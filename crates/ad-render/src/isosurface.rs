//! Vortex-core isosurfaces, drawn by direct raymarching.
//!
//! # Why not marching cubes
//!
//! Marching cubes is the reflex answer and it is the wrong one here.
//!
//! A compute-shader marching cubes over the 512^3 derived grid runs about
//! 110 ms on a 4090 — nine frames a second for the isosurface *alone*, before
//! the volume, the geometry and the post chain get a look in. Worse, its cost is
//! proportional to the number of voxels and is paid again in full every time the
//! isolevel moves, which is exactly the interaction this feature exists for:
//! dragging a slider and watching structures appear. It also needs somewhere to
//! put the triangles, and the vertex count of a Q-criterion isosurface in a
//! turbulent bend is not something you can budget for in advance.
//!
//! Raymarching inverts all three properties:
//!
//! - **Cost scales with pixels, not voxels.** At 1440p that is 3.7 M rays
//!   whatever the grid resolution, and the brick accelerator kills most of them
//!   in the first few steps.
//! - **Depth is free and correct.** The fragment writes the true reverse-Z depth
//!   of the hit, so the duct occludes the surface with no sorting and no
//!   pre-pass.
//! - **Nothing pops.** The isolevel is a uniform; moving it moves the surface
//!   continuously, because there is no mesh to re-tessellate. Marching cubes
//!   changes topology one cell at a time and the surface visibly snaps.
//!
//! There is no allocation proportional to the surface, no vertex buffer, and no
//! upper bound to guess at.
//!
//! # The isolevel, and why there is a histogram
//!
//! Q-criterion has no natural scale — [`crate::fields`] normalises it against
//! `U^2 / D_h^2` so it is dimensionless, but "the right isolevel" still depends
//! entirely on the flow. Shipping a default and hoping is how people conclude a
//! duct has no vortices in it. [`QHistogram`] reduces the actual Q distribution
//! over fluid cells so the UI can draw it under the slider and
//! [`QHistogram::suggested_isolevel`] can put the handle somewhere useful on
//! first sight.
//!
//! # Accuracy
//!
//! The march finds a *bracket* — the first step across which `Q~ - iso` changes
//! sign — and then bisects it. Four halvings take a one-voxel bracket to a
//! sixteenth of a voxel, which is below what the trilinear filter itself
//! resolves, so more would be wasted work. Both halves are mirrored on the CPU
//! as [`first_crossing`] and [`bisect_crossing`] so the logic is testable
//! without an adapter.
//!
//! # Layering
//!
//! Like every overlay in this crate, this pass depth-*tests* against the scene's
//! G-buffer and does not write to it — [`crate::OverlayContext::depth_view`]
//! says so explicitly, and TAA reprojects that buffer with the mesh's motion
//! vectors, which are not this surface's. The consequence is that overlays
//! cannot occlude *one another*, only overdraw in registration order. Register
//! the opaque ones (this and [`crate::slice`]) before the additive tracers, so
//! the emissive layer ends up on top where it belongs. Correct mutual occlusion
//! would need an overlay-owned depth buffer; see the notes in the build report.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::colormap::{self, ColorMap, Interpolation};
use crate::fields::DerivedFields;
use crate::particles::{overlay_defines, overlay_shader_loader};
use crate::{util, OverlayContext, OverlayPass};

/// Workgroup edge for the histogram reduction. 4^3 = 64 threads.
pub(crate) const HISTOGRAM_WG: u32 = 4;

/// What the surface colour encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsosurfaceShading {
    /// Speed in m/s through the colour map. The default: the shape says where
    /// the vortex is, the colour says how fast the air in it is going.
    #[default]
    Speed,
    /// The surface normal as RGB. A debugging view, and the fastest way to see
    /// whether the gradient is right.
    Normal,
    /// A single neutral colour, for when the geometry is the whole message.
    Constant,
}

impl IsosurfaceShading {
    fn code(self) -> u32 {
        match self {
            IsosurfaceShading::Speed => 0,
            IsosurfaceShading::Normal => 1,
            IsosurfaceShading::Constant => 2,
        }
    }
}

/// Everything the user can turn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsosurfaceSettings {
    /// Normalised Q-criterion the surface is drawn at. See [`QHistogram`] for
    /// how to pick one that is not a guess.
    pub iso_level: f32,
    /// March step as a fraction of a derived voxel. 0.5 is the useful default:
    /// the bracket only has to be small enough to hold a single crossing, and
    /// the bisection does the rest.
    pub step_scale: f32,
    /// Hard cap on march steps per ray, so a pathological view cannot hang the
    /// GPU.
    pub max_steps: u32,
    /// Bisection iterations after the sign change. Four is the point past which
    /// the trilinear filter, not the root finder, sets the accuracy.
    pub refine_steps: u32,
    /// Skip bricks that provably contain no crossing. Turning it off must not
    /// change the image; that is the invariant the accelerator is tested by.
    pub skip_empty_space: bool,
    /// Surface opacity. Below 1 the structures behind show through, which is
    /// often the only way to read a nested pair of vortex tubes.
    pub opacity: f32,
    pub shading: IsosurfaceShading,
    pub color_map: ColorMap,
    /// Speed range the colour map spans, m/s.
    pub speed_range: [f32; 2],
    /// Ambient term. Not zero: a pure Lambert vortex tube is black on half its
    /// circumference and reads as two surfaces rather than one.
    pub ambient: f32,
    pub specular: f32,
    /// Rim light along the silhouette. This is what separates two tubes lying
    /// one behind the other.
    pub rim: f32,
    /// Overall shading strength; 0 gives flat colour.
    pub light_strength: f32,
    /// Per-pixel start jitter, in march steps. The surface is hard-edged, so
    /// without this its silhouette steps along the march grid. With it the error
    /// becomes noise the progressive accumulator averages away.
    pub jitter: f32,
    /// Upper end of the Q range the histogram covers.
    pub histogram_max: f32,
}

impl Default for IsosurfaceSettings {
    fn default() -> Self {
        Self {
            iso_level: 0.55,
            step_scale: 0.5,
            max_steps: 512,
            refine_steps: 4,
            skip_empty_space: true,
            opacity: 1.0,
            shading: IsosurfaceShading::default(),
            // Viridis, not Turbo. Turbo is a rainbow map: it is not
            // perceptually uniform, its luminance is non-monotonic so it
            // invents banding that is not in the data, and Google's own
            // announcement is explicit that it is not colourblind-safe.
            // It stays available for users who ask for it, but a rainbow
            // must not be what the tool reaches for on its own.
            color_map: ColorMap::Viridis,
            speed_range: [0.0, 10.0],
            ambient: 0.25,
            specular: 0.35,
            rim: 0.25,
            light_strength: 1.0,
            jitter: 1.0,
            histogram_max: 4.0,
        }
    }
}

impl IsosurfaceSettings {
    pub fn sanitise(&mut self) {
        if !self.iso_level.is_finite() {
            self.iso_level = 0.0;
        }
        self.step_scale = self.step_scale.clamp(0.05, 4.0);
        self.max_steps = self.max_steps.clamp(16, 4096);
        // Zero refinement is legal and gives the raw march grid, which is a
        // useful thing to be able to look at when debugging the bracket.
        self.refine_steps = self.refine_steps.min(16);
        self.opacity = self.opacity.clamp(0.0, 1.0);
        self.ambient = self.ambient.clamp(0.0, 2.0);
        self.specular = self.specular.clamp(0.0, 4.0);
        self.rim = self.rim.clamp(0.0, 4.0);
        self.light_strength = self.light_strength.clamp(0.0, 1.0);
        self.jitter = self.jitter.clamp(0.0, 4.0);
        self.histogram_max = self.histogram_max.max(1e-4);
        if self.speed_range[1] <= self.speed_range[0] {
            self.speed_range[1] = self.speed_range[0] + 1e-3;
        }
    }
}

// -- the CPU twins of the shader arithmetic ----------------------------------

/// Refine a bracketed sign change by bisection, mirroring `fs_iso`.
///
/// Requires `f(t_lo) < 0 <= f(t_hi)`; the caller is the one that establishes
/// that, because bisection on an unbracketed interval converges confidently to
/// the wrong answer rather than failing.
///
/// The residual bracket after `steps` halvings is `(t_hi - t_lo) / 2^steps`, so
/// the returned root is within half of that.
pub fn bisect_crossing(t_lo: f32, t_hi: f32, steps: u32, f: impl Fn(f32) -> f32) -> f32 {
    let (mut a, mut b) = (t_lo, t_hi);
    for _ in 0..steps {
        let m = 0.5 * (a + b);
        if f(m) < 0.0 {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// First crossing of `f` from negative to non-negative along `[t0, t1]`,
/// stepping by `h` and refining with `refine` bisections.
///
/// The CPU twin of the march loop in `fs_iso`, minus the brick skipping — which
/// is an optimisation and by construction cannot change the answer. Returns
/// `t0` immediately if the interval starts inside the surface, which is what the
/// shader does when the eye sits inside a vortex core.
pub fn first_crossing(
    t0: f32,
    t1: f32,
    h: f32,
    refine: u32,
    f: impl Fn(f32) -> f32,
) -> Option<f32> {
    // `is_finite` first, so a NaN step is rejected rather than silently
    // comparing false against every bound and marching forever.
    if !h.is_finite() || h <= 0.0 || t1 <= t0 {
        return None;
    }
    let mut f_prev = f(t0);
    if f_prev >= 0.0 {
        return Some(t0);
    }
    let mut t_prev = t0;
    let mut t = t0;
    while t < t1 {
        let t_next = (t + h).min(t1);
        let f_next = f(t_next);
        if f_prev < 0.0 && f_next >= 0.0 {
            return Some(bisect_crossing(t_prev, t_next, refine, &f));
        }
        t_prev = t_next;
        f_prev = f_next;
        if t_next <= t {
            break;
        }
        t = t_next;
    }
    None
}

// -- the histogram -----------------------------------------------------------

/// Distribution of normalised Q over the fluid cells of the domain.
///
/// Exists so that "what isolevel?" is a question with an answer on screen rather
/// than one the user has to discover by scrubbing. The UI draws it under the
/// isolevel slider; [`Self::suggested_isolevel`] places the handle.
///
/// Solid cells are excluded on the GPU side. That is not a detail: a duct fills
/// perhaps a fifth of its bounding box, so including the wall would put 80% of
/// the mass in bin 0 and drag every percentile to zero.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct QHistogram {
    /// Counts, low Q first. Out-of-range samples are clamped into the end bins.
    pub bins: Vec<u32>,
    /// `[lo, hi]` of the covered Q range, as f32 bits so the type stays `Eq`.
    lo_bits: u32,
    hi_bits: u32,
}

impl QHistogram {
    /// Bins in a histogram. 128 is finer than any slider is wide and still only
    /// 512 bytes to read back.
    pub const BINS: usize = 128;

    pub fn new(bins: Vec<u32>, range: [f32; 2]) -> Self {
        Self {
            bins,
            lo_bits: range[0].to_bits(),
            hi_bits: range[1].to_bits(),
        }
    }

    pub fn range(&self) -> [f32; 2] {
        [f32::from_bits(self.lo_bits), f32::from_bits(self.hi_bits)]
    }

    /// Total fluid samples counted. Zero means either no fluid or no histogram
    /// has been computed yet.
    pub fn total(&self) -> u64 {
        self.bins.iter().map(|c| *c as u64).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Q value at the low edge of bin `i`.
    pub fn bin_edge(&self, i: usize) -> f32 {
        let [lo, hi] = self.range();
        let n = self.bins.len().max(1) as f32;
        lo + (hi - lo) * (i as f32 / n)
    }

    pub fn bin_center(&self, i: usize) -> f32 {
        0.5 * (self.bin_edge(i) + self.bin_edge(i + 1))
    }

    /// Counts scaled so the tallest bin is 1. What a UI actually draws.
    pub fn normalised(&self) -> Vec<f32> {
        let peak = self.bins.iter().copied().max().unwrap_or(0).max(1) as f32;
        self.bins.iter().map(|c| *c as f32 / peak).collect()
    }

    /// Q value below which a fraction `p` of the fluid sits.
    ///
    /// Linear interpolation *within* the straddling bin, not a bin centre: with
    /// 128 bins over a range of 4, a bin is 0.03 wide, and snapping to bin
    /// centres would make the suggested isolevel visibly quantised as the
    /// solver evolves.
    pub fn percentile(&self, p: f32) -> f32 {
        let [lo, hi] = self.range();
        let total = self.total();
        if total == 0 || self.bins.is_empty() {
            return lo;
        }
        let p = p.clamp(0.0, 1.0);
        let target = p as f64 * total as f64;
        let mut cum = 0.0f64;
        for (i, c) in self.bins.iter().enumerate() {
            let next = cum + *c as f64;
            if next >= target && *c > 0 {
                let within = ((target - cum) / *c as f64) as f32;
                let width = (hi - lo) / self.bins.len() as f32;
                return self.bin_edge(i) + within.clamp(0.0, 1.0) * width;
            }
            cum = next;
        }
        hi
    }

    /// An isolevel worth showing on first sight.
    ///
    /// The 99th percentile of fluid Q: vortex cores are by definition the
    /// high-Q tail, and a surface at the median would enclose half the duct and
    /// look like a solid plug. Floored at a small positive value, because Q~
    /// below zero is strain-dominated and has no core to draw.
    pub fn suggested_isolevel(&self) -> f32 {
        if self.is_empty() {
            return IsosurfaceSettings::default().iso_level;
        }
        self.percentile(0.99).max(1.0e-3)
    }
}

// -- GPU plumbing ------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct IsoUniform {
    iso_level: f32,
    step_scale: f32,
    max_steps: u32,
    refine_steps: u32,

    color_lo: f32,
    color_inv_span: f32,
    opacity: f32,
    shading_mode: u32,

    light_dir: [f32; 4],
    surface: [f32; 4],

    skip_enabled: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct HistUniform {
    lo: f32,
    hi: f32,
    bin_count: u32,
    _pad0: u32,
}

/// A Q-criterion isosurface, raymarched into the scene.
pub struct IsosurfaceOverlay {
    pub settings: IsosurfaceSettings,
    /// Direction toward the key light. Kept here rather than read from
    /// `PostSettings` because the overlay does not get to see it; the app syncs
    /// the two, exactly as the renderer syncs the mesh and the volume.
    pub light_dir: Vec3,
    enabled: bool,

    uniform: wgpu::Buffer,
    lut: wgpu::Texture,
    _lut_view: wgpu::TextureView,
    _lut_sampler: wgpu::Sampler,
    _draw_layout: wgpu::BindGroupLayout,
    draw_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,

    hist_uniform: wgpu::Buffer,
    hist_bins: wgpu::Buffer,
    hist_readback: wgpu::Buffer,
    hist_layout: wgpu::BindGroupLayout,
    hist_group: wgpu::BindGroup,
    hist_pipeline: wgpu::ComputePipeline,

    lut_dirty: bool,
    last_color_map: ColorMap,
}

impl IsosurfaceOverlay {
    /// Build the pass. `camera_layout`, `fields_layout` and `brick_layout` are
    /// [`crate::Renderer::camera_bind_group_layout`],
    /// [`crate::Renderer::fields_bind_group_layout`] and
    /// [`crate::Renderer::brick_bind_group_layout`].
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_layout: &wgpu::BindGroupLayout,
        fields_layout: &wgpu::BindGroupLayout,
        brick_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self> {
        let mut settings = IsosurfaceSettings::default();
        settings.sanitise();

        let uniform = util::uniform_buffer::<IsoUniform>(device, "isosurface uniform");
        let lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("isosurface colour LUT"),
            size: wgpu::Extent3d {
                width: colormap::LUT_SIZE as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let lut_view = lut.create_view(&Default::default());
        let lut_sampler = util::linear_clamp_sampler(device, "isosurface LUT");

        let fs = wgpu::ShaderStages::FRAGMENT;
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("isosurface draw"),
            entries: &[
                util::uniform_entry(0, fs),
                util::sampled_float_entry(1, fs, wgpu::TextureViewDimension::D2),
                util::sampler_entry(2, fs, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let draw_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("isosurface draw"),
            layout: &draw_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                },
            ],
        });

        let loader = overlay_shader_loader();
        let defines = overlay_defines();
        let module = loader.create_module(device, "isosurface.wgsl", &defines)?;
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("isosurface"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(fields_layout),
                Some(brick_layout),
                Some(&draw_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("isosurface"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_iso"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // Depth-tested against the scene so the duct occludes the surface,
            // but *not* written. The depth buffer is the G-buffer's, and TAA
            // reprojects with the mesh motion vectors that belong to it —
            // stamping a raymarched surface into it would pair those pixels with
            // motion vectors that are not theirs and smear under camera motion.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::mesh::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(crate::mesh::DEPTH_COMPARE),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_iso"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::post::HDR_FORMAT,
                    // Premultiplied over. The shader multiplies through by
                    // alpha, so `One` and not `SrcAlpha` on the source side.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // -- histogram --
        let hist_uniform = util::uniform_buffer::<HistUniform>(device, "Q histogram uniform");
        let hist_bytes = (QHistogram::BINS * 4) as u64;
        let hist_bins = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Q histogram bins"),
            size: hist_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hist_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Q histogram readback"),
            size: hist_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let cs = wgpu::ShaderStages::COMPUTE;
        let hist_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Q histogram"),
            entries: &[
                util::uniform_entry(0, cs),
                util::storage_buffer_entry(1, cs, false),
            ],
        });
        let hist_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Q histogram"),
            layout: &hist_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hist_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hist_bins.as_entire_binding(),
                },
            ],
        });
        let hist_pipeline = util::compute_pipeline(
            device,
            &loader,
            "isosurface.wgsl",
            "histogram_main",
            &defines.clone().flag("ISO_HISTOGRAM"),
            &[Some(fields_layout), Some(&hist_layout)],
            "Q histogram",
        )?;

        let this = Self {
            settings,
            light_dir: Vec3::new(-0.35, 0.78, 0.52).normalize(),
            enabled: true,
            uniform,
            lut,
            _lut_view: lut_view,
            _lut_sampler: lut_sampler,
            _draw_layout: draw_layout,
            draw_group,
            pipeline,
            hist_uniform,
            hist_bins,
            hist_readback,
            hist_layout,
            hist_group,
            hist_pipeline,
            lut_dirty: true,
            last_color_map: IsosurfaceSettings::default().color_map,
        };
        this.upload_lut(queue);
        Ok(this)
    }

    pub fn settings(&self) -> &IsosurfaceSettings {
        &self.settings
    }
    /// Mutate the settings. The colour LUT is re-baked on the next frame.
    pub fn settings_mut(&mut self) -> &mut IsosurfaceSettings {
        self.lut_dirty = true;
        &mut self.settings
    }

    /// Draw or skip the surface. Cheaper than adding and removing the overlay,
    /// and it keeps the pipeline warm.
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    fn upload_lut(&self, queue: &wgpu::Queue) {
        let colors = self.settings.color_map.lut(Interpolation::default());
        let entries: Vec<[f32; 4]> = colors.iter().map(|c| [c.x, c.y, c.z, 1.0]).collect();
        let bytes = colormap::pack_rgba16f(&entries);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.lut,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(colormap::LUT_SIZE as u32 * 8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: colormap::LUT_SIZE as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn build_uniform(&self) -> IsoUniform {
        let s = &self.settings;
        let (lo, hi) = (s.speed_range[0], s.speed_range[1]);
        let dir = self.light_dir.normalize_or(Vec3::Y);
        IsoUniform {
            iso_level: s.iso_level,
            step_scale: s.step_scale,
            max_steps: s.max_steps,
            refine_steps: s.refine_steps,
            color_lo: lo,
            color_inv_span: 1.0 / (hi - lo).max(1e-9),
            opacity: s.opacity,
            shading_mode: s.shading.code(),
            light_dir: [dir.x, dir.y, dir.z, s.light_strength],
            surface: [s.ambient, s.specular, s.rim, s.jitter],
            skip_enabled: u32::from(s.skip_empty_space),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    /// Reduce the Q distribution over every fluid cell and read it back.
    ///
    /// **Blocking.** It submits its own command buffer and waits for the GPU, so
    /// it belongs on a solver step or a UI event, never in the render loop. That
    /// is a deliberate trade: the alternative is a multi-frame readback ring
    /// whose only purpose is to update a slider's background, and the derived
    /// field changes at solver rate anyway.
    pub fn compute_histogram(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fields: &DerivedFields,
    ) -> QHistogram {
        self.settings.sanitise();
        let range = [0.0f32, self.settings.histogram_max];
        queue.write_buffer(
            &self.hist_uniform,
            0,
            bytemuck::bytes_of(&HistUniform {
                lo: range[0],
                hi: range[1],
                bin_count: QHistogram::BINS as u32,
                _pad0: 0,
            }),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Q histogram"),
        });
        enc.clear_buffer(&self.hist_bins, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Q histogram"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hist_pipeline);
            pass.set_bind_group(0, fields.read_bind_group(), &[]);
            pass.set_bind_group(1, &self.hist_group, &[]);
            let d = fields.dims();
            pass.dispatch_workgroups(
                util::dispatch_count(d.x, HISTOGRAM_WG),
                util::dispatch_count(d.y, HISTOGRAM_WG),
                util::dispatch_count(d.z, HISTOGRAM_WG),
            );
        }
        let bytes = (QHistogram::BINS * 4) as u64;
        enc.copy_buffer_to_buffer(&self.hist_bins, 0, &self.hist_readback, 0, bytes);
        queue.submit([enc.finish()]);

        let slice = self.hist_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let _ = rx.recv();
        let mut bins = vec![0u32; QHistogram::BINS];
        if let Ok(view) = slice.get_mapped_range() {
            for (i, b) in bins.iter_mut().enumerate() {
                *b = u32::from_le_bytes(view[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        self.hist_readback.unmap();
        QHistogram::new(bins, range)
    }

    /// The histogram bind group layout, exposed so a caller can see what the
    /// compute pass binds without reaching into the module.
    pub fn histogram_layout(&self) -> &wgpu::BindGroupLayout {
        &self.hist_layout
    }
}

impl OverlayPass for IsosurfaceOverlay {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "isosurface"
    }

    fn is_static(&self) -> bool {
        // Nothing here animates on its own: the surface only moves when the
        // field or the isolevel does, and both of those already reset the
        // accumulator through the renderer.
        true
    }

    fn record(&mut self, ctx: &mut OverlayContext<'_>) {
        if !self.enabled {
            return;
        }
        self.settings.sanitise();
        if self.lut_dirty || self.last_color_map != self.settings.color_map {
            self.upload_lut(ctx.queue);
            self.last_color_map = self.settings.color_map;
            self.lut_dirty = false;
        }
        ctx.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.build_uniform()));

        let ts = ctx.profiler.render_scope("isosurface");
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("isosurface"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.hdr_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.camera_bind_group, &[]);
        pass.set_bind_group(1, ctx.fields.read_bind_group(), &[]);
        pass.set_bind_group(2, ctx.bricks.read_bind_group(), &[]);
        pass.set_bind_group(3, &self.draw_group, &[]);
        // One fullscreen triangle. Every pixel is a ray; the brick test kills
        // the ones that miss in the first few steps.
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::{GpuContext, Grid};
    use glam::UVec3;

    fn gpu() -> Option<GpuContext> {
        GpuContext::for_tests()
    }

    #[test]
    fn bisection_halves_the_bracket_every_iteration() {
        // The claim that justifies stopping at four: each step halves the
        // interval, so four take one voxel to a sixteenth of one.
        let root = 3.7f32;
        let f = |t: f32| t - root;
        for steps in 0..12u32 {
            let got = bisect_crossing(0.0, 8.0, steps, f);
            let bound = 8.0 / 2f32.powi(steps as i32 + 1);
            assert!(
                (got - root).abs() <= bound + 1e-5,
                "{steps} steps gave {got}, outside the {bound} bracket"
            );
        }
        // Four steps on a one-voxel bracket, stated in the units that matter.
        let voxel = 0.75f32;
        let err = (bisect_crossing(0.0, voxel, 4, |t| t - 0.31 * voxel) - 0.31 * voxel).abs();
        assert!(
            err < voxel / 16.0,
            "four bisections left {err} mm of a {voxel} mm voxel"
        );
    }

    #[test]
    fn bisection_converges_on_a_nonlinear_implicit_too() {
        // A real Q field is nothing like linear across a voxel. Bisection does
        // not care, which is exactly why it is used instead of a secant step
        // that would be fooled by curvature.
        let f = |t: f32| (t * t * t) - 2.0;
        let root = 2f32.cbrt();
        let got = bisect_crossing(0.0, 4.0, 20, f);
        assert!((got - root).abs() < 1e-4, "got {got}, want {root}");
    }

    #[test]
    fn the_march_finds_the_first_crossing_and_not_a_later_one() {
        // Two nested shells along the ray. The visible surface is the near one;
        // returning the far one draws the inside of the structure and looks
        // convincingly like a normal map bug.
        let f = |t: f32| {
            // Negative, positive on [2, 3], negative, positive on [6, 7].
            if (2.0..3.0).contains(&t) || (6.0..7.0).contains(&t) {
                1.0
            } else {
                -1.0
            }
        };
        let hit = first_crossing(0.0, 10.0, 0.1, 8, f).expect("a crossing exists");
        assert!(
            (hit - 2.0).abs() < 0.05,
            "found the crossing at {hit}, not at 2.0"
        );
    }

    #[test]
    fn a_ray_that_misses_everything_reports_no_hit() {
        assert!(first_crossing(0.0, 10.0, 0.1, 4, |_| -1.0).is_none());
        // A degenerate step must not spin forever.
        assert!(first_crossing(0.0, 10.0, 0.0, 4, |_| -1.0).is_none());
        assert!(first_crossing(5.0, 5.0, 0.1, 4, |_| -1.0).is_none());
    }

    #[test]
    fn starting_inside_the_surface_hits_immediately() {
        // The eye inside a vortex core. The shader reports the entry point
        // rather than searching for a crossing that is behind the camera.
        let hit = first_crossing(1.5, 10.0, 0.1, 4, |_| 1.0).expect("inside counts as a hit");
        assert_eq!(hit, 1.5);
    }

    #[test]
    fn a_finer_march_never_moves_the_root_by_more_than_the_refinement() {
        // Halving the step must not change the answer, because the bisection
        // resolves whatever the bracket was. If it does, the bracket is not
        // being established correctly.
        let f = |t: f32| t - 4.321;
        let coarse = first_crossing(0.0, 10.0, 0.5, 6, f).unwrap();
        let fine = first_crossing(0.0, 10.0, 0.05, 6, f).unwrap();
        assert!((coarse - fine).abs() < 0.02, "{coarse} vs {fine}");
    }

    // -- histogram -----------------------------------------------------------

    fn synthetic_histogram() -> QHistogram {
        // 100 samples spread evenly over the first half of the range and 1 in
        // the top bin: roughly the shape of a real Q distribution, which is a
        // heap near zero with a thin high tail that is the actual vortices.
        let mut bins = vec![0u32; QHistogram::BINS];
        for b in bins.iter_mut().take(QHistogram::BINS / 2) {
            *b = 2;
        }
        bins[QHistogram::BINS - 1] = 1;
        QHistogram::new(bins, [0.0, 4.0])
    }

    #[test]
    fn percentiles_are_monotone_and_span_the_range() {
        let h = synthetic_histogram();
        assert_eq!(h.total(), 2 * (QHistogram::BINS as u64 / 2) + 1);
        let mut last = f32::NEG_INFINITY;
        for i in 0..=100 {
            let p = i as f32 / 100.0;
            let v = h.percentile(p);
            assert!(
                v >= last - 1e-4,
                "percentile {p} went backwards: {last} -> {v}"
            );
            assert!(
                (0.0..=4.0).contains(&v),
                "percentile {p} = {v} is outside the range"
            );
            last = v;
        }
        // The median of a flat block over the low half sits in the middle of it.
        let median = h.percentile(0.5);
        assert!((median - 1.0).abs() < 0.05, "median is {median}, want ~1.0");
    }

    #[test]
    fn the_suggested_isolevel_lands_in_the_high_tail() {
        // The property that makes the suggestion useful: it must sit above the
        // bulk of the distribution, or the surface encloses the whole duct and
        // the user sees a plug rather than vortices.
        let h = synthetic_histogram();
        let iso = h.suggested_isolevel();
        assert!(
            iso > h.percentile(0.5),
            "suggestion {iso} is below the median"
        );
        assert!(iso > 1.9, "suggestion {iso} is inside the low-Q bulk");
        assert!(iso <= 4.0);

        // An empty histogram must not suggest a NaN or a zero that hides the
        // surface entirely.
        let empty = QHistogram::new(vec![0; QHistogram::BINS], [0.0, 4.0]);
        assert!(empty.is_empty());
        assert!(empty.suggested_isolevel().is_finite());
        assert!(empty.suggested_isolevel() > 0.0);
    }

    #[test]
    fn bin_edges_tile_the_range_without_gaps() {
        let h = synthetic_histogram();
        assert_eq!(h.bin_edge(0), 0.0);
        assert!((h.bin_edge(QHistogram::BINS) - 4.0).abs() < 1e-5);
        for i in 0..QHistogram::BINS {
            let c = h.bin_center(i);
            assert!(c > h.bin_edge(i) && c < h.bin_edge(i + 1));
        }
        let n = h.normalised();
        assert_eq!(n.len(), QHistogram::BINS);
        assert!(n.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(n.iter().cloned().fold(0.0f32, f32::max) == 1.0);
    }

    #[test]
    fn settings_sanitise_into_the_shader_contract() {
        let mut s = IsosurfaceSettings {
            iso_level: f32::NAN,
            step_scale: 0.0,
            max_steps: 1,
            refine_steps: 99,
            opacity: 4.0,
            speed_range: [5.0, 1.0],
            jitter: -1.0,
            histogram_max: 0.0,
            ..Default::default()
        };
        s.sanitise();
        assert_eq!(s.iso_level, 0.0);
        assert!(s.step_scale > 0.0);
        assert!(s.max_steps >= 16);
        assert!(s.refine_steps <= 16);
        assert_eq!(s.opacity, 1.0);
        assert!(s.speed_range[1] > s.speed_range[0]);
        assert!(s.jitter >= 0.0);
        assert!(s.histogram_max > 0.0);
    }

    #[test]
    fn the_uniforms_are_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<IsoUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<HistUniform>() % 16, 0);
    }

    // -- GPU -----------------------------------------------------------------

    fn test_grid() -> Grid {
        Grid {
            dims: UVec3::new(48, 32, 32),
            dx_mm: 0.75,
            origin_mm: Vec3::splat(-12.0),
        }
    }

    /// A pair of counter-rotating blobs. Chosen because it has genuine rotation
    /// *and* genuine strain between the two cores, so Q-criterion is non-trivial
    /// and the isosurface has a shape rather than a sphere — and because it is
    /// sparse, which is what leaves most bricks inactive and gives the
    /// accelerator something to skip.
    fn swirl_field(dims: UVec3) -> Vec<[f32; 4]> {
        let c = dims.as_vec3() * 0.5;
        let radius = dims.y as f32 * 0.22;
        let mut out = Vec::with_capacity((dims.x * dims.y * dims.z) as usize);
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let p = Vec3::new(x as f32, y as f32, z as f32) - c;
                    let mut u = Vec3::ZERO;
                    for sign in [-1.0f32, 1.0] {
                        let d = p - Vec3::new(sign * radius * 1.3, 0.0, 0.0);
                        let g = (-(d.length_squared()) / (radius * radius)).exp();
                        u += Vec3::new(-d.y, d.x, 0.35 * sign) * (sign * 0.9 * g);
                    }
                    out.push([u.x, u.y, u.z, 0.0]);
                }
            }
        }
        out
    }

    /// Upload a synthetic field and derive it, returning the views the renderer
    /// needs. Unit scales, so the texture holds exactly what the test wrote.
    fn upload_field(
        gpu: &GpuContext,
        grid: Grid,
        data: &[[f32; 4]],
    ) -> (wgpu::Texture, wgpu::Texture) {
        let dims = grid.dims;
        let (mac, flg) = crate::fields::create_source_textures(&gpu.device, grid);
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for t in data {
            for c in t {
                bytes.extend_from_slice(&colormap::f32_to_f16_bits(*c).to_le_bytes());
            }
        }
        let extent = wgpu::Extent3d {
            width: dims.x,
            height: dims.y,
            depth_or_array_layers: dims.z,
        };
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &mac,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims.x * 8),
                rows_per_image: Some(dims.y),
            },
            extent,
        );
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &flg,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &vec![ad_gpu::flags::FLUID; (dims.x * dims.y * dims.z) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims.x),
                rows_per_image: Some(dims.y),
            },
            extent,
        );
        (mac, flg)
    }

    #[test]
    fn the_brick_skip_does_not_change_the_isosurface() {
        // The headline invariant, and the same one the volume raymarch is held
        // to: the accelerator is an optimisation, so turning it off must produce
        // the same picture. A difference is a hole in the surface that the
        // skipping is jumping over, and a hole in an isosurface reads as a real
        // feature of the flow rather than as a bug.
        //
        // The third case is the subtle one. The brick min/max is reduced over
        // whichever channel is *displayed*, so the exact interval test is only
        // available while Q-criterion is on screen. With speed displayed the
        // shader must fall back to the channel-independent "no fluid here" flag
        // — and still draw exactly the same surface.
        let Some(gpu) = gpu() else { return };
        let grid = Grid {
            dims: UVec3::new(64, 48, 48),
            dx_mm: 0.75,
            origin_mm: Vec3::splat(-18.0),
        };
        let (w, h) = (192u32, 144u32);
        let (mac, flg) = upload_field(&gpu, grid, &swirl_field(grid.dims));
        let d3 = wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        };
        let mac_view = mac.create_view(&d3);
        let flg_view = flg.create_view(&d3);
        let scales = crate::DeriveScales {
            speed_ms: 1.0,
            q_tilde: 1.0,
            vorticity: 1.0,
            pressure_pa: 1.0,
        };

        let mut cam = crate::Camera::default();
        cam.aspect = w as f32 / h as f32;
        cam.frame_bbox(grid.bbox(), 0.05);
        cam.pitch = 0.35;

        let render = |field: crate::DerivedField, skip: bool, iso: f32| -> Vec<u8> {
            let mut r = crate::Renderer::new(
                &gpu,
                crate::RendererConfig {
                    width: w,
                    height: h,
                    target_format: wgpu::TextureFormat::Rgba8Unorm,
                    grid,
                    field_resolution: crate::FieldResolution::Full,
                    profiling: false,
                },
            )
            .expect("renderer");
            // Everything temporal or stochastic off, and the volume made
            // transparent, so the comparison is of the isosurface and nothing
            // else.
            {
                let s = r.post_settings_mut();
                s.taa = false;
                s.bloom = false;
                s.ssao = false;
                s.dither = 0.0;
            }
            let mut vs = *r.volume_settings();
            vs.density = 0.0;
            r.set_volume_settings(vs);
            r.set_field(field);

            let mut o = IsosurfaceOverlay::new(
                &gpu.device,
                &gpu.queue,
                r.camera_bind_group_layout(),
                r.fields_bind_group_layout(),
                r.brick_bind_group_layout(),
            )
            .expect("isosurface");
            {
                let s = o.settings_mut();
                s.iso_level = iso;
                s.skip_empty_space = skip;
                // Deterministic: the start jitter is a stochastic antialiasing
                // trick and would swamp the comparison with its own noise.
                s.jitter = 0.0;
            }
            r.add_overlay(Box::new(o));

            let target = util::color_target(
                &gpu.device,
                "iso skip test",
                w,
                h,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            let view = target.create_view(&Default::default());
            let scene = crate::Scene::new();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            r.render(
                &mut enc,
                crate::FrameInput {
                    camera: &cam,
                    scene: &scene,
                    sources: Some(crate::FieldSources {
                        macro_view: &mac_view,
                        flags_view: Some(&flg_view),
                        scales,
                    }),
                    sdf: None,
                    target: &view,
                    dt: 1.0 / 60.0,
                },
            )
            .expect("frame");
            util::readback_rgba8(&gpu.device, &gpu.queue, &target, w, h, enc)
        };

        // Pick the isolevel from the field's own distribution rather than from a
        // constant, so the test cannot quietly stop drawing anything if the
        // synthetic field is ever retuned.
        let iso = {
            let mut r = crate::Renderer::new(
                &gpu,
                crate::RendererConfig {
                    width: 64,
                    height: 64,
                    target_format: wgpu::TextureFormat::Rgba8Unorm,
                    grid,
                    field_resolution: crate::FieldResolution::Full,
                    profiling: false,
                },
            )
            .unwrap();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            let target = util::color_target(
                &gpu.device,
                "hist",
                64,
                64,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            let view = target.create_view(&Default::default());
            r.render(
                &mut enc,
                crate::FrameInput {
                    camera: &cam,
                    scene: &crate::Scene::new(),
                    sources: Some(crate::FieldSources {
                        macro_view: &mac_view,
                        flags_view: Some(&flg_view),
                        scales,
                    }),
                    sdf: None,
                    target: &view,
                    dt: 0.0,
                },
            )
            .unwrap();
            gpu.queue.submit([enc.finish()]);
            let mut o = IsosurfaceOverlay::new(
                &gpu.device,
                &gpu.queue,
                r.camera_bind_group_layout(),
                r.fields_bind_group_layout(),
                r.brick_bind_group_layout(),
            )
            .unwrap();
            let hist = o.compute_histogram(&gpu.device, &gpu.queue, r.fields());
            assert!(!hist.is_empty(), "the histogram counted no fluid cells");
            // Well inside the structure. The blobs occupy a small fraction of
            // the box, so a lower percentile lands in the near-zero Q of the
            // still surroundings and the "surface" is a noise shell whose every
            // pixel is a grazing hit -- which tests the accelerator against
            // sampling noise rather than against itself.
            let iso = hist.percentile(0.995);
            assert!(
                iso > 0.05,
                "the synthetic field has no real Q structure: {iso}"
            );
            iso
        };

        let a = render(crate::DerivedField::QCriterion, true, iso);
        let b = render(crate::DerivedField::QCriterion, false, iso);
        let c = render(crate::DerivedField::Speed, true, iso);
        assert_eq!(a.len(), b.len());

        // The surface has to actually be on screen, or the three images agree
        // about a picture of the background.
        let lum: Vec<f32> = a.chunks_exact(4).map(|p| p[0] as f32).collect();
        let mean = lum.iter().sum::<f32>() / lum.len() as f32;
        let var = lum.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / lum.len() as f32;
        assert!(var > 40.0, "nothing was drawn: red-channel variance {var}");

        for (other, what) in [
            (&b, "turning the skip off"),
            (&c, "displaying another field"),
        ] {
            let mut worst = 0i32;
            let mut differing = 0usize;
            for (x, y) in a.iter().zip(other.iter()) {
                let d = (*x as i32 - *y as i32).abs();
                worst = worst.max(d);
                differing += usize::from(d > 0);
            }
            let frac = differing as f64 / a.len() as f64;
            assert!(
                worst <= 2,
                "{what} moved a pixel by {worst}/255 -- the brick skip is jumping a crossing"
            );
            assert!(
                frac < 0.005,
                "{what} changed {:.2}% of pixels",
                frac * 100.0
            );
        }
    }

    #[test]
    fn the_isosurface_records_inside_a_real_frame() {
        let Some(gpu) = gpu() else { return };
        let mut r = crate::Renderer::new(
            &gpu,
            crate::RendererConfig {
                width: 160,
                height: 120,
                target_format: wgpu::TextureFormat::Rgba8Unorm,
                grid: test_grid(),
                field_resolution: crate::FieldResolution::Full,
                profiling: false,
            },
        )
        .expect("renderer");
        let iso = IsosurfaceOverlay::new(
            &gpu.device,
            &gpu.queue,
            r.camera_bind_group_layout(),
            r.fields_bind_group_layout(),
            r.brick_bind_group_layout(),
        )
        .expect("isosurface");
        r.add_overlay(Box::new(iso));

        let target = util::color_target(
            &gpu.device,
            "iso target",
            160,
            120,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let view = target.create_view(&Default::default());
        let scene = crate::Scene::new();
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        r.render(
            &mut enc,
            crate::FrameInput {
                camera: &crate::Camera::default(),
                scene: &scene,
                sources: None,
                sdf: None,
                target: &view,
                dt: 1.0 / 60.0,
            },
        )
        .expect("frame");
        gpu.queue.submit([enc.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }
}

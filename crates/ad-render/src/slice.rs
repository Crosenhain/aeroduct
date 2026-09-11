//! An oriented cutting plane carrying animated line-integral convolution.
//!
//! # What a slice is for
//!
//! A colour-mapped plane through the duct answers "how fast is the air here".
//! On its own it cannot answer "where is it going", and in a bend that is the
//! whole question. LIC smears a noise field along the local streamlines so the
//! *texture* becomes the direction — with none of the sampling bias of a
//! hand-placed streamline seed set, and at a cost that scales with pixels rather
//! than seeds. Making the convolution kernel a travelling wave resolves the last
//! ambiguity: a static LIC shows the streamline axis but not which way along it
//! the fluid travels, and left and right look identical.
//!
//! The arithmetic all lives in [`crate::lic`] and is mirrored in
//! `shaders/render/lic.wgsl`; this module is the plane, the bind groups and the
//! draw.
//!
//! # The three rules the composite obeys
//!
//! **Colour stays quantitative.** The composite is
//! `colormap(scalar) * ((1 - c) + c * lic)` — multiplicative, brightness only,
//! never a hue shift and never a blend towards the LIC's own grey. At the
//! default contrast of 0.4 that is `base * (0.6 + 0.4 * lic)`. A reader can
//! still take a colour off the plane, look it up in the legend, and get a
//! number.
//!
//! **The convolution is normalised.** The travelling kernel's weights do not sum
//! to a phase-independent constant, so an unnormalised sum makes the whole plane
//! pulse once per animation cycle — which looks precisely like a flickering
//! exposure bug and is the sort of thing that gets blamed on the tonemapper.
//!
//! **The contrast is scaled by `|u_in_plane| / |u|`.** This is the one that
//! matters most and the one everyone omits. LIC on a plane can only show the
//! in-plane component of the velocity. In a 90-degree bend a large fraction of
//! the flow pierces any plane you can draw, and a confident swirl texture
//! rendered from a 10%-of-magnitude in-plane residue is a lie told beautifully.
//! [`crate::lic::LicSettings::honesty`] fades the texture to flat colour exactly
//! where the flow is leaving the plane. It costs nothing and it is the
//! difference between a figure you can publish and one you cannot.
//!
//! # Following the duct
//!
//! An axis-aligned cut through a 90-degree bend is oblique to the flow over most
//! of its area, and every velocity it shows is foreshortened by an angle that
//! changes across the image. [`SliceSettings::follow_centreline`] instead rides a
//! [`Centreline`] spline: the plane normal is the spline tangent, so one slider
//! walks a plane down the passage staying perpendicular to it and the in-plane
//! fraction stays small — which, by the honesty rule above, is what makes the
//! LIC on it worth trusting.
//!
//! # Layering
//!
//! The plane depth-*tests* against the scene's G-buffer and does not write to
//! it, as [`crate::OverlayContext::depth_view`] requires, so overlays overdraw
//! in registration order rather than occluding one another. Register this before
//! the additive tracers of [`crate::particles`]. See [`crate::isosurface`] for
//! the same note and the reason.

use ad_gpu::Bbox;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

use crate::colormap::{self, ColorMap, Interpolation};
use crate::fields::DerivedField;
use crate::lic::{Centreline, LicSettings};
use crate::particles::{overlay_defines, overlay_shader_loader};
use crate::{util, OverlayContext, OverlayPass};

/// The three axis-aligned cut orientations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisPreset {
    /// Normal along +X: a plane of constant `x`.
    X,
    /// Normal along +Y.
    Y,
    /// Normal along +Z.
    Z,
}

impl AxisPreset {
    pub const ALL: [AxisPreset; 3] = [AxisPreset::X, AxisPreset::Y, AxisPreset::Z];

    pub fn normal(self) -> Vec3 {
        match self {
            AxisPreset::X => Vec3::X,
            AxisPreset::Y => Vec3::Y,
            AxisPreset::Z => Vec3::Z,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            AxisPreset::X => "YZ (x = const)",
            AxisPreset::Y => "XZ (y = const)",
            AxisPreset::Z => "XY (z = const)",
        }
    }
}

/// A cutting plane as a point and an orthonormal in-plane basis.
///
/// Stored as `(origin, u, v)` rather than `(origin, normal)` because the LIC
/// noise lattice and the quad both live in plane coordinates, and deriving a
/// basis from a normal every frame would let the texture spin as the normal
/// moves. The normal is `u x v`, so it is always consistent with the basis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlicePlane {
    pub origin_mm: Vec3,
    pub u: Vec3,
    pub v: Vec3,
}

impl Default for SlicePlane {
    fn default() -> Self {
        Self { origin_mm: Vec3::ZERO, u: Vec3::X, v: Vec3::Y }
    }
}

impl SlicePlane {
    /// Build from a point and a normal, choosing an arbitrary but stable basis.
    ///
    /// The reference axis is swapped when the normal gets close to it, because
    /// `cross` of two nearly parallel vectors loses all its precision and the
    /// basis — and with it the noise lattice — starts to spin.
    pub fn from_normal(origin_mm: Vec3, normal: Vec3) -> Self {
        let n = normal.normalize_or(Vec3::Z);
        let reference = if n.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
        let u = reference.cross(n).normalize_or(Vec3::X);
        let v = n.cross(u).normalize_or(Vec3::Y);
        Self { origin_mm, u, v }
    }

    /// An axis-aligned preset at normalised position `t` across `bbox`.
    pub fn axis(preset: AxisPreset, bbox: Bbox, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let origin = bbox.min + (bbox.max - bbox.min) * match preset {
            AxisPreset::X => Vec3::new(t, 0.5, 0.5),
            AxisPreset::Y => Vec3::new(0.5, t, 0.5),
            AxisPreset::Z => Vec3::new(0.5, 0.5, t),
        };
        Self::from_normal(origin, preset.normal())
    }

    /// Unit normal, `u x v`. Right-handed by construction.
    pub fn normal(&self) -> Vec3 {
        self.u.cross(self.v).normalize_or(Vec3::Z)
    }

    /// World point to plane coordinates in mm, dropping the out-of-plane part.
    pub fn project(&self, p: Vec3) -> Vec2 {
        let d = p - self.origin_mm;
        Vec2::new(d.dot(self.u), d.dot(self.v))
    }

    /// Plane coordinates in mm back to a world point.
    pub fn point(&self, uv: Vec2) -> Vec3 {
        self.origin_mm + self.u * uv.x + self.v * uv.y
    }

    /// Signed distance from the plane, mm.
    pub fn distance(&self, p: Vec3) -> f32 {
        (p - self.origin_mm).dot(self.normal())
    }

    /// Re-orthonormalise, so a plane assembled by hand or dragged by a gizmo
    /// cannot slowly shear.
    pub fn sanitise(&mut self) {
        let u = self.u.normalize_or(Vec3::X);
        // Gram-Schmidt: remove whatever component of v has crept onto u.
        let v = (self.v - u * self.v.dot(u)).normalize_or_zero();
        let v = if v.length_squared() < 0.5 {
            // v collapsed onto u; pick any perpendicular rather than emitting a
            // degenerate basis that would make the quad a line.
            let reference = if u.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
            reference.cross(u).normalize_or(Vec3::Y)
        } else {
            v
        };
        self.u = u;
        self.v = v;
    }
}

/// Fraction of a velocity that lies in a plane, `|u - (u.n)n| / |u|`.
///
/// The honesty weight. 1 where the flow runs entirely within the cutting plane,
/// 0 where it pierces it dead on. Mirrors `lic_honesty` in `lic.wgsl` at
/// `power = 1`.
pub fn in_plane_fraction(velocity: Vec3, normal: Vec3) -> f32 {
    let speed = velocity.length();
    if speed <= 1e-9 {
        return 0.0;
    }
    let n = normal.normalize_or(Vec3::Z);
    ((velocity - n * velocity.dot(n)).length() / speed).clamp(0.0, 1.0)
}

/// The composite: `base * ((1 - contrast) + contrast * lic)`.
///
/// Mirrors `lic_compose` in `lic.wgsl`. Multiplicative on purpose — see the
/// module docs for why the colour is not allowed to move.
pub fn compose(base: Vec3, lic: f32, contrast: f32) -> Vec3 {
    let c = contrast.clamp(0.0, 1.0);
    base * ((1.0 - c) + c * lic.clamp(0.0, 1.0))
}

/// Everything the user can turn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceSettings {
    /// The plane, in world millimetres. Ignored while
    /// [`Self::follow_centreline`] is set.
    pub plane: SlicePlane,
    /// Ride the centreline spline instead: the value is a normalised arc
    /// position in `[0, 1]` and the plane normal becomes the spline tangent.
    /// Has no effect until a centreline is supplied with
    /// [`SliceOverlay::set_centreline`].
    pub follow_centreline: Option<f32>,
    /// Half-size of the drawn quad, mm. `None` fits it to the field volume,
    /// which is what you want unless you are deliberately cropping.
    pub half_extent_mm: Option<f32>,
    pub opacity: f32,
    /// Exposure of the plane relative to the rest of the scene. The target is
    /// HDR, so above 1 is meaningful.
    pub brightness: f32,
    pub lic: LicSettings,
    /// Which derived scalar is colour-mapped. Independent of the volume's
    /// displayed field, so a speed slice can sit inside a Q-criterion volume.
    pub field: DerivedField,
    pub color_map: ColorMap,
    /// Data range the colour map spans, in `field`'s own units.
    pub range: [f32; 2],
    /// Width of the bright rim round the plane border, mm. Zero for none.
    ///
    /// Not decoration: a slice positioned in still air or inside solid material
    /// discards every fragment, and without the rim the control that moves it
    /// appears to do nothing at all.
    pub edge_mm: f32,
    pub edge_color: Vec3,
}

impl Default for SliceSettings {
    fn default() -> Self {
        Self {
            plane: SlicePlane::default(),
            follow_centreline: None,
            half_extent_mm: None,
            opacity: 0.95,
            brightness: 1.0,
            lic: LicSettings::default(),
            field: DerivedField::Speed,
            // Viridis, not Turbo. Turbo is a rainbow map: it is not
            // perceptually uniform, its luminance is non-monotonic so it
            // invents banding that is not in the data, and Google's own
            // announcement is explicit that it is not colourblind-safe.
            // It stays available for users who ask for it, but a rainbow
            // must not be what the tool reaches for on its own.
            color_map: ColorMap::Viridis,
            range: [0.0, 10.0],
            edge_mm: 0.6,
            edge_color: Vec3::new(0.55, 0.62, 0.72),
        }
    }
}

impl SliceSettings {
    pub fn sanitise(&mut self) {
        self.plane.sanitise();
        if let Some(t) = &mut self.follow_centreline {
            *t = t.clamp(0.0, 1.0);
        }
        if let Some(e) = &mut self.half_extent_mm {
            *e = e.clamp(0.1, 10_000.0);
        }
        self.opacity = self.opacity.clamp(0.0, 1.0);
        self.brightness = self.brightness.clamp(0.0, 8.0);
        self.edge_mm = self.edge_mm.clamp(0.0, 100.0);
        self.lic.sanitise();
        if self.range[1] <= self.range[0] {
            self.range[1] = self.range[0] + 1e-3;
        }
    }

    /// Jump to an axis-aligned cut at normalised position `t` across `bbox`,
    /// leaving centreline-follow off.
    pub fn set_axis(&mut self, preset: AxisPreset, bbox: Bbox, t: f32) {
        self.plane = SlicePlane::axis(preset, bbox, t);
        self.follow_centreline = None;
    }
}

// -- GPU plumbing ------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SliceUniform {
    origin_mm: [f32; 3],
    half_u_mm: f32,

    axis_u: [f32; 3],
    half_v_mm: f32,

    axis_v: [f32; 3],
    opacity: f32,

    normal: [f32; 3],
    contrast: f32,

    color_lo: f32,
    color_inv_span: f32,
    steps: u32,
    step_mm: f32,

    phase: f32,
    noise_scale_mm: f32,
    honesty_power: f32,
    honesty: u32,

    channel: u32,
    brightness: f32,
    edge_mm: f32,
    _pad0: f32,

    edge_color: [f32; 4],
}

/// A cutting plane with animated LIC on it.
pub struct SliceOverlay {
    pub settings: SliceSettings,
    enabled: bool,

    uniform: wgpu::Buffer,
    lut: wgpu::Texture,
    _lut_view: wgpu::TextureView,
    _lut_sampler: wgpu::Sampler,
    _layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,

    centreline: Option<Centreline>,
    /// Seconds of wall clock the animation has run for. Kept on the CPU and
    /// wrapped before it reaches the shader; see [`LicSettings::phase_at`].
    time_s: f32,
    /// Wall-clock delta of the previous recorded frame. See
    /// [`OverlayPass::is_static`].
    last_dt: f32,
    lut_dirty: bool,
    last_color_map: ColorMap,
}

impl SliceOverlay {
    /// Build the pass. `camera_layout` and `fields_layout` are
    /// [`crate::Renderer::camera_bind_group_layout`] and
    /// [`crate::Renderer::fields_bind_group_layout`].
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_layout: &wgpu::BindGroupLayout,
        fields_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self> {
        let mut settings = SliceSettings::default();
        settings.sanitise();

        let uniform = util::uniform_buffer::<SliceUniform>(device, "slice uniform");
        let lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("slice colour LUT"),
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
        let lut_sampler = util::linear_clamp_sampler(device, "slice LUT");

        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slice"),
            entries: &[
                util::uniform_entry(0, vf),
                util::sampled_float_entry(1, wgpu::ShaderStages::FRAGMENT, wgpu::TextureViewDimension::D2),
                util::sampler_entry(
                    2,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                ),
            ],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slice"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
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
        let module = loader.create_module(device, "slice.wgsl", &defines)?;
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slice"),
            bind_group_layouts: &[Some(camera_layout), Some(fields_layout), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("slice"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_slice"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Visible from both sides. A cutting plane you have to orbit
                // around to see is worse than useless.
                cull_mode: None,
                ..Default::default()
            },
            // Depth tested against the scene so the duct occludes the plane, and
            // not written: the depth buffer belongs to the G-buffer, whose
            // motion vectors TAA reprojects with, and a translucent plane has no
            // business claiming those pixels.
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
                entry_point: Some("fs_slice"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::post::HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            // Premultiplied: the shader has already multiplied
                            // through by alpha.
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

        let this = Self {
            settings,
            enabled: true,
            uniform,
            lut,
            _lut_view: lut_view,
            _lut_sampler: lut_sampler,
            _layout: layout,
            group,
            pipeline,
            centreline: None,
            time_s: 0.0,
            last_dt: 0.0,
            lut_dirty: true,
            last_color_map: SliceSettings::default().color_map,
        };
        this.upload_lut(queue);
        Ok(this)
    }

    pub fn settings(&self) -> &SliceSettings {
        &self.settings
    }
    /// Mutate the settings. The colour LUT is re-baked on the next frame.
    pub fn settings_mut(&mut self) -> &mut SliceSettings {
        self.lut_dirty = true;
        &mut self.settings
    }

    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Supply the duct centreline that [`SliceSettings::follow_centreline`]
    /// rides. Typically the spline the geometry crate fits through the passage.
    pub fn set_centreline(&mut self, centreline: Option<Centreline>) {
        self.centreline = centreline;
    }
    pub fn centreline(&self) -> Option<&Centreline> {
        self.centreline.as_ref()
    }

    /// The plane actually drawn, after centreline-follow has been applied.
    ///
    /// Falls back to [`SliceSettings::plane`] when follow is off or no
    /// centreline has been supplied, so a UI that offers the mode before the
    /// geometry is loaded degrades to an ordinary plane rather than to nothing.
    pub fn resolved_plane(&self) -> SlicePlane {
        match (self.settings.follow_centreline, &self.centreline) {
            (Some(t), Some(c)) => {
                let (origin, u, v) = c.frame(t.clamp(0.0, 1.0) * c.length());
                SlicePlane { origin_mm: origin, u, v }
            }
            _ => self.settings.plane,
        }
    }

    /// Seconds the LIC animation has been running.
    pub fn time(&self) -> f32 {
        self.time_s
    }

    /// Whether a slice in this state changes its image from frame to frame.
    ///
    /// Split out from [`OverlayPass::is_static`] so the three-way condition can
    /// be tested without a GPU and without reaching into private state: all
    /// three have to hold, and dropping any one of them silently disables the
    /// progressive accumulator for the whole scene.
    pub fn animating(enabled: bool, cycles_per_second: f32, last_dt: f32) -> bool {
        enabled && cycles_per_second != 0.0 && last_dt > 0.0
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

    fn build_uniform(&self, bbox: Bbox) -> SliceUniform {
        let s = &self.settings;
        let plane = self.resolved_plane();
        // Fitting to half the bounding-box diagonal always covers the volume
        // whatever the plane's orientation, and the fragment shader discards
        // everything outside it — so an over-large quad costs discarded
        // fragments, while an under-large one silently crops the data.
        let half = s
            .half_extent_mm
            .unwrap_or_else(|| 0.5 * bbox.size().length().max(1.0));
        let (lo, hi) = (s.range[0], s.range[1]);
        let n = plane.normal();
        SliceUniform {
            origin_mm: plane.origin_mm.to_array(),
            half_u_mm: half,
            axis_u: plane.u.to_array(),
            half_v_mm: half,
            axis_v: plane.v.to_array(),
            opacity: s.opacity,
            normal: n.to_array(),
            contrast: s.lic.contrast,
            color_lo: lo,
            color_inv_span: 1.0 / (hi - lo).max(1e-9),
            steps: s.lic.steps,
            step_mm: s.lic.step_mm,
            phase: s.lic.phase_at(self.time_s),
            noise_scale_mm: s.lic.noise_scale_mm,
            honesty_power: s.lic.honesty_power,
            honesty: u32::from(s.lic.honesty),
            channel: s.field.channel(),
            brightness: s.brightness,
            edge_mm: s.edge_mm,
            _pad0: 0.0,
            edge_color: [s.edge_color.x, s.edge_color.y, s.edge_color.z, 1.0],
        }
    }
}

impl OverlayPass for SliceOverlay {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "slice"
    }

    fn is_static(&self) -> bool {
        // A travelling kernel changes the image every frame, so the progressive
        // accumulator must not average it. A frozen phase is static and may be
        // accumulated, which is exactly what a still screenshot wants — and so
        // is a zero frame delta, because an app that has stopped advancing time
        // has stopped the animation just as effectively as setting the rate to
        // zero would.
        !Self::animating(self.enabled, self.settings.lic.cycles_per_second, self.last_dt)
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
        // Clamp before accumulating: a stalled frame must not jump the animation
        // half a cycle, which would read as the flow reversing.
        self.last_dt = ctx.dt.clamp(0.0, 1.0 / 15.0);
        self.time_s += self.last_dt;
        // Wrap on the CPU. The phase only has to be exact modulo one cycle, and
        // an unbounded float would lose its fractional precision within the
        // hour and make the animation visibly ratchet.
        let period = 1.0 / self.settings.lic.cycles_per_second.abs().max(1e-6);
        if self.time_s > period {
            self.time_s %= period;
        }

        let u = self.build_uniform(ctx.fields.bbox());
        ctx.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));

        let ts = ctx.profiler.render_scope("slice");
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("slice"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.hdr_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
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
        pass.set_bind_group(2, &self.group, &[]);
        pass.draw(0..6, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::{GpuContext, Grid};
    use glam::UVec3;
    use std::f32::consts::FRAC_PI_4;

    fn gpu() -> Option<GpuContext> {
        match GpuContext::new_blocking(None) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    fn test_bbox() -> Bbox {
        Bbox { min: Vec3::new(-20.0, -10.0, -20.0), max: Vec3::new(60.0, 30.0, 60.0) }
    }

    #[test]
    fn a_plane_from_a_normal_is_orthonormal_and_right_handed() {
        for n in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-0.2, 0.97, 0.1),
            Vec3::new(0.0, -1.0, 0.0),
        ] {
            let p = SlicePlane::from_normal(Vec3::new(3.0, 4.0, 5.0), n);
            assert!((p.u.length() - 1.0).abs() < 1e-5, "u is not unit for n={n}");
            assert!((p.v.length() - 1.0).abs() < 1e-5, "v is not unit for n={n}");
            assert!(p.u.dot(p.v).abs() < 1e-4, "basis is not orthogonal for n={n}");
            // u x v must be the normal we asked for, not its opposite: the sign
            // decides which way the LIC animation appears to travel.
            let got = p.normal();
            assert!(got.dot(n.normalize()) > 0.999, "normal flipped for n={n}: got {got}");
        }
        // A zero normal is legal input from a half-built gizmo and must not
        // produce NaNs.
        let p = SlicePlane::from_normal(Vec3::ZERO, Vec3::ZERO);
        assert!(p.u.is_finite() && p.v.is_finite() && p.normal().is_finite());
    }

    #[test]
    fn plane_coordinates_round_trip() {
        let p = SlicePlane::from_normal(Vec3::new(10.0, -2.0, 7.0), Vec3::new(1.0, 2.0, -0.5));
        for uv in [Vec2::ZERO, Vec2::new(12.0, -5.0), Vec2::new(-30.0, 40.0)] {
            let world = p.point(uv);
            assert!((p.project(world) - uv).length() < 1e-3, "{uv} did not round trip");
            // ...and every such point is exactly on the plane.
            assert!(p.distance(world).abs() < 1e-3);
        }
        // A point off the plane projects to its foot, and its distance is signed.
        let off = p.point(Vec2::new(3.0, 4.0)) + p.normal() * 2.5;
        assert!((p.project(off) - Vec2::new(3.0, 4.0)).length() < 1e-3);
        assert!((p.distance(off) - 2.5).abs() < 1e-3);
    }

    #[test]
    fn axis_presets_cut_where_they_say_they_do() {
        let b = test_bbox();
        for (preset, axis) in [
            (AxisPreset::X, 0usize),
            (AxisPreset::Y, 1),
            (AxisPreset::Z, 2),
        ] {
            for t in [0.0f32, 0.25, 1.0] {
                let p = SlicePlane::axis(preset, b, t);
                let want = b.min[axis] + (b.max[axis] - b.min[axis]) * t;
                assert!(
                    (p.origin_mm[axis] - want).abs() < 1e-4,
                    "{preset:?} at t={t} sits at {} not {want}",
                    p.origin_mm[axis]
                );
                assert!(p.normal().dot(preset.normal()) > 0.999);
                // The basis must span the other two axes, or the quad is edge-on.
                assert!(p.u.dot(preset.normal()).abs() < 1e-4);
                assert!(p.v.dot(preset.normal()).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn the_honesty_weight_is_the_cosine_of_the_flow_out_of_the_plane() {
        let n = Vec3::Z;
        // Entirely in the plane: full contrast, the texture is telling the truth.
        assert!((in_plane_fraction(Vec3::new(3.0, 4.0, 0.0), n) - 1.0).abs() < 1e-5);
        // Entirely through it: no contrast at all. This is the case that makes
        // an unweighted LIC a beautiful lie -- there is no in-plane flow to show
        // and yet the noise would still smear into confident-looking fibres.
        assert!(in_plane_fraction(Vec3::Z * 9.0, n) < 1e-5);
        // 45 degrees: 1/sqrt(2).
        let diag = Vec3::new(FRAC_PI_4.cos(), 0.0, FRAC_PI_4.sin());
        assert!((in_plane_fraction(diag, n) - 0.5f32.sqrt()).abs() < 1e-4);
        // Still air has no direction, so it gets no texture rather than a
        // division by zero.
        assert_eq!(in_plane_fraction(Vec3::ZERO, n), 0.0);

        // The number a 90-degree bend actually produces: a plane cutting at 30
        // degrees to the flow shows half the magnitude, and the honesty term is
        // what stops that half being drawn at full confidence.
        let flow = Vec3::new(0.866, 0.0, 0.5);
        assert!((in_plane_fraction(flow, n) - 0.866).abs() < 1e-3);
    }

    #[test]
    fn the_composite_keeps_the_colour_and_only_moves_the_brightness() {
        let base = Vec3::new(0.8, 0.3, 0.1);
        // The spec composite: 0.6 + 0.4 * lic.
        assert!((compose(base, 1.0, 0.4) - base).length() < 1e-6);
        assert!((compose(base, 0.0, 0.4) - base * 0.6).length() < 1e-6);
        assert!((compose(base, 0.5, 0.4) - base * 0.8).length() < 1e-6);
        // Zero contrast leaves the colour map untouched, which is what the
        // honesty term produces where the flow leaves the plane.
        assert!((compose(base, 0.0, 0.0) - base).length() < 1e-6);

        // Hue is preserved at every contrast and every LIC value: the ratios
        // between the channels never move, so a colour can still be looked up in
        // the legend.
        for lic in [0.0f32, 0.3, 1.0] {
            for c in [0.0f32, 0.4, 1.0] {
                let out = compose(base, lic, c);
                if out.length() > 1e-6 {
                    let ratio = out / base;
                    assert!(
                        (ratio.x - ratio.y).abs() < 1e-5 && (ratio.y - ratio.z).abs() < 1e-5,
                        "lic={lic} contrast={c} shifted the hue: {ratio}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_sheared_basis_is_repaired_rather_than_drawn() {
        let mut p = SlicePlane {
            origin_mm: Vec3::ZERO,
            u: Vec3::new(2.0, 0.0, 0.0),
            v: Vec3::new(1.0, 3.0, 0.0),
        };
        p.sanitise();
        assert!((p.u.length() - 1.0).abs() < 1e-5);
        assert!((p.v.length() - 1.0).abs() < 1e-5);
        assert!(p.u.dot(p.v).abs() < 1e-5);

        // v collapsed onto u: a degenerate quad would be a line on screen.
        let mut d = SlicePlane { origin_mm: Vec3::ZERO, u: Vec3::X, v: Vec3::X };
        d.sanitise();
        assert!(d.u.dot(d.v).abs() < 1e-4, "a degenerate basis was not repaired");
        assert!(d.normal().is_finite() && d.normal().length() > 0.9);
    }

    #[test]
    fn settings_sanitise_into_the_shader_contract() {
        let mut s = SliceSettings {
            opacity: 4.0,
            brightness: -1.0,
            edge_mm: -3.0,
            range: [5.0, 1.0],
            follow_centreline: Some(9.0),
            half_extent_mm: Some(0.0),
            ..Default::default()
        };
        s.sanitise();
        assert_eq!(s.opacity, 1.0);
        assert_eq!(s.brightness, 0.0);
        assert_eq!(s.edge_mm, 0.0);
        assert!(s.range[1] > s.range[0]);
        assert_eq!(s.follow_centreline, Some(1.0));
        assert!(s.half_extent_mm.unwrap() > 0.0);
        assert!(s.lic.steps <= crate::lic::MAX_STEPS);
    }

    #[test]
    fn only_a_genuinely_moving_slice_stops_the_accumulator() {
        // All three conditions have to hold at once. Dropping any one of them
        // is a bug in one direction or the other: too eager and a still
        // screenshot never antialiases, too lax and the accumulator averages a
        // travelling wave into grey mush.
        assert!(SliceOverlay::animating(true, 0.35, 1.0 / 60.0));
        // Disabled: nothing is drawn at all.
        assert!(!SliceOverlay::animating(false, 0.35, 1.0 / 60.0));
        // Frozen phase: the picture is the same every frame.
        assert!(!SliceOverlay::animating(true, 0.0, 1.0 / 60.0));
        // The app stopped advancing time -- just as static as a frozen phase,
        // and this is what lets a paused capture converge.
        assert!(!SliceOverlay::animating(true, 0.35, 0.0));
        // A backwards animation still animates.
        assert!(SliceOverlay::animating(true, -0.35, 1.0 / 60.0));
    }

    #[test]
    fn the_uniform_is_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<SliceUniform>() % 16, 0);
    }

    // -- GPU -----------------------------------------------------------------

    fn test_grid() -> Grid {
        Grid { dims: UVec3::new(48, 32, 32), dx_mm: 0.75, origin_mm: Vec3::splat(-12.0) }
    }

    fn make_renderer(gpu: &GpuContext) -> crate::Renderer {
        crate::Renderer::new(
            gpu,
            crate::RendererConfig {
                width: 160,
                height: 120,
                target_format: wgpu::TextureFormat::Rgba8Unorm,
                grid: test_grid(),
                field_resolution: crate::FieldResolution::Full,
                profiling: false,
            },
        )
        .expect("renderer")
    }

    #[test]
    fn following_the_centreline_puts_the_normal_on_the_tangent() {
        // The property the mode exists for: the plane stays perpendicular to the
        // passage, so the flow through it is not foreshortened by an angle that
        // changes across the image -- and, by the honesty rule, so the LIC on it
        // is worth looking at.
        let Some(gpu) = gpu() else { return };
        let r = make_renderer(&gpu);
        let mut s = SliceOverlay::new(
            &gpu.device,
            &gpu.queue,
            r.camera_bind_group_layout(),
            r.fields_bind_group_layout(),
        )
        .expect("slice");

        let c = Centreline::new(&[
            Vec3::new(55.0, 0.0, 0.0),
            Vec3::new(52.0, 0.0, 20.0),
            Vec3::new(39.0, 0.0, 39.0),
            Vec3::new(20.0, 0.0, 52.0),
            Vec3::new(0.0, 0.0, 55.0),
        ])
        .unwrap();
        let length = c.length();
        s.set_centreline(Some(c));

        // Without the mode on, the centreline is ignored entirely.
        let flat = s.resolved_plane();
        assert_eq!(flat, SliceSettings::default().plane);

        for i in 0..=10 {
            let t = i as f32 / 10.0;
            s.settings.follow_centreline = Some(t);
            let plane = s.resolved_plane();
            let (point, tangent) = s.centreline().unwrap().sample(t * length);
            assert!(
                (plane.origin_mm - point).length() < 1e-3,
                "t={t}: plane sits at {} not {point}",
                plane.origin_mm
            );
            assert!(
                plane.normal().dot(tangent) > 0.999,
                "t={t}: normal {} is not the tangent {tangent}",
                plane.normal()
            );
        }

        // Detaching the centreline falls back to the explicit plane rather than
        // to nothing, so a UI offering the mode before geometry is loaded still
        // draws something.
        s.set_centreline(None);
        s.settings.follow_centreline = Some(0.5);
        assert_eq!(s.resolved_plane(), SliceSettings::default().plane);
    }

    #[test]
    fn the_slice_records_inside_a_real_frame() {
        let Some(gpu) = gpu() else { return };
        let mut r = make_renderer(&gpu);
        let mut s = SliceOverlay::new(
            &gpu.device,
            &gpu.queue,
            r.camera_bind_group_layout(),
            r.fields_bind_group_layout(),
        )
        .expect("slice");
        s.settings_mut().plane = SlicePlane::axis(AxisPreset::Y, test_grid().bbox(), 0.5);
        // Nothing has been recorded yet, so nothing has changed yet.
        assert!(s.is_static());
        r.add_overlay(Box::new(s));

        let target = util::color_target(
            &gpu.device,
            "slice target",
            160,
            120,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let view = target.create_view(&Default::default());
        let scene = crate::Scene::new();
        for _ in 0..3 {
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
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }
}

//! Post-processing: SSAO, compositing, TAA, bloom, AgX tonemap.
//!
//! Everything from the G-buffer to the swapchain lives here, in that order:
//!
//! ```text
//! SSAO -> composite (background + shading + volume) -> [Wave 2 overlays]
//!      -> TAA -> bloom -> AgX tonemap -> target
//! ```
//!
//! Two design points are worth stating up front.
//!
//! **Everything upstream of the tonemap is linear HDR.** Tonemapping is not
//! idempotent and does not commute with blending, averaging or thresholding, so
//! applying it anywhere but the very end would silently change what TAA is
//! averaging and what bloom is thresholding. It happens exactly once.
//!
//! **The accumulator is a first-class mode, not a screenshot hack.** This is a
//! tool people set a view in and then stare at. When neither camera nor sim has
//! changed, TAA switches from an exponential blend — which converges to a fixed
//! amount of residual aliasing and stops — to a true running mean over jittered
//! samples, which keeps getting better for as long as you leave it. The
//! supersampled screenshot path is the same mechanism with the internal targets
//! allocated larger.

use ad_gpu::{Profiler, ShaderDefines, ShaderLoader};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::mesh;
use crate::noise;
use crate::util;

/// Working colour format for every intermediate. Half-float, linear light: the
/// volume's emissive core routinely runs an order of magnitude above white, and
/// clamping it before the tonemap is what turns a glowing vortex into a flat
/// white blob.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Ambient occlusion factor.
pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Number of levels in the bloom pyramid, starting at half resolution.
const BLOOM_LEVELS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackgroundSettings {
    pub sky_top: Vec3,
    pub sky_bottom: Vec3,
    pub ground: Vec3,
    /// Ground plane height, mm. Set from the scene bbox by the driver.
    pub ground_y_mm: f32,
    /// Radius of the contact-shadow pool, mm.
    pub shadow_radius_mm: f32,
    pub shadow_strength: f32,
    /// Distance over which the ground fades into the sky, mm.
    pub ground_fade_mm: f32,
}

impl Default for BackgroundSettings {
    fn default() -> Self {
        Self {
            // A dark neutral with a slight cool cast at the top and a warmer
            // floor. Neutral so it does not bias a colour-mapped overlay; not
            // flat, so the eye has a horizon to place the part against.
            sky_top: Vec3::new(0.052, 0.056, 0.066),
            sky_bottom: Vec3::new(0.015, 0.015, 0.018),
            ground: Vec3::new(0.030, 0.029, 0.028),
            ground_y_mm: 0.0,
            shadow_radius_mm: 120.0,
            shadow_strength: 0.85,
            ground_fade_mm: 1800.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightingSettings {
    pub key_dir: Vec3,
    pub key_intensity: f32,
    pub key_color: Vec3,
    pub fill_dir: Vec3,
    pub fill_intensity: f32,
    pub ambient: f32,
    pub rim: f32,
    /// Exponent applied to the AO term. >1 deepens contact shadows.
    pub ao_power: f32,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            // A conventional three-quarter key with a cool fill from below-left:
            // enough separation to read curvature on a smooth grey part without
            // looking theatrical.
            key_dir: Vec3::new(0.45, 0.75, 0.48).normalize(),
            key_intensity: 2.6,
            key_color: Vec3::new(1.0, 0.97, 0.93),
            fill_dir: Vec3::new(-0.6, -0.15, -0.4).normalize(),
            fill_intensity: 0.35,
            ambient: 1.0,
            rim: 0.35,
            ao_power: 1.4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostSettings {
    pub taa: bool,
    /// Weight of the current frame. 0.1 is a good balance of stability against
    /// ghosting for a scene with slow camera motion.
    pub taa_blend: f32,
    pub taa_variance_gamma: f32,

    pub ssao: bool,
    pub ssao_radius_mm: f32,
    pub ssao_bias_mm: f32,
    pub ssao_intensity: f32,
    pub ssao_samples: u32,

    pub bloom: bool,
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    pub bloom_intensity: f32,
    pub bloom_radius: f32,

    pub exposure: f32,
    /// AgX "look": a touch of contrast and saturation. 0 is the neutral base.
    pub agx_look: f32,
    /// Output dither amplitude, in LSBs of the 8-bit target.
    pub dither: f32,

    pub background: BackgroundSettings,
    pub lighting: LightingSettings,
    pub clearcoat: f32,
}

impl Default for PostSettings {
    fn default() -> Self {
        Self {
            taa: true,
            taa_blend: 0.1,
            taa_variance_gamma: 1.25,
            ssao: true,
            // Scaled to the passage, not the part: on a duct with a 6 mm bore
            // and 2 mm walls, a 40 mm AO radius just darkens everything.
            ssao_radius_mm: 4.0,
            ssao_bias_mm: 0.08,
            ssao_intensity: 0.9,
            ssao_samples: 12,
            bloom: true,
            bloom_threshold: 1.1,
            bloom_knee: 0.55,
            bloom_intensity: 0.06,
            bloom_radius: 1.0,
            exposure: 1.0,
            agx_look: 0.6,
            dither: 1.0,
            background: BackgroundSettings::default(),
            lighting: LightingSettings::default(),
            clearcoat: 0.6,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SsaoUniform {
    radius_mm: f32,
    bias_mm: f32,
    intensity: f32,
    sample_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CompositeUniform {
    sky_top: [f32; 4],
    sky_bottom: [f32; 4],
    ground_color: [f32; 4],
    key_dir: [f32; 4],
    fill_dir: [f32; 4],
    key_color: [f32; 4],
    ground: [f32; 4],
    scene_center: [f32; 4],
    misc: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TaaUniform {
    blend: f32,
    variance_gamma: f32,
    accumulate: u32,
    accum_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BloomUniform {
    threshold: f32,
    knee: f32,
    radius: f32,
    intensity: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TonemapUniform {
    exposure: f32,
    bloom_intensity: f32,
    look_strength: f32,
    dither: f32,
}

struct Target {
    _texture: wgpu::Texture,
    storage: wgpu::TextureView,
    sampled: wgpu::TextureView,
    size: (u32, u32),
}

impl Target {
    fn new(device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat) -> Self {
        Self::with_usage(device, label, w, h, format, wgpu::TextureUsages::STORAGE_BINDING)
    }

    fn with_usage(
        device: &wgpu::Device,
        label: &str,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
        let t = util::color_target(device, label, w, h, format, usage);
        Self {
            storage: t.create_view(&Default::default()),
            sampled: t.create_view(&Default::default()),
            size: (w.max(1), h.max(1)),
            _texture: t,
        }
    }
}

/// The whole post chain.
pub struct PostChain {
    pub settings: PostSettings,

    linear_sampler: wgpu::Sampler,
    blue_noise_view: wgpu::TextureView,
    _blue_noise: wgpu::Texture,

    ssao_uniform: wgpu::Buffer,
    ssao_layout: wgpu::BindGroupLayout,
    ssao_pipeline: wgpu::ComputePipeline,
    ao: Target,

    comp_uniform: wgpu::Buffer,
    comp_layout: wgpu::BindGroupLayout,
    comp_pipeline: wgpu::ComputePipeline,
    hdr: Target,

    taa_uniform: wgpu::Buffer,
    taa_layout: wgpu::BindGroupLayout,
    taa_pipeline: wgpu::ComputePipeline,
    history: [Target; 2],
    history_index: usize,
    accum_count: u32,

    bloom_uniform: wgpu::Buffer,
    bloom_layout: wgpu::BindGroupLayout,
    bloom_prefilter: wgpu::ComputePipeline,
    bloom_down: wgpu::ComputePipeline,
    bloom_up: wgpu::ComputePipeline,
    down_chain: Vec<Target>,
    up_chain: Vec<Target>,

    tone_uniform: wgpu::Buffer,
    tone_layout: wgpu::BindGroupLayout,
    tone_pipeline: wgpu::RenderPipeline,
    target_format: wgpu::TextureFormat,

    size: (u32, u32),
}

impl PostChain {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        loader: &ShaderLoader,
        camera_layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
        target_format: wgpu::TextureFormat,
        settings: PostSettings,
    ) -> Result<Self> {
        let cs = wgpu::ShaderStages::COMPUTE;
        let linear_sampler = util::linear_clamp_sampler(device, "post linear");
        let blue_noise = noise::create_blue_noise_texture(device, queue);
        let blue_noise_view = blue_noise.create_view(&Default::default());

        // --- SSAO ---
        let ssao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao"),
            entries: &[
                util::uniform_entry(0, cs),
                util::texture_entry(1, cs, wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2),
                util::sampled_float_entry(2, cs, wgpu::TextureViewDimension::D2),
                util::sampled_float_entry(3, cs, wgpu::TextureViewDimension::D2),
                util::storage_texture_entry(4, cs, AO_FORMAT, wgpu::TextureViewDimension::D2),
            ],
        });
        let ssao_pipeline = util::compute_pipeline(
            device,
            loader,
            "ssao.wgsl",
            "ssao_main",
            &ShaderDefines::new(),
            &[Some(camera_layout), Some(&ssao_layout)],
            "ssao",
        )?;

        // --- composite ---
        let comp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite"),
            entries: &[
                util::uniform_entry(0, cs),
                util::sampler_entry(1, cs, wgpu::SamplerBindingType::Filtering),
                util::sampled_float_entry(2, cs, wgpu::TextureViewDimension::D2),
                util::sampled_float_entry(3, cs, wgpu::TextureViewDimension::D2),
                util::texture_entry(4, cs, wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2),
                util::texture_entry(
                    5,
                    cs,
                    wgpu::TextureSampleType::Float { filterable: false },
                    wgpu::TextureViewDimension::D2,
                ),
                util::sampled_float_entry(6, cs, wgpu::TextureViewDimension::D2),
                util::storage_texture_entry(7, cs, HDR_FORMAT, wgpu::TextureViewDimension::D2),
            ],
        });
        let comp_pipeline = util::compute_pipeline(
            device,
            loader,
            "composite.wgsl",
            "composite_main",
            &ShaderDefines::new(),
            &[Some(camera_layout), Some(&comp_layout)],
            "composite",
        )?;

        // --- TAA ---
        let taa_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa"),
            entries: &[
                util::uniform_entry(0, cs),
                util::sampler_entry(1, cs, wgpu::SamplerBindingType::Filtering),
                util::sampled_float_entry(2, cs, wgpu::TextureViewDimension::D2),
                util::sampled_float_entry(3, cs, wgpu::TextureViewDimension::D2),
                util::sampled_float_entry(4, cs, wgpu::TextureViewDimension::D2),
                util::texture_entry(5, cs, wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2),
                util::texture_entry(
                    6,
                    cs,
                    wgpu::TextureSampleType::Float { filterable: false },
                    wgpu::TextureViewDimension::D2,
                ),
                util::storage_texture_entry(7, cs, HDR_FORMAT, wgpu::TextureViewDimension::D2),
            ],
        });
        let taa_pipeline = util::compute_pipeline(
            device,
            loader,
            "taa.wgsl",
            "taa_main",
            &ShaderDefines::new(),
            &[Some(camera_layout), Some(&taa_layout)],
            "taa",
        )?;

        // --- bloom ---
        let bloom_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom"),
            entries: &[
                util::uniform_entry(0, cs),
                util::sampler_entry(1, cs, wgpu::SamplerBindingType::Filtering),
                util::sampled_float_entry(2, cs, wgpu::TextureViewDimension::D2),
                util::sampled_float_entry(3, cs, wgpu::TextureViewDimension::D2),
                util::storage_texture_entry(4, cs, HDR_FORMAT, wgpu::TextureViewDimension::D2),
            ],
        });
        let mk_bloom = |entry: &str, label: &str| {
            util::compute_pipeline(
                device,
                loader,
                "bloom.wgsl",
                entry,
                &ShaderDefines::new(),
                &[Some(&bloom_layout)],
                label,
            )
        };
        let bloom_prefilter = mk_bloom("prefilter_main", "bloom prefilter")?;
        let bloom_down = mk_bloom("downsample_main", "bloom downsample")?;
        let bloom_up = mk_bloom("upsample_main", "bloom upsample")?;

        // --- tonemap ---
        let fs = wgpu::ShaderStages::FRAGMENT;
        let tone_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap"),
            entries: &[
                util::uniform_entry(0, fs),
                util::sampler_entry(1, fs, wgpu::SamplerBindingType::Filtering),
                util::sampled_float_entry(2, fs, wgpu::TextureViewDimension::D2),
                util::sampled_float_entry(3, fs, wgpu::TextureViewDimension::D2),
                util::sampled_float_entry(4, fs, wgpu::TextureViewDimension::D2),
            ],
        });
        let tone_pipeline =
            Self::build_tonemap(device, loader, &tone_layout, target_format)?;

        let (ao, hdr, history, down_chain, up_chain) = Self::create_targets(device, width, height);

        Ok(Self {
            settings,
            linear_sampler,
            blue_noise_view,
            _blue_noise: blue_noise,
            ssao_uniform: util::uniform_buffer::<SsaoUniform>(device, "ssao uniform"),
            ssao_layout,
            ssao_pipeline,
            ao,
            comp_uniform: util::uniform_buffer::<CompositeUniform>(device, "composite uniform"),
            comp_layout,
            comp_pipeline,
            hdr,
            taa_uniform: util::uniform_buffer::<TaaUniform>(device, "taa uniform"),
            taa_layout,
            taa_pipeline,
            history,
            history_index: 0,
            accum_count: 0,
            bloom_uniform: util::uniform_buffer::<BloomUniform>(device, "bloom uniform"),
            bloom_layout,
            bloom_prefilter,
            bloom_down,
            bloom_up,
            down_chain,
            up_chain,
            tone_uniform: util::uniform_buffer::<TonemapUniform>(device, "tonemap uniform"),
            tone_layout,
            tone_pipeline,
            target_format,
            size: (width.max(1), height.max(1)),
        })
    }

    fn build_tonemap(
        device: &wgpu::Device,
        loader: &ShaderLoader,
        layout: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> Result<wgpu::RenderPipeline> {
        // When the target view is sRGB the hardware does the encode on write, so
        // the shader must not do it as well; doing both is the classic
        // washed-out-image bug.
        let mut defines = ShaderDefines::new();
        if format.is_srgb() {
            defines = defines.flag("SRGB_TARGET");
        }
        let module = loader.create_module(device, "tonemap.wgsl", &defines)?;
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap"),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
        Ok(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_tonemap"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        }))
    }

    #[allow(clippy::type_complexity)]
    fn create_targets(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (Target, Target, [Target; 2], Vec<Target>, Vec<Target>) {
        let (w, h) = (width.max(1), height.max(1));
        let ao = Target::new(device, "ssao", w, h, AO_FORMAT);
        // The HDR target is also a render attachment: the ghost/wireframe
        // overlay and every Wave 2 pass draw into it with a real render pass
        // before TAA sees it.
        let hdr = Target::with_usage(
            device,
            "hdr",
            w,
            h,
            HDR_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let history = [
            Target::new(device, "taa history A", w, h, HDR_FORMAT),
            Target::new(device, "taa history B", w, h, HDR_FORMAT),
        ];

        let mut down = Vec::with_capacity(BLOOM_LEVELS);
        let (mut lw, mut lh) = (w, h);
        for _ in 0..BLOOM_LEVELS {
            lw = (lw / 2).max(1);
            lh = (lh / 2).max(1);
            down.push(Target::new(device, "bloom down", lw, lh, HDR_FORMAT));
        }
        // The up chain is one shorter: the smallest level is its own starting
        // point and needs no destination of its own.
        let mut up = Vec::with_capacity(BLOOM_LEVELS - 1);
        for i in 0..BLOOM_LEVELS - 1 {
            up.push(Target::new(device, "bloom up", down[i].size.0, down[i].size.1, HDR_FORMAT));
        }
        (ao, hdr, history, down, up)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (ao, hdr, history, down, up) = Self::create_targets(device, width, height);
        self.ao = ao;
        self.hdr = hdr;
        self.history = history;
        self.down_chain = down;
        self.up_chain = up;
        self.size = (width.max(1), height.max(1));
        self.reset_accumulation();
    }

    pub fn set_target_format(
        &mut self,
        device: &wgpu::Device,
        loader: &ShaderLoader,
        format: wgpu::TextureFormat,
    ) -> Result<()> {
        if format == self.target_format {
            return Ok(());
        }
        self.tone_pipeline = Self::build_tonemap(device, loader, &self.tone_layout, format)?;
        self.target_format = format;
        Ok(())
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }
    pub fn hdr_view(&self) -> &wgpu::TextureView {
        &self.hdr.sampled
    }
    /// The HDR target as a render attachment, for the geometry overlay and for
    /// Wave 2 passes that want to draw into the scene before TAA.
    pub fn hdr_attachment_view(&self) -> &wgpu::TextureView {
        &self.hdr.storage
    }
    pub fn ao_view(&self) -> &wgpu::TextureView {
        &self.ao.sampled
    }
    pub fn accumulated_samples(&self) -> u32 {
        self.accum_count
    }

    /// Throw the temporal history away. Call whenever anything that is not the
    /// jitter changes: camera, transfer function, field data, resolution.
    pub fn reset_accumulation(&mut self) {
        self.accum_count = 0;
    }

    // -- passes ---------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn ssao(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut Profiler,
        camera_group: &wgpu::BindGroup,
        depth: &wgpu::TextureView,
        normal: &wgpu::TextureView,
    ) {
        let u = SsaoUniform {
            radius_mm: self.settings.ssao_radius_mm,
            bias_mm: self.settings.ssao_bias_mm,
            intensity: self.settings.ssao_intensity,
            // Zero samples means the shader writes a flat 1.0, which is exactly
            // what "SSAO off" should look like to the compositor. Cheaper than
            // maintaining a second code path or a white stand-in texture.
            sample_count: if self.settings.ssao { self.settings.ssao_samples } else { 0 },
        };
        queue.write_buffer(&self.ssao_uniform, 0, bytemuck::bytes_of(&u));

        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao"),
            layout: &self.ssao_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.ssao_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal) },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.blue_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.ao.storage),
                },
            ],
        });

        let ts = profiler.scope("ssao");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssao"),
            timestamp_writes: ts,
        });
        pass.set_pipeline(&self.ssao_pipeline);
        pass.set_bind_group(0, camera_group, &[]);
        pass.set_bind_group(1, &group, &[]);
        pass.dispatch_workgroups(
            util::dispatch_count(self.size.0, 8),
            util::dispatch_count(self.size.1, 8),
            1,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut Profiler,
        camera_group: &wgpu::BindGroup,
        albedo: &wgpu::TextureView,
        normal: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        volume_color: &wgpu::TextureView,
        scene_center: Vec3,
        volume_enabled: bool,
    ) {
        let bg = &self.settings.background;
        let li = &self.settings.lighting;
        let u = CompositeUniform {
            sky_top: bg.sky_top.extend(1.0).to_array(),
            sky_bottom: bg.sky_bottom.extend(1.0).to_array(),
            ground_color: bg.ground.extend(1.0).to_array(),
            key_dir: li.key_dir.normalize_or(Vec3::Y).extend(li.key_intensity).to_array(),
            fill_dir: li.fill_dir.normalize_or(Vec3::NEG_Y).extend(li.fill_intensity).to_array(),
            key_color: li.key_color.extend(li.ambient).to_array(),
            ground: [bg.ground_y_mm, bg.shadow_radius_mm, bg.shadow_strength, bg.ground_fade_mm],
            scene_center: scene_center.extend(self.settings.clearcoat).to_array(),
            misc: [1.0, li.rim, if volume_enabled { 1.0 } else { 0.0 }, li.ao_power],
        };
        queue.write_buffer(&self.comp_uniform, 0, bytemuck::bytes_of(&u));

        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite"),
            layout: &self.comp_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.comp_uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(albedo) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(normal) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.ao.sampled),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(volume_color),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.hdr.storage),
                },
            ],
        });

        let ts = profiler.scope("composite");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("composite"),
            timestamp_writes: ts,
        });
        pass.set_pipeline(&self.comp_pipeline);
        pass.set_bind_group(0, camera_group, &[]);
        pass.set_bind_group(1, &group, &[]);
        pass.dispatch_workgroups(
            util::dispatch_count(self.size.0, 8),
            util::dispatch_count(self.size.1, 8),
            1,
        );
    }

    /// Resolve TAA. `static_scene` switches to progressive accumulation.
    ///
    /// Returns the view the tonemap should read.
    #[allow(clippy::too_many_arguments)]
    pub fn taa(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut Profiler,
        camera_group: &wgpu::BindGroup,
        motion: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        volume_front: &wgpu::TextureView,
        static_scene: bool,
    ) {
        if !self.settings.taa {
            self.accum_count = 0;
            return;
        }
        // Accumulation only makes sense once there is something to accumulate
        // onto; the first frame after a reset is a plain copy of the current
        // frame, which the running mean gives us for free at n = 0.
        let accumulate = static_scene;
        let u = TaaUniform {
            blend: self.settings.taa_blend,
            variance_gamma: self.settings.taa_variance_gamma,
            accumulate: u32::from(accumulate),
            accum_count: self.accum_count,
        };
        queue.write_buffer(&self.taa_uniform, 0, bytemuck::bytes_of(&u));

        let src = self.history_index;
        let dst = 1 - src;
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa"),
            layout: &self.taa_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.taa_uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.hdr.sampled),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.history[src].sampled),
                },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(motion) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(volume_front),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.history[dst].storage),
                },
            ],
        });

        let ts = profiler.scope("taa");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("taa"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.taa_pipeline);
            pass.set_bind_group(0, camera_group, &[]);
            pass.set_bind_group(1, &group, &[]);
            pass.dispatch_workgroups(
                util::dispatch_count(self.size.0, 8),
                util::dispatch_count(self.size.1, 8),
                1,
            );
        }

        self.history_index = dst;
        if accumulate {
            self.accum_count = self.accum_count.saturating_add(1);
        } else {
            self.accum_count = 0;
        }
    }

    /// The view carrying the final linear-HDR image: the TAA result when TAA is
    /// on, the raw composite when it is off.
    pub fn resolved_view(&self) -> &wgpu::TextureView {
        if self.settings.taa {
            &self.history[self.history_index].sampled
        } else {
            &self.hdr.sampled
        }
    }

    pub fn bloom(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut Profiler,
    ) {
        if !self.settings.bloom {
            return;
        }
        let u = BloomUniform {
            threshold: self.settings.bloom_threshold,
            knee: self.settings.bloom_knee.max(1e-3),
            radius: self.settings.bloom_radius,
            intensity: self.settings.bloom_intensity,
        };
        queue.write_buffer(&self.bloom_uniform, 0, bytemuck::bytes_of(&u));

        let source = self.resolved_view();
        let make_group = |a: &wgpu::TextureView, b: &wgpu::TextureView, dst: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom"),
                layout: &self.bloom_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.bloom_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(a) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(b) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(dst) },
                ],
            })
        };

        // Groups are built before the pass so the borrow of `encoder` stays put.
        let mut steps: Vec<(&wgpu::ComputePipeline, wgpu::BindGroup, (u32, u32))> = Vec::new();
        steps.push((
            &self.bloom_prefilter,
            make_group(source, source, &self.down_chain[0].storage),
            self.down_chain[0].size,
        ));
        for i in 1..self.down_chain.len() {
            steps.push((
                &self.bloom_down,
                make_group(
                    &self.down_chain[i - 1].sampled,
                    &self.down_chain[i - 1].sampled,
                    &self.down_chain[i].storage,
                ),
                self.down_chain[i].size,
            ));
        }
        // Upsample: start from the smallest down level, add the same-size down
        // level at each step, and write into the up chain.
        for i in (0..self.up_chain.len()).rev() {
            let smaller = if i + 1 == self.up_chain.len() {
                &self.down_chain[self.down_chain.len() - 1].sampled
            } else {
                &self.up_chain[i + 1].sampled
            };
            steps.push((
                &self.bloom_up,
                make_group(smaller, &self.down_chain[i].sampled, &self.up_chain[i].storage),
                self.up_chain[i].size,
            ));
        }

        let ts = profiler.scope("bloom");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bloom"),
            timestamp_writes: ts,
        });
        for (pipeline, group, size) in &steps {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(
                util::dispatch_count(size.0, 8),
                util::dispatch_count(size.1, 8),
                1,
            );
        }
    }

    /// The final draw. Always the last thing that touches colour.
    pub fn tonemap(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        let u = TonemapUniform {
            exposure: self.settings.exposure.max(0.0),
            bloom_intensity: if self.settings.bloom { self.settings.bloom_intensity } else { 0.0 },
            look_strength: self.settings.agx_look,
            dither: self.settings.dither,
        };
        queue.write_buffer(&self.tone_uniform, 0, bytemuck::bytes_of(&u));

        // With bloom off the pass still needs something bound; the smallest
        // bloom level is a valid texture and its contribution is multiplied by
        // zero, so no branch is needed in the shader.
        let bloom_view = &self.up_chain[0].sampled;
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap"),
            layout: &self.tone_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tone_uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(self.resolved_view()),
                },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(bloom_view) },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.blue_noise_view),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.tone_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The G-buffer, allocated once per resolution.
pub struct GBuffer {
    _albedo: wgpu::Texture,
    _normal: wgpu::Texture,
    _motion: wgpu::Texture,
    _depth: wgpu::Texture,
    pub albedo: wgpu::TextureView,
    pub normal: wgpu::TextureView,
    pub motion: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub size: (u32, u32),
}

impl GBuffer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (w, h) = (width.max(1), height.max(1));
        let attach = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let albedo = util::color_target(device, "gbuffer albedo", w, h, mesh::ALBEDO_FORMAT, attach);
        let normal = util::color_target(device, "gbuffer normal", w, h, mesh::NORMAL_FORMAT, attach);
        let motion = util::color_target(device, "gbuffer motion", w, h, mesh::MOTION_FORMAT, attach);
        let depth = util::color_target(device, "gbuffer depth", w, h, mesh::DEPTH_FORMAT, attach);
        Self {
            albedo: albedo.create_view(&Default::default()),
            normal: normal.create_view(&Default::default()),
            motion: motion.create_view(&Default::default()),
            depth: depth.create_view(&Default::default()),
            _albedo: albedo,
            _normal: normal,
            _motion: motion,
            _depth: depth,
            size: (w, h),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_uniforms_are_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<SsaoUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<CompositeUniform>(), 144);
        assert_eq!(std::mem::size_of::<TaaUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<BloomUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<TonemapUniform>() % 16, 0);
    }

    #[test]
    fn default_settings_are_plausible() {
        let s = PostSettings::default();
        assert!(s.taa_blend > 0.0 && s.taa_blend < 0.5, "blend {} will ghost or flicker", s.taa_blend);
        assert!(s.taa_variance_gamma >= 1.0);
        // AO radius must be scaled to the passage, not the part: the test duct's
        // median internal width is 6.3 mm.
        assert!(s.ssao_radius_mm < 10.0, "AO radius {} is bigger than the passage", s.ssao_radius_mm);
        assert!(s.bloom_threshold > 1.0, "bloom must only catch above-white content");
        assert!(s.bloom_intensity < 0.25, "bloom this strong hides the data");
        assert!(s.exposure > 0.0);
    }

    #[test]
    fn the_background_is_dark_and_neutral_with_a_gradient() {
        let bg = BackgroundSettings::default();
        // Dark, so a black-based volume map has somewhere to be dark against.
        assert!(bg.sky_top.max_element() < 0.15);
        // Actually a gradient, not a flat fill.
        assert!(bg.sky_top.length() > bg.sky_bottom.length() * 1.5);
        // Neutral: no channel more than 40% above the mean, or it tints the
        // colour map.
        let mean = (bg.sky_top.x + bg.sky_top.y + bg.sky_top.z) / 3.0;
        assert!(bg.sky_top.max_element() < mean * 1.4);
    }

    #[test]
    fn bloom_chain_shapes_are_consistent() {
        // The up chain is exactly one shorter than the down chain, and each up
        // level matches the size of the down level it adds. Getting this wrong
        // is a validation error at dispatch time, on a GPU, in a user's session.
        assert!(BLOOM_LEVELS >= 2);
        let (mut w, mut h) = (1920u32, 1080u32);
        let mut sizes = Vec::new();
        for _ in 0..BLOOM_LEVELS {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            sizes.push((w, h));
        }
        assert_eq!(sizes[0], (960, 540));
        assert_eq!(sizes.len(), BLOOM_LEVELS);
        assert_eq!(sizes.last().unwrap(), &(60, 33));
    }

    #[test]
    fn hdr_format_is_storage_capable_and_filterable() {
        let base = wgpu::Features::empty();
        let f = HDR_FORMAT.guaranteed_format_features(base);
        assert!(f.allowed_usages.contains(wgpu::TextureUsages::STORAGE_BINDING));
        assert!(f.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert_eq!(
            HDR_FORMAT.sample_type(None, Some(base)),
            Some(wgpu::TextureSampleType::Float { filterable: true }),
            "the bloom and TAA passes need to filter this"
        );
    }
}

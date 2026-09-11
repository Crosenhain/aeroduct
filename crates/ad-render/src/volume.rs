//! The volume raymarching pass.
//!
//! Runs at a configurable fraction of the output resolution (half by default)
//! into a premultiplied `Rgba16Float` target, plus an `R32Float` "front depth"
//! target that TAA uses to reproject a medium that has no motion vectors of its
//! own. See `shaders/render/volume.wgsl` for the marching itself.
//!
//! # The reference step
//!
//! Opacities coming out of the transfer function are defined *per reference
//! step*, and the reference step is one derived voxel. The raymarch then takes
//! whatever step it likes and corrects with
//! `alpha_c = 1 - (1 - alpha_ref)^(h / h_ref)`. That is what makes the image
//! independent of `step_scale`, which in turn is what makes empty-space skipping
//! debuggable: with the correction in place, "skipping on" and "skipping off"
//! must match, so any difference is a bug rather than a tuning question.

use ad_gpu::{Profiler, ShaderDefines, ShaderLoader};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::colormap;
use crate::noise;
use crate::transfer::TransferFunction;
use crate::util;

/// Format of the half-resolution volume target. Premultiplied RGBA.
pub const VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Format of the volume "front distance" target, in millimetres along the ray.
pub const FRONT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// User-facing raymarch controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeSettings {
    /// Raymarch step as a fraction of a derived voxel. 1.0 is the sane default:
    /// the field is already band-limited to the voxel grid, so stepping finer
    /// costs time and adds nothing the transfer function can see.
    pub step_scale: f32,
    /// Extra opacity multiplier on top of the baked LUT. Left at 1.0 normally;
    /// the density *slider* re-bakes the LUT instead, so there is one source of
    /// truth for what a value's opacity is.
    pub density: f32,
    /// Hard cap on marching iterations, as a backstop against a degenerate ray.
    pub max_steps: u32,
    /// Gradient-based shading strength, 0 = pure emission-absorption.
    pub shading: f32,
    pub specular: f32,
    pub ambient: Vec3,
    /// Direction *toward* the key light.
    pub light_dir: Vec3,
    /// Turn empty-space skipping off. Only useful for verifying that it is
    /// invisible, which — with the opacity correction — it must be.
    pub skip_empty_space: bool,
    /// Accumulated alpha at which the volume's "front" is recorded for TAA.
    pub front_alpha: f32,
    /// Render resolution as a fraction of the output. 2 = half res.
    pub resolution_divisor: u32,
}

impl Default for VolumeSettings {
    fn default() -> Self {
        Self {
            step_scale: 1.0,
            density: 1.0,
            max_steps: 2048,
            shading: 0.75,
            specular: 0.35,
            ambient: Vec3::new(0.16, 0.18, 0.22),
            light_dir: Vec3::new(0.45, 0.75, 0.48).normalize(),
            skip_empty_space: true,
            front_alpha: 0.1,
            resolution_divisor: 2,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VolumeUniform {
    step_scale: f32,
    h_ref_mm: f32,
    density: f32,
    max_steps: u32,

    tf_lo: f32,
    tf_inv_span: f32,
    tf_log: u32,
    tf_show_clamp: u32,

    under_color: [f32; 4],
    over_color: [f32; 4],
    light_dir: [f32; 4],

    jitter_offset: f32,
    skip_enabled: u32,
    sdf_present: u32,
    front_alpha: f32,

    sdf_min_mm: [f32; 4],
    sdf_inv_size: [f32; 4],
    ambient: [f32; 4],
}

/// A signed distance field of the solid geometry, supplied by the geometry
/// crate. Optional, and purely an accelerator: the raymarcher sphere-traces
/// through solid material instead of stepping through it.
pub struct SdfVolume<'a> {
    pub view: &'a wgpu::TextureView,
    /// World-space corner of the SDF volume, mm.
    pub min_mm: Vec3,
    /// World-space size of the SDF volume, mm.
    pub size_mm: Vec3,
    /// Offset applied to the sampled distance, mm. Use a small positive value to
    /// keep the march clear of the wall and avoid sampling half-solid voxels.
    pub surface_offset_mm: f32,
}

pub struct VolumeRenderer {
    settings: VolumeSettings,
    uniform: wgpu::Buffer,

    lut: wgpu::Texture,
    lut_view: wgpu::TextureView,
    lut_sampler: wgpu::Sampler,
    blue_noise_view: wgpu::TextureView,
    _blue_noise: wgpu::Texture,

    /// 1x1x1 stand-in bound when the caller has no SDF. Its single texel holds a
    /// large positive distance, so `sdf < 0` is never true and the sphere-trace
    /// branch is inert.
    _sdf_fallback: wgpu::Texture,
    sdf_fallback_view: wgpu::TextureView,

    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,

    color: wgpu::Texture,
    color_storage: wgpu::TextureView,
    color_sampled: wgpu::TextureView,
    front: wgpu::Texture,
    front_storage: wgpu::TextureView,
    front_sampled: wgpu::TextureView,
    size: (u32, u32),

    jitter: f32,
}

impl VolumeRenderer {
    const WG: (u32, u32) = (8, 8);

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        loader: &ShaderLoader,
        camera_layout: &wgpu::BindGroupLayout,
        fields_layout: &wgpu::BindGroupLayout,
        brick_layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
        settings: VolumeSettings,
    ) -> Result<Self> {
        let cs = wgpu::ShaderStages::COMPUTE;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume"),
            entries: &[
                util::uniform_entry(0, cs),
                util::sampled_float_entry(1, cs, wgpu::TextureViewDimension::D2),
                util::sampler_entry(2, cs, wgpu::SamplerBindingType::Filtering),
                util::texture_entry(
                    3,
                    cs,
                    wgpu::TextureSampleType::Float { filterable: false },
                    wgpu::TextureViewDimension::D2,
                ),
                util::texture_entry(4, cs, wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2),
                util::sampled_float_entry(5, cs, wgpu::TextureViewDimension::D3),
                util::storage_texture_entry(6, cs, VOLUME_FORMAT, wgpu::TextureViewDimension::D2),
                util::storage_texture_entry(7, cs, FRONT_FORMAT, wgpu::TextureViewDimension::D2),
            ],
        });

        let defines = ShaderDefines::new()
            .value("VOLUME_WG_X", Self::WG.0)
            .value("VOLUME_WG_Y", Self::WG.1);
        let pipeline = util::compute_pipeline(
            device,
            loader,
            "volume.wgsl",
            "volume_main",
            &defines,
            &[
                Some(camera_layout),
                Some(fields_layout),
                Some(brick_layout),
                Some(&layout),
            ],
            "volume raymarch",
        )?;

        // 256x1 transfer-function LUT. Rgba16Float so the colour stays in linear
        // light with plenty of headroom at the dark end, where an 8-bit LUT
        // would quantise a black-based volume map into visible steps.
        let lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("transfer function LUT"),
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
        let lut_sampler = util::linear_clamp_sampler(device, "transfer function LUT");

        let blue_noise = noise::create_blue_noise_texture(device, queue);
        let blue_noise_view = blue_noise.create_view(&Default::default());

        let sdf_fallback = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("empty SDF stand-in"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &sdf_fallback,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::bytes_of(&1.0e6f32),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let sdf_fallback_view = sdf_fallback.create_view(&wgpu::TextureViewDescriptor {
            label: Some("empty SDF stand-in"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let (color, color_storage, color_sampled, front, front_storage, front_sampled, size) =
            Self::create_targets(device, width, height, settings.resolution_divisor);

        Ok(Self {
            settings,
            uniform: util::uniform_buffer::<VolumeUniform>(device, "volume uniform"),
            lut,
            lut_view,
            lut_sampler,
            blue_noise_view,
            _blue_noise: blue_noise,
            _sdf_fallback: sdf_fallback,
            sdf_fallback_view,
            layout,
            pipeline,
            color,
            color_storage,
            color_sampled,
            front,
            front_storage,
            front_sampled,
            size,
            jitter: 0.0,
        })
    }

    #[allow(clippy::type_complexity)]
    fn create_targets(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        divisor: u32,
    ) -> (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::TextureView,
        (u32, u32),
    ) {
        let d = divisor.max(1);
        let w = width.div_ceil(d).max(1);
        let h = height.div_ceil(d).max(1);
        let color = util::color_target(
            device,
            "volume colour",
            w,
            h,
            VOLUME_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING,
        );
        let front = util::color_target(
            device,
            "volume front distance",
            w,
            h,
            FRONT_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING,
        );
        // Two views of each target: one bound as a write-only storage image by
        // the raymarch, one bound as a sampled texture by the compositor. WGSL
        // storage textures cannot be read, so a single view cannot do both jobs.
        let cs = color.create_view(&Default::default());
        let cr = color.create_view(&Default::default());
        let fs = front.create_view(&Default::default());
        let fr = front.create_view(&Default::default());
        (color, cs, cr, front, fs, fr, (w, h))
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (c, cs, cr, f, fs, fr, size) =
            Self::create_targets(device, width, height, self.settings.resolution_divisor);
        self.color = c;
        self.color_storage = cs;
        self.color_sampled = cr;
        self.front = f;
        self.front_storage = fs;
        self.front_sampled = fr;
        self.size = size;
    }

    pub fn settings(&self) -> &VolumeSettings {
        &self.settings
    }

    /// Apply new settings. Returns true when the targets had to be reallocated,
    /// which the caller must treat as a history reset for TAA.
    pub fn set_settings(
        &mut self,
        device: &wgpu::Device,
        settings: VolumeSettings,
        width: u32,
        height: u32,
    ) -> bool {
        let resized = settings.resolution_divisor != self.settings.resolution_divisor;
        self.settings = settings;
        if resized {
            self.resize(device, width, height);
        }
        resized
    }

    /// Re-upload the LUT. Cheap (2 KiB), so the UI can call it on every drag of
    /// the isolevel and the picture keeps up.
    pub fn upload_transfer_function(&self, queue: &wgpu::Queue, tf: &TransferFunction) {
        let entries = tf.bake_lut();
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

    /// Premultiplied half-resolution colour, for the compositing pass.
    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.color_sampled
    }
    /// Distance along each ray at which accumulated alpha first crossed the
    /// front threshold, or -1. TAA reprojects the volume with this.
    pub fn front_view(&self) -> &wgpu::TextureView {
        &self.front_sampled
    }
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
    pub fn color_texture(&self) -> &wgpu::Texture {
        &self.color
    }
    pub fn front_texture(&self) -> &wgpu::Texture {
        &self.front
    }

    /// Advance the blue-noise offset by one frame.
    pub fn advance_jitter(&mut self) {
        self.jitter = noise::golden_ratio_advance(self.jitter);
    }
    pub fn reset_jitter(&mut self) {
        self.jitter = 0.0;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut Profiler,
        camera_group: &wgpu::BindGroup,
        fields: &crate::fields::DerivedFields,
        bricks: &crate::accel::BrickGrid,
        opaque_depth: &wgpu::TextureView,
        sdf: Option<&SdfVolume<'_>>,
        tf: &TransferFunction,
    ) {
        let tfu = tf.uniform();
        let (sdf_min, sdf_inv, present) = match sdf {
            Some(s) => (
                s.min_mm.extend(0.0).to_array(),
                (Vec3::ONE / s.size_mm.max(Vec3::splat(1e-6)))
                    .extend(s.surface_offset_mm)
                    .to_array(),
                1u32,
            ),
            None => ([0.0; 4], [0.0, 0.0, 0.0, 0.0], 0u32),
        };
        let u = VolumeUniform {
            step_scale: self.settings.step_scale.clamp(0.05, 8.0),
            // The reference step is one derived voxel; see the module docs.
            h_ref_mm: fields.voxel_mm(),
            density: self.settings.density,
            max_steps: self.settings.max_steps.clamp(1, 65536),
            tf_lo: tfu.lo,
            tf_inv_span: tfu.inv_span,
            tf_log: tfu.log_scale,
            tf_show_clamp: tfu.show_clamp,
            under_color: tfu.under_color,
            over_color: tfu.over_color,
            light_dir: self
                .settings
                .light_dir
                .normalize_or(Vec3::Y)
                .extend(self.settings.shading)
                .to_array(),
            jitter_offset: self.jitter,
            skip_enabled: u32::from(self.settings.skip_empty_space),
            sdf_present: present,
            front_alpha: self.settings.front_alpha.clamp(0.001, 0.999),
            sdf_min_mm: sdf_min,
            sdf_inv_size: sdf_inv,
            ambient: self.settings.ambient.extend(self.settings.specular).to_array(),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));

        let sdf_view = sdf.map(|s| s.view).unwrap_or(&self.sdf_fallback_view);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.blue_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(opaque_depth),
                },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(sdf_view) },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.color_storage),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.front_storage),
                },
            ],
        });

        let ts = profiler.scope("volume");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("volume raymarch"),
            timestamp_writes: ts,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera_group, &[]);
        pass.set_bind_group(1, fields.read_bind_group(), &[]);
        pass.set_bind_group(2, bricks.read_bind_group(), &[]);
        pass.set_bind_group(3, &group, &[]);
        pass.dispatch_workgroups(
            util::dispatch_count(self.size.0, Self::WG.0),
            util::dispatch_count(self.size.1, Self::WG.1),
            1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_uniform_is_16_byte_aligned() {
        // WGSL rounds a uniform struct's size up to its alignment; a mismatch
        // here shows up as the shader reading the wrong field, not as an error.
        assert_eq!(std::mem::size_of::<VolumeUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<VolumeUniform>(), 144);
    }

    #[test]
    fn default_settings_are_sane() {
        let s = VolumeSettings::default();
        assert!(s.step_scale > 0.0 && s.step_scale <= 2.0);
        assert!(s.skip_empty_space, "skipping must be on by default");
        assert!((s.light_dir.length() - 1.0).abs() < 1e-5, "light must be normalised");
        assert!(s.resolution_divisor >= 1);
        // The front threshold has to be low enough that a wispy vortex still
        // gets a reprojection depth, and high enough not to latch onto noise.
        assert!(s.front_alpha > 0.02 && s.front_alpha < 0.5);
    }

    #[test]
    fn volume_and_front_formats_are_storage_capable() {
        let base = wgpu::Features::empty();
        for f in [VOLUME_FORMAT, FRONT_FORMAT] {
            assert!(
                f.guaranteed_format_features(base)
                    .allowed_usages
                    .contains(wgpu::TextureUsages::STORAGE_BINDING),
                "{f:?} cannot be a storage target in the base feature set"
            );
        }
    }
}

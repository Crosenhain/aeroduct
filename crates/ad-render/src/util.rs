//! Small wgpu conveniences shared by every pass in the crate.
//!
//! Nothing clever here. The point is that bind-group layouts and their bind
//! groups are written next to each other in the same order, because the most
//! common way to lose an hour in wgpu is a layout whose entry 3 is a sampler and
//! a bind group whose entry 3 is a texture.

use ad_gpu::{ShaderDefines, ShaderLoader};
use anyhow::Result;

/// Every WGSL file in `shaders/render/` is embedded at compile time and
/// registered as a virtual file with the `ad-gpu` preprocessor.
///
/// Embedding rather than reading from disk means the renderer works from any
/// working directory, works in `cargo test`, and works from a shipped binary
/// with no data directory. The files on disk remain the source of truth: they
/// are what `include_str!` reads, so editing one and rebuilding is the whole
/// workflow.
pub fn shader_loader() -> ShaderLoader {
    let mut l = ShaderLoader::new("shaders/render");
    macro_rules! embed {
        ($($name:literal),* $(,)?) => {
            $( l.add_virtual($name, include_str!(concat!("../../../shaders/render/", $name))); )*
        };
    }
    embed!(
        "common.wgsl",
        "derive.wgsl",
        "brick.wgsl",
        "volume.wgsl",
        "mesh.wgsl",
        "ssao.wgsl",
        "composite.wgsl",
        "taa.wgsl",
        "bloom.wgsl",
        "tonemap.wgsl",
    );
    l
}

pub fn compute_pipeline(
    device: &wgpu::Device,
    loader: &ShaderLoader,
    entry_file: &str,
    entry_point: &str,
    defines: &ShaderDefines,
    layouts: &[Option<&wgpu::BindGroupLayout>],
    label: &str,
) -> Result<wgpu::ComputePipeline> {
    let module = loader.create_module(device, entry_file, defines)?;
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: layouts,
        immediate_size: 0,
    });
    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        }),
    )
}

/// A bind-group-layout entry for a uniform buffer.
pub fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn storage_buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn texture_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// A *sampled* view of a texture that a compute pass also writes to through a
/// separate storage view. See [`storage_texture_entry`] for why there are two.
pub fn sampled_float_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    texture_entry(
        binding,
        visibility,
        wgpu::TextureSampleType::Float { filterable: true },
        view_dimension,
    )
}

/// A write-only storage texture binding.
///
/// WGSL storage textures are **write-only** in the portable spec: there is no
/// `textureLoad` on a `texture_storage_*<_, write>`. Any pass that needs to read
/// back what it wrote — or that hands its output to a later pass that wants
/// filtering — must create a second, *sampled* view of the same texture and
/// bind that separately. Every 3D texture in this crate is created with
/// `STORAGE_BINDING | TEXTURE_BINDING` for exactly that reason.
pub fn storage_texture_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    format: wgpu::TextureFormat,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension,
        },
        count: None,
    }
}

pub fn sampler_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

/// Number of workgroups needed to cover `n` items with `size`-wide groups.
#[inline]
pub fn dispatch_count(n: u32, size: u32) -> u32 {
    n.div_ceil(size.max(1))
}

/// Create a uniform buffer sized for `T`.
pub fn uniform_buffer<T: bytemuck::Pod>(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>().max(16) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A linear, clamp-to-edge sampler. The default for every 3D field read: clamp
/// rather than repeat, because a volume that tiles at the domain boundary looks
/// like a solver instability and wastes an afternoon.
pub fn linear_clamp_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}

pub fn nearest_clamp_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}

/// Create a 2D render target.
pub fn color_target(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    extra: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | extra,
        view_formats: &[],
    })
}

/// Copy an `Rgba8*` texture back to the CPU, submitting `encoder` as it goes.
///
/// Handles the 256-byte row alignment that `copy_texture_to_buffer` demands and
/// that catches everyone once: the buffer rows are padded, the returned `Vec` is
/// not. Blocking, so this is for screenshots and tests, never the render loop.
pub fn readback_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    mut encoder: wgpu::CommandEncoder,
) -> Vec<u8> {
    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let _ = rx.recv();

    let mut out = Vec::with_capacity((unpadded * height) as usize);
    if let Ok(view) = slice.get_mapped_range() {
        for row in 0..height as usize {
            let start = row * padded as usize;
            out.extend_from_slice(&view[start..start + unpadded as usize]);
        }
    }
    buffer.unmap();
    out
}

/// Copy any texture back to the CPU as raw bytes, submitting `encoder`.
///
/// Rows come back unpadded and layers are concatenated, so the result indexes as
/// `((z * height + y) * width + x) * bytes_per_texel`. Blocking; for tests and
/// debugging only.
pub fn readback_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: wgpu::Extent3d,
    bytes_per_texel: u32,
    mut encoder: wgpu::CommandEncoder,
) -> Vec<u8> {
    let unpadded = size.width * bytes_per_texel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let rows = size.height * size.depth_or_array_layers;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("texture readback"),
        size: (padded * rows) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let _ = rx.recv();

    let mut out = Vec::with_capacity((unpadded * rows) as usize);
    if let Ok(view) = slice.get_mapped_range() {
        for row in 0..rows as usize {
            let start = row * padded as usize;
            out.extend_from_slice(&view[start..start + unpadded as usize]);
        }
    }
    buffer.unmap();
    out
}

/// Box-average an RGBA8 image down by an integer factor.
///
/// The second half of the supersampled screenshot path: render at `factor`
/// times the resolution, accumulate jittered samples, read back, then average
/// here. A box filter and not a bilinear resample, because a bilinear resample
/// by 4 only ever touches 2x2 of every 4x4 block and throws three quarters of
/// the samples away — which would make the expensive render pointless.
pub fn box_downsample_rgba8(pixels: &[u8], width: u32, height: u32, factor: u32) -> Vec<u8> {
    let f = factor.max(1);
    if f == 1 {
        return pixels.to_vec();
    }
    let (ow, oh) = (width / f, height / f);
    let mut out = Vec::with_capacity((ow * oh * 4) as usize);
    let n = (f * f) as u32;
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = [0u32; 4];
            for sy in 0..f {
                for sx in 0..f {
                    let i = (((oy * f + sy) * width + ox * f + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += pixels[i + c] as u32;
                    }
                }
            }
            for c in acc {
                // Round to nearest rather than truncating: truncation biases
                // the whole image down by half an LSB, which is visible as a
                // slightly darker screenshot than the thing on screen.
                out.push(((c + n / 2) / n) as u8);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_shader_preprocesses_without_a_gpu() {
        // Catches `#include` typos, unterminated `#if`s and missing files at
        // `cargo test` time rather than at pipeline creation on a user's
        // machine. It does not type-check WGSL; the GPU tests do that.
        let loader = shader_loader();
        let base = ShaderDefines::new()
            .value("VOLUME_WG_X", 8u32)
            .value("VOLUME_WG_Y", 8u32)
            .value("DERIVE_WG_X", 4u32)
            .value("DERIVE_WG_Y", 4u32)
            .value("DERIVE_WG_Z", 4u32)
            .value("BRICK_SIZE", 8u32)
            .value("REDUCE_WG", 64u32)
            .value("DILATE_WG_X", 4u32)
            .value("DILATE_WG_Y", 4u32)
            .value("DILATE_WG_Z", 4u32)
            .flag("SRGB_TARGET");

        // brick.wgsl compiles once per pass; each variant must expand cleanly
        // and must contain exactly the one entry point it is meant to.
        for (flag, entry) in [
            ("BRICK_PASS_MINMAX", "fn minmax_main"),
            ("BRICK_PASS_SEED", "fn seed_main"),
            ("BRICK_PASS_DILATE", "fn dilate_main"),
        ] {
            let mut d = base.clone();
            if flag != "BRICK_PASS_DILATE" {
                d = d.flag(flag);
            }
            let src = loader.load("brick.wgsl", &d).unwrap();
            assert!(src.contains(entry), "brick.wgsl/{flag} is missing {entry}");
            let others = ["fn minmax_main", "fn seed_main", "fn dilate_main"]
                .iter()
                .filter(|e| **e != entry && src.contains(*e))
                .count();
            assert_eq!(
                others, 0,
                "brick.wgsl/{flag} leaked another pass's entry point"
            );
        }

        for f in [
            "derive.wgsl",
            "volume.wgsl",
            "mesh.wgsl",
            "ssao.wgsl",
            "composite.wgsl",
            "taa.wgsl",
            "bloom.wgsl",
            "tonemap.wgsl",
        ] {
            let src = loader.load(f, &base).unwrap_or_else(|e| panic!("{f}: {e}"));
            assert!(src.len() > 100, "{f} preprocessed to nothing");
            assert!(!src.contains("#include"), "{f} has an unexpanded include");
            // Every `#NAME` must have been substituted; a leftover token is a
            // define the Rust side forgot to set, and it fails at pipeline
            // creation with an unhelpful parse error.
            for line in src.lines() {
                assert!(
                    !line.trim_start().starts_with('#'),
                    "{f} has an unhandled directive: {line}"
                );
            }
        }
    }

    #[test]
    fn box_downsample_averages_each_block() {
        // 4x4 image, two 2x2 blocks per row, with known means.
        let mut px = Vec::new();
        for y in 0..4u32 {
            for x in 0..4u32 {
                let v = (x + y * 4) as u8 * 16;
                px.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let out = box_downsample_rgba8(&px, 4, 4, 2);
        assert_eq!(out.len(), 2 * 2 * 4);
        // Top-left block covers values 0, 16, 64, 80 -> mean 40.
        assert_eq!(out[0], 40);
        // Alpha survives untouched.
        assert!(out.chunks_exact(4).all(|p| p[3] == 255));
    }

    #[test]
    fn box_downsample_is_a_no_op_at_factor_one() {
        let px: Vec<u8> = (0..64).collect();
        assert_eq!(box_downsample_rgba8(&px, 4, 4, 1), px);
    }

    #[test]
    fn box_downsample_rounds_rather_than_truncating() {
        // Four values averaging 10.5. Truncation would give 10 and darken the
        // whole screenshot by half an LSB.
        let px = vec![
            10, 10, 10, 255, 10, 10, 10, 255, //
            11, 11, 11, 255, 11, 11, 11, 255,
        ];
        assert_eq!(box_downsample_rgba8(&px, 2, 2, 2)[0], 11);
    }

    #[test]
    fn dispatch_count_rounds_up() {
        assert_eq!(dispatch_count(0, 8), 0);
        assert_eq!(dispatch_count(1, 8), 1);
        assert_eq!(dispatch_count(8, 8), 1);
        assert_eq!(dispatch_count(9, 8), 2);
        assert_eq!(dispatch_count(347, 8), 44);
    }
}

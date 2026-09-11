//! What the metrics passes read: the solver's macroscopic fields, plus the flag
//! byte that says which cells are real.
//!
//! # Three bindings, one meaning
//!
//! Every metrics kernel binds the same group 0:
//!
//! | binding | resource | contents |
//! |---|---|---|
//! | 0 | `texture_3d<f32>` | velocity, **lattice units**, `Rgba16Float` |
//! | 1 | `texture_3d<f32>` | density, **lattice units**, `R32Float` |
//! | 2 | `array<u32>` storage | [`ad_gpu::flags`] bytes, four cells per word |
//!
//! The two textures are the ones `ad_solver::Solver::velocity_view` and
//! `density_view` hand out. Per CONTRACT.md rule 5 they are bound as *sampled*
//! views, never as the storage views the macroscopic pass writes through. The
//! flag words are what `ad_geom::Voxelizer::flags_buffer` produces, which is
//! already the right layout over the same interior grid.
//!
//! # Why `textureLoad` and not a sampler
//!
//! The kernels do their own trilinear interpolation from eight `textureLoad`s
//! rather than letting a linear sampler do it. Three reasons, in order of
//! importance:
//!
//! 1. `R32Float` is only filterable with the optional `FLOAT32_FILTERABLE`
//!    feature. Doing it by hand removes that dependency entirely.
//! 2. Hardware filter weights are specified to only 8 bits of subpixel
//!    precision. Over 262,144 samples that is a systematic, not a random,
//!    error in an integral.
//! 3. We need to know exactly what happens at a solid corner. Doing the
//!    interpolation ourselves makes the answer inspectable instead of
//!    depending on what the texture happens to contain.
//!
//! # What a solid cell contains
//!
//! `shaders/lbm/macroscopic.wgsl` writes `u = 0`, `rho = 1` for every solid
//! cell. That is exactly the no-slip Dirichlet value, so a solid corner
//! contributing to a trilinear stencil is physically the right thing and needs
//! no special case. The flag byte is used for something different: deciding
//! whether the *sample point itself* lies in fluid, which is what
//! [`crate::plane`] reports as the covered-area fraction.

use ad_gpu::types::Grid;
use anyhow::Result;
use glam::Vec4;

/// Velocity texture format the solver produces.
pub const VELOCITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Density texture format the solver produces.
pub const DENSITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Velocity format for verification fixtures, where fp16 storage error would
/// swamp the tolerance an analytic test wants to assert.
pub const VELOCITY_FORMAT_EXACT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Borrowed references to one snapshot of the solver state.
///
/// Deliberately not a reference to `Solver`: the metrics kernels do not need the
/// distribution functions, the pipelines or the step counter, and taking only
/// what they read is what lets every analytic test in this crate run against a
/// synthetic field with no solver at all.
#[derive(Clone, Copy)]
pub struct FieldRefs<'a> {
    /// The **interior** grid, matching the texture dimensions exactly.
    pub grid: Grid,
    pub velocity: &'a wgpu::TextureView,
    pub density: &'a wgpu::TextureView,
    /// One [`ad_gpu::flags`] byte per interior cell, four per `u32`, X fastest.
    pub flags: &'a wgpu::Buffer,
}

/// Bind group layout for group 0 of every metrics kernel.
///
/// `filterable: false` on both textures is the weaker requirement and is
/// satisfied by `Rgba16Float`, `R32Float` and `Rgba32Float` alike, which is what
/// lets the verification fixtures swap in a 32-bit velocity texture without a
/// second layout.
pub fn field_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let tex = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("metrics field"),
        entries: &[
            tex(0),
            tex(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub fn field_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    field: &FieldRefs<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("metrics field"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(field.velocity),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(field.density),
            },
            wgpu::BindGroupEntry { binding: 2, resource: field.flags.as_entire_binding() },
        ],
    })
}

/// Pack one [`ad_gpu::flags`] byte per cell into `u32` words, four cells per
/// word, X fastest — the layout both `ad_solver` and `ad_geom` already use.
pub fn pack_flags(bytes: &[u8]) -> Vec<u32> {
    let mut out = vec![0u32; bytes.len().div_ceil(4).max(1)];
    for (i, b) in bytes.iter().enumerate() {
        out[i >> 2] |= (*b as u32) << ((i & 3) * 8);
    }
    out
}

/// Inverse of [`pack_flags`], for tests and for anything that wants to inspect
/// a flag buffer it read back.
pub fn unpack_flags(words: &[u32], count: usize) -> Vec<u8> {
    (0..count).map(|i| ((words[i >> 2] >> ((i & 3) * 8)) & 0xff) as u8).collect()
}

/// IEEE binary32 to binary16, round-to-nearest-even, with overflow saturating to
/// infinity and subnormals handled.
///
/// Needed because `Rgba16Float` is what the solver writes and
/// `Queue::write_texture` takes raw bytes. Small enough to own rather than pull
/// in a dependency for, and exercised by a round-trip test.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let mant = x & 0x007f_ffff;

    if ((x >> 23) & 0xff) == 0xff {
        // Inf or NaN. Keep NaN a NaN rather than letting it become infinity.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        // Subnormal: shift the implicit 1 back in and round.
        let m = mant | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = m + half - 1 + ((m >> shift) & 1);
        return sign | (rounded >> shift) as u16;
    }
    // Round-to-nearest-even on the 13 discarded mantissa bits.
    let rounded = mant + 0x0000_0fff + ((mant >> 13) & 1);
    if rounded & 0x0080_0000 != 0 {
        exp += 1;
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
    }
    sign | ((exp as u16) << 10) | ((rounded >> 13) & 0x3ff) as u16
}

/// binary16 back to binary32. Only used by tests, but it belongs next to its
/// inverse.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Normalise the subnormal.
        let shift = mant.leading_zeros() - 21;
        let e = 127 - 15 - shift;
        let m = (mant << (shift + 1)) & 0x03ff;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// A velocity texture, a density texture and a flag buffer sized to one grid.
///
/// The application gets these from the solver; this owns its own set so that
/// every analytic test in this crate — uniform flow through a tilted plane,
/// Poiseuille, solid-body rotation — can be run against a field whose answer is
/// known in closed form, with no solver and no geometry in the loop.
pub struct FieldTextures {
    pub grid: Grid,
    velocity: wgpu::Texture,
    density: wgpu::Texture,
    flags: wgpu::Buffer,
    velocity_format: wgpu::TextureFormat,
}

impl FieldTextures {
    /// Solver-matching formats: fp16 velocity, fp32 density.
    pub fn new(device: &wgpu::Device, grid: Grid) -> Self {
        Self::with_format(device, grid, VELOCITY_FORMAT)
    }

    /// fp32 velocity, so an analytic test can assert to 1e-5 instead of 1e-3.
    pub fn new_exact(device: &wgpu::Device, grid: Grid) -> Self {
        Self::with_format(device, grid, VELOCITY_FORMAT_EXACT)
    }

    fn with_format(device: &wgpu::Device, grid: Grid, velocity_format: wgpu::TextureFormat) -> Self {
        let size = wgpu::Extent3d {
            width: grid.dims.x,
            height: grid.dims.y,
            depth_or_array_layers: grid.dims.z,
        };
        let make = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let words = (grid.cell_count() as usize).div_ceil(4).max(1);
        Self {
            grid,
            velocity: make("metrics fixture velocity", velocity_format),
            density: make("metrics fixture density", DENSITY_FORMAT),
            flags: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("metrics fixture flags"),
                size: (words * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            velocity_format,
        }
    }

    /// Upload one `(ux, uy, uz, rho)` value per cell, X fastest, plus the flag
    /// byte per cell. All in **lattice units**, matching what the solver writes.
    pub fn upload(&self, queue: &wgpu::Queue, values: &[Vec4], flags: &[u8]) -> Result<()> {
        let n = self.grid.cell_count() as usize;
        anyhow::ensure!(values.len() == n, "expected {n} field values, got {}", values.len());
        anyhow::ensure!(flags.len() == n, "expected {n} flag bytes, got {}", flags.len());

        let (w, h, d) = (self.grid.dims.x, self.grid.dims.y, self.grid.dims.z);
        let extent = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d };

        let vel_bytes: Vec<u8> = match self.velocity_format {
            VELOCITY_FORMAT_EXACT => values
                .iter()
                .flat_map(|v| {
                    [v.x, v.y, v.z, 0.0f32].into_iter().flat_map(f32::to_le_bytes)
                })
                .collect(),
            _ => values
                .iter()
                .flat_map(|v| {
                    [v.x, v.y, v.z, 0.0f32]
                        .into_iter()
                        .flat_map(|c| f32_to_f16_bits(c).to_le_bytes())
                })
                .collect(),
        };
        let bpp = if self.velocity_format == VELOCITY_FORMAT_EXACT { 16 } else { 8 };
        queue.write_texture(
            self.velocity.as_image_copy(),
            &vel_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * bpp),
                rows_per_image: Some(h),
            },
            extent,
        );

        let den: Vec<f32> = values.iter().map(|v| v.w).collect();
        queue.write_texture(
            self.density.as_image_copy(),
            bytemuck::cast_slice(&den),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            extent,
        );

        queue.write_buffer(&self.flags, 0, bytemuck::cast_slice(&pack_flags(flags)));
        Ok(())
    }

    /// Fill from a closure of cell centre position (mm) to `(u, rho)` and a flag.
    pub fn fill(
        &self,
        queue: &wgpu::Queue,
        f: impl Fn(glam::UVec3, glam::Vec3) -> (glam::Vec3, f32, u8),
    ) -> Result<()> {
        let d = self.grid.dims;
        let n = self.grid.cell_count() as usize;
        let mut values = Vec::with_capacity(n);
        let mut flags = Vec::with_capacity(n);
        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    let c = glam::UVec3::new(x, y, z);
                    let (u, rho, fl) = f(c, self.grid.cell_center_mm(c));
                    values.push(Vec4::new(u.x, u.y, u.z, rho));
                    flags.push(fl);
                }
            }
        }
        self.upload(queue, &values, &flags)
    }

    pub fn velocity_view(&self) -> wgpu::TextureView {
        self.velocity.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn density_view(&self) -> wgpu::TextureView {
        self.density.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn flags_buffer(&self) -> &wgpu::Buffer {
        &self.flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::types::flags;

    #[test]
    fn flag_packing_round_trips_and_matches_the_solver_layout() {
        let bytes: Vec<u8> = (0..37).map(|i| (i * 7 % 64) as u8).collect();
        let words = pack_flags(&bytes);
        assert_eq!(words.len(), 10);
        assert_eq!(unpack_flags(&words, bytes.len()), bytes);
        // The exact word the solver's own test asserts.
        assert_eq!(pack_flags(&[0x01, 0x02, 0x04, 0x08, 0x10])[0], 0x08040201);
    }

    #[test]
    fn f16_round_trips_the_values_a_lattice_velocity_actually_takes() {
        // Lattice velocities live in +/-0.2 and densities near 1.
        for v in [0.0f32, 1.0, -1.0, 0.05, -0.05, 0.1, 1e-4, -1e-4, 0.123_456, 65504.0] {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            let err = if v == 0.0 { back.abs() } else { (back - v).abs() / v.abs() };
            assert!(err < 1e-3, "{v} round-tripped to {back}");
        }
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::INFINITY)).is_infinite());
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        // Overflow saturates rather than wrapping to zero.
        assert!(f16_bits_to_f32(f32_to_f16_bits(1e30)).is_infinite());
        // Subnormal range.
        let tiny = f16_bits_to_f32(f32_to_f16_bits(1e-6));
        assert!(tiny > 0.0 && tiny < 2e-6, "1e-6 became {tiny}");
    }

    #[test]
    fn a_fixture_uploads_and_binds_against_the_shared_layout() {
        let Some(gpu) = crate::test_gpu() else { return };
        let grid = Grid::covering(
            ad_gpu::Bbox { min: glam::Vec3::ZERO, max: glam::Vec3::splat(4.0) },
            1.0,
        );
        for exact in [false, true] {
            let f = if exact {
                FieldTextures::new_exact(&gpu.device, grid)
            } else {
                FieldTextures::new(&gpu.device, grid)
            };
            f.fill(&gpu.queue, |_, p| (glam::Vec3::new(0.05, 0.0, 0.0), 1.0 + p.x * 1e-4, flags::FLUID))
                .unwrap();
            let layout = field_layout(&gpu.device);
            let (vv, dv) = (f.velocity_view(), f.density_view());
            // Validation failures surface at bind group creation.
            let _ = field_bind_group(
                &gpu.device,
                &layout,
                &FieldRefs { grid, velocity: &vv, density: &dv, flags: f.flags_buffer() },
            );
        }
    }
}

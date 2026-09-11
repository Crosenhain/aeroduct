//! Empty-space skipping: an 8^3 brick min/max grid plus a Chebyshev distance
//! transform.
//!
//! A duct volume is mostly nothing. The domain is a 260 x 180 x 180 mm box
//! around a part that is 145 x 72 x 69 mm and hollow, and the transfer function
//! is usually a narrow soft-isosurface bump, so the fraction of voxels that
//! contribute any opacity at all is routinely under 2%. Marching through the
//! other 98% at `dx` steps is the single largest cost in the frame.
//!
//! # The structure, and why this one
//!
//! Three stages, each rebuilt only when its input changes:
//!
//! 1. **Brick min/max.** Reduce the displayed scalar over each 8^3 block into a
//!    `(min, max)` pair. Rebuilt when the field data or the displayed channel
//!    changes. The reduction includes a one-voxel apron, because the raymarcher
//!    samples with trilinear filtering and can therefore see slightly outside
//!    the block it is standing in; without the apron you get a faint grid of
//!    missing shells that only appears at certain camera angles.
//! 2. **Binarisation.** A brick is *active* if its `[min, max]` interval
//!    intersects the transfer function's non-zero support. Rebuilt when the
//!    transfer function changes — which is cheap, and is why re-tuning the
//!    isolevel stays interactive.
//! 3. **Chebyshev distance transform** over the binary grid, so the raymarcher
//!    can read one texel and skip `d` bricks in one go.
//!
//! The last stage is the one that earns its keep. Reported measurements put a
//! min/max grid plus a distance transform at roughly **2x faster than an octree
//! or SparseLeap**, for a fraction of the implementation. The reason is that the
//! skip is a single `textureLoad` and a ray-box exit, with no traversal stack, no
//! pointer chasing and no divergence between neighbouring rays.
//!
//! Chebyshev rather than Euclidean because Chebyshev distance in the brick grid
//! is exactly graph distance under 26-connectivity, so the transform is a
//! trivially parallel dilation; and because the thing being skipped is an
//! axis-aligned *box* of bricks, which is what Chebyshev measures.
//!
//! # The skip is safe
//!
//! If `d(b) = k`, then every brick within Chebyshev distance `k - 1` of `b` is
//! inactive (otherwise `d(b)` would be smaller). So the raymarcher may advance
//! straight to the exit of the `(2k-1)^3` box of bricks centred on `b`. Nothing
//! inside it can contribute opacity, by construction — which, combined with the
//! opacity correction in [`crate::transfer`], means turning skipping on and off
//! must produce a **pixel-identical** image. If it does not, there is a bug, and
//! that unambiguity is the whole reason both mechanisms are non-negotiable.

use ad_gpu::{Profiler, ShaderDefines, ShaderLoader};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::UVec3;

use crate::fields::DerivedFields;
use crate::transfer::TransferFunction;
use crate::util;

/// Bricks are 8 voxels on a side. Small enough that a brick straddling the wall
/// of a 6 mm passage does not drag half the passage into "active"; large enough
/// that the brick grid is 1/512 the size of the field.
pub const BRICK_SIZE: u32 = 8;

/// Distances are capped here. A skip of 15 bricks is 120 voxels, which at the
/// interactive tier is 90 mm — a third of the domain — so a larger cap buys
/// nothing and costs a dilation pass per unit.
pub const MAX_SKIP: u32 = 15;

/// Dilation passes run. One more than [`MAX_SKIP`] so the ping-pong ends on the
/// texture the read bind group points at, which keeps that bind group static.
const DILATE_PASSES: u32 = MAX_SKIP + 1;

/// Iterative Chebyshev distance transform over a binary brick grid.
///
/// `active[i]` is true where the brick contributes opacity; the result is the
/// Chebyshev distance to the nearest active brick, clamped at `cap`. Cells
/// outside the grid count as inactive.
///
/// This mirrors `brick.wgsl`'s `dilate_main` exactly, one iteration per pass. It
/// exists on the CPU so the recurrence can be checked against a brute-force
/// reference without a GPU — the shader is then correct by construction, since
/// the two implement the same three lines.
pub fn chebyshev_distance_transform(active: &[bool], dims: UVec3, cap: u32) -> Vec<u32> {
    let n = (dims.x * dims.y * dims.z) as usize;
    assert_eq!(active.len(), n, "active grid does not match dims");
    let idx = |x: u32, y: u32, z: u32| ((z * dims.y + y) * dims.x + x) as usize;

    let mut cur: Vec<u32> = active.iter().map(|a| if *a { 0 } else { cap }).collect();
    let mut next = cur.clone();

    // `cap` iterations suffice: each one can only lower a value by 1, and no
    // value starts above `cap`.
    for _ in 0..cap {
        let mut changed = false;
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let here = cur[idx(x, y, z)];
                    let mut best = here;
                    if here > 0 {
                        for dz in -1i32..=1 {
                            for dy in -1i32..=1 {
                                for dx in -1i32..=1 {
                                    let (nx, ny, nz) = (
                                        x as i32 + dx,
                                        y as i32 + dy,
                                        z as i32 + dz,
                                    );
                                    if nx < 0
                                        || ny < 0
                                        || nz < 0
                                        || nx >= dims.x as i32
                                        || ny >= dims.y as i32
                                        || nz >= dims.z as i32
                                    {
                                        continue;
                                    }
                                    let nd = cur[idx(nx as u32, ny as u32, nz as u32)];
                                    best = best.min(nd.saturating_add(1).min(cap));
                                }
                            }
                        }
                    }
                    if best != here {
                        changed = true;
                    }
                    next[idx(x, y, z)] = best;
                }
            }
        }
        std::mem::swap(&mut cur, &mut next);
        if !changed {
            break;
        }
    }
    cur
}

/// Brute-force reference: the definition, evaluated directly. `O(n^2)`, so only
/// for tests.
pub fn chebyshev_distance_brute_force(active: &[bool], dims: UVec3, cap: u32) -> Vec<u32> {
    let sites: Vec<(i32, i32, i32)> = (0..dims.z)
        .flat_map(|z| (0..dims.y).flat_map(move |y| (0..dims.x).map(move |x| (x, y, z))))
        .filter(|(x, y, z)| active[((z * dims.y + y) * dims.x + x) as usize])
        .map(|(x, y, z)| (x as i32, y as i32, z as i32))
        .collect();

    (0..dims.z)
        .flat_map(|z| (0..dims.y).flat_map(move |y| (0..dims.x).map(move |x| (x, y, z))))
        .map(|(x, y, z)| {
            let (px, py, pz) = (x as i32, y as i32, z as i32);
            sites
                .iter()
                .map(|(sx, sy, sz)| {
                    (px - sx).abs().max((py - sy).abs()).max((pz - sz).abs()) as u32
                })
                .min()
                .unwrap_or(cap)
                .min(cap)
        })
        .collect()
}

/// Number of bricks covering a field of `dims` voxels.
pub fn brick_dims(dims: UVec3) -> UVec3 {
    UVec3::new(
        dims.x.div_ceil(BRICK_SIZE).max(1),
        dims.y.div_ceil(BRICK_SIZE).max(1),
        dims.z.div_ceil(BRICK_SIZE).max(1),
    )
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BrickUniform {
    brick_dims: [u32; 3],
    brick_size: u32,
    field_dims: [u32; 3],
    max_skip: u32,
    /// Transfer-function support, clamped to finite values so a driver with
    /// aggressive fast-math cannot turn an infinity into a NaN and mark the
    /// whole grid inactive.
    support_lo: f32,
    support_hi: f32,
    /// Interior transparent band, or an empty interval (`lo > hi`) when there is
    /// none. See `transfer::Support`.
    gap_lo: f32,
    gap_hi: f32,
    channel: u32,
    _pad: [u32; 3],
}

/// Signature used to decide whether the binarisation is stale.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SupportKey([u32; 4]);

impl SupportKey {
    fn of(support: Option<crate::transfer::Support>) -> Self {
        match support {
            Some(s) => {
                let (ga, gb) = s.gap_or_empty();
                Self([s.range.0.to_bits(), s.range.1.to_bits(), ga.to_bits(), gb.to_bits()])
            }
            // An empty interval: nothing is active.
            None => Self([1.0f32.to_bits(), 0.0f32.to_bits(), 1.0f32.to_bits(), 0.0f32.to_bits()]),
        }
    }
}

/// The brick acceleration structure.
pub struct BrickGrid {
    dims: UVec3,
    field_dims: UVec3,

    /// `[min/max, distance A, distance B]`. Held so the textures outlive the
    /// bind groups that reference them; the views are not kept, since every one
    /// of them is baked into a bind group at construction and never rebound.
    textures: [wgpu::Texture; 3],

    uniform: wgpu::Buffer,

    minmax_pipeline: wgpu::ComputePipeline,
    minmax_group: wgpu::BindGroup,

    seed_pipeline: wgpu::ComputePipeline,
    seed_group: wgpu::BindGroup,

    dilate_pipeline: wgpu::ComputePipeline,
    /// `dilate_groups[i]` reads `dist[i]` and writes `dist[1 - i]`.
    dilate_groups: [wgpu::BindGroup; 2],

    read_layout: wgpu::BindGroupLayout,
    read_group: wgpu::BindGroup,

    cached_generation: Option<u64>,
    cached_channel: Option<u32>,
    cached_support: Option<SupportKey>,
}

impl BrickGrid {
    /// Workgroup shape of the min/max reduction: one workgroup per brick.
    const REDUCE_WG: u32 = 64;
    const DILATE_WG: UVec3 = UVec3::new(4, 4, 4);

    pub fn new(
        device: &wgpu::Device,
        loader: &ShaderLoader,
        fields: &DerivedFields,
    ) -> Result<Self> {
        let field_dims = fields.dims();
        let dims = brick_dims(field_dims);
        let size = wgpu::Extent3d {
            width: dims.x,
            height: dims.y,
            depth_or_array_layers: dims.z,
        };

        // Rg32Float for the min/max pair: guaranteed storage-capable in the base
        // WebGPU feature set, and only ever read with `textureLoad`, so its lack
        // of guaranteed filtering costs nothing.
        let minmax = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brick min/max"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rg32Float,
            // COPY_SRC costs nothing and buys inspectability: the brick grid is
            // the one structure whose bugs are invisible in the final image
            // (they look like missing data, not like wrong data), so being able
            // to read it back is worth a usage flag.
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mk_dist = |i: usize| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(if i == 0 { "brick distance A" } else { "brick distance B" }),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let dist = [mk_dist(0), mk_dist(1)];

        let view = |t: &wgpu::Texture, label: &str| {
            t.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            })
        };
        let minmax_storage = view(&minmax, "brick min/max (storage)");
        let minmax_sampled = view(&minmax, "brick min/max (sampled)");
        let dist_storage = [view(&dist[0], "brick dist A (storage)"), view(&dist[1], "brick dist B (storage)")];
        let dist_sampled = [view(&dist[0], "brick dist A (sampled)"), view(&dist[1], "brick dist B (sampled)")];

        let uniform = util::uniform_buffer::<BrickUniform>(device, "brick uniform");
        let cs = wgpu::ShaderStages::COMPUTE;

        // --- min/max: reads the derived fields (group 0), writes group 1.
        let minmax_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brick min/max"),
            entries: &[
                util::uniform_entry(0, cs),
                util::storage_texture_entry(
                    1,
                    cs,
                    wgpu::TextureFormat::Rg32Float,
                    wgpu::TextureViewDimension::D3,
                ),
            ],
        });
        let minmax_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brick min/max"),
            layout: &minmax_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&minmax_storage),
                },
            ],
        });

        // --- seed: min/max -> binary distance field.
        let seed_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brick seed"),
            entries: &[
                util::uniform_entry(0, cs),
                util::texture_entry(
                    1,
                    cs,
                    wgpu::TextureSampleType::Float { filterable: false },
                    wgpu::TextureViewDimension::D3,
                ),
                util::storage_texture_entry(
                    2,
                    cs,
                    wgpu::TextureFormat::R32Uint,
                    wgpu::TextureViewDimension::D3,
                ),
            ],
        });
        let seed_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brick seed"),
            layout: &seed_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&minmax_sampled),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dist_storage[0]),
                },
            ],
        });

        // --- dilate: one Chebyshev step, ping-ponged.
        let dilate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brick dilate"),
            entries: &[
                util::uniform_entry(0, cs),
                util::texture_entry(
                    1,
                    cs,
                    wgpu::TextureSampleType::Uint,
                    wgpu::TextureViewDimension::D3,
                ),
                util::storage_texture_entry(
                    2,
                    cs,
                    wgpu::TextureFormat::R32Uint,
                    wgpu::TextureViewDimension::D3,
                ),
            ],
        });
        let mk_dilate = |src: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("brick dilate"),
                layout: &dilate_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&dist_sampled[src]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&dist_storage[1 - src]),
                    },
                ],
            })
        };
        let dilate_groups = [mk_dilate(0), mk_dilate(1)];

        // --- read side, for the raymarcher and for Wave 2.
        let vis = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT;
        let read_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brick grid (read)"),
            entries: &[
                util::uniform_entry(0, vis),
                util::texture_entry(
                    1,
                    vis,
                    wgpu::TextureSampleType::Uint,
                    wgpu::TextureViewDimension::D3,
                ),
                util::texture_entry(
                    2,
                    vis,
                    wgpu::TextureSampleType::Float { filterable: false },
                    wgpu::TextureViewDimension::D3,
                ),
            ],
        });
        let read_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brick grid (read)"),
            layout: &read_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                // DILATE_PASSES is even, so the final result always lands back
                // in slot 0 and this bind group never has to be rebuilt.
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dist_sampled[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&minmax_sampled),
                },
            ],
        });

        let defines = ShaderDefines::new()
            .value("BRICK_SIZE", BRICK_SIZE)
            .value("REDUCE_WG", Self::REDUCE_WG)
            .value("DILATE_WG_X", Self::DILATE_WG.x)
            .value("DILATE_WG_Y", Self::DILATE_WG.y)
            .value("DILATE_WG_Z", Self::DILATE_WG.z);

        // Three passes, three modules. They want different resources at group 0,
        // and WGSL requires every `@group`/`@binding` pair to be unique within a
        // module, so the preprocessor selects one pass per compile.
        let minmax_pipeline = util::compute_pipeline(
            device,
            loader,
            "brick.wgsl",
            "minmax_main",
            &defines.clone().flag("BRICK_PASS_MINMAX"),
            &[Some(fields.read_bind_group_layout()), Some(&minmax_layout)],
            "brick min/max",
        )?;
        let seed_pipeline = util::compute_pipeline(
            device,
            loader,
            "brick.wgsl",
            "seed_main",
            &defines.clone().flag("BRICK_PASS_SEED"),
            &[Some(&seed_layout)],
            "brick seed",
        )?;
        let dilate_pipeline = util::compute_pipeline(
            device,
            loader,
            "brick.wgsl",
            "dilate_main",
            &defines,
            &[Some(&dilate_layout)],
            "brick dilate",
        )?;

        debug_assert_eq!(DILATE_PASSES % 2, 0, "the read bind group assumes an even pass count");

        let [dist_a, dist_b] = dist;
        Ok(Self {
            dims,
            field_dims,
            textures: [minmax, dist_a, dist_b],
            uniform,
            minmax_pipeline,
            minmax_group,
            seed_pipeline,
            seed_group,
            dilate_pipeline,
            dilate_groups,
            read_layout,
            read_group,
            cached_generation: None,
            cached_channel: None,
            cached_support: None,
        })
    }

    pub fn dims(&self) -> UVec3 {
        self.dims
    }

    /// Layout of the read bind group: `(uniform, distance, min/max)`.
    /// **Wave 2 extension point** — an isosurface pass can skip empty space with
    /// exactly the same traversal.
    pub fn read_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.read_layout
    }
    pub fn read_bind_group(&self) -> &wgpu::BindGroup {
        &self.read_group
    }

    /// `Rg32Float` per-brick `(min, max)` of the displayed scalar. A brick with
    /// no fluid voxels is flagged by `min > max`.
    pub fn minmax_texture(&self) -> &wgpu::Texture {
        &self.textures[0]
    }
    /// `R32Uint` Chebyshev distance, in bricks, to the nearest active brick.
    /// Slot 0 always holds the final result: the dilation pass count is even.
    pub fn distance_texture(&self) -> &wgpu::Texture {
        &self.textures[1]
    }

    /// Force a full rebuild on the next [`BrickGrid::update`].
    pub fn invalidate(&mut self) {
        self.cached_generation = None;
        self.cached_support = None;
    }

    /// Rebuild whatever is stale. Cheap and safe to call every frame.
    ///
    /// Returns true if any work was recorded, which is useful for the
    /// progressive accumulator: a rebuild means the image changed.
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut Profiler,
        fields: &DerivedFields,
        tf: &TransferFunction,
    ) -> bool {
        let channel = fields.field().channel();
        let support = tf.support();
        let key = SupportKey::of(support);

        let minmax_stale =
            self.cached_generation != Some(fields.generation()) || self.cached_channel != Some(channel);
        let binary_stale = minmax_stale || self.cached_support != Some(key);
        if !binary_stale {
            return false;
        }

        // An absent support means "entirely transparent"; an empty interval
        // marks every brick inactive, which is exactly right.
        let (lo, hi) = support.map(|s| s.range).unwrap_or((1.0, 0.0));
        let (gap_lo, gap_hi) = support.map(|s| s.gap_or_empty()).unwrap_or((1.0, 0.0));
        let u = BrickUniform {
            brick_dims: self.dims.to_array(),
            brick_size: BRICK_SIZE,
            field_dims: self.field_dims.to_array(),
            max_skip: MAX_SKIP,
            support_lo: lo.clamp(-f32::MAX, f32::MAX),
            support_hi: hi.clamp(-f32::MAX, f32::MAX),
            gap_lo: gap_lo.clamp(-f32::MAX, f32::MAX),
            gap_hi: gap_hi.clamp(-f32::MAX, f32::MAX),
            channel,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));

        if minmax_stale {
            let ts = profiler.scope("brick min/max");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brick min/max"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.minmax_pipeline);
            pass.set_bind_group(0, fields.read_bind_group(), &[]);
            pass.set_bind_group(1, &self.minmax_group, &[]);
            // One workgroup per brick.
            pass.dispatch_workgroups(self.dims.x, self.dims.y, self.dims.z);
        }

        {
            let ts = profiler.scope("brick seed");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brick seed"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.seed_pipeline);
            pass.set_bind_group(0, &self.seed_group, &[]);
            pass.dispatch_workgroups(
                util::dispatch_count(self.dims.x, Self::DILATE_WG.x),
                util::dispatch_count(self.dims.y, Self::DILATE_WG.y),
                util::dispatch_count(self.dims.z, Self::DILATE_WG.z),
            );
        }

        {
            let ts = profiler.scope("brick chebyshev");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brick chebyshev"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.dilate_pipeline);
            for i in 0..DILATE_PASSES {
                pass.set_bind_group(0, &self.dilate_groups[(i % 2) as usize], &[]);
                pass.dispatch_workgroups(
                    util::dispatch_count(self.dims.x, Self::DILATE_WG.x),
                    util::dispatch_count(self.dims.y, Self::DILATE_WG.y),
                    util::dispatch_count(self.dims.z, Self::DILATE_WG.z),
                );
            }
        }

        self.cached_generation = Some(fields.generation());
        self.cached_channel = Some(channel);
        self.cached_support = Some(key);
        true
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic, dependency-free PRNG so the tests are reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }
    }

    #[test]
    fn distance_transform_matches_brute_force_on_random_grids() {
        // The reference implementation is the definition; the iterative one is
        // what the shader does. If they ever disagree, the raymarcher will skip
        // over something it should have drawn.
        let mut rng = Lcg(0x5EED);
        for (dims, density) in [
            (UVec3::new(13, 11, 9), 12u32),
            (UVec3::new(8, 8, 8), 3),
            (UVec3::new(20, 4, 7), 40),
            (UVec3::new(5, 5, 5), 90),
        ] {
            let n = (dims.x * dims.y * dims.z) as usize;
            let active: Vec<bool> = (0..n).map(|_| rng.next() % 100 < density).collect();
            let got = chebyshev_distance_transform(&active, dims, MAX_SKIP);
            let want = chebyshev_distance_brute_force(&active, dims, MAX_SKIP);
            assert_eq!(got, want, "mismatch for {dims:?} at density {density}%");
        }
    }

    #[test]
    fn a_single_active_brick_produces_concentric_cubic_shells() {
        // Chebyshev distance draws cubes, not spheres. This is the property the
        // raymarcher's box-exit skip relies on.
        let dims = UVec3::new(11, 11, 11);
        let n = (dims.x * dims.y * dims.z) as usize;
        let mut active = vec![false; n];
        let idx = |x: u32, y: u32, z: u32| ((z * dims.y + y) * dims.x + x) as usize;
        active[idx(5, 5, 5)] = true;

        let d = chebyshev_distance_transform(&active, dims, MAX_SKIP);
        assert_eq!(d[idx(5, 5, 5)], 0);
        for (p, want) in [
            ((6u32, 5u32, 5u32), 1u32),
            ((6, 6, 6), 1), // the diagonal is also distance 1
            ((7, 7, 7), 2),
            ((5, 5, 8), 3),
            ((0, 0, 0), 5),
            ((10, 10, 10), 5),
        ] {
            assert_eq!(d[idx(p.0, p.1, p.2)], want, "at {p:?}");
        }
    }

    #[test]
    fn an_entirely_inactive_grid_saturates_at_the_cap() {
        let dims = UVec3::new(6, 6, 6);
        let active = vec![false; 216];
        let d = chebyshev_distance_transform(&active, dims, MAX_SKIP);
        assert!(d.iter().all(|v| *v == MAX_SKIP), "empty grid must be all-cap");
    }

    #[test]
    fn an_entirely_active_grid_is_all_zero() {
        let dims = UVec3::new(6, 6, 6);
        let active = vec![true; 216];
        let d = chebyshev_distance_transform(&active, dims, MAX_SKIP);
        assert!(d.iter().all(|v| *v == 0));
    }

    #[test]
    fn distances_are_capped_but_never_wrong_below_the_cap() {
        // A long thin grid where the true distance exceeds the cap: everything
        // far away must read exactly `cap`, and everything near must be exact.
        let dims = UVec3::new(64, 1, 1);
        let mut active = vec![false; 64];
        active[0] = true;
        let d = chebyshev_distance_transform(&active, dims, MAX_SKIP);
        for x in 0..64usize {
            assert_eq!(d[x], (x as u32).min(MAX_SKIP), "at x={x}");
        }
    }

    #[test]
    fn the_skip_box_around_a_brick_is_genuinely_empty() {
        // Restates the safety argument in the module docs as a test: if
        // d(b) = k, no brick within Chebyshev distance k-1 of b is active.
        let mut rng = Lcg(0xC0FFEE);
        let dims = UVec3::new(15, 13, 12);
        let n = (dims.x * dims.y * dims.z) as usize;
        let active: Vec<bool> = (0..n).map(|_| rng.next() % 100 < 8).collect();
        let d = chebyshev_distance_transform(&active, dims, MAX_SKIP);
        let idx = |x: i32, y: i32, z: i32| ((z * dims.y as i32 + y) * dims.x as i32 + x) as usize;

        for z in 0..dims.z as i32 {
            for y in 0..dims.y as i32 {
                for x in 0..dims.x as i32 {
                    let k = d[idx(x, y, z)] as i32;
                    if k == 0 {
                        continue;
                    }
                    let r = k - 1;
                    for dz in -r..=r {
                        for dy in -r..=r {
                            for dx in -r..=r {
                                let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                                if nx < 0
                                    || ny < 0
                                    || nz < 0
                                    || nx >= dims.x as i32
                                    || ny >= dims.y as i32
                                    || nz >= dims.z as i32
                                {
                                    continue;
                                }
                                assert!(
                                    !active[idx(nx, ny, nz)],
                                    "skipping {r} bricks from ({x},{y},{z}) would miss an active brick"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn support_intersection_matches_the_shader_test() {
        use crate::transfer::Support;
        // A diverging support: opaque below -10 and above 10, transparent in
        // between. This is the pressure case, and the whole reason `gap` exists.
        let s = Support { range: (f32::NEG_INFINITY, f32::INFINITY), gap: Some((-10.0, 10.0)) };
        assert!(!s.intersects(-3.0, 3.0), "a brick entirely in the gap must be skipped");
        assert!(!s.contains(0.0));
        assert!(s.intersects(-3.0, 30.0), "a brick straddling the gap must not be");
        assert!(s.intersects(-30.0, 30.0));
        assert!(s.contains(50.0) && s.contains(-50.0));
        // Exactly on the gap edge is active: the LUT filters across it.
        assert!(s.intersects(-10.0, 10.0));

        // A plain bracket with no gap behaves like a simple interval.
        let t = Support { range: (1.0, 5.0), gap: None };
        assert!(t.intersects(0.0, 2.0));
        assert!(!t.intersects(6.0, 9.0));
        assert!(!t.intersects(-4.0, 0.5));
        // An impossible interval (an empty brick) is never active.
        assert!(!t.intersects(1.0, -1.0));
    }

    #[test]
    fn brick_dims_round_up_and_never_collapse() {
        assert_eq!(brick_dims(UVec3::new(8, 8, 8)), UVec3::new(1, 1, 1));
        assert_eq!(brick_dims(UVec3::new(9, 16, 1)), UVec3::new(2, 2, 1));
        // The interactive tier at half resolution.
        assert_eq!(brick_dims(UVec3::new(174, 120, 120)), UVec3::new(22, 15, 15));
    }

    #[test]
    fn dilate_pass_count_is_even_and_reaches_the_cap() {
        // Even, so the ping-pong ends where the read bind group points; and at
        // least MAX_SKIP, so the transform actually converges.
        assert_eq!(DILATE_PASSES % 2, 0);
        assert!(DILATE_PASSES >= MAX_SKIP);
    }
}

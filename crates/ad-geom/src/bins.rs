//! A uniform spatial bin over triangles.
//!
//! Two things need "which triangles are near here?". The CPU ray-parity
//! reference would otherwise be `rays * triangles` — 5,325 rays against 85,180
//! triangles is 450 million intersection tests — and the incremental
//! re-voxelisation that runs while the user drags an obstruction needs to touch
//! only the triangles whose bins intersect the box the part swept through.
//!
//! A uniform grid rather than a BVH on purpose: the bins have to be rebuilt
//! every time a transform changes, and a uniform grid rebuilds in a single
//! counting sort over the triangles. A BVH would query faster and rebuild far
//! slower, which is the wrong trade for geometry that moves every frame.

use ad_gpu::Bbox;
use glam::{UVec3, Vec3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct TriangleBins {
    /// The region the bins cover. Queries outside it are clamped, which is
    /// correct because no triangle lies outside it either.
    pub bounds: Bbox,
    pub dims: UVec3,
    pub bin_mm: Vec3,
    /// Prefix offsets into `items`, length `dims.x*dims.y*dims.z + 1`.
    starts: Vec<u32>,
    /// Triangle indices, grouped by bin.
    items: Vec<u32>,
}

impl TriangleBins {
    /// Build bins of roughly `target_bin_mm` on a side over the given triangle
    /// AABBs.
    ///
    /// The bin size is a tuning knob, not a correctness one. 8-16 cells per bin
    /// keeps the per-bin triangle lists short enough to scan while keeping the
    /// bin count low enough that a rebuild is trivial.
    pub fn build(tri_bounds: &[Bbox], target_bin_mm: f32) -> Self {
        let bounds = tri_bounds
            .iter()
            .copied()
            .fold(Bbox::EMPTY, |a, b| a.union(b))
            .expanded(Vec3::splat(target_bin_mm * 0.5));
        let bounds = if tri_bounds.is_empty() {
            Bbox {
                min: Vec3::ZERO,
                max: Vec3::ONE,
            }
        } else {
            bounds
        };

        let size = bounds.size().max(Vec3::splat(1e-6));
        let dims = (size / target_bin_mm.max(1e-6))
            .ceil()
            .clamp(Vec3::ONE, Vec3::splat(512.0))
            .as_uvec3();
        let bin_mm = size / dims.as_vec3();
        let nbins = (dims.x * dims.y * dims.z) as usize;

        let index = |c: UVec3| ((c.z * dims.y + c.y) * dims.x + c.x) as usize;
        let cell_of = |p: Vec3| -> UVec3 {
            ((p - bounds.min) / bin_mm)
                .floor()
                .clamp(Vec3::ZERO, (dims - UVec3::ONE).as_vec3())
                .as_uvec3()
        };

        // Counting sort. Two parallel passes over the triangles with a serial
        // prefix sum in between; the prefix sum is over bins, of which there are
        // a few thousand at most.
        let counts: Vec<AtomicU32> = (0..nbins).map(|_| AtomicU32::new(0)).collect();
        tri_bounds.par_iter().for_each(|b| {
            let (lo, hi) = (cell_of(b.min), cell_of(b.max));
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        counts[index(UVec3::new(x, y, z))].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

        let mut starts = Vec::with_capacity(nbins + 1);
        let mut total = 0u32;
        for c in &counts {
            starts.push(total);
            total += c.load(Ordering::Relaxed);
        }
        starts.push(total);

        // Reuse `counts` as the per-bin write cursor.
        for (c, s) in counts.iter().zip(&starts) {
            c.store(*s, Ordering::Relaxed);
        }
        let items: Vec<AtomicU32> = (0..total as usize).map(|_| AtomicU32::new(0)).collect();
        tri_bounds.par_iter().enumerate().for_each(|(t, b)| {
            let (lo, hi) = (cell_of(b.min), cell_of(b.max));
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let slot =
                            counts[index(UVec3::new(x, y, z))].fetch_add(1, Ordering::Relaxed);
                        items[slot as usize].store(t as u32, Ordering::Relaxed);
                    }
                }
            }
        });

        Self {
            bounds,
            dims,
            bin_mm,
            starts,
            items: items.into_iter().map(|a| a.into_inner()).collect(),
        }
    }

    /// Build directly from a triangle soup.
    pub fn of_triangles(tris: &[[Vec3; 3]], target_bin_mm: f32) -> Self {
        let bounds: Vec<Bbox> = tris
            .par_iter()
            .map(|t| Bbox::from_points(t.iter().copied()))
            .collect();
        Self::build(&bounds, target_bin_mm)
    }

    pub fn bin_count(&self) -> usize {
        (self.dims.x * self.dims.y * self.dims.z) as usize
    }

    pub fn entry_count(&self) -> usize {
        self.items.len()
    }

    #[inline]
    fn cell_of(&self, p: Vec3) -> UVec3 {
        ((p - self.bounds.min) / self.bin_mm)
            .floor()
            .clamp(Vec3::ZERO, (self.dims - UVec3::ONE).as_vec3())
            .as_uvec3()
    }

    #[inline]
    fn index(&self, c: UVec3) -> usize {
        ((c.z * self.dims.y + c.y) * self.dims.x + c.x) as usize
    }

    /// Visit every triangle in a bin overlapping `query`. A triangle spanning
    /// several bins is visited once per bin, so the caller must dedupe if that
    /// matters; [`Self::collect_in_aabb`] does.
    pub fn for_each_in_aabb(&self, query: Bbox, mut f: impl FnMut(u32)) {
        if query.is_empty() {
            return;
        }
        let lo = self.cell_of(query.min);
        let hi = self.cell_of(query.max);
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let b = self.index(UVec3::new(x, y, z));
                    for t in &self.items[self.starts[b] as usize..self.starts[b + 1] as usize] {
                        f(*t);
                    }
                }
            }
        }
    }

    /// Sorted, deduplicated triangle indices whose bins overlap `query`.
    pub fn collect_in_aabb(&self, query: Bbox, out: &mut Vec<u32>) {
        out.clear();
        self.for_each_in_aabb(query, |t| out.push(t));
        out.sort_unstable();
        out.dedup();
    }

    /// Triangles that a ray along +X at `(y, z)` could possibly hit: the whole
    /// bin column at that `(y, z)`.
    pub fn collect_x_column(&self, y: f32, z: f32, out: &mut Vec<u32>) {
        let q = Bbox {
            min: Vec3::new(self.bounds.min.x, y, z),
            max: Vec3::new(self.bounds.max.x, y, z),
        };
        self.collect_in_aabb(q, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    /// The property that matters: a query must never miss a triangle whose AABB
    /// it overlaps. Extra triangles are merely wasted work.
    #[test]
    fn queries_never_miss_an_overlapping_triangle() {
        let mesh = primitives::uv_sphere(Vec3::new(3.0, -2.0, 1.0), 12.0, 24, 12);
        let tris: Vec<[Vec3; 3]> = (0..mesh.triangle_count())
            .map(|t| mesh.triangle(t))
            .collect();
        let bounds: Vec<Bbox> = tris
            .iter()
            .map(|t| Bbox::from_points(t.iter().copied()))
            .collect();
        let bins = TriangleBins::of_triangles(&tris, 4.0);

        let mut seed = 7u32;
        let mut rnd = |lo: f32, hi: f32| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            lo + (seed >> 8) as f32 / (1 << 24) as f32 * (hi - lo)
        };
        let mut got = Vec::new();
        for _ in 0..200 {
            let c = Vec3::new(rnd(-16.0, 20.0), rnd(-20.0, 16.0), rnd(-14.0, 16.0));
            let h = Vec3::new(rnd(0.1, 6.0), rnd(0.1, 6.0), rnd(0.1, 6.0));
            let q = Bbox {
                min: c - h,
                max: c + h,
            };
            bins.collect_in_aabb(q, &mut got);

            for (t, b) in bounds.iter().enumerate() {
                let overlaps = b.min.cmple(q.max).all() && b.max.cmpge(q.min).all();
                if overlaps {
                    assert!(
                        got.binary_search(&(t as u32)).is_ok(),
                        "triangle {t} overlaps the query but was not returned"
                    );
                }
            }
        }
    }

    #[test]
    fn every_triangle_lands_in_at_least_one_bin() {
        let mesh = primitives::torus(Vec3::ZERO, 10.0, 3.0, 32, 16);
        let tris: Vec<[Vec3; 3]> = (0..mesh.triangle_count())
            .map(|t| mesh.triangle(t))
            .collect();
        let bins = TriangleBins::of_triangles(&tris, 5.0);
        assert!(bins.entry_count() >= tris.len());

        let mut all = Vec::new();
        bins.collect_in_aabb(bins.bounds, &mut all);
        assert_eq!(
            all.len(),
            tris.len(),
            "a triangle went missing from the bins"
        );
    }

    #[test]
    fn an_x_column_contains_everything_that_row_can_hit() {
        let mesh = primitives::box_mesh(Vec3::new(-5.0, -5.0, -5.0), Vec3::splat(5.0));
        let tris: Vec<[Vec3; 3]> = (0..mesh.triangle_count())
            .map(|t| mesh.triangle(t))
            .collect();
        let bins = TriangleBins::of_triangles(&tris, 3.0);
        let mut out = Vec::new();
        bins.collect_x_column(0.0, 0.0, &mut out);
        // A ray through the centre of a box must at least see the two faces it
        // enters and leaves through, which are two triangles each.
        assert!(
            out.len() >= 4,
            "only {} candidates through the centre",
            out.len()
        );
    }
}

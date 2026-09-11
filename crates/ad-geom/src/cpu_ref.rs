//! CPU reference implementations, used as ground truth for the GPU voxeliser.
//!
//! Two independent methods, which is the point: they share no code, so when
//! they agree on a mask the agreement means something.
//!
//! * [`ray_parity_voxelize`] shoots one ray per grid row and calls a cell solid
//!   when an odd number of surface crossings lie behind it. It knows nothing
//!   about distances, normals or pseudonormals, and it detects a leaky mesh for
//!   free: a watertight surface must give every ray an even crossing count.
//! * [`CpuSdf`] computes the exact signed distance by brute force over the
//!   triangles near a point, signed with the same angle-weighted pseudonormal
//!   the shader uses. This is what the GPU field is compared against value by
//!   value.

use crate::bins::TriangleBins;
use crate::mesh::{closest_point_on_triangle, Feature, TriMesh};
use crate::scene::FlatGeometry;
use ad_gpu::{Bbox, Grid};
use glam::{UVec3, Vec3};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Ray parity
// ---------------------------------------------------------------------------

/// The outcome of a ray-parity voxelisation, including the leak diagnostics.
#[derive(Debug, Clone)]
pub struct RayParityResult {
    pub grid: Grid,
    /// One entry per cell, X fastest, matching [`Grid::linear`].
    pub solid: Vec<bool>,
    pub solid_cells: u64,
    /// One ray per (y, z) row of the grid.
    pub rows_total: usize,
    /// Rows that crossed the surface at all. The rest are empty space and carry
    /// no information either way.
    pub rows_with_hits: usize,
    /// Rows with an odd number of crossings. On a watertight surface this is
    /// zero by definition, so any nonzero value is a hole.
    pub odd_parity_rows: usize,
    /// Up to a handful of `(y, z)` cell coordinates of odd rows, so the caller
    /// can say *where* the leak is rather than only that there is one.
    pub odd_parity_examples: Vec<[u32; 2]>,
    /// Hits that landed within a whisker of a triangle edge. The ray is jittered
    /// off the cell centre precisely to avoid these; a nonzero count means the
    /// jitter was unlucky and the parity may be unreliable.
    pub grazing_hits: usize,
}

impl RayParityResult {
    /// Occupied volume, mm^3. One cell contributes `dx^3`.
    pub fn volume_mm3(&self) -> f64 {
        let dx = self.grid.dx_mm as f64;
        self.solid_cells as f64 * dx * dx * dx
    }

    /// True when every ray that met the surface met it an even number of times.
    pub fn is_watertight(&self) -> bool {
        self.odd_parity_rows == 0
    }

    pub fn report(&self) -> String {
        format!(
            "ray parity: {} of {} rows hit the surface, {} with odd parity{}; \
             {} solid cells = {:.0} mm^3 at dx = {} mm{}",
            self.rows_with_hits,
            self.rows_total,
            self.odd_parity_rows,
            if self.odd_parity_examples.is_empty() {
                String::new()
            } else {
                format!(" (first at cell y,z = {:?})", self.odd_parity_examples[0])
            },
            self.solid_cells,
            self.volume_mm3(),
            self.grid.dx_mm,
            if self.grazing_hits > 0 {
                format!(
                    "; {} grazing hits, parity may be unreliable",
                    self.grazing_hits
                )
            } else {
                String::new()
            },
        )
    }
}

/// Where a +X ray at `(y, z)` crosses a triangle, as an absolute x coordinate.
///
/// Moller-Trumbore, specialised for the fixed direction so the cross products
/// collapse: with `dir = (1,0,0)`, `dir x e2 = (0, -e2.z, e2.y)` and
/// `dir . qvec = qvec.x`.
#[inline]
fn x_ray_hit(y: f32, z: f32, tri: &[Vec3; 3], grazing: &mut usize) -> Option<f32> {
    let [a, b, c] = *tri;
    let e1 = b - a;
    let e2 = c - a;
    let pvec = Vec3::new(0.0, -e2.z, e2.y);
    let det = e1.dot(pvec);
    // A ray parallel to the triangle plane contributes no crossing. Counting one
    // would break parity, so the test is on the determinant and not on `t`.
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = Vec3::new(-a.x, y - a.y, z - a.z);
    let u = tvec.dot(pvec) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(e1);
    let v = qvec.x * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    // Within a whisker of an edge the parity of two adjacent triangles can both
    // fire or both miss. The jitter applied by the caller makes this vanishingly
    // rare, but it is worth counting rather than assuming.
    const GRAZE: f32 = 1e-6;
    if u < GRAZE || v < GRAZE || u + v > 1.0 - GRAZE {
        *grazing += 1;
    }
    Some(e2.dot(qvec) * inv)
}

/// Classify every cell of `grid` as solid or fluid by ray parity.
///
/// One ray per `(y, z)` row, travelling in +X through the row's cell centres.
/// The ray is nudged off the exact centre by a fraction of a cell so it cannot
/// hit a shared edge or a vertex head on, which is the one way parity goes
/// wrong on an otherwise perfect mesh. The nudge is far below the cell size, so
/// it cannot change which cells the ray classifies.
pub fn ray_parity_voxelize(tris: &[[Vec3; 3]], grid: Grid) -> RayParityResult {
    let bins = TriangleBins::of_triangles(tris, (grid.dx_mm * 12.0).max(1e-3));
    let dims = grid.dims;
    let nx = dims.x as usize;

    // Irrational fractions of a cell: no mesh feature can be aligned to both.
    // 1/phi and sqrt(2) - 1, scaled to a thousandth of a cell.
    let jitter_y = grid.dx_mm * 1.0e-3 * 0.618_034;
    let jitter_z = grid.dx_mm * 1.0e-3 * 0.414_213_6;

    let mut solid = vec![false; grid.cell_count() as usize];

    #[derive(Default)]
    struct Stats {
        rows_with_hits: usize,
        odd: usize,
        examples: Vec<[u32; 2]>,
        grazing: usize,
        solid_cells: u64,
    }

    let stats = solid
        .par_chunks_mut(nx)
        .enumerate()
        .map(|(row, cells)| {
            let j = (row % dims.y as usize) as u32;
            let k = (row / dims.y as usize) as u32;
            let y = grid.origin_mm.y + j as f32 * grid.dx_mm + jitter_y;
            let z = grid.origin_mm.z + k as f32 * grid.dx_mm + jitter_z;

            let mut st = Stats::default();
            let mut cand = Vec::new();
            bins.collect_x_column(y, z, &mut cand);
            if cand.is_empty() {
                return st;
            }

            let mut xs: Vec<f32> = Vec::with_capacity(8);
            for t in &cand {
                if let Some(x) = x_ray_hit(y, z, &tris[*t as usize], &mut st.grazing) {
                    xs.push(x);
                }
            }
            if xs.is_empty() {
                return st;
            }
            xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            st.rows_with_hits = 1;
            if xs.len() % 2 == 1 {
                st.odd = 1;
                st.examples.push([j, k]);
            }

            // Walk the row alongside the sorted crossing list: a cell is inside
            // when an odd number of crossings lie behind it.
            let mut crossed = 0usize;
            for (i, cell) in cells.iter_mut().enumerate() {
                let x = grid.origin_mm.x + i as f32 * grid.dx_mm;
                while crossed < xs.len() && xs[crossed] <= x {
                    crossed += 1;
                }
                if crossed % 2 == 1 {
                    *cell = true;
                    st.solid_cells += 1;
                }
            }
            st
        })
        .reduce(Stats::default, |mut a, b| {
            a.rows_with_hits += b.rows_with_hits;
            a.odd += b.odd;
            a.grazing += b.grazing;
            a.solid_cells += b.solid_cells;
            for e in b.examples {
                if a.examples.len() < 8 {
                    a.examples.push(e);
                }
            }
            a
        });

    RayParityResult {
        grid,
        solid,
        solid_cells: stats.solid_cells,
        rows_total: (dims.y * dims.z) as usize,
        rows_with_hits: stats.rows_with_hits,
        odd_parity_rows: stats.odd,
        odd_parity_examples: stats.examples,
        grazing_hits: stats.grazing,
    }
}

// ---------------------------------------------------------------------------
// Exact signed distance
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub tri: usize,
    pub feature: Feature,
    pub point: Vec3,
    /// Unsigned distance, mm.
    pub distance: f32,
    /// Signed distance, negative inside.
    pub signed: f32,
}

/// Brute-force exact signed distance field, accelerated only by spatial bins.
///
/// Slow on purpose: there is nothing clever here to be wrong.
pub struct CpuSdf {
    pub geom: FlatGeometry,
    bins: TriangleBins,
}

impl CpuSdf {
    pub fn new(geom: FlatGeometry) -> Self {
        // Bin size chosen from the mean triangle size rather than from a cell
        // size, because this type has no grid.
        let bounds = geom.bounds();
        let mean = if bounds.is_empty() {
            1.0
        } else {
            bounds.iter().map(|b| b.size().max_element()).sum::<f32>() / bounds.len() as f32
        };
        let bins = TriangleBins::build(&bounds, (mean * 8.0).max(1e-3));
        Self { geom, bins }
    }

    pub fn from_mesh(mesh: &TriMesh) -> Self {
        Self::new(FlatGeometry::from_mesh(mesh))
    }

    pub fn triangle_count(&self) -> usize {
        self.geom.len()
    }

    /// Closest surface point to `p`, with the sign from the pseudonormal.
    ///
    /// The search radius doubles until the best distance found is inside it,
    /// which is what makes the answer exact: any triangle closer than `best`
    /// has a point within `best` of `p`, so its AABB overlaps the query box and
    /// its bin was scanned.
    pub fn closest(&self, p: Vec3) -> Option<Hit> {
        if self.geom.is_empty() {
            return None;
        }
        let mut radius = self.bins.bin_mm.max_element();
        let limit = self.bins.bounds.size().length() + radius;
        let mut cand = Vec::new();

        loop {
            let q = Bbox {
                min: p - Vec3::splat(radius),
                max: p + Vec3::splat(radius),
            };
            self.bins.collect_in_aabb(q, &mut cand);
            if let Some(hit) = self.best_of(p, cand.iter().copied()) {
                if hit.distance <= radius || radius >= limit {
                    return Some(hit);
                }
            } else if radius >= limit {
                // Nothing in any bin: fall back to every triangle. Only
                // reachable for degenerate inputs, but a silent `None` here
                // would look like empty space.
                return self.best_of(p, 0..self.geom.len() as u32);
            }
            radius *= 2.0;
        }
    }

    fn best_of(&self, p: Vec3, candidates: impl Iterator<Item = u32>) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for t in candidates {
            let t = t as usize;
            let [a, b, c] = self.geom.tris[t];
            let (q, feature) = closest_point_on_triangle(p, a, b, c);
            let d = (p - q).length();
            if best.is_none_or(|h| d < h.distance) {
                let n = self.geom.pseudonormal(t, feature);
                let signed = if (p - q).dot(n) > 0.0 { d } else { -d };
                best = Some(Hit {
                    tri: t,
                    feature,
                    point: q,
                    distance: d,
                    signed,
                });
            }
        }
        best
    }

    pub fn signed_distance(&self, p: Vec3) -> f32 {
        self.closest(p).map(|h| h.signed).unwrap_or(f32::INFINITY)
    }

    /// The sign the *naive* nearest-face-normal test would produce.
    ///
    /// Exists only so a test can demonstrate that it is wrong on geometry with
    /// sharp edges, which is the entire justification for the pseudonormal
    /// machinery. Never use this for anything real.
    pub fn naive_signed_distance(&self, p: Vec3) -> f32 {
        match self.closest(p) {
            Some(h) => {
                let n = self.geom.pn[h.tri][0];
                if (p - h.point).dot(n) > 0.0 {
                    h.distance
                } else {
                    -h.distance
                }
            }
            None => f32::INFINITY,
        }
    }

    /// Sample the exact signed distance at every cell centre within `band_mm` of
    /// the surface. Cells outside the band get `NaN`, because a narrow-band
    /// field makes no promise about them.
    pub fn sample_band(&self, grid: Grid, band_mm: f32) -> Vec<f32> {
        let nx = grid.dims.x as usize;
        let mut out = vec![f32::NAN; grid.cell_count() as usize];
        out.par_chunks_mut(nx).enumerate().for_each(|(row, cells)| {
            let j = (row % grid.dims.y as usize) as u32;
            let k = (row / grid.dims.y as usize) as u32;
            for (i, cell) in cells.iter_mut().enumerate() {
                let p = grid.cell_center_mm(UVec3::new(i as u32, j, k));
                if let Some(h) = self.closest(p) {
                    if h.distance <= band_mm {
                        *cell = h.signed;
                    }
                }
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn soup(mesh: &TriMesh) -> Vec<[Vec3; 3]> {
        (0..mesh.triangle_count())
            .map(|t| mesh.triangle(t))
            .collect()
    }

    fn grid_around(b: Bbox, dx: f32) -> Grid {
        Grid::covering(b.expanded(Vec3::splat(dx * 6.0)), dx)
    }

    #[test]
    fn ray_parity_recovers_the_volume_of_a_box() {
        let (min, max) = (Vec3::new(2.0, 3.0, 4.0), Vec3::new(22.0, 13.0, 34.0));
        let m = primitives::box_mesh(min, max);
        let g = grid_around(m.bbox(), 0.5);
        let r = ray_parity_voxelize(&soup(&m), g);
        assert!(r.is_watertight(), "{}", r.report());
        assert_eq!(r.grazing_hits, 0, "{}", r.report());
        let exact = 20.0 * 10.0 * 30.0;
        assert!(
            (r.volume_mm3() / exact - 1.0).abs() < 0.02,
            "{} vs {exact}: {}",
            r.volume_mm3(),
            r.report()
        );
    }

    #[test]
    fn ray_parity_recovers_the_volume_of_a_sphere() {
        let m = primitives::uv_sphere(Vec3::new(1.0, -2.0, 0.5), 12.0, 64, 32);
        let g = grid_around(m.bbox(), 0.5);
        let r = ray_parity_voxelize(&soup(&m), g);
        assert!(r.is_watertight(), "{}", r.report());
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 12.0f64.powi(3);
        assert!(
            (r.volume_mm3() / exact - 1.0).abs() < 0.02,
            "{}",
            r.report()
        );
    }

    /// A hole in the mesh must be reported, not silently voxelised into
    /// something plausible-looking. This is the failure mode that produces a
    /// solver that diverges for no visible reason.
    #[test]
    fn a_missing_triangle_shows_up_as_odd_parity() {
        let mut m = primitives::uv_sphere(Vec3::ZERO, 10.0, 32, 16);
        assert!(m.topology().is_watertight_manifold());
        // Remove one triangle from the equator, where plenty of rows pass
        // through it.
        let victim = m.triangle_count() / 2;
        m.indices.remove(victim);
        assert!(!m.topology().is_watertight_manifold());

        let g = grid_around(m.bbox(), 0.5);
        let r = ray_parity_voxelize(&soup(&m), g);
        assert!(
            !r.is_watertight(),
            "the hole went unnoticed: {}",
            r.report()
        );
        assert!(!r.odd_parity_examples.is_empty());
        assert!(r.report().contains("odd parity"));
    }

    #[test]
    fn exact_sdf_matches_the_analytic_box_everywhere_including_the_corners() {
        let (min, max) = (Vec3::new(-4.0, -3.0, -6.0), Vec3::new(5.0, 7.0, 2.0));
        let sdf = CpuSdf::from_mesh(&primitives::box_mesh(min, max));

        let mut worst = 0.0f32;
        // Deliberately walk right through the corners and edges, on a lattice
        // offset so points land exactly on face planes as well as off them.
        let mut z = -9.0;
        while z <= 5.0 {
            let mut y = -6.0;
            while y <= 10.0 {
                let mut x = -7.0;
                while x <= 8.0 {
                    let p = Vec3::new(x, y, z);
                    let got = sdf.signed_distance(p);
                    let want = primitives::box_sdf(p, min, max);
                    worst = worst.max((got - want).abs());
                    x += 0.5;
                }
                y += 0.5;
            }
            z += 0.5;
        }
        assert!(
            worst < 1e-3,
            "worst error {worst} mm against the analytic box SDF"
        );
    }

    #[test]
    fn exact_sdf_matches_an_analytic_sphere_within_the_tessellation_error() {
        let r = 10.0f32;
        let n = 160u32;
        let sdf = CpuSdf::from_mesh(&primitives::uv_sphere(Vec3::ZERO, r, n, n / 2));
        // An inscribed polyhedron sits inside the sphere by the sagitta of the
        // longest chord across a facet, which is a quad *diagonal*, not an
        // edge: sqrt(2) * theta, hence r * (sqrt(2) theta)^2 / 8 = r theta^2 / 4.
        let theta = std::f32::consts::TAU / n as f32;
        let tol = r * theta * theta / 4.0 * 1.5;

        let mut seed = 3u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
        };
        for _ in 0..3000 {
            let p = Vec3::new(rnd(), rnd(), rnd()) * 15.0;
            let want = p.length() - r;
            let got = sdf.signed_distance(p);
            assert!(
                (got - want).abs() < tol,
                "{got} vs {want} at {p:?} (tol {tol})"
            );
        }
    }

    #[test]
    fn ray_parity_and_the_exact_sdf_agree_on_a_torus() {
        // Two completely independent methods, one shape with a hole through it.
        let m = primitives::torus(Vec3::ZERO, 12.0, 4.0, 96, 48);
        let g = grid_around(m.bbox(), 0.6);
        let parity = ray_parity_voxelize(&soup(&m), g);
        assert!(parity.is_watertight(), "{}", parity.report());

        let sdf = CpuSdf::from_mesh(&m);
        let mut disagree = 0usize;
        let mut checked = 0usize;
        // Every 13th cell: enough coverage without an O(cells * triangles) test.
        for c in (0..g.cell_count() as usize).step_by(13) {
            let x = (c % g.dims.x as usize) as u32;
            let y = ((c / g.dims.x as usize) % g.dims.y as usize) as u32;
            let z = (c / (g.dims.x as usize * g.dims.y as usize)) as u32;
            let p = g.cell_center_mm(UVec3::new(x, y, z));
            let d = sdf.signed_distance(p);
            // Skip cells within half a cell of the surface: parity classifies by
            // the cell centre and so does the SDF, but a centre sitting within
            // float noise of the surface can legitimately go either way.
            if d.abs() < 1e-3 {
                continue;
            }
            checked += 1;
            if (d < 0.0) != parity.solid[c] {
                disagree += 1;
            }
        }
        assert!(checked > 5_000, "only checked {checked} cells");
        assert_eq!(disagree, 0, "{disagree} of {checked} cells disagreed");
    }
}

//! Indexed triangle mesh, its topology, and the closest-point query the whole
//! voxeliser is built on.
//!
//! Everything downstream — sign determination, `q` extraction, mouth detection,
//! subdivision — needs the same three things from a mesh: consistent winding,
//! shared vertices, and edge adjacency. So they are computed once here and the
//! results are handed around rather than rederived.
//!
//! Units are millimetres, per the build contract.

use ad_gpu::Bbox;
use glam::Vec3;
use std::collections::HashMap;

/// Which feature of a triangle a closest-point query landed on.
///
/// This matters far more than it looks. The *sign* of the distance field is
/// determined by the surface normal at the closest point, and on a triangle
/// mesh that normal is only well defined in the interior of a face. On an edge
/// or a vertex the correct object is the angle-weighted pseudonormal, so the
/// query has to say which case it hit. See [`crate::mesh::PseudoNormals`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// Corner `k` of the triangle, `k` in 0..3.
    Vertex(u8),
    /// Edge `k`, joining corner `k` to corner `(k + 1) % 3`.
    Edge(u8),
    /// Strictly inside the face.
    Face,
}

impl Feature {
    /// Encoding shared with the WGSL side: 0..3 vertices, 3..6 edges, 6 face.
    pub fn code(self) -> u32 {
        match self {
            Feature::Vertex(k) => k as u32,
            Feature::Edge(k) => 3 + k as u32,
            Feature::Face => 6,
        }
    }
}

/// Closest point on triangle `(a, b, c)` to `p`, plus which feature it lies on.
///
/// Ericson, *Real-Time Collision Detection* 5.1.5, extended to report the
/// feature. The WGSL implementation in `shaders/geom/closest_point.wgsl` is a
/// line-for-line translation of this function; if you change one, change both,
/// and `gpu_and_cpu_sdf_agree` is the test that will notice if you do not.
///
/// Edge numbering is `edge 0 = (a, b)`, `edge 1 = (b, c)`, `edge 2 = (c, a)`,
/// which is the same convention [`TriMesh::edges_of`] uses.
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Feature) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(0));
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(1));
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let t = if denom != 0.0 { d1 / denom } else { 0.0 };
        return (a + ab * t, Feature::Edge(0));
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(2));
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let t = if denom != 0.0 { d2 / denom } else { 0.0 };
        return (a + ac * t, Feature::Edge(2));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let t = if denom != 0.0 { (d4 - d3) / denom } else { 0.0 };
        return (b + (c - b) * t, Feature::Edge(1));
    }

    // Interior. A zero-area triangle reaches here with `va + vb + vc == 0`;
    // welding drops those, but a sliver that survives must not produce a NaN.
    let sum = va + vb + vc;
    if sum == 0.0 {
        return (a, Feature::Vertex(0));
    }
    let denom = 1.0 / sum;
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, Feature::Face)
}

/// An indexed triangle mesh in millimetres.
#[derive(Debug, Clone, Default)]
pub struct TriMesh {
    pub positions: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
    /// Normals as they appeared in the source file, if there were any.
    ///
    /// Kept only so [`MeshHealth`] can report how far the file disagrees with
    /// the winding. Nothing computes with them: plenty of exporters write
    /// zeros, unnormalised vectors, or normals that contradict the vertex order,
    /// and trusting them silently inverts the sign of the distance field.
    pub file_normals: Option<Vec<Vec3>>,
}

impl TriMesh {
    pub fn new(positions: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Self {
        Self { positions, indices, file_normals: None }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    #[inline]
    pub fn triangle(&self, t: usize) -> [Vec3; 3] {
        let i = self.indices[t];
        [
            self.positions[i[0] as usize],
            self.positions[i[1] as usize],
            self.positions[i[2] as usize],
        ]
    }

    /// The three edges of triangle `t` as vertex-index pairs, in the
    /// `(0,1), (1,2), (2,0)` order [`Feature::Edge`] refers to.
    #[inline]
    pub fn edges_of(&self, t: usize) -> [(u32, u32); 3] {
        let i = self.indices[t];
        [(i[0], i[1]), (i[1], i[2]), (i[2], i[0])]
    }

    /// Unnormalised face normal: twice the area, direction from the winding.
    #[inline]
    pub fn face_normal_raw(&self, t: usize) -> Vec3 {
        let [a, b, c] = self.triangle(t);
        (b - a).cross(c - a)
    }

    /// Unit face normal, recomputed from the winding. Zero for a degenerate
    /// triangle.
    #[inline]
    pub fn face_normal(&self, t: usize) -> Vec3 {
        self.face_normal_raw(t).normalize_or_zero()
    }

    pub fn face_normals(&self) -> Vec<Vec3> {
        (0..self.triangle_count()).map(|t| self.face_normal(t)).collect()
    }

    pub fn bbox(&self) -> Bbox {
        Bbox::from_points(self.positions.iter().copied())
    }

    pub fn surface_area(&self) -> f64 {
        (0..self.triangle_count())
            .map(|t| self.face_normal_raw(t).length() as f64 * 0.5)
            .sum()
    }

    /// Signed volume by the divergence theorem, mm^3.
    ///
    /// `V = 1/6 * sum over triangles of a . (b x c)`. Positive means the winding
    /// is outward (counter-clockwise seen from outside), which is the STL
    /// convention. A negative value means the whole mesh is inside-out; a value
    /// near zero on a mesh with real volume means the winding is inconsistent.
    ///
    /// Accumulated in `f64`: at 85k triangles the `f32` sum loses about three
    /// significant figures to cancellation, which is enough to fail a 2% check.
    pub fn signed_volume(&self) -> f64 {
        (0..self.triangle_count())
            .map(|t| {
                let [a, b, c] = self.triangle(t);
                let (a, b, c) = (a.as_dvec3(), b.as_dvec3(), c.as_dvec3());
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    /// Number of triangles with zero area.
    pub fn degenerate_count(&self) -> usize {
        (0..self.triangle_count())
            .filter(|&t| self.face_normal_raw(t).length_squared() == 0.0)
            .count()
    }

    /// Flip every triangle's winding. Used when the signed volume comes out
    /// negative, i.e. the file was authored inside-out.
    pub fn flip_winding(&mut self) {
        for i in self.indices.iter_mut() {
            i.swap(1, 2);
        }
        if let Some(n) = &mut self.file_normals {
            for v in n.iter_mut() {
                *v = -*v;
            }
        }
    }

    pub fn topology(&self) -> Topology {
        Topology::of(self)
    }

    pub fn health(&self) -> MeshHealth {
        MeshHealth::of(self)
    }

    pub fn pseudo_normals(&self) -> PseudoNormals {
        PseudoNormals::of(self)
    }

    /// Exact unsigned distance from `p` to triangle `t`, plus the closest point
    /// and its feature.
    #[inline]
    pub fn closest_point(&self, t: usize, p: Vec3) -> (Vec3, Feature) {
        let [a, b, c] = self.triangle(t);
        closest_point_on_triangle(p, a, b, c)
    }
}

/// Edge adjacency and the invariants derived from it.
#[derive(Debug, Clone)]
pub struct Topology {
    /// Unique undirected edges.
    pub edge_count: usize,
    /// `valence_histogram[n]` = number of edges shared by exactly `n`
    /// triangles. Index 0 is unused; anything past index 2 is non-manifold.
    pub valence_histogram: Vec<usize>,
    /// Edges used by exactly one triangle: the mesh has a hole there.
    pub boundary_edges: usize,
    /// Edges used by three or more triangles.
    pub non_manifold_edges: usize,
    /// Undirected edges whose two triangles traverse them the same way round,
    /// meaning their windings disagree.
    pub inconsistent_edges: usize,
    /// Euler characteristic `V - E + F` over the vertices actually referenced.
    pub euler_characteristic: i64,
    /// For each triangle and each of its three edges, the index of the triangle
    /// on the other side, or `u32::MAX` if there is not exactly one.
    pub edge_neighbour: Vec<[u32; 3]>,
}

impl Topology {
    pub fn of(mesh: &TriMesh) -> Self {
        // (min, max) vertex index -> up to two (triangle, edge) uses. A third
        // use makes the edge non-manifold, and we only need to count those.
        let mut map: HashMap<(u32, u32), EdgeUses> =
            HashMap::with_capacity(mesh.triangle_count() * 2);
        let mut used_vertices = vec![false; mesh.vertex_count()];

        for t in 0..mesh.triangle_count() {
            for (k, (u, v)) in mesh.edges_of(t).into_iter().enumerate() {
                used_vertices[u as usize] = true;
                used_vertices[v as usize] = true;
                let key = if u < v { (u, v) } else { (v, u) };
                // `forward` records whether this triangle walks the edge from
                // the lower index to the higher one. Two triangles sharing an
                // edge must disagree, or their windings are inconsistent.
                let forward = u < v;
                let e = map.entry(key).or_default();
                e.count += 1;
                if e.count <= 2 {
                    e.tri[e.count as usize - 1] = (t as u32, k as u8, forward);
                }
            }
        }

        let mut valence_histogram = vec![0usize; 3];
        let mut inconsistent_edges = 0;
        let mut non_manifold_edges = 0;
        let mut edge_neighbour = vec![[u32::MAX; 3]; mesh.triangle_count()];

        for uses in map.values() {
            let n = uses.count as usize;
            if n >= valence_histogram.len() {
                valence_histogram.resize(n + 1, 0);
            }
            valence_histogram[n] += 1;
            if n > 2 {
                non_manifold_edges += 1;
            }
            if n == 2 {
                let (t0, k0, f0) = uses.tri[0];
                let (t1, k1, f1) = uses.tri[1];
                edge_neighbour[t0 as usize][k0 as usize] = t1;
                edge_neighbour[t1 as usize][k1 as usize] = t0;
                if f0 == f1 {
                    inconsistent_edges += 1;
                }
            }
        }

        let v = used_vertices.iter().filter(|u| **u).count() as i64;
        let e = map.len() as i64;
        let f = mesh.triangle_count() as i64;

        Self {
            edge_count: map.len(),
            boundary_edges: valence_histogram[1],
            non_manifold_edges,
            inconsistent_edges,
            euler_characteristic: v - e + f,
            valence_histogram,
            edge_neighbour,
        }
    }

    /// Every edge shared by exactly two consistently wound triangles.
    pub fn is_watertight_manifold(&self) -> bool {
        self.boundary_edges == 0 && self.non_manifold_edges == 0 && self.inconsistent_edges == 0
    }

    /// Genus, for a closed orientable surface: `chi = 2 - 2g`.
    pub fn genus(&self) -> Option<i64> {
        if !self.is_watertight_manifold() || (2 - self.euler_characteristic) % 2 != 0 {
            return None;
        }
        Some((2 - self.euler_characteristic) / 2)
    }
}

#[derive(Default, Clone, Copy)]
struct EdgeUses {
    count: u32,
    tri: [(u32, u8, bool); 2],
}

/// Everything worth telling the user about a mesh before simulating it.
#[derive(Debug, Clone)]
pub struct MeshHealth {
    pub triangle_count: usize,
    pub vertex_count: usize,
    pub bbox: Bbox,
    pub surface_area_mm2: f64,
    /// Signed volume from the divergence theorem, mm^3. Meaningless unless the
    /// mesh is watertight.
    pub signed_volume_mm3: f64,
    pub degenerate_triangles: usize,
    pub topology: Topology,
    /// Triangles whose recomputed normal disagrees with the file's by more than
    /// 1 degree. Purely informational; the recomputed normal always wins.
    pub normal_disagreements: usize,
    /// Triangles whose file normal points the opposite way to the winding.
    pub normal_inversions: usize,
}

impl MeshHealth {
    pub fn of(mesh: &TriMesh) -> Self {
        let mut normal_disagreements = 0;
        let mut normal_inversions = 0;
        if let Some(file_normals) = &mesh.file_normals {
            for (t, fnorm) in file_normals.iter().enumerate().take(mesh.triangle_count()) {
                let computed = mesh.face_normal(t);
                let stated = fnorm.normalize_or_zero();
                if stated == Vec3::ZERO || computed == Vec3::ZERO {
                    continue;
                }
                let d = stated.dot(computed);
                // cos(1 degree) = 0.99985.
                if d < 0.99985 {
                    normal_disagreements += 1;
                }
                if d < 0.0 {
                    normal_inversions += 1;
                }
            }
        }

        Self {
            triangle_count: mesh.triangle_count(),
            vertex_count: mesh.vertex_count(),
            bbox: mesh.bbox(),
            surface_area_mm2: mesh.surface_area(),
            signed_volume_mm3: mesh.signed_volume(),
            degenerate_triangles: mesh.degenerate_count(),
            topology: mesh.topology(),
            normal_disagreements,
            normal_inversions,
        }
    }

    pub fn is_watertight_manifold(&self) -> bool {
        self.topology.is_watertight_manifold()
    }

    /// A human-readable report. Written to be pasted into a bug report as-is.
    pub fn report(&self) -> String {
        let size = self.bbox.size();
        let mut s = String::new();
        s.push_str(&format!(
            "{} triangles, {} vertices\n\
             bbox {:.3} x {:.3} x {:.3} mm  (min {:.3},{:.3},{:.3})\n\
             surface area {:.1} mm^2, signed volume {:.1} mm^3\n\
             edges {}, chi = {}",
            self.triangle_count,
            self.vertex_count,
            size.x,
            size.y,
            size.z,
            self.bbox.min.x,
            self.bbox.min.y,
            self.bbox.min.z,
            self.surface_area_mm2,
            self.signed_volume_mm3,
            self.topology.edge_count,
            self.topology.euler_characteristic,
        ));
        if let Some(g) = self.topology.genus() {
            s.push_str(&format!(" (genus {g})"));
        }
        s.push('\n');
        s.push_str("edge valence:");
        for (n, count) in self.topology.valence_histogram.iter().enumerate() {
            if *count > 0 {
                s.push_str(&format!(" {n}x{count}"));
            }
        }
        s.push('\n');
        if self.is_watertight_manifold() {
            s.push_str("watertight 2-manifold: yes\n");
        } else {
            s.push_str(&format!(
                "watertight 2-manifold: NO ({} boundary edges, {} non-manifold edges, \
                 {} inconsistently wound edges)\n",
                self.topology.boundary_edges,
                self.topology.non_manifold_edges,
                self.topology.inconsistent_edges,
            ));
        }
        if self.degenerate_triangles > 0 {
            s.push_str(&format!("{} degenerate triangles\n", self.degenerate_triangles));
        }
        if self.normal_disagreements > 0 {
            s.push_str(&format!(
                "{} of {} file normals disagree with the winding by >1 degree ({} fully \
                 inverted); the winding is used\n",
                self.normal_disagreements, self.triangle_count, self.normal_inversions,
            ));
        }
        s
    }
}

/// Angle-weighted pseudonormals, per Baerentzen & Aanaes 2005,
/// *Signed distance computation using the angle weighted pseudonormal*
/// (IEEE TVCG 11(3)).
///
/// # Why the obvious thing is wrong
///
/// To sign a distance field you need the outward normal at the closest point on
/// the surface. On a triangle mesh that is only defined in the interior of a
/// face. The tempting shortcut — take the nearest triangle's face normal and
/// test `dot(p - closest, n)` — is *provably* wrong whenever the closest point
/// lands on an edge or a vertex shared by faces more than 90 degrees apart.
///
/// Concretely: at a convex edge whose interior dihedral angle is less than 90
/// degrees (a knife edge, which a printed duct lip is), a point sitting just
/// outside, almost directly above face A, is equidistant from faces A and B. If
/// the search happens to return B, then `dot(p - closest, n_B) < 0` and the
/// voxel is declared solid even though it is plainly outside. That produces
/// isolated solid cells hanging in the fluid and isolated fluid cells inside the
/// wall, and the solver diverges within a few hundred steps.
///
/// The fix is Baerentzen's theorem: replace the normal at a non-face feature
/// with
///
/// - **edge**: `n_1 + n_2`, the two incident face normals summed (each weighted
///   by its incident angle of pi, so the weights cancel);
/// - **vertex**: `sum_i alpha_i * n_i` over incident faces, where `alpha_i` is
///   the interior angle of face `i` *at that vertex*.
///
/// `dot(p - closest, N)` then has the correct sign for every point, because the
/// pseudonormal is guaranteed to lie strictly inside the normal cone of the
/// feature. Area weighting or plain averaging do **not** have this property; the
/// angle weight is the entire content of the result.
///
/// The whole thing is a dozen lines and it is the difference between a solver
/// that runs and one that does not.
#[derive(Debug, Clone)]
pub struct PseudoNormals {
    /// Unit face normal per triangle.
    pub face: Vec<Vec3>,
    /// Per triangle, per edge `k` in the `(0,1), (1,2), (2,0)` order.
    /// Unnormalised sums are fine; only the direction is ever used, but they
    /// are normalised anyway so the GPU can compare magnitudes if it wants to.
    pub edge: Vec<[Vec3; 3]>,
    /// Per *vertex* of the mesh, indexed by vertex index.
    pub vertex: Vec<Vec3>,
}

impl PseudoNormals {
    pub fn of(mesh: &TriMesh) -> Self {
        let face = mesh.face_normals();

        // Vertex pseudonormals: accumulate angle * face normal.
        let mut vertex = vec![Vec3::ZERO; mesh.vertex_count()];
        for t in 0..mesh.triangle_count() {
            let n = face[t];
            if n == Vec3::ZERO {
                continue;
            }
            let [a, b, c] = mesh.triangle(t);
            let idx = mesh.indices[t];
            let corners = [(a, b, c), (b, c, a), (c, a, b)];
            for (k, (p, q, r)) in corners.into_iter().enumerate() {
                let u = (q - p).normalize_or_zero();
                let v = (r - p).normalize_or_zero();
                // Interior angle at this corner. `clamp` guards acos against a
                // dot product that rounds a hair outside [-1, 1].
                let angle = u.dot(v).clamp(-1.0, 1.0).acos();
                if angle.is_finite() {
                    vertex[idx[k] as usize] += n * angle;
                }
            }
        }
        for v in vertex.iter_mut() {
            *v = v.normalize_or_zero();
        }

        // Edge pseudonormals: the sum of the two incident face normals. An edge
        // that is not shared by exactly two triangles has no defined
        // pseudonormal; fall back to the face normal so a slightly broken mesh
        // still produces a usable field rather than a NaN. `MeshHealth` will
        // already have said the mesh is not watertight.
        let topo = mesh.topology();
        let mut edge = vec![[Vec3::ZERO; 3]; mesh.triangle_count()];
        for t in 0..mesh.triangle_count() {
            for k in 0..3 {
                let other = topo.edge_neighbour[t][k];
                let sum = if other == u32::MAX {
                    face[t]
                } else {
                    face[t] + face[other as usize]
                };
                edge[t][k] = sum.normalize_or_zero();
                if edge[t][k] == Vec3::ZERO {
                    // Two exactly opposed faces (a zero-thickness fin). No
                    // meaningful outward direction; use this face's own normal.
                    edge[t][k] = face[t];
                }
            }
        }

        Self { face, edge, vertex }
    }

    /// The outward pseudonormal to use for a closest point on `feature` of
    /// triangle `t`.
    #[inline]
    pub fn at(&self, mesh: &TriMesh, t: usize, feature: Feature) -> Vec3 {
        match feature {
            Feature::Face => self.face[t],
            Feature::Edge(k) => self.edge[t][k as usize],
            Feature::Vertex(k) => self.vertex[mesh.indices[t][k as usize] as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn closest_point_finds_each_feature_of_a_triangle() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);

        // Above the interior.
        let (p, f) = closest_point_on_triangle(Vec3::new(0.2, 0.2, 1.0), a, b, c);
        assert_eq!(f, Feature::Face);
        assert!((p - Vec3::new(0.2, 0.2, 0.0)).length() < 1e-6);

        // Beyond each corner.
        for (q, k, want) in [
            (Vec3::new(-1.0, -1.0, 0.0), 0u8, a),
            (Vec3::new(2.0, -1.0, 0.0), 1, b),
            (Vec3::new(-1.0, 2.0, 0.0), 2, c),
        ] {
            let (p, f) = closest_point_on_triangle(q, a, b, c);
            assert_eq!(f, Feature::Vertex(k), "expected vertex {k} for {q:?}");
            assert!((p - want).length() < 1e-6);
        }

        // Off the middle of each edge. Edge k joins corner k to corner k+1.
        for (q, k) in [
            (Vec3::new(0.5, -1.0, 0.0), 0u8),
            (Vec3::new(1.0, 1.0, 0.0), 1),
            (Vec3::new(-1.0, 0.5, 0.0), 2),
        ] {
            let (_, f) = closest_point_on_triangle(q, a, b, c);
            assert_eq!(f, Feature::Edge(k), "expected edge {k} for {q:?}");
        }
    }

    #[test]
    fn closest_point_never_returns_a_point_outside_the_triangle() {
        // Random-ish sampling: the returned point must be a convex combination
        // of the corners and must be at least as close as any corner.
        let a = Vec3::new(0.3, -1.0, 0.2);
        let b = Vec3::new(2.0, 0.5, -0.4);
        let c = Vec3::new(-0.7, 1.3, 1.1);
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 * 6.0 - 3.0
        };
        for _ in 0..2000 {
            let p = Vec3::new(rnd(), rnd(), rnd());
            let (q, _) = closest_point_on_triangle(p, a, b, c);
            let d = (q - p).length();
            for corner in [a, b, c] {
                assert!(d <= (corner - p).length() + 1e-4, "a corner was closer than {q:?}");
            }
        }
    }

    #[test]
    fn a_cube_is_a_watertight_manifold_of_genus_zero() {
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        let h = m.health();
        assert_eq!(h.triangle_count, 12);
        assert_eq!(h.vertex_count, 8);
        assert!(h.is_watertight_manifold(), "{}", h.report());
        assert_eq!(h.topology.euler_characteristic, 2);
        assert_eq!(h.topology.genus(), Some(0));
        // Every edge shared by exactly two triangles.
        assert_eq!(h.topology.valence_histogram[2], h.topology.edge_count);
        assert!((h.signed_volume_mm3 - 1000.0).abs() < 1e-3, "{}", h.signed_volume_mm3);
        assert!((h.surface_area_mm2 - 600.0).abs() < 1e-3);
    }

    #[test]
    fn removing_a_triangle_shows_up_as_boundary_edges() {
        let mut m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        m.indices.pop();
        let h = m.health();
        assert!(!h.is_watertight_manifold());
        assert_eq!(h.topology.boundary_edges, 3);
        assert_eq!(h.topology.genus(), None);
    }

    #[test]
    fn inverted_winding_shows_up_as_negative_volume() {
        let mut m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        m.flip_winding();
        assert!(m.signed_volume() < 0.0);
        // Still a manifold, just inside-out.
        assert!(m.topology().is_watertight_manifold());
        m.flip_winding();
        assert!(m.signed_volume() > 0.0);
    }

    #[test]
    fn sphere_volume_and_area_converge_on_the_analytic_values() {
        let m = primitives::uv_sphere(Vec3::ZERO, 10.0, 128, 64);
        let h = m.health();
        assert!(h.is_watertight_manifold(), "{}", h.report());
        assert_eq!(h.topology.genus(), Some(0));
        let exact_v = 4.0 / 3.0 * std::f64::consts::PI * 1000.0;
        let exact_a = 4.0 * std::f64::consts::PI * 100.0;
        // A polyhedron inscribed in the sphere under-reports both.
        assert!((h.signed_volume_mm3 / exact_v - 1.0).abs() < 2e-3, "{}", h.signed_volume_mm3);
        assert!((h.surface_area_mm2 / exact_a - 1.0).abs() < 2e-3, "{}", h.surface_area_mm2);
    }

    #[test]
    fn a_torus_has_genus_one() {
        let m = primitives::torus(Vec3::ZERO, 10.0, 3.0, 48, 24);
        let t = m.topology();
        assert!(t.is_watertight_manifold());
        assert_eq!(t.euler_characteristic, 0);
        assert_eq!(t.genus(), Some(1));
    }

    #[test]
    fn edge_pseudonormal_bisects_the_two_faces_of_a_cube_edge() {
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        let pn = m.pseudo_normals();
        // Find a triangle/edge pair whose neighbour has a different normal.
        let topo = m.topology();
        let mut checked = 0;
        for t in 0..m.triangle_count() {
            for k in 0..3 {
                let o = topo.edge_neighbour[t][k] as usize;
                let (n1, n2) = (pn.face[t], pn.face[o]);
                if n1.dot(n2).abs() > 0.9 {
                    continue; // coplanar diagonal of a face
                }
                let want = (n1 + n2).normalize();
                assert!((pn.edge[t][k] - want).length() < 1e-5);
                // A cube edge pseudonormal points 45 degrees out from both.
                assert!((pn.edge[t][k].dot(n1) - 0.5f32.sqrt()).abs() < 1e-5);
                checked += 1;
            }
        }
        assert!(checked >= 24, "expected to check every cube edge twice, got {checked}");
    }

    #[test]
    fn vertex_pseudonormal_of_a_cube_corner_is_the_body_diagonal() {
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        let pn = m.pseudo_normals();
        // The corner at (10,10,10) must point along (1,1,1)/sqrt(3). This is the
        // case angle weighting gets right and area weighting does not: the
        // corner is shared by three faces, but two of them are split into
        // triangles that touch the corner with different areas.
        let corner = m
            .positions
            .iter()
            .position(|p| (*p - Vec3::splat(10.0)).length() < 1e-6)
            .expect("cube corner");
        let want = Vec3::splat(1.0).normalize();
        assert!(
            (pn.vertex[corner] - want).length() < 1e-5,
            "corner pseudonormal was {:?}",
            pn.vertex[corner]
        );
    }
}

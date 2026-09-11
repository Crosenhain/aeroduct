//! Analytic test shapes.
//!
//! These exist so the voxeliser can be checked against a distance field that is
//! known in closed form rather than against another approximation. A box gives
//! an exact SDF everywhere, including at its edges and corners, which is the
//! only place the sign logic can go wrong. A convex prism gives an exact
//! *inside/outside* test even where its distance field is awkward, which is
//! enough to catch a sign flip.
//!
//! They are `pub` rather than test-only because the validation harness in
//! `validation/geom/` is a separate compilation unit and needs them too.

use crate::mesh::TriMesh;
use glam::{Vec2, Vec3};

/// Axis-aligned box, outward wound, 12 triangles over 8 shared vertices.
pub fn box_mesh(min: Vec3, max: Vec3) -> TriMesh {
    let p = |i: usize| {
        Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        )
    };
    let positions: Vec<Vec3> = (0..8).map(p).collect();
    // Each face as a quad of corner indices, wound counter-clockwise seen from
    // outside.
    let quads: [[u32; 4]; 6] = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    let mut indices = Vec::with_capacity(12);
    for q in quads {
        indices.push([q[0], q[1], q[2]]);
        indices.push([q[0], q[2], q[3]]);
    }
    TriMesh::new(positions, indices)
}

/// Exact signed distance to an axis-aligned box. Negative inside.
///
/// This is the closed form the voxeliser is measured against, so it is
/// deliberately the textbook one (Quilez): the outside term is the distance to
/// the clamped point, the inside term is the largest (least negative) face
/// distance.
pub fn box_sdf(p: Vec3, min: Vec3, max: Vec3) -> f32 {
    let center = (min + max) * 0.5;
    let half = (max - min) * 0.5;
    let d = (p - center).abs() - half;
    let outside = d.max(Vec3::ZERO).length();
    let inside = d.x.max(d.y.max(d.z)).min(0.0);
    outside + inside
}

/// UV sphere. `segments` around the equator, `rings` latitude bands.
///
/// Poles are single shared vertices, so this is a closed genus-0 manifold. The
/// polyhedron is inscribed, so its surface sits *inside* the true sphere by up
/// to `r * theta^2 / 8`; pick the tessellation with that in mind when comparing
/// against the analytic SDF.
pub fn uv_sphere(center: Vec3, radius: f32, segments: u32, rings: u32) -> TriMesh {
    assert!(segments >= 3 && rings >= 2);
    let mut positions = Vec::with_capacity((segments * (rings - 1) + 2) as usize);
    let mut indices = Vec::new();

    positions.push(center + Vec3::new(0.0, 0.0, radius)); // north pole = 0
    for j in 1..rings {
        let theta = std::f32::consts::PI * j as f32 / rings as f32;
        let (st, ct) = theta.sin_cos();
        for i in 0..segments {
            let phi = std::f32::consts::TAU * i as f32 / segments as f32;
            let (sp, cp) = phi.sin_cos();
            positions.push(center + Vec3::new(radius * st * cp, radius * st * sp, radius * ct));
        }
    }
    positions.push(center + Vec3::new(0.0, 0.0, -radius)); // south pole
    let south = (positions.len() - 1) as u32;
    let ring = |j: u32, i: u32| 1 + (j - 1) * segments + (i % segments);

    for i in 0..segments {
        indices.push([0, ring(1, i), ring(1, i + 1)]);
    }
    for j in 1..rings - 1 {
        for i in 0..segments {
            let (a, b) = (ring(j, i), ring(j, i + 1));
            let (c, d) = (ring(j + 1, i), ring(j + 1, i + 1));
            indices.push([a, c, d]);
            indices.push([a, d, b]);
        }
    }
    for i in 0..segments {
        indices.push([south, ring(rings - 1, i + 1), ring(rings - 1, i)]);
    }

    TriMesh::new(positions, indices)
}

/// Torus in the XY plane. Genus 1, which is what the test part is, so this is
/// the shape that proves the Euler characteristic bookkeeping.
pub fn torus(center: Vec3, major: f32, minor: f32, nu: u32, nv: u32) -> TriMesh {
    assert!(nu >= 3 && nv >= 3);
    let mut positions = Vec::with_capacity((nu * nv) as usize);
    for i in 0..nu {
        let u = std::f32::consts::TAU * i as f32 / nu as f32;
        let (su, cu) = u.sin_cos();
        for j in 0..nv {
            let v = std::f32::consts::TAU * j as f32 / nv as f32;
            let (sv, cv) = v.sin_cos();
            let r = major + minor * cv;
            positions.push(center + Vec3::new(r * cu, r * su, minor * sv));
        }
    }
    let idx = |i: u32, j: u32| (i % nu) * nv + (j % nv);
    let mut indices = Vec::with_capacity((nu * nv * 2) as usize);
    for i in 0..nu {
        for j in 0..nv {
            let (a, b, c, d) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
    }
    TriMesh::new(positions, indices)
}

/// Extrude a convex, counter-clockwise polygon in XY between two Z planes.
///
/// Convexity is required because the caps are fan-triangulated.
pub fn prism(polygon: &[Vec2], z0: f32, z1: f32) -> TriMesh {
    let n = polygon.len() as u32;
    assert!(n >= 3, "a prism needs at least a triangle");
    let mut positions = Vec::with_capacity(polygon.len() * 2);
    for p in polygon {
        positions.push(Vec3::new(p.x, p.y, z0));
    }
    for p in polygon {
        positions.push(Vec3::new(p.x, p.y, z1));
    }

    let mut indices = Vec::new();
    // Sides. For a CCW polygon the outward 2D normal of edge (p_i, p_i+1) is
    // (e.y, -e.x), and this winding reproduces it.
    for i in 0..n {
        let j = (i + 1) % n;
        indices.push([i, j, j + n]);
        indices.push([i, j + n, i + n]);
    }
    // Caps: bottom faces -Z so its fan is reversed, top faces +Z.
    for i in 1..n - 1 {
        indices.push([0, i + 1, i]);
        indices.push([n, n + i, n + i + 1]);
    }
    TriMesh::new(positions, indices)
}

/// A prism with a deliberately sharp apex: the case the naive
/// nearest-face-normal sign test gets wrong.
///
/// The apex sits at the origin pointing along -X, with an interior dihedral
/// angle of `apex_deg`. For `apex_deg < 90` there is a wedge of points just
/// outside the solid, nearly directly above one of the two faces, for which the
/// *other* face is equidistant and whose normal points away from them — so a
/// nearest-triangle sign test flips there. See [`crate::mesh::PseudoNormals`].
pub fn sharp_wedge(apex_deg: f32, length: f32, depth: f32) -> TriMesh {
    let half = apex_deg.to_radians() * 0.5;
    let (s, c) = half.sin_cos();
    let poly = [
        Vec2::new(0.0, 0.0),
        Vec2::new(length * c, -length * s),
        Vec2::new(length * c, length * s),
    ];
    prism(&poly, -depth * 0.5, depth * 0.5)
}

/// Half-space description of a convex solid, for exact inside/outside tests.
///
/// `inside(p)` is `max_i(dot(n_i, p) - d_i) <= 0`, which is exact for any
/// intersection of half-spaces regardless of how awkward the true distance
/// field is near an edge.
pub struct ConvexSolid {
    pub planes: Vec<(Vec3, f32)>,
}

impl ConvexSolid {
    /// Derive the half-spaces from a convex mesh by deduplicating face planes.
    pub fn from_convex_mesh(mesh: &TriMesh) -> Self {
        let mut planes: Vec<(Vec3, f32)> = Vec::new();
        for t in 0..mesh.triangle_count() {
            let n = mesh.face_normal(t);
            if n == Vec3::ZERO {
                continue;
            }
            let d = n.dot(mesh.triangle(t)[0]);
            if !planes
                .iter()
                .any(|(pn, pd)| pn.dot(n) > 0.9999 && (pd - d).abs() < 1e-4)
            {
                planes.push((n, d));
            }
        }
        Self { planes }
    }

    /// Signed "algebraic" distance: exact inside, a lower bound outside.
    pub fn plane_max(&self, p: Vec3) -> f32 {
        self.planes
            .iter()
            .map(|(n, d)| n.dot(p) - d)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn contains(&self, p: Vec3) -> bool {
        self.plane_max(p) <= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_sdf_matches_brute_force_distance_to_the_mesh() {
        let (min, max) = (Vec3::new(-3.0, -2.0, -5.0), Vec3::new(4.0, 6.0, 1.0));
        let m = box_mesh(min, max);
        let mut seed = 9u32;
        let mut rnd = |lo: f32, hi: f32| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            lo + (seed >> 8) as f32 / (1 << 24) as f32 * (hi - lo)
        };
        for _ in 0..500 {
            let p = Vec3::new(rnd(-8.0, 9.0), rnd(-7.0, 11.0), rnd(-10.0, 6.0));
            let brute = (0..m.triangle_count())
                .map(|t| (m.closest_point(t, p).0 - p).length())
                .fold(f32::INFINITY, f32::min);
            assert!(
                (box_sdf(p, min, max).abs() - brute).abs() < 1e-3,
                "analytic {} vs brute {brute} at {p:?}",
                box_sdf(p, min, max)
            );
        }
    }

    #[test]
    fn prism_and_wedge_are_watertight() {
        for m in [
            sharp_wedge(25.0, 20.0, 10.0),
            prism(
                &[
                    Vec2::new(0.0, 0.0),
                    Vec2::new(5.0, 0.0),
                    Vec2::new(5.0, 5.0),
                    Vec2::new(0.0, 5.0),
                ],
                -1.0,
                1.0,
            ),
        ] {
            let h = m.health();
            assert!(h.is_watertight_manifold(), "{}", h.report());
            assert!(h.signed_volume_mm3 > 0.0, "prism wound inside-out");
        }
    }

    #[test]
    fn convex_solid_agrees_with_the_box_sdf_sign() {
        let (min, max) = (Vec3::splat(-2.0), Vec3::splat(3.0));
        let solid = ConvexSolid::from_convex_mesh(&box_mesh(min, max));
        assert_eq!(solid.planes.len(), 6);
        for p in [
            Vec3::ZERO,
            Vec3::splat(2.9),
            Vec3::splat(3.1),
            Vec3::new(0.0, 0.0, 10.0),
        ] {
            assert_eq!(solid.contains(p), box_sdf(p, min, max) <= 0.0, "at {p:?}");
        }
    }
}

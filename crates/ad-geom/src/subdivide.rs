//! Midpoint subdivision, for mapping per-cell wall quantities back onto the
//! surface.
//!
//! Wall shear, pressure and heat flux come out of the solver on the lattice. To
//! draw them on the duct they have to be sampled per surface vertex, and a
//! vertex can only carry one value, so any triangle much larger than a cell will
//! show the field as flat facets. Subdividing until every edge is at most one
//! cell long fixes that.
//!
//! Two properties are non-negotiable:
//!
//! * **The surface does not move.** New vertices are edge midpoints, which lie
//!   exactly on the original surface. No smoothing, no projection: a subdivided
//!   duct must voxelise to exactly the same solid mask as the original, or the
//!   two views of the same part disagree.
//! * **No cracks.** Whether an edge is split is decided per *edge*, not per
//!   triangle, so two triangles sharing an edge always agree. A triangle with
//!   only one or two split edges is retriangulated to match (the "red-green"
//!   scheme), which is what keeps the mesh watertight without refining
//!   everything to the finest level.

use crate::mesh::TriMesh;
use glam::Vec3;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Subdivided {
    pub mesh: TriMesh,
    /// For each triangle of `mesh`, the index of the triangle it came from in
    /// the input. Lets a value computed per output vertex be attributed back to
    /// the face the user actually sees.
    pub source: Vec<u32>,
    /// How many refinement rounds ran.
    pub levels: u32,
    /// True if the target edge length was reached everywhere. False means a cap
    /// stopped the refinement early.
    pub converged: bool,
    /// Longest edge in the result, mm.
    pub max_edge_mm: f32,
}

/// Subdivide until no edge is longer than `max_edge_mm`.
///
/// `max_triangles` and `max_levels` are guards, not goals: a mesh with one
/// enormous triangle and a target of 0.3 mm would otherwise grow without bound
/// and take the UI thread with it.
pub fn subdivide_to_edge_length(
    mesh: &TriMesh,
    max_edge_mm: f32,
    max_triangles: usize,
    max_levels: u32,
) -> Subdivided {
    let mut positions = mesh.positions.clone();
    let mut indices = mesh.indices.clone();
    let mut source: Vec<u32> = (0..mesh.triangle_count() as u32).collect();
    let target = max_edge_mm.max(1e-6);
    let target2 = target * target;

    let mut levels = 0;
    let mut converged = false;
    while levels < max_levels {
        let vertices_before = positions.len();
        // Decide per undirected edge, so neighbours cannot disagree.
        let mut midpoint: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in &indices {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                if midpoint.contains_key(&key) {
                    continue;
                }
                let (pa, pb) = (positions[a as usize], positions[b as usize]);
                if (pb - pa).length_squared() > target2 {
                    let idx = positions.len() as u32;
                    positions.push((pa + pb) * 0.5);
                    midpoint.insert(key, idx);
                }
            }
        }
        if midpoint.is_empty() {
            converged = true;
            break;
        }
        // A split at most quadruples the triangle count; stop before blowing the
        // budget rather than after, and drop the midpoints nothing will use.
        if indices.len() * 4 > max_triangles {
            positions.truncate(vertices_before);
            break;
        }

        let mid = |a: u32, b: u32| -> Option<u32> {
            midpoint.get(&if a < b { (a, b) } else { (b, a) }).copied()
        };
        let mut out = Vec::with_capacity(indices.len() * 2);
        let mut out_src = Vec::with_capacity(indices.len() * 2);
        for (t, tri) in indices.iter().enumerate() {
            let [v0, v1, v2] = *tri;
            let m = [mid(v0, v1), mid(v1, v2), mid(v2, v0)];
            let src = source[t];
            let mut emit = |a: u32, b: u32, c: u32| {
                out.push([a, b, c]);
                out_src.push(src);
            };
            match m {
                [None, None, None] => emit(v0, v1, v2),
                // One split edge: bisect toward the opposite corner.
                [Some(m0), None, None] => {
                    emit(v0, m0, v2);
                    emit(m0, v1, v2);
                }
                [None, Some(m1), None] => {
                    emit(v0, v1, m1);
                    emit(v0, m1, v2);
                }
                [None, None, Some(m2)] => {
                    emit(v0, v1, m2);
                    emit(m2, v1, v2);
                }
                // Two split edges: one triangle on the shared corner, and the
                // remaining quad fanned across its shorter diagonal so the
                // result does not degenerate into slivers.
                [Some(m0), Some(m1), None] => {
                    emit(m0, v1, m1);
                    quad(&mut emit, &positions, v0, m0, m1, v2);
                }
                [None, Some(m1), Some(m2)] => {
                    emit(m1, v2, m2);
                    quad(&mut emit, &positions, v0, v1, m1, m2);
                }
                [Some(m0), None, Some(m2)] => {
                    emit(v0, m0, m2);
                    quad(&mut emit, &positions, m0, v1, v2, m2);
                }
                [Some(m0), Some(m1), Some(m2)] => {
                    emit(v0, m0, m2);
                    emit(m0, v1, m1);
                    emit(m2, m1, v2);
                    emit(m0, m1, m2);
                }
            }
        }
        indices = out;
        source = out_src;
        levels += 1;
    }

    let mesh_out = TriMesh::new(positions, indices);
    let max_edge = longest_edge(&mesh_out);
    Subdivided {
        converged: converged || max_edge <= target,
        max_edge_mm: max_edge,
        mesh: mesh_out,
        source,
        levels,
    }
}

/// Split a planar quad `a-b-c-d` (in order) along its shorter diagonal.
fn quad(emit: &mut impl FnMut(u32, u32, u32), positions: &[Vec3], a: u32, b: u32, c: u32, d: u32) {
    let p = |i: u32| positions[i as usize];
    if (p(a) - p(c)).length_squared() <= (p(b) - p(d)).length_squared() {
        emit(a, b, c);
        emit(a, c, d);
    } else {
        emit(a, b, d);
        emit(b, c, d);
    }
}

pub fn longest_edge(mesh: &TriMesh) -> f32 {
    let mut m = 0.0f32;
    for t in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.triangle(t);
        m = m
            .max((b - a).length())
            .max((c - b).length())
            .max((a - c).length());
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn subdivision_reaches_the_target_edge_length() {
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        let s = subdivide_to_edge_length(&m, 1.0, 4_000_000, 16);
        assert!(
            s.converged,
            "did not converge: longest edge {}",
            s.max_edge_mm
        );
        assert!(
            s.max_edge_mm <= 1.0 + 1e-5,
            "longest edge {}",
            s.max_edge_mm
        );
        assert!(s.mesh.triangle_count() > m.triangle_count());
    }

    /// The property the whole module exists to guarantee: the surface is
    /// unchanged. Same volume, same area, still closed.
    #[test]
    fn subdivision_preserves_the_surface_exactly() {
        for m in [
            primitives::box_mesh(Vec3::new(-2.0, 1.0, 0.0), Vec3::new(8.0, 6.0, 13.0)),
            primitives::uv_sphere(Vec3::ZERO, 8.0, 24, 12),
            primitives::sharp_wedge(20.0, 30.0, 8.0),
        ] {
            let before = m.health();
            let s = subdivide_to_edge_length(&m, 1.0, 4_000_000, 16);
            let after = s.mesh.health();

            assert!(
                after.is_watertight_manifold(),
                "cracked: {}",
                after.report()
            );
            assert!(
                (after.signed_volume_mm3 - before.signed_volume_mm3).abs()
                    / before.signed_volume_mm3.abs()
                    < 1e-4,
                "volume changed: {} -> {}",
                before.signed_volume_mm3,
                after.signed_volume_mm3
            );
            assert!(
                (after.surface_area_mm2 - before.surface_area_mm2).abs() / before.surface_area_mm2
                    < 1e-4,
                "area changed: {} -> {}",
                before.surface_area_mm2,
                after.surface_area_mm2
            );
            // A closed surface keeps its Euler characteristic under refinement.
            assert_eq!(
                after.topology.euler_characteristic,
                before.topology.euler_characteristic
            );
        }
    }

    #[test]
    fn every_output_triangle_maps_back_to_an_input_triangle() {
        let m = primitives::uv_sphere(Vec3::ZERO, 10.0, 16, 8);
        let s = subdivide_to_edge_length(&m, 1.5, 4_000_000, 16);
        assert_eq!(s.source.len(), s.mesh.triangle_count());
        assert!(s.source.iter().all(|i| (*i as usize) < m.triangle_count()));

        // Every original triangle must still be represented, and the children of
        // one parent must cover exactly the parent's area.
        let mut area = vec![0.0f64; m.triangle_count()];
        for t in 0..s.mesh.triangle_count() {
            area[s.source[t] as usize] += s.mesh.face_normal_raw(t).length() as f64 * 0.5;
        }
        for t in 0..m.triangle_count() {
            let want = m.face_normal_raw(t).length() as f64 * 0.5;
            assert!(
                (area[t] - want).abs() / want < 1e-4,
                "triangle {t}: children cover {} of {want}",
                area[t]
            );
        }
    }

    #[test]
    fn children_keep_the_parent_winding() {
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(4.0));
        let s = subdivide_to_edge_length(&m, 0.9, 1_000_000, 16);
        for t in 0..s.mesh.triangle_count() {
            let parent = m.face_normal(s.source[t] as usize);
            let child = s.mesh.face_normal(t);
            assert!(
                child.dot(parent) > 0.99,
                "triangle {t} flipped: {child:?} vs {parent:?}"
            );
        }
    }

    #[test]
    fn an_already_fine_mesh_is_returned_untouched() {
        let m = primitives::uv_sphere(Vec3::ZERO, 2.0, 8, 4);
        let long = longest_edge(&m);
        let s = subdivide_to_edge_length(&m, long * 2.0, 1_000_000, 16);
        assert_eq!(s.levels, 0);
        assert!(s.converged);
        assert_eq!(s.mesh.triangle_count(), m.triangle_count());
    }

    #[test]
    fn the_triangle_budget_stops_runaway_refinement() {
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(1000.0));
        let s = subdivide_to_edge_length(&m, 0.01, 50_000, 32);
        assert!(!s.converged);
        assert!(
            s.mesh.triangle_count() <= 50_000,
            "{} triangles",
            s.mesh.triangle_count()
        );
        // Even a truncated refinement must not tear the surface.
        assert!(s.mesh.topology().is_watertight_manifold());
    }
}

//! Automatic detection of duct mouths.
//!
//! A printed duct is normally exported with its openings flush against the
//! bounding box, because that is how it was cut in CAD. So a mouth shows up as
//! a flat *rim*: an annulus of triangles lying in one of the six bounding-box
//! planes, with a hole in the middle. The hole is the opening, and its boundary
//! loop is what we want.
//!
//! The detection is therefore topological rather than geometric. Collect the
//! triangles coplanar with a box face, orient them consistently, extract the
//! boundary loops of that patch, and read off the winding: with every triangle
//! wound counter-clockwise in the plane, an outer boundary comes out
//! anticlockwise (positive area) and a hole comes out clockwise (negative).
//! Every negative loop is a mouth. That handles several mouths on one face, and
//! a mouth with an island in the middle of it, without any special cases.

use crate::mesh::TriMesh;
use crate::scene::{MeshRole, Scene};
use ad_gpu::{Bbox, FlowPatch};
use glam::{Vec2, Vec3};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct MouthConfig {
    /// How far a vertex may sit from the box plane and still count as being in
    /// it. Exporters round coordinates, so this cannot be zero.
    pub plane_tol_mm: f32,
    /// A face normal must be at least this parallel to the box face's normal.
    /// Keeps a near-tangent sliver from joining the rim.
    pub normal_tol: f32,
    /// Holes smaller than this are screw clearances and drain slots, not
    /// mouths.
    pub min_area_mm2: f32,
}

impl Default for MouthConfig {
    fn default() -> Self {
        Self {
            plane_tol_mm: 0.05,
            normal_tol: 0.99,
            min_area_mm2: 4.0,
        }
    }
}

/// One detected opening.
#[derive(Debug, Clone)]
pub struct Mouth {
    /// The patch, with `normal` pointing **into** the fluid domain, which is
    /// what an inlet needs.
    pub patch: FlowPatch,
    /// True open area of the hole, mm^2. This is the number to use for a
    /// velocity-to-flow-rate conversion; `patch.area_mm2()` is the area of the
    /// bounding rectangle, which is larger for anything not rectangular.
    pub open_area_mm2: f32,
    /// Which box face it was found on: axis 0/1/2 and whether it was the min or
    /// the max side.
    pub axis: u8,
    pub on_min_side: bool,
    /// The hole's boundary, in order, in world space.
    pub boundary: Vec<Vec3>,
}

impl Mouth {
    /// Perimeter of the opening, mm.
    pub fn perimeter_mm(&self) -> f32 {
        let n = self.boundary.len();
        (0..n)
            .map(|i| (self.boundary[(i + 1) % n] - self.boundary[i]).length())
            .sum()
    }

    /// Hydraulic diameter, `4A / P`. The Reynolds numbers in the contract are
    /// quoted against this.
    pub fn hydraulic_diameter_mm(&self) -> f32 {
        let p = self.perimeter_mm();
        if p <= 0.0 {
            0.0
        } else {
            4.0 * self.open_area_mm2 / p
        }
    }
}

/// In-plane axes for box face `axis`/`side`, ordered so `u x v` is the box's
/// *outward* normal. That ordering is what makes an outer boundary come out
/// positive and a hole negative.
fn plane_axes(axis: usize, on_min_side: bool) -> (Vec3, Vec3, Vec3) {
    let e = [Vec3::X, Vec3::Y, Vec3::Z];
    let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
    // e_{a+1} x e_{a+2} = e_a for any axis, so this pair is right-handed about
    // +axis; swap them for the min face, whose outward normal is -axis.
    if on_min_side {
        (e[j], e[i], -e[axis])
    } else {
        (e[i], e[j], e[axis])
    }
}

/// Find every mouth on the six faces of `bbox`.
pub fn detect_mouths(mesh: &TriMesh, bbox: Bbox, cfg: MouthConfig) -> Vec<Mouth> {
    let mut out = Vec::new();
    if bbox.is_empty() {
        return out;
    }
    for axis in 0..3usize {
        for on_min_side in [true, false] {
            let w = if on_min_side {
                bbox.min[axis]
            } else {
                bbox.max[axis]
            };
            out.extend(detect_on_plane(mesh, axis, on_min_side, w, cfg));
        }
    }
    out
}

/// Convenience: every mouth of every visible duct in the scene, in world space.
///
/// Against the ducts' own box, not the scene's: an obstruction that sticks out
/// past a duct face would otherwise move that face off the mouth, and the duct
/// would have none.
pub fn detect_in_scene(scene: &Scene, cfg: MouthConfig) -> Vec<Mouth> {
    let bbox = scene.bbox_of_role(MeshRole::Duct);
    let mut out = Vec::new();
    for (_, inst) in scene.visible() {
        if inst.role != MeshRole::Duct {
            continue;
        }
        out.extend(detect_mouths(&inst.world_mesh(), bbox, cfg));
    }
    out
}

fn detect_on_plane(
    mesh: &TriMesh,
    axis: usize,
    on_min_side: bool,
    w: f32,
    cfg: MouthConfig,
) -> Vec<Mouth> {
    let (u_axis, v_axis, outward) = plane_axes(axis, on_min_side);

    // Triangles lying in the plane, each re-wound counter-clockwise in (u, v).
    let mut oriented: Vec<[u32; 3]> = Vec::new();
    for t in 0..mesh.triangle_count() {
        let n = mesh.face_normal(t);
        if n.dot(outward).abs() < cfg.normal_tol {
            continue;
        }
        let tri = mesh.triangle(t);
        if tri.iter().any(|p| (p[axis] - w).abs() > cfg.plane_tol_mm) {
            continue;
        }
        let i = mesh.indices[t];
        let p: Vec<Vec2> = tri
            .iter()
            .map(|q| Vec2::new(q.dot(u_axis), q.dot(v_axis)))
            .collect();
        let area2 = (p[1] - p[0]).perp_dot(p[2] - p[0]);
        if area2 == 0.0 {
            continue;
        }
        oriented.push(if area2 > 0.0 { i } else { [i[0], i[2], i[1]] });
    }
    if oriented.is_empty() {
        return Vec::new();
    }

    // Boundary of the patch: directed edges with no matching reverse. Counting
    // rather than a set membership test, so a rim that touches itself at a
    // point still produces the right number of loops.
    let mut counts: HashMap<(u32, u32), i32> = HashMap::new();
    for t in &oriented {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *counts.entry((a, b)).or_insert(0) += 1;
        }
    }
    let mut next: HashMap<u32, Vec<u32>> = HashMap::new();
    for ((a, b), n) in &counts {
        let back = counts.get(&(*b, *a)).copied().unwrap_or(0);
        for _ in 0..(n - back).max(0) {
            next.entry(*a).or_default().push(*b);
        }
    }

    let mut mouths = Vec::new();
    let starts: Vec<u32> = next.keys().copied().collect();
    for start in starts {
        while next.get(&start).is_some_and(|v| !v.is_empty()) {
            let Some(loop_ix) = walk_loop(start, &mut next) else {
                continue;
            };
            if loop_ix.len() < 3 {
                continue;
            }
            let pts2: Vec<Vec2> = loop_ix
                .iter()
                .map(|i| {
                    let p = mesh.positions[*i as usize];
                    Vec2::new(p.dot(u_axis), p.dot(v_axis))
                })
                .collect();
            let (area, centroid) = polygon_area_and_centroid(&pts2);

            // Positive area is the outer edge of the rim; only the clockwise
            // loops are holes, and a hole in a rim is a mouth.
            if area >= 0.0 || -area < cfg.min_area_mm2 {
                continue;
            }
            let open_area = -area;

            let lo = pts2
                .iter()
                .fold(Vec2::splat(f32::INFINITY), |a, b| a.min(*b));
            let hi = pts2
                .iter()
                .fold(Vec2::splat(f32::NEG_INFINITY), |a, b| a.max(*b));
            let half = (hi - lo) * 0.5;
            let plane_point = |c: Vec2| u_axis * c.x + v_axis * c.y + outward.abs() * w;

            mouths.push(Mouth {
                patch: FlowPatch {
                    center_mm: plane_point(centroid),
                    // Into the fluid: away from the box face, i.e. the opposite
                    // of the face's outward normal.
                    normal: -outward,
                    half_u: u_axis * half.x,
                    half_v: v_axis * half.y,
                },
                open_area_mm2: open_area,
                axis: axis as u8,
                on_min_side,
                boundary: loop_ix
                    .iter()
                    .map(|i| mesh.positions[*i as usize])
                    .collect(),
            });
        }
    }
    mouths
}

/// Follow directed boundary edges from `start` until they return to it,
/// consuming each edge as it is used.
fn walk_loop(start: u32, next: &mut HashMap<u32, Vec<u32>>) -> Option<Vec<u32>> {
    let mut out = vec![start];
    let mut here = start;
    // A loop cannot be longer than the number of remaining edges.
    let budget: usize = next.values().map(|v| v.len()).sum();
    for _ in 0..budget {
        let to = next.get_mut(&here)?.pop()?;
        if next.get(&here).is_some_and(|v| v.is_empty()) {
            next.remove(&here);
        }
        if to == start {
            return Some(out);
        }
        out.push(to);
        here = to;
    }
    None
}

/// Shoelace area and area centroid. Positive for a counter-clockwise polygon.
fn polygon_area_and_centroid(p: &[Vec2]) -> (f32, Vec2) {
    let n = p.len();
    let mut a2 = 0.0f64;
    let mut c = glam::DVec2::ZERO;
    for i in 0..n {
        let (q, r) = (p[i].as_dvec2(), p[(i + 1) % n].as_dvec2());
        let cross = q.x * r.y - r.x * q.y;
        a2 += cross;
        c += (q + r) * cross;
    }
    let area = a2 * 0.5;
    if a2.abs() < 1e-12 {
        return (0.0, p[0]);
    }
    ((area) as f32, (c / (3.0 * a2)).as_vec2())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;
    use glam::Vec3;

    /// A rectangular tube: a box with a rectangular hole through it in Z, so
    /// both ends are mouths flush with the bounding box.
    fn tube(outer: Vec2, inner: Vec2, length: f32) -> TriMesh {
        let (ho, hi) = (outer * 0.5, inner * 0.5);
        let mut positions = Vec::new();
        // 0..4 outer ring at z=0, 4..8 inner at z=0, then the same at z=length.
        for z in [0.0, length] {
            for h in [ho, hi] {
                for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    positions.push(Vec3::new(h.x * sx, h.y * sy, z));
                }
            }
        }
        let o0 = 0u32;
        let i0 = 4;
        let o1 = 8;
        let i1 = 12;
        let mut indices = Vec::new();
        let mut quad = |a: u32, b: u32, c: u32, d: u32| {
            indices.push([a, b, c]);
            indices.push([a, c, d]);
        };
        for k in 0..4u32 {
            let (k0, k1) = (k, (k + 1) % 4);
            // Rim at z=0, outward normal -Z: wound clockwise seen from +Z.
            quad(o0 + k0, i0 + k0, i0 + k1, o0 + k1);
            // Rim at z=length, outward +Z.
            quad(o1 + k0, o1 + k1, i1 + k1, i1 + k0);
            // Outer wall, normal pointing away from the axis.
            quad(o0 + k0, o0 + k1, o1 + k1, o1 + k0);
            // Inner wall, normal pointing toward the axis.
            quad(i0 + k0, i1 + k0, i1 + k1, i0 + k1);
        }
        TriMesh::new(positions, indices)
    }

    #[test]
    fn the_test_tube_is_a_watertight_genus_one_solid() {
        let m = tube(Vec2::new(40.0, 20.0), Vec2::new(30.0, 10.0), 50.0);
        let h = m.health();
        assert!(h.is_watertight_manifold(), "{}", h.report());
        assert_eq!(h.topology.genus(), Some(1), "a tube is a torus");
        assert!(
            h.signed_volume_mm3 > 0.0,
            "wound inside-out: {}",
            h.signed_volume_mm3
        );
    }

    #[test]
    fn both_ends_of_a_tube_are_found_with_the_right_area_and_normal() {
        let m = tube(Vec2::new(40.0, 20.0), Vec2::new(30.0, 10.0), 50.0);
        let mut found = detect_mouths(&m, m.bbox(), MouthConfig::default());
        assert_eq!(
            found.len(),
            2,
            "expected exactly two mouths, got {}",
            found.len()
        );
        found.sort_by(|a, b| {
            a.patch
                .center_mm
                .z
                .partial_cmp(&b.patch.center_mm.z)
                .unwrap()
        });

        let want_area = 30.0 * 10.0;
        for (mouth, sign) in found.iter().zip([1.0f32, -1.0]) {
            assert!(
                (mouth.open_area_mm2 - want_area).abs() / want_area < 0.001,
                "area {} vs {want_area}",
                mouth.open_area_mm2
            );
            // The normal must point into the duct, not out of the box.
            assert!(
                (mouth.patch.normal - Vec3::new(0.0, 0.0, sign)).length() < 1e-5,
                "normal {:?} does not point into the fluid",
                mouth.patch.normal
            );
            assert!((mouth.patch.center_mm.x).abs() < 1e-4);
            assert!((mouth.patch.center_mm.y).abs() < 1e-4);
            // The bounding rectangle of a rectangular hole is the hole.
            assert!((mouth.patch.area_mm2() - want_area).abs() / want_area < 0.001);
            // 4A/P for a 30x10 rectangle is 15.
            assert!((mouth.hydraulic_diameter_mm() - 15.0).abs() < 1e-3);
        }
        assert!((found[0].patch.center_mm.z - 0.0).abs() < 1e-4);
        assert!((found[1].patch.center_mm.z - 50.0).abs() < 1e-4);
    }

    #[test]
    fn a_solid_box_has_no_mouths() {
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(20.0));
        assert!(detect_mouths(&m, m.bbox(), MouthConfig::default()).is_empty());
    }

    #[test]
    fn an_obstruction_past_the_duct_does_not_hide_its_mouths() {
        use crate::scene::{MeshAsset, Transform};
        let mut s = Scene::new();
        let duct = tube(Vec2::new(40.0, 20.0), Vec2::new(30.0, 10.0), 50.0);
        s.add(
            "duct",
            MeshAsset::new(duct),
            Transform::IDENTITY,
            MeshRole::Duct,
        );
        // Beside the tube and longer than it, so it overhangs both mouth faces.
        let vane = primitives::box_mesh(Vec3::new(25.0, -5.0, -20.0), Vec3::new(35.0, 5.0, 70.0));
        s.add(
            "vane",
            MeshAsset::new(vane),
            Transform::IDENTITY,
            MeshRole::Obstruction,
        );
        assert_eq!(detect_in_scene(&s, MouthConfig::default()).len(), 2);
    }

    #[test]
    fn tiny_holes_are_filtered_out() {
        let m = tube(Vec2::new(40.0, 20.0), Vec2::new(1.0, 1.0), 50.0);
        let cfg = MouthConfig {
            min_area_mm2: 4.0,
            ..Default::default()
        };
        assert!(
            detect_mouths(&m, m.bbox(), cfg).is_empty(),
            "a 1 mm^2 hole is not a mouth"
        );
        let cfg = MouthConfig {
            min_area_mm2: 0.1,
            ..Default::default()
        };
        assert_eq!(detect_mouths(&m, m.bbox(), cfg).len(), 2);
    }

    #[test]
    fn a_mouth_off_the_bbox_plane_is_not_detected() {
        // Shift the tube so neither end sits on the scene box: the openings are
        // then interior features and must not be reported as mouths.
        let m = tube(Vec2::new(40.0, 20.0), Vec2::new(30.0, 10.0), 50.0);
        let bbox = m.bbox().expanded(Vec3::splat(5.0));
        assert!(detect_mouths(&m, bbox, MouthConfig::default()).is_empty());
    }

    #[test]
    fn mouths_are_found_on_every_axis() {
        // Same tube, rotated so its axis lies along X and then along Y. The
        // detector must not be biased toward any particular plane.
        for axis in 0..3usize {
            let base = tube(Vec2::new(40.0, 20.0), Vec2::new(30.0, 10.0), 50.0);
            let mut m = base.clone();
            for p in m.positions.iter_mut() {
                *p = match axis {
                    0 => Vec3::new(p.z, p.x, p.y),
                    1 => Vec3::new(p.y, p.z, p.x),
                    _ => *p,
                };
            }
            let found = detect_mouths(&m, m.bbox(), MouthConfig::default());
            assert_eq!(found.len(), 2, "axis {axis} found {} mouths", found.len());
            for mouth in &found {
                assert_eq!(mouth.axis as usize, axis);
                assert!((mouth.open_area_mm2 - 300.0).abs() < 0.5);
            }
        }
    }

    #[test]
    fn polygon_area_sign_follows_the_winding() {
        let ccw = [
            Vec2::ZERO,
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 3.0),
            Vec2::new(0.0, 3.0),
        ];
        let (a, c) = polygon_area_and_centroid(&ccw);
        assert!((a - 6.0).abs() < 1e-5);
        assert!((c - Vec2::new(1.0, 1.5)).length() < 1e-5);

        let mut cw = ccw;
        cw.reverse();
        let (a, _) = polygon_area_and_centroid(&cw);
        assert!((a + 6.0).abs() < 1e-5);
    }
}

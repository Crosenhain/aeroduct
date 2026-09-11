//! Validation: the sign of the distance field.
//!
//! Everything downstream of the voxeliser depends on one boolean per cell being
//! right, and the only place it can go wrong is where a cell's closest surface
//! point lands on an edge or a vertex rather than in the middle of a face. On a
//! printed duct that is a large fraction of the near-wall cells, because the
//! lips and the internal fillets are sharp.
//!
//! These tests need no GPU: they exercise `ad_geom::CpuSdf`, which is the same
//! logic the shader runs and the reference the GPU field is compared against in
//! `gpu_voxelize.rs`.

use ad_geom::primitives::{self, ConvexSolid};
use ad_geom::CpuSdf;
use glam::Vec3;

/// The whole justification for the pseudonormal machinery, as an executable
/// claim.
///
/// A wedge with an interior dihedral angle below 90 degrees has an exterior
/// region — points just outside, nearly directly above one of the two faces —
/// where the closest point is the shared apex edge and *both* faces are exactly
/// equidistant. Whichever face the search happens to return, the naive
/// `dot(p - closest, face_normal)` test is negative for points on the far side
/// of that face's plane, so those cells come out solid when they are plainly
/// fluid.
///
/// The angle-weighted pseudonormal at the edge is `n1 + n2`, which for this
/// wedge points straight out along the bisector and is positive across the whole
/// exterior cone.
///
/// The test asserts both halves: the pseudonormal is right everywhere, and the
/// naive alternative is demonstrably wrong here. Without the second assertion
/// this test would still pass if someone quietly swapped the implementation for
/// the naive one on a shape where both happen to agree.
#[test]
fn the_naive_face_normal_sign_test_fails_on_a_sharp_edge() {
    let apex_deg = 25.0f32;
    let mesh = primitives::sharp_wedge(apex_deg, 30.0, 12.0);
    let solid = ConvexSolid::from_convex_mesh(&mesh);
    let sdf = CpuSdf::from_mesh(&mesh);

    // The exterior normal cone at the apex edge spans the angle between the two
    // face normals, which is 180 - apex.
    let half_cone = (180.0 - apex_deg).to_radians() * 0.5;

    let mut pseudonormal_wrong = 0;
    let mut naive_wrong = 0;
    let mut sampled = 0;

    for i in 0..=40 {
        // Sweep the exterior cone around the apex bisector, which points along
        // -X for this wedge.
        let a = -half_cone + 2.0 * half_cone * i as f32 / 40.0;
        let dir = Vec3::new(-a.cos(), a.sin(), 0.0);
        for step in 1..=12 {
            let t = 0.05 * step as f32;
            for z in [-3.0f32, 0.0, 3.0] {
                let p = dir * t + Vec3::new(0.0, 0.0, z);
                let truth_inside = solid.contains(p);
                assert!(!truth_inside, "the sample points must all be outside the wedge");
                sampled += 1;
                if sdf.signed_distance(p) < 0.0 {
                    pseudonormal_wrong += 1;
                }
                if sdf.naive_signed_distance(p) < 0.0 {
                    naive_wrong += 1;
                }
            }
        }
    }

    assert!(sampled > 500, "only {sampled} samples");
    assert_eq!(
        pseudonormal_wrong, 0,
        "the angle-weighted pseudonormal misclassified {pseudonormal_wrong} of {sampled} \
         exterior points near a {apex_deg} degree edge"
    );
    assert!(
        naive_wrong > sampled / 10,
        "the naive nearest-face-normal test only failed {naive_wrong} of {sampled} times; if it \
         now agrees everywhere, this test has stopped proving anything and needs a sharper wedge"
    );
}

/// The same claim on a shape with an interior sharp edge rather than an exterior
/// one: two boxes meeting at a shallow angle produce a reflex crease, and points
/// *inside* the material near it must stay inside.
#[test]
fn signs_are_correct_on_both_sides_of_a_cube() {
    let (min, max) = (Vec3::new(-5.0, -7.0, -3.0), Vec3::new(6.0, 4.0, 9.0));
    let sdf = CpuSdf::from_mesh(&primitives::box_mesh(min, max));

    // Walk a shell of points around every edge and corner at a range of offsets,
    // including diagonal offsets that land squarely in a corner's normal cone.
    let mut checked = 0;
    for corner in 0..8 {
        let c = Vec3::new(
            if corner & 1 == 0 { min.x } else { max.x },
            if corner & 2 == 0 { min.y } else { max.y },
            if corner & 4 == 0 { min.z } else { max.z },
        );
        let away = Vec3::new(
            if corner & 1 == 0 { -1.0 } else { 1.0 },
            if corner & 2 == 0 { -1.0 } else { 1.0 },
            if corner & 4 == 0 { -1.0 } else { 1.0 },
        );
        for d in [0.01f32, 0.05, 0.2, 0.7, 1.9] {
            for mask in 1..8u32 {
                let off = Vec3::new(
                    if mask & 1 != 0 { away.x * d } else { 0.0 },
                    if mask & 2 != 0 { away.y * d } else { 0.0 },
                    if mask & 4 != 0 { away.z * d } else { 0.0 },
                );
                for sign in [1.0f32, -1.0] {
                    let p = c + off * sign;
                    let want = primitives::box_sdf(p, min, max);
                    let got = sdf.signed_distance(p);
                    assert!(
                        (got - want).abs() < 1e-3,
                        "at {p:?} near corner {c:?}: got {got}, analytic {want}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 500, "only {checked} points around the corners");
}

/// The pseudonormal must be a property of the *feature*, not of whichever
/// triangle the search returned. If it were not, the tie at a shared edge would
/// make the sign depend on the order triangles happen to be stored in.
#[test]
fn the_sign_does_not_depend_on_triangle_order() {
    let mesh = primitives::sharp_wedge(25.0, 30.0, 12.0);
    let mut reversed = mesh.clone();
    reversed.indices.reverse();

    let a = CpuSdf::from_mesh(&mesh);
    let b = CpuSdf::from_mesh(&reversed);

    let mut seed = 11u32;
    let mut rnd = |lo: f32, hi: f32| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        lo + (seed >> 8) as f32 / (1 << 24) as f32 * (hi - lo)
    };
    for _ in 0..4000 {
        let p = Vec3::new(rnd(-4.0, 34.0), rnd(-10.0, 10.0), rnd(-9.0, 9.0));
        let (da, db) = (a.signed_distance(p), b.signed_distance(p));
        assert_eq!(
            da < 0.0,
            db < 0.0,
            "reordering the triangles changed the sign at {p:?}: {da} vs {db}"
        );
        assert!((da.abs() - db.abs()).abs() < 1e-4);
    }
}

/// A missing triangle has to be *reported*, not quietly voxelised into something
/// that looks plausible. A leak is the most common reason a simulation of a real
/// part behaves strangely, and it is cheap to detect.
#[test]
fn a_hole_in_the_mesh_is_detected_by_ray_parity() {
    let mut mesh = primitives::uv_sphere(Vec3::new(2.0, -1.0, 0.0), 9.0, 40, 20);
    assert!(mesh.topology().is_watertight_manifold());

    let grid = ad_gpu::Grid::covering(mesh.bbox().expanded(Vec3::splat(2.0)), 0.5);
    let soup = |m: &ad_geom::TriMesh| -> Vec<[Vec3; 3]> {
        (0..m.triangle_count()).map(|t| m.triangle(t)).collect()
    };

    let intact = ad_geom::ray_parity_voxelize(&soup(&mesh), grid);
    assert!(intact.is_watertight(), "the intact sphere should be clean: {}", intact.report());

    // Knock out one triangle near the equator, where many rows pass through it.
    let victim = mesh.triangle_count() / 2;
    mesh.indices.remove(victim);

    let health = mesh.health();
    assert!(!health.is_watertight_manifold());
    assert_eq!(health.topology.boundary_edges, 3, "{}", health.report());

    let leaky = ad_geom::ray_parity_voxelize(&soup(&mesh), grid);
    assert!(!leaky.is_watertight(), "the hole went unnoticed: {}", leaky.report());
    assert!(leaky.odd_parity_rows > 0);
    assert!(!leaky.odd_parity_examples.is_empty(), "no location reported for the leak");
    assert!(leaky.report().contains("odd parity"));
}

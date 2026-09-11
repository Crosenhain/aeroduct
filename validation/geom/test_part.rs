//! Validation against the real part, `Airflow redirector - Part 1.stl`.
//!
//! The file lives outside the repository, so every test here skips when it is
//! not found; set `AERODUCT_TEST_STL` to point at it. The numbers asserted are
//! the ones in `CONTRACT.md`: 85,180 triangles, every edge shared by exactly two
//! faces, Euler characteristic 0 (genus 1), 55,805 mm^3, and two mouths of 2,116
//! and 1,141 mm^2.

use ad_geom::scene::FlatGeometry;
use ad_geom::{MouthConfig, TriMesh, Voxelizer};
use ad_gpu::{flags, Bbox, Grid};
use glam::Vec3;

fn part() -> Option<ad_geom::StlLoad> {
    let path = ad_geom::test_stl_path()?;
    match ad_geom::load_stl(&path) {
        Ok(l) => Some(l),
        Err(e) => panic!(
            "the test part exists at {} but would not load: {e:#}",
            path.display()
        ),
    }
}

fn soup(m: &TriMesh) -> Vec<[Vec3; 3]> {
    (0..m.triangle_count()).map(|t| m.triangle(t)).collect()
}

#[test]
fn the_part_is_a_watertight_genus_one_solid() {
    let Some(load) = part() else { return };
    eprintln!("{}", load.summary());
    let h = load.health();
    eprintln!("{}", h.report());

    assert_eq!(h.triangle_count, 85_180);
    assert_eq!(load.format, ad_geom::StlFormat::Binary);

    // Every edge shared by exactly two consistently wound triangles.
    assert!(h.is_watertight_manifold(), "{}", h.report());
    assert_eq!(h.topology.valence_histogram.get(1).copied().unwrap_or(0), 0);
    assert_eq!(h.topology.valence_histogram[2], h.topology.edge_count);
    assert_eq!(h.topology.euler_characteristic, 0);
    assert_eq!(h.topology.genus(), Some(1));

    // Welding must have found the shared vertices: V - E + F = 0 with F = 85180
    // forces E = 3F/2 and V = E - F.
    assert_eq!(h.topology.edge_count, 85_180 * 3 / 2);
    assert_eq!(h.vertex_count, h.topology.edge_count - 85_180);

    // Divergence-theorem volume, positive because the winding is outward.
    assert!(
        (h.signed_volume_mm3 / 55_805.0 - 1.0).abs() < 0.01,
        "volume {} mm^3, expected 55805",
        h.signed_volume_mm3
    );

    let size = h.bbox.size();
    for (got, want) in [(size.x, 145.0f32), (size.y, 72.185), (size.z, 68.94)] {
        assert!(
            (got - want).abs() < 0.05,
            "bbox {size:?} does not match the contract"
        );
    }
}

#[test]
fn the_part_has_exactly_the_two_documented_mouths() {
    let Some(load) = part() else { return };
    let mesh = &load.mesh;
    let mut found = ad_geom::detect_mouths(mesh, mesh.bbox(), MouthConfig::default());
    for m in &found {
        eprintln!(
            "mouth on axis {} {} plane: {:.0} mm^2, D_h {:.1} mm, centre {:?}, normal {:?}",
            m.axis,
            if m.on_min_side { "min" } else { "max" },
            m.open_area_mm2,
            m.hydraulic_diameter_mm(),
            m.patch.center_mm,
            m.patch.normal,
        );
    }
    assert_eq!(
        found.len(),
        2,
        "expected exactly two mouths, found {}",
        found.len()
    );

    found.sort_by(|a, b| b.open_area_mm2.partial_cmp(&a.open_area_mm2).unwrap());
    let bbox = mesh.bbox();

    // Mouth A: z = 0 plane, 2116 mm^2.
    let a = &found[0];
    assert_eq!(a.axis, 2, "the larger mouth should be on a Z plane");
    assert!(a.on_min_side);
    assert!(
        (a.open_area_mm2 / 2116.0 - 1.0).abs() < 0.05,
        "mouth A area {} mm^2, expected 2116 +/- 5%",
        a.open_area_mm2
    );
    assert!((a.patch.center_mm.z - bbox.min.z).abs() < 0.05);
    assert!(
        (a.patch.normal - Vec3::Z).length() < 1e-5,
        "mouth A normal {:?} must point into the fluid",
        a.patch.normal
    );

    // Mouth B: y = 0 plane, 1141 mm^2.
    let b = &found[1];
    assert_eq!(b.axis, 1, "the smaller mouth should be on a Y plane");
    assert!(b.on_min_side);
    assert!(
        (b.open_area_mm2 / 1141.0 - 1.0).abs() < 0.05,
        "mouth B area {} mm^2, expected 1141 +/- 5%",
        b.open_area_mm2
    );
    assert!((b.patch.center_mm.y - bbox.min.y).abs() < 0.05);
    assert!((b.patch.normal - Vec3::Y).length() < 1e-5);

    // The contract's ~1.85:1 area contraction.
    let ratio = a.open_area_mm2 / b.open_area_mm2;
    assert!(
        (ratio - 1.85).abs() < 0.15,
        "area ratio {ratio}, expected about 1.85"
    );
}

#[test]
fn ray_parity_finds_no_leaks_in_the_part() {
    let Some(load) = part() else { return };
    let grid = Grid::covering(load.mesh.bbox().expanded(Vec3::splat(3.0)), 1.0);
    let r = ad_geom::ray_parity_voxelize(&soup(&load.mesh), grid);
    eprintln!("{}", r.report());

    assert!(r.is_watertight(), "{}", r.report());
    assert_eq!(r.grazing_hits, 0, "{}", r.report());
    // The part is a bend, so most rows of its bounding box miss it entirely.
    // What matters is that a substantial number do and every one of them is
    // even.
    assert!(
        r.rows_with_hits > 1_000,
        "only {} rows met the part",
        r.rows_with_hits
    );
    // A 1 mm voxelisation of a thin-walled part loses a little volume to
    // partially-filled cells; 1.2% is what the CPU reference was measured at.
    assert!(
        (r.volume_mm3() / 55_805.0 - 1.0).abs() < 0.02,
        "voxel volume {:.0} mm^3 against 55805",
        r.volume_mm3()
    );
}

#[test]
fn the_gpu_mask_matches_the_cpu_reference_on_the_part() {
    let Some(load) = part() else { return };
    let Some(gpu) = ad_geom::test_gpu() else {
        return;
    };

    let grid = Grid::covering(load.mesh.bbox().expanded(Vec3::splat(3.0)), 1.0);
    let reference = ad_geom::ray_parity_voxelize(&soup(&load.mesh), grid);
    assert!(reference.is_watertight(), "{}", reference.report());

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    let stats = vox
        .voxelize(FlatGeometry::from_mesh(&load.mesh), grid)
        .expect("voxelize");
    eprintln!("{}", stats.report());
    assert!(stats.fill_converged, "{}", stats.report());

    let cell_flags = vox.read_flags().expect("flags");
    let phi = vox.read_phi().expect("phi");
    let mut disagree = 0usize;
    let mut borderline = 0usize;
    let mut first: Option<usize> = None;
    for (i, f) in cell_flags.iter().enumerate() {
        if (*f & flags::SOLID != 0) == reference.solid[i] {
            continue;
        }
        if phi[i].abs() <= 1e-3 {
            borderline += 1;
        } else {
            disagree += 1;
            first.get_or_insert(i);
        }
    }
    if let Some(i) = first {
        let c = glam::UVec3::new(
            (i as u32) % grid.dims.x,
            ((i as u32) / grid.dims.x) % grid.dims.y,
            (i as u32) / (grid.dims.x * grid.dims.y),
        );
        eprintln!("first disagreement at cell {c:?}, phi = {}", phi[i]);
    }
    eprintln!(
        "{disagree} hard disagreements, {borderline} borderline, out of {} cells",
        cell_flags.len()
    );

    // Two independent methods on a real 85k-triangle part. Anything above a
    // handful of cells means one of them is wrong, not that the part is awkward.
    assert!(
        disagree * 10_000 < cell_flags.len(),
        "{disagree} of {} cells disagreed between the GPU field and ray parity",
        cell_flags.len()
    );

    // ...and the occupied volume must land on the analytic figure.
    assert!(
        (stats.solid_volume_mm3() / 55_805.0 - 1.0).abs() < 0.02,
        "GPU volume {:.0} mm^3 against the analytic 55805",
        stats.solid_volume_mm3()
    );
}

/// The interactive requirement: a full re-voxelisation at the contract's
/// interactive tier, and an incremental one after a drag.
///
/// The timing is reported rather than asserted tightly, because a debug build,
/// a busy GPU or a different card all move it. The assertion is only there to
/// catch something pathological, like the flood fill failing to converge and
/// running its iteration cap every time.
#[test]
fn voxelising_the_part_at_the_interactive_tier_is_fast_enough() {
    let Some(load) = part() else { return };
    let Some(gpu) = ad_geom::test_gpu() else {
        return;
    };

    // The contract's domain: 260 x 180 x 180 mm around the scene, dx = 0.75.
    let b = load.mesh.bbox();
    let c = b.center();
    let domain = Bbox {
        min: c - Vec3::new(130.0, 90.0, 90.0),
        max: c + Vec3::new(130.0, 90.0, 90.0),
    };
    let grid = Grid::covering(domain, 0.75);
    eprintln!(
        "grid {}x{}x{} = {:.1} M cells",
        grid.dims.x,
        grid.dims.y,
        grid.dims.z,
        grid.cell_count() as f64 / 1e6
    );

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    let geom = FlatGeometry::from_mesh(&load.mesh);

    // First call includes shader warm-up and the initial allocations.
    let warm = vox.voxelize(geom.clone(), grid).expect("voxelize");
    eprintln!("cold: {}", warm.report());
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let s = vox.voxelize(geom.clone(), grid).expect("voxelize");
        eprintln!("warm: {}", s.report());
        best = best.min(s.total_ms);
        assert!(s.fill_converged, "{}", s.report());
        // Sweeps are checked for convergence in batches of four, so this counts
        // in fours; needing more than three batches would mean the fill is
        // crawling rather than sweeping.
        assert!(
            s.fill_iterations <= 12,
            "flood fill needed {} iterations",
            s.fill_iterations
        );
    }
    eprintln!("best full re-voxelisation: {best:.1} ms at dx = 0.75 mm");
    assert!(
        best < 5_000.0,
        "a full re-voxelisation took {best:.0} ms, which is not interactive"
    );
}

/// The drag path, at the interactive tier, with the real part in the scene.
///
/// This is what actually has to keep up with the mouse: a small obstruction
/// moving next to an 85k-triangle duct. Only the triangles near the box the
/// obstruction swept through should be dispatched.
#[test]
fn dragging_an_obstruction_next_to_the_part_stays_interactive() {
    let Some(load) = part() else { return };
    let Some(gpu) = ad_geom::test_gpu() else {
        return;
    };

    let b = load.mesh.bbox();
    let c = b.center();
    let grid = Grid::covering(
        Bbox {
            min: c - Vec3::new(130.0, 90.0, 90.0),
            max: c + Vec3::new(130.0, 90.0, 90.0),
        },
        0.75,
    );

    let mut scene = ad_geom::Scene::new();
    scene.add(
        "duct",
        ad_geom::MeshAsset::new(load.mesh.clone()),
        ad_geom::Transform::IDENTITY,
        ad_geom::MeshRole::Duct,
    );
    let blob = scene.add(
        "blob",
        ad_geom::MeshAsset::new(ad_geom::primitives::uv_sphere(Vec3::ZERO, 6.0, 32, 16)),
        ad_geom::Transform::from_translation(c + Vec3::new(0.0, 60.0, 0.0)),
        ad_geom::MeshRole::Obstruction,
    );

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    let first = vox.sync(&mut scene, grid).expect("initial");
    eprintln!("initial: {}", first.report());

    let mut best = f64::INFINITY;
    let mut worst = 0.0f64;
    for step in 1..=8 {
        scene.set_transform(
            blob,
            ad_geom::Transform::from_translation(c + Vec3::new(0.0, 60.0 - 2.0 * step as f32, 0.0)),
        );
        let s = vox.sync(&mut scene, grid).expect("drag");
        assert!(s.incremental, "the drag forced a full rebuild");
        assert!(s.fill_converged, "{}", s.report());
        best = best.min(s.total_ms);
        worst = worst.max(s.total_ms);
        if step == 1 {
            eprintln!("drag: {}", s.report());
        }
    }
    eprintln!("drag re-voxelisation: {best:.1} to {worst:.1} ms per step at dx = 0.75 mm");
    assert!(worst < 5_000.0, "a drag step took {worst:.0} ms");
}

#[test]
fn subdividing_the_part_preserves_its_surface() {
    let Some(load) = part() else { return };
    let before = load.mesh.health();
    let s = ad_geom::subdivide_to_edge_length(&load.mesh, 0.75, 8_000_000, 8);
    eprintln!(
        "subdivided {} -> {} triangles in {} levels, longest edge {:.3} mm (converged: {})",
        before.triangle_count,
        s.mesh.triangle_count(),
        s.levels,
        s.max_edge_mm,
        s.converged
    );
    let after = s.mesh.health();

    assert!(
        after.is_watertight_manifold(),
        "subdivision cracked the mesh"
    );
    assert_eq!(
        after.topology.euler_characteristic,
        before.topology.euler_characteristic
    );
    assert!(
        (after.signed_volume_mm3 - before.signed_volume_mm3).abs() / before.signed_volume_mm3
            < 1e-4,
        "volume moved: {} -> {}",
        before.signed_volume_mm3,
        after.signed_volume_mm3
    );
    assert!(
        (after.surface_area_mm2 - before.surface_area_mm2).abs() / before.surface_area_mm2 < 1e-4
    );
    assert_eq!(s.source.len(), s.mesh.triangle_count());
    assert!(s
        .source
        .iter()
        .all(|i| (*i as usize) < before.triangle_count));
}

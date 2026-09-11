//! Validation: the GPU voxeliser against analytic fields and against the CPU
//! reference.
//!
//! Every test here skips cleanly when no adapter is present, per the build
//! contract — CI machines have no Vulkan device, and a suite that is red there
//! by design trains everyone to ignore it.

use ad_geom::primitives::{self, ConvexSolid};
use ad_geom::{
    scene::FlatGeometry, CpuSdf, MeshAsset, MeshRole, Scene, Transform, TriMesh, Voxelizer,
};
use ad_gpu::{flags, Bbox, GpuContext, Grid};
use glam::{UVec3, Vec3};

/// `None` means "no GPU"; every test treats that as a skip.
fn gpu() -> Option<GpuContext> {
    ad_geom::test_gpu()
}

fn grid_around(mesh: &TriMesh, dx: f32, margin_cells: f32) -> Grid {
    Grid::covering(mesh.bbox().expanded(Vec3::splat(dx * margin_cells)), dx)
}

fn cell_at(grid: Grid, i: usize) -> UVec3 {
    let nx = grid.dims.x as usize;
    let ny = grid.dims.y as usize;
    UVec3::new(
        (i % nx) as u32,
        ((i / nx) % ny) as u32,
        (i / (nx * ny)) as u32,
    )
}

fn soup(m: &TriMesh) -> Vec<[Vec3; 3]> {
    (0..m.triangle_count()).map(|t| m.triangle(t)).collect()
}

/// The distance values themselves, against a field known in closed form.
///
/// A box is the right shape for this: its SDF is exact everywhere including at
/// its edges and corners, and its mesh has no tessellation error to hide behind.
#[test]
fn narrow_band_matches_the_analytic_box_sdf() {
    let Some(gpu) = gpu() else { return };
    let (min, max) = (Vec3::new(-6.0, -4.0, -9.0), Vec3::new(7.0, 11.0, 3.0));
    let mesh = primitives::box_mesh(min, max);
    let grid = grid_around(&mesh, 0.5, 8.0);

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    let stats = vox
        .voxelize(FlatGeometry::from_mesh(&mesh), grid)
        .expect("voxelize");
    let phi = vox.read_phi().expect("phi readback");

    let mut worst = 0.0f32;
    let mut checked = 0usize;
    for (i, got) in phi.iter().enumerate() {
        let p = grid.cell_center_mm(cell_at(grid, i));
        let want = primitives::box_sdf(p, min, max);
        // Only the band makes a promise about its value.
        if want.abs() > stats.band_mm * 0.98 {
            continue;
        }
        checked += 1;
        worst = worst.max((got - want).abs());
    }
    assert!(checked > 20_000, "only {checked} band cells");
    assert!(
        worst < 0.01 * grid.dx_mm,
        "worst narrow-band error {worst} mm exceeds 0.01*dx = {} mm over {checked} cells",
        0.01 * grid.dx_mm
    );
}

/// The same, for a curved surface. The comparison target is the *mesh's* exact
/// field rather than the sphere's, so this measures the voxeliser and not the
/// tessellation.
#[test]
fn narrow_band_matches_the_cpu_reference_on_a_sphere() {
    let Some(gpu) = gpu() else { return };
    let mesh = primitives::uv_sphere(Vec3::new(1.0, -2.0, 0.5), 9.0, 96, 48);
    let grid = grid_around(&mesh, 0.5, 8.0);

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    let stats = vox
        .voxelize(FlatGeometry::from_mesh(&mesh), grid)
        .expect("voxelize");
    let phi = vox.read_phi().expect("phi readback");

    let reference = CpuSdf::from_mesh(&mesh).sample_band(grid, stats.band_mm);
    let mut worst = 0.0f32;
    let mut checked = 0usize;
    for (i, want) in reference.iter().enumerate() {
        if want.is_nan() || want.abs() > stats.band_mm * 0.98 {
            continue;
        }
        checked += 1;
        worst = worst.max((phi[i] - want).abs());
    }
    assert!(checked > 20_000, "only {checked} band cells");
    assert!(
        worst < 0.01 * grid.dx_mm,
        "worst error {worst} mm over {checked} cells"
    );

    // ...and against the analytic sphere, within the inscribed-polyhedron error.
    let theta = std::f32::consts::TAU / 96.0;
    let tol = 9.0 * theta * theta / 4.0 * 1.5 + 0.01 * grid.dx_mm;
    for (i, got) in phi.iter().enumerate() {
        let p = grid.cell_center_mm(cell_at(grid, i));
        let want = (p - Vec3::new(1.0, -2.0, 0.5)).length() - 9.0;
        if want.abs() > stats.band_mm * 0.98 {
            continue;
        }
        assert!(
            (got - want).abs() < tol,
            "{got} vs analytic {want} at {p:?}"
        );
    }
}

/// Signs at a sharp edge, on the GPU this time. The CPU version of this claim is
/// in `pseudonormal.rs`; this checks the shader implements the same rule.
#[test]
fn signs_are_right_at_a_sharp_edge() {
    let Some(gpu) = gpu() else { return };
    let mesh = primitives::sharp_wedge(25.0, 30.0, 12.0);
    let solid = ConvexSolid::from_convex_mesh(&mesh);
    let grid = grid_around(&mesh, 0.25, 8.0);

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    vox.voxelize(FlatGeometry::from_mesh(&mesh), grid)
        .expect("voxelize");
    let phi = vox.read_phi().expect("phi readback");

    let mut wrong = 0usize;
    let mut checked = 0usize;
    for (i, got) in phi.iter().enumerate() {
        let p = grid.cell_center_mm(cell_at(grid, i));
        // A cell centre within a whisker of the surface may legitimately fall
        // either way; everything else must be decided correctly.
        if got.abs() < 1e-3 {
            continue;
        }
        checked += 1;
        if (*got < 0.0) != solid.contains(p) {
            wrong += 1;
        }
    }
    assert!(checked > 50_000, "only {checked} cells");
    assert_eq!(
        wrong, 0,
        "{wrong} of {checked} cells were signed wrongly near a 25 degree edge"
    );
}

/// The solid mask against a completely independent method: ray parity, which
/// shares no code with the distance field at all.
#[test]
fn solid_mask_matches_the_ray_parity_reference() {
    let Some(gpu) = gpu() else { return };
    // The box's faces must not land on cell centres. `grid_around` puts the
    // centres at `min - 2.2 + k * 0.4`, so a face `size` from `min` sits on one
    // whenever `size + 2.2` is a multiple of 0.4; a 9 mm side does exactly
    // that (k = 28) and hands the SDF ~1,100 cells with `phi = +/-0` decided
    // by rounding. The RTX 4090 happened to round them to agree with the ray
    // parity reference; the DX12 software rasteriser on a CI runner rounded
    // 575 of them the other way. The sides here (9.3, 9.1, 10) put every face
    // between centres.
    for mesh in [
        primitives::box_mesh(Vec3::new(-4.0, -6.0, -2.0), Vec3::new(5.3, 3.1, 8.0)),
        primitives::uv_sphere(Vec3::ZERO, 10.0, 64, 32),
        primitives::torus(Vec3::ZERO, 11.0, 3.5, 96, 48),
        primitives::sharp_wedge(25.0, 24.0, 10.0),
    ] {
        let grid = grid_around(&mesh, 0.4, 6.0);
        let reference = ad_geom::ray_parity_voxelize(&soup(&mesh), grid);
        assert!(reference.is_watertight(), "{}", reference.report());

        let mut vox = Voxelizer::new(&gpu).expect("pipelines");
        let stats = vox
            .voxelize(FlatGeometry::from_mesh(&mesh), grid)
            .expect("voxelize");
        assert!(stats.fill_converged, "{}", stats.report());
        let cell_flags = vox.read_flags().expect("flag readback");

        let mut disagree = 0usize;
        let mut borderline = 0usize;
        let phi = vox.read_phi().expect("phi readback");
        for (i, f) in cell_flags.iter().enumerate() {
            let gpu_solid = *f & flags::SOLID != 0;
            if gpu_solid != reference.solid[i] {
                // A cell centre sitting within float noise of the surface can
                // legitimately be classified either way by either method, since
                // both decide by the sign at the centre. Count those separately
                // rather than pretending they do not exist.
                if phi[i].abs() > 1e-3 {
                    disagree += 1;
                } else {
                    borderline += 1;
                }
            }
        }
        let total = cell_flags.len();
        assert_eq!(
            disagree,
            0,
            "{disagree} of {total} cells disagreed with ray parity; {}",
            stats.report()
        );
        assert!(
            borderline * 1000 < stats.boundary_cells as usize,
            "{borderline} cells sat exactly on the surface, out of {} boundary cells; that is \
             more coincidence than a real mesh should produce",
            stats.boundary_cells
        );
        // And the mask has to have the right volume, not merely agree cell by
        // cell. The tolerance covers only the borderline cells counted above.
        assert!(
            (stats.solid_volume_mm3() / reference.volume_mm3() - 1.0).abs() < 1e-3,
            "{} vs {}",
            stats.solid_volume_mm3(),
            reference.volume_mm3()
        );
    }
}

/// The link list is the part the solver actually consumes. Check it against the
/// field it was derived from rather than against a stored number.
#[test]
fn boundary_links_are_consistent_with_the_field() {
    let Some(gpu) = gpu() else { return };
    let mesh = primitives::uv_sphere(Vec3::ZERO, 8.0, 64, 32);
    let grid = grid_around(&mesh, 0.5, 6.0);

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    let stats = vox
        .voxelize(FlatGeometry::from_mesh(&mesh), grid)
        .expect("voxelize");
    let phi = vox.read_phi().expect("phi");
    let cell_flags = vox.read_flags().expect("flags");
    let links = vox.read_links().expect("links");

    assert_eq!(links.len(), stats.link_count as usize);
    assert!(
        !links.is_empty(),
        "a sphere in a box must have boundary links"
    );

    let dirs = ad_gpu::lattice::D3Q19_DIRS;
    let mut cells_with_links = std::collections::HashSet::new();
    for link in &links {
        let cell = link.cell as usize;
        cells_with_links.insert(link.cell);

        // Every link starts on a fluid cell that is flagged as a boundary.
        assert!(
            flags::is_fluid(cell_flags[cell]),
            "link from a solid cell {cell}"
        );
        assert!(
            cell_flags[cell] & flags::SOLID_BOUNDARY != 0,
            "cell {cell} has a link but is not flagged SOLID_BOUNDARY"
        );

        let d = link.direction as usize;
        assert!(
            (1..19).contains(&d),
            "direction {d} is not a D3Q19 non-rest link"
        );

        // ...and ends on a solid one.
        let c = cell_at(grid, cell).as_ivec3() + dirs[d];
        assert!(c.cmpge(glam::IVec3::ZERO).all() && c.cmplt(grid.dims.as_ivec3()).all());
        let n = grid.linear(c.as_uvec3()) as usize;
        assert!(
            !flags::is_fluid(cell_flags[n]),
            "link {d} from {cell} does not reach solid"
        );

        // q must reproduce the linear interpolation along the link.
        let want = (phi[cell] / (phi[cell] - phi[n])).clamp(1.0 / 255.0, 1.0);
        assert!(
            (link.q() - want).abs() <= 1.0 / 255.0 + 1e-6,
            "q {} for link {d} of cell {cell} should be {want}",
            link.q()
        );
        assert!(link.q() > 0.0 && link.q() <= 1.0);
    }

    // Every boundary-flagged cell has to appear, or the solver would apply plain
    // bounce-back where interpolation was expected.
    let flagged: usize = cell_flags
        .iter()
        .filter(|f| **f & flags::SOLID_BOUNDARY != 0)
        .count();
    assert_eq!(
        flagged,
        cells_with_links.len(),
        "flagged boundary cells without links"
    );
    assert_eq!(flagged as u64, stats.boundary_cells);
}

/// A partial re-voxelisation must produce exactly the same field as a full one.
/// This is the path that runs while the user drags an obstruction, so a
/// discrepancy would show up as the mask slowly rotting over a drag.
#[test]
fn an_incremental_update_matches_a_full_rebuild() {
    let Some(gpu) = gpu() else { return };
    let duct = MeshAsset::new(primitives::box_mesh(
        Vec3::new(-20.0, -12.0, -12.0),
        Vec3::new(20.0, 12.0, 12.0),
    ));
    let blob = MeshAsset::new(primitives::uv_sphere(Vec3::ZERO, 5.0, 32, 16));

    let mut scene = Scene::new();
    scene.add("duct", duct, Transform::IDENTITY, MeshRole::Duct);
    let obstruction = scene.add(
        "blob",
        blob,
        Transform::from_translation(Vec3::new(-30.0, 0.0, 0.0)),
        MeshRole::Obstruction,
    );
    let grid = Grid::covering(
        Bbox {
            min: Vec3::new(-45.0, -20.0, -20.0),
            max: Vec3::new(45.0, 20.0, 20.0),
        },
        0.5,
    );

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    vox.sync(&mut scene, grid).expect("initial");

    // Drag the obstruction, then update incrementally.
    let moved = Transform::from_translation(Vec3::new(-24.0, 4.0, 2.0));
    scene.set_transform(obstruction, moved);
    let incremental = vox.sync(&mut scene, grid).expect("incremental");
    assert!(
        incremental.incremental,
        "the drag should not have forced a full rebuild"
    );
    assert!(
        incremental.active_triangles < incremental.triangles,
        "an incremental pass dispatched all {} triangles",
        incremental.triangles
    );
    let inc_flags = vox.read_flags().expect("flags");
    let inc_phi = vox.read_phi().expect("phi");

    // Now rebuild from scratch with the part already in its new place.
    let mut fresh = Scene::new();
    fresh.add(
        "duct",
        MeshAsset::new(primitives::box_mesh(
            Vec3::new(-20.0, -12.0, -12.0),
            Vec3::new(20.0, 12.0, 12.0),
        )),
        Transform::IDENTITY,
        MeshRole::Duct,
    );
    fresh.add(
        "blob",
        MeshAsset::new(primitives::uv_sphere(Vec3::ZERO, 5.0, 32, 16)),
        moved,
        MeshRole::Obstruction,
    );
    let mut vox2 = Voxelizer::new(&gpu).expect("pipelines");
    let full = vox2.sync(&mut fresh, grid).expect("full");
    let full_flags = vox2.read_flags().expect("flags");
    let full_phi = vox2.read_phi().expect("phi");

    assert_eq!(
        incremental.solid_cells, full.solid_cells,
        "solid cell counts differ"
    );
    assert_eq!(incremental.boundary_cells, full.boundary_cells);
    assert_eq!(incremental.link_count, full.link_count);
    assert_eq!(inc_flags, full_flags, "the flag fields differ");
    let worst = inc_phi
        .iter()
        .zip(&full_phi)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-4,
        "the distance fields differ by up to {worst} mm"
    );
}

/// Moving a part back to where it started must restore the original mask
/// exactly. A one-way incremental update that leaves debris behind would pass
/// the test above and still be wrong over a long drag.
#[test]
fn dragging_a_part_and_back_restores_the_original_mask() {
    let Some(gpu) = gpu() else { return };
    let mut scene = Scene::new();
    scene.add(
        "duct",
        MeshAsset::new(primitives::box_mesh(Vec3::splat(-15.0), Vec3::splat(15.0))),
        Transform::IDENTITY,
        MeshRole::Duct,
    );
    let blob = scene.add(
        "blob",
        MeshAsset::new(primitives::uv_sphere(Vec3::ZERO, 4.0, 24, 12)),
        Transform::from_translation(Vec3::new(-24.0, 0.0, 0.0)),
        MeshRole::Obstruction,
    );
    let grid = Grid::covering(
        Bbox {
            min: Vec3::splat(-32.0),
            max: Vec3::splat(32.0),
        },
        0.6,
    );

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    vox.sync(&mut scene, grid).expect("initial");
    let before = vox.read_flags().expect("flags");

    for x in [-20.0f32, -14.0, -8.0, -14.0, -20.0, -24.0] {
        scene.set_transform(blob, Transform::from_translation(Vec3::new(x, 0.0, 0.0)));
        vox.sync(&mut scene, grid).expect("drag step");
    }
    let after = vox.read_flags().expect("flags");
    assert_eq!(before, after, "a round trip left the mask changed");
}

/// Two shapes at once, with the second inside the first, so the flood fill has
/// to handle a cavity that is not reachable from the domain boundary.
#[test]
fn a_sealed_cavity_stays_fluid_and_an_enclosed_solid_stays_solid() {
    let Some(gpu) = gpu() else { return };
    // A hollow shell: outer box, inner box wound inside-out so the material is
    // the space between them and the middle is a sealed void.
    let mut mesh = primitives::box_mesh(Vec3::splat(-10.0), Vec3::splat(10.0));
    let mut inner = primitives::box_mesh(Vec3::splat(-6.0), Vec3::splat(6.0));
    inner.flip_winding();
    let offset = mesh.positions.len() as u32;
    mesh.positions.extend(inner.positions);
    mesh.indices.extend(
        inner
            .indices
            .iter()
            .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset]),
    );

    let grid = grid_around(&mesh, 0.5, 8.0);
    let reference = ad_geom::ray_parity_voxelize(&soup(&mesh), grid);
    assert!(reference.is_watertight(), "{}", reference.report());

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    let stats = vox
        .voxelize(FlatGeometry::from_mesh(&mesh), grid)
        .expect("voxelize");
    let cell_flags = vox.read_flags().expect("flags");
    let phi = vox.read_phi().expect("phi");

    let mut disagree = 0usize;
    for (i, f) in cell_flags.iter().enumerate() {
        if ((*f & flags::SOLID != 0) != reference.solid[i]) && phi[i].abs() > 1e-3 {
            disagree += 1;
        }
    }
    assert_eq!(
        disagree,
        0,
        "{disagree} cells disagreed; {}",
        stats.report()
    );

    // The sealed void in the middle must be fluid, and the shell around it solid.
    let centre = grid.linear(
        ((Vec3::ZERO - grid.origin_mm) / grid.dx_mm)
            .round()
            .as_uvec3(),
    ) as usize;
    assert!(
        flags::is_fluid(cell_flags[centre]),
        "the sealed cavity was filled in"
    );
    let in_wall = grid.linear(
        ((Vec3::new(8.0, 0.0, 0.0) - grid.origin_mm) / grid.dx_mm)
            .round()
            .as_uvec3(),
    ) as usize;
    assert!(
        !flags::is_fluid(cell_flags[in_wall]),
        "the shell wall came out fluid"
    );
}

/// A grid whose cells outnumber a single dispatch's workgroup limit, to prove
/// the grid-stride loops cover everything. 65535 workgroups of 64 threads is
/// 4.2 M cells, so this grid needs several passes per thread.
#[test]
fn a_grid_larger_than_one_dispatch_is_fully_covered() {
    let Some(gpu) = gpu() else { return };
    let mesh = primitives::box_mesh(Vec3::splat(-20.0), Vec3::splat(20.0));
    // 220^3 = 10.6 M cells, comfortably past the 4.2 M a single dispatch covers.
    let grid = Grid::covering(
        Bbox {
            min: Vec3::splat(-27.5),
            max: Vec3::splat(27.5),
        },
        0.25,
    );
    assert!(
        grid.cell_count() > 65535 * 64,
        "grid too small to exercise the stride loop"
    );

    let mut vox = Voxelizer::new(&gpu).expect("pipelines");
    let stats = vox
        .voxelize(FlatGeometry::from_mesh(&mesh), grid)
        .expect("voxelize");
    assert!(stats.fill_converged, "{}", stats.report());
    let exact = 40.0f64.powi(3);
    assert!(
        (stats.solid_volume_mm3() / exact - 1.0).abs() < 0.01,
        "{} vs {exact}: {}",
        stats.solid_volume_mm3(),
        stats.report()
    );
}

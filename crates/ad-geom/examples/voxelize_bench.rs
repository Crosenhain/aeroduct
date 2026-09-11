//! Measure the voxeliser on a real part at every resolution tier.
//!
//! ```text
//! cargo run --release -p ad-geom --example voxelize_bench [-- path/to/part.stl]
//! ```
//!
//! Reports a full re-voxelisation and a drag step at each tier, which are the
//! two numbers that decide whether dragging an obstruction feels live.

use ad_geom::scene::FlatGeometry;
use ad_geom::{MeshAsset, MeshRole, Scene, Transform, Voxelizer};
use ad_gpu::{Bbox, GpuContext, Grid};
use glam::Vec3;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let path = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .or_else(ad_geom::test_stl_path)
        .ok_or_else(|| {
            anyhow::anyhow!("pass an STL path, or set AERODUCT_TEST_STL to the test part")
        })?;

    let t0 = std::time::Instant::now();
    let load = ad_geom::load_stl(&path)?;
    println!("{}", load.summary());
    println!(
        "loaded and welded in {:.0} ms",
        t0.elapsed().as_secs_f64() * 1e3
    );
    println!("{}", load.health().report());

    for m in ad_geom::detect_mouths(&load.mesh, load.mesh.bbox(), Default::default()) {
        println!(
            "mouth: axis {} {}, {:.0} mm^2, D_h {:.1} mm, normal {:?}",
            m.axis,
            if m.on_min_side { "min" } else { "max" },
            m.open_area_mm2,
            m.hydraulic_diameter_mm(),
            m.patch.normal,
        );
    }

    let gpu = GpuContext::new_blocking(None)?;
    println!("\nGPU: {} ({:?})\n", gpu.info.name, gpu.info.backend);

    let centre = load.mesh.bbox().center();
    let domain = Bbox {
        min: centre - Vec3::new(130.0, 90.0, 90.0),
        max: centre + Vec3::new(130.0, 90.0, 90.0),
    };

    for (tier, dx) in [("interactive", 0.75f32), ("quality", 0.40), ("max", 0.30)] {
        let grid = Grid::covering(domain, dx);
        println!(
            "--- {tier}: dx = {dx} mm, {}x{}x{} = {:.1} M cells ---",
            grid.dims.x,
            grid.dims.y,
            grid.dims.z,
            grid.cell_count() as f64 / 1e6
        );

        let mut vox = match Voxelizer::new(&gpu) {
            Ok(v) => v,
            Err(e) => {
                println!("  skipped: {e}");
                continue;
            }
        };
        let geom = FlatGeometry::from_mesh(&load.mesh);
        match vox.voxelize(geom.clone(), grid) {
            Ok(s) => println!("  cold  {}", s.report()),
            Err(e) => {
                println!("  skipped: {e}");
                continue;
            }
        }
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let s = vox.voxelize(geom.clone(), grid)?;
            best = best.min(s.total_ms);
        }
        println!("  full re-voxelisation: {best:.1} ms");

        // Drag an obstruction beside the duct and time the incremental path.
        let mut scene = Scene::new();
        scene.add(
            "duct",
            MeshAsset::new(load.mesh.clone()),
            Transform::IDENTITY,
            MeshRole::Duct,
        );
        let blob = scene.add(
            "blob",
            MeshAsset::new(ad_geom::primitives::uv_sphere(Vec3::ZERO, 6.0, 32, 16)),
            Transform::from_translation(centre + Vec3::new(0.0, 60.0, 0.0)),
            MeshRole::Obstruction,
        );
        let mut vox = Voxelizer::new(&gpu)?;
        vox.sync(&mut scene, grid)?;
        let mut drag = f64::INFINITY;
        let mut dispatched = 0;
        for step in 1..=8 {
            scene.set_transform(
                blob,
                Transform::from_translation(centre + Vec3::new(0.0, 60.0 - step as f32, 0.0)),
            );
            let s = vox.sync(&mut scene, grid)?;
            drag = drag.min(s.total_ms);
            dispatched = s.active_triangles;
        }
        println!("  drag step: {drag:.1} ms ({dispatched} triangles dispatched)\n");
    }

    Ok(())
}

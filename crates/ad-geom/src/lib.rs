//! Geometry pipeline for AeroDuct: STL in, solver-ready voxels out.
//!
//! The chain is:
//!
//! ```text
//! stl::load_stl        -> TriMesh (welded, wound from the file's vertex order)
//! mesh::MeshHealth     -> is it watertight? what volume? what genus?
//! scene::Scene         -> several meshes, each with a transform and a role
//! voxelize::Voxelizer  -> flags + narrow-band SDF + BoundaryLink list, on the GPU
//! mouths::detect_mouths-> where the air goes in and out
//! subdivide            -> a surface fine enough to paint solver results onto
//! ```
//!
//! Two design decisions are load-bearing and are argued for where they live:
//!
//! * The sign of the distance field comes from the **angle-weighted
//!   pseudonormal**, not from the nearest face normal. See
//!   [`mesh::PseudoNormals`]. Getting this wrong produces stray solid cells in
//!   the passage and a solver that diverges, and it is ten lines of code to get
//!   right.
//! * The `atomicMin` that finds the nearest triangle operates on the **IEEE bit
//!   pattern of a non-negative float**, which is monotonic as an unsigned
//!   integer. See `atomic_min_dist` in `shaders/geom/common.wgsl`.
//!
//! Everything in [`cpu_ref`] exists to check the GPU path against an
//! independent method, and [`primitives`] supplies shapes whose distance fields
//! are known in closed form.

pub mod bins;
pub mod cpu_ref;
pub mod mesh;
pub mod mouths;
pub mod primitives;
pub mod scene;
pub mod stl;
pub mod subdivide;
pub mod voxelize;

pub use bins::TriangleBins;
pub use cpu_ref::{ray_parity_voxelize, CpuSdf, RayParityResult};
pub use mesh::{Feature, MeshHealth, PseudoNormals, Topology, TriMesh};
pub use mouths::{detect_in_scene, detect_mouths, Mouth, MouthConfig};
pub use scene::{FlatGeometry, MeshAsset, MeshInstance, MeshRole, Scene, SceneDirty, Transform};
pub use stl::{load_stl, parse_stl, StlFormat, StlLoad};
pub use subdivide::{subdivide_to_edge_length, Subdivided};
pub use voxelize::{VoxelStats, VoxelizeConfig, Voxelizer};

/// Locate the test STL, if it is available.
///
/// A test duct, not a synthetic fixture: it lives in `parts/` at the workspace
/// root (or beside the repository, its old home), and every test that wants it
/// has to skip cleanly when it is absent. Set `AERODUCT_TEST_STL` to point at
/// it explicitly.
pub fn test_stl_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("AERODUCT_TEST_STL") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    // `CARGO_MANIFEST_DIR` is crates/ad-geom, so the workspace root is two up.
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        "../../parts/Airflow redirector - Part 1.stl",
        "../../../Airflow redirector - Part 1.stl",
        "../../Airflow redirector - Part 1.stl",
        "../Airflow redirector - Part 1.stl",
    ];
    candidates.iter().map(|c| manifest.join(c)).find(|p| p.exists())
}

/// Acquire a GPU context for a test, or `None` if this machine has no adapter.
///
/// Tests that need a GPU must skip rather than fail when one is missing, per the
/// build contract: CI runners generally have no Vulkan device and a red test
/// suite there teaches everyone to ignore red test suites.
pub fn test_gpu() -> Option<ad_gpu::GpuContext> {
    match ad_gpu::GpuContext::new_blocking(None) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("skipping GPU test: no adapter available ({e})");
            None
        }
    }
}

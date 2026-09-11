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

/// The test duct's file name inside `parts/`.
pub const TEST_STL_NAME: &str = "Airflow redirector - Part 1.stl";

/// Locate the test STL, if it is available.
///
/// A test duct, not a synthetic fixture: it lives in `parts/` at the workspace
/// root, and every test that wants it has to skip cleanly when it is absent.
/// Set `AERODUCT_TEST_STL` to point at it explicitly.
///
/// A shipped binary has no workspace, so the search also covers `parts/` next
/// to the executable (which is how a release package is laid out) and under
/// the current directory. See [`test_stl_candidates`] for the order.
pub fn test_stl_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("AERODUCT_TEST_STL") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    // `CARGO_MANIFEST_DIR` is crates/ad-geom, so the workspace root is two up.
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let cwd = std::env::current_dir().ok();
    test_stl_candidates(&workspace, exe_dir.as_deref(), cwd.as_deref())
        .into_iter()
        .find(|p| p.exists())
}

/// Where [`test_stl_path`] looks, in order: the workspace the binary was built
/// from, then `parts/` beside the executable, then `parts/` under the current
/// directory, then the pre-`parts/` location beside the repository.
///
/// The workspace comes first so a developer's `cargo run` always picks up the
/// file under version control, even when a stale copy sits in `target/`.
pub fn test_stl_candidates(
    workspace: &std::path::Path,
    exe_dir: Option<&std::path::Path>,
    cwd: Option<&std::path::Path>,
) -> Vec<std::path::PathBuf> {
    let mut out = vec![workspace.join("parts").join(TEST_STL_NAME)];
    out.extend(exe_dir.map(|d| d.join("parts").join(TEST_STL_NAME)));
    out.extend(cwd.map(|d| d.join("parts").join(TEST_STL_NAME)));
    out.push(workspace.join("..").join(TEST_STL_NAME));
    out
}

#[cfg(test)]
mod test_stl_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn the_shipped_layout_is_searched_after_the_workspace() {
        let c = test_stl_candidates(
            Path::new("/ws"),
            Some(Path::new("/opt/aeroduct")),
            Some(Path::new("/home/me")),
        );
        assert_eq!(c[0], Path::new("/ws/parts").join(TEST_STL_NAME));
        assert_eq!(c[1], Path::new("/opt/aeroduct/parts").join(TEST_STL_NAME));
        assert_eq!(c[2], Path::new("/home/me/parts").join(TEST_STL_NAME));
        assert_eq!(c[3], Path::new("/ws/..").join(TEST_STL_NAME));
    }

    #[test]
    fn missing_exe_or_cwd_just_shortens_the_list() {
        assert_eq!(test_stl_candidates(Path::new("/ws"), None, None).len(), 2);
    }

    #[test]
    fn the_checked_in_test_duct_is_found() {
        // Not a skip: the file is under version control in parts/.
        let p = test_stl_path().expect("parts/ test duct");
        assert!(p.ends_with(TEST_STL_NAME), "{}", p.display());
    }
}

/// Acquire a GPU context for a test, or `None` if this machine has no adapter.
///
/// Tests that need a GPU must skip rather than fail when one is missing, per the
/// build contract: CI runners generally have no Vulkan device and a red test
/// suite there teaches everyone to ignore red test suites. See
/// [`ad_gpu::GpuContext::for_tests`] for the software-adapter rule.
pub fn test_gpu() -> Option<ad_gpu::GpuContext> {
    ad_gpu::GpuContext::for_tests()
}

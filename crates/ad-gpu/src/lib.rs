//! Foundation crate for AeroDuct.
//!
//! Owns the GPU device, the shared data types every other crate codes against,
//! the WGSL preprocessor, the structure-of-arrays distribution-function storage,
//! and the roofline profiler.
//!
//! Nothing here knows about ducts, meshes or rendering. If a type is needed by
//! two sibling crates, it belongs in [`types`]; if only one crate needs it, it
//! does not belong here at all.

pub mod context;
pub mod ddf;
pub mod lattice;
pub mod profiler;
pub mod shader;
pub mod types;

pub use context::{GpuCapabilities, GpuContext};
pub use ddf::{bytes_per_cell, direction_bytes, predicted_steps_per_second, DdfBuffers};
pub use lattice::{opposite, EsotericPull, LatticeDef};
pub use profiler::Profiler;
pub use shader::{ShaderDefines, ShaderLoader};
pub use types::{
    air, flags, Bbox, BoundaryLink, DdfPrecision, FlowPatch, Grid, LatticeUnits, MetricSample,
    SimUniforms, VelocitySet,
};

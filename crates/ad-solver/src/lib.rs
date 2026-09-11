//! AeroDuct lattice-Boltzmann solver.

pub mod boundary;
pub mod collision;
pub mod config;
pub mod precision;
pub mod reference;
pub mod shaders;
pub mod solver;

pub use boundary::{CellKind, LinkTable, PaddedDomain};
pub use collision::CollisionModel;
pub use config::{InletSpec, SolverConfig};
pub use reference::{Macro, ReferenceLbm};
pub use solver::{LbmUniforms, Solver};

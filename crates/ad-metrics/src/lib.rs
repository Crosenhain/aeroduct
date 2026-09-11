//! AeroDuct engineering metrics: the numbers that say whether a duct is any
//! good, and how much to trust each one.
//!
//! # The layers
//!
//! ```text
//! field      what the kernels read: velocity, density, flags (one bind group)
//! shaders    WGSL assembly, plus the GENERATED accumulator layout
//!   plane    512x512 quadrature over a FlowPatch -> flow, pressure, uniformity
//!   volume   every cell -> peak velocity, separation, stagnation, residual
//!   wall     one record per triangle -> wall pressure, shear, y+
//! readback   a non-blocking ring; the frame loop never waits for a status bar
//! stats      Welford, autocorrelation-corrected SEM, convergence, resets
//! rtd        residence time distribution from tracer ages
//! metrics    the aggregate the UI binds to
//! ```
//!
//! # Three rules everything here follows
//!
//! **1. Every reported scalar carries its standard error.** A turbulent duct
//! never reaches a steady state, so one frame's flow rate is a sample from a
//! distribution, not a measurement. `dp = 47.3 +/- 0.6 Pa` is an engineering
//! statement; `47.31 Pa` is a statement about one time step. See
//! [`stats::Estimate`], and [`stats`]'s module docs for why the naive
//! `sigma/sqrt(N)` overstates the precision by a factor of four or more.
//!
//! **2. Mass imbalance gates everything.** `|mdot_in - mdot_out| / mdot_in` must
//! be under 1%. It is the only number here whose right answer is known in
//! advance, which makes it the only one that can catch a mistake nobody was
//! looking for — a patch clipping geometry, a plane inside the sponge layer, a
//! run still in its transient. [`stats::Monitor`] refuses to call anything
//! converged while it is broken.
//!
//! It is a **mass** flux imbalance, not a volumetric one. LBM is weakly
//! compressible, so across the test part's ~100 Pa the air genuinely expands by
//! 7% between the two planes; differencing volumetric flows would report that
//! physics as a 7% conservation error and bury the 1% gate. The volumetric
//! imbalance is still reported next to it, because the gap between the two *is*
//! the compressibility.
//!
//! **3. What was not measured is said, not assumed.** A patch that clips the
//! wall reports its covered-area fraction. A wall triangle that found no fluid
//! reports itself dry rather than reporting zero shear. A plane that saw nothing
//! reports zero samples rather than a plausible-looking zero.
//!
//! # Units
//!
//! Millimetres in model space and lattice units on the GPU, per CONTRACT.md.
//! Every conversion to SI happens once, on the CPU, through
//! [`ad_gpu::LatticeUnits`] — so no kernel carries a unit conversion and there
//! is one place to look when a number is out by a factor of `dx/dt`.

pub mod field;
pub mod metrics;
pub mod plane;
pub mod readback;
pub mod rtd;
pub mod shaders;
pub mod stats;
pub mod volume;
pub mod wall;

pub use field::{field_bind_group, field_layout, FieldRefs, FieldTextures};
pub use metrics::{
    throw_distance_m, DuctMetrics, Flow, Jet, LossBand, LossCoefficient, MetricsConfig,
    MetricsReport, ReferenceVelocity, Snapshot, Uniformity, Wall,
};
pub use plane::{GpuGrid, PlaneAccum, PlaneMetrics, PlaneReading, DEFAULT_SAMPLES};
pub use readback::{Frame, ReadbackRing};
pub use rtd::{tau_ideal_s, AgeSample, Rtd, RTD_BINS};
pub use stats::{Estimate, Health, Monitor, MonitorConfig, ParameterHash, Series, Welford};
pub use volume::{VolumeAccum, VolumeConfig, VolumeMetrics, VolumeReading};
pub use wall::{WallConfig, WallField, WallMetrics, WallSummary, WallTriangle};

/// Acquire a GPU context for a test, or `None` if this machine has no adapter.
///
/// Per CONTRACT.md, GPU tests must **skip** rather than fail when no adapter is
/// available: CI runners generally have no Vulkan device, and a suite that is
/// permanently red there teaches everyone to ignore red suites.
#[cfg(test)]
pub(crate) fn test_gpu() -> Option<ad_gpu::GpuContext> {
    match ad_gpu::GpuContext::new_blocking(None) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("skipping GPU test: no adapter available ({e})");
            None
        }
    }
}

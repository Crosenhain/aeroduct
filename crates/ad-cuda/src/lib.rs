//! A CUDA compute backend for the AeroDuct lattice-Boltzmann solver, headless.
//!
//! # Why this exists, and what it is not for
//!
//! It is **not** expected to be faster. D3Q19 stream-collide has an arithmetic
//! intensity of ~2.9 FLOP/byte against an RTX 4090's machine balance point of
//! 82, so it is bandwidth-bound by an order of magnitude and no API can beat the
//! memory controller. The wgpu kernel already measures 84-98% of the 1008 GB/s
//! DRAM roofline in isolation. There is essentially nothing left on the table.
//!
//! What there *is* is a measurement we could not otherwise make. The isolated
//! kernel reaches 84% of roofline; the full application reads 53%. Three
//! explanations fit — the extra per-step boundary work, the larger working set,
//! or a wgpu/naga codegen ceiling — and timestamp queries cannot separate them.
//! A second implementation of the same algorithm on the same silicon can. That
//! is the deliverable: an honest A/B, whichever way it falls.
//!
//! Secondarily, a headless backend is what parameter sweeps and neural-surrogate
//! training-data generation actually want, and it puts tensor cores and cuBLAS
//! within reach later.
//!
//! # Measured: the A/B
//!
//! RTX 4090, driver 616.56, CUDA 13.3, peak DRAM 1008 GB/s. D3Q19, TRT at
//! Lambda = 3/16, Smagorinsky Cs = 0.11, FP32, open box on every axis, 64-wide
//! workgroups, 157 B/cell/step on both sides. Backends run strictly one at a
//! time (`cargo run --release -p ad-cuda --features cuda --example ab_bench`);
//! "best" is the fastest of 200 ten-step batches, "GPU" the device timer with
//! the same 0.05 EWMA on both.
//!
//! | backend | grid | Mcells | GPU ms/step | best ms/step | best MLUPS | GPU % roofline | best % roofline |
//! |---|---|---|---|---|---|---|---|
//! | wgpu/Vulkan | 256x128x128 | 4.36 | 0.834 | **0.783** | 5567 | 81.4 | **86.7** |
//! | CUDA | 256x128x128 | 4.36 | 0.848 | **0.821** | 5313 | 80.0 | **82.7** |
//! | wgpu/Vulkan | 384x256x256 | 25.69 | 4.892 | **4.835** | 5314 | 81.8 | **82.8** |
//! | CUDA | 384x256x256 | 25.69 | 5.107 | **5.025** | 5114 | 78.3 | **79.6** |
//!
//! **CUDA is 2-5% slower than wgpu/Vulkan, at both grid sizes and on both
//! timers.** That is the expected result and it is the useful one: it confirms
//! the roofline analysis and rules out a wgpu/naga codegen ceiling. There is
//! nothing to be won by moving the solver to CUDA, and the reason to keep this
//! backend is batch work and diagnosis, not speed.
//!
//! # Measured: what the kernel actually achieves
//!
//! Nsight Compute could not run — GPU performance counters need elevation on
//! Windows (`ERR_NVGPUCTRPERM`) and enabling non-admin profiling is a driver
//! registry change, not ours to make. Nsight Systems traces without counters,
//! which is enough for the question that mattered. Per-launch durations over 410
//! launches, same configuration:
//!
//! | grid | min | p10 | median | mean | max |
//! |---|---|---|---|---|---|
//! | 4.36 M cells | 0.716 ms (**94.8%**) | 0.729 ms (93.2%) | 0.747 ms (90.9%) | 0.910 ms (74.7%) | 2.096 ms (32.4%) |
//! | 25.69 M cells | 4.405 ms (**90.8%**) | 4.439 ms (90.2%) | 5.281 ms (75.8%) | 5.126 ms (78.1%) | 5.928 ms (67.5%) |
//!
//! Two things fall out.
//!
//! 1. **The kernel reaches 91-95% of the DRAM roofline.** The spread is other
//!    GPU clients on a desktop, not the kernel: the distribution has a hard
//!    floor and a long right tail, which is the signature of contention rather
//!    than of anything the kernel does differently from launch to launch. Any
//!    mean-based figure on this machine understates by 15-20 points.
//! 2. **Launch overhead is not a factor.** Median gap between consecutive
//!    kernels is 1.31 us against a 747 us kernel — 0.18%. Whatever costs the
//!    application throughput, it is not per-step dispatch cost.
//!
//! # Measured: where the "53% in the app" actually goes
//!
//! It is an accounting artefact, at least on the production plenum domain.
//! `ad_solver::Solver::bytes_per_step` bills every *padded* cell at 157 B, but a
//! solid cell reads its flag byte and returns before touching a single
//! distribution — and the plenum carve removes 87.7% of them. From the app's own
//! `dx = 0.4 mm` run: 43.1 M padded cells but **5.117 M fluid cells**, at
//! 0.991 ms/step. Billed that way it prints "6823 GB/s, 677% of roofline", which
//! is impossible and is right there in `sweep-out/dx0.4.log`. Billing only the
//! cells that move bytes gives 157 x 5.117 M = 803 MB/step over 0.991 ms =
//! **811 GB/s, 80.4% of roofline** — the same band as the isolated kernel here,
//! and the same band as CUDA.
//!
//! So the in-app kernel is not slower than the isolated one in any way this
//! backend can detect; the denominator is wrong. That does not make the app's
//! `ms/step` wrong, only its bandwidth and roofline lines, and it is worth
//! fixing before any future optimisation is judged against them.
//!
//! One smaller observation from the same trace, shared by both backends: the
//! launch is `grid(5, 130, 130) x block(64, 1, 1)` for a padded x of 258, so 320
//! lanes are started per row to cover 258 cells and the fifth block runs 2 of 64
//! lanes. Harmless for bandwidth — the idle lanes exit before any load — but a
//! padded x that is a multiple of the workgroup size would remove it.
//!
//! # Feature gate
//!
//! Everything that touches a device is behind the off-by-default `cuda` feature.
//! With it off this crate has no `cudarc` dependency and compiles anywhere, and
//! [`kernel`] — the part that can silently drift away from `ad_gpu::lattice` —
//! is still compiled and still tested. That is deliberate: the generator is the
//! risky half, and it is the half that does not need hardware.
//!
//! ```text
//! cargo test --workspace                    # CUDA off, generator tests run
//! cargo test -p ad-cuda --features cuda     # CUDA on, device tests run
//! ```
//!
//! # Validation
//!
//! The four gates in `tests/validation.rs` are the wgpu backend's own, re-run
//! against this one. Measured:
//!
//! | gate | wgpu | CUDA |
//! |---|---|---|
//! | mass drift, closed periodic box, 5000 steps | exactly 0 | **exactly 0** |
//! | Poiseuille effective-width spread over tau = 0.51/0.6/1.0, TRT | 0.0066% | **0.0070%** |
//! | the same with BGK (negative control) | ~0.45% | **0.4458%** |
//! | worst velocity vs `ad_solver::reference`, obstacle | 1.7e-7 | **1.9e-7** |
//! | worst velocity, CUDA vs wgpu, 2000 steps | — | **2.6e-8** |
//!
//! # Scope of this pass
//!
//! Headless only. Vulkan-CUDA rendering interop is deliberately not attempted:
//! wgpu does not cleanly expose external-memory handles, and round-tripping a
//! velocity field through host memory is ~676 MB/frame at the quality tier,
//! which is not viable. See the note at the end of this comment.
//!
//! D3Q19 (or D3Q27) stream-collide with Esoteric Pull in-place streaming, TRT
//! collision at the magic parameter 3/16 plus Smagorinsky, FP32 storage. The
//! FP16C codec is not ported; `ad_solver::precision` stays the only
//! implementation of it.
//!
//! # Interop, as future work
//!
//! Sharing the velocity texture with the Vulkan renderer needs
//! `VK_KHR_external_memory_win32` on the wgpu side and
//! `cuImportExternalMemory` on this one. The CUDA half exists today
//! (`cudarc::driver::external_memory`); the wgpu half does not — `wgpu-hal`
//! creates images without `VkExportMemoryAllocateInfo` and exposes no handle, so
//! it would take either a `wgpu-hal` Vulkan escape hatch or an upstream change.
//! Until then the two backends cannot share a surface, which is exactly why this
//! pass is headless and why the wgpu path remains the one the application runs.

pub mod kernel;

#[cfg(feature = "cuda")]
mod backend;

#[cfg(feature = "cuda")]
pub use backend::{CudaSolver, DeviceInfo};

pub use kernel::{CudaParams, KernelSpec};

use anyhow::Result;

/// The two compute backends, for callers that pick one at run time.
///
/// This is the whole of the "backend selection" surface, and it lives here
/// rather than in `ad-solver` on purpose: nothing in the solver needs to know
/// this crate exists, so the wgpu path carries no CUDA-shaped abstraction and
/// the contract's "never bypass ad-gpu" rule is untouched. A caller that wants
/// both links this crate; one that does not, does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// `ad_solver::Solver`. Always available, renders, and is what the
    /// application runs.
    #[default]
    Wgpu,
    /// [`CudaSolver`]. Headless. Requires the `cuda` feature *and* a device.
    Cuda,
}

impl Backend {
    pub const fn name(self) -> &'static str {
        match self {
            Backend::Wgpu => "wgpu/Vulkan",
            Backend::Cuda => "CUDA",
        }
    }

    /// Whether this backend can actually run here.
    ///
    /// For CUDA this is a real probe — the feature being compiled in says
    /// nothing about whether a driver is installed or a device is present, and
    /// the failure mode of assuming otherwise is a panic deep inside a batch run
    /// rather than a clean fallback.
    pub fn available(self) -> bool {
        match self {
            Backend::Wgpu => true,
            Backend::Cuda => cuda_device_count().unwrap_or(0) > 0,
        }
    }
}

/// Number of CUDA devices, or `Ok(0)` when the feature is off.
///
/// Returns `Err` only when the driver is present but refuses to answer, which is
/// worth surfacing; a missing driver is reported as zero devices.
pub fn cuda_device_count() -> Result<usize> {
    #[cfg(feature = "cuda")]
    {
        backend::device_count()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(0)
    }
}

/// The common surface both backends offer, so a benchmark or a batch sweep can
/// hold either behind one type.
///
/// Deliberately small: stepping, resetting, reading the macroscopic field back,
/// and the four performance numbers. Anything backend-specific — textures, bind
/// groups, NVRTC options — stays off it, because a trait wide enough to cover
/// both would stop being an abstraction and start being a union.
pub trait LbmBackend {
    fn backend(&self) -> Backend;
    fn steps_taken(&self) -> u64;
    /// Advance `n` steps. Both implementations bound the work per submission to
    /// stay inside the Windows TDR window; see `CudaSolver::step`.
    fn step(&mut self, n: u32);
    fn reset(&mut self);
    /// `(u.x, u.y, u.z, rho)` per *interior* cell, X-fastest. Blocking.
    fn read_macroscopic(&mut self) -> Result<Vec<[f32; 4]>>;
    fn set_profiling(&mut self, on: bool);
    /// Pick up any timings that have landed. The wgpu profiler reads back
    /// asynchronously and needs polling; CUDA events do not, so this is a no-op
    /// there. Callers should call it regardless.
    fn collect_profiling(&mut self);
    fn wait_idle(&self);
    fn ms_per_step(&self) -> Option<f64>;
    fn mlups(&self) -> Option<f64>;
    fn roofline_fraction(&self) -> Option<f64>;
    /// Bytes of memory traffic one step moves. On the trait because a wall-clock
    /// measurement has to be turned into a roofline percentage by the caller,
    /// and it must use the *same* traffic model both backends report against or
    /// the comparison is meaningless.
    fn bytes_per_step(&self) -> u64;
    /// Total cells one step updates, including the solid halo. Same reason.
    fn cell_count(&self) -> u64;
}

impl LbmBackend for ad_solver::Solver {
    fn backend(&self) -> Backend {
        Backend::Wgpu
    }
    fn steps_taken(&self) -> u64 {
        ad_solver::Solver::steps_taken(self)
    }
    fn step(&mut self, n: u32) {
        ad_solver::Solver::step(self, n)
    }
    fn reset(&mut self) {
        ad_solver::Solver::reset(self)
    }
    fn read_macroscopic(&mut self) -> Result<Vec<[f32; 4]>> {
        ad_solver::Solver::read_macroscopic(self)
    }
    fn set_profiling(&mut self, on: bool) {
        ad_solver::Solver::set_profiling(self, on)
    }
    fn collect_profiling(&mut self) {
        ad_solver::Solver::collect_profiling(self)
    }
    fn wait_idle(&self) {
        ad_solver::Solver::wait_idle(self)
    }
    fn ms_per_step(&self) -> Option<f64> {
        ad_solver::Solver::ms_per_step(self)
    }
    fn mlups(&self) -> Option<f64> {
        ad_solver::Solver::mlups(self)
    }
    fn roofline_fraction(&self) -> Option<f64> {
        ad_solver::Solver::roofline_fraction(self)
    }
    fn bytes_per_step(&self) -> u64 {
        ad_solver::Solver::bytes_per_step(self)
    }
    fn cell_count(&self) -> u64 {
        self.domain.padded_cell_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wgpu_backend_is_always_available_and_cuda_is_probed() {
        assert!(Backend::Wgpu.available());
        assert_eq!(Backend::default(), Backend::Wgpu);
        // With the feature off this must be false without touching a driver.
        if cfg!(not(feature = "cuda")) {
            assert!(!Backend::Cuda.available());
            assert_eq!(cuda_device_count().unwrap_or(usize::MAX), 0);
        }
    }
}

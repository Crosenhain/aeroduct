//! Sequential A/B of the wgpu/Vulkan and CUDA backends on one device.
//!
//! # Read this before quoting a number from it
//!
//! The kernel is memory-bandwidth-bound: D3Q19 stream-collide moves 157 B/cell
//! and does ~2.9 FLOP/byte against an RTX 4090's machine balance point of 82. So
//! **the only figure that means anything on its own is percent of roofline**,
//! and the only way to measure it is on an otherwise idle GPU. A second client
//! on the same memory controller moves every number here by 2x or more: this
//! project has measured the same build at 98% of roofline alone and 61%
//! alongside the test suite.
//!
//! Hence: one backend at a time, never concurrently, with a full device
//! synchronisation between them. That is why this is an example rather than a
//! `#[test]` — `cargo test` runs test binaries in parallel, which is precisely
//! the contention this measurement cannot tolerate.
//!
//! ```text
//! cargo run --release -p ad-cuda --features cuda --example ab_bench
//! cargo run --release -p ad-cuda --features cuda --example ab_bench -- cuda
//! ncu --set full --kernel-name lbm_stream_collide --launch-count 5 \
//!     target/release/examples/ab_bench.exe cuda small
//! ```
//!
//! Both backends are driven through [`ad_cuda::LbmBackend`], so the loop below
//! is literally the same code for each — no chance of one getting an extra
//! synchronisation or a different batch size.
//!
//! # What the two numbers per row mean
//!
//! *Wall* is the host-side minimum over the batches, and the minimum rather than
//! the mean on purpose: contention only ever makes a batch slower, so the
//! minimum is the closest thing to an uncontended sample. *GPU* is the device
//! timer — timestamp queries for wgpu, events for CUDA — smoothed identically on
//! both sides. They should agree to a few percent; if they do not, the
//! accounting is wrong, not the kernel.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!(
        "ab_bench needs the `cuda` feature:\n  \
         cargo run --release -p ad-cuda --features cuda --example ab_bench"
    );
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    bench::run()
}

#[cfg(feature = "cuda")]
mod bench {
    use ad_cuda::{Backend, CudaSolver, LbmBackend};
    use ad_gpu::types::{flags, DdfPrecision, Grid, VelocitySet};
    use ad_solver::{CollisionModel, Solver, SolverConfig};
    use anyhow::{Context as _, Result};
    use glam::{UVec3, Vec3};
    use std::time::Instant;

    /// Steps per timed batch, and how many batches. Overridable with
    /// `AD_BENCH_BATCH` / `AD_BENCH_BATCHES`.
    ///
    /// One batch has to be long enough that launch overhead and the batch's own
    /// synchronisation are negligible against the work — at the small grid a
    /// step is ~0.7 ms, so ten of them is 7 ms against ~30 us of overhead.
    ///
    /// The batch *count* matters for a different reason. On a desktop the GPU is
    /// never truly idle, and a traced run shows this kernel's duration spread
    /// from 0.72 ms (p10 0.73) up to 2.10 ms with a heavy tail: the mean sits at
    /// 75% of roofline while the median is 91% and the fastest launches reach
    /// 95%. The tail is other clients, not the kernel. Averaging keeps the tail;
    /// taking a *minimum* over many batches throws it away, and more batches
    /// means a better chance of catching a genuinely uncontended one. 20 is
    /// enough for a quick read, a few hundred for a number worth quoting.
    const DEFAULT_BATCH: u32 = 10;
    const DEFAULT_BATCHES: u32 = 20;

    fn env_u32(name: &str, default: u32) -> u32 {
        match std::env::var(name) {
            Ok(v) => v.trim().parse().unwrap_or_else(|_| {
                eprintln!("note: {name}={v:?} is not a positive integer; using {default}");
                default
            }),
            Err(_) => default,
        }
        .max(1)
    }

    /// The grids. The first matches `throughput_at_a_realistic_grid` in
    /// `validation/lbm/gpu.rs` exactly, so this can be read against the numbers
    /// already in the project; the second is large enough that the working set
    /// leaves any cache argument behind — 25 M cells is 1.9 GiB of FP32 DDFs
    /// against 72 MB of L2.
    fn grids(which: &str) -> Vec<(&'static str, UVec3)> {
        let small = ("small", UVec3::new(256, 128, 128));
        let large = ("large", UVec3::new(384, 256, 256));
        match which {
            "small" => vec![small],
            "large" => vec![large],
            _ => vec![small, large],
        }
    }

    fn config() -> SolverConfig {
        SolverConfig {
            set: VelocitySet::D3Q19,
            precision: DdfPrecision::Fp32,
            collision: CollisionModel::Trt,
            tau0: 0.55,
            smagorinsky_c: 0.11,
            initial_velocity: Vec3::new(0.05, 0.0, 0.0),
            // Open on every axis, so the padded halo and the boundary branches
            // are present exactly as in a production run. A fully periodic box
            // would flatter both backends equally, but it would not be the
            // kernel the application executes.
            periodic: [false; 3],
            workgroup_size: 64,
            ..Default::default()
        }
    }

    struct Row {
        backend: &'static str,
        grid: &'static str,
        cells: u64,
        /// Best per-step time over all batches. Contention only ever makes a
        /// batch slower, so this is the closest thing to an uncontended sample.
        wall_ms: f64,
        gpu_ms: Option<f64>,
        mlups: Option<f64>,
        roofline: Option<f64>,
        bytes_per_step: u64,
    }

    impl Row {
        /// Roofline percentage implied by the *best* wall-clock batch, against
        /// the device peak. Uses the backend's own traffic model, which is the
        /// same 157 B/cell model on both sides.
        fn wall_roofline(&self, peak: f64) -> f64 {
            self.bytes_per_step as f64 / (self.wall_ms * 1e-3) / peak
        }
        fn wall_mlups(&self) -> f64 {
            self.cells as f64 / (self.wall_ms * 1e-3) / 1e6
        }
    }

    /// Warm up, time the wall clock, then time the device. Identical for both
    /// backends because it only ever sees [`LbmBackend`].
    fn measure(
        b: &mut dyn LbmBackend,
        backend: &'static str,
        grid: &'static str,
        cells: u64,
        batch: u32,
        batches: u32,
    ) -> Row {
        // Shader/PTX compile, first touch, clock ramp.
        b.step(batch);
        b.wait_idle();

        // Wall clock with profiling off, so the measurement is of batched
        // submission with no per-call instrumentation.
        let mut wall_ms = f64::INFINITY;
        for _ in 0..batches {
            let t = Instant::now();
            b.step(batch);
            b.wait_idle();
            wall_ms = wall_ms.min(t.elapsed().as_secs_f64() * 1e3 / batch as f64);
        }

        // Then the device timers. Every timed call must batch the same number of
        // steps: both backends average pass times, and one pass holds all `batch`
        // launches, so mixing batch lengths averages incommensurable numbers.
        b.set_profiling(true);
        for _ in 0..batches {
            b.step(batch);
            b.wait_idle();
            b.collect_profiling();
        }
        b.set_profiling(false);

        Row {
            backend,
            grid,
            cells,
            wall_ms,
            gpu_ms: b.ms_per_step(),
            mlups: b.mlups(),
            roofline: b.roofline_fraction(),
            bytes_per_step: b.bytes_per_step(),
        }
    }

    pub fn run() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).map(|a| a.to_lowercase()).collect();
        let which_grid =
            args.iter().find(|a| *a == "small" || *a == "large").cloned().unwrap_or_default();
        // Naming no backend means both, so `ab_bench small` is not silently a
        // no-op. Naming one runs only that one, which is what `ncu` wants.
        let named = |n: &str| args.iter().any(|a| a == n);
        let any_backend = named("wgpu") || named("cuda") || named("both");
        let run_wgpu = !any_backend || named("wgpu") || named("both");
        let run_cuda = !any_backend || named("cuda") || named("both");

        let batch = env_u32("AD_BENCH_BATCH", DEFAULT_BATCH);
        let batches = env_u32("AD_BENCH_BATCHES", DEFAULT_BATCHES);
        println!(
            "AeroDuct backend A/B - D3Q19, TRT (Lambda = 3/16), Smagorinsky Cs = 0.11, FP32,\n\
             open box (padded halo on every axis), {batch} steps per timed batch, \
             {batches} batches.\n\
             Run this on an idle GPU. The kernel is bandwidth-bound; a second client \
             invalidates every number below.\n"
        );

        let mut rows = Vec::new();
        // The device peak both backends' percentages divide by. wgpu looks it up
        // by adapter name; CUDA derives it from the memory clock and bus width.
        // On the RTX 4090 the two agree at 1008 GB/s, which is the check that
        // the derivation is right rather than merely plausible.
        let mut peak: Option<f64> = None;
        for (name, dims) in grids(&which_grid) {
            let grid = Grid { dims, dx_mm: 1.0, origin_mm: Vec3::ZERO };
            let mask = vec![flags::FLUID; (dims.x as usize) * (dims.y as usize) * (dims.z as usize)];
            let cfg = config();

            // Strictly one at a time. Each solver is dropped - freeing its VRAM
            // and its context - before the next is built, so the two never share
            // the memory controller and neither is measured while the other's
            // allocation is still resident.
            if run_wgpu {
                let ctx = ad_gpu::GpuContext::new_blocking(None)
                    .map_err(|e| anyhow::anyhow!("no wgpu adapter: {e}"))?;
                if !ctx.caps.timestamps {
                    eprintln!("note: adapter has no timestamp queries; wgpu GPU timings will be blank");
                }
                let mut s = Solver::new(&ctx, grid, &mask, &[], cfg)
                    .context("building the wgpu solver")?;
                let cells = s.domain.padded_cell_count();
                peak = peak.or(ctx.caps.peak_bandwidth);
                rows.push(measure(&mut s, "wgpu/Vulkan", name, cells, batch, batches));
                s.wait_idle();
            }

            if run_cuda {
                if !Backend::Cuda.available() {
                    eprintln!("note: no CUDA device; skipping the CUDA rows");
                } else {
                    let mut s =
                        CudaSolver::new(grid, &mask, &[], cfg).context("building the CUDA solver")?;
                    let cells = s.domain.padded_cell_count();
                    peak = peak.or(s.device().peak_bandwidth);
                    println!(
                        "CUDA device: {} (sm_{}{}, {} SMs, peak {:.0} GB/s)",
                        s.device().name,
                        s.device().compute_capability.0,
                        s.device().compute_capability.1,
                        s.device().multiprocessors,
                        s.device().peak_bandwidth.unwrap_or(0.0) / 1e9,
                    );
                    rows.push(measure(&mut s, "CUDA", name, cells, batch, batches));
                    s.wait_idle();
                }
            }
        }

        println!(
            "\npeak DRAM bandwidth {:.0} GB/s\n\n\
             {:<12} {:<6} {:>8} {:>7} {:>9} {:>9} {:>10} {:>10} {:>9} {:>8}",
            peak.unwrap_or(0.0) / 1e9,
            "backend",
            "grid",
            "Mcells",
            "B/cell",
            "GPU ms",
            "best ms",
            "GPU MLUPS",
            "best MLUPS",
            "GPU %rl",
            "best %rl",
        );
        for r in &rows {
            let opt = |v: Option<f64>, scale: f64, dp: usize| {
                v.map(|x| format!("{:.*}", dp, x * scale)).unwrap_or_else(|| "-".to_string())
            };
            println!(
                "{:<12} {:<6} {:>8.2} {:>7} {:>9} {:>9.3} {:>10} {:>10.0} {:>9} {:>8}",
                r.backend,
                r.grid,
                r.cells as f64 / 1e6,
                r.bytes_per_step / r.cells,
                opt(r.gpu_ms, 1.0, 3),
                r.wall_ms,
                opt(r.mlups, 1.0, 0),
                r.wall_mlups(),
                opt(r.roofline, 100.0, 1),
                peak.map(|p| format!("{:.1}", r.wall_roofline(p) * 100.0))
                    .unwrap_or_else(|| "-".to_string()),
            );
        }
        println!(
            "\n\"GPU\" columns are the device timer, smoothed identically on both backends, so they\n\
             keep the contention tail; \"best\" is the minimum over the {batches} batches, which\n\
             discards it. On a desktop the two differ by a lot and the minimum is the honest one."
        );

        // The comparison, stated rather than left to the reader. If CUDA merely
        // matches, that confirms the roofline analysis and is the expected
        // result; if it is materially faster, that is a finding about the wgpu
        // path rather than about CUDA.
        for (name, _) in grids(&which_grid) {
            let pick = |b: &str| rows.iter().find(|r| r.grid == name && r.backend == b);
            let (Some(w), Some(c)) = (pick("wgpu/Vulkan"), pick("CUDA")) else { continue };
            if let (Some(wg), Some(cg)) = (w.gpu_ms, c.gpu_ms) {
                println!(
                    "\n{name}: CUDA is {:.3}x the wgpu step time on the device timer \
                     ({:+.1}% throughput).",
                    cg / wg,
                    (wg / cg - 1.0) * 100.0
                );
            }
            println!(
                "{name}: CUDA is {:.3}x the wgpu step time on the best batch ({:+.1}% throughput).",
                c.wall_ms / w.wall_ms,
                (w.wall_ms / c.wall_ms - 1.0) * 100.0
            );
        }
        Ok(())
    }
}

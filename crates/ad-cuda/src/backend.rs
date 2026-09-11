//! The device side: NVRTC compilation, buffers, the step loop and the timing.
//!
//! # Shape of a step
//!
//! One kernel launch. Esoteric Pull streams in place, so there is no ping-pong
//! buffer to swap and no separate streaming pass — the load *is* the stream. The
//! only per-step state is the parity bit, which rides in the by-value parameter
//! block, so a run of `n` steps is `n` launches on one stream with no host
//! involvement beyond building the argument list.
//!
//! This is the same shape as `ad_solver::solver::Solver::step_batch`, which is
//! the point: the A/B is only worth anything if both sides do the same work in
//! the same order.
//!
//! # Buffers
//!
//! | buffer | contents |
//! |---|---|
//! | `ddf` | `q * padded_cells` FP32 shifted populations, SoA |
//! | `flags` | one `ad_gpu::types::flags` byte per padded cell, four per word |
//! | `link_mask` | one `u32` per padded cell; bit `i` = "flip the load parity" |
//! | `macro_buf` | `(u.x, u.y, u.z, rho)` per *interior* cell |
//!
//! The DDFs are one allocation rather than `q`, because CUDA has no equivalent
//! of wgpu's 2 GiB binding clamp. The bytes are identical either way — plane `i`
//! of the flat array is byte-for-byte what `DdfBuffers::buffers[i]` holds — so
//! traffic, coalescing and the roofline denominator are unchanged. See
//! [`crate::kernel`].
//!
//! # Timing
//!
//! CUDA events around a whole `step(n)` call, divided by `n`, then folded into
//! the same 0.05 exponential moving average `ad_gpu::Profiler` uses, so
//! `ms_per_step`, `mlups` and `roofline_fraction` mean the same thing on both
//! backends and can be printed side by side without an asterisk.
//!
//! Timing is off by default. An event pair costs a stream synchronisation point
//! per batch, which is exactly what a batch sweep does not want.
//!
//! The moving average is not the whole story, and on a desktop it is the
//! misleading part. A traced run of 410 launches at 4.36 M cells spreads from
//! 0.716 ms (94.8% of the 1008 GB/s roofline) through a median of 0.747 ms
//! (90.9%) to a 2.096 ms tail, with a mean of 0.910 ms (74.7%). The floor is
//! hard and the tail is long, which is contention from other GPU clients rather
//! than anything the kernel varies. Prefer the *minimum* over many batches when
//! quoting a number; `examples/ab_bench.rs` reports both and says which is
//! which. Launch gaps in the same trace are 1.31 us median against a 747 us
//! kernel - 0.18% - so per-step dispatch cost is not a factor either way.

use std::sync::Arc;

use ad_gpu::types::{BoundaryLink, DdfPrecision, Grid};
use ad_solver::boundary::{LinkTable, PaddedDomain};
use ad_solver::solver::max_steps_per_submit_for;
use ad_solver::SolverConfig;
use anyhow::{bail, Context as _, Result};
use cudarc::driver::sys::{CUdevice_attribute, CUevent_flags};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use glam::{UVec3, Vec3};

use crate::kernel::{self, CudaParams, KernelSpec};
use crate::{Backend, LbmBackend};

// The parameter block goes to the device by value, as a plain C struct. It is
// `#[repr(C)]` and made only of `u32`/`f32`, and
// `param_struct_matches_the_generated_c_struct` checks it against the emitted
// `struct Params` field by field, which is what makes this sound.
unsafe impl DeviceRepr for CudaParams {}

/// Number of CUDA devices, or zero when no driver is present.
pub(crate) fn device_count() -> Result<usize> {
    match CudaContext::device_count() {
        Ok(n) => Ok(n.max(0) as usize),
        // A machine with no NVIDIA driver reports an error rather than zero, and
        // that is not a failure worth propagating to a caller who only asked
        // whether the backend was usable.
        Err(e) => {
            log::debug!("CUDA device enumeration failed: {e}");
            Ok(0)
        }
    }
}

/// What the device says about itself, including the peak bandwidth the roofline
/// percentage divides by.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    /// Bytes per second, from the memory clock and bus width rather than a
    /// lookup table.
    ///
    /// `2 * clock * width / 8`: the factor of two is DDR. On the RTX 4090 the
    /// driver reports 10,501,000 kHz over a 384-bit bus, giving
    /// **1008.1 GB/s** — which is exactly the figure CONTRACT.md records for the
    /// verified hardware baseline, and exactly what `ad_gpu::context`'s name
    /// lookup returns. Deriving it means the two backends' roofline percentages
    /// share a denominator without sharing a hardcoded table.
    ///
    /// `None` when the driver declines to report a memory clock, which newer
    /// drivers may do on some parts; the percentage is then simply unavailable
    /// rather than silently wrong.
    pub peak_bandwidth: Option<f64>,
    pub multiprocessors: i32,
}

impl DeviceInfo {
    fn query(ctx: &Arc<CudaContext>) -> Result<Self> {
        let name = ctx.name().context("querying the CUDA device name")?;
        let compute_capability = ctx
            .compute_capability()
            .context("querying the compute capability")?;
        let clock_khz = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
            .unwrap_or(0);
        let bus_bits = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
            .unwrap_or(0);
        let peak_bandwidth = (clock_khz > 0 && bus_bits > 0)
            .then(|| 2.0 * clock_khz as f64 * 1.0e3 * bus_bits as f64 / 8.0);
        let multiprocessors = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(0);
        Ok(Self {
            name,
            compute_capability,
            peak_bandwidth,
            multiprocessors,
        })
    }
}

/// One pass' worth of timing, with the same smoothing `ad_gpu::Profiler` applies.
#[derive(Debug, Clone, Copy, Default)]
struct Timing {
    mean_ms: f64,
    samples: u64,
}

impl Timing {
    /// First sample sets the mean outright; after that a 0.05 EWMA. Identical to
    /// `ad_gpu::profiler`, deliberately — a benchmark that smoothed the two
    /// backends differently would be comparing filters, not kernels.
    fn push(&mut self, ms: f64) {
        if self.samples == 0 {
            self.mean_ms = ms;
        } else {
            self.mean_ms += (ms - self.mean_ms) * 0.05;
        }
        self.samples += 1;
    }
}

/// The CUDA lattice-Boltzmann solver. Headless: no textures, no rendering.
///
/// Mirrors `ad_solver::Solver` closely enough that the two can be driven from
/// one loop through [`crate::LbmBackend`], and produces the macroscopic field in
/// exactly the format `Solver::read_macroscopic` returns, so a cross-backend
/// comparison is a direct element-wise difference with nothing to reconcile.
pub struct CudaSolver {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    // The functions borrow the module, so it has to outlive them.
    _module: Arc<CudaModule>,
    f_init: CudaFunction,
    f_step: CudaFunction,
    f_macro: CudaFunction,

    pub grid: Grid,
    pub domain: PaddedDomain,
    pub links: LinkTable,
    cfg: SolverConfig,
    spec: KernelSpec,
    info: DeviceInfo,

    ddf: CudaSlice<f32>,
    flags: CudaSlice<u32>,
    link_mask: CudaSlice<u32>,
    macro_buf: CudaSlice<f32>,

    profiling: bool,
    /// A recorded but not yet collected event pair, with the number of steps it
    /// spans. Kept rather than waited on, so `step` does not stall.
    pending: Option<(CudaEvent, CudaEvent, u32)>,
    timing: Timing,
    profiled_steps: u32,
    steps: u64,
}

impl CudaSolver {
    /// Build a solver for `grid`, with one [`ad_gpu::types::flags`] byte per
    /// interior cell.
    ///
    /// `boundary_links` is accepted and remapped for parity with
    /// `ad_solver::Solver`, whose Wave-3 interpolated bounce-back will need it.
    /// Nothing in this kernel consumes it yet, so unlike the wgpu path it is not
    /// uploaded — a device allocation nothing reads is not a data path, it is
    /// wasted VRAM.
    pub fn new(
        grid: Grid,
        mask: &[u8],
        boundary_links: &[BoundaryLink],
        cfg: SolverConfig,
    ) -> Result<Self> {
        if cfg.precision != DdfPrecision::Fp32 {
            bail!(
                "the CUDA backend stores FP32 only; {:?} is implemented in ad_solver::precision \
                 for the wgpu path. See crates/ad-cuda/src/kernel.rs.",
                cfg.precision
            );
        }

        let domain = PaddedDomain::new(grid.dims, cfg.periodic, mask, cfg.set);
        let links = LinkTable::remap(boundary_links, grid.dims, &domain);
        let cells = domain.padded_cell_count();
        let q = cfg.set.q() as u64;

        // 19 x 312 M cells x 4 B is 23.7 GB, which no consumer card holds, but a
        // request that large should fail here with an intelligible message
        // rather than as an opaque CUDA_ERROR_OUT_OF_MEMORY.
        let ddf_elems = cells
            .checked_mul(q)
            .and_then(|n| usize::try_from(n).ok())
            .with_context(|| {
                format!(
                    "grid {}x{}x{} overflows an allocation",
                    grid.dims.x, grid.dims.y, grid.dims.z
                )
            })?;

        let ctx = CudaContext::new(0).context(
            "opening CUDA device 0 - is an NVIDIA driver installed and a device present?",
        )?;
        let info = DeviceInfo::query(&ctx)?;
        let stream = ctx.default_stream();

        let spec = KernelSpec {
            set: cfg.set,
            collision: cfg.collision,
            block_x: cfg.workgroup_size.max(1),
        };
        let (module, f_init, f_step, f_macro) = compile(&ctx, spec, &info)?;

        let flag_words = pack_flags(&domain.flags);
        let flags = stream
            .clone_htod(&flag_words)
            .context("uploading the flag bytes")?;
        let link_mask = stream
            .clone_htod(&domain.link_mask)
            .context("uploading the link mask")?;
        // Explicit, and not redundant. cudarc issues host copies with
        // `cuMemcpy*Async` and only synchronises for *pinned* host memory - a
        // plain `Vec` gets `SyncOnDrop::Sync(None)`, i.e. nothing. The driver
        // does treat a pageable copy as synchronous in practice, but `flag_words`
        // is a local about to be dropped, and "the DMA has probably already read
        // it" is not a thing to rely on for a use-after-free.
        stream
            .synchronize()
            .context("waiting for the static uploads")?;

        let ddf = stream.alloc_zeros::<f32>(ddf_elems).with_context(|| {
            format!(
                "allocating {:.2} GiB of distribution functions ({:.1} M cells x {q})",
                (ddf_elems * 4) as f64 / (1u64 << 30) as f64,
                cells as f64 / 1e6
            )
        })?;
        let interior = usize::try_from(domain.interior_cell_count() * 4)
            .context("interior grid overflows an allocation")?;
        let macro_buf = stream
            .alloc_zeros::<f32>(interior)
            .context("allocating the macroscopic readback buffer")?;

        log::info!(
            "CUDA backend on {} (sm_{}{}, {} SMs): {:?} FP32, {:.1} M padded cells, \
             {:.2} GiB of DDFs",
            info.name,
            info.compute_capability.0,
            info.compute_capability.1,
            info.multiprocessors,
            cfg.set,
            cells as f64 / 1e6,
            (ddf_elems * 4) as f64 / (1u64 << 30) as f64,
        );

        let mut me = Self {
            ctx,
            stream,
            _module: module,
            f_init,
            f_step,
            f_macro,
            grid,
            domain,
            links,
            cfg,
            spec,
            info,
            ddf,
            flags,
            link_mask,
            macro_buf,
            profiling: false,
            pending: None,
            timing: Timing::default(),
            profiled_steps: 1,
            steps: 0,
        };
        me.reset()?;
        Ok(me)
    }

    pub fn config(&self) -> &SolverConfig {
        &self.cfg
    }

    pub fn device(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn kernel_spec(&self) -> KernelSpec {
        self.spec
    }

    pub fn steps_taken(&self) -> u64 {
        self.steps
    }

    /// Physical time simulated so far, from `SolverConfig::dt_s`.
    pub fn sim_time_seconds(&self) -> f64 {
        self.steps as f64 * self.cfg.dt_s
    }

    /// Apply new parameters. Errors when the change would need a different
    /// kernel, matching `ad_solver::Solver::update`.
    pub fn update(&mut self, cfg: SolverConfig) -> Result<()> {
        if self.cfg.needs_rebuild(&cfg) {
            bail!(
                "changing the velocity set, storage precision, collision model, periodicity or \
                 block size requires rebuilding the CUDA solver"
            );
        }
        self.cfg = cfg;
        Ok(())
    }

    /// Hot-update just the inlet velocity, in lattice units.
    ///
    /// Free here, unlike the wgpu path: the parameters ride in the launch
    /// argument list rather than a uniform buffer, so there is nothing to write.
    pub fn set_inlet_velocity(&mut self, v: Vec3) {
        self.cfg.inlet_velocity = v;
    }

    /// Reinitialise every fluid cell to equilibrium and zero the step counter.
    pub fn reset(&mut self) -> Result<()> {
        self.steps = 0;
        // Zeroing is not strictly required - every slot the solver reads is
        // written by `init` - but it costs one clear per reset and removes any
        // chance of a stray value in an unread slot surviving a future change.
        self.stream
            .memset_zeros(&mut self.ddf)
            .context("clearing the distribution functions")?;

        let params = self.params(0);
        let cfg = self.launch_config(self.domain.padded);
        let stream = self.stream.clone();
        let mut b = stream.launch_builder(&self.f_init);
        b.arg(&params).arg(&mut self.ddf).arg(&self.flags);
        // SAFETY: the argument list matches `lbm_init`'s signature - checked by
        // `param_struct_matches_the_generated_c_struct` for the parameter block,
        // and by the generator for the pointers - and every buffer is sized to
        // the padded grid the parameters describe.
        unsafe { b.launch(cfg) }.context("launching lbm_init")?;
        Ok(())
    }

    /// Advance `n` steps.
    ///
    /// # Why the work per launch batch is bounded
    ///
    /// Windows kills a GPU job that does not yield within roughly two seconds
    /// (Timeout Detection and Recovery) and resets the driver. That cost the
    /// wgpu path a 58-minute resolution run before
    /// `ad_solver::solver::max_steps_per_submit_for` existed, because one
    /// `submit` holding 500 dispatches is *one* command packet.
    ///
    /// CUDA is structurally safer here — each launch is its own packet, and at
    /// the grids this app runs no single stream-collide is within two orders of
    /// magnitude of the watchdog — but the same budget is applied anyway, from
    /// the same function, for two reasons. It bounds how far the host can run
    /// ahead of the device (the launch queue is finite, and past its depth
    /// `cuLaunchKernel` blocks unpredictably), and it means a future change that
    /// fuses launches cannot reintroduce the failure. The cost is one stream
    /// synchronisation per ~10^9 cell-steps, which at the interactive tier is
    /// once every 150 steps and is unmeasurable next to the work it carries.
    pub fn step(&mut self, n: u32) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if self.profiling {
            // Collect anything outstanding before overwriting the slot, or a
            // batch's timing is silently dropped.
            self.collect(true);
            let start = self.new_timing_event()?;
            start
                .record(&self.stream)
                .context("recording the start event")?;
            self.launch_steps(n)?;
            let end = self.new_timing_event()?;
            end.record(&self.stream)
                .context("recording the end event")?;
            self.pending = Some((start, end, n));
            self.profiled_steps = n;
        } else {
            self.launch_steps(n)?;
        }
        Ok(())
    }

    fn new_timing_event(&self) -> Result<CudaEvent> {
        // CU_EVENT_DEFAULT, not the `None` default: cudarc's default is
        // CU_EVENT_DISABLE_TIMING, and `elapsed_ms` on such a pair fails rather
        // than returning a wrong number - but only at the point of reading it,
        // by which time the run is over.
        self.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .context("creating a CUDA timing event")
    }

    fn launch_steps(&mut self, n: u32) -> Result<()> {
        let per_batch = max_steps_per_submit_for(self.domain.padded_cell_count());
        let cfg = self.launch_config(self.domain.padded);
        let mut left = n;
        let mut done = 0u32;
        while left > 0 {
            let chunk = left.min(per_batch);
            for k in 0..chunk {
                let parity = ((self.steps + (done + k) as u64) % 2) as u32;
                let params = self.params(parity);
                let stream = self.stream.clone();
                let mut b = stream.launch_builder(&self.f_step);
                b.arg(&params)
                    .arg(&mut self.ddf)
                    .arg(&self.flags)
                    .arg(&self.link_mask);
                // SAFETY: as `reset`. The argument list matches
                // `lbm_stream_collide`, and Esoteric Pull makes the in-place
                // update race-free - every (address, half) is owned by exactly
                // one cell for the whole step - which is what lets one buffer be
                // both the input and the output of a single launch.
                unsafe { b.launch(cfg) }.context("launching lbm_stream_collide")?;
            }
            done += chunk;
            left -= chunk;
            if left > 0 {
                self.stream.synchronize().context("draining a step batch")?;
            }
        }
        self.steps += n as u64;
        Ok(())
    }

    /// Turn CUDA event timing on or off. Off by default: an event pair per batch
    /// is a synchronisation point, and a batch sweep does not want one.
    pub fn set_profiling(&mut self, on: bool) {
        self.profiling = on;
        if !on {
            self.collect(true);
        }
    }

    pub fn is_profiling(&self) -> bool {
        self.profiling
    }

    /// Fold any completed event pair into the moving average.
    ///
    /// Non-blocking, like `Solver::collect_profiling`: if the batch is still in
    /// flight the pair is put back and picked up next time.
    pub fn collect_profiling(&mut self) {
        self.collect(false);
    }

    fn collect(&mut self, block: bool) {
        let Some((start, end, steps)) = self.pending.take() else {
            return;
        };
        if !block && !end.is_complete() {
            self.pending = Some((start, end, steps));
            return;
        }
        match start.elapsed_ms(&end) {
            Ok(ms) => {
                self.profiled_steps = steps.max(1);
                self.timing.push(ms as f64);
            }
            // A failed elapsed_ms means the pair is unusable, not that the run
            // is broken. Drop the sample and say so rather than poisoning the
            // average or taking the process down over a statistic.
            Err(e) => log::warn!("CUDA event timing failed, sample dropped: {e}"),
        }
    }

    /// Block until every launched step has finished on the device.
    pub fn wait_idle(&self) {
        if let Err(e) = self.stream.synchronize() {
            log::error!("CUDA stream synchronise failed: {e}");
        }
    }

    /// Read the macroscopic field back as `(u.x, u.y, u.z, rho)` per interior
    /// cell, X-fastest — byte-identical in meaning to
    /// `ad_solver::Solver::read_macroscopic`.
    pub fn read_macroscopic(&mut self) -> Result<Vec<[f32; 4]>> {
        let parity = (self.steps % 2) as u32;
        let params = self.params(parity);
        let cfg = self.launch_config(self.domain.interior);
        {
            let stream = self.stream.clone();
            let mut b = stream.launch_builder(&self.f_macro);
            b.arg(&params)
                .arg(&self.ddf)
                .arg(&self.flags)
                .arg(&self.link_mask)
                .arg(&mut self.macro_buf);
            // SAFETY: as `reset`; `lbm_macroscopic` only reads the populations.
            unsafe { b.launch(cfg) }.context("launching lbm_macroscopic")?;
        }
        let flat: Vec<f32> = self
            .stream
            .clone_dtoh(&self.macro_buf)
            .context("reading the macroscopic field back")?;
        // As in `new`: cudarc's device-to-host copy into a pageable `Vec` is
        // issued asynchronously and synchronised by nothing. Reading `flat`
        // before this line would be a race with the copy that fills it - and one
        // that would usually appear to work, which is the worst kind.
        self.stream
            .synchronize()
            .context("waiting for the macroscopic readback")?;
        Ok(flat
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect())
    }

    /// Bytes of memory traffic one step moves, for the roofline percentage.
    ///
    /// Identical accounting to `ad_solver::Solver::bytes_per_step`, including
    /// the 4 B/cell for the link mask that `ad_gpu::bytes_per_cell` does not
    /// budget. Same numerator, same denominator, so the two backends' roofline
    /// percentages are directly comparable — which is the entire point.
    pub fn bytes_per_step(&self) -> u64 {
        let (_, traffic) = ad_gpu::bytes_per_cell(self.cfg.set, self.cfg.precision);
        (traffic + 4) * self.domain.padded_cell_count()
    }

    /// Milliseconds per step, from the last collected event pair.
    pub fn ms_per_step(&self) -> Option<f64> {
        (self.timing.samples > 0).then(|| self.timing.mean_ms / self.profiled_steps.max(1) as f64)
    }

    /// Achieved million lattice updates per second.
    pub fn mlups(&self) -> Option<f64> {
        let ms = self.ms_per_step()?;
        if ms <= 0.0 {
            return None;
        }
        Some(self.domain.padded_cell_count() as f64 / (ms * 1e-3) / 1e6)
    }

    /// Achieved bandwidth as a fraction of the device's peak. For a
    /// bandwidth-bound kernel this is the only performance number that means
    /// anything on its own.
    pub fn roofline_fraction(&self) -> Option<f64> {
        let ms = self.ms_per_step()?;
        let peak = self.info.peak_bandwidth?;
        if ms <= 0.0 || peak <= 0.0 {
            return None;
        }
        Some(self.bytes_per_step() as f64 / (ms * 1e-3) / peak)
    }

    /// One-line summary in the same shape `ad_gpu::Profiler::summary` produces.
    pub fn profiler_summary(&self) -> String {
        match (self.ms_per_step(), self.mlups(), self.roofline_fraction()) {
            (Some(ms), Some(m), Some(r)) => {
                format!(
                    "stream_collide: {ms:.3} ms/step, {m:.0} MLUPS, {:.0}% of roofline",
                    r * 100.0
                )
            }
            (Some(ms), Some(m), None) => format!("stream_collide: {ms:.3} ms/step, {m:.0} MLUPS"),
            _ => "stream_collide: no timing collected".to_string(),
        }
    }

    pub fn padded_dims(&self) -> UVec3 {
        self.domain.padded
    }

    /// Block `(block_x, 1, 1)`, grid `(ceil(dims.x / block_x), dims.y, dims.z)`.
    ///
    /// The same decomposition the WGSL dispatch uses. X is the fastest-varying
    /// axis, so a block covers a contiguous run of cells and consecutive lanes
    /// touch consecutive addresses in every direction plane.
    fn launch_config(&self, dims: UVec3) -> LaunchConfig {
        let b = self.spec.block_x.max(1);
        LaunchConfig {
            grid_dim: (dims.x.div_ceil(b), dims.y, dims.z),
            block_dim: (b, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn params(&self, parity: u32) -> CudaParams {
        let c = &self.cfg;
        let periodic =
            (c.periodic[0] as u32) | ((c.periodic[1] as u32) << 1) | ((c.periodic[2] as u32) << 2);
        let d = self.domain.padded;
        let i = self.domain.interior;
        let o = self.domain.offset;
        CudaParams {
            dims_x: d.x,
            dims_y: d.y,
            dims_z: d.z,
            step_parity: parity,
            interior_x: i.x,
            interior_y: i.y,
            interior_z: i.z,
            cell_count: self.domain.padded_cell_count() as u32,
            offset_x: o.x,
            offset_y: o.y,
            offset_z: o.z,
            flag_words: self.domain.padded_cell_count().div_ceil(4) as u32,
            inlet_velocity_x: c.inlet_velocity.x,
            inlet_velocity_y: c.inlet_velocity.y,
            inlet_velocity_z: c.inlet_velocity.z,
            tau0: c.tau0,
            initial_velocity_x: c.initial_velocity.x,
            initial_velocity_y: c.initial_velocity.y,
            initial_velocity_z: c.initial_velocity.z,
            trt_lambda: c.trt_lambda,
            body_force_x: c.body_force.x,
            body_force_y: c.body_force.y,
            body_force_z: c.body_force.z,
            smagorinsky_c: c.smagorinsky_c,
            tau_max: c.tau_max,
            outflow_velocity: c.outflow_velocity,
            rho_ref: c.rho_ref,
            sponge_strength: c.sponge_strength,
            sponge_cells: c.sponge_cells,
            periodic,
            total_steps: self.steps as u32,
            outlet_anti_bounce_back: c.outlet_anti_bounce_back as u32,
            outlet_normal_x: c.outlet_normal.x,
            outlet_normal_y: c.outlet_normal.y,
            outlet_normal_z: c.outlet_normal.z,
            inlet_normal_x: c.inlet_normal.x,
            inlet_normal_y: c.inlet_normal.y,
            inlet_normal_z: c.inlet_normal.z,
        }
    }
}

impl LbmBackend for CudaSolver {
    fn backend(&self) -> Backend {
        Backend::Cuda
    }
    fn steps_taken(&self) -> u64 {
        self.steps
    }
    /// Panic-free: a launch failure is logged and the step count stops
    /// advancing, because the trait exists for benchmark loops and a `Result`
    /// there would only be unwrapped.
    fn step(&mut self, n: u32) {
        if let Err(e) = CudaSolver::step(self, n) {
            log::error!("CUDA step failed: {e:#}");
        }
    }
    fn reset(&mut self) {
        if let Err(e) = CudaSolver::reset(self) {
            log::error!("CUDA reset failed: {e:#}");
        }
    }
    fn read_macroscopic(&mut self) -> Result<Vec<[f32; 4]>> {
        CudaSolver::read_macroscopic(self)
    }
    fn set_profiling(&mut self, on: bool) {
        CudaSolver::set_profiling(self, on)
    }
    fn collect_profiling(&mut self) {
        CudaSolver::collect_profiling(self)
    }
    fn wait_idle(&self) {
        CudaSolver::wait_idle(self)
    }
    fn ms_per_step(&self) -> Option<f64> {
        CudaSolver::ms_per_step(self)
    }
    fn mlups(&self) -> Option<f64> {
        CudaSolver::mlups(self)
    }
    fn roofline_fraction(&self) -> Option<f64> {
        CudaSolver::roofline_fraction(self)
    }
    fn bytes_per_step(&self) -> u64 {
        CudaSolver::bytes_per_step(self)
    }
    fn cell_count(&self) -> u64 {
        self.domain.padded_cell_count()
    }
}

/// Compile the generated source with NVRTC and pull the three entry points out.
///
/// Runtime compilation rather than a build-time `nvcc` step, for three reasons:
/// the kernel is generated from Rust tables that a build script would have to
/// duplicate, the source depends on run-time configuration (velocity set,
/// collision operator, block size) the way the WGSL does, and it means the crate
/// builds with headers alone and never needs an import library.
///
/// The architecture is the device's own, as `compute_XX`, so NVRTC emits PTX and
/// the driver JITs it. Asking for `sm_XX` would produce a cubin instead, which
/// `nvrtcGetPTX` cannot return.
///
/// **No `--use_fast_math`.** It would change the numerics away from the WGSL and
/// make the cross-backend agreement test measure the flag rather than the
/// algorithm. FMA contraction is left at the CUDA default (`--fmad=true`), which
/// matches what the Vulkan driver does with naga's SPIR-V, since naga emits no
/// `NoContraction` decorations.
fn compile(
    ctx: &Arc<CudaContext>,
    spec: KernelSpec,
    info: &DeviceInfo,
) -> Result<(Arc<CudaModule>, CudaFunction, CudaFunction, CudaFunction)> {
    let src = kernel::generate(spec);
    let (major, minor) = info.compute_capability;
    let mut opts = CompileOptions {
        options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
        name: Some("aeroduct_lbm.cu".to_string()),
        ..Default::default()
    };
    // Setting AD_CUDA_SOURCE_DIR writes the generated source there and compiles
    // with a line table, so Nsight Compute can attribute a stall or a sector
    // count to a *line* rather than only to SASS. NVRTC compiles from a string,
    // so without a real file on disk the profiler has nothing to correlate
    // against - and source correlation is most of why this backend exists.
    //
    // `--generate-line-info` adds a debug section and does not change codegen,
    // so a profiled kernel is the same kernel a timed one is.
    if let Some(dir) = std::env::var_os("AD_CUDA_SOURCE_DIR") {
        let path = std::path::Path::new(&dir).join("aeroduct_lbm.cu");
        match std::fs::write(&path, &src) {
            Ok(()) => {
                opts.name = Some(path.to_string_lossy().into_owned());
                opts.options.push("--generate-line-info".to_string());
                log::info!("generated CUDA source written to {}", path.display());
            }
            Err(e) => log::warn!(
                "AD_CUDA_SOURCE_DIR is set but {} could not be written: {e}",
                path.display()
            ),
        }
    }
    let ptx = compile_ptx_with_opts(&src, opts).map_err(|e| {
        // The NVRTC log is the only useful thing in a compile failure, and it
        // refers to line numbers in generated source nobody has on disk. Dump
        // the source next to the log so the two can be read together.
        let path = std::env::temp_dir().join("aeroduct_lbm.cu");
        let hint = match std::fs::write(&path, &src) {
            Ok(()) => format!(" (generated source written to {})", path.display()),
            Err(_) => String::new(),
        };
        anyhow::anyhow!("NVRTC rejected the generated kernel{hint}: {e}")
    })?;

    let module = ctx
        .load_module(ptx)
        .context("loading the compiled PTX module")?;
    let f_init = module.load_function(kernel::FN_INIT).context("lbm_init")?;
    let f_step = module
        .load_function(kernel::FN_STREAM_COLLIDE)
        .context("lbm_stream_collide")?;
    let f_macro = module
        .load_function(kernel::FN_MACROSCOPIC)
        .context("lbm_macroscopic")?;
    Ok((module, f_init, f_step, f_macro))
}

/// Pack one flag byte per cell into `u32` words, four cells per word.
///
/// Byte-for-byte what `ad_solver::solver::pack_flags` produces, and the kernel's
/// `get_flags` is the inverse; the layout is fixed by `ad_gpu::ddf`'s traffic
/// model budgeting exactly one byte per cell per step.
fn pack_flags(bytes: &[u8]) -> Vec<u32> {
    let mut out = vec![0u32; bytes.len().div_ceil(4)];
    for (i, b) in bytes.iter().enumerate() {
        out[i >> 2] |= (*b as u32) << ((i & 3) * 8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_pack_four_cells_per_word_the_same_way_the_wgpu_path_does() {
        let bytes = [0x01u8, 0x02, 0x04, 0x08, 0x10];
        let words = pack_flags(&bytes);
        assert_eq!(words, vec![0x08040201, 0x00000010]);
        for (i, b) in bytes.iter().enumerate() {
            let got = (words[i >> 2] >> ((i & 3) * 8)) & 0xff;
            assert_eq!(got as u8, *b, "cell {i}");
        }
    }

    /// The measured RTX 4090 numbers, so the derivation cannot drift away from
    /// the figure CONTRACT.md records for the verified baseline.
    #[test]
    fn peak_bandwidth_derivation_reproduces_the_contract_baseline() {
        // Driver-reported: 10,501,000 kHz over a 384-bit bus.
        let bw: f64 = 2.0 * 10_501_000.0 * 1.0e3 * 384.0 / 8.0;
        assert!(
            (bw - 1008.0e9).abs() < 1.0e9,
            "derived {:.1} GB/s, CONTRACT.md says 1008",
            bw / 1e9
        );
    }
}

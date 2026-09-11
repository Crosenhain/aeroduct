//! The metrics adapter — **the one file that connects `ad-metrics`.**
//!
//! # What this does
//!
//! [`DuctMetricsSource`] owns an [`ad_metrics::DuctMetrics`], drives it from the
//! frame loop against the solver's velocity and density textures plus the app's
//! own interior flag mask, and maps the resulting [`MetricsReport`] onto
//! [`ad_ui::view::MetricsView`]. Nothing in `ad-ui` knows any of this happened:
//! the panels only ever see view models, which is what let both crates be
//! written at the same time.
//!
//! The field table, and where each number now comes from:
//!
//! | `MetricsView` field | unit | source |
//! |---|---|---|
//! | `flow_rate` | m^3/s | `report.flow_in` — inlet patch integral of `u . n` |
//! | `pressure_drop` | Pa | `report.total_pressure_drop_pa`, mass-flow weighted |
//! | `loss_coefficient` | - | `report.loss_coefficient.k`, on the inlet bulk |
//! | `uniformity` | - | `report.uniformity.gamma`, Weltens on the outlet patch |
//! | `deflection_deg` | deg | `report.jet.deflection_deg` |
//! | `max_speed` | m/s | `report.peak_speed_ms`, the volume reduction |
//! | `inlet` / `outlet` | mixed | mouth geometry + the two plane readings |
//! | `convergence.residuals` | - | `report.residual`, `log_y = true` |
//! | `convergence.flow_in` / `flow_out` | m^3/s | the two curves whose gap converges |
//! | `convergence.pressure_drop` | Pa | against step |
//! | `convergence.mass_imbalance` | - | `report.mass_imbalance`, a **mass** flux |
//! | `outlet_velocity_histogram` | m/s | the plane pass's GPU-binned histogram |
//! | `residence_time` | s | [`crate::tracers`], opt-in; see below |
//! | `spectrum` | Hz / Pa | **deliberately empty**: acoustics is out of scope |
//!
//! # The two invariants the UI cannot check
//!
//! 1. **`sem` is the standard error of the mean**, not the standard deviation of
//!    the signal. [`ad_metrics::Estimate::sem`] already is, so [`reading`] simply
//!    passes it through; the thing to never do is reach for `std_dev`.
//! 2. **`samples` counts *independent* samples.** Consecutive LBM steps are not
//!    independent — at `dx = 0.75 mm` one step is 12.5 microseconds — so
//!    [`reading`] publishes the autocorrelation-corrected
//!    [`ad_metrics::Estimate::n_eff`], never the raw `n`. On a real run those
//!    differ by an order of magnitude, and quoting `n` would claim a confidence
//!    the data does not support.
//!
//! [`MetricsView::apply_default_thresholds`] applies the contract's traffic
//! lights afterwards, so this file never sets a [`Health`] by hand.
//!
//! # The reset rule
//!
//! Averaging across a parameter change produces a confidently wrong number: the
//! error bar measures scatter, not bias, so a window that is half 3 m/s and half
//! 5 m/s looks *more* converged than either half. Two mechanisms guard it and
//! they must not double-report:
//!
//! * `ad_ui`'s [`ad_ui::ParamWatcher`] sees the slider move, raises the toast
//!   itself, and the app calls [`MetricsSource::reset`].
//! * [`ad_metrics::MetricsConfig::parameter_hash`] catches everything the
//!   watcher cannot see — a mouth patch that moved because the geometry changed,
//!   a lattice-unit change that arrived by another route — and clears the
//!   averages from inside `DuctMetrics`.
//!
//! So a reset the application asked for is swallowed (`ad-ui` has already
//! announced it) and a reset it did not ask for is surfaced through
//! [`MetricsSource::take_unannounced_reset`], which `main.rs` turns into the
//! same toast. A silent clear is the one outcome that must not happen.

use std::sync::Arc;

use ad_gpu::types::{FlowPatch, Grid};
use ad_gpu::GpuContext;
use ad_metrics::field::pack_flags;
use ad_metrics::stats::Health as RunHealth;
use ad_metrics::{
    field_bind_group, field_layout, AgeSample, DuctMetrics, Estimate, FieldRefs, MetricsConfig,
    MetricsReport, ReferenceVelocity,
};
use ad_ui::view::{
    ConvergenceState, Health, HistogramView, MetricsView, PatchView, Reading, ResetCause,
    ResidenceTimeView, StatsWindowView, Trace,
};
use ad_ui::UiState;
use anyhow::{Context as _, Result};
use wgpu::util::DeviceExt as _;

use crate::sim::Sim;

/// Points kept per convergence trace. At one sample per frame this is about a
/// minute of history at 60 fps, which is longer than any transient worth
/// watching and still a trivial memmove in [`Trace::push`].
const TRACE_CAPACITY: usize = 4096;

/// What the app knows at the moment a sample is taken.
pub struct SampleContext<'a> {
    pub sim: &'a Sim,
    /// Solver steps taken.
    pub step: u64,
    /// Steps since the averaging window opened.
    pub steps_in_window: u64,
    /// Physical seconds simulated.
    pub sim_time_s: f64,
    /// Inlet bulk velocity the user asked for, m/s.
    pub inlet_velocity_ms: f32,
    /// Which detected mouth is the inlet, and which is the outlet. Not always
    /// 0 and 1: the toolbar's A<->B toggle moves them, and reporting mouth 0's
    /// area as the inlet after a swap quotes a flow rate for the wrong hole.
    pub inlet: usize,
    pub outlet: usize,
    /// The solver advanced this frame. When it did not — the run is paused, or
    /// the tuner asked for zero steps — nothing is recorded: re-measuring an
    /// unchanged field would fold the same value in again and again, which
    /// inflates the raw sample count without adding any information.
    pub stepped: bool,
}

/// Fills the UI's view models. One implementation per source of numbers.
pub trait MetricsSource {
    /// Update `out` in place. Called once per frame, after the solver has
    /// stepped and before the UI is built.
    fn sample(&mut self, cx: &SampleContext<'_>, out: &mut MetricsView);

    /// Throw away the averaging window. `cause` is what the UI has already
    /// toasted, and is echoed in [`StatsWindowView::last_reset`].
    fn reset(&mut self, cause: ResetCause);

    /// One-shot: true when the averages were cleared for a reason the
    /// *application* did not ask for, so `main.rs` can raise the toast that
    /// `ad-ui` raises for the ones it does know about. See the module docs.
    fn take_unannounced_reset(&mut self) -> bool {
        false
    }

    /// Fold tracer exits into the residence-time distribution. Separate from
    /// [`Self::sample`] because capturing the velocity field needs `&mut Sim`,
    /// which the sample context deliberately does not carry.
    fn record_ages(&mut self, _exits: &[AgeSample], _trapped_fraction: f64) {}

    /// A line for the status bar saying where the numbers come from, so a
    /// placeholder can never be mistaken for a measurement.
    fn provenance(&self) -> &'static str;

    /// GPU time of the measurement passes, ms, when they are timed. For the
    /// frame breakdown in the Diagnostics panel.
    fn gpu_ms(&self) -> Option<f64> {
        None
    }

    /// One line per timed measurement pass, for the Diagnostics panel.
    fn gpu_report(&self) -> Vec<String> {
        Vec::new()
    }

    /// Every scalar with its error bar, as text.
    ///
    /// Written to the log when a headless run captures its screenshot: a PNG
    /// proves the HUD drew, and this proves what it drew — which is the half an
    /// automated run can actually check.
    fn summary(&self) -> String {
        String::new()
    }
}

// ---------------------------------------------------------------------------
// the real thing
// ---------------------------------------------------------------------------

/// [`DuctMetrics`] driven from the frame loop.
///
/// # What it binds
///
/// Group 0 of every metrics kernel is `(velocity, density, flags)`. The first
/// two are the solver's *sampled* views, per CONTRACT.md rule 5. The third is
/// **not** [`ad_solver::Solver::flags_buffer`]: that one covers the solver's
/// padded domain, one halo cell larger on every non-periodic axis, while
/// [`FieldRefs`] wants one byte per *interior* cell. Binding the padded buffer
/// would shear the flag field against the textures by a growing offset and
/// silently mis-classify which samples are fluid. So the mask is uploaded here
/// from `Sim::mask`, which is the interior array the boundary conditions were
/// written into anyway.
pub struct DuctMetricsSource {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    field_group: wgpu::BindGroup,
    /// Kept alive because the bind group refers to it; never read after upload.
    _flags: wgpu::Buffer,
    metrics: DuctMetrics,
    inlet: usize,
    outlet: usize,
    /// Fluid volume of the duct passage, mm^3. `V/Q` from this is the reference
    /// the measured residence time is compared against. `None` when the flood
    /// fill could not isolate the passage — see [`crate::tracers::passage_volume_mm3`].
    passage_volume_mm3: Option<f64>,
    /// Streamwise length a flow-through is counted against, mm.
    duct_length_mm: f64,
    /// Tracer exits recorded so far, and the fraction that never left.
    rtd_exits: u64,
    trapped_fraction: f64,
    /// Set by [`MetricsSource::reset`]; echoed into the stats window.
    last_reset: Option<ResetCause>,
    /// A reset the application asked for is about to arrive as a monitor notice.
    /// Swallow exactly one, so only the ones nobody announced reach the toast.
    expect_reset: bool,
    unannounced_reset: bool,
    /// Last step a convergence point was pushed at, so a frame whose readback
    /// did not advance does not stack points on one x.
    last_trace_step: Option<u64>,
    /// Times the plane and volume passes, which `ad-metrics` records under
    /// scopes on it.
    profiler: ad_gpu::Profiler,
}

impl DuctMetricsSource {
    /// Build the passes and bind them to `sim`'s current textures.
    ///
    /// The bind group captures the solver's textures, so this must be rebuilt
    /// whenever the solver is — which is exactly when `main.rs` adopts a new
    /// [`Sim`].
    pub fn new(gpu: &GpuContext, sim: &Sim, inlet: usize, outlet: usize) -> Result<Self> {
        let layout = field_layout(&gpu.device);

        // `to_le_bytes` rather than a `bytemuck` cast: it is one allocation at
        // construction and it saves the crate a dependency it needs nowhere
        // else. Little-endian is what the shader's `word >> (i & 3) * 8` unpack
        // assumes, and every target this runs on is little-endian.
        let words = pack_flags(&sim.mask);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let flags = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("metrics interior flags"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let velocity = sim.solver.velocity_view();
        let density = sim.solver.density_view();
        let field_group = field_bind_group(
            &gpu.device,
            &layout,
            &FieldRefs {
                grid: sim.grid,
                velocity: &velocity,
                density: &density,
                flags: &flags,
            },
        );

        // The passage volume has to be known before the configuration is built:
        // it is `MetricsConfig::passage_volume_mm3`, the `V` in `tau_ideal =
        // V/Q`, and `ad-metrics` reports the ideal residence time as unknown
        // rather than guessing when it is `None`.
        let passage_volume_mm3 = crate::tracers::passage_volume_mm3(sim);
        match passage_volume_mm3 {
            Some(v) => log::info!("duct passage volume {v:.0} mm^3 (from the interior flag mask)"),
            None => log::warn!(
                "could not isolate the duct passage from the flag mask; the ideal residence \
                 time will not be reported"
            ),
        }

        let cfg = metrics_config(sim, inlet, outlet, passage_volume_mm3);
        let duct_length_mm = cfg.duct_length_mm;
        let metrics =
            DuctMetrics::new(&gpu.device, &layout, cfg).context("building the metrics passes")?;

        Ok(Self {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            field_group,
            _flags: flags,
            metrics,
            inlet,
            outlet,
            passage_volume_mm3,
            duct_length_mm,
            rtd_exits: 0,
            trapped_fraction: f64::NAN,
            last_reset: None,
            expect_reset: false,
            unannounced_reset: false,
            last_trace_step: None,
            profiler: ad_gpu::Profiler::new(&gpu.device, &gpu.queue, 2, gpu.caps.timestamps, None),
        })
    }
}

impl MetricsSource for DuctMetricsSource {
    fn gpu_ms(&self) -> Option<f64> {
        let planes = self.profiler.timing("metrics planes").map(|t| t.mean_ms);
        let volume = self.profiler.timing("metrics volume").map(|t| t.mean_ms);
        match (planes, volume) {
            (None, None) => None,
            (p, v) => Some(p.unwrap_or(0.0) + v.unwrap_or(0.0)),
        }
    }

    fn gpu_report(&self) -> Vec<String> {
        ["metrics planes", "metrics volume"]
            .into_iter()
            .filter(|name| self.profiler.timing(name).is_some())
            .map(|name| self.profiler.summary(name, 0))
            .collect()
    }

    fn sample(&mut self, cx: &SampleContext<'_>, out: &mut MetricsView) {
        // 1. Keep the configuration in step with the operating point. This is
        //    the call that arms `MetricsConfig::parameter_hash`, so a change the
        //    UI never saw still throws the averages away instead of blending two
        //    operating points into one confident-looking mean.
        self.metrics.set_config(metrics_config(
            cx.sim,
            self.inlet,
            self.outlet,
            self.passage_volume_mm3,
        ));

        // 2. Record, then collect. `record` resets its accumulators through
        //    `queue.write_buffer`, so the encoder it writes into has to be
        //    submitted before anything else touches them — hence its own
        //    encoder and its own submit rather than sharing the frame's.
        if cx.stepped {
            self.profiler.begin_frame();
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("metrics"),
                });
            if let Err(e) = self.metrics.record(
                &self.queue,
                &mut enc,
                &self.field_group,
                cx.step,
                Some(&mut self.profiler),
            ) {
                log::warn!("metrics record failed: {e:#}");
            }
            self.profiler.resolve(&mut enc);
            self.queue.submit([enc.finish()]);
        }
        self.profiler.collect(&self.device);
        self.metrics.poll(&self.device);

        // 3. A reset nobody announced has to reach the user.
        if self.metrics.monitor_mut().take_reset_notice() && !std::mem::take(&mut self.expect_reset)
        {
            self.unannounced_reset = true;
        }

        let report = self.metrics.report();
        map_report(
            &report,
            patch_view(cx, cx.inlet),
            patch_view(cx, cx.outlet),
            out,
        );

        // 4. Traces, on the step the readback actually came from.
        if self.last_trace_step != Some(report.step) && report.frames > 0 {
            self.last_trace_step = Some(report.step);
            push_traces(&report, out);
        }
        out.convergence.x_is_steps = true;

        // 5. Residence time. `tau_ideal = V/Q` is already seeded from
        //    `MetricsConfig::passage_volume_mm3`, which step 1 keeps current, so
        //    there is nothing to correct here.
        map_rtd(
            self.metrics.rtd(),
            self.rtd_exits,
            self.trapped_fraction,
            &mut out.residence_time,
        );

        // `report.flow_throughs` is the number the convergence gate actually
        // used, counted from the last reset by `DuctMetrics` itself, so the
        // progress bar and the traffic light can never disagree about how far
        // through the averaging window the run is.
        let per = cx
            .sim
            .units
            .steps_per_flow_through(self.duct_length_mm)
            .max(1.0);
        let cfg = *self.metrics.monitor().config();
        out.stats = StatsWindowView {
            steps_in_window: cx.steps_in_window,
            steps_target: (per * (cfg.discard_flow_throughs + cfg.average_flow_throughs)) as u64,
            flow_throughs: report.flow_throughs,
            flow_throughs_target: cfg.discard_flow_throughs + cfg.average_flow_throughs,
            last_reset: self.last_reset,
            steps_since_reset: cx.steps_in_window,
        };

        // 1 / (2 dt): the highest frequency this time step can represent. Real,
        // and worth drawing even with no spectrum behind it. Everything else in
        // `SpectrumView` stays empty on purpose — acoustics is out of scope, and
        // an empty plot is the honest way to say so.
        out.spectrum.nyquist_hz = 0.5 / cx.sim.units.dt_s;

        if !report.mass_balance_ok && report.frames > 0 {
            out.warnings.push(format!(
                "mass imbalance {:.2}% exceeds the 1% gate: no number here is trustworthy yet",
                report.mass_imbalance.mean * 100.0
            ));
        }
        // Not a fault, but not something to leave the user to work out either:
        // the two flow rates on the convergence plot converge to *different*
        // values on any run with a real pressure drop, and this says why.
        if report.compressibility_is_material() {
            out.warnings.push(format!(
                "Q_out exceeds Q_in by {:.2}%, of which {:.2}% is the measured expansion across \
                 the {:.0} Pa drop: this solver is weakly compressible, and the 1% gate above is \
                 on mass flux, which is what is conserved",
                report.volumetric_imbalance.mean * 100.0,
                report.volumetric_expansion.mean * 100.0,
                report.static_pressure_drop_pa.mean
            ));
        }
        if self.passage_volume_mm3.is_none() {
            out.warnings
                .push("duct passage volume unknown: the ideal residence time is not shown".into());
        }

        out.apply_default_thresholds();
    }

    fn reset(&mut self, cause: ResetCause) {
        self.metrics.monitor_mut().reset();
        self.metrics.rtd_mut().clear();
        self.rtd_exits = 0;
        self.trapped_fraction = f64::NAN;
        self.last_reset = Some(cause);
        self.last_trace_step = None;
        // `Monitor::reset` raises its own notice; the UI has already toasted
        // this one, so the next `sample` swallows it.
        self.expect_reset = true;
    }

    fn take_unannounced_reset(&mut self) -> bool {
        std::mem::take(&mut self.unannounced_reset)
    }

    fn record_ages(&mut self, exits: &[AgeSample], trapped_fraction: f64) {
        let rtd = self.metrics.rtd_mut();
        for s in exits {
            rtd.record(*s);
        }
        self.rtd_exits += exits.len() as u64;
        self.trapped_fraction = trapped_fraction;
    }

    fn provenance(&self) -> &'static str {
        "metrics: ad-metrics (GPU plane + volume reductions)"
    }

    fn summary(&self) -> String {
        let r = self.metrics.report();
        let mut s = r.summary();
        s.push_str(&format!(
            "Q_in = {} CFM, K on {} = {}\n",
            r.flow_in.cfm(),
            r.loss_coefficient.reference.label(),
            r.loss_coefficient.k
        ));
        // The plane geometry, not the flow. A patch that clipped the wall or
        // hung off the domain would under-read in silence, so the coverage goes
        // in the log next to the number it would have corrupted.
        for (name, plane) in [("inlet", &r.inlet), ("outlet", &r.outlet)] {
            if let Some(p) = plane {
                s.push_str(&format!(
                    "{name} plane: {:.0} of {:.0} mm^2 open ({:.1}% covered, {:.1}% off-grid), \
                     |u| mean {:.3} peak {:.3} m/s, p_s {:.3} Pa, p_t {:.3} Pa\n",
                    p.open_area_mm2,
                    p.patch_area_mm2,
                    p.covered_fraction * 100.0,
                    p.outside_fraction * 100.0,
                    p.mean_speed_ms,
                    p.max_speed_ms,
                    p.static_pressure_pa,
                    p.total_pressure_pa,
                ));
            }
        }
        s.push_str(&self.metrics.rtd().summary());
        s.push('\n');
        s
    }
}

/// The metrics configuration for one operating point.
///
/// The only judgement in here is the outlet normal. `Mouth::patch.normal` points
/// *into* the fluid, which is what an inlet wants; [`MetricsConfig`] wants both
/// normals pointing **downstream**, so the outlet's is reversed. With it a
/// working duct reports `Q_in > 0` and `Q_out > 0` and the mass balance is a
/// difference; without it the two flows have opposite signs and the imbalance
/// reads 200% on a perfectly conserved solution.
fn metrics_config(
    sim: &Sim,
    inlet: usize,
    outlet: usize,
    passage_volume_mm3: Option<f64>,
) -> MetricsConfig {
    let inlet_patch = into_the_duct(sim.mouths[inlet.min(sim.mouths.len() - 1)].patch, sim.grid);
    let mut outlet_patch =
        into_the_duct(sim.mouths[outlet.min(sim.mouths.len() - 1)].patch, sim.grid);
    outlet_patch.normal = -outlet_patch.normal;

    let mut cfg = MetricsConfig::new(sim.grid, sim.units, inlet_patch, outlet_patch);
    // A part fed through more than one mouth: the other inlets' flow counts
    // toward `Q_in`, so the mass balance is a statement about the whole part.
    cfg.extra_inlets = sim
        .inlets
        .iter()
        .filter(|m| **m != inlet && **m < sim.mouths.len())
        .map(|m| into_the_duct(sim.mouths[*m].patch, sim.grid))
        .collect();
    cfg.reference = ReferenceVelocity::Inlet;
    // The air inside the part, from the flood fill in `crate::tracers` — never
    // the volume pass's fluid volume, which is the whole 261 mm box and would
    // make `tau_ideal` seventy times too long.
    cfg.passage_volume_mm3 = passage_volume_mm3;
    // A flow-through is a length of *duct*, not of the 261 mm box around it.
    // `MetricsConfig::new` defaults to the grid's longest side, which would
    // stretch every flow-through by 1.8x and make the transient look shorter
    // than it is.
    cfg.duct_length_mm = sim.duct_bbox().size().max_element() as f64;
    cfg
}

/// Cells to move a measurement plane off the mouth and into the passage.
///
/// One cell, so the plane lands on the first cell-centre plane that is
/// unambiguously duct fluid.
const MEASUREMENT_OFFSET_CELLS: f32 = 1.0;

/// Move a mouth patch off the opening and one cell into the duct.
///
/// # Why not measure at the mouth itself
///
/// The mouth plane is where the field is *discontinuous*. At the inlet it is
/// literally the boundary-condition cells: the quadrature's trilinear stencil
/// straddles them and the room air in front, so it returns a weighted average of
/// the prescribed velocity and whatever the room is doing. On the test part that
/// under-reads `Q_in` by 6% — 5.8 L/s against the 6.27 the boundary is actually
/// pushing — which is six times the 1% mass-balance gate all the other numbers
/// are judged by. At the outlet the same stencil mixes duct fluid with the exit
/// jet's entrainment.
///
/// One cell in, snapped to the cell-centre plane, the stencil is entirely inside
/// the passage and the reading is of the fluid rather than of the boundary
/// condition.
///
/// The snap only applies to an axis-aligned mouth, which every mouth detected on
/// a bounding-box face is. Anything else is simply translated along its normal,
/// where there is no cell-centre plane to snap to.
fn into_the_duct(mut patch: FlowPatch, grid: Grid) -> FlowPatch {
    let n = patch.normal.normalize_or_zero();
    if n == glam::Vec3::ZERO {
        return patch;
    }
    let axis = (0..3)
        .max_by(|a, b| n[*a].abs().total_cmp(&n[*b].abs()))
        .unwrap_or(0);
    if n[axis].abs() > 0.999 {
        let k = ((patch.center_mm[axis] - grid.origin_mm[axis]) / grid.dx_mm).round();
        let k = k + n[axis].signum() * MEASUREMENT_OFFSET_CELLS;
        patch.center_mm[axis] = grid.origin_mm[axis] + k * grid.dx_mm;
    } else {
        patch.center_mm += n * (MEASUREMENT_OFFSET_CELLS * grid.dx_mm);
    }
    patch
}

// ---------------------------------------------------------------------------
// the mapping
// ---------------------------------------------------------------------------

/// One [`Estimate`] as the UI's [`Reading`].
///
/// Two things happen here and both are load-bearing:
///
/// * `samples` is [`Estimate::n_eff`], the autocorrelation-corrected count, not
///   the raw `n`. On a run sampled once per frame the two differ by 5-50x, and
///   `n` would tell the user the average is far better established than it is.
/// * An estimate with no samples becomes [`Reading::unknown`] rather than a
///   reading of zero. [`Estimate::ZERO`] has `mean = 0.0`, which is finite and
///   would render as a confident `0.000` — the exact failure the four-valued
///   [`Health`] exists to prevent.
fn reading(e: Estimate) -> Reading {
    if e.n == 0 || !e.mean.is_finite() {
        return Reading::unknown();
    }
    let samples = e.n_eff.max(1.0).round().min(u32::MAX as f64) as u32;
    Reading::new(e.mean, e.sem, Health::Unknown, samples)
}

/// One *instantaneous* quantity, marked as such.
///
/// `sem = NaN` is what [`ad_ui::format::uncertain`] renders with a leading `~`,
/// which is the distinction between "this is the latest frame" and "this is a
/// converged average". Used for the two per-plane peaks, which `ad-metrics` does
/// not keep a series for: adding one would drag a noisy maximum into
/// [`ad_metrics::Monitor::overall`] and hold the whole run at "averaging"
/// forever.
fn instantaneous(v: f64) -> Reading {
    if v.is_finite() {
        Reading::new(v, f64::NAN, Health::Unknown, 1)
    } else {
        Reading::unknown()
    }
}

/// `ad-metrics`' run state as the UI's.
///
/// [`RunHealth::Transient`] maps to `Warmup` rather than `Averaging`: the
/// statistics *are* accumulating, but they are contaminated by the initial
/// condition, and the panel's job is to stop anyone quoting them yet.
fn convergence_state(h: RunHealth) -> ConvergenceState {
    match h {
        RunHealth::Diverged => ConvergenceState::Diverged,
        RunHealth::Converged => ConvergenceState::Converged,
        RunHealth::Converging => ConvergenceState::Averaging,
        RunHealth::Unknown | RunHealth::Transient => ConvergenceState::Warmup,
    }
}

/// Everything a [`MetricsReport`] says, in the UI's terms.
///
/// Pure: no GPU, no solver, no clock. That is what makes the unit conversions
/// and the SEM handling testable from a struct literal, which is where the two
/// invariants at the top of this file are actually checked.
///
/// `inlet` and `outlet` arrive carrying the mouth *geometry*, which is known the
/// moment the STL is voxelised and has nothing to do with the flow; this fills
/// in the measured half.
fn map_report(report: &MetricsReport, inlet: PatchView, outlet: PatchView, out: &mut MetricsView) {
    out.flow_rate = reading(report.flow_in.m3s());
    out.pressure_drop = reading(report.total_pressure_drop_pa);
    out.loss_coefficient = reading(report.loss_coefficient.k);
    out.uniformity = reading(report.uniformity.gamma);
    out.deflection_deg = reading(report.jet.deflection_deg);
    // `Jet::direction` falls back to the outlet normal before anything has been
    // measured; only a real deflection reading makes it a measurement.
    out.jet_direction = out
        .deflection_deg
        .value
        .is_finite()
        .then_some(report.jet.direction);
    out.max_speed = reading(report.peak_speed_ms);

    out.reynolds = report.reynolds;
    out.mach_lb = report.mach_lb;
    out.tau0 = report.tau0;

    out.inlet = PatchView {
        mean_velocity: reading(report.inlet_velocity_ms),
        ..inlet
    };
    out.outlet = PatchView {
        mean_velocity: reading(report.outlet_velocity_ms),
        backflow_fraction: reading(report.uniformity.backflow_fraction),
        ..outlet
    };
    if let Some(r) = &report.inlet {
        out.inlet.max_velocity = instantaneous(r.max_speed_ms);
        out.inlet.backflow_fraction = instantaneous(r.backflow_fraction);
    }
    if let Some(r) = &report.outlet {
        out.outlet.max_velocity = instantaneous(r.max_speed_ms);
    }

    out.convergence.state = convergence_state(report.health);
    out.convergence.mass_imbalance = reading(report.mass_imbalance);

    out.outlet_velocity_histogram = histogram_view(report);
    out.warnings = report.warnings.clone();
}

/// The outlet velocity histogram, pre-binned on the GPU.
///
/// `Uniformity::histogram` is `(bin centre m/s, area fraction)` over
/// `histogram_range_ms`; [`HistogramView`] wants `edges.len() == counts.len() + 1`.
/// Reconstructing the edges from the range rather than from the centres keeps
/// the bars aligned with the range the shader actually binned over, and an
/// inconsistent pair is rejected by [`HistogramView::is_valid`] rather than
/// drawn half a bin out.
fn histogram_view(report: &MetricsReport) -> HistogramView {
    let (lo, hi) = report.uniformity.histogram_range_ms;
    let bins = report.uniformity.histogram.len();
    if bins == 0 || !(hi > lo) {
        return HistogramView::default();
    }
    let width = (hi - lo) / bins as f64;
    HistogramView {
        name: "U_out".into(),
        unit: "m/s".into(),
        edges: (0..=bins).map(|i| lo + i as f64 * width).collect(),
        counts: report
            .uniformity
            .histogram
            .iter()
            .map(|(_, f)| *f)
            .collect(),
        mean: reading(report.outlet_velocity_ms),
    }
}

/// The residence-time distribution as the panel wants it.
///
/// `mean_residence_s` gets a real error bar because tracers *are* independent
/// samples — unlike consecutive solver steps — so `sigma / sqrt(n)` is the
/// honest SEM here with no autocorrelation correction to make. The variance
/// ratio has no closed-form error, so it is marked instantaneous rather than
/// given an invented one.
fn map_rtd(rtd: &ad_metrics::Rtd, exits: u64, trapped_fraction: f64, out: &mut ResidenceTimeView) {
    // Read `tau_ideal` back off the distribution rather than recomputing it, so
    // the panel and `Rtd::summary` can never quote two different ideal times for
    // the same duct.
    out.ideal_residence_s = rtd.tau_ideal_s();
    if exits == 0 || rtd.total_weight() <= 0.0 {
        out.histogram = HistogramView::default();
        out.mean_residence_s = Reading::unknown();
        out.variance_ratio = Reading::unknown();
        out.trapped_fraction = Reading::unknown();
        return;
    }

    let n = exits.min(u32::MAX as u64) as u32;
    let mean = rtd.mean_age_s();
    let sem = (rtd.variance_s2() / exits as f64).sqrt();
    out.mean_residence_s = Reading::new(mean, sem, Health::Unknown, n);
    let dispersion = rtd.dispersion();
    out.variance_ratio = Reading::new(dispersion * dispersion, f64::NAN, Health::Unknown, n);
    out.trapped_fraction = instantaneous(trapped_fraction);

    let dt = rtd.bin_width_s();
    let e = rtd.distribution();
    out.histogram = HistogramView {
        name: "E(t)".into(),
        unit: "s".into(),
        edges: (0..=e.len()).map(|i| i as f64 * dt).collect(),
        counts: e.iter().map(|(_, v)| *v).collect(),
        mean: out.mean_residence_s,
    };
}

/// Append one point to each convergence trace.
///
/// The x axis is the solver step the readback came from, not wall-clock time:
/// the whole point of these curves is to say how far the *solution* has come,
/// and a run whose frame rate changes would otherwise stretch and squash its own
/// history.
fn push_traces(report: &MetricsReport, out: &mut MetricsView) {
    let conv = &mut out.convergence;
    let x = report.step as f64;

    if conv.flow_in.name.is_empty() {
        conv.flow_in = Trace::new("Q_in", "m^3/s");
        conv.flow_out = Trace::new("Q_out", "m^3/s");
        conv.pressure_drop = Trace::new("dp_total", "Pa");
    }
    conv.flow_in
        .push(x, report.flow_in.m3s().mean, TRACE_CAPACITY);
    conv.flow_out
        .push(x, report.flow_out.m3s().mean, TRACE_CAPACITY);
    conv.pressure_drop
        .push(x, report.total_pressure_drop_pa.mean, TRACE_CAPACITY);

    if let Some(r) = report.residual {
        if conv.residuals.is_empty() {
            let mut t = Trace::new("|du| / |u|", "");
            // Residuals span decades and flatten at the amplitude of the real
            // unsteadiness; on a linear axis that plateau is invisible.
            t.log_y = true;
            conv.residuals.push(t);
        }
        conv.residuals[0].push(x, r, TRACE_CAPACITY);
    }
}

// ---------------------------------------------------------------------------
// the fallback
// ---------------------------------------------------------------------------

/// The stand-in used when the metrics passes cannot be built.
///
/// Reports only what is true by construction and marks everything else unknown.
/// It is no longer the default — [`DuctMetricsSource`] is — but a shader that
/// fails to compile on some other driver must degrade to "we do not know"
/// rather than to a plausible number, because a fabricated pressure drop with an
/// error bar is indistinguishable from a real one.
#[derive(Debug, Default)]
pub struct PlaceholderMetrics {
    /// Set by [`MetricsSource::reset`], consumed on the next sample. The reset
    /// call does not know the step count, and guessing zero there would report
    /// "reset 4,200,000 steps ago" the moment anyone touched the slider.
    last_reset: Option<ResetCause>,
}

impl MetricsSource for PlaceholderMetrics {
    fn sample(&mut self, cx: &SampleContext<'_>, out: &mut MetricsView) {
        let units = &cx.sim.units;

        // Deterministic from the operating point, so these are real.
        out.reynolds = units.reynolds(cx.sim.d_h_mm);
        out.mach_lb = units.mach_lb();
        out.tau0 = units.tau0;

        // Prescribed, not measured. Rendered with a leading `~` by
        // `format::uncertain` because `sem` is NaN, which is exactly the
        // distinction between "this is the boundary condition" and "this is a
        // converged average".
        let u_in = cx.inlet_velocity_ms as f64;

        out.inlet = patch_view(cx, cx.inlet);
        out.inlet.mean_velocity = instantaneous(u_in);
        out.outlet = patch_view(cx, cx.outlet);

        let area_m2 = out.inlet.open_area_mm2 as f64 * 1e-6;
        out.flow_rate = instantaneous(u_in * area_m2);

        // Everything genuinely measured stays unknown.
        out.pressure_drop = Reading::unknown();
        out.loss_coefficient = Reading::unknown();
        out.uniformity = Reading::unknown();
        out.deflection_deg = Reading::unknown();
        out.jet_direction = None;
        out.max_speed = Reading::unknown();
        out.convergence.mass_imbalance = Reading::unknown();

        // Window bookkeeping the app really does know.
        let length_mm = cx.sim.duct_bbox().size().max_element() as f64;
        let per_flow_through = units.steps_per_flow_through(length_mm).max(1.0);
        out.stats = StatsWindowView {
            steps_in_window: cx.steps_in_window,
            steps_target: (per_flow_through * 15.0) as u64,
            flow_throughs: cx.steps_in_window as f64 / per_flow_through,
            flow_throughs_target: 15.0,
            last_reset: self.last_reset,
            steps_since_reset: cx.steps_in_window,
        };

        // The one honest trace available without a readback: the inlet flow the
        // boundary condition is prescribing, against simulated time. It is a
        // constant while the slider is still, and it steps when the slider
        // moves, which is exactly what it should show.
        let flow = &mut out.convergence.flow_in;
        if flow.name.is_empty() {
            *flow = Trace::new("Q_in (prescribed)", "m^3/s");
        }
        if flow.last().map(|(t, _)| cx.sim_time_s > t).unwrap_or(true) {
            flow.push(cx.sim_time_s, out.flow_rate.value, TRACE_CAPACITY);
        }
        out.convergence.x_is_steps = false;
        out.spectrum.nyquist_hz = 0.5 / units.dt_s;

        out.warnings = units.warnings();
        out.warnings.push(
            "metrics adapter not connected: measured values read '--' because the ad-metrics \
             passes could not be built on this device"
                .to_string(),
        );

        out.apply_default_thresholds();
    }

    fn reset(&mut self, cause: ResetCause) {
        self.last_reset = Some(cause);
    }

    fn provenance(&self) -> &'static str {
        "metrics: placeholder (deterministic values only)"
    }
}

/// Geometry of one mouth, which the UI needs whether or not anything is measured
/// there.
fn patch_view(cx: &SampleContext<'_>, which: usize) -> PatchView {
    let Some(m) = cx.sim.mouths.get(which) else {
        return PatchView::default();
    };
    PatchView {
        name: format!("{}", (b'A' + which as u8) as char),
        open_area_mm2: m.open_area_mm2,
        hydraulic_diameter_mm: m.hydraulic_diameter_mm(),
        center_mm: m.patch.center_mm,
        normal: m.patch.normal,
        mean_velocity: Reading::unknown(),
        max_velocity: Reading::unknown(),
        backflow_fraction: Reading::unknown(),
        is_inlet: cx.sim.inlets.contains(&which),
        is_outlet: which == cx.sim.outlet,
    }
}

/// Copy the mouth geometry into the UI's toggle list.
///
/// Separate from [`MetricsSource`] because it is pure geometry: it is known the
/// moment the STL is voxelised, long before any flow has been measured.
pub fn publish_mouths(sim: &Sim, state: &mut UiState) {
    state.mouths = (0..sim.mouths.len())
        .map(|i| {
            let m = &sim.mouths[i];
            PatchView {
                name: format!("{}", (b'A' + i as u8) as char),
                open_area_mm2: m.open_area_mm2,
                hydraulic_diameter_mm: m.hydraulic_diameter_mm(),
                center_mm: m.patch.center_mm,
                normal: m.patch.normal,
                is_inlet: sim.inlets.contains(&i),
                is_outlet: i == sim.outlet,
                ..PatchView::default()
            }
        })
        .collect();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_metrics::plane::PlaneReading;
    use ad_metrics::{Flow, Jet, LossBand, LossCoefficient, Uniformity, Wall};
    use glam::Vec3;

    fn estimate(mean: f64, sem: f64, n: u64, n_eff: f64) -> Estimate {
        Estimate {
            mean,
            sem,
            std_dev: sem * n_eff.sqrt(),
            n,
            n_eff,
            tau_int: n as f64 / (2.0 * n_eff),
        }
    }

    /// A plane reading with the fields the mapping reads. Air at the reference
    /// density, so the mass flow is `rho_phys Q`.
    fn plane(u: f64, max: f64, backflow: f64) -> PlaneReading {
        PlaneReading {
            patch_area_mm2: 2116.0,
            covered_fraction: 0.99,
            outside_fraction: 0.0,
            open_area_mm2: 2095.0,
            hydraulic_diameter_mm: 27.4,
            flow_rate_m3s: u * 2095.0e-6,
            mass_flow_kgs: u * 2095.0e-6 * 1.2,
            bulk_velocity_ms: u,
            total_pressure_pa: 20.0,
            total_pressure_area_pa: 20.0,
            static_pressure_pa: 18.0,
            uniformity: 0.93,
            cv: 0.21,
            p05_ms: u * 0.5,
            p95_ms: u * 1.4,
            backflow_fraction: backflow,
            max_speed_ms: max,
            mean_speed_ms: u,
            momentum_dir: Vec3::Z,
            deflection_deg: 3.1,
            cone_half_angle_deg: 14.0,
            histogram: Vec::new(),
            hist_min_ms: 0.0,
            hist_max_ms: max,
            fluid_samples: 250_000,
        }
    }

    /// A report shaped like a converged 3 m/s run on the contract's test part.
    fn synthetic_report() -> MetricsReport {
        let q_in = estimate(6.35e-3, 8.0e-6, 4000, 190.0);
        let q_out = estimate(6.32e-3, 9.0e-6, 4000, 180.0);
        MetricsReport {
            step: 72_000,
            frames: 4000,
            flow_throughs: 18.6,
            health: RunHealth::Converged,
            health_reason: None,
            converged_but_unbalanced: false,
            flow_in: Flow(q_in),
            flow_out: Flow(q_out),
            mass_flow_in_kgs: estimate(7.62e-3, 1.0e-5, 4000, 190.0),
            mass_flow_out_kgs: estimate(7.60e-3, 1.1e-5, 4000, 180.0),
            mass_imbalance: estimate(0.0047, 0.0002, 4000, 150.0),
            volumetric_imbalance: estimate(0.0047, 0.0002, 4000, 150.0),
            volumetric_expansion: estimate(0.0, 0.0, 4000, 150.0),
            mass_balance_ok: true,
            total_pressure_drop_pa: estimate(14.2, 0.31, 4000, 120.0),
            static_pressure_drop_pa: estimate(21.0, 0.4, 4000, 120.0),
            loss_coefficient: LossCoefficient {
                k: estimate(0.58, 0.013, 4000, 120.0),
                band: LossBand::Green,
                reference: ReferenceVelocity::Inlet,
                v_ref_ms: estimate(3.03, 0.004, 4000, 200.0),
            },
            inlet_velocity_ms: estimate(3.03, 0.004, 4000, 200.0),
            outlet_velocity_ms: estimate(5.64, 0.01, 4000, 180.0),
            uniformity: Uniformity {
                gamma: estimate(0.91, 0.002, 4000, 160.0),
                cv: estimate(0.22, 0.001, 4000, 160.0),
                p05_ms: estimate(2.1, 0.02, 4000, 160.0),
                p95_ms: estimate(8.0, 0.03, 4000, 160.0),
                backflow_fraction: estimate(0.002, 0.0001, 4000, 160.0),
                // Three bins spanning 0..3 m/s, so the edges are known by hand.
                histogram: vec![(0.5, 0.2), (1.5, 0.5), (2.5, 0.3)],
                histogram_range_ms: (0.0, 3.0),
            },
            jet: Jet {
                direction: Vec3::NEG_Y,
                deflection_deg: estimate(3.2, 0.05, 4000, 140.0),
                cone_half_angle_deg: estimate(15.0, 0.2, 4000, 140.0),
                throw_distance_m: estimate(1.1, 0.01, 4000, 140.0),
                throw_target_ms: 0.25,
            },
            peak_speed_ms: estimate(11.4, 0.06, 4000, 130.0),
            reverse_volume_fraction: estimate(0.01, 0.0005, 4000, 130.0),
            stagnant_volume_fraction: estimate(0.2, 0.002, 4000, 130.0),
            residual: Some(2.4e-3),
            wall: Wall {
                summary: None,
                mean_shear_pa: Estimate::ZERO,
                max_shear_pa: Estimate::ZERO,
                mean_y_plus: Estimate::ZERO,
                max_y_plus: Estimate::ZERO,
            },
            reynolds: 5400.0,
            mach_lb: 0.0866,
            tau0: 0.5011,
            hydraulic_diameter_mm: 27.4,
            inlet: Some(plane(3.03, 4.6, 0.0)),
            outlet: Some(plane(5.64, 11.0, 0.002)),
            volume: None,
            warnings: vec!["tau = 0.501100 is very close to 0.5".into()],
        }
    }

    fn geometry(name: &str, area: f32) -> PatchView {
        PatchView {
            name: name.into(),
            open_area_mm2: area,
            ..PatchView::default()
        }
    }

    #[test]
    fn a_report_maps_onto_the_view_with_its_units_and_error_bars_intact() {
        let r = synthetic_report();
        let mut v = MetricsView::default();
        map_report(&r, geometry("A", 2095.0), geometry("B", 1120.0), &mut v);

        // SI throughout, and the *mean* carries its own SEM rather than the
        // standard deviation of the signal, which is 14x larger here.
        assert!((v.flow_rate.value - 6.35e-3).abs() < 1e-12);
        assert!(
            (v.flow_rate.sem - 8.0e-6).abs() < 1e-15,
            "the SEM was not preserved"
        );
        assert!(
            v.flow_rate.sem < r.flow_in.m3s().std_dev,
            "std_dev leaked in as the error bar"
        );
        assert!((v.pressure_drop.value - 14.2).abs() < 1e-12);
        assert!((v.pressure_drop.sem - 0.31).abs() < 1e-12);
        assert!((v.loss_coefficient.value - 0.58).abs() < 1e-12);
        assert!((v.uniformity.value - 0.91).abs() < 1e-12);
        assert!((v.max_speed.value - 11.4).abs() < 1e-12);
        assert!((v.convergence.mass_imbalance.value - 0.0047).abs() < 1e-12);

        // The relative error is what has to survive a unit change, and the UI
        // converts at the moment of display. Check the conversion the status bar
        // actually performs.
        let cfm = r.flow_in.cfm();
        assert!((cfm.mean - 6.35e-3 * 2118.88).abs() < 1e-9);
        assert!(
            (cfm.relative_error() - v.flow_rate.relative_error().unwrap()).abs() < 1e-12,
            "converting to CFM moved the relative error"
        );

        // Geometry passes through untouched; the flow half is filled in.
        assert_eq!(v.inlet.name, "A");
        assert_eq!(v.inlet.open_area_mm2, 2095.0);
        assert!((v.inlet.mean_velocity.value - 3.03).abs() < 1e-12);
        assert!((v.outlet.mean_velocity.value - 5.64).abs() < 1e-12);
        assert!((v.outlet.backflow_fraction.value - 0.002).abs() < 1e-12);

        // Deterministic quantities are bare f64s, not readings.
        assert!((v.reynolds - 5400.0).abs() < 1e-9);
        assert_eq!(v.convergence.state, ConvergenceState::Converged);
        assert_eq!(v.warnings.len(), 1);
    }

    #[test]
    fn samples_count_independent_samples_not_solver_steps() {
        // The invariant the UI cannot check. 4000 correlated frames are worth
        // 190 independent ones here, and publishing 4000 would claim an error
        // bar 4.6x tighter than the data supports.
        let r = synthetic_report();
        let mut v = MetricsView::default();
        map_report(&r, geometry("A", 2095.0), geometry("B", 1120.0), &mut v);
        assert_eq!(v.flow_rate.samples, 190);
        assert_ne!(v.flow_rate.samples, r.flow_in.m3s().n as u32);
        assert_eq!(v.pressure_drop.samples, 120);
    }

    #[test]
    fn an_empty_window_reads_unknown_rather_than_a_confident_zero() {
        // `Estimate::ZERO` has a finite mean of 0.0. Passed through naively it
        // would render as "0.000 +/- 0.000" with a green light next to it, which
        // is the single most dangerous thing this application could display.
        // `Reading` derives `PartialEq`, but NaN never equals itself, so the
        // comparison has to be field by field.
        let empty = reading(Estimate::ZERO);
        assert!(!empty.is_known());
        assert!(empty.value.is_nan() && empty.sem.is_nan());
        assert_eq!(empty.samples, 0);
        assert_eq!(empty.state, Health::Unknown);

        let mut r = synthetic_report();
        r.frames = 0;
        r.flow_in = Flow(Estimate::ZERO);
        r.total_pressure_drop_pa = Estimate::ZERO;
        r.loss_coefficient.k = Estimate::ZERO;
        r.uniformity.gamma = Estimate::ZERO;
        r.peak_speed_ms = Estimate::ZERO;
        r.mass_imbalance = Estimate::ZERO;
        let mut v = MetricsView::default();
        map_report(&r, geometry("A", 2095.0), geometry("B", 1120.0), &mut v);

        for (name, reading) in [
            ("flow rate", v.flow_rate),
            ("pressure drop", v.pressure_drop),
            ("loss coefficient", v.loss_coefficient),
            ("uniformity", v.uniformity),
            ("peak speed", v.max_speed),
            ("mass imbalance", v.convergence.mass_imbalance),
        ] {
            assert!(!reading.is_known(), "{name} claimed to know something");
            assert_eq!(reading.samples, 0, "{name}");
            assert!(reading.value.is_nan(), "{name} rendered a zero");
        }
        // ...and the thresholds must not paint a light on nothing.
        v.apply_default_thresholds();
        assert_eq!(v.loss_coefficient.state, Health::Unknown);
        assert_eq!(v.convergence.mass_imbalance.state, Health::Unknown);
    }

    #[test]
    fn the_outlet_histogram_keeps_its_bars_aligned_with_the_binned_range() {
        let r = synthetic_report();
        let h = histogram_view(&r);
        assert!(h.is_valid(), "edges must be one longer than counts");
        assert_eq!(h.edges, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(h.centers(), vec![0.5, 1.5, 2.5]);
        assert_eq!(h.unit, "m/s");
        // Area fractions, so they sum to one.
        assert!((h.counts.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // A degenerate range must produce nothing rather than a bar chart with
        // zero-width bins.
        let mut empty = r.clone();
        empty.uniformity.histogram_range_ms = (0.0, 0.0);
        assert!(!histogram_view(&empty).is_valid());
    }

    #[test]
    fn traces_advance_on_the_step_the_readback_came_from() {
        let mut r = synthetic_report();
        let mut v = MetricsView::default();
        push_traces(&r, &mut v);
        r.step = 73_000;
        push_traces(&r, &mut v);

        assert_eq!(v.convergence.flow_in.x, vec![72_000.0, 73_000.0]);
        assert_eq!(v.convergence.flow_out.y.len(), 2);
        assert_eq!(v.convergence.pressure_drop.y[0], 14.2);
        assert_eq!(v.convergence.residuals.len(), 1);
        assert!(
            v.convergence.residuals[0].log_y,
            "a residual needs a log axis"
        );

        // No residual, no trace: an absent number must not become a zero on a
        // log plot, where it would read as perfect convergence.
        r.residual = None;
        let mut fresh = MetricsView::default();
        push_traces(&r, &mut fresh);
        assert!(fresh.convergence.residuals.is_empty());
    }

    #[test]
    fn an_rtd_with_no_exits_reports_nothing_and_still_quotes_the_ideal_time() {
        // V/Q for the measured passage: 128 cm^3 at 6.3 L/s.
        let mut rtd = ad_metrics::Rtd::new(ad_metrics::tau_ideal_s(128_250.0, 6.3e-3));
        let mut v = ResidenceTimeView::default();
        map_rtd(&rtd, 0, f64::NAN, &mut v);
        assert!(!v.mean_residence_s.is_known());
        assert!(!v.histogram.is_valid());
        assert!(
            (v.ideal_residence_s - 0.020_357).abs() < 1e-5,
            "{}",
            v.ideal_residence_s
        );

        // With no passage volume the app hands the distribution an infinite
        // ideal time, which must read as "unknown" rather than as a number the
        // panel would draw a reference line at.
        rtd.set_tau_ideal(f64::INFINITY);
        map_rtd(&rtd, 0, f64::NAN, &mut v);
        assert!(!v.ideal_residence_s.is_finite());
    }

    #[test]
    fn a_measured_rtd_carries_a_real_error_bar_because_tracers_are_independent() {
        let mut rtd = ad_metrics::Rtd::new(0.03);
        // Two equally weighted arrival times: mean 0.03, sigma 0.01.
        for _ in 0..500 {
            rtd.record(AgeSample::new(0.02, 1.0));
            rtd.record(AgeSample::new(0.04, 1.0));
        }
        let mut v = ResidenceTimeView::default();
        map_rtd(&rtd, 1000, 0.03, &mut v);

        assert!((v.mean_residence_s.value - 0.03).abs() < 1e-12);
        assert_eq!(v.mean_residence_s.samples, 1000);
        let want_sem = 0.01 / (1000.0f64).sqrt();
        assert!(
            (v.mean_residence_s.sem - want_sem).abs() < 1e-9,
            "sem {} vs {want_sem}",
            v.mean_residence_s.sem
        );
        // sigma^2 / t_bar^2 for this pair is (0.01/0.03)^2 = 1/9.
        assert!((v.variance_ratio.value - 1.0 / 9.0).abs() < 1e-9);
        assert!((v.trapped_fraction.value - 0.03).abs() < 1e-12);
        assert!(v.histogram.is_valid());
        // E(t) integrates to one over the histogram.
        let dt = v.histogram.bin_width();
        assert!((v.histogram.counts.iter().sum::<f64>() * dt - 1.0).abs() < 1e-9);
    }

    /// The measurement planes must land on a cell-centre plane inside the duct,
    /// whichever way the mouth faces.
    ///
    /// Left on the mouth itself, the inlet plane's stencil straddles the
    /// boundary-condition cells and the room air in front of them and under-reads
    /// `Q_in` by 6% — six times the mass-balance gate every other number is
    /// judged by.
    #[test]
    fn a_measurement_plane_sits_one_cell_inside_the_duct() {
        let grid = ad_gpu::types::Grid {
            dims: glam::UVec3::splat(16),
            dx_mm: 1.0,
            origin_mm: Vec3::splat(0.5),
        };
        let at = |normal: Vec3, w: f32| {
            let p = FlowPatch {
                center_mm: Vec3::new(0.0, 0.0, w),
                normal,
                half_u: Vec3::X * 4.0,
                half_v: Vec3::Y * 4.0,
            };
            into_the_duct(p, grid).center_mm.z
        };
        // Mouth at z = 2 facing +z: nearest cell centre is 2.5, one further in
        // is 3.5. Both are cell centres, which is the point of the snap.
        assert_eq!(at(Vec3::Z, 2.0), 3.5);
        // The same mouth facing the other way moves the other way.
        assert_eq!(at(-Vec3::Z, 2.0), 1.5);
        // A tilted patch has no cell-centre plane to snap to, so it is simply
        // translated by one cell along its own normal.
        let tilted = Vec3::new(0.0, 0.6, 0.8);
        let moved = at(tilted, 2.0);
        assert!(
            (moved - (2.0 + 0.8)).abs() < 1e-6,
            "tilted patch moved to {moved}"
        );
        // A degenerate normal must not produce a NaN centre.
        assert_eq!(at(Vec3::ZERO, 2.0), 2.0);
    }

    /// The gate is on mass flux; the volumetric gap is explained rather than
    /// hidden.
    ///
    /// On the test part at 3 m/s the two flow meters disagree by 7% with mass
    /// exactly conserved, because the air expands across the ~100 Pa drop. The
    /// panel must not raise a conservation alarm for that — but it must also not
    /// leave a user staring at two convergence curves that settle a visible
    /// distance apart with no explanation.
    #[test]
    fn a_compressible_run_explains_its_volumetric_gap_without_failing_the_gate() {
        let mut r = synthetic_report();
        r.mass_imbalance = estimate(0.0015, 0.0001, 4000, 150.0);
        r.volumetric_imbalance = estimate(0.0761, 0.0004, 4000, 150.0);
        r.volumetric_expansion = estimate(0.0714, 0.0004, 4000, 150.0);
        assert!(r.mass_balance_ok, "mass is conserved to 0.15%");
        assert!(
            r.compressibility_is_material(),
            "a 7.6% volume gap is worth saying"
        );
        // The expansion accounts for all but half a percent of it.
        assert!(
            r.unexplained_volumetric_imbalance().abs() < 0.005,
            "{} unexplained",
            r.unexplained_volumetric_imbalance()
        );

        // An incompressible-looking run says nothing, because there is nothing
        // to say: the line would be noise on every reading.
        let mut flat = r.clone();
        flat.volumetric_imbalance = estimate(0.0016, 0.0001, 4000, 150.0);
        flat.volumetric_expansion = estimate(0.0001, 0.0001, 4000, 150.0);
        assert!(!flat.compressibility_is_material());

        // Nothing measured yet: no claim either way.
        let mut empty = r.clone();
        empty.volumetric_imbalance = Estimate::ZERO;
        assert!(!empty.compressibility_is_material());
    }

    #[test]
    fn the_run_state_distinguishes_a_warm_up_from_an_average() {
        // A transient is not "averaging": the numbers exist but are still
        // contaminated by the initial condition.
        assert_eq!(
            convergence_state(RunHealth::Transient),
            ConvergenceState::Warmup
        );
        assert_eq!(
            convergence_state(RunHealth::Unknown),
            ConvergenceState::Warmup
        );
        assert_eq!(
            convergence_state(RunHealth::Converging),
            ConvergenceState::Averaging
        );
        assert_eq!(
            convergence_state(RunHealth::Converged),
            ConvergenceState::Converged
        );
        assert_eq!(
            convergence_state(RunHealth::Diverged),
            ConvergenceState::Diverged
        );
    }

    #[test]
    fn a_prescribed_reading_is_marked_provisional_rather_than_converged() {
        use ad_ui::format::{uncertain, Quantity, UnitSystem};
        let s = uncertain(instantaneous(3.0), Quantity::velocity(UnitSystem::Metric));
        assert!(
            s.starts_with('~'),
            "a boundary condition must not read as a measurement: {s}"
        );
        assert!(!s.contains("+/-"));
    }
}

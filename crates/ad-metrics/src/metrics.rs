//! The aggregate: everything the UI reads, with an error bar on every number.
//!
//! [`DuctMetrics`] owns the plane pass, the volume pass and the statistics. It
//! is recorded once per frame, polled once per frame, and asked for a
//! [`MetricsReport`] whenever something wants to draw.
//!
//! # The one number to check first
//!
//! **Mass imbalance, `|mdot_in - mdot_out| / mdot_in`.** It must be under 1%.
//!
//! It is the most trustworthy correctness check in the tool, because it is the
//! only one that has a known right answer. Every other quantity here — pressure
//! drop, loss coefficient, uniformity — is a number nobody knows in advance, so
//! a plausible-looking wrong value is indistinguishable from a right one. Mass
//! conservation is not like that: air in equals air out, always, and an
//! imbalance is a direct measure of everything that can be wrong at once. A
//! patch that clips the geometry, a plane placed inside the sponge layer, an
//! outlet still passing a starting transient, a solver that is quietly losing
//! mass at the boundary — all of them show up here first.
//!
//! So [`crate::stats::Monitor`] gates convergence on it: no scalar is reported
//! [`Health::Converged`] while the mass balance is broken, however steady it
//! looks. A steady wrong answer is worse than an unsteady one.
//!
//! ## Why it is a mass flux and not a volume flux
//!
//! `rho u` is conserved; `u` is not. Lattice Boltzmann is weakly compressible,
//! so a plane sitting at +100 Pa carries air that is 7 parts in a thousand
//! denser than a plane at ambient — in lattice terms `rho - 1 = 0.070` — and the
//! same mass leaves through the outlet as a 7% larger *volume*. On the test part
//! at 3 m/s the volumetric imbalance reads 7.6% while mass is conserved to
//! 1.5%. Gating on the volumetric figure would therefore condemn a healthy run
//! seven times over the tolerance and, worse, teach everyone to ignore the one
//! number in this tool that cannot lie.
//!
//! Both are reported. [`MetricsReport::mass_imbalance`] is the gate;
//! [`MetricsReport::volumetric_imbalance`] is what a pair of flow meters would
//! read, and [`MetricsReport::volumetric_expansion`] — the density ratio between
//! the two planes, measured, not assumed — is the amount of the difference that
//! compressibility accounts for. When the second and third agree and the first
//! is small, the duct is conserving mass and expanding the air; when the first
//! is large, something is actually wrong.
//!
//! # Why the pressure drop is mass-flow weighted
//!
//! Total pressure drop is `p_t(inlet) - p_t(outlet)` with
//!
//! ```text
//! p_t = sum(rho w p_t dA) / sum(rho w dA)      mass-flow weighted
//! ```
//!
//! not the area average. The two agree exactly on a uniform plane and diverge
//! the moment there is any backflow, because area weighting counts a reversed
//! corner as carrying just as much air as the core. Since a reversed corner has
//! low total pressure, area weighting drags the outlet average down and
//! **inflates** the reported `dp` — precisely at the operating points where a
//! designer is trying to decide whether a change helped. The area-weighted value
//! is still reported, as [`PlaneReading::total_pressure_area_pa`], so the gap
//! can be seen; it is not the number to quote.
//!
//! Static pressure drop is reported separately and area-weighted, because static
//! pressure is what a manometer tapping the wall measures and mass-flow
//! weighting it would not correspond to any instrument.
//!
//! # Loss coefficient
//!
//! `K = dp_total / (0.5 rho V_ref^2)`, dimensionless, and **the reference
//! velocity is part of the answer**. Quoting `K = 0.8` without saying whether
//! `V_ref` is the inlet bulk, the outlet bulk or the peak is meaningless: this
//! duct has a 1.85:1 contraction, so those three differ by more than a factor of
//! three in `V^2`. [`LossCoefficient`] therefore carries its
//! [`ReferenceVelocity`] and the value used.
//!
//! Reference points, from ASHRAE: a smooth radiused elbow runs `K = 0.2..0.3`; a
//! mitred 90 degree bend with no turning vanes runs `K = 2.0..3.5`. The
//! contract's pass mark is `K < 1`, ideally 0.3-0.6.
//!
//! # Every scalar is a time average
//!
//! Nothing here quotes an instantaneous value. Each frame's readings are pushed
//! into a [`crate::stats::Series`], which corrects the standard error for
//! autocorrelation — successive solver steps advance the physical clock by
//! microseconds and are nothing like independent samples — and the report hands
//! back [`Estimate`]s. `dp = 47.3 +/- 0.6 Pa` is an engineering statement;
//! `47.31 Pa` is a statement about one time step.
//!
//! Derived quantities (`K`, the imbalance, the throw distance) are formed
//! **per frame and then averaged**, not by combining averages. That propagates
//! the fluctuation correctly and needs no covariance bookkeeping.

use ad_gpu::types::{FlowPatch, Grid, LatticeUnits, MetricSample};
use anyhow::Result;
use glam::Vec3;

use crate::plane::{PlaneMetrics, PlaneReading};
use crate::rtd::{tau_ideal_s, Rtd};
use crate::stats::{Estimate, Health, Monitor, MonitorConfig, ParameterHash};
use crate::volume::{VolumeConfig, VolumeMetrics, VolumeReading};
use crate::wall::WallSummary;

/// Series names, so the monitor keys and the report cannot drift apart.
pub mod series {
    pub const Q_IN: &str = "flow_in";
    pub const Q_OUT: &str = "flow_out";
    pub const MDOT_IN: &str = "mass_flow_in";
    pub const MDOT_OUT: &str = "mass_flow_out";
    /// `|mdot_in - mdot_out| / mdot_in`. The gate.
    pub const IMBALANCE: &str = "mass_imbalance";
    /// `|Q_in - Q_out| / Q_in`. Reported, never gated on: see the module docs.
    pub const VOLUME_IMBALANCE: &str = "volume_imbalance";
    /// `rho_in / rho_out - 1`, the volumetric expansion the measured densities
    /// predict. Explains the gap between the two imbalances above.
    pub const EXPANSION: &str = "volumetric_expansion";
    pub const DP_TOTAL: &str = "dp_total";
    pub const DP_STATIC: &str = "dp_static";
    pub const K: &str = "loss_coefficient";
    pub const V_REF: &str = "v_ref";
    pub const U_IN: &str = "u_inlet";
    pub const U_OUT: &str = "u_outlet";
    pub const GAMMA: &str = "uniformity";
    pub const CV: &str = "cv";
    pub const P05: &str = "p05";
    pub const P95: &str = "p95";
    pub const BACKFLOW: &str = "backflow_out";
    pub const DEFLECTION: &str = "deflection_deg";
    pub const CONE: &str = "cone_half_angle_deg";
    pub const THROW: &str = "throw_m";
    pub const PEAK_SPEED: &str = "peak_speed";
    pub const REVERSE: &str = "reverse_volume_fraction";
    pub const STAGNANT: &str = "stagnant_volume_fraction";
    pub const RESIDUAL: &str = "residual";
    pub const WALL_SHEAR: &str = "wall_shear_mean";
    pub const WALL_SHEAR_MAX: &str = "wall_shear_max";
    pub const Y_PLUS: &str = "y_plus_mean";
    pub const Y_PLUS_MAX: &str = "y_plus_max";
}

/// Which velocity the loss coefficient is normalised by.
///
/// Part of the answer, not a detail: `K` scales as `1/V_ref^2`, and this duct's
/// inlet and outlet bulk velocities differ by 1.85:1, so choosing differently
/// changes `K` by 3.4x.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceVelocity {
    /// Inlet bulk velocity, `Q / A_open`. The usual convention for a fitting.
    Inlet,
    /// Outlet bulk velocity. The convention when the fitting is a nozzle and the
    /// exit condition is what matters.
    Outlet,
    /// The larger of the two. Conservative: it gives the smallest `K`.
    Faster,
}

impl ReferenceVelocity {
    pub fn label(self) -> &'static str {
        match self {
            ReferenceVelocity::Inlet => "inlet bulk",
            ReferenceVelocity::Outlet => "outlet bulk",
            ReferenceVelocity::Faster => "faster of inlet/outlet bulk",
        }
    }

    fn pick(self, u_in: f64, u_out: f64) -> f64 {
        match self {
            ReferenceVelocity::Inlet => u_in,
            ReferenceVelocity::Outlet => u_out,
            ReferenceVelocity::Faster => u_in.abs().max(u_out.abs()),
        }
    }
}

/// Traffic light for the loss coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossBand {
    /// `K < 0.6`. The contract's "ideal" range; a well-radiused bend.
    Green,
    /// `0.6 <= K < 1.5`. Past ideal but inside the `K < 1` pass mark or close
    /// to it.
    Amber,
    /// `K >= 1.5`. Heading for uncut-mitred-bend territory.
    Red,
}

impl LossBand {
    pub fn of(k: f64) -> Self {
        if !k.is_finite() || k >= 1.5 {
            LossBand::Red
        } else if k < 0.6 {
            LossBand::Green
        } else {
            LossBand::Amber
        }
    }

    pub fn color(self) -> &'static str {
        match self {
            LossBand::Green => "green",
            LossBand::Amber => "amber",
            LossBand::Red => "red",
        }
    }

    /// What the band means, in the terms an ASHRAE table uses.
    pub fn describe(self) -> &'static str {
        match self {
            LossBand::Green => "at or below a well-radiused elbow (K = 0.2-0.3)",
            LossBand::Amber => "above the ideal range but inside the K < 1 target",
            LossBand::Red => "approaching an uncut mitred bend (K = 2.0-3.5)",
        }
    }
}

/// Everything the metrics layer needs to know about the case.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// The interior grid the field textures cover.
    pub grid: Grid,
    pub units: LatticeUnits,
    /// Inlet patch. Its normal points **downstream**, into the domain.
    pub inlet: FlowPatch,
    /// Outlet patch. Its normal also points **downstream**, out of the domain,
    /// so a working duct gives `Q_in > 0` and `Q_out > 0` and the imbalance is
    /// a difference rather than a sum.
    pub outlet: FlowPatch,
    /// Additional measurement planes, reported but not used for any aggregate.
    pub extra_planes: Vec<FlowPatch>,
    /// Further inlet planes, normals downstream like `inlet`. Their flow and
    /// mass flow add to the inlet's for `Q_in`, the mass balance and the
    /// expansion; pressures and the reference velocity stay the primary
    /// inlet's. For a part fed through more than one mouth.
    pub extra_inlets: Vec<FlowPatch>,
    pub reference: ReferenceVelocity,
    pub volume: VolumeConfig,
    /// Fluid volume **of the duct passage itself**, mm^3: the air inside the
    /// printed part, between the two mouths. `None` when the caller cannot
    /// isolate it, which reports the ideal residence time as unknown rather than
    /// as a number.
    ///
    /// This is the `V` in `tau_ideal = V / Q`, so it decides the whole residence
    /// time panel — and it is **not** the volume pass's
    /// [`VolumeReading::fluid_volume_mm3`], which counts every fluid cell in the
    /// 260 x 180 x 180 mm domain box, room air included. On the contract's test
    /// part those are 128,250 mm^3 and roughly 9,000,000 mm^3: a factor of
    /// seventy, which would make `1 - t_bar/tau_ideal` read as 99% dead volume
    /// for any real flow. The metrics layer cannot tell the two apart by itself
    /// — a flood fill from the mouths is what separates them, and that is
    /// geometry this crate deliberately knows nothing about — so the caller must
    /// say, and the field is named after what it must be given.
    pub passage_volume_mm3: Option<f64>,
    /// Streamwise length used to convert steps into flow-through times, mm.
    pub duct_length_mm: f64,
    /// Terminal velocity the throw distance is quoted to, m/s. 0.25 is the
    /// ASHRAE convention for the edge of an occupied zone.
    pub throw_target_ms: f64,
    /// ASHRAE throw constant `K` in `V_x = K V_0 sqrt(A_0) / x`. 6.0 is the
    /// usual value for a compact free jet.
    pub throw_constant: f64,
    pub monitor: MonitorConfig,
}

impl MetricsConfig {
    pub fn new(grid: Grid, units: LatticeUnits, inlet: FlowPatch, outlet: FlowPatch) -> Self {
        let length = grid.dims.as_vec3().max_element() as f64 * grid.dx_mm as f64;
        Self {
            grid,
            units,
            inlet,
            outlet,
            extra_planes: Vec::new(),
            extra_inlets: Vec::new(),
            reference: ReferenceVelocity::Inlet,
            volume: VolumeConfig {
                axis: outlet.normal.normalize_or_zero(),
                // 5% of the inlet bulk lattice velocity.
                stagnation_threshold: (units.u_lb * 0.05) as f32,
                ..Default::default()
            },
            // Unknown until the caller floods the passage. Left `None` rather
            // than defaulted to anything, because every plausible default here
            // is a wrong number that would still draw a reference line.
            passage_volume_mm3: None,
            duct_length_mm: length,
            throw_target_ms: 0.25,
            throw_constant: 6.0,
            monitor: MonitorConfig::default(),
        }
    }

    /// Everything that invalidates an average, hashed. Fed to
    /// [`crate::stats::Monitor::set_parameters`] every frame, so moving the
    /// inlet or changing the velocity throws the old average away instead of
    /// silently mixing two operating points.
    ///
    /// [`Self::passage_volume_mm3`] is deliberately *not* in it: it rescales
    /// `tau_ideal`, which is recomputed every frame anyway, and it often arrives
    /// a few frames after the run starts. Hashing it would throw away a
    /// perfectly good average the moment the flood fill finished.
    pub fn parameter_hash(&self) -> u64 {
        let patch = |h: ParameterHash, p: &FlowPatch| {
            h.vec3(p.center_mm)
                .vec3(p.normal)
                .vec3(p.half_u)
                .vec3(p.half_v)
        };
        let mut h = ParameterHash::new()
            .u64(self.grid.dims.x as u64)
            .u64(self.grid.dims.y as u64)
            .u64(self.grid.dims.z as u64)
            .f32(self.grid.dx_mm)
            .vec3(self.grid.origin_mm)
            .f64(self.units.dx_m)
            .f64(self.units.dt_s)
            .f64(self.units.rho_phys)
            .f64(self.units.nu_phys)
            .f64(self.units.u_lb)
            .f64(self.units.u_phys)
            .f64(self.units.tau0)
            .str(self.reference.label())
            .f64(self.throw_target_ms)
            .f64(self.throw_constant);
        h = patch(h, &self.inlet);
        h = patch(h, &self.outlet);
        for p in &self.extra_planes {
            h = patch(h, p);
        }
        for p in &self.extra_inlets {
            h = patch(h, p);
        }
        h.finish()
    }

    /// Inlet, outlet, any extra planes, then any extra inlets, in the slot
    /// order the plane pass dispatches them.
    pub fn patches(&self) -> Vec<FlowPatch> {
        let mut v = Vec::with_capacity(2 + self.extra_planes.len() + self.extra_inlets.len());
        v.push(self.inlet);
        v.push(self.outlet);
        v.extend_from_slice(&self.extra_planes);
        v.extend_from_slice(&self.extra_inlets);
        v
    }

    /// The snapshot slots of the extra inlets: after the two mains and the
    /// extra planes.
    pub fn extra_inlet_slots(&self) -> std::ops::Range<usize> {
        let start = 2 + self.extra_planes.len();
        start..start + self.extra_inlets.len()
    }
}

/// One frame's decoded measurements, before averaging.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub step: u64,
    /// One per patch, in [`MetricsConfig::patches`] order.
    pub planes: Vec<PlaneReading>,
    pub volume: Option<VolumeReading>,
}

impl Snapshot {
    pub fn inlet(&self) -> &PlaneReading {
        &self.planes[0]
    }
    pub fn outlet(&self) -> &PlaneReading {
        &self.planes[1]
    }
}

/// The two GPU passes. Absent in a headless aggregator.
struct GpuPasses {
    planes: PlaneMetrics,
    volume: VolumeMetrics,
}

/// The whole metrics layer: two GPU passes, the statistics, and the residence
/// time distribution.
pub struct DuctMetrics {
    /// `None` for a headless aggregator built by [`DuctMetrics::headless`],
    /// which folds readings that came from somewhere else.
    gpu: Option<GpuPasses>,
    monitor: Monitor,
    rtd: Rtd,
    cfg: MetricsConfig,
    latest: Option<Snapshot>,
    latest_wall: Option<WallSummary>,
    /// The step of the most recent frame folded into the averages.
    last_step: u64,
    /// Solver step the current averaging window opened at. Flow-throughs are
    /// counted from **here**, not from step zero: see [`Self::sync_window`].
    window_origin_step: u64,
    /// [`Monitor::reset_count`] as of the last frame, so a reset that happened
    /// by any route — a parameter-hash change, or the application calling
    /// [`Monitor::reset`] directly — is noticed here exactly once.
    resets_seen: u64,
    frames: u64,
}

impl DuctMetrics {
    pub fn new(
        device: &wgpu::Device,
        field_layout: &wgpu::BindGroupLayout,
        cfg: MetricsConfig,
    ) -> Result<Self> {
        Self::with_samples(device, field_layout, cfg, crate::plane::DEFAULT_SAMPLES)
    }

    pub fn with_samples(
        device: &wgpu::Device,
        field_layout: &wgpu::BindGroupLayout,
        cfg: MetricsConfig,
        samples: u32,
    ) -> Result<Self> {
        let n_planes = cfg.patches().len();
        let planes = PlaneMetrics::with_samples(device, field_layout, n_planes, samples, 3)?;
        let volume = VolumeMetrics::new(device, field_layout, cfg.grid, cfg.volume)?;
        let mut me = Self::headless(cfg);
        me.gpu = Some(GpuPasses { planes, volume });
        Ok(me)
    }

    /// An aggregator with no GPU passes.
    ///
    /// [`Self::record`] and [`Self::poll`] do nothing; [`Self::observe`] still
    /// folds readings, and [`Self::report`] still produces the full report. That
    /// is what lets the derivations — imbalance, pressure drop, `K`, the reset
    /// rule — be tested against synthetic readings with no device in the loop,
    /// and it is also the shape a batch replay of saved readings wants.
    pub fn headless(cfg: MetricsConfig) -> Self {
        let mut monitor = Monitor::new(cfg.monitor);
        monitor.set_parameters(cfg.parameter_hash());
        let resets_seen = monitor.reset_count();
        Self {
            gpu: None,
            monitor,
            rtd: Rtd::new(f64::INFINITY),
            cfg,
            latest: None,
            latest_wall: None,
            last_step: 0,
            window_origin_step: 0,
            resets_seen,
            frames: 0,
        }
    }

    pub fn config(&self) -> &MetricsConfig {
        &self.cfg
    }

    /// Apply a new configuration. The averages are cleared automatically if the
    /// operating point changed; they survive a change that cannot affect the
    /// physics (for instance the reference-velocity label is hashed, so it does
    /// reset — deliberately, since `K` would otherwise mix two definitions).
    ///
    /// Changing the grid or the residual stride needs a new [`DuctMetrics`],
    /// because the volume pass's buffers are sized to them.
    pub fn set_config(&mut self, cfg: MetricsConfig) {
        let changed = self.monitor.set_parameters(cfg.parameter_hash());
        if let Some(g) = &mut self.gpu {
            g.volume.set_config(cfg.volume);
            if changed {
                g.volume.invalidate_history();
            }
        }
        if changed {
            self.rtd.clear();
            self.latest = None;
            self.latest_wall = None;
            self.frames = 0;
        }
        self.cfg = cfg;
    }

    pub fn monitor(&self) -> &Monitor {
        &self.monitor
    }

    pub fn monitor_mut(&mut self) -> &mut Monitor {
        &mut self.monitor
    }

    /// The residence time distribution. The particle system feeds it
    /// [`crate::rtd::AgeSample`]s; this crate only supplies `V/Q`.
    pub fn rtd(&self) -> &Rtd {
        &self.rtd
    }

    pub fn rtd_mut(&mut self) -> &mut Rtd {
        &mut self.rtd
    }

    pub fn latest(&self) -> Option<&Snapshot> {
        self.latest.as_ref()
    }

    pub fn frames_folded(&self) -> u64 {
        self.frames
    }

    pub fn has_capacity(&self) -> bool {
        match &self.gpu {
            Some(g) => g.planes.has_capacity() && g.volume.has_capacity(),
            None => false,
        }
    }

    /// Record both passes into `encoder`. Returns `false` when a readback ring
    /// is saturated and this frame was skipped, which is not an error: the
    /// measurement would have been stale.
    ///
    /// The accumulators are reset through `queue.write_buffer`, so `encoder`
    /// must be submitted on `queue` before anything else touches them.
    ///
    /// `profiler`, when given, times the two passes as `metrics planes` and
    /// `metrics volume`. A scope is claimed only when its pass is recorded, so
    /// a skipped frame never resolves a timestamp that was not written.
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        field_group: &wgpu::BindGroup,
        step: u64,
        mut profiler: Option<&mut ad_gpu::Profiler>,
    ) -> Result<bool> {
        let patches = self.cfg.patches();
        let grid = self.cfg.grid;
        let Some(g) = &mut self.gpu else {
            return Ok(false);
        };
        let a = g.planes.record(
            queue,
            encoder,
            field_group,
            grid,
            &patches,
            step,
            profiler.as_deref_mut(),
        )?;
        let b = g.volume.record(queue, encoder, field_group, step, profiler);
        Ok(a || b)
    }

    /// Collect whatever has landed and fold it into the averages. Never blocks.
    ///
    /// Returns the number of frames folded. Plane and volume readbacks are
    /// matched by step; a volume frame whose plane partner never arrived (or
    /// vice versa) still contributes what it has, because dropping a whole
    /// frame's statistics to keep two independent passes in lockstep would cost
    /// more than it buys.
    pub fn poll(&mut self, device: &wgpu::Device) -> usize {
        let (plane_frames, volume_frames) = match &mut self.gpu {
            Some(g) => (g.planes.poll(device), g.volume.poll(device)),
            None => return 0,
        };
        let patches = self.cfg.patches();
        let lu = self.cfg.units;

        let mut folded = 0;
        for f in &plane_frames {
            // A frame recorded before a patch was added carries fewer slots than
            // the current configuration wants. Dropping it is right: those slots
            // never existed, and the parameter hash has already cleared the
            // averages it would have joined.
            if f.data.len() < patches.len() {
                continue;
            }
            let readings: Vec<PlaneReading> = patches
                .iter()
                .enumerate()
                .map(|(i, p)| PlaneReading::from_accum(&f.data[i], p, &lu))
                .collect();
            let volume = volume_frames
                .iter()
                .find(|v| v.step == f.step)
                .or_else(|| volume_frames.last())
                .map(|v| VolumeReading::from_accums(&v.data[0], &v.data[1], self.cfg.grid, &lu));
            self.observe(Snapshot {
                step: f.step,
                planes: readings,
                volume,
            });
            folded += 1;
        }
        // A volume frame with no plane partner still updates the peak velocity
        // and the residual, which are the two numbers a user watches while the
        // run is warming up.
        if plane_frames.is_empty() {
            for v in &volume_frames {
                let vr = VolumeReading::from_accums(&v.data[0], &v.data[1], self.cfg.grid, &lu);
                self.observe_volume(&vr);
                self.last_step = self.last_step.max(v.step);
                folded += 1;
            }
        }
        folded
    }

    /// Fold one decoded frame into the statistics.
    ///
    /// Separated from [`Self::poll`] and public so the derivation can be tested
    /// against synthetic readings with no GPU in the loop — which is how every
    /// analytic case in this crate is checked.
    pub fn observe(&mut self, snap: Snapshot) {
        if snap.planes.len() < 2 {
            // Nothing here is defined without both an inlet and an outlet, and
            // silently folding half a frame would move the averages by an
            // amount nobody could later account for.
            log::debug!(
                "metrics: ignoring a snapshot with {} planes",
                snap.planes.len()
            );
            return;
        }
        self.monitor.set_parameters(self.cfg.parameter_hash());
        self.sync_window(snap.step);

        let lu = &self.cfg.units;
        let inlet = &snap.planes[0];
        let outlet = &snap.planes[1];

        // Every inlet mouth feeds the part, so the flow in is the sum. The
        // pressures below stay the primary inlet's.
        let extra: Vec<&PlaneReading> = self
            .cfg
            .extra_inlet_slots()
            .filter_map(|i| snap.planes.get(i))
            .collect();
        let q_in = inlet.flow_rate_m3s + extra.iter().map(|p| p.flow_rate_m3s).sum::<f64>();
        let q_out = outlet.flow_rate_m3s;
        self.monitor.observe(series::Q_IN, q_in);
        self.monitor.observe(series::Q_OUT, q_out);

        let mdot_in = inlet.mass_flow_kgs + extra.iter().map(|p| p.mass_flow_kgs).sum::<f64>();
        let mdot_out = outlet.mass_flow_kgs;
        self.monitor.observe(series::MDOT_IN, mdot_in);
        self.monitor.observe(series::MDOT_OUT, mdot_out);

        // Both imbalances are normalised by the inlet, which is the plane the
        // boundary condition sets and therefore the one that is not itself in
        // question. Only the mass one gates: `rho u` is what the continuity
        // equation conserves, and `u` alone differs between the planes by the
        // compressibility of the solver. See the module docs.
        let ratio = |a: f64, b: f64| {
            if a.abs() > 1e-30 {
                Some((a - b).abs() / a.abs())
            } else {
                None
            }
        };
        if let Some(imbalance) = ratio(mdot_in, mdot_out).filter(|v| v.is_finite()) {
            self.monitor.observe(series::IMBALANCE, imbalance);
            self.monitor.set_mass_imbalance(imbalance);
        }
        if let Some(v) = ratio(q_in, q_out).filter(|v| v.is_finite()) {
            self.monitor.observe(series::VOLUME_IMBALANCE, v);
        }
        // How much of the volumetric gap the measured densities account for.
        // Formed from each plane's own flux-weighted mean density, so it is a
        // measurement rather than an estimate from the pressure drop, and it is
        // exactly the quantity that reconciles the two imbalances above.
        if let (Some(rho_in), Some(rho_out)) =
            (inlet.mean_density_kgm3(), outlet.mean_density_kgm3())
        {
            if rho_out.abs() > 1e-30 {
                self.monitor
                    .observe(series::EXPANSION, rho_in / rho_out - 1.0);
            }
        }

        let dp_total = inlet.total_pressure_pa - outlet.total_pressure_pa;
        let dp_static = inlet.static_pressure_pa - outlet.static_pressure_pa;
        self.monitor.observe(series::DP_TOTAL, dp_total);
        self.monitor.observe(series::DP_STATIC, dp_static);

        let u_in = inlet.bulk_velocity_ms;
        let u_out = outlet.bulk_velocity_ms;
        self.monitor.observe(series::U_IN, u_in);
        self.monitor.observe(series::U_OUT, u_out);
        let v_ref = self.cfg.reference.pick(u_in, u_out);
        self.monitor.observe(series::V_REF, v_ref);
        // Formed per frame, then averaged: dividing two averages would need a
        // covariance to get the error bar right, and would be wrong whenever the
        // flow rate and the pressure drop fluctuate together, which they do.
        let dyn_head = 0.5 * lu.rho_phys * v_ref * v_ref;
        if dyn_head > 1e-12 {
            self.monitor.observe(series::K, dp_total / dyn_head);
        }

        self.monitor.observe(series::GAMMA, outlet.uniformity);
        self.monitor.observe(series::CV, outlet.cv);
        self.monitor.observe(series::P05, outlet.p05_ms);
        self.monitor.observe(series::P95, outlet.p95_ms);
        self.monitor
            .observe(series::BACKFLOW, outlet.backflow_fraction);
        self.monitor
            .observe(series::DEFLECTION, outlet.deflection_deg);
        self.monitor
            .observe(series::CONE, outlet.cone_half_angle_deg);
        self.monitor.observe(
            series::THROW,
            throw_distance_m(
                u_out,
                outlet.open_area_mm2,
                self.cfg.throw_constant,
                self.cfg.throw_target_ms,
            ),
        );

        if let Some(v) = &snap.volume {
            self.observe_volume(v);
        }

        // V/Q, so the residence time panel has something to compare its measured
        // mean against. `V` is the passage the caller supplied — never the
        // volume pass's whole-domain fluid volume, which is the room as well as
        // the duct — and the *outlet* flow, which is the flow the tracers
        // actually left through. Unknown passage means an infinite tau_ideal,
        // which `Rtd::dead_volume_fraction` reports as `None`.
        self.rtd.set_tau_ideal(match self.cfg.passage_volume_mm3 {
            Some(v) if v > 0.0 => tau_ideal_s(v, q_out),
            _ => f64::INFINITY,
        });

        self.last_step = snap.step;
        self.frames += 1;
        self.monitor
            .set_flow_throughs(self.flow_throughs_at(snap.step));
        self.latest = Some(snap);
    }

    /// Fold one wall capture into the averages.
    ///
    /// [`crate::wall::WallMetrics`] is deliberately *not* owned by this struct:
    /// it needs the surface triangles, which come from `ad-geom` and which
    /// [`MetricsConfig`] does not carry — the metrics layer never has to know
    /// what a mesh is. The application drives the wall pass, decodes it with
    /// [`crate::wall::WallField`], and hands the summary here so that the four
    /// scalars anyone quotes from it (mean and peak shear, mean and peak `y+`)
    /// get the same time-averaging and error bars as everything else.
    pub fn observe_wall(&mut self, w: &WallSummary) {
        self.monitor.observe(series::WALL_SHEAR, w.mean_shear_pa);
        self.monitor.observe(series::WALL_SHEAR_MAX, w.max_shear_pa);
        self.monitor.observe(series::Y_PLUS, w.mean_y_plus);
        self.monitor.observe(series::Y_PLUS_MAX, w.max_y_plus);
        self.latest_wall = Some(*w);
    }

    fn observe_volume(&mut self, v: &VolumeReading) {
        self.monitor.observe(series::PEAK_SPEED, v.peak_speed_ms);
        self.monitor.observe(series::REVERSE, v.reverse_fraction);
        self.monitor.observe(series::STAGNANT, v.stagnant_fraction);
        if let Some(r) = v.residual {
            self.monitor.observe(series::RESIDUAL, r);
        }
    }

    /// Re-open the averaging window if anything reset the monitor since the last
    /// frame.
    ///
    /// Flow-throughs are the convergence gate's clock: below
    /// [`MonitorConfig::discard_flow_throughs`] every scalar reports
    /// [`Health::Transient`] and nothing is quotable. They must therefore be
    /// counted **from the last reset**, not from step zero.
    ///
    /// Counting from zero is not a small error, it inverts the gate. A reset
    /// happens precisely when a boundary condition changed, which is exactly
    /// when a new startup transient begins — and a run that had been going for
    /// twenty flow-throughs would carry that count straight across the reset,
    /// declare the transient long over, and start quoting converged averages of
    /// the first few frames after the change. The gate would be at its most
    /// confident at the one moment it is most wrong.
    ///
    /// Detected through [`Monitor::reset_count`] rather than by trusting the
    /// caller, because a reset arrives by two routes — the parameter hash from
    /// inside [`Self::observe`], and the application calling [`Monitor::reset`]
    /// on the slider — and only one of them passes through here.
    fn sync_window(&mut self, step: u64) {
        let resets = self.monitor.reset_count();
        if resets != self.resets_seen {
            self.resets_seen = resets;
            self.window_origin_step = step;
            // `frames` is documented as "since the last reset", and the series
            // behind it have just been emptied.
            self.frames = 0;
        }
    }

    /// Flow-throughs elapsed since the window opened.
    fn flow_throughs_at(&self, step: u64) -> f64 {
        let per = self
            .cfg
            .units
            .steps_per_flow_through(self.cfg.duct_length_mm);
        if per > 0.0 && per.is_finite() {
            step.saturating_sub(self.window_origin_step) as f64 / per
        } else {
            0.0
        }
    }

    /// Solver steps folded into the current averaging window: `last_step` minus
    /// the step the window opened at. The application's own window counter and
    /// this should agree; where they cannot (a reset between two readbacks),
    /// this is the one the convergence gate used.
    pub fn steps_in_window(&self) -> u64 {
        self.last_step.saturating_sub(self.window_origin_step)
    }

    fn est(&self, name: &str) -> Estimate {
        self.monitor.estimate(name).unwrap_or(Estimate::ZERO)
    }

    /// Everything, averaged, with error bars.
    pub fn report(&self) -> MetricsReport {
        let lu = &self.cfg.units;
        let last = self.latest.as_ref();
        let k = self.est(series::K);
        let inlet_last = last.map(|s| s.inlet().clone());
        let outlet_last = last.map(|s| s.outlet().clone());

        let d_h = inlet_last
            .as_ref()
            .map(|r| r.hydraulic_diameter_mm)
            .unwrap_or(0.0);

        MetricsReport {
            step: self.last_step,
            frames: self.frames,
            flow_throughs: self.monitor.flow_throughs(),
            health: self.monitor.overall(),
            health_reason: self
                .monitor
                .names()
                .map(str::to_string)
                .collect::<Vec<_>>()
                .into_iter()
                // First scalar with anything to say. `explain` is None for a
                // converged metric, so this lands on the first one that is not,
                // rather than on whichever happens to be first.
                .find_map(|n| self.monitor.explain(&n).map(|w| format!("{n}: {w}"))),
            converged_but_unbalanced: self.monitor.statistically_converged_but_unbalanced(),

            flow_in: Flow(self.est(series::Q_IN)),
            flow_out: Flow(self.est(series::Q_OUT)),
            mass_flow_in_kgs: self.est(series::MDOT_IN),
            mass_flow_out_kgs: self.est(series::MDOT_OUT),
            mass_imbalance: self.est(series::IMBALANCE),
            volumetric_imbalance: self.est(series::VOLUME_IMBALANCE),
            volumetric_expansion: self.est(series::EXPANSION),
            mass_balance_ok: self.monitor.mass_balance_ok(),

            total_pressure_drop_pa: self.est(series::DP_TOTAL),
            static_pressure_drop_pa: self.est(series::DP_STATIC),

            loss_coefficient: LossCoefficient {
                k,
                band: LossBand::of(k.mean),
                reference: self.cfg.reference,
                v_ref_ms: self.est(series::V_REF),
            },

            inlet_velocity_ms: self.est(series::U_IN),
            outlet_velocity_ms: self.est(series::U_OUT),

            uniformity: Uniformity {
                gamma: self.est(series::GAMMA),
                cv: self.est(series::CV),
                p05_ms: self.est(series::P05),
                p95_ms: self.est(series::P95),
                backflow_fraction: self.est(series::BACKFLOW),
                histogram: outlet_last
                    .as_ref()
                    .map(|r| r.histogram.clone())
                    .unwrap_or_default(),
                histogram_range_ms: outlet_last
                    .as_ref()
                    .map(|r| (r.hist_min_ms, r.hist_max_ms))
                    .unwrap_or((0.0, 0.0)),
            },

            jet: Jet {
                direction: outlet_last
                    .as_ref()
                    .map(|r| r.momentum_dir)
                    .unwrap_or(self.cfg.outlet.normal),
                deflection_deg: self.est(series::DEFLECTION),
                cone_half_angle_deg: self.est(series::CONE),
                throw_distance_m: self.est(series::THROW),
                throw_target_ms: self.cfg.throw_target_ms,
            },

            peak_speed_ms: self.est(series::PEAK_SPEED),
            reverse_volume_fraction: self.est(series::REVERSE),
            stagnant_volume_fraction: self.est(series::STAGNANT),
            residual: self.monitor.series(series::RESIDUAL).and_then(|s| s.last()),

            reynolds: lu.reynolds(d_h),
            mach_lb: lu.mach_lb(),
            tau0: lu.tau0,
            hydraulic_diameter_mm: d_h,

            wall: Wall {
                summary: self.latest_wall,
                mean_shear_pa: self.est(series::WALL_SHEAR),
                max_shear_pa: self.est(series::WALL_SHEAR_MAX),
                mean_y_plus: self.est(series::Y_PLUS),
                max_y_plus: self.est(series::Y_PLUS_MAX),
            },

            inlet: inlet_last,
            outlet: outlet_last,
            volume: last.and_then(|s| s.volume),
            warnings: self.warnings(),
        }
    }

    /// [`LatticeUnits::warnings`] plus anything the metrics layer itself can see
    /// is wrong with how it has been driven.
    fn warnings(&self) -> Vec<String> {
        let mut w = self.cfg.units.warnings();
        // The domain-volume mistake, caught rather than reported as 99% dead
        // volume. The passage is a hole through a part sitting in a room-sized
        // box; if it is within a factor of ten of every fluid cell in the
        // domain, what was passed is the domain.
        if let (Some(v), Some(domain)) = (
            self.cfg.passage_volume_mm3,
            self.latest
                .as_ref()
                .and_then(|s| s.volume)
                .map(|v| v.fluid_volume_mm3),
        ) {
            if domain > 0.0 && v > 0.1 * domain {
                w.push(format!(
                    "passage volume {v:.0} mm^3 is {:.0}% of the whole domain's fluid volume: \
                     that looks like the domain rather than the duct, and the ideal residence \
                     time is {:.0}x too long if it is",
                    v / domain * 100.0,
                    (v / domain).max(1.0)
                ));
            }
        }
        w
    }
}

/// ASHRAE throw: the distance at which a free jet's centreline velocity has
/// decayed to `target`.
///
/// `V_x = K V_0 sqrt(A_0) / x`, so `x = K V_0 sqrt(A_0) / V_x`. `K ~ 6` for a
/// compact opening. Returns 0 when the outlet is already slower than the target,
/// which is the honest answer — the jet never had a throw to speak of — rather
/// than a distance derived from a formula that does not apply.
///
/// This is a *correlation*, not a simulation result: the domain does not extend
/// far enough downstream to measure a metre of throw directly, and extrapolating
/// a decay law from the near field is exactly what the correlation is for.
pub fn throw_distance_m(v0_ms: f64, area_mm2: f64, k: f64, target_ms: f64) -> f64 {
    if !(v0_ms > target_ms) || !(area_mm2 > 0.0) || !(target_ms > 0.0) {
        return 0.0;
    }
    k * v0_ms * (area_mm2 * 1e-6).sqrt() / target_ms
}

/// A volumetric flow with an error bar, convertible without losing it.
///
/// The relative error is invariant under a unit change, so converting a mean
/// without converting its SEM would silently claim extra precision. Wrapping the
/// pair makes that impossible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flow(pub Estimate);

impl Flow {
    /// Cubic metres per second.
    pub fn m3s(&self) -> Estimate {
        self.0
    }
    /// Cubic feet per minute.
    pub fn cfm(&self) -> Estimate {
        scale(self.0, 2118.88)
    }
    /// Litres per second.
    pub fn litres_per_second(&self) -> Estimate {
        scale(self.0, 1000.0)
    }
}

impl std::fmt::Display for Flow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} L/s ({} CFM)", self.litres_per_second(), self.cfm())
    }
}

/// Multiply a mean and every dispersion measure by the same factor.
///
/// `n`, `n_eff` and `tau_int` are properties of the *sampling*, not of the unit,
/// so they pass through unchanged.
pub fn scale(e: Estimate, k: f64) -> Estimate {
    Estimate {
        mean: e.mean * k,
        sem: e.sem * k.abs(),
        std_dev: e.std_dev * k.abs(),
        n: e.n,
        n_eff: e.n_eff,
        tau_int: e.tau_int,
    }
}

/// The loss coefficient, with the reference it was normalised by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossCoefficient {
    pub k: Estimate,
    pub band: LossBand,
    pub reference: ReferenceVelocity,
    /// The reference velocity actually used, m/s.
    pub v_ref_ms: Estimate,
}

impl std::fmt::Display for LossCoefficient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "K = {} [{}] on V_ref = {} m/s ({}), {}",
            self.k,
            self.band.color(),
            self.v_ref_ms,
            self.reference.label(),
            self.band.describe()
        )
    }
}

/// Outlet profile quality.
#[derive(Debug, Clone, PartialEq)]
pub struct Uniformity {
    /// Weltens index, `gamma = 1 - sum(|w_i - w_bar| A_i)/(2 A w_bar)`.
    pub gamma: Estimate,
    /// Coefficient of variation of the through-plane velocity.
    pub cv: Estimate,
    pub p05_ms: Estimate,
    pub p95_ms: Estimate,
    pub backflow_fraction: Estimate,
    /// Most recent instantaneous histogram, `(centre m/s, area fraction)`.
    pub histogram: Vec<(f64, f64)>,
    pub histogram_range_ms: (f64, f64),
}

/// Where the air goes after it leaves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jet {
    /// Mass-flux-weighted momentum direction at the outlet, most recent frame.
    pub direction: Vec3,
    /// Angle between that direction and the outlet normal, degrees.
    pub deflection_deg: Estimate,
    /// Half-angle of the cone containing 90% of the forward momentum flux.
    pub cone_half_angle_deg: Estimate,
    /// Distance to [`Self::throw_target_ms`], metres, from the ASHRAE
    /// correlation. See [`throw_distance_m`].
    pub throw_distance_m: Estimate,
    pub throw_target_ms: f64,
}

/// Wall loading, averaged. Empty until [`DuctMetrics::observe_wall`] is fed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wall {
    /// The most recent capture, with the per-triangle aggregates.
    pub summary: Option<WallSummary>,
    /// Area-weighted mean wall shear stress, Pa.
    pub mean_shear_pa: Estimate,
    pub max_shear_pa: Estimate,
    pub mean_y_plus: Estimate,
    /// Peak `y+`. Above ~11 the first fluid node has left the viscous sublayer
    /// and CONTRACT.md's no-wall-function decision no longer holds.
    pub max_y_plus: Estimate,
}

impl Wall {
    pub fn measured(&self) -> bool {
        self.summary.is_some()
    }

    /// Whether the wall treatment is still inside its range of validity.
    pub fn resolves_viscous_sublayer(&self) -> bool {
        self.max_y_plus.mean < 11.0
    }
}

/// Everything, averaged. This is what the UI adapter reads.
#[derive(Debug, Clone)]
pub struct MetricsReport {
    pub step: u64,
    /// Frames folded into the averages since the last reset.
    pub frames: u64,
    pub flow_throughs: f64,
    pub health: Health,
    /// Why the light is not green, in words, or `None` when it is.
    ///
    /// Amber has two completely different causes with opposite remedies: the
    /// statistics have not settled (run longer), or the domain is leaking mass
    /// (running longer will never help). A single lamp cannot say which, and the
    /// second is the one worth acting on.
    pub health_reason: Option<String>,
    /// True when every statistical test passes and only the mass balance is
    /// holding the light amber -- the "stop and fix the boundary conditions"
    /// signal.
    pub converged_but_unbalanced: bool,

    /// Volumetric flow through the inlet plane. What a flow meter reads.
    pub flow_in: Flow,
    pub flow_out: Flow,
    /// Mass flow through the inlet plane, kg/s. What is conserved.
    pub mass_flow_in_kgs: Estimate,
    pub mass_flow_out_kgs: Estimate,
    /// `|mdot_in - mdot_out| / mdot_in`.
    /// **Under 1% or nothing else means anything.**
    pub mass_imbalance: Estimate,
    /// `|Q_in - Q_out| / Q_in`. Reported, never gated on: a weakly compressible
    /// solver expands the air across a real pressure drop, so this is nonzero on
    /// a perfectly conserving run. Compare it with
    /// [`Self::volumetric_expansion`]; what is left over is real.
    pub volumetric_imbalance: Estimate,
    /// `rho_in / rho_out - 1` from the two planes' flux-weighted mean densities:
    /// how much volumetric expansion the compressibility actually accounts for.
    pub volumetric_expansion: Estimate,
    pub mass_balance_ok: bool,

    /// Mass-flow-weighted total-pressure drop, Pa. The number to quote.
    pub total_pressure_drop_pa: Estimate,
    /// Area-weighted static-pressure drop, Pa. What a wall tapping measures.
    pub static_pressure_drop_pa: Estimate,

    pub loss_coefficient: LossCoefficient,
    pub inlet_velocity_ms: Estimate,
    pub outlet_velocity_ms: Estimate,

    pub uniformity: Uniformity,
    pub jet: Jet,

    pub peak_speed_ms: Estimate,
    pub reverse_volume_fraction: Estimate,
    pub stagnant_volume_fraction: Estimate,
    /// Latest `||du||/||u||`, not an average: a residual is read as a trace.
    pub residual: Option<f64>,

    /// Wall pressure, shear and `y+`. Populated only when the application
    /// drives [`crate::wall::WallMetrics`] and feeds
    /// [`DuctMetrics::observe_wall`].
    pub wall: Wall,

    pub reynolds: f64,
    pub mach_lb: f64,
    pub tau0: f64,
    pub hydraulic_diameter_mm: f64,

    /// Most recent instantaneous plane readings, for the patch panels.
    pub inlet: Option<PlaneReading>,
    pub outlet: Option<PlaneReading>,
    pub volume: Option<VolumeReading>,
    /// Verbatim from [`LatticeUnits::warnings`].
    pub warnings: Vec<String>,
}

impl MetricsReport {
    /// The contract's reduced record for the inlet plane, if one has arrived.
    pub fn inlet_sample(&self) -> Option<MetricSample> {
        self.inlet
            .as_ref()
            .map(|r| r.metric_sample(self.step as u32))
    }

    pub fn outlet_sample(&self) -> Option<MetricSample> {
        self.outlet
            .as_ref()
            .map(|r| r.metric_sample(self.step as u32))
    }

    /// Is the mass balance inside the 1% gate?
    pub fn trustworthy(&self) -> bool {
        self.mass_balance_ok && self.health != Health::Diverged
    }

    /// The volumetric imbalance the measured expansion does *not* explain.
    ///
    /// `|Q_in - Q_out| / Q_in` minus `|rho_in/rho_out - 1|`. Near zero means the
    /// two flow meters disagree purely because the air expanded on the way
    /// through; materially positive means they disagree for some other reason,
    /// and that reason is [`Self::mass_imbalance`]. Reported so the gap between
    /// the volumetric and mass figures is a number rather than a puzzle.
    pub fn unexplained_volumetric_imbalance(&self) -> f64 {
        self.volumetric_imbalance.mean - self.volumetric_expansion.mean.abs()
    }

    /// Do the volumetric and mass imbalances differ by enough to be worth
    /// saying? The 1% gate is the natural yardstick: a gap smaller than it
    /// changes no decision.
    pub fn compressibility_is_material(&self) -> bool {
        self.volumetric_imbalance.n > 0
            && (self.volumetric_imbalance.mean - self.mass_imbalance.mean).abs() > 0.01
    }

    /// Every scalar in the report, as `(name, estimate, unit)`, for a table or
    /// a CSV export. The order is the order a report should be read in.
    pub fn scalars(&self) -> Vec<(&'static str, Estimate, &'static str)> {
        let mut v = vec![
            ("flow in", self.flow_in.litres_per_second(), "L/s"),
            ("flow out", self.flow_out.litres_per_second(), "L/s"),
            ("mass flow in", scale(self.mass_flow_in_kgs, 1000.0), "g/s"),
            (
                "mass flow out",
                scale(self.mass_flow_out_kgs, 1000.0),
                "g/s",
            ),
            ("mass imbalance", self.mass_imbalance, "-"),
            ("volumetric imbalance", self.volumetric_imbalance, "-"),
            ("volumetric expansion", self.volumetric_expansion, "-"),
            ("total pressure drop", self.total_pressure_drop_pa, "Pa"),
            ("static pressure drop", self.static_pressure_drop_pa, "Pa"),
            ("loss coefficient", self.loss_coefficient.k, "-"),
            ("reference velocity", self.loss_coefficient.v_ref_ms, "m/s"),
            ("inlet bulk velocity", self.inlet_velocity_ms, "m/s"),
            ("outlet bulk velocity", self.outlet_velocity_ms, "m/s"),
            ("uniformity", self.uniformity.gamma, "-"),
            ("coefficient of variation", self.uniformity.cv, "-"),
            ("outlet P5", self.uniformity.p05_ms, "m/s"),
            ("outlet P95", self.uniformity.p95_ms, "m/s"),
            ("outlet backflow", self.uniformity.backflow_fraction, "-"),
            ("jet deflection", self.jet.deflection_deg, "deg"),
            ("90% cone half-angle", self.jet.cone_half_angle_deg, "deg"),
            ("throw distance", self.jet.throw_distance_m, "m"),
            ("peak speed", self.peak_speed_ms, "m/s"),
            ("reverse volume", self.reverse_volume_fraction, "-"),
            ("stagnant volume", self.stagnant_volume_fraction, "-"),
        ];
        if self.wall.measured() {
            v.push(("mean wall shear", self.wall.mean_shear_pa, "Pa"));
            v.push(("peak wall shear", self.wall.max_shear_pa, "Pa"));
            v.push(("mean y+", self.wall.mean_y_plus, "-"));
            v.push(("peak y+", self.wall.max_y_plus, "-"));
        }
        v
    }

    /// A few lines for a status bar or a log, every one carrying its error bar.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Q_in = {} L/s ({} CFM), Q_out = {} L/s, mass imbalance = {:.2}% [{}]\n",
            self.flow_in.litres_per_second(),
            self.flow_in.cfm(),
            self.flow_out.litres_per_second(),
            self.mass_imbalance.mean * 100.0,
            if self.mass_balance_ok {
                "ok"
            } else {
                "BROKEN: nothing below is trustworthy"
            }
        ));
        // Only when the two disagree by enough to change a decision. On an
        // incompressible-looking run this line would be noise; on the test part
        // it is the difference between a user chasing a conservation bug and a
        // user reading a compressibility figure.
        if self.compressibility_is_material() {
            s.push_str(&format!(
                "volumetric imbalance = {:.2}%, of which {:.2}% is the measured expansion \
                 rho_in/rho_out (weakly compressible solver, not lost mass)\n",
                self.volumetric_imbalance.mean * 100.0,
                self.volumetric_expansion.mean * 100.0
            ));
        }
        s.push_str(&format!(
            "dp_total = {} Pa, dp_static = {} Pa\n",
            self.total_pressure_drop_pa, self.static_pressure_drop_pa
        ));
        s.push_str(&format!("{}\n", self.loss_coefficient));
        s.push_str(&format!(
            "uniformity gamma = {}, CV = {}, P5/P95 = {} / {} m/s, backflow = {}\n",
            self.uniformity.gamma,
            self.uniformity.cv,
            self.uniformity.p05_ms,
            self.uniformity.p95_ms,
            self.uniformity.backflow_fraction
        ));
        s.push_str(&format!(
            "jet: deflection {} deg, 90% cone half-angle {} deg, throw to {:.2} m/s = {} m\n",
            self.jet.deflection_deg,
            self.jet.cone_half_angle_deg,
            self.jet.throw_target_ms,
            self.jet.throw_distance_m
        ));
        s.push_str(&format!(
            "peak |u| = {} m/s, reverse volume = {}, stagnant volume = {}\n",
            self.peak_speed_ms, self.reverse_volume_fraction, self.stagnant_volume_fraction
        ));
        if self.wall.measured() {
            s.push_str(&format!(
                "wall: tau = {} Pa mean, {} Pa peak; y+ = {} mean, {} peak ({})\n",
                self.wall.mean_shear_pa,
                self.wall.max_shear_pa,
                self.wall.mean_y_plus,
                self.wall.max_y_plus,
                if self.wall.resolves_viscous_sublayer() {
                    "viscous sublayer resolved"
                } else {
                    "OUTSIDE the viscous sublayer; shear is under-read"
                }
            ));
        }
        s.push_str(&format!(
            "state: {} after {} frames / {:.1} flow-throughs\n",
            self.health.traffic_light(),
            self.frames,
            self.flow_throughs
        ));
        if let Some(why) = &self.health_reason {
            s.push_str(&format!("  not green because: {why}\n"));
        }
        if self.converged_but_unbalanced {
            s.push_str(
                "  the statistics have settled; what is left is a domain problem, \
                 not a patience problem\n",
            );
        }
        for w in &self.warnings {
            s.push_str(&format!("warning: {w}\n"));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{field_layout, FieldRefs, FieldTextures};
    use ad_gpu::types::{flags, Bbox};

    fn units() -> LatticeUnits {
        LatticeUnits::for_air(1.0, 2.0, 0.05)
    }

    fn grid() -> Grid {
        Grid::covering(
            Bbox {
                min: Vec3::ZERO,
                max: Vec3::splat(48.0),
            },
            1.0,
        )
    }

    fn patches() -> (FlowPatch, FlowPatch) {
        // Two planes normal to +X, both facing downstream.
        let inlet = FlowPatch {
            center_mm: Vec3::new(12.0, 24.0, 24.0),
            normal: Vec3::X,
            half_u: Vec3::Y * 8.0,
            half_v: Vec3::Z * 8.0,
        };
        let outlet = FlowPatch {
            center_mm: Vec3::new(36.0, 24.0, 24.0),
            ..inlet
        };
        (inlet, outlet)
    }

    fn config() -> MetricsConfig {
        let (inlet, outlet) = patches();
        MetricsConfig::new(grid(), units(), inlet, outlet)
    }

    /// A synthetic plane reading with the fields the derivations use. The air on
    /// it is at the reference density, so `mdot = rho_phys Q` and the two
    /// imbalances agree; [`compressed`] is the case where they do not.
    fn reading(q: f64, u: f64, pt: f64, ps: f64) -> PlaneReading {
        PlaneReading {
            patch_area_mm2: 256.0,
            covered_fraction: 1.0,
            outside_fraction: 0.0,
            open_area_mm2: 256.0,
            hydraulic_diameter_mm: 16.0,
            flow_rate_m3s: q,
            mass_flow_kgs: q * units().rho_phys,
            bulk_velocity_ms: u,
            total_pressure_pa: pt,
            total_pressure_area_pa: pt,
            static_pressure_pa: ps,
            uniformity: 0.95,
            cv: 0.2,
            p05_ms: u * 0.6,
            p95_ms: u * 1.3,
            backflow_fraction: 0.0,
            max_speed_ms: u * 1.4,
            mean_speed_ms: u,
            momentum_dir: Vec3::X,
            deflection_deg: 0.0,
            cone_half_angle_deg: 12.0,
            histogram: Vec::new(),
            hist_min_ms: 0.0,
            hist_max_ms: u * 1.4,
            fluid_samples: 1000,
        }
    }

    /// A plane carrying `mdot` kg/s of air whose lattice density is `rho_lb`.
    /// The volumetric flow follows from the two, exactly as it does on a real
    /// plane, so a pair of these conserves mass by construction.
    fn compressed(mdot: f64, rho_lb: f64, pt: f64, ps: f64) -> PlaneReading {
        let rho = rho_lb * units().rho_phys;
        let q = mdot / rho;
        PlaneReading {
            flow_rate_m3s: q,
            mass_flow_kgs: mdot,
            bulk_velocity_ms: q / (256.0 * 1e-6),
            ..reading(q, q / (256.0 * 1e-6), pt, ps)
        }
    }

    /// A volume reading whose only load-bearing field is the fluid volume: the
    /// whole domain's, which is what seeding `tau_ideal` from this pass gets
    /// wrong.
    fn domain_volume(fluid_volume_mm3: f64) -> VolumeReading {
        VolumeReading {
            cells_visited: 20_000_000,
            fluid_cells: 9_000_000,
            fluid_volume_mm3,
            reverse_fraction: 0.01,
            stagnant_fraction: 0.2,
            peak_speed_ms: 11.4,
            mean_speed_ms: 1.2,
            rms_speed_ms: 2.0,
            mean_axial_ms: 0.8,
            min_pressure_pa: -20.0,
            max_pressure_pa: 100.0,
            residual: Some(2.0e-3),
            probe_cells: 140_000,
        }
    }

    /// Drive the derivations with synthetic readings — no GPU, no solver.
    fn cpu_metrics(cfg: MetricsConfig) -> DuctMetrics {
        DuctMetrics::headless(cfg)
    }

    #[test]
    fn the_loss_bands_sit_where_the_ashrae_reference_points_do() {
        // A well-radiused elbow.
        assert_eq!(LossBand::of(0.25), LossBand::Green);
        assert_eq!(LossBand::of(0.55), LossBand::Green);
        // Past ideal, inside the contract's K < 1 pass mark.
        assert_eq!(LossBand::of(0.6), LossBand::Amber);
        assert_eq!(LossBand::of(0.95), LossBand::Amber);
        // An uncut mitred bend, and anything worse.
        assert_eq!(LossBand::of(2.5), LossBand::Red);
        assert_eq!(LossBand::of(f64::NAN), LossBand::Red, "a NaN is not a pass");
        assert!(LossBand::Red.describe().contains("mitred"));
    }

    #[test]
    fn the_reference_velocity_changes_k_and_says_so() {
        // Inlet 2 m/s, outlet 3.7 m/s: a 1.85:1 contraction, as in the contract.
        assert_eq!(ReferenceVelocity::Inlet.pick(2.0, 3.7), 2.0);
        assert_eq!(ReferenceVelocity::Outlet.pick(2.0, 3.7), 3.7);
        assert_eq!(ReferenceVelocity::Faster.pick(2.0, 3.7), 3.7);
        // K scales as 1/V^2, so the choice is worth a factor of 3.4 here.
        let ratio = (3.7f64 / 2.0).powi(2);
        assert!((ratio - 3.42).abs() < 0.01, "{ratio}");
        assert!(ReferenceVelocity::Outlet.label().contains("outlet"));
    }

    #[test]
    fn flow_conversions_carry_the_error_bar_with_the_mean() {
        let e = Estimate {
            mean: 0.00635,
            sem: 0.00004,
            std_dev: 0.0004,
            n: 100,
            n_eff: 40.0,
            tau_int: 1.25,
        };
        let f = Flow(e);
        assert!((f.litres_per_second().mean - 6.35).abs() < 1e-9);
        assert!((f.cfm().mean - 13.4549).abs() < 1e-3);
        // The relative error is what must be invariant.
        for c in [f.m3s(), f.cfm(), f.litres_per_second()] {
            assert!(
                (c.relative_error() - e.relative_error()).abs() < 1e-12,
                "a unit change moved the relative error"
            );
            assert_eq!(c.n, e.n);
            assert_eq!(c.n_eff, e.n_eff);
        }
    }

    #[test]
    fn the_throw_correlation_scales_as_velocity_and_the_square_root_of_area() {
        // x = K V0 sqrt(A0) / V_target.
        let x = throw_distance_m(4.0, 2116.0, 6.0, 0.25);
        let want = 6.0 * 4.0 * (2116.0e-6f64).sqrt() / 0.25;
        assert!((x - want).abs() < 1e-9, "{x} vs {want}");
        // Doubling V0 doubles the throw; quadrupling the area doubles it too.
        assert!((throw_distance_m(8.0, 2116.0, 6.0, 0.25) / x - 2.0).abs() < 1e-9);
        assert!((throw_distance_m(4.0, 4.0 * 2116.0, 6.0, 0.25) / x - 2.0).abs() < 1e-9);
        // Already below the target: no throw, rather than a formula misapplied.
        assert_eq!(throw_distance_m(0.2, 2116.0, 6.0, 0.25), 0.0);
        assert_eq!(throw_distance_m(4.0, 0.0, 6.0, 0.25), 0.0);
    }

    /// The derivations, driven by hand: a duct passing 6.35 L/s with a 20 Pa
    /// total-pressure drop and a perfect mass balance.
    #[test]
    fn the_aggregate_derives_the_imbalance_pressure_drop_and_k_it_is_given() {
        let mut m = cpu_metrics(config());
        let lu = units();
        let (q, u_in, u_out) = (0.00635, 2.0, 3.71);
        for step in 0..600u64 {
            let snap = Snapshot {
                step,
                planes: vec![reading(q, u_in, 20.0, 18.0), reading(q, u_out, 0.0, -8.0)],
                volume: None,
            };
            m.observe(snap);
        }
        let r = m.report();
        assert!(
            r.mass_imbalance.mean.abs() < 1e-12,
            "{}",
            r.mass_imbalance.mean
        );
        assert!(r.mass_balance_ok);
        assert!((r.total_pressure_drop_pa.mean - 20.0).abs() < 1e-9);
        assert!((r.static_pressure_drop_pa.mean - 26.0).abs() < 1e-9);

        let want_k = 20.0 / (0.5 * lu.rho_phys * u_in * u_in);
        assert!(
            (r.loss_coefficient.k.mean - want_k).abs() < 1e-9,
            "K = {} vs {want_k}",
            r.loss_coefficient.k.mean
        );
        assert_eq!(r.loss_coefficient.reference, ReferenceVelocity::Inlet);
        assert!((r.loss_coefficient.v_ref_ms.mean - u_in).abs() < 1e-9);
        assert!(r.loss_coefficient.to_string().contains("inlet bulk"));
        assert!((r.flow_in.litres_per_second().mean - 6.35).abs() < 1e-9);

        // A constant series has zero spread, so the SEM is zero and the
        // formatter still refuses to invent digits.
        assert_eq!(r.total_pressure_drop_pa.sem, 0.0);
        assert!(r.summary().contains("dp_total"));
    }

    /// The wall pass is driven separately (it needs geometry the metrics layer
    /// has no business knowing about), but its four headline scalars must still
    /// arrive with error bars, and the report must not pretend to know them
    /// before anything has been fed in.
    #[test]
    fn wall_loading_joins_the_report_with_error_bars_or_not_at_all() {
        let mut m = cpu_metrics(config());
        for step in 0..200u64 {
            m.observe(Snapshot {
                step,
                planes: vec![
                    reading(0.00635, 2.0, 20.0, 18.0),
                    reading(0.00635, 3.71, 0.0, -8.0),
                ],
                volume: None,
            });
        }
        let before = m.report();
        assert!(!before.wall.measured(), "nothing has been measured yet");
        assert!(!before.summary().contains("wall:"));
        assert!(before.scalars().iter().all(|(n, _, _)| !n.contains("y+")));

        // A run whose peak y+ wanders either side of 6.
        for i in 0..200 {
            let jitter = ((i % 7) as f64 - 3.0) * 0.1;
            m.observe_wall(&crate::wall::WallSummary {
                triangles: 100,
                wetted_triangles: 90,
                wetted_area_mm2: 900.0,
                pressure_force_n: glam::DVec3::ZERO,
                viscous_force_n: glam::DVec3::new(1e-4, 0.0, 0.0),
                mean_shear_pa: 0.05 + 0.001 * jitter,
                max_shear_pa: 0.2,
                mean_pressure_pa: 12.0,
                mean_probe_distance_cells: 1.5,
                mean_y_plus: 3.0,
                max_y_plus: 6.0 + jitter,
            });
        }
        let r = m.report();
        assert!(r.wall.measured());
        assert!((r.wall.mean_y_plus.mean - 3.0).abs() < 1e-9);
        assert!(
            r.wall.max_shear_pa.sem == 0.0,
            "a constant peak has no scatter"
        );
        assert!(
            r.wall.mean_shear_pa.sem > 0.0,
            "a jittering mean must show one"
        );
        assert!(
            r.wall.resolves_viscous_sublayer(),
            "y+ ~ 6 is inside the sublayer"
        );
        assert!(r.summary().contains("viscous sublayer resolved"));
        let names: Vec<&str> = r.scalars().iter().map(|(n, _, _)| *n).collect();
        assert!(names.contains(&"peak y+") && names.contains(&"mean wall shear"));
    }

    /// A 5% mass imbalance must block convergence no matter how steady the rest
    /// of the numbers are. This is the gate the whole module is built around.
    #[test]
    fn a_broken_mass_balance_blocks_convergence_and_says_so() {
        let mut m = cpu_metrics(config());
        for step in 0..2000u64 {
            m.observe(Snapshot {
                step,
                planes: vec![
                    reading(0.00635, 2.0, 20.0, 18.0),
                    // 5% short.
                    reading(0.00635 * 0.95, 3.71, 0.0, -8.0),
                ],
                volume: None,
            });
        }
        m.monitor_mut().set_flow_throughs(20.0);
        let r = m.report();
        assert!((r.mass_imbalance.mean - 0.05).abs() < 1e-9);
        assert!(!r.mass_balance_ok);
        assert!(!r.trustworthy());
        assert_ne!(r.health, Health::Converged);
        assert!(r.summary().contains("BROKEN"));
    }

    /// The imbalance that gates everything must be a **mass** imbalance.
    ///
    /// The case is the contract's test part at 3 m/s: the inlet plane sits at
    /// +99.5 Pa (`rho - 1 = 0.070`) and the outlet at -1.8 Pa, so the same air
    /// leaves as a 7.1% larger volume. Mass is conserved here *exactly*, by
    /// construction. Differencing volumetric flows called that a 7.1%
    /// conservation error — seven times the gate — and blocked convergence on a
    /// run that was perfectly healthy.
    #[test]
    fn a_compressible_duct_conserves_mass_while_its_volume_flow_expands() {
        let lu = units();
        let (rho_in_lb, rho_out_lb) = (1.0700f64, 0.9987f64);
        let want_expansion = rho_in_lb / rho_out_lb - 1.0;
        assert!((want_expansion - 0.0714).abs() < 1e-3, "{want_expansion}");

        // One mass flow, two planes, two different densities.
        let mdot = 0.00635 * lu.rho_phys * rho_in_lb;
        let mut m = cpu_metrics(config());
        for step in 0..600u64 {
            m.observe(Snapshot {
                step,
                planes: vec![
                    compressed(mdot, rho_in_lb, 120.0, 99.5),
                    compressed(mdot, rho_out_lb, 105.0, -1.8),
                ],
                volume: None,
            });
        }
        let r = m.report();

        // The gate sees conserved mass and passes.
        assert!(
            r.mass_imbalance.mean.abs() < 1e-12,
            "mass imbalance {}",
            r.mass_imbalance.mean
        );
        assert!(
            r.mass_balance_ok,
            "a mass-conserving duct must pass the gate"
        );
        assert!(r.trustworthy());

        // The volumetric imbalance is emphatically *not* zero, and is still
        // reported: it is what a pair of flow meters would read.
        assert!(
            (r.volumetric_imbalance.mean - want_expansion).abs() < 1e-9,
            "volumetric imbalance {}",
            r.volumetric_imbalance.mean
        );
        assert!(
            r.volumetric_imbalance.mean > 0.05,
            "the two must not have been conflated"
        );
        assert!(
            r.flow_out.m3s().mean > r.flow_in.m3s().mean,
            "the air expanded"
        );

        // ...and the difference between them is accounted for, not left hanging.
        assert!((r.volumetric_expansion.mean - want_expansion).abs() < 1e-9);
        assert!(
            r.unexplained_volumetric_imbalance().abs() < 1e-9,
            "the expansion should explain all of it, {} left over",
            r.unexplained_volumetric_imbalance()
        );
        assert!(r.compressibility_is_material());
        assert!(
            r.summary().contains("volumetric imbalance"),
            "{}",
            r.summary()
        );

        // The mass flows themselves survive to the report, both equal to what
        // was fed in.
        assert!((r.mass_flow_in_kgs.mean - mdot).abs() < 1e-15);
        assert!((r.mass_flow_out_kgs.mean - mdot).abs() < 1e-15);

        // Genuinely lost mass still fails, at the same 1% gate as before: half a
        // percent of the outlet mass flow removed on top of the expansion.
        let mut leaky = cpu_metrics(config());
        for step in 0..600u64 {
            leaky.observe(Snapshot {
                step,
                planes: vec![
                    compressed(mdot, rho_in_lb, 120.0, 99.5),
                    compressed(mdot * 0.95, rho_out_lb, 105.0, -1.8),
                ],
                volume: None,
            });
        }
        let r = leaky.report();
        assert!(
            (r.mass_imbalance.mean - 0.05).abs() < 1e-9,
            "{}",
            r.mass_imbalance.mean
        );
        assert!(
            !r.mass_balance_ok,
            "5% of the mass went missing and nothing noticed"
        );
    }

    /// A reset mid-run must re-arm the transient gate.
    ///
    /// The averages are thrown away precisely when a boundary condition
    /// changed — which is exactly when a new startup transient begins. Counting
    /// flow-throughs from the absolute solver step carried the old count across
    /// the reset, so the gate declared the transient long over and started
    /// quoting averages of the first few frames after the change.
    #[test]
    fn a_mid_run_reset_re_arms_the_transient_gate() {
        let mut m = cpu_metrics(config());
        let discard = m.monitor().config().discard_flow_throughs;
        let per = units().steps_per_flow_through(m.config().duct_length_mm);
        assert!(per > 0.0 && per.is_finite(), "steps per flow-through {per}");

        // Ten flow-throughs of a perfectly steady duct: well past the gate.
        let frame = || Snapshot {
            step: 0,
            planes: vec![
                reading(0.00635, 2.0, 20.0, 18.0),
                reading(0.00635, 3.71, 0.0, -8.0),
            ],
            volume: None,
        };
        let end = (per * 10.0) as u64;
        for i in 0..200u64 {
            m.observe(Snapshot {
                step: i * end / 200,
                ..frame()
            });
        }
        assert!(
            m.report().flow_throughs > discard,
            "{}",
            m.report().flow_throughs
        );
        assert_ne!(
            m.report().health,
            Health::Transient,
            "the run really was past the transient"
        );

        // The user drags the inlet slider. The application clears the averages,
        // and the solver keeps counting from where it was.
        m.monitor_mut().reset();
        for i in 0..40u64 {
            m.observe(Snapshot {
                step: end + i,
                ..frame()
            });
        }

        let r = m.report();
        assert!(
            r.flow_throughs < 1.0,
            "flow-throughs survived the reset: {} (per flow-through {per} steps, reset at {end})",
            r.flow_throughs
        );
        assert_eq!(
            r.health,
            Health::Transient,
            "the gate must discard the transient the reset just started"
        );
        assert_eq!(r.frames, 40, "frames are counted from the reset too");
        assert_eq!(m.steps_in_window(), 39);
    }

    /// `tau_ideal = V/Q` is the *passage* over the flow, not the domain over the
    /// flow.
    ///
    /// The volume pass counts every fluid cell in the 260 x 180 x 180 mm box —
    /// room air included — which on the test part is about seventy times the
    /// duct's own passage. Seeding `tau_ideal` from it made the dead-volume
    /// fraction read 99% for any real flow, since `1 - t_bar/tau_ideal` with a
    /// `tau_ideal` seventy times too large is 99% whatever the tracers did.
    #[test]
    fn the_ideal_residence_time_comes_from_the_passage_not_the_domain() {
        const PASSAGE_MM3: f64 = 128_250.0;
        const DOMAIN_MM3: f64 = 9_000_000.0;

        let mut cfg = config();
        cfg.passage_volume_mm3 = Some(PASSAGE_MM3);
        let mut m = cpu_metrics(cfg);

        let q = 0.00635;
        m.observe(Snapshot {
            step: 1,
            planes: vec![reading(q, 2.0, 20.0, 18.0), reading(q, 3.71, 0.0, -8.0)],
            volume: Some(domain_volume(DOMAIN_MM3)),
        });

        let want = tau_ideal_s(PASSAGE_MM3, q);
        assert!(
            (want - 0.0202).abs() < 1e-3,
            "20 ms through a 128 cm^3 passage: {want}"
        );
        assert!(
            (m.rtd().tau_ideal_s() - want).abs() / want < 1e-12,
            "tau_ideal = {} s, passage V/Q = {want} s (domain V/Q would be {} s)",
            m.rtd().tau_ideal_s(),
            tau_ideal_s(DOMAIN_MM3, q)
        );

        // ...so a plug-flow tracer population reads as no dead volume, rather
        // than as 99% of the duct not participating.
        for _ in 0..100 {
            m.rtd_mut().record(crate::rtd::AgeSample::new(want, 1.0));
        }
        let dead = m
            .rtd()
            .dead_volume_fraction()
            .expect("a known passage gives a known tau");
        assert!(dead.abs() < 1e-9, "dead volume {dead}");

        // Handed the domain volume by mistake, the report says so rather than
        // quietly reporting 99% dead volume.
        let mut cfg = m.config().clone();
        cfg.passage_volume_mm3 = Some(DOMAIN_MM3);
        m.set_config(cfg);
        m.observe(Snapshot {
            step: 2,
            planes: vec![reading(q, 2.0, 20.0, 18.0), reading(q, 3.71, 0.0, -8.0)],
            volume: Some(domain_volume(DOMAIN_MM3)),
        });
        assert!(
            m.report()
                .warnings
                .iter()
                .any(|w| w.contains("looks like the domain")),
            "{:?}",
            m.report().warnings
        );

        // No passage volume at all: unknown, not a plausible number.
        let mut cfg = m.config().clone();
        cfg.passage_volume_mm3 = None;
        m.set_config(cfg);
        m.observe(Snapshot {
            step: 3,
            planes: vec![reading(q, 2.0, 20.0, 18.0), reading(q, 3.71, 0.0, -8.0)],
            volume: Some(domain_volume(DOMAIN_MM3)),
        });
        assert!(!m.rtd().tau_ideal_s().is_finite());
        assert!(m.rtd().dead_volume_fraction().is_none());
    }

    /// Changing the operating point must clear the average rather than blending
    /// two flows into a confident-looking mean. Without this, dragging the inlet
    /// slider produces a converged answer for a duct that does not exist.
    #[test]
    fn changing_the_inlet_velocity_throws_the_average_away() {
        let mut m = cpu_metrics(config());
        for step in 0..500u64 {
            m.observe(Snapshot {
                step,
                planes: vec![
                    reading(0.004, 2.0, 10.0, 9.0),
                    reading(0.004, 3.7, 0.0, -4.0),
                ],
                volume: None,
            });
        }
        assert!((m.report().flow_in.m3s().mean - 0.004).abs() < 1e-12);

        let mut cfg = m.config().clone();
        cfg.units = LatticeUnits::for_air(1.0, 5.0, 0.05);
        m.set_config(cfg);
        for step in 500..1000u64 {
            m.observe(Snapshot {
                step,
                planes: vec![
                    reading(0.010, 5.0, 60.0, 55.0),
                    reading(0.010, 9.3, 0.0, -20.0),
                ],
                volume: None,
            });
        }
        let r = m.report();
        assert!(
            (r.flow_in.m3s().mean - 0.010).abs() < 1e-12,
            "the old operating point survived: {}",
            r.flow_in.m3s().mean
        );
        assert_eq!(r.frames, 500);
    }

    /// A noisy but stationary run: the mean is right, the error bar is honest,
    /// and the correction for autocorrelation is in force.
    #[test]
    fn a_fluctuating_duct_reports_a_mean_with_a_real_error_bar() {
        let mut m = cpu_metrics(config());
        // Deterministic AR(1) noise so the test cannot flake.
        let (mut x, phi) = (0.0f64, 0.8);
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for step in 0..4000u64 {
            x = phi * x + (1.0 - phi * phi).sqrt() * next();
            let dp = 20.0 + 0.5 * x;
            m.observe(Snapshot {
                step,
                planes: vec![
                    reading(0.00635, 2.0, dp, dp - 2.0),
                    reading(0.00635, 3.71, 0.0, -8.0),
                ],
                volume: None,
            });
        }
        let r = m.report();
        assert!(
            (r.total_pressure_drop_pa.mean - 20.0).abs() < 0.1,
            "{}",
            r.total_pressure_drop_pa
        );
        assert!(
            r.total_pressure_drop_pa.sem > 0.0,
            "a fluctuating signal must have an error bar"
        );
        // Correlated samples: fewer effective ones than raw ones.
        assert!(
            r.total_pressure_drop_pa.n_eff < r.total_pressure_drop_pa.n as f64 * 0.5,
            "N_eff = {} of N = {}",
            r.total_pressure_drop_pa.n_eff,
            r.total_pressure_drop_pa.n
        );
        assert!(r.total_pressure_drop_pa.tau_int > 0.5);
        // ...and the display never shows more digits than the SEM supports.
        let text = format!("{}", r.total_pressure_drop_pa);
        assert!(text.contains("+/-"), "{text}");
    }

    /// End to end on the GPU: uniform flow through two identical parallel
    /// planes. `Q_in` and `Q_out` must match to the quadrature's precision, the
    /// pressure drop must be zero, and `K` with it.
    #[test]
    fn a_uniform_channel_conserves_mass_and_loses_no_pressure() {
        let Some(gpu) = crate::test_gpu() else { return };
        let g = grid();
        let cfg = config();
        let tex = FieldTextures::new_exact(&gpu.device, g);
        let u_lb = 0.05f32;
        tex.fill(&gpu.queue, |_, _| (Vec3::X * u_lb, 1.0, flags::FLUID))
            .unwrap();

        let layout = field_layout(&gpu.device);
        let (vv, dv) = (tex.velocity_view(), tex.density_view());
        let group = crate::field::field_bind_group(
            &gpu.device,
            &layout,
            &FieldRefs {
                grid: g,
                velocity: &vv,
                density: &dv,
                flags: tex.flags_buffer(),
            },
        );
        let mut m = DuctMetrics::with_samples(&gpu.device, &layout, cfg, 256).unwrap();

        for step in 0..3u64 {
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            m.record(&gpu.queue, &mut enc, &group, step, None).unwrap();
            gpu.queue.submit([enc.finish()]);
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            m.poll(&gpu.device);
        }
        // Anything still in flight.
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        m.poll(&gpu.device);

        let r = m.report();
        assert!(r.frames > 0, "no frames were folded");
        assert!(
            r.mass_imbalance.mean < 1e-4,
            "imbalance {} between two identical planes",
            r.mass_imbalance.mean
        );
        assert!(r.mass_balance_ok);
        // Zero to the precision of two independent fp32 plane accumulations;
        // the dynamic head here is 2.37 Pa, so this is 4 decimal places of it.
        assert!(
            r.total_pressure_drop_pa.mean.abs() < 1e-3,
            "dp = {} Pa across nothing",
            r.total_pressure_drop_pa.mean
        );
        assert!(r.loss_coefficient.k.mean.abs() < 1e-3);

        let want_q = u_lb as f64 * units().c_u() * (256.0 * 1e-6);
        assert!(
            (r.flow_in.m3s().mean / want_q - 1.0).abs() < 2e-3,
            "Q = {} vs analytic {want_q}",
            r.flow_in.m3s().mean
        );
        // Uniform flow: the profile is perfect and points along the normal.
        assert!((r.uniformity.gamma.mean - 1.0).abs() < 1e-3);
        assert!(r.jet.deflection_deg.mean < 0.1);
        assert!(
            r.peak_speed_ms.mean > 0.0,
            "the volume pass produced nothing"
        );
        assert_eq!(r.reverse_volume_fraction.mean, 0.0);
        // ...and the contract's reduced record comes out of the same reading.
        let s = r.outlet_sample().expect("an outlet sample should exist");
        assert!((s.flow_rate as f64 / want_q - 1.0).abs() < 2e-3);
        assert!((s.litres_per_second() as f64 / (want_q * 1000.0) - 1.0).abs() < 2e-3);
    }
}

//! The view models: plain data the UI renders, and the **only** thing Wave 3's
//! adapter has to fill.
//!
//! # Why these exist at all
//!
//! `ad-metrics` and the flow-viz overlays are being written in parallel with
//! this crate, so their types do not exist yet. Coding the panels against a
//! guess at someone else's API would produce a UI that compiles once and then
//! rots. Instead every number the UI can display lives in a struct defined
//! *here*, in terms the UI needs (a value, its uncertainty, a traffic light, a
//! unit label). Wave 3 writes one adapter that fills them; nothing in
//! `panels/` ever learns where a number came from.
//!
//! That indirection also buys the tests. Every formatting rule, threshold and
//! delta in this crate is exercised without a window, a GPU or a solver,
//! because the input is a struct literal.
//!
//! # The rule that shapes everything here
//!
//! **Every scalar carries its uncertainty.** [`Reading`] has no constructor
//! that takes a bare value without also taking a standard error, and
//! [`crate::format::uncertain`] is the only way the UI is allowed to print one.
//! A lattice-Boltzmann duct simulation produces a *fluctuating* signal; quoting
//! `47.3184 Pa` off one time step is not precision, it is noise with decimal
//! places. Making the uncertainty structurally unavoidable is cheaper than
//! remembering to add it.

use glam::Vec3;

/// The traffic light next to a number.
///
/// Deliberately four-valued: [`Health::Unknown`] is not [`Health::Good`]. A
/// statistic that has not accumulated enough samples yet must look different
/// from one that has and is fine, or the user reads a warm-up transient as a
/// result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Health {
    /// Not enough data yet, or no threshold defined for this quantity.
    #[default]
    Unknown,
    /// Within the design target.
    Good,
    /// Outside the target but not alarming.
    Watch,
    /// Outside anything acceptable, or physically suspicious.
    Bad,
}

impl Health {
    /// Linear-space RGBA for the label. Chosen to stay distinguishable under
    /// deuteranopia: the amber and green differ in luminance as well as hue,
    /// and red is the darkest of the three.
    pub fn color(self) -> [f32; 4] {
        match self {
            Health::Unknown => [0.62, 0.64, 0.68, 1.0],
            Health::Good => [0.36, 0.82, 0.47, 1.0],
            Health::Watch => [0.95, 0.72, 0.18, 1.0],
            Health::Bad => [0.94, 0.33, 0.31, 1.0],
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Health::Unknown => "unknown",
            Health::Good => "good",
            Health::Watch => "watch",
            Health::Bad => "bad",
        }
    }

    /// Classify `v` against a band where **lower is better**: good below `good`,
    /// bad above `bad`.
    pub fn below(v: f64, good: f64, bad: f64) -> Health {
        if !v.is_finite() {
            Health::Unknown
        } else if v <= good {
            Health::Good
        } else if v >= bad {
            Health::Bad
        } else {
            Health::Watch
        }
    }

    /// Classify `v` where **higher is better**: good above `good`, bad below
    /// `bad`.
    pub fn above(v: f64, good: f64, bad: f64) -> Health {
        if !v.is_finite() {
            Health::Unknown
        } else if v >= good {
            Health::Good
        } else if v <= bad {
            Health::Bad
        } else {
            Health::Watch
        }
    }
}

/// One measured scalar: the mean, its standard error, and a judgement.
///
/// `sem` is the standard error **of the mean**, not the standard deviation of
/// the signal. Those differ by `sqrt(n)` and confusing them is the classic way
/// to publish an error bar an order of magnitude too wide. Wave 3's adapter is
/// responsible for dividing; this crate only renders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    /// Mean over the averaging window, in the canonical SI unit for the
    /// quantity (see each field of [`MetricsView`]).
    pub value: f64,
    /// Standard error of the mean, same unit. Negative or non-finite means
    /// "not known yet" and renders as a bare value with a `~` marker.
    pub sem: f64,
    pub state: Health,
    /// Independent samples behind `value`. Zero means the window is empty, and
    /// the UI greys the number out rather than showing a confident zero.
    pub samples: u32,
}

impl Default for Reading {
    fn default() -> Self {
        Self::unknown()
    }
}

impl Reading {
    /// A reading with no data behind it. This is what a fresh app shows, and
    /// what a statistics reset returns every metric to.
    pub const fn unknown() -> Self {
        Self { value: f64::NAN, sem: f64::NAN, state: Health::Unknown, samples: 0 }
    }

    pub const fn new(value: f64, sem: f64, state: Health, samples: u32) -> Self {
        Self { value, sem, state, samples }
    }

    /// True when there is something worth showing. A reading with no samples
    /// is not the same as a reading of zero.
    pub fn is_known(&self) -> bool {
        self.samples > 0 && self.value.is_finite()
    }

    /// True when `sem` is a usable error bar.
    pub fn has_error_bar(&self) -> bool {
        self.sem.is_finite() && self.sem > 0.0
    }

    /// Relative standard error, as a fraction. `None` when either half is
    /// unusable or the value is (near) zero, where a relative error is
    /// meaningless rather than merely large.
    pub fn relative_error(&self) -> Option<f64> {
        if !self.has_error_bar() || !self.value.is_finite() || self.value.abs() < 1e-12 {
            return None;
        }
        Some(self.sem / self.value.abs())
    }

    /// Whether two readings differ by more than their combined error bars.
    ///
    /// The test is `|a - b| > k * sqrt(sem_a^2 + sem_b^2)`. This is what stops
    /// the A/B panel from painting a green arrow on a change that is pure
    /// noise, which on a design tool is worse than showing nothing: it invites
    /// the user to keep a modification that did nothing.
    pub fn differs_from(&self, other: &Reading, k: f64) -> bool {
        if !self.is_known() || !other.is_known() {
            return false;
        }
        let combined = (self.sem.max(0.0).powi(2) + other.sem.max(0.0).powi(2)).sqrt();
        (self.value - other.value).abs() > k * combined.max(f64::MIN_POSITIVE)
    }
}

/// Where the averaging window stands, and why it last restarted.
///
/// Surfaced because silently averaging across a parameter change produces a
/// confidently wrong number. The UI shows this, and [`crate::toast`] shouts
/// about it, precisely so that failure mode cannot happen quietly.
#[derive(Debug, Clone, PartialEq)]
pub struct StatsWindowView {
    /// Solver steps since the window opened.
    pub steps_in_window: u64,
    /// Steps the window wants before it will call itself converged.
    pub steps_target: u64,
    /// Flow-throughs of the domain accumulated, and the target. The headline
    /// convergence number for a duct: a statistic averaged over less than a
    /// couple of flow-throughs has not seen the flow it is describing.
    pub flow_throughs: f64,
    pub flow_throughs_target: f64,
    /// Why the window last restarted. `None` on a fresh run.
    pub last_reset: Option<ResetCause>,
    /// Steps ago the last reset happened, for "reset 4,200 steps ago".
    pub steps_since_reset: u64,
}

impl Default for StatsWindowView {
    fn default() -> Self {
        Self {
            steps_in_window: 0,
            steps_target: 0,
            flow_throughs: 0.0,
            flow_throughs_target: 15.0,
            last_reset: None,
            steps_since_reset: 0,
        }
    }
}

impl StatsWindowView {
    /// Fraction of the way to the flow-through target, clamped to `[0, 1]`.
    pub fn progress(&self) -> f32 {
        if self.flow_throughs_target <= 0.0 {
            return 0.0;
        }
        (self.flow_throughs / self.flow_throughs_target).clamp(0.0, 1.0) as f32
    }
}

/// Why the statistics window restarted.
///
/// The variants are the *user-visible causes*, not the internal mechanism,
/// because the toast quotes them verbatim and "the parameter hash changed" is
/// not a sentence anyone wants to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetCause {
    InletVelocity,
    /// The louver aim, or the install pose while a louver aim is set: either
    /// changes the angle the air enters the part at.
    InletAngle,
    InletOutletSwapped,
    /// A vent's speed or aim.
    VentChanged,
    GeometryChanged,
    /// The simulated box's margins.
    DomainChanged,
    ResolutionChanged,
    FluidChanged,
    SolverRestarted,
    Manual,
}

impl ResetCause {
    pub fn message(self) -> &'static str {
        match self {
            ResetCause::InletVelocity => "inlet velocity changed",
            ResetCause::InletAngle => "inlet air angle changed",
            ResetCause::InletOutletSwapped => "inlet and outlet swapped",
            ResetCause::VentChanged => "vent air changed",
            ResetCause::GeometryChanged => "geometry changed",
            ResetCause::DomainChanged => "simulated box changed",
            ResetCause::ResolutionChanged => "grid resolution changed",
            ResetCause::FluidChanged => "fluid properties changed",
            ResetCause::SolverRestarted => "solver restarted",
            ResetCause::Manual => "statistics reset by hand",
        }
    }
}

/// How the run is doing, independent of any one metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvergenceState {
    /// Transient still washing through; nothing quoted yet.
    #[default]
    Warmup,
    /// Accumulating statistics.
    Averaging,
    /// Residuals flat and the window long enough.
    Converged,
    /// NaN or a runaway velocity. The one state that must be impossible to
    /// miss, because every number downstream of it is meaningless.
    Diverged,
}

impl ConvergenceState {
    pub fn label(self) -> &'static str {
        match self {
            ConvergenceState::Warmup => "warming up",
            ConvergenceState::Averaging => "averaging",
            ConvergenceState::Converged => "converged",
            ConvergenceState::Diverged => "DIVERGED",
        }
    }
    pub fn health(self) -> Health {
        match self {
            ConvergenceState::Warmup => Health::Unknown,
            ConvergenceState::Averaging => Health::Watch,
            ConvergenceState::Converged => Health::Good,
            ConvergenceState::Diverged => Health::Bad,
        }
    }
}

/// A named trace for the convergence plot.
///
/// Two parallel `Vec`s rather than a `Vec<(f64, f64)>` because ImPlot takes
/// separate x and y slices, and an interleaved layout would need a copy at
/// every frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trace {
    pub name: String,
    /// Unit of `y`, for the axis label. Empty for a dimensionless residual.
    pub unit: String,
    /// Solver step (or seconds; the panel labels the axis from
    /// [`ConvergenceView::x_is_steps`]).
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// Plot this one on a log axis. Residuals span decades; flow rates do not.
    pub log_y: bool,
}

impl Trace {
    pub fn new(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self { name: name.into(), unit: unit.into(), x: Vec::new(), y: Vec::new(), log_y: false }
    }

    /// Append a point, discarding the oldest once `capacity` is reached.
    ///
    /// A plain `Vec` with a front-drain rather than a ring buffer: ImPlot wants
    /// contiguous slices in draw order, and at the few-thousand-point sizes a
    /// trace reaches, the memmove is far cheaper than the two-slice plumbing a
    /// ring would need at every call site.
    pub fn push(&mut self, x: f64, y: f64, capacity: usize) {
        self.x.push(x);
        self.y.push(y);
        let cap = capacity.max(2);
        if self.x.len() > cap {
            let drop = self.x.len() - cap;
            self.x.drain(..drop);
            self.y.drain(..drop);
        }
    }

    pub fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn last(&self) -> Option<(f64, f64)> {
        Some((*self.x.last()?, *self.y.last()?))
    }
}

/// Everything the convergence panel draws.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConvergenceView {
    pub state: ConvergenceState,
    /// Residual traces, typically `|du|/u` and `|drho|/rho`.
    pub residuals: Vec<Trace>,
    /// Inlet and outlet flow rate against step. Their *separation* is the
    /// honest convergence signal for a duct: mass in must equal mass out, and
    /// the gap between the two curves is a direct measure of how far from
    /// steady the solution still is.
    pub flow_in: Trace,
    pub flow_out: Trace,
    /// Pressure drop against step.
    pub pressure_drop: Trace,
    /// True when the x axis is solver steps rather than seconds.
    pub x_is_steps: bool,
    /// Mass imbalance `|mdot_in - mdot_out| / mdot_in`, dimensionless. The single
    /// number that says whether the solution is trustworthy at all.
    ///
    /// Deliberately mass flux, not volume flux. The flow is weakly compressible,
    /// so the *volumetric* rates differ by the density ratio across the duct --
    /// on this part about 7% at 3 m/s -- which is real physics, not solver error.
    /// Gating on volume flow would therefore condemn a perfectly conserved
    /// solution. The volumetric imbalance and the expansion that explains it are
    /// reported separately.
    pub mass_imbalance: Reading,
}

/// One inlet or outlet plane, as the toolbar and HUD need it.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchView {
    pub name: String,
    /// True open area, mm^2 — the hole, not its bounding rectangle.
    pub open_area_mm2: f32,
    pub hydraulic_diameter_mm: f32,
    pub center_mm: Vec3,
    /// Unit normal pointing into the fluid.
    pub normal: Vec3,
    /// Bulk velocity through the plane, m/s.
    pub mean_velocity: Reading,
    /// Peak speed on the plane, m/s.
    pub max_velocity: Reading,
    /// Fraction of the area flowing the wrong way. Non-zero at an outlet means
    /// recirculation is being sucked back in, which invalidates the pressure
    /// reading there.
    pub backflow_fraction: Reading,
    /// Air comes in here: the chosen inlet mouth, or every mouth a vent is
    /// sealed to. Set by the app from the lattice it built.
    pub is_inlet: bool,
    /// The mouth the outflow face is taken from. Set by the app.
    pub is_outlet: bool,
}

impl Default for PatchView {
    fn default() -> Self {
        Self {
            is_inlet: false,
            is_outlet: false,
            name: String::new(),
            open_area_mm2: 0.0,
            hydraulic_diameter_mm: 0.0,
            center_mm: Vec3::ZERO,
            normal: Vec3::X,
            mean_velocity: Reading::unknown(),
            max_velocity: Reading::unknown(),
            backflow_fraction: Reading::unknown(),
        }
    }
}

/// The cell-size control.
///
/// Apply-on-commit, never live. Changing `dx` reallocates every lattice buffer,
/// re-voxelises the part and restarts the statistics — seconds of stall and
/// possibly gigabytes of VRAM — so it has to happen once, on purpose, rather
/// than on every pixel of a slider drag. The panel holds what has been typed;
/// the app fills in [`Self::estimate`] so the cost is on screen before the
/// button is pressed.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionPanel {
    /// Cell size in the input box, mm. Nothing happens to it until Apply.
    pub pending_dx_mm: f32,
    /// What `pending_dx_mm` would cost. `None` until the app has looked.
    pub estimate: Option<ResolutionEstimate>,
}

impl ResolutionPanel {
    /// One-click tiers: 1 mm for a quick look, 0.75 mm the default, and 0.5,
    /// 0.4 and 0.3 mm the fine end of the resolution study.
    pub const PRESETS_MM: [f32; 5] = [1.0, 0.75, 0.5, 0.4, 0.3];
    /// Finest cell size the control accepts. 0.2 mm is already past what a
    /// 24 GB card holds for this part in FP32; the pre-flight reports the real
    /// figure for whatever is loaded.
    pub const MIN_MM: f32 = 0.2;
    /// Coarsest. 3 mm leaves about two cells across this part's 6.3 mm median
    /// passage, which is a picture of the flow and nothing more.
    pub const MAX_MM: f32 = 3.0;

    pub fn new(dx_mm: f32) -> Self {
        Self { pending_dx_mm: dx_mm, estimate: None }
    }

    /// The estimate, if it describes the cell size in the box *and* the box
    /// margins the user is asking for.
    pub fn estimate_for(&self, domain_mm: Option<[f32; 6]>) -> Option<&ResolutionEstimate> {
        self.current_estimate().filter(|e| {
            e.domain_mm.map(|m| m.map(f32::to_bits)) == domain_mm.map(|m| m.map(f32::to_bits))
        })
    }

    /// The estimate, if it describes what is in the box right now.
    pub fn current_estimate(&self) -> Option<&ResolutionEstimate> {
        self.estimate
            .as_ref()
            .filter(|e| e.dx_mm.to_bits() == self.pending_dx_mm.to_bits())
    }
}

/// What a candidate cell size would cost, worked out before anything is
/// allocated.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionEstimate {
    /// The cell size this describes, mm.
    pub dx_mm: f32,
    /// The box margins it was costed with; `None` is the automatic box.
    pub domain_mm: Option<[f32; 6]>,
    /// Lattice dimensions, cells.
    pub dims: [u32; 3],
    /// Lattice cells.
    pub cells: u64,
    /// Device memory the lattice needs once built, bytes.
    pub vram_bytes: u64,
    /// Device memory the app lets itself use, bytes. `None` when the GPU was
    /// not recognised and `AERODUCT_VRAM_GB` is unset; the memory check is then
    /// skipped.
    pub budget_bytes: Option<u64>,
    /// Expected solver steps per second. `None` with neither a measured rate to
    /// scale nor a known memory bandwidth to predict from.
    pub steps_per_s: Option<f64>,
    /// Expected wall-clock seconds per flow-through at that rate.
    pub flow_through_s: Option<f64>,
    /// Base relaxation time at the current operating point.
    pub tau0: f64,
    /// Worth knowing, but the change is allowed.
    pub warnings: Vec<String>,
    /// Why the change will be refused. `None` means it fits.
    pub blocker: Option<String>,
}

/// A histogram ready to draw: `edges.len() == counts.len() + 1`.
///
/// Pre-binned rather than raw samples, because the sample set lives on the GPU
/// and the UI must never pull a million values across the bus to draw forty
/// bars.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HistogramView {
    pub name: String,
    pub unit: String,
    pub edges: Vec<f64>,
    pub counts: Vec<f64>,
    /// Area-weighted mean of the underlying quantity, for the annotation line.
    pub mean: Reading,
}

impl HistogramView {
    pub fn is_valid(&self) -> bool {
        self.edges.len() >= 2 && self.counts.len() + 1 == self.edges.len()
    }

    /// Bin centres, which is what ImPlot's bar plot wants for x.
    pub fn centers(&self) -> Vec<f64> {
        if !self.is_valid() {
            return Vec::new();
        }
        self.edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect()
    }

    pub fn bin_width(&self) -> f64 {
        if self.edges.len() < 2 {
            0.0
        } else {
            (self.edges[self.edges.len() - 1] - self.edges[0]) / (self.edges.len() - 1) as f64
        }
    }
}

/// Residence-time distribution: how long air actually spends inside the duct.
///
/// Kept separate from a plain [`HistogramView`] because the two summary
/// statistics — mean residence time and the variance ratio — are what the panel
/// is really for. A long tail on `E(t)` is a stagnant pocket, and that is a
/// design defect no pressure-drop number reveals.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResidenceTimeView {
    pub histogram: HistogramView,
    /// First moment of `E(t)`, seconds.
    pub mean_residence_s: Reading,
    /// `sigma^2 / mean^2`. 0 is plug flow, 1 is a perfectly stirred tank.
    /// Anything above ~0.5 in a duct means a large recirculation.
    pub variance_ratio: Reading,
    /// Ideal residence time from volume over flow rate, seconds. The reference
    /// the measured mean should be compared against.
    pub ideal_residence_s: f64,
    /// Particles that never left, as a fraction. Directly the trapped volume.
    pub trapped_fraction: Reading,
}

/// A one-sided spectrum, for the acoustics work the contract anticipates.
///
/// A stub on purpose: the panel draws whatever is in here and labels itself as
/// provisional. Wave 3 fills `freq_hz`/`magnitude` from a probe's pressure
/// history; nothing else in the UI has to change.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpectrumView {
    pub source: String,
    pub freq_hz: Vec<f64>,
    /// Magnitude in dB re 20 uPa when `db` is set, otherwise linear Pa.
    pub magnitude: Vec<f64>,
    pub db: bool,
    /// Highest frequency the time step can represent, `1 / (2 dt)`. Drawn as a
    /// vertical marker so nobody reads a number above it.
    pub nyquist_hz: f64,
    /// Samples the transform was taken over. Zero means "not computed yet".
    pub window_samples: u32,
}

/// A point probe dropped in the viewport.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeView {
    pub id: u32,
    pub label: String,
    pub position_mm: Vec3,
    /// Static pressure history, Pa.
    pub pressure: Trace,
    /// Speed history, m/s.
    pub speed: Trace,
    /// Instantaneous velocity vector, m/s, for the arrow drawn at the probe.
    pub velocity_ms: Vec3,
    /// False when the probe landed inside solid, where every reading is a lie.
    pub in_fluid: bool,
    pub visible: bool,
    /// Selected in the list, so the gizmo attaches to it.
    pub selected: bool,
}

impl ProbeView {
    pub fn new(id: u32, position_mm: Vec3) -> Self {
        Self {
            id,
            label: format!("probe {id}"),
            position_mm,
            pressure: Trace::new("p", "Pa"),
            speed: Trace::new("|u|", "m/s"),
            velocity_ms: Vec3::ZERO,
            in_fluid: true,
            visible: true,
            selected: false,
        }
    }
}

/// The engineering numbers, all of them, in canonical SI.
///
/// **Units are fixed here and converted only at the moment of display.** Wave 3
/// fills SI; [`crate::format`] converts to CFM or inches of water for the
/// screen. Storing display units would make the A/B deltas wrong the moment
/// someone toggles the unit system mid-comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsView {
    /// Volumetric flow, m^3/s. Displayed as CFM or L/s.
    pub flow_rate: Reading,
    /// Total-pressure drop inlet to outlet, Pa.
    pub pressure_drop: Reading,
    /// Loss coefficient `K = dp / (0.5 rho U^2)`, dimensionless. The contract's
    /// design target is `K < 1`, ideally 0.3-0.6.
    pub loss_coefficient: Reading,
    /// Weltens uniformity index at the outlet, dimensionless, 1 = perfectly
    /// uniform.
    pub uniformity: Reading,
    /// Angle between the outlet momentum vector and the outlet normal, degrees.
    pub deflection_deg: Reading,
    /// Measured direction of the air leaving the outlet, unit, in the
    /// **lattice** frame; the UI turns it through the install pose to say where
    /// it points in the car. `None` until the outlet has been measured.
    pub jet_direction: Option<Vec3>,
    /// Peak speed anywhere in the domain, m/s. Watched because a runaway here
    /// is the first visible symptom of an instability.
    pub max_speed: Reading,
    /// Reynolds number on the inlet hydraulic diameter. Deterministic, not
    /// measured, so it is a bare `f64`.
    pub reynolds: f64,
    /// Lattice Mach number. Same.
    pub mach_lb: f64,
    /// Base relaxation time. Same. Shown because `tau -> 0.5` is the single
    /// best predictor of an unstable run.
    pub tau0: f64,

    pub inlet: PatchView,
    pub outlet: PatchView,
    pub convergence: ConvergenceView,
    pub stats: StatsWindowView,

    pub outlet_velocity_histogram: HistogramView,
    pub residence_time: ResidenceTimeView,
    pub spectrum: SpectrumView,

    /// Verbatim strings from `LatticeUnits::warnings()`, plus anything the
    /// adapter wants to add. Shown in the status bar and, if non-empty, as a
    /// persistent badge — they explain an instability *before* it happens.
    pub warnings: Vec<String>,
}

impl Default for MetricsView {
    fn default() -> Self {
        Self {
            flow_rate: Reading::unknown(),
            pressure_drop: Reading::unknown(),
            loss_coefficient: Reading::unknown(),
            uniformity: Reading::unknown(),
            deflection_deg: Reading::unknown(),
            jet_direction: None,
            max_speed: Reading::unknown(),
            reynolds: f64::NAN,
            mach_lb: f64::NAN,
            tau0: f64::NAN,
            inlet: PatchView::default(),
            outlet: PatchView::default(),
            convergence: ConvergenceView::default(),
            stats: StatsWindowView::default(),
            outlet_velocity_histogram: HistogramView::default(),
            residence_time: ResidenceTimeView::default(),
            spectrum: SpectrumView::default(),
            warnings: Vec::new(),
        }
    }
}

impl MetricsView {
    /// Apply the contract's design targets to every metric that has one.
    ///
    /// Kept here rather than in the adapter so the thresholds are in one place
    /// and are unit-tested. `Health::Unknown` is left alone for anything with
    /// no sensible target (flow rate depends entirely on the operating point).
    pub fn apply_default_thresholds(&mut self) {
        // K < 1 is the contract's pass mark; 0.6 is the "ideal" ceiling. An
        // uncut mitred bend runs 2.0-3.5, so anything above 1.5 is a genuine
        // failure rather than a merely disappointing design.
        self.loss_coefficient.state =
            Health::below(self.loss_coefficient.value, 0.6, 1.5);
        // Weltens uniformity: 0.95 is a good outlet, below 0.8 is a jet with a
        // dead zone beside it.
        self.uniformity.state = Health::above(self.uniformity.value, 0.95, 0.8);
        // Deflection from the outlet normal.
        self.deflection_deg.state = Health::below(self.deflection_deg.value, 5.0, 15.0);
        // Peak speed against the inlet bulk. A well-behaved 1.85:1 contraction
        // peaks around 2-3x inlet; 6x means separation or an instability.
        let u_in = self.inlet.mean_velocity.value;
        if u_in.is_finite() && u_in > 1e-6 {
            self.max_speed.state =
                Health::below(self.max_speed.value / u_in, 3.5, 6.0);
        }
        // Mass imbalance: 1% is fine, 5% means the answer is not converged.
        self.convergence.mass_imbalance.state =
            Health::below(self.convergence.mass_imbalance.value.abs(), 0.01, 0.05);
        // Backflow at the outlet invalidates the pressure reading there.
        self.outlet.backflow_fraction.state =
            Health::below(self.outlet.backflow_fraction.value, 0.005, 0.05);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_reading_is_not_a_zero_reading() {
        // The distinction the whole HUD rests on: a metric that has not
        // accumulated samples must not render as a confident number.
        let r = Reading::unknown();
        assert!(!r.is_known());
        assert!(!r.has_error_bar());
        assert_eq!(r.state, Health::Unknown);
        assert_eq!(Reading::new(0.0, 0.1, Health::Good, 0).is_known(), false);
        assert!(Reading::new(0.0, 0.1, Health::Good, 12).is_known());
    }

    #[test]
    fn traffic_lights_are_ordered_and_reject_nonfinite() {
        assert_eq!(Health::below(0.4, 0.6, 1.5), Health::Good);
        assert_eq!(Health::below(1.0, 0.6, 1.5), Health::Watch);
        assert_eq!(Health::below(2.0, 0.6, 1.5), Health::Bad);
        assert_eq!(Health::below(f64::NAN, 0.6, 1.5), Health::Unknown);

        assert_eq!(Health::above(0.99, 0.95, 0.8), Health::Good);
        assert_eq!(Health::above(0.9, 0.95, 0.8), Health::Watch);
        assert_eq!(Health::above(0.5, 0.95, 0.8), Health::Bad);
        // Exactly on a boundary counts as the better verdict, so a target of
        // "K <= 0.6" reads green at 0.6 rather than amber.
        assert_eq!(Health::below(0.6, 0.6, 1.5), Health::Good);
        assert_eq!(Health::above(0.95, 0.95, 0.8), Health::Good);
    }

    #[test]
    fn contract_targets_land_on_the_expected_colours() {
        // The numbers in CONTRACT.md, driven through the real thresholds.
        let mut m = MetricsView::default();
        m.inlet.mean_velocity = Reading::new(3.0, 0.01, Health::Unknown, 100);
        m.loss_coefficient = Reading::new(0.58, 0.02, Health::Unknown, 100);
        m.uniformity = Reading::new(0.91, 0.01, Health::Unknown, 100);
        m.deflection_deg = Reading::new(3.2, 0.2, Health::Unknown, 100);
        m.max_speed = Reading::new(18.4, 0.3, Health::Unknown, 100);
        m.apply_default_thresholds();

        assert_eq!(m.loss_coefficient.state, Health::Good, "K = 0.58 is in the ideal band");
        assert_eq!(m.uniformity.state, Health::Watch, "gamma = 0.91 is not great");
        assert_eq!(m.deflection_deg.state, Health::Good);
        // 18.4 / 3.0 = 6.1x the inlet bulk: that is the red one in the mock-up.
        assert_eq!(m.max_speed.state, Health::Bad);
    }

    #[test]
    fn a_mitred_bend_fails_the_loss_coefficient_target() {
        // ASHRAE puts an unvaned mitred 90 at K = 2.0-3.5, which must read red.
        let mut m = MetricsView::default();
        m.loss_coefficient = Reading::new(2.6, 0.1, Health::Unknown, 500);
        m.apply_default_thresholds();
        assert_eq!(m.loss_coefficient.state, Health::Bad);
    }

    #[test]
    fn differs_from_respects_the_error_bars() {
        // The A/B panel's gate. Two readings a hair apart with fat error bars
        // are the same reading.
        let a = Reading::new(47.3, 0.6, Health::Good, 200);
        let b = Reading::new(47.6, 0.6, Health::Good, 200);
        assert!(!a.differs_from(&b, 2.0), "0.3 Pa apart with +/-0.6 is noise");
        let c = Reading::new(52.0, 0.6, Health::Good, 200);
        assert!(a.differs_from(&c, 2.0), "4.7 Pa apart with +/-0.6 is real");
        // An unknown reading never "differs": there is nothing to compare.
        assert!(!a.differs_from(&Reading::unknown(), 2.0));
    }

    #[test]
    fn a_trace_drops_the_oldest_points_when_full() {
        let mut t = Trace::new("residual", "");
        for i in 0..10 {
            t.push(i as f64, (i * i) as f64, 4);
        }
        assert_eq!(t.x.len(), 4);
        assert_eq!(t.y.len(), 4);
        assert_eq!(t.x[0], 6.0, "the window must slide, not restart");
        assert_eq!(t.last(), Some((9.0, 81.0)));
        // A degenerate capacity must not panic or produce an empty trace.
        t.push(10.0, 100.0, 0);
        assert_eq!(t.x.len(), 2);
    }

    #[test]
    fn histogram_geometry_is_self_consistent() {
        let h = HistogramView {
            name: "U_out".into(),
            unit: "m/s".into(),
            edges: vec![0.0, 1.0, 2.0, 3.0],
            counts: vec![5.0, 9.0, 2.0],
            mean: Reading::new(1.4, 0.05, Health::Unknown, 300),
        };
        assert!(h.is_valid());
        assert_eq!(h.centers(), vec![0.5, 1.5, 2.5]);
        assert!((h.bin_width() - 1.0).abs() < 1e-12);

        // One count too many is the classic off-by-one; it must be rejected
        // rather than drawn misaligned.
        let bad = HistogramView { counts: vec![1.0, 2.0, 3.0, 4.0], ..h };
        assert!(!bad.is_valid());
    }

    #[test]
    fn stats_progress_is_clamped_and_safe_at_zero_target() {
        let mut s = StatsWindowView { flow_throughs: 7.3, flow_throughs_target: 15.0, ..Default::default() };
        assert!((s.progress() - 7.3 / 15.0).abs() < 1e-6);
        s.flow_throughs = 40.0;
        assert_eq!(s.progress(), 1.0);
        s.flow_throughs_target = 0.0;
        assert_eq!(s.progress(), 0.0, "no target must not divide by zero");
    }
}

//! AeroDuct UI shell.
//!
//! An Ansys-Discovery-Live-shaped interface over a live lattice-Boltzmann
//! solver: the simulation is **always running**, edits apply to it live, and
//! there is no Run button and no modal dialog anywhere in the crate.
//!
//! ```text
//! TOOLBAR:  play/pause/step/reset | steps-per-frame | cell size + Apply
//!           view presets | inlet A<->B | U slider
//! HUD:      Q | dp | K | gamma | theta | U_max     (each with its error bar and traffic light)
//! +----------+-------------------------------------+--------------+
//! | LAYERS   |            3D VIEWPORT              | PROPERTIES   |
//! |          |                        +---------+  | colormap     |
//! |          |                        | LEGEND  |  | range/opacity|
//! +----------+------------------------+---------+--+--------------+
//! | [Q_in/Q_out] [dp] [probe p(t)] [RTD E(t)]   flow-throughs 7.3/15 |
//! +---------------------------------------------------------------+
//! ```
//!
//! # How it is put together
//!
//! | layer | what it is | tested without a window |
//! |---|---|---|
//! | [`view`] | plain view models Wave 3 fills | yes |
//! | [`format`], [`compare`], [`params`], [`autotune`], [`layers`], [`legend`], [`camera_input`], [`overlays`], [`toast`] | the logic | yes |
//! | [`state`] | the aggregate, plus the action queue | yes |
//! | [`platform`] | winit events into ImGui | partly |
//! | [`backend`] | ImGui + ImPlot + ImGuizmo on wgpu | no |
//! | [`panels`] | the widgets | the dock layout only |
//!
//! The split is deliberate and load-bearing: everything above `platform` is
//! ordinary data and pure functions, so the parts that are easy to get subtly
//! wrong — uncertainty rounding, A/B significance, the parameter-change
//! detection that resets statistics — are covered by unit tests that need no
//! GPU and no display.
//!
//! # What Wave 3 has to do
//!
//! Fill [`view::MetricsView`], [`state::UiState::mouths`] and the probe traces
//! from `ad-metrics` and the flow-viz overlays, and service the
//! [`state::UiAction`] queue. Nothing in [`panels`] refers to a type from
//! either crate, and this crate does not depend on `ad-metrics` at all, so
//! neither of them can break this one — see the comment in `Cargo.toml`.
//! `crates/ad-app/src/metrics.rs` is the single file that connects them.
//!
//! # Three rules the code enforces rather than documents
//!
//! 1. **No modal dialogs, ever.** File pickers are the one native modal, and
//!    they are the OS's, not ours.
//! 2. **Every scalar carries its uncertainty.** [`view::Reading`] has no
//!    bare-value constructor and [`format::uncertain`] is the only renderer.
//! 3. **A statistics reset is impossible to miss.** [`state::UiState::sync_params`]
//!    is the single place a parameter change is detected, and it raises the
//!    toast itself so no future code path can forget to.

pub mod autotune;
pub mod backend;
pub mod camera_input;
pub mod compare;
pub mod format;
pub mod layers;
pub mod legend;
pub mod overlays;
pub mod panels;
pub mod params;
pub mod platform;
pub mod pose;
pub mod state;
pub mod toast;
pub mod view;

pub use autotune::{AutoTuneConfig, Clock, RateMeter, StepAutoTuner};
pub use backend::{UiBackend, UiFrame};
pub use camera_input::{CameraInput, Gesture, MouseState, PointerOwnership};
pub use compare::{Baseline, Better, Delta, MetricDeltas, Significance};
pub use format::{Quantity, UnitSystem};
pub use layers::{Layer, LayerId, LayerKind, LayerStack};
pub use legend::{Handle, LegendRange};
pub use overlays::{
    GizmoMode, GizmoSpace, IsoSettings, ParticleSeed, ParticleSettings, Placement, Probes,
    SliceSettings, StreamlineSettings, VentSettings,
};
pub use params::{ParamChange, ParamWatcher, SimParams};
pub use platform::WinitPlatform;
pub use pose::InstallPose;
pub use state::{PanelVisibility, PlotTab, UiAction, UiState};
pub use toast::{Severity, Toast, Toasts};
pub use view::{
    ConvergenceState, ConvergenceView, Health, HistogramView, MetricsView, PatchView, ProbeView,
    Reading, ResetCause, ResidenceTimeView, ResolutionEstimate, ResolutionPanel, SpectrumView,
    StatsWindowView, Trace,
};

/// Version of the Dear ImGui the app is linked against, for the About line.
pub fn imgui_version() -> &'static str {
    dear_imgui_rs::dear_imgui_version()
}

/// Whether the linked Dear ImGui has docking compiled in.
///
/// Checked at run time and reported in the diagnostics panel rather than
/// assumed: the layout degrades to plain floating windows without it, and a
/// silent degradation is exactly the sort of thing that gets mistaken for a
/// layout bug.
pub const HAS_DOCKING: bool = dear_imgui_rs::HAS_DOCKING;

//! The whole UI state, and the action list the app drains each frame.
//!
//! # Why actions instead of callbacks
//!
//! The panels do not own the solver, the renderer or the scene, and they must
//! not: an immediate-mode panel runs inside an ImGui frame, where taking a
//! mutable borrow of half the application to service a button press is both
//! awkward and a good way to end up rebuilding a solver in the middle of
//! recording a draw list.
//!
//! Instead every panel pushes a [`UiAction`] and returns. The app drains the
//! queue at a point of its choosing — after the UI frame, before the solver
//! step — where it has clean mutable access to everything. That also makes the
//! interesting half of the UI testable: a test can call the state machine,
//! inspect the actions it emitted, and never open a window.
//!
//! # What is *not* in here
//!
//! Anything the renderer or solver already owns: the camera, the transfer
//! functions, the mesh visibility. The UI reads and edits those in place
//! through the borrow the app hands it. Duplicating them here would create two
//! sources of truth for the same value, and the copy the user is dragging would
//! be the wrong one about half the time.

use std::path::PathBuf;

use ad_render::{DerivedField, ViewPreset};
use glam::Vec3;

use crate::autotune::{Clock, StepAutoTuner};
use crate::camera_input::CameraInput;
use crate::compare::{Baseline, MetricDeltas};
use crate::format::UnitSystem;
use crate::layers::{LayerId, LayerStack};
use crate::legend::LegendRange;
use crate::overlays::{
    GizmoMode, GizmoSpace, IsoSettings, ParticleSettings, Placement, Probes, SliceSettings,
    StreamlineSettings, VentSettings,
};
use crate::params::{ParamWatcher, SimParams};
use crate::pose::{DuctGeometry, InstallPose};
use crate::toast::Toasts;
use crate::view::{MetricsView, PatchView, ResetCause, ResolutionPanel};

/// Something the app must do. Drained once per frame.
#[derive(Debug, Clone, PartialEq)]
pub enum UiAction {
    /// Play / pause the sim. The sim being *always on* is the model, so pause
    /// exists only for inspecting a transient, never as a prerequisite to
    /// editing anything.
    SetPlaying(bool),
    /// Advance exactly `n` steps while paused.
    StepOnce(u32),
    /// Reinitialise the field to equilibrium and zero the step counter.
    ResetSolver,
    /// Throw away accumulated statistics, with a reason for the toast.
    ResetStatistics(ResetCause),

    /// Hot-apply a new inlet velocity, m/s.
    SetInletVelocity(f32),
    /// Make mouth `i` the inlet; the other detected mouth becomes the outlet.
    SetInletMouth(usize),
    /// One-click swap of the two auto-detected mouths.
    SwapInletOutlet,
    /// New cell size, mm. Reallocates.
    SetResolution(f32),
    /// New lattice velocity (`u_lb`).
    SetLatticeVelocity(f64),
    /// Smagorinsky constant.
    SetSmagorinsky(f32),

    ApplyViewPreset(ViewPreset),
    /// Fit the camera to the scene.
    FrameScene,
    SetField(DerivedField),
    /// Auto-range the legend to the field's measured extremes.
    AutoRangeLegend,

    /// Load the main duct STL, replacing whatever is loaded.
    LoadDuct(PathBuf),
    /// Add an obstruction STL.
    LoadObstruction(PathBuf),
    RemoveObstruction(usize),
    SetObstructionPlacement(usize, Placement),
    /// Add a vent at the inlet mouth. Removal is an edit of
    /// [`UiState::vents`]; the app notices.
    AddVent,
    /// Turn the part itself by the quarter turns nearest the install pose, and
    /// take those out of the pose, so the picture stays put while the mouths
    /// move to other lattice faces. Rebuilds.
    BakePose,

    /// Rebuild with these six box margins, mm (`[-x, +x, -y, +y, -z, +z]`,
    /// part frame), if they fit. Pre-flighted like [`Self::SetResolution`].
    SetDomainMargins([f32; 6]),
    /// Back to the automatic box.
    ResetDomainMargins,

    /// Drop a probe where a viewport ray hits the flow. The app does the
    /// picking, because it owns the depth buffer and the geometry.
    ProbeFromRay {
        origin_mm: Vec3,
        direction: Vec3,
    },
    MoveProbe {
        id: u32,
        position_mm: Vec3,
    },
    RemoveProbe(u32),

    AddSlice,
    RemoveSlice(usize),

    /// Freeze the current metrics as the A/B baseline.
    SaveBaseline,
    ClearBaseline,

    /// Write a supersampled PNG.
    Screenshot,
    RequestQuit,
}

/// Which bottom-strip plot is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlotTab {
    #[default]
    Flow,
    PressureDrop,
    Probes,
    Residence,
    Histogram,
    Spectrum,
}

impl PlotTab {
    pub fn label(self) -> &'static str {
        match self {
            PlotTab::Flow => "Q_in / Q_out",
            PlotTab::PressureDrop => "dp",
            PlotTab::Probes => "probe p(t)",
            PlotTab::Residence => "RTD E(t)",
            PlotTab::Histogram => "U_out histogram",
            PlotTab::Spectrum => "spectrum",
        }
    }
    pub const ALL: [PlotTab; 6] = [
        PlotTab::Flow,
        PlotTab::PressureDrop,
        PlotTab::Probes,
        PlotTab::Residence,
        PlotTab::Histogram,
        PlotTab::Spectrum,
    ];
}

/// Which panels are open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelVisibility {
    pub layers: bool,
    pub properties: bool,
    pub plots: bool,
    pub hud: bool,
    pub legend: bool,
    pub diagnostics: bool,
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            layers: true,
            properties: true,
            plots: true,
            hud: true,
            legend: true,
            diagnostics: false,
        }
    }
}

/// Everything the UI owns.
pub struct UiState {
    /// The simulation runs unless this is false. There is no "Run" button; this
    /// is a pause for inspecting a transient.
    pub playing: bool,
    pub tuner: StepAutoTuner,
    pub clock: Clock,

    /// The parameters as the user has them set. The app compares this against
    /// [`Self::watcher`] to decide what to apply.
    pub params: SimParams,
    watcher: ParamWatcher,

    pub layers: LayerStack,
    pub probes: Probes,
    pub slices: Vec<SliceSettings>,
    pub particles: ParticleSettings,
    pub iso: IsoSettings,
    pub streamlines: StreamlineSettings,

    pub gizmo_mode: GizmoMode,
    pub gizmo_space: GizmoSpace,
    /// Translation snap in mm; `None` disables snapping. 0.5 mm is a useful
    /// default on a part whose median passage is 6.3 mm wide.
    pub gizmo_snap_mm: Option<f32>,
    /// Rotation snap in degrees; `None` disables it.
    pub gizmo_snap_deg: Option<f32>,

    /// Where the duct sits in the car: the lattice (STL) frame onto the world
    /// frame. The solver never sees it; see [`crate::pose`].
    pub install: InstallPose,
    /// The point `install` turns about, lattice mm: the duct's bounding-box
    /// centre. Written by the app whenever the geometry is built.
    pub install_pivot_mm: Vec3,
    /// Vent louver aim, degrees, car terms: `[up_down, sideways]`. The source
    /// of truth for the inlet air angle; [`Self::sync_inlet_tilt`] turns it
    /// into the part-frame tilt the solver takes.
    pub inlet_louver_deg: [f32; 2],
    /// Where each obstruction sits in the car (world frame), indexed like
    /// `LayerKind::Obstruction`. Edited in place by the gizmo and the
    /// properties panel; the app voxelises a change once it has settled.
    pub obstructions: Vec<Placement>,
    /// The part's own scale and quarter turns in the lattice: a real change
    /// of geometry, rebuilt when it settles. See [`crate::pose::DuctGeometry`].
    pub duct: DuctGeometry,
    /// The vents in the room (car frame), indexed like `LayerKind::Vent`.
    /// Edited in place; the app rebuilds when one's cells settle and
    /// hot-applies when only its air changed.
    pub vents: Vec<VentSettings>,

    /// The simulated box's six margins beyond the duct as built, mm,
    /// `[-x, +x, -y, +y, -z, +z]` in the part's frame. Written by the app.
    pub domain_margins_mm: [f32; 6],
    /// The margins in the box, waiting for Apply.
    pub pending_domain_mm: [f32; 6],

    pub legend: LegendRange,
    pub units: UnitSystem,
    pub camera_input: CameraInput,
    pub field: DerivedField,

    /// Filled by Wave 3's adapter. Everything the HUD and the plots read.
    pub metrics: MetricsView,
    /// The detected mouths, in detection order. Index 0 and 1 are what the
    /// inlet toggle switches between.
    pub mouths: Vec<PatchView>,
    /// The cell-size control: what is typed in it, and what the app says it
    /// would cost.
    pub resolution: ResolutionPanel,

    pub baseline: Option<Baseline>,
    /// Recomputed each frame when a baseline exists.
    pub deltas: Option<MetricDeltas>,

    pub toasts: Toasts,
    pub panels: PanelVisibility,
    pub plot_tab: PlotTab,

    /// Screen-space rectangle of the 3D viewport: `[x, y, w, h]` in physical
    /// pixels. Written by the viewport panel each frame; read by the gizmo and
    /// by click-to-probe, both of which need to convert screen to NDC.
    pub viewport_rect: [f32; 4],
    /// True while ImGui wants the mouse.
    pub ui_capture_mouse: bool,
    /// True while a gizmo is hovered or being dragged.
    pub gizmo_active: bool,
    /// Physical pixels per logical (ImGui) pixel: the window's DPI factor.
    /// `viewport_rect` and the cursor are physical; anything drawn through
    /// ImGui — the gizmo, the viewport overlays — wants logical.
    pub ui_scale: f32,

    /// One line per GPU pass, from the renderer's profiler.
    pub profiling: Vec<String>,
    /// Free-form line the app writes for the status bar (device name, grid
    /// size, VRAM).
    pub status: String,

    actions: Vec<UiAction>,
}

impl Default for UiState {
    fn default() -> Self {
        Self::new(SimParams::default())
    }
}

impl UiState {
    pub fn new(params: SimParams) -> Self {
        Self {
            playing: true,
            tuner: StepAutoTuner::default(),
            clock: Clock::default(),
            params,
            watcher: ParamWatcher::new(params),
            layers: LayerStack::defaults(),
            probes: Probes::new(),
            slices: Vec::new(),
            particles: ParticleSettings::default(),
            iso: IsoSettings::default(),
            streamlines: StreamlineSettings::default(),
            gizmo_mode: GizmoMode::Translate,
            gizmo_space: GizmoSpace::World,
            gizmo_snap_mm: Some(0.5),
            gizmo_snap_deg: Some(15.0),
            install: InstallPose::IDENTITY,
            install_pivot_mm: Vec3::ZERO,
            inlet_louver_deg: [0.0, 0.0],
            obstructions: Vec::new(),
            duct: DuctGeometry::IDENTITY,
            vents: Vec::new(),
            domain_margins_mm: [0.0; 6],
            pending_domain_mm: [0.0; 6],
            legend: LegendRange::default(),
            units: UnitSystem::Metric,
            camera_input: CameraInput::default(),
            field: DerivedField::Speed,
            metrics: MetricsView::default(),
            mouths: Vec::new(),
            resolution: ResolutionPanel::new(params.dx_mm),
            baseline: None,
            deltas: None,
            toasts: Toasts::new(),
            panels: PanelVisibility::default(),
            plot_tab: PlotTab::Flow,
            viewport_rect: [0.0, 0.0, 1.0, 1.0],
            ui_capture_mouse: false,
            gizmo_active: false,
            ui_scale: 1.0,
            profiling: Vec::new(),
            status: String::new(),
            actions: Vec::new(),
        }
    }

    /// Queue an action for the app to service.
    pub fn push_action(&mut self, action: UiAction) {
        self.actions.push(action);
    }

    /// Take the queued actions. The app calls this once per frame.
    pub fn drain_actions(&mut self) -> Vec<UiAction> {
        std::mem::take(&mut self.actions)
    }

    pub fn pending_actions(&self) -> &[UiAction] {
        &self.actions
    }

    /// Detect parameter changes and raise the reset toast.
    ///
    /// This is where the "statistics reset must be visible" requirement is
    /// actually enforced, and it is one call rather than a rule people have to
    /// remember: any path that edits [`Self::params`] gets the toast for free,
    /// including one that has not been written yet.
    ///
    /// Returns the classification so the app can rebuild or hot-apply.
    pub fn sync_params(&mut self) -> crate::params::ParamChange {
        let change = self.watcher.observe(self.params);
        if let Some(cause) = change.cause {
            if change.reset_statistics {
                self.toasts
                    .push(crate::toast::Toast::statistics_reset(cause));
                // A probe trace spanning a boundary-condition change shows a
                // step the flow never took, so it goes with the averages.
                self.probes.clear_history();
            }
        }
        change
    }

    /// What [`Self::sync_params`] would report now, without committing it,
    /// raising a toast or clearing a probe.
    ///
    /// The frame loop asks this before a rebuild, because a rebuild can fail —
    /// the GPU can refuse the new lattice — and a failed one has to be undone
    /// with [`Self::revert_params`] before anything announces it.
    pub fn peek_params(&self) -> crate::params::ParamChange {
        self.watcher.classify(&self.params)
    }

    /// The parameters the running solver actually has: the last set
    /// [`Self::sync_params`] committed.
    pub fn committed_params(&self) -> SimParams {
        *self.watcher.current()
    }

    /// Discard edits to [`Self::params`] that have not been committed.
    ///
    /// For the one case where the app cannot honour an edit: a rebuild that
    /// failed. The controls then show what the solver is really running rather
    /// than a resolution nobody is simulating, and the next
    /// [`Self::sync_params`] finds nothing to announce.
    pub fn revert_params(&mut self) {
        self.params = self.committed_params();
    }

    /// Force the next [`Self::sync_params`] to report a reset. Used by the
    /// explicit "reset statistics" button so the toast still appears.
    pub fn invalidate_statistics(&mut self) {
        self.watcher.invalidate();
    }

    /// Recompute the A/B deltas. Cheap, and doing it every frame keeps them in
    /// step with a live-updating [`Self::metrics`].
    pub fn refresh_deltas(&mut self) {
        self.deltas = self
            .baseline
            .as_ref()
            .map(|b| crate::compare::compare(&self.metrics, &b.metrics));
    }

    /// Freeze the current metrics as the baseline.
    pub fn save_baseline(&mut self, label: impl Into<String>, step: u64) {
        let b = Baseline::capture(label, step, self.params.inlet_velocity_ms, &self.metrics);
        let l = b.label.clone();
        self.baseline = Some(b);
        self.refresh_deltas();
        self.toasts.push(
            crate::toast::Toast::new(crate::toast::Severity::Info, "Baseline saved")
                .with_detail(format!("\"{l}\" at step {step}; the HUD now shows deltas"))
                .with_key("baseline"),
        );
    }

    pub fn clear_baseline(&mut self) {
        self.baseline = None;
        self.deltas = None;
    }

    /// Whether the baseline is at the same operating point as the live run.
    /// `None` when there is no baseline.
    pub fn baseline_comparable(&self) -> Option<bool> {
        self.baseline
            .as_ref()
            .map(|b| b.same_operating_point(self.params.inlet_velocity_ms))
    }

    /// Which mouth is the outlet, given the chosen inlet. With exactly two
    /// mouths this is the other one; with more, the largest of the rest, on the
    /// grounds that a duct's main outlet is not a drain hole.
    pub fn outlet_mouth(&self) -> Option<usize> {
        if self.mouths.len() < 2 {
            return None;
        }
        let inlet = self.params.inlet_mouth;
        // What the app built, when it has said: it knows which mouths the
        // vents are sealed to, which this view model does not.
        if let Some(o) = self.mouths.iter().position(|m| m.is_outlet) {
            if o != inlet {
                return Some(o);
            }
        }
        if let Some(o) = self
            .params
            .outlet_mouth
            .filter(|o| *o < self.mouths.len() && *o != inlet)
        {
            return Some(o);
        }
        if self.mouths.len() == 2 {
            return Some(1 - inlet.min(1));
        }
        self.mouths
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != inlet)
            .max_by(|a, b| a.1.open_area_mm2.total_cmp(&b.1.open_area_mm2))
            .map(|(i, _)| i)
    }

    /// Queue the one-click inlet/outlet swap.
    pub fn swap_inlet_outlet(&mut self) {
        if let Some(out) = self.outlet_mouth() {
            self.params.inlet_mouth = out;
            self.push_action(UiAction::SetInletMouth(out));
        }
    }

    /// Convert a point in the viewport rectangle to normalised device
    /// coordinates, `y` up. Returns `None` outside the rectangle.
    ///
    /// Used by click-to-probe. The `y` flip is the classic off-by-a-sign here:
    /// window coordinates grow downward and NDC grows upward, and getting it
    /// wrong drops probes mirrored about the horizon, which looks like a
    /// picking bug in the *depth* buffer and sends you debugging the wrong
    /// thing.
    pub fn viewport_ndc(&self, screen: [f32; 2]) -> Option<glam::Vec2> {
        let [x, y, w, h] = self.viewport_rect;
        if w <= 0.0 || h <= 0.0 {
            return None;
        }
        let u = (screen[0] - x) / w;
        let v = (screen[1] - y) / h;
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        Some(glam::Vec2::new(u * 2.0 - 1.0, 1.0 - v * 2.0))
    }

    /// Pointer ownership for the camera controller.
    pub fn pointer_ownership(&self) -> crate::camera_input::PointerOwnership {
        crate::camera_input::PointerOwnership {
            ui_capture: self.ui_capture_mouse,
            gizmo_active: self.gizmo_active,
        }
    }

    /// The layer currently armed for the gizmo, if it is one that can be moved.
    pub fn gizmo_target(&self) -> Option<LayerId> {
        let l = self.layers.selected()?;
        (l.kind.is_transformable() && self.gizmo_mode != GizmoMode::Off).then_some(l.id)
    }

    /// The viewport rectangle in ImGui's logical pixels.
    pub fn viewport_rect_logical(&self) -> [f32; 4] {
        let s = if self.ui_scale.is_finite() && self.ui_scale > 0.0 {
            self.ui_scale
        } else {
            1.0
        };
        self.viewport_rect.map(|v| v / s)
    }

    /// Lattice → world for the current install pose.
    pub fn install_matrix(&self) -> glam::Mat4 {
        self.install.matrix(self.install_pivot_mm)
    }

    /// A lattice-frame point (probe, slice, mouth) where the user sees it.
    pub fn to_world(&self, p: Vec3) -> Vec3 {
        self.install.to_world(self.install_pivot_mm, p)
    }

    /// A world point (a camera ray, a gizmo drag) in the lattice frame.
    pub fn to_lattice(&self, p: Vec3) -> Vec3 {
        self.install.to_lattice(self.install_pivot_mm, p)
    }

    /// Set the part-frame inlet tilt the solver takes from the louver aim the
    /// user gave in car terms.
    ///
    /// Cheap and idempotent, so the app calls it every frame: the louver is
    /// fixed in the car, so re-posing the part or swapping the inlet mouth
    /// changes the tilt relative to the part, and that change then goes
    /// through [`Self::sync_params`] like any other — toast and all.
    pub fn sync_inlet_tilt(&mut self) {
        let Some(m) = self.mouths.get(self.params.inlet_mouth) else {
            return;
        };
        let axis = crate::pose::lattice_axis(m.normal);
        self.params.inlet_tilt_deg = crate::pose::louver_to_tilt(
            self.inlet_louver_deg,
            m.normal,
            axis,
            self.install.rotation,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{Health, Reading};

    #[test]
    fn actions_queue_and_drain_once() {
        let mut s = UiState::default();
        s.push_action(UiAction::ResetSolver);
        s.push_action(UiAction::SetPlaying(false));
        assert_eq!(s.pending_actions().len(), 2);
        let drained = s.drain_actions();
        assert_eq!(drained.len(), 2);
        assert!(
            s.drain_actions().is_empty(),
            "actions must not be delivered twice"
        );
    }

    #[test]
    fn changing_a_parameter_raises_exactly_one_visible_reset() {
        // The requirement, end to end: edit a parameter, get a toast.
        let mut s = UiState::default();
        assert!(s.sync_params().is_none());
        assert!(s.toasts.is_empty());

        s.params.inlet_velocity_ms = 5.0;
        let change = s.sync_params();
        assert!(change.reset_statistics && change.hot_apply && !change.rebuild);
        assert_eq!(s.toasts.len(), 1);
        assert!(
            s.toasts
                .iter()
                .next()
                .unwrap()
                .detail
                .contains("inlet velocity"),
            "the toast must name the cause"
        );
        // A quiet frame afterwards must not raise another.
        s.sync_params();
        assert_eq!(s.toasts.len(), 1);
    }

    #[test]
    fn a_change_that_was_reverted_is_never_announced() {
        // The failed-rebuild path: the app peeks, tries the rebuild, the GPU
        // refuses, and the edit is reverted. Nothing in that sequence may leave
        // a "statistics reset" toast behind, because nothing was reset.
        let mut s = UiState::default();
        let running = s.committed_params();
        s.params.dx_mm = 0.3;

        let preview = s.peek_params();
        assert!(preview.rebuild && preview.reset_statistics);
        assert!(s.toasts.is_empty(), "peeking must not toast");
        assert_eq!(s.committed_params(), running, "peeking must not commit");

        s.revert_params();
        assert_eq!(s.params, running);
        assert!(s.sync_params().is_none());
        assert!(
            s.toasts.is_empty(),
            "a change that never took effect must not be announced"
        );
    }

    #[test]
    fn a_statistics_reset_also_drops_probe_history() {
        // A probe trace spanning a boundary change shows a step the flow never
        // took, which reads as a physical event.
        let mut s = UiState::default();
        let id = s.probes.add(Vec3::ZERO);
        for i in 0..10 {
            s.probes.record(id, i as f64, 1.0, Vec3::X, true);
        }
        assert!(!s.probes.get(id).unwrap().pressure.is_empty());
        s.params.inlet_velocity_ms = 4.0;
        s.sync_params();
        assert!(s.probes.get(id).unwrap().pressure.is_empty());
    }

    #[test]
    fn an_explicit_reset_still_produces_a_toast() {
        let mut s = UiState::default();
        s.invalidate_statistics();
        let c = s.sync_params();
        assert!(c.reset_statistics);
    }

    #[test]
    fn swapping_the_inlet_is_one_action_and_is_reversible() {
        let mut s = UiState::default();
        s.mouths = vec![PatchView::default(), PatchView::default()];
        assert_eq!(s.outlet_mouth(), Some(1));
        s.swap_inlet_outlet();
        assert_eq!(s.params.inlet_mouth, 1);
        assert_eq!(s.drain_actions(), vec![UiAction::SetInletMouth(1)]);
        s.swap_inlet_outlet();
        assert_eq!(s.params.inlet_mouth, 0);
    }

    #[test]
    fn with_more_than_two_mouths_the_outlet_is_the_largest_of_the_rest() {
        let mut s = UiState::default();
        let mouth = |a: f32| PatchView {
            open_area_mm2: a,
            ..Default::default()
        };
        s.mouths = vec![mouth(2116.0), mouth(30.0), mouth(1141.0)];
        assert_eq!(s.outlet_mouth(), Some(2), "a drain hole is not the outlet");
        s.params.inlet_mouth = 2;
        assert_eq!(s.outlet_mouth(), Some(0));
    }

    #[test]
    fn a_single_mouth_has_no_outlet_and_swapping_does_nothing() {
        let mut s = UiState::default();
        s.mouths = vec![PatchView::default()];
        assert_eq!(s.outlet_mouth(), None);
        s.swap_inlet_outlet();
        assert!(s.pending_actions().is_empty());
    }

    #[test]
    fn viewport_picking_flips_y_and_rejects_clicks_outside() {
        let mut s = UiState::default();
        s.viewport_rect = [100.0, 50.0, 800.0, 400.0];
        let c = s.viewport_ndc([500.0, 250.0]).unwrap();
        assert!(
            c.x.abs() < 1e-6 && c.y.abs() < 1e-6,
            "centre should be NDC origin: {c:?}"
        );
        // Top of the viewport is NDC +1.
        assert!(s.viewport_ndc([500.0, 50.0]).unwrap().y > 0.99);
        assert!(s.viewport_ndc([500.0, 450.0]).unwrap().y < -0.99);
        assert!(
            s.viewport_ndc([50.0, 250.0]).is_none(),
            "a click on the layer panel is not a pick"
        );
        assert!(s.viewport_ndc([500.0, 500.0]).is_none());
    }

    #[test]
    fn a_degenerate_viewport_does_not_divide_by_zero() {
        let mut s = UiState::default();
        s.viewport_rect = [0.0, 0.0, 0.0, 0.0];
        assert!(s.viewport_ndc([0.0, 0.0]).is_none());
    }

    #[test]
    fn the_baseline_drives_the_deltas_and_reports_comparability() {
        let mut s = UiState::default();
        s.metrics.pressure_drop = Reading::new(47.3, 0.6, Health::Good, 200);
        s.save_baseline("as loaded", 12_000);
        assert_eq!(s.baseline_comparable(), Some(true));
        assert!(s.deltas.is_some());
        assert!(!s.toasts.is_empty(), "saving a baseline should say so");

        s.metrics.pressure_drop = Reading::new(41.0, 0.6, Health::Good, 200);
        s.refresh_deltas();
        assert_eq!(s.deltas.unwrap().pressure_drop.improved, Some(true));

        // Move the operating point: the deltas still compute, but the panel is
        // told they are not a like-for-like comparison.
        s.params.inlet_velocity_ms = 8.0;
        assert_eq!(s.baseline_comparable(), Some(false));

        s.clear_baseline();
        assert!(s.deltas.is_none());
        assert_eq!(s.baseline_comparable(), None);
    }

    #[test]
    fn the_gizmo_only_arms_on_a_placeable_layer() {
        use crate::layers::LayerKind;
        let mut s = UiState::default();
        let volume = s.layers.find(LayerKind::Volume).unwrap();
        s.layers.select(Some(volume));
        assert!(
            s.gizmo_target().is_none(),
            "the volume is not a placeable object"
        );

        let obstruction = s.layers.push(LayerKind::Obstruction(0), "Vane");
        s.layers.select(Some(obstruction));
        assert_eq!(s.gizmo_target(), Some(obstruction));

        s.gizmo_mode = GizmoMode::Off;
        assert!(
            s.gizmo_target().is_none(),
            "'off' must actually suppress the gizmo"
        );
    }

    #[test]
    fn pointer_ownership_reaches_the_camera_mapping() {
        let mut s = UiState::default();
        assert!(s.pointer_ownership().viewport_has_mouse());
        s.ui_capture_mouse = true;
        assert!(!s.pointer_ownership().viewport_has_mouse());
        s.ui_capture_mouse = false;
        s.gizmo_active = true;
        assert!(!s.pointer_ownership().viewport_has_mouse());
    }
}

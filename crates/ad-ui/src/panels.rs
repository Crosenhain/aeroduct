//! The widgets.
//!
//! Everything here renders from [`crate::view`], [`crate::state`] and the
//! renderer's own settings structs. Nothing in this file references `ad-metrics`
//! or a flow-visualisation pass, which is what lets those crates be written in
//! parallel with this one: they fill [`crate::view::MetricsView`], and the
//! panels never learn where a number came from.
//!
//! # Layout
//!
//! One dockspace over the main viewport, with the central node left **empty and
//! passthrough**. The 3D scene is rendered to the whole surface before ImGui
//! draws, so the empty central node is a hole through the UI onto the
//! simulation — which is also why [`crate::state::UiState::viewport_rect`] is
//! the whole framebuffer rather than a sub-rectangle: picking rays and the
//! gizmo are computed against the same full-screen projection the renderer used.
//!
//! ```text
//! +-------------------------------------------------------------+
//! | TOOLBAR  transport | steps/frame | view | inlet A<->B | U     |
//! | HUD      Q  dp  K  gamma  theta  U_max   (each +/- and lit)   |
//! | STATUS   sim | wall | x real-time | steps/s      [warnings]   |
//! +--------+--------------------------------------+-------------+
//! | LAYERS |            (the 3D view)             | PROPERTIES  |
//! |        |                                      +-------------+
//! |        |                                      | LEGEND      |
//! +--------+--------------------------------------+-------------+
//! |  PLOTS  [Q_in/Q_out] [dp] [probe p(t)] [RTD] [hist] [spec]   |
//! +-------------------------------------------------------------+
//! ```
//!
//! # Two rules this file follows without being asked
//!
//! * **No modal dialogs.** Every destructive action is a button that takes
//!   effect immediately and announces itself with a toast. The only blocking
//!   dialog in the app is the OS file picker, which is the OS's.
//! * **No bare numbers.** Every measured scalar goes through
//!   [`crate::format::uncertain`], which is the only renderer that knows how
//!   many digits an error bar justifies.

use ad_render::{
    Camera, ColorMap, DerivedField, MeshDisplay, OpacityMode, RangeScale, ViewPreset,
};
use dear_imgui_rs::{
    Condition, DockLayout, DockLayoutApply, DockNodeFlags, DockSplit, MouseButton, StyleColor,
    TreeNodeFlags, WindowFlags, WindowKey,
};
use dear_implot::{
    AxisFlags, DragToolFlags, DragToolId, LinePlot, Plot as _, PlotCond, PositionalBarPlot,
    ScatterPlot,
};

use crate::backend::UiFrame;
use crate::compare::Delta;
use crate::format::{self, Quantity, UnitSystem};
use crate::layers::{LayerId, LayerKind};
use crate::legend::Handle;
use crate::overlays::{snap_for, GizmoMode, GizmoSpace, ParticleSeed, Placement, VentSettings};
use crate::pose::{describe_direction, DuctGeometry, InstallPose};
use crate::state::{PlotTab, UiAction, UiState};
use crate::view::{
    Health, MetricsView, Reading, ResetCause, ResolutionEstimate, ResolutionPanel, Trace,
};

/// Everything the app hands the panels that they are allowed to edit in place.
///
/// The rule from [`crate::state`]: state the renderer already owns is borrowed,
/// not copied. A second copy of the transfer function living in `UiState` would
/// be wrong roughly half the time — whichever one the user is not dragging.
pub struct PanelContext<'a> {
    pub state: &'a mut UiState,
    /// Transfer function for the field currently displayed.
    pub transfer: &'a mut ad_render::TransferFunction,
    /// The live camera, for the gizmo's view and projection matrices.
    pub camera: &'a Camera,
    pub mesh_display: &'a mut MeshDisplay,
    /// Solver steps taken, for baseline provenance and the status bar.
    pub step: u64,
    /// The lattice the solver is running on.
    pub grid: ad_gpu::Grid,
    /// Measured extremes of the displayed field, when the app knows them. Drives
    /// the legend's double-click auto-range.
    pub field_extremes: Option<(f32, f32)>,
}

/// Stable identities for the docked windows, plus scratch the widgets need to
/// keep between frames.
pub struct Panels {
    keys: Keys,
    layout: DockLayout,
    scratch: Scratch,
}

struct Keys {
    toolbar: WindowKey,
    layers: WindowKey,
    properties: WindowKey,
    legend: WindowKey,
    plots: WindowKey,
    diagnostics: WindowKey,
}

#[derive(Default)]
struct Scratch {
    baseline_label: String,
    /// Manual steps-per-frame, when the auto-tuner is off.
    manual_steps: i32,
    /// Curve point being dragged in the opacity editor, so a drag that leaves
    /// the point behind does not grab its neighbour.
    dragging_curve_point: Option<usize>,
    /// Last view preset applied, so the combo shows what was chosen rather than
    /// snapping back to the first entry on the next frame.
    view_preset: usize,
}

impl Panels {
    pub fn new() -> anyhow::Result<Self> {
        let key = |id: &str, title: &str| -> anyhow::Result<WindowKey> {
            WindowKey::new(id, title).map_err(|e| anyhow::anyhow!("window key {id}: {e}"))
        };
        let keys = Keys {
            toolbar: key("toolbar", "Simulation")?,
            layers: key("layers", "Layers")?,
            properties: key("properties", "Properties")?,
            legend: key("legend", "Legend")?,
            plots: key("plots", "Plots")?,
            diagnostics: key("diagnostics", "Diagnostics")?,
        };

        // Toolbar across the top, layers left, properties over the legend on the
        // right, plots along the bottom, and the remaining leaf deliberately
        // empty: that empty leaf is the central node, and with
        // `PASSTHRU_CENTRAL_NODE` it is the window onto the simulation.
        let layout = DockLayout::split(
            DockSplit::Up,
            0.20,
            DockLayout::tabs([&keys.toolbar]),
            DockLayout::split(
                DockSplit::Left,
                0.17,
                DockLayout::tabs([&keys.layers]),
                DockLayout::split(
                    DockSplit::Right,
                    0.26,
                    DockLayout::split(
                        DockSplit::Down,
                        0.42,
                        DockLayout::tabs([&keys.legend]),
                        DockLayout::tabs([&keys.properties, &keys.diagnostics]),
                    ),
                    DockLayout::split(
                        DockSplit::Down,
                        0.32,
                        DockLayout::tabs([&keys.plots]),
                        DockLayout::Tabs(Vec::new()),
                    ),
                ),
            ),
        );
        layout
            .validate()
            .map_err(|e| anyhow::anyhow!("dock layout: {e}"))?;

        Ok(Self {
            keys,
            layout,
            scratch: Scratch { manual_steps: 16, ..Default::default() },
        })
    }

    /// Draw the whole UI for one frame.
    pub fn build(&mut self, frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
        let ui = frame.ui;
        // Disjoint field borrows: the keys and the layout are read, the scratch
        // is written, and the closures below capture each independently.
        let keys = &self.keys;
        let layout = &self.layout;
        let scratch = &mut self.scratch;

        let docked = ui
            .dockspace()
            .main_viewport()
            .flags(DockNodeFlags::PASSTHRU_CENTRAL_NODE)
            .layout(layout, DockLayoutApply::IfMissing)
            .build()
            .is_ok();

        // The panels are drawn whether or not docking took: without it they are
        // floating windows in the same order, which is a degraded layout rather
        // than a missing UI.
        if cx.state.panels.hud {
            ui.window(&keys.toolbar)
                .flags(WindowFlags::NO_SCROLLBAR)
                .size([1200.0, 150.0], Condition::FirstUseEver)
                .build(|| {
                    toolbar(frame, cx, scratch);
                    ui.separator();
                    hud(frame, cx);
                    ui.separator();
                    status_bar(frame, cx);
                });
        }

        if cx.state.panels.layers {
            ui.window(&keys.layers)
                .size([260.0, 600.0], Condition::FirstUseEver)
                .build(|| layers_panel(frame, cx));
        }

        if cx.state.panels.properties {
            ui.window(&keys.properties)
                .size([320.0, 460.0], Condition::FirstUseEver)
                .build(|| properties_panel(frame, cx, scratch));
        }

        if cx.state.panels.legend {
            ui.window(&keys.legend)
                .size([320.0, 300.0], Condition::FirstUseEver)
                .build(|| legend_panel(frame, cx));
        }

        if cx.state.panels.plots {
            ui.window(&keys.plots)
                .size([1200.0, 280.0], Condition::FirstUseEver)
                .build(|| plots_panel(frame, cx));
        }

        if cx.state.panels.diagnostics {
            ui.window(&keys.diagnostics)
                .size([320.0, 460.0], Condition::FirstUseEver)
                .build(|| diagnostics_panel(frame, cx, docked));
        }

        gizmo(frame, cx);
        inlet_arrow(frame, cx);
        vent_outlines(frame, cx);
        toasts(frame, cx);
    }
}

// ---------------------------------------------------------------------------
// toolbar
// ---------------------------------------------------------------------------

fn toolbar(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>, scratch: &mut Scratch) {
    let ui = frame.ui;
    let s = &mut *cx.state;

    // Transport. Note what is *not* here: a Run button. The sim is always
    // running; pause exists to freeze a transient for inspection.
    let play_label = if s.playing { "Pause" } else { "Play" };
    if ui.button_with_size(play_label, [72.0, 0.0]) {
        s.playing = !s.playing;
        let playing = s.playing;
        s.push_action(UiAction::SetPlaying(playing));
    }
    ui.set_item_tooltip(
        "Freeze the solver to inspect a transient. Editing anything still applies live.",
    );

    ui.same_line();
    let steps = s.tuner.steps();
    if ui.button("Step") {
        s.push_action(UiAction::StepOnce(steps.max(1)));
    }
    ui.set_item_tooltip("Advance one dispatch of the current steps-per-frame.");

    ui.same_line();
    if ui.button("Reset field") {
        s.push_action(UiAction::ResetSolver);
        s.invalidate_statistics();
    }
    ui.set_item_tooltip("Reinitialise every cell to equilibrium and zero the step counter.");

    ui.same_line();
    if ui.button("Reset stats") {
        s.invalidate_statistics();
        s.push_action(UiAction::ResetStatistics(ResetCause::Manual));
    }
    ui.set_item_tooltip("Throw away the averaging window and start the error bars again.");

    // Steps per frame.
    ui.same_line();
    ui.text("|");
    ui.same_line();
    let mut auto = s.tuner.enabled;
    if ui.checkbox("auto steps", &mut auto) {
        if auto {
            s.tuner.enabled = true;
        } else {
            s.tuner.set_manual(steps);
            scratch.manual_steps = steps as i32;
        }
    }
    ui.set_item_tooltip(
        "Pick the steps per frame to hold the frame-time budget. Turn it off to pin the count \
         while reproducing something.",
    );
    ui.same_line();
    ui.set_next_item_width(120.0);
    if s.tuner.enabled {
        let mut budget_ms = s.tuner.config.frame_budget_s * 1000.0;
        if ui.slider_f32("ms/frame", &mut budget_ms, 8.0, 100.0) {
            s.tuner.config.frame_budget_s = budget_ms / 1000.0;
        }
        ui.same_line();
        ui.text_disabled(format!("{steps} steps/frame"));
    } else {
        let mut n = scratch.manual_steps.max(1);
        if ui.slider_i32("steps/frame", &mut n, 1, 2048) {
            scratch.manual_steps = n;
            s.tuner.set_manual(n as u32);
        }
    }

    // The cell size, on the first row beside the other controls of what the
    // solver costs. Not on a row of its own: the toolbar dock is a fixed slice
    // of the window with no scrollbar, and a third row pushes the status bar
    // out of it.
    ui.same_line();
    ui.text("|");
    ui.same_line();
    resolution_control(frame, s);

    // Second row. Deliberately a new line rather than more `same_line`: at
    // 1600 px the transport controls, the tuner and the operating point do not
    // fit on one row, and ImGui clips rather than wraps — the inlet slider
    // simply vanishes off the right edge.
    ui.set_next_item_width(130.0);
    if ui.combo("view", &mut scratch.view_preset, &ViewPreset::ALL, |p| {
        std::borrow::Cow::Borrowed(p.label())
    }) {
        let preset = ViewPreset::ALL[scratch.view_preset.min(ViewPreset::ALL.len() - 1)];
        s.push_action(UiAction::ApplyViewPreset(preset));
    }
    ui.same_line();
    if ui.button("Frame") {
        s.push_action(UiAction::FrameScene);
    }
    ui.set_item_tooltip("Fit the camera to the scene (F).");

    ui.same_line();
    ui.text("|");
    ui.same_line();
    inlet_toggle(frame, s);

    ui.same_line();
    ui.set_next_item_width(220.0);
    let mut u = s.params.inlet_velocity_ms;
    // 0-8 m/s: the contract's vent range, and the slider the user spends the
    // session on. It hot-applies; the statistics reset it causes is announced by
    // `UiState::sync_params`, not here, so no path can forget it.
    if ui.slider_f32("inlet U (m/s)", &mut u, 0.0, 8.0) {
        s.params.inlet_velocity_ms = u;
        s.push_action(UiAction::SetInletVelocity(u));
    }

    ui.same_line();
    let mut imperial = s.units == UnitSystem::Imperial;
    if ui.checkbox("imperial", &mut imperial) {
        s.units = if imperial { UnitSystem::Imperial } else { UnitSystem::Metric };
    }
    ui.set_item_tooltip("CFM and inches of water instead of L/s and pascals. Display only.");

    ui.same_line();
    if ui.button("Baseline") {
        let label = if scratch.baseline_label.trim().is_empty() {
            format!("step {}", format::grouped(cx.step))
        } else {
            scratch.baseline_label.clone()
        };
        let step = cx.step;
        cx.state.save_baseline(label, step);
        cx.state.push_action(UiAction::SaveBaseline);
    }
    ui.set_item_tooltip("Freeze the current numbers; the HUD then shows deltas against them.");
    if cx.state.baseline.is_some() {
        ui.same_line();
        if ui.button("Clear A/B") {
            cx.state.clear_baseline();
            cx.state.push_action(UiAction::ClearBaseline);
        }
    }
}

/// The cell-size control: preset tiers, a custom value, Apply, and the cost.
///
/// Apply only *asks*. It pushes [`UiAction::SetResolution`], and the app
/// pre-flights the size against the GPU's memory and binding limits before it
/// writes `SimParams::dx_mm` — so a refused size never touches the parameters,
/// and never raises a statistics-reset toast for a change that did not happen.
fn resolution_control(frame: &UiFrame<'_>, s: &mut UiState) {
    let ui = frame.ui;
    let running = s.params.dx_mm;
    let same = |a: f32, b: f32| a.to_bits() == b.to_bits();

    ui.text("dx");
    ui.set_item_tooltip(
        "Cell size: the lattice spacing. Finer resolves the passage better and costs about \
         dx^-4 in wall-clock time per flow-through: halving it means 8x the cells and twice \
         the steps per second of air. Nothing changes until Apply.",
    );

    ui.same_line();
    ui.set_next_item_width(95.0);
    let pending = s.resolution.pending_dx_mm;
    let preview = if ResolutionPanel::PRESETS_MM.iter().any(|p| same(*p, pending)) {
        format!("{pending} mm")
    } else {
        "custom".to_string()
    };
    if let Some(combo) = ui.begin_combo("##dx preset", &preview) {
        for preset in ResolutionPanel::PRESETS_MM {
            let label = if same(preset, running) {
                format!("{preset} mm (running)")
            } else {
                format!("{preset} mm")
            };
            if ui.selectable_config(&label).selected(same(preset, pending)).build() {
                s.resolution.pending_dx_mm = preset;
            }
        }
        combo.end();
    }

    ui.same_line();
    ui.set_next_item_width(70.0);
    let mut custom = s.resolution.pending_dx_mm;
    if ui.input_float("mm##dx", &mut custom) && custom.is_finite() {
        s.resolution.pending_dx_mm = custom.clamp(ResolutionPanel::MIN_MM, ResolutionPanel::MAX_MM);
    }
    ui.set_item_tooltip(format!(
        "Any cell size from {} to {} mm.",
        ResolutionPanel::MIN_MM,
        ResolutionPanel::MAX_MM
    ));

    let pending = s.resolution.pending_dx_mm;
    let unchanged = same(pending, running);
    let estimate = s.resolution.current_estimate().cloned();
    // Disabled until the app has costed the value in the box, so a click can
    // never race ahead of the check it is supposed to wait for.
    let blocked = estimate.as_ref().is_none_or(|e| e.blocker.is_some());
    ui.same_line();
    {
        let _off = ui.begin_disabled_with_cond(unchanged || blocked);
        if ui.button("Apply##dx") {
            s.push_action(UiAction::SetResolution(pending));
        }
    }
    ui.set_item_tooltip(if unchanged {
        "This is the cell size the solver is running.".to_string()
    } else {
        format!(
            "Rebuild at {pending} mm: re-voxelises the part and restarts the statistics. If the \
             new lattice cannot be built, the current one keeps running."
        )
    });

    ui.same_line();
    match &estimate {
        None => ui.text_disabled("..."),
        Some(e) if !unchanged && e.blocker.is_some() => {
            ui.text_colored(Health::Bad.color(), "won't fit");
            if let Some(why) = &e.blocker {
                ui.set_item_tooltip(why);
            }
        }
        Some(e) => {
            ui.text_disabled(resolution_summary(e));
            ui.set_item_tooltip(resolution_detail(e, unchanged));
            if !unchanged && !e.warnings.is_empty() {
                ui.same_line();
                ui.text_colored(Health::Watch.color(), format!("[{}]", e.warnings.len()));
                ui.set_item_tooltip(e.warnings.join("\n"));
            }
        }
    }
}

/// The cost line beside Apply: size, memory, time.
fn resolution_summary(e: &ResolutionEstimate) -> String {
    let memory = match e.budget_bytes {
        Some(b) => format!("{} of {}", gib(e.vram_bytes), gib(b)),
        None => gib(e.vram_bytes),
    };
    let time = match e.flow_through_s {
        Some(t) => format!("{} per flow-through", format::duration(t)),
        None => "speed unknown".to_string(),
    };
    format!("{:.1} M cells | {memory} | {time}", e.cells as f64 / 1e6)
}

/// The same in full, for the tooltip.
fn resolution_detail(e: &ResolutionEstimate, running: bool) -> String {
    let mut s = format!(
        "{}{} x {} x {} = {} cells at {} mm\nGPU memory: {} for the lattice",
        if running { "running: " } else { "" },
        e.dims[0],
        e.dims[1],
        e.dims[2],
        format::grouped(e.cells),
        e.dx_mm,
        gib(e.vram_bytes),
    );
    if let Some(b) = e.budget_bytes {
        s += &format!(", of {} the app allows itself", gib(b));
    }
    match e.steps_per_s {
        Some(r) => s += &format!("\nabout {} steps/s", format::grouped(r.round() as u64)),
        None => s += "\nspeed unknown: nothing measured, and no bandwidth figure for this GPU",
    }
    if let Some(t) = e.flow_through_s {
        s += &format!(", so {} per flow-through", format::duration(t));
    }
    s += &format!("\ntau = {:.5}", e.tau0);
    s
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// The mouth A <-> B toggle.
fn inlet_toggle(frame: &UiFrame<'_>, s: &mut UiState) {
    let ui = frame.ui;
    if s.mouths.len() < 2 {
        ui.text_disabled("inlet: (one mouth)");
        return;
    }
    let mouth_name = |s: &UiState, i: usize| {
        if s.mouths[i].name.is_empty() {
            format!("{}", (b'A' + i as u8) as char)
        } else {
            s.mouths[i].name.clone()
        }
    };
    ui.text("inlet");
    for i in 0..s.mouths.len().min(4) {
        ui.same_line();
        let name = format!("{}##in", mouth_name(s, i));
        if ui.radio_button(&name, s.params.inlet_mouth == i) {
            s.params.inlet_mouth = i;
            s.push_action(UiAction::SetInletMouth(i));
        }
        let area = s.mouths[i].open_area_mm2;
        let dh = s.mouths[i].hydraulic_diameter_mm;
        let role = match (s.mouths[i].is_inlet, s.mouths[i].is_outlet) {
            (true, _) => "supplying air",
            (_, true) => "the outflow face",
            _ => "an open mouth",
        };
        ui.set_item_tooltip(format!("open area {area:.0} mm^2, D_h {dh:.1} mm; {role}"));
    }
    ui.same_line();
    if ui.button("Swap") {
        s.swap_inlet_outlet();
        s.push_action(UiAction::SwapInletOutlet);
    }
    ui.set_item_tooltip("Blow the other way. Rebuilds the flag field and resets the statistics.");
    // A third mouth means the outlet is a choice too.
    if s.mouths.len() > 2 {
        ui.same_line();
        ui.text("outlet");
        ui.same_line();
        if ui.radio_button("auto##out", s.params.outlet_mouth.is_none()) {
            s.params.outlet_mouth = None;
        }
        ui.set_item_tooltip("The largest mouth that is not supplying air.");
        for i in 0..s.mouths.len().min(4) {
            ui.same_line();
            let name = format!("{}##out", mouth_name(s, i));
            let _off = ui.begin_disabled_with_cond(i == s.params.inlet_mouth);
            if ui.radio_button(&name, s.params.outlet_mouth == Some(i)) {
                s.params.outlet_mouth = Some(i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

/// One metric: name, value with its uncertainty, traffic light, A/B delta.
fn hud(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let units = cx.state.units;
    let m: &MetricsView = &cx.state.metrics;
    let d = cx.state.deltas;

    let chips: [(&str, Reading, Quantity, Option<Delta>); 6] = [
        ("Q", m.flow_rate, Quantity::flow(units), d.map(|x| x.flow_rate)),
        ("dp", m.pressure_drop, Quantity::pressure(units), d.map(|x| x.pressure_drop)),
        ("K", m.loss_coefficient, Quantity::plain(), d.map(|x| x.loss_coefficient)),
        ("gamma", m.uniformity, Quantity::plain(), d.map(|x| x.uniformity)),
        ("theta", m.deflection_deg, Quantity::degrees(), d.map(|x| x.deflection_deg)),
        ("U_max", m.max_speed, Quantity::velocity(units), d.map(|x| x.max_speed)),
    ];

    for (i, (name, reading, quantity, delta)) in chips.into_iter().enumerate() {
        if i > 0 {
            ui.same_line();
            ui.text_disabled("|");
            ui.same_line();
        }
        ui.group(|| {
            ui.text_disabled(name);
            let colour = reading.state.color();
            ui.text_colored(colour, format::uncertain(reading, quantity));
            if let Some(delta) = delta {
                let rel = if delta.relative.is_finite() {
                    format!("{:+.1}%", delta.relative * 100.0)
                } else {
                    format!("{:+}", format::short(delta.absolute))
                };
                ui.same_line();
                ui.text_colored(
                    delta.color(),
                    format!("{} {rel}", delta.direction.glyph()),
                );
            }
        });
        if ui.is_item_hovered() {
            let mut tip = format!(
                "{name}: {}\nstate: {}\nsamples: {}",
                format::uncertain(reading, quantity),
                reading.state.label(),
                reading.samples
            );
            if let Some(r) = reading.relative_error() {
                tip.push_str(&format!("\nrelative error: {}", format::percent(r)));
            }
            if let Some(delta) = delta {
                tip.push_str(&format!(
                    "\nvs baseline: {} ({:.1} sigma, {})",
                    format::short(delta.absolute),
                    delta.sigma(),
                    if delta.is_real() { "significant" } else { "inside the noise" },
                ));
            }
            ui.set_tooltip(tip);
        }
    }

    // Convergence and the averaging window, which is what says whether any of
    // the numbers above are worth quoting.
    ui.same_line();
    ui.text_disabled("|");
    ui.same_line();
    let conv = cx.state.metrics.convergence.state;
    ui.group(|| {
        ui.text_disabled("window");
        ui.text_colored(conv.health().color(), conv.label());
    });
    ui.same_line();
    let stats = &cx.state.metrics.stats;
    ui.group(|| {
        ui.text_disabled("flow-throughs");
        ui.progress_bar(stats.progress())
            .size([170.0, 0.0])
            .overlay_text(format!(
                "{:.1} / {:.0}",
                stats.flow_throughs, stats.flow_throughs_target
            ))
            .build();
    });
    if ui.is_item_hovered() {
        let mut tip = format!(
            "{} steps in this averaging window.\nA duct statistic averaged over less than a \
             couple of flow-throughs has not seen the flow it describes.",
            format::grouped(stats.steps_in_window)
        );
        if let Some(cause) = stats.last_reset {
            tip.push_str(&format!(
                "\n\nlast reset: {} ({} steps ago)",
                cause.message(),
                format::grouped(stats.steps_since_reset)
            ));
        }
        ui.set_tooltip(tip);
    }

    if let Some(false) = cx.state.baseline_comparable() {
        ui.same_line();
        ui.text_colored(
            Health::Watch.color(),
            "baseline is at a different inlet velocity",
        );
        ui.set_item_tooltip(
            "The deltas are still computed, but they are not a like-for-like design comparison.",
        );
    }
}

// ---------------------------------------------------------------------------
// status bar
// ---------------------------------------------------------------------------

fn status_bar(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    ui.text_disabled(cx.state.clock.status_line());
    if ui.is_item_hovered() {
        ui.set_tooltip(crate::autotune::Clock::REALTIME_NOTE);
    }

    let warnings = &cx.state.metrics.warnings;
    if !warnings.is_empty() {
        ui.same_line();
        ui.text_colored(
            Health::Watch.color(),
            format!("[{} warning(s)]", warnings.len()),
        );
        if ui.is_item_hovered() {
            ui.set_tooltip(warnings.join("\n"));
        }
    }

    if !cx.state.status.is_empty() {
        ui.same_line();
        ui.text_disabled("|");
        ui.same_line();
        ui.text_disabled(&cx.state.status);
    }
}

// ---------------------------------------------------------------------------
// layers
// ---------------------------------------------------------------------------

fn layers_panel(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;

    if ui.button("Show all") {
        cx.state.layers.show_all();
    }
    ui.same_line();
    if ui.button("Add slice") {
        cx.state.push_action(UiAction::AddSlice);
    }
    ui.same_line();
    if ui.button("Add vent") {
        cx.state.push_action(UiAction::AddVent);
    }
    ui.set_item_tooltip(
        "A vent blowing air in the room, placed in front of the inlet mouth to start with. \
         With any vent present the duct's mouths are plain openings: what enters is whatever \
         the vents deliver. Up to three.",
    );
    if ui.button("Add obstruction...") {
        if let Some(path) = pick_stl("Add an obstruction") {
            cx.state.push_action(UiAction::LoadObstruction(path));
        }
    }
    ui.set_item_tooltip(
        "An STL placed in the car beside the duct. It starts where its own coordinates put it, \
         so a part exported from the same CAD assembly as the duct lands in place.",
    );
    ui.same_line();
    if ui.button("Load duct...") {
        if let Some(path) = pick_stl("Load a duct") {
            cx.state.push_action(UiAction::LoadDuct(path));
        }
    }
    ui.set_item_tooltip("Replace the duct, keeping the obstructions and the install pose.");
    ui.separator();

    let ids: Vec<LayerId> = cx.state.layers.layers().iter().map(|l| l.id).collect();
    let mut solo: Option<LayerId> = None;
    let mut remove: Option<LayerId> = None;
    let mut select: Option<LayerId> = None;
    let mut move_up: Option<LayerId> = None;
    let mut move_down: Option<LayerId> = None;

    for id in ids {
        let Some(layer) = cx.state.layers.get(id) else { continue };
        let (kind, enabled, reason, name, selected) = (
            layer.kind,
            layer.enabled,
            layer.disabled_reason.clone(),
            layer.name.clone(),
            layer.selected,
        );
        let _id = ui.push_id(id.0 as i32);

        let mut visible = layer.visible;
        // A layer with nothing to draw is greyed rather than hidden: a checkbox
        // that silently does nothing is worse than one you cannot click.
        let disabled_alpha = if enabled { None } else { Some(ui.push_style_var(
            dear_imgui_rs::StyleVar::Alpha(0.45),
        )) };
        if ui.checkbox("##vis", &mut visible) && enabled {
            cx.state.layers.set_visible(id, visible);
        }
        drop(disabled_alpha);
        if !enabled && ui.is_item_hovered() {
            ui.set_tooltip(if reason.is_empty() { "nothing to draw yet" } else { &reason });
        }

        ui.same_line();
        if ui.selectable_config(&name).selected(selected).build() {
            select = Some(id);
        }

        ui.same_line();
        if ui.small_button("S") {
            solo = Some(id);
        }
        ui.set_item_tooltip("Solo: hide everything else. The fastest way to find an artefact.");
        ui.same_line();
        if ui.arrow_button("up", dear_imgui_rs::Direction::Up) {
            move_up = Some(id);
        }
        ui.same_line();
        if ui.arrow_button("down", dear_imgui_rs::Direction::Down) {
            move_down = Some(id);
        }
        if matches!(kind, LayerKind::Slice(_) | LayerKind::Obstruction(_) | LayerKind::Vent(_)) {
            ui.same_line();
            if ui.small_button("x") {
                remove = Some(id);
            }
        }

        // Opacity for the layers that honour it.
        if !matches!(kind, LayerKind::Volume) {
            let mut opacity = cx.state.layers.get(id).map(|l| l.opacity).unwrap_or(1.0);
            ui.set_next_item_width(-1.0);
            if ui.slider_f32("##opacity", &mut opacity, 0.0, 1.0) {
                if let Some(l) = cx.state.layers.get_mut(id) {
                    l.opacity = opacity;
                }
            }
        }
    }

    if let Some(id) = solo {
        cx.state.layers.solo(id);
    }
    if let Some(id) = select {
        cx.state.layers.select(Some(id));
    }
    if let Some(id) = move_up {
        cx.state.layers.move_up(id);
    }
    if let Some(id) = move_down {
        cx.state.layers.move_down(id);
    }
    if let Some(id) = remove {
        if let Some(layer) = cx.state.layers.remove(id) {
            match layer.kind {
                LayerKind::Slice(i) => {
                    if i < cx.state.slices.len() {
                        cx.state.slices.remove(i);
                    }
                    cx.state.push_action(UiAction::RemoveSlice(i));
                }
                LayerKind::Obstruction(i) => {
                    cx.state.push_action(UiAction::RemoveObstruction(i))
                }
                // An edit of the vent list; the app rebuilds once it settles.
                LayerKind::Vent(i) => {
                    if i < cx.state.vents.len() {
                        cx.state.vents.remove(i);
                    }
                }
                _ => {}
            }
        }
    }

    ui.separator();
    probes_list(frame, cx);
}

/// The OS file picker, for an STL: the one modal the UI allows, and it is the
/// OS's. See the crate docs.
fn pick_stl(title: &str) -> Option<std::path::PathBuf> {
    rfd::FileDialog::new().set_title(title).add_filter("STL", &["stl", "STL"]).pick_file()
}

fn probes_list(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    if !ui.collapsing_header("Probes", TreeNodeFlags::DEFAULT_OPEN) {
        return;
    }
    if cx.state.probes.is_empty() {
        ui.text_disabled("ctrl+click the view to drop one");
        return;
    }
    let ids: Vec<u32> = cx.state.probes.items().iter().map(|p| p.id).collect();
    let mut select = None;
    let mut remove = None;
    for id in ids {
        let Some(p) = cx.state.probes.get(id) else { continue };
        let _id = ui.push_id(id as i32);
        // Stored on the part; shown where it is in the car.
        let w = cx.state.to_world(p.position_mm);
        let label = format!("{} ({:.1}, {:.1}, {:.1}) mm", p.label, w.x, w.y, w.z);
        let selected = p.selected;
        let in_fluid = p.in_fluid;
        if ui.selectable_config(&label).selected(selected).build() {
            select = Some(id);
        }
        if !in_fluid {
            ui.same_line();
            ui.text_colored(Health::Bad.color(), "in solid");
            ui.set_item_tooltip("This probe landed inside the wall; every reading here is a lie.");
        }
        ui.same_line();
        if ui.small_button("x") {
            remove = Some(id);
        }
    }
    if let Some(id) = select {
        cx.state.probes.select(Some(id));
    }
    if let Some(id) = remove {
        cx.state.probes.remove(id);
        cx.state.push_action(UiAction::RemoveProbe(id));
    }
}

// ---------------------------------------------------------------------------
// properties
// ---------------------------------------------------------------------------

fn properties_panel(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>, scratch: &mut Scratch) {
    let ui = frame.ui;

    // Which scalar is displayed.
    let mut field_ix = DerivedField::ALL
        .iter()
        .position(|f| *f == cx.state.field)
        .unwrap_or(0);
    ui.set_next_item_width(-1.0);
    if ui.combo("##field", &mut field_ix, &DerivedField::ALL, |f| {
        std::borrow::Cow::Owned(format!("{} ({})", f.name(), f.unit()))
    }) {
        let field = DerivedField::ALL[field_ix];
        cx.state.field = field;
        cx.state.push_action(UiAction::SetField(field));
    }

    ui.separator_with_text("Colour");
    let mut map_ix = ColorMap::ALL
        .iter()
        .position(|m| *m == cx.transfer.map)
        .unwrap_or(0);
    ui.set_next_item_width(-1.0);
    if ui.combo("##colormap", &mut map_ix, &ColorMap::ALL, |m| {
        std::borrow::Cow::Borrowed(m.label())
    }) {
        cx.transfer.map = ColorMap::ALL[map_ix];
    }
    if !cx.transfer.map.cvd_safe() {
        ui.text_colored(
            Health::Watch.color(),
            "not colourblind-safe",
        );
        ui.set_item_tooltip(
            "This map is not distinguishable under deuteranopia. The renderer's accessibility \
             toggle substitutes an equivalent of the same kind.",
        );
    }

    let mut log = cx.transfer.scale == RangeScale::Log;
    if ui.checkbox("log scale", &mut log) {
        cx.transfer.scale = if log { RangeScale::Log } else { RangeScale::Linear };
        cx.transfer.sanitise();
        cx.state.legend.scale = cx.transfer.scale;
        cx.state.legend.sanitise();
    }
    ui.same_line();
    let mut symmetric = cx.transfer.symmetric_lock;
    if ui.checkbox("symmetric", &mut symmetric) {
        cx.transfer.symmetric_lock = symmetric;
        cx.transfer.sanitise();
        cx.state.legend.symmetric = symmetric;
    }
    ui.set_item_tooltip("Keep zero on the diverging map's neutral colour.");

    ui.separator_with_text("Opacity");
    let modes = [OpacityMode::Curve, OpacityMode::SoftIso];
    let mut mode_ix = modes.iter().position(|m| *m == cx.transfer.mode).unwrap_or(0);
    ui.set_next_item_width(-1.0);
    if ui.combo("##opacitymode", &mut mode_ix, &modes, |m| {
        std::borrow::Cow::Borrowed(match m {
            OpacityMode::Curve => "curve",
            OpacityMode::SoftIso => "soft isosurface",
        })
    }) {
        cx.transfer.mode = modes[mode_ix];
        cx.transfer.sanitise();
    }

    let mut density = cx.transfer.density;
    if ui.slider_f32("density", &mut density, 0.0, 8.0) {
        cx.transfer.density = density;
    }

    match cx.transfer.mode {
        OpacityMode::Curve => opacity_curve_editor(frame, cx, scratch),
        OpacityMode::SoftIso => {
            let (lo, hi) = (cx.transfer.range[0], cx.transfer.range[1]);
            let mut iso = cx.transfer.iso;
            let mut changed = ui.slider_f32("level", &mut iso.center, lo, hi);
            changed |= ui.slider_f32("width", &mut iso.width, (hi - lo).abs() * 1e-3, (hi - lo).abs() * 0.5);
            changed |= ui.slider_f32("amplitude", &mut iso.amplitude, 0.0, 1.0);
            if changed {
                cx.transfer.iso = iso;
                cx.transfer.sanitise();
            }
        }
    }

    ui.separator_with_text("Geometry");
    let displays = [
        MeshDisplay::Ghost,
        MeshDisplay::Solid,
        MeshDisplay::Wireframe,
        MeshDisplay::Off,
    ];
    let mut display_ix = displays
        .iter()
        .position(|d| *d == *cx.mesh_display)
        .unwrap_or(0);
    ui.set_next_item_width(-1.0);
    if ui.combo("##meshdisplay", &mut display_ix, &displays, |d| {
        std::borrow::Cow::Borrowed(d.label())
    }) {
        *cx.mesh_display = displays[display_ix];
    }

    duct_geometry_controls(frame, cx);
    install_pose_controls(frame, cx);
    inlet_air_controls(frame, cx);
    domain_controls(frame, cx);

    ui.separator_with_text("Gizmo");
    for (i, mode) in GizmoMode::ALL.into_iter().enumerate() {
        if i > 0 {
            ui.same_line();
        }
        if ui.radio_button(mode.label(), cx.state.gizmo_mode == mode) {
            cx.state.gizmo_mode = mode;
        }
    }
    let mut world = cx.state.gizmo_space == GizmoSpace::World;
    if ui.checkbox("world axes", &mut world) {
        cx.state.gizmo_space = if world { GizmoSpace::World } else { GizmoSpace::Local };
    }
    let mut snap_on = cx.state.gizmo_snap_mm.is_some();
    if ui.checkbox("snap", &mut snap_on) {
        cx.state.gizmo_snap_mm = snap_on.then_some(0.5);
        cx.state.gizmo_snap_deg = snap_on.then_some(15.0);
    }
    if let Some(snap) = cx.state.gizmo_snap_mm.as_mut() {
        ui.same_line();
        ui.set_next_item_width(110.0);
        ui.slider_f32("mm", snap, 0.05, 5.0);
    }
    if let Some(snap) = cx.state.gizmo_snap_deg.as_mut() {
        ui.same_line();
        ui.set_next_item_width(110.0);
        ui.slider_f32("deg", snap, 1.0, 90.0);
    }

    slice_properties(frame, cx);
    obstruction_properties(frame, cx);
    vent_properties(frame, cx);
    particle_properties(frame, cx);
}

/// The selected vent: where it is, how big, how hard and which way it blows.
/// Car frame. Moving or resizing it rebuilds the lattice once the edit
/// settles; its air applies at once.
fn vent_properties(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let Some(layer) = cx.state.layers.selected() else { return };
    let LayerKind::Vent(i) = layer.kind else { return };
    let mouth = cx
        .state
        .mouths
        .get(cx.state.params.inlet_mouth)
        .map(|m| cx.state.to_world(m.center_mm));
    let Some(v) = cx.state.vents.get_mut(i) else { return };
    ui.separator_with_text("Vent");
    ui.text_disabled("car frame; cells rebuild when you stop editing, air applies at once");

    ui.set_next_item_width(90.0);
    let mut w = v.width_mm;
    if ui.input_float("width (mm)", &mut w) && w.is_finite() {
        v.width_mm = w.clamp(1.0, 1000.0);
    }
    ui.same_line();
    ui.set_next_item_width(90.0);
    let mut h = v.height_mm;
    if ui.input_float("height (mm)", &mut h) && h.is_finite() {
        v.height_mm = h.clamp(1.0, 1000.0);
    }
    ui.slider_f32("speed x inlet U", &mut v.speed_scale, 0.0, 3.0);
    ui.set_item_tooltip("This vent's air speed as a multiple of the inlet U on the toolbar.");
    let max = VentSettings::MAX_AIM_DEG;
    ui.slider_f32("aim up/down (deg)", &mut v.aim_deg[0], -max, max);
    ui.set_item_tooltip("Louvers: tip the air toward the vent's own up without turning its face.");
    ui.slider_f32("aim sideways (deg)", &mut v.aim_deg[1], -max, max);

    let mut origin = v.placement.translation_mm.to_array();
    if ui.drag_float3("origin (mm)##vent", &mut origin) {
        v.placement.translation_mm = glam::Vec3::from(origin);
    }
    let (yaw, pitch, roll) = v.placement.rotation.to_euler(glam::EulerRot::YXZ);
    let mut e = [yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees()];
    if ui.drag_float3("yaw/pitch/roll##vent", &mut e) {
        v.placement.rotation = glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            e[0].to_radians(),
            e[1].to_radians(),
            e[2].to_radians(),
        );
    }
    if let Some(target) = mouth {
        if ui.button("Face the inlet mouth") {
            let to = (target - v.placement.translation_mm).normalize_or(v.normal());
            v.placement.rotation =
                glam::Quat::from_rotation_arc(v.normal(), to) * v.placement.rotation;
        }
        ui.set_item_tooltip("Turn this vent to blow straight at the centre of the inlet mouth.");
    }
    ui.text(&format!("blows {}", describe_direction(v.direction())));
}

/// The vents, drawn where they stand: each rectangle's outline and an arrow
/// the way its air leaves. The selected one is brighter.
fn vent_outlines(frame: &UiFrame<'_>, cx: &PanelContext<'_>) {
    let s = &*cx.state;
    if s.vents.is_empty() {
        return;
    }
    let project = viewport_projector(cx.camera, s.viewport_rect_logical());
    let selected = s.layers.selected().map(|l| l.kind);
    let draw = frame.ui.get_background_draw_list();
    for (i, v) in s.vents.iter().enumerate() {
        let alpha = if selected == Some(LayerKind::Vent(i)) { 1.0 } else { 0.55 };
        let color = [0.35, 0.85, 1.0, alpha];
        let corners = v.corners().map(&project);
        for k in 0..4 {
            if let (Some(a), Some(b)) = (corners[k], corners[(k + 1) % 4]) {
                draw.add_line(a.to_array(), b.to_array(), color).thickness(2.0).build();
            }
        }
        let c = v.placement.translation_mm;
        let tip = c + v.direction() * (0.5 * v.width_mm.max(v.height_mm)).max(10.0);
        if let (Some(a), Some(b)) = (project(c), project(tip)) {
            draw_arrow(&draw, a, b, color);
        }
    }
}

/// The simulated box: how far it reaches past the part on each side.
///
/// Bigger downstream gives an exit jet room to develop and a vent somewhere to
/// stand; bigger anywhere costs cells. Nothing changes until Apply, which is
/// pre-flighted against the GPU's memory like the cell size.
fn domain_controls(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    ui.separator_with_text("Simulated box");
    ui.text_disabled("margins beyond the part, mm, part frame");
    ui.set_item_tooltip(
        "How far the simulated air reaches past the part's bounding box on each side. The \
         faces the mouths open onto are named. Grow the outlet's face to give the exit jet \
         room; grow any face to make room for a vent. Costs cells: the estimate below is \
         what Apply would build.",
    );
    let roles = face_roles(cx.state);
    let mut m = cx.state.pending_domain_mm;
    let mut changed = false;
    for (k, axis) in ["-X", "+X", "-Y", "+Y", "-Z", "+Z"].iter().enumerate() {
        let label = match &roles[k] {
            Some(role) => format!("{axis} ({role})##dom{k}"),
            None => format!("{axis}##dom{k}"),
        };
        ui.set_next_item_width(80.0);
        if ui.input_float(&label, &mut m[k]) && m[k].is_finite() {
            m[k] = m[k].clamp(0.0, 2000.0);
            changed = true;
        }
    }
    if changed {
        cx.state.pending_domain_mm = m;
    }
    let s = &mut *cx.state;
    let bits = |v: [f32; 6]| v.map(f32::to_bits);
    let unchanged = bits(s.pending_domain_mm) == bits(s.domain_margins_mm);
    let estimate = s.resolution.estimate_for(Some(s.pending_domain_mm)).cloned();
    let blocked = estimate.as_ref().is_none_or(|e| e.blocker.is_some());
    {
        let _off = ui.begin_disabled_with_cond(unchanged || blocked);
        if ui.button("Apply##domain") {
            let pending = s.pending_domain_mm;
            s.push_action(UiAction::SetDomainMargins(pending));
        }
    }
    ui.set_item_tooltip("Rebuild the lattice inside the new box. The flow restarts.");
    ui.same_line();
    {
        let _off = ui.begin_disabled_with_cond(s.params.domain_mm.is_none());
        if ui.button("Automatic") {
            s.push_action(UiAction::ResetDomainMargins);
        }
    }
    ui.set_item_tooltip("Back to the margins the domain rules pick from the mouths.");
    ui.same_line();
    match &estimate {
        None => ui.text_disabled("..."),
        Some(e) if !unchanged && e.blocker.is_some() => {
            ui.text_colored(Health::Bad.color(), "won't fit");
            if let Some(why) = &e.blocker {
                ui.set_item_tooltip(why);
            }
        }
        Some(e) => {
            ui.text_disabled(resolution_summary(e));
            ui.set_item_tooltip(resolution_detail(e, unchanged));
        }
    }
}

/// What each face of the box is to the flow, from the mouths' normals: the
/// face the inlet mouth's outside air is on, and the one the outlet exhausts
/// through. Indexed `[-x, +x, -y, +y, -z, +z]`.
fn face_roles(s: &UiState) -> [Option<String>; 6] {
    let mut roles: [Option<String>; 6] = Default::default();
    let mut name = |mouth: Option<&crate::view::PatchView>, what: &str| {
        let Some(m) = mouth else { return };
        let axis = crate::pose::lattice_axis(m.normal);
        // The normal points into the duct, so the outside is on the min side
        // when it is positive.
        let k = axis * 2 + usize::from(m.normal[axis] < 0.0);
        roles[k] = Some(format!("{what} {}", m.name));
    };
    name(s.mouths.get(s.params.inlet_mouth), "behind inlet");
    name(s.outlet_mouth().and_then(|o| s.mouths.get(o)), "past outlet");
    roles
}

/// World point to viewport pixels, or `None` behind the camera.
fn viewport_projector(camera: &Camera, rect: [f32; 4]) -> impl Fn(glam::Vec3) -> Option<glam::Vec2> {
    let vp = camera.projection_unjittered() * camera.view();
    let [x, y, w, h] = rect;
    move |p: glam::Vec3| {
        let c = vp * p.extend(1.0);
        (c.w > 1.0e-6).then(|| {
            let ndc = c.truncate() / c.w;
            glam::Vec2::new(x + (ndc.x * 0.5 + 0.5) * w, y + (0.5 - ndc.y * 0.5) * h)
        })
    }
}

/// A line from `a` to `b` with a head at `b`, in viewport pixels.
fn draw_arrow(draw: &dear_imgui_rs::DrawListMut<'_>, a: glam::Vec2, b: glam::Vec2, color: [f32; 4]) {
    let along = (b - a).normalize_or_zero();
    if along == glam::Vec2::ZERO {
        return; // pointing straight at the camera
    }
    let side = along.perp() * 6.0;
    let head = b - along * 14.0;
    draw.add_line(a.to_array(), b.to_array(), color).thickness(3.0).build();
    draw.add_line(b.to_array(), (head + side).to_array(), color).thickness(3.0).build();
    draw.add_line(b.to_array(), (head - side).to_array(), color).thickness(3.0).build();
}

/// Numeric placement for the selected obstruction, in the car frame. Same data
/// the gizmo edits, so the two stay in step; the lattice is rebuilt once the
/// numbers stop moving.
fn obstruction_properties(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let Some(layer) = cx.state.layers.selected() else { return };
    let LayerKind::Obstruction(i) = layer.kind else { return };
    let Some(p) = cx.state.obstructions.get_mut(i) else { return };
    ui.separator_with_text("Obstruction");
    ui.text_disabled("car frame; applied when you stop editing");
    let mut origin = p.translation_mm.to_array();
    if ui.drag_float3("origin (mm)", &mut origin) {
        p.translation_mm = glam::Vec3::from(origin);
    }
    ui.set_item_tooltip("Where the obstruction's own STL origin sits in the car.");
    let (yaw, pitch, roll) = p.rotation.to_euler(glam::EulerRot::YXZ);
    let mut e = [yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees()];
    if ui.drag_float3("yaw/pitch/roll##obstruction", &mut e) {
        p.rotation = glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            e[0].to_radians(),
            e[1].to_radians(),
            e[2].to_radians(),
        );
    }
    let mut scale = p.scale;
    if ui.slider_f32("scale", &mut scale, 0.1, 10.0) {
        p.scale = scale.max(1e-3);
    }
}

/// The part itself: its scale, and quarter turns of it inside the lattice.
/// Both re-voxelise once the edit settles, unlike the install pose below.
fn duct_geometry_controls(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    ui.text_disabled("part geometry: rebuilds when you stop editing");
    ui.set_item_tooltip(
        "Changes to the STL itself, about its centre. Scale with the number here or with the          gizmo's scale handles on the Duct layer. \"Bake pose\" turns the part by the quarter          turns nearest the install pose and takes them out of the pose, so the picture stays          put while the mouths move to other lattice faces.",
    );
    ui.set_next_item_width(90.0);
    let mut scale = cx.state.duct.scale;
    if ui.input_float("scale x##duct", &mut scale) && scale.is_finite() {
        cx.state.duct.scale = scale.clamp(0.1, 10.0);
    }
    ui.same_line();
    if ui.button("Bake pose") {
        cx.state.push_action(UiAction::BakePose);
    }
    ui.set_item_tooltip("Turn the part itself by the nearest quarter turns of the install pose.");
    if !cx.state.duct.is_identity() {
        ui.same_line();
        if ui.button("Reset part") {
            cx.state.duct = DuctGeometry::IDENTITY;
        }
        let (yaw, pitch, roll) = cx.state.duct.turn.to_euler(glam::EulerRot::YXZ);
        ui.text_disabled(&format!(
            "part: scale {:.3}, turned {:.0}/{:.0}/{:.0} deg",
            cx.state.duct.scale,
            yaw.to_degrees(),
            pitch.to_degrees(),
            roll.to_degrees()
        ));
    }
}

/// How the duct sits in the car.
///
/// Turning it here turns the world around the lattice, not the part inside it:
/// nothing re-voxelises and nothing restarts, so it is as cheap as orbiting the
/// camera — but the view presets, the ground and anything placed in the world
/// now follow the car rather than the STL's axes.
fn install_pose_controls(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    ui.separator_with_text("Install pose");
    ui.text_disabled("car frame, +Y up");
    ui.set_item_tooltip(
        "Turns the part in the world, not in the lattice: the solver keeps running in the \
         part's own frame, so nothing restarts. There is no gravity in the model, so the \
         flow relative to the part is the same in every pose. Select the Duct layer to \
         turn it with the gizmo.",
    );

    let pose = &mut cx.state.install;
    let mut e = pose.euler_deg().to_array();
    if ui.drag_float3("yaw/pitch/roll", &mut e) {
        pose.set_euler_deg(glam::Vec3::from(e));
    }
    for (axis, name) in [(1usize, "yaw  "), (0, "pitch"), (2, "roll ")] {
        ui.text(name);
        ui.same_line();
        if ui.small_button(&format!("-90##{axis}")) {
            pose.nudge(axis, -90.0);
        }
        ui.same_line();
        if ui.small_button(&format!("+90##{axis}")) {
            pose.nudge(axis, 90.0);
        }
    }
    let mut offset = pose.offset_mm.to_array();
    if ui.drag_float3("offset (mm)", &mut offset) {
        pose.offset_mm = glam::Vec3::from(offset);
    }
    if ui.button("Reset pose") {
        *pose = InstallPose::IDENTITY;
    }

    // Where the air goes, in the car. Mouth normals point into the duct, so
    // the air arrives along the inlet's (tilted by any louver aim) and leaves
    // against the outlet's.
    let s = &*cx.state;
    if let Some(m) = s.mouths.get(s.params.inlet_mouth) {
        let d = s.params.inlet_direction(m.normal, crate::pose::lattice_axis(m.normal));
        ui.text(&format!("air in:  {}", describe_direction(s.install.dir_to_world(d))));
    }
    if let Some(m) = s.outlet_mouth().and_then(|o| s.mouths.get(o)) {
        ui.text(&format!("air out: {}", describe_direction(s.install.dir_to_world(-m.normal))));
    }
    if let Some(d) = s.metrics.jet_direction {
        ui.text(&format!("jet:     {}", describe_direction(s.install.dir_to_world(d))));
        ui.set_item_tooltip("Measured momentum direction of the air leaving the outlet.");
    }
}

/// The vent louver aim: the angle the air comes into the inlet at.
///
/// Set in car terms, because the louver is part of the car; the solver gets it
/// relative to the part through [`UiState::sync_inlet_tilt`]. Hot-applied.
fn inlet_air_controls(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    ui.separator_with_text("Inlet air angle");
    if !cx.state.vents.is_empty() {
        ui.text_disabled("the vents supply the air; the mouth is an opening");
        ui.set_item_tooltip("Aim each vent from its own layer. Remove every vent to drive the mouth directly again.");
        return;
    }
    ui.text_disabled("vent louver aim, car frame");
    ui.set_item_tooltip(
        "Turns the air entering the inlet; the component through the mouth stays at the inlet \
         U. Applied at the mouth plane in the room domain; the plenum domain's walled \
         extension straightens most of it out.",
    );
    let max = crate::params::MAX_INLET_TILT_DEG;
    let mut aim = cx.state.inlet_louver_deg;
    let mut changed = ui.slider_f32("up/down (deg)", &mut aim[0], -max, max);
    ui.set_item_tooltip("Positive tips the air toward car up (+Y).");
    changed |= ui.slider_f32("sideways (deg)", &mut aim[1], -max, max);
    ui.set_item_tooltip("Positive turns the air to the right, looking downstream.");
    if ui.small_button("straight in") {
        aim = [0.0, 0.0];
        changed = true;
    }
    if changed {
        cx.state.inlet_louver_deg = aim;
        cx.state.sync_inlet_tilt();
    }

    let s = &*cx.state;
    if s.params.inlet_tilt_deg == [0.0, 0.0] {
        return;
    }
    let Some(m) = s.mouths.get(s.params.inlet_mouth) else { return };
    let d = s.params.inlet_direction(m.normal, crate::pose::lattice_axis(m.normal));
    let off_normal = d.normalize().dot(m.normal).clamp(-1.0, 1.0).acos().to_degrees();
    let speed = s.params.inlet_velocity_ms * d.length();
    ui.text(&format!("{off_normal:.0}° off-normal, |u| {speed:.2} m/s"));
    ui.set_item_tooltip(
        "The velocity through the mouth is held at the inlet U, so the flow rate is nominally \
         the untilted one; the inlet's density shifts with the angle, which moves the measured \
         flow a few percent at 30°. K now includes the cost of turning the stream in the \
         mouth, so it is not comparable with an untilted K.",
    );
    let lattice_speed = s.params.u_lb * d.length() as f64;
    if lattice_speed > 0.1 {
        ui.text_colored(
            Health::Watch.color(),
            &format!("lattice speed {lattice_speed:.3}: compressibility error rising"),
        );
    }
}

/// The louver aim, drawn where the air comes in: an arrow along the tilted
/// inlet direction, ending at the inlet mouth. Only while a tilt is set; at
/// zero it would restate the mouth normal and clutter every screenshot.
fn inlet_arrow(frame: &UiFrame<'_>, cx: &PanelContext<'_>) {
    let s = &*cx.state;
    if s.params.inlet_tilt_deg == [0.0, 0.0] || !s.vents.is_empty() {
        return;
    }
    let Some(m) = s.mouths.get(s.params.inlet_mouth) else { return };
    let d = s.params.inlet_direction(m.normal, crate::pose::lattice_axis(m.normal));
    let d = s.install.dir_to_world(d.normalize_or_zero());
    let tip = s.to_world(m.center_mm);
    let tail = tip - d * (2.0 * m.hydraulic_diameter_mm).max(10.0);
    let project = viewport_projector(cx.camera, s.viewport_rect_logical());
    let (Some(a), Some(b)) = (project(tail), project(tip)) else { return };
    draw_arrow(&frame.ui.get_background_draw_list(), a, b, [0.35, 0.85, 1.0, 0.95]);
}

fn slice_properties(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let Some(layer) = cx.state.layers.selected() else { return };
    let LayerKind::Slice(i) = layer.kind else { return };
    let Some(slice) = cx.state.slices.get_mut(i) else { return };
    ui.separator_with_text("Slice");
    ui.checkbox("scalar", &mut slice.show_scalar);
    ui.same_line();
    ui.checkbox("vectors", &mut slice.show_vectors);
    ui.same_line();
    ui.checkbox("LIC", &mut slice.lic);
    ui.checkbox("clip geometry", &mut slice.clip_geometry);
    ui.slider_f32("arrow spacing (mm)", &mut slice.vector_spacing_mm, 0.5, 10.0);
    ui.slider_f32("opacity##slice", &mut slice.opacity, 0.0, 1.0);
    let mut origin = slice.origin_mm.to_array();
    if ui.drag_float3("origin (mm)", &mut origin) {
        slice.origin_mm = glam::Vec3::from(origin);
    }
}

fn particle_properties(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    if !ui.collapsing_header("Particles", TreeNodeFlags::empty()) {
        return;
    }
    let p = &mut cx.state.particles;
    ui.checkbox("enabled##particles", &mut p.enabled);
    let mut count = p.count as i32;
    if ui.slider_i32("count", &mut count, 1_000, 200_000) {
        p.count = count as u32;
    }
    ui.slider_f32("size (px)", &mut p.size_px, 1.0, 8.0);
    ui.slider_f32("display speed", &mut p.speed_scale, 0.1, 8.0);
    ui.set_item_tooltip(
        "A display gain, not a physical rate: the simulation runs thousands of times slower than \
         real time, so particles moving at the true rate would be motionless.",
    );
    let mut trail = p.trail as i32;
    if ui.slider_i32("trail", &mut trail, 0, 64) {
        p.trail = trail as u32;
    }
    let mut seed_ix = ParticleSeed::ALL.iter().position(|s| *s == p.seed).unwrap_or(0);
    ui.set_next_item_width(-1.0);
    if ui.combo("##seed", &mut seed_ix, &ParticleSeed::ALL, |s| {
        std::borrow::Cow::Borrowed(s.label())
    }) {
        p.seed = ParticleSeed::ALL[seed_ix];
    }
    ui.checkbox("colour by field", &mut p.color_by_field);
}

/// The opacity curve, edited with ImPlot's draggable points.
///
/// Direct manipulation rather than a table of numbers: an opacity curve is a
/// shape, and the only way to tune one is to see the volume change while you
/// drag it. Points are clamped to `[0, 1]` in both axes because
/// `OpacityCurve::sanitise` will do it anyway, and letting a point fly off the
/// plot and then snap back reads as the widget being broken.
fn opacity_curve_editor(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>, scratch: &mut Scratch) {
    let ui = frame.ui;
    let plot = &frame.plot;

    if ui.button("+ point") {
        // Halfway along, at whatever the curve currently says: a new point that
        // changes nothing until it is dragged.
        let alpha = cx.transfer.curve.eval(0.5);
        cx.transfer.curve.try_insert(0.5, alpha);
    }
    ui.same_line();
    if ui.button("- point") {
        if let Some(i) = scratch.dragging_curve_point {
            cx.transfer.curve.try_remove(i);
            scratch.dragging_curve_point = None;
        } else {
            let n = cx.transfer.curve.points.len();
            if n > 2 {
                cx.transfer.curve.try_remove(n / 2);
            }
        }
    }
    ui.same_line();
    ui.text_disabled("drag the points");

    if let Some(token) = plot.begin_plot_with_size("##opacity curve", [-1.0, 150.0]) {
        plot.setup_axes(
            Some("normalised value"),
            Some("alpha"),
            AxisFlags::NO_MENUS,
            AxisFlags::NO_MENUS,
        );
        plot.setup_axes_limits(0.0, 1.0, 0.0, 1.0, PlotCond::Always);
        plot.setup_finish();

        // The curve as the shader will sample it.
        let xs: Vec<f64> = (0..=64).map(|i| i as f64 / 64.0).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|t| cx.transfer.curve.eval(*t as f32) as f64)
            .collect();
        LinePlot::new("alpha", &xs, &ys).plot(plot);

        let mut changed = false;
        for i in 0..cx.transfer.curve.points.len() {
            let p = cx.transfer.curve.points[i];
            let (mut x, mut y) = (p[0] as f64, p[1] as f64);
            let r = plot.drag_point(
                DragToolId::new(i as i32),
                &mut x,
                &mut y,
                [0.52, 0.74, 0.96, 1.0],
                5.0,
                DragToolFlags::NONE,
            );
            if r.held {
                scratch.dragging_curve_point = Some(i);
            }
            if r.changed {
                cx.transfer.curve.points[i] =
                    [(x as f32).clamp(0.0, 1.0), (y as f32).clamp(0.0, 1.0)];
                changed = true;
            }
        }
        if changed {
            cx.transfer.curve.sanitise();
            cx.transfer.sanitise();
        }
        token.end();
    }
}

// ---------------------------------------------------------------------------
// legend
// ---------------------------------------------------------------------------

/// The colour bar, with draggable end handles, a double-click auto-range and a
/// lock.
///
/// The lock is the load-bearing control: two designs rendered with independently
/// auto-ranged colour bars look identical no matter how different they are,
/// because auto-ranging removes exactly the information being compared.
fn legend_panel(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;

    // Keep the legend in step with the transfer function unless it is locked;
    // `sync_from` enforces that, so no path here can quietly unfreeze an A/B.
    cx.state.legend.sync_from(cx.transfer);

    let mut locked = cx.state.legend.locked;
    if ui.checkbox("lock range", &mut locked) {
        cx.state.legend.locked = locked;
    }
    ui.set_item_tooltip(
        "Freeze the range so an A/B comparison is a comparison. Survives a field switch.",
    );
    ui.same_line();
    if ui.button("Auto") {
        if let Some((lo, hi)) = cx.field_extremes {
            cx.state.legend.auto_range(lo, hi);
        }
        cx.state.push_action(UiAction::AutoRangeLegend);
    }
    ui.set_item_tooltip("Fit the bar to the measured extremes (or double-click the bar).");

    let field = cx.state.field;
    ui.text_disabled(format!("{} [{}]", field.name(), field.unit()));

    // The bar itself.
    let avail = ui.content_region_avail();
    let bar_w = 34.0f32;
    let bar_h = (avail[1] - 40.0).clamp(80.0, 400.0);
    let origin = ui.cursor_screen_pos();
    let draw = ui.get_window_draw_list();

    let interp = cx.transfer.interpolation;
    let map = cx.transfer.map;
    const STRIPS: usize = 96;
    for i in 0..STRIPS {
        let t0 = i as f32 / STRIPS as f32;
        let t1 = (i + 1) as f32 / STRIPS as f32;
        let c = map.sample(0.5 * (t0 + t1), interp);
        // Bar runs high at the top, so the strip at parameter t sits at
        // (1 - t) down the bar.
        let y0 = origin[1] + (1.0 - t1) * bar_h;
        let y1 = origin[1] + (1.0 - t0) * bar_h;
        draw.add_rect(
            [origin[0], y0],
            [origin[0] + bar_w, y1],
            [c.x, c.y, c.z, 1.0],
        )
        .filled(true)
        .build();
    }
    draw.add_rect(
        [origin[0], origin[1]],
        [origin[0] + bar_w, origin[1] + bar_h],
        [0.26, 0.28, 0.33, 0.9],
    )
    .build();

    // Ticks.
    let legend = cx.state.legend;
    for tick in legend.ticks(6) {
        let t = legend.normalise(tick).clamp(0.0, 1.0);
        let y = origin[1] + (1.0 - t) * bar_h;
        draw.add_line(
            [origin[0] + bar_w, y],
            [origin[0] + bar_w + 5.0, y],
            [0.75, 0.78, 0.84, 1.0],
        )
        .build();
        draw.add_text(
            [origin[0] + bar_w + 9.0, y - 7.0],
            [0.82, 0.85, 0.90, 1.0],
            format::short(tick as f64),
        );
    }

    // Two invisible buttons over the ends, which is what makes them draggable
    // without inventing hit-testing.
    let handle_h = 16.0;
    let mut dragged: Option<(Handle, f32)> = None;
    for (handle, at_top) in [(Handle::High, true), (Handle::Low, false)] {
        let y = if at_top { origin[1] } else { origin[1] + bar_h - handle_h };
        ui.set_cursor_screen_pos([origin[0] - 6.0, y]);
        let id = if at_top { "##hi" } else { "##lo" };
        ui.invisible_button(id, [bar_w + 12.0, handle_h]);
        let hovered = ui.is_item_hovered();
        if hovered || ui.is_item_active() {
            draw.add_rect(
                [origin[0] - 6.0, y],
                [origin[0] + bar_w + 6.0, y + handle_h],
                [1.0, 1.0, 1.0, 0.35],
            )
            .filled(true)
            .build();
        }
        if ui.is_item_active() {
            cx.state.legend.dragging = Some(handle);
            let delta = ui.mouse_drag_delta(MouseButton::Left);
            if delta[1].abs() > 0.0 {
                // Screen y grows downward, the bar's value grows upward.
                dragged = Some((handle, -delta[1] / bar_h.max(1.0)));
                ui.reset_mouse_drag_delta(MouseButton::Left);
            }
        }
    }
    if !ui.is_mouse_down(MouseButton::Left) {
        cx.state.legend.dragging = None;
    }
    if let Some((handle, frac)) = dragged {
        if cx.state.legend.drag_handle(handle, frac) {
            cx.state.legend.apply_to(cx.transfer);
        }
    }

    // Double-clicking the bar auto-ranges, which is the gesture everyone tries
    // first.
    ui.set_cursor_screen_pos(origin);
    ui.invisible_button("##bar", [bar_w, bar_h]);
    if ui.is_item_hovered() && ui.is_mouse_double_clicked(MouseButton::Left) {
        if let Some((lo, hi)) = cx.field_extremes {
            if cx.state.legend.auto_range(lo, hi) {
                cx.state.legend.apply_to(cx.transfer);
            }
        }
        cx.state.push_action(UiAction::AutoRangeLegend);
    }

    ui.set_cursor_screen_pos([origin[0], origin[1] + bar_h + 8.0]);
    let (mut lo, mut hi) = (cx.state.legend.lo, cx.state.legend.hi);
    ui.set_next_item_width(-1.0);
    let edited = ui.drag_float_range2("range", &mut lo, &mut hi);
    if edited && !cx.state.legend.locked {
        cx.state.legend.lo = lo;
        cx.state.legend.hi = hi;
        cx.state.legend.sanitise();
        cx.state.legend.apply_to(cx.transfer);
    }
    if cx.state.legend.locked {
        ui.text_colored(Health::Watch.color(), "locked");
    }
}

// ---------------------------------------------------------------------------
// plots
// ---------------------------------------------------------------------------

fn plots_panel(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let Some(bar) = ui.tab_bar("##plottabs") else { return };
    for tab in PlotTab::ALL {
        if let Some(item) = ui.tab_item(tab.label()) {
            cx.state.plot_tab = tab;
            match tab {
                PlotTab::Flow => flow_plot(frame, cx),
                PlotTab::PressureDrop => pressure_plot(frame, cx),
                PlotTab::Probes => probe_plot(frame, cx),
                PlotTab::Residence => residence_plot(frame, cx),
                PlotTab::Histogram => histogram_plot(frame, cx),
                PlotTab::Spectrum => spectrum_plot(frame, cx),
            }
            item.end();
        }
    }
    bar.end();
}

fn x_label(cx: &PanelContext<'_>) -> &'static str {
    if cx.state.metrics.convergence.x_is_steps { "step" } else { "time (s)" }
}

fn plot_trace(plot: &dear_implot::PlotUi<'_>, trace: &Trace) {
    if trace.x.is_empty() {
        return;
    }
    if trace.log_y {
        // No log axis without reaching into `implot-sys`, which is not a
        // dependency here; plotting log10 of the value is the same picture and
        // the axis label says so.
        let ys: Vec<f64> = trace
            .y
            .iter()
            .map(|v| if *v > 0.0 { v.log10() } else { -30.0 })
            .collect();
        LinePlot::new(&trace.name, &trace.x, &ys).plot(plot);
    } else {
        LinePlot::new(&trace.name, &trace.x, &trace.y).plot(plot);
    }
}

fn flow_plot(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let plot = &frame.plot;
    let conv = &cx.state.metrics.convergence;

    ui.text(format!(
        "mass imbalance {}",
        format::uncertain(conv.mass_imbalance, Quantity::plain())
    ));
    ui.same_line();
    ui.text_colored(conv.mass_imbalance.state.color(), conv.mass_imbalance.state.label());
    ui.set_item_tooltip(
        "|mdot_in - mdot_out| / mdot_in, on mass flux rather than volume flow: the air \
         expands across the duct, so the volumetric rates legitimately differ. The \
         separation of the two curves is the honest convergence signal for a duct.",
    );

    if let Some(token) = plot.begin_plot_with_size("##flow", [-1.0, -1.0]) {
        plot.setup_axes(
            Some(x_label(cx)),
            Some("Q (m^3/s)"),
            AxisFlags::AUTO_FIT,
            AxisFlags::AUTO_FIT,
        );
        plot.setup_finish();
        plot_trace(plot, &conv.flow_in);
        plot_trace(plot, &conv.flow_out);
        token.end();
    }
}

fn pressure_plot(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let plot = &frame.plot;
    let conv = &cx.state.metrics.convergence;
    if let Some(token) = plot.begin_plot_with_size("##dp", [-1.0, -1.0]) {
        plot.setup_axes(Some(x_label(cx)), Some("dp (Pa)"), AxisFlags::AUTO_FIT, AxisFlags::AUTO_FIT);
        plot.setup_finish();
        plot_trace(plot, &conv.pressure_drop);
        for t in &conv.residuals {
            plot_trace(plot, t);
        }
        token.end();
    }
}

fn probe_plot(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let plot = &frame.plot;
    if cx.state.probes.is_empty() {
        ui.text_disabled("no probes; ctrl+click the view to drop one");
        return;
    }
    if let Some(token) = plot.begin_plot_with_size("##probes", [-1.0, -1.0]) {
        plot.setup_axes(Some("time (s)"), Some("p (Pa)"), AxisFlags::AUTO_FIT, AxisFlags::AUTO_FIT);
        plot.setup_finish();
        for p in cx.state.probes.items() {
            if p.visible {
                plot_trace(plot, &p.pressure);
            }
        }
        token.end();
    }
}

fn residence_plot(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let plot = &frame.plot;
    let rtd = &cx.state.metrics.residence_time;

    ui.text(format!(
        "mean {}   variance ratio {}   trapped {}",
        format::uncertain(rtd.mean_residence_s, Quantity::new(1.0, "s")),
        format::uncertain(rtd.variance_ratio, Quantity::plain()),
        format::uncertain(rtd.trapped_fraction, Quantity::plain()),
    ));
    ui.set_item_tooltip(
        "Variance ratio: 0 is plug flow, 1 is a perfectly stirred tank. Above ~0.5 in a duct \
         means a large recirculation, which no pressure-drop number reveals.",
    );
    if rtd.ideal_residence_s.is_finite() && rtd.ideal_residence_s > 0.0 {
        ui.same_line();
        ui.text_disabled(format!("(ideal {} s)", format::short(rtd.ideal_residence_s)));
    }

    if !rtd.histogram.is_valid() {
        ui.text_disabled("no residence-time distribution yet");
        return;
    }
    if let Some(token) = plot.begin_plot_with_size("##rtd", [-1.0, -1.0]) {
        plot.setup_axes(Some("t (s)"), Some("E(t)"), AxisFlags::AUTO_FIT, AxisFlags::AUTO_FIT);
        plot.setup_finish();
        let centers = rtd.histogram.centers();
        PositionalBarPlot::new("E(t)", &centers, &rtd.histogram.counts)
            .with_bar_size(rtd.histogram.bin_width() * 0.85)
            .plot(plot);
        token.end();
    }
}

fn histogram_plot(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let plot = &frame.plot;
    let h = &cx.state.metrics.outlet_velocity_histogram;
    if !h.is_valid() {
        ui.text_disabled("no outlet histogram yet");
        return;
    }
    ui.text(format!(
        "area-weighted mean {}",
        format::uncertain(h.mean, Quantity::velocity(cx.state.units))
    ));
    if let Some(token) = plot.begin_plot_with_size("##hist", [-1.0, -1.0]) {
        plot.setup_axes(
            Some(if h.unit.is_empty() { "value" } else { h.unit.as_str() }),
            Some("area fraction"),
            AxisFlags::AUTO_FIT,
            AxisFlags::AUTO_FIT,
        );
        plot.setup_finish();
        let centers = h.centers();
        PositionalBarPlot::new(&h.name, &centers, &h.counts)
            .with_bar_size(h.bin_width() * 0.85)
            .plot(plot);
        token.end();
    }
}

fn spectrum_plot(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    let plot = &frame.plot;
    let s = &cx.state.metrics.spectrum;
    ui.text_colored(
        Health::Watch.color(),
        "provisional: the acoustics work is not wired up yet",
    );
    if s.freq_hz.is_empty() {
        ui.text_disabled("no spectrum computed");
        return;
    }
    if let Some(token) = plot.begin_plot_with_size("##spectrum", [-1.0, -1.0]) {
        plot.setup_axes(
            Some("f (Hz)"),
            Some(if s.db { "dB re 20 uPa" } else { "Pa" }),
            AxisFlags::AUTO_FIT,
            AxisFlags::AUTO_FIT,
        );
        plot.setup_finish();
        LinePlot::new(&s.source, &s.freq_hz, &s.magnitude).plot(plot);
        if s.nyquist_hz.is_finite() && s.nyquist_hz > 0.0 {
            let x = [s.nyquist_hz, s.nyquist_hz];
            let lo = s.magnitude.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = s.magnitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let y = [lo, hi];
            ScatterPlot::new("Nyquist", &x, &y).plot(plot);
        }
        token.end();
    }
}

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

fn diagnostics_panel(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>, docked: bool) {
    let ui = frame.ui;
    ui.text(format!("Dear ImGui {}", crate::imgui_version()));
    ui.text(format!(
        "docking compiled in: {}   dockspace submitted: {docked}",
        crate::HAS_DOCKING
    ));
    if !crate::HAS_DOCKING || !docked {
        ui.text_colored(
            Health::Watch.color(),
            "the layout has degraded to floating windows",
        );
    }
    ui.separator();
    ui.text(format!(
        "grid {} x {} x {} at dx = {} mm ({} cells)",
        cx.grid.dims.x,
        cx.grid.dims.y,
        cx.grid.dims.z,
        cx.grid.dx_mm,
        format::grouped(cx.grid.cell_count())
    ));
    let m = &cx.state.metrics;
    ui.text(format!(
        "Re {}   Ma_lb {}   tau0 {}",
        format::short(m.reynolds),
        format::short(m.mach_lb),
        format::short(m.tau0)
    ));
    ui.text(format!("steps taken: {}", format::grouped(cx.step)));

    ui.separator_with_text("lattice warnings");
    if m.warnings.is_empty() {
        ui.text_disabled("none");
    } else {
        for w in &m.warnings {
            ui.text_colored(Health::Watch.color(), w);
        }
    }

    ui.separator_with_text("GPU passes");
    if cx.state.profiling.is_empty() {
        ui.text_disabled("no timings yet");
    } else {
        for line in &cx.state.profiling {
            ui.text_disabled(line);
        }
    }

    ui.separator();
    ui.checkbox("layers", &mut cx.state.panels.layers);
    ui.same_line();
    ui.checkbox("properties", &mut cx.state.panels.properties);
    ui.same_line();
    ui.checkbox("plots", &mut cx.state.panels.plots);
    ui.checkbox("legend", &mut cx.state.panels.legend);
    ui.same_line();
    ui.checkbox("toolbar", &mut cx.state.panels.hud);
    if ui.button("Screenshot") {
        cx.state.push_action(UiAction::Screenshot);
    }
}

// ---------------------------------------------------------------------------
// gizmo
// ---------------------------------------------------------------------------

/// The transform gizmo, when a placeable layer is selected.
///
/// Matrices go across as `[f32; 16]` column-major rather than `glam::Mat4`:
/// `dear-imguizmo` is built against a different glam major than this workspace,
/// so its `Mat4Like` is implemented for *its* `Mat4`, not ours. The array form
/// is the one both agree on, and `to_cols_array` is exact.
fn gizmo(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let g = &frame.gizmo;
    let Some(id) = cx.state.gizmo_target() else {
        cx.state.gizmo_active = false;
        return;
    };
    let Some(layer) = cx.state.layers.get(id) else { return };
    let kind = layer.kind;

    let pivot = cx.state.install_pivot_mm;
    let pose = cx.state.install;

    // What the handles move, as a world-space matrix.
    let world = match kind {
        // A vent's size is its own fields; there is nothing for the scale
        // handles to do.
        LayerKind::Vent(_) if cx.state.gizmo_mode == GizmoMode::Scale => {
            cx.state.gizmo_active = false;
            return;
        }
        // Move and turn set the install pose (the picture); the scale handles
        // scale the part itself, which is re-voxelised once the drag settles.
        LayerKind::Duct => {
            pose.gizmo_matrix(pivot) * glam::Mat4::from_scale(glam::Vec3::splat(cx.state.duct.scale))
        }
        LayerKind::Vent(i) => match cx.state.vents.get(i) {
            Some(v) => v.placement.matrix(),
            None => return,
        },
        // Slices are stored on the part, so they are shown through the install
        // pose and brought back through its inverse.
        LayerKind::Slice(i) => match cx.state.slices.get(i) {
            Some(s) => pose.matrix(pivot) * s.placement().matrix(),
            None => return,
        },
        // Obstructions are placed in the car, which is the gizmo's frame.
        LayerKind::Obstruction(i) => match cx.state.obstructions.get(i) {
            Some(p) => p.matrix(),
            None => return,
        },
        _ => return,
    };

    let [x, y, w, h] = cx.state.viewport_rect_logical();
    g.set_rect(x, y, w, h);
    g.set_drawlist_background();

    let view = cx.camera.view().to_cols_array();
    let proj = cx.camera.projection_unjittered().to_cols_array();
    let mut model = world.to_cols_array();

    let op = match cx.state.gizmo_mode {
        GizmoMode::Translate => dear_imguizmo::Operation::TRANSLATE,
        GizmoMode::Rotate => dear_imguizmo::Operation::ROTATE,
        GizmoMode::Scale => dear_imguizmo::Operation::SCALE,
        GizmoMode::Off => return,
    };
    let mode = match cx.state.gizmo_space {
        GizmoSpace::World => dear_imguizmo::Mode::World,
        GizmoSpace::Local => dear_imguizmo::Mode::Local,
    };
    let snap = snap_for(cx.state.gizmo_mode, cx.state.gizmo_snap_mm, cx.state.gizmo_snap_deg)
        .map(|s| [s, s, s]);

    let used = g.manipulate(
        &view,
        &proj,
        op,
        mode,
        &mut model,
        None,
        snap.as_ref(),
        None,
        None,
    );
    cx.state.gizmo_active = used || g.is_over();

    if used {
        let m = glam::Mat4::from_cols_array(&model);
        match kind {
            LayerKind::Duct => {
                cx.state.install = InstallPose::from_gizmo_matrix(m, pivot);
                if cx.state.gizmo_mode == GizmoMode::Scale {
                    cx.state.duct.scale = Placement::from_matrix(m).scale.clamp(0.1, 10.0);
                }
            }
            LayerKind::Vent(i) => {
                if let Some(v) = cx.state.vents.get_mut(i) {
                    v.placement = Placement { scale: 1.0, ..Placement::from_matrix(m) };
                }
            }
            LayerKind::Slice(i) => {
                let p = Placement::from_matrix(pose.inverse(pivot) * m);
                if let Some(s) = cx.state.slices.get_mut(i) {
                    s.set_placement(p);
                }
            }
            // Edited in place; the app rebuilds the lattice once the drag ends.
            LayerKind::Obstruction(i) => {
                if let Some(o) = cx.state.obstructions.get_mut(i) {
                    *o = Placement::from_matrix(m);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// toasts
// ---------------------------------------------------------------------------

/// Transient notifications, stacked from the top-right of the display.
///
/// Windows rather than a draw-list overlay so they can be clicked to dismiss,
/// and `NO_DOCKING` so a stray drag cannot dock a notification into the layout.
fn toasts(frame: &UiFrame<'_>, cx: &mut PanelContext<'_>) {
    let ui = frame.ui;
    if cx.state.toasts.is_empty() {
        return;
    }
    let display = frame.display_size();
    let width = 380.0f32;
    let mut y = 32.0f32;
    let mut dismiss: Option<usize> = None;

    let toasts: Vec<crate::toast::Toast> = cx.state.toasts.iter().cloned().collect();
    for (i, toast) in toasts.iter().enumerate() {
        let alpha = 0.35 + 0.6 * toast.remaining();
        let flags = WindowFlags::NO_DECORATION
            | WindowFlags::NO_DOCKING
            | WindowFlags::ALWAYS_AUTO_RESIZE
            | WindowFlags::NO_SAVED_SETTINGS
            | WindowFlags::NO_FOCUS_ON_APPEARING
            | WindowFlags::NO_NAV;
        let title = format!("##toast{i}");
        let colour = toast.severity.color();
        let height = ui
            .window(title.as_str())
            .flags(flags)
            .position([display[0] - width - 20.0, y], Condition::Always)
            .size([width, 0.0], Condition::Always)
            .bg_alpha(alpha)
            .build(|| {
                let _c = ui.push_style_color(StyleColor::Text, colour);
                ui.text(toast.display_title());
                drop(_c);
                if !toast.detail.is_empty() {
                    ui.text_wrapped(&toast.detail);
                }
                if ui.small_button("dismiss") {
                    dismiss = Some(i);
                }
                ui.window_size()[1]
            })
            .unwrap_or(60.0);
        y += height + 8.0;
    }
    if let Some(i) = dismiss {
        cx.state.toasts.dismiss(i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dock_layout_is_valid_and_names_every_panel() {
        // A malformed layout fails at `build()` inside a frame, where the only
        // symptom is windows appearing undocked. Validating it here means the
        // failure is a test rather than a shrug.
        let panels = Panels::new().expect("panel layout");
        panels.layout.validate().expect("layout validates");

        let mut seen = Vec::new();
        collect(&panels.layout, &mut seen);
        for want in ["toolbar", "layers", "properties", "legend", "plots", "diagnostics"] {
            assert!(seen.iter().any(|s| s == want), "{want} is not in the layout");
        }
        // Exactly one empty leaf: the central node the 3D view shows through.
        assert_eq!(empty_leaves(&panels.layout), 1, "the viewport hole is missing or duplicated");
    }

    fn collect(layout: &DockLayout, out: &mut Vec<String>) {
        match layout {
            DockLayout::Tabs(keys) => out.extend(keys.iter().map(|k| k.stable_id().to_string())),
            DockLayout::Split { first, second, .. } => {
                collect(first, out);
                collect(second, out);
            }
        }
    }

    fn empty_leaves(layout: &DockLayout) -> usize {
        match layout {
            DockLayout::Tabs(keys) => usize::from(keys.is_empty()),
            DockLayout::Split { first, second, .. } => empty_leaves(first) + empty_leaves(second),
        }
    }
}

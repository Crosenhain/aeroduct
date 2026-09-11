//! AeroDuct: interactive GPU airflow simulation for 3D-printed ducts.
//!
//! # The model
//!
//! Ansys Discovery Live, not a batch solver with a GUI on top. **The simulation
//! is always running**, every edit applies to the running solver, and there is
//! no Run button anywhere. The consequence for this file is that the frame loop
//! has no modes: it applies whatever changed, steps the solver, derives the
//! fields, draws, and does it again.
//!
//! ```text
//! winit event  -> ImGui, then the camera if ImGui did not want it
//! redraw:
//!   drain the UI action queue        (loads, presets, probes, screenshots)
//!   peek_params, rebuild if needed   -> transactional: a failure keeps the old one
//!   sync_params                      -> hot-apply, and a loud reset
//!   N solver steps                   (N from the auto-tuner)
//!   compute_macroscopic              -> velocity/density textures
//!   derive fields -> render 3D       to the whole surface
//!   ImGui over the top               with a passthrough central dock node
//!   present
//! ```
//!
//! # Honesty about "real-time"
//!
//! At `dx = 0.4 mm` one physical second is 200,000 LBM steps, so a second of
//! simulated air costs tens of minutes of wall clock. Everything in this app
//! that says "real-time" means *rendering* in real time — the picture keeps up
//! with the mouse. [`ad_ui::Clock::status_line`] states the physics ratio
//! explicitly rather than letting a frame rate imply anything about it.

mod domain;
mod metrics;
mod plenum;
mod png;
mod profile;
mod resolution;
mod sim;
mod tracers;

use std::sync::Arc;
use std::time::Instant;

use ad_gpu::GpuContext;
use ad_render::{
    DerivedField, FieldSources, FrameInput, GpuMesh, MeshData, MeshDisplay, MeshStyle, MeshVertex,
    OrbitController, Renderer, RendererConfig, ViewPreset,
};
use ad_ui::{
    panels::{PanelContext, Panels},
    Gesture, MouseState, RateMeter, SimParams, UiAction, UiBackend, UiState,
};
use anyhow::{Context as _, Result};
use glam::{IVec3, Mat3, Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::metrics::{DuctMetricsSource, MetricsSource, PlaceholderMetrics, SampleContext};
use crate::profile::Phase;
use crate::sim::Sim;
use crate::tracers::TracerRtd;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new().context("creating the event loop")?;
    // Poll rather than Wait: the simulation is always running, so there is
    // always work even with no input.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .context("running the event loop")?;
    if let Some(e) = app.fatal {
        return Err(e);
    }
    Ok(())
}

#[derive(Default)]
struct App {
    running: Option<Running>,
    fatal: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        match Running::new(event_loop) {
            Ok(r) => self.running = Some(r),
            Err(e) => {
                log::error!("startup failed: {e:#}");
                self.fatal = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(r) = self.running.as_mut() else {
            return;
        };
        if let Err(e) = r.window_event(event_loop, event) {
            log::error!("frame failed: {e:#}");
            self.fatal = Some(e);
            event_loop.exit();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(r) = self.running.as_ref() {
            r.window.request_redraw();
        }
    }
}

struct Running {
    window: Arc<Window>,
    gpu: GpuContext,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    renderer: Renderer,
    render_scene: ad_render::Scene,
    controller: OrbitController,

    sim: Sim,
    metrics: Box<dyn MetricsSource>,
    /// CPU tracer advection for the residence-time distribution. `None` unless
    /// `AERODUCT_RTD` asked for it; see [`crate::tracers`] for what it costs.
    tracers: Option<TracerRtd>,

    ui: UiBackend,
    panels: Panels,
    state: UiState,

    /// The obstructions' meshes, in the order of the scene's obstruction
    /// instances and of `state.obstructions`, which says where each sits.
    obstructions: Vec<ObstructionAsset>,
    /// `state.obstructions` and the install pose as the running lattice has
    /// them. A difference is an edit waiting to be voxelised; see
    /// [`Running::settle_obstructions`].
    committed_placements: Vec<ad_ui::Placement>,
    committed_pose: ad_ui::InstallPose,
    /// The vents (car frame) the running lattice was flagged from, to roll
    /// back to when a rebuild with new ones fails.
    committed_vents: Vec<ad_ui::VentSettings>,
    /// The part's own scale and quarter turns as the running lattice has them.
    committed_duct: ad_ui::pose::DuctGeometry,
    /// The pose last written to the log, so a gizmo drag leaves one line.
    logged_pose: ad_ui::InstallPose,
    /// A pending placement edit — obstructions, vents, or the pose while
    /// either exists — and when it last changed.
    placement_edit: Option<PlacementEdit>,
    /// For each mesh in `render_scene`, the obstruction it draws, or `None`
    /// for the duct.
    render_roles: Vec<Option<usize>>,

    last_frame: Instant,
    step_rate: RateMeter,
    /// Where the frame's time goes; see [`crate::profile`].
    frame_profile: profile::FrameProfile,
    wall_time_s: f64,
    /// Solver step the current averaging window opened at.
    window_start_step: u64,
    /// When the resolution control's cost line was last worked out. See
    /// [`Running::refresh_resolution_estimate`].
    resolution_refreshed: Instant,

    mouse: MouseState,
    cursor: [f32; 2],
    prev_cursor: [f32; 2],
    mesh_display: MeshDisplay,
    screenshot_counter: u32,

    /// Frames rendered since startup.
    frames: u64,
    /// The swapchain reported itself suboptimal; reconfigure before next frame.
    reconfigure_pending: bool,
    /// When the current run of failed surface acquisitions began. `None` while
    /// the swapchain is healthy.
    acquire_failing_since: Option<Instant>,
    /// When the surface was last reconfigured, so the recovery path retries at a
    /// rate rather than as fast as the event loop spins.
    last_reconfigure: Option<Instant>,
    /// `AERODUCT_FRAMES`: render this many frames, capture the window, exit.
    /// The headless acceptance path — a renderer's real test is looking at it,
    /// and this is how an automated run gets something to look at.
    smoke_frames: Option<u64>,
    smoke_path: std::path::PathBuf,
    exit_requested: bool,
}

impl Running {
    fn new(event_loop: &ActiveEventLoop) -> Result<Self> {
        let attrs = Window::default_attributes()
            .with_title("AeroDuct")
            .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 950.0));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .context("creating the window")?,
        );

        // `ad-gpu` owns the instance, so the surface is created from it rather
        // than the other way round. The adapter is picked without a compatible
        // surface, which on the PRIMARY backends resolves to the same discrete
        // device the surface can present to.
        let gpu = GpuContext::new_blocking(None).context("acquiring a GPU")?;
        log::info!("device: {} ({:?})", gpu.info.name, gpu.info.backend);

        let surface = gpu
            .instance
            .create_surface(window.clone())
            .context("creating the window surface")?;
        let caps = surface.get_capabilities(&gpu.adapter);
        // A *non*-sRGB target: the AgX tonemap in `ad-render` already writes
        // display-encoded values, and an sRGB swapchain would encode them twice.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        // COPY_SRC when the platform allows it, so `AERODUCT_FRAMES` can capture
        // the *composited* window — 3D plus UI — rather than a re-render of the
        // scene without the panels on top. That is the only way an automated run
        // can prove the interface actually drew.
        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }
        let surface_config = wgpu::SurfaceConfiguration {
            usage,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            // Mailbox interactively, for the lowest latency while the user is
            // dragging the camera. **Fifo for an unattended run**: a pinned step
            // count drives frames of the better part of a second, and at that
            // rate the Mailbox swapchain on this driver goes permanently invalid
            // after about a minute — `get_current_texture` returns a validation
            // error that reconfiguring does not clear, and an 80,000-step
            // measurement is lost to a presentation mode. Reproduced with the
            // metrics passes disabled, so it is the swapchain and not the work.
            present_mode: present_mode_override(&caps.present_modes).unwrap_or(
                if headless_capture().is_some() {
                    wgpu::PresentMode::Fifo
                } else {
                    caps.present_modes
                        .iter()
                        .copied()
                        .find(|m| *m == wgpu::PresentMode::Mailbox)
                        .unwrap_or(wgpu::PresentMode::Fifo)
                },
            ),
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &surface_config);
        log::info!(
            "surface: {format:?} {}x{}",
            surface_config.width,
            surface_config.height
        );

        // --- geometry and solver ---
        let params = startup_params();
        let path = std::env::args()
            .nth(1)
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
            .or_else(ad_geom::test_stl_path);
        let mut sim = match path {
            Some(p) => {
                log::info!("loading {}", p.display());
                Sim::from_stl(&gpu, &p, &params)?
            }
            None => anyhow::bail!(
                "no STL to load. Pass one on the command line, or put \
                 \"Airflow redirector - Part 1.stl\" in a parts/ folder next to the executable."
            ),
        };

        // --- renderer ---
        let renderer = Renderer::new(
            &gpu,
            RendererConfig::new(
                surface_config.width,
                surface_config.height,
                format,
                sim.grid,
            ),
        )
        .context("building the renderer")?;
        // Where the part sits in the car. Only the picture, the camera and
        // things placed in the world see it; the lattice stays in the STL's
        // frame. See `ad_ui::pose`.
        let install = ad_ui::InstallPose::from_env().unwrap_or_default();
        let pivot = sim.duct_bbox().center();
        let (mut render_scene, render_roles) = build_render_scene(&gpu, &sim);
        render_scene.model = install.matrix(pivot);
        render_scene.end_frame();
        if !install.is_identity() {
            log::info!(
                "install pose: yaw/pitch/roll {:.1?} deg, offset {:?} mm, about {:?} mm",
                install.euler_deg().to_array(),
                install.offset_mm.to_array(),
                pivot.to_array()
            );
            for (i, m) in sim.mouths.iter().enumerate() {
                log::info!(
                    "mouth {}: air enters along {} in the car",
                    (b'A' + i as u8) as char,
                    ad_ui::pose::describe_direction(install.dir_to_world(m.patch.normal))
                );
            }
        }
        let world_bbox = install.world_bbox(pivot, sim.scene_bbox());

        let mut controller = OrbitController::default();
        controller.frame_bbox(world_bbox, FRAME_MARGIN);
        controller.apply_preset(ViewPreset::ALL[0]);
        controller.frame_bbox(world_bbox, FRAME_MARGIN);
        controller.snap();

        // --- UI ---
        let ui = UiBackend::new(&gpu, &window, format)?;
        let panels = Panels::new()?;
        let mut state = UiState::new(params);
        state.install = install;
        state.install_pivot_mm = pivot;
        state.viewport_rect = [
            0.0,
            0.0,
            surface_config.width as f32,
            surface_config.height as f32,
        ];
        state.ui_scale = window.scale_factor() as f32;
        metrics::publish_mouths(&sim, &mut state);
        publish_domain(&sim, &mut state);
        // A louver aim from the environment is in car terms and the solver was
        // built without it, so it is converted now, against the detected inlet
        // and the pose, and hot-applied before the parameters are adopted.
        if let Some(louver) = ad_ui::pose::louver_from_lookup(|k| std::env::var(k).ok()) {
            state.inlet_louver_deg = louver;
            state.sync_inlet_tilt();
            sim.hot_apply(&state.params)?;
            log::info!(
                "inlet louver {louver:?} deg (car) -> tilt {:?} deg (part frame)",
                state.params.inlet_tilt_deg
            );
        }
        // Adopt the parameters without raising a reset toast on the first frame:
        // nothing has changed yet.
        let _ = state.sync_params();

        let (inlet, outlet) = mouth_roles(&sim, &params);
        let metrics = build_metrics(&gpu, &sim, inlet, outlet);
        let tracers = build_tracers(&sim, inlet, outlet);
        state.status = format!(
            "{} | {} | {}",
            gpu.info.name,
            sim.describe(),
            metrics.provenance()
        );
        log::info!("{}", sim.voxel_report);
        state.toasts.push(
            ad_ui::Toast::new(ad_ui::Severity::Notice, "Simulation running")
                .with_detail(
                    "There is no Run button: edits apply to the live solver. Measured numbers \
                     read '--' until the averaging window has something in it.",
                )
                .with_key("startup"),
        );
        if let Some(n) = pinned_steps() {
            state.tuner.set_manual(n);
            log::info!("steps per frame pinned to {n} by AERODUCT_STEPS");
        }
        // Solver timestamps stay on. The step tuner needs the per-step GPU cost
        // to split the frame budget between the solver and everything else, and
        // two timestamps per batch cost nothing next to the batch. The profiler
        // records per step, so its average holds while the tuner moves the
        // batch length.
        sim.solver.set_profiling(true);
        let frame_profile = profile::FrameProfile::new(&gpu);

        let mut running = Self {
            window,
            gpu,
            surface,
            surface_config,
            renderer,
            render_scene,
            controller,
            sim,
            metrics,
            tracers,
            ui,
            panels,
            state,
            obstructions: Vec::new(),
            committed_placements: Vec::new(),
            committed_pose: install,
            committed_vents: Vec::new(),
            committed_duct: ad_ui::pose::DuctGeometry::IDENTITY,
            logged_pose: install,
            placement_edit: None,
            render_roles,
            last_frame: Instant::now(),
            step_rate: RateMeter::new(0.5),
            frame_profile,
            wall_time_s: 0.0,
            window_start_step: 0,
            resolution_refreshed: Instant::now(),
            mouse: MouseState::default(),
            cursor: [0.0, 0.0],
            prev_cursor: [0.0, 0.0],
            mesh_display: mesh_display_override().unwrap_or(MeshDisplay::Ghost),
            screenshot_counter: 0,
            frames: 0,
            reconfigure_pending: false,
            acquire_failing_since: None,
            last_reconfigure: None,
            smoke_frames: headless_capture(),
            smoke_path: std::env::var("AERODUCT_SHOT")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("aeroduct-window.png")),
            exit_requested: false,
        };
        running.load_startup_placements()?;
        Ok(running)
    }

    // -- events ------------------------------------------------------------

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) -> Result<()> {
        let captured = self.ui.handle_window_event(&self.window, &event);

        match &event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                return Ok(());
            }
            WindowEvent::Resized(size) => {
                self.resize(size.width, size.height);
                return Ok(());
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = self.window.inner_size();
                self.resize(size.width, size.height);
                return Ok(());
            }
            WindowEvent::RedrawRequested => {
                self.frame()?;
                if self.exit_requested {
                    event_loop.exit();
                }
                return Ok(());
            }
            _ => {}
        }

        self.input(&event, captured);
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) {
        // A minimised window reports zero; configuring a surface at zero is a
        // validation error, so the swapchain is simply left alone and the frame
        // loop skips until it comes back.
        if width == 0 || height == 0 {
            self.surface_config.width = 0;
            self.surface_config.height = 0;
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface
            .configure(&self.gpu.device, &self.surface_config);
        self.renderer.resize(width, height);
        self.ui.resized(&self.window);
        self.ui.set_surface_size(width, height);
        self.state.viewport_rect = [0.0, 0.0, width as f32, height as f32];
        self.state.ui_scale = self.window.scale_factor() as f32;
    }

    fn input(&mut self, event: &WindowEvent, captured: bool) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.prev_cursor = self.cursor;
                self.cursor = [position.x as f32, position.y as f32];
                let owner = self.state.pointer_ownership();
                let gesture = self.state.camera_input.gesture(self.mouse, owner);
                if gesture != Gesture::None {
                    let dx = self.cursor[0] - self.prev_cursor[0];
                    let dy = self.cursor[1] - self.prev_cursor[1];
                    let h = self.surface_config.height.max(1) as f32;
                    self.state
                        .camera_input
                        .apply_drag(&mut self.controller, gesture, dx, dy, h);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = *state == ElementState::Pressed;
                match button {
                    winit::event::MouseButton::Left => self.mouse.left = down,
                    winit::event::MouseButton::Middle => self.mouse.middle = down,
                    winit::event::MouseButton::Right => self.mouse.right = down,
                    _ => {}
                }
                // Ctrl+click in the viewport drops a probe.
                if down
                    && *button == winit::event::MouseButton::Left
                    && self.mouse.ctrl
                    && !captured
                    && self.state.pointer_ownership().viewport_has_mouse()
                {
                    if let Some(ndc) = self.state.viewport_ndc(self.cursor) {
                        let (origin, dir) = self.controller.current.ray(ndc);
                        self.state.push_action(UiAction::ProbeFromRay {
                            origin_mm: origin,
                            direction: dir,
                        });
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 30.0,
                };
                let owner = self.state.pointer_ownership();
                self.state
                    .camera_input
                    .apply_wheel(&mut self.controller, notches, owner);
            }
            WindowEvent::ModifiersChanged(m) => {
                let s = m.state();
                self.mouse.ctrl = s.control_key();
                self.mouse.shift = s.shift_key();
                self.mouse.alt = s.alt_key();
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic,
                ..
            } => {
                if *is_synthetic || captured || event.state != ElementState::Pressed {
                    return;
                }
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                self.shortcut(code);
            }
            _ => {}
        }
    }

    fn shortcut(&mut self, code: KeyCode) {
        let digit = match code {
            KeyCode::Digit1 => Some(1),
            KeyCode::Digit2 => Some(2),
            KeyCode::Digit3 => Some(3),
            KeyCode::Digit4 => Some(4),
            KeyCode::Digit5 => Some(5),
            KeyCode::Digit6 => Some(6),
            KeyCode::Digit7 => Some(7),
            _ => None,
        };
        if let Some(d) = digit.and_then(ad_ui::camera_input::preset_for_digit) {
            self.state.push_action(UiAction::ApplyViewPreset(d));
            return;
        }
        match code {
            KeyCode::KeyF => self.state.push_action(UiAction::FrameScene),
            KeyCode::Space => {
                self.state.playing = !self.state.playing;
            }
            KeyCode::KeyG => {
                self.mesh_display = self.mesh_display.cycle();
            }
            KeyCode::F12 => self.state.push_action(UiAction::Screenshot),
            _ => {}
        }
    }

    // -- the frame ---------------------------------------------------------

    fn frame(&mut self) -> Result<()> {
        if self.surface_config.width == 0 || self.surface_config.height == 0 {
            return Ok(()); // minimised
        }

        if std::mem::take(&mut self.reconfigure_pending) {
            self.surface
                .configure(&self.gpu.device, &self.surface_config);
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().clamp(1.0e-4, 0.25);
        self.last_frame = now;
        let mut laps = profile::Laps::start(now);

        if self.smoke_frames.is_some() && self.frames == 2 {
            self.smoke_script();
        }
        if self.smoke_frames.is_some() {
            self.smoke_resolution_step();
        }
        self.handle_actions()?;
        self.settle_placements()?;
        self.refresh_resolution_estimate();

        // Parameter changes. A rebuild is attempted *before* the change is
        // committed, because it can fail: the GPU can refuse the new lattice.
        // `rebuild` then leaves the old solver running and puts the parameters
        // back, so `sync_params` finds nothing to announce — no reset toast for
        // a change that never took effect.
        //
        // The louver aim is in car terms and the solver takes it relative to the
        // part, which the pose and the inlet choice both change; turning one
        // into the other here makes it an ordinary parameter change below.
        self.state.sync_inlet_tilt();
        if self.state.peek_params().rebuild {
            self.rebuild()?;
        }
        // The single place a statistics reset is detected; `sync_params` raises
        // the toast itself. A rebuild that went through reports `rebuild` here,
        // which subsumes the hot apply.
        let change = self.state.sync_params();
        if change.hot_apply {
            self.sim.hot_apply(&self.state.params)?;
            if change.cause == Some(ad_ui::view::ResetCause::InletAngle)
                && self.state.params.inlet_tilt_deg != [0.0, 0.0]
                && self.sim.domain.plenum.is_some()
            {
                self.state.toasts.push(
                    ad_ui::Toast::new(ad_ui::Severity::Notice, "Louver angle mostly lost")
                        .with_detail(
                            "The plenum domain puts the inlet at the end of a walled extension, \
                             which straightens the air before it reaches the mouth. The room \
                             domain applies the angle at the mouth itself.",
                        )
                        .with_key("tilt-plenum"),
                );
            }
        }
        if change.reset_statistics {
            // `sync_params` has already toasted this one, so the adapter is told
            // the cause and swallows the notice its own monitor will raise.
            self.metrics
                .reset(change.cause.unwrap_or(ad_ui::view::ResetCause::Manual));
            self.window_start_step = self.sim.solver.steps_taken();
        }

        self.controller.update(dt);
        self.state.toasts.update(dt);
        laps.lap(Phase::Update);

        // Step the solver, and the macroscopic fields in the same submit: only
        // when there is something new to derive.
        let steps = if self.state.playing {
            self.state.tuner.steps()
        } else {
            0
        };
        if steps > 0 {
            self.sim.solver.step_and_compute_macroscopic(steps);
            self.wall_time_s += dt as f64;
        }
        self.step_rate.tick(steps as f64, dt as f64);
        let stepped = steps > 0;
        laps.lap(Phase::Solver);

        // Tracers, before the metrics sample so a fresh set of exits shows up in
        // the same frame's view. `update` is a no-op on all but one frame in
        // several hundred; when it does fire it captures the velocity field,
        // which is why it needs `&mut Sim` and cannot live inside `sample`.
        if let Some(t) = self.tracers.as_mut().filter(|_| stepped) {
            if let Some((exits, trapped)) = t.update(&mut self.sim, dt) {
                self.metrics.record_ages(&exits, trapped);
            }
        }

        // Metrics -> the view models the panels read.
        let (inlet, outlet) = mouth_roles(&self.sim, &self.state.params);
        let cx = SampleContext {
            sim: &self.sim,
            step: self.sim.solver.steps_taken(),
            steps_in_window: self
                .sim
                .solver
                .steps_taken()
                .saturating_sub(self.window_start_step),
            sim_time_s: self.sim.solver.sim_time_seconds(),
            inlet_velocity_ms: self.state.params.inlet_velocity_ms,
            inlet,
            outlet,
            stepped,
        };
        self.metrics.sample(&cx, &mut self.state.metrics);
        // A reset the UI did not raise itself. Averaging across a parameter
        // change is the one bug that produces a confidently wrong number, so a
        // clear that happens quietly is not allowed to stay quiet.
        if self.metrics.take_unannounced_reset() {
            self.state.toasts.push(
                ad_ui::Toast::new(ad_ui::Severity::Notice, "Statistics reset")
                    .with_detail(
                        "the measurement configuration moved - averages and error bars start \
                         again from zero",
                    )
                    .with_key("stats-reset"),
            );
            self.window_start_step = self.sim.solver.steps_taken();
        }
        self.state.refresh_deltas();
        self.update_clock(steps);
        laps.lap(Phase::Metrics);

        // --- render ---
        use wgpu::CurrentSurfaceTexture as Acquired;
        let surface_texture = match self.surface.get_current_texture() {
            Acquired::Success(t) => t,
            // Suboptimal still presents. Reconfigure *after* this frame rather
            // than now: the texture is already acquired, and reconfiguring a
            // surface with a live frame outstanding is not something wgpu
            // promises anything about.
            Acquired::Suboptimal(t) => {
                self.reconfigure_pending = true;
                t
            }
            Acquired::Outdated | Acquired::Lost => {
                self.surface
                    .configure(&self.gpu.device, &self.surface_config);
                return Ok(());
            }
            Acquired::Timeout | Acquired::Occluded => return Ok(()),
            // Recoverable, up to a point. A long unattended run drives frames
            // that take the best part of a second each, and Windows will
            // invalidate the swapchain of a window it considers unresponsive;
            // reconfiguring gets it back. Bailing on the first one aborted an
            // 80,000-step verification run three minutes in, which is a
            // measurement lost to a presentation problem. It is still a real
            // error if it will not clear, so a run of them in a row is fatal.
            Acquired::Validation => {
                // Counted in *seconds*, not in frames. `about_to_wait` requests
                // another redraw immediately, so a frame-counted budget is spent
                // inside a millisecond and never gives the window state time to
                // settle — which is exactly how a recoverable hiccup became a
                // fatal one on the first attempt at this.
                let now = Instant::now();
                let since = *self.acquire_failing_since.get_or_insert(now);
                anyhow::ensure!(
                    now - since < ACQUIRE_RECOVERY_LIMIT,
                    "surface acquisition has raised a validation error for {:.0?} \
                     despite reconfiguring",
                    now - since
                );
                // ...and reconfigure at a *rate*, not once per frame. The retry
                // loop runs as fast as the event loop will spin, and thousands of
                // `configure` calls a second is itself a way to upset the
                // swapchain; one every quarter second is plenty to recover from a
                // hiccup and cheap enough to be free.
                if self
                    .last_reconfigure
                    .is_none_or(|t| now - t >= RECONFIGURE_INTERVAL)
                {
                    self.last_reconfigure = Some(now);
                    log::warn!("surface acquisition failed validation; reconfiguring");
                    self.surface
                        .configure(&self.gpu.device, &self.surface_config);
                }
                return Ok(());
            }
        };
        if let Some(since) = self.acquire_failing_since.take() {
            log::warn!(
                "surface acquisition recovered after {:.0?} of validation errors",
                since.elapsed()
            );
        }
        let target = surface_texture.texture.create_view(&Default::default());
        laps.lap(Phase::Acquire);

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        self.frame_profile.gpu.begin_frame();

        let velocity_view = self.sim.solver.velocity_view();
        let scales = self.sim.derive_scales();
        self.render_scene.model = self.state.install_matrix();
        let previews = self.obstruction_previews();
        let duct_preview = self.duct_preview();
        for (m, role) in self.render_scene.meshes.iter_mut().zip(&self.render_roles) {
            match *role {
                None => {
                    m.visible = self.state.layers.is_drawn(ad_ui::LayerKind::Duct);
                    m.model = duct_preview;
                }
                Some(i) => {
                    m.visible = self.state.layers.is_drawn(ad_ui::LayerKind::Obstruction(i));
                    m.model = previews.get(i).copied().unwrap_or(Mat4::IDENTITY);
                }
            }
        }
        self.renderer.set_mesh_display(self.mesh_display);
        self.renderer.set_field(self.state.field);
        let span = self.frame_profile.gpu.span_begin(&mut encoder, "render");
        let rendered = self.renderer.render(
            &mut encoder,
            FrameInput {
                camera: &self.controller.current,
                scene: &self.render_scene,
                sources: stepped.then(|| FieldSources {
                    macro_view: &velocity_view,
                    flags_view: Some(&self.sim.flags_view),
                    scales,
                }),
                sdf: None,
                target: &target,
                dt,
            },
        );
        self.frame_profile.gpu.span_end(&mut encoder, span);
        rendered?;
        laps.lap(Phase::Render);

        // --- UI over the top ---
        let extent = dear_extent(&surface_texture.texture);
        let extremes = heuristic_field_range(
            self.state.field,
            self.state.params.inlet_velocity_ms,
            self.sim.d_h_mm as f32,
            self.state.params.rho as f32,
        );
        let step = self.sim.solver.steps_taken();
        let grid = self.sim.grid;
        // The panels edit a *copy* of the transfer function and it is written
        // back only if it actually changed. `Renderer::transfer_function_mut`
        // marks the LUT dirty and resets the progressive accumulator on every
        // call, so handing the panels the live one would re-bake the LUT sixty
        // times a second and stop the still-image accumulation from ever
        // building up — the picture would never resolve while nothing moved.
        let mut transfer = self.renderer.transfer_function().clone();
        let cursor = {
            let frame = self.ui.frame(Some(dt));
            let mut pcx = PanelContext {
                state: &mut self.state,
                transfer: &mut transfer,
                camera: &self.controller.current,
                mesh_display: &mut self.mesh_display,
                step,
                grid,
                field_extremes: Some(extremes),
            };
            self.panels.build(&frame, &mut pcx);
            frame.cursor()
        };
        if transfer != *self.renderer.transfer_function() {
            *self.renderer.transfer_function_mut() = transfer;
        }
        self.ui.apply_cursor(&self.window, cursor);
        self.ui.set_surface_size(extent.0, extent.1);
        let span = self.frame_profile.gpu.span_begin(&mut encoder, "ui");
        let drawn = self.ui.render(&mut encoder, &target, extent.0, extent.1);
        self.frame_profile.gpu.span_end(&mut encoder, span);
        drawn?;
        self.frame_profile.gpu.resolve(&mut encoder);
        self.state.ui_capture_mouse = self.ui.wants_mouse();
        laps.lap(Phase::Ui);

        self.frames += 1;
        let capture = self.smoke_frames == Some(self.frames)
            && self
                .surface_config
                .usage
                .contains(wgpu::TextureUsages::COPY_SRC);
        if capture {
            let (w, h) = (self.surface_config.width, self.surface_config.height);
            let raw = ad_render::util::readback_rgba8(
                &self.gpu.device,
                &self.gpu.queue,
                &surface_texture.texture,
                w,
                h,
                encoder,
            );
            let rgba = to_rgba(raw, self.surface_config.format);
            png::write_rgba(&self.smoke_path, w, h, &rgba)?;
            log::info!(
                "captured the composited window to {}",
                self.smoke_path.display()
            );
            if tracers::rtd_enabled() {
                let (i, o) = mouth_roles(&self.sim, &self.state.params);
                tracers::probe_mouth(&mut self.sim, i, "inlet");
                tracers::probe_mouth(&mut self.sim, o, "outlet");
            }
            log::info!(
                "metrics at step {} ({} flow-throughs):\n{}",
                self.sim.solver.steps_taken(),
                format_args!("{:.1}", self.state.metrics.stats.flow_throughs),
                self.metrics.summary()
            );
            // The cost side of the same run, from GPU timestamps around the
            // stream-collide pass alone. Quoted next to the physics on purpose:
            // a domain change moves both, and a cell count without a measured
            // ms/step is an argument about arithmetic rather than about speed.
            if let Some(ms) = self.sim.solver.ms_per_step() {
                log::info!(
                    "throughput: {ms:.3} ms/step over {} cells | {}",
                    self.sim.grid.cell_count(),
                    self.sim.solver.profiler_summary(),
                );
            }
        } else {
            self.gpu.queue.submit([encoder.finish()]);
        }
        self.gpu.queue.present(surface_texture);
        if self.smoke_frames.is_some_and(|n| self.frames >= n) {
            self.exit_requested = true;
        }
        self.renderer.after_submit();
        self.frame_profile.gpu.collect(&self.gpu.device);
        self.render_scene.end_frame();
        laps.lap(Phase::Present);
        let window_closed = self.frame_profile.end_frame(dt, &laps, steps);
        self.state.profiling =
            self.frame_profile
                .report(&self.sim.solver, &self.renderer, self.metrics.as_ref());
        // An unattended run logs the breakdown once per window, so profiling a
        // configuration needs nothing but the log.
        if self.smoke_frames.is_some() && window_closed {
            log::info!(
                "frame profile at frame {}:\n  {}",
                self.frames,
                self.state.profiling.join("\n  ")
            );
        }

        // Feed the whole measured frame back to the auto-tuner, with the steps
        // this frame actually ran (none while paused) and the GPU's per-step
        // cost. `dt` is the interval between redraws, which is the number the
        // user experiences: the CPU-side frame time would miss the GPU
        // backpressure that a big step batch actually creates. The step cost
        // lets the tuner split that interval into what the step count buys and
        // what it does not; see `ad_ui::StepAutoTuner::update_measured`.
        let step_cost = self
            .sim
            .solver
            .ms_per_step()
            .map(|ms| std::time::Duration::from_secs_f64(ms * 1e-3));
        self.state
            .tuner
            .update_measured(std::time::Duration::from_secs_f32(dt), steps, step_cost);
        Ok(())
    }

    /// Reach the panels that only appear after a click, so an automated run
    /// exercises them.
    ///
    /// The gizmo in particular is a foreign-function call into ImGuizmo that no
    /// unit test can reach: it only runs when a placeable layer is selected, and
    /// "the app crashes the first time you select a slice" is exactly the class
    /// of bug a headless smoke run should be catching.
    fn smoke_script(&mut self) {
        self.state.push_action(UiAction::AddSlice);
        let centre = self.sim.duct_bbox().center();
        self.state.probes.add(centre);
        log::info!("smoke script: added a slice and a probe");
        if let Some(preset) = view_override() {
            self.frame_domain(preset);
        }
    }

    /// `AERODUCT_SET_DX`: drive the resolution control headlessly. One cell
    /// size, or a comma-separated list applied 150 frames apart — the list is
    /// how repeated changes are checked for handing their VRAM back. Each
    /// sends the same action the Apply button does, so the pre-flight, the
    /// rebuild and the rollback all run exactly as they would for a click.
    fn smoke_resolution_step(&mut self) {
        const EVERY: u64 = 150;
        let Some(since) = self.frames.checked_sub(2).filter(|f| f % EVERY == 0) else {
            return;
        };
        let Ok(list) = std::env::var("AERODUCT_SET_DX") else {
            return;
        };
        let dx = list
            .split(',')
            .filter_map(|v| v.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .nth((since / EVERY) as usize);
        if let Some(dx) = dx {
            log::info!("smoke script: requesting dx = {dx} mm");
            self.state.resolution.pending_dx_mm = dx;
            self.state.push_action(UiAction::SetResolution(dx));
        }
    }

    /// Point the camera at the whole *domain* rather than at the part.
    ///
    /// The default framing is [`Sim::scene_bbox`], which is the right shot for
    /// looking at a duct and the wrong one for looking at a domain: the box now
    /// reaches several outlet diameters past the part on the downstream face,
    /// and that is precisely the region the scene framing crops away. The one
    /// question a margin change has to answer — is the exit jet inside the box,
    /// or is it clipped at a face? — is only decidable when the faces are in
    /// shot, and the volume render stops at them, so framing the lattice puts
    /// the jet and its downstream boundary in the same picture.
    ///
    /// `LayerKind::GridOutline` would draw the edges explicitly and is not
    /// turned on here, because nothing renders it yet: it is a toggle in the
    /// layer stack with no pass behind it.
    fn frame_domain(&mut self, preset: ViewPreset) {
        self.controller.apply_preset(preset);
        // A wider margin than [`FRAME_MARGIN`], because the panels are drawn
        // *over* the viewport rather than beside it: the docked stacks cover
        // roughly the outer 40% of the window, so a box framed to the viewport
        // has its downstream face behind the plots. The margin that matters for
        // this shot is the one to the panel edge, not to the window edge.
        let domain = self
            .state
            .install
            .world_bbox(self.state.install_pivot_mm, self.sim.grid.bbox());
        self.controller.frame_bbox(domain, DOMAIN_FRAME_MARGIN);
        self.controller.snap();
        log::info!("smoke script: framed the domain from {}", preset.label());
    }

    /// The scene's box in the world, i.e. through the install pose: what every
    /// "frame the part" shot aims at. Obstructions included, so a part parked
    /// beside the duct is in shot too.
    fn world_scene_bbox(&self) -> ad_gpu::Bbox {
        self.state
            .install
            .world_bbox(self.state.install_pivot_mm, self.sim.scene_bbox())
    }

    // -- obstructions --------------------------------------------------------
    //
    // An obstruction is a car part, so it is placed in the car (world) frame
    // and carried into the duct's lattice frame through the install pose. Every
    // change — a load, a removal, a drag, a new pose — is a full transactional
    // rebuild: seconds, and the flow restarts. So a drag previews on screen and
    // is voxelised once it has settled, not on every mouse move.

    /// The duct plus every obstruction, the obstructions carried from the car
    /// into the duct's lattice frame through the install pose.
    fn assemble_scene(
        &self,
        duct: ad_geom::MeshInstance,
        assets: &[ObstructionAsset],
        placements: &[ad_ui::Placement],
    ) -> ad_geom::Scene {
        let pivot = duct.world_bbox().center();
        let mut scene = ad_geom::Scene::new();
        scene.add(
            duct.name,
            duct.asset,
            duct.transform,
            ad_geom::MeshRole::Duct,
        );
        for (o, p) in assets.iter().zip(placements) {
            let t = self.state.install.placement_to_lattice(pivot, *p);
            scene.add(
                o.name.clone(),
                o.asset.clone(),
                t,
                ad_geom::MeshRole::Obstruction,
            );
        }
        scene
    }

    /// Rebuild the lattice with `assets` at `placements` and `vents` in the
    /// room, keeping the running duct. Transactional like every rebuild: on
    /// failure the old setup keeps running and the edit is undone. Returns
    /// whether it went in.
    fn commit_placements(
        &mut self,
        assets: Vec<ObstructionAsset>,
        placements: Vec<ad_ui::Placement>,
        vents: Vec<ad_ui::VentSettings>,
    ) -> Result<bool> {
        let mut duct = self.sim.scene.instance(0).clone();
        // The part's own scale and turns, about its centre, which is where the
        // install pose pivots: the pivot does not move.
        let geometry = self.state.duct;
        duct.transform = geometry.to_geom(duct.asset.local_bbox.center());
        let scene = self.assemble_scene(duct, &assets, &placements);
        let lattice = lattice_vents(&self.state.install, self.state.install_pivot_mm, &vents);
        // The parameters the running solver has, not pending edits to them: a
        // resolution change in flight gets its own rebuild, and taking it here
        // as well would build it twice.
        let params = self.state.committed_params();
        let started = Instant::now();
        match self.replace_sim(|gpu| Sim::from_scene(gpu, scene, lattice, &params)) {
            Ok(()) => {
                log::info!(
                    "{} obstruction(s), {} vent(s) voxelised in {:.1} s: {}",
                    assets.len(),
                    vents.len(),
                    started.elapsed().as_secs_f64(),
                    self.sim.describe()
                );
                self.obstructions = assets;
                self.state.obstructions = placements.clone();
                self.committed_placements = placements;
                self.state.vents = vents.clone();
                self.committed_vents = vents;
                self.committed_duct = geometry;
                self.committed_pose = self.state.install;
                // A rebuild no parameter change announced, so the statistics
                // reset is raised here, as `UiState::sync_params` would.
                self.state.toasts.push(ad_ui::Toast::statistics_reset(
                    ad_ui::view::ResetCause::GeometryChanged,
                ));
                self.state.probes.clear_history();
                let grid = self.sim.grid.bbox();
                for inst in self
                    .sim
                    .scene
                    .instances()
                    .iter()
                    .filter(|i| i.role == ad_geom::MeshRole::Obstruction)
                {
                    let b = inst.world_bbox();
                    if b.min.cmplt(grid.min).any() || b.max.cmpgt(grid.max).any() {
                        self.state.toasts.push(
                            ad_ui::Toast::new(
                                ad_ui::Severity::Notice,
                                "Obstruction cut at the domain edge",
                            )
                            .with_detail(format!(
                                "The part of {:?} outside the simulated box is left out; the box \
                                     is sized from the duct.",
                                inst.name
                            ))
                            .with_key("obstruction-clipped"),
                        );
                    }
                }
                Ok(true)
            }
            Err(e) => {
                if let Some(why) = self.gpu.lost() {
                    return Err(e.context(format!(
                        "the GPU device was lost placing an obstruction ({why})"
                    )));
                }
                log::error!("placing the obstructions/vents failed; the previous setup keeps running: {e:#}");
                self.state.obstructions = self.committed_placements.clone();
                self.state.vents = self.committed_vents.clone();
                self.state.duct = self.committed_duct;
                let names: Vec<String> = self
                    .committed_vents
                    .iter()
                    .map(|v| v.name.clone())
                    .collect();
                self.state
                    .layers
                    .reset_vents(names.iter().map(String::as_str));
                self.state.install = self.committed_pose;
                self.state.toasts.push(
                    ad_ui::Toast::new(ad_ui::Severity::Warning, "Placement not applied")
                        .with_detail(brief(&e))
                        .with_key("placement-failed"),
                );
                Ok(false)
            }
        }
    }

    /// Load an obstruction STL and voxelise it in. It starts at its own
    /// coordinates in the duct's frame — where a part exported from the same
    /// CAD assembly belongs — moved by `offset_mm` in the car.
    fn add_obstruction(&mut self, path: &std::path::Path, offset_mm: Vec3) -> Result<()> {
        let Some((asset, placement)) = self.load_obstruction(path, offset_mm) else {
            return Ok(());
        };
        let name = asset.name.clone();
        let mut assets = self.obstructions.clone();
        assets.push(asset);
        let mut placements = self.state.obstructions.clone();
        placements.push(placement);
        if self.commit_placements(assets, placements, self.state.vents.clone())? {
            let id = self.state.layers.push(
                ad_ui::LayerKind::Obstruction(self.obstructions.len() - 1),
                name,
            );
            self.state.layers.select(Some(id));
        }
        Ok(())
    }

    /// A new vent, standing 30 mm in front of the inlet mouth and facing it,
    /// the size of the mouth: where a car's vent is relative to a duct that
    /// fits over it. Each extra one stands a little further out so they do
    /// not coincide.
    fn default_vent(&self, i: usize) -> ad_ui::VentSettings {
        let inlet = self
            .state
            .params
            .inlet_mouth
            .min(self.sim.mouths.len().saturating_sub(1));
        // The first vent sits on the inlet mouth, sealed to it: a duct fitted
        // over a car vent. Each further one stands off a little so they do not
        // share cells.
        self.vent_at_mouth(inlet, 20.0 * i as f32, format!("Vent {}", i + 1))
    }

    /// A vent on mouth `m`, the mouth's size, blowing into it, `standoff` mm
    /// out from its plane. Car frame.
    fn vent_at_mouth(&self, m: usize, standoff: f32, name: String) -> ad_ui::VentSettings {
        let s = &self.state;
        let Some(m) = self.sim.mouths.get(m) else {
            return ad_ui::VentSettings::new(name, ad_ui::Placement::default(), 100.0, 20.0);
        };
        let n = s.install.dir_to_world(m.patch.normal.normalize_or(Vec3::Z));
        let u = s.install.dir_to_world(m.patch.half_u.normalize_or(Vec3::X));
        // A right-handed frame with +Z the way the vent blows and +X along the
        // mouth's width, so the rectangle lies the way the slot does.
        let y = n.cross(u).normalize_or(Vec3::Y);
        let x = y.cross(n).normalize_or(Vec3::X);
        let rotation = Quat::from_mat3(&Mat3::from_cols(x, y, n)).normalize();
        let centre = s.to_world(m.patch.center_mm) - n * standoff;
        ad_ui::VentSettings::new(
            name,
            ad_ui::Placement {
                translation_mm: centre,
                rotation,
                scale: 1.0,
            },
            2.0 * m.patch.half_u.length(),
            2.0 * m.patch.half_v.length(),
        )
    }

    /// Ctrl+click on a mouth: the selected vent moves onto it, or a new one is
    /// made there. Sealed to the mouth, the size of the mouth.
    fn place_vent_at_mouth(&mut self, m: usize) {
        let letter = (b'A' + m as u8) as char;
        let selected = self.state.layers.selected().and_then(|l| match l.kind {
            ad_ui::LayerKind::Vent(i) if i < self.state.vents.len() => Some(i),
            _ => None,
        });
        match selected {
            Some(i) => {
                let name = self.state.vents[i].name.clone();
                let mut v = self.vent_at_mouth(m, 0.0, name);
                v.speed_scale = self.state.vents[i].speed_scale;
                v.aim_deg = self.state.vents[i].aim_deg;
                self.state.vents[i] = v;
                log::info!("vent {} moved onto mouth {letter}", i + 1);
            }
            None if self.state.vents.len() >= sim::MAX_VENTS => {
                self.state.toasts.push(
                    ad_ui::Toast::new(ad_ui::Severity::Info, "Three vents at most")
                        .with_detail(
                            "Select a vent layer and ctrl+click a mouth to move it there instead.",
                        )
                        .with_key("vent-slots"),
                );
                return;
            }
            None => {
                let i = self.state.vents.len();
                let v = self.vent_at_mouth(m, 0.0, format!("Vent {}", i + 1));
                let name = v.name.clone();
                self.state.vents.push(v);
                let id = self.state.layers.push(ad_ui::LayerKind::Vent(i), name);
                self.state.layers.select(Some(id));
                log::info!("vent {} placed on mouth {letter}", i + 1);
            }
        }
        self.state.toasts.push(
            ad_ui::Toast::new(
                ad_ui::Severity::Info,
                format!("Air source on mouth {letter}"),
            )
            .with_detail("Sealed to the opening. Drag it off with the gizmo for a free jet.")
            .with_key("vent-placed"),
        );
    }

    /// Turn the part itself by the quarter turns nearest the install pose and
    /// take them out of the pose: the picture stays as it is, the mouths move
    /// to other lattice faces, and the lattice is rebuilt. Probes and slices
    /// live on the part, so they turn with it.
    fn bake_pose(&mut self) {
        let turn = ad_ui::pose::nearest_quarter_turn(self.state.install.rotation);
        if turn == Quat::IDENTITY {
            self.state.toasts.push(
                ad_ui::Toast::new(ad_ui::Severity::Info, "Nothing to bake")
                    .with_detail(
                        "The pose is within 45 degrees of the part's own axes on every axis.",
                    )
                    .with_key("bake"),
            );
            return;
        }
        let c = self.state.install_pivot_mm;
        self.state.duct.turn = ad_ui::pose::nearest_quarter_turn(turn * self.state.duct.turn);
        self.state.install.rotation = (self.state.install.rotation * turn.conjugate()).normalize();
        for p in self.state.probes.items_mut() {
            p.position_mm = turn * (p.position_mm - c) + c;
        }
        for sl in &mut self.state.slices {
            sl.origin_mm = turn * (sl.origin_mm - c) + c;
            sl.normal = (turn * sl.normal).normalize_or(Vec3::Y);
        }
        log::info!(
            "baking a quarter turn into the part: turn {:?}, pose now yaw/pitch/roll {:.1?} deg",
            turn,
            self.state.install.euler_deg().to_array()
        );
    }

    /// The lattice-frame move from the part as built to the part as the user
    /// has it: what makes a scale drag show before the rebuild.
    fn duct_preview(&self) -> Mat4 {
        if self.state.duct == self.committed_duct {
            return Mat4::IDENTITY;
        }
        let inst = self.sim.scene.instance(0);
        let now = self.state.duct.to_geom(inst.asset.local_bbox.center());
        transform_matrix(now) * transform_matrix(inst.transform).inverse()
    }

    /// Read an obstruction STL and place it: at its own coordinates in the
    /// duct's frame, moved by `offset_mm` in the car. `None`, with a toast, if
    /// the file will not load.
    fn load_obstruction(
        &mut self,
        path: &std::path::Path,
        offset_mm: Vec3,
    ) -> Option<(ObstructionAsset, ad_ui::Placement)> {
        let load = match ad_geom::load_stl(path) {
            Ok(l) => l,
            Err(e) => {
                log::error!("loading obstruction {}: {e:#}", path.display());
                self.state.toasts.push(
                    ad_ui::Toast::new(ad_ui::Severity::Critical, "Could not load STL")
                        .with_detail(format!("{path:?}: {e}")),
                );
                return None;
            }
        };
        log::info!(
            "obstruction {}: {} triangles, {}",
            path.display(),
            load.mesh.triangle_count(),
            load.health().report()
        );
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "obstruction".into());
        let mut placement = self
            .state
            .install
            .lattice_to_placement(self.state.install_pivot_mm, ad_geom::Transform::IDENTITY);
        placement.translation_mm += offset_mm;
        Some((
            ObstructionAsset {
                name,
                asset: ad_geom::MeshAsset::new(load.mesh),
            },
            placement,
        ))
    }

    /// Take obstruction `i` out. Its layer went when the button was pressed,
    /// so a removal that cannot be applied puts the layer back.
    fn remove_obstruction(&mut self, i: usize) -> Result<()> {
        if i >= self.obstructions.len() {
            return Ok(());
        }
        let mut assets = self.obstructions.clone();
        let removed = assets.remove(i);
        let mut placements = self.state.obstructions.clone();
        if i < placements.len() {
            placements.remove(i);
        }
        if !self.commit_placements(assets, placements, self.state.vents.clone())? {
            self.state.layers.insert_obstruction(i, removed.name);
        }
        Ok(())
    }

    /// Voxelise a placement edit once it has settled: the mouse is up and
    /// nothing has moved for [`OBSTRUCTION_SETTLE`]. Obstructions, vents, and
    /// the install pose while either exists, since they stay put in the car
    /// while the duct turns.
    ///
    /// A vent whose *air* changed but whose cells did not — its speed or aim —
    /// is a uniform write, applied at once through the parameter watcher so
    /// the statistics reset like any other operating-point change.
    fn settle_placements(&mut self) -> Result<()> {
        let pose = self.state.install;
        let pivot = self.state.install_pivot_mm;
        let wanted = lattice_vents(&pose, pivot, &self.state.vents);
        let cells_same = wanted.len() == self.sim.vents.len()
            && wanted
                .iter()
                .zip(&self.sim.vents)
                .all(|(a, b)| a.same_cells(b));
        if cells_same {
            let mut air_changed = false;
            for (have, want) in self.sim.vents.iter_mut().zip(&wanted) {
                if have.direction != want.direction || have.speed_scale != want.speed_scale {
                    have.direction = want.direction;
                    have.speed_scale = want.speed_scale;
                    air_changed = true;
                }
            }
            if air_changed {
                self.state.params.vent_generation =
                    self.state.params.vent_generation.wrapping_add(1);
            }
        }

        let obstructed = !self.obstructions.is_empty();
        if !obstructed && self.state.vents.is_empty() {
            // Nothing placed, so every pose is already "committed". Kept
            // current so a failed first add cannot roll the pose back to one
            // from before the user last turned the part.
            self.committed_pose = pose;
        }
        if pose != self.logged_pose && !self.mouse.left {
            self.logged_pose = pose;
            log::info!(
                "install pose now yaw/pitch/roll {:.1?} deg, offset {:.1?} mm",
                pose.euler_deg().to_array(),
                pose.offset_mm.to_array()
            );
        }
        let placements = &self.state.obstructions;
        let vents = &self.state.vents;
        let duct = self.state.duct;
        let pending = *placements != self.committed_placements
            || !cells_same
            || duct != self.committed_duct
            || (obstructed && pose != self.committed_pose);
        if !pending {
            self.placement_edit = None;
            return Ok(());
        }
        let now = Instant::now();
        match &self.placement_edit {
            Some(e)
                if e.placements == *placements
                    && e.vents == *vents
                    && e.pose == pose
                    && e.duct == duct =>
            {
                if !self.mouse.left && now.duration_since(e.at) >= OBSTRUCTION_SETTLE {
                    self.placement_edit = None;
                    self.commit_placements(
                        self.obstructions.clone(),
                        self.state.obstructions.clone(),
                        self.state.vents.clone(),
                    )?;
                }
            }
            _ => {
                self.placement_edit = Some(PlacementEdit {
                    at: now,
                    placements: placements.clone(),
                    vents: vents.clone(),
                    pose,
                    duct,
                })
            }
        }
        Ok(())
    }

    /// For each obstruction, the lattice-frame move from where the running
    /// lattice has it to where the user has it now. Identity until an edit is
    /// pending; what makes the mesh follow the gizmo before the rebuild.
    fn obstruction_previews(&self) -> Vec<Mat4> {
        let pivot = self.state.install_pivot_mm;
        self.sim
            .scene
            .instances()
            .iter()
            .filter(|i| i.role == ad_geom::MeshRole::Obstruction)
            .zip(&self.state.obstructions)
            .map(|(inst, p)| {
                let now = self.state.install.placement_to_lattice(pivot, *p);
                if now == inst.transform {
                    Mat4::IDENTITY
                } else {
                    transform_matrix(now) * transform_matrix(inst.transform).inverse()
                }
            })
            .collect()
    }

    /// `AERODUCT_OBSTRUCTION` and `AERODUCT_VENT`, for headless runs. Every
    /// one is loaded first and the lattice rebuilt once, rather than once per
    /// item.
    fn load_startup_placements(&mut self) -> Result<()> {
        let first = self.obstructions.len();
        let mut assets = self.obstructions.clone();
        let mut placements = self.state.obstructions.clone();
        if let Ok(spec) = std::env::var("AERODUCT_OBSTRUCTION") {
            for (path, offset) in parse_obstructions(&spec) {
                log::info!(
                    "loading obstruction {} offset {:?} mm (AERODUCT_OBSTRUCTION)",
                    path.display(),
                    offset
                );
                if let Some((asset, placement)) = self.load_obstruction(&path, offset) {
                    assets.push(asset);
                    placements.push(placement);
                }
            }
        }
        let mut vents = self.state.vents.clone();
        if let Ok(spec) = std::env::var("AERODUCT_VENT") {
            for v in parse_vents(&spec) {
                log::info!(
                    "vent {:?} at {:?} mm facing {:?}, {} x {} mm, {}x U (AERODUCT_VENT)",
                    v.name,
                    v.placement.translation_mm.to_array(),
                    v.normal().to_array(),
                    v.width_mm,
                    v.height_mm,
                    v.speed_scale
                );
                vents.push(v);
            }
        }
        let obstruction_names: Vec<String> =
            assets[first..].iter().map(|a| a.name.clone()).collect();
        let vent_names: Vec<String> = vents.iter().map(|v| v.name.clone()).collect();
        if obstruction_names.is_empty() && vents.len() == self.state.vents.len() {
            return Ok(());
        }
        if self.commit_placements(assets, placements, vents)? {
            for (k, name) in obstruction_names.into_iter().enumerate() {
                self.state
                    .layers
                    .push(ad_ui::LayerKind::Obstruction(first + k), name);
            }
            self.state
                .layers
                .reset_vents(vent_names.iter().map(String::as_str));
        }
        Ok(())
    }

    /// Pre-flight a change of the simulated box and, if it fits, make it.
    /// Like [`Running::request_resolution`]: only `SimParams::domain_mm` is
    /// written, and the rebuild follows through `sync_params`.
    fn request_domain(&mut self, mm: Option<[f32; 6]>) {
        let bits = |m: Option<[f32; 6]>| m.map(|m| m.map(f32::to_bits));
        if bits(mm) == bits(self.state.params.domain_mm) {
            return;
        }
        let dx = self.state.params.dx_mm;
        let estimate = self.estimate_resolution(dx, mm);
        let blocker = estimate.blocker.clone();
        let what = match mm {
            Some(m) => format!("box margins {m:?} mm"),
            None => "the automatic box".to_string(),
        };
        match blocker {
            Some(why) if !preflight_disabled() => {
                log::warn!("{what} refused: {why}");
                self.state.toasts.push(
                    ad_ui::Toast::new(ad_ui::Severity::Warning, "Box not applied")
                        .with_detail(why)
                        .with_key("domain-refused"),
                );
            }
            Some(why) => {
                log::warn!("AERODUCT_PREFLIGHT=off: attempting {what} anyway ({why})");
                self.state.params.domain_mm = mm;
            }
            None => {
                log::info!("{what}: pre-flight passed ({} cells)", estimate.cells);
                self.state.params.domain_mm = mm;
            }
        }
    }

    fn update_clock(&mut self, steps: u32) {
        let c = &mut self.state.clock;
        c.steps = self.sim.solver.steps_taken();
        c.sim_time_s = self.sim.solver.sim_time_seconds();
        c.wall_time_s = self.wall_time_s;
        c.steps_per_second = self.step_rate.rate();
        c.steps_per_physical_second = self.sim.units.steps_per_physical_second();
        c.fps = self.ui.framerate() as f64;
        c.steps_per_frame = steps;
    }

    // -- actions -----------------------------------------------------------

    fn handle_actions(&mut self) -> Result<()> {
        for action in self.state.drain_actions() {
            match action {
                UiAction::SetPlaying(p) => self.state.playing = p,
                UiAction::StepOnce(n) => {
                    self.sim.solver.step(n);
                    self.sim.solver.compute_macroscopic();
                }
                UiAction::ResetSolver => {
                    self.sim.solver.reset();
                    self.window_start_step = 0;
                    self.wall_time_s = 0.0;
                    self.metrics.reset(ad_ui::view::ResetCause::SolverRestarted);
                }
                UiAction::ResetStatistics(cause) => {
                    self.metrics.reset(cause);
                    self.window_start_step = self.sim.solver.steps_taken();
                }
                UiAction::ApplyViewPreset(p) => {
                    self.controller.apply_preset(p);
                    self.controller
                        .frame_bbox(self.world_scene_bbox(), FRAME_MARGIN);
                }
                UiAction::FrameScene => {
                    self.controller
                        .frame_bbox(self.world_scene_bbox(), FRAME_MARGIN);
                }
                UiAction::SetField(f) => self.renderer.set_field(f),
                UiAction::LoadDuct(path) => {
                    let params = self.state.params;
                    // A new duct, with the obstructions kept where they are in
                    // the car. The same transactional swap as a rebuild, so an
                    // STL the GPU cannot take leaves the current part running.
                    let assets = self.obstructions.clone();
                    let placements = self.state.obstructions.clone();
                    let vents = self.state.vents.clone();
                    let scene = Sim::load_duct(&path).map(|duct| {
                        // The vents stand where they stand in the car; the new
                        // duct's centre is the pivot they are carried in by.
                        let pivot = duct.world_bbox().center();
                        let lattice = lattice_vents(&self.state.install, pivot, &vents);
                        (self.assemble_scene(duct, &assets, &placements), lattice)
                    });
                    match scene.and_then(|(s, v)| {
                        self.replace_sim(|gpu| Sim::from_scene(gpu, s, v, &params))
                    }) {
                        Ok(()) => {
                            self.committed_placements = placements;
                            self.committed_vents = vents;
                            self.state.duct = ad_ui::pose::DuctGeometry::IDENTITY;
                            self.committed_duct = self.state.duct;
                            self.committed_pose = self.state.install;
                        }
                        Err(e) if self.gpu.lost().is_some() => {
                            let why = self.gpu.lost().unwrap_or_default();
                            return Err(e.context(format!(
                                "the GPU device was lost loading {} ({why})",
                                path.display()
                            )));
                        }
                        Err(e) => {
                            log::error!("loading {}: {e:#}", path.display());
                            self.state.toasts.push(
                                ad_ui::Toast::new(ad_ui::Severity::Critical, "Could not load STL")
                                    .with_detail(format!("{path:?}: {e}")),
                            );
                        }
                    }
                }
                UiAction::AddSlice => {
                    let centre = self.sim.duct_bbox().center();
                    let i = self.state.slices.len();
                    let mut slice = ad_ui::overlays::SliceSettings::axis_aligned(
                        format!("slice {}", i + 1),
                        centre,
                        1,
                    );
                    // Level in the car whichever way the part is turned. Slices
                    // are stored on the part, like probes.
                    slice.normal = self.state.install.dir_to_lattice(Vec3::Y);
                    self.state.slices.push(slice);
                    let id = self
                        .state
                        .layers
                        .push(ad_ui::LayerKind::Slice(i), format!("Slice {}", i + 1));
                    self.state.layers.select(Some(id));
                }
                UiAction::ProbeFromRay {
                    origin_mm,
                    direction,
                } => {
                    // The ray is the camera's, in the world; the flags are in
                    // the lattice. Probes are kept in the lattice so they ride
                    // along when the part is re-posed.
                    let origin = self.state.to_lattice(origin_mm);
                    let direction = self.state.install.dir_to_lattice(direction);
                    // A click on a mouth's opening puts an air source there.
                    let hit = pick_mouth(&self.sim, origin, direction);
                    log::info!(
                        "ctrl+click: ray from {:.1?} along {:.2?} mm (lattice), mouth {:?}",
                        origin.to_array(),
                        direction.to_array(),
                        hit.map(|m| (b'A' + m as u8) as char)
                    );
                    if let Some(m) = hit {
                        self.place_vent_at_mouth(m);
                        continue;
                    }
                    match pick_probe(&self.sim, origin, direction) {
                        Some(p) => {
                            self.state.probes.add(p);
                        }
                        None => self.state.toasts.push(
                            ad_ui::Toast::new(ad_ui::Severity::Info, "Nothing under the cursor")
                                .with_detail("The ray did not enter the duct passage.")
                                .with_key("probe-miss"),
                        ),
                    }
                }
                UiAction::MoveProbe { id, position_mm } => {
                    if let Some(p) = self.state.probes.get_mut(id) {
                        p.position_mm = position_mm;
                    }
                }
                UiAction::Screenshot => match self.screenshot() {
                    Ok(path) => self.state.toasts.push(
                        ad_ui::Toast::new(ad_ui::Severity::Info, "Screenshot written")
                            .with_detail(path.display().to_string()),
                    ),
                    Err(e) => log::error!("screenshot failed: {e:#}"),
                },
                UiAction::RequestQuit => {
                    self.state.playing = false;
                }
                // Pre-flighted before `SimParams` is touched: a cell size that
                // will not fit is refused here, with the reason, rather than
                // attempted. One that passes goes through `sync_params` like
                // every other change.
                UiAction::SetResolution(dx) => self.request_resolution(dx),
                // Applied through `SimParams` and `sync_params`, which is what
                // makes the statistics reset impossible to forget.
                UiAction::SetInletVelocity(_)
                | UiAction::SetInletMouth(_)
                | UiAction::SwapInletOutlet
                | UiAction::SetLatticeVelocity(_)
                | UiAction::SetSmagorinsky(_) => {}
                // Handled inside the UI, or not yet wired.
                UiAction::AutoRangeLegend
                | UiAction::SaveBaseline
                | UiAction::ClearBaseline
                | UiAction::RemoveSlice(_)
                | UiAction::RemoveProbe(_) => {}
                UiAction::LoadObstruction(path) => self.add_obstruction(&path, Vec3::ZERO)?,
                UiAction::RemoveObstruction(i) => self.remove_obstruction(i)?,
                // An edit of the vent list, like a drag: the lattice follows
                // once it settles.
                UiAction::AddVent => {
                    let i = self.state.vents.len();
                    if i >= sim::MAX_VENTS {
                        self.state.toasts.push(
                            ad_ui::Toast::new(ad_ui::Severity::Info, "Three vents at most")
                                .with_detail(
                                    "The solver has three vent slots. Remove one to add another.",
                                )
                                .with_key("vent-slots"),
                        );
                    } else {
                        let v = self.default_vent(i);
                        let name = v.name.clone();
                        self.state.vents.push(v);
                        let id = self.state.layers.push(ad_ui::LayerKind::Vent(i), name);
                        self.state.layers.select(Some(id));
                    }
                }
                UiAction::BakePose => self.bake_pose(),
                UiAction::SetDomainMargins(mm) => self.request_domain(Some(mm)),
                UiAction::ResetDomainMargins => self.request_domain(None),
                // Voxelised once it settles, like a gizmo drag.
                UiAction::SetObstructionPlacement(i, p) => {
                    if let Some(o) = self.state.obstructions.get_mut(i) {
                        *o = p;
                    }
                }
            }
        }
        Ok(())
    }

    /// Rebuild the solver from the current parameters, transactionally.
    ///
    /// Never fails the frame. If the new lattice cannot be built, the old one is
    /// still running, the parameters snap back to what it was built with, and a
    /// toast says why. Before this, a failed rebuild — a resolution too fine for
    /// the card, say — propagated out of the frame loop and closed the window;
    /// and since the geometry had already been moved out of the old `Sim`, there
    /// was nothing left to fall back to anyway.
    ///
    /// The one failure it cannot absorb is a lost device, which it returns as
    /// an error so the frame loop stops; see [`Running::replace_sim`].
    fn rebuild(&mut self) -> Result<()> {
        let params = self.state.params;
        let was = self.state.committed_params();
        // A clone rather than a move, so the old `Sim` stays whole until the new
        // one exists. Cheap: the meshes are behind `Arc`s.
        let scene = self.sim.scene.clone();
        let vents = lattice_vents(
            &self.state.install,
            self.state.install_pivot_mm,
            &self.state.vents,
        );
        let started = Instant::now();
        match self.replace_sim(|gpu| Sim::from_scene(gpu, scene, vents, &params)) {
            Ok(()) => {
                self.committed_vents = self.state.vents.clone();
                self.state.duct = self.committed_duct;
                log::info!(
                    "rebuilt in {:.1} s: {}",
                    started.elapsed().as_secs_f64(),
                    self.sim.describe()
                )
            }
            Err(e) => {
                if let Some(why) = self.gpu.lost() {
                    return Err(e.context(format!(
                        "the GPU device was lost building the {} mm lattice ({why}); every \
                         resource went with it, the running solver included. Restart, and \
                         choose a coarser cell size",
                        params.dx_mm
                    )));
                }
                log::error!("rebuild failed; the previous solver keeps running: {e:#}");
                self.state.revert_params();
                self.state.toasts.push(
                    ad_ui::Toast::new(ad_ui::Severity::Warning, "Change not applied")
                        .with_detail(format!(
                            "{} Still running the previous setup (dx = {} mm).",
                            brief(&e),
                            was.dx_mm
                        ))
                        .with_key("rebuild-failed"),
                );
            }
        }
        Ok(())
    }

    /// Build a replacement `Sim` beside the running one, and swap it in only if
    /// every step succeeded.
    ///
    /// "Every step" includes what wgpu reports out of band. A buffer the device
    /// refuses does not come back as an `Err` from `create_buffer`: it goes to
    /// the innermost error scope — and with none open, to wgpu's default
    /// handler, which panics. So the whole build (voxeliser, solver, and the
    /// renderer if the grid changed) runs inside an out-of-memory scope and a
    /// validation scope, and anything either catches fails the attempt just as
    /// an `Err` would.
    ///
    /// One failure it cannot contain: running out of memory. Creating a buffer
    /// or texture fails softly, but wgpu 30 treats an out-of-memory in the
    /// staging uploads and submissions around it as losing the whole device,
    /// which frees the running solver along with the half-built one. Callers
    /// check [`GpuContext::lost`] after an error; the pre-flight in
    /// [`crate::resolution`] is what keeps a build from getting there.
    ///
    /// Holds a second lattice's worth of VRAM while it runs, which is what the
    /// pre-flight budgets for.
    fn replace_sim(&mut self, build: impl FnOnce(&GpuContext) -> Result<Sim>) -> Result<()> {
        let oom = self
            .gpu
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self
            .gpu
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        if std::env::var("AERODUCT_FAULT").as_deref() == Ok("validation") {
            // `AERODUCT_FAULT=validation`: one deliberately invalid buffer inside
            // the scopes (MAP_READ with MAP_WRITE needs a feature the device
            // does not request). The only way to prove headlessly that a wgpu
            // error here rolls back rather than reaching the default handler,
            // which would panic and take the window with it.
            log::warn!("AERODUCT_FAULT=validation: injecting a validation error into this build");
            let _ = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("deliberately invalid (AERODUCT_FAULT)"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
                mapped_at_creation: false,
            });
        }
        let built = build(&self.gpu).and_then(|sim| {
            // The derived-field textures and the brick accelerator are sized to
            // the lattice, so a new grid means a new renderer — built here,
            // while failing is still free.
            let renderer = if sim.grid != self.sim.grid {
                let config = self.renderer_config(sim.grid);
                Some(Renderer::new(&self.gpu, config).context("building the renderer")?)
            } else {
                None
            };
            Ok((sim, renderer))
        });
        // Scopes are a stack: the innermost comes off first.
        let validation = pollster::block_on(validation.pop());
        let oom = pollster::block_on(oom.pop());
        let refused = oom
            .map(|e| format!("the GPU ran out of memory: {e}"))
            .or_else(|| validation.map(|e| format!("the GPU rejected the new lattice: {e}")));
        let outcome = match (built, refused) {
            (Ok((sim, renderer)), None) => {
                self.install(sim, renderer);
                Ok(())
            }
            // Built on top of a refused allocation, so some of its handles are
            // invalid. Dropped here, before anything can use one.
            (Ok(_), Some(why)) => Err(anyhow::anyhow!(why)),
            (Err(e), None) => Err(e),
            (Err(e), Some(why)) => Err(e.context(why)),
        };
        // The build stalled this frame for seconds whatever its outcome, and
        // dropping whichever lattice lost adds to it. Restart the frame clock
        // only now, so the next frame does not report any of that as its own
        // duration: the step auto-tuner would read it as a frame far over budget
        // and spend dozens of frames climbing back from one step. Measured on a
        // failed 0.75 mm build: 42 steps in the next thirty frames without a
        // restart, 175 with it, against 475 when the failure was instant.
        self.last_frame = Instant::now();
        outcome
    }

    fn renderer_config(&self, grid: ad_gpu::Grid) -> RendererConfig {
        RendererConfig::new(
            self.surface_config.width,
            self.surface_config.height,
            self.surface_config.format,
            grid,
        )
    }

    /// Swap in a `Sim` that has been fully built. Nothing in here can fail;
    /// that is the point of doing it last.
    fn install(&mut self, sim: Sim, renderer: Option<Renderer>) {
        self.sim = sim;
        // A new solver starts with timestamps off; the tuner needs them.
        self.sim.solver.set_profiling(true);
        if let Some(mut renderer) = renderer {
            // The transfer function lives in the renderer, so a fresh one would
            // quietly reset the colormap, range and opacity curve the user had
            // set up. Everything else it draws with is pushed every frame.
            *renderer.transfer_function_mut() = self.renderer.transfer_function().clone();
            self.renderer = renderer;
        }
        let (render_scene, render_roles) = build_render_scene(&self.gpu, &self.sim);
        self.render_scene = render_scene;
        self.render_roles = render_roles;
        // The pivot is the duct's centre, which only a new STL moves; the pose
        // itself belongs to the user and survives every rebuild.
        self.state.install_pivot_mm = self.sim.duct_bbox().center();
        self.render_scene.model = self.state.install_matrix();
        self.render_scene.end_frame();
        self.sim.upload_flags(&self.gpu.queue);
        // The metrics passes bind the solver's textures and the interior flag
        // mask, both of which the new `Sim` has replaced, so they are rebuilt
        // rather than reset: a stale bind group would keep measuring the
        // geometry that was just thrown away.
        let (inlet, outlet) = mouth_roles(&self.sim, &self.state.params);
        self.metrics = build_metrics(&self.gpu, &self.sim, inlet, outlet);
        self.tracers = build_tracers(&self.sim, inlet, outlet);
        self.state.status = format!(
            "{} | {} | {}",
            self.gpu.info.name,
            self.sim.describe(),
            self.metrics.provenance()
        );
        metrics::publish_mouths(&self.sim, &mut self.state);
        publish_domain(&self.sim, &mut self.state);
        self.metrics.reset(ad_ui::view::ResetCause::GeometryChanged);
        self.window_start_step = 0;
        self.wall_time_s = 0.0;
        self.state.tuner.reset();
        // The cost line described the old lattice; work it out again.
        self.state.resolution.estimate = None;
    }

    /// Keep the resolution control's cost line current.
    ///
    /// Recomputed when the value in the box changes, and once a second
    /// otherwise, because the time estimate scales the live step rate. Not every
    /// frame: it does not move that fast, and the domain planner it runs can log.
    fn refresh_resolution_estimate(&mut self) {
        let stale = self.state.resolution.current_estimate().is_none()
            || self.resolution_refreshed.elapsed() >= std::time::Duration::from_secs(1);
        // Also when the box margins in their control moved: the one estimate
        // serves both controls, costed at the cell size and the margins each
        // is showing.
        let domain = Some(self.state.pending_domain_mm);
        let stale = stale || self.state.resolution.estimate_for(domain).is_none();
        if stale {
            let dx = self.state.resolution.pending_dx_mm;
            self.state.resolution.estimate = Some(self.estimate_resolution(dx, domain));
            self.resolution_refreshed = Instant::now();
        }
    }

    fn estimate_resolution(
        &self,
        dx_mm: f32,
        domain_mm: Option<[f32; 6]>,
    ) -> ad_ui::ResolutionEstimate {
        resolution::estimate(
            &self.gpu,
            &self.sim,
            &self.state.params,
            dx_mm,
            domain_mm,
            self.step_rate.rate(),
        )
    }

    /// Pre-flight a cell-size change and, if it fits, make it.
    ///
    /// "Make it" means writing `SimParams::dx_mm` and nothing else: the rebuild
    /// then happens through `sync_params` like every other change, which keeps
    /// the statistics reset and its toast in one place. A refusal writes
    /// nothing, so the only trace it leaves is the reason.
    fn request_resolution(&mut self, dx_mm: f32) {
        if dx_mm.to_bits() == self.state.params.dx_mm.to_bits() {
            return;
        }
        // Costed with the box the solver is running, not one still being typed.
        let estimate = self.estimate_resolution(dx_mm, self.state.params.domain_mm);
        let blocker = estimate.blocker.clone();
        self.resolution_refreshed = Instant::now();
        match blocker {
            Some(why) if !preflight_disabled() => {
                log::warn!("dx = {dx_mm} mm refused: {why}");
                self.state.toasts.push(
                    ad_ui::Toast::new(ad_ui::Severity::Warning, format!("{dx_mm} mm not applied"))
                        .with_detail(why)
                        .with_key("resolution-refused"),
                );
            }
            Some(why) => {
                log::warn!("AERODUCT_PREFLIGHT=off: attempting dx = {dx_mm} mm anyway ({why})");
                self.state.params.dx_mm = dx_mm;
            }
            None => {
                log::info!(
                    "dx {} -> {dx_mm} mm: pre-flight passed",
                    self.state.params.dx_mm
                );
                self.state.params.dx_mm = dx_mm;
            }
        }
    }

    /// Render a supersampled, temporally accumulated frame to a PNG.
    fn screenshot(&mut self) -> Result<std::path::PathBuf> {
        use ad_render::util;
        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        const SCALE: u32 = 2;
        const SAMPLES: u32 = 32;

        self.renderer.set_target_format(FORMAT)?;
        let (w, h) = self.renderer.begin_supersample(SCALE);
        let target = util::color_target(
            &self.gpu.device,
            "screenshot",
            w,
            h,
            FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let view = target.create_view(&Default::default());

        let velocity_view = self.sim.solver.velocity_view();
        let scales = self.sim.derive_scales();
        let mut pixels = Vec::new();
        for i in 0..SAMPLES {
            let mut enc = self
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("shot"),
                });
            self.renderer.render(
                &mut enc,
                FrameInput {
                    camera: &self.controller.current,
                    scene: &self.render_scene,
                    sources: (i == 0).then(|| FieldSources {
                        macro_view: &velocity_view,
                        flags_view: Some(&self.sim.flags_view),
                        scales,
                    }),
                    sdf: None,
                    target: &view,
                    dt: 1.0 / 60.0,
                },
            )?;
            if i + 1 == SAMPLES {
                pixels =
                    util::readback_rgba8(&self.gpu.device, &self.gpu.queue, &target, w, h, enc);
            } else {
                self.gpu.queue.submit([enc.finish()]);
            }
        }
        self.renderer.end_supersample();
        self.renderer
            .set_target_format(self.surface_config.format)?;

        let down = util::box_downsample_rgba8(&pixels, w, h, SCALE);
        let path = std::env::current_dir()?.join(format!("aeroduct-{:05}.png", {
            self.screenshot_counter += 1;
            self.screenshot_counter
        }));
        png::write_rgba(&path, w / SCALE, h / SCALE, &down)?;
        log::info!("screenshot: {}", path.display());
        Ok(path)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Which mouth is the inlet and which is the outlet, clamped to what was
/// actually detected.
///
/// One function rather than the same two lines at four call sites, because
/// getting them out of step means the metrics passes measure one hole and the
/// HUD labels the other.
fn mouth_roles(sim: &Sim, _params: &SimParams) -> (usize, usize) {
    // Decided when the lattice was built, from the parameters and the vents
    // it was built with; a pending edit rebuilds before it can matter.
    (sim.inlets[0], sim.outlet)
}

/// The real metrics source, or the placeholder if its passes will not build.
///
/// A shader that fails to compile on some other driver must degrade to "we do
/// not know" rather than to a plausible number: a fabricated pressure drop with
/// an error bar beside it is indistinguishable from a measured one.
fn build_metrics(
    gpu: &GpuContext,
    sim: &Sim,
    inlet: usize,
    outlet: usize,
) -> Box<dyn MetricsSource> {
    // `AERODUCT_METRICS=off` forces the placeholder. Purely a bisection handle:
    // when something goes wrong on a long run, the first question is whether the
    // measurement passes are involved, and the cheapest way to answer it is to
    // run the identical binary without them.
    if std::env::var("AERODUCT_METRICS").as_deref() == Ok("off") {
        log::warn!("AERODUCT_METRICS=off: measured values will read '--'");
        return Box::new(PlaceholderMetrics::default());
    }
    match DuctMetricsSource::new(gpu, sim, inlet, outlet) {
        Ok(m) => Box::new(m),
        Err(e) => {
            log::error!("metrics unavailable, falling back to the placeholder: {e:#}");
            Box::new(PlaceholderMetrics::default())
        }
    }
}

fn build_tracers(sim: &Sim, inlet: usize, outlet: usize) -> Option<TracerRtd> {
    if !tracers::rtd_enabled() {
        return None;
    }
    // Two seconds of wall clock between captures. The readback is blocking, so
    // this is the frame-time cost of the feature amortised to roughly 10%.
    let t = TracerRtd::new(sim, inlet, outlet, 2.0);
    if t.is_none() {
        log::warn!("AERODUCT_RTD is set but the inlet mouth gave no seedable tracers");
    }
    t
}

/// `AERODUCT_FRAMES`: render this many frames, capture the window, and exit.
///
/// Read in two places — the swapchain configuration wants to know whether this
/// is an unattended run before the frame loop exists — so it lives in one
/// function rather than being parsed twice.
fn headless_capture() -> Option<u64> {
    std::env::var("AERODUCT_FRAMES")
        .ok()?
        .parse::<u64>()
        .ok()
        .filter(|n| *n > 0)
}

/// `AERODUCT_PRESENT=fifo|mailbox|immediate`: override the present mode.
///
/// A measurement handle. An unattended run defaults to Fifo (see the swapchain
/// configuration for why), and Fifo quantises every frame to a whole number of
/// vblanks — 7 ms steps on a 143 Hz panel — so a headless frame rate says as
/// much about the monitor as about the frame. Profiling what an interactive
/// user sees needs the interactive present mode.
fn present_mode_override(supported: &[wgpu::PresentMode]) -> Option<wgpu::PresentMode> {
    let raw = std::env::var("AERODUCT_PRESENT").ok()?;
    let want = match raw.to_ascii_lowercase().as_str() {
        "fifo" => wgpu::PresentMode::Fifo,
        "mailbox" => wgpu::PresentMode::Mailbox,
        "immediate" => wgpu::PresentMode::Immediate,
        _ => {
            log::warn!("AERODUCT_PRESENT={raw} not recognised; using the default present mode");
            return None;
        }
    };
    if supported.contains(&want) {
        log::info!("present mode {want:?} from AERODUCT_PRESENT");
        Some(want)
    } else {
        log::warn!("AERODUCT_PRESENT asked for {want:?}, which this surface does not support");
        None
    }
}

/// `AERODUCT_MESH`: force the duct display mode for an unattended run.
///
/// Exists so a headless screenshot can answer "is the internal flow hidden
/// behind the shell, or is it just dim?" without a human at the keyboard. That
/// question is not decidable from a single ghosted frame, and it is the one that
/// matters most for a duct: seeing inside is the whole point of the tool.
fn mesh_display_override() -> Option<MeshDisplay> {
    match std::env::var("AERODUCT_MESH")
        .ok()?
        .to_ascii_lowercase()
        .as_str()
    {
        "off" | "none" => Some(MeshDisplay::Off),
        "solid" => Some(MeshDisplay::Solid),
        "ghost" => Some(MeshDisplay::Ghost),
        "wire" | "wireframe" => Some(MeshDisplay::Wireframe),
        other => {
            log::warn!("AERODUCT_MESH={other:?} not recognised; using the default");
            None
        }
    }
}

/// `AERODUCT_VIEW`: frame the whole domain from a named direction.
///
/// Sibling of [`mesh_display_override`], and for the same reason: a headless
/// screenshot has to answer a question, and the default view answers a different
/// one. Here the question is whether the anisotropic margins in [`crate::domain`]
/// left the exit jet room — a side-on view with the grid outline drawn shows the
/// jet and the face it would be clipped against in the same frame, which an
/// iso view framed on the part cannot.
fn view_override() -> Option<ViewPreset> {
    let v = std::env::var("AERODUCT_VIEW").ok()?.to_ascii_lowercase();
    let preset = match v.as_str() {
        "front" => ViewPreset::Front,
        "back" => ViewPreset::Back,
        "left" => ViewPreset::Left,
        "right" => ViewPreset::Right,
        "top" => ViewPreset::Top,
        "bottom" => ViewPreset::Bottom,
        "iso" => ViewPreset::Iso,
        other => {
            log::warn!("AERODUCT_VIEW={other:?} not recognised; keeping the default framing");
            return None;
        }
    };
    Some(preset)
}

/// `AERODUCT_STEPS`: pin the solver steps per frame instead of auto-tuning.
///
/// The auto-tuner targets a 16.7 ms frame, which on a 21.6 M-cell grid is a
/// handful of steps: fine interactively, and hopeless for a headless run that
/// has to reach fifteen flow-throughs. Pinning it trades frame rate for physics,
/// which is exactly the trade an unattended verification run wants.
fn pinned_steps() -> Option<u32> {
    std::env::var("AERODUCT_STEPS")
        .ok()?
        .parse::<u32>()
        .ok()
        .filter(|n| *n > 0 && *n <= 4096)
}

/// How long the swapchain may keep failing validation before the run gives up.
///
/// A window driven at a frame a second is one Windows will happily declare
/// unresponsive and invalidate the swapchain of; reconfiguring gets it back, but
/// not always on the next frame. Ten seconds is far longer than any hiccup
/// observed and far shorter than a run.
const ACQUIRE_RECOVERY_LIMIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Minimum gap between two attempts to reconfigure a failing surface.
const RECONFIGURE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// Extra framing margin around the part.
///
/// The 3D scene is rendered to the *whole* surface and the docked panels are
/// drawn over its edges, so a tight fit puts the ends of the duct underneath the
/// layer and properties columns. This is the fraction that keeps it inside the
/// central node at the default layout.
const FRAME_MARGIN: f32 = 0.55;

/// The same fraction for [`Running::frame_domain`], which frames the lattice
/// rather than the part.
///
/// Larger because the box is larger and because the thing being checked is at
/// its edge: the downstream face is the one an over-tight margin would clip the
/// jet against, and a shot with that face under the plots panel cannot show
/// whether it did.
const DOMAIN_FRAME_MARGIN: f32 = 0.95;

/// Reorder a captured surface into RGBA.
///
/// Swapchains are commonly `Bgra8Unorm`; writing those bytes straight into a PNG
/// produces a picture with the red and blue channels exchanged, which reads as a
/// colour-map bug rather than a capture bug.
fn to_rgba(mut raw: Vec<u8>, format: wgpu::TextureFormat) -> Vec<u8> {
    let swap = matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    for px in raw.chunks_exact_mut(4) {
        if swap {
            px.swap(0, 2);
        }
        px[3] = 255; // the swapchain's alpha is not meaningful in a screenshot
    }
    raw
}

fn dear_extent(texture: &wgpu::Texture) -> (u32, u32) {
    let s = texture.size();
    (s.width, s.height)
}

/// Startup overrides.
///
/// Environment variables rather than command-line flags because the useful
/// thing to do with them is to re-run the same command under a different
/// resolution or operating point while bisecting something. Every one of them
/// is validated: a typo must not produce a zero-cell grid or a NaN velocity,
/// both of which fail a long way from the cause.
///
/// The domain follows the same convention but is not `SimParams`: it is
/// resolved against the detected mouths rather than set by the user, so it
/// lives in [`crate::domain::DomainMargins::from_env`]. `AERODUCT_DOMAIN`
/// picks between `room`, `trimmed` and `plenum` by name — which is what an A/B
/// wants — and `AERODUCT_MARGIN_*` / `AERODUCT_PLENUM_*` tune whichever one is
/// selected.
fn startup_params() -> SimParams {
    let mut params = SimParams::default();
    if let Some(dx) = env_f32("AERODUCT_DX_MM").filter(|v| *v > 0.05) {
        params.dx_mm = dx;
    }
    if let Some(u) = env_f32("AERODUCT_U").filter(|v| (0.0..=50.0).contains(v)) {
        params.inlet_velocity_ms = u;
    }
    // The Smagorinsky constant, for bisecting how much of a reported pressure
    // drop is eddy viscosity. `SimParams::default` ships the *stability* value
    // (0.17), which is well above CONTRACT.md's accuracy range of 0.10-0.12, and
    // the loss coefficient is sensitive to it: the LES model raises `tau_eff`
    // from a base of 0.5011, so a factor of two in `Cs^2` is a large factor in
    // the effective viscosity. Zero disables the model.
    if let Some(c) = env_f32("AERODUCT_CS").filter(|v| (0.0..=0.5).contains(v)) {
        params.smagorinsky_c = c;
    }
    if let Ok(v) = std::env::var("AERODUCT_INLET") {
        // Mouth index, so a run can be repeated blowing the other way without
        // touching the UI.
        if let Ok(i) = v.parse::<usize>() {
            params.inlet_mouth = i.min(7);
        }
    }
    // The lattice velocity, for a run that needs a slower, safer lattice.
    if let Some(u) = env_f32("AERODUCT_ULB").filter(|v| (0.005..=0.2).contains(v)) {
        params.u_lb = u as f64;
    }
    // The six box margins by hand, mm: what the Simulated box control sets.
    if let Ok(v) = std::env::var("AERODUCT_DOMAIN_MM") {
        match parse_margins(&v) {
            Some(m) => params.domain_mm = Some(m),
            None => log::warn!("AERODUCT_DOMAIN_MM={v:?} is not six non-negative numbers; ignored"),
        }
    }
    params
}

/// `"-x,+x,-y,+y,-z,+z"` in mm, all finite and non-negative.
fn parse_margins(s: &str) -> Option<[f32; 6]> {
    let v: Option<Vec<f32>> = s
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f32>()
                .ok()
                .filter(|v| v.is_finite() && *v >= 0.0)
        })
        .collect();
    let v = v?;
    (v.len() == 6).then(|| [v[0], v[1], v[2], v[3], v[4], v[5]])
}

/// `AERODUCT_VENT="cx,cy,cz@nx,ny,nz@w,h[@scale][;...]"`: vents to stand in
/// the room at startup, car frame — a centre in mm, the way it blows, its
/// size in mm, and optionally its speed as a multiple of the inlet U.
fn parse_vents(spec: &str) -> Vec<ad_ui::VentSettings> {
    let nums = |s: &str| -> Option<Vec<f32>> {
        s.split(',')
            .map(|t| t.trim().parse::<f32>().ok().filter(|v| v.is_finite()))
            .collect()
    };
    spec.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .enumerate()
        .filter_map(|(i, item)| {
            let parts: Vec<&str> = item.split('@').collect();
            let (Some(c), Some(n), Some(size)) = (
                parts.first().and_then(|s| nums(s)),
                parts.get(1).and_then(|s| nums(s)),
                parts.get(2).and_then(|s| nums(s)),
            ) else {
                log::warn!("AERODUCT_VENT entry {item:?} is not \"cx,cy,cz@nx,ny,nz@w,h[@scale]\"; skipped");
                return None;
            };
            let (&[cx, cy, cz], &[nx, ny, nz], &[w, h]) = (c.as_slice(), n.as_slice(), size.as_slice()) else {
                log::warn!("AERODUCT_VENT entry {item:?} is not \"cx,cy,cz@nx,ny,nz@w,h[@scale]\"; skipped");
                return None;
            };
            let normal = Vec3::new(nx, ny, nz).normalize_or(Vec3::Z);
            let rotation = Quat::from_rotation_arc(Vec3::Z, normal);
            let mut v = ad_ui::VentSettings::new(
                format!("Vent {}", i + 1),
                ad_ui::Placement { translation_mm: Vec3::new(cx, cy, cz), rotation, scale: 1.0 },
                w,
                h,
            );
            if let Some(scale) = parts.get(3).and_then(|s| s.trim().parse::<f32>().ok()) {
                v.speed_scale = scale.clamp(0.0, 3.0);
            }
            Some(v)
        })
        .collect()
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()?
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
}

/// `AERODUCT_PREFLIGHT=off`: attempt a cell size even when the pre-flight says
/// it will not fit. A test handle for the transactional rebuild, whose failure
/// path otherwise only runs when the pre-flight's arithmetic is wrong.
fn preflight_disabled() -> bool {
    std::env::var("AERODUCT_PREFLIGHT").as_deref() == Ok("off")
}

/// An error cut down to what a toast can hold: the first line of the chain,
/// capped. The whole chain goes to the log; wgpu's validation reports run to
/// dozens of lines.
fn brief(e: &anyhow::Error) -> String {
    const MAX_CHARS: usize = 300;
    let full = format!("{e:#}");
    let line = full.lines().next().unwrap_or_default().trim();
    let mut out: String = line.chars().take(MAX_CHARS).collect();
    if line.chars().count() > MAX_CHARS {
        out.push_str("...");
    } else if !out.ends_with('.') {
        out.push('.');
    }
    out
}

/// How long an obstruction edit must hold still before it is voxelised.
const OBSTRUCTION_SETTLE: std::time::Duration = std::time::Duration::from_millis(300);

/// An obstruction's mesh, loaded once. Where it sits is `UiState::obstructions`.
#[derive(Clone)]
struct ObstructionAsset {
    name: String,
    asset: Arc<ad_geom::MeshAsset>,
}

/// `AERODUCT_OBSTRUCTION="file.stl[@dx,dy,dz][;...]"`: obstructions to load at
/// startup, each moved by an offset in mm in the car frame from where its own
/// coordinates put it. An entry with a malformed offset is skipped, with a
/// warning, rather than loaded somewhere it was not asked to be.
fn parse_obstructions(spec: &str) -> Vec<(std::path::PathBuf, Vec3)> {
    spec.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|item| {
            let (path, offset) = match item.rsplit_once('@') {
                Some((p, o)) => (p, Some(o)),
                None => (item, None),
            };
            let offset = match offset {
                None => Vec3::ZERO,
                Some(o) => {
                    let v: Option<Vec<f32>> = o
                        .split(',')
                        .map(|s| s.trim().parse::<f32>().ok().filter(|v| v.is_finite()))
                        .collect();
                    match v.as_deref() {
                        Some(&[x, y, z]) => Vec3::new(x, y, z),
                        _ => {
                            log::warn!("AERODUCT_OBSTRUCTION entry {item:?}: the offset is not \"dx,dy,dz\"; skipped");
                            return None;
                        }
                    }
                }
            };
            Some((std::path::PathBuf::from(path.trim()), offset))
        })
        .collect()
}

fn transform_matrix(t: ad_geom::Transform) -> Mat4 {
    Mat4::from_scale_rotation_translation(Vec3::splat(t.scale), t.rotation, t.translation)
}

/// A placement edit waiting to settle, and when it last changed.
struct PlacementEdit {
    at: Instant,
    placements: Vec<ad_ui::Placement>,
    vents: Vec<ad_ui::VentSettings>,
    pose: ad_ui::InstallPose,
    duct: ad_ui::pose::DuctGeometry,
}

/// The vents as the lattice needs them: carried from the car into the duct's
/// frame through the install pose, about `pivot`.
fn lattice_vents(
    pose: &ad_ui::InstallPose,
    pivot: Vec3,
    vents: &[ad_ui::VentSettings],
) -> Vec<sim::VentPatch> {
    vents
        .iter()
        .map(|v| {
            let (x, y) = v.axes();
            sim::VentPatch {
                center_mm: pose.to_lattice(pivot, v.placement.translation_mm),
                normal: pose.dir_to_lattice(v.normal()),
                half_u: pose.dir_to_lattice(x) * (v.width_mm * 0.5),
                half_v: pose.dir_to_lattice(y) * (v.height_mm * 0.5),
                direction: pose.dir_to_lattice(v.direction()),
                speed_scale: v.speed_scale,
            }
        })
        .collect()
}

/// Tell the UI the box the lattice was built in. The margins in its control
/// follow along unless the user has started editing them.
fn publish_domain(sim: &Sim, state: &mut UiState) {
    let d = &sim.domain;
    let running = [
        d.lo_mm.x, d.hi_mm.x, d.lo_mm.y, d.hi_mm.y, d.lo_mm.z, d.hi_mm.z,
    ];
    let bits = |m: [f32; 6]| m.map(f32::to_bits);
    if bits(state.pending_domain_mm) == bits(state.domain_margins_mm) {
        state.pending_domain_mm = running;
    }
    state.domain_margins_mm = running;
}

/// Upload the scene's triangles for the ghost/wireframe pass. Also returns, per
/// mesh, which obstruction it is (`None` for the duct), so the frame loop can
/// tie each to its layer.
fn build_render_scene(gpu: &GpuContext, sim: &Sim) -> (ad_render::Scene, Vec<Option<usize>>) {
    let mut scene = ad_render::Scene::new();
    let mut roles = Vec::new();
    let mut next_obstruction = 0;
    for (_, inst) in sim.scene.visible() {
        let (role, style) = match inst.role {
            ad_geom::MeshRole::Duct => (None, MeshStyle::default()),
            ad_geom::MeshRole::Obstruction => {
                next_obstruction += 1;
                // Warm grey-orange, so a car part never reads as more duct.
                let style = MeshStyle {
                    albedo: Vec3::new(0.80, 0.55, 0.38),
                    ..MeshStyle::default()
                };
                (Some(next_obstruction - 1), style)
            }
        };
        roles.push(role);
        let mesh = inst.world_mesh();
        let normals = mesh.pseudo_normals();
        let vertices: Vec<MeshVertex> = mesh
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| MeshVertex {
                position: p.to_array(),
                normal: normals
                    .vertex
                    .get(i)
                    .copied()
                    .unwrap_or(Vec3::Y)
                    .normalize_or(Vec3::Y)
                    .to_array(),
                scalar: 0.0,
            })
            .collect();
        let indices: Vec<u32> = mesh
            .indices
            .iter()
            .flat_map(|t| t.iter().copied())
            .collect();
        scene.meshes.push(GpuMesh::upload(
            &gpu.device,
            &gpu.queue,
            MeshData {
                vertices: &vertices,
                indices: &indices,
            },
            style,
        ));
    }
    (scene, roles)
}

/// A plausible display range for a field at a given operating point.
///
/// **Not a measurement.** The renderer has no cheap way to report the extremes
/// of a 20-million-cell field without a readback, so the legend's auto-range
/// falls back to what the operating point implies: a well-behaved contraction
/// peaks at two or three times the inlet bulk, and the dynamic pressure sets the
/// scale for the pressure field. Wave 3's reduction replaces this with the real
/// min and max, and nothing else has to change.
fn heuristic_field_range(field: DerivedField, u_in: f32, d_h_mm: f32, rho: f32) -> (f32, f32) {
    let u = u_in.max(0.1);
    match field {
        DerivedField::Speed => (0.0, u * 3.0),
        DerivedField::QCriterion => (0.0, 2.0),
        DerivedField::Vorticity => (0.0, 4.0 * u / (d_h_mm.max(1.0) * 1e-3)),
        DerivedField::Pressure => {
            let q = 0.5 * rho * u * u;
            (-3.0 * q, 3.0 * q)
        }
    }
}

/// March a picking ray through the flag field and return the first fluid cell
/// *inside* the duct.
///
/// "Inside" means: after the ray has passed through at least one solid cell. A
/// probe dropped on the near side of the shell would sit in the open air in
/// front of the part, which reads as a bug in the picking rather than as the
/// user having clicked the outside of an opaque object.
fn pick_probe(sim: &Sim, origin_mm: Vec3, direction: Vec3) -> Option<Vec3> {
    let grid = sim.grid;
    let dir = direction.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    let bbox = grid.bbox();
    let diagonal = bbox.size().length();
    let step = grid.dx_mm * 0.5;
    let n = (diagonal / step).ceil() as i32;

    let mut seen_solid = false;
    let mut first_fluid_before_solid = None;
    for i in 0..n.max(1) {
        let p = origin_mm + dir * (i as f32 * step);
        if !bbox.contains(p) {
            if seen_solid {
                break;
            }
            continue;
        }
        let c = grid.cell_containing(p);
        if c.cmplt(IVec3::ZERO).any() || c.cmpge(grid.dims.as_ivec3()).any() {
            continue;
        }
        let flag = sim.mask[grid.linear(c.as_uvec3()) as usize];
        if ad_gpu::flags::is_fluid(flag) {
            if seen_solid {
                return Some(grid.cell_center_mm(c.as_uvec3()));
            }
            first_fluid_before_solid.get_or_insert(grid.cell_center_mm(c.as_uvec3()));
        } else {
            seen_solid = true;
        }
    }
    None
}

/// The mouth a picking ray enters, if any: the nearest whose opening the ray
/// crosses inside its rectangle, with nothing solid in the way. Lattice frame.
fn pick_mouth(sim: &Sim, origin_mm: Vec3, direction: Vec3) -> Option<usize> {
    let dir = direction.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    let grid = sim.grid;
    let mut best: Option<(usize, f32)> = None;
    for (i, m) in sim.mouths.iter().enumerate() {
        let n = m.patch.normal.normalize_or_zero();
        let denom = dir.dot(n);
        if denom.abs() < 1.0e-6 {
            continue;
        }
        let t = (m.patch.center_mm - origin_mm).dot(n) / denom;
        if t <= 0.0 || best.is_some_and(|(_, bt)| t >= bt) {
            continue;
        }
        let p = origin_mm + dir * t;
        let half = m.patch.half_u.abs() + m.patch.half_v.abs();
        let d = (p - m.patch.center_mm).abs();
        if (0..3).any(|a| n[a].abs() < 0.5 && d[a] > half[a] + grid.dx_mm) {
            continue;
        }
        // Not through the part: walk the ray to the opening and give up at the
        // first solid cell.
        let step = grid.dx_mm * 0.5;
        let blocked = (1..(t / step) as i32).any(|k| {
            let q = origin_mm + dir * (k as f32 * step);
            let c = grid.cell_containing(q);
            c.cmpge(IVec3::ZERO).all()
                && c.cmplt(grid.dims.as_ivec3()).all()
                && !ad_gpu::flags::is_fluid(sim.mask[grid.linear(c.as_uvec3()) as usize])
        });
        if !blocked {
            best = Some((i, t));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_render::DerivedField;

    #[test]
    fn vent_specs_parse_place_direction_size_and_speed() {
        let v = parse_vents("0,10,-40 @ 0,0,1 @ 140,15 @ 0.5; 5,5,5@1,0,0@20,20");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].placement.translation_mm, Vec3::new(0.0, 10.0, -40.0));
        assert!(v[0].normal().abs_diff_eq(Vec3::Z, 1e-6));
        assert_eq!(
            (v[0].width_mm, v[0].height_mm, v[0].speed_scale),
            (140.0, 15.0, 0.5)
        );
        assert!(v[1].normal().abs_diff_eq(Vec3::X, 1e-6));
        assert_eq!(v[1].speed_scale, 1.0);
        assert!(parse_vents("1,2,3@0,0,1").is_empty(), "a vent needs a size");
        assert!(parse_vents("").is_empty());
    }

    #[test]
    fn margins_parse_as_six_non_negative_numbers() {
        assert_eq!(
            parse_margins("1, 2,3,4,5,6"),
            Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
        assert_eq!(parse_margins("1,2,3,4,5"), None);
        assert_eq!(parse_margins("1,2,3,4,5,-6"), None);
        assert_eq!(parse_margins("a,2,3,4,5,6"), None);
    }

    #[test]
    fn a_vent_is_carried_into_the_lattice_through_the_pose() {
        let mut pose = ad_ui::InstallPose::IDENTITY;
        pose.nudge(1, 90.0);
        let pivot = Vec3::new(-1.7, 36.1, 34.5);
        let v = ad_ui::VentSettings::new(
            "v",
            ad_ui::Placement {
                translation_mm: pivot + Vec3::X * 40.0,
                rotation: Quat::IDENTITY,
                scale: 1.0,
            },
            140.0,
            15.0,
        );
        let l = lattice_vents(&pose, pivot, &[v]);
        // World +X is lattice +Z after a quarter turn about Y; world +Z is
        // lattice -X.
        assert!(
            l[0].center_mm.abs_diff_eq(pivot + Vec3::Z * 40.0, 1e-3),
            "{}",
            l[0].center_mm
        );
        assert!(
            l[0].normal.abs_diff_eq(Vec3::NEG_X, 1e-5),
            "{}",
            l[0].normal
        );
        assert!(
            (l[0].half_u.length() - 70.0).abs() < 1e-3 && (l[0].half_v.length() - 7.5).abs() < 1e-3
        );
        assert_eq!(l[0].direction, l[0].normal);
    }

    #[test]
    fn obstruction_specs_parse_paths_and_offsets() {
        use std::path::PathBuf;
        let v = parse_obstructions(r"C:\parts\vane.stl@10, 0,-5; b.stl ;;");
        assert_eq!(
            v,
            vec![
                (
                    PathBuf::from(r"C:\parts\vane.stl"),
                    Vec3::new(10.0, 0.0, -5.0)
                ),
                (PathBuf::from("b.stl"), Vec3::ZERO),
            ]
        );
        assert!(
            parse_obstructions("x.stl@1,2").is_empty(),
            "a bad offset drops that entry"
        );
        assert!(parse_obstructions("").is_empty());
    }

    #[test]
    fn a_toast_gets_the_first_line_of_a_gpu_error_not_all_of_it() {
        let e = anyhow::anyhow!("line one\nline two\nline three").context("building the solver");
        assert_eq!(brief(&e), "building the solver: line one.");
        let long = brief(&anyhow::anyhow!("x".repeat(1000)));
        assert!(
            long.chars().count() == 303 && long.ends_with("..."),
            "{long}"
        );
    }

    #[test]
    fn the_heuristic_range_brackets_the_operating_point() {
        // Not a measurement, but it must at least contain what the flow can
        // physically do, or auto-range clips the picture.
        let (lo, hi) = heuristic_field_range(DerivedField::Speed, 3.0, 6.3, 1.184);
        assert_eq!(lo, 0.0);
        assert!(
            hi > 3.0,
            "the range must reach past the inlet bulk, got {hi}"
        );

        let (lo, hi) = heuristic_field_range(DerivedField::Pressure, 8.0, 6.3, 1.184);
        assert!(
            lo < 0.0 && hi > 0.0,
            "pressure must be signed and symmetric"
        );
        assert!((lo + hi).abs() < 1e-4);

        // A zero-velocity slider must not collapse the range to a point.
        let (lo, hi) = heuristic_field_range(DerivedField::Speed, 0.0, 6.3, 1.184);
        assert!(
            hi > lo,
            "a stopped fan must still leave a usable colour bar"
        );
    }

    #[test]
    fn startup_parameters_reject_a_nonsense_resolution() {
        // The env override is a convenience for bisecting VRAM problems; a typo
        // in it must not produce a zero-cell grid.
        let base = SimParams::default();
        assert!(base.dx_mm > 0.0);
        std::env::set_var("AERODUCT_DX_MM", "not-a-number");
        assert_eq!(startup_params().dx_mm, base.dx_mm);
        std::env::set_var("AERODUCT_DX_MM", "0.0");
        assert_eq!(startup_params().dx_mm, base.dx_mm);
        std::env::set_var("AERODUCT_DX_MM", "1.5");
        assert_eq!(startup_params().dx_mm, 1.5);
        std::env::remove_var("AERODUCT_DX_MM");

        // The same guard on the operating point: a NaN inlet velocity would
        // make `LatticeUnits` produce a NaN time step and the whole status bar
        // would read "--" with no clue why.
        std::env::set_var("AERODUCT_U", "NaN");
        assert_eq!(startup_params().inlet_velocity_ms, base.inlet_velocity_ms);
        std::env::set_var("AERODUCT_U", "5.5");
        assert_eq!(startup_params().inlet_velocity_ms, 5.5);
        std::env::remove_var("AERODUCT_U");

        std::env::set_var("AERODUCT_INLET", "1");
        assert_eq!(startup_params().inlet_mouth, 1);
        std::env::remove_var("AERODUCT_INLET");

        // The LES constant, which is the one override that can quietly change a
        // reported loss coefficient by a factor of several.
        std::env::set_var("AERODUCT_CS", "0.11");
        assert!((startup_params().smagorinsky_c - 0.11).abs() < 1e-6);
        std::env::set_var("AERODUCT_CS", "-1");
        assert_eq!(
            startup_params().smagorinsky_c,
            base.smagorinsky_c,
            "a negative Cs is not a model"
        );
        std::env::remove_var("AERODUCT_CS");

        // Steps per frame: a pin outside the tuner's range must be ignored
        // rather than clamped to something the user did not ask for.
        assert_eq!(pinned_steps(), None);
        std::env::set_var("AERODUCT_STEPS", "200");
        assert_eq!(pinned_steps(), Some(200));
        std::env::set_var("AERODUCT_STEPS", "0");
        assert_eq!(pinned_steps(), None);
        std::env::remove_var("AERODUCT_STEPS");

        // The domain view, which is how an unattended run gets a shot that
        // shows whether the exit jet is clipped at a box face. A name that is
        // not a preset must leave the default framing rather than pick one.
        assert_eq!(view_override(), None);
        std::env::set_var("AERODUCT_VIEW", "LEFT");
        assert_eq!(view_override(), Some(ViewPreset::Left));
        std::env::set_var("AERODUCT_VIEW", "sideways");
        assert_eq!(view_override(), None);
        std::env::remove_var("AERODUCT_VIEW");
    }
}

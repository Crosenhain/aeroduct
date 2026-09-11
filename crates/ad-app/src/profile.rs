//! Where a frame's time goes: the breakdown behind the Diagnostics panel's
//! "GPU passes" section and the headless log.
//!
//! Two clocks, because either alone misleads.
//!
//! * **GPU timestamps** say what each piece of work cost the device: the step
//!   batch and the macroscopic pass (the solver's own profiler), the metrics
//!   reductions (the metrics source's), and the whole render graph and the UI
//!   (spans here, around the renderer's per-pass scopes).
//! * **CPU laps** say where the frame loop spent its wall clock. `acquire` is
//!   the tell: a GPU-bound frame spends its slack blocked in
//!   `get_current_texture`, a CPU-bound one does not.
//!
//! Means are over a fixed window of frames rather than exponential, for the
//! same reason as [`ad_ui::RateMeter`]: they describe the last two seconds, not
//! a blend of everything since the last hitch. The step count's range over the
//! window is reported beside its mean, because a tuner that swings between 1
//! and 12 steps and one that holds 6 have the same mean and very different
//! frame pacing.

use std::time::Instant;

use ad_gpu::{GpuContext, Profiler};

/// The stretches of `Running::frame` the laps split it into.
#[derive(Debug, Clone, Copy)]
pub enum Phase {
    /// Actions, parameter sync, camera: everything before the solver.
    Update,
    /// Recording and submitting the step batch and the macroscopic pass.
    Solver,
    /// Tracers and the metrics sample, which records and submits its own pass.
    Metrics,
    /// `get_current_texture`, which blocks while the swapchain has no free image.
    Acquire,
    /// Recording the render graph.
    Render,
    /// Building the panels and recording the UI.
    Ui,
    /// Submit, present, and the profilers' readback.
    Present,
}

const PHASES: usize = 7;
const PHASE_NAMES: [&str; PHASES] = [
    "update", "solver", "metrics", "acquire", "render", "ui", "present",
];

/// CPU time per [`Phase`] for one frame.
pub struct Laps {
    last: Instant,
    ms: [f64; PHASES],
}

impl Laps {
    pub fn start(at: Instant) -> Self {
        Self {
            last: at,
            ms: [0.0; PHASES],
        }
    }

    /// Charge the time since the previous lap to `phase`.
    pub fn lap(&mut self, phase: Phase) {
        let now = Instant::now();
        self.ms[phase as usize] += (now - self.last).as_secs_f64() * 1e3;
        self.last = now;
    }
}

/// Sums for the window being filled.
#[derive(Clone, Copy)]
struct Window {
    frames: u32,
    dt_ms: f64,
    steps: f64,
    min_steps: u32,
    max_steps: u32,
    cpu: [f64; PHASES],
}

impl Window {
    const EMPTY: Self = Self {
        frames: 0,
        dt_ms: 0.0,
        steps: 0.0,
        min_steps: u32::MAX,
        max_steps: 0,
        cpu: [0.0; PHASES],
    };
}

pub struct FrameProfile {
    /// Spans around whole stretches of the frame encoder: `render`, `ui`.
    pub gpu: Profiler,
    filling: Window,
    /// The last closed window, as means. `None` until one closes.
    closed: Option<Window>,
}

impl FrameProfile {
    /// Frames per averaging window: two seconds at 60 fps.
    pub const WINDOW: u32 = 120;

    pub fn new(gpu: &GpuContext) -> Self {
        Self {
            gpu: Profiler::new(&gpu.device, &gpu.queue, 4, gpu.caps.timestamps, None),
            filling: Window::EMPTY,
            closed: None,
        }
    }

    /// Fold in one frame: its wall-clock interval, its laps and the steps it
    /// dispatched. Returns true when this frame closed a window.
    pub fn end_frame(&mut self, dt_s: f32, laps: &Laps, steps: u32) -> bool {
        let w = &mut self.filling;
        w.frames += 1;
        w.dt_ms += dt_s as f64 * 1e3;
        w.steps += steps as f64;
        w.min_steps = w.min_steps.min(steps);
        w.max_steps = w.max_steps.max(steps);
        for (sum, ms) in w.cpu.iter_mut().zip(laps.ms) {
            *sum += ms;
        }
        if w.frames < Self::WINDOW {
            return false;
        }
        let n = w.frames as f64;
        let mut mean = *w;
        mean.dt_ms /= n;
        mean.steps /= n;
        mean.cpu.iter_mut().for_each(|ms| *ms /= n);
        self.closed = Some(mean);
        self.filling = Window::EMPTY;
        true
    }

    /// The breakdown, one line per row of the Diagnostics panel.
    ///
    /// GPU numbers read `--` where nothing times them (an adapter without
    /// timestamp queries). The macroscopic, metrics and derive figures are per
    /// *stepped* frame, since nothing runs them otherwise.
    pub fn report(
        &self,
        solver: &ad_solver::Solver,
        renderer: &ad_render::Renderer,
        metrics: &dyn crate::metrics::MetricsSource,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        let metrics_ms = metrics.gpu_ms();
        if let Some(w) = &self.closed {
            let fmt = |v: Option<f64>| v.map_or_else(|| "--".to_string(), |v| format!("{v:.2}"));
            let per_step = solver.ms_per_step();
            let steps_ms = per_step.map(|s| s * w.steps);
            let macro_ms = solver.macroscopic_ms();
            let render_ms = self.gpu.timing("render").map(|t| t.mean_ms);
            let ui_ms = self.gpu.timing("ui").map(|t| t.mean_ms);
            let gpu_ms: f64 = [steps_ms, macro_ms, metrics_ms, render_ms, ui_ms]
                .into_iter()
                .flatten()
                .sum();
            lines.push(format!(
                "frame {:.2} ms ({:.0} fps), {:.2} steps/frame ({}-{}), GPU {:.2} ms accounted",
                w.dt_ms,
                1000.0 / w.dt_ms.max(1e-3),
                w.steps,
                w.min_steps,
                w.max_steps,
                gpu_ms,
            ));
            lines.push(format!(
                "GPU ms: steps {} ({}/step) | macroscopic {} | metrics {} | render {} | ui {}",
                fmt(steps_ms),
                fmt(per_step),
                fmt(macro_ms),
                fmt(metrics_ms),
                fmt(render_ms),
                fmt(ui_ms),
            ));
            let lapped: f64 = w.cpu.iter().sum();
            let mut cpu = String::from("CPU ms:");
            for (name, ms) in PHASE_NAMES.iter().zip(w.cpu) {
                cpu.push_str(&format!(" {name} {ms:.2} |"));
            }
            cpu.push_str(&format!(" event loop {:.2}", (w.dt_ms - lapped).max(0.0)));
            lines.push(cpu);
        }
        lines.extend(renderer.profiling_report());
        lines.extend(metrics.gpu_report());
        lines
    }
}

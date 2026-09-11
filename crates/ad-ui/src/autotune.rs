//! Auto-tuning the number of solver steps per rendered frame, and the clock
//! that reports what that actually bought.
//!
//! # The controller
//!
//! The app wants two things at once: the highest sim throughput the GPU can
//! deliver, and a UI that stays draggable. Those trade directly, because the
//! solver and the renderer share one device queue — a frame that dispatches 200
//! LBM steps will not present for a third of a second whatever the renderer
//! does.
//!
//! So: pick a frame-time budget, measure the achieved frame time, and move the
//! step count toward the budget. Two details make the difference between a
//! controller that settles and one that oscillates visibly:
//!
//! * **Multiplicative, clamped adjustment.** The correction is
//!   `n *= budget / measured`, clamped to a factor of 2 per frame. Additive
//!   stepping takes hundreds of frames to climb from 1 to 400; unclamped
//!   multiplication overshoots into a 2-second frame on the first hitch and
//!   then hunts.
//! * **A dead band.** Inside +/-12% of the budget nothing changes. Without it
//!   the count jitters every frame, and because each frame's cost depends on
//!   the count, the jitter is self-sustaining — a visible pulsing in the frame
//!   rate that looks like a stutter bug.
//!
//! The measurement is the *whole* frame, sim and render together, because that
//! is what the user experiences and what the budget is a statement about.
//!
//! # Splitting the budget
//!
//! That law has a weakness a unit test without latency never sees: the frame
//! interval it measures is the bill for a batch dispatched a couple of frames
//! earlier, because the swapchain lets the CPU run that far ahead of the GPU.
//! Feedback through a delay goes unstable once the gain is high enough, and the
//! gain here is the solver's share of the frame. At `dx = 0.75 mm`, with 4.4 ms
//! steps and 4.6 ms of everything else, it swung between 1 and 17 steps a frame
//! and averaged 25% over budget.
//!
//! So when the GPU's per-step cost is known (timestamps), the tuner splits the
//! budget instead of chasing it: it estimates the part of the frame the step
//! count does not buy -- `frame - steps * cost`, against the batch that frame
//! was actually billed for -- and asks for `(budget - overhead) / cost` steps.
//! The overhead does not depend on the count, so there is no loop gain left to
//! go unstable. The multiplicative law remains the fallback where timestamps
//! are unavailable.
//!
//! # The honesty problem
//!
//! At `dx = 0.4 mm` one physical second is 200,000 LBM steps. Even at 70
//! steps/s that is 48 minutes of wall clock per simulated second, so the sim
//! runs some 2,800 times *slower* than real time. "Real-time" in this app means
//! real-time *rendering* — the picture keeps up with your hand on the mouse —
//! and it would be trivially easy to let a status bar imply otherwise.
//! [`Clock::realtime_factor`] therefore reports the honest ratio, and
//! [`Clock::status_line`] spells out which "real-time" is meant.

use std::collections::VecDeque;
use std::time::Duration;

/// Tunable limits for [`StepAutoTuner`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoTuneConfig {
    /// Wall-clock budget for one whole frame, seconds. 1/60 keeps the UI
    /// responsive; users who want throughput raise it.
    pub frame_budget_s: f32,
    pub min_steps: u32,
    pub max_steps: u32,
    /// Relative dead band around the budget. Inside it, do nothing.
    pub dead_band: f32,
    /// Largest multiplicative change per frame.
    pub max_ratio: f32,
    /// Exponential-smoothing weight for the frame-time estimate. Small enough
    /// that one hitched frame does not halve the step count.
    pub smoothing: f32,
    /// Frames between dispatching a batch and seeing its cost in the measured
    /// frame interval: how far the swapchain lets the CPU run ahead of the GPU
    /// (`desired_maximum_frame_latency` in the app).
    pub latency_frames: u32,
    /// Smoothing weight for the overhead estimate when the step cost is known.
    pub overhead_smoothing: f32,
}

impl Default for AutoTuneConfig {
    fn default() -> Self {
        Self {
            frame_budget_s: 1.0 / 60.0,
            min_steps: 1,
            max_steps: 4096,
            dead_band: 0.12,
            max_ratio: 2.0,
            smoothing: 0.15,
            latency_frames: 2,
            overhead_smoothing: 0.15,
        }
    }
}

/// Picks a steps-per-frame count that keeps the frame inside a time budget.
#[derive(Debug, Clone)]
pub struct StepAutoTuner {
    pub config: AutoTuneConfig,
    /// Off means the user pinned the count by hand.
    pub enabled: bool,
    steps: f32,
    smoothed_frame_s: f32,
    /// The part of a frame the step count does not buy: render, metrics, the
    /// macroscopic pass, the UI. Estimated only when a step cost is supplied.
    overhead_s: Option<f32>,
    /// Steps actually run by the most recent frames, newest first, so each
    /// frame time is charged to the batch that caused it.
    recent: VecDeque<u32>,
}

impl Default for StepAutoTuner {
    fn default() -> Self {
        Self::new(AutoTuneConfig::default())
    }
}

impl StepAutoTuner {
    pub fn new(config: AutoTuneConfig) -> Self {
        Self {
            steps: config.min_steps.max(1) as f32,
            smoothed_frame_s: config.frame_budget_s,
            overhead_s: None,
            recent: VecDeque::new(),
            config,
            enabled: true,
        }
    }

    /// Steps to dispatch this frame.
    pub fn steps(&self) -> u32 {
        (self.steps.round() as i64).clamp(self.config.min_steps as i64, self.config.max_steps as i64)
            as u32
    }

    /// Smoothed whole-frame time, seconds.
    pub fn frame_time_s(&self) -> f32 {
        self.smoothed_frame_s
    }

    /// Pin the count by hand, which also turns auto-tuning off. Manual control
    /// exists because reproducing a bug sometimes needs a fixed step count.
    pub fn set_manual(&mut self, steps: u32) {
        self.enabled = false;
        self.steps = steps.clamp(self.config.min_steps, self.config.max_steps) as f32;
    }

    /// Feed in the wall-clock duration of the frame just finished and get the
    /// count for the next one. Assumes the frame ran the count asked for and
    /// that no step cost is known: the multiplicative law alone.
    pub fn update(&mut self, frame: Duration) -> u32 {
        self.update_measured(frame, self.steps(), None)
    }

    /// [`Self::update`] with what the frame loop actually knows: the steps the
    /// frame ran (zero while paused) and the GPU cost of one step, when
    /// timestamps provide it. See "Splitting the budget" in the module docs.
    pub fn update_measured(
        &mut self,
        frame: Duration,
        steps_run: u32,
        step_cost: Option<Duration>,
    ) -> u32 {
        let dt = frame.as_secs_f32().max(1e-6);
        let s = self.config.smoothing.clamp(0.01, 1.0);
        self.smoothed_frame_s += (dt - self.smoothed_frame_s) * s;

        // The batch this frame's time is the bill for. Until the pipeline has
        // filled, the oldest one on record.
        self.recent.push_front(steps_run);
        self.recent.truncate(self.config.latency_frames as usize + 1);
        let billed = self.recent.back().copied().unwrap_or(steps_run);

        if !self.enabled {
            return self.steps();
        }

        let budget = self.config.frame_budget_s.max(1e-4);
        let (lo, hi) = (self.config.min_steps as f32, self.config.max_steps as f32);
        let cost = step_cost.map(|c| c.as_secs_f32()).filter(|c| c.is_finite() && *c > 0.0);
        let ratio = match cost {
            Some(c) => {
                // A stall is not overhead the step count could make room for,
                // so one hitched frame moves the estimate by a bounded amount.
                let sample = (dt - billed as f32 * c).clamp(0.0, 2.0 * budget);
                let a = self.config.overhead_smoothing.clamp(0.01, 1.0);
                let overhead = match self.overhead_s {
                    Some(o) => o + (sample - o) * a,
                    None => sample,
                };
                self.overhead_s = Some(overhead);
                ((budget - overhead) / c).clamp(lo, hi) / self.steps.max(1e-3)
            }
            // A paused frame is fast because it ran nothing. Read as headroom,
            // it walked the count up to `max_steps`, and the first frame after
            // resuming dispatched thousands of steps.
            None if steps_run == 0 => return self.steps(),
            None => budget / self.smoothed_frame_s.max(1e-6),
        };
        if (ratio - 1.0).abs() > self.config.dead_band {
            let clamped = ratio.clamp(1.0 / self.config.max_ratio, self.config.max_ratio);
            self.steps = (self.steps * clamped).clamp(lo, hi);
        }
        self.steps()
    }

    /// Forget the frame-time history, e.g. after a resolution change makes
    /// every past measurement irrelevant.
    ///
    /// A pinned count survives: it is what the tuner was told, not something
    /// it learned. Before this, every rebuild under `AERODUCT_STEPS` dropped a
    /// headless run to one step a frame, which reads as a stalled flow.
    pub fn reset(&mut self) {
        if self.enabled {
            self.steps = self.config.min_steps.max(1) as f32;
        }
        self.smoothed_frame_s = self.config.frame_budget_s;
        self.overhead_s = None;
        self.recent.clear();
    }
}

/// The three clocks the status bar has to keep straight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clock {
    /// Physical time the solver has advanced, seconds.
    pub sim_time_s: f64,
    /// Wall-clock time spent stepping, seconds. Excludes paused time, so
    /// "x real-time" does not improve while the sim is stopped.
    pub wall_time_s: f64,
    pub steps: u64,
    /// Measured over a recent window, not since startup.
    pub steps_per_second: f64,
    /// Steps in one physical second at this operating point, `1 / dt`.
    pub steps_per_physical_second: f64,
    /// Rendered frames per second.
    pub fps: f64,
    /// Steps dispatched per rendered frame.
    pub steps_per_frame: u32,
}

impl Default for Clock {
    fn default() -> Self {
        Self {
            sim_time_s: 0.0,
            wall_time_s: 0.0,
            steps: 0,
            steps_per_second: 0.0,
            steps_per_physical_second: f64::NAN,
            fps: 0.0,
            steps_per_frame: 1,
        }
    }
}

impl Clock {
    /// Simulated seconds per wall second. Far below 1 for any real duct case;
    /// the number exists to be looked at, not to be reassuring.
    pub fn realtime_factor(&self) -> f64 {
        if self.steps_per_physical_second.is_finite() && self.steps_per_physical_second > 0.0 {
            self.steps_per_second / self.steps_per_physical_second
        } else {
            f64::NAN
        }
    }

    /// Wall-clock seconds to advance one more second of physical time.
    pub fn wall_seconds_per_sim_second(&self) -> f64 {
        let f = self.realtime_factor();
        if f.is_finite() && f > 0.0 { 1.0 / f } else { f64::NAN }
    }

    /// One line for the status bar, with the "real-time" ambiguity resolved
    /// rather than left to the reader.
    pub fn status_line(&self) -> String {
        let rt = self.realtime_factor();
        let pace = if !rt.is_finite() || rt <= 0.0 {
            "x real-time --".to_string()
        } else if rt >= 1.0 {
            format!("{rt:.2}x real-time")
        } else {
            // A number like 0.00036 is unreadable, and "3.6e-4x real time" is
            // worse. Say how long a simulated second costs instead: that is the
            // form the user can plan around.
            format!(
                "1/{:.0} real-time ({} of wall clock per simulated second)",
                1.0 / rt,
                crate::format::duration(self.wall_seconds_per_sim_second())
            )
        };
        format!(
            "sim {} | wall {} | {} | {:.0} steps/s | {} steps/frame | {:.0} fps",
            crate::format::duration(self.sim_time_s),
            crate::format::duration(self.wall_time_s),
            pace,
            self.steps_per_second,
            self.steps_per_frame,
            self.fps,
        )
    }

    /// The caveat the status bar shows beside the pace, so nobody reads
    /// "60 fps" as "this duct is being simulated in real time".
    pub const REALTIME_NOTE: &'static str =
        "\"real-time\" here means real-time rendering: the picture keeps up with the mouse. \
         The physics does not. At dx = 0.4 mm one physical second is 200,000 LBM steps, \
         so a second of simulated air costs tens of minutes of wall clock.";
}

/// Rolling wall-clock rate estimator.
///
/// Windowed rather than cumulative on purpose. A since-startup average is
/// dominated by the first few seconds — voxelisation, pipeline compiles, the
/// first slow frames — and keeps reporting a rate the machine has not achieved
/// for minutes. A one-second window tracks what is happening now, which is what
/// someone dragging a slider is asking about.
#[derive(Debug, Clone)]
pub struct RateMeter {
    window_s: f64,
    elapsed_s: f64,
    count: f64,
    rate: f64,
}

impl Default for RateMeter {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl RateMeter {
    pub fn new(window_s: f64) -> Self {
        Self { window_s: window_s.max(1e-3), elapsed_s: 0.0, count: 0.0, rate: 0.0 }
    }

    /// Record `n` events over `dt` seconds.
    pub fn tick(&mut self, n: f64, dt: f64) {
        self.count += n;
        self.elapsed_s += dt.max(0.0);
        if self.elapsed_s >= self.window_s {
            self.rate = self.count / self.elapsed_s;
            self.count = 0.0;
            self.elapsed_s = 0.0;
        }
    }

    /// Events per second over the last completed window. Zero until the first
    /// window closes, which is honest: nothing has been measured yet.
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(x: f32) -> Duration {
        Duration::from_secs_f32(x / 1000.0)
    }

    #[test]
    fn the_tuner_climbs_toward_the_budget_and_settles() {
        // Model a machine that takes 0.15 ms per step plus 3 ms of render.
        // At a 16.7 ms budget that is about 91 steps.
        let mut t = StepAutoTuner::default();
        let mut steps = t.steps();
        for _ in 0..400 {
            let frame = 3.0 + steps as f32 * 0.15;
            steps = t.update(ms(frame));
        }
        let frame = 3.0 + steps as f32 * 0.15;
        assert!(
            (frame - 16.67).abs() < 3.0,
            "settled at {steps} steps, {frame:.1} ms/frame"
        );
        // ...and stays there rather than hunting.
        let a = t.steps();
        for _ in 0..60 {
            let f = 3.0 + t.steps() as f32 * 0.15;
            t.update(ms(f));
        }
        let b = t.steps();
        assert!(
            (a as i64 - b as i64).abs() <= (a as i64 / 8).max(2),
            "step count oscillates: {a} then {b}"
        );
    }

    #[test]
    fn the_tuner_backs_off_hard_when_frames_blow_the_budget() {
        let mut t = StepAutoTuner::default();
        t.set_manual(2000);
        t.enabled = true;
        // 400 ms frames: a 24x overrun.
        for _ in 0..40 {
            t.update(ms(400.0));
        }
        assert!(t.steps() < 200, "still at {} steps after a sustained overrun", t.steps());
    }

    #[test]
    fn one_hitched_frame_does_not_collapse_the_step_count() {
        // The smoothing is what stops a single 300 ms stall (a shader compile,
        // an STL load) from dropping the sim to one step per frame and taking
        // several seconds to climb back.
        let mut t = StepAutoTuner::default();
        for _ in 0..200 {
            t.update(ms(16.0));
        }
        let before = t.steps();
        t.update(ms(300.0));
        let after = t.steps();
        assert!(
            after as f32 > before as f32 * 0.45,
            "one hitch took {before} steps down to {after}"
        );
    }

    #[test]
    fn the_dead_band_stops_jitter_near_the_budget() {
        let mut t = StepAutoTuner::default();
        for _ in 0..200 {
            t.update(ms(16.0));
        }
        let settled = t.steps();
        // Alternate 5% either side of the budget: inside the dead band, so
        // nothing should move.
        for i in 0..40 {
            t.update(ms(if i % 2 == 0 { 15.8 } else { 17.5 }));
        }
        assert_eq!(t.steps(), settled, "the count moved inside the dead band");
    }

    #[test]
    fn manual_mode_pins_the_count_and_still_measures_frame_time() {
        let mut t = StepAutoTuner::default();
        t.set_manual(64);
        assert!(!t.enabled);
        for _ in 0..50 {
            t.update(ms(500.0));
        }
        assert_eq!(t.steps(), 64, "manual mode must not be overridden");
        assert!(t.frame_time_s() > 0.2, "frame time must still be tracked");
    }

    #[test]
    fn a_reset_keeps_a_pinned_count() {
        let mut t = StepAutoTuner::default();
        t.set_manual(20);
        t.reset();
        assert_eq!(t.steps(), 20, "a rebuild must not unpin AERODUCT_STEPS");
        let mut auto = StepAutoTuner::default();
        for _ in 0..50 {
            auto.update(ms(2.0));
        }
        auto.reset();
        assert_eq!(auto.steps(), auto.config.min_steps.max(1), "an auto count starts over");
    }

    #[test]
    fn limits_are_respected_at_both_ends() {
        let cfg = AutoTuneConfig { min_steps: 4, max_steps: 32, ..Default::default() };
        let mut t = StepAutoTuner::new(cfg);
        for _ in 0..200 {
            t.update(ms(0.01));
        }
        assert_eq!(t.steps(), 32);
        for _ in 0..200 {
            t.update(ms(5000.0));
        }
        assert_eq!(t.steps(), 4);
    }

    /// A frame loop whose frame time is the bill for the batch issued
    /// `latency` frames earlier, as on a real swapchain. Returns each frame's
    /// step count and duration.
    fn run_plant(
        t: &mut StepAutoTuner,
        frames: usize,
        latency: usize,
        overhead_ms: f32,
        cost_ms: f32,
        timed: bool,
    ) -> Vec<(u32, f32)> {
        let mut in_flight: VecDeque<u32> = std::iter::repeat(t.steps()).take(latency).collect();
        (0..frames)
            .map(|_| {
                let n = t.steps();
                in_flight.push_back(n);
                let billed = in_flight.pop_front().unwrap_or(n);
                let frame = overhead_ms + billed as f32 * cost_ms;
                t.update_measured(ms(frame), n, timed.then(|| ms(cost_ms)));
                (n, frame)
            })
            .collect()
    }

    #[test]
    fn splitting_the_budget_holds_steady_when_the_solver_is_most_of_the_frame() {
        // dx = 0.75 mm on the reference machine, measured: 4.4 ms a step and
        // 4.6 ms of everything else, each frame billed two frames after its
        // dispatch. The frame-time-only law swung between 1 and 17 steps here.
        let mut t = StepAutoTuner::default();
        let log = run_plant(&mut t, 300, 2, 4.6, 4.4, true);
        let tail = &log[240..];
        let (lo, hi) = tail.iter().fold((u32::MAX, 0), |(lo, hi), &(n, _)| (lo.min(n), hi.max(n)));
        assert!(hi - lo <= 1, "the step count still swings {lo}..{hi}");
        let mean = tail.iter().map(|&(_, f)| f).sum::<f32>() / tail.len() as f32;
        assert!(
            (mean - 16.67).abs() < 16.67 * 0.15,
            "settled at {mean:.1} ms/frame against a 16.7 ms budget"
        );
    }

    #[test]
    fn a_change_in_overhead_moves_the_count_without_a_long_overrun() {
        // Something heavier comes into view: 3 ms of overhead becomes 10 ms.
        let mut t = StepAutoTuner::default();
        run_plant(&mut t, 200, 2, 3.0, 1.0, true);
        let before = t.steps();
        let log = run_plant(&mut t, 200, 2, 10.0, 1.0, true);
        let after = t.steps();
        assert!(after < before, "the count did not fall: {before} -> {after}");
        assert!((after as f32 - 6.67).abs() <= 1.5, "settled at {after} steps, want about 6.7");
        let worst = log.iter().map(|&(_, f)| f).fold(0.0, f32::max);
        assert!(worst < 16.67 * 1.6, "worst frame {worst:.1} ms during the transition");
        let over = log.iter().filter(|&&(_, f)| f > 16.67 * 1.12).count();
        assert!(over < 40, "{over} frames over budget before the count came down");
    }

    #[test]
    fn one_hitched_frame_does_not_collapse_the_split_budget() {
        let mut t = StepAutoTuner::default();
        run_plant(&mut t, 200, 2, 3.0, 0.15, true);
        let before = t.steps();
        t.update_measured(ms(300.0), before, Some(ms(0.15)));
        let after = t.steps();
        assert!(
            after as f32 > before as f32 * 0.45,
            "one hitch took {before} steps down to {after}"
        );
    }

    #[test]
    fn a_paused_run_does_not_wind_the_count_up() {
        // Paused frames run no steps and come back fast. Read as headroom, they
        // walked the count up to max_steps, and the first frame after resuming
        // then dispatched thousands of steps at once.
        for timed in [false, true] {
            let mut t = StepAutoTuner::default();
            run_plant(&mut t, 200, 2, 4.6, 4.4, timed);
            let settled = t.steps();
            for _ in 0..200 {
                t.update_measured(ms(1.5), 0, timed.then(|| ms(4.4)));
            }
            assert!(
                t.steps() <= settled + 1,
                "paused for 200 frames (timed: {timed}), the count went {settled} -> {}",
                t.steps()
            );
        }
    }

    #[test]
    fn the_clock_is_honest_about_how_slow_the_physics_is() {
        // CONTRACT.md's quality tier: dx = 0.4 mm, ~70 steps/s, 200,000 steps
        // per physical second. That is 1/2857 of real time, and the status line
        // must say so rather than rounding to "0.00x".
        let c = Clock {
            sim_time_s: 0.0035,
            wall_time_s: 10.0,
            steps: 700,
            steps_per_second: 70.0,
            steps_per_physical_second: 200_000.0,
            fps: 60.0,
            steps_per_frame: 1,
        };
        let f = c.realtime_factor();
        assert!((f - 3.5e-4).abs() < 1e-6, "factor was {f}");
        let line = c.status_line();
        assert!(line.contains("1/2857 real-time"), "{line}");
        assert!(line.contains("per simulated second"), "{line}");
        assert!(!line.contains("0.00x"), "the misleading rounding is back: {line}");
        assert!(
            (c.wall_seconds_per_sim_second() - 2857.0).abs() < 2.0,
            "{}",
            c.wall_seconds_per_sim_second()
        );
    }

    #[test]
    fn a_clock_with_no_measurement_yet_reports_dashes_not_zero() {
        let line = Clock::default().status_line();
        assert!(line.contains("x real-time --"), "{line}");
    }

    #[test]
    fn the_rate_meter_reports_the_recent_rate_not_the_lifetime_one() {
        let mut m = RateMeter::new(1.0);
        // One slow second, then four fast ones.
        m.tick(5.0, 1.0);
        assert!((m.rate() - 5.0).abs() < 1e-9);
        for _ in 0..4 {
            m.tick(500.0, 1.0);
        }
        assert!((m.rate() - 500.0).abs() < 1e-6, "rate was {}", m.rate());
    }

    #[test]
    fn the_rate_meter_is_zero_until_a_window_closes() {
        let mut m = RateMeter::new(1.0);
        m.tick(100.0, 0.2);
        assert_eq!(m.rate(), 0.0, "an unfinished window must not be extrapolated");
    }
}

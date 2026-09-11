//! Running statistics, error bars and convergence.
//!
//! # Why every number carries an error bar
//!
//! A turbulent duct never reaches a steady state. The instantaneous flow rate,
//! pressure drop and uniformity all fluctuate forever, so a single instantaneous
//! reading is a sample from a distribution, not a measurement. Quoting
//! `dp = 47.31 Pa` from one frame is a statement about that frame and nothing
//! else. Quoting `dp = 47.3 +/- 0.6 Pa` is a statement about the duct.
//!
//! So every scalar the metrics layer reports goes through [`Series`], which
//! keeps a Welford running mean and variance and turns them into an
//! [`Estimate`] with a standard error of the mean.
//!
//! # Why the naive SEM is wrong here
//!
//! `SEM = sigma / sqrt(N)` assumes independent samples. Successive solver steps
//! are anything but: at the interactive tier one step advances the physical
//! clock by a few microseconds, so a thousand consecutive samples describe
//! essentially the same instant of the flow. Using `N` directly would make the
//! error bar shrink like `1/sqrt(N)` while the underlying uncertainty did not
//! move at all, and the tool would claim four significant figures it does not
//! have.
//!
//! The fix is the integrated autocorrelation time. With
//! `rho_k = corr(x_i, x_{i+k})`,
//!
//! ```text
//! tau_int = 1/2 + sum_{k>=1} rho_k
//! N_eff   = N / (1 + 2 * sum_{k>=1} rho_k) = N / (2 * tau_int)
//! SEM     = sigma / sqrt(N_eff)
//! ```
//!
//! The sum is truncated by Sokal's initial-positive-sequence rule: stop at the
//! first `k` where the estimated `rho_k` goes non-positive, because past that
//! point the estimator is dominated by its own noise and summing further makes
//! the answer worse, not better.
//!
//! [`Series::n_eff`] is checked against an AR(1) process in the tests, where
//! `rho_k = phi^k` gives the closed form `N_eff = N * (1 - phi) / (1 + phi)`.
//!
//! # Reading a residual plateau
//!
//! [`Monitor`] tracks a pseudo-residual `R = ||u^{n+1} - u^n|| / ||u^n||`. In a
//! *steady* flow it falls monotonically toward zero. In a genuinely unsteady one
//! — which every duct at Re > ~2000 is, and this one runs at Re = 1,600-15,500 —
//! it falls for a few flow-through times and then flattens at some finite value.
//!
//! **That plateau is not a failure to converge.** It is the amplitude of the
//! real, physical unsteadiness, and it is the signal to stop waiting for a
//! steady answer and start time-averaging instead. Misreading it as
//! non-convergence — and responding by lowering the time step, tightening
//! tolerances or refining the grid — is the single most common way to waste a
//! day on a CFD run. [`Monitor`] therefore never reports `Diverged` for a flat
//! residual; it reports [`Health::Converged`] as soon as the *statistics* are
//! stationary, which is the question that actually matters.

use std::collections::VecDeque;

/// Welford's online mean and variance.
///
/// The textbook `sum(x^2) - n*mean^2` form loses catastrophic precision when the
/// mean is large compared with the spread, which is exactly the case for a
/// pressure that hovers around 47 Pa with 0.3 Pa of fluctuation. Welford is
/// numerically stable and costs one extra multiply.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Welford {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }

    pub fn count(&self) -> u64 {
        self.n
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Sample variance, with the Bessel correction.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// A mean with an honest error bar.
///
/// `Display` renders `mean +/- sem` at a precision derived from the error bar
/// itself, so the number never shows digits the statistics do not support.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    pub mean: f64,
    /// Standard error of the mean, corrected for autocorrelation.
    pub sem: f64,
    pub std_dev: f64,
    /// Raw sample count.
    pub n: u64,
    /// Effective (independent) sample count, `N / (1 + 2*sum rho_k)`.
    pub n_eff: f64,
    /// Integrated autocorrelation time, `1/2 + sum rho_k`. 0.5 means the samples
    /// are independent; 50 means fifty consecutive samples are worth one.
    pub tau_int: f64,
}

impl Estimate {
    pub const ZERO: Self = Self { mean: 0.0, sem: 0.0, std_dev: 0.0, n: 0, n_eff: 0.0, tau_int: 0.5 };

    /// Relative error, `SEM / |mean|`. Infinite for a mean of zero, which is the
    /// honest answer: a relative error on nothing is not defined.
    pub fn relative_error(&self) -> f64 {
        if self.mean.abs() > 0.0 {
            self.sem / self.mean.abs()
        } else {
            f64::INFINITY
        }
    }

    /// Number of decimal places worth printing: enough to show one significant
    /// figure of the error bar, capped so a tiny error bar does not produce a
    /// wall of digits.
    fn decimals(&self) -> usize {
        if !(self.sem > 0.0) || !self.sem.is_finite() {
            return 3;
        }
        let d = -(self.sem.log10().floor()) as i32 + 0;
        d.clamp(0, 6) as usize
    }
}

impl std::fmt::Display for Estimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.decimals();
        write!(f, "{:.*} +/- {:.*}", d, self.mean, d, self.sem)
    }
}

/// One monitored scalar: a Welford accumulator plus enough recent history to
/// estimate the autocorrelation and to test for drift.
#[derive(Debug, Clone)]
pub struct Series {
    welford: Welford,
    /// Most recent samples, oldest first. Capacity bounds the memory a long run
    /// can use; the Welford mean is over *every* sample regardless.
    history: VecDeque<f64>,
    capacity: usize,
    /// Samples fed in since the last [`Series::reset`], including any that
    /// scrolled out of `history`.
    seen: u64,
}

impl Series {
    /// `capacity` is how many recent samples to keep for the autocorrelation and
    /// drift tests. 4096 is plenty: the autocorrelation estimate is unusable
    /// past a lag of about `capacity / 50` anyway.
    pub fn new(capacity: usize) -> Self {
        Self {
            welford: Welford::new(),
            history: VecDeque::with_capacity(capacity.max(2)),
            capacity: capacity.max(2),
            seen: 0,
        }
    }

    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.welford.push(x);
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(x);
        self.seen += 1;
    }

    pub fn count(&self) -> u64 {
        self.welford.count()
    }

    pub fn mean(&self) -> f64 {
        self.welford.mean()
    }

    pub fn std_dev(&self) -> f64 {
        self.welford.std_dev()
    }

    pub fn last(&self) -> Option<f64> {
        self.history.back().copied()
    }

    pub fn history(&self) -> impl Iterator<Item = &f64> {
        self.history.iter()
    }

    pub fn reset(&mut self) {
        self.welford.reset();
        self.history.clear();
        self.seen = 0;
    }

    /// Normalised autocorrelation at lag `k`, estimated over the retained
    /// history with the *history* mean (not the lifetime mean — using a mean the
    /// window did not produce biases every lag in the same direction).
    pub fn autocorrelation(&self, k: usize) -> f64 {
        let n = self.history.len();
        if k == 0 {
            return 1.0;
        }
        if n < k + 2 {
            return 0.0;
        }
        let mean = self.history.iter().sum::<f64>() / n as f64;
        let mut c0 = 0.0;
        for x in &self.history {
            let d = x - mean;
            c0 += d * d;
        }
        if c0 <= 0.0 {
            return 0.0;
        }
        let mut ck = 0.0;
        for i in 0..n - k {
            ck += (self.history[i] - mean) * (self.history[i + k] - mean);
        }
        // Both sums are divided by n (the biased estimator). That is deliberate:
        // the biased form tapers rho_k toward zero at large lag, which keeps the
        // truncated sum from being dominated by noisy high-lag terms.
        ck / c0
    }

    /// Integrated autocorrelation time, `tau_int = 1/2 + sum_{k>=1} rho_k`,
    /// truncated by Sokal's initial-positive-sequence rule.
    ///
    /// Returns 0.5 (independent samples) when there is not enough history to
    /// estimate anything, which is the conservative direction only in the sense
    /// that it is the *assumption* the naive SEM already makes — so nothing gets
    /// worse than the status quo while the run is warming up.
    pub fn tau_int(&self) -> f64 {
        let n = self.history.len();
        if n < 16 {
            return 0.5;
        }
        // Never look further than n/8: past that the estimator has fewer than
        // eight independent lag products and is pure noise.
        let max_lag = (n / 8).max(1);
        let mut sum = 0.0;
        for k in 1..=max_lag {
            let r = self.autocorrelation(k);
            if r <= 0.0 {
                break;
            }
            sum += r;
        }
        0.5 + sum
    }

    /// Effective sample count, `N / (2 * tau_int)`, clamped to `[1, N]`.
    pub fn n_eff(&self) -> f64 {
        let n = self.welford.count() as f64;
        if n <= 1.0 {
            return n;
        }
        (n / (2.0 * self.tau_int())).clamp(1.0, n)
    }

    /// The number to report.
    pub fn estimate(&self) -> Estimate {
        let n = self.welford.count();
        if n == 0 {
            return Estimate::ZERO;
        }
        let n_eff = self.n_eff();
        let sd = self.welford.std_dev();
        Estimate {
            mean: self.welford.mean(),
            sem: if n_eff > 0.0 { sd / n_eff.sqrt() } else { f64::INFINITY },
            std_dev: sd,
            n,
            n_eff,
            tau_int: self.tau_int(),
        }
    }

    /// Relative drift between the mean of the last `w` samples and the mean of
    /// the `w` before those, normalised by the overall mean.
    ///
    /// This is the stationarity test. A series whose two halves agree to better
    /// than half a percent has stopped moving in any way that matters, whatever
    /// the residual is doing.
    pub fn window_drift(&self, w: usize) -> Option<f64> {
        let n = self.history.len();
        if w == 0 || n < 2 * w {
            return None;
        }
        let recent: f64 = self.history.iter().skip(n - w).sum::<f64>() / w as f64;
        let prev: f64 = self.history.iter().skip(n - 2 * w).take(w).sum::<f64>() / w as f64;
        let scale = self.welford.mean().abs().max(1e-30);
        Some((recent - prev).abs() / scale)
    }
}

/// Traffic-light state for one monitored scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    /// Not enough samples yet to say anything.
    Unknown,
    /// Still in the startup transient: fewer than
    /// [`MonitorConfig::discard_flow_throughs`] flow-throughs have passed, so
    /// the average is contaminated by the initial condition.
    Transient,
    /// Stationary in the mean but the error bar is still too wide to quote.
    Converging,
    /// Stationary and precise. Quote it.
    Converged,
    /// Produced a non-finite value. The solver has blown up; nothing downstream
    /// means anything.
    Diverged,
}

impl Health {
    /// A colour name for the UI, so the mapping lives in one place.
    pub fn traffic_light(self) -> &'static str {
        match self {
            Health::Converged => "green",
            Health::Converging => "amber",
            Health::Transient | Health::Unknown => "grey",
            Health::Diverged => "red",
        }
    }

    pub fn is_quotable(self) -> bool {
        self == Health::Converged
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MonitorConfig {
    /// Half-window for the drift test, in samples. The test compares the mean of
    /// the last `window` samples with the mean of the `window` before those, so
    /// `2 * window` samples must have accumulated before it says anything.
    pub window: usize,
    /// Recent samples retained per series, for autocorrelation and drift.
    pub history: usize,
    /// Flow-through times to discard before the average is trusted. Literature
    /// practice is 3-5; the transient in a bend is mostly gone by 3.
    pub discard_flow_throughs: f64,
    /// Flow-through times to average over before the answer is publishable.
    /// 10-20 is the usual recommendation.
    pub average_flow_throughs: f64,
    /// Stationarity threshold on `|mean(last W) - mean(prev W)| / |mean|`.
    pub drift_tolerance: f64,
    /// Precision threshold on `SEM / |mean|`.
    pub sem_tolerance: f64,
    /// Above this, the mass balance is broken and *nothing* is trustworthy.
    pub mass_imbalance_tolerance: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            window: 64,
            history: 4096,
            discard_flow_throughs: 3.0,
            average_flow_throughs: 15.0,
            drift_tolerance: 0.005,
            sem_tolerance: 0.005,
            mass_imbalance_tolerance: 0.01,
        }
    }
}

/// Tracks every reported scalar, decides when the run is converged, and — most
/// importantly — **throws the whole average away when a boundary condition
/// changes**.
///
/// # The reset problem
///
/// If the user drags the inlet velocity slider from 3 m/s to 5 m/s and the
/// averages keep accumulating, the tool will report a confident mean of two
/// different flows with a small error bar, because the error bar only measures
/// scatter, not bias. That is worse than reporting nothing: it looks converged.
///
/// Calling [`Monitor::reset`] by hand is easy to forget, so the primary
/// interface is [`Monitor::set_parameters`], which takes a hash of everything
/// that defines the operating point and resets automatically when it changes.
/// Build the hash with [`ParameterHash`]. The application should call it every
/// frame; it is a `u64` comparison when nothing has moved.
///
/// [`Monitor::take_reset_notice`] then lets the UI toast the user, so a reset is
/// visible rather than silent.
#[derive(Debug, Clone)]
pub struct Monitor {
    entries: Vec<(String, Series)>,
    config: MonitorConfig,
    param_hash: Option<u64>,
    reset_count: u64,
    reset_notice: bool,
    flow_throughs: f64,
    mass_imbalance: f64,
    diverged: bool,
}

impl Default for Monitor {
    fn default() -> Self {
        Self::new(MonitorConfig::default())
    }
}

impl Monitor {
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
            param_hash: None,
            reset_count: 0,
            reset_notice: false,
            flow_throughs: 0.0,
            mass_imbalance: 0.0,
            diverged: false,
        }
    }

    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: MonitorConfig) {
        self.config = config;
    }

    /// Feed one sample of a named scalar. Unknown names are created on demand,
    /// so callers do not have to declare their metrics up front.
    ///
    /// A non-finite value marks the whole monitor diverged rather than being
    /// silently dropped: a NaN in the flow rate means the solver has failed, and
    /// quietly continuing to average the finite samples around it would hide
    /// that.
    pub fn observe(&mut self, name: &str, value: f64) {
        if !value.is_finite() {
            self.diverged = true;
            return;
        }
        match self.entries.iter_mut().find(|(n, _)| n == name) {
            Some((_, s)) => s.push(value),
            None => {
                let mut s = Series::new(self.config.history);
                s.push(value);
                self.entries.push((name.to_string(), s));
            }
        }
    }

    pub fn series(&self, name: &str) -> Option<&Series> {
        self.entries.iter().find(|(n, _)| n == name).map(|(_, s)| s)
    }

    pub fn estimate(&self, name: &str) -> Option<Estimate> {
        self.series(name).map(|s| s.estimate())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _)| n.as_str())
    }

    /// Simulated flow-through times, `t * V_bulk / L`. The unit engineers reason
    /// in: "average over 15 flow-throughs" is a statement everyone understands,
    /// where "average over 400,000 steps" depends on `dx`, `u_lb` and the tier.
    pub fn set_flow_throughs(&mut self, t: f64) {
        self.flow_throughs = t;
    }

    pub fn flow_throughs(&self) -> f64 {
        self.flow_throughs
    }

    /// `|Q_in - Q_out| / Q_in`. The gate on everything else.
    pub fn set_mass_imbalance(&mut self, i: f64) {
        self.mass_imbalance = i;
    }

    pub fn mass_imbalance(&self) -> f64 {
        self.mass_imbalance
    }

    pub fn mass_balance_ok(&self) -> bool {
        self.mass_imbalance.is_finite()
            && self.mass_imbalance < self.config.mass_imbalance_tolerance
    }

    /// Traffic light for one scalar.
    pub fn health(&self, name: &str) -> Health {
        if self.diverged {
            return Health::Diverged;
        }
        let Some(s) = self.series(name) else { return Health::Unknown };
        if s.count() < 8 {
            return Health::Unknown;
        }
        if self.flow_throughs < self.config.discard_flow_throughs {
            return Health::Transient;
        }
        let est = s.estimate();
        let Some(drift) = s.window_drift(self.config.window) else {
            return Health::Converging;
        };
        let stationary = drift < self.config.drift_tolerance;
        let precise = est.relative_error() < self.config.sem_tolerance;
        if stationary && precise && self.mass_balance_ok() {
            Health::Converged
        } else {
            Health::Converging
        }
    }

    /// Why the light is not green, in words.
    ///
    /// # Why this exists
    ///
    /// [`Health::Converged`] requires three separate things at once: the running
    /// mean has stopped drifting, its standard error is small enough, and the
    /// mass balance closes. Collapsing all three into one amber lamp hides the
    /// only thing the user needs in order to act, because the remedies are
    /// opposite:
    ///
    /// - *still drifting* or *error bar too wide* -> *run it longer*
    /// - *mass imbalance* -> **running longer will never help**; the boundary
    ///   conditions are leaking and no amount of averaging fixes that
    ///
    /// Without this distinction a run that has statistically converged but has a
    /// 5.6% mass imbalance looks identical to one that simply needs more steps,
    /// and the honest response to the first is to stop and fix the domain.
    /// Returns `None` when the metric is converged and there is nothing to say.
    pub fn explain(&self, name: &str) -> Option<String> {
        match self.health(name) {
            Health::Converged => None,
            Health::Diverged => Some("the field went non-finite".to_string()),
            Health::Unknown => Some("not enough samples yet".to_string()),
            Health::Transient => Some(format!(
                "still in the startup transient ({:.1} of {:.1} flow-throughs discarded)",
                self.flow_throughs, self.config.discard_flow_throughs
            )),
            Health::Converging => {
                let mut why: Vec<String> = Vec::new();
                if let Some(s) = self.series(name) {
                    match s.window_drift(self.config.window) {
                        Some(d) if d >= self.config.drift_tolerance => why.push(format!(
                            "the mean is still drifting ({:.2}% over the last window, want < {:.2}%)",
                            d * 100.0,
                            self.config.drift_tolerance * 100.0
                        )),
                        None => why.push("not enough history to measure drift".to_string()),
                        _ => {}
                    }
                    let e = s.estimate().relative_error();
                    if e >= self.config.sem_tolerance {
                        why.push(format!(
                            "the error bar is still wide ({:.2}%, want < {:.2}%)",
                            e * 100.0,
                            self.config.sem_tolerance * 100.0
                        ));
                    }
                }
                if !self.mass_balance_ok() {
                    // Deliberately worded as a dead end, because it is one.
                    why.push(format!(
                        "mass imbalance is {:.2}% (want < {:.2}%) -- running longer will \
                         NOT fix this, the domain is leaking",
                        self.mass_imbalance * 100.0,
                        self.config.mass_imbalance_tolerance * 100.0
                    ));
                }
                if why.is_empty() {
                    None
                } else {
                    Some(why.join("; "))
                }
            }
        }
    }

    /// True when every statistical test passes and only the mass balance is
    /// holding the light amber.
    ///
    /// This is the "stop running, go fix the boundary conditions" signal.
    pub fn statistically_converged_but_unbalanced(&self) -> bool {
        if self.mass_balance_ok() || self.diverged || self.entries.is_empty() {
            return false;
        }
        self.entries.iter().all(|(name, _)| {
            let Some(s) = self.series(name) else { return false };
            if s.count() < 8 || self.flow_throughs < self.config.discard_flow_throughs {
                return false;
            }
            let drifting = match s.window_drift(self.config.window) {
                Some(d) => d >= self.config.drift_tolerance,
                None => true,
            };
            !drifting && s.estimate().relative_error() < self.config.sem_tolerance
        })
    }

    /// Worst state across every monitored scalar. This is the light the status
    /// bar should show.
    pub fn overall(&self) -> Health {
        if self.diverged {
            return Health::Diverged;
        }
        if self.entries.is_empty() {
            return Health::Unknown;
        }
        let mut worst = Health::Converged;
        for (name, _) in &self.entries {
            let h = self.health(name);
            worst = match (worst, h) {
                (_, Health::Diverged) | (Health::Diverged, _) => Health::Diverged,
                (_, Health::Unknown) | (Health::Unknown, _) => Health::Unknown,
                (_, Health::Transient) | (Health::Transient, _) => Health::Transient,
                (_, Health::Converging) | (Health::Converging, _) => Health::Converging,
                _ => Health::Converged,
            };
        }
        worst
    }

    /// Have we averaged long enough to publish, by the flow-through rule?
    pub fn averaged_long_enough(&self) -> bool {
        self.flow_throughs
            >= self.config.discard_flow_throughs + self.config.average_flow_throughs
    }

    /// Throw every average away. Call this whenever the physics changes.
    pub fn reset(&mut self) {
        for (_, s) in self.entries.iter_mut() {
            s.reset();
        }
        self.flow_throughs = 0.0;
        self.diverged = false;
        self.reset_count += 1;
        self.reset_notice = true;
    }

    /// Compare `hash` against the last one seen; reset and return `true` if it
    /// changed. The first call only records the hash.
    pub fn set_parameters(&mut self, hash: u64) -> bool {
        match self.param_hash {
            Some(h) if h == hash => false,
            Some(_) => {
                self.param_hash = Some(hash);
                self.reset();
                true
            }
            None => {
                self.param_hash = Some(hash);
                false
            }
        }
    }

    pub fn reset_count(&self) -> u64 {
        self.reset_count
    }

    /// One-shot flag for the UI: true exactly once after each reset, so the app
    /// can toast "averages cleared: boundary condition changed" instead of
    /// leaving the user wondering why the error bars grew.
    pub fn take_reset_notice(&mut self) -> bool {
        std::mem::replace(&mut self.reset_notice, false)
    }
}

/// Builder for the parameter hash that drives [`Monitor::set_parameters`].
///
/// Deliberately not `std::hash::Hash` on some config struct: the point is to be
/// *explicit* about which quantities invalidate an average, so that adding a
/// field to a config elsewhere cannot silently start or stop triggering resets.
/// Floats go in through their bit patterns, so `-0.0` and `0.0` are distinct and
/// a NaN is stable — both are fine, because this is an equality test, not an
/// ordering.
#[derive(Debug, Clone, Copy)]
pub struct ParameterHash(u64);

impl Default for ParameterHash {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterHash {
    pub fn new() -> Self {
        // FNV-1a 64-bit offset basis.
        Self(0xcbf2_9ce4_8422_2325)
    }

    pub fn u64(mut self, v: u64) -> Self {
        for b in v.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self
    }

    pub fn f64(self, v: f64) -> Self {
        self.u64(v.to_bits())
    }

    pub fn f32(self, v: f32) -> Self {
        self.u64(v.to_bits() as u64)
    }

    pub fn vec3(self, v: glam::Vec3) -> Self {
        self.f32(v.x).f32(v.y).f32(v.z)
    }

    pub fn str(mut self, s: &str) -> Self {
        for b in s.as_bytes() {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self
    }

    pub fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic normal deviates. A seeded xorshift plus Box-Muller, so the
    /// statistical tests below are reproducible and cannot flake.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn uniform(&mut self) -> f64 {
            // (0, 1), never exactly 0, so the log below is finite.
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-15
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform();
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    #[test]
    fn welford_matches_a_direct_two_pass_computation() {
        let mut rng = Rng(0x1234_5678);
        // A large offset is the case the naive sum-of-squares formula fails on.
        let xs: Vec<f64> = (0..5000).map(|_| 1.0e6 + rng.normal()).collect();

        let mut w = Welford::new();
        for x in &xs {
            w.push(*x);
        }

        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);

        // Relative, not absolute: with a mean of 1e6 the f64 unit in the last
        // place is 1.2e-10, so an absolute 1e-9 would be asserting that two
        // different summation orders agree to eight ulps. 1e-12 relative is
        // 1e-6 in absolute terms and still tight enough to catch the naive
        // sum-of-squares formula, which is out by orders of magnitude here.
        assert!(
            (w.mean() - mean).abs() / mean.abs() < 1e-12,
            "{} vs {mean}",
            w.mean()
        );
        assert!(
            (w.variance() - var).abs() / var < 1e-9,
            "{} vs {var}",
            w.variance()
        );
    }

    #[test]
    fn sem_of_independent_samples_matches_sigma_over_sqrt_n() {
        // White noise: rho_k = 0 for k >= 1, so tau_int = 1/2 and N_eff = N.
        let mut rng = Rng(0xdead_beef);
        let mut s = Series::new(4096);
        for _ in 0..20_000 {
            s.push(rng.normal());
        }
        let e = s.estimate();
        let naive = s.std_dev() / (s.count() as f64).sqrt();
        assert!(
            (e.n_eff / s.count() as f64 - 1.0).abs() < 0.15,
            "white noise should be nearly independent, N_eff/N = {}",
            e.n_eff / s.count() as f64
        );
        assert!(
            (e.sem / naive - 1.0).abs() < 0.1,
            "sem {} vs naive {naive}",
            e.sem
        );
    }

    #[test]
    fn autocorrelation_correction_recovers_the_analytic_ar1_result() {
        // AR(1): x_{i+1} = phi*x_i + sqrt(1-phi^2)*eps.
        // rho_k = phi^k exactly, so 1 + 2*sum_{k>=1} phi^k = (1+phi)/(1-phi)
        // and N_eff = N * (1-phi)/(1+phi).
        for phi in [0.5_f64, 0.8, 0.9] {
            let mut rng = Rng(0xa5a5_0000 + (phi * 1000.0) as u64);
            let mut s = Series::new(8192);
            let mut x = 0.0;
            let sigma = (1.0 - phi * phi).sqrt();
            // Burn in so the process starts stationary.
            for _ in 0..2000 {
                x = phi * x + sigma * rng.normal();
            }
            for _ in 0..100_000 {
                x = phi * x + sigma * rng.normal();
                s.push(x);
            }

            let want_ratio = (1.0 - phi) / (1.0 + phi);
            let got_ratio = s.n_eff() / s.count() as f64;
            assert!(
                (got_ratio / want_ratio - 1.0).abs() < 0.25,
                "phi = {phi}: N_eff/N = {got_ratio:.4}, analytic {want_ratio:.4}"
            );

            // And the correction must actually widen the error bar: for phi=0.9
            // the naive SEM understates by sqrt(19) ~ 4.4x.
            let naive = s.std_dev() / (s.count() as f64).sqrt();
            let ratio = s.estimate().sem / naive;
            let want = (1.0 / want_ratio).sqrt();
            assert!(
                (ratio / want - 1.0).abs() < 0.2,
                "phi = {phi}: SEM inflated {ratio:.3}x, analytic {want:.3}x"
            );
        }
    }

    #[test]
    fn autocorrelation_at_lag_zero_is_one_and_decays_like_phi_to_the_k() {
        let mut rng = Rng(7);
        let mut s = Series::new(8192);
        let phi = 0.7;
        let mut x = 0.0;
        for _ in 0..20_000 {
            x = phi * x + (1.0f64 - phi * phi).sqrt() * rng.normal();
            s.push(x);
        }
        assert_eq!(s.autocorrelation(0), 1.0);
        for k in 1..5 {
            let want = phi.powi(k as i32);
            let got = s.autocorrelation(k);
            assert!((got - want).abs() < 0.06, "lag {k}: {got:.4} vs {want:.4}");
        }
    }

    #[test]
    fn a_drifting_series_is_never_reported_converged() {
        let mut m = Monitor::new(MonitorConfig { window: 32, ..Default::default() });
        m.set_flow_throughs(10.0);
        // A slow ramp: the error bar can be small while the mean is still moving.
        for i in 0..2000 {
            m.observe("dp", 40.0 + i as f64 * 0.01);
        }
        assert_eq!(m.health("dp"), Health::Converging, "a ramp is not converged");

        // Now hold it steady with a little noise and it must go green.
        let mut rng = Rng(99);
        let mut m2 = Monitor::new(MonitorConfig { window: 32, ..Default::default() });
        m2.set_flow_throughs(10.0);
        m2.set_mass_imbalance(0.001);
        for _ in 0..4000 {
            m2.observe("dp", 47.3 + 0.3 * rng.normal());
        }
        assert_eq!(m2.health("dp"), Health::Converged);
    }

    #[test]
    fn broken_mass_balance_blocks_convergence_however_steady_everything_else_is() {
        let mut rng = Rng(5);
        let mut m = Monitor::default();
        m.set_flow_throughs(20.0);
        m.set_mass_imbalance(0.05); // 5%: way over the 1% gate
        for _ in 0..4000 {
            m.observe("dp", 47.3 + 0.05 * rng.normal());
        }
        assert!(!m.mass_balance_ok());
        assert_eq!(m.health("dp"), Health::Converging);
    }

    #[test]
    fn a_changed_parameter_hash_clears_the_average() {
        let a = ParameterHash::new().f32(3.0).str("inlet").finish();
        let b = ParameterHash::new().f32(5.0).str("inlet").finish();
        assert_ne!(a, b);

        let mut m = Monitor::default();
        assert!(!m.set_parameters(a), "the first hash must not count as a change");
        for _ in 0..500 {
            m.observe("q", 10.0);
        }
        assert_eq!(m.series("q").unwrap().count(), 500);

        assert!(m.set_parameters(b), "a different hash must reset");
        assert_eq!(m.series("q").unwrap().count(), 0, "the average survived a parameter change");
        assert!(m.take_reset_notice());
        assert!(!m.take_reset_notice(), "the notice must be one-shot");

        // Same hash again: no reset.
        assert!(!m.set_parameters(b));
    }

    #[test]
    fn averaging_across_a_step_change_would_have_been_badly_wrong() {
        // The bug this whole mechanism exists to prevent, demonstrated.
        let mut naive = Series::new(4096);
        for _ in 0..1000 {
            naive.push(30.0);
        }
        for _ in 0..1000 {
            naive.push(50.0);
        }
        // ...and it looks confident: the reported mean of 40 is 10 away from
        // both of the two values the series actually took, and the error bar is
        // smaller than that gap, so nothing about the number warns you. (The
        // autocorrelation correction does widen it a long way — a step change
        // has rho_k near 1 out to enormous lag — which is exactly the point of
        // that correction, but it still cannot turn bias into visible error.)
        let e = naive.estimate();
        assert!((e.mean - 40.0).abs() < 1e-9, "mean was {}", e.mean);
        assert!(
            e.sem < 10.0,
            "sem {} still understates the 10-unit distance to either true value",
            e.sem
        );

        // With the monitor it cannot happen.
        let mut m = Monitor::default();
        m.set_parameters(1);
        for _ in 0..1000 {
            m.observe("dp", 30.0);
        }
        m.set_parameters(2);
        for _ in 0..1000 {
            m.observe("dp", 50.0);
        }
        assert!((m.estimate("dp").unwrap().mean - 50.0).abs() < 1e-9);
    }

    #[test]
    fn a_non_finite_sample_marks_the_run_diverged() {
        let mut m = Monitor::default();
        m.set_flow_throughs(10.0);
        for _ in 0..100 {
            m.observe("q", 1.0);
        }
        m.observe("q", f64::NAN);
        assert_eq!(m.overall(), Health::Diverged);
        assert_eq!(Health::Diverged.traffic_light(), "red");
    }

    #[test]
    fn a_plateaued_residual_in_an_unsteady_flow_still_converges() {
        // The misreading this module's docs warn about: R flattens at 2e-3 and
        // never falls further, but every reported scalar is stationary. That is
        // a converged time-average, not a failed run.
        let mut rng = Rng(0x5eed);
        // The drift window has to be long enough that the *noise* on a window
        // mean is well inside the 0.5% drift tolerance. The residual fluctuates
        // by 2% of itself, so a 64-sample window would compare two means with
        // 0.35% of scatter each and report drift that is nothing but noise.
        // Widening the window is what a real run does too.
        let mut m = Monitor::new(MonitorConfig { window: 512, history: 8192, ..Default::default() });
        m.set_flow_throughs(20.0);
        m.set_mass_imbalance(0.002);
        for _ in 0..10_000 {
            m.observe("residual", 2.0e-3 * (1.0 + 0.02 * rng.normal()));
            m.observe("dp", 47.3 + 0.4 * rng.normal());
        }
        assert_eq!(m.overall(), Health::Converged);
    }

    #[test]
    fn estimate_display_shows_no_more_digits_than_the_error_bar_supports() {
        let e = Estimate { mean: 47.312_9, sem: 0.61, std_dev: 5.0, n: 100, n_eff: 20.0, tau_int: 2.5 };
        assert_eq!(format!("{e}"), "47.3 +/- 0.6");
        let e = Estimate { mean: 0.004_213, sem: 0.000_08, std_dev: 1.0, n: 10, n_eff: 5.0, tau_int: 1.0 };
        assert_eq!(format!("{e}"), "0.00421 +/- 0.00008");
    }

    #[test]
    fn window_drift_needs_two_full_windows_before_it_answers() {
        let mut s = Series::new(1024);
        for _ in 0..63 {
            s.push(1.0);
        }
        assert!(s.window_drift(32).is_none());
        s.push(1.0);
        assert!(s.window_drift(32).is_some());
    }

    /// A run whose statistics have settled but whose domain leaks must be
    /// distinguishable from one that simply needs more steps -- the first is a
    /// dead end and the second is not.
    #[test]
    fn a_leaking_domain_is_reported_as_a_dead_end_not_as_impatience() {
        let mut m = Monitor::new(MonitorConfig {
            discard_flow_throughs: 1.0,
            ..MonitorConfig::default()
        });
        m.set_flow_throughs(10.0);
        // A dead-steady scalar: no drift, negligible error bar.
        for _ in 0..400 {
            m.observe("dp", 65.0);
        }

        // Balanced: green, and nothing to explain.
        m.set_mass_imbalance(0.001);
        assert_eq!(m.health("dp"), Health::Converged);
        assert!(m.explain("dp").is_none());
        assert!(!m.statistically_converged_but_unbalanced());

        // Same statistics, leaking domain: amber, and the reason says so.
        m.set_mass_imbalance(0.0563);
        assert_eq!(m.health("dp"), Health::Converging);
        let why = m.explain("dp").expect("amber must explain itself");
        assert!(why.contains("mass imbalance"), "got {why:?}");
        assert!(why.contains("NOT fix this"), "the dead end must be explicit: {why:?}");
        assert!(
            m.statistically_converged_but_unbalanced(),
            "this is the stop-and-fix-the-domain signal"
        );
    }

    /// The other amber: genuinely still converging. Here running longer *is*
    /// the right answer, and the explanation must not blame the domain.
    #[test]
    fn a_still_settling_run_is_not_blamed_on_the_domain() {
        let mut m = Monitor::new(MonitorConfig {
            discard_flow_throughs: 1.0,
            ..MonitorConfig::default()
        });
        m.set_flow_throughs(10.0);
        m.set_mass_imbalance(0.001);
        // A ramp: the mean is still moving.
        for i in 0..400 {
            m.observe("dp", 10.0 + i as f64 * 0.5);
        }
        assert_ne!(m.health("dp"), Health::Converged);
        let why = m.explain("dp").expect("amber must explain itself");
        assert!(!why.contains("mass imbalance"), "the domain is fine here: {why:?}");
        assert!(!m.statistically_converged_but_unbalanced());
    }
}


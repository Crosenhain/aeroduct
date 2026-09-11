//! Residence time distribution, measured Lagrangianly.
//!
//! # Why Lagrangian
//!
//! The residence time distribution `E(t)` is *defined* as the age distribution
//! of fluid leaving the system. There is a Eulerian way to get at it — inject a
//! passive scalar, solve a transport equation, watch the outlet concentration —
//! but that costs another field, another set of boundary conditions and a
//! numerical diffusion error that directly fakes mixing. Tracers already exist
//! in this app for the streamline visualisation. Tag each one with a birth time,
//! histogram the ages of the ones that cross the outlet, and `E(t)` falls out
//! with no extra physics and no extra error.
//!
//! # The numbers
//!
//! With `w_i` the weight of exit sample `i` (its mass flux, or 1 for an
//! unweighted tracer) and `t_i` its age:
//!
//! ```text
//! E(t) dt = fraction of exiting fluid with age in [t, t+dt),  integral E = 1
//! t_bar   = sum(w_i t_i) / sum(w_i)          mean residence time
//! sigma^2 = sum(w_i t_i^2)/sum(w_i) - t_bar^2
//! tau_ideal = V / Q                          fluid volume over volumetric flow
//! dead volume fraction = 1 - t_bar / tau_ideal
//! ```
//!
//! `t_bar` is computed from the raw samples, not from the histogram, so the bin
//! width affects only the plotted curve and never the reported mean.
//!
//! # Reading the shape
//!
//! Two limits bracket everything real:
//!
//! * **Plug flow**: `E(t) = delta(t - tau)`, `sigma/t_bar = 0`. Every parcel
//!   takes the same route.
//! * **Perfectly stirred tank**: `E(t) = exp(-t/tau)/tau`, `sigma/t_bar = 1`.
//!
//! A duct should be near the plug-flow end. A long exponential tail with
//! `sigma/t_bar` approaching 1 means a recirculation zone is holding fluid,
//! which is the same defect [`crate::volume`] reports as reverse-flow volume,
//! seen from the other side. A **negative** dead-volume fraction means `t_bar`
//! exceeds `V/Q`, which usually means the tracer seeding is biased toward slow
//! near-wall fluid rather than that the duct is bigger than it is.

/// One tracer crossing the outlet.
///
/// Deliberately the whole interface to the particle system: the metrics layer
/// never touches tracer state, positions, or integration. Anything that can
/// produce `(age, weight)` pairs — the GPU particle system, a CPU streamline
/// integrator, a test — can drive this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgeSample {
    /// Time since the tracer was born, seconds.
    pub age_s: f64,
    /// How much fluid this tracer stands for. Use the local mass flux
    /// `rho * (u . n)` at the crossing when the seeding is uniform in *area*,
    /// or 1.0 when tracers were seeded in proportion to flux already.
    ///
    /// Getting this wrong is the classic RTD error: area-uniform seeding
    /// over-counts the slow near-wall fluid, which stretches the tail and
    /// invents dead volume that is not there.
    pub weight: f64,
}

impl AgeSample {
    pub fn new(age_s: f64, weight: f64) -> Self {
        Self { age_s, weight }
    }
}

/// The measured distribution plus everything derived from it.
#[derive(Debug, Clone, Default)]
pub struct Rtd {
    bins: Vec<f64>,
    bin_width_s: f64,
    /// Sum of weights, sum of w*t, sum of w*t^2 — the exact moments, kept
    /// independently of the histogram.
    total_weight: f64,
    sum_wt: f64,
    sum_wt2: f64,
    count: u64,
    max_age_s: f64,
    tau_ideal_s: f64,
}

/// Number of histogram bins. Fixed, because the bin *width* adapts instead: a
/// fixed bin count keeps the plot a predictable size for the UI.
pub const RTD_BINS: usize = 128;

impl Rtd {
    /// `tau_ideal_s = V / Q` seeds the initial bin width at `tau/16`, which puts
    /// a plug-flow spike near bin 16 of 128 and leaves eight ideal residence
    /// times of tail before the histogram has to grow.
    pub fn new(tau_ideal_s: f64) -> Self {
        let tau = if tau_ideal_s.is_finite() && tau_ideal_s > 0.0 { tau_ideal_s } else { 1.0 };
        Self {
            bins: vec![0.0; RTD_BINS],
            bin_width_s: tau / 16.0,
            total_weight: 0.0,
            sum_wt: 0.0,
            sum_wt2: 0.0,
            count: 0,
            max_age_s: 0.0,
            tau_ideal_s,
        }
    }

    /// Update `V/Q` after the flow rate has been measured. Does not disturb
    /// samples already recorded; only the derived `tau_ideal` and dead-volume
    /// fraction change.
    pub fn set_tau_ideal(&mut self, tau_ideal_s: f64) {
        self.tau_ideal_s = tau_ideal_s;
    }

    pub fn tau_ideal_s(&self) -> f64 {
        self.tau_ideal_s
    }

    /// Record one tracer exit.
    pub fn record(&mut self, s: AgeSample) {
        if !s.age_s.is_finite() || !s.weight.is_finite() || s.weight <= 0.0 || s.age_s < 0.0 {
            return;
        }
        self.total_weight += s.weight;
        self.sum_wt += s.weight * s.age_s;
        self.sum_wt2 += s.weight * s.age_s * s.age_s;
        self.count += 1;
        self.max_age_s = self.max_age_s.max(s.age_s);

        // Grow the histogram to fit, by doubling the bin width and merging
        // neighbouring bins. Doubling preserves every recorded weight exactly
        // (bin j and j+1 fold into bin j/2), so a late long-lived tracer costs
        // resolution but never data.
        while s.age_s >= self.bin_width_s * RTD_BINS as f64 {
            self.coarsen();
        }
        let b = (s.age_s / self.bin_width_s) as usize;
        self.bins[b.min(RTD_BINS - 1)] += s.weight;
    }

    pub fn record_all(&mut self, samples: impl IntoIterator<Item = AgeSample>) {
        for s in samples {
            self.record(s);
        }
    }

    fn coarsen(&mut self) {
        let mut merged = vec![0.0; RTD_BINS];
        for (i, w) in self.bins.iter().enumerate() {
            merged[i / 2] += w;
        }
        self.bins = merged;
        self.bin_width_s *= 2.0;
    }

    pub fn sample_count(&self) -> u64 {
        self.count
    }

    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    pub fn bin_width_s(&self) -> f64 {
        self.bin_width_s
    }

    pub fn max_age_s(&self) -> f64 {
        self.max_age_s
    }

    /// Mean residence time, from the exact weighted moment.
    pub fn mean_age_s(&self) -> f64 {
        if self.total_weight > 0.0 {
            self.sum_wt / self.total_weight
        } else {
            0.0
        }
    }

    /// Variance of the age distribution, seconds squared.
    pub fn variance_s2(&self) -> f64 {
        if self.total_weight <= 0.0 {
            return 0.0;
        }
        let m = self.mean_age_s();
        (self.sum_wt2 / self.total_weight - m * m).max(0.0)
    }

    /// `sigma / t_bar`. 0 is plug flow, 1 is a perfectly stirred tank, and above
    /// 1 means a strongly bimodal path — a short circuit plus a trapped zone.
    pub fn dispersion(&self) -> f64 {
        let m = self.mean_age_s();
        if m > 0.0 {
            self.variance_s2().sqrt() / m
        } else {
            0.0
        }
    }

    /// `1 - t_bar / tau_ideal`. Positive means part of the geometry is not
    /// participating in the flow.
    ///
    /// Returns `None` when `tau_ideal` is unknown, rather than a plausible-
    /// looking zero.
    pub fn dead_volume_fraction(&self) -> Option<f64> {
        if self.tau_ideal_s.is_finite() && self.tau_ideal_s > 0.0 && self.total_weight > 0.0 {
            Some(1.0 - self.mean_age_s() / self.tau_ideal_s)
        } else {
            None
        }
    }

    /// `E(t)` as `(bin centre in seconds, density in 1/s)` pairs, normalised so
    /// that `sum(E_i * dt) = 1`. This is what the UI plots.
    pub fn distribution(&self) -> Vec<(f64, f64)> {
        let norm = self.total_weight * self.bin_width_s;
        (0..RTD_BINS)
            .map(|i| {
                let t = (i as f64 + 0.5) * self.bin_width_s;
                let e = if norm > 0.0 { self.bins[i] / norm } else { 0.0 };
                (t, e)
            })
            .collect()
    }

    /// Cumulative distribution `F(t) = integral_0^t E`. `F(t_50)` = 0.5 gives the
    /// median residence time, which is more robust than the mean when the tail
    /// is long.
    pub fn cumulative(&self) -> Vec<(f64, f64)> {
        let mut acc = 0.0;
        (0..RTD_BINS)
            .map(|i| {
                acc += self.bins[i];
                let t = (i as f64 + 1.0) * self.bin_width_s;
                (t, if self.total_weight > 0.0 { acc / self.total_weight } else { 0.0 })
            })
            .collect()
    }

    /// Age below which fraction `f` of the exiting fluid lies, by linear
    /// interpolation inside the containing bin.
    pub fn quantile(&self, f: f64) -> Option<f64> {
        if self.total_weight <= 0.0 || !(0.0..=1.0).contains(&f) {
            return None;
        }
        let target = f * self.total_weight;
        let mut acc = 0.0;
        for (i, w) in self.bins.iter().enumerate() {
            if acc + w >= target {
                let within = if *w > 0.0 { (target - acc) / w } else { 0.0 };
                return Some((i as f64 + within) * self.bin_width_s);
            }
            acc += w;
        }
        Some(RTD_BINS as f64 * self.bin_width_s)
    }

    pub fn clear(&mut self) {
        let tau = self.tau_ideal_s;
        *self = Self::new(tau);
    }

    /// One line for the status bar.
    pub fn summary(&self) -> String {
        if self.total_weight <= 0.0 {
            return "residence time: no tracer exits recorded yet".into();
        }
        match self.dead_volume_fraction() {
            Some(d) => format!(
                "residence time {:.3} s (ideal {:.3} s, dead volume {:.1}%, sigma/t = {:.2}, n = {})",
                self.mean_age_s(),
                self.tau_ideal_s,
                d * 100.0,
                self.dispersion(),
                self.count
            ),
            None => format!(
                "residence time {:.3} s (sigma/t = {:.2}, n = {})",
                self.mean_age_s(),
                self.dispersion(),
                self.count
            ),
        }
    }
}

/// Ideal residence time `tau = V / Q`.
///
/// `volume_mm3` is the *fluid* volume (the passage, not the printed part), and
/// `flow_m3s` is the measured volumetric flow. Mixing millimetres and SI here is
/// the obvious trap, so the conversion lives in one function.
pub fn tau_ideal_s(volume_mm3: f64, flow_m3s: f64) -> f64 {
    if flow_m3s.abs() < 1e-12 {
        return f64::INFINITY;
    }
    (volume_mm3 * 1e-9) / flow_m3s.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plug_flow_gives_the_ideal_residence_time_and_no_dead_volume() {
        let tau = 0.25;
        let mut r = Rtd::new(tau);
        for _ in 0..1000 {
            r.record(AgeSample::new(tau, 1.0));
        }
        assert!((r.mean_age_s() - tau).abs() < 1e-12);
        assert!(r.dispersion() < 1e-6, "plug flow must have zero spread");
        assert!(r.dead_volume_fraction().unwrap().abs() < 1e-12);
    }

    #[test]
    fn a_stirred_tank_reproduces_its_analytic_moments() {
        // E(t) = exp(-t/tau)/tau has t_bar = tau and sigma = tau exactly.
        // Sample it deterministically by inverse-CDF at 200k evenly spaced
        // quantiles, so the test cannot flake.
        let tau = 0.4;
        let mut r = Rtd::new(tau);
        let n = 200_000;
        for i in 0..n {
            let p = (i as f64 + 0.5) / n as f64;
            r.record(AgeSample::new(-tau * (1.0 - p).ln(), 1.0));
        }
        assert!((r.mean_age_s() / tau - 1.0).abs() < 1e-3, "t_bar/tau = {}", r.mean_age_s() / tau);
        assert!((r.dispersion() - 1.0).abs() < 1e-2, "sigma/t_bar = {}", r.dispersion());
        // Median of an exponential is tau*ln 2.
        let median = r.quantile(0.5).unwrap();
        assert!(
            (median / (tau * std::f64::consts::LN_2) - 1.0).abs() < 0.02,
            "median {median} vs {}",
            tau * std::f64::consts::LN_2
        );
    }

    #[test]
    fn the_distribution_integrates_to_one() {
        let mut r = Rtd::new(1.0);
        for i in 0..5000 {
            r.record(AgeSample::new(0.5 + (i % 97) as f64 * 0.01, 1.0 + (i % 7) as f64));
        }
        let dt = r.bin_width_s();
        let total: f64 = r.distribution().iter().map(|(_, e)| e * dt).sum();
        assert!((total - 1.0).abs() < 1e-9, "integral E dt = {total}");
        let (_, f_last) = *r.cumulative().last().unwrap();
        assert!((f_last - 1.0).abs() < 1e-9, "F(inf) = {f_last}");
    }

    #[test]
    fn dead_volume_is_the_shortfall_against_v_over_q() {
        // Fluid leaves after 0.6 of the ideal time: 40% of the volume is not
        // taking part.
        let tau = 1.0;
        let mut r = Rtd::new(tau);
        for _ in 0..100 {
            r.record(AgeSample::new(0.6, 1.0));
        }
        assert!((r.dead_volume_fraction().unwrap() - 0.4).abs() < 1e-12);
    }

    #[test]
    fn weighting_by_flux_changes_the_answer_and_that_is_the_point() {
        // Half the tracers are slow near-wall fluid carrying a tenth the flux.
        let mut area_weighted = Rtd::new(1.0);
        let mut flux_weighted = Rtd::new(1.0);
        for _ in 0..500 {
            area_weighted.record(AgeSample::new(1.0, 1.0));
            area_weighted.record(AgeSample::new(5.0, 1.0));
            flux_weighted.record(AgeSample::new(1.0, 1.0));
            flux_weighted.record(AgeSample::new(5.0, 0.1));
        }
        assert!((area_weighted.mean_age_s() - 3.0).abs() < 1e-9);
        assert!(
            (flux_weighted.mean_age_s() - 1.5 / 1.1).abs() < 1e-9,
            "flux-weighted mean was {}",
            flux_weighted.mean_age_s()
        );
        assert!(flux_weighted.mean_age_s() < area_weighted.mean_age_s());
    }

    #[test]
    fn a_long_lived_tracer_coarsens_the_histogram_without_losing_weight() {
        let mut r = Rtd::new(1.0);
        let w0 = r.bin_width_s();
        for _ in 0..100 {
            r.record(AgeSample::new(0.5, 1.0));
        }
        let mean_before = r.mean_age_s();
        // 400x the ideal time: recirculation, and far outside the initial range.
        r.record(AgeSample::new(400.0, 1.0));
        assert!(r.bin_width_s() > w0, "the histogram should have grown");
        assert_eq!(r.total_weight(), 101.0, "weight was lost in the rebin");
        let hist_weight: f64 = r.distribution().iter().map(|(_, e)| e * r.bin_width_s()).sum();
        assert!((hist_weight - 1.0).abs() < 1e-9);
        // The exact moments are untouched by binning.
        assert!(r.mean_age_s() > mean_before);
        assert!((r.mean_age_s() - (100.0 * 0.5 + 400.0) / 101.0).abs() < 1e-12);
    }

    #[test]
    fn nonsense_samples_are_ignored_rather_than_poisoning_the_moments() {
        let mut r = Rtd::new(1.0);
        r.record(AgeSample::new(1.0, 1.0));
        r.record(AgeSample::new(f64::NAN, 1.0));
        r.record(AgeSample::new(1.0, f64::INFINITY));
        r.record(AgeSample::new(-1.0, 1.0));
        r.record(AgeSample::new(1.0, 0.0));
        assert_eq!(r.sample_count(), 1);
        assert!(r.mean_age_s().is_finite());
    }

    #[test]
    fn tau_ideal_converts_millimetres_to_seconds() {
        // 10,000 mm^3 = 1e-5 m^3 at 1e-3 m^3/s (1 L/s) is 10 ms.
        assert!((tau_ideal_s(10_000.0, 1.0e-3) - 0.01).abs() < 1e-12);
        assert!(tau_ideal_s(1000.0, 0.0).is_infinite());
    }

    #[test]
    fn an_empty_distribution_reports_nothing_rather_than_zero() {
        let r = Rtd::new(1.0);
        assert_eq!(r.mean_age_s(), 0.0);
        assert!(r.dead_volume_fraction().is_none());
        assert!(r.quantile(0.5).is_none());
        assert!(r.summary().contains("no tracer exits"));
    }
}

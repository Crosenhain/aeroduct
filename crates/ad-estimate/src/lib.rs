//! Instant duct performance from 1D loss correlations: the answer while the
//! user is still dragging.
//!
//! # Why this exists
//!
//! The lattice-Boltzmann solver cannot be interactive and never will be. Its
//! timestep follows *acoustic* scaling — `dt = dx * u_lb / U` — so refining the
//! grid costs a fourth power of work, and measured on the contract's own
//! hardware it is **136x short of real time at 0.75 mm and 1682x short at
//! 0.4 mm**. No amount of kernel tuning closes a gap of that size.
//!
//! But the numbers a duct designer actually asks for — flow rate, pressure
//! drop, loss coefficient — are not obtained by simulation in industry. They
//! are obtained from one-dimensional loss correlations, in *microseconds*, and
//! have been since the 1950s. That is how every HVAC duct you have ever stood
//! under was sized.
//!
//! There is a sharper point than "it is a fast fallback". The LBM currently
//! reports `K = 12.5 +/- 0.9` for the test part, against an ASHRAE expectation
//! of order 1-3 for a mitred bend and a design target under 1. It is
//! under-resolved: at `dx = 0.75 mm` a 6.3 mm passage is nine cells across, and
//! nine cells cannot carry a turbulent boundary layer and a separation bubble at
//! the same time. **A correlation model may well be the more accurate of the
//! two right now.** This crate is a second opinion, not a consolation prize.
//!
//! # The two phases, and why they are separate
//!
//! ```text
//! centreline::extract_passage   geometry -> Passage      once per geometry edit  (~10-100 ms)
//! network::estimate             Passage  -> EstimateReport  every slider tick    (~10 us)
//! ```
//!
//! [`centreline::extract_passage`] does the expensive part: flood-fill the air
//! between the two mouths, walk it with a geodesic wavefront, and reduce it to a
//! few hundred stations carrying area, hydraulic diameter and aspect ratio. That
//! is a function of the *shape* only.
//!
//! [`network::estimate`] then evaluates the loss network over those stations.
//! It touches no voxels, allocates a handful of small vectors and is pure
//! arithmetic, which is what makes it microseconds rather than milliseconds. So
//! a velocity slider re-solves instantly, and only moving geometry re-extracts.
//!
//! # Every number carries a band
//!
//! Loss correlations are curve fits to wind-tunnel data taken on clean,
//! fully-developed, isolated fittings. Applied to a compact printed part they
//! are good to roughly **+/-20-30%**, and saying so is not a disclaimer, it is
//! the result. Every [`Band`] in this crate is a real uncertainty propagated
//! from per-element figures, with a floor at 20% because no stack of
//! correlations is better than its method.
//!
//! # What it deliberately cannot do
//!
//! A 1D loss network knows nothing about *where* the air separates, whether the
//! outlet jet is uniform, or what the flow looks like. It returns a scalar and a
//! band. Everything spatial is still the solver's job — which is the right
//! division of labour, because those are the questions worth waiting for.
//!
//! # Units
//!
//! Millimetres for geometry, per CONTRACT.md rule 6, because that is what CAD
//! exports. SI everywhere else: m/s, Pa, m^3/s, kg/m^3. The conversion happens
//! once, at the boundary of [`network::estimate`].

#![forbid(unsafe_code)]

pub mod centreline;
pub mod loss;
pub mod network;

pub use centreline::{
    extract_passage, extract_passage_from_flags, Bend, Confidence, MouthSpec, Passage,
    PassageConfig, PassageError, Station, Transition,
};
pub use loss::{
    bend_local_k, elbow_k, entry_k, exit_k, friction_factor, gradual_contraction_k,
    gradual_expansion_k, sudden_contraction_k, sudden_expansion_k, EntryCondition, ExitCondition,
    Friction, Regime, Section,
};
pub use network::{
    estimate, Drive, ElementKind, EstimateConfig, EstimateReport, LossCoefficient, LossElement,
};

/// Fluid properties. Defaults to the same air the solver uses, so the two
/// answers are never separated by a density.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fluid {
    /// Density, kg/m^3.
    pub rho: f64,
    /// Kinematic viscosity, m^2/s.
    pub nu: f64,
}

impl Fluid {
    /// Air at 25 C, verbatim from [`ad_gpu::air`].
    pub const AIR: Self = Self { rho: ad_gpu::air::RHO, nu: ad_gpu::air::NU };

    /// Dynamic viscosity, Pa s.
    pub fn mu(&self) -> f64 {
        self.rho * self.nu
    }
}

impl Default for Fluid {
    fn default() -> Self {
        Self::AIR
    }
}

/// Absolute roughness of a well-tuned FDM print, mm.
///
/// Layer lines are the roughness: a 0.2 mm layer height leaves ridges of
/// roughly a quarter of that once the extrusion has slumped. Against a 6 mm
/// passage this is `eps/D_h = 0.008`, which is *rougher in relative terms than
/// galvanised steel ductwork* (0.0005) and firmly in the transitionally-rough
/// regime. It is not a rounding error: at `Re = 10^4` it raises the friction
/// factor by about 25% over a hydraulically smooth wall.
///
/// The default in [`network::EstimateConfig`] is 0.0 (smooth), because a
/// default that silently adds loss makes the analytic validation cases lie.
/// Set this explicitly when estimating a real printed part.
pub const PRINTED_ROUGHNESS_MM: f64 = 0.05;

/// A value with a one-sigma uncertainty.
///
/// Mirrors the *reporting shape* of `ad_metrics::Estimate` — a mean and a
/// dispersion, printed as `mean +/- sigma` — without mirroring its
/// statistics. That type's error bar comes from an autocorrelated time
/// series; this one's comes from the stated accuracy of a published
/// correlation. They mean different things and are computed differently, so
/// they are different types, but they render the same way and the UI can put
/// them side by side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    pub mean: f64,
    /// One standard deviation. Never negative.
    pub sigma: f64,
}

impl Band {
    pub const ZERO: Self = Self { mean: 0.0, sigma: 0.0 };

    pub fn new(mean: f64, sigma: f64) -> Self {
        Self { mean, sigma: sigma.abs() }
    }

    /// A value believed exact — a momentum balance, not a curve fit.
    pub fn exact(mean: f64) -> Self {
        Self { mean, sigma: 0.0 }
    }

    /// A value with a relative uncertainty, e.g. `Band::relative(1.3, 0.30)`
    /// for "1.3, good to 30%".
    pub fn relative(mean: f64, rel: f64) -> Self {
        Self { mean, sigma: (mean * rel).abs() }
    }

    pub fn low(&self) -> f64 {
        self.mean - self.sigma
    }

    pub fn high(&self) -> f64 {
        self.mean + self.sigma
    }

    /// `sigma / |mean|`. Infinite for a mean of zero, which is the honest
    /// answer: a relative error on nothing is undefined.
    pub fn relative_sigma(&self) -> f64 {
        if self.mean.abs() > 0.0 {
            self.sigma / self.mean.abs()
        } else {
            f64::INFINITY
        }
    }

    /// Multiply by an exactly-known factor. A unit conversion, or referencing a
    /// `K` to a different velocity.
    pub fn scale(self, k: f64) -> Self {
        Self { mean: self.mean * k, sigma: self.sigma * k.abs() }
    }

    /// Whether `value` lies inside the band. Used by tests to say "the
    /// handbook figure is inside our band" rather than comparing point values.
    pub fn contains(&self, value: f64) -> bool {
        value >= self.low() && value <= self.high()
    }

    /// Sum of independent contributions: means add, sigmas add in quadrature.
    ///
    /// Quadrature is the right combination for *independent* errors and it is
    /// what is used here, but see [`Band::sum_conservative`] for why the result
    /// is then floored.
    pub fn sum(items: impl IntoIterator<Item = Band>) -> Band {
        let mut mean = 0.0;
        let mut var = 0.0;
        for b in items {
            mean += b.mean;
            var += b.sigma * b.sigma;
        }
        Band { mean, sigma: var.sqrt() }
    }

    /// Quadrature sum, with the total relative uncertainty floored at
    /// `floor_rel`.
    ///
    /// Quadrature assumes the element errors are independent. They are not: a
    /// duct that is rougher than assumed is rougher along its whole length, and
    /// a passage extraction that reads `D_h` 10% low reads it low everywhere.
    /// Stacking eight elements in quadrature would report a *smaller* relative
    /// error than any one of them, which is the classic way to talk yourself
    /// into a number you do not have. The floor — 20%, the accepted accuracy of
    /// the whole correlation method — stops that.
    pub fn sum_conservative(items: impl IntoIterator<Item = Band>, floor_rel: f64) -> Band {
        let b = Band::sum(items);
        Band { mean: b.mean, sigma: b.sigma.max((b.mean * floor_rel).abs()) }
    }
}

impl std::fmt::Display for Band {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Enough decimals to show one significant figure of the error bar, the
        // same convention `ad_metrics::Estimate` uses.
        let d = if self.sigma > 0.0 && self.sigma.is_finite() {
            (-(self.sigma.log10().floor()) as i32).clamp(0, 6) as usize
        } else {
            3
        };
        write!(f, "{:.*} +/- {:.*}", d, self.mean, d, self.sigma)
    }
}

/// Which velocity the loss coefficient is normalised by.
///
/// Mirrors `ad_metrics::ReferenceVelocity` variant for variant, deliberately:
/// `K` scales as `1/V_ref^2`, and the test part's inlet and outlet bulk
/// velocities differ by 1.85:1, so a mismatched convention between the solver
/// panel and the estimator panel would show a **3.4x** disagreement that is
/// purely bookkeeping. Keep the two in step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceVelocity {
    /// Inlet bulk velocity, `Q / A_open`. The usual convention for a fitting.
    #[default]
    Inlet,
    /// Outlet bulk velocity. The convention when the fitting is a nozzle.
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

    pub fn pick(self, u_in: f64, u_out: f64) -> f64 {
        match self {
            ReferenceVelocity::Inlet => u_in,
            ReferenceVelocity::Outlet => u_out,
            ReferenceVelocity::Faster => u_in.abs().max(u_out.abs()),
        }
    }
}

/// Traffic light for the loss coefficient. Thresholds copied from
/// `ad_metrics::LossBand` so the two panels never disagree about a colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossBand {
    /// `K < 0.6`. The contract's "ideal" range; a well-radiused bend.
    Green,
    /// `0.6 <= K < 1.5`. Past ideal but inside the `K < 1` pass mark or close.
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

    pub fn describe(self) -> &'static str {
        match self {
            LossBand::Green => "at or below a well-radiused elbow (K = 0.2-0.3)",
            LossBand::Amber => "above the ideal range but inside the K < 1 target",
            LossBand::Red => "approaching or past an unvaned mitred bend (K = 1.0-1.6)",
        }
    }
}

/// Clamped piecewise-linear interpolation through a table sorted by `x`.
///
/// Handbook fitting data *is* a table; reproducing it by fitting a polynomial
/// would introduce an error that has nothing to do with the physics. Values
/// outside the table's range clamp to its endpoints rather than extrapolating,
/// because extrapolating a curve fit past its data is the single most common
/// way these correlations get abused. Where an extrapolation is genuinely
/// wanted it is written out explicitly and justified at the call site.
pub(crate) fn interpolate(x: f64, table: &[(f64, f64)]) -> f64 {
    let Some(first) = table.first() else { return 0.0 };
    // A NaN compares false against everything, so without this guard it would
    // fall out of the loop below and silently return the table's last entry.
    // Degenerate geometry must produce the conservative end, not an arbitrary
    // one.
    if x.is_nan() || x <= first.0 {
        return first.1;
    }
    let Some(last) = table.last() else { return 0.0 };
    if x >= last.0 {
        return last.1;
    }
    for w in table.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if x <= x1 {
            let t = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.0 };
            return y0 + t * (y1 - y0);
        }
    }
    last.1
}

/// Hermite smoothstep on [0, 1]. Zero derivative at both ends, which is what
/// makes a blended correlation `C^1` rather than merely continuous.
pub(crate) fn smoothstep(t: f64) -> f64 {
    // `max` then `min` rather than `clamp`: `f64::clamp` propagates NaN, and a
    // NaN blend weight would poison a friction factor. `f64::max(NaN, 0.0)` is
    // 0.0, so this maps a NaN input onto the low end.
    let t = t.max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_band_prints_to_the_precision_of_its_error_bar() {
        assert_eq!(Band::new(12.5, 0.9).to_string(), "12.5 +/- 0.9");
        assert_eq!(Band::new(2.0, 0.06).to_string(), "2.00 +/- 0.06");
        // No error bar: fall back to three decimals rather than printing a
        // spuriously exact integer.
        assert_eq!(Band::exact(1.0).to_string(), "1.000 +/- 0.000");
    }

    #[test]
    fn quadrature_summation_never_reports_less_error_than_the_method_has() {
        // Eight elements each good to 25%: naive quadrature would claim the
        // total is good to 9%, which is nonsense.
        let parts: Vec<Band> = (0..8).map(|_| Band::relative(1.0, 0.25)).collect();
        let naive = Band::sum(parts.iter().copied());
        assert!(naive.relative_sigma() < 0.10, "quadrature really does shrink like this");
        let honest = Band::sum_conservative(parts, 0.20);
        assert!((honest.mean - 8.0).abs() < 1e-12);
        assert!(honest.relative_sigma() >= 0.20 - 1e-12);
    }

    #[test]
    fn scaling_a_band_scales_the_error_with_it() {
        let k = Band::new(2.0, 0.4).scale(0.25);
        assert!((k.mean - 0.5).abs() < 1e-12);
        assert!((k.sigma - 0.1).abs() < 1e-12);
        // ...and leaves the relative error alone, which is the invariant that
        // matters when referencing K to a different velocity.
        assert!((k.relative_sigma() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn a_zero_mean_band_reports_undefined_relative_error_rather_than_zero() {
        assert!(Band::ZERO.relative_sigma().is_infinite());
    }

    #[test]
    fn interpolation_clamps_instead_of_extrapolating() {
        let t = [(0.0, 10.0), (1.0, 20.0), (3.0, 0.0)];
        assert!((interpolate(-5.0, &t) - 10.0).abs() < 1e-12);
        assert!((interpolate(0.5, &t) - 15.0).abs() < 1e-12);
        assert!((interpolate(2.0, &t) - 10.0).abs() < 1e-12);
        assert!((interpolate(99.0, &t) - 0.0).abs() < 1e-12);
        assert!(interpolate(1.0, &[]).abs() < 1e-12);
    }

    #[test]
    fn degenerate_inputs_do_not_poison_the_helpers() {
        // NaN must land on a defined end of the range, not fall through.
        assert!((interpolate(f64::NAN, &[(0.0, 7.0), (1.0, 9.0)]) - 7.0).abs() < 1e-12);
        assert!(smoothstep(f64::NAN).abs() < 1e-12);
        assert!(smoothstep(f64::INFINITY) == 1.0);
        assert!(smoothstep(f64::NEG_INFINITY) == 0.0);
    }

    #[test]
    fn smoothstep_is_flat_at_both_ends() {
        assert!(smoothstep(-1.0).abs() < 1e-12);
        assert!((smoothstep(2.0) - 1.0).abs() < 1e-12);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-12);
        // Zero slope at the ends: the finite difference is second order small.
        let h = 1e-4;
        assert!(smoothstep(h) / h < 1e-3);
        assert!((1.0 - smoothstep(1.0 - h)) / h < 1e-3);
    }

    #[test]
    fn the_reference_velocity_choice_moves_k_by_the_area_ratio_squared() {
        // The contract's 1.85:1 contraction. This is why the two panels must
        // agree on the convention.
        let (u_in, u_out) = (2.0, 3.71);
        let k_in = ReferenceVelocity::Inlet.pick(u_in, u_out);
        let k_out = ReferenceVelocity::Outlet.pick(u_in, u_out);
        let ratio = (k_out / k_in).powi(2);
        assert!((ratio - 3.44).abs() < 0.05, "K changes by {ratio}x, expected ~3.4");
        assert!((ReferenceVelocity::Faster.pick(u_in, u_out) - u_out).abs() < 1e-12);
    }

    #[test]
    fn loss_bands_match_the_metrics_crates_thresholds() {
        assert_eq!(LossBand::of(0.3), LossBand::Green);
        assert_eq!(LossBand::of(0.6), LossBand::Amber);
        assert_eq!(LossBand::of(1.49), LossBand::Amber);
        assert_eq!(LossBand::of(1.5), LossBand::Red);
        assert_eq!(LossBand::of(f64::NAN), LossBand::Red);
    }

    #[test]
    fn air_matches_the_solvers_air_exactly() {
        assert!((Fluid::AIR.rho - 1.184).abs() < 1e-12);
        assert!((Fluid::AIR.mu() - 1.184 * 1.55e-5).abs() < 1e-18);
    }
}

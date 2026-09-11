//! A/B comparison: freeze a baseline, then show every HUD number as a delta.
//!
//! This is *the* workflow for a design tool. Nobody cares that a duct has
//! `K = 0.58`; they care whether the fillet they just added made it better than
//! the `K = 0.71` it had five minutes ago. Without a baseline the user has to
//! screenshot the HUD and compare by eye, which is exactly the manual step a
//! live tool exists to remove.
//!
//! # The part that is easy to get wrong
//!
//! A delta must be judged against the *combined* error bars, not against zero.
//! An LBM duct at 15 flow-throughs still fluctuates by a percent or so, so a
//! naive `new - old` produces a coloured arrow on every frame and trains the
//! user to ignore it. [`Delta::significance`] gates on
//! `|d| > k * sqrt(sem_a^2 + sem_b^2)` and reports
//! [`Significance::WithinNoise`] otherwise, which the panel draws in grey with
//! a flat bar rather than an arrow.

use crate::view::{MetricsView, Reading};

/// How to read the sign of a change.
///
/// Necessary because "up" is good for uniformity and bad for pressure drop, and
/// a UI that paints every increase green is worse than one that paints nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Better {
    /// Lower is better: pressure drop, loss coefficient, deflection.
    Lower,
    /// Higher is better: uniformity, flow rate at a fixed fan.
    Higher,
    /// Neither: report the change without a verdict.
    Neutral,
}

/// Whether a change survived the noise floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Significance {
    /// One or both readings had no samples.
    Unknown,
    /// Inside the combined error bars. Draw it flat and grey.
    WithinNoise,
    /// Outside them. Draw the arrow.
    Real,
}

/// The direction an arrow points, once significance has been established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Flat,
}

impl Direction {
    /// A single ASCII glyph. Deliberately not `↑`/`↓`: the default ImGui font
    /// atlas has no arrows and would draw boxes.
    pub fn glyph(self) -> &'static str {
        match self {
            Direction::Up => "^",
            Direction::Down => "v",
            Direction::Flat => "=",
        }
    }
}

/// The result of comparing one reading against its baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Delta {
    /// `now - baseline`, in the reading's own (SI) unit.
    pub absolute: f64,
    /// `(now - baseline) / |baseline|`. `NaN` when the baseline is ~zero, where
    /// a relative change is not defined rather than merely huge.
    pub relative: f64,
    /// Combined standard error, `sqrt(a^2 + b^2)`.
    pub combined_sem: f64,
    pub significance: Significance,
    pub direction: Direction,
    /// `true` when the change is real *and* an improvement, given [`Better`].
    /// `None` for [`Better::Neutral`] or an insignificant change.
    pub improved: Option<bool>,
}

impl Delta {
    /// How many combined standard errors apart the two readings are. Used for
    /// the tooltip, so a borderline result can be inspected rather than merely
    /// judged.
    pub fn sigma(&self) -> f64 {
        if self.combined_sem > 0.0 {
            self.absolute / self.combined_sem
        } else {
            f64::NAN
        }
    }

    pub fn is_real(&self) -> bool {
        self.significance == Significance::Real
    }

    /// Colour for the arrow: green for a real improvement, red for a real
    /// regression, grey for anything the statistics cannot distinguish.
    pub fn color(&self) -> [f32; 4] {
        match (self.significance, self.improved) {
            (Significance::Real, Some(true)) => [0.36, 0.82, 0.47, 1.0],
            (Significance::Real, Some(false)) => [0.94, 0.33, 0.31, 1.0],
            (Significance::Real, None) => [0.72, 0.78, 0.92, 1.0],
            _ => [0.55, 0.57, 0.61, 1.0],
        }
    }
}

/// Number of combined standard errors a change must exceed to count.
///
/// Two, not one. At one sigma roughly a third of pure-noise comparisons would
/// light up, which is often enough to be worthless. Two puts the false-positive
/// rate around 5%, which is the usual engineering compromise and — more to the
/// point — is low enough that a lit arrow is worth looking at.
pub const SIGNIFICANCE_SIGMA: f64 = 2.0;

/// Compare one reading against its baseline.
pub fn delta(now: Reading, base: Reading, better: Better) -> Delta {
    let combined = (now.sem.max(0.0).powi(2) + base.sem.max(0.0).powi(2)).sqrt();
    if !now.is_known() || !base.is_known() {
        return Delta {
            absolute: f64::NAN,
            relative: f64::NAN,
            combined_sem: combined,
            significance: Significance::Unknown,
            direction: Direction::Flat,
            improved: None,
        };
    }

    let absolute = now.value - base.value;
    let relative = if base.value.abs() > 1e-12 {
        absolute / base.value.abs()
    } else {
        f64::NAN
    };

    let significance = if absolute.abs() > SIGNIFICANCE_SIGMA * combined.max(f64::MIN_POSITIVE) {
        Significance::Real
    } else {
        Significance::WithinNoise
    };

    let direction = match significance {
        Significance::Real if absolute > 0.0 => Direction::Up,
        Significance::Real => Direction::Down,
        _ => Direction::Flat,
    };

    let improved = match (significance, better) {
        (Significance::Real, Better::Lower) => Some(absolute < 0.0),
        (Significance::Real, Better::Higher) => Some(absolute > 0.0),
        _ => None,
    };

    Delta { absolute, relative, combined_sem: combined, significance, direction, improved }
}

/// A frozen snapshot to compare against.
///
/// Stores the whole [`MetricsView`] rather than a handful of scalars, so a new
/// metric added in Wave 3 is comparable without touching this file. It also
/// keeps the solver step and a human label, because a baseline whose provenance
/// you cannot recall is a baseline you stop trusting.
#[derive(Debug, Clone)]
pub struct Baseline {
    pub label: String,
    /// Solver step the snapshot was taken at.
    pub step: u64,
    /// Inlet velocity at the time, m/s. Shown next to the label because
    /// comparing two designs at different operating points is the single
    /// easiest way to fool yourself with this feature.
    pub inlet_velocity_ms: f32,
    pub metrics: MetricsView,
}

impl Baseline {
    pub fn capture(
        label: impl Into<String>,
        step: u64,
        inlet_velocity_ms: f32,
        metrics: &MetricsView,
    ) -> Self {
        Self { label: label.into(), step, inlet_velocity_ms, metrics: metrics.clone() }
    }

    /// Whether the live run is at the same operating point as this baseline.
    ///
    /// Compared at 1% because the velocity slider is continuous and an exact
    /// float match would essentially never hold. When this is false the panel
    /// says so: the deltas are still computed (sometimes that *is* the
    /// experiment) but they are not a like-for-like design comparison.
    pub fn same_operating_point(&self, inlet_velocity_ms: f32) -> bool {
        let a = self.inlet_velocity_ms.abs().max(1e-6);
        (self.inlet_velocity_ms - inlet_velocity_ms).abs() / a < 0.01
    }
}

/// Every delta the HUD shows, in one struct so the panel is a straight-line
/// render with no comparison logic in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricDeltas {
    pub flow_rate: Delta,
    pub pressure_drop: Delta,
    pub loss_coefficient: Delta,
    pub uniformity: Delta,
    pub deflection_deg: Delta,
    pub max_speed: Delta,
}

/// Compare the whole HUD row against a baseline.
///
/// The [`Better`] choice per metric is the interesting content of this
/// function, and it is deliberately not configurable: a duct with a lower loss
/// coefficient is a better duct, and letting that be a setting invites someone
/// to flip it to make a bad result look good.
pub fn compare(now: &MetricsView, base: &MetricsView) -> MetricDeltas {
    MetricDeltas {
        // At a fixed inlet velocity more flow means more open area, which is
        // usually the goal but is not unambiguously better, so: neutral.
        flow_rate: delta(now.flow_rate, base.flow_rate, Better::Neutral),
        pressure_drop: delta(now.pressure_drop, base.pressure_drop, Better::Lower),
        loss_coefficient: delta(now.loss_coefficient, base.loss_coefficient, Better::Lower),
        uniformity: delta(now.uniformity, base.uniformity, Better::Higher),
        deflection_deg: delta(now.deflection_deg, base.deflection_deg, Better::Lower),
        // A higher peak speed at the same flow means a tighter separation
        // bubble or a sharper vena contracta: worse.
        max_speed: delta(now.max_speed, base.max_speed, Better::Lower),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::Health;

    fn r(value: f64, sem: f64) -> Reading {
        Reading::new(value, sem, Health::Good, 256)
    }

    #[test]
    fn a_change_inside_the_error_bars_is_not_an_improvement() {
        // The failure this whole module exists to prevent: a coloured arrow on
        // a change that is pure fluctuation.
        let d = delta(r(47.4, 0.6), r(47.3, 0.6), Better::Lower);
        assert_eq!(d.significance, Significance::WithinNoise);
        assert_eq!(d.direction, Direction::Flat);
        assert_eq!(d.improved, None);
        assert_eq!(d.color(), [0.55, 0.57, 0.61, 1.0]);
    }

    #[test]
    fn a_real_reduction_in_pressure_drop_is_green_and_points_down() {
        let d = delta(r(41.0, 0.6), r(47.3, 0.6), Better::Lower);
        assert_eq!(d.significance, Significance::Real);
        assert_eq!(d.direction, Direction::Down);
        assert_eq!(d.improved, Some(true));
        assert!((d.absolute + 6.3).abs() < 1e-9);
        assert!((d.relative + 6.3 / 47.3).abs() < 1e-9);
        assert!(d.sigma() < -7.0, "6.3 Pa on +/-0.85 combined is many sigma");
    }

    #[test]
    fn the_same_rise_is_good_or_bad_depending_on_the_metric() {
        // Uniformity up is good; loss coefficient up is bad. Same arithmetic,
        // opposite verdict, which is the reason `Better` exists.
        let up = |better| delta(r(0.95, 0.005), r(0.88, 0.005), better);
        assert_eq!(up(Better::Higher).improved, Some(true));
        assert_eq!(up(Better::Lower).improved, Some(false));
        assert_eq!(up(Better::Neutral).improved, None);
        assert_eq!(up(Better::Neutral).direction, Direction::Up, "neutral still reports the sign");
    }

    #[test]
    fn comparing_against_an_unmeasured_baseline_reports_unknown() {
        let d = delta(r(47.3, 0.6), Reading::unknown(), Better::Lower);
        assert_eq!(d.significance, Significance::Unknown);
        assert!(d.absolute.is_nan());
        assert_eq!(d.improved, None);
        // ...and the other way round, right after a statistics reset.
        assert_eq!(
            delta(Reading::unknown(), r(47.3, 0.6), Better::Lower).significance,
            Significance::Unknown
        );
    }

    #[test]
    fn a_zero_baseline_has_no_relative_change_but_still_has_an_absolute_one() {
        let d = delta(r(0.4, 0.01), r(0.0, 0.01), Better::Lower);
        assert!((d.absolute - 0.4).abs() < 1e-12);
        assert!(d.relative.is_nan(), "a percentage of zero is not a number");
        assert_eq!(d.significance, Significance::Real);
    }

    #[test]
    fn zero_error_bars_do_not_make_every_comparison_significant_by_dividing_by_zero() {
        // Both readings exact and equal: the change is zero, which is not
        // greater than zero, so it must land in the noise band rather than
        // being declared real by a 0/0.
        let d = delta(r(1.0, 0.0), r(1.0, 0.0), Better::Lower);
        assert_eq!(d.significance, Significance::WithinNoise);
        assert!(d.sigma().is_nan());
        // ...but a genuine difference with no error bars is still real.
        assert_eq!(delta(r(2.0, 0.0), r(1.0, 0.0), Better::Lower).significance, Significance::Real);
    }

    #[test]
    fn the_whole_hud_row_compares_with_the_right_polarity() {
        let mut base = MetricsView::default();
        base.pressure_drop = r(47.3, 0.6);
        base.loss_coefficient = r(0.71, 0.01);
        base.uniformity = r(0.88, 0.005);
        base.deflection_deg = r(8.0, 0.2);
        base.max_speed = r(18.4, 0.3);
        base.flow_rate = r(0.006_35, 0.000_04);

        let mut now = base.clone();
        now.pressure_drop = r(41.0, 0.6);
        now.loss_coefficient = r(0.58, 0.01);
        now.uniformity = r(0.95, 0.005);
        now.deflection_deg = r(3.2, 0.2);
        now.max_speed = r(14.0, 0.3);

        let d = compare(&now, &base);
        assert_eq!(d.pressure_drop.improved, Some(true));
        assert_eq!(d.loss_coefficient.improved, Some(true));
        assert_eq!(d.uniformity.improved, Some(true));
        assert_eq!(d.deflection_deg.improved, Some(true));
        assert_eq!(d.max_speed.improved, Some(true));
        // Flow rate did not move at all.
        assert_eq!(d.flow_rate.significance, Significance::WithinNoise);
    }

    #[test]
    fn a_baseline_knows_whether_it_is_comparable() {
        let base = Baseline::capture("as loaded", 12_000, 3.0, &MetricsView::default());
        assert!(base.same_operating_point(3.0));
        assert!(base.same_operating_point(3.02));
        assert!(!base.same_operating_point(5.0), "comparing 3 m/s to 5 m/s is not an A/B test");
        assert_eq!(base.step, 12_000);
    }
}

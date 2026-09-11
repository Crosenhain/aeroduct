//! Turning numbers into strings, honestly.
//!
//! Two jobs, and both of them are places where a plausible-looking shortcut
//! produces a misleading UI:
//!
//! 1. **Uncertainty.** `47.31843 Pa` is not a measurement, it is a float. The
//!    number of digits worth showing is decided by the error bar and by nothing
//!    else, so [`uncertain`] rounds the value to the precision the error bar
//!    justifies rather than to a fixed format string.
//! 2. **Units.** Everything upstream is SI (or millimetres, per the contract);
//!    the conversion to CFM or inches of water happens here, at the last
//!    moment. Converting earlier would mean the A/B baseline and the live
//!    reading could be in different units without anything noticing.

use crate::view::Reading;

/// Which unit family the display is in.
///
/// Duct work is quoted in CFM and inches of water in North America and in L/s
/// and pascals almost everywhere else, and people genuinely cannot read the
/// other one at a glance. This is a display switch only; nothing upstream of
/// [`crate::format`] ever sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnitSystem {
    /// L/s, Pa, m/s, mm.
    #[default]
    Metric,
    /// CFM, inH2O, ft/min, in.
    Imperial,
}

impl UnitSystem {
    pub fn label(self) -> &'static str {
        match self {
            UnitSystem::Metric => "metric",
            UnitSystem::Imperial => "imperial",
        }
    }
    pub fn toggled(self) -> Self {
        match self {
            UnitSystem::Metric => UnitSystem::Imperial,
            UnitSystem::Imperial => UnitSystem::Metric,
        }
    }
}

/// Cubic feet per minute per cubic metre per second.
pub const CFM_PER_M3S: f64 = 2118.8800032893155;
/// Litres per second per cubic metre per second.
pub const LPS_PER_M3S: f64 = 1000.0;
/// Inches of water column (at 4 C) per pascal.
pub const INH2O_PER_PA: f64 = 1.0 / 249.0889;
/// Feet per minute per metre per second.
pub const FPM_PER_MS: f64 = 196.850393700787;
/// Inches per millimetre.
pub const IN_PER_MM: f64 = 1.0 / 25.4;

/// A linear unit conversion plus the label to print after the number.
///
/// Linear-and-through-the-origin is the only kind of conversion this app needs,
/// which matters: a reading's *uncertainty* scales by the same factor as its
/// value, so [`Quantity::convert`] can transform a whole [`Reading`] without
/// ever having to think about error propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    pub factor: f64,
    pub unit: &'static str,
}

impl Quantity {
    pub const fn new(factor: f64, unit: &'static str) -> Self {
        Self { factor, unit }
    }

    /// Volumetric flow, from m^3/s.
    pub fn flow(system: UnitSystem) -> Self {
        match system {
            UnitSystem::Metric => Self::new(LPS_PER_M3S, "L/s"),
            UnitSystem::Imperial => Self::new(CFM_PER_M3S, "CFM"),
        }
    }

    /// Pressure, from Pa.
    pub fn pressure(system: UnitSystem) -> Self {
        match system {
            UnitSystem::Metric => Self::new(1.0, "Pa"),
            UnitSystem::Imperial => Self::new(INH2O_PER_PA, "inH2O"),
        }
    }

    /// Velocity, from m/s.
    pub fn velocity(system: UnitSystem) -> Self {
        match system {
            UnitSystem::Metric => Self::new(1.0, "m/s"),
            UnitSystem::Imperial => Self::new(FPM_PER_MS, "ft/min"),
        }
    }

    /// Length, from mm.
    pub fn length(system: UnitSystem) -> Self {
        match system {
            UnitSystem::Metric => Self::new(1.0, "mm"),
            UnitSystem::Imperial => Self::new(IN_PER_MM, "in"),
        }
    }

    /// Dimensionless, so nothing is scaled and nothing is appended.
    pub const fn plain() -> Self {
        Self::new(1.0, "")
    }

    /// Degrees. Written as `deg` rather than the degree sign because the ImGui
    /// default font atlas is Latin-1 only and would render `°` as a box.
    pub const fn degrees() -> Self {
        Self::new(1.0, "deg")
    }

    /// Convert a whole reading. Value and error bar scale together, which is
    /// exactly right for a linear conversion through the origin and is why
    /// this type refuses to represent an offset one.
    pub fn convert(&self, r: Reading) -> Reading {
        Reading {
            value: r.value * self.factor,
            sem: r.sem * self.factor,
            state: r.state,
            samples: r.samples,
        }
    }
}

/// Decimal places justified by an uncertainty of `sem`.
///
/// The convention: **one significant figure in the uncertainty, two when its
/// leading digit is 1.** The exception matters more than it looks — going from
/// `+/- 0.1` to `+/- 0.14` is a 40% change in the quoted precision, and
/// rounding it away is the difference between an error bar that means something
/// and one that is itself rounded by a third.
///
/// Returns `None` when `sem` is unusable, so callers can fall back to a fixed
/// format rather than inventing a precision.
pub fn decimals_for(sem: f64) -> Option<usize> {
    if !sem.is_finite() || sem <= 0.0 {
        return None;
    }
    let exp = sem.abs().log10().floor();
    let lead = (sem.abs() / 10f64.powf(exp)).floor() as i32;
    // Two significant figures when the leading digit is 1, one otherwise.
    let sig = if lead == 1 { 2 } else { 1 };
    let places = -(exp as i32) + (sig - 1);
    Some(places.clamp(0, 9) as usize)
}

/// `"47.3 +/- 0.6 Pa"`.
///
/// ASCII `+/-` rather than `±` for the same font-atlas reason as `deg`: the
/// default ImGui atlas has no glyph for it and would draw a hollow box, which
/// looks like a rendering bug rather than a plus-minus sign.
pub fn uncertain(r: Reading, q: Quantity) -> String {
    let r = q.convert(r);
    let unit = if q.unit.is_empty() {
        String::new()
    } else {
        format!(" {}", q.unit)
    };

    if !r.is_known() {
        return format!("--{unit}");
    }
    match decimals_for(r.sem) {
        Some(d) => format!("{:.*} +/- {:.*}{unit}", d, r.value, d, r.sem),
        // No error bar yet: a tilde, so the number never masquerades as
        // converged. Three significant-ish digits, which is all a single
        // instantaneous sample can support.
        None => format!("~{}{unit}", short(r.value)),
    }
}

/// Just the value, at the precision the error bar justifies. For a HUD chip
/// that shows the error bar separately or not at all.
pub fn value_only(r: Reading, q: Quantity) -> String {
    let r = q.convert(r);
    if !r.is_known() {
        return "--".to_string();
    }
    match decimals_for(r.sem) {
        Some(d) => format!("{:.*}", d, r.value),
        None => short(r.value),
    }
}

/// Three-significant-figure rendering for a number with no error bar attached.
///
/// Switches to exponent form outside `[1e-3, 1e6)`, because a duct simulation
/// legitimately produces both `0.00042 m^3/s` and `132000000 cells`, and a
/// fixed format is unreadable for one of them.
pub fn short(v: f64) -> String {
    if !v.is_finite() {
        return "--".to_string();
    }
    let a = v.abs();
    if a == 0.0 {
        return "0".to_string();
    }
    if a < 1e-3 || a >= 1e6 {
        return format!("{v:.2e}");
    }
    let decimals = if a >= 100.0 {
        0
    } else if a >= 10.0 {
        1
    } else if a >= 1.0 {
        2
    } else {
        3
    };
    format!("{v:.*}", decimals)
}

/// Thousands-separated integer, for cell and step counts.
///
/// Uses a thin-looking ASCII apostrophe-free grouping with commas; at
/// 132,000,000 cells the ungrouped form is genuinely hard to read at a glance
/// and the whole point of the status bar is glanceability.
pub fn grouped(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A duration in the largest unit that keeps it readable.
pub fn duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "--".to_string();
    }
    if seconds < 1e-3 {
        format!("{:.0} us", seconds * 1e6)
    } else if seconds < 1.0 {
        format!("{:.1} ms", seconds * 1e3)
    } else if seconds < 60.0 {
        format!("{seconds:.2} s")
    } else if seconds < 3600.0 {
        // Floor the remainder rather than letting `{:.0}` round it: 119.6 s must
        // read "1m 59s", not "1m 60s", and 7,300 s must read "2h 01m" rather
        // than rounding 1m40s up to "2h 02m".
        format!(
            "{:.0}m {:02.0}s",
            (seconds / 60.0).floor(),
            (seconds % 60.0).floor()
        )
    } else {
        format!(
            "{:.0}h {:02.0}m",
            (seconds / 3600.0).floor(),
            ((seconds % 3600.0) / 60.0).floor()
        )
    }
}

/// A ratio as a percentage with one decimal, or `--` when it is not a number.
pub fn percent(x: f64) -> String {
    if x.is_finite() {
        format!("{:.1}%", x * 100.0)
    } else {
        "--".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::Health;

    fn r(value: f64, sem: f64) -> Reading {
        Reading::new(value, sem, Health::Good, 128)
    }

    #[test]
    fn uncertainty_sets_the_precision_of_the_value() {
        // The headline example from the brief.
        assert_eq!(
            uncertain(r(47.31843, 0.6), Quantity::pressure(UnitSystem::Metric)),
            "47.3 +/- 0.6 Pa"
        );
        // Leading digit 1 keeps a second figure, so +/-0.14 is not flattened
        // to +/-0.1.
        assert_eq!(
            uncertain(r(47.31843, 0.14), Quantity::pressure(UnitSystem::Metric)),
            "47.32 +/- 0.14 Pa"
        );
        // A fat error bar strips decimals from the value too.
        assert_eq!(
            uncertain(r(47.31843, 3.0), Quantity::pressure(UnitSystem::Metric)),
            "47 +/- 3 Pa"
        );
        assert_eq!(
            uncertain(r(1470.0, 40.0), Quantity::pressure(UnitSystem::Metric)),
            "1470 +/- 40 Pa"
        );
    }

    #[test]
    fn decimals_track_the_error_bar_across_decades() {
        assert_eq!(decimals_for(0.6), Some(1));
        assert_eq!(decimals_for(0.06), Some(2));
        assert_eq!(decimals_for(0.14), Some(2));
        assert_eq!(decimals_for(0.014), Some(3));
        assert_eq!(decimals_for(3.0), Some(0));
        assert_eq!(decimals_for(14.0), Some(0));
        assert_eq!(
            decimals_for(40.0),
            Some(0),
            "tens are already coarser than one place"
        );
        // Unusable inputs must not invent a precision.
        assert_eq!(decimals_for(0.0), None);
        assert_eq!(decimals_for(-1.0), None);
        assert_eq!(decimals_for(f64::NAN), None);
        assert_eq!(decimals_for(f64::INFINITY), None);
    }

    #[test]
    fn a_reading_with_no_samples_renders_as_a_dash_not_a_number() {
        // If this ever regresses, the HUD shows a confident 0.00 during warm-up
        // and someone screenshots it.
        let q = Quantity::pressure(UnitSystem::Metric);
        assert_eq!(uncertain(Reading::unknown(), q), "-- Pa");
        assert_eq!(
            uncertain(Reading::new(0.0, 0.1, Health::Good, 0), q),
            "-- Pa"
        );
        assert_eq!(value_only(Reading::unknown(), q), "--");
    }

    #[test]
    fn a_reading_with_no_error_bar_is_marked_provisional() {
        let q = Quantity::plain();
        let s = uncertain(Reading::new(1.234, f64::NAN, Health::Unknown, 1), q);
        assert!(
            s.starts_with('~'),
            "an un-averaged sample must be marked: {s}"
        );
        assert!(!s.contains("+/-"));
    }

    #[test]
    fn conversion_scales_the_error_bar_with_the_value() {
        // The reason `Quantity` refuses to model an offset conversion: this
        // identity is what makes unit switching safe.
        let si = r(0.006_35, 0.000_05);
        let cfm = Quantity::flow(UnitSystem::Imperial).convert(si);
        assert!((cfm.value - 13.45).abs() < 0.01, "{} CFM", cfm.value);
        assert!(
            (cfm.sem / cfm.value - si.sem / si.value).abs() < 1e-12,
            "relative error must be invariant under a unit change"
        );
        assert_eq!(cfm.samples, si.samples);
        assert_eq!(cfm.state, si.state);
    }

    #[test]
    fn the_contract_table_round_trips_through_the_flow_conversion() {
        // CONTRACT.md: U_in = 3 m/s gives 6.35 L/s and 13.5 CFM.
        let si = r(0.006_35, 0.000_04);
        assert_eq!(
            uncertain(si, Quantity::flow(UnitSystem::Metric)),
            "6.35 +/- 0.04 L/s"
        );
        // The contract quotes 13.5 CFM to three significant figures; the error
        // bar of +/-0.08 CFM justifies a second decimal, so the honest rendering
        // is 13.45. Rounding to the contract's three figures here would throw
        // away a digit the measurement actually supports.
        let imperial = uncertain(si, Quantity::flow(UnitSystem::Imperial));
        assert!(imperial.starts_with("13.45 +/- 0.08"), "{imperial}");
        assert!(imperial.ends_with(" CFM"));
    }

    #[test]
    fn pressure_in_inches_of_water_keeps_enough_digits() {
        // 47.3 Pa is 0.19 inH2O; if the conversion happened before the
        // rounding decision this would come out as "0.2 +/- 0.0".
        let s = uncertain(r(47.3, 0.6), Quantity::pressure(UnitSystem::Imperial));
        assert_eq!(s, "0.190 +/- 0.002 inH2O");
    }

    #[test]
    fn unit_system_toggles_both_ways() {
        assert_eq!(UnitSystem::Metric.toggled(), UnitSystem::Imperial);
        assert_eq!(
            UnitSystem::Imperial.toggled().toggled(),
            UnitSystem::Imperial
        );
    }

    #[test]
    fn short_form_covers_the_range_a_duct_sim_actually_produces() {
        assert_eq!(short(0.0), "0");
        assert_eq!(short(0.5), "0.500");
        assert_eq!(short(5.25), "5.25");
        assert_eq!(short(52.5), "52.5");
        assert_eq!(short(525.4), "525");
        assert_eq!(short(1.0e-5), "1.00e-5");
        assert_eq!(short(1.32e8), "1.32e8");
        assert_eq!(short(f64::NAN), "--");
    }

    #[test]
    fn grouping_matches_the_obvious_cases() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(132_000_000), "132,000,000");
    }

    #[test]
    fn durations_pick_a_readable_unit() {
        assert_eq!(duration(5.0e-6), "5 us");
        assert_eq!(duration(0.25), "250.0 ms");
        assert_eq!(duration(12.5), "12.50 s");
        assert_eq!(duration(125.0), "2m 05s");
        assert_eq!(duration(7_300.0), "2h 01m");
        assert_eq!(duration(-1.0), "--");
    }
}

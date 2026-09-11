//! The loss network: [`Passage`] plus a fan curve in, engineering numbers out.
//!
//! # What it computes
//!
//! ```text
//! friction     integral over the centreline of f(s)/D_h(s) * 1/2 rho V(s)^2 ds
//! bends        one K per detected turn, on that turn's local velocity
//! transitions  one K per detected area change, on its faster side
//! entry, exit  optional, off by default -- see below
//!             ------------------------------------------------
//! dp_total     the sum
//! dp_static    dp_total + 1/2 rho (V_out^2 - V_in^2)
//! K            dp_total / (1/2 rho V_ref^2)
//! Q            V_in * A_in
//! ```
//!
//! The friction term is *integrated*, not evaluated once with an average
//! `L/D_h`. On the test part the area contracts 1.85:1, so `V` nearly doubles
//! and the friction loading — which goes as `V^2/D_h` — varies by a factor of
//! five along the duct. A single-station friction term would put that loss in
//! the wrong place and get the total wrong by tens of percent.
//!
//! # Entry and exit are off by default, and that is a real decision
//!
//! The solver reports a **total**-pressure drop measured between the two mouth
//! *planes*. At the inlet plane the air is already moving and already inside
//! the duct, so nothing was spent getting it there; at the outlet plane its
//! kinetic energy is still counted in the total pressure, so nothing has been
//! dumped yet. To compare like with like, this crate defaults to
//! [`EntryCondition::None`] and [`ExitCondition::None`].
//!
//! That is *not* what a fan has to supply. For that question set
//! [`ExitCondition::Discharge`] — worth `K = 1` exactly, which on a short duct
//! is usually the single largest term in the whole network. Both numbers are
//! right; they answer different questions, and confusing them is the classic
//! way a duct comes out a factor of two off.
//!
//! # Why it is fast
//!
//! Everything here reads a few hundred `f64`s out of a [`Passage`] and does
//! arithmetic. There is no allocation per station, no voxel access, and no
//! iteration except the dozen `log10` calls inside Colebrook. A slider tick
//! re-solves in microseconds, which is the entire point of the crate.

use crate::centreline::{Confidence, Passage, Station};
use crate::loss::{
    bend_local_k, entry_k, exit_k, friction_factor, gradual_contraction_k, gradual_expansion_k,
    EntryCondition, ExitCondition, Regime, Section,
};
use crate::{Band, Fluid, LossBand, ReferenceVelocity};

/// What drives the flow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Drive {
    /// Bulk velocity at the inlet mouth, m/s. What the app's slider sets.
    InletVelocity(f64),
    /// Volumetric flow, m^3/s.
    Flow(f64),
    /// Available total pressure, Pa — a fan's operating point. Solved for the
    /// flow it produces, which is the question a fan-curve intersection asks.
    TotalPressure(f64),
}

/// Everything the network needs that is not geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimateConfig {
    pub fluid: Fluid,
    /// Absolute wall roughness, mm. **Defaults to 0** — hydraulically smooth —
    /// so the analytic validation cases mean what they say. Set it to
    /// [`crate::PRINTED_ROUGHNESS_MM`] for a real printed part; on a 6 mm
    /// passage it is worth about 25% on the friction term.
    pub roughness_mm: f64,
    pub entry: EntryCondition,
    pub exit: ExitCondition,
    pub reference: ReferenceVelocity,
    /// Relative-uncertainty floor on the total. 0.20 is the accepted accuracy
    /// of the correlation method as a whole; claiming better is claiming
    /// something the method cannot deliver. See [`Band::sum_conservative`].
    pub uncertainty_floor: f64,
}

impl Default for EstimateConfig {
    fn default() -> Self {
        Self {
            fluid: Fluid::AIR,
            roughness_mm: 0.0,
            entry: EntryCondition::None,
            exit: ExitCondition::None,
            reference: ReferenceVelocity::Inlet,
            uncertainty_floor: 0.20,
        }
    }
}

impl EstimateConfig {
    /// The configuration that answers "what must the fan supply": a sharp-edged
    /// entry from a plenum and a discharge to still air.
    pub fn as_installed() -> Self {
        Self {
            entry: EntryCondition::Flush,
            exit: ExitCondition::Discharge,
            roughness_mm: crate::PRINTED_ROUGHNESS_MM,
            ..Self::default()
        }
    }
}

/// What kind of loss an element is. The breakdown is the actionable part of
/// the report: it says which *feature* to change, which a single `K` cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementKind {
    Friction,
    Bend,
    Contraction,
    Expansion,
    Entry,
    Exit,
}

impl ElementKind {
    pub fn label(self) -> &'static str {
        match self {
            ElementKind::Friction => "friction",
            ElementKind::Bend => "bend",
            ElementKind::Contraction => "contraction",
            ElementKind::Expansion => "expansion",
            ElementKind::Entry => "entry",
            ElementKind::Exit => "exit",
        }
    }

    /// One sentence on what to do about it.
    pub fn advice(self) -> &'static str {
        match self {
            ElementKind::Friction => {
                "shorten the passage, open it out, or print it smoother -- friction \
                 scales as L/D_h and as the fifth power of D_h at fixed flow"
            }
            ElementKind::Bend => {
                "radius the turn, or rotate the section so the duct turns about its \
                 short side; r/D_h = 1.5 costs a seventh of a mitre"
            }
            ElementKind::Contraction => "lengthen the taper: the loss goes as sin(theta/2)",
            ElementKind::Expansion => {
                "lengthen the diffuser to under a 45 degree included angle, or the flow \
                 separates and it performs no better than a sudden step"
            }
            ElementKind::Entry => "round the inlet lip; r/D_h = 0.1 removes 80% of it",
            ElementKind::Exit => {
                "open the outlet out: the discharge loss is the whole velocity head and \
                 falls as the square of the exit area"
            }
        }
    }
}

/// One term in the loss network.
#[derive(Debug, Clone, PartialEq)]
pub struct LossElement {
    pub kind: ElementKind,
    /// Human-readable identity, e.g. `"bend 87 deg, r/D_h = 0.6"`.
    pub label: String,
    /// `K` on this element's own local velocity — the number to compare
    /// against a handbook.
    pub k_local: Band,
    /// The velocity `k_local` is referenced to, m/s.
    pub v_local_ms: f64,
    /// `K` re-referenced to the report's reference velocity, so the elements
    /// sum to the total.
    pub k_ref: Band,
    /// Pressure drop across this element, Pa.
    pub dp_pa: Band,
    /// Fraction of the total drop. The column to sort by.
    pub share: f64,
    /// Where along the passage it sits, mm.
    pub s_start_mm: f64,
    pub s_end_mm: f64,
    /// The correlation this came from, for the tooltip.
    pub source: &'static str,
}

/// The loss coefficient with the convention it was computed under.
///
/// Mirrors `ad_metrics::LossCoefficient` field for field so the two panels can
/// be read as one table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossCoefficient {
    pub k: Band,
    pub band: LossBand,
    pub reference: ReferenceVelocity,
    pub v_ref_ms: f64,
}

impl std::fmt::Display for LossCoefficient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "K = {} [{}] on V_ref = {:.2} m/s ({}), {}",
            self.k,
            self.band.color(),
            self.v_ref_ms,
            self.reference.label(),
            self.band.describe()
        )
    }
}

/// The answer.
///
/// Field names deliberately track `ad_metrics::MetricsReport` so the UI can put
/// the estimator and the solver side by side without a translation layer.
#[derive(Debug, Clone, PartialEq)]
pub struct EstimateReport {
    /// Volumetric flow, m^3/s.
    pub flow_m3s: f64,
    pub inlet_velocity_ms: f64,
    pub outlet_velocity_ms: f64,
    /// Mass-flow-weighted total-pressure drop, Pa. The number to quote.
    pub total_pressure_drop_pa: Band,
    /// Static-pressure drop, Pa. What a wall tapping measures. Differs from the
    /// total by the change in velocity head, which on a contracting duct is
    /// large and of the *opposite* sign to intuition.
    pub static_pressure_drop_pa: Band,
    pub loss_coefficient: LossCoefficient,
    /// Every term, largest first.
    pub elements: Vec<LossElement>,
    /// Reynolds number on the length-weighted mean hydraulic diameter and the
    /// inlet bulk velocity.
    pub reynolds: f64,
    pub hydraulic_diameter_mm: f64,
    /// The regime most of the friction was computed in.
    pub regime: Regime,
    /// Length-weighted mean Darcy friction factor.
    pub mean_friction_factor: f64,
    pub confidence: Confidence,
    pub warnings: Vec<String>,
}

impl EstimateReport {
    /// Volumetric flow in litres per second.
    pub fn litres_per_second(&self) -> f64 {
        self.flow_m3s * 1000.0
    }

    /// Volumetric flow in cubic feet per minute. Same constant `MetricSample`
    /// uses, so the two panels never differ in the fourth digit.
    pub fn cfm(&self) -> f64 {
        self.flow_m3s * 2118.88
    }

    /// The single biggest loss. The feature to change first.
    pub fn dominant_element(&self) -> Option<&LossElement> {
        self.elements.first()
    }

    /// Total `K` attributable to one kind of element.
    pub fn k_of(&self, kind: ElementKind) -> Band {
        Band::sum(self.elements.iter().filter(|e| e.kind == kind).map(|e| e.k_ref))
    }

    /// Every scalar as `(name, band, unit)`, in the order a report is read.
    /// Shaped like `MetricsReport::scalars` so one table renderer serves both.
    pub fn scalars(&self) -> Vec<(&'static str, Band, &'static str)> {
        vec![
            ("flow", Band::exact(self.litres_per_second()), "L/s"),
            ("flow", Band::exact(self.cfm()), "CFM"),
            ("total pressure drop", self.total_pressure_drop_pa, "Pa"),
            ("static pressure drop", self.static_pressure_drop_pa, "Pa"),
            ("loss coefficient", self.loss_coefficient.k, "-"),
            ("reference velocity", Band::exact(self.loss_coefficient.v_ref_ms), "m/s"),
            ("inlet bulk velocity", Band::exact(self.inlet_velocity_ms), "m/s"),
            ("outlet bulk velocity", Band::exact(self.outlet_velocity_ms), "m/s"),
            ("Reynolds number", Band::exact(self.reynolds), "-"),
            ("hydraulic diameter", Band::exact(self.hydraulic_diameter_mm), "mm"),
            ("friction factor", Band::exact(self.mean_friction_factor), "-"),
        ]
    }

    pub fn summary(&self) -> String {
        format!(
            "{:.2} L/s ({:.1} CFM), dp_total = {} Pa, {} | Re = {:.0} ({}), confidence {}",
            self.litres_per_second(),
            self.cfm(),
            self.total_pressure_drop_pa,
            self.loss_coefficient,
            self.reynolds,
            self.regime.label(),
            self.confidence.label(),
        )
    }

    /// The breakdown, as a block of text. This is the part a designer acts on.
    pub fn breakdown(&self) -> String {
        let mut s = String::from("  element          K_ref        dp (Pa)      share  source\n");
        for e in &self.elements {
            s.push_str(&format!(
                "  {:<14}  {:>11}  {:>11}  {:>5.0}%  {}\n",
                e.label,
                format!("{:.3}", e.k_ref.mean),
                format!("{:.2}", e.dp_pa.mean),
                100.0 * e.share,
                e.source,
            ));
        }
        s.push_str(&format!(
            "  {:<14}  {:>11}  {:>11}  {:>5}\n",
            "TOTAL",
            format!("{:.3}", self.loss_coefficient.k.mean),
            format!("{:.2}", self.total_pressure_drop_pa.mean),
            "100%"
        ));
        s
    }
}

// ---------------------------------------------------------------------------
// The solve
// ---------------------------------------------------------------------------

/// Evaluate the loss network.
///
/// Microseconds. Call it on every slider tick; only re-extract the
/// [`Passage`] when the geometry itself moves.
pub fn estimate(passage: &Passage, drive: Drive, cfg: &EstimateConfig) -> EstimateReport {
    let a_in = (passage.inlet_area_mm2 * 1e-6).max(1e-12);
    let q = match drive {
        Drive::InletVelocity(v) => v.max(0.0) * a_in,
        Drive::Flow(q) => q.max(0.0),
        Drive::TotalPressure(dp) => flow_for_pressure(passage, dp, cfg),
    };
    solve(passage, q, cfg)
}

/// Fixed-point solve for the flow a given total pressure will push through.
///
/// `dp = K(Q) * 1/2 rho * (Q/A)^2` with `K` only weakly dependent on `Q`
/// (through the Reynolds number in the friction term), so iterating
/// `Q <- A sqrt(2 dp / (rho K))` converges geometrically with a contraction
/// factor of about `1/8`. Six iterations is machine precision; twelve is
/// paranoia and still costs less than a microsecond.
///
/// A bisection would also work and would be unconditionally safe, but it needs
/// thirty-plus evaluations for the same accuracy — and the whole promise of
/// this crate is that the answer is instant.
fn flow_for_pressure(passage: &Passage, dp_pa: f64, cfg: &EstimateConfig) -> f64 {
    if !(dp_pa > 0.0) {
        return 0.0;
    }
    let a_in = (passage.inlet_area_mm2 * 1e-6).max(1e-12);
    // Seed from K = 1, which is the right order for any duct worth printing.
    let mut v = (2.0 * dp_pa / cfg.fluid.rho.max(1e-9)).sqrt();
    for _ in 0..12 {
        let r = solve(passage, v * a_in, cfg);
        let k = r.loss_coefficient.k.mean;
        if !(k > 1e-9) {
            break;
        }
        let next = (2.0 * dp_pa / (cfg.fluid.rho.max(1e-9) * k)).sqrt();
        if !next.is_finite() || next <= 0.0 {
            break;
        }
        if (next - v).abs() <= 1e-12 * next {
            v = next;
            break;
        }
        v = next;
    }
    v * a_in
}

fn solve(passage: &Passage, q_m3s: f64, cfg: &EstimateConfig) -> EstimateReport {
    let rho = cfg.fluid.rho.max(1e-9);
    let nu = cfg.fluid.nu.max(1e-12);
    let a_in = (passage.inlet_area_mm2 * 1e-6).max(1e-12);
    let a_out = (passage.outlet_area_mm2 * 1e-6).max(1e-12);
    let q = q_m3s.max(0.0);
    let v_in = q / a_in;
    let v_out = q / a_out;
    let head = |v: f64| 0.5 * rho * v * v;

    let mut warnings = passage.warnings.clone();
    let mut elements: Vec<LossElement> = Vec::with_capacity(passage.bends.len() + 4);

    // --- Friction, integrated station by station ------------------------
    //
    // Every station's f comes from the same correlation in the same regime, so
    // its errors are perfectly correlated along the duct: they are summed
    // linearly, not in quadrature. Treating them as independent would claim
    // that a duct which is rougher than assumed is rougher only in places.
    let mut dp_fric = 0.0;
    let mut dp_fric_sigma = 0.0;
    let mut f_weighted = 0.0;
    let mut span_total = 0.0;
    let mut regime_span = [0.0f64; 3];
    for st in &passage.stations {
        let a = (st.area_mm2 * 1e-6).max(1e-12);
        let dh_m = (st.hydraulic_diameter_mm * 1e-3).max(1e-9);
        let v = q / a;
        let re = v * dh_m / nu;
        let rel_rough = (cfg.roughness_mm / st.hydraulic_diameter_mm.max(1e-9)).max(0.0);
        let fr = friction_factor(re, rel_rough, Section::Rectangular { aspect: st.aspect });
        let dp = fr.f * (st.span_mm * 1e-3 / dh_m) * head(v);
        dp_fric += dp;
        dp_fric_sigma += dp * fr.sigma_rel;
        f_weighted += fr.f * st.span_mm;
        span_total += st.span_mm;
        regime_span[regime_index(fr.regime)] += st.span_mm;
    }
    let mean_f = if span_total > 0.0 { f_weighted / span_total } else { 0.0 };
    let regime = dominant_regime(&regime_span);
    if regime == Regime::Transitional {
        warnings.push(
            "most of the duct is in the laminar-turbulent transition, where the friction \
             factor is not a function of Reynolds number alone; the friction term here is \
             a blend and is uncertain by 50%"
                .into(),
        );
    }
    elements.push(LossElement {
        kind: ElementKind::Friction,
        label: format!("friction L/D_h={:.1}", passage.length_mm / passage.mean_dh_mm.max(1e-9)),
        k_local: k_from_dp(Band::new(dp_fric, dp_fric_sigma), head(v_in)),
        v_local_ms: v_in,
        k_ref: Band::ZERO,
        dp_pa: Band::new(dp_fric, dp_fric_sigma),
        share: 0.0,
        s_start_mm: 0.0,
        s_end_mm: passage.length_mm,
        source: if regime == Regime::Laminar {
            "Shah & London (1978) f*Re, rectangular duct"
        } else if cfg.roughness_mm > 0.0 {
            "Colebrook-White (1939)"
        } else {
            "Blasius (1913) / Colebrook-White (1939)"
        },
    });

    // --- Bends ----------------------------------------------------------
    for b in &passage.bends {
        let v = velocity_at(passage, q, 0.5 * (b.s_start_mm + b.s_end_mm), v_in);
        let k = bend_local_k(b.angle_deg, b.r_over_dh, b.aspect_hw);
        elements.push(LossElement {
            kind: ElementKind::Bend,
            label: format!("bend {:.0} deg r/D={:.2}", b.angle_deg, b.r_over_dh),
            k_local: k,
            v_local_ms: v,
            k_ref: Band::ZERO,
            dp_pa: k.scale(head(v)),
            share: 0.0,
            s_start_mm: b.s_start_mm,
            s_end_mm: b.s_end_mm,
            source: if b.r_over_dh < 0.5 {
                "Idelchik 6-7 mitre anchor x aspect factor 6-1"
            } else {
                "Idelchik Diagram 6-1 (A1 B1 C1)"
            },
        });
    }

    // --- Area changes ---------------------------------------------------
    for t in &passage.transitions {
        // Both correlations are referenced to the faster side, which is always
        // the smaller area. See the module docs of `loss`.
        let a_small = t.area_in_mm2.min(t.area_out_mm2).max(1e-9) * 1e-6;
        let v = q / a_small;
        let (kind, k, source) = if t.is_contraction {
            (
                ElementKind::Contraction,
                gradual_contraction_k(t.area_ratio(), t.included_angle_deg),
                "Idelchik Diagram 4-9",
            )
        } else {
            (
                ElementKind::Expansion,
                gradual_expansion_k(t.area_ratio(), t.included_angle_deg),
                "Gibson / Idelchik Diagram 5-2",
            )
        };
        elements.push(LossElement {
            kind,
            label: format!(
                "{} {:.2}:1 @{:.0} deg",
                if t.is_contraction { "contraction" } else { "expansion" },
                1.0 / t.area_ratio().max(1e-9),
                t.included_angle_deg
            ),
            k_local: k,
            v_local_ms: v,
            k_ref: Band::ZERO,
            dp_pa: k.scale(head(v)),
            share: 0.0,
            s_start_mm: t.s_start_mm,
            s_end_mm: t.s_end_mm,
            source,
        });
    }

    // --- Entry and exit -------------------------------------------------
    if cfg.entry != EntryCondition::None {
        let k = entry_k(cfg.entry);
        elements.push(LossElement {
            kind: ElementKind::Entry,
            label: "entry".into(),
            k_local: k,
            v_local_ms: v_in,
            k_ref: Band::ZERO,
            dp_pa: k.scale(head(v_in)),
            share: 0.0,
            s_start_mm: 0.0,
            s_end_mm: 0.0,
            source: "Idelchik Diagram 3-4 / ASHRAE ED1",
        });
    }
    if cfg.exit != ExitCondition::None {
        let k = exit_k(cfg.exit);
        elements.push(LossElement {
            kind: ElementKind::Exit,
            label: "exit".into(),
            k_local: k,
            v_local_ms: v_out,
            k_ref: Band::ZERO,
            dp_pa: k.scale(head(v_out)),
            share: 0.0,
            s_start_mm: passage.length_mm,
            s_end_mm: passage.length_mm,
            source: "Borda-Carnot, A2 -> infinity",
        });
    }

    // --- Totals ---------------------------------------------------------
    let v_ref = cfg.reference.pick(v_in, v_out);
    let head_ref = head(v_ref);
    let dp_correlation = Band::sum(elements.iter().map(|e| e.dp_pa));
    // The extraction's own error enters as a multiplicative factor on the
    // whole network -- if D_h is read 10% small, it is read small everywhere --
    // so it combines with the correlation error in quadrature at the end
    // rather than being sprinkled over the elements.
    //
    // The method floor is applied to the correlation part *first* and the
    // geometry error stacks on top of it, rather than the floor being taken
    // over the combined figure. Otherwise a badly extracted passage and a
    // perfect one both report exactly 20% - the floor swallows the very signal
    // it exists to protect - and the user has no way to tell that the number
    // came off a duct three cells wide.
    let geom = passage.confidence.geometry_sigma();
    let correlation =
        dp_correlation.sigma.max((dp_correlation.mean * cfg.uncertainty_floor).abs());
    let sigma = (correlation.powi(2) + (dp_correlation.mean * geom).powi(2)).sqrt();
    let dp_total = Band::new(dp_correlation.mean, sigma);

    for e in elements.iter_mut() {
        // K on the report's reference velocity: K_ref = K_local (V_local/V_ref)^2.
        e.k_ref = k_from_dp(e.dp_pa, head_ref);
        e.share = if dp_total.mean.abs() > 0.0 { e.dp_pa.mean / dp_total.mean } else { 0.0 };
    }
    elements.sort_by(|a, b| {
        b.dp_pa.mean.partial_cmp(&a.dp_pa.mean).unwrap_or(std::cmp::Ordering::Equal)
    });

    let k_total = k_from_dp(dp_total, head_ref);
    if head_ref <= 0.0 {
        warnings.push(
            "the reference velocity is zero, so the loss coefficient is undefined; the \
             pressure drop is still reported and is also zero"
                .into(),
        );
    }
    // p_t = p_s + 1/2 rho V^2, so dp_s = dp_t + 1/2 rho (V_out^2 - V_in^2). On a
    // contracting duct V_out > V_in, which makes the static drop *larger* than
    // the total drop -- the opposite of what most people expect, and the usual
    // reason a wall-tapping measurement looks like it disagrees with a
    // Pitot-based one.
    let dp_static = Band::new(dp_total.mean + head(v_out) - head(v_in), dp_total.sigma);

    let dh = passage.mean_dh_mm;
    EstimateReport {
        flow_m3s: q,
        inlet_velocity_ms: v_in,
        outlet_velocity_ms: v_out,
        total_pressure_drop_pa: dp_total,
        static_pressure_drop_pa: dp_static,
        loss_coefficient: LossCoefficient {
            k: k_total,
            band: LossBand::of(k_total.mean),
            reference: cfg.reference,
            v_ref_ms: v_ref,
        },
        elements,
        reynolds: v_in * dh * 1e-3 / nu,
        hydraulic_diameter_mm: dh,
        regime,
        mean_friction_factor: mean_f,
        confidence: passage.confidence,
        warnings,
    }
}

fn k_from_dp(dp: Band, head_ref: f64) -> Band {
    if head_ref > 0.0 {
        dp.scale(1.0 / head_ref)
    } else {
        Band::ZERO
    }
}

/// Bulk velocity at an arc position, from the nearest station's area.
fn velocity_at(passage: &Passage, q: f64, s_mm: f64, fallback: f64) -> f64 {
    let mut best: Option<&Station> = None;
    let mut best_d = f64::INFINITY;
    for st in &passage.stations {
        let d = (st.s_mm - s_mm).abs();
        if d < best_d {
            best_d = d;
            best = Some(st);
        }
    }
    match best {
        Some(st) if st.area_mm2 > 0.0 => q / (st.area_mm2 * 1e-6),
        _ => fallback,
    }
}

fn regime_index(r: Regime) -> usize {
    match r {
        Regime::Laminar => 0,
        Regime::Transitional => 1,
        Regime::Turbulent => 2,
    }
}

fn dominant_regime(span: &[f64; 3]) -> Regime {
    let mut best = 0;
    for i in 1..3 {
        if span[i] > span[best] {
            best = i;
        }
    }
    match best {
        0 => Regime::Laminar,
        1 => Regime::Transitional,
        _ => Regime::Turbulent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::centreline::{Bend, Transition};
    use glam::Vec3;

    /// A synthetic straight passage of constant section, built without going
    /// near a voxel. Lets the network be tested against a closed-form answer
    /// with the extraction taken out of the loop.
    fn straight(length_mm: f64, w: f64, h: f64, n: usize) -> Passage {
        let area = w * h;
        let dh = 2.0 * w * h / (w + h);
        let span = length_mm / n as f64;
        let stations = (0..n)
            .map(|i| Station {
                s_mm: (i as f64 + 0.5) * span,
                span_mm: span,
                point_mm: Vec3::new(0.0, 0.0, (i as f64 + 0.5) as f32 * span as f32),
                tangent: Vec3::Z,
                area_mm2: area,
                hydraulic_diameter_mm: dh,
                width_mm: w.max(h),
                height_mm: w.min(h),
                aspect: w.max(h) / w.min(h),
                curvature_per_mm: 0.0,
                major_axis: Vec3::X,
                minor_axis: Vec3::Y,
                hydraulic_diameter_faces_mm: dh,
            })
            .collect();
        Passage {
            dx_mm: 0.5,
            stations,
            bends: Vec::new(),
            transitions: Vec::new(),
            length_mm,
            volume_mm3: area * length_mm,
            dead_volume_mm3: 0.0,
            inlet_area_mm2: area,
            outlet_area_mm2: area,
            inlet_dh_mm: dh,
            outlet_dh_mm: dh,
            mean_dh_mm: dh,
            min_cells_across: 20.0,
            confidence: Confidence::Good,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn a_straight_duct_reproduces_darcy_weisbach_exactly() {
        // The whole network on a duct with nothing but friction must equal
        // f (L/D_h) 1/2 rho V^2 evaluated by hand. Any bookkeeping error --
        // a stray millimetre-to-metre, a double-counted span -- shows up here.
        let p = straight(200.0, 20.0, 20.0, 64);
        let cfg = EstimateConfig::default();
        for v in [1.0, 3.0, 8.0] {
            let r = estimate(&p, Drive::InletVelocity(v), &cfg);
            let dh_m = p.mean_dh_mm * 1e-3;
            let re = v * dh_m / cfg.fluid.nu;
            let f = friction_factor(re, 0.0, Section::Rectangular { aspect: 1.0 }).f;
            let want = f * (0.2 / dh_m) * 0.5 * cfg.fluid.rho * v * v;
            assert!(
                (r.total_pressure_drop_pa.mean / want - 1.0).abs() < 1e-9,
                "V = {v}: got {} Pa, hand-calc {want} Pa",
                r.total_pressure_drop_pa.mean
            );
            // ...and K must be exactly f L/D_h.
            assert!((r.loss_coefficient.k.mean - f * 0.2 / dh_m).abs() < 1e-9);
            assert!((r.mean_friction_factor - f).abs() < 1e-12);
        }
    }

    #[test]
    fn flow_scales_with_the_inlet_area_and_the_units_line_up() {
        let p = straight(100.0, 40.0, 10.0, 32);
        let r = estimate(&p, Drive::InletVelocity(5.0), &EstimateConfig::default());
        // 400 mm^2 at 5 m/s = 2.0e-3 m^3/s = 2 L/s.
        assert!((r.flow_m3s - 2.0e-3).abs() < 1e-12);
        assert!((r.litres_per_second() - 2.0).abs() < 1e-9);
        assert!((r.cfm() - 4.23776).abs() < 1e-4);
        assert!((r.inlet_velocity_ms - 5.0).abs() < 1e-12);
        // Constant section: in and out must agree.
        assert!((r.outlet_velocity_ms - 5.0).abs() < 1e-12);
        // ...and driving by flow instead must give the same answer.
        let r2 = estimate(&p, Drive::Flow(2.0e-3), &EstimateConfig::default());
        assert!((r2.total_pressure_drop_pa.mean - r.total_pressure_drop_pa.mean).abs() < 1e-12);
    }

    #[test]
    fn solving_for_a_fan_pressure_inverts_the_forward_solve() {
        // Round trip: pick a velocity, get its dp, feed that dp back and
        // recover the velocity. This is what a fan-curve intersection does.
        let p = straight(150.0, 30.0, 12.0, 48);
        let cfg = EstimateConfig::as_installed();
        for v in [0.5, 2.0, 5.0, 12.0] {
            let forward = estimate(&p, Drive::InletVelocity(v), &cfg);
            let back = estimate(
                &p,
                Drive::TotalPressure(forward.total_pressure_drop_pa.mean),
                &cfg,
            );
            assert!(
                (back.inlet_velocity_ms / v - 1.0).abs() < 1e-6,
                "V = {v} -> dp = {:.3} Pa -> V = {}",
                forward.total_pressure_drop_pa.mean,
                back.inlet_velocity_ms
            );
        }
        // Zero and negative pressure produce no flow, not a NaN.
        assert_eq!(estimate(&p, Drive::TotalPressure(0.0), &cfg).flow_m3s, 0.0);
        assert_eq!(estimate(&p, Drive::TotalPressure(-5.0), &cfg).flow_m3s, 0.0);
    }

    #[test]
    fn the_exit_loss_dominates_a_short_duct_and_is_off_by_default() {
        let p = straight(60.0, 20.0, 20.0, 32);
        let bare = estimate(&p, Drive::InletVelocity(5.0), &EstimateConfig::default());
        let installed = estimate(
            &p,
            Drive::InletVelocity(5.0),
            &EstimateConfig { exit: ExitCondition::Discharge, ..Default::default() },
        );
        // K = 1 exactly is added, on the same velocity.
        assert!(
            (installed.loss_coefficient.k.mean - bare.loss_coefficient.k.mean - 1.0).abs() < 1e-9
        );
        assert!(installed.k_of(ElementKind::Exit).mean > 0.99);
        assert!(bare.k_of(ElementKind::Exit).mean == 0.0);
        // ...and on a short duct it is the biggest term by a mile.
        assert_eq!(installed.dominant_element().map(|e| e.kind), Some(ElementKind::Exit));
        assert!(installed.dominant_element().is_some_and(|e| e.share > 0.8));
    }

    #[test]
    fn static_and_total_drop_differ_by_exactly_the_velocity_head_change() {
        // A contracting duct: the static drop must EXCEED the total drop,
        // which is the counter-intuitive sign everyone gets wrong once.
        let mut p = straight(100.0, 20.0, 20.0, 32);
        p.outlet_area_mm2 = p.inlet_area_mm2 / 2.0;
        let cfg = EstimateConfig::default();
        let r = estimate(&p, Drive::InletVelocity(4.0), &cfg);
        assert!((r.outlet_velocity_ms - 8.0).abs() < 1e-9);
        let expect = r.total_pressure_drop_pa.mean + 0.5 * cfg.fluid.rho * (8.0 * 8.0 - 16.0);
        assert!((r.static_pressure_drop_pa.mean - expect).abs() < 1e-9);
        assert!(r.static_pressure_drop_pa.mean > r.total_pressure_drop_pa.mean);
    }

    #[test]
    fn the_reference_velocity_convention_moves_k_by_the_area_ratio_squared() {
        let mut p = straight(100.0, 20.0, 20.0, 32);
        p.outlet_area_mm2 = p.inlet_area_mm2 / 1.85;
        let at = |reference| {
            estimate(
                &p,
                Drive::InletVelocity(2.0),
                &EstimateConfig { reference, ..Default::default() },
            )
            .loss_coefficient
            .k
            .mean
        };
        let ratio = at(ReferenceVelocity::Inlet) / at(ReferenceVelocity::Outlet);
        assert!((ratio - 1.85 * 1.85).abs() < 0.01, "K moved by {ratio}x, expected 3.42");
        // `Faster` picks the outlet here, which gives the flattering answer.
        assert!((at(ReferenceVelocity::Faster) - at(ReferenceVelocity::Outlet)).abs() < 1e-12);
    }

    #[test]
    fn the_elements_sum_to_the_total_and_the_shares_sum_to_one() {
        let mut p = straight(180.0, 30.0, 12.0, 64);
        p.bends.push(Bend {
            s_start_mm: 40.0,
            s_end_mm: 90.0,
            angle_deg: 90.0,
            radius_mm: 25.0,
            hydraulic_diameter_mm: p.mean_dh_mm,
            r_over_dh: 25.0 / p.mean_dh_mm,
            width_mm: 12.0,
            height_mm: 30.0,
            aspect_hw: 30.0 / 12.0,
        });
        p.transitions.push(Transition {
            s_start_mm: 100.0,
            s_end_mm: 140.0,
            area_in_mm2: 360.0,
            area_out_mm2: 200.0,
            included_angle_deg: 20.0,
            is_contraction: true,
        });
        p.outlet_area_mm2 = 200.0;
        let cfg = EstimateConfig::as_installed();
        let r = estimate(&p, Drive::InletVelocity(3.0), &cfg);

        let dp_sum: f64 = r.elements.iter().map(|e| e.dp_pa.mean).sum();
        assert!((dp_sum - r.total_pressure_drop_pa.mean).abs() < 1e-9);
        let k_sum: f64 = r.elements.iter().map(|e| e.k_ref.mean).sum();
        assert!((k_sum - r.loss_coefficient.k.mean).abs() < 1e-9, "{k_sum} vs {}", r.loss_coefficient.k.mean);
        let share: f64 = r.elements.iter().map(|e| e.share).sum();
        assert!((share - 1.0).abs() < 1e-9);
        // Every kind is present and accounted for.
        for kind in [
            ElementKind::Friction,
            ElementKind::Bend,
            ElementKind::Contraction,
            ElementKind::Entry,
            ElementKind::Exit,
        ] {
            assert!(r.k_of(kind).mean > 0.0, "{} missing", kind.label());
        }
        // Sorted largest first, so `dominant_element` means what it says.
        for w in r.elements.windows(2) {
            assert!(w[0].dp_pa.mean >= w[1].dp_pa.mean);
        }
        assert!(r.breakdown().contains("TOTAL"));
        assert!(r.summary().contains("L/s"));
        assert_eq!(r.scalars().len(), 11);
    }

    #[test]
    fn each_elements_k_local_is_on_its_own_velocity() {
        // K_ref = K_local * (V_local/V_ref)^2. The contraction sits at the
        // small end, so its local velocity is higher than the inlet reference
        // and its K_ref must be correspondingly larger.
        let mut p = straight(120.0, 30.0, 12.0, 48);
        p.transitions.push(Transition {
            s_start_mm: 20.0,
            s_end_mm: 100.0,
            area_in_mm2: 360.0,
            area_out_mm2: 180.0,
            included_angle_deg: 15.0,
            is_contraction: true,
        });
        p.outlet_area_mm2 = 180.0;
        let r = estimate(&p, Drive::InletVelocity(2.0), &EstimateConfig::default());
        let e = r
            .elements
            .iter()
            .find(|e| e.kind == ElementKind::Contraction)
            .expect("the contraction should appear");
        assert!((e.v_local_ms - 4.0).abs() < 1e-9, "local V = {}", e.v_local_ms);
        let scale = (e.v_local_ms / r.loss_coefficient.v_ref_ms).powi(2);
        assert!((e.k_ref.mean / e.k_local.mean - scale).abs() < 1e-9);
        assert!(e.k_ref.mean > e.k_local.mean);
    }

    #[test]
    fn the_reported_band_never_claims_better_than_the_method() {
        let p = straight(200.0, 20.0, 20.0, 64);
        let r = estimate(&p, Drive::InletVelocity(3.0), &EstimateConfig::default());
        assert!(
            r.total_pressure_drop_pa.relative_sigma() >= 0.20 - 1e-9,
            "band was {}",
            r.total_pressure_drop_pa
        );
        assert!(r.loss_coefficient.k.relative_sigma() >= 0.20 - 1e-9);
        // A poorly extracted passage must report a wider band, not the floor.
        let mut poor = p.clone();
        poor.confidence = Confidence::Poor;
        let rp = estimate(&poor, Drive::InletVelocity(3.0), &EstimateConfig::default());
        assert!(rp.total_pressure_drop_pa.relative_sigma() > 0.35);
        assert!((rp.total_pressure_drop_pa.mean - r.total_pressure_drop_pa.mean).abs() < 1e-12);
    }

    #[test]
    fn zero_flow_produces_zeros_rather_than_nan() {
        let p = straight(100.0, 20.0, 20.0, 32);
        let r = estimate(&p, Drive::InletVelocity(0.0), &EstimateConfig::as_installed());
        assert_eq!(r.flow_m3s, 0.0);
        assert!(r.total_pressure_drop_pa.mean.abs() < 1e-15);
        assert!(r.loss_coefficient.k.mean.is_finite());
        assert!(r.reynolds.abs() < 1e-12);
        assert!(r.warnings.iter().any(|w| w.contains("undefined")));
        // A negative velocity is clamped, not run backwards.
        let neg = estimate(&p, Drive::InletVelocity(-4.0), &EstimateConfig::default());
        assert_eq!(neg.flow_m3s, 0.0);
    }

    #[test]
    fn roughness_raises_the_friction_term_and_only_the_friction_term() {
        let mut p = straight(300.0, 8.0, 6.0, 64);
        p.bends.push(Bend {
            s_start_mm: 100.0,
            s_end_mm: 160.0,
            angle_deg: 90.0,
            radius_mm: 10.0,
            hydraulic_diameter_mm: p.mean_dh_mm,
            r_over_dh: 10.0 / p.mean_dh_mm,
            width_mm: 6.0,
            height_mm: 8.0,
            aspect_hw: 8.0 / 6.0,
        });
        // 20 m/s, not 5: a 6.9 mm passage at 5 m/s is only Re = 2200, i.e.
        // laminar, and laminar friction does not see roughness at all. That is
        // itself worth knowing -- a small printed duct at low flow gets its
        // surface finish for free.
        let smooth = estimate(&p, Drive::InletVelocity(20.0), &EstimateConfig::default());
        let printed = estimate(
            &p,
            Drive::InletVelocity(20.0),
            &EstimateConfig { roughness_mm: crate::PRINTED_ROUGHNESS_MM, ..Default::default() },
        );
        assert_eq!(smooth.regime, Regime::Turbulent, "the check needs a turbulent duct");
        assert!(printed.k_of(ElementKind::Friction).mean > 1.15 * smooth.k_of(ElementKind::Friction).mean);
        assert!(
            (printed.k_of(ElementKind::Bend).mean - smooth.k_of(ElementKind::Bend).mean).abs()
                < 1e-12
        );
        assert!(printed.mean_friction_factor > smooth.mean_friction_factor);
    }

    #[test]
    fn a_bend_costs_less_when_the_duct_turns_about_its_short_side() {
        // The actionable claim the breakdown makes. Same duct, same turn, the
        // section rotated 90 degrees about the flow axis.
        let mk = |aspect_hw: f64| {
            let mut p = straight(200.0, 30.0, 10.0, 48);
            p.bends.push(Bend {
                s_start_mm: 60.0,
                s_end_mm: 140.0,
                angle_deg: 90.0,
                radius_mm: 15.0,
                hydraulic_diameter_mm: p.mean_dh_mm,
                r_over_dh: 15.0 / p.mean_dh_mm,
                width_mm: 10.0,
                height_mm: 30.0,
                aspect_hw,
            });
            estimate(&p, Drive::InletVelocity(4.0), &EstimateConfig::default())
                .k_of(ElementKind::Bend)
                .mean
        };
        assert!(mk(0.33) > 1.4 * mk(3.0), "hard way {} vs easy way {}", mk(0.33), mk(3.0));
    }

    #[test]
    fn the_whole_solve_is_microseconds() {
        // The promise of the crate, asserted rather than hoped for. The bound
        // is deliberately loose (a debug build and a busy machine both move
        // it); anything anywhere near a millisecond would mean the network is
        // touching something it should not.
        let mut p = straight(200.0, 30.0, 12.0, 128);
        for i in 0..4 {
            p.bends.push(Bend {
                s_start_mm: 20.0 + 40.0 * i as f64,
                s_end_mm: 50.0 + 40.0 * i as f64,
                angle_deg: 45.0,
                radius_mm: 20.0,
                hydraulic_diameter_mm: p.mean_dh_mm,
                r_over_dh: 20.0 / p.mean_dh_mm,
                width_mm: 12.0,
                height_mm: 30.0,
                aspect_hw: 2.5,
            });
        }
        let cfg = EstimateConfig::as_installed();
        // Warm up, then time a batch so one scheduler hiccup cannot fail it.
        let _ = estimate(&p, Drive::InletVelocity(3.0), &cfg);
        let n = 1000;
        let t0 = std::time::Instant::now();
        let mut sink = 0.0;
        for i in 0..n {
            let v = 1.0 + (i % 8) as f64;
            sink += estimate(&p, Drive::InletVelocity(v), &cfg).total_pressure_drop_pa.mean;
        }
        let per = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        assert!(sink > 0.0);
        eprintln!("loss network: {per:.1} us per solve, 128 stations and 4 bends");
        assert!(per < 500.0, "{per:.1} us per solve is not interactive");
    }
}

//! The classical loss correlations, each with its source and validity range.
//!
//! Every function here is a published curve fit or an exact momentum balance.
//! None of them is tuned to make this app's numbers look good, and where a
//! correlation is being pushed past the range its data covers the doc comment
//! says so and the returned [`Band`] widens.
//!
//! # The one convention that has to be right
//!
//! A fitting's `K` is meaningless without the velocity it is referenced to.
//! Throughout this module:
//!
//! * **contraction**: `K` is on the **downstream** (smaller, faster) velocity
//! * **expansion**: `K` is on the **upstream** (smaller, faster) velocity
//! * **bend, entry, exit, friction**: `K` is on the local duct velocity
//!
//! These are the handbook conventions, and they are all "the faster of the
//! two", which is the mnemonic worth keeping. [`crate::network`] converts every
//! element onto one common reference velocity before summing.
//!
//! # Fitting loss versus friction loss
//!
//! A handbook elbow coefficient is a *lump*: it contains both the local
//! separation loss and the ordinary wall friction over the bend's own arc,
//! because a fitting is sold and tabulated as one object. A duct *network*
//! must not use that number, because its friction integral already runs along
//! the whole centreline including the arc, and adding the lump would count the
//! arc's friction twice.
//!
//! So this module exposes both, and the distinction is load-bearing:
//!
//! * [`elbow_k`] — the handbook lump. Compare this against ASHRAE tables.
//! * [`bend_local_k`] — the separation loss alone. This is what the network
//!   sums, alongside its own friction integral.

use crate::{interpolate, smoothstep, Band};

// ---------------------------------------------------------------------------
// Friction
// ---------------------------------------------------------------------------

/// Reynolds number below which the flow is taken as laminar.
///
/// The transition is not a number, it is a range that depends on inlet
/// disturbance, roughness and entry length; 2300 is the conventional lower
/// bound for pipe flow (Reynolds 1883, and every text since).
pub const RE_LAMINAR_MAX: f64 = 2300.0;

/// Reynolds number above which the flow is taken as fully turbulent. Between
/// this and [`RE_LAMINAR_MAX`] the friction factor is blended and the
/// uncertainty is enormous — see [`friction_factor`].
pub const RE_TURBULENT_MIN: f64 = 4000.0;

/// Cross-section shape, for the *laminar* friction factor only.
///
/// This distinction matters in laminar flow and barely matters in turbulent
/// flow, which is the single most misremembered fact about duct friction. In
/// laminar flow the hydraulic-diameter substitution is wrong by up to 50%
/// (`f*Re` runs from 56.9 for a square to 96 for parallel plates against 64 for
/// a circle). In turbulent flow, using `D_h` in the circular correlation is
/// accurate to about 10%, because the near-wall structure that sets the shear
/// is local and does not care about the far side of the duct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Section {
    /// Round pipe. `f * Re = 64`.
    Circular,
    /// Rectangular duct of the given aspect ratio (long side / short side,
    /// always `>= 1`).
    Rectangular { aspect: f64 },
}

/// Which branch of the friction law produced a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    Laminar,
    /// Between 2300 and 4000. The friction factor here is genuinely not a
    /// function of Reynolds number alone; it depends on how the flow was
    /// disturbed upstream and can sit anywhere between the two branches.
    Transitional,
    Turbulent,
}

impl Regime {
    pub fn label(self) -> &'static str {
        match self {
            Regime::Laminar => "laminar",
            Regime::Transitional => "transitional",
            Regime::Turbulent => "turbulent",
        }
    }
}

/// A friction factor with its regime and its uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Friction {
    /// Darcy-Weisbach `f`. Four times the Fanning factor; check which one a
    /// source means before comparing, because both are written `f`.
    pub f: f64,
    pub regime: Regime,
    /// One-sigma relative uncertainty on `f`.
    pub sigma_rel: f64,
}

impl Friction {
    pub fn band(&self) -> Band {
        Band::relative(self.f, self.sigma_rel)
    }
}

/// Laminar `f * Re` for a rectangular duct, Shah & London (1978), *Laminar Flow
/// Forced Convection in Ducts*, eq. for the rectangular channel.
///
/// In Fanning form the fit is
/// `f_F * Re = 24 (1 - 1.3553a + 1.9467a^2 - 1.7012a^3 + 0.9564a^4 - 0.2537a^5)`
/// with `a = short/long` in `(0, 1]`; this returns the Darcy value, four times
/// larger. Exact endpoints: `a = 1` gives 56.91 (square), `a -> 0` gives 96
/// (parallel plates). Valid for all aspect ratios; the fit is quoted to better
/// than 0.05%.
///
/// Note that neither endpoint is 64. A square duct is 11% *less* resistive than
/// the circular formula predicts and a flat slot is 50% *more*, so a printed
/// slot duct running laminar and sized with `64/Re` is badly wrong.
pub fn laminar_f_re_rectangular(aspect: f64) -> f64 {
    // `a` is short/long, so it is the reciprocal of our aspect convention.
    let a = (1.0 / aspect.max(1.0)).max(0.0).min(1.0);
    96.0 * (1.0 - 1.3553 * a + 1.9467 * a.powi(2) - 1.7012 * a.powi(3) + 0.9564 * a.powi(4)
        - 0.2537 * a.powi(5))
}

/// `f * Re` for a section: 64 for a circle, Shah & London for a rectangle.
pub fn laminar_f_re(section: Section) -> f64 {
    match section {
        Section::Circular => 64.0,
        Section::Rectangular { aspect } => laminar_f_re_rectangular(aspect),
    }
}

/// Blasius (1913): `f = 0.3164 Re^-0.25`, smooth wall.
///
/// **Validity: `3 x 10^3 < Re < 10^5`, hydraulically smooth only.** Outside
/// that it drifts from the data — by `Re = 10^6` it is 25% low — which is why
/// [`friction_factor`] hands off to Colebrook above `10^5`. Accurate to about
/// 2-3% inside its range, which is as good as any measurement of it.
pub fn blasius(re: f64) -> f64 {
    // `!(re > 0.0)` rather than `re <= 0.0` so a NaN Reynolds number -- which a
    // zero-area station would produce -- returns zero instead of propagating.
    if !(re > 0.0) {
        return 0.0;
    }
    0.3164 * re.powf(-0.25)
}

/// Colebrook-White (1939), solved iteratively:
/// `1/sqrt(f) = -2 log10( eps/(3.7 D) + 2.51 / (Re sqrt(f)) )`.
///
/// **Validity: `Re > 4000`, any relative roughness up to about 0.05.** This is
/// the implicit relation the Moody chart is drawn from. It reduces to the
/// smooth-wall Prandtl law as `eps -> 0` and to the fully-rough von Karman law
/// as `Re -> inf`, so it is the correlation to use whenever roughness matters —
/// and on a 3D-printed duct it always does; see [`crate::PRINTED_ROUGHNESS_MM`].
///
/// Seeded with Haaland's explicit approximation and then iterated on the
/// fixed-point form, which converges monotonically in a handful of steps
/// because `1/sqrt(f)` appears only inside a logarithm. Twelve iterations puts
/// it at machine precision; the cost is a dozen `log10` calls, i.e. nothing.
pub fn colebrook(re: f64, rel_roughness: f64) -> f64 {
    if !(re > 0.0) {
        return 0.0;
    }
    let eps = rel_roughness.max(0.0);
    // Haaland (1983), explicit, within 2% of Colebrook. A good starting point.
    let mut inv_sqrt_f = -1.8 * ((eps / 3.7).powf(1.11) + 6.9 / re).log10();
    if !inv_sqrt_f.is_finite() || inv_sqrt_f <= 0.0 {
        inv_sqrt_f = 1.0;
    }
    for _ in 0..12 {
        let next = -2.0 * (eps / 3.7 + 2.51 * inv_sqrt_f / re).log10();
        if !next.is_finite() || next <= 0.0 {
            break;
        }
        inv_sqrt_f = next;
    }
    1.0 / (inv_sqrt_f * inv_sqrt_f)
}

/// Haaland (1983): an explicit approximation to Colebrook, within ~2%.
/// Exposed because it is what a spreadsheet uses and it is useful for checking
/// that the [`colebrook`] iteration landed somewhere sane.
pub fn haaland(re: f64, rel_roughness: f64) -> f64 {
    if !(re > 0.0) {
        return 0.0;
    }
    let x = -1.8 * ((rel_roughness.max(0.0) / 3.7).powf(1.11) + 6.9 / re).log10();
    1.0 / (x * x)
}

/// The Darcy friction factor, across every regime, continuously.
///
/// * `Re <= 2300`: `f = C(shape) / Re`, exact for fully-developed laminar flow.
/// * `Re >= 4000`: Blasius when the wall is smooth and `Re < 10^5`, otherwise
///   Colebrook.
/// * in between: a smoothstep blend of the two branches, evaluated at the
///   *actual* Reynolds number, so the result is `C^1` and lands exactly on
///   `C/Re` at 2300 and exactly on the turbulent branch at 4000.
///
/// The blend is a presentation choice, not physics. Real transitional flow is
/// bistable and hysteretic; the honest statement is "somewhere between these
/// two curves", and that is what the returned `sigma_rel` of 0.5 says. A
/// discontinuous step would be worse in every way — it would make the estimator
/// jump under a slider and it would hide the uncertainty behind a crisp-looking
/// number.
///
/// One artefact is worth knowing about: just above 2300 the blend weight is
/// still near zero while the laminar branch is still falling as `1/Re`, so `f`
/// dips about 6% below its value at 2300 (bottoming out near `Re = 2600`)
/// before climbing to the turbulent branch. Removing the dip would need a kink
/// at 2300 — a jump in slope under the user's slider — to buy a 6% correction
/// inside a band that is 50% wide. It is not worth it, so the dip stays and is
/// documented instead.
///
/// `rel_roughness` is `eps / D_h`, dimensionless.
pub fn friction_factor(re: f64, rel_roughness: f64, section: Section) -> Friction {
    let f_lam = |re: f64| if re > 0.0 { laminar_f_re(section) / re } else { 0.0 };
    let smooth = rel_roughness <= 1e-9;
    let f_turb = |re: f64| {
        // Blasius inside its window, Colebrook everywhere else. They agree to
        // ~2% at the handover, so the switch does not produce a visible step.
        if smooth && re < 1.0e5 {
            blasius(re)
        } else {
            colebrook(re, rel_roughness)
        }
    };

    if !(re > 0.0) {
        return Friction { f: 0.0, regime: Regime::Laminar, sigma_rel: 0.0 };
    }
    if re <= RE_LAMINAR_MAX {
        // Fully-developed laminar friction is an exact solution of the
        // Navier-Stokes equations, not a fit. The 3% allows for the entrance
        // region, which is genuinely more resistive.
        return Friction { f: f_lam(re), regime: Regime::Laminar, sigma_rel: 0.03 };
    }
    if re >= RE_TURBULENT_MIN {
        let sigma = if smooth && re < 1.0e5 {
            // Blasius in its own window.
            0.05
        } else if smooth {
            0.08
        } else {
            // Colebrook's own scatter against the data it was fitted to is
            // +/-10-15%, and the roughness of a printed surface is not known to
            // better than a factor of two anyway.
            0.15
        };
        return Friction { f: f_turb(re), regime: Regime::Turbulent, sigma_rel: sigma };
    }

    let w = smoothstep((re - RE_LAMINAR_MAX) / (RE_TURBULENT_MIN - RE_LAMINAR_MAX));
    Friction {
        f: (1.0 - w) * f_lam(re) + w * f_turb(re),
        regime: Regime::Transitional,
        sigma_rel: 0.50,
    }
}

// ---------------------------------------------------------------------------
// Bends
// ---------------------------------------------------------------------------

/// Idelchik's angle factor `A1`, Diagram 6-1: how a bend's loss scales with
/// turn angle at fixed radius ratio. Normalised so `A1(90 deg) = 1`.
///
/// I. E. Idelchik, *Handbook of Hydraulic Resistance*, 3rd ed., Diagram 6-1.
/// Tabulated 0-180 degrees; clamped outside.
fn angle_factor(angle_deg: f64) -> f64 {
    const TABLE: [(f64, f64); 11] = [
        (0.0, 0.0),
        (20.0, 0.31),
        (30.0, 0.45),
        (45.0, 0.60),
        (60.0, 0.78),
        (75.0, 0.90),
        (90.0, 1.00),
        (110.0, 1.13),
        (130.0, 1.20),
        (150.0, 1.28),
        (180.0, 1.40),
    ];
    interpolate(angle_deg.abs(), &TABLE)
}

/// Idelchik's radius factor `B1`, Diagram 6-1:
///
/// ```text
/// B1 = 0.21 / (r/D_h)^2.5   for r/D_h <  1
/// B1 = 0.21 / (r/D_h)^0.5   for r/D_h >= 1
/// ```
///
/// **Validity: `0.5 <= r/D_h <= 10`.** Both branches meet at `B1 = 0.21` where
/// `r/D_h = 1`, which is why the exponent is allowed to change there. Below
/// 0.5 the `-2.5` branch runs away — it would give 6.8 at `r/D_h = 0.25` and
/// infinity at zero — so [`bend_local_k`] switches to the mitre correlation
/// instead of extrapolating it. That switch is the single most consequential
/// decision in this module and it is why the mitre gets its own anchor rather
/// than a limit.
fn radius_factor(r_over_dh: f64) -> f64 {
    let r = r_over_dh.max(1e-6);
    if r < 1.0 {
        0.21 / r.powf(2.5)
    } else {
        0.21 / r.sqrt()
    }
}

/// Idelchik's aspect factor `C1`, Diagram 6-1, as a function of `H/W`, where
/// `W` is the section dimension **in the plane of the turn** and `H` the one
/// perpendicular to it.
///
/// The trend is the useful part for a designer: a *tall* bend (`H/W` around 2-3,
/// turning about its short side) is 15% cheaper than square, and a *flat wide*
/// one (`H/W = 0.25`, turning the hard way) is 30% dearer. Rotating a duct 90
/// degrees about its own axis is free and buys you that.
///
/// I. E. Idelchik, Diagram 6-1, tabulated `H/W = 0.25 .. 8`.
///
/// Below `H/W = 0.25` the table stops, and rather than clamping — which would
/// claim a flat slot is no worse than 1.30 however flat it gets — this
/// continues the table's own local power law, `C1 ~ (H/W)^-0.152`, fitted to
/// the 0.25-to-0.5 interval. Anything below 0.25 is therefore an
/// **extrapolation**, and [`bend_local_k`] widens the band when it is used.
fn aspect_factor(h_over_w: f64) -> f64 {
    const TABLE: [(f64, f64); 12] = [
        (0.25, 1.30),
        (0.50, 1.17),
        (0.75, 1.09),
        (1.00, 1.00),
        (1.50, 0.90),
        (2.00, 0.85),
        (3.00, 0.85),
        (4.00, 0.90),
        (5.00, 0.95),
        (6.00, 0.98),
        (7.00, 1.00),
        (8.00, 1.00),
    ];
    let hw = h_over_w.max(1e-6);
    if hw >= 0.25 {
        return interpolate(hw, &TABLE);
    }
    // ln(1.30/1.17) / ln(0.50/0.25) = 0.152. Floored at H/W = 0.02 (a 50:1
    // slot) so a degenerate one-cell-thick station cannot drive the factor to
    // infinity; C1(0.02) = 1.91, which is already past anything measured.
    const P: f64 = 0.152_25;
    1.30 * (0.25 / hw.max(0.02)).powf(P)
}

/// True when [`aspect_factor`] had to leave its table.
fn aspect_is_extrapolated(h_over_w: f64) -> bool {
    h_over_w < 0.25 || h_over_w > 8.0
}

/// Local (separation) loss coefficient of a bend, on the duct velocity.
///
/// `K_loc = A1(angle) * B1(r/D_h) * C1(H/W)` for a radiused bend, and a
/// separate anchored value for a mitre. **Friction along the bend's own arc is
/// deliberately excluded** — see the module docs — so this is what a duct
/// network sums next to its friction integral. For the handbook lump, use
/// [`elbow_k`].
///
/// The mitre branch (`r/D_h < 0.5`) is anchored at `K = 1.30` for a square
/// 90-degree unvaned mitre and scaled by the same aspect factor. Sources for
/// that anchor: Idelchik Diagram 6-7 gives 1.2-1.4 for a sharp 90-degree
/// elbow; Crane TP-410 gives `60 f_T ~ 1.2`; ASHRAE's rectangular mitred elbow
/// without vanes runs 1.0-1.5 across `H/W`. All three agree on "about 1.2-1.5
/// at square aspect", and the aspect factor then reproduces the ASHRAE spread:
/// `1.30 * C1(0.25) = 1.69` for a flat wide mitre against ASHRAE's ~1.5, and
/// `1.30 * C1(2) = 1.11` for a tall one against ASHRAE's ~1.0-1.1.
///
/// Between `r/D_h = 0.5` and the mitre there is no data and the geometry is
/// ambiguous anyway (is a 0.3-radius corner a bend or a chamfered mitre?), so
/// the two branches are blended across `0.25 < r/D_h < 0.5` and the band is
/// widened there.
///
/// **Uncertainty: 25% for a tabulated radiused bend, 30% for a mitre, 40% when
/// the aspect ratio is off the end of the table or the radius is in the
/// blend.** Handbook fitting coefficients are measured on isolated fittings
/// with fully-developed approach flow and a long straight downstream. A printed
/// part gives them none of that.
pub fn bend_local_k(angle_deg: f64, r_over_dh: f64, aspect_hw: f64) -> Band {
    /// Anchor for a square, unvaned, 90-degree mitre.
    const MITRE_K90: f64 = 1.30;
    const MITRE_LIMIT: f64 = 0.25;
    const RADIUS_LIMIT: f64 = 0.5;

    let a1 = angle_factor(angle_deg);
    let c1 = aspect_factor(aspect_hw);
    let r = r_over_dh.max(0.0);

    let k_mitre = MITRE_K90 * a1 * c1;
    let k_radius = a1 * radius_factor(r.max(RADIUS_LIMIT)) * c1;

    let (k, mut sigma): (f64, f64) = if r <= MITRE_LIMIT {
        (k_mitre, 0.30)
    } else if r >= RADIUS_LIMIT {
        (k_radius, 0.25)
    } else {
        let w = smoothstep((r - MITRE_LIMIT) / (RADIUS_LIMIT - MITRE_LIMIT));
        ((1.0 - w) * k_mitre + w * k_radius, 0.40)
    };
    if aspect_is_extrapolated(aspect_hw) {
        sigma = sigma.max(0.40);
    }
    if r > 10.0 {
        // Past the table; a very gentle curve is barely a fitting at all and
        // the friction term dominates anyway.
        sigma = sigma.max(0.40);
    }
    Band::relative(k, sigma)
}

/// The **handbook lump** for an elbow: separation loss plus the friction over
/// the bend's own arc.
///
/// `K = K_loc + f * (theta * r) / D_h`, where `theta * r` is the developed
/// length of the arc. This is the number an ASHRAE or Idelchik table quotes and
/// the number to compare against them; it is *not* the number a duct network
/// should sum, because the network's friction integral already covers the arc.
///
/// A worked check, which is also the crate's headline validation case: a smooth
/// rectangular 90-degree elbow at `r/D_h = 1.5`, square aspect, `f = 0.03`
/// gives `K_loc = 1.0 * 0.21/sqrt(1.5) * 1.0 = 0.171` and
/// `K_fr = 0.03 * (pi/2) * 1.5 = 0.071`, so `K = 0.243` — inside the
/// universally quoted 0.2-0.3 for a well-radiused elbow, from two independent
/// terms neither of which was tuned to land there.
pub fn elbow_k(angle_deg: f64, r_over_dh: f64, aspect_hw: f64, f_darcy: f64) -> Band {
    let local = bend_local_k(angle_deg, r_over_dh, aspect_hw);
    let arc_over_dh = angle_deg.abs().to_radians() * r_over_dh.max(0.0);
    // The arc friction is as well known as f is; give it f's own 10%. `f` is
    // clamped to [0, 1]: no real Darcy factor exceeds about 0.1, so a larger one
    // is a caller bug, and letting an infinity through here would turn the whole
    // report into NaN.
    let friction = Band::relative(f_darcy.max(0.0).min(1.0) * arc_over_dh, 0.10);
    Band::sum([local, friction])
}

// ---------------------------------------------------------------------------
// Area changes
// ---------------------------------------------------------------------------

/// Sudden contraction, `K = 0.5 (1 - A2/A1)`, **on the downstream velocity**.
///
/// Weisbach's form, reproduced in Idelchik Diagram 4-9 and every text since.
/// The physics is not the contraction itself — accelerating flow is stable and
/// nearly lossless — but the *vena contracta* just downstream of it and the
/// re-expansion from it back to the pipe wall. That is why the coefficient is
/// referenced to the downstream velocity and why it goes to 0.5 for a
/// contraction from a plenum (`A2/A1 -> 0`), matching the sharp-edged entry
/// loss, which is the same event.
///
/// **Validity: turbulent flow, sharp square-edged step, `0 <= A2/A1 <= 1`.**
/// Accurate to about 15%; Idelchik's own refinement `0.5(1 - A2/A1)^0.75` sits
/// inside that band across the whole range, so the simpler form is kept.
pub fn sudden_contraction_k(a2_over_a1: f64) -> Band {
    let ratio = a2_over_a1.max(0.0).min(1.0);
    Band::relative(0.5 * (1.0 - ratio), 0.15)
}

/// Sudden expansion, `K = (1 - A1/A2)^2`, **on the upstream velocity**.
///
/// Borda-Carnot. Uniquely among the coefficients here this is not a curve fit:
/// it falls out of a control-volume momentum balance across the step, assuming
/// only that the static pressure acts uniformly on the shoulder. It matches
/// experiment to a few percent for square-edged steps, which is why its band is
/// 5% and not 25%.
///
/// Two limits worth remembering: `A2 -> inf` gives `K = 1`, i.e. discharging
/// into a room costs exactly the whole velocity head (see [`exit_k`]), and
/// `A1 = A2` gives zero.
///
/// **Validity: turbulent flow, square-edged step, `0 <= A1/A2 <= 1`.**
pub fn sudden_expansion_k(a1_over_a2: f64) -> Band {
    let ratio = a1_over_a2.max(0.0).min(1.0);
    let k = (1.0 - ratio) * (1.0 - ratio);
    Band::relative(k, 0.05)
}

/// Gradual contraction (a nozzle), on the downstream velocity.
///
/// `K = K_sudden * m(theta)` with Idelchik's included-angle multiplier
/// (Diagram 4-9):
///
/// ```text
/// m = sin(theta/2)        for theta <= 45 deg
/// m = sqrt(sin(theta/2))  for 45 < theta <= 180 deg
/// ```
///
/// `theta` is the **included** angle (the full cone angle, not the half-angle),
/// so `theta = 180 deg` is a sudden step and reduces exactly to
/// [`sudden_contraction_k`]. A 30-degree taper costs about a quarter of the
/// sudden value; below about 20 degrees the loss is nearly all friction and a
/// nozzle can be treated as free.
///
/// **Validity: turbulent flow, `0 < theta <= 180 deg`.** Accurate to about 30%
/// — worse than the sudden case, because the transition length and the wall
/// curvature both matter and neither is in the correlation.
pub fn gradual_contraction_k(a2_over_a1: f64, included_angle_deg: f64) -> Band {
    let sudden = sudden_contraction_k(a2_over_a1);
    let theta = included_angle_deg.abs().max(0.0).min(180.0);
    let half = (theta * 0.5).to_radians();
    let m = if theta <= 45.0 { half.sin() } else { half.sin().sqrt() };
    Band::relative(sudden.mean * m, if theta >= 179.0 { 0.15 } else { 0.30 })
}

/// Gradual expansion (a diffuser), on the upstream velocity.
///
/// `K = C(theta) * (1 - A1/A2)^2`, the Gibson form quoted by Idelchik
/// (Diagram 5-2) and ASHRAE:
///
/// ```text
/// C = 2.6 sin(theta/2)   for theta <= 45 deg
/// C = 1.0                for theta >  45 deg
/// ```
///
/// `theta` is the included angle. At `theta = 180 deg` this is exactly
/// Borda-Carnot, and at `theta = 45 deg` the two branches meet (`2.6 sin 22.5 =
/// 0.995`). The minimum is around 6-8 degrees, where `C` drops to about 0.15;
/// past 45 degrees the flow has separated off the wall and the diffuser is
/// doing nothing that a sudden step would not do — which is the practical
/// lesson, and the reason a duct that opens out too fast is worse than one that
/// does not taper at all.
///
/// **Validity: turbulent flow, `0 < theta <= 180 deg`, `A1/A2 <= 1`.** Around
/// 30%; diffuser performance is notoriously sensitive to the inlet boundary
/// layer, which a 1D model cannot see.
pub fn gradual_expansion_k(a1_over_a2: f64, included_angle_deg: f64) -> Band {
    let sudden = sudden_expansion_k(a1_over_a2);
    let theta = included_angle_deg.abs().max(0.0).min(180.0);
    let c = if theta <= 45.0 { 2.6 * (theta * 0.5).to_radians().sin() } else { 1.0 };
    Band::relative(sudden.mean * c, if theta >= 179.0 { 0.05 } else { 0.30 })
}

// ---------------------------------------------------------------------------
// Entry and exit
// ---------------------------------------------------------------------------

/// How the duct is fed at its inlet mouth.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EntryCondition {
    /// No entry loss. **The default**, and the right choice when comparing
    /// against the solver: the LBM measures total pressure *at the inlet
    /// plane*, with the flow already inside the duct, so whatever it cost to
    /// get there is not in its `dp` and must not be in ours either.
    #[default]
    None,
    /// Sharp-edged opening flush with a wall, drawing from still air.
    /// `K = 0.5`.
    Flush,
    /// Rounded lip of the given radius ratio `r/D_h`.
    Rounded { r_over_dh: f64 },
    /// A tube protruding into the plenum (Borda mouthpiece). `K = 0.8`, the
    /// worst practical inlet.
    ReEntrant,
    /// A properly contoured bellmouth. `K = 0.03`.
    BellMouth,
}

/// How the duct discharges at its outlet mouth.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ExitCondition {
    /// No exit loss. **The default**, matching a *total*-pressure drop measured
    /// between the two mouth planes: the exit kinetic energy is still in the
    /// total pressure at the outlet plane, so charging for it as well would
    /// double count.
    #[default]
    None,
    /// Discharge to still air. `K = 1.0` exactly — the whole velocity head is
    /// dumped into the room. This is the term to include when the question is
    /// "what must the fan supply", and it is usually the largest single loss in
    /// a short duct, which is worth knowing.
    Discharge,
}

/// Entry loss coefficient, on the duct velocity just inside the mouth.
///
/// The rounded table is Idelchik Diagram 3-4 / ASHRAE ED1: the loss collapses
/// astonishingly fast with a little radius, from 0.50 sharp to 0.09 at
/// `r/D_h = 0.10` and a floor of about 0.03. On a printed part that radius
/// costs nothing to add, which makes this the cheapest loss in the whole
/// network to remove.
///
/// **Validity: turbulent flow, entry from a large plenum.** About 20%.
pub fn entry_k(entry: EntryCondition) -> Band {
    match entry {
        EntryCondition::None => Band::ZERO,
        EntryCondition::Flush => Band::relative(0.50, 0.20),
        EntryCondition::ReEntrant => Band::relative(0.80, 0.25),
        EntryCondition::BellMouth => Band::relative(0.03, 0.50),
        EntryCondition::Rounded { r_over_dh } => {
            const TABLE: [(f64, f64); 8] = [
                (0.00, 0.50),
                (0.02, 0.28),
                (0.04, 0.24),
                (0.06, 0.15),
                (0.10, 0.09),
                (0.15, 0.06),
                (0.20, 0.03),
                (0.50, 0.03),
            ];
            Band::relative(interpolate(r_over_dh.max(0.0), &TABLE), 0.25)
        }
    }
}

/// Exit loss coefficient, on the duct velocity just inside the mouth.
///
/// Exactly 1.0 for a discharge to still air, from Borda-Carnot with
/// `A2 -> infinity`. Not a fit, so no error bar beyond the assumption that the
/// room really is still.
pub fn exit_k(exit: ExitCondition) -> Band {
    match exit {
        ExitCondition::None => Band::ZERO,
        ExitCondition::Discharge => Band::relative(1.0, 0.05),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CIRC: Section = Section::Circular;

    #[test]
    fn laminar_friction_is_exactly_64_over_re_for_a_pipe() {
        // The analytic case. No excuses, no tolerance beyond float noise.
        for re in [100.0, 500.0, 1000.0, 2000.0, 2300.0] {
            let fr = friction_factor(re, 0.0, CIRC);
            assert_eq!(fr.regime, Regime::Laminar);
            assert!(
                (fr.f - 64.0 / re).abs() < 1e-12,
                "Re = {re}: f = {} against 64/Re = {}",
                fr.f,
                64.0 / re
            );
        }
    }

    #[test]
    fn shah_and_london_hits_both_analytic_endpoints() {
        // Square duct: f*Re = 56.91. Parallel plates: 96. Both are exact
        // solutions of the Navier-Stokes equations, so the fit must reproduce
        // them, and 64 (the circular value) must lie between them.
        assert!(
            (laminar_f_re_rectangular(1.0) - 56.91).abs() < 0.05,
            "square gave {}",
            laminar_f_re_rectangular(1.0)
        );
        assert!(
            (laminar_f_re_rectangular(1.0e6) - 96.0).abs() < 0.01,
            "parallel plates gave {}",
            laminar_f_re_rectangular(1.0e6)
        );
        // A 1000:1 slot is within 0.2% of the limit, which is the practical
        // statement: any duct flat enough to matter is at 96.
        assert!((laminar_f_re_rectangular(1000.0) / 96.0 - 1.0).abs() < 0.002);
        // Monotone in aspect ratio, and straddling the circular 64.
        let mut last = laminar_f_re_rectangular(1.0);
        for aspect in [1.5, 2.0, 3.0, 5.0, 10.0, 50.0] {
            let now = laminar_f_re_rectangular(aspect);
            assert!(now > last, "f*Re fell from {last} to {now} at aspect {aspect}");
            last = now;
        }
        assert!(laminar_f_re_rectangular(1.0) < 64.0);
        assert!(laminar_f_re_rectangular(4.0) > 64.0);
    }

    #[test]
    fn blasius_is_reproduced_across_its_whole_validity_range() {
        for re in [4.0e3, 1.0e4, 3.0e4, 9.9e4] {
            let fr = friction_factor(re, 0.0, CIRC);
            assert_eq!(fr.regime, Regime::Turbulent);
            let want = 0.3164 * re.powf(-0.25);
            assert!(
                (fr.f / want - 1.0).abs() < 1e-12,
                "Re = {re}: f = {} against Blasius {want}",
                fr.f
            );
        }
        // The textbook value everyone remembers.
        assert!((blasius(1.0e4) - 0.031_64).abs() < 1e-5);
    }

    #[test]
    fn colebrook_agrees_with_blasius_on_a_smooth_wall() {
        // Two independent correlations, one of them implicit. Where both are
        // valid they must agree; the accepted spread is a few percent.
        for re in [4.0e3, 1.0e4, 5.0e4, 1.0e5] {
            let (b, c) = (blasius(re), colebrook(re, 0.0));
            assert!(
                (b / c - 1.0).abs() < 0.05,
                "Re = {re}: Blasius {b:.5} against Colebrook {c:.5}"
            );
        }
    }

    #[test]
    fn colebrook_reaches_the_fully_rough_von_karman_limit() {
        // As Re -> inf the Re term vanishes and Colebrook must collapse onto
        // 1/sqrt(f) = -2 log10(eps/3.7D), which is a completely different
        // formula. If the iteration were broken this would not hold.
        for eps in [0.001, 0.005, 0.01, 0.03] {
            let f = colebrook(1.0e9, eps);
            let inv = -2.0 * (eps / 3.7).log10();
            let want = 1.0 / (inv * inv);
            // Not exact: at Re = 1e9 the viscous term is 1e-5 of the roughness
            // term, not zero. That residue is the whole reason Colebrook is
            // used instead of von Karman below Re = 1e7.
            assert!((f / want - 1.0).abs() < 1e-4, "eps = {eps}: {f} against {want}");
        }
    }

    #[test]
    fn haaland_is_within_its_advertised_two_percent_of_colebrook() {
        for re in [5.0e3, 1.0e5, 1.0e7] {
            for eps in [0.0, 0.0005, 0.01] {
                let (h, c) = (haaland(re, eps), colebrook(re, eps));
                assert!(
                    (h / c - 1.0).abs() < 0.02,
                    "Re = {re}, eps = {eps}: Haaland {h:.5} against Colebrook {c:.5}"
                );
            }
        }
    }

    #[test]
    fn printed_roughness_is_not_a_rounding_error() {
        // eps/D_h = 0.05 mm / 6.3 mm = 0.008, which is rougher in relative
        // terms than galvanised steel duct. If this were negligible the crate
        // would not need a roughness knob at all.
        let re = 1.0e4;
        let smooth = friction_factor(re, 0.0, CIRC).f;
        let printed = friction_factor(re, crate::PRINTED_ROUGHNESS_MM / 6.3, CIRC).f;
        let rise = printed / smooth - 1.0;
        assert!(rise > 0.15, "printed roughness only raised f by {:.1}%", rise * 100.0);
        assert!(rise < 0.60, "printed roughness raised f by {:.1}%, implausible", rise * 100.0);
    }

    #[test]
    fn the_transition_is_continuous_at_both_ends_and_bracketed_between_them() {
        let lam = friction_factor(RE_LAMINAR_MAX, 0.0, CIRC);
        assert!((lam.f - 64.0 / RE_LAMINAR_MAX).abs() < 1e-12);
        let turb = friction_factor(RE_TURBULENT_MIN, 0.0, CIRC);
        assert!((turb.f - blasius(RE_TURBULENT_MIN)).abs() < 1e-12);

        // Approaching each end from inside the blend must converge to the same
        // value: no step.
        let just_above = friction_factor(RE_LAMINAR_MAX + 1e-6, 0.0, CIRC);
        assert!((just_above.f - lam.f).abs() < 1e-6, "step of {}", just_above.f - lam.f);
        let just_below = friction_factor(RE_TURBULENT_MIN - 1e-6, 0.0, CIRC);
        assert!((just_below.f - turb.f).abs() < 1e-6, "step of {}", just_below.f - turb.f);

        // Through the transition the blend must stay bracketed by the two
        // branches it is blending, and must end well above where it started.
        // It is NOT monotone: just past 2300 the blend weight is still tiny
        // while the laminar branch is still falling as 1/Re, so f dips by a few
        // percent before climbing. That dip is real behaviour of the blend and
        // is documented on `friction_factor`; it bottoms out near Re = 2600 at
        // about 6%, an order of magnitude inside the 50% band the transition
        // carries, and pretending otherwise would need a kink at 2300.
        let mut dip = 0.0f64;
        for i in 1..=40 {
            let re = RE_LAMINAR_MAX + (RE_TURBULENT_MIN - RE_LAMINAR_MAX) * i as f64 / 40.0;
            let f = friction_factor(re, 0.0, CIRC);
            assert_eq!(
                f.regime,
                if re >= RE_TURBULENT_MIN { Regime::Turbulent } else { Regime::Transitional }
            );
            let (lo, hi) = {
                let (a, b) = (64.0 / re, blasius(re));
                (a.min(b), a.max(b))
            };
            assert!(f.f >= lo - 1e-12 && f.f <= hi + 1e-12, "f = {} outside [{lo}, {hi}] at Re = {re}", f.f);
            dip = dip.max(1.0 - f.f / lam.f);
        }
        assert!(dip > 0.02, "the dip is real; if it vanished the blend changed");
        assert!(dip < 0.08, "the blend dips {:.1}% below the laminar endpoint", dip * 100.0);
        assert!(turb.f > lam.f * 1.4, "the transition should raise f substantially");
        // ...and it says loudly that it does not know.
        assert!(friction_factor(3000.0, 0.0, CIRC).sigma_rel >= 0.5);
    }

    #[test]
    fn a_smooth_ninety_degree_elbow_at_r_over_d_1_5_lands_in_the_handbook_band() {
        // The canonical check: everyone's table says 0.2-0.3.
        let f = friction_factor(2.0e4, 0.0, CIRC).f;
        let k = elbow_k(90.0, 1.5, 1.0, f);
        assert!(
            k.mean >= 0.20 && k.mean <= 0.30,
            "K = {k} for a smooth 90 deg elbow at r/D = 1.5, expected 0.2-0.3"
        );
        // The two terms should be of the same order; if the friction part
        // dominated, the radius factor would not be doing any work.
        let local = bend_local_k(90.0, 1.5, 1.0);
        assert!(local.mean > 0.10 && local.mean < 0.25, "K_loc = {local}");
    }

    #[test]
    fn a_mitred_rectangular_elbow_matches_the_handbook_range() {
        // ASHRAE's rectangular mitred elbow without vanes runs C = 1.0-1.5
        // across H/W; Idelchik Diagram 6-7 gives 1.2-1.4 at square aspect;
        // Crane TP-410's 60 f_T is ~1.2. This must reproduce that consensus.
        let f = friction_factor(2.0e4, 0.0, CIRC).f;
        for (hw, lo, hi) in [(0.25, 1.4, 1.8), (1.0, 1.1, 1.5), (2.0, 0.9, 1.3), (8.0, 1.0, 1.5)]
        {
            let k = elbow_k(90.0, 0.0, hw, f);
            assert!(
                k.mean >= lo && k.mean <= hi,
                "mitre at H/W = {hw}: K = {k}, expected {lo}-{hi}"
            );
        }
        // A mitre must be several times worse than a well-radiused elbow, or
        // the whole point of radiusing a bend is lost.
        let mitre = elbow_k(90.0, 0.0, 1.0, f).mean;
        let radiused = elbow_k(90.0, 1.5, 1.0, f).mean;
        assert!(mitre / radiused > 4.0, "mitre {mitre} vs radiused {radiused}");
    }

    #[test]
    fn the_contracts_quoted_mitre_band_needs_an_unfavourable_aspect_ratio() {
        // CONTRACT.md quotes K = 2.0-3.5 for "a rectangular mitred 90 degree
        // bend with no turning vanes (ASHRAE)". No handbook this crate cites
        // reaches 2.0 for the bare fitting at ordinary aspect ratios, and this
        // test pins down exactly how far you have to push the geometry to get
        // there. It is documentation of a disagreement, not a target.
        let f = friction_factor(2.0e4, 0.0, CIRC).f;
        let square = elbow_k(90.0, 0.0, 1.0, f);
        assert!(square.high() < 2.0, "square mitre band {square} already reaches the contract");

        // Turning the hard way in a very flat duct does approach it.
        let flat = elbow_k(90.0, 0.0, 0.05, f);
        assert!(flat.mean > 1.7, "H/W = 0.05 mitre gave only K = {flat}");
        assert!(flat.high() > 2.0, "even a 20:1 flat mitre band {flat} misses 2.0");
        // ...and it says it is extrapolating while it does so.
        assert!(flat.relative_sigma() >= 0.40 - 1e-9);
    }

    #[test]
    fn bend_loss_falls_monotonically_with_radius_and_rises_with_angle() {
        let mut last = f64::INFINITY;
        for r in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0] {
            let k = bend_local_k(90.0, r, 1.0).mean;
            assert!(k <= last + 1e-9, "K rose from {last} to {k} going to r/D = {r}");
            last = k;
        }
        let mut last = 0.0;
        for a in [15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0] {
            let k = bend_local_k(a, 1.0, 1.0).mean;
            assert!(k >= last - 1e-9, "K fell from {last} to {k} at {a} deg");
            last = k;
        }
        // The two Idelchik radius branches must meet at r/D = 1.
        let below = bend_local_k(90.0, 1.0 - 1e-9, 1.0).mean;
        let above = bend_local_k(90.0, 1.0 + 1e-9, 1.0).mean;
        assert!((below - above).abs() < 1e-6);
        assert!((above - 0.21).abs() < 1e-6, "B1(1) should be 0.21, got {above}");
    }

    #[test]
    fn turning_a_duct_the_easy_way_is_measurably_cheaper() {
        // The actionable output: rotating the section 90 degrees about the flow
        // axis is free and changes the bend loss by ~35%.
        let hard = bend_local_k(90.0, 1.0, 0.25).mean;
        let easy = bend_local_k(90.0, 1.0, 3.0).mean;
        assert!(hard / easy > 1.4, "hard {hard} vs easy {easy}");
    }

    #[test]
    fn sudden_expansion_is_exactly_borda_carnot() {
        // This one is a momentum balance, not a fit, so "exactly" means
        // exactly.
        for (a1, a2) in [(1.0f64, 1.0f64), (1.0, 2.0), (1.0, 4.0), (1.0, 100.0), (3.0, 4.0)] {
            let want = (1.0 - a1 / a2).powi(2);
            let got = sudden_expansion_k(a1 / a2);
            assert!((got.mean - want).abs() < 1e-15, "{got} against {want}");
        }
        // Discharging into a room is the A2 -> inf limit and costs exactly one
        // velocity head.
        assert!((sudden_expansion_k(0.0).mean - 1.0).abs() < 1e-15);
        assert!((sudden_expansion_k(0.0).mean - exit_k(ExitCondition::Discharge).mean).abs() < 1e-15);
        assert!(sudden_expansion_k(1.0).mean.abs() < 1e-15);
    }

    #[test]
    fn sudden_contraction_matches_the_weisbach_form_and_the_entry_limit() {
        for r in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!((sudden_contraction_k(r).mean - 0.5 * (1.0 - r)).abs() < 1e-15);
        }
        // Contracting from a plenum is the same event as a sharp-edged entry,
        // so the two correlations must agree at the limit.
        assert!(
            (sudden_contraction_k(0.0).mean - entry_k(EntryCondition::Flush).mean).abs() < 1e-12
        );
    }

    #[test]
    fn gradual_area_changes_degenerate_to_the_sudden_ones_at_180_degrees() {
        let (a_c, a_e) = (0.4, 0.4);
        assert!(
            (gradual_contraction_k(a_c, 180.0).mean - sudden_contraction_k(a_c).mean).abs() < 1e-12
        );
        assert!(
            (gradual_expansion_k(a_e, 180.0).mean - sudden_expansion_k(a_e).mean).abs() < 1e-12
        );
        // ...and a gentle taper is much cheaper than a step.
        assert!(gradual_contraction_k(a_c, 20.0).mean < 0.25 * sudden_contraction_k(a_c).mean);
        assert!(gradual_expansion_k(a_e, 10.0).mean < 0.30 * sudden_expansion_k(a_e).mean);
        // The diffuser branches must meet at 45 degrees, where 2.6 sin 22.5 = 1.
        let below = gradual_expansion_k(a_e, 45.0 - 1e-9).mean;
        let above = gradual_expansion_k(a_e, 45.0 + 1e-9).mean;
        assert!((below / above - 1.0).abs() < 0.01, "{below} against {above}");
    }

    #[test]
    fn expansion_beats_contraction_only_below_a_two_to_one_area_change() {
        // `(1-r)^2` against `0.5(1-r)` cross at exactly `r = 0.5`. Below that a
        // step-out costs more than the step-in that undoes it, above it the
        // other way round -- worth knowing when choosing which end of a
        // transition to put an area change at, and a sharp check that neither
        // formula has been transcribed with the wrong exponent.
        for ratio in [0.05, 0.1, 0.25, 0.49] {
            assert!(
                sudden_expansion_k(ratio).mean > sudden_contraction_k(ratio).mean,
                "at A_small/A_large = {ratio} the expansion should dominate"
            );
        }
        for ratio in [0.51, 0.75, 0.9] {
            assert!(sudden_expansion_k(ratio).mean < sudden_contraction_k(ratio).mean);
        }
        let cross = (sudden_expansion_k(0.5).mean - sudden_contraction_k(0.5).mean).abs();
        assert!(cross < 1e-15, "the curves should cross exactly at 0.5, gap {cross}");
    }

    #[test]
    fn a_little_entry_radius_removes_most_of_the_entry_loss() {
        let sharp = entry_k(EntryCondition::Flush).mean;
        let rounded = entry_k(EntryCondition::Rounded { r_over_dh: 0.10 }).mean;
        assert!(rounded < 0.25 * sharp, "sharp {sharp}, r/D = 0.1 gives {rounded}");
        assert!((entry_k(EntryCondition::Rounded { r_over_dh: 0.0 }).mean - sharp).abs() < 1e-15);
        assert!(entry_k(EntryCondition::ReEntrant).mean > sharp);
        assert!(entry_k(EntryCondition::BellMouth).mean < 0.05);
        assert_eq!(entry_k(EntryCondition::None), Band::ZERO);
        assert_eq!(exit_k(ExitCondition::None), Band::ZERO);
    }

    #[test]
    fn every_coefficient_is_finite_for_every_degenerate_input() {
        // No unwrap, no NaN, no infinity, whatever the geometry extractor hands
        // over — including the zeros and the absurd values a broken passage
        // produces.
        let inputs = [0.0f64, -1.0, 1e-12, 1e12, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for &x in &inputs {
            assert!(bend_local_k(90.0, x, 1.0).mean.is_finite(), "bend r/D = {x}");
            assert!(bend_local_k(x, 1.0, 1.0).mean.is_finite(), "bend angle = {x}");
            assert!(bend_local_k(90.0, 1.0, x).mean.is_finite(), "bend H/W = {x}");
            assert!(elbow_k(90.0, 1.0, 1.0, x).mean.is_finite(), "elbow f = {x}");
            assert!(sudden_contraction_k(x).mean.is_finite());
            assert!(sudden_expansion_k(x).mean.is_finite());
            assert!(gradual_contraction_k(0.5, x).mean.is_finite());
            assert!(gradual_expansion_k(0.5, x).mean.is_finite());
            assert!(entry_k(EntryCondition::Rounded { r_over_dh: x }).mean.is_finite());
            for section in [CIRC, Section::Rectangular { aspect: x }] {
                let fr = friction_factor(x, 0.0, section);
                assert!(fr.f.is_finite(), "f = {} at Re = {x}", fr.f);
                assert!(fr.f >= 0.0);
            }
            assert!(colebrook(1.0e5, x).is_finite(), "colebrook eps = {x}");
        }
    }
}

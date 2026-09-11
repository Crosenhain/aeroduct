//! Where the simulation domain ends.
//!
//! # The three answers
//!
//! There are three domains in the tree, and [`DomainMargins`] selects between
//! them. In order of how much room they leave around the part:
//!
//! | | what it is | cells | ms/step | default? |
//! |---|---|---|---|---|
//! | **room** | one isotropic margin, equilibrium on all six faces | 21.57 M | 3.633 | **yes** |
//! | trimmed | the same room, shrunk per-face from the mouths' normals | 15.2 M | ~2.6 | no |
//! | plenum | no room at all: bounding box plus a walled extension on each mouth ([`crate::plenum`]) | 6.55 M | 0.152 | no |
//!
//! The room is 24x the cost per step of the plenum and is still the default,
//! which is the sort of thing that needs its reasons written down rather than
//! inferred.
//!
//! The rest of this module is the room and the trim. The plenum lives next
//! door, and [`DomainMargins::default`] is the decision record for why the
//! expensive one is still the one that ships.
//!
//! # The measurement that motivated this
//!
//! The domain used to be one isotropic margin — 40% of the scene's longest side
//! on all six faces — which around the test part is a 261 x 188 x 185 mm box.
//! Of that 9.08e6 mm^3:
//!
//! * the part's own bounding box is 7.9%
//! * **the duct passage is 1.41%**
//!
//! So 98.6% of every LBM step was spent relaxing room air toward the state it
//! was already in. At `dx = 0.75 mm` that is 21.6 M cells of which only about
//! 300 k are inside the duct.
//!
//! # The rule
//!
//! Air does not care about the bounding box; it cares about where it is going.
//! So the six margins are derived from the two detected mouths, each from the
//! direction its own normal points:
//!
//! | face | margin | why |
//! |---|---|---|
//! | the one the outlet mouth exhausts through | [`DomainMargins::downstream_d_h`] x `D_h` | the exit jet, the sponge, and anything the user parks in the flow |
//! | the one behind the inlet mouth | [`DomainMargins::upstream_d_h`] x `D_h` | the inlet plenum CONTRACT.md asks for |
//! | the other four | [`DomainMargins::lateral_frac`] x the part's extent on that axis, floored at [`DomainMargins::lateral_d_h`] x `D_h` | entrainment, and staying outside the mouths' own pressure field |
//!
//! `Mouth::patch.normal` points **into** the fluid, and every detected mouth is
//! flush with a bounding-box face, so a mouth whose normal is `+axis` sits on
//! that axis' *min* face and its outside is the min side. That one fact is what
//! makes both the upstream and the downstream face fall out of the normal alone,
//! with no per-axis special cases and nothing to keep in step when the user
//! swaps which mouth is the inlet.
//!
//! # What the clamps are for
//!
//! `D_h` of a slot is roughly twice its width, so a 15 mm slot yields `D_h`
//! around 27 mm and four of them is 110 mm — comparable to the part itself. The
//! absolute clamps stop a very large or very small mouth from producing a domain
//! that is mostly empty room or one that has the outlet plane sitting in the
//! recirculation behind the part.

use ad_geom::Mouth;
use ad_gpu::Bbox;
use glam::Vec3;

use crate::plenum::{Plenum, PlenumWalls};

/// The isotropic margin this replaced: 40% of the scene's longest side on every
/// face, which is CONTRACT.md's 260 x 180 x 180 mm around the test part.
///
/// Kept as the value [`DomainMargins::isotropic_frac`] reproduces, so an A/B
/// run against the old behaviour is an environment variable rather than a
/// rebuild.
pub const LEGACY_ISOTROPIC_MARGIN: f32 = 0.4;

/// Downstream margin, in outlet hydraulic diameters.
///
/// Four rather than six. A plane jet's potential core is about five slot widths
/// and `D_h` of a high-aspect slot is about two of them, so 4 `D_h` is roughly
/// eight slot widths: past the core, into the self-similar region, and far
/// enough that the sponge damps a jet that has already spread rather than one
/// still leaving the mouth. Six would be more comfortable and costs another
/// 20% of the cells for flow the user is no longer looking at.
const DEFAULT_DOWNSTREAM_D_H: f32 = 4.0;

/// Upstream margin, in inlet hydraulic diameters.
///
/// CONTRACT.md asks for ~2 `D_h` of plenum in front of the inlet so a boundary
/// layer develops instead of plug flow being injected; the extra quarter is
/// slack so the equilibrium face is not sitting exactly on the last streamline
/// that turns into the mouth.
const DEFAULT_UPSTREAM_D_H: f32 = 2.25;

/// Lateral margin, as a fraction of the part's extent on that axis.
///
/// Per-axis rather than one number off the longest side: the room beside a
/// 145 mm part and the room behind a 69 mm one are different questions, and
/// tying both to 145 is how the old box grew to five times the part's volume.
const DEFAULT_LATERAL_FRAC: f32 = 0.15;

/// Floor on every lateral margin, in the larger mouth's hydraulic diameters.
///
/// # Why a fraction of the part is not enough on its own
///
/// A mouth is a sink: the inlet draws air toward its plane from every direction
/// that is not blocked by the part, and the pressure perturbation that does the
/// drawing decays over a length set by the *opening*, not by the part. An
/// equilibrium face pins `p` at the reference value, so putting one inside that
/// perturbation does not merely truncate the domain, it holds the near field at
/// the wrong pressure and the inlet has to work against it.
///
/// Measured, at `dx = 0.75 mm` and 3 m/s, 150 k steps: with the lateral margins
/// on the part's extent alone the tight faces landed 10-11 mm from a mouth of
/// `D_h = 27 mm`, and the inlet total pressure read **114 Pa against the
/// isotropic domain's 101** — a 14% error in `dp_total` while `Q` was unmoved at
/// 0.7%. `Q` cannot see it because the inlet is a velocity boundary: it delivers
/// the flow it was asked for whatever the pressure costs, so a confined inlet is
/// invisible in the flow rate and shows up entirely in the pressure.
///
/// One `D_h` puts the face outside that perturbation. It is the margin the
/// anisotropy has to spend rather than save.
const DEFAULT_LATERAL_D_H: f32 = 1.0;

/// Absolute clamps on the downstream margin, mm.
const DOWNSTREAM_RANGE_MM: (f32, f32) = (25.0, 150.0);
/// Absolute clamps on the upstream margin, mm.
const UPSTREAM_RANGE_MM: (f32, f32) = (10.0, 80.0);
/// Absolute clamps on a lateral margin, mm.
///
/// The ceiling is above `DEFAULT_LATERAL_D_H` times the test part's 27 mm mouth
/// on purpose: a clamp that bit at the default would make the mouth-scaled floor
/// decorative on the only geometry it has been measured against.
const LATERAL_RANGE_MM: (f32, f32) = (8.0, 50.0);

/// Floor on a lateral margin in cells, so a coarse `dx` still leaves the
/// equilibrium face a few cells clear of the geometry.
const LATERAL_FLOOR_CELLS: u32 = 6;

/// Cells of undisturbed fluid demanded between the geometry side of the sponge
/// and the jet's near field.
const SPONGE_CLEARANCE_CELLS: u32 = 2;

/// Extent of the exit jet's near field, in outlet hydraulic diameters.
///
/// The sponge and the outlet plane have to start beyond this. An outlet plane
/// inside the recirculation just past the mouth reads the pressure of an eddy
/// rather than of the room, which is a wrong `dp` rather than a noisy one — so
/// when the requested margin cannot hold both, the margin grows.
const JET_NEAR_FIELD_D_H: f32 = 1.0;

/// Smallest hydraulic diameter treated as real, mm. Below this the mouth's
/// boundary loop is degenerate and the absolute clamps carry the margin.
const MIN_D_H_MM: f32 = 1.0;

/// How far past the scene the domain reaches, per direction.
///
/// Every field is a *shape* rather than a distance, so one set of numbers works
/// for any part: they are resolved against the detected mouths at the moment
/// the domain is built, which is also the moment the user's choice of inlet is
/// known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomainMargins {
    /// Downstream of the outlet mouth, in its hydraulic diameters.
    pub downstream_d_h: f32,
    /// Upstream of the inlet mouth, in its hydraulic diameters.
    pub upstream_d_h: f32,
    /// Everywhere else, as a fraction of the scene's extent on that axis.
    pub lateral_frac: f32,
    /// Floor under every lateral margin, in the larger mouth's hydraulic
    /// diameters. See [`DEFAULT_LATERAL_D_H`] for the measurement that put it
    /// there.
    pub lateral_d_h: f32,
    /// Escape hatch: one isotropic margin as a fraction of the longest side,
    /// exactly as the domain was built before this module existed. `Some(0.4)`
    /// reproduces the old box, which is what makes a before/after comparison a
    /// pair of runs of the same binary.
    pub isotropic_frac: Option<f32>,
    /// When set, the domain is not a room at all: it is the part's bounding box
    /// with a walled straight extension on each mouth, and the boundary
    /// conditions live at the ends of those. See [`crate::plenum`].
    ///
    /// Takes precedence over both fields above, because they describe how much
    /// room to leave and this one says there is no room.
    pub plenum: Option<Plenum>,
    /// The six margins set by hand, mm, as `[-x, +x, -y, +y, -z, +z]`. Wins
    /// over every rule above: a user who has typed a distance wants that
    /// distance, and the domain control is the one place where a bigger box
    /// is asked for on purpose — room for an exit jet to develop, or for a
    /// vent standing off the part.
    pub explicit_mm: Option<[f32; 6]>,
}

impl Default for DomainMargins {
    /// The isotropic room, still, and this is the third time that has been a
    /// deliberate choice rather than an oversight.
    ///
    /// # The three domains, measured
    ///
    /// `dx = 0.75 mm`, 3 m/s on mouth A, run one at a time on an otherwise idle
    /// RTX 4090. At **70,000 steps**, which is the step count every previous
    /// comparison in this file used:
    ///
    /// | | **isotropic room** | anisotropic trim | plenum ([`crate::plenum`]) |
    /// |---|---|---|---|
    /// | cells | 21.57 M | 15.2 M | **6.55 M** |
    /// | ms/step | 3.634 | ~2.6 | **0.153** |
    /// | `Q_in` | 6.131 +/- 0.008 L/s | 6.13 +/- 0.02 | 6.34 +/- 0.03 |
    /// | `Q_out` | 6.41 +/- 0.05 | 7.06 +/- 0.05 | 7.20 +/- 0.09 |
    /// | mass imbalance | **5.63%** | 9.10% | **6.83%** |
    /// | `dp_total` | 65 +/- 2 Pa | 72 +/- 5 | 74 +/- 12 |
    ///
    /// **The middle column is why the right column is not just "trim harder".**
    /// Shrinking the room while keeping equilibrium far-field sides bought 1.42x
    /// and cost mass conservation, because an equilibrium plane is a pressure
    /// boundary and moving one to within ~10 mm of a 27 mm mouth is a modelling
    /// error rather than a truncation. `Q_in` did not move, which proves
    /// nothing: a velocity inlet delivers the flow it was asked for whatever the
    /// pressure costs, so the damage lands entirely on `Q_out` and on the
    /// balance between them.
    ///
    /// # Why the plenum is not the default either
    ///
    /// Because of the two bold numbers in the last row but one, and because the
    /// same comparison run five times longer does not rescue them. At
    /// **350,000 steps**, both domains, same machine, same conditions:
    ///
    /// | | **isotropic room** | plenum |
    /// |---|---|---|
    /// | ms/step | 3.633 | **0.152** |
    /// | `Q_in` | 6.17 +/- 0.01 L/s | 6.354 +/- 0.006 |
    /// | `Q_out` | 6.66 +/- 0.01 | 6.85 +/- 0.09 |
    /// | mass imbalance | **3.02%** | **3.56%** |
    /// | `dp_total` | 60 +/- 2 Pa | 57 +/- 2 Pa |
    /// | outlet backflow | 0.09 | 0.04 |
    ///
    /// The plenum domain does not beat the room on mass conservation at either
    /// step count. It is close — 3.56% against 3.02%, where the difference is
    /// entirely fluctuation, since both satisfy continuity in the *mean* to
    /// about 1.2% once the measured expansion is accounted for — but "close" is
    /// not the gate. The gate was that it must not get worse, and it does.
    ///
    /// Everything else about the plenum domain is better: 23.8x faster per step,
    /// backflow at the outlet plane down from 15% to 0.9%, the inlet development
    /// length CONTRACT.md asks for actually present, and no equilibrium cells at
    /// all, so mass has exactly two doors instead of 381,270 windows. It ships
    /// tested and one environment variable away rather than as the default.
    ///
    /// # Two findings from the attempt that outlive it
    ///
    /// **1. Neither domain is converged at 70,000 steps, and the 65 Pa this
    /// file has been quoting is not a converged number.** The room's own
    /// `dp_total` moves from 65 +/- 2 to 60 +/- 2 Pa between 70 k and 350 k
    /// steps, and its mass imbalance from 5.63% to 3.02%, on an unchanged
    /// domain. Any future comparison anchored to "65 +/- 2 Pa at 70,000 steps"
    /// is anchored to a transient. The plenum sweep makes the same point from
    /// the other side: at 70 k, outlet plenums of 2, 3, 4, 5, 6 and 8 `D_h` give
    /// mass imbalances of 7.18, 6.83, 8.83, 4.31, 8.27 and 6.76% and `dp_total`
    /// from 65 to 84 Pa. The solver is deterministic, so that is not run-to-run
    /// noise; it is six samples of a state that has not settled.
    ///
    /// **2. The two domains agree on the answer.** At 350 k, `dp_total` is 57
    /// against 60 Pa — 5% apart, about two error bars — from geometries that
    /// have almost nothing in common downstream of the duct. That is the
    /// strongest evidence available that the duct's own pressure drop is being
    /// measured rather than the box around it, and it took 53 seconds of solver
    /// time on one side against 1,272 on the other.
    ///
    /// What would change the default: a mass imbalance at equal steps that is
    /// *better* rather than nearly the same. The plenum's is fluctuation-limited
    /// at the outlet plane (`Q_out` error bar +/- 0.09 against the room's
    /// +/- 0.01), which points at the confined separated flow in the outlet
    /// extension rather than at the termination — so the thing to try next is a
    /// wider outlet plenum, a chamber rather than a tube, not a deeper one.
    ///
    /// All three domains are one environment variable apart (`AERODUCT_DOMAIN`),
    /// because a before/after on this should stay a pair of runs of one binary.
    fn default() -> Self {
        Self {
            downstream_d_h: DEFAULT_DOWNSTREAM_D_H,
            upstream_d_h: DEFAULT_UPSTREAM_D_H,
            lateral_frac: DEFAULT_LATERAL_FRAC,
            lateral_d_h: DEFAULT_LATERAL_D_H,
            isotropic_frac: Some(LEGACY_ISOTROPIC_MARGIN),
            plenum: None,
            explicit_mm: None,
        }
    }
}

impl DomainMargins {
    /// With the six margins fixed by hand, or the rules left alone.
    pub fn with_explicit(self, mm: Option<[f32; 6]>) -> Self {
        Self {
            explicit_mm: mm.filter(|m| m.iter().all(|v| v.is_finite())),
            ..self
        }
    }

    /// Read the `AERODUCT_MARGIN_*` overrides.
    ///
    /// Same convention and same discipline as [`crate::startup_params`]: every
    /// value is validated, and a typo falls back to the default rather than
    /// producing a domain that is zero cells thick or larger than VRAM.
    pub fn from_env() -> Self {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// [`Self::from_env`] against an arbitrary lookup, so the parsing is
    /// testable without mutating the process environment — which is shared by
    /// every test in this binary and cannot be done safely in parallel.
    fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let num = |name: &str| -> Option<f32> {
            get(name)?.parse::<f32>().ok().filter(|v| v.is_finite())
        };
        let mut m = Self::default();

        // --- the plenum knobs -------------------------------------------
        //
        // Same rule as the anisotropic ones below: asking for a plenum depth is
        // asking for the domain that has plenums in it. That matters more here
        // than it looks, because a plenum knob set while the domain was a room
        // would parse cleanly, store faithfully, and change nothing.
        let mut plenum = m.plenum.unwrap_or_default();
        let mut plenum_requested = m.plenum.is_some();
        if let Some(v) = num("AERODUCT_PLENUM_UPSTREAM_DH").filter(|v| (0.0..=40.0).contains(v)) {
            plenum.upstream_d_h = v;
            plenum_requested = true;
        }
        if let Some(v) = num("AERODUCT_PLENUM_DOWNSTREAM_DH").filter(|v| (0.0..=40.0).contains(v)) {
            plenum.downstream_d_h = v;
            plenum_requested = true;
        }
        if let Some(v) = num("AERODUCT_PLENUM_LATERAL_CELLS").filter(|v| (0.0..=64.0).contains(v)) {
            plenum.lateral_cells = v as u32;
            plenum_requested = true;
        }
        match get("AERODUCT_PLENUM_WALLS").as_deref() {
            Some("solid") | Some("walled") | Some("duct") => {
                plenum.walls = PlenumWalls::Solid;
                plenum_requested = true;
            }
            Some("open") | Some("equilibrium") | Some("chamber") => {
                plenum.walls = PlenumWalls::Open;
                plenum_requested = true;
            }
            Some(v) => log::warn!(
                "AERODUCT_PLENUM_WALLS={v:?} is neither \"solid\" nor \"open\"; keeping the default"
            ),
            None => {}
        }
        m.plenum = plenum_requested.then_some(plenum);

        // Setting any anisotropic knob opts into the anisotropic domain.
        //
        // The default is the isotropic box, which ignores all four of these. So
        // without this, asking for `AERODUCT_MARGIN_DOWNSTREAM_DH=6` would parse
        // cleanly, be stored faithfully, and then change nothing at all -- the
        // worst kind of knob. Requesting a margin is a request for the mode that
        // uses it. An explicit `AERODUCT_MARGIN_ISOTROPIC` still wins, because
        // it is set below and can put the isotropic box back.
        let mut anisotropic_requested = false;
        if let Some(v) = num("AERODUCT_MARGIN_DOWNSTREAM_DH").filter(|v| (0.0..=40.0).contains(v)) {
            m.downstream_d_h = v;
            anisotropic_requested = true;
        }
        if let Some(v) = num("AERODUCT_MARGIN_UPSTREAM_DH").filter(|v| (0.0..=40.0).contains(v)) {
            m.upstream_d_h = v;
            anisotropic_requested = true;
        }
        if let Some(v) = num("AERODUCT_MARGIN_LATERAL").filter(|v| (0.0..=4.0).contains(v)) {
            m.lateral_frac = v;
            anisotropic_requested = true;
        }
        if let Some(v) = num("AERODUCT_MARGIN_LATERAL_DH").filter(|v| (0.0..=40.0).contains(v)) {
            m.lateral_d_h = v;
            anisotropic_requested = true;
        }
        if anisotropic_requested {
            m.isotropic_frac = None;
            m.plenum = None;
        }
        // A fraction, or the word `legacy` for the margin this replaced — which
        // is the spelling an A/B run against the old domain actually wants, and
        // spares whoever does it from having to remember the number.
        match get("AERODUCT_MARGIN_ISOTROPIC").as_deref() {
            Some("legacy") | Some("old") => {
                m.isotropic_frac = Some(LEGACY_ISOTROPIC_MARGIN);
                m.plenum = None;
            }
            Some(v) => {
                match v
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite() && (0.0..=4.0).contains(v))
                {
                    Some(v) => {
                        m.isotropic_frac = Some(v);
                        m.plenum = None;
                    }
                    None => log::warn!(
                        "AERODUCT_MARGIN_ISOTROPIC={v:?} is neither a fraction nor \"legacy\"; \
                         keeping the default domain"
                    ),
                }
            }
            None => {}
        }

        // The one that names a domain outright, evaluated last so it wins.
        //
        // Every knob above selects a mode as a side effect of asking for one of
        // its parameters, which is right when you are tuning and wrong when you
        // are running an A/B: "give me the old domain" should not require
        // knowing which parameter happens to identify it.
        match get("AERODUCT_DOMAIN").as_deref() {
            Some("plenum") | Some("duct") => m.plenum = Some(m.plenum.unwrap_or_default()),
            Some("room") | Some("legacy") | Some("isotropic") => {
                m.plenum = None;
                m.isotropic_frac = Some(m.isotropic_frac.unwrap_or(LEGACY_ISOTROPIC_MARGIN));
            }
            Some("trimmed") | Some("anisotropic") => {
                m.plenum = None;
                m.isotropic_frac = None;
            }
            Some(v) => log::warn!(
                "AERODUCT_DOMAIN={v:?} is not one of \"plenum\", \"room\" or \"trimmed\"; \
                 keeping the default domain"
            ),
            None => {}
        }
        m
    }

    /// Resolve the margins against a scene and its mouths.
    ///
    /// `scene` is the **whole** scene box, obstructions included, so anything
    /// the user drops in the flow is inside the domain by construction and the
    /// downstream margin is measured past it rather than through it.
    pub fn plan(
        &self,
        scene: Bbox,
        mouths: &[Mouth],
        inlet: usize,
        outlet: usize,
        dx_mm: f32,
        sponge_cells: u32,
    ) -> Domain {
        if scene.is_empty() {
            return Domain {
                bbox: scene,
                lo_mm: Vec3::ZERO,
                hi_mm: Vec3::ZERO,
                plenum: None,
            };
        }
        if let Some(m) = self.explicit_mm {
            let lo = Vec3::new(m[0], m[2], m[4]).max(Vec3::ZERO);
            let hi = Vec3::new(m[1], m[3], m[5]).max(Vec3::ZERO);
            return Domain {
                bbox: Bbox {
                    min: scene.min - lo,
                    max: scene.max + hi,
                },
                lo_mm: lo,
                hi_mm: hi,
                plenum: None,
            };
        }
        // The plenum domain first, and it may decline: a mouth with no
        // axis-aligned face has no direction to extrude a plenum along, and a
        // box built anyway would put the boundary conditions somewhere the flow
        // does not go. Falling back to a room there is the conservative failure
        // — expensive, but the expense is the thing that was always safe.
        if let Some(p) = self.plenum {
            if let Some(d) = p.plan(scene, mouths, inlet, outlet, dx_mm, sponge_cells) {
                return d;
            }
            log::warn!(
                "no axis-aligned face to build a plenum on; falling back to the isotropic room"
            );
            let m = Vec3::splat(scene.size().max_element() * LEGACY_ISOTROPIC_MARGIN);
            return Domain {
                bbox: scene.expanded(m),
                lo_mm: m,
                hi_mm: m,
                plenum: None,
            };
        }
        if let Some(frac) = self.isotropic_frac {
            let m = Vec3::splat(scene.size().max_element() * frac.max(0.0));
            return Domain {
                bbox: scene.expanded(m),
                lo_mm: m,
                hi_mm: m,
                plenum: None,
            };
        }

        let dx = if dx_mm.is_finite() && dx_mm > 0.0 {
            dx_mm
        } else {
            1.0
        };
        let size = scene.size();
        // The mouth the lateral faces have to stay clear of is the larger of the
        // two, whichever way round the user is blowing: swapping the inlet must
        // not move a box face, or the same duct would report two pressures.
        let d_h_max = [inlet, outlet]
            .into_iter()
            .filter_map(|i| axis_face(mouths.get(i)))
            .map(|(_, d_h)| d_h)
            .fold(0.0f32, f32::max);
        let mouth_floor =
            (self.lateral_d_h * d_h_max).clamp(LATERAL_RANGE_MM.0, LATERAL_RANGE_MM.1);
        let floor = LATERAL_RANGE_MM
            .0
            .max(LATERAL_FLOOR_CELLS as f32 * dx)
            .max(mouth_floor);
        // The ceiling yields to the floor rather than the other way round: a
        // very coarse `dx` can push the six-cell floor past the ceiling, and
        // `f32::clamp` panics outright when handed an inverted range.
        let ceiling = LATERAL_RANGE_MM.1.max(floor);
        let lateral = |extent: f32| (self.lateral_frac * extent).clamp(floor, ceiling);

        let mut lo = Vec3::new(lateral(size.x), lateral(size.y), lateral(size.z));
        let mut hi = lo;

        // Downstream first: it is the larger of the two, so if both mouths
        // somehow exhaust through the same face the max below keeps it.
        if let Some((face, d_h)) = axis_face(mouths.get(outlet)) {
            let sponge = (sponge_cells + SPONGE_CLEARANCE_CELLS) as f32 * dx;
            let want = (self.downstream_d_h * d_h)
                .clamp(DOWNSTREAM_RANGE_MM.0, DOWNSTREAM_RANGE_MM.1)
                // Grow rather than overlap: the sponge plus the jet's near
                // field is a hard floor, whatever the multiplier asked for.
                .max(JET_NEAR_FIELD_D_H * d_h + sponge);
            face.widen(&mut lo, &mut hi, want);
        }
        if let Some((face, d_h)) = axis_face(mouths.get(inlet)) {
            let want = (self.upstream_d_h * d_h).clamp(UPSTREAM_RANGE_MM.0, UPSTREAM_RANGE_MM.1);
            face.widen(&mut lo, &mut hi, want);
        }

        Domain {
            bbox: Bbox {
                min: scene.min - lo,
                max: scene.max + hi,
            },
            lo_mm: lo,
            hi_mm: hi,
            plenum: None,
        }
    }
}

/// A resolved domain: the box, and the six margins that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Domain {
    pub bbox: Bbox,
    /// Margin added on the min side of each axis, mm.
    pub lo_mm: Vec3,
    /// Margin added on the max side of each axis, mm.
    pub hi_mm: Vec3,
    /// `Some` when the two generous margins are *plenums* rather than room, and
    /// the mask still has to be carved to match. Carried on the resolved domain
    /// rather than re-read from the environment downstream, so the box and the
    /// walls inside it can never come from two different decisions.
    pub plenum: Option<Plenum>,
}

impl Domain {
    /// One line for the log: which face got what, so an unexpected cell count
    /// can be traced to a margin without a debugger.
    pub fn describe(&self) -> String {
        let s = self.bbox.size();
        format!(
            "{}{:.0} x {:.0} x {:.0} mm, margins -x {:.0} +x {:.0} / -y {:.0} +y {:.0} / \
             -z {:.0} +z {:.0} mm",
            match self.plenum.map(|p| p.walls) {
                Some(PlenumWalls::Solid) => "plenum (walled) ",
                Some(PlenumWalls::Open) => "plenum (open outlet chamber) ",
                // Six equal margins is the isotropic room; anything else came
                // from the per-face rule. Inferred rather than stored, because
                // the margins *are* the distinction and a flag beside them
                // could disagree with them.
                None if self.lo_mm.abs_diff_eq(self.hi_mm, 1e-4)
                    && self.lo_mm.max_element() - self.lo_mm.min_element() < 1e-4 =>
                {
                    "room "
                }
                None => "room (trimmed) ",
            },
            s.x,
            s.y,
            s.z,
            self.lo_mm.x,
            self.hi_mm.x,
            self.lo_mm.y,
            self.hi_mm.y,
            self.lo_mm.z,
            self.hi_mm.z,
        )
    }
}

/// One face of the domain box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Face {
    pub axis: usize,
    pub min_side: bool,
}

impl Face {
    /// Raise this face's margin to `mm`, never lower it.
    pub(crate) fn widen(self, lo: &mut Vec3, hi: &mut Vec3, mm: f32) {
        let slot = if self.min_side {
            &mut lo[self.axis]
        } else {
            &mut hi[self.axis]
        };
        *slot = slot.max(mm);
    }
}

/// The outside face a mouth opens onto, and its hydraulic diameter.
///
/// Derived from `patch.normal`, which points *into* the fluid. A mouth flush
/// with the box face at `axis` min has its normal along `+axis`, so a positive
/// component means the outside air is on the min side — the same face
/// `sim::apply_boundaries` puts the outlet condition on, reached without
/// consulting `Mouth::axis`. A mouth whose normal is not axis-aligned has no
/// box face to claim and contributes nothing but the lateral margin.
///
/// Shared with [`crate::plenum`], which extrudes its plenums from exactly this
/// face. Deriving both from one function is what stops a plenum being built on
/// one face while the margin that has to make room for it lands on another.
pub(crate) fn axis_face(mouth: Option<&Mouth>) -> Option<(Face, f32)> {
    let m = mouth?;
    let n = m.patch.normal;
    if !n.is_finite() {
        return None;
    }
    let axis = (0..3usize).max_by(|a, b| n[*a].abs().total_cmp(&n[*b].abs()))?;
    if n[axis].abs() < 0.9 * n.length() || n[axis].abs() < 1e-3 {
        return None;
    }
    let d_h = m.hydraulic_diameter_mm();
    if !d_h.is_finite() {
        return None;
    }
    Some((
        Face {
            axis,
            min_side: n[axis] > 0.0,
        },
        d_h.max(MIN_D_H_MM),
    ))
}

/// The contract's test part, as data, for the tests in this crate.
///
/// Shared with [`crate::plenum`] rather than copied into it: both modules build
/// a domain around the *same* part, and two fixtures that drifted apart would
/// let the two domains be compared against different geometry — which is
/// precisely the comparison neither of them is allowed to get wrong.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;
    use ad_gpu::FlowPatch;

    /// A mouth on one face of `bbox`, with a rectangular opening `a x b` mm.
    ///
    /// The boundary loop is what `hydraulic_diameter_mm` measures, so it is a
    /// real rectangle rather than four coincident points: a fixture with no
    /// perimeter would give `D_h = 0` and every margin would fall to its clamp,
    /// which is the one case these tests must not accidentally be testing.
    pub(crate) fn mouth(bbox: Bbox, axis: usize, min_side: bool, a: f32, b: f32) -> Mouth {
        let e = [Vec3::X, Vec3::Y, Vec3::Z];
        let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut centre = bbox.center();
        centre[axis] = if min_side {
            bbox.min[axis]
        } else {
            bbox.max[axis]
        };
        let (half_u, half_v) = (e[i] * a * 0.5, e[j] * b * 0.5);
        let corner = |s: f32, t: f32| centre + half_u * s + half_v * t;
        Mouth {
            patch: FlowPatch {
                center_mm: centre,
                // Into the fluid: away from the box face it sits on.
                normal: if min_side { e[axis] } else { -e[axis] },
                half_u,
                half_v,
            },
            open_area_mm2: a * b,
            axis: axis as u8,
            on_min_side: min_side,
            boundary: vec![
                corner(-1.0, -1.0),
                corner(1.0, -1.0),
                corner(1.0, 1.0),
                corner(-1.0, 1.0),
            ],
        }
    }

    /// The contract's test part, to scale: 145 x 72.2 x 68.9 mm with mouth A a
    /// 139 x 15 slot on the z = 0 face and mouth B a 74 x 15 slot on y = 0.
    pub(crate) fn part() -> (Bbox, Vec<Mouth>) {
        let bbox = Bbox {
            min: Vec3::new(-74.226, 0.0, 0.0),
            max: Vec3::new(70.774, 72.185, 68.940),
        };
        let a = mouth(bbox, 2, true, 139.0, 15.0);
        let b = mouth(bbox, 1, true, 15.0, 74.0);
        (bbox, vec![a, b])
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::{mouth, part};
    use super::*;

    /// The anisotropic margins, explicitly.
    ///
    /// The tests below exercise the anisotropic *rule*. That rule is no longer
    /// what `DomainMargins::default()` returns -- see the comment there for the
    /// measurement that demoted it -- so they have to ask for it by name. A test
    /// that reads the default would silently start testing the isotropic box.
    fn anisotropic() -> DomainMargins {
        DomainMargins {
            isotropic_frac: None,
            plenum: None,
            ..DomainMargins::default()
        }
    }

    #[test]
    fn hand_set_margins_are_used_as_typed_and_win_over_every_rule() {
        let (bbox, mouths) = part();
        let mm = [10.0, 20.0, 30.0, 140.0, 50.0, 60.0];
        for base in [DomainMargins::default(), anisotropic()] {
            let d = base
                .with_explicit(Some(mm))
                .plan(bbox, &mouths, 0, 1, 0.75, 20);
            assert_eq!(d.lo_mm, Vec3::new(10.0, 30.0, 50.0));
            assert_eq!(d.hi_mm, Vec3::new(20.0, 140.0, 60.0));
            assert_eq!(d.bbox.min, bbox.min - d.lo_mm);
            assert_eq!(d.bbox.max, bbox.max + d.hi_mm);
            assert!(d.plenum.is_none());
        }
        let d = DomainMargins::default()
            .with_explicit(Some([f32::NAN; 6]))
            .plan(bbox, &mouths, 0, 1, 0.75, 20);
        assert_eq!(d.lo_mm, d.hi_mm, "a NaN margin falls back to the rules");
    }

    #[test]
    fn the_generous_margin_lands_on_the_face_the_outlet_exhausts_through() {
        // Mouth B is on y = 0 and its normal points +y into the duct, so the
        // jet leaves through the domain's y-min face. That is the face
        // `sim::apply_boundaries` puts the outlet and its sponge on, and it is
        // the only one allowed to be the largest.
        let (bbox, mouths) = part();
        let d = anisotropic().plan(bbox, &mouths, 0, 1, 0.75, 20);

        let every_other = [d.lo_mm.x, d.hi_mm.x, d.hi_mm.y, d.lo_mm.z, d.hi_mm.z];
        for m in every_other {
            assert!(
                d.lo_mm.y > m,
                "the downstream margin {} is not the largest; found {m}",
                d.lo_mm.y
            );
        }
        // ...and it really is scaled by the jet, not by the part.
        let d_h = mouths[1].hydraulic_diameter_mm();
        assert!(
            (d.lo_mm.y - DEFAULT_DOWNSTREAM_D_H * d_h).abs() < 1e-3,
            "{} vs {} D_h of {d_h}",
            d.lo_mm.y,
            DEFAULT_DOWNSTREAM_D_H
        );
        // The upstream face is the one behind the inlet mouth: mouth A is on
        // z = 0, so the plenum is at z-min and the far z face stays tight.
        assert!(
            d.lo_mm.z > d.hi_mm.z,
            "the plenum landed on the wrong z face"
        );
        assert!(
            d.lo_mm.z < d.lo_mm.y,
            "the plenum is supposed to be the modest one"
        );
    }

    /// The whole point of deriving from the normals: the user can swap which
    /// mouth blows, and the box has to turn round with them.
    #[test]
    fn swapping_the_inlet_moves_the_generous_margin_to_the_other_face() {
        let (bbox, mouths) = part();
        let m = anisotropic();
        let forward = m.plan(bbox, &mouths, 0, 1, 0.75, 20);
        let reversed = m.plan(bbox, &mouths, 1, 0, 0.75, 20);

        // Forward: generous at y-min (mouth B exhausts), plenum at z-min.
        // Reversed: generous at z-min (mouth A exhausts), plenum at y-min.
        assert!(
            reversed.lo_mm.z > reversed.lo_mm.y,
            "the jet margin did not follow the outlet"
        );
        assert!(
            reversed.lo_mm.z > forward.lo_mm.z,
            "z-min should have grown from plenum to jet"
        );
        assert!(
            reversed.lo_mm.y < forward.lo_mm.y,
            "y-min should have shrunk from jet to plenum"
        );
        assert_ne!(
            forward.bbox, reversed.bbox,
            "a swap must re-derive the box, not reuse it"
        );
    }

    /// An outlet plane sitting inside the recirculation reads the pressure of
    /// an eddy. When the sponge will not fit in the requested margin, the
    /// margin grows.
    #[test]
    fn the_sponge_and_the_outlet_plane_always_fit_downstream() {
        let (bbox, mouths) = part();
        let d_h = mouths[1].hydraulic_diameter_mm();

        for (dx, sponge, downstream_d_h) in [
            (0.75f32, 20u32, 4.0f32),
            (0.5, 20, 4.0),
            (0.75, 200, 4.0),
            (1.0, 20, 0.0),
        ] {
            let m = DomainMargins {
                downstream_d_h,
                ..anisotropic()
            };
            let d = m.plan(bbox, &mouths, 0, 1, dx, sponge);
            let needed = (sponge + SPONGE_CLEARANCE_CELLS) as f32 * dx + JET_NEAR_FIELD_D_H * d_h;
            assert!(
                d.lo_mm.y >= needed - 1e-3,
                "dx {dx}, {sponge} sponge cells, {downstream_d_h} D_h: margin {} < {needed}",
                d.lo_mm.y
            );
        }
        // A 200-cell sponge at 0.75 mm needs 151.5 mm plus the near field, well
        // past the 150 mm ceiling: the floor has to win over the clamp, or the
        // sponge would silently overlap the mouth.
        let wide = anisotropic().plan(bbox, &mouths, 0, 1, 0.75, 200);
        assert!(wide.lo_mm.y > DOWNSTREAM_RANGE_MM.1);
    }

    /// The lateral faces have to clear the mouths' own pressure field.
    ///
    /// This is the one the first cut of the module got wrong. Sizing the lateral
    /// margins off the part alone put the tight faces 10-11 mm from a mouth of
    /// `D_h = 27 mm`; the equilibrium boundary pins `p` there, and the inlet
    /// total pressure came out 14% high against the isotropic domain while `Q`
    /// was unmoved. A velocity inlet delivers the flow it was asked for whatever
    /// the pressure costs, so `Q` is exactly the wrong quantity to look for this
    /// in and the assertion has to be about the geometry instead.
    #[test]
    fn every_lateral_face_clears_the_larger_mouth_by_a_hydraulic_diameter() {
        let (bbox, mouths) = part();
        let d_h = mouths
            .iter()
            .map(|m| m.hydraulic_diameter_mm())
            .fold(0.0f32, f32::max);
        let d = anisotropic().plan(bbox, &mouths, 0, 1, 0.75, 20);

        // The four faces that are neither upstream nor downstream. The part's
        // own extent would have asked for 10.8 mm on +y and 10.3 mm on +z.
        let want = DEFAULT_LATERAL_D_H * d_h;
        for (name, got) in [
            ("-x", d.lo_mm.x),
            ("+x", d.hi_mm.x),
            ("+y", d.hi_mm.y),
            ("+z", d.hi_mm.z),
        ] {
            assert!(
                got >= want - 1e-3,
                "{name} margin {got} mm is inside {want} mm of the mouth"
            );
        }
        // ...and the floor is a floor, not an override: the downstream face is
        // still the jet's, several times larger.
        assert!(d.lo_mm.y > 3.0 * want);

        // Swapping which mouth blows must not move a lateral face. The two
        // mouths have different `D_h`, so a floor taken from "the inlet" rather
        // than "the larger" would give the same duct two different lateral
        // boxes and two different pressures depending on the button pressed.
        let swapped = anisotropic().plan(bbox, &mouths, 1, 0, 0.75, 20);
        assert_eq!(d.lo_mm.x, swapped.lo_mm.x);
        assert_eq!(d.hi_mm.x, swapped.hi_mm.x);
        assert_eq!(d.hi_mm.y, swapped.hi_mm.y);
        assert_eq!(d.hi_mm.z, swapped.hi_mm.z);
    }

    /// Obstructions are part of the scene box, so they are inside the domain by
    /// construction — including one parked out in the jet, which is the case
    /// the anisotropy is most likely to clip.
    #[test]
    fn an_obstruction_in_the_jet_stays_inside_the_domain() {
        let (part_bbox, mouths) = part();
        let blocker = Bbox {
            min: Vec3::new(-20.0, -60.0, 10.0),
            max: Vec3::new(20.0, -50.0, 30.0),
        };
        let scene = part_bbox.union(blocker);
        let d = anisotropic().plan(scene, &mouths, 0, 1, 0.75, 20);

        assert!(d.bbox.contains(blocker.min) && d.bbox.contains(blocker.max));
        // ...and the downstream margin is measured past it, not through it.
        assert!(
            d.bbox.min.y
                <= blocker.min.y - DEFAULT_DOWNSTREAM_D_H * mouths[1].hydraulic_diameter_mm()
                    + 1e-3
        );
    }

    #[test]
    fn the_anisotropic_box_is_a_fraction_of_the_isotropic_one_and_still_holds_the_part() {
        let (bbox, mouths) = part();
        let tight = anisotropic().plan(bbox, &mouths, 0, 1, 0.75, 20);
        let legacy = DomainMargins {
            isotropic_frac: Some(LEGACY_ISOTROPIC_MARGIN),
            plenum: None,
            ..DomainMargins::default()
        }
        .plan(bbox, &mouths, 0, 1, 0.75, 20);

        // The old box, reproduced: CONTRACT.md's 261 x 188 x 185 mm.
        let s = legacy.bbox.size();
        assert!(
            (s.x - 261.0).abs() < 1.0 && (s.y - 188.0).abs() < 1.0 && (s.z - 185.0).abs() < 1.0
        );

        let vol = |b: Bbox| {
            let s = b.size();
            (s.x as f64) * (s.y as f64) * (s.z as f64)
        };
        let ratio = vol(tight.bbox) / vol(legacy.bbox);
        assert!(
            ratio < 0.75,
            "the anisotropic box saved nothing: {ratio:.2} of the old volume"
        );
        // Whatever else it does, it must contain the geometry.
        assert!(tight.bbox.contains(bbox.min) && tight.bbox.contains(bbox.max));
    }

    /// The face this module widens and the face `sim::apply_boundaries` puts
    /// the outlet plus its sponge on must be the same one.
    ///
    /// They are reached by different routes on purpose — this one from
    /// `patch.normal`, so the margins follow the flow rather than a stored
    /// index; the boundary conditions from `Mouth::axis` / `on_min_side`. If
    /// they ever disagreed the convective outflow would sit on a face with only
    /// a lateral margin behind it and the exit jet would be pinned against the
    /// box, which is the failure the generous margin exists to prevent.
    #[test]
    fn the_face_derived_from_the_normal_is_the_one_the_boundary_conditions_use() {
        let (bbox, _) = part();
        for axis in 0..3usize {
            for min_side in [true, false] {
                let m = mouth(bbox, axis, min_side, 30.0, 10.0);
                let (face, d_h) = axis_face(Some(&m)).expect("an axis-aligned mouth has a face");
                assert_eq!(face.axis, m.axis as usize, "axis {axis} min {min_side}");
                assert_eq!(face.min_side, m.on_min_side, "axis {axis} min {min_side}");
                // 4A/P for a 30 x 10 rectangle is 15, which is what makes the
                // margins scale with the opening rather than with a clamp.
                assert!((d_h - 15.0).abs() < 1e-3, "D_h {d_h}");
            }
        }
    }

    #[test]
    fn a_mouth_with_no_usable_normal_falls_back_to_the_lateral_margin() {
        let (bbox, mouths) = part();
        let mut broken = mouths.clone();
        broken[1].patch.normal = Vec3::new(0.6, 0.6, 0.5).normalize();
        let d = anisotropic().plan(bbox, &broken, 0, 1, 0.75, 20);
        // No face claimed the downstream margin, so y-min is lateral like the
        // rest — a smaller box than it should be, but never a NaN one.
        assert!((d.lo_mm.y - d.hi_mm.y).abs() < 1e-6);
        assert!(d.bbox.size().min_element() > 0.0);

        // The degenerate cases: no mouths at all, and an empty scene.
        let none = anisotropic().plan(bbox, &[], 0, 1, 0.75, 20);
        assert!(none.bbox.contains(bbox.min) && none.bbox.contains(bbox.max));
        let empty = anisotropic().plan(Bbox::EMPTY, &mouths, 0, 1, 0.75, 20);
        assert!(
            empty.bbox.is_empty(),
            "an empty scene must not produce a box out of nothing"
        );
    }

    /// The shipped default is the isotropic room, and that is a measured choice
    /// rather than an oversight — the third time in this file's history that
    /// sentence has been true. The plenum domain is 23.8x faster per step and at
    /// a fixed 70,000 steps its mass imbalance is 6.83% against the room's
    /// 5.63%, which is the one number this work was gated on. See
    /// `DomainMargins::default` for the full table and for what would change it.
    ///
    /// All three domains stay reachable by name, because an A/B between any two
    /// of them should be a pair of runs of the same binary.
    #[test]
    fn the_default_is_the_room_until_a_cheaper_domain_conserves_mass_at_equal_steps() {
        let d = DomainMargins::default();
        assert_eq!(
            d.plenum, None,
            "the default domain silently became a plenum"
        );
        assert_eq!(
            d.isotropic_frac,
            Some(LEGACY_ISOTROPIC_MARGIN),
            "the default domain silently became anisotropic"
        );

        let (bbox, mouths) = part();
        let plan = d.plan(bbox, &mouths, 0, 1, 0.75, 20);
        assert!(plan.plenum.is_none(), "the default plan is not the room");
        let s = plan.bbox.size();
        assert!(
            (s.x - 261.0).abs() < 1.0 && (s.y - 188.0).abs() < 1.0 && (s.z - 185.0).abs() < 1.0
        );

        // The plenum, by name, and it really is the terminated duct.
        let named =
            DomainMargins::from_lookup(|k| (k == "AERODUCT_DOMAIN").then(|| "plenum".into()));
        assert_eq!(named.plenum, Some(Plenum::DEFAULT));
        assert!(named.plan(bbox, &mouths, 0, 1, 0.75, 20).plenum.is_some());

        // The anisotropic trim, by name.
        let trim =
            DomainMargins::from_lookup(|k| (k == "AERODUCT_DOMAIN").then(|| "trimmed".into()));
        assert_eq!(trim.plenum, None);
        assert_eq!(trim.isotropic_frac, None);
        assert_eq!(
            trim.plan(bbox, &mouths, 0, 1, 0.75, 20),
            anisotropic().plan(bbox, &mouths, 0, 1, 0.75, 20)
        );
    }

    /// A mouth that cannot be extruded is not a reason to build a broken plenum.
    #[test]
    fn a_plenum_that_cannot_be_built_falls_back_to_the_room_rather_than_to_nothing() {
        let (bbox, mouths) = part();
        let mut skew = mouths.clone();
        skew[1].patch.normal = Vec3::new(0.6, 0.6, 0.5).normalize();

        let d = DomainMargins::default().plan(bbox, &skew, 0, 1, 0.75, 20);
        assert!(
            d.plenum.is_none(),
            "a plenum was built on a mouth with no face"
        );
        assert!(d.bbox.contains(bbox.min) && d.bbox.contains(bbox.max));
        // The fallback is the *room*, deliberately: it is the expensive answer,
        // and expensive is the one that was always safe.
        let s = d.bbox.size();
        assert!(
            (s.x - 261.0).abs() < 1.0,
            "the fallback is not the isotropic room: {s:?}"
        );
    }

    #[test]
    fn the_environment_overrides_parse_and_reject_nonsense() {
        let with = |pairs: &[(&str, &str)]| {
            let owned: Vec<(String, String)> = pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            DomainMargins::from_lookup(move |k| {
                owned.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())
            })
        };

        let base = DomainMargins::default();
        assert_eq!(with(&[]), base);
        let m = with(&[
            ("AERODUCT_MARGIN_DOWNSTREAM_DH", "6"),
            ("AERODUCT_MARGIN_UPSTREAM_DH", "1.5"),
            ("AERODUCT_MARGIN_LATERAL", "0.05"),
            ("AERODUCT_MARGIN_LATERAL_DH", "0.5"),
        ]);
        assert_eq!(m.downstream_d_h, 6.0);
        assert_eq!(m.upstream_d_h, 1.5);
        assert_eq!(m.lateral_frac, 0.05);
        assert_eq!(m.lateral_d_h, 0.5);
        assert_eq!(m.isotropic_frac, None);

        // An anisotropic knob on its own opts into the anisotropic domain --
        // otherwise it would parse, store, and do nothing.
        assert_eq!(
            with(&[("AERODUCT_MARGIN_DOWNSTREAM_DH", "6")]).isotropic_frac,
            None,
            "asking for a downstream margin should select the domain that uses it"
        );
        // ...but asking for the isotropic box explicitly still wins.
        assert_eq!(
            with(&[
                ("AERODUCT_MARGIN_DOWNSTREAM_DH", "6"),
                ("AERODUCT_MARGIN_ISOTROPIC", "legacy"),
            ])
            .isotropic_frac,
            Some(LEGACY_ISOTROPIC_MARGIN)
        );
        // A knob that fails validation is not a request for anything.
        assert_eq!(
            with(&[("AERODUCT_MARGIN_DOWNSTREAM_DH", "fnord")]).isotropic_frac,
            Some(LEGACY_ISOTROPIC_MARGIN)
        );

        // A typo, a NaN and an out-of-range value must all leave the default
        // standing rather than produce a domain nobody asked for.
        for bad in ["fnord", "NaN", "-1", "1e9"] {
            assert_eq!(
                with(&[("AERODUCT_MARGIN_DOWNSTREAM_DH", bad)]).downstream_d_h,
                base.downstream_d_h,
                "{bad} was accepted"
            );
        }
        // The escape hatch is opt-in, takes a fraction, and answers to the name
        // of the margin it reproduces.
        assert_eq!(
            with(&[("AERODUCT_MARGIN_ISOTROPIC", "0.4")]).isotropic_frac,
            Some(LEGACY_ISOTROPIC_MARGIN)
        );
        assert_eq!(
            with(&[("AERODUCT_MARGIN_ISOTROPIC", "legacy")]).isotropic_frac,
            Some(LEGACY_ISOTROPIC_MARGIN)
        );
        // A value that fails validation leaves the *default* standing, and the
        // default is the plenum: a typo must not silently opt the run into a
        // different domain, in either direction.
        assert_eq!(
            with(&[("AERODUCT_MARGIN_ISOTROPIC", "fnord")]),
            DomainMargins::default()
        );
        assert_eq!(
            with(&[("AERODUCT_DOMAIN", "fnord")]),
            DomainMargins::default()
        );
        assert_eq!(
            with(&[("AERODUCT_PLENUM_WALLS", "fnord")]),
            DomainMargins::default()
        );
    }

    #[test]
    fn the_plenum_overrides_parse_and_select_the_domain_that_uses_them() {
        let with = |pairs: &[(&str, &str)]| {
            let owned: Vec<(String, String)> = pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            DomainMargins::from_lookup(move |k| {
                owned.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())
            })
        };

        let m = with(&[
            ("AERODUCT_PLENUM_UPSTREAM_DH", "3"),
            ("AERODUCT_PLENUM_DOWNSTREAM_DH", "5"),
            ("AERODUCT_PLENUM_LATERAL_CELLS", "4"),
            ("AERODUCT_PLENUM_WALLS", "open"),
        ]);
        let p = m.plenum.expect("plenum knobs select the plenum domain");
        assert_eq!(p.upstream_d_h, 3.0);
        assert_eq!(p.downstream_d_h, 5.0);
        assert_eq!(p.lateral_cells, 4);
        assert_eq!(p.walls, PlenumWalls::Open);

        // A plenum knob set while the domain has been named as a room is a
        // contradiction, and the name is the later, more explicit statement.
        assert_eq!(
            with(&[
                ("AERODUCT_PLENUM_DOWNSTREAM_DH", "5"),
                ("AERODUCT_DOMAIN", "room")
            ])
            .plenum,
            None
        );
        // ...and the older margin knobs still take the run out of the plenum,
        // or they would parse, store, and change nothing.
        assert_eq!(with(&[("AERODUCT_MARGIN_DOWNSTREAM_DH", "6")]).plenum, None);
        assert_eq!(
            with(&[("AERODUCT_MARGIN_ISOTROPIC", "legacy")]).plenum,
            None
        );

        // A knob that fails validation is not a request for anything: it must
        // neither build a plenum of zero cells nor, since the default is the
        // room, opt the run into a different domain on the strength of a typo.
        for bad in ["fnord", "NaN", "-1", "1e9"] {
            assert_eq!(
                with(&[("AERODUCT_PLENUM_UPSTREAM_DH", bad)]),
                DomainMargins::default(),
                "{bad} was accepted"
            );
        }
        // ...while a valid one is, and carries its value through.
        assert_eq!(
            with(&[
                ("AERODUCT_DOMAIN", "plenum"),
                ("AERODUCT_PLENUM_UPSTREAM_DH", "1.5")
            ])
            .plenum
            .unwrap()
            .upstream_d_h,
            1.5
        );
    }
}

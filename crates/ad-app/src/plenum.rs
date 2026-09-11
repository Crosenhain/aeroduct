//! Terminating the domain with duct instead of with room.
//!
//! # The problem this exists to solve
//!
//! [`crate::domain`] answers "how much room is there around the part". Both of
//! the answers it can give are expensive, and the cheap one is wrong:
//!
//! * the isotropic room is 261 x 188 x 185 mm around a 145 mm part, 21.6 M cells
//!   at `dx = 0.75 mm`, of which **98.6% is room air** being relaxed toward the
//!   state it is already in;
//! * trimming that room anisotropically saved 1.42x and cost mass conservation,
//!   because it moved an *equilibrium plane* — which pins `p` at the reference —
//!   to within ~10 mm of a 27 mm mouth. See the measurement recorded on
//!   [`crate::domain::DomainMargins::default`].
//!
//! The mistake both share is treating a nearby plane of prescribed equilibrium
//! as if it were open sky. It is not; it is a pressure boundary condition, and
//! putting one close to a mouth is a modelling error that no amount of margin
//! *tuning* fixes, only margin *spending*.
//!
//! So this module does something else: it does not shrink the room, it replaces
//! it. The domain becomes the part's bounding box plus two **plenums** — a
//! walled straight extension on each mouth — and the boundary conditions move
//! out to the ends of those extensions, where the flow is one-dimensional and a
//! velocity inlet and a pressure outlet are the *right* conditions rather than
//! tolerable ones.
//!
//! That is not an approximation of a duct test; it is the geometry of one. A
//! fitting's loss coefficient is measured (AMCA 210 / ASHRAE) by bolting
//! straight duct to both ends and instrumenting a few diameters away, which is
//! exactly what is built here.
//!
//! # What it costs and what it buys
//!
//! Measured against the isotropic room, same `dx`, same step count,
//! sequentially on an idle GPU — the full tables are on [`Plenum::default`] and
//! [`crate::domain::DomainMargins::default`].
//!
//! The cell count falls 3.3x, from 21.57 M to 6.55 M. The **time** falls 23.8x,
//! from 3.634 to 0.153 ms/step, and the gap between those two numbers is the
//! interesting part: a solid cell returns from `stream_collide` after reading
//! one flag byte, while a fluid cell moves 157 bytes of distribution functions.
//! The room domain is 21.44 M fluid cells and this one is 0.78 M, so what
//! actually got 27x smaller is the memory traffic, on a kernel that is
//! bandwidth-bound by construction.
//!
//! The corollary is worth keeping in mind when reading a profile: the room
//! domain runs at 95% of roofline and this one reports 690%, because
//! `Solver::bytes_per_step` counts every cell as if it moved a full population
//! set. The ms/step is real; the roofline percentage is not.
//!
//! **This is not the default.** It is faster, it is a better-posed problem, and
//! at 350,000 steps it agrees with the room's pressure drop to 5% — and at
//! neither 70,000 nor 350,000 steps does it beat the room on the mass imbalance
//! that gates everything. [`crate::domain::DomainMargins::default`] has both
//! tables and the reasoning.
//!
//! # Where the walls are
//!
//! Everything that is not the duct passage or one of the two extensions becomes
//! [`ad_gpu::flags::SOLID`]:
//!
//! 1. each plenum slab is filled solid except for the mouth's own footprint,
//!    swept out to the domain face — that is the extension tube;
//! 2. anything still fluid that a flood fill from the inlet cannot reach is
//!    filled too, which removes the pockets of trapped air between the part and
//!    its own bounding box.
//!
//! Step 2 is what makes the mass balance mean something: afterwards the fluid
//! region is *connected*, and it has exactly two openings, both of them boundary
//! conditions we chose. There is no third place for mass to go.
//!
//! The voxeliser's [`ad_gpu::flags::SOLID_BOUNDARY`] is deliberately not
//! maintained across the carve. Nothing consumes it — bounce-back runs off
//! `ad_solver::PaddedDomain::link_mask`, which is rebuilt from the flags this
//! module produces — and a second full pass over six million cells to keep a
//! cosmetic bit accurate would be the most expensive no-op in the startup path.

use ad_geom::Mouth;
use ad_gpu::{flags, Bbox, Grid};
use glam::{UVec3, Vec3};

use crate::domain::{axis_face, Domain, Face};

/// Depth of the inlet plenum, in inlet hydraulic diameters.
///
/// CONTRACT.md: "Place it ~2 D_h upstream inside a straight extension so a
/// boundary layer develops rather than injecting plug flow." Two is that number.
/// It was only ever approximated before — the domain reserved 2.25 `D_h` of
/// *open room* behind the mouth and then put the velocity boundary on the mouth
/// itself, so the plug profile went straight into the duct and the reserved room
/// did nothing but cost cells.
pub const DEFAULT_UPSTREAM_D_H: f32 = 2.0;

/// Depth of the outlet plenum, in outlet hydraulic diameters.
///
/// # What the sweep actually said
///
/// Three, chosen for cost, because depth is not what the answer depends on.
/// Swept at 2, 3, 4, 5, 6 and 8 `D_h`, all at `dx = 0.75 mm` and 70,000 steps:
///
/// | `D_h` | 2 | 3 | 4 | 5 | 6 | 8 |
/// |---|---|---|---|---|---|---|
/// | cells | 5.4 M | 6.5 M | 7.7 M | 8.8 M | 9.9 M | 11.9 M |
/// | mass imbalance | 7.18% | 6.83% | 8.83% | 4.31% | 8.27% | 6.76% |
/// | `dp_total` | 72 +/- 9 | 74 +/- 12 | 84 +/- 7 | 65 +/- 8 | 84 +/- 6 | 69 +/- 6 |
///
/// That is scatter, not a trend: doubling the plenum and doubling it again moves
/// the answer by less than the run moves on its own. Which is the useful
/// finding, and it says two things at once. The good one: the outflow condition
/// is *terminating* the domain rather than reflecting off it, because a
/// reflective outlet shows up first as an answer that depends on where you put
/// the outlet, and this one does not. The bad one: at 70,000 steps none of these
/// runs is converged well enough for a 10% difference to mean anything, and no
/// depth fixes that. See [`Plenum::default`] for what does.
///
/// So the depth is set by the floors that have a physical argument behind them —
/// the sponge, its clearance, and [`NEAR_FIELD_D_H`] — and then by cost.
pub const DEFAULT_DOWNSTREAM_D_H: f32 = 3.0;

/// Lateral slack around the part's bounding box, in cells.
///
/// One. The lateral faces are behind solid in this mode, so the margin is not
/// buying clearance from a pressure boundary the way [`crate::domain`]'s does;
/// it is only keeping `Grid::covering`'s round-up from putting the part's own
/// surface exactly on the outermost cell row.
pub const DEFAULT_LATERAL_CELLS: u32 = 1;

/// Absolute clamps on the plenum depths, mm. A degenerate mouth must not
/// produce a plenum of zero cells or one larger than the part.
const UPSTREAM_RANGE_MM: (f32, f32) = (10.0, 120.0);
const DOWNSTREAM_RANGE_MM: (f32, f32) = (15.0, 200.0);

/// Cells of clearance demanded between the sponge and the mouth.
const SPONGE_CLEARANCE_CELLS: u32 = 2;

/// Extent of the exit flow's near field, in outlet hydraulic diameters.
///
/// The outflow plane has to sit beyond this whatever the multiplier asked for.
/// A `dp` read inside a recirculation is the pressure of an eddy, which is a
/// wrong number rather than a noisy one; and unlike noise it does not average
/// out. With walled plenums there is no free jet to have a near field, but the
/// floor is kept for the open-walled variant and for a mouth whose `D_h` is
/// large next to the requested depth.
const NEAR_FIELD_D_H: f32 = 1.0;

/// What the sides of the outlet plenum do.
///
/// This is the question [`crate::domain`] got wrong twice, so it is a knob with
/// two measured settings rather than a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlenumWalls {
    /// Solid. The outlet plenum is a straight duct extension, so the domain has
    /// exactly two openings and no entrainment anywhere.
    Solid,
    /// Open. The outlet plenum is a chamber whose sides are the domain's
    /// equilibrium faces, so the exit flow can entrain — the arrangement
    /// CONTRACT.md's "box sides: equilibrium" describes, at plenum scale.
    ///
    /// Kept because "should the sides entrain?" deserved an answer from a
    /// measurement rather than from an argument. It got one; see
    /// [`Plenum::default`].
    Open,
}

/// A terminated duct domain: bounding box plus two plenums.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plenum {
    /// Inlet plenum depth, in inlet hydraulic diameters.
    pub upstream_d_h: f32,
    /// Outlet plenum depth, in outlet hydraulic diameters.
    pub downstream_d_h: f32,
    /// Slack on the four faces that are neither plenum, in cells.
    pub lateral_cells: u32,
    pub walls: PlenumWalls,
}

impl Plenum {
    /// The measured configuration, as a constant so it can be named in one.
    /// See [`Plenum::default`] for the numbers that chose it.
    pub const DEFAULT: Self = Self {
        upstream_d_h: DEFAULT_UPSTREAM_D_H,
        downstream_d_h: DEFAULT_DOWNSTREAM_D_H,
        lateral_cells: DEFAULT_LATERAL_CELLS,
        walls: PlenumWalls::Solid,
    };
}

impl Default for Plenum {
    /// The measured configuration. **Not what the app runs** — see
    /// [`crate::domain::DomainMargins::default`] for why the room is still the
    /// shipped domain and what would have to be true to change that.
    ///
    /// # Walled or open: the A/B the sides question deserved
    ///
    /// `dx = 0.75 mm`, 3 m/s on mouth A, 70,000 steps, run one at a time on an
    /// otherwise idle RTX 4090. Same box, same inlet extension, same cell count;
    /// the only difference is what the outlet plenum's sides do.
    ///
    /// | | room (shipped) | **plenum, walled** | plenum, open sides |
    /// |---|---|---|---|
    /// | cells | 21.57 M | **6.55 M** | 6.55 M |
    /// | *fluid* cells | 21.44 M | **0.78 M** | 3.83 M |
    /// | ms/step | 3.634 | **0.153** | 1.042 |
    /// | `Q_in` | 6.131 +/- 0.008 L/s | 6.34 +/- 0.03 | 6.38 +/- 0.04 |
    /// | `Q_out` | 6.41 +/- 0.05 | 7.20 +/- 0.09 | 7.14 +/- 0.07 |
    /// | mass imbalance | 5.63% | 6.83% | 5.24% |
    /// | `dp_total` | 65 +/- 2 Pa | 74 +/- 12 | 77 +/- 7 |
    /// | outlet backflow | 0.15 | **0.009** | 0.023 |
    ///
    /// Read that honestly: **open sides win the mass imbalance at this step
    /// count** (5.24% against 6.83%, and against the room's own 5.63%), and lose
    /// everything else. They cost 6.8x the time — 3.83 M fluid cells against
    /// 0.78 M, because the outlet chamber is open air that has to be simulated —
    /// they leave `dp_total` 18% high at 77 +/- 7 Pa where walled sits at 74 and
    /// the room at 65, and they put an equilibrium plane, which is a pressure
    /// boundary, back within a plenum's reach of the flow. That last one is the
    /// mistake the anisotropic trim was demoted for, and it is not fixed by
    /// being further away, only made smaller.
    ///
    /// The 5.24% is also not a stable win: the same metric moves between 4.31%
    /// and 8.83% across walled plenums of different depths (see
    /// [`DEFAULT_DOWNSTREAM_D_H`]) purely because neither configuration is
    /// converged at 70,000 steps. Preferring open sides on the strength of 1.6
    /// points of an unconverged number, at 6.8x the cost, would be reading noise.
    ///
    /// So the sides question resolves to **neither** equilibrium nor free-slip:
    /// put them behind the duct wall and there is no side in contact with fluid
    /// to have a condition on. That needs no boundary condition the solver did
    /// not already have. **No solver or shader change was required for any of
    /// this** — `crates/ad-solver` and `shaders/lbm` are untouched.
    ///
    /// # What the extension fixed outright
    ///
    /// * **Backflow at the outlet plane: 15% -> 0.9%.** The measurement plane
    ///   sits one cell inside mouth B either way; in the room domain that cell
    ///   is inside the recirculation of a free jet. This is the "the outlet
    ///   plane sits in 15% backflow, which is itself suspect" problem, gone.
    /// * **The inlet plane stopped reading a fan disc.** The room domain's
    ///   velocity boundary is the mouth itself, in open air, so air arrives at
    ///   it sideways: `|u|` mean 3.47 m/s against a through-plane 2.97, peak
    ///   10.6. With 2 `D_h` of walled extension in front, the same plane reads
    ///   `|u|` mean 3.06 against a through-plane 3.05, peak 5.5 — a duct
    ///   profile, which is what CONTRACT.md asked for and what the room domain
    ///   never actually delivered.
    /// * **Mass has nowhere else to go.** The room domain has 381,270
    ///   equilibrium cells, each of which resets its populations every step and
    ///   is therefore a mass source or sink. The walled plenum domain has zero:
    ///   3,730 inlet cells, 1,990 outlet cells, and wall.
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Plenum {
    /// Resolve the plenum depths for a given pair of mouths.
    ///
    /// `None` when a mouth has no axis-aligned face to extrude from, which is
    /// the caller's cue to fall back to a room domain rather than to build a
    /// plenum pointing in a direction nothing flows.
    pub fn depths(
        &self,
        mouths: &[Mouth],
        inlet: usize,
        outlet: usize,
        dx_mm: f32,
        sponge_cells: u32,
    ) -> Option<Depths> {
        let dx = if dx_mm.is_finite() && dx_mm > 0.0 {
            dx_mm
        } else {
            1.0
        };
        let (in_face, d_h_in) = axis_face(mouths.get(inlet))?;
        let (out_face, d_h_out) = axis_face(mouths.get(outlet))?;

        let upstream = (self.upstream_d_h * d_h_in).clamp(UPSTREAM_RANGE_MM.0, UPSTREAM_RANGE_MM.1);
        // The sponge, its clearance and the near field are a hard floor: the
        // requested depth may be raised by them but never lowered past them.
        let sponge = (sponge_cells + SPONGE_CLEARANCE_CELLS) as f32 * dx;
        let downstream = (self.downstream_d_h * d_h_out)
            .clamp(DOWNSTREAM_RANGE_MM.0, DOWNSTREAM_RANGE_MM.1)
            .max(NEAR_FIELD_D_H * d_h_out + sponge);

        Some(Depths {
            in_face,
            out_face,
            d_h_in,
            d_h_out,
            upstream,
            downstream,
            lateral: self.lateral_cells as f32 * dx,
        })
    }

    /// The domain box: the scene, plus a plenum on each mouth's own face and a
    /// cell of slack on the other four.
    ///
    /// `scene` is the whole scene box, obstructions included, so an STL the user
    /// parks anywhere is inside the domain by construction. In this mode it will
    /// also be *entombed* unless it sits in the passage or a plenum, which is
    /// the honest consequence of a domain that has no room in it: there is no
    /// exit jet left to park something in.
    pub fn plan(
        &self,
        scene: Bbox,
        mouths: &[Mouth],
        inlet: usize,
        outlet: usize,
        dx_mm: f32,
        sponge_cells: u32,
    ) -> Option<Domain> {
        if scene.is_empty() {
            return None;
        }
        let d = self.depths(mouths, inlet, outlet, dx_mm, sponge_cells)?;

        let mut lo = Vec3::splat(d.lateral);
        let mut hi = lo;
        // Downstream first, so that two mouths exhausting through the same face
        // keep the larger of the two margins rather than the last one written.
        d.out_face.widen(&mut lo, &mut hi, d.downstream);
        d.in_face.widen(&mut lo, &mut hi, d.upstream);

        Some(Domain {
            bbox: Bbox {
                min: scene.min - lo,
                max: scene.max + hi,
            },
            lo_mm: lo,
            hi_mm: hi,
            plenum: Some(*self),
        })
    }

    /// Fill everything that is not passage or plenum with solid, and say where
    /// the velocity inlet should go.
    ///
    /// Those two are one call on purpose. The inlet belongs at the far end of
    /// the inlet extension — that is the whole point of building it, because the
    /// boundary injects plug flow and the [`DEFAULT_UPSTREAM_D_H`] hydraulic
    /// diameters between it and the mouth are what turn plug flow into a profile
    /// with a boundary layer on it. But if the extension could not be built, the
    /// inlet must go back on the mouth: a velocity boundary on the domain face
    /// of an *un-walled* plenum covers the whole face, and drives a sheet of
    /// moving air across the entire box. Returning the plane from the function
    /// that built the walls is what makes those two facts impossible to get out
    /// of step.
    ///
    /// Order within the carve is not arbitrary either: the plenum slabs are
    /// walled off *first* so that the flood fill afterwards cannot escape
    /// through them into the trapped air around the part, which would defeat the
    /// whole exercise by keeping every cell it was meant to remove.
    pub fn carve(
        &self,
        mask: &mut [u8],
        grid: Grid,
        mouths: &[Mouth],
        inlet: usize,
        outlet: usize,
    ) -> CarveReport {
        let fluid_before = mask.iter().filter(|f| flags::is_fluid(**f)).count();
        let mut report = CarveReport {
            fluid_before,
            fluid_after: fluid_before,
            connected: true,
            inlet_plane_mm: None,
        };

        // Both cross-sections are read off the *un-carved* mask, before either
        // wall goes in. On this part the two plenum slabs overlap in the corner
        // where the two mouths' outsides meet, so walling one first would blank
        // cells the other still has to read to find its own opening.
        let inlet_section = section(grid, mask, &mouths[inlet]);
        let outlet_section = section(grid, mask, &mouths[outlet]);

        // 1. The extensions. The inlet one always: CONTRACT.md asks for a
        //    straight extension there and an open inlet plenum is just the fan
        //    disc in free air that this replaces.
        if let Some(s) = &inlet_section {
            wall_off_plenum(mask, grid, s);
            let a = s.axis;
            let k = if s.min_side { 0 } else { grid.dims[a] - 1 };
            report.inlet_plane_mm = Some(grid.origin_mm[a] + k as f32 * grid.dx_mm);
        }
        if self.walls == PlenumWalls::Solid {
            if let Some(s) = &outlet_section {
                wall_off_plenum(mask, grid, s);
            }
        }

        // 2. Everything the inlet cannot reach. Only meaningful once both
        //    plenums are walled: with an open outlet plenum the trapped air is
        //    connected to the chamber, so there is nothing to remove and the
        //    fill would be an expensive way to change nothing.
        if self.walls == PlenumWalls::Solid {
            if let (Some(a), Some(b)) = (&inlet_section, &outlet_section) {
                report.connected = retain_connected(mask, grid, a, b);
            }
        }

        report.fluid_after = mask.iter().filter(|f| flags::is_fluid(**f)).count();
        report
    }
}

/// The resolved plenum geometry, in mm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Depths {
    pub in_face: Face,
    pub out_face: Face,
    pub d_h_in: f32,
    pub d_h_out: f32,
    /// Inlet plenum depth, mm.
    pub upstream: f32,
    /// Outlet plenum depth, mm.
    pub downstream: f32,
    /// Slack on the other four faces, mm.
    pub lateral: f32,
}

/// What [`Plenum::carve`] removed, for the log.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarveReport {
    pub fluid_before: usize,
    pub fluid_after: usize,
    /// Whether the flood fill reached the outlet. False means the fill was
    /// abandoned and the mask left alone.
    pub connected: bool,
    /// Where the velocity inlet goes, as a coordinate along the inlet mouth's
    /// own axis. `None` means no extension was built and the inlet belongs back
    /// on the mouth — see [`Plenum::carve`].
    pub inlet_plane_mm: Option<f32>,
}

impl CarveReport {
    pub fn describe(&self) -> String {
        format!(
            "plenum carve: {:.2} M fluid cells -> {:.2} M ({:.1}% removed){}",
            self.fluid_before as f64 / 1e6,
            self.fluid_after as f64 / 1e6,
            100.0 * (1.0 - self.fluid_after as f64 / self.fluid_before.max(1) as f64),
            if self.connected {
                ""
            } else {
                " [ABANDONED: the inlet does not reach the outlet]"
            },
        )
    }
}

/// The cross-section a plenum is extruded from: the mouth's *voxelised*
/// opening.
///
/// # Why not the mouth's bounding rectangle
///
/// Because the rectangle is bigger than the hole, and the difference is flow.
/// `Mouth::patch` carries a bounding rectangle — 140 x 15 mm on the test part —
/// while the opening inside it is 2095 mm^2. Extruding the rectangle builds a
/// tube 6% wider than the duct it feeds, which does two things, both bad:
///
/// * the velocity inlet at the far end covers 3,948 cells rather than the
///   3,730 the opening actually has, and `Q` is `u` times whatever area the flag
///   ended up covering: measured, `Q_in` read 6.68 L/s from the rectangle and
///   6.34 from the opening, a 5% error in the one quantity a velocity inlet is
///   supposed to control exactly;
/// * the tube meets the duct at a step, which is a sudden contraction with its
///   own loss, invented by the mesh rather than present in the part.
///
/// Taking the fluid cells on the mouth's own plane instead makes the extension
/// exactly as open as the duct, for any mouth shape, without ever needing to
/// know that this one happens to be a slot. The footprint rectangle is still
/// used, but only to *clip*: the mouth's plane runs the width of the domain, and
/// the fluid on it beyond the rim is the open air beside the part.
struct Section {
    axis: usize,
    /// The first cell row at or inside the mouth's plane. The slab this plenum
    /// fills is everything outside it.
    plane_row: u32,
    min_side: bool,
    u_axis: usize,
    v_axis: usize,
    /// One flag per in-plane cell: is the duct open here? Indexed
    /// `v * u_len + u`.
    open: Vec<bool>,
    u_len: usize,
}

impl Section {
    #[inline]
    fn is_open(&self, c: UVec3) -> bool {
        let (u, v) = (c[self.u_axis] as usize, c[self.v_axis] as usize);
        self.open[v * self.u_len + u]
    }
}

fn section(grid: Grid, mask: &[u8], mouth: &Mouth) -> Option<Section> {
    let (face, _) = axis_face(Some(mouth))?;
    let axis = face.axis;
    let t = (mouth.patch.center_mm[axis] - grid.origin_mm[axis]) / grid.dx_mm;
    if !t.is_finite() {
        return None;
    }
    let n = grid.dims[axis];
    // The first row at or *inside* the plane, from whichever side the mouth
    // opens onto. See `wall_off_plenum` for why this is ceil/floor rather than
    // round, and what one row of slack cost when it was not.
    let plane_row = if face.min_side {
        t.ceil().clamp(0.0, (n - 1) as f32) as u32
    } else {
        t.floor().clamp(0.0, (n - 1) as f32) as u32
    };

    let (u_axis, v_axis) = ((axis + 1) % 3, (axis + 2) % 3);
    let half = footprint_half(mouth, grid.dx_mm);
    let centre = mouth.patch.center_mm;
    let (nu, nv) = (grid.dims[u_axis] as usize, grid.dims[v_axis] as usize);
    let mut open = vec![false; nu * nv];
    let mut count = 0usize;
    for v in 0..nv {
        for u in 0..nu {
            let mut c = UVec3::ZERO;
            c[axis] = plane_row;
            c[u_axis] = u as u32;
            c[v_axis] = v as u32;
            let p = grid.cell_center_mm(c);
            let in_footprint = (p[u_axis] - centre[u_axis]).abs() <= half[u_axis]
                && (p[v_axis] - centre[v_axis]).abs() <= half[v_axis];
            if in_footprint && flags::is_fluid(mask[grid.linear(c) as usize]) {
                open[v * nu + u] = true;
                count += 1;
            }
        }
    }
    if count == 0 {
        log::warn!(
            "the mouth at {:?} has no open cells on its own plane; the plenum would be a \
             blind hole, so it has not been built",
            mouth.patch.center_mm
        );
        return None;
    }
    Some(Section {
        axis,
        plane_row,
        min_side: face.min_side,
        u_axis,
        v_axis,
        open,
        u_len: nu,
    })
}

/// The mouth's footprint, as a half-extent per axis.
///
/// Component-wise rather than "`half_u` is along `(axis+1)%3`", for the reason
/// spelled out on [`crate::sim::mark_inlet`]: `ad_geom::mouths::plane_axes`
/// swaps the in-plane pair on a min-side face, and both mouths of the test part
/// are on min-side faces. Adding the two absolute vectors gives the half-size on
/// every axis at once whichever order they arrived in.
///
/// The half-cell of slack is the same one `mark_inlet` uses, and for the same
/// reason: without it a 15 mm slot at `dx = 0.75 mm` loses its outermost row to
/// rounding — here that would be a tube one row narrower than the mouth feeding
/// it, which is a sudden contraction nobody asked for.
fn footprint_half(mouth: &Mouth, dx_mm: f32) -> Vec3 {
    mouth.patch.half_u.abs() + mouth.patch.half_v.abs() + Vec3::splat(dx_mm * 0.5)
}

/// Fill one plenum slab with solid, except for the mouth's own opening swept
/// out to the domain face.
///
/// Only ever *adds* solid. An obstruction the user parked in the extension was
/// marked solid by the voxeliser and stays that way; nothing here can open a
/// hole in geometry.
///
/// # Why the slab boundary is `ceil`, not `round`
///
/// The slab is every cell row whose *centre* lies outside the mouth's plane, and
/// nothing else. Rounding to the nearest row instead is off by one whenever the
/// plane happens to fall in the outer half of a cell — and on the test part it
/// does, on mouth B: the plane sat at 101.45 cells from the origin, `round` made
/// row 101 "the mouth plane" and left it alone, and row 101's centre is 0.34 mm
/// *outside* the part, where the voxeliser had quite correctly marked the whole
/// row fluid.
///
/// One uncarved row is not a rounding blemish, it is a hole. That row ran the
/// full width of the domain, so the outlet extension was connected to the open
/// air beside the part through a 0.75 mm gap: the flood fill walked straight out
/// of it and kept 2.19 M cells instead of 0.81 M, and the simulation would have
/// leaked flow through it too. It is the one arithmetic detail in this module
/// that has to be exactly right, which is why it has a test of its own.
fn wall_off_plenum(mask: &mut [u8], grid: Grid, section: &Section) {
    let axis = section.axis;
    let n = grid.dims[axis];
    let rows: std::ops::Range<u32> = if section.min_side {
        0..section.plane_row
    } else {
        (section.plane_row + 1).min(n)..n
    };
    for row in rows {
        for a in 0..grid.dims[section.u_axis] {
            for b in 0..grid.dims[section.v_axis] {
                let mut c = UVec3::ZERO;
                c[axis] = row;
                c[section.u_axis] = a;
                c[section.v_axis] = b;
                if !section.is_open(c) {
                    mask[grid.linear(c) as usize] = flags::SOLID;
                }
            }
        }
    }
}

/// Solid-fill every fluid cell the inlet cannot reach.
///
/// Returns false, having changed nothing, when the fill does not reach the
/// outlet mouth. That is not a tuning failure, it is a broken problem — a duct
/// whose two openings are not connected through the fluid — and quietly
/// entombing the outlet would turn it into a sealed box that runs happily and
/// reports nothing.
fn retain_connected(mask: &mut [u8], grid: Grid, inlet: &Section, outlet: &Section) -> bool {
    let n = mask.len();
    let seeds = plenum_face_cells(grid, inlet);
    let targets = plenum_face_cells(grid, outlet);

    let mut seen = vec![false; n];
    let mut stack: Vec<u32> = Vec::with_capacity(1 << 16);
    for c in seeds {
        let i = grid.linear(c) as usize;
        if flags::is_fluid(mask[i]) && !seen[i] {
            seen[i] = true;
            stack.push(grid.linear(c));
        }
    }
    // Six-connected, matching the sense in which a lattice cell is "reachable":
    // a diagonal-only connection is one lattice link wide and is not a passage.
    while let Some(cell) = stack.pop() {
        let c = UVec3::new(
            cell % grid.dims.x,
            (cell / grid.dims.x) % grid.dims.y,
            cell / (grid.dims.x * grid.dims.y),
        );
        for axis in 0..3usize {
            for step in [-1i32, 1] {
                let v = c[axis] as i32 + step;
                if v < 0 || v >= grid.dims[axis] as i32 {
                    continue;
                }
                let mut nb = c;
                nb[axis] = v as u32;
                let j = grid.linear(nb) as usize;
                if flags::is_fluid(mask[j]) && !seen[j] {
                    seen[j] = true;
                    stack.push(grid.linear(nb));
                }
            }
        }
    }

    let reached_outlet = targets.into_iter().any(|c| seen[grid.linear(c) as usize]);
    if !reached_outlet {
        return false;
    }
    for i in 0..n {
        if !seen[i] && flags::is_fluid(mask[i]) {
            mask[i] = flags::SOLID;
        }
    }
    true
}

/// Cells on the domain face at the far end of a plenum: the extension's own
/// cross-section, at the row the boundary condition lives on.
///
/// The seeds of the flood fill at one end and its target at the other, and the
/// same set the velocity inlet is marked over — so an inlet cell, a seed cell
/// and a tube cell can never be three different ideas of "the opening".
fn plenum_face_cells(grid: Grid, section: &Section) -> Vec<UVec3> {
    let axis = section.axis;
    let row = if section.min_side {
        0
    } else {
        grid.dims[axis] - 1
    };
    let mut out = Vec::new();
    for a in 0..grid.dims[section.u_axis] {
        for b in 0..grid.dims[section.v_axis] {
            let mut c = UVec3::ZERO;
            c[axis] = row;
            c[section.u_axis] = a;
            c[section.v_axis] = b;
            if section.is_open(c) {
                out.push(c);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::test_fixtures::{mouth, part};

    fn grid_for(p: &Plenum, scene: Bbox, mouths: &[Mouth], dx: f32) -> (Domain, Grid) {
        let d = p
            .plan(scene, mouths, 0, 1, dx, 20)
            .expect("axis-aligned mouths give a plan");
        (d, Grid::covering(d.bbox, dx))
    }

    /// The depth of each plenum is set by *its own* mouth, not by the part.
    #[test]
    fn plenum_depth_scales_with_the_hydraulic_diameter_of_its_own_mouth() {
        let (_scene, mouths) = part();
        let p = Plenum::default();
        let d = p.depths(&mouths, 0, 1, 0.75, 20).unwrap();

        assert!(
            (d.upstream - DEFAULT_UPSTREAM_D_H * d.d_h_in).abs() < 1e-3,
            "{d:?}"
        );
        assert!(
            (d.downstream - DEFAULT_DOWNSTREAM_D_H * d.d_h_out).abs() < 1e-3,
            "{d:?}"
        );
        // ...and the two mouths really do have different D_h, so a depth taken
        // from the wrong one would be visible here.
        assert!(
            (d.d_h_in - d.d_h_out).abs() > 1.0,
            "the fixture cannot tell the two apart"
        );

        // Doubling the multiplier doubles the plenum, which is the property that
        // makes a depth sweep a meaningful experiment rather than a clamp sweep.
        let deep = Plenum {
            upstream_d_h: 4.0,
            ..p
        }
        .depths(&mouths, 0, 1, 0.75, 20)
        .unwrap();
        assert!((deep.upstream - 2.0 * d.upstream).abs() < 1e-3);
    }

    /// The outflow plane has to clear the near field and the sponge, whatever
    /// depth was asked for.
    #[test]
    fn the_outlet_plane_sits_outside_the_near_field_and_the_sponge() {
        let (scene, mouths) = part();
        for (dx, sponge, downstream_d_h) in [
            (0.75f32, 20u32, 3.0f32),
            (0.75, 20, 0.0),
            (0.5, 200, 3.0),
            (1.0, 20, 1.0),
        ] {
            let p = Plenum {
                downstream_d_h,
                ..Plenum::default()
            };
            let d = p.depths(&mouths, 0, 1, dx, sponge).unwrap();
            let needed = NEAR_FIELD_D_H * d.d_h_out + (sponge + SPONGE_CLEARANCE_CELLS) as f32 * dx;
            assert!(
                d.downstream >= needed - 1e-3,
                "dx {dx}, {sponge} sponge cells, {downstream_d_h} D_h: {} < {needed}",
                d.downstream
            );
            // ...and the box really is that deep on the outlet's own face. Mouth
            // B is on y = 0 with its normal +y, so the plenum is at y-min.
            let plan = p.plan(scene, &mouths, 0, 1, dx, sponge).unwrap();
            assert!(
                plan.lo_mm.y >= needed - 1e-3,
                "the plan did not spend the depth it computed"
            );
        }
    }

    /// Swapping which mouth blows has to re-derive *both* plenums: the extension
    /// the boundary layer develops in must follow the inlet, and the one the
    /// outflow condition terminates in must follow the outlet.
    #[test]
    fn an_inlet_swap_re_derives_both_plenums() {
        let (scene, mouths) = part();
        let p = Plenum::default();
        let fwd = p.plan(scene, &mouths, 0, 1, 0.75, 20).unwrap();
        let rev = p.plan(scene, &mouths, 1, 0, 0.75, 20).unwrap();

        // Forward: inlet on A (z-min), outlet on B (y-min).
        // Reversed: inlet on B (y-min), outlet on A (z-min).
        let d_fwd = p.depths(&mouths, 0, 1, 0.75, 20).unwrap();
        let d_rev = p.depths(&mouths, 1, 0, 0.75, 20).unwrap();
        assert!(
            (fwd.lo_mm.z - d_fwd.upstream).abs() < 1e-3,
            "inlet plenum is not on z-min"
        );
        assert!(
            (fwd.lo_mm.y - d_fwd.downstream).abs() < 1e-3,
            "outlet plenum is not on y-min"
        );
        assert!(
            (rev.lo_mm.y - d_rev.upstream).abs() < 1e-3,
            "the inlet plenum did not follow"
        );
        assert!(
            (rev.lo_mm.z - d_rev.downstream).abs() < 1e-3,
            "the outlet plenum did not follow"
        );
        assert_ne!(
            fwd.bbox, rev.bbox,
            "a swap must re-derive the box, not reuse it"
        );

        // The four faces that are neither plenum keep their slack either way.
        for (name, a, b) in [
            ("-x", fwd.lo_mm.x, rev.lo_mm.x),
            ("+x", fwd.hi_mm.x, rev.hi_mm.x),
            ("+y", fwd.hi_mm.y, rev.hi_mm.y),
            ("+z", fwd.hi_mm.z, rev.hi_mm.z),
        ] {
            assert_eq!(a, b, "the {name} face moved when the inlet was swapped");
        }
    }

    /// Anything the user drops in the scene is inside the domain box, because
    /// the plan is built from the scene box rather than from the duct's.
    #[test]
    fn an_obstruction_stl_stays_inside_the_domain() {
        let (part_bbox, mouths) = part();
        for blocker in [
            // Out in what used to be the exit jet.
            Bbox {
                min: Vec3::new(-20.0, -60.0, 10.0),
                max: Vec3::new(20.0, -50.0, 30.0),
            },
            // Behind the inlet, inside what is now the inlet plenum.
            Bbox {
                min: Vec3::new(-10.0, 55.0, -30.0),
                max: Vec3::new(10.0, 65.0, -20.0),
            },
            // Off to one side, where the plenum domain has almost no margin.
            Bbox {
                min: Vec3::new(80.0, 10.0, 10.0),
                max: Vec3::new(95.0, 20.0, 20.0),
            },
        ] {
            let scene = part_bbox.union(blocker);
            let d = Plenum::default()
                .plan(scene, &mouths, 0, 1, 0.75, 20)
                .unwrap();
            assert!(
                d.bbox.contains(blocker.min) && d.bbox.contains(blocker.max),
                "{blocker:?} fell outside {:?}",
                d.bbox
            );
            // ...and the plenum is measured past it rather than through it.
            let depths = Plenum::default().depths(&mouths, 0, 1, 0.75, 20).unwrap();
            assert!(d.bbox.min.y <= scene.min.y - depths.downstream + 1e-3);
        }
    }

    /// The whole cost argument in one assertion.
    #[test]
    fn the_plenum_box_is_a_fraction_of_the_room_and_still_holds_the_part() {
        let (scene, mouths) = part();
        let plenum = Plenum::default()
            .plan(scene, &mouths, 0, 1, 0.75, 20)
            .unwrap();
        let room = crate::domain::DomainMargins {
            isotropic_frac: Some(crate::domain::LEGACY_ISOTROPIC_MARGIN),
            plenum: None,
            ..crate::domain::DomainMargins::default()
        }
        .plan(scene, &mouths, 0, 1, 0.75, 20);

        let cells = |b: Bbox| Grid::covering(b, 0.75).cell_count() as f64;
        let ratio = cells(room.bbox) / cells(plenum.bbox);
        assert!(
            ratio > 3.0,
            "the plenum domain saved almost nothing: {ratio:.2}x"
        );
        assert!(plenum.bbox.contains(scene.min) && plenum.bbox.contains(scene.max));
    }

    // --- the carve ---------------------------------------------------------

    /// A duct with two mouths, in the plenum domain, carved.
    ///
    /// The "part" is a solid slab pierced by a straight bore between the two
    /// mouths, plus a lump of geometry that leaves a pocket of trapped air
    /// beside it. Both features matter: the bore is what the flood fill must
    /// keep, the pocket is what it must remove.
    fn carved() -> (Grid, Vec<u8>, Vec<Mouth>, Plenum) {
        // A 20 x 20 x 20 mm part with a 4 x 4 mm bore from z = 0 to y = 0,
        // through a corner elbow, and open air in the rest of the box.
        let scene = Bbox {
            min: Vec3::ZERO,
            max: Vec3::splat(20.0),
        };
        let a = mouth(scene, 2, true, 4.0, 4.0); // on z = 0, centred
        let b = mouth(scene, 1, true, 4.0, 4.0); // on y = 0, centred
        let p = Plenum::default();
        let domain = p
            .plan(scene, &[a.clone(), b.clone()], 0, 1, 1.0, 2)
            .unwrap();
        let grid = Grid::covering(domain.bbox, 1.0);

        // Everything inside the scene box is solid except an L of bore joining
        // the two mouth centres, plus a sealed pocket of air in a far corner.
        let mut mask = vec![flags::FLUID; grid.cell_count() as usize];
        let c = scene.center();
        for z in 0..grid.dims.z {
            for y in 0..grid.dims.y {
                for x in 0..grid.dims.x {
                    let cell = UVec3::new(x, y, z);
                    let pmm = grid.cell_center_mm(cell);
                    if !scene.contains(pmm) {
                        continue;
                    }
                    // The bore: down the z axis at (cx, cy), then out the y axis
                    // at (cx, cz) — an elbow meeting in the middle.
                    let leg_z = (pmm.x - c.x).abs() <= 2.0 && (pmm.y - c.y).abs() <= 2.0;
                    let leg_y = (pmm.x - c.x).abs() <= 2.0 && (pmm.z - c.z).abs() <= 2.0;
                    let bore = (leg_z && pmm.z <= c.z + 2.0) || (leg_y && pmm.y <= c.y + 2.0);
                    // A pocket of air with no way out, in the +x +y +z corner.
                    let pocket = pmm.x > 15.0 && pmm.y > 15.0 && pmm.z > 15.0;
                    if !bore && !pocket {
                        mask[grid.linear(cell) as usize] = flags::SOLID;
                    }
                }
            }
        }
        (grid, mask, vec![a, b], p)
    }

    #[test]
    fn the_carve_keeps_the_passage_and_the_two_extensions_and_nothing_else() {
        let (grid, mut mask, mouths, p) = carved();
        let report = p.carve(&mut mask, grid, &mouths, 0, 1);
        assert!(report.connected, "the flood fill lost the outlet");
        assert!(
            report.fluid_after < report.fluid_before / 4,
            "{}",
            report.describe()
        );

        // The inlet extension is fluid all the way from the domain face to the
        // mouth, and only over the mouth's own footprint.
        let a = &mouths[0];
        let half = footprint_half(a, grid.dx_mm);
        let k_mouth = ((a.patch.center_mm.z - grid.origin_mm.z) / grid.dx_mm).round() as u32;
        for z in 0..k_mouth {
            let mut open = 0;
            for y in 0..grid.dims.y {
                for x in 0..grid.dims.x {
                    let cell = UVec3::new(x, y, z);
                    let pmm = grid.cell_center_mm(cell);
                    let inside = (pmm.x - a.patch.center_mm.x).abs() <= half.x
                        && (pmm.y - a.patch.center_mm.y).abs() <= half.y;
                    let fluid = flags::is_fluid(mask[grid.linear(cell) as usize]);
                    assert!(!(fluid && !inside), "the inlet plenum leaked at {cell:?}");
                    open += usize::from(fluid);
                }
            }
            assert!(open > 0, "the inlet extension is blocked at row z = {z}");
        }

        // The sealed pocket is gone.
        let corner = grid
            .cell_range(Bbox {
                min: Vec3::splat(16.0),
                max: Vec3::splat(19.0),
            })
            .expect("the pocket is inside the domain");
        for z in corner.0.z..=corner.1.z {
            for y in corner.0.y..=corner.1.y {
                for x in corner.0.x..=corner.1.x {
                    let f = mask[grid.linear(UVec3::new(x, y, z)) as usize];
                    assert!(!flags::is_fluid(f), "trapped air survived at ({x},{y},{z})");
                }
            }
        }
    }

    /// The extension is exactly as open as the duct it is bolted to.
    ///
    /// This is the difference between a velocity inlet that delivers the flow it
    /// was asked for and one that delivers 9% too much. The boundary sets `u`,
    /// so `Q` is `u` times *the area the flag ended up covering* — and if the
    /// extension is the mouth's bounding rectangle rather than its opening, that
    /// area is the rectangle's. Measured on the test part before this was fixed:
    /// 3,948 inlet cells against the opening's 3,730, and `Q_in` = 6.68 L/s
    /// against 6.34.
    ///
    /// The fixture makes the point sharply by giving the mouth a rim: the hole
    /// is a 2 mm cross inside a 4 x 4 rectangle, so a rectangular tube would be
    /// more than twice the open area.
    #[test]
    fn the_extension_has_the_mouths_own_open_area_not_its_bounding_rectangle() {
        let scene = Bbox {
            min: Vec3::ZERO,
            max: Vec3::splat(20.0),
        };
        let m = mouth(scene, 2, true, 4.0, 4.0);
        let plan = Plenum::default()
            .plan(scene, &[m.clone()], 0, 0, 1.0, 2)
            .unwrap();
        let grid = Grid::covering(plan.bbox, 1.0);

        // Everything inside the part is solid except a cross-shaped bore.
        let c = scene.center();
        let mut mask = vec![flags::FLUID; grid.cell_count() as usize];
        for z in 0..grid.dims.z {
            for y in 0..grid.dims.y {
                for x in 0..grid.dims.x {
                    let cell = UVec3::new(x, y, z);
                    let p = grid.cell_center_mm(cell);
                    if !scene.contains(p) {
                        continue;
                    }
                    let cross = (p.x - c.x).abs() <= 1.0 || (p.y - c.y).abs() <= 1.0;
                    let in_rect = (p.x - c.x).abs() <= 2.0 && (p.y - c.y).abs() <= 2.0;
                    if !(cross && in_rect) {
                        mask[grid.linear(cell) as usize] = flags::SOLID;
                    }
                }
            }
        }

        let s = section(grid, &mask, &m).expect("the cross-shaped mouth has an opening");
        let plane_open = (0..grid.dims.y)
            .flat_map(|y| (0..grid.dims.x).map(move |x| (x, y)))
            .filter(|(x, y)| s.is_open(UVec3::new(*x, *y, s.plane_row)))
            .count();

        wall_off_plenum(&mut mask, grid, &s);
        let face = plenum_face_cells(grid, &s);
        assert_eq!(
            face.len(),
            plane_open,
            "the extension's far end is not the same cross-section as the mouth"
        );
        // The rectangle would have been 5 x 5 = 25 cells at this dx; the cross
        // inside it is fewer, and that gap is the flow error.
        assert!(
            face.len() < 25,
            "the extension is the bounding rectangle after all"
        );
        assert!(
            face.len() > 8,
            "the extension collapsed to nothing: {}",
            face.len()
        );
        // Every row of the extension has exactly that cross-section, so there is
        // no step for the flow to separate off anywhere along it.
        for z in 0..s.plane_row {
            let open = (0..grid.dims.y)
                .flat_map(|y| (0..grid.dims.x).map(move |x| UVec3::new(x, y, z)))
                .filter(|c| flags::is_fluid(mask[grid.linear(*c) as usize]))
                .count();
            assert_eq!(
                open, plane_open,
                "the extension changes area at row z = {z}"
            );
        }
    }

    /// Both openings must survive the carve, or the run measures a sealed box.
    #[test]
    fn the_inlet_and_outlet_faces_stay_open_after_carving() {
        let (grid, mut mask, mouths, p) = carved();
        let before = mask.clone();
        p.carve(&mut mask, grid, &mouths, 0, 1);
        for (name, m) in [("inlet", &mouths[0]), ("outlet", &mouths[1])] {
            let s = section(grid, &before, m).expect("the fixture mouth has an opening");
            let cells = plenum_face_cells(grid, &s);
            let open = cells
                .iter()
                .filter(|c| flags::is_fluid(mask[grid.linear(**c) as usize]))
                .count();
            assert!(open > 0, "the {name} face was sealed by the carve");
        }
    }

    /// A duct whose two openings are not connected is a broken problem, and the
    /// carve has to say so rather than entomb the outlet and run happily.
    #[test]
    fn a_blocked_duct_abandons_the_fill_rather_than_sealing_the_outlet() {
        let (grid, mut mask, mouths, p) = carved();
        // Plug the elbow.
        let mid = grid
            .cell_range(Bbox {
                min: Vec3::splat(8.0),
                max: Vec3::splat(12.0),
            })
            .unwrap();
        for z in mid.0.z..=mid.1.z {
            for y in mid.0.y..=mid.1.y {
                for x in mid.0.x..=mid.1.x {
                    mask[grid.linear(UVec3::new(x, y, z)) as usize] = flags::SOLID;
                }
            }
        }
        // The plenum walls are still built — they are geometry, derived from the
        // mouths, not a guess about connectivity — but the flood fill on top of
        // them must have changed nothing.
        let mut walls_only = mask.clone();
        wall_off_plenum(
            &mut walls_only,
            grid,
            &section(grid, &mask, &mouths[0]).unwrap(),
        );
        wall_off_plenum(
            &mut walls_only,
            grid,
            &section(grid, &mask, &mouths[1]).unwrap(),
        );

        let report = p.carve(&mut mask, grid, &mouths, 0, 1);
        assert!(
            !report.connected,
            "a plugged duct was reported as connected"
        );
        assert_eq!(
            mask, walls_only,
            "the fill entombed cells after reporting that it had been abandoned"
        );
        // ...and in particular the outlet is still an opening rather than a wall.
        let s = section(grid, &walls_only, &mouths[1]).unwrap();
        let open = plenum_face_cells(grid, &s)
            .into_iter()
            .filter(|c| flags::is_fluid(mask[grid.linear(*c) as usize]))
            .count();
        assert!(
            open > 0,
            "the outlet was sealed by a fill that claimed to have given up"
        );
    }

    /// Open walls leave the outlet chamber open, which is the only difference
    /// between the two settings — the box, the inlet extension and the cell
    /// count are identical.
    #[test]
    fn open_walls_change_the_outlet_plenum_and_nothing_else() {
        let (grid, mut solid, mouths, p) = carved();
        let mut open = solid.clone();
        p.carve(&mut solid, grid, &mouths, 0, 1);
        Plenum {
            walls: PlenumWalls::Open,
            ..p
        }
        .carve(&mut open, grid, &mouths, 0, 1);

        assert!(
            open.iter().filter(|f| flags::is_fluid(**f)).count()
                > solid.iter().filter(|f| flags::is_fluid(**f)).count(),
            "the open chamber has no more fluid in it than the walled tube"
        );
        // The inlet extension is walled either way.
        let k_mouth =
            ((mouths[0].patch.center_mm.z - grid.origin_mm.z) / grid.dx_mm).round() as u32;
        for z in 0..k_mouth {
            for y in 0..grid.dims.y {
                for x in 0..grid.dims.x {
                    let i = grid.linear(UVec3::new(x, y, z)) as usize;
                    assert_eq!(
                        flags::is_fluid(solid[i]),
                        flags::is_fluid(open[i]),
                        "the inlet plenum differs at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    /// The off-by-one that cost 1.4 M cells and would have leaked flow.
    ///
    /// Every row whose centre is outside the mouth's plane has to be walled,
    /// including the one the plane very nearly bisects. A single uncarved row
    /// spans the whole domain, so it joins the extension to the open air beside
    /// the part — and that is a hole in the duct, not a modelling nicety.
    ///
    /// The sweep over sub-cell offsets is the point: the failing case only
    /// appears when the plane lands in the outer half of a cell, which is a
    /// property of the part's dimensions and `dx` together and therefore
    /// something no single fixture can be trusted to hit.
    #[test]
    fn every_row_outside_the_mouth_plane_is_walled_whatever_the_rounding() {
        for offset in [0.0f32, 0.1, 0.25, 0.49, 0.5, 0.51, 0.75, 0.9, 0.99] {
            let scene = Bbox {
                min: Vec3::ZERO,
                max: Vec3::splat(20.0),
            };
            let m = mouth(scene, 2, true, 4.0, 4.0);
            // Shift the grid so the mouth plane sits `offset` of a cell above a
            // row centre, without moving the geometry.
            let dx = 1.0;
            let plan = Plenum::default()
                .plan(scene, &[m.clone()], 0, 0, dx, 2)
                .unwrap();
            let mut grid = Grid::covering(plan.bbox, dx);
            grid.origin_mm.z -= offset * dx;

            // An all-fluid mask, so the extruded cross-section is exactly the
            // footprint rectangle and any discrepancy below is the row
            // arithmetic rather than the geometry.
            let mut mask = vec![flags::FLUID; grid.cell_count() as usize];
            let s = section(grid, &mask, &m).unwrap();
            wall_off_plenum(&mut mask, grid, &s);

            for z in 0..grid.dims.z {
                let zc = grid.cell_center_mm(UVec3::new(0, 0, z)).z;
                if zc >= m.patch.center_mm.z {
                    continue; // the geometry's own rows; not this function's business
                }
                for y in 0..grid.dims.y {
                    for x in 0..grid.dims.x {
                        let cell = UVec3::new(x, y, z);
                        let p = grid.cell_center_mm(cell);
                        let half = footprint_half(&m, dx);
                        let inside = (p.x - m.patch.center_mm.x).abs() <= half.x
                            && (p.y - m.patch.center_mm.y).abs() <= half.y;
                        let fluid = flags::is_fluid(mask[grid.linear(cell) as usize]);
                        assert_eq!(
                            fluid,
                            inside,
                            "offset {offset}: row z = {z} ({zc:.2} mm, outside the plane at \
                             {:.2}) is {} at {cell:?}",
                            m.patch.center_mm.z,
                            if fluid { "open" } else { "walled" }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_inlet_plane_is_the_far_end_of_the_extension_not_the_mouth() {
        let (scene, mouths) = part();
        let p = Plenum::default();
        let (_, grid) = grid_for(&p, scene, &mouths, 0.75);
        // An all-fluid mask stands in for the voxelised part here: what is under
        // test is where the plane lands, not what shape it is.
        let mut mask = vec![flags::FLUID; grid.cell_count() as usize];
        let plane = p
            .carve(&mut mask, grid, &mouths, 0, 1)
            .inlet_plane_mm
            .unwrap();
        let depths = p.depths(&mouths, 0, 1, 0.75, 20).unwrap();

        // Mouth A is on z = 0; the extension runs to z = -upstream.
        assert!(
            plane < 0.0,
            "the inlet plane is not upstream of the mouth: {plane}"
        );
        let gap = mouths[0].patch.center_mm.z - plane;
        assert!(
            (gap - depths.upstream).abs() < grid.dx_mm,
            "{gap} mm of development length, wanted {}",
            depths.upstream
        );
        // ...and it is a real cell row, not a coordinate between two.
        let k = (plane - grid.origin_mm.z) / grid.dx_mm;
        assert!(
            (k - k.round()).abs() < 1e-4,
            "the inlet plane is off-lattice: {k}"
        );

        // Swapped, it moves to the other mouth's face — and to that mouth's own
        // development length, which is a different number because the two
        // mouths have different `D_h`.
        let swapped = p.plan(scene, &mouths, 1, 0, 0.75, 20).unwrap();
        let g = Grid::covering(swapped.bbox, 0.75);
        let mut mask = vec![flags::FLUID; g.cell_count() as usize];
        let plane = p.carve(&mut mask, g, &mouths, 1, 0).inlet_plane_mm.unwrap();
        let rev = p.depths(&mouths, 1, 0, 0.75, 20).unwrap();
        assert!(
            plane < 0.0,
            "the swapped inlet plane is not upstream of mouth B: {plane}"
        );
        let gap = mouths[1].patch.center_mm.y - plane;
        assert!(
            (gap - rev.upstream).abs() < g.dx_mm,
            "{gap} mm, wanted {}",
            rev.upstream
        );
    }

    /// The property the whole domain exists for: after the carve and the
    /// boundary conditions, mass has exactly two doors and no windows.
    ///
    /// An `EQUILIBRIUM` cell overwrites its populations with `f^eq(rho_ref, 0)`
    /// every step, so it is a mass source or sink wherever it is. The room
    /// domain has 381,270 of them and the mass balance is a statement about
    /// what is left over after they have all had their say. The walled plenum
    /// domain must have none at all, or it is just a smaller room.
    #[test]
    fn the_walled_plenum_domain_has_no_equilibrium_cells_anywhere() {
        let (grid, mut mask, mouths, p) = carved();
        let report = p.carve(&mut mask, grid, &mouths, 0, 1);
        crate::sim::apply_boundaries(
            &mut mask,
            grid,
            &mouths,
            0,
            1,
            crate::sim::BoundaryPlan {
                inlet_plane_mm: report.inlet_plane_mm,
                sponge_cells: 4,
            },
            &[],
        );

        let count = |bit: u8| {
            mask.iter()
                .filter(|f| flags::is_fluid(**f) && **f & bit != 0)
                .count()
        };
        assert_eq!(
            count(flags::EQUILIBRIUM),
            0,
            "the plenum domain grew an open face"
        );
        assert!(count(flags::INLET) > 0, "the inlet was carved away");
        assert!(count(flags::OUTLET) > 0, "the outlet was carved away");
        // The sponge is a layer, not a plane: four rows of the outlet
        // extension's cross-section.
        assert_eq!(
            count(flags::SPONGE),
            4 * count(flags::OUTLET),
            "the sponge is not four rows deep"
        );

        // ...and with the sides open, the equilibrium faces come back — which is
        // what makes the assertion above about this domain rather than about
        // `apply_boundaries` refusing to flag anything.
        let (grid, mut mask, mouths, _) = carved();
        let open = Plenum {
            walls: PlenumWalls::Open,
            ..p
        };
        let report = open.carve(&mut mask, grid, &mouths, 0, 1);
        crate::sim::apply_boundaries(
            &mut mask,
            grid,
            &mouths,
            0,
            1,
            crate::sim::BoundaryPlan {
                inlet_plane_mm: report.inlet_plane_mm,
                sponge_cells: 4,
            },
            &[],
        );
        assert!(
            mask.iter()
                .filter(|f| flags::is_fluid(**f) && **f & flags::EQUILIBRIUM != 0)
                .count()
                > 0,
            "the open chamber has no equilibrium faces either"
        );
    }

    #[test]
    fn a_mouth_with_no_axis_face_has_no_plenum_to_build() {
        let (scene, mouths) = part();
        let mut skew = mouths.clone();
        skew[1].patch.normal = Vec3::new(0.6, 0.6, 0.5).normalize();
        assert!(Plenum::default()
            .plan(scene, &skew, 0, 1, 0.75, 20)
            .is_none());
        assert!(Plenum::default().depths(&skew, 0, 1, 0.75, 20).is_none());
        // ...and an empty scene never produces a box out of nothing.
        assert!(Plenum::default()
            .plan(Bbox::EMPTY, &mouths, 0, 1, 0.75, 20)
            .is_none());
    }
}

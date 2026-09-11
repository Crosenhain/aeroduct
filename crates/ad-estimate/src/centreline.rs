//! Turning a voxel mask into the half-dozen numbers a loss network needs.
//!
//! # What has to come out
//!
//! `A(s)`, `D_h(s)`, aspect ratio, developed length, bend angles with their
//! radius ratios, and where the area changes. Nothing else, and nothing
//! spatial: everything downstream of here is one-dimensional.
//!
//! # How the centreline is found, and why not the other way
//!
//! The two standard options are a **distance-transform ridge** and **marched
//! slice centroids**. Neither is used here, for the same reason: both are
//! fragile on exactly the geometry this app sees.
//!
//! A distance-transform ridge is the medial axis, and the medial axis of a
//! *flat slot* is a sheet, not a curve. The test part's passage is 139 x 15 mm
//! at the inlet — an aspect ratio of nine — so its ridge is a two-dimensional
//! set, and any curve extracted from it is an arbitrary choice among many.
//! Worse, the medial axis is famously unstable: a millimetre of surface noise
//! sprouts a spurious branch.
//!
//! Marched slice centroids need a local flow direction to slice perpendicular
//! to, and that direction is what we are trying to find. The march can be made
//! to work, but it fails in exactly the situation that matters — a tight bend,
//! where the slicing plane starts cutting the duct twice.
//!
//! So instead: a **two-sided geodesic coordinate**. Flood the passage from the
//! inlet mouth, then run a Dijkstra over the flooded cells with 26-neighbour
//! chamfer weights - and do it again from the outlet. That gives `T_in(x)` and
//! `T_out(x)`, the distances through the air to each mouth, and the flow
//! coordinate is
//!
//! ```text
//! u(x) = T_in / (T_in + T_out),   0 at the inlet, 1 at the outlet
//! ```
//!
//! One field alone is not enough, and the reason is worth stating because it
//! took a validation bend to find. In a bend the inner wall is a shorter path
//! than the outer wall, so a wavefront launched from the inlet reaches the
//! outlet plane at `r * pi/2` - a different time for every radius. Level sets
//! of `T_in` therefore *tilt* across the section, and near the outlet they cut
//! the duct at an angle instead of across it: the last bands come out with half
//! the true area, the centreline hooks inward, and a 90 degree turn measures
//! 64. Normalising by the total path kills that exactly, because both fields
//! tilt the same way - `T_in ~ r*theta` and `T_out ~ r*(pi/2 - theta)`, so
//! `u = theta / (pi/2)` with the radius cancelled. Level sets of `u` are the
//! true cross-sections.
//!
//! Slice the passage into bands of constant `u` and:
//!
//! * the **centroid** of each band is a centreline point — an average over
//!   thousands of cells, so voxel noise cancels rather than accumulates;
//! * the **volume** of each band divided by the arc length it spans on the
//!   centreline is `A(s)`, which makes `integral A ds = V` hold *exactly*
//!   (the band spans are constructed to partition the centreline). Each cell's
//!   volume is split between the two nearest bands in proportion to where its
//!   `u` falls, rather than dropped whole into one - with three cells to a band
//!   a hard assignment quantises the count to +/- 1 and puts a 20% ripple on
//!   `A(s)`, and since the friction term goes as `V^2 = (Q/A)^2` that ripple
//!   rectifies into a real bias on the total;
//! * the **second moments** of each band about its centroid, projected
//!   perpendicular to the local tangent, give the aspect ratio.
//!
//! This has no free parameter beyond the band thickness, it cannot produce a
//! branch, it works in a bend because the wavefront turns with the duct, and it
//! degrades into a diagnosable failure — the wavefront never reaches the
//! outlet — rather than a plausible wrong answer.
//!
//! # Why the section is fitted as a rectangle
//!
//! `D_h = 4A/P` needs a perimeter, and a voxelised perimeter is a staircase
//! that overestimates by up to `sqrt(3)` on an oblique wall. Instead the
//! section's two in-plane principal second moments are converted to the sides
//! of an **equivalent rectangle**, `a = sqrt(12 lambda + dx^2)` — Sheppard's
//! correction, which makes that exact for a rectangle sampled on a lattice —
//! and then rescaled to match the area, which comes from the cell count and is
//! unbiased. For a genuinely rectangular duct, which is what a printed slot
//! duct is and what every ASHRAE rectangular correlation assumes, this is
//! exact. For a round one it reads `D_h` about 11% low, well inside the band
//! the method carries anyway. The staircase perimeter is computed too, purely
//! as a cross-check, and a large disagreement raises a warning.
//!
//! # Degrading honestly
//!
//! Every failure mode here has a name in [`PassageError`] or a sentence in
//! [`Passage::warnings`]. A duct with no clean passage reports that it has no
//! clean passage; it never returns a confident wrong number.

use crate::Band;
use ad_gpu::{flags, Grid};
use glam::{DVec3, UVec3, Vec3};
use std::collections::{BinaryHeap, VecDeque};

/// One end of the passage: an opening, its plane, and the outline of the hole.
///
/// Deliberately plain data rather than a re-export, so the numerical core has
/// no opinion about how the mouth was found. [`ad_geom::Mouth`] converts into
/// it, and that conversion is the whole integration surface between the two
/// crates.
#[derive(Debug, Clone, PartialEq)]
pub struct MouthSpec {
    /// Centroid of the opening, mm.
    pub center_mm: Vec3,
    /// Unit normal pointing **into** the fluid, matching `FlowPatch`'s
    /// convention for an inlet.
    pub normal: Vec3,
    /// True open area of the hole, mm^2 — not the bounding rectangle.
    pub open_area_mm2: f32,
    /// The hole's boundary loop, in order, in world space.
    pub boundary: Vec<Vec3>,
}

impl From<&ad_geom::Mouth> for MouthSpec {
    fn from(m: &ad_geom::Mouth) -> Self {
        Self {
            center_mm: m.patch.center_mm,
            normal: m.patch.normal,
            open_area_mm2: m.open_area_mm2,
            boundary: m.boundary.clone(),
        }
    }
}

impl MouthSpec {
    /// Which axis the mouth's plane is perpendicular to.
    pub fn axis(&self) -> usize {
        let n = self.normal.abs();
        if n.x >= n.y && n.x >= n.z {
            0
        } else if n.y >= n.z {
            1
        } else {
            2
        }
    }

    /// True when the flow enters in the `+axis` direction, i.e. the mouth caps
    /// the low-coordinate end of the passage.
    pub fn faces_positive(&self) -> bool {
        self.normal[self.axis()] >= 0.0
    }

    /// Coordinate of the mouth plane along its axis, mm. Taken from the
    /// boundary loop where there is one, because the loop lies exactly in the
    /// plane while an area centroid can be pulled off it by a ragged rim.
    pub fn plane_mm(&self) -> f32 {
        let a = self.axis();
        if self.boundary.is_empty() {
            return self.center_mm[a];
        }
        self.boundary.iter().map(|p| p[a]).sum::<f32>() / self.boundary.len() as f32
    }

    /// Perimeter of the opening, mm. Zero when there is no boundary loop.
    pub fn perimeter_mm(&self) -> f64 {
        let n = self.boundary.len();
        if n < 3 {
            return 0.0;
        }
        (0..n).map(|i| (self.boundary[(i + 1) % n] - self.boundary[i]).length() as f64).sum()
    }

    /// `4A/P` of the opening, mm. Falls back to the diameter of an equal-area
    /// circle when the loop is missing, rather than returning zero.
    pub fn hydraulic_diameter_mm(&self) -> f64 {
        let p = self.perimeter_mm();
        let a = (self.open_area_mm2 as f64).max(0.0);
        if p > 1e-9 {
            4.0 * a / p
        } else {
            2.0 * (a / std::f64::consts::PI).sqrt()
        }
    }

    /// Is `p`, projected onto the mouth's plane, inside the hole?
    ///
    /// Crossing-number test on the two in-plane coordinates. Handles a
    /// non-convex opening, which is what the topological mouth detector
    /// routinely produces.
    fn contains_in_plane(&self, p: Vec3) -> bool {
        let a = self.axis();
        let (i, j) = ((a + 1) % 3, (a + 2) % 3);
        let n = self.boundary.len();
        if n < 3 {
            return false;
        }
        let (px, py) = (p[i], p[j]);
        let mut inside = false;
        for k in 0..n {
            let q = self.boundary[k];
            let r = self.boundary[(k + 1) % n];
            let (qx, qy) = (q[i], q[j]);
            let (rx, ry) = (r[i], r[j]);
            if (qy > py) != (ry > py) {
                let t = (py - qy) / (ry - qy);
                if px < qx + t * (rx - qx) {
                    inside = !inside;
                }
            }
        }
        inside
    }
}

/// Knobs for the extraction. The defaults suit the test part and anything else
/// this app is likely to see; they are exposed because a pathological part will
/// need them moved, and failing silently would be worse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassageConfig {
    /// Index into the mouth slice of the opening the air enters through.
    pub inlet: usize,
    /// Index of the opening it leaves through.
    pub outlet: usize,
    /// Minimum band thickness, in cells. Below about three cells the band
    /// volume is quantised badly enough to show as ripple in `A(s)`.
    pub min_band_cells: f64,
    /// Upper bound on the number of stations. 128 is more than a loss network
    /// can use and keeps the per-slider-tick evaluation in the tens of
    /// microseconds.
    pub max_bands: usize,
    /// Gaussian smoothing radius applied to the centreline, in bands. The
    /// centroids are already an average over thousands of cells; this removes
    /// only the residual wobble that would otherwise read as curvature.
    pub smooth_bands: f64,
    /// A curvature run counts as a bend when its radius is below this multiple
    /// of `D_h`. Anything gentler is not a fitting: its excess loss over a
    /// straight duct of the same length is below the method's noise floor.
    pub bend_radius_limit: f64,
    /// ...and when it turns through at least this many degrees.
    pub min_bend_deg: f64,
    /// Fractional area change below which a transition is not a fitting.
    pub min_area_change: f64,
}

impl Default for PassageConfig {
    fn default() -> Self {
        Self {
            inlet: 0,
            outlet: 1,
            min_band_cells: 3.0,
            max_bands: 128,
            smooth_bands: 1.5,
            bend_radius_limit: 10.0,
            min_bend_deg: 15.0,
            min_area_change: 0.08,
        }
    }
}

/// Why an extraction produced nothing usable.
///
/// Each variant is a *diagnosis*, not a generic failure: every one can be put
/// in front of the user as a sentence about their geometry.
#[derive(Debug, Clone, PartialEq)]
pub enum PassageError {
    /// The mask length does not match the grid.
    MaskSize { expected: usize, got: usize },
    /// Fewer than two mouths, or the configured indices are out of range.
    NeedTwoMouths { have: usize },
    /// The inlet and outlet indices are the same.
    SameMouth,
    /// The inlet opening has no fluid cells behind it. Usually the mouth plane
    /// and the voxel grid disagree, or the opening voxelised shut because `dx`
    /// is coarser than the passage.
    InletBlocked,
    /// The flood fill never reached the outlet: the two mouths are not
    /// connected through air.
    Disconnected { filled_cells: usize },
    /// The passage is too short or too thin to say anything about.
    TooSmall { cells: usize, length_mm: f64 },
}

impl std::fmt::Display for PassageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassageError::MaskSize { expected, got } => {
                write!(f, "voxel mask has {got} cells, grid has {expected}")
            }
            PassageError::NeedTwoMouths { have } => write!(
                f,
                "need an inlet and an outlet, found {have} usable mouth(s); place them \
                 by hand, or check that the duct's openings sit on a bounding-box plane"
            ),
            PassageError::SameMouth => write!(f, "the inlet and the outlet are the same mouth"),
            PassageError::InletBlocked => write!(
                f,
                "no air behind the inlet opening: the mouth plane and the voxel grid \
                 disagree, or dx is coarser than the passage"
            ),
            PassageError::Disconnected { filled_cells } => write!(
                f,
                "the inlet and outlet are not connected through air ({filled_cells} cells \
                 reached); the duct is blocked, or dx is too coarse and has welded the \
                 passage shut"
            ),
            PassageError::TooSmall { cells, length_mm } => write!(
                f,
                "the passage is {cells} cells and {length_mm:.1} mm long, too little to \
                 characterise; refine dx"
            ),
        }
    }
}

impl std::error::Error for PassageError {}

/// How much to believe the extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Well resolved and sealed. Correlation error dominates.
    Good,
    /// Something is off — coarse resolution, a ragged section, a mouth that
    /// does not match the mask. Usable, but read the band as a floor.
    Marginal,
    /// A passage was found but the extraction does not believe its own
    /// geometry. Report the number only next to the reason.
    Poor,
}

impl Confidence {
    pub fn label(self) -> &'static str {
        match self {
            Confidence::Good => "good",
            Confidence::Marginal => "marginal",
            Confidence::Poor => "poor",
        }
    }

    /// Extra relative uncertainty the extraction itself contributes to a
    /// velocity-head term. `K` goes as `1/D_h` through friction and as the
    /// square of an area through everything else, so a 5% geometry error is
    /// already a 5-10% error on `K`.
    pub fn geometry_sigma(self) -> f64 {
        match self {
            Confidence::Good => 0.05,
            Confidence::Marginal => 0.15,
            Confidence::Poor => 0.40,
        }
    }
}

/// One station along the passage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Station {
    /// Arc length from the inlet mouth, mm.
    pub s_mm: f64,
    /// Arc length this station is responsible for, mm. The spans partition
    /// `[0, length_mm]` exactly, which is what makes both `integral A ds = V`
    /// and the friction integral consistent.
    pub span_mm: f64,
    /// Centreline point, mm.
    pub point_mm: Vec3,
    /// Unit flow direction.
    pub tangent: Vec3,
    /// Cross-sectional area, mm^2.
    pub area_mm2: f64,
    /// `4A/P` of the equivalent rectangle, mm.
    pub hydraulic_diameter_mm: f64,
    /// Long side of the equivalent rectangle, mm.
    pub width_mm: f64,
    /// Short side, mm.
    pub height_mm: f64,
    /// `width / height`, always `>= 1`.
    pub aspect: f64,
    /// Curvature magnitude, 1/mm.
    pub curvature_per_mm: f64,
    /// Direction of the long side of the section.
    pub major_axis: Vec3,
    /// Direction of the short side.
    pub minor_axis: Vec3,
    /// `4A/P` using the staircase-counted wetted perimeter, mm. A cross-check
    /// only: it carries a staircase bias of up to `sqrt(3)` and must not be
    /// used for a friction calculation.
    pub hydraulic_diameter_faces_mm: f64,
}

/// A turn in the passage, reduced to what a bend correlation asks for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bend {
    pub s_start_mm: f64,
    pub s_end_mm: f64,
    /// Total turn through the run, degrees.
    pub angle_deg: f64,
    /// Centreline radius, mm: the run's arc length over its turn in radians.
    pub radius_mm: f64,
    /// Mean hydraulic diameter over the run, mm.
    pub hydraulic_diameter_mm: f64,
    /// The number a bend correlation is indexed on.
    pub r_over_dh: f64,
    /// Section dimension **in the plane of the turn**, mm. ASHRAE's `W`.
    pub width_mm: f64,
    /// Section dimension perpendicular to the plane of the turn, mm. ASHRAE's
    /// `H`. The ratio of these two is worth about 30% of the bend's loss and
    /// costs nothing to change, which is why they are reported separately.
    pub height_mm: f64,
    /// `H/W`, the argument of the aspect factor.
    pub aspect_hw: f64,
}

/// A monotone area change: a contraction or a diffuser.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transition {
    pub s_start_mm: f64,
    pub s_end_mm: f64,
    pub area_in_mm2: f64,
    pub area_out_mm2: f64,
    /// Included (full) angle of the equivalent cone, degrees. 180 is a step.
    pub included_angle_deg: f64,
    pub is_contraction: bool,
}

impl Transition {
    /// Smaller area over larger. Always in `(0, 1]`.
    pub fn area_ratio(&self) -> f64 {
        let (a, b) = (self.area_in_mm2, self.area_out_mm2);
        if a <= 0.0 || b <= 0.0 {
            return 1.0;
        }
        (a / b).min(b / a)
    }
}

/// Everything the loss network needs, and nothing it does not.
///
/// Produced once per geometry edit; [`crate::network::estimate`] then reads it
/// many times without touching a voxel.
#[derive(Debug, Clone, PartialEq)]
pub struct Passage {
    pub dx_mm: f64,
    /// Inlet to outlet, `s` increasing.
    pub stations: Vec<Station>,
    pub bends: Vec<Bend>,
    pub transitions: Vec<Transition>,
    /// Developed centreline length, mm.
    pub length_mm: f64,
    /// Air volume between the two mouth planes, mm^3.
    pub volume_mm3: f64,
    /// Air the wavefront reached but which lies past the outlet's arrival
    /// time: dead pockets and side branches. A large value means the passage
    /// is not a single channel.
    pub dead_volume_mm3: f64,
    /// True open area of the inlet mouth, mm^2, from its boundary loop rather
    /// than from the voxels — the loop is exact and the voxels are not.
    pub inlet_area_mm2: f64,
    pub outlet_area_mm2: f64,
    /// Hydraulic diameters of the two openings, mm.
    pub inlet_dh_mm: f64,
    pub outlet_dh_mm: f64,
    /// Length-weighted mean hydraulic diameter: the `D_h` a single-number
    /// Reynolds number should be quoted against.
    pub mean_dh_mm: f64,
    /// Smallest section dimension anywhere, in cells. Under about four, the
    /// extraction is guessing.
    pub min_cells_across: f64,
    pub confidence: Confidence,
    pub warnings: Vec<String>,
}

impl Passage {
    /// `A_inlet / A_outlet`. Greater than one for a contracting duct.
    pub fn area_ratio(&self) -> f64 {
        if self.outlet_area_mm2 > 0.0 {
            self.inlet_area_mm2 / self.outlet_area_mm2
        } else {
            1.0
        }
    }

    /// Total turn through every detected bend, degrees. Should be close to the
    /// angle between the two mouth normals; when it is much larger, the
    /// passage wanders and the loss network is summing turns that partly
    /// cancel.
    pub fn total_turn_deg(&self) -> f64 {
        self.bends.iter().map(|b| b.angle_deg).sum()
    }

    /// Where the biggest single area change happens, as a fraction of the
    /// developed length. `None` when the duct has no material transition.
    pub fn contraction_location(&self) -> Option<f64> {
        if self.length_mm <= 0.0 {
            return None;
        }
        let t = self.transitions.iter().min_by(|a, b| {
            a.area_ratio().partial_cmp(&b.area_ratio()).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        Some(0.5 * (t.s_start_mm + t.s_end_mm) / self.length_mm)
    }

    /// Length-weighted mean aspect ratio, for a single-number section shape.
    pub fn mean_aspect(&self) -> f64 {
        self.weighted_mean(|st| st.aspect)
    }

    /// The geometry contribution to the uncertainty, as a band around 1.0.
    /// Multiplying a `K` by this propagates the extraction's own error into
    /// the report.
    pub fn geometry_factor(&self) -> Band {
        Band::relative(1.0, self.confidence.geometry_sigma())
    }

    pub(crate) fn weighted_mean(&self, f: impl Fn(&Station) -> f64) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for st in &self.stations {
            num += f(st) * st.span_mm;
            den += st.span_mm;
        }
        if den > 0.0 {
            num / den
        } else {
            self.stations.first().map(&f).unwrap_or(0.0)
        }
    }

    /// A one-line summary for a log or a status bar.
    pub fn summary(&self) -> String {
        format!(
            "passage: L = {:.1} mm, V = {:.0} mm^3, D_h = {:.2} mm (in {:.2}, out {:.2}), \
             A_in/A_out = {:.2}, {} bend(s) totalling {:.0} deg, {} transition(s), \
             {:.1} cells across at the narrowest, confidence {}",
            self.length_mm,
            self.volume_mm3,
            self.mean_dh_mm,
            self.inlet_dh_mm,
            self.outlet_dh_mm,
            self.area_ratio(),
            self.bends.len(),
            self.total_turn_deg(),
            self.transitions.len(),
            self.min_cells_across,
            self.confidence.label(),
        )
    }
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// Extract the passage from the solver's per-cell flag bytes.
///
/// The convenience entry point for the app: `ad_geom::Voxelizer::read_flags`
/// hands back exactly this, and `flags::is_fluid` is the only bit that
/// matters. Inlet, outlet and sponge markings are irrelevant to a geometric
/// extraction and are deliberately ignored — this crate decides for itself
/// where the flow enters, from the mouth polygons.
pub fn extract_passage_from_flags(
    grid: Grid,
    cell_flags: &[u8],
    mouths: &[MouthSpec],
    cfg: &PassageConfig,
) -> Result<Passage, PassageError> {
    let solid: Vec<bool> = cell_flags.iter().map(|f| !flags::is_fluid(*f)).collect();
    extract_passage(grid, &solid, mouths, cfg)
}

/// Extract the passage from a solid mask.
///
/// `solid` is one `bool` per cell in [`Grid::linear`] order — X fastest — and
/// must be exactly `grid.cell_count()` long. Nothing here touches the GPU: pass
/// whatever mask you have, from `ad_geom::ray_parity_voxelize` on the CPU or
/// from the GPU voxeliser's flags.
///
/// The grid may be larger than the part; the mouth planes act as caps, so the
/// extraction measures the air between them and nothing else.
pub fn extract_passage(
    grid: Grid,
    solid: &[bool],
    mouths: &[MouthSpec],
    cfg: &PassageConfig,
) -> Result<Passage, PassageError> {
    let n_cells = grid.cell_count() as usize;
    if solid.len() != n_cells {
        return Err(PassageError::MaskSize { expected: n_cells, got: solid.len() });
    }
    if mouths.len() < 2 || cfg.inlet >= mouths.len() || cfg.outlet >= mouths.len() {
        return Err(PassageError::NeedTwoMouths { have: mouths.len() });
    }
    if cfg.inlet == cfg.outlet {
        return Err(PassageError::SameMouth);
    }

    let dx = grid.dx_mm as f64;
    let mut warnings = Vec::new();
    let caps = Caps::new(&grid, mouths, cfg, &mut warnings);
    let passable = caps.passable_mask(&grid, solid, mouths);

    let inlet_seeds = caps.slab_cells(cfg.inlet, &grid, &passable);
    if inlet_seeds.is_empty() {
        return Err(PassageError::InletBlocked);
    }
    let outlet_seeds = caps.slab_cells(cfg.outlet, &grid, &passable);

    // 1. Flood the air behind the inlet, 6-connected.
    let fill = flood(&grid, &passable, &caps, &inlet_seeds);
    if !outlet_seeds.iter().any(|&i| fill.inside[i]) {
        return Err(PassageError::Disconnected { filled_cells: fill.cells });
    }
    if fill.leak_contacts > 0 {
        warnings.push(format!(
            "the passage fill reached the grid boundary at {} cells away from a mouth: \
             the duct wall has a hole at this dx, or the mask is not the part you think \
             it is",
            fill.leak_contacts
        ));
    }

    // 2. Geodesic distance from each mouth, 26-connected, over the filled set.
    let t_in = geodesic(&grid, &fill.inside, &inlet_seeds);
    let t_out = geodesic(&grid, &fill.inside, &outlet_seeds);

    // 3. The shortest inlet-to-outlet path through any cell: the natural scale
    //    for both the band width and the dead-pocket cut.
    let mut l_min = f64::INFINITY;
    for i in 0..n_cells {
        if t_in[i] == UNREACHED || t_out[i] == UNREACHED {
            continue;
        }
        let total = (t_in[i] as u64 + t_out[i] as u64) as f64 * dx / QUANT;
        if total < l_min {
            l_min = total;
        }
    }
    if !l_min.is_finite() || l_min <= 2.0 * dx {
        return Err(PassageError::TooSmall {
            cells: fill.cells,
            length_mm: if l_min.is_finite() { l_min } else { 0.0 },
        });
    }

    // 4. Bands of constant flow coordinate.
    let want = (cfg.min_band_cells.max(1.0) * dx).max(l_min / cfg.max_bands.max(4) as f64);
    let n_bands = ((l_min / want).floor() as usize).clamp(4, cfg.max_bands.max(4));
    let ds = l_min / n_bands as f64;
    let (acc, dead_cells) = accumulate(&grid, &t_in, &t_out, l_min, n_bands);

    let populated = acc.iter().filter(|a| a.w > 0.5).count();
    if populated < 4 {
        return Err(PassageError::TooSmall { cells: fill.cells, length_mm: l_min });
    }
    if populated < n_bands {
        warnings.push(format!(
            "{} of {n_bands} wavefront bands were empty; the passage is not a single \
             channel of slowly varying section",
            n_bands - populated
        ));
    }

    // 5. Centroids -> centreline -> stations. The ends are anchored on the
    //    mouth centres, which are the only two exact points in the extraction.
    let inlet = &mouths[cfg.inlet];
    let outlet = &mouths[cfg.outlet];
    let mut core: Vec<DVec3> = Vec::with_capacity(populated);
    let mut kept: Vec<usize> = Vec::with_capacity(populated);
    for (b, a) in acc.iter().enumerate() {
        if a.w > 0.5 {
            core.push(a.sum / a.w);
            kept.push(b);
        }
    }
    // Differentiate the band centroids *alone*, and do it before smoothing.
    //
    // Two separate traps, both found by the validation bend rather than by
    // reading the code:
    //
    // * The mouth centres are exact points on the mouth plane, but they are the
    //   centroid of the opening's *outline* while a band centroid is the
    //   centroid of a *volume* -- and in a bend the second sits further out,
    //   because there is more air at larger radius. Mixing the two puts a
    //   half-millimetre kink at each end of an otherwise smooth curve, and
    //   Menger curvature reads a kink as a very tight radius: that alone turned
    //   a 90 degree bend into 114 degrees. So the anchors set the arc length,
    //   which they get right, and stay out of the curvature, which they do not.
    //
    // * Smoothing displaces the end points of a polyline no matter how the
    //   kernel is continued past them, because the first band's centroid is
    //   genuinely off the interior's spacing -- its splat kernel is clipped by
    //   the mouth plane. Curvature is a second difference, so it amplifies that
    //   displacement enormously: the two end stations came out at r = 15.5 mm
    //   on a 24 mm bend. Taking the curvature from the raw centroids avoids it
    //   entirely, and the noise immunity that smoothing was providing comes
    //   instead from a wider Menger stencil, which is *exact* for a circular
    //   arc at any width. Smoothing then still happens, but only for the
    //   positions and the arc length, where a displaced end point costs a
    //   fraction of a millimetre and nothing else.
    let stencil = (cfg.smooth_bands.round() as usize).max(1);
    let kappa = curvature_vectors(&core, stencil);
    smooth_polyline(&mut core, cfg.smooth_bands);

    let mut centroids: Vec<DVec3> = Vec::with_capacity(core.len() + 2);
    centroids.push(as_dvec3(inlet.center_mm));
    centroids.extend_from_slice(&core);
    centroids.push(as_dvec3(outlet.center_mm));

    let mut s_at = vec![0.0f64; centroids.len()];
    for i in 1..centroids.len() {
        s_at[i] = s_at[i - 1] + (centroids[i] - centroids[i - 1]).length();
    }
    let length_mm = s_at.last().copied().unwrap_or(0.0);
    if length_mm <= 2.0 * dx {
        return Err(PassageError::TooSmall { cells: fill.cells, length_mm });
    }

    let cell_vol = dx * dx * dx;
    let last = centroids.len() - 1;
    let mut stations = Vec::with_capacity(kept.len());
    for (slot, &b) in kept.iter().enumerate() {
        let c = slot + 1; // centroids[0] is the inlet anchor
        let a = &acc[b];
        // The span each band owns on the centreline: half of each neighbouring
        // gap, with the first and last extended to the mouths so the spans
        // partition [0, L] exactly. That is what makes integral A ds = V.
        let lo = if slot == 0 { 0.0 } else { 0.5 * (s_at[c - 1] + s_at[c]) };
        let hi =
            if slot + 1 == kept.len() { s_at[last] } else { 0.5 * (s_at[c] + s_at[c + 1]) };
        let span = (hi - lo).max(0.25 * ds);
        let area = a.w * cell_vol / span;

        let mut tangent = (centroids[c + 1] - centroids[c - 1]).normalize_or_zero();
        if tangent.length_squared() < 0.5 {
            tangent = DVec3::Z;
        }
        let (major, minor, lam1, lam2) = a.principal(tangent);
        // Sheppard's correction: cell centres spanning a width W on a lattice
        // of pitch dx have variance (W^2 - dx^2)/12, so inverting it exactly
        // recovers W for a rectangle and gives dx for a single row of cells.
        let ea = (12.0 * lam1 + dx * dx).sqrt();
        let eb = (12.0 * lam2 + dx * dx).sqrt();
        let aspect = if eb > 0.0 { (ea / eb).max(1.0) } else { 1.0 };
        // Rescale to the measured area: the aspect ratio comes from the shape
        // (robust) and the scale from the cell count (unbiased).
        let height = (area / aspect).max(0.0).sqrt();
        let width = aspect * height;
        let dh = if width + height > 0.0 { 2.0 * width * height / (width + height) } else { 0.0 };
        let perim_faces = a.faces * dx * dx / span;
        let dh_faces = if perim_faces > 0.0 { 4.0 * area / perim_faces } else { 0.0 };

        stations.push(Station {
            s_mm: s_at[c],
            span_mm: span,
            point_mm: as_vec3(centroids[c]),
            tangent: as_vec3(tangent),
            area_mm2: area,
            hydraulic_diameter_mm: dh,
            width_mm: width,
            height_mm: height,
            aspect,
            curvature_per_mm: kappa.get(slot).map(|k| k.length()).unwrap_or(0.0),
            major_axis: as_vec3(major),
            minor_axis: as_vec3(minor),
            hydraulic_diameter_faces_mm: dh_faces,
        });
    }

    // 6. Reduce to fittings.
    let bends = find_bends(&stations, &kappa, cfg);
    let transitions = find_transitions(&stations, cfg);

    // 7. Say how much of this to believe.
    let min_cells = stations.iter().map(|s| s.height_mm / dx).fold(f64::INFINITY, f64::min);
    let mut confidence = Confidence::Good;
    if min_cells < 4.0 {
        confidence = Confidence::Marginal;
        warnings.push(format!(
            "the passage is only {min_cells:.1} cells across at its narrowest; at this \
             dx the extracted D_h is uncertain by roughly {:.0}%",
            100.0 / min_cells.max(1.0)
        ));
    }
    if min_cells < 2.5 {
        confidence = Confidence::Poor;
    }
    if fill.leak_contacts > 0 {
        confidence = confidence.max(Confidence::Poor);
    }
    let dead_volume = dead_cells * cell_vol;
    let volume = (fill.cells as f64 * cell_vol - dead_volume).max(0.0);
    if dead_volume > 0.15 * volume.max(1e-9) {
        confidence = confidence.max(Confidence::Marginal);
        warnings.push(format!(
            "{:.0}% of the air the fill reached lies past the outlet wavefront: the \
             passage has a large dead pocket or a second branch",
            100.0 * dead_volume / volume.max(1e-9)
        ));
    }

    let mut passage = Passage {
        dx_mm: dx,
        stations,
        bends,
        transitions,
        length_mm,
        volume_mm3: volume,
        dead_volume_mm3: dead_volume,
        inlet_area_mm2: inlet.open_area_mm2 as f64,
        outlet_area_mm2: outlet.open_area_mm2 as f64,
        inlet_dh_mm: inlet.hydraulic_diameter_mm(),
        outlet_dh_mm: outlet.hydraulic_diameter_mm(),
        mean_dh_mm: 0.0,
        min_cells_across: min_cells,
        confidence,
        warnings,
    };
    passage.mean_dh_mm = passage.weighted_mean(|s| s.hydraulic_diameter_mm);

    // The staircase cross-check. A big disagreement means the section is not
    // rectangle-like and the fitted D_h should not be trusted.
    let ratio = passage.weighted_mean(|s| {
        if s.hydraulic_diameter_faces_mm > 0.0 {
            s.hydraulic_diameter_mm / s.hydraulic_diameter_faces_mm
        } else {
            1.0
        }
    });
    if !(0.6..=1.7).contains(&ratio) {
        passage.confidence = passage.confidence.max(Confidence::Marginal);
        passage.warnings.push(format!(
            "the fitted-rectangle D_h and the voxel-counted D_h differ by {:.0}%: the \
             section is not rectangle-like",
            100.0 * (ratio - 1.0).abs()
        ));
    }

    Ok(passage)
}

fn as_vec3(v: DVec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

fn as_dvec3(v: Vec3) -> DVec3 {
    DVec3::new(v.x as f64, v.y as f64, v.z as f64)
}

// ---------------------------------------------------------------------------
// The mouth caps
// ---------------------------------------------------------------------------

/// A mouth reduced to a half-space test on the grid.
struct Cap {
    axis: usize,
    /// Cell index along `axis` of the slab the mouth plane falls in.
    slab: i64,
    /// True when the fluid lies at *higher* indices than the slab.
    fluid_above: bool,
    /// Whether cells inside the opening are passable. False for a third mouth
    /// that is neither the inlet nor the outlet: it gets sealed, so the
    /// extraction measures one channel rather than a manifold.
    open: bool,
    index: usize,
}

struct Caps {
    caps: Vec<Cap>,
}

impl Caps {
    fn new(
        grid: &Grid,
        mouths: &[MouthSpec],
        cfg: &PassageConfig,
        warnings: &mut Vec<String>,
    ) -> Self {
        let dx = grid.dx_mm as f64;
        let caps = mouths
            .iter()
            .enumerate()
            .map(|(index, m)| {
                let axis = m.axis();
                let plane = m.plane_mm() as f64;
                let origin = grid.origin_mm[axis] as f64;
                let dims = grid.dims[axis] as i64;
                // The mouth plane sits on a cell *face*, half a cell below the
                // centre of the first fluid cell, so the slab is the cell whose
                // centre is nearest half a cell downstream of the plane.
                let raw = (plane - origin) / dx;
                let slab = if raw.is_finite() {
                    (raw.round() as i64).clamp(0, dims - 1)
                } else {
                    0
                };
                let open = index == cfg.inlet || index == cfg.outlet;
                if !open {
                    warnings.push(format!(
                        "mouth {index} is neither the inlet nor the outlet and has been \
                         sealed; the estimate covers one channel only"
                    ));
                }
                Cap { axis, slab, fluid_above: m.faces_positive(), open, index }
            })
            .collect();
        Self { caps }
    }

    /// Which cells the fill is allowed into: fluid, between the mouth planes,
    /// and — on a mouth plane itself — inside the opening.
    ///
    /// Precomputed in one pass rather than tested per neighbour, because the
    /// point-in-polygon test is the expensive part and only the two mouth slabs
    /// need it.
    fn passable_mask(&self, grid: &Grid, solid: &[bool], mouths: &[MouthSpec]) -> Vec<bool> {
        let (nx, ny, nz) = (grid.dims.x as usize, grid.dims.y as usize, grid.dims.z as usize);
        let mut out = vec![false; solid.len()];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = (z * ny + y) * nx + x;
                    if solid[i] {
                        continue;
                    }
                    let c = [x as i64, y as i64, z as i64];
                    let mut blocked = false;
                    let mut opened = false;
                    let mut on_a_slab = false;
                    for cap in &self.caps {
                        let k = c[cap.axis];
                        if k == cap.slab {
                            on_a_slab = true;
                        } else if (cap.fluid_above && k < cap.slab)
                            || (!cap.fluid_above && k > cap.slab)
                        {
                            blocked = true;
                        }
                    }
                    if on_a_slab {
                        let centre =
                            grid.cell_center_mm(UVec3::new(x as u32, y as u32, z as u32));
                        for cap in &self.caps {
                            if c[cap.axis] != cap.slab {
                                continue;
                            }
                            let inside = cap.open
                                && mouths
                                    .get(cap.index)
                                    .is_some_and(|m| m.contains_in_plane(centre));
                            if inside {
                                opened = true;
                            } else {
                                blocked = true;
                            }
                        }
                    }
                    out[i] = !blocked || opened;
                }
            }
        }
        out
    }

    /// The passable cells of one mouth's slab: where a fill starts and where a
    /// wavefront's arrival is read off.
    fn slab_cells(&self, index: usize, grid: &Grid, passable: &[bool]) -> Vec<usize> {
        let Some(cap) = self.caps.iter().find(|c| c.index == index) else { return Vec::new() };
        let dims = [grid.dims.x as i64, grid.dims.y as i64, grid.dims.z as i64];
        let (i, j) = ((cap.axis + 1) % 3, (cap.axis + 2) % 3);
        let mut out = Vec::new();
        let mut c = [0i64; 3];
        c[cap.axis] = cap.slab;
        for u in 0..dims[i] {
            for v in 0..dims[j] {
                c[i] = u;
                c[j] = v;
                let idx = (((c[2] * dims[1]) + c[1]) * dims[0] + c[0]) as usize;
                if passable.get(idx).copied().unwrap_or(false) {
                    out.push(idx);
                }
            }
        }
        out
    }

    /// Is this cell on a mouth slab? Used only to decide whether touching the
    /// grid boundary counts as a leak.
    fn on_any_slab(&self, c: [i64; 3]) -> bool {
        self.caps.iter().any(|cap| c[cap.axis] == cap.slab)
    }
}

// ---------------------------------------------------------------------------
// Flood fill and geodesic distance
// ---------------------------------------------------------------------------

const NEIGHBOURS6: [(i64, i64, i64); 6] =
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];

struct Fill {
    inside: Vec<bool>,
    cells: usize,
    leak_contacts: usize,
}

/// Flood the air behind the inlet opening, **6-connected**.
///
/// Six and not twenty-six on purpose: a 26-connected fill slips diagonally
/// through a wall that is one cell thick on the diagonal, and a 2 mm duct wall
/// at a coarse `dx` is exactly that kind of feature. Six-connectivity cannot
/// cross a wall at all. The cost is that a passage which is *only* diagonally
/// connected reads as blocked — but such a passage is one voxel wide, which is
/// a resolution problem the caller needs to be told about anyway, and
/// [`PassageError::Disconnected`] tells them.
///
/// A filled cell that reaches the grid boundary anywhere other than on a mouth
/// slab means the fill escaped the duct. That is counted rather than fixed: the
/// number ends up in a warning, because the right response is to refine `dx` or
/// repair the mesh, not to paper over it here.
fn flood(grid: &Grid, passable: &[bool], caps: &Caps, seeds: &[usize]) -> Fill {
    let (nx, ny, nz) = (grid.dims.x as i64, grid.dims.y as i64, grid.dims.z as i64);
    let mut inside = vec![false; passable.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &s in seeds {
        if passable[s] && !inside[s] {
            inside[s] = true;
            queue.push_back(s);
        }
    }
    let mut cells = 0usize;
    let mut leak_contacts = 0usize;
    while let Some(i) = queue.pop_front() {
        cells += 1;
        let x = (i as i64) % nx;
        let y = ((i as i64) / nx) % ny;
        let z = (i as i64) / (nx * ny);
        let on_edge = x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1;
        if on_edge && !caps.on_any_slab([x, y, z]) {
            leak_contacts += 1;
        }
        for (ox, oy, oz) in NEIGHBOURS6 {
            let (jx, jy, jz) = (x + ox, y + oy, z + oz);
            if jx < 0 || jy < 0 || jz < 0 || jx >= nx || jy >= ny || jz >= nz {
                continue;
            }
            let j = ((jz * ny + jy) * nx + jx) as usize;
            if passable[j] && !inside[j] {
                inside[j] = true;
                queue.push_back(j);
            }
        }
    }
    Fill { inside, cells, leak_contacts }
}

const UNREACHED: u32 = u32::MAX;

/// A cell whose shortest inlet-to-outlet path runs more than this multiple of
/// the shortest path anywhere is a dead pocket rather than part of the channel,
/// and is left out of `A(s)` - though not out of the reported volume, because
/// the air is really there.
///
/// Generous on purpose. In a bend of `r/W = 1` the outer wall is already 40%
/// longer than the inner one, so a tight threshold would write off the outside
/// of every bend as dead - exactly backwards, since that is where the flow is.
const DEAD_PATH_FACTOR: f64 = 2.5;

/// Distances are carried as integers in units of `dx / 256`, so a binary heap
/// can order them without a float wrapper and with no chance of a NaN
/// comparison making the priority queue incoherent. The quantisation error is
/// under 0.4% of a cell — an order of magnitude below the chamfer error that
/// 26-neighbour Dijkstra carries anyway.
const QUANT: f64 = 256.0;

fn geodesic(grid: &Grid, inside: &[bool], seeds: &[usize]) -> Vec<u32> {
    let (nx, ny, nz) = (grid.dims.x as i64, grid.dims.y as i64, grid.dims.z as i64);
    let mut dist = vec![UNREACHED; inside.len()];
    let mut heap: BinaryHeap<std::cmp::Reverse<(u32, u32)>> = BinaryHeap::new();
    for &s in seeds {
        if inside.get(s).copied().unwrap_or(false) && dist[s] != 0 {
            dist[s] = 0;
            heap.push(std::cmp::Reverse((0, s as u32)));
        }
    }
    // 26-neighbour chamfer weights: 1, sqrt(2), sqrt(3) cells.
    let mut offsets: Vec<(i64, i64, i64, u32)> = Vec::with_capacity(26);
    for dz in -1i64..=1 {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let len = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                offsets.push((dx, dy, dz, (len * QUANT).round() as u32));
            }
        }
    }

    while let Some(std::cmp::Reverse((d, i))) = heap.pop() {
        let i = i as usize;
        if d > dist[i] {
            continue;
        }
        let x = (i as i64) % nx;
        let y = ((i as i64) / nx) % ny;
        let z = (i as i64) / (nx * ny);
        for &(ox, oy, oz, w) in &offsets {
            let (jx, jy, jz) = (x + ox, y + oy, z + oz);
            if jx < 0 || jy < 0 || jz < 0 || jx >= nx || jy >= ny || jz >= nz {
                continue;
            }
            let j = ((jz * ny + jy) * nx + jx) as usize;
            if !inside[j] {
                continue;
            }
            let nd = d.saturating_add(w);
            if nd < dist[j] {
                dist[j] = nd;
                heap.push(std::cmp::Reverse((nd, j as u32)));
            }
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// Band accumulation and section fitting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct BandAcc {
    /// Cell-equivalents in this band. Fractional, because a cell's volume is
    /// split between the two bands its flow coordinate falls between.
    w: f64,
    sum: DVec3,
    /// Weighted sums of `x^2, y^2, z^2`.
    sum_sq: DVec3,
    /// Weighted sums of `xy, yz, zx`.
    sum_cross: DVec3,
    /// Cell faces onto non-passage: the wetted perimeter, before the staircase
    /// bias is worried about.
    faces: f64,
}

/// Bin every passage cell by its flow coordinate, splitting each cell's volume
/// between the two nearest band centres.
///
/// Returns the bands and the cell-equivalents written off as dead pockets. The
/// linear split is a tent filter of width `2/n_bands` in `u`; by Poisson
/// summation its residual ripple against a lattice of `m` cells per band falls
/// as `1/(pi m)^2`, so at the default three cells per band the quantisation
/// error in `A(s)` is about 1% rather than the 20% a hard assignment gives.
fn accumulate(
    grid: &Grid,
    t_in: &[u32],
    t_out: &[u32],
    l_min: f64,
    n_bands: usize,
) -> (Vec<BandAcc>, f64) {
    let dx = grid.dx_mm as f64;
    let (nx, ny, nz) = (grid.dims.x as usize, grid.dims.y as usize, grid.dims.z as usize);
    let mut acc = vec![BandAcc::default(); n_bands];
    let mut dead = 0.0f64;
    let last = n_bands as i64 - 1;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = (z * ny + y) * nx + x;
                if t_in[i] == UNREACHED || t_out[i] == UNREACHED {
                    continue;
                }
                let total = (t_in[i] as u64 + t_out[i] as u64) as f64;
                if total <= 0.0 {
                    continue;
                }
                if total * dx / QUANT > DEAD_PATH_FACTOR * l_min {
                    dead += 1.0;
                    continue;
                }
                let u = (t_in[i] as f64 / total).max(0.0).min(1.0);
                // Band centres sit at `(b + 1/2) / n_bands`, so a cell at `u`
                // straddles bands `floor(g)` and `floor(g) + 1`.
                let g = u * n_bands as f64 - 0.5;
                let b0 = g.floor();
                let frac = g - b0;
                let lo = (b0 as i64).clamp(0, last) as usize;
                let hi = (b0 as i64 + 1).clamp(0, last) as usize;

                let p = DVec3::new(
                    grid.origin_mm.x as f64 + x as f64 * dx,
                    grid.origin_mm.y as f64 + y as f64 * dx,
                    grid.origin_mm.z as f64 + z as f64 * dx,
                );
                // Faces onto non-passage: the wetted perimeter. Band-to-band
                // faces drop out automatically, because the neighbour is still
                // in the passage, so only walls count.
                let mut faces = 0.0f64;
                for (ox, oy, oz) in NEIGHBOURS6 {
                    let (jx, jy, jz) = (x as i64 + ox, y as i64 + oy, z as i64 + oz);
                    let outside = jx < 0
                        || jy < 0
                        || jz < 0
                        || jx >= nx as i64
                        || jy >= ny as i64
                        || jz >= nz as i64;
                    if outside
                        || t_in[(jz as usize * ny + jy as usize) * nx + jx as usize] == UNREACHED
                    {
                        faces += 1.0;
                    }
                }

                for (b, w) in [(lo, 1.0 - frac), (hi, frac)] {
                    if w <= 0.0 {
                        continue;
                    }
                    let a = &mut acc[b];
                    a.w += w;
                    a.sum += p * w;
                    a.sum_sq += p * p * w;
                    a.sum_cross += DVec3::new(p.x * p.y, p.y * p.z, p.z * p.x) * w;
                    a.faces += faces * w;
                }
            }
        }
    }
    (acc, dead)
}

impl BandAcc {
    /// Principal in-plane axes of the section and their variances, largest
    /// first.
    ///
    /// The full 3x3 covariance is projected onto the two directions
    /// perpendicular to the flow, which reduces the eigenproblem to a 2x2 —
    /// solvable in closed form, with no iteration and so no chance of failing
    /// to converge on a degenerate section.
    fn principal(&self, tangent: DVec3) -> (DVec3, DVec3, f64, f64) {
        let (u, v) = orthonormal_pair(tangent);
        if self.w < 2.0 {
            return (u, v, 0.0, 0.0);
        }
        let n = self.w;
        let mean = self.sum / n;
        let cxx = self.sum_sq.x / n - mean.x * mean.x;
        let cyy = self.sum_sq.y / n - mean.y * mean.y;
        let czz = self.sum_sq.z / n - mean.z * mean.z;
        let cxy = self.sum_cross.x / n - mean.x * mean.y;
        let cyz = self.sum_cross.y / n - mean.y * mean.z;
        let czx = self.sum_cross.z / n - mean.z * mean.x;
        let cov = |a: DVec3, b: DVec3| {
            a.x * b.x * cxx
                + a.y * b.y * cyy
                + a.z * b.z * czz
                + (a.x * b.y + a.y * b.x) * cxy
                + (a.y * b.z + a.z * b.y) * cyz
                + (a.z * b.x + a.x * b.z) * czx
        };
        let (cuu, cuv, cvv) = (cov(u, u), cov(u, v), cov(v, v));
        let tr = cuu + cvv;
        let det = cuu * cvv - cuv * cuv;
        let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
        let l1 = (0.5 * (tr + disc)).max(0.0);
        let l2 = (0.5 * (tr - disc)).max(0.0);
        // Eigenvector of the 2x2 for l1; both closed forms degenerate when the
        // section is isotropic, so fall back to the basis itself there.
        let (e1u, e1v) = if cuv.abs() > 1e-12 {
            (cuv, l1 - cuu)
        } else if cuu >= cvv {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };
        let mut major = (u * e1u + v * e1v).normalize_or_zero();
        if major.length_squared() < 0.5 {
            major = u;
        }
        let mut minor = tangent.cross(major).normalize_or_zero();
        if minor.length_squared() < 0.5 {
            minor = v;
        }
        (major, minor, l1, l2)
    }
}

/// Two unit vectors completing `t` into an orthonormal frame.
fn orthonormal_pair(t: DVec3) -> (DVec3, DVec3) {
    let helper = if t.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let mut u = t.cross(helper).normalize_or_zero();
    if u.length_squared() < 0.5 {
        u = DVec3::X;
    }
    let mut v = t.cross(u).normalize_or_zero();
    if v.length_squared() < 0.5 {
        v = DVec3::Y;
    }
    (u, v)
}

/// Second-order Newton extrapolation through `a, b, c` at parameter `t`, where
/// `t = 0` is `a`, `t = 1` is `b` and `t = -1` is one step *before* `a`.
///
/// Exact for anything quadratic in the sampling parameter, which is what makes
/// it the right way to continue a curved polyline past its end.
fn extrapolate(a: DVec3, b: DVec3, c: DVec3, t: f64) -> DVec3 {
    a + (b - a) * t + (c - b * 2.0 + a) * (t * (t - 1.0) * 0.5)
}

/// Gaussian smoothing along a polyline, continued **quadratically** past both
/// ends.
///
/// How the ends are handled turned out to matter twice, and both failures were
/// found by the validation bend rather than by inspection:
///
/// * Truncating the kernel and renormalising — the obvious implementation —
///   biases every end point toward the interior, because the surviving half of
///   the window is all on one side. On a *straight* duct that pushed the first
///   station 0.8 mm downstream and shrank its span, which read as a 25% area
///   error on a duct of constant section.
/// * Continuing linearly (`p[-1] = 2 p[0] - p[1]`) fixes that, and is exact for
///   a straight line — but a bend is not a straight line, and extending an arc
///   along its tangent flattens the first two points. That halved the measured
///   curvature at each end of a 90 degree bend and reported it as 66 degrees.
///
/// A quadratic continuation is exact for a circular arc to third order in the
/// step, so it reproduces both a line and a bend. The smoothing radius is also
/// capped at a sixth of the polyline, because a kernel spanning most of a short
/// centreline is not removing noise, it is removing the duct.
fn smooth_polyline(points: &mut [DVec3], radius_bands: f64) {
    let n = points.len();
    if n < 5 || !(radius_bands > 0.0) {
        return;
    }
    let sigma = radius_bands.min(n as f64 / 6.0).max(1e-3);
    let r = (sigma * 2.0).ceil() as isize;
    let src = points.to_vec();
    let last = n as isize - 1;
    let at = |j: isize| -> DVec3 {
        if j < 0 {
            extrapolate(src[0], src[1], src[2], j as f64)
        } else if j > last {
            extrapolate(src[n - 1], src[n - 2], src[n - 3], -((j - last) as f64))
        } else {
            src[j as usize]
        }
    };
    for (i, out) in points.iter_mut().enumerate() {
        let mut num = DVec3::ZERO;
        let mut den = 0.0;
        for k in -r..=r {
            let w = (-(k as f64 * k as f64) / (2.0 * sigma * sigma)).exp();
            num += at(i as isize + k) * w;
            den += w;
        }
        if den > 0.0 {
            *out = num / den;
        }
    }
}

/// The **curvature vector** at every vertex: magnitude `1/R` from the Menger
/// circle through three points spaced `stencil` apart, direction along the
/// binormal of that circle.
///
/// A vector and not a scalar, and that is the point. Menger's scalar form is
/// `2 |(b-a) x (c-a)| / (|ab||bc||ca|)`, and the absolute value **rectifies
/// noise**: a wobble that should turn the centreline left as often as right
/// instead adds to the curvature every time. Integrated along a bend that bias
/// does not cancel — it reported a 24 mm quarter bend as 104 degrees rather
/// than 90, and got worse the gentler the bend, because the bias is fixed while
/// the signal is not. Keeping the cross product as a vector lets
/// [`find_bends`] project it onto the bend's own plane, where the two signs
/// cancel as they should.
///
/// Menger rather than a finite-difference second derivative because it is
/// **exact for a circular arc at any sampling density** — and a circular arc is
/// precisely the shape a bend is being fitted to.
///
/// The stencil width is doing real work too. Curvature from three adjacent
/// points amplifies a perpendicular wobble `d` by `4d/h^2`, so at the default
/// band spacing a fiftieth of a millimetre of centroid noise is worth a quarter
/// of the curvature of a 24 mm bend. Widening the stencil to `m h` divides that
/// by `m^2` while leaving a genuine arc's answer untouched, which is exactly
/// the trade a smoother cannot make.
///
/// The guard band at each end matters as much. A mouth plane is a Dirichlet
/// boundary for both distance fields, and within a couple of bands of it the
/// level sets of `u` are not yet parallel to the true cross-section — on the
/// validation bend the band centroids drift inward by 0.3 mm over the first
/// three stations. That is a 1% radial error and utterly harmless to `A(s)`,
/// but curvature is a second derivative of position: differentiated twice it
/// turned `1/24 mm` into `1/17 mm` and inflated a 90 degree turn to 101. So
/// curvature is computed only where a full stencil of undistorted stations is
/// available, and the nearest good value is carried out to each end. Carrying
/// the value out rather than zeroing it is deliberate — the duct is still
/// turning there, and declaring the ends straight would shave a slice off every
/// bend, always in the direction that flatters the design.
fn curvature_vectors(points: &[DVec3], stencil: usize) -> Vec<DVec3> {
    let n = points.len();
    let mut out = vec![DVec3::ZERO; n];
    if n < 3 {
        return out;
    }
    let m = stencil.max(1).min((n - 1) / 2);
    for i in 0..n {
        let lo = i.saturating_sub(m);
        let hi = (i + m).min(n - 1);
        if hi - lo < 2 {
            continue;
        }
        let mid = i.clamp(lo + 1, hi - 1);
        let (a, b, c) = (points[lo], points[mid], points[hi]);
        let ab = (b - a).length();
        let bc = (c - b).length();
        let ca = (a - c).length();
        if ab * bc * ca < 1e-12 {
            continue;
        }
        out[i] = (b - a).cross(c - a) * (2.0 / (ab * bc * ca));
    }

    // Carry the first and last fully-supported values out over the guard band.
    let guard = (m + 1).min((n - 1) / 2);
    let (first, last) = (out[guard], out[n - 1 - guard]);
    for o in out.iter_mut().take(guard) {
        *o = first;
    }
    for o in out.iter_mut().skip(n - guard) {
        *o = last;
    }

    // A running mean over the same stencil. Constant on a true arc, so it
    // changes nothing there; on noise it averages the two signs toward zero.
    let src = out.clone();
    for (i, o) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(m);
        let hi = (i + m).min(n - 1);
        let mut acc = DVec3::ZERO;
        for v in &src[lo..=hi] {
            acc += *v;
        }
        *o = acc / (hi - lo + 1) as f64;
    }
    out
}

/// Segment the centreline into bends and reduce each to what a correlation
/// wants.
///
/// A run starts where the local radius drops below `bend_radius_limit * D_h`
/// and continues while it stays under twice that. The hysteresis is not
/// decoration: without it a single noisy station in the middle of a gentle
/// 90 degree bend splits it into three separate fittings, and three 30 degree
/// bends cost noticeably less than one 90 degree bend — `A1(30) = 0.45` each
/// against `A1(90) = 1.0` — so the fragmentation would quietly under-report the
/// loss.
fn find_bends(stations: &[Station], kappa: &[DVec3], cfg: &PassageConfig) -> Vec<Bend> {
    let mut bends = Vec::new();
    if stations.len() < 3 || kappa.len() != stations.len() {
        return bends;
    }
    let threshold = |st: &Station| {
        1.0 / (cfg.bend_radius_limit.max(0.1) * st.hydraulic_diameter_mm.max(1e-6))
    };

    let mut i = 0;
    while i < stations.len() {
        if stations[i].curvature_per_mm <= threshold(&stations[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i + 1 < stations.len()
            && stations[i + 1].curvature_per_mm > 0.5 * threshold(&stations[i + 1])
        {
            i += 1;
        }
        let end = i;
        i += 1;

        // The plane the run turns in: the arc-weighted mean curvature vector.
        // It is also the bend's binormal, so projecting each station's
        // curvature onto it gives a *signed* turn -- noise that curls the other
        // way subtracts instead of adding.
        let mut n_b = DVec3::ZERO;
        for (st, k) in stations[start..=end].iter().zip(&kappa[start..=end]) {
            n_b += *k * st.span_mm;
        }
        let n_b = n_b.normalize_or_zero();
        if n_b.length_squared() < 0.5 {
            continue;
        }

        let mut turn_rad = 0.0;
        let mut dh_sum = 0.0;
        let mut arc = 0.0;
        for (st, k) in stations[start..=end].iter().zip(&kappa[start..=end]) {
            turn_rad += k.dot(n_b) * st.span_mm;
            dh_sum += st.hydraulic_diameter_mm * st.span_mm;
            arc += st.span_mm;
        }
        let angle_deg = turn_rad.to_degrees();
        if angle_deg < cfg.min_bend_deg || arc <= 0.0 {
            continue;
        }
        let dh = dh_sum / arc;
        let radius = if turn_rad > 1e-9 { arc / turn_rad } else { f64::INFINITY };

        // Which section dimension lies in the plane of the turn? That is the
        // `W` a bend correlation is indexed on, and getting it the wrong way
        // round moves the answer by 30% in the direction that flatters the
        // design.
        let mid = &stations[(start + end) / 2];
        let t = as_dvec3(mid.tangent);
        let in_plane = t.cross(n_b).normalize_or_zero();
        let major = as_dvec3(mid.major_axis);
        let (width, height) = if major.dot(in_plane).abs() >= 0.5 {
            (mid.width_mm, mid.height_mm)
        } else {
            (mid.height_mm, mid.width_mm)
        };

        bends.push(Bend {
            s_start_mm: stations[start].s_mm,
            s_end_mm: stations[end].s_mm,
            angle_deg,
            radius_mm: radius,
            hydraulic_diameter_mm: dh,
            r_over_dh: if dh > 0.0 { radius / dh } else { f64::INFINITY },
            width_mm: width,
            height_mm: height,
            aspect_hw: if width > 0.0 { height / width } else { 1.0 },
        });
    }
    bends
}

fn find_transitions(stations: &[Station], cfg: &PassageConfig) -> Vec<Transition> {
    let mut out = Vec::new();
    let n = stations.len();
    if n < 3 {
        return out;
    }
    // Segment on a smoothed area profile, or voxel ripple becomes a dozen
    // imaginary fittings.
    let mut area: Vec<f64> = stations.iter().map(|s| s.area_mm2).collect();
    smooth_scalar(&mut area, 2.0);

    let mut start = 0usize;
    let mut dir = 0i32;
    for i in 1..n {
        let d = match area[i].partial_cmp(&area[i - 1]) {
            Some(std::cmp::Ordering::Greater) => 1,
            Some(std::cmp::Ordering::Less) => -1,
            _ => 0,
        };
        if d == 0 {
            continue;
        }
        if dir == 0 {
            dir = d;
            continue;
        }
        if d != dir {
            push_transition(&mut out, stations, &area, start, i - 1, cfg);
            start = i - 1;
            dir = d;
        }
    }
    push_transition(&mut out, stations, &area, start, n - 1, cfg);
    out
}

fn push_transition(
    out: &mut Vec<Transition>,
    stations: &[Station],
    area: &[f64],
    lo: usize,
    hi: usize,
    cfg: &PassageConfig,
) {
    if hi <= lo {
        return;
    }
    let (a1, a2) = (area[lo], area[hi]);
    if a1 <= 0.0 || a2 <= 0.0 {
        return;
    }
    if (a2 - a1).abs() / a1.max(a2) < cfg.min_area_change {
        return;
    }
    let ds = (stations[hi].s_mm - stations[lo].s_mm).max(1e-6);
    // Included angle of the equivalent cone, from the change in equal-area
    // diameter over the run. `atan` of a ratio, so a step (ds -> 0) saturates
    // at 180 degrees rather than overflowing.
    let d1 = 2.0 * (a1 / std::f64::consts::PI).sqrt();
    let d2 = 2.0 * (a2 / std::f64::consts::PI).sqrt();
    let included = 2.0 * ((d2 - d1).abs() / (2.0 * ds)).atan().to_degrees();
    out.push(Transition {
        s_start_mm: stations[lo].s_mm,
        s_end_mm: stations[hi].s_mm,
        area_in_mm2: a1,
        area_out_mm2: a2,
        included_angle_deg: included.min(180.0),
        is_contraction: a2 < a1,
    });
}

fn smooth_scalar(v: &mut [f64], sigma: f64) {
    let n = v.len();
    if n < 3 || !(sigma > 0.0) {
        return;
    }
    let r = (sigma * 2.0).ceil() as isize;
    let src = v.to_vec();
    for (i, out) in v.iter_mut().enumerate() {
        let mut num = 0.0;
        let mut den = 0.0;
        for k in -r..=r {
            let j = i as isize + k;
            if j < 0 || j >= n as isize {
                continue;
            }
            let w = (-(k as f64 * k as f64) / (2.0 * sigma * sigma)).exp();
            num += src[j as usize] * w;
            den += w;
        }
        if den > 0.0 {
            *out = num / den;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` points spread over `sweep` radians of a circle of radius `r`, in the
    /// XY plane, going anticlockwise.
    fn arc(r: f64, sweep: f64, n: usize) -> Vec<DVec3> {
        (0..n)
            .map(|i| {
                let t = sweep * i as f64 / (n - 1) as f64;
                DVec3::new(r * t.cos(), r * t.sin(), 0.0)
            })
            .collect()
    }

    fn rect_mouth(centre: Vec3, normal: Vec3, u: Vec3, v: Vec3, hu: f32, hv: f32) -> MouthSpec {
        MouthSpec {
            center_mm: centre,
            normal,
            open_area_mm2: 4.0 * hu * hv,
            boundary: vec![
                centre - u * hu - v * hv,
                centre + u * hu - v * hv,
                centre + u * hu + v * hv,
                centre - u * hu + v * hv,
            ],
        }
    }

    #[test]
    fn a_mouth_reports_its_own_plane_orientation_and_hydraulic_diameter() {
        // A 30 x 10 opening on the z = 4 plane, air entering along +Z.
        let m = rect_mouth(Vec3::new(1.0, 2.0, 4.0), Vec3::Z, Vec3::X, Vec3::Y, 15.0, 5.0);
        assert_eq!(m.axis(), 2);
        assert!(m.faces_positive(), "+Z means the fluid is above the plane");
        assert!((m.plane_mm() - 4.0).abs() < 1e-6);
        assert!((m.perimeter_mm() - 80.0).abs() < 1e-6);
        // 4A/P for a 30 x 10 rectangle is 15.
        assert!((m.hydraulic_diameter_mm() - 15.0).abs() < 1e-6);

        let back = rect_mouth(Vec3::ZERO, -Vec3::Y, Vec3::Z, Vec3::X, 3.0, 3.0);
        assert_eq!(back.axis(), 1);
        assert!(!back.faces_positive());

        // No boundary loop: fall back to an equal-area circle rather than
        // dividing by a zero perimeter.
        let bare = MouthSpec {
            center_mm: Vec3::ZERO,
            normal: Vec3::X,
            open_area_mm2: std::f32::consts::PI * 25.0,
            boundary: Vec::new(),
        };
        assert_eq!(bare.axis(), 0);
        assert!((bare.hydraulic_diameter_mm() - 10.0).abs() < 1e-3);
        assert!(bare.perimeter_mm() == 0.0);
    }

    #[test]
    fn point_in_plane_handles_a_non_convex_opening() {
        // An L, which is what the topological mouth detector produces from a
        // rim that is not a simple rectangle. A convex test would wrongly
        // include the notch, and the flood fill would start outside the duct.
        let m = MouthSpec {
            center_mm: Vec3::ZERO,
            normal: Vec3::Z,
            open_area_mm2: 30.0,
            boundary: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(6.0, 0.0, 0.0),
                Vec3::new(6.0, 2.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(2.0, 6.0, 0.0),
                Vec3::new(0.0, 6.0, 0.0),
            ],
        };
        assert!(m.contains_in_plane(Vec3::new(1.0, 1.0, 0.0)));
        assert!(m.contains_in_plane(Vec3::new(5.0, 1.0, 0.0)));
        assert!(m.contains_in_plane(Vec3::new(1.0, 5.0, 0.0)));
        // The notch: outside the L but inside its bounding box.
        assert!(!m.contains_in_plane(Vec3::new(5.0, 5.0, 0.0)));
        assert!(!m.contains_in_plane(Vec3::new(-1.0, 1.0, 0.0)));
        assert!(!m.contains_in_plane(Vec3::new(7.0, 1.0, 0.0)));
    }

    #[test]
    fn curvature_vectors_are_exact_on_a_circle_including_at_the_ends() {
        // The property the whole bend measurement rests on. A 20 mm arc must
        // read 1/20 everywhere, ends included -- ends that read half the
        // curvature turned a 90 degree bend into 66 before the guard band.
        let r = 20.0;
        let pts = arc(r, std::f64::consts::FRAC_PI_2, 25);
        let k = curvature_vectors(&pts, 2);
        assert_eq!(k.len(), pts.len());
        for (i, v) in k.iter().enumerate() {
            assert!(
                (v.length() * r - 1.0).abs() < 0.02,
                "station {i}: kappa = {} against 1/{r}",
                v.length()
            );
            // Anticlockwise in XY, so the binormal is +Z.
            assert!(v.z > 0.0, "station {i}: binormal {v:?} has the wrong sign");
            assert!(v.x.abs() + v.y.abs() < 1e-9 * v.z);
        }

        // ...and it tracks the radius, rather than returning something
        // plausible-looking once.
        for r in [5.0, 12.0, 40.0, 200.0] {
            let k = curvature_vectors(&arc(r, 1.0, 31), 2);
            let mid = k[15].length();
            assert!((mid * r - 1.0).abs() < 0.02, "r = {r} read as {}", 1.0 / mid);
        }
    }

    #[test]
    fn a_straight_polyline_has_no_curvature_and_is_not_moved_by_smoothing() {
        // The two failures that together produced a 25% area error on a duct of
        // constant section, pinned as one test.
        let dir = DVec3::new(1.0, 2.0, -0.5).normalize();
        let mut pts: Vec<DVec3> = (0..20).map(|i| dir * (i as f64 * 1.7)).collect();
        for v in curvature_vectors(&pts, 2) {
            assert!(v.length() < 1e-9, "a straight line curved: {v:?}");
        }
        let before = pts.clone();
        smooth_polyline(&mut pts, 1.5);
        for (a, b) in before.iter().zip(&pts) {
            assert!((*a - *b).length() < 1e-9, "smoothing moved {a:?} to {b:?}");
        }
    }

    #[test]
    fn smoothing_barely_shrinks_an_arc_and_keeps_it_circular() {
        // Gaussian smoothing pulls points on a circle toward their chords, so
        // it must shrink the radius a little -- but uniformly, and by far less
        // than the curvature signal it is protecting.
        let r = 20.0;
        let mut pts = arc(r, std::f64::consts::FRAC_PI_2, 25);
        smooth_polyline(&mut pts, 1.5);
        let radii: Vec<f64> = pts.iter().map(|p| p.length()).collect();
        let lo = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(lo > 0.99 * r, "smoothing shrank the arc to {lo} from {r}");
        assert!(hi <= r + 1e-9, "smoothing should not grow the arc: {hi}");
        assert!(hi - lo < 0.02 * r, "the shrink is uneven: {lo}..{hi}");
    }

    #[test]
    fn a_transitions_area_ratio_is_always_the_smaller_over_the_larger() {
        let contract = Transition {
            s_start_mm: 0.0,
            s_end_mm: 10.0,
            area_in_mm2: 200.0,
            area_out_mm2: 100.0,
            included_angle_deg: 30.0,
            is_contraction: true,
        };
        let expand = Transition { area_in_mm2: 100.0, area_out_mm2: 200.0, ..contract };
        assert!((contract.area_ratio() - 0.5).abs() < 1e-12);
        assert!((expand.area_ratio() - 0.5).abs() < 1e-12);
        // A degenerate zero area must not divide by zero.
        let bad = Transition { area_out_mm2: 0.0, ..contract };
        assert!((bad.area_ratio() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn confidence_widens_the_band_as_it_falls() {
        assert!(Confidence::Good < Confidence::Marginal);
        assert!(Confidence::Marginal < Confidence::Poor);
        assert_eq!(Confidence::Good.max(Confidence::Poor), Confidence::Poor);
        assert!(Confidence::Good.geometry_sigma() < Confidence::Marginal.geometry_sigma());
        assert!(Confidence::Marginal.geometry_sigma() < Confidence::Poor.geometry_sigma());
    }

    #[test]
    fn an_orthonormal_pair_is_orthonormal_for_every_tangent_including_degenerate_ones() {
        for t in [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-0.6, 0.0, 0.8),
            DVec3::ZERO,
        ] {
            let (u, v) = orthonormal_pair(t);
            assert!((u.length() - 1.0).abs() < 1e-9, "u = {u:?} for t = {t:?}");
            assert!((v.length() - 1.0).abs() < 1e-9, "v = {v:?} for t = {t:?}");
            assert!(u.dot(v).abs() < 1e-9, "u.v = {} for t = {t:?}", u.dot(v));
            if t.length_squared() > 0.5 {
                assert!(u.dot(t).abs() < 1e-9);
                assert!(v.dot(t).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn the_equivalent_rectangle_recovers_a_rectangle_it_was_given() {
        // Fill a band with cell centres covering 20 x 12 mm and check the
        // principal moments come back as 20 and 12 after Sheppard's correction.
        // This step decides D_h, so it has to be exact for the shape every
        // rectangular-duct correlation assumes.
        let dx = 0.5;
        let (w, h) = (20.0, 12.0);
        let mut acc = BandAcc::default();
        let nx = (w / dx) as i32;
        let ny = (h / dx) as i32;
        for i in 0..nx {
            for j in 0..ny {
                let p = DVec3::new(
                    (i as f64 + 0.5) * dx - w * 0.5,
                    (j as f64 + 0.5) * dx - h * 0.5,
                    0.0,
                );
                acc.w += 1.0;
                acc.sum += p;
                acc.sum_sq += p * p;
                acc.sum_cross += DVec3::new(p.x * p.y, p.y * p.z, p.z * p.x);
            }
        }
        let (major, minor, l1, l2) = acc.principal(DVec3::Z);
        let a = (12.0 * l1 + dx * dx).sqrt();
        let b = (12.0 * l2 + dx * dx).sqrt();
        assert!((a - w).abs() < 1e-6, "long side {a}, expected {w}");
        assert!((b - h).abs() < 1e-6, "short side {b}, expected {h}");
        assert!(major.x.abs() > 0.999, "major axis {major:?} should be X");
        assert!(minor.y.abs() > 0.999, "minor axis {minor:?} should be Y");
        // 4A/P for 20 x 12 is 15.
        assert!((2.0 * a * b / (a + b) - 15.0).abs() < 1e-5);
    }
}

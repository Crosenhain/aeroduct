//! Residence time, measured with tracers, and the passage volume it is judged
//! against.
//!
//! # Why this lives in the app
//!
//! `ad_metrics::rtd` is deliberately fed rather than self-driving: it takes
//! `(age, weight)` pairs and never touches tracer state. The *source* of those
//! pairs has to be something that can advect a particle through the velocity
//! field, and the only such thing in this workspace — `ad_render`'s streakline
//! system — is a visualisation that never reports where its particles went. So
//! the app integrates its own, on the CPU, from a snapshot of the solver field.
//!
//! # Why it is off by default
//!
//! Reading the velocity field back needs `SolverConfig::macroscopic_buffer`,
//! which costs 16 bytes per cell of storage plus the same again in a staging
//! buffer — 691 MB at the interactive tier — and the readback itself is
//! blocking. That is a bad trade for a plot, so it is behind `AERODUCT_RTD=1`
//! and the panel honestly says it has nothing when it is off.
//!
//! The cheap fix is one accessor on `ad_solver::Solver`: the velocity texture is
//! already created with `COPY_SRC`, so exposing it would let this copy just the
//! part's own bounding box (14 MB here, not 345 MB) with no extra allocation at
//! all. That is a change for the solver's owner, not for this crate.
//!
//! # What the numbers mean
//!
//! Tracers are seeded uniformly over the *area* of the inlet mouth, so each one
//! is weighted by the through-plane flux at its seed — the flow rate of the
//! streamtube it stands for. Weighting them equally instead is the classic RTD
//! error: it over-counts slow near-wall fluid, stretches the tail, and invents
//! dead volume that is not there.
//!
//! A tracer ends in one of three ways:
//!
//! * **exit** — it crossed the outlet mouth plane. Its age enters `E(t)`.
//! * **trapped** — it aged out or stalled below a thousandth of the inlet speed.
//!   Its weight is what [`ad_ui::view::ResidenceTimeView::trapped_fraction`]
//!   reports, and it is the recirculation seen from the Lagrangian side.
//! * **lost** — it left the captured region without crossing the outlet. Counted
//!   in neither, because "we stopped looking" is not a measurement.

use ad_geom::Mouth;
use ad_gpu::{flags, Bbox, Grid};
use ad_metrics::AgeSample;
use glam::{UVec3, Vec3};

use crate::sim::Sim;

/// Tracers per side of the seed grid over the inlet patch. 576 streamtubes is
/// enough for a smooth `E(t)` at 128 bins and costs a few tens of milliseconds.
const SEEDS_PER_SIDE: u32 = 24;

/// Hard ceiling on integration substeps per tracer, so a stagnant pocket cannot
/// turn one particle into an unbounded loop.
const MAX_SUBSTEPS: u32 = 20_000;

/// Fraction of a cell a tracer may cross per substep.
const CFL: f32 = 0.5;

/// Is the tracer path enabled? See the module docs for what it costs.
pub fn rtd_enabled() -> bool {
    matches!(
        std::env::var("AERODUCT_RTD").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Log the through-plane flow on a stack of grid planes around a mouth, read
/// straight off the CPU-side macroscopic field.
///
/// A ground truth that owes nothing to the GPU reduction: it visits cell centres
/// with no interpolation and no quadrature, so a disagreement between this and
/// [`ad_metrics::PlaneMetrics`] localises itself immediately. It also prints how
/// many cells on each plane are flagged `INLET`, which is what caught the
/// transposed inlet footprint documented in `sim::mark_inlet` — 420 of 3,714
/// driven cells, and a flow rate a factor of 26 below the hand calculation.
///
/// Runs only with `AERODUCT_RTD`, because it needs the same macroscopic buffer.
pub fn probe_mouth(sim: &mut Sim, which: usize, label: &str) {
    let Ok(full) = sim.solver.read_macroscopic() else {
        return;
    };
    let grid = sim.grid;
    let m = &sim.mouths[which];
    let axis = (m.axis as usize) % 3;
    let n = m.patch.normal.normalize_or(Vec3::Z);
    let c = m.patch.center_mm;
    let half = m.patch.half_u.abs() + m.patch.half_v.abs();
    let (ua, va) = ((axis + 1) % 3, (axis + 2) % 3);
    let k0 = ((c[axis] - grid.origin_mm[axis]) / grid.dx_mm).round() as i32;
    let c_u = sim.units.c_u();
    for dk in -2i32..=8 {
        let k = k0 + dk;
        if k < 0 || k as u32 >= grid.dims[axis] {
            continue;
        }
        let (mut n_fluid, mut n_inlet, mut sum_w, mut sum_speed) = (0u64, 0u64, 0.0f64, 0.0f64);
        for a in 0..grid.dims[ua] {
            for b in 0..grid.dims[va] {
                let mut cell = UVec3::ZERO;
                cell[axis] = k as u32;
                cell[ua] = a;
                cell[va] = b;
                let p = grid.cell_center_mm(cell);
                if (p[ua] - c[ua]).abs() > half[ua] || (p[va] - c[va]).abs() > half[va] {
                    continue;
                }
                let i = grid.linear(cell) as usize;
                let fl = sim.mask[i];
                if !flags::is_fluid(fl) {
                    continue;
                }
                n_fluid += 1;
                if fl & flags::INLET != 0 {
                    n_inlet += 1;
                }
                let v = full[i];
                let u = Vec3::new(v[0], v[1], v[2]);
                sum_w += u.dot(n) as f64;
                sum_speed += u.length() as f64;
            }
        }
        let nf = n_fluid.max(1) as f64;
        log::info!(
            "{label} plane k={k} ({:+.3} mm): {n_fluid} fluid ({n_inlet} inlet), \
             u.n {:.4} m/s, |u| {:.4} m/s, Q {:.3} L/s",
            grid.origin_mm[axis] + k as f32 * grid.dx_mm,
            sum_w / nf * c_u,
            sum_speed / nf * c_u,
            sum_w * c_u * (grid.dx_mm as f64).powi(2) * 1e-6 * 1000.0,
        );
    }
}

/// A cuboid of the interior grid, and the mapping to and from mm.
#[derive(Clone, Copy)]
struct Region {
    grid: Grid,
    lo: UVec3,
    dims: UVec3,
}

impl Region {
    /// The cells covering the part's own bounding box.
    ///
    /// Everything both jobs here care about — the passage, and where a tracer
    /// can be — is inside the printed part. The rest of the 261 mm domain is
    /// entrained room air, twelve times the volume and none of the answer.
    fn of_scene(sim: &Sim) -> Option<Self> {
        let bbox = sim.duct_bbox();
        // One cell of margin, so the mouth planes (which sit exactly on the
        // bounding box) have a neighbour on both sides to interpolate from.
        let pad = Vec3::splat(sim.grid.dx_mm);
        let (lo, hi) = sim.grid.cell_range(Bbox {
            min: bbox.min - pad,
            max: bbox.max + pad,
        })?;
        Some(Self {
            grid: sim.grid,
            lo,
            dims: hi - lo + UVec3::ONE,
        })
    }

    fn len(&self) -> usize {
        (self.dims.x as usize) * (self.dims.y as usize) * (self.dims.z as usize)
    }

    fn local(&self, x: u32, y: u32, z: u32) -> usize {
        ((z * self.dims.y + y) * self.dims.x + x) as usize
    }

    fn cell(&self, li: usize) -> UVec3 {
        let (nx, ny) = (self.dims.x as usize, self.dims.y as usize);
        let x = li % nx;
        let y = (li / nx) % ny;
        let z = li / (nx * ny);
        self.lo + UVec3::new(x as u32, y as u32, z as u32)
    }

    fn on_shell(&self, li: usize) -> bool {
        let c = self.cell(li) - self.lo;
        c.x == 0
            || c.y == 0
            || c.z == 0
            || c.x + 1 == self.dims.x
            || c.y + 1 == self.dims.y
            || c.z + 1 == self.dims.z
    }
}

/// Fluid volume of the duct passage, mm^3, or `None` if it could not be
/// isolated.
///
/// # The definition
///
/// The passage is the fluid **enclosed by the part**: the cells inside the
/// part's bounding box that cannot be reached from the outside of that box
/// except through a mouth. So the fill is seeded from the box's shell and the
/// two mouth planes are **walled off**, and whatever fluid the fill never
/// reaches is the passage.
///
/// Walling the mouths off is the part that has to be right. Excluding them from
/// the *seeds* alone is not enough: the free air directly in front of a mouth is
/// on the shell too, is seeded, and walks straight in through the opening — at
/// which point the fill reports the entire bounding box. Blocking the mouth's
/// own cell plane seals the passage at exactly the surface that closes it.
///
/// Simply flooding from the inlet instead would be wrong in the other
/// direction: this part is an elbow, so the concave side of the bend is open air
/// that is inside the bounding box, and a fill that escaped into it would report
/// the whole box.
///
/// # Why it matters
///
/// `ad_metrics::DuctMetrics` seeds `tau_ideal = V/Q` from the *volume pass*,
/// whose `V` is every fluid cell in the 261 mm domain. That is about 9,000,000
/// mm^3 against a passage of order 100,000 — a `tau_ideal` roughly ninety times
/// too long, which would make any real residence time look like 99% dead volume.
/// The app overrides it with this.
pub fn passage_volume_mm3(sim: &Sim) -> Option<f64> {
    let cell_mm3 = (sim.grid.dx_mm as f64).powi(3);
    // In the *walled* plenum domain the question is already answered.
    // `crate::plenum` has filled everything that is not passage or extension
    // with solid, and both extensions are strictly outside the part's bounding
    // box — so the fluid inside that box *is* the passage, exactly, with no fill
    // needed.
    //
    // Which is fortunate, because the fill below cannot run there: it is seeded
    // from the outside of the part, and in a walled plenum domain there is no
    // outside. Left to itself it finds no seeds, reports the whole box as
    // enclosed, and is rejected by the leak guard — a correct refusal, but a
    // refusal, and the ideal residence time then goes unreported for a domain
    // that knows the answer better than the room does.
    //
    // The open-walled variant keeps its trapped air, so it is *not* this case:
    // short-circuiting there would count the pockets between the part and its
    // bounding box as passage and report five times the real volume.
    if sim.domain.plenum.map(|p| p.walls) == Some(crate::plenum::PlenumWalls::Solid) {
        let bbox = sim.duct_bbox();
        let (lo, hi) = sim.grid.cell_range(bbox)?;
        let mut passage = 0u64;
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let c = glam::UVec3::new(x, y, z);
                    if flags::is_fluid(sim.mask[sim.grid.linear(c) as usize])
                        && bbox.contains(sim.grid.cell_center_mm(c))
                    {
                        passage += 1;
                    }
                }
            }
        }
        return (passage > 0).then(|| passage as f64 * cell_mm3);
    }

    let region = Region::of_scene(sim)?;
    let (passage, fluid) = enclosed_cells(region, &sim.mask, &sim.mouths);
    // A fill that leaked reports most of the bounding box; a fill that found
    // nothing reports zero. Both are "we do not know", and saying so beats
    // quoting a residence time against a volume that is not the duct's.
    if passage == 0 || passage as f64 > 0.6 * fluid as f64 {
        log::debug!("passage fill rejected: {passage} of {fluid} fluid cells inside the part bbox");
        return None;
    }
    Some(passage as f64 * cell_mm3)
}

/// The flood fill itself: `(enclosed fluid cells, total fluid cells)` in
/// `region`.
///
/// Split out so it can be driven from a fixture with no GPU and no `Sim`, which
/// is the only way the leak this function exists to avoid can be tested.
fn enclosed_cells(region: Region, mask: &[u8], mouths: &[Mouth]) -> (u64, u64) {
    let grid = region.grid;
    let n = region.len();
    let fluid_at = |li: usize| flags::is_fluid(mask[grid.linear(region.cell(li)) as usize]);

    // `outside` starts true on the mouth planes so the fill treats them as wall:
    // they are never enqueued, so nothing propagates through them, and they are
    // excluded from the passage count as well. One cell layer per mouth is a
    // fraction of a percent of the volume and the alternative is a fill that
    // walks in through the opening.
    let mut outside: Vec<bool> = (0..n)
        .map(|li| in_a_mouth(mouths, grid.dx_mm, grid.cell_center_mm(region.cell(li))))
        .collect();
    let mut stack: Vec<u32> = Vec::new();
    for li in 0..n {
        if !region.on_shell(li) || !fluid_at(li) || outside[li] {
            continue;
        }
        outside[li] = true;
        stack.push(li as u32);
    }

    // Six-connected flood through fluid only. Twenty-six-connected would leak
    // diagonally through a staircased wall that is watertight face-to-face,
    // which is exactly the wall a voxelised 2 mm shell produces.
    while let Some(li) = stack.pop() {
        let c = region.cell(li as usize) - region.lo;
        let mut visit = |x: u32, y: u32, z: u32| {
            let nb = region.local(x, y, z);
            if !outside[nb] && fluid_at(nb) {
                outside[nb] = true;
                stack.push(nb as u32);
            }
        };
        if c.x > 0 {
            visit(c.x - 1, c.y, c.z);
        }
        if c.y > 0 {
            visit(c.x, c.y - 1, c.z);
        }
        if c.z > 0 {
            visit(c.x, c.y, c.z - 1);
        }
        if c.x + 1 < region.dims.x {
            visit(c.x + 1, c.y, c.z);
        }
        if c.y + 1 < region.dims.y {
            visit(c.x, c.y + 1, c.z);
        }
        if c.z + 1 < region.dims.z {
            visit(c.x, c.y, c.z + 1);
        }
    }

    let (mut passage, mut fluid) = (0u64, 0u64);
    for li in 0..n {
        if !fluid_at(li) {
            continue;
        }
        fluid += 1;
        if !outside[li] {
            passage += 1;
        }
    }
    (passage, fluid)
}

/// Does `p` lie in the plane of a detected mouth, inside its opening rectangle?
///
/// The plane tolerance is *half* a cell, so exactly one cell-centre plane — the
/// one `sim::mark_inlet` rounds to — answers yes. A full cell would catch the
/// layer of room air in front of the mouth as well and count it as duct.
fn in_a_mouth(mouths: &[Mouth], dx_mm: f32, p: Vec3) -> bool {
    let slack = dx_mm * 0.5;
    mouths.iter().any(|m| {
        let axis = (m.axis as usize) % 3;
        let c = m.patch.center_mm;
        if (p[axis] - c[axis]).abs() > slack {
            return false;
        }
        // `half_u` and `half_v` are axis-aligned in-plane extents, one along
        // each of the other two axes, so their component-wise sum is the
        // rectangle's half-size on every axis at once.
        let half = m.patch.half_u.abs() + m.patch.half_v.abs() + Vec3::splat(slack);
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        (p[u] - c[u]).abs() <= half[u] && (p[v] - c[v]).abs() <= half[v]
    })
}

/// One seeded streamtube.
struct Seed {
    /// Start point, mm, one cell downstream of the mouth plane.
    position_mm: Vec3,
    /// Inlet normal, for the flux weight.
    normal: Vec3,
}

/// How a tracer finished, in weight.
#[derive(Debug, Default, Clone, Copy)]
struct Tally {
    exited: f64,
    trapped: f64,
    lost: f64,
}

/// ...and which of the three it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// Crossed the outlet mouth plane. Its age is a sample of `E(t)`.
    Exited,
    /// Aged out, stalled, or exhausted its substep budget: still in the duct.
    Trapped,
    /// Left the captured region without crossing the outlet. Counted in
    /// neither, because "we stopped looking" is not a measurement.
    Lost,
}

/// The CPU tracer integrator.
pub struct TracerRtd {
    region: Region,
    seeds: Vec<Seed>,
    /// Outlet mouth plane, with the normal pointing **downstream** so a tracer
    /// that has left satisfies `(p - centre) . n >= 0`.
    outlet_center_mm: Vec3,
    outlet_normal: Vec3,
    /// Age at which a tracer is declared trapped, seconds.
    max_age_s: f64,
    /// Wall-clock seconds between captures, and the accumulator.
    interval_s: f64,
    since_s: f64,
    /// Turned off after a failed readback, so a run cannot stall once a frame
    /// on an error it will keep hitting.
    live: bool,
}

impl TracerRtd {
    /// Seed over the inlet mouth. `None` when the geometry gives nothing to
    /// seed, which is not an error — the panel just stays empty.
    pub fn new(sim: &Sim, inlet: usize, outlet: usize, interval_s: f64) -> Option<Self> {
        let region = Region::of_scene(sim)?;
        let inlet_mouth = sim.mouths.get(inlet)?;
        let outlet_mouth = sim.mouths.get(outlet)?;
        let patch = inlet_mouth.patch;
        let normal = patch.normal.normalize_or(Vec3::X);
        let dx = sim.grid.dx_mm;

        let mut seeds = Vec::new();
        let n = SEEDS_PER_SIDE.max(2);
        for i in 0..n {
            for j in 0..n {
                let s = -1.0 + 2.0 * (i as f32 + 0.5) / n as f32;
                let t = -1.0 + 2.0 * (j as f32 + 0.5) / n as f32;
                // A cell downstream of the plane, so the first interpolation
                // stencil sits inside the passage rather than straddling the
                // mouth and the room air behind it.
                let p = patch.point(s, t) + normal * dx;
                let c = sim.grid.cell_containing(p);
                if c.cmplt(glam::IVec3::ZERO).any() || c.cmpge(sim.grid.dims.as_ivec3()).any() {
                    continue;
                }
                if !flags::is_fluid(sim.mask[sim.grid.linear(c.as_uvec3()) as usize]) {
                    continue;
                }
                seeds.push(Seed {
                    position_mm: p,
                    normal,
                });
            }
        }
        if seeds.is_empty() {
            return None;
        }

        // Twenty times the straight-line transit at the inlet bulk speed. Long
        // enough that a genuinely slow path still exits and short enough that a
        // recirculating one is called trapped rather than integrated forever.
        let length_mm = sim.duct_bbox().size().max_element() as f64;
        let u = sim.units.u_phys.abs().max(0.05);
        let max_age_s = 20.0 * (length_mm * 1e-3) / u;

        Some(Self {
            region,
            seeds,
            outlet_center_mm: outlet_mouth.patch.center_mm,
            outlet_normal: -outlet_mouth.patch.normal.normalize_or(Vec3::X),
            max_age_s,
            interval_s: interval_s.max(0.1),
            since_s: f64::INFINITY,
            live: true,
        })
    }

    /// Advance the throttle and, when it fires, capture the field and integrate.
    ///
    /// Returns the exits and the trapped weight fraction, or `None` when this
    /// frame did nothing — which is nearly every frame, by design.
    pub fn update(&mut self, sim: &mut Sim, dt_s: f32) -> Option<(Vec<AgeSample>, f64)> {
        if !self.live {
            return None;
        }
        self.since_s += dt_s as f64;
        if self.since_s < self.interval_s {
            return None;
        }
        self.since_s = 0.0;

        let field = match self.capture(sim) {
            Some(f) => f,
            None => return None,
        };
        // Lattice velocity to mm/s. Recomputed every capture because a hot-
        // applied velocity change moves `dt` and with it `c_u`.
        let c_u_mm_s = (sim.units.c_u() * 1e3) as f32;
        // A tracer below a thousandth of the inlet bulk is not going anywhere on
        // any timescale this run covers, so it is stagnant rather than slow.
        let stall_mm_s = 1e-3 * (sim.units.u_lb as f32).abs() * c_u_mm_s;
        Some(self.integrate(&field, c_u_mm_s, stall_mm_s))
    }

    /// Pull the velocity field back and keep only the part's bounding box.
    fn capture(&mut self, sim: &mut Sim) -> Option<Vec<Vec3>> {
        let t0 = std::time::Instant::now();
        let full = match sim.solver.read_macroscopic() {
            Ok(v) => v,
            Err(e) => {
                log::warn!("residence time disabled: {e:#}");
                self.live = false;
                return None;
            }
        };
        let grid = sim.grid;
        let r = self.region;
        let mut out = vec![Vec3::ZERO; r.len()];
        for z in 0..r.dims.z {
            for y in 0..r.dims.y {
                for x in 0..r.dims.x {
                    let c = r.lo + UVec3::new(x, y, z);
                    let v = full[grid.linear(c) as usize];
                    out[r.local(x, y, z)] = Vec3::new(v[0], v[1], v[2]);
                }
            }
        }
        log::debug!(
            "tracer field capture: {} cells in {:.0} ms",
            r.len(),
            t0.elapsed().as_secs_f64() * 1e3
        );
        Some(out)
    }

    /// Trilinear velocity in lattice units, or `None` outside the region.
    fn sample(&self, field: &[Vec3], p: Vec3) -> Option<Vec3> {
        let g = self.region.grid;
        let f = (p - g.origin_mm) / g.dx_mm - self.region.lo.as_vec3();
        if f.x < 0.0 || f.y < 0.0 || f.z < 0.0 {
            return None;
        }
        let i = f.floor();
        let (ix, iy, iz) = (i.x as u32, i.y as u32, i.z as u32);
        let d = self.region.dims;
        if ix + 1 >= d.x || iy + 1 >= d.y || iz + 1 >= d.z {
            return None;
        }
        let t = f - i;
        let at = |dx: u32, dy: u32, dz: u32| field[self.region.local(ix + dx, iy + dy, iz + dz)];
        let lerp = |a: Vec3, b: Vec3, w: f32| a + (b - a) * w;
        let c00 = lerp(at(0, 0, 0), at(1, 0, 0), t.x);
        let c10 = lerp(at(0, 1, 0), at(1, 1, 0), t.x);
        let c01 = lerp(at(0, 0, 1), at(1, 0, 1), t.x);
        let c11 = lerp(at(0, 1, 1), at(1, 1, 1), t.x);
        Some(lerp(lerp(c00, c10, t.y), lerp(c01, c11, t.y), t.z))
    }

    /// Advect every seed and tally what happened to it.
    fn integrate(&self, field: &[Vec3], c_u_mm_s: f32, stall_mm_s: f32) -> (Vec<AgeSample>, f64) {
        let dx = self.region.grid.dx_mm;
        let mut exits = Vec::with_capacity(self.seeds.len());
        let mut tally = Tally::default();

        for seed in &self.seeds {
            let Some(u0) = self.sample(field, seed.position_mm) else {
                continue;
            };
            // The streamtube's own flow rate, per unit seed area. Only the
            // component through the plane counts: a seed on a recirculating
            // corner carries fluid the *wrong* way and stands for no inflow.
            let w = (u0.dot(seed.normal) * c_u_mm_s) as f64;
            if !(w > 0.0) {
                continue;
            }

            let mut p = seed.position_mm;
            let mut age = 0.0f64;
            // Running out of substeps is a trapped tracer, not a lost one: it
            // is still inside the duct, it is just going nowhere.
            let mut outcome = Outcome::Trapped;
            for _ in 0..MAX_SUBSTEPS {
                let Some(u) = self.sample(field, p) else {
                    outcome = Outcome::Lost;
                    break;
                };
                let v = u * c_u_mm_s;
                let speed = v.length();
                if speed < stall_mm_s {
                    outcome = Outcome::Trapped;
                    break;
                }
                // Midpoint RK2 at a fixed fraction of a cell per step: the
                // truncation error is then bounded by the grid rather than by
                // however fast this particular streamline happens to be.
                let h = (CFL * dx / speed) as f64;
                let mid = p + v * (0.5 * h as f32);
                let Some(um) = self.sample(field, mid) else {
                    outcome = Outcome::Lost;
                    break;
                };
                p += um * (c_u_mm_s * h as f32);
                age += h;

                if (p - self.outlet_center_mm).dot(self.outlet_normal) >= 0.0 {
                    outcome = Outcome::Exited;
                    break;
                }
                if age >= self.max_age_s {
                    outcome = Outcome::Trapped;
                    break;
                }
            }
            match outcome {
                Outcome::Exited => {
                    exits.push(AgeSample::new(age, w));
                    tally.exited += w;
                }
                Outcome::Trapped => tally.trapped += w,
                Outcome::Lost => tally.lost += w,
            }
        }

        let total = tally.exited + tally.trapped + tally.lost;
        let trapped_fraction = if total > 0.0 {
            tally.trapped / total
        } else {
            f64::NAN
        };
        log::debug!(
            "tracers: {} exits, weights exited {:.3} trapped {:.3} lost {:.3}",
            exits.len(),
            tally.exited,
            tally.trapped,
            tally.lost
        );
        (exits, trapped_fraction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::FlowPatch;

    /// A 6 x 2 cell bore through a block set **two cells in** from the min-x
    /// face of a 20 x 12 x 12 grid, so there is a slab of free air in front of
    /// the mouth. That gap is the whole point: without it the fill has no route
    /// to the opening and the test cannot tell a sealed mouth from an open one.
    ///
    /// The bore is not square, so a transposed footprint is visible too.
    fn fixture() -> (Grid, Vec<u8>, Vec<Mouth>) {
        let grid = Grid {
            dims: UVec3::new(20, 12, 12),
            dx_mm: 1.0,
            origin_mm: Vec3::splat(0.5),
        };
        let mut mask = vec![flags::FLUID; grid.cell_count() as usize];
        for z in 0..grid.dims.z {
            for y in 0..grid.dims.y {
                for x in BLOCK {
                    let inside = (3..9).contains(&y) && (5..7).contains(&z);
                    if !inside {
                        mask[grid.linear(UVec3::new(x, y, z)) as usize] = flags::SOLID;
                    }
                }
            }
        }
        // Min-side and max-side faces of the block, in the axis order
        // `ad_geom::mouths::plane_axes` produces for each.
        let mouth = |on_min: bool, x: f32| Mouth {
            patch: FlowPatch {
                center_mm: Vec3::new(x, 6.0, 6.0),
                normal: if on_min { Vec3::X } else { -Vec3::X },
                half_u: if on_min { Vec3::Z * 1.0 } else { Vec3::Y * 3.0 },
                half_v: if on_min { Vec3::Y * 3.0 } else { Vec3::Z * 1.0 },
            },
            open_area_mm2: 12.0,
            axis: 0,
            on_min_side: on_min,
            boundary: Vec::new(),
        };
        (grid, mask, vec![mouth(true, 2.0), mouth(false, 20.0)])
    }

    /// Cells the block spans along x. It stops short of the min face so the
    /// mouth has open air in front of it, and runs to the grid edge at the far
    /// end so the bore exits through the region shell.
    const BLOCK: std::ops::Range<u32> = 2..20;

    /// The flood fill's job: find the bore and nothing else.
    ///
    /// Seeding from the shell while merely *skipping* the mouths is not enough —
    /// the free air in front of a mouth is seeded, walks in through the opening
    /// and floods the passage, which reports the whole bounding box as duct and
    /// makes `tau_ideal = V/Q` tens of times too long. The mouth planes have to
    /// be walled off, and that is what this checks.
    #[test]
    fn the_passage_fill_finds_the_bore_and_not_the_air_around_it() {
        let (grid, mask, mouths) = fixture();
        let region = Region {
            grid,
            lo: UVec3::ZERO,
            dims: grid.dims,
        };
        let (passage, fluid) = enclosed_cells(region, &mask, &mouths);

        // The bore runs x = 2..19 inclusive; the two mouth planes (x = 2 and
        // x = 19) are walled off, leaving x = 3..18.
        let bore_slice = 6 * 2;
        assert_eq!(
            passage,
            (BLOCK.len() as u64 - 2) * bore_slice,
            "the fill did not isolate the bore"
        );
        assert!(
            (passage as f64) < 0.6 * fluid as f64,
            "{passage} of {fluid} would be rejected as a leak"
        );
    }

    /// A mouth with no open air in front of it is still found, and an *open*
    /// grid with no mouths at all reports no passage rather than a plausible
    /// number.
    #[test]
    fn an_open_box_has_no_enclosed_passage() {
        let (grid, mask, _) = fixture();
        let region = Region {
            grid,
            lo: UVec3::ZERO,
            dims: grid.dims,
        };
        // No mouths: the bore is open at both ends and reachable from the shell,
        // so nothing is enclosed.
        let (passage, fluid) = enclosed_cells(region, &mask, &[]);
        assert_eq!(passage, 0, "an open bore is not a sealed passage");
        assert!(fluid > 0);
    }

    #[test]
    fn the_mouth_test_covers_the_opening_and_stops_at_its_rim() {
        let (grid, _, mouths) = fixture();
        let at = |p: Vec3| in_a_mouth(&mouths, grid.dx_mm, p);
        assert!(
            at(Vec3::new(2.5, 6.0, 6.0)),
            "the mouth's own cell plane is in the mouth"
        );
        assert!(
            at(Vec3::new(19.5, 8.5, 5.5)),
            "so is the far mouth's corner"
        );
        assert!(
            !at(Vec3::new(10.0, 6.0, 6.0)),
            "the middle of the bore is not a mouth"
        );
        assert!(
            !at(Vec3::new(2.5, 11.5, 6.0)),
            "outside the rim is not a mouth"
        );
        // Half a cell of tolerance: the room air a full cell in front of the
        // mouth must not be counted as duct.
        assert!(
            !at(Vec3::new(0.5, 6.0, 6.0)),
            "the layer in front of the mouth is not duct"
        );
        // ...and the footprint is read component-wise, so a transposed
        // description of the same rectangle answers identically.
        assert!(at(Vec3::new(2.5, 8.5, 6.5)) && !at(Vec3::new(2.5, 6.0, 8.5)));
    }

    #[test]
    fn the_region_index_round_trips() {
        let grid = Grid {
            dims: UVec3::new(9, 7, 5),
            dx_mm: 0.5,
            origin_mm: Vec3::ZERO,
        };
        let r = Region {
            grid,
            lo: UVec3::new(2, 1, 1),
            dims: UVec3::new(4, 3, 2),
        };
        assert_eq!(r.len(), 24);
        for z in 0..r.dims.z {
            for y in 0..r.dims.y {
                for x in 0..r.dims.x {
                    let li = r.local(x, y, z);
                    assert_eq!(r.cell(li), r.lo + UVec3::new(x, y, z));
                }
            }
        }
        // The shell is everything but the (empty) 2x1x0 interior of a 4x3x2 box.
        assert_eq!((0..r.len()).filter(|li| r.on_shell(*li)).count(), 24);
    }

    #[test]
    fn the_environment_switch_is_off_unless_it_is_asked_for() {
        std::env::remove_var("AERODUCT_RTD");
        assert!(!rtd_enabled());
        std::env::set_var("AERODUCT_RTD", "0");
        assert!(
            !rtd_enabled(),
            "a stray value must not turn a 691 MB buffer on"
        );
        std::env::set_var("AERODUCT_RTD", "1");
        assert!(rtd_enabled());
        std::env::remove_var("AERODUCT_RTD");
    }
}

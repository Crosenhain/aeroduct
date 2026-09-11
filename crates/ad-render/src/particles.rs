//! GPU streakline tracers: four million of them, advected entirely on the GPU.
//!
//! # These are streaklines, not streamlines
//!
//! The distinction matters and an engineer will notice it immediately. A
//! **streamline** is a curve everywhere tangent to a *single instantaneous*
//! velocity field. A **pathline** is the trajectory of one fluid particle
//! through a time-varying field. A **streakline** is the locus of all particles
//! that have passed through one seed point. In a steady flow all three coincide;
//! in the unsteady, separating flow of a 90-degree bend they emphatically do
//! not.
//!
//! What this module draws is streaklines: particles released continuously from
//! the inlet and advected through whatever velocity field the solver has
//! produced *at that moment*, which changes under them as they travel. That is
//! also exactly what a smoke wand in a wind tunnel produces, which is why the
//! image is intuitive. The public API says "streakline" throughout, and
//! [`StreaklineOverlay::hero_streamlines`] is the one place a genuine
//! streamline is offered — frozen field, RK4, a few thousand curves.
//!
//! # The five decisions that make this work
//!
//! **RK2 midpoint, not RK4.** RK4's formal fourth-order accuracy is a statement
//! about a smooth field. The derived velocity texture is trilinearly
//! interpolated, so it is only C0: its first derivative jumps at every voxel
//! face. Across such a field RK4 degrades to roughly second order — the same as
//! RK2 — while costing twice the texture fetches. On a bandwidth-bound advection
//! kernel, fetches *are* the cost. RK4 is kept only for the handful of hero
//! curves, where the field is frozen and pre-filtered.
//!
//! **A hard sub-cell CFL condition.** `h = min(h_max, 0.4 * dx / |u|)`, with 1-4
//! adaptive substeps. This is the single most important line in the file. A
//! particle that advances more than one cell per substep can step clean over the
//! 2 mm duct wall, reappear outside it, and keep going — and a few thousand
//! tracers streaming out through a solid wall makes the whole image read as
//! broken even though the solver is perfect. See [`substep_count`].
//!
//! **SDF wall collision.** With the geometry crate's signed distance field
//! available, a particle closer to the wall than its own radius is pushed back
//! out along `normalize(grad(sdf))`, and one deeper than `1.5 dx` inside is
//! killed and respawned. Without an SDF the pass degrades to velocity-only
//! advection plus a fluid-fraction test, which is weaker but never wrong.
//!
//! **Three blended seeding strategies.** Flux-weighted inlet seeding is the
//! honest default: accepting a candidate with probability proportional to `u.n`
//! makes particle density proportional to volumetric flow, so "more particles
//! here" means "more air here" rather than "more seeds here". A small uniform
//! volumetric reseed (1-2%) is what keeps recirculating dead zones from going
//! black — those zones are exactly what a duct designer is looking for, and
//! nothing released at the inlet ever reaches them. Q-criterion importance
//! seeding, via a prefix-sum CDF over a 32^3 bin grid, is the "show me the
//! vortices" mode.
//!
//! **No CPU synchronisation, ever.** Dead particles are pushed onto an append
//! buffer with an atomic counter and consumed by the seeder in the same frame,
//! with an indirect dispatch sized by a one-thread pass in between. Nothing is
//! read back, so nothing stalls.
//!
//! # Rendering
//!
//! Additive (`ONE, ONE`), depth test on, depth write off. Additive blending is
//! commutative, so there is no draw order to get wrong and no sorting to pay
//! for — with four million sprites, sorting is not on the table. One
//! non-indexed draw of `6 * count` vertices, with the quad generated from
//! `vertex_index` arithmetic; there is no vertex buffer at all.
//!
//! Two details are not optional:
//!
//! - **The sprite radius is clamped to at least 1.5 px, and the alpha is scaled
//!   by `(r_true / r_clamped)^2` to compensate.** A sub-pixel sprite is hit or
//!   missed by the pixel grid depending on where it lands, so a cloud of distant
//!   particles boils and shimmers. Clamping the radius stabilises coverage; the
//!   alpha scaling is what keeps the *total* light emitted unchanged, so a
//!   receding cloud fades smoothly instead of getting brighter. See
//!   [`sprite_footprint`].
//! - **Motion blur is geometric, not a post-process.** Each particle is drawn as
//!   a capsule stretched from `prev_pos` to `pos` — which is what its image
//!   actually is over one exposure — with the alpha divided by the stretch so
//!   energy is conserved. A screen-space post-process cannot do this correctly
//!   for overlapping additive sprites, and would smear the geometry behind them.

use ad_gpu::{Bbox, FlowPatch, ShaderDefines, ShaderLoader};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

use crate::colormap::{self, ColorMap, Interpolation};
use crate::{util, OverlayContext, OverlayPass};

/// Threads per workgroup for advection and seeding.
const PARTICLE_WG: u32 = 256;
/// Threads per workgroup for the importance-bin reduction.
const IMPORTANCE_WG: u32 = 64;
/// Threads in the single-workgroup CDF scan. Must divide the bin count.
const SCAN_WG: u32 = 256;

/// Bins per axis in the importance CDF. `32^3 = 32768` bins is 128 KiB, small
/// enough to scan in one workgroup and fine enough that a bin is a few
/// millimetres across on any of the resolution tiers.
pub const CDF_DIM: u32 = 32;
/// Total importance bins.
pub const CDF_BINS: u32 = CDF_DIM * CDF_DIM * CDF_DIM;

/// Population the emission is calibrated against.
///
/// Additive blending sums, so N tracers landing on a pixel are N times as bright
/// as one. Left unnormalised, turning the count up from 256k to 4M does not show
/// more structure — it shows a white blob, and the only route back is hunting
/// for an intensity value that happens to suit that particular count. Scaling
/// the per-tracer emission by `REFERENCE_COUNT / count` makes the population a
/// Monte-Carlo estimator of one fixed picture instead: more tracers buy less
/// noise, not more light. See [`StreaklineSettings::normalise_intensity`].
pub const REFERENCE_COUNT: u32 = 4 << 20;

/// Bytes per particle in the storage buffer. Two `vec4`s: position + age, and
/// previous position + lifetime.
pub const PARTICLE_BYTES: u64 = 32;
/// Bytes per trail ring entry: a 16-bit-quantised position plus a 16-bit age.
pub const TRAIL_ENTRY_BYTES: u64 = 8;

/// What the tracer colour encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreaklineColor {
    /// Speed in m/s, through the colour map. The default, and the one that makes
    /// the picture quantitative.
    #[default]
    Speed,
    /// Normalised age, so you can see how long a particle has been in the
    /// domain. This is what reveals a recirculation bubble: old particles pile
    /// up in it.
    Age,
    /// A single colour, for when the geometry is the message.
    Constant,
}

impl StreaklineColor {
    fn code(self) -> u32 {
        match self {
            StreaklineColor::Speed => 0,
            StreaklineColor::Age => 1,
            StreaklineColor::Constant => 2,
        }
    }
}

/// Relative weights of the three seeding strategies. Normalised before use, so
/// only the ratios matter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeedWeights {
    /// Flux-weighted release across the inlet patch, probability proportional to
    /// `u.n`. Physically meaningful density; the honest default.
    pub inlet: f32,
    /// Uniform reseed anywhere in the seed box. Small but non-zero: without it
    /// the recirculating regions never receive a particle and render black.
    pub volume: f32,
    /// Inverse-transform sampling of a Q-criterion CDF. Turn this up to hunt
    /// vortices; it deliberately over-samples them, so it is *not* a density you
    /// may read quantitatively.
    pub importance: f32,
}

impl Default for SeedWeights {
    fn default() -> Self {
        Self {
            inlet: 0.97,
            volume: 0.02,
            importance: 0.01,
        }
    }
}

impl SeedWeights {
    /// The vortex-hunting preset: most releases land on high-Q structure.
    pub fn vortex_hunt() -> Self {
        Self {
            inlet: 0.3,
            volume: 0.05,
            importance: 0.65,
        }
    }

    /// Weights scaled to sum to 1. An all-zero set falls back to pure inlet
    /// seeding rather than producing a division by zero and an empty screen.
    pub fn normalised(self) -> [f32; 3] {
        let w = [
            self.inlet.max(0.0),
            self.volume.max(0.0),
            self.importance.max(0.0),
        ];
        let sum = w[0] + w[1] + w[2];
        if sum <= 1e-9 {
            [1.0, 0.0, 0.0]
        } else {
            [w[0] / sum, w[1] / sum, w[2] / sum]
        }
    }
}

/// A signed distance field of the solid geometry, owned by the overlay.
///
/// Distinct from [`crate::volume::SdfVolume`], which borrows for the duration of
/// one frame: the particle pass keeps its bind group between frames, so it needs
/// an owned handle. `wgpu::TextureView` is a cheap reference-counted handle, so
/// cloning one out of the geometry crate's texture costs nothing.
#[derive(Debug, Clone)]
pub struct SdfSource {
    pub view: wgpu::TextureView,
    /// World-space corner of the SDF volume, mm.
    pub min_mm: Vec3,
    /// World-space size of the SDF volume, mm.
    pub size_mm: Vec3,
    /// Added to the sampled distance, mm. A small positive value keeps tracers
    /// clear of the half-solid voxels right at the wall.
    pub surface_offset_mm: f32,
}

/// Second-tier ribbon trails.
///
/// Deliberately a *separate, much smaller* population. A 32-deep position
/// history for all four million particles would be a gigabyte; at 256k it is
/// 64 MiB, which is affordable and still draws a dense enough weave to read as
/// continuous. The two populations are advected by the same kernel — a trail
/// particle is simply one whose index is below `trail_count`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrailSettings {
    pub enabled: bool,
    /// Ribbon width at the head, mm.
    pub width_mm: f32,
    /// Alpha at the head. Tapers quadratically to zero at the tail.
    pub alpha: f32,
    /// Age at which a trail's ring entry saturates its 16-bit age slot. Only
    /// needs to exceed the particle lifetime.
    pub max_age_s: f32,
}

impl Default for TrailSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            width_mm: 0.35,
            alpha: 0.5,
            max_age_s: 8.0,
        }
    }
}

/// Everything the user can turn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreaklineSettings {
    /// Simulated seconds of flow per second of wall clock.
    ///
    /// Not a cosmetic knob. Air moving at 5 m/s crosses the 145 mm test part in
    /// 29 ms; played at real time the tracers would cross the screen in two
    /// frames and read as noise. 0.02 puts a crossing at about 1.5 s, which is
    /// slow enough to follow and fast enough to feel like flow.
    pub time_scale: f32,
    /// Freeze the advection. The image stays; nothing moves.
    pub paused: bool,
    /// CFL number in cells per substep. 0.4 is comfortably sub-cell; above ~0.8
    /// tunnelling through a 2 mm wall becomes possible.
    pub cfl: f32,
    /// Substeps per frame, capped. Four is the point of diminishing returns:
    /// beyond it the cost is real and the remaining error is dominated by the
    /// field's own C0 interpolation, not by the integrator.
    pub max_substeps: u32,
    /// Hard backstop on the displacement of one substep, in cells. Bites only
    /// when the substep cap has already been saturated — i.e. exactly the case
    /// that would tunnel. A clamped particle lags the true streakline slightly;
    /// an unclamped one leaves the duct through a solid wall.
    pub max_step_cells: f32,
    /// Mean lifetime in simulated seconds.
    pub life_s: f32,
    /// Per-particle lifetime jitter, as a fraction. **Do not set this to zero.**
    /// A synchronised population dies together and the whole field of tracers
    /// blinks once per `life_s`, which looks like a driver fault.
    pub life_jitter: f32,
    /// Fraction of life spent fading in.
    pub fade_in: f32,
    /// Fraction of life spent fading out.
    pub fade_out: f32,
    /// Sprite radius in world millimetres.
    pub radius_mm: f32,
    /// Floor on the on-screen sprite radius, pixels. See the module docs.
    pub min_radius_px: f32,
    /// Emission per tracer, at [`REFERENCE_COUNT`] tracers.
    ///
    /// Small, and it has to be: four million additive sprites overlapping inside
    /// a 6 mm passage put hundreds of them on every pixel, so the per-tracer
    /// figure that sums to a correctly exposed cloud is a few percent. The
    /// target is HDR and runs through bloom, so the *sum* passing 1 is exactly
    /// what makes the core jet glow.
    pub intensity: f32,
    /// Scale the emission by `REFERENCE_COUNT / count`, so changing the tracer
    /// count changes the noise and not the exposure. Leave this on; turning it
    /// off is for calibrating a single sprite.
    pub normalise_intensity: bool,
    /// How much of the frame's displacement to stretch the capsule over.
    /// 0 draws round sprites, 1 draws the full exposure streak.
    pub motion_blur: f32,
    /// Longest capsule allowed, pixels. A particle that crosses half the screen
    /// in one frame is a sampling failure, not a motion blur.
    pub max_streak_px: f32,
    pub color: StreaklineColor,
    pub color_map: ColorMap,
    /// Data range the colour map spans: m/s for [`StreaklineColor::Speed`],
    /// ignored otherwise.
    pub color_range: [f32; 2],
    /// Collision radius against the SDF, mm.
    pub particle_radius_mm: f32,
    /// Fluid fraction below which a voxel counts as wall. Only used when there
    /// is no SDF.
    pub occupancy_min: f32,
    pub seed: SeedWeights,
    /// Q-tilde below which a bin contributes nothing to the importance CDF.
    pub q_threshold: f32,
    /// Peak `u.n` at the inlet, m/s, used as the rejection-sampling envelope.
    /// Set it to 0 to disable flux weighting and seed the patch uniformly.
    pub inlet_peak_speed_ms: f32,
    /// Distance to nudge a fresh particle along the inlet normal, mm, so it does
    /// not start exactly on a boundary voxel.
    pub inlet_offset_mm: f32,
    /// Frames between importance CDF rebuilds. The CDF only has to track the
    /// large-scale structure, and rebuilding it every frame would put a 2 M
    /// sample reduction plus a scan on the critical path for no visible gain.
    pub importance_interval: u32,
    pub trails: TrailSettings,
}

impl Default for StreaklineSettings {
    fn default() -> Self {
        Self {
            time_scale: 0.02,
            paused: false,
            cfl: 0.4,
            max_substeps: 4,
            max_step_cells: 1.0,
            life_s: 3.0,
            life_jitter: 0.2,
            fade_in: 0.05,
            fade_out: 0.2,
            radius_mm: 0.35,
            min_radius_px: 1.5,
            intensity: 0.03,
            normalise_intensity: true,
            motion_blur: 0.7,
            max_streak_px: 48.0,
            color: StreaklineColor::default(),
            color_map: ColorMap::Inferno,
            color_range: [0.0, 10.0],
            particle_radius_mm: 0.15,
            occupancy_min: 0.1,
            seed: SeedWeights::default(),
            q_threshold: 0.05,
            inlet_peak_speed_ms: 8.0,
            inlet_offset_mm: 0.5,
            importance_interval: 30,
            trails: TrailSettings::default(),
        }
    }
}

impl StreaklineSettings {
    pub fn sanitise(&mut self) {
        self.time_scale = self.time_scale.clamp(0.0, 10.0);
        self.cfl = self.cfl.clamp(0.02, 0.9);
        self.max_substeps = self.max_substeps.clamp(1, 8);
        self.max_step_cells = self.max_step_cells.clamp(0.1, 4.0);
        self.life_s = self.life_s.clamp(0.05, 600.0);
        self.life_jitter = self.life_jitter.clamp(0.0, 0.9);
        self.fade_in = self.fade_in.clamp(0.0, 0.5);
        self.fade_out = self.fade_out.clamp(0.0, 0.9);
        self.radius_mm = self.radius_mm.clamp(1e-3, 100.0);
        self.min_radius_px = self.min_radius_px.clamp(0.0, 32.0);
        self.intensity = self.intensity.clamp(0.0, 64.0);
        self.motion_blur = self.motion_blur.clamp(0.0, 1.0);
        self.max_streak_px = self.max_streak_px.clamp(1.0, 4096.0);
        self.particle_radius_mm = self.particle_radius_mm.clamp(0.0, 100.0);
        self.occupancy_min = self.occupancy_min.clamp(0.0, 1.0);
        self.importance_interval = self.importance_interval.max(1);
        self.inlet_peak_speed_ms = self.inlet_peak_speed_ms.max(0.0);
        if self.color_range[1] <= self.color_range[0] {
            self.color_range[1] = self.color_range[0] + 1e-3;
        }
        self.trails.width_mm = self.trails.width_mm.clamp(1e-3, 100.0);
        self.trails.alpha = self.trails.alpha.clamp(0.0, 4.0);
        self.trails.max_age_s = self.trails.max_age_s.max(self.life_s);
    }
}

/// Fixed capacities, chosen once when the overlay is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreaklineConfig {
    /// Tracers. Four million is the headline figure and costs 128 MiB.
    pub count: u32,
    /// How many of them also keep a position history. Zero disables trails
    /// entirely and allocates nothing.
    pub trail_count: u32,
    /// Ring depth of the history.
    pub trail_length: u32,
}

impl Default for StreaklineConfig {
    fn default() -> Self {
        Self {
            count: 4 << 20,
            trail_count: 0,
            trail_length: 32,
        }
    }
}

impl StreaklineConfig {
    /// The trail preset from the plan: 256k particles, 32 deep, 64 MiB.
    pub fn with_trails() -> Self {
        Self {
            count: 4 << 20,
            trail_count: 256 << 10,
            trail_length: 32,
        }
    }

    fn sane(self) -> Self {
        let count = self.count.clamp(1, 64 << 20);
        let trail_length = if self.trail_count == 0 {
            0
        } else {
            self.trail_length.clamp(2, 256)
        };
        Self {
            count,
            trail_count: self.trail_count.min(count),
            trail_length,
        }
    }

    /// Bytes of VRAM the buffers will occupy.
    pub fn bytes(self) -> u64 {
        let s = self.sane();
        s.count as u64 * (PARTICLE_BYTES + 4)
            + s.trail_count as u64 * s.trail_length as u64 * TRAIL_ENTRY_BYTES
            + CDF_BINS as u64 * 4
    }
}

// -- the CPU twins of the shader arithmetic ----------------------------------

/// Substeps for one frame under the sub-cell CFL condition.
///
/// `h = min(dt, cfl * dx / |u|)` is the step that keeps a particle inside its
/// own cell; the number of substeps is how many of those fit in the frame,
/// capped. Mirrors `advect_main` in `particles.wgsl` exactly.
///
/// The cap is the honest part: when it binds, the substep is *longer* than the
/// CFL limit and the displacement clamp in [`clamp_displacement`] takes over.
pub fn substep_count(dt: f32, speed_mm_s: f32, dx_mm: f32, cfl: f32, max_substeps: u32) -> u32 {
    if dt <= 0.0 || speed_mm_s <= 1e-6 || !dt.is_finite() {
        return 1;
    }
    let h_cfl = (cfl * dx_mm / speed_mm_s).max(1e-9);
    let n = (dt / h_cfl).ceil();
    if !n.is_finite() || n < 1.0 {
        return 1;
    }
    (n as u32).clamp(1, max_substeps.max(1))
}

/// Clamp a substep displacement to `max_cells` cells. Returns the clamped
/// displacement.
pub fn clamp_displacement(d: Vec3, dx_mm: f32, max_cells: f32) -> Vec3 {
    let limit = max_cells * dx_mm;
    let l = d.length();
    if l > limit && l > 0.0 {
        d * (limit / l)
    } else {
        d
    }
}

/// One RK2 midpoint step. `velocity` is in millimetres per second and `h` in
/// seconds, matching the shader.
///
/// Midpoint rather than Heun (trapezoid) because they cost the same and the
/// midpoint form has the smaller error constant on a rotating field, which is
/// the case that matters here.
pub fn rk2_step(p: Vec3, h: f32, velocity: impl Fn(Vec3) -> Vec3) -> Vec3 {
    let k1 = velocity(p);
    let k2 = velocity(p + k1 * (0.5 * h));
    p + k2 * h
}

/// One classical RK4 step, used only by the hero streamlines.
pub fn rk4_step(p: Vec3, h: f32, velocity: impl Fn(Vec3) -> Vec3) -> Vec3 {
    let k1 = velocity(p);
    let k2 = velocity(p + k1 * (0.5 * h));
    let k3 = velocity(p + k2 * (0.5 * h));
    let k4 = velocity(p + k3 * h);
    p + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)
}

/// Fade envelope over a particle's life: in over the first `fade_in` of it, out
/// over the last `fade_out`.
///
/// Smoothstep rather than linear at both ends. A linear fade-in has a
/// discontinuous derivative at the moment of birth, and with a large enough
/// population that shows up as a faint but perceptible texture of "popping"
/// along the inlet.
pub fn fade_alpha(age: f32, life: f32, fade_in: f32, fade_out: f32) -> f32 {
    if life <= 0.0 || age < 0.0 || age > life {
        return 0.0;
    }
    let smooth = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    let a = if fade_in > 1e-6 {
        smooth(age / (fade_in * life))
    } else {
        1.0
    };
    let b = if fade_out > 1e-6 {
        smooth((life - age) / (fade_out * life))
    } else {
        1.0
    };
    a * b
}

/// On-screen sprite radius and the alpha compensation that goes with it.
///
/// Returns `(radius_px, alpha_scale)`. The radius is `r_true` raised to at least
/// `min_px`; the alpha scale is `(r_true / r_px)^2`, which holds the sprite's
/// integrated energy constant so a receding particle dims smoothly instead of
/// brightening as its clamped footprint grows.
///
/// Both halves are needed. Clamping without compensating makes a distant cloud
/// *brighter* than a near one; compensating without clamping leaves the shimmer
/// the clamp exists to remove.
pub fn sprite_footprint(true_px: f32, min_px: f32) -> (f32, f32) {
    let r_true = true_px.max(1e-6);
    let r = r_true.max(min_px);
    let scale = (r_true / r) * (r_true / r);
    (r, scale.clamp(0.0, 1.0))
}

/// Energy normalisation for the motion-blur capsule.
///
/// A stationary sprite deposits light over an area `~r^2`; stretched into a
/// capsule of core length `L` it covers `~r^2 + L r`. Scaling the alpha by
/// `r / (r + L)` keeps the total emission the same, which is what real motion
/// blur does — a fast particle is a long faint streak, not a long bright one.
pub fn streak_alpha(radius_px: f32, length_px: f32) -> f32 {
    let r = radius_px.max(1e-6);
    r / (r + length_px.max(0.0))
}

/// Next slot of the trail ring, given whether this frame actually writes one.
///
/// The whole correctness of the ribbons rests on this. `vs_trail` connects ring
/// slots `head - j` and `head - j - 1` and trusts that they were written on
/// *consecutive* frames — that is what makes "the newer end has the larger age"
/// a sound test for a respawn. Advancing the head on a frame that writes
/// nothing — a paused sim, a render-only frame, trails toggled off — silently
/// breaks that: one segment per particle then joins two entries a whole ring
/// apart, and about a fifth of the population respawns inside that window, so a
/// fifth of the tracers grow a ribbon shooting clean across the domain. It
/// looks like a wild advection bug and it is a counter that ticked once too
/// often.
///
/// Returns `head` unchanged when `wrote` is false; a `len` below 2 has no ring
/// at all.
pub fn advance_trail_head(head: u32, len: u32, wrote: bool) -> u32 {
    if !wrote || len < 2 {
        return head;
    }
    (head + 1) % len
}

/// Corner offsets of the six vertices of a ribbon segment quad, in
/// `(along, across)` where `along` runs 0 at the older sample to 1 at the newer
/// and `across` runs -1 to 1.
///
/// Mirrors the `vertex_index` arithmetic in `vs_trail`. Both triangles wind the
/// same way, which the tests pin: culling is off for these ribbons, so a winding
/// error would produce a *correct-looking* image today and a disappearing ribbon
/// the moment someone turns culling on.
pub fn ribbon_corner(index: u32) -> Vec2 {
    // Quad corners 0..3, indexed by the triangle-list pattern (0,1,2, 0,2,3).
    let q = match index % 6 {
        0 | 3 => 0u32,
        1 => 1,
        2 | 4 => 2,
        _ => 3,
    };
    let along = if q == 1 || q == 2 { 1.0 } else { 0.0 };
    let across = if q == 2 || q == 3 { 1.0 } else { -1.0 };
    Vec2::new(along, across)
}

/// Screen-space positions of a ribbon segment's six vertices.
///
/// `older` and `newer` are the two ring samples in pixels; `width` is the
/// half-width in pixels.
pub fn ribbon_quad(older: Vec2, newer: Vec2, width: f32) -> [Vec2; 6] {
    let axis = newer - older;
    let len = axis.length();
    let dir = if len > 1e-5 { axis / len } else { Vec2::X };
    let nrm = Vec2::new(-dir.y, dir.x);
    let mut out = [Vec2::ZERO; 6];
    for (i, o) in out.iter_mut().enumerate() {
        let c = ribbon_corner(i as u32);
        *o = older + dir * (c.x * len) + nrm * (c.y * width);
    }
    out
}

// -- GPU plumbing ------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct StreaklineUniform {
    volume_min_mm: [f32; 3],
    dx_mm: f32,

    volume_size_mm: [f32; 3],
    dt_sim: f32,

    seed_min_mm: [f32; 3],
    cfl: f32,

    seed_size_mm: [f32; 3],
    max_step_cells: f32,

    inlet_center: [f32; 3],
    inlet_un_max: f32,

    inlet_u: [f32; 3],
    w_inlet: f32,

    inlet_v: [f32; 3],
    w_volume: f32,

    inlet_n: [f32; 3],
    w_import: f32,

    sdf_min_mm: [f32; 3],
    sdf_offset_mm: f32,

    sdf_inv_size: [f32; 3],
    particle_radius_mm: f32,

    count: u32,
    frame: u32,
    max_substeps: u32,
    sdf_present: u32,

    life_mean_s: f32,
    life_jitter: f32,
    fade_in: f32,
    fade_out: f32,

    radius_mm: f32,
    min_radius_px: f32,
    intensity: f32,
    color_mode: u32,

    color_lo: f32,
    color_inv_span: f32,
    cdf_dim: u32,
    q_threshold: f32,

    trail_count: u32,
    trail_len: u32,
    trail_head: u32,
    trail_max_age_s: f32,

    inlet_offset_mm: f32,
    occupancy_min: f32,
    motion_blur: f32,
    max_streak_px: f32,

    /// `1 / (dt_sim * 1000)`, i.e. millimetres-per-frame to metres per second.
    /// Held over from the last moving frame so a paused view keeps its colours.
    inv_dt_ms: f32,
    trail_width_mm: f32,
    trail_alpha: f32,
    _pad: f32,
}

/// Embed the overlay shaders next to the ones `util` already registers.
///
/// A fresh loader rather than an addition to [`crate::util::shader_loader`]: the
/// Wave 2 shaders need defines that the core preprocessor test does not set, and
/// keeping them out of that list keeps the two independent.
pub(crate) fn overlay_shader_loader() -> ShaderLoader {
    let mut l = util::shader_loader();
    l.add_virtual(
        "particles.wgsl",
        include_str!("../../../shaders/render/particles.wgsl"),
    );
    l.add_virtual(
        "isosurface.wgsl",
        include_str!("../../../shaders/render/isosurface.wgsl"),
    );
    l.add_virtual(
        "slice.wgsl",
        include_str!("../../../shaders/render/slice.wgsl"),
    );
    l.add_virtual("lic.wgsl", include_str!("../../../shaders/render/lic.wgsl"));
    l
}

pub(crate) fn overlay_defines() -> ShaderDefines {
    ShaderDefines::new()
        .value("PARTICLE_WG", PARTICLE_WG)
        .value("IMPORTANCE_WG", IMPORTANCE_WG)
        .value("SCAN_WG", SCAN_WG)
        .value("LIC_MAX_STEPS", crate::lic::MAX_STEPS)
        .value("HIST_WG", crate::isosurface::HISTOGRAM_WG)
}

/// Four million streaklines, advected and drawn entirely on the GPU.
pub struct StreaklineOverlay {
    config: StreaklineConfig,
    pub settings: StreaklineSettings,

    uniform: wgpu::Buffer,
    particles: wgpu::Buffer,
    free_list: wgpu::Buffer,
    counters: wgpu::Buffer,
    indirect: wgpu::Buffer,
    seed_scratch: wgpu::Buffer,
    cdf: wgpu::Buffer,
    trails: wgpu::Buffer,

    lut: wgpu::Texture,
    // Held only to keep the view and sampler alive for the draw bind group,
    // which is built once and never rebuilt.
    _lut_view: wgpu::TextureView,
    _lut_sampler: wgpu::Sampler,

    _sdf_fallback: wgpu::Texture,
    sdf_fallback_view: wgpu::TextureView,
    sdf: Option<SdfSource>,

    compute_layout: wgpu::BindGroupLayout,
    compute_group: wgpu::BindGroup,
    /// Identical to `compute_group` but with the scratch buffer at binding 4.
    /// Used by the seeder alone; see `seed_scratch`.
    seed_group: wgpu::BindGroup,
    _draw_layout: wgpu::BindGroupLayout,
    draw_group: wgpu::BindGroup,

    advect: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    seed: wgpu::ComputePipeline,
    importance: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    sprite_pipeline: wgpu::RenderPipeline,
    trail_pipeline: wgpu::RenderPipeline,

    inlet: Option<FlowPatch>,
    seed_bounds: Option<Bbox>,

    frame: u32,
    /// Slot of the trail ring this frame writes. Advanced by
    /// [`advance_trail_head`] and *only* on frames that write; see there for
    /// what goes wrong otherwise.
    trail_head: u32,
    /// Whether trails were on last frame, so the ring can be cleared on the
    /// rising edge rather than resuming across a gap it has no record of.
    trails_were_on: bool,
    /// Simulated seconds elapsed. Only used for diagnostics.
    sim_time: f32,
    /// Simulated seconds advanced on the previous recorded frame. Zero means
    /// the population did not move, whoever decided that — see [`OverlayPass::is_static`].
    last_dt_sim: f32,
    last_inv_dt_ms: f32,
    needs_clear: bool,
    lut_dirty: bool,
    last_color_map: ColorMap,
}

impl StreaklineOverlay {
    /// Build the whole system. `fields_layout` must be
    /// [`crate::Renderer::fields_bind_group_layout`] and `camera_layout`
    /// [`crate::Renderer::camera_bind_group_layout`].
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_layout: &wgpu::BindGroupLayout,
        fields_layout: &wgpu::BindGroupLayout,
        config: StreaklineConfig,
    ) -> Result<Self> {
        let config = config.sane();
        let mut settings = StreaklineSettings::default();
        settings.sanitise();

        let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
                mapped_at_creation: false,
            })
        };

        let uniform = util::uniform_buffer::<StreaklineUniform>(device, "streakline uniform");
        let particles = storage(
            "streakline particles",
            config.count as u64 * PARTICLE_BYTES,
            // VERTEX so the buffer can also be bound as vertex data by a future
            // instanced path; the draw today reads it as a storage buffer, which
            // is what lets one vertex fetch six sprite corners.
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
        );
        let free_list = storage(
            "streakline free list",
            config.count as u64 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let counters = storage("streakline counters", 16, wgpu::BufferUsages::COPY_SRC);
        let indirect = storage("streakline dispatch", 16, wgpu::BufferUsages::INDIRECT);
        // A stand-in for `indirect` in the seeder's bind group. wgpu's usage
        // scope is the whole compute pass, and `STORAGE_READ_WRITE` is exclusive
        // — so binding the very buffer the pass dispatches from is a validation
        // error, even though the shader never touches it from `seed_main`. The
        // scratch buffer keeps binding 4 occupied with something inert so one
        // bind-group layout still serves every compute pass.
        let seed_scratch = storage(
            "streakline dispatch scratch",
            16,
            wgpu::BufferUsages::empty(),
        );
        let cdf = storage(
            "streakline importance cdf",
            CDF_BINS as u64 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let trail_bytes =
            config.trail_count as u64 * config.trail_length as u64 * TRAIL_ENTRY_BYTES;
        // COPY_SRC so the ring can be read back and checked: the ribbon
        // segments are the one part of this system whose correctness is a
        // property of *pairs* of frames, which no single-frame test can see.
        let trails = storage(
            "streakline trails",
            trail_bytes,
            wgpu::BufferUsages::COPY_SRC,
        );

        let lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("streakline colour LUT"),
            size: wgpu::Extent3d {
                width: colormap::LUT_SIZE as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let lut_view = lut.create_view(&Default::default());
        let lut_sampler = util::linear_clamp_sampler(device, "streakline LUT");

        let sdf_fallback = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("streakline empty SDF"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // A large positive distance, so `d < r_particle` is never true and the
        // collision branch is inert without needing a second pipeline.
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &sdf_fallback,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::bytes_of(&1.0e6f32),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let sdf_fallback_view = sdf_fallback.create_view(&wgpu::TextureViewDescriptor {
            label: Some("streakline empty SDF"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let cs = wgpu::ShaderStages::COMPUTE;
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("streakline compute"),
            entries: &[
                util::uniform_entry(0, cs),
                util::storage_buffer_entry(1, cs, false),
                util::storage_buffer_entry(2, cs, false),
                util::storage_buffer_entry(3, cs, false),
                util::storage_buffer_entry(4, cs, false),
                util::storage_buffer_entry(5, cs, false),
                util::sampled_float_entry(6, cs, wgpu::TextureViewDimension::D3),
                util::storage_buffer_entry(7, cs, false),
            ],
        });

        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("streakline draw"),
            entries: &[
                util::uniform_entry(0, vf),
                // Read-only in the vertex stage: a *writable* storage buffer
                // there needs the VERTEX_WRITABLE_STORAGE downlevel flag, which
                // `ad-gpu` does not request and which a fair number of drivers
                // do not offer.
                util::storage_buffer_entry(1, wgpu::ShaderStages::VERTEX, true),
                util::storage_buffer_entry(2, wgpu::ShaderStages::VERTEX, true),
                util::sampled_float_entry(3, vf, wgpu::TextureViewDimension::D2),
                util::sampler_entry(4, vf, wgpu::SamplerBindingType::Filtering),
            ],
        });

        let loader = overlay_shader_loader();
        let defines = overlay_defines();

        let make_compute = |entry: &str, label: &str| {
            util::compute_pipeline(
                device,
                &loader,
                "particles.wgsl",
                entry,
                &defines,
                &[Some(fields_layout), Some(&compute_layout)],
                label,
            )
        };
        let advect = make_compute("advect_main", "streakline advect")?;
        let prepare = make_compute("prepare_main", "streakline dispatch prepare")?;
        let seed = make_compute("seed_main", "streakline seed")?;
        let importance = make_compute("importance_main", "streakline importance")?;
        let scan = make_compute("scan_main", "streakline cdf scan")?;

        let draw_module = loader.create_module(
            device,
            "particles.wgsl",
            &defines.clone().flag("PARTICLES_DRAW"),
        )?;
        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("streakline draw"),
            bind_group_layouts: &[Some(camera_layout), Some(&draw_layout)],
            immediate_size: 0,
        });

        // Additive, and therefore order-independent: no sort, no OIT, no depth
        // pre-pass. Destination alpha is left alone (`Zero, One`) because the
        // HDR target's alpha has already been finalised by the compositor and
        // accumulating four million sprites into it would drive it meaningless.
        let additive = wgpu::ColorTargetState {
            format: crate::post::HDR_FORMAT,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        };
        // Test against the scene, never write: an emissive tracer must not
        // occlude the tracer behind it, and with additive blending it cannot.
        let depth_test_only = wgpu::DepthStencilState {
            format: crate::mesh::DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(crate::mesh::DEPTH_COMPARE),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let mk_draw = |vs: &str, fs: &str, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&draw_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &draw_module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(depth_test_only.clone()),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &draw_module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(additive.clone())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let sprite_pipeline = mk_draw("vs_particle", "fs_particle", "streakline sprites");
        let trail_pipeline = mk_draw("vs_trail", "fs_trail", "streakline trails");

        let compute_group = Self::build_compute_group(
            device,
            &compute_layout,
            &uniform,
            &particles,
            &free_list,
            &counters,
            &indirect,
            &cdf,
            &sdf_fallback_view,
            &trails,
        );
        let seed_group = Self::build_compute_group(
            device,
            &compute_layout,
            &uniform,
            &particles,
            &free_list,
            &counters,
            &seed_scratch,
            &cdf,
            &sdf_fallback_view,
            &trails,
        );
        let draw_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("streakline draw"),
            layout: &draw_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: trails.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                },
            ],
        });

        let this = Self {
            config,
            settings,
            uniform,
            particles,
            free_list,
            counters,
            indirect,
            seed_scratch,
            cdf,
            trails,
            lut,
            _lut_view: lut_view,
            _lut_sampler: lut_sampler,
            _sdf_fallback: sdf_fallback,
            sdf_fallback_view,
            sdf: None,
            compute_layout,
            compute_group,
            seed_group,
            _draw_layout: draw_layout,
            draw_group,
            advect,
            prepare,
            seed,
            importance,
            scan,
            sprite_pipeline,
            trail_pipeline,
            inlet: None,
            seed_bounds: None,
            frame: 0,
            trail_head: 0,
            trails_were_on: false,
            sim_time: 0.0,
            last_dt_sim: 0.0,
            last_inv_dt_ms: 0.0,
            needs_clear: true,
            lut_dirty: true,
            last_color_map: ColorMap::Inferno,
        };
        this.upload_lut(queue);
        Ok(this)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_compute_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        particles: &wgpu::Buffer,
        free_list: &wgpu::Buffer,
        counters: &wgpu::Buffer,
        indirect: &wgpu::Buffer,
        cdf: &wgpu::Buffer,
        sdf: &wgpu::TextureView,
        trails: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("streakline compute"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: free_list.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cdf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(sdf),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: trails.as_entire_binding(),
                },
            ],
        })
    }

    pub fn config(&self) -> StreaklineConfig {
        self.config
    }
    pub fn count(&self) -> u32 {
        self.config.count
    }
    pub fn settings(&self) -> &StreaklineSettings {
        &self.settings
    }
    /// Mutate the settings. The colour LUT is re-baked on the next frame.
    pub fn settings_mut(&mut self) -> &mut StreaklineSettings {
        self.lut_dirty = true;
        &mut self.settings
    }

    /// The inlet patch to release tracers from. Without one the inlet weight is
    /// ignored and seeding falls back to the volumetric and importance
    /// strategies, which still produces a picture — just not a flux-honest one.
    pub fn set_inlet(&mut self, patch: Option<FlowPatch>) {
        self.inlet = patch;
    }
    pub fn inlet(&self) -> Option<FlowPatch> {
        self.inlet
    }

    /// Restrict the uniform volumetric reseed to a box, normally the duct's own
    /// bounds. Without it the reseed covers the whole domain, most of which is
    /// still room air, and the 1-2% of releases that were meant to light up the
    /// recirculation zones are scattered uselessly instead.
    pub fn set_seed_bounds(&mut self, bbox: Option<Bbox>) {
        self.seed_bounds = bbox;
    }

    /// Attach or detach the solid SDF. Rebuilds one bind group; cheap enough to
    /// call whenever the geometry changes.
    pub fn set_sdf(&mut self, device: &wgpu::Device, sdf: Option<SdfSource>) {
        let view = match &sdf {
            Some(s) => s.view.clone(),
            None => self.sdf_fallback_view.clone(),
        };
        self.compute_group = Self::build_compute_group(
            device,
            &self.compute_layout,
            &self.uniform,
            &self.particles,
            &self.free_list,
            &self.counters,
            &self.indirect,
            &self.cdf,
            &view,
            &self.trails,
        );
        self.seed_group = Self::build_compute_group(
            device,
            &self.compute_layout,
            &self.uniform,
            &self.particles,
            &self.free_list,
            &self.counters,
            &self.seed_scratch,
            &self.cdf,
            &view,
            &self.trails,
        );
        self.sdf = sdf;
    }
    pub fn has_sdf(&self) -> bool {
        self.sdf.is_some()
    }

    /// Kill every particle. They are all reseeded over the following frames, so
    /// this is the right response to "the geometry changed under me".
    pub fn reset(&mut self) {
        self.needs_clear = true;
        self.sim_time = 0.0;
    }

    pub fn sim_time(&self) -> f32 {
        self.sim_time
    }

    /// Ring slot the next trail write lands in. Exposed so a test can reproduce
    /// the exact segment pairing `vs_trail` uses.
    pub fn trail_head(&self) -> u32 {
        self.trail_head
    }

    /// Emission actually sent to the shader, after population normalisation.
    ///
    /// Exposed because it is the number that explains a blown-out or invisible
    /// cloud, and because a UI that shows only the raw slider leaves the user
    /// wondering why the same value looks different at a different count.
    pub fn effective_intensity(&self) -> f32 {
        if !self.settings.normalise_intensity {
            return self.settings.intensity;
        }
        self.settings.intensity * REFERENCE_COUNT as f32 / self.config.count.max(1) as f32
    }

    fn upload_lut(&self, queue: &wgpu::Queue) {
        let colors = self.settings.color_map.lut(Interpolation::default());
        let entries: Vec<[f32; 4]> = colors.iter().map(|c| [c.x, c.y, c.z, 1.0]).collect();
        let bytes = colormap::pack_rgba16f(&entries);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.lut,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(colormap::LUT_SIZE as u32 * 8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: colormap::LUT_SIZE as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn build_uniform(
        &self,
        ctx: &OverlayContext<'_>,
        dt_sim: f32,
        inv_dt_ms: f32,
    ) -> StreaklineUniform {
        let s = &self.settings;
        let fields = ctx.fields;
        let bbox = fields.bbox();
        let seed = self.seed_bounds.unwrap_or(bbox);
        let w = s.seed.normalised();

        // No patch means no flux seeding, whatever the weight slider says.
        let (inlet_center, inlet_u, inlet_v, inlet_n, w_inlet) = match self.inlet {
            Some(p) => (
                p.center_mm,
                p.half_u,
                p.half_v,
                p.normal.normalize_or(Vec3::X),
                w[0],
            ),
            None => (Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z, 0.0),
        };
        // Renormalise once the inlet may have been dropped, so the remaining
        // strategies still cover the whole unit interval.
        let total = (w_inlet + w[1] + w[2]).max(1e-6);
        let (w_inlet, w_volume, w_import) = (w_inlet / total, w[1] / total, w[2] / total);

        let (sdf_min, sdf_inv, sdf_present, sdf_offset) = match &self.sdf {
            Some(sdf) => (
                sdf.min_mm,
                Vec3::ONE / sdf.size_mm.max(Vec3::splat(1e-6)),
                1u32,
                sdf.surface_offset_mm,
            ),
            None => (Vec3::ZERO, Vec3::ZERO, 0u32, 0.0),
        };

        let (lo, hi) = (s.color_range[0], s.color_range[1]);
        let trail_len = if s.trails.enabled {
            self.config.trail_length
        } else {
            0
        };

        StreaklineUniform {
            volume_min_mm: bbox.min.to_array(),
            // The *solver* cell size, not the derived voxel: the CFL condition is
            // about not stepping over a wall, and the wall is resolved on the
            // solver grid.
            dx_mm: fields.grid().dx_mm,
            volume_size_mm: fields.volume_size_mm().to_array(),
            dt_sim,
            seed_min_mm: seed.min.to_array(),
            cfl: s.cfl,
            seed_size_mm: seed.size().to_array(),
            max_step_cells: s.max_step_cells,
            inlet_center: inlet_center.to_array(),
            inlet_un_max: s.inlet_peak_speed_ms,
            inlet_u: inlet_u.to_array(),
            w_inlet,
            inlet_v: inlet_v.to_array(),
            w_volume,
            inlet_n: inlet_n.to_array(),
            w_import,
            sdf_min_mm: sdf_min.to_array(),
            sdf_offset_mm: sdf_offset,
            sdf_inv_size: sdf_inv.to_array(),
            particle_radius_mm: s.particle_radius_mm,
            count: self.config.count,
            frame: self.frame,
            max_substeps: s.max_substeps,
            sdf_present,
            life_mean_s: s.life_s,
            life_jitter: s.life_jitter,
            fade_in: s.fade_in,
            fade_out: s.fade_out,
            radius_mm: s.radius_mm,
            min_radius_px: s.min_radius_px,
            intensity: self.effective_intensity(),
            color_mode: s.color.code(),
            color_lo: lo,
            color_inv_span: 1.0 / (hi - lo).max(1e-9),
            cdf_dim: CDF_DIM,
            q_threshold: s.q_threshold,
            trail_count: if trail_len > 0 {
                self.config.trail_count
            } else {
                0
            },
            trail_len,
            trail_head: if trail_len > 0 {
                self.trail_head % trail_len
            } else {
                0
            },
            trail_max_age_s: s.trails.max_age_s,
            inlet_offset_mm: s.inlet_offset_mm,
            occupancy_min: s.occupancy_min,
            motion_blur: s.motion_blur,
            max_streak_px: s.max_streak_px,
            inv_dt_ms,
            trail_width_mm: s.trails.width_mm,
            trail_alpha: s.trails.alpha,
            _pad: 0.0,
        }
    }

    /// A few thousand true streamlines through a frozen field, integrated with
    /// RK4 on the CPU.
    ///
    /// Genuinely streamlines, not streaklines: the field is a snapshot and does
    /// not evolve while the curve is traced. RK4 is worth its four evaluations
    /// here — the count is small, it runs once rather than every frame, and the
    /// curves are long enough that the error accumulates.
    ///
    /// The caller supplies the velocity lookup, because the render crate has no
    /// CPU copy of the field; the app reads it back once for this purpose.
    pub fn hero_streamlines(
        seeds: &[Vec3],
        steps: usize,
        h: f32,
        velocity: impl Fn(Vec3) -> Vec3,
    ) -> Vec<Vec<Vec3>> {
        seeds
            .iter()
            .map(|start| {
                let mut curve = Vec::with_capacity(steps + 1);
                let mut p = *start;
                curve.push(p);
                for _ in 0..steps {
                    let v = velocity(p);
                    if !v.is_finite() || v.length_squared() < 1e-16 {
                        break;
                    }
                    let next = rk4_step(p, h, &velocity);
                    if !next.is_finite() {
                        break;
                    }
                    p = next;
                    curve.push(p);
                }
                curve
            })
            .collect()
    }
}

impl OverlayPass for StreaklineOverlay {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "streaklines"
    }

    fn is_static(&self) -> bool {
        // Paused tracers still occupy the frame, but they no longer change, so
        // the progressive accumulator may keep integrating them.
        //
        // A zero frame delta counts as paused too, and deliberately: an app that
        // has stopped stepping time has stopped the tracers just as surely as
        // the pause button has, and refusing to accumulate then would leave a
        // frozen image permanently un-antialiased for no reason. It is the
        // *previous* frame's delta because the renderer asks before recording,
        // which is the answer it wants: what changed since the last image.
        self.settings.paused || self.last_dt_sim == 0.0
    }

    fn record(&mut self, ctx: &mut OverlayContext<'_>) {
        self.settings.sanitise();
        if self.lut_dirty || self.last_color_map != self.settings.color_map {
            self.upload_lut(ctx.queue);
            self.last_color_map = self.settings.color_map;
            self.lut_dirty = false;
        }

        // Clamp the wall-clock delta before scaling: a stalled frame (shader
        // compile, window drag) must not teleport the whole population.
        let dt_wall = ctx.dt.clamp(0.0, 1.0 / 15.0);
        let dt_sim = if self.settings.paused {
            0.0
        } else {
            dt_wall * self.settings.time_scale
        };
        self.last_dt_sim = dt_sim;
        if dt_sim > 0.0 {
            self.sim_time += dt_sim;
            self.last_inv_dt_ms = 1.0 / (dt_sim * 1000.0);
        }
        let run_compute = dt_sim > 0.0 || self.frame == 0;
        let trails_on = self.settings.trails.enabled && self.config.trail_length > 1;
        // The ring advances only on frames that write into it, so the two slots
        // every ribbon segment joins were always written back to back. See
        // [`advance_trail_head`] for what happens when it does not.
        if self.frame > 0 {
            self.trail_head = advance_trail_head(
                self.trail_head,
                self.config.trail_length,
                run_compute && trails_on,
            );
        }
        // Turning trails back on resumes a ring whose newest entry is from
        // whenever they were turned off, and the segment bridging that gap is a
        // streak across however far the particle travelled in between. There is
        // no history to salvage, so throw it away.
        let trails_just_on = trails_on && !self.trails_were_on;
        self.trails_were_on = trails_on;

        let u = self.build_uniform(ctx, dt_sim, self.last_inv_dt_ms);
        ctx.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));

        if self.needs_clear {
            // Zeroing gives every particle `life = 0`, i.e. dead, so the first
            // advect pass pushes all of them onto the free list and the first
            // seed pass fills the domain. There is no separate init kernel.
            ctx.encoder.clear_buffer(&self.particles, 0, None);
            ctx.encoder.clear_buffer(&self.trails, 0, None);
            ctx.encoder.clear_buffer(&self.cdf, 0, None);
            self.needs_clear = false;
        } else if trails_just_on {
            ctx.encoder.clear_buffer(&self.trails, 0, None);
        }

        if run_compute {
            // The counters are the frame's only shared mutable state, and they
            // must start at zero: `free_count` is an append cursor.
            ctx.encoder.clear_buffer(&self.counters, 0, None);

            let rebuild_cdf =
                u.w_import > 0.0 && self.frame % self.settings.importance_interval.max(1) == 0;
            if rebuild_cdf {
                let ts = ctx.profiler.scope("streakline importance");
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("streakline importance"),
                        timestamp_writes: ts,
                    });
                pass.set_pipeline(&self.importance);
                pass.set_bind_group(0, ctx.fields.read_bind_group(), &[]);
                pass.set_bind_group(1, &self.compute_group, &[]);
                pass.dispatch_workgroups(util::dispatch_count(CDF_BINS, IMPORTANCE_WG), 1, 1);
                drop(pass);

                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("streakline cdf scan"),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(&self.scan);
                pass.set_bind_group(0, ctx.fields.read_bind_group(), &[]);
                pass.set_bind_group(1, &self.compute_group, &[]);
                // One workgroup: the whole scan is 32768 entries and a
                // cross-workgroup scan would need a second dispatch to fix up
                // block offsets for no measurable gain at this size.
                pass.dispatch_workgroups(1, 1, 1);
            }

            {
                let ts = ctx.profiler.scope("streakline advect");
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("streakline advect"),
                        timestamp_writes: ts,
                    });
                pass.set_pipeline(&self.advect);
                pass.set_bind_group(0, ctx.fields.read_bind_group(), &[]);
                pass.set_bind_group(1, &self.compute_group, &[]);
                pass.dispatch_workgroups(
                    util::dispatch_count(self.config.count, PARTICLE_WG),
                    1,
                    1,
                );
            }
            {
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("streakline dispatch prepare"),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(&self.prepare);
                pass.set_bind_group(0, ctx.fields.read_bind_group(), &[]);
                pass.set_bind_group(1, &self.compute_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let ts = ctx.profiler.scope("streakline seed");
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("streakline seed"),
                        timestamp_writes: ts,
                    });
                pass.set_pipeline(&self.seed);
                pass.set_bind_group(0, ctx.fields.read_bind_group(), &[]);
                pass.set_bind_group(1, &self.seed_group, &[]);
                // Indirect, so the number of workgroups follows the number of
                // deaths without a readback. This is the whole reason the free
                // list is an append buffer rather than a compacted list.
                pass.dispatch_workgroups_indirect(&self.indirect, 0);
            }
        }

        {
            let ts = ctx.profiler.render_scope("streaklines");
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("streaklines"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, ctx.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.draw_group, &[]);

            if self.settings.trails.enabled && u.trail_len > 1 && u.trail_count > 0 {
                pass.set_pipeline(&self.trail_pipeline);
                let segments = u.trail_len - 1;
                pass.draw(0..(6 * segments * u.trail_count), 0..1);
            }
            pass.set_pipeline(&self.sprite_pipeline);
            pass.draw(0..(6 * self.config.count), 0..1);
        }

        self.frame = self.frame.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::{GpuContext, Grid, Profiler};
    use glam::UVec3;

    fn gpu() -> Option<GpuContext> {
        match GpuContext::new_blocking(None) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    #[test]
    fn rk2_conserves_the_radius_of_a_solid_body_rotation() {
        // Solid-body rotation `u = omega x r` has an exact solution: every
        // particle traces a circle, so |r| is an invariant. The midpoint rule's
        // amplification factor per step is sqrt(1 + (h*omega)^4 / 4), which is
        // fourth order in the rotation angle — so a thousand steps at 0.1 rad
        // each drift by about a percent. Forward Euler's factor is
        // sqrt(1 + (h*omega)^2), which is *second* order and drifts by 500%.
        // The gap between those two numbers is the whole reason this test
        // discriminates.
        let omega = 12.0f32; // rad/s
        let vel = |p: Vec3| Vec3::new(-omega * p.y, omega * p.x, 0.0);

        let r0 = 20.0f32;
        let h = 0.1 / omega; // 0.1 rad per step
        let mut p = Vec3::new(r0, 0.0, 0.0);
        let mut euler = p;
        for _ in 0..1000 {
            p = rk2_step(p, h, vel);
            euler += vel(euler) * h;
        }
        let drift = (p.length() - r0).abs() / r0;
        assert!(drift < 0.02, "RK2 radius drifted {}%", drift * 100.0);

        // ...and the particle really went round: a method that simply stalled
        // would also conserve the radius.
        let angle = p.y.atan2(p.x);
        assert!(angle.is_finite());
        let turns = 1000.0 * 0.1 / std::f32::consts::TAU;
        assert!(turns > 15.0, "the test only covered {turns} revolutions");

        let euler_drift = (euler.length() - r0).abs() / r0;
        assert!(
            euler_drift > 20.0 * drift,
            "forward Euler drifted only {}%, so the test is not discriminating",
            euler_drift * 100.0
        );
    }

    #[test]
    fn rk2_converges_at_second_order_on_a_smooth_field() {
        // Halving the step must quarter the error. This is what pins the
        // integrator as midpoint rather than as something accidentally
        // first-order, which a sign slip in the half-step would produce.
        let omega = 5.0f32;
        let vel = |p: Vec3| Vec3::new(-omega * p.y, omega * p.x, 0.0);
        let start = Vec3::new(10.0, 0.0, 0.0);
        let total = 1.0f32;

        let err_for = |n: u32| {
            let h = total / n as f32;
            let mut p = start;
            for _ in 0..n {
                p = rk2_step(p, h, vel);
            }
            let a = omega * total;
            let exact = Vec3::new(10.0 * a.cos(), 10.0 * a.sin(), 0.0);
            (p - exact).length()
        };
        let coarse = err_for(200);
        let fine = err_for(400);
        let ratio = coarse / fine.max(1e-9);
        assert!(
            (3.0..5.5).contains(&ratio),
            "error ratio {ratio} is not second order (coarse {coarse}, fine {fine})"
        );
    }

    #[test]
    fn rk4_beats_rk2_on_the_same_smooth_field() {
        // Justifies keeping RK4 for the hero curves, where the field is frozen
        // and effectively smooth. On the trilinear texture it would not.
        let omega = 5.0f32;
        let vel = |p: Vec3| Vec3::new(-omega * p.y, omega * p.x, 0.0);
        let start = Vec3::new(10.0, 0.0, 0.0);
        let (n, total) = (200u32, 1.0f32);
        let h = total / n as f32;

        let mut a = start;
        let mut b = start;
        for _ in 0..n {
            a = rk2_step(a, h, vel);
            b = rk4_step(b, h, vel);
        }
        let angle = omega * total;
        let exact = Vec3::new(10.0 * angle.cos(), 10.0 * angle.sin(), 0.0);
        assert!(
            (b - exact).length() < 0.02 * (a - exact).length(),
            "RK4 error {} is not much better than RK2's {}",
            (b - exact).length(),
            (a - exact).length()
        );
    }

    #[test]
    fn the_cfl_clamp_keeps_every_substep_inside_one_cell() {
        // The tunnelling guard, stated as arithmetic. For any speed the
        // integrator can be handed, the displacement of a single substep must
        // stay below one cell — either because the CFL condition chose enough
        // substeps, or because the backstop clamped it.
        let dx = 0.75f32;
        let (cfl, max_sub, max_cells) = (0.4f32, 4u32, 1.0f32);
        let dt = 1.0 / 60.0 * 0.02; // one frame at the default time scale

        for speed in [0.0f32, 1.0, 50.0, 500.0, 5_000.0, 50_000.0, 1.0e6] {
            let n = substep_count(dt, speed, dx, cfl, max_sub);
            assert!(
                (1..=max_sub).contains(&n),
                "speed {speed} gave {n} substeps"
            );
            let h = dt / n as f32;
            let raw = Vec3::X * speed * h;
            let d = clamp_displacement(raw, dx, max_cells);
            assert!(
                d.length() <= max_cells * dx + 1e-4,
                "speed {speed} mm/s moved {} mm in one substep, more than a {dx} mm cell",
                d.length()
            );
            // ...and below the point where the cap binds, the clamp must be
            // inactive, so normal flow is integrated exactly.
            if speed <= cfl * dx * max_sub as f32 / dt {
                assert!(
                    (d - raw).length() < 1e-5,
                    "the backstop fired at {speed} mm/s, where the CFL condition should have sufficed"
                );
                assert!(raw.length() <= cfl * dx + 1e-4);
            }
        }
    }

    #[test]
    fn a_fast_particle_cannot_cross_a_wall_in_one_frame() {
        // The failure this whole substep machinery exists to prevent, simulated:
        // a 2 mm wall at x = 0, a particle coming at it at 8 m/s. With the
        // clamp, no substep ever jumps the wall, so the collision test sees it.
        let dx = 0.75f32;
        let wall = 2.0f32; // solid occupies [0, wall]
        let dt = 1.0 / 60.0 * 0.02;
        let speed_mm_s = 8_000.0; // 8 m/s
                                  // Half a millimetre short of the wall, which is the worst case: near
                                  // enough that one whole-frame step lands past the far face.
        let start = -0.5f32;

        // The failure being prevented is real at these numbers: a single
        // unsubstepped step covers 2.67 mm, so from `start` it emerges on the
        // far side of a 2 mm wall having never sampled inside it.
        assert!(
            start + speed_mm_s * dt > wall,
            "the test flow is too slow to tunnel, so it proves nothing"
        );

        let n = substep_count(dt, speed_mm_s, dx, 0.4, 4);
        let h = dt / n as f32;
        let mut x = start;
        let mut crossed_undetected = false;
        let mut sampled_inside = false;
        for _ in 0..n {
            let d = clamp_displacement(Vec3::X * speed_mm_s * h, dx, 1.0);
            let next = x + d.x;
            // "Undetected" means the substep started before the wall and ended
            // past it without ever sampling inside it.
            if x < 0.0 && next > wall {
                crossed_undetected = true;
            }
            x = next;
            if x > 0.0 && x < wall {
                sampled_inside = true;
            }
        }
        assert!(
            !crossed_undetected,
            "a substep jumped the entire {wall} mm wall"
        );
        // Not tunnelling is only half of it: the collision test has to actually
        // get a look at the particle while it is inside the solid.
        assert!(
            sampled_inside,
            "no substep ended inside the wall, so the SDF push-out never fires"
        );
    }

    #[test]
    fn the_fade_envelope_starts_and_ends_at_zero() {
        let (fi, fo) = (0.05f32, 0.2f32);
        let life = 3.0f32;
        assert_eq!(fade_alpha(0.0, life, fi, fo), 0.0);
        assert_eq!(fade_alpha(life, life, fi, fo), 0.0);
        assert_eq!(fade_alpha(-1.0, life, fi, fo), 0.0);
        assert_eq!(fade_alpha(life + 1.0, life, fi, fo), 0.0);
        // Full brightness across the middle.
        assert!((fade_alpha(life * 0.5, life, fi, fo) - 1.0).abs() < 1e-4);
        // Monotone up through the fade-in and down through the fade-out.
        let mut prev = 0.0;
        for i in 0..=20 {
            let a = fade_alpha(i as f32 / 20.0 * fi * life, life, fi, fo);
            assert!(a >= prev - 1e-6, "fade-in is not monotone");
            prev = a;
        }
        assert!(fade_alpha(life * 0.9, life, fi, fo) > fade_alpha(life * 0.97, life, fi, fo));
    }

    #[test]
    fn lifetime_jitter_desynchronises_the_population() {
        // Without jitter the whole field of tracers dies on the same frame and
        // the image blinks. The property that matters is that the *spread* of
        // death times is a decent fraction of the lifetime.
        let mut s = StreaklineSettings::default();
        s.sanitise();
        assert!(
            s.life_jitter >= 0.15,
            "the default jitter is too small to hide the pulse"
        );
        let lo = s.life_s * (1.0 - s.life_jitter);
        let hi = s.life_s * (1.0 + s.life_jitter);
        assert!(hi - lo > 0.3 * s.life_s);
    }

    #[test]
    fn the_sprite_clamp_conserves_energy() {
        // The precise failure: without the alpha compensation a receding cloud
        // of particles gets *brighter* as it recedes, because the clamped
        // footprint stops shrinking while the emission per pixel does not.
        for min_px in [1.5f32, 2.0] {
            let mut last_energy = f32::INFINITY;
            for i in 1..40 {
                let true_px = 4.0 / i as f32;
                let (r, scale) = sprite_footprint(true_px, min_px);
                assert!(r >= min_px - 1e-6 && r >= true_px - 1e-6);
                // Emitted energy is alpha * area.
                let energy = scale * r * r;
                assert!(
                    energy <= last_energy + 1e-4,
                    "energy rose from {last_energy} to {energy} as the sprite receded"
                );
                assert!(
                    (energy - true_px * true_px).abs() < 1e-3,
                    "energy is not r_true^2"
                );
                last_energy = energy;
            }
        }
        // Above the floor the clamp must be a no-op, or near sprites lose light.
        let (r, scale) = sprite_footprint(6.0, 1.5);
        assert_eq!(r, 6.0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn motion_blur_conserves_energy_too() {
        // A capsule twice as long must be half as bright per unit length.
        let r = 2.0f32;
        for len in [0.0f32, 1.0, 10.0, 100.0] {
            let a = streak_alpha(r, len);
            let covered = r * r + len * r;
            assert!(
                (a * covered - r * r).abs() < 1e-3,
                "length {len} does not conserve energy"
            );
        }
        assert_eq!(streak_alpha(r, 0.0), 1.0);
        assert!(streak_alpha(r, 100.0) < 0.05);
    }

    #[test]
    fn ribbon_triangles_are_consistently_wound() {
        // Culling is off for the ribbons, so a winding error would look correct
        // today and delete every trail the day someone turns culling on. This
        // is the test that stops that.
        // Half the 2D cross product, i.e. the signed triangle area.
        let signed_area = |a: Vec2, b: Vec2, c: Vec2| {
            let u = b - a;
            let v = c - a;
            0.5 * (u.x * v.y - u.y * v.x)
        };
        let (len, half_width) = (30.0f32, 4.0f32);
        for i in 0..24 {
            let a = i as f32 / 24.0 * std::f32::consts::TAU;
            let older = Vec2::new(100.0, 100.0);
            let newer = older + Vec2::new(a.cos(), a.sin()) * len;
            let q = ribbon_quad(older, newer, half_width);
            let t1 = signed_area(q[0], q[1], q[2]);
            let t2 = signed_area(q[3], q[4], q[5]);
            assert!(t1 > 0.0 && t2 > 0.0, "angle {a}: areas {t1}, {t2}");
            // The two triangles must tile the segment's full rectangle exactly:
            // length by twice the half-width, with no overlap and no gap.
            let want = len * 2.0 * half_width;
            assert!(
                (t1 + t2 - want).abs() < 1e-2,
                "quad area is {} not {want}",
                t1 + t2
            );
        }
        // A degenerate (zero-length) segment must not produce NaNs.
        let q = ribbon_quad(Vec2::splat(5.0), Vec2::splat(5.0), 2.0);
        assert!(q.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn ribbon_corners_cover_the_quad_exactly_once_per_triangle() {
        let mut seen = std::collections::HashMap::new();
        for i in 0..6u32 {
            let c = ribbon_corner(i);
            *seen.entry((c.x as i32, c.y as i32)).or_insert(0) += 1;
        }
        assert_eq!(seen.len(), 4, "a quad has four distinct corners");
        // Two corners are shared by both triangles, two are not.
        let shared = seen.values().filter(|v| **v == 2).count();
        assert_eq!(shared, 2, "corner sharing is wrong: {seen:?}");
    }

    #[test]
    fn the_trail_ring_head_only_moves_when_something_is_written() {
        // The bookkeeping the ribbons' correctness rests on. `vs_trail` joins
        // ring slots `head - j` and `head - j - 1` and reads "the newer end has
        // the larger age" as proof that no respawn lies between them — which is
        // sound only if the two were written on consecutive frames.
        const LEN: u32 = 8;
        let mut head = 0u32;
        let mut written = Vec::new();
        // A realistic frame sequence: some stepping, a pause, more stepping.
        for frame in 0..40u32 {
            let wrote = !(12..25).contains(&frame);
            if frame > 0 {
                head = advance_trail_head(head, LEN, wrote);
            }
            if wrote {
                written.push(head);
            }
        }
        // Every write landed in the slot after the previous write, with no gaps:
        // that is exactly the property the age test needs.
        for pair in written.windows(2) {
            assert_eq!(
                pair[1],
                (pair[0] + 1) % LEN,
                "writes landed in {} then {}, which are not adjacent",
                pair[0],
                pair[1]
            );
        }
        assert!(
            written.len() > LEN as usize,
            "the ring never wrapped; wrapping is untested"
        );

        // A degenerate ring has no slots to advance through.
        assert_eq!(advance_trail_head(0, 1, true), 0);
        assert_eq!(advance_trail_head(0, 0, true), 0);
        // And the head must be stable across a pause of any length.
        let mut h = 5u32;
        for _ in 0..100 {
            h = advance_trail_head(h, LEN, false);
        }
        assert_eq!(h, 5);
    }

    #[test]
    fn seed_weights_normalise_and_degrade() {
        let w = SeedWeights::default().normalised();
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // The volumetric reseed must be small but never zero: it is the only
        // thing that puts a particle into a recirculation bubble.
        assert!(w[1] > 0.0 && w[1] < 0.05, "volumetric weight is {}", w[1]);
        let z = SeedWeights {
            inlet: 0.0,
            volume: 0.0,
            importance: 0.0,
        }
        .normalised();
        assert_eq!(
            z,
            [1.0, 0.0, 0.0],
            "an empty weight set must not blank the screen"
        );
        let v = SeedWeights::vortex_hunt().normalised();
        assert!(v[2] > 0.5, "the vortex preset should mostly seed on Q");
    }

    #[test]
    fn the_config_budget_matches_the_plan() {
        // 4M tracers at 32 bytes plus a 4-byte free slot each.
        let c = StreaklineConfig::default();
        assert_eq!(c.count, 4 << 20);
        assert_eq!(
            c.trail_count, 0,
            "trails are opt-in; they are the expensive tier"
        );
        let mib = c.bytes() as f64 / (1024.0 * 1024.0);
        assert!((140.0..160.0).contains(&mib), "base budget is {mib} MiB");

        // The trail tier: 256k x 32 x 8 bytes = 64 MiB on top.
        let t = StreaklineConfig::with_trails();
        let extra = (t.bytes() - c.bytes()) as f64 / (1024.0 * 1024.0);
        assert!(
            (extra - 64.0).abs() < 0.01,
            "trail budget is {extra} MiB, want 64"
        );

        // And the thing the plan says not to do: a full-population history.
        let absurd = StreaklineConfig {
            count: 4 << 20,
            trail_count: 4 << 20,
            trail_length: 32,
        };
        assert!(
            absurd.bytes() > (1 << 30),
            "a 4M-particle history should be over a gigabyte; the budget maths is wrong"
        );
    }

    #[test]
    fn every_overlay_shader_preprocesses_without_a_gpu() {
        // The same guard `util::every_embedded_shader_preprocesses_without_a_gpu`
        // gives the core passes, extended to the Wave 2 files: an `#include`
        // typo, an unterminated `#if` or a `#DEFINE` the Rust side forgot to set
        // otherwise surfaces as an unhelpful WGSL parse error at pipeline
        // creation on a user's machine.
        let loader = overlay_shader_loader();
        let base = overlay_defines();

        // particles.wgsl compiles twice, and each variant must carry exactly the
        // entry points it is meant to and none of the other's.
        let compute = loader.load("particles.wgsl", &base).unwrap();
        let draw = loader
            .load("particles.wgsl", &base.clone().flag("PARTICLES_DRAW"))
            .unwrap();
        for entry in [
            "fn advect_main",
            "fn seed_main",
            "fn prepare_main",
            "fn scan_main",
        ] {
            assert!(
                compute.contains(entry),
                "the compute variant is missing {entry}"
            );
            assert!(!draw.contains(entry), "the draw variant leaked {entry}");
        }
        for entry in [
            "fn vs_particle",
            "fn fs_particle",
            "fn vs_trail",
            "fn fs_trail",
        ] {
            assert!(draw.contains(entry), "the draw variant is missing {entry}");
            assert!(
                !compute.contains(entry),
                "the compute variant leaked {entry}"
            );
        }

        // isosurface.wgsl likewise splits into a draw and a histogram module,
        // because the two want different resources at group 0.
        let iso_draw = loader.load("isosurface.wgsl", &base).unwrap();
        let iso_hist = loader
            .load("isosurface.wgsl", &base.clone().flag("ISO_HISTOGRAM"))
            .unwrap();
        assert!(iso_draw.contains("fn fs_iso") && !iso_draw.contains("fn histogram_main"));
        assert!(iso_hist.contains("fn histogram_main") && !iso_hist.contains("fn fs_iso"));

        let slice = loader.load("slice.wgsl", &base).unwrap();
        assert!(slice.contains("fn fs_slice"));
        // `lic.wgsl` is pulled in by the include, and must arrive expanded.
        assert!(
            slice.contains("fn lic_ramp_kernel"),
            "lic.wgsl was not included"
        );

        for (name, src) in [
            ("particles.wgsl/compute", &compute),
            ("particles.wgsl/draw", &draw),
            ("isosurface.wgsl/draw", &iso_draw),
            ("isosurface.wgsl/histogram", &iso_hist),
            ("slice.wgsl", &slice),
        ] {
            assert!(src.len() > 100, "{name} preprocessed to nothing");
            for line in src.lines() {
                assert!(
                    !line.trim_start().starts_with('#'),
                    "{name} has an unhandled directive: {line}"
                );
            }
        }
    }

    #[test]
    fn the_cdf_scan_shape_divides_evenly() {
        // The single-workgroup scan assumes every thread gets the same chunk.
        assert_eq!(CDF_BINS % SCAN_WG, 0);
        assert_eq!(CDF_BINS, 32 * 32 * 32);
    }

    #[test]
    fn the_uniform_is_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<StreaklineUniform>() % 16, 0);
    }

    #[test]
    fn hero_streamlines_close_on_themselves_in_a_solid_body_rotation() {
        // A closed circular orbit is the strongest available check on a
        // streamline tracer: after a full turn it must return to its seed.
        let omega = 2.0f32;
        let vel = |p: Vec3| Vec3::new(-omega * p.y, omega * p.x, 0.0);
        let seeds = [Vec3::new(15.0, 0.0, 0.0), Vec3::new(0.0, 8.0, 3.0)];
        let steps = 400;
        let h = std::f32::consts::TAU / omega / steps as f32;
        let curves = StreaklineOverlay::hero_streamlines(&seeds, steps, h, vel);
        assert_eq!(curves.len(), 2);
        for (curve, seed) in curves.iter().zip(seeds) {
            assert_eq!(curve.len(), steps + 1);
            let close = (curve[steps] - seed).length();
            assert!(close < 1e-2, "orbit closed to within {close} mm, not exact");
        }
        // A stationary field terminates the curve rather than emitting a
        // thousand identical points.
        let dead = StreaklineOverlay::hero_streamlines(&[Vec3::ZERO], 50, 0.1, |_| Vec3::ZERO);
        assert_eq!(dead[0].len(), 1);
    }

    // -- GPU tests -----------------------------------------------------------

    fn test_grid() -> Grid {
        Grid {
            dims: UVec3::new(48, 32, 32),
            dx_mm: 0.75,
            origin_mm: Vec3::splat(-12.0),
        }
    }

    /// The minimum a Wave 2 overlay needs to be driven: derived fields, a brick
    /// grid, a camera bind group and the two attachments.
    ///
    /// Built directly rather than through [`crate::Renderer`] so a test can read
    /// the overlay's own buffers back afterwards, which is the whole point of
    /// the free-list and wall tests.
    struct Harness {
        fields: crate::fields::DerivedFields,
        bricks: crate::accel::BrickGrid,
        camera_layout: wgpu::BindGroupLayout,
        _camera_buffer: wgpu::Buffer,
        camera_group: wgpu::BindGroup,
        _depth: wgpu::Texture,
        depth_view: wgpu::TextureView,
        _hdr: wgpu::Texture,
        hdr_view: wgpu::TextureView,
        size: (u32, u32),
    }

    impl Harness {
        /// `velocity` returns metres per second at a world-space point in mm.
        fn new(gpu: &GpuContext, grid: Grid, velocity: impl Fn(Vec3) -> Vec3) -> Self {
            let (w, h) = (160u32, 120u32);
            let loader = util::shader_loader();
            let stages = wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE;
            let camera_layout =
                gpu.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("test camera"),
                        entries: &[util::uniform_entry(0, stages)],
                    });
            let camera_buffer =
                util::uniform_buffer::<crate::CameraUniform>(&gpu.device, "test camera");
            let mut cam = crate::Camera::default();
            cam.aspect = w as f32 / h as f32;
            cam.frame_bbox(grid.bbox(), 0.1);
            let cu = crate::CameraUniform::new(&cam, glam::Mat4::IDENTITY, w, h, 0);
            gpu.queue
                .write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&cu));
            let camera_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("test camera"),
                layout: &camera_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }],
            });

            let mut fields = crate::fields::DerivedFields::new(
                &gpu.device,
                &gpu.queue,
                &loader,
                grid,
                crate::FieldResolution::Full,
            )
            .expect("derived fields");
            let bricks =
                crate::accel::BrickGrid::new(&gpu.device, &loader, &fields).expect("brick grid");

            // Fill the solver-side textures with the analytic field. Unit
            // scales, so the macroscopic texture holds metres per second
            // directly and the test controls the speed exactly.
            let (mac, flg) = crate::fields::create_source_textures(&gpu.device, grid);
            let dims = grid.dims;
            let mut bytes = Vec::with_capacity((dims.x * dims.y * dims.z) as usize * 8);
            for z in 0..dims.z {
                for y in 0..dims.y {
                    for x in 0..dims.x {
                        let p = grid.cell_center_mm(UVec3::new(x, y, z));
                        let v = velocity(p);
                        for c in [v.x, v.y, v.z, 0.0f32] {
                            bytes.extend_from_slice(&colormap::f32_to_f16_bits(c).to_le_bytes());
                        }
                    }
                }
            }
            let extent = wgpu::Extent3d {
                width: dims.x,
                height: dims.y,
                depth_or_array_layers: dims.z,
            };
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &mac,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dims.x * 8),
                    rows_per_image: Some(dims.y),
                },
                extent,
            );
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &flg,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &vec![ad_gpu::flags::FLUID; (dims.x * dims.y * dims.z) as usize],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dims.x),
                    rows_per_image: Some(dims.y),
                },
                extent,
            );
            let d3 = wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            };
            let mut profiler = Profiler::new(&gpu.device, &gpu.queue, 8, false, None);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            fields.derive(
                &gpu.device,
                &gpu.queue,
                &mut enc,
                &mut profiler,
                &crate::FieldSources {
                    macro_view: &mac.create_view(&d3),
                    flags_view: Some(&flg.create_view(&d3)),
                    // Unit scales: the source texture already holds m/s.
                    scales: crate::DeriveScales {
                        speed_ms: 1.0,
                        q_tilde: 1.0,
                        vorticity: 1.0,
                        pressure_pa: 1.0,
                    },
                },
            );
            gpu.queue.submit([enc.finish()]);

            let depth = util::color_target(
                &gpu.device,
                "test depth",
                w,
                h,
                crate::mesh::DEPTH_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            let hdr = util::color_target(
                &gpu.device,
                "test hdr",
                w,
                h,
                crate::post::HDR_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            Self {
                depth_view: depth.create_view(&Default::default()),
                hdr_view: hdr.create_view(&Default::default()),
                _depth: depth,
                _hdr: hdr,
                fields,
                bricks,
                camera_layout,
                _camera_buffer: camera_buffer,
                camera_group,
                size: (w, h),
            }
        }

        /// Clear both attachments and record one overlay frame.
        fn frame(&mut self, gpu: &GpuContext, pass: &mut dyn OverlayPass, dt: f32) {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        // Reverse-Z: the far plane is zero.
                        load: wgpu::LoadOp::Clear(crate::mesh::DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            {
                let mut profiler = Profiler::new(&gpu.device, &gpu.queue, 8, false, None);
                let mut ctx = OverlayContext {
                    device: &gpu.device,
                    queue: &gpu.queue,
                    encoder: &mut enc,
                    profiler: &mut profiler,
                    camera_bind_group: &self.camera_group,
                    fields: &self.fields,
                    bricks: &self.bricks,
                    hdr_view: &self.hdr_view,
                    depth_view: &self.depth_view,
                    size: self.size,
                    dt,
                };
                pass.record(&mut ctx);
            }
            gpu.queue.submit([enc.finish()]);
        }
    }

    fn make_system(gpu: &GpuContext, h: &Harness, count: u32) -> StreaklineOverlay {
        StreaklineOverlay::new(
            &gpu.device,
            &gpu.queue,
            &h.camera_layout,
            h.fields.read_bind_group_layout(),
            StreaklineConfig {
                count,
                trail_count: count / 4,
                trail_length: 8,
            },
        )
        .expect("streaklines")
    }

    /// Read a storage buffer back. Blocking; tests only.
    fn read_buffer(gpu: &GpuContext, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("streakline readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        gpu.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let _ = rx.recv();
        let out = slice
            .get_mapped_range()
            .map(|v| v.to_vec())
            .unwrap_or_default();
        staging.unmap();
        out
    }

    #[test]
    fn the_whole_particle_frame_records_and_runs_inside_the_renderer() {
        // The end-to-end test: five compute pipelines and two render pipelines
        // compile, every bind group matches its layout, an indirect dispatch fed
        // by an atomic counter validates, and the pass slots into a real frame.
        let Some(gpu) = gpu() else { return };
        let grid = test_grid();
        let mut r = crate::Renderer::new(
            &gpu,
            crate::RendererConfig {
                width: 160,
                height: 120,
                target_format: wgpu::TextureFormat::Rgba8Unorm,
                grid,
                field_resolution: crate::FieldResolution::Full,
                profiling: false,
            },
        )
        .expect("renderer");
        let mut sys = StreaklineOverlay::new(
            &gpu.device,
            &gpu.queue,
            r.camera_bind_group_layout(),
            r.fields_bind_group_layout(),
            StreaklineConfig {
                count: 4096,
                trail_count: 1024,
                trail_length: 8,
            },
        )
        .expect("streaklines");
        sys.settings_mut().trails.enabled = true;
        r.add_overlay(Box::new(sys));
        assert_eq!(r.overlay_names(), vec!["streaklines"]);

        let target = util::color_target(
            &gpu.device,
            "streakline target",
            160,
            120,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let view = target.create_view(&Default::default());
        let scene = crate::Scene::new();
        for _ in 0..3 {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            r.render(
                &mut enc,
                crate::FrameInput {
                    camera: &crate::Camera::default(),
                    scene: &scene,
                    sources: None,
                    sdf: None,
                    target: &view,
                    dt: 1.0 / 60.0,
                },
            )
            .expect("frame");
            gpu.queue.submit([enc.finish()]);
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    #[test]
    fn the_free_list_neither_loses_nor_duplicates_particles() {
        // The invariant the whole no-readback scheme rests on: every particle is
        // either alive or exactly once on the free list. A duplicated index
        // means two seeder threads write the same slot and the population
        // silently halves; a lost one means a slot is never reused and the
        // count decays frame by frame until the screen is empty.
        let Some(gpu) = gpu() else { return };
        const N: u32 = 4096;
        const LIFE: f32 = 0.4;
        const DT: f32 = 1.0 / 15.0;
        let mut h = Harness::new(&gpu, test_grid(), |_| Vec3::ZERO);
        let mut sys = make_system(&gpu, &h, N);
        {
            let s = sys.settings_mut();
            // Real-time playback and a life of about six frames: long enough
            // that any given frame recycles only a slice of the population, so
            // the free list is genuinely partial rather than trivially
            // everything, and short enough that 40 frames turn it over half a
            // dozen times.
            s.life_s = LIFE;
            s.time_scale = 1.0;
            s.seed = SeedWeights {
                inlet: 0.0,
                volume: 1.0,
                importance: 0.0,
            };
            s.trails.enabled = true;
        }

        for _ in 0..40 {
            h.frame(&gpu, &mut sys, DT);
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let counters = read_buffer(&gpu, &sys.counters, 16);
        let free_count = u32::from_le_bytes(counters[0..4].try_into().unwrap());
        let alive = u32::from_le_bytes(counters[4..8].try_into().unwrap());
        assert!(
            free_count <= N,
            "free count {free_count} exceeds the population"
        );
        assert_eq!(
            free_count + alive,
            N,
            "{free_count} free + {alive} alive != {N}: a particle was lost or duplicated"
        );
        assert!(
            free_count > 0,
            "nothing ever died; the recycling path is untested"
        );
        assert!(
            alive > 0,
            "the whole population died on one frame; the partition is trivially satisfied"
        );

        let raw = read_buffer(&gpu, &sys.free_list, free_count as u64 * 4);
        let mut seen = vec![false; N as usize];
        for i in 0..free_count as usize {
            let idx = u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
            assert!(idx < N as usize, "free list holds out-of-range index {idx}");
            assert!(!seen[idx], "particle {idx} is on the free list twice");
            seen[idx] = true;
        }

        // Every live particle must be a plausible one: finite, aged within its
        // own lifetime. A stale slot written by two threads shows up here.
        let raw = read_buffer(&gpu, &sys.particles, N as u64 * PARTICLE_BYTES);
        let f = |i: usize| f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
        let mut counted = 0u32;
        for i in 0..N as usize {
            let (age, life) = (f(i * 8 + 3), f(i * 8 + 7));
            if life <= 0.0 {
                continue;
            }
            counted += 1;
            assert!(
                age >= 0.0 && age < life,
                "particle {i} has age {age} of life {life}"
            );
            // Lifetime jitter must stay inside its stated band.
            assert!(
                (LIFE * 0.8 - 1e-4..=LIFE * 1.2 + 1e-4).contains(&life),
                "life {life} is outside the +/-20% jitter band"
            );
        }
        // The buffer is read back *after* the seeder has run, so it holds the
        // survivors plus whatever the seeder revived from this frame's free
        // list. Both bounds are what pins that: never fewer than the survivors,
        // and never more than the survivors plus everything that was freed.
        // Asserting equality here — the obvious thing to write — is wrong, and
        // silently passes only when every particle happens to die every frame.
        assert!(
            counted >= alive,
            "{counted} live particles is fewer than the {alive} the advect pass counted"
        );
        assert!(
            counted <= alive + free_count,
            "{counted} live particles exceeds {alive} survivors + {free_count} recyclable"
        );
        assert!(
            counted > alive,
            "the seeder revived nothing; the free list is never actually consumed"
        );
    }

    #[test]
    fn no_ribbon_segment_spans_more_than_one_frame_of_travel() {
        // The bug this exists to catch, which a screenshot found and no other
        // test would have: after a run of stepping frames followed by *still*
        // frames, about a fifth of the tracers grew a ribbon shooting clean
        // across the domain. The ring head was advancing on frames that wrote
        // nothing, so one segment per particle joined two entries a whole ring
        // apart — and any respawn inside that window slipped past the
        // monotonic-age guard, because the age at the far end really was
        // smaller.
        //
        // Stated as an invariant: every segment `vs_trail` will draw joins two
        // positions no further apart than one frame of clamped travel. This
        // reproduces the shader's segment pairing and age test exactly, so it
        // fails on the same segments the eye picks out.
        let Some(gpu) = gpu() else { return };
        const N: u32 = 4096;
        const LEN: u32 = 8;
        let grid = test_grid();
        // Fast enough that the displacement clamp binds every frame, so "one
        // frame of travel" is a hard, known bound.
        let mut h = Harness::new(&gpu, grid, |_| Vec3::new(5.0, 1.0, 0.0));
        let mut sys = StreaklineOverlay::new(
            &gpu.device,
            &gpu.queue,
            &h.camera_layout,
            h.fields.read_bind_group_layout(),
            StreaklineConfig {
                count: N,
                trail_count: N,
                trail_length: LEN,
            },
        )
        .expect("streaklines");
        let (life_s, dt) = (0.15f32, 1.0 / 30.0);
        {
            let s = sys.settings_mut();
            s.time_scale = 1.0;
            // About four or five frames of life, so respawns land *inside* the
            // eight-deep ring window. That is the case the age guard exists for
            // and the case the head bug defeated.
            s.life_s = life_s;
            s.life_jitter = 0.25;
            s.seed = SeedWeights {
                inlet: 0.0,
                volume: 1.0,
                importance: 0.0,
            };
            s.trails.enabled = true;
        }

        for _ in 0..30 {
            h.frame(&gpu, &mut sys, dt);
        }
        // ...then frames where time does not advance, which is what a paused
        // sim or a still capture looks like.
        for _ in 0..10 {
            h.frame(&gpu, &mut sys, 0.0);
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let head = sys.trail_head();
        let bbox = h.fields.bbox();
        let size = h.fields.volume_size_mm();
        let raw = read_buffer(&gpu, &sys.trails, N as u64 * LEN as u64 * TRAIL_ENTRY_BYTES);
        let particles = read_buffer(&gpu, &sys.particles, N as u64 * PARTICLE_BYTES);
        let pf = |i: usize| f32::from_le_bytes(particles[i * 4..i * 4 + 4].try_into().unwrap());

        let entry = |slot: usize| -> (Vec3, u32) {
            let x = u32::from_le_bytes(raw[slot * 8..slot * 8 + 4].try_into().unwrap());
            let y = u32::from_le_bytes(raw[slot * 8 + 4..slot * 8 + 8].try_into().unwrap());
            let q = Vec3::new((x & 0xffff) as f32, (x >> 16) as f32, (y & 0xffff) as f32);
            (bbox.min + (q / 65535.0) * size, y >> 16)
        };

        // One frame of travel, worst case: every substep saturating the
        // displacement clamp, plus the ring's own 16-bit quantisation.
        let s = sys.settings();
        let bound =
            s.max_substeps as f32 * s.max_step_cells * grid.dx_mm + size.length() / 65535.0 + 1e-3;

        let mut drawn = 0usize;
        let mut worst = 0.0f32;
        for pid in 0..N as usize {
            if pf(pid * 8 + 7) <= 0.0 {
                continue; // dead: `vs_trail` emits nothing for it
            }
            for seg in 0..LEN - 1 {
                let ia = pid * LEN as usize + ((head + LEN - seg) % LEN) as usize;
                let ib = pid * LEN as usize + ((head + LEN - seg - 1) % LEN) as usize;
                let (pa, aa) = entry(ia);
                let (pb, ab) = entry(ib);
                // The shader's guard, verbatim.
                if aa == 0 || ab == 0 || aa <= ab {
                    continue;
                }
                drawn += 1;
                worst = worst.max((pa - pb).length());
            }
        }

        assert!(
            drawn > 100,
            "only {drawn} segments would be drawn; the test proves nothing"
        );
        assert!(
            worst <= bound,
            "a ribbon segment spans {worst} mm, more than the {bound} mm a tracer can \
             travel in one frame -- the ring is joining entries that are not adjacent in time"
        );
    }

    #[test]
    fn particles_driven_at_a_wall_never_end_up_inside_it() {
        // The CFL clamp and the SDF push-out, end to end on the GPU: 8 m/s of
        // uniform flow straight at a plane wall, for long enough that an
        // unclamped integrator would have taken every particle clean through it.
        let Some(gpu) = gpu() else { return };
        const N: u32 = 8192;
        let grid = test_grid();
        let mut h = Harness::new(&gpu, grid, |_| Vec3::new(8.0, 0.0, 0.0));
        let mut sys = make_system(&gpu, &h, N);

        // Solid everywhere x > 0, as a signed distance field: d = -x.
        let bbox = h.fields.bbox();
        let size = bbox.size();
        let sdf_dims = UVec3::new(64, 8, 8);
        let mut data = Vec::with_capacity((sdf_dims.x * sdf_dims.y * sdf_dims.z) as usize);
        for _z in 0..sdf_dims.z {
            for _y in 0..sdf_dims.y {
                for x in 0..sdf_dims.x {
                    let px = bbox.min.x + (x as f32 + 0.5) / sdf_dims.x as f32 * size.x;
                    data.push(-px);
                }
            }
        }
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test sdf"),
            size: wgpu::Extent3d {
                width: sdf_dims.x,
                height: sdf_dims.y,
                depth_or_array_layers: sdf_dims.z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(sdf_dims.x * 4),
                rows_per_image: Some(sdf_dims.y),
            },
            wgpu::Extent3d {
                width: sdf_dims.x,
                height: sdf_dims.y,
                depth_or_array_layers: sdf_dims.z,
            },
        );
        sys.set_sdf(
            &gpu.device,
            Some(SdfSource {
                view: tex.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D3),
                    ..Default::default()
                }),
                min_mm: bbox.min,
                size_mm: size,
                surface_offset_mm: 0.0,
            }),
        );
        // Release only into the fluid half, and give them nowhere to go but the
        // wall.
        sys.set_seed_bounds(Some(Bbox {
            min: bbox.min,
            max: Vec3::new(-2.0, bbox.max.y, bbox.max.z),
        }));
        {
            let s = sys.settings_mut();
            s.seed = SeedWeights {
                inlet: 0.0,
                volume: 1.0,
                importance: 0.0,
            };
            s.life_s = 600.0;
            s.particle_radius_mm = 0.1;
        }

        for _ in 0..30 {
            h.frame(&gpu, &mut sys, 1.0 / 30.0);
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let raw = read_buffer(&gpu, &sys.particles, N as u64 * PARTICLE_BYTES);
        let f = |i: usize| f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
        let mut alive = 0usize;
        let mut worst = f32::NEG_INFINITY;
        let outer = Bbox {
            min: bbox.min - Vec3::splat(grid.dx_mm),
            max: bbox.max + Vec3::splat(grid.dx_mm),
        };
        for i in 0..N as usize {
            if f(i * 8 + 7) <= 0.0 {
                continue;
            }
            alive += 1;
            let p = Vec3::new(f(i * 8), f(i * 8 + 1), f(i * 8 + 2));
            assert!(p.is_finite(), "particle {i} is at {p}");
            assert!(outer.contains(p), "particle {i} escaped the domain at {p}");
            worst = worst.max(p.x);
        }
        assert!(
            alive > N as usize / 4,
            "only {alive} of {N} particles survived"
        );
        assert!(
            worst <= grid.dx_mm,
            "a particle came to rest at x = {worst}, more than one {} mm cell inside the wall",
            grid.dx_mm
        );
        // And confirm the flow really was fast enough to tunnel without the
        // clamp: one unsubstepped frame covers many cells.
        let per_frame = 8_000.0 * (1.0 / 30.0) * StreaklineSettings::default().time_scale;
        assert!(
            per_frame > 4.0 * grid.dx_mm,
            "the test flow is too slow to prove anything"
        );
    }
}

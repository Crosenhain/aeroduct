//! Shared types every crate in the workspace codes against.
//!
//! These are deliberately plain data. Nothing here touches the GPU, so the
//! solver, geometry, render and metrics crates can all agree on shapes without
//! depending on each other.

use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec3};

/// Air properties at 25 degrees C. Used as the default fluid everywhere.
pub mod air {
    /// Density, kg/m^3.
    pub const RHO: f64 = 1.184;
    /// Kinematic viscosity, m^2/s.
    pub const NU: f64 = 1.55e-5;
    /// Speed of sound, m/s. Only used for reporting Mach number.
    pub const C_SOUND: f64 = 346.0;
}

/// An axis-aligned box in millimetres. Model space is millimetres throughout,
/// because that is what STL files from CAD are authored in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bbox {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bbox {
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        points
            .into_iter()
            .fold(Self::EMPTY, |b, p| b.union_point(p))
    }

    pub fn union_point(self, p: Vec3) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.min.cmpgt(self.max).any()
    }

    pub fn size(&self) -> Vec3 {
        (self.max - self.min).max(Vec3::ZERO)
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Grow by a fixed margin on every side.
    pub fn expanded(self, margin: Vec3) -> Self {
        Self {
            min: self.min - margin,
            max: self.max + margin,
        }
    }

    /// Grow by a fraction of the current size on every side.
    pub fn expanded_relative(self, frac: Vec3) -> Self {
        self.expanded(self.size() * frac)
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    /// Whether two boxes overlap, touching counting as overlap.
    pub fn intersects(&self, other: Self) -> bool {
        self.min.cmple(other.max).all() && self.max.cmpge(other.min).all()
    }

    pub fn intersection(self, other: Self) -> Self {
        Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }
}

/// The uniform Cartesian lattice the solver runs on.
///
/// `origin_mm` is the world-space position of the *centre* of cell (0,0,0), so
/// cell `(i,j,k)` sits at `origin_mm + dx_mm * (i,j,k)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub dims: UVec3,
    /// Cell size in millimetres.
    pub dx_mm: f32,
    pub origin_mm: Vec3,
}

impl Grid {
    /// Build a grid covering `bbox` at cell size `dx_mm`, rounding dimensions up.
    pub fn covering(bbox: Bbox, dx_mm: f32) -> Self {
        let n = (bbox.size() / dx_mm).ceil().max(Vec3::ONE);
        Self {
            dims: n.as_uvec3(),
            dx_mm,
            // Offset by half a cell so cell centres sit inside the box.
            origin_mm: bbox.min + Vec3::splat(dx_mm * 0.5),
        }
    }

    pub fn cell_count(&self) -> u64 {
        self.dims.x as u64 * self.dims.y as u64 * self.dims.z as u64
    }

    /// Linear index. X is fastest-varying, which is what the streaming kernel
    /// wants for coalesced access along the +/-X links.
    #[inline]
    pub fn linear(&self, c: UVec3) -> u32 {
        (c.z * self.dims.y + c.y) * self.dims.x + c.x
    }

    #[inline]
    pub fn cell_center_mm(&self, c: UVec3) -> Vec3 {
        self.origin_mm + c.as_vec3() * self.dx_mm
    }

    pub fn bbox(&self) -> Bbox {
        let h = self.dx_mm * 0.5;
        Bbox {
            min: self.origin_mm - Vec3::splat(h),
            max: self.origin_mm + self.dims.as_vec3() * self.dx_mm - Vec3::splat(h),
        }
    }

    /// Workgroup dispatch counts for a given workgroup size.
    pub fn dispatch(&self, wg: UVec3) -> UVec3 {
        (self.dims + wg - UVec3::ONE) / wg
    }

    /// Cell whose centre is nearest `p`. May fall outside the grid, so the
    /// result is signed; use [`Grid::cell_range`] when you need clamped indices.
    #[inline]
    pub fn cell_containing(&self, p: Vec3) -> glam::IVec3 {
        ((p - self.origin_mm) / self.dx_mm).round().as_ivec3()
    }

    /// Inclusive cell range covering `bbox`, clamped to the grid.
    ///
    /// Returns `None` when the box lies entirely outside, which callers should
    /// treat as "no work to do" rather than as an error.
    pub fn cell_range(&self, bbox: Bbox) -> Option<(UVec3, UVec3)> {
        if bbox.is_empty() || !self.bbox().intersects(bbox) {
            return None;
        }
        let last = (self.dims.as_ivec3() - glam::IVec3::ONE).max(glam::IVec3::ZERO);
        let lo = self
            .cell_containing(bbox.min)
            .clamp(glam::IVec3::ZERO, last);
        let hi = self
            .cell_containing(bbox.max)
            .clamp(glam::IVec3::ZERO, last);
        Some((lo.as_uvec3(), hi.as_uvec3()))
    }
}

/// Which lattice velocity set the solver was built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocitySet {
    D3Q19,
    D3Q27,
}

impl VelocitySet {
    pub const fn q(self) -> usize {
        match self {
            VelocitySet::D3Q19 => 19,
            VelocitySet::D3Q27 => 27,
        }
    }
    pub const fn shader_define(self) -> &'static str {
        match self {
            VelocitySet::D3Q19 => "D3Q19",
            VelocitySet::D3Q27 => "D3Q27",
        }
    }
}

/// Storage precision for the distribution functions. Arithmetic is always FP32.
///
/// `Fp16c` is Lehmann's custom 1-sign / 4-exponent / 11-mantissa format with the
/// range clamped to +/-2. It halves truncation error versus IEEE binary16 and is
/// *faster* on a bandwidth-bound kernel, because the extra integer ops are free
/// while we wait on memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdfPrecision {
    /// Verification mode. Use this for numbers you intend to quote.
    Fp32,
    Fp16c,
}

impl DdfPrecision {
    pub const fn bytes_per_ddf(self) -> u64 {
        match self {
            DdfPrecision::Fp32 => 4,
            DdfPrecision::Fp16c => 2,
        }
    }
    pub const fn shader_define(self) -> &'static str {
        match self {
            DdfPrecision::Fp32 => "DDF_FP32",
            DdfPrecision::Fp16c => "DDF_FP16C",
        }
    }
}

/// Per-cell classification. One byte per cell, packed as a bitfield.
///
/// Kept as a plain `u8` rather than a Rust enum because a cell can be both
/// `SOLID_BOUNDARY` and `SPONGE`, and because the shader indexes it directly.
pub mod flags {
    pub const FLUID: u8 = 0x00;
    /// Inside the geometry. No collision, no streaming.
    pub const SOLID: u8 = 0x01;
    /// Fluid cell with at least one link crossing into solid. Has entries in the
    /// boundary link list, and is where interpolated bounce-back applies.
    pub const SOLID_BOUNDARY: u8 = 0x02;
    /// Velocity inlet.
    pub const INLET: u8 = 0x04;
    /// Convective outflow plane; also carries the pressure reference.
    pub const OUTLET: u8 = 0x08;
    /// Graded relaxation toward the target state, to absorb outgoing waves.
    pub const SPONGE: u8 = 0x10;
    /// Open box side: relaxed toward equilibrium so the exit jet can entrain
    /// surrounding air rather than being confined by a wall.
    pub const EQUILIBRIUM: u8 = 0x20;
    /// The two high bits say *which* velocity inlet an [`INLET`] cell belongs
    /// to: slot 0 is the duct-mouth inlet, slots 1-3 are vents standing in the
    /// room, each with its own velocity and normal in the solver's uniforms.
    pub const INLET_SLOT_SHIFT: u8 = 6;
    pub const INLET_SLOT_MASK: u8 = 0xC0;
    pub const INLET_SLOTS: usize = 4;

    #[inline]
    pub fn is_fluid(f: u8) -> bool {
        f & SOLID == 0
    }

    /// Which velocity inlet an `INLET` cell takes its velocity from.
    #[inline]
    pub fn inlet_slot(f: u8) -> u8 {
        (f & INLET_SLOT_MASK) >> INLET_SLOT_SHIFT
    }

    /// The flag byte for an inlet cell in `slot`.
    #[inline]
    pub fn inlet_in_slot(slot: u8) -> u8 {
        INLET | ((slot & 3) << INLET_SLOT_SHIFT)
    }

    /// A fluid cell with at least one link crossing into solid, so it needs
    /// bounce-back treatment. Solid cells are never boundary cells.
    #[inline]
    pub fn is_boundary(f: u8) -> bool {
        is_fluid(f) && f & SOLID_BOUNDARY != 0
    }
}

/// Lattice-unit conversion. The bridge between SI and the solver.
///
/// The procedure, in order (Kruger et al., ch. 7):
///   1. pick `dx` from the resolution you need
///   2. pick `u_lb <= 0.1` (0.05 when you care about pressure accuracy)
///   3. `dt = dx * u_lb / U`
///   4. `nu_lb = nu * dt / dx^2`
///   5. `tau = 3 * nu_lb + 1/2`
///
/// The number worth internalising: `re_cell = u_lb / nu_lb = U * dx / nu` is
/// **independent of `u_lb`**. You cannot buy stability by lowering the lattice
/// velocity, only by refining `dx`. Raising `u_lb` raises `tau` (more stable) at
/// the cost of O(Ma^2) compressibility error.
#[derive(Debug, Clone, Copy)]
pub struct LatticeUnits {
    /// Cell size, metres.
    pub dx_m: f64,
    /// Time step, seconds.
    pub dt_s: f64,
    /// Physical density that lattice density 1.0 corresponds to, kg/m^3.
    pub rho_phys: f64,
    /// Physical kinematic viscosity, m^2/s.
    pub nu_phys: f64,
    /// Reference (inlet bulk) velocity in lattice units.
    pub u_lb: f64,
    /// Reference velocity, m/s.
    pub u_phys: f64,
    /// Base relaxation time. The LES model raises this locally.
    pub tau0: f64,
}

impl LatticeUnits {
    pub fn new(dx_mm: f64, u_phys: f64, u_lb: f64, rho_phys: f64, nu_phys: f64) -> Self {
        let dx_m = dx_mm * 1e-3;
        // Guard against a zero-velocity slider: fall back to a nominal 1 m/s so
        // dt stays finite. The solver then simply produces a quiescent field.
        let u_ref = if u_phys.abs() < 1e-9 {
            1.0
        } else {
            u_phys.abs()
        };
        let dt_s = dx_m * u_lb / u_ref;
        let nu_lb = nu_phys * dt_s / (dx_m * dx_m);
        Self {
            dx_m,
            dt_s,
            rho_phys,
            nu_phys,
            u_lb,
            u_phys,
            tau0: 3.0 * nu_lb + 0.5,
        }
    }

    /// Convenience: air at 25 C.
    pub fn for_air(dx_mm: f64, u_phys: f64, u_lb: f64) -> Self {
        Self::new(dx_mm, u_phys, u_lb, air::RHO, air::NU)
    }

    pub fn nu_lb(&self) -> f64 {
        (self.tau0 - 0.5) / 3.0
    }

    /// Velocity conversion factor: m/s per lattice velocity unit.
    pub fn c_u(&self) -> f64 {
        self.dx_m / self.dt_s
    }

    /// Cell Reynolds number. Governs stability; independent of `u_lb`.
    pub fn re_cell(&self) -> f64 {
        self.u_phys.abs() * self.dx_m / self.nu_phys
    }

    /// Reynolds number for a given hydraulic diameter in millimetres.
    pub fn reynolds(&self, d_h_mm: f64) -> f64 {
        self.u_phys.abs() * (d_h_mm * 1e-3) / self.nu_phys
    }

    /// Lattice Mach number. Compressibility error is O(Ma^2); keep below ~0.17.
    pub fn mach_lb(&self) -> f64 {
        self.u_lb * f64::sqrt(3.0)
    }

    /// Lattice density deviation to physical pressure, Pa.
    /// `p = c_s^2 * (rho_lb - 1) * rho_phys * (dx/dt)^2`, with `c_s^2 = 1/3`.
    pub fn pressure_pa(&self, rho_lb_minus_one: f64) -> f64 {
        rho_lb_minus_one * self.rho_phys * self.c_u() * self.c_u() / 3.0
    }

    /// Lattice velocity to m/s.
    pub fn velocity_ms(&self, u_lattice: f64) -> f64 {
        u_lattice * self.c_u()
    }

    pub fn steps_per_physical_second(&self) -> f64 {
        1.0 / self.dt_s
    }

    /// Steps needed to advance one flow-through of a domain `length_mm` long.
    pub fn steps_per_flow_through(&self, length_mm: f64) -> f64 {
        let u = if self.u_phys.abs() < 1e-9 {
            1.0
        } else {
            self.u_phys.abs()
        };
        (length_mm * 1e-3 / u) / self.dt_s
    }

    /// Sanity gate. Returns every reason this parameter set looks dangerous, so
    /// the UI can surface it rather than letting the user wonder why it blew up.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.tau0 <= 0.5 {
            w.push(format!(
                "tau = {:.6} <= 0.5: negative viscosity, the solver will diverge",
                self.tau0
            ));
        } else if self.tau0 < 0.501 {
            w.push(format!(
                "tau = {:.6} is very close to 0.5; TRT plus the LES model is mandatory here, plain BGK will not survive",
                self.tau0
            ));
        }
        // Threshold on u_lb, not on Mach. u_lb = 0.1 is the canonical ceiling in
        // the LBM literature and is what every tier here uses, but it yields
        // Ma = 0.1*sqrt(3) = 0.1732 -- so a "Mach > 0.17" test fires on the
        // default settings of every run. A warning that is always on is a
        // warning nobody reads, which is worse than not having one.
        if self.u_lb > 0.1 + 1e-6 {
            w.push(format!(
                "lattice velocity {:.3} exceeds the usual 0.1 ceiling (Mach {:.3}): \
                 compressibility error grows as O(Ma^2)",
                self.u_lb,
                self.mach_lb()
            ));
        }
        if self.re_cell() > 600.0 {
            w.push(format!(
                "cell Reynolds number {:.0} is high; expect heavy reliance on the LES model",
                self.re_cell()
            ));
        }
        w
    }
}

/// GPU-side mirror of the simulation parameters, bound as a uniform buffer.
///
/// `repr(C)` with explicit padding to satisfy the 16-byte alignment rules WGSL
/// applies to uniform buffers.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SimUniforms {
    pub dims: [u32; 3],
    pub step_parity: u32,

    pub origin_mm: [f32; 3],
    pub dx_mm: f32,

    /// Inlet velocity in lattice units.
    pub inlet_velocity: [f32; 3],
    pub tau0: f32,

    /// Smagorinsky constant. 0 disables the LES model.
    pub smagorinsky_c: f32,
    /// TRT magic parameter. 3/16 puts the bounce-back wall exactly halfway
    /// regardless of viscosity; 1/4 is optimal for linear stability.
    pub trt_lambda: f32,
    /// Upper clamp on the LES-raised relaxation time.
    pub tau_max: f32,
    /// Convective outflow speed, lattice units.
    pub outflow_velocity: f32,

    /// Sponge layer thickness in cells; 0 disables.
    pub sponge_cells: u32,
    pub sponge_strength: f32,
    pub total_steps: u32,
    pub _pad: u32,
}

/// One entry per (boundary cell, link) pair needing interpolated bounce-back.
///
/// `q` is the normalised wall distance along the link,
/// `q = phi(x_f) / (phi(x_f) - phi(x_f + c_i))`, in (0, 1]. A value of 0.5 means
/// the wall sits exactly halfway, which is what plain bounce-back assumes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BoundaryLink {
    pub cell: u32,
    pub direction: u8,
    pub q_quantised: u8,
    pub _pad: [u8; 2],
}

impl BoundaryLink {
    #[inline]
    pub fn q(&self) -> f32 {
        // Maps 1..=255 onto (0, 1]. 0 is reserved for "no intersection".
        self.q_quantised as f32 / 255.0
    }
    #[inline]
    pub fn quantise(q: f32) -> u8 {
        (q.clamp(1.0 / 255.0, 1.0) * 255.0).round() as u8
    }
}

/// A planar patch used as an inlet, an outlet, or a measurement plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowPatch {
    /// Centre, mm.
    pub center_mm: Vec3,
    /// Unit normal. Points *into* the fluid domain for an inlet.
    pub normal: Vec3,
    /// In-plane axes scaled to the patch half-extents, mm.
    pub half_u: Vec3,
    pub half_v: Vec3,
}

impl FlowPatch {
    pub fn area_mm2(&self) -> f32 {
        4.0 * self.half_u.cross(self.half_v).length()
    }
    /// Parametric point, with `s` and `t` each in [-1, 1].
    pub fn point(&self, s: f32, t: f32) -> Vec3 {
        self.center_mm + self.half_u * s + self.half_v * t
    }
}

/// A single reduced measurement handed back from the GPU.
///
/// Everything the metrics crate reads back lands in one of these, so the async
/// readback ring has exactly one element type.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct MetricSample {
    /// Volumetric flow, m^3/s.
    pub flow_rate: f32,
    /// Mass-flow-weighted total pressure, Pa.
    pub total_pressure: f32,
    /// Area-weighted static pressure, Pa.
    pub static_pressure: f32,
    /// Weltens uniformity index, dimensionless.
    pub uniformity: f32,
    /// Mass-flux-weighted momentum direction.
    pub momentum_dir: [f32; 3],
    /// Peak speed on the patch, m/s.
    pub max_speed: f32,
    /// Fraction of patch area with reversed through-flow.
    pub backflow_fraction: f32,
    /// Step index this sample was taken at, so late readbacks can be ordered.
    pub step: u32,
    pub _pad: [u32; 2],
}

impl MetricSample {
    /// Volumetric flow in cubic feet per minute.
    pub fn cfm(&self) -> f32 {
        self.flow_rate * 2118.88
    }
    /// Volumetric flow in litres per second.
    pub fn litres_per_second(&self) -> f32 {
        self.flow_rate * 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_units_reproduce_the_planning_table() {
        // dx = 0.4 mm, U = 8 m/s, u_lb = 0.1 -> dt = 5 us, tau = 0.50145.
        let lu = LatticeUnits::for_air(0.4, 8.0, 0.1);
        assert!((lu.dt_s - 5.0e-6).abs() < 1e-12, "dt was {}", lu.dt_s);
        assert!((lu.tau0 - 0.501453).abs() < 1e-5, "tau was {}", lu.tau0);
        assert!((lu.steps_per_physical_second() - 200_000.0).abs() < 1.0);
        assert!(
            (lu.re_cell() - 206.45).abs() < 0.1,
            "re_cell was {}",
            lu.re_cell()
        );
    }

    #[test]
    fn cell_reynolds_is_independent_of_lattice_velocity() {
        let a = LatticeUnits::for_air(0.4, 8.0, 0.1);
        let b = LatticeUnits::for_air(0.4, 8.0, 0.05);
        assert!((a.re_cell() - b.re_cell()).abs() < 1e-9);
        // ...but tau is not: halving u_lb halves (tau - 1/2).
        assert!(b.tau0 < a.tau0);
    }

    #[test]
    fn the_canonical_lattice_velocity_does_not_warn() {
        // u_lb = 0.1 is the standard ceiling and is what every resolution tier
        // uses. If this warns, the app shows a permanent warning badge and the
        // real warnings stop being read.
        let lu = LatticeUnits::for_air(0.75, 3.0, 0.1);
        assert!(
            !lu.warnings().iter().any(|w| w.contains("lattice velocity")),
            "u_lb = 0.1 should not warn, got {:?}",
            lu.warnings()
        );
        // ...but going above it should.
        let hot = LatticeUnits::for_air(0.75, 3.0, 0.2);
        assert!(hot
            .warnings()
            .iter()
            .any(|w| w.contains("lattice velocity")));
    }

    #[test]
    fn finer_grids_are_more_stable() {
        let coarse = LatticeUnits::for_air(1.0, 8.0, 0.1);
        let fine = LatticeUnits::for_air(0.3, 8.0, 0.1);
        assert!(fine.tau0 > coarse.tau0);
        assert!(!coarse.warnings().is_empty(), "the coarse tier should warn");
    }

    #[test]
    fn grid_covers_the_requested_box() {
        let bbox = Bbox {
            min: Vec3::ZERO,
            max: Vec3::new(260.0, 180.0, 180.0),
        };
        let g = Grid::covering(bbox, 0.75);
        assert_eq!(g.dims, UVec3::new(347, 240, 240));
        assert_eq!(g.cell_count(), 347 * 240 * 240);
        assert!(g.bbox().min.x <= bbox.min.x + 1e-4);
    }

    #[test]
    fn cell_range_clamps_and_rejects_disjoint_boxes() {
        let g = Grid::covering(
            Bbox {
                min: Vec3::ZERO,
                max: Vec3::splat(10.0),
            },
            1.0,
        );
        // A box hanging off the corner still yields the clamped overlap.
        let (lo, hi) = g
            .cell_range(Bbox {
                min: Vec3::splat(-5.0),
                max: Vec3::splat(2.0),
            })
            .expect("overlapping box should produce a range");
        assert_eq!(lo, UVec3::ZERO);
        assert!(hi.x >= 1 && hi.x < g.dims.x);
        // A box entirely outside is not an error, just no work.
        assert!(g
            .cell_range(Bbox {
                min: Vec3::splat(100.0),
                max: Vec3::splat(110.0)
            })
            .is_none());
    }

    #[test]
    fn cell_containing_round_trips_through_cell_centres() {
        let g = Grid::covering(
            Bbox {
                min: Vec3::ZERO,
                max: Vec3::splat(8.0),
            },
            0.5,
        );
        for c in [UVec3::ZERO, UVec3::new(3, 5, 2), g.dims - UVec3::ONE] {
            assert_eq!(g.cell_containing(g.cell_center_mm(c)), c.as_ivec3());
        }
    }

    #[test]
    fn flag_predicates_agree_on_every_bit_pattern() {
        // Solid wins over everything: a solid cell is never fluid and never a
        // boundary cell, whatever else is set.
        for f in 0u8..=255 {
            if f & flags::SOLID != 0 {
                assert!(
                    !flags::is_fluid(f) && !flags::is_boundary(f),
                    "flags {f:#04x}"
                );
            } else {
                assert_eq!(flags::is_boundary(f), f & flags::SOLID_BOUNDARY != 0);
            }
        }
    }

    #[test]
    fn bbox_intersection_is_symmetric_and_matches_contains() {
        let a = Bbox {
            min: Vec3::ZERO,
            max: Vec3::splat(2.0),
        };
        let b = Bbox {
            min: Vec3::ONE,
            max: Vec3::splat(3.0),
        };
        let c = Bbox {
            min: Vec3::splat(5.0),
            max: Vec3::splat(6.0),
        };
        assert!(a.intersects(b) && b.intersects(a));
        assert!(!a.intersects(c) && !c.intersects(a));
        assert!(a.intersection(b).contains(Vec3::splat(1.5)));
    }

    #[test]
    fn boundary_link_q_round_trips() {
        for q in [0.05_f32, 0.25, 0.5, 0.75, 1.0] {
            let back = BoundaryLink {
                cell: 0,
                direction: 1,
                q_quantised: BoundaryLink::quantise(q),
                _pad: [0; 2],
            }
            .q();
            assert!(
                (back - q).abs() < 1.0 / 255.0,
                "{q} round-tripped to {back}"
            );
        }
    }
}

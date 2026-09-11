//! The parameters the user can change, and the change detection that decides
//! whether a change resets the statistics or rebuilds the solver.
//!
//! # Why a hash and not a `PartialEq`
//!
//! The question the app asks every frame is not "did anything change" but
//! "**what kind** of change was that". Three answers matter and they cost very
//! different amounts:
//!
//! | change | cost | consequence |
//! |---|---|---|
//! | inlet velocity, tau, LES constant | a uniform write | hot-applied, statistics reset |
//! | inlet/outlet swap, geometry, dx | revoxelise / reallocate | solver rebuilt, statistics reset |
//! | colormap, layer visibility, camera | nothing | statistics untouched |
//!
//! Splitting the parameters into two hashes — one for "the boundary conditions
//! the statistics are conditioned on" and one for "the things baked into the
//! shader" — makes that a two-comparison question instead of a dozen `if`s
//! scattered through the frame loop, and makes it testable.
//!
//! The reason this matters so much: **silently averaging across a boundary
//! condition change produces a confidently wrong number.** Half the window at
//! 3 m/s and half at 5 m/s yields a pressure drop that corresponds to no
//! operating point at all, with a small error bar, because the two halves are
//! each internally consistent. That is the single most dangerous failure this
//! UI can have, so the detection is structural and the reset is loud.

use std::hash::{Hash, Hasher};

use glam::{Vec2, Vec3};

use crate::view::ResetCause;

/// Largest off-normal angle the inlet air may have, degrees.
///
/// The normal component is held fixed (see [`SimParams::inlet_direction`]), so
/// at 60° the lattice speed is twice `u_lb`: a lattice Mach number of about
/// 0.17 at the default `u_lb = 0.05`. Past that the LBM's compressibility error
/// stops being small.
pub const MAX_INLET_TILT_DEG: f32 = 60.0;

/// Everything the user can turn that the solver sees.
///
/// Millimetres and SI, per the contract. Deliberately plain and `Copy`: the
/// change detector snapshots it every frame, and anything requiring a clone
/// would put an allocation in the frame loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimParams {
    /// Inlet bulk velocity, m/s. The slider the user spends the session on.
    pub inlet_velocity_ms: f32,
    /// Which detected mouth is the inlet. The other one becomes the outlet.
    pub inlet_mouth: usize,
    /// Which mouth the outflow face is taken from. `None` picks the largest
    /// mouth that is not supplying air — the other mouth of a two-mouth duct,
    /// or, with vents sealed to mouths, the largest mouth without one.
    pub outlet_mouth: Option<usize>,
    /// Off-normal tilt of the inlet air, degrees, toward the inlet mouth's two
    /// in-plane lattice axes `e1 = (axis + 1) % 3` and `e2 = (axis + 2) % 3`:
    /// the air enters along `n + tan(a) e1 + tan(b) e2`. In the part's frame,
    /// so the solver never needs the install pose; the UI derives it from a
    /// louver aim given in car terms (`crate::pose::louver_to_tilt`).
    pub inlet_tilt_deg: [f32; 2],
    /// Cell size, mm. Changing this reallocates everything.
    pub dx_mm: f32,
    /// Lattice velocity, the `u_lb` of `LatticeUnits`. 0.05 for pressure
    /// accuracy, 0.1 for speed.
    pub u_lb: f64,
    /// Fluid density, kg/m^3.
    pub rho: f64,
    /// Kinematic viscosity, m^2/s.
    pub nu: f64,
    /// Smagorinsky constant; 0 disables the LES model.
    ///
    /// The default is the **stability** value, not the accuracy one. See
    /// `ad_solver::SolverConfig::smagorinsky_for_tau`: every resolution tier in
    /// CONTRACT.md lands at `tau0 < 0.515`, where the measured sweep in
    /// `validation/lbm/gpu.rs` shows 0.11 diverging and 0.17 surviving. Shipping
    /// 0.11 here made the app open on a field that went to NaN inside a
    /// thousand steps and rendered as a solid yellow block — which reads as a
    /// broken renderer, not as an unstable solver.
    pub smagorinsky_c: f32,
    /// TRT magic parameter.
    pub trt_lambda: f32,
    /// Sponge layer thickness, cells.
    pub sponge_cells: u32,
    /// Bumped by the app whenever the geometry changes (a mesh loaded, moved,
    /// hidden). An opaque counter rather than a hash of the scene, because the
    /// scene already tracks its own dirtiness far more cheaply than we could.
    pub geometry_generation: u64,
    /// The simulated box's six margins beyond the duct, mm, as
    /// `[-x, +x, -y, +y, -z, +z]` in the part's frame. `None` leaves them to
    /// the domain rules. Changing this reallocates everything.
    pub domain_mm: Option<[f32; 6]>,
    /// Bumped by the app when a vent's air changes (speed or aim) without its
    /// plane moving: a uniform write, but the statistics are conditioned on it.
    pub vent_generation: u64,
    /// Bumped when the velocity set, storage precision or collision operator
    /// changes. Those are shader defines, so they force a rebuild.
    pub shader_generation: u64,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            inlet_velocity_ms: 3.0,
            inlet_mouth: 0,
            outlet_mouth: None,
            inlet_tilt_deg: [0.0, 0.0],
            dx_mm: 0.75,
            u_lb: 0.05,
            rho: ad_gpu::types::air::RHO,
            nu: ad_gpu::types::air::NU,
            smagorinsky_c: 0.17,
            trt_lambda: 3.0 / 16.0,
            sponge_cells: 20,
            geometry_generation: 0,
            domain_mm: None,
            vent_generation: 0,
            shader_generation: 0,
        }
    }
}

fn hash_margins<H: Hasher>(m: Option<[f32; 6]>, h: &mut H) {
    match m {
        None => 0u8.hash(h),
        Some(m) => {
            1u8.hash(h);
            for v in m {
                hash_f32(v, h);
            }
        }
    }
}

/// Hash an `f64` by its bit pattern, with every NaN folded to one value.
///
/// Bit-pattern hashing is what makes a *hash* of floating-point parameters
/// well-defined at all. The NaN folding matters because a NaN slider value
/// would otherwise never compare equal to itself and would reset the statistics
/// on every single frame — a livelock that looks like the app refusing to
/// average.
fn hash_f64<H: Hasher>(v: f64, h: &mut H) {
    if v.is_nan() {
        u64::MAX.hash(h)
    } else {
        v.to_bits().hash(h)
    }
}

fn hash_f32<H: Hasher>(v: f32, h: &mut H) {
    hash_f64(v as f64, h)
}

impl SimParams {
    /// Hash of everything the accumulated statistics are conditioned on.
    ///
    /// If this changes, every averaged number in the app now describes a
    /// different physical problem and must be thrown away.
    pub fn statistics_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        hash_f32(self.inlet_velocity_ms, &mut h);
        self.inlet_mouth.hash(&mut h);
        self.outlet_mouth.hash(&mut h);
        hash_f32(self.inlet_tilt_deg[0], &mut h);
        hash_f32(self.inlet_tilt_deg[1], &mut h);
        hash_f32(self.dx_mm, &mut h);
        hash_f64(self.u_lb, &mut h);
        hash_f64(self.rho, &mut h);
        hash_f64(self.nu, &mut h);
        hash_f32(self.smagorinsky_c, &mut h);
        hash_f32(self.trt_lambda, &mut h);
        self.sponge_cells.hash(&mut h);
        self.geometry_generation.hash(&mut h);
        hash_margins(self.domain_mm, &mut h);
        self.vent_generation.hash(&mut h);
        self.shader_generation.hash(&mut h);
        h.finish()
    }

    /// Hash of everything that cannot be hot-applied.
    ///
    /// Strictly a subset of [`Self::statistics_hash`]'s inputs: anything that
    /// forces a rebuild also invalidates the statistics, but not the reverse —
    /// the inlet velocity is the whole point of the split.
    pub fn rebuild_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        hash_f32(self.dx_mm, &mut h);
        hash_f64(self.u_lb, &mut h);
        hash_f64(self.rho, &mut h);
        hash_f64(self.nu, &mut h);
        self.inlet_mouth.hash(&mut h);
        self.outlet_mouth.hash(&mut h);
        self.sponge_cells.hash(&mut h);
        self.geometry_generation.hash(&mut h);
        hash_margins(self.domain_mm, &mut h);
        self.shader_generation.hash(&mut h);
        h.finish()
    }

    /// The most specific reason `self` differs from `prev`, for the toast.
    ///
    /// Ordered by how surprising the consequence is, not by how big the change
    /// is: a resolution change is checked before a velocity change because when
    /// both moved at once, "grid resolution changed" is the sentence that
    /// explains the ten-second stall the user just sat through.
    pub fn reset_cause(&self, prev: &SimParams) -> Option<ResetCause> {
        if self.shader_generation != prev.shader_generation {
            return Some(ResetCause::SolverRestarted);
        }
        if self.dx_mm.to_bits() != prev.dx_mm.to_bits()
            || self.u_lb.to_bits() != prev.u_lb.to_bits()
        {
            return Some(ResetCause::ResolutionChanged);
        }
        if self.geometry_generation != prev.geometry_generation {
            return Some(ResetCause::GeometryChanged);
        }
        if self.domain_mm.map(|m| m.map(f32::to_bits))
            != prev.domain_mm.map(|m| m.map(f32::to_bits))
        {
            return Some(ResetCause::DomainChanged);
        }
        if self.inlet_mouth != prev.inlet_mouth || self.outlet_mouth != prev.outlet_mouth {
            return Some(ResetCause::InletOutletSwapped);
        }
        if self.rho.to_bits() != prev.rho.to_bits() || self.nu.to_bits() != prev.nu.to_bits() {
            return Some(ResetCause::FluidChanged);
        }
        if self.inlet_velocity_ms.to_bits() != prev.inlet_velocity_ms.to_bits() {
            return Some(ResetCause::InletVelocity);
        }
        if self.inlet_tilt_deg.map(f32::to_bits) != prev.inlet_tilt_deg.map(f32::to_bits) {
            return Some(ResetCause::InletAngle);
        }
        if self.vent_generation != prev.vent_generation {
            return Some(ResetCause::VentChanged);
        }
        if self.smagorinsky_c.to_bits() != prev.smagorinsky_c.to_bits()
            || self.trt_lambda.to_bits() != prev.trt_lambda.to_bits()
            || self.sponge_cells != prev.sponge_cells
        {
            return Some(ResetCause::SolverRestarted);
        }
        None
    }

    /// The direction the inlet blows, lattice frame, for a mouth with inward
    /// unit normal `n` on lattice axis `axis`.
    ///
    /// The **normal component is held at 1**, so `u_lb * direction` prescribes
    /// the same velocity *through* the mouth as an untilted inlet: a louver
    /// turns the air, it does not change what the fan pushes. The lattice units
    /// and the loss coefficient's reference velocity keep their meaning; the
    /// price is `|u| = u_lb sec(theta)`, which is why the angle is capped at
    /// [`MAX_INLET_TILT_DEG`]. The density the equilibrium inlet floats to does
    /// move with the tilt, so the *measured* flow drifts a few percent at large
    /// angles (+2.6 % at 30° on the test part); see the tilted-inlet test in
    /// `ad_solver::reference`.
    ///
    /// Zero tilt returns `n` itself, bit for bit.
    pub fn inlet_direction(&self, n: Vec3, axis: usize) -> Vec3 {
        let [a, b] = self.inlet_tilt_deg;
        if a == 0.0 && b == 0.0 {
            return n;
        }
        let mut t = Vec2::new(a.to_radians().tan(), b.to_radians().tan());
        if !t.is_finite() {
            return n;
        }
        let cap = MAX_INLET_TILT_DEG.to_radians().tan();
        if t.length() > cap {
            t *= cap / t.length();
        }
        n + Vec3::AXES[(axis + 1) % 3] * t.x + Vec3::AXES[(axis + 2) % 3] * t.y
    }

    /// Lattice units for this operating point, so the UI can show `tau`, the
    /// Mach number and `LatticeUnits::warnings()` without going through the
    /// solver.
    pub fn lattice_units(&self) -> ad_gpu::LatticeUnits {
        ad_gpu::LatticeUnits::new(
            self.dx_mm as f64,
            self.inlet_velocity_ms as f64,
            self.u_lb,
            self.rho,
            self.nu,
        )
    }
}

/// What the frame loop must do about a parameter change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamChange {
    /// Uniform data changed; call `Solver::update` / `set_inlet_velocity`.
    pub hot_apply: bool,
    /// The solver, the voxelisation or the grid must be rebuilt.
    pub rebuild: bool,
    /// Averaged statistics are now invalid.
    pub reset_statistics: bool,
    /// What to tell the user. `None` only when nothing changed.
    pub cause: Option<ResetCause>,
}

impl ParamChange {
    pub const NONE: Self = Self {
        hot_apply: false,
        rebuild: false,
        reset_statistics: false,
        cause: None,
    };

    pub fn is_none(&self) -> bool {
        !self.hot_apply && !self.rebuild && !self.reset_statistics
    }
}

/// Watches [`SimParams`] and classifies each change.
///
/// Holds the previous *values* as well as the two hashes: the hashes answer
/// "did it change" in constant time regardless of how many fields there are,
/// and the values answer "which field" for the message. Keeping both is four
/// dozen bytes and removes the temptation to derive the cause from a hash,
/// which cannot be done.
#[derive(Debug, Clone)]
pub struct ParamWatcher {
    prev: SimParams,
    stats_hash: u64,
    rebuild_hash: u64,
}

impl ParamWatcher {
    pub fn new(params: SimParams) -> Self {
        Self {
            prev: params,
            stats_hash: params.statistics_hash(),
            rebuild_hash: params.rebuild_hash(),
        }
    }

    pub fn current(&self) -> &SimParams {
        &self.prev
    }

    /// Compare `next` against the last observed state and adopt it.
    ///
    /// Returns [`ParamChange::NONE`] on the overwhelmingly common path where
    /// nothing moved, which is two `u64` comparisons.
    pub fn observe(&mut self, next: SimParams) -> ParamChange {
        let stats = next.statistics_hash();
        let rebuild = next.rebuild_hash();
        let change = self.classify_hashed(&next, stats, rebuild);
        self.prev = next;
        self.stats_hash = stats;
        self.rebuild_hash = rebuild;
        change
    }

    /// What [`Self::observe`] would report for `next`, without adopting it.
    ///
    /// For a change that can fail. A rebuild allocates a new lattice and the
    /// GPU can refuse it, so the frame loop asks first, attempts the rebuild,
    /// and only then observes — or, if the rebuild failed, puts the old
    /// parameters back so there is nothing to observe. Committing first would
    /// announce a statistics reset for a change that never took effect.
    pub fn classify(&self, next: &SimParams) -> ParamChange {
        self.classify_hashed(next, next.statistics_hash(), next.rebuild_hash())
    }

    fn classify_hashed(&self, next: &SimParams, stats: u64, rebuild: u64) -> ParamChange {
        if stats == self.stats_hash && rebuild == self.rebuild_hash {
            return ParamChange::NONE;
        }
        let needs_rebuild = rebuild != self.rebuild_hash;
        ParamChange {
            // A rebuild subsumes a hot apply; doing both would write uniforms
            // into a solver that is about to be dropped.
            hot_apply: !needs_rebuild,
            rebuild: needs_rebuild,
            reset_statistics: stats != self.stats_hash,
            cause: next.reset_cause(&self.prev),
        }
    }

    /// Force the next [`Self::observe`] to report a change. Used after a manual
    /// statistics reset so the toast and the window agree.
    pub fn invalidate(&mut self) {
        self.stats_hash = self.stats_hash.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_box_margins_rebuild_and_a_vent_edit_hot_applies() {
        let p = SimParams::default();
        let mut w = ParamWatcher::new(p);
        let mut q = p;
        q.domain_mm = Some([58.0, 58.0, 58.0, 150.0, 58.0, 58.0]);
        let c = w.observe(q);
        assert!(c.rebuild && c.reset_statistics, "{c:?}");
        assert_eq!(c.cause, Some(ResetCause::DomainChanged));
        let mut r = q;
        r.vent_generation += 1;
        let c = w.observe(r);
        assert!(c.hot_apply && !c.rebuild && c.reset_statistics, "{c:?}");
        assert_eq!(c.cause, Some(ResetCause::VentChanged));
    }

    #[test]
    fn an_inlet_angle_is_hot_applied_and_resets_the_statistics() {
        let p = SimParams::default();
        let mut w = ParamWatcher::new(p);
        let mut q = p;
        q.inlet_tilt_deg = [20.0, 0.0];
        let c = w.observe(q);
        assert!(c.hot_apply && !c.rebuild && c.reset_statistics, "{c:?}");
        assert_eq!(c.cause, Some(ResetCause::InletAngle));
    }

    #[test]
    fn a_tilted_inlet_keeps_its_normal_component() {
        let mut p = SimParams::default();
        for n in [Vec3::Z, Vec3::NEG_Z] {
            p.inlet_tilt_deg = [0.0, 0.0];
            assert_eq!(
                p.inlet_direction(n, 2),
                n,
                "zero tilt is the normal, exactly"
            );
            p.inlet_tilt_deg = [30.0, -10.0];
            let d = p.inlet_direction(n, 2);
            assert_eq!(d.dot(n), 1.0, "the flow rate is the untilted one");
            assert!(
                (d.x - 30f32.to_radians().tan()).abs() < 1e-6,
                "e1 of axis 2 is x"
            );
            assert!(
                (d.y - (-10f32).to_radians().tan()).abs() < 1e-6,
                "e2 of axis 2 is y"
            );
        }
        p.inlet_tilt_deg = [89.0, 89.0];
        let d = p.inlet_direction(Vec3::Z, 2);
        assert!(
            d.length() <= 2.0 + 1e-5,
            "capped at 60 degrees off-normal: {d}"
        );
        p.inlet_tilt_deg = [f32::NAN, 0.0];
        assert_eq!(p.inlet_direction(Vec3::Z, 2), Vec3::Z);
    }

    #[test]
    fn nothing_changing_costs_nothing_and_reports_nothing() {
        let p = SimParams::default();
        let mut w = ParamWatcher::new(p);
        assert!(w.observe(p).is_none());
        assert!(
            w.observe(p).is_none(),
            "repeated identical frames must stay quiet"
        );
    }

    #[test]
    fn classify_previews_a_change_without_committing_it() {
        // The frame loop asks before it rebuilds, so that a rebuild the GPU
        // refuses can be undone without a trace. That only works if asking
        // changes nothing.
        let p = SimParams::default();
        let mut w = ParamWatcher::new(p);
        let finer = SimParams { dx_mm: 0.5, ..p };

        let preview = w.classify(&finer);
        assert!(preview.rebuild, "a resolution change must rebuild");
        assert!(preview.reset_statistics);
        assert_eq!(preview.cause, Some(ResetCause::ResolutionChanged));
        assert_eq!(
            w.classify(&finer),
            preview,
            "asking twice must see the same change"
        );
        assert_eq!(*w.current(), p, "asking must not adopt");

        // Backing out costs nothing: the original parameters are still the
        // committed ones, so observing them reports no change at all.
        assert!(w.observe(p).is_none());

        // Going ahead reports exactly what was previewed, and only once.
        assert_eq!(w.observe(finer), preview);
        assert!(w.observe(finer).is_none());
    }

    #[test]
    fn the_inlet_slider_hot_applies_and_resets_statistics() {
        // The exact case the brief calls out: `set_inlet_velocity` applies
        // without a rebuild, but the averages are now describing a different
        // problem and must be thrown away visibly.
        let p = SimParams::default();
        let mut w = ParamWatcher::new(p);
        let c = w.observe(SimParams {
            inlet_velocity_ms: 5.0,
            ..p
        });
        assert!(c.hot_apply);
        assert!(!c.rebuild, "the velocity must never trigger a rebuild");
        assert!(c.reset_statistics);
        assert_eq!(c.cause, Some(ResetCause::InletVelocity));
    }

    #[test]
    fn swapping_the_inlet_rebuilds_and_says_so() {
        let p = SimParams::default();
        let mut w = ParamWatcher::new(p);
        let c = w.observe(SimParams {
            inlet_mouth: 1,
            ..p
        });
        assert!(
            c.rebuild,
            "the flag field changes, so the solver is rebuilt"
        );
        assert!(!c.hot_apply, "a rebuild subsumes the uniform write");
        assert!(c.reset_statistics);
        assert_eq!(c.cause, Some(ResetCause::InletOutletSwapped));
    }

    #[test]
    fn a_resolution_change_outranks_a_simultaneous_velocity_change_in_the_message() {
        // Both moved; the user needs to be told about the expensive one.
        let p = SimParams::default();
        let mut w = ParamWatcher::new(p);
        let c = w.observe(SimParams {
            dx_mm: 0.4,
            inlet_velocity_ms: 8.0,
            ..p
        });
        assert_eq!(c.cause, Some(ResetCause::ResolutionChanged));
        assert!(c.rebuild && c.reset_statistics);
    }

    #[test]
    fn geometry_and_shader_generations_are_detected() {
        let p = SimParams::default();
        let mut w = ParamWatcher::new(p);
        assert_eq!(
            w.observe(SimParams {
                geometry_generation: 1,
                ..p
            })
            .cause,
            Some(ResetCause::GeometryChanged)
        );
        let p = *w.current();
        assert_eq!(
            w.observe(SimParams {
                shader_generation: 1,
                ..p
            })
            .cause,
            Some(ResetCause::SolverRestarted)
        );
    }

    #[test]
    fn the_rebuild_hash_ignores_the_inlet_velocity_but_the_stats_hash_does_not() {
        // This is the whole point of having two hashes; if it ever collapses to
        // one, dragging the velocity slider would stall on a solver rebuild.
        let a = SimParams::default();
        let b = SimParams {
            inlet_velocity_ms: 7.5,
            ..a
        };
        assert_eq!(a.rebuild_hash(), b.rebuild_hash());
        assert_ne!(a.statistics_hash(), b.statistics_hash());
    }

    #[test]
    fn every_rebuild_trigger_also_invalidates_the_statistics() {
        // The invariant that makes `hot_apply = !rebuild` safe: there is no
        // parameter that needs a rebuild but leaves the averages valid.
        let base = SimParams::default();
        let variants = [
            SimParams { dx_mm: 0.4, ..base },
            SimParams { u_lb: 0.1, ..base },
            SimParams { rho: 1.0, ..base },
            SimParams { nu: 1.0e-5, ..base },
            SimParams {
                inlet_mouth: 1,
                ..base
            },
            SimParams {
                sponge_cells: 4,
                ..base
            },
            SimParams {
                geometry_generation: 9,
                ..base
            },
            SimParams {
                shader_generation: 9,
                ..base
            },
        ];
        for v in variants {
            assert_ne!(
                v.rebuild_hash(),
                base.rebuild_hash(),
                "{v:?} should rebuild"
            );
            assert_ne!(
                v.statistics_hash(),
                base.statistics_hash(),
                "{v:?} rebuilds but claims the statistics survive"
            );
        }
    }

    #[test]
    fn a_nan_parameter_does_not_reset_the_statistics_on_every_frame() {
        // A NaN never equals itself, so a naive comparison would livelock the
        // averaging. Bit-pattern hashing with folded NaNs fixes it.
        let p = SimParams {
            inlet_velocity_ms: f32::NAN,
            ..SimParams::default()
        };
        let mut w = ParamWatcher::new(p);
        assert!(w.observe(p).is_none());
        assert_eq!(p.statistics_hash(), p.statistics_hash());
    }

    #[test]
    fn invalidate_forces_one_reset_and_then_settles() {
        let p = SimParams::default();
        let mut w = ParamWatcher::new(p);
        w.invalidate();
        let c = w.observe(p);
        assert!(c.reset_statistics);
        assert!(w.observe(p).is_none(), "the forced reset must not repeat");
    }

    #[test]
    fn lattice_units_reproduce_the_contract_numbers() {
        // dx = 0.4 mm at 8 m/s with u_lb = 0.1: dt = 5 us, 200,000 steps per
        // physical second. The number the status bar has to be honest about.
        let p = SimParams {
            dx_mm: 0.4,
            inlet_velocity_ms: 8.0,
            u_lb: 0.1,
            ..SimParams::default()
        };
        let u = p.lattice_units();
        assert!((u.dt_s - 5.0e-6).abs() < 1e-12, "dt was {}", u.dt_s);
        assert!((u.steps_per_physical_second() - 200_000.0).abs() < 1.0);
    }
}

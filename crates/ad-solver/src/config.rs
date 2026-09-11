//! Everything that parameterises a run, shared by the GPU solver and the CPU
//! reference so the two cannot be configured differently by accident.

use ad_gpu::types::{DdfPrecision, LatticeUnits, SimUniforms, VelocitySet};
use glam::Vec3;

use crate::collision::CollisionModel;

/// One velocity inlet as the solver sees it: the lattice velocity it imposes,
/// the inward normal of the plane its cells lie on, and where its density
/// comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InletSpec {
    pub velocity: Vec3,
    pub normal: Vec3,
    /// Take the density from the populations that arrived rather than from
    /// the plane closure. Every neighbour of a plane standing free in the
    /// fluid is real, so the raw moment *is* the local density, and resetting
    /// the cell to `f_eq(rho, u)` conserves mass: a fan disc that pushes on the
    /// air rather than a source of it. Measured in a one-dimensional column
    /// (`reference.rs`), that disc drives only 0.46 of its velocity downstream
    /// and draws 0.54 from behind; the closure drives the full velocity and
    /// draws nothing, which is what a vent with a duct behind it does. So the
    /// vents use the closure and this stays available for a true fan.
    pub local_density: bool,
}

impl Default for InletSpec {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
            normal: Vec3::X,
            local_density: false,
        }
    }
}

/// Solver parameters.
///
/// Split into two groups by mutability: the fields above `inlet_velocity` are
/// baked into the compiled shader, so changing one means constructing a new
/// [`crate::Solver`] (and [`Self::needs_rebuild`] says when that is necessary);
/// the fields below are uniform data and [`crate::Solver::update`] applies them
/// without touching a pipeline. The inlet velocity in particular must be
/// hot-updatable, because that is the slider the user spends the whole session
/// dragging.
#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    // ---- compile-time (shader defines) ----
    pub set: VelocitySet,
    pub precision: DdfPrecision,
    pub collision: CollisionModel,
    /// Per-axis periodicity. Production runs are open on every side (the
    /// contract forbids periodic box faces); the validation cases need it.
    ///
    /// A non-periodic axis costs one cell of solid halo at each end, because
    /// Esoteric Pull parks half of a boundary cell's distributions in the slot
    /// belonging to the cell *outside* the domain. See [`crate::Solver`].
    pub periodic: [bool; 3],
    /// Threads per workgroup along X. X is the fastest-varying axis, so this is
    /// the direction that coalesces; 64 and 128 are the values worth trying.
    pub workgroup_size: u32,
    /// Also write the macroscopic fields to a plain storage buffer, for CPU
    /// readback in validation. Costs an extra 16 bytes per cell of write traffic
    /// on the macroscopic pass only, so it is off in production.
    pub macroscopic_buffer: bool,

    // ---- hot-updatable (uniform data) ----
    /// Inlet velocity, lattice units.
    pub inlet_velocity: Vec3,
    /// Inward normal of the inlet plane, matching `FlowPatch::normal`.
    ///
    /// The inlet density is closed from the populations that are actually
    /// known, which means knowing which links are the unknown ones — and that
    /// is exactly the set with `c_i . n > 0`. Taking the raw moment instead
    /// reads several percent low, because the "incoming" populations at the
    /// domain edge are this cell's own, bounced back with the wrong sign of
    /// momentum. See `shaders/lbm/boundary.wgsl::inlet_drho`.
    pub inlet_normal: Vec3,
    /// Inlet slots 1-3, for cells whose flag byte carries a slot index (see
    /// `ad_gpu::flags::inlet_slot`). Slot 0 is `inlet_velocity` / `inlet_normal`
    /// above, with the plane closure.
    pub extra_inlets: [InletSpec; 3],
    /// Velocity the field is initialised to, lattice units. Used by the Galilean
    /// invariance test and as a warm start for the duct.
    pub initial_velocity: Vec3,
    /// Uniform body force per unit volume, lattice units. Drives the periodic
    /// channel cases; zero in production.
    pub body_force: Vec3,
    /// Base relaxation time. Normally `units.tau0`.
    pub tau0: f32,
    /// TRT magic parameter. 3/16 places the bounce-back wall exactly halfway.
    pub trt_lambda: f32,
    /// Smagorinsky constant; 0 disables the LES model.
    pub smagorinsky_c: f32,
    /// Upper clamp on the LES-raised relaxation time.
    pub tau_max: f32,
    /// Convective outflow speed, lattice units. Enters as `U/(1+U)`, the
    /// relaxation weight of the first-order upwind convective condition.
    pub outflow_velocity: f32,
    /// Outward normal of the outlet plane.
    ///
    /// Anti-bounce-back is applied only to links arriving from *outside* along
    /// this normal. Without it, an outlet cell that also sits on a solid wall —
    /// a corner of the domain, or where the outlet plane meets the duct body —
    /// would have its wall links turned into pressure boundaries too. Three
    /// orthogonal anti-bounce-back reflections meeting at one cell diverge:
    /// anti-bounce-back negates the population, and a corner cell negating in
    /// three directions at once is an amplifying map. That is not hypothetical;
    /// it is what the first run of
    /// `the_production_boundary_configuration_is_stable_and_conserves_flux` did,
    /// with the first NaN appearing at the outlet corner after 2500 steps.
    ///
    /// A zero normal restores the old "every blocked link" behaviour, which is
    /// correct only when the outlet plane touches nothing else.
    pub outlet_normal: Vec3,
    /// Pin the outlet pressure with anti-bounce-back. **Off by default.**
    ///
    /// CONTRACT.md calls for anti-bounce-back as the pressure reference, and it
    /// is implemented and available — but measured under real through-flow it
    /// diverges. `the_production_boundary_configuration_is_stable_and_conserves_flux`
    /// NaNs at the outlet plane within 500-2500 steps with it on, at every
    /// relaxation time from 0.502 to 0.55 and for every collision operator; with
    /// it off, every one of those cases runs to 10,000 steps and settles at a
    /// peak of 2.1x the inlet mean, which is the textbook square-duct profile.
    /// Ablation isolated it: it is the anti-bounce-back specifically, not the
    /// convective term, not the sponge, not the open box sides, and not the
    /// velocity term inside it (zeroing that changes nothing).
    ///
    /// Anti-bounce-back negates a population, so it is not a contraction, and at
    /// an outlet cell carrying strong non-equilibrium there is nothing bounding
    /// the result. The convective relaxation below reaches the same density
    /// without that: it drives the outlet toward `f^eq(rho_ref, u)` every step,
    /// which is a pressure reference in everything but name and is a contraction
    /// by construction.
    ///
    /// Left in and reachable rather than deleted, because the acoustics work the
    /// contract anticipates will want a *hard* pressure reference, and the fix
    /// is likely a relaxed form `(1-theta) bounce-back + theta anti-bounce-back`
    /// rather than a rewrite.
    pub outlet_anti_bounce_back: bool,
    pub sponge_cells: u32,
    pub sponge_strength: f32,
    /// Reference density the outlet pressure is pinned to.
    pub rho_ref: f32,
    /// Seconds of physical time per lattice step. Reporting only — nothing in
    /// the solver reads it, but `Solver::sim_time_seconds` does, and the UI
    /// needs a number it can put next to "0.42 s simulated".
    pub dt_s: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            set: VelocitySet::D3Q19,
            precision: DdfPrecision::Fp32,
            collision: CollisionModel::Trt,
            periodic: [false; 3],
            workgroup_size: 64,
            macroscopic_buffer: false,
            inlet_velocity: Vec3::ZERO,
            inlet_normal: Vec3::X,
            extra_inlets: [InletSpec::default(); 3],
            initial_velocity: Vec3::ZERO,
            body_force: Vec3::ZERO,
            tau0: 0.6,
            trt_lambda: 3.0 / 16.0,
            // 0.17, not the accuracy-motivated 0.11, because `Default` has no
            // tau to consult and must therefore be the safe one. Every tier in
            // the plan lands at tau0 < 0.515, where the measured sweep says 0.11
            // diverges -- and a `..Default::default()` at a coarse tau is
            // exactly how that bites: the app opened on a NaN field rendering as
            // a solid yellow block before this was corrected. Callers who want
            // the accurate value should build through `from_units`, which picks
            // by resolution.
            smagorinsky_c: 0.17,
            tau_max: 1.0,
            outflow_velocity: 0.0,
            outlet_normal: Vec3::X,
            outlet_anti_bounce_back: false,
            sponge_cells: 0,
            sponge_strength: 0.0,
            rho_ref: 1.0,
            dt_s: 1.0,
        }
    }
}

impl SolverConfig {
    /// Config for a production duct run at the given physical operating point.
    ///
    /// Picks the Smagorinsky constant from `tau0` via
    /// [`SolverConfig::smagorinsky_for_tau`], because the right value depends on
    /// the grid and getting it wrong is the difference between a run that
    /// converges and one that goes NaN.
    pub fn from_units(units: LatticeUnits) -> Self {
        Self {
            tau0: units.tau0 as f32,
            dt_s: units.dt_s,
            smagorinsky_c: Self::smagorinsky_for_tau(units.tau0 as f32),
            ..Default::default()
        }
    }

    /// Smagorinsky constant appropriate to a given base relaxation time.
    ///
    /// There are two defensible values and they disagree, so this picks between
    /// them by resolution rather than pretending one answer fits both ends:
    ///
    /// - **0.11** is the standard internal-flow value, chosen for *accuracy*. The
    ///   theoretical Lilly constant of 0.17 is well known to be over-dissipative
    ///   in shear and near walls, which is exactly where this duct's interesting
    ///   physics lives.
    /// - **0.17** is what the coarse tier actually needs to *survive*. Measured
    ///   on the bent-duct smoke test: `tau0` of 0.502, 0.505 and 0.51 all diverge
    ///   at 0.11 and are stable at 0.17; 0.52 is stable at 0.11; and `Cs = 0`
    ///   diverges everywhere below 0.53.
    ///
    /// The two are reconcilable once you notice what the model is *for* at each
    /// end. `tau0` rises as `dx` falls, so small `tau0` means a coarse grid. On
    /// the coarse interactive tier the subgrid model is doing real work and is
    /// mainly a numerical stabiliser, so the dissipative value is both necessary
    /// and honest. On the fine quality tier the first fluid node sits at
    /// `y+ = 2..7`, inside the viscous sublayer, where the model contributes
    /// almost nothing and its only effect is the accuracy penalty -- so use the
    /// accurate value there.
    ///
    /// The threshold sits at `tau0 = 0.515`, between the measured 0.51 (needs
    /// 0.17) and 0.52 (fine at 0.11).
    ///
    /// This is a default, not a constraint: the UI exposes `smagorinsky_c`, and
    /// a user chasing a converged pressure drop can lower it and watch for
    /// divergence.
    pub fn smagorinsky_for_tau(tau0: f32) -> f32 {
        if tau0 < 0.515 {
            0.17
        } else {
            0.11
        }
    }

    /// The inlet a cell in `slot` (0-3) is driven by. Slot 0 is the duct-mouth
    /// inlet with the plane closure; the rest are [`Self::extra_inlets`].
    pub fn inlet_slot(&self, slot: u8) -> InletSpec {
        match slot {
            0 => InletSpec {
                velocity: self.inlet_velocity,
                normal: self.inlet_normal,
                local_density: false,
            },
            s => self.extra_inlets[(s as usize - 1).min(2)],
        }
    }

    /// True when a shader rebuild is needed to go from `self` to `other`.
    pub fn needs_rebuild(&self, other: &Self) -> bool {
        self.set != other.set
            || self.precision != other.precision
            || self.collision != other.collision
            || self.periodic != other.periodic
            || self.workgroup_size != other.workgroup_size
            || self.macroscopic_buffer != other.macroscopic_buffer
    }

    /// Fill the shared [`SimUniforms`] mirror. The solver adds its own private
    /// fields (body force, periodicity, reference density) in
    /// [`crate::LbmUniforms`]; those are deliberately not in `SimUniforms`
    /// because nothing outside the solver needs them.
    pub fn sim_uniforms(
        &self,
        grid: &ad_gpu::types::Grid,
        step_parity: u32,
        total_steps: u32,
    ) -> SimUniforms {
        SimUniforms {
            dims: [grid.dims.x, grid.dims.y, grid.dims.z],
            step_parity,
            origin_mm: grid.origin_mm.to_array(),
            dx_mm: grid.dx_mm,
            inlet_velocity: self.inlet_velocity.to_array(),
            tau0: self.tau0,
            smagorinsky_c: self.smagorinsky_c,
            trt_lambda: self.trt_lambda,
            tau_max: self.tau_max,
            outflow_velocity: self.outflow_velocity,
            sponge_cells: self.sponge_cells,
            sponge_strength: self.sponge_strength,
            total_steps,
            _pad: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::LatticeUnits;

    #[test]
    fn coarse_grids_get_the_stabilising_smagorinsky_constant() {
        // The interactive tier: dx = 0.75 mm at 8 m/s gives tau0 ~ 0.5008, which
        // the duct smoke test showed diverges at Cs = 0.11.
        let coarse = SolverConfig::from_units(LatticeUnits::for_air(0.75, 8.0, 0.1));
        assert!(
            coarse.tau0 < 0.515,
            "expected a coarse tau0, got {}",
            coarse.tau0
        );
        assert_eq!(coarse.smagorinsky_c, 0.17);
    }

    #[test]
    fn fine_grids_get_the_accurate_smagorinsky_constant() {
        // Push tau0 above the threshold and the accuracy-motivated value applies.
        assert_eq!(SolverConfig::smagorinsky_for_tau(0.52), 0.11);
        assert_eq!(SolverConfig::smagorinsky_for_tau(0.60), 0.11);
    }

    #[test]
    fn the_smagorinsky_threshold_brackets_the_measured_cases() {
        // Measured: 0.51 diverges at 0.11, 0.52 does not. The threshold must
        // separate them, or the default silently reintroduces the divergence.
        assert_eq!(SolverConfig::smagorinsky_for_tau(0.510), 0.17);
        assert_eq!(SolverConfig::smagorinsky_for_tau(0.520), 0.11);
    }

    #[test]
    fn the_bare_default_is_the_stable_constant_not_the_accurate_one() {
        // `Default` cannot know the grid, so it must not hand out the value that
        // diverges on every tier this app actually ships.
        assert_eq!(SolverConfig::default().smagorinsky_c, 0.17);
    }

    #[test]
    fn from_units_carries_the_timestep_through_for_pressure_conversion() {
        let units = LatticeUnits::for_air(0.4, 3.0, 0.1);
        let cfg = SolverConfig::from_units(units);
        assert_eq!(cfg.dt_s, units.dt_s);
        assert!((cfg.tau0 as f64 - units.tau0).abs() < 1e-6);
    }
}

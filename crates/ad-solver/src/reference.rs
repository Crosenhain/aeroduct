//! A CPU lattice-Boltzmann solver that mirrors the GPU kernel step for step.
//!
//! [`ad_gpu::lattice`] anticipates this: "so the CPU-side reference solver and
//! the validation harness agree with the shader by construction". It exists for
//! three reasons.
//!
//! 1. **The physics tests run without a GPU.** Mass conservation, Galilean
//!    invariance, tau-independence and the Poiseuille convergence order are all
//!    small-grid problems. Running them on the CPU means they gate every commit
//!    on any machine, not just one with a Vulkan adapter.
//! 2. **It localises failures.** When a GPU result is wrong, the question is
//!    always "physics or plumbing?". `validation/lbm/gpu.rs` answers it
//!    in one assertion by comparing the two field-for-field.
//! 3. **It is where the streaming scheme was derived and checked.** In
//!    particular the `a_population_aimed_at_a_wall_comes_back_reversed` test is
//!    the direct proof of the bounce-back rule below, which is not obvious and
//!    which a subtly wrong version of would still produce plausible-looking flow.
//!
//! # The bounce-back rule
//!
//! Esoteric Pull gives a cell two addresses per direction pair `(m, m+1)`:
//! `X = slot(n - c_m, alpha)` and `Y = slot(n, !alpha)`, with
//! `alpha = p ? m : m+1` and `p` the step parity. `X` carries the link between
//! `n` and `n - c_m`; `Y` carries the link between `n` and `n + c_m`. Read and
//! write use the *same* address — that is the in-place property — and the two
//! cells sharing a link swap which of the pair's two slots they own on every
//! step.
//!
//! Now let `s = n - c_m` be solid. Cell `n` owns *both* slots at cell-index `s`,
//! because nothing on the other side of the link ever executes. At step `p` it
//! stores its outgoing `f_{m+1}` into `slot(s, alpha)`. At step `p+1` the
//! unmodified rule would have it load `f_m` from `slot(s, !alpha)`, which nobody
//! wrote. Loading with the parity *flipped* instead reads `slot(s, alpha)` —
//! exactly the value it stored — and that is
//! `f_m(n, t+1) = f_{m+1}^post(n, t)`, which is halfway bounce-back verbatim.
//!
//! So: **stores are never flipped; a load flips exactly when the upstream
//! neighbour `n - c_i` is solid.** No branch on the address, no extra memory
//! traffic, no separate boundary kernel. The flip bits are precomputed once into
//! [`crate::boundary::PaddedDomain::link_mask`].
//!
//! The same rule makes initialisation trivial: writing the initial equilibrium
//! with `store_slot(..., odd_step = true)` lands every value, wall links
//! included, exactly where step 0 will look for it.

use ad_gpu::lattice::{EsotericPull, LatticeDef};
use ad_gpu::types::flags;
use glam::{UVec3, Vec3};

use crate::boundary::{CellKind, PaddedDomain};
use crate::collision::{apply_force, collide, pi_norm, smagorinsky_tau, trt_rates};
use crate::config::SolverConfig;
use crate::precision::{quantise_fp16c, shifted_equilibrium};

/// Density and velocity at one cell, in lattice units.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Macro {
    pub rho: f32,
    pub u: Vec3,
}

/// The CPU reference solver. Same index scheme, same operators, same boundary
/// treatment as `shaders/lbm/stream_collide.wgsl`.
pub struct ReferenceLbm {
    pub domain: PaddedDomain,
    pub cfg: SolverConfig,
    def: &'static LatticeDef,
    ep: EsotericPull,
    /// Shifted populations `g_i = f_i - w_i`, structure-of-arrays.
    g: Vec<f32>,
    steps: u64,
}

impl ReferenceLbm {
    pub fn new(domain: PaddedDomain, cfg: SolverConfig) -> Self {
        let def = cfg.set.def();
        let cells = domain.padded_cell_count();
        let mut me = Self {
            ep: EsotericPull::new(cells, def.q),
            g: vec![0.0; cells as usize * def.q],
            domain,
            cfg,
            def,
            steps: 0,
        };
        me.reset();
        me
    }

    pub fn steps_taken(&self) -> u64 {
        self.steps
    }

    fn odd_step(&self) -> bool {
        self.steps % 2 == 1
    }

    #[inline]
    fn neighbour_index(&self, c: UVec3, d: usize) -> u64 {
        self.domain
            .linear(self.domain.neighbour(c, self.def.directions[d])) as u64
    }

    /// Storage round trip, so the CPU reference truncates exactly as the GPU
    /// does when running in FP16C.
    #[inline]
    fn quantise(&self, v: f32) -> f32 {
        match self.cfg.precision {
            ad_gpu::types::DdfPrecision::Fp32 => v,
            ad_gpu::types::DdfPrecision::Fp16c => quantise_fp16c(v),
        }
    }

    /// Reset to equilibrium at `(1, initial_velocity)`.
    ///
    /// Writes with `odd_step = true`, i.e. as the store half of a virtual step
    /// `-1`, which is exactly what step 0's loads expect — for ordinary links
    /// *and* for wall links, where the flipped load lands on the slot this write
    /// fills. Anything else leaves the first step reading uninitialised memory
    /// on every boundary link.
    pub fn reset(&mut self) {
        self.steps = 0;
        let u = self.cfg.initial_velocity;
        self.set_equilibrium(|_| (1.0, u));
    }

    /// Overwrite the whole field with equilibrium at a caller-supplied state,
    /// given as a function of *interior* coordinates.
    ///
    /// Used by [`Self::reset`] and by validation cases that need a specific
    /// initial condition (a seeded shear wave, an analytic warm start). The
    /// write goes through `store_slot` at the parity of the step *before* the
    /// current one, which is the only choice that leaves wall links populated;
    /// see the module comment.
    pub fn set_equilibrium(&mut self, state: impl Fn(UVec3) -> (f32, Vec3)) {
        self.g.fill(0.0);
        let store_parity = !self.odd_step();
        let offset = self.domain.offset;
        let cells = self.domain.padded_cell_count();
        for cell in 0..cells {
            if !flags::is_fluid(self.domain.flags[cell as usize]) {
                continue;
            }
            let c = self.domain.coords(cell as u32);
            let (rho, u) = state(c - offset);
            let u = u.to_array();
            for i in 0..self.def.q {
                let d = self.def.directions[i];
                let geq = shifted_equilibrium(
                    self.def.weights[i],
                    [d.x as f32, d.y as f32, d.z as f32],
                    rho - 1.0,
                    rho,
                    u,
                );
                let idx = self
                    .ep
                    .store_slot(cell, i, store_parity, |k| self.neighbour_index(c, k));
                self.g[idx as usize] = self.quantise(geq);
            }
        }
    }

    /// Load a cell's full population set for the current step.
    fn load(&self, cell: u64, c: UVec3, odd: bool, out: &mut [f32]) {
        let mask = self.domain.link_mask[cell as usize];
        out[0] = self.g[self.ep.load_slot(cell, 0, odd, |_| 0) as usize];
        for i in 1..self.def.q {
            let flip = mask & (1 << i) != 0;
            let idx = self
                .ep
                .load_slot(cell, i, odd ^ flip, |k| self.neighbour_index(c, k));
            out[i] = self.g[idx as usize];
        }
    }

    /// `rho - 1` and `rho * u` from shifted populations, in the order the
    /// precision argument in [`crate::precision`] demands: the small deviations
    /// are summed first and the 1 is added exactly once at the end.
    fn moments(&self, g: &[f32]) -> (f32, [f32; 3]) {
        let mut drho = 0.0f32;
        let mut m = [0.0f32; 3];
        for i in 0..self.def.q {
            let v = g[i];
            drho += v;
            let c = self.def.directions[i];
            m[0] += v * c.x as f32;
            m[1] += v * c.y as f32;
            m[2] += v * c.z as f32;
        }
        (drho, m)
    }

    fn equilibrium(&self, drho: f32, rho: f32, u: [f32; 3], out: &mut [f32]) {
        for i in 0..self.def.q {
            let c = self.def.directions[i];
            out[i] = shifted_equilibrium(
                self.def.weights[i],
                [c.x as f32, c.y as f32, c.z as f32],
                drho,
                rho,
                u,
            );
        }
    }

    /// Advance one step.
    pub fn step(&mut self) {
        let q = self.def.q;
        let odd = self.odd_step();
        let cells = self.domain.padded_cell_count();
        let force = self.cfg.body_force.to_array();
        let rho_ref = self.cfg.rho_ref;

        // Esoteric Pull is in-place and thread-safe: every (address, half) is
        // owned by exactly one cell for the whole step, so a single buffer with
        // no ordering constraints is correct. The GPU relies on precisely this.
        let mut new = self.g.clone();
        let mut g = vec![0.0f32; q];
        let mut geq = vec![0.0f32; q];
        let mut neq = vec![0.0f32; q];

        for cell in 0..cells {
            let cf = self.domain.flags[cell as usize];
            let kind = CellKind::of(cf);
            if kind == CellKind::Solid {
                continue;
            }
            let c = self.domain.coords(cell as u32);
            self.load(cell, c, odd, &mut g);

            let (mut drho, mut mom) = self.moments(&g);

            if kind == CellKind::Outlet && self.cfg.outlet_anti_bounce_back {
                // Anti-bounce-back pins the pressure at the outlet plane. It
                // needs the wall velocity, which we do not have until the
                // moments are in, so predict with the plain bounce-back result
                // and then correct. One pass is enough: the correction is
                // O(Ma^2) in the velocity term.
                let rho_p = 1.0 + drho;
                let up = [mom[0] / rho_p, mom[1] / rho_p, mom[2] / rho_p];
                self.anti_bounce_back(cell, c, &mut g, up, rho_ref);
                let (d2, m2) = self.moments(&g);
                drho = d2;
                mom = m2;
            }

            let mut rho = 1.0 + drho;
            let mut u = [
                (mom[0] + 0.5 * force[0]) / rho,
                (mom[1] + 0.5 * force[1]) / rho,
                (mom[2] + 0.5 * force[2]) / rho,
            ];

            match kind {
                CellKind::Inlet => {
                    let spec = self.inlet_spec(cell);
                    u = self.inlet_velocity_at(cell, spec).to_array();
                    drho = if spec.local_density {
                        // Free-standing: the raw moment is the local density.
                        // See `InletSpec::local_density`.
                        let r = rho_ref - 1.0;
                        drho.clamp(r - 0.2, r + 0.2)
                    } else {
                        self.inlet_drho(&g, spec.normal, u)
                    };
                    rho = 1.0 + drho;
                }
                CellKind::Equilibrium => {
                    rho = rho_ref;
                    drho = rho_ref - 1.0;
                    u = [0.0; 3];
                }
                _ => {}
            }

            self.equilibrium(drho, rho, u, &mut geq);

            for i in 0..q {
                neq[i] = g[i] - geq[i];
            }
            let tau = smagorinsky_tau(
                self.cfg.tau0,
                self.cfg.tau_max,
                self.cfg.smagorinsky_c,
                rho,
                pi_norm(self.def, &neq),
            );
            let lambda = match self.cfg.collision {
                crate::collision::CollisionModel::Trt => self.cfg.trt_lambda,
                _ => 0.0,
            };
            let (s_e, s_o) = trt_rates(tau, lambda);

            match kind {
                // Equilibrium boundaries discard the non-equilibrium entirely.
                // Simple, unconditionally stable, and the reason this is the v1
                // inlet: it has no mechanism by which to diverge.
                CellKind::Inlet | CellKind::Equilibrium => g.copy_from_slice(&geq),
                _ => {
                    collide(self.cfg.collision, self.def, &mut g, &geq, s_e, s_o);
                    apply_force(self.def, &mut g, u, force, s_e, s_o);
                }
            }

            if kind == CellKind::Outlet && self.cfg.outflow_velocity > 0.0 {
                // Convective outflow df/dt + U df/dn = 0, discretised upwind and
                // with the upstream population approximated by the local
                // equilibrium at the reference density. Weight U/(1+U) is the
                // exact factor from f(t+1) = (f + U f_up)/(1 + U).
                let uu = self.cfg.outflow_velocity;
                let w = uu / (1.0 + uu);
                let mut target = vec![0.0f32; q];
                self.equilibrium(rho_ref - 1.0, rho_ref, u, &mut target);
                for i in 0..q {
                    g[i] += w * (target[i] - g[i]);
                }
            }

            let sigma =
                self.domain
                    .sponge_sigma(c, cf, self.cfg.sponge_cells, self.cfg.sponge_strength);
            if sigma > 0.0 {
                // Absorb rather than reflect: relax toward equilibrium at the
                // reference density but the *local* velocity, so the layer eats
                // acoustic and vortical content without imposing a mean flow.
                let mut target = vec![0.0f32; q];
                self.equilibrium(rho_ref - 1.0, rho_ref, u, &mut target);
                for i in 0..q {
                    g[i] += sigma * (target[i] - g[i]);
                }
            }

            for i in 0..q {
                let idx = self
                    .ep
                    .store_slot(cell, i, odd, |k| self.neighbour_index(c, k));
                new[idx as usize] = self.quantise(g[i]);
            }
        }

        self.g = new;
        self.steps += 1;
    }

    /// Replace the populations arriving on wall links with the anti-bounce-back
    /// value, which fixes the density at `rho_ref` instead of the velocity at
    /// zero.
    ///
    /// `f_i = -f_ibar^post + 2 w_i rho_ref (1 + (c.u)^2/(2 c_s^4) - u^2/(2 c_s^2))`,
    /// written in shifted form so no unshifted population is ever built:
    /// `g_i = -g_loaded + 2 w_i [(rho_ref - 1) + rho_ref ((9/2)(c.u)^2 - (3/2)u^2)]`.
    /// Density at a velocity inlet, closed from the *known* populations.
    ///
    /// With `n` the inward normal the unknowns are exactly the links with
    /// `c_i . n > 0`, and `rho = (A + 2B)/(1 - u_n)` with `A` the tangential sum
    /// and `B` the outgoing sum. In shifted form the weight sums cancel exactly
    /// (`W_0 + 2 W_- = 1` for an axis-aligned normal), leaving
    /// `drho = (A_g + 2 B_g + u_n)/(1 - u_n)` with no unshifted quantity.
    ///
    /// The raw moment cannot be used: at a plane inlet the populations arriving
    /// from outside are this cell's own, bounced off the domain halo, so they
    /// carry `-u` instead of `+u` and the density reads low by `u_n` every step.
    /// Measured fixed point at `u_n = 0.05`: `rho = 0.864`, with the duct
    /// downstream stagnant.
    fn inlet_drho(&self, g: &[f32], n: Vec3, u: [f32; 3]) -> f32 {
        let (mut a, mut b) = (0.0f32, 0.0f32);
        for i in 0..self.def.q {
            let cn = self.def.directions[i].as_vec3().dot(n);
            if cn == 0.0 {
                a += g[i];
            } else if cn < 0.0 {
                b += g[i];
            }
        }
        let un = u[0] * n.x + u[1] * n.y + u[2] * n.z;
        let drho = (a + 2.0 * b + un) / (1.0 - un).max(0.1);
        // Clamped: the closure assumes a flat plane, and at the edge of a narrow
        // inlet patch some "known" links are blocked by the duct wall instead.
        // A generous band keeps the inlet's defining property - it cannot blow
        // up - without ever binding on a well-posed problem. See
        // `shaders/lbm/boundary.wgsl::inlet_drho`.
        let r = self.cfg.rho_ref - 1.0;
        drho.clamp(r - 0.2, r + 0.2)
    }

    /// The velocity one inlet cell imposes: the configured one, less its
    /// component normal to any wall beside the cell.
    ///
    /// A tilted inlet (a louver aim) carries a velocity *along* the inlet
    /// plane. Where the plane meets the duct wall, part of that is a velocity
    /// through an impermeable wall — into it on one side of the duct, out of it
    /// on the other — and the wall's bounce-back hands it straight to
    /// [`Self::inlet_drho`], which reads the reflections as data: the axis link
    /// and both in-plane diagonals toward the wall all carry it with the same
    /// sign, so the density closure sees a pile-up on one rim and suction on
    /// the other (±0.06 in density, measured in
    /// `a_tilted_inlet_turns_the_air_and_keeps_the_flow_rate`, against a
    /// physical ~3 u_t² of 0.002). A component *along* the wall is harmless:
    /// its two diagonals toward the wall carry it with opposite signs and
    /// cancel. So the wall-normal component goes — which is also just the
    /// no-penetration condition — and the rest stays. Only the four axis links
    /// in the plane are tested, which covers axis-aligned walls; a corner loses
    /// both in-plane components. A smaller residual, one row in from the rim,
    /// is inherent to the density closure; the test says how big.
    ///
    /// With no tilt the in-plane components are exactly zero, so this returns
    /// the configured velocity bit for bit. `shaders/lbm/boundary.wgsl` has the
    /// GPU twin.
    fn inlet_velocity_at(&self, cell: u64, spec: crate::config::InletSpec) -> Vec3 {
        let n = spec.normal;
        let mask = self.domain.link_mask[cell as usize];
        let mut u = spec.velocity;
        for i in 1..self.def.q {
            let c = self.def.directions[i];
            if c.abs().element_sum() != 1 || c.as_vec3().dot(n) != 0.0 {
                continue;
            }
            // Bit i: the neighbour this population streams in from, x - c_i,
            // is solid, so there is a wall on the -c_i side.
            if mask & (1 << i) != 0 {
                let c = c.as_vec3();
                u -= c * u.dot(c);
            }
        }
        u
    }

    /// Which inlet drives this cell: the slot in the two high bits of its flag
    /// byte, looked up in the config. Twin of the shader's `inlet_slot`.
    fn inlet_spec(&self, cell: u64) -> crate::config::InletSpec {
        self.cfg
            .inlet_slot(flags::inlet_slot(self.domain.flags[cell as usize]))
    }

    /// Is link `i` at padded cell `cc` a genuine outflow link?
    ///
    /// The direction must point back into the domain against the outlet normal,
    /// *and* the upstream cell must leave the domain only along the normal axis.
    /// The second condition is what keeps the corner where the outlet plane
    /// meets a side wall from turning that wall into a pressure boundary; see
    /// `SolverConfig::outlet_normal`.
    fn is_outflow_link(&self, cc: UVec3, i: usize) -> bool {
        let n = self.cfg.outlet_normal;
        if n == Vec3::ZERO {
            return true;
        }
        let c = self.def.directions[i];
        if c.as_vec3().dot(n) >= 0.0 {
            return false;
        }
        let up = cc.as_ivec3() - c;
        let lo = self.domain.offset.as_ivec3();
        let hi = lo + self.domain.interior.as_ivec3() - glam::IVec3::ONE;
        for a in 0..3 {
            if n[a].abs() > 0.5 {
                continue; // the normal axis may be outside
            }
            if up[a] < lo[a] || up[a] > hi[a] {
                return false;
            }
        }
        true
    }

    fn anti_bounce_back(&self, cell: u64, cc: UVec3, g: &mut [f32], u: [f32; 3], rho_ref: f32) {
        let mask = self.domain.link_mask[cell as usize];
        let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        for i in 1..self.def.q {
            if mask & (1 << i) == 0 || !self.is_outflow_link(cc, i) {
                continue;
            }
            let c = self.def.directions[i];
            let cu = c.x as f32 * u[0] + c.y as f32 * u[1] + c.z as f32 * u[2];
            let abb = 2.0
                * self.def.weights[i]
                * ((rho_ref - 1.0) + rho_ref * (4.5 * cu * cu - 1.5 * uu));
            g[i] = -g[i] + abb;
        }
    }

    /// Density and velocity at one *interior* cell.
    pub fn macroscopic(&self, ic: UVec3) -> Macro {
        let cell = self.domain.interior_linear(ic) as u64;
        self.macroscopic_padded(cell)
    }

    /// Density and velocity at a padded cell index.
    ///
    /// Equilibrium boundaries report the state they *impose*, not the moment of
    /// their incoming populations. At an inlet the post-step state is exactly
    /// `f^eq(rho, u_inlet)`; the pre-collision moment there is dominated by
    /// links reflecting off the domain wall behind the plane and can have the
    /// wrong sign entirely. `shaders/lbm/macroscopic.wgsl` applies the same rule,
    /// so the CPU and GPU fields stay comparable.
    pub fn macroscopic_padded(&self, cell: u64) -> Macro {
        let cf = self.domain.flags[cell as usize];
        if !flags::is_fluid(cf) {
            return Macro {
                rho: 1.0,
                u: Vec3::ZERO,
            };
        }
        let c = self.domain.coords(cell as u32);
        let mut g = vec![0.0f32; self.def.q];
        self.load(cell, c, self.odd_step(), &mut g);
        let (drho, mom) = self.moments(&g);
        let rho = 1.0 + drho;
        let f = self.cfg.body_force.to_array();
        let mut m = Macro {
            rho,
            u: Vec3::new(
                (mom[0] + 0.5 * f[0]) / rho,
                (mom[1] + 0.5 * f[1]) / rho,
                (mom[2] + 0.5 * f[2]) / rho,
            ),
        };
        match CellKind::of(cf) {
            CellKind::Inlet => m.u = self.inlet_velocity_at(cell, self.inlet_spec(cell)),
            CellKind::Equilibrium => {
                m.rho = self.cfg.rho_ref;
                m.u = Vec3::ZERO;
            }
            _ => {}
        }
        m
    }

    /// Total lattice mass over the fluid cells. Conserved to round-off in a
    /// closed periodic box, which makes it the cheapest possible streaming test.
    pub fn total_mass(&self) -> f64 {
        let mut m = 0.0f64;
        for cell in 0..self.domain.padded_cell_count() {
            if flags::is_fluid(self.domain.flags[cell as usize]) {
                m += self.macroscopic_padded(cell).rho as f64;
            }
        }
        m
    }

    /// Raw access to the shifted populations, for the GPU comparison test.
    pub fn raw(&self) -> &[f32] {
        &self.g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::CollisionModel;
    use ad_gpu::lattice::{opposite, D3Q19_DIRS};
    use ad_gpu::types::VelocitySet;

    fn periodic_box(n: u32, cfg: SolverConfig) -> ReferenceLbm {
        let d = PaddedDomain::uniform_fluid(UVec3::splat(n), [true; 3], cfg.set);
        ReferenceLbm::new(d, cfg)
    }

    /// The single sharpest check on the streaming scheme, and the one to run
    /// first: a closed periodic box must conserve total density exactly. Any
    /// index mistake either duplicates or destroys populations, and shows up
    /// here within a handful of steps.
    #[test]
    fn mass_is_conserved_in_a_closed_periodic_box() {
        let cfg = SolverConfig {
            tau0: 0.6,
            smagorinsky_c: 0.0,
            initial_velocity: Vec3::new(0.05, -0.03, 0.02),
            ..Default::default()
        };
        let mut lbm = periodic_box(8, cfg);
        // Perturb so there is something to stream: a density blob in the middle.
        let m0 = lbm.total_mass();
        for _ in 0..500 {
            lbm.step();
        }
        let m1 = lbm.total_mass();
        let cells = lbm.domain.interior_cell_count() as f64;
        assert!(
            (m1 - m0).abs() / cells < 1e-6,
            "mass drifted from {m0} to {m1} over 500 steps ({} per cell)",
            (m1 - m0) / cells
        );
    }

    /// Uniform flow in a periodic box is an exact solution of the lattice
    /// Boltzmann equation. If streaming, collision or forcing has an asymmetry,
    /// the field will not stay uniform.
    #[test]
    fn uniform_flow_stays_exactly_uniform() {
        for model in [
            CollisionModel::Trt,
            CollisionModel::Bgk,
            CollisionModel::RegularizedBgk,
        ] {
            let cfg = SolverConfig {
                collision: model,
                tau0: 0.8,
                smagorinsky_c: 0.0,
                initial_velocity: Vec3::new(0.06, 0.02, -0.04),
                ..Default::default()
            };
            let mut lbm = periodic_box(6, cfg);
            for _ in 0..200 {
                lbm.step();
            }
            let want = cfg.initial_velocity;
            for z in 0..6u32 {
                for y in 0..6u32 {
                    for x in 0..6u32 {
                        let m = lbm.macroscopic(UVec3::new(x, y, z));
                        assert!(
                            (m.rho - 1.0).abs() < 1e-5,
                            "{}: density {} at ({x},{y},{z})",
                            model.name(),
                            m.rho
                        );
                        assert!(
                            (m.u - want).length() < 1e-5,
                            "{}: velocity {:?} at ({x},{y},{z}), wanted {want:?}",
                            model.name(),
                            m.u
                        );
                    }
                }
            }
        }
    }

    /// Direct proof of the bounce-back rule in the module comment. Put a single
    /// population on a link aimed at a wall and check it comes back as its own
    /// opposite exactly one step later, with nothing else touched.
    #[test]
    fn a_population_aimed_at_a_wall_comes_back_reversed() {
        // A 3x3x3 fluid region with a solid halo, no collision at all
        // (tau = 1 with the equilibrium set to the current state would still
        // collide, so instead we test the raw transport by disabling relaxation
        // via s_e = 1 and comparing against the reference's own equilibrium).
        // Simpler: use the streaming addresses directly.
        let cfg = SolverConfig {
            set: VelocitySet::D3Q19,
            ..Default::default()
        };
        let d = PaddedDomain::uniform_fluid(UVec3::splat(1), [false; 3], cfg.set);
        // One fluid cell, solid on all sides: every link is a wall link.
        let lbm = ReferenceLbm::new(d, cfg);
        assert_eq!(
            lbm.domain.link_mask[lbm.domain.interior_linear(UVec3::ZERO) as usize] >> 1,
            (1u32 << 18) - 1
        );

        let cell = lbm.domain.interior_linear(UVec3::ZERO) as u64;
        let c = lbm.domain.coords(cell as u32);
        let ep = EsotericPull::new(lbm.domain.padded_cell_count(), 19);
        let nb = |k: usize| lbm.domain.linear(lbm.domain.neighbour(c, D3Q19_DIRS[k])) as u64;

        // For each step parity and each direction, the address a cell stores
        // f_ibar into must be the address it loads f_i from on the next step
        // with the flip applied. That *is* halfway bounce-back.
        for odd in [false, true] {
            for i in 1..19usize {
                let ibar = opposite(i);
                let stored = ep.store_slot(cell, ibar, odd, nb);
                // Next step, flipped because the upstream neighbour is solid.
                let loaded = ep.load_slot(cell, i, !odd ^ true, nb);
                assert_eq!(
                    stored, loaded,
                    "direction {i} (opposite {ibar}), odd={odd}: stored {stored}, loads {loaded}"
                );
            }
        }
    }

    /// A quiescent closed box must stay quiescent forever. This catches sign
    /// errors in bounce-back that a periodic test cannot see.
    #[test]
    fn a_closed_box_of_still_fluid_stays_still() {
        let cfg = SolverConfig {
            tau0: 0.55,
            smagorinsky_c: 0.0,
            ..Default::default()
        };
        let d = PaddedDomain::uniform_fluid(UVec3::new(5, 5, 5), [false; 3], cfg.set);
        let mut lbm = ReferenceLbm::new(d, cfg);
        for _ in 0..300 {
            lbm.step();
        }
        let mut worst = 0.0f32;
        for z in 0..5u32 {
            for y in 0..5u32 {
                for x in 0..5u32 {
                    let m = lbm.macroscopic(UVec3::new(x, y, z));
                    assert!(
                        (m.rho - 1.0).abs() < 1e-5,
                        "density {} at ({x},{y},{z})",
                        m.rho
                    );
                    worst = worst.max(m.u.length());
                }
            }
        }
        assert!(worst < 1e-6, "still fluid developed a velocity of {worst}");
    }

    /// Mass conservation with a wall present. Bounce-back is mass-conserving by
    /// construction — the population that goes in comes back out — so a closed
    /// box with an obstacle must also hold its total density.
    #[test]
    fn mass_is_conserved_around_an_obstacle() {
        let cfg = SolverConfig {
            tau0: 0.6,
            smagorinsky_c: 0.0,
            initial_velocity: Vec3::new(0.05, 0.0, 0.0),
            periodic: [true, true, true],
            ..Default::default()
        };
        let mut d = PaddedDomain::uniform_fluid(UVec3::splat(8), [true; 3], cfg.set);
        for z in 3..5u32 {
            for y in 3..5u32 {
                for x in 3..5u32 {
                    let idx = d.linear(UVec3::new(x, y, z)) as usize;
                    d.flags[idx] = flags::SOLID;
                }
            }
        }
        d.rebuild_link_mask(cfg.set.def());
        let mut lbm = ReferenceLbm::new(d, cfg);
        let m0 = lbm.total_mass();
        for _ in 0..400 {
            lbm.step();
        }
        let m1 = lbm.total_mass();
        let cells = lbm.domain.interior_cell_count() as f64;
        assert!(
            (m1 - m0).abs() / cells < 1e-6,
            "mass drifted from {m0} to {m1} with an obstacle present"
        );
    }

    #[test]
    fn fp16c_storage_still_conserves_mass() {
        // The codec is lossy, so the bound is looser than FP32 - but it must not
        // *drift*, because bounce-back and streaming only ever move values
        // around. A systematic leak would show up as a growing error.
        let cfg = SolverConfig {
            precision: ad_gpu::types::DdfPrecision::Fp16c,
            tau0: 0.6,
            smagorinsky_c: 0.0,
            initial_velocity: Vec3::new(0.05, -0.02, 0.01),
            ..Default::default()
        };
        let mut lbm = periodic_box(6, cfg);
        let m0 = lbm.total_mass();
        for _ in 0..300 {
            lbm.step();
        }
        let m1 = lbm.total_mass();
        let cells = lbm.domain.interior_cell_count() as f64;
        assert!(
            (m1 - m0).abs() / cells < 5e-5,
            "FP16C mass drifted by {} per cell",
            (m1 - m0) / cells
        );
    }

    /// A tilted inlet — a velocity along the inlet plane on top of the one
    /// through it — is what a louver aim hands the solver. This checks it on
    /// the CPU twin of the production arrangement: the field stays finite, the
    /// flow rate is set by the normal component alone, the air really is
    /// turned, and the inlet density does not move. The last is the sharp one:
    /// where the plane meets the duct wall the tangential velocity points into
    /// the wall, and an inlet that imposes it there piles air against the wall
    /// and reads the pile-up back through the density closure.
    #[test]
    fn a_tilted_inlet_turns_the_air_and_keeps_the_flow_rate() {
        // `validation/lbm/gpu.rs`'s production configuration in miniature: a
        // 6 x 6 bore through a solid block, opening into a box with open sides
        // and a convective, sponged far face.
        let dims = UVec3::new(24, 12, 12);
        let (nx, ny, nz) = (dims.x, dims.y, dims.z);
        let duct_end = 14u32;
        let bore = 3..9u32;
        let run = |inlet_velocity: Vec3| -> ReferenceLbm {
            let at = |x: u32, y: u32, z: u32| ((z * ny + y) * nx + x) as usize;
            let mut mask = vec![flags::FLUID; (nx * ny * nz) as usize];
            for z in 0..nz {
                for y in 0..ny {
                    let inside = bore.contains(&y) && bore.contains(&z);
                    for x in 0..duct_end {
                        if !inside {
                            mask[at(x, y, z)] = flags::SOLID;
                        }
                    }
                    if inside {
                        mask[at(0, y, z)] = flags::INLET;
                    }
                    mask[at(nx - 1, y, z)] = flags::OUTLET | flags::SPONGE;
                }
            }
            for x in duct_end..nx - 1 {
                for k in 0..ny {
                    mask[at(x, 0, k)] = flags::EQUILIBRIUM;
                    mask[at(x, ny - 1, k)] = flags::EQUILIBRIUM;
                    mask[at(x, k, 0)] = flags::EQUILIBRIUM;
                    mask[at(x, k, nz - 1)] = flags::EQUILIBRIUM;
                }
            }
            let cfg = SolverConfig {
                tau0: 0.55,
                smagorinsky_c: 0.0,
                inlet_velocity,
                inlet_normal: Vec3::X,
                outflow_velocity: 0.05,
                outlet_normal: Vec3::X,
                sponge_cells: 4,
                sponge_strength: 0.4,
                ..Default::default()
            };
            let mut lbm =
                ReferenceLbm::new(PaddedDomain::new(dims, [false; 3], &mask, cfg.set), cfg);
            for _ in 0..3000 {
                lbm.step();
            }
            lbm
        };
        let sum_over_bore = |lbm: &ReferenceLbm, x: u32, f: &dyn Fn(Macro) -> f32| -> f32 {
            let mut s = 0.0;
            for z in bore.clone() {
                for y in bore.clone() {
                    s += f(lbm.macroscopic(UVec3::new(x, y, z)));
                }
            }
            s
        };

        let straight = run(Vec3::new(0.05, 0.0, 0.0));
        // tan(26.6°) off the normal, with the normal component unchanged.
        let tilted = run(Vec3::new(0.05, 0.025, 0.0));

        for x in 0..duct_end {
            let bad = sum_over_bore(&tilted, x, &|m| {
                (!(m.rho.is_finite() && m.u.is_finite())) as u32 as f32
            });
            assert_eq!(bad, 0.0, "non-finite cells at x = {x}");
        }
        let flux = |lbm: &ReferenceLbm| sum_over_bore(lbm, duct_end / 2, &|m| m.rho * m.u.x);
        let (q0, q1) = (flux(&straight), flux(&tilted));
        assert!(q0 > 0.0, "no flow through the straight duct");
        assert!(
            (q1 - q0).abs() / q0 < 0.01,
            "the tilt changed the flow rate: {q0} -> {q1}"
        );

        let turned = sum_over_bore(&tilted, 1, &|m| m.u.y);
        assert!(
            turned > 0.0,
            "the air was not turned toward +y (sum of u_y {turned})"
        );

        let mut shift = 0.0f32;
        let mut worst = 0.0f32;
        for z in bore.clone() {
            for y in bore.clone() {
                let c = UVec3::new(0, y, z);
                let (s, t) = (
                    straight.macroscopic(c).rho - 1.0,
                    tilted.macroscopic(c).rho - 1.0,
                );
                shift = shift.max((t - s).abs());
                worst = worst.max(t.abs());
            }
        }
        println!(
            "tilted inlet: flow {q0:.5} -> {q1:.5}, inlet drho worst {worst:.4}, moved by up to {shift:.4}"
        );
        assert!(
            worst < 0.1,
            "inlet density {worst} is heading for the ±0.2 clamp"
        );
        // With the rim cells pushing air through the wall beside them, this was
        // 0.063. What is left sits one row in, where the imposed tangential
        // velocity steps from zero at the rim to its full value: the closure
        // reads that in-plane divergence at first order in u_t (0.025 here),
        // where the physics has it at second (~3 u_t^2 = 0.002). The cure is the
        // velocity bounce-back inlet `boundary.wgsl` already asks for, which
        // needs no density closure at all; until then this bound holds the line.
        assert!(
            shift < 0.035,
            "the tilt moved the inlet density by {shift}: air is piling up on the rim"
        );
    }

    /// The two density rules for an inlet plane standing in open fluid, in a
    /// laterally periodic column so the flow is one-dimensional. The plane
    /// closure is a *source*: the column downstream moves at the full inlet
    /// velocity and nothing is drawn from behind. The local moment is a fan
    /// disc that merely pushes: measured 0.46 U downstream and 0.54 U drawn in
    /// from behind, with a pressure jump across it, at tau 0.502 and 0.55
    /// alike. A vent has a duct behind it, so the vents use the closure; this
    /// pins the numbers that decided it.
    #[test]
    fn a_plane_inlet_in_open_fluid_is_a_source_with_the_closure_and_a_disc_without() {
        let dims = UVec3::new(60, 4, 4);
        let (nx, ny, nz) = (dims.x, dims.y, dims.z);
        let at = |x: u32, y: u32, z: u32| ((z * ny + y) * nx + x) as usize;
        let mut mask = vec![flags::FLUID; (nx * ny * nz) as usize];
        for z in 0..nz {
            for y in 0..ny {
                mask[at(0, y, z)] = flags::EQUILIBRIUM;
                mask[at(nx - 1, y, z)] = flags::EQUILIBRIUM;
                mask[at(15, y, z)] = flags::inlet_in_slot(1);
            }
        }
        let column = |local_density: bool| -> (f32, f32) {
            let mut cfg = SolverConfig {
                tau0: 0.502,
                smagorinsky_c: 0.0,
                periodic: [false, true, true],
                ..Default::default()
            };
            cfg.extra_inlets[0] = crate::config::InletSpec {
                velocity: Vec3::new(0.05, 0.0, 0.0),
                normal: Vec3::X,
                local_density,
            };
            let mut lbm =
                ReferenceLbm::new(PaddedDomain::new(dims, cfg.periodic, &mask, cfg.set), cfg);
            for _ in 0..3000 {
                lbm.step();
            }
            let u = |x: u32| lbm.macroscopic(UVec3::new(x, 2, 2)).u.x / 0.05;
            (u(30), u(6))
        };
        let (down, up) = column(false);
        println!("closure: {down:.3} U downstream, {up:.3} U upstream");
        assert!(
            (down - 1.0).abs() < 0.01,
            "the closure column is not at the inlet velocity: {down}"
        );
        assert!(up.abs() < 0.01, "the closure draws from behind: {up}");
        let (down, up) = column(true);
        println!("local moment: {down:.3} U downstream, {up:.3} U upstream");
        assert!(
            (0.3..0.6).contains(&down),
            "the disc's jet changed strength: {down}"
        );
        assert!(up > 0.3, "the disc stopped drawing from behind: {up}");
    }

    /// A vent standing free in the room: an inlet in slot 1, taking its
    /// density from the local moment. It must blow a jet at the speed it was
    /// given, stay finite, and not pile density up on itself — the closure's
    /// failure mode, which the local moment is there to avoid.
    #[test]
    fn a_free_standing_vent_blows_a_jet_at_its_own_speed() {
        let dims = UVec3::new(24, 12, 12);
        let (nx, ny, nz) = (dims.x, dims.y, dims.z);
        let at = |x: u32, y: u32, z: u32| ((z * ny + y) * nx + x) as usize;
        let mut mask = vec![flags::FLUID; (nx * ny * nz) as usize];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1 {
                        mask[at(x, y, z)] = flags::EQUILIBRIUM;
                    }
                }
            }
        }
        for z in 4..8u32 {
            for y in 4..8u32 {
                mask[at(6, y, z)] = flags::inlet_in_slot(1);
            }
        }
        let mut cfg = SolverConfig {
            tau0: 0.55,
            smagorinsky_c: 0.0,
            ..Default::default()
        };
        cfg.extra_inlets[0] = crate::config::InletSpec {
            velocity: Vec3::new(0.05, 0.0, 0.0),
            normal: Vec3::X,
            local_density: true,
        };
        let mut lbm = ReferenceLbm::new(PaddedDomain::new(dims, [false; 3], &mask, cfg.set), cfg);
        for _ in 0..1500 {
            lbm.step();
        }
        let mut worst_drho = 0.0f32;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let m = lbm.macroscopic(UVec3::new(x, y, z));
                    assert!(
                        m.rho.is_finite() && m.u.is_finite(),
                        "non-finite at ({x},{y},{z})"
                    );
                    if x == 6 && (4..8).contains(&y) && (4..8).contains(&z) {
                        worst_drho = worst_drho.max((m.rho - 1.0).abs());
                        assert!(
                            m.u.abs_diff_eq(Vec3::new(0.05, 0.0, 0.0), 1e-6),
                            "the vent reports its own velocity"
                        );
                    }
                }
            }
        }
        let leaving = lbm.macroscopic(UVec3::new(7, 5, 5)).u;
        let downstream = lbm.macroscopic(UVec3::new(9, 5, 5)).u;
        let upstream = lbm.macroscopic(UVec3::new(3, 5, 5)).u;
        println!(
            "vent: drho on the disc {worst_drho:.4}, u leaving {leaving}, 3 cells downstream {downstream}, 3 upstream {upstream}"
        );
        // A four-cell disc at this viscosity (Re ~ 12) spreads at once, so the
        // jet is checked where it leaves and only for its existence further on.
        assert!(
            leaving.x > 0.035,
            "the air leaving the disc is not at its speed: {leaving}"
        );
        assert!(
            downstream.x > 0.02,
            "the jet died within three cells: {downstream}"
        );
        assert!(
            upstream.x > 0.0,
            "a fan disc draws air in from behind: {upstream}"
        );
        assert!(
            worst_drho < 0.02,
            "density piled up on the disc: {worst_drho}"
        );
    }
}

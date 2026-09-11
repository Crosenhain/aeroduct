//! Collision operators, and the Smagorinsky closure that feeds them.
//!
//! Three operators, all behind [`CollisionModel`] so swapping one costs a config
//! field and a shader define rather than a restructure:
//!
//! - **TRT** (default). Two relaxation rates: the symmetric (even) part of the
//!   non-equilibrium carries the viscous stress and relaxes at `s_e = 1/tau`; the
//!   antisymmetric (odd) part carries no physics in the Navier-Stokes limit, so
//!   its rate `s_o` is free. Ginzburg's magic parameter fixes it:
//!   `Lambda = (1/s_e - 1/2)(1/s_o - 1/2)`. At `Lambda = 3/16` the bounce-back
//!   wall sits exactly halfway between the last fluid node and the first solid
//!   node **independently of viscosity**, which is the single most important
//!   property for us: `tau` here is 0.5006-0.502, and with BGK the wall would
//!   move by a large fraction of a cell as the resolution changes.
//!
//! - **BGK**. One rate. Kept because it makes TRT bugs obvious: if TRT is wired
//!   up wrongly it usually degenerates to something BGK-like, and the
//!   tau-independence test in `validation/lbm/poiseuille.rs` compares the two directly.
//!   At our operating point plain BGK is expected to be unstable; that is the
//!   result the test is designed to show, not a defect.
//!
//! - **Regularized BGK**. Projects `f^neq` onto the second-order Hermite basis
//!   before relaxing, throwing away the ghost moments entirely. Strictly more
//!   dissipative than TRT and first-order accurate at the wall, so it is a
//!   stability escape hatch, not a default.
//!
//! Everything operates on *shifted* populations `g_i = f_i - w_i`. That is free:
//! `f - f^eq == g - g^eq`, so the shift cancels identically inside every operator
//! and no intermediate of order 1 is ever formed. See [`crate::precision`].

use ad_gpu::lattice::{opposite, LatticeDef};

/// Speed of sound squared on a D3Qn lattice.
pub const CS2: f32 = 1.0 / 3.0;

/// Which collision operator the solver runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CollisionModel {
    /// Two-relaxation-time. The default, and the only one whose wall position is
    /// viscosity-independent.
    #[default]
    Trt,
    /// Single-relaxation-time. Comparison baseline.
    Bgk,
    /// Hermite-regularized BGK. Stability escape hatch.
    RegularizedBgk,
}

impl CollisionModel {
    /// Preprocessor flag selecting the operator in `shaders/lbm/collision.wgsl`.
    pub const fn shader_define(self) -> &'static str {
        match self {
            CollisionModel::Trt => "COLLIDE_TRT",
            CollisionModel::Bgk => "COLLIDE_BGK",
            CollisionModel::RegularizedBgk => "COLLIDE_RBGK",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            CollisionModel::Trt => "TRT",
            CollisionModel::Bgk => "BGK",
            CollisionModel::RegularizedBgk => "regularized BGK",
        }
    }
}

/// The two TRT relaxation rates for a given relaxation time and magic parameter.
///
/// `s_e = 1/tau` fixes the viscosity, `nu = c_s^2 (1/s_e - 1/2)`. Then
/// `Lambda = (1/s_e - 1/2)(1/s_o - 1/2)` with `1/s_e - 1/2 = tau - 1/2` gives
/// `s_o = 1 / (Lambda/(tau - 1/2) + 1/2)`.
///
/// `lambda <= 0` degenerates the operator to BGK (`s_o = s_e`), which is what
/// [`CollisionModel::Bgk`] uses so that one code path serves both.
#[inline]
pub fn trt_rates(tau: f32, lambda: f32) -> (f32, f32) {
    let s_e = 1.0 / tau;
    if lambda <= 0.0 {
        return (s_e, s_e);
    }
    let s_o = 1.0 / (lambda / (tau - 0.5) + 0.5);
    (s_e, s_o)
}

/// Smagorinsky-Lilly eddy viscosity, evaluated from the local non-equilibrium
/// second moment.
///
/// The reason to do LES this way in LBM rather than with finite differences is
/// that the strain-rate tensor is available *locally and algebraically*:
/// `Pi_ab = sum_i c_ia c_ib f_i^neq` is proportional to `S_ab` with no stencil at
/// all. On a staircased voxel wall that matters enormously — a finite-difference
/// velocity gradient across the steps would produce spurious strain and a wall of
/// artificial eddy viscosity exactly where the boundary layer is thinnest.
///
/// `tau_eff = 0.5 (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |Pi| / rho))`, with
/// `|Pi| = sqrt(Pi_ab Pi_ab)` over all nine components. `cs_smag = 0` returns
/// `tau0` exactly, so the model switches off cleanly.
///
/// `Cs` is 0.10-0.12 rather than Lilly's theoretical 0.17: the theoretical value
/// assumes homogeneous isotropic turbulence in the inertial range, and applied
/// unmodified near a wall it produces an eddy viscosity where the real flow is
/// laminar, thickening the boundary layer and inflating the pressure drop.
#[inline]
pub fn smagorinsky_tau(tau0: f32, tau_max: f32, cs_smag: f32, rho: f32, pi_norm: f32) -> f32 {
    if cs_smag <= 0.0 {
        return tau0;
    }
    const EIGHTEEN_ROOT_TWO: f32 = 25.455_845; // 18 * sqrt(2)
    let inner = tau0 * tau0 + EIGHTEEN_ROOT_TWO * cs_smag * cs_smag * pi_norm / rho.max(1e-6);
    (0.5 * (tau0 + inner.sqrt())).clamp(tau0, tau_max)
}

/// Frobenius norm of the non-equilibrium second moment, `sqrt(Pi_ab Pi_ab)`.
///
/// `neq[i]` is `f_i - f_i^eq`, equivalently `g_i - g_i^eq`.
pub fn pi_norm(def: &LatticeDef, neq: &[f32]) -> f32 {
    let mut pi = [[0.0f32; 3]; 3];
    for (i, n) in neq.iter().enumerate() {
        let c = def.directions[i];
        let c = [c.x as f32, c.y as f32, c.z as f32];
        for a in 0..3 {
            for b in 0..3 {
                pi[a][b] += n * c[a] * c[b];
            }
        }
    }
    let mut s = 0.0f32;
    for row in &pi {
        for v in row {
            s += v * v;
        }
    }
    s.sqrt()
}

/// Apply the collision operator in place to shifted populations.
///
/// `g` and `geq` are the shifted populations and their shifted equilibrium;
/// `s_e`/`s_o` come from [`trt_rates`] with the *effective* (LES-raised)
/// relaxation time. This is the reference implementation that
/// `shaders/lbm/collision.wgsl` mirrors; the GPU-vs-CPU test in
/// `validation/lbm/gpu.rs` is what keeps the two honest.
pub fn collide(model: CollisionModel, def: &LatticeDef, g: &mut [f32], geq: &[f32], s_e: f32, s_o: f32) {
    let q = def.q;
    match model {
        CollisionModel::Bgk => {
            for i in 0..q {
                g[i] -= s_e * (g[i] - geq[i]);
            }
        }
        CollisionModel::Trt => {
            // The rest population is purely symmetric, so it sees s_e alone.
            g[0] -= s_e * (g[0] - geq[0]);
            // Pairs are (1,2), (3,4), ...; walking i by 2 from 1 visits each once.
            let mut i = 1;
            while i < q {
                let j = i + 1;
                debug_assert_eq!(opposite(i), j);
                let (ni, nj) = (g[i] - geq[i], g[j] - geq[j]);
                let sym = 0.5 * (ni + nj);
                let asym = 0.5 * (ni - nj);
                g[i] -= s_e * sym + s_o * asym;
                g[j] -= s_e * sym - s_o * asym;
                i += 2;
            }
        }
        CollisionModel::RegularizedBgk => {
            // Rebuild f^neq from its second moment only: everything above second
            // order (the ghost moments) is discarded rather than relaxed.
            let mut pi = [[0.0f32; 3]; 3];
            for i in 0..q {
                let c = def.directions[i];
                let c = [c.x as f32, c.y as f32, c.z as f32];
                let n = g[i] - geq[i];
                for a in 0..3 {
                    for b in 0..3 {
                        pi[a][b] += n * c[a] * c[b];
                    }
                }
            }
            let trace = pi[0][0] + pi[1][1] + pi[2][2];
            let keep = 1.0 - s_e;
            for i in 0..q {
                let c = def.directions[i];
                let c = [c.x as f32, c.y as f32, c.z as f32];
                let mut qpi = 0.0f32;
                for a in 0..3 {
                    for b in 0..3 {
                        qpi += c[a] * c[b] * pi[a][b];
                    }
                }
                // Q_iab : Pi_ab with Q_iab = c_ia c_ib - c_s^2 delta_ab, and the
                // 1/(2 c_s^4) = 4.5 Hermite normalisation.
                let reg = def.weights[i] * 4.5 * (qpi - CS2 * trace);
                g[i] = geq[i] + keep * reg;
            }
        }
    }
}

/// Guo's forcing term, split for TRT.
///
/// `F_i = w_i [ 3 (c_i - u).F + 9 (c_i.u)(c_i.F) ]`, applied as
/// `(1 - s/2) F_i` with `s = s_e` on the symmetric part and `s_o` on the
/// antisymmetric part. For BGK (`s_e == s_o`) this collapses to the textbook
/// `(1 - 1/(2 tau)) F_i`.
///
/// The velocity passed in must already include the `F/(2 rho)` half-step
/// correction; see `ReferenceLbm::macroscopic`. Getting that wrong shows
/// up immediately as a first-order-in-`dx` Poiseuille profile.
pub fn apply_force(def: &LatticeDef, g: &mut [f32], u: [f32; 3], force: [f32; 3], s_e: f32, s_o: f32) {
    if force == [0.0; 3] {
        return;
    }
    let q = def.q;
    let udotf = u[0] * force[0] + u[1] * force[1] + u[2] * force[2];
    let term = |i: usize| -> f32 {
        let c = def.directions[i];
        let c = [c.x as f32, c.y as f32, c.z as f32];
        let cu = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
        let cf = c[0] * force[0] + c[1] * force[1] + c[2] * force[2];
        def.weights[i] * (3.0 * (cf - udotf) + 9.0 * cu * cf)
    };
    // c_0 = 0, so F_0 is entirely symmetric.
    g[0] += (1.0 - 0.5 * s_e) * term(0);
    let mut i = 1;
    while i < q {
        let j = i + 1;
        let (fi, fj) = (term(i), term(j));
        let sym = 0.5 * (fi + fj);
        let asym = 0.5 * (fi - fj);
        g[i] += (1.0 - 0.5 * s_e) * sym + (1.0 - 0.5 * s_o) * asym;
        g[j] += (1.0 - 0.5 * s_e) * sym - (1.0 - 0.5 * s_o) * asym;
        i += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::shifted_equilibrium;
    use ad_gpu::lattice::D3Q19;

    fn equilibrium_state(rho: f32, u: [f32; 3]) -> Vec<f32> {
        (0..19)
            .map(|i| {
                let c = D3Q19.directions[i];
                shifted_equilibrium(
                    D3Q19.weights[i],
                    [c.x as f32, c.y as f32, c.z as f32],
                    rho - 1.0,
                    rho,
                    u,
                )
            })
            .collect()
    }

    fn moments(g: &[f32]) -> (f32, [f32; 3]) {
        let mut drho = 0.0f32;
        let mut m = [0.0f32; 3];
        for (i, v) in g.iter().enumerate() {
            drho += v;
            let c = D3Q19.directions[i];
            m[0] += v * c.x as f32;
            m[1] += v * c.y as f32;
            m[2] += v * c.z as f32;
        }
        (1.0 + drho, m)
    }

    #[test]
    fn magic_parameter_recovers_the_definition() {
        for tau in [0.5006f32, 0.51, 0.6, 1.0, 2.0] {
            for lambda in [3.0 / 16.0f32, 0.25, 1.0 / 12.0] {
                let (s_e, s_o) = trt_rates(tau, lambda);
                let got = (1.0 / s_e - 0.5) * (1.0 / s_o - 0.5);
                assert!(
                    (got - lambda).abs() < 1e-5,
                    "tau {tau} lambda {lambda}: recovered {got}"
                );
                assert!(s_o > 0.0 && s_o < 2.0, "s_o = {s_o} is outside the stable range");
            }
        }
    }

    #[test]
    fn trt_degenerates_to_bgk_when_the_rates_coincide() {
        // Lambda = (tau - 1/2)^2 makes s_o == s_e, so TRT must reproduce BGK
        // exactly. If the symmetric/antisymmetric split is wrong this fails.
        let tau = 0.8f32;
        let lambda = (tau - 0.5) * (tau - 0.5);
        let (s_e, s_o) = trt_rates(tau, lambda);
        assert!((s_e - s_o).abs() < 1e-6);

        let geq = equilibrium_state(1.002, [0.05, -0.03, 0.02]);
        let mut a: Vec<f32> = geq.iter().enumerate().map(|(i, v)| v + 0.001 * (i as f32 - 9.0)).collect();
        let mut b = a.clone();
        collide(CollisionModel::Trt, &D3Q19, &mut a, &geq, s_e, s_o);
        collide(CollisionModel::Bgk, &D3Q19, &mut b, &geq, s_e, s_e);
        for i in 0..19 {
            assert!((a[i] - b[i]).abs() < 1e-6, "direction {i}: TRT {} vs BGK {}", a[i], b[i]);
        }
    }

    /// Project an arbitrary perturbation onto the space of genuine
    /// non-equilibria, i.e. fields whose zeroth and first moments vanish.
    ///
    /// `delta_i = p_i - w_i (dm + 3 c_i . dmom)` removes exactly the mass and
    /// momentum `p` was carrying, using the fact that `sum_i w_i = 1`,
    /// `sum_i w_i c_i = 0` and `sum_i w_i c_ia c_ib = delta_ab / 3`.
    fn project_out_conserved_moments(p: &[f32]) -> Vec<f32> {
        let (rho, mom) = moments(p);
        let dm = rho - 1.0;
        (0..19)
            .map(|i| {
                let c = D3Q19.directions[i];
                let cm = c.x as f32 * mom[0] + c.y as f32 * mom[1] + c.z as f32 * mom[2];
                p[i] - D3Q19.weights[i] * (dm + 3.0 * cm)
            })
            .collect()
    }

    #[test]
    fn every_operator_conserves_mass_and_momentum() {
        let geq = equilibrium_state(1.004, [0.06, 0.01, -0.04]);
        let raw: Vec<f32> = (0..19).map(|i| 0.002 * ((i * 7 % 11) as f32 - 5.0)).collect();
        let neq0 = project_out_conserved_moments(&raw);
        // Sanity: the projection really did produce a valid non-equilibrium, or
        // the rest of this test would be vacuous.
        let (r, m) = moments(&neq0);
        assert!((r - 1.0).abs() < 1e-6 && m.iter().all(|v| v.abs() < 1e-6), "projection failed");
        let start: Vec<f32> = geq.iter().zip(&neq0).map(|(a, b)| a + b).collect();

        for model in [CollisionModel::Trt, CollisionModel::Bgk, CollisionModel::RegularizedBgk] {
            let (s_e, s_o) = trt_rates(0.55, 3.0 / 16.0);
            let mut g = start.clone();
            let (rho_before, m_before) = moments(&g);
            collide(model, &D3Q19, &mut g, &geq, s_e, s_o);
            let (rho_after, m_after) = moments(&g);
            assert!(
                (rho_after - rho_before).abs() < 2e-6,
                "{}: density {rho_before} -> {rho_after}",
                model.name()
            );
            for a in 0..3 {
                assert!(
                    (m_after[a] - m_before[a]).abs() < 2e-6,
                    "{}: momentum {a} {} -> {}",
                    model.name(),
                    m_before[a],
                    m_after[a]
                );
            }
        }
    }

    #[test]
    fn equilibrium_is_a_fixed_point_of_every_operator() {
        let geq = equilibrium_state(1.001, [0.04, -0.02, 0.03]);
        for model in [CollisionModel::Trt, CollisionModel::Bgk, CollisionModel::RegularizedBgk] {
            let mut g = geq.clone();
            let (s_e, s_o) = trt_rates(0.7, 3.0 / 16.0);
            collide(model, &D3Q19, &mut g, &geq, s_e, s_o);
            for i in 0..19 {
                assert!(
                    (g[i] - geq[i]).abs() < 1e-7,
                    "{} moved direction {i} away from equilibrium",
                    model.name()
                );
            }
        }
    }

    #[test]
    fn regularization_preserves_the_second_moment_it_keeps() {
        // A regularized f^neq must reproduce the same Pi_ab it was built from,
        // which is the defining property of the Hermite projection.
        let geq = equilibrium_state(1.0, [0.03, 0.0, 0.0]);
        let mut g: Vec<f32> = geq.iter().enumerate().map(|(i, v)| v + 0.0005 * (i as f32 % 5.0 - 2.0)).collect();
        let neq_before: Vec<f32> = g.iter().zip(&geq).map(|(a, b)| a - b).collect();
        let pi_before = pi_norm(&D3Q19, &neq_before);

        // s_e = 0 leaves f^neq untouched apart from the projection itself.
        collide(CollisionModel::RegularizedBgk, &D3Q19, &mut g, &geq, 0.0, 0.0);
        let neq_after: Vec<f32> = g.iter().zip(&geq).map(|(a, b)| a - b).collect();
        let pi_after = pi_norm(&D3Q19, &neq_after);
        assert!(
            (pi_after - pi_before).abs() < 1e-6 * pi_before.max(1e-6),
            "|Pi| changed under projection: {pi_before} -> {pi_after}"
        );
    }

    #[test]
    fn smagorinsky_switches_off_cleanly_and_only_ever_raises_tau() {
        let tau0 = 0.5006f32;
        assert_eq!(smagorinsky_tau(tau0, 1.0, 0.0, 1.0, 100.0), tau0);
        for pi in [0.0f32, 1e-6, 1e-3, 1e-1, 10.0] {
            let t = smagorinsky_tau(tau0, 1.0, 0.11, 1.0, pi);
            assert!(t >= tau0 - 1e-7, "tau_eff {t} fell below tau0 {tau0}");
            assert!(t <= 1.0, "tau_eff {t} exceeded the clamp");
        }
        // Monotone in |Pi|: more strain, more eddy viscosity.
        let a = smagorinsky_tau(tau0, 1.0, 0.11, 1.0, 1e-4);
        let b = smagorinsky_tau(tau0, 1.0, 0.11, 1.0, 1e-2);
        assert!(b > a, "eddy viscosity did not increase with strain: {a} then {b}");
    }

    #[test]
    fn guo_forcing_injects_exactly_the_requested_momentum() {
        // sum_i F_i == 0 and sum_i c_i F_i == (1 - s/2) ... the standard result is
        // that Guo's term adds F/2 to the momentum during collision, with the
        // other F/2 coming from the half-step correction in the velocity.
        let def = &D3Q19;
        let u = [0.05f32, 0.02, -0.01];
        let force = [1e-5f32, -2e-5, 3e-6];
        let (s_e, s_o) = trt_rates(0.6, 3.0 / 16.0);
        // Apply to a zero field so the moments measure the source term itself.
        // Measuring the delta on top of a full population set would drown a
        // 1e-5 force in the f32 rounding of numbers a thousand times larger.
        let mut g = vec![0.0f32; 19];
        apply_force(def, &mut g, u, force, s_e, s_o);
        let (rho_after, m_after) = moments(&g);
        assert!(
            (rho_after - 1.0).abs() < 1e-10,
            "forcing changed the density by {}",
            rho_after - 1.0
        );
        for a in 0..3 {
            let want = (1.0 - 0.5 * s_o) * force[a];
            assert!(
                (m_after[a] - want).abs() < 1e-11,
                "axis {a}: momentum gain {}, expected {want}",
                m_after[a]
            );
        }
    }
}

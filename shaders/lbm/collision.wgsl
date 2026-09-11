// Collision operators and the Smagorinsky closure.
//
// A line-for-line mirror of ad_solver::collision. Two things keep them honest:
// validation/lbm/gpu.rs compares the two field-by-field on a real device, and the
// analytic cases in validation/lbm/ run against the CPU version, so a
// divergence shows up as a GPU-only failure with the physics already cleared.
//
// Everything operates on *shifted* populations g = f - w. The shift cancels
// identically inside every operator (f - f^eq == g - g^eq), so no intermediate
// of order 1 is ever formed and the FP16C storage keeps its accuracy. See
// ad_solver::precision for why that ordering is load-bearing.

#include "lbm/common.wgsl"

// f_i^eq - w_i, built already-shifted.
//
// `drho` is passed separately from `rho` on purpose: the caller obtained it as a
// sum of small numbers, and recomputing it as `rho - 1.0` here would throw away
// exactly the precision the DDF shift was introduced to protect.
fn shifted_equilibrium(i: u32, drho: f32, rho: f32, u: vec3<f32>) -> f32 {
    let c = vec3<f32>(C[i]);
    let cu = dot(c, u);
    let uu = dot(u, u);
    return W[i] * (drho + rho * (3.0 * cu + 4.5 * cu * cu - 1.5 * uu));
}

// (s_e, s_o) for TRT. s_e = 1/tau fixes the viscosity; the magic parameter
// Lambda = (1/s_e - 1/2)(1/s_o - 1/2) fixes the free odd rate. lambda <= 0
// degenerates to BGK, which is how COLLIDE_BGK reuses this path.
fn trt_rates(tau: f32) -> vec2<f32> {
    let s_e = 1.0 / tau;
#if COLLIDE_TRT
    let s_o = 1.0 / (U.trt_lambda / (tau - 0.5) + 0.5);
    return vec2<f32>(s_e, s_o);
#else
    return vec2<f32>(s_e, s_e);
#endif
}

// Non-equilibrium second moment, Pi_ab = sum_i c_ia c_ib (g_i - geq_i).
//
// This is why LES is cheap in LBM: the strain rate is available locally and
// algebraically, with no stencil. On a staircased voxel wall that is decisive -
// a finite-difference velocity gradient across the steps would manufacture
// strain, and with it a layer of artificial eddy viscosity exactly where the
// boundary layer is thinnest.
fn nonequilibrium_stress(
    g: ptr<function, array<f32, Q_CONST>>,
    geq: ptr<function, array<f32, Q_CONST>>,
) -> mat3x3<f32> {
    var pi = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var i = 0u; i < Q; i = i + 1u) {
        let c = vec3<f32>(C[i]);
        let n = (*g)[i] - (*geq)[i];
        pi[0] = pi[0] + n * c.x * c;
        pi[1] = pi[1] + n * c.y * c;
        pi[2] = pi[2] + n * c.z * c;
    }
    return pi;
}

fn frobenius(pi: mat3x3<f32>) -> f32 {
    return sqrt(dot(pi[0], pi[0]) + dot(pi[1], pi[1]) + dot(pi[2], pi[2]));
}

// tau_eff = 0.5 (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |Pi| / rho)), clamped.
//
// Cs is 0.10-0.12 rather than Lilly's 0.17: the theoretical constant assumes
// inertial-range isotropic turbulence, and near a wall it produces eddy
// viscosity where the real flow is laminar.
fn smagorinsky_tau(rho: f32, pi_norm: f32) -> f32 {
    if (U.smagorinsky_c <= 0.0) { return U.tau0; }
    let inner = U.tau0 * U.tau0
        + 25.455845 * U.smagorinsky_c * U.smagorinsky_c * pi_norm / max(rho, 1e-6);
    return clamp(0.5 * (U.tau0 + sqrt(inner)), U.tau0, U.tau_max);
}

fn collide(
    g: ptr<function, array<f32, Q_CONST>>,
    geq: ptr<function, array<f32, Q_CONST>>,
    s_e: f32,
    s_o: f32,
) {
#if COLLIDE_RBGK
    // Regularized BGK: rebuild f^neq from its second moment alone, discarding
    // every ghost moment rather than relaxing it. Strictly more dissipative than
    // TRT and only first-order at the wall, so this is a stability escape hatch,
    // not a default.
    let pi = nonequilibrium_stress(g, geq);
    let trace = pi[0].x + pi[1].y + pi[2].z;
    let keep = 1.0 - s_e;
    for (var i = 0u; i < Q; i = i + 1u) {
        let c = vec3<f32>(C[i]);
        let qpi = dot(c, pi[0]) * c.x + dot(c, pi[1]) * c.y + dot(c, pi[2]) * c.z;
        // Q_iab : Pi_ab with Q_iab = c_ia c_ib - c_s^2 delta_ab, times the
        // 1/(2 c_s^4) = 4.5 Hermite normalisation.
        (*g)[i] = (*geq)[i] + keep * W[i] * 4.5 * (qpi - CS2 * trace);
    }
#else
    // The rest population is purely symmetric, so it only ever sees s_e.
    (*g)[0] = (*g)[0] - s_e * ((*g)[0] - (*geq)[0]);
    for (var i = 1u; i < Q; i = i + 2u) {
        let j = i + 1u;
        let ni = (*g)[i] - (*geq)[i];
        let nj = (*g)[j] - (*geq)[j];
        let sym = 0.5 * (ni + nj);
        let asym = 0.5 * (ni - nj);
        (*g)[i] = (*g)[i] - (s_e * sym + s_o * asym);
        (*g)[j] = (*g)[j] - (s_e * sym - s_o * asym);
    }
#endif
}

// Guo's forcing term, split for TRT.
//
// F_i = w_i [3 (c_i - u).F + 9 (c_i.u)(c_i.F)], applied as (1 - s/2) F_i with
// s_e on the symmetric part and s_o on the antisymmetric part. For BGK the two
// rates coincide and this collapses to the textbook (1 - 1/(2 tau)) F_i.
//
// The velocity passed in must already carry the F/(2 rho) half-step correction.
// Getting that wrong turns the Poiseuille profile first-order, which is exactly
// what validation/lbm/poiseuille.rs would catch.
fn apply_force(g: ptr<function, array<f32, Q_CONST>>, u: vec3<f32>, s_e: f32, s_o: f32) {
    let force = U.body_force;
    if (all(force == vec3<f32>(0.0))) { return; }
    let udotf = dot(u, force);
    let ke = 1.0 - 0.5 * s_e;
    let ko = 1.0 - 0.5 * s_o;
    (*g)[0] = (*g)[0] + ke * W[0] * (3.0 * (0.0 - udotf));
    for (var i = 1u; i < Q; i = i + 2u) {
        let j = i + 1u;
        let ci = vec3<f32>(C[i]);
        let cj = vec3<f32>(C[j]);
        let fi = W[i] * (3.0 * (dot(ci, force) - udotf) + 9.0 * dot(ci, u) * dot(ci, force));
        let fj = W[j] * (3.0 * (dot(cj, force) - udotf) + 9.0 * dot(cj, u) * dot(cj, force));
        let sym = 0.5 * (fi + fj);
        let asym = 0.5 * (fi - fj);
        (*g)[i] = (*g)[i] + ke * sym + ko * asym;
        (*g)[j] = (*g)[j] + ke * sym - ko * asym;
    }
}

// rho - 1 and rho*u, in the order the DDF shift demands: the small deviations
// are summed first and the 1 is added exactly once, at the end. Summing f_i
// directly would put 19 numbers of order w_i through an addition whose
// interesting part is the 1e-4 deviation from unity.
struct Moments {
    drho: f32,
    momentum: vec3<f32>,
};

fn moments(g: ptr<function, array<f32, Q_CONST>>) -> Moments {
    var m: Moments;
    m.drho = 0.0;
    m.momentum = vec3<f32>(0.0);
    for (var i = 0u; i < Q; i = i + 1u) {
        let v = (*g)[i];
        m.drho = m.drho + v;
        m.momentum = m.momentum + v * vec3<f32>(C[i]);
    }
    return m;
}

fn fill_equilibrium(
    geq: ptr<function, array<f32, Q_CONST>>,
    drho: f32,
    rho: f32,
    u: vec3<f32>,
) {
    for (var i = 0u; i < Q; i = i + 1u) {
        (*geq)[i] = shifted_equilibrium(i, drho, rho, u);
    }
}

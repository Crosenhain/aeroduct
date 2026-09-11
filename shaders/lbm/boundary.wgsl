// Boundary conditions, dispatched off the ad_gpu::flags bitfield.
//
// Priority is SOLID > INLET > OUTLET > EQUILIBRIUM, with SPONGE orthogonal to
// all of them. ad_solver::boundary::CellKind is the same decision in Rust and
// carries a test for the ordering.
//
// Solid walls do not appear here at all. Halfway bounce-back is implicit in the
// Esoteric Pull load: a link whose upstream neighbour is solid reads with the
// opposite step parity, which retrieves the population this cell pushed into the
// wall link last step. Zero extra memory traffic, no separate kernel, no branch
// beyond selecting one of a pair's two slots. What is left for this file is
// everything that is *not* a no-slip wall.
//
// # What is deliberately not here yet
//
// Single-node interpolated bounce-back (Marson et al., arXiv:2009.04604) needs
// the per-link wall distance q from ad_gpu::BoundaryLink. The link is already
// identified at the point of the load - the same mask bit that selects the flip
// would select the interpolation - so Wave 3 adds a lookup and a lerp, not a
// restructure. The link table is built and uploaded by ad_solver::boundary
// today so that the data path exists before the shader needs it.

#include "lbm/common.wgsl"

// Anti-bounce-back: replace the populations arriving on blocked links with the
// value that fixes the *density* at rho_ref instead of the velocity at zero.
//
//   f_i = -f_ibar^post + 2 w_i rho_ref (1 + (c.u)^2/(2 c_s^4) - u^2/(2 c_s^2))
//
// written in shifted form, so no unshifted population is ever built:
//
//   g_i = -g_loaded + 2 w_i [(rho_ref - 1) + rho_ref ((9/2)(c.u)^2 - (3/2) u^2)]
//
// `g_loaded` on a blocked link is exactly f_ibar^post - w, which is why the
// bounce-back value can be reused directly rather than re-read.
// Is link `i` at cell `c` a genuine outflow link?
//
// Two conditions, and both matter. The direction must point back into the domain
// against the outlet normal (so it is an *unknown* population arriving from
// outside), and the upstream cell must leave the domain **only** along the
// normal axis. The second is what keeps a corner sane: at the edge where the
// outlet plane meets a side wall, a diagonal like (-1, +1, 0) does point inward
// against a +x normal, but its upstream cell is outside in y as well. Treating
// it as outflow puts an anti-bounce-back reflection on what is really a wall,
// and several of those meeting at one cell is an amplifying map. Measured: the
// production duct test NaNs at the outlet corner after 1500-4000 steps without
// this second condition, for every collision operator.
fn is_outflow_link(c: vec3<u32>, i: u32) -> bool {
    let n = U.outlet_normal;
    if (!any(n != vec3<f32>(0.0))) { return true; }     // ungated: every blocked link
    if (dot(vec3<f32>(C[i]), n) >= 0.0) { return false; }
    let up = vec3<i32>(c) - C[i];
    let lo = vec3<i32>(U.offset);
    let hi = lo + vec3<i32>(U.interior) - vec3<i32>(1);
    let outside = (up < lo) | (up > hi);
    let on_normal_axis = abs(n) > vec3<f32>(0.5);
    // Outside only where the normal points.
    return !any(outside & !on_normal_axis);
}

fn anti_bounce_back(c: vec3<u32>, mask: u32, g: ptr<function, array<f32, Q_CONST>>, u: vec3<f32>) {
    let uu = dot(u, u);
    let rho_ref = U.rho_ref;
    for (var i = 1u; i < Q; i = i + 1u) {
        if ((mask & (1u << i)) == 0u) { continue; }
        if (!is_outflow_link(c, i)) { continue; }
        let cu = dot(vec3<f32>(C[i]), u);
        let abb = 2.0 * W[i] * ((rho_ref - 1.0) + rho_ref * (4.5 * cu * cu - 1.5 * uu));
        (*g)[i] = -(*g)[i] + abb;
    }
}

// Density at a velocity inlet, from the populations that are actually known.
//
// The naive thing - take rho from the raw moment of everything loaded - is
// wrong, and wrong in a way that quietly kills the flow rather than blowing it
// up. An inlet plane sits against the edge of the domain, so the populations
// arriving from "outside" are really this cell's own outgoing populations
// bounced off the halo. Those carry momentum -u instead of +u, and the moment
// they contribute is short by 6 u_n * sum_{c.n>0} w_i = u_n. The inlet then
// emits at a density a few percent low, which lowers it further next step. The
// measured fixed point for u_n = 0.05 was rho = 0.864, with the duct downstream
// completely stagnant: the inlet had no stagnation pressure left to drive it.
//
// The standard closure instead uses only the known populations. With `n` the
// inward normal, the unknowns are exactly the links with `c_i . n > 0`, so
//
//   rho = (A + 2B) / (1 - u_n),   A = sum_{c.n=0} f_i,  B = sum_{c.n<0} f_i
//
// and in shifted form the weight sums cancel exactly - `W_0 + 2 W_- = 1` for any
// axis-aligned normal, by the symmetry of the weight table - leaving
//
//   drho = (A_g + 2 B_g + u_n) / (1 - u_n)
//
// with no unshifted quantity anywhere. See ad_solver::precision for why that
// matters.
// The closure assumes a flat plane: it treats every link with `c_i . n <= 0` as
// carrying genuine neighbour data. At the *edge* of an inlet patch - a bore only
// six cells across is mostly edge - some of those links are blocked by the duct
// wall instead, and the value they carry is a reflection rather than data. The
// closure then returns a density that is merely wrong; fed back through the
// equilibrium it emits, it becomes a density that runs away.
//
// Clamping to a physically sane band is what makes the inlet keep the property
// it was chosen for: it cannot blow up. +/-20% of the reference density is far
// wider than any duct pressure this solver will see (0.2 in lattice density is
// 0.067 in lattice pressure, hundreds of Pascals at the operating point), so the
// clamp never binds on a well-posed problem and always binds on a broken one.
//
// The real fix is a velocity bounce-back inlet - halfway bounce-back with a
// prescribed wall velocity, `f_i += 6 w_i rho (c_i . u_wall)` on blocked links -
// which needs no density closure at all, is unconditionally mass-conserving, and
// is the same mechanism the moving lid in validation/lbm/cavity.rs wants. That
// belongs with Wave 3's interpolated bounce-back, which touches the same lines.
fn inlet_drho(g: ptr<function, array<f32, Q_CONST>>, n: vec3<f32>, u: vec3<f32>) -> f32 {
    var a = 0.0;
    var b = 0.0;
    for (var i = 0u; i < Q; i = i + 1u) {
        let cn = dot(vec3<f32>(C[i]), n);
        if (cn == 0.0) { a = a + (*g)[i]; }
        else if (cn < 0.0) { b = b + (*g)[i]; }
    }
    let un = dot(u, n);
    let drho = (a + 2.0 * b + un) / max(1.0 - un, 0.1);
    return clamp(drho, U.rho_ref - 1.0 - 0.2, U.rho_ref - 1.0 + 0.2);
}

// The velocity one inlet cell imposes: the configured one, less its component
// normal to any wall beside the cell.
//
// A tilted inlet (a louver aim) carries a velocity *along* the inlet plane.
// Where the plane meets the duct wall, part of that goes through an
// impermeable wall - into it on one side of the duct, out of it on the other -
// and bounce-back hands it straight to `inlet_drho`, which reads the
// reflections as data: pile-up on one rim, suction on the other, ±0.06 in
// density against a physical ~3 u_t^2. A component *along* the wall cancels
// between its two diagonals and is harmless. So the wall-normal component goes
// (the no-penetration condition) and the rest stays. Only the four axis links
// in the plane are tested, which covers axis-aligned walls; a corner loses
// both in-plane components.
//
// With no tilt the in-plane components are exactly zero and `u` comes back bit
// for bit. CPU twin, with the measurement: ReferenceLbm::inlet_velocity_at.
fn inlet_velocity_at(mask: u32, u_in: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var u = u_in;
    for (var i = 1u; i < Q; i = i + 1u) {
        let c = vec3<f32>(C[i]);
        if (dot(abs(c), vec3<f32>(1.0)) != 1.0 || dot(c, n) != 0.0) { continue; }
        // Bit i: the neighbour this population streams in from, x - c_i, is
        // solid, so there is a wall on the -c_i side.
        if ((mask & (1u << i)) != 0u) {
            u = u - c * dot(u, c);
        }
    }
    return u;
}

// Relax the whole population set toward a target equilibrium by `weight`.
// Used by the convective outlet and by the sponge; both are "absorb, do not
// reflect" treatments and differ only in what they aim at and how hard.
fn relax_toward_equilibrium(
    g: ptr<function, array<f32, Q_CONST>>,
    weight: f32,
    drho: f32,
    rho: f32,
    u: vec3<f32>,
) {
    for (var i = 0u; i < Q; i = i + 1u) {
        (*g)[i] = (*g)[i] + weight * (shifted_equilibrium(i, drho, rho, u) - (*g)[i]);
    }
}

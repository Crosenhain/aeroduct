// The fused stream-collide kernel, and the initialiser that feeds it.
//
// One dispatch per time step. Esoteric Pull streams in place, so there is no
// second DDF copy and no separate streaming pass: the load *is* the stream, and
// the store is next step's stream. See ad_solver::reference for the derivation
// of the addressing and of the implicit bounce-back.
//
// Workgroup shape is (WG_X, 1, 1). X is the fastest-varying axis, so a workgroup
// covers a contiguous run of cells and every direction buffer is read with
// consecutive lanes touching consecutive addresses - the coalescing optimum.
// WG_X is a define so it can be swept; 64 and 128 are the values worth trying.

#include "lbm/common.wgsl"
#include "lbm/generated_ddf.wgsl"
#include "lbm/collision.wgsl"
#include "lbm/boundary.wgsl"

// Initialise every fluid cell to equilibrium at (1, initial_velocity).
//
// The store uses the parity of the step *before* step 0, which is the only
// choice that also populates wall links: on a blocked link the flipped load of
// step 0 lands precisely on the slot this write fills. Any other convention
// leaves the first step reading uninitialised memory on every boundary link,
// which shows up as a boundary layer of garbage that then diffuses inward.
@compute @workgroup_size(#WG_X, 1, 1)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= U.dims)) { return; }
    let cell = cell_index(gid);
    if (!is_fluid(get_flags(cell))) { return; }

    var g: array<f32, Q_CONST>;
    fill_equilibrium(&g, 0.0, 1.0, U.initial_velocity);
    store_ddf(cell, gid, U.step_parity == 0u, &g);
}

@compute @workgroup_size(#WG_X, 1, 1)
fn stream_collide(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= U.dims)) { return; }
    let cell = cell_index(gid);
    let fl = get_flags(cell);
    // Solid cells never execute. They own no populations: both slots on a wall
    // link belong to the fluid cell across it.
    if (!is_fluid(fl)) { return; }

    let odd = U.step_parity != 0u;
    let mask = link_mask[cell];

    var g: array<f32, Q_CONST>;
    load_ddf(cell, gid, odd, mask, &g);

    var mom = moments(&g);

    let is_inlet = (fl & FLAG_INLET) != 0u;
    let is_outlet = !is_inlet && (fl & FLAG_OUTLET) != 0u;
    let is_equil = !is_inlet && !is_outlet && (fl & FLAG_EQUILIBRIUM) != 0u;

    if (is_outlet && U.outlet_anti_bounce_back != 0u) {
        // Anti-bounce-back needs the wall velocity, which is not known until the
        // moments are in. Predict with the plain bounce-back result and correct
        // once; the correction to the velocity term is O(Ma^2) of an already
        // small quantity, so a second pass buys nothing.
        //
        // OFF BY DEFAULT. See SolverConfig::outlet_anti_bounce_back: this is
        // unstable under real through-flow, and the convective term below pins
        // the outlet density just as effectively without negating anything.
        let rho_p = 1.0 + mom.drho;
        anti_bounce_back(gid, mask, &g, mom.momentum / rho_p);
        mom = moments(&g);
    }

    var drho = mom.drho;
    var rho = 1.0 + drho;
    var u = (mom.momentum + 0.5 * U.body_force) / rho;

    if (is_inlet) {
        // Equilibrium inlet: prescribe the velocity and let the density float,
        // so the inlet pressure is free and a duct pressure drop is measurable.
        // The density comes from the *known* populations only - see
        // `inlet_drho` for why the raw moment is not usable here. The boundary
        // has no mechanism by which to diverge, which is why it is the v1 inlet.
        let slot = inlet_slot(fl);
        let spec = U.inlet_vel[slot];
        let n = U.inlet_nrm[slot].xyz;
        u = inlet_velocity_at(mask, spec.xyz, n);
        if (spec.w > 0.5) {
            // Free-standing (a vent in the room): every neighbour is fluid, so
            // the arriving populations are all data and their sum is the local
            // density. f_eq(rho, u) sums to the rho that arrived, so the reset
            // conserves mass: a fan disc, not a leak. The clamp is the same
            // sanity band the closure has, and binds only on a broken problem.
            drho = clamp(mom.drho, U.rho_ref - 1.0 - 0.2, U.rho_ref - 1.0 + 0.2);
        } else {
            drho = inlet_drho(&g, n, u);
        }
        rho = 1.0 + drho;
    } else if (is_equil) {
        // Open box side. Ambient pressure and no mean flow, so the exit jet can
        // entrain surrounding air instead of being confined by a wall.
        rho = U.rho_ref;
        drho = U.rho_ref - 1.0;
        u = vec3<f32>(0.0);
    }

    var geq: array<f32, Q_CONST>;
    fill_equilibrium(&geq, drho, rho, u);

    let pi = nonequilibrium_stress(&g, &geq);
    let tau = smagorinsky_tau(rho, frobenius(pi));
    let rates = trt_rates(tau);

    if (is_inlet || is_equil) {
        // Discard the non-equilibrium entirely.
        for (var i = 0u; i < Q; i = i + 1u) { g[i] = geq[i]; }
    } else {
        collide(&g, &geq, rates.x, rates.y);
        apply_force(&g, u, rates.x, rates.y);
    }

    if (is_outlet && U.outflow_velocity > 0.0) {
        // Convective outflow, df/dt + U df/dn = 0, discretised upwind with the
        // upstream population approximated by the local equilibrium at the
        // reference density. The weight U/(1+U) is the exact factor from
        // f(t+1) = (f + U f_up) / (1 + U).
        //
        // The approximation is the upstream *population*, not the condition: a
        // true upwind stencil would need the neighbour's full population set,
        // which Esoteric Pull does not make addressable. Pressure is pinned
        // separately by the anti-bounce-back above, so what this term supplies
        // is the outgoing-wave damping, and that is what it is good at.
        let w = U.outflow_velocity / (1.0 + U.outflow_velocity);
        relax_toward_equilibrium(&g, w, U.rho_ref - 1.0, U.rho_ref, u);
    }

    let sigma = sponge_sigma(gid, fl);
    if (sigma > 0.0) {
        // Absorb rather than reflect: relax toward equilibrium at the reference
        // density but the *local* velocity, so the layer eats acoustic and
        // vortical content without imposing a mean flow of its own.
        relax_toward_equilibrium(&g, sigma, U.rho_ref - 1.0, U.rho_ref, u);
    }

    store_ddf(cell, gid, odd, &g);
}

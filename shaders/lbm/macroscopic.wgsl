// Export rho and u for the render and metrics crates.
//
// Runs at the same step parity the *next* stream-collide would load with, so it
// sees the current state without disturbing it: the DDF bindings are read-only
// here in practice, and the bounce-back flip is applied identically, because
// anything that wants "all q distributions of cell x at time t" has to ask the
// Esoteric Pull addressing rather than index the buffers.
//
// Textures are sized to the *interior* grid, so consumers never see the solid
// halo the solver needs. Per CONTRACT.md rule 5, storage textures are write-only
// in the portable spec: this pass writes through a storage binding, and the
// render and metrics passes read through a separate *sampled* view of the same
// texture with a linear sampler.

#include "lbm/common.wgsl"
#include "lbm/generated_ddf.wgsl"
#include "lbm/collision.wgsl"
#include "lbm/boundary.wgsl"

@group(2) @binding(0) var velocity_out: texture_storage_3d<rgba16float, write>;
@group(2) @binding(1) var density_out: texture_storage_3d<r32float, write>;
#if MACRO_BUFFER
// Validation only. A texture readback needs 256-byte row alignment and a
// staging copy per slice; a plain buffer is one map_async and exact. Costs
// 16 B/cell of write traffic on this pass only, so it is off in production.
@group(2) @binding(2) var<storage, read_write> macro_buffer: array<vec4<f32>>;
#endif

@compute @workgroup_size(#WG_X, 1, 1)
fn macroscopic(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= U.interior)) { return; }
    let c = gid + U.offset;
    let cell = cell_index(c);

    let fl = get_flags(cell);
    var rho = 1.0;
    var u = vec3<f32>(0.0);
    if (is_fluid(fl)) {
        var g: array<f32, Q_CONST>;
        load_ddf(cell, c, U.step_parity != 0u, link_mask[cell], &g);
        let m = moments(&g);
        rho = 1.0 + m.drho;
        u = (m.momentum + 0.5 * U.body_force) / rho;

        // Report what an equilibrium boundary actually holds, not the moment of
        // the incoming populations.
        //
        // At an inlet the cell's state after the step *is* f^eq(rho, u_inlet) -
        // that is the definition of the boundary. Its pre-collision moment is
        // something else entirely: half of its links come back off the domain
        // wall carrying the reversed prescribed velocity, so the raw moment can
        // even have the wrong sign. Reporting it would put a nonsense vector on
        // the inlet plane of every visualisation and in every metric taken
        // there. ad_solver::reference::macroscopic applies the same rule.
        if ((fl & FLAG_INLET) != 0u) {
            let slot = inlet_slot(fl);
            u = inlet_velocity_at(link_mask[cell], U.inlet_vel[slot].xyz, U.inlet_nrm[slot].xyz);
        } else if ((fl & FLAG_EQUILIBRIUM) != 0u) {
            rho = U.rho_ref;
            u = vec3<f32>(0.0);
        }
    }

    let p = vec3<i32>(gid);
    // The w channel carries the density deviation, not a pad. `ad-render`'s
    // derive pass reads it as `rho - 1` to build the pressure field, so writing
    // 0 here renders Pressure as a flat zero everywhere -- a plausible-looking
    // picture that is entirely wrong. Store the deviation rather than the raw
    // density so the value stays small and keeps its precision in f16.
    textureStore(velocity_out, p, vec4<f32>(u, rho - 1.0));
    textureStore(density_out, p, vec4<f32>(rho, 0.0, 0.0, 0.0));
#if MACRO_BUFFER
    macro_buffer[interior_index(gid)] = vec4<f32>(u, rho);
#endif
}

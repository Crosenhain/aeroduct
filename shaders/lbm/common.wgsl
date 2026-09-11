// Uniforms, grid indexing and the sponge profile.
//
// Included by every LBM entry point *after* the generated prelude, which brings
// in the lattice tables (C, W, opposite, pair_first), the FP16C codec, the DDF
// bindings and the Esoteric Pull transport bodies.
//
// The include below must stay the first thing in the file that is not a comment.
// The generated head begins with the module's `enable` directives - FP16C
// storage needs `enable wgpu_int16;` - and WGSL rejects a directive that follows
// any declaration, so anything declared above it would be a compile error rather
// than a subtle one.
//
// The dimensions in `U.dims` are the *padded* ones. Every non-periodic axis
// carries one cell of solid halo at each end, because Esoteric Pull parks half a
// boundary cell's populations in the slot belonging to the cell outside the
// domain; see ad_solver::boundary for the full argument. Interior cell (i,j,k)
// is padded cell (i,j,k) + U.offset.

#include "lbm/generated_head.wgsl"

// The generated prelude and the ad-gpu lattice prelude must agree about q.
const_assert Q_CONST == Q;

struct Uniforms {
    dims: vec3<u32>,            // padded grid
    step_parity: u32,           // 0 or 1
    interior: vec3<u32>,        // the caller's grid
    cell_count: u32,            // padded cell count
    offset: vec3<u32>,          // padded = interior + offset
    flag_words: u32,
    inlet_velocity: vec3<f32>,
    tau0: f32,
    initial_velocity: vec3<f32>,
    trt_lambda: f32,
    body_force: vec3<f32>,
    smagorinsky_c: f32,
    tau_max: f32,
    outflow_velocity: f32,
    rho_ref: f32,
    sponge_strength: f32,
    sponge_cells: u32,
    periodic: u32,              // bit a set means axis a wraps
    total_steps: u32,
    pad0: u32,
    outlet_normal: vec3<f32>,   // outward, for the anti-bounce-back gate
    pad1: f32,
    inlet_normal: vec3<f32>,    // inward, for the inlet density closure
    outlet_anti_bounce_back: u32,
    // The four inlet slots, chosen per cell by the two high bits of its flag
    // byte: slot 0 is the duct-mouth inlet, 1-3 are vents standing in the room.
    // xyz = lattice velocity, w > 0 = free-standing (density from the local
    // moment rather than the plane closure).
    inlet_vel: array<vec4<f32>, 4>,
    inlet_nrm: array<vec4<f32>, 4>,   // xyz = inward normal of the slot's plane
};

fn inlet_slot(fl: u32) -> u32 {
    return (fl >> 6u) & 3u;
}

@group(0) @binding(0) var<uniform> U: Uniforms;
// One flag byte per cell, four per word. Keeping it a byte matters: the traffic
// model in ad_gpu::ddf budgets exactly 1 B/cell/step for it.
@group(0) @binding(1) var<storage, read> cell_flags: array<u32>;
// Bit i set means "direction i's upstream neighbour is solid, so flip the load
// parity". Precomputed on the CPU because the geometry is static, which turns
// q-1 scattered neighbour-flag reads into one coalesced word.
@group(0) @binding(2) var<storage, read> link_mask: array<u32>;
// Reserved for Wave 3's single-node interpolated bounce-back. Packed
// ad_gpu::BoundaryLink records: cell in the low word, direction and the
// quantised wall distance q in the high word. Bound and uploaded today so the
// data path exists before the shader consumes it; unused declarations cost
// nothing, and having the binding in place keeps the layout stable.
@group(0) @binding(3) var<storage, read> boundary_links: array<vec2<u32>>;

const CS2: f32 = 0.3333333333333333;

fn cell_index(c: vec3<u32>) -> u32 {
    return (c.z * U.dims.y + c.y) * U.dims.x + c.x;
}

fn interior_index(c: vec3<u32>) -> u32 {
    return (c.z * U.interior.y + c.y) * U.interior.x + c.x;
}

// Neighbour in direction `d`, wrapping on every axis.
//
// Wrapping unconditionally is both safe and free. On a padded axis the wrap can
// only ever trigger for a halo cell, and halo cells return before they get here;
// on a periodic axis it is the intended behaviour. The payoff is that no
// invocation can compute an out-of-range index, so there is no bounds check on
// the hot path and no way for a boundary cell to read someone else's slot.
fn neighbour_index(c: vec3<u32>, d: u32) -> u32 {
    let n = vec3<i32>(U.dims);
    let p = vec3<i32>(c) + C[d];
    let w = ((p % n) + n) % n;
    return u32((w.z * n.y + w.y) * n.x + w.x);
}

fn get_flags(cell: u32) -> u32 {
    return (cell_flags[cell >> 2u] >> ((cell & 3u) * 8u)) & 0xffu;
}

fn is_fluid(f: u32) -> bool {
    return (f & FLAG_SOLID) == 0u;
}

// Distance in cells to the nearest closed domain face, saturating at `limit`.
// Periodic axes have no face and are skipped.
fn face_distance(c: vec3<u32>, limit: u32) -> u32 {
    var d = limit;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if ((U.periodic & (1u << a)) != 0u) { continue; }
        let n = U.dims[a];
        let v = c[a];
        d = min(d, min(v, n - 1u - v));
    }
    return d;
}

// Quadratic sponge ramp: zero at the inner edge of the layer, full strength at
// the face. A step change in damping reflects nearly as much as the wall it
// replaced, which is the whole point of grading it.
//
// Gated on the SPONGE flag, not on distance alone. The distance is only the
// *profile*; the flag is the caller's statement of where the absorbing layer
// actually is. Grading by distance alone puts damping wherever the domain
// happens to be narrow — inside the duct, if the duct passes within
// `sponge_cells` of a face — which silently eats the flow being measured. That
// is not hypothetical: it cost 12% of the streamwise flux in the first run of
// the production-configuration test, and combined with the outlet's
// anti-bounce-back it diverged outright.
fn sponge_sigma(c: vec3<u32>, fl: u32) -> f32 {
    if ((fl & FLAG_SPONGE) == 0u || U.sponge_cells == 0u || U.sponge_strength <= 0.0) {
        return 0.0;
    }
    let d = face_distance(c, U.sponge_cells);
    if (d >= U.sponge_cells) { return 0.0; }
    let t = f32(U.sponge_cells - d) / f32(U.sponge_cells);
    return U.sponge_strength * t * t;
}

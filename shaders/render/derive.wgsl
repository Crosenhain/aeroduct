// Derived fields: solver macroscopic state -> four sampled scalars + velocity.
//
// See `crates/ad-render/src/fields.rs` for the physics and the unit algebra.
// The two things worth restating at the point of use:
//
//   * gradients are masked so a stencil arm never reaches into a solid cell;
//   * Q is stored normalised, `Q~ = Q_lattice * (D_h_cells / u_lb)^2`, which is
//     one uniform multiply here and a stable isolevel for the user.

#include "common.wgsl"

struct DeriveUniform {
    src_dims: vec3<u32>,
    downsample: u32,
    dst_dims: vec3<u32>,
    flags_present: u32,
    speed_scale: f32,
    q_scale: f32,
    vort_scale: f32,
    pressure_scale: f32,
};

// ad_gpu::types::flags::SOLID
const FLAG_SOLID: u32 = 1u;

@group(0) @binding(0) var<uniform> du: DeriveUniform;
@group(0) @binding(1) var src_macro: texture_3d<f32>;
@group(0) @binding(2) var src_flags: texture_3d<u32>;
@group(0) @binding(3) var out_scalars: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var out_velocity: texture_storage_3d<rgba16float, write>;

fn in_domain(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(du.src_dims));
}

// Only ever called after `in_domain`, so the flag texture is never sampled out
// of range and the 1x1x1 stand-in used when there are no flags is never touched.
fn cell_is_solid(c: vec3<i32>) -> bool {
    if (du.flags_present == 0u) {
        return false;
    }
    return (textureLoad(src_flags, c, 0).x & FLAG_SOLID) != 0u;
}

fn cell_velocity(c: vec3<i32>) -> vec3<f32> {
    return textureLoad(src_macro, c, 0).xyz;
}

fn usable(c: vec3<i32>) -> bool {
    return in_domain(c) && !cell_is_solid(c);
}

// du/dx_axis by central difference, degrading to one-sided rather than
// differencing across a wall or off the edge of the domain.
//
// Differencing into a solid cell reads whatever the solver left there (zero, in
// practice) and manufactures a shear of `u / dx` right at the wall. The result
// is a bright shell of fake Q-criterion wrapped around the entire duct, which
// looks exactly like a boundary layer and is entirely an artefact.
fn axis_derivative(c: vec3<i32>, axis: vec3<i32>, u0: vec3<f32>) -> vec3<f32> {
    let cp = c + axis;
    let cm = c - axis;
    let okp = usable(cp);
    let okm = usable(cm);
    if (okp && okm) {
        return (cell_velocity(cp) - cell_velocity(cm)) * 0.5;
    }
    if (okp) {
        return cell_velocity(cp) - u0;
    }
    if (okm) {
        return u0 - cell_velocity(cm);
    }
    return vec3<f32>(0.0);
}

fn frobenius_sq(m: mat3x3<f32>) -> f32 {
    return dot(m[0], m[0]) + dot(m[1], m[1]) + dot(m[2], m[2]);
}

struct CellResult {
    scalars: vec4<f32>,
    velocity: vec3<f32>,
};

fn derive_cell(c: vec3<i32>) -> CellResult {
    let raw = textureLoad(src_macro, c, 0);
    let u0 = raw.xyz;

    // Column k of J is du/dx_k, i.e. J[k][i] = du_i / dx_k in WGSL's
    // column-major indexing. Matches `fields::q_criterion` on the CPU.
    let j = mat3x3<f32>(
        axis_derivative(c, vec3<i32>(1, 0, 0), u0),
        axis_derivative(c, vec3<i32>(0, 1, 0), u0),
        axis_derivative(c, vec3<i32>(0, 0, 1), u0),
    );
    let jt = transpose(j);
    let s = (j + jt) * 0.5;
    let om = (j - jt) * 0.5;
    let q = 0.5 * (frobenius_sq(om) - frobenius_sq(s));
    let w = vec3<f32>(
        j[1].z - j[2].y,
        j[2].x - j[0].z,
        j[0].y - j[1].x,
    );

    var r: CellResult;
    r.velocity = u0 * du.speed_scale;
    r.scalars = vec4<f32>(
        length(u0) * du.speed_scale,
        q * du.q_scale,
        length(w) * du.vort_scale,
        raw.w * du.pressure_scale,
    );
    return r;
}

@compute @workgroup_size(#DERIVE_WG_X, #DERIVE_WG_Y, #DERIVE_WG_Z)
fn derive_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= du.dst_dims)) {
        return;
    }

    let f = i32(du.downsample);
    let base = vec3<i32>(gid) * f;

    // Box filter over the block. Note that the *derivatives* are still taken on
    // the fine lattice and only the resulting scalars are averaged: coarsening
    // the velocity first and differencing afterwards would halve the gradient
    // resolution exactly where it matters, in thin shear layers.
    var acc = vec4<f32>(0.0);
    var vel = vec3<f32>(0.0);
    var fluid = 0.0;
    var total = 0.0;

    for (var z = 0; z < f; z = z + 1) {
        for (var y = 0; y < f; y = y + 1) {
            for (var x = 0; x < f; x = x + 1) {
                let c = base + vec3<i32>(x, y, z);
                if (!in_domain(c)) {
                    continue;
                }
                total = total + 1.0;
                if (cell_is_solid(c)) {
                    continue;
                }
                let r = derive_cell(c);
                acc = acc + r.scalars;
                vel = vel + r.velocity;
                fluid = fluid + 1.0;
            }
        }
    }

    var inv = 0.0;
    if (fluid > 0.0) {
        inv = 1.0 / fluid;
    }
    var frac = 0.0;
    if (total > 0.0) {
        frac = fluid / total;
    }

    let p = vec3<i32>(gid);
    textureStore(out_scalars, p, acc * inv);
    // The fluid fraction in `w` is what stops the volume bleeding into the duct
    // walls, and what tells the brick reduction which voxels are real.
    textureStore(out_velocity, p, vec4<f32>(vel * inv, frac));
}

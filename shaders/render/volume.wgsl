// Front-to-back volume raymarch with empty-space skipping.
//
// Runs as a compute pass at half resolution into a premultiplied RGBA target.
// Compute rather than a blended fullscreen draw for two reasons: the output is
// half-res and has to be depth-aware-upsampled anyway, so there is nothing to
// blend into directly; and a compute pass can carry a profiler timestamp scope
// through `ad_gpu::Profiler`, which today only hands out compute-pass writes.
// The compositing arithmetic in `composite.wgsl` is exactly the
// `ONE, ONE_MINUS_SRC_ALPHA` blend it replaces.
//
// Three invariants hold this pass together. Break any one and the other two stop
// being debuggable:
//
//   1. **Opacity correction.** `alpha_c = 1 - (1 - alpha_ref)^(h / h_ref)`, every
//      sample, no exceptions. Without it the image depends on the step size, so
//      any change to the accelerator changes the picture and you can no longer
//      tell a skipping bug from a skipping success.
//   2. **The skip is conservative.** A brick distance of `k` means the
//      `(2k-1)^3` box of bricks around it is inactive, so advancing to that
//      box's exit cannot miss anything. Proven in accel.rs.
//   3. **Reverse-Z everywhere.** The opaque depth buffer is reverse-Z; it is
//      linearised with `z_near / depth` and nothing else.

#include "common.wgsl"

struct VolumeUniform {
    // Raymarch step as a fraction of a voxel. 1.0 is the Nyquist-ish default.
    step_scale: f32,
    // Reference step the transfer function's opacities are defined against.
    h_ref_mm: f32,
    // Extra global density on top of the baked LUT.
    density: f32,
    max_steps: u32,

    tf_lo: f32,
    tf_inv_span: f32,
    tf_log: u32,
    tf_show_clamp: u32,

    under_color: vec4<f32>,
    over_color: vec4<f32>,

    // xyz = unit direction toward the key light, w = shading strength (0 = off).
    light_dir: vec4<f32>,

    // Advanced by the golden ratio each frame, see `post::golden_ratio_advance`.
    jitter_offset: f32,
    skip_enabled: u32,
    sdf_present: u32,
    front_alpha: f32,

    sdf_min_mm: vec4<f32>,
    // xyz = 1 / sdf extent, w = surface offset in mm.
    sdf_inv_size: vec4<f32>,
    // xyz = ambient tint, w = specular strength.
    ambient: vec4<f32>,
};

struct BrickUniform {
    brick_dims: vec3<u32>,
    brick_size: u32,
    field_dims: vec3<u32>,
    max_skip: u32,
    support_lo: f32,
    support_hi: f32,
    gap_lo: f32,
    gap_hi: f32,
    channel: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> cam: Camera;

@group(1) @binding(0) var<uniform> fu: FieldsInfo;
@group(1) @binding(1) var field_sampler: sampler;
@group(1) @binding(2) var field_scalars: texture_3d<f32>;
@group(1) @binding(3) var field_velocity: texture_3d<f32>;

@group(2) @binding(0) var<uniform> bu: BrickUniform;
@group(2) @binding(1) var brick_dist: texture_3d<u32>;
@group(2) @binding(2) var brick_minmax: texture_3d<f32>;

@group(3) @binding(0) var<uniform> vu: VolumeUniform;
@group(3) @binding(1) var tf_lut: texture_2d<f32>;
@group(3) @binding(2) var lut_sampler: sampler;
@group(3) @binding(3) var blue_noise: texture_2d<f32>;
@group(3) @binding(4) var opaque_depth: texture_depth_2d;
@group(3) @binding(5) var solid_sdf: texture_3d<f32>;
@group(3) @binding(6) var out_color: texture_storage_2d<rgba16float, write>;
@group(3) @binding(7) var out_front: texture_storage_2d<r32float, write>;

fn field_uvw(p: vec3<f32>) -> vec3<f32> {
    return (p - fu.volume_min_mm) / fu.volume_size_mm;
}

fn sample_scalar(uvw: vec3<f32>) -> f32 {
    return pick_channel(textureSampleLevel(field_scalars, field_sampler, uvw, 0.0), fu.channel);
}

// Data value -> normalised LUT coordinate. Mirrors
// `TransferFunction::normalise`; the log branch is a uniform test.
fn tf_coord(v: f32) -> f32 {
    var x = v;
    if (vu.tf_log != 0u) {
        x = log(max(v, 1.0e-30));
    }
    return (x - vu.tf_lo) * vu.tf_inv_span;
}

// Exit parameter of the axis-aligned box of bricks guaranteed inactive around
// `brick` when its Chebyshev distance is `d`.
//
// d = k means every brick within Chebyshev distance k-1 is inactive, so the box
// runs from brick - (k-1) to brick + (k-1) inclusive, i.e. one brick past that
// on the high side in world units.
fn brick_skip_exit(origin: vec3<f32>, dir: vec3<f32>, brick: vec3<i32>, d: u32) -> f32 {
    let r = f32(d) - 1.0;
    let bw = f32(bu.brick_size) * fu.voxel_mm;
    let lo = fu.volume_min_mm + (vec3<f32>(brick) - vec3<f32>(r)) * bw;
    let hi = fu.volume_min_mm + (vec3<f32>(brick) + vec3<f32>(r + 1.0)) * bw;
    let t = ray_box(origin, dir, lo, hi);
    return t.y;
}

fn brick_coord(p: vec3<f32>) -> vec3<i32> {
    let bw = f32(bu.brick_size) * fu.voxel_mm;
    return vec3<i32>(floor((p - fu.volume_min_mm) / bw));
}

// Signed distance to the duct wall, in mm. Positive outside the solid.
fn sample_sdf(p: vec3<f32>) -> f32 {
    let uvw = (p - vu.sdf_min_mm.xyz) * vu.sdf_inv_size.xyz;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
        return 1.0e6;
    }
    return textureSampleLevel(solid_sdf, field_sampler, uvw, 0.0).x - vu.sdf_inv_size.w;
}

// Central-difference gradient of the displayed scalar, in texture space. Used
// as a surface normal for shading: on a soft isosurface it is exactly the
// isosurface normal, which is what makes the mode look like a surface at all.
fn field_gradient(uvw: vec3<f32>) -> vec3<f32> {
    let e = 1.0 / vec3<f32>(fu.dims);
    let dx = sample_scalar(uvw + vec3<f32>(e.x, 0.0, 0.0)) - sample_scalar(uvw - vec3<f32>(e.x, 0.0, 0.0));
    let dy = sample_scalar(uvw + vec3<f32>(0.0, e.y, 0.0)) - sample_scalar(uvw - vec3<f32>(0.0, e.y, 0.0));
    let dz = sample_scalar(uvw + vec3<f32>(0.0, 0.0, e.z)) - sample_scalar(uvw - vec3<f32>(0.0, 0.0, e.z));
    return vec3<f32>(dx, dy, dz);
}

fn shade(base: vec3<f32>, uvw: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let g = field_gradient(uvw);
    let len = length(g);
    if (len < 1.0e-8) {
        return base;
    }
    // The scalar rises into the feature, so the outward normal is -grad.
    let n = -g / len;
    let l = normalize(vu.light_dir.xyz);
    let ndl = max(dot(n, l), 0.0);
    // Half-Lambert: a hard terminator on a translucent medium reads as a
    // shading error rather than as shape.
    let wrapped = ndl * 0.5 + 0.5;
    let h = normalize(l - view);
    let spec = pow(max(dot(n, h), 0.0), 48.0) * vu.ambient.w;
    let lit = base * (vu.ambient.xyz + vec3<f32>(wrapped)) + vec3<f32>(spec);
    return mix(base, lit, clamp(vu.light_dir.w, 0.0, 1.0));
}

@compute @workgroup_size(#VOLUME_WG_X, #VOLUME_WG_Y, 1)
fn volume_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(out_color);
    if (gid.x >= out_dims.x || gid.y >= out_dims.y) {
        return;
    }
    let px = vec2<i32>(gid.xy);

    let ndc = pixel_to_ndc(vec2<f32>(gid.xy) + vec2<f32>(0.5), vec2<f32>(out_dims));
    let world_ray = camera_ray(cam, ndc);
    // March in the lattice frame, where the field, the bricks and the SDF are
    // all axis-aligned boxes. The install pose is rigid, so `t` measures the
    // same distance in both frames: the depth clip below and the TAA front
    // stay in world terms with no conversion.
    var ray: Ray;
    ray.origin = (fu.world_to_lattice * vec4<f32>(world_ray.origin, 1.0)).xyz;
    ray.dir = normalize((fu.world_to_lattice * vec4<f32>(world_ray.dir, 0.0)).xyz);

    let lo = fu.volume_min_mm;
    let hi = fu.volume_min_mm + fu.volume_size_mm;
    let span = ray_box(ray.origin, ray.dir, lo, hi);

    var t0 = span.x;
    var t1 = span.y;
    // Clamp the entry to the near plane. Without this, a camera inside the duct
    // has a negative t_enter and the marcher starts behind the eye, which turns
    // into a wall of opaque fog the moment you fly in for a close look.
    t0 = max(t0, cam.eye_near.w);

    // Composite against the opaque depth buffer. The half-res pixel maps to a
    // 2x2 block at full res; taking the nearest of the block would leak the
    // volume through thin geometry, so we take the *closest* depth, i.e. the
    // largest reverse-Z value, which is conservative in the safe direction.
    let full = textureDimensions(opaque_depth);
    let scale = vec2<f32>(full) / vec2<f32>(out_dims);
    let base = vec2<i32>(vec2<f32>(gid.xy) * scale);
    var dmax = 0.0;
    for (var j = 0; j < 2; j = j + 1) {
        for (var i = 0; i < 2; i = i + 1) {
            let c = clamp(base + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(full) - vec2<i32>(1));
            dmax = max(dmax, textureLoad(opaque_depth, c, 0));
        }
    }
    if (dmax > 0.0) {
        // View-space distance along the view axis, converted to distance along
        // this ray by dividing out the cosine to the view direction.
        let t_view = linear_depth(cam.eye_near.w, dmax);
        let cosine = max(dot(world_ray.dir, cam.forward_tan.xyz), 1.0e-4);
        t1 = min(t1, t_view / cosine);
    }

    if (t1 <= t0 || span.y <= span.x) {
        textureStore(out_color, px, vec4<f32>(0.0));
        textureStore(out_front, px, vec4<f32>(-1.0, 0.0, 0.0, 0.0));
        return;
    }

    let h = max(vu.step_scale * fu.voxel_mm, 1.0e-4);

    // Blue-noise first-sample jitter, advanced by the golden ratio each frame.
    // Blue noise puts the error in the high spatial frequencies where TAA can
    // remove it; a white-noise or per-pixel-hash jitter leaves low-frequency
    // blotches that TAA preserves faithfully.
    let nsize = textureDimensions(blue_noise);
    let ncoord = vec2<i32>(px % vec2<i32>(nsize));
    let noise = textureLoad(blue_noise, ncoord, 0).x;
    var t = t0 + fract(noise + vu.jitter_offset) * h;

    var color = vec3<f32>(0.0);
    var alpha = 0.0;
    var t_front = -1.0;
    var steps = 0u;

    loop {
        if (t >= t1 || alpha > 0.99 || steps >= vu.max_steps) {
            break;
        }
        steps = steps + 1u;
        let p = ray.origin + ray.dir * t;

        if (vu.skip_enabled != 0u) {
            let b = brick_coord(p);
            var d = i32(bu.max_skip);
            if (all(b >= vec3<i32>(0)) && all(b < vec3<i32>(bu.brick_dims))) {
                d = i32(textureLoad(brick_dist, b, 0).x);
            }
            if (d > 0) {
                let exit = brick_skip_exit(ray.origin, ray.dir, b, u32(d));
                // Always advance by at least a hair, so a numerically
                // degenerate box can never spin the loop.
                t = max(exit, t) + 1.0e-3;
                continue;
            }
        }

        // Free acceleration: the geometry crate's signed distance field lets us
        // sphere-trace straight through the duct wall instead of stepping
        // through solid material that can never contribute.
        if (vu.sdf_present != 0u) {
            let s = sample_sdf(p);
            if (s < 0.0) {
                t = t + max(-s, h);
                continue;
            }
        }

        let uvw = field_uvw(p);
        let occupancy = textureSampleLevel(field_velocity, field_sampler, uvw, 0.0).w;
        if (occupancy > 0.0) {
            let v = sample_scalar(uvw);
            let tn = tf_coord(v);
            var rgba = textureSampleLevel(
                tf_lut, lut_sampler, vec2<f32>(clamp(tn, 0.0, 1.0), 0.5), 0.0);
            if (vu.tf_show_clamp != 0u) {
                if (tn < 0.0) {
                    rgba = vec4<f32>(vu.under_color.xyz, rgba.w);
                } else if (tn > 1.0) {
                    rgba = vec4<f32>(vu.over_color.xyz, rgba.w);
                }
            }

            let a_ref = clamp(rgba.w * vu.density * occupancy, 0.0, 1.0);
            if (a_ref > 0.0) {
                // Opacity correction. See invariant 1 at the top of this file.
                let a = 1.0 - pow(1.0 - a_ref, h / vu.h_ref_mm);
                var rgb = rgba.xyz;
                if (vu.light_dir.w > 0.0 && a > 0.01) {
                    rgb = shade(rgb, uvw, ray.dir);
                }
                // Front-to-back, premultiplied.
                color = color + (1.0 - alpha) * a * rgb;
                alpha = alpha + (1.0 - alpha) * a;
                if (t_front < 0.0 && alpha > vu.front_alpha) {
                    t_front = t;
                }
            }
        }

        t = t + h;
    }

    textureStore(out_color, px, vec4<f32>(color, alpha));
    // The depth of the volume's "front" is what TAA reprojects the volume with,
    // since a participating medium has no motion vectors of its own.
    textureStore(out_front, px, vec4<f32>(t_front, 0.0, 0.0, 0.0));
}

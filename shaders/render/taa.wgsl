// Temporal antialiasing, with a progressive-accumulation mode.
//
// Two things about this scene make plain TAA insufficient:
//
//   * **The volume has no motion vectors.** A participating medium is not a
//     surface, so there is nothing to reproject. What we do have is the depth at
//     which the accumulated alpha first crossed a threshold — the volume's
//     visual "front" — which is written by volume.wgsl. Reprojecting that point
//     as if it were a surface is not exact, but it is right wherever the volume
//     looks like a surface, which is exactly where temporal error is visible.
//   * **The camera stops moving a lot.** This is a design tool: the user sets a
//     view and then stares at it. When both camera and sim are static, the
//     blend switches from a fixed-weight exponential filter to a true running
//     mean over jittered samples, which converges to a properly supersampled
//     image instead of hovering at whatever the blend weight allows. That is
//     also the 4x-supersampled screenshot path.

#include "common.wgsl"

struct TaaUniform {
    // Weight given to the current frame in the moving-average mode.
    blend: f32,
    // Neighbourhood clipping width in standard deviations.
    variance_gamma: f32,
    // Non-zero to switch to progressive accumulation.
    accumulate: u32,
    // Number of samples already integrated, in accumulation mode.
    accum_count: u32,
};

@group(0) @binding(0) var<uniform> cam: Camera;

@group(1) @binding(0) var<uniform> tu: TaaUniform;
@group(1) @binding(1) var linear_sampler: sampler;
@group(1) @binding(2) var current_tex: texture_2d<f32>;
@group(1) @binding(3) var history_tex: texture_2d<f32>;
@group(1) @binding(4) var motion_tex: texture_2d<f32>;
@group(1) @binding(5) var gbuf_depth: texture_depth_2d;
@group(1) @binding(6) var volume_front: texture_2d<f32>;
@group(1) @binding(7) var out_tex: texture_storage_2d<rgba16float, write>;

// Clip `history` toward `center` until it lies inside the AABB.
//
// Clipping, not clamping. Clamping each channel independently moves the colour
// to the nearest corner of the box, which changes its hue; clipping along the
// line back to the neighbourhood mean preserves it. The visible difference is
// coloured fringing on moving edges.
fn clip_aabb(lo: vec3<f32>, hi: vec3<f32>, history: vec3<f32>) -> vec3<f32> {
    let center = 0.5 * (hi + lo);
    let extent = max(0.5 * (hi - lo), vec3<f32>(1.0e-5));
    let v = history - center;
    let a = abs(v / extent);
    let m = max(a.x, max(a.y, a.z));
    if (m > 1.0) {
        return center + v / m;
    }
    return history;
}

// NDC delta -> UV delta. The y axis flips between the two spaces, and getting
// that sign wrong produces smearing that looks exactly like a too-high blend
// weight, which is a great way to spend an afternoon tuning the wrong number.
fn ndc_delta_to_uv(d: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(d.x * 0.5, -d.y * 0.5);
}

@compute @workgroup_size(8, 8, 1)
fn taa_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_tex);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let px = vec2<i32>(gid.xy);
    let fsize = vec2<f32>(size);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / fsize;
    let current = textureLoad(current_tex, px, 0).xyz;

    if (tu.accumulate != 0u) {
        // Progressive mode: the camera is static, so the history is aligned by
        // construction and only the jitter differs. A running mean converges;
        // an exponential blend does not.
        let history = textureLoad(history_tex, px, 0).xyz;
        let n = f32(tu.accum_count);
        let w = 1.0 / (n + 1.0);
        textureStore(out_tex, px, vec4<f32>(mix(history, current, w), 1.0));
        return;
    }

    // --- reprojection ---
    var motion = vec2<f32>(0.0);
    var valid = false;

    let depth = textureLoad(gbuf_depth, px, 0);
    if (depth > 0.0) {
        motion = textureLoad(motion_tex, px, 0).xy;
        valid = true;
    } else {
        // No geometry here, so try the volume's front surface.
        let vsize = textureDimensions(volume_front);
        let vpx = clamp(
            vec2<i32>(vec2<f32>(gid.xy) * vec2<f32>(vsize) / fsize),
            vec2<i32>(0),
            vec2<i32>(vsize) - vec2<i32>(1),
        );
        let front = textureLoad(volume_front, vpx, 0).x;
        if (front > 0.0) {
            let ndc = pixel_to_ndc(vec2<f32>(gid.xy) + vec2<f32>(0.5), fsize);
            let ray = camera_ray(cam, ndc);
            let world = vec4<f32>(ray.origin + ray.dir * front, 1.0);
            let cur_clip = cam.view_proj_unjit * world;
            let prev_clip = cam.prev_view_proj_unjit * world;
            if (cur_clip.w > 0.0 && prev_clip.w > 0.0) {
                motion = cur_clip.xy / cur_clip.w - prev_clip.xy / prev_clip.w;
                valid = true;
            }
        }
    }

    let prev_uv = uv - ndc_delta_to_uv(motion);
    if (!valid || any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0))) {
        // Nothing to reproject onto: take the current frame and start again.
        textureStore(out_tex, px, vec4<f32>(current, 1.0));
        return;
    }

    // --- neighbourhood statistics ---
    // Variance clipping (Salvi) rather than a min/max box: the min/max of a 3x3
    // neighbourhood is set by its two most extreme pixels, so a single bright
    // sample opens the box wide enough to let ghosting straight through. The
    // first two moments describe the neighbourhood as a distribution instead.
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let c = clamp(px + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
            let s = textureLoad(current_tex, c, 0).xyz;
            m1 = m1 + s;
            m2 = m2 + s * s;
        }
    }
    let inv_n = 1.0 / 9.0;
    let mean = m1 * inv_n;
    let sigma = sqrt(max(m2 * inv_n - mean * mean, vec3<f32>(0.0)));
    let gamma = max(tu.variance_gamma, 0.1);
    let lo = mean - sigma * gamma;
    let hi = mean + sigma * gamma;

    let history = textureSampleLevel(history_tex, linear_sampler, prev_uv, 0.0).xyz;
    let clipped = clip_aabb(lo, hi, history);

    let blend = clamp(tu.blend, 0.02, 1.0);
    let result = mix(clipped, current, blend);
    textureStore(out_tex, px, vec4<f32>(max(result, vec3<f32>(0.0)), 1.0));
}

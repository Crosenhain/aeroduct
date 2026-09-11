// Screen-space ambient occlusion on the opaque G-buffer.
//
// Deliberately a single pass with a modest sample count and a blue-noise
// rotation, and *no* blur. The blur is TAA's job: the jitter sequence already
// decorrelates the sample pattern across frames, so the temporal filter
// integrates the noise away for free and keeps contact shadows crisp. A spatial
// blur would soften exactly the thin creases the AO exists to draw — and on a
// duct with 2 mm walls, those creases are most of the geometry.

#include "common.wgsl"

struct SsaoUniform {
    // World-space sampling radius, mm. Scaled to the passage size, not the part.
    radius_mm: f32,
    // Depth bias, mm. Stops a flat surface shadowing itself.
    bias_mm: f32,
    intensity: f32,
    sample_count: u32,
};

@group(0) @binding(0) var<uniform> cam: Camera;

@group(1) @binding(0) var<uniform> su: SsaoUniform;
@group(1) @binding(1) var gbuf_depth: texture_depth_2d;
@group(1) @binding(2) var gbuf_normal: texture_2d<f32>;
@group(1) @binding(3) var blue_noise: texture_2d<f32>;
@group(1) @binding(4) var out_ao: texture_storage_2d<r32float, write>;

const MAX_SAMPLES: u32 = 24u;

// Cosine-weighted hemisphere direction from two uniform numbers, built around
// +Z and then rotated onto the surface normal.
fn hemisphere_dir(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u1, 0.0)));
    // Frisvad's branchless orthonormal basis, with the sign trick that keeps it
    // stable when the normal points near -Z.
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t1 = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let t2 = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return normalize(t1 * local.x + t2 * local.y + n * local.z);
}

@compute @workgroup_size(8, 8, 1)
fn ssao_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_ao);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let px = vec2<i32>(gid.xy);
    let depth = textureLoad(gbuf_depth, px, 0);
    if (depth <= 0.0) {
        // No geometry: fully unoccluded, so the background is untouched.
        textureStore(out_ao, px, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    // Ghost mode fills depth but leaves the normal cleared. Normalising a zero
    // vector produces NaN, which would propagate through the whole AO buffer,
    // so a degenerate normal means "no surface" and reports no occlusion.
    let n_raw = textureLoad(gbuf_normal, px, 0).xyz;
    if (dot(n_raw, n_raw) < 0.25) {
        textureStore(out_ao, px, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }
    let ndc = pixel_to_ndc(vec2<f32>(gid.xy) + vec2<f32>(0.5), vec2<f32>(size));
    let origin = unproject(cam.inv_view_proj, ndc, depth);
    let n = normalize(n_raw);
    let center_dist = linear_depth(cam.eye_near.w, depth);

    let nsize = textureDimensions(blue_noise);
    let noise = textureLoad(blue_noise, px % vec2<i32>(nsize), 0).x;
    // Advance the noise per frame so TAA has something to average.
    let rot = fract(noise + f32(u32(cam.inv_resolution_frame.z) % 64u) * 0.6180339887);

    let count = min(su.sample_count, MAX_SAMPLES);
    var occlusion = 0.0;
    var taken = 0.0;

    for (var i = 0u; i < MAX_SAMPLES; i = i + 1u) {
        if (i >= count) {
            break;
        }
        // Golden-ratio stratification within the hemisphere, offset by the
        // per-pixel blue noise so neighbouring pixels never share a pattern.
        let u1 = fract((f32(i) + 0.5) / f32(count) + rot);
        let u2 = fract(f32(i) * 0.7548776662 + rot);
        let dir = hemisphere_dir(n, u1, u2);
        // Vary the radius so the samples fill the hemisphere volume rather than
        // clustering on its shell.
        let scale = 0.25 + 0.75 * u2 * u2;
        let p = origin + dir * (su.radius_mm * scale);

        let clip = cam.view_proj_unjit * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }
        let suv = ndc_to_uv(clip.xy / clip.w);
        if (any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0))) {
            continue;
        }
        let spx = vec2<i32>(suv * vec2<f32>(size));
        let sd = textureLoad(gbuf_depth, clamp(spx, vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1)), 0);
        if (sd <= 0.0) {
            continue;
        }
        taken = taken + 1.0;

        let sample_dist = linear_depth(cam.eye_near.w, clip.z / clip.w);
        let scene_dist = linear_depth(cam.eye_near.w, sd);
        // Reverse-Z: a larger depth value is nearer. Working in linearised
        // distances keeps this readable and keeps the bias in millimetres.
        if (scene_dist < sample_dist - su.bias_mm) {
            // Range check, so a distant wall behind a near edge does not cast
            // occlusion across the gap.
            let delta = abs(center_dist - scene_dist);
            occlusion = occlusion + clamp(su.radius_mm / max(delta, 1.0e-4), 0.0, 1.0);
        }
    }

    var ao = 1.0;
    if (taken > 0.0) {
        ao = clamp(1.0 - su.intensity * occlusion / taken, 0.0, 1.0);
    }
    textureStore(out_ao, px, vec4<f32>(ao, 0.0, 0.0, 0.0));
}

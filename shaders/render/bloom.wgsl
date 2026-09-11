// Bloom: soft-knee threshold, a downsample chain, and a tent upsample back up.
//
// Bloom is applied to *additive* content — the volume's emissive core, specular
// highlights on the clearcoat — which on this scene is a small fraction of the
// frame. A hard threshold on such content flickers under TAA as pixels cross the
// cut, so the prefilter uses a quadratic soft knee: values well below the
// threshold are removed exactly, values around it ramp in smoothly.
//
// All three passes share one bind group layout so they can share a module.
// `src_b` is only meaningful for the upsample; the other passes bind the same
// texture twice and ignore it.

#include "common.wgsl"

struct BloomUniform {
    threshold: f32,
    // Width of the soft knee, in the same units as the threshold.
    knee: f32,
    // Radius of the upsample tent, in destination texels.
    radius: f32,
    intensity: f32,
};

@group(0) @binding(0) var<uniform> bl: BloomUniform;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var src_a: texture_2d<f32>;
@group(0) @binding(3) var src_b: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

fn soft_knee(c: vec3<f32>) -> vec3<f32> {
    let br = max(c.x, max(c.y, c.z));
    let knee = max(bl.knee, 1.0e-4);
    let soft = clamp(br - bl.threshold + knee, 0.0, 2.0 * knee);
    let contribution = max(soft * soft / (4.0 * knee), br - bl.threshold);
    return c * (contribution / max(br, 1.0e-4));
}

fn dst_uv(gid: vec2<u32>, size: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(gid) + vec2<f32>(0.5)) / vec2<f32>(size);
}

// Nine-tap box (a 3x3 of bilinear taps) downsample. Wider than a plain 2x2, and
// the extra width is what stops single bright pixels from strobing as they move
// between destination texels.
fn downsample13(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    let a = textureSampleLevel(src_a, linear_sampler, uv + texel * vec2<f32>(-1.0, -1.0), 0.0).xyz;
    let b = textureSampleLevel(src_a, linear_sampler, uv + texel * vec2<f32>(0.0, -1.0), 0.0).xyz;
    let c = textureSampleLevel(src_a, linear_sampler, uv + texel * vec2<f32>(1.0, -1.0), 0.0).xyz;
    let d = textureSampleLevel(src_a, linear_sampler, uv + texel * vec2<f32>(-1.0, 0.0), 0.0).xyz;
    let e = textureSampleLevel(src_a, linear_sampler, uv, 0.0).xyz;
    let f = textureSampleLevel(src_a, linear_sampler, uv + texel * vec2<f32>(1.0, 0.0), 0.0).xyz;
    let g = textureSampleLevel(src_a, linear_sampler, uv + texel * vec2<f32>(-1.0, 1.0), 0.0).xyz;
    let h = textureSampleLevel(src_a, linear_sampler, uv + texel * vec2<f32>(0.0, 1.0), 0.0).xyz;
    let i = textureSampleLevel(src_a, linear_sampler, uv + texel * vec2<f32>(1.0, 1.0), 0.0).xyz;
    return (a + c + g + i) * 0.0625 + (b + d + f + h) * 0.125 + e * 0.25;
}

fn tent(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    let r = texel * bl.radius;
    let a = textureSampleLevel(src_a, linear_sampler, uv + vec2<f32>(-r.x, -r.y), 0.0).xyz;
    let b = textureSampleLevel(src_a, linear_sampler, uv + vec2<f32>(0.0, -r.y), 0.0).xyz;
    let c = textureSampleLevel(src_a, linear_sampler, uv + vec2<f32>(r.x, -r.y), 0.0).xyz;
    let d = textureSampleLevel(src_a, linear_sampler, uv + vec2<f32>(-r.x, 0.0), 0.0).xyz;
    let e = textureSampleLevel(src_a, linear_sampler, uv, 0.0).xyz;
    let f = textureSampleLevel(src_a, linear_sampler, uv + vec2<f32>(r.x, 0.0), 0.0).xyz;
    let g = textureSampleLevel(src_a, linear_sampler, uv + vec2<f32>(-r.x, r.y), 0.0).xyz;
    let h = textureSampleLevel(src_a, linear_sampler, uv + vec2<f32>(0.0, r.y), 0.0).xyz;
    let i = textureSampleLevel(src_a, linear_sampler, uv + vec2<f32>(r.x, r.y), 0.0).xyz;
    return (a + c + g + i) * 0.0625 + (b + d + f + h) * 0.125 + e * 0.25;
}

@compute @workgroup_size(8, 8, 1)
fn prefilter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let uv = dst_uv(gid.xy, size);
    let texel = 1.0 / vec2<f32>(textureDimensions(src_a));
    let c = downsample13(uv, texel);
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(soft_knee(max(c, vec3<f32>(0.0))), 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn downsample_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let uv = dst_uv(gid.xy, size);
    let texel = 1.0 / vec2<f32>(textureDimensions(src_a));
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(downsample13(uv, texel), 1.0));
}

// Upsample and add. `src_a` is the smaller level being expanded, `src_b` is the
// same-size level from the downsample chain. Storage textures are write-only in
// WGSL, so this cannot accumulate in place — hence the second source.
@compute @workgroup_size(8, 8, 1)
fn upsample_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let uv = dst_uv(gid.xy, size);
    let texel = 1.0 / vec2<f32>(textureDimensions(src_a));
    let up = tent(uv, texel);
    let same = textureSampleLevel(src_b, linear_sampler, uv, 0.0).xyz;
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(same + up, 1.0));
}

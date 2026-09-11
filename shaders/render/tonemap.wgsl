// AgX tonemapping, applied exactly once, at the very end.
//
// # Why AgX and not ACES
//
// ACES RRT+ODT is a film-emulation transform. Its notorious behaviour on bright
// saturated colour is to rotate the hue on its way to white — bright blues go
// purple, bright oranges go yellow. On a scientific visualisation that is not a
// stylistic quirk, it is a *data* error: the whole contract of a colour map is
// that the colour you see indexes the value, and a tone curve that moves hue as
// a function of intensity breaks that contract exactly where the interesting
// values are. Inferno's top end is a saturated orange, and it is precisely the
// core of the jet.
//
// AgX instead desaturates toward white along a path that keeps hue stable, which
// is what a real film stock does and what the eye expects. Highlights roll off
// rather than clipping, and the mapping stays monotone in luminance, so a
// brighter value never renders darker.
//
// # Applied exactly once
//
// Tonemapping is not idempotent and does not commute with anything. Everything
// upstream of this file — the volume, the composite, TAA, bloom — works in
// linear HDR. Any "just clamp it for now" applied earlier would silently change
// what TAA is averaging and what bloom is thresholding.

#include "common.wgsl"

struct TonemapUniform {
    // Linear exposure multiplier applied before the transform.
    exposure: f32,
    bloom_intensity: f32,
    // Amount of the AgX "look" contrast, 0 = neutral.
    look_strength: f32,
    // Ordered-dither amplitude in output LSBs. Kills 8-bit banding in the
    // background gradient, which is otherwise the first thing anyone notices.
    dither: f32,
};

@group(0) @binding(0) var<uniform> tm: TonemapUniform;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var hdr_tex: texture_2d<f32>;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var blue_noise: texture_2d<f32>;

// AgX base transform. Rotates into a slightly desaturated working space where
// the sigmoid can be applied per channel without pulling hue around.
const AGX_MAT = mat3x3<f32>(
    vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
    vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336000000000),
    vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
);

const AGX_MAT_INV = mat3x3<f32>(
    vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
    vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
    vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
);

// The log2 window AgX maps onto [0, 1]: about 12.5 stops below and 4 above mid
// grey. Wide enough that the volume's emissive core has somewhere to roll off
// into instead of clipping to white.
const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

// Sixth-order polynomial fit of the AgX sigmoid on [0, 1].
fn agx_sigmoid(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - 0.00232;
}

// Optional "look": slope / power / saturation, in the AgX working space. A
// touch of contrast and a touch of desaturation reads as photographic; the
// neutral look is available by setting `look_strength` to zero.
fn agx_look(v: vec3<f32>, strength: f32) -> vec3<f32> {
    let luma = luminance(v);
    let slope = mix(1.0, 1.05, strength);
    let power = mix(1.0, 1.08, strength);
    let sat = mix(1.0, 1.12, strength);
    let out = pow(max(v * slope, vec3<f32>(0.0)), vec3<f32>(power));
    return max(vec3<f32>(luma) + sat * (out - vec3<f32>(luma)), vec3<f32>(0.0));
}

fn agx(color: vec3<f32>) -> vec3<f32> {
    var v = AGX_MAT * max(color, vec3<f32>(0.0));
    // Log2 encode into the display window.
    v = clamp(log2(max(v, vec3<f32>(1.0e-10))), vec3<f32>(AGX_MIN_EV), vec3<f32>(AGX_MAX_EV));
    v = (v - vec3<f32>(AGX_MIN_EV)) / (AGX_MAX_EV - AGX_MIN_EV);
    v = agx_sigmoid(v);
    v = agx_look(v, clamp(tm.look_strength, 0.0, 1.0));
    v = AGX_MAT_INV * v;
    // The sigmoid output is display-referred with an implied 2.2 gamma, so undo
    // it to get back to linear light. The caller then encodes for the target.
    return max(pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.2)), vec3<f32>(0.0));
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    out.clip = fullscreen_position(vi);
    out.uv = fullscreen_uv(vi);
    return out;
}

@fragment
fn fs_tonemap(input: VsOut) -> @location(0) vec4<f32> {
    var hdr = textureSampleLevel(hdr_tex, linear_sampler, input.uv, 0.0).xyz;
    let bloom = textureSampleLevel(bloom_tex, linear_sampler, input.uv, 0.0).xyz;
    hdr = hdr + bloom * tm.bloom_intensity;
    var mapped = agx(hdr * tm.exposure);

#if SRGB_TARGET
    // The target view is sRGB, so the hardware does the encode on write.
#else
    mapped = srgb_encode(mapped);
#endif

    // Triangular-PDF-ish dither from the blue-noise tile. Applied after the
    // encode, in output code values, because that is where the quantisation is.
    let nsize = textureDimensions(blue_noise);
    let px = vec2<i32>(input.clip.xy);
    let n = textureLoad(blue_noise, px % vec2<i32>(nsize), 0).x;
    mapped = mapped + (n - 0.5) * tm.dither * (1.0 / 255.0);

    return vec4<f32>(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

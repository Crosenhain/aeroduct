// Direct isosurface raymarching of the normalised Q-criterion.
//
// See `crates/ad-render/src/isosurface.rs` for why this is a raymarcher and not
// marching cubes. The short version: marching cubes costs O(voxels) every time
// the isolevel moves and pops as it is dragged; a raymarch costs O(pixels), gets
// correct depth for free, and moves continuously.
//
// Four things this file must not get wrong:
//
//   1. **The brick skip must be exact, not merely plausible.** A brick is
//      skippable for this pass only when its stored interval provably cannot
//      contain the isolevel — which holds only if the brick min/max was reduced
//      over the *same channel* we march. `bu.channel` says which, so the shader
//      derives that itself rather than trusting a flag from the CPU.
//   2. **A crossing must be bracketed before it is refined.** Bisection is valid
//      only on a sign change, so the marcher keeps the previous sample and
//      refines only when `f_prev < 0 <= f_next`.
//   3. **Solid voxels are not "below the isolevel", they are absent.** Treating
//      a wall as low-Q fluid draws a shell over the whole duct interior, which
//      looks like a real structure and is not.
//   4. **Reverse-Z.** The fragment writes its own depth so the surface is
//      occluded by the duct, and `depth_from_linear` is the only conversion.
//
// Two modules come out of this file: the draw, and the Q histogram compute pass
// selected by `ISO_HISTOGRAM`. They cannot share one module, because WGSL
// requires every `@group`/`@binding` pair to be unique within a module and the
// two want entirely different resources at group 0.

#include "common.wgsl"

// Channel of the normalised Q-criterion in the packed scalar texture. Mirrors
// `DerivedField::QCriterion::channel()`.
const Q_CHANNEL: u32 = 1u;
const SPEED_CHANNEL: u32 = 0u;

#if ISO_HISTOGRAM

// -- Q histogram -------------------------------------------------------------

@group(0) @binding(0) var<uniform> fu: FieldsInfo;
@group(0) @binding(1) var field_sampler: sampler;
@group(0) @binding(2) var field_scalars: texture_3d<f32>;
@group(0) @binding(3) var field_velocity: texture_3d<f32>;

struct HistUniform {
    lo: f32,
    hi: f32,
    bin_count: u32,
    pad0: u32,
};

@group(1) @binding(0) var<uniform> hu: HistUniform;
@group(1) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;

// One thread per derived voxel. Solid voxels are excluded, so the histogram
// describes the *flow* rather than the bounding box — which matters, because a
// duct fills perhaps a fifth of its box and the wall would otherwise pile into
// bin 0 and drag every percentile down with it.
@compute @workgroup_size(#HIST_WG, #HIST_WG, #HIST_WG)
fn histogram_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= fu.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    if (textureLoad(field_velocity, c, 0).w <= 0.0) {
        return;
    }
    let q = pick_channel(textureLoad(field_scalars, c, 0), Q_CHANNEL);
    let span = max(hu.hi - hu.lo, 1.0e-9);
    let t = (q - hu.lo) / span;
    // Clamp into the end bins rather than dropping out-of-range samples: the
    // tails are the interesting part of a Q distribution, and discarding them
    // silently would put every suggested isolevel too low.
    let b = u32(clamp(t, 0.0, 0.999999) * f32(hu.bin_count));
    atomicAdd(&bins[min(b, max(hu.bin_count, 1u) - 1u)], 1u);
}

#else

// -- the draw ----------------------------------------------------------------

struct IsoUniform {
    // Q-tilde value the surface is drawn at.
    iso_level: f32,
    // March step as a fraction of a derived voxel.
    step_scale: f32,
    max_steps: u32,
    // Bisection iterations after the sign change. Four halvings take a
    // one-voxel bracket to a sixteenth of a voxel, comfortably under the
    // trilinear filter's own resolution.
    refine_steps: u32,

    // Speed range the colour map spans, m/s.
    color_lo: f32,
    color_inv_span: f32,
    opacity: f32,
    shading_mode: u32,

    // xyz = unit direction toward the key light, w = shading strength.
    light_dir: vec4<f32>,
    // x = ambient, y = specular, z = rim, w = jitter amplitude in steps.
    surface: vec4<f32>,

    skip_enabled: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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

@group(3) @binding(0) var<uniform> iu: IsoUniform;
@group(3) @binding(1) var color_lut: texture_2d<f32>;
@group(3) @binding(2) var lut_sampler: sampler;

fn field_uvw(p: vec3<f32>) -> vec3<f32> {
    return (p - fu.volume_min_mm) / fu.volume_size_mm;
}

// The implicit function whose zero set is the surface: `Q~ - iso`, forced
// strongly negative outside the fluid so a wall can never register as a
// crossing. Large negative rather than zero on purpose — zero would put the
// wall face exactly on the isolevel whenever `iso == 0`.
fn implicit(p: vec3<f32>) -> f32 {
    let uvw = field_uvw(p);
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
        return -1.0e6;
    }
    if (textureSampleLevel(field_velocity, field_sampler, uvw, 0.0).w <= 0.0) {
        return -1.0e6;
    }
    let s = textureSampleLevel(field_scalars, field_sampler, uvw, 0.0);
    return pick_channel(s, Q_CHANNEL) - iu.iso_level;
}

fn brick_coord(p: vec3<f32>) -> vec3<i32> {
    let bw = f32(bu.brick_size) * fu.voxel_mm;
    return vec3<i32>(floor((p - fu.volume_min_mm) / bw));
}

// Exit parameter of one brick's own box: the DDA of `volume.wgsl` specialised to
// a single cell. Advancing to the exit of a brick that provably holds no
// crossing cannot miss one.
fn brick_exit(origin: vec3<f32>, dir: vec3<f32>, brick: vec3<i32>) -> f32 {
    let bw = f32(bu.brick_size) * fu.voxel_mm;
    let lo = fu.volume_min_mm + vec3<f32>(brick) * bw;
    let hi = lo + vec3<f32>(bw);
    return ray_box(origin, dir, lo, hi).y;
}

// Whether a brick can be skipped outright. Two independent reasons, and the
// first holds whatever field is on screen:
//
//   * `min > max` is the brick grid's flag for "not one fluid voxel in here",
//     which is a property of the geometry rather than of the channel. The duct
//     wall and the solid interior are skipped on that alone.
//   * When the min/max happens to have been reduced over the Q channel — i.e.
//     Q-criterion is the displayed field — a brick whose whole interval sits
//     *below* the isolevel provably contains no crossing. That is the exact
//     test, and it is what makes this pass fast in the mode where it is the
//     thing being looked at.
//
// The mirror-image test — the whole interval sitting *above* the isolevel, so
// the brick is entirely inside the structure — looks equally sound and is not.
// `minmax_main` reduces over fluid voxels only, while `implicit` reports solid
// as far below the isolevel; so a brick that is all high-Q fluid *plus some
// wall* does contain a crossing, right where the vortex tube meets the duct.
// Skipping it opens the end of the tube. The gain would have been negligible
// anyway — a ray already inside the structure has already found its surface.
//
// Anything else marches. Being conservative here is the difference between an
// optimisation and a hole in the surface.
fn brick_skippable(b: vec3<i32>) -> bool {
    if (any(b < vec3<i32>(0)) || any(b >= vec3<i32>(bu.brick_dims))) {
        return true;
    }
    let mm = textureLoad(brick_minmax, b, 0).xy;
    if (mm.x > mm.y) {
        return true;
    }
    if (bu.channel == Q_CHANNEL) {
        // `minmax_main` already widens each interval by a one-voxel apron, so
        // the trilinear filter footprint is covered and no extra slack is
        // needed here.
        return iu.iso_level > mm.y;
    }
    return false;
}

// Central-difference gradient of the implicit function, in world units. The
// scalar rises into the vortex core, so the outward normal is `-grad`.
fn q_gradient(p: vec3<f32>) -> vec3<f32> {
    let e = fu.voxel_mm;
    return vec3<f32>(
        implicit(p + vec3<f32>(e, 0.0, 0.0)) - implicit(p - vec3<f32>(e, 0.0, 0.0)),
        implicit(p + vec3<f32>(0.0, e, 0.0)) - implicit(p - vec3<f32>(0.0, e, 0.0)),
        implicit(p + vec3<f32>(0.0, 0.0, e)) - implicit(p - vec3<f32>(0.0, 0.0, e)),
    );
}

fn lut_color(t: f32) -> vec3<f32> {
    return textureSampleLevel(color_lut, lut_sampler, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0).rgb;
}

struct FragOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

// Reverse-Z far plane and a transparent colour: what a discarded fragment
// returns, since WGSL's `discard` demotes the invocation rather than ending it
// and every path still has to produce a value.
fn miss() -> FragOut {
    var o: FragOut;
    o.color = vec4<f32>(0.0);
    o.depth = 0.0;
    return o;
}

@vertex
fn vs_iso(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_position(vi);
}

@fragment
fn fs_iso(@builtin(position) frag: vec4<f32>) -> FragOut {
    let res = cam.jitter_resolution.zw;
    let ndc = pixel_to_ndc(frag.xy, res);
    let ray = camera_ray(cam, ndc);

    let lo = fu.volume_min_mm;
    let hi = fu.volume_min_mm + fu.volume_size_mm;
    let span = ray_box(ray.origin, ray.dir, lo, hi);
    // Clamp the entry to the near plane, or a camera inside the duct starts the
    // march behind its own eye.
    var t = max(span.x, cam.eye_near.w);
    let t_end = span.y;
    if (t_end <= t || span.y <= span.x) {
        discard;
        return miss();
    }

    let h = max(iu.step_scale * fu.voxel_mm, 1.0e-4);

    // Per-pixel, per-frame start jitter. The surface is hard, so without it the
    // silhouette steps along the march grid; with it that error becomes noise
    // the accumulator integrates into a clean antialiased edge.
    if (iu.surface.w > 0.0) {
        let seed = hash_u32(
            u32(frag.x) * 1973u + u32(frag.y) * 9277u
                + u32(cam.inv_resolution_frame.z) * 26699u
        );
        t = t + f32(seed) * (1.0 / 4294967296.0) * h * iu.surface.w;
    }

    // The march runs on a fixed lattice `t_start + i * h`, and the skip moves
    // the *index* rather than the parameter.
    //
    // That is the whole trick, and it took a test to find. Jumping straight to a
    // brick's exit and restarting the bracket there leaves the accelerated ray
    // sampling a different set of points from the unaccelerated one, so the two
    // bracket different intervals and — on a hard surface with several crossings
    // inside one step — pick different roots. The image then differs along every
    // silhouette by an entire pixel value, which is indistinguishable from the
    // skip actually eating part of the surface. Landing on the last lattice
    // point at or before the exit instead makes the accelerated march evaluate a
    // strict *subset* of the same brackets, so the answer is bit-identical and
    // the accelerator is provably invisible.
    let t_start = t;
    let inv_h = 1.0 / h;

    var f_prev = implicit(ray.origin + ray.dir * t_start);
    var t_hit = -1.0;
    // Already inside the structure at the entry point: the eye sits within a
    // vortex core, so the visible surface is the one we are standing on. Report
    // the entry rather than hunting a crossing that is behind the camera.
    if (f_prev >= 0.0) {
        t_hit = t_start;
    }

    var i = 0u;
    var steps = 0u;
    loop {
        if (t_hit >= 0.0 || steps >= iu.max_steps) {
            break;
        }
        steps = steps + 1u;
        let t_cur = t_start + f32(i) * h;
        if (t_cur >= t_end) {
            break;
        }

        if (iu.skip_enabled != 0u) {
            let b = brick_coord(ray.origin + ray.dir * t_cur);
            if (brick_skippable(b)) {
                let exit = brick_exit(ray.origin, ray.dir, b);
                // The last lattice point strictly before the exit. The interval
                // from here to there lies inside the brick and is proven
                // crossing-free; the bracket that straddles the brick boundary
                // is still evaluated in full.
                let j = i32(ceil((exit - t_start) * inv_h)) - 1;
                if (j > i32(i)) {
                    i = u32(j);
                    f_prev = implicit(ray.origin + ray.dir * (t_start + f32(i) * h));
                    continue;
                }
                // The exit is less than two steps away, so skipping would buy
                // nothing and cannot be done without moving off the lattice.
                // March instead.
            }
        }

        let t_next = min(t_start + f32(i + 1u) * h, t_end);
        let f_next = implicit(ray.origin + ray.dir * t_next);
        if (f_prev < 0.0 && f_next >= 0.0) {
            // --- bracketed: refine by bisection ---
            var a = t_cur;
            var b2 = t_next;
            for (var k = 0u; k < iu.refine_steps; k = k + 1u) {
                let m = 0.5 * (a + b2);
                if (implicit(ray.origin + ray.dir * m) < 0.0) {
                    a = m;
                } else {
                    b2 = m;
                }
            }
            t_hit = 0.5 * (a + b2);
            break;
        }
        f_prev = f_next;
        i = i + 1u;
    }

    if (t_hit < 0.0) {
        discard;
        return miss();
    }

    let p = ray.origin + ray.dir * t_hit;
    let scalars = textureSampleLevel(field_scalars, field_sampler, field_uvw(p), 0.0);

    let g = q_gradient(p);
    let gl = length(g);
    var n = -ray.dir;
    if (gl > 1.0e-12) {
        n = -g / gl;
    }

    var base = vec3<f32>(0.78, 0.80, 0.84);
    if (iu.shading_mode == 0u) {
        // Colour by speed. The surface says *where* the vortex is; the colour is
        // what puts units on it. An isosurface with no scalar mapped onto it is
        // a shape and nothing more.
        let speed = pick_channel(scalars, SPEED_CHANNEL);
        base = lut_color((speed - iu.color_lo) * iu.color_inv_span);
    } else if (iu.shading_mode == 1u) {
        base = n * 0.5 + vec3<f32>(0.5);
    }

    let l = normalize(iu.light_dir.xyz);
    // Half-Lambert: a hard terminator on a thin twisted structure reads as a
    // hole in the surface rather than as shape.
    let wrapped = max(dot(n, l), 0.0) * 0.5 + 0.5;
    let hv = normalize(l - ray.dir);
    let spec = pow(max(dot(n, hv), 0.0), 48.0) * iu.surface.y;
    // Rim light along the silhouette, which is what separates two vortex tubes
    // lying one behind the other.
    let rim = pow(1.0 - max(dot(n, -ray.dir), 0.0), 3.0) * iu.surface.z;
    var lit = base * (iu.surface.x + wrapped + rim) + vec3<f32>(spec);
    lit = mix(base, lit, clamp(iu.light_dir.w, 0.0, 1.0));

    // Distance along the view *axis*, not along the ray: reverse-Z depth is
    // defined against the former.
    let cosine = max(dot(ray.dir, cam.forward_tan.xyz), 1.0e-4);

    var o: FragOut;
    let a = clamp(iu.opacity, 0.0, 1.0);
    // Premultiplied, for a `One / OneMinusSrcAlpha` blend.
    o.color = vec4<f32>(lit * a, a);
    o.depth = depth_from_linear(cam.eye_near.w, t_hit * cosine);
    return o;
}

#endif

// An oriented cutting plane with animated line-integral convolution on it.
//
// See `crates/ad-render/src/slice.rs` for the reasoning. The plane is two
// triangles generated from the vertex index — no vertex buffer — so the
// rasteriser produces correct reverse-Z depth for free and the duct occludes the
// plane with no work at all.
//
// Three things this file must not get wrong:
//
//   1. **The convolution must be normalised.** The travelling kernel's weights
//      do not sum to a phase-independent constant, so an unnormalised sum makes
//      the whole plane pulse once per animation cycle. It looks exactly like a
//      flickering exposure bug and people blame the tonemapper for an hour.
//   2. **The LIC contrast must be scaled by the in-plane fraction of the
//      velocity.** LIC can only show the in-plane component; drawing full
//      contrast where the flow is piercing the plane is a confident lie.
//   3. **The colour stays quantitative.** The texture modulates brightness
//      multiplicatively and never shifts hue, so a colour on screen still means
//      the number the legend says it means.

#include "common.wgsl"
#include "lic.wgsl"

struct SliceUniform {
    origin_mm: vec3<f32>,
    half_u_mm: f32,

    axis_u: vec3<f32>,
    half_v_mm: f32,

    axis_v: vec3<f32>,
    opacity: f32,

    normal: vec3<f32>,
    contrast: f32,

    // Data range the colour map spans, in the displayed field's own units.
    color_lo: f32,
    color_inv_span: f32,
    // Half-length of the convolution, in integration steps each way.
    steps: u32,
    step_mm: f32,

    // Wrapped into [0, 1) on the CPU: after an hour at 60 fps a raw `t * rate`
    // has only a few bits of fractional part left and the animation ratchets.
    phase: f32,
    noise_scale_mm: f32,
    honesty_power: f32,
    honesty: u32,

    channel: u32,
    brightness: f32,
    // Width of the bright rim drawn round the plane's border, mm. Zero for none.
    edge_mm: f32,
    pad0: f32,

    edge_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;

@group(1) @binding(0) var<uniform> fu: FieldsInfo;
@group(1) @binding(1) var field_sampler: sampler;
@group(1) @binding(2) var field_scalars: texture_3d<f32>;
@group(1) @binding(3) var field_velocity: texture_3d<f32>;

@group(2) @binding(0) var<uniform> su: SliceUniform;
@group(2) @binding(1) var color_lut: texture_2d<f32>;
@group(2) @binding(2) var lut_sampler: sampler;

struct SliceOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_mm: vec3<f32>,
    // Millimetres from the plane origin along (axis_u, axis_v). The LIC noise
    // lattice lives in this space, so the texture stays put as the plane is
    // scrubbed rather than crawling across it.
    @location(1) plane_mm: vec2<f32>,
};

// Two triangles, (0,1,2, 0,2,3) round a unit quad.
fn quad_corner(index: u32) -> vec2<f32> {
    var q = index;
    if (index == 3u) {
        q = 0u;
    } else if (index == 4u) {
        q = 2u;
    } else if (index == 5u) {
        q = 3u;
    }
    let x = select(-1.0, 1.0, q == 1u || q == 2u);
    let y = select(-1.0, 1.0, q == 2u || q == 3u);
    return vec2<f32>(x, y);
}

@vertex
fn vs_slice(@builtin(vertex_index) vi: u32) -> SliceOut {
    let c = quad_corner(vi % 6u);
    let plane = vec2<f32>(c.x * su.half_u_mm, c.y * su.half_v_mm);
    let world = su.origin_mm + su.axis_u * plane.x + su.axis_v * plane.y;

    var o: SliceOut;
    o.clip = cam.view_proj * vec4<f32>(world, 1.0);
    o.world_mm = world;
    o.plane_mm = plane;
    return o;
}

fn field_uvw(p: vec3<f32>) -> vec3<f32> {
    return (p - fu.volume_min_mm) / fu.volume_size_mm;
}

fn inside_volume(uvw: vec3<f32>) -> bool {
    return all(uvw >= vec3<f32>(0.0)) && all(uvw <= vec3<f32>(1.0));
}

// Velocity in m/s, with the fluid fraction in `w`. Outside the volume it reports
// `w < 0`, which is distinct from `w == 0` ("inside a wall") — one means there
// is nothing to draw, the other means there is solid material there.
fn sample_velocity(p: vec3<f32>) -> vec4<f32> {
    let uvw = field_uvw(p);
    if (!inside_volume(uvw)) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    return textureSampleLevel(field_velocity, field_sampler, uvw, 0.0);
}

// Component of the velocity lying in the cutting plane, as a unit direction.
// Returns the zero vector where there is nothing to follow, which the caller
// uses to stop walking rather than integrating noise.
fn plane_direction(p: vec3<f32>) -> vec3<f32> {
    let v = sample_velocity(p);
    if (v.w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let inplane = v.xyz - su.normal * dot(v.xyz, su.normal);
    let l = length(inplane);
    if (l < 1.0e-9) {
        return vec3<f32>(0.0);
    }
    return inplane / l;
}

// Snap back onto the plane. The direction is projected already, so this only
// removes accumulated round-off — but over 28 steps that round-off is enough to
// walk a whole voxel out of the plane and sample a different sheet of flow.
fn onto_plane(p: vec3<f32>) -> vec3<f32> {
    return p - su.normal * dot(p - su.origin_mm, su.normal);
}

fn plane_coords(p: vec3<f32>) -> vec2<f32> {
    let d = p - su.origin_mm;
    return vec2<f32>(dot(d, su.axis_u), dot(d, su.axis_v));
}

// One RK2 midpoint step of arc length `h` along the unit in-plane direction.
// Midpoint rather than Euler because a duct bend curves the streamline
// appreciably over a 28-step convolution, and Euler cuts every corner outward —
// which shows up as LIC fibres that drift off the actual flow near the inner
// wall, exactly where a designer is looking hardest.
fn walk(p: vec3<f32>, h: f32) -> vec3<f32> {
    let d1 = plane_direction(p);
    if (all(d1 == vec3<f32>(0.0))) {
        return p;
    }
    let mid = onto_plane(p + d1 * (0.5 * h));
    var d2 = plane_direction(mid);
    if (all(d2 == vec3<f32>(0.0))) {
        d2 = d1;
    }
    return onto_plane(p + d2 * h);
}

@fragment
fn fs_slice(in: SliceOut) -> @location(0) vec4<f32> {
    // The rim first, so the plane's outline stays visible even where it cuts
    // through solid material or still air and everything else is discarded.
    // Without it a slice through an empty part of the domain simply vanishes and
    // the control that moves it appears to do nothing.
    var edge = 0.0;
    if (su.edge_mm > 0.0) {
        let du = su.half_u_mm - abs(in.plane_mm.x);
        let dv = su.half_v_mm - abs(in.plane_mm.y);
        edge = 1.0 - smoothstep(0.0, su.edge_mm, min(du, dv));
    }

    let p = in.world_mm;
    let uvw = field_uvw(p);
    let v = sample_velocity(p);
    if (!inside_volume(uvw) || v.w <= 0.0) {
        if (edge <= 0.001) {
            discard;
        }
        let a = clamp(su.opacity * edge, 0.0, 1.0);
        return vec4<f32>(su.edge_color.rgb * a, a);
    }

    // --- the quantitative channel ---
    let scalars = textureSampleLevel(field_scalars, field_sampler, uvw, 0.0);
    let value = pick_channel(scalars, su.channel);
    let t = (value - su.color_lo) * su.color_inv_span;
    let base = textureSampleLevel(
        color_lut, lut_sampler, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0).rgb;

    // --- the directional channel ---
    let speed = length(v.xyz);
    let inplane = length(v.xyz - su.normal * dot(v.xyz, su.normal));

    var acc = lic_ramp_kernel(0.0, su.phase) * lic_noise(in.plane_mm, su.noise_scale_mm);
    var wsum = lic_ramp_kernel(0.0, su.phase);
    // Counted, not assumed. A walk that stalls against a wall contributes one
    // tap, and the gain has to know that or it turns a single noise sample into
    // a stark black-and-white lattice.
    var taps = 1.0;
    var fwd = p;
    var bwd = p;
    var fwd_live = true;
    var bwd_live = true;
    let inv_l = 1.0 / f32(max(su.steps, 1u));

    for (var k = 1u; k <= #LIC_MAX_STEPSu; k = k + 1u) {
        if (k > su.steps) {
            break;
        }
        let s = f32(k) * inv_l;

        if (fwd_live) {
            let next = walk(fwd, su.step_mm);
            // A stalled walk means the streamline ran into a wall or a
            // stagnation point. Continuing would pile every remaining tap onto
            // one noise cell and stamp a bright blob there.
            if (distance(next, fwd) < 0.25 * su.step_mm) {
                fwd_live = false;
            } else {
                fwd = next;
                let w = lic_ramp_kernel(s, su.phase);
                acc = acc + w * lic_noise(plane_coords(fwd), su.noise_scale_mm);
                wsum = wsum + w;
                taps = taps + 1.0;
            }
        }
        if (bwd_live) {
            let next = walk(bwd, -su.step_mm);
            if (distance(next, bwd) < 0.25 * su.step_mm) {
                bwd_live = false;
            } else {
                bwd = next;
                let w = lic_ramp_kernel(-s, su.phase);
                acc = acc + w * lic_noise(plane_coords(bwd), su.noise_scale_mm);
                wsum = wsum + w;
                taps = taps + 1.0;
            }
        }
        if (!fwd_live && !bwd_live) {
            break;
        }
    }

    // Normalised: the weighted *mean*, not the sum. This is what stops the whole
    // plane breathing once per animation cycle.
    var lic = 0.5;
    if (wsum > 1.0e-20) {
        let gain = lic_contrast_gain(
            lic_effective_taps(taps, su.step_mm, su.noise_scale_mm));
        lic = lic_apply_gain(acc / wsum, gain);
    }

    var contrast = su.contrast;
    if (su.honesty != 0u) {
        contrast = contrast * lic_honesty(inplane, speed, su.honesty_power);
    }

    var rgb = lic_compose(base, lic, contrast) * su.brightness;
    rgb = mix(rgb, su.edge_color.rgb, clamp(edge, 0.0, 1.0));

    // Fade out across the last voxel of fluid rather than cutting at it. `w` is
    // a fluid *fraction*, and the trilinear interpolation of a 0/1 mask is
    // exactly zero throughout the solid — so a hard `w <= 0` test puts the
    // boundary on voxel faces and the wall comes out as an axis-aligned
    // staircase running the length of the duct. Using the fraction as coverage
    // is both prettier and more truthful: it is what the fraction means.
    let coverage = smoothstep(0.0, 0.5, v.w);
    let a = clamp(su.opacity * max(coverage, edge), 0.0, 1.0);
    // Premultiplied, for a `One / OneMinusSrcAlpha` blend.
    return vec4<f32>(rgb * a, a);
}

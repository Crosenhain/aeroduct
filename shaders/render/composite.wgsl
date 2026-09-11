// Background, opaque shading, and volume compositing into the HDR target.
//
// This is where the picture is actually assembled:
//
//   1. an analytic background (vertical gradient + ground plane + contact
//      shadow), so an empty scene still looks like a studio and not like a void;
//   2. deferred shading of the G-buffer with a three-light rig and a clearcoat
//      lobe, so the duct reads as *plastic*;
//   3. the half-resolution volume, upsampled and composited with premultiplied
//      `ONE, ONE_MINUS_SRC_ALPHA` over the top.

#include "common.wgsl"

struct CompositeUniform {
    // Background gradient, linear light.
    sky_top: vec4<f32>,
    sky_bottom: vec4<f32>,
    ground_color: vec4<f32>,

    // xyz = direction toward the key light, w = intensity.
    key_dir: vec4<f32>,
    // xyz = fill light direction, w = intensity.
    fill_dir: vec4<f32>,
    // xyz = key colour, w = ambient strength.
    key_color: vec4<f32>,

    // x = ground plane height mm, y = contact-shadow radius mm,
    // z = contact-shadow strength, w = ground fade distance mm.
    ground: vec4<f32>,
    // xyz = scene centre mm, w = clearcoat strength.
    scene_center: vec4<f32>,
    // x = exposure applied to the opaque pass, y = rim strength,
    // z = 1 when the volume should be composited, w = ambient occlusion power.
    misc: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;

@group(1) @binding(0) var<uniform> cu: CompositeUniform;
@group(1) @binding(1) var linear_sampler: sampler;
@group(1) @binding(2) var gbuf_albedo: texture_2d<f32>;
@group(1) @binding(3) var gbuf_normal: texture_2d<f32>;
@group(1) @binding(4) var gbuf_depth: texture_depth_2d;
@group(1) @binding(5) var ssao_tex: texture_2d<f32>;
@group(1) @binding(6) var volume_color: texture_2d<f32>;
@group(1) @binding(7) var out_hdr: texture_storage_2d<rgba16float, write>;

// Vertical gradient plus a ground plane. A flat dark grey background makes
// every render look like a screenshot of a bug report; a subtle gradient gives
// the eye a horizon to place the part against, at the cost of nothing.
fn background(origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    var col = mix(cu.sky_bottom.xyz, cu.sky_top.xyz, smoothstep(0.0, 1.0, t));

    // Ground plane, only when looking down at it.
    if (dir.y < -1.0e-4) {
        let t_ground = (cu.ground.x - origin.y) / dir.y;
        if (t_ground > 0.0) {
            let p = origin + dir * t_ground;
            let r = length(p.xz - cu.scene_center.xz) / max(cu.ground.y, 1.0e-3);
            // Soft contact shadow: a Gaussian pool under the part. Cheaper and
            // steadier than a shadow map, and for a single object on a plane it
            // is indistinguishable at the sizes anyone looks at.
            let shadow = exp(-r * r * 2.0) * clamp(cu.ground.z, 0.0, 1.0);
            let ground = cu.ground_color.xyz * (1.0 - shadow);
            // Fade the ground into the sky with distance so there is no hard
            // horizon line to draw the eye away from the part.
            let fade = clamp(t_ground / max(cu.ground.w, 1.0), 0.0, 1.0);
            col = mix(ground, col, fade * fade);
        }
    }
    return col;
}

// Schlick Fresnel.
fn fresnel(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// GGX specular, single term. Enough for a plastic part under a studio rig.
fn ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let h = normalize(v + l);
    let a = max(roughness * roughness, 1.0e-3);
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 1.0e-4);
    let ndl = max(dot(n, l), 0.0);
    let d = a2 / max(PI * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0), 1.0e-6);
    let k = a * 0.5;
    let gv = ndv / (ndv * (1.0 - k) + k);
    let gl = ndl / (ndl * (1.0 - k) + k);
    return d * gv * gl / max(4.0 * ndv * ndl, 1.0e-4);
}

fn shade_opaque(albedo: vec3<f32>, roughness: f32, n: vec3<f32>, p: vec3<f32>, ao: f32) -> vec3<f32> {
    let v = normalize(cam.eye_near.xyz - p);
    let key = normalize(cu.key_dir.xyz);
    let fill = normalize(cu.fill_dir.xyz);

    let ndl_key = max(dot(n, key), 0.0);
    let ndl_fill = max(dot(n, fill), 0.0);

    // Dielectric base reflectance.
    let f0 = vec3<f32>(0.04);
    var col = albedo * (ndl_key * cu.key_dir.w * cu.key_color.xyz + ndl_fill * cu.fill_dir.w);

    let spec_key = ggx(n, v, key, roughness) * ndl_key * cu.key_dir.w;
    col = col + fresnel(f0, max(dot(n, v), 0.0)) * spec_key;

    // Clearcoat: a second, much smoother lobe over the top. This is the whole
    // difference between "3D-printed plastic part" and "grey blob". The coat
    // has its own fixed IOR 1.5 reflectance and does not tint with the albedo.
    let coat = ggx(n, v, key, 0.08) * ndl_key * cu.key_dir.w * clamp(cu.scene_center.w, 0.0, 1.0);
    col = col + vec3<f32>(coat) * fresnel(vec3<f32>(0.04), max(dot(n, v), 0.0)).x * 4.0;

    // Ambient from the same gradient the background uses, so the part sits in
    // its environment instead of on top of it.
    let ambient = mix(cu.sky_bottom.xyz, cu.sky_top.xyz, n.y * 0.5 + 0.5) * cu.key_color.w;
    let occ = pow(clamp(ao, 0.0, 1.0), max(cu.misc.w, 0.01));
    col = col + albedo * ambient * occ;

    // Rim light along the silhouette: a duct is a smooth dark object and needs
    // an edge to read against the background.
    let rim = pow(clamp(1.0 - max(dot(n, v), 0.0), 0.0, 1.0), 4.0) * cu.misc.y;
    col = col + cu.sky_top.xyz * rim;

    return col * cu.misc.x;
}

@compute @workgroup_size(8, 8, 1)
fn composite_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_hdr);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let px = vec2<i32>(gid.xy);
    let fsize = vec2<f32>(size);
    let ndc = pixel_to_ndc(vec2<f32>(gid.xy) + vec2<f32>(0.5), fsize);
    let ray = camera_ray(cam, ndc);

    var color = background(ray.origin, ray.dir);

    let depth = textureLoad(gbuf_depth, px, 0);
    let nrm = textureLoad(gbuf_normal, px, 0);
    // Depth alone is not enough to say "there is a surface here to shade".
    // Ghost mode writes depth and motion but masks off albedo and normal, so
    // the volume clips correctly against the far wall while the shell itself is
    // drawn later as a translucent overlay. A populated G-buffer fragment always
    // has a unit normal; the cleared value is zero. Without this test the ghosted
    // duct shades as an opaque black silhouette over the flow.
    if (depth > 0.0 && dot(nrm.xyz, nrm.xyz) > 0.25) {
        let albedo = textureLoad(gbuf_albedo, px, 0);
        let ao = textureLoad(ssao_tex, px, 0).x;
        let world = unproject(cam.inv_view_proj, ndc, depth);
        color = shade_opaque(albedo.xyz, albedo.w, normalize(nrm.xyz), world, ao);
    }

    if (cu.misc.z > 0.5) {
        // Bilinear upsample of the half-resolution volume.
        //
        // No bilateral weighting, on purpose: the raymarch already clips against
        // the *closest* depth in each 2x2 block, so at a silhouette it stops
        // early rather than late. The worst case is a half-res-pixel-wide gap
        // where the volume falls slightly short of an edge, which is invisible;
        // the alternative failure — volume leaking through geometry — is not.
        let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / fsize;
        let vol = textureSampleLevel(volume_color, linear_sampler, uv, 0.0);
        // Premultiplied over. Identical arithmetic to a `ONE,
        // ONE_MINUS_SRC_ALPHA` blend state, done here because the source is at
        // a different resolution and has to be resampled anyway.
        color = vol.xyz + color * (1.0 - clamp(vol.w, 0.0, 1.0));
    }

    textureStore(out_hdr, px, vec4<f32>(color, 1.0));
}

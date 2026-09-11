// Duct and obstruction geometry: a G-buffer pass plus two forward overlays.
//
// The G-buffer carries albedo, a world normal, a per-vertex scalar channel for
// later wall-quantity mapping (shear stress, wall pressure), and a motion vector
// for TAA. Shading happens in composite.wgsl, so the material model can change
// without touching the rasterisation path.
//
// Depth is **reverse-Z**: cleared to 0, tested `Greater`. See camera.rs.

#include "common.wgsl"

struct MeshUniform {
    model: mat4x4<f32>,
    prev_model: mat4x4<f32>,
    // rgb = albedo, a = alpha used by the ghost mode.
    base_color: vec4<f32>,
    // x = roughness, y = clearcoat, z = Fresnel power, w = rim strength.
    params: vec4<f32>,
    // xyz = direction toward the key light, w = intensity.
    light_dir: vec4<f32>,
    // x = ghost min alpha, y = ghost max alpha, z/w = per-vertex scalar range.
    ghost: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> mesh: MeshUniform;

struct VsIn {
    // Model-space position in millimetres, as exported by CAD.
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Per-vertex scalar carried through to the G-buffer so wall quantities can
    // be colour-mapped later without a second geometry pass.
    @location(2) scalar: f32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) scalar: f32,
    // Un-jittered clip positions, this frame and last, for motion vectors.
    @location(3) cur_clip: vec4<f32>,
    @location(4) prev_clip: vec4<f32>,
};

fn transform(input: VsIn) -> VsOut {
    let world = mesh.model * vec4<f32>(input.position, 1.0);
    let prev_world = mesh.prev_model * vec4<f32>(input.position, 1.0);
    var out: VsOut;
    // Rasterise with the *jittered* matrix so TAA sees a moving sample point.
    out.clip = cam.view_proj * world;
    out.world_pos = world.xyz;
    // Uniform scale is assumed (CAD parts are not sheared), so the upper 3x3
    // rotates normals correctly without an inverse transpose.
    out.world_normal = normalize((mesh.model * vec4<f32>(input.normal, 0.0)).xyz);
    out.scalar = input.scalar;
    // ...but motion vectors use the *un-jittered* matrices, or the jitter shows
    // up as scene motion and TAA spends every frame fighting its own sampling.
    out.cur_clip = cam.view_proj_unjit * world;
    out.prev_clip = cam.prev_view_proj_unjit * prev_world;
    return out;
}

@vertex
fn vs_mesh(input: VsIn) -> VsOut {
    return transform(input);
}

struct GBuffer {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

fn motion_vector(cur: vec4<f32>, prev: vec4<f32>) -> vec2<f32> {
    if (cur.w <= 0.0 || prev.w <= 0.0) {
        return vec2<f32>(0.0);
    }
    // NDC delta. TAA converts to UV, where the y axis flips.
    return cur.xy / cur.w - prev.xy / prev.w;
}

@fragment
fn fs_gbuffer(input: VsOut, @builtin(front_facing) front: bool) -> GBuffer {
    var n = normalize(input.world_normal);
    // Duct interiors are seen from the inside constantly. Flipping to the
    // visible side means back faces shade as surfaces instead of as black.
    if (!front) {
        n = -n;
    }
    var out: GBuffer;
    out.albedo = vec4<f32>(mesh.base_color.xyz, mesh.params.x);
    out.normal = vec4<f32>(n, input.scalar);
    out.motion = motion_vector(input.cur_clip, input.prev_clip);
    return out;
}

// --- forward overlays -------------------------------------------------------
//
// Ghost and wireframe draw straight into the HDR target after the volume has
// been composited, with `ONE, ONE_MINUS_SRC_ALPHA` blending and no depth write.
//
// Ghost mode culls **front** faces and draws only back faces. That is one
// transparent layer per pixel with no sorting and no depth peeling, and for a
// closed shell it is exactly the right layer: you see the far wall of the duct
// through the near wall, which is what makes the internal flow readable.

@fragment
fn fs_ghost(input: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var n = normalize(input.world_normal);
    if (!front) {
        n = -n;
    }
    let v = normalize(cam.eye_near.xyz - input.world_pos);
    let facing = abs(dot(n, v));

    // Fresnel: grazing angles are opaque, face-on is nearly clear. This is what
    // makes a ghosted shell read as a solid object rather than as a stain, and
    // it is physically the right behaviour for a dielectric.
    let fres = pow(clamp(1.0 - facing, 0.0, 1.0), max(mesh.params.z, 0.5));
    let a = clamp(mix(mesh.ghost.x, mesh.ghost.y, fres), 0.0, 1.0);

    let l = normalize(mesh.light_dir.xyz);
    let diffuse = max(dot(n, l), 0.0) * 0.5 + 0.5;
    // Rim light along the silhouette, which is the only place a transparent
    // shell has enough contrast to show its shape.
    let rim = pow(clamp(1.0 - facing, 0.0, 1.0), 3.0) * mesh.params.w;
    let rgb = mesh.base_color.xyz * diffuse * mesh.light_dir.w + vec3<f32>(rim);

    // Premultiplied, to match the blend state.
    return vec4<f32>(rgb * a, a);
}

@fragment
fn fs_wire(input: VsOut) -> @location(0) vec4<f32> {
    // Drawn from a de-duplicated edge index buffer as a line list, because
    // `PolygonMode::Line` needs an optional wgpu feature that `ad-gpu` does not
    // request, and a real edge list gives cleaner lines than triangle outlines
    // anyway.
    let a = clamp(mesh.base_color.w, 0.0, 1.0);
    let l = normalize(mesh.light_dir.xyz);
    let shade = 0.55 + 0.45 * max(dot(normalize(input.world_normal), l), 0.0);
    let rgb = mesh.base_color.xyz * shade;
    return vec4<f32>(rgb * a, a);
}

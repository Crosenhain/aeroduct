// Shared struct layouts and pure helper functions for the render passes.
//
// Deliberately declares NO bindings. Different passes want the camera at
// different group indices, and a header that hard-codes `@group(0)` forces every
// consumer into the same layout, which is how bind-group layouts end up with
// unused slots. Structs and functions here; bindings in the file that uses them.

const PI: f32 = 3.141592653589793;

// Mirrors `camera::CameraUniform`. Jittered and un-jittered matrices are both
// present because rasterisation and ray generation want the jittered pair while
// motion vectors must use the un-jittered pair; see camera.rs.
struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view_proj_unjit: mat4x4<f32>,
    prev_view_proj_unjit: mat4x4<f32>,
    // xyz = eye position mm, w = near plane mm
    eye_near: vec4<f32>,
    // xyz = unit forward, w = tan(fov_y / 2)
    forward_tan: vec4<f32>,
    // xy = NDC jitter, zw = viewport size in pixels
    jitter_resolution: vec4<f32>,
    // xy = 1 / viewport size, z = frame index, w = aspect
    inv_resolution_frame: vec4<f32>,
};

// Mirrors `fields::FieldsUniform`.
struct FieldsInfo {
    dims: vec3<u32>,
    channel: u32,
    volume_min_mm: vec3<f32>,
    voxel_mm: f32,
    volume_size_mm: vec3<f32>,
    diagonal_mm: f32,
    // The fields live in the lattice frame and the camera in the world frame;
    // these are the install pose that maps one onto the other. Rigid, so a ray
    // keeps its parameter `t` across the map.
    lattice_to_world: mat4x4<f32>,
    world_to_lattice: mat4x4<f32>,
};

// Pick the displayed scalar out of the packed vec4. `ch` comes from a uniform,
// so every invocation in a workgroup takes the same branch: no divergence.
fn pick_channel(v: vec4<f32>, ch: u32) -> f32 {
    var r = v.x;
    if (ch == 1u) {
        r = v.y;
    } else if (ch == 2u) {
        r = v.z;
    } else if (ch == 3u) {
        r = v.w;
    }
    return r;
}

// Reverse-Z depth -> positive view-space distance, mm.
//
// This is the single most copy-pasted three lines in a renderer and the single
// easiest place to lose reverse-Z consistency. It is the exact inverse of
// `Camera::projection` in camera.rs: depth = z_near / distance.
// Depth 0 is the cleared value and means "nothing here", reported as a huge
// distance rather than a division by zero.
fn linear_depth(z_near: f32, depth: f32) -> f32 {
    if (depth <= 0.0) {
        return 3.0e38;
    }
    return z_near / depth;
}

fn depth_from_linear(z_near: f32, dist: f32) -> f32 {
    if (dist <= 0.0) {
        return 1.0;
    }
    return min(z_near / dist, 1.0);
}

// Pixel centre -> NDC, with y flipped because texture space runs downward.
fn pixel_to_ndc(px: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    let uv = px / size;
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn unproject(inv_view_proj: mat4x4<f32>, ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let p = inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return p.xyz / p.w;
}

// World-space ray for a pixel. Built from two unprojected points because the
// infinite far plane makes the depth-0 row singular; depths 1.0 and 0.5 are
// both well conditioned. Matches `Camera::ray` on the CPU.
struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
};

fn camera_ray(cam: Camera, ndc: vec2<f32>) -> Ray {
    let near = unproject(cam.inv_view_proj, ndc, 1.0);
    let mid = unproject(cam.inv_view_proj, ndc, 0.5);
    var r: Ray;
    r.origin = near;
    r.dir = normalize(mid - near);
    return r;
}

// Reciprocal that never produces a 0 * inf NaN when a ray is exactly axis
// aligned and its origin sits exactly on a slab plane.
fn safe_rcp(d: vec3<f32>) -> vec3<f32> {
    let s = select(vec3<f32>(1.0), vec3<f32>(-1.0), d < vec3<f32>(0.0));
    return s / max(abs(d), vec3<f32>(1.0e-8));
}

// Slab test. Returns (t_enter, t_exit); the box is missed when exit <= enter.
fn ray_box(origin: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec2<f32> {
    let inv = safe_rcp(dir);
    let a = (lo - origin) * inv;
    let b = (hi - origin) * inv;
    let tmin = min(a, b);
    let tmax = max(a, b);
    return vec2<f32>(
        max(tmin.x, max(tmin.y, tmin.z)),
        min(tmax.x, min(tmax.y, tmax.z)),
    );
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

// Cheap integer hash, for per-pixel decorrelation where blue noise is overkill.
fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 17u);
    h = h * 0xed5ad4bbu;
    h = h ^ (h >> 11u);
    h = h * 0xac4c1b51u;
    h = h ^ (h >> 15u);
    h = h * 0x31848babu;
    h = h ^ (h >> 14u);
    return h;
}

fn hash_f32(x: u32) -> f32 {
    return f32(hash_u32(x)) * (1.0 / 4294967296.0);
}

// Fullscreen triangle. Three vertices, no vertex buffer, no index buffer, and
// one fewer edge across the middle of the screen than two triangles would have.
fn fullscreen_position(vertex_index: u32) -> vec4<f32> {
    let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(vertex_index) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn fullscreen_uv(vertex_index: u32) -> vec2<f32> {
    let x = f32(i32(vertex_index) / 2) * 2.0;
    let y = 1.0 - f32(i32(vertex_index) & 1) * 2.0;
    return vec2<f32>(x, y);
}

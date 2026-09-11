// Brick min/max, binarisation, and the Chebyshev distance transform.
//
// See `crates/ad-render/src/accel.rs` for why this structure and not an octree.
// `dilate_main` below is line-for-line the same recurrence as
// `accel::chebyshev_distance_transform`, which is checked against a brute-force
// reference on the CPU; that is how this shader is verified.
//
// The three passes live in one file but compile as three modules, selected by
// `BRICK_PASS_*`. They cannot share a module: WGSL requires every `@group`/
// `@binding` pair to be unique within a module, and these three want entirely
// different resources at group 0.

#include "common.wgsl"

const BRICK: i32 = #BRICK_SIZE;
const REDUCE_N: u32 = #REDUCE_WG;

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

#if BRICK_PASS_MINMAX

// group 0 is the shared derived-fields layout, verbatim.
@group(0) @binding(0) var<uniform> fu: FieldsInfo;
@group(0) @binding(1) var field_sampler: sampler;
@group(0) @binding(2) var field_scalars: texture_3d<f32>;
@group(0) @binding(3) var field_velocity: texture_3d<f32>;

@group(1) @binding(0) var<uniform> bu: BrickUniform;
@group(1) @binding(1) var out_minmax: texture_storage_3d<rg32float, write>;

var<workgroup> red_min: array<f32, REDUCE_N>;
var<workgroup> red_max: array<f32, REDUCE_N>;

// One workgroup per brick, reducing a 10^3 window: the 8^3 brick plus a
// one-voxel apron on every side.
//
// The apron is not paranoia. The raymarcher samples with trilinear filtering, so
// standing inside a brick it can read values up to half a voxel outside it, and
// near a brick boundary the filter footprint straddles the seam. Without the
// apron, thin structures lying exactly on a brick face get skipped, which shows
// up as a faint regular grid of missing shells visible only from certain angles
// — the worst kind of bug to chase.
@compute @workgroup_size(#REDUCE_WG, 1, 1)
fn minmax_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let base = vec3<i32>(wid) * BRICK - vec3<i32>(1);
    let span = BRICK + 2;
    let total = u32(span * span * span);
    let dims = vec3<i32>(fu.dims);

    var vmin = 1.0e30;
    var vmax = -1.0e30;

    var i = lid;
    loop {
        if (i >= total) {
            break;
        }
        let si = i32(i);
        let c = base + vec3<i32>(si % span, (si / span) % span, si / (span * span));
        if (all(c >= vec3<i32>(0)) && all(c < dims)) {
            // Fully solid voxels carry meaningless zeros. Folding them in would
            // drag zero into every brick's interval, and a soft isosurface whose
            // support straddles zero would then mark the whole duct wall active.
            if (textureLoad(field_velocity, c, 0).w > 0.0) {
                let v = pick_channel(textureLoad(field_scalars, c, 0), bu.channel);
                vmin = min(vmin, v);
                vmax = max(vmax, v);
            }
        }
        i = i + REDUCE_N;
    }

    red_min[lid] = vmin;
    red_max[lid] = vmax;
    workgroupBarrier();

    var s = REDUCE_N / 2u;
    loop {
        if (s == 0u) {
            break;
        }
        if (lid < s) {
            red_min[lid] = min(red_min[lid], red_min[lid + s]);
            red_max[lid] = max(red_max[lid], red_max[lid + s]);
        }
        workgroupBarrier();
        s = s / 2u;
    }

    if (lid == 0u) {
        var result = vec4<f32>(red_min[0], red_max[0], 0.0, 0.0);
        // A brick with no fluid voxels at all is flagged by min > max, which is
        // impossible for a real interval and so needs no extra channel.
        if (red_min[0] > red_max[0]) {
            result = vec4<f32>(1.0, -1.0, 0.0, 0.0);
        }
        textureStore(out_minmax, vec3<i32>(wid), result);
    }
}

#elif BRICK_PASS_SEED

@group(0) @binding(0) var<uniform> bu: BrickUniform;
@group(0) @binding(1) var seed_minmax: texture_3d<f32>;
@group(0) @binding(2) var seed_out: texture_storage_3d<r32uint, write>;

@compute @workgroup_size(#DILATE_WG_X, #DILATE_WG_Y, #DILATE_WG_Z)
fn seed_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= bu.brick_dims)) {
        return;
    }
    let mm = textureLoad(seed_minmax, vec3<i32>(gid), 0).xy;
    // Interval intersection against the transfer function's non-zero support.
    // Mirrors `transfer::Support::intersects` exactly, including the interior
    // gap that a diverging transfer function leaves around zero -- without the
    // gap test a pressure view marks every brick active and skips nothing.
    let valid = mm.x <= mm.y;
    let overlaps = mm.y >= bu.support_lo && mm.x <= bu.support_hi;
    let inside_gap = mm.x > bu.gap_lo && mm.y < bu.gap_hi;
    var d = bu.max_skip;
    if (valid && overlaps && !inside_gap) {
        d = 0u;
    }
    textureStore(seed_out, vec3<i32>(gid), vec4<u32>(d, 0u, 0u, 0u));
}

#else

@group(0) @binding(0) var<uniform> bu: BrickUniform;
@group(0) @binding(1) var dil_src: texture_3d<u32>;
@group(0) @binding(2) var dil_dst: texture_storage_3d<r32uint, write>;

// Outside the grid counts as "at least max_skip away", which can never lower a
// value that is already capped. Same convention as the CPU reference.
fn load_dist(c: vec3<i32>) -> u32 {
    if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(bu.brick_dims))) {
        return bu.max_skip;
    }
    return textureLoad(dil_src, c, 0).x;
}

@compute @workgroup_size(#DILATE_WG_X, #DILATE_WG_Y, #DILATE_WG_Z)
fn dilate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= bu.brick_dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    var best = load_dist(c);
    // Chebyshev distance is graph distance under 26-connectivity, so "min over
    // the 3x3x3 neighbourhood, plus one" is exactly one unit of the transform.
    // Run it max_skip times and it converges on the true distance.
    if (best > 0u) {
        for (var dz = -1; dz <= 1; dz = dz + 1) {
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    let nd = load_dist(c + vec3<i32>(dx, dy, dz));
                    best = min(best, min(nd + 1u, bu.max_skip));
                }
            }
        }
    }
    textureStore(dil_dst, c, vec4<u32>(best, 0u, 0u, 0u));
}

#endif

// Shared declarations for the signed-distance voxeliser.
//
// Every entry point in geom/voxelize.wgsl uses this one binding layout, so a
// single bind group serves the whole pipeline and the stages can be reordered
// or re-run without rebuilding anything.

struct GeomUniforms {
    dims: vec3<u32>,
    // Number of entries in `active_tris`, not the size of `tris`.
    tri_count: u32,

    origin_mm: vec3<f32>,
    dx_mm: f32,

    // Narrow-band half-width in millimetres. Distances beyond this are not
    // recorded; the flood fill supplies the sign out there.
    band_mm: f32,
    // Sweep axis for the flood fill: 0 = X lines, 1 = Y, 2 = Z.
    axis: u32,
    // 0 means "count links but write none", which is how the exact link count
    // is obtained before the output buffer is sized.
    link_capacity: u32,
    _pad_a: u32,

    // Inclusive cell range this invocation is allowed to touch. A full
    // voxelisation passes the whole grid; a drag passes only the box the moved
    // obstruction swept through.
    region_lo: vec3<u32>,
    _pad_b: u32,
    region_hi: vec3<u32>,
    _pad_c: u32,
};

struct Tri {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
};

// Angle-weighted pseudonormals for one triangle, precomputed on the CPU because
// they need edge adjacency and per-vertex accumulation over the whole mesh.
// `e0/e1/e2` follow the edge numbering (a,b), (b,c), (c,a); `v0/v1/v2` are the
// pseudonormals of corners a, b, c.
struct TriPn {
    face: vec3<f32>,
    e0: vec3<f32>,
    e1: vec3<f32>,
    e2: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
};

// Mirrors ad_gpu::BoundaryLink: cell index, then direction in the low byte and
// the quantised q in the second byte.
struct Link {
    cell: u32,
    packed: u32,
};

@group(0) @binding(0) var<uniform> U: GeomUniforms;
@group(0) @binding(1) var<storage, read> tris: array<Tri>;
@group(0) @binding(2) var<storage, read> tri_pn: array<TriPn>;
// Indirection so a partial re-voxelisation can dispatch over just the triangles
// whose spatial bins intersect the dirty region.
@group(0) @binding(3) var<storage, read> active_tris: array<u32>;

// Packed IEEE bits of the smallest |distance| seen at each cell. See
// `atomic_min_f32` for why a plain integer atomicMin is correct here.
@group(0) @binding(4) var<storage, read_write> dist: array<atomic<u32>>;
// Index of the triangle that achieved it.
@group(0) @binding(5) var<storage, read_write> tri_idx: array<atomic<u32>>;
// Signed distance, millimetres, negative inside the solid.
@group(0) @binding(6) var<storage, read_write> phi: array<f32>;
@group(0) @binding(7) var<storage, read_write> state: array<u32>;
// ad_gpu::flags bytes, four cells per word.
@group(0) @binding(8) var<storage, read_write> cell_flags: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> links: array<Link>;
// 0: link count, 1: solid cells, 2: boundary cells, 3: fill changed flag,
// 4: link overflow count.
@group(0) @binding(10) var<storage, read_write> counters: array<atomic<u32>>;

const NO_TRI: u32 = 0xFFFFFFFFu;
// 1e30, a finite float larger than any distance we will ever record, so the
// slot can always be bitcast back without producing a NaN.
const FAR_BITS: u32 = 0x7149F2CAu;

// Flood-fill states. The two band states are set by the pseudonormal pass and
// are never overwritten; only UNKNOWN cells are filled.
const ST_UNKNOWN: u32 = 0u;
const ST_EXTERIOR: u32 = 1u;
const ST_BAND_OUT: u32 = 2u;
const ST_BAND_IN: u32 = 3u;

fn cell_count() -> u32 {
    return U.dims.x * U.dims.y * U.dims.z;
}

// X fastest, matching ad_gpu::Grid::linear.
fn cell_index(c: vec3<u32>) -> u32 {
    return (c.z * U.dims.y + c.y) * U.dims.x + c.x;
}

fn cell_center(c: vec3<u32>) -> vec3<f32> {
    return U.origin_mm + vec3<f32>(c) * U.dx_mm;
}

// Atomic minimum on a non-negative f32, done as an integer atomicMin.
//
// For any two non-negative floats, a < b if and only if their IEEE-754 bit
// patterns compare the same way as unsigned integers: the sign bit is 0, the
// exponent occupies the next-most-significant bits, and the mantissa the rest,
// so the encoding is monotonic. Distances here are absolute values, hence
// always non-negative, so the bitcast needs no sign-flip fix-up. (Negative
// floats would need one; do not reuse this on the signed field.)
fn atomic_min_dist(cell: u32, d: f32) {
    atomicMin(&dist[cell], bitcast<u32>(d));
}

struct Closest {
    p: vec3<f32>,
    // 0..3 = vertex a/b/c, 3..6 = edge (a,b)/(b,c)/(c,a), 6 = face interior.
    feature: u32,
};

// Closest point on a triangle, reporting which feature it landed on.
//
// Ericson, Real-Time Collision Detection 5.1.5. This is a line-for-line
// translation of ad_geom::mesh::closest_point_on_triangle; the test
// `gpu_and_cpu_sdf_agree` compares the two over a whole grid and will fail if
// they drift apart.
//
// The feature is what makes the sign correct later: on an edge or a vertex the
// face normal is the wrong object to test against, and only the pseudonormal of
// that specific feature gives the right answer.
fn closest_point_on_tri(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Closest {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return Closest(a, 0u);
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return Closest(b, 1u);
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let den = d1 - d3;
        var t = 0.0;
        if (den != 0.0) { t = d1 / den; }
        return Closest(a + ab * t, 3u);
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return Closest(c, 2u);
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let den = d2 - d6;
        var t = 0.0;
        if (den != 0.0) { t = d2 / den; }
        return Closest(a + ac * t, 5u);
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let den = (d4 - d3) + (d5 - d6);
        var t = 0.0;
        if (den != 0.0) { t = (d4 - d3) / den; }
        return Closest(b + (c - b) * t, 4u);
    }

    let sum = va + vb + vc;
    if (sum == 0.0) {
        return Closest(a, 0u);
    }
    let den = 1.0 / sum;
    return Closest(a + ab * (vb * den) + ac * (vc * den), 6u);
}

// The outward pseudonormal for a closest point on `feature` of triangle `t`.
fn pseudonormal(t: u32, feature: u32) -> vec3<f32> {
    let pn = tri_pn[t];
    switch (feature) {
        case 0u: { return pn.v0; }
        case 1u: { return pn.v1; }
        case 2u: { return pn.v2; }
        case 3u: { return pn.e0; }
        case 4u: { return pn.e1; }
        case 5u: { return pn.e2; }
        default: { return pn.face; }
    }
}

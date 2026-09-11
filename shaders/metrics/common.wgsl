// Shared field sampling for every metrics kernel.
//
// Included after the generated prelude (flag constants and accumulator layout,
// emitted from the Rust definitions by ad_metrics::shaders so the two cannot
// drift). Everything here works in LATTICE UNITS: velocity is the solver's
// lattice velocity, density is the lattice density near 1.0. Conversion to SI
// happens once, on the CPU, in ad_metrics::metrics, because doing it here would
// mean threading four more uniforms through three kernels for no gain.
//
// Group 0 is the same for every metrics pass:
//   0  velocity, texture_3d<f32>, sampled view (NOT the storage view)
//   1  density,  texture_3d<f32>, sampled view
//   2  flag bytes, four cells per u32 word, X fastest
//
// Per CONTRACT.md rule 5 these are sampled views of the textures the solver's
// macroscopic pass writes through a separate storage view.

@group(0) @binding(0) var velocity_tex: texture_3d<f32>;
@group(0) @binding(1) var density_tex: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> cell_flags: array<u32>;

// The grid the textures cover. Embedded in each pass's uniform rather than
// bound separately, so a pass needs exactly one uniform.
struct GridInfo {
    dims: vec3<u32>,
    dx_mm: f32,
    origin_mm: vec3<f32>,   // centre of cell (0,0,0)
    pad: f32,
};

const CS2: f32 = 0.3333333333333333;
const PI: f32 = 3.14159265358979;

fn cell_linear(g: GridInfo, c: vec3<u32>) -> u32 {
    return (c.z * g.dims.y + c.y) * g.dims.x + c.x;
}

fn flags_at(g: GridInfo, c: vec3<u32>) -> u32 {
    let i = cell_linear(g, c);
    return (cell_flags[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

fn is_solid(f: u32) -> bool {
    return (f & FLAG_SOLID) != 0u;
}

struct FieldSample {
    u: vec3<f32>,
    rho: f32,
    // The sample's containing cell exists in the grid.
    inside: bool,
    // ...and is not solid. Only `fluid` samples enter an integral; the rest are
    // counted so the covered-area fraction can be reported honestly.
    fluid: bool,
};

// Trilinear interpolation of (u, rho) at a point in millimetres.
//
// Coverage is decided by the *containing* cell (nearest cell centre, matching
// ad_gpu::Grid::cell_containing), not by the interpolation stencil: a point half
// a cell inside the passage is inside the passage even though four of its eight
// stencil corners are wall.
//
// The stencil corners are clamped to the grid. That only ever matters in the
// outer half-cell, because a point whose containing cell is outside was already
// rejected, and there nearest-neighbour extension is the right extrapolation:
// the alternative, letting the fetch wrap, would read the far side of the
// domain.
//
// Solid corners need no special case. shaders/lbm/macroscopic.wgsl writes u = 0
// and rho = 1 into every solid cell, which is exactly the no-slip Dirichlet
// value, so the interpolant decays to zero across the wall as it should.
fn sample_field(g: GridInfo, p_mm: vec3<f32>) -> FieldSample {
    var s: FieldSample;
    s.u = vec3<f32>(0.0, 0.0, 0.0);
    s.rho = 1.0;
    s.inside = false;
    s.fluid = false;

    let dims = vec3<i32>(g.dims);
    let gc = (p_mm - g.origin_mm) / g.dx_mm;
    let ci = vec3<i32>(round(gc));
    if (any(ci < vec3<i32>(0, 0, 0)) || any(ci >= dims)) {
        return s;
    }
    s.inside = true;
    if (is_solid(flags_at(g, vec3<u32>(ci)))) {
        return s;
    }
    s.fluid = true;

    let base = vec3<i32>(floor(gc));
    let f = gc - floor(gc);
    var acc_u = vec3<f32>(0.0, 0.0, 0.0);
    var acc_r = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) {
        let o = vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
        let wv = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
        let wt = wv.x * wv.y * wv.z;
        let q = clamp(base + o, vec3<i32>(0, 0, 0), dims - vec3<i32>(1, 1, 1));
        acc_u = acc_u + wt * textureLoad(velocity_tex, q, 0).xyz;
        acc_r = acc_r + wt * textureLoad(density_tex, q, 0).x;
    }
    s.u = acc_u;
    s.rho = acc_r;
    return s;
}

// Read a cell without interpolating. Used by the volume reductions, which visit
// cell centres and must not smear a peak across its neighbours.
fn load_cell(g: GridInfo, c: vec3<u32>) -> FieldSample {
    var s: FieldSample;
    let q = vec3<i32>(c);
    s.u = textureLoad(velocity_tex, q, 0).xyz;
    s.rho = textureLoad(density_tex, q, 0).x;
    s.inside = true;
    s.fluid = !is_solid(flags_at(g, c));
    if (!s.fluid) {
        s.u = vec3<f32>(0.0, 0.0, 0.0);
        s.rho = 1.0;
    }
    return s;
}

// Static pressure in lattice units, p_s = c_s^2 (rho - 1).
fn static_pressure_lb(rho: f32) -> f32 {
    return CS2 * (rho - 1.0);
}

// Total pressure in lattice units, p_t = p_s + 1/2 rho |u|^2.
//
// Both terms convert to pascals through the *same* factor rho_phys * (dx/dt)^2,
// which is why they can be summed here and converted once on the CPU. See
// ad_metrics::metrics::pressure_pa.
fn total_pressure_lb(rho: f32, u: vec3<f32>) -> f32 {
    return static_pressure_lb(rho) + 0.5 * rho * dot(u, u);
}

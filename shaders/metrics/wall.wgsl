// Wall pressure and shear stress, one record per surface triangle.
//
// # Why this is not just "sample the field at the vertices"
//
// The obvious implementation — trilinearly interpolate the Cartesian velocity
// and density at each STL vertex — is wrong, and wrong in a way that looks
// plausible. A vertex sits *on* the boundary, which in a lattice-Boltzmann
// solver is the half-way point between the last fluid node and the first solid
// node. Half the stencil corners of that interpolation are bounce-back cells
// carrying u = 0 and rho = 1 by fiat, so the interpolated "wall velocity" is
// simply the fluid velocity divided by two, and the interpolated pressure is
// pulled toward the reference. Both errors are smooth, both scale with the real
// answer, and neither shows up as a NaN or a spike. You get a wall shear map
// that is about half of the truth and looks beautiful.
//
// So this kernel never evaluates anything at the surface. It walks *outward*
// along the triangle normal until it finds a cell the flag byte says is fluid,
// evaluates there, and uses the triangle itself only as the place where the
// no-slip condition u = 0 is known exactly.
//
// # The stress
//
// The physically correct route is momentum exchange (Ladd 1994; Mei et al.
// 2002): sum `c_i (f_i(x_f) + f_i~(x_f + c_i))` over the boundary links. That
// needs the raw populations, which `ad_solver::Solver` does not expose — the
// DDF buffers and the boundary link list are both private. The documented
// fallback is the non-equilibrium stress at the first fluid cell,
//
//     sigma = -(1 - dt/(2 tau)) * Pi_neq,   Pi_neq_ab = sum_i c_ia c_ib f_i^neq
//
// which needs the populations too. What *is* available is the macroscopic
// velocity field, and the second-order Chapman-Enskog closure that makes the
// two equivalent:
//
//     Pi_neq_ab = -2 rho c_s^2 tau S_ab            (dt = 1, lattice units)
//     => sigma_ab = (1 - 1/(2 tau)) * 2 rho c_s^2 tau S_ab
//                 = 2 rho c_s^2 (tau - 1/2) S_ab
//                 = 2 rho nu_lb S_ab
//
// i.e. the Newtonian deviatoric stress, with the `(1 - dt/(2 tau))` correction
// already folded in. That last line is what this kernel computes. It is the
// same number the Pi_neq route returns whenever the closure holds, and it
// degrades in exactly the same place the closure does (very strong gradients,
// tau far from 1/2).
//
// # The strain rate, and where the wall actually is
//
// `S_ab = 1/2 (d_a u_b + d_b u_a)` is assembled in the *triangle's* local frame
// rather than in lattice axes:
//
//   * along the normal, `du/dn = u(P1) / y1`. This is the one derivative that
//     matters for wall shear, and taking it against the known wall value u = 0
//     at the triangle plane — not against a neighbouring solid cell centre —
//     is what avoids the factor-of-two above. It is exact wherever the profile
//     is linear, and CONTRACT.md puts the first fluid node at y+ = 1.8..6.9,
//     i.e. inside the viscous sublayer, precisely where it is.
//   * along the two in-plane tangents, a central difference between fluid
//     samples, falling back to one-sided when a neighbour is inside the wall.
//     These terms are small for a wall-bounded flow but they are what makes the
//     tensor frame-independent, so a curved surface does not report shear that
//     is really a rotation.
//
// Then, with `n` the unit normal pointing from the solid into the fluid, the
// traction the fluid exerts on the solid is `t = sigma . n`, and
// `tau_w = |t - (t.n) n|`.
//
// # y+
//
// `y+ = u_tau y / nu` with `u_tau = sqrt(tau_w / rho)`. Dimensionless, so it is
// the same number in lattice and SI units and needs no conversion. It is
// reported because the contract's decision to run **no wall function** is only
// valid while y+ stays inside the viscous sublayer; if this climbs past ~11 the
// first fluid node has left the linear region and both the shear above and the
// solver's plain bounce-back are being asked for more than they can give.
//
// # Units
//
// Everything is in lattice units, as everywhere else in `shaders/metrics`. The
// one exception is area: the per-triangle records carry `stress_lb * area_mm2`,
// because the triangles arrive in millimetres and converting them per-triangle
// on the GPU would mean carrying a second length scale for no reason. The CPU
// multiplies by `rho_phys * (dx/dt)^2 * 1e-6` once, in `ad_metrics::wall`.

#include "metrics/common.wgsl"

struct WallUniforms {
    grid: GridInfo,
    /// Triangles in the buffer. Threads past this return immediately.
    tri_count: u32,
    /// Base kinematic viscosity in lattice units, `c_s^2 (tau0 - 1/2)`.
    nu_lb: f32,
    /// Smagorinsky constant, already multiplied by nothing: `dx = 1` in lattice
    /// units so the filter width is 1. Set to 0 to report the molecular stress
    /// only, which is the right choice when the solver's LES is also off.
    smagorinsky_c: f32,
    /// First probe offset from the triangle plane, in cells.
    probe_start: f32,

    /// Spacing between successive probes, in cells.
    probe_step: f32,
    /// How many probes to try before giving up on this triangle.
    probe_count: u32,
    /// Tangential finite-difference step, in cells.
    tangent_h: f32,
    pad: f32,
};

/// One triangle in millimetres. Explicit padding because a `vec3` in a storage
/// buffer is 16-byte aligned, so the Rust mirror must be 48 bytes too.
struct Tri {
    a: vec3<f32>,
    pa: f32,
    b: vec3<f32>,
    pb: f32,
    c: vec3<f32>,
    pc: f32,
};

@group(1) @binding(0) var<uniform> W: WallUniforms;
// One WALL_STRIDE-slot record per triangle. Written by exactly one invocation
// each, so unlike the plane and volume accumulators this needs no atomics.
@group(1) @binding(1) var<storage, read_write> wall: array<f32>;
@group(1) @binding(2) var<storage, read> tris: array<Tri>;

/// `M_ij = a_i b_j`. WGSL matrices are column-major and `m[j][i]` is row `i` of
/// column `j`, so column `j` of the outer product is `a * b[j]`.
fn outer(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(a * b.x, a * b.y, a * b.z);
}

/// Directional derivative of velocity along a unit direction, per *cell*.
///
/// Central where both neighbours are fluid, one-sided where only one is, and
/// zero where the sample is pinched between two walls — which is the honest
/// answer for a passage one cell wide, not a reason to fabricate a gradient.
fn deriv_along(p_mm: vec3<f32>, dir: vec3<f32>, h_cells: f32, u0: vec3<f32>) -> vec3<f32> {
    let step_mm = dir * (h_cells * W.grid.dx_mm);
    let sp = sample_field(W.grid, p_mm + step_mm);
    let sm = sample_field(W.grid, p_mm - step_mm);
    if (sp.fluid && sm.fluid) {
        return (sp.u - sm.u) / (2.0 * h_cells);
    }
    if (sp.fluid) {
        return (sp.u - u0) / h_cells;
    }
    if (sm.fluid) {
        return (u0 - sm.u) / h_cells;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64, 1, 1)
fn wall_stress(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= W.tri_count) {
        return;
    }
    let base = t * WALL_STRIDE;

    // Clear first and return early on any failure, so a triangle that never
    // found fluid reports W_AREA = 0 rather than last frame's numbers. The CPU
    // uses that zero as the "dry" marker; see ad_metrics::wall::WallRecord.
    for (var k = 0u; k < WALL_STRIDE; k = k + 1u) {
        wall[base + k] = 0.0;
    }

    let tri = tris[t];
    let cr = cross(tri.b - tri.a, tri.c - tri.a);
    let two_area = length(cr);
    if (!(two_area > 0.0)) {
        return; // degenerate sliver
    }
    // Winding is outward per the STL convention, so this points from the solid
    // into the fluid — into the passage for an internal face.
    let n = cr / two_area;
    let area_mm2 = 0.5 * two_area;
    let centroid = (tri.a + tri.b + tri.c) * (1.0 / 3.0);

    // Walk outward until the flag byte says the containing cell is fluid.
    var y_cells = 0.0;
    var fs: FieldSample;
    var found = false;
    for (var k = 0u; k < W.probe_count; k = k + 1u) {
        let yc = W.probe_start + f32(k) * W.probe_step;
        let s = sample_field(W.grid, centroid + n * (yc * W.grid.dx_mm));
        if (s.fluid) {
            y_cells = yc;
            fs = s;
            found = true;
            break;
        }
    }
    if (!found || !(y_cells > 0.0)) {
        return;
    }
    let p = centroid + n * (y_cells * W.grid.dx_mm);

    // An orthonormal in-plane frame. Seeding from whichever axis is least
    // aligned with the normal keeps the cross product well conditioned.
    var seed = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(n.x) > 0.9) {
        seed = vec3<f32>(0.0, 1.0, 0.0);
    }
    let t1 = normalize(seed - n * dot(n, seed));
    let t2 = cross(n, t1);

    // du/dn against the exact no-slip value at the triangle plane; the two
    // in-plane derivatives by finite difference among fluid samples.
    let dudn = fs.u / y_cells;
    let dudt1 = deriv_along(p, t1, W.tangent_h, fs.u);
    let dudt2 = deriv_along(p, t2, W.tangent_h, fs.u);

    // G_ab = d u_b / d x_a, assembled from the three directional derivatives.
    let g = outer(n, dudn) + outer(t1, dudt1) + outer(t2, dudt2);
    let s_rate = 0.5 * (g + transpose(g));

    // |S| = sqrt(2 S:S), the Smagorinsky-Lilly strain-rate invariant. The eddy
    // viscosity is (Cs * filter width)^2 |S| and the filter width is one cell,
    // which is 1 in lattice units.
    var ss = 0.0;
    for (var i = 0; i < 3; i = i + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            ss = ss + s_rate[j][i] * s_rate[j][i];
        }
    }
    let s_mag = sqrt(2.0 * ss);
    let nu_eff = W.nu_lb + W.smagorinsky_c * W.smagorinsky_c * s_mag;

    // sigma = 2 rho nu_eff S; see the header for why this already carries the
    // (1 - dt/(2 tau)) factor of the Pi_neq form.
    let sigma = (2.0 * fs.rho * nu_eff) * s_rate;
    let trac = sigma * n;
    let shear = trac - n * dot(trac, n);
    let tau_w = length(shear);

    let p_lb = static_pressure_lb(fs.rho);
    // y+ is defined with the MOLECULAR viscosity, not the effective one. Using
    // nu_eff would shrink y+ exactly where the eddy viscosity is largest, i.e.
    // exactly where the first fluid node is most likely to have left the viscous
    // sublayer, and would hide the one thing this number exists to warn about.
    let u_tau = sqrt(max(tau_w, 0.0) / max(fs.rho, 1.0e-6));
    var y_plus = 0.0;
    if (W.nu_lb > 0.0) {
        y_plus = u_tau * y_cells / W.nu_lb;
    }

    // Pressure acts along -n (into the solid); the viscous traction is stored
    // whole, normal component included, so the CPU can sum a genuine force.
    let fp = n * (-p_lb * area_mm2);
    let fv = trac * area_mm2;

    wall[base + W_FP_X + 0u] = fp.x;
    wall[base + W_FP_X + 1u] = fp.y;
    wall[base + W_FP_X + 2u] = fp.z;
    wall[base + W_FV_X + 0u] = fv.x;
    wall[base + W_FV_X + 1u] = fv.y;
    wall[base + W_FV_X + 2u] = fv.z;
    wall[base + W_TAU_A] = tau_w * area_mm2;
    wall[base + W_YPLUS_A] = y_plus * area_mm2;
    wall[base + W_AREA] = area_mm2;
    wall[base + W_P_A] = p_lb * area_mm2;
    wall[base + W_Y_A] = y_cells * area_mm2;
}

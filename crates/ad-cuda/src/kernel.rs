//! Generated CUDA C: the lattice tables, the parameter block, and the Esoteric
//! Pull load/store bodies.
//!
//! # Why this is generated, and generated *this* way
//!
//! `ad_solver::shaders` makes the argument for WGSL and every word of it applies
//! here: the transport is `q` near-identical blocks of index arithmetic, and a
//! hand transcription of them is the sort of thing that goes wrong once and then
//! produces plausible-looking physics forever. A second backend doubles that
//! risk, because now there are two transcriptions that must agree with each
//! other *and* with the Rust.
//!
//! So this module does not re-derive anything. It calls
//! [`ad_solver::shaders::probe`] — the same function that emits the WGSL — which
//! runs the real [`ad_gpu::lattice::EsotericPull`] against a sentinel cell count
//! so `slot()` becomes an invertible encoding of `(direction, which cell)`, and
//! reads the answer back out. The CUDA and the WGSL therefore descend from one
//! implementation, the one that
//! `esoteric_pull_store_then_load_round_trips` covers exhaustively. There is no
//! third place for the scheme to be written down wrongly.
//!
//! The direction vectors and weights come from [`ad_gpu::lattice`] and the flag
//! bits from [`ad_gpu::types::flags`], for the same reason.
//!
//! # Where this deliberately differs from the WGSL
//!
//! Two places, both forced and both noted so an A/B against the Vulkan backend
//! is read correctly.
//!
//! 1. **One allocation, not `q`.** WGSL needs `q` separate storage buffers
//!    because wgpu clamps a *binding* to 2 GiB - 1 (see [`ad_gpu::ddf`]); CUDA
//!    has no such limit, so the DDFs are one flat array and
//!    `slot(cell, dir) = dir * cell_count + cell` is literal pointer arithmetic.
//!    The bytes and their order are identical — plane `i` of the flat array is
//!    byte-for-byte what `ddf[i]` holds in the wgpu path — so this changes
//!    addressing, not layout, not traffic and not coalescing. It does mean the
//!    slot index can exceed 32 bits on a large grid (19 x 312 M = 5.9e9), which
//!    the per-binding WGSL indices never can, so the accessors widen to 64-bit.
//!
//! 2. **FP32 only.** The FP16C codec is not ported. The measurement this backend
//!    exists to make is against the FP32 kernel, and FP16C would add a second
//!    variable to it. `ad_solver::precision` remains the only implementation.
//!
//! Everything else — the `((p % n) + n) % n` wrapping neighbour, the branch-free
//! bounce-back via the load-parity flip, the unrolled pair loop, the workgroup
//! shape, the order of operations inside the moments — is mirrored exactly,
//! including the parts that are not obviously optimal. A backend comparison is
//! only worth anything if both sides run the same algorithm.

use ad_gpu::lattice::pair_first;
use ad_gpu::types::VelocitySet;
use ad_solver::collision::CollisionModel;
use ad_solver::shaders::probe;
use std::fmt::Write as _;

/// The parameter block, mirroring `ad_solver::solver::LbmUniforms` field for
/// field.
///
/// Every field is four bytes and there is no padding anywhere, because a CUDA
/// kernel parameter is a plain C struct rather than a WGSL uniform — none of the
/// 16-byte `vec3` alignment that forces `LbmUniforms` to interleave a scalar
/// after each `[T; 3]` applies. The layout is asserted against
/// [`PARAM_FIELDS`] by `param_struct_matches_the_generated_c_struct`, which is
/// what stops the two from drifting: a mismatch here is not a compile error, it
/// is a kernel silently reading `tau0` out of `trt_lambda`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaParams {
    pub dims_x: u32,
    pub dims_y: u32,
    pub dims_z: u32,
    pub step_parity: u32,
    pub interior_x: u32,
    pub interior_y: u32,
    pub interior_z: u32,
    pub cell_count: u32,
    pub offset_x: u32,
    pub offset_y: u32,
    pub offset_z: u32,
    pub flag_words: u32,
    pub inlet_velocity_x: f32,
    pub inlet_velocity_y: f32,
    pub inlet_velocity_z: f32,
    pub tau0: f32,
    pub initial_velocity_x: f32,
    pub initial_velocity_y: f32,
    pub initial_velocity_z: f32,
    pub trt_lambda: f32,
    pub body_force_x: f32,
    pub body_force_y: f32,
    pub body_force_z: f32,
    pub smagorinsky_c: f32,
    pub tau_max: f32,
    pub outflow_velocity: f32,
    pub rho_ref: f32,
    pub sponge_strength: f32,
    pub sponge_cells: u32,
    pub periodic: u32,
    pub total_steps: u32,
    pub outlet_anti_bounce_back: u32,
    pub outlet_normal_x: f32,
    pub outlet_normal_y: f32,
    pub outlet_normal_z: f32,
    pub inlet_normal_x: f32,
    pub inlet_normal_y: f32,
    pub inlet_normal_z: f32,
}

const _: () = assert!(std::mem::size_of::<CudaParams>() == PARAM_FIELDS.len() * 4);

/// `(C type, field name)` for every member of `struct Params`, in declaration
/// order. The single source for the generated C struct; [`CudaParams`] is
/// checked against it field by field.
pub const PARAM_FIELDS: &[(&str, &str)] = &[
    ("unsigned int", "dims_x"),
    ("unsigned int", "dims_y"),
    ("unsigned int", "dims_z"),
    ("unsigned int", "step_parity"),
    ("unsigned int", "interior_x"),
    ("unsigned int", "interior_y"),
    ("unsigned int", "interior_z"),
    ("unsigned int", "cell_count"),
    ("unsigned int", "offset_x"),
    ("unsigned int", "offset_y"),
    ("unsigned int", "offset_z"),
    ("unsigned int", "flag_words"),
    ("float", "inlet_velocity_x"),
    ("float", "inlet_velocity_y"),
    ("float", "inlet_velocity_z"),
    ("float", "tau0"),
    ("float", "initial_velocity_x"),
    ("float", "initial_velocity_y"),
    ("float", "initial_velocity_z"),
    ("float", "trt_lambda"),
    ("float", "body_force_x"),
    ("float", "body_force_y"),
    ("float", "body_force_z"),
    ("float", "smagorinsky_c"),
    ("float", "tau_max"),
    ("float", "outflow_velocity"),
    ("float", "rho_ref"),
    ("float", "sponge_strength"),
    ("unsigned int", "sponge_cells"),
    ("unsigned int", "periodic"),
    ("unsigned int", "total_steps"),
    ("unsigned int", "outlet_anti_bounce_back"),
    ("float", "outlet_normal_x"),
    ("float", "outlet_normal_y"),
    ("float", "outlet_normal_z"),
    ("float", "inlet_normal_x"),
    ("float", "inlet_normal_y"),
    ("float", "inlet_normal_z"),
];

/// Entry point names, so the Rust side cannot misspell one and get a runtime
/// "named symbol not found" instead of a compile error.
pub const FN_INIT: &str = "lbm_init";
pub const FN_STREAM_COLLIDE: &str = "lbm_stream_collide";
pub const FN_MACROSCOPIC: &str = "lbm_macroscopic";

/// Everything the kernel needs to know at compile time.
///
/// These are exactly the fields `SolverConfig::needs_rebuild` treats as
/// compile-time, minus the storage precision (FP32 only here) and the
/// macroscopic-buffer flag (always on: headless has no textures, so the buffer
/// is the only output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSpec {
    pub set: VelocitySet,
    pub collision: CollisionModel,
    /// Threads per block along X. X is the fastest-varying axis, so this is the
    /// direction that coalesces, and it is the same knob as
    /// `SolverConfig::workgroup_size`.
    pub block_x: u32,
}

/// The lattice tables, as `__constant__` arrays.
///
/// `__constant__` rather than literals baked into the unrolled code: every lane
/// in a warp reads the same element on the same iteration, which is the
/// broadcast case the constant cache exists for, and it keeps the emitted source
/// readable. The direction components are stored as `float` even though they are
/// all in `{-1, 0, 1}`, because every use in the physics is a float multiply;
/// the two places that need integers (`neighbour_index` and `is_outflow_link`)
/// cast, exactly as the WGSL's `vec3<f32>(C[i])` does in reverse.
fn tables(set: VelocitySet) -> String {
    let def = set.def();
    let mut s = String::with_capacity(2048);
    let q = def.q;
    let _ = writeln!(s, "#define Q {q}");
    let _ = writeln!(s, "#define Q_CONST {q}");
    s.push('\n');
    for (name, pick) in [
        ("CX", 0usize),
        ("CY", 1),
        ("CZ", 2),
    ] {
        let _ = write!(s, "__constant__ float {name}[Q] = {{");
        for (n, d) in def.directions.iter().enumerate() {
            let v = [d.x, d.y, d.z][pick];
            if n > 0 {
                s.push_str(", ");
            }
            let _ = write!(s, "{v}.0f");
        }
        s.push_str("};\n");
    }
    s.push_str("__constant__ float W[Q] = {");
    for (n, w) in def.weights.iter().enumerate() {
        if n > 0 {
            s.push_str(", ");
        }
        // 17 significant digits, matching `lattice::wgsl_prelude`, so the two
        // backends start from bit-identical weights.
        let _ = write!(s, "{w:.17}f");
    }
    s.push_str("};\n");
    s
}

/// The [`ad_gpu::types::flags`] bitfield, emitted so the kernel cannot hold a
/// stale copy of a value that lives in another crate.
fn flag_constants() -> String {
    use ad_gpu::types::flags;
    let mut s = String::from("// from ad_gpu::types::flags\n");
    for (name, value) in [
        ("SOLID", flags::SOLID),
        ("SOLID_BOUNDARY", flags::SOLID_BOUNDARY),
        ("INLET", flags::INLET),
        ("OUTLET", flags::OUTLET),
        ("SPONGE", flags::SPONGE),
        ("EQUILIBRIUM", flags::EQUILIBRIUM),
    ] {
        let _ = writeln!(s, "#define FLAG_{name} {value}u");
    }
    s
}

fn param_struct() -> String {
    let mut s = String::from("struct Params {\n");
    for (ty, name) in PARAM_FIELDS {
        let _ = writeln!(s, "    {ty} {name};");
    }
    s.push_str("};\n");
    s
}

/// Per-direction typed accessors into the flat SoA array.
///
/// One function per direction rather than one indexed by `dir`, so that after
/// inlining the `(size_t)dir * cell_count` base is a compile-time-scaled
/// loop-invariant the compiler hoists once per kernel instead of recomputing per
/// access. This mirrors the WGSL's `ddf_get_{i}` / `ddf_put_{i}`, which exist
/// there because WGSL has no arrays of bindings; here it is purely for codegen.
fn accessors(set: VelocitySet) -> String {
    let q = set.q();
    let mut s = String::with_capacity(4096);
    s.push_str(
        "// FP32 storage. The value stored is the *shifted* population g = f - w;\n\
         // see ad_solver::precision for why that matters even when storage is exact.\n\
         //\n\
         // The index is widened to 64 bits because dir * cell_count overflows a u32\n\
         // past ~226 M cells, which is inside the resolution tiers this app ships.\n\
         // The wgpu path cannot hit this: it indexes q separate bindings by cell.\n",
    );
    for i in 0..q {
        let _ = writeln!(
            s,
            "__device__ __forceinline__ float ddf_get_{i}(const float* ddf, \
             unsigned int cc, unsigned int cell) {{ return ddf[(unsigned long long){i} * cc + cell]; }}"
        );
        let _ = writeln!(
            s,
            "__device__ __forceinline__ void ddf_put_{i}(float* ddf, \
             unsigned int cc, unsigned int cell, float v) {{ ddf[(unsigned long long){i} * cc + cell] = v; }}"
        );
    }
    s
}

/// The unrolled Esoteric Pull load and store bodies.
///
/// A near-line-for-line translation of `ad_solver::shaders::transport`, from the
/// same [`probe`] calls. `mask` bit `i` set means the upstream neighbour of
/// direction `i` is solid, in which case the *load* uses the opposite step
/// parity — which reads back the population this cell pushed into the wall link
/// last step, i.e. halfway bounce-back at zero extra traffic and with a two-way
/// branch that is uniform everywhere except at a wall. Stores are never flipped.
fn transport(set: VelocitySet) -> String {
    let def = set.def();
    let q = def.q;
    let mut s = String::with_capacity(16384);
    s.push_str(
        "// Addresses come from ad_gpu::lattice::EsotericPull itself, via\n\
         // ad_solver::shaders::probe - the same call the WGSL generator makes. This\n\
         // is a transcription of what that code returns, not a re-derivation of it.\n\n",
    );

    // ---- load ----
    s.push_str(
        "__device__ __forceinline__ void load_ddf(const Params& P, const float* ddf,\n\
         \x20       unsigned int cell, unsigned int cx, unsigned int cy, unsigned int cz,\n\
         \x20       bool odd, unsigned int mask, float* g) {\n\
         \x20   const unsigned int cc = P.cell_count;\n",
    );
    let a0 = probe(q, 0, false, false);
    assert_eq!(a0.dir, 0, "the rest population must live in buffer 0");
    s.push_str("    g[0] = ddf_get_0(ddf, cc, cell);\n");

    let mut i = 1;
    while i < q {
        assert_eq!(pair_first(i), i, "pairs must start on odd indices");
        let even = probe(q, i, false, false);
        let odd = probe(q, i, true, false);
        let nd = even
            .neighbour_dir
            .or(odd.neighbour_dir)
            .expect("the odd member of a pair must load through a neighbour");
        let d = def.directions[nd];
        let _ = writeln!(s, "    {{ // pair ({i}, {})", i + 1);
        let _ = writeln!(
            s,
            "        const unsigned int nb = neighbour_index(P, cx, cy, cz, {}, {}, {});",
            d.x, d.y, d.z
        );
        for k in 0..2 {
            let dir = i + k;
            let e = probe(q, dir, false, false);
            let o = probe(q, dir, true, false);
            let idx_e = if e.neighbour_dir.is_some() { "nb" } else { "cell" };
            let idx_o = if o.neighbour_dir.is_some() { "nb" } else { "cell" };
            assert_eq!(idx_e, idx_o, "a load's address base must not depend on parity");
            let _ = writeln!(
                s,
                "        if (odd != (((mask >> {dir}u) & 1u) != 0u)) {{ g[{dir}] = ddf_get_{}(ddf, cc, {idx_o}); }}\n\
                 \x20       else {{ g[{dir}] = ddf_get_{}(ddf, cc, {idx_e}); }}",
                o.dir, e.dir
            );
        }
        s.push_str("    }\n");
        i += 2;
    }
    s.push_str("}\n\n");

    // ---- store ----
    s.push_str(
        "__device__ __forceinline__ void store_ddf(const Params& P, float* ddf,\n\
         \x20       unsigned int cell, unsigned int cx, unsigned int cy, unsigned int cz,\n\
         \x20       bool odd, const float* g) {\n\
         \x20   const unsigned int cc = P.cell_count;\n\
         \x20   ddf_put_0(ddf, cc, cell, g[0]);\n",
    );
    let mut i = 1;
    while i < q {
        let even = probe(q, i, false, true);
        let odd_a = probe(q, i, true, true);
        let even_b = probe(q, i + 1, false, true);
        let odd_b = probe(q, i + 1, true, true);
        let nd = even_b
            .neighbour_dir
            .or(odd_b.neighbour_dir)
            .expect("the even member's store must use a neighbour");
        assert!(
            even.neighbour_dir.is_none() && odd_a.neighbour_dir.is_none(),
            "the odd member's store must be local"
        );
        let d = def.directions[nd];
        let _ = writeln!(s, "    {{ // pair ({i}, {})", i + 1);
        let _ = writeln!(
            s,
            "        const unsigned int nb = neighbour_index(P, cx, cy, cz, {}, {}, {});",
            d.x, d.y, d.z
        );
        let _ = writeln!(
            s,
            "        if (odd) {{ ddf_put_{}(ddf, cc, cell, g[{i}]); ddf_put_{}(ddf, cc, nb, g[{}]); }}\n\
             \x20       else {{ ddf_put_{}(ddf, cc, cell, g[{i}]); ddf_put_{}(ddf, cc, nb, g[{}]); }}",
            odd_a.dir,
            odd_b.dir,
            i + 1,
            even.dir,
            even_b.dir,
            i + 1
        );
        s.push_str("    }\n");
        i += 2;
    }
    s.push_str("}\n");
    s
}

/// Indexing, flag unpacking and the sponge profile. Mirrors
/// `shaders/lbm/common.wgsl`.
const COMMON: &str = r#"
// Neighbour in a direction, wrapping on every axis.
//
// Wrapping unconditionally is both safe and free. On a padded axis the wrap can
// only ever trigger for a halo cell, and halo cells return before they get here;
// on a periodic axis it is the intended behaviour. The payoff is that no thread
// can compute an out-of-range index, so there is no bounds check on the hot path
// and no way for a boundary cell to read someone else's slot.
//
// The `((p % n) + n) % n` is kept even though a conditional add would be
// cheaper, because the WGSL does exactly this and the point of this backend is
// to run the same arithmetic.
__device__ __forceinline__ unsigned int neighbour_index(
        const Params& P, unsigned int cx, unsigned int cy, unsigned int cz,
        int dx, int dy, int dz) {
    const int nx = (int)P.dims_x, ny = (int)P.dims_y, nz = (int)P.dims_z;
    int px = (int)cx + dx;
    int py = (int)cy + dy;
    int pz = (int)cz + dz;
    px = ((px % nx) + nx) % nx;
    py = ((py % ny) + ny) % ny;
    pz = ((pz % nz) + nz) % nz;
    return (unsigned int)((pz * ny + py) * nx + px);
}

__device__ __forceinline__ unsigned int cell_index(
        const Params& P, unsigned int cx, unsigned int cy, unsigned int cz) {
    return (cz * P.dims_y + cy) * P.dims_x + cx;
}

__device__ __forceinline__ unsigned int interior_index(
        const Params& P, unsigned int cx, unsigned int cy, unsigned int cz) {
    return (cz * P.interior_y + cy) * P.interior_x + cx;
}

// One flag byte per cell, four per word. Keeping it a byte matters: the traffic
// model in ad_gpu::ddf budgets exactly 1 B/cell/step for it.
__device__ __forceinline__ unsigned int get_flags(
        const unsigned int* __restrict__ cell_flags, unsigned int cell) {
    return (cell_flags[cell >> 2] >> ((cell & 3u) * 8u)) & 0xffu;
}

__device__ __forceinline__ bool is_fluid(unsigned int f) {
    return (f & FLAG_SOLID) == 0u;
}

// Distance in cells to the nearest closed domain face, saturating at `limit`.
// Periodic axes have no face and are skipped.
__device__ __forceinline__ unsigned int face_distance(
        const Params& P, unsigned int cx, unsigned int cy, unsigned int cz,
        unsigned int limit) {
    const unsigned int n[3] = { P.dims_x, P.dims_y, P.dims_z };
    const unsigned int v[3] = { cx, cy, cz };
    unsigned int d = limit;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        if ((P.periodic & (1u << a)) != 0u) { continue; }
        const unsigned int lo = v[a];
        const unsigned int hi = n[a] - 1u - v[a];
        const unsigned int near = lo < hi ? lo : hi;
        d = near < d ? near : d;
    }
    return d;
}

// Quadratic sponge ramp: zero at the inner edge of the layer, full strength at
// the face. A step change in damping reflects nearly as much as the wall it
// replaced, which is the whole point of grading it.
//
// Gated on the SPONGE flag, not on distance alone. The distance is only the
// profile; the flag is the caller's statement of where the absorbing layer
// actually is. See ad_solver::boundary::PaddedDomain::sponge_sigma for the 12%
// flux loss that grading by distance alone cost.
__device__ __forceinline__ float sponge_sigma(
        const Params& P, unsigned int cx, unsigned int cy, unsigned int cz,
        unsigned int fl) {
    if ((fl & FLAG_SPONGE) == 0u || P.sponge_cells == 0u || P.sponge_strength <= 0.0f) {
        return 0.0f;
    }
    const unsigned int d = face_distance(P, cx, cy, cz, P.sponge_cells);
    if (d >= P.sponge_cells) { return 0.0f; }
    const float t = (float)(P.sponge_cells - d) / (float)P.sponge_cells;
    return P.sponge_strength * t * t;
}
"#;

/// Collision operators, the Smagorinsky closure and the moment sums. Mirrors
/// `shaders/lbm/collision.wgsl`, which mirrors `ad_solver::collision`.
const COLLISION: &str = r#"
#define CS2 0.3333333333333333f

// f_i^eq - w_i, built already-shifted.
//
// `drho` is passed separately from `rho` on purpose: the caller obtained it as a
// sum of small numbers, and recomputing it as `rho - 1.0` here would throw away
// exactly the precision the DDF shift was introduced to protect.
__device__ __forceinline__ float shifted_equilibrium(
        int i, float drho, float rho, float ux, float uy, float uz) {
    const float cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
    const float uu = ux * ux + uy * uy + uz * uz;
    return W[i] * (drho + rho * (3.0f * cu + 4.5f * cu * cu - 1.5f * uu));
}

// (s_e, s_o) for TRT. s_e = 1/tau fixes the viscosity; the magic parameter
// Lambda = (1/s_e - 1/2)(1/s_o - 1/2) fixes the free odd rate. Under BGK or
// regularized BGK the two rates coincide, which is how one code path serves all
// three operators.
__device__ __forceinline__ void trt_rates(const Params& P, float tau, float* s_e, float* s_o) {
    *s_e = 1.0f / tau;
#if COLLIDE_TRT
    *s_o = 1.0f / (P.trt_lambda / (tau - 0.5f) + 0.5f);
#else
    *s_o = *s_e;
#endif
}

// Non-equilibrium second moment, Pi_ab = sum_i c_ia c_ib (g_i - geq_i).
//
// This is why LES is cheap in LBM: the strain rate is available locally and
// algebraically, with no stencil. On a staircased voxel wall that is decisive -
// a finite-difference velocity gradient across the steps would manufacture
// strain, and with it a layer of artificial eddy viscosity exactly where the
// boundary layer is thinnest.
__device__ __forceinline__ void nonequilibrium_stress(
        const float* g, const float* geq, float pi[3][3]) {
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        #pragma unroll
        for (int b = 0; b < 3; ++b) { pi[a][b] = 0.0f; }
    }
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        const float n = g[i] - geq[i];
        const float c[3] = { CX[i], CY[i], CZ[i] };
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            #pragma unroll
            for (int b = 0; b < 3; ++b) { pi[a][b] += n * c[a] * c[b]; }
        }
    }
}

__device__ __forceinline__ float frobenius(const float pi[3][3]) {
    float s = 0.0f;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        #pragma unroll
        for (int b = 0; b < 3; ++b) { s += pi[a][b] * pi[a][b]; }
    }
    return sqrtf(s);
}

// tau_eff = 0.5 (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |Pi| / rho)), clamped.
//
// Cs is 0.10-0.12 rather than Lilly's 0.17: the theoretical constant assumes
// inertial-range isotropic turbulence, and near a wall it produces eddy
// viscosity where the real flow is laminar. Cs = 0 returns tau0 exactly, so the
// model switches off cleanly.
__device__ __forceinline__ float smagorinsky_tau(const Params& P, float rho, float pi_norm) {
    if (P.smagorinsky_c <= 0.0f) { return P.tau0; }
    const float inner = P.tau0 * P.tau0
        + 25.455845f * P.smagorinsky_c * P.smagorinsky_c * pi_norm / fmaxf(rho, 1e-6f);
    const float t = 0.5f * (P.tau0 + sqrtf(inner));
    return fminf(fmaxf(t, P.tau0), P.tau_max);
}

__device__ __forceinline__ void collide(float* g, const float* geq, float s_e, float s_o) {
#if COLLIDE_RBGK
    // Regularized BGK: rebuild f^neq from its second moment alone, discarding
    // every ghost moment rather than relaxing it. Strictly more dissipative than
    // TRT and only first-order at the wall, so this is a stability escape hatch,
    // not a default.
    float pi[3][3];
    nonequilibrium_stress(g, geq, pi);
    const float trace = pi[0][0] + pi[1][1] + pi[2][2];
    const float keep = 1.0f - s_e;
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        const float c[3] = { CX[i], CY[i], CZ[i] };
        float qpi = 0.0f;
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            #pragma unroll
            for (int b = 0; b < 3; ++b) { qpi += c[a] * c[b] * pi[a][b]; }
        }
        // Q_iab : Pi_ab with Q_iab = c_ia c_ib - c_s^2 delta_ab, times the
        // 1/(2 c_s^4) = 4.5 Hermite normalisation.
        g[i] = geq[i] + keep * W[i] * 4.5f * (qpi - CS2 * trace);
    }
#else
    // The rest population is purely symmetric, so it only ever sees s_e.
    g[0] = g[0] - s_e * (g[0] - geq[0]);
    #pragma unroll
    for (int i = 1; i < Q; i += 2) {
        const int j = i + 1;
        const float ni = g[i] - geq[i];
        const float nj = g[j] - geq[j];
        const float sym = 0.5f * (ni + nj);
        const float asym = 0.5f * (ni - nj);
        g[i] = g[i] - (s_e * sym + s_o * asym);
        g[j] = g[j] - (s_e * sym - s_o * asym);
    }
#endif
}

// Guo's forcing term, split for TRT.
//
// F_i = w_i [3 (c_i - u).F + 9 (c_i.u)(c_i.F)], applied as (1 - s/2) F_i with
// s_e on the symmetric part and s_o on the antisymmetric part. The velocity
// passed in must already carry the F/(2 rho) half-step correction; getting that
// wrong turns the Poiseuille profile first-order.
__device__ __forceinline__ void apply_force(
        const Params& P, float* g, float ux, float uy, float uz, float s_e, float s_o) {
    const float fx = P.body_force_x, fy = P.body_force_y, fz = P.body_force_z;
    if (fx == 0.0f && fy == 0.0f && fz == 0.0f) { return; }
    const float udotf = ux * fx + uy * fy + uz * fz;
    const float ke = 1.0f - 0.5f * s_e;
    const float ko = 1.0f - 0.5f * s_o;
    g[0] = g[0] + ke * W[0] * (3.0f * (0.0f - udotf));
    #pragma unroll
    for (int i = 1; i < Q; i += 2) {
        const int j = i + 1;
        const float ciu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
        const float cif = CX[i] * fx + CY[i] * fy + CZ[i] * fz;
        const float cju = CX[j] * ux + CY[j] * uy + CZ[j] * uz;
        const float cjf = CX[j] * fx + CY[j] * fy + CZ[j] * fz;
        const float fi = W[i] * (3.0f * (cif - udotf) + 9.0f * ciu * cif);
        const float fj = W[j] * (3.0f * (cjf - udotf) + 9.0f * cju * cjf);
        const float sym = 0.5f * (fi + fj);
        const float asym = 0.5f * (fi - fj);
        g[i] = g[i] + ke * sym + ko * asym;
        g[j] = g[j] + ke * sym - ko * asym;
    }
}

// rho - 1 and rho*u, in the order the DDF shift demands: the small deviations
// are summed first and the 1 is added exactly once, at the end. Summing f_i
// directly would put q numbers of order w_i through an addition whose
// interesting part is the 1e-4 deviation from unity.
__device__ __forceinline__ void moments(
        const float* g, float* drho, float* mx, float* my, float* mz) {
    float d = 0.0f, x = 0.0f, y = 0.0f, z = 0.0f;
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        const float v = g[i];
        d += v;
        x += v * CX[i];
        y += v * CY[i];
        z += v * CZ[i];
    }
    *drho = d; *mx = x; *my = y; *mz = z;
}

__device__ __forceinline__ void fill_equilibrium(
        float* geq, float drho, float rho, float ux, float uy, float uz) {
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        geq[i] = shifted_equilibrium(i, drho, rho, ux, uy, uz);
    }
}
"#;

/// Everything that is not a no-slip wall. Mirrors `shaders/lbm/boundary.wgsl`.
///
/// Solid walls do not appear: halfway bounce-back is implicit in the Esoteric
/// Pull load parity flip, so there is no separate boundary kernel and no extra
/// traffic.
const BOUNDARY: &str = r#"
// Is link `i` at cell (cx,cy,cz) a genuine outflow link?
//
// Two conditions, and both matter. The direction must point back into the domain
// against the outlet normal (so it is an *unknown* population arriving from
// outside), and the upstream cell must leave the domain **only** along the
// normal axis. The second is what keeps a corner sane: several anti-bounce-back
// reflections meeting at one cell is an amplifying map, and the production duct
// test NaNs at the outlet corner within 1500-4000 steps without it.
__device__ __forceinline__ bool is_outflow_link(
        const Params& P, unsigned int cx, unsigned int cy, unsigned int cz, int i) {
    const float nx = P.outlet_normal_x, ny = P.outlet_normal_y, nz = P.outlet_normal_z;
    if (nx == 0.0f && ny == 0.0f && nz == 0.0f) { return true; }
    if (CX[i] * nx + CY[i] * ny + CZ[i] * nz >= 0.0f) { return false; }
    const int up[3] = { (int)cx - (int)CX[i], (int)cy - (int)CY[i], (int)cz - (int)CZ[i] };
    const int lo[3] = { (int)P.offset_x, (int)P.offset_y, (int)P.offset_z };
    const int hi[3] = { lo[0] + (int)P.interior_x - 1,
                        lo[1] + (int)P.interior_y - 1,
                        lo[2] + (int)P.interior_z - 1 };
    const float n[3] = { nx, ny, nz };
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        if (fabsf(n[a]) > 0.5f) { continue; }   // the normal axis may be outside
        if (up[a] < lo[a] || up[a] > hi[a]) { return false; }
    }
    return true;
}

// Anti-bounce-back: replace the populations arriving on blocked links with the
// value that fixes the *density* at rho_ref instead of the velocity at zero.
//
//   g_i = -g_loaded + 2 w_i [(rho_ref - 1) + rho_ref ((9/2)(c.u)^2 - (3/2) u^2)]
//
// `g_loaded` on a blocked link is exactly f_ibar^post - w, which is why the
// bounce-back value can be reused directly rather than re-read.
__device__ __forceinline__ void anti_bounce_back(
        const Params& P, unsigned int cx, unsigned int cy, unsigned int cz,
        unsigned int mask, float* g, float ux, float uy, float uz) {
    const float uu = ux * ux + uy * uy + uz * uz;
    const float rho_ref = P.rho_ref;
    #pragma unroll
    for (int i = 1; i < Q; ++i) {
        if ((mask & (1u << i)) == 0u) { continue; }
        if (!is_outflow_link(P, cx, cy, cz, i)) { continue; }
        const float cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
        const float abb = 2.0f * W[i] * ((rho_ref - 1.0f) + rho_ref * (4.5f * cu * cu - 1.5f * uu));
        g[i] = -g[i] + abb;
    }
}

// Density at a velocity inlet, from the populations that are actually known.
//
// The raw moment is not usable: an inlet plane sits against the edge of the
// domain, so the populations arriving from "outside" are really this cell's own
// outgoing populations bounced off the halo, carrying -u instead of +u. The
// density then reads low by u_n every step; measured fixed point at u_n = 0.05
// was rho = 0.864, with the duct downstream completely stagnant.
//
// With `n` the inward normal the unknowns are exactly the links with c_i.n > 0,
// and rho = (A + 2B)/(1 - u_n). In shifted form the weight sums cancel exactly
// (W_0 + 2 W_- = 1 for an axis-aligned normal), leaving
// drho = (A_g + 2 B_g + u_n)/(1 - u_n) with no unshifted quantity anywhere.
//
// Clamped because the closure assumes a flat plane, and at the edge of a narrow
// inlet patch some "known" links are blocked by the duct wall instead. The band
// is far wider than any duct pressure this solver will see, so it never binds on
// a well-posed problem and always binds on a broken one.
__device__ __forceinline__ float inlet_drho(
        const Params& P, const float* g, float nx, float ny, float nz,
        float ux, float uy, float uz) {
    float a = 0.0f, b = 0.0f;
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        const float cn = CX[i] * nx + CY[i] * ny + CZ[i] * nz;
        if (cn == 0.0f) { a += g[i]; }
        else if (cn < 0.0f) { b += g[i]; }
    }
    const float un = ux * nx + uy * ny + uz * nz;
    const float drho = (a + 2.0f * b + un) / fmaxf(1.0f - un, 0.1f);
    const float r = P.rho_ref - 1.0f;
    return fminf(fmaxf(drho, r - 0.2f), r + 0.2f);
}

// Relax the whole population set toward a target equilibrium by `weight`. Used
// by the convective outlet and by the sponge; both are "absorb, do not reflect"
// treatments and differ only in what they aim at and how hard.
__device__ __forceinline__ void relax_toward_equilibrium(
        float* g, float weight, float drho, float rho, float ux, float uy, float uz) {
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        g[i] = g[i] + weight * (shifted_equilibrium(i, drho, rho, ux, uy, uz) - g[i]);
    }
}
"#;

/// The three entry points. Mirrors `shaders/lbm/stream_collide.wgsl` and
/// `shaders/lbm/macroscopic.wgsl`.
///
/// The grid mapping is the WGSL's exactly: block `(BLOCK_X, 1, 1)`, grid
/// `(ceil(dims.x / BLOCK_X), dims.y, dims.z)`, so `global_invocation_id`
/// translates to `blockIdx.x * BLOCK_X + threadIdx.x, blockIdx.y, blockIdx.z`.
/// X is the fastest-varying axis, so a block covers a contiguous run of cells
/// and every direction plane is read with consecutive lanes touching consecutive
/// addresses — the coalescing optimum.
const ENTRY_POINTS: &str = r#"
// Initialise every fluid cell to equilibrium at (1, initial_velocity).
//
// The store uses the parity of the step *before* step 0, which is the only
// choice that also populates wall links: on a blocked link the flipped load of
// step 0 lands precisely on the slot this write fills. Any other convention
// leaves the first step reading uninitialised memory on every boundary link,
// which shows up as a boundary layer of garbage that then diffuses inward.
extern "C" __global__ __launch_bounds__(BLOCK_X) void lbm_init(
        const Params P, float* ddf,
        const unsigned int* __restrict__ cell_flags) {
    const unsigned int cx = blockIdx.x * BLOCK_X + threadIdx.x;
    const unsigned int cy = blockIdx.y;
    const unsigned int cz = blockIdx.z;
    if (cx >= P.dims_x || cy >= P.dims_y || cz >= P.dims_z) { return; }
    const unsigned int cell = cell_index(P, cx, cy, cz);
    if (!is_fluid(get_flags(cell_flags, cell))) { return; }

    float g[Q];
    fill_equilibrium(g, 0.0f, 1.0f, P.initial_velocity_x, P.initial_velocity_y,
                     P.initial_velocity_z);
    store_ddf(P, ddf, cell, cx, cy, cz, P.step_parity == 0u, g);
}

extern "C" __global__ __launch_bounds__(BLOCK_X) void lbm_stream_collide(
        const Params P, float* ddf,
        const unsigned int* __restrict__ cell_flags,
        const unsigned int* __restrict__ link_mask) {
    const unsigned int cx = blockIdx.x * BLOCK_X + threadIdx.x;
    const unsigned int cy = blockIdx.y;
    const unsigned int cz = blockIdx.z;
    if (cx >= P.dims_x || cy >= P.dims_y || cz >= P.dims_z) { return; }
    const unsigned int cell = cell_index(P, cx, cy, cz);
    const unsigned int fl = get_flags(cell_flags, cell);
    // Solid cells never execute. They own no populations: both slots on a wall
    // link belong to the fluid cell across it.
    if (!is_fluid(fl)) { return; }

    const bool odd = P.step_parity != 0u;
    const unsigned int mask = link_mask[cell];

    float g[Q];
    load_ddf(P, ddf, cell, cx, cy, cz, odd, mask, g);

    float drho, mx, my, mz;
    moments(g, &drho, &mx, &my, &mz);

    const bool is_inlet = (fl & FLAG_INLET) != 0u;
    const bool is_outlet = !is_inlet && (fl & FLAG_OUTLET) != 0u;
    const bool is_equil = !is_inlet && !is_outlet && (fl & FLAG_EQUILIBRIUM) != 0u;

    if (is_outlet && P.outlet_anti_bounce_back != 0u) {
        // Anti-bounce-back needs the wall velocity, which is not known until the
        // moments are in. Predict with the plain bounce-back result and correct
        // once; the correction to the velocity term is O(Ma^2) of an already
        // small quantity, so a second pass buys nothing.
        //
        // OFF BY DEFAULT. See SolverConfig::outlet_anti_bounce_back: this is
        // unstable under real through-flow, and the convective term below pins
        // the outlet density just as effectively without negating anything.
        const float rho_p = 1.0f + drho;
        anti_bounce_back(P, cx, cy, cz, mask, g, mx / rho_p, my / rho_p, mz / rho_p);
        moments(g, &drho, &mx, &my, &mz);
    }

    float rho = 1.0f + drho;
    float ux = (mx + 0.5f * P.body_force_x) / rho;
    float uy = (my + 0.5f * P.body_force_y) / rho;
    float uz = (mz + 0.5f * P.body_force_z) / rho;

    if (is_inlet) {
        // Equilibrium inlet: prescribe the velocity and let the density float,
        // so the inlet pressure is free and a duct pressure drop is measurable.
        // The density comes from the *known* populations only. The boundary has
        // no mechanism by which to diverge, which is why it is the v1 inlet.
        ux = P.inlet_velocity_x;
        uy = P.inlet_velocity_y;
        uz = P.inlet_velocity_z;
        drho = inlet_drho(P, g, P.inlet_normal_x, P.inlet_normal_y, P.inlet_normal_z,
                          ux, uy, uz);
        rho = 1.0f + drho;
    } else if (is_equil) {
        // Open box side. Ambient pressure and no mean flow, so the exit jet can
        // entrain surrounding air instead of being confined by a wall.
        rho = P.rho_ref;
        drho = P.rho_ref - 1.0f;
        ux = 0.0f; uy = 0.0f; uz = 0.0f;
    }

    float geq[Q];
    fill_equilibrium(geq, drho, rho, ux, uy, uz);

    float pi[3][3];
    nonequilibrium_stress(g, geq, pi);
    const float tau = smagorinsky_tau(P, rho, frobenius(pi));
    float s_e, s_o;
    trt_rates(P, tau, &s_e, &s_o);

    if (is_inlet || is_equil) {
        // Discard the non-equilibrium entirely.
        #pragma unroll
        for (int i = 0; i < Q; ++i) { g[i] = geq[i]; }
    } else {
        collide(g, geq, s_e, s_o);
        apply_force(P, g, ux, uy, uz, s_e, s_o);
    }

    if (is_outlet && P.outflow_velocity > 0.0f) {
        // Convective outflow, df/dt + U df/dn = 0, discretised upwind with the
        // upstream population approximated by the local equilibrium at the
        // reference density. The weight U/(1+U) is the exact factor from
        // f(t+1) = (f + U f_up) / (1 + U).
        const float w = P.outflow_velocity / (1.0f + P.outflow_velocity);
        relax_toward_equilibrium(g, w, P.rho_ref - 1.0f, P.rho_ref, ux, uy, uz);
    }

    const float sigma = sponge_sigma(P, cx, cy, cz, fl);
    if (sigma > 0.0f) {
        // Absorb rather than reflect: relax toward equilibrium at the reference
        // density but the *local* velocity, so the layer eats acoustic and
        // vortical content without imposing a mean flow of its own.
        relax_toward_equilibrium(g, sigma, P.rho_ref - 1.0f, P.rho_ref, ux, uy, uz);
    }

    store_ddf(P, ddf, cell, cx, cy, cz, odd, g);
}

// Export rho and u as one float4 per *interior* cell, X-fastest.
//
// The wgpu path writes two 3D textures here and a plain buffer only when
// SolverConfig::macroscopic_buffer is on. Headless has no consumer for the
// textures, so this backend writes only the buffer - which is also the exact
// format `Solver::read_macroscopic` returns, so the two are directly comparable.
//
// Runs at the same step parity the *next* stream-collide would load with, so it
// sees the current state without disturbing it.
extern "C" __global__ __launch_bounds__(BLOCK_X) void lbm_macroscopic(
        const Params P, const float* __restrict__ ddf,
        const unsigned int* __restrict__ cell_flags,
        const unsigned int* __restrict__ link_mask,
        float* __restrict__ out) {
    const unsigned int gx = blockIdx.x * BLOCK_X + threadIdx.x;
    const unsigned int gy = blockIdx.y;
    const unsigned int gz = blockIdx.z;
    if (gx >= P.interior_x || gy >= P.interior_y || gz >= P.interior_z) { return; }
    const unsigned int cx = gx + P.offset_x;
    const unsigned int cy = gy + P.offset_y;
    const unsigned int cz = gz + P.offset_z;
    const unsigned int cell = cell_index(P, cx, cy, cz);

    const unsigned int fl = get_flags(cell_flags, cell);
    float rho = 1.0f;
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (is_fluid(fl)) {
        float g[Q];
        load_ddf(P, ddf, cell, cx, cy, cz, P.step_parity != 0u, link_mask[cell], g);
        float drho, mx, my, mz;
        moments(g, &drho, &mx, &my, &mz);
        rho = 1.0f + drho;
        ux = (mx + 0.5f * P.body_force_x) / rho;
        uy = (my + 0.5f * P.body_force_y) / rho;
        uz = (mz + 0.5f * P.body_force_z) / rho;

        // Report what an equilibrium boundary actually holds, not the moment of
        // the incoming populations. At an inlet the cell's state after the step
        // *is* f^eq(rho, u_inlet); its pre-collision moment can have the wrong
        // sign entirely. ad_solver::reference::macroscopic_padded and
        // shaders/lbm/macroscopic.wgsl apply the same rule, which is what makes
        // the three fields comparable.
        if ((fl & FLAG_INLET) != 0u) {
            ux = P.inlet_velocity_x; uy = P.inlet_velocity_y; uz = P.inlet_velocity_z;
        } else if ((fl & FLAG_EQUILIBRIUM) != 0u) {
            rho = P.rho_ref;
            ux = 0.0f; uy = 0.0f; uz = 0.0f;
        }
    }

    const unsigned int o = interior_index(P, gx, gy, gz) * 4u;
    out[o + 0u] = ux;
    out[o + 1u] = uy;
    out[o + 2u] = uz;
    out[o + 3u] = rho;
}
"#;

/// The complete CUDA C translation unit for one [`KernelSpec`].
///
/// Handed straight to NVRTC. Deterministic: the same spec always produces the
/// same bytes, which is what makes it usable as a compilation cache key.
pub fn generate(spec: KernelSpec) -> String {
    let mut s = String::with_capacity(48 * 1024);
    s.push_str(
        "// GENERATED by ad_cuda::kernel - do not edit.\n\
         //\n\
         // Lattice tables from ad_gpu::lattice, flag bits from ad_gpu::types::flags,\n\
         // and the Esoteric Pull transport from ad_solver::shaders::probe - the same\n\
         // call that emits the WGSL. Nothing here is transcribed by hand.\n\n",
    );
    let _ = writeln!(s, "#define BLOCK_X {}", spec.block_x.max(1));
    // One operator selected, the other two compiled out, exactly as the WGSL's
    // `#if COLLIDE_*` does. Defined to 0 rather than left undefined so `#if`
    // never depends on the preprocessor's treatment of unknown identifiers.
    for m in [CollisionModel::Trt, CollisionModel::Bgk, CollisionModel::RegularizedBgk] {
        let _ = writeln!(
            s,
            "#define {} {}",
            m.shader_define(),
            u32::from(m == spec.collision)
        );
    }
    s.push('\n');
    s.push_str(&flag_constants());
    s.push('\n');
    s.push_str(&tables(spec.set));
    s.push('\n');
    s.push_str(&param_struct());
    s.push_str(COMMON);
    s.push('\n');
    s.push_str(&accessors(spec.set));
    s.push('\n');
    s.push_str(&transport(spec.set));
    s.push_str(COLLISION);
    s.push_str(BOUNDARY);
    s.push_str(ENTRY_POINTS);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::lattice::{opposite, EsotericPull};

    fn all_specs() -> Vec<KernelSpec> {
        let mut v = Vec::new();
        for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
            for collision in
                [CollisionModel::Trt, CollisionModel::Bgk, CollisionModel::RegularizedBgk]
            {
                for block_x in [64u32, 128] {
                    v.push(KernelSpec { set, collision, block_x });
                }
            }
        }
        v
    }

    /// The Rust parameter block and the generated C struct must agree byte for
    /// byte. A mismatch is not a compile error on either side — it is a kernel
    /// reading `tau0` out of `trt_lambda` and producing a plausible wrong
    /// answer, which is why this is checked field by field rather than by size
    /// alone.
    #[test]
    fn param_struct_matches_the_generated_c_struct() {
        macro_rules! check {
            ($($f:ident),* $(,)?) => {{
                let mut n = 0usize;
                $(
                    let (ty, name) = PARAM_FIELDS[n];
                    assert_eq!(name, stringify!($f), "field {n} is out of order");
                    assert_eq!(
                        std::mem::offset_of!(CudaParams, $f),
                        n * 4,
                        "field {} is not at word {n}", stringify!($f)
                    );
                    assert!(ty == "float" || ty == "unsigned int", "field {name} has type {ty}");
                    n += 1;
                )*
                assert_eq!(n, PARAM_FIELDS.len(), "PARAM_FIELDS has entries the struct does not");
            }};
        }
        check!(
            dims_x, dims_y, dims_z, step_parity,
            interior_x, interior_y, interior_z, cell_count,
            offset_x, offset_y, offset_z, flag_words,
            inlet_velocity_x, inlet_velocity_y, inlet_velocity_z, tau0,
            initial_velocity_x, initial_velocity_y, initial_velocity_z, trt_lambda,
            body_force_x, body_force_y, body_force_z, smagorinsky_c,
            tau_max, outflow_velocity, rho_ref, sponge_strength,
            sponge_cells, periodic, total_steps, outlet_anti_bounce_back,
            outlet_normal_x, outlet_normal_y, outlet_normal_z,
            inlet_normal_x, inlet_normal_y, inlet_normal_z,
        );
        // ...and the C side really does declare each of them, in order.
        let src = generate(KernelSpec {
            set: VelocitySet::D3Q19,
            collision: CollisionModel::Trt,
            block_x: 64,
        });
        let mut cursor = src.find("struct Params {").expect("no Params struct emitted");
        for (ty, name) in PARAM_FIELDS {
            let decl = format!("    {ty} {name};");
            let at = src[cursor..]
                .find(&decl)
                .unwrap_or_else(|| panic!("`{decl}` missing or out of order"));
            cursor += at + decl.len();
        }
    }

    /// The generated transport must address exactly what [`EsotericPull`] says,
    /// checked independently of the generator: for every direction and parity,
    /// find the emitted accessor and compare its buffer index against a fresh
    /// call to the real Rust implementation.
    ///
    /// This is the test that would catch a generator bug the WGSL does not have,
    /// so it deliberately does not reuse `probe` — it goes to `EsotericPull`.
    #[test]
    fn generated_transport_addresses_agree_with_esoteric_pull() {
        for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
            let q = set.q();
            let src = transport(set);
            let (load, store) = src.split_at(src.find("void store_ddf").expect("no store body"));

            // A 6x6x6 periodic lattice, so `slot` is decodable and every link is
            // exercised. Cell 0's neighbour in direction d is a distinct index.
            const N: u64 = 6;
            let cells = N * N * N;
            let ep = EsotericPull::new(cells, q);

            let mut m = 1usize;
            while m < q {
                assert_eq!(opposite(m), m + 1, "pairing broke");
                for odd in [false, true] {
                    for d in [m, m + 1] {
                        // The buffer the load for direction `d` must use.
                        let want_load = ep.load_slot(0, d, odd, |_| 1) / cells;
                        let want_store = ep.store_slot(0, d, odd, |_| 1) / cells;
                        // Emitted form: `g[d] = ddf_get_N(...)` on the odd-parity
                        // arm first, even second (see `transport`).
                        let arm = if odd { 0 } else { 1 };
                        let got_load = nth_accessor(load, &format!("g[{d}] = ddf_get_"), arm);
                        assert_eq!(
                            got_load, want_load as usize,
                            "{set:?} load dir {d} odd={odd}: emitted buffer {got_load}, \
                             EsotericPull says {want_load}"
                        );
                        let got_store = nth_put_for(store, m, d, odd);
                        assert_eq!(
                            got_store, want_store as usize,
                            "{set:?} store dir {d} odd={odd}: emitted buffer {got_store}, \
                             EsotericPull says {want_store}"
                        );
                    }
                }
                m += 2;
            }
        }
    }

    /// The buffer index in the `n`th occurrence of `needle` followed by digits.
    fn nth_accessor(hay: &str, needle: &str, n: usize) -> usize {
        let mut from = 0usize;
        for _ in 0..=n {
            let at = hay[from..]
                .find(needle)
                .unwrap_or_else(|| panic!("`{needle}` occurrence {n} not found"));
            from += at + needle.len();
        }
        let digits: String = hay[from..].chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().expect("accessor index is not a number")
    }

    /// The store body emits one line per pair holding four `ddf_put_N` calls:
    /// odd-parity local, odd-parity neighbour, then even-parity local, even
    /// neighbour. Pick the one for `(pair m, direction d, parity odd)`.
    fn nth_put_for(store: &str, m: usize, d: usize, odd: bool) -> usize {
        let block = store
            .split(&format!("{{ // pair ({m}, {})", m + 1))
            .nth(1)
            .unwrap_or_else(|| panic!("pair ({m}, {}) block missing", m + 1));
        let index = usize::from(!odd) * 2 + usize::from(d != m);
        nth_accessor(block, "ddf_put_", index)
    }

    /// Every direction buffer must be *written* exactly as many times as it is
    /// read, or some slot is being dropped. Same invariant the WGSL generator
    /// checks, and it catches a whole class of copy-paste error.
    #[test]
    fn transport_reads_and_writes_each_buffer_the_same_number_of_times() {
        for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
            let s = transport(set);
            for i in 0..set.q() {
                let gets = s.matches(&format!("ddf_get_{i}(")).count();
                let puts = s.matches(&format!("ddf_put_{i}(")).count();
                assert_eq!(gets, puts, "{set:?}: buffer {i} read {gets} times, written {puts}");
            }
        }
    }

    #[test]
    fn every_direction_and_accessor_appears_for_every_spec() {
        for spec in all_specs() {
            let s = generate(spec);
            let q = spec.set.q();
            for i in 0..q {
                assert!(s.contains(&format!("ddf_get_{i}(")), "{spec:?}: no get for {i}");
                assert!(s.contains(&format!("ddf_put_{i}(")), "{spec:?}: no put for {i}");
                assert!(s.contains(&format!("g[{i}]")), "{spec:?}: direction {i} never transported");
            }
            assert!(s.contains(&format!("#define Q {q}")), "{spec:?}: wrong q");
            assert!(s.contains(&format!("#define BLOCK_X {}", spec.block_x)));
            for name in [FN_INIT, FN_STREAM_COLLIDE, FN_MACROSCOPIC] {
                assert!(
                    s.contains(&format!("void {name}(")),
                    "{spec:?}: entry point {name} missing"
                );
            }
        }
    }

    /// Exactly one collision operator is compiled in, and it is the requested
    /// one. If two were live the `#if` chain would silently pick whichever the
    /// preprocessor saw first.
    #[test]
    fn exactly_one_collision_operator_is_enabled() {
        for spec in all_specs() {
            let s = generate(spec);
            for m in [CollisionModel::Trt, CollisionModel::Bgk, CollisionModel::RegularizedBgk] {
                let want = u32::from(m == spec.collision);
                assert!(
                    s.contains(&format!("#define {} {want}", m.shader_define())),
                    "{spec:?}: {} should be {want}",
                    m.name()
                );
            }
        }
    }

    /// The tables must be the ad-gpu ones, not a copy. Checking the weights to
    /// full precision is the cheap way to prove nobody retyped 1/36 as 0.0278.
    #[test]
    fn tables_come_from_ad_gpu_lattice() {
        for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
            let s = tables(set);
            let def = set.def();
            for w in def.weights {
                assert!(
                    s.contains(&format!("{w:.17}f")),
                    "{set:?}: weight {w} is not in the emitted table"
                );
            }
            // Direction components, counted: q entries in each of three arrays.
            for name in ["CX", "CY", "CZ"] {
                let line = s
                    .lines()
                    .find(|l| l.starts_with(&format!("__constant__ float {name}[Q]")))
                    .unwrap_or_else(|| panic!("{name} table missing"));
                assert_eq!(line.matches(", ").count(), def.q - 1, "{name} has the wrong length");
            }
        }
    }

    /// Every loop over the population array has to be unrolled or the array
    /// spills from registers to local memory, and a bandwidth-bound kernel that
    /// spills is not measuring what it claims to. Cheap structural check: the
    /// number of `for (int i` loops and the number of `#pragma unroll` must
    /// match up.
    #[test]
    fn every_population_loop_is_unrolled() {
        let s = generate(KernelSpec {
            set: VelocitySet::D3Q19,
            collision: CollisionModel::Trt,
            block_x: 64,
        });
        let lines: Vec<&str> = s.lines().map(|l| l.trim()).collect();
        for (n, l) in lines.iter().enumerate() {
            if l.starts_with("for (int i = 0; i < Q") || l.starts_with("for (int i = 1; i < Q") {
                assert_eq!(
                    lines[n - 1],
                    "#pragma unroll",
                    "population loop on line {n} is not unrolled: {l}"
                );
            }
        }
    }
}

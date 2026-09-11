// Plane integrals over a FlowPatch.
//
// A 512x512 midpoint quadrature over the patch, trilinearly sampling the
// macroscopic fields, reduced to one small record per patch.
//
// # Why fp32 accumulation, always
//
// 262,144 samples of a lattice velocity around 0.05 sum to about 1.3e4. In fp16
// the representable spacing at that magnitude is 8, so once the partial sum
// passes a couple of thousand every further sample rounds to a multiple of 8 and
// the integral stops moving. That is not noise that averages out; it is a
// systematic truncation toward the running total, and it visibly biases the flow
// rate. The per-thread values, the workgroup reduction and the global
// accumulator are therefore all fp32.
//
// # The two-level reduction
//
// The launch is a fixed number of workgroups (REDUCE_GROUPS_PER_SIDE squared,
// in plane.rs) whatever the sample count, and each thread strides across the
// sample grid by the whole launch, folding into its own partial slots. Each
// workgroup of 64 threads then folds those in workgroup memory (a halving tree,
// six barriers) and performs exactly ONE atomic update per accumulator. The
// global buffer sees 256 atomic updates per scalar - which matters, because
// WGSL has no native f32 atomic add and each one is a compare-exchange loop
// whose cost grows with contention. The first version ran one workgroup per 64
// samples: 4096 contended loops per scalar, per plane, per pass. That is the
// same failure that took the whole-volume reduction in volume.wgsl to 180 ms a
// frame at dx = 0.75 mm, at a smaller scale.
//
// # The two passes
//
// Weltens' uniformity index needs the plane mean before it can accumulate
// |w_i - w_bar|, and the momentum cone needs the mean momentum direction before
// it can bin angles about it. Neither is exactly recoverable from moments, so
// the same kernel runs twice over the same sample grid:
//
//   pass 0  counts, sums, momentum vector, extrema
//   pass 1  sum |w - w_bar|, velocity histogram, momentum-angle histogram
//
// WebGPU orders dispatches within a compute pass and inserts the storage barrier
// between them, so pass 1 reads what pass 0 wrote with no extra synchronisation.
// The second pass costs another 262k texture fetches, which is microseconds.

#include "metrics/common.wgsl"

struct PlaneUniforms {
    grid: GridInfo,
    center_mm: vec3<f32>,
    pass_index: u32,
    half_u: vec3<f32>,
    slot: u32,
    half_v: vec3<f32>,
    samples: u32,
    normal: vec3<f32>,
    pad: f32,
};

@group(1) @binding(0) var<uniform> P: PlaneUniforms;
// One flat array of atomics rather than an array of structs: WGSL has no f32
// atomics, so every field is a u32 holding either a count or the bit pattern of
// an f32, and a flat layout keeps the Rust mirror (ad_metrics::plane::PlaneAccum)
// a byte-for-byte match with generated index constants.
@group(1) @binding(1) var<storage, read_write> acc: array<atomic<u32>>;

const WG: u32 = 64u;
// Workgroup partial slots: four counts followed by the scalars, in the same
// order as the global record.
const SLOTS: u32 = 4u + N_SCALARS;
const WG_SLOTS: u32 = WG * SLOTS;
// Slots [0, K_SUM_END) fold by addition; the last three fold by max/max/min.
const K_SUM_END: u32 = 4u + S_MAX_SPEED;
const K_MAX_SPEED: u32 = 4u + S_MAX_SPEED;
const K_MAX_W: u32 = 4u + S_MAX_W;
const K_MIN_W: u32 = 4u + S_MIN_W;
const_assert SLOTS == 20u;
const_assert K_MIN_W == SLOTS - 1u;

var<workgroup> part: array<f32, WG_SLOTS>;
var<workgroup> hist_local: array<atomic<u32>, HIST_BINS>;
var<workgroup> angle_local: array<atomic<u32>, ANGLE_BINS>;

// ---------------------------------------------------------------------------
// f32 atomics, built from compare-exchange.
//
// WGSL provides atomics only on u32/i32. Accumulating in fixed point instead
// would trade the fp16 bias this file exists to avoid for a quantisation bias,
// so the sums go through a CAS loop on the bit pattern. With one update per
// workgroup and a few hundred workgroups, the loop rarely spins for long.
// ---------------------------------------------------------------------------

fn atomic_add_f32(i: u32, v: f32) {
    if (v == 0.0) { return; }
    var old = atomicLoad(&acc[i]);
    loop {
        let sum = bitcast<f32>(old) + v;
        let r = atomicCompareExchangeWeak(&acc[i], old, bitcast<u32>(sum));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}

fn atomic_max_f32(i: u32, v: f32) {
    var old = atomicLoad(&acc[i]);
    loop {
        if (bitcast<f32>(old) >= v) { break; }
        let r = atomicCompareExchangeWeak(&acc[i], old, bitcast<u32>(v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}

fn atomic_min_f32(i: u32, v: f32) {
    var old = atomicLoad(&acc[i]);
    loop {
        if (bitcast<f32>(old) <= v) { break; }
        let r = atomicCompareExchangeWeak(&acc[i], old, bitcast<u32>(v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}

fn read_scalar(base: u32, s: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&acc[base + A_SCALAR_BASE + s]));
}

// Fold `v` into this thread's own partial slot `k`. Only the owning thread
// touches a slot before the reduction barrier, so no atomics are needed.
fn add_part(k: u32, tid: u32, v: f32) {
    part[k * WG + tid] = part[k * WG + tid] + v;
}

@compute @workgroup_size(8, 8, 1)
fn plane(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let base = P.slot * ACC_STRIDE;

    // Clear the workgroup partials. Every thread reaches every barrier below,
    // including threads whose sample falls outside the grid, because a
    // workgroupBarrier in non-uniform control flow is undefined behaviour.
    for (var k = 0u; k < SLOTS; k = k + 1u) {
        part[k * WG + tid] = 0.0;
    }
    part[K_MAX_SPEED * WG + tid] = -1.0e30;
    part[K_MAX_W * WG + tid] = -1.0e30;
    part[K_MIN_W * WG + tid] = 1.0e30;
    if (P.pass_index == 1u) {
        for (var b = tid; b < HIST_BINS; b = b + WG) {
            atomicStore(&hist_local[b], 0u);
        }
        for (var b = tid; b < ANGLE_BINS; b = b + WG) {
            atomicStore(&angle_local[b], 0u);
        }
    }
    workgroupBarrier();

    // What pass 1 bins against, from pass 0's totals: read once per thread
    // rather than once per sample.
    var w_bar = 0.0;
    var lo = 0.0;
    var span = 1.0;
    var j = vec3<f32>(0.0, 0.0, 0.0);
    var jm = 0.0;
    var total_flux = 0.0;
    if (P.pass_index == 1u) {
        let n_fluid = f32(atomicLoad(&acc[base + A_N_FLUID]));
        w_bar = select(0.0, read_scalar(base, S_W) / n_fluid, n_fluid > 0.0);
        lo = read_scalar(base, S_MIN_W);
        span = max(read_scalar(base, S_MAX_W) - lo, 1.0e-20);
        j = vec3<f32>(read_scalar(base, S_JX), read_scalar(base, S_JY), read_scalar(base, S_JZ));
        jm = length(j);
        total_flux = read_scalar(base, S_MOMFLUX);
    }

    // Each thread strides over the sample grid by the whole launch and folds
    // every sample it visits into its own partial slots; see "The two-level
    // reduction" above.
    let inv = 1.0 / f32(P.samples);
    let reach = nwg.xy * 8u;
    for (var iy = gid.y; iy < P.samples; iy = iy + reach.y) {
        for (var ix = gid.x; ix < P.samples; ix = ix + reach.x) {
            // Cell-centred parametric coordinates: s_i = -1 + 2(i + 1/2)/N. The
            // midpoint rule this gives is exact for any field linear in (s, t)
            // and has no bias toward either edge of the patch, which an
            // endpoint rule would.
            let s = 2.0 * (f32(ix) + 0.5) * inv - 1.0;
            let t = 2.0 * (f32(iy) + 0.5) * inv - 1.0;
            let fs = sample_field(P.grid, P.center_mm + P.half_u * s + P.half_v * t);
            let w = dot(fs.u, P.normal);
            let speed = length(fs.u);

            if (P.pass_index == 0u) {
                add_part(0u, tid, 1.0);
                add_part(1u, tid, select(0.0, 1.0, fs.inside));
                if (fs.fluid) {
                    let pt = total_pressure_lb(fs.rho, fs.u);
                    add_part(2u, tid, 1.0);
                    add_part(3u, tid, select(0.0, 1.0, w < 0.0));
                    add_part(4u + S_W, tid, w);
                    add_part(4u + S_RHO_W, tid, fs.rho * w);
                    add_part(4u + S_PS, tid, static_pressure_lb(fs.rho));
                    add_part(4u + S_PT, tid, pt);
                    add_part(4u + S_MDOT_PT, tid, fs.rho * w * pt);
                    add_part(4u + S_W2, tid, w * w);
                    add_part(4u + S_SPEED, tid, speed);
                    add_part(4u + S_ABS_W, tid, abs(w));
                    add_part(4u + S_JX, tid, fs.rho * fs.u.x * w);
                    add_part(4u + S_JY, tid, fs.rho * fs.u.y * w);
                    add_part(4u + S_JZ, tid, fs.rho * fs.u.z * w);
                    // Forward momentum flux magnitude only, so a slow reversed
                    // corner does not vote in the jet direction; see the cone
                    // estimate below.
                    add_part(4u + S_MOMFLUX, tid, fs.rho * speed * max(w, 0.0));
                    part[K_MAX_SPEED * WG + tid] = max(part[K_MAX_SPEED * WG + tid], speed);
                    part[K_MAX_W * WG + tid] = max(part[K_MAX_W * WG + tid], w);
                    part[K_MIN_W * WG + tid] = min(part[K_MIN_W * WG + tid], w);
                }
            } else if (fs.fluid) {
                add_part(4u + S_ABS_DEV, tid, abs(w - w_bar));

                // Velocity histogram over the through-plane component, binned
                // across the range pass 0 measured. Counts are exact u32.
                let b = clamp(u32(max((w - lo) / span, 0.0) * f32(HIST_BINS)), 0u, HIST_BINS - 1u);
                atomicAdd(&hist_local[b], 1u);

                // Momentum-angle histogram about the mass-flux-weighted mean
                // direction, weighted by each sample's share of the forward
                // momentum flux. The weight is quantised to 1/2^24 of the total
                // so it can use a u32 atomic: the whole plane sums to about
                // 2^24, well inside u32, and each sample carries ~64 counts,
                // far finer than a 32-bin cone estimate can resolve.
                let flux = fs.rho * speed * max(w, 0.0);
                if (jm > 0.0 && total_flux > 0.0 && flux > 0.0 && speed > 0.0) {
                    let ang = acos(clamp(dot(fs.u / speed, j / jm), -1.0, 1.0));
                    let ab = clamp(u32(ang / PI * f32(ANGLE_BINS)), 0u, ANGLE_BINS - 1u);
                    atomicAdd(&angle_local[ab], u32(clamp(flux / total_flux * 16777216.0, 0.0, 4.0e9)));
                }
            }
        }
    }
    workgroupBarrier();

    // Halving tree over the workgroup. The three extrema cannot fold as sums,
    // and folding them here is what keeps the global buffer to one update each.
    var stride = WG / 2u;
    loop {
        if (tid < stride) {
            for (var k = 0u; k < K_SUM_END; k = k + 1u) {
                part[k * WG + tid] = part[k * WG + tid] + part[k * WG + tid + stride];
            }
            part[K_MAX_SPEED * WG + tid] = max(part[K_MAX_SPEED * WG + tid], part[K_MAX_SPEED * WG + tid + stride]);
            part[K_MAX_W * WG + tid] = max(part[K_MAX_W * WG + tid], part[K_MAX_W * WG + tid + stride]);
            part[K_MIN_W * WG + tid] = min(part[K_MIN_W * WG + tid], part[K_MIN_W * WG + tid + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        if (P.pass_index == 0u) {
            atomicAdd(&acc[base + A_N_TOTAL], u32(part[0u * WG]));
            atomicAdd(&acc[base + A_N_INSIDE], u32(part[1u * WG]));
            atomicAdd(&acc[base + A_N_FLUID], u32(part[2u * WG]));
            atomicAdd(&acc[base + A_N_BACK], u32(part[3u * WG]));
            for (var s = 0u; s < S_MAX_SPEED; s = s + 1u) {
                atomic_add_f32(base + A_SCALAR_BASE + s, part[(4u + s) * WG]);
            }
            // Only workgroups that actually saw fluid may move the extrema; a
            // workgroup entirely inside the wall would otherwise push its
            // sentinel into the global maximum.
            if (part[2u * WG] > 0.0) {
                atomic_max_f32(base + A_SCALAR_BASE + S_MAX_SPEED, part[K_MAX_SPEED * WG]);
                atomic_max_f32(base + A_SCALAR_BASE + S_MAX_W, part[K_MAX_W * WG]);
                atomic_min_f32(base + A_SCALAR_BASE + S_MIN_W, part[K_MIN_W * WG]);
            }
        } else {
            atomic_add_f32(base + A_SCALAR_BASE + S_ABS_DEV, part[(4u + S_ABS_DEV) * WG]);
        }
    }

    if (P.pass_index == 1u) {
        for (var b = tid; b < HIST_BINS; b = b + WG) {
            let v = atomicLoad(&hist_local[b]);
            if (v != 0u) { atomicAdd(&acc[base + A_HIST_BASE + b], v); }
        }
        for (var b = tid; b < ANGLE_BINS; b = b + WG) {
            let v = atomicLoad(&angle_local[b]);
            if (v != 0u) { atomicAdd(&acc[base + A_ANGLE_BASE + b], v); }
        }
    }
}

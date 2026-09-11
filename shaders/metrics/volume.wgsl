// Whole-domain reductions: peak velocity, separation, stagnation, residual.
//
// # Why peak velocity is the headline number
//
// Dipole aeroacoustic sound power scales as U^6 (Curle 1955; Lighthill's
// eighth-power law is the quadrupole case, which is not what a duct at Mach 0.02
// radiates). Sixty times the log of a 20% peak reduction is 5.8 dB - audible,
// and achievable by rounding one corner. No other single number in the app
// points at a design change that directly, which is why this pass reduces over
// EVERY fluid cell rather than a sample: a peak that only exists in one cell is
// still the cell that makes the noise.
//
// # Two entry points, two costs
//
// `volume_stats` visits every cell. At the interactive tier that is 20 M cells
// reading 12 bytes each, about 0.25 ms on the reference hardware - cheap enough
// to run every frame, expensive enough to be worth knowing about.
//
// `volume_residual` visits a strided subset, because it needs the *previous*
// field to difference against and storing 20 M velocities would cost 320 MB.
// At stride 4 that becomes 5 MB and 312 k probes, which estimates the field-wide
// L2 norm to well inside the precision anyone reads a residual to. The residual
// is a convergence indicator, not a conserved quantity; spending 64x the memory
// to sharpen its third digit would be a poor trade.
//
// # Why a fixed number of workgroups
//
// Both entry points walk their index space with a stride of the whole launch,
// and volume.rs dispatches them as a fixed number of workgroups whatever the
// grid size. The alternative -- one workgroup per 64 cells -- costed out at
// 0.25 ms from bandwidth and measured at 30-300 ms: every workgroup ends in a
// compare-exchange loop on the same handful of accumulator words, and at
// dx = 0.75 mm that is 337,000 of them per word, serialised in L2. The cost
// grew as the flow filled the domain (a zero sum skips its atomic) until the
// frame ran at 5 fps. The atomic count now stays at the workgroup count.

#include "metrics/common.wgsl"

struct VolumeUniforms {
    grid: GridInfo,
    // Duct axis, unit. Reverse flow is u . axis < 0.
    axis: vec3<f32>,
    // |u| below this counts as stagnant. Set to 0.05 * V_bulk in lattice units.
    stagnation_threshold: f32,
    // Probe grid for the residual pass: ceil(dims / stride).
    probe_dims: vec3<u32>,
    stride: u32,
    // 0 on the first residual pass, when there is nothing to difference against.
    has_prev: u32,
    pad: vec3<u32>,
};

@group(1) @binding(0) var<uniform> V: VolumeUniforms;
@group(1) @binding(1) var<storage, read_write> acc: array<atomic<u32>>;
// Previous velocity at each probe point. Written by every residual pass.
@group(1) @binding(2) var<storage, read_write> prev: array<vec4<f32>>;

const WG: u32 = 64u;
const SLOTS: u32 = 4u + V_N_SCALARS;
const WG_SLOTS: u32 = WG * SLOTS;
// Slots [0, K_SUM_END) fold by addition; then max, max, min.
const K_SUM_END: u32 = 4u + VS_MAX_SPEED;
const K_MAX_SPEED: u32 = 4u + VS_MAX_SPEED;
const K_MAX_RHO: u32 = 4u + VS_MAX_RHO;
const K_MIN_RHO: u32 = 4u + VS_MIN_RHO;
const_assert K_MIN_RHO == SLOTS - 1u;

var<workgroup> part: array<f32, WG_SLOTS>;

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

fn clear_partials(tid: u32) {
    for (var k = 0u; k < SLOTS; k = k + 1u) {
        part[k * WG + tid] = 0.0;
    }
    part[K_MAX_SPEED * WG + tid] = -1.0e30;
    part[K_MAX_RHO * WG + tid] = -1.0e30;
    part[K_MIN_RHO * WG + tid] = 1.0e30;
}

fn reduce_and_flush(tid: u32) {
    var stride = WG / 2u;
    loop {
        if (tid < stride) {
            for (var k = 0u; k < K_SUM_END; k = k + 1u) {
                part[k * WG + tid] = part[k * WG + tid] + part[k * WG + tid + stride];
            }
            part[K_MAX_SPEED * WG + tid] = max(part[K_MAX_SPEED * WG + tid], part[K_MAX_SPEED * WG + tid + stride]);
            part[K_MAX_RHO * WG + tid] = max(part[K_MAX_RHO * WG + tid], part[K_MAX_RHO * WG + tid + stride]);
            part[K_MIN_RHO * WG + tid] = min(part[K_MIN_RHO * WG + tid], part[K_MIN_RHO * WG + tid + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride >> 1u;
    }
    if (tid == 0u) {
        atomicAdd(&acc[V_N_VISITED], u32(part[0u * WG]));
        atomicAdd(&acc[V_N_FLUID], u32(part[1u * WG]));
        atomicAdd(&acc[V_N_REVERSE], u32(part[2u * WG]));
        atomicAdd(&acc[V_N_STAGNANT], u32(part[3u * WG]));
        for (var s = 0u; s < VS_MAX_SPEED; s = s + 1u) {
            atomic_add_f32(V_SCALAR_BASE + s, part[(4u + s) * WG]);
        }
        if (part[1u * WG] > 0.0) {
            atomic_max_f32(V_SCALAR_BASE + VS_MAX_SPEED, part[K_MAX_SPEED * WG]);
            atomic_max_f32(V_SCALAR_BASE + VS_MAX_RHO, part[K_MAX_RHO * WG]);
            atomic_min_f32(V_SCALAR_BASE + VS_MIN_RHO, part[K_MIN_RHO * WG]);
        }
    }
}

// Peak speed, separation and stagnation over every cell.
//
// Separation is measured against a supplied duct axis rather than against the
// local streamline direction, because "the flow here is going backwards" is only
// meaningful relative to where the duct is trying to send it. A bend turns the
// flow through 90 degrees; against the local direction nothing is ever reversed.
@compute @workgroup_size(64, 1, 1)
fn volume_stats(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    clear_partials(tid);

    // Per-thread partials in registers across the stride loop; counts stay
    // exact in f32 up to 2^24 per workgroup, far beyond any grid this runs.
    let d = V.grid.dims;
    let total = d.x * d.y * d.z;
    var n_visited = 0.0;
    var n_fluid = 0.0;
    var n_reverse = 0.0;
    var n_stagnant = 0.0;
    var sum_speed = 0.0;
    var sum_speed2 = 0.0;
    var sum_axial = 0.0;
    var max_speed = -1.0e30;
    var max_rho = -1.0e30;
    var min_rho = 1.0e30;
    for (var i = wid.x * WG + tid; i < total; i = i + nwg.x * WG) {
        let fs = load_cell(V.grid, vec3<u32>(i % d.x, (i / d.x) % d.y, i / (d.x * d.y)));
        n_visited = n_visited + 1.0;
        if (fs.fluid) {
            let speed = length(fs.u);
            let axial = dot(fs.u, V.axis);
            n_fluid = n_fluid + 1.0;
            n_reverse = n_reverse + select(0.0, 1.0, axial < 0.0);
            n_stagnant = n_stagnant + select(0.0, 1.0, speed < V.stagnation_threshold);
            sum_speed = sum_speed + speed;
            sum_speed2 = sum_speed2 + speed * speed;
            sum_axial = sum_axial + axial;
            max_speed = max(max_speed, speed);
            max_rho = max(max_rho, fs.rho);
            min_rho = min(min_rho, fs.rho);
        }
    }
    part[0u * WG + tid] = n_visited;
    part[1u * WG + tid] = n_fluid;
    part[2u * WG + tid] = n_reverse;
    part[3u * WG + tid] = n_stagnant;
    part[(4u + VS_SUM_SPEED) * WG + tid] = sum_speed;
    part[(4u + VS_SUM_SPEED2) * WG + tid] = sum_speed2;
    part[(4u + VS_SUM_AXIAL) * WG + tid] = sum_axial;
    part[K_MAX_SPEED * WG + tid] = max_speed;
    part[K_MAX_RHO * WG + tid] = max_rho;
    part[K_MIN_RHO * WG + tid] = min_rho;
    workgroupBarrier();
    reduce_and_flush(tid);
}

// Pseudo-residual R = ||u^{n+1} - u^n|| / ||u^n||, over a strided probe set.
//
// Both norms are accumulated as squares here; the CPU takes the ratio of the
// square roots. Solid cells are skipped entirely rather than contributing a
// pair of zeros: they would add nothing to the numerator and nothing to the
// denominator, but they would make the probe count meaningless.
@compute @workgroup_size(64, 1, 1)
fn volume_residual(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    clear_partials(tid);

    let p = V.probe_dims;
    let total = p.x * p.y * p.z;
    var n_visited = 0.0;
    var n_fluid = 0.0;
    var sum_du2 = 0.0;
    var sum_u2 = 0.0;
    // `i` is the probe's linear index, X-fastest: the slot in `prev`.
    for (var i = wid.x * WG + tid; i < total; i = i + nwg.x * WG) {
        let g = vec3<u32>(i % p.x, (i / p.x) % p.y, i / (p.x * p.y));
        let c = min(g * V.stride, V.grid.dims - vec3<u32>(1u, 1u, 1u));
        let fs = load_cell(V.grid, c);
        n_visited = n_visited + 1.0;
        if (fs.fluid) {
            n_fluid = n_fluid + 1.0;
            if (V.has_prev != 0u) {
                let du = fs.u - prev[i].xyz;
                sum_du2 = sum_du2 + dot(du, du);
                sum_u2 = sum_u2 + dot(fs.u, fs.u);
            }
        }
        prev[i] = vec4<f32>(fs.u, fs.rho);
    }
    part[0u * WG + tid] = n_visited;
    part[1u * WG + tid] = n_fluid;
    part[(4u + VS_SUM_DU2) * WG + tid] = sum_du2;
    part[(4u + VS_SUM_U2) * WG + tid] = sum_u2;
    workgroupBarrier();
    reduce_and_flush(tid);
}

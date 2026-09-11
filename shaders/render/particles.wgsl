// GPU streakline tracers: advection, recycling, seeding and drawing.
//
// See `crates/ad-render/src/particles.rs` for the reasoning. The five things
// this file must not get wrong, in order of how badly they show:
//
//   1. **Sub-cell CFL.** `h = min(dt, cfl * dx / |u|)` with 1..max_substeps
//      substeps, plus a hard displacement backstop for when the substep cap
//      saturates. A particle that moves more than a cell per substep steps over
//      the duct wall and the image reads as broken.
//   2. **The free list is an append buffer.** Every particle is either alive or
//      pushed exactly once per frame onto `free_list` by `advect_main`. The
//      seeder consumes it in the same frame through an indirect dispatch, so
//      nothing is ever read back to the CPU.
//   3. **Sprites are clamped to a minimum radius and alpha-compensated.**
//      `alpha *= (r_true / r_clamped)^2` conserves emitted energy; without it a
//      receding particle cloud gets brighter.
//   4. **Motion blur is the geometry.** The sprite is a capsule from `prev` to
//      `pos`, with `alpha *= r / (r + L)` so the streak is long and faint rather
//      than long and bright.
//   5. **Additive blending.** No sorting, no order dependence, no OIT.

#include "common.wgsl"

struct ParticleUniform {
    volume_min_mm: vec3<f32>,
    // Solver cell size, not the derived voxel: the CFL condition is about not
    // stepping over a wall, and walls are resolved on the solver grid.
    dx_mm: f32,

    volume_size_mm: vec3<f32>,
    dt_sim: f32,

    seed_min_mm: vec3<f32>,
    cfl: f32,

    seed_size_mm: vec3<f32>,
    max_step_cells: f32,

    inlet_center: vec3<f32>,
    inlet_un_max: f32,

    inlet_u: vec3<f32>,
    w_inlet: f32,

    inlet_v: vec3<f32>,
    w_volume: f32,

    inlet_n: vec3<f32>,
    w_import: f32,

    sdf_min_mm: vec3<f32>,
    sdf_offset_mm: f32,

    sdf_inv_size: vec3<f32>,
    particle_radius_mm: f32,

    count: u32,
    frame: u32,
    max_substeps: u32,
    sdf_present: u32,

    life_mean_s: f32,
    life_jitter: f32,
    fade_in: f32,
    fade_out: f32,

    radius_mm: f32,
    min_radius_px: f32,
    intensity: f32,
    color_mode: u32,

    color_lo: f32,
    color_inv_span: f32,
    cdf_dim: u32,
    q_threshold: f32,

    trail_count: u32,
    trail_len: u32,
    trail_head: u32,
    trail_max_age_s: f32,

    inlet_offset_mm: f32,
    occupancy_min: f32,
    motion_blur: f32,
    max_streak_px: f32,

    inv_dt_ms: f32,
    trail_width_mm: f32,
    trail_alpha: f32,
    pad0: f32,
};

// `life <= 0` is the one and only definition of "dead". The buffer is zeroed at
// creation, so every particle starts dead and the first frame's seeder fills the
// whole population with no separate initialisation kernel.
struct Particle {
    pos: vec3<f32>,
    age: f32,
    prev: vec3<f32>,
    life: f32,
};

// Millimetres per metre. The derived velocity texture is in m/s; positions are
// in mm, like everything else in the app.
const MM_PER_M: f32 = 1000.0;

#if PARTICLES_DRAW

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> pu: ParticleUniform;
@group(1) @binding(1) var<storage, read> particles: array<Particle>;
@group(1) @binding(2) var<storage, read> trails: array<vec2<u32>>;
@group(1) @binding(3) var color_lut: texture_2d<f32>;
@group(1) @binding(4) var lut_sampler: sampler;

#else

@group(0) @binding(0) var<uniform> fu: FieldsInfo;
@group(0) @binding(1) var field_sampler: sampler;
@group(0) @binding(2) var field_scalars: texture_3d<f32>;
@group(0) @binding(3) var field_velocity: texture_3d<f32>;

@group(1) @binding(0) var<uniform> pu: ParticleUniform;
@group(1) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(1) @binding(2) var<storage, read_write> free_list: array<u32>;
@group(1) @binding(3) var<storage, read_write> counters: Counters;
@group(1) @binding(4) var<storage, read_write> dispatch_args: array<u32>;
@group(1) @binding(5) var<storage, read_write> cdf: array<f32>;
@group(1) @binding(6) var solid_sdf: texture_3d<f32>;
@group(1) @binding(7) var<storage, read_write> trails: array<vec2<u32>>;

struct Counters {
    free_count: atomic<u32>,
    alive: atomic<u32>,
    pad0: u32,
    pad1: u32,
};

#endif

// -- shared helpers ----------------------------------------------------------

// Trail ring entries are 16-bit-quantised positions plus a 16-bit age.
//
// Quantising against the field bounding box rather than storing f16 is not a
// micro-optimisation: over a 260 mm domain f16 resolves only ~0.13 mm, which is
// a fifth of a cell and visibly kinks a ribbon. Unorm16 over the same box
// resolves 0.004 mm, and costs the same eight bytes.
//
// The age slot doubles as a validity marker. It is stored in 1..65535, so zero
// means "never written"; and because it only ever increases along a live
// particle's history, a segment whose newer end has a *smaller* age than its
// older end is one that straddles a respawn, and is dropped. That is what stops
// a recycled particle from drawing a streak across the whole domain.
fn pack_trail(p: vec3<f32>, age: f32, max_age: f32) -> vec2<u32> {
    let t = clamp((p - pu.volume_min_mm) / pu.volume_size_mm, vec3<f32>(0.0), vec3<f32>(1.0));
    let q = vec3<u32>(t * 65535.0 + vec3<f32>(0.5));
    let a = u32(clamp(age / max(max_age, 1.0e-6), 0.0, 1.0) * 65534.0) + 1u;
    return vec2<u32>(q.x | (q.y << 16u), q.z | (a << 16u));
}

fn trail_pos(v: vec2<u32>) -> vec3<f32> {
    let q = vec3<f32>(f32(v.x & 0xffffu), f32(v.x >> 16u), f32(v.y & 0xffffu));
    return pu.volume_min_mm + (q / 65535.0) * pu.volume_size_mm;
}

fn trail_age(v: vec2<u32>) -> u32 {
    return v.y >> 16u;
}

// Fade in over the first `fade_in` of a life and out over the last `fade_out`.
// Smoothstep at both ends: a linear fade-in has a derivative discontinuity at
// birth, which across a few million particles reads as a faint popping texture
// along the inlet. Mirrors `particles::fade_alpha`.
fn fade_alpha(age: f32, life: f32) -> f32 {
    if (life <= 0.0 || age < 0.0 || age > life) {
        return 0.0;
    }
    var a = 1.0;
    if (pu.fade_in > 1.0e-6) {
        a = smoothstep(0.0, 1.0, clamp(age / (pu.fade_in * life), 0.0, 1.0));
    }
    var b = 1.0;
    if (pu.fade_out > 1.0e-6) {
        b = smoothstep(0.0, 1.0, clamp((life - age) / (pu.fade_out * life), 0.0, 1.0));
    }
    return a * b;
}

#if !PARTICLES_DRAW

// -- compute side ------------------------------------------------------------

// PCG-style hash-and-advance. Cheap, and decorrelated enough that neighbouring
// particle indices in the same frame do not seed at the same place — which a
// plain LCG on the index absolutely does, and which shows up as diagonal stripes
// across the inlet.
fn rand(state: ptr<function, u32>) -> f32 {
    *state = (*state) * 747796405u + 2891336453u;
    var w = ((*state >> ((*state >> 28u) + 4u)) ^ (*state)) * 277803737u;
    w = (w >> 22u) ^ w;
    return f32(w) * (1.0 / 4294967296.0);
}

// Velocity and fluid fraction. `w < 0` means "outside the field volume", which
// is distinct from `w == 0` ("inside a wall") and has to be, because one kills
// the particle and the other is handled by the SDF.
fn sample_field(p: vec3<f32>) -> vec4<f32> {
    let uvw = (p - fu.volume_min_mm) / fu.volume_size_mm;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    return textureSampleLevel(field_velocity, field_sampler, uvw, 0.0);
}

fn velocity_mm_s(p: vec3<f32>) -> vec3<f32> {
    return sample_field(p).xyz * MM_PER_M;
}

// RK2 midpoint. Not RK4: the field is trilinear and therefore only C0, so RK4's
// fourth order collapses to about second across a voxel face while still costing
// four fetches instead of two. See the module docs.
fn rk2(p: vec3<f32>, h: f32) -> vec3<f32> {
    let k1 = velocity_mm_s(p);
    let k2 = velocity_mm_s(p + k1 * (0.5 * h));
    return p + k2 * h;
}

fn sample_sdf(p: vec3<f32>) -> f32 {
    let uvw = (p - pu.sdf_min_mm) * pu.sdf_inv_size;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
        return 1.0e6;
    }
    return textureSampleLevel(solid_sdf, field_sampler, uvw, 0.0).x - pu.sdf_offset_mm;
}

// Central difference at half a cell. Half rather than a full cell because the
// push-out direction has to be accurate right at the surface, which is exactly
// where a wide stencil straddles the zero crossing and points sideways.
fn sdf_gradient(p: vec3<f32>) -> vec3<f32> {
    let e = 0.5 * pu.dx_mm;
    return vec3<f32>(
        sample_sdf(p + vec3<f32>(e, 0.0, 0.0)) - sample_sdf(p - vec3<f32>(e, 0.0, 0.0)),
        sample_sdf(p + vec3<f32>(0.0, e, 0.0)) - sample_sdf(p - vec3<f32>(0.0, e, 0.0)),
        sample_sdf(p + vec3<f32>(0.0, 0.0, e)) - sample_sdf(p - vec3<f32>(0.0, 0.0, e)),
    );
}

fn push_free(i: u32) {
    let slot = atomicAdd(&counters.free_count, 1u);
    if (slot < pu.count) {
        free_list[slot] = i;
    }
}

@compute @workgroup_size(#PARTICLE_WG)
fn advect_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pu.count) {
        return;
    }
    var p = particles[i];
    if (p.life <= 0.0) {
        push_free(i);
        return;
    }

    // --- sub-cell CFL. The single most important block in this file. ---
    var pos = p.pos;
    let speed = length(velocity_mm_s(pos));
    var n = 1u;
    if (speed > 1.0e-6 && pu.dt_sim > 0.0) {
        let h_cfl = max(pu.cfl * pu.dx_mm / speed, 1.0e-9);
        n = u32(ceil(pu.dt_sim / h_cfl));
    }
    n = clamp(n, 1u, pu.max_substeps);
    let h = pu.dt_sim / f32(n);
    let dmax = pu.max_step_cells * pu.dx_mm;

    var alive = true;
    for (var s = 0u; s < n; s = s + 1u) {
        var d = rk2(pos, h) - pos;
        // The backstop. It only fires once the substep cap has saturated, i.e.
        // exactly in the case that would otherwise tunnel. A clamped particle
        // lags the true streakline; an unclamped one leaves through a wall.
        let dl = length(d);
        if (dl > dmax) {
            d = d * (dmax / dl);
        }
        pos = pos + d;

        if (pu.sdf_present != 0u) {
            let sd = sample_sdf(pos);
            if (sd < pu.particle_radius_mm) {
                if (sd < -1.5 * pu.dx_mm) {
                    // Deep inside solid: something went wrong upstream (a moved
                    // geometry, a coarse SDF). Recycle rather than trying to
                    // recover a position whose gradient is meaningless.
                    alive = false;
                    break;
                }
                let g = sdf_gradient(pos);
                let gl = length(g);
                if (gl > 1.0e-9) {
                    pos = pos + (g / gl) * (pu.particle_radius_mm - sd);
                }
            }
        }

        let occ = sample_field(pos).w;
        if (occ < 0.0) {
            alive = false;
            break;
        }
        if (pu.sdf_present == 0u && occ < pu.occupancy_min) {
            // No SDF: the derived fluid fraction is the only wall signal there
            // is, and it has no usable gradient at half resolution, so this
            // degrades to killing rather than to pushing out.
            alive = false;
            break;
        }
    }

    p.prev = p.pos;
    p.pos = pos;
    p.age = p.age + pu.dt_sim;

    if (!alive || p.age >= p.life) {
        p.life = 0.0;
        particles[i] = p;
        push_free(i);
        return;
    }

    particles[i] = p;
    atomicAdd(&counters.alive, 1u);
    if (pu.trail_len > 0u && i < pu.trail_count) {
        trails[i * pu.trail_len + pu.trail_head] = pack_trail(p.pos, p.age, pu.trail_max_age_s);
    }
}

// One thread, one job: size the seeder's dispatch from the atomic death count.
// This is what keeps the free list free of a CPU round trip.
@compute @workgroup_size(1)
fn prepare_main() {
    let n = min(atomicLoad(&counters.free_count), pu.count);
    dispatch_args[0] = (n + #PARTICLE_WGu - 1u) / #PARTICLE_WGu;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

// --- seeding ---------------------------------------------------------------

// Flux-weighted release: rejection sampling against `u.n`, so the probability of
// a release at a point is proportional to the volume flow through it. That is
// what makes particle density mean something — particles per second through any
// area becomes proportional to the volumetric flow through it, which is the
// property that lets a viewer read the picture quantitatively.
//
// Failing all attempts leaves the particle dead, and it is simply retried next
// frame. That is deliberate: placing it anyway would bias the density towards
// the low-flux corners, which is the one thing this function exists to avoid.
fn seed_inlet(rng: ptr<function, u32>, out: ptr<function, vec3<f32>>) -> bool {
    for (var k = 0u; k < 16u; k = k + 1u) {
        let s = rand(rng) * 2.0 - 1.0;
        let t = rand(rng) * 2.0 - 1.0;
        let p = pu.inlet_center + pu.inlet_u * s + pu.inlet_v * t
            + pu.inlet_n * pu.inlet_offset_mm;
        let f = sample_field(p);
        if (f.w <= 0.0) {
            continue;
        }
        // An envelope of zero means the caller has turned flux weighting off,
        // so accept uniformly.
        if (pu.inlet_un_max <= 0.0) {
            *out = p;
            return true;
        }
        let un = dot(f.xyz, pu.inlet_n);
        if (un <= 0.0) {
            continue;
        }
        if (rand(rng) * pu.inlet_un_max <= un) {
            *out = p;
            return true;
        }
    }
    return false;
}

// Uniform reseed inside the seed box. Small in weight but not optional: nothing
// released at the inlet ever reaches a recirculating dead zone, and a dead zone
// that renders black is indistinguishable from one that is not there.
fn seed_volume(rng: ptr<function, u32>, out: ptr<function, vec3<f32>>) -> bool {
    for (var k = 0u; k < 16u; k = k + 1u) {
        let r = vec3<f32>(rand(rng), rand(rng), rand(rng));
        let p = pu.seed_min_mm + r * pu.seed_size_mm;
        if (sample_field(p).w >= pu.occupancy_min) {
            *out = p;
            return true;
        }
    }
    return false;
}

// Inverse-transform sampling of the Q-criterion CDF: draw uniformly on
// [0, total) and binary search for the bin that bracket contains. The CDF is
// inclusive, so the answer is the first index whose value exceeds the draw.
fn seed_importance(rng: ptr<function, u32>, out: ptr<function, vec3<f32>>) -> bool {
    let d = pu.cdf_dim;
    let n = d * d * d;
    if (n == 0u) {
        return false;
    }
    let total = cdf[n - 1u];
    if (!(total > 0.0)) {
        return false;
    }
    let r = rand(rng) * total;
    var lo = 0u;
    var hi = n - 1u;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        if (cdf[mid] > r) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    let b = vec3<f32>(f32(lo % d), f32((lo / d) % d), f32(lo / (d * d)));
    let cell = fu.volume_size_mm / f32(d);
    let jitter = vec3<f32>(rand(rng), rand(rng), rand(rng));
    *out = fu.volume_min_mm + (b + jitter) * cell;
    return true;
}

@compute @workgroup_size(#PARTICLE_WG)
fn seed_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let n = min(atomicLoad(&counters.free_count), pu.count);
    if (k >= n) {
        return;
    }
    let i = free_list[k];
    var rng = hash_u32((i * 0x9e3779b9u) ^ (pu.frame * 0x85ebca6bu) ^ 0x2545f491u);

    var pos = vec3<f32>(0.0);
    var ok = false;
    let r = rand(&rng);
    if (r < pu.w_inlet) {
        ok = seed_inlet(&rng, &pos);
    } else if (r < pu.w_inlet + pu.w_volume) {
        ok = seed_volume(&rng, &pos);
    } else {
        ok = seed_importance(&rng, &pos);
    }
    if (!ok) {
        // Still dead. It stays on next frame's free list and tries again, which
        // keeps the population conserved without biasing where it lands.
        return;
    }

    var p: Particle;
    p.pos = pos;
    p.prev = pos;
    p.age = 0.0;
    // +/- jitter on the lifetime. Without it the whole population dies on the
    // same frame and the image blinks once per `life_mean_s`.
    p.life = pu.life_mean_s * (1.0 - pu.life_jitter + 2.0 * pu.life_jitter * rand(&rng));
    particles[i] = p;

    if (pu.trail_len > 0u && i < pu.trail_count) {
        // Stamp the head with age 0 so every segment reaching back into the
        // previous incarnation fails the monotonic-age test and is dropped.
        trails[i * pu.trail_len + pu.trail_head] = pack_trail(pos, 0.0, pu.trail_max_age_s);
    }
}

// --- importance CDF --------------------------------------------------------

// Bin weight: mean positive excess of normalised Q over the threshold, inside
// fluid only. Sampling the bin rather than reducing every voxel in it is
// deliberate — the CDF only has to steer seeding towards structure, and 4^3
// stratified samples per bin is plenty for that at a 512th of the cost.
@compute @workgroup_size(#IMPORTANCE_WG)
fn importance_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = pu.cdf_dim;
    let n = d * d * d;
    let idx = gid.x;
    if (idx >= n) {
        return;
    }
    let b = vec3<f32>(f32(idx % d), f32((idx / d) % d), f32(idx / (d * d)));
    let cell = fu.volume_size_mm / f32(d);
    let base = fu.volume_min_mm + b * cell;

    var acc = 0.0;
    for (var z = 0u; z < 4u; z = z + 1u) {
        for (var y = 0u; y < 4u; y = y + 1u) {
            for (var x = 0u; x < 4u; x = x + 1u) {
                let f = (vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5)) * 0.25;
                let p = base + f * cell;
                let uvw = (p - fu.volume_min_mm) / fu.volume_size_mm;
                let q = textureSampleLevel(field_scalars, field_sampler, uvw, 0.0).y;
                let occ = textureSampleLevel(field_velocity, field_sampler, uvw, 0.0).w;
                if (occ >= pu.occupancy_min) {
                    acc = acc + max(q - pu.q_threshold, 0.0);
                }
            }
        }
    }
    cdf[idx] = acc;
}

var<workgroup> scan_partials: array<f32, #SCAN_WG>;

// Inclusive prefix sum of the whole bin grid in one workgroup.
//
// Two levels: each thread serially sums its own contiguous chunk, the 256 chunk
// totals are scanned in shared memory, then each thread rewrites its chunk with
// the running total. One workgroup because 32768 entries is small enough that a
// multi-block scan's extra dispatch and fix-up pass would cost more than the
// parallelism it buys.
@compute @workgroup_size(#SCAN_WG)
fn scan_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let threads = #SCAN_WGu;
    let n = pu.cdf_dim * pu.cdf_dim * pu.cdf_dim;
    let chunk = (n + threads - 1u) / threads;
    let start = min(lid.x * chunk, n);
    let end = min(start + chunk, n);

    var sum = 0.0;
    for (var i = start; i < end; i = i + 1u) {
        sum = sum + cdf[i];
    }
    scan_partials[lid.x] = sum;
    workgroupBarrier();

    // Hillis-Steele over the chunk totals. Read before the barrier, write after:
    // the two-barrier form is what makes it correct without a double buffer.
    for (var off = 1u; off < threads; off = off * 2u) {
        var v = 0.0;
        if (lid.x >= off) {
            v = scan_partials[lid.x - off];
        }
        workgroupBarrier();
        if (lid.x >= off) {
            scan_partials[lid.x] = scan_partials[lid.x] + v;
        }
        workgroupBarrier();
    }

    var running = 0.0;
    if (lid.x > 0u) {
        running = scan_partials[lid.x - 1u];
    }
    for (var i = start; i < end; i = i + 1u) {
        running = running + cdf[i];
        cdf[i] = running;
    }
}

#else

// -- draw side ---------------------------------------------------------------

struct SpriteOut {
    @builtin(position) clip: vec4<f32>,
    // Pixel offset from the capsule centre, in the capsule's own frame:
    // x along the streak, y across it.
    @location(0) local: vec2<f32>,
    // x = Gaussian radius in pixels, y = capsule core half-length in pixels.
    @location(1) shape: vec2<f32>,
    @location(2) color: vec4<f32>,
};

fn ndc_to_px(ndc: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return vec2<f32>((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y);
}

// Six vertices, two triangles, generated from the vertex index. Corner order is
// (0,1,2, 0,2,3) around a unit quad.
fn quad_corner(index: u32) -> vec2<f32> {
    var q = index;
    if (index == 3u) {
        q = 0u;
    } else if (index == 4u) {
        q = 2u;
    } else if (index == 5u) {
        q = 3u;
    }
    let x = select(-1.0, 1.0, q == 1u || q == 2u);
    let y = select(-1.0, 1.0, q == 2u || q == 3u);
    return vec2<f32>(x, y);
}

// Everything off screen collapses to a single point, so the two triangles are
// degenerate and the rasteriser discards them without a discard.
fn degenerate() -> SpriteOut {
    var o: SpriteOut;
    o.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    o.local = vec2<f32>(0.0);
    o.shape = vec2<f32>(1.0, 0.0);
    o.color = vec4<f32>(0.0);
    return o;
}

fn lut_color(t: f32) -> vec3<f32> {
    return textureSampleLevel(color_lut, lut_sampler, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0).rgb;
}

// Colour coordinate for a particle, in [0, 1] along the LUT.
fn color_coord(p: Particle) -> f32 {
    if (pu.color_mode == 1u) {
        return clamp(p.age / max(p.life, 1.0e-6), 0.0, 1.0);
    }
    if (pu.color_mode == 2u) {
        return 1.0;
    }
    // Speed, recovered from the frame's own displacement rather than stored:
    // it is exactly the advected speed, and it costs no bytes per particle.
    let speed_ms = length(p.pos - p.prev) * pu.inv_dt_ms;
    return (speed_ms - pu.color_lo) * pu.color_inv_span;
}

@vertex
fn vs_particle(@builtin(vertex_index) vi: u32) -> SpriteOut {
    let pid = vi / 6u;
    if (pid >= pu.count) {
        return degenerate();
    }
    let p = particles[pid];
    if (p.life <= 0.0) {
        return degenerate();
    }

    let c1 = cam.view_proj * vec4<f32>(p.pos, 1.0);
    if (c1.w <= 0.0) {
        return degenerate();
    }
    var c0 = cam.view_proj * vec4<f32>(p.prev, 1.0);
    if (c0.w <= 0.0) {
        c0 = c1;
    }

    let res = cam.jitter_resolution.zw;
    let s1 = ndc_to_px(c1.xy / c1.w, res);
    let s0 = ndc_to_px(c0.xy / c0.w, res);

    // Perspective footprint: half the viewport height divided by tan(fov/2) is
    // the pixels-per-millimetre at one millimetre of view distance, and `w` is
    // exactly that view distance because the projection sets w = -z_view.
    let proj_scale = 0.5 * res.y / max(cam.forward_tan.w, 1.0e-6);
    let r_true = pu.radius_mm * proj_scale / c1.w;
    let r_px = max(r_true, pu.min_radius_px);
    // Energy conservation under the clamp. Without this a receding cloud gets
    // brighter as its footprint stops shrinking.
    let clamp_scale = clamp((r_true / r_px) * (r_true / r_px), 0.0, 1.0);

    // Motion blur: the capsule *is* the exposure, so the alpha is divided by the
    // stretch and a fast particle is a long faint streak.
    var streak = length(s1 - s0) * pu.motion_blur;
    streak = min(streak, pu.max_streak_px);
    let streak_scale = r_px / (r_px + streak);

    let axis = s1 - s0;
    let al = length(axis);
    var dir = vec2<f32>(1.0, 0.0);
    if (al > 1.0e-4) {
        dir = axis / al;
    }
    let nrm = vec2<f32>(-dir.y, dir.x);
    let mid = 0.5 * (s0 + s1);
    let half_len = 0.5 * streak;
    // 2.5 sigma of the Gaussian: beyond that the contribution is under 0.2% and
    // the fragment shader would discard it anyway.
    let pad = r_px * 2.5;

    let corner = quad_corner(vi % 6u);
    let offset = dir * (corner.x * (half_len + pad)) + nrm * (corner.y * pad);
    let px = mid + offset;
    let ndc = pixel_to_ndc(px, res);

    var o: SpriteOut;
    // The whole sprite sits at the particle's depth: a billboard, so it is
    // depth-tested against the scene as a point, which is what we want.
    o.clip = vec4<f32>(ndc * c1.w, c1.z, c1.w);
    o.local = offset_in_frame(offset, dir, nrm);
    o.shape = vec2<f32>(r_px, half_len);
    let a = fade_alpha(p.age, p.life) * pu.intensity * clamp_scale * streak_scale;
    o.color = vec4<f32>(lut_color(color_coord(p)), a);
    return o;
}

fn offset_in_frame(offset: vec2<f32>, dir: vec2<f32>, nrm: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(dot(offset, dir), dot(offset, nrm));
}

@fragment
fn fs_particle(in: SpriteOut) -> @location(0) vec4<f32> {
    // Distance to the capsule's core segment.
    let d = length(vec2<f32>(max(abs(in.local.x) - in.shape.y, 0.0), in.local.y));
    let x = d / max(in.shape.x, 1.0e-4);
    let w = exp(-2.5 * x * x);
    if (w < 0.002) {
        discard;
    }
    let a = in.color.a * w;
    // Premultiplied for an additive blend. Destination alpha is left alone by
    // the pipeline's `Zero, One` alpha blend.
    return vec4<f32>(in.color.rgb * a, a);
}

// --- ribbon trails ---------------------------------------------------------

struct TrailOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) across: f32,
    @location(1) color: vec4<f32>,
};

fn trail_degenerate() -> TrailOut {
    var o: TrailOut;
    o.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    o.across = 0.0;
    o.color = vec4<f32>(0.0);
    return o;
}

@vertex
fn vs_trail(@builtin(vertex_index) vi: u32) -> TrailOut {
    let segments = max(pu.trail_len, 1u) - 1u;
    if (segments == 0u || pu.trail_count == 0u) {
        return trail_degenerate();
    }
    let per_particle = 6u * segments;
    let pid = vi / per_particle;
    if (pid >= pu.trail_count) {
        return trail_degenerate();
    }
    let rem = vi % per_particle;
    let seg = rem / 6u;
    let corner = rem % 6u;

    let p = particles[pid];
    if (p.life <= 0.0) {
        return trail_degenerate();
    }

    // Walk backwards from the ring head: entry j is j frames old.
    let len = pu.trail_len;
    let ia = pid * len + ((pu.trail_head + len - seg) % len);
    let ib = pid * len + ((pu.trail_head + len - seg - 1u) % len);
    let ea = trails[ia];
    let eb = trails[ib];
    let aa = trail_age(ea);
    let ab = trail_age(eb);
    // Both ends written, and age increasing towards the head. A respawn resets
    // the age, so this is what drops the segment that would otherwise draw a
    // streak from the old position to the new one.
    if (aa == 0u || ab == 0u || aa <= ab) {
        return trail_degenerate();
    }

    let wa = cam.view_proj * vec4<f32>(trail_pos(ea), 1.0);
    let wb = cam.view_proj * vec4<f32>(trail_pos(eb), 1.0);
    if (wa.w <= 0.0 || wb.w <= 0.0) {
        return trail_degenerate();
    }
    let res = cam.jitter_resolution.zw;
    let sa = ndc_to_px(wa.xy / wa.w, res);
    let sb = ndc_to_px(wb.xy / wb.w, res);

    let c = quad_corner(corner);
    // `quad_corner` runs -1..1 along the segment; the ribbon wants 0..1, so the
    // older end is at 0 and the newer at 1.
    let along = c.x * 0.5 + 0.5;
    let across = c.y;

    let t = f32(seg) / f32(segments);
    let proj_scale = 0.5 * res.y / max(cam.forward_tan.w, 1.0e-6);
    let depth_w = mix(wb.w, wa.w, along);
    // Taper the width and, quadratically, the alpha towards the tail. The
    // quadratic makes the tail vanish rather than end.
    let width_px = max(pu.trail_width_mm * proj_scale / depth_w, 0.5) * (1.0 - t);

    let axis = sa - sb;
    let al = length(axis);
    var dir = vec2<f32>(1.0, 0.0);
    if (al > 1.0e-4) {
        dir = axis / al;
    }
    let nrm = vec2<f32>(-dir.y, dir.x);
    let px = sb + dir * (along * al) + nrm * (across * width_px);
    let ndc = pixel_to_ndc(px, res);

    var o: TrailOut;
    o.clip = vec4<f32>(
        ndc * depth_w,
        mix(wb.z, wa.z, along),
        depth_w,
    );
    o.across = across;
    let fade = (1.0 - t) * (1.0 - t);
    let a = fade_alpha(p.age, p.life) * pu.trail_alpha * pu.intensity * fade;
    o.color = vec4<f32>(lut_color(color_coord(p)), a);
    return o;
}

@fragment
fn fs_trail(in: TrailOut) -> @location(0) vec4<f32> {
    // Soft across the ribbon so it reads as a filament rather than a strip.
    let w = 1.0 - in.across * in.across;
    let a = in.color.a * max(w, 0.0);
    if (a < 0.001) {
        discard;
    }
    return vec4<f32>(in.color.rgb * a, a);
}

#endif

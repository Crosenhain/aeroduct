//! GPU validation: the shader against the CPU reference, and the analytic cases
//! re-run on the device.
//!
//! Every test here skips cleanly when no adapter is present, so CI without a GPU
//! still runs the whole physics suite through `ReferenceLbm`.
//!
//! The comparison against the CPU reference is the load-bearing one. The
//! analytic cases in `poiseuille.rs` and `streaming.rs` have already established
//! that the *physics* is right; what is left to establish is that the WGSL is the
//! same physics. Comparing the two field-for-field answers that in one assertion,
//! and when it fails it says "plumbing", not "physics".
//!
//! Every test that touches the device takes `shared_gpu()` or `exclusive_gpu()`
//! first; `GPU` below says why.

#[path = "harness.rs"]
mod harness;

use ad_gpu::types::{flags, DdfPrecision, Grid, VelocitySet};
use ad_solver::{CollisionModel, PaddedDomain, ReferenceLbm, Solver, SolverConfig};
use glam::{UVec3, Vec3};
use harness::*;
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Who else may be on the GPU while a test runs.
///
/// libtest runs the tests in this file on parallel threads, each with its own
/// device on the same adapter. (Cargo runs test *binaries* one at a time, so
/// this file is the only source of that contention.) The correctness tests do
/// not care: they compare numbers, not times. `throughput_at_a_realistic_grid`
/// does. With the rest of this file on the GPU its timestamped passes read ~3x
/// slow against a quieter wall-clock batch, which fails the build and says
/// nothing about the profiler.
///
/// So the correctness tests share the device and the timing test takes it
/// alone. A read-write lock rather than a mutex, because the correctness tests
/// are most of this binary's run time and gain nothing from queueing behind
/// each other.
///
/// A new GPU test that takes neither guard can land in the middle of a
/// measurement and bring the flakiness back.
static GPU: RwLock<()> = RwLock::new(());

/// For tests whose outcome does not depend on timing.
fn shared_gpu() -> RwLockReadGuard<'static, ()> {
    // A panic in the timing test poisons the lock. That failure has already
    // been reported and the lock guards no data, so carry on regardless.
    GPU.read().unwrap_or_else(PoisonError::into_inner)
}

/// For tests that measure time: nothing else in this process is on the GPU
/// while the guard is held.
fn exclusive_gpu() -> RwLockWriteGuard<'static, ()> {
    GPU.write().unwrap_or_else(PoisonError::into_inner)
}

fn grid_of(dims: UVec3) -> Grid {
    Grid {
        dims,
        dx_mm: 1.0,
        origin_mm: Vec3::ZERO,
    }
}

/// Build the same problem on both solvers.
fn pair(
    dims: UVec3,
    mask: Vec<u8>,
    cfg: SolverConfig,
    gpu: &ad_gpu::GpuContext,
) -> (ReferenceLbm, Solver) {
    let domain = PaddedDomain::new(dims, cfg.periodic, &mask, cfg.set);
    let cpu = ReferenceLbm::new(domain, cfg);
    let gpu = Solver::new(gpu, grid_of(dims), &mask, &[], cfg).expect("solver");
    (cpu, gpu)
}

/// Worst absolute difference in density and velocity over the interior.
fn compare(cpu: &ReferenceLbm, field: &[[f32; 4]], dims: UVec3) -> (f32, f32, UVec3) {
    let mut worst_rho = 0.0f32;
    let mut worst_u = 0.0f32;
    let mut at = UVec3::ZERO;
    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let i = ((z * dims.y + y) * dims.x + x) as usize;
                let g = field[i];
                let c = cpu.macroscopic(UVec3::new(x, y, z));
                let du = (Vec3::new(g[0], g[1], g[2]) - c.u).length();
                worst_rho = worst_rho.max((g[3] - c.rho).abs());
                if du > worst_u {
                    worst_u = du;
                    at = UVec3::new(x, y, z);
                }
            }
        }
    }
    (worst_rho, worst_u, at)
}

#[test]
fn gpu_matches_the_cpu_reference_in_a_periodic_box() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
        for collision in [
            CollisionModel::Trt,
            CollisionModel::Bgk,
            CollisionModel::RegularizedBgk,
        ] {
            let dims = UVec3::new(12, 10, 8);
            let mut cfg = clean_config(0.7);
            cfg.set = set;
            cfg.collision = collision;
            cfg.periodic = [true; 3];
            cfg.macroscopic_buffer = true;
            cfg.initial_velocity = Vec3::new(0.05, -0.03, 0.02);
            cfg.body_force = Vec3::new(1e-5, 0.0, 0.0);

            let (mut cpu, mut gpu) = pair(dims, vec![flags::FLUID; 12 * 10 * 8], cfg, &ctx);
            for _ in 0..200 {
                cpu.step();
            }
            gpu.step(200);
            let field = gpu.read_macroscopic().expect("readback");
            let (drho, du, at) = compare(&cpu, &field, dims);
            println!("{set:?}/{collision:?}: worst drho {drho:e}, worst du {du:e} at {at:?}");
            assert!(
                drho < 5e-6,
                "{set:?}/{collision:?}: density differs by {drho:e}"
            );
            assert!(
                du < 5e-6,
                "{set:?}/{collision:?}: velocity differs by {du:e} at {at:?}"
            );
        }
    }
}

/// The same comparison with walls, which exercises the bounce-back parity flip
/// and the link mask. If the shader's transport is subtly wrong this is where it
/// shows: the interior would still agree, and only the cells next to the
/// obstacle would drift.
#[test]
fn gpu_matches_the_cpu_reference_around_an_obstacle() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    let dims = UVec3::new(14, 12, 10);
    let mut mask = vec![flags::FLUID; (dims.x * dims.y * dims.z) as usize];
    let at = |x: u32, y: u32, z: u32| ((z * dims.y + y) * dims.x + x) as usize;
    // An L-shaped obstacle: concave corners are where a wrong bounce-back rule
    // is most likely to be visible, because a cell can have several blocked
    // links at once.
    for z in 2..8u32 {
        for y in 4..7u32 {
            mask[at(6, y, z)] = flags::SOLID;
        }
        for x in 6..10u32 {
            mask[at(x, 6, z)] = flags::SOLID;
        }
    }

    let mut cfg = clean_config(0.65);
    cfg.periodic = [true; 3];
    cfg.macroscopic_buffer = true;
    cfg.initial_velocity = Vec3::new(0.06, 0.0, 0.0);
    cfg.body_force = Vec3::new(2e-5, 0.0, 0.0);

    let (mut cpu, mut gpu) = pair(dims, mask, cfg, &ctx);
    for _ in 0..300 {
        cpu.step();
    }
    gpu.step(300);
    let field = gpu.read_macroscopic().expect("readback");
    let (drho, du, at) = compare(&cpu, &field, dims);
    println!("obstacle: worst drho {drho:e}, worst du {du:e} at {at:?}");
    assert!(drho < 5e-6, "density differs by {drho:e}");
    assert!(du < 5e-6, "velocity differs by {du:e} at {at:?}");
}

/// A tilted inlet — what a louver aim hands the solver — on the device against
/// the CPU reference, in the production boundary arrangement. The inlet's rim
/// cells drop the velocity component through the wall beside them
/// (`inlet_velocity_at`, once in WGSL and once in Rust); a disagreement over
/// which links count as "beside a wall" shows up at exactly those cells, and
/// in the macroscopic pass, which reports the inlet velocity rather than
/// measuring it.
#[test]
fn gpu_matches_the_cpu_reference_with_a_tilted_inlet() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    let dims = UVec3::new(20, 12, 12);
    let (nx, ny, nz) = (dims.x, dims.y, dims.z);
    let at = |x: u32, y: u32, z: u32| ((z * ny + y) * nx + x) as usize;
    let (duct_end, bore) = (12u32, 3..9u32);
    let mut mask = vec![flags::FLUID; (nx * ny * nz) as usize];
    for z in 0..nz {
        for y in 0..ny {
            let inside = bore.contains(&y) && bore.contains(&z);
            for x in 0..duct_end {
                if !inside {
                    mask[at(x, y, z)] = flags::SOLID;
                }
            }
            if inside {
                mask[at(0, y, z)] = flags::INLET;
            }
            mask[at(nx - 1, y, z)] = flags::OUTLET | flags::SPONGE;
        }
    }
    for x in duct_end..nx - 1 {
        for k in 0..ny {
            mask[at(x, 0, k)] = flags::EQUILIBRIUM;
            mask[at(x, ny - 1, k)] = flags::EQUILIBRIUM;
            mask[at(x, k, 0)] = flags::EQUILIBRIUM;
            mask[at(x, k, nz - 1)] = flags::EQUILIBRIUM;
        }
    }

    let mut cfg = clean_config(0.6);
    cfg.macroscopic_buffer = true;
    // Tilted about both in-plane axes, so every rim of the bore has a wall
    // normal component to remove.
    cfg.inlet_velocity = Vec3::new(0.05, 0.02, -0.01);
    cfg.inlet_normal = Vec3::X;
    cfg.outflow_velocity = 0.05;
    cfg.outlet_normal = Vec3::X;
    cfg.sponge_cells = 4;
    cfg.sponge_strength = 0.4;

    let (mut cpu, mut gpu) = pair(dims, mask, cfg, &ctx);
    for _ in 0..300 {
        cpu.step();
    }
    gpu.step(300);
    let field = gpu.read_macroscopic().expect("readback");
    let (drho, du, at) = compare(&cpu, &field, dims);
    println!("tilted inlet: worst drho {drho:e}, worst du {du:e} at {at:?}");
    assert!(drho < 5e-6, "density differs by {drho:e}");
    assert!(du < 5e-6, "velocity differs by {du:e} at {at:?}");
}

/// Two inlet slots at once — the mouth inlet in slot 0 with the plane closure,
/// a free-standing vent in slot 1 with the local density — on the device
/// against the CPU reference. The slot is two bits of the flag byte and the
/// density rule a flag in the uniform; either decoded differently on the two
/// sides shows up at the inlet cells within a few steps.
#[test]
fn gpu_matches_the_cpu_reference_with_a_vent_in_its_own_slot() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    let dims = UVec3::new(24, 12, 12);
    let (nx, ny, nz) = (dims.x, dims.y, dims.z);
    let at = |x: u32, y: u32, z: u32| ((z * ny + y) * nx + x) as usize;
    let mut mask = vec![flags::FLUID; (nx * ny * nz) as usize];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if y == 0 || z == 0 || y == ny - 1 || z == nz - 1 {
                    mask[at(x, y, z)] = flags::EQUILIBRIUM;
                }
            }
            if (3..9).contains(&y) && (3..9).contains(&z) {
                mask[at(0, y, z)] = flags::INLET;
            }
            mask[at(nx - 1, y, z)] = flags::OUTLET | flags::SPONGE;
        }
    }
    // A tilted vent halfway along, so its staircase and its slot both matter.
    let n = Vec3::new(1.0, 0.3, 0.0).normalize();
    for z in 3..9u32 {
        for y in 2..10u32 {
            let x = 12.0 - 0.3 * (y as f32 - 6.0);
            let x = x.round() as u32;
            mask[at(x, y, z)] = flags::inlet_in_slot(1);
        }
    }
    let mut cfg = clean_config(0.6);
    cfg.macroscopic_buffer = true;
    cfg.inlet_velocity = Vec3::new(0.03, 0.0, 0.0);
    cfg.inlet_normal = Vec3::X;
    cfg.extra_inlets[0] = ad_solver::InletSpec {
        velocity: n * 0.05,
        normal: n,
        local_density: true,
    };
    cfg.outflow_velocity = 0.05;
    cfg.outlet_normal = Vec3::X;
    cfg.sponge_cells = 4;
    cfg.sponge_strength = 0.4;

    let (mut cpu, mut gpu) = pair(dims, mask, cfg, &ctx);
    for _ in 0..300 {
        cpu.step();
    }
    gpu.step(300);
    let field = gpu.read_macroscopic().expect("readback");
    let (drho, du, at_worst) = compare(&cpu, &field, dims);
    println!("vent slot: worst drho {drho:e}, worst du {du:e} at {at_worst:?}");
    assert!(drho < 5e-6, "density differs by {drho:e}");
    assert!(du < 5e-6, "velocity differs by {du:e} at {at_worst:?}");
}

/// FP16C on the device against FP16C on the CPU. Because the CPU reference
/// truncates through `ad_solver::precision::quantise_fp16c`, agreement here is
/// evidence that `wgsl_codec()` and the Rust codec produce the *same bits* — a
/// one-ulp difference in the encoder would decorrelate the two fields within a
/// few hundred steps.
#[test]
fn gpu_fp16c_storage_matches_the_rust_codec() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    let dims = UVec3::new(10, 8, 8);
    let mut cfg = clean_config(0.7);
    cfg.precision = DdfPrecision::Fp16c;
    cfg.periodic = [true; 3];
    cfg.macroscopic_buffer = true;
    cfg.initial_velocity = Vec3::new(0.05, -0.02, 0.03);

    let (mut cpu, mut gpu) = pair(dims, vec![flags::FLUID; 10 * 8 * 8], cfg, &ctx);
    // The fast layout must actually be the one under test wherever the adapter
    // can provide it. A regression that silently dropped back to the packed
    // fallback would still pass every assertion below, and cost half the
    // throughput of the format for no visible reason.
    assert_eq!(
        gpu.ddf_buffers().fp16c_per_cell,
        ctx.caps.shader_i16,
        "FP16C picked the wrong storage layout for this adapter"
    );

    for _ in 0..200 {
        cpu.step();
    }
    gpu.step(200);
    let field = gpu.read_macroscopic().expect("readback");
    let (drho, du, at) = compare(&cpu, &field, dims);
    println!("FP16C: worst drho {drho:e}, worst du {du:e} at {at:?}");
    // Both sides quantise identically, so the only residual is the order of the
    // f32 sums in the moments. If the codecs disagreed at all, this would be
    // ~1e-4 (an ulp of the storage format), not ~1e-7.
    assert!(drho < 2e-5, "FP16C density differs by {drho:e}");
    assert!(du < 2e-5, "FP16C velocity differs by {du:e} at {at:?}");
}

/// The same comparison on the *packed* FP16C layout, which needs a device that
/// has been denied 16-bit storage to reach at all on this hardware.
///
/// The fallback carries the same populations through `atomicAnd` + `atomicOr`
/// instead of a plain 16-bit store. It is slower, and it is what runs on an
/// adapter without `SHADER_I16` — so it has to stay just as correct as the fast
/// path, and it has to be checked somewhere that a 16-bit-capable machine will
/// actually run.
#[test]
fn gpu_fp16c_packed_fallback_matches_the_rust_codec() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu_without_16bit_storage() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };
    let dims = UVec3::new(10, 8, 8);
    let mut cfg = clean_config(0.7);
    cfg.precision = DdfPrecision::Fp16c;
    cfg.periodic = [true; 3];
    cfg.macroscopic_buffer = true;
    cfg.initial_velocity = Vec3::new(0.05, -0.02, 0.03);

    let (mut cpu, mut gpu) = pair(dims, vec![flags::FLUID; 10 * 8 * 8], cfg, &ctx);
    assert!(
        !gpu.ddf_buffers().fp16c_per_cell,
        "this context was supposed to have no 16-bit storage, so the packed \
         fallback is what should have been selected"
    );

    for _ in 0..200 {
        cpu.step();
    }
    gpu.step(200);
    let field = gpu.read_macroscopic().expect("readback");
    let (drho, du, at) = compare(&cpu, &field, dims);
    println!("FP16C packed fallback: worst drho {drho:e}, worst du {du:e} at {at:?}");
    assert!(drho < 2e-5, "packed FP16C density differs by {drho:e}");
    assert!(
        du < 2e-5,
        "packed FP16C velocity differs by {du:e} at {at:?}"
    );

    // Both layouts hold two bytes a cell; only the addressable element differs.
    // If that ever stops being true the roofline model is billing the wrong
    // number of bytes for one of them.
    assert_eq!(
        gpu.ddf_buffers().bytes_per_direction,
        ad_gpu::direction_bytes(gpu.ddf_buffers().cell_count, DdfPrecision::Fp16c),
        "the fallback must not cost more memory than the fast path"
    );
}

/// Mass conservation, measured on the device.
#[test]
fn gpu_conserves_mass_in_a_closed_periodic_box() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    let dims = UVec3::new(16, 16, 16);
    let mut cfg = clean_config(0.6);
    cfg.periodic = [true; 3];
    cfg.macroscopic_buffer = true;
    cfg.initial_velocity = Vec3::new(0.06, -0.03, 0.02);

    let mask = vec![flags::FLUID; (dims.x * dims.y * dims.z) as usize];
    let mut gpu = Solver::new(&ctx, grid_of(dims), &mask, &[], cfg).expect("solver");
    let total = |f: &[[f32; 4]]| f.iter().map(|v| v[3] as f64).sum::<f64>();

    let m0 = total(&gpu.read_macroscopic().unwrap());
    let cells = (dims.x * dims.y * dims.z) as f64;
    let mut series = Vec::new();
    for _ in 0..10 {
        gpu.step(500);
        series.push((total(&gpu.read_macroscopic().unwrap()) - m0) / cells);
    }
    println!("GPU mass drift per cell over 5000 steps: {:e}", series[9]);
    let worst = series.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
    assert!(
        worst < 1e-6,
        "GPU mass drifted {worst:e} per cell; drift series {series:?}"
    );
}

/// Poiseuille on the device, including the tau-independence property. This is
/// the physics acceptance criterion re-checked on the hardware that will
/// actually produce the numbers the user sees.
#[test]
fn gpu_poiseuille_wall_is_halfway_independent_of_tau() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    const N: u32 = 12;
    const RE: f32 = 6.0;
    let dims = UVec3::new(1, N, 1);
    let mask = vec![flags::FLUID; N as usize];

    for model in [CollisionModel::Trt, CollisionModel::Bgk] {
        let mut widths = Vec::new();
        for tau in [0.51f32, 0.6, 1.0] {
            let nu = nu_of(tau);
            let u_max = RE * nu / N as f32;
            let force = force_for_u_max(N, u_max, 1.0, nu);
            let mut cfg = clean_config(tau);
            cfg.collision = model;
            cfg.periodic = [true, false, true];
            cfg.macroscopic_buffer = true;
            cfg.body_force = Vec3::new(force, 0.0, 0.0);

            let mut gpu = Solver::new(&ctx, grid_of(dims), &mask, &[], cfg).expect("solver");
            // Diffusive relaxation is h^2/nu; at tau = 0.51 that is ~4400 steps
            // per e-folding, so 120k steps is ~27 of them.
            gpu.step(120_000);
            let field = gpu.read_macroscopic().unwrap();
            let profile: Vec<f32> = (0..N as usize).map(|j| field[j][0]).collect();
            let h_eff = effective_width(fitted_u_max(&profile), force, 1.0, nu);
            println!("GPU {model:?} tau={tau}: effective width {h_eff:.6} (geometric {N})");
            widths.push(h_eff);
        }
        let spread = (widths.iter().cloned().fold(f32::MIN, f32::max)
            - widths.iter().cloned().fold(f32::MAX, f32::min))
            / N as f32;
        println!(
            "GPU {model:?}: width spread across tau {:.4}%",
            spread * 100.0
        );
        match model {
            CollisionModel::Trt => {
                assert!(
                    spread < 1e-3,
                    "GPU TRT width varied by {:.4}%",
                    spread * 100.0
                );
                for w in &widths {
                    assert!(
                        (w - N as f32).abs() / (N as f32) < 2e-3,
                        "GPU TRT wall at width {w}, not {N}"
                    );
                }
            }
            _ => assert!(
                spread > 1e-3,
                "GPU BGK should visibly drift; got {:.4}%",
                spread * 100.0
            ),
        }
    }
}

/// Throughput, reported as MLUPS and as a percentage of the memory roofline.
///
/// LBM does almost no arithmetic, so achieved bandwidth is the entire
/// performance story: a bare millisecond figure means nothing, a roofline
/// percentage says immediately whether to go looking for a layout problem.
///
/// Three configurations, not two. FP16C has two storage layouts — one `u16` per
/// cell, and two cells packed into a `u32` with atomic stores — and the second
/// one is the reason FP16C used to be *slower* than FP32 despite moving half the
/// bytes. Measuring both in the same run on the same machine is what makes the
/// comparison mean anything; quoting a number from a previous build does not.
///
/// It runs in the default `cargo test --workspace`, on a GPU that is also
/// driving a desktop, so it has to shrug off contention without loosening what
/// it guards. Two things do that. `exclusive_gpu` keeps the rest of this file
/// off the device for the duration, which removes the contention behind the
/// flaky failures. And both sides of the wall-versus-timestamp cross-check are
/// the fastest of `BATCHES` batches, so whatever load remains from outside the
/// process has to hit every batch to move either figure. Marking the test
/// `#[ignore]` would have been simpler, and would have taken the only check on
/// the profiler's per-step accounting out of the run everyone actually does;
/// verify.sh still runs it on its own for the headline numbers.
#[test]
fn throughput_at_a_realistic_grid() {
    let _gpu = exclusive_gpu();
    let Some(ctx) = harness::gpu() else { return };
    if !ctx.caps.timestamps {
        eprintln!("SKIP: adapter has no timestamp queries");
        return;
    }
    // 256 x 128 x 128 = 4.2 M cells: large enough to be bandwidth-bound and
    // small enough to fit comfortably in FP32 alongside anything else running.
    let dims = UVec3::new(256, 128, 128);
    let mask = vec![flags::FLUID; (dims.x * dims.y * dims.z) as usize];

    // The packed layout only appears on a device without 16-bit storage, so on
    // capable hardware it has to be asked for explicitly. Same adapter, same
    // roofline; only the one feature is withheld.
    let packed_ctx = harness::gpu_without_16bit_storage().map(|mut c| {
        c.caps.peak_bandwidth = ctx.caps.peak_bandwidth;
        c
    });

    let mut configs: Vec<(&str, &ad_gpu::GpuContext, DdfPrecision)> = vec![
        ("FP32", &ctx, DdfPrecision::Fp32),
        ("FP16C", &ctx, DdfPrecision::Fp16c),
    ];
    if let Some(p) = &packed_ctx {
        if p.caps.timestamps {
            configs.push(("FP16C-packed", p, DdfPrecision::Fp16c));
        }
    }

    for (name, ctx, precision) in configs {
        for wg in [64u32, 128] {
            let mut cfg = clean_config(0.55);
            cfg.precision = precision;
            cfg.workgroup_size = wg;
            cfg.smagorinsky_c = 0.11;
            cfg.initial_velocity = Vec3::new(0.05, 0.0, 0.0);
            let mut gpu = match Solver::new(ctx, grid_of(dims), &mask, &[], cfg) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP {name} wg={wg}: {e}");
                    continue;
                }
            };
            // Say which layout actually got built, so a run that quietly fell
            // back cannot be read as a measurement of the fast path.
            assert_eq!(
                gpu.ddf_buffers().fp16c_per_cell,
                name == "FP16C" && ctx.caps.shader_i16,
                "{name}: got the wrong FP16C storage layout"
            );
            const BATCH: u32 = 10;
            const BATCHES: u32 = 20;

            gpu.step(BATCH); // warm up: shader compile, first touch, clock ramp
            gpu.wait_idle();

            // Wall clock first, with profiling *off* so the measurement is of
            // batched submission with no per-call synchronisation.
            //
            // Take the *minimum* batch rather than the mean. `exclusive_gpu`
            // keeps this file's other tests off the device, but not the
            // compositor, a browser or anything else on the machine, and
            // contention only ever makes a batch slower. The mean of 20 batches
            // drifts by 4x or more under load, which is close enough to `BATCH`
            // to make a mean-based cross-check useless as a guard against the
            // accounting bug it exists to catch.
            let mut wall_ms_per_step = f64::INFINITY;
            for _ in 0..BATCHES {
                let wall = std::time::Instant::now();
                gpu.step(BATCH);
                gpu.wait_idle();
                let ms = wall.elapsed().as_secs_f64() * 1e3 / BATCH as f64;
                wall_ms_per_step = wall_ms_per_step.min(ms);
            }

            // Then the GPU timestamps. Every timed call must batch the same
            // number of steps: the profiler averages pass times, and one pass
            // holds all `BATCH` dispatches.
            gpu.set_profiling(true);
            for _ in 0..BATCHES {
                gpu.step(BATCH);
                // `step` issues the timestamp readback without waiting for it,
                // so a headless caller has to poll for the result. An
                // application gets this for free from presentation.
                gpu.wait_idle();
                gpu.collect_profiling();
            }
            let summary = gpu.profiler_summary();

            // The same statistic on the timestamp side. The profiler keeps an
            // exponential moving average seeded by the first pass -- after 20
            // passes that one still holds 38% of the weight -- and every
            // contended pass drags it up. Held against a wall-clock *minimum*,
            // that asymmetry alone failed the band below: 1.755 ms/step
            // timestamped against 0.611 on the wall, with the rest of this file
            // running alongside.
            //
            // The fastest pass comes in by rescaling the solver's own figures,
            // not by computing `min_ms / BATCH` here. `ms_per_step`, `mlups` and
            // `roofline_fraction` carry the per-step and bytes-per-pass
            // accounting under test, and all three are built on `mean_ms`, so
            // scaling by `min_ms / mean_ms` swaps the statistic and nothing
            // else. An accounting error multiplies every pass by the same
            // factor, so it passes through the minimum untouched.
            let best = gpu
                .profiler()
                .timing(ad_solver::solver::STEP_SCOPE)
                .filter(|t| t.min_ms > 0.0 && t.mean_ms > 0.0)
                .map(|t| t.min_ms / t.mean_ms);
            let ms = gpu.ms_per_step().zip(best).map(|(ms, b)| ms * b);
            let mlups = gpu.mlups().zip(best).map(|(m, b)| m / b);
            let roofline = gpu.roofline_fraction().zip(best).map(|(r, b)| r / b);

            match (mlups, roofline, ms, best) {
                (Some(m), Some(r), Some(ms), Some(b)) => println!(
                    "{name} wg={wg}: {ms:.3} ms/step ({wall_ms_per_step:.3} wall), \
                     {m:.0} MLUPS, {:.0}% of roofline  [timestamp average {:.2}x the fastest pass]",
                    r * 100.0,
                    1.0 / b
                ),
                (Some(m), None, _, _) => println!("{name} wg={wg}: {m:.0} MLUPS  [{summary}]"),
                _ => println!("{name} wg={wg}: no timing collected  [{summary}]"),
            }
            if let Some(ms) = ms {
                // The failure this guards against is an order-of-magnitude one:
                // reporting the whole pass as if it were a single step puts
                // `ms_per_step` out by exactly `BATCH` (10x). Both figures are
                // now the fastest of `BATCHES` batches, so the ratio compares two
                // quiet batches. Over three runs on an RTX 4090 it sat between
                // 0.79 and 1.12 across all six configurations, lowest for FP32,
                // whose wall batch reads 10-20% under its fastest timestamped
                // pass even on an idle GPU. The band leaves more than 2x below
                // and 3.5x above that range, and still excludes 10x and 1/10x.
                let ratio = wall_ms_per_step / ms;
                assert!(
                    (0.35..4.0).contains(&ratio),
                    "{name} wg={wg}: GPU timestamps say {ms:.3} ms/step but the wall clock \
                     says {wall_ms_per_step:.3} (ratio {ratio:.2}, both the fastest of \
                     {BATCHES} batches); the profiler accounting is wrong"
                );
            }
            if let Some(r) = roofline {
                // A regression guard, not a performance target. Alone on an
                // RTX 4090 this kernel's fastest FP32 pass reaches 92-95% of the
                // DRAM roofline.
                // Contention has measured 40% (this file's other tests running
                // alongside it) and 13% (an unrelated process on the GPU), both
                // on the moving average. So the bound is set where only a real
                // layout regression can trip it — an uncoalesced access pattern
                // or a spilled population array would land an order of magnitude
                // below the idle figure.
                //
                // A job that holds the GPU through every batch can still trip
                // it, and that is the right answer: the number would be fiction.
                if precision == DdfPrecision::Fp32 {
                    assert!(
                        r > 0.15,
                        "FP32 reached only {:.0}% of roofline on its fastest pass; \
                         if another GPU job was running, rerun this alone",
                        r * 100.0
                    );
                }
            }
        }
    }
}

/// Hot parameter updates must not need a rebuild, and must actually take effect.
#[test]
fn inlet_velocity_updates_without_a_rebuild() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    let dims = UVec3::new(8, 8, 8);
    let mut mask = vec![flags::FLUID; 8 * 8 * 8];
    for z in 0..8u32 {
        for y in 0..8u32 {
            mask[((z * 8 + y) * 8) as usize] = flags::INLET;
        }
    }
    let mut cfg = clean_config(0.6);
    cfg.macroscopic_buffer = true;
    cfg.inlet_velocity = Vec3::new(0.02, 0.0, 0.0);

    let mut gpu = Solver::new(&ctx, grid_of(dims), &mask, &[], cfg).expect("solver");
    gpu.step(100);
    let before = gpu.read_macroscopic().unwrap()[0][0];

    gpu.set_inlet_velocity(Vec3::new(0.06, 0.0, 0.0));
    gpu.step(100);
    let after = gpu.read_macroscopic().unwrap()[0][0];

    println!("inlet cell u.x: {before:.5} -> {after:.5}");
    assert!(
        (before - 0.02).abs() < 1e-4,
        "inlet did not hold its prescribed velocity"
    );
    assert!(
        (after - 0.06).abs() < 1e-4,
        "hot update did not take effect: got {after}"
    );

    // A change that *does* need a rebuild must say so rather than silently doing
    // nothing.
    let mut other = *gpu.config();
    other.precision = DdfPrecision::Fp16c;
    assert!(
        gpu.update(other).is_err(),
        "changing storage precision should demand a rebuild"
    );
}

/// `reset()` must return the solver to its initial state exactly.
#[test]
fn reset_restores_the_initial_condition() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    let dims = UVec3::new(8, 8, 8);
    let mask = vec![flags::FLUID; 8 * 8 * 8];
    let mut cfg = clean_config(0.6);
    cfg.periodic = [true; 3];
    cfg.macroscopic_buffer = true;
    cfg.initial_velocity = Vec3::new(0.04, 0.01, -0.02);

    let mut gpu = Solver::new(&ctx, grid_of(dims), &mask, &[], cfg).expect("solver");
    let initial = gpu.read_macroscopic().unwrap();
    gpu.step(137);
    assert_eq!(gpu.steps_taken(), 137);
    gpu.reset();
    assert_eq!(gpu.steps_taken(), 0);
    let after = gpu.read_macroscopic().unwrap();
    let worst = initial
        .iter()
        .zip(&after)
        .map(|(a, b)| (0..4).map(|k| (a[k] - b[k]).abs()).fold(0.0f32, f32::max))
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-6, "reset left a residual of {worst:e}");
}

/// The production boundary configuration, end to end: velocity inlet, convective
/// outlet with a graded sponge and an anti-bounce-back pressure reference, solid
/// duct walls, and open equilibrium sides beyond the duct exit.
///
/// This is the arrangement the duct case will actually run, so it is worth a
/// smoke test of its own. Three things are checked, in the order they would go
/// wrong: it stays finite, it conserves mass (flux in equals flux out), and the
/// outlet really does pin the pressure.
#[test]
fn the_production_boundary_configuration_is_stable_and_conserves_flux() {
    let _gpu = shared_gpu();
    let Some(ctx) = harness::gpu() else { return };
    // A square duct along x, opening into free space for the last third.
    let dims = UVec3::new(48, 24, 24);
    let (nx, ny, nz) = (dims.x, dims.y, dims.z);
    let at = |x: u32, y: u32, z: u32| ((z * ny + y) * nx + x) as usize;
    let duct_end = 32u32;
    let mut mask = vec![flags::FLUID; (nx * ny * nz) as usize];

    for z in 0..nz {
        for y in 0..ny {
            // Duct walls: a 6-cell-square bore through a solid block.
            for x in 0..duct_end {
                let inside = (9..15).contains(&y) && (9..15).contains(&z);
                if !inside {
                    mask[at(x, y, z)] = flags::SOLID;
                }
            }
            // Velocity inlet across the bore.
            if (9..15).contains(&y) && (9..15).contains(&z) {
                mask[at(0, y, z)] = flags::INLET;
            }
            // Convective outlet with the pressure reference, plus a sponge.
            mask[at(nx - 1, y, z)] = flags::OUTLET | flags::SPONGE;
        }
    }
    // Open box sides downstream of the duct, so the exit jet can entrain.
    for x in duct_end..nx - 1 {
        for k in 0..ny {
            mask[at(x, 0, k)] = flags::EQUILIBRIUM;
            mask[at(x, ny - 1, k)] = flags::EQUILIBRIUM;
            mask[at(x, k, 0)] = flags::EQUILIBRIUM;
            mask[at(x, k, nz - 1)] = flags::EQUILIBRIUM;
        }
    }

    // tau = 0.502 is the coarse tier's relaxation time from CONTRACT.md's
    // resolution table. The coarse tier is the stability challenge, not the fine
    // one, so that is where this runs.
    //
    // Cs = 0.17 here, not the 0.10-0.12 the contract recommends for internal
    // flow. That is a measured requirement, not a preference. Sweeping this
    // exact case over tau and Cs (16k steps each, "stable" meaning finite and
    // peaking below 8x the inlet velocity):
    //
    //   tau     Cs=0    Cs=0.11   Cs=0.17
    //   0.502    no       no        yes
    //   0.505    no       no        yes
    //   0.51     no       no        yes
    //   0.52     no      yes        yes
    //   0.53    yes      yes        yes
    //   0.55    yes      yes        yes
    //
    // So the LES model is doing exactly the job it is here for - with Cs = 0 the
    // solver cannot run the coarse tier at all - but a sharp-edged six-cell jet
    // needs more eddy viscosity than a smooth internal flow does. The contract's
    // 0.10-0.12 is right for the duct bore and not enough at the exit plane.
    let mut cfg = clean_config(0.502);
    cfg.macroscopic_buffer = true;
    cfg.smagorinsky_c = 0.17;
    cfg.inlet_velocity = Vec3::new(0.05, 0.0, 0.0);
    cfg.inlet_normal = Vec3::X;
    cfg.outflow_velocity = 0.05;
    cfg.outlet_normal = Vec3::X;
    cfg.sponge_cells = 6;
    cfg.sponge_strength = 0.4;

    let mut gpu = Solver::new(&ctx, grid_of(dims), &mask, &[], cfg).expect("solver");
    for _ in 0..8 {
        gpu.step(2_500);
    }
    let f = gpu.read_macroscopic().expect("readback");

    // 1. Finite and bounded. A fully developed square duct peaks at about 2.1x
    //    its bulk velocity, so anything far above that has diverged.
    let mut worst = 0.0f32;
    for v in &f {
        for k in 0..4 {
            assert!(v[k].is_finite(), "field went non-finite: {v:?}");
        }
        worst = worst.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
    }
    println!(
        "open duct at tau={}: peak speed {worst:.4} (inlet 0.05)",
        cfg.tau0
    );
    assert!(
        worst > 0.07,
        "peak speed {worst} is too low; the duct never got going"
    );
    assert!(
        worst < 0.25,
        "peak speed {worst} is far above the inlet velocity; this diverged"
    );

    // 2. Continuity. Integrate the streamwise flux over three planes: just after
    //    the inlet, mid-duct, and just past the duct exit. A leak in the
    //    boundary treatment shows up here long before it distorts a profile.
    let flux = |x: u32| -> f32 {
        let mut s = 0.0;
        for z in 0..nz {
            for y in 0..ny {
                let v = f[at(x, y, z)];
                s += v[0] * v[3];
            }
        }
        s
    };
    let (f_in, f_mid, f_out) = (flux(1), flux(duct_end / 2), flux(duct_end + 1));
    println!("open duct flux: inlet {f_in:.5}, mid {f_mid:.5}, past the exit {f_out:.5}");
    assert!(f_in > 0.0, "no flow entered the duct");
    // Only the two planes *inside* the duct are held to strict continuity. Past
    // the exit the box sides are open equilibrium boundaries, so mass crossing
    // them is the entrainment the contract asks for rather than a leak, and the
    // streamwise flux there is not conserved by construction. The exit number is
    // printed rather than asserted on for exactly that reason.
    assert!(
        (f_mid - f_in).abs() / f_in < 0.03,
        "streamwise flux mid-duct is {f_mid:.5} against {f_in:.5} at the inlet;          continuity inside a solid-walled duct must hold"
    );
    assert!(
        f_out > 0.5 * f_in,
        "the jet lost most of its flux one cell past the exit"
    );

    // 3. The outlet holds the reference pressure. With anti-bounce-back off (see
    //    SolverConfig::outlet_anti_bounce_back) the convective term is what does
    //    this, by relaxing the outlet plane toward f^eq(rho_ref, u) every step.
    //    If it did not, injected mass would raise the density without bound and
    //    any quoted pressure drop would be meaningless.
    let mut rho_out = 0.0f64;
    for z in 0..nz {
        for y in 0..ny {
            rho_out += f[at(nx - 1, y, z)][3] as f64;
        }
    }
    rho_out /= (ny * nz) as f64;
    println!(
        "open duct: mean outlet density {rho_out:.6} (reference {})",
        cfg.rho_ref
    );
    assert!(
        (rho_out - cfg.rho_ref as f64).abs() < 5e-3,
        "outlet density {rho_out} drifted from the reference {}",
        cfg.rho_ref
    );

    // 4. And the duct is pressurised relative to it, or there is no pressure drop
    //    to measure and the inlet is not doing its job.
    let mut rho_in = 0.0f64;
    let mut n_in = 0u32;
    for z in 9..15u32 {
        for y in 9..15u32 {
            rho_in += f[at(1, y, z)][3] as f64;
            n_in += 1;
        }
    }
    rho_in /= n_in as f64;
    println!(
        "open duct: inlet-plane density {rho_in:.6}, drop {:.6}",
        rho_in - rho_out
    );
    assert!(
        rho_in > rho_out,
        "the duct inlet ({rho_in}) is not above the outlet ({rho_out}); there is no pressure drop"
    );
}

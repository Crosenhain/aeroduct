//! CUDA validation: the four gates the wgpu backend already passes.
//!
//! These are not new physics tests. They are *the same* tests, run against the
//! second backend, because the claim being made is "CUDA implements the same
//! algorithm" and the only way to establish that is to hold it to the same bar:
//!
//! 1. **Mass conservation** in a closed periodic box over 5000 steps. The
//!    sharpest possible check on the streaming indices — any mistake either
//!    duplicates or destroys populations.
//! 2. **Poiseuille wall position independent of tau.** TRT at the magic
//!    parameter 3/16 puts the bounce-back wall exactly halfway regardless of
//!    viscosity; BGK does not. This is the sharpest correctness test in the
//!    project because it fails for a *subtly* wrong collision operator, which
//!    the other tests do not.
//! 3. **Agreement with `ad_solver::reference`**, the CPU solver. Localises a
//!    failure to "plumbing" rather than "physics" in one assertion.
//! 4. **Cross-backend agreement.** Same initial condition, same steps, CUDA
//!    against wgpu. This is the test that proves the two backends are running
//!    the same simulation rather than two plausible ones.
//!
//! Every test skips cleanly when its device is missing, and the whole file is
//! behind the `cuda` feature, so `cargo test --workspace` on a machine with no
//! toolkit does not see it at all.

// The Poiseuille gate has to estimate the effective channel width with the same
// parabola fit the wgpu test uses, or it is comparing estimators rather than
// backends. Including the harness is how the solver's own validation targets get
// at it (see the [[test]] entries in ad-solver/Cargo.toml); this is the same
// mechanism, read-only, and a rename over there fails the build loudly here
// rather than quietly diverging.
#[cfg(feature = "cuda")]
#[path = "../../../validation/lbm/harness.rs"]
mod harness;

#[cfg(feature = "cuda")]
mod cuda_gates {
    use super::harness::{self, *};
    use ad_cuda::CudaSolver;
    use ad_gpu::types::{flags, Grid, VelocitySet};
    use ad_solver::{CollisionModel, PaddedDomain, ReferenceLbm, Solver, SolverConfig};
    use glam::{UVec3, Vec3};

    fn grid_of(dims: UVec3) -> Grid {
        Grid {
            dims,
            dx_mm: 1.0,
            origin_mm: Vec3::ZERO,
        }
    }

    /// A CUDA solver, or `None` with a printed reason.
    ///
    /// Skipping rather than failing keeps this suite meaningful on a machine
    /// without an NVIDIA card, which is the same contract the wgpu validation
    /// targets honour.
    fn cuda(dims: UVec3, mask: &[u8], cfg: SolverConfig) -> Option<CudaSolver> {
        if !ad_cuda::Backend::Cuda.available() {
            eprintln!("SKIP: no CUDA device available");
            return None;
        }
        match CudaSolver::new(grid_of(dims), mask, &[], cfg) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("SKIP: CUDA solver could not be built: {e:#}");
                None
            }
        }
    }

    /// Worst absolute difference in density and velocity between a CUDA field
    /// and the CPU reference, over the interior.
    fn compare_to_reference(
        cpu: &ReferenceLbm,
        field: &[[f32; 4]],
        dims: UVec3,
    ) -> (f32, f32, UVec3) {
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

    /// Worst element-wise difference between two macroscopic fields.
    fn compare_fields(a: &[[f32; 4]], b: &[[f32; 4]], dims: UVec3) -> (f32, f32, UVec3) {
        assert_eq!(
            a.len(),
            b.len(),
            "the two backends returned different field sizes"
        );
        let mut worst_rho = 0.0f32;
        let mut worst_u = 0.0f32;
        let mut at = UVec3::ZERO;
        for (i, (p, q)) in a.iter().zip(b).enumerate() {
            worst_rho = worst_rho.max((p[3] - q[3]).abs());
            let du = (Vec3::new(p[0], p[1], p[2]) - Vec3::new(q[0], q[1], q[2])).length();
            if du > worst_u {
                worst_u = du;
                let i = i as u32;
                at = UVec3::new(i % dims.x, (i / dims.x) % dims.y, i / (dims.x * dims.y));
            }
        }
        (worst_rho, worst_u, at)
    }

    /// The L-shaped obstacle from `validation/lbm/gpu.rs`. Concave corners are
    /// where a wrong bounce-back rule is most likely to show, because a cell can
    /// have several blocked links at once.
    fn l_obstacle(dims: UVec3) -> Vec<u8> {
        let mut mask = vec![flags::FLUID; (dims.x * dims.y * dims.z) as usize];
        let at = |x: u32, y: u32, z: u32| ((z * dims.y + y) * dims.x + x) as usize;
        for z in 2..8u32 {
            for y in 4..7u32 {
                mask[at(6, y, z)] = flags::SOLID;
            }
            for x in 6..10u32 {
                mask[at(x, 6, z)] = flags::SOLID;
            }
        }
        mask
    }

    // ---------------------------------------------------------------- gate 1

    /// **Gate 1.** A closed periodic box must conserve total density. The wgpu
    /// backend drifts *exactly* zero over 5000 steps; anything else here means
    /// the Esoteric Pull addressing loses or duplicates a population, and the
    /// bound is set at zero rather than at a tolerance because a scheme that
    /// only moves values around has no mechanism by which to drift at all.
    #[test]
    fn cuda_conserves_mass_in_a_closed_periodic_box() {
        let dims = UVec3::new(16, 16, 16);
        let mut cfg = clean_config(0.6);
        cfg.periodic = [true; 3];
        cfg.initial_velocity = Vec3::new(0.06, -0.03, 0.02);
        let mask = vec![flags::FLUID; (dims.x * dims.y * dims.z) as usize];
        let Some(mut gpu) = cuda(dims, &mask, cfg) else {
            return;
        };

        let total = |f: &[[f32; 4]]| f.iter().map(|v| v[3] as f64).sum::<f64>();
        let cells = (dims.x * dims.y * dims.z) as f64;
        let m0 = total(&gpu.read_macroscopic().expect("readback"));
        let mut series = Vec::new();
        for _ in 0..10 {
            gpu.step(500).expect("step");
            series.push((total(&gpu.read_macroscopic().expect("readback")) - m0) / cells);
        }
        println!("CUDA mass drift per cell over 5000 steps: {:e}", series[9]);
        let worst = series.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        assert_eq!(
            worst, 0.0,
            "CUDA mass drifted {worst:e} per cell; the wgpu backend drifts exactly 0. \
             Drift series: {series:?}"
        );
    }

    // ---------------------------------------------------------------- gate 2

    /// **Gate 2.** TRT at Lambda = 3/16 must hold the bounce-back wall exactly
    /// halfway between the last fluid node and the halo, *independently of
    /// viscosity*. The profile stays parabolic either way; what moves is the
    /// channel width its peak implies, which is what this measures.
    ///
    /// The acceptance band is 0.01% of the geometric width. That is tighter than
    /// the wgpu target's own 0.1% guard and matches what wgpu actually measures
    /// (0.0066%), because a second backend agreeing to the loose bound would not
    /// be evidence of anything — BGK, the negative control, spreads ~0.45%.
    #[test]
    fn cuda_poiseuille_wall_is_halfway_independent_of_tau() {
        if !ad_cuda::Backend::Cuda.available() {
            eprintln!("SKIP: no CUDA device available");
            return;
        }
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
                cfg.body_force = Vec3::new(force, 0.0, 0.0);

                let Some(mut gpu) = cuda(dims, &mask, cfg) else {
                    return;
                };
                // Diffusive relaxation is h^2/nu; at tau = 0.51 that is ~4400
                // steps per e-folding, so 120k steps is ~27 of them.
                gpu.step(120_000).expect("step");
                let field = gpu.read_macroscopic().expect("readback");
                let profile: Vec<f32> = (0..N as usize).map(|j| field[j][0]).collect();
                let h_eff = effective_width(fitted_u_max(&profile), force, 1.0, nu);
                println!("CUDA {model:?} tau={tau}: effective width {h_eff:.6} (geometric {N})");
                widths.push(h_eff);
            }
            let spread = (widths.iter().cloned().fold(f32::MIN, f32::max)
                - widths.iter().cloned().fold(f32::MAX, f32::min))
                / N as f32;
            println!(
                "CUDA {model:?}: width spread across tau {:.4}%",
                spread * 100.0
            );
            match model {
                CollisionModel::Trt => {
                    assert!(
                        spread < 1e-4,
                        "CUDA TRT width varied by {:.4}%, wanted under 0.01%",
                        spread * 100.0
                    );
                    for w in &widths {
                        assert!(
                            (w - N as f32).abs() / (N as f32) < 2e-3,
                            "CUDA TRT wall at width {w}, not {N}"
                        );
                    }
                }
                // The negative control. If BGK does *not* drift, TRT is not
                // being selected and gate 2 is passing vacuously.
                _ => assert!(
                    spread > 1e-3,
                    "CUDA BGK should visibly drift; got {:.4}%",
                    spread * 100.0
                ),
            }
        }
    }

    // ---------------------------------------------------------------- gate 3

    /// **Gate 3a.** CUDA against the CPU reference in a periodic box, every
    /// velocity set and every collision operator. The wgpu backend matches to
    /// 1.7e-7 worst-case velocity; the assertion band is the wgpu target's own
    /// 5e-6, and the *printed* number is what says whether the agreement is
    /// really at f32 round-off.
    #[test]
    fn cuda_matches_the_cpu_reference_in_a_periodic_box() {
        if !ad_cuda::Backend::Cuda.available() {
            eprintln!("SKIP: no CUDA device available");
            return;
        }
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
                cfg.initial_velocity = Vec3::new(0.05, -0.03, 0.02);
                cfg.body_force = Vec3::new(1e-5, 0.0, 0.0);

                let mask = vec![flags::FLUID; (dims.x * dims.y * dims.z) as usize];
                let domain = PaddedDomain::new(dims, cfg.periodic, &mask, cfg.set);
                let mut cpu = ReferenceLbm::new(domain, cfg);
                let Some(mut gpu) = cuda(dims, &mask, cfg) else {
                    return;
                };

                for _ in 0..200 {
                    cpu.step();
                }
                gpu.step(200).expect("step");
                let field = gpu.read_macroscopic().expect("readback");
                let (drho, du, at) = compare_to_reference(&cpu, &field, dims);
                println!(
                    "CUDA {set:?}/{collision:?}: worst drho {drho:e}, worst du {du:e} at {at:?}"
                );
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

    /// **Gate 3b.** The same comparison with walls, which exercises the
    /// bounce-back parity flip and the link mask. If the transport is subtly
    /// wrong this is where it shows: the interior would still agree and only the
    /// cells next to the obstacle would drift.
    #[test]
    fn cuda_matches_the_cpu_reference_around_an_obstacle() {
        let dims = UVec3::new(14, 12, 10);
        let mask = l_obstacle(dims);
        let mut cfg = clean_config(0.65);
        cfg.periodic = [true; 3];
        cfg.initial_velocity = Vec3::new(0.06, 0.0, 0.0);
        cfg.body_force = Vec3::new(2e-5, 0.0, 0.0);

        let domain = PaddedDomain::new(dims, cfg.periodic, &mask, cfg.set);
        let mut cpu = ReferenceLbm::new(domain, cfg);
        let Some(mut gpu) = cuda(dims, &mask, cfg) else {
            return;
        };

        for _ in 0..300 {
            cpu.step();
        }
        gpu.step(300).expect("step");
        let field = gpu.read_macroscopic().expect("readback");
        let (drho, du, at) = compare_to_reference(&cpu, &field, dims);
        println!("CUDA obstacle: worst drho {drho:e}, worst du {du:e} at {at:?}");
        assert!(drho < 5e-6, "density differs by {drho:e}");
        assert!(du < 5e-6, "velocity differs by {du:e} at {at:?}");
    }

    // ---------------------------------------------------------------- gate 4

    /// **Gate 4.** The two backends, same initial condition, same step count,
    /// compared field for field.
    ///
    /// Gate 3 already establishes that each backend matches the CPU, so this is
    /// in one sense implied — but only in one sense. The CPU comparison runs at
    /// 200-300 steps on a small grid; this runs long enough for a one-ulp
    /// difference in, say, the order of a moment sum to grow visible, and it is
    /// the assertion that would actually be quoted if someone asked "are these
    /// the same simulation?". The bound is the same 5e-6, and the printed number
    /// is the answer.
    #[test]
    fn cuda_matches_the_wgpu_backend() {
        let dims = UVec3::new(14, 12, 10);
        let mask = l_obstacle(dims);
        let mut cfg = clean_config(0.65);
        cfg.periodic = [true; 3];
        cfg.macroscopic_buffer = true; // the wgpu path needs this to read back
        cfg.initial_velocity = Vec3::new(0.06, 0.0, 0.0);
        cfg.body_force = Vec3::new(2e-5, 0.0, 0.0);

        let Some(mut cu) = cuda(dims, &mask, cfg) else {
            return;
        };
        let Some(ctx) = harness::gpu() else { return };
        let mut wg = Solver::new(&ctx, grid_of(dims), &mask, &[], cfg).expect("wgpu solver");

        // Run them one after the other, never at once. This kernel is
        // bandwidth-bound; two clients on one memory controller would not change
        // the *fields*, but it is the habit that keeps the timing numbers in
        // examples/ab_bench.rs honest, and it costs nothing here.
        const STEPS: u32 = 2000;
        cu.step(STEPS).expect("cuda step");
        cu.wait_idle();
        let a = cu.read_macroscopic().expect("cuda readback");

        wg.step(STEPS);
        wg.wait_idle();
        let b = wg.read_macroscopic().expect("wgpu readback");

        let (drho, du, at) = compare_fields(&a, &b, dims);
        println!(
            "CUDA vs wgpu after {STEPS} steps: worst drho {drho:e}, worst du {du:e} at {at:?}"
        );
        assert!(drho < 5e-6, "backends disagree on density by {drho:e}");
        assert!(
            du < 5e-6,
            "backends disagree on velocity by {du:e} at {at:?}"
        );
    }

    /// The generated kernel must survive NVRTC for every configuration the
    /// solver can be asked for, not just the default. A `#if` that only compiles
    /// under TRT is a failure nobody sees until a sweep asks for BGK.
    #[test]
    fn every_configuration_compiles_on_the_device() {
        if !ad_cuda::Backend::Cuda.available() {
            eprintln!("SKIP: no CUDA device available");
            return;
        }
        let dims = UVec3::new(8, 8, 8);
        let mask = vec![flags::FLUID; 8 * 8 * 8];
        for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
            for collision in [
                CollisionModel::Trt,
                CollisionModel::Bgk,
                CollisionModel::RegularizedBgk,
            ] {
                for workgroup_size in [64u32, 128] {
                    let mut cfg = clean_config(0.6);
                    cfg.set = set;
                    cfg.collision = collision;
                    cfg.workgroup_size = workgroup_size;
                    cfg.periodic = [true; 3];
                    let s = CudaSolver::new(grid_of(dims), &mask, &[], cfg).unwrap_or_else(|e| {
                        panic!("{set:?}/{collision:?}/wg{workgroup_size}: {e:#}")
                    });
                    assert_eq!(s.kernel_spec().set, set);
                    assert_eq!(s.kernel_spec().block_x, workgroup_size);
                }
            }
        }
    }

    /// FP16C is not ported, and asking for it must say so rather than silently
    /// storing FP32 and reporting half the traffic — which would inflate the
    /// roofline percentage by 2x in exactly the measurement this crate exists
    /// to make.
    #[test]
    fn asking_for_fp16c_is_refused_rather_than_silently_downgraded() {
        let dims = UVec3::new(8, 8, 8);
        let mask = vec![flags::FLUID; 8 * 8 * 8];
        let mut cfg = clean_config(0.6);
        cfg.precision = ad_gpu::types::DdfPrecision::Fp16c;
        let msg = match CudaSolver::new(grid_of(dims), &mask, &[], cfg) {
            Ok(_) => panic!("FP16C must be refused, not silently stored as FP32"),
            Err(e) => format!("{e:#}"),
        };
        assert!(msg.contains("FP32"), "unhelpful refusal: {msg}");
    }
}

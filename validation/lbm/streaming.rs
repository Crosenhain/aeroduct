//! The two cheap invariants, written first and run first.
//!
//! Mass conservation and Galilean invariance cost seconds and catch an Esoteric
//! Pull mistake immediately â€” an index error either duplicates or loses
//! populations, and both show up here long before any physics test would notice.
//! Everything else in `validation/lbm/` assumes these pass.

#[path = "harness.rs"]
mod harness;

use ad_gpu::types::{flags, DdfPrecision, VelocitySet};
use ad_solver::{PaddedDomain, ReferenceLbm, SolverConfig};
use glam::{UVec3, Vec3};
use harness::*;

fn periodic_box(n: u32, cfg: SolverConfig) -> ReferenceLbm {
    ReferenceLbm::new(PaddedDomain::uniform_fluid(UVec3::splat(n), [true; 3], cfg.set), cfg)
}

/// Total density in a closed periodic box is conserved to round-off. Thousands
/// of steps, because a per-step leak of 1e-9 is invisible over ten.
#[test]
fn mass_is_conserved_over_thousands_of_steps() {
    for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
        let mut cfg = clean_config(0.6);
        cfg.set = set;
        cfg.initial_velocity = Vec3::new(0.06, -0.03, 0.02);
        let mut lbm = periodic_box(8, cfg);

        let cells = lbm.domain.interior_cell_count() as f64;
        let m0 = lbm.total_mass();
        let mut worst = 0.0f64;
        for k in 1..=20 {
            for _ in 0..250 {
                lbm.step();
            }
            let drift = (lbm.total_mass() - m0).abs() / cells;
            worst = worst.max(drift);
            assert!(
                drift < 1e-6,
                "{set:?}: mass drifted {drift:e} per cell after {} steps",
                k * 250
            );
        }
        println!("{set:?}: worst mass drift over 5000 steps was {worst:e} per cell");
        // A drift that grows linearly would be a leak; one that stays at the
        // f32 noise floor is round-off. 5000 steps of f32 accumulation over 512
        // cells cannot exceed this without something being wrong.
        assert!(worst < 1e-6);
    }
}

/// Bounce-back moves populations around without creating or destroying them, so
/// an obstacle must not change the answer.
#[test]
fn mass_is_conserved_with_a_wall_in_the_box() {
    let mut cfg = clean_config(0.6);
    cfg.initial_velocity = Vec3::new(0.05, 0.0, 0.0);
    let mut d = PaddedDomain::uniform_fluid(UVec3::splat(10), [true; 3], cfg.set);
    // A plate, not a cube: a flat obstacle has links of every orientation
    // terminating on it, including the diagonals that are easiest to get wrong.
    for z in 0..10u32 {
        for y in 4..6u32 {
            let i = d.linear(UVec3::new(5, y, z)) as usize;
            d.flags[i] = flags::SOLID;
        }
    }
    d.rebuild_link_mask(cfg.set.def());
    let mut lbm = ReferenceLbm::new(d, cfg);

    let cells = lbm.domain.interior_cell_count() as f64;
    let m0 = lbm.total_mass();
    for _ in 0..3000 {
        lbm.step();
    }
    let drift = (lbm.total_mass() - m0).abs() / cells;
    println!("mass drift with an obstacle: {drift:e} per cell over 3000 steps");
    assert!(drift < 1e-6, "mass drifted {drift:e} per cell around an obstacle");
}

/// Uniform flow in a periodic box is an exact solution. Any asymmetry in
/// streaming, collision or forcing breaks it, and the failure is a *pattern* in
/// the residual rather than a magnitude, so this catches things a norm would not.
#[test]
fn uniform_flow_stays_uniform_in_every_direction() {
    // Deliberately not axis-aligned: an axis-aligned velocity leaves most of the
    // diagonal links carrying equal populations, which hides pair-ordering bugs.
    let velocities = [
        Vec3::new(0.05, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -0.05),
        Vec3::new(0.04, 0.03, 0.02),
        Vec3::new(-0.03, 0.05, -0.01),
    ];
    for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
        for v in velocities {
            let mut cfg = clean_config(0.7);
            cfg.set = set;
            cfg.initial_velocity = v;
            let mut lbm = periodic_box(6, cfg);
            for _ in 0..500 {
                lbm.step();
            }
            let mut worst_u = 0.0f32;
            let mut worst_rho = 0.0f32;
            for z in 0..6u32 {
                for y in 0..6u32 {
                    for x in 0..6u32 {
                        let m = lbm.macroscopic(UVec3::new(x, y, z));
                        worst_u = worst_u.max((m.u - v).length());
                        worst_rho = worst_rho.max((m.rho - 1.0).abs());
                    }
                }
            }
            assert!(
                worst_u < 1e-5 && worst_rho < 1e-5,
                "{set:?} u={v:?}: worst velocity error {worst_u:e}, worst density error {worst_rho:e}"
            );
        }
    }
}

/// Galilean invariance proper: the *same* flow seen from a moving frame must
/// evolve the same way. A decaying shear wave carried along by a uniform drift
/// must decay at the same rate as one standing still — advection may rotate its
/// phase, but it must not change how fast the amplitude falls.
///
/// The amplitude is measured as the magnitude of the complex Fourier
/// coefficient, precisely so the phase rotation caused by the drift does not
/// masquerade as decay. (A sin-only projection would report a *negative*
/// amplitude after the wave has travelled half a wavelength, which is how this
/// test failed the first time it was run.)
#[test]
fn a_shear_wave_decays_at_the_same_rate_in_a_moving_frame() {
    let tau = 0.8f32;
    let nu = nu_of(tau);
    const N: u32 = 16;

    let k = 2.0 * std::f32::consts::PI / N as f32;
    let steps = 400u64;
    // Analytic decay rate of a sinusoidal shear wave in an incompressible fluid.
    let analytic_rate = nu * k * k;

    let mut rates = Vec::new();
    for drift in [0.0f32, 0.05] {
        let cfg = clean_config(tau);
        let mut lbm = periodic_box(N, cfg);
        let amp = 0.02f32;
        lbm.set_equilibrium(|c| (1.0, Vec3::new(drift, amp * (k * c.x as f32).sin(), 0.0)));
        let before = shear_amplitude(&lbm, k);
        for _ in 0..steps {
            lbm.step();
        }
        let after = shear_amplitude(&lbm, k);
        let rate = -(after / before).ln() / steps as f32;
        println!(
            "drift {drift}: amplitude {before:.6} -> {after:.6}, decay rate {rate:.6e}              (analytic {analytic_rate:.6e})"
        );
        rates.push(rate);
    }
    assert!(
        (rates[0] - analytic_rate).abs() / analytic_rate < 0.03,
        "shear wave decayed at {} instead of {analytic_rate}",
        rates[0]
    );
    // The residual difference is the known O(Ma^2) Galilean error of the D3Q19
    // equilibrium (the missing cubic velocity term), which at u = 0.05 is a few
    // parts in a thousand. Anything larger is a bug, not lattice physics.
    assert!(
        (rates[1] - rates[0]).abs() / rates[0] < 0.02,
        "decay rate changed with the frame velocity: {:e} vs {:e}",
        rates[1],
        rates[0]
    );
}

/// Magnitude of the `exp(i k x)` component of `u_y`.
fn shear_amplitude(lbm: &ReferenceLbm, k: f32) -> f32 {
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    let mut count = 0u64;
    for z in 0..lbm.domain.interior.z {
        for y in 0..lbm.domain.interior.y {
            for x in 0..lbm.domain.interior.x {
                let uy = lbm.macroscopic(UVec3::new(x, y, z)).u.y as f64;
                let phase = (k * x as f32) as f64;
                re += uy * phase.cos();
                im += uy * phase.sin();
                count += 1;
            }
        }
    }
    let n = count as f64;
    (2.0 * ((re / n).powi(2) + (im / n).powi(2)).sqrt()) as f32
}

/// FP16C storage must not introduce a systematic mass leak. The codec is lossy,
/// but streaming only ever moves values, so the error must stay bounded rather
/// than accumulate.
#[test]
fn fp16c_storage_does_not_leak_mass() {
    let mut cfg = clean_config(0.6);
    cfg.precision = DdfPrecision::Fp16c;
    cfg.initial_velocity = Vec3::new(0.05, -0.02, 0.01);
    let mut lbm = periodic_box(8, cfg);
    let cells = lbm.domain.interior_cell_count() as f64;
    let m0 = lbm.total_mass();
    let mut series = Vec::new();
    for _ in 0..8 {
        for _ in 0..250 {
            lbm.step();
        }
        series.push((lbm.total_mass() - m0) / cells);
    }
    println!("FP16C mass drift trajectory: {series:?}");
    let worst = series.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
    assert!(worst < 1e-4, "FP16C mass drifted {worst:e} per cell");
    // Not growing without bound: the last quarter must not be much worse than
    // the first, which is what distinguishes round-off from a leak.
    assert!(
        series[7].abs() < 4.0 * series[1].abs().max(1e-9),
        "FP16C mass error is growing: {series:?}"
    );
}

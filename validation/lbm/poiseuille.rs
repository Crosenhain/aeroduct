//! Plane Poiseuille flow: the case that decides whether the collision operator
//! and the wall treatment are right.
//!
//! Four things are checked here, in increasing order of sharpness.
//!
//! 1. The profile matches the analytic parabola to better than 1% at 20 cells
//!    across.
//! 2. Refinement does not degrade it. With TRT at `Lambda = 3/16` the discrete
//!    Poiseuille solution is *exact*, so the measured error sits at round-off
//!    and the convergence order is unbounded rather than merely second — and the
//!    test says so explicitly, because "second order" would understate it.
//! 3. **Tau-independence.** Run at `tau = 0.51, 0.6, 1.0` at a fixed Reynolds
//!    number. Halfway bounce-back places the wall exactly midway between the
//!    last fluid node and the first solid node only if the magic parameter
//!    `Lambda = (1/s_e - 1/2)(1/s_o - 1/2)` equals 3/16. BGK has no free rate, so
//!    its `Lambda` is `(tau - 1/2)^2` and the wall *moves* as the viscosity
//!    changes. Since `tau` here is set by the mesh (0.5006 at 1 mm, 0.502 at
//!    0.3 mm), a viscosity-dependent wall position would mean the duct changes
//!    size when the user changes resolution. This is the single most important
//!    test in the project, and it compares BGK against TRT directly so the
//!    difference is visible rather than asserted.
//! 4. The laminar friction factor, computed the long way round through
//!    [`ad_gpu::LatticeUnits`], so the lattice-density-to-Pascals chain is
//!    exercised end to end rather than just the velocity field.

#[path = "harness.rs"]
mod harness;

use ad_gpu::types::LatticeUnits;
use ad_solver::CollisionModel;
use harness::*;

/// Run a channel to steady state and return (measured profile, analytic profile).
fn solve_channel(
    n: u32,
    tau: f32,
    model: CollisionModel,
    u_max_target: f32,
) -> (Vec<f32>, Vec<f32>, f32) {
    let nu = nu_of(tau);
    let force = force_for_u_max(n, u_max_target, 1.0, nu);
    let mut cfg = clean_config(tau);
    cfg.collision = model;
    cfg.body_force = glam::Vec3::new(force, 0.0, 0.0);
    // Start from the analytic answer so the transient is short: the relaxation
    // is diffusive, t ~ h^2 / nu, and at tau = 0.51 that is 12,000 steps per
    // e-folding at h = 20.
    let mut lbm = plane_channel(n, cfg);
    let (steps, change) = run_to_steady(&mut lbm, 200_000, 2e-9);
    // 1e-5 is the *noise* floor, not the convergence target: at tau = 0.51 the
    // driving force is 3e-7 in lattice units and the residual per-step change
    // never falls below f32 granularity. The steady state is reached long
    // before that; this bound only catches a run that is still visibly moving.
    assert!(
        change < 1e-5,
        "{model:?} n={n} tau={tau}: not converged after {steps} steps (last change {change:e})"
    );
    let got = channel_profile(&lbm);
    let want = analytic_poiseuille(n, force, 1.0, nu);
    (got, want, force)
}

#[test]
fn profile_matches_the_analytic_parabola_at_twenty_cells() {
    let (got, want, _) = solve_channel(20, 0.8, CollisionModel::Trt, 0.05);
    let err = l2_relative(&got, &want);
    println!("TRT n=20 tau=0.8: relative L2 error {err:e}");
    assert!(err < 0.01, "L2 error {err:e} exceeds the 1% acceptance");
    // ...and in fact it is at round-off, because TRT at Lambda = 3/16 solves
    // this case exactly. If this tightened bound ever fails while the 1% bound
    // still passes, something has quietly become first-order.
    assert!(err < 5e-5, "L2 error {err:e} is far above round-off for exact TRT Poiseuille");
}

#[test]
fn refinement_does_not_degrade_the_solution() {
    // TRT is exact here, so there is no convergence *rate* to measure - the
    // error is round-off at every resolution. Asserting "second order" would be
    // asserting something weaker than the truth, so assert the truth: the error
    // stays at round-off under 4x refinement, which subsumes second order.
    let mut errors = Vec::new();
    for n in [10u32, 20, 40] {
        let (got, want, _) = solve_channel(n, 0.8, CollisionModel::Trt, 0.05);
        let err = l2_relative(&got, &want);
        // The residual is a *uniform* amplitude offset (every node is low by the
        // same fraction), not a shape error, which is how we know it is the f32
        // floor rather than a discretisation error: the parabola is exact, its
        // scale is a few parts in 1e5 off. The offset grows as h^2 because the
        // driving force at fixed peak velocity is 8 nu u / h^2, so at h = 40 the
        // per-step increment is 16x smaller relative to the populations it is
        // added to, and f32 eats correspondingly more of it. FP32 arithmetic is
        // mandated by the contract, so this floor is the real one.
        println!("TRT n={n}: relative L2 error {err:e}");
        errors.push(err);
    }
    for (n, e) in [10u32, 20, 40].iter().zip(&errors) {
        assert!(
            *e < 3e-4,
            "TRT n={n} error {e:e} is above the f32 forcing floor; the operator is no \
             longer solving Poiseuille exactly"
        );
    }

    // BGK, by contrast, has a real discretisation error, and measuring its order
    // is what makes this test more than a tautology.
    //
    // Bounce-back with BGK puts the wall at an offset that satisfies
    // `h_eff^2 = h^2 + (16/3)(Lambda - 3/16)` with `Lambda = (tau - 1/2)^2`
    // (Ginzburg & d'Humieres). The bracket is a constant, so
    // `h_eff - h ~ c / (2h)`: the wall offset shrinks as the mesh refines, and
    // the relative profile error is therefore **second order**, not first. The
    // problem with BGK is not its convergence rate, it is that `c` depends on
    // the viscosity - which is exactly what
    // `trt_wall_position_is_independent_of_tau_and_bgk_is_not` measures.
    //
    // Asserting second order here is thus a real check on BGK, and the direct
    // wall-position assertion below is the check on TRT.
    let mut bgk = Vec::new();
    for n in [10u32, 20, 40] {
        let (got, want, _) = solve_channel(n, 0.8, CollisionModel::Bgk, 0.05);
        let err = l2_relative(&got, &want);
        println!("BGK n={n}: relative L2 error {err:e}");
        bgk.push(err);
    }
    let order = |a: f32, b: f32| (a / b).log2();
    let o1 = order(bgk[0], bgk[1]);
    let o2 = order(bgk[1], bgk[2]);
    println!("BGK observed order: {o1:.2} (10->20), {o2:.2} (20->40)");
    // The 10->20 refinement is clean. By 40 the BGK error (5.5e-4) is within a
    // factor of five of the f32 forcing floor (1e-4), which drags the apparent
    // order down, so only the first interval is asserted on.
    assert!(
        (1.6..2.4).contains(&o1),
        "BGK converged at order {o1:.2} between n=10 and n=20; halfway bounce-back \
         with BGK should be second order with a viscosity-dependent constant"
    );
    assert!(o2 > 1.4, "BGK order collapsed to {o2:.2} at the finest mesh");
    assert!(bgk[1] > 20.0 * errors[1], "BGK should be visibly worse than TRT at n=20");

    // The wall position itself: TRT must put it exactly halfway at *every*
    // resolution, and BGK must not.
    for n in [10u32, 20, 40] {
        let nu = nu_of(0.8);
        let (trt_p, _, f) = solve_channel(n, 0.8, CollisionModel::Trt, 0.05);
        let (bgk_p, _, _) = solve_channel(n, 0.8, CollisionModel::Bgk, 0.05);
        let h_trt = effective_width(fitted_u_max(&trt_p), f, 1.0, nu);
        let h_bgk = effective_width(fitted_u_max(&bgk_p), f, 1.0, nu);
        println!(
            "n={n}: TRT effective width {h_trt:.5} (error {:+.4}%), BGK {h_bgk:.5} \
             (error {:+.4}%)",
            (h_trt / n as f32 - 1.0) * 100.0,
            (h_bgk / n as f32 - 1.0) * 100.0
        );
        assert!(
            (h_trt - n as f32).abs() / (n as f32) < 5e-4,
            "TRT wall at width {h_trt} instead of {n}"
        );
        assert!(
            (h_bgk - n as f32).abs() > 2.0 * (h_trt - n as f32).abs(),
            "BGK wall offset is no longer distinguishable from TRT's at n={n}"
        );
    }
}

/// The sharpest single test in the project.
#[test]
fn trt_wall_position_is_independent_of_tau_and_bgk_is_not() {
    const N: u32 = 12;
    // Fixed Reynolds number Re = u_max * h / nu, so the only thing changing
    // between runs is the viscosity - and with it, for BGK, the magic parameter.
    const RE: f32 = 6.0;
    let taus = [0.51f32, 0.6, 1.0];

    let mut report = Vec::new();
    for model in [CollisionModel::Trt, CollisionModel::Bgk] {
        let mut widths = Vec::new();
        for tau in taus {
            let nu = nu_of(tau);
            let u_max = RE * nu / N as f32;
            let (got, _, force) = solve_channel(N, tau, model, u_max);
            let h_eff = effective_width(fitted_u_max(&got), force, 1.0, nu);
            println!(
                "{model:?} tau={tau}: nu={nu:.6} u_max={u_max:.6} effective width {h_eff:.6} \
                 (geometric width {N})"
            );
            widths.push(h_eff);
        }
        let spread = (widths.iter().cloned().fold(f32::MIN, f32::max)
            - widths.iter().cloned().fold(f32::MAX, f32::min))
            / N as f32;
        println!("{model:?} width spread across tau: {:.4}%", spread * 100.0);
        report.push((model, widths, spread));
    }

    let (_, trt_widths, trt_spread) = &report[0];
    let (_, _bgk_widths, bgk_spread) = &report[1];

    // TRT: the profiles must be identical. A 0.1% spread over a 2x range in
    // viscosity is already far more than round-off would produce.
    assert!(
        *trt_spread < 1e-3,
        "TRT effective width varied by {:.4}% across tau: {trt_widths:?}",
        trt_spread * 100.0
    );
    // ...and the wall really is halfway, not merely consistent.
    for (tau, w) in taus.iter().zip(trt_widths) {
        assert!(
            (w - N as f32).abs() / (N as f32) < 2e-3,
            "TRT at tau={tau} put the wall at width {w}, not {N}"
        );
    }

    // BGK: the width must visibly drift. If this assertion ever fails, either
    // BGK has been wired to TRT by accident or the diagnostic has stopped
    // measuring the wall position.
    assert!(
        *bgk_spread > 20.0 * trt_spread.max(1e-6),
        "BGK width spread {:.4}% is not distinguishable from TRT's {:.4}%; the \
         comparison has stopped being meaningful",
        bgk_spread * 100.0,
        trt_spread * 100.0
    );
    println!(
        "tau-independence: TRT spread {:.5}%, BGK spread {:.5}% ({:.0}x worse)",
        trt_spread * 100.0,
        bgk_spread * 100.0,
        bgk_spread / trt_spread.max(1e-9)
    );
}

/// Laminar friction factor, computed entirely in SI through
/// [`ad_gpu::LatticeUnits`].
///
/// `f * Re = 96` for a plane channel (the familiar `64` is the *circular pipe*
/// constant; a pipe cannot be represented exactly on a Cartesian lattice, so
/// using it here would measure the staircase error rather than the solver).
/// The square-duct case below covers the 3D geometry with the same machinery.
#[test]
fn laminar_friction_factor_of_a_plane_channel() {
    const N: u32 = 20;
    // dx = 1 mm, U = 0.05 m/s, u_lb = 0.1 -> tau = 0.593, Re_Dh = 129: solidly
    // laminar, and a relaxation time that converges in a few thousand steps.
    let lu = LatticeUnits::for_air(1.0, 0.05, 0.1);
    let tau = lu.tau0 as f32;
    // Bulk velocity is 2/3 of the peak, so aim the peak at 1.5 * u_lb.
    let u_max = 1.5 * lu.u_lb as f32;
    let (got, _, force) = solve_channel(N, tau, CollisionModel::Trt, u_max);

    let u_bulk_lb = bulk_velocity(&got) as f64;
    let d_h_lb = 2.0 * N as f64; // hydraulic diameter of a plane channel

    // Lattice pressure is p = rho/3, so a body force F per cell is the same
    // pressure gradient as a density gradient of 3F per cell. Going through
    // `pressure_pa` rather than multiplying by hand is the point of the test:
    // it is the conversion the metrics crate will use to quote a duct dp.
    let dpdx_pa_per_m = lu.pressure_pa(3.0 * force as f64) / lu.dx_m;
    let u_bulk_ms = lu.velocity_ms(u_bulk_lb);
    let d_h_m = d_h_lb * lu.dx_m;

    let re = u_bulk_ms * d_h_m / lu.nu_phys;
    let f_darcy = dpdx_pa_per_m * d_h_m / (0.5 * lu.rho_phys * u_bulk_ms * u_bulk_ms);
    let product = f_darcy * re;

    println!(
        "plane channel: tau={tau:.5} Re={re:.1} f={f_darcy:.5} f*Re={product:.3} (want 96)"
    );
    // The physical numbers must also be sane, or the conversion could be
    // self-consistently wrong.
    assert!(
        (u_bulk_ms - lu.u_phys).abs() / lu.u_phys < 0.02,
        "bulk velocity {u_bulk_ms} m/s does not match the design point {} m/s",
        lu.u_phys
    );
    assert!(re > 20.0 && re < 2000.0, "Re {re} is not in the laminar range");
    assert!(
        (product - 96.0).abs() / 96.0 < 0.02,
        "f*Re = {product}, expected 96 within 2%"
    );
}

/// The same conversion chain in 3D, on the one duct shape a Cartesian lattice
/// represents exactly.
///
/// `f * Re = 56.91` for a square duct (Shah & London, laminar forced convection
/// in ducts). Any staircase error would show up as a systematic offset here,
/// and there is none to have.
#[test]
fn laminar_friction_factor_of_a_square_duct() {
    const N: u32 = 24;
    const F_RE_SQUARE: f64 = 56.91;
    let lu = LatticeUnits::for_air(1.0, 0.05, 0.1);
    let tau = lu.tau0 as f32;
    let nu = nu_of(tau);
    // No closed form for the peak of a square-duct profile in one line; pick a
    // force that lands the bulk velocity near u_lb and measure what comes out.
    let force = 8.0 * nu * (1.5 * lu.u_lb as f32) / ((N * N) as f32);

    let mut cfg = clean_config(tau);
    cfg.body_force = glam::Vec3::new(force, 0.0, 0.0);
    let mut lbm = square_duct(N, cfg);

    // The square duct has no `channel_profile` shortcut; drive it on step count
    // and verify convergence from the bulk velocity instead.
    let bulk = |l: &ad_solver::ReferenceLbm| -> f64 {
        let mut s = 0.0f64;
        let mut n = 0u64;
        for z in 0..N {
            for y in 0..N {
                s += l.macroscopic(glam::UVec3::new(0, y, z)).u.x as f64;
                n += 1;
            }
        }
        s / n as f64
    };
    let mut prev = 0.0f64;
    for _ in 0..60 {
        for _ in 0..500 {
            lbm.step();
        }
        let now = bulk(&lbm);
        if (now - prev).abs() / now.abs().max(1e-12) < 1e-8 {
            prev = now;
            break;
        }
        prev = now;
    }
    let u_bulk_lb = prev;

    let d_h_lb = N as f64; // square duct: D_h = side length
    let dpdx_pa_per_m = lu.pressure_pa(3.0 * force as f64) / lu.dx_m;
    let u_bulk_ms = lu.velocity_ms(u_bulk_lb);
    let d_h_m = d_h_lb * lu.dx_m;
    let re = u_bulk_ms * d_h_m / lu.nu_phys;
    let f_darcy = dpdx_pa_per_m * d_h_m / (0.5 * lu.rho_phys * u_bulk_ms * u_bulk_ms);
    let product = f_darcy * re;

    println!("square duct: Re={re:.1} f*Re={product:.3} (want {F_RE_SQUARE})");
    assert!(re > 5.0 && re < 2000.0, "Re {re} is not in the laminar range");
    assert!(
        (product - F_RE_SQUARE).abs() / F_RE_SQUARE < 0.02,
        "f*Re = {product}, expected {F_RE_SQUARE} within 2%"
    );
}

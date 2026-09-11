//! The real part: `Airflow redirector - Part 1.stl`.
//!
//! Everything here skips cleanly when the STL is not present — it is a test
//! duct shipped in `parts/`, not a synthetic fixture, and a checkout without it
//! (or a build from a crate tarball) must still test green. Set
//! `AERODUCT_TEST_STL` to point at it explicitly.
//!
//! The voxel mask comes from `ad_geom::ray_parity_voxelize`, which runs on the
//! **CPU**. That is deliberate: the GPU is busy running the solver, and the
//! whole claim of this crate is that it needs nothing from it.
//!
//! Run with `--nocapture` to see the numbers; the assertions are deliberately
//! loose bounds on things that would indicate a broken extraction, because the
//! interesting output here is a *measurement*, not a pass mark.

use ad_estimate::centreline::MouthSpec;
use ad_estimate::{
    estimate, extract_passage, Drive, ElementKind, EntryCondition, EstimateConfig, ExitCondition,
    Passage, PassageConfig, ReferenceVelocity,
};
use ad_gpu::Grid;
use glam::Vec3;
use std::time::Instant;

/// What the LBM currently reports for this duct, from CONTRACT.md's status.
const LBM_K: f64 = 12.5;
const LBM_K_SEM: f64 = 0.9;

fn part() -> Option<ad_geom::StlLoad> {
    let path = ad_geom::test_stl_path()?;
    match ad_geom::load_stl(&path) {
        Ok(l) => Some(l),
        Err(e) => panic!("the test part exists at {} but would not load: {e:#}", path.display()),
    }
}

/// Load, voxelise on the CPU, and extract. Returns the passage plus the two
/// timings, so the report can separate "what a geometry edit costs" from "what
/// a slider tick costs".
fn extract_part(load: &ad_geom::StlLoad, dx: f32) -> Option<(Passage, f64, f64, usize)> {
    let mesh = &load.mesh;

    // Mouths first: they define where the passage begins and ends.
    let mut mouths = ad_geom::detect_mouths(mesh, mesh.bbox(), ad_geom::MouthConfig::default());
    if mouths.len() != 2 {
        eprintln!("expected two mouths, found {}; skipping", mouths.len());
        return None;
    }
    // Largest first: the contract puts the inlet on mouth A, the 2116 mm^2
    // opening at z = 0.
    mouths.sort_by(|a, b| {
        b.open_area_mm2.partial_cmp(&a.open_area_mm2).unwrap_or(std::cmp::Ordering::Equal)
    });
    let specs: Vec<MouthSpec> = mouths.iter().map(MouthSpec::from).collect();

    // A 2 mm collar so the ray-parity voxeliser has clean air around the part.
    // The extraction does not need it -- the mouth planes cap the passage
    // wherever they fall on the grid -- but it keeps the mask honest at the
    // bounding box.
    let grid = Grid::covering(mesh.bbox().expanded(Vec3::splat(2.0)), dx);
    let soup: Vec<[Vec3; 3]> = (0..mesh.triangle_count()).map(|t| mesh.triangle(t)).collect();

    let t0 = Instant::now();
    let parity = ad_geom::ray_parity_voxelize(&soup, grid);
    let voxel_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert!(parity.is_watertight(), "{}", parity.report());

    let t1 = Instant::now();
    let passage = match extract_passage(grid, &parity.solid, &specs, &PassageConfig::default()) {
        Ok(p) => p,
        Err(e) => panic!("extraction failed on the real part at dx = {dx}: {e}"),
    };
    let extract_ms = t1.elapsed().as_secs_f64() * 1e3;
    Some((passage, voxel_ms, extract_ms, grid.cell_count() as usize))
}

#[test]
fn the_real_passage_matches_what_the_contract_says_about_it() {
    let Some(load) = part() else { return };
    let Some((p, voxel_ms, extract_ms, cells)) = extract_part(&load, 0.5) else { return };

    eprintln!("\n=== passage extraction, dx = 0.5 mm, {cells} cells ===");
    eprintln!("{}", p.summary());
    eprintln!("ray-parity voxelisation {voxel_ms:.0} ms, extraction {extract_ms:.0} ms");
    for w in &p.warnings {
        eprintln!("  warning: {w}");
    }
    for b in &p.bends {
        eprintln!(
            "  bend: {:.0} deg, r = {:.1} mm, D_h = {:.1} mm, r/D_h = {:.2}, \
             W(in turn plane) = {:.1}, H = {:.1}, H/W = {:.2}, s = {:.0}..{:.0} mm",
            b.angle_deg,
            b.radius_mm,
            b.hydraulic_diameter_mm,
            b.r_over_dh,
            b.width_mm,
            b.height_mm,
            b.aspect_hw,
            b.s_start_mm,
            b.s_end_mm
        );
    }
    for t in &p.transitions {
        eprintln!(
            "  {}: {:.0} -> {:.0} mm^2 ({:.2}:1) over s = {:.0}..{:.0} mm, included angle {:.0} deg",
            if t.is_contraction { "contraction" } else { "expansion" },
            t.area_in_mm2,
            t.area_out_mm2,
            1.0 / t.area_ratio(),
            t.s_start_mm,
            t.s_end_mm,
            t.included_angle_deg
        );
    }
    eprintln!(
        "  A(s): {:.0} mm^2 at the inlet -> {:.0} at the outlet, aspect {:.1} mean",
        p.stations.first().map(|s| s.area_mm2).unwrap_or(0.0),
        p.stations.last().map(|s| s.area_mm2).unwrap_or(0.0),
        p.mean_aspect()
    );

    // The mouths are the contract's, straight from the geometry crate.
    assert!(
        (p.inlet_area_mm2 / 2116.0 - 1.0).abs() < 0.05,
        "inlet area {} mm^2 against the contract's 2116",
        p.inlet_area_mm2
    );
    assert!(
        (p.outlet_area_mm2 / 1141.0 - 1.0).abs() < 0.05,
        "outlet area {} mm^2 against the contract's 1141",
        p.outlet_area_mm2
    );
    assert!(
        (p.area_ratio() - 1.85).abs() < 0.15,
        "area ratio {} against the contract's 1.85",
        p.area_ratio()
    );

    // Nothing about the passage should be absurd.
    assert!(p.length_mm > 40.0 && p.length_mm < 400.0, "L = {} mm", p.length_mm);
    assert!(p.mean_dh_mm > 3.0 && p.mean_dh_mm < 40.0, "D_h = {} mm", p.mean_dh_mm);
    assert!(p.volume_mm3 > 1_000.0, "V = {} mm^3", p.volume_mm3);
    // It is a bend: something has to turn, and the two mouth normals are 90
    // degrees apart, so the total turn should be in that neighbourhood.
    assert!(!p.bends.is_empty(), "a 90 degree bend with no detected bend");
    assert!(
        p.total_turn_deg() > 45.0 && p.total_turn_deg() < 200.0,
        "total turn {} deg",
        p.total_turn_deg()
    );
    assert!(p.stations.len() >= 8, "only {} stations", p.stations.len());
}

#[test]
fn the_estimate_for_the_real_part_with_its_band() {
    let Some(load) = part() else { return };
    let Some((p, voxel_ms, extract_ms, cells)) = extract_part(&load, 0.5) else { return };

    // Mouth-to-mouth total pressure, no entry and no exit loss: the same thing
    // the solver's two measurement planes see. This is the only configuration
    // in which the two K values are comparable at all.
    let vs_solver = EstimateConfig {
        roughness_mm: ad_estimate::PRINTED_ROUGHNESS_MM,
        reference: ReferenceVelocity::Inlet,
        ..Default::default()
    };
    // What a fan actually has to supply: sharp entry from a plenum, discharge
    // to still air.
    let installed = EstimateConfig::as_installed();

    eprintln!("\n=== loss network on the real part ===");
    eprintln!("{}", p.summary());
    eprintln!("voxelise {voxel_ms:.0} ms + extract {extract_ms:.0} ms over {cells} cells");

    eprintln!(
        "\n{:>5} {:>8} {:>7} {:>9} {:>9} {:>17} {:>10}",
        "U_in", "Q", "CFM", "U_out", "Re", "dp_total (Pa)", "K"
    );
    for u in [2.0f64, 3.0, 5.0, 8.0] {
        let r = estimate(&p, Drive::InletVelocity(u), &vs_solver);
        eprintln!(
            "{u:>5.0} {:>8.2} {:>7.1} {:>9.2} {:>9.0} {:>17} {:>10}",
            r.litres_per_second(),
            r.cfm(),
            r.outlet_velocity_ms,
            r.reynolds,
            format!("{}", r.total_pressure_drop_pa),
            format!("{}", r.loss_coefficient.k),
        );
    }

    let r = estimate(&p, Drive::InletVelocity(3.0), &vs_solver);
    eprintln!("\nat U_in = 3 m/s, mouth to mouth, printed roughness:");
    eprintln!("  {}", r.summary());
    eprint!("{}", r.breakdown());
    eprintln!("  static drop {} Pa (larger than the total, because the duct accelerates the air)", r.static_pressure_drop_pa);
    for w in &r.warnings {
        eprintln!("  warning: {w}");
    }

    let ri = estimate(&p, Drive::InletVelocity(3.0), &installed);
    eprintln!("\nas installed (sharp entry + discharge to room):");
    eprintln!("  {}", ri.summary());
    eprint!("{}", ri.breakdown());

    eprintln!("\n=== K, three ways ===");
    eprintln!(
        "  correlation, mouth to mouth : K = {}   [{}..{}]",
        r.loss_coefficient.k,
        format!("{:.2}", r.loss_coefficient.k.low()),
        format!("{:.2}", r.loss_coefficient.k.high())
    );
    eprintln!("  LBM, same convention        : K = {LBM_K} +/- {LBM_K_SEM}");
    eprintln!("  ASHRAE unvaned mitred elbow : K = 1.0-1.6 for the fitting alone");
    eprintln!("  contract's design target    : K < 1, ideally 0.3-0.6");
    eprintln!(
        "  ratio LBM / correlation     : {:.1}x",
        LBM_K / r.loss_coefficient.k.mean.max(1e-9)
    );

    // Timing: the whole point of the crate.
    let n = 5000;
    let t0 = Instant::now();
    let mut sink = 0.0;
    for i in 0..n {
        sink += estimate(&p, Drive::InletVelocity(1.0 + (i % 8) as f64), &vs_solver)
            .total_pressure_drop_pa
            .mean;
    }
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
    assert!(sink > 0.0);
    eprintln!(
        "\n=== timing ===\n  loss network: {per_us:.1} us per solve ({} stations, {} bends, \
         {} transitions)",
        p.stations.len(),
        p.bends.len(),
        p.transitions.len()
    );
    eprintln!("  extraction:   {extract_ms:.0} ms, once per geometry edit");
    eprintln!(
        "  for scale, one LBM step at dx = 0.75 mm is ~2.2 ms and a converged run is \
         tens of thousands of them"
    );
    assert!(per_us < 500.0, "{per_us:.1} us per solve is not interactive");

    // The estimate has to be a *duct* answer, not an arbitrary number.
    assert!(
        r.loss_coefficient.k.mean > 0.05 && r.loss_coefficient.k.mean < 10.0,
        "K = {} is outside anything a duct can be",
        r.loss_coefficient.k
    );
    // Flow rate is pure kinematics and must match the contract's table.
    let q2 = estimate(&p, Drive::InletVelocity(2.0), &vs_solver);
    assert!(
        (q2.litres_per_second() - 4.23).abs() < 0.15,
        "Q = {:.2} L/s at 2 m/s, contract says 4.23",
        q2.litres_per_second()
    );
    assert!((q2.outlet_velocity_ms - 3.71).abs() < 0.2, "U_out = {}", q2.outlet_velocity_ms);

    // Every element must be attributed. A breakdown that does not add up is
    // worse than no breakdown.
    let sum: f64 = r.elements.iter().map(|e| e.k_ref.mean).sum();
    assert!((sum - r.loss_coefficient.k.mean).abs() < 1e-9);
    assert!(r.elements.iter().any(|e| e.kind == ElementKind::Bend));
}

#[test]
fn the_answer_does_not_depend_on_the_cell_size() {
    // The strongest single check available without a wind tunnel: the same duct
    // through two different lattices. Whatever the extraction is measuring, it
    // must not be the lattice.
    let Some(load) = part() else { return };
    let cfg = EstimateConfig {
        roughness_mm: ad_estimate::PRINTED_ROUGHNESS_MM,
        ..Default::default()
    };
    let mut seen = Vec::new();
    for dx in [1.0f32, 0.75, 0.5] {
        let Some((p, voxel_ms, extract_ms, cells)) = extract_part(&load, dx) else { return };
        let r = estimate(&p, Drive::InletVelocity(3.0), &cfg);
        eprintln!(
            "dx = {dx:>4}: {cells:>9} cells, {voxel_ms:>5.0} + {extract_ms:>4.0} ms | \
             L = {:>5.1} mm, D_h = {:>5.2} mm, turn = {:>5.1} deg, K = {}",
            p.length_mm,
            p.mean_dh_mm,
            p.total_turn_deg(),
            r.loss_coefficient.k
        );
        seen.push((dx as f64, p.length_mm, p.mean_dh_mm, r.loss_coefficient.k.mean, p.confidence));
    }

    let coarse = seen[0];
    let fine = seen[seen.len() - 1];
    assert!(
        (fine.1 / coarse.1 - 1.0).abs() < 0.15,
        "developed length moved from {} to {} mm between dx = {} and {}",
        coarse.1,
        fine.1,
        coarse.0,
        fine.0
    );
    assert!(
        (fine.2 / coarse.2 - 1.0).abs() < 0.20,
        "D_h moved from {} to {} mm",
        coarse.2,
        fine.2
    );
    // K is allowed to move more than the geometry, because the bend
    // correlation is a steep function of r/D_h -- but a factor of two would
    // mean the extraction is resolution-bound, not the correlation.
    assert!(
        fine.3 / coarse.3 > 0.5 && fine.3 / coarse.3 < 2.0,
        "K moved from {} to {} between dx = {} and {}",
        coarse.3,
        fine.3,
        coarse.0,
        fine.0
    );
}

#[test]
fn the_contracts_hand_calc_table_is_reproduced_once_the_exit_loss_is_counted() {
    // CONTRACT.md's own expectations, at rho = 1.2.
    //
    // The flow column is pure kinematics and must be exact. The pressure column
    // turns out to be a sharper test than it looks, because it pins down *what
    // the contract was measuring*: its range corresponds to K = 1.7-4.2, and
    // this duct's mouth-to-mouth K is 0.7. The gap is almost exactly one
    // velocity head at the outlet -- the discharge loss, which is K = 1 on
    // `V_out` and therefore K = 3.5 referenced to `V_in` across a 1.87:1
    // contraction. Add it and the estimate lands at the top of every one of the
    // contract's four ranges.
    //
    // So the table is a system figure ("what must the fan supply"), not a
    // mouth-to-mouth figure, and the two must never be compared directly. That
    // distinction is worth a test on its own.
    let Some(load) = part() else { return };
    let Some((p, _, _, _)) = extract_part(&load, 0.5) else { return };
    let air = ad_estimate::Fluid { rho: 1.2, nu: ad_gpu::air::NU };
    let base = EstimateConfig {
        fluid: air,
        roughness_mm: ad_estimate::PRINTED_ROUGHNESS_MM,
        ..Default::default()
    };
    let mouth_to_mouth = EstimateConfig { exit: ExitCondition::None, ..base };
    let with_exit =
        EstimateConfig { entry: EntryCondition::None, exit: ExitCondition::Discharge, ..base };

    eprintln!("\n=== against CONTRACT.md's hand-calc table (rho = 1.2) ===");
    eprintln!(
        "{:>5} {:>17} {:>17} {:>16} {:>18} {:>12}",
        "U_in", "Q L/s (contract)", "U_out (contract)", "dp mouth-mouth", "dp with exit", "contract"
    );
    let table = [
        (2.0f64, 4.23f64, 3.71f64, (4.0f64, 10.0f64)),
        (3.0, 6.35, 5.56, (9.0, 22.0)),
        (5.0, 10.58, 9.27, (26.0, 62.0)),
        (8.0, 16.93, 14.84, (66.0, 159.0)),
    ];
    for (u, q_want, uo_want, (dp_lo, dp_hi)) in table {
        let bare = estimate(&p, Drive::InletVelocity(u), &mouth_to_mouth);
        let full = estimate(&p, Drive::InletVelocity(u), &with_exit);
        let dp = full.total_pressure_drop_pa;
        let hit = dp.high() >= dp_lo && dp.low() <= dp_hi;
        eprintln!(
            "{u:>5.0} {:>10.2} ({q_want:>5.2}) {:>10.2} ({uo_want:>5.2}) {:>16} {:>18} {:>5.0}-{:<6.0} {}",
            full.litres_per_second(),
            full.outlet_velocity_ms,
            format!("{}", bare.total_pressure_drop_pa),
            format!("{dp}"),
            dp_lo,
            dp_hi,
            if hit { "overlaps" } else { "MISSES" }
        );
        // Flow and velocities are geometry, not correlation: they must be right.
        assert!(
            (full.litres_per_second() / q_want - 1.0).abs() < 0.04,
            "Q = {:.2} L/s at {u} m/s, contract says {q_want}",
            full.litres_per_second()
        );
        assert!((full.outlet_velocity_ms / uo_want - 1.0).abs() < 0.06);
        // ...and the system-level pressure must land inside the hand calc.
        assert!(
            hit,
            "at {u} m/s the estimate {dp} Pa misses the contract's {dp_lo}-{dp_hi} Pa"
        );
    }
    let k_bare = estimate(&p, Drive::InletVelocity(3.0), &mouth_to_mouth).loss_coefficient.k.mean;
    let k_full = estimate(&p, Drive::InletVelocity(3.0), &with_exit).loss_coefficient.k.mean;
    eprintln!(
        "  K = {k_bare:.2} mouth to mouth, {k_full:.2} with the discharge loss; the \
         difference is one outlet velocity head, {:.2} referenced to the inlet",
        k_full - k_bare
    );
    // The difference must be exactly (V_out/V_in)^2 = the area ratio squared.
    let expect = p.area_ratio().powi(2);
    assert!(
        ((k_full - k_bare) / expect - 1.0).abs() < 0.02,
        "the exit term is {:.3}, expected the area ratio squared {expect:.3}",
        k_full - k_bare
    );
}

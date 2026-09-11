//! Validation against geometry whose answer is known in closed form.
//!
//! Two independent things are being checked and they are kept apart on purpose:
//!
//! 1. **The correlations**, against the analytic friction laws and the handbook
//!    fitting coefficients. Those live in the unit tests inside `loss.rs`,
//!    where they belong.
//! 2. **The extraction**, against voxel masks built from shapes whose `L`,
//!    `A(s)`, `D_h` and `r/D_h` are known exactly. That is what this file does.
//!
//! The masks here are built by evaluating a predicate at every cell centre, so
//! they contain no geometry-crate code at all: if `ad-geom` and this crate ever
//! disagree about what a voxel is, these tests still hold one of them to the
//! analytic answer.
//!
//! Every duct below is defined so that the passage boundaries fall exactly on
//! cell faces. That is not making it easy — it removes the discretisation error
//! from the *reference*, so any error that shows up belongs to the extraction.

use ad_estimate::centreline::{MouthSpec, PassageError};
use ad_estimate::loss::{friction_factor, Section};
use ad_estimate::{
    estimate, extract_passage, Drive, ElementKind, EstimateConfig, Passage, PassageConfig,
};
use ad_gpu::{Bbox, Grid};
use glam::{UVec3, Vec3};

/// Build a solid mask from a passage predicate: everything that is not air is
/// wall. A block with a hole drilled through it, which is the simplest object
/// that has a passage at all.
fn mask(grid: Grid, is_air: impl Fn(Vec3) -> bool) -> Vec<bool> {
    let mut out = vec![true; grid.cell_count() as usize];
    for z in 0..grid.dims.z {
        for y in 0..grid.dims.y {
            for x in 0..grid.dims.x {
                let c = UVec3::new(x, y, z);
                out[grid.linear(c) as usize] = !is_air(grid.cell_center_mm(c));
            }
        }
    }
    out
}

/// A rectangular opening as a mouth: four corners wound in the plane.
fn rect_mouth(centre: Vec3, normal: Vec3, u: Vec3, v: Vec3, half_u: f32, half_v: f32) -> MouthSpec {
    MouthSpec {
        center_mm: centre,
        normal,
        open_area_mm2: 4.0 * half_u * half_v,
        boundary: vec![
            centre - u * half_u - v * half_v,
            centre + u * half_u - v * half_v,
            centre + u * half_u + v * half_v,
            centre - u * half_u + v * half_v,
        ],
    }
}

/// A straight rectangular duct along +Z, `w` by `h`, `length` long, with 2 mm
/// of wall around it. Dimensions are chosen so the passage walls land on cell
/// faces at `dx = 0.5`.
fn straight_duct(w: f32, h: f32, length: f32, dx: f32) -> (Grid, Vec<bool>, Vec<MouthSpec>) {
    let wall = 2.0;
    let bbox = Bbox {
        min: Vec3::new(-w / 2.0 - wall, -h / 2.0 - wall, 0.0),
        max: Vec3::new(w / 2.0 + wall, h / 2.0 + wall, length),
    };
    let grid = Grid::covering(bbox, dx);
    let solid = mask(grid, |p| p.x.abs() <= w / 2.0 && p.y.abs() <= h / 2.0);
    let mouths = vec![
        rect_mouth(Vec3::ZERO, Vec3::Z, Vec3::X, Vec3::Y, w / 2.0, h / 2.0),
        rect_mouth(
            Vec3::new(0.0, 0.0, length),
            -Vec3::Z,
            Vec3::X,
            Vec3::Y,
            w / 2.0,
            h / 2.0,
        ),
    ];
    (grid, solid, mouths)
}

/// A 90-degree bend of centreline radius `r`, turning in the YZ plane from the
/// `z = 0` face to the `y = 0` face. Section is `w` radially (so `w` is the
/// dimension **in the plane of the turn**) by `h` along X.
fn quarter_bend(r: f32, w: f32, h: f32, dx: f32) -> (Grid, Vec<bool>, Vec<MouthSpec>) {
    let outer = r + w / 2.0;
    let bbox = Bbox {
        min: Vec3::new(-h / 2.0 - 2.0, 0.0, 0.0),
        max: Vec3::new(h / 2.0 + 2.0, outer + 2.0, outer + 2.0),
    };
    let grid = Grid::covering(bbox, dx);
    let solid = mask(grid, |p| {
        let radius = (p.y * p.y + p.z * p.z).sqrt();
        p.x.abs() <= h / 2.0 && p.y >= 0.0 && p.z >= 0.0 && (radius - r).abs() <= w / 2.0
    });
    let mouths = vec![
        // Inlet on z = 0, flow entering along +Z, section spanning y = r +/- w/2.
        rect_mouth(Vec3::new(0.0, r, 0.0), Vec3::Z, Vec3::X, Vec3::Y, h / 2.0, w / 2.0),
        // Outlet on y = 0, flow leaving along -Y.
        rect_mouth(Vec3::new(0.0, 0.0, r), Vec3::Y, Vec3::X, Vec3::Z, h / 2.0, w / 2.0),
    ];
    (grid, solid, mouths)
}

/// A duct that tapers linearly from `w0` to `w1` at constant height.
fn tapered_duct(
    w0: f32,
    w1: f32,
    h: f32,
    length: f32,
    dx: f32,
) -> (Grid, Vec<bool>, Vec<MouthSpec>) {
    let wall = 2.0;
    let wmax = w0.max(w1);
    let bbox = Bbox {
        min: Vec3::new(-wmax / 2.0 - wall, -h / 2.0 - wall, 0.0),
        max: Vec3::new(wmax / 2.0 + wall, h / 2.0 + wall, length),
    };
    let grid = Grid::covering(bbox, dx);
    let solid = mask(grid, |p| {
        let t = (p.z / length).clamp(0.0, 1.0);
        let w = w0 + (w1 - w0) * t;
        p.x.abs() <= w / 2.0 && p.y.abs() <= h / 2.0
    });
    let mouths = vec![
        rect_mouth(Vec3::ZERO, Vec3::Z, Vec3::X, Vec3::Y, w0 / 2.0, h / 2.0),
        rect_mouth(Vec3::new(0.0, 0.0, length), -Vec3::Z, Vec3::X, Vec3::Y, w1 / 2.0, h / 2.0),
    ];
    (grid, solid, mouths)
}

fn extract(grid: Grid, solid: &[bool], mouths: &[MouthSpec]) -> Passage {
    match extract_passage(grid, solid, mouths, &PassageConfig::default()) {
        Ok(p) => p,
        Err(e) => panic!("extraction failed on synthetic geometry: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Extraction against known geometry
// ---------------------------------------------------------------------------

#[test]
fn a_straight_rectangular_duct_is_recovered_exactly() {
    // 20 x 12 mm, 100 mm long, at dx = 0.5. Every boundary lands on a cell
    // face, so the *correct* answers are exactly 20, 12, 15 (= 4A/P) and 100.
    let (grid, solid, mouths) = straight_duct(20.0, 12.0, 100.0, 0.5);
    let p = extract(grid, &solid, &mouths);
    eprintln!("{}", p.summary());
    for w in &p.warnings {
        eprintln!("  warning: {w}");
    }

    assert!(p.warnings.is_empty(), "a perfect duct should produce no warnings: {:?}", p.warnings);
    assert!((p.length_mm - 100.0).abs() < 0.6, "L = {} mm, expected 100", p.length_mm);
    assert!(
        (p.volume_mm3 / (20.0 * 12.0 * 100.0) - 1.0).abs() < 0.02,
        "V = {} mm^3, expected 24000",
        p.volume_mm3
    );
    assert!(p.dead_volume_mm3 < 0.01 * p.volume_mm3, "a straight duct has no dead volume");
    assert!((p.mean_dh_mm - 15.0).abs() < 0.15, "D_h = {} mm, expected 15", p.mean_dh_mm);
    assert!((p.mean_aspect() - 20.0 / 12.0).abs() < 0.02, "aspect = {}", p.mean_aspect());
    assert!(p.bends.is_empty(), "a straight duct has no bends, found {:?}", p.bends);
    assert!(p.transitions.is_empty(), "constant section, found {:?}", p.transitions);
    assert!((p.area_ratio() - 1.0).abs() < 1e-6);

    // Section, station by station. The interior is held to 1%: on a duct of
    // constant section there is nothing for A(s) to do but sit still, and any
    // ripple there is the band quantisation leaking through. The first and
    // last bands are allowed 5%, because their splat kernel is truncated by the
    // mouth plane -- a real edge effect, not a bug, and an order of magnitude
    // inside the band the method carries.
    let n = p.stations.len();
    for (i, st) in p.stations.iter().enumerate() {
        let tol = if i < 2 || i + 2 >= n { 12.0 } else { 5.0 };
        assert!(
            (st.area_mm2 - 240.0).abs() < tol,
            "A = {} at s = {} (station {i} of {n})",
            st.area_mm2,
            st.s_mm
        );
        assert!((st.width_mm - 20.0).abs() < 0.5, "W = {}", st.width_mm);
        assert!((st.height_mm - 12.0).abs() < 0.5, "H = {}", st.height_mm);
        assert!(st.curvature_per_mm < 1e-3, "curvature {} on a straight duct", st.curvature_per_mm);
        // The long side of the section is along X by construction.
        assert!(st.major_axis.x.abs() > 0.99, "major axis {:?}", st.major_axis);
    }
    // The band spans must partition the centreline, or the friction integral
    // is over the wrong length.
    let span: f64 = p.stations.iter().map(|s| s.span_mm).sum();
    assert!((span - p.length_mm).abs() < 1e-9, "spans sum to {span}, L = {}", p.length_mm);
    // ...and integral A ds must return the volume it came from.
    let integral: f64 = p.stations.iter().map(|s| s.area_mm2 * s.span_mm).sum();
    assert!(
        (integral / p.volume_mm3 - 1.0).abs() < 1e-9,
        "integral A ds = {integral} against V = {}",
        p.volume_mm3
    );
}

#[test]
fn the_extraction_is_insensitive_to_cell_size() {
    // The same duct at three resolutions. If the extracted D_h drifted with dx
    // the method would be measuring the lattice, not the duct.
    let mut seen = Vec::new();
    for dx in [1.0f32, 0.5, 0.25] {
        let (grid, solid, mouths) = straight_duct(20.0, 12.0, 80.0, dx);
        let p = extract(grid, &solid, &mouths);
        eprintln!("dx = {dx}: {}", p.summary());
        seen.push((p.length_mm, p.mean_dh_mm, p.mean_aspect()));
    }
    for (l, dh, aspect) in &seen {
        assert!((l - 80.0).abs() < 1.0, "L = {l}");
        assert!((dh - 15.0).abs() < 0.25, "D_h = {dh}");
        assert!((aspect - 20.0 / 12.0).abs() < 0.03, "aspect = {aspect}");
    }
}

#[test]
fn a_ninety_degree_bend_recovers_its_angle_radius_and_orientation() {
    // r = 24 mm, section 8 mm radially by 16 mm across, so D_h = 10.67 mm and
    // r/D_h = 2.25 by hand. The section is deliberately NOT square: the
    // extraction has to work out which of 8 and 16 lies in the plane of the
    // turn, and getting that backwards changes a bend correlation by 30%.
    let (grid, solid, mouths) = quarter_bend(24.0, 8.0, 16.0, 0.5);
    let p = extract(grid, &solid, &mouths);
    eprintln!("{}", p.summary());
    for b in &p.bends {
        eprintln!("  bend {b:?}");
    }

    let dh_exact = 2.0 * 8.0 * 16.0 / (8.0 + 16.0);
    assert!((p.mean_dh_mm - dh_exact).abs() < 0.4, "D_h = {} against {dh_exact}", p.mean_dh_mm);
    // Arc length of the centreline. The wavefront centroid sits a little
    // outside the nominal radius because there is more volume at larger radius,
    // so the developed length comes out slightly long; 5% covers it.
    let arc = 24.0 * std::f64::consts::FRAC_PI_2;
    assert!(
        (p.length_mm / arc - 1.0).abs() < 0.06,
        "L = {} mm against pi/2 * 24 = {arc}",
        p.length_mm
    );

    assert_eq!(p.bends.len(), 1, "expected exactly one bend, got {}", p.bends.len());
    let b = p.bends[0];
    assert!((b.angle_deg - 90.0).abs() < 10.0, "turn = {} deg", b.angle_deg);
    assert!((b.radius_mm / 24.0 - 1.0).abs() < 0.10, "r = {} mm", b.radius_mm);
    assert!((b.r_over_dh - 2.25).abs() < 0.35, "r/D_h = {}", b.r_over_dh);
    // The turn happens in the YZ plane, so the 8 mm dimension is `W`.
    assert!((b.width_mm - 8.0).abs() < 0.6, "in-plane W = {} mm, expected 8", b.width_mm);
    assert!((b.height_mm - 16.0).abs() < 0.8, "out-of-plane H = {} mm, expected 16", b.height_mm);
    assert!((b.aspect_hw - 2.0).abs() < 0.25, "H/W = {}", b.aspect_hw);
    // Constant section around the turn: no imaginary transitions.
    assert!(
        p.transitions.is_empty(),
        "a constant-section bend has no area change, found {:?}",
        p.transitions
    );
}

#[test]
fn a_tighter_bend_is_reported_as_a_tighter_bend() {
    // Sweep the radius and check `r/D_h` tracks it. A bend detector that
    // returned a plausible constant would pass every single-case test.
    let mut last = 0.0;
    for r in [12.0f32, 18.0, 24.0, 36.0] {
        let (grid, solid, mouths) = quarter_bend(r, 8.0, 8.0, 0.5);
        let p = extract(grid, &solid, &mouths);
        eprintln!("r = {r}: {}", p.summary());
        assert_eq!(p.bends.len(), 1, "r = {r}: {} bends", p.bends.len());
        let got = p.bends[0].r_over_dh;
        let want = r as f64 / 8.0;
        eprintln!("r = {r}: r/D_h = {got:.2} against {want:.2}");
        assert!((got / want - 1.0).abs() < 0.15, "r = {r}: r/D_h = {got} against {want}");
        assert!(got > last, "r/D_h did not increase with radius");
        last = got;
    }
}

#[test]
fn a_taper_is_found_once_with_the_right_ratio_and_angle() {
    // 20 x 12 down to 10 x 12 over 60 mm: a 2:1 area contraction with an
    // included angle of 2 atan(5.12/120) = 4.9 degrees by hand.
    let (grid, solid, mouths) = tapered_duct(20.0, 10.0, 12.0, 60.0, 0.5);
    let p = extract(grid, &solid, &mouths);
    eprintln!("{}", p.summary());
    for t in &p.transitions {
        eprintln!("  transition {t:?}");
    }

    assert!((p.area_ratio() - 2.0).abs() < 0.02, "A_in/A_out = {}", p.area_ratio());
    assert_eq!(p.transitions.len(), 1, "expected one transition, got {:?}", p.transitions);
    let t = p.transitions[0];
    assert!(t.is_contraction, "the duct narrows");
    assert!((t.area_ratio() - 0.5).abs() < 0.06, "area ratio {}", t.area_ratio());
    assert!(
        t.included_angle_deg > 3.0 && t.included_angle_deg < 8.0,
        "included angle {} deg, hand-calc 4.9",
        t.included_angle_deg
    );
    // It runs essentially the whole duct, so the contraction is reported near
    // the middle.
    let at = p.contraction_location().expect("a taper has a location");
    assert!((at - 0.5).abs() < 0.15, "contraction located at {at} of the length");
    assert!(p.bends.is_empty());

    // The area profile must fall monotonically through the interior, not
    // wander. The two stations at each end straddle a mouth plane, where the
    // splat kernel is clipped and the area carries a few percent of edge
    // effect.
    let n = p.stations.len();
    for w in p.stations[2..n - 2].windows(2) {
        assert!(
            w[1].area_mm2 < w[0].area_mm2 + 2.5,
            "area rose from {} to {}",
            w[0].area_mm2,
            w[1].area_mm2
        );
    }
}

// ---------------------------------------------------------------------------
// Degrading honestly
// ---------------------------------------------------------------------------

#[test]
fn a_blocked_duct_is_reported_as_blocked_rather_than_estimated() {
    let (grid, mut solid, mouths) = straight_duct(20.0, 12.0, 100.0, 0.5);
    // Weld the passage shut halfway along.
    for z in 0..grid.dims.z {
        let zc = grid.cell_center_mm(UVec3::new(0, 0, z)).z;
        if (zc - 50.0).abs() < 2.0 {
            for y in 0..grid.dims.y {
                for x in 0..grid.dims.x {
                    solid[grid.linear(UVec3::new(x, y, z)) as usize] = true;
                }
            }
        }
    }
    let err = extract_passage(grid, &solid, &mouths, &PassageConfig::default())
        .expect_err("a blocked duct must not produce an estimate");
    assert!(matches!(err, PassageError::Disconnected { .. }), "{err:?}");
    // ...and the message has to be something a user can act on.
    let text = err.to_string();
    assert!(text.contains("not connected"), "{text}");
    assert!(text.contains("dx"), "the message should suggest the resolution: {text}");
}

#[test]
fn every_malformed_input_is_a_named_diagnosis() {
    let (grid, solid, mouths) = straight_duct(20.0, 12.0, 40.0, 1.0);
    let cfg = PassageConfig::default();

    assert!(matches!(
        extract_passage(grid, &solid[..10], &mouths, &cfg),
        Err(PassageError::MaskSize { .. })
    ));
    assert!(matches!(
        extract_passage(grid, &solid, &mouths[..1], &cfg),
        Err(PassageError::NeedTwoMouths { have: 1 })
    ));
    assert!(matches!(
        extract_passage(grid, &solid, &mouths, &PassageConfig { outlet: 0, ..cfg }),
        Err(PassageError::SameMouth)
    ));
    assert!(matches!(
        extract_passage(grid, &solid, &mouths, &PassageConfig { outlet: 7, ..cfg }),
        Err(PassageError::NeedTwoMouths { .. })
    ));
    // A mouth over solid material: nothing behind it.
    let all_solid = vec![true; solid.len()];
    assert!(matches!(
        extract_passage(grid, &all_solid, &mouths, &cfg),
        Err(PassageError::InletBlocked)
    ));
    // Every diagnosis must render as a sentence.
    for e in [
        PassageError::MaskSize { expected: 1, got: 2 },
        PassageError::NeedTwoMouths { have: 0 },
        PassageError::SameMouth,
        PassageError::InletBlocked,
        PassageError::Disconnected { filled_cells: 3 },
        PassageError::TooSmall { cells: 1, length_mm: 0.5 },
    ] {
        assert!(e.to_string().len() > 20, "{e:?} has no message");
    }
}

#[test]
fn an_under_resolved_passage_says_so_instead_of_guessing() {
    // A 3 mm duct at dx = 1 mm is three cells across. The extraction still
    // returns something -- it must, or a coarse interactive grid would show
    // nothing at all -- but it has to widen its own error bar and say why.
    let (grid, solid, mouths) = straight_duct(3.0, 3.0, 40.0, 1.0);
    let p = extract(grid, &solid, &mouths);
    eprintln!("{}", p.summary());
    for w in &p.warnings {
        eprintln!("  warning: {w}");
    }
    assert!(p.min_cells_across < 4.0);
    assert_ne!(p.confidence, ad_estimate::Confidence::Good);
    assert!(
        p.warnings.iter().any(|w| w.contains("cells across")),
        "no resolution warning: {:?}",
        p.warnings
    );
    // ...and that must reach the report as a wider band.
    let coarse = estimate(&p, Drive::InletVelocity(4.0), &EstimateConfig::default());
    let (fine_grid, fine_solid, fine_mouths) = straight_duct(3.0, 3.0, 40.0, 0.25);
    let fine = extract(fine_grid, &fine_solid, &fine_mouths);
    let fine_report = estimate(&fine, Drive::InletVelocity(4.0), &EstimateConfig::default());
    assert_eq!(fine.confidence, ad_estimate::Confidence::Good);
    assert!(
        coarse.loss_coefficient.k.relative_sigma()
            > fine_report.loss_coefficient.k.relative_sigma(),
        "the coarse extraction should report a wider band"
    );
}

// ---------------------------------------------------------------------------
// Extraction and network together, against the analytic answer
// ---------------------------------------------------------------------------

#[test]
fn a_voxelised_straight_duct_reproduces_darcy_weisbach() {
    // End to end: voxels in, pressure drop out, compared against
    // f (L/D_h) 1/2 rho V^2 computed by hand from the *nominal* geometry.
    // Nothing is allowed to hide here -- if the extraction reads D_h 5% low,
    // this fails.
    let (grid, solid, mouths) = straight_duct(20.0, 12.0, 100.0, 0.5);
    let p = extract(grid, &solid, &mouths);
    let cfg = EstimateConfig::default();
    let (l_m, dh_m, area_m2) = (0.100, 0.015, 20.0e-3 * 12.0e-3);

    for v in [0.5, 2.0, 5.0, 12.0] {
        let r = estimate(&p, Drive::InletVelocity(v), &cfg);
        let re = v * dh_m / cfg.fluid.nu;
        let f = friction_factor(re, 0.0, Section::Rectangular { aspect: 20.0 / 12.0 }).f;
        let want = f * (l_m / dh_m) * 0.5 * cfg.fluid.rho * v * v;
        let got = r.total_pressure_drop_pa.mean;
        eprintln!(
            "V = {v:>5} m/s  Re = {re:>8.0}  {:<13} f = {f:.5}  dp = {got:.4} Pa (hand {want:.4})",
            r.regime.label()
        );
        assert!(
            (got / want - 1.0).abs() < 0.03,
            "V = {v}: dp = {got} Pa against the hand calculation {want} Pa"
        );
        // Q = V A, exactly.
        assert!((r.flow_m3s / (v * area_m2) - 1.0).abs() < 1e-9);
        // No bend, no transition: friction is the whole story.
        assert!((r.k_of(ElementKind::Friction).mean / r.loss_coefficient.k.mean - 1.0).abs() < 1e-9);
        assert!(r.total_pressure_drop_pa.contains(want), "the band must cover the truth");
    }
}

#[test]
fn the_laminar_branch_reproduces_the_analytic_solution_end_to_end() {
    // A 3 mm square duct at 1 m/s is Re = 194: fully laminar, where the
    // friction factor is not a correlation at all but an exact solution of the
    // Navier-Stokes equations. f*Re = 56.91 for a square section (Shah &
    // London), NOT the 64 of the circular formula -- an 11% difference that a
    // duct sizer using 64/Re would carry straight into the answer.
    let (grid, solid, mouths) = straight_duct(3.0, 3.0, 60.0, 0.125);
    let p = extract(grid, &solid, &mouths);
    let cfg = EstimateConfig::default();
    let r = estimate(&p, Drive::InletVelocity(1.0), &cfg);
    eprintln!("{}", r.summary());

    assert_eq!(r.regime, ad_estimate::Regime::Laminar);
    let re = 1.0 * 0.003 / cfg.fluid.nu;
    assert!((r.reynolds / re - 1.0).abs() < 0.02, "Re = {}", r.reynolds);
    assert!(
        (r.mean_friction_factor * r.reynolds - 56.91).abs() < 1.0,
        "f*Re = {}",
        r.mean_friction_factor * r.reynolds
    );
    let want = 56.91 / re * (0.060 / 0.003) * 0.5 * cfg.fluid.rho * 1.0;
    assert!(
        (r.total_pressure_drop_pa.mean / want - 1.0).abs() < 0.04,
        "dp = {} Pa against the analytic {want} Pa",
        r.total_pressure_drop_pa.mean
    );
    // ...and the circular formula would have been 12% out, which is the point.
    let circular = 64.0 / re * (0.060 / 0.003) * 0.5 * cfg.fluid.rho;
    assert!((circular / want - 1.0).abs() > 0.10);
}

#[test]
fn a_voxelised_bend_lands_on_the_handbook_coefficient() {
    // r/D_h = 2.25, H/W = 2: Idelchik gives K_loc = A1 B1 C1
    // = 1.0 * 0.21/sqrt(2.25) * 0.85 = 0.119. The extraction has to find that
    // geometry from voxels and the network has to look it up.
    let (grid, solid, mouths) = quarter_bend(24.0, 8.0, 16.0, 0.5);
    let p = extract(grid, &solid, &mouths);
    let r = estimate(&p, Drive::InletVelocity(6.0), &EstimateConfig::default());
    eprintln!("{}", r.summary());
    eprintln!("{}", r.breakdown());

    let k_bend = r.k_of(ElementKind::Bend).mean;
    let want = 0.21 / 2.25f64.sqrt() * 0.85;
    assert!(
        (k_bend / want - 1.0).abs() < 0.25,
        "bend K = {k_bend} against the hand-worked Idelchik value {want}"
    );
    // A well-radiused bend of this length: friction and bend are comparable,
    // and the total is nowhere near a mitre.
    assert!(r.loss_coefficient.k.mean < 0.5, "K = {}", r.loss_coefficient.k);
    assert!(r.k_of(ElementKind::Friction).mean > 0.0);
}

#[test]
fn extraction_is_fast_enough_to_sit_behind_a_drag() {
    // Not the microsecond path -- that is the network, timed in its own unit
    // test -- but the extraction still has to keep up with a geometry edit.
    let (grid, solid, mouths) = straight_duct(20.0, 12.0, 100.0, 0.5);
    eprintln!("grid {:?} = {} cells", grid.dims, grid.cell_count());
    let t0 = std::time::Instant::now();
    let p = extract(grid, &solid, &mouths);
    let extract_ms = t0.elapsed().as_secs_f64() * 1e3;

    let cfg = EstimateConfig::as_installed();
    let _ = estimate(&p, Drive::InletVelocity(3.0), &cfg);
    let n = 2000;
    let t1 = std::time::Instant::now();
    let mut sink = 0.0;
    for i in 0..n {
        sink += estimate(&p, Drive::InletVelocity(1.0 + (i % 8) as f64), &cfg)
            .total_pressure_drop_pa
            .mean;
    }
    let solve_us = t1.elapsed().as_secs_f64() * 1e6 / n as f64;
    assert!(sink > 0.0);
    eprintln!(
        "extraction {extract_ms:.1} ms over {} cells, then {solve_us:.1} us per solve \
         ({} stations)",
        grid.cell_count(),
        p.stations.len()
    );
    assert!(solve_us < 500.0, "{solve_us:.1} us per solve is not interactive");
}

//! Shader assembly for the metrics kernels.
//!
//! Two things live here: the WGSL sources, embedded with `include_str!` so the
//! crate works from any working directory and cannot pick up a stale file, and a
//! **generated prelude** that emits the flag constants and the accumulator
//! layout from the Rust definitions.
//!
//! The generated prelude is the important half. The plane accumulator is a flat
//! array of `u32` whose meaning is entirely positional — index 4 is the sum of
//! the through-plane velocity, index 20 is the first histogram bin — and if the
//! Rust mirror and the WGSL indices ever disagreed, every reported number would
//! be silently wrong in a way that still looked plausible. Emitting one from the
//! other makes that impossible, which is the same reason
//! [`ad_gpu::lattice::wgsl_prelude`] exists.

use ad_gpu::types::flags;
use ad_gpu::{ShaderDefines, ShaderLoader};

// ---------------------------------------------------------------------------
// Plane accumulator layout. One record per measurement plane.
//
//   [0]        sample count dispatched
//   [1]        samples whose containing cell is inside the grid
//   [2]        ...and in fluid: the covered-area count
//   [3]        ...and flowing backwards through the plane
//   [4  .. 20] the sixteen f32 scalars below, as bit patterns
//   [20 ..148] through-plane velocity histogram, exact counts
//   [148..180] momentum-angle histogram, fixed point in 1/2^24 of the total
//   [180..192] padding to a round 768 bytes
// ---------------------------------------------------------------------------

/// `u32` words per plane record.
pub const ACC_STRIDE: usize = 192;
pub const A_N_TOTAL: usize = 0;
pub const A_N_INSIDE: usize = 1;
pub const A_N_FLUID: usize = 2;
pub const A_N_BACK: usize = 3;
pub const A_SCALAR_BASE: usize = 4;
pub const N_SCALARS: usize = 16;
pub const A_HIST_BASE: usize = A_SCALAR_BASE + N_SCALARS;
pub const HIST_BINS: usize = 128;
pub const A_ANGLE_BASE: usize = A_HIST_BASE + HIST_BINS;
pub const ANGLE_BINS: usize = 32;

/// Sum of the through-plane velocity `w = u . n`.
pub const S_W: usize = 0;
/// Sum of `rho * w`: the mass flux, up to the area element.
pub const S_RHO_W: usize = 1;
/// Sum of static pressure `c_s^2 (rho - 1)`.
pub const S_PS: usize = 2;
/// Sum of total pressure, area-weighted. Reported for comparison only.
pub const S_PT: usize = 3;
/// Sum of `rho * w * p_t`: the numerator of the mass-flow-weighted total
/// pressure, which is the one that is physically meaningful.
pub const S_MDOT_PT: usize = 4;
/// Sum of `w^2`, for the coefficient of variation.
pub const S_W2: usize = 5;
/// Sum of `|u|`.
pub const S_SPEED: usize = 6;
/// Sum of `|w|`.
pub const S_ABS_W: usize = 7;
/// Momentum flux vector `J = sum rho * u * w`, x/y/z.
pub const S_JX: usize = 8;
pub const S_JY: usize = 9;
pub const S_JZ: usize = 10;
/// Sum of the forward momentum-flux magnitude `rho |u| max(w, 0)`.
pub const S_MOMFLUX: usize = 11;
/// Sum of `|w - w_bar|`, the Weltens numerator. Pass 1 only.
pub const S_ABS_DEV: usize = 12;
/// Peak `|u|` on the plane. Reduced by max, not by sum.
pub const S_MAX_SPEED: usize = 13;
pub const S_MAX_W: usize = 14;
pub const S_MIN_W: usize = 15;

// ---------------------------------------------------------------------------
// Volume accumulator layout. One record for the whole domain.
// ---------------------------------------------------------------------------

/// `u32` words in the volume record.
pub const VOL_STRIDE: usize = 16;
pub const V_N_VISITED: usize = 0;
pub const V_N_FLUID: usize = 1;
pub const V_N_REVERSE: usize = 2;
pub const V_N_STAGNANT: usize = 3;
pub const V_SCALAR_BASE: usize = 4;
pub const V_N_SCALARS: usize = 8;

// The summed scalars come first and the three extrema last, exactly as in the
// plane layout, because `volume.wgsl`'s halving tree folds slots
// `[0, 4 + VS_MAX_SPEED)` by addition and the remaining three by max/max/min.
// Putting an extremum in the middle would make the loop bound truncate the
// sums: with `VS_MAX_SPEED = 0` the tree summed *nothing* but the four counts,
// and every mean, RMS and residual came back as zero. The `const_assert` in
// `volume.wgsl` and `the_accumulator_layout_is_self_consistent` below both
// exist to catch that; the assert is what caught it.

pub const VS_SUM_SPEED: usize = 0;
pub const VS_SUM_SPEED2: usize = 1;
/// `sum |u^{n+1} - u^n|^2` over the probe points.
pub const VS_SUM_DU2: usize = 2;
/// `sum |u^n|^2` over the probe points.
pub const VS_SUM_U2: usize = 3;
pub const VS_SUM_AXIAL: usize = 4;
/// Peak `|u|` anywhere in the domain. Max-reduced; must stay third from last.
pub const VS_MAX_SPEED: usize = 5;
pub const VS_MAX_RHO: usize = 6;
pub const VS_MIN_RHO: usize = 7;

// ---------------------------------------------------------------------------
// Wall record layout. One per triangle.
// ---------------------------------------------------------------------------

/// `f32` slots per triangle in the wall buffer.
///
/// Every slot except [`W_AREA`] is *already multiplied by the triangle's area*,
/// so summing a column over triangles and dividing by the summed [`W_AREA`]
/// gives an area-weighted mean with no second pass and no per-triangle
/// bookkeeping on the CPU. `W_AREA` doubles as the coverage marker: it is zero
/// exactly when the probe walk found no fluid, which is how a triangle buried
/// inside another part is told apart from one sitting in dead-still air.
pub const WALL_STRIDE: usize = 12;
/// Pressure (normal) force on this triangle, lattice stress times mm^2, x/y/z.
pub const W_FP_X: usize = 0;
/// Viscous force, lattice stress times mm^2, x/y/z. The full traction, normal
/// component included, so summing it gives a real force rather than a
/// projection.
pub const W_FV_X: usize = 3;
/// Area-weighted sum of the wall shear magnitude.
pub const W_TAU_A: usize = 6;
/// Area-weighted sum of `y+`.
pub const W_YPLUS_A: usize = 7;
/// Wetted area, mm^2. Zero means no fluid was found off this triangle, and
/// every other slot in the record is then meaningless.
pub const W_AREA: usize = 8;
/// Area-weighted sum of the static wall pressure, lattice units.
pub const W_P_A: usize = 9;
/// Area-weighted sum of the probe distance, in cells. Reported so a wall map
/// can say how far from the surface it actually measured.
pub const W_Y_A: usize = 10;

/// Every constant above, plus the [`ad_gpu::flags`] bits, as WGSL.
pub fn generated_prelude() -> String {
    let mut s = String::with_capacity(2048);
    s.push_str("// GENERATED by ad_metrics::shaders::generated_prelude - do not edit\n");
    let mut c = |name: &str, v: usize| {
        s.push_str(&format!("const {name}: u32 = {v}u;\n"));
    };
    c("FLAG_SOLID", flags::SOLID as usize);
    c("FLAG_SOLID_BOUNDARY", flags::SOLID_BOUNDARY as usize);
    c("FLAG_INLET", flags::INLET as usize);
    c("FLAG_OUTLET", flags::OUTLET as usize);
    c("FLAG_SPONGE", flags::SPONGE as usize);
    c("FLAG_EQUILIBRIUM", flags::EQUILIBRIUM as usize);

    c("ACC_STRIDE", ACC_STRIDE);
    c("A_N_TOTAL", A_N_TOTAL);
    c("A_N_INSIDE", A_N_INSIDE);
    c("A_N_FLUID", A_N_FLUID);
    c("A_N_BACK", A_N_BACK);
    c("A_SCALAR_BASE", A_SCALAR_BASE);
    c("N_SCALARS", N_SCALARS);
    c("A_HIST_BASE", A_HIST_BASE);
    c("HIST_BINS", HIST_BINS);
    c("A_ANGLE_BASE", A_ANGLE_BASE);
    c("ANGLE_BINS", ANGLE_BINS);

    c("S_W", S_W);
    c("S_RHO_W", S_RHO_W);
    c("S_PS", S_PS);
    c("S_PT", S_PT);
    c("S_MDOT_PT", S_MDOT_PT);
    c("S_W2", S_W2);
    c("S_SPEED", S_SPEED);
    c("S_ABS_W", S_ABS_W);
    c("S_JX", S_JX);
    c("S_JY", S_JY);
    c("S_JZ", S_JZ);
    c("S_MOMFLUX", S_MOMFLUX);
    c("S_ABS_DEV", S_ABS_DEV);
    c("S_MAX_SPEED", S_MAX_SPEED);
    c("S_MAX_W", S_MAX_W);
    c("S_MIN_W", S_MIN_W);

    c("VOL_STRIDE", VOL_STRIDE);
    c("V_N_VISITED", V_N_VISITED);
    c("V_N_FLUID", V_N_FLUID);
    c("V_N_REVERSE", V_N_REVERSE);
    c("V_N_STAGNANT", V_N_STAGNANT);
    c("V_SCALAR_BASE", V_SCALAR_BASE);
    c("V_N_SCALARS", V_N_SCALARS);
    c("VS_MAX_SPEED", VS_MAX_SPEED);
    c("VS_SUM_SPEED", VS_SUM_SPEED);
    c("VS_SUM_SPEED2", VS_SUM_SPEED2);
    c("VS_SUM_DU2", VS_SUM_DU2);
    c("VS_SUM_U2", VS_SUM_U2);
    c("VS_SUM_AXIAL", VS_SUM_AXIAL);
    c("VS_MIN_RHO", VS_MIN_RHO);
    c("VS_MAX_RHO", VS_MAX_RHO);

    c("WALL_STRIDE", WALL_STRIDE);
    c("W_FP_X", W_FP_X);
    c("W_FV_X", W_FV_X);
    c("W_TAU_A", W_TAU_A);
    c("W_YPLUS_A", W_YPLUS_A);
    c("W_AREA", W_AREA);
    c("W_P_A", W_P_A);
    c("W_Y_A", W_Y_A);
    s
}

/// A loader with every metrics shader registered.
pub fn build_loader() -> ShaderLoader {
    let mut loader = ShaderLoader::new(".");
    loader.add_virtual("metrics/generated.wgsl", generated_prelude());
    loader.add_virtual(
        "metrics/common.wgsl",
        format!(
            "#include \"metrics/generated.wgsl\"\n{}",
            include_str!("../../../shaders/metrics/common.wgsl")
        ),
    );
    loader.add_virtual("metrics/plane.wgsl", include_str!("../../../shaders/metrics/plane.wgsl"));
    loader.add_virtual("metrics/volume.wgsl", include_str!("../../../shaders/metrics/volume.wgsl"));
    loader.add_virtual("metrics/wall.wgsl", include_str!("../../../shaders/metrics/wall.wgsl"));
    loader
}

pub fn defines() -> ShaderDefines {
    ShaderDefines::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_accumulator_layout_is_self_consistent() {
        assert_eq!(A_HIST_BASE, 20);
        assert_eq!(A_ANGLE_BASE, 148);
        assert!(A_ANGLE_BASE + ANGLE_BINS <= ACC_STRIDE);
        assert_eq!(ACC_STRIDE * 4, 768, "one plane record should be 768 bytes");
        // The three extrema must be the last scalars, because the workgroup
        // reduction sums every slot below S_MAX_SPEED and folds only those three
        // by max/min. Reordering them silently turns a maximum into a sum.
        assert_eq!(S_MAX_SPEED, N_SCALARS - 3);
        assert_eq!(S_MAX_W, N_SCALARS - 2);
        assert_eq!(S_MIN_W, N_SCALARS - 1);
        assert!(V_SCALAR_BASE + V_N_SCALARS <= VOL_STRIDE);
        // The volume record folds the same way, and gets the same rule: the
        // three extrema must be the last three scalars, in max/max/min order.
        // With VS_MAX_SPEED anywhere earlier, `volume.wgsl`'s
        // `for (s = 0; s < VS_MAX_SPEED; ...)` truncates the summed scalars and
        // every mean and residual silently reads back as zero.
        assert_eq!(VS_MAX_SPEED, V_N_SCALARS - 3);
        assert_eq!(VS_MAX_RHO, V_N_SCALARS - 2);
        assert_eq!(VS_MIN_RHO, V_N_SCALARS - 1);
        // The wall record's three-wide force slots must not overlap each other
        // or the scalars that follow them.
        assert!(W_FP_X + 3 <= W_FV_X);
        assert!(W_FV_X + 3 <= W_TAU_A);
        for s in [W_TAU_A, W_YPLUS_A, W_AREA, W_P_A, W_Y_A] {
            assert!(s < WALL_STRIDE, "wall slot {s} is outside the record");
        }
    }

    #[test]
    fn every_metrics_shader_preprocesses_and_gets_the_generated_constants() {
        // Catches an unbalanced #if, a missing include or a renamed constant
        // without needing a GPU.
        let loader = build_loader();
        let d = defines();
        for entry in ["metrics/plane.wgsl", "metrics/volume.wgsl", "metrics/wall.wgsl"] {
            let src = loader.load(entry, &d).unwrap_or_else(|e| panic!("{entry}: {e}"));
            assert!(src.contains("@compute"), "{entry}: no entry point emitted");
            assert!(
                src.contains(&format!("const ACC_STRIDE: u32 = {ACC_STRIDE}u;")),
                "{entry}: generated prelude missing"
            );
            assert!(
                src.contains(&format!("const FLAG_SOLID: u32 = {}u;", flags::SOLID)),
                "{entry}: flag constants missing"
            );
            assert!(src.contains("fn sample_field"), "{entry}: common.wgsl was not included");
        }
    }

    #[test]
    fn the_flag_constants_track_ad_gpu() {
        // If someone renumbers ad_gpu::flags, this is what notices.
        let p = generated_prelude();
        assert!(p.contains("const FLAG_SOLID: u32 = 1u;"));
        assert!(p.contains("const FLAG_INLET: u32 = 4u;"));
        assert!(p.contains("const FLAG_OUTLET: u32 = 8u;"));
    }
}

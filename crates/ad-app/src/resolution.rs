//! What a change of cell size would cost, worked out before anything is
//! allocated.
//!
//! # Two guards, not one
//!
//! The rebuild itself is transactional (`Running::replace_sim` in `main.rs`):
//! the new lattice is built beside the old one and swapped in only if every
//! step succeeded, so a failure costs the attempt and nothing else. That is the
//! guard of last resort. This module is the first guard, for the failures a
//! rebuild reports badly or not at all:
//!
//! * **A storage binding that is too big.** wgpu clamps one binding to
//!   2 GiB - 1 whatever the driver reports (see
//!   [`ad_gpu::GpuContext::max_binding_bytes`]), and every per-cell buffer is
//!   one binding. The build would fail and say so, but only after the stall.
//! * **A lattice that does not fit.** Past the end of VRAM a Vulkan driver on
//!   Windows need not fail the allocation at all: it can put it in system
//!   memory, where a bandwidth-bound solver runs at PCIe speed — about thirty
//!   times slower on this card, indistinguishable from a hang, with no error
//!   anywhere. And where it does fail, the rollback cannot help: wgpu 30 treats
//!   an out-of-memory in the staging uploads and submissions around a big
//!   allocation as losing the whole device, which frees the running solver
//!   along with the half-built one (measured: the rebuild rolled back, then the
//!   next touch of a live buffer found it destroyed). So for memory this is the
//!   only guard; the rollback covers everything else.
//!
//! A transactional rebuild holds both lattices while the new one is built, so
//! the memory check counts both.
//!
//! Everything except [`estimate`] is arithmetic on a [`Grid`], tested without a
//! GPU.

use ad_gpu::{DdfPrecision, GpuContext, Grid, VelocitySet};
use ad_ui::{ResolutionEstimate, ResolutionPanel, SimParams};
use glam::UVec3;

use crate::sim::Sim;

/// Device bytes per lattice cell outside the distribution functions, once
/// built.
///
/// Counted from the allocations: the solver's packed flags (1) and link mask
/// (4), its velocity (8, `Rgba16Float`) and density (4, `R32Float`) textures,
/// the flag texture the renderer masks walls with (1), the metrics passes'
/// packed interior mask (1), and the renderer's two derived-field textures at
/// half resolution (2 x 8 / 8 = 2): 21 bytes. Measured with nvidia-smi on this
/// part's room domain at 1.0 and 0.5 mm (9.1 M and 72.8 M cells), the slope
/// beyond the DDFs came out at 22.3 bytes, the difference most likely 3D
/// texture tiling. Rounded up, because this figure only ever refuses.
pub const AUX_BYTES_PER_CELL: u64 = 23;

/// The residence-time path's macroscopic buffer and its staging copy, when
/// `AERODUCT_RTD` turns it on.
pub const RTD_BYTES_PER_CELL: u64 = 32;

/// Voxeliser scratch per cell, alive until the new solver has been built: four
/// `u32` fields (distance, nearest triangle, signed distance, fill state) and
/// the packed flags. An upper bound: the transient measured at 0.5 mm (sampled
/// every 200 ms) was under half of this.
pub const BUILD_BYTES_PER_CELL: u64 = 17;

/// Device memory held back for everything outside the lattices, as measured on
/// this machine: the desktop and other applications (2.2-2.3 GiB), the app's
/// fixed cost of render targets, pipelines and UI (0.6 GiB), and what the
/// allocator keeps after a resolution change (0.2-0.8 GiB across four changes
/// in one run, not accumulating). That is about 3.6 GiB at most; the rest of
/// the margin is [`BUILD_BYTES_PER_CELL`], which measured at under half its
/// value.
pub const RESERVE_BYTES: u64 = 4 << 30;

/// Fraction of peak bandwidth the step kernel reaches on this GPU, for
/// predicting a rate when there is no measured one to scale. About 80% once the
/// profiler stopped billing the padding cells as useful work.
const ROOFLINE_EFFICIENCY: f64 = 0.8;

/// What one lattice costs the GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Footprint {
    pub grid: Grid,
    /// Cells including the solver's one-cell pad on every face, which is what
    /// the step kernel actually sweeps.
    pub padded_cells: u64,
    /// The largest single storage binding among the per-cell buffers.
    pub binding_bytes: u64,
    /// Device memory once built.
    pub resident_bytes: u64,
    /// Device memory at the peak of the build, voxeliser scratch included.
    pub build_bytes: u64,
}

impl Footprint {
    pub fn of(grid: Grid, precision: DdfPrecision, rtd: bool) -> Self {
        let set = VelocitySet::D3Q19;
        let cells = grid.cell_count();
        let p = grid.dims + UVec3::splat(2);
        let padded_cells = p.x as u64 * p.y as u64 * p.z as u64;
        // Asked of the solver rather than restated, so this is the allocation
        // and not a model of it. The app never runs periodic.
        let ddf = ad_solver::solver::ddf_bytes(grid, [false; 3], set, precision);
        // Every per-cell storage buffer is one binding, and whichever is largest
        // meets wgpu's clamp first. In FP32 that is a DDF direction; in FP16C
        // the u32-per-cell fields overtake it.
        let binding_bytes = [
            ddf / set.q() as u64,             // one DDF direction
            padded_cells * 4,                 // the solver's link mask
            cells * 4,                        // each voxeliser field
            if rtd { cells * 16 } else { 0 }, // the RTD macroscopic buffer
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let aux = AUX_BYTES_PER_CELL + if rtd { RTD_BYTES_PER_CELL } else { 0 };
        let resident_bytes = ddf + cells * aux;
        Self {
            grid,
            padded_cells,
            binding_bytes,
            resident_bytes,
            build_bytes: resident_bytes + cells * BUILD_BYTES_PER_CELL,
        }
    }
}

/// What a lattice has to fit inside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// wgpu's ceiling on one storage binding.
    pub max_binding_bytes: u64,
    /// Longest edge of a 3D texture. The solver's velocity and density textures
    /// are the size of the lattice.
    pub max_texture_3d: u32,
    /// Device memory the app may use, if the GPU's size is known.
    pub budget_bytes: Option<u64>,
}

impl Limits {
    pub fn of(gpu: &GpuContext) -> Self {
        Self {
            max_binding_bytes: gpu.max_binding_bytes(),
            max_texture_3d: gpu.limits.max_texture_dimension_3d,
            budget_bytes: gpu.caps.vram_bytes.map(|v| v.saturating_sub(RESERVE_BYTES)),
        }
    }
}

/// Why `target` cannot replace `current`, or `None` if it can.
///
/// Written for a toast, so each sentence says what to do about it.
pub fn blocker(current: &Footprint, target: &Footprint, limits: &Limits) -> Option<String> {
    let d = target.grid.dims;
    if d.max_element() > limits.max_texture_3d {
        return Some(format!(
            "the lattice would be {} x {} x {} cells, and the GPU's 3D textures stop at {} on a \
             side. Use a coarser cell size.",
            d.x, d.y, d.z, limits.max_texture_3d
        ));
    }
    if target.binding_bytes > limits.max_binding_bytes {
        return Some(format!(
            "one of its per-cell buffers would be {}, over the {} wgpu allows a single buffer \
             binding. Use a coarser cell size.",
            gib(target.binding_bytes),
            gib(limits.max_binding_bytes)
        ));
    }
    let budget = limits.budget_bytes?;
    if target.build_bytes > budget {
        return Some(format!(
            "it needs {} of GPU memory to build and {} is available. Use a coarser cell size.",
            gib(target.build_bytes),
            gib(budget)
        ));
    }
    let swap = current.resident_bytes + target.build_bytes;
    if swap > budget {
        return Some(format!(
            "it fits on its own ({}), but not beside the running lattice ({}), which stays \
             resident until the new one is built: {} of {}. Apply 1 mm first to release it, \
             then this.",
            gib(target.build_bytes),
            gib(current.resident_bytes),
            gib(swap),
            gib(budget)
        ));
    }
    None
}

/// What is known about solver speed, for scaling to another lattice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Throughput {
    /// Steps per second measured on the running lattice; zero or less when
    /// there is no measurement (paused, or the first frames).
    pub measured_steps_per_s: f64,
    /// Peak memory bandwidth, bytes per second, for the fallback prediction.
    pub peak_bandwidth: Option<f64>,
    pub precision: DdfPrecision,
}

/// Expected steps per second on `target`, from what is known on `current`.
///
/// The solver is bandwidth-bound, so a step costs in proportion to the padded
/// cells it sweeps and the measured rate scales by their ratio. Scaling a
/// measurement carries over everything a model would not know about this GPU
/// and driver; only with nothing measured does it fall back to the roofline.
pub fn steps_per_s(current: &Footprint, target: &Footprint, t: &Throughput) -> Option<f64> {
    if t.measured_steps_per_s > 0.0 && current.padded_cells > 0 && target.padded_cells > 0 {
        return Some(
            t.measured_steps_per_s * current.padded_cells as f64 / target.padded_cells as f64,
        );
    }
    let bandwidth = t.peak_bandwidth?;
    Some(ad_gpu::predicted_steps_per_second(
        target.padded_cells,
        VelocitySet::D3Q19,
        t.precision,
        bandwidth,
        ROOFLINE_EFFICIENCY,
    ))
}

/// Everything the resolution control shows about `dx_mm`, for the scene `sim`
/// is running with `params`.
pub fn estimate(
    gpu: &GpuContext,
    sim: &Sim,
    params: &SimParams,
    dx_mm: f32,
    domain_mm: Option<[f32; 6]>,
    measured_steps_per_s: f64,
) -> ResolutionEstimate {
    let in_range =
        dx_mm.is_finite() && (ResolutionPanel::MIN_MM..=ResolutionPanel::MAX_MM).contains(&dx_mm);
    if !in_range {
        return ResolutionEstimate {
            dx_mm,
            domain_mm,
            dims: [0; 3],
            cells: 0,
            vram_bytes: 0,
            budget_bytes: None,
            steps_per_s: None,
            flow_through_s: None,
            tau0: 0.0,
            warnings: Vec::new(),
            blocker: Some(format!(
                "the control takes {} to {} mm.",
                ResolutionPanel::MIN_MM,
                ResolutionPanel::MAX_MM
            )),
        };
    }

    let precision = crate::sim::precision_from_env();
    let rtd = crate::tracers::rtd_enabled();
    let target_params = SimParams {
        dx_mm,
        domain_mm,
        ..*params
    };
    // The planner the build itself uses, so this is the grid Apply would get.
    let (_, grid) = crate::sim::plan_lattice(
        sim.duct_bbox(),
        &sim.mouths,
        sim.inlets[0],
        sim.outlet,
        &target_params,
    );
    let current = Footprint::of(sim.grid, precision, rtd);
    let target = Footprint::of(grid, precision, rtd);
    let limits = Limits::of(gpu);

    let throughput = Throughput {
        measured_steps_per_s,
        peak_bandwidth: gpu.caps.peak_bandwidth,
        precision,
    };
    let rate = steps_per_s(&current, &target, &throughput);
    let units = target_params.lattice_units();
    // The length the metrics count a flow-through against, so the two agree.
    let length_mm = sim.duct_bbox().size().max_element() as f64;
    let flow_through_s = rate
        .filter(|r| *r > 0.0)
        .map(|r| units.steps_per_flow_through(length_mm) / r);

    let mut warnings = units.warnings();
    if limits.budget_bytes.is_none() {
        warnings.push(
            "GPU memory size unknown, so the memory check is off; set AERODUCT_VRAM_GB to turn \
             it on"
                .into(),
        );
    }

    ResolutionEstimate {
        dx_mm,
        domain_mm,
        dims: grid.dims.to_array(),
        cells: grid.cell_count(),
        vram_bytes: target.resident_bytes,
        budget_bytes: limits.budget_bytes,
        steps_per_s: rate,
        flow_through_s,
        tau0: units.tau0,
        warnings,
        blocker: blocker(&current, &target, &limits),
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::Bbox;
    use glam::Vec3;

    const GIB: u64 = 1 << 30;

    /// The room domain this part gets at the default margin: the 145 x 72 x 69
    /// mm part plus 58 mm on every side. About 21.6 M cells at 0.75 mm.
    fn room(dx_mm: f32) -> Grid {
        Grid::covering(
            Bbox {
                min: Vec3::ZERO,
                max: Vec3::new(261.0, 188.0, 185.0),
            },
            dx_mm,
        )
    }

    fn fp32(dx_mm: f32) -> Footprint {
        Footprint::of(room(dx_mm), DdfPrecision::Fp32, false)
    }

    /// This machine: an RTX 4090, and the binding clamp wgpu applies to it.
    fn rtx_4090() -> Limits {
        Limits {
            max_binding_bytes: (1 << 31) - 1,
            max_texture_3d: 16384,
            budget_bytes: Some(24 * GIB - RESERVE_BYTES),
        }
    }

    #[test]
    fn the_footprint_is_the_solvers_own_allocation_plus_the_counted_extras() {
        let g = room(0.75);
        let f = Footprint::of(g, DdfPrecision::Fp32, false);
        let ddf =
            ad_solver::solver::ddf_bytes(g, [false; 3], VelocitySet::D3Q19, DdfPrecision::Fp32);
        assert_eq!(f.resident_bytes, ddf + g.cell_count() * AUX_BYTES_PER_CELL);
        assert_eq!(
            f.build_bytes,
            f.resident_bytes + g.cell_count() * BUILD_BYTES_PER_CELL
        );
        // In FP32 the widest binding is one DDF direction: 4 bytes per padded
        // cell, the same as the link mask, and more than any interior field.
        assert_eq!(f.binding_bytes, ddf / 19);
        assert_eq!(f.binding_bytes, f.padded_cells * 4);

        // RTD adds its buffers to both the total and, at 16 bytes a cell, the
        // widest binding.
        let r = Footprint::of(g, DdfPrecision::Fp32, true);
        assert_eq!(
            r.resident_bytes,
            f.resident_bytes + g.cell_count() * RTD_BYTES_PER_CELL
        );
        assert_eq!(r.binding_bytes, g.cell_count() * 16);
    }

    #[test]
    fn in_fp16c_the_u32_fields_not_the_ddfs_set_the_binding_limit() {
        // Halving the DDFs does not halve the finest grid that binds: the link
        // mask and the voxeliser's fields are still four bytes a cell.
        let f = Footprint::of(room(0.5), DdfPrecision::Fp16c, false);
        let direction = f.padded_cells * 2;
        assert!(f.binding_bytes > direction);
        assert_eq!(f.binding_bytes, f.padded_cells * 4);
    }

    #[test]
    fn on_this_card_the_quality_tier_fits_and_the_finest_does_not() {
        let running = fp32(0.75);
        let limits = rtx_4090();
        // 0.5 and 0.4 mm are the quality tiers, and both must be reachable
        // straight from the default with the default still resident.
        assert_eq!(blocker(&running, &fp32(0.5), &limits), None);
        assert_eq!(blocker(&running, &fp32(0.4), &limits), None);
        // 0.3 mm on the room domain is ~337 M cells, ~26 GB of DDFs alone.
        let why = blocker(&running, &fp32(0.3), &limits).expect("0.3 mm cannot fit in 24 GB");
        assert!(why.contains("to build"), "{why}");
        // Coarsening always fits.
        assert_eq!(blocker(&fp32(0.4), &fp32(1.0), &limits), None);
    }

    #[test]
    fn each_limit_refuses_with_its_own_reason() {
        let current = fp32(1.0);
        let target = fp32(0.75);

        let textures = Limits {
            max_texture_3d: 256,
            ..rtx_4090()
        };
        let why = blocker(&current, &target, &textures).unwrap();
        assert!(why.contains("3D textures"), "{why}");

        let binding = Limits {
            max_binding_bytes: 1 << 20,
            ..rtx_4090()
        };
        let why = blocker(&current, &target, &binding).unwrap();
        assert!(why.contains("binding"), "{why}");

        let tiny = Limits {
            budget_bytes: Some(target.build_bytes - 1),
            ..rtx_4090()
        };
        let why = blocker(&current, &target, &tiny).unwrap();
        assert!(why.contains("to build"), "{why}");
    }

    #[test]
    fn a_target_that_fits_alone_but_not_beside_the_running_lattice_says_how_to_get_there() {
        let current = fp32(0.5);
        let target = fp32(0.4);
        let limits = Limits {
            budget_bytes: Some(current.resident_bytes + target.build_bytes - 1),
            ..rtx_4090()
        };
        assert!(
            target.build_bytes < limits.budget_bytes.unwrap(),
            "fits on its own"
        );
        let why = blocker(&current, &target, &limits).unwrap();
        assert!(why.contains("Apply 1 mm first"), "{why}");

        // One byte more and it goes.
        let roomier = Limits {
            budget_bytes: limits.budget_bytes.map(|b| b + 1),
            ..limits
        };
        assert_eq!(blocker(&current, &target, &roomier), None);
    }

    #[test]
    fn with_the_memory_size_unknown_only_the_hard_limits_apply() {
        let unknown = Limits {
            budget_bytes: None,
            ..rtx_4090()
        };
        assert_eq!(blocker(&fp32(0.75), &fp32(0.3), &unknown), None);
    }

    #[test]
    fn throughput_scales_the_measurement_and_falls_back_to_the_roofline() {
        let current = fp32(0.75);
        let target = fp32(0.5);
        let measured = Throughput {
            measured_steps_per_s: 1000.0,
            peak_bandwidth: Some(1008.0e9),
            precision: DdfPrecision::Fp32,
        };
        let scaled = steps_per_s(&current, &target, &measured).unwrap();
        let expected = 1000.0 * current.padded_cells as f64 / target.padded_cells as f64;
        assert!((scaled - expected).abs() < 1e-9);
        // (0.75 / 0.5)^3 = 3.4: a finer grid is slower by the cell ratio.
        assert!((1000.0 / scaled - 3.375).abs() < 0.1, "{scaled}");

        let paused = Throughput {
            measured_steps_per_s: 0.0,
            ..measured
        };
        let predicted = steps_per_s(&current, &target, &paused).unwrap();
        let roofline = ad_gpu::predicted_steps_per_second(
            target.padded_cells,
            VelocitySet::D3Q19,
            DdfPrecision::Fp32,
            1008.0e9,
            ROOFLINE_EFFICIENCY,
        );
        assert_eq!(predicted, roofline);

        let nothing = Throughput {
            peak_bandwidth: None,
            ..paused
        };
        assert_eq!(steps_per_s(&current, &target, &nothing), None);
    }
}

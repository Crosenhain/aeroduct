//! Whole-domain reductions: the driver for `shaders/metrics/volume.wgsl`.
//!
//! Two entry points, two very different costs and two very different meanings:
//!
//! * `volume_stats` visits **every** cell and reports the peak speed, the mean
//!   and RMS speed, the reverse-flow and stagnation volume fractions and the
//!   density extremes.
//! * `volume_residual` visits a strided probe set and reports
//!   `R = ||u^{n+1} - u^n|| / ||u^n||`, differencing against a stored copy of
//!   the previous probe field.
//!
//! # Why they cannot share an accumulator
//!
//! Both entry points call the same `reduce_and_flush`, which writes to absolute
//! slots — `acc[V_N_VISITED]`, not `acc[base + V_N_VISITED]`. Pointing both at
//! one record would make the cell count the sum of the whole grid *and* the
//! probe set, and every fraction derived from it would be wrong by whatever
//! ratio the stride happened to give. So they get one storage buffer each, and
//! a small staging buffer collects the pair for a single readback.
//!
//! # Reading the residual
//!
//! It will plateau. See the long note in [`crate::stats`]: at Re = 1,600-15,500
//! this duct is genuinely unsteady, so `R` falls for a few flow-through times
//! and then flattens at the amplitude of the real fluctuation. That plateau is
//! the signal to **start time-averaging**, not evidence of a failed run.
//! [`crate::stats::Monitor`] is built around that distinction.
//!
//! # Why the peak speed gets its own headline
//!
//! Dipole sound power scales as `U^6` (Curle 1955), so a 20% cut in peak
//! velocity is 5.8 dB — audible, and often achievable by rounding one corner.
//! That is why `volume_stats` reduces over every cell rather than a sample: a
//! peak that exists in one cell is still the cell that makes the noise.

use ad_gpu::types::{Grid, LatticeUnits};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec3};

use crate::plane::GpuGrid;
use crate::readback::{Frame, ReadbackRing};
use crate::shaders;
use crate::shaders::{
    VOL_STRIDE, VS_MAX_RHO, VS_MAX_SPEED, VS_MIN_RHO, VS_SUM_AXIAL, VS_SUM_DU2, VS_SUM_SPEED,
    VS_SUM_SPEED2, VS_SUM_U2, V_N_FLUID, V_N_REVERSE, V_N_STAGNANT, V_N_VISITED, V_SCALAR_BASE,
};

/// Workgroups each reduction is dispatched as, whatever the grid size.
///
/// The shaders stride across the grid by the whole launch, so this is also
/// how many float atomics each accumulator word takes per pass. It used to be
/// one workgroup per 64 cells: at dx = 0.75 mm, 337,000 compare-exchange loops
/// on the same words, which serialise in L2 and took the frame from 40 ms to
/// 190 ms as the flow filled the domain. 1024 x 64 threads still keeps enough
/// loads in flight to stream the field at close to full bandwidth.
const REDUCE_WORKGROUPS: u32 = 1024;

/// Mirror of `VolumeUniforms` in `volume.wgsl`.
///
/// The trailing padding is not decorative. WGSL aligns `vec3<u32>` to 16 bytes,
/// so the shader's `pad: vec3<u32>` lands at offset 80 and the struct rounds up
/// to 96 — even though only 68 bytes carry data. A Rust mirror that stopped at
/// 68 would be rejected as too small for the binding.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VolumeUniforms {
    grid: GpuGrid,
    axis: [f32; 3],
    stagnation_threshold: f32,
    probe_dims: [u32; 3],
    stride: u32,
    has_prev: u32,
    _pad: [u32; 7],
}

const _: () = assert!(std::mem::size_of::<VolumeUniforms>() == 96);

/// One volume record, exactly as the GPU wrote it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VolumeAccum {
    pub words: [u32; VOL_STRIDE],
}

impl Default for VolumeAccum {
    fn default() -> Self {
        Self::initial()
    }
}

impl VolumeAccum {
    /// Zeroed except the three extrema, which fold by `max`/`max`/`min` and
    /// would otherwise be clamped at zero. See [`crate::plane::PlaneAccum`].
    pub fn initial() -> Self {
        let mut words = [0u32; VOL_STRIDE];
        words[V_SCALAR_BASE + VS_MAX_SPEED] = (-1.0e30f32).to_bits();
        words[V_SCALAR_BASE + VS_MAX_RHO] = (-1.0e30f32).to_bits();
        words[V_SCALAR_BASE + VS_MIN_RHO] = (1.0e30f32).to_bits();
        Self { words }
    }

    pub fn visited(&self) -> u32 {
        self.words[V_N_VISITED]
    }
    pub fn fluid(&self) -> u32 {
        self.words[V_N_FLUID]
    }
    pub fn reversed(&self) -> u32 {
        self.words[V_N_REVERSE]
    }
    pub fn stagnant(&self) -> u32 {
        self.words[V_N_STAGNANT]
    }
    pub fn scalar(&self, s: usize) -> f32 {
        f32::from_bits(self.words[V_SCALAR_BASE + s])
    }
}

/// How the volume pass classifies a cell.
#[derive(Debug, Clone, Copy)]
pub struct VolumeConfig {
    /// Duct axis, unit. "Reverse flow" is `u . axis < 0`.
    ///
    /// Measured against a *fixed* axis rather than against the local streamline
    /// direction, because "the flow here is going backwards" only means anything
    /// relative to where the duct is trying to send the air. Against the local
    /// direction nothing is ever reversed, and a 90 degree bend would report a
    /// clean run no matter how badly it separated.
    pub axis: Vec3,
    /// Speed below which a cell counts as stagnant, **lattice units**. A useful
    /// default is 5% of the inlet bulk lattice velocity.
    pub stagnation_threshold: f32,
    /// Probe stride for the residual pass. 4 turns 20 M cells into 312 k probes
    /// and 5 MB of stored previous field instead of 320 MB, which estimates the
    /// field-wide L2 norm far more precisely than anyone reads a residual.
    pub residual_stride: u32,
}

impl Default for VolumeConfig {
    fn default() -> Self {
        Self { axis: Vec3::X, stagnation_threshold: 0.0025, residual_stride: 4 }
    }
}

/// Pipelines, buffers and readback for the two volume passes.
pub struct VolumeMetrics {
    stats_pipeline: wgpu::ComputePipeline,
    residual_pipeline: wgpu::ComputePipeline,
    uniform: wgpu::Buffer,
    stats_group: wgpu::BindGroup,
    residual_group: wgpu::BindGroup,
    acc_stats: wgpu::Buffer,
    acc_residual: wgpu::Buffer,
    staging: wgpu::Buffer,
    ring: ReadbackRing<VolumeAccum>,
    grid: Grid,
    probe_dims: UVec3,
    cfg: VolumeConfig,
    has_prev: bool,
}

impl VolumeMetrics {
    pub fn new(
        device: &wgpu::Device,
        field_layout: &wgpu::BindGroupLayout,
        grid: Grid,
        cfg: VolumeConfig,
    ) -> Result<Self> {
        let stride = cfg.residual_stride.max(1);
        let probe_dims = UVec3::new(
            grid.dims.x.div_ceil(stride),
            grid.dims.y.div_ceil(stride),
            grid.dims.z.div_ceil(stride),
        );
        let probes = (probe_dims.x as u64) * (probe_dims.y as u64) * (probe_dims.z as u64);

        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("metrics volume group1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(2, false),
            ],
        });

        let loader = shaders::build_loader();
        let module = loader.create_module(device, "metrics/volume.wgsl", &shaders::defines())?;
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("metrics volume layout"),
            bind_group_layouts: &[Some(field_layout), Some(&group1_layout)],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let acc = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (VOL_STRIDE * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let acc_stats = acc("metrics volume stats");
        let acc_residual = acc("metrics volume residual");
        let prev = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metrics volume previous field"),
            size: probes.max(1) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metrics volume staging"),
            size: (VOL_STRIDE * 4 * 2) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metrics volume uniforms"),
            size: std::mem::size_of::<VolumeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let group = |label, target: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &group1_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: target.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: prev.as_entire_binding() },
                ],
            })
        };
        let stats_group = group("metrics volume stats group", &acc_stats);
        let residual_group = group("metrics volume residual group", &acc_residual);

        Ok(Self {
            stats_pipeline: make("volume_stats"),
            residual_pipeline: make("volume_residual"),
            uniform,
            stats_group,
            residual_group,
            acc_stats,
            acc_residual,
            staging,
            ring: ReadbackRing::new(device, "metrics volume", 2, 3),
            grid,
            probe_dims,
            cfg,
            has_prev: false,
        })
    }

    pub fn config(&self) -> &VolumeConfig {
        &self.cfg
    }

    /// Update the axis or thresholds without rebuilding. Changing the residual
    /// stride needs a rebuild, because the stored previous field is sized to it,
    /// so that field is ignored here.
    pub fn set_config(&mut self, cfg: VolumeConfig) {
        self.cfg = VolumeConfig { residual_stride: self.cfg.residual_stride, ..cfg };
    }

    /// Forget the stored previous field, so the next residual is skipped rather
    /// than differenced against a field from before a parameter change.
    pub fn invalidate_history(&mut self) {
        self.has_prev = false;
    }

    pub fn has_capacity(&self) -> bool {
        self.ring.has_capacity()
    }

    /// Record both passes and the readback copy.
    ///
    /// As with [`crate::plane::PlaneMetrics::record`], the accumulators are
    /// reset through `queue.write_buffer` because three slots need `+/-1e30`
    /// sentinels, so `encoder` must be submitted before anything else touches
    /// them.
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        field_group: &wgpu::BindGroup,
        step: u64,
        profiler: Option<&mut ad_gpu::Profiler>,
    ) -> bool {
        if !self.ring.has_capacity() {
            return false;
        }
        let u = VolumeUniforms {
            grid: GpuGrid::new(self.grid),
            axis: self.cfg.axis.normalize_or_zero().to_array(),
            stagnation_threshold: self.cfg.stagnation_threshold,
            probe_dims: self.probe_dims.to_array(),
            stride: self.cfg.residual_stride.max(1),
            has_prev: self.has_prev as u32,
            _pad: [0; 7],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        let init = VolumeAccum::initial();
        queue.write_buffer(&self.acc_stats, 0, bytemuck::bytes_of(&init));
        queue.write_buffer(&self.acc_residual, 0, bytemuck::bytes_of(&init));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("metrics volume"),
                timestamp_writes: profiler.and_then(|p| p.scope("metrics volume")),
            });
            pass.set_bind_group(0, field_group, &[]);

            pass.set_pipeline(&self.stats_pipeline);
            pass.set_bind_group(1, &self.stats_group, &[]);
            pass.dispatch_workgroups(REDUCE_WORKGROUPS, 1, 1);

            pass.set_pipeline(&self.residual_pipeline);
            pass.set_bind_group(1, &self.residual_group, &[]);
            pass.dispatch_workgroups(REDUCE_WORKGROUPS, 1, 1);
        }
        let bytes = (VOL_STRIDE * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.acc_stats, 0, &self.staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.acc_residual, 0, &self.staging, bytes, bytes);
        self.has_prev = true;

        self.ring.record(encoder, &self.staging, 0, step)
    }

    pub fn poll(&mut self, device: &wgpu::Device) -> Vec<Frame<VolumeAccum>> {
        self.ring.poll(device)
    }

    pub fn drain_blocking(&mut self, device: &wgpu::Device) -> Vec<Frame<VolumeAccum>> {
        self.ring.drain_blocking(device)
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The volume pass in SI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeReading {
    pub cells_visited: u64,
    pub fluid_cells: u64,
    /// Fluid volume, mm^3. `V/Q` from this is what [`crate::rtd`] compares the
    /// measured mean residence time against.
    pub fluid_volume_mm3: f64,
    /// Fraction of the fluid volume moving against the duct axis.
    pub reverse_fraction: f64,
    /// Fraction of the fluid volume below the stagnation threshold.
    pub stagnant_fraction: f64,
    /// Peak speed anywhere in the domain, m/s.
    pub peak_speed_ms: f64,
    pub mean_speed_ms: f64,
    pub rms_speed_ms: f64,
    /// Mean velocity component along the duct axis, m/s.
    pub mean_axial_ms: f64,
    /// Static pressure range across the domain, Pa. A run whose minimum is
    /// diving is about to cavitate numerically.
    pub min_pressure_pa: f64,
    pub max_pressure_pa: f64,
    /// `||u^{n+1} - u^n|| / ||u^n||` over the probe set, or `None` on the first
    /// pass after a reset, when there is nothing to difference against.
    pub residual: Option<f64>,
    pub probe_cells: u64,
}

impl VolumeReading {
    /// `frame` is one readback: `[stats, residual]`.
    pub fn from_accums(stats: &VolumeAccum, residual: &VolumeAccum, grid: Grid, lu: &LatticeUnits) -> Self {
        let fluid = stats.fluid() as f64;
        let c_u = lu.c_u();
        let cell_mm3 = (grid.dx_mm as f64).powi(3);
        let p_factor = lu.rho_phys * c_u * c_u / 3.0; // c_s^2 (rho - 1) -> Pa

        let (mean_speed, rms_speed, mean_axial) = if fluid > 0.0 {
            (
                stats.scalar(VS_SUM_SPEED) as f64 / fluid * c_u,
                (stats.scalar(VS_SUM_SPEED2) as f64 / fluid).max(0.0).sqrt() * c_u,
                stats.scalar(VS_SUM_AXIAL) as f64 / fluid * c_u,
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        let peak = if stats.fluid() > 0 { stats.scalar(VS_MAX_SPEED) as f64 * c_u } else { 0.0 };
        let (min_rho, max_rho) = if stats.fluid() > 0 {
            (stats.scalar(VS_MIN_RHO) as f64, stats.scalar(VS_MAX_RHO) as f64)
        } else {
            (1.0, 1.0)
        };

        let sum_u2 = residual.scalar(VS_SUM_U2) as f64;
        let sum_du2 = residual.scalar(VS_SUM_DU2) as f64;
        let res = if sum_u2 > 0.0 { Some((sum_du2 / sum_u2).sqrt()) } else { None };

        Self {
            cells_visited: stats.visited() as u64,
            fluid_cells: stats.fluid() as u64,
            fluid_volume_mm3: fluid * cell_mm3,
            reverse_fraction: if fluid > 0.0 { stats.reversed() as f64 / fluid } else { 0.0 },
            stagnant_fraction: if fluid > 0.0 { stats.stagnant() as f64 / fluid } else { 0.0 },
            peak_speed_ms: peak,
            mean_speed_ms: mean_speed,
            rms_speed_ms: rms_speed,
            mean_axial_ms: mean_axial,
            min_pressure_pa: (min_rho - 1.0) * p_factor,
            max_pressure_pa: (max_rho - 1.0) * p_factor,
            residual: res,
            probe_cells: residual.fluid() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{field_layout, FieldTextures};
    use crate::field::FieldRefs;
    use ad_gpu::types::{flags, Bbox};

    fn units() -> LatticeUnits {
        LatticeUnits::for_air(1.0, 2.0, 0.05)
    }

    /// Run the volume pass twice over a field the closure defines, so the second
    /// frame has a previous field to difference against.
    fn measure(
        gpu: &ad_gpu::GpuContext,
        grid: Grid,
        cfg: VolumeConfig,
        frames: usize,
        fill: impl Fn(usize, glam::UVec3, Vec3) -> (Vec3, f32, u8),
    ) -> VolumeReading {
        let tex = FieldTextures::new_exact(&gpu.device, grid);
        let layout = field_layout(&gpu.device);
        let (vv, dv) = (tex.velocity_view(), tex.density_view());
        let group = crate::field::field_bind_group(
            &gpu.device,
            &layout,
            &FieldRefs { grid, velocity: &vv, density: &dv, flags: tex.flags_buffer() },
        );
        let mut vm = VolumeMetrics::new(&gpu.device, &layout, grid, cfg).unwrap();
        let mut last = None;
        for f in 0..frames {
            tex.fill(&gpu.queue, |c, p| fill(f, c, p)).unwrap();
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            assert!(vm.record(&gpu.queue, &mut enc, &group, f as u64, None));
            gpu.queue.submit([enc.finish()]);
            for fr in vm.drain_blocking(&gpu.device) {
                last = Some(VolumeReading::from_accums(&fr.data[0], &fr.data[1], grid, &units()));
            }
        }
        last.expect("no volume readback arrived")
    }

    #[test]
    fn the_accumulator_starts_from_sentinels_not_zero() {
        let a = VolumeAccum::initial();
        assert!(a.scalar(VS_MAX_SPEED) < -1.0e29);
        assert!(a.scalar(VS_MIN_RHO) > 1.0e29);
        assert_eq!(a.fluid(), 0);
        assert_eq!(std::mem::size_of::<VolumeAccum>(), VOL_STRIDE * 4);
    }

    /// Every fraction this pass reports is a count over `n_fluid`, so the first
    /// thing to check is that the counts are the counts: a known number of solid
    /// cells, a known number reversed, a known number stagnant.
    #[test]
    fn the_volume_fractions_are_exact_cell_counts() {
        let Some(gpu) = crate::test_gpu() else { return };
        let grid = Grid::covering(Bbox { min: Vec3::ZERO, max: Vec3::splat(20.0) }, 1.0);
        let n = grid.cell_count();
        let cfg = VolumeConfig {
            axis: Vec3::X,
            // Half the inlet speed, so the "slow" third counts as stagnant.
            stagnation_threshold: 0.01,
            residual_stride: 2,
        };
        // x < 5: solid. 5 <= x < 10: reversed at full speed. 10 <= x < 15:
        // stagnant. x >= 15: forward at full speed.
        let r = measure(&gpu, grid, cfg, 1, |_, c, _| match c.x {
            0..=4 => (Vec3::ZERO, 1.0, flags::SOLID),
            5..=9 => (Vec3::new(-0.05, 0.0, 0.0), 1.0, flags::FLUID),
            10..=14 => (Vec3::new(0.001, 0.0, 0.0), 1.0, flags::FLUID),
            _ => (Vec3::new(0.05, 0.0, 0.0), 1.0, flags::FLUID),
        });
        assert_eq!(r.cells_visited, n);
        assert_eq!(r.fluid_cells, n * 3 / 4, "a quarter of the grid is solid");
        assert!((r.reverse_fraction - 1.0 / 3.0).abs() < 1e-9, "{}", r.reverse_fraction);
        assert!((r.stagnant_fraction - 1.0 / 3.0).abs() < 1e-9, "{}", r.stagnant_fraction);
        // Fluid volume: 3/4 of a 20 mm cube.
        assert!((r.fluid_volume_mm3 - 6000.0).abs() < 1.0, "{}", r.fluid_volume_mm3);
        // Peak speed comes from the fastest cell, not an average.
        let want = 0.05 * units().c_u();
        assert!((r.peak_speed_ms / want - 1.0).abs() < 1e-3, "peak {}", r.peak_speed_ms);
        // Reversed and forward halves cancel; the stagnant third is what is left.
        let want_axial = (-0.05 + 0.001 + 0.05) / 3.0 * units().c_u();
        assert!(
            (r.mean_axial_ms - want_axial).abs() < 1e-6 * units().c_u(),
            "axial {} vs {want_axial}",
            r.mean_axial_ms
        );
    }

    /// A field that does not move has a residual of exactly zero; one that
    /// changes by a known relative amount reports exactly that amount. Anything
    /// else means the previous field is being stored or read wrongly, which is
    /// the only way this pass can fail silently.
    #[test]
    fn the_residual_measures_the_relative_change_it_is_given() {
        let Some(gpu) = crate::test_gpu() else { return };
        let grid = Grid::covering(Bbox { min: Vec3::ZERO, max: Vec3::splat(16.0) }, 1.0);
        let cfg = VolumeConfig { residual_stride: 2, ..Default::default() };

        let steady = measure(&gpu, grid, cfg, 2, |_, _, _| {
            (Vec3::new(0.05, 0.0, 0.0), 1.0, flags::FLUID)
        });
        assert_eq!(steady.residual, Some(0.0), "a frozen field must have no residual");

        // Frame 1 is 1% faster than frame 0, uniformly, so R = 0.01 exactly.
        let moving = measure(&gpu, grid, cfg, 2, |f, _, _| {
            let s = if f == 0 { 0.05 } else { 0.05 * 1.01 };
            (Vec3::new(s, 0.0, 0.0), 1.0, flags::FLUID)
        });
        let r = moving.residual.expect("second frame should produce a residual");
        assert!((r - 0.01).abs() < 1e-4, "residual {r} should be 1%");
    }

    /// Reverse flow is judged against the *duct* axis, not the local streamline
    /// direction. Rotating the axis by 180 degrees must flip the answer; without
    /// that convention a bend would never report separation at all.
    #[test]
    fn reverse_flow_is_measured_against_the_duct_axis() {
        let Some(gpu) = crate::test_gpu() else { return };
        let grid = Grid::covering(Bbox { min: Vec3::ZERO, max: Vec3::splat(12.0) }, 1.0);
        let fwd = measure(
            &gpu,
            grid,
            VolumeConfig { axis: Vec3::X, ..Default::default() },
            1,
            |_, _, _| (Vec3::new(0.05, 0.0, 0.0), 1.0, flags::FLUID),
        );
        assert_eq!(fwd.reverse_fraction, 0.0);
        let back = measure(
            &gpu,
            grid,
            VolumeConfig { axis: -Vec3::X, ..Default::default() },
            1,
            |_, _, _| (Vec3::new(0.05, 0.0, 0.0), 1.0, flags::FLUID),
        );
        assert_eq!(back.reverse_fraction, 1.0);
    }

    /// The density extremes must survive the `min`/`max` fold and convert to a
    /// symmetric pressure range about the reference.
    #[test]
    fn the_pressure_range_comes_from_the_density_extremes() {
        let Some(gpu) = crate::test_gpu() else { return };
        let grid = Grid::covering(Bbox { min: Vec3::ZERO, max: Vec3::splat(16.0) }, 1.0);
        let lu = units();
        let r = measure(&gpu, grid, VolumeConfig::default(), 1, |_, c, _| {
            let rho = if c.x == 0 { 0.999 } else if c.x == 1 { 1.001 } else { 1.0 };
            (Vec3::new(0.05, 0.0, 0.0), rho, flags::FLUID)
        });
        let want_hi = lu.pressure_pa(0.001);
        assert!(
            (r.max_pressure_pa / want_hi - 1.0).abs() < 1e-3,
            "p_max {} vs {want_hi}",
            r.max_pressure_pa
        );
        assert!(
            (r.min_pressure_pa / -want_hi - 1.0).abs() < 1e-3,
            "p_min {} vs {}",
            r.min_pressure_pa,
            -want_hi
        );
    }
}

//! Plane integrals: the driver for `shaders/metrics/plane.wgsl`.
//!
//! Given an [`ad_gpu::FlowPatch`] and a snapshot of the solver fields, this
//! dispatches a 512 x 512 midpoint quadrature over the patch and turns the
//! reduced record into engineering numbers — flow rate, total and static
//! pressure, uniformity, backflow, the velocity histogram and the momentum
//! direction — in SI.
//!
//! # The quadrature
//!
//! Sample `(s_i, t_j)` with `s_i = -1 + 2(i + 1/2)/N`, so the samples are cell
//! centres of an `N x N` subdivision of the patch and each one owns an equal
//! area `dA = A / N^2`. Every integral is then a plain sum times `dA`:
//!
//! ```text
//! Q      = sum_fluid (u . n) dA
//! p_t    = sum_fluid (rho w p_t) / sum_fluid (rho w)      mass-flow weighted
//! p_s    = sum_fluid p_s / n_fluid                        area weighted
//! ```
//!
//! `N = 512` is 262,144 samples per plane per frame. That is deliberately far
//! finer than the lattice — at `dx = 0.4 mm` a 2116 mm^2 mouth is about 13,000
//! cells — because the quadrature error and the discretisation error are then
//! not the same size, and the flow rate stops jittering as the patch is nudged
//! by a fraction of a cell.
//!
//! # Skipped samples are reported, not hidden
//!
//! A patch is a rectangle; a duct mouth is not. Samples whose containing cell is
//! solid, or which fall outside the grid entirely, contribute nothing to any
//! integral — which is right, since no air flows through the wall — but they are
//! counted. [`PlaneReading::covered_fraction`] is that count, and it is the
//! difference between "this duct passes 4.2 L/s" and "this duct passes 4.2 L/s
//! through the 61% of the patch rectangle that is actually a hole". A patch
//! placed carelessly across a wall would otherwise under-read in silence, which
//! is the failure mode that makes a measurement tool worthless.
//!
//! # fp32, everywhere
//!
//! See the header of `plane.wgsl`. Summing 262,144 lattice velocities of ~0.05
//! in fp16 stalls the accumulator once the partial sum passes a couple of
//! thousand, because the representable spacing there is 8. That is a systematic
//! bias in an integral, not noise that averages away.

use ad_gpu::types::{FlowPatch, Grid, LatticeUnits, MetricSample};
use anyhow::{ensure, Result};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::readback::{Frame, ReadbackRing};
use crate::shaders;
use crate::shaders::{
    ACC_STRIDE, ANGLE_BINS, A_ANGLE_BASE, A_HIST_BASE, A_N_BACK, A_N_FLUID, A_N_INSIDE,
    A_N_TOTAL, A_SCALAR_BASE, HIST_BINS, N_SCALARS, S_ABS_DEV, S_JX, S_JY, S_JZ, S_MAX_SPEED,
    S_MAX_W, S_MDOT_PT, S_MIN_W, S_PS, S_PT, S_RHO_W, S_SPEED, S_W, S_W2,
};

/// Workgroups along each side of the plane launch, whatever the sample count.
///
/// The shader strides over the sample grid, so this squared is also how many
/// float atomics each accumulator word takes per dispatch. See the reduction
/// note in `plane.wgsl` for why that count, not the sample count, sets the
/// cost.
const REDUCE_GROUPS_PER_SIDE: u32 = 16;

/// Default samples per side. 512 x 512 = 262,144 samples per plane.
pub const DEFAULT_SAMPLES: u32 = 512;

/// The `GridInfo` struct every metrics shader's uniform starts with.
///
/// Lives here rather than in [`crate::field`] because it is a *uniform* layout
/// concern, not a binding concern: 32 bytes, `vec3<u32>` then `f32` then
/// `vec3<f32>` then padding, which is what WGSL's 16-byte uniform alignment
/// forces. [`crate::volume`] and [`crate::wall`] embed the same struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuGrid {
    pub dims: [u32; 3],
    pub dx_mm: f32,
    pub origin_mm: [f32; 3],
    pub pad: f32,
}

impl GpuGrid {
    pub fn new(grid: Grid) -> Self {
        Self {
            dims: grid.dims.to_array(),
            dx_mm: grid.dx_mm,
            origin_mm: grid.origin_mm.to_array(),
            pad: 0.0,
        }
    }
}

const _: () = assert!(std::mem::size_of::<GpuGrid>() == 32);

/// Mirror of `PlaneUniforms` in `plane.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PlaneUniforms {
    grid: GpuGrid,
    center_mm: [f32; 3],
    pass_index: u32,
    half_u: [f32; 3],
    slot: u32,
    half_v: [f32; 3],
    samples: u32,
    normal: [f32; 3],
    pad: f32,
}

const _: () = assert!(std::mem::size_of::<PlaneUniforms>() == 96);

/// One plane's raw accumulator record, exactly as the GPU wrote it.
///
/// Kept as opaque words rather than a struct of named fields because the layout
/// is generated: [`crate::shaders::generated_prelude`] emits the same indices
/// into WGSL, so there is precisely one definition of "slot 4 is the sum of the
/// through-plane velocity" and it cannot drift.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PlaneAccum {
    pub words: [u32; ACC_STRIDE],
}

impl Default for PlaneAccum {
    fn default() -> Self {
        Self::initial()
    }
}

impl std::fmt::Debug for PlaneAccum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaneAccum")
            .field("n_total", &self.n_total())
            .field("n_fluid", &self.n_fluid())
            .field("sum_w", &self.scalar(S_W))
            .finish_non_exhaustive()
    }
}

impl PlaneAccum {
    /// The state the buffer must be in *before* a dispatch.
    ///
    /// Not all zeros. Three slots are folded by `max`/`max`/`min` rather than by
    /// addition, and a zeroed maximum would clamp a wholly reversed plane to
    /// zero — reporting no backflow on a plane that is entirely backflow. The
    /// sentinels are `+/-1e30` rather than infinities so that a plane which sees
    /// no fluid at all yields a finite, obviously-wrong number instead of a NaN
    /// that propagates.
    pub fn initial() -> Self {
        let mut words = [0u32; ACC_STRIDE];
        words[A_SCALAR_BASE + S_MAX_SPEED] = (-1.0e30f32).to_bits();
        words[A_SCALAR_BASE + S_MAX_W] = (-1.0e30f32).to_bits();
        words[A_SCALAR_BASE + S_MIN_W] = (1.0e30f32).to_bits();
        Self { words }
    }

    /// Samples dispatched over the patch, `N^2`.
    pub fn n_total(&self) -> u32 {
        self.words[A_N_TOTAL]
    }
    /// ...of which the containing cell lies inside the grid.
    pub fn n_inside(&self) -> u32 {
        self.words[A_N_INSIDE]
    }
    /// ...of which the containing cell is fluid. The covered-area count.
    pub fn n_fluid(&self) -> u32 {
        self.words[A_N_FLUID]
    }
    /// ...of which the through-plane velocity is negative.
    pub fn n_backflow(&self) -> u32 {
        self.words[A_N_BACK]
    }

    /// One of the sixteen f32 scalars, by its `S_*` index.
    pub fn scalar(&self, s: usize) -> f32 {
        debug_assert!(s < N_SCALARS);
        f32::from_bits(self.words[A_SCALAR_BASE + s])
    }

    /// Through-plane velocity histogram, exact counts over
    /// `[scalar(S_MIN_W), scalar(S_MAX_W)]`.
    pub fn histogram(&self) -> &[u32] {
        &self.words[A_HIST_BASE..A_HIST_BASE + HIST_BINS]
    }

    /// Momentum-angle histogram about the mean momentum direction, in fixed
    /// point (1/2^24 of the total forward momentum flux per count).
    pub fn angle_histogram(&self) -> &[u32] {
        &self.words[A_ANGLE_BASE..A_ANGLE_BASE + ANGLE_BINS]
    }
}

/// Pipelines, buffers and readback ring for the plane pass.
///
/// One instance serves every measurement plane in the scene: they share the
/// accumulator buffer (one `ACC_STRIDE` record each), the pipeline and the
/// readback, so adding a plane costs two dispatches and 768 bytes.
pub struct PlaneMetrics {
    pipeline: wgpu::ComputePipeline,
    /// `[slot * 2 + pass]`. Each holds one uniform buffer plus the shared
    /// accumulator, so a dispatch needs no dynamic offsets.
    groups: Vec<wgpu::BindGroup>,
    uniforms: Vec<wgpu::Buffer>,
    acc: wgpu::Buffer,
    ring: ReadbackRing<PlaneAccum>,
    max_planes: usize,
    samples: u32,
}

impl PlaneMetrics {
    /// `field_layout` must be [`crate::field::field_layout`]; `max_planes` is
    /// the largest number of patches a single [`Self::record`] may carry.
    pub fn new(
        device: &wgpu::Device,
        field_layout: &wgpu::BindGroupLayout,
        max_planes: usize,
    ) -> Result<Self> {
        Self::with_samples(device, field_layout, max_planes, DEFAULT_SAMPLES, 3)
    }

    pub fn with_samples(
        device: &wgpu::Device,
        field_layout: &wgpu::BindGroupLayout,
        max_planes: usize,
        samples: u32,
        depth: usize,
    ) -> Result<Self> {
        let max_planes = max_planes.max(1);
        let samples = samples.max(1);

        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("metrics plane group1"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let loader = shaders::build_loader();
        let module = loader.create_module(device, "metrics/plane.wgsl", &shaders::defines())?;
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("metrics plane layout"),
            bind_group_layouts: &[Some(field_layout), Some(&group1_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("metrics plane"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("plane"),
            compilation_options: Default::default(),
            cache: None,
        });

        let acc = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metrics plane accumulator"),
            size: (ACC_STRIDE * 4 * max_planes) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // One uniform buffer and one bind group per (slot, pass). Dynamic
        // offsets would save the allocations but would drag in
        // `min_uniform_buffer_offset_alignment`, and with at most a handful of
        // planes there is nothing to save.
        let mut uniforms = Vec::with_capacity(max_planes * 2);
        let mut groups = Vec::with_capacity(max_planes * 2);
        for _ in 0..max_planes * 2 {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("metrics plane uniforms"),
                size: std::mem::size_of::<PlaneUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metrics plane group1"),
                layout: &group1_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: acc.as_entire_binding() },
                ],
            }));
            uniforms.push(buf);
        }

        Ok(Self {
            pipeline,
            groups,
            uniforms,
            acc,
            ring: ReadbackRing::new(device, "metrics plane", max_planes, depth),
            max_planes,
            samples,
        })
    }

    pub fn max_planes(&self) -> usize {
        self.max_planes
    }

    pub fn samples_per_side(&self) -> u32 {
        self.samples
    }

    /// Samples dispatched per plane, `N^2`.
    pub fn samples_per_plane(&self) -> u64 {
        self.samples as u64 * self.samples as u64
    }

    pub fn has_capacity(&self) -> bool {
        self.ring.has_capacity()
    }

    /// Record both passes for every patch, plus the readback copy.
    ///
    /// The accumulator is reset through `queue.write_buffer` rather than
    /// `clear_buffer` because three of its slots need `+/-1e30` sentinels, not
    /// zero — see [`PlaneAccum::initial`]. wgpu applies queued writes before the
    /// command buffers of the next submit, so **the encoder must be submitted on
    /// the same queue before anything else touches the accumulator**.
    ///
    /// Returns `false` when the readback ring is saturated, in which case
    /// nothing was recorded: the GPU is more than `depth` frames behind and this
    /// measurement would have been stale anyway.
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        field_group: &wgpu::BindGroup,
        grid: Grid,
        patches: &[FlowPatch],
        step: u64,
        profiler: Option<&mut ad_gpu::Profiler>,
    ) -> Result<bool> {
        ensure!(
            patches.len() <= self.max_planes,
            "{} patches exceeds the {} this PlaneMetrics was built for",
            patches.len(),
            self.max_planes
        );
        if patches.is_empty() || !self.ring.has_capacity() {
            return Ok(false);
        }

        let init = vec![PlaneAccum::initial(); self.max_planes];
        queue.write_buffer(&self.acc, 0, bytemuck::cast_slice(&init));

        for (slot, patch) in patches.iter().enumerate() {
            for pass in 0..2u32 {
                let u = PlaneUniforms {
                    grid: GpuGrid::new(grid),
                    center_mm: patch.center_mm.to_array(),
                    pass_index: pass,
                    half_u: patch.half_u.to_array(),
                    slot: slot as u32,
                    half_v: patch.half_v.to_array(),
                    samples: self.samples,
                    normal: patch.normal.normalize_or_zero().to_array(),
                    pad: 0.0,
                };
                queue.write_buffer(
                    &self.uniforms[slot * 2 + pass as usize],
                    0,
                    bytemuck::bytes_of(&u),
                );
            }
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("metrics plane"),
                timestamp_writes: profiler.and_then(|p| p.scope("metrics planes")),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, field_group, &[]);
            // A fixed launch that the shader strides across; see
            // REDUCE_GROUPS_PER_SIDE.
            let groups = self.samples.div_ceil(8).min(REDUCE_GROUPS_PER_SIDE);
            // Pass 0 for every plane, then pass 1 for every plane. WebGPU orders
            // dispatches inside a compute pass and inserts the storage barrier,
            // so pass 1 sees pass 0's sums with no explicit synchronisation.
            for p in 0..2usize {
                for slot in 0..patches.len() {
                    pass.set_bind_group(1, &self.groups[slot * 2 + p], &[]);
                    pass.dispatch_workgroups(groups, groups, 1);
                }
            }
        }

        Ok(self.ring.record(encoder, &self.acc, 0, step))
    }

    /// Non-blocking. Each frame carries one [`PlaneAccum`] per slot.
    pub fn poll(&mut self, device: &wgpu::Device) -> Vec<Frame<PlaneAccum>> {
        self.ring.poll(device)
    }

    /// Blocking drain. Tests and batch runs only.
    pub fn drain_blocking(&mut self, device: &wgpu::Device) -> Vec<Frame<PlaneAccum>> {
        self.ring.drain_blocking(device)
    }

    pub fn dropped(&self) -> u64 {
        self.ring.dropped()
    }
}

/// Everything one plane record means, in SI.
///
/// Every field here is one *instantaneous* measurement. Feed them to
/// [`crate::stats::Series`] (which [`crate::metrics::DuctMetrics`] does) before
/// quoting anything: a turbulent duct never settles, so a single frame is a
/// sample from a distribution and not a measurement of the duct.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneReading {
    /// Area of the patch *rectangle*, mm^2.
    pub patch_area_mm2: f64,
    /// Fraction of that rectangle whose samples landed in fluid.
    pub covered_fraction: f64,
    /// Fraction that landed outside the grid entirely. Non-zero means the patch
    /// hangs off the domain, which is a placement error, not a measurement.
    pub outside_fraction: f64,
    /// True open area, mm^2: the hole, not the rectangle.
    pub open_area_mm2: f64,
    /// Hydraulic diameter of the open area on the assumption of a wide slot,
    /// `4A/P`, with the perimeter taken from the patch rectangle. Reported for
    /// the Reynolds number only.
    pub hydraulic_diameter_mm: f64,

    /// Volumetric flow through the plane, m^3/s. Signed: positive along the
    /// patch normal. This is what a flow meter reads.
    pub flow_rate_m3s: f64,
    /// Mass flow through the plane, kg/s. Signed like [`Self::flow_rate_m3s`].
    ///
    /// `mdot = sum(rho w) dA`, from the same accumulator slot the mass-flow
    /// weighting already uses, so it costs nothing extra to report.
    ///
    /// **This, not the volumetric flow, is the conserved quantity.** The solver
    /// is weakly compressible: across the test part's ~100 Pa the lattice
    /// density falls by 7%, so `Q_out` genuinely exceeds `Q_in` by 7% with not a
    /// milligram of mass lost. Differencing the volumetric flows therefore
    /// reports physics as though it were error, which is why
    /// [`crate::metrics::DuctMetrics`] gates on the mass imbalance instead.
    pub mass_flow_kgs: f64,
    /// `Q / A_open`, m/s. The bulk velocity an engineer would compute by hand.
    pub bulk_velocity_ms: f64,
    /// Mass-flow-weighted mean total pressure, Pa.
    ///
    /// `p_t = sum(rho w p_t) / sum(rho w)`. This, not the area average, is the
    /// one to difference across a duct: area weighting counts a stagnant or
    /// reversed corner as though it carried as much air as the core, which
    /// inflates the apparent total-pressure drop whenever there is any backflow.
    pub total_pressure_pa: f64,
    /// Area-weighted mean total pressure, Pa. Reported only so the gap between
    /// the two can be seen; it is the wrong number to quote.
    pub total_pressure_area_pa: f64,
    /// Area-weighted mean static pressure, Pa.
    pub static_pressure_pa: f64,

    /// Weltens uniformity index, `gamma = 1 - sum(|w_i - w_bar| A_i)/(2 A w_bar)`.
    /// 1 is a perfectly flat profile.
    pub uniformity: f64,
    /// Coefficient of variation of the through-plane velocity, `sigma / w_bar`.
    ///
    /// From the raw moments, `sigma^2 = <w^2> - <w>^2`, which is a cancellation
    /// of two nearly equal fp32 sums. On a real duct profile (`CV = 0.2..0.5`)
    /// that costs nothing; on an almost perfectly flat profile the two sums
    /// agree to the accumulator's precision and the result floors out around
    /// `CV ~ 0.01` rather than reaching zero. Read a CV below a percent as
    /// "flat", not as a measurement. [`Self::uniformity`] has no such floor,
    /// because `sum |w - w_bar|` is accumulated directly in the second pass.
    pub cv: f64,
    /// 5th and 95th percentile of the through-plane velocity, m/s, by area.
    pub p05_ms: f64,
    pub p95_ms: f64,
    /// Fraction of the *covered* area flowing backwards through the plane.
    pub backflow_fraction: f64,
    /// Peak speed anywhere on the plane, m/s.
    pub max_speed_ms: f64,
    /// Mean speed on the plane, m/s.
    pub mean_speed_ms: f64,

    /// Unit mass-flux-weighted momentum direction, `J/|J|` with
    /// `J = sum rho u (u.n) dA`.
    pub momentum_dir: Vec3,
    /// Angle between [`Self::momentum_dir`] and the patch normal, degrees.
    pub deflection_deg: f64,
    /// Half-angle of the cone about the momentum direction containing 90% of
    /// the forward momentum flux, degrees.
    pub cone_half_angle_deg: f64,

    /// Through-plane velocity histogram as `(bin centre m/s, area fraction)`.
    pub histogram: Vec<(f64, f64)>,
    /// Range the histogram spans, m/s.
    pub hist_min_ms: f64,
    pub hist_max_ms: f64,

    /// Samples that landed in fluid. Zero means every reading above is a
    /// placeholder and nothing on this plane should be shown.
    pub fluid_samples: u32,
}

impl PlaneReading {
    /// Interpret one accumulator record.
    ///
    /// `patch` must be the same patch the record was dispatched over, and `lu`
    /// the lattice units in force at the time. Both are CPU-side, so the shader
    /// never carries a unit conversion.
    pub fn from_accum(acc: &PlaneAccum, patch: &FlowPatch, lu: &LatticeUnits) -> Self {
        let n_total = acc.n_total().max(1) as f64;
        let n_fluid = acc.n_fluid();
        let nf = n_fluid as f64;

        let patch_area_mm2 = patch.area_mm2() as f64;
        let covered_fraction = nf / n_total;
        let outside_fraction = (acc.n_total() - acc.n_inside()) as f64 / n_total;
        let open_area_mm2 = patch_area_mm2 * covered_fraction;
        // dA per sample. Solid samples own their share of the rectangle too;
        // they simply carry zero flow, which is what makes Q come out right.
        let d_a_mm2 = patch_area_mm2 / n_total;

        let c_u = lu.c_u();
        // Lattice velocity * mm^2 -> m^3/s.
        let to_flow = d_a_mm2 * 1e-6 * c_u;

        if n_fluid == 0 {
            return Self {
                patch_area_mm2,
                covered_fraction: 0.0,
                outside_fraction,
                open_area_mm2: 0.0,
                hydraulic_diameter_mm: 0.0,
                flow_rate_m3s: 0.0,
                mass_flow_kgs: 0.0,
                bulk_velocity_ms: 0.0,
                total_pressure_pa: 0.0,
                total_pressure_area_pa: 0.0,
                static_pressure_pa: 0.0,
                uniformity: 0.0,
                cv: 0.0,
                p05_ms: 0.0,
                p95_ms: 0.0,
                backflow_fraction: 0.0,
                max_speed_ms: 0.0,
                mean_speed_ms: 0.0,
                momentum_dir: patch.normal.normalize_or_zero(),
                deflection_deg: 0.0,
                cone_half_angle_deg: 0.0,
                histogram: Vec::new(),
                hist_min_ms: 0.0,
                hist_max_ms: 0.0,
                fluid_samples: 0,
            };
        }

        let sum_w = acc.scalar(S_W) as f64;
        let sum_w2 = acc.scalar(S_W2) as f64;
        let sum_rho_w = acc.scalar(S_RHO_W) as f64;
        let sum_mdot_pt = acc.scalar(S_MDOT_PT) as f64;
        let w_bar = sum_w / nf;

        let flow_rate_m3s = sum_w * to_flow;
        // Lattice density is normalised so that rho = 1 is `rho_phys`, so the
        // one extra factor of rho_phys turns the same integral into kg/s. No
        // other conversion appears here, which is what keeps the mass flow and
        // the volumetric flow exactly consistent with the density that relates
        // them.
        let mass_flow_kgs = sum_rho_w * to_flow * lu.rho_phys;
        let bulk_velocity_ms =
            if open_area_mm2 > 0.0 { flow_rate_m3s / (open_area_mm2 * 1e-6) } else { 0.0 };

        // Both pressures are lattice pressures already (p = c_s^2 (rho - 1) and
        // p_t = p + rho|u|^2/2), so the single factor rho_phys (dx/dt)^2 turns
        // either into pascals. LatticeUnits::pressure_pa folds in the c_s^2 that
        // is already applied here, so use the raw factor instead.
        let p_factor = lu.rho_phys * c_u * c_u;
        let total_pressure_pa = if sum_rho_w.abs() > 1e-30 {
            sum_mdot_pt / sum_rho_w * p_factor
        } else {
            acc.scalar(S_PT) as f64 / nf * p_factor
        };
        let total_pressure_area_pa = acc.scalar(S_PT) as f64 / nf * p_factor;
        let static_pressure_pa = acc.scalar(S_PS) as f64 / nf * p_factor;

        // Weltens: with equal-area samples, A_i = dA and A = n_fluid dA, so the
        // area elements cancel and only the sample count survives.
        //   gamma = 1 - sum|w_i - w_bar| / (2 n_fluid |w_bar|)
        let uniformity = if w_bar.abs() > 1e-30 {
            1.0 - acc.scalar(S_ABS_DEV) as f64 / (2.0 * nf * w_bar.abs())
        } else {
            0.0
        };
        let variance = (sum_w2 / nf - w_bar * w_bar).max(0.0);
        let cv = if w_bar.abs() > 1e-30 { variance.sqrt() / w_bar.abs() } else { 0.0 };

        let backflow_fraction = acc.n_backflow() as f64 / nf;
        let max_speed_ms = acc.scalar(S_MAX_SPEED) as f64 * c_u;
        let mean_speed_ms = acc.scalar(S_SPEED) as f64 / nf * c_u;

        let j = Vec3::new(acc.scalar(S_JX), acc.scalar(S_JY), acc.scalar(S_JZ));
        let n = patch.normal.normalize_or_zero();
        let momentum_dir = if j.length_squared() > 0.0 { j.normalize() } else { n };
        let deflection_deg =
            (momentum_dir.dot(n).clamp(-1.0, 1.0) as f64).acos().to_degrees();

        let lo = acc.scalar(S_MIN_W) as f64 * c_u;
        let hi = acc.scalar(S_MAX_W) as f64 * c_u;
        let (histogram, p05_ms, p95_ms) = velocity_histogram(acc.histogram(), lo, hi);
        let cone_half_angle_deg = cone_half_angle(acc.angle_histogram(), 0.9);

        // 4A/P for the patch rectangle. Only ever used for a Reynolds number,
        // which nobody reads to three figures.
        let (a, b) = (
            2.0 * patch.half_u.length() as f64,
            2.0 * patch.half_v.length() as f64,
        );
        let perimeter = 2.0 * (a + b);
        let hydraulic_diameter_mm =
            if perimeter > 0.0 { 4.0 * open_area_mm2 / perimeter } else { 0.0 };

        Self {
            patch_area_mm2,
            covered_fraction,
            outside_fraction,
            open_area_mm2,
            hydraulic_diameter_mm,
            flow_rate_m3s,
            mass_flow_kgs,
            bulk_velocity_ms,
            total_pressure_pa,
            total_pressure_area_pa,
            static_pressure_pa,
            uniformity,
            cv,
            p05_ms,
            p95_ms,
            backflow_fraction,
            max_speed_ms,
            mean_speed_ms,
            momentum_dir,
            deflection_deg,
            cone_half_angle_deg,
            histogram,
            hist_min_ms: lo,
            hist_max_ms: hi,
            fluid_samples: n_fluid,
        }
    }

    /// Through-flow-weighted mean density on the plane, kg/m^3, or `None` when
    /// no net flow crossed it.
    ///
    /// `mdot / Q = sum(rho w) / sum(w) * rho_phys`. It is the density that
    /// relates this plane's two flow rates, so the ratio of two planes' values
    /// is exactly the volumetric expansion between them — which is the honest
    /// way to say how much of a volumetric imbalance is compressibility rather
    /// than lost mass. Taking it from the static pressure instead would mix an
    /// area-weighted mean into a pair of flux integrals.
    pub fn mean_density_kgm3(&self) -> Option<f64> {
        if self.flow_rate_m3s.abs() > 0.0 && self.mass_flow_kgs.is_finite() {
            Some(self.mass_flow_kgs / self.flow_rate_m3s)
        } else {
            None
        }
    }

    /// Volumetric flow in cubic feet per minute.
    pub fn cfm(&self) -> f64 {
        self.flow_rate_m3s * 2118.88
    }

    /// Volumetric flow in litres per second.
    pub fn litres_per_second(&self) -> f64 {
        self.flow_rate_m3s * 1000.0
    }

    /// The shared [`ad_gpu::MetricSample`] form, for anything that wants the
    /// contract's reduced record rather than this crate's full one.
    pub fn metric_sample(&self, step: u32) -> MetricSample {
        MetricSample {
            flow_rate: self.flow_rate_m3s as f32,
            total_pressure: self.total_pressure_pa as f32,
            static_pressure: self.static_pressure_pa as f32,
            uniformity: self.uniformity as f32,
            momentum_dir: self.momentum_dir.to_array(),
            max_speed: self.max_speed_ms as f32,
            backflow_fraction: self.backflow_fraction as f32,
            step,
            _pad: [0; 2],
        }
    }
}

/// Turn the exact bin counts into `(centre, area fraction)` pairs and pull the
/// 5th and 95th percentiles out of the same cumulative sum.
///
/// Percentiles are interpolated inside the containing bin. With 128 bins over
/// the measured range that puts the quantisation error under 1% of the spread,
/// which is finer than the difference between P95 and P95 ever matters.
fn velocity_histogram(bins: &[u32], lo: f64, hi: f64) -> (Vec<(f64, f64)>, f64, f64) {
    let total: u64 = bins.iter().map(|&c| c as u64).sum();
    if total == 0 || !(hi > lo) {
        return (Vec::new(), lo, hi);
    }
    let width = (hi - lo) / bins.len() as f64;
    let out: Vec<(f64, f64)> = bins
        .iter()
        .enumerate()
        .map(|(i, &c)| (lo + (i as f64 + 0.5) * width, c as f64 / total as f64))
        .collect();

    let quantile = |q: f64| -> f64 {
        let target = q * total as f64;
        let mut acc = 0.0;
        for (i, &c) in bins.iter().enumerate() {
            let c = c as f64;
            if acc + c >= target && c > 0.0 {
                return lo + (i as f64 + (target - acc) / c) * width;
            }
            acc += c;
        }
        hi
    };
    (out, quantile(0.05), quantile(0.95))
}

/// Half-angle of the cone about the mean momentum direction that contains
/// `frac` of the forward momentum flux.
///
/// The shader binned each sample's share of the flux by its angle to the mean
/// direction, in fixed point. Walking the cumulative distribution and
/// interpolating inside the bin gives a half-angle whose resolution is better
/// than the 5.6 degrees one of 32 bins spans.
///
/// This is the number that says whether a duct throws a *jet* or a *spray*: two
/// outlets with identical flow and identical mean direction can differ by 20
/// degrees here, and the wide one will feel like a draught rather than a beam.
fn cone_half_angle(bins: &[u32], frac: f64) -> f64 {
    let total: u64 = bins.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0.0;
    }
    let bin_deg = 180.0 / bins.len() as f64;
    let target = frac * total as f64;
    let mut acc = 0.0;
    for (i, &c) in bins.iter().enumerate() {
        let c = c as f64;
        if acc + c >= target && c > 0.0 {
            return (i as f64 + (target - acc) / c) * bin_deg;
        }
        acc += c;
    }
    180.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::types::{flags, Bbox};
    use crate::field::{field_layout, FieldRefs, FieldTextures};

    /// A cube of grid covering `[0, size]` mm at `dx`.
    fn grid(size: f32, dx: f32) -> Grid {
        Grid::covering(Bbox { min: Vec3::ZERO, max: Vec3::splat(size) }, dx)
    }

    fn units() -> LatticeUnits {
        // dx = 1 mm, U = 2 m/s, u_lb = 0.05. c_u = 40 m/s per lattice unit.
        LatticeUnits::for_air(1.0, 2.0, 0.05)
    }

    /// Run one plane over a synthetic field and return the reading.
    fn measure(
        gpu: &ad_gpu::GpuContext,
        g: Grid,
        patch: FlowPatch,
        samples: u32,
        fill: impl Fn(glam::UVec3, Vec3) -> (Vec3, f32, u8),
    ) -> PlaneReading {
        let tex = FieldTextures::new_exact(&gpu.device, g);
        tex.fill(&gpu.queue, fill).unwrap();
        let layout = field_layout(&gpu.device);
        let (vv, dv) = (tex.velocity_view(), tex.density_view());
        let group = crate::field::field_bind_group(
            &gpu.device,
            &layout,
            &FieldRefs { grid: g, velocity: &vv, density: &dv, flags: tex.flags_buffer() },
        );
        let mut pm = PlaneMetrics::with_samples(&gpu.device, &layout, 1, samples, 2).unwrap();
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(pm.record(&gpu.queue, &mut enc, &group, g, &[patch], 1, None).unwrap());
        gpu.queue.submit([enc.finish()]);
        let frames = pm.drain_blocking(&gpu.device);
        assert!(!frames.is_empty(), "no plane readback arrived");
        PlaneReading::from_accum(&frames[0].data[0], &patch, &units())
    }

    #[test]
    fn accumulator_sentinels_survive_a_round_trip_through_bit_patterns() {
        let a = PlaneAccum::initial();
        assert_eq!(a.n_total(), 0);
        assert!(a.scalar(S_MAX_W) < -1.0e29, "max sentinel was {}", a.scalar(S_MAX_W));
        assert!(a.scalar(S_MIN_W) > 1.0e29);
        assert_eq!(a.scalar(S_W), 0.0);
        assert_eq!(a.histogram().len(), HIST_BINS);
        assert_eq!(a.angle_histogram().len(), ANGLE_BINS);
        assert_eq!(std::mem::size_of::<PlaneAccum>(), ACC_STRIDE * 4);
    }

    #[test]
    fn a_histogram_of_one_value_puts_both_percentiles_on_it() {
        let mut bins = vec![0u32; HIST_BINS];
        bins[64] = 1000;
        let (h, p05, p95) = velocity_histogram(&bins, 0.0, 1.0);
        assert_eq!(h.len(), HIST_BINS);
        assert!((h[64].1 - 1.0).abs() < 1e-12, "all the weight should be in one bin");
        assert!((p05 - 0.5).abs() < 0.01 && (p95 - 0.5).abs() < 0.01, "{p05} {p95}");
    }

    #[test]
    fn a_uniform_angle_distribution_gives_the_right_cone() {
        // All weight in the first four bins of 32: 90% of it lies inside
        // 4 * 5.625 = 22.5 degrees, and the interpolation should land there.
        let mut bins = vec![0u32; ANGLE_BINS];
        for b in bins.iter_mut().take(4) {
            *b = 100;
        }
        let a = cone_half_angle(&bins, 0.9);
        assert!((a - 20.25).abs() < 0.2, "cone half-angle was {a}");
        assert_eq!(cone_half_angle(&[0; ANGLE_BINS], 0.9), 0.0);
    }

    /// The exactness test: `Q = U A cos(theta)` for a uniform field through a
    /// tilted plane, for a spread of tilts. Nothing about this depends on the
    /// lattice, so any error is quadrature or unit conversion.
    #[test]
    fn uniform_flow_through_a_tilted_plane_gives_u_a_cos_theta() {
        let Some(gpu) = crate::test_gpu() else { return };
        let g = grid(64.0, 1.0);
        let lu = units();
        let u_lb = 0.05f32;
        let flow = Vec3::X;

        for theta_deg in [0.0f32, 15.0, 30.0, 45.0, 60.0] {
            let theta = theta_deg.to_radians();
            // Tilt the plane about Z: the normal leans away from the flow.
            let n = Vec3::new(theta.cos(), theta.sin(), 0.0);
            let t = Vec3::new(-theta.sin(), theta.cos(), 0.0);
            let half_u = t * 12.0;
            let half_v = Vec3::Z * 12.0;
            let patch = FlowPatch {
                center_mm: Vec3::splat(32.0),
                normal: n,
                half_u,
                half_v,
            };
            let r = measure(&gpu, g, patch, 256, |_, _| (flow * u_lb, 1.0, flags::FLUID));

            let area_m2 = patch.area_mm2() as f64 * 1e-6;
            let want = (u_lb as f64 * lu.c_u()) * area_m2 * theta.cos() as f64;
            assert!(
                (r.flow_rate_m3s - want).abs() / want < 2e-3,
                "theta = {theta_deg}: Q = {} vs analytic {want}",
                r.flow_rate_m3s
            );
            assert!((r.covered_fraction - 1.0).abs() < 1e-9);
            // A uniform field is perfectly uniform and has no spread. The
            // uniformity index is exact because its numerator is accumulated
            // directly; the CV floors out near 1% on a flat profile because it
            // is a cancellation of two nearly equal fp32 sums. See the note on
            // PlaneReading::cv.
            assert!((r.uniformity - 1.0).abs() < 1e-4, "gamma = {}", r.uniformity);
            assert!(r.cv < 0.01, "cv = {}", r.cv);
            assert_eq!(r.backflow_fraction, 0.0);
            // ...and its momentum points exactly along the flow.
            assert!(
                r.momentum_dir.dot(flow) > 0.999_9,
                "momentum {:?} should be +X",
                r.momentum_dir
            );
            assert!(
                (r.deflection_deg - theta_deg as f64).abs() < 0.05,
                "deflection {} vs {theta_deg}",
                r.deflection_deg
            );
        }
    }

    /// Plane Poiseuille flow has closed forms for both shape statistics:
    /// `CV = 1/sqrt(5)` and `gamma = 1 - 1/(3 sqrt 3)`. Neither depends on the
    /// peak velocity, the channel height or the units, which is what makes them
    /// such a good check on the two-pass reduction.
    #[test]
    fn poiseuille_reproduces_the_closed_form_uniformity_and_cv() {
        let Some(gpu) = crate::test_gpu() else { return };
        // Channel between y = 0 and y = H, flow along +X.
        let h = 40.0f32;
        let g = grid(48.0, 0.25);
        let u_max = 0.06f32;
        let patch = FlowPatch {
            center_mm: Vec3::new(24.0, h * 0.5, 24.0),
            normal: Vec3::X,
            // Exactly the channel height, so the quadrature covers y in [0, H].
            half_u: Vec3::Y * (h * 0.5),
            half_v: Vec3::Z * 12.0,
        };
        let r = measure(&gpu, g, patch, 512, |_, p| {
            let eta = p.y / h;
            let u = 4.0 * u_max * eta * (1.0 - eta);
            (Vec3::new(u.max(0.0), 0.0, 0.0), 1.0, flags::FLUID)
        });

        let want_cv = 1.0 / 5.0f64.sqrt();
        let want_gamma = 1.0 - 1.0 / (3.0 * 3.0f64.sqrt());
        assert!(
            (r.cv - want_cv).abs() < 0.01,
            "CV = {:.4}, analytic {:.4}",
            r.cv,
            want_cv
        );
        assert!(
            (r.uniformity - want_gamma).abs() < 0.01,
            "gamma = {:.4}, analytic {:.4}",
            r.uniformity,
            want_gamma
        );
        // Bulk velocity is 2/3 of the peak for a parabola.
        let want_bulk = (2.0 / 3.0) * u_max as f64 * units().c_u();
        assert!(
            (r.bulk_velocity_ms / want_bulk - 1.0).abs() < 0.01,
            "bulk {} vs {want_bulk}",
            r.bulk_velocity_ms
        );
        // ...and the peak of the histogram range is the centreline velocity.
        assert!(
            (r.hist_max_ms / (u_max as f64 * units().c_u()) - 1.0).abs() < 0.02,
            "hist max {}",
            r.hist_max_ms
        );
    }

    /// Solid-body rotation about the plane normal carries no net flux at all,
    /// and every sample is in-plane. If the reduction leaked any component of
    /// the swirl into `w`, this is where it would show.
    #[test]
    fn solid_body_rotation_passes_no_flow_through_its_own_axis() {
        let Some(gpu) = crate::test_gpu() else { return };
        let g = grid(64.0, 1.0);
        let centre = Vec3::splat(32.0);
        let omega = 0.002f32;
        let patch = FlowPatch {
            center_mm: centre,
            normal: Vec3::Z,
            half_u: Vec3::X * 16.0,
            half_v: Vec3::Y * 16.0,
        };
        let r = measure(&gpu, g, patch, 256, |_, p| {
            let d = p - centre;
            (Vec3::new(-omega * d.y, omega * d.x, 0.0), 1.0, flags::FLUID)
        });
        // Q is zero to the precision of the sum, not merely small.
        let scale = (omega as f64 * 16.0 * units().c_u()) * (patch.area_mm2() as f64 * 1e-6);
        assert!(
            r.flow_rate_m3s.abs() < 1e-6 * scale.max(1e-12),
            "Q = {} for pure swirl (scale {scale})",
            r.flow_rate_m3s
        );
        // Peak speed is omega * R at the rectangle's corner.
        let want_peak = omega as f64 * (16.0f64 * 2.0f64.sqrt()) * units().c_u();
        assert!(
            (r.max_speed_ms / want_peak - 1.0).abs() < 0.02,
            "peak {} vs {want_peak}",
            r.max_speed_ms
        );
    }

    /// A linear shear `u = (a y, 0, 0)` through an x-normal plane integrates to
    /// `Q = a y_c A` exactly, and half the plane flows backwards when the plane
    /// straddles `y = 0`.
    #[test]
    fn linear_shear_integrates_to_the_centreline_value_and_splits_the_backflow() {
        let Some(gpu) = crate::test_gpu() else { return };
        let g = grid(64.0, 1.0);
        let a = 0.001f32; // lattice velocity per mm
        let y0 = 32.0f32;

        // Centred on the zero crossing: mean w = 0, exactly half reversed.
        let patch = FlowPatch {
            center_mm: Vec3::new(32.0, y0, 32.0),
            normal: Vec3::X,
            half_u: Vec3::Y * 16.0,
            half_v: Vec3::Z * 16.0,
        };
        let r = measure(&gpu, g, patch, 256, |_, p| {
            (Vec3::new(a * (p.y - y0), 0.0, 0.0), 1.0, flags::FLUID)
        });
        let scale = (a as f64 * 16.0 * units().c_u()) * (patch.area_mm2() as f64 * 1e-6);
        assert!(r.flow_rate_m3s.abs() < 1e-5 * scale, "Q = {}", r.flow_rate_m3s);
        assert!(
            (r.backflow_fraction - 0.5).abs() < 0.01,
            "backflow {} should be half the plane",
            r.backflow_fraction
        );

        // Offset so the whole plane flows forward: Q = a * (y_c - y0) * A.
        let offset = 8.0f32;
        let patch2 = FlowPatch {
            center_mm: Vec3::new(32.0, y0 + offset, 32.0),
            normal: Vec3::X,
            half_u: Vec3::Y * 4.0,
            half_v: Vec3::Z * 16.0,
        };
        let r2 = measure(&gpu, g, patch2, 256, |_, p| {
            (Vec3::new(a * (p.y - y0), 0.0, 0.0), 1.0, flags::FLUID)
        });
        let want =
            (a * offset) as f64 * units().c_u() * (patch2.area_mm2() as f64 * 1e-6);
        assert!(
            (r2.flow_rate_m3s / want - 1.0).abs() < 2e-3,
            "Q = {} vs analytic {want}",
            r2.flow_rate_m3s
        );
        assert_eq!(r2.backflow_fraction, 0.0);
    }

    /// Half the patch buried in wall: the flow rate must halve and the covered
    /// fraction must say so. This is the check that stops a clipped patch from
    /// quietly under-reading.
    #[test]
    fn a_patch_clipping_geometry_reports_its_coverage_and_halves_the_flow() {
        let Some(gpu) = crate::test_gpu() else { return };
        let g = grid(64.0, 0.5);
        let u_lb = 0.05f32;
        let patch = FlowPatch {
            center_mm: Vec3::splat(32.0),
            normal: Vec3::X,
            half_u: Vec3::Y * 10.0,
            half_v: Vec3::Z * 10.0,
        };
        // Solid everywhere above the patch centre line.
        let r = measure(&gpu, g, patch, 256, |_, p| {
            if p.y > 32.0 {
                (Vec3::ZERO, 1.0, flags::SOLID)
            } else {
                (Vec3::X * u_lb, 1.0, flags::FLUID)
            }
        });
        assert!(
            (r.covered_fraction - 0.5).abs() < 0.02,
            "covered fraction {}",
            r.covered_fraction
        );
        let full = u_lb as f64 * units().c_u() * (patch.area_mm2() as f64 * 1e-6);
        assert!(
            (r.flow_rate_m3s / (0.5 * full) - 1.0).abs() < 0.03,
            "Q = {} vs half of {full}",
            r.flow_rate_m3s
        );
        // The open area is the hole, not the rectangle.
        assert!(
            (r.open_area_mm2 / (patch.area_mm2() as f64 * 0.5) - 1.0).abs() < 0.03,
            "open area {}",
            r.open_area_mm2
        );
        // ...and the bulk velocity is still the true one, because it divides by
        // the open area rather than the rectangle.
        assert!(
            (r.bulk_velocity_ms / (u_lb as f64 * units().c_u()) - 1.0).abs() < 0.03,
            "bulk {}",
            r.bulk_velocity_ms
        );
    }

    /// Total pressure differs from static by exactly `rho |u|^2 / 2`, and the
    /// mass-flow weighting must not change that for a uniform field.
    #[test]
    fn total_pressure_exceeds_static_by_the_dynamic_head() {
        let Some(gpu) = crate::test_gpu() else { return };
        let g = grid(32.0, 1.0);
        let lu = units();
        let u_lb = 0.05f32;
        let patch = FlowPatch {
            center_mm: Vec3::splat(16.0),
            normal: Vec3::X,
            half_u: Vec3::Y * 6.0,
            half_v: Vec3::Z * 6.0,
        };
        let r = measure(&gpu, g, patch, 128, |_, _| (Vec3::X * u_lb, 1.0, flags::FLUID));

        let u_ms = u_lb as f64 * lu.c_u();
        let want_dynamic = 0.5 * lu.rho_phys * u_ms * u_ms;
        let got = r.total_pressure_pa - r.static_pressure_pa;
        assert!(
            (got / want_dynamic - 1.0).abs() < 1e-3,
            "dynamic head {got} vs {want_dynamic}"
        );
        // rho = 1 exactly, so the static pressure is the reference: zero.
        assert!(r.static_pressure_pa.abs() < 1e-6, "p_s = {}", r.static_pressure_pa);
        // Uniform flow: the two weightings agree, to the precision of two
        // independent fp32 accumulations over 16,384 samples.
        assert!(
            (r.total_pressure_pa - r.total_pressure_area_pa).abs() / want_dynamic < 1e-4,
            "weightings disagree on a uniform field: {} vs {}",
            r.total_pressure_pa,
            r.total_pressure_area_pa
        );
    }

    /// The mass flow is the volumetric flow times the local density, and on a
    /// compressed plane the two are *not* interchangeable. This is the reading
    /// the mass balance is built on, so it has to be right at `rho != 1`.
    #[test]
    fn mass_flow_is_the_volume_flow_times_the_density_on_the_plane() {
        let Some(gpu) = crate::test_gpu() else { return };
        let g = grid(32.0, 1.0);
        let lu = units();
        let (u_lb, rho) = (0.05f32, 1.05f32);
        let patch = FlowPatch {
            center_mm: Vec3::splat(16.0),
            normal: Vec3::X,
            half_u: Vec3::Y * 6.0,
            half_v: Vec3::Z * 6.0,
        };
        let r = measure(&gpu, g, patch, 128, |_, _| (Vec3::X * u_lb, rho, flags::FLUID));

        let want = rho as f64 * lu.rho_phys * r.flow_rate_m3s;
        assert!(
            (r.mass_flow_kgs / want - 1.0).abs() < 1e-4,
            "mdot = {} vs rho Q = {want}",
            r.mass_flow_kgs
        );
        // ...and the density that relates them comes back out.
        let d = r.mean_density_kgm3().expect("a flowing plane has a mean density");
        assert!((d / (rho as f64 * lu.rho_phys) - 1.0).abs() < 1e-4, "rho_bar = {d}");
    }

    /// An empty plane reports nothing rather than a plausible zero with a
    /// -1e30 peak velocity leaking out of the sentinel.
    #[test]
    fn a_plane_entirely_inside_solid_reports_no_samples_and_no_nonsense() {
        let Some(gpu) = crate::test_gpu() else { return };
        let g = grid(16.0, 1.0);
        let patch = FlowPatch {
            center_mm: Vec3::splat(8.0),
            normal: Vec3::X,
            half_u: Vec3::Y * 4.0,
            half_v: Vec3::Z * 4.0,
        };
        let r = measure(&gpu, g, patch, 64, |_, _| (Vec3::ZERO, 1.0, flags::SOLID));
        assert_eq!(r.fluid_samples, 0);
        assert_eq!(r.covered_fraction, 0.0);
        assert_eq!(r.flow_rate_m3s, 0.0);
        assert_eq!(r.max_speed_ms, 0.0, "the -1e30 sentinel escaped");
        assert!(r.histogram.is_empty());
    }
}

//! The GPU solver: resources, pipelines and the step loop.
//!
//! # Shape of a step
//!
//! One dispatch. Esoteric Pull streams in place, so there is no ping-pong buffer
//! to swap and no separate streaming pass — the load is the stream. The only
//! per-step state is the parity bit, and even that costs nothing at run time:
//! two uniform buffers are written once, and stepping alternates between two
//! prebuilt bind groups. A run of `n` steps is `n` dispatches recorded into one
//! command buffer and submitted together, so the CPU touches nothing per step.
//!
//! # Why the grid is padded
//!
//! See [`crate::boundary`]. Every non-periodic axis gains one cell of solid halo
//! at each end, because a boundary cell parks half its populations in the slot
//! belonging to the cell outside the domain. The macroscopic textures are sized
//! to the caller's *interior* grid, so nothing downstream ever sees the halo.
//!
//! # Bind groups
//!
//! | group | contents |
//! |---|---|
//! | 0 | uniforms, flag bytes, link mask |
//! | 1 | the `q` DDF buffers, one per lattice direction |
//! | 2 | macroscopic outputs (macroscopic pass only) |
//!
//! Group 0 exists twice, once per step parity. Group 1 is
//! [`ad_gpu::DdfBuffers`]' own layout, unchanged.

use std::sync::Arc;

use ad_gpu::types::{flags, BoundaryLink, DdfPrecision, Grid, VelocitySet};
use ad_gpu::{DdfBuffers, GpuContext, Profiler, ShaderDefines, ShaderLoader};
use anyhow::{Context as _, Result};
use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec3};
use wgpu::util::DeviceExt as _;

use crate::boundary::{LinkTable, PaddedDomain};
use crate::config::SolverConfig;
use crate::{precision, shaders};

/// GPU mirror of the solver parameters.
///
/// [`ad_gpu::SimUniforms`] is the *shared* mirror, and stays exactly as the
/// contract defines it — nothing outside the solver needs a body force or a
/// periodicity mask, so those live here instead. [`SolverConfig::sim_uniforms`]
/// still produces the shared struct for anyone who wants it.
///
/// Layout note: every `vec3` in a WGSL uniform is 16-byte aligned, so each
/// `[T; 3]` here is followed by a scalar that fills the slot. Total 160 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LbmUniforms {
    pub dims: [u32; 3],
    pub step_parity: u32,
    pub interior: [u32; 3],
    pub cell_count: u32,
    pub offset: [u32; 3],
    pub flag_words: u32,
    pub inlet_velocity: [f32; 3],
    pub tau0: f32,
    pub initial_velocity: [f32; 3],
    pub trt_lambda: f32,
    pub body_force: [f32; 3],
    pub smagorinsky_c: f32,
    pub tau_max: f32,
    pub outflow_velocity: f32,
    pub rho_ref: f32,
    pub sponge_strength: f32,
    pub sponge_cells: u32,
    pub periodic: u32,
    pub total_steps: u32,
    pub _pad: u32,
    pub outlet_normal: [f32; 3],
    pub _pad1: f32,
    pub inlet_normal: [f32; 3],
    pub outlet_anti_bounce_back: u32,
    /// The four inlet slots (`ad_gpu::flags::inlet_slot`): xyz = lattice
    /// velocity, w > 0 = free-standing (density from the local moment). Slot 0
    /// repeats `inlet_velocity` / `inlet_normal`, so the shader reads one table.
    pub inlet_vel: [[f32; 4]; 4],
    /// xyz = inward normal of each slot's plane.
    pub inlet_nrm: [[f32; 4]; 4],
}

const _: () = assert!(std::mem::size_of::<LbmUniforms>() == 288);

/// Name of the profiler scope the step kernel reports under.
pub const STEP_SCOPE: &str = "stream_collide";
/// Name of the profiler scope the macroscopic pass reports under.
pub const MACRO_SCOPE: &str = "macroscopic";

pub struct Solver {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pub grid: Grid,
    pub domain: PaddedDomain,
    pub links: LinkTable,
    cfg: SolverConfig,

    ddf: DdfBuffers,
    uniform_buffers: [wgpu::Buffer; 2],
    flags_buffer: wgpu::Buffer,
    _link_buffer: wgpu::Buffer,
    boundary_link_buffer: wgpu::Buffer,

    group0: [wgpu::BindGroup; 2],
    group1: wgpu::BindGroup,
    group2: wgpu::BindGroup,

    init_pipeline: wgpu::ComputePipeline,
    step_pipeline: wgpu::ComputePipeline,
    macro_pipeline: wgpu::ComputePipeline,

    velocity_texture: wgpu::Texture,
    density_texture: wgpu::Texture,
    macro_buffer: Option<wgpu::Buffer>,
    macro_readback: Option<wgpu::Buffer>,

    profiler: Profiler,
    /// One profiler scope wraps a whole `step(n)` batch, and it is recorded per
    /// step (`Profiler::scope_per`) rather than divided by `n` afterwards:
    /// dividing a moving average of mixed batch lengths by the latest `n` read
    /// 4-5x high whenever the step tuner was moving.
    profiling: bool,
    steps: u64,
}

impl Solver {
    /// Build a solver for `grid`, with one [`flags`] byte per interior cell.
    ///
    /// `boundary_links` may be empty; it is uploaded and bound regardless so the
    /// Wave-3 interpolated bounce-back has its data path already in place.
    pub fn new(
        gpu: &GpuContext,
        grid: Grid,
        mask: &[u8],
        boundary_links: &[BoundaryLink],
        cfg: SolverConfig,
    ) -> Result<Self> {
        let domain = PaddedDomain::new(grid.dims, cfg.periodic, mask, cfg.set);
        let links = LinkTable::remap(boundary_links, grid.dims, &domain);

        // The DDF allocation is sized to the padded grid, since that is what the
        // index scheme addresses. The halo is 2/N of each axis: 2.5% of cells at
        // the interactive tier, and it buys branch-free boundary handling.
        let padded_grid = Grid {
            dims: domain.padded,
            ..grid
        };
        let ddf = DdfBuffers::allocate(
            &gpu.device,
            &gpu.limits,
            padded_grid,
            cfg.set,
            cfg.precision,
        )
        .context("allocating distribution function buffers")?;

        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        // ---- static per-cell data ----
        let flag_words = pack_flags(&domain.flags);
        let flags_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm flags"),
            contents: bytemuck::cast_slice(&flag_words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let link_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm link mask"),
            contents: bytemuck::cast_slice(&domain.link_mask),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // A zero-length storage buffer is invalid, so keep one dummy entry.
        let boundary_link_data: Vec<BoundaryLink> = if links.links.is_empty() {
            vec![BoundaryLink {
                cell: 0,
                direction: 0,
                q_quantised: 0,
                _pad: [0; 2],
            }]
        } else {
            links.links.clone()
        };
        let boundary_link_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm boundary links"),
            contents: bytemuck::cast_slice(&boundary_link_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffers = [0u32, 1].map(|parity| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if parity == 0 {
                    "lbm uniforms (even)"
                } else {
                    "lbm uniforms (odd)"
                }),
                size: std::mem::size_of::<LbmUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // ---- macroscopic outputs ----
        let extent = wgpu::Extent3d {
            width: grid.dims.x,
            height: grid.dims.y,
            depth_or_array_layers: grid.dims.z,
        };
        let make_texture = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                // STORAGE_BINDING to write here, TEXTURE_BINDING so render and
                // metrics can read through a *sampled* view. Two views of one
                // texture; storage textures are write-only in the portable spec.
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let velocity_texture = make_texture("lbm velocity", wgpu::TextureFormat::Rgba16Float);
        let density_texture = make_texture("lbm density", wgpu::TextureFormat::R32Float);

        let interior_cells = domain.interior_cell_count();
        let (macro_buffer, macro_readback) = if cfg.macroscopic_buffer {
            let bytes = interior_cells * 16;
            (
                Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("lbm macroscopic buffer"),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })),
                Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("lbm macroscopic readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })),
            )
        } else {
            (None, None)
        };

        // ---- layouts ----
        let l0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lbm group0"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
            ],
        });
        let l1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lbm ddf"),
            entries: &ddf.layout_entries(0),
        });
        let mut l2_entries = vec![
            storage_texture_entry(0, wgpu::TextureFormat::Rgba16Float),
            storage_texture_entry(1, wgpu::TextureFormat::R32Float),
        ];
        if cfg.macroscopic_buffer {
            l2_entries.push(storage_entry(2, false));
        }
        let l2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lbm macroscopic"),
            entries: &l2_entries,
        });

        // ---- pipelines ----
        //
        // The layout comes from the allocation, never from the config: FP16C has
        // two of them over identical bytes and only `DdfBuffers` knows which one
        // the device could give us.
        let layout = shaders::DdfLayout::of(cfg.precision, ddf.fp16c_per_cell);
        let loader = build_loader(cfg, layout);
        let defines = build_defines(cfg, layout);
        let step_module = loader
            .create_module(&device, "lbm/stream_collide.wgsl", &defines)
            .context("compiling stream_collide.wgsl")?;
        let macro_module = loader
            .create_module(&device, "lbm/macroscopic.wgsl", &defines)
            .context("compiling macroscopic.wgsl")?;

        let step_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lbm step layout"),
            bind_group_layouts: &[Some(&l0), Some(&l1)],
            immediate_size: 0,
        });
        let macro_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lbm macroscopic layout"),
            bind_group_layouts: &[Some(&l0), Some(&l1), Some(&l2)],
            immediate_size: 0,
        });
        let compute = |label, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let init_pipeline = compute("lbm init", &step_layout, &step_module, "init");
        let step_pipeline = compute(
            "lbm stream_collide",
            &step_layout,
            &step_module,
            "stream_collide",
        );
        let macro_pipeline = compute(
            "lbm macroscopic",
            &macro_layout,
            &macro_module,
            "macroscopic",
        );

        // ---- bind groups ----
        let group0 = [0usize, 1].map(|p| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lbm group0"),
                layout: &l0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffers[p].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: flags_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: link_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: boundary_link_buffer.as_entire_binding(),
                    },
                ],
            })
        });
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbm ddf"),
            layout: &l1,
            entries: &ddf.bind_entries(0),
        });
        let vel_view = velocity_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let den_view = density_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut g2_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&vel_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&den_view),
            },
        ];
        if let Some(b) = &macro_buffer {
            g2_entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: b.as_entire_binding(),
            });
        }
        let group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbm macroscopic"),
            layout: &l2,
            entries: &g2_entries,
        });

        let profiler = Profiler::new(
            &device,
            &queue,
            8,
            gpu.caps.timestamps,
            gpu.caps.peak_bandwidth,
        );

        let mut me = Self {
            device,
            queue,
            grid,
            domain,
            links,
            cfg,
            ddf,
            uniform_buffers,
            flags_buffer,
            _link_buffer: link_buffer,
            boundary_link_buffer,
            group0,
            group1,
            group2,
            init_pipeline,
            step_pipeline,
            macro_pipeline,
            velocity_texture,
            density_texture,
            macro_buffer,
            macro_readback,
            profiler,
            profiling: false,
            steps: 0,
        };
        me.write_uniforms();
        me.reset();
        Ok(me)
    }

    pub fn config(&self) -> &SolverConfig {
        &self.cfg
    }

    pub fn steps_taken(&self) -> u64 {
        self.steps
    }

    /// Physical time simulated so far, from `SolverConfig::dt_s`.
    pub fn sim_time_seconds(&self) -> f64 {
        self.steps as f64 * self.cfg.dt_s
    }

    /// Apply new parameters without rebuilding anything.
    ///
    /// Returns an error if the change would need a different shader — the inlet
    /// velocity, the relaxation time, the Smagorinsky constant and the whole
    /// outlet configuration are all uniform data and apply immediately, which is
    /// the point: the inlet slider must not stall on a pipeline compile.
    pub fn update(&mut self, cfg: SolverConfig) -> Result<()> {
        if self.cfg.needs_rebuild(&cfg) {
            anyhow::bail!(
                "changing the velocity set, storage precision, collision model, periodicity, \
                 workgroup size or macroscopic-buffer flag requires rebuilding the solver"
            );
        }
        self.cfg = cfg;
        self.write_uniforms();
        Ok(())
    }

    /// Hot-update just the inlet velocity, in lattice units.
    pub fn set_inlet_velocity(&mut self, v: Vec3) {
        self.cfg.inlet_velocity = v;
        self.write_uniforms();
    }

    /// The raw distribution functions.
    ///
    /// Exposed so wall stress can be computed by **momentum exchange**, which
    /// needs the populations themselves:
    ///
    /// ```text
    /// F = (dx^3/dt) * sum over boundary links of c_i * [f_i(x_f, t) + f_ibar(x_f, t+dt)]
    /// ```
    ///
    /// That is the accurate route, and it integrates exactly to the total force
    /// on the duct. Reconstructing the strain rate from the sampled velocity
    /// field instead is second-best: it differences a field that is itself
    /// interpolated, right where the staircase boundary makes interpolation
    /// least trustworthy.
    ///
    /// Anything reading these must index through [`ad_gpu::EsotericPull`] --
    /// half of a cell's populations physically live in a neighbour's slot, so a
    /// direct index will appear to work and be quietly wrong.
    pub fn ddf_buffers(&self) -> &ad_gpu::DdfBuffers {
        &self.ddf
    }

    /// Per-cell flag bytes, packed four cells per `u32`, little-endian, X-fastest.
    ///
    /// **This covers the padded domain, not the interior grid.** The solver adds
    /// one cell of solid halo per non-periodic axis, because Esoteric Pull parks
    /// half of a boundary cell's populations in the slot belonging to the cell
    /// outside the domain. So indices here are strides of `padded_dims()`, and
    /// feeding this straight to something that expects interior dimensions
    /// misaligns every cell by a plane -- which looks like a plausible field
    /// rather than an error. Use [`Solver::padded_dims`] to index it, or build
    /// your own interior-sized flag buffer if that is what you need.
    pub fn flags_buffer(&self) -> &wgpu::Buffer {
        &self.flags_buffer
    }

    /// The velocity texture itself, for `copy_texture_to_buffer`.
    ///
    /// [`Solver::velocity_view`] is enough to *sample* the field, but reading a
    /// region back on the CPU needs the texture. Without this the only route is
    /// `macroscopic_buffer`, which allocates a full interleaved copy of the
    /// domain -- 691 MB at the interactive tier -- when a sub-region copy of a
    /// few megabytes would do.
    ///
    /// `xyz` is velocity in **lattice units** (multiply by `LatticeUnits::c_u()`
    /// for m/s) and `w` is `rho - 1`.
    pub fn velocity_texture(&self) -> &wgpu::Texture {
        &self.velocity_texture
    }

    /// The density texture, `R32Float`, holding raw lattice density.
    pub fn density_texture(&self) -> &wgpu::Texture {
        &self.density_texture
    }

    /// The sparse boundary-link list, byte-identical to `[ad_gpu::BoundaryLink]`.
    pub fn boundary_link_buffer(&self) -> &wgpu::Buffer {
        &self.boundary_link_buffer
    }

    /// Reinitialise every fluid cell to equilibrium and zero the step counter.
    pub fn reset(&mut self) {
        self.steps = 0;
        self.write_uniforms();

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbm reset"),
            });
        // Zeroing is not strictly required — every slot the solver reads is
        // written by `init` — but it costs one clear per reset and removes any
        // chance of a NaN in an unread slot propagating through a future change.
        for b in &self.ddf.buffers {
            enc.clear_buffer(b, 0, None);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lbm init"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.init_pipeline);
            pass.set_bind_group(0, &self.group0[0], &[]);
            pass.set_bind_group(1, &self.group1, &[]);
            let d = self.padded_dispatch();
            pass.dispatch_workgroups(d.x, d.y, d.z);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Advance `n` steps.
    ///
    /// All `n` dispatches go into one command buffer. The only thing that
    /// changes between them is which of the two prebuilt group-0 bind groups is
    /// bound, so there is no per-step buffer write and no CPU/GPU round trip.
    pub fn step(&mut self, n: u32) {
        self.advance(n, false);
    }

    /// [`Self::step`], then [`Self::compute_macroscopic`] recorded into the
    /// last step batch's command buffer.
    ///
    /// The frame loop's path. It saves a submit, and it is the only way the
    /// macroscopic pass gets timed: recorded on its own, its profiler scope is
    /// claimed after the step batch has already resolved, so its timestamps
    /// are written and never read.
    pub fn step_and_compute_macroscopic(&mut self, n: u32) {
        self.advance(n, true);
    }

    fn advance(&mut self, n: u32, macroscopic: bool) {
        if n == 0 {
            if macroscopic {
                self.compute_macroscopic();
            }
            return;
        }
        // Split into several submits when `n` is large.
        //
        // # Why, and what happens without it
        //
        // Every dispatch used to go into one command buffer and one submit. That
        // is the right shape for an interactive frame, where `n` is a handful of
        // steps, and it is how the CPU stays out of the way. But an unattended
        // convergence run asks for hundreds of steps at once, and on a large grid
        // a single submit then occupies the GPU for seconds.
        //
        // Windows kills a GPU job that does not yield within roughly two seconds
        // (Timeout Detection and Recovery) and resets the driver. wgpu surfaces
        // that as `Device::poll: Validation Error / Parent device is lost`, from
        // which nothing can recover: the queue, every buffer and the whole
        // pipeline cache are gone. It cost a 58-minute resolution run at
        // `dx = 0.3 mm` with 500 steps per submit, and the message says nothing
        // about the real cause -- there is no hint that the *batch size* was the
        // problem, so the obvious reading is that the physics diverged.
        //
        // So bound the work per submit rather than trusting the caller. The
        // budget is in cell-steps, because the cost of a submit is the grid size
        // times the number of dispatches in it. Submits are cheap next to the
        // work they carry (tens of microseconds against tens of milliseconds), so
        // being conservative here costs essentially nothing: at `dx = 0.3 mm`
        // this is ~10 steps per submit, adding under two seconds across a
        // 350,000-step run.
        let per_submit = self.max_steps_per_submit();
        let mut left = n;
        while left > 0 {
            let chunk = left.min(per_submit);
            left -= chunk;
            self.step_batch(chunk, macroscopic && left == 0);
        }
    }

    /// Largest number of steps to put in one submit, from the grid size.
    ///
    /// See [`Solver::step`] for why this exists. The budget is deliberately well
    /// under what the driver actually tolerates, because the failure mode is a
    /// lost device rather than a slow frame.
    fn max_steps_per_submit(&self) -> u32 {
        let p = self.domain.padded;
        max_steps_per_submit_for(p.x as u64 * p.y as u64 * p.z as u64)
    }

    /// One submit's worth of steps. Callers should use [`Solver::step`], which
    /// enforces the per-submit budget.
    fn step_batch(&mut self, n: u32, macroscopic: bool) {
        if n == 0 {
            return;
        }
        self.profiler.begin_frame();
        let d = self.padded_dispatch();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbm step"),
            });
        {
            let scope = if self.profiling {
                self.profiler.scope_per(STEP_SCOPE, n)
            } else {
                None
            };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lbm stream_collide"),
                timestamp_writes: scope,
            });
            pass.set_pipeline(&self.step_pipeline);
            pass.set_bind_group(1, &self.group1, &[]);
            for k in 0..n {
                let parity = ((self.steps + k as u64) % 2) as usize;
                pass.set_bind_group(0, &self.group0[parity], &[]);
                pass.dispatch_workgroups(d.x, d.y, d.z);
            }
        }
        self.steps += n as u64;
        // After the count moves: `record_macroscopic` takes its parity from it.
        if macroscopic {
            self.record_macroscopic(&mut enc);
        }
        self.profiler.resolve(&mut enc);
        self.queue.submit(Some(enc.finish()));

        if self.profiling {
            // Non-blocking. `Profiler::collect` tracks its own map state and
            // never leaves the readback buffer pending, so the timings simply
            // arrive a call or two late — which is exactly right for a number
            // that ends up in a status bar. Nothing here stalls the frame loop.
            self.profiler.collect(&self.device);
        }
    }

    /// Turn GPU timestamp profiling on or off.
    ///
    /// Off by default, because timestamp queries are not free and most runs do
    /// not want them. Turn it on to read [`Self::mlups`],
    /// [`Self::roofline_fraction`] or [`Self::profiler_summary`]; the first
    /// numbers appear a couple of `step` calls later, once the readback lands.
    pub fn set_profiling(&mut self, on: bool) {
        self.profiling = on && self.profiler.is_enabled();
    }

    pub fn is_profiling(&self) -> bool {
        self.profiling
    }

    /// Pick up any GPU timings that have landed since the last call.
    ///
    /// [`Self::step`] issues the readback but never waits for it, so the numbers
    /// arrive a call or two later and something has to poll for them.
    /// Applications get that for free from presentation; a headless caller — a
    /// benchmark, a batch run — has to ask, and this is where. Cheap and safe to
    /// call every frame.
    pub fn collect_profiling(&mut self) {
        if self.profiling {
            let _ = self.device.poll(wgpu::PollType::Poll);
            self.profiler.collect(&self.device);
        }
    }

    /// Block until every submitted step has finished on the device.
    ///
    /// Needed only for wall-clock measurement and for tests; the frame loop
    /// should never call it, because the whole point of batching `step(n)` is
    /// that the CPU runs ahead.
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Recompute the macroscopic textures from the current populations.
    pub fn compute_macroscopic(&mut self) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbm macroscopic"),
            });
        self.record_macroscopic(&mut enc);
        self.queue.submit(Some(enc.finish()));
    }

    fn record_macroscopic(&mut self, enc: &mut wgpu::CommandEncoder) {
        let parity = (self.steps % 2) as usize;
        let d = self.interior_dispatch();
        let scope = if self.profiling {
            self.profiler.scope(MACRO_SCOPE)
        } else {
            None
        };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lbm macroscopic"),
            timestamp_writes: scope,
        });
        pass.set_pipeline(&self.macro_pipeline);
        pass.set_bind_group(0, &self.group0[parity], &[]);
        pass.set_bind_group(1, &self.group1, &[]);
        pass.set_bind_group(2, &self.group2, &[]);
        pass.dispatch_workgroups(d.x, d.y, d.z);
    }

    /// Velocity field as a sampled view, for the render and metrics crates.
    ///
    /// Storage textures are write-only in the portable WGSL spec, so consumers
    /// must bind *this* view (plus a sampler), never the storage view the
    /// macroscopic pass writes through.
    pub fn velocity_view(&self) -> wgpu::TextureView {
        self.velocity_texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn density_view(&self) -> wgpu::TextureView {
        self.density_texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Read the macroscopic field back to the CPU as `(u.x, u.y, u.z, rho)` per
    /// interior cell, X-fastest.
    ///
    /// Blocking, and only available when [`SolverConfig::macroscopic_buffer`] is
    /// set. This is a validation path, not a runtime one.
    pub fn read_macroscopic(&mut self) -> Result<Vec<[f32; 4]>> {
        if self.macro_buffer.is_none() {
            anyhow::bail!("read_macroscopic needs SolverConfig::macroscopic_buffer = true");
        }
        let bytes = self.domain.interior_cell_count() * 16;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbm macroscopic readback"),
            });
        self.record_macroscopic(&mut enc);
        let (Some(src), Some(dst)) = (&self.macro_buffer, &self.macro_readback) else {
            unreachable!("checked above")
        };
        enc.copy_buffer_to_buffer(src, 0, dst, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = dst.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow::anyhow!("device poll failed: {e:?}"))?;
        rx.recv()??;
        let out = {
            let view = slice.get_mapped_range()?;
            bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec()
        };
        dst.unmap();
        Ok(out)
    }

    /// Bytes of memory traffic one step moves, for the roofline percentage.
    pub fn bytes_per_step(&self) -> u64 {
        let (_, traffic) = ad_gpu::bytes_per_cell(self.cfg.set, self.cfg.precision);
        // The link mask is 4 B/cell on top of ad-gpu's model, which budgets only
        // the flag byte. Counting it keeps the roofline percentage honest rather
        // than flattering.
        (traffic + 4) * self.domain.padded_cell_count()
    }

    /// One-line profiler summary, including percent of roofline.
    ///
    /// Only meaningful when [`Self::set_profiling`] is on. The step pass is
    /// timed per step ([`ad_gpu::Profiler::scope_per`]), so the moving average
    /// holds across batches of different lengths.
    pub fn profiler_summary(&self) -> String {
        self.profiler.summary(STEP_SCOPE, self.bytes_per_step())
    }

    /// Achieved million lattice updates per second.
    pub fn mlups(&self) -> Option<f64> {
        let ms = self.ms_per_step()?;
        if ms <= 0.0 {
            return None;
        }
        Some(self.domain.padded_cell_count() as f64 / (ms * 1e-3) / 1e6)
    }

    /// Achieved bandwidth as a fraction of the device's peak. For a
    /// bandwidth-bound kernel this is the only performance number that means
    /// anything on its own.
    pub fn roofline_fraction(&self) -> Option<f64> {
        self.profiler
            .roofline_fraction(STEP_SCOPE, self.bytes_per_step())
    }

    /// Milliseconds per step, averaged over recent timed batches.
    pub fn ms_per_step(&self) -> Option<f64> {
        Some(self.profiler.timing(STEP_SCOPE)?.mean_ms)
    }

    /// GPU time of the macroscopic pass, ms. Only passes recorded by
    /// [`Self::step_and_compute_macroscopic`] are timed.
    pub fn macroscopic_ms(&self) -> Option<f64> {
        Some(self.profiler.timing(MACRO_SCOPE)?.mean_ms)
    }

    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    fn padded_dispatch(&self) -> UVec3 {
        let wg = self.cfg.workgroup_size.max(1);
        UVec3::new(
            self.domain.padded.x.div_ceil(wg),
            self.domain.padded.y,
            self.domain.padded.z,
        )
    }

    fn interior_dispatch(&self) -> UVec3 {
        let wg = self.cfg.workgroup_size.max(1);
        UVec3::new(
            self.domain.interior.x.div_ceil(wg),
            self.domain.interior.y,
            self.domain.interior.z,
        )
    }

    fn write_uniforms(&mut self) {
        for parity in 0..2u32 {
            let u = self.uniforms(parity);
            self.queue.write_buffer(
                &self.uniform_buffers[parity as usize],
                0,
                bytemuck::bytes_of(&u),
            );
        }
    }

    fn uniforms(&self, parity: u32) -> LbmUniforms {
        let periodic = (self.cfg.periodic[0] as u32)
            | ((self.cfg.periodic[1] as u32) << 1)
            | ((self.cfg.periodic[2] as u32) << 2);
        let mut inlet_vel = [[0.0f32; 4]; 4];
        let mut inlet_nrm = [[0.0f32; 4]; 4];
        for slot in 0..4u8 {
            let s = self.cfg.inlet_slot(slot);
            inlet_vel[slot as usize] = s
                .velocity
                .extend(if s.local_density { 1.0 } else { 0.0 })
                .to_array();
            inlet_nrm[slot as usize] = s.normal.extend(0.0).to_array();
        }
        LbmUniforms {
            dims: self.domain.padded.to_array(),
            step_parity: parity,
            interior: self.domain.interior.to_array(),
            cell_count: self.domain.padded_cell_count() as u32,
            offset: self.domain.offset.to_array(),
            flag_words: self.domain.padded_cell_count().div_ceil(4) as u32,
            inlet_velocity: self.cfg.inlet_velocity.to_array(),
            tau0: self.cfg.tau0,
            initial_velocity: self.cfg.initial_velocity.to_array(),
            trt_lambda: self.cfg.trt_lambda,
            body_force: self.cfg.body_force.to_array(),
            smagorinsky_c: self.cfg.smagorinsky_c,
            tau_max: self.cfg.tau_max,
            outflow_velocity: self.cfg.outflow_velocity,
            rho_ref: self.cfg.rho_ref,
            sponge_strength: self.cfg.sponge_strength,
            sponge_cells: self.cfg.sponge_cells,
            periodic,
            total_steps: self.steps as u32,
            _pad: 0,
            outlet_normal: self.cfg.outlet_normal.to_array(),
            _pad1: 0.0,
            inlet_normal: self.cfg.inlet_normal.to_array(),
            outlet_anti_bounce_back: self.cfg.outlet_anti_bounce_back as u32,
            inlet_vel,
            inlet_nrm,
        }
    }
}

/// Pack one flag byte per cell into `u32` words, four cells per word.
fn pack_flags(bytes: &[u8]) -> Vec<u32> {
    let mut out = vec![0u32; bytes.len().div_ceil(4)];
    for (i, b) in bytes.iter().enumerate() {
        out[i >> 2] |= (*b as u32) << ((i & 3) * 8);
    }
    out
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
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

fn storage_texture_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D3,
        },
        count: None,
    }
}

/// A loader with every LBM shader registered as a virtual file.
///
/// The sources are embedded with `include_str!` rather than read from disk, so
/// the binary carries them and nothing depends on the working directory. They
/// still live in `shaders/lbm/` and are still preprocessed by
/// [`ad_gpu::ShaderLoader`] — this only changes where the bytes come from.
pub fn build_loader(cfg: SolverConfig, layout: shaders::DdfLayout) -> ShaderLoader {
    let mut loader = ShaderLoader::new(".");
    // Generated: the lattice tables from ad-gpu, the FP16C codec, the flag
    // constants, the DDF bindings and the Esoteric Pull transport bodies. Split
    // in two so `common.wgsl` can sit between them: it needs Q_CONST and the
    // flags, and the transport bodies need its `neighbour_index`.
    //
    // The `enable` directives go first and nowhere else: WGSL requires every one
    // of them to precede every declaration in the module, and this head is the
    // first thing any entry point pulls in.
    let mut head = String::from(shaders::enable_directives(layout));
    head.push_str(&ad_gpu::lattice::wgsl_prelude(cfg.set));
    head.push('\n');
    head.push_str(&precision::wgsl_codec());
    head.push('\n');
    let generated = shaders::generated_prelude(cfg.set, layout);
    let split = generated
        .find("@group(1)")
        .expect("generated prelude must contain the DDF bindings");
    head.push_str(&generated[..split]);
    loader.add_virtual("lbm/generated_head.wgsl", head);
    loader.add_virtual("lbm/generated_ddf.wgsl", generated[split..].to_string());

    loader.add_virtual(
        "lbm/common.wgsl",
        include_str!("../../../shaders/lbm/common.wgsl"),
    );
    loader.add_virtual(
        "lbm/collision.wgsl",
        include_str!("../../../shaders/lbm/collision.wgsl"),
    );
    loader.add_virtual(
        "lbm/boundary.wgsl",
        include_str!("../../../shaders/lbm/boundary.wgsl"),
    );
    loader.add_virtual(
        "lbm/stream_collide.wgsl",
        include_str!("../../../shaders/lbm/stream_collide.wgsl"),
    );
    loader.add_virtual(
        "lbm/macroscopic.wgsl",
        include_str!("../../../shaders/lbm/macroscopic.wgsl"),
    );
    loader
}

pub fn build_defines(cfg: SolverConfig, layout: shaders::DdfLayout) -> ShaderDefines {
    let mut d = ShaderDefines::new()
        .flag(cfg.set.shader_define())
        .flag(cfg.precision.shader_define())
        // The layout is not implied by the precision - FP16C has two - and it
        // changes the generated source, so it has to be part of the cache key.
        .flag(layout.shader_define())
        .flag(cfg.collision.shader_define())
        .value("WG_X", cfg.workgroup_size);
    if cfg.macroscopic_buffer {
        d = d.flag("MACRO_BUFFER");
    }
    d
}

/// Convenience: a fully solid-free mask for `grid`, for analytic cases.
pub fn open_mask(dims: UVec3) -> Vec<u8> {
    vec![flags::FLUID; (dims.x * dims.y * dims.z) as usize]
}

/// Bytes of VRAM the DDFs for `grid` would need, including the halo.
pub fn ddf_bytes(
    grid: Grid,
    periodic: [bool; 3],
    set: VelocitySet,
    precision: DdfPrecision,
) -> u64 {
    let pad = UVec3::new(
        if periodic[0] { 0 } else { 2 },
        if periodic[1] { 0 } else { 2 },
        if periodic[2] { 0 } else { 2 },
    );
    let d = grid.dims + pad;
    let cells = d.x as u64 * d.y as u64 * d.z as u64;
    // Ask ad-gpu rather than restating its arithmetic; a second copy here is how
    // a VRAM prediction quietly stops matching what actually gets allocated.
    ad_gpu::direction_bytes(cells, precision) * set.q() as u64
}

/// Largest number of steps to put in one submit, for a grid of `cells` cells.
///
/// Cell-steps per submit, calibrated against a run that survived (42.3 M cells
/// x 500 steps) and one that did not (100 M x 500), with a wide margin: being
/// wrong one way costs a few microseconds of submit overhead, and the other way
/// costs a lost device and the whole run. See [`Solver::step`].
pub fn max_steps_per_submit_for(cells: u64) -> u32 {
    // Two measured points bracket this:
    //
    //   42.3 M cells x 500 steps = 2.1e10 cell-steps  -- ran fine
    //   101.3 M cells x 500 steps = 5.05e10           -- lost the device
    //
    // 1e10 sits 2x below the known-good point and 5x below the known-bad one,
    // and lands at a roughly uniform ~250 ms per submit across every tier this
    // app runs. That is well inside the ~2 s watchdog while keeping submits
    // large enough that per-submit overhead stays negligible.
    //
    // The first version of this used 1e9, which is ~24 ms per submit -- about
    // 20x more conservative than needed. It traded the crash for a slowdown:
    // at dx = 0.3 mm it meant 38,888 submits, and the GPU spent a large share
    // of the run draining the pipeline between them rather than solving.
    const BUDGET_CELL_STEPS: u64 = 10_000_000_000;
    (BUDGET_CELL_STEPS / cells.max(1)).clamp(1, 4096) as u32
}

#[cfg(test)]
mod tests {
    /// A convergence run asks for hundreds of steps at once. On a large grid
    /// that must not become one multi-second submit, because Windows resets the
    /// driver after about two seconds and wgpu reports it as a lost device --
    /// unrecoverable, and with nothing in the message pointing at batch size.
    /// This cost a 58-minute run at dx = 0.3 mm before the split existed.
    #[test]
    fn a_large_grid_never_batches_a_whole_convergence_run_into_one_submit() {
        // The grids this app actually runs, from the resolution study.
        let survived = 42_300_000u64; // dx = 0.4 mm, ran fine at 500/submit
        let died = 101_300_000u64; // dx = 0.3 mm, lost the device at 500/submit
        let interactive = 6_500_000u64; // dx = 0.75 mm

        assert!(
            max_steps_per_submit_for(died) < 500,
            "the grid that killed the driver must now be split"
        );
        // ...and with real margin, not just barely.
        assert!(
            max_steps_per_submit_for(died) * 4 < 500,
            "the split should be well clear of the batch size that failed"
        );
        // But not so fine that submit overhead dominates the work it carries.
        assert!(
            max_steps_per_submit_for(died) >= 20,
            "over-splitting trades a crash for a slowdown"
        );
        assert!(
            max_steps_per_submit_for(died) >= 1,
            "it must still make progress"
        );
        // Bigger grid, smaller batch: the budget is work, not step count.
        assert!(max_steps_per_submit_for(died) < max_steps_per_submit_for(survived));
        assert!(max_steps_per_submit_for(survived) < max_steps_per_submit_for(interactive));

        // Submit overhead has to stay negligible against the work it carries.
        // At the finest grid, a 350k-step run should not need a runaway number
        // of submits.
        let submits = 350_000 / max_steps_per_submit_for(died).max(1);
        assert!(submits < 100_000, "{submits} submits is too many");

        // A degenerate grid must not divide by zero or ask for zero steps.
        assert_eq!(
            max_steps_per_submit_for(0).max(1),
            max_steps_per_submit_for(0)
        );
        assert!(max_steps_per_submit_for(u64::MAX) >= 1);
    }

    use super::*;

    #[test]
    fn flags_pack_four_cells_per_word() {
        let bytes = [0x01u8, 0x02, 0x04, 0x08, 0x10];
        let words = pack_flags(&bytes);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], 0x08040201);
        assert_eq!(words[1], 0x00000010);
        // ...and the shader's unpack is the inverse.
        for (i, b) in bytes.iter().enumerate() {
            let got = (words[i >> 2] >> ((i & 3) * 8)) & 0xff;
            assert_eq!(got as u8, *b, "cell {i}");
        }
    }

    #[test]
    fn uniform_layout_matches_the_wgsl_struct() {
        // Every vec3 in a WGSL uniform is 16-byte aligned, so each [T; 3] must be
        // followed by a scalar. If someone reorders these fields, this catches it
        // before the GPU silently reads the wrong offsets.
        use std::mem::offset_of;
        assert_eq!(offset_of!(LbmUniforms, dims), 0);
        assert_eq!(offset_of!(LbmUniforms, step_parity), 12);
        assert_eq!(offset_of!(LbmUniforms, interior), 16);
        assert_eq!(offset_of!(LbmUniforms, offset), 32);
        assert_eq!(offset_of!(LbmUniforms, inlet_velocity), 48);
        assert_eq!(offset_of!(LbmUniforms, initial_velocity), 64);
        assert_eq!(offset_of!(LbmUniforms, body_force), 80);
        assert_eq!(offset_of!(LbmUniforms, tau_max), 96);
        assert_eq!(offset_of!(LbmUniforms, sponge_cells), 112);
        assert_eq!(offset_of!(LbmUniforms, outlet_normal), 128);
        assert_eq!(offset_of!(LbmUniforms, inlet_normal), 144);
        assert_eq!(offset_of!(LbmUniforms, outlet_anti_bounce_back), 156);
        // Two `array<vec4<f32>, 4>` in WGSL: 16-byte stride, 16-aligned start.
        assert_eq!(offset_of!(LbmUniforms, inlet_vel), 160);
        assert_eq!(offset_of!(LbmUniforms, inlet_nrm), 224);
        assert_eq!(std::mem::size_of::<LbmUniforms>(), 288);
    }

    #[test]
    fn shader_sources_preprocess_for_every_configuration() {
        // Catches an unbalanced #if or a missing include without needing a GPU.
        for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
            for precision in [DdfPrecision::Fp32, DdfPrecision::Fp16c] {
                for collision in [
                    crate::CollisionModel::Trt,
                    crate::CollisionModel::Bgk,
                    crate::CollisionModel::RegularizedBgk,
                ] {
                    for macro_buffer in [false, true] {
                        let cfg = SolverConfig {
                            set,
                            precision,
                            collision,
                            macroscopic_buffer: macro_buffer,
                            ..Default::default()
                        };
                        // Both FP16C layouts, so the adapter-without-16-bit-
                        // storage fallback is preprocessed here too rather than
                        // only on hardware that happens to lack the feature.
                        for per_cell in [false, true] {
                            let layout = shaders::DdfLayout::of(precision, per_cell);
                            let loader = build_loader(cfg, layout);
                            let defines = build_defines(cfg, layout);
                            for entry in ["lbm/stream_collide.wgsl", "lbm/macroscopic.wgsl"] {
                                let src = loader.load(entry, &defines).unwrap_or_else(|e| {
                                    panic!("{entry} {set:?} {precision:?} {layout:?}: {e}")
                                });
                                assert!(
                                    src.contains("@compute"),
                                    "{entry}: no entry point emitted"
                                );
                                assert!(
                                    !src.contains("#WG_X"),
                                    "{entry}: workgroup size was not substituted"
                                );
                                assert!(
                                    src.contains(&format!("const Q: u32 = {}u;", set.q())),
                                    "{entry}: lattice prelude missing"
                                );
                                // A directive that follows any declaration is a
                                // WGSL parse error, and it is the preprocessor -
                                // which pastes includes together - that decides
                                // where it lands. Comments and blank lines may
                                // precede it; nothing else may.
                                //
                                // Matched line by line rather than by substring:
                                // the shader sources talk *about* the directive
                                // in their comments, and a substring search
                                // cannot tell that from emitting one.
                                let mut directives = Vec::new();
                                let mut seen_declaration = false;
                                for line in src.lines() {
                                    let l = line.trim();
                                    if l.is_empty() || l.starts_with("//") {
                                        continue;
                                    }
                                    if let Some(rest) = l.strip_prefix("enable ") {
                                        assert!(
                                            !seen_declaration,
                                            "{entry} {layout:?}: `{l}` comes after a declaration, \
                                             which WGSL will reject"
                                        );
                                        directives.push(rest.trim_end_matches(';').to_string());
                                    } else {
                                        seen_declaration = true;
                                    }
                                }
                                let wants_int16 = layout == shaders::DdfLayout::Fp16cPerCell;
                                assert_eq!(
                                    directives.contains(&"wgpu_int16".to_string()),
                                    wants_int16,
                                    "{entry} {layout:?}: emitted directives {directives:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_halo_costs_what_the_contract_budget_can_absorb() {
        // Interactive tier, 347x240x240 at D3Q19/FP16C: 1.1 GB per the contract.
        // The halo must not push that materially.
        let grid = Grid {
            dims: UVec3::new(347, 240, 240),
            dx_mm: 0.75,
            origin_mm: Vec3::ZERO,
        };
        let padded = ddf_bytes(grid, [false; 3], VelocitySet::D3Q19, DdfPrecision::Fp16c);
        let bare = ddf_bytes(
            Grid {
                dims: grid.dims - UVec3::splat(0),
                ..grid
            },
            [true; 3],
            VelocitySet::D3Q19,
            DdfPrecision::Fp16c,
        );
        let overhead = padded as f64 / bare as f64 - 1.0;
        assert!(overhead < 0.03, "halo overhead is {:.2}%", overhead * 100.0);
        assert!(
            padded < 1_250_000_000,
            "{padded} bytes at the interactive tier"
        );
    }
}

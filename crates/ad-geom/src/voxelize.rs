//! GPU signed-distance voxeliser.
//!
//! Turns a triangle soup into the three things the rest of the app needs — the
//! per-cell flag byte, a narrow-band signed distance field, and the list of
//! boundary links with their `q` values — in one pass over the geometry. They
//! all fall out of the same distance field, so computing them together is
//! strictly less work than computing any two of them apart.
//!
//! The shader stages and the reasoning behind each are documented in
//! `shaders/geom/voxelize.wgsl`. This module is the driver: it owns the
//! buffers, keeps them across calls so a re-voxelisation allocates nothing, and
//! implements the incremental path that runs while the user drags a part.
//!
//! # Cost model
//!
//! The narrow-band stages are parallel over *triangles* and each triangle only
//! touches the cells within `band + dx` of its own AABB, so their cost scales
//! with surface area rather than with grid volume. The remaining stages are
//! grid-wide but read and write at most a handful of words per cell, so they run
//! at streaming bandwidth. The target is 50-300 ms at full resolution, which is
//! slow enough to notice and fast enough to drag against.

use crate::bins::TriangleBins;
use crate::scene::{FlatGeometry, Scene};
use ad_gpu::types::flags;
use ad_gpu::{Bbox, BoundaryLink, Grid, GpuContext, ShaderDefines, ShaderLoader, VelocitySet};
use anyhow::{bail, Context as _, Result};
use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec3};
use std::sync::Arc;

/// Tunables. The defaults are what the contract's resolution tiers were sized
/// against.
#[derive(Debug, Clone, Copy)]
pub struct VoxelizeConfig {
    /// Narrow-band half-width in cells. Must be at least 2: the band has to be
    /// a closed shell thick enough that no 6-connected path can cross the
    /// surface without passing through an interior-signed band cell, or the
    /// flood fill leaks through the wall.
    pub band_voxels: f32,
    /// Spatial bin size in cells, for picking which triangles a partial
    /// re-voxelisation has to look at.
    pub bin_voxels: f32,
    /// Give up on the flood fill after this many sweep iterations. Convergence
    /// is normally 1-3; the cap only exists so a pathological input cannot hang
    /// the UI thread.
    pub max_fill_iterations: u32,
}

impl Default for VoxelizeConfig {
    fn default() -> Self {
        Self { band_voxels: 3.5, bin_voxels: 12.0, max_fill_iterations: 24 }
    }
}

/// What a voxelisation produced and what it cost.
#[derive(Debug, Clone)]
pub struct VoxelStats {
    pub grid: Grid,
    pub band_mm: f32,
    pub triangles: usize,
    /// Triangles actually dispatched. Equal to `triangles` for a full rebuild,
    /// and much smaller for a drag.
    pub active_triangles: usize,
    pub solid_cells: u64,
    pub boundary_cells: u64,
    pub link_count: u32,
    pub fill_iterations: u32,
    pub fill_converged: bool,
    /// Wall-clock time including buffer uploads and the readbacks that the
    /// flood-fill convergence test needs.
    pub total_ms: f64,
    pub incremental: bool,
}

impl VoxelStats {
    pub fn solid_volume_mm3(&self) -> f64 {
        let dx = self.grid.dx_mm as f64;
        self.solid_cells as f64 * dx * dx * dx
    }

    pub fn report(&self) -> String {
        format!(
            "voxelised {} tris ({} dispatched) into {}x{}x{} at dx = {} mm in {:.1} ms{}: \
             {} solid cells = {:.0} mm^3, {} boundary cells, {} links, fill converged in {} \
             iteration(s){}",
            self.triangles,
            self.active_triangles,
            self.grid.dims.x,
            self.grid.dims.y,
            self.grid.dims.z,
            self.grid.dx_mm,
            self.total_ms,
            if self.incremental { " (incremental)" } else { "" },
            self.solid_cells,
            self.solid_volume_mm3(),
            self.boundary_cells,
            self.link_count,
            self.fill_iterations,
            if self.fill_converged { "" } else { " -- DID NOT CONVERGE" },
        )
    }
}

// ---------------------------------------------------------------------------
// GPU-side data layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GeomUniforms {
    dims: [u32; 3],
    tri_count: u32,
    origin_mm: [f32; 3],
    dx_mm: f32,
    band_mm: f32,
    axis: u32,
    link_capacity: u32,
    _pad_a: u32,
    region_lo: [u32; 3],
    _pad_b: u32,
    region_hi: [u32; 3],
    _pad_c: u32,
}

/// One triangle, padded to the `vec3<f32>` alignment WGSL applies inside a
/// storage struct: three 16-byte slots, 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuTri {
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub c: [f32; 4],
}

/// Pseudonormals for one triangle: face, three edges, three corners.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuTriPn {
    pub n: [[f32; 4]; 7],
}

fn to_gpu_tri(t: &[Vec3; 3]) -> GpuTri {
    let f = |v: Vec3| [v.x, v.y, v.z, 0.0];
    GpuTri { a: f(t[0]), b: f(t[1]), c: f(t[2]) }
}

fn to_gpu_pn(p: &[Vec3; 7]) -> GpuTriPn {
    let mut n = [[0.0f32; 4]; 7];
    for (dst, src) in n.iter_mut().zip(p) {
        *dst = [src.x, src.y, src.z, 0.0];
    }
    GpuTriPn { n }
}

// Counter slots, matching shaders/geom/common.wgsl.
const CT_LINKS: usize = 0;
const CT_SOLID: usize = 1;
const CT_BOUNDARY: usize = 2;
const CT_CHANGED: usize = 3;
const CT_OVERFLOW: usize = 4;
const COUNTER_WORDS: u64 = 5;

/// Uniform variant slots. One bind group with a dynamic offset selects between
/// them, so the sweep axis can change between passes inside a single command
/// buffer -- `Queue::write_buffer` only takes effect at a submit boundary, so
/// rewriting one uniform between dispatches would not work.
const UV_REGION: usize = 0;
const UV_SWEEP_X: usize = 1;
const UV_CLASSIFY: usize = 4;
const UNIFORM_SLOTS: usize = 5;

// ---------------------------------------------------------------------------

struct Pipelines {
    clear_region: wgpu::ComputePipeline,
    seed_distance: wgpu::ComputePipeline,
    assign_triangle: wgpu::ComputePipeline,
    sign_band: wgpu::ComputePipeline,
    reset_fill: wgpu::ComputePipeline,
    seed_exterior: wgpu::ComputePipeline,
    sweep: wgpu::ComputePipeline,
    resolve_unknown: wgpu::ComputePipeline,
    classify: wgpu::ComputePipeline,
}

pub struct Voxelizer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: VoxelizeConfig,
    layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    uniform_stride: u64,

    // Persistent state, reused across calls.
    geom: FlatGeometry,
    bins: Option<TriangleBins>,
    grid: Option<Grid>,
    active: Vec<u32>,
    /// Triangle ranges whose GPU copy is stale. `None` means "all of them".
    upload_ranges: Option<Vec<std::ops::Range<usize>>>,

    uniforms: wgpu::Buffer,
    tris: wgpu::Buffer,
    pn: wgpu::Buffer,
    active_buf: wgpu::Buffer,
    dist: wgpu::Buffer,
    tri_idx: wgpu::Buffer,
    phi: wgpu::Buffer,
    state: wgpu::Buffer,
    flags: wgpu::Buffer,
    links: wgpu::Buffer,
    counters: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,

    tri_capacity: usize,
    cell_capacity: u64,
    link_capacity: u32,

    stats: Option<VoxelStats>,
}

impl Voxelizer {
    pub fn new(gpu: &GpuContext) -> Result<Self> {
        Self::with_config(gpu, VoxelizeConfig::default())
    }

    pub fn with_config(gpu: &GpuContext, config: VoxelizeConfig) -> Result<Self> {
        anyhow::ensure!(
            config.band_voxels >= 2.0,
            "band_voxels must be at least 2 or the flood fill leaks through thin walls; got {}",
            config.band_voxels
        );
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        // Catch WGSL compilation and validation failures as an error rather than
        // letting wgpu log them and hand back a poisoned pipeline.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = build_module(&device)?;
        let layout = bind_group_layout(&device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("geom voxelize layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mk = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            clear_region: mk("clear_region"),
            seed_distance: mk("seed_distance"),
            assign_triangle: mk("assign_triangle"),
            sign_band: mk("sign_band"),
            reset_fill: mk("reset_fill"),
            seed_exterior: mk("seed_exterior"),
            sweep: mk("sweep"),
            resolve_unknown: mk("resolve_unknown"),
            classify: mk("classify"),
        };
        if let Some(err) = pollster::block_on(scope.pop()) {
            bail!("geometry voxeliser shaders failed to compile: {err}");
        }

        let uniform_stride =
            (gpu.limits.min_uniform_buffer_offset_alignment as u64).max(256);

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("geom uniforms"),
            size: uniform_stride * UNIFORM_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let placeholder = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage,
                mapped_at_creation: false,
            })
        };

        Ok(Self {
            config,
            layout,
            pipelines,
            uniform_stride,
            geom: FlatGeometry::default(),
            bins: None,
            grid: None,
            active: Vec::new(),
            upload_ranges: None,
            uniforms,
            tris: placeholder("geom tris", 48, storage),
            pn: placeholder("geom tri pn", 112, storage),
            active_buf: placeholder("geom active tris", 4, storage),
            dist: placeholder("geom dist", 4, storage),
            tri_idx: placeholder("geom tri index", 4, storage),
            phi: placeholder("geom phi", 4, storage | wgpu::BufferUsages::COPY_SRC),
            state: placeholder("geom fill state", 4, storage),
            flags: placeholder("geom flags", 4, storage | wgpu::BufferUsages::COPY_SRC),
            links: placeholder("geom links", 8, storage | wgpu::BufferUsages::COPY_SRC),
            counters: placeholder(
                "geom counters",
                COUNTER_WORDS * 4,
                storage | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: placeholder(
                "geom readback",
                COUNTER_WORDS * 4,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            ),
            bind_group: None,
            tri_capacity: 0,
            cell_capacity: 0,
            link_capacity: 0,
            stats: None,
            device,
            queue,
        })
    }

    pub fn config(&self) -> VoxelizeConfig {
        self.config
    }

    pub fn stats(&self) -> Option<&VoxelStats> {
        self.stats.as_ref()
    }

    pub fn grid(&self) -> Option<Grid> {
        self.grid
    }

    pub fn band_mm(&self) -> f32 {
        self.grid.map(|g| g.dx_mm * self.config.band_voxels).unwrap_or(0.0)
    }

    /// Narrow-band signed distance, one `f32` per cell, negative inside.
    pub fn phi_buffer(&self) -> &wgpu::Buffer {
        &self.phi
    }

    /// `ad_gpu::flags` bytes, four cells to a word, X fastest.
    ///
    /// Only `SOLID` and `SOLID_BOUNDARY` are set here: those are the only two
    /// the geometry knows about. `INLET`, `OUTLET`, `SPONGE` and `EQUILIBRIUM`
    /// describe the domain boundary and the chosen flow conditions, so the
    /// solver ORs them in afterwards. The buffer is not rewritten between
    /// voxelisations, so anything the solver adds survives only until the next
    /// [`Self::sync`], which clears and rebuilds it.
    pub fn flags_buffer(&self) -> &wgpu::Buffer {
        &self.flags
    }

    /// `ad_gpu::BoundaryLink` entries, valid up to `stats().link_count`.
    pub fn links_buffer(&self) -> &wgpu::Buffer {
        &self.links
    }

    /// Rebuild everything from a triangle soup. This is the entry point tests
    /// and one-shot callers want.
    ///
    /// The geometry must be **watertight**. Stage 5 fills the sign outside the
    /// narrow band by flooding from the domain boundary, and a hole in the
    /// surface lets the fill leak into the interior, turning the part hollow.
    /// Nothing here can detect that — a triangle soup has no topology — so check
    /// [`crate::MeshHealth::is_watertight_manifold`] on load, or run
    /// [`crate::ray_parity_voxelize`], which reports leaks directly.
    pub fn voxelize(&mut self, geom: FlatGeometry, grid: Grid) -> Result<VoxelStats> {
        self.geom = geom;
        self.bins = None;
        self.upload_ranges = None;
        self.grid = Some(grid);
        self.run(grid, None)
    }

    /// Bring the voxelisation up to date with a scene, doing as little work as
    /// the scene's dirty record allows.
    ///
    /// A pure transform change re-uploads only the triangles that moved and
    /// re-voxelises only the box they swept through. Anything structural — an
    /// instance added, removed or hidden — invalidates the buffer offsets, so it
    /// falls back to a full rebuild.
    pub fn sync(&mut self, scene: &mut Scene, grid: Grid) -> Result<VoxelStats> {
        let dirty = scene.take_dirty();
        let grid_changed = self.grid != Some(grid);
        let never_run = self.stats.is_none();
        let full = match &dirty {
            None => grid_changed || never_run,
            Some(d) => d.structural || grid_changed || never_run,
        };

        if !full && dirty.is_none() {
            return Ok(self.stats.clone().expect("stats exist once a run has happened"));
        }

        if full {
            self.geom = FlatGeometry::from_scene(scene);
            self.bins = None;
            self.upload_ranges = None;
            self.grid = Some(grid);
            return self.run(grid, None);
        }

        let dirty = dirty.expect("checked above");
        let mut ranges = Vec::new();
        for i in &dirty.instances {
            if let Some(r) = self.geom.refresh_instance(scene, *i) {
                ranges.push(r);
            }
        }
        self.upload_ranges = Some(ranges);
        // The bins describe where the triangles are *now*, so they must be
        // rebuilt before the dirty region is used to select them. The region
        // already covers where the moved part used to be, so cells it vacated
        // are recomputed from whatever is nearest them now.
        self.bins = None;
        self.grid = Some(grid);
        self.run(grid, Some(dirty.region))
    }

    fn run(&mut self, grid: Grid, dirty_region: Option<Bbox>) -> Result<VoxelStats> {
        let t0 = std::time::Instant::now();
        anyhow::ensure!(!self.geom.is_empty(), "cannot voxelise an empty scene");

        let band_mm = grid.dx_mm * self.config.band_voxels;
        let reach = band_mm + grid.dx_mm;

        if self.bins.is_none() {
            self.bins = Some(TriangleBins::build(
                &self.geom.bounds(),
                grid.dx_mm * self.config.bin_voxels,
            ));
        }

        // Which cells, and therefore which triangles, this call has to touch.
        let (region_lo, region_hi, incremental) = match dirty_region {
            Some(b) if !b.is_empty() => {
                let (lo, hi) = cell_range(grid, b.expanded(Vec3::splat(reach)));
                (lo, hi, true)
            }
            _ => (UVec3::ZERO, grid.dims - UVec3::ONE, false),
        };

        self.active.clear();
        if incremental {
            let world = Bbox {
                min: grid.cell_center_mm(region_lo) - Vec3::splat(reach),
                max: grid.cell_center_mm(region_hi) + Vec3::splat(reach),
            };
            self.bins.as_ref().unwrap().collect_in_aabb(world, &mut self.active);
        } else {
            self.active.extend(0..self.geom.len() as u32);
        }

        self.ensure_buffers(grid)?;
        self.upload_geometry();

        let uniforms = GeomUniforms {
            dims: grid.dims.to_array(),
            tri_count: self.active.len() as u32,
            origin_mm: grid.origin_mm.to_array(),
            dx_mm: grid.dx_mm,
            band_mm,
            axis: 0,
            link_capacity: 0,
            _pad_a: 0,
            region_lo: region_lo.to_array(),
            _pad_b: 0,
            region_hi: region_hi.to_array(),
            _pad_c: 0,
        };
        self.write_uniform(UV_REGION, uniforms);
        for axis in 0..3u32 {
            self.write_uniform(
                UV_SWEEP_X + axis as usize,
                GeomUniforms {
                    axis,
                    region_lo: [0, 0, 0],
                    region_hi: (grid.dims - UVec3::ONE).to_array(),
                    ..uniforms
                },
            );
        }
        self.write_uniform(
            UV_CLASSIFY,
            GeomUniforms {
                link_capacity: 0,
                region_lo: [0, 0, 0],
                region_hi: (grid.dims - UVec3::ONE).to_array(),
                ..uniforms
            },
        );

        let region_cells = {
            let d = region_hi - region_lo + UVec3::ONE;
            d.x as u64 * d.y as u64 * d.z as u64
        };
        let cells = grid.cell_count();

        // Stage 1-4 plus the fill seed, in one submit.
        let mut enc = self.encoder("geom narrow band");
        self.pass(&mut enc, "clear", &self.pipelines.clear_region, UV_REGION, groups(region_cells));
        self.pass(
            &mut enc,
            "seed distance",
            &self.pipelines.seed_distance,
            UV_REGION,
            self.active.len().clamp(1, 65535) as u32,
        );
        self.pass(
            &mut enc,
            "assign triangle",
            &self.pipelines.assign_triangle,
            UV_REGION,
            self.active.len().clamp(1, 65535) as u32,
        );
        self.pass(&mut enc, "sign", &self.pipelines.sign_band, UV_REGION, groups(region_cells));
        if incremental {
            self.pass(&mut enc, "reset fill", &self.pipelines.reset_fill, UV_SWEEP_X, groups(cells));
        }
        self.pass(&mut enc, "seed fill", &self.pipelines.seed_exterior, UV_SWEEP_X, groups(cells));
        self.queue.submit([enc.finish()]);

        // Flood fill. Sweeps are batched between convergence checks because each
        // check costs a round trip to the GPU, and a sweep costs far less.
        const BATCH: u32 = 4;
        let lines = |axis: usize| -> u64 {
            let d = grid.dims;
            match axis {
                0 => d.y as u64 * d.z as u64,
                1 => d.x as u64 * d.z as u64,
                _ => d.x as u64 * d.y as u64,
            }
        };
        let mut iterations = 0u32;
        let mut converged = false;
        while iterations < self.config.max_fill_iterations {
            let mut enc = self.encoder("geom flood fill");
            enc.clear_buffer(&self.counters, 0, None);
            let n = BATCH.min(self.config.max_fill_iterations - iterations);
            for _ in 0..n {
                for axis in 0..3usize {
                    self.pass(
                        &mut enc,
                        "sweep",
                        &self.pipelines.sweep,
                        UV_SWEEP_X + axis,
                        groups(lines(axis)),
                    );
                }
            }
            enc.copy_buffer_to_buffer(&self.counters, 0, &self.readback, 0, COUNTER_WORDS * 4);
            self.queue.submit([enc.finish()]);
            iterations += n;
            if self.read_counters()?[CT_CHANGED] == 0 {
                converged = true;
                break;
            }
        }

        // Resolve, then classify twice: once to learn the exact link count, once
        // to fill the buffer now that it can be sized.
        let mut enc = self.encoder("geom classify");
        enc.clear_buffer(&self.flags, 0, None);
        enc.clear_buffer(&self.counters, 0, None);
        self.pass(&mut enc, "resolve", &self.pipelines.resolve_unknown, UV_CLASSIFY, groups(cells));
        self.pass(&mut enc, "count links", &self.pipelines.classify, UV_CLASSIFY, groups(cells));
        enc.copy_buffer_to_buffer(&self.counters, 0, &self.readback, 0, COUNTER_WORDS * 4);
        self.queue.submit([enc.finish()]);

        let counted = self.read_counters()?;
        let link_count = counted[CT_LINKS];
        self.ensure_links(link_count.max(1))?;
        self.write_uniform(
            UV_CLASSIFY,
            GeomUniforms {
                link_capacity: self.link_capacity,
                region_lo: [0, 0, 0],
                region_hi: (grid.dims - UVec3::ONE).to_array(),
                ..uniforms
            },
        );

        let mut enc = self.encoder("geom links");
        enc.clear_buffer(&self.counters, 0, None);
        self.pass(&mut enc, "write links", &self.pipelines.classify, UV_CLASSIFY, groups(cells));
        enc.copy_buffer_to_buffer(&self.counters, 0, &self.readback, 0, COUNTER_WORDS * 4);
        self.queue.submit([enc.finish()]);
        let final_counts = self.read_counters()?;

        if final_counts[CT_OVERFLOW] != 0 {
            bail!(
                "boundary link buffer overflowed by {} entries; the count pass said {} but the \
                 write pass produced {}",
                final_counts[CT_OVERFLOW],
                link_count,
                final_counts[CT_LINKS],
            );
        }

        let stats = VoxelStats {
            grid,
            band_mm,
            triangles: self.geom.len(),
            active_triangles: self.active.len(),
            solid_cells: final_counts[CT_SOLID] as u64,
            boundary_cells: final_counts[CT_BOUNDARY] as u64,
            link_count: final_counts[CT_LINKS],
            fill_iterations: iterations,
            fill_converged: converged,
            total_ms: t0.elapsed().as_secs_f64() * 1e3,
            incremental,
        };
        if !converged {
            log::warn!(
                "flood fill did not converge in {} iterations; some interior cells may be \
                 misclassified",
                iterations
            );
        }
        log::info!("{}", stats.report());
        self.stats = Some(stats.clone());
        Ok(stats)
    }

    // -- readback helpers ---------------------------------------------------

    /// Copy the flag byte for every cell back to the CPU. Intended for
    /// validation and for saving a mask; the solver binds the buffer directly.
    pub fn read_flags(&self) -> Result<Vec<u8>> {
        let cells = self.grid.context("nothing voxelised yet")?.cell_count();
        let bytes = self.read_buffer(&self.flags, cells.div_ceil(4) * 4)?;
        Ok(bytes[..cells as usize].to_vec())
    }

    /// Copy the narrow-band signed distance field back to the CPU.
    pub fn read_phi(&self) -> Result<Vec<f32>> {
        let cells = self.grid.context("nothing voxelised yet")?.cell_count();
        let bytes = self.read_buffer(&self.phi, cells * 4)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Copy the boundary link list back to the CPU.
    pub fn read_links(&self) -> Result<Vec<BoundaryLink>> {
        let n = self.stats.as_ref().map(|s| s.link_count).unwrap_or(0) as u64;
        if n == 0 {
            return Ok(Vec::new());
        }
        let bytes = self.read_buffer(&self.links, n * 8)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    fn read_buffer(&self, src: &wgpu::Buffer, len: u64) -> Result<Vec<u8>> {
        // COPY_BUFFER_ALIGNMENT is 4, and a mappable buffer must be a multiple
        // of it.
        let len = len.div_ceil(4) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("geom staging"),
            size: len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.encoder("geom readback");
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, len);
        self.queue.submit([enc.finish()]);
        map_and_read(&self.device, &staging, len)
    }

    fn read_counters(&self) -> Result<[u32; COUNTER_WORDS as usize]> {
        let bytes = map_and_read(&self.device, &self.readback, COUNTER_WORDS * 4)?;
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        let mut out = [0u32; COUNTER_WORDS as usize];
        out.copy_from_slice(&words[..COUNTER_WORDS as usize]);
        Ok(out)
    }

    // -- plumbing -----------------------------------------------------------

    fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    fn pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        uniform_slot: usize,
        workgroups: u32,
    ) {
        let bg = self.bind_group.as_ref().expect("bind group");
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        p.set_pipeline(pipeline);
        p.set_bind_group(0, bg, &[(uniform_slot as u64 * self.uniform_stride) as u32]);
        p.dispatch_workgroups(workgroups.max(1), 1, 1);
    }

    fn write_uniform(&self, slot: usize, u: GeomUniforms) {
        self.queue.write_buffer(
            &self.uniforms,
            slot as u64 * self.uniform_stride,
            bytemuck::bytes_of(&u),
        );
    }

    /// Push the triangle data the GPU does not already have.
    ///
    /// Each instance occupies a contiguous range, so a part that moved can be
    /// re-uploaded as one slice. That matters: the whole soup for the test part
    /// is 13 MB across the two buffers, which is several milliseconds of PCIe
    /// traffic per drag frame if it is sent every time.
    fn upload_geometry(&mut self) {
        let ranges = match self.upload_ranges.take() {
            Some(r) => r,
            None => vec![0..self.geom.len()],
        };
        for r in ranges {
            if r.is_empty() {
                continue;
            }
            let tris: Vec<GpuTri> = self.geom.tris[r.clone()].iter().map(to_gpu_tri).collect();
            let pn: Vec<GpuTriPn> = self.geom.pn[r.clone()].iter().map(to_gpu_pn).collect();
            self.queue.write_buffer(&self.tris, r.start as u64 * 48, bytemuck::cast_slice(&tris));
            self.queue.write_buffer(&self.pn, r.start as u64 * 112, bytemuck::cast_slice(&pn));
        }
        self.queue.write_buffer(&self.active_buf, 0, bytemuck::cast_slice(&self.active));
    }

    fn ensure_buffers(&mut self, grid: Grid) -> Result<()> {
        let cells = grid.cell_count();
        let tri_count = self.geom.len();
        let mut rebuild = self.bind_group.is_none();

        if tri_count > self.tri_capacity {
            let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            let n = tri_count.next_power_of_two().max(1024);
            self.tris = self.buffer("geom tris", n as u64 * 48, storage);
            self.pn = self.buffer("geom tri pn", n as u64 * 112, storage);
            self.active_buf = self.buffer("geom active tris", n as u64 * 4, storage);
            self.tri_capacity = n;
            // Fresh buffers hold nothing, so any partial upload plan is void.
            self.upload_ranges = None;
            rebuild = true;
        }

        if cells > self.cell_capacity {
            let max_binding = 2u64 * 1024 * 1024 * 1024 - 4;
            if cells * 4 > max_binding {
                bail!(
                    "grid {}x{}x{} needs {:.2} GiB per scalar field, above the {:.2} GiB storage \
                     binding limit; use a coarser dx",
                    grid.dims.x,
                    grid.dims.y,
                    grid.dims.z,
                    cells as f64 * 4.0 / (1u64 << 30) as f64,
                    max_binding as f64 / (1u64 << 30) as f64,
                );
            }
            let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            self.dist = self.buffer("geom dist", cells * 4, storage);
            self.tri_idx = self.buffer("geom tri index", cells * 4, storage);
            self.phi =
                self.buffer("geom phi", cells * 4, storage | wgpu::BufferUsages::COPY_SRC);
            self.state = self.buffer("geom fill state", cells * 4, storage);
            self.flags = self.buffer(
                "geom flags",
                cells.div_ceil(4) * 4,
                storage | wgpu::BufferUsages::COPY_SRC,
            );
            self.cell_capacity = cells;
            rebuild = true;
        }

        if self.link_capacity == 0 {
            self.ensure_links(1 << 16)?;
            rebuild = true;
        }

        if rebuild {
            self.rebuild_bind_group();
        }
        Ok(())
    }

    fn ensure_links(&mut self, needed: u32) -> Result<()> {
        if needed <= self.link_capacity {
            return Ok(());
        }
        let n = needed.next_power_of_two().max(1 << 16);
        self.links = self.buffer(
            "geom links",
            n as u64 * 8,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        self.link_capacity = n;
        self.rebuild_bind_group();
        Ok(())
    }

    fn buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage,
            mapped_at_creation: false,
        })
    }

    fn rebuild_bind_group(&mut self) {
        let storage = [
            &self.tris,
            &self.pn,
            &self.active_buf,
            &self.dist,
            &self.tri_idx,
            &self.phi,
            &self.state,
            &self.flags,
            &self.links,
            &self.counters,
        ];
        let mut entries = Vec::with_capacity(11);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.uniforms,
                offset: 0,
                size: std::num::NonZeroU64::new(std::mem::size_of::<GeomUniforms>() as u64),
            }),
        });
        for (i, buf) in storage.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: buf.as_entire_binding(),
            });
        }
        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("geom voxelize"),
            layout: &self.layout,
            entries: &entries,
        }));
    }
}

fn groups(items: u64) -> u32 {
    items.div_ceil(64).clamp(1, 65535) as u32
}

/// Inclusive cell range covering a world-space box, clamped to the grid.
fn cell_range(grid: Grid, b: Bbox) -> (UVec3, UVec3) {
    let last = (grid.dims - UVec3::ONE).as_vec3();
    let lo = ((b.min - grid.origin_mm) / grid.dx_mm).floor().clamp(Vec3::ZERO, last);
    let hi = ((b.max - grid.origin_mm) / grid.dx_mm).ceil().clamp(Vec3::ZERO, last);
    (lo.as_uvec3(), hi.as_uvec3())
}

fn map_and_read(device: &wgpu::Device, buffer: &wgpu::Buffer, len: u64) -> Result<Vec<u8>> {
    let slice = buffer.slice(..len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().context("GPU readback channel closed")?.context("GPU buffer map failed")?;
    let out = match slice.get_mapped_range() {
        Ok(view) => view.to_vec(),
        Err(e) => {
            buffer.unmap();
            bail!("GPU readback range unavailable: {e:?}");
        }
    };
    buffer.unmap();
    Ok(out)
}

fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("geom voxelize"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    // One buffer holds several parameter sets; the dynamic
                    // offset picks between them, which is what lets the three
                    // sweep axes share a command buffer.
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<GeomUniforms>() as u64,
                    ),
                },
                count: None,
            },
            storage(1, true),
            storage(2, true),
            storage(3, true),
            storage(4, false),
            storage(5, false),
            storage(6, false),
            storage(7, false),
            storage(8, false),
            storage(9, false),
            storage(10, false),
        ],
    })
}

/// Assemble the shader from the files in `shaders/geom` plus two generated
/// preludes.
///
/// The sources are embedded with `include_str!` rather than read at run time so
/// the crate works from any working directory and cannot silently pick up a
/// stale file. The lattice tables and flag values are generated from the Rust
/// definitions so the two can never drift, which is the whole point of
/// `ad_gpu::lattice::wgsl_prelude`.
fn build_module(device: &wgpu::Device) -> Result<wgpu::ShaderModule> {
    let mut loader = ShaderLoader::new(".");
    loader.add_virtual("geom/common.wgsl", include_str!("../../../shaders/geom/common.wgsl"));
    loader
        .add_virtual("geom/voxelize.wgsl", include_str!("../../../shaders/geom/voxelize.wgsl"));
    loader.add_virtual("geom/lattice.wgsl", ad_gpu::lattice::wgsl_prelude(VelocitySet::D3Q19));
    loader.add_virtual("geom/flags.wgsl", flags_prelude());
    loader.create_module(device, "geom/voxelize.wgsl", &ShaderDefines::new())
}

fn flags_prelude() -> String {
    format!(
        "// GENERATED from ad_gpu::types::flags - do not edit\n\
         const FLAG_SOLID: u32 = {}u;\n\
         const FLAG_SOLID_BOUNDARY: u32 = {}u;\n",
        flags::SOLID,
        flags::SOLID_BOUNDARY,
    )
}

/// The preprocessed shader source, for a test that wants to inspect it without
/// a GPU.
pub fn shader_source() -> Result<String> {
    let mut loader = ShaderLoader::new(".");
    loader.add_virtual("geom/common.wgsl", include_str!("../../../shaders/geom/common.wgsl"));
    loader
        .add_virtual("geom/voxelize.wgsl", include_str!("../../../shaders/geom/voxelize.wgsl"));
    loader.add_virtual("geom/lattice.wgsl", ad_gpu::lattice::wgsl_prelude(VelocitySet::D3Q19));
    loader.add_virtual("geom/flags.wgsl", flags_prelude());
    loader.load("geom/voxelize.wgsl", &ShaderDefines::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_struct_matches_the_wgsl_layout() {
        // WGSL puts vec3<u32> on a 16-byte boundary, so the padding words are
        // load-bearing. If this ever drifts, every dispatch silently reads the
        // wrong grid dimensions.
        assert_eq!(std::mem::size_of::<GeomUniforms>(), 80);
        assert_eq!(std::mem::size_of::<GpuTri>(), 48);
        assert_eq!(std::mem::size_of::<GpuTriPn>(), 112);
        assert_eq!(std::mem::size_of::<BoundaryLink>(), 8);
    }

    #[test]
    fn shader_preprocesses_and_pulls_in_the_generated_tables() {
        let src = shader_source().expect("preprocessing must not depend on a GPU");
        assert!(src.contains("const Q: u32 = 19u;"), "lattice prelude missing");
        assert!(src.contains("const FLAG_SOLID: u32 = 1u;"), "flag prelude missing");
        assert!(src.contains("fn closest_point_on_tri"), "common.wgsl missing");
        for entry in [
            "clear_region",
            "seed_distance",
            "assign_triangle",
            "sign_band",
            "reset_fill",
            "seed_exterior",
            "sweep",
            "resolve_unknown",
            "classify",
        ] {
            assert!(src.contains(&format!("fn {entry}(")), "entry point {entry} missing");
        }
    }

    #[test]
    fn cell_range_clamps_to_the_grid() {
        let grid = Grid::covering(
            Bbox { min: Vec3::ZERO, max: Vec3::splat(10.0) },
            1.0,
        );
        let (lo, hi) = cell_range(grid, Bbox { min: Vec3::splat(-100.0), max: Vec3::splat(100.0) });
        assert_eq!(lo, UVec3::ZERO);
        assert_eq!(hi, grid.dims - UVec3::ONE);

        let (lo, hi) = cell_range(
            grid,
            Bbox { min: Vec3::splat(3.4), max: Vec3::splat(6.6) },
        );
        assert!(lo.x <= 3 && hi.x >= 7, "range {lo:?}..{hi:?} must cover the box");
    }
}

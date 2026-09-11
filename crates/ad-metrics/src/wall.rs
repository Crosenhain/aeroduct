//! Wall pressure and shear stress per triangle: the driver for
//! `shaders/metrics/wall.wgsl`.
//!
//! # Which route this takes, and why
//!
//! There are three ways to get wall shear out of a lattice-Boltzmann solver,
//! in decreasing order of directness:
//!
//! 1. **Momentum exchange** over the boundary links (Ladd 1994; Mei et al.
//!    2002). Sum `c_i (f_i(x_f) + f_i~(x_f + c_i))` over every link that crosses
//!    the surface. This is the reference method: it is exact for the discrete
//!    system, needs no closure and no gradient, and conserves momentum with the
//!    fluid by construction.
//! 2. **Non-equilibrium stress at the first fluid cell**,
//!    `sigma = -(1 - dt/(2 tau)) Pi_neq` with
//!    `Pi_neq_ab = sum_i c_ia c_ib f_i^neq`.
//! 3. **The strain rate reconstructed from the macroscopic velocity field**,
//!    which is what route 2 collapses to under the second-order Chapman-Enskog
//!    closure.
//!
//! Routes 1 and 2 both need the raw populations. `ad_solver::Solver` keeps the
//! DDF buffers, the flag buffer and the boundary link buffer private and exposes
//! only `velocity_view()` and `density_view()`, so **this module takes route 3**.
//! The algebra that makes it the same number is written out in the shader
//! header; the short version is
//!
//! ```text
//! Pi_neq_ab = -2 rho c_s^2 tau S_ab
//!   =>  sigma_ab = (1 - 1/(2 tau)) 2 rho c_s^2 tau S_ab
//!                = 2 rho c_s^2 (tau - 1/2) S_ab  =  2 rho nu_lb S_ab
//! ```
//!
//! so the `(1 - dt/(2 tau))` factor the second route applies is already present.
//! What is lost relative to route 1 is the discrete exactness at a curved
//! boundary and the local `tau_eff` the LES model produces, which is why
//! [`WallConfig::smagorinsky_c`] exists: the same resolved strain rate that
//! gives `S` also gives the Smagorinsky eddy viscosity, so the closure can be
//! kept consistent with the solver's own.
//!
//! **If `ad-solver` later exposes the populations or the link table, route 1
//! belongs here instead** — see the report accompanying this crate.
//!
//! # What this module refuses to do
//!
//! It never evaluates the field at a triangle vertex, an edge or the face
//! itself. Those points lie *on* the boundary, where half of any interpolation
//! stencil is bounce-back cells holding `u = 0, rho = 1` by fiat; the result is
//! a wall shear map that is about half the truth and looks completely
//! plausible. Instead the kernel walks outward along the normal to the first
//! cell the flag byte calls fluid, and uses the triangle only as the place where
//! `u = 0` is known exactly.
//!
//! # y+
//!
//! `y+ = u_tau y / nu`, `u_tau = sqrt(tau_w / rho)`. Dimensionless, so it needs
//! no unit conversion. It is reported because CONTRACT.md's decision to run with
//! **no wall function** is only valid while the first fluid node stays inside
//! the viscous sublayer. The planning figure is `y+ = 1.8..6.9` at
//! `dx = 0.4 mm`; if [`WallSummary::max_y_plus`] climbs past ~11 the linear
//! profile this module assumes has stopped being true and the shear is being
//! under-read.

use ad_gpu::types::{Grid, LatticeUnits};
use anyhow::{ensure, Result};
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Vec3};

use crate::plane::GpuGrid;
use crate::readback::{Frame, ReadbackRing};
use crate::shaders;
use crate::shaders::{WALL_STRIDE, W_AREA, W_FP_X, W_FV_X, W_P_A, W_TAU_A, W_YPLUS_A, W_Y_A};

/// Mirror of `WallUniforms` in `wall.wgsl`. All scalars after the grid, so
/// there is no `vec3` alignment hole to get wrong.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WallUniforms {
    grid: GpuGrid,
    tri_count: u32,
    nu_lb: f32,
    smagorinsky_c: f32,
    probe_start: f32,
    probe_step: f32,
    probe_count: u32,
    tangent_h: f32,
    pad: f32,
}

const _: () = assert!(std::mem::size_of::<WallUniforms>() == 64);

/// Mirror of `Tri` in `wall.wgsl`: three `vec3<f32>` at 16-byte stride.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuTri {
    pub a: [f32; 3],
    pub _pa: f32,
    pub b: [f32; 3],
    pub _pb: f32,
    pub c: [f32; 3],
    pub _pc: f32,
}

const _: () = assert!(std::mem::size_of::<GpuTri>() == 48);

impl GpuTri {
    pub fn new(t: [Vec3; 3]) -> Self {
        Self {
            a: t[0].to_array(),
            _pa: 0.0,
            b: t[1].to_array(),
            _pb: 0.0,
            c: t[2].to_array(),
            _pc: 0.0,
        }
    }
}

/// How far off the surface to look, and what closure to apply.
#[derive(Debug, Clone, Copy)]
pub struct WallConfig {
    /// First probe offset from the triangle plane, in cells.
    ///
    /// 1.5 is deliberate. With halfway bounce-back the wall sits half a cell
    /// from the first fluid node, so 0.5 would land *on* the node nearest the
    /// wall — whose trilinear stencil still reaches into bounce-back cells.
    /// Starting a full cell further out costs a little resolution of the
    /// gradient and buys a sample whose whole stencil is fluid.
    pub probe_start_cells: f32,
    /// Spacing between successive probes when the first lands in solid.
    pub probe_step_cells: f32,
    /// How many probes to try before declaring the triangle dry. Four covers a
    /// staircase artefact on a diagonal surface without letting a triangle
    /// buried inside a neighbouring part borrow a distant cell's stress.
    pub probe_count: u32,
    /// Tangential finite-difference step, in cells.
    pub tangent_h_cells: f32,
    /// Smagorinsky constant for the eddy-viscosity correction, matching
    /// `SolverConfig::smagorinsky_c`. Zero reports the molecular stress only,
    /// which is right when the solver's LES is also off.
    pub smagorinsky_c: f32,
}

impl Default for WallConfig {
    fn default() -> Self {
        Self {
            probe_start_cells: 1.5,
            probe_step_cells: 1.0,
            probe_count: 4,
            tangent_h_cells: 1.0,
            smagorinsky_c: 0.0,
        }
    }
}

/// Pipeline, triangle buffer, per-triangle records and readback.
///
/// The records buffer stays resident on the GPU so a renderer can paint the
/// surface straight from it ([`Self::records_buffer`]); the readback is only for
/// the reported aggregate and is opt-in per call, because at 85,180 triangles
/// one capture is 4 MB and nobody needs that every frame.
pub struct WallMetrics {
    pipeline: wgpu::ComputePipeline,
    uniform: wgpu::Buffer,
    tris: wgpu::Buffer,
    records: wgpu::Buffer,
    group1: wgpu::BindGroup,
    ring: ReadbackRing<f32>,
    grid: Grid,
    tri_count: u32,
    capacity: u32,
    cfg: WallConfig,
}

impl WallMetrics {
    /// `triangles` are in **millimetres**, wound outward per the STL convention,
    /// so the face normal points from the solid into the fluid.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        field_layout: &wgpu::BindGroupLayout,
        grid: Grid,
        triangles: &[[Vec3; 3]],
        cfg: WallConfig,
    ) -> Result<Self> {
        ensure!(
            !triangles.is_empty(),
            "wall metrics need at least one triangle"
        );
        let capacity = triangles.len() as u32;

        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("metrics wall group1"),
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
                storage_entry(2, true),
            ],
        });

        let loader = shaders::build_loader();
        let module = loader.create_module(device, "metrics/wall.wgsl", &shaders::defines())?;
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("metrics wall layout"),
            bind_group_layouts: &[Some(field_layout), Some(&group1_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("metrics wall"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("wall_stress"),
            compilation_options: Default::default(),
            cache: None,
        });

        let tris = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metrics wall triangles"),
            size: capacity as u64 * std::mem::size_of::<GpuTri>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let records = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metrics wall records"),
            size: capacity as u64 * (WALL_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metrics wall uniforms"),
            size: std::mem::size_of::<WallUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("metrics wall group1"),
            layout: &group1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: records.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tris.as_entire_binding(),
                },
            ],
        });

        let mut me = Self {
            pipeline,
            uniform,
            tris,
            records,
            group1,
            ring: ReadbackRing::new(device, "metrics wall", capacity as usize * WALL_STRIDE, 2),
            grid,
            tri_count: 0,
            capacity,
            cfg,
        };
        me.set_triangles(queue, triangles)?;
        Ok(me)
    }

    /// Replace the triangle set. Must not exceed the capacity the instance was
    /// built with, because the records buffer and the readback ring are sized to
    /// it; a larger set needs a new [`WallMetrics`].
    pub fn set_triangles(&mut self, queue: &wgpu::Queue, triangles: &[[Vec3; 3]]) -> Result<()> {
        ensure!(
            triangles.len() as u32 <= self.capacity,
            "{} triangles exceeds the capacity of {}",
            triangles.len(),
            self.capacity
        );
        let gpu: Vec<GpuTri> = triangles.iter().copied().map(GpuTri::new).collect();
        queue.write_buffer(&self.tris, 0, bytemuck::cast_slice(&gpu));
        self.tri_count = triangles.len() as u32;
        Ok(())
    }

    pub fn triangle_count(&self) -> u32 {
        self.tri_count
    }

    pub fn config(&self) -> &WallConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: WallConfig) {
        self.cfg = cfg;
    }

    /// The per-triangle record buffer, for a surface-painting render pass.
    /// `WALL_STRIDE` `f32`s per triangle; see [`crate::shaders`] for the slots.
    pub fn records_buffer(&self) -> &wgpu::Buffer {
        &self.records
    }

    /// Dispatch one thread per triangle. `capture` also queues a readback, which
    /// costs `12 * 4 * tri_count` bytes over the bus — do it every few hundred
    /// frames, not every frame.
    ///
    /// `nu_lb` is the solver's base lattice viscosity, `c_s^2 (tau0 - 1/2)`.
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        field_group: &wgpu::BindGroup,
        nu_lb: f32,
        step: u64,
        capture: bool,
    ) -> bool {
        if self.tri_count == 0 {
            return false;
        }
        let u = WallUniforms {
            grid: GpuGrid::new(self.grid),
            tri_count: self.tri_count,
            nu_lb,
            smagorinsky_c: self.cfg.smagorinsky_c,
            probe_start: self.cfg.probe_start_cells,
            probe_step: self.cfg.probe_step_cells,
            probe_count: self.cfg.probe_count.max(1),
            tangent_h: self.cfg.tangent_h_cells,
            pad: 0.0,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("metrics wall"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, field_group, &[]);
            pass.set_bind_group(1, &self.group1, &[]);
            pass.dispatch_workgroups(self.tri_count.div_ceil(64), 1, 1);
        }
        if capture && self.ring.has_capacity() {
            return self.ring.record(encoder, &self.records, 0, step);
        }
        false
    }

    pub fn poll(&mut self, device: &wgpu::Device) -> Vec<Frame<f32>> {
        self.ring.poll(device)
    }

    pub fn drain_blocking(&mut self, device: &wgpu::Device) -> Vec<Frame<f32>> {
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

/// One triangle's wall state, in SI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallTriangle {
    /// False when the probe walk never found fluid. Every other field is then
    /// meaningless and must not be plotted as zero shear — a triangle inside
    /// another part is not a triangle in still air.
    pub wetted: bool,
    pub area_mm2: f64,
    /// Wall shear stress magnitude, Pa.
    pub shear_pa: f64,
    /// Static pressure at the probe, Pa, relative to the solver's reference.
    pub pressure_pa: f64,
    pub y_plus: f64,
    /// Distance from the triangle to the probe, in cells.
    pub probe_distance_cells: f64,
    /// Pressure force on this triangle, newtons.
    pub pressure_force_n: DVec3,
    /// Viscous traction force on this triangle, newtons.
    pub viscous_force_n: DVec3,
}

/// A readback of the per-triangle records, plus the units to read them in.
///
/// Borrowed rather than owned: at 85k triangles the record set is 4 MB and
/// there is no reason to copy it to compute a handful of sums.
pub struct WallField<'a> {
    records: &'a [f32],
    /// Pascals per unit of lattice stress, `rho_phys (dx/dt)^2`.
    stress_factor: f64,
}

impl<'a> WallField<'a> {
    pub fn new(records: &'a [f32], lu: &LatticeUnits) -> Self {
        let c_u = lu.c_u();
        Self {
            records,
            stress_factor: lu.rho_phys * c_u * c_u,
        }
    }

    pub fn len(&self) -> usize {
        self.records.len() / WALL_STRIDE
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn slot(&self, i: usize, s: usize) -> f64 {
        self.records[i * WALL_STRIDE + s] as f64
    }

    fn vec(&self, i: usize, s: usize) -> DVec3 {
        DVec3::new(self.slot(i, s), self.slot(i, s + 1), self.slot(i, s + 2))
    }

    /// Triangle `i`, in SI.
    ///
    /// The record holds every quantity pre-multiplied by the triangle area, so
    /// the per-triangle value is a division and the aggregate is a sum — which
    /// is what makes an area-weighted mean a one-pass operation.
    pub fn triangle(&self, i: usize) -> WallTriangle {
        let area = self.slot(i, W_AREA);
        if !(area > 0.0) {
            return WallTriangle {
                wetted: false,
                area_mm2: 0.0,
                shear_pa: 0.0,
                pressure_pa: 0.0,
                y_plus: 0.0,
                probe_distance_cells: 0.0,
                pressure_force_n: DVec3::ZERO,
                viscous_force_n: DVec3::ZERO,
            };
        }
        // Forces carry lattice stress times mm^2; 1e-6 turns mm^2 into m^2.
        let f = self.stress_factor * 1e-6;
        WallTriangle {
            wetted: true,
            area_mm2: area,
            shear_pa: self.slot(i, W_TAU_A) / area * self.stress_factor,
            pressure_pa: self.slot(i, W_P_A) / area * self.stress_factor,
            y_plus: self.slot(i, W_YPLUS_A) / area,
            probe_distance_cells: self.slot(i, W_Y_A) / area,
            pressure_force_n: self.vec(i, W_FP_X) * f,
            viscous_force_n: self.vec(i, W_FV_X) * f,
        }
    }

    /// Every aggregate, in one pass.
    pub fn summary(&self) -> WallSummary {
        let n = self.len();
        let mut s = WallSummary {
            triangles: n,
            wetted_triangles: 0,
            wetted_area_mm2: 0.0,
            pressure_force_n: DVec3::ZERO,
            viscous_force_n: DVec3::ZERO,
            mean_shear_pa: 0.0,
            max_shear_pa: 0.0,
            mean_pressure_pa: 0.0,
            mean_y_plus: 0.0,
            max_y_plus: 0.0,
            mean_probe_distance_cells: 0.0,
        };
        let (mut tau_a, mut yp_a, mut p_a, mut y_a) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let area = self.slot(i, W_AREA);
            if !(area > 0.0) {
                continue;
            }
            s.wetted_triangles += 1;
            s.wetted_area_mm2 += area;
            s.pressure_force_n += self.vec(i, W_FP_X);
            s.viscous_force_n += self.vec(i, W_FV_X);
            tau_a += self.slot(i, W_TAU_A);
            yp_a += self.slot(i, W_YPLUS_A);
            p_a += self.slot(i, W_P_A);
            y_a += self.slot(i, W_Y_A);
            s.max_shear_pa = s.max_shear_pa.max(self.slot(i, W_TAU_A) / area);
            s.max_y_plus = s.max_y_plus.max(self.slot(i, W_YPLUS_A) / area);
        }
        let f = self.stress_factor * 1e-6;
        s.pressure_force_n *= f;
        s.viscous_force_n *= f;
        if s.wetted_area_mm2 > 0.0 {
            let a = s.wetted_area_mm2;
            s.mean_shear_pa = tau_a / a * self.stress_factor;
            s.mean_pressure_pa = p_a / a * self.stress_factor;
            s.mean_y_plus = yp_a / a;
            s.mean_probe_distance_cells = y_a / a;
        }
        s.max_shear_pa *= self.stress_factor;
        s
    }
}

/// Aggregates over the whole wetted surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallSummary {
    pub triangles: usize,
    /// Triangles that found fluid. A large shortfall means the surface is buried
    /// or the probe walk is too short, not that the duct is sealed.
    pub wetted_triangles: usize,
    pub wetted_area_mm2: f64,
    /// Net pressure force on the surface, newtons.
    pub pressure_force_n: DVec3,
    /// Net viscous force on the surface, newtons.
    pub viscous_force_n: DVec3,
    /// Area-weighted mean wall shear stress, Pa.
    pub mean_shear_pa: f64,
    pub max_shear_pa: f64,
    /// Area-weighted mean static wall pressure, Pa.
    pub mean_pressure_pa: f64,
    pub mean_y_plus: f64,
    pub max_y_plus: f64,
    /// Area-weighted mean probe distance, in cells. Says where the shear was
    /// actually measured, which is the honest caveat on all of the above.
    pub mean_probe_distance_cells: f64,
}

impl WallSummary {
    /// Total force the air exerts on the part, newtons.
    pub fn total_force_n(&self) -> DVec3 {
        self.pressure_force_n + self.viscous_force_n
    }

    /// Fraction of the supplied triangles that found fluid.
    pub fn wetted_fraction(&self) -> f64 {
        if self.triangles == 0 {
            0.0
        } else {
            self.wetted_triangles as f64 / self.triangles as f64
        }
    }

    /// Whether the first fluid node is still inside the viscous sublayer.
    ///
    /// CONTRACT.md's decision to use **no wall function** rests on this. Above
    /// `y+ ~ 11` the profile has left the linear region and both the shear
    /// computed here and the solver's plain bounce-back are being asked for more
    /// than they can give.
    pub fn resolves_viscous_sublayer(&self) -> bool {
        self.max_y_plus < 11.0
    }

    pub fn summary_line(&self) -> String {
        format!(
            "wall: tau = {:.4} Pa mean, {:.4} Pa peak; y+ = {:.2} mean, {:.2} peak ({}); \
             force = {:.4} N over {:.0} mm^2 of {} triangles",
            self.mean_shear_pa,
            self.max_shear_pa,
            self.mean_y_plus,
            self.max_y_plus,
            if self.resolves_viscous_sublayer() {
                "viscous sublayer resolved"
            } else {
                "OUTSIDE the viscous sublayer; shear is under-read"
            },
            self.total_force_n().length(),
            self.wetted_area_mm2,
            self.wetted_triangles,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{field_layout, FieldRefs, FieldTextures};
    use ad_gpu::types::{flags, Bbox};

    fn units() -> LatticeUnits {
        LatticeUnits::for_air(1.0, 2.0, 0.05)
    }

    /// A 32 mm cube of grid at dx = 1 mm, with a flat wall at `y = y_w` and
    /// everything below it solid.
    fn shear_case(
        gpu: &ad_gpu::GpuContext,
        y_w: f32,
        gradient: f32,
        tris: &[[Vec3; 3]],
    ) -> Vec<f32> {
        let grid = Grid::covering(
            Bbox {
                min: Vec3::ZERO,
                max: Vec3::splat(32.0),
            },
            1.0,
        );
        let tex = FieldTextures::new_exact(&gpu.device, grid);
        tex.fill(&gpu.queue, |_, p| {
            if p.y < y_w {
                (Vec3::ZERO, 1.0, flags::SOLID)
            } else {
                (
                    Vec3::new(gradient * (p.y - y_w), 0.0, 0.0),
                    1.0,
                    flags::FLUID,
                )
            }
        })
        .unwrap();
        run(gpu, grid, &tex, tris, WallConfig::default())
    }

    fn run(
        gpu: &ad_gpu::GpuContext,
        grid: Grid,
        tex: &FieldTextures,
        tris: &[[Vec3; 3]],
        cfg: WallConfig,
    ) -> Vec<f32> {
        let layout = field_layout(&gpu.device);
        let (vv, dv) = (tex.velocity_view(), tex.density_view());
        let group = crate::field::field_bind_group(
            &gpu.device,
            &layout,
            &FieldRefs {
                grid,
                velocity: &vv,
                density: &dv,
                flags: tex.flags_buffer(),
            },
        );
        let mut wm = WallMetrics::new(&gpu.device, &gpu.queue, &layout, grid, tris, cfg).unwrap();
        let nu_lb = units().nu_lb() as f32;
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(wm.record(&gpu.queue, &mut enc, &group, nu_lb, 1, true));
        gpu.queue.submit([enc.finish()]);
        let frames = wm.drain_blocking(&gpu.device);
        assert!(!frames.is_empty(), "no wall readback arrived");
        frames[0].data.clone()
    }

    /// Two triangles at `y = y_w` with normal +Y, forming a 2x2 mm square.
    fn upward_pair(y_w: f32) -> Vec<[Vec3; 3]> {
        let p = |x: f32, z: f32| Vec3::new(x, y_w, z);
        // Wound so that (b - a) x (c - a) points along +Y.
        vec![
            [p(10.0, 10.0), p(10.0, 12.0), p(12.0, 10.0)],
            [p(12.0, 12.0), p(12.0, 10.0), p(10.0, 12.0)],
        ]
    }

    /// Couette shear against a flat wall: `tau_w = rho nu du/dy` exactly.
    ///
    /// This is the test that would catch the classic failure — evaluating the
    /// field at the surface instead of off it, which halves the answer because
    /// the interpolation stencil straddles bounce-back cells. A factor of two is
    /// far outside the 2% tolerance here.
    #[test]
    fn a_linear_shear_layer_reproduces_mu_du_dy() {
        let Some(gpu) = crate::test_gpu() else { return };
        let lu = units();
        let g_per_mm = 0.01f32; // lattice velocity per millimetre
        let tris = upward_pair(8.0);
        let records = shear_case(&gpu, 8.0, g_per_mm, &tris);

        let field = WallField::new(&records, &lu);
        assert_eq!(field.len(), 2);
        // dx = 1 mm, so the gradient per cell equals the gradient per mm.
        let want_tau_lb = lu.nu_lb() * g_per_mm as f64;
        let want_tau_pa = want_tau_lb * lu.rho_phys * lu.c_u() * lu.c_u();
        let want_y_plus = (want_tau_lb).sqrt() * 1.5 / lu.nu_lb();

        for i in 0..2 {
            let t = field.triangle(i);
            assert!(t.wetted, "triangle {i} found no fluid");
            assert!(
                (t.shear_pa / want_tau_pa - 1.0).abs() < 0.02,
                "triangle {i}: tau_w = {} Pa, analytic {want_tau_pa} Pa (ratio {})",
                t.shear_pa,
                t.shear_pa / want_tau_pa
            );
            assert!(
                (t.y_plus / want_y_plus - 1.0).abs() < 0.02,
                "triangle {i}: y+ = {}, analytic {want_y_plus}",
                t.y_plus
            );
            assert!((t.probe_distance_cells - 1.5).abs() < 1e-4);
            // rho = 1 everywhere, so there is no pressure force at all.
            assert!(
                t.pressure_force_n.length() < 1e-18,
                "{:?}",
                t.pressure_force_n
            );
            // The viscous force is tangential: along +X, the flow direction.
            let f = t.viscous_force_n;
            assert!(
                f.x > 0.0 && f.y.abs() < 1e-3 * f.x.abs(),
                "viscous force {f:?}"
            );
        }

        let s = field.summary();
        assert_eq!(s.wetted_triangles, 2);
        assert!(
            (s.wetted_area_mm2 - 4.0).abs() < 1e-4,
            "area {}",
            s.wetted_area_mm2
        );
        assert!((s.mean_shear_pa / want_tau_pa - 1.0).abs() < 0.02);
        assert!((s.max_shear_pa / want_tau_pa - 1.0).abs() < 0.02);
        assert!(s.wetted_fraction() == 1.0);
        // At y+ ~ 7.6 the contract's no-wall-function decision still holds.
        assert!(
            s.resolves_viscous_sublayer(),
            "y+ peak was {}",
            s.max_y_plus
        );
        assert!(s.summary_line().contains("resolved"));
    }

    /// Doubling the gradient doubles the shear and multiplies y+ by sqrt(2).
    /// Two runs of the same code, so any constant-factor error cancels and only
    /// the scaling is under test.
    #[test]
    fn wall_shear_scales_linearly_and_y_plus_as_its_square_root() {
        let Some(gpu) = crate::test_gpu() else { return };
        let lu = units();
        let tris = upward_pair(8.0);
        let a = shear_case(&gpu, 8.0, 0.005, &tris);
        let b = shear_case(&gpu, 8.0, 0.010, &tris);
        let ta = WallField::new(&a, &lu).triangle(0);
        let tb = WallField::new(&b, &lu).triangle(0);
        assert!(
            (tb.shear_pa / ta.shear_pa - 2.0).abs() < 0.02,
            "shear ratio {}",
            tb.shear_pa / ta.shear_pa
        );
        assert!(
            (tb.y_plus / ta.y_plus - 2.0f64.sqrt()).abs() < 0.02,
            "y+ ratio {}",
            tb.y_plus / ta.y_plus
        );
    }

    /// A triangle whose normal points into the solid finds no fluid, and says
    /// so rather than reporting zero shear on a surface it never measured.
    #[test]
    fn a_triangle_facing_into_the_wall_reports_itself_dry() {
        let Some(gpu) = crate::test_gpu() else { return };
        let lu = units();
        // Same square, wound the other way, so the normal is -Y.
        let p = |x: f32, z: f32| Vec3::new(x, 8.0, z);
        let tris = vec![[p(10.0, 10.0), p(12.0, 10.0), p(10.0, 12.0)]];
        let records = shear_case(&gpu, 8.0, 0.01, &tris);
        let field = WallField::new(&records, &lu);
        let t = field.triangle(0);
        assert!(!t.wetted, "a downward face should find only solid");
        assert_eq!(t.shear_pa, 0.0);
        let s = field.summary();
        assert_eq!(s.wetted_triangles, 0);
        assert_eq!(s.wetted_fraction(), 0.0);
        assert_eq!(s.wetted_area_mm2, 0.0);
    }

    /// A stationary fluid at uniform over-pressure puts a pure normal force on
    /// the surface, `F = -p n A`, and no shear at all.
    #[test]
    fn a_uniform_pressure_field_gives_a_pure_normal_force() {
        let Some(gpu) = crate::test_gpu() else { return };
        let lu = units();
        let grid = Grid::covering(
            Bbox {
                min: Vec3::ZERO,
                max: Vec3::splat(32.0),
            },
            1.0,
        );
        let drho = 0.002f32;
        let tex = FieldTextures::new_exact(&gpu.device, grid);
        tex.fill(&gpu.queue, |_, p| {
            if p.y < 8.0 {
                (Vec3::ZERO, 1.0, flags::SOLID)
            } else {
                (Vec3::ZERO, 1.0 + drho, flags::FLUID)
            }
        })
        .unwrap();
        let tris = upward_pair(8.0);
        let records = run(&gpu, grid, &tex, &tris, WallConfig::default());
        let field = WallField::new(&records, &lu);

        let s = field.summary();
        let want_p = lu.pressure_pa(drho as f64);
        assert!(
            (s.mean_pressure_pa / want_p - 1.0).abs() < 1e-3,
            "wall pressure {} vs {want_p}",
            s.mean_pressure_pa
        );
        assert!(
            s.mean_shear_pa < 1e-12,
            "still air has no shear: {}",
            s.mean_shear_pa
        );
        // F = -p n A, with n = +Y and A = 4 mm^2, so the force is -Y.
        let want_f = -want_p * 4.0e-6;
        assert!(
            (s.pressure_force_n.y / want_f - 1.0).abs() < 1e-3,
            "F_y = {} vs {want_f}",
            s.pressure_force_n.y
        );
        assert!(s.pressure_force_n.x.abs() < 1e-12 * want_f.abs().max(1e-12));
        assert!(s.viscous_force_n.length() < 1e-15);
    }

    #[test]
    fn an_empty_field_summarises_to_nothing_rather_than_dividing_by_zero() {
        let lu = units();
        let f = WallField::new(&[], &lu);
        assert!(f.is_empty());
        let s = f.summary();
        assert_eq!(s.triangles, 0);
        assert_eq!(s.mean_shear_pa, 0.0);
        assert_eq!(s.wetted_fraction(), 0.0);
        assert!(s.resolves_viscous_sublayer());
    }
}

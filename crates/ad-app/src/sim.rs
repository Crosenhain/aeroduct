//! Geometry in, running solver out.
//!
//! This is the integration seam the other crates were written to meet:
//!
//! ```text
//! STL -> ad_geom::Scene -> Voxelizer -> flag mask + boundary links
//!                                    -> boundary conditions (this file)
//!                                    -> ad_solver::Solver
//! ```
//!
//! # The part that is not mechanical
//!
//! The voxeliser produces `SOLID` and `SOLID_BOUNDARY` and nothing else — it
//! knows about geometry, not about flow. Turning that into a solvable problem
//! means deciding where the air comes in, where it leaves and what the sides of
//! the box do, and those three decisions are [`apply_boundaries`]. They follow
//! CONTRACT.md exactly:
//!
//! * **Inlet**: a plane one cell thick, restricted to the mouth's own
//!   footprint. Everything else on that plane is open air, and flagging it
//!   would inject a sheet of flow across the whole domain.
//! * **Outlet**: the domain face the *other* mouth points at, as a convective
//!   outflow with a sponge. Not the mouth plane itself: the exit flow needs
//!   room to leave, and pinning the outlet at the mouth would measure the duct
//!   with its exit sealed against a boundary condition.
//! * **Box sides**: equilibrium, so the flow can entrain surrounding air. Never
//!   periodic.
//!
//! Where those sides *are* is [`crate::domain`], and in one of the two domains
//! it can build there are no fluid box sides at all. [`crate::plenum`] replaces
//! the room with a walled straight extension on each mouth, which moves the
//! inlet plane 2 `D_h` upstream of the mouth — the development length
//! CONTRACT.md asks for and the room domain only ever approximated — and leaves
//! the sides behind the duct wall where an equilibrium plane cannot impose a
//! pressure on anything.
//!
//! Either way, mouth detection happens before the grid exists, because every
//! margin and every plenum is derived from where the mouths are and which way
//! they face. That is the one ordering constraint in [`Sim::from_scene`].
//!
//! # What the room domain is missing, and what the plenum domain supplies
//!
//! In the room domain there is no plate behind the inlet plane. The velocity
//! boundary sets `f = f^eq(rho, u)` in those cells, which drives flow along
//! `+n` into the duct and draws air toward the plane from behind, so it behaves
//! like a fan disc rather than a leak. Measured, that costs more than it looks:
//! one cell downstream of the mouth the room domain reads `|u|` mean 3.47 m/s
//! against a through-plane component of 2.97, with a peak of 10.6 — air
//! arriving sideways and turning into the opening. The plenum domain reads 3.06
//! against 3.05 with a peak of 5.5 at the same plane, which is a duct profile.

use ad_geom::{
    FlatGeometry, MeshAsset, MeshInstance, MeshRole, Mouth, MouthConfig, Scene, Transform,
    Voxelizer,
};
use ad_gpu::{flags, Bbox, BoundaryLink, DdfPrecision, GpuContext, Grid, LatticeUnits};
use ad_solver::{Solver, SolverConfig};
use ad_ui::SimParams;
use anyhow::{Context as _, Result};
use glam::{UVec3, Vec3};

use crate::domain::DomainMargins;

/// A vent standing free in the room: one cell's thickness of velocity inlet on
/// a rectangle, in the lattice frame.
///
/// With any vent present the duct's mouths are plain openings and the vents
/// are the only air supply: what reaches the inlet mouth is whatever the jets
/// deliver, measured rather than prescribed. Up to three (the solver's inlet
/// slots 1-3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VentPatch {
    pub center_mm: Vec3,
    /// Unit, the way the vent faces. The plane its cells go on is the nearest
    /// lattice axis to this — the inlet density closure needs an axis-aligned
    /// plane — while the air leaves along [`Self::direction`] as given.
    pub normal: Vec3,
    /// In-plane half extents, as vectors along the rectangle's two axes.
    pub half_u: Vec3,
    pub half_v: Vec3,
    /// Unit direction the air leaves in; the louver aim, when it differs from
    /// the normal.
    pub direction: Vec3,
    /// Air speed as a multiple of the operating point's inlet velocity.
    pub speed_scale: f32,
}

impl VentPatch {
    /// The lattice axis nearest the vent's normal, signed: the plane its cells
    /// lie on and the normal the density closure is given.
    pub fn axis_normal(&self) -> Vec3 {
        let n = self.normal;
        let axis = (0..3)
            .max_by(|a, b| n[*a].abs().total_cmp(&n[*b].abs()))
            .unwrap_or(2);
        let mut a = Vec3::ZERO;
        a[axis] = if n[axis] < 0.0 { -1.0 } else { 1.0 };
        a
    }

    /// Whether the cells this marks are the same: the plane and the rectangle,
    /// not the air. The air is uniform data; the cells need a rebuild.
    pub fn same_cells(&self, o: &Self) -> bool {
        self.center_mm == o.center_mm
            && self.normal == o.normal
            && self.half_u == o.half_u
            && self.half_v == o.half_v
    }
}

/// Everything downstream of the geometry: the lattice, the flags, the solver.
pub struct Sim {
    pub scene: Scene,
    /// The vents the lattice was flagged with, lattice frame. Their air
    /// (`direction`, `speed_scale`) may be edited in place and hot-applied;
    /// their cells may not.
    pub vents: Vec<VentPatch>,
    /// The resolved domain, kept because *which* domain was built changes what
    /// other passes may assume. [`crate::tracers::passage_volume_mm3`] is the
    /// case in point: its fill is seeded from the outside of the part, which
    /// only exists in a room.
    pub domain: crate::domain::Domain,
    pub grid: Grid,
    pub units: LatticeUnits,
    pub solver: Solver,
    /// Per-cell flag byte, interior grid, X-fastest. Kept on the CPU because
    /// the flags texture the renderer reads is uploaded from it and because the
    /// inlet/outlet marking has to be redone whenever the mouths change.
    pub mask: Vec<u8>,
    pub links: Vec<BoundaryLink>,
    /// Detected mouths, largest open area first. Index 0 is "A".
    pub mouths: Vec<Mouth>,
    /// The mouths air comes in through, primary first: the chosen inlet mouth
    /// when it is driven directly, or every mouth a vent is sealed to. The
    /// primary is the one the metrics take their reference from.
    pub inlets: Vec<usize>,
    /// The mouth whose face carries the outflow condition.
    pub outlet: usize,
    /// `R8Uint` copy of `mask`, for `ad_render`'s derive pass to mask walls.
    pub flags_texture: wgpu::Texture,
    pub flags_view: wgpu::TextureView,
    /// Hydraulic diameter of the inlet mouth, mm. Feeds `DeriveScales` and the
    /// Reynolds number.
    pub d_h_mm: f64,
    /// Triangles in the scene, for the status bar.
    pub triangle_count: usize,
    /// What the last voxelisation cost, for the status bar.
    pub voxel_report: String,
}

impl Sim {
    /// One line for the status bar: what was voxelised, onto what, and how much
    /// interpolated-bounce-back data came out of it.
    pub fn describe(&self) -> String {
        format!(
            "{} tris | {}x{}x{} @ {} mm | {} boundary links",
            self.triangle_count,
            self.grid.dims.x,
            self.grid.dims.y,
            self.grid.dims.z,
            self.grid.dx_mm,
            self.links.len(),
        )
    }

    /// Re-upload the flag mask to the texture the renderer masks walls with.
    ///
    /// Idempotent, and cheap next to a voxelisation. Called whenever the mask
    /// changes so the derive pass can never be one boundary-condition edit
    /// behind the solver — a stale wall mask puts a bright shell on the
    /// Q-criterion view exactly where the geometry is.
    pub fn upload_flags(&self, queue: &wgpu::Queue) {
        write_flags(queue, &self.flags_texture, self.grid, &self.mask);
    }
}

impl Sim {
    /// Load an STL and build everything on top of it.
    pub fn from_stl(gpu: &GpuContext, path: &std::path::Path, params: &SimParams) -> Result<Self> {
        let duct = Self::load_duct(path)?;
        let mut scene = Scene::new();
        scene.add(duct.name, duct.asset, duct.transform, duct.role);
        Self::from_scene(gpu, scene, Vec::new(), params)
    }

    /// Load an STL as the duct: in its own file frame, which is the lattice
    /// frame. Split out so a new duct can join the obstructions already placed.
    pub fn load_duct(path: &std::path::Path) -> Result<MeshInstance> {
        let load = ad_geom::load_stl(path)?;
        log::info!(
            "{}: {} triangles, {}",
            path.display(),
            load.mesh.triangle_count(),
            load.health().report()
        );
        if load.flipped {
            log::warn!("the STL was wound inside-out and has been reversed");
        }
        Ok(MeshInstance {
            name: path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "duct".into()),
            asset: MeshAsset::new(load.mesh),
            transform: Transform::default(),
            role: MeshRole::Duct,
            visible: true,
        })
    }

    /// Build from an already-populated scene, with `vents` (lattice frame)
    /// supplying the air if there are any; otherwise the inlet mouth does.
    pub fn from_scene(
        gpu: &GpuContext,
        mut scene: Scene,
        mut vents: Vec<VentPatch>,
        params: &SimParams,
    ) -> Result<Self> {
        anyhow::ensure!(!scene.is_empty(), "the scene has no geometry");
        if vents.len() > MAX_VENTS {
            log::warn!(
                "{} vents asked for; the solver has slots for {MAX_VENTS}, the rest are ignored",
                vents.len()
            );
            vents.truncate(MAX_VENTS);
        }
        let triangle_count = scene.triangle_count();

        // Mouths first, and *before* the domain: the box is anisotropic and every
        // one of its six margins is derived from where the mouths are and which
        // way they face. Detection reads the scene mesh and the scene bbox, so it
        // owes the lattice nothing and can run ahead of it.
        let mut mouths = ad_geom::detect_in_scene(&scene, MouthConfig::default());
        // Largest first, so index 0 is CONTRACT.md's "mouth A" whatever order
        // the face sweep happened to find them in.
        mouths.sort_by(|a, b| b.open_area_mm2.total_cmp(&a.open_area_mm2));
        for (i, m) in mouths.iter().enumerate() {
            // The centre and the footprint go in the log alongside the area:
            // they are what `crate::domain` scales the margins from, so a domain
            // that came out an unexpected shape can be traced to the mouth that
            // asked for it without attaching a debugger.
            let half = m.patch.half_u.abs() + m.patch.half_v.abs();
            log::info!(
                "mouth {}: {:.0} mm^2 open, D_h {:.1} mm, axis {} {}, normal {:?}, \
                 centre ({:.1}, {:.1}, {:.1}) mm, footprint {:.0} x {:.0} x {:.0} mm",
                (b'A' + i as u8) as char,
                m.open_area_mm2,
                m.hydraulic_diameter_mm(),
                m.axis,
                if m.on_min_side { "min" } else { "max" },
                m.patch.normal,
                m.patch.center_mm.x,
                m.patch.center_mm.y,
                m.patch.center_mm.z,
                2.0 * half.x,
                2.0 * half.y,
                2.0 * half.z,
            );
        }
        anyhow::ensure!(
            mouths.len() >= 2,
            "found {} mouth(s); a duct needs two openings flush with its bounding box",
            mouths.len()
        );

        let (inlets, outlet) = mouth_roles(&mouths, &vents, params, params.dx_mm);
        let inlet = inlets[0];
        if inlets.len() > 1 {
            log::info!(
                "air comes in through mouths {}; outlet {}",
                inlets
                    .iter()
                    .map(|i| ((b'A' + *i as u8) as char).to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                (b'A' + outlet as u8) as char
            );
        }
        let (domain, grid) = plan_lattice(
            scene.bbox_of_role(MeshRole::Duct),
            &mouths,
            inlet,
            outlet,
            params,
        );
        log::info!(
            "domain {} -> grid {} x {} x {} at dx = {} mm ({:.1} M cells)",
            domain.describe(),
            grid.dims.x,
            grid.dims.y,
            grid.dims.z,
            grid.dx_mm,
            grid.cell_count() as f64 / 1e6,
        );
        if let Some(p) = domain.plenum {
            let d = p.depths(&mouths, inlet, outlet, params.dx_mm, params.sponge_cells);
            if let Some(d) = d {
                log::info!(
                    "plenums: {:.0} mm ({:.2} D_h) upstream of the inlet, {:.0} mm ({:.2} D_h) \
                     downstream of the outlet, {} sides",
                    d.upstream,
                    d.upstream / d.d_h_in,
                    d.downstream,
                    d.downstream / d.d_h_out,
                    if p.walls == crate::plenum::PlenumWalls::Solid {
                        "walled"
                    } else {
                        "open"
                    },
                );
            }
        }

        // Local: the voxeliser's scratch buffers are large and a geometry change
        // rebuilds the whole `Sim` anyway, so there is nothing to keep it for.
        let mut voxelizer = Voxelizer::new(gpu).context("creating the voxeliser")?;
        let stats = voxelizer
            .voxelize(FlatGeometry::from_scene(&scene), grid)
            .context("voxelising the scene")?;
        let voxel_report = stats.report();
        log::info!("{voxel_report}");
        scene.take_dirty();

        let mut mask = voxelizer
            .read_flags()
            .context("reading the flag field back")?;
        let mut links = voxelizer
            .read_links()
            .context("reading the boundary links back")?;
        anyhow::ensure!(
            mask.len() as u64 == grid.cell_count(),
            "the voxeliser returned {} flags for {} cells",
            mask.len(),
            grid.cell_count()
        );
        let filled = fill_obstructions(&mut mask, grid, &scene);
        if filled > 0 {
            log::info!("ray parity made {filled} more obstruction cells solid");
            // A link whose fluid end has just been filled no longer has one.
            links.retain(|l| flags::is_fluid(mask[l.cell as usize]));
        }

        // The plenum walls go in before the boundary conditions, not after: the
        // conditions are placed on whatever fluid is left on the domain faces,
        // and carving afterwards would solid-fill cells that had just been made
        // the inlet.
        let plan = match domain.plenum {
            Some(p) => {
                let report = p.carve(&mut mask, grid, &mouths, inlet, outlet);
                log::info!("{}", report.describe());
                if !report.connected {
                    log::warn!(
                        "the inlet plenum does not reach the outlet through the fluid: the mask \
                         was left alone, but this duct is either not watertight or not open"
                    );
                }
                BoundaryPlan {
                    inlet_plane_mm: report.inlet_plane_mm,
                    sponge_cells: params.sponge_cells,
                }
            }
            // The room domain, byte for byte as it was measured: its pressure
            // reference is 381 k equilibrium cells rather than the sponge, and
            // deepening the layer there would change the baseline an A/B is
            // supposed to be against.
            None => BoundaryPlan::default(),
        };
        if !vents.is_empty() && domain.plenum.is_some() {
            log::warn!(
                "vents in the plenum domain: the mouth extension is walled, so a vent outside \
                 the duct's box has nowhere to stand. Use the room domain."
            );
        }
        apply_boundaries(&mut mask, grid, &mouths, inlet, outlet, plan, &vents);

        let d_h_mm = mouths[inlet].hydraulic_diameter_mm().max(0.1) as f64;
        let units = lattice_units_for(params, &mouths, &inlets, outlet, &vents);
        let cfg = solver_config(params, &units, &mouths, inlet, outlet, &vents);
        for w in units.warnings() {
            log::warn!("{w}");
        }

        let solver = Solver::new(gpu, grid, &mask, &links, cfg).context("building the solver")?;

        let (flags_texture, flags_view) = flags_texture(&gpu.device, &gpu.queue, grid, &mask);

        Ok(Self {
            scene,
            vents,
            domain,
            grid,
            units,
            solver,
            mask,
            links,
            mouths,
            inlets,
            outlet,
            flags_texture,
            flags_view,
            d_h_mm,
            triangle_count,
            voxel_report,
        })
    }

    /// Hot-apply everything that does not need a rebuild.
    pub fn hot_apply(&mut self, params: &SimParams) -> Result<()> {
        self.units =
            lattice_units_for(params, &self.mouths, &self.inlets, self.outlet, &self.vents);
        let cfg = solver_config(
            params,
            &self.units,
            &self.mouths,
            self.inlets[0],
            self.outlet,
            &self.vents,
        );
        self.solver.update(cfg)
    }

    /// Scale factors the renderer's derive pass needs.
    pub fn derive_scales(&self) -> ad_render::DeriveScales {
        ad_render::DeriveScales::new(&self.units, self.d_h_mm)
    }

    /// Everything visible: the duct and any obstructions. What the camera
    /// frames.
    pub fn scene_bbox(&self) -> Bbox {
        self.scene.bbox()
    }

    /// The duct alone: the part under test. What the lattice, the install
    /// pose's pivot and every per-duct length are measured from.
    pub fn duct_bbox(&self) -> Bbox {
        self.scene.bbox_of_role(MeshRole::Duct)
    }
}

/// Make every cell inside an obstruction solid, by ray parity over the part of
/// the grid around it. Returns how many cells it changed.
///
/// The voxeliser's signed distance is right for one closed mesh and wrong in
/// two places an obstruction reaches. Where it crosses a domain face, the
/// exterior flood comes in through the cut and marks its inside as air. Where
/// it overlaps the duct wall, the nearest triangle to a cell inside it can be
/// the duct's, which votes "outside". Parity along +X counts only the
/// obstruction's own crossings, over the whole line, so it is right in both.
pub(crate) fn fill_obstructions(mask: &mut [u8], grid: Grid, scene: &Scene) -> u64 {
    let mut changed = 0;
    for (_, inst) in scene
        .visible()
        .filter(|(_, i)| i.role == MeshRole::Obstruction)
    {
        let b = inst.world_bbox();
        let Some((lo, hi)) = grid.cell_range(b) else {
            log::warn!(
                "obstruction {:?} is entirely outside the domain and has no effect",
                inst.name
            );
            continue;
        };
        let g = grid.bbox();
        if b.min.cmplt(g.min).any() || b.max.cmpgt(g.max).any() {
            log::warn!(
                "obstruction {:?} crosses the domain boundary; the part outside is left out",
                inst.name
            );
        }
        let dims = hi - lo + UVec3::ONE;
        let window = Grid {
            dims,
            dx_mm: grid.dx_mm,
            origin_mm: grid.cell_center_mm(lo),
        };
        let mesh = inst.world_mesh();
        let tris: Vec<[Vec3; 3]> = (0..mesh.triangle_count())
            .map(|t| mesh.triangle(t))
            .collect();
        let parity = ad_geom::ray_parity_voxelize(&tris, window);
        for (i, _) in parity.solid.iter().enumerate().filter(|(_, s)| **s) {
            let i = i as u32;
            let c = lo + UVec3::new(i % dims.x, (i / dims.x) % dims.y, i / (dims.x * dims.y));
            let cell = &mut mask[grid.linear(c) as usize];
            if *cell != flags::SOLID {
                *cell = flags::SOLID;
                changed += 1;
            }
        }
    }
    changed
}

/// The lattice units for this operating point, with the lattice velocity
/// lowered when the outlet would otherwise run too fast for the solver.
///
/// `u_lb` is the *inlet* speed in lattice units; the outlet runs faster by the
/// ratio of the air supplied to its area. One 2,095 mm^2 mouth into a
/// 1,120 mm^2 outlet is 1.9x — 0.05 becomes 0.09, which the coarse tier
/// survives. Two mouths sealed to vents feeding a 1,390 mm^2 outlet is 3.9x:
/// 0.19 at the outlet, a lattice Mach number of a third, and the field is NaN
/// within 3,000 steps (measured). So the lattice velocity is capped at
/// `0.1 / ratio`, which keeps the outlet at the literature's ceiling of 0.1.
/// The cost is proportionally more steps per second of air and a `tau0`
/// nearer 0.5; the LES model carries that.
fn lattice_units_for(
    params: &SimParams,
    mouths: &[Mouth],
    inlets: &[usize],
    outlet: usize,
    vents: &[VentPatch],
) -> LatticeUnits {
    let supply: f32 = if vents.is_empty() {
        inlets
            .first()
            .and_then(|i| mouths.get(*i))
            .map_or(0.0, |m| m.open_area_mm2)
    } else {
        vents
            .iter()
            .map(|v| 4.0 * v.half_u.length() * v.half_v.length() * v.speed_scale.max(0.0))
            .sum()
    };
    let exit = mouths.get(outlet).map_or(1.0, |m| m.open_area_mm2).max(1.0);
    let ratio = (supply / exit).max(1.0) as f64;
    let u_lb = params.u_lb.min(0.1 / ratio);
    if u_lb < params.u_lb {
        log::info!(
            "lattice velocity lowered from {} to {u_lb:.4}: the outlet takes {ratio:.1}x the supply speed",
            params.u_lb
        );
    }
    LatticeUnits::new(
        params.dx_mm as f64,
        params.inlet_velocity_ms as f64,
        u_lb,
        params.rho,
        params.nu,
    )
}

/// The largest mouth not in `taken`. A duct's main outlet is not a drain
/// hole, which is the same rule `UiState::outlet_mouth` applies to the view
/// models.
fn outlet_excluding(mouths: &[Mouth], taken: &[usize]) -> usize {
    mouths
        .iter()
        .enumerate()
        .filter(|(i, _)| !taken.contains(i))
        .max_by(|a, b| a.1.open_area_mm2.total_cmp(&b.1.open_area_mm2))
        .map(|(i, _)| i)
        .unwrap_or(usize::from(taken.first() == Some(&0)))
}

/// Whether a vent is sealed to a mouth: its centre on the mouth's plane, to
/// within a couple of cells, and inside the mouth's footprint.
fn vent_on_mouth(v: &VentPatch, m: &Mouth, dx_mm: f32) -> bool {
    let n = m.patch.normal.normalize_or_zero();
    let d = v.center_mm - m.patch.center_mm;
    if d.dot(n).abs() > 1.5 * dx_mm.max(0.1) {
        return false;
    }
    let half = m.patch.half_u.abs() + m.patch.half_v.abs() + Vec3::splat(dx_mm);
    (0..3).all(|a| n[a].abs() > 0.5 || d[a].abs() <= half[a])
}

/// Which mouths supply air (primary first) and which carries the outflow.
///
/// Without vents the chosen inlet mouth supplies the air. With vents, the
/// mouths they are sealed to do; a vent standing free supplies nothing at a
/// mouth, and the chosen inlet mouth stays the primary for the metrics. The
/// outlet is the user's choice if it is not supplying air, else the largest
/// mouth that is not.
pub(crate) fn mouth_roles(
    mouths: &[Mouth],
    vents: &[VentPatch],
    params: &SimParams,
    dx_mm: f32,
) -> (Vec<usize>, usize) {
    let primary = params.inlet_mouth.min(mouths.len().saturating_sub(1));
    let mut inlets = vec![primary];
    for (i, m) in mouths.iter().enumerate() {
        if i != primary && vents.iter().any(|v| vent_on_mouth(v, m, dx_mm)) {
            inlets.push(i);
        }
    }
    let outlet = match params.outlet_mouth {
        Some(o) if o < mouths.len() && !inlets.contains(&o) => o,
        _ => outlet_excluding(mouths, &inlets),
    };
    (inlets, outlet)
}

/// Vents the solver has inlet slots for: slots 1-3, slot 0 being the mouth.
pub const MAX_VENTS: usize = ad_gpu::flags::INLET_SLOTS - 1;

fn solver_config(
    params: &SimParams,
    units: &LatticeUnits,
    mouths: &[Mouth],
    inlet: usize,
    outlet: usize,
    vents: &[VentPatch],
) -> SolverConfig {
    let inlet_normal = mouths[inlet].patch.normal.normalize_or(Vec3::X);
    // `Mouth::patch.normal` points *into* the fluid. The solver's outlet normal
    // is the outward one, so it is negated: anti-bounce-back and the convective
    // term are applied only to links arriving from outside along it, and getting
    // the sign wrong turns the outlet into a second inlet.
    let outlet_outward = -mouths[outlet].patch.normal.normalize_or(Vec3::X);
    // The inlet lattice velocity, derived rather than taken from
    // `LatticeUnits::u_lb`. The two agree whenever the operating point is
    // non-zero — that is what `u_lb` means — but at `U = 0` they do not:
    // `LatticeUnits::new` substitutes a nominal 1 m/s so that `dt` stays finite,
    // which leaves `u_lb` at its configured 0.05 while `c_u` becomes `1/u_lb`.
    // Using `u_lb` there would drive the inlet at 1 m/s with the slider on zero,
    // and "stop the fan" would quietly mean "run it slowly".
    let c_u = units.c_u();
    let u_lb = if c_u.abs() > 1e-12 {
        (units.u_phys / c_u) as f32
    } else {
        0.0
    };

    // A vent blows at `speed_scale` times the operating point along its aim:
    // the speed is the speed, unlike the mouth inlet's louver, which holds the
    // flow rate. A vent that misses the mouth delivers less; that is the point
    // of simulating it. The same plane closure as the mouth inlet, on the
    // axis plane the cells were put on: a source of air with a duct behind it,
    // not a fan disc pushing room air (see `InletSpec::local_density`).
    let mut extra_inlets = [ad_solver::InletSpec::default(); 3];
    for (spec, v) in extra_inlets.iter_mut().zip(vents.iter().take(MAX_VENTS)) {
        let n = v.axis_normal();
        *spec = ad_solver::InletSpec {
            velocity: v.direction.normalize_or(n) * u_lb * v.speed_scale.max(0.0),
            normal: n,
            local_density: false,
        };
    }

    SolverConfig {
        precision: precision_from_env(),
        // Only for the tracer residence-time path, which needs the velocity
        // field on the CPU. It costs 16 bytes per cell of storage plus a staging
        // buffer of the same size — 691 MB at the interactive tier — so it stays
        // off unless `AERODUCT_RTD` asks for it. See `crate::tracers`.
        macroscopic_buffer: crate::tracers::rtd_enabled(),
        periodic: [false; 3],
        // Tilted by the louver aim, with the normal component held at `u_lb`
        // so the flow rate is the untilted one. The normal itself stays the
        // mouth's: the inlet's density closure needs the plane, not the jet.
        inlet_velocity: params.inlet_direction(inlet_normal, mouths[inlet].axis as usize) * u_lb,
        inlet_normal,
        extra_inlets,
        outlet_normal: outlet_outward,
        // Off by default, and deliberately: `SolverConfig::outlet_anti_bounce_back`
        // documents that it diverges under real through-flow.
        outlet_anti_bounce_back: false,
        outflow_velocity: u_lb.abs(),
        sponge_cells: params.sponge_cells,
        sponge_strength: 0.4,
        smagorinsky_c: params.smagorinsky_c,
        trt_lambda: params.trt_lambda,
        ..SolverConfig::from_units(*units)
    }
}

/// The domain and the lattice `params` would build around a scene.
///
/// Shared by the build and by the resolution control's pre-flight, so the grid
/// the control quotes before Apply is the grid the build then allocates — not a
/// second copy of the arithmetic, free to drift from the first.
pub(crate) fn plan_lattice(
    duct: Bbox,
    mouths: &[Mouth],
    inlet: usize,
    outlet: usize,
    params: &SimParams,
) -> (crate::domain::Domain, Grid) {
    // The duct's box, not the scene's. An obstruction parked in the exit jet
    // would be better measured past than through, but an obstruction is a car
    // part — a trim panel, a dashboard — and growing the room around one asks
    // for more cells than any card has. The room's margins already reach well
    // past the mouths; an obstruction that goes further is cut at the face,
    // and `fill_obstructions` keeps the cut part solid.
    let domain = DomainMargins::from_env()
        .with_explicit(params.domain_mm)
        .plan(
            duct,
            mouths,
            inlet,
            outlet,
            params.dx_mm,
            params.sponge_cells,
        );
    let grid = Grid::covering(domain.bbox, params.dx_mm);
    (domain, grid)
}

pub(crate) fn precision_from_env() -> DdfPrecision {
    match std::env::var("AERODUCT_PRECISION").as_deref() {
        Ok("fp16") | Ok("FP16") | Ok("fp16c") | Ok("FP16C") => DdfPrecision::Fp16c,
        // FP32 by default, even though FP16C halves the memory. The packing in
        // `ad_gpu::ddf` shares one u32 between two cells, so without the shader
        // f16 feature every store becomes a read-modify-write and the format
        // measures 4447 MLUPS against FP32's 6305 -- the "cheaper" option is
        // 0.7x the speed. Until that layout is fixed, FP16C buys capacity, not
        // performance, so it should be asked for rather than assumed.
        _ => DdfPrecision::Fp32,
    }
}

/// Where the boundary conditions go, beyond what the mouths already say.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BoundaryPlan {
    /// Coordinate, along the inlet mouth's own axis, of the plane the velocity
    /// inlet goes on.
    ///
    /// `None` puts it on the mouth itself, which is the room domain's
    /// arrangement. In the plenum domain it is the far end of the inlet
    /// extension, so that the prescribed plug profile has
    /// [`crate::plenum::DEFAULT_UPSTREAM_D_H`] hydraulic diameters of wall to
    /// grow a boundary layer against before the duct sees it — which is what
    /// CONTRACT.md asks for and what the mouth-plane inlet only approximated.
    pub inlet_plane_mm: Option<f32>,
    /// How many cell rows of absorbing layer to flag inward from the outflow
    /// plane. Zero leaves the sponge one cell thick.
    ///
    /// # Why this is not simply `params.sponge_cells` everywhere
    ///
    /// `ad_solver::PaddedDomain::sponge_sigma` takes the layer *thickness* from
    /// the solver config and the layer's *location* from the `SPONGE` flag, and
    /// until now only the outflow plane itself was ever flagged. So the shipped
    /// "20-cell sponge" was one cell of relaxation at strength 0.4, and the
    /// remaining 19 cells of grading existed only in the profile function.
    ///
    /// In the room domain that was survivable, because the sponge was never the
    /// pressure reference: 381,270 equilibrium cells on the five other faces
    /// were, and an equilibrium cell resets its populations outright. The plenum
    /// domain has *no* equilibrium cells — that is the point of it — so the
    /// outflow region is the only reference there is, and one plane of it is not
    /// enough. Measured: with a one-cell sponge the walled plenum domain runs
    /// away to a mean static pressure of ~340 Pa, far enough that the inlet's
    /// own density clamp binds and `Q_in` reads 5.04 L/s against the 6.3 the
    /// boundary is pushing. With the layer, it settles.
    ///
    /// The room path still passes zero, so its behaviour is unchanged and the
    /// A/B against it stays a comparison of domains rather than of two edits at
    /// once.
    pub sponge_cells: u32,
}

/// Turn a geometry-only flag field into a solvable boundary-value problem.
///
/// With `vents` the mouth is left an opening and the vents are the inlets,
/// each in its own solver slot.
pub fn apply_boundaries(
    mask: &mut [u8],
    grid: Grid,
    mouths: &[Mouth],
    inlet: usize,
    outlet: usize,
    plan: BoundaryPlan,
    vents: &[VentPatch],
) {
    let BoundaryPlan {
        inlet_plane_mm,
        sponge_cells,
    } = plan;
    let dims = grid.dims;
    let at = |c: UVec3| grid.linear(c) as usize;

    // 1. Every domain face is an open equilibrium boundary, so the exit jet can
    //    entrain surrounding air instead of being confined by a wall.
    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let on_face = x == 0
                    || y == 0
                    || z == 0
                    || x + 1 == dims.x
                    || y + 1 == dims.y
                    || z + 1 == dims.z;
                if !on_face {
                    continue;
                }
                let i = at(UVec3::new(x, y, z));
                if flags::is_fluid(mask[i]) {
                    mask[i] = flags::EQUILIBRIUM;
                }
            }
        }
    }

    // 2. The face the outlet mouth points at becomes the convective outflow,
    //    with the sponge that absorbs what the outflow condition reflects.
    //
    //    The outflow condition itself is one plane — there is only one place a
    //    domain can end — but the sponge behind it is `sponge_cells` deep, which
    //    is what makes it an absorbing *layer* rather than a single step change
    //    in damping. See `BoundaryPlan::sponge_cells` for the measurement that
    //    made the depth matter.
    let (axis, on_min) = (mouths[outlet].axis as usize % 3, mouths[outlet].on_min_side);
    let plane = if on_min { 0 } else { dims[axis] - 1 };
    let depth = sponge_cells.clamp(1, dims[axis]);
    for step in 0..depth {
        // Inward from the outflow plane, whichever face it is on.
        let row = if on_min { plane + step } else { plane - step };
        for a in 0..dims[(axis + 1) % 3] {
            for b in 0..dims[(axis + 2) % 3] {
                let mut c = UVec3::ZERO;
                c[axis] = row;
                c[(axis + 1) % 3] = a;
                c[(axis + 2) % 3] = b;
                let i = at(c);
                if !flags::is_fluid(mask[i]) {
                    continue;
                }
                if step == 0 {
                    mask[i] = flags::OUTLET | flags::SPONGE;
                } else {
                    // Only the sponge bit inland: a second plane of `OUTLET`
                    // would be a second convective outflow condition in the
                    // middle of the fluid, and `CellKind` would stop collision
                    // running there at all.
                    mask[i] |= flags::SPONGE;
                }
            }
        }
    }

    // 3. The inlet: a plane restricted to the mouth's footprint — the mouth's
    //    own in the room domain, the upstream end of its extension otherwise.
    //    Or, with vents in the room, those instead: the mouth stays open and
    //    takes whatever their jets bring it.
    if vents.is_empty() {
        mark_inlet(mask, grid, &mouths[inlet], inlet_plane_mm);
    } else {
        for (i, v) in vents.iter().take(MAX_VENTS).enumerate() {
            let marked = mark_vent(mask, grid, v, (i + 1) as u8);
            let area = 4.0 * v.half_u.length() * v.half_v.length();
            log::info!(
                "vent {}: {marked} cells at ({:.1}, {:.1}, {:.1}) mm facing {:?} ({area:.0} mm^2 nominal)",
                i + 1,
                v.center_mm.x,
                v.center_mm.y,
                v.center_mm.z,
                v.normal
            );
            if marked == 0 {
                log::warn!(
                    "vent {} lies outside the simulated box (or inside a solid) and blows nothing",
                    i + 1
                );
            }
        }
    }

    // What the three decisions above actually produced, counted.
    //
    // Worth a line because the *ratio* of these is the domain's pressure
    // reference, and a domain that cannot hold its mean pressure looks exactly
    // like a domain with a bad solver. An equilibrium cell resets its
    // populations outright, so the open faces are a hard reference in
    // proportion to their number; an outlet cell only relaxes partway. The room
    // domain has ~400 k equilibrium cells and the plenum domain has none, which
    // is the single biggest behavioural difference between them and is invisible
    // in any other log line.
    let (mut inlet_cells, mut outlet_cells, mut equil_cells, mut sponge, mut fluid) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    for f in mask.iter() {
        if !flags::is_fluid(*f) {
            continue;
        }
        fluid += 1;
        inlet_cells += u64::from(*f & flags::INLET != 0);
        outlet_cells += u64::from(*f & flags::OUTLET != 0);
        equil_cells += u64::from(*f & flags::EQUILIBRIUM != 0);
        sponge += u64::from(*f & flags::SPONGE != 0);
    }
    log::info!(
        "boundaries: {fluid} fluid cells, {inlet_cells} inlet, {outlet_cells} outlet, \
         {equil_cells} equilibrium, {sponge} sponge"
    );
}

/// Flag one cell's thickness of the fluid cells on a vent's rectangle as an
/// inlet in `slot`. Returns how many.
///
/// The cells go on the plane of cell centres nearest the vent's centre along
/// the lattice axis nearest its normal — the closure that gives the inlet its
/// density needs an axis-aligned plane. The rectangle is the vent's own,
/// projected onto that plane, so a vent turned away from the axes emits from
/// a face rotated by up to 45° while its air still leaves the way it was
/// aimed. Logged when the turn is more than a few degrees.
fn mark_vent(mask: &mut [u8], grid: Grid, v: &VentPatch, slot: u8) -> usize {
    let given = v.normal.normalize_or_zero();
    let (lu, lv) = (v.half_u.length(), v.half_v.length());
    if given == Vec3::ZERO || lu <= 0.0 || lv <= 0.0 || !grid.dx_mm.is_finite() {
        return 0;
    }
    let n = v.axis_normal();
    let turned = given.dot(n).clamp(-1.0, 1.0).acos().to_degrees();
    if turned > 5.0 {
        log::info!(
            "vent plane snapped to the {n:?} axis, {turned:.0} deg from the face it was given"
        );
    }
    // In-plane axes: the rectangle's own, flattened onto the axis plane.
    let flatten = |a: Vec3| (a - n * a.dot(n)).normalize_or_zero();
    let (mut u_hat, mut v_hat) = (flatten(v.half_u / lu), flatten(v.half_v / lv));
    if u_hat == Vec3::ZERO || v_hat == Vec3::ZERO {
        // The rectangle was edge-on to the axis plane; lay it out square to it.
        let axis = (0..3).find(|a| n[*a] != 0.0).unwrap_or(2);
        u_hat = Vec3::AXES[(axis + 1) % 3];
        v_hat = Vec3::AXES[(axis + 2) % 3];
    }
    let dx = grid.dx_mm;
    let mut c = v.center_mm;
    let axis = (0..3).find(|a| n[*a] != 0.0).unwrap_or(2);
    c[axis] = grid.origin_mm[axis] + ((c[axis] - grid.origin_mm[axis]) / dx).round() * dx;
    let thick = 0.5 * dx + 1.0e-4 * dx;
    // The cells whose centres lie on the rectangle, and no more: a vent's
    // size is stated, so unlike the mouth there is no outermost row to keep.
    let slack = 1.0e-4 * dx;
    let reach = u_hat.abs() * (lu + slack) + v_hat.abs() * (lv + slack) + Vec3::splat(thick + dx);
    let Some((lo, hi)) = grid.cell_range(Bbox {
        min: c - reach,
        max: c + reach,
    }) else {
        return 0;
    };
    let mut marked = 0;
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let cell = UVec3::new(x, y, z);
                let d = grid.cell_center_mm(cell) - c;
                if d.dot(n).abs() > thick
                    || d.dot(u_hat).abs() > lu + slack
                    || d.dot(v_hat).abs() > lv + slack
                {
                    continue;
                }
                let f = &mut mask[grid.linear(cell) as usize];
                // Not over the outflow plane and its sponge: a vent on the far
                // face would otherwise blow straight out of the box.
                if flags::is_fluid(*f) && *f & flags::OUTLET == 0 {
                    *f = flags::inlet_in_slot(slot);
                    marked += 1;
                }
            }
        }
    }
    marked
}

/// Flag the fluid cells lying in the mouth's plane and inside its opening.
///
/// The footprint test is the whole point. The mouth sits flush with the part's
/// bounding box, so its plane extends across the entire domain; flagging the
/// plane without the footprint test injects a sheet of moving air across the
/// whole box, which produces a picture that looks like a wind tunnel and numbers
/// that mean nothing.
///
/// # `half_u` is not necessarily along `(axis + 1) % 3`
///
/// `ad_geom::mouths::plane_axes` orders the in-plane basis so that `u x v` is
/// the box's *outward* normal, which means it **swaps** the pair on a min-side
/// face. Both mouths of the test part are on min-side faces, so `half_u` there
/// is the extent along `(axis + 2) % 3`, not `(axis + 1) % 3`.
///
/// Pairing `half_u` with the wrong axis transposes the footprint. On this part
/// that turned a 139 x 15 mm mouth into a 15.75 x 139.75 mm strip running out
/// through the free air beside the duct: 420 of the mouth's 3,714 cells were
/// driven and the rest of the sheet blew into the room, which measured as
/// `Q_in = 0.24 L/s` against a hand calculation of 6.35 and a 46% mass
/// imbalance. Taking the extents component-wise instead removes the assumption
/// entirely — the basis vectors are axis-aligned, so `|half_u| + |half_v|` is
/// the half-size on every axis at once whichever way round they came.
fn mark_inlet(mask: &mut [u8], grid: Grid, mouth: &Mouth, plane_mm: Option<f32>) {
    let axis = mouth.axis as usize % 3;
    let patch = &mouth.patch;
    let plane_w = plane_mm
        .filter(|v| v.is_finite())
        .unwrap_or(patch.center_mm[axis]);
    let k = ((plane_w - grid.origin_mm[axis]) / grid.dx_mm).round();
    if !k.is_finite() || k < 0.0 || k as u32 >= grid.dims[axis] {
        log::warn!("the inlet mouth plane at {plane_w} mm falls outside the grid");
        return;
    }
    let k = k as u32;

    let (u_axis, v_axis) = ((axis + 1) % 3, (axis + 2) % 3);
    // `half_u` / `half_v` are the mouth's bounding rectangle in its own plane,
    // as axis-aligned vectors in an order that depends on which face the mouth
    // was found on. A half-cell of slack keeps the outermost row of the opening:
    // without it a 15 mm slot at dx = 0.75 mm loses two of its twenty rows to
    // rounding.
    let slack = grid.dx_mm * 0.5;
    let half = patch.half_u.abs() + patch.half_v.abs() + Vec3::splat(slack);
    let centre = patch.center_mm;

    let mut marked = 0usize;
    for a in 0..grid.dims[u_axis] {
        for b in 0..grid.dims[v_axis] {
            let mut c = UVec3::ZERO;
            c[axis] = k;
            c[u_axis] = a;
            c[v_axis] = b;
            let p = grid.cell_center_mm(c);
            if (p[u_axis] - centre[u_axis]).abs() > half[u_axis]
                || (p[v_axis] - centre[v_axis]).abs() > half[v_axis]
            {
                continue;
            }
            let i = grid.linear(c) as usize;
            if flags::is_fluid(mask[i]) {
                mask[i] = flags::INLET;
                marked += 1;
            }
        }
    }
    log::info!(
        "inlet: {marked} cells on the plane at {plane_w:.2} mm ({:.0} mm^2 nominal open area)",
        mouth.open_area_mm2
    );
}

/// Upload the flag mask as an `R8Uint` 3D texture.
///
/// The renderer's derive pass masks walls out of its finite differences with
/// this: without it, every cell next to the duct wall differentiates against a
/// solid neighbour holding zero velocity, and the Q-criterion view grows a bright
/// shell exactly where the geometry is.
pub fn flags_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    grid: Grid,
    mask: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let extent = wgpu::Extent3d {
        width: grid.dims.x,
        height: grid.dims.y,
        depth_or_array_layers: grid.dims.z,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cell flags"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_flags(queue, &texture, grid, mask);
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });
    (texture, view)
}

fn write_flags(queue: &wgpu::Queue, texture: &wgpu::Texture, grid: Grid, mask: &[u8]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        mask,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(grid.dims.x),
            rows_per_image: Some(grid.dims.y),
        },
        wgpu::Extent3d {
            width: grid.dims.x,
            height: grid.dims.y,
            depth_or_array_layers: grid.dims.z,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::FlowPatch;

    fn test_grid() -> Grid {
        Grid {
            dims: UVec3::new(20, 12, 12),
            dx_mm: 1.0,
            origin_mm: Vec3::new(0.5, 0.5, 0.5),
        }
    }

    /// A mouth on the min-x face of a 6x6 bore centred in the domain.
    fn mouth(axis: u8, on_min_side: bool, centre: Vec3, half: (Vec3, Vec3), area: f32) -> Mouth {
        Mouth {
            patch: FlowPatch {
                center_mm: centre,
                normal: if on_min_side {
                    Vec3::AXES[axis as usize]
                } else {
                    -Vec3::AXES[axis as usize]
                },
                half_u: half.0,
                half_v: half.1,
            },
            open_area_mm2: area,
            axis,
            on_min_side,
            boundary: Vec::new(),
        }
    }

    #[test]
    fn an_obstruction_cut_by_the_domain_face_is_solid_not_a_shell() {
        let grid = Grid {
            dims: UVec3::splat(20),
            dx_mm: 1.0,
            origin_mm: Vec3::splat(0.5),
        };
        let mut mask = vec![flags::FLUID; grid.cell_count() as usize];
        let mut scene = Scene::new();
        // Centred on the -x face, so half of it is outside the grid.
        let ball = ad_geom::primitives::uv_sphere(Vec3::new(0.0, 10.0, 10.0), 6.0, 48, 24);
        scene.add(
            "ball",
            MeshAsset::new(ball),
            Transform::IDENTITY,
            MeshRole::Obstruction,
        );
        let filled = fill_obstructions(&mut mask, grid, &scene);
        let half_ball = 2.0 / 3.0 * std::f32::consts::PI * 6f32.powi(3);
        assert!(
            (filled as f32 - half_ball).abs() / half_ball < 0.1,
            "{filled} cells for a {half_ball:.0} mm^3 half-ball"
        );
        assert_eq!(
            mask[grid.linear(UVec3::new(0, 10, 10)) as usize],
            flags::SOLID,
            "solid at the cut"
        );
        assert_eq!(
            mask[grid.linear(UVec3::new(10, 10, 10)) as usize],
            flags::FLUID
        );
    }

    #[test]
    fn the_duct_is_left_to_the_voxeliser() {
        let grid = Grid {
            dims: UVec3::splat(20),
            dx_mm: 1.0,
            origin_mm: Vec3::splat(0.5),
        };
        let mut mask = vec![flags::FLUID; grid.cell_count() as usize];
        let mut scene = Scene::new();
        let block = ad_geom::primitives::box_mesh(Vec3::splat(4.0), Vec3::splat(12.0));
        scene.add(
            "duct",
            MeshAsset::new(block),
            Transform::IDENTITY,
            MeshRole::Duct,
        );
        assert_eq!(fill_obstructions(&mut mask, grid, &scene), 0);
    }

    fn two_x_mouths() -> [Mouth; 2] {
        let half = (Vec3::Y * 3.0, Vec3::Z);
        [
            mouth(0, true, Vec3::new(0.5, 6.0, 6.0), half, 12.0),
            mouth(0, false, Vec3::new(19.5, 6.0, 6.0), half, 12.0),
        ]
    }

    #[test]
    fn an_untilted_inlet_is_bit_for_bit_what_it_was() {
        let params = SimParams::default();
        let units = params.lattice_units();
        let cfg = solver_config(&params, &units, &two_x_mouths(), 0, 1, &[]);
        let u_lb = (units.u_phys / units.c_u()) as f32;
        assert_eq!(
            cfg.inlet_velocity.to_array().map(f32::to_bits),
            (Vec3::X * u_lb).to_array().map(f32::to_bits)
        );
    }

    #[test]
    fn a_tilted_inlet_turns_the_air_without_changing_the_flow() {
        let mut params = SimParams::default();
        params.inlet_tilt_deg = [30.0, -10.0];
        let units = params.lattice_units();
        let cfg = solver_config(&params, &units, &two_x_mouths(), 0, 1, &[]);
        let u_lb = (units.u_phys / units.c_u()) as f32;
        assert_eq!(
            cfg.inlet_normal,
            Vec3::X,
            "the density closure still needs the mouth's plane"
        );
        assert!((cfg.inlet_velocity.dot(cfg.inlet_normal) - u_lb).abs() < 1e-7);
        assert!(
            cfg.inlet_velocity.y > 0.0 && cfg.inlet_velocity.z < 0.0,
            "{}",
            cfg.inlet_velocity
        );
    }

    /// A **6 x 2** bore through a solid block that stops two thirds of the way
    /// along x, so the last third is open air. That is the real arrangement — a
    /// duct exhausting into a room — and it is what makes the open box sides
    /// meaningful: with the block filling the whole domain there is no free
    /// surface for them to be on.
    ///
    /// The bore is deliberately *not* square. A square one cannot tell a
    /// transposed footprint from a correct one, which is exactly how the real
    /// part shipped with 420 of its 3,714 inlet cells driven.
    const DUCT_END: u32 = 14;
    const BORE_Y: std::ops::Range<u32> = 3..9;
    const BORE_Z: std::ops::Range<u32> = 5..7;

    fn duct_mask(grid: Grid) -> Vec<u8> {
        let mut mask = vec![flags::FLUID; grid.cell_count() as usize];
        for z in 0..grid.dims.z {
            for y in 0..grid.dims.y {
                for x in 0..DUCT_END {
                    let inside = BORE_Y.contains(&y) && BORE_Z.contains(&z);
                    if !inside {
                        mask[grid.linear(UVec3::new(x, y, z)) as usize] = flags::SOLID;
                    }
                }
            }
        }
        mask
    }

    /// The mouth at one end of [`duct_mask`]'s bore, with its in-plane axes in
    /// the order `ad_geom::mouths::plane_axes` really produces.
    ///
    /// For a **min-side** face that order is swapped: `u x v` has to be the
    /// box's *outward* normal, so `half_u` is the extent along `(axis + 2) % 3`
    /// and `half_v` the one along `(axis + 1) % 3`. Both mouths of the contract's
    /// test part are on min-side faces, so a fixture that used the naive order
    /// would test a case that never occurs.
    fn bore_mouth(on_min_side: bool) -> Mouth {
        let x = if on_min_side { 0.5 } else { 19.5 };
        let (half_y, half_z) = (Vec3::Y * 3.0, Vec3::Z * 1.0);
        let (half_u, half_v) = if on_min_side {
            (half_z, half_y)
        } else {
            (half_y, half_z)
        };
        mouth(
            0,
            on_min_side,
            Vec3::new(x, 6.0, 6.0),
            (half_u, half_v),
            12.0,
        )
    }

    fn vent(center: Vec3, normal: Vec3, half_u: Vec3, half_v: Vec3) -> VentPatch {
        VentPatch {
            center_mm: center,
            normal,
            half_u,
            half_v,
            direction: normal,
            speed_scale: 1.0,
        }
    }

    /// With a vent, the mouth is an opening and the vent's cells carry its
    /// slot; an axis-aligned vent is one cell thick and covers its rectangle.
    #[test]
    fn the_lattice_velocity_yields_to_a_fast_outlet() {
        let params = SimParams::default();
        // The contract's part: one mouth into one about half its area. Fine.
        let a = mouth(2, true, Vec3::ZERO, (Vec3::X * 70.0, Vec3::Y * 7.5), 2095.0);
        let b = mouth(1, true, Vec3::ZERO, (Vec3::X * 37.0, Vec3::Z * 7.5), 1120.0);
        let u = lattice_units_for(&params, &[a.clone(), b.clone()], &[0], 1, &[]);
        assert_eq!(u.u_lb, params.u_lb, "1.9x at the outlet leaves 0.05 alone");
        // Two sealed vents into a small outlet: nearly four times the speed.
        let c = mouth(1, true, Vec3::ZERO, (Vec3::X * 39.0, Vec3::Z * 9.0), 1390.0);
        let va = vent(Vec3::ZERO, Vec3::Z, Vec3::X * 79.0, Vec3::Y * 9.0);
        let vb = vent(Vec3::ZERO, Vec3::Z, Vec3::X * 71.5, Vec3::Y * 9.0);
        let u = lattice_units_for(&params, &[a, b, c], &[0, 1], 2, &[va, vb]);
        let ratio = (4.0 * 79.0 * 9.0 + 4.0 * 71.5 * 9.0) / 1390.0;
        assert!(
            (u.u_lb - 0.1 / ratio as f64).abs() < 1e-9,
            "u_lb {} for a {ratio:.2}x outlet",
            u.u_lb
        );
        assert!(u.u_lb < 0.03);
    }

    #[test]
    fn vents_sealed_to_mouths_make_them_inlets_and_free_the_outlet() {
        // Three mouths: A (largest, +x face), B, C. Default roles: A in, B out.
        let big = mouth(
            0,
            false,
            Vec3::new(19.5, 6.0, 6.0),
            (Vec3::Y * 4.0, Vec3::Z * 2.0),
            32.0,
        );
        let mid = mouth(
            0,
            true,
            Vec3::new(0.5, 6.0, 6.0),
            (Vec3::Y * 3.0, Vec3::Z * 1.0),
            12.0,
        );
        let small = mouth(
            1,
            true,
            Vec3::new(10.0, 0.5, 6.0),
            (Vec3::X * 2.0, Vec3::Z * 1.0),
            8.0,
        );
        let mouths = [big, mid, small];
        let params = SimParams::default();
        assert_eq!(mouth_roles(&mouths, &[], &params, 1.0), (vec![0], 1));

        // A vent sealed on B: B supplies air too, so the outlet moves to C.
        let on_b = vent(
            Vec3::new(0.5, 6.0, 6.0),
            Vec3::X,
            Vec3::Y * 3.0,
            Vec3::Z * 1.0,
        );
        assert_eq!(mouth_roles(&mouths, &[on_b], &params, 1.0), (vec![0, 1], 2));
        // A vent standing off the mouth is a free jet, not a mouth inlet.
        let free = vent(
            Vec3::new(-5.0, 6.0, 6.0),
            Vec3::X,
            Vec3::Y * 3.0,
            Vec3::Z * 1.0,
        );
        assert_eq!(mouth_roles(&mouths, &[free], &params, 1.0), (vec![0], 1));
        // The user's outlet choice holds unless that mouth is supplying air.
        let chosen = SimParams {
            outlet_mouth: Some(2),
            ..params
        };
        assert_eq!(mouth_roles(&mouths, &[], &chosen, 1.0), (vec![0], 2));
        let taken = SimParams {
            outlet_mouth: Some(1),
            ..params
        };
        assert_eq!(mouth_roles(&mouths, &[on_b], &taken, 1.0), (vec![0, 1], 2));
    }

    #[test]
    fn a_vent_replaces_the_mouth_inlet_and_is_one_cell_thick() {
        let grid = test_grid();
        let mut mask = duct_mask(grid);
        // Facing +x, 4 mm along y by 2 mm along z, standing in the open air
        // past the duct exit at x = 16.5.
        let v = vent(
            Vec3::new(16.5, 6.0, 6.0),
            Vec3::X,
            Vec3::Y * 2.0,
            Vec3::Z * 1.0,
        );
        apply_boundaries(
            &mut mask,
            grid,
            &[bore_mouth(true), bore_mouth(false)],
            0,
            1,
            BoundaryPlan::default(),
            &[v],
        );
        let inlets: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, f)| **f & flags::INLET != 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            inlets.len(),
            4 * 2,
            "{} cells for a 4 x 2 mm vent at 1 mm",
            inlets.len()
        );
        for i in &inlets {
            assert_eq!(i % grid.dims.x as usize, 16, "a vent cell is off its plane");
            assert_eq!(flags::inlet_slot(mask[*i]), 1, "vent 1 drives slot 1");
        }
        // The mouth is an opening now: its cell on the x = 0 face is the open
        // box side every face gets, not an inlet.
        assert_eq!(
            mask[grid.linear(UVec3::new(0, 6, 6)) as usize],
            flags::EQUILIBRIUM
        );
    }

    /// A vent turned away from the axes goes on the nearest axis plane, its
    /// rectangle flattened onto it, one cell thick.
    #[test]
    fn a_turned_vent_lands_on_the_nearest_axis_plane() {
        let grid = Grid {
            dims: UVec3::splat(24),
            dx_mm: 1.0,
            origin_mm: Vec3::splat(0.5),
        };
        let mut mask = vec![flags::FLUID; grid.cell_count() as usize];
        // 30 degrees off +X toward +Y.
        let n = Vec3::new(30f32.to_radians().cos(), 30f32.to_radians().sin(), 0.0);
        let u = Vec3::new(-n.y, n.x, 0.0);
        let v = vent(Vec3::splat(12.0), n, u * 4.0, Vec3::Z * 3.0);
        assert_eq!(v.axis_normal(), Vec3::X);
        let marked = mark_vent(&mut mask, grid, &v, 2);
        let cells: Vec<UVec3> = (0..grid.cell_count() as u32)
            .filter(|i| mask[*i as usize] & flags::INLET != 0)
            .map(|i| UVec3::new(i % 24, (i / 24) % 24, i / 576))
            .collect();
        assert_eq!(cells.len(), marked);
        assert!(
            cells.iter().all(|c| c.x == 12),
            "not all on the x = 12.5 plane: {cells:?}"
        );
        // 8 mm flattened onto the plane's y axis covers 8 cells, 6 mm along z 6.
        assert_eq!(marked, 8 * 6, "{marked} cells");
        assert!(cells
            .iter()
            .all(|c| flags::inlet_slot(mask[grid.linear(*c) as usize]) == 2));
    }

    #[test]
    fn vents_become_the_solver_s_extra_inlets() {
        let params = SimParams::default();
        let units = params.lattice_units();
        let mut v = vent(Vec3::ZERO, Vec3::Z, Vec3::X * 10.0, Vec3::Y * 5.0);
        v.direction = Vec3::new(0.0, 1.0, 1.0).normalize();
        v.speed_scale = 0.5;
        let cfg = solver_config(&params, &units, &two_x_mouths(), 0, 1, &[v]);
        let u_lb = (units.u_phys / units.c_u()) as f32;
        let s = cfg.extra_inlets[0];
        assert!(
            !s.local_density,
            "a vent is a source, with the plane closure"
        );
        assert_eq!(s.normal, Vec3::Z);
        assert!(
            (s.velocity.length() - 0.5 * u_lb).abs() < 1e-7,
            "the speed is the speed"
        );
        assert!(s.velocity.y > 0.0 && s.velocity.z > 0.0);
        assert_eq!(
            cfg.extra_inlets[1],
            ad_solver::InletSpec::default(),
            "unused slots stay quiet"
        );
    }

    #[test]
    fn the_inlet_lands_on_the_bore_and_nowhere_else() {
        // Two failures this guards. The first: flagging the whole plane, which
        // injects a sheet of moving air across the entire domain rather than
        // into the duct. The second, subtler one: pairing `half_u` with the
        // wrong in-plane axis, which flags a *transposed* rectangle — mostly the
        // free air beside the duct, with only the overlap of the two rectangles
        // actually inside the bore.
        let grid = test_grid();
        let mut mask = duct_mask(grid);
        apply_boundaries(
            &mut mask,
            grid,
            &[bore_mouth(true), bore_mouth(false)],
            0,
            1,
            BoundaryPlan::default(),
            &[],
        );

        let inlet_cells = mask.iter().filter(|f| **f & flags::INLET != 0).count();
        let bore = (BORE_Y.len() * BORE_Z.len()) as usize;
        assert_eq!(
            inlet_cells, bore,
            "{inlet_cells} inlet cells against a {bore}-cell bore: the footprint is wrong \
             (a transposed one marks 4 of the bore's cells and a strip of open air)"
        );
        // Every inlet cell must be on the x = 0 plane.
        for (i, f) in mask.iter().enumerate() {
            if *f & flags::INLET != 0 {
                assert_eq!(
                    i % grid.dims.x as usize,
                    0,
                    "an inlet cell escaped the plane"
                );
            }
        }
    }

    /// A mouth on a max-side face keeps the un-swapped axis order, and the same
    /// footprint test has to handle both without knowing which it was given.
    #[test]
    fn the_footprint_survives_either_in_plane_axis_order() {
        let grid = test_grid();
        let mut swapped = duct_mask(grid);
        let mut plain = duct_mask(grid);
        // Same rectangle, described both ways round.
        let m = |half_u, half_v| mouth(0, true, Vec3::new(0.5, 6.0, 6.0), (half_u, half_v), 12.0);
        mark_inlet(&mut swapped, grid, &m(Vec3::Z * 1.0, Vec3::Y * 3.0), None);
        mark_inlet(&mut plain, grid, &m(Vec3::Y * 3.0, Vec3::Z * 1.0), None);
        assert_eq!(
            swapped.iter().filter(|f| **f & flags::INLET != 0).count(),
            plain.iter().filter(|f| **f & flags::INLET != 0).count(),
        );
        assert_eq!(swapped, plain, "the two orderings must mark the same cells");
    }

    #[test]
    fn the_outlet_face_is_the_one_the_outlet_mouth_points_at() {
        let grid = test_grid();
        let mut mask = duct_mask(grid);
        let (a, b) = (bore_mouth(true), bore_mouth(false));
        apply_boundaries(&mut mask, grid, &[a, b], 0, 1, BoundaryPlan::default(), &[]);

        let last = grid.dims.x - 1;
        let outlet_cells: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, f)| **f & flags::OUTLET != 0)
            .map(|(i, _)| i)
            .collect();
        assert!(!outlet_cells.is_empty(), "no outlet cells");
        for i in outlet_cells {
            assert_eq!(
                i % grid.dims.x as usize,
                last as usize,
                "an outlet cell is not on the far face"
            );
        }
        // ...and the sponge travels with it, one plane deep by default.
        let sponge = |m: &[u8]| m.iter().filter(|f| **f & flags::SPONGE != 0).count();
        let outlet = mask.iter().filter(|f| **f & flags::OUTLET != 0).count();
        assert_eq!(
            sponge(&mask),
            outlet,
            "the default plan is supposed to leave the room domain's one-cell sponge alone"
        );

        // Asked for a layer, it builds a layer — and only the outflow plane
        // itself keeps the `OUTLET` flag, because a second convective outflow
        // condition inland would stop collision running there at all.
        let mut deep = duct_mask(grid);
        apply_boundaries(
            &mut deep,
            grid,
            &[bore_mouth(true), bore_mouth(false)],
            0,
            1,
            BoundaryPlan {
                inlet_plane_mm: None,
                sponge_cells: 5,
            },
            &[],
        );
        assert_eq!(
            sponge(&deep),
            5 * outlet,
            "the sponge layer is not five rows deep"
        );
        assert_eq!(
            deep.iter().filter(|f| **f & flags::OUTLET != 0).count(),
            outlet,
            "the outflow condition spread inland with the sponge"
        );
    }

    #[test]
    fn the_box_sides_are_open_and_the_solid_is_untouched() {
        let grid = test_grid();
        let mut mask = duct_mask(grid);
        let (a, b) = (bore_mouth(true), bore_mouth(false));
        let solid_before = mask.iter().filter(|f| **f & flags::SOLID != 0).count();
        apply_boundaries(&mut mask, grid, &[a, b], 0, 1, BoundaryPlan::default(), &[]);
        let solid_after = mask.iter().filter(|f| **f & flags::SOLID != 0).count();
        assert_eq!(
            solid_before, solid_after,
            "a boundary flag overwrote solid geometry"
        );

        // Along the duct the y = 0 face is solid wall and must stay solid: an
        // equilibrium flag on a wall cell would open a hole through the side of
        // the part.
        for z in 0..grid.dims.z {
            for x in 0..DUCT_END {
                let i = grid.linear(UVec3::new(x, 0, z)) as usize;
                assert_eq!(mask[i], flags::SOLID, "the duct wall was opened at x = {x}");
            }
        }
        // Past the duct exit the same face is free air, and there it must be an
        // open equilibrium boundary so the jet can entrain.
        for z in 0..grid.dims.z {
            for x in DUCT_END..grid.dims.x - 1 {
                let i = grid.linear(UVec3::new(x, 0, z)) as usize;
                assert_eq!(
                    mask[i],
                    flags::EQUILIBRIUM,
                    "the box side at x = {x} is closed; the exit jet cannot entrain"
                );
            }
        }
    }

    /// Zero on the slider must mean zero at the inlet.
    ///
    /// `LatticeUnits::new` substitutes a nominal 1 m/s at `U = 0` so that `dt`
    /// stays finite, which leaves `u_lb` at its configured value while `c_u`
    /// grows to compensate. Feeding `u_lb` straight to the boundary there drives
    /// a 1 m/s inlet with the fan nominally off — a duct quietly passing 2 L/s
    /// that the user believes is stopped.
    #[test]
    fn a_zero_inlet_velocity_drives_nothing() {
        let mouths = [bore_mouth(true), bore_mouth(false)];

        let stopped = SimParams {
            inlet_velocity_ms: 0.0,
            ..SimParams::default()
        };
        let cfg = solver_config(&stopped, &stopped.lattice_units(), &mouths, 0, 1, &[]);
        assert_eq!(
            cfg.inlet_velocity,
            Vec3::ZERO,
            "a stopped fan must inject nothing"
        );
        assert_eq!(
            cfg.outflow_velocity, 0.0,
            "and the convective outflow has nothing to convect"
        );

        // ...while a real operating point still lands exactly on `u_lb`, which is
        // what makes `c_u` the conversion the metrics layer relies on.
        let running = SimParams {
            inlet_velocity_ms: 3.0,
            ..SimParams::default()
        };
        let units = running.lattice_units();
        let cfg = solver_config(&running, &units, &mouths, 0, 1, &[]);
        assert!(
            (cfg.inlet_velocity.length() as f64 - units.u_lb).abs() < 1e-9,
            "inlet lattice velocity {} vs u_lb {}",
            cfg.inlet_velocity.length(),
            units.u_lb
        );
        assert!((cfg.inlet_velocity.length() as f64 * units.c_u() - 3.0).abs() < 1e-6);
        assert_eq!(cfg.inlet_normal, Vec3::X);
    }

    #[test]
    fn the_outlet_is_the_largest_of_the_remaining_mouths() {
        let m = |area: f32| mouth(0, true, Vec3::ZERO, (Vec3::Y, Vec3::Z), area);
        let mouths = [m(2116.0), m(30.0), m(1141.0)];
        assert_eq!(
            outlet_excluding(&mouths, &[0]),
            2,
            "a drain hole is not the outlet"
        );
        assert_eq!(outlet_excluding(&mouths, &[2]), 0);
    }
}

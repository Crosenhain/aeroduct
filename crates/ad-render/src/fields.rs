//! Derived fields: the layer that decouples the solver from everything visual.
//!
//! The solver produces distribution functions. Nothing downstream of this module
//! ever touches them. Instead, one compute pass reduces the macroscopic
//! velocity and density into a small set of **sampled 3D textures**, and every
//! later pass — raymarch, slices, isosurfaces, particles — reads only those.
//!
//! That indirection is what makes "switch the displayed field from velocity to
//! Q-criterion" instantaneous: all four scalars live in the four channels of one
//! `Rgba16Float` texture, so switching is a one-word uniform change, not a
//! recompute. It is also what lets the render rate float free of the sim rate:
//! the derive pass runs when the sim has stepped, the raymarch runs when the
//! camera moved, and neither waits for the other.
//!
//! # The four scalars
//!
//! | channel | quantity | units |
//! |---|---|---|
//! | 0 | speed `\|u\|` | m/s |
//! | 1 | **normalised** Q-criterion `Q~` | dimensionless |
//! | 2 | normalised vorticity magnitude | dimensionless |
//! | 3 | pressure | Pa |
//!
//! ## Why Q is normalised
//!
//! The number-one usability failure of Q-criterion visualisation is that the
//! isolevel has units of `1/s^2`, so it needs re-tuning every single time the
//! inlet velocity changes. Users conclude the feature is broken. Storing
//!
//! ```text
//! Q~ = Q * (D_h / U_ref)^2
//! ```
//!
//! makes the isolevel dimensionless, and a value that worked at 2 m/s still
//! works at 8 m/s. The useful band is roughly `0.1 .. 2`, which is why that is
//! the default range in [`crate::transfer::TransferFunction::preset`].
//!
//! In lattice units the conversion collapses to a single scalar:
//! `Q~ = Q_lattice * (D_h_in_cells / u_lb)^2`. See [`DeriveScales`].
//!
//! ## Why the gradients are masked
//!
//! `J_ij = du_i/dx_j` is a central difference, **except** that a stencil arm
//! reaching into a solid cell is dropped and the difference becomes one-sided.
//! Differencing across a wall manufactures an enormous fake shear layer inside
//! the solid, which then shows up as a bright shell of spurious Q-criterion
//! wrapped around the entire duct — the single most convincing-looking wrong
//! result this renderer could produce.

use ad_gpu::{flags, Grid, LatticeUnits, Profiler, ShaderDefines, ShaderLoader};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, UVec3, Vec3};

use crate::util;

/// Which scalar the colour-mapped passes display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DerivedField {
    #[default]
    Speed,
    /// Normalised Q-criterion. Positive where rotation dominates strain.
    QCriterion,
    Vorticity,
    Pressure,
}

impl DerivedField {
    pub const ALL: [DerivedField; 4] = [
        DerivedField::Speed,
        DerivedField::QCriterion,
        DerivedField::Vorticity,
        DerivedField::Pressure,
    ];

    /// Channel of the scalar texture. This *is* the switch-the-field mechanism.
    pub fn channel(self) -> u32 {
        match self {
            DerivedField::Speed => 0,
            DerivedField::QCriterion => 1,
            DerivedField::Vorticity => 2,
            DerivedField::Pressure => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DerivedField::Speed => "speed",
            DerivedField::QCriterion => "Q-criterion",
            DerivedField::Vorticity => "vorticity",
            DerivedField::Pressure => "pressure",
        }
    }

    pub fn unit(self) -> &'static str {
        match self {
            DerivedField::Speed => "m/s",
            DerivedField::QCriterion | DerivedField::Vorticity => "-",
            DerivedField::Pressure => "Pa",
        }
    }

    /// True for quantities that take both signs, which is what drives the
    /// symmetric-range lock and the choice of a diverging colour map.
    pub fn is_signed(self) -> bool {
        matches!(self, DerivedField::Pressure | DerivedField::QCriterion)
    }
}

/// Whether derived fields are computed at the solver's resolution or at half.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldResolution {
    #[default]
    Full,
    /// Box-filtered 2x2x2. Eight times less memory and eight times less
    /// bandwidth for every downstream pass, and through a transfer function it
    /// is invisible: the transfer function is a low-pass filter already, and the
    /// raymarch step is comparable to a voxel either way.
    ///
    /// Note that the *gradients* are still evaluated at full resolution and then
    /// averaged, not evaluated on the coarse lattice. Coarsening first would
    /// halve the gradient accuracy, which matters most exactly where Q-criterion
    /// matters — in thin shear layers.
    Half,
}

impl FieldResolution {
    pub fn factor(self) -> u32 {
        match self {
            FieldResolution::Full => 1,
            FieldResolution::Half => 2,
        }
    }
}

// -- unit conversion ---------------------------------------------------------

/// Scale factors turning lattice-unit quantities into the stored units.
///
/// Derived once on the CPU per parameter change and handed to the shader as
/// four floats, so the inner loop multiplies rather than converting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeriveScales {
    /// Lattice velocity -> m/s. This is `c_u = dx/dt`.
    pub speed_ms: f32,
    /// Lattice Q -> dimensionless `Q~`. Equals `(D_h_cells / u_lb)^2`.
    pub q_tilde: f32,
    /// Lattice vorticity -> dimensionless. Equals `D_h_cells / u_lb`.
    pub vorticity: f32,
    /// `rho_lb - 1` -> Pa.
    pub pressure_pa: f32,
}

impl DeriveScales {
    /// `d_h_mm` is the hydraulic diameter used as the normalising length. For
    /// the test duct that is the ~6.3 mm median passage width, not the 145 mm
    /// overall size: `Q~` should measure vortices against the *passage*.
    pub fn new(units: &LatticeUnits, d_h_mm: f64) -> Self {
        let c_u = units.c_u();
        let dx_m = units.dx_m;
        let d_h_cells = (d_h_mm * 1e-3) / dx_m;
        // Both of these follow from J_phys = J_lattice * c_u / dx, then
        // multiplying by the requested power of (D_h / U_ref). The c_u and the
        // U_ref cancel into 1/u_lb, which is why neither appears below.
        let g = d_h_cells / units.u_lb.max(1e-9);
        Self {
            speed_ms: c_u as f32,
            q_tilde: (g * g) as f32,
            vorticity: g as f32,
            pressure_pa: units.pressure_pa(1.0) as f32,
        }
    }
}

/// `Q = 0.5 * (|Omega|_F^2 - |S|_F^2)` for a velocity gradient tensor.
///
/// `j` holds `J_ij = du_i/dx_j`, i.e. **column `j` is the derivative with
/// respect to axis `j`**, which is `glam`'s column-major convention and the
/// same as the WGSL `mat3x3` built in `derive.wgsl`.
///
/// Present in Rust as well as WGSL so the algebra can be tested against known
/// flows without a GPU. The two implementations must stay in step; the tests
/// below pin the properties that would break first.
pub fn q_criterion(j: Mat3) -> f32 {
    let jt = j.transpose();
    let s = (j + jt) * 0.5;
    let o = (j - jt) * 0.5;
    0.5 * (frobenius_sq(o) - frobenius_sq(s))
}

/// `omega = curl u` from the same gradient tensor.
pub fn vorticity(j: Mat3) -> Vec3 {
    Vec3::new(
        j.y_axis.z - j.z_axis.y,
        j.z_axis.x - j.x_axis.z,
        j.x_axis.y - j.y_axis.x,
    )
}

fn frobenius_sq(m: Mat3) -> f32 {
    m.x_axis.length_squared() + m.y_axis.length_squared() + m.z_axis.length_squared()
}

// -- GPU-side contract -------------------------------------------------------

/// Format the solver must write its macroscopic field into.
///
/// `xyz` = velocity in **lattice units**, `w` = `rho_lb - 1`. Lattice units
/// rather than SI because the solver already has them and the conversion is one
/// multiply that the derive pass has to do anyway.
pub const MACRO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Format of the per-cell flag texture: one [`ad_gpu::flags`] bitfield per cell.
///
/// `R8Uint` is not a *storage*-capable format in the portable WebGPU feature
/// set, so the solver cannot write it from a compute shader directly. Fill it
/// with `copy_buffer_to_texture` from the flag buffer it already has — it only
/// changes when the geometry changes, so the copy is not on the hot path.
pub const FLAGS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Uint;

/// Format of the derived scalar and velocity textures.
///
/// `Rgba16Float` and not `Rg16Float`: only the RGBA 16-bit float format is
/// guaranteed to support a `STORAGE_BINDING` in the base WebGPU feature set.
/// `Rg16Float` is renderable and samplable but *not* storage-capable without an
/// optional feature, and discovering that at pipeline creation on a user's
/// machine is not a good way to find out.
pub const DERIVED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// What the derive pass reads.
pub struct FieldSources<'a> {
    /// 3D [`MACRO_FORMAT`] texture at the solver grid resolution.
    pub macro_view: &'a wgpu::TextureView,
    /// 3D [`FLAGS_FORMAT`] texture, same dimensions. `None` means "treat every
    /// cell as fluid", which is correct for a bare-box test and degrades the
    /// wall masking to plain central differences.
    pub flags_view: Option<&'a wgpu::TextureView>,
    pub scales: DeriveScales,
}

/// Convenience allocation of a correctly-shaped source texture pair. The solver
/// is welcome to create its own; this exists so tests, and the app before the
/// solver is wired up, have something valid to point at.
pub fn create_source_textures(device: &wgpu::Device, grid: Grid) -> (wgpu::Texture, wgpu::Texture) {
    let size = wgpu::Extent3d {
        width: grid.dims.x.max(1),
        height: grid.dims.y.max(1),
        depth_or_array_layers: grid.dims.z.max(1),
    };
    let mac = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("solver macroscopic field"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: MACRO_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let flg = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("solver cell flags"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: FLAGS_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    (mac, flg)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DeriveUniform {
    src_dims: [u32; 3],
    downsample: u32,
    dst_dims: [u32; 3],
    flags_present: u32,
    speed_scale: f32,
    q_scale: f32,
    vort_scale: f32,
    pressure_scale: f32,
}

/// Uniform describing the derived textures to every downstream pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FieldsUniform {
    pub dims: [u32; 3],
    /// Channel of the scalar texture currently being displayed.
    pub channel: u32,
    /// World-space corner of the volume (voxel-0 centre minus half a voxel), mm.
    pub volume_min_mm: [f32; 3],
    /// Derived voxel size, mm. This is the reference step `h_ref` the opacity
    /// correction is defined against.
    pub voxel_mm: f32,
    /// World-space size of the whole volume, mm.
    pub volume_size_mm: [f32; 3],
    /// Length of the volume diagonal, mm. Precomputed because the raymarcher
    /// wants it for its step budget.
    pub diagonal_mm: f32,
    /// Where the lattice sits in the world (the duct's install pose), and the
    /// way back. Everything above is in the lattice frame; the camera is in the
    /// world frame. Rigid, so a ray's parameter is the same distance in both,
    /// which lets the raymarcher march in lattice space against a world-space
    /// depth buffer.
    pub lattice_to_world: [f32; 16],
    pub world_to_lattice: [f32; 16],
}

/// The derived-field textures plus the compute pass that fills them.
///
/// # Two views, two bind groups
///
/// Each texture is created once and viewed twice: a **storage** view that the
/// derive pass writes through, and a **sampled** view plus a linear sampler that
/// everything downstream reads through. This is not an optimisation, it is a
/// requirement — WGSL storage textures are write-only in the portable spec, so a
/// single binding cannot do both jobs.
pub struct DerivedFields {
    grid: Grid,
    resolution: FieldResolution,
    dims: UVec3,
    voxel_mm: f32,
    volume_min_mm: Vec3,
    /// Lattice-to-world placement; see [`Self::set_placement`].
    placement: Mat4,

    scalars: wgpu::Texture,
    velocity: wgpu::Texture,
    scalars_storage: wgpu::TextureView,
    velocity_storage: wgpu::TextureView,
    scalars_sampled: wgpu::TextureView,
    velocity_sampled: wgpu::TextureView,
    sampler: wgpu::Sampler,

    derive_uniform: wgpu::Buffer,
    fields_uniform: wgpu::Buffer,
    derive_layout: wgpu::BindGroupLayout,
    read_layout: wgpu::BindGroupLayout,
    read_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,

    /// 1x1x1 all-fluid stand-in, bound when the caller has no flag texture.
    fallback_flags: wgpu::TextureView,

    field: DerivedField,
    /// Bumped every time [`DerivedFields::derive`] records a pass, so the
    /// accelerator knows its brick min/max is stale.
    generation: u64,
}

impl DerivedFields {
    /// Workgroup shape of the derive kernel. 4x4x4 keeps the 3x3x3-ish gradient
    /// stencils of neighbouring threads inside the same cache lines without
    /// needing explicit shared-memory staging.
    const WG: UVec3 = UVec3::new(4, 4, 4);

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        loader: &ShaderLoader,
        grid: Grid,
        resolution: FieldResolution,
    ) -> Result<Self> {
        let f = resolution.factor();
        let dims = UVec3::new(
            grid.dims.x.div_ceil(f).max(1),
            grid.dims.y.div_ceil(f).max(1),
            grid.dims.z.div_ceil(f).max(1),
        );
        let voxel_mm = grid.dx_mm * f as f32;
        // Centre of derived voxel 0 sits at the centroid of the source cells it
        // covers, which is half a source cell in from the source origin per
        // extra level of coarsening.
        let derived_origin = grid.origin_mm + Vec3::splat(grid.dx_mm * (f as f32 - 1.0) * 0.5);
        let volume_min_mm = derived_origin - Vec3::splat(voxel_mm * 0.5);

        let make = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: dims.x,
                    height: dims.y,
                    depth_or_array_layers: dims.z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: DERIVED_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let scalars = make("derived scalars");
        let velocity = make("derived velocity");

        let view = |t: &wgpu::Texture, label: &str| {
            t.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            })
        };
        let scalars_storage = view(&scalars, "derived scalars (storage)");
        let scalars_sampled = view(&scalars, "derived scalars (sampled)");
        let velocity_storage = view(&velocity, "derived velocity (storage)");
        let velocity_sampled = view(&velocity, "derived velocity (sampled)");

        let sampler = util::linear_clamp_sampler(device, "derived field sampler");

        let fallback = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("all-fluid flag stand-in"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: FLAGS_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // The shader clamps its flag lookups into range, so a 1x1x1 texture
        // holding FLUID makes every cell fluid deterministically, rather than
        // relying on out-of-bounds `textureLoad` returning zero (which WGSL
        // leaves implementation-defined).
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[flags::FLUID],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let fallback_flags = fallback.create_view(&wgpu::TextureViewDescriptor {
            label: Some("all-fluid flag stand-in"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let derive_uniform = util::uniform_buffer::<DeriveUniform>(device, "derive uniform");
        let fields_uniform = util::uniform_buffer::<FieldsUniform>(device, "fields uniform");

        let cs = wgpu::ShaderStages::COMPUTE;
        let derive_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("derive"),
            entries: &[
                util::uniform_entry(0, cs),
                util::sampled_float_entry(1, cs, wgpu::TextureViewDimension::D3),
                util::texture_entry(
                    2,
                    cs,
                    wgpu::TextureSampleType::Uint,
                    wgpu::TextureViewDimension::D3,
                ),
                util::storage_texture_entry(3, cs, DERIVED_FORMAT, wgpu::TextureViewDimension::D3),
                util::storage_texture_entry(4, cs, DERIVED_FORMAT, wgpu::TextureViewDimension::D3),
            ],
        });

        let all =
            wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX;
        let read_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("derived fields (read)"),
            entries: &[
                util::uniform_entry(0, all),
                util::sampler_entry(1, all, wgpu::SamplerBindingType::Filtering),
                util::sampled_float_entry(2, all, wgpu::TextureViewDimension::D3),
                util::sampled_float_entry(3, all, wgpu::TextureViewDimension::D3),
            ],
        });
        let read_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("derived fields (read)"),
            layout: &read_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fields_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&scalars_sampled),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&velocity_sampled),
                },
            ],
        });

        let defines = ShaderDefines::new()
            .value("DERIVE_WG_X", Self::WG.x)
            .value("DERIVE_WG_Y", Self::WG.y)
            .value("DERIVE_WG_Z", Self::WG.z);
        let pipeline = util::compute_pipeline(
            device,
            loader,
            "derive.wgsl",
            "derive_main",
            &defines,
            &[Some(&derive_layout)],
            "derive fields",
        )?;

        let this = Self {
            grid,
            resolution,
            dims,
            voxel_mm,
            volume_min_mm,
            placement: Mat4::IDENTITY,
            scalars,
            velocity,
            scalars_storage,
            velocity_storage,
            scalars_sampled,
            velocity_sampled,
            sampler,
            derive_uniform,
            fields_uniform,
            derive_layout,
            read_layout,
            read_group,
            pipeline,
            fallback_flags,
            field: DerivedField::Speed,
            generation: 0,
        };
        this.upload_fields_uniform(queue);
        Ok(this)
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }
    pub fn resolution(&self) -> FieldResolution {
        self.resolution
    }
    pub fn dims(&self) -> UVec3 {
        self.dims
    }
    /// Derived voxel size in mm. This is the reference step `h_ref`.
    pub fn voxel_mm(&self) -> f32 {
        self.voxel_mm
    }
    pub fn volume_min_mm(&self) -> Vec3 {
        self.volume_min_mm
    }
    pub fn volume_size_mm(&self) -> Vec3 {
        self.dims.as_vec3() * self.voxel_mm
    }
    pub fn bbox(&self) -> ad_gpu::Bbox {
        ad_gpu::Bbox {
            min: self.volume_min_mm,
            max: self.volume_min_mm + self.volume_size_mm(),
        }
    }
    pub fn field(&self) -> DerivedField {
        self.field
    }
    /// Generation counter; changes whenever the field data changes.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Layout of the read-side bind group. **Wave 2 extension point**: build
    /// particle / isosurface / slice pipelines against this and they can sample
    /// the same textures with no further plumbing.
    pub fn read_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.read_layout
    }
    pub fn read_bind_group(&self) -> &wgpu::BindGroup {
        &self.read_group
    }
    /// Sampled views, for a pass that wants its own bind group layout.
    pub fn scalar_view(&self) -> &wgpu::TextureView {
        &self.scalars_sampled
    }
    pub fn velocity_view(&self) -> &wgpu::TextureView {
        &self.velocity_sampled
    }
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
    pub fn scalar_texture(&self) -> &wgpu::Texture {
        &self.scalars
    }
    pub fn velocity_texture(&self) -> &wgpu::Texture {
        &self.velocity
    }

    /// Switch the displayed scalar. One uniform word; no recompute.
    pub fn set_field(&mut self, queue: &wgpu::Queue, field: DerivedField) {
        if self.field != field {
            self.field = field;
            self.upload_fields_uniform(queue);
        }
    }

    /// Place the whole field in the world: the lattice-to-world transform the
    /// meshes are drawn with (`Scene::model`). Uploads only when it changed.
    pub fn set_placement(&mut self, queue: &wgpu::Queue, lattice_to_world: Mat4) {
        if self.placement != lattice_to_world {
            self.placement = lattice_to_world;
            self.upload_fields_uniform(queue);
        }
    }

    pub fn placement(&self) -> Mat4 {
        self.placement
    }

    fn upload_fields_uniform(&self, queue: &wgpu::Queue) {
        let size = self.volume_size_mm();
        let u = FieldsUniform {
            dims: self.dims.to_array(),
            channel: self.field.channel(),
            volume_min_mm: self.volume_min_mm.to_array(),
            voxel_mm: self.voxel_mm,
            volume_size_mm: size.to_array(),
            diagonal_mm: size.length(),
            lattice_to_world: self.placement.to_cols_array(),
            world_to_lattice: self.placement.inverse().to_cols_array(),
        };
        queue.write_buffer(&self.fields_uniform, 0, bytemuck::bytes_of(&u));
    }

    /// Record the derive pass. Call once per sim update, not once per frame.
    pub fn derive(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut Profiler,
        sources: &FieldSources<'_>,
    ) {
        let u = DeriveUniform {
            src_dims: self.grid.dims.to_array(),
            downsample: self.resolution.factor(),
            dst_dims: self.dims.to_array(),
            flags_present: u32::from(sources.flags_view.is_some()),
            speed_scale: sources.scales.speed_ms,
            q_scale: sources.scales.q_tilde,
            vort_scale: sources.scales.vorticity,
            pressure_scale: sources.scales.pressure_pa,
        };
        queue.write_buffer(&self.derive_uniform, 0, bytemuck::bytes_of(&u));

        let flags = sources.flags_view.unwrap_or(&self.fallback_flags);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("derive"),
            layout: &self.derive_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.derive_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(sources.macro_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(flags),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.scalars_storage),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.velocity_storage),
                },
            ],
        });

        let ts = profiler.scope("derive");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("derive fields"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(
                util::dispatch_count(self.dims.x, Self::WG.x),
                util::dispatch_count(self.dims.y, Self::WG.y),
                util::dispatch_count(self.dims.z, Self::WG.z),
            );
        }
        self.generation += 1;
    }

    /// Bytes moved by one derive pass, for the profiler's roofline percentage.
    pub fn traffic_bytes(&self) -> u64 {
        // Every source cell is read once for itself plus roughly six times as a
        // neighbour, at 8 bytes; every derived voxel is written twice at 8.
        let src = self.grid.cell_count() * 8 * 7;
        let dst = self.dims.x as u64 * self.dims.y as u64 * self.dims.z as u64 * 16;
        src + dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `J_ij = du_i/dx_j`, built column-major to match `glam` and WGSL.
    fn gradient(rows: [[f32; 3]; 3]) -> Mat3 {
        // rows[i][j] = du_i/dx_j, so column j gathers rows[*][j].
        Mat3::from_cols(
            Vec3::new(rows[0][0], rows[1][0], rows[2][0]),
            Vec3::new(rows[0][1], rows[1][1], rows[2][1]),
            Vec3::new(rows[0][2], rows[1][2], rows[2][2]),
        )
    }

    #[test]
    fn pure_shear_has_zero_q() {
        // u = (y, 0, 0). Strain and rotation are exactly equal, so Q = 0. This
        // is the whole reason Q exists: it separates a vortex from a shear
        // layer, and a shear layer must not light up.
        let j = gradient([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        assert!(
            q_criterion(j).abs() < 1e-6,
            "pure shear gave Q = {}",
            q_criterion(j)
        );
        // ...but its vorticity is emphatically not zero, which is exactly why
        // vorticity magnitude alone is a bad vortex detector.
        assert!((vorticity(j).length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn solid_body_rotation_has_positive_q() {
        // u = omega x r with omega = (0, 0, 1): u = (-y, x, 0). Pure rotation,
        // no strain, so Q = 0.5 |Omega|^2 = 1 and |curl u| = 2.
        let j = gradient([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        assert!(
            (q_criterion(j) - 1.0).abs() < 1e-6,
            "got Q = {}",
            q_criterion(j)
        );
        assert!((vorticity(j) - Vec3::new(0.0, 0.0, 2.0)).length() < 1e-6);
    }

    #[test]
    fn pure_strain_has_negative_q() {
        // Extensional flow u = (x, -y, 0): all strain, no rotation.
        let j = gradient([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]);
        assert!(q_criterion(j) < -0.5, "got Q = {}", q_criterion(j));
        assert!(vorticity(j).length() < 1e-6);
    }

    #[test]
    fn q_is_invariant_under_a_rigid_rotation_of_the_frame() {
        // Q is a scalar invariant of the gradient tensor. If the WGSL version
        // ever picks up a transpose error, this is what catches it, because a
        // transposed J flips the sign of Q for a mixed flow.
        let j = gradient([[0.3, 1.2, -0.4], [0.7, -0.1, 0.9], [0.2, 0.5, -0.2]]);
        let r = Mat3::from_rotation_y(0.7) * Mat3::from_rotation_x(-0.4);
        let rotated = r * j * r.transpose();
        assert!(
            (q_criterion(j) - q_criterion(rotated)).abs() < 1e-4,
            "{} vs {}",
            q_criterion(j),
            q_criterion(rotated)
        );
        assert!((vorticity(j).length() - vorticity(rotated).length()).abs() < 1e-4);
    }

    #[test]
    fn the_row_column_convention_cannot_corrupt_q_or_vorticity_magnitude() {
        // Worth pinning explicitly, because it is counter-intuitive and it tells
        // you where a convention bug *would* show up. Transposing J leaves the
        // symmetric part alone and only negates the antisymmetric part, so both
        // Frobenius norms — and therefore Q, and therefore |omega| — are
        // unchanged. The vorticity *vector* does flip, so anything that uses its
        // direction (Wave 2's vortex-core seeding, for one) is the thing that
        // would break, not the scalars stored here.
        let j = gradient([[0.0, -2.0, 0.3], [0.5, 0.0, -0.7], [0.1, 0.4, 0.0]]);
        let t = j.transpose();
        assert!((q_criterion(j) - q_criterion(t)).abs() < 1e-5);
        assert!((vorticity(j).length() - vorticity(t).length()).abs() < 1e-5);
        assert!(
            (vorticity(j) + vorticity(t)).length() < 1e-5,
            "the vector must flip"
        );
        // And the case itself is rotation-dominated, so Q is genuinely positive
        // rather than accidentally zero on both sides.
        assert!(q_criterion(j) > 0.1, "got Q = {}", q_criterion(j));
    }

    #[test]
    fn normalised_q_is_independent_of_the_lattice_velocity() {
        // The property the whole normalisation exists for. Two solver setups of
        // the *same physical flow* at different `u_lb` must report the same Q~.
        let d_h_mm = 6.3;
        let dx_mm = 0.4;
        let u_phys = 5.0;

        let q_tilde_for = |u_lb: f64| {
            let units = LatticeUnits::for_air(dx_mm, u_phys, u_lb);
            let scales = DeriveScales::new(&units, d_h_mm);
            // A fixed physical vortex: u = omega x r with omega = 400 rad/s.
            // In lattice units the gradient per cell is (domega/dx_phys) * dx / c_u.
            let omega_phys = 400.0;
            let g = (omega_phys * units.dx_m / units.c_u()) as f32;
            let j = gradient([[0.0, -g, 0.0], [g, 0.0, 0.0], [0.0, 0.0, 0.0]]);
            q_criterion(j) * scales.q_tilde
        };

        let a = q_tilde_for(0.1);
        let b = q_tilde_for(0.02);
        assert!(a > 1e-6, "test produced a degenerate Q~ of {a}");
        assert!((a - b).abs() / a < 1e-3, "Q~ moved with u_lb: {a} vs {b}");
    }

    #[test]
    fn normalised_q_is_independent_of_the_grid_spacing() {
        // Refining the mesh must not move the isolevel either.
        let q_tilde_for = |dx_mm: f64| {
            let units = LatticeUnits::for_air(dx_mm, 5.0, 0.05);
            let scales = DeriveScales::new(&units, 6.3);
            let g = (400.0 * units.dx_m / units.c_u()) as f32;
            let j = gradient([[0.0, -g, 0.0], [g, 0.0, 0.0], [0.0, 0.0, 0.0]]);
            q_criterion(j) * scales.q_tilde
        };
        let a = q_tilde_for(0.75);
        let b = q_tilde_for(0.3);
        assert!((a - b).abs() / a < 1e-3, "Q~ moved with dx: {a} vs {b}");
    }

    #[test]
    fn scales_agree_with_the_lattice_unit_conversions() {
        let units = LatticeUnits::for_air(0.4, 8.0, 0.1);
        let s = DeriveScales::new(&units, 6.3);
        // Speed: one lattice velocity unit is c_u m/s, and the inlet at u_lb
        // must come back out as u_phys.
        assert!((s.speed_ms * units.u_lb as f32 - 8.0).abs() < 1e-3);
        // Pressure: matches ad-gpu's own converter exactly.
        assert!((s.pressure_pa - units.pressure_pa(1.0) as f32).abs() < 1e-3);
        // Vorticity scale is the square root of the Q scale, by construction.
        assert!((s.vorticity * s.vorticity - s.q_tilde).abs() / s.q_tilde < 1e-4);
    }

    #[test]
    fn field_channels_are_distinct_and_in_range() {
        let mut seen = [false; 4];
        for f in DerivedField::ALL {
            let c = f.channel() as usize;
            assert!(c < 4, "{} has channel {c}", f.name());
            assert!(!seen[c], "channel {c} used twice");
            seen[c] = true;
        }
        assert!(seen.iter().all(|s| *s), "not every channel is used");
    }

    #[test]
    fn derived_formats_are_storage_capable_without_optional_features() {
        // The whole reason DERIVED_FORMAT is Rgba16Float and not Rg16Float.
        let base = wgpu::Features::empty();
        for f in [DERIVED_FORMAT] {
            let feats = f.guaranteed_format_features(base);
            assert!(
                feats
                    .allowed_usages
                    .contains(wgpu::TextureUsages::STORAGE_BINDING),
                "{f:?} is not storage-capable in the base feature set"
            );
            assert!(feats
                .allowed_usages
                .contains(wgpu::TextureUsages::TEXTURE_BINDING));
        }
        // ...and the trap we avoided.
        assert!(!wgpu::TextureFormat::Rg16Float
            .guaranteed_format_features(base)
            .allowed_usages
            .contains(wgpu::TextureUsages::STORAGE_BINDING));
        // Flags only ever need to be sampled and copied into.
        let ff = FLAGS_FORMAT.guaranteed_format_features(base);
        assert!(ff
            .allowed_usages
            .contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert!(ff.allowed_usages.contains(wgpu::TextureUsages::COPY_DST));
    }

    #[test]
    fn uniforms_are_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<DeriveUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<FieldsUniform>() % 16, 0);
        // Three vec4 rows then two mat4x4: the layout of `FieldsInfo` in
        // common.wgsl, which every render pass that samples the fields includes.
        assert_eq!(std::mem::size_of::<FieldsUniform>(), 48 + 2 * 64);
    }
}

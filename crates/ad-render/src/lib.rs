//! AeroDuct render core.
//!
//! An interactive GPU renderer for a lattice-Boltzmann duct simulation, built so
//! that the picture stays honest and the frame rate stays free of the sim rate.
//!
//! # The frame
//!
//! ```text
//! derived fields  (only when the solver has stepped)
//!   -> brick min/max + Chebyshev distance   (only when the field or TF changed)
//!     -> opaque G-buffer                    (albedo, normal, depth, motion)
//!       -> SSAO
//!         -> volume raymarch (half res, empty-space skipped, depth-composited)
//!           -> composite  (background, shading, volume over)
//!             -> geometry overlay (ghost / wireframe)
//!               -> Wave 2 overlays (particles, isosurfaces, slices)
//!                 -> TAA  (or progressive accumulation when nothing is moving)
//!                   -> bloom
//!                     -> AgX tonemap -> target
//! ```
//!
//! # The three things that hold it together
//!
//! **Derived fields.** Nothing downstream of [`fields`] ever touches the
//! distribution functions. Four scalars live in the four channels of one
//! `Rgba16Float` 3D texture, so switching the displayed field is a uniform
//! write, not a recompute.
//!
//! **Reverse-Z, consistently.** Depth runs 1 at the near plane to 0 at infinity,
//! is cleared to 0, and is tested `Greater`. [`camera::Camera::linear_depth`] is
//! the single definition of how to invert it, mirrored once in
//! `shaders/render/common.wgsl`.
//!
//! **Opacity correction.** Every volume sample applies
//! `alpha_c = 1 - (1 - alpha_ref)^(h / h_ref)`. Without it the image depends on
//! the step size, and every empty-space-skipping optimisation silently changes
//! the picture — which makes the accelerator impossible to debug. With it,
//! `skip_empty_space` on and off must match, so a difference is a bug.
//!
//! # Extending it (Wave 2)
//!
//! Particles, isosurfaces and slice planes all want the same thing: draw into
//! the HDR target with the scene's depth, sampling the derived fields. Implement
//! [`OverlayPass`] and register it with [`Renderer::add_overlay`]. Build
//! pipelines against [`Renderer::camera_bind_group_layout`],
//! [`Renderer::fields_bind_group_layout`] and
//! [`Renderer::brick_bind_group_layout`], targeting [`post::HDR_FORMAT`] and
//! [`mesh::DEPTH_FORMAT`]. The overlay runs after the volume and before TAA, so
//! it is antialiased and tonemapped with everything else.

pub mod accel;
pub mod camera;
pub mod colormap;
pub mod fields;
pub mod isosurface;
pub mod lic;
pub mod mesh;
pub mod noise;
pub mod particles;
pub mod post;
pub mod slice;
pub mod transfer;
pub mod util;
pub mod volume;

use std::sync::Arc;

use ad_gpu::{Bbox, GpuContext, Grid, Profiler, ShaderLoader};
use anyhow::Result;
use glam::{Mat4, Vec3};

pub use accel::BrickGrid;
pub use camera::{Camera, CameraUniform, Flythrough, Keyframe, OrbitController, ViewPreset};
pub use colormap::{ColorMap, Interpolation, MapKind};
pub use fields::{
    DeriveScales, DerivedField, DerivedFields, FieldResolution, FieldSources, FieldsUniform,
};
pub use isosurface::{IsosurfaceOverlay, IsosurfaceSettings, IsosurfaceShading, QHistogram};
pub use lic::{Centreline, LicSettings};
pub use mesh::{GpuMesh, MeshData, MeshDisplay, MeshStyle, MeshVertex};
pub use particles::{
    SdfSource, SeedWeights, StreaklineColor, StreaklineConfig, StreaklineOverlay,
    StreaklineSettings, TrailSettings,
};
pub use post::{BackgroundSettings, LightingSettings, PostSettings};
pub use slice::{AxisPreset, SliceOverlay, SlicePlane, SliceSettings};
pub use transfer::{OpacityCurve, OpacityMode, RangeScale, SoftIso, Support, TransferFunction};
pub use volume::{SdfVolume, VolumeSettings};

/// Everything the renderer draws that is not the flow itself.
pub struct Scene {
    pub meshes: Vec<GpuMesh>,
    /// Model-to-world transform, mm. One transform for the whole scene: the
    /// meshes come out of the same STL space.
    pub model: Mat4,
    prev_model: Mat4,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        Self { meshes: Vec::new(), model: Mat4::IDENTITY, prev_model: Mat4::IDENTITY }
    }

    /// World-space bounds of everything visible.
    pub fn bbox(&self) -> Bbox {
        self.meshes
            .iter()
            .filter(|m| m.visible)
            .fold(Bbox::EMPTY, |b, m| {
                // Transform the eight corners rather than the two extremes, so a
                // rotated model still gets a correct (if loose) box.
                let mut acc = b;
                for i in 0..8 {
                    let c = Vec3::new(
                        if i & 1 == 0 { m.bbox.min.x } else { m.bbox.max.x },
                        if i & 2 == 0 { m.bbox.min.y } else { m.bbox.max.y },
                        if i & 4 == 0 { m.bbox.min.z } else { m.bbox.max.z },
                    );
                    acc = acc.union_point((self.model * m.model).transform_point3(c));
                }
                acc
            })
    }

    /// Whether the scene moved since the last frame. Motion vectors are exact
    /// for a rigid transform, but the accumulator still has to be told.
    pub fn is_static(&self) -> bool {
        self.model.abs_diff_eq(self.prev_model, 1e-7)
            && self.meshes.iter().all(|m| m.model.abs_diff_eq(m.prev_model, 1e-7))
    }

    /// Call once per frame after rendering, so next frame's motion vectors have
    /// something to reproject from.
    pub fn end_frame(&mut self) {
        self.prev_model = self.model;
        for m in &mut self.meshes {
            m.prev_model = m.model;
        }
    }
}

/// Context handed to a [`OverlayPass`].
pub struct OverlayContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub profiler: &'a mut Profiler,
    /// Bind group 0 in every render pass in this crate.
    pub camera_bind_group: &'a wgpu::BindGroup,
    pub fields: &'a DerivedFields,
    pub bricks: &'a BrickGrid,
    /// `Rgba16Float`, linear HDR, already carrying the background, the geometry
    /// and the volume. Bind as a render attachment with `LoadOp::Load`.
    pub hdr_view: &'a wgpu::TextureView,
    /// The opaque depth buffer, reverse-Z. Test against it; do not write.
    pub depth_view: &'a wgpu::TextureView,
    pub size: (u32, u32),
    /// Seconds since the previous frame, for anything that integrates.
    pub dt: f32,
}

/// A pass that draws into the scene between the volume and TAA.
///
/// This is the Wave 2 seam: particles, isosurfaces and slice planes are all
/// exactly this. Running before TAA means they are antialiased and tonemapped
/// along with everything else, which is what stops an overlay from looking
/// pasted on.
pub trait OverlayPass: std::any::Any {
    fn name(&self) -> &str;
    fn record(&mut self, ctx: &mut OverlayContext<'_>);
    /// Return false while the pass is animating, so progressive accumulation
    /// does not average a moving overlay into a still image.
    fn is_static(&self) -> bool {
        true
    }

    /// Upcast so [`Renderer::overlay_as_mut`] can recover the concrete type.
    ///
    /// This exists because registration moves the pass into the renderer, and
    /// the UI still needs to reach it afterwards -- to drag an isolevel, pause
    /// the tracers, or move a slice plane. Without it an overlay is
    /// write-once, which makes every setting on it dead on arrival.
    ///
    /// Implement it as `fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }`.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub target_format: wgpu::TextureFormat,
    /// The solver's grid. Derived fields are allocated against it.
    pub grid: Grid,
    pub field_resolution: FieldResolution,
    /// GPU timestamps. Costs a little; worth having on by default while the
    /// renderer is young.
    pub profiling: bool,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, target_format: wgpu::TextureFormat, grid: Grid) -> Self {
        Self {
            width,
            height,
            target_format,
            grid,
            field_resolution: FieldResolution::Half,
            profiling: true,
        }
    }
}

/// Inputs for one frame.
pub struct FrameInput<'a> {
    pub camera: &'a Camera,
    pub scene: &'a Scene,
    /// Supplied only on frames where the solver produced new state. Passing
    /// `None` is the normal case: the renderer runs faster than the sim.
    pub sources: Option<FieldSources<'a>>,
    /// Optional solid SDF from the geometry crate, used to sphere-trace through
    /// the duct wall.
    pub sdf: Option<SdfVolume<'a>>,
    /// Where the tonemapped image goes.
    pub target: &'a wgpu::TextureView,
    pub dt: f32,
}

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    loader: ShaderLoader,

    camera_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    camera_group: wgpu::BindGroup,

    fields: DerivedFields,
    bricks: BrickGrid,
    volume: volume::VolumeRenderer,
    mesh: mesh::MeshRenderer,
    gbuffer: post::GBuffer,
    post: post::PostChain,
    profiler: Profiler,

    /// One transfer function per derived field, indexed by channel.
    ///
    /// Per-field rather than one shared: the fields have wildly different units
    /// and signs, so a single transfer function would be wrong for three of the
    /// four at any moment. Keeping them separate also means switching to
    /// Q-criterion and back does not throw away the isolevel you just spent a
    /// minute tuning, which is the difference between a switch you use and one
    /// you avoid.
    tf: [TransferFunction; 4],
    tf_dirty: bool,
    accessible_colors: bool,

    overlays: Vec<Box<dyn OverlayPass>>,

    size: (u32, u32),
    base_size: (u32, u32),
    supersample: u32,
    frame: u32,
    prev_view_proj: Mat4,
    scene_center: Vec3,
}

impl Renderer {
    /// Number of timestamp pairs the profiler makes room for. Nine core scopes
    /// plus headroom for Wave 2 overlays.
    const PROFILER_SCOPES: u32 = 24;

    pub fn new(gpu: &GpuContext, config: RendererConfig) -> Result<Self> {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let loader = util::shader_loader();

        let stages = wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE;
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera"),
            entries: &[util::uniform_entry(0, stages)],
        });
        let camera_buffer = util::uniform_buffer::<CameraUniform>(&device, "camera uniform");
        let camera_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
        });

        let fields = DerivedFields::new(
            &device,
            &queue,
            &loader,
            config.grid,
            config.field_resolution,
        )?;
        let bricks = BrickGrid::new(&device, &loader, &fields)?;
        let volume = volume::VolumeRenderer::new(
            &device,
            &queue,
            &loader,
            &camera_layout,
            fields.read_bind_group_layout(),
            bricks.read_bind_group_layout(),
            config.width,
            config.height,
            VolumeSettings::default(),
        )?;
        let mesh = mesh::MeshRenderer::new(&device, &loader, &camera_layout, post::HDR_FORMAT)?;
        let gbuffer = post::GBuffer::new(&device, config.width, config.height);
        let post_chain = post::PostChain::new(
            &device,
            &queue,
            &loader,
            &camera_layout,
            config.width,
            config.height,
            config.target_format,
            PostSettings::default(),
        )?;
        let profiler = Profiler::new(
            &device,
            &queue,
            Self::PROFILER_SCOPES,
            config.profiling && gpu.caps.timestamps,
            gpu.caps.peak_bandwidth,
        );

        let tf = [
            TransferFunction::preset(DerivedField::Speed),
            TransferFunction::preset(DerivedField::QCriterion),
            TransferFunction::preset(DerivedField::Vorticity),
            TransferFunction::preset(DerivedField::Pressure),
        ];
        volume.upload_transfer_function(&queue, &tf[0]);

        Ok(Self {
            device,
            queue,
            loader,
            camera_layout,
            camera_buffer,
            camera_group,
            fields,
            bricks,
            volume,
            mesh,
            gbuffer,
            post: post_chain,
            profiler,
            tf,
            tf_dirty: false,
            accessible_colors: false,
            overlays: Vec::new(),
            size: (config.width.max(1), config.height.max(1)),
            base_size: (config.width.max(1), config.height.max(1)),
            supersample: 1,
            frame: 0,
            prev_view_proj: Mat4::IDENTITY,
            scene_center: Vec3::ZERO,
        })
    }

    // -- configuration --------------------------------------------------------

    pub fn resize(&mut self, width: u32, height: u32) {
        self.base_size = (width.max(1), height.max(1));
        self.reallocate();
    }

    fn reallocate(&mut self) {
        let (w, h) = (
            self.base_size.0 * self.supersample,
            self.base_size.1 * self.supersample,
        );
        if (w, h) == self.size && self.gbuffer.size == (w, h) {
            return;
        }
        self.size = (w, h);
        self.gbuffer = post::GBuffer::new(&self.device, w, h);
        self.volume.resize(&self.device, w, h);
        self.post.resize(&self.device, w, h);
    }

    /// Render at `scale` times the display resolution, for a supersampled
    /// capture.
    ///
    /// Returns the internal resolution now in use, which is also the size the
    /// caller's target texture must be. The full recipe:
    ///
    /// ```text
    /// let (w, h) = renderer.begin_supersample(2);
    /// // ...allocate a w x h RENDER_ATTACHMENT | COPY_SRC target...
    /// while renderer.accumulated_samples() < 64 {
    ///     renderer.render(&mut encoder, FrameInput { .. });   // camera held still
    /// }
    /// let px = util::readback_rgba8(device, queue, &target, w, h, encoder);
    /// let out = util::box_downsample_rgba8(&px, w, h, 2);
    /// renderer.end_supersample();
    /// ```
    ///
    /// The two mechanisms compose: `scale^2` spatial samples from the larger
    /// targets times `n` temporal samples from the jittered accumulator. Doing
    /// it purely spatially would need a 16x buffer for the same result, and
    /// doing it purely temporally cannot fix the resolution of the raymarch.
    pub fn begin_supersample(&mut self, scale: u32) -> (u32, u32) {
        self.supersample = scale.clamp(1, 4);
        self.reallocate();
        self.post.reset_accumulation();
        self.size
    }

    pub fn end_supersample(&mut self) {
        self.supersample = 1;
        self.reallocate();
        self.post.reset_accumulation();
    }

    pub fn supersample(&self) -> u32 {
        self.supersample
    }
    /// Internal render resolution, which differs from the display size while
    /// supersampling.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
    /// Number of jittered samples currently integrated. Grows while nothing is
    /// moving and resets to 0 the moment anything does.
    pub fn accumulated_samples(&self) -> u32 {
        self.post.accumulated_samples()
    }

    pub fn set_target_format(&mut self, format: wgpu::TextureFormat) -> Result<()> {
        self.post.set_target_format(&self.device, &self.loader, format)
    }

    /// Switch the displayed scalar.
    ///
    /// All four scalars already live in the four channels of one texture, so
    /// this costs a uniform word, a 2 KiB LUT upload and a brick min/max rebuild
    /// — no re-derivation from the solver state, and nothing that scales with
    /// the number of fields. That is the whole reason the derive pass exists.
    pub fn set_field(&mut self, field: DerivedField) {
        if self.fields.field() != field {
            self.fields.set_field(&self.queue, field);
            self.tf_dirty = true;
        }
    }
    pub fn field(&self) -> DerivedField {
        self.fields.field()
    }

    /// The transfer function for the field currently displayed.
    pub fn transfer_function(&self) -> &TransferFunction {
        &self.tf[self.fields.field().channel() as usize]
    }
    /// Mutate it. The LUT is re-baked and the brick binarisation invalidated on
    /// the next frame; both are cheap enough to do on every drag of a slider.
    pub fn transfer_function_mut(&mut self) -> &mut TransferFunction {
        self.tf_dirty = true;
        &mut self.tf[self.fields.field().channel() as usize]
    }
    /// The transfer function for a field that is not currently displayed, so the
    /// UI can show all four at once.
    pub fn transfer_function_for(&self, field: DerivedField) -> &TransferFunction {
        &self.tf[field.channel() as usize]
    }
    /// Restore a field's transfer function to its preset.
    pub fn reset_transfer_function(&mut self, field: DerivedField) {
        self.tf[field.channel() as usize] = TransferFunction::preset(field);
        self.tf_dirty = true;
    }

    /// Swap every active colour map for a colourblind-safe equivalent of the
    /// same kind.
    pub fn set_accessible_colors(&mut self, on: bool) {
        if self.accessible_colors != on {
            self.accessible_colors = on;
            self.tf_dirty = true;
        }
    }
    pub fn accessible_colors(&self) -> bool {
        self.accessible_colors
    }

    pub fn volume_settings(&self) -> &VolumeSettings {
        self.volume.settings()
    }
    pub fn set_volume_settings(&mut self, s: VolumeSettings) {
        let (w, h) = self.size;
        if self.volume.set_settings(&self.device, s, w, h) {
            self.post.reset_accumulation();
        }
    }

    pub fn post_settings(&self) -> &PostSettings {
        &self.post.settings
    }
    pub fn post_settings_mut(&mut self) -> &mut PostSettings {
        self.post.reset_accumulation();
        &mut self.post.settings
    }

    pub fn mesh_display(&self) -> MeshDisplay {
        self.mesh.display
    }
    pub fn set_mesh_display(&mut self, d: MeshDisplay) {
        if self.mesh.display != d {
            self.mesh.display = d;
            self.post.reset_accumulation();
        }
    }
    pub fn cycle_mesh_display(&mut self) -> MeshDisplay {
        self.set_mesh_display(self.mesh.display.cycle());
        self.mesh.display
    }

    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    // -- extension points -----------------------------------------------------

    /// Register a Wave 2 pass. Overlays run in registration order, after the
    /// volume and before TAA.
    pub fn add_overlay(&mut self, pass: Box<dyn OverlayPass>) {
        self.overlays.push(pass);
    }
    pub fn remove_overlay(&mut self, name: &str) {
        self.overlays.retain(|p| p.name() != name);
    }
    /// Borrow a registered overlay by name, without knowing its type.
    pub fn overlay_mut(&mut self, name: &str) -> Option<&mut dyn OverlayPass> {
        self.overlays
            .iter_mut()
            .find(|o| o.name() == name)
            .map(|o| o.as_mut() as &mut dyn OverlayPass)
    }

    /// Borrow a registered overlay by name and concrete type.
    ///
    /// Returns `None` if no overlay of that name is registered, or if one is but
    /// is a different type -- so a renamed or replaced overlay degrades to "the
    /// control does nothing" rather than a panic.
    pub fn overlay_as_mut<T: OverlayPass>(&mut self, name: &str) -> Option<&mut T> {
        self.overlay_mut(name)?.as_any_mut().downcast_mut::<T>()
    }

    pub fn overlay_names(&self) -> Vec<&str> {
        self.overlays.iter().map(|p| p.name()).collect()
    }

    /// Bind group 0 in every pass. Contains a [`CameraUniform`].
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_layout
    }
    /// `(uniform, sampler, scalars, velocity)`. The velocity texture in
    /// particular is what a particle advection pass wants.
    pub fn fields_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.fields.read_bind_group_layout()
    }
    /// `(uniform, Chebyshev distance, min/max)`, for a pass that wants the same
    /// empty-space skipping the raymarch uses.
    pub fn brick_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.bricks.read_bind_group_layout()
    }
    pub fn fields(&self) -> &DerivedFields {
        &self.fields
    }
    pub fn bricks(&self) -> &BrickGrid {
        &self.bricks
    }
    pub fn shader_loader(&self) -> &ShaderLoader {
        &self.loader
    }
    pub fn hdr_format(&self) -> wgpu::TextureFormat {
        post::HDR_FORMAT
    }
    pub fn depth_format(&self) -> wgpu::TextureFormat {
        mesh::DEPTH_FORMAT
    }

    // -- the frame ------------------------------------------------------------

    /// Record the whole frame into `encoder`. The caller submits.
    pub fn render(&mut self, encoder: &mut wgpu::CommandEncoder, input: FrameInput<'_>) -> Result<()> {
        self.profiler.begin_frame();

        // 1. Derived fields, when the solver has stepped.
        let stepped = input.sources.is_some();
        if let Some(sources) = &input.sources {
            self.fields.derive(
                &self.device,
                &self.queue,
                encoder,
                &mut self.profiler,
                sources,
            );
        }

        // 2. Transfer function, when it changed.
        let channel = self.fields.field().channel() as usize;
        if self.tf_dirty {
            self.tf[channel].sanitise();
            let mut effective = self.tf[channel].clone();
            if self.accessible_colors {
                effective.map = effective.map.accessible_substitute();
            }
            self.volume.upload_transfer_function(&self.queue, &effective);
            self.tf_dirty = false;
            self.post.reset_accumulation();
        }

        // 3. Acceleration structure, when the field or the TF changed.
        let rebuilt = self.bricks.update(
            &self.queue,
            encoder,
            &mut self.profiler,
            &self.fields,
            &self.tf[channel],
        );

        // 3b. One light rig, three consumers.
        //
        // The deferred shading, the ghost shell and the volume's gradient
        // shading each need a key direction, and `PostSettings::lighting` is the
        // single place it lives. Syncing here rather than making callers set it
        // three times removes a whole class of "why is the shell lit from the
        // other side" confusion, which on a translucent object reads as a
        // geometry bug rather than a lighting one.
        let key = self.post.settings.lighting.key_dir.normalize_or(Vec3::Y);
        self.mesh.light_dir = key;
        self.mesh.light_intensity = self.post.settings.lighting.key_intensity.min(1.5);
        {
            // The volume shades with field gradients, which are in lattice
            // axes, so its light turns with the install pose; the meshes are
            // lit in the world.
            let model = input.scene.model;
            let key_lattice = if model == Mat4::IDENTITY {
                key
            } else {
                model.inverse().transform_vector3(key).normalize_or(key)
            };
            let mut vs = *self.volume.settings();
            if vs.light_dir != key_lattice {
                vs.light_dir = key_lattice;
                self.volume.set_settings(&self.device, vs, self.size.0, self.size.1);
            }
        }

        // 3c. The fields sit in the world where the meshes do: through the
        //     install pose. One transform for the whole scene, as for meshes.
        self.fields.set_placement(&self.queue, input.scene.model);

        // 4. Camera. The jitter moves every frame; everything else deciding
        //    "static" must not.
        let (w, h) = self.size;
        let mut cam = *input.camera;
        cam.aspect = w as f32 / h.max(1) as f32;
        cam.jitter = camera::taa_jitter(self.frame, w, h);
        cam.sanitise();

        let vp = cam.view_projection_unjittered();
        let camera_static = vp.abs_diff_eq(self.prev_view_proj, 1.0e-7);
        let overlays_static = self.overlays.iter().all(|p| p.is_static());
        let static_scene = camera_static
            && input.scene.is_static()
            && overlays_static
            && !stepped
            && !rebuilt;

        let cu = CameraUniform::new(&cam, self.prev_view_proj, w, h, self.frame);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cu));

        // Keep the background's ground plane and contact shadow under the part.
        let bbox = input.scene.bbox();
        if !bbox.is_empty() {
            self.scene_center = bbox.center();
            let s = &mut self.post.settings.background;
            s.ground_y_mm = bbox.min.y - bbox.size().y * 0.02;
            s.shadow_radius_mm = (bbox.size().x.max(bbox.size().z) * 0.75).max(1.0);
        }

        // 5. Opaque G-buffer. Runs even in ghost/wireframe/off, because clearing
        //    depth to the reverse-Z far plane is what tells the raymarcher there
        //    is nothing in the way.
        self.mesh.upload(&self.queue, &input.scene.meshes, input.scene.model, input.scene.prev_model);
        self.mesh.render_gbuffer(
            encoder,
            &mut self.profiler,
            &self.camera_group,
            &input.scene.meshes,
            &self.gbuffer.albedo,
            &self.gbuffer.normal,
            &self.gbuffer.motion,
            &self.gbuffer.depth,
        );

        // 6. SSAO on the geometry.
        self.post.ssao(
            &self.device,
            &self.queue,
            encoder,
            &mut self.profiler,
            &self.camera_group,
            &self.gbuffer.depth,
            &self.gbuffer.normal,
        );

        // 7. Volume.
        self.volume.advance_jitter();
        self.volume.render(
            &self.device,
            &self.queue,
            encoder,
            &mut self.profiler,
            &self.camera_group,
            &self.fields,
            &self.bricks,
            &self.gbuffer.depth,
            input.sdf.as_ref(),
            &self.tf[channel],
        );

        // 8. Background, deferred shading, volume over.
        self.post.composite(
            &self.device,
            &self.queue,
            encoder,
            &mut self.profiler,
            &self.camera_group,
            &self.gbuffer.albedo,
            &self.gbuffer.normal,
            &self.gbuffer.depth,
            self.volume.color_view(),
            self.scene_center,
            true,
        );

        // 9. Ghost shell or wireframe, over the composited flow.
        self.mesh.render_overlay(
            encoder,
            &self.camera_group,
            &input.scene.meshes,
            self.post.hdr_attachment_view(),
            &self.gbuffer.depth,
        );

        // 10. Wave 2 overlays. Taken out of `self` so they can borrow the rest
        //     of the renderer immutably while holding the encoder mutably.
        let mut overlays = std::mem::take(&mut self.overlays);
        if !overlays.is_empty() {
            let mut ctx = OverlayContext {
                device: &self.device,
                queue: &self.queue,
                encoder,
                profiler: &mut self.profiler,
                camera_bind_group: &self.camera_group,
                fields: &self.fields,
                bricks: &self.bricks,
                hdr_view: self.post.hdr_attachment_view(),
                depth_view: &self.gbuffer.depth,
                size: self.size,
                dt: input.dt,
            };
            for pass in overlays.iter_mut() {
                pass.record(&mut ctx);
            }
        }
        self.overlays = overlays;

        // 11. TAA / accumulation, bloom, tonemap.
        self.post.taa(
            &self.device,
            &self.queue,
            encoder,
            &mut self.profiler,
            &self.camera_group,
            &self.gbuffer.motion,
            &self.gbuffer.depth,
            self.volume.front_view(),
            static_scene,
        );
        self.post.bloom(&self.device, &self.queue, encoder, &mut self.profiler);
        self.post.tonemap(&self.device, &self.queue, encoder, input.target);

        self.profiler.resolve(encoder);

        self.prev_view_proj = vp;
        self.frame = self.frame.wrapping_add(1);
        Ok(())
    }

    /// Call after submitting the frame's command buffer.
    ///
    /// Collects every frame, without blocking. This used to collect only every
    /// 120 frames behind a `poll(wait_indefinitely)`, to work around an
    /// `ad_gpu::Profiler` that left its readback buffer mapped on a miss. The
    /// profiler now tracks its own mapping and never leaves it pending, and the
    /// workaround had become a cost of its own: a full queue drain every two
    /// seconds, which is a hitch the step tuner then reacts to, and no timings
    /// at all in a headless run shorter than 120 frames.
    pub fn after_submit(&mut self) {
        self.profiler.collect(&self.device);
    }

    /// One line per timed pass, for a status bar.
    pub fn profiling_report(&self) -> Vec<String> {
        const ORDER: [&str; 9] = [
            "derive",
            "brick min/max",
            "brick seed",
            "brick chebyshev",
            "ssao",
            "volume",
            "composite",
            "taa",
            "bloom",
        ];
        let mut lines: Vec<String> = Vec::new();
        for name in ORDER {
            if self.profiler.timing(name).is_some() {
                let bytes = if name == "derive" { self.fields.traffic_bytes() } else { 0 };
                lines.push(self.profiler.summary(name, bytes));
            }
        }
        // Then anything else that reported: overlays time themselves under
        // their own names, and a pass missing from the list above should still
        // show up rather than cost time invisibly.
        let mut rest: Vec<&str> = self
            .profiler
            .timings()
            .map(|(name, _)| name.as_str())
            .filter(|name| !ORDER.contains(name))
            .collect();
        rest.sort_unstable();
        lines.extend(rest.into_iter().map(|name| self.profiler.summary(name, 0)));
        lines
    }
}

#[cfg(test)]
mod tests {
    /// Registration moves an overlay into the renderer. If it cannot be reached
    /// again afterwards, every UI control bound to it is dead on arrival, so
    /// this pins both the lookup and the wrong-type path.
    #[test]
    fn a_registered_overlay_can_be_recovered_by_name_and_type() {
        struct Probe {
            name: &'static str,
            isolevel: f32,
        }
        impl OverlayPass for Probe {
            fn name(&self) -> &str {
                self.name
            }
            fn record(&mut self, _ctx: &mut OverlayContext<'_>) {}
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        }
        struct Other;
        impl OverlayPass for Other {
            fn name(&self) -> &str {
                "other"
            }
            fn record(&mut self, _ctx: &mut OverlayContext<'_>) {}
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        }

        // Exercise the lookup against a bare Vec rather than a full Renderer,
        // which would need a GPU device just to answer a question about names.
        let mut overlays: Vec<Box<dyn OverlayPass>> = vec![
            Box::new(Probe { name: "q-iso", isolevel: 0.5 }),
            Box::new(Other),
        ];

        fn find<'a>(
            v: &'a mut [Box<dyn OverlayPass>],
            name: &str,
        ) -> Option<&'a mut dyn OverlayPass> {
            v.iter_mut().find(|o| o.name() == name).map(|o| o.as_mut() as &mut dyn OverlayPass)
        }

        // The happy path: recover the concrete type and mutate it.
        let probe = find(&mut overlays, "q-iso")
            .and_then(|o| o.as_any_mut().downcast_mut::<Probe>())
            .expect("q-iso should be registered");
        probe.isolevel = 1.25;
        let probe = find(&mut overlays, "q-iso")
            .and_then(|o| o.as_any_mut().downcast_mut::<Probe>())
            .unwrap();
        assert_eq!(probe.isolevel, 1.25, "the mutation did not stick");

        // Asking for the wrong type must yield None, not panic: a renamed or
        // swapped overlay should make the control inert, not crash the app.
        assert!(find(&mut overlays, "other")
            .and_then(|o| o.as_any_mut().downcast_mut::<Probe>())
            .is_none());

        // And an unregistered name is simply absent.
        assert!(find(&mut overlays, "nope").is_none());
    }

    use super::*;
    use ad_gpu::Bbox;
    use glam::UVec3;

    fn test_grid() -> Grid {
        Grid {
            dims: UVec3::new(64, 48, 48),
            dx_mm: 0.75,
            origin_mm: Vec3::new(-24.0, -18.0, -18.0),
        }
    }

    /// Acquire a device, or return `None` so the test skips cleanly on a
    /// machine with no GPU. CI without an adapter must not fail the suite.
    fn gpu() -> Option<GpuContext> {
        match GpuContext::new_blocking(None) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    #[test]
    fn scene_bbox_follows_the_model_transform() {
        let mut scene = Scene::new();
        assert!(scene.bbox().is_empty(), "an empty scene has no bounds");
        assert!(scene.is_static());
        scene.model = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        assert!(!scene.is_static(), "a moved model must reset accumulation");
        scene.end_frame();
        assert!(scene.is_static());
    }

    #[test]
    fn renderer_builds_and_records_a_frame() {
        // The end-to-end test: every pipeline compiles, every bind group
        // matches its layout, and a whole frame records without a validation
        // error. That is most of what can go wrong in a renderer this size.
        let Some(gpu) = gpu() else { return };
        let grid = test_grid();
        let mut r = Renderer::new(
            &gpu,
            RendererConfig::new(320, 200, wgpu::TextureFormat::Rgba8UnormSrgb, grid),
        )
        .expect("renderer construction");

        let (mac, flg) = fields::create_source_textures(&gpu.device, grid);
        let mac_view = mac.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let flg_view = flg.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let target = util::color_target(
            &gpu.device,
            "test target",
            320,
            200,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let target_view = target.create_view(&Default::default());

        let mut scene = Scene::new();
        let verts = [
            MeshVertex { position: [-10.0, -10.0, 0.0], normal: [0.0, 0.0, 1.0], scalar: 0.0 },
            MeshVertex { position: [10.0, -10.0, 0.0], normal: [0.0, 0.0, 1.0], scalar: 0.5 },
            MeshVertex { position: [0.0, 10.0, 0.0], normal: [0.0, 0.0, 1.0], scalar: 1.0 },
        ];
        scene.meshes.push(GpuMesh::upload(
            &gpu.device,
            &gpu.queue,
            MeshData { vertices: &verts, indices: &[0, 1, 2] },
            MeshStyle::default(),
        ));

        let mut cam = Camera::default();
        cam.frame_bbox(Bbox { min: Vec3::splat(-24.0), max: Vec3::splat(24.0) }, 0.1);

        let units = ad_gpu::LatticeUnits::for_air(grid.dx_mm as f64, 5.0, 0.05);
        let scales = DeriveScales::new(&units, 6.3);

        for frame in 0..3 {
            if frame == 2 {
                // A posed part: the fields uniform gains a non-identity
                // placement and the volume light turns into the lattice.
                scene.model = Mat4::from_rotation_y(0.7) * Mat4::from_translation(Vec3::X * 3.0);
            }
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test") });
            r.render(
                &mut enc,
                FrameInput {
                    camera: &cam,
                    scene: &scene,
                    // Derive on the first frame only, exercising both the
                    // "solver stepped" and "render-only" paths.
                    sources: (frame == 0).then(|| FieldSources {
                        macro_view: &mac_view,
                        flags_view: Some(&flg_view),
                        scales,
                    }),
                    sdf: None,
                    target: &target_view,
                    dt: 1.0 / 60.0,
                },
            )
            .expect("frame recording");
            gpu.queue.submit([enc.finish()]);
            r.after_submit();
            scene.end_frame();
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    #[test]
    fn a_rendered_frame_is_not_uniformly_black() {
        // Weak-looking, but it is exactly the assertion that catches a renderer
        // that compiles, validates, runs — and draws nothing. The background
        // gradient alone guarantees a non-constant image, so both the mean and
        // the variance must be non-zero.
        let Some(gpu) = gpu() else { return };
        let grid = test_grid();
        let (w, h) = (128u32, 96u32);
        let mut r = Renderer::new(
            &gpu,
            RendererConfig::new(w, h, wgpu::TextureFormat::Rgba8Unorm, grid),
        )
        .expect("renderer construction");
        // The background gradient is the thing under test, so turn off the
        // dither that would otherwise contribute a little variance of its own.
        r.post_settings_mut().dither = 0.0;

        let target = util::color_target(
            &gpu.device,
            "readback target",
            w,
            h,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let target_view = target.create_view(&Default::default());

        let mut cam = Camera::default();
        cam.frame_bbox(Bbox { min: Vec3::splat(-24.0), max: Vec3::splat(24.0) }, 0.1);
        // Look slightly downward so the ground plane is in frame.
        cam.pitch = 0.35;

        let scene = Scene::new();
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        r.render(
            &mut enc,
            FrameInput {
                camera: &cam,
                scene: &scene,
                sources: None,
                sdf: None,
                target: &target_view,
                dt: 1.0 / 60.0,
            },
        )
        .expect("frame recording");

        let pixels = util::readback_rgba8(&gpu.device, &gpu.queue, &target, w, h, enc);
        assert_eq!(pixels.len(), (w * h * 4) as usize);

        let lum: Vec<f32> = pixels
            .chunks_exact(4)
            .map(|p| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32)
            .collect();
        let mean = lum.iter().sum::<f32>() / lum.len() as f32;
        let var = lum.iter().map(|l| (l - mean) * (l - mean)).sum::<f32>() / lum.len() as f32;

        assert!(mean > 0.5, "the frame is black: mean luminance {mean}");
        assert!(mean < 250.0, "the frame is blown out: mean luminance {mean}");
        assert!(var > 0.05, "the frame is a flat fill: luminance variance {var}");
        assert!(pixels.chunks_exact(4).all(|p| p[3] == 255), "alpha must be opaque");
    }

    #[test]
    fn resize_and_supersample_reallocate_cleanly() {
        let Some(gpu) = gpu() else { return };
        let mut r = Renderer::new(
            &gpu,
            RendererConfig::new(256, 256, wgpu::TextureFormat::Rgba8Unorm, test_grid()),
        )
        .expect("renderer construction");
        assert_eq!(r.size(), (256, 256));
        r.resize(400, 300);
        assert_eq!(r.size(), (400, 300));
        assert_eq!(r.begin_supersample(2), (800, 600));
        r.end_supersample();
        assert_eq!(r.size(), (400, 300));
    }

    // -- synthetic solver state, for the numerical GPU tests ------------------

    /// Grid used by the numeric tests. `x = 32` makes an `Rgba16Float` row
    /// exactly 256 bytes, which is the `copy_texture_to_buffer` row alignment.
    fn numeric_grid() -> Grid {
        Grid { dims: UVec3::new(32, 24, 24), dx_mm: 0.75, origin_mm: Vec3::splat(-8.0) }
    }

    /// A Taylor-Green vortex in lattice units, plus a density ripple. Chosen
    /// because it has genuine rotation *and* genuine strain, so Q-criterion is
    /// non-trivial and both halves of `Q = (|Omega|^2 - |S|^2)/2` matter.
    fn synthetic_macro(dims: UVec3) -> Vec<[f32; 4]> {
        let k = std::f32::consts::TAU / dims.x as f32;
        let mut out = Vec::with_capacity((dims.x * dims.y * dims.z) as usize);
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let (fx, fy, fz) = (x as f32 * k, y as f32 * k * 1.3, z as f32 * k * 0.7);
                    out.push([
                        0.06 * fx.sin() * fy.cos() * fz.cos(),
                        -0.06 * fx.cos() * fy.sin() * fz.cos(),
                        0.015 * (fx + fz).sin(),
                        0.002 * (fx * 0.5).cos(),
                    ]);
                }
            }
        }
        out
    }

    /// A single swirling blob in the middle of an otherwise still domain.
    ///
    /// The empty-space-skipping tests need a *sparse* field: a Taylor-Green
    /// vortex fills every brick with the whole range of speeds, so binarising it
    /// marks everything active and the distance transform never gets exercised.
    fn sparse_macro(dims: UVec3) -> Vec<[f32; 4]> {
        let c = dims.as_vec3() * 0.5;
        let radius = dims.x as f32 * 0.18;
        let mut out = Vec::with_capacity((dims.x * dims.y * dims.z) as usize);
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let p = Vec3::new(x as f32, y as f32, z as f32) - c;
                    let g = (-(p.length_squared()) / (radius * radius)).exp();
                    let tangent = Vec3::new(-p.y, p.x, p.z * 0.2).normalize_or_zero();
                    let u = tangent * 0.08 * g;
                    out.push([u.x, u.y, u.z, 0.001 * g]);
                }
            }
        }
        out
    }

    /// A solid slab through the middle of the domain, so the wall masking in
    /// `derive.wgsl` is actually exercised.
    fn synthetic_flags(dims: UVec3) -> Vec<u8> {
        let mut f = vec![ad_gpu::flags::FLUID; (dims.x * dims.y * dims.z) as usize];
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    if (8..11).contains(&y) && x >= 4 && x < dims.x - 4 {
                        f[((z * dims.y + y) * dims.x + x) as usize] = ad_gpu::flags::SOLID;
                    }
                    let _ = z;
                }
            }
        }
        f
    }

    /// Round a value through f16, so the CPU reference reads exactly what the
    /// shader reads out of an `Rgba16Float` texture.
    fn quantise(v: f32) -> f32 {
        colormap::f16_bits_to_f32(colormap::f32_to_f16_bits(v))
    }

    fn upload_macro(queue: &wgpu::Queue, tex: &wgpu::Texture, dims: UVec3, data: &[[f32; 4]]) {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for t in data {
            for c in t {
                bytes.extend_from_slice(&colormap::f32_to_f16_bits(*c).to_le_bytes());
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims.x * 8),
                rows_per_image: Some(dims.y),
            },
            wgpu::Extent3d {
                width: dims.x,
                height: dims.y,
                depth_or_array_layers: dims.z,
            },
        );
    }

    fn upload_flags(queue: &wgpu::Queue, tex: &wgpu::Texture, dims: UVec3, data: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims.x),
                rows_per_image: Some(dims.y),
            },
            wgpu::Extent3d {
                width: dims.x,
                height: dims.y,
                depth_or_array_layers: dims.z,
            },
        );
    }

    /// The CPU twin of `derive.wgsl`, including the wall masking. Written
    /// against `fields::q_criterion` / `fields::vorticity`, which are themselves
    /// pinned by the analytic-flow tests in `fields`.
    fn cpu_derive(
        data: &[[f32; 4]],
        flags: &[u8],
        dims: UVec3,
        scales: DeriveScales,
    ) -> Vec<[f32; 4]> {
        let idx = |c: UVec3| ((c.z * dims.y + c.y) * dims.x + c.x) as usize;
        let in_domain =
            |c: glam::IVec3| c.cmpge(glam::IVec3::ZERO).all() && c.cmplt(dims.as_ivec3()).all();
        let usable = |c: glam::IVec3| {
            in_domain(c) && ad_gpu::flags::is_fluid(flags[idx(c.as_uvec3())])
        };
        let vel = |c: glam::IVec3| {
            let t = data[idx(c.as_uvec3())];
            Vec3::new(quantise(t[0]), quantise(t[1]), quantise(t[2]))
        };

        let mut out = vec![[0.0f32; 4]; data.len()];
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let c = glam::IVec3::new(x as i32, y as i32, z as i32);
                    if !ad_gpu::flags::is_fluid(flags[idx(c.as_uvec3())]) {
                        continue;
                    }
                    let u0 = vel(c);
                    let axis = |a: glam::IVec3| {
                        let (cp, cm) = (c + a, c - a);
                        match (usable(cp), usable(cm)) {
                            (true, true) => (vel(cp) - vel(cm)) * 0.5,
                            (true, false) => vel(cp) - u0,
                            (false, true) => u0 - vel(cm),
                            (false, false) => Vec3::ZERO,
                        }
                    };
                    let j = glam::Mat3::from_cols(
                        axis(glam::IVec3::X),
                        axis(glam::IVec3::Y),
                        axis(glam::IVec3::Z),
                    );
                    out[idx(c.as_uvec3())] = [
                        u0.length() * scales.speed_ms,
                        fields::q_criterion(j) * scales.q_tilde,
                        fields::vorticity(j).length() * scales.vorticity,
                        quantise(data[idx(c.as_uvec3())][3]) * scales.pressure_pa,
                    ];
                }
            }
        }
        out
    }

    #[test]
    fn the_derive_pass_matches_the_cpu_formulas_including_wall_masking() {
        // The one test that pins `derive.wgsl` numerically. If the shader ever
        // transposes J, drops the wall masking, or applies a scale to the wrong
        // channel, this is what catches it — and it catches it against the same
        // `q_criterion` that the analytic-flow tests in `fields` verify.
        let Some(gpu) = gpu() else { return };
        let grid = numeric_grid();
        let dims = grid.dims;
        let loader = util::shader_loader();

        let mut f = DerivedFields::new(
            &gpu.device,
            &gpu.queue,
            &loader,
            grid,
            FieldResolution::Full,
        )
        .unwrap();

        let (mac, flg) = fields::create_source_textures(&gpu.device, grid);
        let data = synthetic_macro(dims);
        let flags = synthetic_flags(dims);
        upload_macro(&gpu.queue, &mac, dims, &data);
        upload_flags(&gpu.queue, &flg, dims, &flags);
        let d3 = wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        };
        let mac_view = mac.create_view(&d3);
        let flg_view = flg.create_view(&d3);

        let units = ad_gpu::LatticeUnits::for_air(grid.dx_mm as f64, 5.0, 0.05);
        let scales = DeriveScales::new(&units, 6.3);
        let mut profiler = Profiler::new(&gpu.device, &gpu.queue, 4, false, None);

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        f.derive(
            &gpu.device,
            &gpu.queue,
            &mut enc,
            &mut profiler,
            &FieldSources { macro_view: &mac_view, flags_view: Some(&flg_view), scales },
        );

        let raw = util::readback_texture(
            &gpu.device,
            &gpu.queue,
            f.scalar_texture(),
            wgpu::Extent3d {
                width: dims.x,
                height: dims.y,
                depth_or_array_layers: dims.z,
            },
            8,
            enc,
        );
        assert_eq!(raw.len(), (dims.x * dims.y * dims.z) as usize * 8);

        let expect = cpu_derive(&data, &flags, dims, scales);
        let mut worst = [0.0f32; 4];
        let mut checked = 0usize;
        for i in 0..expect.len() {
            if !ad_gpu::flags::is_fluid(flags[i]) {
                continue;
            }
            checked += 1;
            for c in 0..4 {
                let got = colormap::f16_bits_to_f32(u16::from_le_bytes([
                    raw[i * 8 + c * 2],
                    raw[i * 8 + c * 2 + 1],
                ]));
                let want = expect[i][c];
                // Relative, with an absolute floor scaled to the channel's own
                // range: `f16` output has ~1e-3 relative precision.
                let scale = want.abs().max(0.02 * expect.iter().map(|e| e[c].abs()).fold(0.0, f32::max));
                let err = (got - want).abs() / scale.max(1e-12);
                worst[c] = worst[c].max(err);
            }
        }
        assert!(checked > 1000, "the test grid produced only {checked} fluid cells");
        for (c, name) in ["speed", "Q~", "|omega|", "pressure"].iter().enumerate() {
            assert!(
                worst[c] < 0.02,
                "channel {name} differs from the CPU reference by {} relative",
                worst[c]
            );
        }
    }

    #[test]
    fn the_gpu_chebyshev_transform_matches_the_cpu_reference() {
        // The GPU dilation and `accel::chebyshev_distance_transform` implement
        // the same recurrence; the CPU one is checked against brute force in
        // `accel`. Chaining the two makes the shader verified rather than
        // merely plausible.
        let Some(gpu) = gpu() else { return };
        // A larger, sparser setup than the derive test: enough bricks for the
        // transform to have somewhere to propagate, and a field concentrated
        // enough that most of them are empty.
        let grid = Grid {
            dims: UVec3::new(64, 48, 48),
            dx_mm: 0.75,
            origin_mm: Vec3::splat(-16.0),
        };
        let dims = grid.dims;
        let loader = util::shader_loader();

        let mut f =
            DerivedFields::new(&gpu.device, &gpu.queue, &loader, grid, FieldResolution::Full)
                .unwrap();
        let mut bricks = BrickGrid::new(&gpu.device, &loader, &f).unwrap();

        let (mac, flg) = fields::create_source_textures(&gpu.device, grid);
        upload_macro(&gpu.queue, &mac, dims, &sparse_macro(dims));
        upload_flags(&gpu.queue, &flg, dims, &vec![ad_gpu::flags::FLUID; (dims.x * dims.y * dims.z) as usize]);
        let d3 = wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        };
        let mac_view = mac.create_view(&d3);
        let flg_view = flg.create_view(&d3);

        let units = ad_gpu::LatticeUnits::for_air(grid.dx_mm as f64, 5.0, 0.05);
        let scales = DeriveScales::new(&units, 6.3);
        let mut profiler = Profiler::new(&gpu.device, &gpu.queue, 8, false, None);

        // A soft isosurface near the peak speed, so only the bricks around the
        // blob's shell survive binarisation.
        let mut tf = TransferFunction::preset(DerivedField::Speed);
        tf.mode = OpacityMode::SoftIso;
        tf.iso = SoftIso { center: 5.0, width: 0.25, amplitude: 0.9, cutoff_widths: 2.5 };
        tf.range = [0.0, 9.0];
        tf.sanitise();

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        f.derive(
            &gpu.device,
            &gpu.queue,
            &mut enc,
            &mut profiler,
            &FieldSources { macro_view: &mac_view, flags_view: Some(&flg_view), scales },
        );
        assert!(bricks.update(&gpu.queue, &mut enc, &mut profiler, &f, &tf));

        let bd = bricks.dims();
        let extent = wgpu::Extent3d {
            width: bd.x,
            height: bd.y,
            depth_or_array_layers: bd.z,
        };
        // Both textures come from the same encoder submission, so the min/max
        // read back is exactly the one the seed pass binarised.
        let minmax_raw =
            util::readback_texture(&gpu.device, &gpu.queue, bricks.minmax_texture(), extent, 8, enc);
        let enc2 = gpu.device.create_command_encoder(&Default::default());
        let dist_raw = util::readback_texture(
            &gpu.device,
            &gpu.queue,
            bricks.distance_texture(),
            extent,
            4,
            enc2,
        );

        let n = (bd.x * bd.y * bd.z) as usize;
        assert_eq!(minmax_raw.len(), n * 8);
        assert_eq!(dist_raw.len(), n * 4);

        let support = tf.support().expect("the test transfer function must have support");
        let active: Vec<bool> = (0..n)
            .map(|i| {
                let mn = f32::from_le_bytes(minmax_raw[i * 8..i * 8 + 4].try_into().unwrap());
                let mx = f32::from_le_bytes(minmax_raw[i * 8 + 4..i * 8 + 8].try_into().unwrap());
                support.intersects(mn, mx)
            })
            .collect();

        let active_count = active.iter().filter(|a| **a).count();
        assert!(active_count > 0, "no brick is active; the test proves nothing");
        assert!(
            active_count < n,
            "every brick is active; the distance transform is untested"
        );

        let expect = accel::chebyshev_distance_transform(&active, bd, accel::MAX_SKIP);
        let reach = expect.iter().copied().max().unwrap_or(0);
        assert!(
            reach > 1,
            "no brick is more than one step from an active one; the dilation loop is untested"
        );
        let got: Vec<u32> = (0..n)
            .map(|i| u32::from_le_bytes(dist_raw[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();
        assert_eq!(got, expect, "GPU Chebyshev transform disagrees with the CPU reference");
    }

    #[test]
    fn empty_space_skipping_does_not_change_the_image() {
        // The headline invariant, and the reason the opacity correction is
        // mandatory: with the correction in place, the accelerator is an
        // optimisation and nothing else, so skipping on and skipping off must
        // produce the same picture. Any difference is a bug in one of them.
        let Some(gpu) = gpu() else { return };
        let grid = numeric_grid();
        let dims = grid.dims;
        let (w, h) = (192u32, 144u32);

        let (mac, flg) = fields::create_source_textures(&gpu.device, grid);
        upload_macro(&gpu.queue, &mac, dims, &synthetic_macro(dims));
        upload_flags(&gpu.queue, &flg, dims, &synthetic_flags(dims));
        let d3 = wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        };
        let mac_view = mac.create_view(&d3);
        let flg_view = flg.create_view(&d3);
        let units = ad_gpu::LatticeUnits::for_air(grid.dx_mm as f64, 5.0, 0.05);
        let scales = DeriveScales::new(&units, 6.3);

        let mut cam = Camera::default();
        cam.frame_bbox(grid.bbox(), 0.05);
        cam.pitch = 0.4;

        let render_once = |skip: bool| -> Vec<u8> {
            let mut r = Renderer::new(
                &gpu,
                RendererConfig {
                    width: w,
                    height: h,
                    target_format: wgpu::TextureFormat::Rgba8Unorm,
                    grid,
                    field_resolution: FieldResolution::Full,
                    profiling: false,
                },
            )
            .unwrap();
            // Everything temporal or stochastic off, so the comparison is of
            // the raymarch and nothing else.
            {
                let s = r.post_settings_mut();
                s.taa = false;
                s.bloom = false;
                s.ssao = false;
                s.dither = 0.0;
            }
            r.set_field(DerivedField::Speed);
            {
                let tf = r.transfer_function_mut();
                tf.mode = OpacityMode::SoftIso;
                tf.iso = SoftIso { center: 2.0, width: 0.35, amplitude: 0.9, cutoff_widths: 2.5 };
                tf.range = [0.0, 6.0];
                tf.sanitise();
            }
            let mut vs = *r.volume_settings();
            vs.skip_empty_space = skip;
            vs.resolution_divisor = 1;
            r.set_volume_settings(vs);

            let target = util::color_target(
                &gpu.device,
                "skip test",
                w,
                h,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            let view = target.create_view(&Default::default());
            let scene = Scene::new();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            r.render(
                &mut enc,
                FrameInput {
                    camera: &cam,
                    scene: &scene,
                    sources: Some(FieldSources {
                        macro_view: &mac_view,
                        flags_view: Some(&flg_view),
                        scales,
                    }),
                    sdf: None,
                    target: &view,
                    dt: 1.0 / 60.0,
                },
            )
            .unwrap();
            util::readback_rgba8(&gpu.device, &gpu.queue, &target, w, h, enc)
        };

        let with_skip = render_once(true);
        let without = render_once(false);
        assert_eq!(with_skip.len(), without.len());

        // The volume must actually be visible, or the test is comparing two
        // pictures of the background.
        let variance = {
            let l: Vec<f32> = with_skip.chunks_exact(4).map(|p| p[0] as f32).collect();
            let m = l.iter().sum::<f32>() / l.len() as f32;
            l.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / l.len() as f32
        };
        assert!(variance > 5.0, "nothing was drawn; variance {variance}");

        let mut worst = 0i32;
        let mut differing = 0usize;
        for (a, b) in with_skip.iter().zip(&without) {
            let d = (*a as i32 - *b as i32).abs();
            worst = worst.max(d);
            if d > 0 {
                differing += 1;
            }
        }
        let frac = differing as f64 / with_skip.len() as f64;
        // Not bit-exact: skipping lands the ray on a different sub-step phase,
        // so a sample can straddle a slightly different point. The opacity
        // correction is what keeps that difference at the quantisation floor
        // instead of turning it into a systematic density change.
        assert!(
            worst <= 2,
            "skipping changed a pixel by {worst}/255 -- the accelerator or the opacity correction is wrong"
        );
        assert!(frac < 0.25, "{:.1}% of pixels changed", frac * 100.0);
    }

    #[test]
    fn changing_the_field_swaps_the_transfer_function_preset() {
        let Some(gpu) = gpu() else { return };
        let mut r = Renderer::new(
            &gpu,
            RendererConfig::new(64, 64, wgpu::TextureFormat::Rgba8Unorm, test_grid()),
        )
        .expect("renderer construction");
        assert_eq!(r.field(), DerivedField::Speed);
        r.set_field(DerivedField::QCriterion);
        assert_eq!(r.field(), DerivedField::QCriterion);
        // Q-criterion must land in soft-isosurface mode on a dimensionless
        // range: that is the whole point of normalising it.
        assert_eq!(r.transfer_function().mode, OpacityMode::SoftIso);
        assert_eq!(r.transfer_function().range, [0.0, 2.0]);

        r.set_field(DerivedField::Pressure);
        // Pressure is signed, so it must come up on a diverging map with the
        // symmetric lock already on.
        assert!(r.transfer_function().symmetric_lock);
        assert_eq!(r.transfer_function().map.kind(), MapKind::Diverging);
    }

    #[test]
    fn switching_fields_remembers_each_transfer_function() {
        // Tuning an isolevel takes a minute; throwing it away on every field
        // switch is what makes people stop switching fields.
        let Some(gpu) = gpu() else { return };
        let mut r = Renderer::new(
            &gpu,
            RendererConfig::new(64, 64, wgpu::TextureFormat::Rgba8Unorm, test_grid()),
        )
        .expect("renderer construction");

        r.set_field(DerivedField::QCriterion);
        r.transfer_function_mut().iso.center = 1.234;
        r.set_field(DerivedField::Pressure);
        assert_ne!(r.transfer_function().mode, OpacityMode::SoftIso);
        r.set_field(DerivedField::QCriterion);
        assert_eq!(r.transfer_function().iso.center, 1.234, "the tuning was lost");

        // ...and the other fields are visible without switching to them.
        assert!(r.transfer_function_for(DerivedField::Pressure).symmetric_lock);
        r.reset_transfer_function(DerivedField::QCriterion);
        assert_eq!(
            r.transfer_function().iso.center,
            TransferFunction::preset(DerivedField::QCriterion).iso.center
        );
    }

    #[test]
    fn one_light_direction_reaches_all_three_shading_paths() {
        // Deferred shading, the ghost shell and the volume's gradient shading
        // must agree about where the key light is, or a translucent duct looks
        // structurally wrong rather than merely oddly lit.
        let Some(gpu) = gpu() else { return };
        let grid = test_grid();
        let mut r = Renderer::new(
            &gpu,
            RendererConfig::new(64, 64, wgpu::TextureFormat::Rgba8Unorm, grid),
        )
        .expect("renderer construction");

        let dir = Vec3::new(-0.3, 0.2, 0.93).normalize();
        r.post_settings_mut().lighting.key_dir = dir;

        let target = util::color_target(
            &gpu.device,
            "light sync",
            64,
            64,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let view = target.create_view(&Default::default());
        let scene = Scene::new();
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        r.render(
            &mut enc,
            FrameInput {
                camera: &Camera::default(),
                scene: &scene,
                sources: None,
                sdf: None,
                target: &view,
                dt: 1.0 / 60.0,
            },
        )
        .unwrap();
        gpu.queue.submit([enc.finish()]);

        assert!(
            (r.volume_settings().light_dir - dir).length() < 1e-5,
            "the volume is lit from {} instead of {dir}",
            r.volume_settings().light_dir
        );
        assert!((r.mesh.light_dir - dir).length() < 1e-5, "the mesh is lit from elsewhere");
    }

    #[test]
    fn the_accessibility_toggle_replaces_unsafe_maps() {
        let Some(gpu) = gpu() else { return };
        let mut r = Renderer::new(
            &gpu,
            RendererConfig::new(64, 64, wgpu::TextureFormat::Rgba8Unorm, test_grid()),
        )
        .expect("renderer construction");
        r.transfer_function_mut().map = ColorMap::Turbo;
        assert!(!r.transfer_function().map.cvd_safe());
        r.set_accessible_colors(true);
        assert!(r.accessible_colors());
        // The stored choice is preserved -- only what gets baked into the LUT
        // changes, so turning the toggle off restores the user's map.
        assert_eq!(r.transfer_function().map, ColorMap::Turbo);
        assert!(r.transfer_function().map.accessible_substitute().cvd_safe());
    }
}

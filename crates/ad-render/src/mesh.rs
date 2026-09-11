//! Duct and obstruction geometry: G-buffer, ghost shell, wireframe.
//!
//! # Display modes
//!
//! `solid -> ghost -> wireframe -> off`, cycled with one key. All four get used:
//! solid to judge the part, ghost to watch flow inside it, wireframe to check
//! that an STL actually meshed the way you think, off to look at nothing but the
//! flow.
//!
//! **Ghost** is the one that matters. It culls *front* faces and draws only back
//! faces, which gives exactly one transparent layer per pixel with no sorting
//! and no depth peeling.
//!
//! It deliberately writes **no depth**. An earlier version ran a depth prepass
//! on the theory that front-face culling leaves "the far wall" in the depth
//! buffer, so the volume would clip there and the flow inside would show. That
//! is only true of a solid convex body. A duct is a hollow shell, so the nearest
//! back-facing surface is the *inner* face of the near wall, and the volume
//! clipped at the start of the bore -- hiding precisely the flow the mode
//! exists to reveal. See the comment on the prepass pipeline below. Fresnel weighting
//! keeps grazing angles opaque and face-on angles nearly clear, so the shell
//! reads as a solid object rather than a stain, and a rim light picks out the
//! silhouette, which is the only place a transparent surface has enough contrast
//! to show its shape.
//!
//! **Wireframe** draws a de-duplicated edge list as a line list rather than
//! using `PolygonMode::Line`, which needs an optional wgpu feature `ad-gpu` does
//! not request. Real edges also look better than triangle outlines.
//!
//! # Why the material has a clearcoat
//!
//! A single Lambert-plus-GGX lobe on a mid-grey albedo renders as a grey blob,
//! and a grey blob is very hard to judge shape on. A second, much smoother
//! specular lobe over the top — a clearcoat, IOR 1.5, untinted — is what makes a
//! surface read as moulded or printed *plastic*. It costs one extra GGX
//! evaluation and it is the difference between a screenshot people trust and one
//! they squint at. The lobe is evaluated in `composite.wgsl`, since shading is
//! deferred.

use ad_gpu::{Bbox, Profiler, ShaderDefines, ShaderLoader};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::HashSet;


/// G-buffer formats.
pub const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// World normal in `xyz`, the per-vertex scalar channel in `w`.
pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Screen-space motion in NDC units.
pub const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// Reverse-Z depth. `Depth32Float` because reverse-Z spends its precision in the
/// float exponent, and a 24-bit integer format cannot express that.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Depth comparison for reverse-Z. Everything in this crate that touches depth
/// uses this constant rather than writing `Greater` inline, so there is exactly
/// one place to look when something z-fights.
pub const DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::Greater;
/// Clear value for a reverse-Z depth buffer: the far plane.
pub const DEPTH_CLEAR: f32 = 0.0;

/// Uniform buffer stride per mesh. Must be at least
/// `min_uniform_buffer_offset_alignment`, which is 256 on every desktop adapter.
const MESH_UNIFORM_STRIDE: u64 = 256;
/// Meshes that can be drawn in one pass. A duct plus a handful of obstructions.
pub const MAX_MESHES: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct MeshVertex {
    /// Model space, millimetres.
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Per-vertex scalar carried into the G-buffer, for later wall-quantity
    /// mapping (wall shear, wall pressure). Zero when unused.
    pub scalar: f32,
}

impl MeshVertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32],
    };
}

/// Plain CPU-side geometry. Deliberately not a geometry-crate type: the render
/// core takes slices, so the geometry crate can hand it whatever it already has.
#[derive(Debug, Clone, Copy)]
pub struct MeshData<'a> {
    pub vertices: &'a [MeshVertex],
    pub indices: &'a [u32],
}

/// Per-mesh appearance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshStyle {
    /// Linear-light albedo.
    pub albedo: Vec3,
    pub roughness: f32,
    pub clearcoat: f32,
    /// Fresnel exponent for the ghost mode. Lower = more uniformly opaque.
    pub fresnel_power: f32,
    pub rim_strength: f32,
    /// Ghost opacity face-on and at grazing angles.
    pub ghost_alpha: (f32, f32),
    pub wire_alpha: f32,
}

impl Default for MeshStyle {
    fn default() -> Self {
        Self {
            // A light warm grey. Neutral enough not to bias a colour-mapped
            // overlay, light enough that the clearcoat has something to sit on.
            albedo: Vec3::new(0.42, 0.43, 0.46),
            roughness: 0.38,
            clearcoat: 0.6,
            fresnel_power: 2.2,
            rim_strength: 0.35,
            // Face-on is nearly clear so the flow reads through it; grazing
            // angles stay solid enough to describe the silhouette. Pushing the
            // grazing value much past this turns the shell into a milky bag
            // that the volume has to fight through.
            ghost_alpha: (0.035, 0.42),
            wire_alpha: 0.35,
        }
    }
}

/// How the geometry is drawn. Cycled with a single control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshDisplay {
    #[default]
    Solid,
    Ghost,
    Wireframe,
    Off,
}

impl MeshDisplay {
    pub fn cycle(self) -> Self {
        match self {
            MeshDisplay::Solid => MeshDisplay::Ghost,
            MeshDisplay::Ghost => MeshDisplay::Wireframe,
            MeshDisplay::Wireframe => MeshDisplay::Off,
            MeshDisplay::Off => MeshDisplay::Solid,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            MeshDisplay::Solid => "solid",
            MeshDisplay::Ghost => "ghost",
            MeshDisplay::Wireframe => "wireframe",
            MeshDisplay::Off => "off",
        }
    }
    /// Solid mode is the only one that fills albedo and normals, and therefore
    /// the only one the deferred shading pass draws.
    pub fn writes_gbuffer(self) -> bool {
        self == MeshDisplay::Solid
    }
    /// Ghost mode fills **depth and motion only**, from back faces.
    ///
    /// That sounds like a contradiction - ghosting exists so the flow is not
    /// clipped by the shell - but it is the far wall that gets written, not the
    /// near one. The ray enters through the invisible near wall, marches the
    /// flow inside, and stops at the far wall, which is exactly the occlusion a
    /// translucent-but-solid object should produce. Without it the volume keeps
    /// marching out through the back of the duct and the ghost tints the result
    /// uniformly, which reads as fog rather than as a shell.
    ///
    /// The motion vectors come along because TAA needs them: a shell region with
    /// zero motion smears the moment the camera turns.
    pub fn writes_depth_prepass(self) -> bool {
        self == MeshDisplay::Ghost
    }
    pub fn draws_overlay(self) -> bool {
        matches!(self, MeshDisplay::Ghost | MeshDisplay::Wireframe)
    }
}

/// Build a de-duplicated edge list from a triangle index buffer.
///
/// Every interior edge is shared by two triangles, so drawing triangle outlines
/// directly would submit each line twice — visibly heavier where two faces meet,
/// and twice the vertex work.
pub fn edge_list(indices: &[u32]) -> Vec<u32> {
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(indices.len());
    let mut out = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            if seen.insert(key) {
                out.push(key.0);
                out.push(key.1);
            }
        }
    }
    out
}

/// GPU-resident geometry.
pub struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    edges: wgpu::Buffer,
    edge_count: u32,
    pub style: MeshStyle,
    pub bbox: Bbox,
    pub visible: bool,
    /// This mesh's own transform, applied before the scene's. Identity for
    /// everything built from the scene the solver runs on; set while an
    /// obstruction is being moved, so it follows the mouse on screen before
    /// the lattice has been rebuilt around it.
    pub model: Mat4,
    pub(crate) prev_model: Mat4,
}

impl GpuMesh {
    pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue, data: MeshData<'_>, style: MeshStyle) -> Self {
        let edges = edge_list(data.indices);
        let bbox = Bbox::from_points(data.vertices.iter().map(|v| Vec3::from_array(v.position)));

        let make = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
            // Buffers must be non-empty; a degenerate mesh gets a 4-byte stub
            // and a zero draw count rather than a validation error.
            let size = (bytes.len().max(4) as u64).next_multiple_of(4);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !bytes.is_empty() {
                queue.write_buffer(&buf, 0, bytes);
            }
            buf
        };

        Self {
            vertices: make("mesh vertices", bytemuck::cast_slice(data.vertices), wgpu::BufferUsages::VERTEX),
            indices: make("mesh indices", bytemuck::cast_slice(data.indices), wgpu::BufferUsages::INDEX),
            index_count: data.indices.len() as u32,
            edges: make("mesh edges", bytemuck::cast_slice(&edges), wgpu::BufferUsages::INDEX),
            edge_count: edges.len() as u32,
            style,
            bbox,
            visible: true,
            model: Mat4::IDENTITY,
            prev_model: Mat4::IDENTITY,
        }
    }

    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }
    pub fn edge_count(&self) -> u32 {
        self.edge_count / 2
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MeshUniform {
    model: [[f32; 4]; 4],
    prev_model: [[f32; 4]; 4],
    base_color: [f32; 4],
    params: [f32; 4],
    light_dir: [f32; 4],
    ghost: [f32; 4],
}

pub struct MeshRenderer {
    uniform: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
    gbuffer_pipeline: wgpu::RenderPipeline,
    ghost_depth_pipeline: wgpu::RenderPipeline,
    ghost_pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    pub display: MeshDisplay,
    pub light_dir: Vec3,
    pub light_intensity: f32,
}

impl MeshRenderer {
    pub fn new(
        device: &wgpu::Device,
        loader: &ShaderLoader,
        camera_layout: &wgpu::BindGroupLayout,
        hdr_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let stages = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    // One slot per mesh, selected with a dynamic offset, so a
                    // duct and its obstructions can have different materials
                    // inside a single render pass.
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<MeshUniform>() as u64
                    ),
                },
                count: None,
            }],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh uniforms"),
            size: MESH_UNIFORM_STRIDE * MAX_MESHES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<MeshUniform>() as u64),
                }),
            }],
        });

        let defines = ShaderDefines::new();
        let module = loader.create_module(device, "mesh.wgsl", &defines)?;
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh"),
            bind_group_layouts: &[Some(camera_layout), Some(&layout)],
            immediate_size: 0,
        });

        let depth_write = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(DEPTH_COMPARE),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let depth_test_only = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(DEPTH_COMPARE),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh g-buffer"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_mesh"),
                compilation_options: Default::default(),
                buffers: &[Some(MeshVertex::LAYOUT)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // No back-face culling: the camera is expected to go *inside*
                // the duct, and culling would open a hole in the wall the moment
                // it does. The fragment shader flips the normal instead.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_write),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_gbuffer"),
                compilation_options: Default::default(),
                targets: &[
                    Some(ALBEDO_FORMAT.into()),
                    Some(NORMAL_FORMAT.into()),
                    Some(MOTION_FORMAT.into()),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Ghost mode's depth-and-motion prepass: the same vertex work, front
        // faces culled so only the far wall lands in depth, and the colour write
        // masks off everywhere except motion. Leaving the *normal* target
        // cleared is what tells `composite.wgsl` not to shade these pixels: a
        // populated G-buffer fragment always has a unit normal.
        let masked = |format: wgpu::TextureFormat, write: bool| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: if write { wgpu::ColorWrites::ALL } else { wgpu::ColorWrites::empty() },
            })
        };
        let ghost_depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh ghost depth"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_mesh"),
                compilation_options: Default::default(),
                buffers: &[Some(MeshVertex::LAYOUT)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            // Motion vectors only. The prepass must NOT write depth.
            //
            // # Why this was wrong, and why it matters most for a duct
            //
            // The original reasoning was: cull front faces, so "only the far
            // wall lands in depth", and the volume then clips against the far
            // wall and renders the flow inside. That holds for a *solid convex*
            // body, where the back faces really are the far side.
            //
            // A duct is not that. It is a thin hollow shell, so a view ray
            // crosses four surfaces: the near wall's outer face (front-facing,
            // culled), the near wall's INNER face (back-facing, drawn), the far
            // wall's inner face (culled) and its outer face (drawn). With
            // reverse-Z and `CompareFunction::Greater` the nearest fragment
            // wins, and the nearest one drawn is the near wall's *inner*
            // surface -- which is exactly where the passage begins.
            //
            // So the volume raymarch clipped at the start of the duct bore and
            // the entire internal flow was invisible. Measured directly:
            // rendering the same converged field with the mesh off showed the
            // passage in full, and with ghost showed a featureless shell. The
            // ghost alpha was never the problem -- it is 0.035 face-on, nearly
            // clear.
            //
            // Not writing depth means the volume is no longer occluded by the
            // shell at all, so flow behind the duct shows through it. For a
            // 3.5%-alpha shell whose entire purpose is "watch the flow inside",
            // that is the right trade: seeing into the part is the feature, and
            // a duct designer is looking at the bore, not judging occlusion.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_gbuffer"),
                compilation_options: Default::default(),
                targets: &[
                    masked(ALBEDO_FORMAT, false),
                    masked(NORMAL_FORMAT, false),
                    masked(MOTION_FORMAT, true),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Premultiplied "over". The fragment shaders return `rgb * a`, so this
        // is the same arithmetic the volume compositor does by hand.
        let premultiplied = wgpu::ColorTargetState {
            format: hdr_format,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let ghost_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh ghost"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_mesh"),
                compilation_options: Default::default(),
                buffers: &[Some(MeshVertex::LAYOUT)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Front faces culled: one transparent layer, no depth peeling.
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            // GreaterEqual, not Greater: these are the same triangles the depth
            // prepass just wrote, so their depth is bit-identical to what is in
            // the buffer and a strict test would reject every fragment.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_ghost"),
                compilation_options: Default::default(),
                targets: &[Some(premultiplied.clone())],
            }),
            multiview_mask: None,
            cache: None,
        });

        let wire_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh wireframe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_mesh"),
                compilation_options: Default::default(),
                buffers: &[Some(MeshVertex::LAYOUT)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_test_only),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_wire"),
                compilation_options: Default::default(),
                targets: &[Some(premultiplied)],
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            uniform,
            layout,
            group,
            gbuffer_pipeline,
            ghost_depth_pipeline,
            ghost_pipeline,
            wire_pipeline,
            display: MeshDisplay::default(),
            light_dir: Vec3::new(0.45, 0.75, 0.48).normalize(),
            light_intensity: 1.0,
        })
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Upload one uniform slot per mesh.
    pub fn upload(&self, queue: &wgpu::Queue, meshes: &[GpuMesh], model: Mat4, prev_model: Mat4) {
        for (i, mesh) in meshes.iter().take(MAX_MESHES).enumerate() {
            let u = MeshUniform {
                model: (model * mesh.model).to_cols_array_2d(),
                prev_model: (prev_model * mesh.prev_model).to_cols_array_2d(),
                base_color: mesh.style.albedo.extend(mesh.style.wire_alpha).to_array(),
                params: [
                    mesh.style.roughness,
                    mesh.style.clearcoat,
                    mesh.style.fresnel_power,
                    mesh.style.rim_strength,
                ],
                light_dir: self.light_dir.normalize_or(Vec3::Y).extend(self.light_intensity).to_array(),
                ghost: [
                    mesh.style.ghost_alpha.0,
                    mesh.style.ghost_alpha.1,
                    0.0,
                    1.0,
                ],
            };
            queue.write_buffer(&self.uniform, i as u64 * MESH_UNIFORM_STRIDE, bytemuck::bytes_of(&u));
        }
    }

    /// Fill the G-buffer. Clears all four attachments, so it must run even when
    /// there is nothing to draw — the depth clear is what tells the raymarcher
    /// there is no geometry in the way.
    #[allow(clippy::too_many_arguments)]
    pub fn render_gbuffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        _profiler: &mut Profiler,
        camera_group: &wgpu::BindGroup,
        meshes: &[GpuMesh],
        albedo: &wgpu::TextureView,
        normal: &wgpu::TextureView,
        motion: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) {
        fn clear(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh g-buffer"),
            color_attachments: &[clear(albedo), clear(normal), clear(motion)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    // Reverse-Z: the far plane is 0.
                    load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let pipeline = if self.display.writes_gbuffer() {
            &self.gbuffer_pipeline
        } else if self.display.writes_depth_prepass() {
            &self.ghost_depth_pipeline
        } else {
            return;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, camera_group, &[]);
        for (i, mesh) in meshes.iter().take(MAX_MESHES).enumerate() {
            if !mesh.visible || mesh.index_count == 0 {
                continue;
            }
            pass.set_bind_group(1, &self.group, &[i as u32 * MESH_UNIFORM_STRIDE as u32]);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }

    /// Draw the ghost shell or the wireframe into the HDR target, after the
    /// volume has been composited so the flow shows through.
    pub fn render_overlay(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_group: &wgpu::BindGroup,
        meshes: &[GpuMesh],
        hdr: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) {
        if !self.display.draws_overlay() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                // Test but do not write: a ghosted shell must not occlude
                // anything drawn after it.
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let wire = self.display == MeshDisplay::Wireframe;
        pass.set_pipeline(if wire { &self.wire_pipeline } else { &self.ghost_pipeline });
        pass.set_bind_group(0, camera_group, &[]);
        for (i, mesh) in meshes.iter().take(MAX_MESHES).enumerate() {
            if !mesh.visible {
                continue;
            }
            let (buffer, count) = if wire {
                (&mesh.edges, mesh.edge_count)
            } else {
                (&mesh.indices, mesh.index_count)
            };
            if count == 0 {
                continue;
            }
            pass.set_bind_group(1, &self.group, &[i as u32 * MESH_UNIFORM_STRIDE as u32]);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_index_buffer(buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..count, 0, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_cycles_through_all_four_modes_and_returns() {
        let mut m = MeshDisplay::Solid;
        let mut seen = Vec::new();
        for _ in 0..4 {
            seen.push(m);
            m = m.cycle();
        }
        assert_eq!(m, MeshDisplay::Solid, "cycle must return to its start");
        assert_eq!(seen.len(), 4);
        for a in [MeshDisplay::Solid, MeshDisplay::Ghost, MeshDisplay::Wireframe, MeshDisplay::Off] {
            assert!(seen.contains(&a), "{} was skipped", a.label());
        }
    }

    #[test]
    fn each_display_mode_touches_exactly_the_right_targets() {
        // Solid shades deferred; ghost contributes depth and motion but no
        // albedo or normal, so the compositor leaves its pixels to the volume
        // and to the overlay; wireframe and off touch neither.
        assert!(MeshDisplay::Solid.writes_gbuffer());
        assert!(!MeshDisplay::Solid.writes_depth_prepass());

        assert!(!MeshDisplay::Ghost.writes_gbuffer());
        assert!(MeshDisplay::Ghost.writes_depth_prepass());
        assert!(MeshDisplay::Ghost.draws_overlay());

        assert!(!MeshDisplay::Wireframe.writes_gbuffer());
        assert!(!MeshDisplay::Wireframe.writes_depth_prepass());
        assert!(MeshDisplay::Wireframe.draws_overlay());

        assert!(!MeshDisplay::Off.writes_gbuffer());
        assert!(!MeshDisplay::Off.writes_depth_prepass());
        assert!(!MeshDisplay::Off.draws_overlay());
    }

    #[test]
    fn edge_list_deduplicates_shared_edges() {
        // Two triangles sharing an edge: 5 unique edges, not 6.
        let indices = [0u32, 1, 2, 2, 1, 3];
        let edges = edge_list(&indices);
        assert_eq!(edges.len(), 10, "expected 5 edges, got {}", edges.len() / 2);

        let mut pairs: Vec<(u32, u32)> = edges
            .chunks_exact(2)
            .map(|c| if c[0] < c[1] { (c[0], c[1]) } else { (c[1], c[0]) })
            .collect();
        pairs.sort();
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn edge_list_of_a_closed_tetrahedron_has_six_edges() {
        // Euler: V - E + F = 2 for a closed genus-0 surface, so 4 - E + 4 = 2.
        let indices = [0u32, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
        assert_eq!(edge_list(&indices).len() / 2, 6);
    }

    #[test]
    fn edge_list_handles_degenerate_input() {
        assert!(edge_list(&[]).is_empty());
        // A trailing partial triangle is ignored rather than panicking.
        assert_eq!(edge_list(&[0, 1, 2, 3, 4]).len() / 2, 3);
    }

    #[test]
    fn vertex_layout_matches_the_struct() {
        assert_eq!(MeshVertex::LAYOUT.array_stride, 28);
        assert_eq!(MeshVertex::LAYOUT.attributes.len(), 3);
        assert_eq!(MeshVertex::LAYOUT.attributes[1].offset, 12);
        assert_eq!(MeshVertex::LAYOUT.attributes[2].offset, 24);
    }

    #[test]
    fn mesh_uniform_fits_the_dynamic_offset_stride() {
        let size = std::mem::size_of::<MeshUniform>() as u64;
        assert_eq!(size % 16, 0);
        assert!(size <= MESH_UNIFORM_STRIDE, "{size} exceeds the {MESH_UNIFORM_STRIDE}-byte stride");
        // 256 is the worst-case min_uniform_buffer_offset_alignment.
        assert_eq!(MESH_UNIFORM_STRIDE % 256, 0);
    }

    #[test]
    fn reverse_z_constants_agree_with_the_camera() {
        // The depth buffer is cleared to the far plane and the test keeps the
        // *larger* value. Both of these are backwards from the usual convention
        // and both must change together.
        assert_eq!(DEPTH_CLEAR, 0.0);
        assert_eq!(DEPTH_COMPARE, wgpu::CompareFunction::Greater);
        let cam = crate::camera::Camera::default();
        assert_eq!(cam.linear_depth(DEPTH_CLEAR), f32::INFINITY);
    }

    #[test]
    fn gbuffer_formats_are_renderable() {
        let base = wgpu::Features::empty();
        for f in [ALBEDO_FORMAT, NORMAL_FORMAT, MOTION_FORMAT, DEPTH_FORMAT] {
            assert!(
                f.guaranteed_format_features(base)
                    .allowed_usages
                    .contains(wgpu::TextureUsages::RENDER_ATTACHMENT),
                "{f:?} is not renderable in the base feature set"
            );
        }
    }
}

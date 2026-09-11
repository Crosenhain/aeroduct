//! Render one frame of a synthetic bent duct to `preview.png`.
//!
//! ```text
//! cargo run --release -p ad-render --example preview -- [field] [out.png] [display] [overlay]
//! ```
//!
//! `field` is one of `speed`, `q`, `vorticity`, `pressure`; `display` one of
//! `ghost`, `solid`, `wire`, `off`; `overlay` one of `none`, `particles`,
//! `trails`, `few`, `few-trails`, `iso`, `slice`, `slice-follow`, `all`.
//!
//! This exists because a renderer's real acceptance test is looking at it. The
//! automated tests can tell you the maths is right and nothing panics; only a
//! picture tells you whether the thing is worth staring at for an hour. It also
//! gives the other build agents something to run before the app shell exists.
//!
//! The Wave 2 overlays need a *warm-up*, which the still-image path would
//! otherwise hide: the tracer population starts entirely dead, so frame 0 seeds
//! all of it at the inlet at once and an immediate screenshot catches a single
//! coherent sheet of particles rather than a duct full of flow. The overlay
//! modes therefore advect for a few hundred frames, then freeze the animation
//! and let the progressive accumulator converge on a clean image.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use ad_gpu::{flags, Bbox, FlowPatch, GpuContext, Grid, LatticeUnits};
use ad_render::{
    fields, util, AxisPreset, Camera, Centreline, DeriveScales, DerivedField, FieldResolution,
    FieldSources, FrameInput, GpuMesh, IsosurfaceOverlay, MeshData, MeshDisplay, MeshStyle,
    MeshVertex, OpacityMode, Renderer, RendererConfig, Scene, SdfSource, SeedWeights, SliceOverlay,
    SlicePlane, SoftIso, StreaklineConfig, StreaklineOverlay, TransferFunction,
};
use glam::{UVec3, Vec3};

/// Which Wave 2 overlays to add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Overlays {
    particles: bool,
    trails: bool,
    /// A small tracer population, so individual sprites, motion-blur capsules
    /// and ribbons are separable by eye. Four million of them merge into a
    /// continuous cloud, which is the right picture but a useless one for
    /// checking that a sprite is drawn correctly.
    sparse: bool,
    isosurface: bool,
    slice: bool,
    follow_centreline: bool,
}

impl Overlays {
    const NONE: Self = Self {
        particles: false,
        trails: false,
        sparse: false,
        isosurface: false,
        slice: false,
        follow_centreline: false,
    };

    fn parse(name: Option<&str>) -> Self {
        match name {
            Some("particles") => Self { particles: true, ..Self::NONE },
            Some("trails") => Self { particles: true, trails: true, ..Self::NONE },
            Some("few") => Self { particles: true, sparse: true, ..Self::NONE },
            Some("few-trails") => {
                Self { particles: true, trails: true, sparse: true, ..Self::NONE }
            }
            Some("iso") | Some("isosurface") => Self { isosurface: true, ..Self::NONE },
            Some("slice") => Self { slice: true, ..Self::NONE },
            Some("slice-follow") => {
                Self { slice: true, follow_centreline: true, ..Self::NONE }
            }
            Some("all") => Self {
                particles: true,
                trails: true,
                sparse: false,
                isosurface: true,
                slice: true,
                follow_centreline: false,
            },
            _ => Self::NONE,
        }
    }

    fn any(self) -> bool {
        self.particles || self.isosurface || self.slice
    }
}

// Geometry of the synthetic duct, mm. Loosely the shape of the real test part:
// a ~90 degree bend with an area contraction along it.
const BEND_RADIUS: f32 = 55.0;
const INLET_RADIUS: f32 = 13.0;
const OUTLET_RADIUS: f32 = 9.5;
const WALL: f32 = 2.0;

/// Centreline point and tangent at parameter `s` in `[0, 1]`.
fn centreline(s: f32) -> (Vec3, Vec3) {
    let a = s * FRAC_PI_2;
    let (sa, ca) = a.sin_cos();
    (
        Vec3::new(BEND_RADIUS * ca, 0.0, BEND_RADIUS * sa),
        Vec3::new(-sa, 0.0, ca),
    )
}

/// Passage radius at parameter `s`, tapering from inlet to outlet.
fn passage_radius(s: f32) -> f32 {
    INLET_RADIUS + (OUTLET_RADIUS - INLET_RADIUS) * smoothstep(s)
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Nearest centreline parameter to a world point, by closed form on the bend.
fn nearest_s(p: Vec3) -> f32 {
    (p.z.atan2(p.x) / FRAC_PI_2).clamp(0.0, 1.0)
}

/// Sweep the outer wall into a triangle mesh.
fn build_duct_mesh() -> (Vec<MeshVertex>, Vec<u32>) {
    const ALONG: u32 = 96;
    const AROUND: u32 = 48;
    let mut verts = Vec::new();
    let mut idx = Vec::new();

    for i in 0..=ALONG {
        let s = i as f32 / ALONG as f32;
        let (c, t) = centreline(s);
        let r = passage_radius(s) + WALL;
        // The bend lies in the XZ plane, so the binormal is world up.
        let up = Vec3::Y;
        let right = t.cross(up).normalize();
        for j in 0..AROUND {
            let a = j as f32 / AROUND as f32 * TAU;
            let (sa, ca) = a.sin_cos();
            let n = (right * ca + up * sa).normalize();
            verts.push(MeshVertex {
                position: (c + n * r).to_array(),
                normal: n.to_array(),
                // Per-vertex scalar channel: here just the sweep parameter, so
                // the G-buffer's fourth channel is carrying something real.
                scalar: s,
            });
        }
    }
    for i in 0..ALONG {
        for j in 0..AROUND {
            let a = i * AROUND + j;
            let b = i * AROUND + (j + 1) % AROUND;
            let c = (i + 1) * AROUND + j;
            let d = (i + 1) * AROUND + (j + 1) % AROUND;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (verts, idx)
}

/// Synthetic solver state: a swirling jet inside the passage, still air outside,
/// and a solid shell in between.
fn build_field(grid: Grid, u_lb: f32) -> (Vec<[f32; 4]>, Vec<u8>) {
    let n = (grid.dims.x * grid.dims.y * grid.dims.z) as usize;
    let mut macro_data = vec![[0.0f32; 4]; n];
    let mut flag_data = vec![flags::FLUID; n];

    for z in 0..grid.dims.z {
        for y in 0..grid.dims.y {
            for x in 0..grid.dims.x {
                let c = UVec3::new(x, y, z);
                let p = grid.cell_center_mm(c);
                let i = grid.linear(c) as usize;

                let s = nearest_s(p);
                let (q, t) = centreline(s);
                let d = p - q;
                let r = d.length();
                let r_in = passage_radius(s);

                // Only the swept region has a wall; the mouths are open.
                let on_bend = p.x > -1.0 && p.z > -1.0;
                if on_bend && r > r_in && r < r_in + WALL {
                    flag_data[i] = flags::SOLID;
                    continue;
                }
                if !on_bend || r > r_in {
                    continue; // still air outside the duct
                }

                // Turbulent-looking core: a plug profile with a wall boundary
                // layer, a swirl that grows through the bend, and a couple of
                // streamwise waves so the Q-criterion view has structure.
                let wall_profile = (1.0 - (r / r_in).powi(6)).max(0.0);
                let phase = d.y.atan2(d.dot(t.cross(Vec3::Y)));
                let ripple = 1.0 + 0.35 * (6.0 * PI * s + 3.0 * phase).sin();
                let swirl_axis = t.cross(d).normalize_or_zero();
                let swirl = 0.45 * smoothstep(s * 1.4) * (r / r_in);
                let u = (t * ripple + swirl_axis * swirl) * (u_lb * wall_profile);

                macro_data[i] = [u.x, u.y, u.z, 0.004 * (1.0 - s) * wall_profile];
            }
        }
    }
    (macro_data, flag_data)
}

/// Signed distance to the solid shell, mm, positive outside it.
///
/// A radial approximation rather than a true SDF — exact for a straight tube and
/// a little short on the inside of the bend — which is all the particle wall
/// projection needs, because what it uses is the *direction* of the gradient and
/// that is radial either way.
fn solid_distance(p: Vec3) -> f32 {
    // Beyond either mouth there is no wall to hit.
    if p.x <= -1.0 || p.z <= -1.0 {
        return 1.0e4;
    }
    let s = nearest_s(p);
    let (c, _) = centreline(s);
    let r = (p - c).length();
    let r_in = passage_radius(s);
    if r < r_in {
        r_in - r
    } else if r > r_in + WALL {
        r - (r_in + WALL)
    } else {
        -(r - r_in).min(r_in + WALL - r)
    }
}

/// Sample [`solid_distance`] into an `R32Float` 3D texture for the particle
/// pass. 128^3 at 4 bytes is 8 MiB, which resolves the 2 mm wall about four
/// times over.
fn build_sdf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bbox: Bbox,
) -> (wgpu::Texture, SdfSource) {
    const N: u32 = 128;
    let size = bbox.size();
    let mut data = Vec::with_capacity((N * N * N) as usize);
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let f = (Vec3::new(x as f32, y as f32, z as f32) + Vec3::splat(0.5)) / N as f32;
                data.push(solid_distance(bbox.min + f * size));
            }
        }
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("preview sdf"),
        size: wgpu::Extent3d { width: N, height: N, depth_or_array_layers: N },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(N * 4),
            rows_per_image: Some(N),
        },
        wgpu::Extent3d { width: N, height: N, depth_or_array_layers: N },
    );
    let source = SdfSource {
        view: tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        }),
        min_mm: bbox.min,
        size_mm: size,
        // Half a wall thickness of standoff, so tracers do not graze the
        // half-solid voxels right at the surface.
        surface_offset_mm: 0.0,
    };
    (tex, source)
}

/// The inlet mouth, as the flux-weighted seeding patch.
fn inlet_patch() -> FlowPatch {
    let (c, t) = centreline(0.0);
    let up = Vec3::Y;
    let right = t.cross(up).normalize();
    FlowPatch {
        center_mm: c,
        normal: t,
        half_u: right * INLET_RADIUS,
        half_v: up * INLET_RADIUS,
    }
}

/// Control points along the synthetic duct's centreline, for the
/// "slice follows the duct" mode.
fn centreline_points() -> Vec<Vec3> {
    (0..=8).map(|i| centreline(i as f32 / 8.0).0).collect()
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut args = std::env::args().skip(1);
    let field = match args.next().as_deref() {
        Some("q") | Some("Q") => DerivedField::QCriterion,
        Some("vorticity") => DerivedField::Vorticity,
        Some("pressure") => DerivedField::Pressure,
        _ => DerivedField::Speed,
    };
    let out = args.next().unwrap_or_else(|| "preview.png".to_string());
    let display = match args.next().as_deref() {
        Some("solid") => MeshDisplay::Solid,
        Some("wire") | Some("wireframe") => MeshDisplay::Wireframe,
        Some("off") => MeshDisplay::Off,
        _ => MeshDisplay::Ghost,
    };
    let overlays = Overlays::parse(args.next().as_deref());

    let gpu = GpuContext::new_blocking(None)?;
    println!("device: {}", gpu.info.name);

    // A domain around the bend, at a resolution that renders in a second or two.
    let grid = Grid {
        dims: UVec3::new(160, 96, 160),
        dx_mm: 0.75,
        origin_mm: Vec3::new(-20.0, -36.0, -20.0),
    };
    let units = LatticeUnits::for_air(grid.dx_mm as f64, 5.0, 0.05);
    let scales = DeriveScales::new(&units, (INLET_RADIUS * 2.0) as f64);

    let (w, h) = (1280u32, 800u32);
    let mut renderer = Renderer::new(
        &gpu,
        RendererConfig {
            width: w,
            height: h,
            target_format: wgpu::TextureFormat::Rgba8Unorm,
            grid,
            field_resolution: FieldResolution::Full,
            profiling: false,
        },
    )?;

    // --- geometry ---
    let (verts, idx) = build_duct_mesh();
    println!("duct mesh: {} vertices, {} triangles", verts.len(), idx.len() / 3);
    let mut scene = Scene::new();
    scene.meshes.push(GpuMesh::upload(
        &gpu.device,
        &gpu.queue,
        MeshData { vertices: &verts, indices: &idx },
        MeshStyle::default(),
    ));
    renderer.set_mesh_display(display);

    // --- solver state ---
    let (macro_data, flag_data) = build_field(grid, units.u_lb as f32);
    let solid = flag_data.iter().filter(|f| **f == flags::SOLID).count();
    println!("field: {} cells, {solid} solid", macro_data.len());

    let (mac_tex, flg_tex) = fields::create_source_textures(&gpu.device, grid);
    let mut bytes = Vec::with_capacity(macro_data.len() * 8);
    for t in &macro_data {
        for c in t {
            bytes.extend_from_slice(&ad_render::colormap::f32_to_f16_bits(*c).to_le_bytes());
        }
    }
    let extent = wgpu::Extent3d {
        width: grid.dims.x,
        height: grid.dims.y,
        depth_or_array_layers: grid.dims.z,
    };
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &mac_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(grid.dims.x * 8),
            rows_per_image: Some(grid.dims.y),
        },
        extent,
    );
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &flg_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &flag_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(grid.dims.x),
            rows_per_image: Some(grid.dims.y),
        },
        extent,
    );
    let d3 = wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    };
    let mac_view = mac_tex.create_view(&d3);
    let flg_view = flg_tex.create_view(&d3);

    // --- look ---
    renderer.set_field(field);
    {
        let tf = renderer.transfer_function_mut();
        match field {
            DerivedField::Speed => {
                tf.mode = OpacityMode::SoftIso;
                tf.range = [0.0, 7.0];
                tf.iso = SoftIso { center: 4.4, width: 1.4, amplitude: 0.45, cutoff_widths: 2.0 };
            }
            DerivedField::QCriterion => {
                tf.mode = OpacityMode::SoftIso;
                tf.range = [0.0, 2.0];
                tf.iso = SoftIso { center: 0.55, width: 0.26, amplitude: 0.85, cutoff_widths: 2.0 };
            }
            DerivedField::Vorticity => {
                tf.mode = OpacityMode::SoftIso;
                tf.range = [0.0, 6.0];
                tf.iso = SoftIso { center: 2.4, width: 0.9, amplitude: 0.5, cutoff_widths: 2.0 };
            }
            DerivedField::Pressure => {
                *tf = TransferFunction::preset(DerivedField::Pressure);
                tf.fit_range(-30.0, 30.0);
            }
        }
        tf.sanitise();
    }

    // --- Wave 2 overlays ---
    //
    // Built after the transfer function, because the isosurface wants the brick
    // grid binarised against the Q channel and that follows `set_field`.
    let duct_bbox = scene.bbox();
    let _sdf_keepalive;
    // Set once the derive pass has run, so the frame loop below does not repeat
    // it. The isosurface path derives early to get a histogram.
    let mut derived = false;
    // Registration order matters, and only in one direction. No overlay writes
    // into the shared depth buffer — `OverlayContext` says to test against it
    // and not write — so they cannot occlude one another, only overdraw. Opaque
    // surfaces therefore go first and the additive tracers last, which puts the
    // emissive layer on top where it belongs.
    if overlays.slice {
        let mut slice = SliceOverlay::new(
            &gpu.device,
            &gpu.queue,
            renderer.camera_bind_group_layout(),
            renderer.fields_bind_group_layout(),
        )?;
        {
            let s = slice.settings_mut();
            s.range = [0.0, 7.0];
            // The bend lies in the XZ plane at y = 0, so a plane of constant y
            // cuts the passage lengthwise: the flow is almost entirely in-plane
            // and the honesty weight stays near 1, which is what makes this the
            // right slice to demonstrate LIC on.
            s.plane = SlicePlane::axis(AxisPreset::Y, grid.bbox(), 0.5);
            // Crop to the part rather than to the domain. The default fits the
            // whole field volume, which is right in the app — the still air
            // around the duct is real data and hiding it would be a choice made
            // on the user's behalf — but for a picture of the duct it is 80% of
            // the frame in flat colour-map purple.
            s.half_extent_mm = Some(0.5 * duct_bbox.size().length());
            if overlays.particles || overlays.isosurface {
                // Layered with something behind it, so let that show through.
                s.opacity = 0.55;
            }
        }
        if overlays.follow_centreline {
            slice.set_centreline(Centreline::new(&centreline_points()));
            // Two thirds of the way round the bend, where an axis-aligned cut
            // would be at 60 degrees to the flow and show nothing useful.
            slice.settings_mut().follow_centreline = Some(0.66);
            // A cut perpendicular to a 26 mm passage wants to be about the size
            // of the passage; the default fit-the-volume extent turns it into a
            // wall behind the part.
            slice.settings_mut().half_extent_mm = Some(20.0);
        }
        renderer.add_overlay(Box::new(slice));
    }
    if overlays.isosurface {
        let mut iso = IsosurfaceOverlay::new(
            &gpu.device,
            &gpu.queue,
            renderer.camera_bind_group_layout(),
            renderer.fields_bind_group_layout(),
            renderer.brick_bind_group_layout(),
        )?;
        iso.settings_mut().speed_range = [0.0, 7.0];
        iso.light_dir = renderer.post_settings().lighting.key_dir;

        // Derive the fields once up front so the Q histogram has something to
        // reduce, then let it choose the isolevel. This is the flow the UI takes
        // — a histogram under the slider and the handle parked in the high tail
        // — and it is the difference between seeing vortices on first sight and
        // concluding the duct has none.
        {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            let scratch = util::color_target(
                &gpu.device,
                "derive only",
                64,
                64,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            renderer.render(
                &mut enc,
                FrameInput {
                    camera: &Camera::default(),
                    scene: &Scene::new(),
                    sources: Some(FieldSources {
                        macro_view: &mac_view,
                        flags_view: Some(&flg_view),
                        scales,
                    }),
                    sdf: None,
                    target: &scratch.create_view(&Default::default()),
                    dt: 0.0,
                },
            )?;
            gpu.queue.submit([enc.finish()]);
            derived = true;

            let hist = iso.compute_histogram(&gpu.device, &gpu.queue, renderer.fields());
            let level = hist.suggested_isolevel();
            println!(
                "Q histogram: {} fluid cells, median {:.3}, p99 {:.3} -> isolevel {level:.3}",
                hist.total(),
                hist.percentile(0.5),
                hist.percentile(0.99),
            );
            iso.settings_mut().iso_level = level;
        }
        renderer.add_overlay(Box::new(iso));
    }
    if overlays.particles {
        // Tracers *are* the flow visualisation, so the volume raymarch would
        // draw a second, very similar cloud on top of them and make it
        // impossible to tell which is which. In the app they are alternatives on
        // the same toggle.
        let mut vs = *renderer.volume_settings();
        vs.density = 0.0;
        renderer.set_volume_settings(vs);

        let mut sys = StreaklineOverlay::new(
            &gpu.device,
            &gpu.queue,
            renderer.camera_bind_group_layout(),
            renderer.fields_bind_group_layout(),
            match (overlays.sparse, overlays.trails) {
                (true, t) => StreaklineConfig {
                    count: 24 << 10,
                    trail_count: if t { 24 << 10 } else { 0 },
                    trail_length: 32,
                },
                (false, true) => StreaklineConfig::with_trails(),
                (false, false) => StreaklineConfig::default(),
            },
        )?;
        let (tex, source) = build_sdf(&gpu.device, &gpu.queue, duct_bbox);
        _sdf_keepalive = tex;
        sys.set_sdf(&gpu.device, Some(source));
        sys.set_inlet(Some(inlet_patch()));
        sys.set_seed_bounds(Some(duct_bbox));
        {
            let s = sys.settings_mut();
            // A lifetime of 0.05 simulated seconds is about 250 mm of travel at
            // 5 m/s, i.e. comfortably longer than the duct, and — at the default
            // time scale — about 150 frames. That is what lets the warm-up below
            // reach a genuinely mixed population instead of a single coherent
            // sheet of particles all born on frame 0.
            s.life_s = 0.05;
            s.life_jitter = 0.25;
            s.color_range = [0.0, 7.0];
            s.inlet_peak_speed_ms = 7.0;
            s.seed = SeedWeights::default();
            s.trails.enabled = overlays.trails;
            if overlays.sparse {
                // The emission normalisation holds the *total* light constant,
                // so a small population puts the same light through far fewer
                // sprites and the peaks clip. That is the honest behaviour of a
                // Monte-Carlo estimator with fewer samples; the answer is to
                // stop down, which is what a photographer would do.
                s.intensity *= 0.25;
            }
        }
        println!(
            "streaklines: {} tracers, {} with trails, {:.0} MiB",
            sys.config().count,
            sys.config().trail_count,
            sys.config().bytes() as f64 / (1024.0 * 1024.0)
        );
        renderer.add_overlay(Box::new(sys));
    }
    if overlays.any() {
        println!("overlays: {:?}", renderer.overlay_names());
    }

    // Angle and aspect first, then frame: the fit is exact for the direction it
    // is given, so setting the orbit afterwards would loosen it again.
    let mut cam = Camera::default();
    cam.aspect = w as f32 / h as f32;
    cam.yaw = -0.75;
    cam.pitch = 0.42;
    cam.frame_bbox(scene.bbox(), 0.08);

    let target = util::color_target(
        &gpu.device,
        "preview",
        w,
        h,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let view = target.create_view(&Default::default());

    // Two phases.
    //
    // The warm-up advects the overlays with a real frame delta, because a tracer
    // population starts entirely dead and an immediate screenshot would catch
    // one coherent sheet of particles all born on frame 0 rather than a duct
    // full of flow. Then time stops — `dt = 0` — which every overlay reports as
    // static, so the progressive accumulator converges on the frozen image
    // exactly as it does for a still camera.
    const FRAMES: u32 = 48;
    let warmup = if overlays.particles { 400 } else { 0 };
    let mut last = Vec::new();
    let started = std::time::Instant::now();

    for frame in 0..warmup + FRAMES {
        let dt = if frame < warmup { 1.0 / 60.0 } else { 0.0 };
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let sources = (!derived).then(|| FieldSources {
            macro_view: &mac_view,
            flags_view: Some(&flg_view),
            scales,
        });
        derived = true;
        renderer.render(
            &mut enc,
            FrameInput { camera: &cam, scene: &scene, sources, sdf: None, target: &view, dt },
        )?;
        if frame + 1 == warmup + FRAMES {
            last = util::readback_rgba8(&gpu.device, &gpu.queue, &target, w, h, enc);
        } else {
            gpu.queue.submit([enc.finish()]);
            renderer.after_submit();
        }
        scene.end_frame();
    }
    println!(
        "{warmup} warm-up + {FRAMES} still frames in {:.1} s, accumulated {} samples",
        started.elapsed().as_secs_f32(),
        renderer.accumulated_samples()
    );

    write_png(&out, w, h, &last)?;
    println!("wrote {out}");
    Ok(())
}

// -- a minimal PNG writer ----------------------------------------------------
//
// Stored (uncompressed) deflate blocks inside a zlib wrapper. Thirty lines and
// no dependency, versus pulling an image crate into a render crate that has no
// other use for one.

fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity((w * h * 4 + h) as usize);
    for y in 0..h as usize {
        raw.push(0u8); // filter type: none
        let start = y * w as usize * 4;
        raw.extend_from_slice(&rgba[start..start + w as usize * 4]);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA, no interlace
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);

    std::fs::write(path, out)
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut i = 0;
    loop {
        let n = (data.len() - i).min(65535);
        let last = i + n >= data.len();
        out.push(u8::from(last));
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(!(n as u16)).to_le_bytes());
        out.extend_from_slice(&data[i..i + n]);
        i += n;
        if last {
            break;
        }
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for byte in data {
        a = (a + *byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

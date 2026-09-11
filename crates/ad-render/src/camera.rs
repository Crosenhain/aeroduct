//! Camera, orbit controller, view presets and keyframed flythrough.
//!
//! # Reverse-Z, everywhere
//!
//! Depth is stored **reversed**: the near plane maps to 1.0 and infinity maps to
//! 0.0, and the depth test is `Greater` against a buffer cleared to 0.0.
//!
//! The reason is that floating-point depth and the perspective divide have
//! opposite error distributions. `1/z` bunches almost all of its precision near
//! the near plane; float32 bunches almost all of *its* precision near zero.
//! Standard 0..1 depth stacks those two effects and wastes both. Reversing one
//! of them makes the exponent do useful work across the whole range, which turns
//! a ~10^-3 relative depth resolution at the far end into ~10^-7. On a duct
//! scene that spans 145 mm of geometry inside a 260 mm domain, with a near plane
//! that has to sit at a fraction of a millimetre so the camera can go *inside*
//! the duct, that difference is the difference between clean surfaces and
//! z-fighting confetti.
//!
//! The consequence is that **every** depth comparison in this crate is
//! reversed. The volume compositing test in `volume.wgsl` reads the same buffer
//! and must linearise it the same way; [`Camera::linear_depth`] is the single
//! definition, mirrored once in `shaders/render/common.wgsl`.
//!
//! We also use an *infinite* far plane. With reverse-Z there is no precision
//! reason to have a finite far plane, and removing it removes a knob that can
//! silently clip the sponge layer or a flythrough path.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

use ad_gpu::Bbox;

/// Where the camera is and how it sees. Millimetres, like everything else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// The point being orbited, mm.
    pub target: Vec3,
    /// Distance from `target` to the eye, mm.
    pub distance: f32,
    /// Rotation about the world up axis, radians.
    pub yaw: f32,
    /// Elevation, radians. Clamped away from the poles so `up` never degenerates.
    pub pitch: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    pub aspect: f32,
    /// Near plane, mm. Small, because the camera is expected to fly *inside* a
    /// 6 mm passage. Reverse-Z is what makes this affordable.
    pub z_near: f32,
    /// World up. `+Y` matches the STL convention used elsewhere in the app.
    pub up: Vec3,
    /// Sub-pixel offset in NDC applied to the projection, for TAA. Units are
    /// NDC, i.e. `2 * pixel_offset / resolution`.
    pub jitter: Vec2,
}

/// How close to straight-up the camera may get. Exactly at the pole the view
/// matrix has no defined roll and the controller would flip.
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 1.0e-3;

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 400.0,
            yaw: 0.7,
            pitch: 0.45,
            fov_y: 45.0_f32.to_radians(),
            aspect: 16.0 / 9.0,
            z_near: 0.25,
            up: Vec3::Y,
            jitter: Vec2::ZERO,
        }
    }
}

impl Camera {
    /// Eye position in world space, mm.
    pub fn eye(&self) -> Vec3 {
        self.target + self.offset()
    }

    /// Vector from target to eye.
    fn offset(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy) * self.distance
    }

    /// Unit vector the camera looks along.
    pub fn forward(&self) -> Vec3 {
        (-self.offset()).normalize_or_zero()
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, self.up)
    }

    /// Reverse-Z, infinite-far perspective projection.
    ///
    /// Derivation, for right-handed view space looking down `-Z`:
    /// we want `w_clip = -z_view` (the usual perspective divide) and
    /// `z_clip = z_near`, so `depth = z_near / (-z_view)`. That is 1 at the near
    /// plane and tends to 0 at infinity, monotonically. Two entries of the
    /// matrix do the whole job; there is no far plane in it at all.
    pub fn projection(&self) -> Mat4 {
        let f = 1.0 / (self.fov_y * 0.5).tan();
        let mut p = Mat4::from_cols(
            Vec4::new(f / self.aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, self.z_near, 0.0),
        );
        // Jitter must be a pure shift in NDC: `ndc.xy += jitter`. Since
        // `w_clip = -z_view`, that means adding `jitter * w` to `clip.xy`, i.e.
        // `-jitter * z_view` — which is the `z` column, negated. Getting this
        // sign wrong costs half a pixel of permanent bias in the TAA result and
        // reads as a soft double edge that no amount of clamping tuning fixes.
        p.z_axis.x -= self.jitter.x;
        p.z_axis.y -= self.jitter.y;
        p
    }

    /// Projection with no TAA jitter. Motion vectors must be computed from this,
    /// otherwise the jitter shows up as spurious motion and TAA fights itself.
    pub fn projection_unjittered(&self) -> Mat4 {
        let mut c = *self;
        c.jitter = Vec2::ZERO;
        c.projection()
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection() * self.view()
    }

    pub fn view_projection_unjittered(&self) -> Mat4 {
        self.projection_unjittered() * self.view()
    }

    /// Turn a reverse-Z depth buffer value back into a positive view-space
    /// distance along the view axis, mm.
    ///
    /// This is the inverse of the two matrix entries above and nothing else, so
    /// it stays correct if the FOV or aspect changes. `depth == 0` means "no
    /// geometry / infinitely far", which is reported as `f32::INFINITY` rather
    /// than a division blow-up.
    pub fn linear_depth(&self, depth: f32) -> f32 {
        if depth <= 0.0 {
            f32::INFINITY
        } else {
            self.z_near / depth
        }
    }

    /// The forward map of [`Camera::linear_depth`]. Handy in tests and for
    /// placing debug markers at a known distance.
    pub fn depth_from_linear(&self, view_distance: f32) -> f32 {
        if view_distance <= 0.0 {
            1.0
        } else {
            (self.z_near / view_distance).min(1.0)
        }
    }

    /// World-space ray through a point in NDC (`x`,`y` in `[-1, 1]`, `y` up).
    ///
    /// Built from two unprojected points rather than from `inv_proj` columns,
    /// because with an infinite far plane the depth-0 row is singular. Depths
    /// 1.0 and 0.5 are both well-conditioned and give the same direction.
    pub fn ray(&self, ndc: Vec2) -> (Vec3, Vec3) {
        let inv = self.view_projection().inverse();
        let unproject = |z: f32| {
            let p = inv * Vec4::new(ndc.x, ndc.y, z, 1.0);
            p.truncate() / p.w
        };
        let near = unproject(1.0);
        let mid = unproject(0.5);
        (near, (mid - near).normalize_or_zero())
    }

    /// Move the camera so `bbox` exactly fills the frame, with `margin` slack
    /// (0.1 = 10% padding on the tightest axis).
    ///
    /// Fits the eight corners in view space rather than the bounding *sphere*.
    /// The sphere fit is one line and always safe, but on a part like the test
    /// duct — 145 x 72 x 69 mm, so a diagonal 1.7x its largest side — it leaves
    /// the model occupying under a third of the frame and every screenshot
    /// looking like it was taken from the next room.
    ///
    /// The algebra is exact. Rotating so the eye is at view-space `(0, 0, d)`
    /// puts corner `c` at `(vx, vy, vz - d)`, and the frustum needs
    /// `|vx| <= (d - vz) * tan(fov_x/2)`. Solve for `d` and take the largest.
    pub fn frame_bbox(&mut self, bbox: Bbox, margin: f32) {
        if bbox.is_empty() {
            return;
        }
        self.target = bbox.center();

        // Basis of the current view direction, independent of distance.
        let forward = self.forward();
        let right = forward.cross(self.up).normalize_or(Vec3::X);
        let up = right.cross(forward).normalize_or(Vec3::Y);

        let ty = (self.fov_y * 0.5).tan() / (1.0 + margin.max(0.0));
        let tx = ty * self.aspect;

        let mut d = self.z_near * 4.0;
        for i in 0..8 {
            let c = Vec3::new(
                if i & 1 == 0 { bbox.min.x } else { bbox.max.x },
                if i & 2 == 0 { bbox.min.y } else { bbox.max.y },
                if i & 4 == 0 { bbox.min.z } else { bbox.max.z },
            ) - self.target;
            // View-space coordinates relative to the target, with `-forward`
            // being the view-space +Z axis.
            let v = Vec3::new(c.dot(right), c.dot(up), -c.dot(forward));
            d = d.max(v.z + v.x.abs() / tx).max(v.z + v.y.abs() / ty);
        }
        self.distance = d;
    }

    /// Yaw/pitch that look along `dir`.
    pub fn look_along(&mut self, dir: Vec3) {
        let d = dir.normalize_or_zero();
        if d.length_squared() < 0.5 {
            return;
        }
        // The eye sits opposite the look direction, so the offset is `-d`.
        let o = -d;
        self.pitch = o.y.clamp(-1.0, 1.0).asin().clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.yaw = o.x.atan2(o.z);
    }

    /// Clamp state that must stay in range regardless of how it was set.
    pub fn sanitise(&mut self) {
        self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.distance = self.distance.max(self.z_near * 2.0);
        self.fov_y = self
            .fov_y
            .clamp(5.0_f32.to_radians(), 140.0_f32.to_radians());
        if self.aspect <= 0.0 || !self.aspect.is_finite() {
            self.aspect = 1.0;
        }
    }
}

/// Standard views. "Along the duct axis" is the one that actually gets used:
/// looking straight down the inlet or outlet normal is how you judge whether a
/// bend is separating, and eyeballing it from an arbitrary orbit angle is a good
/// way to convince yourself of the wrong thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewPreset {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
    /// Three-quarter view; the default the app opens on.
    Iso,
}

impl ViewPreset {
    /// Direction the camera looks *along*.
    pub fn direction(self) -> Vec3 {
        match self {
            ViewPreset::Front => Vec3::NEG_Z,
            ViewPreset::Back => Vec3::Z,
            ViewPreset::Left => Vec3::X,
            ViewPreset::Right => Vec3::NEG_X,
            ViewPreset::Top => Vec3::NEG_Y,
            ViewPreset::Bottom => Vec3::Y,
            ViewPreset::Iso => Vec3::new(-0.6, -0.5, -0.62).normalize(),
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            ViewPreset::Front => "front (-Z)",
            ViewPreset::Back => "back (+Z)",
            ViewPreset::Left => "left (+X)",
            ViewPreset::Right => "right (-X)",
            ViewPreset::Top => "top (-Y)",
            ViewPreset::Bottom => "bottom (+Y)",
            ViewPreset::Iso => "iso",
        }
    }
    pub const ALL: [ViewPreset; 7] = [
        ViewPreset::Iso,
        ViewPreset::Front,
        ViewPreset::Back,
        ViewPreset::Left,
        ViewPreset::Right,
        ViewPreset::Top,
        ViewPreset::Bottom,
    ];
}

/// Orbit / pan / dolly with critically-damped smoothing.
///
/// The controller keeps a *goal* camera and a *displayed* camera and eases one
/// toward the other. Two things make this feel right rather than merely smooth:
///
/// 1. The easing is `1 - exp(-dt / tau)`, not a fixed `lerp(a, b, 0.2)`. The
///    fixed form is frame-rate dependent, so the camera visibly accelerates when
///    the sim is paused and the renderer jumps from 60 to 300 fps.
/// 2. Dolly is multiplicative and pan is scaled by distance, so the control
///    gain in *screen* space is constant. Panning at 500 mm out and panning at
///    3 mm out feel identical, which is what lets you inspect a 6 mm passage.
#[derive(Debug, Clone)]
pub struct OrbitController {
    /// Where the camera is being asked to go.
    pub goal: Camera,
    /// Where it currently is; this is what you render with.
    pub current: Camera,
    /// Time constant, seconds. ~0.08 reads as responsive but not twitchy.
    pub smoothing_tau: f32,
    pub orbit_speed: f32,
    pub pan_speed: f32,
    pub zoom_speed: f32,
    /// Distance limits, mm.
    pub distance_range: (f32, f32),
}

impl Default for OrbitController {
    fn default() -> Self {
        let cam = Camera::default();
        Self {
            goal: cam,
            current: cam,
            smoothing_tau: 0.08,
            orbit_speed: 0.008,
            pan_speed: 1.0,
            zoom_speed: 0.12,
            distance_range: (0.5, 5000.0),
        }
    }
}

impl OrbitController {
    pub fn new(camera: Camera) -> Self {
        Self {
            goal: camera,
            current: camera,
            ..Default::default()
        }
    }

    /// Mouse drag, pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.goal.yaw -= dx * self.orbit_speed;
        self.goal.pitch =
            (self.goal.pitch + dy * self.orbit_speed).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Mouse drag, pixels. `viewport_height` keeps the gain in screen units.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let view = self.goal.view();
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        // World mm covered by one pixel at the target plane.
        let mm_per_px =
            2.0 * self.goal.distance * (self.goal.fov_y * 0.5).tan() / viewport_height.max(1.0);
        let k = mm_per_px * self.pan_speed;
        self.goal.target += (-right * dx + up * dy) * k;
    }

    /// Wheel notches. Multiplicative so each notch covers the same fraction.
    pub fn dolly(&mut self, notches: f32) {
        let f = (-notches * self.zoom_speed).exp();
        self.goal.distance =
            (self.goal.distance * f).clamp(self.distance_range.0, self.distance_range.1);
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.goal.aspect = aspect;
        self.current.aspect = aspect;
    }

    pub fn apply_preset(&mut self, preset: ViewPreset) {
        self.goal.look_along(preset.direction());
    }

    pub fn frame_bbox(&mut self, bbox: Bbox, margin: f32) {
        self.goal.frame_bbox(bbox, margin);
        self.goal.distance = self
            .goal
            .distance
            .clamp(self.distance_range.0, self.distance_range.1);
    }

    /// Snap the displayed camera onto the goal. Use after a preset change that
    /// should not animate, and after loading a file.
    pub fn snap(&mut self) {
        self.current = self.goal;
    }

    /// Advance the smoothing. Returns true while still moving, which is the
    /// signal the progressive accumulator uses to decide whether it may keep
    /// integrating samples.
    pub fn update(&mut self, dt: f32) -> bool {
        self.goal.sanitise();
        // Frame-rate independent exponential approach. At dt = tau this covers
        // 63% of the remaining distance, at dt >> tau it saturates at 1.
        let k = 1.0 - (-dt.max(0.0) / self.smoothing_tau.max(1e-4)).exp();

        let before = self.current;
        // Yaw is interpolated on the shortest arc so crossing +/-pi does not
        // spin the model all the way round.
        let dyaw = wrap_angle(self.goal.yaw - self.current.yaw);
        self.current.yaw += dyaw * k;
        self.current.pitch += (self.goal.pitch - self.current.pitch) * k;
        self.current.target += (self.goal.target - self.current.target) * k;
        // Distance eases geometrically: linear easing of a log-scaled control
        // crawls when zoomed in and lurches when zoomed out.
        self.current.distance *= (self.goal.distance / self.current.distance).powf(k);
        self.current.fov_y += (self.goal.fov_y - self.current.fov_y) * k;
        self.current.aspect = self.goal.aspect;
        self.current.z_near = self.goal.z_near;
        self.current.sanitise();

        !approx_equal_camera(&before, &self.current)
    }
}

fn wrap_angle(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut x = (a + PI) % TAU;
    if x < 0.0 {
        x += TAU;
    }
    x - PI
}

/// "Has the camera effectively stopped?" Tolerances are in the units of each
/// quantity and chosen so a sub-pixel residual counts as stopped.
fn approx_equal_camera(a: &Camera, b: &Camera) -> bool {
    (a.target - b.target).length() < 1e-3
        && (a.distance - b.distance).abs() < 1e-3
        && (a.yaw - b.yaw).abs() < 1e-5
        && (a.pitch - b.pitch).abs() < 1e-5
        && (a.fov_y - b.fov_y).abs() < 1e-5
}

/// One node of a flythrough path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keyframe {
    /// Time along the path, seconds.
    pub time: f32,
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
}

impl Keyframe {
    pub fn from_camera(time: f32, cam: &Camera) -> Self {
        Self {
            time,
            eye: cam.eye(),
            target: cam.target,
            up: cam.up,
            fov_y: cam.fov_y,
        }
    }
}

/// Catmull-Rom keyframed camera path, for recording videos of a result.
///
/// Catmull-Rom rather than a Bezier because it **passes through** its control
/// points: the keyframes you set are the poses you get, which is the only
/// behaviour anyone wants from a camera path editor.
///
/// The parameterisation is centripetal (`alpha = 0.5`). Uniform Catmull-Rom
/// overshoots into a cusp whenever two keyframes are close together, and camera
/// paths are full of close-together keyframes because that is how you slow down.
#[derive(Debug, Clone, Default)]
pub struct Flythrough {
    pub keyframes: Vec<Keyframe>,
    /// Loop back to the start rather than holding the last pose.
    pub looping: bool,
}

impl Flythrough {
    pub fn new(keyframes: Vec<Keyframe>) -> Self {
        let mut s = Self {
            keyframes,
            looping: false,
        };
        s.sort();
        s
    }

    pub fn push(&mut self, kf: Keyframe) {
        self.keyframes.push(kf);
        self.sort();
    }

    fn sort(&mut self) {
        self.keyframes.sort_by(|a, b| a.time.total_cmp(&b.time));
    }

    pub fn duration(&self) -> f32 {
        match (self.keyframes.first(), self.keyframes.last()) {
            (Some(a), Some(b)) => (b.time - a.time).max(0.0),
            _ => 0.0,
        }
    }

    /// Sample the path. Returns `None` for an empty path.
    pub fn sample(&self, t: f32) -> Option<Keyframe> {
        let n = self.keyframes.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(self.keyframes[0]);
        }

        let t0 = self.keyframes[0].time;
        let span = self.duration();
        let mut t = t;
        if self.looping && span > 0.0 {
            t = t0 + (t - t0).rem_euclid(span);
        }
        let t = t.clamp(t0, t0 + span);

        // Segment containing t.
        let mut i = 0;
        while i + 2 < n && self.keyframes[i + 1].time <= t {
            i += 1;
        }
        let p1 = self.keyframes[i];
        let p2 = self.keyframes[i + 1];
        let dt = (p2.time - p1.time).max(1e-6);
        let u = ((t - p1.time) / dt).clamp(0.0, 1.0);

        // End tangents: duplicate the endpoint, which makes the path start and
        // stop cleanly instead of flicking outward.
        let p0 = self.keyframes[i.saturating_sub(1)];
        let p3 = self.keyframes[(i + 2).min(n - 1)];

        Some(Keyframe {
            time: t,
            eye: catmull_rom_centripetal(p0.eye, p1.eye, p2.eye, p3.eye, u),
            target: catmull_rom_centripetal(p0.target, p1.target, p2.target, p3.target, u),
            // Up and FOV are scalars/near-constant; plain smoothstep avoids any
            // chance of an up-vector overshooting through the pole.
            up: p1.up.lerp(p2.up, smoothstep(u)).normalize_or(Vec3::Y),
            fov_y: p1.fov_y + (p2.fov_y - p1.fov_y) * smoothstep(u),
        })
    }

    /// Write a sampled pose into a camera, converting eye/target back into the
    /// orbit parameters the rest of the app uses.
    pub fn apply(&self, t: f32, cam: &mut Camera) -> bool {
        let Some(kf) = self.sample(t) else {
            return false;
        };
        let d = kf.eye - kf.target;
        cam.target = kf.target;
        cam.distance = d.length().max(cam.z_near * 2.0);
        cam.up = kf.up;
        cam.fov_y = kf.fov_y;
        let n = d.normalize_or(Vec3::Z);
        cam.pitch = n.y.clamp(-1.0, 1.0).asin().clamp(-PITCH_LIMIT, PITCH_LIMIT);
        cam.yaw = n.x.atan2(n.z);
        true
    }
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Centripetal Catmull-Rom on a 4-point stencil, `u` in `[0, 1]` across
/// `p1 -> p2`. Falls back to the uniform form when points coincide.
fn catmull_rom_centripetal(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, u: f32) -> Vec3 {
    let knot = |ti: f32, a: Vec3, b: Vec3| ti + (a - b).length().sqrt().max(1e-4);
    let t0 = 0.0;
    let t1 = knot(t0, p1, p0);
    let t2 = knot(t1, p2, p1);
    let t3 = knot(t2, p3, p2);
    let t = t1 + u * (t2 - t1);

    // Barry-Goldman pyramidal formulation. Reads as three nested lerps and is
    // numerically better behaved than expanding the basis polynomials.
    let lerp = |a: Vec3, b: Vec3, ta: f32, tb: f32| -> Vec3 {
        let d = tb - ta;
        if d.abs() < 1e-9 {
            a
        } else {
            a * ((tb - t) / d) + b * ((t - ta) / d)
        }
    };
    let a1 = lerp(p0, p1, t0, t1);
    let a2 = lerp(p1, p2, t1, t2);
    let a3 = lerp(p2, p3, t2, t3);
    let b1 = lerp(a1, a2, t0, t2);
    let b2 = lerp(a2, a3, t1, t3);
    lerp(b1, b2, t1, t2)
}

/// Halton sequence, used for the TAA sub-pixel jitter.
///
/// Halton(2,3) rather than a random offset because it is low-discrepancy: any
/// prefix of it covers the pixel evenly, so a TAA history that gets reset after
/// 4 frames is still well-distributed. A random sequence clumps.
pub fn halton(index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let mut i = index;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

/// Sub-pixel jitter in NDC for frame `index` at a given resolution.
pub fn taa_jitter(index: u32, width: u32, height: u32) -> Vec2 {
    // +1 because Halton(0) is 0 for every base, which would waste a frame on
    // the un-jittered sample.
    let x = halton(index + 1, 2) - 0.5;
    let y = halton(index + 1, 3) - 0.5;
    Vec2::new(
        2.0 * x / width.max(1) as f32,
        2.0 * y / height.max(1) as f32,
    )
}

/// GPU mirror of the camera. Bound by every pass in the crate at group 0.
///
/// Both jittered and un-jittered matrices are present on purpose: rasterisation
/// and ray generation use the jittered pair so TAA sees a moving sample point,
/// while motion vectors use the un-jittered pair so the jitter does not appear
/// as scene motion.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub view_proj_unjittered: [[f32; 4]; 4],
    pub prev_view_proj_unjittered: [[f32; 4]; 4],

    /// `xyz` = eye position mm, `w` = near plane mm.
    pub eye_near: [f32; 4],
    /// `xyz` = unit forward, `w` = `tan(fov_y / 2)`.
    pub forward_tan: [f32; 4],
    /// `xy` = NDC jitter, `zw` = viewport size in pixels.
    pub jitter_resolution: [f32; 4],
    /// `xy` = 1 / viewport size, `z` = frame index as float, `w` = aspect.
    pub inv_resolution_frame: [f32; 4],
}

impl CameraUniform {
    pub fn new(cam: &Camera, prev_view_proj: Mat4, width: u32, height: u32, frame: u32) -> Self {
        let view = cam.view();
        let proj = cam.projection();
        let vp = proj * view;
        Self {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            view_proj_unjittered: cam.view_projection_unjittered().to_cols_array_2d(),
            prev_view_proj_unjittered: prev_view_proj.to_cols_array_2d(),
            eye_near: cam.eye().extend(cam.z_near).to_array(),
            forward_tan: cam.forward().extend((cam.fov_y * 0.5).tan()).to_array(),
            jitter_resolution: [cam.jitter.x, cam.jitter.y, width as f32, height as f32],
            inv_resolution_frame: [
                1.0 / width.max(1) as f32,
                1.0 / height.max(1) as f32,
                frame as f32,
                cam.aspect,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> Camera {
        Camera {
            target: Vec3::new(10.0, -3.0, 4.0),
            distance: 250.0,
            yaw: 0.9,
            pitch: 0.3,
            fov_y: 50.0_f32.to_radians(),
            aspect: 1.6,
            z_near: 0.25,
            up: Vec3::Y,
            jitter: Vec2::ZERO,
        }
    }

    #[test]
    fn reverse_z_maps_near_to_one_and_far_to_zero() {
        let cam = test_camera();
        let p = cam.projection();
        // A point exactly on the near plane, in view space.
        let near_clip = p * Vec4::new(0.0, 0.0, -cam.z_near, 1.0);
        assert!((near_clip.z / near_clip.w - 1.0).abs() < 1e-5);

        // Depth must decrease monotonically with distance, and approach 0.
        let mut last = 2.0;
        for d in [0.25_f32, 1.0, 10.0, 100.0, 1000.0, 1.0e6] {
            let c = p * Vec4::new(0.0, 0.0, -d, 1.0);
            let z = c.z / c.w;
            assert!(z < last, "depth must decrease with distance: {z} !< {last}");
            assert!((0.0..=1.0).contains(&z), "depth {z} out of range at d={d}");
            last = z;
        }
        assert!(last < 1e-5, "far depth should approach zero, got {last}");
    }

    #[test]
    fn depth_linearisation_round_trips_through_the_projection() {
        // This is the exact chain volume.wgsl performs: rasterise a point,
        // read the depth buffer, recover the view distance. If reverse-Z is
        // applied inconsistently anywhere, this is the test that fails.
        let cam = test_camera();
        let vp = cam.view_projection();
        let view = cam.view();

        for p in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(50.0, 20.0, -30.0),
            Vec3::new(-120.0, 5.0, 200.0),
        ] {
            let clip = vp * p.extend(1.0);
            if clip.w <= 0.0 {
                continue; // behind the camera
            }
            let depth = clip.z / clip.w;
            let expected = -(view * p.extend(1.0)).z; // positive distance along -Z
            let recovered = cam.linear_depth(depth);
            let err = (recovered - expected).abs() / expected;
            assert!(
                err < 1e-4,
                "linear depth {recovered} vs {expected} (rel {err})"
            );
            // And the forward map agrees.
            assert!((cam.depth_from_linear(expected) - depth).abs() < 1e-5);
        }
    }

    #[test]
    fn linear_depth_of_cleared_buffer_is_infinite() {
        // Depth 0 is the clear value in reverse-Z, i.e. "nothing here". The
        // raymarcher relies on this to mean "march to the bbox exit".
        assert_eq!(test_camera().linear_depth(0.0), f32::INFINITY);
    }

    #[test]
    fn ray_through_ndc_centre_is_the_forward_axis() {
        let cam = test_camera();
        let (origin, dir) = cam.ray(Vec2::ZERO);
        assert!(
            dir.dot(cam.forward()) > 0.9999,
            "centre ray must look forward"
        );
        // The origin lies on the near plane, i.e. z_near in front of the eye.
        let along = (origin - cam.eye()).dot(cam.forward());
        assert!(
            (along - cam.z_near).abs() < 1e-3,
            "ray starts at {along}, want {}",
            cam.z_near
        );
    }

    #[test]
    fn corner_rays_open_out_by_the_field_of_view() {
        let cam = test_camera();
        let (_, up_edge) = cam.ray(Vec2::new(0.0, 1.0));
        let half = up_edge.dot(cam.forward()).clamp(-1.0, 1.0).acos();
        assert!(
            (half - cam.fov_y * 0.5).abs() < 1e-3,
            "top edge ray at {half} rad, want {}",
            cam.fov_y * 0.5
        );
    }

    #[test]
    fn framing_puts_the_whole_box_inside_the_frustum_and_fills_it() {
        // The real part's bounding box, at several orbit angles. Every corner
        // must be inside the frame (correctness) *and* at least one must be
        // close to an edge (tightness) — the second half is what a bounding
        // sphere fit fails, leaving the model tiny in the middle of the frame.
        let bbox = Bbox {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(145.0, 72.2, 68.9),
        };
        for (yaw, pitch) in [(0.0, 0.0), (0.9, 0.3), (2.4, -0.8), (-1.1, 1.2)] {
            let mut cam = test_camera();
            cam.yaw = yaw;
            cam.pitch = pitch;
            cam.frame_bbox(bbox, 0.05);

            let vp = cam.view_projection();
            let mut extreme = 0.0f32;
            for i in 0..8 {
                let c = Vec3::new(
                    if i & 1 == 0 { bbox.min.x } else { bbox.max.x },
                    if i & 2 == 0 { bbox.min.y } else { bbox.max.y },
                    if i & 4 == 0 { bbox.min.z } else { bbox.max.z },
                );
                let clip = vp * c.extend(1.0);
                assert!(clip.w > 0.0, "corner {c} ended up behind the camera");
                let ndc = clip.truncate() / clip.w;
                assert!(
                    ndc.x.abs() <= 1.0 + 1e-4 && ndc.y.abs() <= 1.0 + 1e-4,
                    "yaw {yaw}: corner {c} at NDC {ndc}"
                );
                extreme = extreme.max(ndc.x.abs()).max(ndc.y.abs());
            }
            // 5% margin means the tightest corner should sit at ~0.95.
            assert!(
                extreme > 0.9,
                "yaw {yaw}: the box only reaches {extreme} of the frame; the fit is loose"
            );
        }
    }

    #[test]
    fn look_along_recovers_the_requested_direction() {
        let mut cam = test_camera();
        for preset in ViewPreset::ALL {
            cam.look_along(preset.direction());
            let got = cam.forward();
            let want = preset.direction();
            // Top/bottom are clamped off the pole, so allow a milliradian.
            assert!(
                got.dot(want) > 0.99999,
                "{}: got {got}, want {want}",
                preset.label()
            );
        }
    }

    #[test]
    fn smoothing_is_frame_rate_independent() {
        // Same wall-clock elapsed time must land in the same place whether we
        // took one big step or many small ones. A naive lerp fails this badly.
        let mut a = OrbitController::new(test_camera());
        let mut b = a.clone();
        a.goal.yaw += 1.0;
        b.goal.yaw += 1.0;

        a.update(0.1);
        for _ in 0..100 {
            b.update(0.001);
        }
        assert!(
            (a.current.yaw - b.current.yaw).abs() < 2e-3,
            "1x0.1s gave {}, 100x0.001s gave {}",
            a.current.yaw,
            b.current.yaw
        );
    }

    #[test]
    fn smoothing_settles_and_reports_when_it_stops() {
        let mut c = OrbitController::new(test_camera());
        c.goal.distance = 30.0;
        let mut moving = true;
        for _ in 0..2000 {
            moving = c.update(1.0 / 60.0);
            if !moving {
                break;
            }
        }
        assert!(!moving, "controller never reported settling");
        assert!((c.current.distance - 30.0).abs() < 0.05);
    }

    #[test]
    fn yaw_takes_the_short_way_round() {
        let mut c = OrbitController::new(test_camera());
        c.current.yaw = 3.0;
        c.goal.yaw = -3.0; // 0.28 rad away going forwards, 6.0 rad going back
        c.update(0.02);
        assert!(
            c.current.yaw > 3.0,
            "yaw went the long way: {}",
            c.current.yaw
        );
    }

    #[test]
    fn flythrough_passes_exactly_through_its_keyframes() {
        // The whole point of Catmull-Rom over Bezier.
        let kfs: Vec<Keyframe> = (0..5)
            .map(|i| Keyframe {
                time: i as f32,
                eye: Vec3::new(i as f32 * 30.0, (i as f32).sin() * 20.0, 100.0),
                target: Vec3::new(i as f32 * 10.0, 0.0, 0.0),
                up: Vec3::Y,
                fov_y: 0.8,
            })
            .collect();
        let path = Flythrough::new(kfs.clone());
        for kf in &kfs {
            let s = path.sample(kf.time).unwrap();
            assert!(
                (s.eye - kf.eye).length() < 1e-3,
                "at t={}: {} vs {}",
                kf.time,
                s.eye,
                kf.eye
            );
            assert!((s.target - kf.target).length() < 1e-3);
        }
    }

    #[test]
    fn flythrough_is_continuous_and_does_not_overshoot_wildly() {
        let kfs = vec![
            Keyframe {
                time: 0.0,
                eye: Vec3::new(0.0, 0.0, 100.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                fov_y: 0.8,
            },
            Keyframe {
                time: 1.0,
                eye: Vec3::new(1.0, 0.0, 100.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                fov_y: 0.8,
            },
            // Deliberately close together: the case where uniform Catmull-Rom cusps.
            Keyframe {
                time: 1.02,
                eye: Vec3::new(1.05, 0.0, 100.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                fov_y: 0.8,
            },
            Keyframe {
                time: 3.0,
                eye: Vec3::new(60.0, 0.0, 100.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                fov_y: 0.8,
            },
        ];
        let path = Flythrough::new(kfs);
        let mut prev = path.sample(0.0).unwrap().eye;
        let mut max_step = 0.0f32;
        for i in 1..=600 {
            let t = i as f32 / 200.0;
            let p = path.sample(t).unwrap().eye;
            max_step = max_step.max((p - prev).length());
            prev = p;
        }
        // Whole path is 60 mm over 3 s sampled at 200 Hz: ~0.3 mm/step nominal.
        // Anything above a few mm means a cusp blew up.
        assert!(
            max_step < 3.0,
            "path has a discontinuity: max step {max_step} mm"
        );
    }

    #[test]
    fn flythrough_apply_round_trips_eye_and_target() {
        let mut cam = test_camera();
        let path = Flythrough::new(vec![
            Keyframe {
                time: 0.0,
                eye: Vec3::new(30.0, 40.0, 50.0),
                target: Vec3::new(1.0, 2.0, 3.0),
                up: Vec3::Y,
                fov_y: 0.7,
            },
            Keyframe {
                time: 1.0,
                eye: Vec3::new(-30.0, 10.0, 5.0),
                target: Vec3::new(1.0, 2.0, 3.0),
                up: Vec3::Y,
                fov_y: 0.7,
            },
        ]);
        assert!(path.apply(0.0, &mut cam));
        assert!(
            (cam.eye() - Vec3::new(30.0, 40.0, 50.0)).length() < 1e-2,
            "eye was {}",
            cam.eye()
        );
        assert!((cam.target - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
    }

    #[test]
    fn halton_jitter_is_centred_and_bounded() {
        // Both properties matter: an off-centre jitter sequence biases the whole
        // TAA image by a fraction of a pixel and looks like a soft double edge.
        let (w, h) = (1920u32, 1080u32);
        let mut sum = Vec2::ZERO;
        const N: u32 = 64;
        for i in 0..N {
            let j = taa_jitter(i, w, h);
            assert!(
                j.x.abs() <= 1.0 / w as f32 + 1e-6,
                "jitter x out of a pixel: {}",
                j.x
            );
            assert!(j.y.abs() <= 1.0 / h as f32 + 1e-6);
            sum += j;
        }
        let mean = sum / N as f32;
        assert!(
            mean.length() < 0.15 / w as f32,
            "jitter mean {mean} is not centred"
        );
    }

    #[test]
    fn camera_uniform_is_16_byte_aligned() {
        // Uniform buffers in WGSL round struct size up to 16; a mismatch here
        // shows up as garbled matrices rather than a validation error.
        assert_eq!(std::mem::size_of::<CameraUniform>() % 16, 0);
        let cam = test_camera();
        let u = CameraUniform::new(&cam, Mat4::IDENTITY, 1920, 1080, 3);
        assert_eq!(u.eye_near[3], cam.z_near);
        assert!((u.forward_tan[3] - (cam.fov_y * 0.5).tan()).abs() < 1e-6);
    }
}

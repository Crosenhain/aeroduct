//! Where the duct sits in the car: its **install pose**.
//!
//! The solver never sees this. The lattice is laid out in the STL's own frame,
//! where the auto-detected mouths sit flush on axis-aligned faces — the one
//! arrangement the inlet, the outlet and the domain planner are built for — and
//! with no gravity in the model, turning the part turns the air with it: the
//! flow *relative to the part* is the same in every pose. So rotating the part
//! is a change of frame, not of physics, and it is done as one rigid map from
//! the lattice frame onto the world frame that everything is drawn, framed and
//! placed in.
//!
//! Every spatial quantity belongs to exactly one of the two frames:
//!
//! | frame | what lives there |
//! |---|---|
//! | lattice (STL) | voxels, fields, mouths, metric planes, probes, slices |
//! | world (car, `+Y` up) | the camera, view presets, ground plane, obstructions |
//!
//! Probes and slices are lattice-frame because they are placed *on the part*:
//! re-posing it must carry them along rather than leave them in mid-air.
//!
//! The rotation pivots about the part's centre rather than the STL origin,
//! which CAD exports routinely leave on a corner of the part: turning about
//! that swings the duct out of shot.

use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};

use crate::overlays::Placement;

/// Lattice → world, as a rotation about a pivot plus an offset:
///
/// `world = rotation * (lattice - pivot) + pivot + offset_mm`
///
/// The pivot is not stored. It is the duct's bounding-box centre, which the app
/// owns and which only changes when a different STL is loaded, so every method
/// that needs it takes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstallPose {
    pub rotation: Quat,
    pub offset_mm: Vec3,
}

impl Default for InstallPose {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl InstallPose {
    pub const IDENTITY: Self = Self {
        rotation: Quat::IDENTITY,
        offset_mm: Vec3::ZERO,
    };

    /// Exactly the file orientation. Tested exactly, not approximately: the
    /// default pose has to render bit-identically to a build without poses.
    pub fn is_identity(&self) -> bool {
        self.rotation == Quat::IDENTITY && self.offset_mm == Vec3::ZERO
    }

    /// The lattice-to-world matrix. Exactly [`Mat4::IDENTITY`] for the default
    /// pose, not merely close to it.
    pub fn matrix(&self, pivot: Vec3) -> Mat4 {
        if self.is_identity() {
            return Mat4::IDENTITY;
        }
        Mat4::from_rotation_translation(self.rotation, self.translation(pivot))
    }

    /// The world-to-lattice matrix, built from the conjugate rather than by a
    /// general 4x4 inverse: the map is rigid, so this is exact.
    pub fn inverse(&self, pivot: Vec3) -> Mat4 {
        if self.is_identity() {
            return Mat4::IDENTITY;
        }
        let inv = self.rotation.conjugate();
        Mat4::from_rotation_translation(inv, -(inv * self.translation(pivot)))
    }

    /// Where the lattice origin lands in the world.
    fn translation(&self, pivot: Vec3) -> Vec3 {
        pivot + self.offset_mm - self.rotation * pivot
    }

    pub fn to_world(&self, pivot: Vec3, p: Vec3) -> Vec3 {
        self.rotation * (p - pivot) + pivot + self.offset_mm
    }

    pub fn to_lattice(&self, pivot: Vec3, p: Vec3) -> Vec3 {
        self.rotation.conjugate() * (p - pivot - self.offset_mm) + pivot
    }

    pub fn dir_to_world(&self, d: Vec3) -> Vec3 {
        self.rotation * d
    }

    pub fn dir_to_lattice(&self, d: Vec3) -> Vec3 {
        self.rotation.conjugate() * d
    }

    /// The same map as the geometry crate's transform, whose eight-corner
    /// [`ad_geom::Transform::bbox`] gives the posed part's world box.
    pub fn to_geom(&self, pivot: Vec3) -> ad_geom::Transform {
        ad_geom::Transform {
            translation: self.translation(pivot),
            rotation: self.rotation,
            scale: 1.0,
        }
    }

    /// World-space box around a lattice-space box.
    pub fn world_bbox(&self, pivot: Vec3, b: ad_gpu::Bbox) -> ad_gpu::Bbox {
        self.to_geom(pivot).bbox(b)
    }

    /// Where an obstruction placed in the car sits in the lattice: the
    /// transform the voxeliser takes. The obstruction belongs to the car, so
    /// its placement is kept in the world and this is redone whenever the pose
    /// changes.
    pub fn placement_to_lattice(&self, pivot: Vec3, p: Placement) -> ad_geom::Transform {
        if self.is_identity() {
            return p.to_geom();
        }
        ad_geom::Transform {
            translation: self.to_lattice(pivot, p.translation_mm),
            rotation: (self.rotation.conjugate() * p.rotation).normalize(),
            scale: p.scale,
        }
    }

    /// A lattice-frame transform as a placement in the car.
    ///
    /// A newly loaded obstruction starts at the lattice identity — its own STL
    /// coordinates — which for a part exported from the same CAD assembly as
    /// the duct is exactly where it belongs.
    pub fn lattice_to_placement(&self, pivot: Vec3, t: ad_geom::Transform) -> Placement {
        if self.is_identity() {
            return Placement::from_geom(t);
        }
        Placement {
            translation_mm: self.to_world(pivot, t.translation),
            rotation: (self.rotation * t.rotation).normalize(),
            scale: t.scale,
        }
    }

    /// Yaw about world up (`Y`), then pitch about `X`, then roll about `Z`,
    /// degrees. Derived for display; the quaternion is the truth, so the
    /// numbers jumping at ±90° pitch is cosmetic.
    pub fn euler_deg(&self) -> Vec3 {
        let (yaw, pitch, roll) = self.rotation.to_euler(EulerRot::YXZ);
        Vec3::new(yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees())
    }

    pub fn set_euler_deg(&mut self, e: Vec3) {
        self.rotation = canonical(Quat::from_euler(
            EulerRot::YXZ,
            e.x.to_radians(),
            e.y.to_radians(),
            e.z.to_radians(),
        ));
    }

    /// Turn by `degrees` about world axis `axis` (0 = X, 1 = Y, 2 = Z), about
    /// the part's centre.
    pub fn nudge(&mut self, axis: usize, degrees: f32) {
        let a = [Vec3::X, Vec3::Y, Vec3::Z][axis.min(2)];
        self.rotation = canonical(Quat::from_axis_angle(a, degrees.to_radians()) * self.rotation);
    }

    /// The matrix the gizmo manipulates: the rotation, placed at the pivot's
    /// world position, so the handles sit on the part and turn it about its
    /// centre rather than about wherever the STL origin happens to be.
    pub fn gizmo_matrix(&self, pivot: Vec3) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, pivot + self.offset_mm)
    }

    /// Read a pose back out of a gizmo-edited [`Self::gizmo_matrix`]. Scale is
    /// discarded: the pose is rigid by construction.
    pub fn from_gizmo_matrix(m: Mat4, pivot: Vec3) -> Self {
        let (_, rotation, translation) = m.to_scale_rotation_translation();
        Self {
            rotation: canonical(rotation),
            offset_mm: translation - pivot,
        }
    }

    /// `AERODUCT_POSE="yaw,pitch,roll[,x,y,z]"`: degrees, then an optional
    /// offset in mm. `None` when unset; a malformed value is logged and ignored
    /// rather than half-applied.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let raw = get("AERODUCT_POSE")?;
        let values: Option<Vec<f32>> = raw
            .split(',')
            .map(|s| s.trim().parse::<f32>().ok().filter(|v| v.is_finite()))
            .collect();
        let mut pose = Self::IDENTITY;
        match values.as_deref() {
            Some(&[yaw, pitch, roll]) => pose.set_euler_deg(Vec3::new(yaw, pitch, roll)),
            Some(&[yaw, pitch, roll, x, y, z]) => {
                pose.set_euler_deg(Vec3::new(yaw, pitch, roll));
                pose.offset_mm = Vec3::new(x, y, z);
            }
            _ => {
                log::warn!("AERODUCT_POSE={raw:?} is not \"yaw,pitch,roll[,x,y,z]\"; ignored");
                return None;
            }
        }
        Some(pose)
    }

    pub fn from_env() -> Option<Self> {
        Self::from_lookup(|k| std::env::var(k).ok())
    }
}

/// Unit length, `w >= 0`, and float dust from a chain of quarter turns cleaned
/// up, so ±90° steps land on exact values: four nudges about one axis give back
/// exactly the identity, and the readout says 90, not 89.99998.
fn canonical(q: Quat) -> Quat {
    const EPS: f32 = 1.0e-6;
    let mut q = q.normalize();
    if q.w < 0.0 {
        q = -q;
    }
    let clean = |c: f32| -> f32 {
        for exact in [0.0, 0.5, std::f32::consts::FRAC_1_SQRT_2, 1.0] {
            if (c.abs() - exact).abs() < EPS {
                return if exact == 0.0 { 0.0 } else { exact.copysign(c) };
            }
        }
        c
    };
    Quat::from_xyzw(clean(q.x), clean(q.y), clean(q.z), clean(q.w))
}

/// The duct's own transform in the lattice: a real change of the part, unlike
/// the install pose, which only turns the picture.
///
/// Two things can be done to the geometry without breaking the lattice's
/// assumptions: a uniform scale, and a turn by quarter turns, which maps the
/// mouths' faces onto other box faces. Both are about the part's centre, so
/// the part stays where it was and the install pose's pivot does not move.
/// Changing either re-voxelises and restarts the flow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuctGeometry {
    pub scale: f32,
    /// One of the 24 quarter-turn rotations, or the identity.
    pub turn: Quat,
}

impl Default for DuctGeometry {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl DuctGeometry {
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        turn: Quat::IDENTITY,
    };

    pub fn is_identity(&self) -> bool {
        self.scale == 1.0 && self.turn == Quat::IDENTITY
    }

    /// The geometry crate's transform, keeping `centre` (the STL's bounding-box
    /// centre in its own coordinates) exactly where it is.
    pub fn to_geom(&self, centre: Vec3) -> ad_geom::Transform {
        let s = self.scale.max(1.0e-3);
        ad_geom::Transform {
            translation: centre - self.turn * (centre * s),
            rotation: self.turn,
            scale: s,
        }
    }
}

/// The quarter-turn rotation nearest `q`: one of the 24 that map the lattice
/// axes onto themselves. What "bake the pose into the part" turns by.
pub fn nearest_quarter_turn(q: Quat) -> Quat {
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let mut best = (Quat::IDENTITY, -1.0f32);
    for p in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for signs in 0..8u32 {
            let col = |k: usize| axes[p[k]] * if signs & (1 << k) != 0 { -1.0 } else { 1.0 };
            let m = glam::Mat3::from_cols(col(0), col(1), col(2));
            if m.determinant() < 0.0 {
                continue;
            }
            let c = canonical(Quat::from_mat3(&m));
            let d = q.dot(c).abs();
            if d > best.1 {
                best = (c, d);
            }
        }
    }
    best.0
}

/// The inlet's off-normal tilt in the part's frame — what
/// [`crate::params::SimParams::inlet_tilt_deg`] holds — from a vent louver aim
/// given in car terms.
///
/// `louver_deg = [up_down, sideways]`: `up_down` tips the air toward car up
/// (`+Y`) as seen in the inlet plane, `sideways` toward `n × up`, which is to
/// the right looking downstream. The louver is part of the car, so one aim is a
/// different tilt relative to the part in every pose; that is why the aim, not
/// the tilt, is what the UI keeps.
///
/// `n` is the inlet mouth's inward normal in the lattice frame, `axis` its
/// lattice axis, `rotation` the install pose's.
pub fn louver_to_tilt(louver_deg: [f32; 2], n: Vec3, axis: usize, rotation: Quat) -> [f32; 2] {
    if louver_deg == [0.0, 0.0] {
        return [0.0, 0.0];
    }
    let n = n.normalize_or(Vec3::AXES[axis % 3]);
    let e1 = Vec3::AXES[(axis + 1) % 3];
    let e2 = Vec3::AXES[(axis + 2) % 3];
    // Car up, seen from the part, flattened into the inlet plane. An inlet
    // facing straight up or down has no "up" in its plane; any fixed in-plane
    // axis will do there, and e1 is one that does not move.
    let up_lattice = rotation.conjugate() * Vec3::Y;
    let flat = up_lattice - n * up_lattice.dot(n);
    let up = if flat.length() > 1.0e-3 {
        flat.normalize()
    } else {
        e1
    };
    let right = n.cross(up);
    let t = up * louver_deg[0].to_radians().tan() + right * louver_deg[1].to_radians().tan();
    // `t` lies in the inlet plane by construction, so this is an exact change of
    // 2D basis. `+ 0.0` folds a -0.0, which would hash differently from 0.0 and
    // reset the statistics for nothing.
    [
        t.dot(e1).atan().to_degrees() + 0.0,
        t.dot(e2).atan().to_degrees() + 0.0,
    ]
}

/// The lattice axis an axis-aligned mouth normal lies along.
pub fn lattice_axis(n: Vec3) -> usize {
    (0..3)
        .max_by(|a, b| n[*a].abs().total_cmp(&n[*b].abs()))
        .unwrap_or(0)
}

/// `AERODUCT_INLET_TILT="up_down,sideways"`: a louver aim in degrees, car terms,
/// each clamped to [`crate::params::MAX_INLET_TILT_DEG`].
pub fn louver_from_lookup(get: impl Fn(&str) -> Option<String>) -> Option<[f32; 2]> {
    let raw = get("AERODUCT_INLET_TILT")?;
    let values: Option<Vec<f32>> = raw
        .split(',')
        .map(|s| s.trim().parse::<f32>().ok().filter(|v| v.is_finite()))
        .collect();
    let max = crate::params::MAX_INLET_TILT_DEG;
    match values.as_deref() {
        Some(&[a, b]) => Some([a.clamp(-max, max), b.clamp(-max, max)]),
        _ => {
            log::warn!("AERODUCT_INLET_TILT={raw:?} is not \"up_down,sideways\"; ignored");
            None
        }
    }
}

/// A world direction in the words someone standing in the car uses: how far
/// above or below level it points, and which way across the floor.
pub fn describe_direction(d: Vec3) -> String {
    let d = d.normalize_or_zero();
    if d == Vec3::ZERO {
        return "-".into();
    }
    if Vec2::new(d.x, d.z).length() < 1.0e-3 {
        return if d.y > 0.0 {
            "straight up".into()
        } else {
            "straight down".into()
        };
    }
    let elevation = d.y.clamp(-1.0, 1.0).asin().to_degrees();
    let vertical = if elevation.abs() < 0.5 {
        "level".to_string()
    } else if elevation > 0.0 {
        format!("{elevation:.0}° up")
    } else {
        format!("{:.0}° down", -elevation)
    };
    let (axis, sign) = if d.x.abs() >= d.z.abs() {
        ("X", d.x)
    } else {
        ("Z", d.z)
    };
    format!(
        "{vertical}, toward {}{axis}",
        if sign >= 0.0 { "+" } else { "-" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose() -> InstallPose {
        let mut p = InstallPose::IDENTITY;
        p.set_euler_deg(Vec3::new(30.0, -20.0, 45.0));
        p.offset_mm = Vec3::new(5.0, -3.0, 12.0);
        p
    }

    const PIVOT: Vec3 = Vec3::new(-1.7, 36.1, 34.5);

    #[test]
    fn the_default_pose_is_exactly_the_identity() {
        let p = InstallPose::default();
        assert!(p.is_identity());
        assert_eq!(p.matrix(PIVOT), Mat4::IDENTITY);
        assert_eq!(p.inverse(PIVOT), Mat4::IDENTITY);
        assert_eq!(
            p.to_world(PIVOT, Vec3::new(1.0, 2.0, 3.0)),
            Vec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn matrix_point_map_and_inverse_agree() {
        let p = pose();
        let m = p.matrix(PIVOT);
        assert!((p.inverse(PIVOT) * m).abs_diff_eq(Mat4::IDENTITY, 1e-5));
        for q in [
            Vec3::ZERO,
            PIVOT,
            Vec3::new(70.8, 72.2, 68.9),
            Vec3::new(-74.2, 0.0, 0.0),
        ] {
            let w = p.to_world(PIVOT, q);
            assert!(m.transform_point3(q).abs_diff_eq(w, 1e-3), "{q} -> {w}");
            assert!(p.to_lattice(PIVOT, w).abs_diff_eq(q, 1e-3));
            assert!(p.to_geom(PIVOT).point(q).abs_diff_eq(w, 1e-3));
        }
    }

    #[test]
    fn the_part_turns_about_its_own_centre() {
        let mut p = InstallPose::IDENTITY;
        p.nudge(1, 90.0);
        assert!(p.to_world(PIVOT, PIVOT).abs_diff_eq(PIVOT, 1e-5));
        p.offset_mm = Vec3::new(10.0, 0.0, 0.0);
        assert!(p
            .to_world(PIVOT, PIVOT)
            .abs_diff_eq(PIVOT + Vec3::X * 10.0, 1e-5));
    }

    #[test]
    fn quarter_turns_are_exact() {
        for axis in 0..3 {
            let mut p = InstallPose::IDENTITY;
            for _ in 0..4 {
                p.nudge(axis, 90.0);
            }
            assert!(p.is_identity(), "axis {axis}: {:?}", p.rotation);
            p.nudge(axis, 90.0);
            p.nudge(axis, -90.0);
            assert!(
                p.is_identity(),
                "axis {axis} there and back: {:?}",
                p.rotation
            );
        }
        let mut p = InstallPose::IDENTITY;
        p.nudge(1, 90.0);
        assert!(
            p.dir_to_world(Vec3::Z).abs_diff_eq(Vec3::X, 1e-6),
            "a quarter turn maps axes onto axes"
        );
    }

    #[test]
    fn a_car_placement_and_its_lattice_transform_agree() {
        let p = pose();
        let placed = Placement {
            translation_mm: Vec3::new(30.0, -12.0, 80.0),
            rotation: Quat::from_euler(EulerRot::YXZ, 0.3, -0.2, 0.9),
            scale: 1.5,
        };
        let t = p.placement_to_lattice(PIVOT, placed);
        for v in [Vec3::ZERO, Vec3::new(10.0, 2.0, -7.0), Vec3::splat(40.0)] {
            let world = placed.rotation * (v * placed.scale) + placed.translation_mm;
            assert!(
                p.to_world(PIVOT, t.point(v)).abs_diff_eq(world, 1e-3),
                "{v}"
            );
        }
        let back = p.lattice_to_placement(PIVOT, t);
        assert!(back.translation_mm.abs_diff_eq(placed.translation_mm, 1e-3));
        assert!(back.rotation.abs_diff_eq(placed.rotation, 1e-5));
        assert_eq!(back.scale, placed.scale);
        // Unposed, the two frames are one frame, exactly.
        let id = InstallPose::IDENTITY;
        assert_eq!(id.placement_to_lattice(PIVOT, placed), placed.to_geom());
        assert_eq!(id.lattice_to_placement(PIVOT, placed.to_geom()), placed);
    }

    #[test]
    fn euler_angles_round_trip() {
        let e = pose().euler_deg();
        assert!(e.abs_diff_eq(Vec3::new(30.0, -20.0, 45.0), 1e-3), "{e}");
    }

    #[test]
    fn the_gizmo_matrix_round_trips_and_sits_on_the_part() {
        let p = pose();
        let g = p.gizmo_matrix(PIVOT);
        assert!(g
            .w_axis
            .truncate()
            .abs_diff_eq(p.to_world(PIVOT, PIVOT), 1e-4));
        let back = InstallPose::from_gizmo_matrix(g, PIVOT);
        assert!(back.rotation.abs_diff_eq(p.rotation, 1e-5));
        assert!(back.offset_mm.abs_diff_eq(p.offset_mm, 1e-4));
        // A stray scale from the gizmo does not survive.
        let scaled = g * Mat4::from_scale(Vec3::splat(2.0));
        let back = InstallPose::from_gizmo_matrix(scaled, PIVOT);
        assert!(back.rotation.abs_diff_eq(p.rotation, 1e-5));
    }

    #[test]
    fn the_env_var_parses_or_is_refused_whole() {
        let get = |v: &'static str| move |k: &str| (k == "AERODUCT_POSE").then(|| v.to_string());
        assert_eq!(InstallPose::from_lookup(|_| None), None);
        assert_eq!(
            InstallPose::from_lookup(get("0,0,0")),
            Some(InstallPose::IDENTITY)
        );
        let p = InstallPose::from_lookup(get(" 0, 90 ,0")).unwrap();
        assert!(
            p.dir_to_world(Vec3::Z).abs_diff_eq(Vec3::NEG_Y, 1e-6),
            "pitch +90 tips +Z down"
        );
        let p = InstallPose::from_lookup(get("0,0,0,4,5,6")).unwrap();
        assert_eq!(p.offset_mm, Vec3::new(4.0, 5.0, 6.0));
        for bad in ["a,b,c", "1,2", "1,2,3,4", "nan,0,0", "inf,0,0", ""] {
            assert_eq!(InstallPose::from_lookup(get(bad)), None, "{bad:?}");
        }
    }

    fn blow(tilt: [f32; 2], n: Vec3, axis: usize) -> Vec3 {
        let mut p = crate::params::SimParams::default();
        p.inlet_tilt_deg = tilt;
        p.inlet_direction(n, axis)
    }

    #[test]
    fn a_level_louver_is_no_tilt_at_all() {
        let q = pose().rotation;
        let zero = [0.0f32.to_bits(); 2];
        assert_eq!(
            louver_to_tilt([0.0, 0.0], Vec3::Z, 2, q).map(f32::to_bits),
            zero
        );
        assert_eq!(
            louver_to_tilt([-0.0, 0.0], Vec3::Z, 2, q).map(f32::to_bits),
            zero
        );
    }

    /// "20° up" means 20° above level in the car, however the part is turned,
    /// for an inlet that faces horizontally in the car.
    #[test]
    fn louver_up_is_car_up_in_every_pose() {
        let mut poses = vec![InstallPose::IDENTITY];
        for (axis, deg) in [(1, 90.0), (2, 90.0), (2, -90.0), (1, 180.0)] {
            let mut p = InstallPose::IDENTITY;
            p.nudge(axis, deg);
            poses.push(p);
        }
        let mut p = InstallPose::IDENTITY;
        p.set_euler_deg(Vec3::new(37.0, 0.0, 25.0));
        poses.push(p);

        // Yaw about Y and roll about Z both keep a +Z inlet horizontal.
        let n = Vec3::Z;
        for pose in poses {
            let up =
                pose.dir_to_world(blow(louver_to_tilt([20.0, 0.0], n, 2, pose.rotation), n, 2));
            let elevation = up.normalize().y.asin().to_degrees();
            assert!((elevation - 20.0).abs() < 1e-3, "{pose:?}: {elevation}");

            let side =
                pose.dir_to_world(blow(louver_to_tilt([0.0, 15.0], n, 2, pose.rotation), n, 2));
            let side = side.normalize();
            assert!(side.y.abs() < 1e-5, "sideways stays level: {pose:?}");
            let off = side.dot(pose.dir_to_world(n)).acos().to_degrees();
            assert!((off - 15.0).abs() < 1e-3, "{pose:?}: {off}");
            let right = pose.dir_to_world(n).cross(Vec3::Y);
            assert!(
                side.dot(right) > 0.0,
                "positive is to the right, looking downstream"
            );
        }
    }

    #[test]
    fn an_inlet_facing_straight_up_still_tilts() {
        let d = blow(
            louver_to_tilt([20.0, 0.0], Vec3::Y, 1, Quat::IDENTITY),
            Vec3::Y,
            1,
        );
        assert!(d.is_finite());
        let off = d.normalize().dot(Vec3::Y).acos().to_degrees();
        assert!((off - 20.0).abs() < 1e-3, "{off}");
    }

    #[test]
    fn the_louver_env_var_parses_or_is_refused() {
        let get =
            |v: &'static str| move |k: &str| (k == "AERODUCT_INLET_TILT").then(|| v.to_string());
        assert_eq!(louver_from_lookup(|_| None), None);
        assert_eq!(louver_from_lookup(get("20, -5")), Some([20.0, -5.0]));
        assert_eq!(
            louver_from_lookup(get("90,0")),
            Some([60.0, 0.0]),
            "clamped"
        );
        for bad in ["20", "a,b", "1,2,3", "nan,0"] {
            assert_eq!(louver_from_lookup(get(bad)), None, "{bad:?}");
        }
    }

    #[test]
    fn the_nearest_quarter_turn_snaps_and_keeps_the_identity() {
        assert_eq!(nearest_quarter_turn(Quat::IDENTITY), Quat::IDENTITY);
        let near = Quat::from_rotation_y(80f32.to_radians());
        let q = nearest_quarter_turn(near);
        assert!(
            q.abs_diff_eq(Quat::from_rotation_y(90f32.to_radians()), 1e-6),
            "{q:?}"
        );
        let small = Quat::from_rotation_x(30f32.to_radians());
        assert_eq!(nearest_quarter_turn(small), Quat::IDENTITY);
        // Every candidate is a proper rotation that maps axes onto axes.
        let q = nearest_quarter_turn(Quat::from_euler(EulerRot::YXZ, 1.4, -1.6, 0.2));
        for a in [Vec3::X, Vec3::Y, Vec3::Z] {
            let r = q * a;
            assert!((r.abs().max_element() - 1.0).abs() < 1e-6, "{r}");
        }
    }

    #[test]
    fn duct_geometry_keeps_the_part_centred() {
        let c = Vec3::new(-1.7, 36.1, 34.5);
        let g = DuctGeometry {
            scale: 1.5,
            turn: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        };
        let t = g.to_geom(c);
        assert!(t.point(c).abs_diff_eq(c, 1e-4));
        let corner = c + Vec3::new(10.0, 0.0, 0.0);
        assert!(
            t.point(corner)
                .abs_diff_eq(c + Vec3::new(0.0, 15.0, 0.0), 1e-4),
            "{}",
            t.point(corner)
        );
        assert_eq!(
            DuctGeometry::IDENTITY.to_geom(c),
            ad_geom::Transform::IDENTITY
        );
    }

    #[test]
    fn directions_read_like_a_person_would_say_them() {
        assert_eq!(describe_direction(Vec3::Y), "straight up");
        assert_eq!(describe_direction(Vec3::NEG_Y * 3.0), "straight down");
        assert_eq!(describe_direction(Vec3::X), "level, toward +X");
        assert_eq!(
            describe_direction(Vec3::new(0.0, -1.0, -1.0)),
            "45° down, toward -Z"
        );
        assert_eq!(describe_direction(Vec3::ZERO), "-");
    }
}

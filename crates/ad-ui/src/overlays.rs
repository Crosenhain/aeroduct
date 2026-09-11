//! Settings for the flow-visualisation overlays, and the probe list.
//!
//! These are **view models too**: the overlay passes themselves are being
//! written in `ad-render` in parallel with this crate, so the UI edits plain
//! structs here and Wave 3's adapter copies them into whatever the passes turn
//! out to want. The panels never reference an overlay type.
//!
//! Everything spatial is in millimetres and world space, matching the contract,
//! so a slice plane's origin can be handed straight to ImGuizmo alongside the
//! duct's own transform without a unit change in between.

use glam::{Mat4, Quat, Vec3};

/// What a transform gizmo is currently doing.
///
/// Mirrors ImGuizmo's operation set, but as our own enum: the UI stores this in
/// state that is saved and tested, and pinning it to a third-party bitflag
/// would drag that dependency into every test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoMode {
    #[default]
    Translate,
    Rotate,
    Scale,
    /// Gizmo suppressed even though something is selected. Useful when the
    /// handles are covering the thing you are trying to look at.
    Off,
}

impl GizmoMode {
    pub fn label(self) -> &'static str {
        match self {
            GizmoMode::Translate => "move",
            GizmoMode::Rotate => "rotate",
            GizmoMode::Scale => "scale",
            GizmoMode::Off => "off",
        }
    }
    pub const ALL: [GizmoMode; 4] =
        [GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale, GizmoMode::Off];
}

/// Local versus world axes for the gizmo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoSpace {
    #[default]
    World,
    Local,
}

impl GizmoSpace {
    pub fn label(self) -> &'static str {
        match self {
            GizmoSpace::World => "world",
            GizmoSpace::Local => "local",
        }
    }
}

/// The snap ImGuizmo should use in `mode`: millimetres when moving, degrees
/// when turning, none when scaling.
///
/// ImGuizmo takes one snap value and reads it in the units of whatever the
/// operation is, so handing it the translation snap in rotate mode turns a
/// 0.5 mm grid into 0.5° steps — which feels like no snap at all.
pub fn snap_for(mode: GizmoMode, snap_mm: Option<f32>, snap_deg: Option<f32>) -> Option<f32> {
    match mode {
        GizmoMode::Translate => snap_mm,
        GizmoMode::Rotate => snap_deg,
        GizmoMode::Scale | GizmoMode::Off => None,
    }
}

/// A rigid transform in millimetres, in the shape ImGuizmo wants.
///
/// Stored decomposed rather than as a matrix because that is what
/// `ad_geom::Transform` takes and what the numeric fields in the properties
/// panel edit; the matrix is generated on demand for the gizmo and consumed
/// back through [`Placement::from_matrix`]. Round-tripping through a matrix
/// every frame would accumulate drift in the rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub translation_mm: Vec3,
    pub rotation: Quat,
    pub scale: f32,
}

impl Default for Placement {
    fn default() -> Self {
        Self { translation_mm: Vec3::ZERO, rotation: Quat::IDENTITY, scale: 1.0 }
    }
}

impl Placement {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            self.rotation,
            self.translation_mm,
        )
    }

    /// Read a placement back out of a gizmo-edited matrix.
    ///
    /// Uniform scale is recovered as the mean of the three column lengths and
    /// clamped positive. A negative or zero scale mirrors the mesh, which flips
    /// every triangle winding and turns the signed distance field inside out —
    /// the voxeliser would then fill the duct solid and the solver would see no
    /// fluid at all. `ad_geom::Scene::set_transform` asserts on it; refusing it
    /// here means the user sees a gizmo that will not shrink past zero instead
    /// of a panic.
    pub fn from_matrix(m: Mat4) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        let s = ((scale.x.abs() + scale.y.abs() + scale.z.abs()) / 3.0).max(1e-4);
        Self { translation_mm: translation, rotation, scale: s }
    }

    /// Convert to the geometry crate's transform, which the voxeliser takes.
    pub fn to_geom(&self) -> ad_geom::Transform {
        ad_geom::Transform {
            translation: self.translation_mm,
            rotation: self.rotation,
            scale: self.scale,
        }
    }

    pub fn from_geom(t: ad_geom::Transform) -> Self {
        Self { translation_mm: t.translation, rotation: t.rotation, scale: t.scale }
    }
}

/// How the particle overlay behaves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleSettings {
    pub enabled: bool,
    pub count: u32,
    /// Seconds a particle lives before being re-seeded. Bounded because a
    /// particle caught in a recirculation would otherwise never leave, and the
    /// visualisation would slowly fill the eddy with stale dots.
    pub lifetime_s: f32,
    /// Multiplier on the advection rate. Not physical: the sim is thousands of
    /// times slower than real time, so particles moving at the true rate would
    /// be motionless on screen. Labelled as a display speed in the UI so nobody
    /// reads a residence time off it.
    pub speed_scale: f32,
    /// Screen-space radius, pixels.
    pub size_px: f32,
    /// Trail length in frames; 0 draws points.
    pub trail: u32,
    /// Where they come from. Seeding at the inlet answers "where does the air
    /// go"; seeding everywhere answers "what does the whole field look like".
    pub seed: ParticleSeed,
    /// Colour particles by the displayed scalar rather than a flat colour.
    pub color_by_field: bool,
    /// Fraction of particles re-seeded per second, so the display refreshes
    /// even in a steady flow.
    pub reseed_rate: f32,
}

impl Default for ParticleSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            count: 20_000,
            lifetime_s: 4.0,
            speed_scale: 1.0,
            size_px: 2.5,
            trail: 8,
            seed: ParticleSeed::Inlet,
            color_by_field: true,
            reseed_rate: 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParticleSeed {
    /// Uniformly over the inlet patch.
    #[default]
    Inlet,
    /// Uniformly through the fluid volume.
    Volume,
    /// From the selected slice plane, which makes the slice a rake.
    Slice,
    /// From the probe positions.
    Probes,
}

impl ParticleSeed {
    pub fn label(self) -> &'static str {
        match self {
            ParticleSeed::Inlet => "inlet",
            ParticleSeed::Volume => "volume",
            ParticleSeed::Slice => "slice plane",
            ParticleSeed::Probes => "probes",
        }
    }
    pub const ALL: [ParticleSeed; 4] =
        [ParticleSeed::Inlet, ParticleSeed::Volume, ParticleSeed::Slice, ParticleSeed::Probes];
}

/// One cutting plane.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceSettings {
    pub name: String,
    pub visible: bool,
    /// A point on the plane, mm.
    pub origin_mm: Vec3,
    /// Plane normal, unit. Edited by the gizmo's rotation handles.
    pub normal: Vec3,
    /// Draw the scalar field on the plane.
    pub show_scalar: bool,
    /// Draw in-plane velocity arrows.
    pub show_vectors: bool,
    /// Arrow spacing, mm.
    pub vector_spacing_mm: f32,
    /// Arrow length per m/s, mm. A display gain, not a physical quantity.
    pub vector_scale: f32,
    /// Draw line-integral-convolution texture instead of arrows. Reads the
    /// structure of a separated region far better than arrows do, at the cost
    /// of hiding the magnitude.
    pub lic: bool,
    /// Clip the geometry on the far side of the plane, so the plane cuts the
    /// duct open rather than floating inside it.
    pub clip_geometry: bool,
    pub opacity: f32,
}

impl Default for SliceSettings {
    fn default() -> Self {
        Self {
            name: "slice".into(),
            visible: true,
            origin_mm: Vec3::ZERO,
            normal: Vec3::Y,
            show_scalar: true,
            show_vectors: false,
            vector_spacing_mm: 2.0,
            vector_scale: 1.0,
            lic: false,
            clip_geometry: false,
            opacity: 1.0,
        }
    }
}

impl SliceSettings {
    /// A plane through `center` facing along `axis` (0/1/2).
    pub fn axis_aligned(name: impl Into<String>, center: Vec3, axis: u8) -> Self {
        let normal = match axis % 3 {
            0 => Vec3::X,
            1 => Vec3::Y,
            _ => Vec3::Z,
        };
        Self { name: name.into(), origin_mm: center, normal, ..Default::default() }
    }

    /// Placement for the gizmo: the translation is the plane origin and the
    /// rotation takes `+Y` onto the normal.
    ///
    /// `+Y` rather than `+Z` because ImGuizmo draws its rotation rings around
    /// the local axes and the plane visual is a quad in the local XZ plane;
    /// choosing the axis that is *normal* to that quad puts the useful ring
    /// where the user expects to grab it.
    pub fn placement(&self) -> Placement {
        Placement {
            translation_mm: self.origin_mm,
            rotation: Quat::from_rotation_arc(Vec3::Y, self.normal.normalize_or(Vec3::Y)),
            scale: 1.0,
        }
    }

    pub fn set_placement(&mut self, p: Placement) {
        self.origin_mm = p.translation_mm;
        self.normal = (p.rotation * Vec3::Y).normalize_or(Vec3::Y);
    }

    /// Signed distance from the plane, mm. Positive on the normal's side.
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        (p - self.origin_mm).dot(self.normal.normalize_or(Vec3::Y))
    }
}

/// A vent: a rectangle standing in the car that blows air. With any vent
/// present the duct's mouths are plain openings and the vents are the only
/// supply, so what enters the duct is whatever their jets bring — measured,
/// not prescribed.
#[derive(Debug, Clone, PartialEq)]
pub struct VentSettings {
    pub name: String,
    /// Where it is and which way it faces, car frame. Local `+Z` is the way
    /// it blows, `+X` runs along its width, `+Y` along its height. Scale is
    /// ignored; the size is below.
    pub placement: Placement,
    pub width_mm: f32,
    pub height_mm: f32,
    /// Air speed as a multiple of the inlet U on the toolbar.
    pub speed_scale: f32,
    /// Louver aim relative to the vent's face, degrees: toward its `+Y` (up)
    /// and toward its `+X` (right, looking downstream).
    pub aim_deg: [f32; 2],
}

impl VentSettings {
    /// Largest louver aim, degrees. Same cap as the mouth inlet's.
    pub const MAX_AIM_DEG: f32 = 60.0;

    pub fn new(name: impl Into<String>, placement: Placement, width_mm: f32, height_mm: f32) -> Self {
        Self {
            name: name.into(),
            placement: Placement { scale: 1.0, ..placement },
            width_mm: width_mm.max(1.0),
            height_mm: height_mm.max(1.0),
            speed_scale: 1.0,
            aim_deg: [0.0, 0.0],
        }
    }

    /// The way the face points, unit, car frame.
    pub fn normal(&self) -> Vec3 {
        (self.placement.rotation * Vec3::Z).normalize_or(Vec3::Z)
    }

    /// The rectangle's width and height directions, unit, car frame.
    pub fn axes(&self) -> (Vec3, Vec3) {
        let r = self.placement.rotation;
        ((r * Vec3::X).normalize_or(Vec3::X), (r * Vec3::Y).normalize_or(Vec3::Y))
    }

    /// The way the air leaves, unit, car frame: the normal turned by the aim.
    pub fn direction(&self) -> Vec3 {
        let n = self.normal();
        let (x, y) = self.axes();
        let cap = Self::MAX_AIM_DEG;
        let a = self.aim_deg[0].clamp(-cap, cap).to_radians().tan();
        let b = self.aim_deg[1].clamp(-cap, cap).to_radians().tan();
        (n + y * a + x * b).normalize_or(n)
    }

    /// The four corners, car frame, for drawing.
    pub fn corners(&self) -> [Vec3; 4] {
        let c = self.placement.translation_mm;
        let (x, y) = self.axes();
        let (hw, hh) = (x * self.width_mm * 0.5, y * self.height_mm * 0.5);
        [c - hw - hh, c + hw - hh, c + hw + hh, c - hw + hh]
    }

    /// Whether the cells this flags are the same: place and size, not air.
    pub fn same_cells(&self, o: &Self) -> bool {
        self.placement == o.placement
            && self.width_mm.to_bits() == o.width_mm.to_bits()
            && self.height_mm.to_bits() == o.height_mm.to_bits()
    }
}

/// The soft-isosurface overlay, which is the transfer function's soft-iso mode
/// promoted to its own layer so it can be shown alongside a volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsoSettings {
    pub enabled: bool,
    /// Isolevel in **data units** of the displayed field, for the same reason
    /// `ad_render::SoftIso::center` is: re-ranging the colour bar must not move
    /// the surface.
    pub level: f32,
    pub opacity: f32,
    /// Colour the surface by a second scalar rather than flat.
    pub color_by_field: bool,
    /// Show only the side facing the camera, so the surface does not hide its
    /// own interior.
    pub single_sided: bool,
}

impl Default for IsoSettings {
    fn default() -> Self {
        // 0.5 in Q-tilde: the middle of the useful 0.1-2 band, matching
        // `ad_render::SoftIso::default`.
        Self { enabled: false, level: 0.5, opacity: 0.85, color_by_field: false, single_sided: false }
    }
}

/// Streamline seeding and integration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamlineSettings {
    pub enabled: bool,
    pub count: u32,
    /// Integration steps per line.
    pub max_steps: u32,
    /// Step length in cells.
    pub step_cells: f32,
    /// Integrate backwards as well, so a line through a probe shows where the
    /// air came from as well as where it goes.
    pub bidirectional: bool,
    pub width_px: f32,
}

impl Default for StreamlineSettings {
    fn default() -> Self {
        Self { enabled: false, count: 256, max_steps: 2000, step_cells: 0.5, bidirectional: true, width_px: 1.5 }
    }
}

/// The probe list, with stable ids.
///
/// Ids never repeat within a session, because a probe's time series is keyed by
/// id and reusing one would splice two probes' histories into a single plot —
/// which looks like a physical transient rather than a bookkeeping error.
#[derive(Debug, Clone, Default)]
pub struct Probes {
    items: Vec<crate::view::ProbeView>,
    next_id: u32,
    /// How many points each probe's traces keep.
    pub history: usize,
}

impl Probes {
    pub fn new() -> Self {
        Self { items: Vec::new(), next_id: 1, history: 2048 }
    }

    pub fn add(&mut self, position_mm: Vec3) -> u32 {
        let id = self.next_id.max(1);
        self.next_id = id + 1;
        let mut p = crate::view::ProbeView::new(id, position_mm);
        p.selected = true;
        for other in &mut self.items {
            other.selected = false;
        }
        self.items.push(p);
        id
    }

    pub fn remove(&mut self, id: u32) -> bool {
        let before = self.items.len();
        self.items.retain(|p| p.id != id);
        self.items.len() != before
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn get(&self, id: u32) -> Option<&crate::view::ProbeView> {
        self.items.iter().find(|p| p.id == id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut crate::view::ProbeView> {
        self.items.iter_mut().find(|p| p.id == id)
    }

    pub fn items(&self) -> &[crate::view::ProbeView] {
        &self.items
    }

    pub fn items_mut(&mut self) -> &mut [crate::view::ProbeView] {
        &mut self.items
    }

    pub fn select(&mut self, id: Option<u32>) {
        for p in &mut self.items {
            p.selected = Some(p.id) == id;
        }
    }

    pub fn selected(&self) -> Option<&crate::view::ProbeView> {
        self.items.iter().find(|p| p.selected)
    }

    pub fn selected_mut(&mut self) -> Option<&mut crate::view::ProbeView> {
        self.items.iter_mut().find(|p| p.selected)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Append a sample to one probe's traces.
    pub fn record(&mut self, id: u32, t: f64, pressure_pa: f64, velocity: Vec3, in_fluid: bool) {
        let history = self.history;
        if let Some(p) = self.get_mut(id) {
            p.in_fluid = in_fluid;
            p.velocity_ms = velocity;
            p.pressure.push(t, pressure_pa, history);
            p.speed.push(t, velocity.length() as f64, history);
        }
    }

    /// Drop every recorded sample, keeping the probes themselves.
    ///
    /// Called on a statistics reset: a probe trace that spans a boundary
    /// condition change shows a step the flow never took.
    pub fn clear_history(&mut self) {
        for p in &mut self.items {
            p.pressure.clear();
            p.speed.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slice_placement_round_trips_through_the_gizmo_matrix() {
        // The gizmo hands back a matrix; the plane has to come out pointing the
        // same way it went in, or dragging a slice slowly rotates it.
        let mut s = SliceSettings::axis_aligned("s", Vec3::new(1.0, 2.0, 3.0), 2);
        let p = s.placement();
        let back = Placement::from_matrix(p.matrix());
        s.set_placement(back);
        assert!((s.origin_mm - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
        assert!((s.normal - Vec3::Z).length() < 1e-4, "normal drifted to {:?}", s.normal);
    }

    #[test]
    fn repeated_round_trips_do_not_drift() {
        let mut s = SliceSettings::axis_aligned("s", Vec3::ZERO, 1);
        s.normal = Vec3::new(0.3, 0.5, -0.8).normalize();
        let want = s.normal;
        for _ in 0..64 {
            let p = Placement::from_matrix(s.placement().matrix());
            s.set_placement(p);
        }
        assert!((s.normal - want).length() < 1e-3, "drifted from {want:?} to {:?}", s.normal);
    }

    #[test]
    fn a_mirrored_gizmo_matrix_is_refused_rather_than_panicking_the_voxeliser() {
        // `ad_geom::Scene::set_transform` asserts scale > 0, because a mirror
        // flips every winding and inverts the SDF. The gizmo can produce one by
        // dragging a scale handle through zero.
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(-2.0),
            Quat::IDENTITY,
            Vec3::ZERO,
        );
        let p = Placement::from_matrix(m);
        assert!(p.scale > 0.0, "scale came out {}", p.scale);
        assert!(p.to_geom().scale > 0.0);

        let zero = Mat4::from_scale_rotation_translation(Vec3::ZERO, Quat::IDENTITY, Vec3::ZERO);
        assert!(Placement::from_matrix(zero).scale > 0.0);
    }

    #[test]
    fn a_vent_blows_along_its_face_turned_by_its_aim() {
        let mut v = VentSettings::new("v", Placement::default(), 140.0, 15.0);
        assert_eq!(v.direction(), Vec3::Z);
        v.aim_deg = [45.0, 0.0];
        let d = v.direction();
        assert!((d.y - d.z).abs() < 1e-5 && d.x.abs() < 1e-6, "45 up: {d}");
        v.aim_deg = [0.0, 30.0];
        assert!(v.direction().x > 0.0, "sideways is toward local +X");
        v.placement.rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        v.aim_deg = [0.0, 0.0];
        assert!(v.direction().abs_diff_eq(Vec3::X, 1e-5), "yawed a quarter turn, it blows along +X");
        let c = v.corners();
        assert!((c[1] - c[0]).length() - 140.0 < 1e-3 && (c[3] - c[0]).length() - 15.0 < 1e-3);
        let mut moved = v.clone();
        moved.speed_scale = 2.0;
        assert!(v.same_cells(&moved), "speed is air, not cells");
        moved.width_mm += 1.0;
        assert!(!v.same_cells(&moved));
    }

    #[test]
    fn each_gizmo_operation_snaps_in_its_own_units() {
        let (mm, deg) = (Some(0.5), Some(15.0));
        assert_eq!(snap_for(GizmoMode::Translate, mm, deg), mm);
        assert_eq!(snap_for(GizmoMode::Rotate, mm, deg), deg);
        assert_eq!(snap_for(GizmoMode::Scale, mm, deg), None);
        assert_eq!(snap_for(GizmoMode::Rotate, mm, None), None, "no snap means none");
    }

    #[test]
    fn placement_converts_to_and_from_the_geometry_transform() {
        let p = Placement {
            translation_mm: Vec3::new(5.0, -2.0, 1.0),
            rotation: Quat::from_rotation_z(0.7),
            scale: 1.5,
        };
        let back = Placement::from_geom(p.to_geom());
        assert!((back.translation_mm - p.translation_mm).length() < 1e-6);
        assert!((back.scale - p.scale).abs() < 1e-6);
        assert!(back.rotation.abs_diff_eq(p.rotation, 1e-6));
    }

    #[test]
    fn signed_distance_has_the_sign_of_the_normal_side() {
        let s = SliceSettings::axis_aligned("s", Vec3::new(0.0, 10.0, 0.0), 1);
        assert!(s.signed_distance(Vec3::new(0.0, 15.0, 0.0)) > 0.0);
        assert!(s.signed_distance(Vec3::new(0.0, 5.0, 0.0)) < 0.0);
        assert!(s.signed_distance(Vec3::new(3.0, 10.0, -4.0)).abs() < 1e-6);
    }

    #[test]
    fn probe_ids_are_never_reused() {
        // Reusing one would splice two probes' histories into a single plot and
        // look like a physical transient.
        let mut p = Probes::new();
        let a = p.add(Vec3::ZERO);
        let b = p.add(Vec3::X);
        assert!(p.remove(a));
        let c = p.add(Vec3::Y);
        assert_ne!(c, a);
        assert_ne!(c, b);
        assert_eq!(p.len(), 2);
        assert!(!p.remove(a), "removing twice must report nothing happened");
    }

    #[test]
    fn adding_a_probe_selects_it_and_deselects_the_rest() {
        let mut p = Probes::new();
        p.add(Vec3::ZERO);
        let b = p.add(Vec3::X);
        assert_eq!(p.selected().map(|x| x.id), Some(b));
        assert_eq!(p.items().iter().filter(|x| x.selected).count(), 1);
        p.select(None);
        assert!(p.selected().is_none());
    }

    #[test]
    fn probe_history_is_bounded_and_clearable() {
        let mut p = Probes::new();
        p.history = 16;
        let id = p.add(Vec3::ZERO);
        for i in 0..100 {
            p.record(id, i as f64, i as f64 * 0.1, Vec3::X * i as f32, true);
        }
        let probe = p.get(id).unwrap();
        assert_eq!(probe.pressure.x.len(), 16);
        assert_eq!(probe.speed.x.len(), 16);
        assert!((probe.velocity_ms.x - 99.0).abs() < 1e-6);
        p.clear_history();
        assert!(p.get(id).unwrap().pressure.is_empty(), "a reset must drop probe history too");
        assert_eq!(p.len(), 1, "the probes themselves survive");
    }

    #[test]
    fn recording_marks_a_probe_that_landed_in_solid() {
        let mut p = Probes::new();
        let id = p.add(Vec3::ZERO);
        p.record(id, 0.0, 0.0, Vec3::ZERO, false);
        assert!(!p.get(id).unwrap().in_fluid, "a probe inside the wall reads nonsense");
    }

    #[test]
    fn defaults_are_sane_for_a_fresh_session() {
        assert!(ParticleSettings::default().enabled);
        assert!(!IsoSettings::default().enabled, "an empty isosurface looks like a bug");
        assert!(!StreamlineSettings::default().enabled);
        assert_eq!(GizmoMode::default(), GizmoMode::Translate);
        assert_eq!(GizmoSpace::default(), GizmoSpace::World);
    }
}

//! The set of meshes being simulated, and the bookkeeping that makes moving one
//! of them cheap.
//!
//! The user drags obstructions around with a gizmo while the simulation runs, so
//! a transform change has to be an O(1) update that also says *what changed*.
//! The scene therefore records a dirty region — the union of where a moved part
//! was and where it now is — so the voxeliser can rebuild only that box instead
//! of the whole 20-million-cell grid.

use crate::mesh::{PseudoNormals, TriMesh};
use ad_gpu::Bbox;
use glam::{Quat, Vec3};
use std::sync::Arc;

/// Translate, rotate, uniform-scale. Deliberately not a general matrix: uniform
/// scale keeps normals valid under plain rotation (no inverse-transpose) and
/// keeps the signed distance field a true distance field, merely rescaled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: 1.0,
    };

    pub fn from_translation(t: Vec3) -> Self {
        Self {
            translation: t,
            ..Self::IDENTITY
        }
    }

    #[inline]
    pub fn point(&self, p: Vec3) -> Vec3 {
        self.rotation * (p * self.scale) + self.translation
    }

    /// Directions ignore the translation, and uniform scale does not change
    /// them, so this is just the rotation.
    #[inline]
    pub fn direction(&self, d: Vec3) -> Vec3 {
        self.rotation * d
    }

    /// The AABB of a transformed AABB, by transforming all eight corners.
    /// Tighter than transforming the extents when there is a rotation.
    pub fn bbox(&self, b: Bbox) -> Bbox {
        if b.is_empty() {
            return b;
        }
        let mut out = Bbox::EMPTY;
        for i in 0..8 {
            let c = Vec3::new(
                if i & 1 == 0 { b.min.x } else { b.max.x },
                if i & 2 == 0 { b.min.y } else { b.max.y },
                if i & 4 == 0 { b.min.z } else { b.max.z },
            );
            out = out.union_point(self.point(c));
        }
        out
    }
}

/// What a mesh is for. The solver treats both as no-slip walls; the difference
/// is that the duct defines the flow path and its mouths, while obstructions are
/// things the user drops in to see what happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshRole {
    Duct,
    Obstruction,
}

/// A loaded mesh plus everything derived from it that does not depend on the
/// transform. Shared behind an `Arc` so the same STL can be instanced without
/// recomputing the pseudonormals, which is the expensive part.
pub struct MeshAsset {
    pub mesh: TriMesh,
    pub pseudo: PseudoNormals,
    pub local_bbox: Bbox,
}

impl MeshAsset {
    pub fn new(mesh: TriMesh) -> Arc<Self> {
        let pseudo = mesh.pseudo_normals();
        let local_bbox = mesh.bbox();
        Arc::new(Self {
            mesh,
            pseudo,
            local_bbox,
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.mesh.triangle_count()
    }
}

/// Cheap to clone: the mesh itself is behind an `Arc`, so a clone copies a name,
/// a transform and a pointer. The app relies on that to build a replacement
/// lattice beside the running one; see [`Scene`].
#[derive(Clone)]
pub struct MeshInstance {
    pub name: String,
    pub asset: Arc<MeshAsset>,
    pub transform: Transform,
    pub role: MeshRole,
    pub visible: bool,
}

impl MeshInstance {
    pub fn world_bbox(&self) -> Bbox {
        self.transform.bbox(self.asset.local_bbox)
    }

    /// A copy of the mesh with the transform baked in.
    ///
    /// Mouth detection and subdivision both work on an indexed mesh rather than
    /// a triangle soup — they need vertex identity to find boundary loops and to
    /// share midpoints — so they need this rather than [`FlatGeometry`].
    pub fn world_mesh(&self) -> TriMesh {
        let mut m = self.asset.mesh.clone();
        for p in m.positions.iter_mut() {
            *p = self.transform.point(*p);
        }
        if let Some(n) = &mut m.file_normals {
            for v in n.iter_mut() {
                *v = self.transform.direction(*v);
            }
        }
        m
    }
}

/// What changed since the voxeliser last looked.
#[derive(Debug, Clone)]
pub struct SceneDirty {
    /// The instances that need their triangles re-uploaded.
    pub instances: Vec<usize>,
    /// World-space box covering everything that moved, both from and to.
    pub region: Bbox,
    /// The set of triangles itself changed (an instance appeared, vanished or
    /// was hidden), so buffer offsets are no longer valid and everything must
    /// be rebuilt.
    pub structural: bool,
}

impl SceneDirty {
    fn clean() -> Self {
        Self {
            instances: Vec::new(),
            region: Bbox::EMPTY,
            structural: false,
        }
    }

    pub fn is_clean(&self) -> bool {
        !self.structural && self.instances.is_empty()
    }
}

/// `Clone` so the app can build a replacement lattice from a copy while the
/// running one keeps its own: a rebuild that fails must leave the old `Sim`
/// whole. Cheap, because every [`MeshInstance`] shares its mesh by `Arc`.
#[derive(Default, Clone)]
pub struct Scene {
    instances: Vec<MeshInstance>,
    dirty: Option<SceneDirty>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            dirty: None,
        }
    }

    pub fn add(
        &mut self,
        name: impl Into<String>,
        asset: Arc<MeshAsset>,
        transform: Transform,
        role: MeshRole,
    ) -> usize {
        let inst = MeshInstance {
            name: name.into(),
            asset,
            transform,
            role,
            visible: true,
        };
        let region = inst.world_bbox();
        self.instances.push(inst);
        self.mark(None, region, true);
        self.instances.len() - 1
    }

    pub fn remove(&mut self, index: usize) -> MeshInstance {
        let inst = self.instances.remove(index);
        let region = inst.world_bbox();
        self.mark(None, region, true);
        inst
    }

    pub fn clear(&mut self) {
        let region = self.bbox();
        self.instances.clear();
        self.mark(None, region, true);
    }

    pub fn instances(&self) -> &[MeshInstance] {
        &self.instances
    }

    pub fn instance(&self, i: usize) -> &MeshInstance {
        &self.instances[i]
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Instances that actually contribute geometry, in buffer order.
    pub fn visible(&self) -> impl Iterator<Item = (usize, &MeshInstance)> {
        self.instances.iter().enumerate().filter(|(_, i)| i.visible)
    }

    pub fn triangle_count(&self) -> usize {
        self.visible().map(|(_, i)| i.asset.triangle_count()).sum()
    }

    /// Move an instance. This is the interactive path: it must stay O(1) and it
    /// must record both the old and the new footprint, because the cells the
    /// part *left* need rebuilding just as much as the ones it now covers.
    pub fn set_transform(&mut self, index: usize, transform: Transform) {
        assert!(
            transform.scale > 0.0,
            "scale must be positive; a mirror flips every winding"
        );
        let before = self.instances[index].world_bbox();
        self.instances[index].transform = transform;
        let after = self.instances[index].world_bbox();
        self.mark(Some(index), before.union(after), false);
    }

    /// Hiding or showing an instance changes which triangles exist, so it is a
    /// structural change even though nothing moved.
    pub fn set_visible(&mut self, index: usize, visible: bool) {
        if self.instances[index].visible == visible {
            return;
        }
        self.instances[index].visible = visible;
        let region = self.instances[index].world_bbox();
        self.mark(None, region, true);
    }

    pub fn set_role(&mut self, index: usize, role: MeshRole) {
        self.instances[index].role = role;
    }

    fn mark(&mut self, instance: Option<usize>, region: Bbox, structural: bool) {
        let d = self.dirty.get_or_insert_with(SceneDirty::clean);
        d.structural |= structural;
        if !region.is_empty() {
            d.region = d.region.union(region);
        }
        if let Some(i) = instance {
            if !d.instances.contains(&i) {
                d.instances.push(i);
            }
        }
    }

    /// Union of every visible instance's world-space AABB.
    pub fn bbox(&self) -> Bbox {
        self.visible()
            .fold(Bbox::EMPTY, |b, (_, i)| b.union(i.world_bbox()))
    }

    /// Union of the visible instances' world boxes for one role.
    ///
    /// The duct's alone is what the mouths and the lattice are planned from:
    /// an obstruction must neither move the faces the mouths are found on nor
    /// grow the domain around itself, which for a part the size of a dashboard
    /// would ask for more cells than any card has.
    pub fn bbox_of_role(&self, role: MeshRole) -> Bbox {
        self.visible()
            .filter(|(_, i)| i.role == role)
            .fold(Bbox::EMPTY, |b, (_, i)| b.union(i.world_bbox()))
    }

    /// The simulation domain: the scene box grown so the exit jet has somewhere
    /// to go and the inlet has room for an upstream extension.
    ///
    /// The contract fixes 260 x 180 x 180 mm around the test part, which is
    /// roughly 1.8x its longest side; expressed as a relative margin so it
    /// scales with whatever the user loads.
    pub fn domain_bbox(&self, margin_frac: Vec3) -> Bbox {
        let b = self.bbox();
        if b.is_empty() {
            return b;
        }
        b.expanded(b.size().max_element() * margin_frac)
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty.is_some()
    }

    pub fn dirty(&self) -> Option<&SceneDirty> {
        self.dirty.as_ref()
    }

    /// Consume the dirty record. The caller is expected to have just rebuilt
    /// whatever it described.
    pub fn take_dirty(&mut self) -> Option<SceneDirty> {
        self.dirty.take()
    }

    /// Force the next voxelisation to rebuild everything.
    pub fn mark_all_dirty(&mut self) {
        let region = self.bbox();
        self.mark(None, region, true);
    }
}

/// The scene flattened to world space: one triangle soup plus one pseudonormal
/// record per triangle, in the exact order the GPU buffers use.
///
/// Flattening the *vertex* pseudonormals per triangle corner costs 36 bytes a
/// triangle over keeping them indexed, and removes an indirection from the
/// hottest shader loop. At 85k triangles that is 3 MB, which is nothing next to
/// a 20-million-cell grid.
#[derive(Debug, Clone, Default)]
pub struct FlatGeometry {
    pub tris: Vec<[Vec3; 3]>,
    /// `[face, edge01, edge12, edge20, vertex0, vertex1, vertex2]`, matching the
    /// `TriPn` struct and the feature codes in `shaders/geom/common.wgsl`.
    pub pn: Vec<[Vec3; 7]>,
    /// For each visible instance, its scene index and its triangle range.
    pub ranges: Vec<(usize, std::ops::Range<usize>)>,
}

impl FlatGeometry {
    pub fn from_scene(scene: &Scene) -> Self {
        let mut out = Self::default();
        for (idx, inst) in scene.visible() {
            let start = out.tris.len();
            out.append_instance(inst);
            out.ranges.push((idx, start..out.tris.len()));
        }
        out
    }

    pub fn from_mesh(mesh: &TriMesh) -> Self {
        let mut out = Self::default();
        let pn = mesh.pseudo_normals();
        out.append(mesh, &pn, Transform::IDENTITY);
        out.ranges.push((0, 0..out.tris.len()));
        out
    }

    fn append_instance(&mut self, inst: &MeshInstance) {
        self.append(&inst.asset.mesh, &inst.asset.pseudo, inst.transform);
    }

    fn append(&mut self, mesh: &TriMesh, pn: &PseudoNormals, xf: Transform) {
        self.tris.reserve(mesh.triangle_count());
        self.pn.reserve(mesh.triangle_count());
        for t in 0..mesh.triangle_count() {
            let [a, b, c] = mesh.triangle(t);
            self.tris.push([xf.point(a), xf.point(b), xf.point(c)]);
            let i = mesh.indices[t];
            self.pn.push([
                xf.direction(pn.face[t]),
                xf.direction(pn.edge[t][0]),
                xf.direction(pn.edge[t][1]),
                xf.direction(pn.edge[t][2]),
                xf.direction(pn.vertex[i[0] as usize]),
                xf.direction(pn.vertex[i[1] as usize]),
                xf.direction(pn.vertex[i[2] as usize]),
            ]);
        }
    }

    /// Rewrite one instance's triangles in place after it moved. The triangle
    /// *count* cannot change, so the buffer offsets stay valid and only this
    /// slice needs re-uploading.
    pub fn refresh_instance(
        &mut self,
        scene: &Scene,
        scene_index: usize,
    ) -> Option<std::ops::Range<usize>> {
        let (_, range) = self.ranges.iter().find(|(i, _)| *i == scene_index)?.clone();
        let inst = scene.instance(scene_index);
        let mesh = &inst.asset.mesh;
        let pn = &inst.asset.pseudo;
        let xf = inst.transform;
        for (k, t) in range.clone().enumerate() {
            let [a, b, c] = mesh.triangle(k);
            self.tris[t] = [xf.point(a), xf.point(b), xf.point(c)];
            let i = mesh.indices[k];
            self.pn[t] = [
                xf.direction(pn.face[k]),
                xf.direction(pn.edge[k][0]),
                xf.direction(pn.edge[k][1]),
                xf.direction(pn.edge[k][2]),
                xf.direction(pn.vertex[i[0] as usize]),
                xf.direction(pn.vertex[i[1] as usize]),
                xf.direction(pn.vertex[i[2] as usize]),
            ];
        }
        Some(range)
    }

    pub fn len(&self) -> usize {
        self.tris.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    pub fn bounds(&self) -> Vec<Bbox> {
        self.tris
            .iter()
            .map(|t| Bbox::from_points(t.iter().copied()))
            .collect()
    }

    pub fn bbox(&self) -> Bbox {
        self.tris
            .iter()
            .flatten()
            .copied()
            .fold(Bbox::EMPTY, |b, p| b.union_point(p))
    }

    /// The pseudonormal for a closest-point feature, matching
    /// [`crate::mesh::Feature::code`].
    #[inline]
    pub fn pseudonormal(&self, tri: usize, feature: crate::mesh::Feature) -> Vec3 {
        use crate::mesh::Feature;
        let p = &self.pn[tri];
        match feature {
            Feature::Face => p[0],
            Feature::Edge(k) => p[1 + k as usize],
            Feature::Vertex(k) => p[4 + k as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn asset() -> Arc<MeshAsset> {
        MeshAsset::new(primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0)))
    }

    #[test]
    fn scene_bbox_is_the_union_of_transformed_boxes() {
        let mut s = Scene::new();
        s.add("a", asset(), Transform::IDENTITY, MeshRole::Duct);
        s.add(
            "b",
            asset(),
            Transform::from_translation(Vec3::new(100.0, 0.0, 0.0)),
            MeshRole::Obstruction,
        );
        let b = s.bbox();
        assert_eq!(b.min, Vec3::ZERO);
        assert_eq!(b.max, Vec3::new(110.0, 10.0, 10.0));
    }

    #[test]
    fn a_role_box_leaves_the_other_role_out() {
        let mut s = Scene::new();
        s.add("duct", asset(), Transform::IDENTITY, MeshRole::Duct);
        s.add(
            "vane",
            asset(),
            Transform::from_translation(Vec3::new(100.0, 0.0, 0.0)),
            MeshRole::Obstruction,
        );
        assert_eq!(s.bbox_of_role(MeshRole::Duct).max, Vec3::splat(10.0));
        assert_eq!(
            s.bbox_of_role(MeshRole::Obstruction).min,
            Vec3::new(100.0, 0.0, 0.0)
        );
    }

    #[test]
    fn hiding_an_instance_removes_it_from_the_bbox() {
        let mut s = Scene::new();
        s.add("a", asset(), Transform::IDENTITY, MeshRole::Duct);
        let b = s.add(
            "b",
            asset(),
            Transform::from_translation(Vec3::new(100.0, 0.0, 0.0)),
            MeshRole::Obstruction,
        );
        s.set_visible(b, false);
        assert_eq!(s.bbox().max, Vec3::splat(10.0));
        assert_eq!(s.triangle_count(), 12);
    }

    #[test]
    fn rotation_gives_a_tight_box_not_an_extent_scaled_one() {
        // A 45 degree rotation about Z of a 10 mm cube: the footprint in XY
        // becomes 10*sqrt(2) wide, and Z is untouched.
        let t = Transform {
            translation: Vec3::ZERO,
            rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
            scale: 1.0,
        };
        let b = t.bbox(Bbox {
            min: Vec3::ZERO,
            max: Vec3::splat(10.0),
        });
        let size = b.size();
        assert!((size.x - 10.0 * 2.0f32.sqrt()).abs() < 1e-4, "{size:?}");
        assert!((size.z - 10.0).abs() < 1e-5, "{size:?}");
    }

    #[test]
    fn moving_an_instance_dirties_both_the_old_and_the_new_footprint() {
        let mut s = Scene::new();
        let a = s.add("a", asset(), Transform::IDENTITY, MeshRole::Obstruction);
        s.take_dirty();
        assert!(!s.is_dirty());

        s.set_transform(a, Transform::from_translation(Vec3::new(50.0, 0.0, 0.0)));
        let d = s.take_dirty().expect("moving must dirty the scene");
        assert!(!d.structural, "a move must not force a full rebuild");
        assert_eq!(d.instances, vec![a]);
        // The dirty box has to span from where it was to where it went.
        assert_eq!(d.region.min, Vec3::ZERO);
        assert_eq!(d.region.max, Vec3::new(60.0, 10.0, 10.0));
    }

    #[test]
    fn adding_or_hiding_is_structural_but_moving_is_not() {
        let mut s = Scene::new();
        let a = s.add("a", asset(), Transform::IDENTITY, MeshRole::Duct);
        assert!(s.take_dirty().unwrap().structural);

        s.set_transform(a, Transform::from_translation(Vec3::X));
        assert!(!s.take_dirty().unwrap().structural);

        s.set_visible(a, false);
        assert!(s.take_dirty().unwrap().structural);
    }

    #[test]
    fn a_transform_that_does_not_move_still_reports_the_instance() {
        // The voxeliser re-uploads whatever it is told changed; reporting a
        // no-op move is wasteful but never wrong, whereas missing a real one is
        // a stale mask. Assert the conservative behaviour explicitly.
        let mut s = Scene::new();
        let a = s.add("a", asset(), Transform::IDENTITY, MeshRole::Duct);
        s.take_dirty();
        s.set_transform(a, Transform::IDENTITY);
        assert_eq!(s.take_dirty().unwrap().instances, vec![a]);
    }

    #[test]
    fn transform_round_trips_a_point_through_its_inverse() {
        let t = Transform {
            translation: Vec3::new(1.0, -2.0, 3.0),
            rotation: Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.7, 1.1),
            scale: 2.5,
        };
        let p = Vec3::new(4.0, 5.0, -6.0);
        let q = t.point(p);
        let back = t.rotation.inverse() * (q - t.translation) / t.scale;
        assert!((back - p).length() < 1e-4, "{back:?} != {p:?}");
        // Uniform scale scales distances uniformly, which is why the SDF stays
        // a distance field.
        let p2 = Vec3::new(-1.0, 0.5, 2.0);
        let d_local = (p - p2).length();
        let d_world = (t.point(p) - t.point(p2)).length();
        assert!((d_world - d_local * t.scale).abs() < 1e-3);
    }
}

//! The layer stack: what is drawn, in what order, and what each layer is
//! attached to.
//!
//! A CFD viewport is a pile of semi-transparent things — a ghost duct, a
//! volume, particles, an isosurface, two slice planes — and the only way to
//! work out why the picture looks wrong is to turn them off one at a time. So
//! the layer list is not decoration: it is the debugging tool.
//!
//! Two decisions worth stating:
//!
//! * **Layers are identified by a stable [`LayerId`], not by index.** Slices and
//!   obstructions are added and removed at run time, and an index-keyed
//!   selection silently retargets when the list shifts — the classic "I deleted
//!   slice 1 and slice 2's settings changed" bug.
//! * **Visibility is a pair: `visible` and `enabled`.** A layer whose backing
//!   data does not exist yet (no particles seeded, no obstruction loaded) is
//!   *disabled*, drawn greyed with its reason, rather than hidden. A checkbox
//!   that silently does nothing is worse than one you cannot click.

/// Stable handle for a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(pub u32);

/// What a layer draws. The payload is the index into whatever list owns the
/// underlying object, which is why removal must go through
/// [`LayerStack::remove`] rather than a raw `Vec::remove`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// The duct shell.
    Duct,
    /// An additional STL dropped into the flow; the index is into the app's
    /// obstruction list.
    Obstruction(usize),
    /// A vent blowing air in the room; the index is into
    /// [`crate::state::UiState::vents`].
    Vent(usize),
    /// Advected tracer particles.
    Particles,
    /// The direct volume rendering of the current scalar.
    Volume,
    /// A soft isosurface of the current scalar.
    Isosurface,
    /// A cutting plane; the index is into [`crate::overlays::SliceSettings`].
    Slice(usize),
    /// Streamlines seeded from a rake.
    Streamlines,
    /// The wireframe of the lattice, for checking resolution against features.
    GridOutline,
    /// Probe markers and their labels.
    Probes,
}

impl LayerKind {
    /// Whether this layer can be moved with a gizmo, and therefore whether
    /// selecting it should arm the transform widget. The duct's gizmo sets its
    /// install pose in the car, not its place in the lattice; see
    /// [`crate::pose`].
    pub fn is_transformable(self) -> bool {
        matches!(
            self,
            LayerKind::Duct | LayerKind::Obstruction(_) | LayerKind::Vent(_) | LayerKind::Slice(_)
        )
    }
}

/// One row of the layer panel.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub id: LayerId,
    pub kind: LayerKind,
    pub name: String,
    pub visible: bool,
    /// False when the layer has nothing to draw. Shows the checkbox greyed with
    /// [`Layer::disabled_reason`] as a tooltip.
    pub enabled: bool,
    pub disabled_reason: String,
    /// 0-1. Applied by the overlay pass; the volume uses its transfer
    /// function's density instead, so this is ignored there.
    pub opacity: f32,
    /// True while the gizmo is attached to this layer.
    pub selected: bool,
}

impl Layer {
    pub fn new(id: LayerId, kind: LayerKind, name: impl Into<String>) -> Self {
        Self {
            id,
            kind,
            name: name.into(),
            visible: true,
            enabled: true,
            disabled_reason: String::new(),
            opacity: 1.0,
            selected: false,
        }
    }

    /// Whether the renderer should actually draw it.
    pub fn is_drawn(&self) -> bool {
        self.visible && self.enabled && self.opacity > 0.0
    }

    pub fn disable(&mut self, reason: impl Into<String>) {
        self.enabled = false;
        self.disabled_reason = reason.into();
    }

    pub fn enable(&mut self) {
        self.enabled = true;
        self.disabled_reason.clear();
    }
}

/// The ordered stack, drawn front of the list first in the panel.
#[derive(Debug, Clone, Default)]
pub struct LayerStack {
    layers: Vec<Layer>,
    next_id: u32,
}

impl LayerStack {
    /// The stack the app opens with. Matches the mock-up in the brief: the
    /// isosurface is present but off, because on a fresh run there is nothing
    /// to make an isosurface of and an empty one looks like a broken renderer.
    pub fn defaults() -> Self {
        let mut s = Self::default();
        s.push(LayerKind::Duct, "Duct");
        s.push(LayerKind::Particles, "Particles");
        s.push(LayerKind::Volume, "Volume");
        let iso = s.push(LayerKind::Isosurface, "Q-isosurface");
        s.set_visible(iso, false);
        s.push(LayerKind::Streamlines, "Streamlines");
        s.set_visible(s.find(LayerKind::Streamlines).unwrap(), false);
        s.push(LayerKind::Probes, "Probes");
        let grid = s.push(LayerKind::GridOutline, "Grid outline");
        s.set_visible(grid, false);
        s
    }

    pub fn push(&mut self, kind: LayerKind, name: impl Into<String>) -> LayerId {
        let id = LayerId(self.next_id);
        self.next_id += 1;
        self.layers.push(Layer::new(id, kind, name));
        id
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [Layer] {
        &mut self.layers
    }

    pub fn get(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn get_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    /// First layer of a given kind. Handy for the singleton layers (Volume,
    /// Particles); ambiguous for the indexed ones, which is why the index is
    /// part of [`LayerKind`].
    pub fn find(&self, kind: LayerKind) -> Option<LayerId> {
        self.layers.iter().find(|l| l.kind == kind).map(|l| l.id)
    }

    pub fn is_drawn(&self, kind: LayerKind) -> bool {
        self.layers.iter().any(|l| l.kind == kind && l.is_drawn())
    }

    pub fn set_visible(&mut self, id: LayerId, visible: bool) {
        if let Some(l) = self.get_mut(id) {
            l.visible = visible;
        }
    }

    pub fn toggle(&mut self, id: LayerId) {
        if let Some(l) = self.get_mut(id) {
            l.visible = !l.visible;
        }
    }

    /// Show only `id`, hiding everything else. The "solo" gesture: the fastest
    /// way to answer "is that artefact coming from the volume or the mesh?".
    pub fn solo(&mut self, id: LayerId) {
        for l in &mut self.layers {
            l.visible = l.id == id;
        }
    }

    pub fn show_all(&mut self) {
        for l in &mut self.layers {
            l.visible = true;
        }
    }

    /// Select exactly one layer, or nothing when `id` is `None`. Selection is
    /// single because the gizmo is single; multi-select would need a group
    /// transform and there is nothing asking for one.
    pub fn select(&mut self, id: Option<LayerId>) {
        for l in &mut self.layers {
            l.selected = Some(l.id) == id;
        }
    }

    pub fn selected(&self) -> Option<&Layer> {
        self.layers.iter().find(|l| l.selected)
    }

    /// Remove a layer, keeping the payload indices of its siblings correct.
    ///
    /// This is the reason removal is a method rather than a `Vec::remove` at
    /// the call site: every `Obstruction(i)` or `Slice(i)` above the removed
    /// one has to shift down, and forgetting that leaves gizmos attached to the
    /// wrong object — which reads as "the gizmo moved the wrong slice", a bug
    /// that is very hard to see and very easy to create.
    /// Put an obstruction layer back at index `i`, shifting the ones at or
    /// above it up: the inverse of [`Self::remove`], for when the app could not
    /// take the obstruction out after all.
    pub fn insert_obstruction(&mut self, i: usize, name: impl Into<String>) -> LayerId {
        for l in &mut self.layers {
            if let LayerKind::Obstruction(j) = &mut l.kind {
                if *j >= i {
                    *j += 1;
                }
            }
        }
        self.push(LayerKind::Obstruction(i), name)
    }

    pub fn remove(&mut self, id: LayerId) -> Option<Layer> {
        let pos = self.layers.iter().position(|l| l.id == id)?;
        let removed = self.layers.remove(pos);
        match removed.kind {
            LayerKind::Obstruction(i) => self.shift_obstructions(i),
            LayerKind::Vent(i) => self.shift_vents(i),
            LayerKind::Slice(i) => self.shift_slices(i),
            _ => {}
        }
        Some(removed)
    }

    /// Drop every vent layer and put back one per name, in order: for when the
    /// app has to roll the vent list back to what the lattice was built with.
    pub fn reset_vents<'a>(&mut self, names: impl IntoIterator<Item = &'a str>) {
        self.layers.retain(|l| !matches!(l.kind, LayerKind::Vent(_)));
        for (i, name) in names.into_iter().enumerate() {
            self.push(LayerKind::Vent(i), name);
        }
    }

    fn shift_vents(&mut self, above: usize) {
        for l in &mut self.layers {
            if let LayerKind::Vent(i) = &mut l.kind {
                if *i > above {
                    *i -= 1;
                }
            }
        }
    }

    fn shift_obstructions(&mut self, above: usize) {
        for l in &mut self.layers {
            if let LayerKind::Obstruction(i) = &mut l.kind {
                if *i > above {
                    *i -= 1;
                }
            }
        }
    }

    fn shift_slices(&mut self, above: usize) {
        for l in &mut self.layers {
            if let LayerKind::Slice(i) = &mut l.kind {
                if *i > above {
                    *i -= 1;
                }
            }
        }
    }

    /// Move a layer one position toward the front of the list.
    pub fn move_up(&mut self, id: LayerId) {
        if let Some(p) = self.layers.iter().position(|l| l.id == id) {
            if p > 0 {
                self.layers.swap(p, p - 1);
            }
        }
    }

    pub fn move_down(&mut self, id: LayerId) {
        if let Some(p) = self.layers.iter().position(|l| l.id == id) {
            if p + 1 < self.layers.len() {
                self.layers.swap(p, p + 1);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_stack_matches_the_mock_up() {
        let s = LayerStack::defaults();
        assert!(s.is_drawn(LayerKind::Duct));
        assert!(s.is_drawn(LayerKind::Volume));
        assert!(s.is_drawn(LayerKind::Particles));
        assert!(!s.is_drawn(LayerKind::Isosurface), "an empty isosurface looks like a bug");
        assert!(!s.is_drawn(LayerKind::GridOutline));
    }

    #[test]
    fn a_disabled_layer_is_not_drawn_even_when_visible() {
        // The distinction that stops a checkbox from silently doing nothing.
        let mut s = LayerStack::defaults();
        let id = s.find(LayerKind::Particles).unwrap();
        s.get_mut(id).unwrap().disable("no particles seeded");
        assert!(s.get(id).unwrap().visible);
        assert!(!s.is_drawn(LayerKind::Particles));
        assert_eq!(s.get(id).unwrap().disabled_reason, "no particles seeded");
        s.get_mut(id).unwrap().enable();
        assert!(s.is_drawn(LayerKind::Particles));
        assert!(s.get(id).unwrap().disabled_reason.is_empty());
    }

    #[test]
    fn zero_opacity_counts_as_not_drawn() {
        let mut s = LayerStack::defaults();
        let id = s.find(LayerKind::Volume).unwrap();
        s.get_mut(id).unwrap().opacity = 0.0;
        assert!(!s.is_drawn(LayerKind::Volume));
    }

    #[test]
    fn solo_leaves_exactly_one_layer_visible() {
        let mut s = LayerStack::defaults();
        let id = s.find(LayerKind::Volume).unwrap();
        s.solo(id);
        assert_eq!(s.layers().iter().filter(|l| l.visible).count(), 1);
        assert!(s.get(id).unwrap().visible);
        s.show_all();
        assert!(s.layers().iter().all(|l| l.visible));
    }

    #[test]
    fn removing_a_slice_renumbers_the_ones_above_it() {
        // The bug this prevents: delete slice 0 and the gizmo silently starts
        // moving a different plane.
        let mut s = LayerStack::default();
        let a = s.push(LayerKind::Slice(0), "Slice 1");
        let b = s.push(LayerKind::Slice(1), "Slice 2");
        let c = s.push(LayerKind::Slice(2), "Slice 3");
        s.remove(a);
        assert_eq!(s.get(b).unwrap().kind, LayerKind::Slice(0));
        assert_eq!(s.get(c).unwrap().kind, LayerKind::Slice(1));
        // Ids are stable across the renumbering, which is the point of having
        // them separate from the payload index.
        assert!(s.get(b).is_some() && s.get(c).is_some());
    }

    #[test]
    fn removing_an_obstruction_does_not_disturb_slice_indices() {
        let mut s = LayerStack::default();
        let o0 = s.push(LayerKind::Obstruction(0), "Vane A");
        let o1 = s.push(LayerKind::Obstruction(1), "Vane B");
        let sl = s.push(LayerKind::Slice(1), "Slice 2");
        s.remove(o0);
        assert_eq!(s.get(o1).unwrap().kind, LayerKind::Obstruction(0));
        assert_eq!(s.get(sl).unwrap().kind, LayerKind::Slice(1), "slices must be untouched");
    }

    #[test]
    fn vent_layers_shift_and_reset_like_the_others() {
        let mut s = LayerStack::default();
        let a = s.push(LayerKind::Vent(0), "Vent 1");
        let b = s.push(LayerKind::Vent(1), "Vent 2");
        let o = s.push(LayerKind::Obstruction(0), "Vane");
        s.remove(a);
        assert_eq!(s.get(b).unwrap().kind, LayerKind::Vent(0));
        assert_eq!(s.get(o).unwrap().kind, LayerKind::Obstruction(0), "obstructions untouched");
        s.reset_vents(["x", "y", "z"]);
        let vents: Vec<LayerKind> =
            s.layers().iter().filter(|l| matches!(l.kind, LayerKind::Vent(_))).map(|l| l.kind).collect();
        assert_eq!(vents, vec![LayerKind::Vent(0), LayerKind::Vent(1), LayerKind::Vent(2)]);
        assert!(LayerKind::Vent(0).is_transformable());
    }

    #[test]
    fn an_obstruction_layer_put_back_restores_the_indices() {
        let mut s = LayerStack::default();
        let a = s.push(LayerKind::Obstruction(0), "Vane A");
        let b = s.push(LayerKind::Obstruction(1), "Vane B");
        s.remove(a);
        assert_eq!(s.get(b).unwrap().kind, LayerKind::Obstruction(0));
        let a = s.insert_obstruction(0, "Vane A");
        assert_eq!(s.get(a).unwrap().kind, LayerKind::Obstruction(0));
        assert_eq!(s.get(b).unwrap().kind, LayerKind::Obstruction(1));
    }

    #[test]
    fn selection_is_single_and_clearable() {
        let mut s = LayerStack::defaults();
        let a = s.find(LayerKind::Duct).unwrap();
        let b = s.find(LayerKind::Volume).unwrap();
        s.select(Some(a));
        s.select(Some(b));
        assert_eq!(s.layers().iter().filter(|l| l.selected).count(), 1);
        assert_eq!(s.selected().map(|l| l.id), Some(b));
        s.select(None);
        assert!(s.selected().is_none());
    }

    #[test]
    fn only_placeable_layers_arm_the_gizmo() {
        assert!(LayerKind::Obstruction(0).is_transformable());
        assert!(LayerKind::Slice(2).is_transformable());
        assert!(!LayerKind::Volume.is_transformable());
        assert!(LayerKind::Duct.is_transformable(), "the duct's gizmo sets its install pose");
    }

    #[test]
    fn reordering_stays_inside_the_list() {
        let mut s = LayerStack::default();
        let a = s.push(LayerKind::Duct, "a");
        let b = s.push(LayerKind::Volume, "b");
        s.move_up(a);
        assert_eq!(s.layers()[0].id, a, "moving the first item up must be a no-op");
        s.move_down(b);
        assert_eq!(s.layers()[1].id, b, "moving the last item down must be a no-op");
        s.move_down(a);
        assert_eq!(s.layers()[0].id, b);
    }
}

//! Mapping mouse and keyboard input onto camera moves.
//!
//! `ad_render::OrbitController` already does the maths — smoothing, gain,
//! clamping. What is missing, and what belongs in the UI layer, is the
//! *decision*: given the buttons currently held and the modifiers, which of
//! orbit, pan or dolly is this drag?
//!
//! It is a small function and it is worth isolating for two reasons. First, it
//! is the part users notice instantly when it is wrong, and the part that has
//! no automated feedback otherwise — a controller mapped backwards still
//! compiles, still renders, and simply feels bad. Second, it has to interact
//! with two other systems that also want the mouse:
//!
//! * **ImGui.** When `io.want_capture_mouse` is set, the pointer belongs to a
//!   panel and the camera must ignore it entirely. Getting this wrong means the
//!   view spins while you drag a slider.
//! * **ImGuizmo.** While a gizmo is hovered or in use, the drag belongs to the
//!   gizmo. Same failure, harder to notice, because it only happens when a
//!   gizmo is on screen.
//!
//! Both are inputs to [`CameraInput::gesture`] rather than checks scattered
//! through the event handler, so the precedence is stated once and tested.

use ad_render::{OrbitController, ViewPreset};

/// What a drag is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gesture {
    None,
    Orbit,
    Pan,
    /// Dolly driven by vertical drag rather than the wheel, for tablets and
    /// for people who learned it in CAD.
    DollyDrag,
}

/// Buttons and modifiers, as the event loop sees them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MouseState {
    pub left: bool,
    pub middle: bool,
    pub right: bool,
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

/// Who owns the pointer this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PointerOwnership {
    /// ImGui wants the mouse: it is over a panel or dragging a widget.
    pub ui_capture: bool,
    /// A transform gizmo is hovered or being dragged.
    pub gizmo_active: bool,
}

impl PointerOwnership {
    pub fn viewport_has_mouse(self) -> bool {
        !self.ui_capture && !self.gizmo_active
    }
}

/// Camera control tuning and the mapping itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraInput {
    /// Wheel notches per dolly step. Positive means "wheel forward zooms in",
    /// which is the near-universal convention; inverting it is the first thing
    /// anyone complains about.
    pub wheel_gain: f32,
    /// Pixels of vertical drag per wheel notch, for [`Gesture::DollyDrag`].
    pub dolly_drag_pixels: f32,
    /// Invert vertical orbit, for people who expect the CAD convention.
    pub invert_orbit_y: bool,
}

impl Default for CameraInput {
    fn default() -> Self {
        Self { wheel_gain: 1.0, dolly_drag_pixels: 60.0, invert_orbit_y: false }
    }
}

impl CameraInput {
    /// Decide what a drag means.
    ///
    /// The precedence is: nobody else wants the mouse, then middle (or
    /// shift+left) pans, right or alt+left dollies, plain left orbits. Middle
    /// before left matters because a mouse can report both while a chorded
    /// drag is in progress, and pan is the one the user meant.
    pub fn gesture(&self, mouse: MouseState, owner: PointerOwnership) -> Gesture {
        if !owner.viewport_has_mouse() {
            return Gesture::None;
        }
        if mouse.middle || (mouse.left && mouse.shift) {
            Gesture::Pan
        } else if mouse.right || (mouse.left && mouse.alt) {
            Gesture::DollyDrag
        } else if mouse.left {
            Gesture::Orbit
        } else {
            Gesture::None
        }
    }

    /// Apply a mouse delta in pixels to the controller.
    pub fn apply_drag(
        &self,
        controller: &mut OrbitController,
        gesture: Gesture,
        dx: f32,
        dy: f32,
        viewport_height: f32,
    ) {
        match gesture {
            Gesture::None => {}
            Gesture::Orbit => {
                let sy = if self.invert_orbit_y { -1.0 } else { 1.0 };
                controller.orbit(dx, dy * sy);
            }
            Gesture::Pan => controller.pan(dx, dy, viewport_height),
            Gesture::DollyDrag => {
                controller.dolly(-dy / self.dolly_drag_pixels.max(1.0));
            }
        }
    }

    /// Apply a wheel event. Ignored when the pointer is not the viewport's,
    /// which is what stops the wheel from zooming the scene while it scrolls a
    /// list.
    pub fn apply_wheel(
        &self,
        controller: &mut OrbitController,
        notches: f32,
        owner: PointerOwnership,
    ) -> bool {
        if !owner.viewport_has_mouse() || notches == 0.0 {
            return false;
        }
        controller.dolly(notches * self.wheel_gain);
        true
    }
}

/// Keyboard shortcut for a view preset.
///
/// Numeric keys, in the order of `ViewPreset::ALL`, plus `f` to frame. Deliberately
/// not the CAD numpad convention: this app is used on laptops as much as on
/// workstations, and a shortcut on a key half the machines do not have is not a
/// shortcut.
pub fn preset_for_digit(digit: u32) -> Option<ViewPreset> {
    ViewPreset::ALL.get(digit.checked_sub(1)? as usize).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::Bbox;
    use glam::Vec3;

    fn owner() -> PointerOwnership {
        PointerOwnership::default()
    }

    #[test]
    fn the_ui_and_the_gizmo_take_precedence_over_the_camera() {
        // The bug this prevents: the view spins while you drag a slider.
        let ci = CameraInput::default();
        let left = MouseState { left: true, ..Default::default() };
        assert_eq!(ci.gesture(left, owner()), Gesture::Orbit);
        assert_eq!(
            ci.gesture(left, PointerOwnership { ui_capture: true, gizmo_active: false }),
            Gesture::None
        );
        assert_eq!(
            ci.gesture(left, PointerOwnership { ui_capture: false, gizmo_active: true }),
            Gesture::None
        );
    }

    #[test]
    fn button_and_modifier_combinations_map_as_documented() {
        let ci = CameraInput::default();
        let m = |f: fn(&mut MouseState)| {
            let mut s = MouseState::default();
            f(&mut s);
            ci.gesture(s, owner())
        };
        assert_eq!(m(|s| s.left = true), Gesture::Orbit);
        assert_eq!(m(|s| s.middle = true), Gesture::Pan);
        assert_eq!(m(|s| s.right = true), Gesture::DollyDrag);
        assert_eq!(m(|s| { s.left = true; s.shift = true; }), Gesture::Pan);
        assert_eq!(m(|s| { s.left = true; s.alt = true; }), Gesture::DollyDrag);
        assert_eq!(m(|_| {}), Gesture::None);
        // Middle wins over a simultaneously-reported left during a chord.
        assert_eq!(m(|s| { s.left = true; s.middle = true; }), Gesture::Pan);
    }

    #[test]
    fn orbit_and_pan_move_the_camera_in_the_expected_direction() {
        let ci = CameraInput::default();
        let mut c = OrbitController::default();
        let yaw0 = c.goal.yaw;
        ci.apply_drag(&mut c, Gesture::Orbit, 50.0, 0.0, 800.0);
        assert!(c.goal.yaw < yaw0, "dragging right must swing the model to the right");

        let mut c = OrbitController::default();
        let target0 = c.goal.target;
        ci.apply_drag(&mut c, Gesture::Pan, 40.0, 0.0, 800.0);
        assert_ne!(c.goal.target, target0);
        // Panning does not change the orbit angles or the distance.
        assert_eq!(c.goal.yaw, OrbitController::default().goal.yaw);
        assert_eq!(c.goal.distance, OrbitController::default().goal.distance);
    }

    #[test]
    fn inverting_the_vertical_axis_actually_inverts_it() {
        let mut c = OrbitController::default();
        let mut c2 = c.clone();
        CameraInput::default().apply_drag(&mut c, Gesture::Orbit, 0.0, 30.0, 800.0);
        CameraInput { invert_orbit_y: true, ..Default::default() }
            .apply_drag(&mut c2, Gesture::Orbit, 0.0, 30.0, 800.0);
        let base = OrbitController::default().goal.pitch;
        assert!((c.goal.pitch - base).signum() != (c2.goal.pitch - base).signum());
    }

    #[test]
    fn the_wheel_zooms_in_when_pushed_forward_and_is_multiplicative() {
        let ci = CameraInput::default();
        let mut c = OrbitController::default();
        let d0 = c.goal.distance;
        assert!(ci.apply_wheel(&mut c, 1.0, owner()));
        assert!(c.goal.distance < d0, "wheel forward must zoom in");

        // Each notch covers the same fraction, so two notches from 400 mm and
        // two from 40 mm feel identical. This is what makes inspecting a 6 mm
        // passage possible at all.
        let ratio_far = c.goal.distance / d0;
        let mut near = OrbitController::default();
        near.goal.distance = 40.0;
        let n0 = near.goal.distance;
        ci.apply_wheel(&mut near, 1.0, owner());
        assert!((near.goal.distance / n0 - ratio_far).abs() < 1e-5);
    }

    #[test]
    fn the_wheel_is_ignored_when_a_panel_owns_the_pointer() {
        let ci = CameraInput::default();
        let mut c = OrbitController::default();
        let d0 = c.goal.distance;
        assert!(!ci.apply_wheel(&mut c, 3.0, PointerOwnership { ui_capture: true, gizmo_active: false }));
        assert_eq!(c.goal.distance, d0);
    }

    #[test]
    fn dolly_drag_pulls_the_camera_in_when_dragged_up() {
        let ci = CameraInput::default();
        let mut c = OrbitController::default();
        let d0 = c.goal.distance;
        // Screen y grows downward, so a negative dy is "drag up".
        ci.apply_drag(&mut c, Gesture::DollyDrag, 0.0, -60.0, 800.0);
        assert!(c.goal.distance < d0, "drag up should zoom in, got {}", c.goal.distance);
    }

    #[test]
    fn digit_shortcuts_cover_every_preset_and_reject_the_rest() {
        for (i, preset) in ViewPreset::ALL.iter().enumerate() {
            assert_eq!(preset_for_digit(i as u32 + 1), Some(*preset));
        }
        assert_eq!(preset_for_digit(0), None, "there is no view 0");
        assert_eq!(preset_for_digit(9), None);
    }

    #[test]
    fn framing_through_the_controller_respects_the_distance_clamps() {
        // A tiny part must not put the camera inside its own near plane.
        let mut c = OrbitController::default();
        c.frame_bbox(Bbox { min: Vec3::splat(-0.05), max: Vec3::splat(0.05) }, 0.1);
        assert!(c.goal.distance >= c.distance_range.0);
        c.frame_bbox(Bbox { min: Vec3::splat(-1.0e5), max: Vec3::splat(1.0e5) }, 0.1);
        assert!(c.goal.distance <= c.distance_range.1);
    }
}

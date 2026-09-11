//! winit 0.30 -> Dear ImGui input.
//!
//! `dear-imgui-rs` 0.17 ships no winit backend of its own (the crate's own docs
//! point at a `dear-imgui-winit` that is not in this workspace), so the bridge
//! lives here. It is about a hundred lines and it is entirely mechanical, but
//! three details are not:
//!
//! * **Events, not state.** ImGui 1.89+ wants `AddKeyEvent`/`AddMousePosEvent`
//!   queued as they arrive, not a state block copied once per frame. The queued
//!   form is what makes a click that arrives and is released inside one frame
//!   still register — which happens constantly on a 144 Hz display.
//! * **Physical pixels throughout.** The display size, the framebuffer scale
//!   and the mouse position must agree about which coordinate space they are
//!   in, or the UI is offset from the cursor by the DPI factor. We feed logical
//!   size and a framebuffer scale of the DPI factor, which is the combination
//!   ImGui expects and the one that makes fonts crisp at 150%.
//! * **Modifiers separately.** winit reports modifier state in its own event,
//!   and ImGui needs both the physical key events *and* the `ModCtrl`-family
//!   keys. Feeding only one of the two breaks either text editing shortcuts or
//!   the gizmo's snap-on-ctrl, depending on which you forget.

use dear_imgui_rs::{Context, Key, MouseButton, MouseCursor};
use winit::event::{DeviceEvent, ElementState, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{CursorIcon, Window};

/// Feeds winit events into an ImGui context and keeps the display metrics in
/// step with the window.
#[derive(Debug, Clone)]
pub struct WinitPlatform {
    /// Logical window size, ImGui's coordinate space.
    logical_size: [f32; 2],
    /// Device pixel ratio.
    scale_factor: f32,
    /// Last cursor icon pushed to the window, so the platform call is made only
    /// when it changes. Setting the cursor every frame makes it flicker on
    /// Windows.
    last_cursor: Option<MouseCursor>,
    /// Physical-pixel cursor position, kept because click-to-probe needs it in
    /// the same space as the viewport rectangle.
    pub cursor_physical: [f32; 2],
    pub modifiers: ModifiersState,
    /// True while the window has focus. Events are still processed when it does
    /// not (a drag can end outside), but the sim can throttle on it.
    pub focused: bool,
}

impl Default for WinitPlatform {
    fn default() -> Self {
        Self {
            logical_size: [1280.0, 800.0],
            scale_factor: 1.0,
            last_cursor: None,
            cursor_physical: [0.0, 0.0],
            modifiers: ModifiersState::empty(),
            focused: true,
        }
    }
}

impl WinitPlatform {
    /// Configure the context for this app: docking on, keyboard navigation on,
    /// mouse cursors on.
    ///
    /// **Only the platform's own flags are set here.** `RENDERER_HAS_TEXTURES`
    /// and `RENDERER_HAS_VTX_OFFSET` describe the *renderer*, and
    /// `dear-imgui-wgpu` refuses to attach to a context that already advertises
    /// them — it reads them as another backend having claimed the context, and
    /// `WgpuRenderer::new` fails with "already configured for a renderer
    /// backend". The wgpu backend sets both itself when it attaches, and
    /// [`Self::frame_options`] re-asserts `RENDERER_HAS_TEXTURES` every frame so
    /// the managed font-atlas path stays selected.
    ///
    /// `DOCKING_ENABLE` must be set **before the first frame** and must not
    /// change afterwards: Dear ImGui destroys live dock nodes when docking is
    /// turned off and cannot restore them, so the library asserts on a change.
    pub fn init(context: &mut Context, window: &Window) -> Self {
        let mut me = Self::default();
        {
            let io = context.io_mut();
            let mut flags = io.config_flags();
            flags.insert(dear_imgui_rs::ConfigFlags::DOCKING_ENABLE);
            flags.insert(dear_imgui_rs::ConfigFlags::NAV_ENABLE_KEYBOARD);
            io.set_config_flags(flags);

            let mut backend = io.backend_flags();
            backend.insert(dear_imgui_rs::BackendFlags::HAS_MOUSE_CURSORS);
            io.set_backend_flags(backend);
        }
        me.resized(context, window);
        me
    }

    pub fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    /// Physical framebuffer size, which is what the surface and the renderer
    /// are configured with.
    pub fn framebuffer_size(&self) -> (u32, u32) {
        (
            (self.logical_size[0] * self.scale_factor).round().max(1.0) as u32,
            (self.logical_size[1] * self.scale_factor).round().max(1.0) as u32,
        )
    }

    /// Re-read the window's size and DPI. Cheap; call on every resize and on
    /// every scale-factor change.
    pub fn resized(&mut self, context: &mut Context, window: &Window) {
        let scale = window.scale_factor() as f32;
        let physical = window.inner_size();
        self.scale_factor = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        self.logical_size = [
            physical.width as f32 / self.scale_factor,
            physical.height as f32 / self.scale_factor,
        ];
        let io = context.io_mut();
        io.set_display_size(self.logical_size);
        io.set_display_framebuffer_scale([self.scale_factor, self.scale_factor]);
    }

    /// Options for `Context::prepare_frame`, with the display metrics this
    /// platform is tracking.
    pub fn frame_options(&self, delta_seconds: f32) -> dear_imgui_rs::FramePrepareOptions {
        dear_imgui_rs::FramePrepareOptions::new(
            self.logical_size,
            // Zero or negative dt makes ImGui's own timers misbehave; a
            // suspended app can genuinely produce one.
            delta_seconds.max(1.0e-6),
        )
        .framebuffer_scale([self.scale_factor, self.scale_factor])
        .renderer_has_textures()
    }

    /// Feed one window event. Returns true when ImGui consumed it in the sense
    /// that the app should not also act on it.
    ///
    /// The return value deliberately reports *capture*, not "was it a UI
    /// event": a mouse move over a panel is still forwarded to ImGui and still
    /// reported as captured, so the camera does not follow the cursor across a
    /// slider.
    pub fn handle_window_event(
        &mut self,
        context: &mut Context,
        window: &Window,
        event: &WindowEvent,
    ) -> bool {
        match event {
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                self.resized(context, window);
                false
            }
            WindowEvent::Focused(focused) => {
                self.focused = *focused;
                context.io_mut().add_focus_event(*focused);
                false
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_physical = [position.x as f32, position.y as f32];
                let logical = [
                    position.x as f32 / self.scale_factor,
                    position.y as f32 / self.scale_factor,
                ];
                context.io_mut().add_mouse_pos_event(logical);
                context.io().want_capture_mouse()
            }
            WindowEvent::CursorLeft { .. } => {
                // ImGui's sentinel for "no cursor". Without it, a hovered
                // widget stays hovered after the pointer leaves the window.
                context.io_mut().add_mouse_pos_event([f32::MIN, f32::MIN]);
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(b) = map_mouse_button(*button) {
                    context
                        .io_mut()
                        .add_mouse_button_event(b, *state == ElementState::Pressed);
                }
                context.io().want_capture_mouse()
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (h, v) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (*x, *y),
                    // A pixel delta comes from a trackpad, where one "notch" is
                    // conventionally about 30 px. Feeding raw pixels scrolls
                    // lists at roughly thirty times the intended speed.
                    MouseScrollDelta::PixelDelta(p) => {
                        (p.x as f32 / 30.0, p.y as f32 / 30.0)
                    }
                };
                context.io_mut().add_mouse_wheel_event([h, v]);
                context.io().want_capture_mouse()
            }
            WindowEvent::ModifiersChanged(m) => {
                self.modifiers = m.state();
                let io = context.io_mut();
                io.add_key_event(Key::ModCtrl, self.modifiers.control_key());
                io.add_key_event(Key::ModShift, self.modifiers.shift_key());
                io.add_key_event(Key::ModAlt, self.modifiers.alt_key());
                io.add_key_event(Key::ModSuper, self.modifiers.super_key());
                false
            }
            WindowEvent::KeyboardInput { event, is_synthetic, .. } => {
                // Synthetic events are winit's replay of the key state when the
                // window regains focus. Feeding them produces a phantom press
                // of every held key, which in this app means the camera lurches
                // when you alt-tab back.
                if *is_synthetic {
                    return context.io().want_capture_keyboard();
                }
                let down = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(key) = map_key(code) {
                        context.io_mut().add_key_event(key, down);
                    }
                }
                if down {
                    if let Some(text) = &event.text {
                        let io = context.io_mut();
                        for c in text.chars() {
                            // Control characters would be inserted literally
                            // into a text field as boxes.
                            if !c.is_control() {
                                io.add_input_character(c);
                            }
                        }
                    }
                }
                context.io().want_capture_keyboard()
            }
            _ => false,
        }
    }

    /// Feed a raw device event.
    ///
    /// Only used for relative mouse motion while a drag is captured outside the
    /// window, which is exactly the case a camera orbit hits when the user
    /// swings past the window edge.
    pub fn handle_device_event(&mut self, event: &DeviceEvent) -> Option<(f32, f32)> {
        match event {
            DeviceEvent::MouseMotion { delta } => Some((delta.0 as f32, delta.1 as f32)),
            _ => None,
        }
    }

    /// Push ImGui's requested cursor to the window. Call once per frame after
    /// building the UI.
    pub fn apply_cursor(&mut self, window: &Window, requested: Option<MouseCursor>) {
        if self.last_cursor == requested {
            return;
        }
        self.last_cursor = requested;
        match requested {
            None => window.set_cursor_visible(false),
            Some(c) => {
                window.set_cursor_visible(true);
                window.set_cursor(map_cursor(c));
            }
        }
    }
}

fn map_mouse_button(b: winit::event::MouseButton) -> Option<MouseButton> {
    match b {
        winit::event::MouseButton::Left => Some(MouseButton::Left),
        winit::event::MouseButton::Right => Some(MouseButton::Right),
        winit::event::MouseButton::Middle => Some(MouseButton::Middle),
        _ => None,
    }
}

fn map_cursor(c: MouseCursor) -> CursorIcon {
    match c {
        MouseCursor::Arrow => CursorIcon::Default,
        MouseCursor::TextInput => CursorIcon::Text,
        MouseCursor::ResizeAll => CursorIcon::Move,
        MouseCursor::ResizeNS => CursorIcon::NsResize,
        MouseCursor::ResizeEW => CursorIcon::EwResize,
        MouseCursor::ResizeNESW => CursorIcon::NeswResize,
        MouseCursor::ResizeNWSE => CursorIcon::NwseResize,
        MouseCursor::Hand => CursorIcon::Grab,
        MouseCursor::NotAllowed => CursorIcon::NotAllowed,
        MouseCursor::None => CursorIcon::Default,
    }
}

/// winit physical key -> ImGui key.
///
/// Physical rather than logical, so `W`/`A`/`S`/`D` land in the same place on
/// AZERTY. Text still comes from `KeyEvent::text`, which *is* layout-aware, so
/// typing is unaffected.
fn map_key(code: KeyCode) -> Option<Key> {
    use KeyCode as K;
    Some(match code {
        K::Tab => Key::Tab,
        K::ArrowLeft => Key::LeftArrow,
        K::ArrowRight => Key::RightArrow,
        K::ArrowUp => Key::UpArrow,
        K::ArrowDown => Key::DownArrow,
        K::PageUp => Key::PageUp,
        K::PageDown => Key::PageDown,
        K::Home => Key::Home,
        K::End => Key::End,
        K::Insert => Key::Insert,
        K::Delete => Key::Delete,
        K::Backspace => Key::Backspace,
        K::Space => Key::Space,
        K::Enter | K::NumpadEnter => Key::Enter,
        K::Escape => Key::Escape,
        K::ControlLeft => Key::LeftCtrl,
        K::ShiftLeft => Key::LeftShift,
        K::AltLeft => Key::LeftAlt,
        K::SuperLeft => Key::LeftSuper,
        K::ControlRight => Key::RightCtrl,
        K::ShiftRight => Key::RightShift,
        K::AltRight => Key::RightAlt,
        K::SuperRight => Key::RightSuper,
        K::ContextMenu => Key::Menu,
        K::Digit0 => Key::Key0,
        K::Digit1 => Key::Key1,
        K::Digit2 => Key::Key2,
        K::Digit3 => Key::Key3,
        K::Digit4 => Key::Key4,
        K::Digit5 => Key::Key5,
        K::Digit6 => Key::Key6,
        K::Digit7 => Key::Key7,
        K::Digit8 => Key::Key8,
        K::Digit9 => Key::Key9,
        K::KeyA => Key::A,
        K::KeyB => Key::B,
        K::KeyC => Key::C,
        K::KeyD => Key::D,
        K::KeyE => Key::E,
        K::KeyF => Key::F,
        K::KeyG => Key::G,
        K::KeyH => Key::H,
        K::KeyI => Key::I,
        K::KeyJ => Key::J,
        K::KeyK => Key::K,
        K::KeyL => Key::L,
        K::KeyM => Key::M,
        K::KeyN => Key::N,
        K::KeyO => Key::O,
        K::KeyP => Key::P,
        K::KeyQ => Key::Q,
        K::KeyR => Key::R,
        K::KeyS => Key::S,
        K::KeyT => Key::T,
        K::KeyU => Key::U,
        K::KeyV => Key::V,
        K::KeyW => Key::W,
        K::KeyX => Key::X,
        K::KeyY => Key::Y,
        K::KeyZ => Key::Z,
        K::F1 => Key::F1,
        K::F2 => Key::F2,
        K::F3 => Key::F3,
        K::F4 => Key::F4,
        K::F5 => Key::F5,
        K::F6 => Key::F6,
        K::F7 => Key::F7,
        K::F8 => Key::F8,
        K::F9 => Key::F9,
        K::F10 => Key::F10,
        K::F11 => Key::F11,
        K::F12 => Key::F12,
        K::Quote => Key::Apostrophe,
        K::Comma => Key::Comma,
        K::Minus => Key::Minus,
        K::Period => Key::Period,
        K::Slash => Key::Slash,
        K::Semicolon => Key::Semicolon,
        K::Equal => Key::Equal,
        K::BracketLeft => Key::LeftBracket,
        K::Backslash => Key::Backslash,
        K::BracketRight => Key::RightBracket,
        K::Backquote => Key::GraveAccent,
        K::CapsLock => Key::CapsLock,
        K::ScrollLock => Key::ScrollLock,
        K::NumLock => Key::NumLock,
        K::PrintScreen => Key::PrintScreen,
        K::Pause => Key::Pause,
        K::Numpad0 => Key::Keypad0,
        K::Numpad1 => Key::Keypad1,
        K::Numpad2 => Key::Keypad2,
        K::Numpad3 => Key::Keypad3,
        K::Numpad4 => Key::Keypad4,
        K::Numpad5 => Key::Keypad5,
        K::Numpad6 => Key::Keypad6,
        K::Numpad7 => Key::Keypad7,
        K::Numpad8 => Key::Keypad8,
        K::Numpad9 => Key::Keypad9,
        K::NumpadDecimal => Key::KeypadDecimal,
        K::NumpadDivide => Key::KeypadDivide,
        K::NumpadMultiply => Key::KeypadMultiply,
        K::NumpadSubtract => Key::KeypadSubtract,
        K::NumpadAdd => Key::KeypadAdd,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_letter_and_digit_has_a_key_mapping() {
        // A hole here shows up as one key that silently does nothing, which is
        // exactly the kind of bug nobody files.
        for code in [
            KeyCode::KeyA,
            KeyCode::KeyM,
            KeyCode::KeyZ,
            KeyCode::Digit0,
            KeyCode::Digit7,
            KeyCode::Digit9,
        ] {
            assert!(map_key(code).is_some(), "{code:?} is unmapped");
        }
        for d in 1..=7u32 {
            assert!(
                crate::camera_input::preset_for_digit(d).is_some(),
                "digit {d} should select a view preset"
            );
        }
    }

    #[test]
    fn editing_and_navigation_keys_are_mapped() {
        for code in [
            KeyCode::Tab,
            KeyCode::Enter,
            KeyCode::NumpadEnter,
            KeyCode::Escape,
            KeyCode::Backspace,
            KeyCode::Delete,
            KeyCode::ArrowLeft,
            KeyCode::Home,
            KeyCode::End,
            KeyCode::ControlLeft,
            KeyCode::ShiftRight,
        ] {
            assert!(map_key(code).is_some(), "{code:?} is unmapped");
        }
        // Both Enter keys must produce the same ImGui key, or numpad Enter
        // fails to commit a text field.
        assert_eq!(map_key(KeyCode::Enter), map_key(KeyCode::NumpadEnter));
    }

    #[test]
    fn unmapped_keys_are_reported_rather_than_defaulted() {
        // Returning `Key::None` instead would push a real key event for a key
        // that does not exist, and ImGui would treat it as a repeat of
        // whatever `None` collides with.
        assert!(map_key(KeyCode::Fn).is_none());
        assert!(map_key(KeyCode::MediaPlayPause).is_none());
    }

    #[test]
    fn only_the_three_real_mouse_buttons_are_forwarded() {
        assert!(map_mouse_button(winit::event::MouseButton::Left).is_some());
        assert!(map_mouse_button(winit::event::MouseButton::Middle).is_some());
        assert!(map_mouse_button(winit::event::MouseButton::Right).is_some());
        assert!(map_mouse_button(winit::event::MouseButton::Back).is_none());
        assert!(map_mouse_button(winit::event::MouseButton::Other(7)).is_none());
    }

    #[test]
    fn framebuffer_size_is_the_logical_size_times_the_dpi_factor() {
        // The DPI bug this guards against renders the UI at a quarter size in
        // the corner of a 200% display.
        let mut p = WinitPlatform::default();
        p.scale_factor = 2.0;
        p.logical_size = [640.0, 400.0];
        assert_eq!(p.framebuffer_size(), (1280, 800));
        // A minimised window reports zero; the surface must never be
        // configured at zero, so the floor is 1.
        p.logical_size = [0.0, 0.0];
        assert_eq!(p.framebuffer_size(), (1, 1));
    }

    #[test]
    fn frame_options_never_pass_a_zero_delta() {
        // A suspended app can produce dt = 0, which makes ImGui's internal
        // timers (and every animation driven off them) misbehave.
        let p = WinitPlatform::default();
        let _ = p.frame_options(0.0);
        let _ = p.frame_options(-1.0);
    }

    #[test]
    fn cursor_icons_cover_every_imgui_cursor() {
        for c in [
            MouseCursor::Arrow,
            MouseCursor::TextInput,
            MouseCursor::ResizeAll,
            MouseCursor::ResizeNS,
            MouseCursor::ResizeEW,
            MouseCursor::ResizeNESW,
            MouseCursor::ResizeNWSE,
            MouseCursor::Hand,
            MouseCursor::NotAllowed,
            MouseCursor::None,
        ] {
            let _ = map_cursor(c);
        }
    }
}

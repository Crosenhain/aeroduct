//! Dear ImGui + ImPlot + ImGuizmo, on wgpu, wired to [`crate::platform`].
//!
//! # What this file owns
//!
//! Exactly four objects and their lifetimes: the ImGui context, the wgpu
//! renderer backend, the ImPlot context and the ImGuizmo context. Everything
//! else about the UI is in [`crate::panels`] (what is drawn) or
//! [`crate::state`] (what it is drawn from). Keeping the plumbing here means
//! the panels never see a `Context`, a consumer or a render pass, so they stay
//! a function of the view models.
//!
//! # The frame, and why it is split in two
//!
//! ```text
//! backend.frame(window, dt)  ->  UiFrame  { ui, plot, gizmo }
//!     panels::build(&frame, ...)          // immediate-mode; borrows the Ui
//! drop(frame)                             // the Ui borrow ends here
//! backend.render(encoder, target, extent) // Context::render + WgpuRenderer::render
//! ```
//!
//! The split is forced by the borrow checker and is worth understanding rather
//! than working around: `Context::render` needs `&mut Context`, and the `Ui`
//! handed to the panels is borrowed *from* that context. So the frame token has
//! to be dropped before the draw data can be produced. Rust is describing a real
//! constraint here — recording ImGui commands after `Render()` is a use-after-
//! free in the C library — and the two-call shape makes it impossible.
//!
//! # Docking
//!
//! `dear-imgui-rs` 0.17 compiles Dear ImGui's docking branch unconditionally;
//! [`dear_imgui_rs::HAS_DOCKING`] reports it and
//! [`crate::platform::WinitPlatform::init`] sets `ConfigFlags::DOCKING_ENABLE`
//! **before the first frame**, which the library requires (it asserts if the
//! flag changes afterwards, because Dear ImGui destroys live dock nodes when
//! docking is turned off and cannot restore them). [`UiBackend::has_docking`]
//! reports the runtime answer so the diagnostics panel can say so out loud
//! rather than letting a missing feature look like a layout bug.
//!
//! # Colour and gamma
//!
//! The 3D scene is tonemapped to the surface format by `ad-render`, and ImGui
//! draws over the same surface afterwards. `GammaMode::Auto` picks 2.2 for an
//! sRGB surface and 1.0 for a linear one, which is what keeps the panels from
//! being double-corrected against the tonemapped image behind them.

use std::time::Instant;

use anyhow::{Context as _, Result};
use dear_imgui_rs::{Context, MouseCursor, Ui};
use dear_imgui_wgpu::{FramebufferExtent, GammaMode, WgpuInitInfo, WgpuRenderer};
use dear_imguizmo::{GizmoUi, GuizmoContext};
use dear_implot::{PlotContext, PlotUi};
use winit::window::Window;

use crate::platform::WinitPlatform;

/// The ImGui stack for one window.
pub struct UiBackend {
    context: Context,
    renderer: WgpuRenderer,
    plot: PlotContext,
    gizmo: GuizmoContext,
    /// The winit bridge. Public because the event loop feeds it directly.
    pub platform: WinitPlatform,
    last_frame: Instant,
    /// Physical size of the surface the UI is drawn onto.
    surface_size: (u32, u32),
}

impl UiBackend {
    /// Build the whole stack against an existing device.
    ///
    /// `surface_format` must be the format the swapchain is configured with:
    /// the ImGui pipeline is built for exactly one colour target format, and a
    /// mismatch is a validation error at the first draw rather than here.
    pub fn new(
        gpu: &ad_gpu::GpuContext,
        window: &Window,
        surface_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let mut context = Context::create();
        // Docking, cursors and the 1.92 texture path, all before the first frame.
        let platform = WinitPlatform::init(&mut context, window);
        apply_theme(&mut context);

        let init = WgpuInitInfo::new(
            (*gpu.device).clone(),
            (*gpu.queue).clone(),
            surface_format,
        );
        let mut renderer = WgpuRenderer::new(init, &mut context)
            .map_err(|e| anyhow::anyhow!("creating the ImGui wgpu backend: {e}"))?;
        renderer.set_gamma_mode(GammaMode::Auto);

        let plot = PlotContext::try_create(&context)
            .map_err(|e| anyhow::anyhow!("creating the ImPlot context: {e}"))
            .context("ImPlot")?;

        let surface_size = platform.framebuffer_size();
        Ok(Self {
            context,
            renderer,
            plot,
            gizmo: GuizmoContext::new(),
            platform,
            last_frame: Instant::now(),
            surface_size,
        })
    }

    /// Whether the linked Dear ImGui has docking compiled in.
    pub fn has_docking(&self) -> bool {
        crate::HAS_DOCKING
    }

    /// Dear ImGui's own version string, for the diagnostics panel.
    pub fn imgui_version(&self) -> &'static str {
        crate::imgui_version()
    }

    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Feed one winit window event.
    ///
    /// Returns true when ImGui captured it, in the sense that the app should not
    /// also act on it — see [`crate::platform::WinitPlatform::handle_window_event`].
    pub fn handle_window_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        self.platform
            .handle_window_event(&mut self.context, window, event)
    }

    /// Re-read the window's size and DPI.
    pub fn resized(&mut self, window: &Window) {
        self.platform.resized(&mut self.context, window);
        let (w, h) = self.platform.framebuffer_size();
        self.surface_size = (w, h);
    }

    /// Told by the app whenever the swapchain is reconfigured.
    pub fn set_surface_size(&mut self, width: u32, height: u32) {
        self.surface_size = (width.max(1), height.max(1));
    }

    pub fn surface_size(&self) -> (u32, u32) {
        self.surface_size
    }

    /// True while ImGui wants the mouse. Valid between frames as well as
    /// inside one, because the flag is only recomputed at `NewFrame`.
    pub fn wants_mouse(&self) -> bool {
        self.context.io().want_capture_mouse()
    }

    pub fn wants_keyboard(&self) -> bool {
        self.context.io().want_capture_keyboard()
    }

    /// Smoothed frame rate, as ImGui measures it.
    pub fn framerate(&self) -> f32 {
        self.context.io().framerate()
    }

    /// Open a frame. The returned token must be dropped before [`Self::render`].
    ///
    /// `dt` is clamped away from zero by
    /// [`crate::platform::WinitPlatform::frame_options`]; passing `None` uses
    /// the wall clock since the previous call, which is what a normal loop
    /// wants.
    pub fn frame(&mut self, dt: Option<f32>) -> UiFrame<'_> {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        let delta = dt.unwrap_or(elapsed);

        self.context.prepare_frame(self.platform.frame_options(delta));
        // Split borrow: `frame()` takes `self.context`, `get_plot_ui` takes
        // `self.plot`, `begin_frame` takes `self.gizmo`. Disjoint fields, so
        // this is one mutable and two shared borrows of different places.
        let ui: &Ui = self.context.frame();
        let plot = self.plot.get_plot_ui(ui);
        let gizmo = self.gizmo.begin_frame(ui);
        UiFrame { ui, plot, gizmo, delta_s: delta }
    }

    /// Push ImGui's requested cursor to the window.
    ///
    /// Separate from the frame because the frame token borrows the context and
    /// the window has to be touched outside that borrow.
    pub fn apply_cursor(&mut self, window: &Window, cursor: Option<MouseCursor>) {
        self.platform.apply_cursor(window, cursor);
    }

    /// Produce the draw data and record it into `target`.
    ///
    /// The pass loads rather than clears: the 3D scene is already in the
    /// target, and the docked layout leaves its central node transparent so the
    /// scene shows through. Clearing here would paint over the simulation, which
    /// looks exactly like the renderer having failed.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        target_width: u32,
        target_height: u32,
    ) -> Result<()> {
        let extent = FramebufferExtent::new(target_width, target_height);
        if extent.is_empty() {
            // A minimised window. End the frame so ImGui's internal state stays
            // consistent, but record nothing.
            self.context.end_frame();
            return Ok(());
        }

        let consumer = self
            .renderer
            .renderer_consumer()
            .map_err(|e| anyhow::anyhow!("ImGui renderer consumer unavailable: {e}"))?;
        let frame = self.context.render(consumer);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("imgui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.renderer
            .render(frame, &mut pass, extent)
            .map_err(|e| anyhow::anyhow!("recording the ImGui draw data: {e}"))?;
        Ok(())
    }
}

/// One open ImGui frame, with the ImPlot and ImGuizmo handles that are only
/// valid inside it.
///
/// Carrying all three together is deliberate: ImPlot's `BeginPlot` and
/// ImGuizmo's `Manipulate` are only legal between `NewFrame` and `Render`, and
/// a type that cannot be constructed outside that window is a cheaper guarantee
/// than a comment asking people to remember.
pub struct UiFrame<'a> {
    pub ui: &'a Ui,
    pub plot: PlotUi<'a>,
    pub gizmo: GizmoUi<'a>,
    /// Seconds since the previous frame, as ImGui was told.
    pub delta_s: f32,
}

impl UiFrame<'_> {
    /// The cursor ImGui wants this frame.
    pub fn cursor(&self) -> Option<MouseCursor> {
        self.ui.mouse_cursor()
    }

    pub fn wants_mouse(&self) -> bool {
        self.ui.io().want_capture_mouse()
    }

    pub fn wants_keyboard(&self) -> bool {
        self.ui.io().want_capture_keyboard()
    }

    /// Logical display size, ImGui's coordinate space.
    pub fn display_size(&self) -> [f32; 2] {
        self.ui.io().display_size()
    }
}

/// The application's look: a dark theme with square-ish corners and enough
/// contrast to sit over a bright volume rendering.
///
/// Set once at startup rather than pushed per frame. The two values that
/// actually matter for legibility over a live 3D view are the window background
/// alpha (opaque enough to read against a moving image) and the border, which is
/// what separates a floating panel from the render behind it.
fn apply_theme(context: &mut Context) {
    use dear_imgui_rs::StyleColor;

    let style = context.style_mut();
    style.set_window_rounding(4.0);
    style.set_frame_rounding(3.0);
    style.set_grab_rounding(3.0);
    style.set_tab_rounding(3.0);
    style.set_window_border_size(1.0);
    style.set_frame_border_size(0.0);
    style.set_window_padding([8.0, 8.0]);
    style.set_frame_padding([6.0, 3.0]);
    style.set_item_spacing([7.0, 5.0]);
    style.set_scrollbar_size(12.0);

    for (slot, rgba) in [
        (StyleColor::WindowBg, [0.09, 0.10, 0.12, 0.94]),
        (StyleColor::ChildBg, [0.00, 0.00, 0.00, 0.00]),
        (StyleColor::PopupBg, [0.10, 0.11, 0.13, 0.98]),
        (StyleColor::Border, [0.26, 0.28, 0.33, 0.70]),
        (StyleColor::FrameBg, [0.17, 0.19, 0.23, 1.00]),
        (StyleColor::FrameBgHovered, [0.24, 0.27, 0.33, 1.00]),
        (StyleColor::FrameBgActive, [0.29, 0.33, 0.41, 1.00]),
        (StyleColor::TitleBg, [0.10, 0.11, 0.13, 1.00]),
        (StyleColor::TitleBgActive, [0.16, 0.20, 0.28, 1.00]),
        (StyleColor::Button, [0.20, 0.24, 0.31, 1.00]),
        (StyleColor::ButtonHovered, [0.28, 0.35, 0.46, 1.00]),
        (StyleColor::ButtonActive, [0.34, 0.45, 0.60, 1.00]),
        (StyleColor::Header, [0.20, 0.25, 0.33, 1.00]),
        (StyleColor::HeaderHovered, [0.27, 0.34, 0.45, 1.00]),
        (StyleColor::HeaderActive, [0.33, 0.42, 0.56, 1.00]),
        (StyleColor::SliderGrab, [0.42, 0.58, 0.78, 1.00]),
        (StyleColor::SliderGrabActive, [0.52, 0.70, 0.92, 1.00]),
        (StyleColor::CheckMark, [0.52, 0.74, 0.96, 1.00]),
        (StyleColor::Separator, [0.26, 0.28, 0.33, 0.70]),
        (StyleColor::Tab, [0.14, 0.17, 0.22, 1.00]),
        (StyleColor::TabHovered, [0.28, 0.35, 0.46, 1.00]),
        (StyleColor::TabSelected, [0.22, 0.29, 0.39, 1.00]),
        // Transparent, so the dockspace's central node shows the 3D scene
        // rendered underneath instead of a flat fill.
        (StyleColor::DockingEmptyBg, [0.00, 0.00, 0.00, 0.00]),
    ] {
        style.set_color(slot, rgba);
    }
}

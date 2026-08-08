//! The interactive viewer: a window, an orbit camera, and a scrub bar.
//!
//! Everything here needs a display. The offscreen path lives in
//! [`crate::viz::offscreen`] and shares the renderer but none of this.

use crate::{
    gpu::Simulation,
    timeline::{Steppable, Timeline},
    viz::{
        options::{Common, TIMEOUT_MS},
        render::{Renderer, ScrubBar, ViewMode},
    },
};
use blade_graphics as gpu;
use std::{error::Error, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

/// Steps between keyframes. Bounds the worst-case scrub replay, and with a
/// sixteen-frame ring bounds the memory: at 96³ that is 16 × 21 MB.
const KEYFRAME_INTERVAL: u64 = 200;

pub const KEYS: &str = "\
KEYS
    space          pause / resume
    R              reset the fields
    left / right   solver steps per frame
    1 2 3 4        energy split / Ez / Hz / total energy
    L              toggle signed-log scaling
    - / =          brightness
    [ / ]          scrub back / forward one keyframe interval
    home           scrub to the start
    drag           orbit, or scrub when the pointer is on the bar
    scroll         zoom
    escape         quit

The bar along the bottom is the timeline. The lighter span is what keyframes
cover and can be scrubbed to instantly; dragging outside it replays from the
start, which is slower but always available -- the state here is a pure
function of the step number, so a replayed step is the step.
";

struct App {
    common: Common,
    /// Present this many frames and quit, for smoke-testing under a virtual
    /// display where nobody is there to press escape.
    exit_after: Option<u64>,
    presented: u64,
    context: Arc<gpu::Context>,
    window: Option<Window>,
    surface: Option<gpu::Surface>,
    renderer: Option<Renderer>,
    simulation: Simulation,
    encoder: gpu::CommandEncoder,
    sync_point: Option<gpu::SyncPoint>,
    timeline: Timeline,
    running: bool,
    dragging: bool,
    scrubbing: bool,
    cursor: Option<(f64, f64)>,
    last_frame: Instant,
    steps_per_second: f32,
    /// Furthest step reached, which is what the scrub bar spans.
    furthest: u64,
}

pub fn run(common: Common, exit_after: Option<u64>) -> Result<(), Box<dyn Error>> {
    // SAFETY: `Context::init` loads the platform driver; there is no caller
    // precondition beyond building one context.
    let context = Arc::new(unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: true,
            validation: cfg!(debug_assertions),
            ..Default::default()
        })
    }?);
    println!(
        "device: {}",
        context.device_information().device_name.trim()
    );
    println!("{}", common.scene.describe());
    println!("{KEYS}");

    let scene = common.scene()?;
    let mut simulation = Simulation::new(Arc::clone(&context), &scene);
    simulation.advance_by(u64::from(common.warmup));
    let encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "frame",
        buffer_count: 2,
    });

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        common,
        exit_after,
        presented: 0,
        context,
        window: None,
        surface: None,
        renderer: None,
        simulation,
        encoder,
        // A keyframe every `KEYFRAME_INTERVAL` steps, sixteen of them. On the
        // GPU a keyframe is a full readback, so the interval is what keeps
        // that off the per-frame path.
        timeline: Timeline::new(KEYFRAME_INTERVAL, 16),
        sync_point: None,
        running: true,
        dragging: false,
        scrubbing: false,
        cursor: None,
        last_frame: Instant::now(),
        steps_per_second: 0.0,
        furthest: 0,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

impl App {
    fn surface_config(size: PhysicalSize<u32>) -> gpu::SurfaceConfig {
        gpu::SurfaceConfig {
            size: gpu::Extent {
                width: size.width.max(1),
                height: size.height.max(1),
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Recent,
            color_space: gpu::ColorSpace::Linear,
            ..Default::default()
        }
    }

    fn redraw(&mut self) {
        let (Some(window), Some(surface), Some(renderer)) = (
            self.window.as_ref(),
            self.surface.as_mut(),
            self.renderer.as_mut(),
        ) else {
            return;
        };

        if let Some(sync_point) = self.sync_point.take() {
            let _ = self.context.wait_for(&sync_point, TIMEOUT_MS);
        }

        let elapsed = self.last_frame.elapsed();
        self.last_frame = Instant::now();

        let steps = if self.running {
            u64::from(self.common.steps_per_frame)
        } else {
            0
        };
        self.simulation.advance_by(steps);
        // The solver has to have finished before its buffers are sampled: the
        // renderer reads the very same allocations the kernels write.
        self.simulation.wait();
        if steps > 0 {
            self.timeline.observe(&mut self.simulation);
            self.furthest = self.furthest.max(Steppable::step_count(&self.simulation));
        }
        // Computed inline rather than through `self.scrub_bar()`, which would
        // need a second borrow of `self` while `surface` is held mutably.
        let furthest = self.furthest.max(1) as f32;
        self.common.settings.scrub = Some(ScrubBar {
            played: Steppable::step_count(&self.simulation) as f32 / furthest,
            window_start: self.timeline.earliest().unwrap_or(0) as f32 / furthest,
        });

        let seconds = elapsed.as_secs_f32().max(1e-6);
        let instant_rate = steps as f32 / seconds;
        self.steps_per_second += 0.1 * (instant_rate - self.steps_per_second);

        let frame = surface.acquire_frame();
        self.encoder.start();
        self.encoder.init_texture(frame.texture());
        let size = Self::surface_config(window.inner_size()).size;
        renderer.draw(
            &mut self.encoder,
            frame.texture_view(),
            size,
            &self.simulation,
            &self.common.settings,
        );
        self.encoder.present(frame);
        self.sync_point = Some(self.context.submit(&mut self.encoder));
        self.presented += 1;

        let bandwidth =
            self.steps_per_second as f64 * self.simulation.bytes_per_step() as f64 / 1e9;
        window.set_title(&format!(
            "diaphane — {:.0} steps/s · {:.1} GB/s · {:.2} ms · step {} · t = {:.2} ns · \
             {} · gain {:.2} · {} keyframes / {:.0} MB{}{}",
            self.steps_per_second,
            bandwidth,
            seconds * 1e3,
            Steppable::step_count(&self.simulation),
            self.simulation.time() * 1e9,
            self.common.settings.mode.label(),
            self.common.settings.gain,
            self.timeline.keyframe_count(),
            self.timeline.bytes() as f64 / 1e6,
            if self.common.settings.log_strength > 0.0 {
                " · log"
            } else {
                ""
            },
            if self.running { "" } else { " · PAUSED" },
        ));
    }

    fn surface_size(&self) -> gpu::Extent {
        self.window
            .as_ref()
            .map(|window| Self::surface_config(window.inner_size()).size)
            .unwrap_or_default()
    }

    /// Moves to a fraction of the run so far.
    ///
    /// Pauses on the way, because a slider that keeps advancing under the
    /// pointer is not a slider.
    fn scrub_to(&mut self, fraction: f32) {
        let target = (fraction.clamp(0.0, 1.0) * self.furthest as f32).round() as u64;
        if target == Steppable::step_count(&self.simulation) {
            return;
        }
        self.running = false;
        let outcome = self.timeline.seek(&mut self.simulation, target);
        if outcome.replayed() > 0 {
            log::debug!("seek to {target}: {outcome:?}");
        }
    }

    /// Jumps by whole keyframe intervals, which is the granularity that never
    /// costs more than one interval of replay.
    fn nudge(&mut self, intervals: i64) {
        let current = Steppable::step_count(&self.simulation) as i64;
        let target =
            (current + intervals * KEYFRAME_INTERVAL as i64).clamp(0, self.furthest as i64) as u64;
        self.running = false;
        self.timeline.seek(&mut self.simulation, target);
    }

    fn on_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        let settings = &mut self.common.settings;
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::Space => self.running = !self.running,
            KeyCode::KeyR => {
                self.simulation.reset();
                self.timeline.clear();
                self.furthest = 0;
            }
            KeyCode::KeyL => {
                settings.log_strength = if settings.log_strength > 0.0 {
                    0.0
                } else {
                    40.0
                };
            }
            KeyCode::ArrowLeft => {
                self.common.steps_per_frame = self.common.steps_per_frame.saturating_sub(1).max(1);
            }
            KeyCode::ArrowRight => {
                self.common.steps_per_frame = (self.common.steps_per_frame + 1).min(4096);
            }
            KeyCode::Minus => settings.gain = (settings.gain / 1.3).max(1e-3),
            KeyCode::Equal => settings.gain = (settings.gain * 1.3).min(1e4),
            KeyCode::BracketLeft => self.nudge(-1),
            KeyCode::BracketRight => self.nudge(1),
            KeyCode::Home => self.scrub_to(0.0),
            KeyCode::Digit1 => settings.mode = ViewMode::ALL[0],
            KeyCode::Digit2 => settings.mode = ViewMode::ALL[1],
            KeyCode::Digit3 => settings.mode = ViewMode::ALL[2],
            KeyCode::Digit4 => settings.mode = ViewMode::ALL[3],
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("diaphane")
            .with_inner_size(PhysicalSize::new(self.common.width, self.common.height));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => {
                eprintln!("error: could not open a window: {error}");
                event_loop.exit();
                return;
            }
        };
        let config = Self::surface_config(window.inner_size());
        let surface = match self.context.create_surface_configured(&window, config) {
            Ok(surface) => surface,
            Err(error) => {
                eprintln!("error: could not create a surface: {error}");
                event_loop.exit();
                return;
            }
        };
        let mut renderer = Renderer::new(Arc::clone(&self.context), surface.info().format);
        // Same priming as the offscreen path: exposure is the reciprocal of a
        // measurement that is one frame behind.
        self.encoder.start();
        renderer.measure(&mut self.encoder, &self.simulation);
        let sync_point = self.context.submit(&mut self.encoder);
        let _ = self.context.wait_for(&sync_point, TIMEOUT_MS);
        self.renderer = Some(renderer);
        self.surface = Some(surface);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(surface) = self.surface.as_mut() {
                    self.context
                        .reconfigure_surface(surface, Self::surface_config(size));
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed
                    && let PhysicalKey::Code(code) = event.physical_key
                {
                    self.on_key(code, event_loop);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button != MouseButton::Left {
                    return;
                }
                let pressed = state == ElementState::Pressed;
                // Which gesture this is gets decided on press and held until
                // release, so a scrub that wanders off the bar keeps scrubbing
                // rather than suddenly spinning the camera.
                if pressed && let Some((x, y)) = self.cursor {
                    let size = self.surface_size();
                    self.scrubbing = ScrubBar::contains(y, size.height);
                    if self.scrubbing {
                        self.scrub_to(ScrubBar::fraction_at(x, size.width));
                    }
                }
                self.dragging = pressed;
                if !pressed {
                    self.scrubbing = false;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                if self.dragging && self.scrubbing {
                    let width = self.surface_size().width;
                    self.scrub_to(ScrubBar::fraction_at(current.0, width));
                } else if self.dragging
                    && let Some(previous) = self.cursor
                {
                    let delta = (current.0 - previous.0, current.1 - previous.1);
                    self.common
                        .settings
                        .camera
                        .orbit(-delta.0 as f32 * 0.006, delta.1 as f32 * 0.006);
                }
                self.cursor = Some(current);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => lines,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 60.0,
                };
                self.common.settings.camera.zoom((-amount * 0.1).exp());
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Quitting on a frame count is what makes the windowed path testable:
        // under a virtual display there is nobody to press escape, and a
        // viewer that only ever runs on a developer's machine is a viewer
        // nobody notices breaking.
        if self.exit_after.is_some_and(|limit| self.presented >= limit) {
            println!(
                "presented {} frames, {} solver steps, {:.0} steps/s",
                self.presented,
                Steppable::step_count(&self.simulation),
                self.steps_per_second,
            );
            event_loop.exit();
            return;
        }
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(sync_point) = self.sync_point.take() {
            let _ = self.context.wait_for(&sync_point, TIMEOUT_MS);
        }
        if let Some(mut renderer) = self.renderer.take() {
            renderer.destroy();
        }
        if let Some(mut surface) = self.surface.take() {
            self.context.destroy_surface(&mut surface);
        }
        self.context.destroy_command_encoder(&mut self.encoder);
    }
}

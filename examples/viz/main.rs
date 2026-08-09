//! Watch a 3D electromagnetic field evolve, in a window.
//!
//! Everything here needs a display. The offscreen path is the `render` example,
//! which shares the renderer but none of this.

#[path = "../common/mod.rs"]
mod common;

use crate::common::{
    options::{Args, COMMON_HELP, Common, TIMEOUT_MS, init_logging},
    panel::{self, Readout},
    render::{Renderer, ViewMode},
};
use blade_graphics as gpu;
use diaphane::{
    gpu::Simulation,
    timeline::{Steppable, Timeline},
};
use std::{error::Error, process, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

/// Steps between keyframes. Bounds the worst-case scrub replay.
const KEYFRAME_INTERVAL: u64 = 200;

/// What the scrub window is allowed to cost.
///
/// A keyframe is 24 bytes per cell, so a fixed *count* is a memory budget that
/// scales with the domain: sixteen of them at 96³ is 340 MB, reached over the
/// first three thousand steps, which is indistinguishable from a leak while it
/// is happening. Fixing the bytes instead makes a big domain get a shorter
/// window rather than a bigger bill.
const KEYFRAME_BUDGET: usize = 96 << 20;

fn keyframe_capacity(cells: usize) -> usize {
    let bytes = cells * 6 * std::mem::size_of::<f32>();
    // The floor is one, not two: when a single keyframe busts the budget --
    // a 256³ domain is 400 MB a frame -- keeping a second "for the window"
    // would double a bill that is already eight times the promise. One
    // keyframe still means the scrub bar works; it just replays more.
    (KEYFRAME_BUDGET / bytes.max(1)).clamp(1, 32)
}

const KEYS: &str = "\
The panel along the bottom has the transport, the scrub slider, the view mode
and the numbers. The keyboard shadows it, for the things worth doing without
aiming:

    space          pause / resume
    R              reset the fields
    left / right   solver steps per frame
    1..7           E+H / energy / E / H / total / the grid / ribbons
    L              toggle signed-log scaling
    - / =          brightness
    [ / ]          scrub back / forward one keyframe interval
    home           scrub to the start
    drag           orbit
    scroll         zoom
    escape         quit

Scrubbing inside the keyframed span restores a snapshot; outside it the run
replays from the start, which is slower and always correct -- the state is a
pure function of the step number, so a replayed step is the step. Stepping
*backwards* is that same replay: the GPU solver only runs forwards.
";

fn help() -> String {
    format!(
        "\
cargo viz — watch a 3D electromagnetic field evolve

USAGE:
    cargo viz [OPTIONS]

OPTIONS:
{COMMON_HELP}\
    --exit-after <FRAMES>         present this many frames, then quit
                                  (--steps 0 opens the window paused)

{KEYS}
Use `cargo render` to write PNGs instead, on a machine with no display.
"
    )
}

fn parse() -> Result<(Common, Option<u64>), String> {
    let mut common = Common::default();
    let mut exit_after = None;
    let mut args = Args::from_env();
    while let Some(flag) = args.next_flag() {
        if common.accept(&flag, &mut args)? {
            continue;
        }
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{}", help());
                process::exit(0);
            }
            "--exit-after" => exit_after = Some(args.parse(&flag)?),
            other => return Err(format!("unknown flag {other:?}; try --help")),
        }
    }
    Ok((common, exit_after))
}

fn main() {
    init_logging();
    let (common, exit_after) = match parse() {
        Ok(parsed) => parsed,
        Err(complaint) => {
            eprintln!("error: {complaint}");
            process::exit(2);
        }
    };
    if let Err(error) = run(common, exit_after) {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

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
    egui_context: egui::Context,
    egui_winit: Option<egui_winit::State>,
    gui: Option<blade_egui::GuiPainter>,
    running: bool,
    dragging: bool,
    cursor: Option<(f64, f64)>,
    last_frame: Instant,
    steps_per_second: f32,
    /// Tessellated egui primitives in the last frame.
    ///
    /// Reported on exit because a panel that silently stops drawing looks
    /// exactly like a panel that is drawing correctly, from out here. Under a
    /// virtual display there is nobody to notice, so the count is the evidence.
    panel_primitives: usize,
    /// Furthest step reached, which is what the scrub bar spans.
    furthest: u64,
}

fn run(mut common: Common, exit_after: Option<u64>) -> Result<(), Box<dyn Error>> {
    // `--steps 0` reads as "open it paused", so that is what it does. Taking
    // it literally meant advancing by nothing every frame, which also left
    // the auto-exposure at zero and the screen blown out white.
    let start_running = common.steps_per_frame > 0;
    common.steps_per_frame = common.steps_per_frame.max(1);
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

    let keyframes = keyframe_capacity(scene.grid.extent.total());
    println!(
        "timeline: {keyframes} keyframes of {:.0} MB, covering {} steps",
        scene.grid.extent.total() as f64 * 24.0 / 1e6,
        keyframes as u64 * KEYFRAME_INTERVAL,
    );

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
        // On the GPU a keyframe is a full readback, so the interval is what
        // keeps that off the per-frame path, and the budget is what keeps the
        // window from growing without bound.
        timeline: Timeline::new(KEYFRAME_INTERVAL, keyframes),
        egui_context: egui::Context::default(),
        egui_winit: None,
        gui: None,
        sync_point: None,
        running: start_running,
        dragging: false,
        cursor: None,
        last_frame: Instant::now(),
        steps_per_second: 0.0,
        panel_primitives: 0,
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
        let (Some(window), Some(surface), Some(renderer), Some(gui), Some(egui_winit)) = (
            self.window.as_ref(),
            self.surface.as_mut(),
            self.renderer.as_mut(),
            self.gui.as_mut(),
            self.egui_winit.as_mut(),
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

        let seconds = elapsed.as_secs_f32().max(1e-6);
        let instant_rate = steps as f32 / seconds;
        self.steps_per_second += 0.1 * (instant_rate - self.steps_per_second);

        // Sampled before the panel runs, so every widget in one frame reports
        // the same instant rather than each reading a different one.
        let readout = Readout {
            step: Steppable::step_count(&self.simulation),
            furthest: self.furthest,
            window_start: self.timeline.earliest().unwrap_or(0),
            time: self.simulation.time(),
            steps_per_second: self.steps_per_second,
            gigabytes_per_second: self.steps_per_second as f64
                * self.simulation.bytes_per_step() as f64
                / 1e9,
            frame_milliseconds: seconds * 1e3,
            keyframes: self.timeline.keyframe_count(),
            keyframe_megabytes: self.timeline.bytes() as f64 / 1e6,
            running: self.running,
            steps_per_frame: self.common.steps_per_frame,
        };

        let raw_input = egui_winit.take_egui_input(window);
        let mut commands = None;
        let output = self.egui_context.run_ui(raw_input, |ui| {
            commands = Some(panel::draw(ui, &readout, &mut self.common.settings));
        });
        egui_winit.handle_platform_output(window, output.platform_output);
        // A clicked widget keeps egui's keyboard focus, and focus reroutes
        // the keyboard: Space re-presses the last button instead of pausing,
        // and Escape needs one press to blur and another to quit. Nothing in
        // the panel takes typed input, so focus has no job here at all.
        self.egui_context.memory_mut(|memory| {
            if let Some(focused) = memory.focused() {
                memory.surrender_focus(focused);
            }
        });
        let jobs = self
            .egui_context
            .tessellate(output.shapes, output.pixels_per_point);
        self.panel_primitives = jobs.len();

        // Applied here rather than inside the panel: the UI records intent and
        // the simulation is moved in one place, so "the slider moved" and "the
        // solver stepped" cannot interleave differently per widget.
        if let Some(commands) = commands {
            if commands.toggle_running {
                self.running = !self.running;
            }
            if let Some(per_frame) = commands.steps_per_frame {
                self.common.steps_per_frame = per_frame;
            }
            if commands.reset {
                self.simulation.reset();
                self.timeline.clear();
                self.furthest = 0;
                self.running = false;
            }
            let current = Steppable::step_count(&self.simulation);
            // Stepping is not clamped to the furthest step reached: at the
            // head of the run -- which is where a paused viewer usually is --
            // stepping forward means advancing the solver, and clamping there
            // made the button a no-op in its main use case.
            let target = commands.seek.or_else(|| {
                (commands.step_by != 0).then(|| current.saturating_add_signed(commands.step_by))
            });
            if let Some(target) = target
                && target != current
            {
                // Any move on the timeline pauses, because a slider that keeps
                // advancing under the pointer is not a slider.
                self.running = false;
                self.timeline.seek(&mut self.simulation, target);
                self.simulation.wait();
                self.timeline.observe(&mut self.simulation);
                self.furthest = self.furthest.max(Steppable::step_count(&self.simulation));
            }
        }

        let frame = surface.acquire_frame();
        self.encoder.start();
        self.encoder.init_texture(frame.texture());
        let size = Self::surface_config(window.inner_size()).size;
        gui.update_textures(&mut self.encoder, &output.textures_delta, &self.context);
        renderer.draw(
            &mut self.encoder,
            frame.texture_view(),
            size,
            &self.simulation,
            &self.common.settings,
            steps > 0,
        );
        {
            // A second pass that loads rather than clears, so the panel lands
            // on top of the volume. Keeping it out of `Renderer` is what lets
            // the offscreen program share the render pass without linking a UI
            // toolkit it has no display for.
            let mut pass = self.encoder.render(
                "panel",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: frame.texture_view(),
                        init_op: gpu::InitOp::Load,
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            );
            gui.paint(
                &mut pass,
                &jobs,
                &blade_egui::ScreenDescriptor {
                    physical_size: (size.width, size.height),
                    scale_factor: output.pixels_per_point,
                },
                &self.context,
            );
        }
        self.encoder.present(frame);
        let sync_point = self.context.submit(&mut self.encoder);
        gui.after_submit(&sync_point);
        self.sync_point = Some(sync_point);
        self.presented += 1;
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
                // Pausing too, exactly like the panel's ⏮: a reset means
                // "back to the start, look at it", and the two spellings of
                // the same action should not disagree about what happens next.
                self.simulation.reset();
                self.timeline.clear();
                self.furthest = 0;
                self.running = false;
            }
            KeyCode::KeyL => settings.toggle_log(),
            KeyCode::ArrowLeft => {
                self.common.steps_per_frame = self
                    .common
                    .steps_per_frame
                    .saturating_sub(1)
                    .max(*panel::STEPS_PER_FRAME.start());
            }
            KeyCode::ArrowRight => {
                self.common.steps_per_frame =
                    (self.common.steps_per_frame + 1).min(*panel::STEPS_PER_FRAME.end());
            }
            KeyCode::Minus => settings.gain = (settings.gain / 1.3).max(*panel::GAIN.start()),
            KeyCode::Equal => settings.gain = (settings.gain * 1.3).min(*panel::GAIN.end()),
            KeyCode::BracketLeft => self.nudge(-1),
            KeyCode::BracketRight => self.nudge(1),
            KeyCode::Home => self.scrub_to(0.0),
            KeyCode::Digit1 => settings.mode = ViewMode::ALL[0],
            KeyCode::Digit2 => settings.mode = ViewMode::ALL[1],
            KeyCode::Digit3 => settings.mode = ViewMode::ALL[2],
            KeyCode::Digit4 => settings.mode = ViewMode::ALL[3],
            KeyCode::Digit5 => settings.mode = ViewMode::ALL[4],
            KeyCode::Digit6 => settings.mode = ViewMode::ALL[5],
            KeyCode::Digit7 => settings.mode = ViewMode::ALL[6],
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
        self.gui = Some(blade_egui::GuiPainter::new(surface.info(), &self.context));
        self.egui_winit = Some(egui_winit::State::new(
            self.egui_context.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        ));
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
        // egui sees every event first and says whether it wants it. Without
        // that, dragging a slider also orbits the camera and typing in the
        // panel also toggles the view mode -- the pointer is over a widget, so
        // the widget owns it.
        let consumed = match (self.window.as_ref(), self.egui_winit.as_mut()) {
            (Some(window), Some(state)) => state.on_window_event(window, &event).consumed,
            _ => false,
        };
        if consumed && !matches!(event, WindowEvent::RedrawRequested) {
            return;
        }
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
                // Always an orbit now. Scrubbing is the panel's slider, and a
                // drag that reaches here is one egui declined -- so the pointer
                // is over the volume, and over the volume a drag turns it.
                self.dragging = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                if self.dragging
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
                "presented {} frames, {} solver steps, {:.0} steps/s, \
                 {} panel primitives",
                self.presented,
                Steppable::step_count(&self.simulation),
                self.steps_per_second,
                self.panel_primitives,
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
        if let Some(mut gui) = self.gui.take() {
            gui.destroy(&self.context);
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

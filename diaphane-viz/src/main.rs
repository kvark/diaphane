//! The visualizer.
//!
//! Runs a [`diaphane`] simulation on the GPU and draws its field buffers
//! directly, hundreds of solver steps per displayed frame.
//!
//! Two modes. With no `--frames` it opens a window and runs interactively.
//! With `--frames N` it renders offscreen to PNGs and exits, which needs no
//! display at all — that is what CI runs, and what makes the render path
//! something that gets exercised rather than merely compiled.

mod camera;
mod render;

use crate::render::{Capture, Renderer, ViewMode, ViewSettings};
use blade_graphics as gpu;
use diaphane::{Extent, Scene, gpu::Simulation};
use std::{env, error::Error, path::PathBuf, process, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const TIMEOUT_MS: u32 = 120_000;

const HELP: &str = "\
diaphane-viz — watch a 3D electromagnetic field evolve

USAGE:
    diaphane-viz [OPTIONS]

OPTIONS:
    --scene <photon|cavity|slab>  what to simulate           [default: photon]
    --extent <CELLS>              cube side in cells         [default: 96]
    --steps <N>                   solver steps per frame     [default: 8]
    --warmup <N>                  steps to run before the first frame [default: 0]
    --mode <energy|electric|magnetic|magnitude>              [default: energy]
    --gain <F>                    brightness multiplier      [default: 1.0]
    --log <F>                     signed-log strength, 0 = linear [default: 6]
    --frames <N>                  render N frames offscreen and exit
    --output-dir <PATH>           where the PNGs go          [default: frames]
    --size <WxH>                  offscreen resolution       [default: 720x540]
    -h, --help                    this

KEYS (windowed)
    space          pause / resume
    R              reset the fields
    left / right   solver steps per frame
    1 2 3 4        energy split / Ez / Hz / total energy
    L              toggle signed-log scaling
    - / =          brightness
    drag           orbit
    scroll         zoom
    escape         quit
";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SceneKind {
    Photon,
    Cavity,
    Slab,
}

impl SceneKind {
    fn build(self, extent: Extent) -> Scene {
        match self {
            Self::Photon => Scene::photon(extent),
            Self::Cavity => Scene::cavity(extent),
            Self::Slab => Scene::slab(extent, 1.8),
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Self::Photon => "a wave packet crossing free space",
            Self::Cavity => "a dipole ringing a closed conducting box",
            Self::Slab => "a wave packet meeting a dielectric slab",
        }
    }
}

struct Options {
    scene: SceneKind,
    extent: u32,
    steps_per_frame: u32,
    warmup: u32,
    frames: Option<u32>,
    output_dir: PathBuf,
    width: u32,
    height: u32,
    settings: ViewSettings,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut options = Self {
            scene: SceneKind::Photon,
            extent: 96,
            steps_per_frame: 8,
            warmup: 0,
            frames: None,
            output_dir: PathBuf::from("frames"),
            width: 720,
            height: 540,
            settings: ViewSettings::new(),
        };
        let mut arguments = env::args().skip(1);
        while let Some(flag) = arguments.next() {
            let mut value = || {
                arguments
                    .next()
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match flag.as_str() {
                "-h" | "--help" => {
                    print!("{HELP}");
                    process::exit(0);
                }
                "--scene" => {
                    options.scene = match value()?.as_str() {
                        "photon" => SceneKind::Photon,
                        "cavity" => SceneKind::Cavity,
                        "slab" => SceneKind::Slab,
                        other => return Err(format!("unknown scene {other:?}")),
                    }
                }
                "--mode" => {
                    options.settings.mode = match value()?.as_str() {
                        "energy" => ViewMode::Energy,
                        "electric" => ViewMode::Electric,
                        "magnetic" => ViewMode::Magnetic,
                        "magnitude" => ViewMode::Magnitude,
                        other => return Err(format!("unknown mode {other:?}")),
                    }
                }
                "--extent" => options.extent = parse(&value()?, "--extent")?,
                "--steps" => options.steps_per_frame = parse(&value()?, "--steps")?,
                "--warmup" => options.warmup = parse(&value()?, "--warmup")?,
                "--frames" => options.frames = Some(parse(&value()?, "--frames")?),
                "--gain" => options.settings.gain = parse(&value()?, "--gain")?,
                "--log" => options.settings.log_strength = parse(&value()?, "--log")?,
                "--output-dir" => options.output_dir = PathBuf::from(value()?),
                "--size" => {
                    let text = value()?;
                    let (width, height) = text
                        .split_once(['x', 'X'])
                        .ok_or_else(|| format!("--size wants WxH, got {text:?}"))?;
                    options.width = parse(width, "--size")?;
                    options.height = parse(height, "--size")?;
                }
                other => return Err(format!("unknown flag {other:?}; try --help")),
            }
        }
        Ok(options)
    }

    fn scene(&self) -> Scene {
        let scene = self.scene.build(Extent::cube(self.extent));
        if let Err(complaint) = scene.validate() {
            eprintln!("warning: {complaint}");
        }
        scene
    }
}

fn parse<T: std::str::FromStr>(text: &str, flag: &str) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("{flag} could not read {text:?}"))
}

fn main() {
    env_logger_init();
    let options = match Options::parse() {
        Ok(options) => options,
        Err(complaint) => {
            eprintln!("error: {complaint}");
            process::exit(2);
        }
    };
    let result = match options.frames {
        Some(frames) => render_offscreen(&options, frames),
        None => run_windowed(options),
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

fn env_logger_init() {
    // Blade logs device selection at info level, which is worth seeing when a
    // machine has more than one.
    if env::var("RUST_LOG").is_err() {
        unsafe { env::set_var("RUST_LOG", "warn") };
    }
}

/// Renders `frames` frames to PNGs without ever touching a window system.
fn render_offscreen(options: &Options, frames: u32) -> Result<(), Box<dyn Error>> {
    let context = diaphane::gpu::headless_context()?;
    println!(
        "device: {}",
        context.device_information().device_name.trim()
    );

    let scene = options.scene();
    let mut simulation = Simulation::new(Arc::clone(&context), &scene);
    let mut renderer = Renderer::new(Arc::clone(&context), Capture::FORMAT);
    let mut capture = Capture::new(Arc::clone(&context), options.width, options.height);
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "capture",
        buffer_count: 2,
    });

    println!(
        "{}: {}³ cells, {} steps per frame, {} frame(s)",
        options.scene.describe(),
        options.extent,
        options.steps_per_frame,
        frames
    );

    // Fast-forward to the moment worth looking at, so a short capture does not
    // have to start from an empty domain.
    if options.warmup > 0 {
        simulation.advance_by(u64::from(options.warmup));
        simulation.wait();
    }

    for frame in 0..frames {
        simulation.advance_by(u64::from(options.steps_per_frame));
        simulation.wait();

        // Offscreen has no latency budget to protect, so the auto-range is
        // measured and consumed within the same frame. A pulse's peak can
        // climb by orders of magnitude between frames while it is building,
        // and a scale one frame stale would blow the exposure out completely.
        encoder.start();
        renderer.measure(&mut encoder, &simulation);
        let sync_point = context.submit(&mut encoder);
        context.wait_for(&sync_point, TIMEOUT_MS)?;

        encoder.start();
        capture.initialize(&mut encoder);
        renderer.draw(
            &mut encoder,
            capture.view(),
            capture.size(),
            &simulation,
            &options.settings,
        );
        capture.copy_out(&mut encoder);
        let sync_point = context.submit(&mut encoder);
        context.wait_for(&sync_point, TIMEOUT_MS)?;

        let path = options.output_dir.join(format!("frame{frame:04}.png"));
        capture.write_png(&path)?;
        println!(
            "{}  step {}  t = {:.3} ns  peak = {:.3e}",
            path.display(),
            simulation.step_count(),
            simulation.time() * 1e9,
            renderer.scale()
        );
    }

    context.destroy_command_encoder(&mut encoder);
    capture.destroy();
    renderer.destroy();
    Ok(())
}

/// Interactive state that survives across frames.
struct App {
    options: Options,
    context: Arc<gpu::Context>,
    window: Option<Window>,
    surface: Option<gpu::Surface>,
    renderer: Option<Renderer>,
    simulation: Simulation,
    encoder: gpu::CommandEncoder,
    sync_point: Option<gpu::SyncPoint>,
    running: bool,
    dragging: bool,
    cursor: Option<(f64, f64)>,
    last_frame: Instant,
    steps_per_second: f32,
}

fn run_windowed(options: Options) -> Result<(), Box<dyn Error>> {
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
    println!("{}", options.scene.describe());
    println!("{}", &HELP[HELP.find("KEYS").unwrap_or(0)..]);

    let scene = options.scene();
    let mut simulation = Simulation::new(Arc::clone(&context), &scene);
    simulation.advance_by(u64::from(options.warmup));
    let encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "frame",
        buffer_count: 2,
    });

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        options,
        context,
        window: None,
        surface: None,
        renderer: None,
        simulation,
        encoder,
        sync_point: None,
        running: true,
        dragging: false,
        cursor: None,
        last_frame: Instant::now(),
        steps_per_second: 0.0,
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
            u64::from(self.options.steps_per_frame)
        } else {
            0
        };
        self.simulation.advance_by(steps);
        // The solver has to have finished before its buffers are sampled: the
        // renderer reads the very same allocations the kernels write.
        self.simulation.wait();

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
            &self.options.settings,
        );
        self.encoder.present(frame);
        self.sync_point = Some(self.context.submit(&mut self.encoder));

        let bandwidth =
            self.steps_per_second as f64 * self.simulation.bytes_per_step() as f64 / 1e9;
        window.set_title(&format!(
            "diaphane — {:.0} steps/s · {:.1} GB/s · {:.2} ms · t = {:.2} ns · {} · gain {:.2}{}{}",
            self.steps_per_second,
            bandwidth,
            seconds * 1e3,
            self.simulation.time() * 1e9,
            self.options.settings.mode.label(),
            self.options.settings.gain,
            if self.options.settings.log_strength > 0.0 {
                " · log"
            } else {
                ""
            },
            if self.running { "" } else { " · PAUSED" },
        ));
    }

    fn on_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        let settings = &mut self.options.settings;
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::Space => self.running = !self.running,
            KeyCode::KeyR => self.simulation.reset(),
            KeyCode::KeyL => {
                settings.log_strength = if settings.log_strength > 0.0 {
                    0.0
                } else {
                    40.0
                };
            }
            KeyCode::ArrowLeft => {
                self.options.steps_per_frame =
                    self.options.steps_per_frame.saturating_sub(1).max(1);
            }
            KeyCode::ArrowRight => {
                self.options.steps_per_frame = (self.options.steps_per_frame + 1).min(4096);
            }
            KeyCode::Minus => settings.gain = (settings.gain / 1.3).max(1e-3),
            KeyCode::Equal => settings.gain = (settings.gain * 1.3).min(1e4),
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
            .with_inner_size(PhysicalSize::new(self.options.width, self.options.height));
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
                if button == MouseButton::Left {
                    self.dragging = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                if self.dragging
                    && let Some(previous) = self.cursor
                {
                    let delta = (current.0 - previous.0, current.1 - previous.1);
                    self.options
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
                self.options.settings.camera.zoom((-amount * 0.1).exp());
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
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

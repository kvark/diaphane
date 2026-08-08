//! Render a simulation to PNG files, with no window system involved.
//!
//! This is what CI runs, and what makes the render path something that gets
//! exercised on every push rather than only ever on a developer's desktop. It
//! is a separate program from the viewer because it shares the renderer and
//! nothing else: no event loop, no surface, no display.

#[path = "../common/mod.rs"]
mod common;

use crate::common::{
    options::{Args, COMMON_HELP, Common, TIMEOUT_MS, init_logging},
    render::{Capture, Renderer, ScrubBar},
};
use blade_graphics as gpu;
use diaphane::gpu::Simulation;
use std::{
    error::Error,
    path::{Path, PathBuf},
    process,
    sync::Arc,
};

/// Flags for the offscreen renderer, on top of the shared ones.
struct Options {
    common: Common,
    frames: u32,
    output_dir: PathBuf,
    save_scene: Option<PathBuf>,
    /// Draw the scrub bar into the frames too. Off by default because these
    /// are meant to be clean images; CI turns it on so that branch of the
    /// shader is not dead code.
    timeline: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            common: Common::default(),
            frames: 1,
            output_dir: PathBuf::from("frames"),
            save_scene: None,
            timeline: false,
        }
    }
}

fn help() -> String {
    format!(
        "\
cargo render — render a 3D electromagnetic field to PNG files

USAGE:
    cargo render [OPTIONS]

OPTIONS:
{COMMON_HELP}\
    --frames <N>                  how many frames to write   [default: 1]
    --output-dir <PATH>           where the PNGs go          [default: frames]
    --save-scene <PATH>           write the scene out and exit
    --timeline                    draw the scrub bar into the frames

Needs no window system, so this is what CI runs.
"
    )
}

fn parse() -> Result<Options, String> {
    let mut options = Options::default();
    let mut args = Args::from_env();
    while let Some(flag) = args.next_flag() {
        if options.common.accept(&flag, &mut args)? {
            continue;
        }
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{}", help());
                process::exit(0);
            }
            "--frames" => options.frames = args.parse(&flag)?,
            "--output-dir" => options.output_dir = PathBuf::from(args.value(&flag)?),
            "--save-scene" => options.save_scene = Some(PathBuf::from(args.value(&flag)?)),
            "--timeline" => options.timeline = true,
            other => return Err(format!("unknown flag {other:?}; try --help")),
        }
    }
    Ok(options)
}

fn main() {
    init_logging();
    let options = match parse() {
        Ok(options) => options,
        Err(complaint) => {
            eprintln!("error: {complaint}");
            process::exit(2);
        }
    };
    let result = match options.save_scene.clone() {
        Some(path) => save_scene(&options, &path),
        None => run(&options),
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

/// Writes the resolved scene out as RON and stops.
///
/// The intended way to start authoring: take a preset, dump it, edit the file.
fn save_scene(options: &Options, path: &Path) -> Result<(), Box<dyn Error>> {
    let scene = options.common.scene()?;
    scene.save(path)?;
    println!(
        "wrote {} ({} cells)",
        path.display(),
        scene.grid.extent.total()
    );
    Ok(())
}

/// Renders `frames` frames to PNGs without ever touching a window system.
fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    let frames = options.frames;
    let context = diaphane::gpu::headless_context()?;
    println!(
        "device: {}",
        context.device_information().device_name.trim()
    );

    let scene = options.common.scene()?;
    let mut simulation = Simulation::new(Arc::clone(&context), &scene);
    let mut renderer = Renderer::new(Arc::clone(&context), Capture::FORMAT);
    let mut capture = Capture::new(
        Arc::clone(&context),
        options.common.width,
        options.common.height,
    );
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "capture",
        buffer_count: 2,
    });

    println!(
        "{}: {}³ cells, {} steps per frame, {} frame(s)",
        options.common.scene.describe(),
        options.common.extent,
        options.common.steps_per_frame,
        frames
    );

    // Fast-forward to the moment worth looking at, so a short capture does not
    // have to start from an empty domain.
    if options.common.warmup > 0 {
        simulation.advance_by(u64::from(options.common.warmup));
        simulation.wait();
    }

    let mut settings = options.common.settings;
    for frame in 0..frames {
        simulation.advance_by(u64::from(options.common.steps_per_frame));
        simulation.wait();

        // Offscreen frames are meant to be clean images, so the bar is opt-in.
        // CI turns it on, which is what keeps that branch of the shader from
        // being dead code that only ever runs on someone's desktop.
        if options.timeline {
            settings.scrub = Some(ScrubBar {
                played: (frame + 1) as f32 / frames as f32,
                window_start: 0.35,
            });
        }

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
            &settings,
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

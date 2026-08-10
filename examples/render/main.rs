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
    render::{Animation, Capture, Renderer, ScrubBar, ViewMode},
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
    /// Collect the frames into one animated GIF instead of writing PNGs.
    gif: Option<PathBuf>,
    /// Hundredths of a second per GIF frame.
    gif_delay: u16,
    /// Draw the scrub bar into the frames too. Off by default because these
    /// are meant to be clean images; CI turns it on so that branch of the
    /// shader is not dead code.
    timeline: bool,
    /// Run the film backwards once the solver reaches this step.
    reverse_at: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            common: Common::default(),
            frames: 1,
            output_dir: PathBuf::from("frames"),
            save_scene: None,
            gif: None,
            gif_delay: 5,
            timeline: false,
            reverse_at: None,
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
    --gif <PATH>                  collect the frames into one animated GIF
    --gif-delay <CENTISECONDS>    per GIF frame                [default: 5]
    --timeline                    draw the scrub bar into the frames
    --reverse-at <STEP>           run the film backwards from this step on
                                  (needs a lossless scene: PEC walls, no metal)

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
            "--gif" => options.gif = Some(PathBuf::from(args.value(&flag)?)),
            "--gif-delay" => options.gif_delay = args.parse(&flag)?,
            "--timeline" => options.timeline = true,
            "--reverse-at" => options.reverse_at = Some(args.parse(&flag)?),
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
    // Refused here with a sentence rather than three frames deep with an
    // assert: the flag is the reversal feature's front door.
    if options.reverse_at.is_some() && !scene.is_reversible() {
        return Err(
            "--reverse-at needs a lossless scene: PEC walls (the cavity \
                    preset, or `boundary: Pec`) and no conductive material"
                .into(),
        );
    }
    let mut simulation = Simulation::new(Arc::clone(&context), &scene);
    // Before the warmup, so `--mode intensity` averages over it too -- the
    // whole point of the view is a long integration.
    simulation.set_accumulate_intensity(options.common.settings.mode == ViewMode::Intensity);
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

    // The resolved extent, not the --extent flag: a scene file carries its
    // own grid, and a graded one has more cells than its recipe names.
    let extent = scene.grid.extent;
    println!(
        "{}: {}×{}×{} cells, {} steps per frame, {} frame(s)",
        options.common.scene.describe(),
        extent.x,
        extent.y,
        extent.z,
        options.common.steps_per_frame,
        frames
    );

    // Fast-forward to the moment worth looking at, so a short capture does not
    // have to start from an empty domain.
    if options.common.warmup > 0 {
        simulation.advance_by(u64::from(options.common.warmup));
        simulation.wait();
    }

    let mut animation = options.gif.as_ref().map(|_| {
        Animation::new(
            options.common.width,
            options.common.height,
            options.gif_delay,
        )
    });
    let mut settings = options.common.settings;
    // Latched, not re-derived: the first reversal drops the step back below
    // the threshold, and re-checking would advance again -- a ping-pong
    // around the turnaround instead of a film run backwards.
    let mut backwards = false;
    for frame in 0..frames {
        // Past the turnaround the film runs backwards -- the leapfrog is an
        // involution, so this is the solver actually un-stepping, not a
        // replay. At step zero it parks rather than underflowing.
        backwards = backwards
            || options
                .reverse_at
                .is_some_and(|at| simulation.step_count() >= at);
        if backwards {
            let steps = u64::from(options.common.steps_per_frame).min(simulation.step_count());
            simulation.reverse_by(steps);
        } else {
            simulation.advance_by(u64::from(options.common.steps_per_frame));
        }
        simulation.wait();

        // Offscreen frames are meant to be clean images, so the bar is opt-in.
        // CI turns it on, which is what keeps that branch of the shader from
        // being dead code that only ever runs on someone's desktop.
        if options.timeline {
            settings.scrub = Some(ScrubBar {
                played: (frame + 1) as f32 / frames as f32,
                // The offscreen renderer keeps no keyframes, so nothing is
                // instantly reachable and none of the bar gets the window
                // tint. It used to fabricate 0.35 to demonstrate the colour.
                window_start: 1.0,
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
            true,
        );
        capture.copy_out(&mut encoder);
        let sync_point = context.submit(&mut encoder);
        context.wait_for(&sync_point, TIMEOUT_MS)?;

        let where_it_went = match animation.as_mut() {
            Some(animation) => {
                animation.push(capture.pixels());
                format!("frame {}", animation.len())
            }
            None => {
                let path = options.output_dir.join(format!("frame{frame:04}.png"));
                capture.write_png(&path)?;
                path.display().to_string()
            }
        };
        println!(
            "{where_it_went}  step {}  t = {:.3} ns  peak = {:.3e}",
            simulation.step_count(),
            simulation.time() * 1e9,
            renderer.scale()
        );
    }

    if let (Some(animation), Some(path)) = (animation.as_ref(), options.gif.as_ref()) {
        animation.write(path)?;
        println!("wrote {} ({} frames)", path.display(), animation.len());
    }

    context.destroy_command_encoder(&mut encoder);
    capture.destroy();
    renderer.destroy();
    Ok(())
}

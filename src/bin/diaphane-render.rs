//! Render a simulation to PNG files. Needs no display.

use diaphane::viz::{
    offscreen,
    options::{Args, COMMON_HELP, init_logging},
};
use std::{path::PathBuf, process};

fn help() -> String {
    format!(
        "\
diaphane-render — render a 3D electromagnetic field to PNG files

USAGE:
    diaphane-render [OPTIONS]

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

fn parse() -> Result<offscreen::Options, String> {
    let mut options = offscreen::Options::default();
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
        Some(path) => offscreen::save_scene(&options, &path),
        None => offscreen::run(&options),
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

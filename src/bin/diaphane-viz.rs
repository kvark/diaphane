//! Watch a 3D electromagnetic field evolve, in a window.

use diaphane::viz::{
    app,
    options::{Args, COMMON_HELP, Common, init_logging},
};
use std::process;

fn help() -> String {
    format!(
        "\
diaphane-viz — watch a 3D electromagnetic field evolve

USAGE:
    diaphane-viz [OPTIONS]

OPTIONS:
{COMMON_HELP}\
    --exit-after <FRAMES>         present this many frames, then quit

{}
Use `diaphane-render` to write PNGs instead, on a machine with no display.
",
        app::KEYS
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
    if let Err(error) = app::run(common, exit_after) {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

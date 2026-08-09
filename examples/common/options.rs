//! Command-line parsing shared by the two programs.
//!
//! `cargo viz` opens a window; `cargo render` writes PNGs and needs no display
//! at all. They share everything about *what* to simulate and how to colour it,
//! and share nothing about where the pixels go — so the common part lives here
//! and each program owns its own flags and its own help text. A single program
//! with an `--offscreen` switch would mean every flag has to document which
//! mode it applies in.

use crate::common::render::{ViewMode, ViewSettings};
use diaphane::{Extent, Scene};
use std::{env, path::PathBuf, str::FromStr};

/// Millisecond timeout on every GPU wait. Generous, because a batch of a few
/// thousand steps on a software rasterizer is not fast.
pub const TIMEOUT_MS: u32 = 120_000;

/// A flag stream, with the small conveniences both parsers want.
pub struct Args {
    items: std::vec::IntoIter<String>,
}

impl Args {
    pub fn from_env() -> Self {
        Self {
            items: env::args().skip(1).collect::<Vec<_>>().into_iter(),
        }
    }

    pub fn next_flag(&mut self) -> Option<String> {
        self.items.next()
    }

    pub fn value(&mut self, flag: &str) -> Result<String, String> {
        self.items
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))
    }

    pub fn parse<T: FromStr>(&mut self, flag: &str) -> Result<T, String> {
        let text = self.value(flag)?;
        text.parse()
            .map_err(|_| format!("{flag} could not read {text:?}"))
    }

    /// `WxH`.
    pub fn size(&mut self, flag: &str) -> Result<(u32, u32), String> {
        let text = self.value(flag)?;
        let (width, height) = text
            .split_once(['x', 'X'])
            .ok_or_else(|| format!("{flag} wants WxH, got {text:?}"))?;
        let parse = |part: &str| {
            part.parse::<u32>()
                .map_err(|_| format!("{flag} could not read {text:?}"))
        };
        Ok((parse(width)?, parse(height)?))
    }
}

/// Where the scene comes from: a built-in preset, or a file.
#[derive(Clone, Debug, PartialEq)]
pub enum SceneSource {
    Photon,
    Cavity,
    Slab,
    File(PathBuf),
}

impl SceneSource {
    /// Anything that is not a preset name is taken as a path, so
    /// `--scene scenes/double-slit.ron` needs no extra flag.
    pub fn parse(argument: &str) -> Self {
        match argument {
            "photon" => Self::Photon,
            "cavity" => Self::Cavity,
            "slab" => Self::Slab,
            path => Self::File(PathBuf::from(path)),
        }
    }

    fn build(&self, extent: Extent) -> Result<Scene, String> {
        // `--extent` sizes the presets, which are defined at a millimetre per
        // cell. A scene file carries its own domain, so the flag does not
        // apply to it -- use `--resolution` to refine one instead.
        Ok(match *self {
            Self::Photon => Scene::photon(extent),
            Self::Cavity => Scene::cavity(extent),
            Self::Slab => Scene::slab(extent, 1.8),
            Self::File(ref path) => Scene::load(path)?,
        })
    }

    pub fn describe(&self) -> String {
        match *self {
            Self::Photon => "a wave packet crossing free space".to_string(),
            Self::Cavity => "a dipole ringing a closed conducting box".to_string(),
            Self::Slab => "a wave packet meeting a dielectric slab".to_string(),
            Self::File(ref path) => path.display().to_string(),
        }
    }
}

/// The flags both binaries accept.
#[derive(Clone, Debug)]
pub struct Common {
    pub scene: SceneSource,
    pub extent: u32,
    pub resolution: Option<f32>,
    pub steps_per_frame: u32,
    pub warmup: u32,
    pub width: u32,
    pub height: u32,
    pub settings: ViewSettings,
}

/// The shared part of both help texts, so the two cannot drift apart.
///
/// `\x20` rather than a plain space because a `\` line continuation eats the
/// leading whitespace of the next line as well as the newline, which silently
/// unindented the first flag and only the first flag.
pub const COMMON_HELP: &str = "\
\x20   --scene <NAME|PATH.ron>       a preset or a scene file   [default: photon]
                                  presets: photon, cavity, slab
    --extent <CELLS>              cube side in cells         [default: 96]
    --resolution <CELLS/M>        rediscretize without moving anything
    --steps <N>                   solver steps per frame     [default: 8]
    --warmup <N>                  steps to run before the first frame
    --mode <fields|energy|electric|magnetic|magnitude|grid>  [default: fields]
    --gain <F>                    brightness multiplier      [default: 1.0]
    --log <F>                     signed-log strength, 0 = linear [default: 6]
    --size <WxH>                  resolution                 [default: 720x540]
    -h, --help                    this
";

impl Common {
    /// Consumes `flag` if it is one of the shared ones.
    ///
    /// Returns false when it is not, so the caller can try its own.
    pub fn accept(&mut self, flag: &str, args: &mut Args) -> Result<bool, String> {
        match flag {
            "--scene" => self.scene = SceneSource::parse(&args.value(flag)?),
            "--extent" => self.extent = args.parse(flag)?,
            "--resolution" => self.resolution = Some(args.parse(flag)?),
            "--steps" => self.steps_per_frame = args.parse(flag)?,
            "--warmup" => self.warmup = args.parse(flag)?,
            "--gain" => self.settings.gain = args.parse(flag)?,
            "--log" => self.settings.log_strength = args.parse(flag)?,
            "--size" => (self.width, self.height) = args.size(flag)?,
            "--mode" => {
                self.settings.mode = match args.value(flag)?.as_str() {
                    "fields" => ViewMode::Fields,
                    "energy" => ViewMode::Energy,
                    "electric" => ViewMode::Electric,
                    "magnetic" => ViewMode::Magnetic,
                    "magnitude" => ViewMode::Magnitude,
                    "grid" => ViewMode::Grid,
                    other => return Err(format!("unknown mode {other:?}")),
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Resolves the scene, warning about anything questionable rather than
    /// refusing to run — a scene can be under-resolved on purpose.
    pub fn scene(&self) -> Result<Scene, String> {
        let mut scene = self.scene.build(Extent::cube(self.extent))?;
        // Geometry and sources are in metres, so this rediscretizes the same
        // physical problem rather than resizing it.
        if let Some(resolution) = self.resolution {
            scene = scene.with_resolution(resolution);
        }
        if let Err(complaint) = scene.validate() {
            eprintln!("warning: {complaint}");
        }
        Ok(scene)
    }
}

impl Default for Common {
    fn default() -> Self {
        Self {
            scene: SceneSource::Photon,
            extent: 96,
            resolution: None,
            steps_per_frame: 8,
            warmup: 0,
            width: 720,
            height: 540,
            settings: ViewSettings::new(),
        }
    }
}

/// Blade logs device selection at info level, which is worth seeing when a
/// machine has more than one.
pub fn init_logging() {
    if env::var("RUST_LOG").is_err() {
        // SAFETY: called once, before any thread is spawned.
        unsafe { env::set_var("RUST_LOG", "warn") };
    }
}

#[cfg(test)]
mod tests {
    use super::{Args, Common, SceneSource};
    use crate::common::render::ViewMode;
    use std::path::PathBuf;

    fn args(items: &[&str]) -> Args {
        Args {
            items: items
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .into_iter(),
        }
    }

    #[test]
    fn accepts_the_shared_flags() {
        let mut common = Common::default();
        let mut args = args(&["electric", "2.5", "1600x900", "48"]);
        assert!(common.accept("--mode", &mut args).unwrap());
        assert!(common.accept("--gain", &mut args).unwrap());
        assert!(common.accept("--size", &mut args).unwrap());
        assert!(common.accept("--extent", &mut args).unwrap());

        assert_eq!(common.settings.mode, ViewMode::Electric);
        assert_eq!(common.settings.gain, 2.5);
        assert_eq!((common.width, common.height), (1600, 900));
        assert_eq!(common.extent, 48);
    }

    #[test]
    fn declines_flags_it_does_not_own() {
        // This is what lets each binary keep its own flags without the shared
        // parser having to know about them.
        let mut common = Common::default();
        let mut args = args(&["7"]);
        assert!(!common.accept("--frames", &mut args).unwrap());
    }

    #[test]
    fn reports_a_missing_or_unreadable_value() {
        let mut common = Common::default();
        assert!(common.accept("--extent", &mut args(&[])).is_err());
        assert!(common.accept("--extent", &mut args(&["wide"])).is_err());
        assert!(common.accept("--size", &mut args(&["1600"])).is_err());
        assert!(common.accept("--mode", &mut args(&["sideways"])).is_err());
    }

    #[test]
    fn a_scene_argument_is_a_preset_name_or_a_path() {
        assert_eq!(SceneSource::parse("cavity"), SceneSource::Cavity);
        assert_eq!(
            SceneSource::parse("scenes/double-slit.ron"),
            SceneSource::File(PathBuf::from("scenes/double-slit.ron"))
        );
    }

    #[test]
    fn resolution_rediscretizes_rather_than_resizing() {
        let mut common = Common {
            extent: 48,
            ..Default::default()
        };
        let coarse = common.scene().unwrap();
        common.resolution = Some(2.0 * coarse.grid.resolution());
        let fine = common.scene().unwrap();

        assert_eq!(fine.grid.extent.x, 2 * coarse.grid.extent.x);
        // Same physical box, twice the cells.
        for axis in 0..3 {
            let (a, b) = (coarse.grid.size()[axis], fine.grid.size()[axis]);
            assert!((a - b).abs() < 1e-6, "axis {axis}: {a} vs {b}");
        }
    }
}

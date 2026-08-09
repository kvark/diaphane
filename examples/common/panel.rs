//! The control panel: what the viewer can be told to do, and what it reports.
//!
//! Kept apart from the event loop because a UI that mutates the simulation
//! while it is being drawn is a borrow tangle and, worse, a place where "the
//! slider moved" and "the solver stepped" can interleave differently depending
//! on which widget was touched. So the panel is a pure function of a
//! [`Readout`] into a [`Commands`]: it renders numbers it was handed and
//! records intent it does not act on. The caller applies the intent afterwards,
//! in one place, in a fixed order.

use crate::common::render::{ViewMode, ViewSettings};

/// One range for a slider and its keyboard shortcut, so neither can set a
/// value the other would silently snap back into its own bounds.
pub const STEPS_PER_FRAME: std::ops::RangeInclusive<u32> = 1..=4096;
pub const GAIN: std::ops::RangeInclusive<f32> = 1e-3..=1e4;

/// What the panel displays. Sampled once per frame, before any of it moves.
pub struct Readout {
    pub step: u64,
    pub furthest: u64,
    /// First step the keyframe window can reach instantly.
    pub window_start: u64,
    pub time: f32,
    pub steps_per_second: f32,
    pub gigabytes_per_second: f64,
    pub frame_milliseconds: f32,
    pub keyframes: usize,
    pub keyframe_megabytes: f64,
    pub running: bool,
    pub steps_per_frame: u32,
}

/// What the panel asks for. Nothing here happens until the caller applies it.
#[derive(Default)]
pub struct Commands {
    pub toggle_running: bool,
    pub reset: bool,
    /// Steps to move, signed. Backwards goes through the timeline, since the
    /// GPU solver has no reverse kernel.
    pub step_by: i64,
    /// Absolute step to seek to, from the scrub slider.
    pub seek: Option<u64>,
    pub steps_per_frame: Option<u32>,
}

/// Builds the panel. Returns what it was asked to do.
pub fn draw(root: &mut egui::Ui, readout: &Readout, settings: &mut ViewSettings) -> Commands {
    let mut commands = Commands::default();
    egui::Panel::bottom("controls")
        .exact_size(112.0)
        .show_inside(root, |ui| {
            ui.add_space(4.0);
            transport(ui, readout, &mut commands);
            ui.add_space(2.0);
            scrubber(ui, readout, &mut commands);
            ui.add_space(2.0);
            appearance(ui, settings);
            ui.separator();
            statistics(ui, readout);
        });
    commands
}

fn transport(ui: &mut egui::Ui, readout: &Readout, commands: &mut Commands) {
    ui.horizontal(|ui| {
        if ui
            .button("⏮")
            .on_hover_text("back to the start and clear the fields")
            .clicked()
        {
            commands.reset = true;
        }
        // Stepping backwards is a seek, not an inverse update: the GPU solver
        // only runs forwards. The timeline restores the nearest earlier
        // keyframe and replays, so one step back can cost up to a keyframe
        // interval of replay -- correct always, instant only inside the window.
        if ui
            .button("◀")
            .on_hover_text("one step back — replays from the nearest keyframe")
            .clicked()
        {
            commands.step_by = -1;
        }
        let label = if readout.running { "⏸" } else { "▶" };
        if ui.button(label).on_hover_text("space").clicked() {
            commands.toggle_running = true;
        }
        if ui.button("▶|").on_hover_text("one step forward").clicked() {
            commands.step_by = 1;
        }

        ui.separator();
        let mut per_frame = readout.steps_per_frame;
        if ui
            .add(
                egui::Slider::new(&mut per_frame, STEPS_PER_FRAME)
                    .logarithmic(true)
                    .text("steps / frame"),
            )
            .changed()
        {
            commands.steps_per_frame = Some(per_frame);
        }
    });
}

fn scrubber(ui: &mut egui::Ui, readout: &Readout, commands: &mut Commands) {
    ui.horizontal(|ui| {
        let furthest = readout.furthest.max(1);
        let mut target = readout.step;
        let slider = ui.add(
            egui::Slider::new(&mut target, 0..=furthest)
                .text("step")
                .clamping(egui::SliderClamping::Always),
        );
        if slider.changed() {
            commands.seek = Some(target);
        }
        // Which part of the run is cheap to reach. Outside it a seek replays
        // from zero, which is slower and always correct, and saying so is what
        // makes a long drag's pause explicable rather than alarming.
        let covered = 100.0 * (furthest - readout.window_start) as f32 / furthest as f32;
        ui.label(
            egui::RichText::new(format!("keyframed: last {covered:.0}%"))
                .weak()
                .small(),
        )
        .on_hover_text(
            "dragging inside this span restores a keyframe; outside it, the run \
             replays from the start",
        );
    });
}

fn appearance(ui: &mut egui::Ui, settings: &mut ViewSettings) {
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("mode")
            .selected_text(settings.mode.label())
            .show_ui(ui, |ui| {
                for mode in ViewMode::ALL {
                    ui.selectable_value(&mut settings.mode, mode, mode.label());
                }
            });
        ui.add(
            egui::Slider::new(&mut settings.gain, GAIN)
                .logarithmic(true)
                .text("gain"),
        );
        let mut log = settings.log_strength > 0.0;
        if ui
            .checkbox(&mut log, "signed log")
            .on_hover_text("compresses the range so a weak tail stays visible next to a peak")
            .changed()
        {
            settings.toggle_log();
        }
    });
}

fn statistics(ui: &mut egui::Ui, readout: &Readout) {
    // In the panel rather than the title bar, because these are the numbers
    // that say whether what you are looking at is worth trusting, and a title
    // bar is where numbers go to be ignored.
    ui.horizontal(|ui| {
        let stat = |ui: &mut egui::Ui, text: String| {
            ui.label(egui::RichText::new(text).monospace().small());
        };
        stat(ui, format!("{:>6.0} steps/s", readout.steps_per_second));
        stat(ui, format!("{:>5.1} GB/s", readout.gigabytes_per_second));
        stat(ui, format!("{:>5.1} ms", readout.frame_milliseconds));
        stat(ui, format!("t = {:>7.2} ns", readout.time * 1e9));
        stat(
            ui,
            format!(
                "{} keyframes / {:.0} MB",
                readout.keyframes, readout.keyframe_megabytes
            ),
        );
    });
}

//! Scrubbing back and forth in time.
//!
//! # The mechanism is determinism, not reversal
//!
//! The state of a simulation here is a pure function of `(scene, step_count)`:
//! fixed initial conditions, sources that are analytic functions of time, no
//! randomness anywhere. So "go to step *k*" never needs a recording — reset and
//! re-run. That alone is a working seek, and it costs `k / rate` seconds, which
//! at a few thousand steps per second is fine for a jump and hopeless for a
//! drag.
//!
//! Keyframes bound that cost. Snapshot the fields every so often, and seeking
//! backwards becomes "restore the nearest earlier keyframe, replay the
//! remainder". [`Timeline`] is that ring buffer.
//!
//! [`crate::cpu::Simulation::reverse`] is a different tool for a different job.
//! It is exact and needs no memory at all, but it only works in a lossless box
//! and it costs one step per step, so it is the right answer for "nudge back a
//! frame" and the wrong one for a slider.
//!
//! # What it costs
//!
//! A keyframe is both fields in full: `6 × cells × 4` bytes, so **24 bytes per
//! cell**. At 128³ that is 50 MB each. Keyframe spacing sets the worst-case
//! scrub latency — allowing ~100 ms of replay at 3000 steps/s means one every
//! ~300 steps — and the product of the two is the memory bill. A long run
//! therefore gets a *window*, not a complete history: recent steps scrub
//! instantly, older ones fall back to replaying from the start, which is always
//! available and never wrong.
//!
//! # Live editing would break this
//!
//! All of it rests on the state depending only on the step number. Mutating
//! geometry while the solver runs makes it depend on the *history of edits*
//! instead, and the slider's premise is gone. The brief's own answer — commit
//! the geometry and reset the fields — generalizes: an edit starts a new take
//! with its own `t = 0`, and a timeline scrubs within one take.

use std::collections::VecDeque;

/// A complete field state, and the step it belongs to.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub step: u64,
    /// All three components of `E`, laid out one after another.
    pub electric: Vec<f32>,
    /// All three components of `H`.
    pub magnetic: Vec<f32>,
}

impl Snapshot {
    pub fn bytes(&self) -> usize {
        size_of_val(self.electric.as_slice()) + size_of_val(self.magnetic.as_slice())
    }
}

/// What a [`Timeline`] needs from a solver.
///
/// Implemented by both [`crate::cpu::Simulation`] and
/// [`crate::gpu::Simulation`], so a timeline works over either. On the GPU a
/// snapshot is a readback and a restore is an upload; both are expensive, which
/// is exactly why they happen on keyframe boundaries rather than every step.
pub trait Steppable {
    fn step_count(&self) -> u64;
    fn advance_by(&mut self, steps: u64);
    fn reset(&mut self);
    /// Takes `&mut` because on the GPU this is a readback: it has to wait for
    /// the last submission and drive the encoder.
    fn snapshot(&mut self) -> Snapshot;
    /// Restores a state captured by [`Self::snapshot`] from the same scene.
    fn restore(&mut self, snapshot: &Snapshot);
}

/// A ring of keyframes covering the recent past.
#[derive(Clone, Debug)]
pub struct Timeline {
    keyframes: VecDeque<Snapshot>,
    interval: u64,
    capacity: usize,
}

/// What a [`Timeline::seek`] actually had to do, for a HUD to report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Seek {
    /// Already there.
    Unchanged,
    /// Stepped forward from where it was.
    Forward { replayed: u64 },
    /// Restored a keyframe, then replayed the remainder.
    FromKeyframe { at: u64, replayed: u64 },
    /// Fell outside the window, so it restarted from an empty domain.
    FromStart { replayed: u64 },
}

impl Seek {
    /// Steps the solver had to run, which is what the latency is proportional
    /// to.
    pub fn replayed(&self) -> u64 {
        match *self {
            Self::Unchanged => 0,
            Self::Forward { replayed }
            | Self::FromKeyframe { replayed, .. }
            | Self::FromStart { replayed } => replayed,
        }
    }
}

impl Timeline {
    /// Keeps `capacity` keyframes spaced `interval` steps apart.
    ///
    /// `interval` trades scrub latency against memory: the worst replay is
    /// `interval` steps, and the bill is `capacity` full field states.
    pub fn new(interval: u64, capacity: usize) -> Self {
        assert!(interval > 0, "keyframe interval must be positive");
        assert!(
            capacity > 0,
            "a timeline needs room for at least one keyframe"
        );
        Self {
            keyframes: VecDeque::with_capacity(capacity),
            interval,
            capacity,
        }
    }

    /// Records a keyframe if one is due. Call after advancing.
    ///
    /// Cheap when nothing is due, which is almost always.
    pub fn observe(&mut self, simulation: &mut impl Steppable) {
        let step = simulation.step_count();
        if !step.is_multiple_of(self.interval) {
            return;
        }
        if self.keyframes.back().is_some_and(|last| last.step == step) {
            return;
        }
        if self.keyframes.len() == self.capacity {
            self.keyframes.pop_front();
        }
        self.keyframes.push_back(simulation.snapshot());
    }

    /// Throws away every keyframe, for when the scene underneath has changed.
    pub fn clear(&mut self) {
        self.keyframes.clear();
    }

    /// Oldest step that can be reached without replaying from zero.
    pub fn earliest(&self) -> Option<u64> {
        self.keyframes.front().map(|frame| frame.step)
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Memory the keyframes currently occupy.
    pub fn bytes(&self) -> usize {
        self.keyframes.iter().map(Snapshot::bytes).sum()
    }

    /// Moves the simulation to `target`, by the cheapest route available.
    ///
    /// Forward is just stepping. Backward restores the newest keyframe at or
    /// before the target and replays; if none exists, it resets and replays
    /// from the start, which is slow but always correct.
    pub fn seek(&self, simulation: &mut impl Steppable, target: u64) -> Seek {
        let current = simulation.step_count();
        if target == current {
            return Seek::Unchanged;
        }
        if target > current {
            simulation.advance_by(target - current);
            return Seek::Forward {
                replayed: target - current,
            };
        }

        match self
            .keyframes
            .iter()
            .rev()
            .find(|frame| frame.step <= target)
        {
            Some(frame) => {
                simulation.restore(frame);
                simulation.advance_by(target - frame.step);
                Seek::FromKeyframe {
                    at: frame.step,
                    replayed: target - frame.step,
                }
            }
            None => {
                simulation.reset();
                simulation.advance_by(target);
                Seek::FromStart { replayed: target }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Seek, Snapshot, Steppable, Timeline};
    use crate::{Axis, Boundary, Extent, Scene, cpu};

    fn scene() -> Scene {
        Scene::photon(Extent::cube(24))
    }

    fn fingerprint(simulation: &cpu::Simulation) -> Vec<f32> {
        Axis::ALL
            .iter()
            .flat_map(|&axis| simulation.electric(axis).iter().copied())
            .collect()
    }

    #[test]
    fn a_snapshot_restores_the_exact_state() {
        let mut simulation = cpu::Simulation::new(&scene());
        simulation.advance_by(60);
        let snapshot = simulation.snapshot();
        let expected = fingerprint(&simulation);

        simulation.advance_by(40);
        assert_ne!(fingerprint(&simulation), expected);

        simulation.restore(&snapshot);
        assert_eq!(simulation.step_count(), 60);
        assert_eq!(fingerprint(&simulation), expected);
    }

    #[test]
    fn seeking_backwards_reproduces_the_forward_run_exactly() {
        // The claim the whole slider rests on: state is a function of the step
        // number alone, so arriving at a step by replay is indistinguishable
        // from having stepped there.
        let mut reference = cpu::Simulation::new(&scene());
        reference.advance_by(90);
        let expected = fingerprint(&reference);

        let mut simulation = cpu::Simulation::new(&scene());
        let mut timeline = Timeline::new(25, 8);
        for _ in 0..150 {
            simulation.advance_by(1);
            timeline.observe(&mut simulation);
        }
        let outcome = timeline.seek(&mut simulation, 90);

        assert_eq!(simulation.step_count(), 90);
        assert_eq!(fingerprint(&simulation), expected);
        // 90 is 15 steps past the keyframe at 75.
        assert_eq!(
            outcome,
            Seek::FromKeyframe {
                at: 75,
                replayed: 15
            }
        );
    }

    #[test]
    fn a_target_older_than_the_window_replays_from_the_start() {
        let mut simulation = cpu::Simulation::new(&scene());
        // Room for two keyframes only, so the early history falls out.
        let mut timeline = Timeline::new(20, 2);
        for _ in 0..120 {
            simulation.advance_by(1);
            timeline.observe(&mut simulation);
        }
        assert_eq!(timeline.keyframe_count(), 2);
        assert_eq!(timeline.earliest(), Some(100));

        let mut reference = cpu::Simulation::new(&scene());
        reference.advance_by(10);
        let outcome = timeline.seek(&mut simulation, 10);

        assert_eq!(outcome, Seek::FromStart { replayed: 10 });
        assert_eq!(fingerprint(&simulation), fingerprint(&reference));
    }

    #[test]
    fn seeking_forward_just_steps() {
        let mut simulation = cpu::Simulation::new(&scene());
        let timeline = Timeline::new(10, 4);
        simulation.advance_by(20);
        assert_eq!(
            timeline.seek(&mut simulation, 35),
            Seek::Forward { replayed: 15 }
        );
        assert_eq!(simulation.step_count(), 35);
        assert_eq!(timeline.seek(&mut simulation, 35), Seek::Unchanged);
    }

    #[test]
    fn the_window_slides_and_the_memory_bill_is_bounded() {
        let mut simulation = cpu::Simulation::new(&scene());
        let mut timeline = Timeline::new(10, 4);
        for _ in 0..200 {
            simulation.advance_by(1);
            timeline.observe(&mut simulation);
        }
        assert_eq!(timeline.keyframe_count(), 4);
        assert_eq!(timeline.earliest(), Some(170));

        // 24 bytes per cell per keyframe, and no more than `capacity` of them
        // however long the run goes on.
        let per_frame = 24 * Extent::cube(24).total();
        assert_eq!(timeline.bytes(), 4 * per_frame);
    }

    #[test]
    fn observing_twice_at_the_same_step_records_one_keyframe() {
        let mut simulation = cpu::Simulation::new(&scene());
        let mut timeline = Timeline::new(10, 8);
        simulation.advance_by(10);
        timeline.observe(&mut simulation);
        timeline.observe(&mut simulation);
        assert_eq!(timeline.keyframe_count(), 1);
    }

    #[test]
    fn a_snapshot_reports_its_own_size() {
        let mut simulation = cpu::Simulation::new(&scene());
        let snapshot: Snapshot = simulation.snapshot();
        assert_eq!(snapshot.bytes(), 24 * Extent::cube(24).total());
    }

    #[test]
    fn scrubbing_a_lossless_scene_agrees_with_stepping_backwards() {
        // Two independent routes to the same state: replay from a keyframe,
        // and the exact inverse of the update. They should land in the same
        // place, which checks each against the other.
        let scene = Scene::cavity(Extent::cube(24)).with_boundary(Boundary::Pec);
        let mut scrubbed = cpu::Simulation::new(&scene);
        let mut timeline = Timeline::new(20, 8);
        for _ in 0..100 {
            scrubbed.advance_by(1);
            timeline.observe(&mut scrubbed);
        }
        let mut reversed = cpu::Simulation::new(&scene);
        reversed.advance_by(100);
        reversed.reverse_by(30);

        timeline.seek(&mut scrubbed, 70);
        assert_eq!(scrubbed.step_count(), reversed.step_count());

        let peak = fingerprint(&scrubbed)
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let worst = fingerprint(&scrubbed)
            .iter()
            .zip(fingerprint(&reversed).iter())
            .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
        assert!(
            worst < 1e-4 * peak,
            "replay and reversal disagree by {:e} of the peak",
            worst / peak
        );
    }
}

//! Sources: what shakes the fields, and how.
//!
//! All sources here are **soft**, meaning they add to the field rather than
//! overwrite it:
//!
//! ```text
//! E[component] += amplitude · waveform(t) · weight(cell)
//! ```
//!
//! A hard source that assigns `E` instead is also a perfect reflector, because a
//! cell whose value is dictated cannot respond to a wave arriving at it. Scattered
//! light returning to the source would bounce off it and the domain would slowly
//! fill with spurious echoes. Soft sources are transparent to everything except
//! the energy they inject.
//!
//! # Waveform choice matters more than it looks
//!
//! Every waveform here is deliberately zero-mean and smoothly switched on. A pulse
//! with DC content deposits a static field that never radiates away and sits in
//! the domain contaminating every later measurement; a sinusoid switched on
//! abruptly is a step discontinuity whose spectrum reaches to the grid's Nyquist
//! limit, where the numerical dispersion is worst, so it injects visible garbage.
//! Both failures look like solver bugs.

use crate::grid::{Axis, Grid};
use std::f32;

/// The time profile of a source.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum Waveform {
    /// The second derivative of a Gaussian, `(1 − 2π²f²τ²)·exp(−π²f²τ²)`.
    ///
    /// The default choice: broadband, compact in time, and exactly zero-mean,
    /// so it leaves nothing behind after it passes.
    Ricker {
        peak_frequency: f32,
        /// Time of the peak. Before it the waveform ramps up from zero; make
        /// it large enough that the switch-on is negligible.
        delay: f32,
    },
    /// A sinusoid under a Gaussian envelope — narrowband, for looking at one
    /// frequency at a time.
    GaussianPulse {
        center_frequency: f32,
        /// Envelope width at half maximum, in periods of the carrier.
        duration_cycles: f32,
        delay: f32,
    },
    /// A steady sinusoid, raised smoothly from zero over `ramp_cycles`.
    ContinuousWave { frequency: f32, ramp_cycles: f32 },
}

impl Waveform {
    /// A Ricker wavelet with a delay long enough that the switch-on transient
    /// is below `1e-6` of the peak.
    pub fn ricker(peak_frequency: f32) -> Self {
        Self::Ricker {
            peak_frequency,
            delay: 1.4 / peak_frequency,
        }
    }

    /// A Gaussian pulse with the delay chosen the same way.
    pub fn gaussian_pulse(center_frequency: f32, duration_cycles: f32) -> Self {
        Self::GaussianPulse {
            center_frequency,
            duration_cycles,
            delay: 2.0 * duration_cycles / center_frequency,
        }
    }

    /// Value at time `t`, normalized so the peak magnitude is about 1.
    pub fn evaluate(&self, time: f32) -> f32 {
        match *self {
            Self::Ricker {
                peak_frequency,
                delay,
            } => {
                let arg = f32::consts::PI * peak_frequency * (time - delay);
                let arg2 = arg * arg;
                (1.0 - 2.0 * arg2) * (-arg2).exp()
            }
            Self::GaussianPulse {
                center_frequency,
                duration_cycles,
                delay,
            } => {
                // FWHM of the envelope is `duration_cycles` periods, so
                // σ = FWHM / (2√(2 ln 2)).
                const FWHM_TO_SIGMA: f32 = 0.424_660_9;
                let sigma = FWHM_TO_SIGMA * duration_cycles / center_frequency;
                let tau = time - delay;
                let envelope = (-0.5 * (tau / sigma) * (tau / sigma)).exp();
                envelope * (2.0 * f32::consts::PI * center_frequency * tau).sin()
            }
            Self::ContinuousWave {
                frequency,
                ramp_cycles,
            } => {
                if time <= 0.0 {
                    return 0.0;
                }
                let ramp_duration = ramp_cycles / frequency;
                // Raised cosine: value and first derivative are both continuous
                // at each end, so the spectrum stays where it belongs.
                let envelope = if time >= ramp_duration {
                    1.0
                } else {
                    0.5 * (1.0 - (f32::consts::PI * time / ramp_duration).cos())
                };
                envelope * (2.0 * f32::consts::PI * frequency * time).sin()
            }
        }
    }

    /// The frequency the source puts the most energy at.
    pub fn dominant_frequency(&self) -> f32 {
        match *self {
            Self::Ricker { peak_frequency, .. } => peak_frequency,
            Self::GaussianPulse {
                center_frequency, ..
            } => center_frequency,
            Self::ContinuousWave { frequency, .. } => frequency,
        }
    }
}

/// Where a source deposits its energy.
///
/// Positions are in metres from the centre of the domain, like [`crate::Shape`]
/// and for the same reason: a source pinned to a cell index moves when the
/// resolution changes.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum SourceShape {
    /// A single Yee cell: an oscillating point dipole, which radiates the
    /// familiar toroidal pattern with a null along its own axis.
    ///
    /// A nonzero `velocity` slides it through the grid, hopping cell to cell,
    /// and it goes *silent* once its true position leaves the domain — a
    /// clamped mover would keep pouring energy into whatever wall cell it
    /// stuck to. Slower than the local phase velocity the wake is a Doppler
    /// pattern; faster than `c/n` inside a dielectric it closes into a Mach
    /// cone, which is Cherenkov radiation with nothing exotic involved.
    Point {
        at: [f32; 3],
        /// Metres per second. Zero — the serde default, so older scene files
        /// still parse — stands still.
        #[cfg_attr(feature = "serde", serde(default))]
        velocity: [f32; 3],
    },
    /// A planar sheet normal to `axis`, at `offset` metres along it, centred
    /// on the domain transversely and apodized by a Gaussian of the given
    /// waist in metres.
    ///
    /// A waist comparable to the domain approximates a plane wave; a small one
    /// launches a diverging beam. The sheet is soft, so it radiates in *both*
    /// directions along `axis` — total-field/scattered-field, which would make
    /// it one-way, is not implemented.
    Sheet { axis: Axis, offset: f32, waist: f32 },
}

/// A source: a shape, a direction, and a time profile.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Source {
    pub shape: SourceShape,
    /// Which component of `E` is driven. For a [`SourceShape::Sheet`] this
    /// should be transverse to the sheet's axis, or the source drives a
    /// longitudinal field that cannot propagate.
    pub polarization: Axis,
    pub waveform: Waveform,
    pub amplitude: f32,
}

impl Source {
    /// A point dipole at a position in metres.
    pub fn point(at: [f32; 3], polarization: Axis, waveform: Waveform) -> Self {
        Self::point_moving(at, [0.0; 3], polarization, waveform)
    }

    /// A point dipole that starts `at` and moves at `velocity` metres per
    /// second — see [`SourceShape::Point`] for what motion means here.
    pub fn point_moving(
        at: [f32; 3],
        velocity: [f32; 3],
        polarization: Axis,
        waveform: Waveform,
    ) -> Self {
        Self {
            shape: SourceShape::Point { at, velocity },
            polarization,
            waveform,
            amplitude: 1.0,
        }
    }

    /// An apodized sheet, which is the shape that launches something looking
    /// like a propagating wave packet rather than a spherical wave.
    ///
    /// `offset` and `waist` are metres.
    pub fn sheet(
        axis: Axis,
        offset: f32,
        waist: f32,
        polarization: Axis,
        waveform: Waveform,
    ) -> Self {
        Self {
            shape: SourceShape::Sheet {
                axis,
                offset,
                waist,
            },
            polarization,
            waveform,
            amplitude: 1.0,
        }
    }

    pub fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude;
        self
    }

    /// Where this source sits at `t = 0`, in metres. A sheet reports the
    /// centre of the plane it occupies; a moving point reports where it
    /// starts, which is the position validation checks.
    pub fn position(&self) -> [f32; 3] {
        match self.shape {
            SourceShape::Point { at, .. } => at,
            SourceShape::Sheet { axis, offset, .. } => {
                let mut position = [0.0; 3];
                position[axis.index()] = offset;
                position
            }
        }
    }

    /// Resolves the source against a grid at one instant.
    ///
    /// Both solvers consume this: it reduces every shape to a box of cells and
    /// a weight function, so there is exactly one injection kernel rather than
    /// one per shape.
    pub fn injection(&self, grid: &Grid, time: f32) -> Injection {
        let extent = grid.extent.as_array();
        let value = self.amplitude * self.waveform.evaluate(time);
        match self.shape {
            SourceShape::Point { at, velocity } => {
                let position = [
                    at[0] + velocity[0] * time,
                    at[1] + velocity[1] * time,
                    at[2] + velocity[2] * time,
                ];
                Injection {
                    origin: grid.cell_containing(position),
                    extent: [1, 1, 1],
                    center: [0.0; 3],
                    inverse_waist_squared: 0.0,
                    component: self.polarization.index(),
                    // Gone means silent. The position is a pure function of
                    // time, so the retraction that time reversal performs
                    // silences at exactly the same instant.
                    value: if grid.contains(position) { value } else { 0.0 },
                }
            }
            SourceShape::Sheet {
                axis,
                offset,
                waist,
            } => {
                let a = axis.index();
                let mut position = [0.0; 3];
                position[a] = offset;

                let mut origin = [0, 0, 0];
                let mut region = extent;
                origin[a] = grid.cell_containing(position)[a];
                region[a] = 1;

                // Both the centre and the waist stay in metres, so the taper is
                // the same shape whatever the cells under it are doing. Stated
                // in cells it would narrow wherever the grid was refined --
                // which is precisely where somebody put a refinement because
                // they cared what the field was doing.
                Injection {
                    origin,
                    extent: region,
                    center: [0.0; 3],
                    inverse_waist_squared: 1.0 / (waist * waist),
                    component: self.polarization.index(),
                    value,
                }
            }
        }
    }
}

/// A source flattened into something a kernel can execute: a box of cells, a
/// component, an amplitude, and an apodization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Injection {
    /// First cell of the affected region.
    pub origin: [usize; 3],
    /// Size of the affected region in cells.
    pub extent: [usize; 3],
    /// Centre of the Gaussian apodization, in metres from the domain centre.
    pub center: [f32; 3],
    /// `1/waist²` in 1/m². Zero means no apodization — every cell gets full
    /// weight.
    pub inverse_waist_squared: f32,
    /// Index of the driven `E` component.
    pub component: usize,
    /// `amplitude · waveform(t)`, evaluated on the host so the kernel never
    /// needs to know what a Ricker wavelet is.
    pub value: f32,
}

impl Injection {
    /// Number of cells this injection touches.
    pub fn cell_count(&self) -> usize {
        self.extent[0] * self.extent[1] * self.extent[2]
    }

    /// Apodization weight at a physical position, in metres.
    pub fn weight(&self, position: [f32; 3]) -> f32 {
        if self.inverse_waist_squared == 0.0 {
            return 1.0;
        }
        let mut radius_squared = 0.0;
        for (axis, &coordinate) in position.iter().enumerate() {
            // Only the directions the region actually spans are apodized;
            // the sheet's own normal contributes nothing.
            if self.extent[axis] > 1 {
                let offset = coordinate - self.center[axis];
                radius_squared += offset * offset;
            }
        }
        (-radius_squared * self.inverse_waist_squared).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::{Source, SourceShape, Waveform};
    use crate::grid::{Axis, Extent, Grid};

    fn grid() -> Grid {
        Grid::new(Extent::new(32, 40, 48), 1e-3)
    }

    #[test]
    fn ricker_peaks_at_its_delay_and_is_zero_mean() {
        let frequency = 1e9;
        let waveform = Waveform::ricker(frequency);
        let Waveform::Ricker { delay, .. } = waveform else {
            unreachable!()
        };
        assert!((waveform.evaluate(delay) - 1.0).abs() < 1e-6);
        // Starts from rest: no step at t = 0.
        assert!(waveform.evaluate(0.0).abs() < 1e-5);

        // A DC component would leave a static field behind forever, so this is
        // load-bearing rather than cosmetic.
        let steps = 20_000;
        let span = 4.0 * delay;
        let mean: f32 = (0..steps)
            .map(|i| waveform.evaluate(span * i as f32 / steps as f32))
            .sum::<f32>()
            / steps as f32;
        assert!(mean.abs() < 1e-4, "mean {mean} is not zero");
    }

    #[test]
    fn gaussian_pulse_envelope_has_the_requested_width() {
        let frequency = 1e9;
        let cycles = 10.0;
        let waveform = Waveform::gaussian_pulse(frequency, cycles);
        let Waveform::GaussianPulse { delay, .. } = waveform else {
            unreachable!()
        };
        // Recover the envelope by taking the peak magnitude within one carrier
        // period, which is where |sin| reaches 1.
        let period = 1.0 / frequency;
        let envelope = |periods_from_peak: f32| {
            let center = delay + periods_from_peak * period;
            (0..256)
                .map(|i| {
                    waveform
                        .evaluate(center + (i as f32 / 256.0 - 0.5) * period)
                        .abs()
                })
                .fold(0.0f32, f32::max)
        };
        assert!((envelope(0.0) - 1.0).abs() < 0.02, "{}", envelope(0.0));
        // `duration_cycles` is defined as the full width at half maximum, so
        // half of it out from the peak is where the envelope reaches 0.5.
        let half = envelope(0.5 * cycles);
        assert!(
            (half - 0.5).abs() < 0.05,
            "envelope at half width is {half}"
        );
        assert!(envelope(2.0 * cycles) < 0.01, "pulse does not die out");
        // The delay has to hide the switch-on, or the pulse starts with a step.
        assert!(waveform.evaluate(0.0).abs() < 1e-4);
    }

    #[test]
    fn continuous_wave_ramps_smoothly_from_rest() {
        let waveform = Waveform::ContinuousWave {
            frequency: 1e9,
            ramp_cycles: 3.0,
        };
        assert_eq!(waveform.evaluate(-1e-12), 0.0);
        assert_eq!(waveform.evaluate(0.0), 0.0);
        // Small compared to a hard switch-on, which would already be at full
        // amplitude a quarter period in.
        let quarter = 0.25e-9;
        assert!(waveform.evaluate(quarter).abs() < 0.15);
        // Fully on well past the ramp.
        let late = 10e-9;
        let peak = (0..400)
            .map(|i| waveform.evaluate(late + i as f32 * 1e-12).abs())
            .fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 0.01, "peak {peak}");
    }

    #[test]
    fn point_injection_covers_exactly_one_cell() {
        // 32x40x48 cells at 1 mm: cell (4,5,6) centres at (-11.5, -14.5, -17.5) mm.
        let source = Source::point(
            [-11.5e-3, -14.5e-3, -17.5e-3],
            Axis::Z,
            Waveform::ricker(1e9),
        );
        let injection = source.injection(&grid(), 0.0);
        assert_eq!(injection.cell_count(), 1);
        assert_eq!(injection.origin, [4, 5, 6]);
        assert_eq!(injection.component, 2);
        assert_eq!(injection.weight([-11.5e-3, -14.5e-3, -17.5e-3]), 1.0);
    }

    #[test]
    fn sheet_injection_spans_the_plane_and_apodizes_transversely() {
        let grid = grid();
        // x = -8 mm is cell 8 of 32; a 6 mm waist is 6 cells.
        let source = Source::sheet(Axis::X, -8e-3, 6e-3, Axis::Z, Waveform::ricker(1e9));
        let injection = source.injection(&grid, 0.0);
        assert_eq!(injection.origin, [8, 0, 0]);
        assert_eq!(injection.extent, [1, 40, 48]);

        // Peak weight on the axis of the sheet, falling off transversely. The
        // taper is stated in metres, so these are metres — one waist out is
        // `exp(-1)`.
        let center = [-8e-3, 0.0, 0.0];
        assert!((injection.weight(center) - 1.0).abs() < 1e-6);
        let offset = injection.weight([-8e-3, 6e-3, 0.0]);
        assert!(
            (offset - (-1.0f32).exp()).abs() < 1e-6,
            "one waist out should be exp(-1), got {offset}"
        );
        // The sheet normal must not participate: moving along x is outside the
        // region entirely, and would otherwise damp the whole sheet.
        assert_eq!(injection.weight([0.05, 0.0, 0.0]), injection.weight(center));
    }

    #[test]
    fn injection_value_follows_the_waveform() {
        let grid = grid();
        let waveform = Waveform::ricker(1e9);
        let source = Source::point([0.0; 3], Axis::X, waveform).with_amplitude(3.0);
        let time = 0.7e-9;
        let injection = source.injection(&grid, time);
        assert!((injection.value - 3.0 * waveform.evaluate(time)).abs() < 1e-6);
    }

    #[test]
    fn a_moving_point_tracks_its_velocity() {
        // Half of c along +x from the centre of a 32-cell axis: 15 mm in
        // 100 ps, which is cell 16 to cell 31.
        let source =
            Source::point_moving([0.0; 3], [1.5e8, 0.0, 0.0], Axis::Z, Waveform::ricker(1e9));
        assert_eq!(source.injection(&grid(), 0.0).origin, [16, 20, 24]);
        assert_eq!(source.injection(&grid(), 1e-10).origin, [31, 20, 24]);
    }

    #[test]
    fn a_moving_point_goes_silent_when_it_leaves() {
        // At the Ricker's own peak time the mover is far outside the domain,
        // while a static twin is at full drive -- so the zero is the
        // position's doing, not the waveform's.
        let waveform = Waveform::ricker(1e9);
        let Waveform::Ricker { delay, .. } = waveform else {
            unreachable!()
        };
        let mover = Source::point_moving([0.0; 3], [1.5e8, 0.0, 0.0], Axis::Z, waveform);
        let sitter = Source::point([0.0; 3], Axis::Z, waveform);
        assert_eq!(mover.injection(&grid(), delay).value, 0.0);
        assert!(sitter.injection(&grid(), delay).value.abs() > 0.99);
    }

    #[test]
    fn sheet_clamps_a_position_past_the_far_wall() {
        let grid = grid();
        let source = Source::sheet(Axis::Y, 9.9, 4e-3, Axis::X, Waveform::ricker(1e9));
        let injection = source.injection(&grid, 0.0);
        assert_eq!(injection.origin[1], 39);
        assert!(matches!(source.shape, SourceShape::Sheet { .. }));
    }
}

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
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum SourceShape {
    /// A single Yee cell: an oscillating point dipole, which radiates the
    /// familiar toroidal pattern with a null along its own axis.
    Point { at: [u32; 3] },
    /// A planar sheet normal to `axis`, centred on the domain and apodized by
    /// a Gaussian of the given waist in cells.
    ///
    /// A waist comparable to the domain approximates a plane wave; a small one
    /// launches a diverging beam. The sheet is soft, so it radiates in *both*
    /// directions along `axis` — total-field/scattered-field, which would make
    /// it one-way, is not implemented.
    Sheet {
        axis: Axis,
        position: u32,
        waist: f32,
    },
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
    /// A point dipole at a cell.
    pub fn point(at: [u32; 3], polarization: Axis, waveform: Waveform) -> Self {
        Self {
            shape: SourceShape::Point { at },
            polarization,
            waveform,
            amplitude: 1.0,
        }
    }

    /// An apodized sheet, which is the shape that launches something looking
    /// like a propagating wave packet rather than a spherical wave.
    pub fn sheet(
        axis: Axis,
        position: u32,
        waist: f32,
        polarization: Axis,
        waveform: Waveform,
    ) -> Self {
        Self {
            shape: SourceShape::Sheet {
                axis,
                position,
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

    /// Resolves the source against a grid at one instant.
    ///
    /// Both solvers consume this: it reduces every shape to a box of cells and
    /// a weight function, so there is exactly one injection kernel rather than
    /// one per shape.
    pub fn injection(&self, grid: &Grid, time: f32) -> Injection {
        let extent = grid.extent.as_array();
        let value = self.amplitude * self.waveform.evaluate(time);
        match self.shape {
            SourceShape::Point { at } => Injection {
                origin: [at[0] as usize, at[1] as usize, at[2] as usize],
                extent: [1, 1, 1],
                center: [0.0; 3],
                inverse_waist_squared: 0.0,
                component: self.polarization.index(),
                value,
            },
            SourceShape::Sheet {
                axis,
                position,
                waist,
            } => {
                let a = axis.index();
                let mut origin = [0, 0, 0];
                let mut region = extent;
                origin[a] = (position as usize).min(extent[a] - 1);
                region[a] = 1;
                let center = [
                    0.5 * extent[0] as f32,
                    0.5 * extent[1] as f32,
                    0.5 * extent[2] as f32,
                ];
                Injection {
                    origin,
                    extent: region,
                    center,
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
    /// Centre of the Gaussian apodization, in cells.
    pub center: [f32; 3],
    /// `1/waist²`. Zero means no apodization — every cell gets full weight.
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

    /// Apodization weight at an absolute cell coordinate.
    pub fn weight(&self, coord: [usize; 3]) -> f32 {
        if self.inverse_waist_squared == 0.0 {
            return 1.0;
        }
        let mut radius_squared = 0.0;
        for (axis, &position) in coord.iter().enumerate() {
            // Only the directions the region actually spans are apodized;
            // the sheet's own normal contributes nothing.
            if self.extent[axis] > 1 {
                let offset = position as f32 - self.center[axis];
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
        let source = Source::point([4, 5, 6], Axis::Z, Waveform::ricker(1e9));
        let injection = source.injection(&grid(), 0.0);
        assert_eq!(injection.cell_count(), 1);
        assert_eq!(injection.origin, [4, 5, 6]);
        assert_eq!(injection.component, 2);
        assert_eq!(injection.weight([4, 5, 6]), 1.0);
    }

    #[test]
    fn sheet_injection_spans_the_plane_and_apodizes_transversely() {
        let grid = grid();
        let source = Source::sheet(Axis::X, 8, 6.0, Axis::Z, Waveform::ricker(1e9));
        let injection = source.injection(&grid, 0.0);
        assert_eq!(injection.origin, [8, 0, 0]);
        assert_eq!(injection.extent, [1, 40, 48]);

        // Peak weight at the centre of the sheet, falling off transversely.
        let center = [8, 20, 24];
        assert!((injection.weight(center) - 1.0).abs() < 1e-6);
        let offset = injection.weight([8, 26, 24]);
        assert!(offset < 0.4 && offset > 0.3, "weight {offset}");
        // The sheet normal must not participate: moving along x is outside the
        // region entirely, and would otherwise damp the whole sheet.
        assert_eq!(injection.weight([99, 20, 24]), injection.weight(center));
    }

    #[test]
    fn injection_value_follows_the_waveform() {
        let grid = grid();
        let waveform = Waveform::ricker(1e9);
        let source = Source::point([1, 1, 1], Axis::X, waveform).with_amplitude(3.0);
        let time = 0.7e-9;
        let injection = source.injection(&grid, time);
        assert!((injection.value - 3.0 * waveform.evaluate(time)).abs() < 1e-6);
    }

    #[test]
    fn sheet_clamps_a_position_past_the_far_wall() {
        let grid = grid();
        let source = Source::sheet(Axis::Y, 999, 4.0, Axis::X, Waveform::ricker(1e9));
        let injection = source.injection(&grid, 0.0);
        assert_eq!(injection.origin[1], 39);
        assert!(matches!(source.shape, SourceShape::Sheet { .. }));
    }
}

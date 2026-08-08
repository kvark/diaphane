//! What happens at the edge of the domain.
//!
//! # Two options
//!
//! [`Boundary::Pec`] costs nothing and is not really a feature: clamping the
//! stencil to the array bounds leaves the outermost tangential `E` samples at
//! their initial zero forever, which *is* a perfect electric conductor. A closed
//! PEC box is lossless, so it is the reference case for the energy-conservation
//! test.
//!
//! [`Boundary::Absorbing`] is the default. It is a graded, impedance-matched
//! conductive layer: both the electric and magnetic loss rates are set to the same
//! value, which in the continuum is the condition for a wave to cross the
//! interface without reflecting. Only the discretization reflects, and grading the
//! loss smoothly from zero at the inner edge to a maximum at the wall is what
//! keeps that reflection small.
//!
//! This is not CPML. CPML additionally stretches the coordinate into the complex
//! plane, which suppresses evanescent and grazing-incidence components that a real
//! conductivity cannot touch; it needs auxiliary convolution fields and it is,
//! per the design brief, where FDTD implementations go to die. The layer here
//! buys most of the benefit for none of the machinery, and the validation suite
//! puts a measured number on what "most" means.
//!
//! # Why the profile is separable
//!
//! A conductivity that ramps in from all six faces is, at an edge or a corner, the
//! *sum* of the contributions from each face that the cell is inside of:
//!
//! ```text
//! r(i, j, k) = rx(i) + ry(j) + rz(k)
//! ```
//!
//! That is exact, needs no special case for edges and corners, and costs three 1D
//! arrays of length `nx`, `ny`, `nz` instead of a full-domain field — a few
//! kilobytes rather than a few tens of megabytes.
//!
//! Each axis carries *two* profiles, sampled at integer and half-integer
//! positions. That is not an optimization, it is a correctness requirement: `E`
//! and `H` are staggered by half a cell, and an impedance-matched layer only
//! matches if the loss damping each field is evaluated where that field lives.
//! Sampling both from one array is the kind of half-cell error that produces a
//! layer which looks like it works and reflects an order of magnitude more than
//! it should.

use crate::grid::{Axis, Grid};

/// The condition applied at all six faces of the domain.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum Boundary {
    /// Perfectly conducting walls. Lossless — energy stays in the box forever.
    Pec,
    /// Graded impedance-matched conductive layer.
    Absorbing {
        /// Layer depth in cells. Ten is the usual working value; fewer than
        /// about six starts to reflect visibly.
        thickness: u32,
        /// Reflection coefficient the grading is designed for, e.g. `1e-6`.
        /// The achieved figure is worse than this — the formula accounts for
        /// the continuum problem, not the discrete one — but it is the knob
        /// that sets how aggressive the layer is.
        target_reflection: f32,
    },
}

impl Boundary {
    /// Ten cells targeting `1e-6`, which is the configuration the validation
    /// suite measures.
    pub const DEFAULT: Self = Self::Absorbing {
        thickness: 10,
        target_reflection: 1e-6,
    };

    /// Polynomial grading order. Cubic is the standard choice: quadratic
    /// reflects more, quartic concentrates too much loss at the wall.
    const GRADING_ORDER: f32 = 3.0;
}

impl Default for Boundary {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Per-axis loss profiles, sampled at both integer and half-integer positions.
///
/// Values are stored as the dimensionless half-step loss `r·Δt/2`, which is
/// what the update equations consume directly.
#[derive(Clone, Debug, PartialEq)]
pub struct AbsorbingProfile {
    integer: [Vec<f32>; 3],
    half: [Vec<f32>; 3],
}

impl AbsorbingProfile {
    /// Builds the profiles for a grid.
    pub fn new(grid: &Grid, boundary: Boundary) -> Self {
        let extent = grid.extent.as_array();
        let (thickness, target) = match boundary {
            Boundary::Pec => (0, 1.0),
            Boundary::Absorbing {
                thickness,
                target_reflection,
            } => (thickness as usize, target_reflection),
        };

        if thickness == 0 {
            return Self {
                integer: extent.map(|n| vec![0.0; n]),
                half: extent.map(|n| vec![0.0; n]),
            };
        }
        for (axis, &n) in extent.iter().enumerate() {
            assert!(
                2 * thickness < n,
                "absorbing layers of {thickness} cells do not fit in axis {axis} of {n} cells",
            );
        }
        assert!(
            target > 0.0 && target < 1.0,
            "target reflection {target} must lie in (0, 1)",
        );

        // σ_max = −(m+1)·ln(R₀) / (2·η₀·d) with d = T·Δ, converted to the
        // dimensionless half-step loss r·Δt/2. The cell size and the speed of
        // light both cancel, leaving a number that depends only on the
        // grading order, the target, the Courant number and the thickness.
        let order = Boundary::GRADING_ORDER;
        let peak = -(order + 1.0) * target.ln() * grid.courant / (4.0 * thickness as f32);
        let profile = |position: f32, n: usize| -> f32 {
            let outer = (n - 1) as f32;
            let depth_low = (thickness as f32 - position) / thickness as f32;
            let depth_high = (thickness as f32 - (outer - position)) / thickness as f32;
            let depth = depth_low.max(depth_high).clamp(0.0, 1.0);
            peak * depth.powf(order)
        };

        Self {
            integer: extent.map(|n| (0..n).map(|i| profile(i as f32, n)).collect()),
            half: extent.map(|n| (0..n).map(|i| profile(i as f32 + 0.5, n)).collect()),
        }
    }

    /// Loss seen by the `axis` component of `E` at a cell.
    ///
    /// `E[a]` sits at a half position along `a` and integer positions along the
    /// other two — see the staggering table in [`crate::grid`].
    ///
    /// Defined in terms of [`Self::electric_row`] rather than alongside it, so
    /// that the two cannot drift apart — not even by the last bit, which they
    /// would if each summed the three profiles in its own order.
    pub fn electric(&self, axis: Axis, coord: [usize; 3]) -> f32 {
        let (varying, constant) = self.electric_row(axis, coord[1], coord[2]);
        varying[coord[0]] + constant
    }

    /// Loss seen by the `axis` component of `H` at a cell — the mirror image
    /// of [`Self::electric`].
    pub fn magnetic(&self, axis: Axis, coord: [usize; 3]) -> f32 {
        let (varying, constant) = self.magnetic_row(axis, coord[1], coord[2]);
        varying[coord[0]] + constant
    }

    /// Splits the [`Self::electric`] lookup into the part that varies along
    /// `x` and the constant remainder for a given `(y, z)` row.
    ///
    /// The solvers sweep `x` innermost, so two of the three profile reads are
    /// loop-invariant. Hoisting them here rather than in the loop keeps the
    /// staggering rule stated once.
    pub fn electric_row(&self, axis: Axis, y: usize, z: usize) -> (&[f32], f32) {
        let a = axis.index();
        let pick = |t: usize| {
            if t == a {
                &self.half[t]
            } else {
                &self.integer[t]
            }
        };
        (pick(0), pick(1)[y] + pick(2)[z])
    }

    /// [`Self::magnetic`] split the same way.
    pub fn magnetic_row(&self, axis: Axis, y: usize, z: usize) -> (&[f32], f32) {
        let a = axis.index();
        let pick = |t: usize| {
            if t == a {
                &self.integer[t]
            } else {
                &self.half[t]
            }
        };
        (pick(0), pick(1)[y] + pick(2)[z])
    }

    /// The profiles laid out for GPU upload: all three integer profiles, then
    /// all three half profiles, each in `x, y, z` order.
    ///
    /// Offsets are recoverable from the grid extent alone, so no side table
    /// needs to travel with the buffer.
    pub fn packed(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(2 * self.integer.iter().map(Vec::len).sum::<usize>());
        for axis in &self.integer {
            out.extend_from_slice(axis);
        }
        for axis in &self.half {
            out.extend_from_slice(axis);
        }
        out
    }

    /// Largest half-step loss anywhere in the layer, useful for reporting.
    pub fn peak(&self) -> f32 {
        self.half
            .iter()
            .chain(self.integer.iter())
            .flat_map(|axis| axis.iter())
            .fold(0.0f32, |acc, &v| acc.max(v))
    }
}

#[cfg(test)]
mod tests {
    use super::{AbsorbingProfile, Boundary};
    use crate::grid::{Axis, Extent, Grid};

    fn grid() -> Grid {
        Grid::new(Extent::new(40, 44, 48), 1e-3)
    }

    #[test]
    fn pec_has_no_loss_anywhere() {
        let profile = AbsorbingProfile::new(&grid(), Boundary::Pec);
        assert_eq!(profile.peak(), 0.0);
        assert_eq!(profile.electric(Axis::X, [0, 0, 0]), 0.0);
        assert_eq!(profile.magnetic(Axis::Z, [39, 43, 47]), 0.0);
    }

    #[test]
    fn interior_is_untouched_and_walls_are_damped() {
        let grid = grid();
        let profile = AbsorbingProfile::new(&grid, Boundary::DEFAULT);
        // Dead centre of the domain: no absorber at all, or the layer would be
        // eating the physics it is supposed to be letting out.
        assert_eq!(profile.electric(Axis::X, [20, 22, 24]), 0.0);
        assert_eq!(profile.magnetic(Axis::Y, [20, 22, 24]), 0.0);
        // Corner cell: three faces contribute, so the loss is the largest in
        // the domain.
        let corner = profile.electric(Axis::X, [0, 0, 0]);
        let face = profile.electric(Axis::X, [0, 22, 24]);
        assert!(corner > face, "corner {corner} should exceed face {face}");
        assert!(face > 0.0);
    }

    #[test]
    fn grading_is_monotone_into_the_wall() {
        let profile = AbsorbingProfile::new(&grid(), Boundary::DEFAULT);
        let mut previous = f32::INFINITY;
        for i in 0..10 {
            let value = profile.electric(Axis::Y, [i, 22, 24]);
            assert!(
                value <= previous,
                "not monotone at {i}: {value} > {previous}"
            );
            previous = value;
        }
        assert_eq!(profile.electric(Axis::Y, [12, 22, 24]), 0.0);
    }

    #[test]
    fn both_walls_are_absorbing() {
        let profile = AbsorbingProfile::new(&grid(), Boundary::DEFAULT);
        let low = profile.magnetic(Axis::Z, [20, 22, 0]);
        let high = profile.magnetic(Axis::Z, [20, 22, 47]);
        assert!(low > 0.0 && high > 0.0, "low {low}, high {high}");
    }

    #[test]
    fn peak_matches_the_closed_form() {
        let grid = grid();
        let profile = AbsorbingProfile::new(&grid, Boundary::DEFAULT);
        // −(m+1)·ln(R₀)·S / (4T) with m = 3, R₀ = 1e-6, S = 0.5, T = 10.
        let expected = -4.0 * 1e-6f32.ln() * 0.5 / 40.0;
        // The sampled peak is at the outermost integer position, which sits
        // exactly at the wall, so it should hit the closed form.
        assert!(
            (profile.peak() - expected).abs() < 1e-5,
            "{} vs {expected}",
            profile.peak()
        );
    }

    #[test]
    fn electric_and_magnetic_sample_different_positions() {
        // If these ever coincide the layer has stopped being impedance
        // matched, which is the failure mode that still looks plausible.
        let profile = AbsorbingProfile::new(&grid(), Boundary::DEFAULT);
        let coord = [3, 22, 24];
        assert_ne!(
            profile.electric(Axis::X, coord),
            profile.magnetic(Axis::X, coord)
        );
    }

    #[test]
    fn row_split_agrees_with_the_direct_lookup() {
        let profile = AbsorbingProfile::new(&grid(), Boundary::DEFAULT);
        for axis in Axis::ALL {
            for coord in [
                [0, 0, 0],
                [3, 5, 44],
                [20, 22, 24],
                [39, 43, 47],
                [7, 2, 40],
            ] {
                let (varying, constant) = profile.electric_row(axis, coord[1], coord[2]);
                assert_eq!(varying[coord[0]] + constant, profile.electric(axis, coord));
                let (varying, constant) = profile.magnetic_row(axis, coord[1], coord[2]);
                assert_eq!(varying[coord[0]] + constant, profile.magnetic(axis, coord));
            }
        }
    }

    #[test]
    fn packed_layout_round_trips() {
        let grid = grid();
        let profile = AbsorbingProfile::new(&grid, Boundary::DEFAULT);
        let packed = profile.packed();
        let [nx, ny, nz] = grid.extent.as_array();
        assert_eq!(packed.len(), 2 * (nx + ny + nz));
        assert_eq!(packed[0], profile.integer[0][0]);
        assert_eq!(packed[nx], profile.integer[1][0]);
        assert_eq!(packed[nx + ny + nz], profile.half[0][0]);
    }

    #[test]
    #[should_panic(expected = "do not fit")]
    fn rejects_a_layer_thicker_than_the_domain() {
        let grid = Grid::new(Extent::cube(12), 1e-3);
        let _ = AbsorbingProfile::new(&grid, Boundary::DEFAULT);
    }
}

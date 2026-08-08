//! The Yee grid: extents, indexing, and the staggering convention.
//!
//! # Staggering
//!
//! Every FDTD bug is an off-by-half, so the convention is written down once, here,
//! and every other module refers back to it. Positions are in units of the cell
//! size `Δ`, with the origin at the corner of cell `(0, 0, 0)`:
//!
//! | Field | Position          | Time    |
//! |-------|-------------------|---------|
//! | `Ex`  | `(i+½, j,   k  )` | `n`     |
//! | `Ey`  | `(i,   j+½, k  )` | `n`     |
//! | `Ez`  | `(i,   j,   k+½)` | `n`     |
//! | `Hx`  | `(i,   j+½, k+½)` | `n+½`   |
//! | `Hy`  | `(i+½, j,   k+½)` | `n+½`   |
//! | `Hz`  | `(i+½, j+½, k  )` | `n+½`   |
//!
//! Read as a rule rather than a table: **`E` along axis `a` is offset by half a
//! cell along `a` and sits on integer positions along the other two. `H` along `a`
//! is the mirror image — integer along `a`, half along the other two.**
//!
//! All six components are stored as plain `[x × y × z]` arrays of the same shape.
//! The staggering is a statement about what an index *means*, not about layout.
//!
//! # Cyclic symmetry
//!
//! The curl updates are invariant under the cyclic relabelling `x → y → z → x`, so
//! both solvers here write the update once and run it for each axis `a` with
//! `b = a.next()` and `c = a.prev()`:
//!
//! ```text
//! H[a] -= (S/μr) · ( ∂E[c]/∂b − ∂E[b]/∂c )     forward differences
//! E[a] += (S/εr) · ( ∂H[c]/∂b − ∂H[b]/∂c )     backward differences
//! ```
//!
//! Forward for `H` and backward for `E` is what places each derivative exactly at
//! the position of the field it updates.

use std::f64;

/// Speed of light in vacuum, m/s.
pub const SPEED_OF_LIGHT: f32 = 299_792_458.0;
/// Vacuum magnetic permeability, H/m.
pub const VACUUM_PERMEABILITY: f32 = 1.256_637_1e-6;
/// Vacuum electric permittivity, F/m.
pub const VACUUM_PERMITTIVITY: f32 = 8.854_188e-12;
/// Wave impedance of free space, Ω.
pub const VACUUM_IMPEDANCE: f32 = 376.730_32;

/// One of the three coordinate axes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// All three axes, in cyclic order.
    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// Position of this axis in a `[x, y, z]` array.
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    /// The next axis in the cyclic order `x → y → z → x`.
    pub const fn next(self) -> Self {
        match self {
            Self::X => Self::Y,
            Self::Y => Self::Z,
            Self::Z => Self::X,
        }
    }

    /// The previous axis in the cyclic order.
    pub const fn prev(self) -> Self {
        self.next().next()
    }
}

/// Size of the grid in cells.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Extent {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Extent {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn cube(side: u32) -> Self {
        Self::new(side, side, side)
    }

    /// Extent as an `[x, y, z]` array.
    pub const fn as_array(&self) -> [usize; 3] {
        [self.x as usize, self.y as usize, self.z as usize]
    }

    /// Number of cells, which is the length of every field array.
    pub const fn total(&self) -> usize {
        self.x as usize * self.y as usize * self.z as usize
    }

    /// Distance in elements between neighbours along each axis.
    ///
    /// The layout is x-major (adjacent `x` are adjacent in memory), which is
    /// what makes the innermost loop of both solvers a unit-stride sweep.
    pub const fn strides(&self) -> [usize; 3] {
        [1, self.x as usize, self.x as usize * self.y as usize]
    }

    /// Linear index of a cell.
    pub const fn index(&self, coord: [usize; 3]) -> usize {
        (coord[2] * self.y as usize + coord[1]) * self.x as usize + coord[0]
    }
}

/// The discretization: how big a cell is, and how big a time step is.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Grid {
    pub extent: Extent,
    /// Cell size `Δ` in metres. Cells are cubic.
    pub cell_size: f32,
    /// Courant number `S = c·Δt/Δ`. Must stay below [`Grid::COURANT_LIMIT`].
    pub courant: f32,
}

impl Grid {
    /// Stability limit for 3D FDTD on a cubic grid: `1/√3`.
    ///
    /// Above it the leapfrog scheme has eigenvalues off the unit circle and
    /// the fields grow without bound. There is no graceful degradation — it
    /// either steps stably or it explodes exponentially.
    pub const COURANT_LIMIT: f32 = 0.577_350_26;

    /// A grid with the default Courant number of `0.5`, which leaves headroom
    /// below the limit for lossy media and the absorbing layer.
    pub fn new(extent: Extent, cell_size: f32) -> Self {
        Self {
            extent,
            cell_size,
            courant: 0.5,
        }
    }

    /// A grid covering `size` metres at `resolution` cells per metre.
    ///
    /// This is the constructor scenes are authored against: it is the one where
    /// changing the resolution leaves the physical problem alone.
    pub fn for_size(size: [f32; 3], resolution: f32) -> Self {
        assert!(
            resolution > 0.0 && resolution.is_finite(),
            "resolution must be positive and finite, got {resolution}"
        );
        let cells = size.map(|metres| (metres * resolution).round().max(2.0) as u32);
        Self::new(Extent::new(cells[0], cells[1], cells[2]), 1.0 / resolution)
    }

    /// Cells per metre.
    pub fn resolution(&self) -> f32 {
        1.0 / self.cell_size
    }

    /// Physical size of the domain in metres.
    pub fn size(&self) -> [f32; 3] {
        self.extent.as_array().map(|n| n as f32 * self.cell_size)
    }

    /// Time step `Δt = S·Δ/c`, in seconds.
    pub fn time_step(&self) -> f32 {
        self.courant * self.cell_size / SPEED_OF_LIGHT
    }

    /// Cell coordinate of a physical position.
    ///
    /// # The origin is the centre of the domain
    ///
    /// Not the corner. Geometry written against a centred origin survives a
    /// change of resolution *and* a change of domain size without moving,
    /// which is the whole point of specifying it in metres. Against a corner
    /// origin, growing the domain to give a wave more room to fly would drag
    /// every object along with the far wall.
    pub fn to_cell(&self, position: [f32; 3]) -> [f32; 3] {
        let extent = self.extent.as_array();
        std::array::from_fn(|axis| position[axis] / self.cell_size + 0.5 * extent[axis] as f32)
    }

    /// Physical position of a possibly fractional cell coordinate.
    pub fn to_position(&self, cell: [f32; 3]) -> [f32; 3] {
        let extent = self.extent.as_array();
        std::array::from_fn(|axis| (cell[axis] - 0.5 * extent[axis] as f32) * self.cell_size)
    }

    /// Index of the cell containing a physical position, clamped to the domain.
    ///
    /// The *floor*, not the nearest integer: [`Self::to_cell`] measures from
    /// cell corners, so cell `i` spans `[i, i+1)` and its centre is at `i+0.5`.
    /// Rounding instead would put every cell centre in its right-hand
    /// neighbour, which is a half-cell error that looks like nothing.
    pub fn cell_containing(&self, position: [f32; 3]) -> [usize; 3] {
        let cell = self.to_cell(position);
        let extent = self.extent.as_array();
        std::array::from_fn(|axis| (cell[axis].floor().max(0.0) as usize).min(extent[axis] - 1))
    }

    /// Physical position of the centre of a cell, which is where material
    /// membership is decided.
    pub fn cell_center(&self, coord: [usize; 3]) -> [f32; 3] {
        self.to_position(std::array::from_fn(|axis| coord[axis] as f32 + 0.5))
    }

    /// Whether a physical position lies inside the domain.
    pub fn contains(&self, position: [f32; 3]) -> bool {
        let size = self.size();
        (0..3).all(|axis| position[axis].abs() <= 0.5 * size[axis])
    }

    /// Cells per wavelength at `frequency` in a medium of the given refractive
    /// index.
    ///
    /// Below about 10 the phase error is visible; 20 is the working minimum.
    pub fn cells_per_wavelength(&self, frequency: f32, refractive_index: f32) -> f32 {
        SPEED_OF_LIGHT / (refractive_index * frequency * self.cell_size)
    }

    /// The frequency whose free-space wavelength spans `cells` cells.
    pub fn frequency_for_resolution(&self, cells: f32) -> f32 {
        SPEED_OF_LIGHT / (cells * self.cell_size)
    }

    /// Panics if the configuration cannot step stably.
    pub fn validate(&self) {
        assert!(
            self.extent.x >= 2 && self.extent.y >= 2 && self.extent.z >= 2,
            "grid must be at least 2 cells across on every axis, got {:?}",
            self.extent
        );
        assert!(
            self.cell_size > 0.0 && self.cell_size.is_finite(),
            "cell size must be positive and finite, got {}",
            self.cell_size
        );
        assert!(
            self.courant > 0.0 && self.courant <= Self::COURANT_LIMIT,
            "Courant number {} is outside (0, {}]; the scheme would diverge",
            self.courant,
            Self::COURANT_LIMIT
        );
    }
}

/// The analytic phase velocity of the discrete scheme, as a fraction of `c`.
///
/// The Yee scheme does not propagate all wavelengths at `c`. For a plane wave
/// with `n` cells per wavelength travelling along `direction` (a unit vector),
/// the numerical dispersion relation
///
/// ```text
/// (1/S)² sin²(π S / n) = Σ_a sin²(π d_a / n)
/// ```
///
/// is solved for the numerical wavenumber. The error is `O((Δ/λ)²)`, worst
/// along the grid axes and best along the body diagonal. This function is what
/// the validation suite compares the measured velocity against — the point of
/// the test is that the solver matches the *discrete* physics, not that it
/// matches `c`, which it provably cannot.
pub fn numerical_phase_velocity(
    courant: f32,
    cells_per_wavelength: f32,
    direction: [f32; 3],
) -> f32 {
    // The relative gap between the ideal and numerical wavenumbers is of order
    // 1e-4 and lands on 1e-7-relative quantities, so the root find runs in
    // `f64`. This is analysis code, not an inner loop.
    let courant = courant as f64;
    let cells = cells_per_wavelength as f64;
    let norm = direction
        .iter()
        .map(|&d| (d as f64) * (d as f64))
        .sum::<f64>()
        .sqrt();
    let unit = direction.map(|d| d as f64 / norm);

    let target = {
        let s = (f64::consts::PI * courant / cells).sin() / courant;
        s * s
    };
    // The sum of squared sines rises monotonically with `k` over the bracket,
    // so plain bisection converges without any guarding.
    let residual = |k_delta: f64| -> f64 {
        unit.iter()
            .map(|&d| {
                let s = (0.5 * k_delta * d).sin();
                s * s
            })
            .sum::<f64>()
            - target
    };

    let ideal = 2.0 * f64::consts::PI / cells;
    let (mut lo, mut hi) = (0.5 * ideal, 2.0 * ideal);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if residual(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // `v/c = ω/(kc)`, and `ω = k_ideal·c` by construction of `cells`.
    (ideal / (0.5 * (lo + hi))) as f32
}

#[cfg(test)]
mod tests {
    use super::{Axis, Extent, Grid, numerical_phase_velocity};

    #[test]
    fn axis_cycles() {
        for axis in Axis::ALL {
            assert_eq!(axis.next().next().next(), axis);
            assert_eq!(axis.prev(), axis.next().next());
            assert_ne!(axis.next(), axis);
        }
    }

    #[test]
    fn indexing_matches_strides() {
        let extent = Extent::new(5, 7, 11);
        let strides = extent.strides();
        assert_eq!(extent.total(), 5 * 7 * 11);
        for (axis, stride) in strides.iter().enumerate() {
            let mut coord = [1, 2, 3];
            let base = extent.index(coord);
            coord[axis] += 1;
            assert_eq!(extent.index(coord), base + stride);
        }
    }

    #[test]
    fn dispersion_is_slower_along_axes_than_diagonals() {
        // The classic signature of Yee numerical dispersion: axial waves lag,
        // diagonal waves are nearly exact, and both approach `c` as the mesh
        // is refined.
        let axial = numerical_phase_velocity(0.5, 20.0, [1.0, 0.0, 0.0]);
        let diagonal = numerical_phase_velocity(0.5, 20.0, [1.0, 1.0, 1.0]);
        assert!(axial < diagonal, "{axial} should lag {diagonal}");
        assert!(axial < 1.0 && axial > 0.99, "axial velocity {axial}");
        assert!(
            diagonal <= 1.0 && diagonal > 0.999,
            "diagonal velocity {diagonal}"
        );

        let fine = numerical_phase_velocity(0.5, 40.0, [1.0, 0.0, 0.0]);
        assert!(1.0 - fine < 1.0 - axial, "refining should reduce the error");
    }

    #[test]
    fn physical_coordinates_are_centred_and_round_trip() {
        let grid = Grid::new(Extent::new(40, 60, 80), 1e-3);
        let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|axis| (a[axis] - b[axis]).abs() < 1e-7);
        assert!(close(grid.size(), [0.04, 0.06, 0.08]), "{:?}", grid.size());
        // The origin sits at the centre, so the domain runs from −L/2 to +L/2.
        assert_eq!(grid.to_cell([0.0; 3]), [20.0, 30.0, 40.0]);
        assert_eq!(grid.to_position([20.0, 30.0, 40.0]), [0.0; 3]);
        assert!(grid.contains([0.019, -0.029, 0.0]));
        assert!(!grid.contains([0.021, 0.0, 0.0]));

        for position in [[0.0; 3], [0.011, -0.007, 0.03], [-0.02, 0.03, -0.04]] {
            let round_trip = grid.to_position(grid.to_cell(position));
            for axis in 0..3 {
                assert!((round_trip[axis] - position[axis]).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn geometry_does_not_move_when_the_resolution_changes() {
        // The property the whole change to physical units exists to provide:
        // the same point in metres lands at the same fraction of the domain
        // regardless of how finely it is discretized.
        let size = [0.04, 0.06, 0.08];
        let coarse = Grid::for_size(size, 1000.0);
        let fine = Grid::for_size(size, 2000.0);
        assert_eq!(coarse.extent, Extent::new(40, 60, 80));
        assert_eq!(fine.extent, Extent::new(80, 120, 160));

        let position = [0.012, -0.018, 0.031];
        let coarse_cell = coarse.to_cell(position);
        let fine_cell = fine.to_cell(position);
        for axis in 0..3 {
            assert!((fine_cell[axis] - 2.0 * coarse_cell[axis]).abs() < 1e-3);
        }
    }

    #[test]
    fn a_cell_centre_lands_back_in_its_own_cell() {
        // The half-cell error this guards against: rounding rather than
        // flooring puts every centre in the next cell along.
        let grid = Grid::new(Extent::new(12, 14, 16), 1e-3);
        for coord in [[0, 0, 0], [4, 5, 6], [11, 13, 15], [6, 7, 8]] {
            assert_eq!(grid.cell_containing(grid.cell_center(coord)), coord);
        }
        // Positions outside the domain clamp rather than wrapping or panicking.
        assert_eq!(grid.cell_containing([-9.9, 0.0, 0.0]), [0, 7, 8]);
        assert_eq!(grid.cell_containing([9.9, 9.9, 9.9]), [11, 13, 15]);
    }

    #[test]
    fn cell_centres_are_half_a_cell_in_from_the_corner() {
        let grid = Grid::new(Extent::cube(10), 1e-3);
        for axis in 0..3 {
            assert!((grid.cell_center([0, 0, 0])[axis] + 0.0045).abs() < 1e-7);
            assert!((grid.cell_center([9, 9, 9])[axis] - 0.0045).abs() < 1e-7);
        }
        assert!((grid.resolution() - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn time_step_respects_courant() {
        let grid = Grid::new(Extent::cube(16), 1e-3);
        grid.validate();
        assert!(grid.courant < Grid::COURANT_LIMIT);
        let expected = 0.5 * 1e-3 / super::SPEED_OF_LIGHT;
        assert!((grid.time_step() - expected).abs() < 1e-16);
    }

    #[test]
    #[should_panic(expected = "Courant number")]
    fn rejects_unstable_courant() {
        let mut grid = Grid::new(Extent::cube(16), 1e-3);
        grid.courant = 0.9;
        grid.validate();
    }
}

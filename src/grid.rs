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

/// Cell widths along one axis.
///
/// Cells are boxes, not cubes: each axis carries its own list of widths, so
/// resolution can be spent where the physics is small and saved where it is
/// not. A uniform grid is not a special case in the solvers — it is a
/// `Spacing` whose widths happen to be equal — so there is one code path and
/// the graded one cannot rot from disuse.
///
/// # Two spacings, and confusing them is the graded off-by-half
///
/// | | measured between | used by |
/// |---|---|---|
/// | primary `Δ[i]` | corner `i` and corner `i+1` | the `H` update |
/// | dual `Δ̃[i]` | centre `i−1` and centre `i` | the `E` update |
///
/// This follows from the staggering at the top of this module and nothing
/// else. `E` sits on integer positions and `H` on half positions, so a
/// difference of `E` spans one whole cell, while a difference of `H` spans two
/// half cells that may be different sizes. On a uniform grid the two coincide,
/// which is exactly why a graded grid finds the bug that a uniform one hides.
#[derive(Clone, Debug, PartialEq)]
pub struct Spacing {
    primary: Vec<f32>,
    dual: Vec<f32>,
    /// Corner positions from the low edge; `primary.len() + 1` of them.
    ///
    /// In `f64`, and the corners are where a graded grid would otherwise leak
    /// precision into places nothing else touches. A running sum of a few
    /// hundred `f32` widths drifts by enough to land a source one cell over —
    /// which reads as a physics bug, not an arithmetic one. These are three
    /// short 1D arrays built once, so the wider type costs nothing that
    /// matters.
    corners: Vec<f64>,
}

impl Spacing {
    /// Equal cells, which is what every scene gets until it asks otherwise.
    pub fn uniform(count: u32, width: f32) -> Self {
        let count = count as usize;
        assert!(count >= 2, "an axis needs at least 2 cells, got {count}");
        // Multiplied rather than accumulated, so a uniform axis lands on
        // exactly the corners it would have had before there was such a thing
        // as a `Spacing` — including the domain centre falling on a corner,
        // which is what makes `to_position(to_cell(p)) == p` hold to the bit.
        let corners = (0..=count).map(|i| i as f64 * f64::from(width)).collect();
        Self::assemble(vec![width; count], corners)
    }

    /// Arbitrary cell widths, low edge first.
    pub fn from_widths(primary: Vec<f32>) -> Self {
        let mut corners = Vec::with_capacity(primary.len() + 1);
        let mut offset = 0.0;
        corners.push(offset);
        for &width in &primary {
            offset += f64::from(width);
            corners.push(offset);
        }
        Self::assemble(primary, corners)
    }

    fn assemble(primary: Vec<f32>, corners: Vec<f64>) -> Self {
        assert!(
            primary.len() >= 2,
            "an axis needs at least 2 cells, got {}",
            primary.len()
        );
        assert!(
            primary.iter().all(|&w| w > 0.0 && w.is_finite()),
            "cell widths must be positive and finite"
        );
        // The dual width at `i` spans the two half cells either side of corner
        // `i`. At `i = 0` the left half lies outside the domain; the stencil is
        // clamped there, so the value is never read, and mirroring the first
        // cell keeps it finite rather than leaving a trap for later.
        let dual = (0..primary.len())
            .map(|i| 0.5 * (primary[i.saturating_sub(1)] + primary[i]))
            .collect();
        Self {
            primary,
            dual,
            corners,
        }
    }

    pub fn count(&self) -> u32 {
        self.primary.len() as u32
    }

    /// Cell widths, for the `H` update.
    pub fn primary(&self) -> &[f32] {
        &self.primary
    }

    /// Centre-to-centre distances, for the `E` update.
    pub fn dual(&self) -> &[f32] {
        &self.dual
    }

    /// Total length of the axis, in metres.
    pub fn length(&self) -> f32 {
        self.span() as f32
    }

    fn span(&self) -> f64 {
        self.corners[self.corners.len() - 1]
    }

    pub fn finest(&self) -> f32 {
        self.primary.iter().copied().fold(f32::INFINITY, f32::min)
    }

    pub fn coarsest(&self) -> f32 {
        self.primary
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Largest ratio between neighbouring cells, which is the number that says
    /// whether the grading is smooth enough to stay second-order accurate.
    pub fn worst_ratio(&self) -> f32 {
        self.primary
            .windows(2)
            .map(|pair| (pair[1] / pair[0]).max(pair[0] / pair[1]))
            .fold(1.0, f32::max)
    }

    /// Position of a possibly fractional cell coordinate, measured from the
    /// *centre* of the axis.
    ///
    /// Centred here rather than in the caller so that the half-length is
    /// subtracted in `f64`, against the same corners the forward map uses. Do
    /// it in `f32` outside and the domain centre stops landing exactly on the
    /// corner it is, which is a fraction of a cell — right up until it is the
    /// wrong side of one.
    pub fn position(&self, coordinate: f32) -> f32 {
        (self.offset_of(coordinate) - 0.5 * self.span()) as f32
    }

    /// Cell coordinate of a position measured from the centre — the inverse of
    /// [`Self::position`], which on a graded axis is a search rather than a
    /// division.
    pub fn coordinate(&self, position: f32) -> f32 {
        let offset = f64::from(position) + 0.5 * self.span();
        // `partition_point` gives the first corner strictly past `offset`, so
        // one less is the cell containing it.
        let index = self
            .corners
            .partition_point(|&corner| corner <= offset)
            .saturating_sub(1)
            .min(self.primary.len() - 1);
        (index as f64 + (offset - self.corners[index]) / f64::from(self.primary[index])) as f32
    }

    fn offset_of(&self, coordinate: f32) -> f64 {
        let last = self.primary.len() - 1;
        let index = (coordinate.floor().max(0.0) as usize).min(last);
        self.corners[index] + f64::from(coordinate - index as f32) * f64::from(self.primary[index])
    }
}

/// A request to resolve part of the domain more finely than the rest.
///
/// In metres, from the centre of the domain, exactly like [`crate::Shape`]: a
/// refinement is a statement about which physics you want resolved, so it has
/// to survive a change of base resolution or of domain size without moving,
/// the same way geometry does.
///
/// Refinement is per axis and therefore a *tensor product*: asking for a fine
/// box refines the three slabs it projects onto, all the way through the
/// domain. That is the price of keeping the grid logically dense — and keeping
/// it dense is what keeps the kernel a flat coalesced dispatch with no
/// interfaces, no interpolation and no late-time instability. It is right for
/// layered stacks, wires and boundary layers, and wasteful for one small ball
/// in a large empty box.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Refinement {
    pub center: [f32; 3],
    pub size: [f32; 3],
    /// Target cell size inside the region, in metres.
    pub cell_size: f32,
}

/// The discretization: how big the cells are, and how big a time step is.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(from = "GridSpec", into = "GridSpec")
)]
pub struct Grid {
    /// Size of the grid in cells. Derived from the spacing when the grid is
    /// graded, so it is not the same as the base extent that was asked for.
    ///
    /// Public for reading. It is kept consistent with `spacing` by the
    /// constructors — `spacing` is private so a `Grid` cannot be assembled any
    /// other way — and [`Grid::validate`] rechecks the pair rather than
    /// trusting it.
    pub extent: Extent,
    /// Courant number `S = c·Δt/Δ_ref`. Must stay below
    /// [`Grid::COURANT_LIMIT`].
    pub courant: f32,
    /// The recipe, kept so the grid can be rebuilt at another resolution and
    /// so a scene file stays something a person can read.
    base_extent: Extent,
    base_cell_size: f32,
    refinements: Vec<Refinement>,
    max_ratio: f32,
    spacing: [Spacing; 3],
}

/// The serialized form of a [`Grid`]: the recipe, not the resolved arrays.
///
/// A scene file says "cells this big, and this region finer"; writing out a
/// few hundred cell widths per axis would be unreadable, undiffable, and
/// welded to one resolution.
#[derive(Clone, Debug)]
// Named `Grid` on the wire, because that is the concept a scene file is
// stating. That the resolved grid is a different type is this module's
// business, not the reader's -- and it keeps every scene written before
// grading existed parsing unchanged.
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename = "Grid")
)]
pub struct GridSpec {
    /// Cells the domain would have at `cell_size`, which is what fixes its
    /// physical size. Refinements only subdivide; they never move a wall.
    pub extent: Extent,
    /// The coarse cell size in metres, away from any refinement.
    pub cell_size: f32,
    pub courant: f32,
    #[cfg_attr(feature = "serde", serde(default))]
    pub refinements: Vec<Refinement>,
    #[cfg_attr(feature = "serde", serde(default = "default_max_ratio"))]
    pub max_ratio: f32,
}

/// Growth cap between neighbouring cells.
///
/// The Yee scheme's second-order accuracy rests on the centred difference
/// being centred, which a change of spacing breaks: the local truncation error
/// drops to first order wherever the width changes. Grading gently keeps the
/// *global* error close to second order, and 1.15 is the usual working figure
/// — enough to cross an order of magnitude in about sixteen cells.
pub fn default_max_ratio() -> f32 {
    1.15
}

impl Grid {
    /// Stability limit for 3D FDTD on a cubic grid: `1/√3`.
    ///
    /// Above it the leapfrog scheme has eigenvalues off the unit circle and
    /// the fields grow without bound. There is no graceful degradation — it
    /// either steps stably or it explodes exponentially.
    pub const COURANT_LIMIT: f32 = 0.577_350_26;

    /// A uniform grid with the default Courant number of `0.5`, which leaves
    /// headroom below the limit for lossy media and the absorbing layer.
    pub fn new(extent: Extent, cell_size: f32) -> Self {
        Self::build(extent, cell_size, 0.5, Vec::new(), default_max_ratio())
    }

    /// A uniform grid covering `size` metres at `resolution` cells per metre.
    ///
    /// This is the constructor scenes are authored against: it is the one where
    /// changing the resolution leaves the physical problem alone.
    pub fn for_size(size: [f32; 3], resolution: f32) -> Self {
        Self::new(Self::extent_for(size, resolution), 1.0 / resolution)
    }

    /// A grid covering `size` metres at `resolution` cells per metre, refined
    /// where asked.
    ///
    /// The refinements only subdivide: the domain keeps the size and the walls
    /// it would have had without them, so adding one does not move the
    /// geometry or the absorbing layer.
    pub fn graded(size: [f32; 3], resolution: f32, refinements: Vec<Refinement>) -> Self {
        Self::build(
            Self::extent_for(size, resolution),
            1.0 / resolution,
            0.5,
            refinements,
            default_max_ratio(),
        )
    }

    fn extent_for(size: [f32; 3], resolution: f32) -> Extent {
        assert!(
            resolution > 0.0 && resolution.is_finite(),
            "resolution must be positive and finite, got {resolution}"
        );
        let cells = size.map(|metres| (metres * resolution).round().max(2.0) as u32);
        Extent::new(cells[0], cells[1], cells[2])
    }

    fn build(
        base_extent: Extent,
        base_cell_size: f32,
        courant: f32,
        refinements: Vec<Refinement>,
        max_ratio: f32,
    ) -> Self {
        assert!(
            base_cell_size > 0.0 && base_cell_size.is_finite(),
            "cell size must be positive and finite, got {base_cell_size}"
        );
        assert!(
            max_ratio > 1.0 && max_ratio.is_finite(),
            "the growth cap must exceed 1, got {max_ratio}"
        );
        let base = base_extent.as_array();
        let spacing: [Spacing; 3] = std::array::from_fn(|axis| {
            // With nothing to refine, the answer is the grid that was asked
            // for, cell for cell. Routing the uniform case through the marcher
            // would perturb every existing scene by a rounding remainder for
            // no reason at all.
            if refinements.is_empty() {
                Spacing::uniform(base[axis] as u32, base_cell_size)
            } else {
                grade(
                    axis,
                    base[axis] as f32 * base_cell_size,
                    base_cell_size,
                    &refinements,
                    max_ratio,
                )
            }
        });
        Self {
            extent: Extent::new(spacing[0].count(), spacing[1].count(), spacing[2].count()),
            courant,
            base_extent,
            base_cell_size,
            refinements,
            max_ratio,
            spacing,
        }
    }

    /// The same domain and the same refinements, rediscretized.
    pub fn with_resolution(&self, resolution: f32) -> Self {
        Self::build(
            Self::extent_for(self.size(), resolution),
            1.0 / resolution,
            self.courant,
            self.refinements.clone(),
            self.max_ratio,
        )
    }

    /// Cell widths along one axis.
    pub fn spacing(&self, axis: Axis) -> &Spacing {
        &self.spacing[axis.index()]
    }

    pub fn refinements(&self) -> &[Refinement] {
        &self.refinements
    }

    /// Whether every cell is the same cube, which is worth knowing because it
    /// is the case the analytic tests can be written against.
    pub fn is_uniform(&self) -> bool {
        self.refinements.is_empty()
    }

    /// Cells per metre at the coarse spacing.
    pub fn resolution(&self) -> f32 {
        1.0 / self.base_cell_size
    }

    /// The smallest and largest cell in the domain, in metres.
    pub fn finest(&self) -> f32 {
        Axis::ALL
            .iter()
            .map(|&a| self.spacing(a).finest())
            .fold(f32::INFINITY, f32::min)
    }

    pub fn coarsest(&self) -> f32 {
        Axis::ALL
            .iter()
            .map(|&a| self.spacing(a).coarsest())
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// The cubic cell that would impose the same stability limit as this one.
    ///
    /// The 3D Courant condition is `c·Δt·√(Σ_a 1/Δ_a²) ≤ 1`, taking the
    /// smallest cell on each axis. Rolling that sum into a single equivalent
    /// cube keeps `courant` meaning what it always meant — for a cubic grid
    /// this *is* the cell size — so the same `1/√3` limit still applies and
    /// nothing downstream has to learn a second convention.
    pub fn reference_cell_size(&self) -> f32 {
        let inverse_squares: f32 = Axis::ALL
            .iter()
            .map(|&a| {
                let finest = self.spacing(a).finest();
                1.0 / (finest * finest)
            })
            .sum();
        (3.0 / inverse_squares).sqrt()
    }

    /// Physical size of the domain in metres.
    pub fn size(&self) -> [f32; 3] {
        std::array::from_fn(|axis| self.spacing[axis].length())
    }

    /// Time step `Δt = S·Δ_ref/c`, in seconds.
    pub fn time_step(&self) -> f32 {
        self.courant * self.reference_cell_size() / SPEED_OF_LIGHT
    }

    /// `c·Δt/Δ` per cell along one axis — the factor the curl updates carry.
    ///
    /// On a uniform grid every entry is exactly `courant`, which is what the
    /// coefficient tables used to fold in directly. The `H` update differences
    /// `E` across a whole cell and the `E` update differences `H` across two
    /// half cells, so they read the primary and the dual list respectively.
    pub fn magnetic_gains(&self, axis: Axis) -> Vec<f32> {
        self.gains(self.spacing(axis).primary())
    }

    pub fn electric_gains(&self, axis: Axis) -> Vec<f32> {
        self.gains(self.spacing(axis).dual())
    }

    fn gains(&self, widths: &[f32]) -> Vec<f32> {
        let travel = SPEED_OF_LIGHT * self.time_step();
        widths.iter().map(|&width| travel / width).collect()
    }

    /// Centre of each cell along one axis, in metres from the domain centre.
    pub fn cell_centers(&self, axis: Axis) -> Vec<f32> {
        let spacing = self.spacing(axis);
        (0..spacing.count())
            .map(|index| spacing.position(index as f32 + 0.5))
            .collect()
    }

    /// Everything per-axis a kernel needs, packed the way the shaders unpack
    /// it: three sections of three axes each, back to back. The offsets follow
    /// from the extent, so nothing extra has to travel alongside — the same
    /// arrangement the absorber profile uses.
    ///
    /// | section | contents | read by |
    /// |---|---|---|
    /// | 0 | `c·Δt/Δ`, primary | the `H` update |
    /// | 1 | `c·Δt/Δ̃`, dual | the `E` update |
    /// | 2 | cell centre, metres | source apodization, and the renderer |
    pub fn packed_geometry(&self) -> Vec<f32> {
        let mut packed = Vec::new();
        for axis in Axis::ALL {
            packed.extend(self.magnetic_gains(axis));
        }
        for axis in Axis::ALL {
            packed.extend(self.electric_gains(axis));
        }
        for axis in Axis::ALL {
            packed.extend(self.cell_centers(axis));
        }
        packed
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
    /// # And it is not a scale
    ///
    /// On a graded axis this is the inverse of the cumulative cell width, so
    /// it is a search rather than a division. Writing it as a division is the
    /// mistake that puts a source in the wrong cell only in the scenes that
    /// asked for refinement — which are the ones nobody checks by hand.
    pub fn to_cell(&self, position: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|axis| self.spacing[axis].coordinate(position[axis]))
    }

    /// Physical position of a possibly fractional cell coordinate.
    pub fn to_position(&self, cell: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|axis| self.spacing[axis].position(cell[axis]))
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
    /// Cells per wavelength at `frequency` in a medium of the given refractive
    /// index, taken at the *coarsest* cell — the question being asked is
    /// whether the wave is resolved everywhere, and the answer is set by the
    /// worst place, not the average one.
    pub fn cells_per_wavelength(&self, frequency: f32, refractive_index: f32) -> f32 {
        SPEED_OF_LIGHT / (refractive_index * frequency * self.coarsest())
    }

    /// The frequency whose free-space wavelength spans `cells` of the coarsest
    /// cells.
    pub fn frequency_for_resolution(&self, cells: f32) -> f32 {
        SPEED_OF_LIGHT / (cells * self.coarsest())
    }

    /// Panics if the configuration cannot step stably.
    pub fn validate(&self) {
        assert!(
            self.extent.x >= 2 && self.extent.y >= 2 && self.extent.z >= 2,
            "grid must be at least 2 cells across on every axis, got {:?}",
            self.extent
        );
        // The pair is maintained by the constructors rather than derived on
        // every access, so it is worth rechecking: a grid whose spacing and
        // extent disagree indexes past the end of an axis somewhere deep in a
        // solver rather than failing here.
        let extent = self.extent.as_array();
        for axis in Axis::ALL {
            assert_eq!(
                self.spacing(axis).count() as usize,
                extent[axis.index()],
                "the {axis:?} spacing has {} cells but the extent says {}",
                self.spacing(axis).count(),
                extent[axis.index()],
            );
        }
        assert!(
            self.courant > 0.0 && self.courant <= Self::COURANT_LIMIT,
            "Courant number {} is outside (0, {}]; the scheme would diverge",
            self.courant,
            Self::COURANT_LIMIT
        );
    }
}

/// Cell widths along one axis, fine where a refinement asks and smoothly
/// graded back to `base` everywhere else.
///
/// Three steps, and the middle one is the whole idea:
///
/// 1. Sample the target width on a lattice: `base`, or the smallest refinement
///    covering that point.
/// 2. Limit how fast it may change. Between neighbouring cells the position
///    advances by about one width, so a per-cell ratio cap of `max_ratio` is a
///    Lipschitz bound of `max_ratio − 1` on width against distance. A forward
///    sweep and a backward sweep make that bound hold in both directions —
///    it is a distance transform, and two passes are exact.
/// 3. March out cells, then scale every width by one common factor so the last
///    corner lands exactly on the far wall. A uniform scale cannot break the
///    bound, because it leaves every ratio alone.
fn grade(
    axis: usize,
    length: f32,
    base: f32,
    refinements: &[Refinement],
    max_ratio: f32,
) -> Spacing {
    let finest = refinements
        .iter()
        .map(|refinement| refinement.cell_size)
        .fold(base, f32::min);
    assert!(
        finest > 0.0 && finest.is_finite(),
        "refinement cell sizes must be positive and finite, got {finest}"
    );
    // Four samples across the smallest cell asked for, so the marcher never
    // steps over a refinement narrower than it can see.
    let samples = ((length / finest * 4.0).ceil() as usize).clamp(64, 1 << 20);
    let step = length / samples as f32;

    let mut target = vec![base; samples + 1];
    for (index, width) in target.iter_mut().enumerate() {
        // Refinements are centred on the domain, like all geometry.
        let position = index as f32 * step - 0.5 * length;
        for refinement in refinements {
            let half = 0.5 * refinement.size[axis];
            if (position - refinement.center[axis]).abs() <= half {
                *width = width.min(refinement.cell_size);
            }
        }
    }

    let slope = (max_ratio - 1.0) * step;
    for index in 1..target.len() {
        target[index] = target[index].min(target[index - 1] + slope);
    }
    for index in (0..target.len() - 1).rev() {
        target[index] = target[index].min(target[index + 1] + slope);
    }

    let mut widths = Vec::new();
    let mut offset = 0.0f32;
    while offset < length {
        let index = ((offset / step) as usize).min(target.len() - 1);
        widths.push(target[index]);
        offset += target[index];
        assert!(
            widths.len() <= 1 << 16,
            "grading the {axis} axis ran to {} cells; the refinements are \
             finer than the domain can carry",
            widths.len()
        );
    }

    // The march reads a *sampled* profile, so quantization lets neighbours
    // drift a little past the cap — measurably: 1.18 where 1.15 was asked for.
    // Reapplying the bound to the cells themselves puts it where the accuracy
    // argument actually lives, and makes `Spacing::worst_ratio` a contract
    // rather than an observation. Both sweeps only ever shrink a cell, so the
    // rescale below still has something to correct.
    for index in 1..widths.len() {
        widths[index] = widths[index].min(widths[index - 1] * max_ratio);
    }
    for index in (0..widths.len() - 1).rev() {
        widths[index] = widths[index].min(widths[index + 1] * max_ratio);
    }

    // The march always ends past the far wall, by up to a whole cell. Dropping
    // the last one may land closer than keeping it, and taking whichever is
    // nearer halves the worst-case correction the rescale has to smear over
    // every cell in the axis.
    let mut total: f32 = widths.iter().sum();
    if widths.len() > 2 {
        let without = total - widths[widths.len() - 1];
        if (length - without).abs() < (total - length).abs() {
            widths.pop();
            total = without;
        }
    }
    // One common factor, so the last corner lands exactly on the far wall
    // without disturbing a single ratio.
    let scale = length / total;
    for width in &mut widths {
        *width *= scale;
    }
    Spacing::from_widths(widths)
}

impl From<GridSpec> for Grid {
    fn from(spec: GridSpec) -> Self {
        Self::build(
            spec.extent,
            spec.cell_size,
            spec.courant,
            spec.refinements,
            spec.max_ratio,
        )
    }
}

impl From<Grid> for GridSpec {
    fn from(grid: Grid) -> Self {
        Self {
            extent: grid.base_extent,
            cell_size: grid.base_cell_size,
            courant: grid.courant,
            refinements: grid.refinements,
            max_ratio: grid.max_ratio,
        }
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

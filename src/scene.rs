//! A scene: everything needed to reproduce a run.
//!
//! The scene is the unit of persistence and the unit the validation suite is
//! written against, so both solvers are constructed from one and nothing else.
//! Geometry is rasterized into the per-cell material index once, at construction;
//! after that the solver only ever reads indices.

use crate::{
    boundary::Boundary,
    grid::{Axis, Extent, Grid, SPEED_OF_LIGHT},
    material::{Material, MaterialTable},
    source::{Source, Waveform},
};

/// Cell size the built-in presets are defined at: one millimetre.
const PRESET_CELL_SIZE: f32 = 1e-3;

/// A primitive painted into the material index.
///
/// # Everything is in metres, with the origin at the centre of the domain
///
/// Not cell indices. Geometry written in cells is welded to one resolution:
/// change the cell size and every object moves, so you cannot run the same
/// scene at twice the resolution to check the answer stopped changing — which
/// is the single most useful thing a saved scene enables. See
/// [`Grid::to_cell`] for why the origin is the centre rather than a corner.
///
/// Later shapes overwrite earlier ones, so a scene reads top to bottom like a
/// stack of paint.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum Shape {
    /// Axis-aligned box given by its centre and its full side lengths.
    ///
    /// Centre-and-size rather than min-and-max because it is what survives a
    /// change of domain size unchanged, and because it is how every photonics
    /// tool states a box.
    Block {
        center: [f32; 3],
        size: [f32; 3],
        material: u32,
    },
    Sphere {
        center: [f32; 3],
        radius: f32,
        material: u32,
    },
    /// A slab spanning the whole domain except along `axis`, where it is
    /// `thickness` metres thick and centred at `offset`.
    Slab {
        axis: Axis,
        offset: f32,
        thickness: f32,
        material: u32,
    },
}

impl Shape {
    /// Whether a physical position is inside this shape.
    fn contains(&self, position: [f32; 3]) -> bool {
        match *self {
            Self::Block { center, size, .. } => {
                (0..3).all(|axis| (position[axis] - center[axis]).abs() <= 0.5 * size[axis])
            }
            Self::Sphere { center, radius, .. } => {
                let distance_squared: f32 = (0..3)
                    .map(|axis| {
                        let offset = position[axis] - center[axis];
                        offset * offset
                    })
                    .sum();
                distance_squared <= radius * radius
            }
            Self::Slab {
                axis,
                offset,
                thickness,
                ..
            } => (position[axis.index()] - offset).abs() <= 0.5 * thickness,
        }
    }

    /// Half-extent of the shape along each axis, measured from its centre.
    /// A slab is unbounded in its two transverse directions.
    fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        match *self {
            Self::Block { center, size, .. } => (center, size.map(|s| 0.5 * s)),
            Self::Sphere { center, radius, .. } => (center, [radius; 3]),
            Self::Slab {
                axis,
                offset,
                thickness,
                ..
            } => {
                let mut center = [0.0; 3];
                center[axis.index()] = offset;
                let mut half = [f32::INFINITY; 3];
                half[axis.index()] = 0.5 * thickness;
                (center, half)
            }
        }
    }

    fn material(&self) -> u32 {
        match *self {
            Self::Block { material, .. }
            | Self::Sphere { material, .. }
            | Self::Slab { material, .. } => material,
        }
    }
}

/// A complete, reproducible simulation setup.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Scene {
    pub grid: Grid,
    pub boundary: Boundary,
    pub materials: MaterialTable,
    pub shapes: Vec<Shape>,
    pub sources: Vec<Source>,
}

impl Scene {
    /// An empty domain of the given physical size, at `resolution` cells per
    /// metre, with absorbing walls and no sources.
    ///
    /// This is the constructor to author against. [`Self::empty`] is the
    /// grid-level one, for when the exact cell count is what matters.
    pub fn sized(size: [f32; 3], resolution: f32) -> Self {
        Self::on_grid(Grid::for_size(size, resolution))
    }

    /// An empty domain on an explicitly chosen grid.
    pub fn empty(extent: Extent, cell_size: f32) -> Self {
        Self::on_grid(Grid::new(extent, cell_size))
    }

    /// An empty domain on a grid you built yourself — the way in for a graded
    /// one, since [`Grid::graded`] is where refinements are stated.
    pub fn on_grid(grid: Grid) -> Self {
        Self {
            grid,
            boundary: Boundary::DEFAULT,
            materials: MaterialTable::new(),
            shapes: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// The same physical scene, rediscretized at `resolution` cells per metre.
    ///
    /// Nothing moves: geometry and sources are in metres, so only the grid
    /// changes. This is what makes a convergence study a one-liner — run the
    /// same scene at 1x and 2x and see whether the answer stopped changing.
    /// It is also the check that catches an under-resolved result, which
    /// otherwise looks entirely convincing.
    pub fn with_resolution(&self, resolution: f32) -> Self {
        Self {
            grid: self.grid.with_resolution(resolution),
            ..self.clone()
        }
    }

    pub fn with_boundary(mut self, boundary: Boundary) -> Self {
        self.boundary = boundary;
        self
    }

    pub fn with_source(mut self, source: Source) -> Self {
        self.sources.push(source);
        self
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shapes.push(shape);
        self
    }

    /// Rasterizes the shapes into a per-cell material index.
    pub fn material_indices(&self) -> Vec<u32> {
        let [nx, ny, nz] = self.grid.extent.as_array();
        let mut indices = vec![MaterialTable::VACUUM; self.grid.extent.total()];
        if self.shapes.is_empty() {
            return indices;
        }
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let coord = [x, y, z];
                    // Membership is decided at the cell centre. Subpixel
                    // smoothing -- averaging the permittivity across a
                    // partially filled cell -- is the highest-leverage
                    // accuracy upgrade available here, and this is the line it
                    // would replace.
                    let position = self.grid.cell_center(coord);
                    for shape in self.shapes.iter() {
                        if shape.contains(position) {
                            indices[self.grid.extent.index(coord)] = shape.material();
                        }
                    }
                }
            }
        }
        indices
    }

    /// Checks that the scene is actually resolvable, and reports what is
    /// wrong if it is not.
    ///
    /// Under-resolving is the most common way to get a result that looks
    /// convincing and is wrong: the wave still propagates, just at the wrong
    /// speed, so nothing announces the problem.
    /// For each material present, its refractive index paired with the largest
    /// cell it sits in — the place where that material is least well resolved.
    ///
    /// Vacuum is included, because an under-resolved *empty* region is just as
    /// wrong; it simply needs a bigger cell to get there.
    fn worst_resolved(&self) -> Vec<(f32, f32)> {
        let extent = self.grid.extent.as_array();
        let widths = Axis::ALL.map(|axis| self.grid.spacing(axis).primary());
        let indices = self.material_indices();
        let mut coarsest = vec![0.0f32; self.materials.len()];
        for z in 0..extent[2] {
            for y in 0..extent[1] {
                let slab = widths[1][y].max(widths[2][z]);
                let row = (z * extent[1] + y) * extent[0];
                for (x, &width) in widths[0].iter().enumerate() {
                    let cell = slab.max(width);
                    let material = indices[row + x] as usize;
                    coarsest[material] = coarsest[material].max(cell);
                }
            }
        }
        coarsest
            .iter()
            .enumerate()
            .filter(|&(_, &cell)| cell > 0.0)
            .map(|(material, &cell)| (self.materials.get(material as u32).refractive_index(), cell))
            .collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        self.grid.validate();
        // Structural checks first. Everything below rasterizes the shapes and
        // looks materials up by the index it finds, so a dangling reference has
        // to be reported here rather than panicking three lines later.
        //
        // The table invariants cannot be broken through the API -- `push`
        // never touches slot 0 -- but a scene file states the table in text.
        if self.materials.is_empty() {
            return Err("the material table is empty; slot 0 must be vacuum".to_string());
        }
        if self.materials.get(0) != Material::VACUUM {
            return Err(
                "material 0 must be vacuum: every unpainted cell wears index 0".to_string(),
            );
        }
        // Mirrors what the absorber's constructor will panic over. Catching it
        // here means "this scene cannot run at this resolution" arrives as a
        // report instead of detonating inside `Simulation::new` -- which is
        // how a scene that validates at its native resolution used to fail
        // after `with_resolution` coarsened it.
        if let Boundary::Absorbing {
            thickness,
            target_reflection,
        } = self.boundary
        {
            for (axis, &cells) in self.grid.extent.as_array().iter().enumerate() {
                if 2 * thickness as usize >= cells {
                    return Err(format!(
                        "absorbing layers {thickness} cells thick meet in the middle of \
                         axis {axis}, which is only {cells} cells across; thin the layer \
                         or raise the resolution"
                    ));
                }
            }
            if !(target_reflection > 0.0 && target_reflection < 1.0) {
                return Err(format!(
                    "target reflection {target_reflection} must lie in (0, 1)"
                ));
            }
        }
        for shape in self.shapes.iter() {
            if shape.material() as usize >= self.materials.len() {
                return Err(format!(
                    "shape references material {} but the table has {}",
                    shape.material(),
                    self.materials.len()
                ));
            }
        }
        // Resolution is checked against the cells each material *actually
        // occupies*, not against the coarsest cell in the domain. Pairing the
        // highest index in the scene with the largest cell anywhere is only the
        // right question on a uniform grid — on a graded one it rejects
        // precisely the scenes grading exists for, where the slow material is
        // small and the refinement is over it.
        let worst = self.worst_resolved();
        for source in self.sources.iter() {
            let frequency = source.waveform.dominant_frequency();
            for &(index, cell_size) in worst.iter() {
                let cells = SPEED_OF_LIGHT / (index * frequency * cell_size);
                if cells < 10.0 {
                    return Err(format!(
                        "{cells:.1} cells per wavelength at {:.3} GHz in index {index:.2}, \
                         where the cells are {:.3} mm; below 10 the phase error is severe",
                        frequency * 1e-9,
                        cell_size * 1e3,
                    ));
                }
            }
        }
        for source in self.sources.iter() {
            let position = source.position();
            if !self.grid.contains(position) {
                let size = self.grid.size();
                return Err(format!(
                    "source at {position:?} m is outside the {size:?} m domain; \
                     positions are in metres from the centre, not cell indices"
                ));
            }
        }
        for shape in self.shapes.iter() {
            // A shape entirely outside the domain paints nothing, and after
            // the move to metres the way that happens is passing cell indices
            // by mistake. Silently rendering an empty scene is the worst
            // possible response to that.
            let (center, half) = shape.bounds();
            let size = self.grid.size();
            if (0..3).any(|axis| center[axis].abs() - half[axis] > 0.5 * size[axis]) {
                return Err(format!(
                    "shape centred at {center:?} m lies entirely outside the \
                     {size:?} m domain; positions are in metres from the centre"
                ));
            }
        }
        Ok(())
    }

    /// The signature demo: a transversely apodized wave packet launched into
    /// free space, with absorbing walls so it leaves cleanly.
    ///
    /// The packet is polarized along `z` and travels along `x`, so `Ez` and
    /// `Hy` carry it and the exchange between them is visible side-on.
    pub fn photon(extent: Extent) -> Self {
        // The presets take a cell count because that is what sets the cost of
        // a run, and pin the resolution at a millimetre per cell — so `extent`
        // reads as "how many millimetres across". To refine one instead of
        // enlarging it, call [`Self::with_resolution`].
        let grid = Grid::new(extent, PRESET_CELL_SIZE);
        let size = grid.size();
        // Sixteen cells per wavelength: numerical dispersion stays a fraction
        // of a percent, and the packet is short enough to be an object in the
        // box rather than filling it.
        let frequency = grid.frequency_for_resolution(16.0);
        Self {
            boundary: Boundary::DEFAULT,
            materials: MaterialTable::new(),
            shapes: Vec::new(),
            sources: vec![Source::sheet(
                Axis::X,
                // A fifth of the way in from the low wall, which with a centred
                // origin is three tenths back from the middle.
                -0.3 * size[0],
                0.25 * size[1].min(size[2]),
                Axis::Z,
                // Two cycles. Longer and the packet is no longer a packet — at
                // four it is eighty cells from end to end, which is most of a
                // usable domain.
                Waveform::gaussian_pulse(frequency, 2.0),
            )],
            grid,
        }
    }

    /// A closed perfectly conducting box driven by a point dipole.
    ///
    /// Lossless, so it is the energy-conservation reference; and once the
    /// standing-wave pattern settles, the electric and magnetic energy
    /// densities alternate in place, which is the clearest view of the two
    /// fields trading energy back and forth.
    pub fn cavity(extent: Extent) -> Self {
        Self::cavity_on(Grid::new(extent, PRESET_CELL_SIZE))
    }

    /// The same cavity on a grid you built yourself, graded or not.
    pub fn cavity_on(grid: Grid) -> Self {
        let size = grid.size();
        let frequency = grid.frequency_for_resolution(16.0);
        Self {
            boundary: Boundary::Pec,
            materials: MaterialTable::new(),
            shapes: Vec::new(),
            sources: vec![Source::point(
                // Off-centre, so the dipole is not sitting on a node of every
                // mode it would otherwise excite.
                [-size[0] / 6.0, 0.0, 0.0],
                Axis::Z,
                Waveform::ricker(frequency),
            )],
            grid,
        }
    }

    /// A wave packet meeting a dielectric slab head on: part reflects, part
    /// refracts, and the wavelength visibly shortens inside the glass.
    pub fn slab(extent: Extent, refractive_index: f32) -> Self {
        let mut scene = Self::photon(extent);
        let size = scene.grid.size();
        let material = scene.materials.push(Material::refractive(refractive_index));
        scene.shapes.push(Shape::Slab {
            axis: Axis::X,
            offset: size[0] / 8.0,
            thickness: size[0] / 4.0,
            material,
        });
        // Keep the resolution inside the glass too, where the wavelength is
        // `n` times shorter — the easiest resolution mistake to make, and one
        // that shows up as the wrong refraction angle rather than as an error.
        for source in scene.sources.iter_mut() {
            let frequency = scene.grid.frequency_for_resolution(16.0 * refractive_index);
            source.waveform = Waveform::gaussian_pulse(frequency, 2.0);
        }
        scene
    }
}

/// Reading and writing scenes as files.
///
/// RON rather than JSON: it round-trips Rust enums without inventing a tag
/// convention, it allows comments, and a scene file is meant to be read and
/// edited by a person. `Scene` is a plain data structure precisely so this is
/// possible — the alternative, which most FDTD packages take, is for a scene to
/// be a *program*, which you cannot diff, hash, or hand to a solver you did not
/// compile yourself.
#[cfg(feature = "serde")]
impl Scene {
    /// Parses a scene, reporting the line and column of a syntax error.
    ///
    /// Does not validate — call [`Self::validate`] afterwards. Parsing and
    /// meaning are separate failures and a caller may want to fix one without
    /// the other stopping it.
    pub fn from_ron(text: &str) -> Result<Self, String> {
        ron::from_str(text).map_err(|error| error.to_string())
    }

    /// Serializes a scene, formatted for a human to edit afterwards.
    pub fn to_ron(&self) -> Result<String, String> {
        let config = ron::ser::PrettyConfig::new()
            .struct_names(true)
            .separate_tuple_members(false);
        ron::ser::to_string_pretty(self, config).map_err(|error| error.to_string())
    }

    /// Reads a scene from disk.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Self::from_ron(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    /// Writes a scene to disk.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let path = path.as_ref();
        std::fs::write(path, self.to_ron()?).map_err(|error| format!("{}: {error}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::{Scene, Shape};
    use crate::{
        boundary::Boundary,
        grid::{Axis, Extent},
        material::{Material, MaterialTable},
        source::{Source, Waveform},
    };

    #[test]
    fn empty_scene_is_all_vacuum() {
        // 24 cells, not fewer: the default boundary is a ten-cell absorbing
        // layer per wall, and validation now checks that it fits -- on the
        // old eight-cell domain this scene validated and could not run.
        let scene = Scene::empty(Extent::cube(24), 1e-3);
        let indices = scene.material_indices();
        assert_eq!(indices.len(), 24 * 24 * 24);
        assert!(indices.iter().all(|&i| i == MaterialTable::VACUUM));
        assert_eq!(scene.validate(), Ok(()));
    }

    #[test]
    fn shapes_rasterize_and_later_ones_win() {
        let mut scene = Scene::empty(Extent::cube(16), 1e-3);
        let glass = scene.materials.push(Material::refractive(1.5));
        let metal = scene.materials.push(Material::PERFECT_CONDUCTOR);
        // A 16 mm box at 1 mm per cell: the domain runs from -8 mm to +8 mm.
        scene.shapes.push(Shape::Block {
            center: [0.0; 3],
            size: [8e-3; 3],
            material: glass,
        });
        scene.shapes.push(Shape::Sphere {
            center: [0.0; 3],
            radius: 2e-3,
            material: metal,
        });

        let indices = scene.material_indices();
        let at = |c: [usize; 3]| indices[scene.grid.extent.index(c)];
        assert_eq!(at([0, 0, 0]), MaterialTable::VACUUM);
        assert_eq!(at([5, 5, 5]), glass);
        // Centre of the sphere, painted after the block.
        assert_eq!(at([8, 8, 8]), metal);
        // Inside the block but outside the sphere.
        assert_eq!(at([11, 11, 11]), glass);
    }

    #[test]
    fn slab_shape_spans_one_axis_only() {
        let mut scene = Scene::empty(Extent::cube(12), 1e-3);
        let glass = scene.materials.push(Material::refractive(2.0));
        // 12 cells at 1 mm spans -6 mm to +6 mm; cells 3..6 are -2.5..-0.5 mm.
        scene.shapes.push(Shape::Slab {
            axis: Axis::Y,
            offset: -1.5e-3,
            thickness: 3e-3,
            material: glass,
        });
        let indices = scene.material_indices();
        let at = |c: [usize; 3]| indices[scene.grid.extent.index(c)];
        assert_eq!(at([0, 4, 11]), glass);
        assert_eq!(at([11, 4, 0]), glass);
        assert_eq!(at([5, 2, 5]), MaterialTable::VACUUM);
        assert_eq!(at([5, 6, 5]), MaterialTable::VACUUM);
    }

    #[test]
    fn validation_rejects_an_unresolved_source() {
        let mut scene = Scene::photon(Extent::cube(48));
        // Ten times the frequency is half a wavelength per cell: nonsense.
        for source in scene.sources.iter_mut() {
            let frequency = scene.grid.frequency_for_resolution(2.0);
            source.waveform = Waveform::ricker(frequency);
        }
        let error = scene.validate().unwrap_err();
        assert!(error.contains("cells per wavelength"), "{error}");
    }

    #[test]
    fn validation_rejects_a_dangling_material_reference() {
        let mut scene = Scene::empty(Extent::cube(24), 1e-3);
        scene.shapes.push(Shape::Block {
            center: [0.0; 3],
            size: [2e-3; 3],
            material: 7,
        });
        assert!(scene.validate().unwrap_err().contains("material 7"));
    }

    #[test]
    fn geometry_occupies_the_same_volume_at_any_resolution() {
        // The whole point of metres. If geometry were still in cell indices,
        // doubling the resolution would halve every object's physical size and
        // these fractions would fall by 8x.
        let mut scene = Scene::sized([0.032; 3], 1000.0);
        let glass = scene.materials.push(Material::refractive(1.5));
        let metal = scene.materials.push(Material::PERFECT_CONDUCTOR);
        scene.shapes.push(Shape::Sphere {
            center: [4e-3, -2e-3, 0.0],
            radius: 6e-3,
            material: glass,
        });
        scene.shapes.push(Shape::Block {
            center: [-8e-3, 0.0, 0.0],
            size: [4e-3, 8e-3, 8e-3],
            material: metal,
        });

        let fraction = |scene: &Scene, material: u32| {
            let indices = scene.material_indices();
            indices.iter().filter(|&&i| i == material).count() as f64 / indices.len() as f64
        };
        for material in [glass, metal] {
            let coarse = fraction(&scene, material);
            let fine = fraction(&scene.with_resolution(2000.0), material);
            assert!(coarse > 0.001, "material {material} painted nothing");
            // Only the staircase at the surface differs, and refining shrinks
            // it, so the volumes agree to a couple of percent.
            let difference = (fine - coarse).abs() / coarse;
            assert!(
                difference < 0.03,
                "material {material} occupies {coarse:.4} coarse vs {fine:.4} fine"
            );
        }
        assert_eq!(scene.with_resolution(2000.0).grid.extent, Extent::cube(64));
    }

    #[test]
    fn sources_keep_their_physical_position_across_resolutions() {
        let mut scene = Scene::sized([0.032; 3], 1000.0);
        scene.sources.push(Source::point(
            [8e-3, -4e-3, 0.0],
            Axis::Z,
            Waveform::ricker(scene.grid.frequency_for_resolution(20.0)),
        ));
        let fine = scene.with_resolution(2000.0);

        let coarse_cell = scene.sources[0].injection(&scene.grid, 0.0).origin;
        let fine_cell = fine.sources[0].injection(&fine.grid, 0.0).origin;
        // Same place in metres means twice the cell index at twice the
        // resolution, to within the cell the position falls in.
        for axis in 0..3 {
            let expected = 2 * coarse_cell[axis];
            assert!(
                fine_cell[axis].abs_diff(expected) <= 1,
                "axis {axis}: coarse cell {} became {}",
                coarse_cell[axis],
                fine_cell[axis]
            );
        }
    }

    #[test]
    fn validation_catches_geometry_written_in_cells_by_mistake() {
        // The failure mode this whole change introduces: passing what used to
        // be a cell index into a field that now wants metres. A sphere at
        // "20" is twenty metres out and paints nothing at all.
        let mut scene = Scene::sized([0.032; 3], 1000.0);
        let glass = scene.materials.push(Material::refractive(1.5));
        scene.shapes.push(Shape::Sphere {
            center: [20.0, 20.0, 20.0],
            radius: 5.0,
            material: glass,
        });
        let error = scene.validate().unwrap_err();
        assert!(error.contains("entirely outside"), "{error}");
    }

    #[test]
    fn validation_catches_a_source_outside_the_domain() {
        let mut scene = Scene::sized([0.032; 3], 1000.0);
        scene.sources.push(Source::point(
            [20.0, 0.0, 0.0],
            Axis::Z,
            Waveform::ricker(scene.grid.frequency_for_resolution(20.0)),
        ));
        let error = scene.validate().unwrap_err();
        assert!(error.contains("outside"), "{error}");
    }

    #[test]
    fn validation_catches_an_absorber_that_does_not_fit() {
        // Two ten-cell layers meet in the middle of a nineteen-cell axis.
        // This used to validate cleanly and then panic in `Simulation::new`,
        // which is exactly how a scene that runs fine at its native
        // resolution failed after `with_resolution` coarsened it.
        let scene = Scene::empty(Extent::cube(19), 1e-3).with_boundary(Boundary::Absorbing {
            thickness: 10,
            target_reflection: 1e-6,
        });
        let error = scene.validate().unwrap_err();
        assert!(error.contains("thick"), "{error}");
    }

    #[test]
    fn validation_catches_a_nonsense_target_reflection() {
        let scene = Scene::empty(Extent::cube(32), 1e-3).with_boundary(Boundary::Absorbing {
            thickness: 8,
            target_reflection: 0.0,
        });
        let error = scene.validate().unwrap_err();
        assert!(error.contains("reflection"), "{error}");
    }

    #[test]
    fn presets_are_valid_and_have_the_intended_boundaries() {
        let photon = Scene::photon(Extent::cube(64));
        assert_eq!(photon.validate(), Ok(()));
        assert_ne!(photon.boundary, Boundary::Pec);

        let cavity = Scene::cavity(Extent::cube(64));
        assert_eq!(cavity.validate(), Ok(()));
        assert_eq!(cavity.boundary, Boundary::Pec);

        // The slab preset has to stay resolved inside the glass, where the
        // wavelength is shorter — the easiest resolution mistake to make.
        let slab = Scene::slab(Extent::cube(64), 2.0);
        assert_eq!(slab.validate(), Ok(()));
        assert!(slab.materials.peak_refractive_index() >= 2.0);
    }
}

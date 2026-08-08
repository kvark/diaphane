//! A scene: everything needed to reproduce a run.
//!
//! The scene is the unit of persistence and the unit the validation suite is
//! written against, so both solvers are constructed from one and nothing else.
//! Geometry is rasterized into the per-cell material index once, at construction;
//! after that the solver only ever reads indices.

use crate::{
    boundary::Boundary,
    grid::{Axis, Extent, Grid},
    material::{Material, MaterialTable},
    source::{Source, Waveform},
};

/// A primitive painted into the material index.
///
/// Later shapes overwrite earlier ones, so a scene reads top to bottom like a
/// stack of paint.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum Shape {
    /// Axis-aligned box, inclusive of `min` and exclusive of `max`.
    Block {
        min: [u32; 3],
        max: [u32; 3],
        material: u32,
    },
    /// Sphere, in cell coordinates.
    Sphere {
        center: [f32; 3],
        radius: f32,
        material: u32,
    },
    /// A slab filling the domain except along `axis`, where it spans
    /// `start..end`.
    Slab {
        axis: Axis,
        start: u32,
        end: u32,
        material: u32,
    },
}

impl Shape {
    /// Whether a cell is inside this shape.
    fn contains(&self, coord: [usize; 3]) -> bool {
        match *self {
            Self::Block { min, max, .. } => (0..3)
                .all(|axis| coord[axis] >= min[axis] as usize && coord[axis] < max[axis] as usize),
            Self::Sphere { center, radius, .. } => {
                let distance_squared: f32 = (0..3)
                    .map(|axis| {
                        let d = coord[axis] as f32 + 0.5 - center[axis];
                        d * d
                    })
                    .sum();
                distance_squared <= radius * radius
            }
            Self::Slab {
                axis, start, end, ..
            } => {
                let c = coord[axis.index()];
                c >= start as usize && c < end as usize
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
    /// An empty domain with absorbing walls and no sources.
    pub fn empty(extent: Extent, cell_size: f32) -> Self {
        Self {
            grid: Grid::new(extent, cell_size),
            boundary: Boundary::DEFAULT,
            materials: MaterialTable::new(),
            shapes: Vec::new(),
            sources: Vec::new(),
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
                    for shape in self.shapes.iter() {
                        if shape.contains(coord) {
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
    pub fn validate(&self) -> Result<(), String> {
        self.grid.validate();
        for source in self.sources.iter() {
            let index = self.materials.peak_refractive_index();
            let cells = self
                .grid
                .cells_per_wavelength(source.waveform.dominant_frequency(), index);
            if cells < 10.0 {
                return Err(format!(
                    "{cells:.1} cells per wavelength at {:.3} GHz in index {index:.2}; \
                     below 10 the phase error is severe",
                    source.waveform.dominant_frequency() * 1e-9,
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
        Ok(())
    }

    /// The signature demo: a transversely apodized wave packet launched into
    /// free space, with absorbing walls so it leaves cleanly.
    ///
    /// The packet is polarized along `z` and travels along `x`, so `Ez` and
    /// `Hy` carry it and the exchange between them is visible side-on.
    pub fn photon(extent: Extent) -> Self {
        let cell_size = 1e-3;
        let grid = Grid::new(extent, cell_size);
        // Twenty cells per wavelength: enough that numerical dispersion is a
        // fraction of a percent rather than something you can see.
        let frequency = grid.frequency_for_resolution(20.0);
        let waist = 0.25 * extent.y.min(extent.z) as f32;
        Self {
            grid,
            boundary: Boundary::DEFAULT,
            materials: MaterialTable::new(),
            shapes: Vec::new(),
            sources: vec![Source::sheet(
                Axis::X,
                extent.x / 5,
                waist,
                Axis::Z,
                Waveform::gaussian_pulse(frequency, 4.0),
            )],
        }
    }

    /// A closed perfectly conducting box driven by a point dipole.
    ///
    /// Lossless, so it is the energy-conservation reference; and once the
    /// standing-wave pattern settles, the electric and magnetic energy
    /// densities alternate in place, which is the clearest view of the two
    /// fields trading energy back and forth.
    pub fn cavity(extent: Extent) -> Self {
        let cell_size = 1e-3;
        let grid = Grid::new(extent, cell_size);
        let frequency = grid.frequency_for_resolution(16.0);
        // Off-centre so the dipole is not sitting on a node of every mode it
        // would otherwise excite.
        let at = [extent.x / 3, extent.y / 2, extent.z / 2];
        Self {
            grid,
            boundary: Boundary::Pec,
            materials: MaterialTable::new(),
            shapes: Vec::new(),
            sources: vec![Source::point(at, Axis::Z, Waveform::ricker(frequency))],
        }
    }

    /// A wave packet meeting a dielectric slab head on: part reflects, part
    /// refracts, and the wavelength visibly shortens inside the glass.
    pub fn slab(extent: Extent, refractive_index: f32) -> Self {
        let mut scene = Self::photon(extent);
        let material = scene.materials.push(Material::refractive(refractive_index));
        scene.shapes.push(Shape::Slab {
            axis: Axis::X,
            start: extent.x / 2,
            end: extent.x * 3 / 4,
            material,
        });
        // Keep 20 cells per wavelength inside the glass too, where the
        // wavelength is `n` times shorter.
        for source in scene.sources.iter_mut() {
            let frequency = scene.grid.frequency_for_resolution(20.0 * refractive_index);
            source.waveform = Waveform::gaussian_pulse(frequency, 4.0);
        }
        scene
    }
}

#[cfg(test)]
mod tests {
    use super::{Scene, Shape};
    use crate::{
        boundary::Boundary,
        grid::{Axis, Extent},
        material::{Material, MaterialTable},
    };

    #[test]
    fn empty_scene_is_all_vacuum() {
        let scene = Scene::empty(Extent::cube(8), 1e-3);
        let indices = scene.material_indices();
        assert_eq!(indices.len(), 8 * 8 * 8);
        assert!(indices.iter().all(|&i| i == MaterialTable::VACUUM));
        assert_eq!(scene.validate(), Ok(()));
    }

    #[test]
    fn shapes_rasterize_and_later_ones_win() {
        let mut scene = Scene::empty(Extent::cube(16), 1e-3);
        let glass = scene.materials.push(Material::refractive(1.5));
        let metal = scene.materials.push(Material::PERFECT_CONDUCTOR);
        scene.shapes.push(Shape::Block {
            min: [4, 4, 4],
            max: [12, 12, 12],
            material: glass,
        });
        scene.shapes.push(Shape::Sphere {
            center: [8.0, 8.0, 8.0],
            radius: 2.0,
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
        scene.shapes.push(Shape::Slab {
            axis: Axis::Y,
            start: 3,
            end: 6,
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
            source.waveform = crate::source::Waveform::ricker(frequency);
        }
        let error = scene.validate().unwrap_err();
        assert!(error.contains("cells per wavelength"), "{error}");
    }

    #[test]
    fn validation_rejects_a_dangling_material_reference() {
        let mut scene = Scene::empty(Extent::cube(8), 1e-3);
        scene.shapes.push(Shape::Block {
            min: [0, 0, 0],
            max: [2, 2, 2],
            material: 7,
        });
        assert!(scene.validate().unwrap_err().contains("material 7"));
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

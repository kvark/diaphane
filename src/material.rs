//! Materials, and the update coefficients derived from them.
//!
//! The grid stores a `u32` index per cell into a small table, never the
//! coefficients themselves. Repainting geometry is then a single index write with
//! no recomputation, and adding per-material parameters later costs nothing per
//! cell.
//!
//! # Normalized fields
//!
//! Everything below is written for the impedance-normalized electric field
//! `Ẽ = √(ε₀/μ₀)·E = E/η₀`. Substituting it into Maxwell's curl equations gives
//!
//! ```text
//! ∂Ẽ/∂t = (c/εr)·∇×H − (σ/(ε₀ εr))·Ẽ
//! ∂H/∂t = −(c/μr)·∇×Ẽ − (σ*/(μ₀ μr))·H
//! ```
//!
//! which is *symmetric*: the two gains are `c/εr` and `c/μr`, the two loss rates
//! have the same units, and in free space they are numerically identical. That
//! symmetry is the whole reason for normalizing — it keeps both fields at the same
//! order of magnitude, which matters for `f32`, and it collapses two different
//! update kernels into one shape used twice.

use crate::grid::{Grid, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};

/// A linear, non-dispersive, possibly lossy medium.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Material {
    /// Relative permittivity `εr`.
    pub relative_permittivity: f32,
    /// Relative permeability `μr`.
    pub relative_permeability: f32,
    /// Electric conductivity `σ`, in S/m. Infinite means a perfect conductor.
    pub conductivity: f32,
    /// Magnetic loss `σ*`, in Ω/m. Not a physical property of ordinary matter;
    /// it exists so that absorbing regions can be impedance-matched.
    pub magnetic_loss: f32,
}

impl Material {
    pub const VACUUM: Self = Self {
        relative_permittivity: 1.0,
        relative_permeability: 1.0,
        conductivity: 0.0,
        magnetic_loss: 0.0,
    };

    /// A perfect electric conductor: tangential `E` is pinned to zero.
    pub const PERFECT_CONDUCTOR: Self = Self {
        relative_permittivity: 1.0,
        relative_permeability: 1.0,
        conductivity: f32::INFINITY,
        magnetic_loss: 0.0,
    };

    /// A lossless dielectric of the given relative permittivity.
    pub const fn dielectric(relative_permittivity: f32) -> Self {
        Self {
            relative_permittivity,
            ..Self::VACUUM
        }
    }

    /// A lossless dielectric specified by its refractive index `n = √(εr μr)`,
    /// with the permeability left at 1.
    pub fn refractive(index: f32) -> Self {
        Self::dielectric(index * index)
    }

    /// A conductive medium whose magnetic loss is chosen to match its
    /// wave impedance to the lossless version of itself.
    ///
    /// The matching condition is `σ*/μ = σ/ε`. A wave crossing into such a
    /// medium sees no impedance step and therefore does not reflect — it only
    /// decays. This is the mechanism behind [`crate::Boundary::Absorbing`].
    pub fn matched_lossy(relative_permittivity: f32, conductivity: f32) -> Self {
        Self {
            relative_permittivity,
            relative_permeability: 1.0,
            conductivity,
            magnetic_loss: conductivity * VACUUM_PERMEABILITY
                / (VACUUM_PERMITTIVITY * relative_permittivity),
        }
    }

    /// Refractive index `√(εr μr)` of the lossless part of the medium.
    pub fn refractive_index(&self) -> f32 {
        (self.relative_permittivity * self.relative_permeability).sqrt()
    }

    /// Update coefficients for this material on the given grid.
    ///
    /// The gains are the *material* half only — `1/εr` and `1/μr`. The
    /// geometric half, `c·Δt/Δ`, is per axis and per cell once the grid can be
    /// graded, so it lives in [`Grid::electric_gains`] and
    /// [`Grid::magnetic_gains`] and the two multiply in the kernel. On a
    /// uniform grid the geometric factor is the Courant number everywhere,
    /// which is exactly what used to be folded in here.
    pub fn coefficients(&self, grid: &Grid) -> Coefficients {
        // A perfect conductor is the `loss = 1` limit rather than a branch in
        // the kernel: it makes the electric update `E ← (0·E + 0·curl)/2 = 0`,
        // which is precisely the boundary condition, with no divergence and
        // no special case downstream.
        if self.conductivity.is_infinite() {
            return Coefficients {
                electric_gain: 0.0,
                electric_loss: 1.0,
                magnetic_gain: 1.0 / self.relative_permeability,
                magnetic_loss: 0.0,
            };
        }
        let half_step = 0.5 * grid.time_step();
        Coefficients {
            electric_gain: 1.0 / self.relative_permittivity,
            electric_loss: half_step * self.conductivity
                / (VACUUM_PERMITTIVITY * self.relative_permittivity),
            magnetic_gain: 1.0 / self.relative_permeability,
            magnetic_loss: half_step * self.magnetic_loss
                / (VACUUM_PERMEABILITY * self.relative_permeability),
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::VACUUM
    }
}

/// The four numbers a cell needs to advance one step.
///
/// Both solvers apply them in the same semi-implicit form, which is what makes
/// a lossy medium stable at any conductivity:
///
/// ```text
/// let l = electric_loss + absorber_loss;
/// E ← ((1 − l)·E + electric_gain·(∇×H)) / (1 + l)
///
/// let m = magnetic_loss + absorber_loss;
/// H ← ((1 − m)·H − magnetic_gain·(∇×E)) / (1 + m)
/// ```
///
/// With no loss this collapses to the plain leapfrog `E += gain·curl`. The
/// absorber term is added at *use* rather than baked in, because it varies
/// across the grid while the material does not.
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Coefficients {
    /// `S/εr`.
    pub electric_gain: f32,
    /// `σ·Δt / (2 ε₀ εr)`, dimensionless.
    pub electric_loss: f32,
    /// `S/μr`.
    pub magnetic_gain: f32,
    /// `σ*·Δt / (2 μ₀ μr)`, dimensionless.
    pub magnetic_loss: f32,
}

/// The palette of materials a scene can paint with.
///
/// Index 0 is always vacuum, so a freshly zeroed material-index buffer
/// describes an empty domain.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct MaterialTable {
    materials: Vec<Material>,
}

impl MaterialTable {
    /// A table containing only vacuum.
    pub fn new() -> Self {
        Self {
            materials: vec![Material::VACUUM],
        }
    }

    /// Index of vacuum, which is always present.
    pub const VACUUM: u32 = 0;

    /// Adds a material and returns its index. Duplicates are merged, so
    /// callers can add freely without growing the table.
    pub fn push(&mut self, material: Material) -> u32 {
        match self.materials.iter().position(|m| *m == material) {
            Some(index) => index as u32,
            None => {
                self.materials.push(material);
                (self.materials.len() - 1) as u32
            }
        }
    }

    pub fn get(&self, index: u32) -> Material {
        self.materials[index as usize]
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// Always false: vacuum occupies index 0 of every table.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The coefficient table to upload alongside the material indices.
    pub fn coefficients(&self, grid: &Grid) -> Vec<Coefficients> {
        self.materials
            .iter()
            .map(|m| m.coefficients(grid))
            .collect()
    }

    /// Highest refractive index present, which sets the resolution the grid
    /// has to meet.
    pub fn peak_refractive_index(&self) -> f32 {
        self.materials
            .iter()
            .filter(|m| m.conductivity.is_finite())
            .map(|m| m.refractive_index())
            .fold(1.0, f32::max)
    }
}

impl Default for MaterialTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{Material, MaterialTable};
    use crate::grid::{Extent, Grid};

    fn grid() -> Grid {
        Grid::new(Extent::cube(16), 1e-3)
    }

    #[test]
    fn vacuum_is_lossless_and_symmetric() {
        let c = Material::VACUUM.coefficients(&grid());
        assert_eq!(c.electric_loss, 0.0);
        assert_eq!(c.magnetic_loss, 0.0);
        // The point of impedance normalization: both curls carry the same gain.
        assert_eq!(c.electric_gain, c.magnetic_gain);
        // The material half only. The geometric half is `c·Δt/Δ`, which is per
        // axis and per cell once the grid can be graded, and on this uniform
        // grid comes out as the Courant number everywhere.
        assert_eq!(c.electric_gain, 1.0);
        let grid = grid();
        for axis in crate::Axis::ALL {
            for gain in grid
                .electric_gains(axis)
                .iter()
                .chain(&grid.magnetic_gains(axis))
            {
                assert!((gain - grid.courant).abs() < 1e-6, "gain {gain}");
            }
        }
    }

    #[test]
    fn perfect_conductor_pins_the_field_to_zero() {
        let c = Material::PERFECT_CONDUCTOR.coefficients(&grid());
        let updated = |e: f32, curl: f32| {
            ((1.0 - c.electric_loss) * e + c.electric_gain * curl) / (1.0 + c.electric_loss)
        };
        assert_eq!(updated(1.0, 5.0), 0.0);
        assert_eq!(updated(-3.0, -2.0), 0.0);
    }

    #[test]
    fn matched_loss_has_equal_electric_and_magnetic_rates() {
        // Equal normalized loss rates is exactly the zero-reflection
        // condition, so this is the invariant the absorber depends on.
        for &permittivity in &[1.0, 2.25, 12.0] {
            let material = Material::matched_lossy(permittivity, 0.05);
            let c = material.coefficients(&grid());
            let relative = (c.electric_loss - c.magnetic_loss).abs() / c.electric_loss;
            assert!(relative < 1e-5, "{permittivity}: {c:?}");
        }
    }

    #[test]
    fn conductivity_raises_loss_without_touching_gain() {
        let lossless = Material::dielectric(4.0).coefficients(&grid());
        let lossy = Material::matched_lossy(4.0, 0.02).coefficients(&grid());
        assert_eq!(lossless.electric_gain, lossy.electric_gain);
        assert_eq!(lossless.electric_loss, 0.0);
        assert!(lossy.electric_loss > 0.0);
    }

    #[test]
    fn table_starts_with_vacuum_and_merges_duplicates() {
        let mut table = MaterialTable::new();
        assert_eq!(table.get(MaterialTable::VACUUM), Material::VACUUM);
        let glass = table.push(Material::refractive(1.5));
        assert_eq!(glass, 1);
        assert_eq!(table.push(Material::refractive(1.5)), glass);
        assert_eq!(table.push(Material::VACUUM), MaterialTable::VACUUM);
        assert_eq!(table.len(), 2);
        assert!((table.peak_refractive_index() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn perfect_conductor_does_not_inflate_the_resolution_requirement() {
        let mut table = MaterialTable::new();
        table.push(Material::PERFECT_CONDUCTOR);
        assert_eq!(table.peak_refractive_index(), 1.0);
    }
}

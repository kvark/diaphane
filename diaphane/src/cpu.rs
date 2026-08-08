//! The reference solver.
//!
//! Plain `f32` loops, no intrinsics, no threads. Its job is to be *obviously* the
//! equations in [`crate::grid`] and nothing else, so that when the GPU kernel and
//! this disagree, this one is right. It doubles as the baseline the benchmarks
//! measure against, since every FDTD package in the Rust ecosystem today is
//! CPU-resident and comparing a GPU number against them would measure the hardware
//! rather than the code.
//!
//! # The step
//!
//! ```text
//! update_magnetic()   H^{n−½} → H^{n+½}, reading E^n
//! update_electric()   E^n     → E^{n+1}, reading H^{n+½}
//! inject()            add the sources into E^{n+1}
//! ```
//!
//! No double buffering anywhere: the `H` update reads only `E` and the `E` update
//! reads only `H`, so each can be done in place. The instinct on a GPU is to
//! ping-pong, and here it would double the memory traffic for nothing.

use crate::{
    boundary::AbsorbingProfile,
    grid::{Axis, Grid},
    material::{Coefficients, MaterialTable},
    scene::Scene,
    source::Source,
};

/// Field energy, split into its two halves.
///
/// In normalized units the densities are `½·εr·Ẽ²` and `½·μr·H²`; multiplying
/// the total by `μ₀·Δ³` recovers joules. The split is the interesting part:
/// in a propagating wave the two halves track each other, and in a standing
/// wave they alternate, which is the same statement as "the fields are trading
/// energy back and forth".
///
/// # This total is not exactly the conserved quantity
///
/// `E` and `H` live half a time step apart, so `½Σ(εE² + μH²)` mixes two
/// instants. The quantity leapfrog conserves exactly is
/// `½Σ(εEⁿ·Eⁿ + μHⁿ⁻½·Hⁿ⁺½)`, and the difference between them oscillates at
/// the step scale with a relative amplitude of order `ω·Δt` — a few percent at
/// typical resolutions. It does not accumulate. Testing conservation therefore
/// means comparing time *averages*, and a run that looks like it is gaining
/// energy over a handful of steps is almost always looking at this instead.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Energy {
    pub electric: f64,
    pub magnetic: f64,
}

impl Energy {
    pub fn total(&self) -> f64 {
        self.electric + self.magnetic
    }
}

/// A running simulation on the CPU.
pub struct Simulation {
    grid: Grid,
    electric: [Vec<f32>; 3],
    magnetic: [Vec<f32>; 3],
    material_index: Vec<u32>,
    materials: MaterialTable,
    coefficients: Vec<Coefficients>,
    absorber: AbsorbingProfile,
    sources: Vec<Source>,
    step: u64,
}

impl Simulation {
    /// Builds a simulation from a scene. Panics if the grid cannot step
    /// stably; use [`Scene::validate`] first for the recoverable problems.
    pub fn new(scene: &Scene) -> Self {
        scene.grid.validate();
        let total = scene.grid.extent.total();
        Self {
            grid: scene.grid,
            electric: std::array::from_fn(|_| vec![0.0; total]),
            magnetic: std::array::from_fn(|_| vec![0.0; total]),
            material_index: scene.material_indices(),
            materials: scene.materials.clone(),
            coefficients: scene.materials.coefficients(&scene.grid),
            absorber: AbsorbingProfile::new(&scene.grid, scene.boundary),
            sources: scene.sources.clone(),
            step: 0,
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Steps completed so far.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Simulated time in seconds, which is the time `E` currently holds.
    pub fn time(&self) -> f32 {
        self.step as f32 * self.grid.time_step()
    }

    pub fn electric(&self, axis: Axis) -> &[f32] {
        &self.electric[axis.index()]
    }

    pub fn magnetic(&self, axis: Axis) -> &[f32] {
        &self.magnetic[axis.index()]
    }

    /// Direct access for seeding an initial condition in a test.
    pub fn electric_mut(&mut self, axis: Axis) -> &mut [f32] {
        &mut self.electric[axis.index()]
    }

    pub fn magnetic_mut(&mut self, axis: Axis) -> &mut [f32] {
        &mut self.magnetic[axis.index()]
    }

    /// Value of an `E` component at a cell.
    pub fn sample_electric(&self, axis: Axis, coord: [usize; 3]) -> f32 {
        self.electric[axis.index()][self.grid.extent.index(coord)]
    }

    pub fn sample_magnetic(&self, axis: Axis, coord: [usize; 3]) -> f32 {
        self.magnetic[axis.index()][self.grid.extent.index(coord)]
    }

    /// Clears the fields and rewinds the clock, leaving geometry alone.
    pub fn reset(&mut self) {
        for axis in 0..3 {
            self.electric[axis].fill(0.0);
            self.magnetic[axis].fill(0.0);
        }
        self.step = 0;
    }

    /// Advances one full time step.
    pub fn advance(&mut self) {
        self.update_magnetic();
        self.update_electric();
        // `E` now holds time `(step + 1)·Δt`, which is when the source acts.
        let time = (self.step + 1) as f32 * self.grid.time_step();
        self.inject(time);
        self.step += 1;
    }

    pub fn advance_by(&mut self, steps: u64) {
        for _ in 0..steps {
            self.advance();
        }
    }

    /// `H[a] ← ((1−m)·H[a] − gain·(∂E[c]/∂b − ∂E[b]/∂c)) / (1+m)`, forward
    /// differences.
    fn update_magnetic(&mut self) {
        let extent = self.grid.extent.as_array();
        let strides = self.grid.extent.strides();
        for axis in Axis::ALL {
            let (a, b, c) = (axis.index(), axis.next().index(), axis.prev().index());
            let (stride_b, stride_c) = (strides[b], strides[c]);
            // A forward difference along `b` and `c` needs a neighbour there,
            // so the last plane along each is left untouched. Those samples sit
            // outside the physical domain.
            let mut limit = extent;
            limit[b] -= 1;
            limit[c] -= 1;

            let target = &mut self.magnetic[a];
            let field_b = &self.electric[b];
            let field_c = &self.electric[c];
            for z in 0..limit[2] {
                for y in 0..limit[1] {
                    let row = (z * extent[1] + y) * extent[0];
                    let (absorb_x, absorb_row) = self.absorber.magnetic_row(axis, y, z);
                    for (x, &absorb) in absorb_x.iter().enumerate().take(limit[0]) {
                        let index = row + x;
                        let curl = (field_c[index + stride_b] - field_c[index])
                            - (field_b[index + stride_c] - field_b[index]);
                        let coefficients = self.coefficients[self.material_index[index] as usize];
                        let loss = coefficients.magnetic_loss + absorb_row + absorb;
                        target[index] = ((1.0 - loss) * target[index]
                            - coefficients.magnetic_gain * curl)
                            / (1.0 + loss);
                    }
                }
            }
        }
    }

    /// `E[a] ← ((1−l)·E[a] + gain·(∂H[c]/∂b − ∂H[b]/∂c)) / (1+l)`, backward
    /// differences.
    fn update_electric(&mut self) {
        let extent = self.grid.extent.as_array();
        let strides = self.grid.extent.strides();
        for axis in Axis::ALL {
            let (a, b, c) = (axis.index(), axis.next().index(), axis.prev().index());
            let (stride_b, stride_c) = (strides[b], strides[c]);
            // A backward difference needs the previous plane, so index 0 along
            // `b` and `c` is never updated. It stays at zero, which is exactly
            // a perfect electric conductor at the two low faces — the PEC
            // boundary costs nothing because it is what the stencil already
            // does.
            let mut start = [0; 3];
            start[b] = 1;
            start[c] = 1;

            let target = &mut self.electric[a];
            let field_b = &self.magnetic[b];
            let field_c = &self.magnetic[c];
            for z in start[2]..extent[2] {
                for y in start[1]..extent[1] {
                    let row = (z * extent[1] + y) * extent[0];
                    let (absorb_x, absorb_row) = self.absorber.electric_row(axis, y, z);
                    for (x, &absorb) in absorb_x.iter().enumerate().skip(start[0]) {
                        let index = row + x;
                        let curl = (field_c[index] - field_c[index - stride_b])
                            - (field_b[index] - field_b[index - stride_c]);
                        let coefficients = self.coefficients[self.material_index[index] as usize];
                        let loss = coefficients.electric_loss + absorb_row + absorb;
                        target[index] = ((1.0 - loss) * target[index]
                            + coefficients.electric_gain * curl)
                            / (1.0 + loss);
                    }
                }
            }
        }
    }

    /// Adds every source into `E`. Additive, never assigning — see
    /// [`crate::source`].
    fn inject(&mut self, time: f32) {
        for source in self.sources.iter() {
            let injection = source.injection(&self.grid, time);
            if injection.value == 0.0 {
                continue;
            }
            let target = &mut self.electric[injection.component];
            for dz in 0..injection.extent[2] {
                for dy in 0..injection.extent[1] {
                    for dx in 0..injection.extent[0] {
                        let coord = [
                            injection.origin[0] + dx,
                            injection.origin[1] + dy,
                            injection.origin[2] + dz,
                        ];
                        target[self.grid.extent.index(coord)] +=
                            injection.value * injection.weight(coord);
                    }
                }
            }
        }
    }

    /// Total field energy in normalized units.
    ///
    /// Accumulated in `f64`: the point of this quantity is to detect drift
    /// over 10⁵ steps, and an `f32` accumulator over a million cells would
    /// lose more precision than the drift being measured.
    pub fn energy(&self) -> Energy {
        let mut energy = Energy::default();
        for (index, &material) in self.material_index.iter().enumerate() {
            let material = self.materials.get(material);
            let mut electric = 0.0f32;
            let mut magnetic = 0.0f32;
            for axis in 0..3 {
                let e = self.electric[axis][index];
                let h = self.magnetic[axis][index];
                electric += e * e;
                magnetic += h * h;
            }
            energy.electric += 0.5 * f64::from(material.relative_permittivity * electric);
            energy.magnetic += 0.5 * f64::from(material.relative_permeability * magnetic);
        }
        energy
    }

    /// Per-cell energy density, summed over both fields, for visualization and
    /// for locating a wave packet.
    pub fn energy_density(&self) -> Vec<f32> {
        (0..self.grid.extent.total())
            .map(|index| {
                let mut density = 0.0;
                for axis in 0..3 {
                    let e = self.electric[axis][index];
                    let h = self.magnetic[axis][index];
                    density += 0.5 * (e * e + h * h);
                }
                density
            })
            .collect()
    }

    /// Whether any field value has stopped being a finite number, which is
    /// what a Courant violation looks like a few hundred steps in.
    pub fn is_finite(&self) -> bool {
        self.electric
            .iter()
            .chain(self.magnetic.iter())
            .flat_map(|field| field.iter())
            .all(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::Simulation;
    use crate::{
        boundary::Boundary,
        grid::{Axis, Extent},
        scene::Scene,
        source::{Source, Waveform},
    };

    fn pulse_scene(extent: Extent) -> Scene {
        let scene = Scene::empty(extent, 1e-3);
        let frequency = scene.grid.frequency_for_resolution(20.0);
        let center = [extent.x / 2, extent.y / 2, extent.z / 2];
        scene.with_source(Source::point(center, Axis::Z, Waveform::ricker(frequency)))
    }

    #[test]
    fn starts_at_rest() {
        let simulation = Simulation::new(&pulse_scene(Extent::cube(24)));
        assert_eq!(simulation.step_count(), 0);
        assert_eq!(simulation.time(), 0.0);
        assert_eq!(simulation.energy().total(), 0.0);
    }

    #[test]
    fn a_source_puts_energy_in_and_the_absorber_takes_it_out() {
        let mut simulation = Simulation::new(&pulse_scene(Extent::cube(48)));
        simulation.advance_by(120);
        let peak = simulation.energy().total();
        assert!(peak > 0.0, "the source did nothing");
        assert!(simulation.is_finite());

        // The Ricker wavelet is long finished; everything it launched should
        // reach the walls and be absorbed.
        simulation.advance_by(600);
        let remaining = simulation.energy().total();
        assert!(
            remaining < peak * 1e-4,
            "{remaining:e} left of {peak:e} — the absorber is leaking"
        );
    }

    /// Mean total energy over a window, which averages away the half-step
    /// oscillation described on [`super::Energy`].
    fn mean_energy(simulation: &mut Simulation, steps: u64) -> f64 {
        let mut sum = 0.0;
        for _ in 0..steps {
            simulation.advance();
            sum += simulation.energy().total();
        }
        sum / steps as f64
    }

    #[test]
    fn a_pec_box_holds_on_to_its_energy() {
        let scene = pulse_scene(Extent::cube(40)).with_boundary(Boundary::Pec);
        let mut simulation = Simulation::new(&scene);
        simulation.advance_by(200);
        let settled = mean_energy(&mut simulation, 200);
        simulation.advance_by(2000);
        let later = mean_energy(&mut simulation, 200);
        let drift = (later - settled).abs() / settled;
        assert!(drift < 1e-3, "energy drifted by {drift:e}");
    }

    #[test]
    fn the_two_energies_alternate_in_a_cavity() {
        // The physical claim the visualizer exists to show: in a standing wave
        // the electric and magnetic halves are in quadrature, so their
        // difference oscillates while the sum does not.
        let mut simulation = Simulation::new(&Scene::cavity(Extent::cube(40)));
        simulation.advance_by(400);
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for _ in 0..200 {
            simulation.advance();
            let energy = simulation.energy();
            let split = (energy.electric - energy.magnetic) / energy.total();
            minimum = minimum.min(split);
            maximum = maximum.max(split);
        }
        assert!(
            maximum - minimum > 0.3,
            "the split barely moved: {minimum} to {maximum}"
        );
    }

    #[test]
    fn a_pulse_expands_at_about_the_speed_of_light() {
        // A coarse sanity check that the wave goes somewhere at roughly the
        // right rate; `tests/validation.rs` measures the phase velocity
        // properly against the discrete dispersion relation.
        let extent = Extent::cube(80);
        let mut simulation = Simulation::new(&pulse_scene(extent));
        let center = 40;
        let radius = 25;
        let probe = [center + radius, center, center];

        let mut arrival = None;
        for step in 1..400 {
            simulation.advance();
            if simulation.sample_electric(Axis::Z, probe).abs() > 1e-4 {
                arrival = Some(step);
                break;
            }
        }
        let arrival = arrival.expect("the pulse never reached the probe") as f32;
        // The source peaks `1.4/f` after t=0, and the grid runs at S = 0.5, so
        // the wave needs `radius / S` steps of travel plus the source delay.
        let travel = radius as f32 / simulation.grid().courant;
        assert!(
            arrival > travel * 0.9,
            "arrived at step {arrival}, faster than light ({travel} steps of travel)"
        );
        assert!(
            arrival < travel * 3.0,
            "arrived at step {arrival}, far too slow"
        );
    }

    #[test]
    fn a_perfect_conductor_expels_the_field() {
        use crate::{material::Material, scene::Shape};
        let extent = Extent::cube(40);
        let mut scene = pulse_scene(extent);
        let metal = scene.materials.push(Material::PERFECT_CONDUCTOR);
        scene.shapes.push(Shape::Block {
            min: [26, 10, 10],
            max: [34, 30, 30],
            material: metal,
        });

        let mut simulation = Simulation::new(&scene);
        simulation.advance_by(300);
        for z in 10..30 {
            for y in 10..30 {
                for x in 26..34 {
                    let value = simulation.sample_electric(Axis::Z, [x, y, z]);
                    assert_eq!(value, 0.0, "field leaked into the conductor at {x},{y},{z}");
                }
            }
        }
    }

    #[test]
    fn reset_returns_to_the_initial_state() {
        let mut simulation = Simulation::new(&pulse_scene(Extent::cube(24)));
        simulation.advance_by(50);
        assert!(simulation.energy().total() > 0.0);
        simulation.reset();
        assert_eq!(simulation.step_count(), 0);
        assert_eq!(simulation.energy().total(), 0.0);
    }

    #[test]
    fn energy_density_sums_to_the_total_energy() {
        let mut simulation = Simulation::new(&pulse_scene(Extent::cube(24)));
        simulation.advance_by(60);
        let summed: f64 = simulation
            .energy_density()
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        let total = simulation.energy().total();
        assert!((summed - total).abs() < total * 1e-4, "{summed} vs {total}");
    }
}

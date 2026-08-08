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
    timeline::{Snapshot, Steppable},
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

    /// Whether stepping backwards is numerically meaningful here.
    ///
    /// False as soon as anything in the scene is lossy — see [`Self::reverse`].
    pub fn is_reversible(&self) -> bool {
        let lossless = self
            .coefficients
            .iter()
            .all(|c| c.electric_loss == 0.0 && c.magnetic_loss == 0.0);
        lossless && self.absorber.peak() == 0.0
    }

    /// Undoes one step, exactly.
    ///
    /// # Leapfrog is an involution
    ///
    /// The forward step is
    ///
    /// ```text
    /// H ← ((1−m)·H − g_h·curl E) / (1+m)
    /// E ← ((1−l)·E + g_e·curl H) / (1+l)
    /// ```
    ///
    /// and undoing it means running the two in the opposite order with the
    /// algebra inverted:
    ///
    /// ```text
    /// E ← ((1+l)·E − g_e·curl H) / (1−l)
    /// H ← ((1+m)·H + g_h·curl E) / (1−m)
    /// ```
    ///
    /// With no loss that is the *same* arithmetic with the gains negated and
    /// the order swapped. In exact arithmetic the round trip is the identity;
    /// in `f32` it drifts by about `√N·ε`, because `a + b − b` is not `a`.
    /// Sources come back out exactly, since a waveform is a pure function of
    /// time.
    ///
    /// # Why loss forbids it
    ///
    /// The reverse step divides by `(1−l)`. In the absorbing layer the
    /// half-loss peaks near `0.69`, so reversing *amplifies* by more than 3×
    /// per step in the outermost cells — ten steps back is a factor of 10⁵ on
    /// whatever noise is there. A perfect conductor is worse still: `l = 1`
    /// exactly, so the forward map is singular and the information is not
    /// merely hard to recover, it is gone.
    ///
    /// Panics rather than returning garbage, because garbage here looks like
    /// an exponentially growing field and would be blamed on the solver.
    pub fn reverse(&mut self) {
        assert!(
            self.is_reversible(),
            "this scene is lossy, and running a dissipative update backwards \
             amplifies roundoff exponentially; reversal needs Boundary::Pec \
             and no conductive material"
        );
        assert!(self.step > 0, "cannot step back past the start");

        // Forward order is H, E, inject; so backward is un-inject, un-E, un-H.
        let time = self.step as f32 * self.grid.time_step();
        self.retract(time);
        self.reverse_electric();
        self.reverse_magnetic();
        self.step -= 1;
    }

    pub fn reverse_by(&mut self, steps: u64) {
        for _ in 0..steps {
            self.reverse();
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

    /// The inverse of [`Self::update_electric`], over the same cells.
    ///
    /// `E ← ((1+l)·E − gain·curl H) / (1−l)`. The loop bounds have to match
    /// the forward sweep exactly: a cell the forward step skipped and this one
    /// touched would not be undone, it would be corrupted.
    fn reverse_electric(&mut self) {
        let extent = self.grid.extent.as_array();
        let strides = self.grid.extent.strides();
        for axis in Axis::ALL {
            let (a, b, c) = (axis.index(), axis.next().index(), axis.prev().index());
            let (stride_b, stride_c) = (strides[b], strides[c]);
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
                        target[index] = ((1.0 + loss) * target[index]
                            - coefficients.electric_gain * curl)
                            / (1.0 - loss);
                    }
                }
            }
        }
    }

    /// The inverse of [`Self::update_magnetic`], over the same cells.
    fn reverse_magnetic(&mut self) {
        let extent = self.grid.extent.as_array();
        let strides = self.grid.extent.strides();
        for axis in Axis::ALL {
            let (a, b, c) = (axis.index(), axis.next().index(), axis.prev().index());
            let (stride_b, stride_c) = (strides[b], strides[c]);
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
                        target[index] = ((1.0 + loss) * target[index]
                            + coefficients.magnetic_gain * curl)
                            / (1.0 - loss);
                    }
                }
            }
        }
    }

    /// Adds every source into `E`. Additive, never assigning — see
    /// [`crate::source`].
    fn inject(&mut self, time: f32) {
        self.apply_sources(time, 1.0);
    }

    /// Takes them back out again, for [`Self::reverse`]. Exact, because a
    /// waveform is a pure function of time.
    fn retract(&mut self, time: f32) {
        self.apply_sources(time, -1.0);
    }

    fn apply_sources(&mut self, time: f32, sign: f32) {
        for source in self.sources.iter() {
            let injection = source.injection(&self.grid, time);
            if injection.value == 0.0 {
                continue;
            }
            let value = sign * injection.value;
            let target = &mut self.electric[injection.component];
            for dz in 0..injection.extent[2] {
                for dy in 0..injection.extent[1] {
                    for dx in 0..injection.extent[0] {
                        let coord = [
                            injection.origin[0] + dx,
                            injection.origin[1] + dy,
                            injection.origin[2] + dz,
                        ];
                        target[self.grid.extent.index(coord)] += value * injection.weight(coord);
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

impl Steppable for Simulation {
    fn step_count(&self) -> u64 {
        self.step
    }

    fn advance_by(&mut self, steps: u64) {
        Self::advance_by(self, steps);
    }

    fn reset(&mut self) {
        Self::reset(self);
    }

    fn snapshot(&mut self) -> Snapshot {
        Snapshot {
            step: self.step,
            electric: self.electric.concat(),
            magnetic: self.magnetic.concat(),
        }
    }

    fn restore(&mut self, snapshot: &Snapshot) {
        let cells = self.grid.extent.total();
        assert_eq!(
            snapshot.electric.len(),
            3 * cells,
            "snapshot was taken from a different grid"
        );
        for axis in 0..3 {
            let range = axis * cells..(axis + 1) * cells;
            self.electric[axis].copy_from_slice(&snapshot.electric[range.clone()]);
            self.magnetic[axis].copy_from_slice(&snapshot.magnetic[range]);
        }
        self.step = snapshot.step;
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
        scene.with_source(Source::point(
            [0.0; 3],
            Axis::Z,
            Waveform::ricker(frequency),
        ))
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
        // 40 cells at 1 mm spans -20..20 mm; cells 26..34 are 6..14 mm.
        scene.shapes.push(Shape::Block {
            center: [10e-3, 0.0, 0.0],
            size: [8e-3, 20e-3, 20e-3],
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
    fn stepping_back_undoes_stepping_forward() {
        // Leapfrog is an exact involution in a lossless box. What this checks
        // that the plane-wave test cannot is that the forward and reverse
        // sweeps cover exactly the same cells: a cell updated one way and not
        // the other would not be undone, it would be corrupted.
        let scene = pulse_scene(Extent::cube(32)).with_boundary(Boundary::Pec);
        let mut simulation = Simulation::new(&scene);
        assert!(simulation.is_reversible());

        simulation.advance_by(60);
        let before: Vec<f32> = Axis::ALL
            .iter()
            .flat_map(|&axis| simulation.electric(axis).iter().copied())
            .chain(
                Axis::ALL
                    .iter()
                    .flat_map(|&axis| simulation.magnetic(axis).iter().copied()),
            )
            .collect();
        let peak = before.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        assert!(peak > 0.0);

        simulation.advance_by(200);
        simulation.reverse_by(200);
        assert_eq!(simulation.step_count(), 60);

        let after: Vec<f32> = Axis::ALL
            .iter()
            .flat_map(|&axis| simulation.electric(axis).iter().copied())
            .chain(
                Axis::ALL
                    .iter()
                    .flat_map(|&axis| simulation.magnetic(axis).iter().copied()),
            )
            .collect();
        let worst = before
            .iter()
            .zip(after.iter())
            .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
        // 400 steps of f32 arithmetic; the drift is roundoff, not error.
        assert!(
            worst < 1e-4 * peak,
            "round trip drifted by {:e} against a peak of {peak:e}",
            worst / peak
        );
    }

    #[test]
    fn reversing_all_the_way_returns_to_an_empty_domain() {
        let scene = pulse_scene(Extent::cube(24)).with_boundary(Boundary::Pec);
        let mut simulation = Simulation::new(&scene);
        simulation.advance_by(120);
        let excited = simulation.energy().total();
        simulation.reverse_by(120);

        assert_eq!(simulation.step_count(), 0);
        // The source put the energy in; running time backwards takes it out
        // again, because injection is a pure function of the step number.
        let residue = simulation.energy().total();
        assert!(
            residue < excited * 1e-8,
            "{residue:e} left of {excited:e} after unwinding to t = 0"
        );
    }

    #[test]
    #[should_panic(expected = "lossy")]
    fn reversal_refuses_a_scene_with_an_absorber() {
        // The absorbing layer amplifies by more than 3x per step run
        // backwards. Returning an exponentially growing field would look like
        // a solver bug, so this refuses instead.
        let mut simulation = Simulation::new(&pulse_scene(Extent::cube(32)));
        assert!(!simulation.is_reversible());
        simulation.advance();
        simulation.reverse();
    }

    #[test]
    #[should_panic(expected = "lossy")]
    fn reversal_refuses_a_scene_with_a_conductor() {
        use crate::{material::Material, scene::Shape};
        let mut scene = pulse_scene(Extent::cube(32)).with_boundary(Boundary::Pec);
        let metal = scene.materials.push(Material::PERFECT_CONDUCTOR);
        scene.shapes.push(Shape::Block {
            center: [8e-3, 0.0, 0.0],
            size: [4e-3; 3],
            material: metal,
        });
        let mut simulation = Simulation::new(&scene);
        assert!(!simulation.is_reversible());
        simulation.advance();
        simulation.reverse();
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

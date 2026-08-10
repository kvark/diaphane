//! Analytic acceptance tests. No GPU required.
//!
//! The hard part of validating an FDTD solver is that a wrong one still looks
//! right: a half-cell offset in the stencil still produces a wave that expands
//! at roughly the correct speed and interferes plausibly. So every test here
//! compares against a closed-form prediction and requires a number, rather than
//! asserting that something looks wave-like.
//!
//! # The scheme does not propagate at c, and testing that it does would be wrong
//!
//! Yee's scheme has its own dispersion relation,
//!
//! ```text
//! sin²(ωΔt/2) = S² · Σ_a sin²(k_a Δ/2)
//! ```
//!
//! which reduces to `ω = ck` only as `Δ → 0`. At 20 cells per wavelength an
//! axial wave runs about 0.3% slow and a diagonal one is nearly exact. The
//! tests below therefore check the solver against the *discrete* physics. A
//! test that demanded `c` would either fail on a correct solver or pass with a
//! tolerance loose enough to hide real bugs.

use diaphane::{
    Axis, Boundary, Extent, Material, Scene, Shape, Source, Waveform, cpu,
    grid::{self, Grid},
};

/// An exact plane-wave solution of the discrete update equations.
///
/// This is not a discretized continuum solution — it is a solution of the
/// stepping scheme itself, so a correct solver reproduces it to `f32`
/// roundoff and forever. Constructing it requires committing to every
/// convention at once: which sample sits at which half-cell, which difference
/// is forward, and what the temporal offset between `E` and `H` is. That is
/// precisely why it is the sharpest test available.
struct PlaneWave {
    /// `k·Δ`, one component per axis.
    wavenumber: [f64; 3],
    /// `ω·Δt`, the phase advance per step.
    phase_rate: f64,
    electric_amplitude: [f64; 3],
    magnetic_amplitude: [f64; 3],
}

impl PlaneWave {
    /// Builds the wave travelling along `direction` with a spatial wavelength
    /// of `cells_per_wavelength` cells, in a medium of the given index.
    fn new(grid: &Grid, direction: [f64; 3], cells_per_wavelength: f64, index: f64) -> Self {
        let norm = direction.iter().map(|d| d * d).sum::<f64>().sqrt();
        let magnitude = 2.0 * std::f64::consts::PI / cells_per_wavelength;
        let wavenumber = direction.map(|d| magnitude * d / norm);

        // `s_a = sin(k_a Δ/2)` is the quantity the discrete curl actually
        // produces, and it plays the role `k` plays in the continuum. Note it
        // is not parallel to `k` off the grid axes, which is numerical
        // dispersion's anisotropy stated as a vector.
        let s = wavenumber.map(|q| (0.5 * q).sin());
        let s_norm = s.iter().map(|v| v * v).sum::<f64>().sqrt();

        // sin(ωΔt/2) = S·|s|/n, the dispersion relation solved for ω.
        let courant = f64::from(grid.courant);
        let phase_rate = 2.0 * (courant * s_norm / index).asin();

        // Any unit vector perpendicular to `s`; transversality in the discrete
        // scheme is with respect to `s`, not `k`. Crossing with the axis `s`
        // leans on least is what keeps the two from being parallel — which for
        // axial propagation they otherwise would be.
        let least = (0..3)
            .min_by(|&i, &j| s[i].abs().total_cmp(&s[j].abs()))
            .unwrap();
        let seed = std::array::from_fn(|axis| if axis == least { 1.0 } else { 0.0 });
        let electric_amplitude = normalize(cross(s, seed));
        assert!(
            electric_amplitude.iter().all(|v| v.is_finite()),
            "degenerate polarization basis for direction {direction:?}"
        );
        // σĤ = (S/μr)(s × Ê); with μr = 1 this makes |Ĥ| = n·|Ê|, which is the
        // free-space impedance relation in normalized units.
        let sigma = (0.5 * phase_rate).sin();
        let magnetic_amplitude = cross(s, electric_amplitude).map(|v| courant * v / sigma);

        Self {
            wavenumber,
            phase_rate,
            electric_amplitude,
            magnetic_amplitude,
        }
    }

    /// Phase velocity as a fraction of `c`: `ω/(kc)`.
    fn phase_velocity(&self, grid: &Grid) -> f64 {
        let k = self.wavenumber.iter().map(|q| q * q).sum::<f64>().sqrt();
        self.phase_rate / (f64::from(grid.courant) * k)
    }

    /// Free-space wavelength in cells corresponding to this wave's frequency,
    /// which is what [`grid::numerical_phase_velocity`] is parameterized by.
    fn free_space_cells(&self, grid: &Grid) -> f64 {
        2.0 * std::f64::consts::PI * f64::from(grid.courant) / self.phase_rate
    }

    /// `E[axis]` at a cell after `step` steps.
    ///
    /// `E[a]` sits half a cell along `a`, at integer time.
    fn electric(&self, axis: usize, coord: [usize; 3], step: f64) -> f64 {
        let phase = self.dot(coord) + 0.5 * self.wavenumber[axis];
        self.electric_amplitude[axis] * (self.phase_rate * step - phase).cos()
    }

    /// `H[axis]` at a cell as stored after `step` steps.
    ///
    /// `H[a]` sits half a cell along each of the *other* two axes, and the
    /// value the solver holds once `step` steps are done is the one for time
    /// `(step − ½)Δt`: `advance` updates `H` first, so it ends the step half a
    /// tick behind `E`.
    fn magnetic(&self, axis: usize, coord: [usize; 3], step: f64) -> f64 {
        let half_sum = 0.5 * self.wavenumber.iter().sum::<f64>();
        let phase = self.dot(coord) + half_sum - 0.5 * self.wavenumber[axis];
        self.magnetic_amplitude[axis] * (self.phase_rate * (step - 0.5) - phase).cos()
    }

    fn dot(&self, coord: [usize; 3]) -> f64 {
        (0..3).map(|a| self.wavenumber[a] * coord[a] as f64).sum()
    }

    /// Writes the wave into a simulation at `step = 0`.
    fn seed(&self, simulation: &mut cpu::Simulation) {
        let extent = simulation.grid().extent;
        let [nx, ny, nz] = extent.as_array();
        for axis in Axis::ALL {
            let a = axis.index();
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let index = extent.index([x, y, z]);
                        simulation.electric_mut(axis)[index] =
                            self.electric(a, [x, y, z], 0.0) as f32;
                        simulation.magnetic_mut(axis)[index] =
                            self.magnetic(a, [x, y, z], 0.0) as f32;
                    }
                }
            }
        }
    }

    /// Worst disagreement between the solver and this solution, over the cells
    /// far enough from the walls that the boundary has not reached them.
    ///
    /// The seeded wave is exact in the interior but wrong at the clamped
    /// boundary planes, and that error walks inwards one cell per step —
    /// the stencil's domain of dependence is exactly one cell.
    fn worst_error(&self, simulation: &cpu::Simulation, margin: usize) -> f64 {
        let extent = simulation.grid().extent;
        let [nx, ny, nz] = extent.as_array();
        let step = simulation.step_count() as f64;
        let mut worst = 0.0f64;
        for z in margin..nz - margin {
            for y in margin..ny - margin {
                for x in margin..nx - margin {
                    for axis in Axis::ALL {
                        let a = axis.index();
                        let electric = f64::from(simulation.sample_electric(axis, [x, y, z]));
                        worst = worst.max((electric - self.electric(a, [x, y, z], step)).abs());
                        let magnetic = f64::from(simulation.sample_magnetic(axis, [x, y, z]));
                        worst = worst.max((magnetic - self.magnetic(a, [x, y, z], step)).abs());
                    }
                }
            }
        }
        worst
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let norm = v.iter().map(|c| c * c).sum::<f64>().sqrt();
    v.map(|c| c / norm)
}

/// Seeds an exact discrete plane wave, steps, and requires the solver to have
/// reproduced it.
fn check_plane_wave(direction: [f64; 3], cells_per_wavelength: f64, permittivity: f32) -> f64 {
    const SIDE: u32 = 96;
    const STEPS: u64 = 30;

    let mut scene = Scene::empty(Extent::cube(SIDE), 1e-3).with_boundary(Boundary::Pec);
    if permittivity != 1.0 {
        let material = scene.materials.push(Material::dielectric(permittivity));
        // Fills the whole domain: an unbounded slab thicker than the box.
        scene.shapes.push(Shape::Slab {
            axis: Axis::X,
            offset: 0.0,
            thickness: f32::INFINITY,
            material,
        });
    }

    let index = f64::from(permittivity).sqrt();
    let wave = PlaneWave::new(&scene.grid, direction, cells_per_wavelength, index);
    let mut simulation = cpu::Simulation::new(&scene);
    wave.seed(&mut simulation);
    simulation.advance_by(STEPS);

    let error = wave.worst_error(&simulation, STEPS as usize + 2);
    let peak = wave
        .magnetic_amplitude
        .iter()
        .chain(wave.electric_amplitude.iter())
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    assert!(
        error < 1e-4 * peak,
        "{direction:?} at {cells_per_wavelength} cells/λ in εr={permittivity}: \
         worst error {error:e} against a peak of {peak:e}",
    );
    wave.phase_velocity(&scene.grid)
}

#[test]
fn an_exact_plane_wave_propagates_exactly() {
    // Axial, body diagonal, and an oblique direction where `s` is genuinely
    // not parallel to `k` — the case a solver that quietly assumes isotropy
    // would get wrong.
    for direction in [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 0.0],
    ] {
        for cells in [10.0, 20.0] {
            check_plane_wave(direction, cells, 1.0);
        }
    }
}

#[test]
fn measured_phase_velocity_matches_the_dispersion_relation() {
    let grid = Grid::new(Extent::cube(96), 1e-3);
    for direction in [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0]] {
        for cells in [10.0, 20.0, 40.0] {
            let wave = PlaneWave::new(&grid, direction, cells, 1.0);
            let measured = wave.phase_velocity(&grid);
            // `numerical_phase_velocity` starts from the frequency and solves
            // for the wavenumber; `PlaneWave` starts from the wavenumber and
            // solves for the frequency. Agreeing means the same relation was
            // inverted correctly in both directions.
            let predicted = f64::from(grid::numerical_phase_velocity(
                grid.courant,
                wave.free_space_cells(&grid) as f32,
                direction.map(|d| d as f32),
            ));
            assert!(
                (measured - predicted).abs() < 1e-4,
                "{direction:?} at {cells} cells/λ: measured {measured}, predicted {predicted}"
            );
            assert!(measured < 1.0, "the scheme cannot be superluminal");
        }
    }
}

#[test]
fn dispersion_is_anisotropic_and_shrinks_with_refinement() {
    let grid = Grid::new(Extent::cube(96), 1e-3);
    let velocity = |direction: [f64; 3], cells: f64| {
        PlaneWave::new(&grid, direction, cells, 1.0).phase_velocity(&grid)
    };
    // The signature of Yee dispersion: axial waves lag most, diagonal least.
    let axial = velocity([1.0, 0.0, 0.0], 20.0);
    let diagonal = velocity([1.0, 1.0, 1.0], 20.0);
    assert!(
        axial < diagonal,
        "axial {axial} should lag diagonal {diagonal}"
    );
    assert!(
        (1.0 - axial) < 0.005,
        "axial error at 20 cells/λ is {:.3}%, worse than the 0.5% the brief asks for",
        100.0 * (1.0 - axial)
    );
    // Second order in Δ/λ: halving the cell size should quarter the error.
    let coarse = 1.0 - velocity([1.0, 0.0, 0.0], 10.0);
    let fine = 1.0 - velocity([1.0, 0.0, 0.0], 20.0);
    let ratio = coarse / fine;
    assert!(
        (3.5..4.5).contains(&ratio),
        "refinement improved the error by {ratio:.2}×, expected ~4× for O((Δ/λ)²)"
    );
}

#[test]
fn a_dielectric_slows_the_wave_by_its_refractive_index() {
    // Same machinery, but the whole domain is glass. This puts the material
    // coefficient path on an exact analytic check rather than a visual one.
    for permittivity in [2.25f32, 4.0, 9.0] {
        let index = f64::from(permittivity).sqrt();
        let velocity = check_plane_wave([1.0, 0.0, 0.0], 20.0 * index, permittivity);
        let expected = 1.0 / index;
        // 20 cells per wavelength *inside the medium*, so the residual
        // discretization error is the same fraction as in vacuum.
        assert!(
            (velocity - expected).abs() < 0.005 * expected,
            "εr={permittivity}: phase velocity {velocity:.5}, expected about {expected:.5}"
        );
    }
}

#[test]
fn the_test_would_notice_a_wrong_dispersion_relation() {
    // A test that passes is only worth something if it can fail. Seed the wave
    // with the ideal ω = ck instead of the numerical one and confirm the
    // solver visibly disagrees — otherwise the check above is measuring
    // nothing.
    const SIDE: u32 = 96;
    const STEPS: u64 = 30;
    let scene = Scene::empty(Extent::cube(SIDE), 1e-3).with_boundary(Boundary::Pec);
    let mut wave = PlaneWave::new(&scene.grid, [1.0, 0.0, 0.0], 10.0, 1.0);

    let honest = wave.phase_rate;
    let wavenumber = wave.wavenumber.iter().map(|q| q * q).sum::<f64>().sqrt();
    wave.phase_rate = f64::from(scene.grid.courant) * wavenumber;
    assert!(wave.phase_rate > honest, "the ideal wave should be faster");

    let mut simulation = cpu::Simulation::new(&scene);
    wave.seed(&mut simulation);
    simulation.advance_by(STEPS);
    let error = wave.worst_error(&simulation, STEPS as usize + 2);
    assert!(
        error > 0.05,
        "seeding the wrong frequency only moved the field by {error:e}; \
         the plane-wave test cannot be detecting anything"
    );
}

#[test]
fn a_closed_conducting_box_conserves_energy() {
    // The brief's canary: a lossless PEC cavity should hold its energy
    // indefinitely. What is being ruled out is secular drift — a scheme with
    // eigenvalues even slightly off the unit circle grows or decays
    // exponentially, and over enough steps that is unmistakable.
    const STEPS: u64 = 40_000;
    let scene = Scene::cavity(Extent::cube(36));
    let mut simulation = cpu::Simulation::new(&scene);
    simulation.advance_by(400);

    // Averaged over a window, because the instantaneous sum mixes E at time n
    // with H at n−½ and therefore wobbles by order ωΔt every step. See
    // `cpu::Energy`.
    let mean_energy = |simulation: &mut cpu::Simulation| {
        let mut sum = 0.0;
        for _ in 0..200 {
            simulation.advance();
            sum += simulation.energy().total();
        }
        sum / 200.0
    };

    let early = mean_energy(&mut simulation);
    simulation.advance_by(STEPS);
    let late = mean_energy(&mut simulation);

    assert!(simulation.is_finite());
    let drift = (late - early).abs() / early;
    assert!(
        drift < 1e-3,
        "energy drifted by {drift:e} over {STEPS} steps ({early:e} → {late:e})"
    );
}

#[test]
fn a_plane_wave_source_is_one_way() {
    // The total-field/scattered-field corrections replay the incident wave at
    // the grid's own numerical phase velocity, so everything radiated
    // backward should cancel. "Should" is measured here: the wave crosses the
    // total region at full strength while the scattered region behind the
    // injection plane stays quiet.
    //
    // What the floor actually is: aperture physics, not injector error. The
    // correction rows span a finite cross-section truncated by the side
    // absorbers, and a truncated plane wave diffracts -- the grazing side
    // layers scatter the incident wave continuously, and some of that
    // radiates backward through the plane. That backwash is real scattered
    // field of the finite configuration, measured here around -30 dB. The
    // injector's own cancellation sits below it; the measurement is sensitive
    // enough to know, because deferring the magnetic correction by half a
    // step -- the mistake this test caught -- raised the floor to -21 dB.
    let mut scene = Scene::empty(Extent::new(120, 32, 32), 1e-3);
    let frequency = scene.grid.frequency_for_resolution(20.0);
    scene.sources.push(Source::plane_wave(
        Axis::X,
        -0.020,
        Axis::Z,
        Waveform::gaussian_pulse(frequency, 4.0),
    ));
    scene.validate().unwrap();

    let mut simulation = cpu::Simulation::new(&scene);
    let plane = 40usize; // cell of x = -20 mm in a 120-cell axis
    let mut total_peak = 0.0f32;
    let mut scattered_peak = 0.0f32;
    for _ in 0..900 {
        simulation.advance();
        let ez = simulation.electric(Axis::Z);
        // On-axis columns, ten cells clear of the plane on each side and of
        // the absorbing walls, so neither region's number is edge backwash.
        for x in 12..plane - 10 {
            scattered_peak = scattered_peak.max(ez[scene.grid.extent.index([x, 16, 16])].abs());
        }
        for x in plane + 10..108 {
            total_peak = total_peak.max(ez[scene.grid.extent.index([x, 16, 16])].abs());
        }
    }

    // The launched wave is the waveform at unit amplitude, within what
    // diffraction of the finite aperture does to an on-axis measurement.
    assert!(
        (0.8..1.25).contains(&total_peak),
        "incident peak {total_peak} is not the waveform's"
    );
    let leak = 20.0 * (scattered_peak / total_peak).log10();
    assert!(
        leak < -25.0,
        "the plane wave leaks {leak:.1} dB backward ({scattered_peak:e} of {total_peak:e})"
    );
}

#[test]
fn a_conducting_box_cannot_tell_its_low_walls_from_its_high_ones() {
    // A pulse launched from the exact centre of a cubic PEC box has no way to
    // tell the wall at -L/2 from the wall at +L/2 -- unless the two implement
    // different boundary conditions. They used to: the low faces froze
    // tangential E (a true electric conductor), while the high faces froze
    // tangential H half a cell inside the wall, which is a perfect *magnetic*
    // conductor. Both are lossless, so the energy test above is blind to it;
    // what gives it away is the echo, which returns sign-flipped from an
    // electric wall and sign-preserved from a magnetic one. Mirror symmetry
    // of the field is therefore the assertion: it survives many bounces on
    // matched walls and is destroyed by the first bounce on mismatched ones.
    let extent = 24usize;
    // 24 cells put a lattice line through the exact centre, where Ez sits at
    // integer x and y -- so a z-polarized dipole there is x- and y-mirror
    // symmetric on the lattice, not merely in the continuum.
    let scene = Scene::empty(Extent::cube(extent as u32), 1e-3).with_boundary(Boundary::Pec);
    let frequency = scene.grid.frequency_for_resolution(12.0);
    let scene = scene.with_source(Source::point(
        [0.0; 3],
        Axis::Z,
        Waveform::ricker(frequency),
    ));
    let mut simulation = cpu::Simulation::new(&scene);
    // c·Δt is half a cell, so the wavefront reaches a wall in about 24
    // steps; this is several round trips off every wall.
    simulation.advance_by(160);
    assert!(simulation.is_finite());

    let ez = simulation.electric(Axis::Z);
    let index = |x: usize, y: usize, z: usize| (z * extent + y) * extent + x;
    let peak = ez.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    assert!(peak > 0.0);

    let mut worst = 0.0f32;
    for z in 0..extent {
        for y in 0..extent {
            // Ez on the low wall is the wall: exactly zero, forever.
            assert_eq!(ez[index(0, y, z)], 0.0);
            // x = i mirrors to x = N - i through the centre lattice line.
            for x in 1..extent {
                let asymmetry = (ez[index(x, y, z)] - ez[index(extent - x, y, z)]).abs();
                worst = worst.max(asymmetry);
            }
        }
    }
    assert!(
        worst < 1e-6 * peak,
        "left/right asymmetry of {:e} of the peak: the walls differ",
        worst / peak
    );
}

/// Records a probe trace for a domain of the given length along x, with
/// everything else held fixed.
fn probe_trace(length: u32, steps: u64) -> Vec<f32> {
    const TRANSVERSE: u32 = 40;
    const CELL: f32 = 1e-3;
    const SOURCE_X: u32 = 20;
    const PROBE_X: u32 = 36;

    let mut scene = Scene::empty(Extent::new(length, TRANSVERSE, TRANSVERSE), CELL);
    let frequency = scene.grid.frequency_for_resolution(20.0);
    // Source and probe are pinned to cell indices in both domains, so the two
    // runs stay comparable when only the far wall moves. Converting through
    // the grid keeps that true now that positions are metres.
    let source_position = scene.grid.to_position([
        SOURCE_X as f32 + 0.5,
        0.5 * TRANSVERSE as f32,
        0.5 * TRANSVERSE as f32,
    ]);
    scene.sources.push(Source::point(
        source_position,
        Axis::Z,
        Waveform::ricker(frequency),
    ));
    let mut simulation = cpu::Simulation::new(&scene);
    (0..steps)
        .map(|_| {
            simulation.advance();
            simulation.sample_electric(
                Axis::Z,
                [
                    PROBE_X as usize,
                    TRANSVERSE as usize / 2,
                    TRANSVERSE as usize / 2,
                ],
            )
        })
        .collect()
}

#[test]
fn the_absorbing_layer_reflects_below_the_stated_level() {
    // Measured against an oversized reference domain rather than eyeballed,
    // because a subtly wrong absorbing layer produces plausible-looking
    // results and a number is the only thing that catches it.
    //
    // Both domains are identical except for how much empty space sits beyond
    // the probe, so their traces differ only by what the near wall sent back.
    // The window stops before the reference domain's own far wall can answer.
    const STEPS: u64 = 380;
    let near = probe_trace(72, STEPS);
    let reference = probe_trace(200, STEPS);

    let incident = reference.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let reflected = near
        .iter()
        .zip(reference.iter())
        .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    let decibels = 20.0 * (reflected / incident).log10();

    println!("absorber reflection: {decibels:.1} dB ({reflected:e} of {incident:e})");
    // Measures about −58 dB; the threshold leaves room for a different
    // rounding order without leaving room for the layer to quietly regress.
    assert!(
        decibels < -50.0,
        "absorbing layer reflects at {decibels:.1} dB, above the -50 dB it claims"
    );
}

#[test]
fn a_propagating_packet_splits_its_energy_evenly() {
    // In a travelling wave the electric and magnetic energies are equal — that
    // is what having a real impedance means. In the standing wave of a cavity
    // they alternate instead, which `cpu` tests separately. Both statements
    // together are the physics the visualizer exists to show.
    let scene = Scene::photon(Extent::new(80, 48, 48));
    let mut simulation = cpu::Simulation::new(&scene);
    simulation.advance_by(140);

    let mut worst: f64 = 0.0;
    for _ in 0..60 {
        simulation.advance();
        let energy = simulation.energy();
        worst = worst.max((energy.electric - energy.magnetic).abs() / energy.total());
    }
    assert!(
        worst < 0.1,
        "the two halves differ by up to {:.1}% in a travelling wave",
        100.0 * worst
    );
}

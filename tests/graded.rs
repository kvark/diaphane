//! What a graded grid has to prove before it is worth having.
//!
//! Non-uniform cells are the cheap end of adaptive refinement: no patches, no
//! interfaces, no interpolation, and therefore none of the late-time
//! instability that subgridding is famous for. "Therefore" is the part that has
//! to be *measured*, so this file does three things a uniform grid cannot:
//!
//! 1. the answer does not depend on where the grading is
//! 2. a change of spacing reflects, and by how much
//! 3. a lossless box still holds its energy over tens of thousands of steps
//!
//! The rest of the suite covers grading by being run at all: every analytic
//! check in `validation.rs` goes through the same per-axis spacing machinery,
//! on a `Spacing` whose widths happen to be equal.

use diaphane::{Axis, Extent, Refinement, Scene, Source, Waveform, cpu, grid::Grid};

const CELL: f32 = 1e-3;
const DOMAIN: f32 = 0.096;
/// Four times finer inside the band, which is the ratio a refinement is worth
/// asking for: below it a uniform grid is simpler, above it the cell count
/// starts to matter more than the accuracy.
const FINE: f32 = 0.25e-3;

/// A band across the middle of x, leaving y and z alone.
///
/// `Refinement::across` rather than a box, and the difference is not cosmetic:
/// refinement is a tensor product, so giving this a transverse size would
/// refine the whole y–z plane to `cell_size` as well. At the sizes here that is
/// 19 million cells instead of 1.2 million, for a field that is not varying in
/// those directions at all.
fn band(half_width: f32, cell_size: f32) -> Refinement {
    Refinement::across(Axis::X, 0.0, 2.0 * half_width, cell_size)
}

#[test]
fn a_uniform_grid_is_a_spacing_whose_widths_are_equal() {
    // The claim the whole design rests on: uniform is not a special case, so
    // there is one code path and the graded one cannot rot. That is only safe
    // if uniform comes out *exactly* as it did before there was such a thing
    // as a spacing.
    let grid = Grid::new(Extent::new(40, 60, 80), CELL);
    assert!(grid.is_uniform());
    assert_eq!(grid.worst_ratio(), 1.0);
    for axis in Axis::ALL {
        let spacing = grid.spacing(axis);
        assert!(spacing.primary().iter().all(|&w| w == CELL));
        // Primary and dual coincide only because the cells are equal. This is
        // the assertion a graded grid is not allowed to satisfy.
        assert_eq!(spacing.primary(), spacing.dual());
        for gain in grid.electric_gains(axis) {
            assert_eq!(gain, grid.courant);
        }
    }
    assert_eq!(grid.reference_cell_size(), CELL);
    // The domain centre lands exactly on a corner, to the bit, which is what
    // makes a centred origin mean anything.
    assert_eq!(grid.to_cell([0.0; 3]), [20.0, 30.0, 40.0]);
    assert_eq!(grid.to_position([20.0, 30.0, 40.0]), [0.0; 3]);
}

#[test]
fn grading_respects_the_growth_cap_and_the_walls() {
    let grid = Grid::graded([DOMAIN; 3], 1.0 / CELL, vec![band(4e-3, FINE)]);
    grid.validate();
    assert!(!grid.is_uniform());

    // The cap is a contract, not an observation: the accuracy argument for a
    // graded grid is precisely that neighbouring cells are nearly equal, so a
    // grid that quietly exceeded it would be second-order in name only.
    assert!(
        grid.worst_ratio() <= 1.15 + 1e-5,
        "neighbouring cells grew by {}",
        grid.worst_ratio()
    );
    // Refinements subdivide; they never move a wall.
    for size in grid.size() {
        assert!((size - DOMAIN).abs() < 1e-6, "domain became {size}");
    }
    assert!(grid.finest() < 1.2 * FINE && grid.coarsest() > 0.9 * CELL);

    // Every cell centre still lands back in its own cell — the graded version
    // of the half-cell error that looks like nothing.
    let extent = grid.extent.as_array();
    for fraction in [0.0, 0.1, 0.5, 0.9] {
        let coord = std::array::from_fn(|axis| ((extent[axis] - 1) as f32 * fraction) as usize);
        assert_eq!(grid.cell_containing(grid.cell_center(coord)), coord);
    }
}

#[test]
fn grading_buys_cells_rather_than_accuracy() {
    // The reason to do any of this. The time step is set by the finest cell
    // either way, so what refinement saves is *cells* — and only on the axes it
    // touches, which is why the comparison here is against the same domain
    // resolved finely along x alone rather than against a uniformly fine cube.
    // Comparing against the cube would report a much larger number by counting
    // a saving on two axes nothing asked to refine.
    let graded = Grid::graded([DOMAIN; 3], 1.0 / CELL, vec![band(4e-3, FINE)]);
    let uniform = (DOMAIN / FINE).round();
    let saving = f64::from(uniform) / f64::from(graded.extent.x);
    println!(
        "graded {:?}: {} cells along x where uniform-fine needs {uniform}: {saving:.1}x",
        graded.extent, graded.extent.x,
    );
    // The ceiling is domain over band — 12x here — and the transition eats into
    // it: relaxing 4x at 15% per cell takes ten cells on each side, and those
    // are cells the band did not ask for. What is left is the honest figure.
    assert!(saving > 2.5, "grading only saved {saving:.1}x along x");
    // The transverse axes must be untouched, or the saving was imaginary.
    assert_eq!(graded.extent.y, (DOMAIN / CELL).round() as u32);
    assert_eq!(graded.extent.z, (DOMAIN / CELL).round() as u32);
    // And the same finest cell, so the band is resolved alike either way.
    assert!((graded.finest() / FINE - 1.0).abs() < 0.05);
}

/// Probe trace along x, with the source and the probe at fixed *physical*
/// positions so grids of different cell counts stay comparable.
fn trace(grid: Grid, steps: u64) -> Vec<f32> {
    const SOURCE_X: f32 = -0.012;
    const PROBE_X: f32 = -0.004;

    let mut scene = Scene::on_grid(grid);
    let frequency = scene.grid.frequency_for_resolution(20.0);
    scene.sources.push(Source::point(
        [SOURCE_X, 0.0, 0.0],
        Axis::Z,
        Waveform::ricker(frequency),
    ));
    let probe = scene.grid.cell_containing([PROBE_X, 0.0, 0.0]);
    let mut simulation = cpu::Simulation::new(&scene);
    (0..steps)
        .map(|_| {
            simulation.advance();
            simulation.sample_electric(Axis::Z, probe)
        })
        .collect()
}

/// Resamples `trace` onto `reference`'s time base.
///
/// Two grids with different finest cells have different time steps, so the
/// traces are not sample-for-sample comparable — the physics is a function of
/// *time*, and only reading it that way compares like with like.
fn resample(trace: &[f32], from: f32, onto: usize, to: f32) -> Vec<f32> {
    (0..onto)
        .map(|index| {
            let position = index as f32 * to / from;
            let low = position.floor() as usize;
            let fraction = position - low as f32;
            let sample = |at: usize| trace.get(at).copied().unwrap_or(0.0);
            sample(low) * (1.0 - fraction) + sample(low + 1) * fraction
        })
        .collect()
}

#[test]
fn a_graded_grid_gives_the_same_answer_as_a_uniformly_fine_one() {
    // Source and probe both sit inside the refined band, so what is being
    // tested is whether the *grading* corrupts the answer — not whether a
    // coarse region has different numerical dispersion, which it provably does
    // and which is the whole reason to refine.
    const STEPS: u64 = 300;
    let graded = Grid::graded([DOMAIN; 3], 1.0 / CELL, vec![band(0.02, FINE)]);
    let fine = Grid::for_size([DOMAIN; 3], 1.0 / FINE);
    let (graded_step, fine_step) = (graded.time_step(), fine.time_step());

    let measured = trace(graded, STEPS);
    let expected = trace(fine, (STEPS as f32 * graded_step / fine_step) as u64 + 2);
    let expected = resample(&expected, fine_step, measured.len(), graded_step);

    let peak = expected.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let worst = measured
        .iter()
        .zip(&expected)
        .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    println!(
        "graded vs uniformly fine: {:.1}% of peak ({worst:e} of {peak:e})",
        100.0 * worst / peak
    );
    assert!(peak > 0.0, "the reference trace never saw the pulse");
    assert!(
        worst < 0.06 * peak,
        "graded run differs from uniformly fine by {:.1}% of peak",
        100.0 * worst / peak
    );
}

#[test]
fn a_grading_transition_reflects_below_the_absorbing_layer() {
    // The number that decides whether grading is worth having. Every change of
    // spacing is a change of numerical phase velocity, and a wave crossing one
    // partially reflects off nothing physical at all.
    //
    // Measured the way the absorbing layer is: against a reference that is
    // identical except for having no transition to reflect from, so the
    // difference between the traces *is* what came back. The window closes
    // before either far wall can answer.
    //
    // The bar is the absorbing layer's own −58 dB. A grading that reflected
    // more than the walls do would make the refinement the loudest artifact in
    // the domain, which is the failure mode that makes subgridding hard.
    const STEPS: u64 = 340;
    let fine = Grid::for_size([DOMAIN; 3], 1.0 / FINE);
    // The band ends at x = −2 mm, between the probe and the far wall, so the
    // transition is the first thing the pulse meets after passing the probe.
    let graded = Grid::graded([DOMAIN; 3], 1.0 / CELL, vec![band(0.014, FINE)]);
    let step = graded.time_step();
    assert!(
        (step / fine.time_step() - 1.0).abs() < 0.05,
        "the two runs must share a time base for the difference to mean anything"
    );

    let measured = trace(graded, STEPS);
    let reference = trace(fine, STEPS);
    let incident = reference.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let reflected = measured
        .iter()
        .zip(&reference)
        .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    let decibels = 20.0 * (reflected / incident).log10();

    println!("grading reflection: {decibels:.1} dB ({reflected:e} of {incident:e})");
    assert!(incident > 0.0);
    assert!(
        decibels < -40.0,
        "a grading transition reflected {decibels:.1} dB, which is louder than the walls"
    );
}

#[test]
fn a_graded_conducting_box_conserves_energy() {
    // The instability canary, and the reason a graded *dense* grid is worth
    // preferring to patches: the scheme stays a plain symmetric leapfrog, so
    // its eigenvalues stay on the unit circle and nothing pumps energy at a
    // change of spacing. Subgridding loses exactly this, and loses it slowly
    // enough that only a long run shows it.
    const STEPS: u64 = 20_000;
    let grid = Grid::graded([0.036; 3], 1000.0, vec![band(6e-3, 0.4e-3)]);
    let scene = Scene::cavity_on(grid);
    let mut simulation = cpu::Simulation::new(&scene);
    simulation.advance_by(400);

    // Averaged over a window: the instantaneous sum mixes E at time n with H at
    // n−½ and wobbles by order ωΔt every step. See `cpu::Energy`.
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
    println!("graded cavity drift over {STEPS} steps: {drift:e}");
    assert!(
        drift < 1e-3,
        "energy drifted by {drift:e} over {STEPS} steps ({early:e} → {late:e})"
    );
}

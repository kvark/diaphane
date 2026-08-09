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
/// Long along x, narrow across it.
///
/// Everything measured here is a trace along x, so the transverse extent only
/// has to hold the absorbing layer and a little room. Making the domain a cube
/// would multiply every run by six for nothing — and these runs include a
/// reference grid four times finer along x and a cavity stepped twenty
/// thousand times.
const DOMAIN: f32 = 0.096;
const TRANSVERSE: f32 = 0.040;

fn domain() -> [f32; 3] {
    [DOMAIN, TRANSVERSE, TRANSVERSE]
}
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
fn stepping_back_survives_a_graded_grid() {
    // The reverse sweeps duplicate the forward sweeps' per-cell gain and
    // wall handling by hand, and a uniform grid cannot referee the copy:
    // every gain it would check is the same Courant constant. A graded
    // cavity makes the two plumbings disagree if they differ anywhere.
    let scene = Scene::cavity_on(Grid::graded(
        [0.040, 0.028, 0.028],
        1.0 / CELL,
        vec![band(0.004, FINE)],
    ));
    let mut simulation = cpu::Simulation::new(&scene);
    simulation.advance_by(200);
    let fingerprint = |simulation: &cpu::Simulation| -> Vec<f32> {
        Axis::ALL
            .iter()
            .flat_map(|&axis| simulation.electric(axis).iter().copied())
            .collect()
    };
    let marked = fingerprint(&simulation);
    let excited = simulation.energy().total();

    simulation.advance_by(150);
    simulation.reverse_by(150);
    let returned = fingerprint(&simulation);
    let peak = marked.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let worst = marked
        .iter()
        .zip(returned.iter())
        .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(peak > 0.0);
    assert!(
        worst < 1e-4 * peak,
        "reversal missed by {:e} of the peak on a graded grid",
        worst / peak
    );

    // All the way back: the source retracts exactly, so an empty domain
    // should be what remains.
    simulation.reverse_by(200);
    let residual = simulation.energy().total();
    assert!(
        residual < 1e-8 * excited,
        "residual energy {residual:e} of {excited:e} after unwinding to step 0"
    );
}

/// The controlled reference: fine along x *everywhere*, transverse
/// discretization identical to the graded grid's.
///
/// A uniformly fine cube would be the obvious reference and the wrong one. It
/// refines two axes the graded grid deliberately leaves alone, so a difference
/// between the traces would confound "the grading corrupted the answer" with
/// "the transverse mesh changed" — and it costs sixteen times the cells.
fn uniformly_fine() -> Grid {
    Grid::graded(
        domain(),
        1.0 / CELL,
        vec![Refinement::across(Axis::X, 0.0, 2.0 * DOMAIN, FINE)],
    )
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
    let grid = Grid::graded(domain(), 1.0 / CELL, vec![band(4e-3, FINE)]);
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
    for (axis, size) in grid.size().iter().enumerate() {
        assert!(
            (size - domain()[axis]).abs() < 1e-6,
            "axis {axis} became {size} m"
        );
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
    let graded = Grid::graded(domain(), 1.0 / CELL, vec![band(4e-3, FINE)]);
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
    assert_eq!(graded.extent.y, (TRANSVERSE / CELL).round() as u32);
    assert_eq!(graded.extent.z, (TRANSVERSE / CELL).round() as u32);
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
    let graded = Grid::graded(domain(), 1.0 / CELL, vec![band(0.02, FINE)]);
    let fine = uniformly_fine();
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

/// Reflection off a grading transition, in dB against a reference with no
/// transition to reflect from.
fn grading_reflection(max_ratio: f32) -> f32 {
    const STEPS: u64 = 340;
    // The band ends at x = −2 mm, between the probe and the far wall, so the
    // transition is the first thing the pulse meets after passing the probe.
    let graded = Grid::graded_at(domain(), 1.0 / CELL, vec![band(0.014, FINE)], max_ratio);
    let measured = trace(graded, STEPS);
    let reference = trace(uniformly_fine(), STEPS);
    let incident = reference.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let reflected = measured
        .iter()
        .zip(&reference)
        .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(incident > 0.0, "the reference trace never saw the pulse");
    20.0 * (reflected / incident).log10()
}

#[test]
fn a_grading_transition_reflects_and_grading_gently_reflects_less() {
    // The honest cost of a graded grid, and the one thing about it that is
    // worse than a uniform one. Every change of spacing is a change of
    // numerical phase velocity, so a wave crossing it partially reflects off
    // nothing physical at all.
    //
    // Measured the way the absorbing layer is: against a reference identical
    // except for having no transition, so the difference between the traces
    // *is* what came back. The window closes before either far wall answers.
    let default = grading_reflection(1.15);
    let gentle = grading_reflection(1.05);
    println!("grading reflection: {default:.1} dB at 1.15, {gentle:.1} dB at 1.05");

    // Reflection is set by how fast the spacing changes, so it is a knob and
    // not a fact. Asserting the *relationship* rather than a threshold is what
    // makes this a test of the mechanism: if grading gently stopped helping,
    // the advice in the docs would be wrong even if the number still passed.
    assert!(
        gentle < default - 3.0,
        "grading at 1.05 should reflect audibly less than at 1.15, got \
         {gentle:.1} dB against {default:.1} dB"
    );
    // And a floor, so a regression that made both worse together is still
    // caught. This is the number to keep an eye on: the absorbing layer
    // measures −58 dB, so at the default cap a refinement boundary is about
    // 25 dB louder than the walls of the domain. That is the price of grading
    // and it is why the cap is a scene-level knob.
    assert!(default < -30.0, "grading reflected {default:.1} dB");
}

#[test]
fn a_graded_conducting_box_conserves_energy() {
    // The instability canary, and the reason a graded *dense* grid is worth
    // preferring to patches: the scheme stays a plain symmetric leapfrog, so
    // its eigenvalues stay on the unit circle and nothing pumps energy at a
    // change of spacing. Subgridding loses exactly this, and loses it slowly
    // enough that only a long run shows it.
    const STEPS: u64 = 20_000;
    let grid = Grid::graded([0.024; 3], 1000.0, vec![band(4e-3, 0.4e-3)]);
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

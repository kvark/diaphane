//! The GPU kernels against the reference solver.
//!
//! An FDTD bug that is a half-cell offset still produces a wave that expands at
//! roughly the right speed and looks entirely convincing. Comparing against an
//! independently written implementation of the same equations is the check that
//! does not care how plausible the picture is.
//!
//! # Why the tolerance is not zero
//!
//! Both solvers do the same arithmetic in `f32`, but not in the same order, and
//! the GPU is free to contract multiply-adds into FMAs while the CPU may not.
//! Each step therefore differs in the last bit or two, and leapfrog carries
//! those differences forward. Over the few hundred steps run here the
//! divergence stays around `1e-5` relative, which is far below the `1e-2`
//! discretization error of the scheme itself but far above zero. Demanding bit
//! equality would mean pinning the shader compiler, which is not a promise
//! worth making.
//!
//! Skips itself if the machine has no usable device. On a headless Linux box
//! `mesa-vulkan-drivers` provides lavapipe, which is slow but real, and is what
//! CI runs against.

use diaphane::{
    Axis, Boundary, Extent, Material, Scene, Shape, Source, Waveform, cpu,
    gpu::{self, headless_context},
};

fn peak_of(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()))
}

fn worst_difference(reference: &[f32], candidate: &[f32]) -> f32 {
    assert_eq!(reference.len(), candidate.len());
    reference
        .iter()
        .zip(candidate.iter())
        .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()))
}

/// Compares one field, scaling the disagreement by the peak across *all three*
/// components rather than by each component's own peak.
///
/// A symmetric source leaves one component orders of magnitude below the
/// others -- a z-directed dipole radiates a purely azimuthal `H`, so `Hz` is
/// nothing but weak scattering. Scaling such a component by its own peak
/// divides noise by noise and reports a large relative error for a difference
/// that is physically nil. The meaningful question is how far apart the two
/// solvers are compared with the size of the field that is actually there.
fn assert_field_parity(
    name: &str,
    reference: [&[f32]; 3],
    candidate: [&[f32]; 3],
    steps: u64,
    tolerance: f32,
) {
    let peak = reference
        .iter()
        .copied()
        .map(peak_of)
        .fold(0.0f32, f32::max);
    assert!(
        peak > 0.0,
        "{name} is identically zero after {steps} steps -- the comparison would pass trivially"
    );
    for (axis, (host, device)) in reference.iter().zip(candidate.iter()).enumerate() {
        let relative = worst_difference(host, device) / peak;
        assert!(
            relative < tolerance,
            "{name}[{axis}] diverged by {relative:e} of the {peak:e} field peak after {steps} steps"
        );
    }
}

/// Steps both solvers over the same scene and requires them to agree.
fn assert_parity(scene: &Scene, steps: u64, tolerance: f32) {
    let context = match headless_context() {
        Ok(context) => context,
        Err(error) => {
            eprintln!("skipping: no usable GPU device ({error})");
            return;
        }
    };

    let mut reference = cpu::Simulation::new(scene);
    let mut candidate = gpu::Simulation::new(context, scene);
    reference.advance_by(steps);
    candidate.advance_by(steps);

    assert_eq!(reference.step_count(), candidate.step_count());
    assert!(reference.is_finite(), "the reference solver diverged");

    let electric = candidate.read_electric();
    let magnetic = candidate.read_magnetic();
    assert_field_parity(
        "E",
        Axis::ALL.map(|axis| reference.electric(axis)),
        Axis::ALL.map(|axis| candidate.component(&electric, axis)),
        steps,
        tolerance,
    );
    assert_field_parity(
        "H",
        Axis::ALL.map(|axis| reference.magnetic(axis)),
        Axis::ALL.map(|axis| candidate.component(&magnetic, axis)),
        steps,
        tolerance,
    );
}

#[test]
fn free_space_with_an_absorbing_boundary() {
    // Exercises the absorber profiles, the sheet source and its apodization.
    assert_parity(&Scene::photon(Extent::new(48, 40, 36)), 300, 1e-4);
}

#[test]
fn a_point_dipole_in_a_perfectly_conducting_box() {
    // No absorber at all, so any disagreement is in the bare stencil. Runs
    // long enough for the wave to bounce off every wall.
    assert_parity(&Scene::cavity(Extent::new(40, 36, 32)), 500, 1e-4);
}

#[test]
fn a_dielectric_slab_and_a_conductor() {
    // Three distinct material indices, so the coefficient table lookup and the
    // perfect-conductor special case are both on the path.
    let mut scene = Scene::empty(Extent::new(44, 40, 40), 1e-3);
    let frequency = scene.grid.frequency_for_resolution(24.0);
    let glass = scene.materials.push(Material::refractive(1.5));
    let lossy = scene.materials.push(Material::matched_lossy(2.0, 0.05));
    let metal = scene.materials.push(Material::PERFECT_CONDUCTOR);
    scene.shapes.push(Shape::Slab {
        axis: Axis::X,
        start: 24,
        end: 32,
        material: glass,
    });
    scene.shapes.push(Shape::Sphere {
        center: [14.0, 20.0, 20.0],
        radius: 5.0,
        material: lossy,
    });
    scene.shapes.push(Shape::Block {
        min: [34, 14, 14],
        max: [38, 26, 26],
        material: metal,
    });
    let scene = scene.with_source(Source::point(
        [12, 20, 20],
        Axis::Z,
        Waveform::ricker(frequency),
    ));
    scene.validate().unwrap();
    assert_parity(&scene, 300, 1e-4);
}

#[test]
fn several_sources_at_once() {
    // Source injection is one dispatch per source with a barrier between, so
    // overlapping sources are the case that would race if the barrier were
    // missing.
    let mut scene = Scene::empty(Extent::cube(40), 1e-3).with_boundary(Boundary::Absorbing {
        thickness: 8,
        target_reflection: 1e-6,
    });
    let frequency = scene.grid.frequency_for_resolution(20.0);
    scene.sources.push(Source::point(
        [20, 20, 20],
        Axis::Z,
        Waveform::ricker(frequency),
    ));
    scene.sources.push(
        Source::point([20, 20, 20], Axis::X, Waveform::ricker(frequency)).with_amplitude(-0.5),
    );
    scene.sources.push(Source::sheet(
        Axis::Y,
        12,
        6.0,
        Axis::Z,
        Waveform::gaussian_pulse(frequency, 3.0),
    ));
    assert_parity(&scene, 250, 1e-4);
}

#[test]
fn resetting_clears_the_device_buffers() {
    let Ok(context) = headless_context() else {
        eprintln!("skipping: no usable GPU device");
        return;
    };
    let scene = Scene::photon(Extent::cube(32));
    let mut simulation = gpu::Simulation::new(context, &scene);
    simulation.advance_by(120);
    let excited = simulation.read_electric();
    assert!(excited.iter().any(|&v| v != 0.0));

    simulation.reset();
    assert_eq!(simulation.step_count(), 0);
    let cleared = simulation.read_electric();
    assert!(
        cleared.iter().all(|&v| v == 0.0),
        "reset left a field behind"
    );
}

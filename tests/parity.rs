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
    timeline::{Steppable, Timeline},
};
use std::sync::Arc;

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
    //
    // The domain is long enough that the packet is still inside it, and driving
    // into the far absorber, when the comparison happens. Running until the
    // field has been absorbed instead would compare two decayed remnants and
    // divide f32 roundoff by whatever was left, which reports a large relative
    // error for an absolute difference of 1e-7.
    assert_parity(&Scene::photon(Extent::new(96, 40, 36)), 150, 1e-4);
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
    // 44 x 40 x 40 cells at 1 mm: the domain runs -22..22 by -20..20 mm.
    let glass = scene.materials.push(Material::refractive(1.5));
    let lossy = scene.materials.push(Material::matched_lossy(2.0, 0.05));
    let metal = scene.materials.push(Material::PERFECT_CONDUCTOR);
    scene.shapes.push(Shape::Slab {
        axis: Axis::X,
        offset: 6e-3,
        thickness: 8e-3,
        material: glass,
    });
    scene.shapes.push(Shape::Sphere {
        center: [-8e-3, 0.0, 0.0],
        radius: 5e-3,
        material: lossy,
    });
    scene.shapes.push(Shape::Block {
        center: [14e-3, 0.0, 0.0],
        size: [4e-3, 12e-3, 12e-3],
        material: metal,
    });
    let scene = scene.with_source(Source::point(
        [-10e-3, 0.0, 0.0],
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
        [0.0; 3],
        Axis::Z,
        Waveform::ricker(frequency),
    ));
    scene
        .sources
        .push(Source::point([0.0; 3], Axis::X, Waveform::ricker(frequency)).with_amplitude(-0.5));
    scene.sources.push(Source::sheet(
        Axis::Y,
        -8e-3,
        6e-3,
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

#[test]
fn a_timeline_scrubs_the_gpu_solver() {
    // On the GPU a snapshot is a full readback and a restore a full upload, so
    // this exercises the path in both directions -- and checks that arriving
    // at a step by replay is indistinguishable from having stepped there,
    // which is the property the whole slider rests on.
    let Ok(context) = headless_context() else {
        eprintln!("skipping: no usable GPU device");
        return;
    };
    let scene = Scene::photon(Extent::cube(32));

    let mut reference = gpu::Simulation::new(Arc::clone(&context), &scene);
    reference.advance_by(70);
    let expected = reference.read_electric();

    let mut simulation = gpu::Simulation::new(context, &scene);
    let mut timeline = Timeline::new(25, 8);
    for _ in 0..120 {
        simulation.advance_by(1);
        timeline.observe(&mut simulation);
    }
    assert!(timeline.keyframe_count() > 1);

    timeline.seek(&mut simulation, 70);
    assert_eq!(Steppable::step_count(&simulation), 70);

    let scrubbed = simulation.read_electric();
    let peak = expected.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let worst = expected
        .iter()
        .zip(scrubbed.iter())
        .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(peak > 0.0);
    assert!(
        worst < 1e-5 * peak,
        "scrubbed state differs from the stepped one by {:e} of the peak",
        worst / peak
    );
}

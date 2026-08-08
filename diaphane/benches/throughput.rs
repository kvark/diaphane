//! How fast the solvers move cells.
//!
//! FDTD is a memory-bandwidth-bound stencil, not an arithmetic problem, so the
//! number that matters is cell-updates per second and the number that explains
//! it is effective bandwidth. Both are reported.
//!
//! A full step reads and writes every component of both fields once, which at
//! `f32` is `6 × 4 × 2 = 48` bytes per cell, plus the material index. Dividing
//! measured throughput by the machine's achievable bandwidth says how much room
//! is left; on a well-fed solver the answer is "not much", and the only
//! remaining lever is moving fewer bytes rather than doing less arithmetic.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use diaphane::{Axis, Boundary, Extent, Scene, Source, Waveform, cpu, gpu};
use std::{hint::black_box, sync::Arc, time::Instant};

/// Bytes of field traffic one step must move per cell, at best.
const BYTES_PER_CELL_STEP: u64 = 6 * 4 * 2;

fn free_space(extent: Extent, boundary: Boundary) -> Scene {
    let scene = Scene::empty(extent, 1e-3).with_boundary(boundary);
    let frequency = scene.grid.frequency_for_resolution(20.0);
    scene.with_source(Source::point(
        [0.0; 3],
        Axis::Z,
        Waveform::ricker(frequency),
    ))
}

fn cpu_solver(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("cpu");
    for side in [32u32, 64, 96] {
        let extent = Extent::cube(side);
        let cells = extent.total() as u64;
        group.throughput(Throughput::Elements(cells));
        group.bench_with_input(
            BenchmarkId::new("free space", side),
            &extent,
            |b, &extent| {
                let scene = free_space(extent, Boundary::DEFAULT);
                let mut simulation = cpu::Simulation::new(&scene);
                b.iter(|| {
                    simulation.advance();
                    black_box(simulation.step_count())
                });
            },
        );
    }
    group.finish();
}

/// The absorbing layer costs three extra loads and a divide per component.
/// Worth knowing whether that is visible against the field traffic — on a
/// bandwidth-bound kernel it should not be.
fn boundary_cost(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("boundary");
    let extent = Extent::cube(64);
    group.throughput(Throughput::Elements(extent.total() as u64));
    for (name, boundary) in [("pec", Boundary::Pec), ("absorbing", Boundary::DEFAULT)] {
        group.bench_function(name, |b| {
            let scene = free_space(extent, boundary);
            let mut simulation = cpu::Simulation::new(&scene);
            b.iter(|| {
                simulation.advance();
                black_box(simulation.step_count())
            });
        });
    }
    group.finish();
}

/// The GPU solver, timed by hand rather than through criterion.
///
/// Criterion's per-iteration timing does not fit a solver whose whole design is
/// to batch hundreds of steps into one submission — measuring a single step
/// would measure submit latency. This runs a large batch and divides.
fn gpu_solver() {
    let context = match gpu::headless_context() {
        Ok(context) => context,
        Err(error) => {
            println!("gpu: skipped, no usable device ({error})");
            return;
        }
    };
    println!("\ngpu: {}", context.device_information().device_name.trim());
    if context.device_information().is_software_emulated {
        println!("gpu: this is a software rasterizer; the numbers describe the CPU it runs on");
    }

    for side in [32u32, 64, 96] {
        let extent = Extent::cube(side);
        let scene = free_space(extent, Boundary::DEFAULT);
        let mut simulation = gpu::Simulation::new(Arc::clone(&context), &scene);

        // Warm up: first submission builds pipelines' first-use state.
        simulation.advance_by(16);
        simulation.wait();

        let steps = 200u64;
        let started = Instant::now();
        simulation.advance_by(steps);
        simulation.wait();
        let elapsed = started.elapsed().as_secs_f64();

        let cells = extent.total() as u64;
        let cell_steps = cells * steps;
        let bandwidth = (cell_steps * BYTES_PER_CELL_STEP) as f64 / elapsed / 1e9;
        println!(
            "gpu: {side}³  {:>8.1} Mcell-steps/s  {:>6.1} GB/s  ({:.0} steps/s)",
            cell_steps as f64 / elapsed / 1e6,
            bandwidth,
            steps as f64 / elapsed,
        );
    }
}

fn report_cpu_scale() {
    // Criterion reports per-element times; this prints the same measurement in
    // the units the design brief is written in, so the two can be compared
    // without arithmetic.
    println!("\ncpu reference, single-threaded:");
    for side in [32u32, 64, 96] {
        let extent = Extent::cube(side);
        let scene = free_space(extent, Boundary::DEFAULT);
        let mut simulation = cpu::Simulation::new(&scene);
        simulation.advance_by(4);

        let steps = match side {
            32 => 400,
            64 => 100,
            _ => 40,
        };
        let started = Instant::now();
        simulation.advance_by(steps);
        let elapsed = started.elapsed().as_secs_f64();

        let cell_steps = extent.total() as u64 * steps;
        println!(
            "cpu: {side}³  {:>8.1} Mcell-steps/s  {:>6.1} GB/s  ({:.0} steps/s)",
            cell_steps as f64 / elapsed / 1e6,
            (cell_steps * BYTES_PER_CELL_STEP) as f64 / elapsed / 1e9,
            steps as f64 / elapsed,
        );
    }
}

fn summary(_criterion: &mut Criterion) {
    report_cpu_scale();
    gpu_solver();
}

criterion_group!(benches, cpu_solver, boundary_cost, summary);
criterion_main!(benches);

//! Against the Rust ecosystem.
//!
//! The comparison is with [`oxiphoton`](https://crates.io/crates/oxiphoton),
//! which is the broadest photonics crate in Rust today and the only one with a
//! 3D FDTD solver. Both are run on the same free-space problem, single
//! threaded, and reported as cell-updates per second.
//!
//! # This is not a like-for-like race, and reporting it as one would be dishonest
//!
//! The two solvers make different choices, and the choices explain most of any
//! gap:
//!
//! | | diaphane | oxiphoton |
//! |---|---|---|
//! | precision | `f32` | `f64` |
//! | absorbing boundary | graded matched conductivity | CPML |
//! | arrays per cell | 6 fields + 1 index | 6 fields + 4 material + 12 CPML `ψ` |
//! | bytes touched per cell | ~28 | ~176 |
//!
//! On a bandwidth-bound stencil, bytes touched per cell is very nearly the
//! whole story, and the ratio above is about 6×. So a large fraction of any
//! measured difference is *by construction* rather than earned: `f64` halves
//! the achievable throughput, and full-domain CPML `ψ` arrays are twelve more
//! streams through memory.
//!
//! Those choices buy something. CPML absorbs grazing incidence and evanescent
//! content that a real conductivity cannot; `f64` matters for high-Q ringdown
//! over 10⁶ steps. Diaphane declines both because its thesis is latency, and
//! latency is bandwidth. What this benchmark shows is the size of that trade,
//! not that one implementation is better written than the other.
//!
//! Run with `cargo bench -p diaphane --bench ecosystem`.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use diaphane::{Axis, Boundary, Extent, Scene, Source, Waveform, cpu};
use oxiphoton::fdtd::{config::BoundaryConfig, dims::fdtd_3d::Fdtd3d};
use std::hint::black_box;

/// Both solvers get the same domain, the same cell size, and the same absorbing
/// layer thickness.
const SIDE: usize = 48;
const CELL_SIZE: f64 = 1e-3;
const LAYER_CELLS: usize = 10;

fn diaphane_scene() -> Scene {
    let extent = Extent::cube(SIDE as u32);
    let scene = Scene::empty(extent, CELL_SIZE as f32).with_boundary(Boundary::Absorbing {
        thickness: LAYER_CELLS as u32,
        target_reflection: 1e-6,
    });
    let frequency = scene.grid.frequency_for_resolution(20.0);
    let center = [extent.x / 2, extent.y / 2, extent.z / 2];
    scene.with_source(Source::point(center, Axis::Z, Waveform::ricker(frequency)))
}

fn oxiphoton_solver() -> Fdtd3d {
    let boundary = BoundaryConfig::pml(LAYER_CELLS);
    Fdtd3d::new(SIDE, SIDE, SIDE, CELL_SIZE, CELL_SIZE, CELL_SIZE, &boundary)
}

fn head_to_head(criterion: &mut Criterion) {
    let cells = (SIDE * SIDE * SIDE) as u64;
    let mut group = criterion.benchmark_group("3d free space, one step");
    // Elements are cell-updates, so the reported throughput is directly
    // comparable between the two regardless of how each is implemented.
    group.throughput(Throughput::Elements(cells));

    group.bench_function("diaphane (f32, matched lossy layer)", |b| {
        let scene = diaphane_scene();
        let mut simulation = cpu::Simulation::new(&scene);
        b.iter(|| {
            simulation.advance();
            black_box(simulation.step_count())
        });
    });

    group.bench_function("oxiphoton (f64, CPML)", |b| {
        let mut solver = oxiphoton_solver();
        // Put a field in it, so neither solver is timed on all-zero data that a
        // memory subsystem might treat unusually.
        solver.inject_ez(SIDE / 2, SIDE / 2, SIDE / 2, 1.0);
        b.iter(|| {
            solver.step();
            black_box(solver.time_step)
        });
    });

    group.finish();
}

/// Both solvers stepped far enough to be sure neither is being flattered by a
/// domain that is still mostly zeros.
fn sustained(criterion: &mut Criterion) {
    let cells = (SIDE * SIDE * SIDE) as u64;
    let mut group = criterion.benchmark_group("3d free space, warmed up");
    group.throughput(Throughput::Elements(cells * 20));
    group.sample_size(20);

    group.bench_function("diaphane", |b| {
        let scene = diaphane_scene();
        let mut simulation = cpu::Simulation::new(&scene);
        simulation.advance_by(120);
        b.iter(|| {
            simulation.advance_by(20);
            black_box(simulation.step_count())
        });
    });

    group.bench_function("oxiphoton", |b| {
        let mut solver = oxiphoton_solver();
        solver.inject_ez(SIDE / 2, SIDE / 2, SIDE / 2, 1.0);
        solver.run(120);
        b.iter(|| {
            solver.run(20);
            black_box(solver.time_step)
        });
    });

    group.finish();
}

criterion_group!(benches, head_to_head, sustained);
criterion_main!(benches);

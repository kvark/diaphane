//! Real-time 3D electrodynamics.
//!
//! Diaphane solves Maxwell's curl equations on a Yee grid in three dimensions
//! and keeps the solver and the renderer in the same GPU memory, so the fields
//! can be watched while they evolve rather than post-processed afterwards. This
//! crate is the headless half: it has no windowing dependency and runs
//! anywhere, including in CI.
//!
//! # Two solvers
//!
//! [`cpu::Simulation`] is the reference implementation -- plain loops, no
//! intrinsics, no threads. [`gpu::Simulation`] is the same physics as a blade
//! compute pipeline. They are written independently and checked against each
//! other so that a shader bug has somewhere to show up; see `tests/parity.rs`.
//!
//! # Where the physics is written down
//!
//! - [`grid`] -- the Yee staggering convention and the discrete dispersion
//!   relation. Read this first; every FDTD bug is an off-by-half.
//! - [`material`] -- impedance normalization and the update coefficients.
//! - [`boundary`] -- what happens at the walls.
//! - [`source`] -- why a source has to be soft and zero-mean.
//! - [`scene`] -- everything needed to reproduce a run.
//! - [`timeline`] -- keyframes, and why a time slider works at all.
//!
//! # Getting started
//!
//! ```
//! use diaphane::{Extent, Scene, cpu};
//!
//! // A transversely apodized wave packet in free space, with absorbing walls.
//! let scene = Scene::photon(Extent::cube(48));
//! scene.validate().unwrap();
//!
//! let mut simulation = cpu::Simulation::new(&scene);
//! simulation.advance_by(100);
//!
//! let energy = simulation.energy();
//! assert!(energy.total() > 0.0);
//! ```

pub mod boundary;
pub mod cpu;
pub mod gpu;
pub mod grid;
pub mod material;
pub mod scene;
pub mod source;
pub mod timeline;

pub use boundary::Boundary;
pub use grid::{Axis, Extent, Grid};
pub use material::{Material, MaterialTable};
pub use scene::{Scene, Shape};
pub use source::{Source, SourceShape, Waveform};

//! Real-time 3D electrodynamics.
//!
//! Diaphane solves Maxwell's curl equations on a Yee grid in three dimensions
//! and keeps the solver and the renderer in the same GPU memory, so the fields
//! can be watched while they evolve rather than post-processed afterwards. This
//! crate is the headless half: it has no windowing dependency and runs
//! anywhere, including in CI.
//!
//! # Where the physics is written down
//!
//! - [`grid`] -- the Yee staggering convention and the discrete dispersion
//!   relation. Read this first; every FDTD bug is an off-by-half.
//! - [`material`] -- impedance normalization and the update coefficients.
//! - [`boundary`] -- what happens at the walls.
//! - [`source`] -- why a source has to be soft and zero-mean.
//! - [`scene`] -- everything needed to reproduce a run.

pub mod boundary;
pub mod grid;
pub mod material;
pub mod scene;
pub mod source;

pub use boundary::Boundary;
pub use grid::{Axis, Extent, Grid};
pub use material::{Material, MaterialTable};
pub use scene::{Scene, Shape};
pub use source::{Source, SourceShape, Waveform};

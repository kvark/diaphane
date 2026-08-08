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
//! [`grid`] holds the Yee staggering convention and the discrete dispersion
//! relation. Read it first; every FDTD bug is an off-by-half.

pub mod grid;

pub use grid::{Axis, Extent, Grid};

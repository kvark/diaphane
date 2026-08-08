//! The visualizer, split into a windowed viewer and an offscreen renderer.
//!
//! Both drive a [`diaphane::gpu::Simulation`] and share [`render`], which
//! binds the solver's own field buffers — nothing is copied between a compute
//! kernel writing a field and the pixel that shows it.
//!
//! They are separate binaries because they share the renderer and nothing
//! else. [`app`] needs a display, an event loop and a surface; [`offscreen`]
//! needs none of those, which is what lets CI run it on a headless machine.

pub mod app;
pub mod camera;
pub mod offscreen;
pub mod options;
pub mod render;

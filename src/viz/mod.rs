//! The visualizer: a windowed viewer and an offscreen renderer.
//!
//! Behind the `viz` feature, because it pulls in a window system and the
//! solver half has to stay runnable without one.
//!
//! Both drive a [`crate::gpu::Simulation`] and share [`render`], whose
//! fragment shader binds the solver's own field buffers — nothing is copied
//! between a compute kernel writing a field and the pixel that shows it.
//! They are separate binary targets because they share that and nothing else:
//! [`app`] needs a display, an event loop and a surface, [`offscreen`] needs
//! none of those, which is what lets CI run it on a headless machine.

pub mod app;
pub mod camera;
pub mod offscreen;
pub mod options;
pub mod render;

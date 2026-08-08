//! Scaffolding shared by the two example programs.
//!
//! Both drive a [`diaphane::gpu::Simulation`] and share [`render`], whose
//! fragment shader binds the solver's own field buffers — nothing is copied
//! between a compute kernel writing a field and the pixel that shows it. They
//! share that and their command line, and nothing else: `viz` needs a display,
//! an event loop and a surface, `render` needs none of those, which is what
//! lets CI run one on a headless machine and the other under a virtual display.
//!
//! This is a module rather than a library because everything under it wants
//! `winit`, `png` and a window system, and those are dev-dependencies — the
//! solver crate has no idea any of them exist. Cargo has no way for one example
//! to depend on another, so both pull this in with `#[path]` and it is compiled
//! once per program.

// Compiled twice, and each program uses a different subset of it: the viewer
// has no use for PNG capture, the renderer has none for the key bindings.
// Whichever half is unused would otherwise be dead code in a build with
// `-D warnings`.
#![allow(dead_code)]

pub mod camera;
pub mod options;
pub mod render;

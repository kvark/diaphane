# diaphane

**Watch light move.**

Diaphane solves Maxwell's curl equations on a 3D Yee grid and keeps the solver
and the renderer in the same GPU memory, so the fields are something you look
at while they evolve rather than something you post-process afterwards. Point
it at a wave packet and you can see the electric and magnetic fields hand
energy back and forth as it goes.

Built on [`blade-graphics`](https://github.com/kvark/blade).

```
cargo viz              # a window
cargo render           # PNGs, no display
```

## Scenes

A scene is data, not a program: geometry, materials, sources and boundary in a
commented [RON](https://github.com/ron-rs/ron) file. Everything is in **metres,
from the centre of the domain**, so a scene is not welded to the resolution it
was written at — `Scene::with_resolution` rediscretizes it without moving
anything, which makes "run it at 2× and check the answer stopped changing" one
call.

```
cargo viz --scene scenes/double-slit.ron
cargo render --scene cavity --save-scene mine.ron
```

[`scenes/`](scenes) has five worked examples — free flight, a conducting
cavity, a glass slab, a metal sphere, and Young's double slit — each commented
with what it shows and what to look at.

## Time

The state is a pure function of `(scene, step)`. Nothing is random, sources are
analytic, so any step can be reproduced rather than recorded. That gives two
independent ways to move backwards:

- **Keyframes.** [`Timeline`](src/timeline.rs) snapshots the fields
  periodically; seeking restores the nearest earlier one and replays. That is
  what the scrub bar along the bottom of the window drags. 24 bytes per cell
  per keyframe, so a long run gets a *window* rather than a full history.
- **Reversal.** Leapfrog is an exact involution, so the solver runs backwards
  with no memory at all — in a lossless box. Through the absorbing layer it
  amplifies by 3× per step, so `reverse` refuses rather than returning an
  exponentially growing field.

## What is here

One crate. The library is the solver and is **headless by default** — nothing
in the default feature set knows what a window is, so `diaphane` can be
depended on, tested in CI, or embedded elsewhere without dragging one in. CI
asserts that rather than trusting it.

The visualizer is two binary targets behind the `viz` feature:

| | |
|---|---|
| `diaphane-viz` | the viewer: a window, an orbit camera, and a scrub bar. |
| `diaphane-render` | the same view, written to PNGs. Needs no display. |

They share the render pass and nothing else — one needs an event loop and a
surface, the other needs neither, which is what lets CI run the second on a
headless machine and the first under Xvfb.

The solver comes in two implementations of the same physics. `diaphane::gpu` is
the blade compute pipeline and is the one that has to be fast.
`diaphane::cpu` is a plain-loops reference: it gives the validation suite
something to run without a GPU, it gives the shader an oracle to be checked
against, and it gives the benchmarks an honest baseline against the CPU-resident
solvers that exist in the Rust ecosystem today.

## The physics

Full-vector 3D FDTD — `Ex, Ey, Ez, Hx, Hy, Hz` on the standard Yee staggering,
leapfrogged in place, `f32` throughout.

Fields are impedance-normalized (`Ẽ = √(ε₀/μ₀)·E`), which makes the two curl
updates structurally identical and keeps both fields in the same numeric range.
Materials are a per-cell `u32` index into a small coefficient table, so
repainting geometry never means recomputing coefficients. Boundaries are either
perfectly conducting walls or a graded, impedance-matched absorbing layer whose
profile is separable and therefore costs three 1D arrays rather than a field.

The conventions that are easy to get wrong — which sample sits at which
half-cell, which difference is forward and which is backward — are written down
in [`src/grid.rs`](src/grid.rs) and referred to from everywhere else. Every FDTD bug is an off-by-half.

## Validation

`cargo test` runs, without needing a GPU:

- **an exact discrete plane wave**, seeded and required to propagate exactly —
  not a discretized continuum solution but a solution of the stepping scheme
  itself, in vacuum and in dielectrics, along axes, diagonals and oblique
  directions. Reproducing it means every convention is right at once: which
  sample sits at which half-cell, which difference is forward, and how far
  apart in time `E` and `H` are.
- **numerical phase velocity** against the analytic dispersion relation. The
  check is that the solver matches the *discrete* physics, which is not `c` and
  provably cannot be — plus a negative control confirming the test would notice
  if it were wrong
- **energy conservation** in a closed PEC box over 40,000 steps
- **absorbing-layer reflection**, measured in dB against an oversized reference
  domain rather than eyeballed. A ten-cell layer comes in at −58 dB.
- **energy equipartition** in a travelling packet, against the alternation seen
  in a cavity — the two halves of the statement the visualizer exists to show

and, when a Vulkan/Metal device is available, **CPU/GPU parity** on scenes with
dielectrics, conductors, absorbers and overlapping sources.

Nothing here is exercised only by compiling it. `mesa-vulkan-drivers` gives a
headless runner a real Vulkan device via lavapipe, and Xvfb gives it a real X11
display, so CI runs the solver, the offscreen renderer *and* the windowed
viewer — the last with `--exit-after`, which presents a fixed number of frames
and quits. CI also cross-checks the Metal backend from Linux, because blade
picks its backend by target and the two differ enough to break independently.

## Benchmarks

FDTD is bandwidth-bound, so the unit is cell-updates per second. On the same
48³ free-space problem, single-threaded:

| | throughput |
|---|---|
| **diaphane** (`f32`, matched lossy layer) | **56 Mcell-steps/s** |
| [`oxiphoton`](https://crates.io/crates/oxiphoton) (`f64`, CPML) | 20–23 Mcell-steps/s |

About 2.5×, and most of it is a design choice rather than better code:
oxiphoton carries `f64` fields and twelve full-domain CPML `ψ` arrays, which is
roughly 6× the bytes per cell. Converting only part of a 6× traffic advantage
into a 2.5× speed advantage means there is headroom left on our side.

Those choices buy oxiphoton real things — CPML handles grazing incidence that a
graded conductivity cannot, and `f64` matters for high-`Q` ringdown. The full
numbers, the caveats, and what is deliberately *not* being compared are in
[`docs/benchmarks.md`](docs/benchmarks.md).

```
cargo bench
```

## Documentation

- [`docs/design.md`](docs/design.md) — the original design brief
- [`docs/architecture.md`](docs/architecture.md) — what was built, and where it
  departs from that brief and why
- [`docs/benchmarks.md`](docs/benchmarks.md) — measured throughput, and what the
  comparison does and does not mean

## License

MIT

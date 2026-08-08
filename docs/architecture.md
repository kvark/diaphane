# Architecture, and where it departs from the design brief

[`design.md`](design.md) is the original handoff brief. It specifies a 2D
TM<sup>z</sup> instrument with 3D deferred to "Phase 5, open question". This
document records what was actually built and why the shape differs.

## Deviation 1: 3D from the start

The brief's own reasoning for staying 2D is storage and simplicity. The stated
purpose of the visualizer, though, is to watch a pulse propagate and see the
electric and magnetic fields trade energy — and that trade is a statement about
two mutually orthogonal vector fields and a propagation direction. In
TM<sup>z</sup> you get `Ez, Hx, Hy`: one scalar and one in-plane vector, which
shows the exchange but hides that it is a rotation of one vector field into
another.

So the solver is **full-vector 3D Yee FDTD** — `Ex, Ey, Ez, Hx, Hy, Hz` — from
the first commit. The costs the brief anticipated are real but bounded:

| | 2D TM<sup>z</sup> | 3D full vector |
|---|---|---|
| Field arrays | 3 | 6 |
| Bytes/cell (fields + material) | 16 | 28 |
| Courant limit | `1/√2 ≈ 0.707` | `1/√3 ≈ 0.577` |
| Visualization | direct heatmap | volume ray march |

Everything else in the brief survives the move: Yee staggering, impedance
normalization, in-place leapfrog, the material-index-plus-coefficient-table
layout, storage buffers over storage textures, and `f32` throughout.

The one place 3D genuinely costs is *interaction*: painting geometry with a
pointer (brief §8) is a 2D idea that does not transfer, and it is not
implemented. What is implemented is the observation half of the instrument.

## Deviation 2: a CPU reference solver alongside the GPU one

The brief describes one solver, on the GPU. This repository has two, computing
the same thing:

- [`Simulation`](../diaphane/src/cpu.rs) — a plain Rust reference. Loops in
  `f32`, no intrinsics, no threads.
- [`gpu::Simulation`](../diaphane/src/gpu.rs) — the blade compute pipeline,
  which is the one that has to be fast.

This is not redundancy for its own sake. It buys three things:

1. **The validation suite runs anywhere.** The analytic checks in
   [`tests/validation.rs`](../diaphane/tests/validation.rs) need no GPU, so
   "is the physics right" is answered independently of "is the shader right".
2. **The shader gets an oracle.** [`tests/parity.rs`](../diaphane/tests/parity.rs)
   steps both solvers over the same scene and requires them to agree to 1e-4 of
   the field peak. An FDTD bug that is a half-cell offset produces a
   plausible-looking wave either way; a numerical comparison against an
   independently written implementation does not care how plausible it looks.
   The tolerance is not zero because the two do the same arithmetic in
   different orders and the GPU may contract multiply-adds where the CPU may
   not — demanding bit equality would mean pinning the shader compiler.
3. **The benchmark has an honest baseline.** Every FDTD package in the Rust
   ecosystem today is CPU-resident. Comparing our GPU number against them
   measures the hardware, not the code. Comparing CPU against CPU measures the
   code, and the GPU speedup is then reported separately against our own CPU
   path. See [`benches/`](../diaphane/benches).

The two implementations share the coefficient derivation
([`material.rs`](../diaphane/src/material.rs)) and the scene description, but
not the stepping loop — a shared loop would make the parity test tautological.

## Deviation 3: a graded matched lossy layer instead of Mur, before CPML

The brief sequences boundaries as Mur (Phase 1) → CPML (Phase 2). We ship a
third thing first, which is strictly better than Mur and much less bug-prone
than CPML:

The absorber is a **graded, impedance-matched conductive layer**. Both the
electric and magnetic loss rates are set to the same value `r` (in 1/s), which
is the condition for zero reflection at a free-space interface in the
continuum. `r` is graded as a cubic polynomial over the layer thickness with
the standard PML `σ_max` formula, so the discretization error — the only
remaining source of reflection — is what limits performance.

The implementation detail that makes this cheap: the profile is **separable**.
`r(i,j,k) = rx(i) + ry(j) + rz(k)`, so it costs three 1D arrays of length
`nx`, `ny`, `nz` rather than a full-domain field, and it composes correctly at
edges and corners with no special cases. Each axis stores two profiles, sampled
at integer and half-integer positions, because `E` and `H` are staggered and a
matched layer requires the loss to be co-located with the field it damps.

Reflection is measured, not eyeballed: `the_absorbing_layer_reflects_below_the_stated_level`
runs the same scene in a domain and in a 3× longer reference domain and takes
the difference between the two probe traces, which is by construction the part
that came back off the near wall. A ten-cell layer measures **−58 dB** against
a point source radiating into it across all angles. The test asserts −50 dB.

That is close to the −60 dB the brief wanted from CPML, for none of the
machinery — no auxiliary `ψ` fields, no complex coordinate stretching. It will
not hold up as well as CPML for strongly evanescent or near-grazing content,
which is what the complex stretch exists to absorb and what a real conductivity
cannot touch; CPML remains the right answer if this ever needs to be
quantitative at grazing incidence.

`Boundary::Pec` is also available and is *free*: clamping the stencil to the
array bounds leaves the outermost tangential `E` samples at zero forever, which
is exactly a perfect electric conductor. That is what the energy-conservation
test runs in.

## Deviation 5: geometry in metres, not cells

The brief does not say which, and the obvious first implementation puts shapes
and sources at cell indices. That is a trap: a scene written in cells is welded
to one resolution, so you cannot run it at 2x and check the answer stopped
changing — which is the single most useful thing a saved scene enables, and the
only routine defence against an under-resolved result that looks entirely
convincing.

So [`Shape`](../diaphane/src/scene.rs) and
[`SourceShape`](../diaphane/src/source.rs) are in **metres, with the origin at
the centre of the domain**. Centred rather than cornered because geometry then
survives a change of *domain size* too — growing the box to give a wave more
room would otherwise drag every object along with the far wall. Boxes are given
as centre-and-size rather than min-and-max for the same reason, and because it
is how every photonics tool states one.

[`Scene::with_resolution`] rediscretizes without moving anything, which makes a
convergence study a one-liner. The `--resolution` flag on the visualizer is the
same thing.

The failure mode this introduces is passing a cell index where metres are
wanted — a sphere at "20" is twenty metres out and silently paints nothing. So
`Scene::validate` rejects geometry that lies entirely outside the domain and
sources outside it, and says so in those words.

[`Scene::with_resolution`]: ../diaphane/src/scene.rs

## Deviation 4: no `egui` yet

The brief's Phase 3 is an interaction layer built on egui. The visualizer here
has time controls, camera control, view-mode switching, brightness and
signed-log toggles on the keyboard, with the perf HUD — steps/s, effective
GB/s, frame time, simulation time — in the window title.

Scene *authoring* is not implemented at all. Scenes are the three presets in
[`scene.rs`](../diaphane/src/scene.rs), selected with `--scene`. The types
derive `serde` behind a feature flag, so file-based scenes are a small step,
but nothing reads or writes one yet and the brief's painting and dragging are
untouched.

## What carried over unchanged

- Impedance-normalized fields `Ẽ = √(ε₀/μ₀)·E`, so the two curl updates have
  identical coefficient structure and `f32` stays well conditioned.
- In-place leapfrog. No ping-pong buffers: the `H` update reads only `E` and
  the `E` update reads only `H`, so two dispatches and a barrier suffice.
- Per-cell `u32` material index into a small coefficient table, never per-cell
  coefficients.
- Storage buffers rather than storage textures, so the fragment shader reads
  the field arrays directly with no copy.
- `S = 0.5` by default, comfortably under the 3D limit of `1/√3`.
- Perf HUD from the first version, not the last.

## What it costs

Measured, on an identical free-space problem: about 2.5× the throughput of the
only other 3D FDTD solver published in Rust, most of which is these design
choices rather than better code. [`benchmarks.md`](benchmarks.md) has the
numbers and the caveats.

## Layout

```
diaphane/          headless solver library. No windowing dependency.
  src/grid.rs        Yee grid, extents, indexing, staggering convention
  src/material.rs    materials and their update coefficients
  src/boundary.rs    separable graded absorbing profiles
  src/source.rs      waveforms and source geometry
  src/scene.rs       serializable scene description
  src/cpu.rs         reference solver
  src/gpu.rs         blade compute solver
  src/shaders/fdtd.wgsl
  tests/             analytic validation and CPU/GPU parity
  benches/           throughput, and comparison against the ecosystem
diaphane-viz/      the visualizer binary: winit + blade volume rendering
docs/
```

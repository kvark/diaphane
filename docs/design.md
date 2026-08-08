# Diaphane — Design Document

**A real-time, interactive 2D electrodynamics instrument built on `blade-graphics`.**

Status: draft v0.1 — intended as a handoff brief for an implementing agent.

---

## 1. Thesis

Every existing FDTD package is a *library*: you describe a scene in a script, wait, and post-process the results. Diaphane is an *instrument*: the solver and the renderer share GPU memory, the simulation runs at hundreds of steps per displayed frame, and you reshape the scene with a pointer while the fields keep evolving around your edits.

The wager is not feature parity with Meep, Lumerical, or `oxiphoton`. It is that **nobody has made wave electrodynamics feel like a video game**, and that the wave regime — interference, diffraction, resonance, evanescent coupling — is the part of optics that is genuinely non-obvious and rewards being watched rather than plotted.

Scope discipline: 2D, wave regime, one mode family, done extremely well. Explicitly **not** a ray tracer, **not** a general photonics toolkit.

### Non-goals

- 3D (until Phase 5 at the earliest; see §9 risk 2)
- Geometric/ray optics — different regime, different tool
- Inverse design, optimization loops, S-parameter extraction pipelines
- Feature-checklist competition with existing crates

---

## 2. Physics

### 2.1 Mode choice

Start with **TM<sup>z</sup>**: fields are `(Ez, Hx, Hy)`, invariant along *z*. Rationale:

- `Ez` is a single scalar → visualizes directly as a signed heatmap with no vector decomposition
- Covers dielectric refraction, diffraction, waveguiding, resonators, photonic crystals
- Half the field storage of the full vector problem

TE<sup>z</sup> (`Hz, Ex, Ey`) is deferred to Phase 4. It matters for metals (surface plasmons live in TE) and should be a compile-time or runtime mode switch sharing the same kernel skeleton.

### 2.2 Governing equations

```
∂Hx/∂t = −(1/μ) ∂Ez/∂y
∂Hy/∂t = +(1/μ) ∂Ez/∂x
∂Ez/∂t = (1/ε) (∂Hy/∂x − ∂Hx/∂y − σ Ez)
```

### 2.3 Yee staggering

Square cells, `Δx = Δy = Δ`. Field sample locations:

| Field | Spatial offset | Temporal offset |
|-------|----------------|-----------------|
| `Ez`  | `(i, j)`       | integer `n`     |
| `Hx`  | `(i, j+½)`     | half `n+½`      |
| `Hy`  | `(i+½, j)`     | half `n+½`      |

In storage all three are plain `[width × height]` arrays; the staggering is a convention about what index means, not a layout difference. Keep the offset convention written at the top of the shader — every FDTD bug is an off-by-half.

### 2.4 Normalized update equations

Use the impedance-normalized electric field `Ẽ = √(ε₀/μ₀) · E`. This makes the free-space update coefficients identical for both curl equations and keeps everything in a well-conditioned range for `f32`:

```
Hx[i,j]  −= (S/μr) · (Ẽz[i,j+1] − Ẽz[i,j])
Hy[i,j]  += (S/μr) · (Ẽz[i+1,j] − Ẽz[i,j])
Ẽz[i,j]  += (S/εr) · ((Hy[i,j] − Hy[i−1,j]) − (Hx[i,j] − Hx[i,j−1]))
```

where `S = c·Δt/Δ` is the Courant number.

With conductivity, the `Ẽz` update takes the standard semi-implicit form:

```
Ẽz = Ca·Ẽz + Cb·(curl H)
Ca = (1 − σΔt/2ε) / (1 + σΔt/2ε)
Cb = (S/εr)      / (1 + σΔt/2ε)
```

`Ca` and `Cb` are precomputed on the CPU per material and looked up by index — the kernel is then a pure multiply-add.

### 2.5 Stability and resolution

- **Courant limit (2D, square cells):** `S ≤ 1/√2 ≈ 0.7071`. Ship with `S = 0.5` for headroom against CPML and dispersive materials.
- **Grid resolution:** ≥ 20 cells per wavelength *in the highest-index material present*. 10 is the absolute floor and will visibly distort phase.
- **Numerical dispersion:** phase-velocity error is `O((Δ/λ)²)`, worst along the grid axes, best along diagonals. This is the dominant physical inaccuracy at reasonable resolutions and should be surfaced in the UI as an estimated ppm error, not hidden.

### 2.6 Leapfrog is in-place

The H update reads only E; the E update reads only H. **No double buffering is required** — two dispatches per timestep with a barrier between them, writing into the same buffers. This is worth stating explicitly because the instinct on GPU is to ping-pong, and here it would double memory traffic for nothing.

---

## 3. Boundaries

**Phase 1:** first-order Mur absorbing boundary. ~10 lines, reflection around −30 dB. Good enough to see a pulse leave without ringing forever.

**Phase 2:** CPML (convolutional PML), the real answer.

- Complex-stretched coordinates with graded `σ`, `κ`, `α`
- Polynomial grading order `m = 3`, thickness 10 cells
- `σ_max = −(m+1)·ln(R₀) / (2·η·d)` with target `R₀ ≈ 1e−6`
- `κ_max ∈ [5, 15]`, `α` graded from `α_max` at the inner edge to 0 at the outer
- Auxiliary recursive-convolution variables `ψ` are needed only for derivatives normal to each PML face: `ψ_Ezx`, `ψ_Ezy`, `ψ_Hyx`, `ψ_Hxy`

**Implementation note:** `ψ` is nonzero only inside the PML slabs. Allocating them full-domain is simpler and in 2D costs little; do that first, and only pack them into slab-shaped buffers if profiling says it matters.

Also provide **PEC** (`Ez = 0`) walls as an option — a closed lossless PEC box is the cleanest energy-conservation test case.

---

## 4. Sources

All sources are **soft/additive** (`Ez += f(t)`) by default. Hard sources (overwrite) act as scatterers and produce spurious reflections.

**Waveforms:**

- **Ricker wavelet** — default. `r(t) = (1 − 2π²f_p²τ²)·exp(−π²f_p²τ²)`, `τ = t − t₀`. Zero DC content, which matters: a source with a DC component slowly builds a static field that never radiates away.
- **Gaussian-modulated sinusoid** — band-limited, for spectral work
- **Continuous wave** — must have a smooth ramp envelope (a few periods of raised cosine). A hard switch-on is a step discontinuity and injects broadband garbage.

**Geometry:** point source (Phase 1), line source, and TF/SF (total-field/scattered-field) plane wave with a 1D auxiliary incident grid (Phase 4). TF/SF is what makes scattering cross-sections meaningful, so it's worth doing properly rather than faking a plane wave with a wide line source.

---

## 5. Materials

**Storage:** a per-cell `u32` material index into a small coefficient table, *not* per-cell coefficients.

- Repainting is a single index write, no coefficient recomputation on the GPU
- Dispersive materials carry extra per-material parameters without bloating the grid
- Material table lives in a small storage buffer, updated whenever the palette changes

**Tiers:**

1. **Non-dispersive dielectric** — `εr`, `μr` (Phase 1–2)
2. **Lossy dielectric** — adds `σ` (Phase 2)
3. **Drude metal via ADE** (Phase 4): `ε(ω) = ε∞ − ωp²/(ω² + iγω)`, implemented as an auxiliary current `∂J/∂t + γJ = ε₀ωp²E` with one extra field `Jz` folded into the E update. This is what unlocks plasmonics and is the main reason to also implement TE<sup>z</sup>.

**Subpixel smoothing** (averaging `ε` at material boundaries according to field orientation) buys roughly an order of magnitude in effective resolution at curved interfaces. Not Phase 1, but architect the material lookup so it can be added without restructuring — it's the single highest-leverage accuracy upgrade available.

---

## 6. GPU architecture (blade)

### 6.1 Data

Field arrays as **storage buffers** of `f32`, not storage textures. Reasons: no format/read-write-access portability friction, and the fragment shader can read them directly for visualization with zero copies.

Buffers:

```
Ez, Hx, Hy            f32 × W×H
mat_index             u32 × W×H
mat_table             small struct array (Ca, Cb, Da, Db, drude params)
psi_*                 f32 × W×H  (CPML, Phase 2)
Jz                    f32 × W×H  (Drude, Phase 4)
intensity_accum       f32 × W×H  (time-averaged |Ez|², visualization)
probe_ring            f32 ring buffer, read back async
```

### 6.2 Per-timestep pipeline

```
dispatch update_h   → barrier
dispatch update_e   → barrier
dispatch inject_sources  (or fold into update_e)
[optional] dispatch accumulate_intensity / dft_monitors
```

Workgroup size `8×8` to start. FDTD is a memory-bandwidth-bound stencil, not an ALU problem — shared-memory tiling to reduce redundant neighbor loads is a real but modest optimization (the L2 already catches most of it). **Do not optimize arithmetic. Optimize traffic.**

### 6.3 Performance budget

Order-of-magnitude, not a promise. Essential traffic per cell per full timestep is roughly 3 arrays streamed twice ≈ 50 B. At 1024×1024 = 1M cells that's ~50 MB per step. A mid-range consumer GPU at ~500–800 GB/s effective gives a theoretical ceiling of ~10–15k steps/s; expect **3–5k steps/s** in practice.

At 60 fps display that is **50–80 solver substeps per rendered frame** — comfortably enough that wave propagation looks like motion rather than a slideshow. Decouple the solver rate from the display rate with an explicit "steps per frame" control, and expose a wall-clock-to-simulation-time ratio in the UI.

### 6.4 Precision

`f32` throughout. Roundoff accumulates roughly as `√N · ε`, so at 10⁴–10⁵ steps expect relative error around 1e−5 — irrelevant for visualization and fine for most engineering. It becomes a real ceiling for high-Q resonator ringdown over 10⁶+ steps. Document the limit rather than reaching for `f64`, which halves bandwidth and therefore halves the frame rate — the entire product thesis.

---

## 7. Rendering

Fullscreen triangle; fragment shader reads `Ez` (or a derived buffer) directly from the storage buffer.

**Colormapping:**

- **Signed diverging map** for `Ez`. Use a perceptually uniform diverging map (Crameri `vik`/`berlin`, or cool-warm). Naive blue→red is not perceptually uniform and actively misrepresents magnitude.
- **Signed-log scaling** — `sign(E)·log(1 + |E|/E₀)` — as a toggle. This is the single biggest usability feature in the whole tool: it lets a weak scattered field be visible in the same frame as the source that produced it. Linear scaling alone makes most interesting physics invisible.
- **Auto-ranging** via exponential moving max with hysteresis, plus a manual clamp. Without hysteresis the display flickers as the pulse amplitude changes.

**Layers:**

- Field (primary)
- Material overlay: Sobel edges on `mat_index` drawn as contour lines, so geometry reads clearly without occluding the field
- Time-averaged intensity (`Σ Ez²`) as an alternate view — this is where interference fringes and standing-wave patterns become *stationary and beautiful*, and it is the screenshot that sells the project
- UI overlay: source markers, probe pins, geometry handles

---

## 8. Interaction

The interaction model is the product. Everything above is table stakes.

**Geometry / materials**

- Paint materials with a brush (size, material selector)
- Place and drag primitives: rectangle, circle, ring, waveguide segment, slit array
- Drag-to-reshape with the solver running — this is the signature interaction

> **Physical honesty:** mutating `ε` while fields are live is a time-varying medium, and it *radiates*. Spurious transients are physically real, not a bug. This is delightful in play mode and unacceptable for measurement, so provide an explicit **"commit geometry & reset fields"** action for quantitative runs, and consider a subtle UI indicator that the current field state is contaminated by edits.

**Sources**

- Click to place; panel for waveform, frequency, amplitude, phase, polarization
- Live frequency slider — watching a waveguide go from guiding to cutoff by dragging one slider is the best two seconds of the whole demo

**Probes and measurement**

- Point probes writing into a GPU ring buffer, read back asynchronously into a scrolling time-trace plot (a few frames of latency is fine)
- Flux monitors across a line (Phase 4)
- Running DFT accumulators at a handful of frequencies, computed in a compute pass with **no per-step readback** — this is how transmission/reflection spectra get computed without stalling the pipeline

**Time control**

- Play / pause / single-step / reset fields / reset all
- Steps-per-frame slider
- Freeze-and-inspect: pause and hover to read field values

**Persistence**

- Scene serialization (RON or JSON): grid, `Δ`, materials, shapes, sources, probes, view settings
- This also makes the validation suite scriptable and gives the CI something to run headless

---

## 9. Risks and honest assessment

1. **"Pretty pulses" is a demo, not a tool.** The gap between an impressive GIF and something a person opens twice is enormous. Mitigation: pick *one* real workflow it is genuinely best at (candidate: interactively tuning a waveguide taper or grating coupler and watching coupling efficiency respond live) and make that path excellent. Ship it with 6–8 curated example scenes that are each a self-contained lesson.

2. **2D is not cheap 3D.** It is the physics of an infinitely extruded structure. For photonics the standard dodge is the effective-index approximation, which is genuinely useful and genuinely approximate. Say so plainly in the docs; a tool that oversells its regime loses expert trust immediately.

3. **Scope creep into "another photonics library."** `oxiphoton` already has broad coverage — FDTD, BPM, S-matrix, RCWA, inverse design. Competing on features is a losing race and would destroy the thing that makes this different. The differentiator is latency and directness, and every feature added is bandwidth taken from that.

4. **Bandwidth ceiling is the hard constraint.** Domain size, framerate, and precision all trade against one another through a single number. Build the perf HUD (steps/s, effective GB/s, sim-time-per-wall-second) in Phase 1, not Phase 4.

5. **CPML is where implementations go to die.** It is the least fun and most bug-prone component, and a subtly wrong PML produces plausible-looking-but-wrong results. Do not hand-tune it into apparent correctness — measure reflection against an oversized reference domain (§10) and require a number.

---

## 10. Validation suite

Acceptance criteria, runnable headless in CI against the scene format:

| Test | Criterion |
|------|-----------|
| Free-space pulse propagation | measured numerical phase velocity matches the analytic 2D dispersion relation to < 0.5% at 20 cells/λ |
| PML reflection | compare against reference run in a 4× larger domain; require < −60 dB |
| Fresnel coefficients | normal and oblique incidence on a dielectric half-space; R and T within 1% of analytic |
| Snell refraction | measured refraction angle within 1° across a range of incidence angles |
| Single/double slit | far-field pattern matches analytic diffraction envelope |
| Mie scattering (cylinder) | scattering cross-section vs. the analytic series solution |
| Energy conservation | lossless PEC box, `U = ½∫(εE² + μH²)` constant to roundoff over 10⁵ steps |

That last one is worth a permanent place in the UI as an energy readout — it is both a live correctness canary and a small piece of physics education.

---

## 11. Workspace layout

```
diaphane/
├── diaphane-sim/     solver: grid, materials, CPML, sources, blade compute
│                     pipelines, WGSL. Headless-capable — no windowing dep.
├── diaphane-app/     winit + egui, interaction, rendering passes, scene I/O
├── diaphane-scenes/  curated example scenes (the teaching material)
└── tests/            validation suite, runs headless against diaphane-sim
```

Keeping `diaphane-sim` headless is load-bearing: it makes the validation suite runnable in CI, and it means the solver can be embedded elsewhere later without dragging a window system along.

---

## 12. Roadmap

**Phase 0 — Skeleton**
Blade device setup, window, fullscreen triangle sampling a dummy buffer, hot-reloadable WGSL, perf HUD.
*Exit:* a colormapped gradient at 60 fps, shader edits visible without restart.

**Phase 1 — Core solver**
TM<sup>z</sup> free space, Mur ABC, soft Ricker point source, substepping, signed diverging colormap.
*Exit:* a pulse spreads outward at the correct numerical phase velocity (within 1%) and leaves the domain without visible ringing.

**Phase 2 — Materials and CPML**
Material index buffer, coefficient table, dielectric and lossy media, full CPML.
*Exit:* Snell refraction and total internal reflection at a slab match analytic; PML reflection < −60 dB.

**Phase 3 — The instrument**
Painting, draggable primitives, source panel with live sliders, probes with time traces, signed-log view, intensity accumulator, time controls, scene save/load.
*Exit:* a double-slit experiment can be built by hand in under 60 seconds and fringes are clearly visible in the intensity view.

**Phase 4 — Depth**
TE<sup>z</sup> mode, Drude metals via ADE, TF/SF plane wave, DFT monitors and spectra, flux lines, subpixel smoothing.
*Exit:* a surface plasmon propagating on a metal film; a transmission spectrum of a grating that matches a reference solver.

**Phase 5 — Open question**
3D. A genuinely different engineering problem (memory, visualization, interaction all change character). Defer the decision until Phase 4 has real users.

---

## 13. Immediate first task for the implementing agent

Phase 0 and Phase 1, in one pass:

1. Blade device + swapchain + winit window
2. `Ez`, `Hx`, `Hy` storage buffers, `256×256` to start
3. Two WGSL compute kernels, `update_h` and `update_e`, free space only, `S = 0.5`
4. Ricker source injected at the grid center
5. Fullscreen fragment shader reading `Ez`, cool-warm diverging colormap, fixed manual range
6. Mur first-order ABC on all four walls
7. Perf HUD: steps/s, frame time, effective bandwidth
8. Keyboard: space = pause, R = reset fields, arrows = steps-per-frame

Success is a pulse expanding as a clean circular ripple and absorbing at the walls. Everything after that is elaboration on a loop that already works.

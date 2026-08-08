# Benchmarks

```
cargo bench -p diaphane --bench throughput   # our two solvers
cargo bench -p diaphane --bench ecosystem    # against oxiphoton
```

FDTD is a memory-bandwidth-bound stencil, so the unit is **cell-updates per
second** and the number that explains it is effective bandwidth. Arithmetic is
not the constraint and optimizing it is not the lever.

## The machine these numbers came from

A shared 4-core cloud VM: Intel Xeon @ 2.8 GHz, 16 GB, **no GPU**. Read
everything below with that in mind, and in particular:

> **The "GPU" numbers are lavapipe**, Mesa's software Vulkan rasterizer. They
> measure the same CPU running WGSL. They are here because they demonstrate the
> pipeline works end to end and because they show the *shape* of the scaling —
> not because they say anything about what a real GPU would do. The design
> brief's estimate for a mid-range consumer GPU is 3–5k steps/s at 1024², which
> is two to three orders of magnitude above anything measured here.

## Our two solvers

Free space, absorbing boundary, one point source. Single-threaded CPU.

| Domain | CPU reference | lavapipe |
|---|---|---|
| 32³ | 49.1 Mcell-steps/s · 2.4 GB/s · 1497 steps/s | 27.3 Mcell-steps/s · 833 steps/s |
| 64³ | 49.1 Mcell-steps/s · 2.4 GB/s · 187 steps/s | 35.4 Mcell-steps/s · 135 steps/s |
| 96³ | 49.6 Mcell-steps/s · 2.4 GB/s · 56 steps/s | 56.6 Mcell-steps/s · 64 steps/s |

Two things are worth reading out of this.

**The CPU reference is flat across domain sizes.** 49 Mcell-steps/s at 32³ and
at 96³, a 27× difference in working set. A solver that were cache-resident at
the small size and streaming at the large one would not be. That is the
signature of a kernel already limited by streaming bandwidth rather than by
cache.

**The lavapipe path improves with size while the CPU path does not.** At 32³ it
is well behind and by 96³ it has pulled ahead. Per-dispatch overhead is fixed
and the work per dispatch grows as the cube, so small domains are dominated by
launch cost. A real GPU shows the same shape more sharply: the batching in
`advance_by` — hundreds of steps encoded into one command buffer — exists
precisely because of this.

### What the absorbing boundary costs

| | 64³ |
|---|---|
| PEC walls | 45.5 Mcell-steps/s |
| Graded matched layer | 46.1 Mcell-steps/s |

Nothing, within the noise of this machine. The absorber adds three array reads
and a reciprocal per component; on a bandwidth-bound kernel those are free
because the loads come from three tiny 1D profiles that live in L1 and the
divide overlaps with waiting for memory. This is the payoff for making the
profile separable rather than a full-domain field — a per-cell `σ` would have
added a fourth streaming array and would have shown up here immediately.

## Against the ecosystem

[`oxiphoton`](https://crates.io/crates/oxiphoton) 0.1.2 is the broadest
photonics crate in Rust and the only one publishing a 3D FDTD solver. Same 48³
free-space problem, same 10-cell absorbing layer, both single-threaded:

| | one step | warmed up |
|---|---|---|
| **diaphane** (`f32`, matched lossy layer) | **56.4 Mcell-steps/s** | **56.3 Mcell-steps/s** |
| oxiphoton (`f64`, CPML) | 19.5 Mcell-steps/s | 23.4 Mcell-steps/s |

About 2.4–2.9× faster. **Most of that is a design choice, not better code**, and
saying otherwise would be dishonest:

| | diaphane | oxiphoton |
|---|---|---|
| precision | `f32` | `f64` |
| absorbing boundary | graded matched conductivity | CPML |
| arrays streamed per cell | 6 fields + 1 index | 6 fields + 4 material + 12 CPML `ψ` |
| bytes touched per cell | ~28 | ~176 |

The byte-traffic ratio is about 6×. We are 2.4–2.9× faster. So diaphane is
converting only *part* of its structural advantage into speed — which is the
interesting number here, and the honest one to keep an eye on. It suggests
oxiphoton's inner loop is doing better per byte than ours, and that there is
headroom on our side that a wider inner loop or explicit SIMD could reach.

The choices oxiphoton makes buy real things. CPML absorbs grazing-incidence and
evanescent content that a graded conductivity cannot; `f64` matters for
high-`Q` ringdown over 10⁶ steps, where our `f32` roundoff accumulates as
`√N·ε` into a real floor. Diaphane declines both because its thesis is latency,
and on this kernel latency *is* bandwidth. What the table measures is the size
of that trade.

### What is not being compared

- **Feature coverage.** oxiphoton has BPM, RCWA, S-matrix, mode solvers,
  inverse design. Diaphane has one solver and a viewer. Competing on breadth is
  a losing race and would destroy the thing that makes this different.
- **Multithreading.** oxiphoton has a `parallel` feature; diaphane's CPU path is
  deliberately single-threaded because it is a *reference*, and its job is to be
  obviously correct. Threading it would make the comparison closer to fair on a
  multi-core box and would make the reference worse at its actual job.
- **Accuracy.** Neither number says anything about whether the answers are
  right. That is what `tests/validation.rs` is for.

## Reproducing

Criterion output lands in `target/criterion`. The `throughput` bench also prints
a plain-text summary in the units above, because criterion's per-iteration
timing on a noisy shared VM disagreed with a straight batch-and-divide by up to
35% at the largest size — the criterion numbers are for tracking regressions
against themselves, the printed summary is for quoting.

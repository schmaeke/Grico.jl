# Remote Phase 4 Scheduler Decision

Benchmarks for Phase 4 were run on `helios` (`Ryzen 9 9950X`, `16` physical cores) with
`1,2,4,8,16` Julia threads on:

- `affine_cell_diffusion / assemble`
- `affine_interface_dg / assemble`
- `nonlinear_interface_dg / residual!`
- `nonlinear_interface_dg / tangent`

The retained benchmark artifact is [remote_phase4_default_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase4_default_9950x.md).

## Retained Design

Keep the Phase 4 scheduler abstraction with:

- static scheduling as the default for regular batched work,
- weighted-static / hybrid selection only for the irregular cell-affine pass,
- dynamic scheduling retained as an internal override for benchmarking and future experiments.

This keeps one coherent traversal backend while selecting a better default for the workloads that actually dominate threaded assembly.

## Why This Is Retained

Compared against the retained pre-Phase-4 state in [remote_current_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_current_9950x.md), the retained Phase 4 default at `16` threads gives:

- `affine_cell_diffusion / assemble`: `196.75 ms -> 189.01 ms` (`1.04x` faster), `1452.75 -> 1445.60 MiB`
- `affine_interface_dg / assemble`: `53.66 ms -> 41.02 ms` (`1.31x` faster), `348.06 -> 307.73 MiB`
- `nonlinear_interface_dg / tangent`: `33.09 ms -> 28.16 ms` (`1.17x` faster), `256.76 -> 238.94 MiB`
- `nonlinear_interface_dg / residual!`: `3.19 ms -> 4.38 ms` (`0.73x`)

The residual regression is real in the benchmark data, but it is small in absolute terms and does not outweigh the much larger gains on the affine interface and tangent paths.

## Variant Comparison

At `16` threads on `helios`:

- Static override in [remote_phase4_static_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase4_static_9950x.md) was close to the retained default on interface-heavy affine assembly (`42.95 ms`) but lost ground on tangent (`34.02 ms`) and cell-affine assembly (`195.62 ms`).
- Hybrid override in [remote_phase4_hybrid_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase4_hybrid_9950x.md) won the regular cell-affine case (`184.10 ms`) but gave back most of the interface-heavy gain (`53.19 ms`) and increased memory.
- Dynamic override in [remote_phase4_dynamic_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase4_dynamic_9950x.md) stayed close to the old behavior and was clearly worse on interface-heavy affine work (`55.25 ms`).

So the retained default is the best balanced policy, not the best policy for every single case.

## Discarded Refinement

A later refinement that split residual and tangent scheduling into separate default categories was benchmarked in [remote_phase4_final_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase4_final_9950x.md) and then discarded.

That refinement regressed the nonlinear path enough that it was not worth keeping:

- `nonlinear_interface_dg / residual!`: `4.38 ms -> 5.98 ms`
- `nonlinear_interface_dg / tangent`: `28.16 ms -> 32.92 ms`
- `affine_interface_dg / assemble`: `41.02 ms -> 45.62 ms`

The codebase therefore keeps the simpler retained Phase 4 default represented by [remote_phase4_default_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase4_default_9950x.md).

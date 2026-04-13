# Remote 9950X Phase 3 Decision

Benchmarks compared on `helios` (`AMD Ryzen 9 9950X`, `16C / 32T`) with thread counts `1, 2, 4, 8, 16`:

- clean baseline `HEAD`: [remote_baseline_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_baseline_9950x.md:1)
- current code: [remote_current_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_current_9950x.md:1)
- current code with Phase 3b direct scatter disabled: [remote_current_no_direct_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_current_no_direct_9950x.md:1)

## Current Vs Baseline

At `16` threads:

| Case / operation | Baseline | Current | Interpretation |
| --- | ---: | ---: | --- |
| `affine_cell_diffusion / assemble` | `199.29 ms` | `196.75 ms` | Effectively flat, slightly faster. |
| `affine_interface_dg / assemble` | `83.87 ms` | `53.66 ms` | Large threaded win, about `1.56x`. |
| `nonlinear_interface_dg / residual!` | `5.61 ms` | `3.19 ms` | Strong win, about `1.76x`. |
| `nonlinear_interface_dg / tangent` | `35.06 ms` | `33.09 ms` | Small but real win, about `1.06x`. |

The serial affine paths are worse than baseline, but the node-level threaded results on the 16-core machine are clearly better where shared-memory scaling matters most, especially for interface-heavy affine assembly.

## Current Vs No Direct Scatter

This isolates Phase 3b.

At `8` threads:

| Case / operation | No direct scatter | Current | Interpretation |
| --- | ---: | ---: | --- |
| `affine_cell_diffusion / assemble` | `243.02 ms` | `222.03 ms` | Direct scatter helps, about `1.09x`. |
| `affine_interface_dg / assemble` | `67.82 ms` | `66.06 ms` | Small win, about `1.03x`. |
| `nonlinear_interface_dg / residual!` | `4.06 ms` | `3.84 ms` | Small win, about `1.06x`. |
| `nonlinear_interface_dg / tangent` | `36.56 ms` | `36.05 ms` | Marginal win, about `1.01x`. |

At `16` threads:

| Case / operation | No direct scatter | Current | Interpretation |
| --- | ---: | ---: | --- |
| `affine_cell_diffusion / assemble` | `194.19 ms` | `196.75 ms` | Current is slightly slower. |
| `affine_interface_dg / assemble` | `53.17 ms` | `53.66 ms` | Effectively flat, slightly slower. |
| `nonlinear_interface_dg / residual!` | `3.16 ms` | `3.19 ms` | Effectively flat, slightly slower. |
| `nonlinear_interface_dg / tangent` | `32.67 ms` | `33.09 ms` | Effectively flat, slightly slower. |

## Recommendation

- Keep Phase 3 batching and traversal prefiltering.
- Do not treat Phase 3b direct scatter as a settled long-term win yet.
- On this 16-core machine it helps around `8` threads, but the benefit disappears by `16` threads and slightly reverses.
- If code simplicity is the priority, Phase 3b is the first candidate to remove or rework.

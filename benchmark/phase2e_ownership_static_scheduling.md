# Phase 2e Ownership Static Scheduling

Generated: `2026-04-12T22:15:30`

Primary thread counts: `1, 2, 4, 6`

Primary scaling set capped at 6 threads on this host because the machine has 6 performance cores.

## Environment

- Julia: `1.12.5`
- OS / arch: `Darwin` / `aarch64`
- CPU: `Apple M2 Pro`
- BLAS threads: `1`
- `OPENBLAS_NUM_THREADS`: `1`
- `OMP_NUM_THREADS`: `1`

## Cases

| Case | Full dofs | Reduced dofs | Leaves | Cells | Interfaces | Max local dofs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `affine_cell_diffusion` | 14641 | 7761 | 1600 | 1600 | 0 | 16 |
| `affine_interface_dg` | 14400 | 14400 | 1600 | 1600 | 3120 | 18 |
| `nonlinear_interface_dg` | 7056 | - | 784 | 784 | 1512 | 18 |
| `adaptive_poisson` | 1279 | - | 140 | - | - | - |

## Results

### `affine_cell_diffusion`

Continuous scalar Poisson problem with volume-dominated affine assembly.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 5.37 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.31 ms | 5.95 MiB | 121226 | 1.25 | 0.62 |
| `adaptivity_plan` | 4 | 4.11 ms | 5.95 MiB | 121293 | 1.31 | 0.33 |
| `adaptivity_plan` | 6 | 3.89 ms | 5.95 MiB | 121355 | 1.38 | 0.23 |
| `assemble` | 1 | 1.774 s | 1445.75 MiB | 59837106 | 1.00 | 1.00 |
| `assemble` | 2 | 1.292 s | 1446.00 MiB | 59834215 | 1.37 | 0.69 |
| `assemble` | 4 | 527.24 ms | 1446.18 MiB | 59828414 | 3.37 | 0.84 |
| `assemble` | 6 | 468.70 ms | 1446.90 MiB | 59822634 | 3.79 | 0.63 |
| `preconditioner_build` | 1 | 12.29 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 79.38 ms | 133.01 MiB | 16369 | 0.15 | 0.08 |
| `preconditioner_build` | 4 | 76.71 ms | 134.02 MiB | 16404 | 0.16 | 0.04 |
| `preconditioner_build` | 6 | 83.59 ms | 134.30 MiB | 16442 | 0.15 | 0.02 |
| `solve_direct` | 1 | 19.02 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 18.98 ms | 28.19 MiB | 136 | 1.00 | 0.50 |
| `solve_direct` | 4 | 19.30 ms | 28.44 MiB | 136 | 0.99 | 0.25 |
| `solve_direct` | 6 | 19.78 ms | 28.44 MiB | 136 | 0.96 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 213.05 ms | 311.17 MiB | 9611491 | 1.00 | 1.00 |
| `assemble` | 2 | 134.96 ms | 317.89 MiB | 9611628 | 1.58 | 0.79 |
| `assemble` | 4 | 83.64 ms | 318.87 MiB | 9611872 | 2.55 | 0.64 |
| `assemble` | 6 | 63.83 ms | 319.93 MiB | 9612150 | 3.34 | 0.56 |
| `preconditioner_build` | 1 | 20.15 ms | 290.46 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 87.65 ms | 290.22 MiB | 16400 | 0.23 | 0.11 |
| `preconditioner_build` | 4 | 88.21 ms | 288.86 MiB | 16435 | 0.23 | 0.06 |
| `preconditioner_build` | 6 | 92.96 ms | 294.39 MiB | 16473 | 0.22 | 0.04 |
| `solve_direct` | 1 | 30.59 ms | 81.46 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 29.98 ms | 86.44 MiB | 108 | 1.02 | 0.51 |
| `solve_direct` | 4 | 30.47 ms | 78.55 MiB | 108 | 1.00 | 0.25 |
| `solve_direct` | 6 | 64.40 ms | 112.00 MiB | 136 | 0.47 | 0.08 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 60.82 ms | 23.42 MiB | 1270468 | 1.00 | 1.00 |
| `residual_bang` | 2 | 32.38 ms | 23.46 MiB | 1270592 | 1.88 | 0.94 |
| `residual_bang` | 4 | 16.94 ms | 23.52 MiB | 1270779 | 3.59 | 0.90 |
| `residual_bang` | 6 | 11.80 ms | 23.60 MiB | 1270999 | 5.15 | 0.86 |
| `tangent` | 1 | 195.78 ms | 235.74 MiB | 11317730 | 1.00 | 1.00 |
| `tangent` | 2 | 113.74 ms | 236.53 MiB | 11317881 | 1.72 | 0.86 |
| `tangent` | 4 | 65.51 ms | 240.05 MiB | 11318118 | 2.99 | 0.75 |
| `tangent` | 6 | 50.97 ms | 237.51 MiB | 11318380 | 3.84 | 0.64 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 747.23 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 981.23 μs | 0.98 MiB | 16730 | 0.76 | 0.38 |
| `adaptivity_plan` | 4 | 1.15 ms | 0.98 MiB | 16787 | 0.65 | 0.16 |
| `adaptivity_plan` | 6 | 834.17 μs | 0.98 MiB | 16854 | 0.90 | 0.15 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.774 s at 1 thread to 468.70 ms at 6 threads, a `3.79x` speedup with `0.63` efficiency. Allocation volume stays essentially flat at about `1445.75` to `1446.90 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 213.05 ms to 63.83 ms at 6 threads, or `3.34x` with `0.56` efficiency, while memory changes by about `1.03x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 60.82 ms to 11.80 ms at 6 threads, which is `5.15x` and `0.86` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 195.78 ms to 50.97 ms at 6 threads, or `3.84x` with `0.64` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.15x` to `0.22x`, and direct solve speedups remain `0.47x` to `0.96x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
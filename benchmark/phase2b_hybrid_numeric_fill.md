# Phase 2b Hybrid Numeric Fill

Generated: `2026-04-12T19:31:09`

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
| `adaptivity_plan` | 1 | 5.38 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.39 ms | 5.95 MiB | 121226 | 1.22 | 0.61 |
| `adaptivity_plan` | 4 | 4.13 ms | 5.96 MiB | 121291 | 1.30 | 0.33 |
| `adaptivity_plan` | 6 | 3.97 ms | 5.96 MiB | 121356 | 1.36 | 0.23 |
| `assemble` | 1 | 2.292 s | 1440.59 MiB | 59837104 | 1.00 | 1.00 |
| `assemble` | 2 | 1.034 s | 1446.81 MiB | 59834207 | 2.22 | 1.11 |
| `assemble` | 4 | 592.83 ms | 1449.49 MiB | 59828312 | 3.87 | 0.97 |
| `assemble` | 6 | 448.17 ms | 1453.79 MiB | 59822390 | 5.11 | 0.85 |
| `preconditioner_build` | 1 | 12.38 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 79.05 ms | 133.01 MiB | 16369 | 0.16 | 0.08 |
| `preconditioner_build` | 4 | 75.60 ms | 134.02 MiB | 16404 | 0.16 | 0.04 |
| `preconditioner_build` | 6 | 80.88 ms | 134.28 MiB | 16442 | 0.15 | 0.03 |
| `solve_direct` | 1 | 18.89 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 18.93 ms | 28.19 MiB | 136 | 1.00 | 0.50 |
| `solve_direct` | 4 | 19.24 ms | 28.44 MiB | 136 | 0.98 | 0.25 |
| `solve_direct` | 6 | 19.27 ms | 28.44 MiB | 136 | 0.98 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 180.92 ms | 294.44 MiB | 9611466 | 1.00 | 1.00 |
| `assemble` | 2 | 110.97 ms | 301.60 MiB | 9611620 | 1.63 | 0.82 |
| `assemble` | 4 | 65.82 ms | 301.58 MiB | 9611770 | 2.75 | 0.69 |
| `assemble` | 6 | 54.95 ms | 299.84 MiB | 9611902 | 3.29 | 0.55 |
| `preconditioner_build` | 1 | 20.55 ms | 293.65 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 86.04 ms | 294.81 MiB | 16400 | 0.24 | 0.12 |
| `preconditioner_build` | 4 | 86.67 ms | 293.99 MiB | 16435 | 0.24 | 0.06 |
| `preconditioner_build` | 6 | 90.45 ms | 294.50 MiB | 16473 | 0.23 | 0.04 |
| `solve_direct` | 1 | 29.45 ms | 88.03 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 30.06 ms | 88.03 MiB | 108 | 0.98 | 0.49 |
| `solve_direct` | 4 | 29.81 ms | 94.60 MiB | 108 | 0.99 | 0.25 |
| `solve_direct` | 6 | 30.53 ms | 94.60 MiB | 108 | 0.96 | 0.16 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 88.70 ms | 23.52 MiB | 1270485 | 1.00 | 1.00 |
| `residual_bang` | 2 | 47.74 ms | 23.50 MiB | 1270587 | 1.86 | 0.93 |
| `residual_bang` | 4 | 23.67 ms | 23.48 MiB | 1270702 | 3.75 | 0.94 |
| `residual_bang` | 6 | 16.59 ms | 23.58 MiB | 1270822 | 5.35 | 0.89 |
| `tangent` | 1 | 213.90 ms | 226.90 MiB | 11317705 | 1.00 | 1.00 |
| `tangent` | 2 | 129.26 ms | 230.41 MiB | 11317867 | 1.65 | 0.83 |
| `tangent` | 4 | 56.12 ms | 230.43 MiB | 11318014 | 3.81 | 0.95 |
| `tangent` | 6 | 56.31 ms | 230.88 MiB | 11318142 | 3.80 | 0.63 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 709.77 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 736.40 μs | 0.97 MiB | 16728 | 0.96 | 0.48 |
| `adaptivity_plan` | 4 | 1.18 ms | 0.98 MiB | 16787 | 0.60 | 0.15 |
| `adaptivity_plan` | 6 | 900.52 μs | 0.99 MiB | 16855 | 0.79 | 0.13 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 2.292 s at 1 thread to 448.17 ms at 6 threads, a `5.11x` speedup with `0.85` efficiency. Allocation volume stays essentially flat at about `1440.59` to `1453.79 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 180.92 ms to 54.95 ms at 6 threads, or `3.29x` with `0.55` efficiency, while memory changes by about `1.02x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 88.70 ms to 16.59 ms at 6 threads, which is `5.35x` and `0.89` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 213.90 ms to 56.31 ms at 6 threads, or `3.80x` with `0.63` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.15x` to `0.23x`, and direct solve speedups remain `0.96x` to `0.98x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
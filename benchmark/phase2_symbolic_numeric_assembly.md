# Phase 2 Symbolic + Numeric Assembly

Generated: `2026-04-12T19:11:23`

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
| `adaptivity_plan` | 1 | 5.12 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.37 ms | 5.94 MiB | 121224 | 1.17 | 0.59 |
| `adaptivity_plan` | 4 | 4.09 ms | 5.95 MiB | 121280 | 1.25 | 0.31 |
| `adaptivity_plan` | 6 | 3.89 ms | 5.95 MiB | 121351 | 1.31 | 0.22 |
| `assemble` | 1 | 1.727 s | 1441.46 MiB | 59837136 | 1.00 | 1.00 |
| `assemble` | 2 | 1.067 s | 1441.55 MiB | 59834211 | 1.62 | 0.81 |
| `assemble` | 4 | 526.42 ms | 1441.52 MiB | 59828310 | 3.28 | 0.82 |
| `assemble` | 6 | 520.94 ms | 1442.27 MiB | 59822388 | 3.32 | 0.55 |
| `preconditioner_build` | 1 | 11.69 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 79.77 ms | 133.92 MiB | 16369 | 0.15 | 0.07 |
| `preconditioner_build` | 4 | 76.75 ms | 134.02 MiB | 16404 | 0.15 | 0.04 |
| `preconditioner_build` | 6 | 83.54 ms | 134.30 MiB | 16442 | 0.14 | 0.02 |
| `solve_direct` | 1 | 18.92 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 18.77 ms | 28.44 MiB | 136 | 1.01 | 0.50 |
| `solve_direct` | 4 | 18.91 ms | 28.44 MiB | 136 | 1.00 | 0.25 |
| `solve_direct` | 6 | 19.56 ms | 28.44 MiB | 136 | 0.97 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 179.78 ms | 300.59 MiB | 9611510 | 1.00 | 1.00 |
| `assemble` | 2 | 106.46 ms | 301.38 MiB | 9611613 | 1.69 | 0.84 |
| `assemble` | 4 | 65.59 ms | 301.68 MiB | 9611757 | 2.74 | 0.69 |
| `assemble` | 6 | 48.96 ms | 299.95 MiB | 9611883 | 3.67 | 0.61 |
| `preconditioner_build` | 1 | 20.98 ms | 293.65 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 88.97 ms | 288.64 MiB | 16400 | 0.24 | 0.12 |
| `preconditioner_build` | 4 | 85.43 ms | 294.10 MiB | 16435 | 0.25 | 0.06 |
| `preconditioner_build` | 6 | 92.13 ms | 294.70 MiB | 16473 | 0.23 | 0.04 |
| `solve_direct` | 1 | 29.76 ms | 77.08 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 30.31 ms | 78.55 MiB | 108 | 0.98 | 0.49 |
| `solve_direct` | 4 | 29.97 ms | 93.44 MiB | 108 | 0.99 | 0.25 |
| `solve_direct` | 6 | 30.95 ms | 86.18 MiB | 108 | 0.96 | 0.16 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 46.69 ms | 23.51 MiB | 1270482 | 1.00 | 1.00 |
| `residual_bang` | 2 | 25.34 ms | 23.49 MiB | 1270581 | 1.84 | 0.92 |
| `residual_bang` | 4 | 13.53 ms | 23.48 MiB | 1270690 | 3.45 | 0.86 |
| `residual_bang` | 6 | 9.12 ms | 23.58 MiB | 1270804 | 5.12 | 0.85 |
| `tangent` | 1 | 174.96 ms | 230.36 MiB | 11317747 | 1.00 | 1.00 |
| `tangent` | 2 | 104.13 ms | 229.75 MiB | 11317862 | 1.68 | 0.84 |
| `tangent` | 4 | 60.36 ms | 230.43 MiB | 11318003 | 2.90 | 0.72 |
| `tangent` | 6 | 45.36 ms | 229.57 MiB | 11318125 | 3.86 | 0.64 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 711.85 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 707.48 μs | 0.98 MiB | 16730 | 1.01 | 0.50 |
| `adaptivity_plan` | 4 | 1.15 ms | 0.98 MiB | 16787 | 0.62 | 0.16 |
| `adaptivity_plan` | 6 | 885.46 μs | 0.98 MiB | 16855 | 0.80 | 0.13 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.727 s at 1 thread to 520.94 ms at 6 threads, a `3.32x` speedup with `0.55` efficiency. Allocation volume stays essentially flat at about `1441.46` to `1442.27 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 179.78 ms to 48.96 ms at 6 threads, or `3.67x` with `0.61` efficiency, while memory changes by about `1.00x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 46.69 ms to 9.12 ms at 6 threads, which is `5.12x` and `0.85` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 174.96 ms to 45.36 ms at 6 threads, or `3.86x` with `0.64` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.14x` to `0.23x`, and direct solve speedups remain `0.96x` to `0.97x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
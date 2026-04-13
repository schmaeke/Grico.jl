# Phase 3b Interface Direct Scatter

Generated: `2026-04-13T08:17:50`

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
| `adaptivity_plan` | 1 | 5.24 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.46 ms | 5.95 MiB | 121226 | 1.18 | 0.59 |
| `adaptivity_plan` | 4 | 4.28 ms | 5.95 MiB | 121293 | 1.22 | 0.31 |
| `adaptivity_plan` | 6 | 3.84 ms | 5.96 MiB | 121360 | 1.37 | 0.23 |
| `assemble` | 1 | 2.161 s | 1444.33 MiB | 59825908 | 1.00 | 1.00 |
| `assemble` | 2 | 1.049 s | 1448.44 MiB | 59823089 | 2.06 | 1.03 |
| `assemble` | 4 | 527.55 ms | 1450.24 MiB | 59817600 | 4.10 | 1.02 |
| `assemble` | 6 | 374.03 ms | 1453.44 MiB | 59812346 | 5.78 | 0.96 |
| `preconditioner_build` | 1 | 12.27 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 78.61 ms | 133.00 MiB | 16369 | 0.16 | 0.08 |
| `preconditioner_build` | 4 | 75.55 ms | 134.02 MiB | 16404 | 0.16 | 0.04 |
| `preconditioner_build` | 6 | 82.64 ms | 134.28 MiB | 16442 | 0.15 | 0.02 |
| `solve_direct` | 1 | 18.99 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 20.95 ms | 28.19 MiB | 136 | 0.91 | 0.45 |
| `solve_direct` | 4 | 18.62 ms | 28.44 MiB | 136 | 1.02 | 0.25 |
| `solve_direct` | 6 | 19.50 ms | 28.44 MiB | 136 | 0.97 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 185.31 ms | 319.61 MiB | 9603495 | 1.00 | 1.00 |
| `assemble` | 2 | 119.62 ms | 337.80 MiB | 9603699 | 1.55 | 0.77 |
| `assemble` | 4 | 91.35 ms | 358.22 MiB | 9604242 | 2.03 | 0.51 |
| `assemble` | 6 | 70.92 ms | 356.01 MiB | 9604970 | 2.61 | 0.44 |
| `preconditioner_build` | 1 | 21.05 ms | 293.97 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 83.82 ms | 294.48 MiB | 16400 | 0.25 | 0.13 |
| `preconditioner_build` | 4 | 85.88 ms | 290.74 MiB | 16435 | 0.25 | 0.06 |
| `preconditioner_build` | 6 | 94.81 ms | 289.27 MiB | 16473 | 0.22 | 0.04 |
| `solve_direct` | 1 | 29.97 ms | 77.08 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 29.27 ms | 90.77 MiB | 108 | 1.02 | 0.51 |
| `solve_direct` | 4 | 29.46 ms | 81.24 MiB | 108 | 1.02 | 0.25 |
| `solve_direct` | 6 | 62.30 ms | 103.74 MiB | 136 | 0.48 | 0.08 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 37.58 ms | 23.42 MiB | 1270475 | 1.00 | 1.00 |
| `residual_bang` | 2 | 21.51 ms | 24.31 MiB | 1270707 | 1.75 | 0.87 |
| `residual_bang` | 4 | 11.34 ms | 24.56 MiB | 1271078 | 3.31 | 0.83 |
| `residual_bang` | 6 | 7.98 ms | 24.59 MiB | 1271530 | 4.71 | 0.78 |
| `tangent` | 1 | 167.14 ms | 235.74 MiB | 11317737 | 1.00 | 1.00 |
| `tangent` | 2 | 104.06 ms | 248.42 MiB | 11318004 | 1.61 | 0.80 |
| `tangent` | 4 | 60.50 ms | 256.86 MiB | 11318541 | 2.76 | 0.69 |
| `tangent` | 6 | 47.77 ms | 259.28 MiB | 11319243 | 3.50 | 0.58 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 710.52 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 744.79 μs | 0.97 MiB | 16728 | 0.95 | 0.48 |
| `adaptivity_plan` | 4 | 1.08 ms | 0.98 MiB | 16787 | 0.66 | 0.16 |
| `adaptivity_plan` | 6 | 853.77 μs | 0.98 MiB | 16855 | 0.83 | 0.14 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 2.161 s at 1 thread to 374.03 ms at 6 threads, a `5.78x` speedup with `0.96` efficiency. Allocation volume stays essentially flat at about `1444.33` to `1453.44 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 185.31 ms to 70.92 ms at 6 threads, or `2.61x` with `0.44` efficiency, while memory changes by about `1.11x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 37.58 ms to 7.98 ms at 6 threads, which is `4.71x` and `0.78` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 167.14 ms to 47.77 ms at 6 threads, or `3.50x` with `0.58` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.15x` to `0.22x`, and direct solve speedups remain `0.48x` to `0.97x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
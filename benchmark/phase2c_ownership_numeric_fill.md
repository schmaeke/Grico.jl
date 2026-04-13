# Phase 2c Ownership-Based Numeric Fill

Generated: `2026-04-12T21:57:22`

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
| `adaptivity_plan` | 1 | 5.26 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.39 ms | 5.95 MiB | 121226 | 1.20 | 0.60 |
| `adaptivity_plan` | 4 | 4.44 ms | 5.95 MiB | 121295 | 1.18 | 0.30 |
| `adaptivity_plan` | 6 | 3.87 ms | 5.96 MiB | 121355 | 1.36 | 0.23 |
| `assemble` | 1 | 4.031 s | 1447.12 MiB | 59837118 | 1.00 | 1.00 |
| `assemble` | 2 | 1.167 s | 1448.24 MiB | 59834276 | 3.45 | 1.73 |
| `assemble` | 4 | 559.24 ms | 1450.28 MiB | 59828777 | 7.21 | 1.80 |
| `assemble` | 6 | 449.45 ms | 1454.13 MiB | 59823501 | 8.97 | 1.49 |
| `preconditioner_build` | 1 | 12.73 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 78.26 ms | 133.01 MiB | 16369 | 0.16 | 0.08 |
| `preconditioner_build` | 4 | 77.20 ms | 134.02 MiB | 16404 | 0.16 | 0.04 |
| `preconditioner_build` | 6 | 81.63 ms | 134.30 MiB | 16442 | 0.16 | 0.03 |
| `solve_direct` | 1 | 19.45 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 19.33 ms | 28.19 MiB | 136 | 1.01 | 0.50 |
| `solve_direct` | 4 | 19.09 ms | 28.44 MiB | 136 | 1.02 | 0.25 |
| `solve_direct` | 6 | 19.29 ms | 28.44 MiB | 136 | 1.01 | 0.17 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 211.50 ms | 309.86 MiB | 9611504 | 1.00 | 1.00 |
| `assemble` | 2 | 120.14 ms | 329.39 MiB | 9611682 | 1.76 | 0.88 |
| `assemble` | 4 | 71.13 ms | 343.45 MiB | 9612200 | 2.97 | 0.74 |
| `assemble` | 6 | 68.87 ms | 348.76 MiB | 9612924 | 3.07 | 0.51 |
| `preconditioner_build` | 1 | 21.14 ms | 290.82 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 87.19 ms | 294.59 MiB | 16400 | 0.24 | 0.12 |
| `preconditioner_build` | 4 | 86.30 ms | 294.20 MiB | 16435 | 0.24 | 0.06 |
| `preconditioner_build` | 6 | 90.66 ms | 294.72 MiB | 16473 | 0.23 | 0.04 |
| `solve_direct` | 1 | 30.21 ms | 81.91 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 30.24 ms | 77.08 MiB | 108 | 1.00 | 0.50 |
| `solve_direct` | 4 | 29.81 ms | 94.60 MiB | 108 | 1.01 | 0.25 |
| `solve_direct` | 6 | 62.62 ms | 112.00 MiB | 136 | 0.48 | 0.08 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 47.49 ms | 23.49 MiB | 1270480 | 1.00 | 1.00 |
| `residual_bang` | 2 | 27.02 ms | 23.67 MiB | 1270627 | 1.76 | 0.88 |
| `residual_bang` | 4 | 11.76 ms | 23.95 MiB | 1270936 | 4.04 | 1.01 |
| `residual_bang` | 6 | 9.19 ms | 24.61 MiB | 1271406 | 5.17 | 0.86 |
| `tangent` | 1 | 181.02 ms | 238.10 MiB | 11317742 | 1.00 | 1.00 |
| `tangent` | 2 | 109.06 ms | 241.97 MiB | 11317924 | 1.66 | 0.83 |
| `tangent` | 4 | 65.29 ms | 252.07 MiB | 11318399 | 2.77 | 0.69 |
| `tangent` | 6 | 48.06 ms | 253.36 MiB | 11319039 | 3.77 | 0.63 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 746.23 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 752.73 μs | 0.97 MiB | 16722 | 0.99 | 0.50 |
| `adaptivity_plan` | 4 | 736.69 μs | 0.98 MiB | 16787 | 1.01 | 0.25 |
| `adaptivity_plan` | 6 | 820.69 μs | 0.98 MiB | 16858 | 0.91 | 0.15 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 4.031 s at 1 thread to 449.45 ms at 6 threads, a `8.97x` speedup with `1.49` efficiency. Allocation volume stays essentially flat at about `1447.12` to `1454.13 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 211.50 ms to 68.87 ms at 6 threads, or `3.07x` with `0.51` efficiency, while memory changes by about `1.13x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 47.49 ms to 9.19 ms at 6 threads, which is `5.17x` and `0.86` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 181.02 ms to 48.06 ms at 6 threads, or `3.77x` with `0.63` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.16x` to `0.23x`, and direct solve speedups remain `0.48x` to `1.01x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
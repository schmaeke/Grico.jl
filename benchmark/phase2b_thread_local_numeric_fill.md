# Phase 2b Thread-Local Numeric Fill

Generated: `2026-04-12T19:26:53`

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
| `adaptivity_plan` | 1 | 5.33 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.46 ms | 5.95 MiB | 121226 | 1.20 | 0.60 |
| `adaptivity_plan` | 4 | 4.46 ms | 5.95 MiB | 121289 | 1.19 | 0.30 |
| `adaptivity_plan` | 6 | 3.88 ms | 5.96 MiB | 121355 | 1.37 | 0.23 |
| `assemble` | 1 | 1.798 s | 1440.59 MiB | 59837104 | 1.00 | 1.00 |
| `assemble` | 2 | 1.014 s | 1446.87 MiB | 59834211 | 1.77 | 0.89 |
| `assemble` | 4 | 572.84 ms | 1449.44 MiB | 59828308 | 3.14 | 0.78 |
| `assemble` | 6 | 476.64 ms | 1453.79 MiB | 59822390 | 3.77 | 0.63 |
| `preconditioner_build` | 1 | 13.49 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 78.70 ms | 133.92 MiB | 16369 | 0.17 | 0.09 |
| `preconditioner_build` | 4 | 77.15 ms | 134.02 MiB | 16404 | 0.17 | 0.04 |
| `preconditioner_build` | 6 | 82.95 ms | 134.29 MiB | 16442 | 0.16 | 0.03 |
| `solve_direct` | 1 | 19.00 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 18.91 ms | 28.44 MiB | 136 | 1.00 | 0.50 |
| `solve_direct` | 4 | 19.08 ms | 28.44 MiB | 136 | 1.00 | 0.25 |
| `solve_direct` | 6 | 19.77 ms | 28.44 MiB | 136 | 0.96 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 199.72 ms | 294.44 MiB | 9611466 | 1.00 | 1.00 |
| `assemble` | 2 | 120.64 ms | 319.65 MiB | 9611597 | 1.66 | 0.83 |
| `assemble` | 4 | 74.71 ms | 339.42 MiB | 9611735 | 2.67 | 0.67 |
| `assemble` | 6 | 69.63 ms | 343.70 MiB | 9611863 | 2.87 | 0.48 |
| `preconditioner_build` | 1 | 19.89 ms | 293.65 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 88.68 ms | 288.64 MiB | 16400 | 0.22 | 0.11 |
| `preconditioner_build` | 4 | 86.94 ms | 291.14 MiB | 16435 | 0.23 | 0.06 |
| `preconditioner_build` | 6 | 92.37 ms | 290.13 MiB | 16473 | 0.22 | 0.04 |
| `solve_direct` | 1 | 29.68 ms | 88.03 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 29.77 ms | 78.55 MiB | 108 | 1.00 | 0.50 |
| `solve_direct` | 4 | 29.76 ms | 86.55 MiB | 108 | 1.00 | 0.25 |
| `solve_direct` | 6 | 30.27 ms | 80.36 MiB | 108 | 0.98 | 0.16 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 47.62 ms | 23.52 MiB | 1270485 | 1.00 | 1.00 |
| `residual_bang` | 2 | 26.87 ms | 23.50 MiB | 1270587 | 1.77 | 0.89 |
| `residual_bang` | 4 | 13.81 ms | 23.48 MiB | 1270702 | 3.45 | 0.86 |
| `residual_bang` | 6 | 8.96 ms | 23.58 MiB | 1270822 | 5.31 | 0.89 |
| `tangent` | 1 | 178.23 ms | 226.90 MiB | 11317705 | 1.00 | 1.00 |
| `tangent` | 2 | 115.83 ms | 245.46 MiB | 11317844 | 1.54 | 0.77 |
| `tangent` | 4 | 66.30 ms | 242.84 MiB | 11317979 | 2.69 | 0.67 |
| `tangent` | 6 | 47.77 ms | 251.14 MiB | 11318115 | 3.73 | 0.62 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 711.37 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 741.33 μs | 0.97 MiB | 16728 | 0.96 | 0.48 |
| `adaptivity_plan` | 4 | 677.54 μs | 0.98 MiB | 16800 | 1.05 | 0.26 |
| `adaptivity_plan` | 6 | 896.98 μs | 0.98 MiB | 16854 | 0.79 | 0.13 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.798 s at 1 thread to 476.64 ms at 6 threads, a `3.77x` speedup with `0.63` efficiency. Allocation volume stays essentially flat at about `1440.59` to `1453.79 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 199.72 ms to 69.63 ms at 6 threads, or `2.87x` with `0.48` efficiency, while memory changes by about `1.17x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 47.62 ms to 8.96 ms at 6 threads, which is `5.31x` and `0.89` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 178.23 ms to 47.77 ms at 6 threads, or `3.73x` with `0.62` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.16x` to `0.22x`, and direct solve speedups remain `0.96x` to `0.98x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
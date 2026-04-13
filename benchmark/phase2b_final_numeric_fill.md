# Phase 2b Final Numeric Fill

Generated: `2026-04-12T19:35:19`

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
| `adaptivity_plan` | 1 | 5.31 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.10 ms | 5.95 MiB | 121226 | 1.29 | 0.65 |
| `adaptivity_plan` | 4 | 4.30 ms | 5.95 MiB | 121280 | 1.24 | 0.31 |
| `adaptivity_plan` | 6 | 3.79 ms | 5.96 MiB | 121358 | 1.40 | 0.23 |
| `assemble` | 1 | 1.786 s | 1441.46 MiB | 59837139 | 1.00 | 1.00 |
| `assemble` | 2 | 1.099 s | 1446.87 MiB | 59834209 | 1.62 | 0.81 |
| `assemble` | 4 | 575.32 ms | 1449.49 MiB | 59828310 | 3.10 | 0.78 |
| `assemble` | 6 | 380.45 ms | 1453.35 MiB | 59822386 | 4.69 | 0.78 |
| `preconditioner_build` | 1 | 12.72 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 78.32 ms | 133.92 MiB | 16369 | 0.16 | 0.08 |
| `preconditioner_build` | 4 | 75.75 ms | 134.02 MiB | 16404 | 0.17 | 0.04 |
| `preconditioner_build` | 6 | 80.88 ms | 134.29 MiB | 16442 | 0.16 | 0.03 |
| `solve_direct` | 1 | 19.11 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 18.73 ms | 28.44 MiB | 136 | 1.02 | 0.51 |
| `solve_direct` | 4 | 18.95 ms | 28.44 MiB | 136 | 1.01 | 0.25 |
| `solve_direct` | 6 | 19.54 ms | 28.44 MiB | 136 | 0.98 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 186.38 ms | 300.59 MiB | 9611513 | 1.00 | 1.00 |
| `assemble` | 2 | 103.96 ms | 301.38 MiB | 9611619 | 1.79 | 0.90 |
| `assemble` | 4 | 68.65 ms | 301.58 MiB | 9611769 | 2.72 | 0.68 |
| `assemble` | 6 | 49.73 ms | 299.95 MiB | 9611901 | 3.75 | 0.62 |
| `preconditioner_build` | 1 | 20.26 ms | 293.65 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 86.05 ms | 293.76 MiB | 16400 | 0.24 | 0.12 |
| `preconditioner_build` | 4 | 84.97 ms | 294.10 MiB | 16435 | 0.24 | 0.06 |
| `preconditioner_build` | 6 | 89.93 ms | 294.59 MiB | 16473 | 0.23 | 0.04 |
| `solve_direct` | 1 | 30.35 ms | 77.08 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 29.94 ms | 88.03 MiB | 108 | 1.01 | 0.51 |
| `solve_direct` | 4 | 29.80 ms | 94.60 MiB | 108 | 1.02 | 0.25 |
| `solve_direct` | 6 | 29.98 ms | 83.49 MiB | 108 | 1.01 | 0.17 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 70.17 ms | 23.52 MiB | 1270485 | 1.00 | 1.00 |
| `residual_bang` | 2 | 33.71 ms | 23.50 MiB | 1270587 | 2.08 | 1.04 |
| `residual_bang` | 4 | 45.16 ms | 23.48 MiB | 1270702 | 1.55 | 0.39 |
| `residual_bang` | 6 | 12.08 ms | 23.58 MiB | 1270822 | 5.81 | 0.97 |
| `tangent` | 1 | 300.78 ms | 230.36 MiB | 11317748 | 1.00 | 1.00 |
| `tangent` | 2 | 106.42 ms | 230.41 MiB | 11317866 | 2.83 | 1.41 |
| `tangent` | 4 | 95.21 ms | 230.43 MiB | 11318013 | 3.16 | 0.79 |
| `tangent` | 6 | 48.37 ms | 229.57 MiB | 11318141 | 6.22 | 1.04 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 728.19 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 697.48 μs | 0.97 MiB | 16728 | 1.04 | 0.52 |
| `adaptivity_plan` | 4 | 1.19 ms | 0.98 MiB | 16787 | 0.61 | 0.15 |
| `adaptivity_plan` | 6 | 836.63 μs | 0.98 MiB | 16855 | 0.87 | 0.15 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.786 s at 1 thread to 380.45 ms at 6 threads, a `4.69x` speedup with `0.78` efficiency. Allocation volume stays essentially flat at about `1441.46` to `1453.35 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 186.38 ms to 49.73 ms at 6 threads, or `3.75x` with `0.62` efficiency, while memory changes by about `1.00x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 70.17 ms to 12.08 ms at 6 threads, which is `5.81x` and `0.97` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 300.78 ms to 48.37 ms at 6 threads, or `6.22x` with `1.04` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.16x` to `0.23x`, and direct solve speedups remain `0.98x` to `1.01x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
# Phase 3 Retained Post Revert

Generated: `2026-04-13T09:04:07`

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

## Results

### `affine_cell_diffusion`

Continuous scalar Poisson problem with volume-dominated affine assembly.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 5.30 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.41 ms | 5.95 MiB | 121226 | 1.20 | 0.60 |
| `adaptivity_plan` | 4 | 4.27 ms | 5.95 MiB | 121291 | 1.24 | 0.31 |
| `adaptivity_plan` | 6 | 3.87 ms | 5.95 MiB | 121362 | 1.37 | 0.23 |
| `assemble` | 1 | 1.899 s | 1444.33 MiB | 59825908 | 1.00 | 1.00 |
| `assemble` | 2 | 1.081 s | 1448.44 MiB | 59823089 | 1.76 | 0.88 |
| `assemble` | 4 | 686.34 ms | 1451.12 MiB | 59817608 | 2.77 | 0.69 |
| `assemble` | 6 | 685.84 ms | 1453.47 MiB | 59812338 | 2.77 | 0.46 |
| `preconditioner_build` | 1 | 12.08 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 79.41 ms | 133.01 MiB | 16369 | 0.15 | 0.08 |
| `preconditioner_build` | 4 | 77.27 ms | 134.02 MiB | 16404 | 0.16 | 0.04 |
| `preconditioner_build` | 6 | 83.28 ms | 134.30 MiB | 16442 | 0.15 | 0.02 |
| `solve_direct` | 1 | 19.36 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 19.17 ms | 28.19 MiB | 136 | 1.01 | 0.50 |
| `solve_direct` | 4 | 19.20 ms | 28.44 MiB | 136 | 1.01 | 0.25 |
| `solve_direct` | 6 | 19.76 ms | 28.44 MiB | 136 | 0.98 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 229.66 ms | 304.02 MiB | 9603495 | 1.00 | 1.00 |
| `assemble` | 2 | 126.38 ms | 337.83 MiB | 9603699 | 1.82 | 0.91 |
| `assemble` | 4 | 91.67 ms | 358.11 MiB | 9604242 | 2.51 | 0.63 |
| `assemble` | 6 | 73.66 ms | 362.14 MiB | 9604994 | 3.12 | 0.52 |
| `preconditioner_build` | 1 | 20.39 ms | 290.10 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 87.89 ms | 290.22 MiB | 16400 | 0.23 | 0.12 |
| `preconditioner_build` | 4 | 86.74 ms | 290.74 MiB | 16435 | 0.24 | 0.06 |
| `preconditioner_build` | 6 | 92.77 ms | 294.50 MiB | 16473 | 0.22 | 0.04 |
| `solve_direct` | 1 | 30.10 ms | 78.16 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 30.24 ms | 79.36 MiB | 108 | 1.00 | 0.50 |
| `solve_direct` | 4 | 30.37 ms | 84.68 MiB | 108 | 0.99 | 0.25 |
| `solve_direct` | 6 | 64.63 ms | 112.00 MiB | 136 | 0.47 | 0.08 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 38.62 ms | 23.42 MiB | 1270475 | 1.00 | 1.00 |
| `residual_bang` | 2 | 22.03 ms | 24.31 MiB | 1270707 | 1.75 | 0.88 |
| `residual_bang` | 4 | 11.76 ms | 24.56 MiB | 1271078 | 3.28 | 0.82 |
| `residual_bang` | 6 | 7.58 ms | 24.59 MiB | 1271530 | 5.09 | 0.85 |
| `tangent` | 1 | 169.82 ms | 235.74 MiB | 11317737 | 1.00 | 1.00 |
| `tangent` | 2 | 112.78 ms | 248.42 MiB | 11318004 | 1.51 | 0.75 |
| `tangent` | 4 | 64.88 ms | 256.86 MiB | 11318537 | 2.62 | 0.65 |
| `tangent` | 6 | 47.23 ms | 260.72 MiB | 11319235 | 3.60 | 0.60 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.899 s at 1 thread to 685.84 ms at 6 threads, a `2.77x` speedup with `0.46` efficiency. Allocation volume stays essentially flat at about `1444.33` to `1453.47 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 229.66 ms to 73.66 ms at 6 threads, or `3.12x` with `0.52` efficiency, while memory changes by about `1.19x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 38.62 ms to 7.58 ms at 6 threads, which is `5.09x` and `0.85` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 169.82 ms to 47.23 ms at 6 threads, or `3.60x` with `0.60` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.15x` to `0.22x`, and direct solve speedups remain `0.47x` to `0.98x`.
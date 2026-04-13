# Remote Current 9950X

Generated: `2026-04-13T08:52:01`

Primary thread counts: `1, 2, 4, 8, 16`

Primary scaling set capped at 6 threads on this host because the machine has 6 performance cores.

## Environment

- Julia: `1.12.5`
- OS / arch: `Linux` / `x86_64`
- CPU: `AMD Ryzen 9 9950X 16-Core Processor`
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
| `adaptivity_plan` | 1 | 3.71 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 3.43 ms | 5.85 MiB | 121226 | 1.08 | 0.54 |
| `adaptivity_plan` | 4 | 2.70 ms | 5.86 MiB | 121301 | 1.38 | 0.34 |
| `adaptivity_plan` | 8 | 2.42 ms | 5.87 MiB | 121439 | 1.53 | 0.19 |
| `adaptivity_plan` | 16 | 2.47 ms | 5.89 MiB | 121666 | 1.50 | 0.09 |
| `assemble` | 1 | 1.448 s | 1443.85 MiB | 59825908 | 1.00 | 1.00 |
| `assemble` | 2 | 639.54 ms | 1447.60 MiB | 59823089 | 2.26 | 1.13 |
| `assemble` | 4 | 353.19 ms | 1449.27 MiB | 59817600 | 4.10 | 1.03 |
| `assemble` | 8 | 222.03 ms | 1451.27 MiB | 59807178 | 6.52 | 0.82 |
| `assemble` | 16 | 196.75 ms | 1452.75 MiB | 59790123 | 7.36 | 0.46 |
| `preconditioner_build` | 1 | 18.01 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 14.54 ms | 127.27 MiB | 16369 | 1.24 | 0.62 |
| `preconditioner_build` | 4 | 22.67 ms | 127.36 MiB | 16404 | 0.79 | 0.20 |
| `preconditioner_build` | 8 | 20.03 ms | 127.76 MiB | 16478 | 0.90 | 0.11 |
| `preconditioner_build` | 16 | 20.38 ms | 128.18 MiB | 16593 | 0.88 | 0.06 |
| `solve_direct` | 1 | 13.24 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 15.20 ms | 27.92 MiB | 136 | 0.87 | 0.44 |
| `solve_direct` | 4 | 14.32 ms | 27.92 MiB | 136 | 0.92 | 0.23 |
| `solve_direct` | 8 | 14.38 ms | 27.92 MiB | 136 | 0.92 | 0.12 |
| `solve_direct` | 16 | 15.16 ms | 27.92 MiB | 136 | 0.87 | 0.05 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 186.80 ms | 303.70 MiB | 9603495 | 1.00 | 1.00 |
| `assemble` | 2 | 118.01 ms | 324.75 MiB | 9603699 | 1.58 | 0.79 |
| `assemble` | 4 | 84.52 ms | 343.98 MiB | 9604242 | 2.21 | 0.55 |
| `assemble` | 8 | 66.06 ms | 345.59 MiB | 9605926 | 2.83 | 0.35 |
| `assemble` | 16 | 53.66 ms | 348.06 MiB | 9610433 | 3.48 | 0.22 |
| `preconditioner_build` | 1 | 21.04 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 33.57 ms | 263.24 MiB | 16400 | 0.63 | 0.31 |
| `preconditioner_build` | 4 | 22.59 ms | 263.44 MiB | 16435 | 0.93 | 0.23 |
| `preconditioner_build` | 8 | 19.53 ms | 264.05 MiB | 16509 | 1.08 | 0.13 |
| `preconditioner_build` | 16 | 19.93 ms | 264.86 MiB | 16624 | 1.06 | 0.07 |
| `solve_direct` | 1 | 21.45 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 21.68 ms | 70.26 MiB | 108 | 0.99 | 0.49 |
| `solve_direct` | 4 | 21.58 ms | 70.26 MiB | 108 | 0.99 | 0.25 |
| `solve_direct` | 8 | 24.10 ms | 70.26 MiB | 108 | 0.89 | 0.11 |
| `solve_direct` | 16 | 24.04 ms | 70.26 MiB | 108 | 0.89 | 0.06 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 28.63 ms | 23.41 MiB | 1270475 | 1.00 | 1.00 |
| `residual_bang` | 2 | 14.58 ms | 24.20 MiB | 1270707 | 1.96 | 0.98 |
| `residual_bang` | 4 | 7.69 ms | 24.40 MiB | 1271078 | 3.73 | 0.93 |
| `residual_bang` | 8 | 3.84 ms | 24.92 MiB | 1272114 | 7.45 | 0.93 |
| `residual_bang` | 16 | 3.19 ms | 25.23 MiB | 1274757 | 8.97 | 0.56 |
| `tangent` | 1 | 136.38 ms | 235.69 MiB | 11317737 | 1.00 | 1.00 |
| `tangent` | 2 | 82.81 ms | 247.74 MiB | 11318004 | 1.65 | 0.82 |
| `tangent` | 4 | 53.13 ms | 255.55 MiB | 11318541 | 2.57 | 0.64 |
| `tangent` | 8 | 36.05 ms | 256.13 MiB | 11320121 | 3.78 | 0.47 |
| `tangent` | 16 | 33.09 ms | 256.76 MiB | 11324364 | 4.12 | 0.26 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.448 s at 1 thread to 196.75 ms at 16 threads, a `7.36x` speedup with `0.46` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1452.75 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 186.80 ms to 53.66 ms at 16 threads, or `3.48x` with `0.22` efficiency, while memory changes by about `1.15x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 28.63 ms to 3.19 ms at 16 threads, which is `8.97x` and `0.56` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 136.38 ms to 33.09 ms at 16 threads, or `4.12x` with `0.26` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.88x` to `1.06x`, and direct solve speedups remain `0.87x` to `0.89x`.
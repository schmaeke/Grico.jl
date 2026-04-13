# Remote Phase 4 Default 9950X

Generated: `2026-04-13T09:28:42`

Primary thread counts: `1, 2, 4, 8, 16`

Primary scaling set uses the requested thread counts on helios.

## Environment

- Julia: `1.12.5`
- OS / arch: `Linux` / `x86_64`
- Host: `helios`
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
| `adaptivity_plan` | 1 | 3.67 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 3.16 ms | 5.85 MiB | 121226 | 1.16 | 0.58 |
| `adaptivity_plan` | 4 | 2.67 ms | 5.85 MiB | 121299 | 1.37 | 0.34 |
| `adaptivity_plan` | 8 | 2.48 ms | 5.87 MiB | 121437 | 1.48 | 0.18 |
| `adaptivity_plan` | 16 | 2.44 ms | 5.89 MiB | 121674 | 1.50 | 0.09 |
| `assemble` | 1 | 1.205 s | 1443.85 MiB | 59825911 | 1.00 | 1.00 |
| `assemble` | 2 | 654.72 ms | 1444.01 MiB | 59823020 | 1.84 | 0.92 |
| `assemble` | 4 | 366.21 ms | 1444.05 MiB | 59817219 | 3.29 | 0.82 |
| `assemble` | 8 | 233.20 ms | 1445.01 MiB | 59805703 | 5.17 | 0.65 |
| `assemble` | 16 | 189.01 ms | 1445.60 MiB | 59786510 | 6.38 | 0.40 |
| `preconditioner_build` | 1 | 15.87 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 14.69 ms | 127.21 MiB | 16369 | 1.08 | 0.54 |
| `preconditioner_build` | 4 | 21.51 ms | 127.36 MiB | 16404 | 0.74 | 0.18 |
| `preconditioner_build` | 8 | 21.16 ms | 127.77 MiB | 16478 | 0.75 | 0.09 |
| `preconditioner_build` | 16 | 20.39 ms | 128.18 MiB | 16601 | 0.78 | 0.05 |
| `solve_direct` | 1 | 12.83 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 14.70 ms | 27.92 MiB | 136 | 0.87 | 0.44 |
| `solve_direct` | 4 | 15.39 ms | 27.92 MiB | 136 | 0.83 | 0.21 |
| `solve_direct` | 8 | 14.03 ms | 27.92 MiB | 136 | 0.91 | 0.11 |
| `solve_direct` | 16 | 14.17 ms | 27.92 MiB | 136 | 0.91 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 190.12 ms | 303.70 MiB | 9603502 | 1.00 | 1.00 |
| `assemble` | 2 | 117.27 ms | 304.07 MiB | 9603658 | 1.62 | 0.81 |
| `assemble` | 4 | 73.19 ms | 304.61 MiB | 9603905 | 2.60 | 0.65 |
| `assemble` | 8 | 52.40 ms | 305.60 MiB | 9604481 | 3.63 | 0.45 |
| `assemble` | 16 | 41.02 ms | 307.73 MiB | 9606024 | 4.63 | 0.29 |
| `preconditioner_build` | 1 | 19.89 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 25.46 ms | 263.24 MiB | 16400 | 0.78 | 0.39 |
| `preconditioner_build` | 4 | 30.16 ms | 263.43 MiB | 16435 | 0.66 | 0.16 |
| `preconditioner_build` | 8 | 17.82 ms | 264.04 MiB | 16509 | 1.12 | 0.14 |
| `preconditioner_build` | 16 | 18.84 ms | 264.77 MiB | 16624 | 1.06 | 0.07 |
| `solve_direct` | 1 | 21.53 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 22.23 ms | 70.26 MiB | 108 | 0.97 | 0.48 |
| `solve_direct` | 4 | 24.00 ms | 70.26 MiB | 108 | 0.90 | 0.22 |
| `solve_direct` | 8 | 23.17 ms | 70.26 MiB | 108 | 0.93 | 0.12 |
| `solve_direct` | 16 | 24.23 ms | 70.26 MiB | 108 | 0.89 | 0.06 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 35.61 ms | 23.41 MiB | 1270489 | 1.00 | 1.00 |
| `residual_bang` | 2 | 16.77 ms | 23.45 MiB | 1270689 | 2.12 | 1.06 |
| `residual_bang` | 4 | 9.33 ms | 23.49 MiB | 1270932 | 3.82 | 0.95 |
| `residual_bang` | 8 | 4.69 ms | 23.61 MiB | 1271504 | 7.59 | 0.95 |
| `residual_bang` | 16 | 4.38 ms | 23.85 MiB | 1273001 | 8.13 | 0.51 |
| `tangent` | 1 | 142.65 ms | 235.69 MiB | 11317751 | 1.00 | 1.00 |
| `tangent` | 2 | 84.85 ms | 235.87 MiB | 11317970 | 1.68 | 0.84 |
| `tangent` | 4 | 50.93 ms | 236.33 MiB | 11318275 | 2.80 | 0.70 |
| `tangent` | 8 | 33.20 ms | 237.22 MiB | 11318955 | 4.30 | 0.54 |
| `tangent` | 16 | 28.16 ms | 238.94 MiB | 11320668 | 5.07 | 0.32 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.205 s at 1 thread to 189.01 ms at 16 threads, a `6.38x` speedup with `0.40` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1445.60 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 190.12 ms to 41.02 ms at 16 threads, or `4.63x` with `0.29` efficiency, while memory changes by about `1.01x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 35.61 ms to 4.38 ms at 16 threads, which is `8.13x` and `0.51` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 142.65 ms to 28.16 ms at 16 threads, or `5.07x` with `0.32` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.78x` to `1.06x`, and direct solve speedups remain `0.89x` to `0.91x`.
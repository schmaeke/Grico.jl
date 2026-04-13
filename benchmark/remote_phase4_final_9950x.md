# Remote Phase 4 Final 9950X

Generated: `2026-04-13T09:46:21`

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
| `adaptivity_plan` | 1 | 3.74 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 3.44 ms | 5.85 MiB | 121226 | 1.09 | 0.54 |
| `adaptivity_plan` | 4 | 2.60 ms | 5.85 MiB | 121299 | 1.44 | 0.36 |
| `adaptivity_plan` | 8 | 2.52 ms | 5.87 MiB | 121437 | 1.48 | 0.19 |
| `adaptivity_plan` | 16 | 2.49 ms | 5.89 MiB | 121685 | 1.50 | 0.09 |
| `assemble` | 1 | 1.209 s | 1443.85 MiB | 59825911 | 1.00 | 1.00 |
| `assemble` | 2 | 689.25 ms | 1444.01 MiB | 59823020 | 1.75 | 0.88 |
| `assemble` | 4 | 370.49 ms | 1444.05 MiB | 59817219 | 3.26 | 0.82 |
| `assemble` | 8 | 238.10 ms | 1445.01 MiB | 59805703 | 5.08 | 0.63 |
| `assemble` | 16 | 185.80 ms | 1445.60 MiB | 59786510 | 6.51 | 0.41 |
| `preconditioner_build` | 1 | 21.15 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 14.82 ms | 127.26 MiB | 16369 | 1.43 | 0.71 |
| `preconditioner_build` | 4 | 22.21 ms | 127.36 MiB | 16404 | 0.95 | 0.24 |
| `preconditioner_build` | 8 | 20.71 ms | 127.76 MiB | 16478 | 1.02 | 0.13 |
| `preconditioner_build` | 16 | 21.34 ms | 128.08 MiB | 16591 | 0.99 | 0.06 |
| `solve_direct` | 1 | 13.60 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 15.32 ms | 27.92 MiB | 136 | 0.89 | 0.44 |
| `solve_direct` | 4 | 14.43 ms | 27.92 MiB | 136 | 0.94 | 0.24 |
| `solve_direct` | 8 | 14.37 ms | 27.92 MiB | 136 | 0.95 | 0.12 |
| `solve_direct` | 16 | 15.16 ms | 27.92 MiB | 136 | 0.90 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 194.59 ms | 303.70 MiB | 9603502 | 1.00 | 1.00 |
| `assemble` | 2 | 116.46 ms | 304.07 MiB | 9603658 | 1.67 | 0.84 |
| `assemble` | 4 | 75.15 ms | 304.61 MiB | 9603905 | 2.59 | 0.65 |
| `assemble` | 8 | 55.65 ms | 305.60 MiB | 9604481 | 3.50 | 0.44 |
| `assemble` | 16 | 45.62 ms | 307.73 MiB | 9606024 | 4.26 | 0.27 |
| `preconditioner_build` | 1 | 22.00 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 26.22 ms | 263.24 MiB | 16400 | 0.84 | 0.42 |
| `preconditioner_build` | 4 | 29.15 ms | 263.44 MiB | 16435 | 0.75 | 0.19 |
| `preconditioner_build` | 8 | 19.20 ms | 264.04 MiB | 16509 | 1.15 | 0.14 |
| `preconditioner_build` | 16 | 35.57 ms | 264.86 MiB | 16624 | 0.62 | 0.04 |
| `solve_direct` | 1 | 25.08 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 22.22 ms | 70.26 MiB | 108 | 1.13 | 0.56 |
| `solve_direct` | 4 | 22.97 ms | 70.26 MiB | 108 | 1.09 | 0.27 |
| `solve_direct` | 8 | 22.73 ms | 70.26 MiB | 108 | 1.10 | 0.14 |
| `solve_direct` | 16 | 25.46 ms | 70.26 MiB | 108 | 0.99 | 0.06 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 60.91 ms | 23.41 MiB | 1270489 | 1.00 | 1.00 |
| `residual_bang` | 2 | 33.56 ms | 24.20 MiB | 1270728 | 1.81 | 0.91 |
| `residual_bang` | 4 | 16.16 ms | 24.40 MiB | 1271099 | 3.77 | 0.94 |
| `residual_bang` | 8 | 8.37 ms | 24.90 MiB | 1272131 | 7.28 | 0.91 |
| `residual_bang` | 16 | 5.98 ms | 25.29 MiB | 1274796 | 10.18 | 0.64 |
| `tangent` | 1 | 175.63 ms | 235.69 MiB | 11317751 | 1.00 | 1.00 |
| `tangent` | 2 | 98.23 ms | 235.87 MiB | 11317970 | 1.79 | 0.89 |
| `tangent` | 4 | 57.82 ms | 236.33 MiB | 11318275 | 3.04 | 0.76 |
| `tangent` | 8 | 34.77 ms | 237.22 MiB | 11318955 | 5.05 | 0.63 |
| `tangent` | 16 | 32.92 ms | 238.94 MiB | 11320668 | 5.33 | 0.33 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.209 s at 1 thread to 185.80 ms at 16 threads, a `6.51x` speedup with `0.41` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1445.60 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 194.59 ms to 45.62 ms at 16 threads, or `4.26x` with `0.27` efficiency, while memory changes by about `1.01x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 60.91 ms to 5.98 ms at 16 threads, which is `10.18x` and `0.64` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 175.63 ms to 32.92 ms at 16 threads, or `5.33x` with `0.33` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.62x` to `0.99x`, and direct solve speedups remain `0.90x` to `0.99x`.
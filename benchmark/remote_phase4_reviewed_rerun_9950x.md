# Remote Phase 4 Reviewed Rerun 9950X

Generated: `2026-04-13T10:17:37`

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
| `adaptivity_plan` | 1 | 3.75 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 3.00 ms | 5.84 MiB | 121224 | 1.25 | 0.62 |
| `adaptivity_plan` | 4 | 2.67 ms | 5.86 MiB | 121301 | 1.40 | 0.35 |
| `adaptivity_plan` | 8 | 2.57 ms | 5.88 MiB | 121439 | 1.46 | 0.18 |
| `adaptivity_plan` | 16 | 2.44 ms | 5.88 MiB | 121676 | 1.53 | 0.10 |
| `assemble` | 1 | 1.215 s | 1443.85 MiB | 59825911 | 1.00 | 1.00 |
| `assemble` | 2 | 649.29 ms | 1444.01 MiB | 59823020 | 1.87 | 0.94 |
| `assemble` | 4 | 427.65 ms | 1444.05 MiB | 59817219 | 2.84 | 0.71 |
| `assemble` | 8 | 240.46 ms | 1445.01 MiB | 59805703 | 5.05 | 0.63 |
| `assemble` | 16 | 194.99 ms | 1445.60 MiB | 59786510 | 6.23 | 0.39 |
| `preconditioner_build` | 1 | 18.70 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 14.97 ms | 127.27 MiB | 16369 | 1.25 | 0.62 |
| `preconditioner_build` | 4 | 22.94 ms | 127.36 MiB | 16404 | 0.82 | 0.20 |
| `preconditioner_build` | 8 | 21.87 ms | 127.77 MiB | 16478 | 0.86 | 0.11 |
| `preconditioner_build` | 16 | 20.84 ms | 128.08 MiB | 16593 | 0.90 | 0.06 |
| `solve_direct` | 1 | 13.64 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 14.36 ms | 27.92 MiB | 136 | 0.95 | 0.47 |
| `solve_direct` | 4 | 15.27 ms | 27.92 MiB | 136 | 0.89 | 0.22 |
| `solve_direct` | 8 | 14.31 ms | 27.92 MiB | 136 | 0.95 | 0.12 |
| `solve_direct` | 16 | 14.86 ms | 27.92 MiB | 136 | 0.92 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 190.88 ms | 303.70 MiB | 9603502 | 1.00 | 1.00 |
| `assemble` | 2 | 116.43 ms | 304.07 MiB | 9603658 | 1.64 | 0.82 |
| `assemble` | 4 | 77.45 ms | 304.61 MiB | 9603905 | 2.46 | 0.62 |
| `assemble` | 8 | 52.03 ms | 305.60 MiB | 9604481 | 3.67 | 0.46 |
| `assemble` | 16 | 44.31 ms | 307.73 MiB | 9606024 | 4.31 | 0.27 |
| `preconditioner_build` | 1 | 19.87 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 25.61 ms | 263.24 MiB | 16400 | 0.78 | 0.39 |
| `preconditioner_build` | 4 | 32.90 ms | 263.44 MiB | 16435 | 0.60 | 0.15 |
| `preconditioner_build` | 8 | 18.98 ms | 264.04 MiB | 16509 | 1.05 | 0.13 |
| `preconditioner_build` | 16 | 20.05 ms | 264.76 MiB | 16624 | 0.99 | 0.06 |
| `solve_direct` | 1 | 21.37 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 22.67 ms | 70.26 MiB | 108 | 0.94 | 0.47 |
| `solve_direct` | 4 | 21.57 ms | 70.26 MiB | 108 | 0.99 | 0.25 |
| `solve_direct` | 8 | 21.67 ms | 70.26 MiB | 108 | 0.99 | 0.12 |
| `solve_direct` | 16 | 24.93 ms | 70.26 MiB | 108 | 0.86 | 0.05 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 68.27 ms | 23.41 MiB | 1270489 | 1.00 | 1.00 |
| `residual_bang` | 2 | 14.17 ms | 23.45 MiB | 1270689 | 4.82 | 2.41 |
| `residual_bang` | 4 | 18.33 ms | 23.49 MiB | 1270932 | 3.72 | 0.93 |
| `residual_bang` | 8 | 9.32 ms | 23.61 MiB | 1271504 | 7.32 | 0.92 |
| `residual_bang` | 16 | 11.76 ms | 23.85 MiB | 1273001 | 5.80 | 0.36 |
| `tangent` | 1 | 180.23 ms | 235.69 MiB | 11317751 | 1.00 | 1.00 |
| `tangent` | 2 | 80.74 ms | 235.87 MiB | 11317970 | 2.23 | 1.12 |
| `tangent` | 4 | 58.24 ms | 236.33 MiB | 11318275 | 3.09 | 0.77 |
| `tangent` | 8 | 39.78 ms | 237.22 MiB | 11318955 | 4.53 | 0.57 |
| `tangent` | 16 | 40.94 ms | 238.94 MiB | 11320668 | 4.40 | 0.28 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.215 s at 1 thread to 194.99 ms at 16 threads, a `6.23x` speedup with `0.39` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1445.60 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 190.88 ms to 44.31 ms at 16 threads, or `4.31x` with `0.27` efficiency, while memory changes by about `1.01x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 68.27 ms to 11.76 ms at 16 threads, which is `5.80x` and `0.36` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 180.23 ms to 40.94 ms at 16 threads, or `4.40x` with `0.28` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.90x` to `0.99x`, and direct solve speedups remain `0.86x` to `0.92x`.
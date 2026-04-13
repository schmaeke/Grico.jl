# Remote Phase 4 Reviewed 9950X

Generated: `2026-04-13T10:09:02`

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
| `adaptivity_plan` | 1 | 3.82 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 3.06 ms | 5.85 MiB | 121226 | 1.25 | 0.62 |
| `adaptivity_plan` | 4 | 2.67 ms | 5.86 MiB | 121301 | 1.43 | 0.36 |
| `adaptivity_plan` | 8 | 2.41 ms | 5.88 MiB | 121441 | 1.59 | 0.20 |
| `adaptivity_plan` | 16 | 2.37 ms | 5.89 MiB | 121684 | 1.61 | 0.10 |
| `assemble` | 1 | 1.309 s | 1443.85 MiB | 59825911 | 1.00 | 1.00 |
| `assemble` | 2 | 936.34 ms | 1444.01 MiB | 59823020 | 1.40 | 0.70 |
| `assemble` | 4 | 499.79 ms | 1444.05 MiB | 59817219 | 2.62 | 0.65 |
| `assemble` | 8 | 235.27 ms | 1445.01 MiB | 59805703 | 5.56 | 0.70 |
| `assemble` | 16 | 205.19 ms | 1445.60 MiB | 59786510 | 6.38 | 0.40 |
| `preconditioner_build` | 1 | 18.61 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 12.87 ms | 127.26 MiB | 16367 | 1.45 | 0.72 |
| `preconditioner_build` | 4 | 22.84 ms | 127.36 MiB | 16404 | 0.81 | 0.20 |
| `preconditioner_build` | 8 | 20.28 ms | 127.76 MiB | 16478 | 0.92 | 0.11 |
| `preconditioner_build` | 16 | 20.25 ms | 128.08 MiB | 16593 | 0.92 | 0.06 |
| `solve_direct` | 1 | 12.76 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 15.38 ms | 27.92 MiB | 136 | 0.83 | 0.42 |
| `solve_direct` | 4 | 14.47 ms | 27.92 MiB | 136 | 0.88 | 0.22 |
| `solve_direct` | 8 | 14.51 ms | 27.92 MiB | 136 | 0.88 | 0.11 |
| `solve_direct` | 16 | 14.04 ms | 27.92 MiB | 136 | 0.91 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 211.50 ms | 303.70 MiB | 9603502 | 1.00 | 1.00 |
| `assemble` | 2 | 117.59 ms | 304.07 MiB | 9603658 | 1.80 | 0.90 |
| `assemble` | 4 | 72.88 ms | 304.61 MiB | 9603905 | 2.90 | 0.73 |
| `assemble` | 8 | 53.03 ms | 305.60 MiB | 9604481 | 3.99 | 0.50 |
| `assemble` | 16 | 41.35 ms | 307.73 MiB | 9606024 | 5.11 | 0.32 |
| `preconditioner_build` | 1 | 22.40 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 22.62 ms | 263.24 MiB | 16400 | 0.99 | 0.50 |
| `preconditioner_build` | 4 | 30.83 ms | 263.43 MiB | 16435 | 0.73 | 0.18 |
| `preconditioner_build` | 8 | 18.97 ms | 263.96 MiB | 16509 | 1.18 | 0.15 |
| `preconditioner_build` | 16 | 21.52 ms | 264.85 MiB | 16624 | 1.04 | 0.07 |
| `solve_direct` | 1 | 25.48 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 22.50 ms | 70.26 MiB | 108 | 1.13 | 0.57 |
| `solve_direct` | 4 | 22.64 ms | 70.26 MiB | 108 | 1.13 | 0.28 |
| `solve_direct` | 8 | 24.97 ms | 70.26 MiB | 108 | 1.02 | 0.13 |
| `solve_direct` | 16 | 24.52 ms | 70.26 MiB | 108 | 1.04 | 0.06 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 64.98 ms | 23.41 MiB | 1270489 | 1.00 | 1.00 |
| `residual_bang` | 2 | 34.22 ms | 23.45 MiB | 1270689 | 1.90 | 0.95 |
| `residual_bang` | 4 | 33.14 ms | 23.49 MiB | 1270932 | 1.96 | 0.49 |
| `residual_bang` | 8 | 11.59 ms | 23.61 MiB | 1271504 | 5.60 | 0.70 |
| `residual_bang` | 16 | 9.19 ms | 23.85 MiB | 1273001 | 7.07 | 0.44 |
| `tangent` | 1 | 182.13 ms | 235.69 MiB | 11317751 | 1.00 | 1.00 |
| `tangent` | 2 | 103.88 ms | 235.87 MiB | 11317970 | 1.75 | 0.88 |
| `tangent` | 4 | 76.67 ms | 236.33 MiB | 11318275 | 2.38 | 0.59 |
| `tangent` | 8 | 36.08 ms | 237.22 MiB | 11318955 | 5.05 | 0.63 |
| `tangent` | 16 | 33.75 ms | 238.94 MiB | 11320668 | 5.40 | 0.34 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.309 s at 1 thread to 205.19 ms at 16 threads, a `6.38x` speedup with `0.40` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1445.60 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 211.50 ms to 41.35 ms at 16 threads, or `5.11x` with `0.32` efficiency, while memory changes by about `1.01x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 64.98 ms to 9.19 ms at 16 threads, which is `7.07x` and `0.44` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 182.13 ms to 33.75 ms at 16 threads, or `5.40x` with `0.34` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.92x` to `1.04x`, and direct solve speedups remain `0.91x` to `1.04x`.
# Remote Phase 4 Dynamic 9950X

Generated: `2026-04-13T09:39:33`

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
- `GRICO_SCHEDULER_OVERRIDE`: `dynamic`

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
| `adaptivity_plan` | 1 | 3.72 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 3.25 ms | 5.84 MiB | 121224 | 1.14 | 0.57 |
| `adaptivity_plan` | 4 | 2.92 ms | 5.86 MiB | 121301 | 1.27 | 0.32 |
| `adaptivity_plan` | 8 | 2.54 ms | 5.87 MiB | 121437 | 1.46 | 0.18 |
| `adaptivity_plan` | 16 | 2.54 ms | 5.89 MiB | 121676 | 1.46 | 0.09 |
| `assemble` | 1 | 1.676 s | 1443.85 MiB | 59825916 | 1.00 | 1.00 |
| `assemble` | 2 | 657.89 ms | 1447.60 MiB | 59823098 | 2.55 | 1.27 |
| `assemble` | 4 | 352.60 ms | 1448.90 MiB | 59817609 | 4.75 | 1.19 |
| `assemble` | 8 | 240.90 ms | 1451.10 MiB | 59807211 | 6.96 | 0.87 |
| `assemble` | 16 | 195.32 ms | 1452.60 MiB | 59790108 | 8.58 | 0.54 |
| `preconditioner_build` | 1 | 17.41 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 14.93 ms | 127.27 MiB | 16369 | 1.17 | 0.58 |
| `preconditioner_build` | 4 | 21.71 ms | 127.36 MiB | 16404 | 0.80 | 0.20 |
| `preconditioner_build` | 8 | 20.04 ms | 127.75 MiB | 16478 | 0.87 | 0.11 |
| `preconditioner_build` | 16 | 21.44 ms | 128.08 MiB | 16594 | 0.81 | 0.05 |
| `solve_direct` | 1 | 12.97 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 14.19 ms | 27.92 MiB | 136 | 0.91 | 0.46 |
| `solve_direct` | 4 | 14.35 ms | 27.92 MiB | 136 | 0.90 | 0.23 |
| `solve_direct` | 8 | 14.98 ms | 27.92 MiB | 136 | 0.87 | 0.11 |
| `solve_direct` | 16 | 14.14 ms | 27.92 MiB | 136 | 0.92 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 190.55 ms | 303.70 MiB | 9603517 | 1.00 | 1.00 |
| `assemble` | 2 | 119.82 ms | 324.75 MiB | 9603724 | 1.59 | 0.80 |
| `assemble` | 4 | 81.96 ms | 343.98 MiB | 9604267 | 2.32 | 0.58 |
| `assemble` | 8 | 67.00 ms | 347.16 MiB | 9605971 | 2.84 | 0.36 |
| `assemble` | 16 | 55.25 ms | 347.10 MiB | 9610422 | 3.45 | 0.22 |
| `preconditioner_build` | 1 | 20.43 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 23.40 ms | 263.24 MiB | 16400 | 0.87 | 0.44 |
| `preconditioner_build` | 4 | 32.04 ms | 263.44 MiB | 16435 | 0.64 | 0.16 |
| `preconditioner_build` | 8 | 18.32 ms | 264.05 MiB | 16509 | 1.11 | 0.14 |
| `preconditioner_build` | 16 | 20.24 ms | 264.86 MiB | 16632 | 1.01 | 0.06 |
| `solve_direct` | 1 | 21.43 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 21.54 ms | 70.26 MiB | 108 | 0.99 | 0.50 |
| `solve_direct` | 4 | 23.00 ms | 70.26 MiB | 108 | 0.93 | 0.23 |
| `solve_direct` | 8 | 24.35 ms | 70.26 MiB | 108 | 0.88 | 0.11 |
| `solve_direct` | 16 | 25.40 ms | 70.26 MiB | 108 | 0.84 | 0.05 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 30.66 ms | 23.41 MiB | 1270524 | 1.00 | 1.00 |
| `residual_bang` | 2 | 17.90 ms | 24.20 MiB | 1270763 | 1.71 | 0.86 |
| `residual_bang` | 4 | 13.47 ms | 24.41 MiB | 1271134 | 2.28 | 0.57 |
| `residual_bang` | 8 | 8.72 ms | 24.92 MiB | 1272170 | 3.52 | 0.44 |
| `residual_bang` | 16 | 3.75 ms | 25.26 MiB | 1274865 | 8.17 | 0.51 |
| `tangent` | 1 | 143.42 ms | 235.69 MiB | 11317786 | 1.00 | 1.00 |
| `tangent` | 2 | 88.22 ms | 247.75 MiB | 11318060 | 1.63 | 0.81 |
| `tangent` | 4 | 58.04 ms | 255.55 MiB | 11318597 | 2.47 | 0.62 |
| `tangent` | 8 | 40.44 ms | 256.96 MiB | 11320169 | 3.55 | 0.44 |
| `tangent` | 16 | 32.64 ms | 256.82 MiB | 11324382 | 4.39 | 0.27 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.676 s at 1 thread to 195.32 ms at 16 threads, a `8.58x` speedup with `0.54` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1452.60 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 190.55 ms to 55.25 ms at 16 threads, or `3.45x` with `0.22` efficiency, while memory changes by about `1.14x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 30.66 ms to 3.75 ms at 16 threads, which is `8.17x` and `0.51` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 143.42 ms to 32.64 ms at 16 threads, or `4.39x` with `0.27` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.81x` to `1.01x`, and direct solve speedups remain `0.84x` to `0.92x`.
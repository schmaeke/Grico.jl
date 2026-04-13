# Remote Current No Direct 9950X

Generated: `2026-04-13T08:54:25`

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
| `adaptivity_plan` | 1 | 3.72 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 3.02 ms | 5.85 MiB | 121226 | 1.23 | 0.62 |
| `adaptivity_plan` | 4 | 2.80 ms | 5.86 MiB | 121301 | 1.33 | 0.33 |
| `adaptivity_plan` | 8 | 2.48 ms | 5.87 MiB | 121437 | 1.50 | 0.19 |
| `adaptivity_plan` | 16 | 2.44 ms | 5.88 MiB | 121666 | 1.52 | 0.10 |
| `assemble` | 1 | 1.279 s | 1443.85 MiB | 59825908 | 1.00 | 1.00 |
| `assemble` | 2 | 702.04 ms | 1447.60 MiB | 59823089 | 1.82 | 0.91 |
| `assemble` | 4 | 359.12 ms | 1449.27 MiB | 59817600 | 3.56 | 0.89 |
| `assemble` | 8 | 243.02 ms | 1451.42 MiB | 59807156 | 5.26 | 0.66 |
| `assemble` | 16 | 194.19 ms | 1452.83 MiB | 59790189 | 6.58 | 0.41 |
| `preconditioner_build` | 1 | 18.80 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 15.59 ms | 127.27 MiB | 16369 | 1.21 | 0.60 |
| `preconditioner_build` | 4 | 21.63 ms | 127.36 MiB | 16404 | 0.87 | 0.22 |
| `preconditioner_build` | 8 | 20.40 ms | 127.74 MiB | 16476 | 0.92 | 0.12 |
| `preconditioner_build` | 16 | 20.24 ms | 128.08 MiB | 16593 | 0.93 | 0.06 |
| `solve_direct` | 1 | 13.27 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 15.75 ms | 27.92 MiB | 136 | 0.84 | 0.42 |
| `solve_direct` | 4 | 14.84 ms | 27.92 MiB | 136 | 0.89 | 0.22 |
| `solve_direct` | 8 | 14.19 ms | 27.92 MiB | 136 | 0.94 | 0.12 |
| `solve_direct` | 16 | 14.49 ms | 27.92 MiB | 136 | 0.92 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 194.57 ms | 303.70 MiB | 9603495 | 1.00 | 1.00 |
| `assemble` | 2 | 116.54 ms | 324.75 MiB | 9603699 | 1.67 | 0.83 |
| `assemble` | 4 | 85.82 ms | 343.98 MiB | 9604242 | 2.27 | 0.57 |
| `assemble` | 8 | 67.82 ms | 347.58 MiB | 9605950 | 2.87 | 0.36 |
| `assemble` | 16 | 53.17 ms | 346.35 MiB | 9610293 | 3.66 | 0.23 |
| `preconditioner_build` | 1 | 22.01 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 21.83 ms | 263.19 MiB | 16398 | 1.01 | 0.50 |
| `preconditioner_build` | 4 | 31.38 ms | 263.43 MiB | 16435 | 0.70 | 0.18 |
| `preconditioner_build` | 8 | 18.90 ms | 264.04 MiB | 16509 | 1.16 | 0.15 |
| `preconditioner_build` | 16 | 18.78 ms | 264.77 MiB | 16624 | 1.17 | 0.07 |
| `solve_direct` | 1 | 21.44 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 21.64 ms | 70.26 MiB | 108 | 0.99 | 0.50 |
| `solve_direct` | 4 | 21.35 ms | 70.26 MiB | 108 | 1.00 | 0.25 |
| `solve_direct` | 8 | 23.36 ms | 70.26 MiB | 108 | 0.92 | 0.11 |
| `solve_direct` | 16 | 23.79 ms | 70.26 MiB | 108 | 0.90 | 0.06 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 27.03 ms | 23.41 MiB | 1270475 | 1.00 | 1.00 |
| `residual_bang` | 2 | 14.79 ms | 24.20 MiB | 1270707 | 1.83 | 0.91 |
| `residual_bang` | 4 | 7.26 ms | 24.40 MiB | 1271078 | 3.72 | 0.93 |
| `residual_bang` | 8 | 4.06 ms | 24.88 MiB | 1272106 | 6.65 | 0.83 |
| `residual_bang` | 16 | 3.16 ms | 25.23 MiB | 1274767 | 8.56 | 0.54 |
| `tangent` | 1 | 135.96 ms | 235.69 MiB | 11317737 | 1.00 | 1.00 |
| `tangent` | 2 | 81.65 ms | 247.74 MiB | 11318004 | 1.67 | 0.83 |
| `tangent` | 4 | 51.28 ms | 254.70 MiB | 11318537 | 2.65 | 0.66 |
| `tangent` | 8 | 36.56 ms | 256.34 MiB | 11320125 | 3.72 | 0.46 |
| `tangent` | 16 | 32.67 ms | 256.36 MiB | 11324334 | 4.16 | 0.26 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.279 s at 1 thread to 194.19 ms at 16 threads, a `6.58x` speedup with `0.41` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1452.83 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 194.57 ms to 53.17 ms at 16 threads, or `3.66x` with `0.23` efficiency, while memory changes by about `1.14x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 27.03 ms to 3.16 ms at 16 threads, which is `8.56x` and `0.54` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 135.96 ms to 32.67 ms at 16 threads, or `4.16x` with `0.26` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.93x` to `1.17x`, and direct solve speedups remain `0.90x` to `0.92x`.
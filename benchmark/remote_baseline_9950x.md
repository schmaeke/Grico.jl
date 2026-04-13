# Remote Baseline 9950X

Generated: `2026-04-13T08:48:43`

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
| `adaptivity_plan` | 1 | 3.74 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 2.94 ms | 5.85 MiB | 121226 | 1.27 | 0.64 |
| `adaptivity_plan` | 4 | 2.95 ms | 5.86 MiB | 121301 | 1.27 | 0.32 |
| `adaptivity_plan` | 8 | 2.56 ms | 5.86 MiB | 121434 | 1.46 | 0.18 |
| `adaptivity_plan` | 16 | 2.46 ms | 5.89 MiB | 121688 | 1.52 | 0.10 |
| `assemble` | 1 | 1.193 s | 1472.78 MiB | 59940922 | 1.00 | 1.00 |
| `assemble` | 2 | 621.64 ms | 1469.44 MiB | 59939037 | 1.92 | 0.96 |
| `assemble` | 4 | 346.60 ms | 1475.87 MiB | 59935212 | 3.44 | 0.86 |
| `assemble` | 8 | 255.62 ms | 1477.42 MiB | 59927508 | 4.67 | 0.58 |
| `assemble` | 16 | 199.29 ms | 1478.04 MiB | 59915525 | 5.99 | 0.37 |
| `preconditioner_build` | 1 | 16.58 ms | 125.99 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 13.32 ms | 126.00 MiB | 16369 | 1.24 | 0.62 |
| `preconditioner_build` | 4 | 21.73 ms | 126.14 MiB | 16408 | 0.76 | 0.19 |
| `preconditioner_build` | 8 | 21.07 ms | 126.54 MiB | 16478 | 0.79 | 0.10 |
| `preconditioner_build` | 16 | 20.35 ms | 126.95 MiB | 16593 | 0.81 | 0.05 |
| `solve_direct` | 1 | 14.62 ms | 28.76 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 16.62 ms | 28.76 MiB | 136 | 0.88 | 0.44 |
| `solve_direct` | 4 | 16.66 ms | 28.76 MiB | 136 | 0.88 | 0.22 |
| `solve_direct` | 8 | 17.29 ms | 28.76 MiB | 136 | 0.85 | 0.11 |
| `solve_direct` | 16 | 16.19 ms | 28.76 MiB | 136 | 0.90 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 133.41 ms | 437.20 MiB | 10008644 | 1.00 | 1.00 |
| `assemble` | 2 | 102.16 ms | 466.05 MiB | 10008794 | 1.31 | 0.65 |
| `assemble` | 4 | 95.27 ms | 468.80 MiB | 10009004 | 1.40 | 0.35 |
| `assemble` | 8 | 93.06 ms | 467.43 MiB | 10009402 | 1.43 | 0.18 |
| `assemble` | 16 | 83.87 ms | 477.53 MiB | 10010146 | 1.59 | 0.10 |
| `preconditioner_build` | 1 | 23.66 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 29.68 ms | 263.24 MiB | 16400 | 0.80 | 0.40 |
| `preconditioner_build` | 4 | 21.97 ms | 263.43 MiB | 16435 | 1.08 | 0.27 |
| `preconditioner_build` | 8 | 27.32 ms | 264.04 MiB | 16509 | 0.87 | 0.11 |
| `preconditioner_build` | 16 | 20.98 ms | 264.76 MiB | 16624 | 1.13 | 0.07 |
| `solve_direct` | 1 | 21.61 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 25.40 ms | 70.26 MiB | 108 | 0.85 | 0.43 |
| `solve_direct` | 4 | 25.86 ms | 70.26 MiB | 108 | 0.84 | 0.21 |
| `solve_direct` | 8 | 24.93 ms | 70.26 MiB | 108 | 0.87 | 0.11 |
| `solve_direct` | 16 | 25.47 ms | 70.26 MiB | 108 | 0.85 | 0.05 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 62.94 ms | 23.35 MiB | 1270452 | 1.00 | 1.00 |
| `residual_bang` | 2 | 30.30 ms | 23.42 MiB | 1270535 | 2.08 | 1.04 |
| `residual_bang` | 4 | 15.23 ms | 23.55 MiB | 1270620 | 4.13 | 1.03 |
| `residual_bang` | 8 | 7.99 ms | 23.80 MiB | 1270784 | 7.88 | 0.98 |
| `residual_bang` | 16 | 5.61 ms | 24.31 MiB | 1271115 | 11.22 | 0.70 |
| `tangent` | 1 | 151.48 ms | 298.00 MiB | 11317808 | 1.00 | 1.00 |
| `tangent` | 2 | 90.38 ms | 285.28 MiB | 11317956 | 1.68 | 0.84 |
| `tangent` | 4 | 56.95 ms | 310.10 MiB | 11318171 | 2.66 | 0.66 |
| `tangent` | 8 | 38.88 ms | 308.50 MiB | 11318541 | 3.90 | 0.49 |
| `tangent` | 16 | 35.06 ms | 302.79 MiB | 11319242 | 4.32 | 0.27 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.193 s at 1 thread to 199.29 ms at 16 threads, a `5.99x` speedup with `0.37` efficiency. Allocation volume stays essentially flat at about `1472.78` to `1478.04 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 133.41 ms to 83.87 ms at 16 threads, or `1.59x` with `0.10` efficiency, while memory changes by about `1.09x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 62.94 ms to 5.61 ms at 16 threads, which is `11.22x` and `0.70` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 151.48 ms to 35.06 ms at 16 threads, or `4.32x` with `0.27` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.81x` to `1.13x`, and direct solve speedups remain `0.85x` to `0.90x`.
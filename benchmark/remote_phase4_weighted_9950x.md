# Remote Phase 4 Weighted 9950X

Generated: `2026-04-13T09:34:27`

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
- `GRICO_SCHEDULER_OVERRIDE`: `weighted_static`

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
| `adaptivity_plan` | 1 | 3.73 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 3.20 ms | 5.84 MiB | 121224 | 1.17 | 0.58 |
| `adaptivity_plan` | 4 | 2.75 ms | 5.86 MiB | 121301 | 1.35 | 0.34 |
| `adaptivity_plan` | 8 | 2.45 ms | 5.87 MiB | 121437 | 1.52 | 0.19 |
| `adaptivity_plan` | 16 | 2.54 ms | 5.89 MiB | 121681 | 1.47 | 0.09 |
| `assemble` | 1 | 1.412 s | 1443.85 MiB | 59825916 | 1.00 | 1.00 |
| `assemble` | 2 | 699.03 ms | 1444.01 MiB | 59823025 | 2.02 | 1.01 |
| `assemble` | 4 | 347.36 ms | 1444.05 MiB | 59817224 | 4.06 | 1.02 |
| `assemble` | 8 | 221.82 ms | 1445.01 MiB | 59805708 | 6.36 | 0.80 |
| `assemble` | 16 | 200.26 ms | 1445.60 MiB | 59786515 | 7.05 | 0.44 |
| `preconditioner_build` | 1 | 17.76 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 15.90 ms | 127.21 MiB | 16367 | 1.12 | 0.56 |
| `preconditioner_build` | 4 | 22.07 ms | 127.36 MiB | 16406 | 0.80 | 0.20 |
| `preconditioner_build` | 8 | 21.32 ms | 127.76 MiB | 16478 | 0.83 | 0.10 |
| `preconditioner_build` | 16 | 20.27 ms | 128.18 MiB | 16597 | 0.88 | 0.05 |
| `solve_direct` | 1 | 14.31 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 15.67 ms | 27.92 MiB | 136 | 0.91 | 0.46 |
| `solve_direct` | 4 | 14.35 ms | 27.92 MiB | 136 | 1.00 | 0.25 |
| `solve_direct` | 8 | 14.40 ms | 27.92 MiB | 136 | 0.99 | 0.12 |
| `solve_direct` | 16 | 14.20 ms | 27.92 MiB | 136 | 1.01 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 190.74 ms | 303.70 MiB | 9603517 | 1.00 | 1.00 |
| `assemble` | 2 | 117.74 ms | 304.07 MiB | 9603673 | 1.62 | 0.81 |
| `assemble` | 4 | 73.64 ms | 304.61 MiB | 9603920 | 2.59 | 0.65 |
| `assemble` | 8 | 52.58 ms | 305.60 MiB | 9604496 | 3.63 | 0.45 |
| `assemble` | 16 | 41.53 ms | 307.73 MiB | 9606039 | 4.59 | 0.29 |
| `preconditioner_build` | 1 | 20.97 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 25.76 ms | 263.24 MiB | 16400 | 0.81 | 0.41 |
| `preconditioner_build` | 4 | 31.61 ms | 263.44 MiB | 16435 | 0.66 | 0.17 |
| `preconditioner_build` | 8 | 19.79 ms | 264.04 MiB | 16509 | 1.06 | 0.13 |
| `preconditioner_build` | 16 | 20.79 ms | 264.86 MiB | 16624 | 1.01 | 0.06 |
| `solve_direct` | 1 | 25.45 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 21.45 ms | 70.26 MiB | 108 | 1.19 | 0.59 |
| `solve_direct` | 4 | 23.43 ms | 70.26 MiB | 108 | 1.09 | 0.27 |
| `solve_direct` | 8 | 25.50 ms | 70.26 MiB | 108 | 1.00 | 0.12 |
| `solve_direct` | 16 | 24.33 ms | 70.26 MiB | 108 | 1.05 | 0.07 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 59.81 ms | 23.41 MiB | 1270524 | 1.00 | 1.00 |
| `residual_bang` | 2 | 16.68 ms | 23.45 MiB | 1270724 | 3.59 | 1.79 |
| `residual_bang` | 4 | 13.02 ms | 23.49 MiB | 1270967 | 4.59 | 1.15 |
| `residual_bang` | 8 | 4.91 ms | 23.61 MiB | 1271539 | 12.17 | 1.52 |
| `residual_bang` | 16 | 7.93 ms | 23.85 MiB | 1273036 | 7.55 | 0.47 |
| `tangent` | 1 | 173.31 ms | 235.69 MiB | 11317786 | 1.00 | 1.00 |
| `tangent` | 2 | 83.29 ms | 235.87 MiB | 11318005 | 2.08 | 1.04 |
| `tangent` | 4 | 56.22 ms | 236.33 MiB | 11318310 | 3.08 | 0.77 |
| `tangent` | 8 | 34.22 ms | 237.22 MiB | 11318990 | 5.06 | 0.63 |
| `tangent` | 16 | 33.46 ms | 238.94 MiB | 11320703 | 5.18 | 0.32 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.412 s at 1 thread to 200.26 ms at 16 threads, a `7.05x` speedup with `0.44` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1445.60 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 190.74 ms to 41.53 ms at 16 threads, or `4.59x` with `0.29` efficiency, while memory changes by about `1.01x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 59.81 ms to 7.93 ms at 16 threads, which is `7.55x` and `0.47` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 173.31 ms to 33.46 ms at 16 threads, or `5.18x` with `0.32` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.88x` to `1.01x`, and direct solve speedups remain `1.01x` to `1.05x`.
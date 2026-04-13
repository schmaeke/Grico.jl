# Remote Phase 4 Static 9950X

Generated: `2026-04-13T09:31:48`

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
- `GRICO_SCHEDULER_OVERRIDE`: `static`

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
| `adaptivity_plan` | 2 | 2.89 ms | 5.85 MiB | 121226 | 1.29 | 0.64 |
| `adaptivity_plan` | 4 | 2.86 ms | 5.85 MiB | 121299 | 1.30 | 0.32 |
| `adaptivity_plan` | 8 | 2.47 ms | 5.88 MiB | 121439 | 1.50 | 0.19 |
| `adaptivity_plan` | 16 | 2.34 ms | 5.88 MiB | 121670 | 1.59 | 0.10 |
| `assemble` | 1 | 1.223 s | 1443.85 MiB | 59825916 | 1.00 | 1.00 |
| `assemble` | 2 | 661.07 ms | 1444.01 MiB | 59823025 | 1.85 | 0.92 |
| `assemble` | 4 | 350.78 ms | 1444.05 MiB | 59817224 | 3.49 | 0.87 |
| `assemble` | 8 | 222.02 ms | 1445.01 MiB | 59805708 | 5.51 | 0.69 |
| `assemble` | 16 | 195.62 ms | 1445.60 MiB | 59786515 | 6.25 | 0.39 |
| `preconditioner_build` | 1 | 18.26 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 13.14 ms | 127.22 MiB | 16367 | 1.39 | 0.69 |
| `preconditioner_build` | 4 | 21.11 ms | 127.36 MiB | 16406 | 0.86 | 0.22 |
| `preconditioner_build` | 8 | 20.41 ms | 127.76 MiB | 16478 | 0.89 | 0.11 |
| `preconditioner_build` | 16 | 21.82 ms | 128.08 MiB | 16593 | 0.84 | 0.05 |
| `solve_direct` | 1 | 13.66 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 15.20 ms | 27.92 MiB | 136 | 0.90 | 0.45 |
| `solve_direct` | 4 | 14.34 ms | 27.92 MiB | 136 | 0.95 | 0.24 |
| `solve_direct` | 8 | 14.27 ms | 27.92 MiB | 136 | 0.96 | 0.12 |
| `solve_direct` | 16 | 14.12 ms | 27.92 MiB | 136 | 0.97 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 187.49 ms | 303.70 MiB | 9603517 | 1.00 | 1.00 |
| `assemble` | 2 | 116.58 ms | 304.07 MiB | 9603673 | 1.61 | 0.80 |
| `assemble` | 4 | 74.86 ms | 304.61 MiB | 9603920 | 2.50 | 0.63 |
| `assemble` | 8 | 53.20 ms | 305.60 MiB | 9604496 | 3.52 | 0.44 |
| `assemble` | 16 | 42.95 ms | 307.73 MiB | 9606039 | 4.37 | 0.27 |
| `preconditioner_build` | 1 | 20.96 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 25.09 ms | 263.24 MiB | 16400 | 0.84 | 0.42 |
| `preconditioner_build` | 4 | 28.37 ms | 263.43 MiB | 16435 | 0.74 | 0.18 |
| `preconditioner_build` | 8 | 19.07 ms | 264.04 MiB | 16509 | 1.10 | 0.14 |
| `preconditioner_build` | 16 | 20.63 ms | 264.77 MiB | 16624 | 1.02 | 0.06 |
| `solve_direct` | 1 | 21.44 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 22.61 ms | 70.26 MiB | 108 | 0.95 | 0.47 |
| `solve_direct` | 4 | 22.48 ms | 70.26 MiB | 108 | 0.95 | 0.24 |
| `solve_direct` | 8 | 23.95 ms | 70.26 MiB | 108 | 0.90 | 0.11 |
| `solve_direct` | 16 | 26.35 ms | 70.26 MiB | 108 | 0.81 | 0.05 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 33.50 ms | 23.41 MiB | 1270524 | 1.00 | 1.00 |
| `residual_bang` | 2 | 16.90 ms | 23.45 MiB | 1270724 | 1.98 | 0.99 |
| `residual_bang` | 4 | 13.08 ms | 23.49 MiB | 1270967 | 2.56 | 0.64 |
| `residual_bang` | 8 | 4.77 ms | 23.61 MiB | 1271539 | 7.02 | 0.88 |
| `residual_bang` | 16 | 6.61 ms | 23.85 MiB | 1273036 | 5.07 | 0.32 |
| `tangent` | 1 | 142.19 ms | 235.69 MiB | 11317786 | 1.00 | 1.00 |
| `tangent` | 2 | 85.86 ms | 235.87 MiB | 11318005 | 1.66 | 0.83 |
| `tangent` | 4 | 53.48 ms | 236.33 MiB | 11318310 | 2.66 | 0.66 |
| `tangent` | 8 | 35.32 ms | 237.22 MiB | 11318990 | 4.03 | 0.50 |
| `tangent` | 16 | 34.02 ms | 238.94 MiB | 11320703 | 4.18 | 0.26 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.223 s at 1 thread to 195.62 ms at 16 threads, a `6.25x` speedup with `0.39` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1445.60 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 187.49 ms to 42.95 ms at 16 threads, or `4.37x` with `0.27` efficiency, while memory changes by about `1.01x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 33.50 ms to 6.61 ms at 16 threads, which is `5.07x` and `0.32` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 142.19 ms to 34.02 ms at 16 threads, or `4.18x` with `0.26` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.84x` to `1.02x`, and direct solve speedups remain `0.81x` to `0.97x`.
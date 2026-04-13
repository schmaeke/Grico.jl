# Remote Phase 4 Hybrid 9950X

Generated: `2026-04-13T09:36:56`

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
- `GRICO_SCHEDULER_OVERRIDE`: `hybrid`

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
| `adaptivity_plan` | 2 | 2.93 ms | 5.85 MiB | 121226 | 1.28 | 0.64 |
| `adaptivity_plan` | 4 | 2.81 ms | 5.87 MiB | 121303 | 1.34 | 0.33 |
| `adaptivity_plan` | 8 | 2.44 ms | 5.87 MiB | 121435 | 1.53 | 0.19 |
| `adaptivity_plan` | 16 | 2.50 ms | 5.88 MiB | 121684 | 1.50 | 0.09 |
| `assemble` | 1 | 1.456 s | 1443.85 MiB | 59825916 | 1.00 | 1.00 |
| `assemble` | 2 | 665.17 ms | 1444.83 MiB | 59823072 | 2.19 | 1.09 |
| `assemble` | 4 | 392.34 ms | 1446.10 MiB | 59817331 | 3.71 | 0.93 |
| `assemble` | 8 | 238.73 ms | 1448.95 MiB | 59805931 | 6.10 | 0.76 |
| `assemble` | 16 | 184.10 ms | 1451.52 MiB | 59787404 | 7.91 | 0.49 |
| `preconditioner_build` | 1 | 17.73 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 14.86 ms | 127.21 MiB | 16369 | 1.19 | 0.60 |
| `preconditioner_build` | 4 | 21.93 ms | 127.36 MiB | 16404 | 0.81 | 0.20 |
| `preconditioner_build` | 8 | 20.95 ms | 127.76 MiB | 16478 | 0.85 | 0.11 |
| `preconditioner_build` | 16 | 20.27 ms | 128.08 MiB | 16593 | 0.87 | 0.05 |
| `solve_direct` | 1 | 13.30 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 15.07 ms | 27.92 MiB | 136 | 0.88 | 0.44 |
| `solve_direct` | 4 | 14.82 ms | 27.92 MiB | 136 | 0.90 | 0.22 |
| `solve_direct` | 8 | 14.17 ms | 27.92 MiB | 136 | 0.94 | 0.12 |
| `solve_direct` | 16 | 14.54 ms | 27.92 MiB | 136 | 0.91 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 192.11 ms | 303.70 MiB | 9603517 | 1.00 | 1.00 |
| `assemble` | 2 | 130.07 ms | 310.42 MiB | 9603714 | 1.48 | 0.74 |
| `assemble` | 4 | 73.22 ms | 317.05 MiB | 9604017 | 2.62 | 0.66 |
| `assemble` | 8 | 47.17 ms | 329.15 MiB | 9604797 | 4.07 | 0.51 |
| `assemble` | 16 | 53.19 ms | 340.74 MiB | 9607148 | 3.61 | 0.23 |
| `preconditioner_build` | 1 | 20.84 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 25.00 ms | 263.24 MiB | 16400 | 0.83 | 0.42 |
| `preconditioner_build` | 4 | 22.59 ms | 263.44 MiB | 16435 | 0.92 | 0.23 |
| `preconditioner_build` | 8 | 19.16 ms | 264.04 MiB | 16509 | 1.09 | 0.14 |
| `preconditioner_build` | 16 | 20.71 ms | 264.76 MiB | 16624 | 1.01 | 0.06 |
| `solve_direct` | 1 | 25.64 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 22.69 ms | 70.26 MiB | 108 | 1.13 | 0.56 |
| `solve_direct` | 4 | 21.82 ms | 70.26 MiB | 108 | 1.17 | 0.29 |
| `solve_direct` | 8 | 25.51 ms | 70.26 MiB | 108 | 1.01 | 0.13 |
| `solve_direct` | 16 | 24.59 ms | 70.26 MiB | 108 | 1.04 | 0.07 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 30.86 ms | 23.41 MiB | 1270524 | 1.00 | 1.00 |
| `residual_bang` | 2 | 16.86 ms | 23.59 MiB | 1270761 | 1.83 | 0.92 |
| `residual_bang` | 4 | 15.62 ms | 23.84 MiB | 1271028 | 1.98 | 0.49 |
| `residual_bang` | 8 | 6.72 ms | 24.30 MiB | 1271668 | 4.59 | 0.57 |
| `residual_bang` | 16 | 4.09 ms | 24.78 MiB | 1273409 | 7.54 | 0.47 |
| `tangent` | 1 | 139.19 ms | 235.69 MiB | 11317786 | 1.00 | 1.00 |
| `tangent` | 2 | 84.32 ms | 239.01 MiB | 11318058 | 1.65 | 0.83 |
| `tangent` | 4 | 57.78 ms | 242.26 MiB | 11318395 | 2.41 | 0.60 |
| `tangent` | 8 | 36.65 ms | 248.72 MiB | 11319197 | 3.80 | 0.47 |
| `tangent` | 16 | 31.60 ms | 252.14 MiB | 11321494 | 4.40 | 0.28 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.456 s at 1 thread to 184.10 ms at 16 threads, a `7.91x` speedup with `0.49` efficiency. Allocation volume stays essentially flat at about `1443.85` to `1451.52 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 192.11 ms to 53.19 ms at 16 threads, or `3.61x` with `0.23` efficiency, while memory changes by about `1.12x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 30.86 ms to 4.09 ms at 16 threads, which is `7.54x` and `0.47` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 139.19 ms to 31.60 ms at 16 threads, or `4.40x` with `0.28` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.87x` to `1.01x`, and direct solve speedups remain `0.91x` to `1.04x`.
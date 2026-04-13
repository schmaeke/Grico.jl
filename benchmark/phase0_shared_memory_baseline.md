# Phase 4 Local Smoke

Generated: `2026-04-13T09:22:17`

Primary thread counts: `1, 6`

Primary scaling set capped at 6 threads on this host because the machine has 6 performance cores.

## Environment

- Julia: `1.12.5`
- OS / arch: `Darwin` / `aarch64`
- Host: `schmaekes-MBP`
- CPU: `Apple M2 Pro`
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
| `adaptivity_plan` | 1 | 5.58 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 6 | 3.91 ms | 5.96 MiB | 121365 | 1.42 | 0.24 |
| `assemble` | 1 | 2.494 s | 1444.33 MiB | 59825911 | 1.00 | 1.00 |
| `assemble` | 6 | 607.97 ms | 1445.48 MiB | 59811439 | 4.10 | 0.68 |
| `preconditioner_build` | 1 | 12.50 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 6 | 80.64 ms | 134.30 MiB | 16442 | 0.16 | 0.03 |
| `solve_direct` | 1 | 19.36 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 6 | 19.05 ms | 28.44 MiB | 136 | 1.02 | 0.17 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 208.58 ms | 304.02 MiB | 9603502 | 1.00 | 1.00 |
| `assemble` | 6 | 59.64 ms | 317.96 MiB | 9604173 | 3.50 | 0.58 |
| `preconditioner_build` | 1 | 21.95 ms | 291.32 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 6 | 95.89 ms | 289.27 MiB | 16473 | 0.23 | 0.04 |
| `solve_direct` | 1 | 29.21 ms | 81.28 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 6 | 62.63 ms | 96.99 MiB | 136 | 0.47 | 0.08 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 77.93 ms | 23.42 MiB | 1270489 | 1.00 | 1.00 |
| `residual_bang` | 6 | 14.57 ms | 23.58 MiB | 1271208 | 5.35 | 0.89 |
| `tangent` | 1 | 160.57 ms | 239.02 MiB | 11317751 | 1.00 | 1.00 |
| `tangent` | 6 | 51.58 ms | 238.81 MiB | 11318613 | 3.11 | 0.52 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 2.494 s at 1 thread to 607.97 ms at 6 threads, a `4.10x` speedup with `0.68` efficiency. Allocation volume stays essentially flat at about `1444.33` to `1445.48 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 208.58 ms to 59.64 ms at 6 threads, or `3.50x` with `0.58` efficiency, while memory changes by about `1.05x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 77.93 ms to 14.57 ms at 6 threads, which is `5.35x` and `0.89` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 160.57 ms to 51.58 ms at 6 threads, or `3.11x` with `0.52` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.16x` to `0.23x`, and direct solve speedups remain `0.47x` to `1.02x`.
# Remote Phase 5 Rerun 9950X

Generated: `2026-04-13T10:49:13`

Primary thread counts: `1, 2, 4, 8, 16`

Thread counts follow the requested benchmark set on this host.

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
| `adaptive_poisson` | 1279 | - | 140 | - | - | - |

## Results

### `affine_cell_diffusion`

Continuous scalar Poisson problem with volume-dominated affine assembly.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 3.74 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 2.96 ms | 5.85 MiB | 121226 | 1.26 | 0.63 |
| `adaptivity_plan` | 4 | 2.67 ms | 5.86 MiB | 121301 | 1.40 | 0.35 |
| `adaptivity_plan` | 8 | 2.39 ms | 5.88 MiB | 121438 | 1.57 | 0.20 |
| `adaptivity_plan` | 16 | 2.40 ms | 5.88 MiB | 121674 | 1.56 | 0.10 |
| `assemble` | 1 | 33.20 ms | 31.35 MiB | 433911 | 1.00 | 1.00 |
| `assemble` | 2 | 22.76 ms | 31.51 MiB | 431020 | 1.46 | 0.73 |
| `assemble` | 4 | 17.63 ms | 31.55 MiB | 425219 | 1.88 | 0.47 |
| `assemble` | 8 | 15.97 ms | 32.51 MiB | 413703 | 2.08 | 0.26 |
| `assemble` | 16 | 14.60 ms | 33.10 MiB | 394510 | 2.27 | 0.14 |
| `preconditioner_build` | 1 | 22.38 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 15.52 ms | 127.27 MiB | 16369 | 1.44 | 0.72 |
| `preconditioner_build` | 4 | 40.62 ms | 127.36 MiB | 16404 | 0.55 | 0.14 |
| `preconditioner_build` | 8 | 40.34 ms | 127.75 MiB | 16478 | 0.55 | 0.07 |
| `preconditioner_build` | 16 | 28.46 ms | 128.18 MiB | 16593 | 0.79 | 0.05 |
| `solve_direct` | 1 | 12.75 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 13.98 ms | 27.92 MiB | 136 | 0.91 | 0.46 |
| `solve_direct` | 4 | 15.32 ms | 27.92 MiB | 136 | 0.83 | 0.21 |
| `solve_direct` | 8 | 14.69 ms | 27.92 MiB | 136 | 0.87 | 0.11 |
| `solve_direct` | 16 | 15.13 ms | 27.92 MiB | 136 | 0.84 | 0.05 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 80.87 ms | 59.47 MiB | 142 | 1.00 | 1.00 |
| `assemble` | 2 | 45.40 ms | 59.84 MiB | 298 | 1.78 | 0.89 |
| `assemble` | 4 | 27.38 ms | 60.39 MiB | 545 | 2.95 | 0.74 |
| `assemble` | 8 | 17.26 ms | 61.38 MiB | 1121 | 4.69 | 0.59 |
| `assemble` | 16 | 12.76 ms | 63.50 MiB | 2664 | 6.34 | 0.40 |
| `preconditioner_build` | 1 | 25.65 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 25.71 ms | 263.24 MiB | 16400 | 1.00 | 0.50 |
| `preconditioner_build` | 4 | 51.93 ms | 263.44 MiB | 16435 | 0.49 | 0.12 |
| `preconditioner_build` | 8 | 62.08 ms | 264.04 MiB | 16509 | 0.41 | 0.05 |
| `preconditioner_build` | 16 | 36.50 ms | 264.86 MiB | 16624 | 0.70 | 0.04 |
| `solve_direct` | 1 | 25.22 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 25.09 ms | 70.26 MiB | 108 | 1.01 | 0.50 |
| `solve_direct` | 4 | 21.96 ms | 70.26 MiB | 108 | 1.15 | 0.29 |
| `solve_direct` | 8 | 24.47 ms | 70.26 MiB | 108 | 1.03 | 0.13 |
| `solve_direct` | 16 | 22.66 ms | 70.26 MiB | 108 | 1.11 | 0.07 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 926.17 μs | 0.15 MiB | 409 | 1.00 | 1.00 |
| `residual_bang` | 2 | 598.57 μs | 0.19 MiB | 609 | 1.55 | 0.77 |
| `residual_bang` | 4 | 373.33 μs | 0.23 MiB | 852 | 2.48 | 0.62 |
| `residual_bang` | 8 | 238.00 μs | 0.35 MiB | 1424 | 3.89 | 0.49 |
| `residual_bang` | 16 | 271.47 μs | 0.59 MiB | 2921 | 3.41 | 0.21 |
| `tangent` | 1 | 25.41 ms | 14.27 MiB | 431 | 1.00 | 1.00 |
| `tangent` | 2 | 14.05 ms | 14.45 MiB | 650 | 1.81 | 0.90 |
| `tangent` | 4 | 8.54 ms | 14.91 MiB | 955 | 2.98 | 0.74 |
| `tangent` | 8 | 5.17 ms | 15.80 MiB | 1635 | 4.92 | 0.61 |
| `tangent` | 16 | 3.89 ms | 17.53 MiB | 3348 | 6.53 | 0.41 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 492.63 μs | 0.96 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 527.80 μs | 0.97 MiB | 16728 | 0.93 | 0.47 |
| `adaptivity_plan` | 4 | 364.65 μs | 0.97 MiB | 16800 | 1.35 | 0.34 |
| `adaptivity_plan` | 8 | 371.17 μs | 0.98 MiB | 16929 | 1.33 | 0.17 |
| `adaptivity_plan` | 16 | 492.18 μs | 1.00 MiB | 17165 | 1.00 | 0.06 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 33.20 ms at 1 thread to 14.60 ms at 16 threads, a `2.27x` speedup with `0.14` efficiency. Allocation volume stays essentially flat at about `31.35` to `33.10 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 80.87 ms to 12.76 ms at 16 threads, or `6.34x` with `0.40` efficiency, while memory changes by about `1.07x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 926.17 μs to 271.47 μs at 16 threads, which is `3.41x` and `0.21` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 25.41 ms to 3.89 ms at 16 threads, or `6.53x` with `0.41` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.70x` to `0.79x`, and direct solve speedups remain `0.84x` to `1.11x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
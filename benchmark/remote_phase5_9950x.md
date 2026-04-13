# Remote Phase 5 9950X

Generated: `2026-04-13T10:46:25`

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
| `adaptivity_plan` | 1 | 3.78 ms | 5.84 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 3.16 ms | 5.84 MiB | 121224 | 1.20 | 0.60 |
| `adaptivity_plan` | 4 | 2.66 ms | 5.85 MiB | 121299 | 1.42 | 0.36 |
| `adaptivity_plan` | 8 | 2.41 ms | 5.86 MiB | 121435 | 1.57 | 0.20 |
| `adaptivity_plan` | 16 | 2.47 ms | 5.89 MiB | 121681 | 1.53 | 0.10 |
| `assemble` | 1 | 30.84 ms | 31.35 MiB | 433911 | 1.00 | 1.00 |
| `assemble` | 2 | 22.92 ms | 31.51 MiB | 431020 | 1.35 | 0.67 |
| `assemble` | 4 | 17.82 ms | 31.55 MiB | 425219 | 1.73 | 0.43 |
| `assemble` | 8 | 15.43 ms | 32.51 MiB | 413703 | 2.00 | 0.25 |
| `assemble` | 16 | 15.41 ms | 33.10 MiB | 394510 | 2.00 | 0.13 |
| `preconditioner_build` | 1 | 19.78 ms | 127.21 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 14.89 ms | 127.26 MiB | 16369 | 1.33 | 0.66 |
| `preconditioner_build` | 4 | 43.21 ms | 127.35 MiB | 16404 | 0.46 | 0.11 |
| `preconditioner_build` | 8 | 39.86 ms | 127.75 MiB | 16478 | 0.50 | 0.06 |
| `preconditioner_build` | 16 | 30.75 ms | 128.08 MiB | 16593 | 0.64 | 0.04 |
| `solve_direct` | 1 | 13.08 ms | 27.92 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 16.10 ms | 27.92 MiB | 136 | 0.81 | 0.41 |
| `solve_direct` | 4 | 14.86 ms | 27.92 MiB | 136 | 0.88 | 0.22 |
| `solve_direct` | 8 | 13.98 ms | 27.92 MiB | 136 | 0.94 | 0.12 |
| `solve_direct` | 16 | 14.37 ms | 27.92 MiB | 136 | 0.91 | 0.06 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 79.23 ms | 59.47 MiB | 142 | 1.00 | 1.00 |
| `assemble` | 2 | 45.10 ms | 59.84 MiB | 298 | 1.76 | 0.88 |
| `assemble` | 4 | 27.77 ms | 60.39 MiB | 545 | 2.85 | 0.71 |
| `assemble` | 8 | 19.44 ms | 61.38 MiB | 1121 | 4.08 | 0.51 |
| `assemble` | 16 | 13.56 ms | 63.50 MiB | 2664 | 5.84 | 0.37 |
| `preconditioner_build` | 1 | 25.49 ms | 263.13 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 26.52 ms | 263.19 MiB | 16400 | 0.96 | 0.48 |
| `preconditioner_build` | 4 | 22.17 ms | 263.44 MiB | 16435 | 1.15 | 0.29 |
| `preconditioner_build` | 8 | 49.02 ms | 264.05 MiB | 16509 | 0.52 | 0.07 |
| `preconditioner_build` | 16 | 24.95 ms | 264.76 MiB | 16624 | 1.02 | 0.06 |
| `solve_direct` | 1 | 25.22 ms | 70.26 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 25.61 ms | 70.26 MiB | 108 | 0.98 | 0.49 |
| `solve_direct` | 4 | 23.68 ms | 70.26 MiB | 108 | 1.06 | 0.27 |
| `solve_direct` | 8 | 23.69 ms | 70.26 MiB | 108 | 1.06 | 0.13 |
| `solve_direct` | 16 | 23.00 ms | 70.26 MiB | 108 | 1.10 | 0.07 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 916.13 μs | 0.15 MiB | 409 | 1.00 | 1.00 |
| `residual_bang` | 2 | 538.98 μs | 0.19 MiB | 609 | 1.70 | 0.85 |
| `residual_bang` | 4 | 295.33 μs | 0.23 MiB | 852 | 3.10 | 0.78 |
| `residual_bang` | 8 | 200.35 μs | 0.35 MiB | 1424 | 4.57 | 0.57 |
| `residual_bang` | 16 | 304.25 μs | 0.59 MiB | 2921 | 3.01 | 0.19 |
| `tangent` | 1 | 24.51 ms | 14.27 MiB | 431 | 1.00 | 1.00 |
| `tangent` | 2 | 14.25 ms | 14.45 MiB | 650 | 1.72 | 0.86 |
| `tangent` | 4 | 8.52 ms | 14.91 MiB | 955 | 2.88 | 0.72 |
| `tangent` | 8 | 5.02 ms | 15.80 MiB | 1635 | 4.88 | 0.61 |
| `tangent` | 16 | 3.92 ms | 17.53 MiB | 3348 | 6.25 | 0.39 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 491.10 μs | 0.96 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 435.79 μs | 0.97 MiB | 16728 | 1.13 | 0.56 |
| `adaptivity_plan` | 4 | 446.21 μs | 0.97 MiB | 16799 | 1.10 | 0.28 |
| `adaptivity_plan` | 8 | 427.59 μs | 0.98 MiB | 16924 | 1.15 | 0.14 |
| `adaptivity_plan` | 16 | 479.77 μs | 1.00 MiB | 17160 | 1.02 | 0.06 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 30.84 ms at 1 thread to 15.41 ms at 16 threads, a `2.00x` speedup with `0.13` efficiency. Allocation volume stays essentially flat at about `31.35` to `33.10 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 79.23 ms to 13.56 ms at 16 threads, or `5.84x` with `0.37` efficiency, while memory changes by about `1.07x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 916.13 μs to 304.25 μs at 16 threads, which is `3.01x` and `0.19` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 24.51 ms to 3.92 ms at 16 threads, or `6.25x` with `0.39` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 16 threads, preconditioner build speedups are only `0.64x` to `1.02x`, and direct solve speedups remain `0.91x` to `1.10x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
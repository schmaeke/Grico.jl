# Phase 3 Batched Local Traversal

Generated: `2026-04-12T22:44:49`

Primary thread counts: `1, 2, 4, 6`

Primary scaling set capped at 6 threads on this host because the machine has 6 performance cores.

## Environment

- Julia: `1.12.5`
- OS / arch: `Darwin` / `aarch64`
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
| `adaptive_poisson` | 1279 | - | 140 | - | - | - |

## Results

### `affine_cell_diffusion`

Continuous scalar Poisson problem with volume-dominated affine assembly.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 5.31 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.43 ms | 5.95 MiB | 121226 | 1.20 | 0.60 |
| `adaptivity_plan` | 4 | 4.33 ms | 5.95 MiB | 121284 | 1.23 | 0.31 |
| `adaptivity_plan` | 6 | 3.85 ms | 5.96 MiB | 121359 | 1.38 | 0.23 |
| `assemble` | 1 | 2.050 s | 1444.33 MiB | 59825908 | 1.00 | 1.00 |
| `assemble` | 2 | 937.61 ms | 1448.44 MiB | 59823089 | 2.19 | 1.09 |
| `assemble` | 4 | 533.03 ms | 1450.84 MiB | 59817616 | 3.85 | 0.96 |
| `assemble` | 6 | 381.10 ms | 1453.32 MiB | 59812342 | 5.38 | 0.90 |
| `preconditioner_build` | 1 | 11.46 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 78.59 ms | 133.01 MiB | 16369 | 0.15 | 0.07 |
| `preconditioner_build` | 4 | 77.33 ms | 134.02 MiB | 16404 | 0.15 | 0.04 |
| `preconditioner_build` | 6 | 83.42 ms | 134.28 MiB | 16442 | 0.14 | 0.02 |
| `solve_direct` | 1 | 18.80 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 18.97 ms | 28.19 MiB | 136 | 0.99 | 0.50 |
| `solve_direct` | 4 | 18.81 ms | 28.44 MiB | 136 | 1.00 | 0.25 |
| `solve_direct` | 6 | 19.71 ms | 28.44 MiB | 136 | 0.95 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 231.28 ms | 304.02 MiB | 9603495 | 1.00 | 1.00 |
| `assemble` | 2 | 136.14 ms | 338.13 MiB | 9603699 | 1.70 | 0.85 |
| `assemble` | 4 | 80.01 ms | 358.44 MiB | 9604242 | 2.89 | 0.72 |
| `assemble` | 6 | 79.28 ms | 361.26 MiB | 9604994 | 2.92 | 0.49 |
| `preconditioner_build` | 1 | 20.68 ms | 290.46 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 87.71 ms | 294.81 MiB | 16400 | 0.24 | 0.12 |
| `preconditioner_build` | 4 | 86.03 ms | 294.31 MiB | 16435 | 0.24 | 0.06 |
| `preconditioner_build` | 6 | 92.84 ms | 294.48 MiB | 16473 | 0.22 | 0.04 |
| `solve_direct` | 1 | 30.56 ms | 78.16 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 30.19 ms | 77.08 MiB | 108 | 1.01 | 0.51 |
| `solve_direct` | 4 | 29.84 ms | 94.60 MiB | 108 | 1.02 | 0.26 |
| `solve_direct` | 6 | 63.79 ms | 112.00 MiB | 136 | 0.48 | 0.08 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 77.33 ms | 23.42 MiB | 1270475 | 1.00 | 1.00 |
| `residual_bang` | 2 | 124.29 ms | 24.31 MiB | 1270707 | 0.62 | 0.31 |
| `residual_bang` | 4 | 22.29 ms | 24.56 MiB | 1271078 | 3.47 | 0.87 |
| `residual_bang` | 6 | 7.63 ms | 24.57 MiB | 1271526 | 10.13 | 1.69 |
| `tangent` | 1 | 215.00 ms | 235.74 MiB | 11317737 | 1.00 | 1.00 |
| `tangent` | 2 | 211.31 ms | 248.42 MiB | 11318004 | 1.02 | 0.51 |
| `tangent` | 4 | 76.15 ms | 256.86 MiB | 11318541 | 2.82 | 0.71 |
| `tangent` | 6 | 49.17 ms | 260.50 MiB | 11319235 | 4.37 | 0.73 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 723.69 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 734.48 μs | 0.98 MiB | 16729 | 0.99 | 0.49 |
| `adaptivity_plan` | 4 | 1.00 ms | 0.98 MiB | 16787 | 0.72 | 0.18 |
| `adaptivity_plan` | 6 | 843.65 μs | 0.98 MiB | 16854 | 0.86 | 0.14 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 2.050 s at 1 thread to 381.10 ms at 6 threads, a `5.38x` speedup with `0.90` efficiency. Allocation volume stays essentially flat at about `1444.33` to `1453.32 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 231.28 ms to 79.28 ms at 6 threads, or `2.92x` with `0.49` efficiency, while memory changes by about `1.19x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 77.33 ms to 7.63 ms at 6 threads, which is `10.13x` and `1.69` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 215.00 ms to 49.17 ms at 6 threads, or `4.37x` with `0.73` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.14x` to `0.22x`, and direct solve speedups remain `0.48x` to `0.95x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
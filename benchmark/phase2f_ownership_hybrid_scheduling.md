# Phase 2f Ownership Hybrid Scheduling

Generated: `2026-04-12T22:19:13`

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
| `adaptivity_plan` | 1 | 5.75 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.18 ms | 5.94 MiB | 121224 | 1.38 | 0.69 |
| `adaptivity_plan` | 4 | 4.14 ms | 5.95 MiB | 121293 | 1.39 | 0.35 |
| `adaptivity_plan` | 6 | 4.28 ms | 5.96 MiB | 121361 | 1.34 | 0.22 |
| `assemble` | 1 | 1.824 s | 1445.75 MiB | 59837107 | 1.00 | 1.00 |
| `assemble` | 2 | 981.67 ms | 1449.85 MiB | 59834288 | 1.86 | 0.93 |
| `assemble` | 4 | 558.60 ms | 1452.10 MiB | 59828803 | 3.26 | 0.82 |
| `assemble` | 6 | 411.48 ms | 1454.91 MiB | 59823541 | 4.43 | 0.74 |
| `preconditioner_build` | 1 | 12.99 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 79.53 ms | 133.01 MiB | 16369 | 0.16 | 0.08 |
| `preconditioner_build` | 4 | 77.05 ms | 134.02 MiB | 16404 | 0.17 | 0.04 |
| `preconditioner_build` | 6 | 83.40 ms | 134.29 MiB | 16442 | 0.16 | 0.03 |
| `solve_direct` | 1 | 19.26 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 19.23 ms | 28.19 MiB | 136 | 1.00 | 0.50 |
| `solve_direct` | 4 | 19.16 ms | 28.44 MiB | 136 | 1.01 | 0.25 |
| `solve_direct` | 6 | 20.32 ms | 28.44 MiB | 136 | 0.95 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 222.81 ms | 304.88 MiB | 9611492 | 1.00 | 1.00 |
| `assemble` | 2 | 132.52 ms | 320.95 MiB | 9611653 | 1.68 | 0.84 |
| `assemble` | 4 | 80.78 ms | 323.57 MiB | 9612069 | 2.76 | 0.69 |
| `assemble` | 6 | 69.87 ms | 324.06 MiB | 9612591 | 3.19 | 0.53 |
| `preconditioner_build` | 1 | 21.07 ms | 290.82 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 88.20 ms | 290.22 MiB | 16400 | 0.24 | 0.12 |
| `preconditioner_build` | 4 | 86.52 ms | 294.10 MiB | 16435 | 0.24 | 0.06 |
| `preconditioner_build` | 6 | 93.29 ms | 294.61 MiB | 16473 | 0.23 | 0.04 |
| `solve_direct` | 1 | 30.82 ms | 79.33 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 30.43 ms | 86.44 MiB | 108 | 1.01 | 0.51 |
| `solve_direct` | 4 | 30.38 ms | 94.60 MiB | 108 | 1.01 | 0.25 |
| `solve_direct` | 6 | 64.41 ms | 112.00 MiB | 136 | 0.48 | 0.08 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 40.74 ms | 23.42 MiB | 1270471 | 1.00 | 1.00 |
| `residual_bang` | 2 | 38.21 ms | 24.30 MiB | 1270627 | 1.07 | 0.53 |
| `residual_bang` | 4 | 10.99 ms | 24.55 MiB | 1270954 | 3.71 | 0.93 |
| `residual_bang` | 6 | 14.08 ms | 24.55 MiB | 1271362 | 2.89 | 0.48 |
| `tangent` | 1 | 191.05 ms | 235.74 MiB | 11317733 | 1.00 | 1.00 |
| `tangent` | 2 | 104.02 ms | 248.41 MiB | 11317924 | 1.84 | 0.92 |
| `tangent` | 4 | 63.30 ms | 256.85 MiB | 11318417 | 3.02 | 0.75 |
| `tangent` | 6 | 59.98 ms | 255.95 MiB | 11319059 | 3.19 | 0.53 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 753.71 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 702.98 μs | 0.98 MiB | 16730 | 1.07 | 0.54 |
| `adaptivity_plan` | 4 | 642.90 μs | 0.98 MiB | 16799 | 1.17 | 0.29 |
| `adaptivity_plan` | 6 | 999.65 μs | 0.98 MiB | 16858 | 0.75 | 0.13 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.824 s at 1 thread to 411.48 ms at 6 threads, a `4.43x` speedup with `0.74` efficiency. Allocation volume stays essentially flat at about `1445.75` to `1454.91 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 222.81 ms to 69.87 ms at 6 threads, or `3.19x` with `0.53` efficiency, while memory changes by about `1.06x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 40.74 ms to 14.08 ms at 6 threads, which is `2.89x` and `0.48` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 191.05 ms to 59.98 ms at 6 threads, or `3.19x` with `0.53` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.16x` to `0.23x`, and direct solve speedups remain `0.48x` to `0.95x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
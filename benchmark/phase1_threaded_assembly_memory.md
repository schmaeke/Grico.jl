# Phase 1 Threaded Assembly Memory

Generated: `2026-04-12T18:46:00`

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
| `adaptivity_plan` | 1 | 5.35 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.30 ms | 5.94 MiB | 121224 | 1.24 | 0.62 |
| `adaptivity_plan` | 4 | 3.77 ms | 5.94 MiB | 121289 | 1.42 | 0.35 |
| `adaptivity_plan` | 6 | 3.92 ms | 5.96 MiB | 121368 | 1.36 | 0.23 |
| `assemble` | 1 | 1.803 s | 1473.70 MiB | 59940949 | 1.00 | 1.00 |
| `assemble` | 2 | 964.61 ms | 1470.38 MiB | 59939074 | 1.87 | 0.93 |
| `assemble` | 4 | 513.45 ms | 1478.30 MiB | 59935281 | 3.51 | 0.88 |
| `assemble` | 6 | 443.06 ms | 1481.38 MiB | 59931457 | 4.07 | 0.68 |
| `preconditioner_build` | 1 | 14.17 ms | 132.10 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 79.63 ms | 132.23 MiB | 16369 | 0.18 | 0.09 |
| `preconditioner_build` | 4 | 79.72 ms | 132.16 MiB | 16404 | 0.18 | 0.04 |
| `preconditioner_build` | 6 | 87.89 ms | 132.60 MiB | 16442 | 0.16 | 0.03 |
| `solve_direct` | 1 | 22.17 ms | 29.06 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 22.90 ms | 29.06 MiB | 136 | 0.97 | 0.48 |
| `solve_direct` | 4 | 22.53 ms | 29.06 MiB | 136 | 0.98 | 0.25 |
| `solve_direct` | 6 | 22.76 ms | 29.06 MiB | 136 | 0.97 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 160.19 ms | 451.36 MiB | 10008647 | 1.00 | 1.00 |
| `assemble` | 2 | 164.70 ms | 494.92 MiB | 10008795 | 0.97 | 0.49 |
| `assemble` | 4 | 111.08 ms | 493.70 MiB | 10009001 | 1.44 | 0.36 |
| `assemble` | 6 | 96.11 ms | 529.11 MiB | 10009211 | 1.67 | 0.28 |
| `preconditioner_build` | 1 | 20.82 ms | 298.02 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 90.53 ms | 292.43 MiB | 16400 | 0.23 | 0.11 |
| `preconditioner_build` | 4 | 89.23 ms | 298.57 MiB | 16435 | 0.23 | 0.06 |
| `preconditioner_build` | 6 | 97.57 ms | 299.30 MiB | 16473 | 0.21 | 0.04 |
| `solve_direct` | 1 | 30.25 ms | 104.66 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 29.49 ms | 100.24 MiB | 108 | 1.03 | 0.51 |
| `solve_direct` | 4 | 29.80 ms | 100.63 MiB | 108 | 1.02 | 0.25 |
| `solve_direct` | 6 | 30.80 ms | 100.72 MiB | 108 | 0.98 | 0.16 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 60.75 ms | 23.51 MiB | 1270480 | 1.00 | 1.00 |
| `residual_bang` | 2 | 32.30 ms | 23.49 MiB | 1270577 | 1.88 | 0.94 |
| `residual_bang` | 4 | 16.84 ms | 23.48 MiB | 1270682 | 3.61 | 0.90 |
| `residual_bang` | 6 | 12.29 ms | 23.58 MiB | 1270792 | 4.94 | 0.82 |
| `tangent` | 1 | 182.84 ms | 300.82 MiB | 11317806 | 1.00 | 1.00 |
| `tangent` | 2 | 110.85 ms | 287.51 MiB | 11317953 | 1.65 | 0.82 |
| `tangent` | 4 | 70.03 ms | 317.72 MiB | 11318166 | 2.61 | 0.65 |
| `tangent` | 6 | 56.22 ms | 314.34 MiB | 11318346 | 3.25 | 0.54 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 726.04 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 718.08 μs | 0.97 MiB | 16728 | 1.01 | 0.51 |
| `adaptivity_plan` | 4 | 686.38 μs | 0.98 MiB | 16796 | 1.06 | 0.26 |
| `adaptivity_plan` | 6 | 886.00 μs | 0.98 MiB | 16859 | 0.82 | 0.14 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 1.803 s at 1 thread to 443.06 ms at 6 threads, a `4.07x` speedup with `0.68` efficiency. Allocation volume stays essentially flat at about `1473.70` to `1481.38 MiB` per call.
- Interface-heavy affine assembly is the weak point: `assemble(plan)` on `affine_interface_dg` only moves from 160.19 ms to 96.11 ms at 6 threads, or `1.67x` with `0.28` efficiency, while memory rises by about `1.17x`. That points directly at the current COO accumulation and interface scatter path as the next design target.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 60.75 ms to 12.29 ms at 6 threads, which is `4.94x` and `0.82` efficiency.
- Nonlinear tangent assembly is noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 182.84 ms to 56.22 ms at 6 threads, or `3.25x` with `0.54` efficiency. That keeps tangent assembly in scope for the later symbolic/numeric assembly phase.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.16x` to `0.21x`, and direct solve speedups remain `0.97x` to `0.98x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
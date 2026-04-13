# Phase 2d Ownership Mailbox Merge

Generated: `2026-04-12T22:11:34`

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
| `adaptivity_plan` | 1 | 5.65 ms | 5.94 MiB | 121115 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 4.19 ms | 5.95 MiB | 121226 | 1.35 | 0.67 |
| `adaptivity_plan` | 4 | 4.04 ms | 5.96 MiB | 121289 | 1.40 | 0.35 |
| `adaptivity_plan` | 6 | 3.88 ms | 5.95 MiB | 121352 | 1.46 | 0.24 |
| `assemble` | 1 | 2.951 s | 1445.75 MiB | 59837107 | 1.00 | 1.00 |
| `assemble` | 2 | 1.019 s | 1449.85 MiB | 59834288 | 2.90 | 1.45 |
| `assemble` | 4 | 635.46 ms | 1452.64 MiB | 59828815 | 4.64 | 1.16 |
| `assemble` | 6 | 395.70 ms | 1454.85 MiB | 59823517 | 7.46 | 1.24 |
| `preconditioner_build` | 1 | 11.64 ms | 133.87 MiB | 16335 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 78.50 ms | 133.01 MiB | 16369 | 0.15 | 0.07 |
| `preconditioner_build` | 4 | 76.81 ms | 134.02 MiB | 16404 | 0.15 | 0.04 |
| `preconditioner_build` | 6 | 83.85 ms | 134.28 MiB | 16442 | 0.14 | 0.02 |
| `solve_direct` | 1 | 19.17 ms | 28.44 MiB | 136 | 1.00 | 1.00 |
| `solve_direct` | 2 | 21.06 ms | 28.19 MiB | 136 | 0.91 | 0.46 |
| `solve_direct` | 4 | 18.84 ms | 28.44 MiB | 136 | 1.02 | 0.25 |
| `solve_direct` | 6 | 19.61 ms | 28.44 MiB | 136 | 0.98 | 0.16 |

### `affine_interface_dg`

Discontinuous scalar mass problem with explicit interior interface work.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `assemble` | 1 | 213.19 ms | 318.61 MiB | 9611493 | 1.00 | 1.00 |
| `assemble` | 2 | 128.35 ms | 338.98 MiB | 9611678 | 1.66 | 0.83 |
| `assemble` | 4 | 92.99 ms | 359.08 MiB | 9612210 | 2.29 | 0.57 |
| `assemble` | 6 | 76.75 ms | 359.38 MiB | 9612940 | 2.78 | 0.46 |
| `preconditioner_build` | 1 | 21.55 ms | 288.86 MiB | 16366 | 1.00 | 1.00 |
| `preconditioner_build` | 2 | 87.23 ms | 294.81 MiB | 16400 | 0.25 | 0.12 |
| `preconditioner_build` | 4 | 86.47 ms | 289.64 MiB | 16435 | 0.25 | 0.06 |
| `preconditioner_build` | 6 | 92.81 ms | 294.39 MiB | 16473 | 0.23 | 0.04 |
| `solve_direct` | 1 | 30.35 ms | 78.55 MiB | 108 | 1.00 | 1.00 |
| `solve_direct` | 2 | 30.16 ms | 77.83 MiB | 108 | 1.01 | 0.50 |
| `solve_direct` | 4 | 29.52 ms | 86.25 MiB | 108 | 1.03 | 0.26 |
| `solve_direct` | 6 | 30.66 ms | 91.88 MiB | 108 | 0.99 | 0.16 |

### `nonlinear_interface_dg`

Nonlinear discontinuous problem with cell, boundary, and interface terms.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_bang` | 1 | 49.83 ms | 23.42 MiB | 1270471 | 1.00 | 1.00 |
| `residual_bang` | 2 | 25.77 ms | 24.30 MiB | 1270627 | 1.93 | 0.97 |
| `residual_bang` | 4 | 13.39 ms | 24.55 MiB | 1270954 | 3.72 | 0.93 |
| `residual_bang` | 6 | 9.31 ms | 24.58 MiB | 1271366 | 5.35 | 0.89 |
| `tangent` | 1 | 193.12 ms | 235.74 MiB | 11317733 | 1.00 | 1.00 |
| `tangent` | 2 | 108.80 ms | 248.41 MiB | 11317924 | 1.77 | 0.89 |
| `tangent` | 4 | 62.83 ms | 260.16 MiB | 11318417 | 3.07 | 0.77 |
| `tangent` | 6 | 50.45 ms | 256.39 MiB | 11319059 | 3.83 | 0.64 |

### `adaptive_poisson`

Adaptivity-planning benchmark on a deterministic manually refined mesh.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 1 | 777.33 μs | 0.97 MiB | 16622 | 1.00 | 1.00 |
| `adaptivity_plan` | 2 | 732.50 μs | 0.98 MiB | 16730 | 1.06 | 0.53 |
| `adaptivity_plan` | 4 | 1.13 ms | 0.98 MiB | 16787 | 0.69 | 0.17 |
| `adaptivity_plan` | 6 | 870.15 μs | 0.98 MiB | 16857 | 0.89 | 0.15 |

## Initial Observations

- Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from 2.951 s at 1 thread to 395.70 ms at 6 threads, a `7.46x` speedup with `1.24` efficiency. Allocation volume stays essentially flat at about `1445.75` to `1454.85 MiB` per call.
- Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from 213.19 ms to 76.75 ms at 6 threads, or `2.78x` with `0.46` efficiency, while memory changes by about `1.13x`.
- Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from 49.83 ms to 9.31 ms at 6 threads, which is `5.35x` and `0.89` efficiency.
- Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from 193.12 ms to 50.45 ms at 6 threads, or `3.83x` with `0.64` efficiency.
- The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At 6 threads, preconditioner build speedups are only `0.14x` to `0.23x`, and direct solve speedups remain `0.98x` to `0.99x`.
- Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.
# Phase 3 Retained Cell Check

Generated: `2026-04-13T09:05:45`

Primary thread counts: `4, 6`

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

## Results

### `affine_cell_diffusion`

Continuous scalar Poisson problem with volume-dominated affine assembly.

| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptivity_plan` | 4 | 3.99 ms | 5.94 MiB | 121289 | - | - |
| `adaptivity_plan` | 6 | 3.84 ms | 5.96 MiB | 121360 | - | - |
| `assemble` | 4 | 754.76 ms | 1450.68 MiB | 59817604 | - | - |
| `assemble` | 6 | 387.97 ms | 1453.36 MiB | 59812338 | - | - |
| `preconditioner_build` | 4 | 76.45 ms | 134.02 MiB | 16404 | - | - |
| `preconditioner_build` | 6 | 85.24 ms | 134.30 MiB | 16442 | - | - |
| `solve_direct` | 4 | 19.02 ms | 28.44 MiB | 136 | - | - |
| `solve_direct` | 6 | 19.19 ms | 28.44 MiB | 136 | - | - |

## Initial Observations

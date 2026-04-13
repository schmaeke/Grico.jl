# Phase 7 Solver Path Cleanup

Generated: `2026-04-13T11:52:10`

## Environment

- Julia: `1.12.5`
- Host: `schmaekes-MBP`
- CPU: `Apple M2 Pro`
- Julia threads: `1`
- BLAS threads: `10`
- `OPENBLAS_NUM_THREADS`: ``
- `OMP_NUM_THREADS`: ``

## `affine_cell_diffusion`

Continuous scalar Poisson problem with symmetric volume-dominated assembly.

- Reduced dofs: `7761`
- Matrix nnz: `170457`
- Symmetric: `true`
- Median assembly time: `155.23 ms`
- Default policy: `Smoothed Aggregation AMG + CG`
- Default cold solve: `127.30 ms`
- Default warm solve: `9.30 ms`

| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `Sparse direct` | 9.76 ms | 335.04 μs | 310.79 μs | 10.09 ms | 165.32 ms | 1 | yes | 1.43e-14 |
| `Smoothed Aggregation AMG + CG` | 3.54 ms | 9.13 ms | 9.39 ms | 12.68 ms | 167.91 ms | 10 | yes | 1.03e-08 |
| `Additive Schwarz + CG` | 19.09 ms | 23.10 ms | 26.37 ms | 42.19 ms | 197.42 ms | 41 | yes | 1.30e-08 |

## `affine_interface_dg`

Discontinuous symmetric scalar problem with explicit interior interface work.

- Reduced dofs: `14400`
- Matrix nnz: `635040`
- Symmetric: `true`
- Median assembly time: `97.39 ms`
- Default policy: `Smoothed Aggregation AMG + CG`
- Default cold solve: `13.80 ms`
- Default warm solve: `2.70 ms`

| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `Sparse direct` | 32.07 ms | 1.04 ms | 871.92 μs | 33.10 ms | 130.49 ms | 1 | yes | 0.00e+00 |
| `Smoothed Aggregation AMG + CG` | 12.13 ms | 2.49 ms | 2.33 ms | 14.62 ms | 112.01 ms | 0 | yes | 0.00e+00 |
| `Additive Schwarz + CG` | 156.18 ms | 890.54 μs | 768.29 μs | 157.07 ms | 254.46 ms | 0 | yes | 0.00e+00 |

## `lid_driven_cavity_step1`

Mixed velocity-pressure reduced system from the DG lid-driven cavity example.

- Reduced dofs: `5120`
- Matrix nnz: `487364`
- Symmetric: `false`
- Median assembly time: `0.00 ns`
- Default policy: `ILU + GMRES`
- Default cold solve: `1.978 s`
- Default warm solve: `1.963 s`

| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `Sparse direct` | 78.13 ms | 4.68 ms | 4.54 ms | 82.81 ms | 82.81 ms | 1 | yes | 1.43e-14 |
| `ILU + GMRES` | 8.64 ms | 1.889 s | 1.892 s | 1.898 s | 1.898 s | 5120 | no | NaN |
| `Additive Schwarz + GMRES` | 12.96 ms | 1.364 s | 1.369 s | 1.377 s | 1.377 s | 2538 | yes | 2.34e-08 |
| `Field-Split Schur + GMRES` | 37.33 ms | 1.135 s | 915.93 ms | 1.172 s | 1.172 s | 1957 | yes | 2.35e-08 |
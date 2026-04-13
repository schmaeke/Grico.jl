# Remote Phase 7 Baseline 9950X

Generated: `2026-04-13T11:53:29`

## Environment

- Julia: `1.12.5`
- Host: `helios`
- CPU: `AMD Ryzen 9 9950X 16-Core Processor`
- Julia threads: `16`
- BLAS threads: `1`
- `OPENBLAS_NUM_THREADS`: ``
- `OMP_NUM_THREADS`: ``

## `affine_cell_diffusion`

Continuous scalar Poisson problem with symmetric volume-dominated assembly.

- Reduced dofs: `7761`
- Matrix nnz: `170457`
- Symmetric: `true`
- Median assembly time: `20.15 ms`
- Default policy: `Smoothed Aggregation AMG + CG`
- Default cold solve: `8.06 ms`
- Default warm solve: `5.30 ms`

| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `Sparse direct` | 11.44 ms | 299.24 μs | 232.13 μs | 11.73 ms | 31.89 ms | 1 | yes | 9.74e-15 |
| `Smoothed Aggregation AMG + CG` | 2.77 ms | 5.19 ms | 5.17 ms | 7.96 ms | 28.11 ms | 10 | yes | 1.03e-08 |
| `Additive Schwarz + CG` | 17.75 ms | 53.01 ms | 49.42 ms | 70.76 ms | 90.91 ms | 41 | yes | 1.30e-08 |

## `affine_interface_dg`

Discontinuous symmetric scalar problem with explicit interior interface work.

- Reduced dofs: `14400`
- Matrix nnz: `635040`
- Symmetric: `true`
- Median assembly time: `15.65 ms`
- Default policy: `Smoothed Aggregation AMG + CG`
- Default cold solve: `13.05 ms`
- Default warm solve: `1.54 ms`

| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `Sparse direct` | 32.67 ms | 737.72 μs | 575.54 μs | 33.41 ms | 49.06 ms | 1 | yes | 0.00e+00 |
| `Smoothed Aggregation AMG + CG` | 13.23 ms | 1.37 ms | 1.35 ms | 14.60 ms | 30.25 ms | 0 | yes | 0.00e+00 |
| `Additive Schwarz + CG` | 55.90 ms | 1.65 ms | 1.72 ms | 57.55 ms | 73.20 ms | 0 | yes | 0.00e+00 |

## `lid_driven_cavity_step1`

Mixed velocity-pressure reduced system from the DG lid-driven cavity example.

- Reduced dofs: `5120`
- Matrix nnz: `487364`
- Symmetric: `false`
- Median assembly time: `0.00 ns`
- Default policy: `ILU + GMRES`
- Default cold solve: `1.097 s`
- Default warm solve: `1.083 s`

| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `Sparse direct` | 68.18 ms | 2.63 ms | 2.33 ms | 70.81 ms | 70.81 ms | 1 | yes | 9.93e-15 |
| `ILU + GMRES` | 5.62 ms | 1.011 s | 1.008 s | 1.017 s | 1.017 s | 5120 | no | NaN |
| `Additive Schwarz + GMRES` | 18.62 ms | 989.32 ms | 994.71 ms | 1.008 s | 1.008 s | 2143 | yes | 2.18e-08 |
| `Field-Split Schur + GMRES` | 21.58 ms | 946.85 ms | 943.35 ms | 968.43 ms | 968.43 ms | 1929 | yes | 2.33e-08 |
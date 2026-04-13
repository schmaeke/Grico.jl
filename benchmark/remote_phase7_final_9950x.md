# Remote Phase 7 Final 9950X

Generated: `2026-04-13T11:57:02`

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
- Median assembly time: `19.50 ms`
- Default policy: `Smoothed Aggregation AMG + CG`
- Default cold solve: `8.09 ms`
- Default warm solve: `5.30 ms`

| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `Sparse direct` | 10.15 ms | 345.85 μs | 237.06 μs | 10.50 ms | 30.00 ms | 1 | yes | 9.74e-15 |
| `Smoothed Aggregation AMG + CG` | 2.79 ms | 5.23 ms | 5.23 ms | 8.01 ms | 27.51 ms | 10 | yes | 1.03e-08 |
| `Additive Schwarz + CG` | 16.73 ms | 56.55 ms | 49.67 ms | 73.28 ms | 92.78 ms | 41 | yes | 1.30e-08 |

## `affine_interface_dg`

Discontinuous symmetric scalar problem with explicit interior interface work.

- Reduced dofs: `14400`
- Matrix nnz: `635040`
- Symmetric: `true`
- Median assembly time: `14.75 ms`
- Default policy: `Smoothed Aggregation AMG + CG`
- Default cold solve: `12.89 ms`
- Default warm solve: `1.55 ms`

| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `Sparse direct` | 45.34 ms | 780.15 μs | 581.18 μs | 46.12 ms | 60.87 ms | 1 | yes | 0.00e+00 |
| `Smoothed Aggregation AMG + CG` | 10.19 ms | 1.37 ms | 1.33 ms | 11.56 ms | 26.31 ms | 0 | yes | 0.00e+00 |
| `Additive Schwarz + CG` | 55.57 ms | 1.72 ms | 1.78 ms | 57.29 ms | 72.04 ms | 0 | yes | 0.00e+00 |

## `lid_driven_cavity_step1`

Mixed velocity-pressure reduced system from the DG lid-driven cavity example.

- Reduced dofs: `5120`
- Matrix nnz: `487364`
- Symmetric: `false`
- Median assembly time: `0.00 ns`
- Default policy: `Sparse direct`
- Default cold solve: `88.18 ms`
- Default warm solve: `2.39 ms`

| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `Sparse direct` | 68.08 ms | 2.92 ms | 2.41 ms | 71.00 ms | 71.00 ms | 1 | yes | 1.07e-14 |
| `ILU + GMRES` | 5.47 ms | 970.60 ms | 969.73 ms | 976.07 ms | 976.07 ms | 5120 | no | NaN |
| `Additive Schwarz + GMRES` | 13.62 ms | 1.002 s | 972.17 ms | 1.015 s | 1.015 s | 2153 | yes | 2.35e-08 |
| `Field-Split Schur + GMRES` | 16.42 ms | 1.126 s | 1.173 s | 1.143 s | 1.143 s | 2027 | yes | 2.34e-08 |
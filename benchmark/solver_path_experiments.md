# Solver Path Experiments

Run date: 2026-04-09

Command:

```bash
julia --project=benchmark benchmark/solver_path_experiments.jl
```

The benchmark script now includes:

- Pure-Julia baselines already tested earlier: sparse direct, AdditiveSchwarz, AlgebraicMultigrid.jl, IncompleteLU.jl, and the existing `FieldSplitSchur` path.
- External-package paths requested here: `HYPRE.jl` and `MUMPS.jl`.
- One-thread defaults for reproducibility in this mixed MPI/OpenMP setup:
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `BLAS.set_num_threads(1)`
- `KMP_DUPLICATE_LIB_OK=TRUE` inside the benchmark script, to avoid the duplicate OpenMP runtime abort triggered by loading both `HYPRE` and `MUMPS` on this macOS setup.

Notes:

- `fill` means:
  - sparse direct: matrix nnz
  - ILU: factor nnz
  - MUMPS: `INFOG(29)` factor entries
- `opcmp` is only reported for `AlgebraicMultigrid.jl` hierarchies in this harness.
- All residuals below are true residuals measured by `Grico._relative_residual_norm`.

## Main Takeaways

- Best SPD default with external dependencies: `HYPRE BoomerAMG + PCG`.
- Best pure-Julia SPD default: `SmoothedAgg AMG + CG`.
- Best generic nonsymmetric single-field path in these tests: `HYPRE GMRES + ILU`.
- Best pure-Julia generic nonsymmetric single-field path: `ILU + GMRES`.
- For structured mixed KH flow systems, the current `FieldSplitSchur + GMRES` path remains strong, but monolithic `HYPRE` preconditioners are now credible fallbacks:
  - `HYPRE GMRES + ILU` is competitive on all sampled steps.
  - `HYPRE FlexGMRES + AMG` is dramatically better than monolithic pure-Julia AMG on the same mixed system.
- `MUMPS` is a good direct-solve reference, but not a good default path for iterative solves because setup dominates.

## Recommendation

If external dependencies are allowed:

- SPD default: `PCG + HYPRE BoomerAMG`
- General default with known mixed/block structure: keep `GMRES + FieldSplitSchur`
- General fallback with no structure: `GMRES + HYPRE ILU`
- Optional direct reference/debug path: `MUMPS`

If pure Julia only:

- SPD default: `CG + SmoothedAgg AMG`
- General default with known mixed/block structure: `GMRES + FieldSplitSchur`
- General fallback with no structure: `GMRES + ILU`

## SPD Scalar Diffusion

Problem:

- Poisson-type scalar diffusion
- roots `(48, 48)`
- degree `2`
- dofs `6721`

| solver | dofs | setup s | solve s | total s | iter | residual | converged | fill | opcmp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Sparse direct | 6721 | 0.000 | 0.022 | 0.022 | 1 | 2.61e-14 | yes | 84929 | - |
| MUMPS direct | 6721 | 0.126 | 0.003 | 0.129 | 1 | 2.58e-14 | yes | 2863143 | - |
| AdditiveSchwarz + CG | 6721 | 0.022 | 0.016 | 0.038 | 27 | 1.18e-08 | yes | - | - |
| RugeStuben AMG + CG | 6721 | 0.002 | 0.010 | 0.012 | 19 | 1.28e-08 | yes | - | 1.09 |
| SmoothedAgg AMG + CG | 6721 | 0.002 | 0.006 | 0.008 | 10 | 1.44e-08 | yes | - | 1.05 |
| HYPRE BoomerAMG + PCG | 6721 | 0.002 | 0.003 | 0.005 | 7 | 4.08e-10 | yes | - | - |

## General Scalar Transport

Sampled system:

- Kelvin-Helmholtz transport equation
- sampled steps `(1, 2, 3)`

### Step 1

| solver | dofs | setup s | solve s | total s | iter | residual | converged | fill | opcmp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Sparse direct | 3216 | 0.000 | 0.007 | 0.007 | 1 | 2.39e-16 | yes | 49824 | - |
| MUMPS direct | 3216 | 0.012 | 0.001 | 0.013 | 1 | 1.86e-16 | yes | 291578 | - |
| AdditiveSchwarz + GMRES | 3216 | 0.010 | 0.006 | 0.016 | 25 | 9.34e-09 | yes | - | - |
| SmoothedAgg AMG + GMRES | 3216 | 0.001 | 0.005 | 0.007 | 16 | 9.10e-09 | yes | - | 1.04 |
| ILU(τ=1e-03) + GMRES | 3216 | 0.002 | 0.001 | 0.003 | 6 | 2.39e-09 | yes | 72259 | - |
| ILU(τ=1e-04) + GMRES | 3216 | 0.008 | 0.001 | 0.009 | 3 | 4.94e-09 | yes | 95523 | - |
| HYPRE GMRES + ILU | 3216 | 0.001 | 0.001 | 0.002 | 10 | 4.65e-09 | yes | - | - |
| HYPRE FlexGMRES + AMG | 3216 | 0.001 | 0.005 | 0.006 | 13 | 8.29e-09 | yes | - | - |

### Step 2

| solver | dofs | setup s | solve s | total s | iter | residual | converged | fill | opcmp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Sparse direct | 3308 | 0.000 | 0.007 | 0.007 | 1 | 2.35e-16 | yes | 51404 | - |
| MUMPS direct | 3308 | 0.012 | 0.001 | 0.013 | 1 | 1.81e-16 | yes | 301224 | - |
| AdditiveSchwarz + GMRES | 3308 | 0.010 | 0.006 | 0.016 | 25 | 1.30e-08 | yes | - | - |
| SmoothedAgg AMG + GMRES | 3308 | 0.001 | 0.005 | 0.007 | 16 | 9.85e-09 | yes | - | 1.03 |
| ILU(τ=1e-03) + GMRES | 3308 | 0.003 | 0.001 | 0.003 | 6 | 5.15e-09 | yes | 72335 | - |
| ILU(τ=1e-04) + GMRES | 3308 | 0.004 | 0.000 | 0.004 | 3 | 3.74e-09 | yes | 96113 | - |
| HYPRE GMRES + ILU | 3308 | 0.001 | 0.001 | 0.003 | 11 | 7.50e-09 | yes | - | - |
| HYPRE FlexGMRES + AMG | 3308 | 0.002 | 0.005 | 0.006 | 13 | 9.70e-09 | yes | - | - |

### Step 3

| solver | dofs | setup s | solve s | total s | iter | residual | converged | fill | opcmp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Sparse direct | 3330 | 0.000 | 0.007 | 0.007 | 1 | 2.16e-16 | yes | 51774 | - |
| MUMPS direct | 3330 | 0.014 | 0.001 | 0.015 | 1 | 1.81e-16 | yes | 302264 | - |
| AdditiveSchwarz + GMRES | 3330 | 0.007 | 0.007 | 0.014 | 26 | 1.32e-08 | yes | - | - |
| SmoothedAgg AMG + GMRES | 3330 | 0.001 | 0.005 | 0.007 | 16 | 1.01e-08 | yes | - | 1.04 |
| ILU(τ=1e-03) + GMRES | 3330 | 0.003 | 0.001 | 0.004 | 7 | 5.39e-09 | yes | 72623 | - |
| ILU(τ=1e-04) + GMRES | 3330 | 0.004 | 0.000 | 0.004 | 3 | 3.17e-09 | yes | 96514 | - |
| HYPRE GMRES + ILU | 3330 | 0.001 | 0.001 | 0.003 | 11 | 7.70e-09 | yes | - | - |
| HYPRE FlexGMRES + AMG | 3330 | 0.002 | 0.005 | 0.006 | 13 | 9.82e-09 | yes | - | - |

## Mixed Kelvin-Helmholtz Flow

Sampled system:

- mixed velocity-pressure flow solve
- sampled steps `(1, 2, 3)`

### Step 1

| solver | dofs | setup s | solve s | total s | iter | residual | converged | fill | opcmp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Sparse direct | 9456 | 0.000 | 0.227 | 0.227 | 1 | 1.09e-15 | yes | 577843 | - |
| MUMPS direct | 9456 | 0.161 | 0.003 | 0.165 | 1 | 3.63e-16 | yes | 5925210 | - |
| FieldSplitSchur + GMRES | 9456 | 0.205 | 0.050 | 0.255 | 31 | 1.18e-08 | yes | - | - |
| SmoothedAgg AMG + GMRES | 9456 | 0.012 | 23.140 | 23.152 | 9456 | 2.48e-07 | no | - | 1.01 |
| ILU(τ=1e-03) + GMRES | 9456 | 0.043 | 0.490 | 0.533 | 646 | 1.49e-08 | yes | 528282 | - |
| ILU(τ=1e-04) + GMRES | 9456 | 0.092 | 0.017 | 0.109 | 16 | 7.89e-09 | yes | 1075150 | - |
| Split(primary AMG, Schur direct) | 9456 | 0.207 | 0.060 | 0.266 | 26 | 9.82e-09 | yes | - | 1.00 |
| Split(primary Schwarz, Schur AMG) | 9456 | 0.033 | 0.091 | 0.124 | 56 | 1.44e-08 | yes | - | 1.00 |
| Split(primary Schwarz, Schur ILU1e-3) | 9456 | 0.032 | 0.270 | 0.302 | 266 | 1.48e-08 | yes | 9862 | - |
| Split(primary Schwarz, Schur ILU1e-4) | 9456 | 0.204 | 0.148 | 0.352 | 142 | 1.44e-08 | yes | 58527 | - |
| HYPRE GMRES + ILU | 9456 | 0.036 | 0.098 | 0.134 | 74 | 1.46e-08 | yes | - | - |
| HYPRE FlexGMRES + AMG | 9456 | 0.024 | 0.137 | 0.161 | 24 | 1.46e-08 | yes | - | - |

### Step 2

| solver | dofs | setup s | solve s | total s | iter | residual | converged | fill | opcmp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Sparse direct | 9732 | 0.000 | 0.214 | 0.214 | 1 | 9.69e-16 | yes | 596987 | - |
| MUMPS direct | 9732 | 0.179 | 0.004 | 0.183 | 1 | 3.71e-16 | yes | 6042928 | - |
| FieldSplitSchur + GMRES | 9732 | 0.053 | 0.053 | 0.107 | 32 | 1.39e-08 | yes | - | - |
| SmoothedAgg AMG + GMRES | 9732 | 0.013 | 5.854 | 5.867 | 2313 | 1.47e-08 | yes | - | 1.01 |
| ILU(τ=1e-03) + GMRES | 9732 | 0.043 | 0.383 | 0.426 | 480 | 1.49e-08 | yes | 536313 | - |
| ILU(τ=1e-04) + GMRES | 9732 | 0.089 | 0.018 | 0.107 | 17 | 1.06e-08 | yes | 1086103 | - |
| Split(primary AMG, Schur direct) | 9732 | 0.210 | 0.063 | 0.272 | 26 | 1.16e-08 | yes | - | 1.00 |
| Split(primary Schwarz, Schur AMG) | 9732 | 0.035 | 0.104 | 0.138 | 62 | 1.48e-08 | yes | - | 1.00 |
| Split(primary Schwarz, Schur ILU1e-3) | 9732 | 0.034 | 0.438 | 0.472 | 425 | 1.48e-08 | yes | 10472 | - |
| Split(primary Schwarz, Schur ILU1e-4) | 9732 | 0.205 | 0.255 | 0.460 | 237 | 1.49e-08 | yes | 59835 | - |
| HYPRE GMRES + ILU | 9732 | 0.044 | 0.119 | 0.163 | 86 | 1.37e-08 | yes | - | - |
| HYPRE FlexGMRES + AMG | 9732 | 0.027 | 0.154 | 0.181 | 26 | 1.29e-08 | yes | - | - |

### Step 3

| solver | dofs | setup s | solve s | total s | iter | residual | converged | fill | opcmp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Sparse direct | 9798 | 0.000 | 0.211 | 0.211 | 1 | 9.89e-16 | yes | 601507 | - |
| MUMPS direct | 9798 | 0.166 | 0.003 | 0.169 | 1 | 3.50e-16 | yes | 6088754 | - |
| FieldSplitSchur + GMRES | 9798 | 0.054 | 0.060 | 0.114 | 35 | 1.21e-08 | yes | - | - |
| SmoothedAgg AMG + GMRES | 9798 | 0.013 | 8.470 | 8.483 | 3321 | 1.49e-08 | yes | - | 1.01 |
| ILU(τ=1e-03) + GMRES | 9798 | 0.045 | 0.497 | 0.542 | 616 | 1.48e-08 | yes | 537626 | - |
| ILU(τ=1e-04) + GMRES | 9798 | 0.117 | 0.019 | 0.135 | 18 | 9.05e-09 | yes | 1091152 | - |
| Split(primary AMG, Schur direct) | 9798 | 0.186 | 0.064 | 0.249 | 26 | 1.06e-08 | yes | - | 1.00 |
| Split(primary Schwarz, Schur AMG) | 9798 | 0.206 | 0.118 | 0.324 | 71 | 1.48e-08 | yes | - | 1.00 |
| Split(primary Schwarz, Schur ILU1e-3) | 9798 | 0.035 | 0.456 | 0.491 | 437 | 1.46e-08 | yes | 10576 | - |
| Split(primary Schwarz, Schur ILU1e-4) | 9798 | 0.037 | 0.182 | 0.219 | 169 | 1.44e-08 | yes | 53978 | - |
| HYPRE GMRES + ILU | 9798 | 0.045 | 0.092 | 0.137 | 66 | 1.44e-08 | yes | - | - |
| HYPRE FlexGMRES + AMG | 9798 | 0.027 | 0.124 | 0.151 | 21 | 1.34e-08 | yes | - | - |

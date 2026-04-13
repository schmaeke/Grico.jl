# Remote Phase 7 9950X Decision

Phase 7 should be kept.

The retained benchmark artifact is [remote_phase7_final_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase7_final_9950x.md:1), compared against the baseline in [remote_phase7_baseline_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase7_baseline_9950x.md:1).

## Findings

- The symmetric default path was already sensible.
  - `affine_cell_diffusion`: default `Smoothed Aggregation AMG + CG` stayed essentially unchanged at `8.06 ms -> 8.09 ms` cold and `5.30 ms -> 5.30 ms` warm.
  - `affine_interface_dg`: default `Smoothed Aggregation AMG + CG` stayed essentially unchanged at `13.05 ms -> 12.89 ms` cold and `1.54 ms -> 1.55 ms` warm.
- The unsymmetric default path was wrong for the cavity-sized mixed system.
  - Baseline default `ILU + GMRES` on `lid_driven_cavity_step1` took `1.097 s` cold and `1.083 s` warm.
  - Explicit sparse direct on the same system took only `70.81 ms` cold and `2.33 ms` warm.
  - Explicit `ILU`, `Additive Schwarz`, and `Field-Split Schur` were all much slower than direct on this case.

## Retained Changes

- Raise the default `ILUPreconditioner` application threshold from `2_000` to `10_000` reduced dofs.
- Cache the ordered direct factorization on `AffineSystem` so repeated direct solves reuse the same factor/operator.

## Result

For the mixed cavity case on `helios`:

- Default cold solve improved from `1.097 s` to `88.18 ms` (`12.44x`)
- Default warm solve improved from `1.083 s` to `2.39 ms` (`453.68x`)

The retained policy is therefore:

- symmetric assembled systems: keep the current AMG/CG default,
- medium unsymmetric assembled systems: use cached sparse direct solves by default,
- explicit ILU/Additive Schwarz/Field-Split remain available, but they are no longer the default for this medium-size regime.

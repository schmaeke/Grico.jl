# Remote Phase 5 9950X Decision

Phase 5 should be kept.

The retained artifact is [remote_phase5_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase5_9950x.md:1), with a confirmation rerun in [remote_phase5_rerun_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase5_rerun_9950x.md:1). The rerun stayed within normal benchmark noise, so the Phase 5 signal is stable on `helios`.

Compared against the retained pre-Phase-5 reference in [remote_phase4_default_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_phase4_default_9950x.md:1), the key gains at `16` threads are:

- `affine_cell_diffusion / assemble`: `189.01 ms -> 15.41 ms` (`12.27x`), memory `1445.60 MiB -> 33.10 MiB`, allocations `59,786,510 -> 394,510`
- `affine_interface_dg / assemble`: `41.02 ms -> 13.56 ms` (`3.03x`), memory `307.73 MiB -> 63.50 MiB`, allocations `9,606,024 -> 2,664`
- `nonlinear_interface_dg / residual!`: `4.38 ms -> 304.25 μs` (`14.39x`), memory `23.85 MiB -> 0.59 MiB`, allocations `1,273,001 -> 2,921`
- `nonlinear_interface_dg / tangent`: `28.16 ms -> 3.92 ms` (`7.18x`), memory `238.94 MiB -> 17.53 MiB`, allocations `11,320,668 -> 3,348`

The single-thread gains are also large, which matches the design of this phase:

- `affine_cell_diffusion / assemble`: `1.205 s -> 30.84 ms`
- `affine_interface_dg / assemble`: `190.12 ms -> 79.23 ms`
- `nonlinear_interface_dg / residual!`: `35.61 ms -> 916.13 μs`
- `nonlinear_interface_dg / tangent`: `142.65 ms -> 24.51 ms`

The most plausible explanation is that the old generic gradient and normal-gradient accessors were still creating substantial tuple/allocation overhead in the hottest local kernels, and the Phase 5 specialization removed that overhead across cell, boundary, and interface operators at once.

Conclusion: retain the Phase 5 accessor specialization and move on.

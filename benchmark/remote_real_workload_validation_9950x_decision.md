# Real-Workload Validation Decision On `helios`

Reference report: [remote_real_workload_validation_9950x.md](/Users/schmaeke/Projects/Grico.jl/benchmark/remote_real_workload_validation_9950x.md:1)

Retained validation scope:
- `annular_plate_nitsche` on a `4 x 4` background mesh with `512` boundary segments
- `origin_singularity_poisson` with the shipped `20` adaptive steps
- `lid_driven_cavity` with the shipped `16 x 16` root grid and `4` adaptive cycles
- thread counts `1, 2, 4, 8, 16`

## Outcome

`Grico` is now in a reasonable state for single-node work, but the validation also shows that the remaining bottlenecks are workload-dependent.

- `annular_plate_nitsche` is not a shared-memory scaling target at its current size. It improved only from `280.17 ms` to `242.97 ms` at `16` threads (`1.15x`). The dominant cost is setup of the finite-cell / embedded-surface workload, not repeated assembly or solve.
- `origin_singularity_poisson` also does not benefit from more threads at its current scale. It moved from `351.80 ms` to `364.15 ms` (`0.97x`) and is dominated by repeated `compile(problem)` work during the adaptive loop.
- `lid_driven_cavity` is the real single-node workload. It improved from `16.752 s` to `7.108 s` at `16` threads (`2.36x`), but scaling is already saturated by `8` threads (`7.129 s`). At `16` threads, solve time (`3.081 s`) is slightly larger than assembly (`2.899 s`), so the next meaningful one-node gains are no longer primarily in assembly.

## Decision

- Keep the shared-memory assembly and kernel work from the earlier phases. The cavity workload confirms that those changes produced a real end-to-end improvement.
- Do not use `annular_plate_nitsche` or `origin_singularity_poisson` as primary thread-scaling benchmarks. They are useful acceptance workloads, but not good node-scaling drivers.
- Use `lid_driven_cavity` as the primary retained single-node acceptance benchmark on `helios`.
- If the goal remains single-node performance only, the next work should focus on:
  - cavity solve-path cost,
  - adaptive-workflow compile/setup reuse,
  - and possibly workload-specific reductions in diagnostics/adaptivity overhead.

# Solver Path Experiments

Run the current solver-path benchmark harness with:

```bash
julia --project=benchmark benchmark/solver_path_experiments.jl
```

The benchmark suite now samples two problem families:

- a symmetric positive definite scalar diffusion problem, and
- mixed velocity-pressure systems taken from the DG lid-driven cavity Picard iteration.

Notes:

- The script forces one-thread defaults for reproducibility in the mixed
  MPI/OpenMP setup:
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `BLAS.set_num_threads(1)`
- `KMP_DUPLICATE_LIB_OK=TRUE` is set inside the benchmark script to avoid the
  duplicate OpenMP runtime abort seen on some macOS configurations when loading
  both `HYPRE` and `MUMPS`.
- The benchmark scripts can now be safely `include`d without running their main
  entry point. Execution only starts when the file is launched as the program.

The previous detailed tables were tied to the removed Kelvin-Helmholtz example
and are intentionally no longer kept here. Rerun the benchmark command above to
generate fresh measurements for the current cavity-based workflow.

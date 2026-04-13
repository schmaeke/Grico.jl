# Phase 2 COO vs Symbolic/Numeric Comparison

Comparison baseline:
- old path: [phase1_threaded_assembly_memory.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase1_threaded_assembly_memory.md:1)
- new path: [phase2_symbolic_numeric_assembly.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2_symbolic_numeric_assembly.md:1)

## Key Deltas

| Case / operation | Metric | Phase 1 | Phase 2 | Change |
| --- | --- | ---: | ---: | ---: |
| `affine_cell_diffusion / assemble / 1 thread` | median time | `1.803 s` | `1.727 s` | `0.96x` |
| `affine_cell_diffusion / assemble / 6 threads` | median time | `443.06 ms` | `520.94 ms` | `1.18x` |
| `affine_cell_diffusion / assemble / 6 threads` | median memory | `1481.38 MiB` | `1442.27 MiB` | `0.97x` |
| `affine_interface_dg / assemble / 1 thread` | median time | `160.19 ms` | `179.78 ms` | `1.12x` |
| `affine_interface_dg / assemble / 6 threads` | median time | `96.11 ms` | `48.96 ms` | `0.51x` |
| `affine_interface_dg / assemble / 6 threads` | median memory | `529.11 MiB` | `299.95 MiB` | `0.57x` |
| `nonlinear_interface_dg / tangent / 1 thread` | median time | `182.84 ms` | `174.96 ms` | `0.96x` |
| `nonlinear_interface_dg / tangent / 6 threads` | median time | `56.22 ms` | `45.36 ms` | `0.81x` |
| `nonlinear_interface_dg / tangent / 6 threads` | median memory | `314.34 MiB` | `229.57 MiB` | `0.73x` |

## Interpretation

- Phase 2 achieved its main structural goal: the runtime no longer rebuilds the sparse graph for the reduced affine matrix or nonlinear tangent.
- The strongest payoff is on the interface-heavy affine case. At 6 threads, `assemble(plan)` improves from `96.11 ms` to `48.96 ms`, while memory drops from `529.11 MiB` to `299.95 MiB`.
- Nonlinear tangent assembly also benefits materially. At 6 threads, `tangent(plan, state)` improves from `56.22 ms` to `45.36 ms`, with memory dropping from `314.34 MiB` to `229.57 MiB`.
- The cell-dominated affine case is mixed: serial assembly improves slightly, but the 6-thread run is slower than the Phase 1 path. Inference: the current chunk-end locked `nzval` reduction is cheaper than full graph reconstruction for irregular/interface-heavy work, but it still introduces contention on regular high-volume cell assembly.
- Residual performance is essentially unchanged in architectural terms, which is expected because Phase 2 targeted sparse matrix assembly rather than vector assembly.

## Conclusion

Phase 2 is a net win and should remain the default path. The remaining shared-memory issue is no longer sparse-graph reconstruction; it is numeric accumulation strategy on highly regular threaded assembly. That becomes the next tuning target.

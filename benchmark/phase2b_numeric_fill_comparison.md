# Phase 2b Numeric Fill Comparison

Comparison baselines:
- Phase 1 memory-behavior refactor: [phase1_threaded_assembly_memory.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase1_threaded_assembly_memory.md:1)
- Phase 2 symbolic + numeric assembly: [phase2_symbolic_numeric_assembly.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2_symbolic_numeric_assembly.md:1)
- Phase 2b final run: [phase2b_final_numeric_fill.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2b_final_numeric_fill.md:1)

## Backend Choice

Phase 2b keeps the symbolic CSC structure from Phase 2 and only changes the numeric fill backend:

- multi-threaded affine and tangent assembly without interface operators use a bounded thread-local slot accumulator,
- interface-coupled assembly keeps the lock-based sparse slot reduction from Phase 2,
- single-thread runs stay on the buffered sparse slot path.

That split is intentional. The thread-local accumulator helps regular cell-dominated work by removing hot-loop synchronization, while interface-heavy paths already performed well enough with the Phase 2 reduction backend and do not benefit from the extra per-thread slot storage.

## Key Deltas At 6 Threads

| Case / operation | Metric | Phase 1 | Phase 2 | Phase 2b | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `affine_cell_diffusion / assemble` | median time | `443.06 ms` | `520.94 ms` | `380.45 ms` | Phase 2b fixes the Phase 2 regression on the regular cell-dominated path and improves on the Phase 1 COO path. |
| `affine_cell_diffusion / assemble` | median memory | `1481.38 MiB` | `1442.27 MiB` | `1453.35 MiB` | Slightly higher than Phase 2 because of bounded thread-local slot storage, still lower than Phase 1. |
| `affine_interface_dg / assemble` | median time | `96.11 ms` | `48.96 ms` | `49.73 ms` | Phase 2b preserves the Phase 2 interface-heavy speedup by leaving that path on the lock-based reducer. |
| `affine_interface_dg / assemble` | median memory | `529.11 MiB` | `299.95 MiB` | `299.95 MiB` | No material change from Phase 2. |
| `nonlinear_interface_dg / tangent` | median time | `56.22 ms` | `45.36 ms` | `48.37 ms` | Slightly slower than the best Phase 2 run, still materially better than Phase 1. |
| `nonlinear_interface_dg / tangent` | median memory | `314.34 MiB` | `229.57 MiB` | `229.57 MiB` | No material change from Phase 2. |

## Interpretation

- Phase 2b completes the numeric-fill part of Phase 2. The regular shared-memory affine path is no longer limited by the chunk-end global `nzval` lock.
- The clean outcome is on `affine_cell_diffusion`: at 6 threads the final backend is about `1.37x` faster than Phase 2 and about `1.16x` faster than the old Phase 1 COO path.
- The interface-heavy affine case keeps the main Phase 2 win, which confirms that the selector should stay structural rather than forcing one numeric fill backend everywhere.
- The nonlinear tangent path remains better than Phase 1, but the gap to the best Phase 2 run is small enough that further gains are unlikely to come from numeric fill alone.

## Benchmark Note

- The 1-thread `nonlinear_interface_dg` residual and tangent numbers varied noticeably across repeated benchmark sessions on this 6-core host even though Phase 2b does not change the residual path and leaves the interface-coupled tangent path on the Phase 2 backend.
- Because of that run-to-run variance, the Phase 2b conclusion is based primarily on the 6-thread assembly/tangent comparisons and on the unchanged memory footprint for the untouched interface-coupled paths.

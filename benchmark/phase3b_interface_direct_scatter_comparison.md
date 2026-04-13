# Phase 3b Interface Direct Scatter Comparison

Comparison baselines:
- Phase 2d retained ownership backend: [phase2d_ownership_mailbox_merge.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2d_ownership_mailbox_merge.md:1)
- Phase 3 batched traversal: [phase3_batched_local_traversal.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase3_batched_local_traversal.md:1)
- Phase 3b direct scatter: [phase3b_interface_direct_scatter.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase3b_interface_direct_scatter.md:1)

## Design Change

Phase 3b keeps the Phase 2 ownership-based accumulation backend and the Phase 3 batched traversal layer, but adds one extra compiled batch property:

- batches whose local dofs all map directly to single global dofs use a dedicated scatter path,
- the fast path still goes through the same ownership accumulators,
- cells, boundary faces, interfaces, and embedded surfaces all share that same structural choice.

This targets DG-style local items, especially interface-heavy batches, without reintroducing separate serial and threaded assembly backends.

## Key Deltas At 6 Threads

| Case / operation | Metric | Phase 2d | Phase 3 | Phase 3b | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `affine_cell_diffusion / assemble` | median time | `395.70 ms` | `381.10 ms` | `374.03 ms` | The direct scatter path also helps regular one-to-one cell work a bit more. |
| `affine_cell_diffusion / assemble` | median memory | `1454.85 MiB` | `1453.32 MiB` | `1453.44 MiB` | Memory is effectively unchanged. |
| `affine_interface_dg / assemble` | median time | `76.75 ms` | `79.28 ms` | `70.92 ms` | Phase 3b recovers and improves the interface-heavy affine path by removing generic term-expansion scatter overhead on DG interfaces. |
| `affine_interface_dg / assemble` | median memory | `359.38 MiB` | `361.26 MiB` | `356.01 MiB` | Memory improves slightly with the direct scatter path. |
| `nonlinear_interface_dg / residual_bang` | median time | `9.31 ms` | `7.63 ms` | `7.98 ms` | Residual remains materially better than Phase 2d, though this particular case is slightly slower than the best Phase 3 run. |
| `nonlinear_interface_dg / tangent` | median time | `50.45 ms` | `49.17 ms` | `47.77 ms` | Tangent benefits modestly from the same direct scatter specialization. |
| `nonlinear_interface_dg / tangent` | median memory | `256.39 MiB` | `260.50 MiB` | `259.28 MiB` | Memory stays in the same range. |

## Interpretation

- The retained Phase 3b change is coherent with the Phase 3 design: it is a local scatter specialization chosen from compiled batch metadata, not a new backend.
- The main intended target improves: interface-heavy affine assembly is now faster than both Phase 2d and Phase 3 on this host.
- The nonlinear tangent path also improves modestly.
- The nonlinear residual path does not get the best overall number, so Phase 3 batching remains the more important change there.

## Conclusion

- Phase 3 now has two clean layers:
  - Phase 3: batch and prefilter local traversal,
  - Phase 3b: specialize one-to-one local/global scatter inside those batches.
- The next step can move on to Phase 4 scheduling, unless later profiling on the larger machine still points at interface-specific kernel structure rather than scheduling.

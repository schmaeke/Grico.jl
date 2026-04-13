# Phase 2c Ownership-Based Numeric Fill Comparison

Comparison baselines:
- Phase 2b final run: [phase2b_final_numeric_fill.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2b_final_numeric_fill.md:1)
- Phase 2c ownership-based run: [phase2c_ownership_numeric_fill.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2c_ownership_numeric_fill.md:1)

## Design Change

Phase 2c removes the backend split introduced in Phase 2b.

- One ownership-based sparse/vector accumulation backend is now used for affine matrix assembly, affine RHS assembly, nonlinear tangent fill, nonlinear residual fill, and the Dirichlet boundary projection helper.
- Serial execution is now the natural `Threads.nthreads() == 1` case of the same accumulation design used for threaded execution.
- Cells, interfaces, boundary faces, and embedded surfaces still keep their distinct local kernels where the mathematics differs, but all global accumulation now flows through the same ownership-based scatter layer.

## Key Deltas At 6 Threads

| Case / operation | Metric | Phase 2b | Phase 2c | Interpretation |
| --- | --- | ---: | ---: | --- |
| `affine_cell_diffusion / assemble` | median time | `380.45 ms` | `449.45 ms` | The unified ownership backend regresses the regular cell-dominated affine path by about `1.18x`. |
| `affine_cell_diffusion / assemble` | median memory | `1453.35 MiB` | `1454.13 MiB` | Memory is effectively unchanged on the regular affine case. |
| `affine_interface_dg / assemble` | median time | `49.73 ms` | `68.87 ms` | Interface-heavy affine assembly regresses by about `1.38x`, which points to remaining overhead in owner handoff for coupled terms. |
| `affine_interface_dg / assemble` | median memory | `299.95 MiB` | `348.76 MiB` | Ownership inbox/foreign buffers increase memory on the interface-heavy affine case. |
| `nonlinear_interface_dg / residual_bang` | median time | `12.08 ms` | `9.19 ms` | Residual assembly improves modestly under the unified vector accumulation path. |
| `nonlinear_interface_dg / tangent` | median time | `48.37 ms` | `48.06 ms` | Tangent fill is effectively unchanged; the ownership backend preserves the best Phase 2b tangent result. |

## Interpretation

- Phase 2c achieves the architectural goal: the sparse assembly layer now has one coherent accumulation model instead of separate serial and threaded numeric-fill backends.
- That coherence is not free. On this 6-core host, the ownership backend is slower than the specialized Phase 2b backend on both affine assembly stress cases.
- The current evidence says the remaining cost sits in ownership handoff and foreign-contribution reduction, not in symbolic sparse structure reuse, which was already solved in Phase 2.
- That makes the next assembly-focused task clearer: keep the ownership model, but reduce foreign-owner traffic and synchronization cost inside the numeric fill backend before moving on to broader kernel specialization work.

## Benchmark Note

- The `1`-thread baseline numbers moved noticeably between Phase 2b and Phase 2c benchmark sessions, so the most reliable comparison here is the `6`-thread steady-state behavior on the targeted assembly paths.
- The benchmark conclusion for Phase 2c is therefore architectural first and performance second: the design is cleaner and more coherent, but the affine numeric fill backend still needs another optimization pass.

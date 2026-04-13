# Phase 2d Ownership Mailbox Merge Comparison

Comparison baselines:
- Phase 2c ownership-based run: [phase2c_ownership_numeric_fill.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2c_ownership_numeric_fill.md:1)
- Phase 2d mailbox-merge run: [phase2d_ownership_mailbox_merge.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2d_ownership_mailbox_merge.md:1)

## Design Change

Phase 2d keeps the ownership-based assembly model from Phase 2c, but removes synchronization from the threaded hot path:

- worker-local matrix and RHS contributions still accumulate into owned buffers and foreign-owner mailboxes,
- foreign contributions are no longer flushed under locks while the worker tasks are running,
- instead, foreign mailboxes are merged once, after the threaded pass has finished,
- the final numeric fill then writes each owner buffer into the preallocated sparse/vector target exactly once.

That keeps serial as the natural `1-thread` case of the same backend while simplifying the threaded path.

## Key Deltas At 6 Threads

| Case / operation | Metric | Phase 2c | Phase 2d | Interpretation |
| --- | --- | ---: | ---: | --- |
| `affine_cell_diffusion / assemble` | median time | `449.45 ms` | `395.70 ms` | The lock-free mailbox merge recovers most of the regular affine gap, improving the ownership backend by about `1.14x`. |
| `affine_cell_diffusion / assemble` | median memory | `1454.13 MiB` | `1454.85 MiB` | Memory is effectively unchanged. |
| `affine_interface_dg / assemble` | median time | `68.87 ms` | `76.75 ms` | Interface-heavy affine assembly regresses somewhat; deferred mailbox merge increases foreign-buffer traffic on this case. |
| `affine_interface_dg / assemble` | median memory | `348.76 MiB` | `359.38 MiB` | Interface-heavy memory increases modestly with larger foreign mailboxes. |
| `nonlinear_interface_dg / residual_bang` | median time | `9.19 ms` | `9.31 ms` | Residual assembly is effectively unchanged. |
| `nonlinear_interface_dg / tangent` | median time | `48.06 ms` | `50.45 ms` | Tangent fill stays in the same range; the main benefit of Phase 2d is on affine assembly. |

## Scheduling Follow-Up

Two follow-up scheduler experiments were run after Phase 2d:

- [phase2e_ownership_static_scheduling.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2e_ownership_static_scheduling.md:1)
- [phase2f_ownership_hybrid_scheduling.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2f_ownership_hybrid_scheduling.md:1)

Those experiments improved some ownership-heavy interface cases, but they did not produce a stable whole-backend win and they made the implementation less coherent. They were therefore not retained in the code.

## Interpretation

- Phase 2d is the clean retained improvement: one ownership-based backend, no hot-path synchronization, and no separate serial/threaded numeric-fill implementations.
- The remaining shared-memory weakness is now much narrower than before. The regular affine path benefits from the mailbox-merge refactor, while the interface-heavy affine path still pays for foreign-owner traffic.
- That makes the next clean step clearer: move on to Phase 3 kernel batching/specialization, then revisit ownership partitioning only if those higher-payoff changes leave interface-heavy affine assembly behind.

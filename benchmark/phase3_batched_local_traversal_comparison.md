# Phase 3 Batched Local Traversal Comparison

Comparison baselines:
- Phase 2d retained backend: [phase2d_ownership_mailbox_merge.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase2d_ownership_mailbox_merge.md:1)
- Phase 3 batched traversal: [phase3_batched_local_traversal.md](/Users/schmaeke/Projects/Grico.jl/benchmark/phase3_batched_local_traversal.md:1)

## Design Change

Phase 3 keeps the Phase 2 ownership-based sparse/vector accumulation backend and changes the local traversal layer:

- cells, faces, interfaces, and surfaces are grouped into compile-time batches by local kernel signature,
- boundary and surface selector filtering is compiled once into batch metadata instead of being re-evaluated inside the hot loops,
- runtime passes now traverse homogeneous batches with one fixed local buffer shape per batch.

This is a local-kernel regularization step, not a new algebra backend.

## Key Deltas At 6 Threads

| Case / operation | Metric | Phase 2d | Phase 3 | Interpretation |
| --- | --- | ---: | ---: | --- |
| `affine_cell_diffusion / assemble` | median time | `395.70 ms` | `381.10 ms` | Cell-dominated affine assembly improves by about `1.04x`; batching pays off on the regular volume path. |
| `affine_cell_diffusion / assemble` | median memory | `1454.85 MiB` | `1453.32 MiB` | Memory is effectively unchanged. |
| `affine_interface_dg / assemble` | median time | `76.75 ms` | `79.28 ms` | Interface-heavy affine assembly is effectively flat, slightly slower by about `3%`. |
| `affine_interface_dg / assemble` | median memory | `359.38 MiB` | `361.26 MiB` | Memory is essentially unchanged. |
| `nonlinear_interface_dg / residual_bang` | median time | `9.31 ms` | `7.63 ms` | Batched traversal materially improves the nonlinear residual path. |
| `nonlinear_interface_dg / tangent` | median time | `50.45 ms` | `49.17 ms` | Tangent fill improves modestly while keeping the same ownership backend. |

## Interpretation

- Phase 3 does what it was supposed to do: it improves the regular local-work side without disturbing the ownership-based assembly design from Phase 2.
- The strongest visible gain is on the nonlinear residual path, where removing repeated selector checks and reusing homogeneous local buffer shapes lowers per-item overhead.
- The regular affine cell path also improves slightly, which is consistent with the intended batching/locality effect.
- The interface-heavy affine case does not yet benefit materially, which suggests that its remaining limit is still more about interface coupling structure than about generic local traversal overhead.

## Conclusion

- Phase 2 and Phase 3 now separate cleanly:
  - Phase 2: coherent ownership-based assembly backend,
  - Phase 3: batched and prefiltered local traversal on top of that backend.
- The next worthwhile step is not another generic traversal rewrite. It should be more targeted kernel work for the remaining interface-heavy and tangent hot spots.

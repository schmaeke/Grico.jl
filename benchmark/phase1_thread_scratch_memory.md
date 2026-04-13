# Phase 1 Threaded Assembly Memory

Generated: `2026-04-12T18:51:51`

Input artifact: `phase1_threaded_assembly_memory.toml`

## Summary

- `_ThreadScratch` no longer stores one dense global RHS vector per worker.
- Persistent scratch size now depends on `max_local_dofs`, not on the global RHS target length.
- The new `rhs_rows` / `rhs_values` buffers are sparse chunk-local accumulators that are flushed after each claimed chunk instead of staying dense for the full solve space.

## Measured Case-Level Effect

| Case / operation | RHS target dofs | Max local dofs | New scratch per worker | Dense RHS removed per worker | Dense RHS removed at 1 threads | Dense RHS removed at 2 threads | Dense RHS removed at 4 threads | Dense RHS removed at 6 threads | Dense RHS removed at 16 threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `affine_cell_diffusion / assemble` | 7761 | 16 | 6.85 KiB | 60.63 KiB | 60.63 KiB | 121.27 KiB | 242.53 KiB | 363.80 KiB | 970.12 KiB |
| `affine_interface_dg / assemble` | 14400 | 18 | 8.48 KiB | 112.50 KiB | 112.50 KiB | 225.00 KiB | 450.00 KiB | 675.00 KiB | 1.76 MiB |
| `nonlinear_interface_dg / residual_bang` | 7056 | 18 | 8.48 KiB | 55.12 KiB | 55.12 KiB | 110.25 KiB | 220.50 KiB | 330.75 KiB | 882.00 KiB |

## Projected Dense RHS Overhead Removed

| RHS target dofs | Removed at 1 thread | Removed at 2 threads | Removed at 4 threads | Removed at 6 threads | Removed at 16 threads |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100000 | 781.25 KiB | 1.53 MiB | 3.05 MiB | 4.58 MiB | 12.21 MiB |
| 1000000 | 7.63 MiB | 15.26 MiB | 30.52 MiB | 45.78 MiB | 122.07 MiB |
| 10000000 | 76.29 MiB | 152.59 MiB | 305.18 MiB | 457.76 MiB | 1.19 GiB |

## Interpretation

- On the current Phase 0 benchmark cases, the removed dense RHS vector is small compared with the total COO and sparse-construction allocation volume, so median per-call memory barely moves.
- The architectural gain is that the persistent RHS path is no longer `O(thread_count × rhs_target_dofs)`. Baseline thread-local scratch now scales with the local integration size, while temporary RHS storage scales with recently claimed work and is cleared at chunk boundaries.
- This is a prerequisite for larger one-node runs, but it does not replace the need for Phase 2. The remaining dominant memory traffic still comes from COO triplet growth and `sparse(...)` reconstruction.
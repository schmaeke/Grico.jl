# Real-Workload Single-Node Validation

Generated: `2026-04-13T12:54:11`

Profile: `smoke`

Thread counts: `1`

Validation runs use the requested physical-core thread counts on this host.

## Environment

- Julia: `1.12.5`
- Host: `schmaekes-MBP`
- CPU: `Apple M2 Pro`
- BLAS threads: `1`

## Cases

### `annular_plate_nitsche`

Unfitted scalar annulus solve with finite-cell quadrature and embedded Nitsche boundary terms.

Configuration:

- `degree`: `4`
- `fcm_subdivision_depth`: `5`
- `penalty`: `40.0`
- `root_counts`: `4, 4`
- `segment_count`: `128`
- `surface_point_count`: `3`

| Threads | Wall time | Speedup | Efficiency | Allocated | GC time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 208.60 ms | 1.00 | 1.00 | 160.18 MiB | 0.00 ns |

Retained outcome at `1` threads:

- `active_leaves`: `16`
- `matrix_nnz`: `3505`
- `reduced_dofs`: `145`
- `relative_l2_error`: `0.0015349281612585734`
- `scalar_dofs`: `161`

Phase breakdown at `1` threads:

| Phase | Time | Share |
| --- | ---: | ---: |
| `setup_seconds` | 108.24 ms | 80.4% |
| `assemble_seconds` | 18.60 ms | 13.8% |
| `compile_seconds` | 7.12 ms | 5.3% |
| `verify_seconds` | 398.25 μs | 0.3% |
| `solve_seconds` | 296.00 μs | 0.2% |

### `origin_singularity_poisson`

Adaptive hp Poisson solve with verification and repeated rebuilds on the singular corner problem.

Configuration:

- `adaptive_steps`: `6`
- `dimension`: `2`
- `initial_degree`: `2`

| Threads | Wall time | Speedup | Efficiency | Allocated | GC time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 121.58 ms | 1.00 | 1.00 | 27.86 MiB | 0.00 ns |

Retained outcome at `1` threads:

- `active_leaves`: `16`
- `completed_steps`: `6`
- `final_matrix_nnz`: `1494`
- `final_relative_l2_error`: `0.010514140040861545`
- `max_active_leaves`: `16`
- `max_scalar_dofs`: `102`
- `reduced_dofs`: `82`
- `scalar_dofs`: `102`
- `stopped_early`: `false`

Phase breakdown at `1` threads:

| Phase | Time | Share |
| --- | ---: | ---: |
| `compile_seconds` | 2.45 ms | 47.9% |
| `transfer_seconds` | 1.35 ms | 26.4% |
| `assemble_seconds` | 524.83 μs | 10.3% |
| `solve_seconds` | 398.08 μs | 7.8% |
| `adaptivity_seconds` | 282.37 μs | 5.5% |
| `verify_seconds` | 72.50 μs | 1.4% |
| `problem_setup_seconds` | 35.04 μs | 0.7% |

### `lid_driven_cavity`

Adaptive mixed DG lid-driven cavity solve with repeated Picard linearizations and mesh adaptation.

Configuration:

- `adaptive_steps`: `1`
- `max_iters`: `6`
- `root_counts`: `8, 8`
- `tolerance`: `0.0001`

| Threads | Wall time | Speedup | Efficiency | Allocated | GC time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.391 s | 1.00 | 1.00 | 1724.41 MiB | 203.24 ms |

Retained outcome at `1` threads:

- `active_leaves`: `80`
- `adaptive_steps_used`: `1`
- `dg_mass_monitor_l2`: `0.37646023583073474`
- `final_picard_iters`: `6`
- `final_relative_update`: `0.0001371309301685526`
- `max_mixed_dofs`: `1600`
- `mixed_dofs`: `1600`
- `velocity_scalar_dofs`: `640`

Phase breakdown at `1` threads:

| Phase | Time | Share |
| --- | ---: | ---: |
| `assemble_seconds` | 1.105 s | 81.3% |
| `solve_seconds` | 163.79 ms | 12.0% |
| `adaptivity_seconds` | 51.84 ms | 3.8% |
| `setup_seconds` | 25.86 ms | 1.9% |
| `diagnostics_seconds` | 13.53 ms | 1.0% |

## Findings

- `annular_plate_nitsche` improves from 208.60 ms at 1 thread to 208.60 ms at 1 thread, or `1.00x` with `1.00` efficiency. The dominant retained phase at 1 thread is `setup_seconds` with 108.24 ms.
- `origin_singularity_poisson` improves from 121.58 ms at 1 thread to 121.58 ms at 1 thread, or `1.00x` with `1.00` efficiency. The dominant retained phase at 1 thread is `compile_seconds` with 2.45 ms.
- `lid_driven_cavity` improves from 1.391 s at 1 thread to 1.391 s at 1 thread, or `1.00x` with `1.00` efficiency. The dominant retained phase at 1 thread is `assemble_seconds` with 1.105 s.
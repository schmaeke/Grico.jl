# Real-Workload Single-Node Validation

Generated: `2026-04-13T12:43:58`

Profile: `validation`

Thread counts: `1, 2, 4, 8, 16`

Validation runs use the requested physical-core thread counts on this host.

## Environment

- Julia: `1.12.5`
- Host: `helios`
- CPU: `AMD Ryzen 9 9950X 16-Core Processor`
- BLAS threads: `1`

## Cases

### `annular_plate_nitsche`

Unfitted scalar annulus solve with finite-cell quadrature and embedded Nitsche boundary terms.

Configuration:

- `degree`: `4`
- `fcm_subdivision_depth`: `7`
- `penalty`: `40.0`
- `root_counts`: `4, 4`
- `segment_count`: `512`
- `surface_point_count`: `3`

| Threads | Wall time | Speedup | Efficiency | Allocated | GC time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 280.17 ms | 1.00 | 1.00 | 388.63 MiB | 15.68 ms |
| 2 | 244.02 ms | 1.15 | 0.57 | 392.08 MiB | 10.57 ms |
| 4 | 240.59 ms | 1.16 | 0.29 | 392.20 MiB | 18.05 ms |
| 8 | 233.80 ms | 1.20 | 0.15 | 392.35 MiB | 20.38 ms |
| 16 | 242.97 ms | 1.15 | 0.07 | 392.58 MiB | 28.48 ms |

Retained outcome at `16` threads:

- `active_leaves`: `16`
- `matrix_nnz`: `3505`
- `reduced_dofs`: `145`
- `relative_l2_error`: `0.0015062075806711156`
- `scalar_dofs`: `161`

Phase breakdown at `16` threads:

| Phase | Time | Share |
| --- | ---: | ---: |
| `setup_seconds` | 116.34 ms | 72.1% |
| `compile_seconds` | 24.45 ms | 15.2% |
| `assemble_seconds` | 20.04 ms | 12.4% |
| `solve_seconds` | 284.32 μs | 0.2% |
| `verify_seconds` | 242.70 μs | 0.2% |

### `origin_singularity_poisson`

Adaptive hp Poisson solve with verification and repeated rebuilds on the singular corner problem.

Configuration:

- `adaptive_steps`: `20`
- `dimension`: `2`
- `initial_degree`: `2`

| Threads | Wall time | Speedup | Efficiency | Allocated | GC time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 351.80 ms | 1.00 | 1.00 | 311.34 MiB | 0.00 ns |
| 2 | 352.74 ms | 1.00 | 0.50 | 315.34 MiB | 0.00 ns |
| 4 | 357.28 ms | 0.98 | 0.25 | 318.93 MiB | 7.65 ms |
| 8 | 356.76 ms | 0.99 | 0.12 | 322.83 MiB | 7.34 ms |
| 16 | 364.15 ms | 0.97 | 0.06 | 326.97 MiB | 14.74 ms |

Retained outcome at `16` threads:

- `active_leaves`: `76`
- `completed_steps`: `20`
- `final_matrix_nnz`: `20718`
- `final_relative_l2_error`: `8.20935384919933e-5`
- `max_active_leaves`: `76`
- `max_scalar_dofs`: `967`
- `reduced_dofs`: `650`
- `scalar_dofs`: `967`
- `stopped_early`: `false`

Phase breakdown at `16` threads:

| Phase | Time | Share |
| --- | ---: | ---: |
| `compile_seconds` | 181.98 ms | 74.0% |
| `solve_seconds` | 21.28 ms | 8.6% |
| `assemble_seconds` | 19.73 ms | 8.0% |
| `transfer_seconds` | 18.75 ms | 7.6% |
| `adaptivity_seconds` | 3.53 ms | 1.4% |
| `verify_seconds` | 664.86 μs | 0.3% |
| `problem_setup_seconds` | 133.45 μs | 0.1% |

### `lid_driven_cavity`

Adaptive mixed DG lid-driven cavity solve with repeated Picard linearizations and mesh adaptation.

Configuration:

- `adaptive_steps`: `4`
- `max_iters`: `24`
- `root_counts`: `16, 16`
- `tolerance`: `1.0e-6`

| Threads | Wall time | Speedup | Efficiency | Allocated | GC time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16.752 s | 1.00 | 1.00 | 22002.22 MiB | 3.579 s |
| 2 | 11.254 s | 1.49 | 0.74 | 22194.70 MiB | 2.530 s |
| 4 | 8.726 s | 1.92 | 0.48 | 22369.85 MiB | 2.046 s |
| 8 | 7.129 s | 2.35 | 0.29 | 22438.11 MiB | 1.621 s |
| 16 | 7.108 s | 2.36 | 0.15 | 22436.78 MiB | 1.824 s |

Retained outcome at `16` threads:

- `active_leaves`: `372`
- `adaptive_steps_used`: `4`
- `dg_mass_monitor_l2`: `0.37504616949060593`
- `final_picard_iters`: `7`
- `final_relative_update`: `6.932391693583903e-7`
- `max_mixed_dofs`: `7440`
- `mixed_dofs`: `7440`
- `velocity_scalar_dofs`: `2976`

Phase breakdown at `16` threads:

| Phase | Time | Share |
| --- | ---: | ---: |
| `solve_seconds` | 3.081 s | 43.6% |
| `assemble_seconds` | 2.899 s | 41.0% |
| `adaptivity_seconds` | 791.68 ms | 11.2% |
| `diagnostics_seconds` | 163.16 ms | 2.3% |
| `setup_seconds` | 140.17 ms | 2.0% |

## Findings

- `annular_plate_nitsche` improves from 280.17 ms at 1 thread to 242.97 ms at 16 threads, or `1.15x` with `0.07` efficiency. The dominant retained phase at 16 threads is `setup_seconds` with 116.34 ms.
- `origin_singularity_poisson` improves from 351.80 ms at 1 thread to 364.15 ms at 16 threads, or `0.97x` with `0.06` efficiency. The dominant retained phase at 16 threads is `compile_seconds` with 181.98 ms.
- `lid_driven_cavity` improves from 16.752 s at 1 thread to 7.108 s at 16 threads, or `2.36x` with `0.15` efficiency. The dominant retained phase at 16 threads is `solve_seconds` with 3.081 s.
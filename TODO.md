# Grico Shared-Memory HPC Roadmap

## Goal

Make `Grico.jl` scale well on modern shared-memory systems before introducing a full distributed-memory architecture.

This roadmap is intentionally staged. Each phase should leave the codebase in a working state with tests passing and benchmark data recorded.

## Guiding Principles

- Optimize the current one-process architecture first.
- Separate discretization logic from algebra/backend choices.
- Prefer measurable wins over speculative refactors.
- Keep all changes benchmark-driven.
- Design new abstractions so they can later support MPI/distributed memory.

## Non-Goals For This Roadmap

- Full MPI support in the core package.
- A complete distributed adaptive mesh infrastructure.
- GPU enablement.

## Definition Of Success

At the end of this roadmap, `Grico.jl` should:

- show useful strong scaling on one node for `assemble`, `residual!`, and `tangent!`,
- avoid pathological memory growth with thread count,
- avoid rebuilding sparse structure on every assembly,
- provide benchmark evidence for which kernels and data paths still limit scaling,
- expose an assembly/algebra structure that can later be extended to distributed memory.

## Baseline Rules

- Every performance change must come with before/after benchmark data.
- Every phase must end with:
  - `test/runtests.jl` passing,
  - at least one benchmark artifact checked into `benchmark/` or documented in a markdown report,
  - a short note about what improved and what remains limiting.
- Unless a phase explicitly says otherwise, preserve the public API.

## Phase 0: Benchmark And Profiling Baseline

### Objective

Establish a trustworthy baseline for serial and threaded behavior before changing internals.

### Tasks

- Add dedicated benchmarks for:
  - `assemble(plan)`,
  - `residual!(result, plan, state)`,
  - `tangent!(matrix, plan, state)` or `tangent(plan, state)`,
  - preconditioner build time,
  - solve time separately from assembly time.
- Benchmark at multiple thread counts, at least:
  - `1, 2, 4, 8`,
  - and higher counts if the machine supports them.
- Record:
  - wall time,
  - allocations,
  - memory,
  - speedup versus 1 thread,
  - efficiency versus 1 thread.
- Create representative benchmark cases:
  - one mostly cell-dominated affine case,
  - one interface-heavy case,
  - one nonlinear residual/tangent case,
  - one adaptivity-relevant case if feasible.
- Add profiling scripts for:
  - CPU time,
  - allocation hot spots,
  - thread scaling regressions.
- Document environment settings used in benchmarks:
  - `JULIA_NUM_THREADS`,
  - `OPENBLAS_NUM_THREADS`,
  - CPU model,
  - Julia version.

### Deliverables

- A benchmark driver in `benchmark/`.
- A short benchmark report in markdown.

### Exit Criteria

- We can answer:
  - where time goes,
  - where allocations happen,
  - when scaling falls off,
  - whether the main limit is assembly, sparse construction, or solve.

## Phase 1: Fix Threaded Assembly Memory Behavior

### Objective

Remove obvious shared-memory scaling blockers from the current assembly path.

### Current Issues

- Each `_ThreadScratch` holds a full-length `global_rhs`.
- Threaded assembly builds large COO triplet arrays and merges them globally.
- Final `sparse(rows, cols, values, ...)` rebuilds structure every time.

### Tasks

- Measure the memory footprint of `_ThreadScratch` as a function of:
  - thread count,
  - global dof count,
  - max local dof count.
- Redesign thread-local RHS accumulation so it does not require one full `ndofs` vector per worker.
- Evaluate alternatives such as:
  - row-blocked thread ownership,
  - sparse thread-local maps for RHS accumulation,
  - segmented reduction buffers,
  - separate strategies for affine assembly versus residual assembly.
- Reduce temporary triplet growth where possible.
- Keep the current semantics intact while making memory costs explicit and bounded.

### Deliverables

- Refactored `_ThreadScratch` and assembly accumulation path.
- Benchmark evidence that memory growth with thread count is materially reduced.

### Exit Criteria

- Thread count no longer multiplies memory usage by `O(ndofs)` in the RHS path.
- Assembly speedup improves on medium and large problems.

## Phase 2: Introduce Symbolic + Numeric Assembly

### Objective

Stop rebuilding the sparse matrix graph on every assembly.

### Why This Matters

This is likely the single biggest structural improvement for shared-memory performance.

### Tasks

- Extend `AssemblyPlan` with precomputed structural assembly data.
- Add a symbolic assembly phase that computes:
  - row/column pattern,
  - stable slot locations for each local contribution,
  - any required reduced-system structural metadata.
- Add a numeric assembly phase that:
  - zeroes numeric storage,
  - scatters directly into preallocated matrix/value buffers.
- Preserve support for:
  - Dirichlet elimination,
  - static condensation,
  - mean constraints,
  - interfaces and embedded surfaces.
- Decide whether residual and tangent should share structural metadata with affine assembly where possible.
- Ensure repeated nonlinear tangents can reuse structure safely.

### Deliverables

- A new symbolic/numeric assembly implementation.
- Benchmarks comparing:
  - old COO path,
  - new symbolic+numeric path.

### Exit Criteria

- Repeated assemblies no longer call the equivalent of full sparse graph reconstruction.
- Tangent assembly for nonlinear solves materially improves.

## Phase 3: Batch And Specialize Local Kernels

### Objective

Improve cache locality and reduce per-item overhead in local integration kernels.

### Why This Matters

`Grico` is flexible and `hp`-adaptive, but that flexibility currently means heterogeneous work items and pointer-heavy local data. Shared-memory performance improves when similar work is processed together.

### Tasks

- Classify cells/faces/interfaces/surfaces into buckets by local signature:
  - dimension,
  - field layout signature,
  - polynomial degree signature,
  - quadrature signature,
  - operator signature if needed.
- Add benchmark variants to compare:
  - original traversal order,
  - bucketed traversal order.
- Reduce pointer chasing in hot paths by flattening or compacting the most frequently accessed local data.
- Audit `CellValues`, `FaceValues`, `InterfaceValues`, and `_FieldValues` for:
  - heap allocations,
  - poor access locality,
  - repeated indirect indexing.
- Consider specialized kernels for common cases:
  - scalar CG,
  - scalar DG,
  - uniform degree blocks,
  - low-order common cases.

### Deliverables

- Signature bucketing infrastructure.
- At least one specialized hot path for a common workload.

### Exit Criteria

- Benchmark evidence shows reduced per-cell overhead.
- Hot loops become more regular and easier to vectorize.

## Phase 4: Improve Thread Scheduling

### Objective

Replace one-size-fits-all atomic chunk claiming with scheduling better suited to FEM workloads.

### Current Issue

The current atomic next-item scheduler is robust but may cause unnecessary overhead, poor locality, and weak NUMA behavior for regular workloads.

### Tasks

- Compare scheduler strategies for:
  - uniform cell work,
  - mildly irregular work,
  - strongly irregular `hp` work.
- Implement and benchmark at least:
  - current dynamic atomic chunking,
  - static partitioning,
  - weighted static partitioning based on estimated local cost,
  - hybrid scheduling for irregular tails.
- Add a lightweight cost model for local work, for example using:
  - local dof count,
  - quadrature point count,
  - operator count,
  - estimated face/interface complexity.
- Select the default scheduler by workload category, not globally.

### Deliverables

- Scheduler abstraction with at least two usable policies.
- Benchmarks showing where each policy wins.

### Exit Criteria

- Scheduler overhead is no longer a dominant factor in regular threaded workloads.
- Large runs show more stable scaling across thread counts.

## Phase 5: SIMD And Node-Local Kernel Optimization

### Objective

Make hot local kernels friendlier to vectorization and low-level optimization.

### Lessons From Trixi

- Flat storage helps.
- Explicit loop kernels often beat tiny BLAS calls.
- Threading backend and SIMD choices should be controlled explicitly.

### Tasks

- Identify the hottest local kernels from profiling.
- Rewrite the worst hot loops to improve:
  - contiguous access,
  - predictable loop structure,
  - reduced branching in inner loops.
- Benchmark explicit loop kernels against any BLAS-based micro-kernels.
- Evaluate targeted use of:
  - `LoopVectorization`,
  - `Polyester`,
  - `Octavian`,
  - or plain optimized Julia loops.
- Keep these optimizations behind small, well-isolated internal kernels.
- Avoid broad dependency additions unless benchmark data justifies them.

### Deliverables

- Optimized kernel implementations for the worst local hotspots.
- Clear benchmark evidence for each new dependency or optimization path.

### Exit Criteria

- The top local kernels show improved throughput in microbenchmarks and end-to-end assembly benchmarks.

## Phase 6: Shared-Memory Algebra Backend Abstraction

### Objective

Decouple the FEM assembly logic from one specific global sparse matrix backend.

### Why This Matters

This is the bridge between "fast threaded serial" and future distributed-memory support.

### Tasks

- Introduce internal algebra/backend interfaces for:
  - vector allocation,
  - symbolic matrix allocation,
  - numeric fill,
  - row/column ownership assumptions.
- Keep the first backend simple:
  - one-process shared-memory sparse backend compatible with current solve paths.
- Ensure `AssemblyPlan` and assembly code depend on backend capabilities rather than directly on `SparseMatrixCSC` construction.
- Keep solve compatibility with the current direct and Krylov paths during transition.
- Document which backend assumptions remain serial-only.

### Deliverables

- Internal backend interface used by assembly.
- Default shared-memory backend implementation.

### Exit Criteria

- Assembly no longer hardcodes one sparse construction strategy end-to-end.
- The new abstraction is sufficient to support a future partitioned backend.

## Phase 7: Solver-Path Cleanup For Large Shared-Memory Runs

### Objective

Ensure solver behavior is appropriate once assembly becomes cheaper and larger one-node runs become practical.

### Tasks

- Re-benchmark the default direct and Krylov paths after Phases 1-6.
- Reassess heuristics such as:
  - direct versus iterative cutoff,
  - AMG/ILU/Additive Schwarz thresholds,
  - reorderings and factor reuse.
- Measure whether preconditioner build time becomes the next dominant bottleneck.
- Add benchmark cases that separate:
  - assembly only,
  - preconditioner setup,
  - solve iterations,
  - total time to solution.
- If beneficial, add explicit caching/reuse paths for repeated solves with unchanged structure.

### Deliverables

- Updated solver heuristics.
- Benchmark report for total time-to-solution on one node.

### Exit Criteria

- The default solve path remains sensible after assembly improvements.
- Time-to-solution improves, not just raw assembly time.

## Cross-Cutting Work Items

These run throughout all phases.

### Testing

- Add tests for any new symbolic/numeric assembly path.
- Add stress tests for repeated assembly.
- Add correctness checks comparing:
  - old and new assembly paths,
  - threaded versus single-threaded results.
- Add tests for deterministic behavior where required.

### Benchmark Hygiene

- Version benchmark inputs.
- Save benchmark outputs with machine and thread metadata.
- Keep at least one stable benchmark suite that can be used in CI or manual regression checks.

### Documentation

- Update developer docs after each architectural phase.
- Record key design decisions and rejected alternatives.

## Suggested Execution Order

1. Phase 0: benchmark baseline.
2. Phase 1: reduce threaded memory blowup.
3. Phase 2: symbolic + numeric assembly.
4. Phase 4: improve scheduling once the assembly backend is cleaner.
5. Phase 3: bucket and specialize local kernels.
6. Phase 5: SIMD and node-local hot-loop tuning.
7. Phase 6: algebra backend abstraction.
8. Phase 7: solver-path cleanup.

## Suggested Acceptance Gates

Do not start distributed-memory work until all of the following are true:

- `assemble` shows meaningful speedup on one node.
- repeated assembly avoids sparse graph rebuilds,
- threaded memory growth is under control,
- benchmark data identifies the remaining node-local bottlenecks,
- assembly logic no longer depends directly on one monolithic sparse construction path.

## Future Distributed-Memory Follow-Up

Once this roadmap is complete, the next roadmap should focus on:

- partitioned leaf ownership,
- ghost cells and ghost dofs,
- distributed vectors and sparse matrices,
- rank-local assembly plus ghost synchronization,
- MPI-aware adaptivity and repartitioning,
- external scalable solvers such as PETSc/HYPRE.

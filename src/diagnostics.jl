# Lightweight runtime diagnostics for matrix-free operators and hpMG.
#
# These routines are deliberately observational: they compile or apply existing
# plans with deterministic sample vectors and report timings plus structural
# metadata. They do not change solver policy, allocate persistent caches, or
# introduce a second instrumentation path inside hot kernels.

"""
    operator_diagnostics(plan; repetitions=3)

Measure basic matrix-free operator application diagnostics for a compiled affine
`plan`.

The returned named tuple records structural counts, full-layout `apply!` time,
and reduced-operator application time using deterministic sample vectors and
reusable workspaces. The function is intended for quick regression checks and
benchmark scripts; for authoritative performance numbers, run it on the target
machine with an appropriate thread count and with enough repetitions to smooth
out noise.
"""
function operator_diagnostics(plan::AssemblyPlan{D,T};
                              repetitions::Integer=3) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  checked_repetitions = _checked_positive(repetitions, "repetitions")
  full_input = _diagnostic_sample_vector(T, dof_count(plan))
  full_output = zeros(T, dof_count(plan))
  reduced_input = _diagnostic_sample_vector(T, reduced_dof_count(plan))
  reduced_output = zeros(T, reduced_dof_count(plan))
  full_workspace = OperatorWorkspace(plan)
  reduced_workspace = _ReducedOperatorWorkspace(plan)
  apply!(full_output, plan, full_input, full_workspace)
  _reduced_apply!(reduced_output, plan, reduced_input, reduced_workspace)
  GC.gc()
  full_timing = @timed begin
    for _ in 1:checked_repetitions
      apply!(full_output, plan, full_input, full_workspace)
    end
  end
  GC.gc()
  reduced_timing = @timed begin
    for _ in 1:checked_repetitions
      _reduced_apply!(reduced_output, plan, reduced_input, reduced_workspace)
    end
  end

  return (; dimension=D, dofs=dof_count(plan), reduced_dofs=reduced_dof_count(plan),
          cells=length(plan.integration.cells),
          boundary_faces=length(plan.integration.boundary_faces),
          interfaces=length(plan.integration.interfaces),
          embedded_surfaces=length(plan.integration.embedded_surfaces),
          cell_operators=length(plan.cell_operators),
          boundary_operators=length(plan.boundary_operators),
          interface_operators=length(plan.interface_operators),
          surface_operators=length(plan.surface_operators), repetitions=checked_repetitions,
          apply_seconds_per_call=Float64(full_timing.time) / checked_repetitions,
          apply_bytes_per_call=Float64(full_timing.bytes) / checked_repetitions,
          reduced_apply_seconds_per_call=Float64(reduced_timing.time) / checked_repetitions,
          reduced_apply_bytes_per_call=Float64(reduced_timing.bytes) / checked_repetitions)
end

"""
    multigrid_diagnostics(problem; preconditioner=GeometricMultigridPreconditioner(), repetitions=3)

Compile hp geometric multigrid for `problem` and measure V-cycle diagnostics.

The report includes hierarchy sizes, the resolved smoother, the selected coarse
solver kind, elapsed compile cost for this call, and average
preconditioner-application time. Compile timing may include first-use Julia
compilation latency; use an external benchmark harness for authoritative
steady-state setup timings.
"""
function multigrid_diagnostics(problem::AffineProblem;
                               preconditioner::GeometricMultigridPreconditioner=GeometricMultigridPreconditioner(),
                               repetitions::Integer=3)
  checked_repetitions = _checked_positive(repetitions, "repetitions")
  hierarchy_ref = Ref{Any}(nothing)
  GC.gc()
  compile_timing = @timed begin
    hierarchy_ref[] = _compile_geometric_multigrid(problem, preconditioner)
  end
  hierarchy = hierarchy_ref[]
  finest = hierarchy.levels[end]
  T = eltype(finest.residual)
  rhs_data = _diagnostic_sample_vector(T, reduced_dof_count(finest.plan))
  result = zeros(T, length(rhs_data))
  _apply_preconditioner!(result, hierarchy, rhs_data)
  GC.gc()
  vcycle_timing = @timed begin
    for _ in 1:checked_repetitions
      _apply_preconditioner!(result, hierarchy, rhs_data)
    end
  end

  level_dofs = Tuple(reduced_dof_count(level.plan) for level in hierarchy.levels)
  level_leaves = Tuple(length(level.plan.integration.cells) for level in hierarchy.levels)
  return (; levels=length(hierarchy.levels), level_dofs, level_leaves, smoother=hierarchy.smoother,
          smoother_restart=hierarchy.smoother_restart,
          coarse_solver=_multigrid_coarse_solver_kind(hierarchy.coarse_solver),
          repetitions=checked_repetitions, compile_seconds=Float64(compile_timing.time),
          compile_bytes=Int(compile_timing.bytes),
          vcycle_seconds_per_call=Float64(vcycle_timing.time) / checked_repetitions,
          vcycle_bytes_per_call=Float64(vcycle_timing.bytes) / checked_repetitions)
end

_multigrid_coarse_solver_kind(::_DenseCholeskyCoarseSolver) = :dense_cholesky
_multigrid_coarse_solver_kind(::_DenseLUCoarseSolver) = :dense_lu
_multigrid_coarse_solver_kind(::_KrylovCoarseSolver) = :krylov

function _diagnostic_sample_vector(::Type{T}, count::Int) where {T<:AbstractFloat}
  values = Vector{T}(undef, count)
  first_frequency = T(19) / T(1000)
  second_frequency = T(43) / T(1000)
  cosine_weight = T(1) / T(10)

  @inbounds for index in 1:count
    point = T(index)
    values[index] = sin(first_frequency * point) + cosine_weight * cos(second_frequency * point)
  end

  return values
end

# Geometric multigrid for matrix-free affine solves.
#
# The production hierarchy is a single hp path: first remove excess polynomial
# order on the fixed fine mesh, then coarsen the dyadic tree one admissible
# frontier at a time. Each level is rediscretized from the original problem on
# adapted fields that preserve field identity. Transfers are compiled once as
# sparse reduced-space prolongations and restriction is their algebraic
# transpose, so constraints remain part of the reduced operator contract.
#
# The file is organized around the pieces of one V-cycle: public policy
# validation, level and transfer construction, local modal transfer kernels,
# smoothers, coarse solvers, and finally the recursive application routine. The
# important architectural rule is that the hierarchy never approximates coarse
# operators by algebraic `RAP` products of the fine operator. Every level is a
# rediscretization of the same problem description on its own adapted space,
# which keeps geometric information and operator callbacks available at coarse
# levels.

const _DEFAULT_COARSE_DIRECT_DOF_LIMIT = 512

"""
    GeometricMultigridPreconditioner(; kwargs...)

Matrix-free geometric multigrid preconditioner policy for affine problems.

The hierarchy is compiled from an `AffineProblem`, not from an already compiled
`AssemblyPlan`, so every level can rediscretize the same operators on adapted
fields. Use it as the preconditioner of [`CGSolver`](@ref) for SPD systems or
[`FGMRESSolver`](@ref) for nonsymmetric, indefinite, or otherwise general
systems.

The implementation builds one configurable hp hierarchy on a shared `HpSpace`;
unsupported coupled-space or non-nested cases throw explicit `ArgumentError`s
when requested explicitly. [`AutoLinearSolver`](@ref) chooses this
preconditioner for supported SPD affine problems and otherwise falls back to a
generic preconditioner unless GMG is requested explicitly.

`smoother=:auto` uses damped Jacobi for SPD operators and a fixed small GMRES
smoothing cycle for nonsymmetric or indefinite operators. The GMRES smoother is
more expensive per V-cycle, but it damps advective and component-coupled error
modes that diagonal smoothing often leaves untouched. When GMRES smoothing is
selected, `smoother_restart` must cover the larger pre/post smoothing count.

`coarse_direct_dof_limit` controls the base-level solve: reduced coarse systems
at or below this size are assembled from local operator matrices and solved
with dense Cholesky for declared SPD operators or dense LU for general
operators. The default is sized for modest scalar dense kernels; larger systems
use the configured coarse Krylov tolerance policy unless the caller explicitly
opts into a larger dense coarse solve.
"""
struct GeometricMultigridPreconditioner <: AbstractPreconditioner
  min_degree::Int
  max_levels::Int
  p_sequence::Symbol
  pre_smoothing_steps::Int
  post_smoothing_steps::Int
  smoother_damping::Float64
  smoother::Symbol
  smoother_restart::Int
  coarse_relative_tolerance::Float64
  coarse_absolute_tolerance::Float64
  coarse_maxiter::Int
  coarse_direct_dof_limit::Int
end

function GeometricMultigridPreconditioner(; min_degree::Integer=1, max_levels::Integer=16,
                                          p_sequence::Symbol=:bisect,
                                          pre_smoothing_steps::Integer=2,
                                          post_smoothing_steps::Integer=2,
                                          smoother_damping::Real=0.8, smoother::Symbol=:auto,
                                          smoother_restart::Integer=8,
                                          coarse_relative_tolerance::Real=1.0e-12,
                                          coarse_absolute_tolerance::Real=0.0,
                                          coarse_maxiter::Integer=10_000,
                                          coarse_direct_dof_limit::Integer=_DEFAULT_COARSE_DIRECT_DOF_LIMIT)
  checked_min_degree = _checked_nonnegative(min_degree, "min_degree")
  checked_max_levels = _checked_positive(max_levels, "max_levels")
  p_sequence in (:bisect, :decrease_by_one, :go_to_one) ||
    throw(ArgumentError("p_sequence must be :bisect, :decrease_by_one, or :go_to_one"))
  checked_pre = _checked_nonnegative(pre_smoothing_steps, "pre_smoothing_steps")
  checked_post = _checked_nonnegative(post_smoothing_steps, "post_smoothing_steps")
  damping = Float64(_checked_solver_positive(smoother_damping, Float64, "smoother_damping"))
  smoother in (:auto, :jacobi, :gmres) ||
    throw(ArgumentError("smoother must be :auto, :jacobi, or :gmres"))
  checked_smoother_restart = _checked_positive(smoother_restart, "smoother_restart")
  coarse_rtol = Float64(_checked_solver_tolerance(coarse_relative_tolerance, Float64,
                                                  "coarse_relative_tolerance"))
  coarse_atol = Float64(_checked_solver_tolerance(coarse_absolute_tolerance, Float64,
                                                  "coarse_absolute_tolerance"))
  checked_coarse_maxiter = _checked_positive(coarse_maxiter, "coarse_maxiter")
  checked_coarse_direct = _checked_nonnegative(coarse_direct_dof_limit, "coarse_direct_dof_limit")
  return GeometricMultigridPreconditioner(checked_min_degree, checked_max_levels, p_sequence,
                                          checked_pre, checked_post, damping, smoother,
                                          checked_smoother_restart, coarse_rtol, coarse_atol,
                                          checked_coarse_maxiter, checked_coarse_direct)
end

# One rediscretized level of the hp hierarchy. The operator and Jacobi object
# are compiled for reduced-space vectors; the residual, operator-value, and
# correction arrays are persistent V-cycle work buffers owned by the level.
struct _MultigridLevel{T<:AbstractFloat,P<:AssemblyPlan,W<:_ReducedOperatorWorkspace{T},
                       O<:_ReducedAffineOperator,J<:_CompiledPreconditioner{T}}
  plan::P
  workspace::W
  operator::O
  jacobi::J
  residual::Vector{T}
  operator_values::Vector{T}
  correction::Vector{T}
end

# Sparse reduced-space prolongation from one coarse level to the next finer
# level. Restriction in the V-cycle is the algebraic transpose of this map, so
# the reduced constraints remain part of the transfer definition.
struct _ReducedTransfer{T<:AbstractFloat,CP<:AssemblyPlan,FP<:AssemblyPlan}
  coarse_plan::CP
  fine_plan::FP
  coarse_indices::Vector{Int}
  fine_indices::Vector{Int}
  coefficients::Vector{T}
end

# Leaf-local tensor-mode transfer contribution before global constraint
# elimination. These entries express one coarse modal basis function exactly in
# the fine modal basis on the child leaf.
struct _LocalTransferEntry{D,T<:AbstractFloat}
  coarse_leaf::Int
  fine_leaf::Int
  coarse_mode::NTuple{D,Int}
  fine_mode::NTuple{D,Int}
  coefficient::T
end

# One admissible h-coarsening operation. A candidate is emitted only when both
# children of a dyadic split are active leaves, so the hierarchy coarsens one
# nested frontier without skipping geometric levels.
struct _MultigridHCoarseningCandidate{D}
  cell::Int
  axis::Int
  children::NTuple{_MIDPOINT_CHILD_COUNT,Int}
  target_degrees::NTuple{D,Int}
end

# Dense direct coarse solves are intentionally preferred for moderate coarse
# systems. The coarse level is the only place where exact inversion is expected;
# using a direct factorization here keeps V-cycles deterministic and avoids a
# fragile nested Krylov solve for common root-grid sizes.
struct _DenseCholeskyCoarseSolver{T<:AbstractFloat}
  factor::Matrix{T}
end

struct _DenseLUCoarseSolver{T<:AbstractFloat}
  factor::Matrix{T}
  pivots::Vector{Int}
end

# Large coarse systems keep a matrix-free Krylov base solve rather than
# assembling an unbounded dense factorization. This is a bounded-memory base
# solve with explicit tolerances, so an inadequate coarse solve fails clearly
# instead of silently returning a poor V-cycle correction.
struct _KrylovCoarseSolver end

# Workspace for one fixed-cycle right-preconditioned GMRES smoother. Columns of
# `krylov` store the Arnoldi basis vᵢ, columns of `preconditioned` store
# zᵢ = M⁻¹vᵢ, and `hessenberg` plus the Givens arrays represent the small
# least-squares problem solved after the prescribed number of smoothing steps.
struct _GmresSmootherWorkspace{T<:AbstractFloat}
  krylov::Matrix{T}
  preconditioned::Matrix{T}
  hessenberg::Matrix{T}
  least_squares_rhs::Vector{T}
  givens_cosine::Vector{T}
  givens_sine::Vector{T}
  coefficients::Vector{T}
end

# Fully compiled hpMG hierarchy. The object owns all level plans, transfers,
# smoothers, coarse solver state, and reusable inter-level right-hand side and
# correction arrays needed by `_apply_preconditioner!`.
struct _CompiledGeometricMultigridPreconditioner{T<:AbstractFloat,CS,OC<:_OperatorClass} <:
       _CompiledPreconditioner{T}
  levels::Vector{_MultigridLevel{T}}
  transfers::Vector{_ReducedTransfer{T}}
  coarse_solver::CS
  operator_class::OC
  level_rhs::Vector{Vector{T}}
  level_solution::Vector{Vector{T}}
  pre_smoothing_steps::Int
  post_smoothing_steps::Int
  smoother_damping::T
  smoother::Symbol
  smoother_restart::Int
  gmres_smoothers::Vector{_GmresSmootherWorkspace{T}}
  coarse_relative_tolerance::T
  coarse_absolute_tolerance::T
  coarse_maxiter::Int
end

function _solve_affine_problem(problem::AffineProblem, ::AutoLinearSolver; kwargs...)
  preconditioner = _default_affine_preconditioner(problem)
  solver = _is_spd_operator_class(operator_class(problem)) ? CGSolver(; preconditioner) :
           FGMRESSolver(; preconditioner)
  return _solve_affine_problem(problem, solver; kwargs...)
end

function _solve_affine_problem(problem::AffineProblem,
                               solver::CGSolver{<:GeometricMultigridPreconditioner};
                               relative_tolerance=nothing, absolute_tolerance=0.0, maxiter=nothing,
                               initial_solution=nothing)
  _check_cg_geometric_multigrid_policy(problem, solver.preconditioner)
  hierarchy = _compile_geometric_multigrid(problem, solver.preconditioner)
  finest = hierarchy.levels[end]
  T = eltype(finest.residual)
  rtol = relative_tolerance === nothing ? sqrt(eps(T)) :
         _checked_solver_tolerance(relative_tolerance, T, "relative_tolerance")
  atol = _checked_solver_tolerance(absolute_tolerance, T, "absolute_tolerance")
  rhs = zeros(T, reduced_dof_count(finest.plan))
  _reduced_rhs!(rhs, finest.plan, finest.workspace)
  outer_maxiter = maxiter === nothing ? max(1_000, 2 * length(rhs)) :
                  _checked_positive(maxiter, "maxiter")
  reduced_values = _cg_solve(finest.operator, rhs, hierarchy; relative_tolerance=rtol,
                             absolute_tolerance=atol, maxiter=outer_maxiter,
                             initial_solution=initial_solution)
  return _state_from_reduced_result(finest.plan, reduced_values)
end

function _solve_affine_problem(problem::AffineProblem,
                               solver::FGMRESSolver{<:GeometricMultigridPreconditioner};
                               relative_tolerance=nothing, absolute_tolerance=0.0, maxiter=nothing,
                               initial_solution=nothing)
  hierarchy = _compile_geometric_multigrid(problem, solver.preconditioner)
  finest = hierarchy.levels[end]
  T = eltype(finest.residual)
  rtol = relative_tolerance === nothing ? sqrt(eps(T)) :
         _checked_solver_tolerance(relative_tolerance, T, "relative_tolerance")
  atol = _checked_solver_tolerance(absolute_tolerance, T, "absolute_tolerance")
  rhs = zeros(T, reduced_dof_count(finest.plan))
  _reduced_rhs!(rhs, finest.plan, finest.workspace)
  outer_maxiter = maxiter === nothing ? max(1_000, 2 * length(rhs)) :
                  _checked_positive(maxiter, "maxiter")
  reduced_values = _fgmres_solve(finest.operator, rhs, hierarchy; restart=solver.restart,
                                 relative_tolerance=rtol, absolute_tolerance=atol,
                                 maxiter=outer_maxiter, initial_solution=initial_solution)
  return _state_from_reduced_result(finest.plan, reduced_values)
end

function _check_cg_geometric_multigrid_policy(problem::AffineProblem,
                                              preconditioner::GeometricMultigridPreconditioner)
  _is_spd_operator_class(operator_class(problem)) ||
    throw(ArgumentError("CG with GeometricMultigridPreconditioner requires an SPD affine problem; use FGMRESSolver or declare the operator SPD if that is mathematically valid"))
  preconditioner.pre_smoothing_steps == preconditioner.post_smoothing_steps ||
    throw(ArgumentError("CG with GeometricMultigridPreconditioner requires equal pre- and post-smoothing steps so the V-cycle remains symmetric"))
  smoother = _resolved_multigrid_smoother(preconditioner.smoother, operator_class(problem))
  smoother === :jacobi ||
    throw(ArgumentError("CG with GeometricMultigridPreconditioner requires the Jacobi smoother; use FGMRESSolver for nonsymmetric smoother policies"))
  return nothing
end

function _default_affine_preconditioner(problem::AffineProblem)
  candidate = GeometricMultigridPreconditioner()
  return _is_spd_operator_class(operator_class(problem)) &&
         _supports_geometric_multigrid(problem, candidate) ? candidate : JacobiPreconditioner()
end

function _supports_geometric_multigrid(problem::AffineProblem,
                                       preconditioner::GeometricMultigridPreconditioner)
  fields_tuple = Tuple(_problem_data(problem).fields)
  space = _maybe_single_multigrid_space(fields_tuple)
  space === nothing && return false
  return _has_multigrid_coarsening_level(space, preconditioner)
end

function _maybe_single_multigrid_space(fields_tuple::Tuple)
  isempty(fields_tuple) && return nothing
  space = field_space(fields_tuple[1])

  for field in fields_tuple
    field_space(field) === space || return nothing
  end

  return space
end

function _has_multigrid_coarsening_level(space::HpSpace, solver::GeometricMultigridPreconditioner)
  solver.max_levels > 1 || return false
  return _has_p_coarsening_level(space, solver) || _has_h_coarsening_level(space)
end

function _has_p_coarsening_level(space::HpSpace, solver::GeometricMultigridPreconditioner)
  active_degrees = space.degree_policy.data
  isempty(active_degrees) && return false
  continuity = continuity_policy(space)

  for degrees in active_degrees
    next_degrees = _coarsened_degree_tuple(degrees, continuity, solver.min_degree,
                                           solver.p_sequence)
    next_degrees != degrees && return true
  end

  return false
end

_has_h_coarsening_level(space::HpSpace) = !isempty(_multigrid_h_coarsening_candidates(space))

# Hierarchy construction crosses several deeply parameterized plan and operator
# types. Keep this setup path behind explicit inference barriers so first-use
# latency does not dominate every GMG solve while the V-cycle data remain typed
# by scalar precision.
Base.@nospecializeinfer function _compile_geometric_multigrid(problem::AffineProblem,
                                                              solver::GeometricMultigridPreconditioner)
  data = Base.inferencebarrier(_problem_data(problem))
  fine_fields = Base.inferencebarrier(Tuple(data.fields))
  fine_space = Base.inferencebarrier(_single_multigrid_space(fine_fields))
  _has_multigrid_coarsening_level(fine_space, solver) ||
    throw(ArgumentError("GeometricMultigridPreconditioner requires at least one admissible h or p coarsening level; use AutoLinearSolver or an explicit non-GMG preconditioner for this problem"))
  T = eltype(origin(field_space(fine_fields[1])))
  spaces = Base.inferencebarrier(_hp_multigrid_spaces(fine_space, solver))
  level_fields = Base.inferencebarrier(_multigrid_level_fields(fine_fields, spaces))
  level_problems = Base.inferencebarrier([_problem_on_fields(problem, fields_tuple)
                                          for fields_tuple in level_fields])
  plans = Base.inferencebarrier(AssemblyPlan[compile(level_problem)
                                             for level_problem in level_problems])
  levels = Base.inferencebarrier(_MultigridLevel{T}[_compile_multigrid_level(plan)
                                                    for plan in plans])
  transfers = Base.inferencebarrier(_ReducedTransfer{T}[_compile_multigrid_transfer(levels[index].plan,
                                                                                    levels[index+1].plan)
                                                        for index in 1:(length(levels)-1)])
  coarse_solver = Base.inferencebarrier(_compile_coarse_solver(levels[1], solver,
                                                               operator_class(problem)))
  smoother = _resolved_multigrid_smoother(solver.smoother, operator_class(problem))
  _check_multigrid_smoother_steps(solver, smoother)
  level_rhs = [zeros(T, reduced_dof_count(levels[index].plan)) for index in 1:(length(levels)-1)]
  level_solution = [zeros(T, reduced_dof_count(levels[index].plan))
                    for index in 1:(length(levels)-1)]
  gmres_smoothers = _compile_multigrid_gmres_smoothers(T, levels, smoother, solver.smoother_restart)
  return _CompiledGeometricMultigridPreconditioner(levels, transfers, coarse_solver,
                                                   operator_class(problem), level_rhs,
                                                   level_solution, solver.pre_smoothing_steps,
                                                   solver.post_smoothing_steps,
                                                   T(solver.smoother_damping), smoother,
                                                   solver.smoother_restart, gmres_smoothers,
                                                   T(solver.coarse_relative_tolerance),
                                                   T(solver.coarse_absolute_tolerance),
                                                   solver.coarse_maxiter)
end

function _resolved_multigrid_smoother(smoother::Symbol, operator_class::_OperatorClass)
  smoother === :auto && return _is_spd_operator_class(operator_class) ? :jacobi : :gmres
  return smoother
end

function _check_multigrid_smoother_steps(solver::GeometricMultigridPreconditioner, smoother::Symbol)
  smoother === :gmres || return nothing
  requested_steps = max(solver.pre_smoothing_steps, solver.post_smoothing_steps)
  requested_steps <= solver.smoother_restart ||
    throw(ArgumentError("GMRES multigrid smoother requires smoother_restart to be at least max(pre_smoothing_steps, post_smoothing_steps)"))
  return nothing
end

function _compile_multigrid_gmres_smoothers(::Type{T}, levels::Vector{_MultigridLevel{T}},
                                            smoother::Symbol, restart::Int) where {T<:AbstractFloat}
  smoother === :gmres || return _GmresSmootherWorkspace{T}[]
  return [_compile_gmres_smoother_workspace(T, reduced_dof_count(levels[index].plan), restart)
          for index in 2:length(levels)]
end

function _compile_gmres_smoother_workspace(::Type{T}, n::Int, restart::Int) where {T<:AbstractFloat}
  count = max(1, min(restart, max(n, 1)))
  return _GmresSmootherWorkspace(zeros(T, n, count + 1), zeros(T, n, count),
                                 zeros(T, count + 1, count), zeros(T, count + 1), zeros(T, count),
                                 zeros(T, count), zeros(T, count))
end

function _compile_multigrid_level(plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  workspace = _ReducedOperatorWorkspace(plan)
  operator = _ReducedAffineOperator(plan, workspace)
  jacobi = _compile_preconditioner(JacobiPreconditioner(), operator)
  n = reduced_dof_count(plan)
  return _MultigridLevel(plan, workspace, operator, jacobi, zeros(T, n), zeros(T, n), zeros(T, n))
end

function _compile_coarse_solver(level::_MultigridLevel{T}, solver::GeometricMultigridPreconditioner,
                                operator_class::_OperatorClass) where {T<:AbstractFloat}
  n = reduced_dof_count(level.plan)
  0 < n <= solver.coarse_direct_dof_limit || return _KrylovCoarseSolver()
  matrix = _assemble_reduced_operator_matrix(level.plan, level.workspace.scratch)
  if _is_spd_operator_class(operator_class)
    _matrix_is_symmetric(matrix) ||
      throw(ArgumentError("declared SPD coarse multigrid matrix is not numerically symmetric"))

    try
      _dense_cholesky_factor!(matrix)
      return _DenseCholeskyCoarseSolver(matrix)
    catch error
      error isa PosDefException || rethrow()
      throw(ArgumentError("declared SPD coarse multigrid matrix is not positive definite"))
    end
  end

  pivots = zeros(Int, n)
  try
    _dense_lu_factor!(matrix, pivots)
    return _DenseLUCoarseSolver(matrix, pivots)
  catch error
    error isa SingularException || rethrow()
    throw(ArgumentError("coarse multigrid matrix is singular; add constraints or use a problem-specific solver policy"))
  end
end

function _matrix_is_symmetric(matrix::AbstractMatrix{T}) where {T<:AbstractFloat}
  n = _require_square_matrix(matrix, "coarse multigrid matrix")
  tolerance = sqrt(eps(T))
  matrix_scale = zero(T)

  for value in matrix
    isfinite(value) || return false
    matrix_scale = max(matrix_scale, abs(value))
  end

  allowed = tolerance * matrix_scale

  for column in 1:n
    for row in (column+1):n
      abs(matrix[row, column] - matrix[column, row]) <= allowed || return false
    end
  end

  return true
end

function _single_multigrid_space(fields_tuple::Tuple)
  isempty(fields_tuple) && throw(ArgumentError("multigrid requires at least one field"))
  space = field_space(fields_tuple[1])

  for field in fields_tuple
    field_space(field) === space ||
      throw(ArgumentError("geometric multigrid currently requires all fields to share one HpSpace; use an explicit solver policy for coupled multi-space systems"))
  end

  return space
end

function _hp_multigrid_spaces(fine_space::HpSpace{D,T},
                              solver::GeometricMultigridPreconditioner) where {D,T<:AbstractFloat}
  fine_to_coarse = HpSpace{D,T}[fine_space]
  current_space = fine_space
  current_degrees = copy(fine_space.degree_policy.data)

  while length(fine_to_coarse) < solver.max_levels
    next_degrees = [_coarsened_degree_tuple(degrees, continuity_policy(current_space),
                                            solver.min_degree, solver.p_sequence)
                    for degrees in current_degrees]
    next_degrees == current_degrees && break
    current_space = _space_with_degrees(current_space, snapshot(current_space).active_leaves,
                                        next_degrees)
    push!(fine_to_coarse, current_space)
    current_degrees = next_degrees
  end

  while length(fine_to_coarse) < solver.max_levels
    next_space = _h_coarsened_space(current_space)
    next_space === nothing && break
    current_space = next_space
    push!(fine_to_coarse, current_space)
  end

  reverse!(fine_to_coarse)
  return fine_to_coarse
end

function _coarsened_degree_tuple(degrees::NTuple{D,Int}, continuity::NTuple{D,Symbol},
                                 min_degree::Int, sequence::Symbol) where {D}
  return ntuple(axis -> begin
                  lower = min(degrees[axis], max(min_degree, continuity[axis] === :cg ? 1 : 0))
                  _coarsened_degree(degrees[axis], lower, sequence)
                end, D)
end

function _coarsened_degree(degree::Int, lower::Int, sequence::Val{:decrease_by_one})
  degree <= lower && return degree
  return max(degree - 1, lower)
end

function _coarsened_degree(degree::Int, lower::Int, sequence::Val{:go_to_one})
  degree <= lower && return degree
  return lower
end

function _coarsened_degree(degree::Int, lower::Int, sequence::Symbol)
  sequence === :bisect && return _coarsened_degree(degree, lower, Val(:bisect))
  sequence === :decrease_by_one && return _coarsened_degree(degree, lower, Val(:decrease_by_one))
  sequence === :go_to_one && return _coarsened_degree(degree, lower, Val(:go_to_one))
  throw(ArgumentError("unsupported p coarsening sequence $sequence"))
end

function _coarsened_degree(degree::Int, lower::Int, ::Val{:bisect})
  degree <= lower && return degree
  return max(div(degree, 2), lower)
end

function _space_with_degrees(source::HpSpace, active, degrees)
  target_snapshot = _snapshot(grid(domain(source)), active)
  return _space_with_snapshot_and_degrees(source, target_snapshot, degrees)
end

function _space_with_snapshot_and_degrees(source::HpSpace, target_snapshot::GridSnapshot, degrees)
  active = target_snapshot.active_leaves
  degree_policy = StoredDegrees(domain(source), active, degrees)
  options = SpaceOptions(basis=basis_family(source), degree=degree_policy,
                         quadrature=source.quadrature_policy, continuity=continuity_policy(source))
  return _compile_snapshot_space(domain(source), target_snapshot, options)
end

function _h_coarsened_space(space::HpSpace{D}) where {D}
  candidates = _multigrid_h_coarsening_candidates(space)
  isempty(candidates) && return nothing

  source_snapshot = snapshot(space)
  mapping = _source_degree_map(space)
  child_to_parent = Dict{Int,Int}()
  emitted_parents = Set{Int}()

  for candidate in candidates
    for child in candidate.children
      haskey(child_to_parent, child) &&
        throw(ArgumentError("child $child appears in multiple multigrid h-coarsening candidates"))
      child_to_parent[child] = candidate.cell
      delete!(mapping, child)
    end

    mapping[candidate.cell] = candidate.target_degrees
  end

  active = Int[]
  sizehint!(active,
            length(source_snapshot.active_leaves) - length(child_to_parent) + length(candidates))

  for leaf in source_snapshot.active_leaves
    parent = get(child_to_parent, leaf, NONE)

    if parent == NONE
      push!(active, leaf)
    elseif !(parent in emitted_parents)
      push!(active, parent)
      push!(emitted_parents, parent)
    end
  end

  target_snapshot = _snapshot(grid(domain(space)), active)
  target_snapshot = _filter_target_snapshot!(domain(space), target_snapshot, mapping)
  active = target_snapshot.active_leaves
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    leaf = active[index]
    haskey(mapping, leaf) || throw(ArgumentError("missing multigrid degree data for leaf $leaf"))
    degrees[index] = mapping[leaf]
  end

  return _space_with_snapshot_and_degrees(space, target_snapshot, degrees)
end

function _multigrid_parent_degrees(space::HpSpace{D},
                                   children::NTuple{_MIDPOINT_CHILD_COUNT,Int}) where {D}
  return ntuple(axis -> minimum(cell_degrees(space, child)[axis] for child in children), D)
end

function _multigrid_h_coarsening_candidates(space::HpSpace{D}) where {D}
  candidates = _MultigridHCoarseningCandidate{D}[]
  space_snapshot = snapshot(space)

  for cell in 1:stored_cell_count(grid(space))
    is_expanded(space_snapshot, cell) || continue
    axis = _snapshot_structural_split_axis(space_snapshot, cell)
    first = _snapshot_first_child(space_snapshot, cell)
    first == NONE && continue
    children = ntuple(offset -> first + offset - 1, _MIDPOINT_CHILD_COUNT)
    all(child -> is_active_leaf(space_snapshot, child) && space_snapshot.leaf_to_index[child] != 0,
        children) || continue
    push!(candidates,
          _MultigridHCoarseningCandidate{D}(cell, axis, children,
                                            _multigrid_parent_degrees(space, children)))
  end

  return candidates
end

function _multigrid_level_fields(fine_fields::Tuple, spaces::Vector)
  finest_index = length(spaces)
  return [level_index == finest_index ? fine_fields :
          ntuple(field_index -> _field_on_space(fine_fields[field_index], spaces[level_index]),
                 length(fine_fields)) for level_index in eachindex(spaces)]
end

_field_on_space(field::ScalarField, space::HpSpace) = _field_with_identity(field, space)

_field_on_space(field::VectorField, space::HpSpace) = _field_with_identity(field, space)

function _problem_on_fields(source::AffineProblem, new_fields::Tuple)
  source_data = _problem_data(source)
  problem = AffineProblem(new_fields...; operator_class=operator_class(source))
  target_data = _problem_data(problem)
  append!(target_data.cell_operators, source_data.cell_operators)
  append!(target_data.boundary_operators, source_data.boundary_operators)
  append!(target_data.interface_operators, source_data.interface_operators)
  append!(target_data.surface_operators, source_data.surface_operators)
  append!(target_data.cell_quadratures, source_data.cell_quadratures)
  append!(target_data.embedded_surfaces, source_data.embedded_surfaces)
  append!(target_data.dirichlet_constraints, source_data.dirichlet_constraints)
  append!(target_data.mean_constraints, source_data.mean_constraints)
  return problem
end

function _compile_multigrid_transfer(coarse_plan::AssemblyPlan{D,T},
                                     fine_plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  coarse_fields = fields(field_layout(coarse_plan))
  fine_fields = fields(field_layout(fine_plan))
  length(coarse_fields) == length(fine_fields) ||
    throw(ArgumentError("multigrid transfer requires matching field counts"))
  coarse_space = _single_multigrid_space(Tuple(coarse_fields))
  fine_space = _single_multigrid_space(Tuple(fine_fields))
  local_entries = _compile_local_transfer_entries(coarse_space, fine_space)
  entries = Dict{Tuple{Int,Int},T}()

  for field_index in eachindex(coarse_fields)
    _append_field_reduced_transfer_entries!(entries, coarse_plan, fine_plan,
                                            coarse_fields[field_index], fine_fields[field_index],
                                            local_entries)
  end

  sorted_entries = sort!(collect(entries); by=entry -> (entry.first[2], entry.first[1]))
  coarse_indices = Int[entry.first[1] for entry in sorted_entries]
  fine_indices = Int[entry.first[2] for entry in sorted_entries]
  coefficients = T[entry.second for entry in sorted_entries]
  return _ReducedTransfer(coarse_plan, fine_plan, coarse_indices, fine_indices, coefficients)
end

function _append_field_reduced_transfer_entries!(entries::Dict{Tuple{Int,Int},T},
                                                 coarse_plan::AssemblyPlan{D,T},
                                                 fine_plan::AssemblyPlan{D,T},
                                                 coarse_field::AbstractField,
                                                 fine_field::AbstractField,
                                                 local_entries::Vector{_LocalTransferEntry{D,T}}) where {D,
                                                                                                         T<:AbstractFloat}
  _field_id(coarse_field) == _field_id(fine_field) ||
    throw(ArgumentError("multigrid transfer requires matching field identities"))
  component_count(coarse_field) == component_count(fine_field) ||
    throw(ArgumentError("multigrid transfer requires matching component counts"))
  coarse_space = field_space(coarse_field)
  fine_space = field_space(fine_field)
  coarse_map = _reduced_map(coarse_plan)
  fine_map = _reduced_map(fine_plan)

  for component in 1:component_count(coarse_field)
    coarse_offset = first(field_component_range(field_layout(coarse_plan), coarse_field, component)) -
                    1
    fine_offset = first(field_component_range(field_layout(fine_plan), fine_field, component)) - 1
    source_expansions = Dict{Tuple{Int,NTuple{D,Int}},Vector{Pair{Int,T}}}()
    target_expansions = Dict{Tuple{Int,NTuple{D,Int}},Dict{Int,T}}()

    for entry in local_entries
      coarse_key = (entry.coarse_leaf, entry.coarse_mode)
      source = get!(source_expansions, coarse_key) do
        _local_mode_reduced_expansion(coarse_map, coarse_space, coarse_offset, entry.coarse_leaf,
                                      entry.coarse_mode)
      end
      target_key = (entry.fine_leaf, entry.fine_mode)
      target = get!(target_expansions, target_key) do
        Dict{Int,T}()
      end

      for pair in source
        _accumulate_transfer_value!(target, pair.first, entry.coefficient * pair.second)
      end
    end

    for pair in target_expansions
      leaf, mode = pair.first
      coordinate = _fine_mode_reduced_coordinate(fine_map, fine_space, fine_offset, leaf, mode)
      coordinate === nothing && continue
      fine_index, inverse_coefficient = coordinate

      for source_pair in pair.second
        coefficient = inverse_coefficient * source_pair.second
        _keep_transfer_coefficient(coefficient) || continue
        _accumulate_transfer_entry!(entries, source_pair.first, fine_index, coefficient)
      end
    end
  end

  return entries
end

function _compile_local_transfer_entries(coarse_space::HpSpace{D,T},
                                         fine_space::HpSpace{D,T}) where {D,T<:AbstractFloat}
  entries = _LocalTransferEntry{D,T}[]
  restriction_cache = Dict{NTuple{4,Int},Matrix{T}}()

  for coarse_leaf in snapshot(coarse_space).active_leaves
    for fine_leaf in _fine_leaves_under_coarse_leaf(coarse_space, fine_space, coarse_leaf)
      matrices = ntuple(axis -> _axis_transfer_matrix!(restriction_cache, coarse_space, fine_space,
                                                       coarse_leaf, fine_leaf, axis), D)

      for coarse_mode in local_modes(coarse_space, coarse_leaf)
        for fine_mode in local_modes(fine_space, fine_leaf)
          coefficient = _tensor_mode_transfer_coefficient(matrices, coarse_mode, fine_mode)
          _keep_transfer_coefficient(coefficient) || continue
          push!(entries,
                _LocalTransferEntry{D,T}(coarse_leaf, fine_leaf, coarse_mode, fine_mode,
                                         coefficient))
        end
      end
    end
  end

  return entries
end

function _keep_transfer_coefficient(coefficient::T) where {T<:AbstractFloat}
  isfinite(coefficient) || throw(ArgumentError("multigrid transfer coefficient must be finite"))
  return !iszero(coefficient)
end

function _fine_leaves_under_coarse_leaf(coarse_space::HpSpace, fine_space::HpSpace,
                                        coarse_leaf::Int)
  coarse_snapshot = snapshot(coarse_space)
  fine_snapshot = snapshot(fine_space)
  coarse_snapshot.leaf_to_index[coarse_leaf] != 0 ||
    throw(ArgumentError("coarse multigrid leaf $coarse_leaf is not active"))

  if coarse_leaf <= length(fine_snapshot.leaf_to_index) &&
     fine_snapshot.leaf_to_index[coarse_leaf] != 0
    return (coarse_leaf,)
  end

  first = _snapshot_first_child(fine_snapshot, coarse_leaf)
  first != NONE ||
    throw(ArgumentError("fine multigrid space does not contain leaf $coarse_leaf or its immediate children"))
  children = ntuple(offset -> first + offset - 1, _MIDPOINT_CHILD_COUNT)
  all(child -> child <= length(fine_snapshot.leaf_to_index) &&
               fine_snapshot.leaf_to_index[child] != 0, children) ||
    throw(ArgumentError("h-multigrid transfer requires adjacent dyadic levels"))
  return children
end

function _axis_transfer_matrix!(cache::Dict{NTuple{4,Int},Matrix{T}}, coarse_space::HpSpace{D,T},
                                fine_space::HpSpace{D,T}, coarse_leaf::Int, fine_leaf::Int,
                                axis::Int) where {D,T<:AbstractFloat}
  source_degree = cell_degrees(coarse_space, coarse_leaf)[axis]
  target_degree = cell_degrees(fine_space, fine_leaf)[axis]
  target_degree >= source_degree ||
    throw(ArgumentError("fine multigrid level must represent every coarse mode exactly"))
  delta, relative = _dyadic_child_position(grid(coarse_space), coarse_leaf, fine_leaf, axis)
  return get!(cache, (source_degree, target_degree, delta, relative)) do
    _affine_restriction_matrix(source_degree, target_degree, delta, relative, T)
  end
end

function _dyadic_child_position(grid_data::CartesianGrid, coarse_leaf::Int, fine_leaf::Int,
                                axis::Int)
  source_level = level(grid_data, coarse_leaf, axis)
  target_level = level(grid_data, fine_leaf, axis)
  delta = target_level - source_level
  delta >= 0 ||
    throw(ArgumentError("fine multigrid leaf $fine_leaf is coarser than coarse leaf $coarse_leaf"))
  source_coord = logical_coordinate(grid_data, coarse_leaf, axis)
  target_coord = logical_coordinate(grid_data, fine_leaf, axis)
  relative_value = Int128(target_coord) - (Int128(source_coord) << delta)
  upper = Int128(1) << delta
  0 <= relative_value <= typemax(Int) ||
    throw(ArgumentError("relative dyadic multigrid offset must be Int-representable"))
  relative_value < upper ||
    throw(ArgumentError("fine multigrid leaf $fine_leaf is not contained in coarse leaf $coarse_leaf"))
  return delta, Int(relative_value)
end

function _tensor_mode_transfer_coefficient(matrices::NTuple{D,Matrix{T}},
                                           coarse_mode::NTuple{D,Int},
                                           fine_mode::NTuple{D,Int}) where {D,T<:AbstractFloat}
  coefficient = one(T)

  for axis in 1:D
    coefficient *= @inbounds matrices[axis][fine_mode[axis]+1, coarse_mode[axis]+1]
  end

  return coefficient
end

function _local_mode_reduced_expansion(map::_ReducedOperatorMap{T}, space::HpSpace{D,T},
                                       field_offset::Int, leaf::Int,
                                       mode::NTuple{D,Int}) where {D,T<:AbstractFloat}
  compiled = _compiled_leaf(space, leaf)
  mode_index = _mode_lookup(compiled, mode)
  mode_index != 0 || throw(ArgumentError("mode $mode is not active on leaf $leaf"))
  expansion = Dict{Int,T}()

  for term_index in _mode_term_range(compiled, mode_index)
    term_coefficient = @inbounds compiled.term_coefficients[term_index]
    scalar_dof = @inbounds compiled.term_indices[term_index]
    full_dof = field_offset + scalar_dof

    @inbounds for pointer in map.row_offsets[full_dof]:(map.row_offsets[full_dof+1]-1)
      _accumulate_transfer_value!(expansion, map.row_indices[pointer],
                                  term_coefficient * map.row_coefficients[pointer])
    end
  end

  return Pair{Int,T}[pair for pair in expansion]
end

function _fine_mode_reduced_coordinate(fine_map::_ReducedOperatorMap{T}, fine_space::HpSpace{D,T},
                                       field_offset::Int, leaf::Int,
                                       mode::NTuple{D,Int}) where {D,T<:AbstractFloat}
  compiled = _compiled_leaf(fine_space, leaf)
  mode_index = _mode_lookup(compiled, mode)
  mode_index != 0 || throw(ArgumentError("mode $mode is not active on leaf $leaf"))
  first_term = @inbounds compiled.term_offsets[mode_index]
  next_term = @inbounds compiled.term_offsets[mode_index+1]
  next_term == first_term + 1 || return nothing
  scalar_dof = @inbounds compiled.term_indices[first_term]
  full_dof = field_offset + scalar_dof
  reduced_index = @inbounds fine_map.reduced_index[full_dof]
  reduced_index == 0 && return nothing
  term_coefficient = @inbounds compiled.term_coefficients[first_term]
  return reduced_index => inv(term_coefficient)
end

function _accumulate_transfer_value!(target::Dict{K,T}, key::K, value::T) where {K,T<:AbstractFloat}
  iszero(value) && return target
  target[key] = get(target, key, zero(T)) + value
  return target
end

function _transfer_values_close(first::T, second::T, ::Type{T}) where {T<:AbstractFloat}
  scale = max(one(T), abs(first), abs(second))
  return abs(first - second) <= 10000 * eps(T) * scale
end

function _accumulate_transfer_entry!(entries::Dict{Tuple{Int,Int},T}, coarse_dof::Int,
                                     fine_dof::Int, coefficient::T) where {T<:AbstractFloat}
  key = (coarse_dof, fine_dof)
  existing = get(entries, key, nothing)

  if existing === nothing
    entries[key] = coefficient
  else
    _transfer_values_close(existing, coefficient, T) ||
      throw(ArgumentError("inconsistent duplicate multigrid transfer entry"))
  end

  return entries
end

function _compile_preconditioner(::GeometricMultigridPreconditioner, ::_ReducedAffineOperator)
  throw(ArgumentError("GeometricMultigridPreconditioner requires the original AffineProblem so every level can rediscretize the operators; call solve(problem; solver=CGSolver(preconditioner=GeometricMultigridPreconditioner())) for SPD systems or solve(problem; solver=FGMRESSolver(preconditioner=GeometricMultigridPreconditioner())) for general systems"))
end

function _apply_preconditioner!(result::AbstractVector{T},
                                preconditioner::_CompiledGeometricMultigridPreconditioner{T},
                                residual::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(result, length(residual), "multigrid result")
  fill!(result, zero(T))
  _vcycle!(preconditioner, length(preconditioner.levels), result, residual)
  return result
end

function _vcycle!(mg::_CompiledGeometricMultigridPreconditioner{T}, level_index::Int,
                  solution::AbstractVector{T}, rhs::AbstractVector{T}) where {T<:AbstractFloat}
  level = mg.levels[level_index]

  if level_index == 1
    return _coarse_solve!(solution, mg.coarse_solver, level, rhs, mg)
  end

  _smooth_multigrid_level!(solution, mg, level_index, rhs, mg.pre_smoothing_steps)
  residual = level.residual
  copyto!(residual, rhs)
  _apply_operator!(level.operator_values, level.operator, solution)
  _axpy!(residual, -one(T), level.operator_values)
  transfer = mg.transfers[level_index-1]
  coarse_rhs = mg.level_rhs[level_index-1]
  _restrict_reduced!(coarse_rhs, transfer, residual)
  coarse_solution = mg.level_solution[level_index-1]
  fill!(coarse_solution, zero(T))
  _vcycle!(mg, level_index - 1, coarse_solution, coarse_rhs)
  _prolongate_reduced_add!(solution, transfer, coarse_solution)
  _smooth_multigrid_level!(solution, mg, level_index, rhs, mg.post_smoothing_steps)
  return solution
end

function _coarse_solve!(solution::AbstractVector{T}, solver::_DenseCholeskyCoarseSolver{T},
                        level::_MultigridLevel{T}, rhs::AbstractVector{T},
                        mg::_CompiledGeometricMultigridPreconditioner{T}) where {T<:AbstractFloat}
  _require_length(solution, length(rhs), "coarse multigrid solution")
  copyto!(solution, rhs)
  _dense_cholesky_solve!(solver.factor, solution)
  return solution
end

function _coarse_solve!(solution::AbstractVector{T}, solver::_DenseLUCoarseSolver{T},
                        level::_MultigridLevel{T}, rhs::AbstractVector{T},
                        mg::_CompiledGeometricMultigridPreconditioner{T}) where {T<:AbstractFloat}
  _require_length(solution, length(rhs), "coarse multigrid solution")
  copyto!(solution, rhs)
  _dense_lu_solve!(solver.factor, solver.pivots, solution)
  return solution
end

function _coarse_solve!(solution::AbstractVector{T}, ::_KrylovCoarseSolver,
                        level::_MultigridLevel{T}, rhs::AbstractVector{T},
                        mg::_CompiledGeometricMultigridPreconditioner{T}) where {T<:AbstractFloat}
  coarse_solution = if _is_spd_operator_class(mg.operator_class)
    _cg_solve(level.operator, rhs, level.jacobi; relative_tolerance=mg.coarse_relative_tolerance,
              absolute_tolerance=mg.coarse_absolute_tolerance, maxiter=mg.coarse_maxiter,
              initial_solution=nothing)
  else
    _fgmres_solve(level.operator, rhs, level.jacobi; restart=max(1, min(30, length(rhs))),
                  relative_tolerance=mg.coarse_relative_tolerance,
                  absolute_tolerance=mg.coarse_absolute_tolerance, maxiter=mg.coarse_maxiter,
                  initial_solution=nothing)
  end
  copyto!(solution, coarse_solution)
  return solution
end

function _smooth_multigrid_level!(solution::AbstractVector{T},
                                  mg::_CompiledGeometricMultigridPreconditioner{T},
                                  level_index::Int, rhs::AbstractVector{T},
                                  steps::Int) where {T<:AbstractFloat}
  steps == 0 && return solution
  level = mg.levels[level_index]

  if mg.smoother === :jacobi
    return _smooth_jacobi!(solution, level, rhs, steps, mg.smoother_damping)
  elseif mg.smoother === :gmres
    return _smooth_gmres!(solution, level, rhs, steps, mg.smoother_damping,
                          mg.gmres_smoothers[level_index-1])
  end

  throw(ArgumentError("unsupported multigrid smoother $(mg.smoother)"))
end

function _smooth_jacobi!(solution::AbstractVector{T}, level::_MultigridLevel{T},
                         rhs::AbstractVector{T}, steps::Int, damping::T) where {T<:AbstractFloat}
  steps == 0 && return solution

  for _ in 1:steps
    copyto!(level.residual, rhs)
    _apply_operator!(level.operator_values, level.operator, solution)
    _axpy!(level.residual, -one(T), level.operator_values)
    _apply_preconditioner!(level.correction, level.jacobi, level.residual)
    _axpy!(solution, damping, level.correction)
  end

  return solution
end

# Fixed-cycle right-preconditioned GMRES smoother. This is used for general
# operators where diagonal Richardson smoothing is often ineffective. The
# routine deliberately performs a prescribed number of Arnoldi steps instead of
# solving to a tolerance; a smoother should remove high-frequency error modes
# at bounded cost, not become an inner Krylov solve whose work varies by level.
function _smooth_gmres!(solution::AbstractVector{T}, level::_MultigridLevel{T},
                        rhs::AbstractVector{T}, steps::Int, damping::T,
                        workspace::_GmresSmootherWorkspace{T}) where {T<:AbstractFloat}
  steps == 0 && return solution
  _require_length(solution, length(rhs), "multigrid smoother solution")
  copyto!(level.residual, rhs)
  _apply_operator!(level.operator_values, level.operator, solution)
  _axpy!(level.residual, -one(T), level.operator_values)
  residual_norm = sqrt(_dot_self(level.residual))
  residual_norm == zero(T) && return solution
  count = min(steps, size(workspace.preconditioned, 2), length(rhs))
  count == 0 && return solution
  fill!(workspace.krylov, zero(T))
  fill!(workspace.preconditioned, zero(T))
  fill!(workspace.hessenberg, zero(T))
  fill!(workspace.least_squares_rhs, zero(T))
  fill!(workspace.givens_cosine, zero(T))
  fill!(workspace.givens_sine, zero(T))
  fill!(workspace.coefficients, zero(T))

  @inbounds for row in eachindex(level.residual)
    workspace.krylov[row, 1] = level.residual[row] / residual_norm
  end

  workspace.least_squares_rhs[1] = residual_norm
  inner_count = 0

  for column in 1:count
    inner_count = column
    z_column = view(workspace.preconditioned, :, column)
    v_column = view(workspace.krylov, :, column)
    _apply_preconditioner!(z_column, level.jacobi, v_column)
    _apply_operator!(level.operator_values, level.operator, z_column)

    for basis_column in 1:column
      v_basis = view(workspace.krylov, :, basis_column)
      workspace.hessenberg[basis_column, column] = _dot(level.operator_values, v_basis)
      _axpy!(level.operator_values, -workspace.hessenberg[basis_column, column], v_basis)
    end

    next_norm = sqrt(_dot_self(level.operator_values))
    workspace.hessenberg[column+1, column] = next_norm

    if next_norm > zero(T) && column < count + 1
      @inbounds for row in eachindex(level.operator_values)
        workspace.krylov[row, column+1] = level.operator_values[row] / next_norm
      end
    end

    _apply_previous_givens_rotations!(workspace.hessenberg, workspace.givens_cosine,
                                      workspace.givens_sine, column)
    cosine, sine = _givens_rotation(workspace.hessenberg[column, column],
                                    workspace.hessenberg[column+1, column])
    workspace.givens_cosine[column] = cosine
    workspace.givens_sine[column] = sine
    _apply_givens_rotation!(workspace.hessenberg, column, column, cosine, sine)
    _apply_givens_rotation!(workspace.least_squares_rhs, column, cosine, sine)

    next_norm == zero(T) && break
  end

  _gmres_smoother_upper_triangular_solve!(workspace.coefficients, workspace.hessenberg,
                                          workspace.least_squares_rhs, inner_count)

  for column in 1:inner_count
    scale = damping * workspace.coefficients[column]
    iszero(scale) && continue
    z_column = view(workspace.preconditioned, :, column)
    _axpy!(solution, scale, z_column)
  end

  return solution
end

function _gmres_smoother_upper_triangular_solve!(result::AbstractVector{T},
                                                 upper::AbstractMatrix{T}, rhs::AbstractVector{T},
                                                 count::Int) where {T<:AbstractFloat}
  _require_length(result, count, "GMRES smoother coefficient work vector")

  for row in count:-1:1
    value = @inbounds rhs[row]

    for column in (row+1):count
      @inbounds value -= upper[row, column] * result[column]
    end

    diagonal = @inbounds upper[row, row]
    isfinite(diagonal) ||
      throw(ArgumentError("GMRES smoother encountered a non-finite Krylov least-squares diagonal"))
    !iszero(diagonal) ||
      throw(ArgumentError("GMRES smoother encountered a singular Krylov least-squares problem"))
    result[row] = value / diagonal
  end

  return result
end

function _prolongate_reduced_add!(fine_reduced::AbstractVector{T}, transfer::_ReducedTransfer{T},
                                  coarse_reduced::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(coarse_reduced, reduced_dof_count(transfer.coarse_plan),
                  "coarse multigrid vector")
  _require_length(fine_reduced, reduced_dof_count(transfer.fine_plan), "fine multigrid vector")

  @inbounds for index in eachindex(transfer.coefficients)
    fine_reduced[transfer.fine_indices[index]] += transfer.coefficients[index] *
                                                  coarse_reduced[transfer.coarse_indices[index]]
  end

  return fine_reduced
end

function _restrict_reduced!(coarse_reduced::AbstractVector{T}, transfer::_ReducedTransfer{T},
                            fine_reduced::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(coarse_reduced, reduced_dof_count(transfer.coarse_plan),
                  "coarse multigrid vector")
  _require_length(fine_reduced, reduced_dof_count(transfer.fine_plan), "fine multigrid vector")
  fill!(coarse_reduced, zero(T))

  @inbounds for index in eachindex(transfer.coefficients)
    coarse_reduced[transfer.coarse_indices[index]] += transfer.coefficients[index] *
                                                      fine_reduced[transfer.fine_indices[index]]
  end

  return coarse_reduced
end

# Geometric multigrid for matrix-free affine solves.
#
# The production hierarchy is a single hp path: first remove excess polynomial
# order on the fixed fine mesh, then coarsen the dyadic tree one admissible
# frontier at a time. Each level is rediscretized from the original weak form on
# adapted fields that preserve field identity. Transfers are compiled once as
# sparse reduced-space prolongations and restriction is their algebraic
# transpose, so constraints remain part of the reduced operator contract.

"""
    GeometricMultigridSolver(; kwargs...)

Matrix-free geometric multigrid solver policy for affine SPD problems.

The hierarchy is compiled from an `AffineProblem`, not from an already compiled
`AssemblyPlan`, so every level can rediscretize the same weak form on adapted
fields. The implementation builds one configurable hp hierarchy on a shared
`HpSpace`; unsupported coupled-space or non-nested cases throw explicit
`ArgumentError`s instead of silently falling back to a weaker method.

`coarse_direct_dof_limit` controls the base-level solve: reduced coarse systems
at or below this size are assembled from local operator matrices and solved
with dense Cholesky when the coarse matrix is numerically symmetric positive
definite. Larger, nonsymmetric, or indefinite coarse systems use the configured
Krylov tolerance policy.
"""
struct GeometricMultigridSolver <: AbstractLinearSolver
  min_degree::Int
  max_levels::Int
  p_sequence::Symbol
  pre_smoothing_steps::Int
  post_smoothing_steps::Int
  smoother_damping::Float64
  coarse_relative_tolerance::Float64
  coarse_absolute_tolerance::Float64
  coarse_maxiter::Int
  coarse_direct_dof_limit::Int
end

function GeometricMultigridSolver(; min_degree::Integer=1, max_levels::Integer=16,
                                  p_sequence::Symbol=:bisect,
                                  pre_smoothing_steps::Integer=2,
                                  post_smoothing_steps::Integer=2,
                                  smoother_damping::Real=0.8,
                                  coarse_relative_tolerance::Real=1.0e-12,
                                  coarse_absolute_tolerance::Real=0.0,
                                  coarse_maxiter::Integer=10_000,
                                  coarse_direct_dof_limit::Integer=512)
  checked_min_degree = _checked_nonnegative(min_degree, "min_degree")
  checked_max_levels = _checked_positive(max_levels, "max_levels")
  p_sequence in (:bisect, :decrease_by_one, :go_to_one) ||
    throw(ArgumentError("p_sequence must be :bisect, :decrease_by_one, or :go_to_one"))
  checked_pre = _checked_nonnegative(pre_smoothing_steps, "pre_smoothing_steps")
  checked_post = _checked_nonnegative(post_smoothing_steps, "post_smoothing_steps")
  damping = Float64(smoother_damping)
  damping > 0 || throw(ArgumentError("smoother_damping must be positive"))
  coarse_rtol = Float64(coarse_relative_tolerance)
  coarse_atol = Float64(coarse_absolute_tolerance)
  coarse_rtol >= 0 || throw(ArgumentError("coarse_relative_tolerance must be nonnegative"))
  coarse_atol >= 0 || throw(ArgumentError("coarse_absolute_tolerance must be nonnegative"))
  checked_coarse_maxiter = _checked_positive(coarse_maxiter, "coarse_maxiter")
  checked_coarse_direct = _checked_nonnegative(coarse_direct_dof_limit,
                                               "coarse_direct_dof_limit")
  return GeometricMultigridSolver(checked_min_degree, checked_max_levels, p_sequence,
                                  checked_pre, checked_post, damping, coarse_rtol,
                                  coarse_atol, checked_coarse_maxiter, checked_coarse_direct)
end

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

struct _ReducedTransfer{T<:AbstractFloat,CP<:AssemblyPlan,FP<:AssemblyPlan}
  coarse_plan::CP
  fine_plan::FP
  coarse_indices::Vector{Int}
  fine_indices::Vector{Int}
  coefficients::Vector{T}
end

struct _LocalTransferEntry{D,T<:AbstractFloat}
  coarse_leaf::Int
  fine_leaf::Int
  coarse_mode::NTuple{D,Int}
  fine_mode::NTuple{D,Int}
  coefficient::T
end

struct _MultigridHCoarseningCandidate{D}
  cell::Int
  axis::Int
  children::NTuple{_MIDPOINT_CHILD_COUNT,Int}
  target_degrees::NTuple{D,Int}
end

struct _DenseCoarseSolver{T<:AbstractFloat}
  factor::Matrix{T}
end

struct _KrylovCoarseSolver end

struct _CompiledGeometricMultigridPreconditioner{T<:AbstractFloat,L,TR,CS} <:
       _CompiledPreconditioner{T}
  levels::L
  transfers::TR
  coarse_solver::CS
  level_rhs::Vector{Vector{T}}
  level_solution::Vector{Vector{T}}
  pre_smoothing_steps::Int
  post_smoothing_steps::Int
  smoother_damping::T
  coarse_relative_tolerance::T
  coarse_absolute_tolerance::T
  coarse_maxiter::Int
end

function _solve_affine_problem(problem::AffineProblem, ::AutoLinearSolver; kwargs...)
  return _solve_affine_problem(problem, GeometricMultigridSolver(); kwargs...)
end

function _solve_affine_problem(problem::AffineProblem,
                               solver::GeometricMultigridSolver;
                               relative_tolerance=sqrt(eps(Float64)),
                               absolute_tolerance=0.0, maxiter=nothing,
                               initial_solution=nothing)
  fine_fields = Tuple(_problem_data(problem).fields)
  fine_space = _single_multigrid_space(fine_fields)
  if !_has_multigrid_coarsening_level(fine_space, solver)
    plan = compile(problem)
    fallback_solver = CGSolver(; preconditioner=JacobiPreconditioner())
    return _solve_affine(plan; solver=fallback_solver,
                         relative_tolerance=relative_tolerance,
                         absolute_tolerance=absolute_tolerance,
                         maxiter=maxiter === nothing ? max(1_000, 2 * reduced_dof_count(plan)) :
                         maxiter,
                         initial_solution=initial_solution)
  end

  hierarchy = _compile_geometric_multigrid(problem, solver)
  levels = hierarchy.levels

  finest = levels[end]
  T = eltype(finest.residual)
  rhs = zeros(T, reduced_dof_count(finest.plan))
  _reduced_rhs!(rhs, finest.plan, finest.workspace)
  outer_maxiter = maxiter === nothing ? max(1_000, 2 * length(rhs)) : maxiter
  reduced_values = _cg_solve(finest.operator, rhs, hierarchy;
                             relative_tolerance=T(relative_tolerance),
                             absolute_tolerance=T(absolute_tolerance),
                             maxiter=outer_maxiter,
                             initial_solution=initial_solution)
  return _state_from_reduced_result(finest.plan, reduced_values)
end

function _has_multigrid_coarsening_level(space::HpSpace, solver::GeometricMultigridSolver)
  solver.max_levels > 1 || return false
  return _has_p_coarsening_level(space, solver) || _has_h_coarsening_level(space)
end

function _has_p_coarsening_level(space::HpSpace, solver::GeometricMultigridSolver)
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

function _compile_geometric_multigrid(problem::AffineProblem,
                                      solver::GeometricMultigridSolver)
  data = _problem_data(problem)
  fine_fields = Tuple(data.fields)
  fine_space = _single_multigrid_space(fine_fields)
  spaces = _hp_multigrid_spaces(fine_space, solver)
  level_fields = _multigrid_level_fields(fine_fields, spaces)
  level_problems = [_problem_on_fields(problem, fields_tuple) for fields_tuple in level_fields]
  plans = [compile(level_problem) for level_problem in level_problems]
  levels = [_compile_multigrid_level(plan) for plan in plans]
  transfers = [_compile_multigrid_transfer(levels[index].plan, levels[index+1].plan)
               for index in 1:(length(levels)-1)]
  coarse_solver = _compile_coarse_solver(levels[1], solver)
  T = eltype(origin(field_space(fine_fields[1])))
  level_rhs = [zeros(T, reduced_dof_count(level.plan)) for level in levels]
  level_solution = [zeros(T, reduced_dof_count(level.plan)) for level in levels]
  return _CompiledGeometricMultigridPreconditioner(levels, transfers, coarse_solver, level_rhs,
                                                   level_solution, solver.pre_smoothing_steps,
                                                   solver.post_smoothing_steps,
                                                   T(solver.smoother_damping),
                                                   T(solver.coarse_relative_tolerance),
                                                   T(solver.coarse_absolute_tolerance),
                                                   solver.coarse_maxiter)
end

function _compile_multigrid_level(plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  workspace = _ReducedOperatorWorkspace(plan)
  operator = _ReducedAffineOperator(plan, workspace)
  jacobi = _compile_preconditioner(JacobiPreconditioner(), operator)
  n = reduced_dof_count(plan)
  return _MultigridLevel(plan, workspace, operator, jacobi, zeros(T, n), zeros(T, n),
                         zeros(T, n))
end

function _compile_coarse_solver(level::_MultigridLevel{T},
                                solver::GeometricMultigridSolver) where {T<:AbstractFloat}
  n = reduced_dof_count(level.plan)
  0 < n <= solver.coarse_direct_dof_limit || return _KrylovCoarseSolver()
  matrix = _assemble_reduced_operator_matrix(level.plan, level.workspace.scratch)
  _matrix_is_symmetric(matrix) || return _KrylovCoarseSolver()

  try
    _dense_cholesky_factor!(matrix)
    return _DenseCoarseSolver(matrix)
  catch error
    error isa PosDefException || rethrow()
    return _KrylovCoarseSolver()
  end
end

function _matrix_is_symmetric(matrix::AbstractMatrix{T}) where {T<:AbstractFloat}
  n = _require_square_matrix(matrix, "coarse multigrid matrix")
  tolerance = sqrt(eps(T))

  for column in 1:n
    for row in (column+1):n
      scale = max(abs(matrix[row, column]), abs(matrix[column, row]), one(T))
      abs(matrix[row, column] - matrix[column, row]) <= tolerance * scale || return false
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
                              solver::GeometricMultigridSolver) where {D,T<:AbstractFloat}
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
                  lower = min(degrees[axis],
                              max(min_degree, continuity[axis] === :cg ? 1 : 0))
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

function _space_with_snapshot_and_degrees(source::HpSpace, target_snapshot::GridSnapshot,
                                         degrees)
  active = target_snapshot.active_leaves
  degree_policy = StoredDegrees(domain(source), active, degrees)
  options = SpaceOptions(basis=basis_family(source), degree=degree_policy,
                         quadrature=source.quadrature_policy,
                         continuity=continuity_policy(source))
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
  sizehint!(active, length(source_snapshot.active_leaves) - length(child_to_parent) +
                    length(candidates))

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
                 length(fine_fields))
          for level_index in eachindex(spaces)]
end

function _field_on_space(field::ScalarField, space::HpSpace)
  return ScalarField(_field_id(field), space, field_name(field))
end

function _field_on_space(field::VectorField, space::HpSpace)
  return VectorField(_field_id(field), space, component_count(field), field_name(field))
end

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
                                                 local_entries::Vector{_LocalTransferEntry{D,T}}) where {
                                                                                                      D,
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
    coarse_offset = first(field_component_range(field_layout(coarse_plan), coarse_field,
                                                component)) - 1
    fine_offset = first(field_component_range(field_layout(fine_plan), fine_field, component)) - 1
    source_expansions = Dict{Tuple{Int,NTuple{D,Int}},Vector{Pair{Int,T}}}()
    target_expansions = Dict{Tuple{Int,NTuple{D,Int}},Dict{Int,T}}()

    for entry in local_entries
      coarse_key = (entry.coarse_leaf, entry.coarse_mode)
      source = get!(source_expansions, coarse_key) do
        _local_mode_reduced_expansion(coarse_map, coarse_space, coarse_offset,
                                      entry.coarse_leaf, entry.coarse_mode)
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
        abs(coefficient) > 1000 * eps(T) || continue
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
      matrices = ntuple(axis -> _axis_transfer_matrix!(restriction_cache, coarse_space,
                                                       fine_space, coarse_leaf, fine_leaf, axis),
                        D)

      for coarse_mode in local_modes(coarse_space, coarse_leaf)
        for fine_mode in local_modes(fine_space, fine_leaf)
          coefficient = _tensor_mode_transfer_coefficient(matrices, coarse_mode, fine_mode)
          abs(coefficient) > 1000 * eps(T) || continue
          push!(entries,
                _LocalTransferEntry{D,T}(coarse_leaf, fine_leaf, coarse_mode, fine_mode,
                                         coefficient))
        end
      end
    end
  end

  return entries
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

function _axis_transfer_matrix!(cache::Dict{NTuple{4,Int},Matrix{T}},
                                coarse_space::HpSpace{D,T},
                                fine_space::HpSpace{D,T},
                                coarse_leaf::Int, fine_leaf::Int,
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

function _local_mode_reduced_expansion(map::_ReducedOperatorMap{T},
                                       space::HpSpace{D,T},
                                       field_offset::Int,
                                       leaf::Int,
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

function _fine_mode_reduced_coordinate(fine_map::_ReducedOperatorMap{T},
                                       fine_space::HpSpace{D,T},
                                       field_offset::Int,
                                       leaf::Int,
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

function _accumulate_transfer_value!(target::Dict{K,T}, key::K, value::T) where {K,
                                                                                 T<:AbstractFloat}
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

  _smooth_jacobi!(solution, level, rhs, mg.pre_smoothing_steps, mg.smoother_damping)
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
  _smooth_jacobi!(solution, level, rhs, mg.post_smoothing_steps, mg.smoother_damping)
  return solution
end

function _coarse_solve!(solution::AbstractVector{T}, solver::_DenseCoarseSolver{T},
                        level::_MultigridLevel{T}, rhs::AbstractVector{T},
                        mg::_CompiledGeometricMultigridPreconditioner{T}) where {T<:AbstractFloat}
  _require_length(solution, length(rhs), "coarse multigrid solution")
  copyto!(solution, rhs)
  _dense_cholesky_solve!(solver.factor, solution)
  return solution
end

function _coarse_solve!(solution::AbstractVector{T}, ::_KrylovCoarseSolver,
                        level::_MultigridLevel{T}, rhs::AbstractVector{T},
                        mg::_CompiledGeometricMultigridPreconditioner{T}) where {T<:AbstractFloat}
  coarse_solution = _cg_solve(level.operator, rhs, level.jacobi;
                              relative_tolerance=mg.coarse_relative_tolerance,
                              absolute_tolerance=mg.coarse_absolute_tolerance,
                              maxiter=mg.coarse_maxiter,
                              initial_solution=nothing)
  copyto!(solution, coarse_solution)
  return solution
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

function _prolongate_reduced_add!(fine_reduced::AbstractVector{T},
                                  transfer::_ReducedTransfer{T},
                                  coarse_reduced::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(coarse_reduced, reduced_dof_count(transfer.coarse_plan),
                  "coarse multigrid vector")
  _require_length(fine_reduced, reduced_dof_count(transfer.fine_plan), "fine multigrid vector")

  @inbounds for index in eachindex(transfer.coefficients)
    fine_reduced[transfer.fine_indices[index]] +=
      transfer.coefficients[index] * coarse_reduced[transfer.coarse_indices[index]]
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
    coarse_reduced[transfer.coarse_indices[index]] +=
      transfer.coefficients[index] * fine_reduced[transfer.fine_indices[index]]
  end

  return coarse_reduced
end

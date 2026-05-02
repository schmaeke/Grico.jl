# This file owns the runtime side of the matrix-free operator layer:
# 1. affine operator and right-hand-side application from an `AssemblyPlan`,
# 2. nonlinear residual and tangent-action evaluation on the full field layout,
# 3. the small amount of reusable local scratch needed by those traversals.
#
# The sparse global assembly path intentionally does not live on this branch.
# Plans still compile the same local integration data and algebraic constraints,
# but runtime work applies local kernels directly to coefficient vectors and
# scatters only vector contributions back to the global field layout.

# Matrix-free plan structure.

# Algebraic map between the full field-layout coefficient vector and the reduced
# vector seen by linear solves. The packed rows store
#
#   u_full = shift + P u_reduced,
#
# and the same coefficients are used for the adjoint projection `P' r_full`.
struct _ReducedOperatorMap{T<:AbstractFloat}
  full_dof_count::Int
  solve_dofs::Vector{Int}
  reduced_index::Vector{Int}
  shift::Vector{T}
  row_offsets::Vector{Int}
  row_indices::Vector{Int}
  row_coefficients::Vector{T}
end

# Plan-local runtime structure. `K` records whether the plan was compiled from
# an affine or residual problem, and `reduced_map` stores the shared constraint
# elimination map used by both solve paths.
struct _MatrixFreeAssemblyStructure{K,T<:AbstractFloat,M}
  reduced_map::M
end

function _compile_assembly_structure(::Val{K}, layout, cell_operators, boundary_operators,
                                     interface_operators, surface_operators, integration, dirichlet,
                                     mean_constraints, constraint_masks) where {K}
  T = eltype(origin(field_space(layout.slots[1].field)))
  reduced_map = (K === :affine || K === :residual) ?
                _compile_reduced_operator_map(T, dof_count(layout), dirichlet, mean_constraints,
                                              constraint_masks) : nothing
  return _MatrixFreeAssemblyStructure{K,T,typeof(reduced_map)}(reduced_map)
end

@inline _matrix_free_kind(::_MatrixFreeAssemblyStructure{K}) where {K} = K

function _require_matrix_free_kind(plan::AssemblyPlan, expected::Symbol)
  structure = plan.assembly_structure
  structure isa _MatrixFreeAssemblyStructure ||
    throw(ArgumentError("plan was not compiled for matrix-free operator evaluation"))
  _matrix_free_kind(structure) == expected ||
    throw(ArgumentError("operation requires a plan compiled from $(expected) problem"))
  return nothing
end

@inline _reduced_map(plan::AssemblyPlan) = plan.assembly_structure.reduced_map

reduced_dof_count(plan::AssemblyPlan) = length(_reduced_map(plan).solve_dofs)
reduced_dof_count(map::_ReducedOperatorMap) = length(map.solve_dofs)

mutable struct _BaseConstraintExpansion{T<:AbstractFloat}
  shift::T
  reduced::Dict{Int,T}
  mean::Dict{Int,T}
end

mutable struct _ReducedExpansion{T<:AbstractFloat}
  shift::T
  reduced::Dict{Int,T}
end

function _BaseConstraintExpansion(::Type{T}) where {T<:AbstractFloat}
  _BaseConstraintExpansion(zero(T), Dict{Int,T}(), Dict{Int,T}())
end

_ReducedExpansion(::Type{T}) where {T<:AbstractFloat} = _ReducedExpansion(zero(T), Dict{Int,T}())

@inline function _accumulate!(target::Dict{Int,T}, index::Int, value::T) where {T<:AbstractFloat}
  iszero(value) && return target
  target[index] = get(target, index, zero(T)) + value
  return target
end

function _add_scaled_expansion!(target::_BaseConstraintExpansion{T},
                                source::_BaseConstraintExpansion{T},
                                scale::T) where {T<:AbstractFloat}
  target.shift += scale * source.shift

  for pair in source.reduced
    _accumulate!(target.reduced, pair.first, scale * pair.second)
  end

  for pair in source.mean
    _accumulate!(target.mean, pair.first, scale * pair.second)
  end

  return target
end

function _add_scaled_expansion!(target::_ReducedExpansion{T}, source::_ReducedExpansion{T},
                                scale::T) where {T<:AbstractFloat}
  target.shift += scale * source.shift

  for pair in source.reduced
    _accumulate!(target.reduced, pair.first, scale * pair.second)
  end

  return target
end

function _compile_reduced_operator_map(::Type{T}, ndofs::Int, dirichlet::_CompiledDirichlet{T},
                                       mean_constraints::Vector{_CompiledLinearConstraint{T}},
                                       masks::_ConstraintMasks{T}) where {T<:AbstractFloat}
  constrained = masks.fixed .| masks.eliminated .| masks.constraint_rows
  solve_dofs = [dof for dof in 1:ndofs if !constrained[dof]]
  reduced_index = zeros(Int, ndofs)

  for index in eachindex(solve_dofs)
    reduced_index[solve_dofs[index]] = index
  end

  mean_pivots = [constraint.pivot for constraint in mean_constraints]
  mean_index = zeros(Int, ndofs)

  for index in eachindex(mean_pivots)
    pivot = mean_pivots[index]
    mean_index[pivot] == 0 || throw(ArgumentError("mean constraint pivots must be unique"))
    mean_index[pivot] = index
  end

  dirichlet_rows = Dict{Int,_CompiledAffineRelation{T}}()

  for row in dirichlet.rows
    haskey(dirichlet_rows, row.pivot) &&
      throw(ArgumentError("Dirichlet relation pivots must be unique"))
    dirichlet_rows[row.pivot] = row
  end

  base_cache = Vector{Union{Nothing,_BaseConstraintExpansion{T}}}(nothing, ndofs)
  visiting = falses(ndofs)
  base_expansion = dof -> _base_constraint_expansion!(base_cache, visiting, dof, reduced_index,
                                                      masks.fixed, masks.fixed_values, mean_index,
                                                      dirichlet_rows, T)
  mean_shift, mean_reduced = _compile_mean_expansions(T, mean_constraints, mean_pivots,
                                                      base_expansion)
  final_cache = Vector{Union{Nothing,_ReducedExpansion{T}}}(nothing, ndofs)
  final_visiting = falses(ndofs)
  final_expansion = dof -> _final_reduced_expansion!(final_cache, final_visiting, dof,
                                                     reduced_index, masks.fixed, masks.fixed_values,
                                                     mean_index, mean_shift, mean_reduced,
                                                     dirichlet_rows, T)
  return _pack_reduced_operator_map(T, ndofs, solve_dofs, reduced_index, final_expansion)
end

function _base_constraint_expansion!(cache, visiting::BitVector, dof::Int,
                                     reduced_index::Vector{Int}, fixed::BitVector,
                                     fixed_values::Vector{T}, mean_index::Vector{Int},
                                     dirichlet_rows, ::Type{T}) where {T<:AbstractFloat}
  cached = cache[dof]
  cached === nothing || return cached
  visiting[dof] && throw(ArgumentError("cyclic Dirichlet constraint relation detected"))
  expansion = _BaseConstraintExpansion(T)

  if fixed[dof]
    expansion.shift = fixed_values[dof]
  elseif reduced_index[dof] != 0
    expansion.reduced[reduced_index[dof]] = one(T)
  elseif mean_index[dof] != 0
    expansion.mean[mean_index[dof]] = one(T)
  elseif haskey(dirichlet_rows, dof)
    visiting[dof] = true
    row = dirichlet_rows[dof]
    expansion.shift = row.rhs

    for index in eachindex(row.indices)
      source = _base_constraint_expansion!(cache, visiting, row.indices[index], reduced_index,
                                           fixed, fixed_values, mean_index, dirichlet_rows, T)
      _add_scaled_expansion!(expansion, source, row.coefficients[index])
    end

    visiting[dof] = false
  else
    throw(ArgumentError("dof $dof has no reduced representation"))
  end

  cache[dof] = expansion
  return expansion
end

function _compile_mean_expansions(::Type{T}, mean_constraints::Vector{_CompiledLinearConstraint{T}},
                                  mean_pivots::Vector{Int}, base_expansion) where {T<:AbstractFloat}
  mean_count = length(mean_constraints)
  mean_shift = zeros(T, mean_count)
  mean_reduced = [Dict{Int,T}() for _ in 1:mean_count]
  mean_count == 0 && return mean_shift, mean_reduced
  mean_matrix = zeros(T, mean_count, mean_count)
  mean_rhs = zeros(T, mean_count)
  reduced_columns = Dict{Int,Vector{T}}()

  for row_index in eachindex(mean_constraints)
    constraint = mean_constraints[row_index]
    mean_rhs[row_index] = constraint.rhs

    for index in eachindex(constraint.indices)
      coefficient = constraint.coefficients[index]
      expansion = base_expansion(constraint.indices[index])
      mean_rhs[row_index] -= coefficient * expansion.shift

      for pair in expansion.mean
        mean_matrix[row_index, pair.first] += coefficient * pair.second
      end

      for pair in expansion.reduced
        column = get!(reduced_columns, pair.first) do
          zeros(T, mean_count)
        end
        column[row_index] -= coefficient * pair.second
      end
    end
  end

  factor = copy(mean_matrix)
  pivots = Vector{Int}(undef, mean_count)
  _dense_lu_factor!(factor, pivots)
  _dense_lu_solve!(factor, pivots, mean_rhs)
  copyto!(mean_shift, mean_rhs)

  for pair in reduced_columns
    rhs_column = pair.second
    _dense_lu_solve!(factor, pivots, rhs_column)

    for mean_index in 1:mean_count
      _accumulate!(mean_reduced[mean_index], pair.first, rhs_column[mean_index])
    end
  end

  return mean_shift, mean_reduced
end

function _final_reduced_expansion!(cache, visiting::BitVector, dof::Int, reduced_index::Vector{Int},
                                   fixed::BitVector, fixed_values::Vector{T},
                                   mean_index::Vector{Int}, mean_shift::Vector{T}, mean_reduced,
                                   dirichlet_rows, ::Type{T}) where {T<:AbstractFloat}
  cached = cache[dof]
  cached === nothing || return cached
  visiting[dof] && throw(ArgumentError("cyclic reduced constraint relation detected"))
  expansion = _ReducedExpansion(T)

  if fixed[dof]
    expansion.shift = fixed_values[dof]
  elseif reduced_index[dof] != 0
    expansion.reduced[reduced_index[dof]] = one(T)
  elseif mean_index[dof] != 0
    index = mean_index[dof]
    expansion.shift = mean_shift[index]

    for pair in mean_reduced[index]
      expansion.reduced[pair.first] = pair.second
    end
  elseif haskey(dirichlet_rows, dof)
    visiting[dof] = true
    row = dirichlet_rows[dof]
    expansion.shift = row.rhs

    for index in eachindex(row.indices)
      source = _final_reduced_expansion!(cache, visiting, row.indices[index], reduced_index, fixed,
                                         fixed_values, mean_index, mean_shift, mean_reduced,
                                         dirichlet_rows, T)
      _add_scaled_expansion!(expansion, source, row.coefficients[index])
    end

    visiting[dof] = false
  else
    throw(ArgumentError("dof $dof has no reduced representation"))
  end

  cache[dof] = expansion
  return expansion
end

function _pack_reduced_operator_map(::Type{T}, ndofs::Int, solve_dofs::Vector{Int},
                                    reduced_index::Vector{Int},
                                    final_expansion) where {T<:AbstractFloat}
  shift = zeros(T, ndofs)
  row_offsets = ones(Int, ndofs + 1)
  expansion_cache = Vector{_ReducedExpansion{T}}(undef, ndofs)
  tolerance = 1000 * eps(T)

  for dof in 1:ndofs
    expansion = final_expansion(dof)
    expansion_cache[dof] = expansion
    shift[dof] = expansion.shift
    stored = count(pair -> abs(pair.second) > tolerance, expansion.reduced)
    row_offsets[dof+1] = row_offsets[dof] + stored
  end

  row_indices = Vector{Int}(undef, row_offsets[end] - 1)
  row_coefficients = Vector{T}(undef, row_offsets[end] - 1)

  for dof in 1:ndofs
    pointer = row_offsets[dof]
    pairs = sort!(collect(expansion_cache[dof].reduced); by=first)

    for pair in pairs
      abs(pair.second) > tolerance || continue
      row_indices[pointer] = pair.first
      row_coefficients[pointer] = pair.second
      pointer += 1
    end
  end

  return _ReducedOperatorMap(ndofs, solve_dofs, reduced_index, shift, row_offsets, row_indices,
                             row_coefficients)
end

# Basic plan/state convenience API.

field_layout(plan::AssemblyPlan) = plan.layout
dof_count(plan::AssemblyPlan) = dof_count(plan.layout)

State(plan::AssemblyPlan) = State(plan.layout)
function State(plan::AssemblyPlan, coefficients::AbstractVector{T}) where {T<:AbstractFloat}
  return State(plan.layout, coefficients)
end

# Reusable local vector storage.

mutable struct _OperatorScratch{T<:AbstractFloat}
  input::Vector{T}
  output::Vector{T}
  rhs::Vector{T}
  kernel::KernelScratch{T}
end

# Matrix-free passes scatter element contributions into global vectors. To keep
# this race-free without imposing a coloring scheme on every traversal type, each
# dense worker slot receives a local scratch block and, except for slot 1, a
# full result buffer. The buffers are reduced once after all cells/faces/surfaces
# in a pass have contributed. The slot count comes from the shared-memory CPU
# runtime boundary in `common.jl`.
struct _ThreadedOperatorScratch{T<:AbstractFloat}
  local_scratch::Vector{_OperatorScratch{T}}
  result_buffers::Vector{Vector{T}}
end

function _OperatorScratch(::Type{T}, local_dof_count::Int) where {T<:AbstractFloat}
  count = max(local_dof_count, 0)
  return _OperatorScratch(zeros(T, count), zeros(T, count), zeros(T, count), KernelScratch(T))
end

function _scratch_buffer(::Type{T}, integration::_CompiledIntegration) where {T<:AbstractFloat}
  return _OperatorScratch(T, _max_local_dof_count(integration))
end

function _threaded_scratch_buffer(::Type{T}, plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  thread_count = _runtime_worker_count(_SHARED_MEMORY_CPU_BACKEND)
  local_dof_count = _max_local_dof_count(plan.integration)
  local_scratch = [_OperatorScratch(T, local_dof_count) for _ in 1:thread_count]
  result_buffers = [zeros(T, dof_count(plan)) for _ in 2:thread_count]
  return _ThreadedOperatorScratch(local_scratch, result_buffers)
end

@inline _threaded_slot_count(workspace::_ThreadedOperatorScratch) = length(workspace.local_scratch)

@inline function _slot_range(item_count::Int, slot::Int, slot_count::Int)
  block_size, remainder = divrem(item_count, slot_count)
  first_index = (slot - 1) * block_size + min(slot - 1, remainder) + 1
  last_index = first_index + block_size - 1 + (slot <= remainder ? 1 : 0)
  return first_index:last_index
end

@inline function _threaded_buffers(workspace::_ThreadedOperatorScratch, result, slot::Int)
  local_scratch = @inbounds workspace.local_scratch[slot]
  local_result = slot == 1 ? result : @inbounds(workspace.result_buffers[slot-1])
  return local_scratch, local_result
end

function _clear_threaded_result!(workspace::_ThreadedOperatorScratch{T},
                                 result::AbstractVector{T}) where {T<:AbstractFloat}
  buffers = workspace.result_buffers
  zero_value = zero(T)

  if isempty(buffers)
    fill!(result, zero_value)
    return result
  end

  @_threaded_loop for index in eachindex(result)
    result[index] = zero_value

    for buffer in buffers
      buffer[index] = zero_value
    end
  end

  return result
end

function _merge_threaded_result!(result::AbstractVector{T},
                                 workspace::_ThreadedOperatorScratch{T}) where {T<:AbstractFloat}
  buffers = workspace.result_buffers
  isempty(buffers) && return result

  @_threaded_loop for index in eachindex(result)
    value = result[index]

    for buffer in buffers
      value += buffer[index]
    end

    result[index] = value
  end

  return result
end

"""
    OperatorWorkspace(plan)

Reusable runtime storage for repeated affine [`rhs!`](@ref) and [`apply!`](@ref)
calls on `plan`.

The default two- and three-argument `rhs!`/`apply!` methods allocate a fresh
workspace for convenience. Performance-critical loops should create one
`OperatorWorkspace` for the compiled plan and pass it to the workspace-aware
methods instead.
"""
struct OperatorWorkspace{T<:AbstractFloat,P<:AssemblyPlan}
  plan::P
  threaded_scratch::_ThreadedOperatorScratch{T}
end

function OperatorWorkspace(plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  return OperatorWorkspace{T,typeof(plan)}(plan, _threaded_scratch_buffer(T, plan))
end

function _check_operator_workspace(plan::AssemblyPlan, workspace::OperatorWorkspace)
  workspace.plan === plan ||
    throw(ArgumentError("operator workspace belongs to a different AssemblyPlan"))
  return nothing
end

"""
    ResidualWorkspace(plan)

Reusable runtime storage for repeated nonlinear residual and tangent-action
evaluations on `plan`.

A workspace is tied to the exact plan used to create it.
"""
struct ResidualWorkspace{T<:AbstractFloat,P<:AssemblyPlan}
  plan::P
  threaded_scratch::_ThreadedOperatorScratch{T}
end

function ResidualWorkspace(plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  return ResidualWorkspace{T,typeof(plan)}(plan, _threaded_scratch_buffer(T, plan))
end

function _check_residual_workspace(plan::AssemblyPlan, workspace::ResidualWorkspace)
  workspace.plan === plan ||
    throw(ArgumentError("residual workspace belongs to a different AssemblyPlan"))
  return nothing
end

function _check_state(plan::AssemblyPlan{D,T}, state::State{T}) where {D,T<:AbstractFloat}
  field_layout(state) === plan.layout ||
    throw(ArgumentError("state belongs to a different field layout"))
  return nothing
end

function _check_no_alias(result::AbstractVector, input::AbstractVector)
  result === input && throw(ArgumentError("result and input vectors must not alias"))
  return nothing
end

# Affine right-hand side.

"""
    rhs(problem)
    rhs(plan)
    rhs!(result, plan)
    rhs!(result, plan, workspace)

Evaluate the affine right-hand side associated with an [`AffineProblem`](@ref)
or a compiled affine [`AssemblyPlan`](@ref).

The result is stored on the full field layout. Rows reserved for strong
Dirichlet data or mean-value constraints contain the explicit algebraic
right-hand side of those constraint equations.
"""
rhs(problem::AffineProblem) = rhs(compile(problem))

function rhs(plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  result = zeros(T, dof_count(plan))
  rhs!(result, plan)
  return result
end

"""
    rhs!(result, plan)
    rhs!(result, plan, workspace)

Store the affine right-hand side of `plan` in `result`.
"""
function rhs!(result::AbstractVector{T}, plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  return rhs!(result, plan, OperatorWorkspace(plan))
end

function rhs!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
              workspace::OperatorWorkspace{T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  _check_operator_workspace(plan, workspace)
  _require_length(result, dof_count(plan), "result")
  scratch = workspace.threaded_scratch
  _clear_threaded_result!(scratch, result)
  traversal = plan.traversal_plan
  masks = plan.constraint_masks
  _rhs_pass!(scratch, result, traversal.cell_batches, plan.integration.cells, plan.cell_operators,
             masks.fixed, masks.blocked_rows, cell_rhs!)
  _boundary_rhs_pass!(scratch, result, traversal.boundary_batches, plan.integration.boundary_faces,
                      plan.boundary_operators, masks.fixed, masks.blocked_rows)
  _rhs_pass!(scratch, result, traversal.interface_batches, plan.integration.interfaces,
             plan.interface_operators, masks.fixed, masks.blocked_rows, interface_rhs!)
  _surface_rhs_pass!(scratch, result, traversal.surface_batches, plan.integration.embedded_surfaces,
                     plan.surface_operators, masks.fixed, masks.blocked_rows)
  _merge_threaded_result!(result, scratch)
  _write_constraint_rhs!(result, plan)
  return result
end

# Affine matrix-free action.

"""
    apply(plan, coefficients)
    apply!(result, plan, coefficients)
    apply!(result, plan, coefficients, workspace)

Apply the affine operator compiled in `plan` directly to `coefficients`.

Both vectors use the full field-layout numbering. Constraint rows are applied
as explicit algebraic equations: fixed Dirichlet rows are identities,
Dirichlet affine rows evaluate their elimination relation, and mean-value rows
evaluate their compiled linear functional.
"""
function apply(plan::AssemblyPlan{D,T}, coefficients::AbstractVector{T}) where {D,T<:AbstractFloat}
  result = zeros(T, dof_count(plan))
  apply!(result, plan, coefficients)
  return result
end

"""
    apply!(result, plan, coefficients)
    apply!(result, plan, coefficients, workspace)

Store the affine matrix-free operator action `plan * coefficients` in `result`.
"""
function apply!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                coefficients::AbstractVector{T}) where {D,T<:AbstractFloat}
  return apply!(result, plan, coefficients, OperatorWorkspace(plan))
end

function apply!(result::AbstractVector{T}, plan::AssemblyPlan{D,T}, coefficients::AbstractVector{T},
                workspace::OperatorWorkspace{T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  _check_operator_workspace(plan, workspace)
  _require_length(result, dof_count(plan), "result")
  _require_length(coefficients, dof_count(plan), "coefficient vector")
  _check_no_alias(result, coefficients)
  scratch = workspace.threaded_scratch
  _clear_threaded_result!(scratch, result)
  traversal = plan.traversal_plan
  masks = plan.constraint_masks
  _apply_pass!(scratch, result, traversal.cell_batches, plan.integration.cells, plan.cell_operators,
               coefficients, masks.fixed, masks.blocked_rows, cell_apply!)
  _boundary_apply_pass!(scratch, result, traversal.boundary_batches,
                        plan.integration.boundary_faces, plan.boundary_operators, coefficients,
                        masks.fixed, masks.blocked_rows)
  _apply_pass!(scratch, result, traversal.interface_batches, plan.integration.interfaces,
               plan.interface_operators, coefficients, masks.fixed, masks.blocked_rows,
               interface_apply!)
  _surface_apply_pass!(scratch, result, traversal.surface_batches,
                       plan.integration.embedded_surfaces, plan.surface_operators, coefficients,
                       masks.fixed, masks.blocked_rows)
  _merge_threaded_result!(result, scratch)
  _write_constraint_action!(result, plan, coefficients)
  return result
end

# Reusable storage for reduced operator applications. `state` wraps
# `full_state`, so updating the vector in place immediately updates the state
# object passed to residual and tangent kernels without allocating a fresh
# wrapper on every Newton step.
mutable struct _ReducedOperatorWorkspace{T<:AbstractFloat,S<:State{T}}
  scratch::_OperatorScratch{T}
  threaded_scratch::_ThreadedOperatorScratch{T}
  full_input::Vector{T}
  full_output::Vector{T}
  full_state::Vector{T}
  state::S
end

function _ReducedOperatorWorkspace(plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  full_state = zeros(T, dof_count(plan))
  return _ReducedOperatorWorkspace(_scratch_buffer(T, plan.integration),
                                   _threaded_scratch_buffer(T, plan), zeros(T, dof_count(plan)),
                                   zeros(T, dof_count(plan)), full_state, State(plan, full_state))
end

function _rhs_physical!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                        scratch::_ThreadedOperatorScratch{T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  _require_length(result, dof_count(plan), "result")
  _clear_threaded_result!(scratch, result)
  traversal = plan.traversal_plan
  _rhs_pass!(scratch, result, traversal.cell_batches, plan.integration.cells, plan.cell_operators,
             nothing, nothing, cell_rhs!)
  _boundary_rhs_pass!(scratch, result, traversal.boundary_batches, plan.integration.boundary_faces,
                      plan.boundary_operators, nothing, nothing)
  _rhs_pass!(scratch, result, traversal.interface_batches, plan.integration.interfaces,
             plan.interface_operators, nothing, nothing, interface_rhs!)
  _surface_rhs_pass!(scratch, result, traversal.surface_batches, plan.integration.embedded_surfaces,
                     plan.surface_operators, nothing, nothing)
  _merge_threaded_result!(result, scratch)
  return result
end

function _apply_physical!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                          coefficients::AbstractVector{T},
                          scratch::_ThreadedOperatorScratch{T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  _require_length(result, dof_count(plan), "result")
  _require_length(coefficients, dof_count(plan), "coefficient vector")
  _clear_threaded_result!(scratch, result)
  traversal = plan.traversal_plan
  _apply_pass!(scratch, result, traversal.cell_batches, plan.integration.cells, plan.cell_operators,
               coefficients, nothing, nothing, cell_apply!)
  _boundary_apply_pass!(scratch, result, traversal.boundary_batches,
                        plan.integration.boundary_faces, plan.boundary_operators, coefficients,
                        nothing, nothing)
  _apply_pass!(scratch, result, traversal.interface_batches, plan.integration.interfaces,
               plan.interface_operators, coefficients, nothing, nothing, interface_apply!)
  _surface_apply_pass!(scratch, result, traversal.surface_batches,
                       plan.integration.embedded_surfaces, plan.surface_operators, coefficients,
                       nothing, nothing)
  _merge_threaded_result!(result, scratch)
  return result
end

function _diagonal_physical!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                             scratch::_OperatorScratch{T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  _require_length(result, dof_count(plan), "result")
  fill!(result, zero(T))
  traversal = plan.traversal_plan
  _diagonal_pass!(scratch, result, traversal.cell_batches, plan.integration.cells,
                  plan.cell_operators, cell_diagonal!)
  _boundary_diagonal_pass!(scratch, result, traversal.boundary_batches,
                           plan.integration.boundary_faces, plan.boundary_operators)
  _diagonal_pass!(scratch, result, traversal.interface_batches, plan.integration.interfaces,
                  plan.interface_operators, interface_diagonal!)
  _surface_diagonal_pass!(scratch, result, traversal.surface_batches,
                          plan.integration.embedded_surfaces, plan.surface_operators)
  return result
end

function _reduced_diagonal!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                            workspace::_ReducedOperatorWorkspace{T}) where {D,T<:AbstractFloat}
  map = _reduced_map(plan)
  _require_length(result, reduced_dof_count(map), "reduced diagonal")

  if !_can_use_local_diagonal_kernels(plan, workspace.scratch, map)
    fill!(result, zero(T))
    return false
  end

  _diagonal_physical!(workspace.full_output, plan, workspace.scratch)
  _copy_direct_reduced_diagonal!(result, map, workspace.full_output)
  return true
end

function _copy_direct_reduced_diagonal!(result::AbstractVector{T}, map::_ReducedOperatorMap{T},
                                        physical_diagonal::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(result, reduced_dof_count(map), "reduced diagonal")
  _require_length(physical_diagonal, map.full_dof_count, "physical diagonal")

  @inbounds for index in eachindex(map.solve_dofs)
    result[index] = physical_diagonal[map.solve_dofs[index]]
  end

  return result
end

function _can_use_local_diagonal_kernels(plan::AssemblyPlan{D,T}, scratch::_OperatorScratch{T},
                                         map::_ReducedOperatorMap{T}) where {D,T<:AbstractFloat}
  return _has_direct_reduced_map(map) && _diagonal_kernel_requirements_satisfied(plan, scratch)
end

function _has_direct_reduced_map(map::_ReducedOperatorMap{T}) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for dof in 1:map.full_dof_count
    first_term = map.row_offsets[dof]
    last_term = map.row_offsets[dof+1] - 1

    if map.reduced_index[dof] != 0
      first_term == last_term || return false
      map.row_indices[first_term] == map.reduced_index[dof] || return false
      abs(map.row_coefficients[first_term] - one(T)) <= tolerance || return false
    elseif first_term <= last_term
      return false
    end
  end

  return true
end

function _item_has_direct_diagonal_map(item::_AssemblyValues, ::Type{T}) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for local_dof in 1:item.local_dof_count
    first_term = item.term_offsets[local_dof]
    last_term = item.term_offsets[local_dof+1] - 1
    first_term == last_term || return false
    global_dof = item.single_term_indices[local_dof]
    global_dof != 0 || return false
    abs(item.single_term_coefficients[local_dof]) > tolerance || return false

    for previous_dof in 1:(local_dof-1)
      item.single_term_indices[previous_dof] == global_dof && return false
    end
  end

  return true
end

# The diagonal scatter is only correct when every local dof maps directly to one
# global dof. Check that condition only for batches whose operators actually
# implement an affine action; pure right-hand-side operators do not affect the
# operator diagonal and should not disable Jacobi data for unrelated cells.
function _diagonal_kernel_requirements_satisfied(plan::AssemblyPlan, scratch::_OperatorScratch)
  traversal = plan.traversal_plan
  return _diagonal_requirements_satisfied(scratch, traversal.cell_batches, plan.integration.cells,
                                          plan.cell_operators, cell_apply!, cell_diagonal!,
                                          _DEFAULT_CELL_APPLY_METHOD,
                                          _DEFAULT_CELL_APPLY_SCRATCH_METHOD,
                                          _DEFAULT_CELL_DIAGONAL_METHOD,
                                          _DEFAULT_CELL_DIAGONAL_SCRATCH_METHOD) &&
         _filtered_diagonal_requirements_satisfied(scratch, traversal.boundary_batches,
                                                   plan.integration.boundary_faces,
                                                   plan.boundary_operators, face_apply!,
                                                   face_diagonal!, _DEFAULT_FACE_APPLY_METHOD,
                                                   _DEFAULT_FACE_APPLY_SCRATCH_METHOD,
                                                   _DEFAULT_FACE_DIAGONAL_METHOD,
                                                   _DEFAULT_FACE_DIAGONAL_SCRATCH_METHOD) &&
         _diagonal_requirements_satisfied(scratch, traversal.interface_batches,
                                          plan.integration.interfaces, plan.interface_operators,
                                          interface_apply!, interface_diagonal!,
                                          _DEFAULT_INTERFACE_APPLY_METHOD,
                                          _DEFAULT_INTERFACE_APPLY_SCRATCH_METHOD,
                                          _DEFAULT_INTERFACE_DIAGONAL_METHOD,
                                          _DEFAULT_INTERFACE_DIAGONAL_SCRATCH_METHOD) &&
         _filtered_diagonal_requirements_satisfied(scratch, traversal.surface_batches,
                                                   plan.integration.embedded_surfaces,
                                                   plan.surface_operators, surface_apply!,
                                                   surface_diagonal!, _DEFAULT_SURFACE_APPLY_METHOD,
                                                   _DEFAULT_SURFACE_APPLY_SCRATCH_METHOD,
                                                   _DEFAULT_SURFACE_DIAGONAL_METHOD,
                                                   _DEFAULT_SURFACE_DIAGONAL_SCRATCH_METHOD)
end

function _diagonal_requirements_satisfied(scratch::_OperatorScratch{T},
                                          batches::Vector{_KernelBatch}, items, operators,
                                          apply_hook, diagonal_hook, default_apply_method,
                                          default_apply_scratch_method, default_diagonal_method,
                                          default_diagonal_scratch_method) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return true

  for kernel_batch in batches
    item = @inbounds items[first(kernel_batch.item_indices)]
    local_input = view(scratch.input, 1:kernel_batch.local_dof_count)
    local_output = view(scratch.output, 1:kernel_batch.local_dof_count)
    local_diagonal = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    has_action = false

    for operator in operators
      _uses_default_operator_method(apply_hook, default_apply_method, default_apply_scratch_method,
                                    local_output, operator, item, local_input, scratch.kernel) &&
        continue
      has_action = true
      _uses_default_operator_method(diagonal_hook, default_diagonal_method,
                                    default_diagonal_scratch_method, local_diagonal, operator, item,
                                    scratch.kernel) && return false
    end

    has_action || continue

    for item_index in kernel_batch.item_indices
      item = @inbounds items[item_index]
      _item_has_direct_diagonal_map(item, T) || return false
    end
  end

  return true
end

function _filtered_diagonal_requirements_satisfied(scratch::_OperatorScratch{T},
                                                   batches::Vector{_FilteredKernelBatch}, items,
                                                   operators, apply_hook, diagonal_hook,
                                                   default_apply_method,
                                                   default_apply_scratch_method,
                                                   default_diagonal_method,
                                                   default_diagonal_scratch_method) where {T<:AbstractFloat}
  isempty(batches) && return true

  for kernel_batch in batches
    item = @inbounds items[first(kernel_batch.item_indices)]
    local_input = view(scratch.input, 1:kernel_batch.local_dof_count)
    local_output = view(scratch.output, 1:kernel_batch.local_dof_count)
    local_diagonal = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    has_action = false

    for operator_index in kernel_batch.operator_indices
      wrapped = @inbounds operators[operator_index]
      operator = wrapped.operator
      _uses_default_operator_method(apply_hook, default_apply_method, default_apply_scratch_method,
                                    local_output, operator, item, local_input, scratch.kernel) &&
        continue
      has_action = true
      _uses_default_operator_method(diagonal_hook, default_diagonal_method,
                                    default_diagonal_scratch_method, local_diagonal, operator, item,
                                    scratch.kernel) && return false
    end

    has_action || continue

    for item_index in kernel_batch.item_indices
      item = @inbounds items[item_index]
      _item_has_direct_diagonal_map(item, T) || return false
    end
  end

  return true
end

function _uses_default_method(hook, default_method, arguments...)
  return which(hook, Base.typesof(arguments...)) === default_method
end

function _uses_default_operator_method(hook, default_method, default_scratch_method,
                                       arguments_and_scratch...)
  scratch = arguments_and_scratch[end]
  arguments = ntuple(index -> arguments_and_scratch[index], length(arguments_and_scratch) - 1)
  return _uses_default_method(hook, default_scratch_method, arguments..., scratch) &&
         _uses_default_method(hook, default_method, arguments...)
end

function _expand_reduced!(full_values::AbstractVector{T}, map::_ReducedOperatorMap{T},
                          reduced_values::AbstractVector{T};
                          include_shift::Bool=true) where {T<:AbstractFloat}
  _require_length(full_values, map.full_dof_count, "full vector")
  _require_length(reduced_values, reduced_dof_count(map), "reduced vector")

  if include_shift
    copyto!(full_values, map.shift)
  else
    fill!(full_values, zero(T))
  end

  for dof in 1:map.full_dof_count
    value = full_values[dof]

    @inbounds for pointer in map.row_offsets[dof]:(map.row_offsets[dof+1]-1)
      value += map.row_coefficients[pointer] * reduced_values[map.row_indices[pointer]]
    end

    full_values[dof] = value
  end

  return full_values
end

function _project_reduced!(reduced_values::AbstractVector{T}, map::_ReducedOperatorMap{T},
                           full_values::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(reduced_values, reduced_dof_count(map), "reduced vector")
  _require_length(full_values, map.full_dof_count, "full vector")
  fill!(reduced_values, zero(T))

  for dof in 1:map.full_dof_count
    value = full_values[dof]

    @inbounds for pointer in map.row_offsets[dof]:(map.row_offsets[dof+1]-1)
      reduced_values[map.row_indices[pointer]] += map.row_coefficients[pointer] * value
    end
  end

  return reduced_values
end

function _compress_reduced!(reduced_values::AbstractVector{T}, map::_ReducedOperatorMap{T},
                            full_values::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(reduced_values, reduced_dof_count(map), "reduced vector")
  _require_length(full_values, map.full_dof_count, "full vector")

  @inbounds for index in eachindex(map.solve_dofs)
    reduced_values[index] = full_values[map.solve_dofs[index]]
  end

  return reduced_values
end

function _reduced_apply!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                         reduced_coefficients::AbstractVector{T},
                         workspace::_ReducedOperatorWorkspace{T}) where {D,T<:AbstractFloat}
  map = _reduced_map(plan)
  _require_length(result, reduced_dof_count(map), "reduced result")
  _require_length(reduced_coefficients, reduced_dof_count(map), "reduced coefficient vector")
  _expand_reduced!(workspace.full_input, map, reduced_coefficients; include_shift=false)
  _apply_physical!(workspace.full_output, plan, workspace.full_input, workspace.threaded_scratch)
  _project_reduced!(result, map, workspace.full_output)
  return result
end

function _reduced_rhs!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                       workspace::_ReducedOperatorWorkspace{T}) where {D,T<:AbstractFloat}
  map = _reduced_map(plan)
  _require_length(result, reduced_dof_count(map), "reduced rhs")
  _rhs_physical!(workspace.full_output, plan, workspace.threaded_scratch)

  if any(!iszero, map.shift)
    _apply_physical!(workspace.full_input, plan, map.shift, workspace.threaded_scratch)

    for index in 1:map.full_dof_count
      workspace.full_output[index] -= workspace.full_input[index]
    end
  end

  _project_reduced!(result, map, workspace.full_output)
  return result
end

function _reduced_residual!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                            reduced_coefficients::AbstractVector{T},
                            workspace::_ReducedOperatorWorkspace{T},
                            residual_workspace::ResidualWorkspace{T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :residual)
  map = _reduced_map(plan)
  _require_length(result, reduced_dof_count(map), "reduced residual")
  _require_length(reduced_coefficients, reduced_dof_count(map), "reduced state")
  _expand_reduced!(workspace.full_state, map, reduced_coefficients; include_shift=true)
  residual!(workspace.full_output, plan, workspace.state, residual_workspace)
  _project_reduced!(result, map, workspace.full_output)
  return result
end

function _reduced_tangent_apply!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                                 reduced_increment::AbstractVector{T},
                                 workspace::_ReducedOperatorWorkspace{T},
                                 residual_workspace::ResidualWorkspace{T}) where {D,
                                                                                  T<:AbstractFloat}
  _require_matrix_free_kind(plan, :residual)
  map = _reduced_map(plan)
  _require_length(result, reduced_dof_count(map), "reduced tangent result")
  _require_length(reduced_increment, reduced_dof_count(map), "reduced increment")
  _expand_reduced!(workspace.full_input, map, reduced_increment; include_shift=false)
  tangent_apply!(workspace.full_output, plan, workspace.state, workspace.full_input,
                 residual_workspace)
  _project_reduced!(result, map, workspace.full_output)
  return result
end

function _reconstruct_reduced_solution!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                                        reduced_solution::AbstractVector{T}) where {D,
                                                                                    T<:AbstractFloat}
  map = _reduced_map(plan)
  return _expand_reduced!(result, map, reduced_solution; include_shift=true)
end

# Nonlinear residual and tangent action.

"""
    residual(plan, state)
    residual!(result, plan, state)
    residual!(result, plan, state, workspace)

Evaluate the nonlinear residual associated with `plan` at `state`.

The residual is formed on the full field layout. Operator rows skip dofs that
are reserved for explicit constraints, and the constraint equations are written
after the local residual contributions have been scattered.
"""
function residual(plan::AssemblyPlan{D,T}, state::State{T}) where {D,T<:AbstractFloat}
  result = zeros(T, dof_count(plan))
  residual!(result, plan, state)
  return result
end

"""
    residual!(result, plan, state)
    residual!(result, plan, state, workspace)

Store the nonlinear residual of `plan` at `state` in `result`.
"""
function residual!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                   state::State{T}) where {D,T<:AbstractFloat}
  return residual!(result, plan, state, ResidualWorkspace(plan))
end

function residual!(result::AbstractVector{T}, plan::AssemblyPlan{D,T}, state::State{T},
                   workspace::ResidualWorkspace{T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :residual)
  _check_state(plan, state)
  _check_residual_workspace(plan, workspace)
  _require_length(result, dof_count(plan), "result")
  scratch = workspace.threaded_scratch
  _clear_threaded_result!(scratch, result)
  traversal = plan.traversal_plan
  masks = plan.constraint_masks
  _residual_pass!(scratch, result, traversal.cell_batches, plan.integration.cells,
                  plan.cell_operators, state, masks.fixed, masks.blocked_rows, cell_residual!)
  _boundary_residual_pass!(scratch, result, traversal.boundary_batches,
                           plan.integration.boundary_faces, plan.boundary_operators, state,
                           masks.fixed, masks.blocked_rows)
  _residual_pass!(scratch, result, traversal.interface_batches, plan.integration.interfaces,
                  plan.interface_operators, state, masks.fixed, masks.blocked_rows,
                  interface_residual!)
  _surface_residual_pass!(scratch, result, traversal.surface_batches,
                          plan.integration.embedded_surfaces, plan.surface_operators, state,
                          masks.fixed, masks.blocked_rows)
  _merge_threaded_result!(result, scratch)
  _write_constraint_residual!(result, plan, coefficients(state))
  return result
end

"""
    tangent_apply(plan, state, increment)
    tangent_apply!(result, plan, state, increment)
    tangent_apply!(result, plan, state, increment, workspace)

Apply the nonlinear tangent operator at `state` to `increment`.

This is the matrix-free counterpart to the removed sparse tangent construction.
"""
function tangent_apply(plan::AssemblyPlan{D,T}, state::State{T},
                       increment::AbstractVector{T}) where {D,T<:AbstractFloat}
  result = zeros(T, dof_count(plan))
  tangent_apply!(result, plan, state, increment)
  return result
end

"""
    tangent_apply!(result, plan, state, increment)
    tangent_apply!(result, plan, state, increment, workspace)

Store the matrix-free tangent action at `state` applied to `increment` in `result`.
"""
function tangent_apply!(result::AbstractVector{T}, plan::AssemblyPlan{D,T}, state::State{T},
                        increment::AbstractVector{T}) where {D,T<:AbstractFloat}
  return tangent_apply!(result, plan, state, increment, ResidualWorkspace(plan))
end

function tangent_apply!(result::AbstractVector{T}, plan::AssemblyPlan{D,T}, state::State{T},
                        increment::AbstractVector{T},
                        workspace::ResidualWorkspace{T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :residual)
  _check_state(plan, state)
  _check_residual_workspace(plan, workspace)
  _require_length(result, dof_count(plan), "result")
  _require_length(increment, dof_count(plan), "increment vector")
  _check_no_alias(result, increment)
  scratch = workspace.threaded_scratch
  _clear_threaded_result!(scratch, result)
  traversal = plan.traversal_plan
  masks = plan.constraint_masks
  _tangent_apply_pass!(scratch, result, traversal.cell_batches, plan.integration.cells,
                       plan.cell_operators, state, increment, masks.fixed, masks.blocked_rows,
                       cell_tangent_apply!)
  _boundary_tangent_apply_pass!(scratch, result, traversal.boundary_batches,
                                plan.integration.boundary_faces, plan.boundary_operators, state,
                                increment, masks.fixed, masks.blocked_rows)
  _tangent_apply_pass!(scratch, result, traversal.interface_batches, plan.integration.interfaces,
                       plan.interface_operators, state, increment, masks.fixed, masks.blocked_rows,
                       interface_tangent_apply!)
  _surface_tangent_apply_pass!(scratch, result, traversal.surface_batches,
                               plan.integration.embedded_surfaces, plan.surface_operators, state,
                               increment, masks.fixed, masks.blocked_rows)
  _merge_threaded_result!(result, scratch)
  _write_constraint_action!(result, plan, increment)
  return result
end

# Local traversal helpers.

@inline _rhs_operators!(local_rhs, ::Tuple{}, item, rhs_hook, scratch) = nothing

@inline function _rhs_operators!(local_rhs, operators::Tuple, item, rhs_hook, scratch)
  rhs_hook(local_rhs, first(operators), item, scratch)
  return _rhs_operators!(local_rhs, Base.tail(operators), item, rhs_hook, scratch)
end

@inline _apply_operators!(local_output, ::Tuple{}, item, local_input, apply_hook, scratch) = nothing

@inline function _apply_operators!(local_output, operators::Tuple, item, local_input, apply_hook,
                                   scratch)
  apply_hook(local_output, first(operators), item, local_input, scratch)
  return _apply_operators!(local_output, Base.tail(operators), item, local_input, apply_hook,
                           scratch)
end

@inline _diagonal_operators!(local_diagonal, ::Tuple{}, item, diagonal_hook, scratch) = nothing

@inline function _diagonal_operators!(local_diagonal, operators::Tuple, item, diagonal_hook,
                                      scratch)
  diagonal_hook(local_diagonal, first(operators), item, scratch)
  return _diagonal_operators!(local_diagonal, Base.tail(operators), item, diagonal_hook, scratch)
end

@inline _residual_operators!(local_rhs, ::Tuple{}, item, state, residual_hook, scratch) = nothing

@inline function _residual_operators!(local_rhs, operators::Tuple, item, state, residual_hook,
                                      scratch)
  residual_hook(local_rhs, first(operators), item, state, scratch)
  return _residual_operators!(local_rhs, Base.tail(operators), item, state, residual_hook, scratch)
end

@inline _tangent_apply_operators!(local_output, ::Tuple{}, item, state, local_input, apply_hook, scratch) = nothing

@inline function _tangent_apply_operators!(local_output, operators::Tuple, item, state, local_input,
                                           apply_hook, scratch)
  apply_hook(local_output, first(operators), item, state, local_input, scratch)
  return _tangent_apply_operators!(local_output, Base.tail(operators), item, state, local_input,
                                   apply_hook, scratch)
end

@inline function _threaded_rhs_item!(scratch::_ThreadedOperatorScratch{T},
                                     result::AbstractVector{T}, slot::Int, items, item_index::Int,
                                     local_dof_count::Int, operators, fixed, blocked_rows,
                                     rhs_hook) where {T<:AbstractFloat}
  local_scratch, local_result = _threaded_buffers(scratch, result, slot)
  local_rhs = view(local_scratch.rhs, 1:local_dof_count)
  item = @inbounds items[item_index]
  fill!(local_rhs, zero(T))
  _rhs_operators!(local_rhs, operators, item, rhs_hook, local_scratch.kernel)
  _scatter_local_vector!(local_result, item, local_rhs, fixed, blocked_rows)
  return nothing
end

@inline function _threaded_filtered_rhs_item!(scratch::_ThreadedOperatorScratch{T},
                                              result::AbstractVector{T}, slot::Int, items,
                                              item_index::Int, local_dof_count::Int, operators,
                                              operator_indices, fixed, blocked_rows,
                                              rhs_hook) where {T<:AbstractFloat}
  local_scratch, local_result = _threaded_buffers(scratch, result, slot)
  local_rhs = view(local_scratch.rhs, 1:local_dof_count)
  item = @inbounds items[item_index]
  fill!(local_rhs, zero(T))

  for operator_index in operator_indices
    wrapped = @inbounds operators[operator_index]
    rhs_hook(local_rhs, wrapped.operator, item, local_scratch.kernel)
  end

  _scatter_local_vector!(local_result, item, local_rhs, fixed, blocked_rows)
  return nothing
end

@inline function _threaded_apply_item!(scratch::_ThreadedOperatorScratch{T},
                                       result::AbstractVector{T}, slot::Int, items, item_index::Int,
                                       local_dof_count::Int, operators,
                                       coefficients::AbstractVector{T}, fixed, blocked_rows,
                                       apply_hook) where {T<:AbstractFloat}
  local_scratch, local_result = _threaded_buffers(scratch, result, slot)
  local_input = view(local_scratch.input, 1:local_dof_count)
  local_output = view(local_scratch.output, 1:local_dof_count)
  item = @inbounds items[item_index]
  _gather_local_coefficients!(local_input, item, coefficients)
  fill!(local_output, zero(T))
  _apply_operators!(local_output, operators, item, local_input, apply_hook, local_scratch.kernel)
  _scatter_local_vector!(local_result, item, local_output, fixed, blocked_rows)
  return nothing
end

@inline function _threaded_filtered_apply_item!(scratch::_ThreadedOperatorScratch{T},
                                                result::AbstractVector{T}, slot::Int, items,
                                                item_index::Int, local_dof_count::Int, operators,
                                                operator_indices, coefficients::AbstractVector{T},
                                                fixed, blocked_rows,
                                                apply_hook) where {T<:AbstractFloat}
  local_scratch, local_result = _threaded_buffers(scratch, result, slot)
  local_input = view(local_scratch.input, 1:local_dof_count)
  local_output = view(local_scratch.output, 1:local_dof_count)
  item = @inbounds items[item_index]
  _gather_local_coefficients!(local_input, item, coefficients)
  fill!(local_output, zero(T))

  for operator_index in operator_indices
    wrapped = @inbounds operators[operator_index]
    apply_hook(local_output, wrapped.operator, item, local_input, local_scratch.kernel)
  end

  _scatter_local_vector!(local_result, item, local_output, fixed, blocked_rows)
  return nothing
end

@inline function _threaded_residual_item!(scratch::_ThreadedOperatorScratch{T},
                                          result::AbstractVector{T}, slot::Int, items,
                                          item_index::Int, local_dof_count::Int, operators,
                                          state::State{T}, fixed, blocked_rows,
                                          residual_hook) where {T<:AbstractFloat}
  local_scratch, local_result = _threaded_buffers(scratch, result, slot)
  local_rhs = view(local_scratch.rhs, 1:local_dof_count)
  item = @inbounds items[item_index]
  fill!(local_rhs, zero(T))
  _residual_operators!(local_rhs, operators, item, state, residual_hook, local_scratch.kernel)
  _scatter_local_vector!(local_result, item, local_rhs, fixed, blocked_rows)
  return nothing
end

@inline function _threaded_filtered_residual_item!(scratch::_ThreadedOperatorScratch{T},
                                                   result::AbstractVector{T}, slot::Int, items,
                                                   item_index::Int, local_dof_count::Int, operators,
                                                   operator_indices, state::State{T}, fixed,
                                                   blocked_rows,
                                                   residual_hook) where {T<:AbstractFloat}
  local_scratch, local_result = _threaded_buffers(scratch, result, slot)
  local_rhs = view(local_scratch.rhs, 1:local_dof_count)
  item = @inbounds items[item_index]
  fill!(local_rhs, zero(T))

  for operator_index in operator_indices
    wrapped = @inbounds operators[operator_index]
    residual_hook(local_rhs, wrapped.operator, item, state, local_scratch.kernel)
  end

  _scatter_local_vector!(local_result, item, local_rhs, fixed, blocked_rows)
  return nothing
end

@inline function _threaded_tangent_apply_item!(scratch::_ThreadedOperatorScratch{T},
                                               result::AbstractVector{T}, slot::Int, items,
                                               item_index::Int, local_dof_count::Int, operators,
                                               state::State{T}, increment::AbstractVector{T}, fixed,
                                               blocked_rows, apply_hook) where {T<:AbstractFloat}
  local_scratch, local_result = _threaded_buffers(scratch, result, slot)
  local_input = view(local_scratch.input, 1:local_dof_count)
  local_output = view(local_scratch.output, 1:local_dof_count)
  item = @inbounds items[item_index]
  _gather_local_coefficients!(local_input, item, increment)
  fill!(local_output, zero(T))
  _tangent_apply_operators!(local_output, operators, item, state, local_input, apply_hook,
                            local_scratch.kernel)
  _scatter_local_vector!(local_result, item, local_output, fixed, blocked_rows)
  return nothing
end

@inline function _threaded_filtered_tangent_apply_item!(scratch::_ThreadedOperatorScratch{T},
                                                        result::AbstractVector{T}, slot::Int, items,
                                                        item_index::Int, local_dof_count::Int,
                                                        operators, operator_indices,
                                                        state::State{T},
                                                        increment::AbstractVector{T}, fixed,
                                                        blocked_rows,
                                                        apply_hook) where {T<:AbstractFloat}
  local_scratch, local_result = _threaded_buffers(scratch, result, slot)
  local_input = view(local_scratch.input, 1:local_dof_count)
  local_output = view(local_scratch.output, 1:local_dof_count)
  item = @inbounds items[item_index]
  _gather_local_coefficients!(local_input, item, increment)
  fill!(local_output, zero(T))

  for operator_index in operator_indices
    wrapped = @inbounds operators[operator_index]
    apply_hook(local_output, wrapped.operator, item, state, local_input, local_scratch.kernel)
  end

  _scatter_local_vector!(local_result, item, local_output, fixed, blocked_rows)
  return nothing
end

function _rhs_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                    batches::Vector{_KernelBatch}, items, operators, fixed, blocked_rows,
                    rhs_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_item_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_rhs_item!(scratch, result, slot, items, item_indices[batch_item_index],
                            local_dof_count, operators, fixed, blocked_rows, rhs_hook)
      end
    end
  end

  return nothing
end

function _boundary_rhs_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                             batches::Vector{_FilteredKernelBatch}, faces, operators, fixed,
                             blocked_rows) where {T<:AbstractFloat}
  isempty(batches) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    operator_indices = kernel_batch.operator_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_face_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_filtered_rhs_item!(scratch, result, slot, faces, item_indices[batch_face_index],
                                     local_dof_count, operators, operator_indices, fixed,
                                     blocked_rows, face_rhs!)
      end
    end
  end

  return nothing
end

function _surface_rhs_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                            batches::Vector{_FilteredKernelBatch}, surfaces, operators, fixed,
                            blocked_rows) where {T<:AbstractFloat}
  isempty(batches) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    operator_indices = kernel_batch.operator_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_surface_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_filtered_rhs_item!(scratch, result, slot, surfaces,
                                     item_indices[batch_surface_index], local_dof_count, operators,
                                     operator_indices, fixed, blocked_rows, surface_rhs!)
      end
    end
  end

  return nothing
end

function _apply_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                      batches::Vector{_KernelBatch}, items, operators,
                      coefficients::AbstractVector{T}, fixed, blocked_rows,
                      apply_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_item_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_apply_item!(scratch, result, slot, items, item_indices[batch_item_index],
                              local_dof_count, operators, coefficients, fixed, blocked_rows,
                              apply_hook)
      end
    end
  end

  return nothing
end

function _boundary_apply_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                               batches::Vector{_FilteredKernelBatch}, faces, operators,
                               coefficients::AbstractVector{T}, fixed,
                               blocked_rows) where {T<:AbstractFloat}
  isempty(batches) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    operator_indices = kernel_batch.operator_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_face_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_filtered_apply_item!(scratch, result, slot, faces, item_indices[batch_face_index],
                                       local_dof_count, operators, operator_indices, coefficients,
                                       fixed, blocked_rows, face_apply!)
      end
    end
  end

  return nothing
end

function _surface_apply_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                              batches::Vector{_FilteredKernelBatch}, surfaces, operators,
                              coefficients::AbstractVector{T}, fixed,
                              blocked_rows) where {T<:AbstractFloat}
  isempty(batches) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    operator_indices = kernel_batch.operator_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_surface_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_filtered_apply_item!(scratch, result, slot, surfaces,
                                       item_indices[batch_surface_index], local_dof_count,
                                       operators, operator_indices, coefficients, fixed,
                                       blocked_rows, surface_apply!)
      end
    end
  end

  return nothing
end

function _residual_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                         batches::Vector{_KernelBatch}, items, operators, state::State{T},
                         fixed::BitVector, blocked_rows::BitVector,
                         residual_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_item_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_residual_item!(scratch, result, slot, items, item_indices[batch_item_index],
                                 local_dof_count, operators, state, fixed, blocked_rows,
                                 residual_hook)
      end
    end
  end

  return nothing
end

function _boundary_residual_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                                  batches::Vector{_FilteredKernelBatch}, faces, operators,
                                  state::State{T}, fixed::BitVector,
                                  blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    operator_indices = kernel_batch.operator_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_face_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_filtered_residual_item!(scratch, result, slot, faces,
                                          item_indices[batch_face_index], local_dof_count,
                                          operators, operator_indices, state, fixed, blocked_rows,
                                          face_residual!)
      end
    end
  end

  return nothing
end

function _surface_residual_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                                 batches::Vector{_FilteredKernelBatch}, surfaces, operators,
                                 state::State{T}, fixed::BitVector,
                                 blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    operator_indices = kernel_batch.operator_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_surface_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_filtered_residual_item!(scratch, result, slot, surfaces,
                                          item_indices[batch_surface_index], local_dof_count,
                                          operators, operator_indices, state, fixed, blocked_rows,
                                          surface_residual!)
      end
    end
  end

  return nothing
end

function _tangent_apply_pass!(scratch::_ThreadedOperatorScratch{T}, result::AbstractVector{T},
                              batches::Vector{_KernelBatch}, items, operators, state::State{T},
                              increment::AbstractVector{T}, fixed::BitVector,
                              blocked_rows::BitVector, apply_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_item_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_tangent_apply_item!(scratch, result, slot, items, item_indices[batch_item_index],
                                      local_dof_count, operators, state, increment, fixed,
                                      blocked_rows, apply_hook)
      end
    end
  end

  return nothing
end

function _boundary_tangent_apply_pass!(scratch::_ThreadedOperatorScratch{T},
                                       result::AbstractVector{T},
                                       batches::Vector{_FilteredKernelBatch}, faces, operators,
                                       state::State{T}, increment::AbstractVector{T},
                                       fixed::BitVector,
                                       blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    operator_indices = kernel_batch.operator_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_face_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_filtered_tangent_apply_item!(scratch, result, slot, faces,
                                               item_indices[batch_face_index], local_dof_count,
                                               operators, operator_indices, state, increment, fixed,
                                               blocked_rows, face_tangent_apply!)
      end
    end
  end

  return nothing
end

function _surface_tangent_apply_pass!(scratch::_ThreadedOperatorScratch{T},
                                      result::AbstractVector{T},
                                      batches::Vector{_FilteredKernelBatch}, surfaces, operators,
                                      state::State{T}, increment::AbstractVector{T},
                                      fixed::BitVector,
                                      blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing
  slot_count = _threaded_slot_count(scratch)

  for kernel_batch in batches
    item_indices = kernel_batch.item_indices
    operator_indices = kernel_batch.operator_indices
    local_dof_count = kernel_batch.local_dof_count

    @_threaded_loop for slot in 1:slot_count
      for batch_surface_index in _slot_range(length(item_indices), slot, slot_count)
        _threaded_filtered_tangent_apply_item!(scratch, result, slot, surfaces,
                                               item_indices[batch_surface_index], local_dof_count,
                                               operators, operator_indices, state, increment, fixed,
                                               blocked_rows, surface_tangent_apply!)
      end
    end
  end

  return nothing
end

function _rhs_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                    batches::Vector{_KernelBatch}, items, operators, fixed, blocked_rows,
                    rhs_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing

  for kernel_batch in batches
    local_rhs = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    for batch_item_index in eachindex(kernel_batch.item_indices)
      item = @inbounds items[kernel_batch.item_indices[batch_item_index]]
      fill!(local_rhs, zero(T))

      _rhs_operators!(local_rhs, operators, item, rhs_hook, scratch.kernel)
      _scatter_local_vector!(result, item, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _boundary_rhs_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                             batches::Vector{_FilteredKernelBatch}, faces, operators, fixed,
                             blocked_rows) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_rhs = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    for batch_face_index in eachindex(kernel_batch.item_indices)
      face = @inbounds faces[kernel_batch.item_indices[batch_face_index]]
      fill!(local_rhs, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        face_rhs!(local_rhs, wrapped.operator, face, scratch.kernel)
      end

      _scatter_local_vector!(result, face, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _surface_rhs_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                            batches::Vector{_FilteredKernelBatch}, surfaces, operators, fixed,
                            blocked_rows) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_rhs = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    for batch_surface_index in eachindex(kernel_batch.item_indices)
      surface = @inbounds surfaces[kernel_batch.item_indices[batch_surface_index]]
      fill!(local_rhs, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        surface_rhs!(local_rhs, wrapped.operator, surface, scratch.kernel)
      end

      _scatter_local_vector!(result, surface, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                      batches::Vector{_KernelBatch}, items, operators,
                      coefficients::AbstractVector{T}, fixed, blocked_rows,
                      apply_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing

  for kernel_batch in batches
    local_input = view(scratch.input, 1:kernel_batch.local_dof_count)
    local_output = view(scratch.output, 1:kernel_batch.local_dof_count)

    for batch_item_index in eachindex(kernel_batch.item_indices)
      item = @inbounds items[kernel_batch.item_indices[batch_item_index]]
      _gather_local_coefficients!(local_input, item, coefficients)
      fill!(local_output, zero(T))

      _apply_operators!(local_output, operators, item, local_input, apply_hook, scratch.kernel)
      _scatter_local_vector!(result, item, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _boundary_apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                               batches::Vector{_FilteredKernelBatch}, faces, operators,
                               coefficients::AbstractVector{T}, fixed,
                               blocked_rows) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_input = view(scratch.input, 1:kernel_batch.local_dof_count)
    local_output = view(scratch.output, 1:kernel_batch.local_dof_count)

    for batch_face_index in eachindex(kernel_batch.item_indices)
      face = @inbounds faces[kernel_batch.item_indices[batch_face_index]]
      _gather_local_coefficients!(local_input, face, coefficients)
      fill!(local_output, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        face_apply!(local_output, wrapped.operator, face, local_input, scratch.kernel)
      end

      _scatter_local_vector!(result, face, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _surface_apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                              batches::Vector{_FilteredKernelBatch}, surfaces, operators,
                              coefficients::AbstractVector{T}, fixed,
                              blocked_rows) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_input = view(scratch.input, 1:kernel_batch.local_dof_count)
    local_output = view(scratch.output, 1:kernel_batch.local_dof_count)

    for batch_surface_index in eachindex(kernel_batch.item_indices)
      surface = @inbounds surfaces[kernel_batch.item_indices[batch_surface_index]]
      _gather_local_coefficients!(local_input, surface, coefficients)
      fill!(local_output, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        surface_apply!(local_output, wrapped.operator, surface, local_input, scratch.kernel)
      end

      _scatter_local_vector!(result, surface, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _diagonal_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                         batches::Vector{_KernelBatch}, items, operators,
                         diagonal_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing

  for kernel_batch in batches
    local_diagonal = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    for batch_item_index in eachindex(kernel_batch.item_indices)
      item = @inbounds items[kernel_batch.item_indices[batch_item_index]]
      fill!(local_diagonal, zero(T))

      _diagonal_operators!(local_diagonal, operators, item, diagonal_hook, scratch.kernel)
      _scatter_local_diagonal!(result, item, local_diagonal)
    end
  end

  return nothing
end

function _boundary_diagonal_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                                  batches::Vector{_FilteredKernelBatch}, faces,
                                  operators) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_diagonal = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    for batch_face_index in eachindex(kernel_batch.item_indices)
      face = @inbounds faces[kernel_batch.item_indices[batch_face_index]]
      fill!(local_diagonal, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        face_diagonal!(local_diagonal, wrapped.operator, face, scratch.kernel)
      end

      _scatter_local_diagonal!(result, face, local_diagonal)
    end
  end

  return nothing
end

function _surface_diagonal_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                                 batches::Vector{_FilteredKernelBatch}, surfaces,
                                 operators) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_diagonal = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    for batch_surface_index in eachindex(kernel_batch.item_indices)
      surface = @inbounds surfaces[kernel_batch.item_indices[batch_surface_index]]
      fill!(local_diagonal, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        surface_diagonal!(local_diagonal, wrapped.operator, surface, scratch.kernel)
      end

      _scatter_local_diagonal!(result, surface, local_diagonal)
    end
  end

  return nothing
end

function _residual_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                         batches::Vector{_KernelBatch}, items, operators, state::State{T},
                         fixed::BitVector, blocked_rows::BitVector,
                         residual_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing

  for kernel_batch in batches
    local_rhs = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    for batch_item_index in eachindex(kernel_batch.item_indices)
      item = @inbounds items[kernel_batch.item_indices[batch_item_index]]
      fill!(local_rhs, zero(T))

      _residual_operators!(local_rhs, operators, item, state, residual_hook, scratch.kernel)
      _scatter_local_vector!(result, item, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _boundary_residual_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                                  batches::Vector{_FilteredKernelBatch}, faces, operators,
                                  state::State{T}, fixed::BitVector,
                                  blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_rhs = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    for batch_face_index in eachindex(kernel_batch.item_indices)
      face = @inbounds faces[kernel_batch.item_indices[batch_face_index]]
      fill!(local_rhs, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        face_residual!(local_rhs, wrapped.operator, face, state, scratch.kernel)
      end

      _scatter_local_vector!(result, face, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _surface_residual_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                                 batches::Vector{_FilteredKernelBatch}, surfaces, operators,
                                 state::State{T}, fixed::BitVector,
                                 blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_rhs = view(scratch.rhs, 1:kernel_batch.local_dof_count)

    for batch_surface_index in eachindex(kernel_batch.item_indices)
      surface = @inbounds surfaces[kernel_batch.item_indices[batch_surface_index]]
      fill!(local_rhs, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        surface_residual!(local_rhs, wrapped.operator, surface, state, scratch.kernel)
      end

      _scatter_local_vector!(result, surface, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _tangent_apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                              batches::Vector{_KernelBatch}, items, operators, state::State{T},
                              increment::AbstractVector{T}, fixed::BitVector,
                              blocked_rows::BitVector, apply_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing

  for kernel_batch in batches
    local_input = view(scratch.input, 1:kernel_batch.local_dof_count)
    local_output = view(scratch.output, 1:kernel_batch.local_dof_count)

    for batch_item_index in eachindex(kernel_batch.item_indices)
      item = @inbounds items[kernel_batch.item_indices[batch_item_index]]
      _gather_local_coefficients!(local_input, item, increment)
      fill!(local_output, zero(T))

      _tangent_apply_operators!(local_output, operators, item, state, local_input, apply_hook,
                                scratch.kernel)
      _scatter_local_vector!(result, item, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _boundary_tangent_apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                                       batches::Vector{_FilteredKernelBatch}, faces, operators,
                                       state::State{T}, increment::AbstractVector{T},
                                       fixed::BitVector,
                                       blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_input = view(scratch.input, 1:kernel_batch.local_dof_count)
    local_output = view(scratch.output, 1:kernel_batch.local_dof_count)

    for batch_face_index in eachindex(kernel_batch.item_indices)
      face = @inbounds faces[kernel_batch.item_indices[batch_face_index]]
      _gather_local_coefficients!(local_input, face, increment)
      fill!(local_output, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        face_tangent_apply!(local_output, wrapped.operator, face, state, local_input,
                            scratch.kernel)
      end

      _scatter_local_vector!(result, face, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _surface_tangent_apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                                      batches::Vector{_FilteredKernelBatch}, surfaces, operators,
                                      state::State{T}, increment::AbstractVector{T},
                                      fixed::BitVector,
                                      blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for kernel_batch in batches
    local_input = view(scratch.input, 1:kernel_batch.local_dof_count)
    local_output = view(scratch.output, 1:kernel_batch.local_dof_count)

    for batch_surface_index in eachindex(kernel_batch.item_indices)
      surface = @inbounds surfaces[kernel_batch.item_indices[batch_surface_index]]
      _gather_local_coefficients!(local_input, surface, increment)
      fill!(local_output, zero(T))

      for operator_index in kernel_batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        surface_tangent_apply!(local_output, wrapped.operator, surface, state, local_input,
                               scratch.kernel)
      end

      _scatter_local_vector!(result, surface, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

# Gather/scatter kernels.

function _gather_local_coefficients!(local_values::AbstractVector{T}, item::_AssemblyValues,
                                     coefficients::AbstractVector{T}) where {T<:AbstractFloat}
  for local_dof in 1:item.local_dof_count
    local_values[local_dof] = _local_coefficient(item, coefficients, local_dof)
  end

  return local_values
end

@inline function _local_coefficient(item::_AssemblyValues, coefficients::AbstractVector{T},
                                    local_dof::Int) where {T<:AbstractFloat}
  single_index = item.single_term_indices[local_dof]
  single_index != 0 && return item.single_term_coefficients[local_dof] * coefficients[single_index]
  value = zero(T)

  @inbounds for term_index in _local_term_range(item, local_dof)
    value += item.term_coefficients[term_index] * coefficients[item.term_indices[term_index]]
  end

  return value
end

function _scatter_local_vector!(result::AbstractVector{T}, item::_AssemblyValues,
                                local_vector::AbstractVector{T}) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for local_row in 1:item.local_dof_count
    value = local_vector[local_row]
    abs(value) > tolerance || continue

    @inbounds for term_index in _local_term_range(item, local_row)
      row = item.term_indices[term_index]
      result[row] += item.term_coefficients[term_index] * value
    end
  end

  return nothing
end

function _scatter_local_vector!(result::AbstractVector{T}, item::_AssemblyValues,
                                local_vector::AbstractVector{T}, fixed::BitVector,
                                blocked_rows::BitVector) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for local_row in 1:item.local_dof_count
    value = local_vector[local_row]
    abs(value) > tolerance || continue

    @inbounds for term_index in _local_term_range(item, local_row)
      row = item.term_indices[term_index]
      (fixed[row] || blocked_rows[row]) && continue
      result[row] += item.term_coefficients[term_index] * value
    end
  end

  return nothing
end

function _scatter_local_vector!(result::AbstractVector{T}, item::_AssemblyValues,
                                local_vector::AbstractVector{T}, ::Nothing,
                                ::Nothing) where {T<:AbstractFloat}
  return _scatter_local_vector!(result, item, local_vector)
end

function _scatter_local_diagonal!(result::AbstractVector{T}, item::_AssemblyValues,
                                  local_diagonal::AbstractVector{T}) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for local_dof in 1:item.local_dof_count
    value = local_diagonal[local_dof]
    abs(value) > tolerance || continue

    @inbounds for term_index in _local_term_range(item, local_dof)
      global_dof = item.term_indices[term_index]
      coefficient = item.term_coefficients[term_index]
      result[global_dof] += coefficient * coefficient * value
    end
  end

  return nothing
end

# Constraint rows.

function _write_constraint_rhs!(result::AbstractVector{T},
                                plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  for index in eachindex(plan.dirichlet.fixed_dofs)
    result[plan.dirichlet.fixed_dofs[index]] = plan.dirichlet.fixed_values[index]
  end

  for row in plan.dirichlet.rows
    result[row.pivot] = row.rhs
  end

  for constraint in plan.mean_constraints
    result[constraint.pivot] = constraint.rhs
  end

  return result
end

function _write_constraint_action!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                                   coefficients::AbstractVector{T}) where {D,T<:AbstractFloat}
  for dof in plan.dirichlet.fixed_dofs
    result[dof] = coefficients[dof]
  end

  for row in plan.dirichlet.rows
    value = coefficients[row.pivot]

    for index in eachindex(row.indices)
      value -= row.coefficients[index] * coefficients[row.indices[index]]
    end

    result[row.pivot] = value
  end

  for constraint in plan.mean_constraints
    value = zero(T)

    for index in eachindex(constraint.indices)
      value += constraint.coefficients[index] * coefficients[constraint.indices[index]]
    end

    result[constraint.pivot] = value
  end

  return result
end

function _write_constraint_residual!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                                     coefficients::AbstractVector{T}) where {D,T<:AbstractFloat}
  for index in eachindex(plan.dirichlet.fixed_dofs)
    dof = plan.dirichlet.fixed_dofs[index]
    result[dof] = coefficients[dof] - plan.dirichlet.fixed_values[index]
  end

  for row in plan.dirichlet.rows
    value = coefficients[row.pivot] - row.rhs

    for index in eachindex(row.indices)
      value -= row.coefficients[index] * coefficients[row.indices[index]]
    end

    result[row.pivot] = value
  end

  for constraint in plan.mean_constraints
    value = -constraint.rhs

    for index in eachindex(constraint.indices)
      value += constraint.coefficients[index] * coefficients[constraint.indices[index]]
    end

    result[constraint.pivot] = value
  end

  return result
end

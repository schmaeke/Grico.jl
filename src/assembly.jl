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

struct _MatrixFreeAssemblyStructure{K} end

function _compile_assembly_structure(::Val{K}, layout, cell_operators, boundary_operators,
                                     interface_operators, surface_operators, integration,
                                     dirichlet, mean_constraints, constraint_masks) where {K}
  return _MatrixFreeAssemblyStructure{K}()
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
end

function _OperatorScratch(::Type{T}, local_dof_count::Int) where {T<:AbstractFloat}
  count = max(local_dof_count, 0)
  return _OperatorScratch(zeros(T, count), zeros(T, count), zeros(T, count))
end

function _scratch_buffer(::Type{T}, integration::_CompiledIntegration) where {T<:AbstractFloat}
  return _OperatorScratch(T, _max_local_dof_count(integration))
end

"""
    ResidualWorkspace(plan)

Reusable runtime storage for repeated nonlinear residual and tangent-action
evaluations on `plan`.

A workspace is tied to the exact plan used to create it.
"""
struct ResidualWorkspace{T<:AbstractFloat,P<:AssemblyPlan}
  plan::P
  scratch::_OperatorScratch{T}
end

function ResidualWorkspace(plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  return ResidualWorkspace{T,typeof(plan)}(plan, _scratch_buffer(T, plan.integration))
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

function rhs!(result::AbstractVector{T}, plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  _require_length(result, dof_count(plan), "result")
  scratch = _scratch_buffer(T, plan.integration)
  fill!(result, zero(T))
  traversal = plan.traversal_plan
  masks = plan.constraint_masks
  _rhs_pass!(scratch, result, traversal.cell_batches, plan.integration.cells,
             plan.cell_operators, masks.fixed, masks.blocked_rows, cell_rhs!)
  _boundary_rhs_pass!(scratch, result, traversal.boundary_batches,
                      plan.integration.boundary_faces, plan.boundary_operators, masks.fixed,
                      masks.blocked_rows)
  _rhs_pass!(scratch, result, traversal.interface_batches, plan.integration.interfaces,
             plan.interface_operators, masks.fixed, masks.blocked_rows, interface_rhs!)
  _surface_rhs_pass!(scratch, result, traversal.surface_batches,
                     plan.integration.embedded_surfaces, plan.surface_operators, masks.fixed,
                     masks.blocked_rows)
  _write_constraint_rhs!(result, plan)
  return result
end

# Affine matrix-free action.

"""
    apply(plan, coefficients)
    apply!(result, plan, coefficients)

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

function apply!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                coefficients::AbstractVector{T}) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  _require_length(result, dof_count(plan), "result")
  _require_length(coefficients, dof_count(plan), "coefficient vector")
  _check_no_alias(result, coefficients)
  scratch = _scratch_buffer(T, plan.integration)
  fill!(result, zero(T))
  traversal = plan.traversal_plan
  masks = plan.constraint_masks
  _apply_pass!(scratch, result, traversal.cell_batches, plan.integration.cells,
               plan.cell_operators, coefficients, masks.fixed, masks.blocked_rows, cell_apply!)
  _boundary_apply_pass!(scratch, result, traversal.boundary_batches,
                        plan.integration.boundary_faces, plan.boundary_operators, coefficients,
                        masks.fixed, masks.blocked_rows)
  _apply_pass!(scratch, result, traversal.interface_batches, plan.integration.interfaces,
               plan.interface_operators, coefficients, masks.fixed, masks.blocked_rows,
               interface_apply!)
  _surface_apply_pass!(scratch, result, traversal.surface_batches,
                       plan.integration.embedded_surfaces, plan.surface_operators, coefficients,
                       masks.fixed, masks.blocked_rows)
  _write_constraint_action!(result, plan, coefficients)
  return result
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
  fill!(result, zero(T))
  traversal = plan.traversal_plan
  masks = plan.constraint_masks
  scratch = workspace.scratch
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
  fill!(result, zero(T))
  traversal = plan.traversal_plan
  masks = plan.constraint_masks
  scratch = workspace.scratch
  _tangent_apply_pass!(scratch, result, traversal.cell_batches, plan.integration.cells,
                       plan.cell_operators, state, increment, masks.fixed, masks.blocked_rows,
                       cell_tangent_apply!)
  _boundary_tangent_apply_pass!(scratch, result, traversal.boundary_batches,
                                plan.integration.boundary_faces, plan.boundary_operators, state,
                                increment, masks.fixed, masks.blocked_rows)
  _tangent_apply_pass!(scratch, result, traversal.interface_batches,
                       plan.integration.interfaces, plan.interface_operators, state, increment,
                       masks.fixed, masks.blocked_rows, interface_tangent_apply!)
  _surface_tangent_apply_pass!(scratch, result, traversal.surface_batches,
                               plan.integration.embedded_surfaces, plan.surface_operators, state,
                               increment, masks.fixed, masks.blocked_rows)
  _write_constraint_action!(result, plan, increment)
  return result
end

# Local traversal helpers.

function _rhs_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                    batches::Vector{_KernelBatch}, items, operators, fixed::BitVector,
                    blocked_rows::BitVector, rhs_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing

  for batch in batches
    local_rhs = view(scratch.rhs, 1:batch.local_dof_count)

    for batch_item_index in eachindex(batch.item_indices)
      item = @inbounds items[batch.item_indices[batch_item_index]]
      fill!(local_rhs, zero(T))

      for operator in operators
        rhs_hook(local_rhs, operator, item)
      end

      _scatter_local_vector!(result, item, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _boundary_rhs_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                             batches::Vector{_FilteredKernelBatch}, faces, operators,
                             fixed::BitVector, blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for batch in batches
    local_rhs = view(scratch.rhs, 1:batch.local_dof_count)

    for batch_face_index in eachindex(batch.item_indices)
      face = @inbounds faces[batch.item_indices[batch_face_index]]
      fill!(local_rhs, zero(T))

      for operator_index in batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        face_rhs!(local_rhs, wrapped.operator, face)
      end

      _scatter_local_vector!(result, face, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _surface_rhs_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                            batches::Vector{_FilteredKernelBatch}, surfaces, operators,
                            fixed::BitVector, blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for batch in batches
    local_rhs = view(scratch.rhs, 1:batch.local_dof_count)

    for batch_surface_index in eachindex(batch.item_indices)
      surface = @inbounds surfaces[batch.item_indices[batch_surface_index]]
      fill!(local_rhs, zero(T))

      for operator_index in batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        surface_rhs!(local_rhs, wrapped.operator, surface)
      end

      _scatter_local_vector!(result, surface, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                      batches::Vector{_KernelBatch}, items, operators,
                      coefficients::AbstractVector{T}, fixed::BitVector,
                      blocked_rows::BitVector, apply_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing

  for batch in batches
    local_input = view(scratch.input, 1:batch.local_dof_count)
    local_output = view(scratch.output, 1:batch.local_dof_count)

    for batch_item_index in eachindex(batch.item_indices)
      item = @inbounds items[batch.item_indices[batch_item_index]]
      _gather_local_coefficients!(local_input, item, coefficients)
      fill!(local_output, zero(T))

      for operator in operators
        apply_hook(local_output, operator, item, local_input)
      end

      _scatter_local_vector!(result, item, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _boundary_apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                               batches::Vector{_FilteredKernelBatch}, faces, operators,
                               coefficients::AbstractVector{T}, fixed::BitVector,
                               blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for batch in batches
    local_input = view(scratch.input, 1:batch.local_dof_count)
    local_output = view(scratch.output, 1:batch.local_dof_count)

    for batch_face_index in eachindex(batch.item_indices)
      face = @inbounds faces[batch.item_indices[batch_face_index]]
      _gather_local_coefficients!(local_input, face, coefficients)
      fill!(local_output, zero(T))

      for operator_index in batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        face_apply!(local_output, wrapped.operator, face, local_input)
      end

      _scatter_local_vector!(result, face, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _surface_apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                              batches::Vector{_FilteredKernelBatch}, surfaces, operators,
                              coefficients::AbstractVector{T}, fixed::BitVector,
                              blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for batch in batches
    local_input = view(scratch.input, 1:batch.local_dof_count)
    local_output = view(scratch.output, 1:batch.local_dof_count)

    for batch_surface_index in eachindex(batch.item_indices)
      surface = @inbounds surfaces[batch.item_indices[batch_surface_index]]
      _gather_local_coefficients!(local_input, surface, coefficients)
      fill!(local_output, zero(T))

      for operator_index in batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        surface_apply!(local_output, wrapped.operator, surface, local_input)
      end

      _scatter_local_vector!(result, surface, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _residual_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                         batches::Vector{_KernelBatch}, items, operators, state::State{T},
                         fixed::BitVector, blocked_rows::BitVector,
                         residual_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing

  for batch in batches
    local_rhs = view(scratch.rhs, 1:batch.local_dof_count)

    for batch_item_index in eachindex(batch.item_indices)
      item = @inbounds items[batch.item_indices[batch_item_index]]
      fill!(local_rhs, zero(T))

      for operator in operators
        residual_hook(local_rhs, operator, item, state)
      end

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

  for batch in batches
    local_rhs = view(scratch.rhs, 1:batch.local_dof_count)

    for batch_face_index in eachindex(batch.item_indices)
      face = @inbounds faces[batch.item_indices[batch_face_index]]
      fill!(local_rhs, zero(T))

      for operator_index in batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        face_residual!(local_rhs, wrapped.operator, face, state)
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

  for batch in batches
    local_rhs = view(scratch.rhs, 1:batch.local_dof_count)

    for batch_surface_index in eachindex(batch.item_indices)
      surface = @inbounds surfaces[batch.item_indices[batch_surface_index]]
      fill!(local_rhs, zero(T))

      for operator_index in batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        surface_residual!(local_rhs, wrapped.operator, surface, state)
      end

      _scatter_local_vector!(result, surface, local_rhs, fixed, blocked_rows)
    end
  end

  return nothing
end

function _tangent_apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                              batches::Vector{_KernelBatch}, items, operators,
                              state::State{T}, increment::AbstractVector{T},
                              fixed::BitVector, blocked_rows::BitVector,
                              apply_hook) where {T<:AbstractFloat}
  (isempty(batches) || isempty(operators)) && return nothing

  for batch in batches
    local_input = view(scratch.input, 1:batch.local_dof_count)
    local_output = view(scratch.output, 1:batch.local_dof_count)

    for batch_item_index in eachindex(batch.item_indices)
      item = @inbounds items[batch.item_indices[batch_item_index]]
      _gather_local_coefficients!(local_input, item, increment)
      fill!(local_output, zero(T))

      for operator in operators
        apply_hook(local_output, operator, item, state, local_input)
      end

      _scatter_local_vector!(result, item, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _boundary_tangent_apply_pass!(scratch::_OperatorScratch{T},
                                       result::AbstractVector{T},
                                       batches::Vector{_FilteredKernelBatch}, faces, operators,
                                       state::State{T}, increment::AbstractVector{T},
                                       fixed::BitVector,
                                       blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for batch in batches
    local_input = view(scratch.input, 1:batch.local_dof_count)
    local_output = view(scratch.output, 1:batch.local_dof_count)

    for batch_face_index in eachindex(batch.item_indices)
      face = @inbounds faces[batch.item_indices[batch_face_index]]
      _gather_local_coefficients!(local_input, face, increment)
      fill!(local_output, zero(T))

      for operator_index in batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        face_tangent_apply!(local_output, wrapped.operator, face, state, local_input)
      end

      _scatter_local_vector!(result, face, local_output, fixed, blocked_rows)
    end
  end

  return nothing
end

function _surface_tangent_apply_pass!(scratch::_OperatorScratch{T}, result::AbstractVector{T},
                                      batches::Vector{_FilteredKernelBatch}, surfaces,
                                      operators, state::State{T},
                                      increment::AbstractVector{T}, fixed::BitVector,
                                      blocked_rows::BitVector) where {T<:AbstractFloat}
  isempty(batches) && return nothing

  for batch in batches
    local_input = view(scratch.input, 1:batch.local_dof_count)
    local_output = view(scratch.output, 1:batch.local_dof_count)

    for batch_surface_index in eachindex(batch.item_indices)
      surface = @inbounds surfaces[batch.item_indices[batch_surface_index]]
      _gather_local_coefficients!(local_input, surface, increment)
      fill!(local_output, zero(T))

      for operator_index in batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        surface_tangent_apply!(local_output, wrapped.operator, surface, state, local_input)
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
  single_index != 0 &&
    return item.single_term_coefficients[local_dof] * coefficients[single_index]
  value = zero(T)

  @inbounds for term_index in _local_term_range(item, local_dof)
    value += item.term_coefficients[term_index] * coefficients[item.term_indices[term_index]]
  end

  return value
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

# This file owns the solve API on the matrix-free branch.
#
# The sparse direct/Krylov/preconditioner stack has intentionally been removed.
# A matrix-free Krylov implementation will be added here once the operator
# action and lightweight preconditioning contract are settled.

"""
    JacobiPreconditioner()

Configuration for generic diagonal Jacobi preconditioning.

When all affine operators provide compatible local diagonal kernels, the
preconditioner builds the reduced diagonal directly from those kernels. It
falls back to exact reduced-operator probing for constraint maps or operator
sets that need the more general path.
"""
struct JacobiPreconditioner end

function default_linear_solve(plan::AssemblyPlan{D,T}, reduced_rhs::AbstractVector{T};
                              workspace=_ReducedOperatorWorkspace(plan), preconditioner=nothing,
                              relative_tolerance=sqrt(eps(T)), absolute_tolerance=zero(T),
                              maxiter=max(1_000, 2 * length(reduced_rhs)),
                              initial_solution=nothing) where {D,T<:AbstractFloat}
  inverse_diagonal = _preconditioner_data(plan, workspace, preconditioner)
  return _cg_solve(plan, reduced_rhs, workspace; relative_tolerance=relative_tolerance,
                   absolute_tolerance=absolute_tolerance, maxiter=maxiter,
                   initial_solution=initial_solution, inverse_diagonal=inverse_diagonal)
end

"""
    solve(problem; linear_solve=default_linear_solve, preconditioner=nothing, kwargs...)
    solve(plan; linear_solve=default_linear_solve, preconditioner=nothing, kwargs...)

Solve a compiled affine problem with a user-supplied matrix-free linear solver.

`linear_solve` is called on the reduced constraint-compatible system as

    linear_solve(plan, reduced_rhs; preconditioner=preconditioner, workspace=workspace, kwargs...)

and must return a reduced coefficient vector. The returned vector is expanded
back to a full-layout [`State`](@ref). The package default solve path is
intentionally a stub on this branch until the matrix-free Krylov and
preconditioning design is implemented.
"""
function solve(problem::AffineProblem; linear_solve=default_linear_solve,
               preconditioner=nothing, kwargs...)
  return solve(compile(problem); linear_solve=linear_solve, preconditioner=preconditioner,
               kwargs...)
end

function solve(plan::AssemblyPlan{D,T}; linear_solve=default_linear_solve,
               preconditioner=nothing, kwargs...) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  workspace = _ReducedOperatorWorkspace(plan)
  reduced_rhs = zeros(T, reduced_dof_count(plan))
  _reduced_rhs!(reduced_rhs, plan, workspace)
  reduced_values = linear_solve(plan, reduced_rhs; preconditioner=preconditioner,
                                workspace=workspace, kwargs...)
  _require_length(reduced_values, reduced_dof_count(plan), "linear solve result")
  eltype(reduced_values) == T ||
    throw(ArgumentError("linear solve result element type must match the plan scalar type"))
  full_values = zeros(T, dof_count(plan))
  _reconstruct_reduced_solution!(full_values, plan, reduced_values)
  return State(plan, full_values)
end

function _cg_solve(plan::AssemblyPlan{D,T}, rhs_data::AbstractVector{T},
                   workspace::_ReducedOperatorWorkspace{T}; relative_tolerance::T,
                   absolute_tolerance::T, maxiter::Int,
                   initial_solution, inverse_diagonal) where {D,T<:AbstractFloat}
  n = length(rhs_data)
  n == 0 && return T[]
  maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
  solution = zeros(T, n)

  if initial_solution !== nothing
    _require_length(initial_solution, n, "initial solution")
    eltype(initial_solution) == T ||
      throw(ArgumentError("initial solution element type must match the plan scalar type"))
    copyto!(solution, initial_solution)
  end

  residual = copy(rhs_data)
  operator_values = zeros(T, n)

  if initial_solution !== nothing
    _reduced_apply!(operator_values, plan, solution, workspace)
    _axpy!(residual, -one(T), operator_values)
  end

  preconditioned_residual = similar(residual)
  _apply_preconditioner!(preconditioned_residual, residual, inverse_diagonal)
  direction = copy(preconditioned_residual)
  residual_inner = _dot(residual, preconditioned_residual)
  rhs_norm = sqrt(_dot_self(rhs_data))
  tolerance = max(absolute_tolerance, relative_tolerance * max(rhs_norm, one(T)))
  residual_norm2 = _dot_self(residual)
  residual_norm2 <= tolerance * tolerance && return solution

  for iteration in 1:maxiter
    _reduced_apply!(operator_values, plan, direction, workspace)
    denominator = _dot(direction, operator_values)
    denominator > zero(T) ||
      throw(ArgumentError("CG encountered a non-positive operator direction; provide a different linear_solve for this problem"))
    alpha = residual_inner / denominator
    _axpy!(solution, alpha, direction)
    _axpy!(residual, -alpha, operator_values)
    next_residual_norm2 = _dot_self(residual)
    next_residual_norm2 <= tolerance * tolerance && return solution
    _apply_preconditioner!(preconditioned_residual, residual, inverse_diagonal)
    next_residual_inner = _dot(residual, preconditioned_residual)
    beta = next_residual_inner / residual_inner
    _update_direction!(direction, preconditioned_residual, beta)
    residual_norm2 = next_residual_norm2
    residual_inner = next_residual_inner
  end

  throw(ArgumentError("CG did not converge in $maxiter iterations"))
end

_preconditioner_data(plan, workspace, ::Nothing) = nothing

function _preconditioner_data(plan::AssemblyPlan{D,T},
                              workspace::_ReducedOperatorWorkspace{T},
                              ::JacobiPreconditioner) where {D,T<:AbstractFloat}
  return _jacobi_inverse_diagonal(plan, workspace)
end

function _preconditioner_data(plan, workspace, preconditioner)
  throw(ArgumentError("unsupported matrix-free preconditioner $(typeof(preconditioner))"))
end

function _jacobi_inverse_diagonal(plan::AssemblyPlan{D,T},
                                  workspace::_ReducedOperatorWorkspace{T}) where {D,
                                                                                  T<:AbstractFloat}
  n = reduced_dof_count(plan)
  inverse_diagonal = zeros(T, n)

  if _reduced_diagonal!(inverse_diagonal, plan, workspace)
    return _invert_jacobi_diagonal!(inverse_diagonal)
  end

  return _probe_jacobi_inverse_diagonal(plan, workspace)
end

function _probe_jacobi_inverse_diagonal(plan::AssemblyPlan{D,T},
                                        workspace::_ReducedOperatorWorkspace{T}) where {D,
                                                                                        T<:AbstractFloat}
  n = reduced_dof_count(plan)
  inverse_diagonal = Vector{T}(undef, n)
  basis = zeros(T, n)
  response = zeros(T, n)

  for index in 1:n
    basis[index] = one(T)
    _reduced_apply!(response, plan, basis, workspace)
    inverse_diagonal[index] = response[index]
    basis[index] = zero(T)
  end

  return _invert_jacobi_diagonal!(inverse_diagonal)
end

function _invert_jacobi_diagonal!(diagonal::AbstractVector{T}) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for index in eachindex(diagonal)
    value = diagonal[index]
    abs(value) > tolerance ||
      throw(ArgumentError("Jacobi preconditioner found a near-zero diagonal entry at reduced dof $index"))
    diagonal[index] = inv(value)
  end

  return diagonal
end

function _apply_preconditioner!(result::AbstractVector{T}, residual::AbstractVector{T},
                                ::Nothing) where {T<:AbstractFloat}
  _require_length(result, length(residual), "preconditioned residual")
  copyto!(result, residual)
  return result
end

function _apply_preconditioner!(result::AbstractVector{T}, residual::AbstractVector{T},
                                inverse_diagonal::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(result, length(residual), "preconditioned residual")
  _require_length(inverse_diagonal, length(residual), "inverse diagonal")

  @inbounds for index in eachindex(residual)
    result[index] = inverse_diagonal[index] * residual[index]
  end

  return result
end

function _dot(first::AbstractVector{T}, second::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(second, length(first), "second vector")
  result = zero(T)

  @inbounds for index in eachindex(first)
    result += first[index] * second[index]
  end

  return result
end

function _dot_self(values::AbstractVector{T}) where {T<:AbstractFloat}
  result = zero(T)

  @inbounds for value in values
    result += value * value
  end

  return result
end

function _axpy!(target::AbstractVector{T}, scale::T,
                source::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(source, length(target), "source vector")

  @inbounds for index in eachindex(target)
    target[index] += scale * source[index]
  end

  return target
end

function _update_direction!(direction::AbstractVector{T}, residual::AbstractVector{T},
                            beta::T) where {T<:AbstractFloat}
  _require_length(residual, length(direction), "residual")

  @inbounds for index in eachindex(direction)
    direction[index] = residual[index] + beta * direction[index]
  end

  return direction
end

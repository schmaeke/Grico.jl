# This file owns the solve API on the matrix-free branch.
#
# The sparse direct/Krylov/preconditioner stack has intentionally been removed.
# A matrix-free Krylov implementation will be added here once the operator
# action and lightweight preconditioning contract are settled.

"""
    JacobiPreconditioner()

Placeholder configuration for the first matrix-free preconditioner.

The implementation is not wired yet; the type exists so solver experiments can
settle on a public keyword/API without reintroducing the old sparse
preconditioner hierarchy.
"""
struct JacobiPreconditioner end

function default_linear_solve(plan::AssemblyPlan{D,T}, reduced_rhs::AbstractVector{T};
                              workspace=_ReducedOperatorWorkspace(plan), preconditioner=nothing,
                              relative_tolerance=sqrt(eps(T)), absolute_tolerance=zero(T),
                              maxiter=max(1_000, 2 * length(reduced_rhs)),
                              initial_solution=nothing) where {D,T<:AbstractFloat}
  preconditioner === nothing ||
    throw(ArgumentError("matrix-free preconditioner $(typeof(preconditioner)) is not implemented yet"))
  return _cg_solve(plan, reduced_rhs, workspace; relative_tolerance=relative_tolerance,
                   absolute_tolerance=absolute_tolerance, maxiter=maxiter,
                   initial_solution=initial_solution)
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
                   initial_solution) where {D,T<:AbstractFloat}
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

  direction = copy(residual)
  residual_norm2 = _dot_self(residual)
  rhs_norm = sqrt(_dot_self(rhs_data))
  tolerance = max(absolute_tolerance, relative_tolerance * max(rhs_norm, one(T)))
  residual_norm2 <= tolerance * tolerance && return solution

  for iteration in 1:maxiter
    _reduced_apply!(operator_values, plan, direction, workspace)
    denominator = _dot(direction, operator_values)
    denominator > zero(T) ||
      throw(ArgumentError("CG encountered a non-positive operator direction; provide a different linear_solve for this problem"))
    alpha = residual_norm2 / denominator
    _axpy!(solution, alpha, direction)
    _axpy!(residual, -alpha, operator_values)
    next_residual_norm2 = _dot_self(residual)
    next_residual_norm2 <= tolerance * tolerance && return solution
    beta = next_residual_norm2 / residual_norm2
    _update_direction!(direction, residual, beta)
    residual_norm2 = next_residual_norm2
  end

  throw(ArgumentError("CG did not converge in $maxiter iterations"))
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

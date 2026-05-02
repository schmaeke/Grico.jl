# This file owns the solve API on the matrix-free branch.
#
# Affine solves run CG on the reduced matrix-free operator. Residual solves run
# Newton on the reduced nonlinear equations and use the same CG kernel for
# matrix-free tangent corrections.

"""
    JacobiPreconditioner()

Configuration for generic diagonal Jacobi preconditioning.

For affine solves, the preconditioner builds the reduced diagonal directly from
local diagonal kernels when every operator contribution can be scattered
safely. It uses identity preconditioning when those kernels are unavailable or
incompatible with the reduced operator map. The default residual tangent solve
does not implement tangent preconditioning; nonlinear users who need one should
provide a custom `linear_solve`.
"""
struct JacobiPreconditioner end

"""
    default_linear_solve(plan, reduced_rhs; workspace=..., preconditioner=nothing, kwargs...)

Default matrix-free CG solve for affine reduced systems.

This is the stable reference implementation for the `linear_solve` callback
used by affine [`solve`](@ref). It accepts the same keyword contract that custom
affine solvers should support: a reusable reduced-operator workspace,
optional preconditioner, tolerances, iteration limit, and optional initial
solution.
"""
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

Solve a compiled affine or residual problem.

For affine problems, `linear_solve` is called on the reduced constraint-
compatible system as

    linear_solve(plan, reduced_rhs; preconditioner=preconditioner, workspace=workspace, kwargs...)

and must return a reduced coefficient vector. The returned vector is expanded
back to a full-layout [`State`](@ref).

For residual problems, the same keyword names configure a Newton solve. The
default tangent correction uses matrix-free CG on `P' J(u) P` without
preconditioning. Passing a non-`nothing` preconditioner to the default residual
solve is rejected so nonlinear preconditioning remains an explicit custom
`linear_solve` decision.
"""
function solve(problem::AffineProblem; linear_solve=default_linear_solve, preconditioner=nothing,
               kwargs...)
  return solve(compile(problem); linear_solve=linear_solve, preconditioner=preconditioner,
               kwargs...)
end

function solve(problem::ResidualProblem; linear_solve=default_tangent_linear_solve,
               preconditioner=nothing, kwargs...)
  return solve(compile(problem); linear_solve=linear_solve, preconditioner=preconditioner,
               kwargs...)
end

function solve(plan::AssemblyPlan{D,T}; linear_solve=default_linear_solve, preconditioner=nothing,
               kwargs...) where {D,T<:AbstractFloat}
  kind = _matrix_free_kind(plan.assembly_structure)
  kind === :affine && return _solve_affine(plan; linear_solve, preconditioner, kwargs...)
  residual_linear_solve = linear_solve === default_linear_solve ? default_tangent_linear_solve :
                          linear_solve
  kind === :residual &&
    return _solve_residual(plan; linear_solve=residual_linear_solve, preconditioner, kwargs...)
  throw(ArgumentError("unsupported matrix-free solve kind $kind"))
end

function _solve_affine(plan::AssemblyPlan{D,T}; linear_solve=default_linear_solve,
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

function _solve_residual(plan::AssemblyPlan{D,T}; linear_solve=default_tangent_linear_solve,
                         preconditioner=nothing, initial_state=nothing,
                         relative_tolerance=sqrt(eps(T)), absolute_tolerance=sqrt(eps(T)),
                         maxiter::Int=20, damping=one(T), linear_relative_tolerance=sqrt(eps(T)),
                         linear_absolute_tolerance=zero(T),
                         linear_maxiter=max(1_000, 2 * reduced_dof_count(plan))) where {D,
                                                                                        T<:AbstractFloat}
  _require_matrix_free_kind(plan, :residual)
  maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
  damping_value = T(damping)
  damping_value > zero(T) || throw(ArgumentError("damping must be positive"))
  map = _reduced_map(plan)
  workspace = _ReducedOperatorWorkspace(plan)
  residual_workspace = ResidualWorkspace(plan)
  reduced_values = zeros(T, reduced_dof_count(map))

  if initial_state !== nothing
    _check_state(plan, initial_state)
    _compress_reduced!(reduced_values, map, coefficients(initial_state))
  end

  reduced_residual = zeros(T, reduced_dof_count(map))
  correction_rhs = zeros(T, reduced_dof_count(map))
  initial_residual_norm = nothing

  for _ in 1:maxiter
    _reduced_residual!(reduced_residual, plan, reduced_values, workspace, residual_workspace)
    residual_norm = sqrt(_dot_self(reduced_residual))
    initial_residual_norm === nothing && (initial_residual_norm = max(residual_norm, one(T)))
    tolerance = max(T(absolute_tolerance), T(relative_tolerance) * initial_residual_norm)
    residual_norm <= tolerance && return _state_from_reduced!(plan, workspace, reduced_values)

    @inbounds for index in eachindex(reduced_residual)
      correction_rhs[index] = -reduced_residual[index]
    end

    correction = linear_solve(plan, workspace.state, correction_rhs; workspace, residual_workspace,
                              preconditioner, relative_tolerance=T(linear_relative_tolerance),
                              absolute_tolerance=T(linear_absolute_tolerance),
                              maxiter=linear_maxiter)
    _require_length(correction, reduced_dof_count(map), "Newton correction")
    eltype(correction) == T ||
      throw(ArgumentError("Newton correction element type must match the plan scalar type"))

    @inbounds for index in eachindex(reduced_values)
      reduced_values[index] += damping_value * correction[index]
    end
  end

  _reduced_residual!(reduced_residual, plan, reduced_values, workspace, residual_workspace)
  residual_norm = sqrt(_dot_self(reduced_residual))
  baseline = initial_residual_norm === nothing ? one(T) : initial_residual_norm
  tolerance = max(T(absolute_tolerance), T(relative_tolerance) * baseline)
  residual_norm <= tolerance && return _state_from_reduced!(plan, workspace, reduced_values)
  throw(ArgumentError("Newton solve did not converge in $maxiter iterations"))
end

function _state_from_reduced!(plan::AssemblyPlan{D,T}, workspace::_ReducedOperatorWorkspace{T},
                              reduced_values::AbstractVector{T}) where {D,T<:AbstractFloat}
  full_values = zeros(T, dof_count(plan))
  _reconstruct_reduced_solution!(full_values, plan, reduced_values)
  return State(plan, full_values)
end

function _cg_solve(plan::AssemblyPlan{D,T}, rhs_data::AbstractVector{T},
                   workspace::_ReducedOperatorWorkspace{T}; relative_tolerance::T,
                   absolute_tolerance::T, maxiter::Int, initial_solution, inverse_diagonal,
                   operator_apply=(target, vector) -> _reduced_apply!(target, plan, vector,
                                                                      workspace)) where {D,
                                                                                         T<:AbstractFloat}
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
    operator_apply(operator_values, solution)
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
    operator_apply(operator_values, direction)
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

function _preconditioner_data(plan::AssemblyPlan{D,T}, workspace::_ReducedOperatorWorkspace{T},
                              ::JacobiPreconditioner) where {D,T<:AbstractFloat}
  return _jacobi_inverse_diagonal(plan, workspace)
end

function _preconditioner_data(plan, workspace, preconditioner)
  throw(ArgumentError("unsupported matrix-free preconditioner $(typeof(preconditioner))"))
end

"""
    default_tangent_linear_solve(plan, state, reduced_rhs; workspace=...,
                                 residual_workspace=..., preconditioner=nothing, kwargs...)

Default matrix-free CG solve for Newton tangent corrections.

Advanced API: this function defines Grico's built-in residual-solve policy. It
is useful as a reference and as the default for [`solve`](@ref) on residual
plans, but nonlinear preconditioning and globalization remain application
policy. Passing a preconditioner to this default method is intentionally
rejected; provide a custom `linear_solve` when tangent preconditioning is
required.
"""
function default_tangent_linear_solve(plan::AssemblyPlan{D,T}, state::State{T},
                                      reduced_rhs::AbstractVector{T};
                                      workspace=_ReducedOperatorWorkspace(plan),
                                      residual_workspace=ResidualWorkspace(plan),
                                      preconditioner=nothing, relative_tolerance=sqrt(eps(T)),
                                      absolute_tolerance=zero(T),
                                      maxiter=max(1_000, 2 * length(reduced_rhs)),
                                      initial_solution=nothing) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :residual)
  _check_state(plan, state)
  _require_length(reduced_rhs, reduced_dof_count(plan), "reduced rhs")
  preconditioner === nothing ||
    throw(ArgumentError("matrix-free tangent preconditioning is not implemented; pass preconditioner=nothing or a custom linear_solve"))
  copyto!(workspace.full_state, coefficients(state))
  operator_apply = (target, vector) -> _reduced_tangent_apply!(target, plan, vector, workspace,
                                                               residual_workspace)
  return _cg_solve(plan, reduced_rhs, workspace; relative_tolerance=relative_tolerance,
                   absolute_tolerance=absolute_tolerance, maxiter=maxiter,
                   initial_solution=initial_solution, inverse_diagonal=nothing, operator_apply)
end

function _jacobi_inverse_diagonal(plan::AssemblyPlan{D,T},
                                  workspace::_ReducedOperatorWorkspace{T}) where {D,
                                                                                  T<:AbstractFloat}
  inverse_diagonal = zeros(T, reduced_dof_count(plan))
  _reduced_diagonal!(inverse_diagonal, plan, workspace) || return nothing
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

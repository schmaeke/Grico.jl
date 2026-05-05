# This file owns the solve API on the matrix-free branch.
#
# Affine solves run CG on the reduced matrix-free operator. Residual solves run
# Newton on the reduced nonlinear equations and use the same reduced-operator CG
# kernel for matrix-free tangent corrections. Solver objects and preconditioners
# are kept separate from the reduced algebra so GMG can plug in as an ordinary
# preconditioner instead of as a special CG code path.

abstract type AbstractLinearSolver end
abstract type AbstractPreconditioner end

"""
    IdentityPreconditioner()

Identity preconditioner for matrix-free reduced systems.

This is mostly useful as an explicit solver policy and as the fallback compiled
form used when a higher-level automatic solver decides that no preconditioner is
available for the current operator class.
"""
struct IdentityPreconditioner <: AbstractPreconditioner end

"""
    JacobiPreconditioner()

Configuration for generic diagonal Jacobi preconditioning.

For affine solves, the preconditioner builds the reduced diagonal directly from
local diagonal kernels when every operator contribution can be scattered
safely. When local diagonal callbacks are unavailable, it probes the reduced
matrix-free operator so users do not need to implement a second operator path
just to obtain a smoother or preconditioner. The default residual tangent solve
does not implement tangent preconditioning; nonlinear users who need one should
provide an explicit tangent solver policy.
"""
struct JacobiPreconditioner <: AbstractPreconditioner end

"""
    CGSolver(; preconditioner=IdentityPreconditioner())

Conjugate-gradient solver configuration for symmetric positive-definite reduced
systems.

The preconditioner is compiled against the concrete reduced operator before the
iteration starts. Unlike the older inverse-diagonal-only path, this solver only
assumes that the compiled preconditioner implements an application operation.
"""
struct CGSolver{P<:AbstractPreconditioner} <: AbstractLinearSolver
  preconditioner::P
end

function CGSolver(; preconditioner=IdentityPreconditioner())
  preconditioner isa AbstractPreconditioner ||
    throw(ArgumentError("preconditioner must be an AbstractPreconditioner"))
  return CGSolver(preconditioner)
end

"""
    FGMRESSolver(; preconditioner=IdentityPreconditioner(), restart=30)

Flexible GMRES solver configuration for general reduced linear systems.

This is the default outer Krylov method for affine problems whose
`operator_class` is not [`SPD`](@ref). The preconditioner is applied on the
right, so variable preconditioners such as multigrid V-cycles remain valid.
"""
struct FGMRESSolver{P<:AbstractPreconditioner} <: AbstractLinearSolver
  preconditioner::P
  restart::Int
end

function FGMRESSolver(; preconditioner=IdentityPreconditioner(), restart::Integer=30)
  preconditioner isa AbstractPreconditioner ||
    throw(ArgumentError("preconditioner must be an AbstractPreconditioner"))
  checked_restart = _checked_positive(restart, "restart")
  return FGMRESSolver(preconditioner, checked_restart)
end

"""
    AutoLinearSolver()

Default linear-solver policy.

For affine problems this policy chooses the outer Krylov method from the
declared operator class: CG for [`SPD`](@ref) problems and FGMRES for general,
nonsymmetric, or indefinite problems. When solving from an `AffineProblem`, it
uses geometric multigrid as the preferred preconditioner whenever a supported
hierarchy can be compiled; otherwise it falls back to Jacobi.
"""
struct AutoLinearSolver <: AbstractLinearSolver end

abstract type _AbstractReducedLinearOperator{T<:AbstractFloat} end

struct _ReducedAffineOperator{D,T<:AbstractFloat,W<:_ReducedOperatorWorkspace{T}} <:
       _AbstractReducedLinearOperator{T}
  plan::AssemblyPlan{D,T}
  workspace::W
end

struct _ReducedTangentOperator{D,T<:AbstractFloat,W<:_ReducedOperatorWorkspace{T},
                               R<:ResidualWorkspace{T}} <:
       _AbstractReducedLinearOperator{T}
  plan::AssemblyPlan{D,T}
  workspace::W
  residual_workspace::R
end

struct _CountingReducedOperator{T<:AbstractFloat,O<:_AbstractReducedLinearOperator{T},C} <:
       _AbstractReducedLinearOperator{T}
  operator::O
  counter::C
end

_operator_size(operator::_AbstractReducedLinearOperator) = reduced_dof_count(operator.plan)
_operator_size(operator::_CountingReducedOperator) = _operator_size(operator.operator)
_operator_class(operator::_AbstractReducedLinearOperator) = operator.plan.operator_class
_operator_class(operator::_CountingReducedOperator) = _operator_class(operator.operator)

function _apply_operator!(result::AbstractVector{T}, operator::_ReducedAffineOperator{D,T},
                          vector::AbstractVector{T}) where {D,T<:AbstractFloat}
  return _reduced_apply!(result, operator.plan, vector, operator.workspace)
end

function _apply_operator!(result::AbstractVector{T}, operator::_ReducedTangentOperator{D,T},
                          vector::AbstractVector{T}) where {D,T<:AbstractFloat}
  return _reduced_tangent_apply!(result, operator.plan, vector, operator.workspace,
                                 operator.residual_workspace)
end

function _apply_operator!(result::AbstractVector{T}, operator::_CountingReducedOperator{T},
                          vector::AbstractVector{T}) where {T<:AbstractFloat}
  operator.counter[] += 1
  return _apply_operator!(result, operator.operator, vector)
end

abstract type _CompiledPreconditioner{T<:AbstractFloat} end

struct _IdentityCompiledPreconditioner{T<:AbstractFloat} <: _CompiledPreconditioner{T} end

struct _JacobiCompiledPreconditioner{T<:AbstractFloat} <: _CompiledPreconditioner{T}
  inverse_diagonal::Vector{T}
end

"""
    solve(problem; solver=AutoLinearSolver(), kwargs...)
    solve(plan; solver=AutoLinearSolver(), kwargs...)

Solve a compiled affine or residual problem.

For affine problems, `solver` is compiled against the reduced constraint-
compatible system and returns a reduced coefficient vector. The returned vector
is expanded back to a full-layout [`State`](@ref). `AutoLinearSolver` uses
geometric multigrid when a supported hierarchy can be built from the problem
description, and otherwise uses matrix-free CG with Jacobi data generated from
the reduced operator.

For residual problems, the same keyword names configure a Newton solve. The
default tangent correction uses matrix-free CG on `P' J(u) P` without
preconditioning. Passing a non-`nothing` preconditioner to the default residual
solve is rejected so nonlinear preconditioning remains an explicit custom
`linear_solve` decision.
"""
function solve(problem::AffineProblem; solver::AbstractLinearSolver=AutoLinearSolver(),
               kwargs...)
  return _solve_affine_problem(problem, solver; kwargs...)
end

function solve(problem::ResidualProblem; linear_solve=default_tangent_linear_solve,
               preconditioner=nothing, kwargs...)
  return solve(compile(problem); linear_solve=linear_solve, preconditioner=preconditioner,
               kwargs...)
end

function solve(plan::AssemblyPlan{D,T}; solver::AbstractLinearSolver=AutoLinearSolver(),
               linear_solve=nothing, preconditioner=nothing, kwargs...) where {D,T<:AbstractFloat}
  kind = _matrix_free_kind(plan.assembly_structure)
  if kind === :affine
    (linear_solve === nothing && preconditioner === nothing) ||
      throw(ArgumentError("affine solves use solver=...; construct a CGSolver, FGMRESSolver, or AutoLinearSolver policy instead of passing linear_solve/preconditioner"))
    return _solve_affine(plan; solver, kwargs...)
  end
  residual_linear_solve = linear_solve === nothing ? default_tangent_linear_solve : linear_solve
  kind === :residual &&
    return _solve_residual(plan; linear_solve=residual_linear_solve, preconditioner, kwargs...)
  throw(ArgumentError("unsupported matrix-free solve kind $kind"))
end

function _solve_affine_problem(problem::AffineProblem, solver::AbstractLinearSolver;
                               kwargs...)
  return _solve_affine(compile(problem); solver, kwargs...)
end

function _solve_affine(plan::AssemblyPlan{D,T}; solver::AbstractLinearSolver=AutoLinearSolver(),
                       relative_tolerance=sqrt(eps(T)), absolute_tolerance=zero(T),
                       maxiter=max(1_000, 2 * reduced_dof_count(plan)),
                       initial_solution=nothing) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  workspace = _ReducedOperatorWorkspace(plan)
  reduced_rhs = zeros(T, reduced_dof_count(plan))
  _reduced_rhs!(reduced_rhs, plan, workspace)
  operator = _ReducedAffineOperator(plan, workspace)
  reduced_values = _solve_reduced_system(solver, operator, reduced_rhs;
                                         relative_tolerance=T(relative_tolerance),
                                         absolute_tolerance=T(absolute_tolerance),
                                         maxiter=maxiter,
                                         initial_solution=initial_solution)
  return _state_from_reduced_result(plan, reduced_values)
end

function _solve_affine_with_callback(plan::AssemblyPlan{D,T}; linear_solve,
                                     preconditioner=nothing, kwargs...) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  workspace = _ReducedOperatorWorkspace(plan)
  reduced_rhs = zeros(T, reduced_dof_count(plan))
  _reduced_rhs!(reduced_rhs, plan, workspace)
  reduced_values = linear_solve(plan, reduced_rhs; preconditioner=preconditioner,
                                workspace=workspace, kwargs...)
  return _state_from_reduced_result(plan, reduced_values)
end

function _state_from_reduced_result(plan::AssemblyPlan{D,T},
                                    reduced_values::AbstractVector{T}) where {D,T<:AbstractFloat}
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

function _solve_reduced_system(::AutoLinearSolver, operator::_AbstractReducedLinearOperator{T},
                               rhs_data::AbstractVector{T}; kwargs...) where {T<:AbstractFloat}
  solver = _is_spd_operator_class(_operator_class(operator)) ?
           CGSolver(; preconditioner=JacobiPreconditioner()) :
           FGMRESSolver(; preconditioner=JacobiPreconditioner())
  return _solve_reduced_system(solver, operator, rhs_data; kwargs...)
end

function _solve_reduced_system(solver::CGSolver, operator::_AbstractReducedLinearOperator{T},
                               rhs_data::AbstractVector{T}; relative_tolerance::T,
                               absolute_tolerance::T, maxiter::Int,
                               initial_solution=nothing) where {T<:AbstractFloat}
  preconditioner = _compile_preconditioner(solver.preconditioner, operator)
  return _cg_solve(operator, rhs_data, preconditioner;
                   relative_tolerance=relative_tolerance,
                   absolute_tolerance=absolute_tolerance, maxiter=maxiter,
                   initial_solution=initial_solution)
end

function _solve_reduced_system(solver::FGMRESSolver,
                               operator::_AbstractReducedLinearOperator{T},
                               rhs_data::AbstractVector{T}; relative_tolerance::T,
                               absolute_tolerance::T, maxiter::Int,
                               initial_solution=nothing) where {T<:AbstractFloat}
  preconditioner = _compile_preconditioner(solver.preconditioner, operator)
  return _fgmres_solve(operator, rhs_data, preconditioner;
                       restart=solver.restart,
                       relative_tolerance=relative_tolerance,
                       absolute_tolerance=absolute_tolerance, maxiter=maxiter,
                       initial_solution=initial_solution)
end

function _cg_solve(operator::_AbstractReducedLinearOperator{T}, rhs_data::AbstractVector{T},
                   preconditioner::_CompiledPreconditioner{T}; relative_tolerance::T,
                   absolute_tolerance::T, maxiter::Int,
                   initial_solution=nothing) where {T<:AbstractFloat}
  n = length(rhs_data)
  n == 0 && return T[]
  n == _operator_size(operator) ||
    throw(ArgumentError("rhs length must match the reduced operator size"))
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
    _apply_operator!(operator_values, operator, solution)
    _axpy!(residual, -one(T), operator_values)
  end

  preconditioned_residual = similar(residual)
  _apply_preconditioner!(preconditioned_residual, preconditioner, residual)
  direction = copy(preconditioned_residual)
  residual_inner = _dot(residual, preconditioned_residual)
  rhs_norm = sqrt(_dot_self(rhs_data))
  tolerance = max(absolute_tolerance, relative_tolerance * max(rhs_norm, one(T)))
  residual_norm2 = _dot_self(residual)
  residual_norm2 <= tolerance * tolerance && return solution

  for iteration in 1:maxiter
    _apply_operator!(operator_values, operator, direction)
    denominator = _dot(direction, operator_values)
    denominator > zero(T) ||
      throw(ArgumentError("CG encountered a non-positive operator direction; provide a different linear_solve for this problem"))
    alpha = residual_inner / denominator
    _axpy!(solution, alpha, direction)
    _axpy!(residual, -alpha, operator_values)
    next_residual_norm2 = _dot_self(residual)
    next_residual_norm2 <= tolerance * tolerance && return solution
    _apply_preconditioner!(preconditioned_residual, preconditioner, residual)
    next_residual_inner = _dot(residual, preconditioned_residual)
    beta = next_residual_inner / residual_inner
    _update_direction!(direction, preconditioned_residual, beta)
    residual_norm2 = next_residual_norm2
    residual_inner = next_residual_inner
  end

  throw(ArgumentError("CG did not converge in $maxiter iterations"))
end

function _fgmres_solve(operator::_AbstractReducedLinearOperator{T},
                       rhs_data::AbstractVector{T},
                       preconditioner::_CompiledPreconditioner{T}; restart::Int,
                       relative_tolerance::T, absolute_tolerance::T, maxiter::Int,
                       initial_solution=nothing) where {T<:AbstractFloat}
  n = length(rhs_data)
  n == 0 && return T[]
  n == _operator_size(operator) ||
    throw(ArgumentError("rhs length must match the reduced operator size"))
  restart >= 1 || throw(ArgumentError("restart must be positive"))
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
    _apply_operator!(operator_values, operator, solution)
    _axpy!(residual, -one(T), operator_values)
  end

  rhs_norm = sqrt(_dot_self(rhs_data))
  tolerance = max(absolute_tolerance, relative_tolerance * max(rhs_norm, one(T)))
  residual_norm = sqrt(_dot_self(residual))
  residual_norm <= tolerance && return solution

  restart_count = min(restart, n)
  krylov = zeros(T, n, restart_count + 1)
  preconditioned = zeros(T, n, restart_count)
  hessenberg = zeros(T, restart_count + 1, restart_count)
  least_squares_rhs = zeros(T, restart_count + 1)
  total_iterations = 0

  while total_iterations < maxiter
    fill!(krylov, zero(T))
    fill!(preconditioned, zero(T))
    fill!(hessenberg, zero(T))
    fill!(least_squares_rhs, zero(T))
    residual_norm = sqrt(_dot_self(residual))
    residual_norm <= tolerance && return solution

    @inbounds for row in 1:n
      krylov[row, 1] = residual[row] / residual_norm
    end

    least_squares_rhs[1] = residual_norm
    inner_limit = min(restart_count, maxiter - total_iterations)
    accepted = false

    for column in 1:inner_limit
      total_iterations += 1
      z_column = view(preconditioned, :, column)
      v_column = view(krylov, :, column)
      _apply_preconditioner!(z_column, preconditioner, v_column)
      _apply_operator!(operator_values, operator, z_column)

      for basis_column in 1:column
        v_basis = view(krylov, :, basis_column)
        hessenberg[basis_column, column] = _dot(operator_values, v_basis)
        _axpy!(operator_values, -hessenberg[basis_column, column], v_basis)
      end

      next_norm = sqrt(_dot_self(operator_values))
      hessenberg[column+1, column] = next_norm

      if next_norm > zero(T) && column < restart_count + 1
        @inbounds for row in 1:n
          krylov[row, column+1] = operator_values[row] / next_norm
        end
      end

      h_view = view(hessenberg, 1:(column+1), 1:column)
      g_view = view(least_squares_rhs, 1:(column+1))
      y = h_view \ g_view
      residual_estimate = sqrt(_dot_self(g_view - h_view * y))

      if residual_estimate <= tolerance || next_norm == zero(T)
        _update_fgmres_solution!(solution, preconditioned, y, column)
        _apply_operator!(operator_values, operator, solution)
        _copy_residual!(residual, rhs_data, operator_values)
        sqrt(_dot_self(residual)) <= tolerance && return solution
        accepted = true
        break
      end

      if column == inner_limit
        _update_fgmres_solution!(solution, preconditioned, y, column)
        accepted = true
      end
    end

    accepted || throw(ArgumentError("FGMRES failed to build a Krylov update"))
    _apply_operator!(operator_values, operator, solution)
    _copy_residual!(residual, rhs_data, operator_values)
  end

  throw(ArgumentError("FGMRES did not converge in $maxiter iterations"))
end

function _update_fgmres_solution!(solution::AbstractVector{T},
                                  preconditioned::AbstractMatrix{T},
                                  coefficients::AbstractVector{T},
                                  count::Int) where {T<:AbstractFloat}
  for column in 1:count
    scale = coefficients[column]
    iszero(scale) && continue
    z_column = view(preconditioned, :, column)
    _axpy!(solution, scale, z_column)
  end

  return solution
end

function _copy_residual!(residual::AbstractVector{T}, rhs_data::AbstractVector{T},
                         operator_values::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(rhs_data, length(residual), "rhs")
  _require_length(operator_values, length(residual), "operator values")

  @inbounds for index in eachindex(residual)
    residual[index] = rhs_data[index] - operator_values[index]
  end

  return residual
end

function _compile_preconditioner(::IdentityPreconditioner,
                                 operator::_AbstractReducedLinearOperator{T}) where {T<:AbstractFloat}
  return _IdentityCompiledPreconditioner{T}()
end

function _compile_preconditioner(::JacobiPreconditioner,
                                 operator::_ReducedAffineOperator{D,T}) where {D,T<:AbstractFloat}
  inverse_diagonal = _jacobi_inverse_diagonal(operator.plan, operator.workspace)
  inverse_diagonal === nothing && return _IdentityCompiledPreconditioner{T}()
  return _JacobiCompiledPreconditioner(inverse_diagonal)
end

function _compile_preconditioner(preconditioner::AbstractPreconditioner,
                                 operator::_AbstractReducedLinearOperator)
  throw(ArgumentError("preconditioner $(typeof(preconditioner)) is not supported for this reduced operator"))
end

function _compile_preconditioner(preconditioner::AbstractPreconditioner,
                                 operator::_CountingReducedOperator)
  return _compile_preconditioner(preconditioner, operator.operator)
end

function _compile_preconditioner(preconditioner, operator::_AbstractReducedLinearOperator)
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
  operator = _ReducedTangentOperator(plan, workspace, residual_workspace)
  return _solve_reduced_system(CGSolver(), operator, reduced_rhs;
                               relative_tolerance=relative_tolerance,
                               absolute_tolerance=absolute_tolerance, maxiter=maxiter,
                               initial_solution=initial_solution)
end

function _jacobi_inverse_diagonal(plan::AssemblyPlan{D,T},
                                  workspace::_ReducedOperatorWorkspace{T}) where {D,
                                                                                  T<:AbstractFloat}
  inverse_diagonal = zeros(T, reduced_dof_count(plan))
  _reduced_diagonal!(inverse_diagonal, plan, workspace) ||
    _probe_reduced_diagonal!(inverse_diagonal, plan, workspace)
  return _invert_jacobi_diagonal!(inverse_diagonal)
end

function _probe_reduced_diagonal!(diagonal::AbstractVector{T}, plan::AssemblyPlan{D,T},
                                  workspace::_ReducedOperatorWorkspace{T}) where {D,
                                                                                  T<:AbstractFloat}
  count = reduced_dof_count(plan)
  _require_length(diagonal, count, "reduced diagonal")
  basis = zeros(T, count)
  response = zeros(T, count)

  for index in 1:count
    basis[index] = one(T)
    _reduced_apply!(response, plan, basis, workspace)
    diagonal[index] = response[index]
    basis[index] = zero(T)
  end

  return diagonal
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

function _apply_preconditioner!(result::AbstractVector{T},
                                ::_IdentityCompiledPreconditioner{T},
                                residual::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(result, length(residual), "preconditioned residual")
  copyto!(result, residual)
  return result
end

function _apply_preconditioner!(result::AbstractVector{T},
                                preconditioner::_JacobiCompiledPreconditioner{T},
                                residual::AbstractVector{T}) where {T<:AbstractFloat}
  _require_length(result, length(residual), "preconditioned residual")
  inverse_diagonal = preconditioner.inverse_diagonal
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

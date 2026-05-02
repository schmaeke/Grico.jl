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

function default_linear_solve(plan::AssemblyPlan, rhs_data::AbstractVector; kwargs...)
  throw(ArgumentError("matrix-free default solve is not implemented yet; pass `linear_solve` to `solve` for experiments"))
end

"""
    solve(problem; linear_solve=default_linear_solve, preconditioner=JacobiPreconditioner(), kwargs...)
    solve(plan; linear_solve=default_linear_solve, preconditioner=JacobiPreconditioner(), kwargs...)

Solve a compiled affine problem with a user-supplied matrix-free linear solver.

`linear_solve` is called as

    linear_solve(plan, rhs(plan); preconditioner=preconditioner, kwargs...)

and must return a full-layout coefficient vector. The package default solve
path is intentionally a stub on this branch until the matrix-free Krylov and
preconditioning design is implemented.
"""
function solve(problem::AffineProblem; linear_solve=default_linear_solve,
               preconditioner=JacobiPreconditioner(), kwargs...)
  return solve(compile(problem); linear_solve=linear_solve, preconditioner=preconditioner,
               kwargs...)
end

function solve(plan::AssemblyPlan{D,T}; linear_solve=default_linear_solve,
               preconditioner=JacobiPreconditioner(), kwargs...) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  rhs_data = rhs(plan)
  values = linear_solve(plan, rhs_data; preconditioner=preconditioner, kwargs...)
  _require_length(values, dof_count(plan), "linear solve result")
  eltype(values) == T ||
    throw(ArgumentError("linear solve result element type must match the plan scalar type"))
  return State(plan, values)
end

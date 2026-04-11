# This file implements the linear-solve layer built on top of `AffineSystem`.
# It owns:
# 1. the default direct/iterative solve policy,
# 2. initial-guess handling and reduced/full-state conversion for solves, and
# 3. a small set of package-provided preconditioners and Krylov wrappers.
#
# The preceding assembly layer produces a reduced affine system together with the
# data needed to reconstruct a full coefficient vector on the original field
# layout. The task of this file is to decide how that reduced system should be
# solved and how the reduced solution should be interpreted.
#
# In symbols, the assembly layer hands this file a reduced system
#
#   A_red u_red = b_red,
#
# together with a reconstruction formula
#
#   u_full = shift + R u_red.
#
# Everything in this file acts on `A_red` and `b_red`, but the public API is
# phrased in terms of the reconstructed full-layout coefficient vector because
# that is what the rest of the package consumes.
#
# The design is intentionally pragmatic rather than fully abstract. The package
# supports a curated set of solver-side features that are useful for the finite-
# element systems it assembles:
#
# - sparse direct solves with fill-reducing orderings,
# - optional preconditioned Krylov iterations on sufficiently large systems,
# - a small collection of package-owned preconditioners,
# - and the ability to accept either reduced or full-layout initial guesses.
#
# This file therefore reads in three layers.
#
# First come user-facing preconditioner configuration objects.
# Second come the internal operators and policies that realize those
# configurations on one assembled system.
# Third come the public solve entry points and the low-level `ldiv!`
# implementations that apply the cached preconditioner/factor operators.

# Public preconditioner configurations.

# Internal supertype for package-provided preconditioner configurations. The
# exported concrete types below represent the small curated set of
# preconditioners the toolbox is willing to support as first-class features.
abstract type _AbstractPreconditioner end

"""
    AdditiveSchwarzPreconditioner(; min_dofs=2_000)

Configuration for a two-level additive Schwarz preconditioner.

This preconditioner builds one dense local solve on each reduced leaf patch and
adds a low-order geometric coarse correction assembled from root-grid vertex
functions. It is intended as an explicit geometric preconditioner for large
assembled affine systems whose sparsity pattern reflects the local support of hp
finite-element basis functions.

The coarse space is not an arbitrary auxiliary basis: it is the multilinear
root-grid space compiled in `assembly.jl` by interpolating retained reduced
dofs to the vertices of the unrefined Cartesian grid. In other words, the
preconditioner combines high-order local patch solves with a low-order global
correction on the coarsest geometric skeleton of the domain.

The `min_dofs` parameter is a usage heuristic rather than an algebraic tuning
parameter: the default solve path only attempts the additive-Schwarz Krylov
iteration when the reduced system has at least that many unknowns. Smaller
systems fall back directly to the sparse direct solve path, which is usually
more reliable and often faster at that scale.
"""
struct AdditiveSchwarzPreconditioner <: _AbstractPreconditioner
  min_dofs::Int

  function AdditiveSchwarzPreconditioner(; min_dofs::Integer=2_000)
    checked = _checked_nonnegative(min_dofs, "min_dofs")
    return new(checked)
  end
end

function Base.:(==)(first::AdditiveSchwarzPreconditioner, second::AdditiveSchwarzPreconditioner)
  first.min_dofs == second.min_dofs
end
function Base.hash(preconditioner::AdditiveSchwarzPreconditioner, seed::UInt)
  hash(preconditioner.min_dofs, hash(AdditiveSchwarzPreconditioner, seed))
end

@inline function _checked_nonnegative_real(value::Real, name::AbstractString)
  checked = float(value)
  isfinite(checked) || throw(ArgumentError("$name must be finite"))
  checked >= 0 || throw(ArgumentError("$name must be non-negative"))
  return checked
end

"""
    SmoothedAggregationAMGPreconditioner(; min_dofs=2_000)

Configuration for a smoothed-aggregation algebraic multigrid preconditioner.

This preconditioner builds a smoothed-aggregation hierarchy with
`AlgebraicMultigrid.jl` and applies it as a linear operator inside the package
Krylov wrappers. It is the package default for systems assembled with the
symmetry flag enabled.

As in the other package preconditioners, `min_dofs` is a policy heuristic: it
controls when the default solve path will bother constructing the multigrid
hierarchy instead of going directly to the sparse direct solve.
"""
struct SmoothedAggregationAMGPreconditioner <: _AbstractPreconditioner
  min_dofs::Int

  function SmoothedAggregationAMGPreconditioner(; min_dofs::Integer=2_000)
    checked = _checked_nonnegative(min_dofs, "min_dofs")
    return new(checked)
  end
end

function Base.:(==)(first::SmoothedAggregationAMGPreconditioner,
                    second::SmoothedAggregationAMGPreconditioner)
  first.min_dofs == second.min_dofs
end
function Base.hash(preconditioner::SmoothedAggregationAMGPreconditioner, seed::UInt)
  hash(preconditioner.min_dofs, hash(SmoothedAggregationAMGPreconditioner, seed))
end

"""
    ILUPreconditioner(; tau=1e-3, min_dofs=2_000)

Configuration for an incomplete-LU preconditioner.

This preconditioner builds an ordered Crout ILU factorization with
`IncompleteLU.jl` and applies it through the package GMRES wrapper. It is the
package default for assembled systems that are not marked symmetric.

The drop parameter `tau` belongs to the ILU construction itself, whereas
`min_dofs` again only decides when the default solve policy attempts the
preconditioned iterative route at all.
"""
struct ILUPreconditioner <: _AbstractPreconditioner
  tau::Float64
  min_dofs::Int

  function ILUPreconditioner(; tau::Real=1e-3, min_dofs::Integer=2_000)
    checked_tau = _checked_nonnegative_real(tau, "tau")
    checked_min_dofs = _checked_nonnegative(min_dofs, "min_dofs")
    return new(checked_tau, checked_min_dofs)
  end
end

function Base.:(==)(first::ILUPreconditioner, second::ILUPreconditioner)
  first.tau == second.tau && first.min_dofs == second.min_dofs
end
function Base.hash(preconditioner::ILUPreconditioner, seed::UInt)
  hashed = hash(preconditioner.tau, hash(ILUPreconditioner, seed))
  return hash(preconditioner.min_dofs, hashed)
end

# Normalize a user-provided field group to a tuple of unique field descriptors.
# The public field-split preconditioner uses field identity rather than raw
# matrix row ranges so the split remains meaningful after Dirichlet elimination
# and static condensation.
function _normalized_field_group(fields, name::AbstractString)
  raw = if fields isa AbstractField
    (fields,)
  elseif fields isa Tuple
    fields
  elseif fields isa AbstractVector
    tuple(fields...)
  else
    throw(ArgumentError("$name must be a field, tuple of fields, or vector of fields"))
  end
  isempty(raw) && throw(ArgumentError("$name must contain at least one field"))
  seen_ids = Set{UInt64}()

  for field in raw
    field isa AbstractField || throw(ArgumentError("$name must contain only field descriptors"))
    !(_field_id(field) in seen_ids) || throw(ArgumentError("$name must not repeat the same field"))
    push!(seen_ids, _field_id(field))
  end

  return raw
end

function _field_group_ids(fields::Tuple{Vararg{AbstractField}})
  ntuple(index -> _field_id(fields[index]), length(fields))
end

# Check that the two user-specified field groups describe a genuine block split.
function _check_disjoint_field_groups(primary_fields::Tuple{Vararg{AbstractField}},
                                      schur_fields::Tuple{Vararg{AbstractField}})
  isempty(intersect(Set(_field_group_ids(primary_fields)), Set(_field_group_ids(schur_fields)))) ||
    throw(ArgumentError("primary_fields and schur_fields must be disjoint"))
  return nothing
end

"""
    FieldSplitSchurPreconditioner(primary_fields, schur_fields;
                                  primary_preconditioner=AdditiveSchwarzPreconditioner(),
                                  min_dofs=2_000)

Configuration for a two-way field-split Schur preconditioner.

`primary_fields` define the leading block `A₁₁` of the reduced matrix, while
`schur_fields` define the complementary Schur block `A₂₂`. The groups are
specified by field descriptors rather than matrix index ranges, so the split
remains tied to the variational unknowns even after constraint elimination and
static condensation. When applied to an assembled system, the two groups must
form a disjoint partition of the system layout.

The current implementation uses a lower-triangular Schur factorization: the
primary block is approximated by `primary_preconditioner` when that is
worthwhile and otherwise by a direct block solve, and the Schur block is
approximated by the matrix-based formula `A₂₂ - A₂₁ diag(A₁₁)⁻¹ A₁₂`. This
keeps the public abstraction stable while leaving room to improve the internal
Schur model later without changing user code.
"""
struct FieldSplitSchurPreconditioner <: _AbstractPreconditioner
  primary_fields::Tuple{Vararg{AbstractField}}
  schur_fields::Tuple{Vararg{AbstractField}}
  primary_preconditioner::AdditiveSchwarzPreconditioner
  min_dofs::Int

  function FieldSplitSchurPreconditioner(primary_fields, schur_fields;
                                         primary_preconditioner::AdditiveSchwarzPreconditioner=AdditiveSchwarzPreconditioner(),
                                         min_dofs::Integer=2_000)
    normalized_primary = _normalized_field_group(primary_fields, "primary_fields")
    normalized_schur = _normalized_field_group(schur_fields, "schur_fields")
    _check_disjoint_field_groups(normalized_primary, normalized_schur)
    checked_min_dofs = _checked_nonnegative(min_dofs, "min_dofs")
    return new(normalized_primary, normalized_schur, primary_preconditioner, checked_min_dofs)
  end
end

function Base.:(==)(first::FieldSplitSchurPreconditioner, second::FieldSplitSchurPreconditioner)
  _field_group_ids(first.primary_fields) == _field_group_ids(second.primary_fields) &&
    _field_group_ids(first.schur_fields) == _field_group_ids(second.schur_fields) &&
    first.primary_preconditioner == second.primary_preconditioner &&
    first.min_dofs == second.min_dofs
end
function Base.hash(preconditioner::FieldSplitSchurPreconditioner, seed::UInt)
  hashed = hash(_field_group_ids(preconditioner.primary_fields),
                hash(FieldSplitSchurPreconditioner, seed))
  hashed = hash(_field_group_ids(preconditioner.schur_fields), hashed)
  hashed = hash(preconditioner.primary_preconditioner, hashed)
  return hash(preconditioner.min_dofs, hashed)
end

# Internal solve-style tags and cached linear-operator types.

# Internal dispatch tag selecting the Krylov method used for a given matrix
# symmetry class.
abstract type _KrylovSolveStyle end

struct _SymmetricKrylovStyle <: _KrylovSolveStyle end

struct _GeneralKrylovStyle <: _KrylovSolveStyle end

# Internal operator contract for cached preconditioners/factors that keep one
# reusable RHS buffer for allocation-free in-place `ldiv!` application.
abstract type _BufferedLinearOperator{T<:AbstractFloat} end

# One local Schwarz patch consisting of its reduced dof set and a factored dense
# patch matrix.
struct _SchwarzPatch{T<:AbstractFloat,F}
  dofs::Vector{Int}
  factor::F
end

function _SchwarzPatch(dofs::Vector{Int}, factor::F) where {T<:AbstractFloat,F<:Factorization{T}}
  return _SchwarzPatch{T,F}(dofs, factor)
end

struct _SchwarzThreadBuffer{T<:AbstractFloat}
  output::Vector{T}
  rhs::Vector{T}
end

# Prepared two-level additive Schwarz operator built for one assembled system.
# The coarse factorization and patch factorizations are created lazily and then
# reused across repeated solves of the same matrix.
mutable struct _AdditiveSchwarzOperator{T<:AbstractFloat,F,CF} <: _BufferedLinearOperator{T}
  patches::Vector{_SchwarzPatch{T,F}}
  coarse_prolongation::SparseMatrixCSC{T,Int}
  coarse_factor::CF
  thread_buffers::Vector{_SchwarzThreadBuffer{T}}
  coarse_rhs::Vector{T}
  coarse_solution::Vector{T}
  apply_rhs::Vector{T}
end

# Reduced-system partition induced by a two-way field split. The stored indices
# are reduced-system row/column numbers, not global layout dofs.
struct _FieldSplitPartition
  primary_indices::Vector{Int}
  schur_indices::Vector{Int}
end

# Prepared block data for the lower-triangular field-split Schur operator.
struct _FieldSplitBlocks{T<:AbstractFloat}
  primary_matrix::SparseMatrixCSC{T,Int}
  coupling12::SparseMatrixCSC{T,Int}
  coupling21::SparseMatrixCSC{T,Int}
  schur_block::SparseMatrixCSC{T,Int}
  primary_topology::_PatchSolveTopology{T}
end

# Small reusable ordered linear operator for cached sparse factorizations. The
# factor is built once, and the work buffers allow repeated `ldiv!`
# applications without allocating.
mutable struct _OrderedFactorOperator{T<:AbstractFloat,F} <: _BufferedLinearOperator{T}
  factor::F
  ordering::Vector{Int}
  inverse_ordering::Vector{Int}
  ordered_rhs::Vector{T}
  ordered_solution::Vector{T}
  apply_rhs::Vector{T}
end

# Prepared field-split Schur operator. The primary block uses a package-owned
# approximate inverse, while the Schur block is handled by a cached direct solve
# of a sparse matrix approximation.
mutable struct _FieldSplitSchurOperator{T<:AbstractFloat,PO,SO} <: _BufferedLinearOperator{T}
  primary_indices::Vector{Int}
  schur_indices::Vector{Int}
  coupling21::SparseMatrixCSC{T,Int}
  primary_operator::PO
  schur_operator::SO
  primary_rhs::Vector{T}
  primary_solution::Vector{T}
  schur_rhs::Vector{T}
  schur_solution::Vector{T}
  apply_rhs::Vector{T}
end

# Ordered reduced-system data used by direct solves and custom solver hooks.
struct _OrderedSystemData{T<:AbstractFloat}
  matrix::SparseMatrixCSC{T,Int}
  rhs::Vector{T}
end

# Default fallback linear solver used by `solve` when the user does not provide
# a custom routine. It is intentionally just sparse backslash; the higher-level
# `solve` logic decides when to prefer a package-provided preconditioned Krylov
# path or reorderings around it.
default_linear_solve(matrix, rhs) = matrix \ rhs

_default_krylov_reltol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))
_default_krylov_maxiter(system::AffineSystem) = max(1_000, size(system.matrix, 1))
_default_gmres_restart(system::AffineSystem) = min(50, size(system.matrix, 1))
function _default_krylov_style(system::AffineSystem)
  system.symmetric ? _SymmetricKrylovStyle() : _GeneralKrylovStyle()
end
function _matrix_krylov_style(matrix_data::SparseMatrixCSC)
  _is_symmetric_matrix(matrix_data) ? _SymmetricKrylovStyle() : _GeneralKrylovStyle()
end

# Resolve the user-facing preconditioner keyword to a concrete package
# configuration. Keeping this normalization in one place makes it explicit that
# `nothing` means "use the package default" rather than "disable
# preconditioning".
function _default_preconditioner(system::AffineSystem)
  return system.symmetric ? SmoothedAggregationAMGPreconditioner() : ILUPreconditioner()
end

function _resolved_preconditioner(system::AffineSystem, preconditioner)
  preconditioner === nothing && return _default_preconditioner(system)
  preconditioner isa _AbstractPreconditioner ||
    throw(ArgumentError("preconditioner must be `nothing` or a package-provided preconditioner configuration"))
  return preconditioner
end

# Default linear-solve policy: try the package-provided preconditioned Krylov
# path when the matrix structure suggests it is worthwhile, otherwise use the
# sparse direct solve path immediately. This is a usage heuristic, not an
# algebraic theorem: unsuccessful Krylov attempts are intentionally allowed to
# fall back to the direct path.
function _default_system_linear_solve(system::AffineSystem{T},
                                      preconditioner::_AbstractPreconditioner,
                                      initial_solution::Union{Nothing,AbstractVector{T}}=nothing) where {T<:AbstractFloat}
  if _preconditioner_is_applicable(system, preconditioner)
    style = _preconditioned_krylov_style(system, preconditioner)
    values, converged = _preconditioned_krylov_solve(system, preconditioner, style,
                                                     initial_solution)
    converged && return values
    @warn "$( _preconditioned_krylov_name(preconditioner, style) ) did not converge; falling back to sparse direct solve" reduced_dofs = size(system.matrix,
                                                                                                                                              1)
  end

  return _default_system_direct_solve(system)
end

# Direct reduced solve with a fill-reducing reordering. Symmetric positive
# definite systems first try a Cholesky factorization; all others fall back to
# the generic sparse backslash or user-provided linear solver.
function _ordered_system_data(system::AffineSystem{T}) where {T<:AbstractFloat}
  return _OrderedSystemData(system.matrix[system.ordering, system.ordering],
                            system.rhs[system.ordering])
end

function _unordered_solution(system::AffineSystem, ordered_values::AbstractVector)
  return ordered_values[system.inverse_ordering]
end

function _default_system_direct_solve(system::AffineSystem{T}) where {T<:AbstractFloat}
  ordered = _ordered_system_data(system)

  if system.symmetric
    try
      return _unordered_solution(system, cholesky(Symmetric(ordered.matrix)) \ ordered.rhs)
    catch
    end
  end

  return _unordered_solution(system, default_linear_solve(ordered.matrix, ordered.rhs))
end

# Small adapter to support user-provided solvers with or without an initial
# guess argument.
function _call_linear_solve(linear_solve, matrix_data, rhs_data, initial_solution)
  if initial_solution !== nothing &&
     applicable(linear_solve, matrix_data, rhs_data, initial_solution)
    return linear_solve(matrix_data, rhs_data, initial_solution)
  end

  return linear_solve(matrix_data, rhs_data)
end

# Layout equivalence test used when interpreting `State` objects as initial
# guesses for an assembled system.
function _matching_layout(first::FieldLayout, second::FieldLayout)
  dof_count(first) == dof_count(second) || return false
  field_count(first) == field_count(second) || return false

  for index in eachindex(first.slots)
    first_slot = first.slots[index]
    second_slot = second.slots[index]
    _field_id(first_slot.field) == _field_id(second_slot.field) || return false
    first_slot.offset == second_slot.offset || return false
    first_slot.scalar_dof_count == second_slot.scalar_dof_count || return false
    first_slot.dof_count == second_slot.dof_count || return false
  end

  return true
end

# Accept either a full-layout initial guess or a reduced one and convert it to
# reduced-system ordering.
function _reduced_initial_solution(system::AffineSystem{T},
                                   initial_solution) where {T<:AbstractFloat}
  initial_solution === nothing && return nothing
  initial_values = if initial_solution isa State
    _matching_layout(field_layout(initial_solution), system.layout) ||
      throw(ArgumentError("initial state layout must match the assembled system layout"))
    coefficients(initial_solution)
  else
    initial_solution
  end
  reduced = if length(initial_values) == dof_count(system.layout)
    Vector{T}(initial_values[system.solve_dofs])
  elseif length(initial_values) == length(system.solve_dofs)
    Vector{T}(initial_values)
  else
    throw(ArgumentError("initial solution length must match either the reduced system or full layout"))
  end
  return reduced
end

function _ordered_initial_solution(system::AffineSystem{T},
                                   initial_solution) where {T<:AbstractFloat}
  reduced = _reduced_initial_solution(system, initial_solution)
  reduced === nothing && return nothing
  return reduced[system.ordering]
end

# Public solve entry points.

"""
    solve(system; linear_solve=default_linear_solve, preconditioner=nothing,
          initial_solution=nothing)
    solve(plan; linear_solve=default_linear_solve, preconditioner=nothing,
          initial_solution=nothing)

Solve an assembled affine system, or assemble and solve directly from a plan.

The returned vector always contains the full coefficient vector on the original
field layout, not just the reduced solve unknowns. By default, the solver uses
either a sparse direct solve or, for sufficiently large systems with available
matrix size, a package-provided preconditioner together with a Krylov iteration
and automatic fallback to the direct path. Systems marked symmetric use
conjugate gradients with
[`SmoothedAggregationAMGPreconditioner`](@ref) by default; all other systems use
GMRES with [`ILUPreconditioner`](@ref). Explicit alternatives such as
[`AdditiveSchwarzPreconditioner`](@ref) and
[`FieldSplitSchurPreconditioner`](@ref) remain available through the
`preconditioner` keyword, which is interpreted only by the default solve path.
If a custom `linear_solve` routine is provided, package preconditioner
configurations are not applied automatically.

`initial_solution` may be given either on the reduced solve space, on the full
field layout, or as a [`State`](@ref) on the matching layout. The return value,
however, is always the full coefficient vector on the original field layout.

This means the solve API deliberately hides the reduced-system bookkeeping from
its callers. Reduced vectors are accepted for efficiency, but full-layout
vectors are returned because that is the natural representation for subsequent
residual evaluation, adaptivity, verification, and export.
"""
function solve(system::AffineSystem; linear_solve=default_linear_solve, preconditioner=nothing,
               initial_solution=nothing)
  reduced_values = if linear_solve === default_linear_solve
    resolved_preconditioner = _resolved_preconditioner(system, preconditioner)
    _with_internal_blas_threads() do
      _default_system_linear_solve(system, resolved_preconditioner,
                                   _reduced_initial_solution(system, initial_solution))
    end
  else
    preconditioner === nothing ||
      throw(ArgumentError("package preconditioner configurations are only used by the default solve path"))
    ordered = _ordered_system_data(system)
    ordered_initial = _ordered_initial_solution(system, initial_solution)
    ordered_values = _call_linear_solve(linear_solve, ordered.matrix, ordered.rhs, ordered_initial)
    _unordered_solution(system, ordered_values)
  end
  return _expand_system_values(system, reduced_values)
end

function solve(plan::AssemblyPlan; linear_solve=default_linear_solve, preconditioner=nothing,
               initial_solution=nothing)
  solve(assemble(plan); linear_solve=linear_solve, preconditioner=preconditioner,
        initial_solution=initial_solution)
end

# Default preconditioned Krylov path and applicability heuristics.

# Package preconditioners participate in the default iterative solve path by
# providing:
#
# 1. an applicability heuristic deciding when the preconditioned route is worth
#    attempting,
# 2. the Krylov method paired with the preconditioner on a given system, and
# 3. a lazily built linear operator that implements `ldiv!`.
#
# The current package keeps this interface intentionally small because the goal
# is a compact toolbox with a curated set of robust solver features rather than
# a generic preconditioner framework.
function _preconditioned_krylov_style(system::AffineSystem, ::_AbstractPreconditioner)
  return _default_krylov_style(system)
end
_preconditioned_krylov_style(::AffineSystem, ::ILUPreconditioner) = _GeneralKrylovStyle()
function _preconditioned_krylov_style(system::AffineSystem, ::FieldSplitSchurPreconditioner)
  _GeneralKrylovStyle()
end

_preconditioner_label(::AdditiveSchwarzPreconditioner) = "Additive Schwarz"
_preconditioner_label(::SmoothedAggregationAMGPreconditioner) = "Smoothed Aggregation AMG"
_preconditioner_label(::ILUPreconditioner) = "ILU"
_preconditioner_label(::FieldSplitSchurPreconditioner) = "Field-Split Schur"

_krylov_method_name(::_SymmetricKrylovStyle) = "CG"
_krylov_method_name(::_GeneralKrylovStyle) = "GMRES"

function _preconditioned_krylov_name(preconditioner::_AbstractPreconditioner,
                                     style::_KrylovSolveStyle)
  return string(_preconditioner_label(preconditioner), " + ", _krylov_method_name(style))
end

# Attempt a preconditioned Krylov solve and accept it only when the true
# relative residual of the assembled system is below the default tolerance. The
# extra residual check keeps the policy robust against optimistic convergence
# reports from the inner Krylov routine or from imperfect preconditioner
# behavior on ill-conditioned systems.
function _preconditioned_krylov_solve(system::AffineSystem{T},
                                      preconditioner::_AbstractPreconditioner,
                                      initial_solution::Union{Nothing,AbstractVector{T}}=nothing) where {T<:AbstractFloat}
  return _preconditioned_krylov_solve(system, preconditioner,
                                      _preconditioned_krylov_style(system, preconditioner),
                                      initial_solution)
end

function _preconditioned_krylov_solve(system::AffineSystem{T},
                                      preconditioner::_AbstractPreconditioner,
                                      style::_KrylovSolveStyle,
                                      initial_solution::Union{Nothing,AbstractVector{T}}=nothing) where {T<:AbstractFloat}
  try
    operator = _preconditioner_operator(system, preconditioner)
    iterate = initial_solution === nothing ? zeros(T, size(system.matrix, 1)) :
              copy(initial_solution)
    solution = _run_preconditioned_krylov!(iterate, system, operator, style,
                                           initial_solution === nothing)
    converged = _relative_residual_norm(system.matrix, system.rhs, solution) <=
                _default_krylov_reltol(T)
    return solution, converged
  catch
    return zeros(T, size(system.matrix, 1)), false
  end
end

function _cg_krylov_options(system::AffineSystem{T}, preconditioner) where {T<:AbstractFloat}
  return (; M=preconditioner, ldiv=true, rtol=_default_krylov_reltol(T),
          itmax=_default_krylov_maxiter(system))
end

function _gmres_krylov_options(system::AffineSystem{T}, preconditioner) where {T<:AbstractFloat}
  return (; N=preconditioner, ldiv=true, rtol=_default_krylov_reltol(T), restart=true,
          memory=_default_gmres_restart(system), itmax=_default_krylov_maxiter(system))
end

function _run_preconditioned_krylov!(iterate::AbstractVector{T}, system::AffineSystem{T},
                                     preconditioner, style::_KrylovSolveStyle,
                                     initially_zero::Bool) where {T<:AbstractFloat}
  solution, _ = initially_zero ? _run_krylov_without_initial_guess(style, system, preconditioner) :
                _run_krylov_with_initial_guess(style, iterate, system, preconditioner)
  return solution
end

function _run_krylov_without_initial_guess(::_SymmetricKrylovStyle, system::AffineSystem{T},
                                           preconditioner) where {T<:AbstractFloat}
  return cg(Symmetric(system.matrix), system.rhs; _cg_krylov_options(system, preconditioner)...)
end

function _run_krylov_with_initial_guess(::_SymmetricKrylovStyle, iterate::AbstractVector{T},
                                        system::AffineSystem{T},
                                        preconditioner) where {T<:AbstractFloat}
  return cg(Symmetric(system.matrix), system.rhs, iterate;
            _cg_krylov_options(system, preconditioner)...)
end

function _run_krylov_without_initial_guess(::_GeneralKrylovStyle, system::AffineSystem{T},
                                           preconditioner) where {T<:AbstractFloat}
  return gmres(system.matrix, system.rhs; _gmres_krylov_options(system, preconditioner)...)
end

function _run_krylov_with_initial_guess(::_GeneralKrylovStyle, iterate::AbstractVector{T},
                                        system::AffineSystem{T},
                                        preconditioner) where {T<:AbstractFloat}
  return gmres(system.matrix, system.rhs, iterate; _gmres_krylov_options(system, preconditioner)...)
end

function _relative_residual_norm(matrix_data::SparseMatrixCSC{T,Int}, rhs_data::AbstractVector{T},
                                 solution::AbstractVector{T}) where {T<:AbstractFloat}
  scale = max(norm(rhs_data), one(T))
  return norm(matrix_data * solution - rhs_data) / scale
end

# Decide whether a preconditioner should be attempted on a given assembled
# system before falling back to the default sparse direct path.
_preconditioner_min_dofs(preconditioner::AdditiveSchwarzPreconditioner) = preconditioner.min_dofs
function _preconditioner_min_dofs(preconditioner::SmoothedAggregationAMGPreconditioner)
  preconditioner.min_dofs
end
_preconditioner_min_dofs(preconditioner::ILUPreconditioner) = preconditioner.min_dofs
_preconditioner_min_dofs(preconditioner::FieldSplitSchurPreconditioner) = preconditioner.min_dofs

function _preconditioner_is_applicable(system::AffineSystem,
                                       preconditioner::_AbstractPreconditioner)
  return size(system.matrix, 1) >= _preconditioner_min_dofs(preconditioner)
end

function _preconditioner_is_applicable(system::AffineSystem,
                                       preconditioner::AdditiveSchwarzPreconditioner)
  size(system.matrix, 1) >= _preconditioner_min_dofs(preconditioner) || return false
  !isempty(system.solve_topology.leaf_patches) || return false
  return true
end
function _preconditioner_is_applicable(system::AffineSystem,
                                       preconditioner::FieldSplitSchurPreconditioner)
  partition = _field_split_partition(system, preconditioner)
  size(system.matrix, 1) >= _preconditioner_min_dofs(preconditioner) || return false
  !isempty(partition.primary_indices) || return false
  !isempty(partition.schur_indices) || return false
  return true
end

# Cached preconditioner construction for one assembled system.

# Lazily build and cache the operator associated with a preconditioner
# configuration. The cache lives on the assembled system so repeated solves on
# the same matrix can reuse the expensive local and coarse factorizations.
function _build_preconditioner_operator(system::AffineSystem{T},
                                        preconditioner::AdditiveSchwarzPreconditioner) where {T<:AbstractFloat}
  style = _preconditioned_krylov_style(system, preconditioner)
  return _build_two_level_additive_schwarz(system.matrix, system.solve_topology, style)
end

function _build_preconditioner_operator(system::AffineSystem{T},
                                        ::SmoothedAggregationAMGPreconditioner) where {T<:AbstractFloat}
  return aspreconditioner(smoothed_aggregation(system.matrix))
end

function _build_preconditioner_operator(system::AffineSystem{T},
                                        preconditioner::ILUPreconditioner) where {T<:AbstractFloat}
  return _build_ordered_ilu_operator(system.matrix, preconditioner)
end

function _build_preconditioner_operator(system::AffineSystem{T},
                                        preconditioner::FieldSplitSchurPreconditioner) where {T<:AbstractFloat}
  partition = _field_split_partition(system, preconditioner)
  isempty(partition.primary_indices) &&
    throw(ArgumentError("field-split primary block leaves no reduced dofs in this system"))
  isempty(partition.schur_indices) &&
    throw(ArgumentError("field-split Schur block leaves no reduced dofs in this system"))
  return _build_field_split_schur(system, partition, preconditioner)
end

function _preconditioner_operator(system::AffineSystem{T},
                                  preconditioner::_AbstractPreconditioner) where {T<:AbstractFloat}
  return get!(system.preconditioner_cache, preconditioner) do
    _build_preconditioner_operator(system, preconditioner)
  end
end

# Field-split Schur preconditioner setup on reduced-system blocks.

function _field_split_group_ids(preconditioner::FieldSplitSchurPreconditioner)
  return Set{UInt64}(_field_group_ids(preconditioner.primary_fields)),
         Set{UInt64}(_field_group_ids(preconditioner.schur_fields))
end

function _check_field_split_groups(system::AffineSystem, primary_field_ids::Set{UInt64},
                                   schur_field_ids::Set{UInt64})
  layout_field_ids = Set{UInt64}(_field_id(slot.field) for slot in system.layout.slots)
  issubset(primary_field_ids, layout_field_ids) ||
    throw(ArgumentError("primary_fields must belong to the assembled system layout"))
  issubset(schur_field_ids, layout_field_ids) ||
    throw(ArgumentError("schur_fields must belong to the assembled system layout"))
  union(primary_field_ids, schur_field_ids) == layout_field_ids ||
    throw(ArgumentError("field split groups must form a complete partition of the assembled system layout"))
  return nothing
end

function _reduced_field_ids(system::AffineSystem)
  field_ids = Vector{UInt64}(undef, length(system.solve_dofs))
  slot_index = 1
  slot = system.layout.slots[slot_index]
  slot_last_dof = slot.offset + slot.dof_count - 1

  for reduced_index in eachindex(system.solve_dofs)
    global_dof = system.solve_dofs[reduced_index]

    while global_dof > slot_last_dof
      slot_index += 1
      slot = system.layout.slots[slot_index]
      slot_last_dof = slot.offset + slot.dof_count - 1
    end

    field_ids[reduced_index] = _field_id(slot.field)
  end

  return field_ids
end

# Resolve a field-split configuration to reduced-system row/column indices. The
# split is validated against the assembled layout so callers do not have to
# reason about Dirichlet-eliminated or statically condensed dofs themselves.
function _field_split_partition(system::AffineSystem, preconditioner::FieldSplitSchurPreconditioner)
  primary_field_ids, schur_field_ids = _field_split_group_ids(preconditioner)
  _check_field_split_groups(system, primary_field_ids, schur_field_ids)

  primary_indices = Int[]
  schur_indices = Int[]

  for (reduced_index, field_id) in enumerate(_reduced_field_ids(system))
    if field_id in primary_field_ids
      push!(primary_indices, reduced_index)
    else
      push!(schur_indices, reduced_index)
    end
  end

  return _FieldSplitPartition(primary_indices, schur_indices)
end

# Restrict the patch topology of the full reduced system to one field block.
# This lets the field-split preconditioner reuse the same additive-Schwarz
# implementation that the package already trusts for whole systems.
function _restricted_solve_topology(topology::_PatchSolveTopology{T},
                                    kept_indices::Vector{Int}) where {T<:AbstractFloat}
  remap = zeros(Int, size(topology.coarse_prolongation, 1))

  for local_index in eachindex(kept_indices)
    remap[kept_indices[local_index]] = local_index
  end

  leaf_patches = Vector{Vector{Int}}(undef, length(topology.leaf_patches))

  for patch_index in eachindex(topology.leaf_patches)
    patch = topology.leaf_patches[patch_index]
    restricted = Int[]
    sizehint!(restricted, length(patch))

    for reduced_dof in patch
      local_dof = remap[reduced_dof]
      local_dof == 0 || push!(restricted, local_dof)
    end

    leaf_patches[patch_index] = restricted
  end

  return _PatchSolveTopology{T}(leaf_patches, topology.coarse_prolongation[kept_indices, :])
end

function _field_split_blocks(system::AffineSystem{T},
                             partition::_FieldSplitPartition) where {T<:AbstractFloat}
  primary_indices = partition.primary_indices
  schur_indices = partition.schur_indices
  return _FieldSplitBlocks(system.matrix[primary_indices, primary_indices],
                           system.matrix[primary_indices, schur_indices],
                           system.matrix[schur_indices, primary_indices],
                           system.matrix[schur_indices, schur_indices],
                           _restricted_solve_topology(system.solve_topology, primary_indices))
end

_has_patch_dofs(topology::_PatchSolveTopology) = any(!isempty, topology.leaf_patches)

# Build the primary block operator used inside the Schur preconditioner. Large
# blocks reuse additive Schwarz; smaller ones fall back to a direct block solve
# because that is usually cheaper and avoids adding a second level of Krylov
# heuristics to the preconditioner application itself.
function _build_primary_block_operator(matrix_data::SparseMatrixCSC{T,Int},
                                       topology::_PatchSolveTopology{T},
                                       preconditioner::AdditiveSchwarzPreconditioner) where {T<:AbstractFloat}
  if size(matrix_data, 1) >= preconditioner.min_dofs && _has_patch_dofs(topology)
    return _build_two_level_additive_schwarz(matrix_data, topology,
                                             _matrix_krylov_style(matrix_data))
  end

  return _build_ordered_direct_operator(matrix_data)
end

# Build the full field-split Schur operator from the assembled matrix and the
# reduced-system partition induced by the user-selected field groups.
function _build_field_split_schur(system::AffineSystem{T}, partition::_FieldSplitPartition,
                                  preconditioner::FieldSplitSchurPreconditioner) where {T<:AbstractFloat}
  blocks = _field_split_blocks(system, partition)
  primary_operator = _build_primary_block_operator(blocks.primary_matrix, blocks.primary_topology,
                                                   preconditioner.primary_preconditioner)
  schur_operator = _build_ordered_direct_operator(_approximate_schur_matrix(blocks.primary_matrix,
                                                                            blocks.coupling12,
                                                                            blocks.coupling21,
                                                                            blocks.schur_block))
  primary_rhs = zeros(T, length(partition.primary_indices))
  primary_solution = zeros(T, length(partition.primary_indices))
  schur_rhs = zeros(T, length(partition.schur_indices))
  schur_solution = zeros(T, length(partition.schur_indices))
  apply_rhs = zeros(T, size(system.matrix, 1))
  return _FieldSplitSchurOperator(copy(partition.primary_indices), copy(partition.schur_indices),
                                  blocks.coupling21, primary_operator, schur_operator, primary_rhs,
                                  primary_solution, schur_rhs, schur_solution, apply_rhs)
end

# Build a sparse matrix approximation to the Schur complement that depends only
# on assembled matrix blocks. Replacing `A₁₁⁻¹` by a safely regularized diagonal
# inverse keeps the approximation generic across mixed systems without baking
# PDE-specific assumptions into the solver core.
function _approximate_schur_matrix(primary_matrix::SparseMatrixCSC{T,Int},
                                   coupling12::SparseMatrixCSC{T,Int},
                                   coupling21::SparseMatrixCSC{T,Int},
                                   schur_block::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
  scaled_coupling12 = _scale_sparse_rows(coupling12, _safe_inverse_diagonal(primary_matrix))
  approximation = sparse(schur_block - coupling21 * scaled_coupling12)
  dropzeros!(approximation)
  return approximation
end

# Form a positive diagonal surrogate for `A₁₁⁻¹`. Tiny or vanishing diagonal
# entries fall back to row-sum scaling so the Schur approximation remains
# defined even when the primary block is not diagonally dominant.
function _safe_inverse_diagonal(matrix_data::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
  diagonal = Vector{T}(diag(matrix_data))
  row_sums = vec(sum(abs, matrix_data; dims=2))
  reference = isempty(row_sums) ? one(T) : maximum(row_sums)
  tolerance = max(reference, one(T)) * 100 * eps(T)
  inverse_diagonal = similar(diagonal)

  for index in eachindex(diagonal)
    scale = abs(diagonal[index])
    scale > tolerance || (scale = max(row_sums[index], tolerance))
    inverse_diagonal[index] = inv(scale)
  end

  return inverse_diagonal
end

# Left multiplication by a diagonal matrix amounts to scaling the sparse rows in
# place. This avoids building an explicit sparse diagonal matrix during Schur
# approximation setup.
function _scale_sparse_rows(matrix_data::SparseMatrixCSC{T,Int},
                            row_scale::AbstractVector{T}) where {T<:AbstractFloat}
  size(matrix_data, 1) == length(row_scale) ||
    throw(ArgumentError("row scaling must match the sparse matrix row count"))
  scaled = copy(matrix_data)
  rows = rowvals(scaled)
  values = nonzeros(scaled)

  for pointer in eachindex(values)
    values[pointer] *= row_scale[rows[pointer]]
  end

  return scaled
end

# Build a small ordered direct solve for a block matrix. The explicit ordering
# keeps block factorizations aligned with the package-wide direct-solve policy
# while still allowing repeated `ldiv!` applications without allocation.
function _build_ordered_factor_operator(factor,
                                        matrix_data::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
  ordering, inverse_ordering = _solve_ordering(matrix_data)
  ordered_matrix = matrix_data[ordering, ordering]
  ordered_rhs = zeros(T, size(matrix_data, 1))
  ordered_solution = zeros(T, size(matrix_data, 1))
  apply_rhs = zeros(T, size(matrix_data, 1))
  return _OrderedFactorOperator(factor(ordered_matrix), ordering, inverse_ordering, ordered_rhs,
                                ordered_solution, apply_rhs)
end

function _build_ordered_direct_operator(matrix_data::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
  return _build_ordered_factor_operator(matrix_data) do ordered_matrix
    _factorize_sparse_direct(ordered_matrix, _matrix_krylov_style(matrix_data))
  end
end

function _build_ordered_ilu_operator(matrix_data::SparseMatrixCSC{T,Int},
                                     preconditioner::ILUPreconditioner) where {T<:AbstractFloat}
  return _build_ordered_factor_operator(matrix_data) do ordered_matrix
    ilu(ordered_matrix; (; Symbol("τ") => preconditioner.tau)...)
  end
end

# Additive Schwarz construction, patch factorizations, and coarse-space setup.

function _factorize_sparse_direct(matrix_data::SparseMatrixCSC{T,Int},
                                  ::_SymmetricKrylovStyle) where {T<:AbstractFloat}
  try
    return cholesky(Symmetric(matrix_data))
  catch
  end

  return lu(matrix_data)
end

function _factorize_sparse_direct(matrix_data::SparseMatrixCSC{T,Int},
                                  ::_GeneralKrylovStyle) where {T<:AbstractFloat}
  return lu(matrix_data)
end

# Gather and scatter helpers used by the block preconditioners. Keeping the
# index motion explicit avoids temporary views and keeps the operator
# application logic readable.
function _gather_entries!(target::AbstractVector, source::AbstractVector, indices::Vector{Int})
  length(target) == length(indices) ||
    throw(ArgumentError("gather target length must match the index count"))

  for local_index in eachindex(indices)
    target[local_index] = source[indices[local_index]]
  end

  return target
end

function _scatter_entries!(target::AbstractVector, source::AbstractVector, indices::Vector{Int})
  length(source) == length(indices) ||
    throw(ArgumentError("scatter source length must match the index count"))

  for local_index in eachindex(indices)
    target[indices[local_index]] = source[local_index]
  end

  return target
end

# Build the two-level additive Schwarz preconditioner:
# - one dense local solve per leaf patch,
# - plus a low-order geometric coarse space assembled by prolongation.
function _build_two_level_additive_schwarz(matrix_data::SparseMatrixCSC{T,Int},
                                           topology::_PatchSolveTopology{T},
                                           style::_KrylovSolveStyle) where {T<:AbstractFloat}
  patches = _build_schwarz_patches(matrix_data, topology.leaf_patches, style)
  coarse_prolongation = _independent_coarse_prolongation(topology.coarse_prolongation)
  coarse_factor = _build_schwarz_coarse_factor(matrix_data, coarse_prolongation, style)
  worker_count = max(1, min(Threads.nthreads(), length(patches)))
  max_patch_size = isempty(patches) ? 1 : maximum(length(patch.dofs) for patch in patches)
  thread_buffers = [_SchwarzThreadBuffer(zeros(T, size(matrix_data, 1)), zeros(T, max_patch_size))
                    for _ in 1:worker_count]
  coarse_rhs = zeros(T, size(coarse_prolongation, 2))
  coarse_solution = zeros(T, size(coarse_prolongation, 2))
  apply_rhs = zeros(T, size(matrix_data, 1))
  return _AdditiveSchwarzOperator(patches, coarse_prolongation, coarse_factor, thread_buffers,
                                  coarse_rhs, coarse_solution, apply_rhs)
end

# Remove linearly dependent coarse basis columns so the coarse solve remains
# well posed.
function _independent_coarse_prolongation(coarse_prolongation::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
  column_count = size(coarse_prolongation, 2)
  column_count == 0 && return coarse_prolongation
  factorization = qr(coarse_prolongation)
  diagonal = abs.(diag(factorization.R))
  maximum_diagonal = isempty(diagonal) ? zero(T) : maximum(diagonal)
  tolerance = max(maximum_diagonal, one(T)) * max(size(coarse_prolongation)...) * 100 * eps(T)
  rank_count = count(>=(tolerance), diagonal)
  rank_count == column_count && return coarse_prolongation
  rank_count == 0 && return spzeros(T, size(coarse_prolongation, 1), 0)
  selected_columns = sort!(collect(factorization.pcol[1:rank_count]))
  return coarse_prolongation[:, selected_columns]
end

function _build_schwarz_coarse_factor(matrix_data::SparseMatrixCSC{T,Int},
                                      coarse_prolongation::SparseMatrixCSC{T,Int},
                                      style::_KrylovSolveStyle) where {T<:AbstractFloat}
  size(coarse_prolongation, 2) == 0 && return nothing
  coarse_matrix = sparse(transpose(coarse_prolongation) * matrix_data * coarse_prolongation)
  return _factorize_sparse_direct(coarse_matrix, style)
end

function _schwarz_patch_factor_type(::Type{T}, ::_SymmetricKrylovStyle) where {T<:AbstractFloat}
  Cholesky{T,Matrix{T}}
end
function _schwarz_patch_factor_type(::Type{T}, ::_GeneralKrylovStyle) where {T<:AbstractFloat}
  LU{T,Matrix{T},Vector{Int}}
end

function _empty_schwarz_patches(::Type{T}, style::_KrylovSolveStyle) where {T<:AbstractFloat}
  return _SchwarzPatch{T,_schwarz_patch_factor_type(T, style)}[]
end

# Build all patch factorizations, in parallel when worthwhile.
function _build_schwarz_patches(matrix_data::SparseMatrixCSC{T,Int},
                                raw_patches::Vector{Vector{Int}},
                                style::_KrylovSolveStyle) where {T<:AbstractFloat}
  active = [patch for patch in raw_patches if !isempty(patch)]
  isempty(active) && return _empty_schwarz_patches(T, style)
  worker_count = min(Threads.nthreads(), length(active))
  built = [_empty_schwarz_patches(T, style) for _ in 1:worker_count]
  _run_chunks_with_scratch!(built, length(active)) do worker_patches, first_patch, last_patch
    for patch_index in first_patch:last_patch
      push!(worker_patches, _build_schwarz_patch(matrix_data, active[patch_index], style))
    end
  end

  patches = _empty_schwarz_patches(T, style)

  for worker_patches in built
    append!(patches, worker_patches)
  end

  return patches
end

# Factor one dense patch matrix for later repeated preconditioner application.
function _factorize_dense_patch!(block::Matrix{T}, ::_SymmetricKrylovStyle) where {T<:AbstractFloat}
  _symmetrize_dense!(block)
  return _with_serialized_blas() do
    cholesky!(Symmetric(block))
  end
end

function _factorize_dense_patch!(block::Matrix{T}, ::_GeneralKrylovStyle) where {T<:AbstractFloat}
  return _with_serialized_blas() do
    lu!(block)
  end
end

function _build_schwarz_patch(matrix_data::SparseMatrixCSC{T,Int}, dofs::Vector{Int},
                              style::_KrylovSolveStyle) where {T<:AbstractFloat}
  block = Matrix(matrix_data[dofs, dofs])
  factorization = _factorize_dense_patch!(block, style)
  return _SchwarzPatch(dofs, factorization)
end

# Average the strictly upper and lower parts of a dense matrix in place. This is
# used before Cholesky factorization to remove small assembly asymmetries.
function _symmetrize_dense!(matrix_data::Matrix{T}) where {T<:AbstractFloat}
  size(matrix_data, 1) == size(matrix_data, 2) || return matrix_data

  for col in 1:size(matrix_data, 2)
    for row in (col+1):size(matrix_data, 1)
      value = (matrix_data[row, col] + matrix_data[col, row]) / 2
      matrix_data[row, col] = value
      matrix_data[col, row] = value
    end
  end

  return matrix_data
end

_buffered_operator_name(::_AdditiveSchwarzOperator) = "Schwarz preconditioner"
_buffered_operator_name(::_OrderedFactorOperator) = "ordered factor"
_buffered_operator_name(::_FieldSplitSchurOperator) = "field-split Schur preconditioner"

function _check_buffered_operator_dimensions(result::AbstractVector, rhs_data::AbstractVector,
                                             operator::_BufferedLinearOperator)
  length(result) == length(rhs_data) == length(operator.apply_rhs) ||
    throw(ArgumentError("$(_buffered_operator_name(operator)) dimensions must match the system"))
  return nothing
end

function _check_buffered_operator_dimensions(rhs_data::AbstractVector,
                                             operator::_BufferedLinearOperator)
  length(rhs_data) == length(operator.apply_rhs) ||
    throw(ArgumentError("$(_buffered_operator_name(operator)) dimensions must match the system"))
  return nothing
end

# In-place application of cached buffered operators.

# Apply the two-level additive Schwarz preconditioner to `rhs_data`.
function LinearAlgebra.ldiv!(result::AbstractVector{T}, preconditioner::_AdditiveSchwarzOperator{T},
                             rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  _check_buffered_operator_dimensions(result, rhs_data, preconditioner)
  fill!(result, zero(T))

  if preconditioner.coarse_factor !== nothing
    mul!(preconditioner.coarse_rhs, transpose(preconditioner.coarse_prolongation), rhs_data)
    ldiv!(preconditioner.coarse_solution, preconditioner.coarse_factor, preconditioner.coarse_rhs)
    mul!(result, preconditioner.coarse_prolongation, preconditioner.coarse_solution, one(T), one(T))
  end

  _apply_schwarz_patches!(result, preconditioner, rhs_data)
  return result
end

# Apply all local Schwarz patch solves and accumulate their contributions.
function _apply_schwarz_patches!(result::AbstractVector{T},
                                 preconditioner::_AdditiveSchwarzOperator{T},
                                 rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  patch_count = length(preconditioner.patches)
  patch_count == 0 && return result
  worker_count = min(length(preconditioner.thread_buffers), patch_count)

  if worker_count == 1
    buffer = preconditioner.thread_buffers[1]
    _apply_schwarz_patch_range!(result, rhs_data, preconditioner.patches, 1, patch_count,
                                buffer.rhs)
    return result
  end

  for worker in 1:worker_count
    fill!(preconditioner.thread_buffers[worker].output, zero(T))
  end

  _run_chunks_with_scratch!(preconditioner.thread_buffers,
                            patch_count) do buffer, first_patch, last_patch
    _apply_schwarz_patch_range!(buffer.output, rhs_data, preconditioner.patches, first_patch,
                                last_patch, buffer.rhs)
  end

  for worker in 1:worker_count
    result .+= preconditioner.thread_buffers[worker].output
  end

  return result
end

function _apply_schwarz_patch_range!(result::AbstractVector{T}, rhs_data::AbstractVector{T},
                                     patches, first_patch::Int, last_patch::Int,
                                     rhs_buffer::Vector{T}) where {T<:AbstractFloat}
  for patch_index in first_patch:last_patch
    patch = patches[patch_index]
    local_rhs = view(rhs_buffer, 1:length(patch.dofs))

    for local_index in eachindex(patch.dofs)
      local_rhs[local_index] = rhs_data[patch.dofs[local_index]]
    end

    _with_serialized_blas() do
      ldiv!(patch.factor, local_rhs)
    end

    for local_index in eachindex(patch.dofs)
      result[patch.dofs[local_index]] += local_rhs[local_index]
    end
  end

  return result
end

# Apply one ordered direct block solve.
function LinearAlgebra.ldiv!(result::AbstractVector{T}, operator::_OrderedFactorOperator{T},
                             rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  _check_buffered_operator_dimensions(result, rhs_data, operator)
  _gather_entries!(operator.ordered_rhs, rhs_data, operator.ordering)
  ldiv!(operator.ordered_solution, operator.factor, operator.ordered_rhs)
  _gather_entries!(result, operator.ordered_solution, operator.inverse_ordering)
  return result
end

# Apply the lower-triangular field-split Schur preconditioner:
# first approximate the primary block solve, then correct the Schur residual and
# solve the Schur block directly.
function LinearAlgebra.ldiv!(result::AbstractVector{T}, preconditioner::_FieldSplitSchurOperator{T},
                             rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  _check_buffered_operator_dimensions(result, rhs_data, preconditioner)
  fill!(result, zero(T))
  _gather_entries!(preconditioner.primary_rhs, rhs_data, preconditioner.primary_indices)
  ldiv!(preconditioner.primary_solution, preconditioner.primary_operator,
        preconditioner.primary_rhs)
  _gather_entries!(preconditioner.schur_rhs, rhs_data, preconditioner.schur_indices)
  mul!(preconditioner.schur_rhs, preconditioner.coupling21, preconditioner.primary_solution,
       -one(T), one(T))
  ldiv!(preconditioner.schur_solution, preconditioner.schur_operator, preconditioner.schur_rhs)
  _scatter_entries!(result, preconditioner.primary_solution, preconditioner.primary_indices)
  _scatter_entries!(result, preconditioner.schur_solution, preconditioner.schur_indices)
  return result
end

function LinearAlgebra.ldiv!(operator::_BufferedLinearOperator{T},
                             rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  _check_buffered_operator_dimensions(rhs_data, operator)
  copyto!(operator.apply_rhs, rhs_data)
  ldiv!(rhs_data, operator, operator.apply_rhs)
  return rhs_data
end

# This file defines the high-level problem containers that users populate before
# compilation. It keeps operator callbacks, quadrature overrides, embedded
# surfaces, and global constraints together in one mutable description that can
# later be compiled into an `AssemblyPlan`.
#
# The compiled `HpSpace` and `FieldLayout` layers fix where degrees of freedom
# live and how global coefficient vectors are partitioned. The job of this file
# is different: it describes which weak forms, boundary data, interface terms,
# embedded-surface terms, and global constraints should act on those unknowns.
#
# The file is organized in the same order in which a reader usually encounters
# the problem description.
#
# First come small selectors and constraint objects such as `BoundaryFace`,
# `Dirichlet`, and `MeanValue`.
#
# Next comes the shared mutable storage used by the public `AffineProblem` and
# `ResidualProblem` wrappers. The storage is deliberately internal: users build
# problems with `add_cell!`, `add_boundary!`, `add_interface!`, `add_surface!`,
# `add_*_quadrature!`, and `add_constraint!`, while later compilation stages
# work with one normalized representation.
#
# After that, the file provides the incremental mutation API used to build
# problems operator by operator and constraint by constraint.
#
# Finally, the file ends with the no-op callback definitions that constitute the
# operator extension surface of the package. User-defined operator types gain
# meaning by specializing these methods.

# Boundary selectors and global constraint types.

"""
    BoundaryFace(axis, side)

Identify one Cartesian domain face by axis and `LOWER`/`UPPER` side.

`axis` is numbered from `1` to the problem dimension `D`, and `side` selects
the lower or upper face orthogonal to that axis. `BoundaryFace` is used to
attach physical boundary operators and strong Dirichlet constraints to the
boundary of a Cartesian domain.

On a periodic axis, the same geometric face identifiers still exist, but they
do not represent physical boundaries. Accordingly, problem validation rejects
boundary operators and Dirichlet constraints placed on periodic faces.
"""
struct BoundaryFace
  axis::Int
  side::Int

  BoundaryFace(axis::Int, side::Int) = new(_checked_positive(axis, "axis"), _checked_side(side))
end

function BoundaryFace(axis::Integer, side::Integer)
  BoundaryFace(_checked_positive(axis, "axis"), _checked_side(side))
end

@inline _all_dirichlet_components(field::AbstractField) = ntuple(identity, component_count(field))

function _checked_dirichlet_components(field::AbstractField, components::Tuple{Vararg{Integer}})
  isempty(components) && throw(ArgumentError("Dirichlet component selectors must not be empty"))
  component_total = component_count(field)
  previous = 0
  checked = Vector{Int}(undef, length(components))

  for index in eachindex(components)
    component = components[index]
    1 <= component <= component_total ||
      throw(ArgumentError("Dirichlet component selector $component must lie in 1:$component_total"))
    component > previous ||
      throw(ArgumentError("Dirichlet component selectors must be strictly increasing and unique"))
    checked_component = Int(component)
    checked[index] = checked_component
    previous = checked_component
  end

  return Tuple(checked)
end

"""
    Dirichlet(field, boundary, data)
    Dirichlet(field, boundary, component, data)
    Dirichlet(field, boundary, components, data)

Strong Dirichlet data for selected components of `field` on `boundary`. `data`
may be a constant or a callable of the physical point.

This constraint prescribes the trace of `field` on the selected physical
boundary face. During compilation, the library projects the prescribed boundary
data onto the active trace degrees of freedom and then eliminates those dofs
from the global affine or residual problem.

When the component selector is omitted, the constraint applies to every
component of `field`. When one component index or a tuple of indices is given,
the data are interpreted relative to that selected subset. For convenience,
vector-valued data that match the full field component count are also accepted,
in which case the selected entries are extracted by their absolute component
indices.

As in the rest of the boundary machinery, Dirichlet constraints are only valid
on non-periodic boundary faces whose normal axis uses `:cg` continuity in the
owning `HpSpace`. For DG boundary treatment, use boundary/interface operators
instead of strong Dirichlet elimination.

The value stored in `data` is intentionally flexible. It may be a scalar,
vector, or callable object, depending on the field and on the intended boundary
data. Later compilation stages interpret it through quadrature and trace-basis
evaluation rather than by assuming one particular concrete representation here.
"""
struct Dirichlet
  field::AbstractField
  boundary::BoundaryFace
  components::Tuple{Vararg{Int}}
  data

  function Dirichlet(field::AbstractField, boundary::BoundaryFace,
                     components::Tuple{Vararg{Integer}}, data)
    return new(field, boundary, _checked_dirichlet_components(field, components), data)
  end
end

function Dirichlet(field::AbstractField, boundary::BoundaryFace, data)
  Dirichlet(field, boundary, _all_dirichlet_components(field), data)
end
function Dirichlet(field::AbstractField, boundary::BoundaryFace, component::Integer, data)
  Dirichlet(field, boundary, (component,), data)
end

"""
    MeanValue(field, target)

Constrain the domain average of `field` to `target`.

For a scalar field `u_h`, this enforces

  (1 / |Ω|) ∫_Ω u_h dΩ = target.

For vector-valued fields, the constraint is applied componentwise. Mean-value
constraints are useful for removing null spaces, for example the additive
pressure constant in incompressible flow problems.

Unlike Dirichlet data, a mean-value constraint is global: it does not refer to
one boundary face or one geometric sub-entity, but to the domain integral of
the selected field over the whole physical domain.
"""
struct MeanValue
  field::AbstractField
  target
end

# Internal symbolic selector used by tagged embedded-surface attachments and
# operators. `nothing` denotes the untagged wildcard behavior "all embedded
# surfaces".
const _SurfaceTag = Union{Nothing,Symbol}

# Internal wrapper pairing one boundary operator with the boundary face on which
# it acts. Boundary operators are stored separately from cell/interface/surface
# operators because they carry this extra geometric selector.
struct _BoundaryContribution{O}
  boundary::BoundaryFace
  operator::O
end

# Internal wrapper pairing one embedded-surface operator with the optional
# symbolic tag that selects the embedded surfaces on which it should act.
struct _SurfaceContribution{O}
  tag::_SurfaceTag
  operator::O
end

# Internal wrapper pairing one embedded-surface attachment with the optional
# symbolic tag under which it is registered in one problem description.
struct _SurfaceAttachment{S}
  tag::_SurfaceTag
  surface::S
end

# Internal storage for user-supplied per-cell quadrature overrides. The public
# constructor lives in `embedded.jl`, but the problem object keeps the validated
# attachments in this compact form.
struct _CellQuadratureAttachment{Q<:AbstractQuadrature}
  leaf::Int
  quadrature::Q
end

# Internal storage used by both affine and residual problem wrappers.

# Shared mutable storage behind the public affine and residual problem wrappers.
# The two problem kinds expose the same editable collections and differ only in
# which operator callbacks later evaluation routines use.
mutable struct _ProblemData
  fields::Vector{AbstractField}
  cell_operators::Vector{Any}
  boundary_operators::Vector{_BoundaryContribution}
  interface_operators::Vector{Any}
  surface_operators::Vector{_SurfaceContribution}
  cell_quadratures::Vector{_CellQuadratureAttachment}
  embedded_surfaces::Vector{_SurfaceAttachment}
  dirichlet_constraints::Vector{Dirichlet}
  mean_constraints::Vector{MeanValue}

  function _ProblemData(fields::Vector{AbstractField})
    return new(_checked_problem_fields(fields), Any[], _BoundaryContribution[], Any[],
               _SurfaceContribution[], _CellQuadratureAttachment[], _SurfaceAttachment[],
               Dirichlet[], MeanValue[])
  end
end

abstract type _AbstractProblem end

function _empty_problem_data(fields::AbstractField...)
  length(fields) >= 1 || throw(ArgumentError("at least one field is required"))
  return _ProblemData(AbstractField[fields...])
end

_problem_data(problem::_AbstractProblem) = getfield(problem, :_data)

const _PROBLEM_STORAGE_PROPERTIES = fieldnames(_ProblemData)

# Keep raw mutable storage out of the public problem API. The field remains
# reachable through Julia's reflective internals, but ordinary property access
# now points users toward the mutation/accessor functions that maintain
# invariants.
function Base.getproperty(problem::_AbstractProblem, name::Symbol)
  if name === :data || name in _PROBLEM_STORAGE_PROPERTIES
    throw(ArgumentError("problem storage is internal; use fields(problem), field_count(problem), and add_*! mutation functions"))
  end

  return getfield(problem, name)
end

function Base.setproperty!(problem::_AbstractProblem, name::Symbol, value)
  if name === :data || name === :_data || name in _PROBLEM_STORAGE_PROPERTIES
    throw(ArgumentError("problem storage is internal; build or change problems with add_*! mutation functions"))
  end

  return setfield!(problem, name, value)
end

function Base.propertynames(::_AbstractProblem, private::Bool=false)
  return private ? (:_data,) : ()
end

# Public problem wrapper types.

"""
    AffineProblem(fields...)

Container for linear operators, quadrature overrides, embedded surfaces, and
constraints that can be compiled into an `AssemblyPlan`.

`AffineProblem` is the main high-level description of a linear variational
problem in this library. It stores

- cell operators acting on volume integrals,
- boundary operators on physical boundary faces,
- interface operators on interior or periodic interfaces,
- embedded-surface operators,
- optional cell quadrature overrides and embedded surfaces,
- and global constraints such as Dirichlet and mean-value conditions.

The operators themselves are plain Julia objects. Their meaning is determined by
the callback methods they implement, such as [`cell_apply!`](@ref),
[`cell_rhs!`](@ref), [`face_apply!`](@ref), or [`interface_apply!`](@ref).

The problem object itself does not evaluate anything yet. It is purely a
mutable declaration of what should later be compiled and traversed by operator
plans.
"""
struct AffineProblem <: _AbstractProblem
  _data::_ProblemData

  function AffineProblem(fields::AbstractField...)
    return new(_empty_problem_data(fields...))
  end
end

"""
    ResidualProblem(fields...)

Container for nonlinear residual/tangent operators and constraints.

`ResidualProblem` plays the same organizational role as [`AffineProblem`](@ref),
but for nonlinear problems. Its operators contribute through residual and
tangent callbacks such as [`cell_residual!`](@ref) and
[`cell_tangent_apply!`](@ref), and the operator is interpreted as a nonlinear
residual equation together with its local linearizations.

The internal storage layout is intentionally the same as for
[`AffineProblem`](@ref). The distinction between the two problem types only
appears later when operator evaluation dispatches to affine versus nonlinear
operator callbacks.
"""
struct ResidualProblem <: _AbstractProblem
  _data::_ProblemData

  function ResidualProblem(fields::AbstractField...)
    return new(_empty_problem_data(fields...))
  end
end

"""
    fields(problem)

Return the fields that define the unknown blocks of `problem`.

The returned tuple is ordered exactly as the fields were passed to the problem
constructor.
"""
fields(problem::_AbstractProblem) = Tuple(_problem_data(problem).fields)

"""
    field_count(problem)

Return the number of fields stored in `problem`.
"""
field_count(problem::_AbstractProblem) = length(_problem_data(problem).fields)

# Validation helpers for problem containers.

# Validate that a problem is defined on at least one mutually compatible field
# and that no field descriptor appears more than once. Field identity is tracked
# by the internal field id rather than by field name.
function _checked_problem_fields(fields::Vector{AbstractField})
  length(fields) >= 1 || throw(ArgumentError("at least one field is required"))
  seen_ids = Set{UInt64}()
  reference = _field_layout_reference(field_space(fields[1]))

  for field in fields
    !(_field_id(field) in seen_ids) ||
      throw(ArgumentError("fields must be unique problem descriptors"))
    push!(seen_ids, _field_id(field))
    _check_field_layout_space(field_space(field), reference)
  end

  return fields
end

_problem_dimension(fields::AbstractVector{<:AbstractField}) = dimension(field_space(fields[1]))
_problem_reference_space(fields::AbstractVector{<:AbstractField}) = field_space(fields[1])

# Check that a referenced field belongs to the problem's declared field list.
function _check_problem_field(fields::AbstractVector{<:AbstractField}, field::AbstractField,
                              context::AbstractString)
  any(existing -> _field_id(existing) == _field_id(field), fields) ||
    throw(ArgumentError("$context field does not belong to this problem"))
  return field
end

# Check that a boundary selector is dimensionally valid for the problem.
function _check_problem_boundary(fields::AbstractVector{<:AbstractField}, boundary::BoundaryFace)
  dimension_count = _problem_dimension(fields)
  boundary.axis <= dimension_count ||
    throw(ArgumentError("boundary axis must lie in 1:$dimension_count for this problem"))
  return boundary
end

# Strengthen the boundary check to exclude periodic axes whenever a physical
# boundary object is required.
function _check_problem_physical_boundary(fields::AbstractVector{<:AbstractField},
                                          boundary::BoundaryFace, context::AbstractString)
  _check_problem_boundary(fields, boundary)
  is_periodic_axis(_problem_reference_space(fields), boundary.axis) &&
    throw(ArgumentError("$context boundary lies on a periodic axis"))
  return boundary
end

function _check_strong_dirichlet_space(field::AbstractField, boundary::BoundaryFace)
  space = field_space(field)
  is_continuous_axis(space, boundary.axis) && return boundary
  field_name_value = field_name(field)
  throw(ArgumentError("Dirichlet constraint on field $field_name_value requires :cg continuity on boundary axis $(boundary.axis); use boundary operators for DG boundaries"))
end

function _check_problem_constraint(fields::AbstractVector{<:AbstractField}, constraint::Dirichlet)
  _check_problem_field(fields, constraint.field, "Dirichlet constraint")
  _check_problem_physical_boundary(fields, constraint.boundary, "Dirichlet constraint")
  _checked_dirichlet_components(constraint.field, constraint.components)
  _check_strong_dirichlet_space(constraint.field, constraint.boundary)
  return constraint
end

function _check_problem_constraint(fields::AbstractVector{<:AbstractField}, constraint::MeanValue)
  _check_problem_field(fields, constraint.field, "mean-value constraint")
  return constraint
end

@inline _problem_constraints(data::_ProblemData, ::Dirichlet) = data.dirichlet_constraints
@inline _problem_constraints(data::_ProblemData, ::MeanValue) = data.mean_constraints

# Revalidate the stored problem data. This is mainly used by later compilation
# stages to assert that no invalid boundary or constraint data slipped into the
# problem container.
function _validate_problem_data(problem::_AbstractProblem)
  data = _problem_data(problem)
  _checked_problem_fields(data.fields)

  for wrapped in data.boundary_operators
    _check_problem_physical_boundary(data.fields, wrapped.boundary, "boundary operator")
  end

  for constraint in data.dirichlet_constraints
    _check_problem_constraint(data.fields, constraint)
  end

  for constraint in data.mean_constraints
    _check_problem_constraint(data.fields, constraint)
  end

  return problem
end

# Mutation API for building problems incrementally.

"""
    add_cell!(problem, operator)

Add a cell operator to `problem`.

For an [`AffineProblem`](@ref), cell operators contribute through callbacks such
as [`cell_apply!`](@ref) and [`cell_rhs!`](@ref). For a
[`ResidualProblem`](@ref), they contribute through [`cell_residual!`](@ref) and
[`cell_tangent_apply!`](@ref). The function returns `problem`, so calls can be
chained.
"""
function add_cell!(problem::_AbstractProblem, operator)
  push!(_problem_data(problem).cell_operators, operator)
  return problem
end

"""
    add_boundary!(problem, boundary, operator)

Add a boundary operator on the physical boundary face `boundary`.

Boundary operators act on [`FaceValues`](@ref) items and are validated
immediately against the problem dimension and periodic topology. In particular,
this function rejects boundaries that lie on periodic axes. The boundary
selector is stored alongside the operator so later compilation/evaluation stages
can route the operator only to matching physical faces.
"""
function add_boundary!(problem::_AbstractProblem, boundary::BoundaryFace, operator)
  data = _problem_data(problem)
  _check_problem_physical_boundary(data.fields, boundary, "boundary operator")
  push!(data.boundary_operators, _BoundaryContribution(boundary, operator))
  return problem
end

"""
    add_surface!(problem, operator)
    add_surface!(problem, tag, operator)

Add an embedded-surface operator to `problem`.

Surface operators act on [`SurfaceValues`](@ref) items generated from embedded
surface quadratures. The untagged form applies on all embedded surfaces in the
problem. The tagged form restricts the operator to embedded surfaces or surface
quadratures attached with the same symbolic `tag`, for example `:outer`.
"""
function add_surface!(problem::_AbstractProblem, operator)
  push!(_problem_data(problem).surface_operators, _SurfaceContribution(nothing, operator))
  return problem
end

function add_surface!(problem::_AbstractProblem, tag::Symbol, operator)
  push!(_problem_data(problem).surface_operators, _SurfaceContribution(tag, operator))
  return problem
end

"""
    add_interface!(problem, operator)

Add an interior/interface operator to `problem`.

Interface operators act on [`InterfaceValues`](@ref) items and therefore see
both sides of an interior or periodic interface.
"""
function add_interface!(problem::_AbstractProblem, operator)
  push!(_problem_data(problem).interface_operators, operator)
  return problem
end

"""
    add_constraint!(problem, constraint)

Add a global constraint to `problem`.

Supported constraint types are [`Dirichlet`](@ref) and [`MeanValue`](@ref). The
function validates that the referenced field belongs to the problem and, for
Dirichlet constraints, that the selected boundary is a non-periodic physical
boundary. Constraints are stored separately from local operators because later
compilation treats them as global algebraic conditions rather than as ordinary
cell or face contributions.
"""
function add_constraint!(problem::_AbstractProblem, constraint::Union{Dirichlet,MeanValue})
  data = _problem_data(problem)
  _check_problem_constraint(data.fields, constraint)
  push!(_problem_constraints(data, constraint), constraint)
  return problem
end

"""
    constrain!(problem, constraint)

Alias for [`add_constraint!`](@ref).
"""
function constrain!(problem::_AbstractProblem, constraint::Union{Dirichlet,MeanValue})
  add_constraint!(problem, constraint)
end

# Operator extension surface.

"""
    KernelScratch(T)
    KernelScratch{T}()

Reusable per-worker scratch storage passed to scratch-aware local kernels.

The matrix-free traversal owns one scratch object per worker slot and reuses it
across cells, faces, interfaces, and embedded surfaces. Local operators may use
[`scratch_vector`](@ref) and [`scratch_matrix`](@ref) to obtain temporary
buffers without allocating inside hot kernels.
"""
mutable struct KernelScratch{T<:AbstractFloat}
  vectors::Vector{Vector{T}}
  matrices::Vector{Matrix{T}}
end

function KernelScratch(::Type{T}) where {T<:AbstractFloat}
  KernelScratch{T}(Vector{Vector{T}}(), Vector{Matrix{T}}())
end
KernelScratch{T}() where {T<:AbstractFloat} = KernelScratch(T)

Base.eltype(::KernelScratch{T}) where {T<:AbstractFloat} = T

_scratch_slot(slot::Integer) = _checked_positive(slot, "scratch slot")

"""
    scratch_vector(scratch, slot, length)

Return a reusable vector buffer from `scratch`.

`slot` lets one kernel keep several independent temporaries alive at once. The
returned vector has exactly `length` entries. Its contents are not initialized.
"""
function scratch_vector(scratch::KernelScratch{T}, slot::Integer,
                        buffer_length::Integer) where {T<:AbstractFloat}
  checked_slot = _scratch_slot(slot)
  checked_length = _checked_nonnegative(buffer_length, "scratch vector length")
  while length(scratch.vectors) < checked_slot
    push!(scratch.vectors, T[])
  end

  buffer = scratch.vectors[checked_slot]
  resize!(buffer, checked_length)
  return buffer
end

"""
    scratch_matrix(scratch, slot, rows, columns)

Return a reusable matrix buffer from `scratch`.

The matrix has exactly `rows` by `columns` entries. Its contents are not
initialized.
"""
function scratch_matrix(scratch::KernelScratch{T}, slot::Integer, rows::Integer,
                        columns::Integer) where {T<:AbstractFloat}
  checked_slot = _scratch_slot(slot)
  checked_rows = _checked_nonnegative(rows, "scratch matrix row count")
  checked_columns = _checked_nonnegative(columns, "scratch matrix column count")

  while length(scratch.matrices) < checked_slot
    push!(scratch.matrices, Matrix{T}(undef, 0, 0))
  end

  matrix = scratch.matrices[checked_slot]
  if size(matrix, 1) != checked_rows || size(matrix, 2) != checked_columns
    matrix = Matrix{T}(undef, checked_rows, checked_columns)
    scratch.matrices[checked_slot] = matrix
  end

  return matrix
end

# These no-op methods define the operator extension surface of the library.
# User-defined operator types implement whichever callbacks are relevant for the
# geometric entities and problem class they participate in. Assembly then calls
# these methods polymorphically while traversing cells, faces, interfaces, and
# embedded surfaces.

"""
    cell_apply!(local_result, operator, values, local_coefficients)

Accumulate the matrix-free affine cell action of `operator` on one
[`CellValues`](@ref) item into `local_result`.

`local_coefficients` and `local_result` use the local coefficient numbering of
`values`. The default method does nothing. Custom affine cell operators should
overload this function when they contribute a bilinear volume term.
"""
cell_apply!(local_result, operator, values, local_coefficients) = nothing

"""
    cell_apply!(local_result, operator, values, local_coefficients, scratch)

Scratch-aware form of [`cell_apply!`](@ref). Operators can overload this method
for allocation-free high-performance kernels. The default method delegates to
the four-argument callback.
"""
function cell_apply!(local_result, operator, values, local_coefficients, scratch::KernelScratch)
  cell_apply!(local_result, operator, values, local_coefficients)
end

"""
    cell_diagonal!(local_diagonal, operator, values)

Accumulate the local diagonal of an affine cell operator into
`local_diagonal`.

This callback supplies the diagonal data used by matrix-free preconditioners.
It should add the diagonal entries of the same local bilinear form represented
by [`cell_apply!`](@ref), using the local coefficient numbering of `values`.
The method must accumulate into `local_diagonal` and leave entries belonging to
other fields or operators unchanged. If a requested Jacobi preconditioner
cannot use the available diagonal callbacks safely, it falls back to identity
preconditioning. The default method does nothing.
"""
cell_diagonal!(local_diagonal, operator, values) = nothing

"""
    cell_diagonal!(local_diagonal, operator, values, scratch)

Scratch-aware form of [`cell_diagonal!`](@ref). The default method delegates to
the three-argument callback.
"""
function cell_diagonal!(local_diagonal, operator, values, scratch::KernelScratch)
  cell_diagonal!(local_diagonal, operator, values)
end

"""
    cell_rhs!(local_rhs, operator, values)

Accumulate the affine cell right-hand-side contribution of `operator` on one
[`CellValues`](@ref) item into `local_rhs`.

The default method does nothing. Custom affine cell operators should overload
this function when they contribute a linear volume term.
"""
cell_rhs!(local_rhs, operator, values) = nothing

"""
    cell_rhs!(local_rhs, operator, values, scratch)

Scratch-aware form of [`cell_rhs!`](@ref). The default method delegates to the
three-argument callback.
"""
function cell_rhs!(local_rhs, operator, values, scratch::KernelScratch)
  cell_rhs!(local_rhs, operator, values)
end

"""
    face_apply!(local_result, operator, values, local_coefficients)

Accumulate the matrix-free affine boundary-face action of `operator` on one
[`FaceValues`](@ref) item into `local_result`.
"""
face_apply!(local_result, operator, values, local_coefficients) = nothing

function face_apply!(local_result, operator, values, local_coefficients, scratch::KernelScratch)
  face_apply!(local_result, operator, values, local_coefficients)
end

"""
    face_diagonal!(local_diagonal, operator, values)

Accumulate the local diagonal of an affine boundary-face operator into
`local_diagonal`.

This is the boundary-face analogue of [`cell_diagonal!`](@ref). It should add
the diagonal of the local bilinear form implemented by [`face_apply!`](@ref) in
the local numbering of the [`FaceValues`](@ref) item. Matrix-free
preconditioners use this method only when the local-to-global maps are direct
enough for local diagonal scattering to be valid. If those requirements are not
met, Jacobi preconditioning falls back to the identity. The default method does
nothing.
"""
face_diagonal!(local_diagonal, operator, values) = nothing

function face_diagonal!(local_diagonal, operator, values, scratch::KernelScratch)
  face_diagonal!(local_diagonal, operator, values)
end

"""
    face_rhs!(local_rhs, operator, values)

Accumulate the affine boundary-face right-hand-side contribution of `operator`
on one [`FaceValues`](@ref) item into `local_rhs`.
"""
face_rhs!(local_rhs, operator, values) = nothing

function face_rhs!(local_rhs, operator, values, scratch::KernelScratch)
  face_rhs!(local_rhs, operator, values)
end

"""
    surface_apply!(local_result, operator, values, local_coefficients)

Accumulate the matrix-free affine embedded-surface action of `operator` on one
[`SurfaceValues`](@ref) item into `local_result`.
"""
surface_apply!(local_result, operator, values, local_coefficients) = nothing

function surface_apply!(local_result, operator, values, local_coefficients, scratch::KernelScratch)
  surface_apply!(local_result, operator, values, local_coefficients)
end

"""
    surface_diagonal!(local_diagonal, operator, values)

Accumulate the local diagonal of an affine embedded-surface operator into
`local_diagonal`.

The callback should represent the diagonal part of the local bilinear surface
contribution implemented by [`surface_apply!`](@ref), using the same local
numbering as the [`SurfaceValues`](@ref) item. If a requested Jacobi
preconditioner cannot use the available diagonal callbacks safely, it falls
back to identity preconditioning.
"""
surface_diagonal!(local_diagonal, operator, values) = nothing

function surface_diagonal!(local_diagonal, operator, values, scratch::KernelScratch)
  surface_diagonal!(local_diagonal, operator, values)
end

"""
    surface_rhs!(local_rhs, operator, values)

Accumulate the affine embedded-surface right-hand-side contribution of
`operator` on one [`SurfaceValues`](@ref) item into `local_rhs`.
"""
surface_rhs!(local_rhs, operator, values) = nothing

function surface_rhs!(local_rhs, operator, values, scratch::KernelScratch)
  surface_rhs!(local_rhs, operator, values)
end

"""
    interface_apply!(local_result, operator, values, local_coefficients)

Accumulate the matrix-free affine interface action of `operator` on one
[`InterfaceValues`](@ref) item into `local_result`.
"""
interface_apply!(local_result, operator, values, local_coefficients) = nothing

function interface_apply!(local_result, operator, values, local_coefficients,
                          scratch::KernelScratch)
  interface_apply!(local_result, operator, values, local_coefficients)
end

"""
    interface_diagonal!(local_diagonal, operator, values)

Accumulate the local diagonal of an affine interface operator into
`local_diagonal`.

Interface operators see both one-sided traces in one [`InterfaceValues`](@ref)
item. This callback should add the diagonal entries of the local bilinear form
implemented by [`interface_apply!`](@ref) in that combined local numbering. It
is intended for preconditioners and therefore does not replace the full
matrix-free action; off-diagonal coupling remains the responsibility of
`interface_apply!`. If the callback cannot be used safely, Jacobi
preconditioning falls back to the identity. The default method does nothing.
"""
interface_diagonal!(local_diagonal, operator, values) = nothing

function interface_diagonal!(local_diagonal, operator, values, scratch::KernelScratch)
  interface_diagonal!(local_diagonal, operator, values)
end

"""
    interface_rhs!(local_rhs, operator, values)

Accumulate the affine interface right-hand-side contribution of `operator` on
one [`InterfaceValues`](@ref) item into `local_rhs`.
"""
interface_rhs!(local_rhs, operator, values) = nothing

function interface_rhs!(local_rhs, operator, values, scratch::KernelScratch)
  interface_rhs!(local_rhs, operator, values)
end

"""
    cell_residual!(local_residual, operator, values, state)

Accumulate the nonlinear cell residual contribution of `operator` on one
[`CellValues`](@ref) item into `local_residual`.

The current discrete state is provided in `state`. The default method does
nothing.
"""
cell_residual!(local_residual, operator, values, state) = nothing

function cell_residual!(local_residual, operator, values, state, scratch::KernelScratch)
  cell_residual!(local_residual, operator, values, state)
end

"""
    cell_tangent_apply!(local_result, operator, values, state, local_increment)

Accumulate the matrix-free cell tangent action of `operator` on one
[`CellValues`](@ref) item into `local_result`.

This is the local linearization of the residual contribution with respect to the
current `state`, applied to `local_increment`. The default method does nothing.
"""
cell_tangent_apply!(local_result, operator, values, state, local_increment) = nothing

function cell_tangent_apply!(local_result, operator, values, state, local_increment,
                             scratch::KernelScratch)
  cell_tangent_apply!(local_result, operator, values, state, local_increment)
end

"""
    face_residual!(local_residual, operator, values, state)

Accumulate the nonlinear boundary-face residual contribution of `operator` on
one [`FaceValues`](@ref) item into `local_residual`.
"""
face_residual!(local_residual, operator, values, state) = nothing

function face_residual!(local_residual, operator, values, state, scratch::KernelScratch)
  face_residual!(local_residual, operator, values, state)
end

"""
    face_tangent_apply!(local_result, operator, values, state, local_increment)

Accumulate the matrix-free boundary-face tangent action of `operator` on one
[`FaceValues`](@ref) item into `local_result`.
"""
face_tangent_apply!(local_result, operator, values, state, local_increment) = nothing

function face_tangent_apply!(local_result, operator, values, state, local_increment,
                             scratch::KernelScratch)
  face_tangent_apply!(local_result, operator, values, state, local_increment)
end

"""
    surface_residual!(local_residual, operator, values, state)

Accumulate the nonlinear embedded-surface residual contribution of `operator`
on one [`SurfaceValues`](@ref) item into `local_residual`.
"""
surface_residual!(local_residual, operator, values, state) = nothing

function surface_residual!(local_residual, operator, values, state, scratch::KernelScratch)
  surface_residual!(local_residual, operator, values, state)
end

"""
    surface_tangent_apply!(local_result, operator, values, state, local_increment)

Accumulate the matrix-free embedded-surface tangent action of `operator` on one
[`SurfaceValues`](@ref) item into `local_result`.
"""
surface_tangent_apply!(local_result, operator, values, state, local_increment) = nothing

function surface_tangent_apply!(local_result, operator, values, state, local_increment,
                                scratch::KernelScratch)
  surface_tangent_apply!(local_result, operator, values, state, local_increment)
end

"""
    interface_residual!(local_residual, operator, values, state)

Accumulate the nonlinear interface residual contribution of `operator` on one
[`InterfaceValues`](@ref) item into `local_residual`.
"""
interface_residual!(local_residual, operator, values, state) = nothing

function interface_residual!(local_residual, operator, values, state, scratch::KernelScratch)
  interface_residual!(local_residual, operator, values, state)
end

"""
    interface_tangent_apply!(local_result, operator, values, state, local_increment)

Accumulate the matrix-free interface tangent action of `operator` on one
[`InterfaceValues`](@ref) item into `local_result`.
"""
interface_tangent_apply!(local_result, operator, values, state, local_increment) = nothing

function interface_tangent_apply!(local_result, operator, values, state, local_increment,
                                  scratch::KernelScratch)
  interface_tangent_apply!(local_result, operator, values, state, local_increment)
end

const _DEFAULT_CELL_DIAGONAL_METHOD = which(cell_diagonal!, Tuple{Any,Any,Any})
const _DEFAULT_FACE_DIAGONAL_METHOD = which(face_diagonal!, Tuple{Any,Any,Any})
const _DEFAULT_SURFACE_DIAGONAL_METHOD = which(surface_diagonal!, Tuple{Any,Any,Any})
const _DEFAULT_INTERFACE_DIAGONAL_METHOD = which(interface_diagonal!, Tuple{Any,Any,Any})
const _DEFAULT_CELL_APPLY_METHOD = which(cell_apply!, Tuple{Any,Any,Any,Any})
const _DEFAULT_FACE_APPLY_METHOD = which(face_apply!, Tuple{Any,Any,Any,Any})
const _DEFAULT_SURFACE_APPLY_METHOD = which(surface_apply!, Tuple{Any,Any,Any,Any})
const _DEFAULT_INTERFACE_APPLY_METHOD = which(interface_apply!, Tuple{Any,Any,Any,Any})
const _DEFAULT_CELL_DIAGONAL_SCRATCH_METHOD = which(cell_diagonal!,
                                                    Tuple{Any,Any,Any,KernelScratch})
const _DEFAULT_FACE_DIAGONAL_SCRATCH_METHOD = which(face_diagonal!,
                                                    Tuple{Any,Any,Any,KernelScratch})
const _DEFAULT_SURFACE_DIAGONAL_SCRATCH_METHOD = which(surface_diagonal!,
                                                       Tuple{Any,Any,Any,KernelScratch})
const _DEFAULT_INTERFACE_DIAGONAL_SCRATCH_METHOD = which(interface_diagonal!,
                                                         Tuple{Any,Any,Any,KernelScratch})
const _DEFAULT_CELL_APPLY_SCRATCH_METHOD = which(cell_apply!, Tuple{Any,Any,Any,Any,KernelScratch})
const _DEFAULT_FACE_APPLY_SCRATCH_METHOD = which(face_apply!, Tuple{Any,Any,Any,Any,KernelScratch})
const _DEFAULT_SURFACE_APPLY_SCRATCH_METHOD = which(surface_apply!,
                                                    Tuple{Any,Any,Any,Any,KernelScratch})
const _DEFAULT_INTERFACE_APPLY_SCRATCH_METHOD = which(interface_apply!,
                                                      Tuple{Any,Any,Any,Any,KernelScratch})

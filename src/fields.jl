# This file defines semantic field descriptors, block layouts, and concrete
# state vectors. It is the layer that connects compiled `HpSpace` objects to the
# flat coefficient arrays used by problem setup, assembly, solves, transfer,
# verification, and output.
#
# Conceptually, the file has three jobs.
#
# First, it introduces field descriptors such as scalar and vector unknowns.
# These objects say what an unknown means and on which compiled space it lives,
# but they do not own coefficient storage.
#
# Second, it combines several such descriptors into one `FieldLayout`. A layout
# fixes the global block ordering of all fields and therefore determines how one
# monolithic algebraic vector is partitioned into named field blocks and
# component blocks.
#
# Third, it provides the concrete `State` wrapper that pairs one validated
# layout with one coefficient vector. Later files use this pairing as the
# canonical representation of a discrete solution, iterate, residual state, or
# right-hand side expressed in field coordinates.
#
# There is also a simple algebraic picture worth keeping in mind. If one
# compiled `HpSpace` represents a scalar finite-element space `V_h`, then a
# `ScalarField` is one copy of `V_h`, a `VectorField` with `m` components is the
# product space `V_h^m`, and a `FieldLayout` is the direct sum of all such field
# blocks. A `State` is then just one coefficient vector in the basis induced by
# that direct-sum ordering. The rest of the library repeatedly moves between
# this semantic block view and the flat vector view, so making that connection
# explicit here helps later files read more naturally.
#
# The file is intentionally arranged in that same order: semantic field
# descriptors first, then validated block layouts, and finally concrete state
# vectors.

"""
    AbstractField

Abstract supertype for unknowns that live on an [`HpSpace`](@ref).

Fields describe the semantic blocks that appear in a discretized problem, for
example a scalar pressure, a concentration, or a vector-valued velocity. A
field does not store coefficient data itself; it only describes how many
components the unknown has, on which space it is defined, and under which name
it should appear in layouts, problems, and output.

Two field objects are considered distinct unknowns even if they carry the same
name and live on the same space. Internally, each field therefore receives a
stable identifier at construction time, and later layout/problem containers use
that identifier rather than the display name to decide field identity.
"""
abstract type AbstractField end

const _FIELD_ID_COUNTER = Base.RefValue{UInt64}(0)

# Field descriptors.

# Every field receives a stable internal identifier on construction. Layouts and
# problems use this identifier rather than names to decide field equality, so
# two distinct fields may legitimately share a descriptive name without being
# mistaken for one another.
@inline function _next_field_id()
  _FIELD_ID_COUNTER[] += 1
  return _FIELD_ID_COUNTER[]
end

"""
    ScalarField(space; name=:field)

Create a scalar unknown that lives on `space`.

Mathematically, this represents one coefficient vector in the scalar finite-
element space associated with `space`. The optional `name` is metadata used to
label the field in problem definitions and output routines; it does not affect
the algebraic layout.
"""
struct ScalarField <: AbstractField
  id::UInt64
  space::HpSpace
  name::Symbol
end

ScalarField(space::HpSpace; name::Symbol=:field) = ScalarField(_next_field_id(), space, name)
function ScalarField(id::Integer, space::HpSpace, name::Symbol)
  return ScalarField(UInt64(_checked_positive(id, "id")), space, name)
end

"""
    VectorField(space, components; name=:field)

Create a vector-valued unknown with `components` components on `space`.

The resulting discrete unknown is stored in block form: each component uses the
same scalar `HpSpace`, and the full vector field is represented by concatenating
the component coefficient blocks in component order. This is appropriate for
standard Cartesian vector unknowns such as velocity or displacement.

The rest of this file consistently uses that component-major convention: the
full coefficient block of component `1` comes first, then component `2`, and so
on. Layout ranges and state views are all defined with respect to this ordering.
"""
struct VectorField <: AbstractField
  id::UInt64
  space::HpSpace
  components::Int
  name::Symbol

  function VectorField(id::UInt64, space::HpSpace, components::Int, name::Symbol)
    return new(id, space, _checked_positive(components, "components"), name)
  end
end

function VectorField(space::HpSpace, components::Integer; name::Symbol=:field)
  return VectorField(_next_field_id(), space, _checked_positive(components, "components"), name)
end
function VectorField(id::Integer, space::HpSpace, components::Integer, name::Symbol)
  return VectorField(UInt64(_checked_positive(id, "id")), space,
                     _checked_positive(components, "components"), name)
end

scalar_field(space::HpSpace; name::Symbol=:field) = ScalarField(space; name)
function vector_field(space::HpSpace, components::Integer; name::Symbol=:field)
  VectorField(space, components; name)
end

"""
    field_space(field)

Return the [`HpSpace`](@ref) on which `field` is defined.

All coefficients associated with the field are interpreted relative to this
space.
"""
field_space(field::ScalarField) = field.space
field_space(field::VectorField) = field.space

"""
    field_name(field)

Return the symbolic name attached to `field`.

Names are descriptive metadata intended for diagnostics, problem definitions,
and output. They are not used to decide field identity.
"""
field_name(field::AbstractField) = field.name

"""
    component_count(field)

Return the number of components of `field`.

Scalar fields always have one component. Vector fields have as many components
as requested at construction time.
"""
component_count(::ScalarField) = 1
component_count(field::VectorField) = field.components

"""
    field_dof_count(field)
    field_dof_count(layout, field)

Return the total degree-of-freedom count associated with `field`.

For a bare field descriptor, this equals

  component_count(field) × scalar_dof_count(field_space(field)).

For a validated [`FieldLayout`](@ref), the two-argument form returns the size of
the block occupied by `field` within that particular global layout.
"""
function field_dof_count(field::AbstractField)
  component_count(field) * scalar_dof_count(field_space(field))
end

@inline _field_id(field::AbstractField) = field.id

# Field layouts.

# Internal layout descriptor for one field block inside a global state vector.
# Each slot stores the owning field, the shared space metadata used for
# validation, and the one-based offset of the field block in the concatenated
# coefficient vector.
struct _FieldSlot{D,T<:AbstractFloat}
  field::AbstractField
  space::HpSpace{D,T}
  offset::Int
  scalar_dof_count::Int
  dof_count::Int
end

"""
    FieldLayout(fields)

Combine one or more fields into a single global degree-of-freedom layout.

`FieldLayout` is the object that turns several field descriptors into one common
block structure for algebraic vectors and matrices. The fields are stored in the
order provided by the user, and their coefficient blocks are concatenated in
that same order. A layout is valid only if all fields share the same dimension,
scalar type, active-leaf topology, physical domain, and periodic topology.

In other words, a `FieldLayout` asserts that all participating fields live on
the same discrete mesh and physical domain, so that one global vector can store
their coefficients side by side without ambiguity. It is therefore the bridge
between semantic unknowns and raw algebraic storage.

Algebraically, the layout fixes the basis ordering of the direct sum of all
field spaces. Later matrix assembly, solver reconstruction, and VTK export all
rely on this one ordering to interpret slices of a monolithic vector as named
field blocks again.
"""
struct FieldLayout{D,T<:AbstractFloat}
  slots::Vector{_FieldSlot{D,T}}
  dof_count::Int

  function FieldLayout{D,T}(slots::Vector{_FieldSlot{D,T}},
                            dof_count::Int) where {D,T<:AbstractFloat}
    _validate_field_layout(slots, dof_count)
    return new{D,T}(slots, dof_count)
  end
end

# Validate the invariants required for a meaningful shared layout. A layout can
# only be formed when all participating fields live on spaces that represent the
# same discrete mesh and physical domain. The checks below also enforce that the
# block offsets form one contiguous one-based partition of the global vector.
function _validate_field_layout(slots::Vector{_FieldSlot{D,T}},
                                layout_dof_count::Int) where {D,T<:AbstractFloat}
  !isempty(slots) || throw(ArgumentError("at least one field is required"))
  seen_ids = Set{UInt64}()
  first_slot = slots[1]
  reference = _field_layout_reference(first_slot.space)
  expected_offset = 1

  for slot in slots
    field = slot.field
    space = slot.space

    _register_layout_field!(seen_ids, field)
    field_space(field) === space ||
      throw(ArgumentError("field slots must store the owning field space"))
    _check_field_layout_space(space, reference)
    slot.offset == expected_offset ||
      throw(ArgumentError("field slots must use contiguous one-based offsets"))
    scalar_dof_count(space) == slot.scalar_dof_count ||
      throw(ArgumentError("field slot scalar-dof counts must match the owning space"))
    component_count(field) * slot.scalar_dof_count == slot.dof_count ||
      throw(ArgumentError("field slot dof counts must match the field component count"))
    expected_offset += slot.dof_count
  end

  layout_dof_count == expected_offset - 1 ||
    throw(ArgumentError("layout dof count must match the field slots"))
  return nothing
end

function FieldLayout(fields)
  field_vector = _checked_layout_fields(fields)
  length(field_vector) >= 1 || throw(ArgumentError("at least one field is required"))
  return _field_layout(field_vector)
end

function _checked_layout_fields(fields)
  field_vector = AbstractField[]

  for field in fields
    field isa AbstractField || throw(ArgumentError("layout entries must be field descriptors"))
    push!(field_vector, field)
  end

  return field_vector
end

# Build the block layout in user-specified field order. The resulting offset
# convention is component-major inside each field: for a vector field, component
# 1 occupies the first scalar-space block of that field, component 2 the next,
# and so on.
function _field_layout(field_vector::Vector{AbstractField})
  length(field_vector) >= 1 || throw(ArgumentError("at least one field is required"))
  first_field = field_vector[1]
  first_space = field_space(first_field)
  reference = _field_layout_reference(first_space)
  offset = 1
  seen_ids = Set{UInt64}()
  slots = _FieldSlot{dimension(first_space),eltype(reference.origin)}[]

  for field in field_vector
    _register_layout_field!(seen_ids, field)
    space = field_space(field)
    _check_field_layout_space(space, reference)
    slot = _field_slot(field, offset)
    push!(slots, slot)
    offset += slot.dof_count
  end

  return FieldLayout{dimension(first_space),eltype(reference.origin)}(slots, offset - 1)
end

# Shared reference data that all fields in one layout must match exactly. The
# active-leaf set, physical box, periodic axes, and physical-region identity
# together identify the common discrete and geometric setting in which all field
# blocks are interpreted.
struct _FieldLayoutReference{L,O,E,P,R}
  dimension::Int
  scalar_type::DataType
  active_leaves::L
  origin::O
  extent::E
  periodic::P
  region::R
end

@inline function _field_layout_reference(space::HpSpace)
  return _FieldLayoutReference(dimension(space), eltype(origin(space)), space.active_leaves,
                               origin(space), extent(space), periodic_axes(space),
                               _physical_region(domain(space)))
end

# Check that one field space is compatible with the common layout reference.
function _check_field_layout_space(space::HpSpace, reference::_FieldLayoutReference)
  dimension(space) == reference.dimension ||
    throw(ArgumentError("all fields must use the same dimension"))
  eltype(origin(space)) == reference.scalar_type ||
    throw(ArgumentError("all fields must use the same scalar type"))
  space.active_leaves == reference.active_leaves ||
    throw(ArgumentError("all fields must share the same active-leaf topology"))
  origin(space) == reference.origin && extent(space) == reference.extent ||
    throw(ArgumentError("all fields must share the same physical domain"))
  periodic_axes(space) == reference.periodic ||
    throw(ArgumentError("all fields must share the same periodic topology"))
  _physical_region(domain(space)) === reference.region ||
    throw(ArgumentError("all fields must share the same physical region"))
  return nothing
end

@inline function _register_layout_field!(seen_ids::Set{UInt64}, field::AbstractField)
  !(_field_id(field) in seen_ids) ||
    throw(ArgumentError("fields must be unique problem descriptors"))
  push!(seen_ids, _field_id(field))
  return seen_ids
end

dimension(::FieldLayout{D}) where {D} = D
dof_count(layout::FieldLayout) = layout.dof_count

"""
    field_count(layout)

Return the number of fields stored in `layout`.

This counts field blocks, not scalar components. A two-component vector field
therefore contributes one field to the layout, not two.
"""
field_count(layout::FieldLayout) = length(layout.slots)

"""
    fields(layout)

Return the fields stored in `layout` as a tuple in layout order.

The returned order is exactly the block order used by the global coefficient
vector.
"""
fields(layout::FieldLayout) = Tuple(slot.field for slot in layout.slots)

# Materialize one field block slot and its contiguous dof span.
function _field_slot(field::AbstractField, offset::Int)
  space = field_space(field)
  scalar_count = scalar_dof_count(space)
  slot_dof_count = component_count(field) * scalar_count
  return _FieldSlot(field, space, offset, scalar_count, slot_dof_count)
end

# Resolve a field descriptor to its slot index by internal field identity. This
# avoids ambiguous lookup by name and keeps range queries insensitive to how
# fields are displayed to the user.
function _field_slot_index(layout::FieldLayout, field::AbstractField)
  field_id = _field_id(field)

  for index in eachindex(layout.slots)
    _field_id(layout.slots[index].field) == field_id && return index
  end

  throw(ArgumentError("field does not belong to this layout"))
end

# Resolve a field descriptor to its slot by internal field identity. This avoids
# ambiguous lookup by name and keeps range queries insensitive to how fields are
# displayed to the user.
@inline function _field_slot(layout::FieldLayout, field::AbstractField)
  return @inbounds layout.slots[_field_slot_index(layout, field)]
end

@inline _field_slot_range(slot::_FieldSlot) = slot.offset:(slot.offset+slot.dof_count-1)

@inline function _field_component_range(slot::_FieldSlot, field::AbstractField, component::Integer)
  checked_component = _require_index(component, component_count(field), "field component")
  first = slot.offset + (checked_component - 1) * slot.scalar_dof_count
  return first:(first+slot.scalar_dof_count-1)
end

"""
    field_dof_range(layout, field)

Return the one-based global index range occupied by `field` inside `layout`.

The returned range covers the entire block of `field`, including all of its
components. For vector-valued fields the block is stored in component-major
order, so this range is the concatenation of the component ranges returned by
[`field_component_range`](@ref).
"""
function field_dof_range(layout::FieldLayout, field::AbstractField)
  return _field_slot_range(_field_slot(layout, field))
end

field_dof_count(layout::FieldLayout, field::AbstractField) = _field_slot(layout, field).dof_count

"""
    field_component_range(layout, field, component)

Return the one-based global index range of one component of `field` inside
`layout`.

Components are numbered from `1` to `component_count(field)` and are stored in
component-major order. Each component occupies one contiguous scalar-space block
of length `scalar_dof_count(field_space(field))`.
"""
function field_component_range(layout::FieldLayout, field::AbstractField, component::Integer)
  return _field_component_range(_field_slot(layout, field), field, component)
end

"""
    State(layout)
    State(layout, coefficients)
    State(plan, coefficients)
    State(system, coefficients)

Store field coefficients on a validated [`FieldLayout`](@ref).

`State` is the concrete carrier of discrete unknown values. It pairs a layout
with a coefficient vector whose entries follow the block ordering prescribed by
that layout. The constructor does not copy the coefficient vector; it stores the
given array directly after checking that its length matches the layout.

The same `State` type is used for many algebraic roles throughout the package:
current solutions, Newton iterates, residual vectors interpreted in field
coordinates, or any other coefficient data that should be accessed by named
field blocks instead of by raw global indices.

Nothing in `State` says whether the coefficients represent primal unknowns,
test-space residuals, corrections, or reconstructed postprocessed data. The
meaning comes entirely from context; `State` only guarantees that the numbers
are organized according to one validated field layout.
"""
struct State{T<:AbstractFloat,V<:AbstractVector{T},L<:FieldLayout}
  layout::L
  coefficients::V

  function State{T,V,L}(layout::L,
                        coefficients::V) where {T<:AbstractFloat,V<:AbstractVector{T},
                                                L<:FieldLayout}
    length(coefficients) == dof_count(layout) ||
      throw(ArgumentError("coefficient vector length must match the layout dof count"))
    return new{T,V,L}(layout, coefficients)
  end
end

function State(layout::FieldLayout{D,T}) where {D,T<:AbstractFloat}
  return State(layout, zeros(T, dof_count(layout)))
end

function State(layout::FieldLayout{D,T}, coefficients::AbstractVector{T}) where {D,T<:AbstractFloat}
  return State{T,typeof(coefficients),typeof(layout)}(layout, coefficients)
end

function State(layout::FieldLayout{D,T}, coefficients::AbstractVector) where {D,T<:AbstractFloat}
  eltype(coefficients) == T ||
    throw(ArgumentError("coefficient vector element type must match the layout scalar type"))
  length(coefficients) == dof_count(layout) ||
    throw(ArgumentError("coefficient vector length must match the layout dof count"))
  throw(ArgumentError("coefficient vector must be an AbstractVector{$T}"))
end

# Concrete state vectors on validated field layouts.

"""
    field_layout(state)

Return the [`FieldLayout`](@ref) that defines the block structure of `state`.
"""
field_layout(state::State) = state.layout

"""
    coefficients(state)

Return the coefficient vector stored in `state`.

The returned array is the live storage of the state, not a copy.
"""
coefficients(state::State) = state.coefficients

"""
    field_values(state, field)

Return a view of the coefficient block of `field` inside `state`.

The view spans all components of the field in the layout ordering. For a scalar
field this is the full field coefficient vector; for a vector field it is the
concatenation of the component blocks in component-major order.
"""
function field_values(state::State, field::AbstractField)
  return view(state.coefficients, _field_slot_range(_field_slot(state.layout, field)))
end

"""
    field_component_values(state, field, component)

Return a view of one component block of `field` inside `state`.

This is the coefficient subvector corresponding to
[`field_component_range`](@ref) for the same `layout`, `field`, and `component`.
The returned object is a live view into the state's storage, so mutating it
updates the corresponding component block of the global coefficient vector.
"""
function field_component_values(state::State, field::AbstractField, component::Integer)
  slot = _field_slot(state.layout, field)
  return view(state.coefficients, _field_component_range(slot, field, component))
end

# This file defines the compiled local evaluation layer used by assembly,
# verification, and output. It stores basis values, gradients, geometric data,
# and local-to-global coefficient maps in a form that lets operators reconstruct
# fields and test functions directly on compiled cells, faces, interfaces, and
# embedded surfaces.
#
# After `space.jl` has fixed the compiled global basis and `problem.jl` has
# declared which operators should act on it, this file performs the next
# compilation step: it turns each relevant geometric entity into a reusable
# local evaluation object. The resulting local data work the same way for CG and
# DG spaces; the only difference is which sparse local-to-global mode expansions
# the compiled `HpSpace` provides.
#
# From an operator writer's perspective, the point of this file is that every
# local weak form can be written against the same interface:
#
#   u_h(x_q),  ∇u_h(x_q),  ϕ_i(x_q),  ∇ϕ_i(x_q),  w_q.
#
# All geometric pullbacks, basis evaluations, continuity substitutions, and
# local block-numbering conventions are compiled once up front.
#
# Each compiled item bundles four kinds of information:
#
# - physical quadrature points and measure weights,
# - shape-function values and physical gradients on those points,
# - the flattened local block layout seen by local assembly kernels,
# - and the sparse local-to-global coefficient maps induced by the compiled
#   `HpSpace`, whether those maps come from continuity constraints or from
#   leaf-local DG dofs.
#
# The result is a layer that lets operators talk in local finite-element terms
# such as "value of the trial field at quadrature point `q`" or "block of the
# local matrix belonging to field `u`", while the expensive geometry and basis
# bookkeeping has already been precomputed.
#
# The file is arranged in that order: local item types first, then the public
# query/evaluation API, then the compilation routines that build those items,
# and finally the low-level basis and geometry kernels on which compilation
# depends.

# Local integration item types and shared storage.

# Internal field-specific payload stored inside each local integration item.
# Besides the basis tables `values` and `gradients`, this stores the sparse
# local-to-global coefficient map for all local dofs of the field so that local
# function values can be reconstructed directly from a global `State`.
struct _FieldValues{D,T<:AbstractFloat}
  field_id::UInt64
  component_count::Int
  scalar_dof_count::Int
  local_mode_count::Int
  block::UnitRange{Int}
  values::Matrix{T}
  gradients::Array{T,3}
  term_offsets::Vector{Int}
  term_indices::Vector{Int}
  term_coefficients::Vector{T}
  single_term_indices::Vector{Int}
  single_term_coefficients::Vector{T}
end

"""
    CellValues

Local evaluation data for one active cell.

`CellValues` stores the physical quadrature points and weights of a cell, along
with precomputed basis-function values, basis gradients, and local-to-global
degree-of-freedom mappings for every field in a [`FieldLayout`](@ref). These
objects are compiled once and then reused during assembly, residual evaluation,
and post-processing.
"""
struct CellValues{D,T<:AbstractFloat,F<:Tuple}
  leaf::Int
  points::Vector{NTuple{D,T}}
  weights::Vector{T}
  fields::F
  term_offsets::Vector{Int}
  term_indices::Vector{Int}
  term_coefficients::Vector{T}
  single_term_indices::Vector{Int}
  single_term_coefficients::Vector{T}
  interior_local_dofs::Vector{Int}
  local_dof_count::Int
end

"""
    FaceValues

Local evaluation data for one boundary face of an active cell.

The stored points and weights represent a surface quadrature rule on the chosen
face, mapped to physical coordinates. In addition to the field evaluation data,
`FaceValues` stores the outward unit normal of the face and the face axis/side
identifiers used by boundary operators.
"""
struct FaceValues{D,T<:AbstractFloat,F<:Tuple}
  leaf::Int
  axis::Int
  side::Int
  normal::NTuple{D,T}
  points::Vector{NTuple{D,T}}
  weights::Vector{T}
  fields::F
  term_offsets::Vector{Int}
  term_indices::Vector{Int}
  term_coefficients::Vector{T}
  single_term_indices::Vector{Int}
  single_term_coefficients::Vector{T}
  local_dof_count::Int
end

"""
    SurfaceValues

Local evaluation data for an embedded surface quadrature item inside one cell.

Unlike [`FaceValues`](@ref), the normal may vary from quadrature point to
quadrature point, so `SurfaceValues` stores one normal vector per point. This
type is used for immersed or embedded surface terms that are not tied to a
topological cell face. The optional `tag` records the symbolic surface label
under which the underlying geometry or quadrature was attached to the problem.
For untagged attachments, `tag === nothing`.
"""
struct SurfaceValues{D,T<:AbstractFloat,F<:Tuple}
  leaf::Int
  tag::_SurfaceTag
  points::Vector{NTuple{D,T}}
  weights::Vector{T}
  normals::Vector{NTuple{D,T}}
  fields::F
  term_offsets::Vector{Int}
  term_indices::Vector{Int}
  term_coefficients::Vector{T}
  single_term_indices::Vector{Int}
  single_term_coefficients::Vector{T}
  local_dof_count::Int
end

"""
    InterfaceValues

Local evaluation data for one interior or periodic interface between two active
cells.

The interface is oriented from the minus side to the plus side. The stored
`normal` points from `minus_leaf` to `plus_leaf`. The minus and plus traces may
use different physical point arrays, which is essential for periodic interfaces:
both sides share the same tangential quadrature patch and weights, but the
physical points on the two faces need not coincide in Euclidean space.

The two sides may also carry different local basis tables and different local
mode-to-global expansions. This is what lets one interface item represent both
matching interfaces and nonmatching hp interfaces in one common format.
"""
struct InterfaceValues{D,T<:AbstractFloat,MF<:Tuple,PF<:Tuple}
  minus_leaf::Int
  plus_leaf::Int
  axis::Int
  normal::NTuple{D,T}
  points::Vector{NTuple{D,T}}
  plus_points::Vector{NTuple{D,T}}
  weights::Vector{T}
  minus_fields::MF
  plus_fields::PF
  term_offsets::Vector{Int}
  term_indices::Vector{Int}
  term_coefficients::Vector{T}
  single_term_indices::Vector{Int}
  single_term_coefficients::Vector{T}
  local_dof_count::Int
end

# Internal side view of one interface trace. This lets the public `plus` and
# `minus` accessors reuse the same field-evaluation machinery as cells, boundary
# faces, and embedded surfaces without duplicating storage.
struct _InterfaceSideValues{D,T<:AbstractFloat,F<:Tuple}
  leaf::Int
  points::Vector{NTuple{D,T}}
  weights::Vector{T}
  normal::NTuple{D,T}
  fields::F
end

const _AssemblyValues = Union{CellValues,FaceValues,SurfaceValues,InterfaceValues}
const _ConstantNormalValues = Union{FaceValues,InterfaceValues,_InterfaceSideValues}
const _FieldEvaluationValues = Union{CellValues,FaceValues,SurfaceValues,_InterfaceSideValues}
const _PointValues = Union{CellValues,FaceValues,SurfaceValues,InterfaceValues,_InterfaceSideValues}

# For each active leaf, record where its boundary-face and embedded-surface
# items live inside the globally compiled integration arrays. This lets assembly
# traverse the local data leaf by leaf without repeatedly searching the global
# vectors.
struct _LeafIntegration
  boundary_faces::UnitRange{Int}
  embedded_surfaces::UnitRange{Int}
end

# Compiled integration cache used by assembly plans. It contains one immutable
# collection of local evaluation items for each geometric entity type together
# with the leaf-local lookup data above.
struct _CompiledIntegration{CV,FV,IV,SV}
  cells::CV
  boundary_faces::FV
  interfaces::IV
  embedded_surfaces::SV
  leaves::Vector{_LeafIntegration}
end

# Generic geometric accessors on compiled local items.

"""
    point_count(values)

Return the number of quadrature points stored in a local integration item.

This method applies to compiled cell, face, interface, and embedded-surface
evaluation data. The returned count matches the length of the associated point
and weight arrays.
"""
point_count(values::_PointValues) = length(values.weights)

"""
    point(values, point_index)

Return the physical quadrature point with one-based index `point_index`.

For [`InterfaceValues`](@ref), this returns the point on the minus-side trace.
Use [`plus`](@ref) first when the plus-side point array is needed.
"""
function point(values::_PointValues, point_index::Integer)
  @inbounds values.points[_checked_index(point_index, point_count(values), "point")]
end

"""
    weight(values, point_index)

Return the physical quadrature weight associated with one point of a compiled
local integration item.

The weight already includes the geometric measure factor of the corresponding
cell, face, interface patch, or embedded-surface piece.
"""
function weight(values::_PointValues, point_index::Integer)
  @inbounds values.weights[_checked_index(point_index, point_count(values), "point")]
end

"""
    face_axis(values)

Return the coordinate axis normal to a face or interface item.

For a face orthogonal to the `a`-th coordinate direction, this function returns
`a`.
"""
face_axis(values::FaceValues) = values.axis
face_axis(values::InterfaceValues) = values.axis

"""
    face_side(values)

Return whether a boundary-face item lies on the `LOWER` or `UPPER` side of its
face axis.
"""
face_side(values::FaceValues) = values.side

"""
    normal(values)
    normal(values, point_index)

Return the unit normal associated with a local integration item.

For [`FaceValues`](@ref) and [`InterfaceValues`](@ref), the normal is constant
over all quadrature points. For [`SurfaceValues`](@ref), the pointwise form
returns the unit normal at the requested quadrature point.
"""
normal(values::_ConstantNormalValues) = values.normal
normal(values::_ConstantNormalValues, point_index::Integer) = values.normal

function normal(values::SurfaceValues, point_index::Integer)
  @inbounds values.normals[_checked_index(point_index, point_count(values), "point")]
end

function _interface_side_values(values::InterfaceValues{D,T}, leaf::Int, points,
                                fields) where {D,T<:AbstractFloat}
  return _InterfaceSideValues{D,T,typeof(fields)}(leaf, points, values.weights, values.normal,
                                                  fields)
end

"""
    plus(values)
    minus(values)

Return the plus- or minus-side trace view of an [`InterfaceValues`](@ref) item.

The returned object behaves like the other local field-evaluation containers:
its shape-function tables, local dof numbering, and point arrays refer only to
the selected side of the interface. This is useful when an interface operator
is written as two one-sided traces `u⁻` and `u⁺`.
"""
function plus(values::InterfaceValues{D,T}) where {D,T<:AbstractFloat}
  return _interface_side_values(values, values.plus_leaf, values.plus_points, values.plus_fields)
end

"""
    minus(values)

Return the minus-side trace view of an [`InterfaceValues`](@ref) item.

This is the one-sided local evaluation object associated with `minus_leaf`.
"""
function minus(values::InterfaceValues{D,T}) where {D,T<:AbstractFloat}
  return _interface_side_values(values, values.minus_leaf, values.points, values.minus_fields)
end

# DG-style trace algebra helpers used by interface kernels.

"""
    jump(minus_value, plus_value)

Return the plus-minus difference `plus_value - minus_value`.

This is a purely algebraic helper for DG-style interface kernels. Scalars are
subtracted directly. Tuples are combined componentwise, recursively, so the
same function works for scalar traces, vector traces, and vector-field
gradients represented as tuples of tuples. The minus/plus orientation matches
the orientation of [`InterfaceValues`](@ref).
"""
@inline jump(minus_value::Number, plus_value::Number) = plus_value - minus_value

function jump(minus_value::Tuple, plus_value::Tuple)
  length(minus_value) == length(plus_value) ||
    throw(ArgumentError("jump expects tuples with matching lengths"))
  return ntuple(index -> jump(minus_value[index], plus_value[index]), length(minus_value))
end

"""
    average(minus_value, plus_value)

Return the arithmetic average `0.5 * (minus_value + plus_value)`.

Like [`jump`](@ref), this helper acts componentwise on tuples and is intended
for DG-style interface kernels after the one-sided traces have been evaluated.
"""
@inline average(minus_value::Number, plus_value::Number) = (minus_value + plus_value) / 2

function average(minus_value::Tuple, plus_value::Tuple)
  length(minus_value) == length(plus_value) ||
    throw(ArgumentError("average expects tuples with matching lengths"))
  return ntuple(index -> average(minus_value[index], plus_value[index]), length(minus_value))
end

"""
    normal_component(value, normal)

Project a physical-space vector quantity onto `normal`.

If `value` is a `D`-tuple, the result is the scalar dot product with `normal`.
If `value` is a tuple of `D`-tuples, the projection is applied componentwise.
This is useful for writing DG fluxes in terms of normal derivatives without
repeating dot-product boilerplate.
"""
@inline function normal_component(value::NTuple{D,<:Number},
                                  normal_value::NTuple{D,<:Number}) where {D}
  return sum(value[axis] * normal_value[axis] for axis in 1:D)
end

function normal_component(value::NTuple{N,<:NTuple{D,<:Number}},
                          normal_value::NTuple{D,<:Number}) where {N,D}
  return ntuple(component -> normal_component(value[component], normal_value), N)
end

# Field-local numbering and basis-table queries.

# Resolve one field descriptor to the precomputed local basis tables and local-
# to-global mapping data stored inside a cell/face/interface/surface item.
function _field_values(values::_FieldEvaluationValues, field::AbstractField)
  field_id = _field_id(field)

  for data in values.fields
    data.field_id == field_id && return data
  end

  throw(ArgumentError("field does not belong to this local integration item"))
end

@inline function _field_component_offset(data::_FieldValues, component::Int)
  return (component - 1) * data.local_mode_count
end

@inline function _field_local_dof(data::_FieldValues, component::Int, mode_index::Int)
  return _field_component_offset(data, component) + mode_index
end

@inline function _checked_point_index(values::_PointValues, point_index::Integer)
  return _checked_index(point_index, point_count(values), "point")
end

@inline function _checked_field_mode(data::_FieldValues, mode_index::Integer)
  return _checked_index(mode_index, data.local_mode_count, "local mode")
end

@inline function _checked_field_component(data::_FieldValues, component::Integer)
  return _checked_index(component, data.component_count, "field component")
end

@inline _point_normal(values::FaceValues, point_index::Int) = values.normal
@inline _point_normal(values::_InterfaceSideValues, point_index::Int) = values.normal
@inline _point_normal(values::SurfaceValues, point_index::Int) = @inbounds values.normals[point_index]

@inline function _shape_gradient(data::_FieldValues{1,T}, mode_index::Int,
                                 point_index::Int) where {T<:AbstractFloat}
  gradient1 = @inbounds data.gradients[1, mode_index, point_index]
  return (gradient1,)
end

@inline function _shape_gradient(data::_FieldValues{2,T}, mode_index::Int,
                                 point_index::Int) where {T<:AbstractFloat}
  gradient1 = @inbounds data.gradients[1, mode_index, point_index]
  gradient2 = @inbounds data.gradients[2, mode_index, point_index]
  return (gradient1, gradient2)
end

@inline function _shape_gradient(data::_FieldValues{3,T}, mode_index::Int,
                                 point_index::Int) where {T<:AbstractFloat}
  gradient1 = @inbounds data.gradients[1, mode_index, point_index]
  gradient2 = @inbounds data.gradients[2, mode_index, point_index]
  gradient3 = @inbounds data.gradients[3, mode_index, point_index]
  return (gradient1, gradient2, gradient3)
end

@inline function _shape_gradient(data::_FieldValues{D,T}, mode_index::Int,
                                 point_index::Int) where {D,T<:AbstractFloat}
  return ntuple(axis -> (@inbounds data.gradients[axis, mode_index, point_index]), D)
end

@inline function _shape_normal_gradient(data::_FieldValues{1,T}, normal_value::NTuple{1,T},
                                        mode_index::Int, point_index::Int) where {T<:AbstractFloat}
  return @inbounds data.gradients[1, mode_index, point_index] * normal_value[1]
end

@inline function _shape_normal_gradient(data::_FieldValues{2,T}, normal_value::NTuple{2,T},
                                        mode_index::Int, point_index::Int) where {T<:AbstractFloat}
  g1 = @inbounds data.gradients[1, mode_index, point_index]
  g2 = @inbounds data.gradients[2, mode_index, point_index]
  return muladd(g1, normal_value[1], g2 * normal_value[2])
end

@inline function _shape_normal_gradient(data::_FieldValues{3,T}, normal_value::NTuple{3,T},
                                        mode_index::Int, point_index::Int) where {T<:AbstractFloat}
  g1 = @inbounds data.gradients[1, mode_index, point_index]
  g2 = @inbounds data.gradients[2, mode_index, point_index]
  g3 = @inbounds data.gradients[3, mode_index, point_index]
  return muladd(g1, normal_value[1], muladd(g2, normal_value[2], g3 * normal_value[3]))
end

@inline function _shape_normal_gradient(data::_FieldValues{D,T}, normal_value::NTuple{D,T},
                                        mode_index::Int,
                                        point_index::Int) where {D,T<:AbstractFloat}
  return sum((@inbounds data.gradients[axis, mode_index, point_index]) * normal_value[axis]
             for axis in 1:D)
end

"""
    field_dof_range(values, field)

Return the one-based local index range occupied by `field` inside a compiled
integration item.

This is the local analogue of [`field_dof_range`](@ref), but now
for the flattened local vectors and matrices used during assembly on one cell,
face, interface trace, or embedded surface.
"""
function field_dof_range(values::_FieldEvaluationValues, field::AbstractField)
  _field_values(values, field).block
end

"""
    field_dof_count(values, field)

Return the number of local degrees of freedom contributed by `field` on one
compiled integration item.
"""
function field_dof_count(values::_FieldEvaluationValues, field::AbstractField)
  length(_field_values(values, field).block)
end

"""
    local_mode_count(values, field)

Return the number of active scalar local modes of `field` on one compiled
integration item.

For vector fields, this counts scalar modes per component, not the total field
block size.
"""
function local_mode_count(values::_FieldEvaluationValues, field::AbstractField)
  _field_values(values, field).local_mode_count
end

"""
    local_dof_index(values, field, component, mode_index)

Return the local degree-of-freedom index corresponding to one field component
and one local mode inside a compiled integration item.

The returned index addresses the flattened local vector/matrix blocks used
during assembly. Components are stored in component-major order, with one block
of all local modes per field component.
"""
function local_dof_index(values::_FieldEvaluationValues, field::AbstractField, component::Integer,
                         mode_index::Integer)
  data = _field_values(values, field)
  checked_component = _checked_field_component(data, component)
  checked_mode = _checked_field_mode(data, mode_index)
  return first(data.block) + _field_local_dof(data, checked_component, checked_mode) - 1
end

"""
    shape_value(values, field, point_index, mode_index)

Return the value of one local shape function at one quadrature point.

The shape functions are the compiled local basis functions after the
space-compilation step has selected the active local tensor-product modes on the
current cell or trace.
"""
function shape_value(values::_FieldEvaluationValues, field::AbstractField, point_index::Integer,
                     mode_index::Integer)
  data = _field_values(values, field)
  checked_point = _checked_point_index(values, point_index)
  checked_mode = _checked_field_mode(data, mode_index)
  return @inbounds data.values[checked_mode, checked_point]
end

"""
    shape_values(values, field)

Return the matrix of local shape-function values for `field`.

The matrix entry `(i, q)` equals the value of the `i`-th active local mode at
the `q`-th quadrature point.
"""
function shape_values(values::_FieldEvaluationValues, field::AbstractField)
  _field_values(values, field).values
end

"""
    shape_gradient(values, field, point_index, mode_index)

Return the physical-space gradient of one local shape function at one
quadrature point.

Gradients are reported in physical coordinates, not reference coordinates.
"""
@inline function shape_gradient(values::_FieldEvaluationValues, field::AbstractField,
                                point_index::Integer, mode_index::Integer)
  data = _field_values(values, field)
  checked_point = _checked_point_index(values, point_index)
  checked_mode = _checked_field_mode(data, mode_index)
  return _shape_gradient(data, checked_mode, checked_point)
end

"""
    shape_gradients(values, field)

Return the array of physical-space local shape-function gradients for `field`.

The array entry `(a, i, q)` equals the `a`-th component of the gradient of the
`i`-th local mode at quadrature point `q`.
"""
function shape_gradients(values::_FieldEvaluationValues, field::AbstractField)
  _field_values(values, field).gradients
end

const _NormalEvaluationValues = Union{FaceValues,SurfaceValues,_InterfaceSideValues}

"""
    shape_normal_gradient(values, field, point_index, mode_index)

Return the normal derivative of one local shape function at one quadrature
point of a face, interface trace, or embedded surface item.

This is shorthand for the physical-space shape gradient dotted with the local
unit normal. It is especially useful in Nitsche, flux, and interior-penalty
terms where the weak form is written in terms of normal derivatives.
"""
@inline function shape_normal_gradient(values::_NormalEvaluationValues, field::AbstractField,
                                       point_index::Integer, mode_index::Integer)
  data = _field_values(values, field)
  checked_point = _checked_point_index(values, point_index)
  checked_mode = _checked_field_mode(data, mode_index)
  return _shape_normal_gradient(data, _point_normal(values, checked_point), checked_mode,
                                checked_point)
end

"""
    block(buffer, values, field)
    block(buffer, values, test_field, trial_field)
    block(buffer, test_values, test_field, trial_values, trial_field)

Return a view of the subvector or submatrix associated with one field block.

These helpers are intended for local assembly code. They translate from
field-level semantics to the contiguous local block layout used by the compiled
integration item.
"""
function block(buffer::AbstractVector, values::_FieldEvaluationValues, field::AbstractField)
  return view(buffer, field_dof_range(values, field))
end

function block(buffer::AbstractMatrix, values::_FieldEvaluationValues, test_field::AbstractField,
               trial_field::AbstractField)
  return view(buffer, field_dof_range(values, test_field), field_dof_range(values, trial_field))
end

function block(buffer::AbstractMatrix, test_values::_FieldEvaluationValues,
               test_field::AbstractField, trial_values::_FieldEvaluationValues,
               trial_field::AbstractField)
  return view(buffer, field_dof_range(test_values, test_field),
              field_dof_range(trial_values, trial_field))
end

# Reconstruct discrete field values and gradients from a global `State`.

"""
    value(values, state, field, point_index)
    value(values, state, field, component, point_index)

Evaluate the discrete field represented by `state` at one quadrature point of a
compiled integration item.

For a scalar field, the one-component form returns a scalar. For a vector field,
the component-free form returns a tuple of component values. Algebraically, this
computes

  u_h(x_q) = Σᵢ ϕᵢ(x_q) ûᵢ,

where `ϕᵢ` are the local compiled basis functions and `ûᵢ` are the local
amplitudes induced by the global coefficient vector. Those amplitudes already
include whatever space compilation has done: continuity substitutions on CG
axes and direct leaf-local dofs on DG axes.
"""
function value(values::_FieldEvaluationValues, state::State{T}, field::AbstractField,
               point_index::Integer) where {T<:AbstractFloat}
  data = _field_values(values, field)
  data.component_count == 1 && return value(values, state, field, 1, point_index)
  return ntuple(component -> value(values, state, field, component, point_index),
                data.component_count)
end

# Evaluate one scalar component by combining the precomputed shape table with
# the local amplitudes induced by the global state coefficients. The amplitude
# lookup accounts for constrained modes whose local basis function expands into
# several global scalar dofs.
function value(values::_FieldEvaluationValues, state::State{T}, field::AbstractField,
               component::Integer, point_index::Integer) where {T<:AbstractFloat}
  data = _field_values(values, field)
  checked_component = _checked_field_component(data, component)
  checked_point = _checked_point_index(values, point_index)
  state_coefficients = coefficients(state)
  result = zero(T)

  for mode_index in 1:data.local_mode_count
    shape = data.values[mode_index, checked_point]
    shape == zero(T) && continue
    local_dof = _field_local_dof(data, checked_component, mode_index)
    amplitude = _term_amplitude(data.term_offsets, data.term_indices, data.term_coefficients,
                                data.single_term_indices, data.single_term_coefficients,
                                state_coefficients, local_dof)
    result += shape * amplitude
  end

  return result
end

"""
    gradient(values, state, field, point_index)
    gradient(values, state, field, component, point_index)

Evaluate the physical-space gradient of a discrete field at one quadrature
point of a compiled integration item.

For scalar fields, the result is a `D`-tuple. For vector fields, the component-
free form returns one such tuple per field component. Algebraically, this
computes

  ∇u_h(x_q) = Σᵢ ∇ϕᵢ(x_q) ûᵢ.
"""
@inline function gradient(values::_FieldEvaluationValues, state::State{T}, field::AbstractField,
                          point_index::Integer) where {T<:AbstractFloat}
  data = _field_values(values, field)
  checked_point = _checked_point_index(values, point_index)
  state_coefficients = coefficients(state)
  data.component_count == 1 && return _field_gradient(data, state_coefficients, 1, checked_point)
  return ntuple(component -> _field_gradient(data, state_coefficients, component, checked_point),
                data.component_count)
end

@inline function gradient(values::_FieldEvaluationValues, state::State{T}, field::AbstractField,
                          component::Integer, point_index::Integer) where {T<:AbstractFloat}
  data = _field_values(values, field)
  checked_component = _checked_field_component(data, component)
  checked_point = _checked_point_index(values, point_index)
  return _field_gradient(data, coefficients(state), checked_component, checked_point)
end

"""
    normal_gradient(values, state, field, point_index)

Return the physical-space gradient of the discrete trace or boundary field,
projected onto the local unit normal.

For scalar fields, the result is a scalar normal derivative. For vector fields,
the result is one scalar normal derivative per component. This is the operator-
level companion of [`shape_normal_gradient`](@ref): one acts on basis functions,
the other on the reconstructed discrete field itself.
"""
@inline function normal_gradient(values::_NormalEvaluationValues, state::State{T},
                                 field::AbstractField,
                                 point_index::Integer) where {T<:AbstractFloat}
  data = _field_values(values, field)
  checked_point = _checked_point_index(values, point_index)
  normal_value = _point_normal(values, checked_point)
  state_coefficients = coefficients(state)
  data.component_count == 1 &&
    return _field_normal_gradient(data, state_coefficients, 1, checked_point, normal_value)
  return ntuple(component -> _field_normal_gradient(data, state_coefficients, component,
                                                    checked_point, normal_value),
                data.component_count)
end

# Low-level gradient evaluator shared by the scalar and vector-field interfaces.
# The gradients stored in `data` are already mapped to physical coordinates, so
# only the constrained-mode amplitudes still need to be applied here.
@inline function _field_gradient(data::_FieldValues{1,T}, state_coefficients::AbstractVector{T},
                                 component::Int, point_index::Int) where {T<:AbstractFloat}
  term_offsets = data.term_offsets
  term_indices = data.term_indices
  term_coefficients = data.term_coefficients
  single_term_indices = data.single_term_indices
  single_term_coefficients = data.single_term_coefficients
  gradients = data.gradients
  result1 = zero(T)

  @inbounds for mode_index in 1:data.local_mode_count
    local_dof = _field_local_dof(data, component, mode_index)
    amplitude = _term_amplitude(term_offsets, term_indices, term_coefficients, single_term_indices,
                                single_term_coefficients, state_coefficients, local_dof)
    amplitude == zero(T) && continue
    result1 = muladd(gradients[1, mode_index, point_index], amplitude, result1)
  end

  return (result1,)
end

@inline function _field_gradient(data::_FieldValues{2,T}, state_coefficients::AbstractVector{T},
                                 component::Int, point_index::Int) where {T<:AbstractFloat}
  term_offsets = data.term_offsets
  term_indices = data.term_indices
  term_coefficients = data.term_coefficients
  single_term_indices = data.single_term_indices
  single_term_coefficients = data.single_term_coefficients
  gradients = data.gradients
  result1 = zero(T)
  result2 = zero(T)

  @inbounds for mode_index in 1:data.local_mode_count
    local_dof = _field_local_dof(data, component, mode_index)
    amplitude = _term_amplitude(term_offsets, term_indices, term_coefficients, single_term_indices,
                                single_term_coefficients, state_coefficients, local_dof)
    amplitude == zero(T) && continue
    result1 = muladd(gradients[1, mode_index, point_index], amplitude, result1)
    result2 = muladd(gradients[2, mode_index, point_index], amplitude, result2)
  end

  return (result1, result2)
end

@inline function _field_gradient(data::_FieldValues{3,T}, state_coefficients::AbstractVector{T},
                                 component::Int, point_index::Int) where {T<:AbstractFloat}
  term_offsets = data.term_offsets
  term_indices = data.term_indices
  term_coefficients = data.term_coefficients
  single_term_indices = data.single_term_indices
  single_term_coefficients = data.single_term_coefficients
  gradients = data.gradients
  result1 = zero(T)
  result2 = zero(T)
  result3 = zero(T)

  @inbounds for mode_index in 1:data.local_mode_count
    local_dof = _field_local_dof(data, component, mode_index)
    amplitude = _term_amplitude(term_offsets, term_indices, term_coefficients, single_term_indices,
                                single_term_coefficients, state_coefficients, local_dof)
    amplitude == zero(T) && continue
    result1 = muladd(gradients[1, mode_index, point_index], amplitude, result1)
    result2 = muladd(gradients[2, mode_index, point_index], amplitude, result2)
    result3 = muladd(gradients[3, mode_index, point_index], amplitude, result3)
  end

  return (result1, result2, result3)
end

@inline function _field_gradient(data::_FieldValues{D,T}, state_coefficients::AbstractVector{T},
                                 component::Int, point_index::Int) where {D,T<:AbstractFloat}
  term_offsets = data.term_offsets
  term_indices = data.term_indices
  term_coefficients = data.term_coefficients
  single_term_indices = data.single_term_indices
  single_term_coefficients = data.single_term_coefficients
  gradients = data.gradients
  result = ntuple(_ -> zero(T), D)

  @inbounds for mode_index in 1:data.local_mode_count
    local_dof = _field_local_dof(data, component, mode_index)
    amplitude = _term_amplitude(term_offsets, term_indices, term_coefficients, single_term_indices,
                                single_term_coefficients, state_coefficients, local_dof)
    amplitude == zero(T) && continue
    result = ntuple(axis -> muladd(gradients[axis, mode_index, point_index], amplitude,
                                   result[axis]), D)
  end

  return result
end

@inline function _field_normal_gradient(data::_FieldValues{1,T},
                                        state_coefficients::AbstractVector{T}, component::Int,
                                        point_index::Int,
                                        normal_value::NTuple{1,T}) where {T<:AbstractFloat}
  term_offsets = data.term_offsets
  term_indices = data.term_indices
  term_coefficients = data.term_coefficients
  single_term_indices = data.single_term_indices
  single_term_coefficients = data.single_term_coefficients
  gradients = data.gradients
  normal1 = normal_value[1]
  result = zero(T)

  @inbounds for mode_index in 1:data.local_mode_count
    local_dof = _field_local_dof(data, component, mode_index)
    amplitude = _term_amplitude(term_offsets, term_indices, term_coefficients, single_term_indices,
                                single_term_coefficients, state_coefficients, local_dof)
    amplitude == zero(T) && continue
    directional = gradients[1, mode_index, point_index] * normal1
    result = muladd(directional, amplitude, result)
  end

  return result
end

@inline function _field_normal_gradient(data::_FieldValues{2,T},
                                        state_coefficients::AbstractVector{T}, component::Int,
                                        point_index::Int,
                                        normal_value::NTuple{2,T}) where {T<:AbstractFloat}
  term_offsets = data.term_offsets
  term_indices = data.term_indices
  term_coefficients = data.term_coefficients
  single_term_indices = data.single_term_indices
  single_term_coefficients = data.single_term_coefficients
  gradients = data.gradients
  normal1 = normal_value[1]
  normal2 = normal_value[2]
  result = zero(T)

  @inbounds for mode_index in 1:data.local_mode_count
    local_dof = _field_local_dof(data, component, mode_index)
    amplitude = _term_amplitude(term_offsets, term_indices, term_coefficients, single_term_indices,
                                single_term_coefficients, state_coefficients, local_dof)
    amplitude == zero(T) && continue
    directional = muladd(gradients[1, mode_index, point_index], normal1,
                         gradients[2, mode_index, point_index] * normal2)
    result = muladd(directional, amplitude, result)
  end

  return result
end

@inline function _field_normal_gradient(data::_FieldValues{3,T},
                                        state_coefficients::AbstractVector{T}, component::Int,
                                        point_index::Int,
                                        normal_value::NTuple{3,T}) where {T<:AbstractFloat}
  term_offsets = data.term_offsets
  term_indices = data.term_indices
  term_coefficients = data.term_coefficients
  single_term_indices = data.single_term_indices
  single_term_coefficients = data.single_term_coefficients
  gradients = data.gradients
  normal1 = normal_value[1]
  normal2 = normal_value[2]
  normal3 = normal_value[3]
  result = zero(T)

  @inbounds for mode_index in 1:data.local_mode_count
    local_dof = _field_local_dof(data, component, mode_index)
    amplitude = _term_amplitude(term_offsets, term_indices, term_coefficients, single_term_indices,
                                single_term_coefficients, state_coefficients, local_dof)
    amplitude == zero(T) && continue
    directional = muladd(gradients[1, mode_index, point_index], normal1,
                         muladd(gradients[2, mode_index, point_index], normal2,
                                gradients[3, mode_index, point_index] * normal3))
    result = muladd(directional, amplitude, result)
  end

  return result
end

@inline function _field_normal_gradient(data::_FieldValues{D,T},
                                        state_coefficients::AbstractVector{T}, component::Int,
                                        point_index::Int,
                                        normal_value::NTuple{D,T}) where {D,T<:AbstractFloat}
  return normal_component(_field_gradient(data, state_coefficients, component, point_index),
                          normal_value)
end

# Compilation of local integration caches for all entity types.

# Compile all local integration data needed by a problem or assembly plan.
# The main idea is to precompute, for every relevant geometric entity, the
# physical quadrature points, mapped weights, basis tables, and sparse local-to-
# global coefficient maps. Later assembly kernels can then focus on the weak
# form itself rather than on repeated geometric or basis evaluation.
function _compile_integration(layout::FieldLayout{D,T}, cell_quadratures=(), embedded_surfaces=();
                              include_interfaces::Bool=false) where {D,T<:AbstractFloat}
  overrides = _cell_quadrature_overrides(layout, cell_quadratures)
  cells = _compile_cells(layout, overrides)
  boundary_faces = _compile_boundary_faces(layout)
  interfaces = include_interfaces ? _compile_interfaces(layout) : InterfaceValues[]
  surfaces = _compile_embedded_surfaces(layout, embedded_surfaces)
  leaves = _compile_leaf_integration(cells, boundary_faces, surfaces)
  return _CompiledIntegration(cells, boundary_faces, interfaces, surfaces, leaves)
end

function _max_local_dof_count(integration::_CompiledIntegration)
  max_local = 0

  for item in integration.cells
    max_local = max(max_local, item.local_dof_count)
  end

  for item in integration.boundary_faces
    max_local = max(max_local, item.local_dof_count)
  end

  for item in integration.interfaces
    max_local = max(max_local, item.local_dof_count)
  end

  for item in integration.embedded_surfaces
    max_local = max(max_local, item.local_dof_count)
  end

  return max_local
end

@inline function _inverse_jacobian(domain_data::AbstractDomain{D,T},
                                   leaf::Int) where {D,T<:AbstractFloat}
  return ntuple(axis -> inv(jacobian_diagonal_from_biunit_cube(domain_data, leaf, axis)), D)
end

function _compile_item_fields(layout::FieldLayout{D,T}, leaf::Int, points::Vector{NTuple{D,T}},
                              reference_points::Vector{NTuple{D,T}}, inverse_jacobian::NTuple{D,T},
                              local_offset::Int=1) where {D,T<:AbstractFloat}
  next_offset = local_offset
  field_data = ntuple(index -> begin
                        data = _compile_field_values(layout.slots[index], leaf, points,
                                                     reference_points, inverse_jacobian,
                                                     next_offset)
                        next_offset = last(data.block) + 1
                        data
                      end, field_count(layout))
  return field_data, next_offset
end

function _compiled_item_terms(field_data::Tuple)
  term_offsets, term_indices, term_coefficients, single_term_indices, single_term_coefficients, local_dof_count = _merge_local_terms(field_data)
  return (; term_offsets, term_indices, term_coefficients, single_term_indices,
          single_term_coefficients, local_dof_count)
end

function _compile_item_collection(compile_item, specs)
  first_item = compile_item(specs[1])
  items = Vector{typeof(first_item)}(undef, length(specs))
  items[1] = first_item
  length(specs) == 1 && return items

  _run_chunks!(length(specs) - 1) do first_spec, last_spec
    for offset in first_spec:last_spec
      item_index = offset + 1
      items[item_index] = compile_item(specs[item_index])
    end
  end

  return items
end

# Entity-specific compilation: cells, boundary faces, interfaces, and embedded surfaces.

# Compile one `CellValues` item per active leaf, optionally using user-supplied
# cell quadrature overrides.
function _compile_cells(layout::FieldLayout{D,T},
                        overrides::Dict{Int,AbstractQuadrature{D,T}}) where {D,T<:AbstractFloat}
  leaves = layout.slots[1].space.active_leaves
  isempty(leaves) && return CellValues[]
  return _compile_item_collection(leaves) do leaf
    _compile_cell(layout, leaf, get(overrides, leaf, nothing))
  end
end

# Build the cell-local evaluation cache. The quadrature shape is chosen as the
# componentwise maximum over all fields on the leaf so every field can be
# evaluated on the same point set. Basis gradients are mapped from reference to
# physical coordinates using the inverse diagonal Jacobian of the affine cell
# map.
function _compile_cell(layout::FieldLayout{D,T}, leaf::Int, override) where {D,T<:AbstractFloat}
  shape = ntuple(axis -> maximum(cell_quadrature_shape(slot.space, leaf)[axis]
                                 for slot in layout.slots), D)
  quadrature = override === nothing ? TensorQuadrature(T, shape) : override
  domain_data = layout.slots[1].space.domain
  qcount = point_count(quadrature)
  reference_points = Vector{NTuple{D,T}}(undef, qcount)
  points = Vector{NTuple{D,T}}(undef, qcount)
  weights = Vector{T}(undef, qcount)
  jacobian = jacobian_determinant_from_biunit_cube(domain_data, leaf)

  for point_index in 1:qcount
    ξ_raw = point(quadrature, point_index)
    ξ = ntuple(axis -> T(ξ_raw[axis]), D)
    reference_points[point_index] = ξ
    points[point_index] = map_from_biunit_cube(domain_data, leaf, ξ)
    weights[point_index] = T(weight(quadrature, point_index)) * jacobian
  end

  inverse_jacobian = _inverse_jacobian(domain_data, leaf)
  field_data, _ = _compile_item_fields(layout, leaf, points, reference_points, inverse_jacobian)
  terms = _compiled_item_terms(field_data)
  interior_local_dofs = _interior_local_dofs(field_data,
                                             _compiled_leaf(layout.slots[1].space, leaf).local_modes)
  return CellValues(leaf, points, weights, field_data, terms.term_offsets, terms.term_indices,
                    terms.term_coefficients, terms.single_term_indices,
                    terms.single_term_coefficients, interior_local_dofs, terms.local_dof_count)
end

# Boundary faces are compiled only on true geometric boundaries. Periodic seams
# therefore appear in the interface array rather than here.
function _compile_boundary_faces(layout::FieldLayout{D,T}) where {D,T<:AbstractFloat}
  space = layout.slots[1].space
  specs = Tuple{Int,Int,Int}[]

  for leaf in space.active_leaves, axis in 1:D, side in (LOWER, UPPER)
    is_domain_boundary(grid(space), leaf, axis, side) || continue
    push!(specs, (leaf, axis, side))
  end

  isempty(specs) && return FaceValues[]
  return _compile_item_collection(specs) do spec
    _compile_boundary_face(layout, spec...)
  end
end

# Embedded surfaces may contribute several local quadrature items on one leaf,
# so compilation first collects worker-owned buffers and then concatenates them.
function _compile_embedded_surfaces(layout::FieldLayout{D,T},
                                    embedded_surfaces) where {D,T<:AbstractFloat}
  isempty(embedded_surfaces) && return SurfaceValues[]
  worker_count = max(1, min(Threads.nthreads(), length(embedded_surfaces)))
  thread_items = [SurfaceValues[] for _ in 1:worker_count]

  _run_chunks_with_scratch!(thread_items,
                            length(embedded_surfaces)) do items, first_surface, last_surface
    for surface_index in first_surface:last_surface
      append!(items, _compile_embedded_surface_items(layout, embedded_surfaces[surface_index]))
    end
  end

  items = SurfaceValues[]

  for local_items in thread_items
    append!(items, local_items)
  end

  sort!(items; by=item -> item.leaf)
  return items
end

# Compile one physical boundary-face quadrature rule. On a Cartesian affine cell,
# the surface measure factor is constant on the face, so the physical weights are
# simply the reference weights times a constant geometric scale.
function _compile_boundary_face(layout::FieldLayout{D,T}, leaf::Int, face_axis::Int,
                                side::Int) where {D,T<:AbstractFloat}
  domain_data = layout.slots[1].space.domain
  shape = ntuple(axis -> axis == face_axis ? 1 :
                         maximum(cell_quadrature_shape(slot.space, leaf)[axis]
                                 for slot in layout.slots), D)
  reference_points, reference_weights = _face_reference_points(T, shape, face_axis, side)
  qcount = length(reference_points)
  points = Vector{NTuple{D,T}}(undef, qcount)
  weights = Vector{T}(undef, qcount)
  scale = D == 1 ? one(T) : ldexp(face_measure(domain_data, leaf, face_axis), -(D - 1))

  for point_index in 1:qcount
    points[point_index] = map_from_biunit_cube(domain_data, leaf, reference_points[point_index])
    weights[point_index] = reference_weights[point_index] * scale
  end

  inverse_jacobian = _inverse_jacobian(domain_data, leaf)
  field_data, _ = _compile_item_fields(layout, leaf, points, reference_points, inverse_jacobian)
  terms = _compiled_item_terms(field_data)
  normal_data = ntuple(axis -> axis == face_axis ? (side == LOWER ? -one(T) : one(T)) : zero(T), D)
  return FaceValues(leaf, face_axis, side, normal_data, points, weights, field_data,
                    terms.term_offsets, terms.term_indices, terms.term_coefficients,
                    terms.single_term_indices, terms.single_term_coefficients,
                    terms.local_dof_count)
end

# Compile one interface item per upper-face neighbor relation. The neighbor
# specification already resolves nonmatching and periodic topology, so this step
# only needs to build the shared quadrature patch and the two one-sided field
# traces.
function _compile_interfaces(layout::FieldLayout{D,T}) where {D,T<:AbstractFloat}
  space = layout.slots[1].space
  specs = _filtered_upper_face_neighbor_specs(grid(space), space.active_leaves, space.leaf_to_index)
  isempty(specs) && return InterfaceValues[]
  return _compile_item_collection(specs) do spec
    _compile_interface(layout, spec...)
  end
end

# Compile a two-sided interface quadrature cache. Minus and plus traces are
# evaluated on separate physical point arrays so that the same code also works
# for periodic interfaces, where the two traces correspond to distinct physical
# locations that share one tangential parametrization.
function _compile_interface(layout::FieldLayout{D,T}, minus_leaf::Int, face_axis::Int,
                            plus_leaf::Int) where {D,T<:AbstractFloat}
  domain_data = layout.slots[1].space.domain
  shape = ntuple(axis -> axis == face_axis ? 1 :
                         maximum(max(cell_quadrature_shape(slot.space, minus_leaf)[axis],
                                     cell_quadrature_shape(slot.space, plus_leaf)[axis])
                                 for slot in layout.slots), D)
  points, plus_points, weights = _interface_face_points(T, domain_data, minus_leaf, face_axis,
                                                        UPPER, plus_leaf, LOWER, shape)
  minus_inverse = _inverse_jacobian(domain_data, minus_leaf)
  plus_inverse = _inverse_jacobian(domain_data, plus_leaf)
  minus_reference_points = [map_to_biunit_cube(domain_data, minus_leaf, point) for point in points]
  plus_reference_points = [map_to_biunit_cube(domain_data, plus_leaf, point)
                           for point in plus_points]
  minus_fields, next_offset = _compile_item_fields(layout, minus_leaf, points,
                                                   minus_reference_points, minus_inverse)
  plus_fields, _ = _compile_item_fields(layout, plus_leaf, plus_points, plus_reference_points,
                                        plus_inverse, next_offset)
  all_fields = (minus_fields..., plus_fields...)
  terms = _compiled_item_terms(all_fields)
  normal_data = ntuple(axis -> axis == face_axis ? one(T) : zero(T), D)
  return InterfaceValues(minus_leaf, plus_leaf, face_axis, normal_data, points, plus_points,
                         weights, minus_fields, plus_fields, terms.term_offsets, terms.term_indices,
                         terms.term_coefficients, terms.single_term_indices,
                         terms.single_term_coefficients, terms.local_dof_count)
end

function _compile_embedded_surface_items(layout::FieldLayout{D,T},
                                         attachment::_SurfaceAttachment) where {D,T<:AbstractFloat}
  return _compile_embedded_surface_items(layout, attachment.surface, attachment.tag)
end

function _compile_embedded_surface_items(layout::FieldLayout{D,T}, surface::SurfaceQuadrature{D},
                                         tag::_SurfaceTag) where {D,T<:AbstractFloat}
  space = layout.slots[1].space
  return [_compile_surface_quadrature(layout, _checked_surface_quadrature(space, surface, D),
                                      tag)]
end

function _compile_embedded_surface_items(layout::FieldLayout{D,T}, surface::SurfaceQuadrature{SD},
                                         tag::_SurfaceTag) where {D,T<:AbstractFloat,SD}
  throw(ArgumentError("surface quadrature dimension $SD does not match the problem dimension $D"))
end

function _compile_embedded_surface_items(layout::FieldLayout{D,T}, surface::EmbeddedSurface,
                                         tag::_SurfaceTag) where {D,T<:AbstractFloat}
  quadratures = surface_quadratures(surface, layout.slots[1].space)
  isempty(quadratures) && return SurfaceValues[]
  return [_compile_surface_quadrature(layout, quadrature, tag) for quadrature in quadratures]
end

# Compile one embedded-surface quadrature item. Here both the integration
# weights and the normals require geometric transformation from the reference
# surface data supplied by the embedded-surface machinery.
function _compile_surface_quadrature(layout::FieldLayout{D,T}, surface::SurfaceQuadrature{D},
                                     tag::_SurfaceTag=nothing) where {D,T<:AbstractFloat}
  domain_data = layout.slots[1].space.domain
  quadrature = surface.quadrature
  qcount = point_count(quadrature)
  reference_points = Vector{NTuple{D,T}}(undef, qcount)
  points = Vector{NTuple{D,T}}(undef, qcount)
  weights = Vector{T}(undef, qcount)
  normals = Vector{NTuple{D,T}}(undef, qcount)

  for point_index in 1:qcount
    ξ = ntuple(axis -> T(point(quadrature, point_index)[axis]), D)
    normal_data = ntuple(axis -> T(surface.normals[point_index][axis]), D)
    reference_points[point_index] = ξ
    points[point_index] = map_from_biunit_cube(domain_data, surface.leaf, ξ)
    weights[point_index] = T(weight(quadrature, point_index)) *
                           _embedded_surface_weight_scale(domain_data, surface.leaf, normal_data)
    normals[point_index] = _physical_surface_normal(domain_data, surface.leaf, normal_data)
  end

  inverse_jacobian = _inverse_jacobian(domain_data, surface.leaf)
  field_data, _ = _compile_item_fields(layout, surface.leaf, points, reference_points,
                                       inverse_jacobian)
  terms = _compiled_item_terms(field_data)
  return SurfaceValues(surface.leaf, tag, points, weights, normals, field_data, terms.term_offsets,
                       terms.term_indices, terms.term_coefficients, terms.single_term_indices,
                       terms.single_term_coefficients, terms.local_dof_count)
end

# Build leaf-local index ranges into the global face and surface arrays. The
# arrays are sorted by leaf, so one linear pass suffices.
function _compile_leaf_integration(cells, boundary_faces, embedded_surfaces)
  leaves = Vector{_LeafIntegration}(undef, length(cells))
  first_boundary = 1
  first_surface = 1

  for leaf_index in eachindex(cells)
    leaf = cells[leaf_index].leaf
    last_boundary = first_boundary - 1

    while last_boundary < length(boundary_faces) && boundary_faces[last_boundary+1].leaf == leaf
      last_boundary += 1
    end

    last_surface = first_surface - 1

    while last_surface < length(embedded_surfaces) && embedded_surfaces[last_surface+1].leaf == leaf
      last_surface += 1
    end

    leaves[leaf_index] = _LeafIntegration(first_boundary:last_boundary, first_surface:last_surface)
    first_boundary = last_boundary + 1
    first_surface = last_surface + 1
  end

  return leaves
end

# Validation and geometric transforms for user-supplied quadratures.

# Parse and validate per-cell quadrature overrides attached to a problem. The
# reference-point checks below enforce that custom rules are still defined on
# the standard biunit reference cell `[-1, 1]^D`. Plain background domains have
# no automatic overrides; `PhysicalDomain`s opt in through `_default_cell_quadrature`.
function _automatic_cell_quadrature_overrides(layout::FieldLayout{D,T}) where {D,T<:AbstractFloat}
  overrides = Dict{Int,AbstractQuadrature{D,T}}()
  space = layout.slots[1].space

  for leaf in active_leaves(space)
    quadrature_shape = ntuple(axis -> maximum(cell_quadrature_shape(slot.space, leaf)[axis]
                                              for slot in layout.slots), D)
    quadrature = _default_cell_quadrature(domain(space), leaf, quadrature_shape)
    quadrature === nothing || (overrides[leaf] = quadrature)
  end

  return overrides
end

function _cell_quadrature_overrides(layout::FieldLayout{D,T},
                                    cell_quadratures) where {D,T<:AbstractFloat}
  overrides = _automatic_cell_quadrature_overrides(layout)
  space = layout.slots[1].space
  grid_data = grid(space)
  explicit = Set{Int}()

  for attachment in cell_quadratures
    if attachment isa _CellQuadratureAttachment
      leaf = attachment.leaf
      quadrature = attachment.quadrature
    elseif attachment isa Pair &&
           attachment.first isa Int &&
           attachment.second isa AbstractQuadrature
      leaf = attachment.first
      quadrature = attachment.second
    else
      throw(ArgumentError("cell quadrature attachments must be added with add_cell_quadrature!(problem, leaf, quadrature)"))
    end

    checked_leaf = _checked_cell(grid_data, leaf)
    _checked_active_leaf_index(grid_data, space.leaf_to_index, checked_leaf, "cell quadrature")
    checked_leaf in explicit &&
      throw(ArgumentError("duplicate cell quadrature attachment for leaf $checked_leaf"))
    dimension(quadrature) == D ||
      throw(ArgumentError("cell quadrature dimension must match the problem dimension"))
    _check_reference_quadrature(quadrature)
    push!(explicit, checked_leaf)
    overrides[checked_leaf] = quadrature
  end

  return overrides
end

function _checked_surface_quadrature(space::HpSpace{D}, surface,
                                     dimension_count::Int) where {D}
  surface isa SurfaceQuadrature ||
    throw(ArgumentError("surface quadrature attachments must be SurfaceQuadrature instances"))
  dimension(surface.quadrature) == dimension_count ||
    throw(ArgumentError("surface quadrature dimension must match the problem dimension"))
  _checked_active_leaf_index(grid(space), space.leaf_to_index, surface.leaf, "surface quadrature")
  _check_reference_quadrature(surface.quadrature)
  return surface
end

# Ensure that user-supplied reference quadratures live on the standard biunit
# reference cell. This lets the rest of the integration code assume a single
# coordinate convention.
function _check_reference_quadrature(quadrature::AbstractQuadrature{D}) where {D}
  for point_index in 1:point_count(quadrature)
    ξ = point(quadrature, point_index)

    for axis in 1:D
      -1 <= ξ[axis] <= 1 ||
        throw(ArgumentError("quadrature points must lie in the biunit reference cell"))
    end
  end

  return quadrature
end

# For embedded surfaces, the reference quadrature carries normals in reference
# coordinates. Mapping a codimension-1 measure requires the usual Piola-type
# scaling: `dΓ = det(J) / ‖J n̂‖ dΓ̂` for diagonal affine `J`.
function _embedded_surface_weight_scale(domain_data::AbstractDomain{D,T}, leaf::Int,
                                        normal_data::NTuple{D,T}) where {D,T<:AbstractFloat}
  det_jacobian = jacobian_determinant_from_biunit_cube(domain_data, leaf)
  mapped_normal = ntuple(axis -> jacobian_diagonal_from_biunit_cube(domain_data, leaf, axis) *
                                 normal_data[axis], D)
  return det_jacobian / sqrt(sum(mapped_normal[axis]^2 for axis in 1:D))
end

# Map a reference-space surface normal to the corresponding physical unit normal.
function _physical_surface_normal(domain_data::AbstractDomain{D,T}, leaf::Int,
                                  normal_data::NTuple{D,T}) where {D,T<:AbstractFloat}
  transformed = ntuple(axis -> normal_data[axis] /
                               jacobian_diagonal_from_biunit_cube(domain_data, leaf, axis), D)
  magnitude = sqrt(sum(transformed[axis]^2 for axis in 1:D))
  return ntuple(axis -> transformed[axis] / magnitude, D)
end

# Field-local basis tables and sparse local-to-global term maps.

# Convenience entry point when the field uses the default tensor quadrature on
# the current entity.
function _compile_field_values(slot::_FieldSlot{D,T}, leaf::Int,
                               physical_points::Vector{NTuple{D,T}},
                               quadrature::TensorQuadrature{D,T}, inverse_jacobian::NTuple{D,T},
                               local_offset::Int) where {D,T<:AbstractFloat}
  reference_points = [point(quadrature, point_index) for point_index in 1:point_count(quadrature)]
  return _compile_field_values(slot, leaf, physical_points, reference_points, inverse_jacobian,
                               local_offset)
end

# Compile one field block on one local integration item. The basis tables are
# purely local to the field and leaf, but the term arrays already encode how
# each local mode expands into the global coefficient vector after continuity
# constraints have been applied in the `HpSpace`.
function _compile_field_values(slot::_FieldSlot{D,T}, leaf::Int,
                               physical_points::Vector{NTuple{D,T}},
                               reference_points::Vector{NTuple{D,T}}, inverse_jacobian::NTuple{D,T},
                               local_offset::Int) where {D,T<:AbstractFloat}
  compiled_leaf = _compiled_leaf(slot.space, leaf)
  mode_count = length(compiled_leaf.local_modes)
  qcount = length(reference_points)
  values = Matrix{T}(undef, mode_count, qcount)
  gradients = Array{T}(undef, D, mode_count, qcount)
  _fill_basis_tables!(values, gradients, compiled_leaf.local_modes, compiled_leaf.degrees,
                      reference_points, inverse_jacobian)

  local_dof_count = mode_count * component_count(slot.field)
  scalar_term_count = length(compiled_leaf.term_indices)
  total_term_count = scalar_term_count * component_count(slot.field)
  term_offsets = Vector{Int}(undef, local_dof_count + 1)
  term_indices = Vector{Int}(undef, total_term_count)
  term_coefficients = Vector{T}(undef, total_term_count)
  term_offsets[1] = 1
  next_term = 1

  for component in 1:component_count(slot.field)
    component_offset = slot.offset + (component - 1) * slot.scalar_dof_count

    for mode_index in 1:mode_count
      for scalar_term_index in _mode_term_range(compiled_leaf, mode_index)
        term_indices[next_term] = component_offset + compiled_leaf.term_indices[scalar_term_index] -
                                  1
        term_coefficients[next_term] = compiled_leaf.term_coefficients[scalar_term_index]
        next_term += 1
      end

      local_dof = (component - 1) * mode_count + mode_index
      term_offsets[local_dof+1] = next_term
    end
  end

  block = local_offset:(local_offset+local_dof_count-1)
  single_term_indices, single_term_coefficients = _single_term_metadata(term_offsets, term_indices,
                                                                        term_coefficients)
  return _FieldValues{D,T}(_field_id(slot.field), component_count(slot.field),
                           slot.scalar_dof_count, mode_count, block, values, gradients,
                           term_offsets, term_indices, term_coefficients, single_term_indices,
                           single_term_coefficients)
end

# Merge the field-local sparse term maps into one contiguous local numbering for
# the whole cell/face/interface item. This is the layout seen by local residual
# and tangent kernels.
function _merge_local_terms(field_data::Tuple)
  local_dof_count = sum(length(data.block) for data in field_data)
  total_term_count = sum(length(data.term_indices) for data in field_data)
  term_offsets = Vector{Int}(undef, local_dof_count + 1)
  T = eltype(first(field_data).values)
  term_indices = Vector{Int}(undef, total_term_count)
  term_coefficients = Vector{T}(undef, total_term_count)
  term_offsets[1] = 1
  next_term = 1
  next_dof = 1

  for data in field_data
    field_local_dofs = length(data.block)

    for local_dof in 1:field_local_dofs
      first = data.term_offsets[local_dof]
      last = data.term_offsets[local_dof+1] - 1

      for term_index in first:last
        term_indices[next_term] = data.term_indices[term_index]
        term_coefficients[next_term] = data.term_coefficients[term_index]
        next_term += 1
      end

      term_offsets[next_dof+1] = next_term
      next_dof += 1
    end
  end

  single_term_indices, single_term_coefficients = _single_term_metadata(term_offsets, term_indices,
                                                                        term_coefficients)
  return term_offsets, term_indices, term_coefficients, single_term_indices,
         single_term_coefficients, local_dof_count
end

# Identify those local dofs whose modes are strictly interior on the cell, i.e.
# every one-dimensional factor uses an interior integrated-Legendre mode
# numbered `> 1`. These are useful in algorithms that distinguish bubble-like
# interior contributions from trace-coupled ones.
function _interior_local_dofs(field_data::Tuple, local_modes)
  interior_modes = falses(length(local_modes))

  for mode_index in eachindex(local_modes)
    interior_modes[mode_index] = all(local_modes[mode_index][axis] > 1
                                     for axis in eachindex(local_modes[mode_index]))
  end

  dofs = Int[]

  for data in field_data
    for component in 1:data.component_count
      offset = first(data.block) + (component - 1) * data.local_mode_count - 1

      for mode_index in 1:data.local_mode_count
        interior_modes[mode_index] && push!(dofs, offset + mode_index)
      end
    end
  end

  return dofs
end

# Tensor-product basis evaluation and face/interface geometry helpers.

# Fill the tensor-product basis tables used by local integration. The algorithm
# first tabulates the one-dimensional integrated Legendre factors and their
# derivatives on each axis, then forms multi-dimensional values and gradients by
# tensor products. Because the geometry maps are axis-aligned affine maps, the
# physical gradient transformation is just multiplication by the inverse diagonal
# Jacobian entries.
function _fill_basis_tables!(values::Matrix{T}, gradients::Array{T,3}, local_modes,
                             degrees::NTuple{D,Int}, reference_points::Vector{NTuple{D,T}},
                             inverse_jacobian::NTuple{D,T}) where {D,T<:AbstractFloat}
  qcount = length(reference_points)
  axis_values = ntuple(axis -> Matrix{T}(undef, degrees[axis] + 1, qcount), D)
  axis_derivatives = ntuple(axis -> Matrix{T}(undef, degrees[axis] + 1, qcount), D)

  for axis in 1:D
    buffer_values = Vector{T}(undef, degrees[axis] + 1)
    buffer_derivatives = Vector{T}(undef, degrees[axis] + 1)

    for point_index in 1:qcount
      _fe_basis_values_and_derivatives!(reference_points[point_index][axis], degrees[axis],
                                        buffer_values, buffer_derivatives)

      for local_index in eachindex(buffer_values)
        axis_values[axis][local_index, point_index] = buffer_values[local_index]
        axis_derivatives[axis][local_index, point_index] = buffer_derivatives[local_index]
      end
    end
  end

  for mode_index in eachindex(local_modes)
    mode = local_modes[mode_index]

    for point_index in 1:qcount
      value = one(T)

      for axis in 1:D
        value *= axis_values[axis][mode[axis]+1, point_index]
      end

      values[mode_index, point_index] = value

      for axis in 1:D
        derivative = inverse_jacobian[axis] * axis_derivatives[axis][mode[axis]+1, point_index]

        for other_axis in 1:D
          other_axis == axis && continue
          derivative *= axis_values[other_axis][mode[other_axis]+1, point_index]
        end

        gradients[axis, mode_index, point_index] = derivative
      end
    end
  end

  return nothing
end

# Construct a common quadrature patch for a two-sided interface. The tangential
# coordinates are integrated over the geometric overlap of the two face patches,
# while the fixed normal coordinate is placed on the minus and plus faces
# separately. This is what allows the same interface machinery to handle both
# ordinary interior faces and periodic face pairings.
function _interface_face_points(::Type{T}, domain_data::AbstractDomain{D,T}, minus_leaf::Int,
                                face_axis::Int, minus_side::Int, plus_leaf::Int, plus_side::Int,
                                shape::NTuple{D,Int}) where {D,T<:AbstractFloat}
  minus_coordinate = minus_side == LOWER ? cell_lower(domain_data, minus_leaf, face_axis) :
                     cell_upper(domain_data, minus_leaf, face_axis)
  plus_coordinate = plus_side == LOWER ? cell_lower(domain_data, plus_leaf, face_axis) :
                    cell_upper(domain_data, plus_leaf, face_axis)

  if D == 1
    return [(minus_coordinate,)], [(plus_coordinate,)], T[one(T)]
  end

  minus_lower = cell_lower(domain_data, minus_leaf)
  minus_upper = cell_upper(domain_data, minus_leaf)
  plus_lower = cell_lower(domain_data, plus_leaf)
  plus_upper = cell_upper(domain_data, plus_leaf)
  tangential_lower = ntuple(index -> begin
                              axis = _face_tangential_axis(index, face_axis)
                              max(minus_lower[axis], plus_lower[axis])
                            end, D - 1)
  tangential_half = ntuple(index -> begin
                             axis = _face_tangential_axis(index, face_axis)
                             lower = tangential_lower[index]
                             upper = min(minus_upper[axis], plus_upper[axis])
                             upper > lower ||
                               throw(ArgumentError("leafs $minus_leaf and $plus_leaf do not share an interface face patch"))
                             (upper - lower) / 2
                           end, D - 1)
  tangential_points, weights = _face_tangential_quadrature(T, shape, face_axis,
                                                           prod(tangential_half))
  points = Vector{NTuple{D,T}}(undef, length(tangential_points))
  plus_points = Vector{NTuple{D,T}}(undef, length(tangential_points))

  for point_index in eachindex(tangential_points)
    tangential_coordinates = _mapped_face_tangential_coordinates(tangential_lower, tangential_half,
                                                                 tangential_points[point_index], T)
    points[point_index] = _face_point(minus_coordinate, tangential_coordinates, face_axis)
    plus_points[point_index] = _face_point(plus_coordinate, tangential_coordinates, face_axis)
  end

  return points, plus_points, weights
end

# Build the reference quadrature points on one face of the biunit cell
# `[-1, 1]^D` by fixing one coordinate to `±1` and letting the remaining
# tangential coordinates run over a tensor-product quadrature rule.
function _face_reference_points(::Type{T}, shape::NTuple{D,Int}, face_axis::Int,
                                side::Int) where {D,T<:AbstractFloat}
  fixed_coordinate = side == LOWER ? -one(T) : one(T)

  if D == 1
    return [(fixed_coordinate,)], T[one(T)]
  end

  tangential_points, weights = _face_tangential_quadrature(T, shape, face_axis)
  points = Vector{NTuple{D,T}}(undef, length(tangential_points))

  for point_index in eachindex(tangential_points)
    points[point_index] = _face_point(fixed_coordinate, tangential_points[point_index], face_axis)
  end

  return points, weights
end

# Insert one fixed normal coordinate into a tuple of tangential coordinates to
# obtain a full point on a face.
@inline _face_tangential_axis(index::Int, face_axis::Int) = index < face_axis ? index : index + 1

@inline function _face_tangential_shape(shape::NTuple{D,Int}, face_axis::Int) where {D}
  return ntuple(index -> shape[_face_tangential_axis(index, face_axis)], D - 1)
end

function _face_tangential_quadrature(::Type{T}, shape::NTuple{D,Int}, face_axis::Int,
                                     weight_scale::T=one(T)) where {D,T<:AbstractFloat}
  quadrature = TensorQuadrature(T, _face_tangential_shape(shape, face_axis))
  tangential_points = Vector{NTuple{D-1,T}}(undef, point_count(quadrature))
  weights = Vector{T}(undef, point_count(quadrature))

  for point_index in 1:point_count(quadrature)
    tangential_points[point_index] = point(quadrature, point_index)
    weights[point_index] = T(weight(quadrature, point_index)) * weight_scale
  end

  return tangential_points, weights
end

@inline function _mapped_face_tangential_coordinates(lower::NTuple{N,T}, half::NTuple{N,T},
                                                     reference_point::NTuple{N,T},
                                                     ::Type{T}) where {N,T<:AbstractFloat}
  return ntuple(index -> lower[index] + half[index] * (one(T) + reference_point[index]), N)
end

function _face_point(fixed_coordinate::T, tangential_coordinates::NTuple{N,T},
                     face_axis::Int) where {N,T<:AbstractFloat}
  D = N + 1
  tangential_offset = 0
  return ntuple(axis -> begin
                  if axis == face_axis
                    fixed_coordinate
                  else
                    tangential_offset += 1
                    tangential_coordinates[tangential_offset]
                  end
                end, D)
end

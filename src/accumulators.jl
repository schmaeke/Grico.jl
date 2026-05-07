# This file implements Grico's accumulator operator API. The central idea is to
# ask user code for the coefficients that multiply a test function and its
# physical gradient at one quadrature point. For a scalar test component `vᵢ`,
# a callback returns coefficients `a₀` and `a∇` representing
#
#   ∫Ω (a₀ vᵢ + a∇ · ∇vᵢ) dΩ.
#
# This is deliberately one level lower than a symbolic weak-form language: the
# user writes the operator once in coefficient form, while Grico owns all ways
# of consuming that definition. The same callback can drive matrix-free apply,
# local matrix construction, diagonal extraction, residual evaluation, tangent
# application, and tensor-product interpolation/projection kernels.
#
# The file is organized in the same order as the execution model:
#
# 1. Small wrapper types store user operators together with their fields.
# 2. Channel types define the callback vocabulary.
# 3. Registration functions attach accumulator forms to problems.
# 4. Callback adapters normalize user return values.
# 5. Basis-loop kernels implement the general path for cells, traces, and
#    embedded surfaces.
# 6. Tensor-product kernels implement the fast cell path used by high-order
#    full-tensor spaces.

# The internal form wrappers are intentionally small and concrete. They keep the
# user-supplied operator object separate from the field metadata needed by local
# assembly. This lets the planner store arbitrary callable physics objects while
# hot kernels still know exactly which test, trial, or state field to evaluate.

struct _CellAccumulatorForm{O,TF<:AbstractField,UF<:AbstractField}
  operator::O
  test_field::TF
  trial_field::UF
end

struct _CellRhsAccumulatorForm{O,TF<:AbstractField}
  operator::O
  test_field::TF
end

struct _CellResidualAccumulatorForm{O,TF<:AbstractField,SF<:AbstractField}
  operator::O
  test_field::TF
  state_field::SF
end

struct _BoundaryAccumulatorForm{O,TF<:AbstractField,UF<:AbstractField}
  operator::O
  test_field::TF
  trial_field::UF
end

struct _BoundaryRhsAccumulatorForm{O,TF<:AbstractField}
  operator::O
  test_field::TF
end

struct _BoundaryResidualAccumulatorForm{O,TF<:AbstractField,SF<:AbstractField}
  operator::O
  test_field::TF
  state_field::SF
end

struct _SurfaceAccumulatorForm{O,TF<:AbstractField,UF<:AbstractField}
  operator::O
  test_field::TF
  trial_field::UF
end

struct _SurfaceRhsAccumulatorForm{O,TF<:AbstractField}
  operator::O
  test_field::TF
end

struct _SurfaceResidualAccumulatorForm{O,TF<:AbstractField,SF<:AbstractField}
  operator::O
  test_field::TF
  state_field::SF
end

struct _InterfaceAccumulatorForm{O,TF<:AbstractField,UF<:AbstractField}
  operator::O
  test_field::TF
  trial_field::UF
end

struct _InterfaceRhsAccumulatorForm{O,TF<:AbstractField}
  operator::O
  test_field::TF
end

struct _InterfaceResidualAccumulatorForm{O,TF<:AbstractField,SF<:AbstractField}
  operator::O
  test_field::TF
  state_field::SF
end

const _AccumulatorOneSidedValues{D,T} = Union{CellValues{D,T},FaceValues{D,T},SurfaceValues{D,T},
                                              _InterfaceSideValues{D,T}}

# Callback channels are the public vocabulary of the accumulator API. They are
# deliberately small immutable objects so Julia can inline user callbacks and
# keep the value and gradient data in registers. Vector-valued fields are not
# represented as dense vectors here; affine and tangent callbacks see one active
# trial component at a time, while nonlinear state callbacks receive the full
# current state as statically sized tuples.

"""
    TrialChannels

Pointwise component-local trial data passed to accumulator callbacks.

`value` is the reconstructed scalar component value and `gradient` is its
physical gradient. For vector fields, Grico evaluates affine and tangent
operators one trial component at a time; `component` records the active column
component.
"""
struct TrialChannels{D,T<:AbstractFloat}
  component::Int
  value::T
  gradient::NTuple{D,T}
end

"""
    StateChannels

Pointwise nonlinear state data passed to residual and tangent accumulator
callbacks.

For scalar fields, `value` is a scalar and `gradient` is one physical-gradient
tuple. For vector fields, `value` is the tuple `(u₁, u₂, ...)` and `gradient`
is the tuple `(∇u₁, ∇u₂, ...)`; `value(state, i)` and `gradient(state, i)`
return one component without exposing tuple indexing in user callbacks.
"""
struct StateChannels{D,T<:AbstractFloat,C,V,G}
  value::V
  gradient::G
end

"""
    TestChannels(value, gradient)

Coefficients multiplying the test function value and physical gradient.

Accumulator callbacks return this object. If a callback returns a scalar
instead, Grico treats it as a value coefficient with zero gradient coefficient.
"""
struct TestChannels{D,T<:AbstractFloat}
  value::T
  gradient::NTuple{D,T}
end

"""
    TracePair(minus, plus)

Oriented minus/plus pair used by interface accumulator callbacks.

The orientation matches [`InterfaceValues`](@ref): [`jump`](@ref) is
`plus - minus`, while [`average`](@ref) is the arithmetic mean. Trace pairs are
returned by `value(trace)` and `gradient(trace)` for interface trial and state
channels.
"""
struct TracePair{M,P}
  minus::M
  plus::P
end

"""
    TraceTrialChannels

Two-sided component-local trial data passed to interface accumulator callbacks.

`minus` and `plus` are ordinary [`TrialChannels`](@ref) objects for the two
traces. The `component` field records the active trial column component.
"""
struct TraceTrialChannels{M<:TrialChannels,P<:TrialChannels}
  component::Int
  minus::M
  plus::P
end

"""
    TraceStateChannels

Two-sided nonlinear state data passed to interface residual and tangent
accumulator callbacks.

Each side is a [`StateChannels`](@ref) object evaluated on the corresponding
interface trace. Scalar and vector state conventions are the same as for
one-sided state channels.
"""
struct TraceStateChannels{M<:StateChannels,P<:StateChannels}
  minus::M
  plus::P
end

"""
    TraceTestChannels(minus, plus)

Coefficients multiplying the minus and plus interface test traces.

Each side is a [`TestChannels`](@ref) object. Interface accumulator callbacks
return this type because a single interface contribution generally acts on both
local trace vectors.
"""
struct TraceTestChannels{M<:TestChannels,P<:TestChannels}
  minus::M
  plus::P
end

# The following accessors keep user code independent of the concrete channel
# fields. They also preserve a uniform vocabulary across ordinary and trace
# channels: `value(trace)` returns a `TracePair`, while `value(trace, i)` returns
# the `i`th state component on both sides of an interface.
@inline minus(pair::Union{TracePair,TraceTrialChannels,TraceStateChannels,TraceTestChannels}) = pair.minus
@inline plus(pair::Union{TracePair,TraceTrialChannels,TraceStateChannels,TraceTestChannels}) = pair.plus
@inline jump(pair::TracePair) = jump(pair.minus, pair.plus)
@inline average(pair::TracePair) = average(pair.minus, pair.plus)

"""
    inner(a, b)

Contract two scalar or tuple-valued quantities in accumulator callbacks.

For numbers this is multiplication. For same-length tuples it is the Euclidean
inner product. For [`TracePair`](@ref) inputs the contraction is applied
side-by-side, preserving the oriented trace pair.
"""
@inline inner(a::Number, b::Number) = a * b

@inline function inner(a::NTuple{D,<:Number}, b::NTuple{D,<:Number}) where {D}
  result = a[1] * b[1]

  for axis in 2:D
    result = muladd(a[axis], b[axis], result)
  end

  return result
end

@inline function inner(a::TracePair, b::TracePair)
  return TracePair(inner(minus(a), minus(b)), inner(plus(a), plus(b)))
end

@inline value(channels::TrialChannels) = channels.value
@inline gradient(channels::TrialChannels) = channels.gradient

"""
    component(channels)

Return the one-based active field component of a component-local accumulator
channel.

Affine and tangent accumulator callbacks are evaluated one trial component at a
time. For scalar fields this value is always `1`; for vector fields it selects
the column component currently represented by [`TrialChannels`](@ref) or
[`TraceTrialChannels`](@ref). Nonlinear state channels intentionally do not have
an active component because they represent the full current state; use
`value(state, i)` or `gradient(state, i)` to access one state component.
"""
@inline component(channels::TrialChannels) = channels.component
@inline value(channels::StateChannels) = channels.value
@inline gradient(channels::StateChannels) = channels.gradient
@inline function value(channels::StateChannels{D,T,1}, component::Integer) where {D,T}
  _require_index(component, 1, "state component")
  return channels.value
end
@inline function value(channels::StateChannels{D,T,C}, component::Integer) where {D,T,C}
  checked_component = _require_index(component, C, "state component")
  return @inbounds channels.value[checked_component]
end
@inline function gradient(channels::StateChannels{D,T,1}, component::Integer) where {D,T}
  _require_index(component, 1, "state component")
  return channels.gradient
end
@inline function gradient(channels::StateChannels{D,T,C}, component::Integer) where {D,T,C}
  checked_component = _require_index(component, C, "state component")
  return @inbounds channels.gradient[checked_component]
end
@inline value(channels::TestChannels) = channels.value
@inline gradient(channels::TestChannels) = channels.gradient
@inline component(channels::TraceTrialChannels) = channels.component
@inline value(channels::TraceTrialChannels) = TracePair(value(minus(channels)),
                                                        value(plus(channels)))
@inline gradient(channels::TraceTrialChannels) = TracePair(gradient(minus(channels)),
                                                           gradient(plus(channels)))
@inline value(channels::TraceStateChannels) = TracePair(value(minus(channels)),
                                                        value(plus(channels)))
@inline gradient(channels::TraceStateChannels) = TracePair(gradient(minus(channels)),
                                                           gradient(plus(channels)))
@inline value(channels::TraceStateChannels, component::Integer) = TracePair(value(minus(channels),
                                                                                  component),
                                                                            value(plus(channels),
                                                                                  component))
@inline gradient(channels::TraceStateChannels, component::Integer) = TracePair(gradient(minus(channels),
                                                                                        component),
                                                                               gradient(plus(channels),
                                                                                        component))
@inline function normal_component(pair::TracePair, normal_value)
  return TracePair(normal_component(minus(pair), normal_value),
                   normal_component(plus(pair), normal_value))
end

# User callbacks may return either `TestChannels` or a scalar shorthand for a
# pure value coefficient. These adapters normalize that surface once, directly
# after the callback call, so all downstream kernels can operate on a single
# representation. The conversions also anchor the result type to the local
# quadrature value type `T`, which avoids type drift in hot loops.
@inline function _zero_test_channels(::Val{D}, ::Type{T}) where {D,T<:AbstractFloat}
  return TestChannels(zero(T), ntuple(_ -> zero(T), Val(D)))
end

@inline function _as_test_channels(channels::TestChannels{D,T}, ::Val{D},
                                   ::Type{T}) where {D,T<:AbstractFloat}
  return channels
end

@inline function _as_test_channels(channels::TestChannels{D}, ::Val{D},
                                   ::Type{T}) where {D,T<:AbstractFloat}
  return TestChannels(T(channels.value), ntuple(axis -> T(channels.gradient[axis]), Val(D)))
end

@inline function _as_test_channels(value::Number, ::Val{D}, ::Type{T}) where {D,T<:AbstractFloat}
  return TestChannels(T(value), ntuple(_ -> zero(T), Val(D)))
end

@inline function _as_trace_test_channels(channels::TraceTestChannels, ::Val{D},
                                         ::Type{T}) where {D,T<:AbstractFloat}
  return TraceTestChannels(_as_test_channels(minus(channels), Val(D), T),
                           _as_test_channels(plus(channels), Val(D), T))
end

"""
    add_cell_accumulator!(problem, test_field, trial_or_state_field, operator)
    add_cell_accumulator!(problem, test_field, operator)

Add a cell accumulator operator to an affine or residual problem.

For an [`AffineProblem`](@ref), define
`Grico.cell_accumulate(operator, q, trial, test_component)` and return
[`TestChannels`](@ref). The `trial` argument is a [`TrialChannels`](@ref)
object for one active trial component. The three-argument field form registers a
right-hand-side contribution and calls
`Grico.cell_rhs_accumulate(operator, q, test_component)`.

For a [`ResidualProblem`](@ref), define
`Grico.cell_residual_accumulate(operator, q, state, test_component)` and
`Grico.cell_tangent_accumulate(operator, q, state, increment, test_component)`.
The residual callback receives full [`StateChannels`](@ref); the tangent
callback also receives component-local increment [`TrialChannels`](@ref).

The same operator definition feeds matrix-free application, diagonal
construction, local matrix assembly, residual evaluation, tangent application,
and tensorized cell kernels.
"""
function add_cell_accumulator!(problem::AffineProblem, test_field::AbstractField,
                               trial_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "cell accumulator test")
  _check_problem_field(data.fields, trial_field, "cell accumulator trial")
  add_cell!(problem, _CellAccumulatorForm(operator, test_field, trial_field))
  return problem
end

function add_cell_accumulator!(problem::AffineProblem, test_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "cell accumulator test")
  add_cell!(problem, _CellRhsAccumulatorForm(operator, test_field))
  return problem
end

function add_cell_accumulator!(problem::ResidualProblem, test_field::AbstractField,
                               state_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "cell accumulator test")
  _check_problem_field(data.fields, state_field, "cell accumulator state")
  add_cell!(problem, _CellResidualAccumulatorForm(operator, test_field, state_field))
  return problem
end

"""
    add_boundary_accumulator!(problem, boundary, test_field, trial_or_state_field, operator)
    add_boundary_accumulator!(problem, boundary, test_field, operator)

Add a boundary-face accumulator operator to `problem`.

Affine bilinear operators implement
`Grico.boundary_accumulate(operator, q, trial, test_component)`. Affine
right-hand-side operators implement
`Grico.boundary_rhs_accumulate(operator, q, test_component)`. Residual
operators implement the matching `boundary_residual_accumulate` and
`boundary_tangent_accumulate` callbacks. All one-sided callbacks return
[`TestChannels`](@ref), or a scalar value coefficient when no test-gradient
coefficient is needed.
"""
function add_boundary_accumulator!(problem::AffineProblem, boundary::BoundaryFace,
                                   test_field::AbstractField, trial_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "boundary accumulator test")
  _check_problem_field(data.fields, trial_field, "boundary accumulator trial")
  add_boundary!(problem, boundary, _BoundaryAccumulatorForm(operator, test_field, trial_field))
  return problem
end

function add_boundary_accumulator!(problem::AffineProblem, boundary::BoundaryFace,
                                   test_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "boundary accumulator test")
  add_boundary!(problem, boundary, _BoundaryRhsAccumulatorForm(operator, test_field))
  return problem
end

function add_boundary_accumulator!(problem::ResidualProblem, boundary::BoundaryFace,
                                   test_field::AbstractField, state_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "boundary accumulator test")
  _check_problem_field(data.fields, state_field, "boundary accumulator state")
  add_boundary!(problem, boundary,
                _BoundaryResidualAccumulatorForm(operator, test_field, state_field))
  return problem
end

"""
    add_surface_accumulator!(problem, test_field, trial_or_state_field, operator)
    add_surface_accumulator!(problem, tag, test_field, trial_or_state_field, operator)
    add_surface_accumulator!(problem, test_field, operator)
    add_surface_accumulator!(problem, tag, test_field, operator)

Add an embedded-surface accumulator operator to `problem`.

The untagged forms apply to every embedded-surface quadrature item. The tagged
forms are restricted to attachments with the same symbolic tag. Callback names
mirror the boundary API, using the `surface_*_accumulate` prefix.
"""
function add_surface_accumulator!(problem::AffineProblem, test_field::AbstractField,
                                  trial_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface accumulator test")
  _check_problem_field(data.fields, trial_field, "surface accumulator trial")
  add_surface!(problem, _SurfaceAccumulatorForm(operator, test_field, trial_field))
  return problem
end

function add_surface_accumulator!(problem::AffineProblem, tag::Symbol, test_field::AbstractField,
                                  trial_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface accumulator test")
  _check_problem_field(data.fields, trial_field, "surface accumulator trial")
  add_surface!(problem, tag, _SurfaceAccumulatorForm(operator, test_field, trial_field))
  return problem
end

function add_surface_accumulator!(problem::AffineProblem, test_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface accumulator test")
  add_surface!(problem, _SurfaceRhsAccumulatorForm(operator, test_field))
  return problem
end

function add_surface_accumulator!(problem::AffineProblem, tag::Symbol, test_field::AbstractField,
                                  operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface accumulator test")
  add_surface!(problem, tag, _SurfaceRhsAccumulatorForm(operator, test_field))
  return problem
end

function add_surface_accumulator!(problem::ResidualProblem, test_field::AbstractField,
                                  state_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface accumulator test")
  _check_problem_field(data.fields, state_field, "surface accumulator state")
  add_surface!(problem, _SurfaceResidualAccumulatorForm(operator, test_field, state_field))
  return problem
end

function add_surface_accumulator!(problem::ResidualProblem, tag::Symbol, test_field::AbstractField,
                                  state_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface accumulator test")
  _check_problem_field(data.fields, state_field, "surface accumulator state")
  add_surface!(problem, tag, _SurfaceResidualAccumulatorForm(operator, test_field, state_field))
  return problem
end

"""
    add_interface_accumulator!(problem, test_field, trial_or_state_field, operator)
    add_interface_accumulator!(problem, test_field, operator)

Add an interface accumulator operator to `problem`.

Interface affine callbacks receive [`TraceTrialChannels`](@ref) and return
[`TraceTestChannels`](@ref). Interface residual callbacks receive
[`TraceStateChannels`](@ref), and tangent callbacks additionally receive a
component-local [`TraceTrialChannels`](@ref). The minus/plus orientation matches
[`InterfaceValues`](@ref).
"""
function add_interface_accumulator!(problem::AffineProblem, test_field::AbstractField,
                                    trial_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "interface accumulator test")
  _check_problem_field(data.fields, trial_field, "interface accumulator trial")
  add_interface!(problem, _InterfaceAccumulatorForm(operator, test_field, trial_field))
  return problem
end

function add_interface_accumulator!(problem::AffineProblem, test_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "interface accumulator test")
  add_interface!(problem, _InterfaceRhsAccumulatorForm(operator, test_field))
  return problem
end

function add_interface_accumulator!(problem::ResidualProblem, test_field::AbstractField,
                                    state_field::AbstractField, operator)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "interface accumulator test")
  _check_problem_field(data.fields, state_field, "interface accumulator state")
  add_interface!(problem, _InterfaceResidualAccumulatorForm(operator, test_field, state_field))
  return problem
end

"""
    cell_accumulate(operator, q, trial, test_component)

Return the local test-channel coefficients for one affine cell operator sample.

This qualified callback is implemented by users for operators registered with
[`add_cell_accumulator!`](@ref) on an [`AffineProblem`](@ref). The point object
`q` provides the current quadrature `point`, `weight`, `coordinate`, and
`cell_size`. The [`TrialChannels`](@ref) argument contains one scalar trial
component value and gradient at `q`; `test_component` identifies the row
component currently being accumulated.

The callback must return [`TestChannels`](@ref), representing the coefficients
`a₀` and `a∇` in `∫Ω (a₀ v + a∇ · ∇v) dΩ`, or a scalar value coefficient when
the gradient coefficient is zero.
"""
function cell_accumulate end

"""
    cell_rhs_accumulate(operator, q, test_component)

Return test-channel coefficients for one affine cell right-hand-side sample.

This callback is used by `add_cell_accumulator!(problem, test_field, operator)`.
It has no trial argument because it contributes only to the load vector.
"""
function cell_rhs_accumulate end

"""
    cell_residual_accumulate(operator, q, state, test_component)

Return the local test-channel coefficients for one nonlinear residual sample.

This callback is used by cell accumulator operators registered on a
[`ResidualProblem`](@ref). The [`StateChannels`](@ref) argument contains the
current scalar or vector state at the quadrature point. The return value has the
same meaning as for [`cell_accumulate`](@ref): it is the coefficient data that
Grico projects onto the test value and test gradient.
"""
function cell_residual_accumulate end

"""
    cell_tangent_accumulate(operator, q, state, increment, test_component)

Return the local test-channel coefficients for one nonlinear tangent sample.

The `state` argument is the current nonlinear state, while `increment` is a
component-local [`TrialChannels`](@ref) object representing the tangent
direction δu. A single implementation of this callback feeds matrix-free
tangent application and any local assembled tangent path supported by the
accumulator layer.
"""
function cell_tangent_accumulate end

"""
    boundary_accumulate(operator, q, trial, test_component)
    surface_accumulate(operator, q, trial, test_component)

Return one-sided test-channel coefficients for affine boundary or embedded
surface operator samples.

The callback contract is the same as [`cell_accumulate`](@ref), but `q` also
provides [`normal`](@ref). Normal-gradient terms are represented by returning a
gradient coefficient proportional to the normal vector.
"""
function boundary_accumulate end
function surface_accumulate end

"""
    boundary_rhs_accumulate(operator, q, test_component)
    surface_rhs_accumulate(operator, q, test_component)

Return one-sided test-channel coefficients for affine boundary or embedded
surface right-hand-side samples.
"""
function boundary_rhs_accumulate end
function surface_rhs_accumulate end

"""
    boundary_residual_accumulate(operator, q, state, test_component)
    surface_residual_accumulate(operator, q, state, test_component)

Return one-sided test-channel coefficients for nonlinear boundary or embedded
surface residual samples.
"""
function boundary_residual_accumulate end
function surface_residual_accumulate end

"""
    boundary_tangent_accumulate(operator, q, state, increment, test_component)
    surface_tangent_accumulate(operator, q, state, increment, test_component)

Return one-sided test-channel coefficients for nonlinear boundary or embedded
surface tangent samples.
"""
function boundary_tangent_accumulate end
function surface_tangent_accumulate end

"""
    interface_accumulate(operator, q, trial, test_component)
    interface_rhs_accumulate(operator, q, test_component)

Return minus/plus test-channel coefficients for affine interface samples.

Interface callbacks return [`TraceTestChannels`](@ref). The affine bilinear
callback receives [`TraceTrialChannels`](@ref), while the RHS callback has no
trial argument.
"""
function interface_accumulate end
function interface_rhs_accumulate end

"""
    interface_residual_accumulate(operator, q, state, test_component)
    interface_tangent_accumulate(operator, q, state, increment, test_component)

Return minus/plus test-channel coefficients for nonlinear interface samples.
"""
function interface_residual_accumulate end
function interface_tangent_accumulate end

# Accumulator callbacks need a compact quadrature-point handle, not the full
# local integration object. This wrapper exposes the same geometric quantities
# as the lower-level integration API but fixes the active point index, keeping
# callback signatures short and avoiding allocation of per-point structures.
struct _AccumulatorQuadraturePoint{V}
  values::V
  point_index::Int
end

@inline point(q::_AccumulatorQuadraturePoint) = point(q.values, q.point_index)
@inline weight(q::_AccumulatorQuadraturePoint) = weight(q.values, q.point_index)
@inline coordinate(q::_AccumulatorQuadraturePoint, axis::Integer) = point(q)[axis]
@inline normal(q::_AccumulatorQuadraturePoint) = normal(q.values, q.point_index)
@inline face_axis(q::_AccumulatorQuadraturePoint{<:Union{FaceValues,InterfaceValues}}) = face_axis(q.values)
@inline face_side(q::_AccumulatorQuadraturePoint{<:FaceValues}) = face_side(q.values)

@inline cell_size(q::_AccumulatorQuadraturePoint{<:CellValues{D}}) where {D} = q.values.cell_sizes

@inline function cell_size(q::_AccumulatorQuadraturePoint{<:Union{FaceValues{D},SurfaceValues{D},
                                                                  InterfaceValues{D},
                                                                  _InterfaceSideValues{D}}}) where {D}
  return q.values.cell_sizes
end

@inline cell_size(q::_AccumulatorQuadraturePoint, field::AbstractField) = cell_size(q)

@inline function cell_size(q::_AccumulatorQuadraturePoint, field::AbstractField, axis::Integer)
  return cell_size(q, axis)
end

@inline function cell_size(q::_AccumulatorQuadraturePoint, axis::Integer)
  checked_axis = _require_index(axis, length(q.values.cell_sizes), "cell-size axis")
  return @inbounds q.values.cell_sizes[checked_axis]
end

# A `TestChannels` value is a linear functional on one local test basis
# function. This helper evaluates `a₀ ϕ + a∇ · ∇ϕ` at the current point. It is
# shared by matrix assembly, matrix-free application, residuals, tangents, RHS
# construction, and all trace variants, which keeps the interpretation of
# callback return values in one place.
@inline function _basis_test_contribution(values, field::AbstractField, point_index::Int,
                                          mode_index::Int,
                                          channels::TestChannels{D,T}) where {D,T<:AbstractFloat}
  return shape_value(values, field, point_index, mode_index) * channels.value +
         inner(shape_gradient(values, field, point_index, mode_index), channels.gradient)
end

# Channel constructors below convert compiled integration data into the public
# callback vocabulary. The distinction between local coefficients, basis
# functions, and global state coefficients matters:
#
# - `_trial_channels` reconstructs a trial component from a local input vector.
# - `_basis_trial_channels` represents one basis column for local matrices and
#   diagonals.
# - `_state_channels` reconstructs the nonlinear state from a global `State`.
# - tensor variants read values that have already been interpolated by
#   sum-factorization.
@inline function _trial_channels(values, local_coefficients::AbstractVector{T},
                                 field::AbstractField, component::Int,
                                 point_index::Int) where {T<:AbstractFloat}
  gradient_data = gradient(values, local_coefficients, field, component, point_index)
  return TrialChannels{length(gradient_data),T}(component,
                                                value(values, local_coefficients, field, component,
                                                      point_index), gradient_data)
end

@inline function _basis_trial_channels(values, field::AbstractField, component::Int,
                                       point_index::Int, mode_index::Int)
  gradient_data = shape_gradient(values, field, point_index, mode_index)
  T = typeof(gradient_data[1])
  return TrialChannels{length(gradient_data),T}(component,
                                                shape_value(values, field, point_index, mode_index),
                                                gradient_data)
end

@inline function _zero_trial_channels(component::Int, ::Val{D},
                                      ::Type{T}) where {D,T<:AbstractFloat}
  return TrialChannels{D,T}(component, zero(T), ntuple(_ -> zero(T), Val(D)))
end

@inline function _trace_basis_trial_channels(minus_values::_InterfaceSideValues{D,T},
                                             plus_values::_InterfaceSideValues{D,T},
                                             field::AbstractField, component::Int, point_index::Int,
                                             mode_index::Int,
                                             plus_side::Bool) where {D,T<:AbstractFloat}
  active_values = plus_side ? plus_values : minus_values
  active = _basis_trial_channels(active_values, field, component, point_index, mode_index)
  inactive = _zero_trial_channels(component, Val(D), T)
  # A trace basis function belongs to exactly one side of the interface. The
  # inactive side is set to zero so a single user callback can compute jumps,
  # averages, and one-sided fluxes for both local matrix columns.
  return plus_side ? TraceTrialChannels(component, inactive, active) :
         TraceTrialChannels(component, active, inactive)
end

@inline function _trace_trial_channels(minus_values::_InterfaceSideValues{D,T},
                                       plus_values::_InterfaceSideValues{D,T},
                                       local_coefficients::AbstractVector{T}, field::AbstractField,
                                       component::Int, point_index::Int) where {D,T<:AbstractFloat}
  minus_trial = _trial_channels(minus_values, local_coefficients, field, component, point_index)
  plus_trial = _trial_channels(plus_values, local_coefficients, field, component, point_index)
  return TraceTrialChannels(component, minus_trial, plus_trial)
end

@inline function _tensor_trial_channels(field::AbstractField, point_index::Int, component::Int,
                                        point_values::AbstractVector{T},
                                        point_gradients::AbstractMatrix{T},
                                        ::Val{D}) where {D,T<:AbstractFloat}
  return TrialChannels{D,T}(component, @inbounds(point_values[point_index]),
                            ntuple(axis -> @inbounds(point_gradients[axis, point_index]), Val(D)))
end

@inline function _tensor_trace_trial_channels(field::AbstractField, point_index::Int,
                                              component::Int, minus_point_values::AbstractVector{T},
                                              plus_point_values::AbstractVector{T},
                                              minus_point_gradients::AbstractMatrix{T},
                                              plus_point_gradients::AbstractMatrix{T},
                                              ::Val{D}) where {D,T<:AbstractFloat}
  minus_trial = _tensor_trial_channels(field, point_index, component, minus_point_values,
                                       minus_point_gradients, Val(D))
  plus_trial = _tensor_trial_channels(field, point_index, component, plus_point_values,
                                      plus_point_gradients, Val(D))
  return TraceTrialChannels(component, minus_trial, plus_trial)
end

@inline function _state_channels(values, state::State{T}, field::AbstractField, point_index::Int,
                                 ::Val{D}) where {D,T<:AbstractFloat}
  data = _field_values(values, field)
  _check_integration_state(values, state)
  return _state_channels(data, coefficients(state), point_index, Val(D))
end

@inline function _state_channels(data::_FieldValues{D,T,1}, state_coefficients::AbstractVector{T},
                                 point_index::Int, ::Val{D}) where {D,T<:AbstractFloat}
  state_value = _field_value_component(data, state_coefficients, 1, point_index)
  state_gradient = _field_gradient(data, state_coefficients, 1, point_index)
  return StateChannels{D,T,1,typeof(state_value),typeof(state_gradient)}(state_value,
                                                                         state_gradient)
end

@inline function _state_channels(data::_FieldValues{D,T,C}, state_coefficients::AbstractVector{T},
                                 point_index::Int, ::Val{D}) where {D,T<:AbstractFloat,C}
  # The component count `C` is encoded in `_FieldValues`, not recovered from the
  # field object at runtime. This is essential for vector nonlinear kernels:
  # `ntuple(..., Val(C))` gives the compiler a statically sized state tuple.
  state_value = ntuple(component -> _field_value_component(data, state_coefficients, component,
                                                           point_index), Val(C))
  state_gradient = ntuple(component -> _field_gradient(data, state_coefficients, component,
                                                       point_index), Val(C))
  return StateChannels{D,T,C,typeof(state_value),typeof(state_gradient)}(state_value,
                                                                         state_gradient)
end

@inline function _trace_state_channels(minus_values::_InterfaceSideValues{D,T},
                                       plus_values::_InterfaceSideValues{D,T}, state::State{T},
                                       field::AbstractField,
                                       point_index::Int) where {D,T<:AbstractFloat}
  return TraceStateChannels(_state_channels(minus_values, state, field, point_index, Val(D)),
                            _state_channels(plus_values, state, field, point_index, Val(D)))
end

@inline function _tensor_state_channels(data::_FieldValues{D,T,1}, point_index::Int,
                                        point_values::AbstractMatrix{T},
                                        point_gradients::AbstractMatrix{T},
                                        ::Val{D}) where {D,T<:AbstractFloat}
  state_value = @inbounds point_values[point_index, 1]
  state_gradient = ntuple(axis -> @inbounds(point_gradients[point_index, axis]), Val(D))
  return StateChannels{D,T,1,typeof(state_value),typeof(state_gradient)}(state_value,
                                                                         state_gradient)
end

@inline function _tensor_state_channels(data::_FieldValues{D,T,C}, point_index::Int,
                                        point_values::AbstractMatrix{T},
                                        point_gradients::AbstractMatrix{T},
                                        ::Val{D}) where {D,T<:AbstractFloat,C}
  # Tensor kernels store vector-field state values columnwise by component. The
  # gradient matrix uses contiguous blocks `(component - 1) * D + axis` so each
  # component gradient can still be reconstructed as a static `D`-tuple.
  state_value = ntuple(component -> @inbounds(point_values[point_index, component]), Val(C))
  state_gradient = ntuple(component -> ntuple(axis -> @inbounds(point_gradients[point_index,
                                                                                (component - 1) * D + axis]),
                                              Val(D)), Val(C))
  return StateChannels{D,T,C,typeof(state_value),typeof(state_gradient)}(state_value,
                                                                         state_gradient)
end

@inline function _tensor_trace_state_channels(minus_data::_FieldValues{D,T,C},
                                              plus_data::_FieldValues{D,T,C}, point_index::Int,
                                              minus_point_values::AbstractMatrix{T},
                                              plus_point_values::AbstractMatrix{T},
                                              minus_point_gradients::AbstractMatrix{T},
                                              plus_point_gradients::AbstractMatrix{T},
                                              ::Val{D}) where {D,T<:AbstractFloat,C}
  return TraceStateChannels(_tensor_state_channels(minus_data, point_index, minus_point_values,
                                                   minus_point_gradients, Val(D)),
                            _tensor_state_channels(plus_data, point_index, plus_point_values,
                                                   plus_point_gradients, Val(D)))
end

# The `_..._accumulation` adapters are the only places where Grico calls user
# methods. They immediately normalize scalar shorthand returns to
# `TestChannels` or `TraceTestChannels`, so the local kernels below never branch
# on user return types. Keeping these wrappers small and `@inline` also lets the
# compiler specialize the hot loops on the concrete user operator type.
@inline function _cell_accumulation(operator::_CellAccumulatorForm, q::_AccumulatorQuadraturePoint,
                                    trial::TrialChannels{D,T},
                                    test_component::Int) where {D,T<:AbstractFloat}
  return _as_test_channels(cell_accumulate(operator.operator, q, trial, test_component), Val(D), T)
end

@inline function _rhs_accumulation(operator::_CellRhsAccumulatorForm,
                                   q::_AccumulatorQuadraturePoint, test_component::Int, ::Val{D},
                                   ::Type{T}) where {D,T<:AbstractFloat}
  return _as_test_channels(cell_rhs_accumulate(operator.operator, q, test_component), Val(D), T)
end

@inline function _boundary_accumulation(operator::_BoundaryAccumulatorForm,
                                        q::_AccumulatorQuadraturePoint, trial::TrialChannels{D,T},
                                        test_component::Int) where {D,T<:AbstractFloat}
  return _as_test_channels(boundary_accumulate(operator.operator, q, trial, test_component), Val(D),
                           T)
end

@inline function _surface_accumulation(operator::_SurfaceAccumulatorForm,
                                       q::_AccumulatorQuadraturePoint, trial::TrialChannels{D,T},
                                       test_component::Int) where {D,T<:AbstractFloat}
  return _as_test_channels(surface_accumulate(operator.operator, q, trial, test_component), Val(D),
                           T)
end

@inline function _rhs_accumulation(operator::_BoundaryRhsAccumulatorForm,
                                   q::_AccumulatorQuadraturePoint, test_component::Int, ::Val{D},
                                   ::Type{T}) where {D,T<:AbstractFloat}
  return _as_test_channels(boundary_rhs_accumulate(operator.operator, q, test_component), Val(D), T)
end

@inline function _rhs_accumulation(operator::_SurfaceRhsAccumulatorForm,
                                   q::_AccumulatorQuadraturePoint, test_component::Int, ::Val{D},
                                   ::Type{T}) where {D,T<:AbstractFloat}
  return _as_test_channels(surface_rhs_accumulate(operator.operator, q, test_component), Val(D), T)
end

@inline function _cell_residual_accumulation(operator::_CellResidualAccumulatorForm,
                                             q::_AccumulatorQuadraturePoint,
                                             state::StateChannels{D,T},
                                             test_component::Int) where {D,T<:AbstractFloat}
  return _as_test_channels(cell_residual_accumulate(operator.operator, q, state, test_component),
                           Val(D), T)
end

@inline function _boundary_residual_accumulation(operator::_BoundaryResidualAccumulatorForm,
                                                 q::_AccumulatorQuadraturePoint,
                                                 state::StateChannels{D,T},
                                                 test_component::Int) where {D,T<:AbstractFloat}
  return _as_test_channels(boundary_residual_accumulate(operator.operator, q, state,
                                                        test_component), Val(D), T)
end

@inline function _surface_residual_accumulation(operator::_SurfaceResidualAccumulatorForm,
                                                q::_AccumulatorQuadraturePoint,
                                                state::StateChannels{D,T},
                                                test_component::Int) where {D,T<:AbstractFloat}
  return _as_test_channels(surface_residual_accumulate(operator.operator, q, state, test_component),
                           Val(D), T)
end

@inline function _cell_tangent_accumulation(operator::_CellResidualAccumulatorForm,
                                            q::_AccumulatorQuadraturePoint,
                                            state::StateChannels{D,T},
                                            increment::TrialChannels{D,T},
                                            test_component::Int) where {D,T<:AbstractFloat}
  return _as_test_channels(cell_tangent_accumulate(operator.operator, q, state, increment,
                                                   test_component), Val(D), T)
end

@inline function _boundary_tangent_accumulation(operator::_BoundaryResidualAccumulatorForm,
                                                q::_AccumulatorQuadraturePoint,
                                                state::StateChannels{D,T},
                                                increment::TrialChannels{D,T},
                                                test_component::Int) where {D,T<:AbstractFloat}
  return _as_test_channels(boundary_tangent_accumulate(operator.operator, q, state, increment,
                                                       test_component), Val(D), T)
end

@inline function _surface_tangent_accumulation(operator::_SurfaceResidualAccumulatorForm,
                                               q::_AccumulatorQuadraturePoint,
                                               state::StateChannels{D,T},
                                               increment::TrialChannels{D,T},
                                               test_component::Int) where {D,T<:AbstractFloat}
  return _as_test_channels(surface_tangent_accumulate(operator.operator, q, state, increment,
                                                      test_component), Val(D), T)
end

@inline function _interface_accumulation(operator::_InterfaceAccumulatorForm,
                                         q::_AccumulatorQuadraturePoint, trial::TraceTrialChannels,
                                         test_component::Int, ::Val{D},
                                         ::Type{T}) where {D,T<:AbstractFloat}
  return _as_trace_test_channels(interface_accumulate(operator.operator, q, trial, test_component),
                                 Val(D), T)
end

@inline function _interface_rhs_accumulation(operator::_InterfaceRhsAccumulatorForm,
                                             q::_AccumulatorQuadraturePoint, test_component::Int,
                                             ::Val{D}, ::Type{T}) where {D,T<:AbstractFloat}
  return _as_trace_test_channels(interface_rhs_accumulate(operator.operator, q, test_component),
                                 Val(D), T)
end

@inline function _interface_residual_accumulation(operator::_InterfaceResidualAccumulatorForm,
                                                  q::_AccumulatorQuadraturePoint,
                                                  state::TraceStateChannels, test_component::Int,
                                                  ::Val{D}, ::Type{T}) where {D,T<:AbstractFloat}
  return _as_trace_test_channels(interface_residual_accumulate(operator.operator, q, state,
                                                               test_component), Val(D), T)
end

@inline function _interface_tangent_accumulation(operator::_InterfaceResidualAccumulatorForm,
                                                 q::_AccumulatorQuadraturePoint,
                                                 state::TraceStateChannels,
                                                 increment::TraceTrialChannels, test_component::Int,
                                                 ::Val{D}, ::Type{T}) where {D,T<:AbstractFloat}
  return _as_trace_test_channels(interface_tangent_accumulate(operator.operator, q, state,
                                                              increment, test_component), Val(D), T)
end

function cell_matrix!(local_matrix::AbstractMatrix{T}, operator::_CellAccumulatorForm,
                      values::CellValues{D,T}, scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field

  # The dense local matrix path probes each trial basis column explicitly. It is
  # not the preferred application path for tensor-product cells, but it provides
  # exact local matrices for diagnostics, reduced assembled operators, and
  # coarse or fallback solver components.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for trial_component in 1:component_count(trial_field)
      for trial_mode in 1:local_mode_count(values, trial_field)
        column = local_dof_index(values, trial_field, trial_component, trial_mode)
        trial = _basis_trial_channels(values, trial_field, trial_component, point_index, trial_mode)

        for test_component in 1:component_count(test_field)
          channels = _cell_accumulation(operator, q, trial, test_component)

          for test_mode in 1:local_mode_count(values, test_field)
            row = local_dof_index(values, test_field, test_component, test_mode)
            local_matrix[row, column] += weighted *
                                         _basis_test_contribution(values, test_field, point_index,
                                                                  test_mode, channels)
          end
        end
      end
    end
  end

  return local_matrix
end

function cell_apply!(local_result::AbstractVector{T}, operator::_CellAccumulatorForm,
                     values::CellValues{D,T}, local_coefficients::AbstractVector{T},
                     scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  trial_tensor = tensor_values(values, operator.trial_field)
  test_tensor = tensor_values(values, operator.test_field)

  # Full-tensor cells use interpolate-callback-project kernels. All other cells
  # use the general basis-loop path, which is simpler and works for trunk bases,
  # anisotropic local modes, and any future non-tensor basis family.
  if trial_tensor !== nothing && test_tensor !== nothing
    _accumulate_cell_accumulator_tensor_project!(local_result, operator, values, local_coefficients,
                                                 test_tensor, trial_tensor, scratch)
  else
    _accumulate_cell_accumulator_basis!(local_result, operator, values, local_coefficients)
  end

  return nothing
end

function _accumulate_cell_accumulator_basis!(local_result::AbstractVector{T},
                                             operator::_CellAccumulatorForm,
                                             values::CellValues{D,T},
                                             local_coefficients::AbstractVector{T}) where {D,
                                                                                           T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field

  # The basis-loop apply path evaluates the reconstructed trial field once per
  # quadrature point and component, calls the user operator, and projects the
  # returned test coefficients onto every local test mode.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for trial_component in 1:component_count(trial_field)
      trial = _trial_channels(values, local_coefficients, trial_field, trial_component, point_index)

      for test_component in 1:component_count(test_field)
        channels = _cell_accumulation(operator, q, trial, test_component)

        for test_mode in 1:local_mode_count(values, test_field)
          row = local_dof_index(values, test_field, test_component, test_mode)
          local_result[row] += weighted *
                               _basis_test_contribution(values, test_field, point_index, test_mode,
                                                        channels)
        end
      end
    end
  end

  return local_result
end

function cell_diagonal!(local_diagonal::AbstractVector{T}, operator::_CellAccumulatorForm,
                        values::CellValues{D,T},
                        scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  _field_id(test_field) == _field_id(trial_field) || return nothing

  # Diagonal extraction is a local basis probe restricted to matching row and
  # column components. Cross-field operators have no contribution to the
  # diagonal of this field block and are skipped by the field-id check above.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for component in 1:component_count(test_field)
      for mode_index in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, component, mode_index)
        trial = _basis_trial_channels(values, trial_field, component, point_index, mode_index)
        channels = _cell_accumulation(operator, q, trial, component)
        local_diagonal[row] += weighted *
                               _basis_test_contribution(values, test_field, point_index, mode_index,
                                                        channels)
      end
    end
  end

  return nothing
end

@inline function _one_sided_accumulation(operator::_BoundaryAccumulatorForm, q, trial,
                                         test_component)
  return _boundary_accumulation(operator, q, trial, test_component)
end

@inline function _one_sided_accumulation(operator::_SurfaceAccumulatorForm, q, trial,
                                         test_component)
  return _surface_accumulation(operator, q, trial, test_component)
end

# Boundary faces, embedded surfaces, and interface sides all expose one local
# trace of a field together with a normal vector. Their one-sided accumulator
# kernels are therefore identical once the callback prefix has been selected.
# The shared routines below cover local matrices, matrix-free apply, diagonals,
# RHS vectors, residuals, and tangents for both boundary and embedded-surface
# attachments.
function _accumulate_one_sided_matrix!(local_matrix::AbstractMatrix{T}, operator,
                                       values::_AccumulatorOneSidedValues{D,T}) where {D,
                                                                                       T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field

  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for trial_component in 1:component_count(trial_field)
      for trial_mode in 1:local_mode_count(values, trial_field)
        column = local_dof_index(values, trial_field, trial_component, trial_mode)
        trial = _basis_trial_channels(values, trial_field, trial_component, point_index, trial_mode)

        for test_component in 1:component_count(test_field)
          channels = _one_sided_accumulation(operator, q, trial, test_component)

          for test_mode in 1:local_mode_count(values, test_field)
            row = local_dof_index(values, test_field, test_component, test_mode)
            local_matrix[row, column] += weighted *
                                         _basis_test_contribution(values, test_field, point_index,
                                                                  test_mode, channels)
          end
        end
      end
    end
  end

  return local_matrix
end

function _accumulate_one_sided_apply_basis!(local_result::AbstractVector{T}, operator,
                                            values::_AccumulatorOneSidedValues{D,T},
                                            local_coefficients::AbstractVector{T}) where {D,
                                                                                          T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field

  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for trial_component in 1:component_count(trial_field)
      trial = _trial_channels(values, local_coefficients, trial_field, trial_component, point_index)

      for test_component in 1:component_count(test_field)
        channels = _one_sided_accumulation(operator, q, trial, test_component)

        for test_mode in 1:local_mode_count(values, test_field)
          row = local_dof_index(values, test_field, test_component, test_mode)
          local_result[row] += weighted *
                               _basis_test_contribution(values, test_field, point_index, test_mode,
                                                        channels)
        end
      end
    end
  end

  return local_result
end

function _accumulate_one_sided_diagonal!(local_diagonal::AbstractVector{T}, operator,
                                         values::_AccumulatorOneSidedValues{D,T}) where {D,
                                                                                         T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  _field_id(test_field) == _field_id(trial_field) || return local_diagonal

  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for component in 1:component_count(test_field)
      for mode_index in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, component, mode_index)
        trial = _basis_trial_channels(values, trial_field, component, point_index, mode_index)
        channels = _one_sided_accumulation(operator, q, trial, component)
        local_diagonal[row] += weighted *
                               _basis_test_contribution(values, test_field, point_index, mode_index,
                                                        channels)
      end
    end
  end

  return local_diagonal
end

function face_matrix!(local_matrix::AbstractMatrix{T}, operator::_BoundaryAccumulatorForm,
                      values::FaceValues{D,T}, scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_matrix!(local_matrix, operator, values)
  return local_matrix
end

function surface_matrix!(local_matrix::AbstractMatrix{T}, operator::_SurfaceAccumulatorForm,
                         values::SurfaceValues{D,T},
                         scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_matrix!(local_matrix, operator, values)
  return local_matrix
end

function face_apply!(local_result::AbstractVector{T}, operator::_BoundaryAccumulatorForm,
                     values::FaceValues{D,T}, local_coefficients::AbstractVector{T},
                     scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_apply_basis!(local_result, operator, values, local_coefficients)
  return nothing
end

function surface_apply!(local_result::AbstractVector{T}, operator::_SurfaceAccumulatorForm,
                        values::SurfaceValues{D,T}, local_coefficients::AbstractVector{T},
                        scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_apply_basis!(local_result, operator, values, local_coefficients)
  return nothing
end

function face_diagonal!(local_diagonal::AbstractVector{T}, operator::_BoundaryAccumulatorForm,
                        values::FaceValues{D,T},
                        scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_diagonal!(local_diagonal, operator, values)
  return nothing
end

function surface_diagonal!(local_diagonal::AbstractVector{T}, operator::_SurfaceAccumulatorForm,
                           values::SurfaceValues{D,T},
                           scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_diagonal!(local_diagonal, operator, values)
  return nothing
end

function _accumulate_one_sided_rhs!(local_rhs::AbstractVector{T}, operator,
                                    values::_AccumulatorOneSidedValues{D,T}) where {D,
                                                                                    T<:AbstractFloat}
  test_field = operator.test_field

  # RHS accumulation has no trial channel. The callback still returns
  # `TestChannels` so load terms may act on both `v` and `∇v`, which covers
  # Neumann, Nitsche consistency, and embedded-surface moment terms.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_component in 1:component_count(test_field)
      channels = _rhs_accumulation(operator, q, test_component, Val(D), T)

      for test_mode in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, test_component, test_mode)
        local_rhs[row] += weighted *
                          _basis_test_contribution(values, test_field, point_index, test_mode,
                                                   channels)
      end
    end
  end

  return local_rhs
end

function cell_rhs!(local_rhs::AbstractVector{T}, operator::_CellRhsAccumulatorForm,
                   values::CellValues{D,T}, scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_rhs!(local_rhs, operator, values)
  return nothing
end

function face_rhs!(local_rhs::AbstractVector{T}, operator::_BoundaryRhsAccumulatorForm,
                   values::FaceValues{D,T}, scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_rhs!(local_rhs, operator, values)
  return nothing
end

function surface_rhs!(local_rhs::AbstractVector{T}, operator::_SurfaceRhsAccumulatorForm,
                      values::SurfaceValues{D,T},
                      scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_rhs!(local_rhs, operator, values)
  return nothing
end

function cell_residual!(local_residual::AbstractVector{T}, operator::_CellResidualAccumulatorForm,
                        values::CellValues{D,T}, state::State{T},
                        scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  state_tensor = tensor_values(values, operator.state_field)
  test_tensor = tensor_values(values, operator.test_field)

  # Nonlinear residuals use the same fast tensor projection as affine cell
  # operators when both the state and the test field have compatible tensor
  # structure. The state is interpolated once into point values before the
  # callback loop, avoiding repeated basis sums.
  if state_tensor !== nothing && test_tensor !== nothing
    _accumulate_cell_residual_accumulator_tensor_project!(local_residual, operator, values, state,
                                                          test_tensor, state_tensor, scratch)
  else
    _accumulate_cell_residual_accumulator_basis!(local_residual, operator, values, state)
  end

  return nothing
end

function cell_tangent_apply!(local_result::AbstractVector{T},
                             operator::_CellResidualAccumulatorForm, values::CellValues{D,T},
                             state::State{T}, local_increment::AbstractVector{T},
                             scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  increment_tensor = tensor_values(values, operator.state_field)
  test_tensor = tensor_values(values, operator.test_field)

  # The tangent callback is linear in the increment but may depend nonlinearly
  # on the already interpolated state. Tensor kernels therefore keep the state
  # point data fixed and sweep over increment components one at a time.
  if increment_tensor !== nothing && test_tensor !== nothing
    _accumulate_cell_tangent_accumulator_tensor_project!(local_result, operator, values, state,
                                                         local_increment, test_tensor,
                                                         increment_tensor, scratch)
  else
    _accumulate_cell_tangent_accumulator_basis!(local_result, operator, values, state,
                                                local_increment)
  end

  return nothing
end

function _accumulate_cell_residual_accumulator_basis!(local_residual::AbstractVector{T},
                                                      operator::_CellResidualAccumulatorForm,
                                                      values::CellValues{D,T},
                                                      state::State{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  state_field = operator.state_field

  # The basis residual path reconstructs the full nonlinear state at each point
  # and projects the callback coefficients onto the test basis. It is used for
  # non-tensor cells and as the reference structure for trace and surface terms.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)
    state_channels = _state_channels(values, state, state_field, point_index, Val(D))

    for test_component in 1:component_count(test_field)
      channels = _cell_residual_accumulation(operator, q, state_channels, test_component)

      for test_mode in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, test_component, test_mode)
        local_residual[row] += weighted *
                               _basis_test_contribution(values, test_field, point_index, test_mode,
                                                        channels)
      end
    end
  end

  return local_residual
end

function _accumulate_cell_tangent_accumulator_basis!(local_result::AbstractVector{T},
                                                     operator::_CellResidualAccumulatorForm,
                                                     values::CellValues{D,T}, state::State{T},
                                                     local_increment::AbstractVector{T}) where {D,
                                                                                                T<:AbstractFloat}
  test_field = operator.test_field
  state_field = operator.state_field

  # Tangent application evaluates δu component by component, mirroring the
  # affine trial-channel convention. This keeps vector-valued tangents compact:
  # user callbacks can write cross-component couplings using
  # `component(increment)` without Grico materializing dense vector objects.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)
    state_channels = _state_channels(values, state, state_field, point_index, Val(D))

    for increment_component in 1:component_count(state_field)
      increment = _trial_channels(values, local_increment, state_field, increment_component,
                                  point_index)

      for test_component in 1:component_count(test_field)
        channels = _cell_tangent_accumulation(operator, q, state_channels, increment,
                                              test_component)

        for test_mode in 1:local_mode_count(values, test_field)
          row = local_dof_index(values, test_field, test_component, test_mode)
          local_result[row] += weighted *
                               _basis_test_contribution(values, test_field, point_index, test_mode,
                                                        channels)
        end
      end
    end
  end

  return local_result
end

# Boundary and embedded-surface nonlinear terms share the same one-sided
# residual and tangent loops as their affine counterparts. These dispatch
# shims select the user callback namespace while keeping the accumulation loops
# below agnostic to the geometric attachment kind.
@inline function _one_sided_residual_accumulation(operator::_BoundaryResidualAccumulatorForm, q,
                                                  state, test_component)
  return _boundary_residual_accumulation(operator, q, state, test_component)
end

@inline function _one_sided_residual_accumulation(operator::_SurfaceResidualAccumulatorForm, q,
                                                  state, test_component)
  return _surface_residual_accumulation(operator, q, state, test_component)
end

@inline function _one_sided_tangent_accumulation(operator::_BoundaryResidualAccumulatorForm, q,
                                                 state, increment, test_component)
  return _boundary_tangent_accumulation(operator, q, state, increment, test_component)
end

@inline function _one_sided_tangent_accumulation(operator::_SurfaceResidualAccumulatorForm, q,
                                                 state, increment, test_component)
  return _surface_tangent_accumulation(operator, q, state, increment, test_component)
end

function _accumulate_one_sided_residual_basis!(local_residual::AbstractVector{T}, operator,
                                               values::_AccumulatorOneSidedValues{D,T},
                                               state::State{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  state_field = operator.state_field

  # The one-sided residual path is the trace analogue of the cell residual
  # basis loop. The state is reconstructed on the active boundary or embedded
  # surface trace, and the returned coefficients are projected only onto that
  # local trace vector.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)
    state_channels = _state_channels(values, state, state_field, point_index, Val(D))

    for test_component in 1:component_count(test_field)
      channels = _one_sided_residual_accumulation(operator, q, state_channels, test_component)

      for test_mode in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, test_component, test_mode)
        local_residual[row] += weighted *
                               _basis_test_contribution(values, test_field, point_index, test_mode,
                                                        channels)
      end
    end
  end

  return local_residual
end

function _accumulate_one_sided_tangent_basis!(local_result::AbstractVector{T}, operator,
                                              values::_AccumulatorOneSidedValues{D,T},
                                              state::State{T},
                                              local_increment::AbstractVector{T}) where {D,
                                                                                         T<:AbstractFloat}
  test_field = operator.test_field
  state_field = operator.state_field

  # The tangent path sweeps the active increment component exactly as the cell
  # tangent path does. This convention keeps vector-valued boundary operators
  # explicit and type-stable while still allowing cross-component coupling in
  # the user callback through `component(increment)`.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)
    state_channels = _state_channels(values, state, state_field, point_index, Val(D))

    for increment_component in 1:component_count(state_field)
      increment = _trial_channels(values, local_increment, state_field, increment_component,
                                  point_index)

      for test_component in 1:component_count(test_field)
        channels = _one_sided_tangent_accumulation(operator, q, state_channels, increment,
                                                   test_component)

        for test_mode in 1:local_mode_count(values, test_field)
          row = local_dof_index(values, test_field, test_component, test_mode)
          local_result[row] += weighted *
                               _basis_test_contribution(values, test_field, point_index, test_mode,
                                                        channels)
        end
      end
    end
  end

  return local_result
end

function face_residual!(local_residual::AbstractVector{T},
                        operator::_BoundaryResidualAccumulatorForm, values::FaceValues{D,T},
                        state::State{T}, scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_residual_basis!(local_residual, operator, values, state)
  return nothing
end

function surface_residual!(local_residual::AbstractVector{T},
                           operator::_SurfaceResidualAccumulatorForm, values::SurfaceValues{D,T},
                           state::State{T}, scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_residual_basis!(local_residual, operator, values, state)
  return nothing
end

function face_tangent_apply!(local_result::AbstractVector{T},
                             operator::_BoundaryResidualAccumulatorForm, values::FaceValues{D,T},
                             state::State{T}, local_increment::AbstractVector{T},
                             scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_tangent_basis!(local_result, operator, values, state, local_increment)
  return nothing
end

function surface_tangent_apply!(local_result::AbstractVector{T},
                                operator::_SurfaceResidualAccumulatorForm,
                                values::SurfaceValues{D,T}, state::State{T},
                                local_increment::AbstractVector{T},
                                scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  _accumulate_one_sided_tangent_basis!(local_result, operator, values, state, local_increment)
  return nothing
end

# Interface kernels differ from one-sided kernels in only one structural
# respect: a single quadrature sample contributes to both adjacent trace
# vectors. `TraceTestChannels` stores the two returned linear functionals, and
# this helper chooses the side-local basis values needed to project one of
# them. The minus/plus orientation is fixed by `InterfaceValues` and is never
# recomputed inside the hot loops.
@inline function _trace_test_contribution(minus_values::_InterfaceSideValues{D,T},
                                          plus_values::_InterfaceSideValues{D,T},
                                          field::AbstractField, point_index::Int, mode_index::Int,
                                          channels::TraceTestChannels,
                                          plus_side::Bool) where {D,T<:AbstractFloat}
  side_values = plus_side ? plus_values : minus_values
  side_channels = plus_side ? plus(channels) : minus(channels)
  return _basis_test_contribution(side_values, field, point_index, mode_index, side_channels)
end

function interface_matrix!(local_matrix::AbstractMatrix{T}, operator::_InterfaceAccumulatorForm,
                           values::InterfaceValues{D,T},
                           scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  minus_values = minus(values)
  plus_values = plus(values)

  # The two-sided local matrix has four side blocks: minus-minus, minus-plus,
  # plus-minus, and plus-plus. We probe one basis column on one side at a time,
  # pass a trace trial with the opposite side set to zero, and then project the
  # returned minus and plus test coefficients into both side-local row blocks.
  # This is the most direct assembled interpretation of the same interface
  # callback used by matrix-free application.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for trial_side in (false, true)
      trial_values = trial_side ? plus_values : minus_values

      for trial_component in 1:component_count(trial_field)
        for trial_mode in 1:local_mode_count(trial_values, trial_field)
          column = local_dof_index(trial_values, trial_field, trial_component, trial_mode)
          trial = _trace_basis_trial_channels(minus_values, plus_values, trial_field,
                                              trial_component, point_index, trial_mode, trial_side)

          for test_component in 1:component_count(test_field)
            channels = _interface_accumulation(operator, q, trial, test_component, Val(D), T)

            for test_side in (false, true)
              test_values = test_side ? plus_values : minus_values

              for test_mode in 1:local_mode_count(test_values, test_field)
                row = local_dof_index(test_values, test_field, test_component, test_mode)
                local_matrix[row, column] += weighted *
                                             _trace_test_contribution(minus_values, plus_values,
                                                                      test_field, point_index,
                                                                      test_mode, channels,
                                                                      test_side)
              end
            end
          end
        end
      end
    end
  end

  return local_matrix
end

function interface_apply!(local_result::AbstractVector{T}, operator::_InterfaceAccumulatorForm,
                          values::InterfaceValues{D,T}, local_coefficients::AbstractVector{T},
                          scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  minus_values = minus(values)
  plus_values = plus(values)

  # Matrix-free interface application reconstructs the complete two-sided trial
  # trace from the local coefficient vector. The user callback may form jumps,
  # averages, normal fluxes, or one-sided penalties, and the returned
  # `TraceTestChannels` is accumulated into the two trace vectors without
  # materializing the four local matrix blocks.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for trial_component in 1:component_count(trial_field)
      trial = _trace_trial_channels(minus_values, plus_values, local_coefficients, trial_field,
                                    trial_component, point_index)

      for test_component in 1:component_count(test_field)
        channels = _interface_accumulation(operator, q, trial, test_component, Val(D), T)

        for test_side in (false, true)
          test_values = test_side ? plus_values : minus_values

          for test_mode in 1:local_mode_count(test_values, test_field)
            row = local_dof_index(test_values, test_field, test_component, test_mode)
            local_result[row] += weighted *
                                 _trace_test_contribution(minus_values, plus_values, test_field,
                                                          point_index, test_mode, channels,
                                                          test_side)
          end
        end
      end
    end
  end

  return nothing
end

function interface_diagonal!(local_diagonal::AbstractVector{T}, operator::_InterfaceAccumulatorForm,
                             values::InterfaceValues{D,T},
                             scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  _field_id(test_field) == _field_id(trial_field) || return nothing
  minus_values = minus(values)
  plus_values = plus(values)

  # Diagonal extraction still has to respect side coupling. A row on the minus
  # trace can receive a diagonal contribution from a minus-side basis column,
  # while the same global local dof may also appear through constrained or
  # shared trace numbering. Comparing local row and column indices preserves
  # the assembled diagonal semantics without building the full interface block.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_side in (false, true)
      test_values = test_side ? plus_values : minus_values

      for component in 1:component_count(test_field)
        for mode_index in 1:local_mode_count(test_values, test_field)
          row = local_dof_index(test_values, test_field, component, mode_index)

          for trial_side in (false, true)
            trial_values = trial_side ? plus_values : minus_values

            for trial_mode in 1:local_mode_count(trial_values, trial_field)
              column = local_dof_index(trial_values, trial_field, component, trial_mode)
              row == column || continue
              trial = _trace_basis_trial_channels(minus_values, plus_values, trial_field, component,
                                                  point_index, trial_mode, trial_side)
              channels = _interface_accumulation(operator, q, trial, component, Val(D), T)
              local_diagonal[row] += weighted *
                                     _trace_test_contribution(minus_values, plus_values, test_field,
                                                              point_index, mode_index, channels,
                                                              test_side)
            end
          end
        end
      end
    end
  end

  return nothing
end

function interface_rhs!(local_rhs::AbstractVector{T}, operator::_InterfaceRhsAccumulatorForm,
                        values::InterfaceValues{D,T},
                        scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  minus_values = minus(values)
  plus_values = plus(values)

  # Interface loads have no trial trace, but they still return separate minus
  # and plus test coefficients. This supports prescribed jumps, mortar-like
  # source terms, and DG flux data whose sign depends on the chosen interface
  # orientation.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_component in 1:component_count(test_field)
      channels = _interface_rhs_accumulation(operator, q, test_component, Val(D), T)

      for test_side in (false, true)
        test_values = test_side ? plus_values : minus_values

        for test_mode in 1:local_mode_count(test_values, test_field)
          row = local_dof_index(test_values, test_field, test_component, test_mode)
          local_rhs[row] += weighted *
                            _trace_test_contribution(minus_values, plus_values, test_field,
                                                     point_index, test_mode, channels, test_side)
        end
      end
    end
  end

  return nothing
end

function interface_residual!(local_residual::AbstractVector{T},
                             operator::_InterfaceResidualAccumulatorForm,
                             values::InterfaceValues{D,T}, state::State{T},
                             scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  state_field = operator.state_field
  minus_values = minus(values)
  plus_values = plus(values)

  # Nonlinear interface residuals reconstruct the state on both sides before
  # calling the user operator. The callback therefore has all data needed for
  # nonlinear numerical fluxes, contact penalties, and jump stabilizations,
  # while the kernel continues to see only test-channel coefficients.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)
    state_channels = _trace_state_channels(minus_values, plus_values, state, state_field,
                                           point_index)

    for test_component in 1:component_count(test_field)
      channels = _interface_residual_accumulation(operator, q, state_channels, test_component,
                                                  Val(D), T)

      for test_side in (false, true)
        test_values = test_side ? plus_values : minus_values

        for test_mode in 1:local_mode_count(test_values, test_field)
          row = local_dof_index(test_values, test_field, test_component, test_mode)
          local_residual[row] += weighted *
                                 _trace_test_contribution(minus_values, plus_values, test_field,
                                                          point_index, test_mode, channels,
                                                          test_side)
        end
      end
    end
  end

  return nothing
end

function interface_tangent_apply!(local_result::AbstractVector{T},
                                  operator::_InterfaceResidualAccumulatorForm,
                                  values::InterfaceValues{D,T}, state::State{T},
                                  local_increment::AbstractVector{T},
                                  scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  state_field = operator.state_field
  minus_values = minus(values)
  plus_values = plus(values)

  # Interface tangent application combines the two nonlinear conventions: the
  # current state is a full minus/plus trace, while the increment is evaluated
  # one component at a time on both sides. This lets tangent callbacks express
  # side-coupled Jacobian actions without forcing the accumulator layer to
  # expose dense vector notation or side-block assembly details.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)
    state_channels = _trace_state_channels(minus_values, plus_values, state, state_field,
                                           point_index)

    for increment_component in 1:component_count(state_field)
      increment = _trace_trial_channels(minus_values, plus_values, local_increment, state_field,
                                        increment_component, point_index)

      for test_component in 1:component_count(test_field)
        channels = _interface_tangent_accumulation(operator, q, state_channels, increment,
                                                   test_component, Val(D), T)

        for test_side in (false, true)
          test_values = test_side ? plus_values : minus_values

          for test_mode in 1:local_mode_count(test_values, test_field)
            row = local_dof_index(test_values, test_field, test_component, test_mode)
            local_result[row] += weighted *
                                 _trace_test_contribution(minus_values, plus_values, test_field,
                                                          point_index, test_mode, channels,
                                                          test_side)
          end
        end
      end
    end
  end

  return nothing
end

# The tensor-product helpers below are the fast full-tensor cell path for
# affine application, nonlinear residuals, and nonlinear tangent application.
# They translate between Grico's compact local vectors and the rectangular
# tensor boxes used by the sum-factorization routines in `quadrature.jl`. The
# tensor box may contain entries for modes that are not active in the local
# polynomial space, so all copies explicitly scatter through
# `tensor.local_modes` instead of assuming a dense local-mode ordering.
function _copy_accumulator_tensor_component_coefficients!(result::AbstractVector{T},
                                                          tensor::TensorProductValues{D,T}, values,
                                                          field::AbstractField, component::Int,
                                                          local_coefficients::AbstractVector{T}) where {D,
                                                                                                        T<:AbstractFloat}
  data = _field_values(values, field)
  offset = first(data.block) + _field_component_offset(data, component) - 1
  fill!(result, zero(T))
  box_shape = tensor_mode_shape(tensor)

  @inbounds for mode_index in 1:tensor_mode_count(tensor)
    mode = tensor.local_modes[mode_index]
    box_indices = ntuple(axis -> mode[axis] + 1, Val(D))
    box_index = _tensor_shape_linear_index(box_indices, box_shape)
    result[box_index] = local_coefficients[offset+mode_index]
  end

  return result
end

function _copy_accumulator_tensor_component_state_coefficients!(result::AbstractVector{T},
                                                                tensor::TensorProductValues{D,T},
                                                                values, state::State{T},
                                                                field::AbstractField,
                                                                component::Int) where {D,
                                                                                       T<:AbstractFloat}
  data = _field_values(values, field)
  return _copy_accumulator_tensor_component_state_coefficients!(result, tensor, data, state,
                                                                component)
end

function _copy_accumulator_tensor_component_state_coefficients!(result::AbstractVector{T},
                                                                tensor::TensorProductValues{D,T},
                                                                data::_FieldValues{D,T},
                                                                state::State{T},
                                                                component::Int) where {D,
                                                                                       T<:AbstractFloat}
  state_coefficients = coefficients(state)
  fill!(result, zero(T))
  box_shape = tensor_mode_shape(tensor)

  # State vectors may contain constrained dofs represented by a small linear
  # combination of reduced coefficients. `_term_amplitude` applies that local
  # expansion while the surrounding scatter maps the active mode into the tensor
  # box used by interpolation and gradient kernels.
  @inbounds for mode_index in 1:tensor_mode_count(tensor)
    mode = tensor.local_modes[mode_index]
    box_indices = ntuple(axis -> mode[axis] + 1, Val(D))
    box_index = _tensor_shape_linear_index(box_indices, box_shape)
    local_dof = _field_local_dof(data, component, mode_index)
    result[box_index] = _term_amplitude(data.term_offsets, data.term_indices,
                                        data.term_coefficients, data.single_term_indices,
                                        data.single_term_coefficients, state_coefficients,
                                        local_dof)
  end

  return result
end

function _fill_accumulator_tensor_state!(point_values::AbstractMatrix{T},
                                         point_gradients::AbstractMatrix{T},
                                         coefficients_buffer::AbstractVector{T},
                                         value_buffer::AbstractVector{T},
                                         gradient_buffer::AbstractMatrix{T},
                                         tensor::TensorProductValues{D,T}, values, state::State{T},
                                         field::AbstractField,
                                         scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  data = _field_values(values, field)
  _check_integration_state(values, state)
  return _fill_accumulator_tensor_state!(point_values, point_gradients, coefficients_buffer,
                                         value_buffer, gradient_buffer, tensor, data, state,
                                         scratch)
end

function _fill_accumulator_tensor_state!(point_values::AbstractMatrix{T},
                                         point_gradients::AbstractMatrix{T},
                                         coefficients_buffer::AbstractVector{T},
                                         value_buffer::AbstractVector{T},
                                         gradient_buffer::AbstractMatrix{T},
                                         tensor::TensorProductValues{D,T},
                                         data::_FieldValues{D,T,C}, state::State{T},
                                         scratch::KernelScratch{T}) where {D,T<:AbstractFloat,C}
  # Nonlinear tensor kernels precompute all state components at all quadrature
  # points before the callback loop. This costs one interpolation and one
  # gradient pass per component, but it avoids repeating those tensor products
  # for every test component or tangent increment component.
  for state_component in 1:C
    _copy_accumulator_tensor_component_state_coefficients!(coefficients_buffer, tensor, data, state,
                                                           state_component)
    tensor_box_interpolate!(value_buffer, tensor, coefficients_buffer, scratch)
    tensor_box_gradient!(gradient_buffer, tensor, coefficients_buffer, scratch)

    @inbounds for point_index in 1:tensor_point_count(tensor)
      point_values[point_index, state_component] = value_buffer[point_index]

      for axis in 1:D
        point_gradients[point_index, (state_component - 1) * D + axis] = gradient_buffer[axis,
                                                                                         point_index]
      end
    end
  end

  return nothing
end

function _project_accumulator_tensor_component!(local_result::AbstractVector{T}, values,
                                                field::AbstractField, component::Int,
                                                tensor::TensorProductValues{D,T},
                                                weighted_values_by_component::AbstractMatrix{T},
                                                weighted_gradients_by_component::AbstractMatrix{T},
                                                scratch::KernelScratch{T}) where {D,
                                                                                  T<:AbstractFloat}
  point_total = tensor_point_count(tensor)
  contribution = scratch_vector(scratch, D + 6, tensor_mode_box_count(tensor))
  weighted_values = scratch_vector(scratch, D + 7, point_total)
  weighted_gradients = scratch_matrix(scratch, 14, D, point_total)
  fill!(contribution, zero(T))

  # Projection routines operate on one scalar test component at a time. The
  # accumulator loops store value coefficients in component columns and gradient
  # coefficients in contiguous `D`-column blocks; this copy extracts the active
  # component into the layout expected by `tensor_box_project_gradient!`.
  @inbounds for point_index in 1:point_total
    weighted_values[point_index] = weighted_values_by_component[point_index, component]

    for axis in 1:D
      source_column = (component - 1) * D + axis
      weighted_gradients[axis, point_index] = weighted_gradients_by_component[point_index,
                                                                              source_column]
    end
  end

  tensor_box_project!(contribution, tensor, weighted_values, scratch)
  tensor_box_project_gradient!(contribution, tensor, weighted_gradients, scratch)
  _add_accumulator_tensor_component_output!(local_result, values, field, component, tensor,
                                            contribution)
  return local_result
end

function _add_accumulator_tensor_component_output!(local_result::AbstractVector{T}, values,
                                                   field::AbstractField, component::Int,
                                                   tensor::TensorProductValues{D,T},
                                                   contribution::AbstractVector{T}) where {D,
                                                                                           T<:AbstractFloat}
  data = _field_values(values, field)
  offset = first(data.block) + _field_component_offset(data, component) - 1
  box_shape = tensor_mode_shape(tensor)

  # The tensor projection returns coefficients for the full tensor box. Only
  # active local modes are scattered back into the element vector; inactive box
  # entries correspond to modes omitted by the local basis and are ignored.
  @inbounds for mode_index in 1:tensor_mode_count(tensor)
    mode = tensor.local_modes[mode_index]
    box_indices = ntuple(axis -> mode[axis] + 1, Val(D))
    box_index = _tensor_shape_linear_index(box_indices, box_shape)
    local_result[offset+mode_index] += contribution[box_index]
  end

  return local_result
end

function _accumulate_cell_accumulator_tensor_project!(local_result::AbstractVector{T},
                                                      operator::_CellAccumulatorForm,
                                                      values::CellValues{D,T},
                                                      local_coefficients::AbstractVector{T},
                                                      test_tensor::TensorProductValues{D,T},
                                                      trial_tensor::TensorProductValues{D,T},
                                                      scratch::KernelScratch{T}) where {D,
                                                                                        T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  test_components = component_count(test_field)
  point_total = tensor_point_count(test_tensor)
  trial_coefficients = scratch_vector(scratch, D + 2, tensor_mode_box_count(trial_tensor))
  trial_values = scratch_vector(scratch, D + 3, tensor_point_count(trial_tensor))
  trial_gradients = scratch_matrix(scratch, 1, D, tensor_point_count(trial_tensor))
  weighted_values = scratch_matrix(scratch, 2, point_total, test_components)
  weighted_gradients = scratch_matrix(scratch, 3, point_total, D * test_components)
  fill!(weighted_values, zero(T))
  fill!(weighted_gradients, zero(T))

  # The fast apply algorithm has three phases. First, interpolate the active
  # trial component from coefficient space to quadrature values and gradients.
  # Second, evaluate the accumulator callback at each point and accumulate the
  # weighted test coefficients. Third, project those weighted coefficients back
  # to the test basis with tensor transposes. This avoids the O(p²D) basis-mode
  # loops of explicit local matrix application on high-order tensor cells.
  for trial_component in 1:component_count(trial_field)
    _copy_accumulator_tensor_component_coefficients!(trial_coefficients, trial_tensor, values,
                                                     trial_field, trial_component,
                                                     local_coefficients)
    tensor_box_interpolate!(trial_values, trial_tensor, trial_coefficients, scratch)
    tensor_box_gradient!(trial_gradients, trial_tensor, trial_coefficients, scratch)

    for point_index in 1:point_count(values)
      q = _AccumulatorQuadraturePoint(values, point_index)
      weighted = weight(q)
      trial = _tensor_trial_channels(trial_field, point_index, trial_component, trial_values,
                                     trial_gradients, Val(D))

      for test_component in 1:test_components
        channels = _cell_accumulation(operator, q, trial, test_component)
        weighted_values[point_index, test_component] += weighted * channels.value

        for axis in 1:D
          target_column = (test_component - 1) * D + axis
          weighted_gradients[point_index, target_column] += weighted * channels.gradient[axis]
        end
      end
    end
  end

  for test_component in 1:test_components
    _project_accumulator_tensor_component!(local_result, values, test_field, test_component,
                                           test_tensor, weighted_values, weighted_gradients,
                                           scratch)
  end

  return local_result
end

function _accumulate_cell_residual_accumulator_tensor_project!(local_residual::AbstractVector{T},
                                                               operator::_CellResidualAccumulatorForm,
                                                               values::CellValues{D,T},
                                                               state::State{T},
                                                               test_tensor::TensorProductValues{D,
                                                                                                T},
                                                               state_tensor::TensorProductValues{D,
                                                                                                 T},
                                                               scratch::KernelScratch{T}) where {D,
                                                                                                 T<:AbstractFloat}
  test_field = operator.test_field
  state_field = operator.state_field
  state_data = _field_values(values, state_field)
  _check_integration_state(values, state)
  test_components = component_count(test_field)
  state_components = _field_component_count(state_data)
  point_total = tensor_point_count(test_tensor)
  state_coefficients = scratch_vector(scratch, D + 2, tensor_mode_box_count(state_tensor))
  state_value_buffer = scratch_vector(scratch, D + 3, tensor_point_count(state_tensor))
  state_gradient_buffer = scratch_matrix(scratch, 4, D, tensor_point_count(state_tensor))
  state_values = scratch_matrix(scratch, 5, tensor_point_count(state_tensor), state_components)
  state_gradients = scratch_matrix(scratch, 6, tensor_point_count(state_tensor),
                                   D * state_components)
  weighted_values = scratch_matrix(scratch, 2, point_total, test_components)
  weighted_gradients = scratch_matrix(scratch, 3, point_total, D * test_components)
  fill!(weighted_values, zero(T))
  fill!(weighted_gradients, zero(T))
  _fill_accumulator_tensor_state!(state_values, state_gradients, state_coefficients,
                                  state_value_buffer, state_gradient_buffer, state_tensor,
                                  state_data, state, scratch)

  # Residual tensor projection is the nonlinear analogue of affine fast apply.
  # The precomputed state arrays are read-only in the callback loop, while the
  # returned value and gradient coefficients are accumulated per test component
  # and projected once at the end.
  for point_index in 1:point_count(values)
    q = _AccumulatorQuadraturePoint(values, point_index)
    weighted = weight(q)
    state_channels = _tensor_state_channels(state_data, point_index, state_values, state_gradients,
                                            Val(D))

    for test_component in 1:test_components
      channels = _cell_residual_accumulation(operator, q, state_channels, test_component)
      weighted_values[point_index, test_component] += weighted * channels.value

      for axis in 1:D
        target_column = (test_component - 1) * D + axis
        weighted_gradients[point_index, target_column] += weighted * channels.gradient[axis]
      end
    end
  end

  for test_component in 1:test_components
    _project_accumulator_tensor_component!(local_residual, values, test_field, test_component,
                                           test_tensor, weighted_values, weighted_gradients,
                                           scratch)
  end

  return local_residual
end

function _accumulate_cell_tangent_accumulator_tensor_project!(local_result::AbstractVector{T},
                                                              operator::_CellResidualAccumulatorForm,
                                                              values::CellValues{D,T},
                                                              state::State{T},
                                                              local_increment::AbstractVector{T},
                                                              test_tensor::TensorProductValues{D,T},
                                                              tensor::TensorProductValues{D,T},
                                                              scratch::KernelScratch{T}) where {D,
                                                                                                T<:AbstractFloat}
  test_field = operator.test_field
  state_field = operator.state_field
  state_data = _field_values(values, state_field)
  _check_integration_state(values, state)
  test_components = component_count(test_field)
  state_components = _field_component_count(state_data)
  point_total = tensor_point_count(test_tensor)
  state_coefficients = scratch_vector(scratch, D + 2, tensor_mode_box_count(tensor))
  state_value_buffer = scratch_vector(scratch, D + 3, tensor_point_count(tensor))
  increment_coefficients = scratch_vector(scratch, D + 4, tensor_mode_box_count(tensor))
  increment_values = scratch_vector(scratch, D + 5, tensor_point_count(tensor))
  increment_gradients = scratch_matrix(scratch, 7, D, tensor_point_count(tensor))
  state_gradient_buffer = scratch_matrix(scratch, 4, D, tensor_point_count(tensor))
  state_values = scratch_matrix(scratch, 5, tensor_point_count(tensor), state_components)
  state_gradients = scratch_matrix(scratch, 6, tensor_point_count(tensor), D * state_components)
  weighted_values = scratch_matrix(scratch, 2, point_total, test_components)
  weighted_gradients = scratch_matrix(scratch, 3, point_total, D * test_components)
  fill!(weighted_values, zero(T))
  fill!(weighted_gradients, zero(T))
  _fill_accumulator_tensor_state!(state_values, state_gradients, state_coefficients,
                                  state_value_buffer, state_gradient_buffer, tensor, state_data,
                                  state, scratch)

  # Tangent tensor projection reuses the precomputed nonlinear state and sweeps
  # one increment component through interpolation, callback evaluation, and
  # accumulation. The weighted test coefficients are shared across increment
  # components because all of them contribute to the same Jacobian-vector
  # product result.
  for increment_component in 1:state_components
    _copy_accumulator_tensor_component_coefficients!(increment_coefficients, tensor, values,
                                                     state_field, increment_component,
                                                     local_increment)
    tensor_box_interpolate!(increment_values, tensor, increment_coefficients, scratch)
    tensor_box_gradient!(increment_gradients, tensor, increment_coefficients, scratch)

    for point_index in 1:point_count(values)
      q = _AccumulatorQuadraturePoint(values, point_index)
      weighted = weight(q)
      state_channels = _tensor_state_channels(state_data, point_index, state_values,
                                              state_gradients, Val(D))
      increment = _tensor_trial_channels(state_field, point_index, increment_component,
                                         increment_values, increment_gradients, Val(D))

      for test_component in 1:test_components
        channels = _cell_tangent_accumulation(operator, q, state_channels, increment,
                                              test_component)
        weighted_values[point_index, test_component] += weighted * channels.value

        for axis in 1:D
          target_column = (test_component - 1) * D + axis
          weighted_gradients[point_index, target_column] += weighted * channels.gradient[axis]
        end
      end
    end
  end

  for test_component in 1:test_components
    _project_accumulator_tensor_component!(local_result, values, test_field, test_component,
                                           test_tensor, weighted_values, weighted_gradients,
                                           scratch)
  end

  return local_result
end

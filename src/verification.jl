# This file provides lightweight verification utilities for comparing a computed
# discrete field with reference data in the `L2` norm. The key design choice is
# that verification does not introduce its own sampling or interpolation layer.
# Instead, it reuses the same compiled cell-evaluation data that drive assembly
# and residual evaluation: the same local basis values, the same physical
# quadrature points, and the same field/component layout. As a result, the
# verification path answers a precise question:
#
#   how large is the error of the discrete field represented by this `State`
#   when integrated with these cell quadratures?
#
# This is useful for both quick regression checks and more careful convergence
# studies. The caller may either reuse an existing compiled `AssemblyPlan` or
# request a modified verification quadrature, for example to integrate the error
# more accurately than the quadrature that happened to be sufficient for
# assembly.
#
# The reported numbers are therefore always with respect to a chosen
# verification quadrature. They approximate the mathematical `L²` norm of the
# error, but they do so through the same explicit weighted sums that the rest of
# the package uses for local operator integration.

# Verification quadrature selection and cell-cache reuse.

@inline _checked_verification_extra(extra_points::Integer) = _checked_nonnegative(extra_points,
                                                                                  "extra_points")

@inline function _verification_axis_quadrature_count(layout::FieldLayout, leaf::Int, axis::Int,
                                                     extra::Int)
  base = maximum(cell_quadrature_shape(slot.space, leaf)[axis] for slot in layout.slots)
  extra <= typemax(Int) - base ||
    throw(ArgumentError("extra_points produces a non-Int-representable verification quadrature count on axis $axis"))
  return base + extra
end

@inline function _verification_quadrature_shape(layout::FieldLayout{D}, leaf::Int,
                                                extra::Int) where {D}
  return ntuple(axis -> _verification_axis_quadrature_count(layout, leaf, axis, extra), D)
end

function _check_verification_quadrature(quadrature::AbstractQuadrature{D,T}) where {D,
                                                                                    T<:AbstractFloat}
  _check_reference_quadrature(quadrature)

  for point_index in 1:point_count(quadrature)
    T(weight(quadrature, point_index)) >= zero(T) ||
      throw(ArgumentError("verification quadrature weights must be non-negative"))
  end

  return quadrature
end

function _verification_cell_overrides(layout::FieldLayout{D,T}, extra::Int,
                                      cell_quadratures) where {D,T<:AbstractFloat}
  space = layout.slots[1].space
  grid_data = grid(space)
  domain_data = domain(space)
  overrides = Dict{Int,AbstractQuadrature{D}}()

  if extra == 0
    for leaf in active_leaves(space)
      shape = _verification_quadrature_shape(layout, leaf, extra)
      quadrature = _physical_cell_quadrature(domain_data, leaf, shape)
      quadrature === nothing || (overrides[leaf] = quadrature)
    end
  else
    for leaf in active_leaves(space)
      shape = _verification_quadrature_shape(layout, leaf, extra)
      quadrature = _physical_cell_quadrature(domain_data, leaf, shape)
      overrides[leaf] = quadrature === nothing ? TensorQuadrature(T, shape) : quadrature
    end
  end

  explicit_overrides = Set{Int}()

  for attachment in cell_quadratures
    attachment isa Pair ||
      throw(ArgumentError("cell quadratures must be provided as leaf => quadrature pairs"))
    attachment.first isa Integer || throw(ArgumentError("cell quadrature leaf must be an integer"))
    attachment.second isa AbstractQuadrature ||
      throw(ArgumentError("cell quadrature value must be an AbstractQuadrature"))
    checked_leaf = _checked_cell(grid_data, attachment.first)
    _checked_active_leaf_index(grid_data, snapshot(space).leaf_to_index, checked_leaf,
                               "verification cell quadrature")
    !(checked_leaf in explicit_overrides) ||
      throw(ArgumentError("duplicate verification cell quadrature attachment for leaf $checked_leaf"))
    dimension(attachment.second) == D ||
      throw(ArgumentError("verification cell quadrature dimension must match the space dimension"))
    _check_verification_quadrature(attachment.second)
    push!(explicit_overrides, checked_leaf)
    overrides[checked_leaf] = attachment.second
  end

  return overrides
end

# Build the cell-evaluation data used for verification. By default this uses
# the layout's physical cell quadrature. This is intentionally distinct from
# assembly on a [`PhysicalDomain`](@ref) with a stabilized cell-measure policy:
# error norms are physical diagnostics by default and should not include
# fictitious-domain volume unless the caller supplies explicit
# `cell_quadratures`. Users may request a uniformly enriched quadrature rule
# via `extra_points` or override individual leaves explicitly.
function _rebuild_verification_cells(layout::FieldLayout{D,T}; extra_points::Integer=0,
                                     cell_quadratures=()) where {D,T<:AbstractFloat}
  extra = _checked_verification_extra(extra_points)
  overrides = _verification_cell_overrides(layout, extra, cell_quadratures)
  return _compile_cells(layout, overrides)
end

@inline _uses_physical_cell_measure(::Domain) = true
@inline _uses_physical_cell_measure(domain::PhysicalDomain) = domain.cell_measure isa
                                                              PhysicalMeasure

# Decide which compiled cell data to use for the verification integral. If the
# caller passes an assembly/verification plan and does not request modified
# quadrature, we can reuse the already compiled cells directly. Otherwise we
# rebuild cell data on demand with the requested quadrature enrichment or
# overrides.
function _verification_cells(state::State, field::AbstractField, plan; extra_points::Integer=0,
                             cell_quadratures=())
  field_dof_range(state.layout, field)
  extra = _checked_verification_extra(extra_points)

  if plan !== nothing
    _check_state(plan, state)
    field_dof_range(plan.layout, field)
  end

  if plan !== nothing &&
     extra == 0 &&
     isempty(cell_quadratures) &&
     _uses_physical_cell_measure(domain(state.layout.slots[1].space))
    return plan.integration.cells
  end

  return _rebuild_verification_cells(state.layout; extra_points=extra,
                                     cell_quadratures=cell_quadratures)
end

# Reference-data normalization and pointwise component evaluation.

# Reference data may be supplied either as a callable `x ↦ u(x)` or as a
# constant scalar/vector value. This helper normalizes both forms to one scalar
# component of type `T`, matching the component structure of the field under
# verification.
@inline _verification_reference_value(data, x) = applicable(data, x) ? data(x) : data

function _checked_verification_reference_data(value, component_total::Int)
  if component_total == 1
    if value isa Tuple || value isa AbstractVector
      length(value) == 1 ||
        throw(ArgumentError("scalar reference data must be scalar or contain exactly one value"))
      return value
    end

    value isa Real && !(value isa Bool) ||
      throw(ArgumentError("scalar reference data must be a finite real value or contain exactly one value"))
    return value
  end

  if value isa Tuple || value isa AbstractVector
    length(value) == component_total ||
      throw(ArgumentError("vector-valued reference data must match the field component count"))
    return value
  end

  throw(ArgumentError("vector-valued reference data must return a tuple or vector"))
end

function _checked_verification_scalar(value, ::Type{T},
                                      name::AbstractString) where {T<:AbstractFloat}
  value isa Real && !(value isa Bool) || throw(ArgumentError("$name must be a finite real value"))
  checked = T(value)
  isfinite(checked) || throw(ArgumentError("$name must be finite"))
  return checked
end

function _verification_component_value(value, component::Int, component_total::Int,
                                       ::Type{T}) where {T<:AbstractFloat}
  raw = value isa Tuple || value isa AbstractVector ? value[component] : value
  name = component_total == 1 ? "scalar reference data" : "vector-valued reference data"
  return _checked_verification_scalar(raw, T, name)
end

function _checked_l2_component(value::T, name::AbstractString) where {T<:AbstractFloat}
  isfinite(value) || throw(ArgumentError("$name must be finite"))
  value >= zero(T) || throw(ArgumentError("$name must be non-negative"))
  return value
end

# Accumulation of absolute and relative `L2` error components.

# Accumulate the numerator and denominator of the relative `L2` error,
#
#   ∫Ω ‖u_h - u_exact‖² dΩ,    ∫Ω ‖u_exact‖² dΩ,
#
# over all active cells. Scalar and vector fields are both reduced to sums over
# components, so the norm is the standard Euclidean field norm integrated over
# the domain.
function _l2_error_components(cells, state::State{T}, field::AbstractField,
                              exact) where {T<:AbstractFloat}
  isempty(cells) && return zero(T), zero(T)
  state_coefficients = coefficients(state)
  component_total = component_count(field)
  numerator = zero(T)
  denominator = zero(T)

  for cell in cells
    data = _field_values(cell, field)

    @inbounds for point_index in 1:point_count(cell)
      x = cell.points[point_index]
      weighted = cell.weights[point_index]
      reference_data = _checked_verification_reference_data(_verification_reference_value(exact, x),
                                                            component_total)

      for component in 1:component_total
        approximate = _field_value_component(data, state_coefficients, component, point_index)
        isfinite(approximate) || throw(ArgumentError("state values must be finite"))
        reference = _verification_component_value(reference_data, component, component_total, T)
        difference = approximate - reference
        numerator += difference * difference * weighted
        denominator += reference * reference * weighted
      end
    end
  end

  return numerator, denominator
end

function _verification_l2_components(state::State{T}, field::AbstractField, exact, plan;
                                     extra_points::Integer=0,
                                     cell_quadratures=()) where {T<:AbstractFloat}
  cells = _verification_cells(state, field, plan; extra_points, cell_quadratures=cell_quadratures)
  numerator, denominator = _l2_error_components(cells, state, field, exact)
  return _checked_l2_component(numerator, "L2 error numerator"),
         _checked_l2_component(denominator, "L2 reference denominator")
end

# Public `L2` verification queries.

"""
    l2_error(state, field, exact; plan=nothing, extra_points=0, cell_quadratures=())

Compute the absolute `L2` error of `field` in `state` against `exact`.

The reference data `exact` may be either

- a callable `x -> u(x)` evaluated at physical quadrature points, or
- a constant scalar/vector value interpreted as a spatially uniform reference.

For vector fields, the returned norm is

`(∫Ω ∑ᵢ (uₕ,ᵢ - u_exact,ᵢ)² dΩ)¹ᐟ²`.

If `plan` is supplied, its compiled cell integration data are reused whenever
`extra_points == 0` and no explicit `cell_quadratures` are given. Otherwise the
verification cells are rebuilt with the requested quadrature settings.

This makes it possible to separate the discrete solution from the quadrature
used to assess it: a nonlinear problem may have been assembled with one rule,
while verification deliberately uses a denser rule to reduce integration error
in the reported norm.

Accordingly, the returned value should be read as "the `L²` error measured by
this verification quadrature," not as a symbolic exact norm independent of
quadrature.
"""
function l2_error(state::State{T}, field::AbstractField, exact; plan=nothing,
                  extra_points::Integer=0, cell_quadratures=()) where {T<:AbstractFloat}
  numerator, _ = _verification_l2_components(state, field, exact, plan; extra_points=extra_points,
                                             cell_quadratures=cell_quadratures)
  return sqrt(numerator)
end

"""
    relative_l2_error(state, field, exact; plan=nothing, extra_points=0, cell_quadratures=())

Compute the relative `L2` error of `field` in `state` against `exact`.

The reported value is

`‖uₕ - u_exact‖ₗ₂ / ‖u_exact‖ₗ₂`.

If the reference norm vanishes exactly, the function returns `0` when the
numerator also vanishes and `Inf` otherwise.

As in [`l2_error`](@ref), the verification integral may either reuse an
available compiled plan or be rebuilt with enriched or explicitly overridden
quadrature rules.

The same quadrature dependence applies here: the ratio is formed from the
numerically integrated numerator and denominator chosen by the verification
settings.
"""
function relative_l2_error(state::State{T}, field::AbstractField, exact; plan=nothing,
                           extra_points::Integer=0, cell_quadratures=()) where {T<:AbstractFloat}
  numerator, denominator = _verification_l2_components(state, field, exact, plan;
                                                       extra_points=extra_points,
                                                       cell_quadratures=cell_quadratures)
  denominator > zero(T) && return sqrt(numerator / denominator)
  return numerator == zero(T) ? zero(T) : T(Inf)
end

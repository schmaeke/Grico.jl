# This file contains the automatic, problem-independent indicator machinery used
# by Grico's compact hp-adaptivity planner. The indicators are deliberately
# derived from the discrete field representation itself rather than from a PDE
# residual. They answer local resolution questions:
#
# - How much energy remains in the highest modal layer of each cell?
# - Do traces agree across DG interfaces?
# - Would collapsing two children into their parent create a large local L²
#   projection defect?
#
# These quantities are not a replacement for application-specific estimators,
# but they provide a stable default policy for examples, diagnostics, and smooth
# manufactured problems. The implementation keeps the three roles separate:
# modal indicators drive p-refinement and p-coarsening, DG jump indicators drive
# h-refinement on discontinuous axes, and projection defects decide whether an
# immediate h-coarsening candidate may be accepted.
#
# The code is arranged in the same order as the planner:
# 1. modal layer energies and decay ratios,
# 2. interface jump indicators for DG axes,
# 3. local L² projection defects for h-coarsening candidates,
# 4. the single-tolerance multiresolution planner that converts indicators into
#    one `AdaptivityPlan`.

# Local mode amplitudes are not always raw global coefficients. On continuous
# spaces and constrained layouts, a local mode may expand into a short sparse
# combination of global dofs. `_term_amplitude` applies that expansion so all
# indicator energies are measured in the actual represented field.
function _local_mode_amplitude(compiled::_CompiledLeaf, coefficients::AbstractVector,
                               mode_index::Int)
  return _term_amplitude(compiled.term_offsets, compiled.term_indices, compiled.term_coefficients,
                         compiled.single_term_indices, compiled.single_term_coefficients,
                         coefficients, mode_index)
end

function _local_mode_energy(component_coefficients, compiled::_CompiledLeaf{D,T},
                            mode_index::Int) where {D,T<:AbstractFloat}
  # Vector-valued fields use the Euclidean component energy of one modal tuple.
  # This keeps scalar and vector indicators comparable without introducing a
  # problem-specific norm at the automatic-planner level.
  energy = zero(T)

  for component in eachindex(component_coefficients)
    amplitude = _local_mode_amplitude(compiled, component_coefficients[component], mode_index)
    energy += amplitude * amplitude
  end

  isfinite(energy) || throw(ArgumentError("modal indicator energies must be finite"))
  return energy
end

@inline function _mode_with_axis_value(mode::NTuple{D,<:Integer}, axis::Int, value::Int) where {D}
  return ntuple(current_axis -> current_axis == axis ? value : Int(mode[current_axis]), D)
end

# Degree-zero DG cells only contain a constant mode and therefore have no modal
# decay information. Degree-one cells only contain the two endpoint modes `ψ₀`
# and `ψ₁` on each axis. Those do not separate constant and linear content by
# themselves, so for `p = 1` we first transform the endpoint pair into its
# constant/linear combination before extracting layer energies.
function _axis_layer_energies(component_coefficients, compiled::_CompiledLeaf{D,T}, axis::Int,
                              mode_energy::AbstractVector{T}) where {D,T<:AbstractFloat}
  degree_value = compiled.degrees[axis]
  top_energy = zero(T)
  previous_energy = zero(T)

  degree_value == 0 && return top_energy, previous_energy

  if degree_value == 1
    half = T(0.5)

    for lower_index in eachindex(compiled.local_modes)
      lower_mode = compiled.local_modes[lower_index]
      lower_mode[axis] == 0 || continue
      upper_index = _mode_lookup(compiled, _mode_with_axis_value(lower_mode, axis, 1))
      upper_index != 0 || continue

      for component in eachindex(component_coefficients)
        lower = _local_mode_amplitude(compiled, component_coefficients[component], lower_index)
        upper = _local_mode_amplitude(compiled, component_coefficients[component], upper_index)
        constant = half * (lower + upper)
        linear = half * (upper - lower)
        constant_energy = constant * constant
        linear_energy = linear * linear
        isfinite(constant_energy) && isfinite(linear_energy) ||
          throw(ArgumentError("modal indicator energies must be finite"))
        previous_energy += constant_energy
        top_energy += linear_energy
      end
    end

    return top_energy, previous_energy
  end

  for mode_index in eachindex(compiled.local_modes)
    mode = compiled.local_modes[mode_index]
    layer = mode[axis]
    layer == degree_value || layer == degree_value - 1 || continue
    amplitude_squared = @inbounds mode_energy[mode_index]

    if layer == degree_value
      top_energy += amplitude_squared
    else
      previous_energy += amplitude_squared
    end
  end

  return top_energy, previous_energy
end

# Interface jumps only need field traces on the two cells adjacent to a face.
# For the finite-element basis used by `HpSpace`, degree-zero DG cells have one
# true constant mode, while degree `>= 1` traces are carried only by endpoint
# modes `0` and `1`; higher integrated-Legendre modes vanish at both endpoints.
# The direct evaluator below exploits this basis invariant instead of compiling
# full interface field tables.
@inline function _trace_mode_axis_value(degree::Int, side::Int)
  degree == 0 && return 0
  return side == LOWER ? 0 : 1
end

function _trace_component_value(compiled::_CompiledLeaf{D,T}, coefficients::AbstractVector{T},
                                axis::Int, side::Int,
                                basis_values::NTuple{D,Vector{T}}) where {D,T<:AbstractFloat}
  boundary_value = _trace_mode_axis_value(compiled.degrees[axis], side)
  result = zero(T)

  for mode_index in eachindex(compiled.local_modes)
    mode = compiled.local_modes[mode_index]
    mode[axis] == boundary_value || continue
    shape = one(T)

    for current_axis in 1:D
      current_axis == axis && continue
      shape *= basis_values[current_axis][mode[current_axis]+1]
    end

    shape == zero(T) && continue
    amplitude = _term_amplitude(compiled.term_offsets, compiled.term_indices,
                                compiled.term_coefficients, compiled.single_term_indices,
                                compiled.single_term_coefficients, coefficients, mode_index)
    result += shape * amplitude
  end

  return result
end

# Interface-jump evaluation keeps one indicator table plus reusable basis
# buffers for both traces of the current face patch.
struct _InterfaceJumpScratch{D,T<:AbstractFloat}
  indicators::Matrix{T}
  minus_basis::NTuple{D,Vector{T}}
  plus_basis::NTuple{D,Vector{T}}
end

function _InterfaceJumpScratch(::Type{T}, ::Val{D},
                               active_leaf_total::Int) where {D,T<:AbstractFloat}
  return _InterfaceJumpScratch{D,T}(zeros(T, active_leaf_total, D), ntuple(_ -> T[], D),
                                    ntuple(_ -> T[], D))
end

@inline function _interface_jump_shape(minus_compiled::_CompiledLeaf{D},
                                       plus_compiled::_CompiledLeaf{D}, axis::Int) where {D}
  return ntuple(current_axis -> current_axis == axis ? 1 :
                                max(minus_compiled.quadrature_shape[current_axis],
                                    plus_compiled.quadrature_shape[current_axis]), D)
end

function _interface_jump_quadrature!(cache::Dict{Tuple{Int,NTuple{D1,Int}},TensorQuadrature{D1,T}},
                                     ::Type{T}, minus_compiled::_CompiledLeaf{D},
                                     plus_compiled::_CompiledLeaf{D},
                                     axis::Int) where {D,D1,T<:AbstractFloat}
  shape = _interface_jump_shape(minus_compiled, plus_compiled, axis)
  key = (axis, _face_tangential_shape(shape, axis))
  return get!(cache, key) do
    TensorQuadrature(T, key[2])
  end
end

@inline _face_tangential_index(axis::Int, face_axis::Int) = axis < face_axis ? axis : axis - 1

@inline function _reference_coordinate(lower::T, upper::T, coordinate::T) where {T<:AbstractFloat}
  return (T(2) * coordinate - lower - upper) / (upper - lower)
end

function _reference_face_point(lower::NTuple{D,T}, upper::NTuple{D,T}, face_axis::Int, side::Int,
                               tangential_coordinates::NTuple{D1,T}) where {D,D1,T<:AbstractFloat}
  fixed_coordinate = side == LOWER ? -one(T) : one(T)
  return ntuple(axis -> axis == face_axis ? fixed_coordinate :
                        _reference_coordinate(lower[axis], upper[axis],
                                              tangential_coordinates[_face_tangential_index(axis,
                                                                                            face_axis)]),
                D)
end

function _interface_jump_energy(component_coefficients, minus_compiled::_CompiledLeaf{D,T},
                                plus_compiled::_CompiledLeaf{D,T}, domain_data::AbstractDomain{D,T},
                                minus_leaf::Int, plus_leaf::Int, axis::Int,
                                quadrature::TensorQuadrature{D1,T},
                                scratch::_InterfaceJumpScratch{D,T}) where {D,D1,T<:AbstractFloat}
  # A face in an adaptive Cartesian grid may be only a patch of either adjacent
  # leaf face. The quadrature is therefore placed on the physical overlap in the
  # tangential coordinates, then mapped back to each leaf's reference face before
  # evaluating the two traces.
  minus_lower = cell_lower(domain_data, minus_leaf)
  minus_upper = cell_upper(domain_data, minus_leaf)
  plus_lower = cell_lower(domain_data, plus_leaf)
  plus_upper = cell_upper(domain_data, plus_leaf)
  tangential_lower = ntuple(index -> begin
                              current_axis = _face_tangential_axis(index, axis)
                              max(minus_lower[current_axis], plus_lower[current_axis])
                            end, D - 1)
  tangential_half = ntuple(index -> begin
                             current_axis = _face_tangential_axis(index, axis)
                             lower = tangential_lower[index]
                             upper = min(minus_upper[current_axis], plus_upper[current_axis])
                             upper > lower ||
                               throw(ArgumentError("leaves $minus_leaf and $plus_leaf do not share an interface face patch"))
                             (upper - lower) / 2
                           end, D - 1)
  weight_scale = one(T)

  for half_width in tangential_half
    weight_scale *= half_width
  end

  jump_energy = zero(T)
  component_total = length(component_coefficients)

  @inbounds for point_index in 1:point_count(quadrature)
    tangential_coordinates = _mapped_face_tangential_coordinates(tangential_lower, tangential_half,
                                                                 point(quadrature, point_index), T)
    minus_reference_point = _reference_face_point(minus_lower, minus_upper, axis, UPPER,
                                                  tangential_coordinates)
    plus_reference_point = _reference_face_point(plus_lower, plus_upper, axis, LOWER,
                                                 tangential_coordinates)
    _fill_leaf_basis!(scratch.minus_basis, minus_compiled.degrees, minus_reference_point)
    _fill_leaf_basis!(scratch.plus_basis, plus_compiled.degrees, plus_reference_point)
    point_jump = zero(T)

    for component in 1:component_total
      minus_value = _trace_component_value(minus_compiled, component_coefficients[component], axis,
                                           UPPER, scratch.minus_basis)
      plus_value = _trace_component_value(plus_compiled, component_coefficients[component], axis,
                                          LOWER, scratch.plus_basis)
      # The jump orientation is plus minus minus, matching the interface
      # convention used by operator callbacks. The squared norm itself is
      # orientation-independent, but keeping the sign convention local avoids
      # later surprises if this evaluator is reused for signed diagnostics.
      difference = plus_value - minus_value
      point_jump += difference * difference
    end

    jump_energy += point_jump * weight(quadrature, point_index) * weight_scale
  end

  return jump_energy
end

"""
    interface_jump_indicators(state, field)

Per-axis interface jump indicators on each active leaf.

For every interior or periodic face patch orthogonal to axis `a`, this
indicator accumulates the squared trace jump of `field` over that patch and
assigns the contribution to both adjacent leaves on axis `a`. Each side uses
its own local DG scaling

  ((p_a + 1)^2 / h_a) ∫_F |[u_h]|² dS,

where `p_a` is the local polynomial degree and `h_a` is the cell size normal to
the face. The returned per-leaf, per-axis values are the square roots of those
accumulated contributions.

This is the problem-independent h-refinement signal on DG axes in the compact
adaptivity planner. The idea is the standard DG heuristic: if two neighboring
leaves represent a smooth solution well, then their traces should already agree
reasonably closely across their common face. Large jumps therefore point to
under-resolution normal to that interface.
"""
function interface_jump_indicators(state::State{T}, field::AbstractField) where {T<:AbstractFloat}
  field_dof_range(field_layout(state), field)
  space = field_space(field)
  D = dimension(space)
  active_leaf_total = active_leaf_count(space)
  space_snapshot = snapshot(space)
  interface_total = interface_count(space_snapshot)
  interface_total == 0 && return [ntuple(_ -> zero(T), D) for _ in 1:active_leaf_total]
  domain_data = domain(space)
  component_coefficients = _component_coefficient_views(state, field)
  quadrature_cache = Dict{Tuple{Int,NTuple{D-1,Int}},TensorQuadrature{D-1,T}}()
  scratch = _InterfaceJumpScratch(T, Val(D), active_leaf_total)

  for interface_index in 1:interface_total
    minus_leaf = @inbounds space_snapshot.interface_minus[interface_index]
    axis = Int(@inbounds space_snapshot.interface_axis[interface_index])
    plus_leaf = @inbounds space_snapshot.interface_plus[interface_index]
    minus_leaf_index = @inbounds space_snapshot.leaf_to_index[minus_leaf]
    plus_leaf_index = @inbounds space_snapshot.leaf_to_index[plus_leaf]
    minus_compiled = @inbounds space.compiled_leaves[minus_leaf_index]
    plus_compiled = @inbounds space.compiled_leaves[plus_leaf_index]
    quadrature = _interface_jump_quadrature!(quadrature_cache, T, minus_compiled, plus_compiled,
                                             axis)
    jump_energy = _interface_jump_energy(component_coefficients, minus_compiled, plus_compiled,
                                         domain_data, minus_leaf, plus_leaf, axis, quadrature,
                                         scratch)
    isfinite(jump_energy) || throw(ArgumentError("interface jump indicators must be finite"))
    minus_scale = ((minus_compiled.degrees[axis] + 1)^2) / cell_size(domain_data, minus_leaf, axis)
    plus_scale = ((plus_compiled.degrees[axis] + 1)^2) / cell_size(domain_data, plus_leaf, axis)
    scratch.indicators[minus_leaf_index, axis] += minus_scale * jump_energy
    scratch.indicators[plus_leaf_index, axis] += plus_scale * jump_energy
  end

  indicators = scratch.indicators
  return [ntuple(axis -> begin
                   value = indicators[leaf_index, axis]
                   isfinite(value) ||
                     throw(ArgumentError("interface jump indicators must be finite"))
                   value >= zero(T) ||
                     throw(ArgumentError("interface jump indicators must be non-negative"))
                   sqrt(value)
                 end, D) for leaf_index in 1:active_leaf_total]
end

"""
    coefficient_coarsening_indicators(state, field)

Per-axis normalized top-layer modal energy ratios on each active leaf.

If the modal energy contained in the highest layer is small relative to the
total local modal energy, removing that layer should have little effect on the
local approximation. If the ratio is large, the same quantity is a natural
signal that the current polynomial degree is still carrying resolved detail.
At `p_axis = 1`, the endpoint pair is interpreted as constant versus linear
content before the top-layer energy is formed. The returned quantity is
`√(E_top / E_total)` per axis.
"""
function coefficient_coarsening_indicators(state::State{T},
                                           field::AbstractField) where {T<:AbstractFloat}
  return _modal_axis_detail_data(state, field).detail
end

# One modal pass produces both normalized top-layer detail and top-to-previous
# decay. Keeping them together avoids re-reading the local modal expansion and
# makes the planner's "magnitude first, regularity second" policy explicit.
struct _ModalAxisDetailData{D,T<:AbstractFloat}
  detail::Vector{NTuple{D,T}}
  decay::Vector{NTuple{D,T}}
end

# Scratch for modal detail evaluation. The arrays are indexed by logical axis
# and reused across leaves to avoid per-leaf temporary allocation.
struct _ModalAxisDetailScratch{T<:AbstractFloat}
  top::Vector{T}
  previous::Vector{T}
  mode_energy::Vector{T}
end

# Convert the two modal layer energies into the decay ratio used by the h/p
# classifier. A nonzero top layer with vanishing previous layer is treated as
# rough, because the current polynomial order does not show reliable decay.
function _modal_decay_value(top_energy::T, previous_energy::T) where {T<:AbstractFloat}
  top_energy == zero(T) && return zero(T)
  previous_energy > eps(T) * top_energy || return floatmax(T)
  ratio = top_energy / previous_energy
  return isfinite(ratio) ? sqrt(ratio) : floatmax(T)
end

# Modal detail and modal decay are computed together because both require the
# same local modal layer energies. The normalized top-layer detail decides
# whether adaptation is needed; the top-to-previous decay ratio is only an
# internal h/p classifier for axes where the detail is significant.
function _modal_axis_detail_data(state::State{T}, field::AbstractField) where {T<:AbstractFloat}
  field_dof_range(field_layout(state), field)
  space = field_space(field)
  D = dimension(space)
  component_coefficients = _component_coefficient_views(state, field)
  detail = Vector{NTuple{D,T}}(undef, active_leaf_count(space))
  decay = Vector{NTuple{D,T}}(undef, active_leaf_count(space))
  scratch = _ModalAxisDetailScratch(zeros(T, D), zeros(T, D), T[])

  for leaf_index in eachindex(snapshot(space).active_leaves)
    compiled = space.compiled_leaves[leaf_index]
    resize!(scratch.mode_energy, length(compiled.local_modes))
    fill!(scratch.top, zero(T))
    fill!(scratch.previous, zero(T))
    total_energy = zero(T)

    for mode_index in eachindex(compiled.local_modes)
      amplitude_squared = _local_mode_energy(component_coefficients, compiled, mode_index)
      scratch.mode_energy[mode_index] = amplitude_squared
      total_energy += amplitude_squared
    end

    for axis in 1:D
      scratch.top[axis], scratch.previous[axis] = _axis_layer_energies(component_coefficients,
                                                                       compiled, axis,
                                                                       scratch.mode_energy)
    end

    detail[leaf_index] = total_energy == zero(T) ? ntuple(_ -> zero(T), D) :
                         ntuple(axis -> sqrt(scratch.top[axis] / total_energy), D)
    decay[leaf_index] = ntuple(axis -> _modal_decay_value(scratch.top[axis], scratch.previous[axis]),
                               D)
  end

  return _ModalAxisDetailData{D,T}(detail, decay)
end

# Threshold handling is centralized so the compact planner and any diagnostic
# entry points share the same admissibility and error semantics.
function _checked_nonnegative_threshold(value::Real, name::AbstractString)
  checked = float(value)
  isfinite(checked) || throw(ArgumentError("$name must be finite"))
  checked >= 0 || throw(ArgumentError("$name must be non-negative"))
  return checked
end

function _require_finite_axis_indicator_data(values, name::AbstractString)
  for leaf_values in values
    for axis in eachindex(leaf_values)
      isfinite(leaf_values[axis]) || throw(ArgumentError("$name must be finite"))
    end
  end

  return values
end

function _active_leaf_index(space::HpSpace, leaf::Integer)
  checked_leaf = _checked_cell(grid(space), leaf)
  index = @inbounds snapshot(space).leaf_to_index[checked_leaf]
  index != 0 || throw(ArgumentError("leaf $checked_leaf is not an active space leaf"))
  return index
end

# Projection-based `h`-coarsening indicators on immediate parent candidates.

# Map child reference coordinates ξ_child ∈ [-1, 1]^D to the parent reference
# coordinates ξ_parent for one immediate dyadic split. This is the geometric key
# to projecting child data onto a parent basis during candidate scoring.
function _child_point_in_parent(space::HpSpace{D,T}, parent::Int, child::Int,
                                ξ_child::NTuple{D,<:Real}) where {D,T<:AbstractFloat}
  grid_data = grid(space)
  split, child_offset = _direct_child_split_axis_and_offset(grid_data, parent, child)
  lower_child = child_offset == 0

  return ntuple(axis -> begin
                  ξ = T(ξ_child[axis])

                  if axis == split
                    lower_child ? (ξ - one(T)) / T(2) : (ξ + one(T)) / T(2)
                  else
                    ξ
                  end
                end, D)
end

# Evaluate one tensor-product parent basis mode from per-axis basis tables. The
# helper is used in the inner projection loops to avoid rebuilding tuple logic
# at each call site.
function _projection_shape_value(modes::AbstractVector{<:NTuple{D,Int}},
                                 basis_values::NTuple{D,<:AbstractVector},
                                 mode_index::Int) where {D}
  mode = modes[mode_index]
  value = one(eltype(basis_values[1]))

  for axis in 1:D
    value *= basis_values[axis][mode[axis]+1]
  end

  return value
end

# Indicator volume integrals use the same reference-cell measure convention as
# assembly: full Cartesian cells fall back to tensor Gauss rules, while physical
# cut cells use the domain's physical or finite-cell quadrature.
function _cached_tensor_cell_quadrature!(cache::Dict{NTuple{D,Int},TensorQuadrature{D,T}},
                                         ::Type{T}, shape::NTuple{D,Int}) where {D,T<:AbstractFloat}
  return get!(cache, shape) do
    TensorQuadrature(T, shape)
  end
end

function _indicator_cell_quadrature!(cache::Dict{NTuple{D,Int},TensorQuadrature{D,T}}, ::Type{T},
                                     domain::Domain{D,T}, leaf::Int,
                                     shape::NTuple{D,Int}) where {D,T<:AbstractFloat}
  return _cached_tensor_cell_quadrature!(cache, T, shape)
end

function _indicator_cell_quadrature!(cache::Dict{NTuple{D,Int},TensorQuadrature{D,T}}, ::Type{T},
                                     domain::PhysicalDomain{D,T}, leaf::Int,
                                     shape::NTuple{D,Int}) where {D,T<:AbstractFloat}
  _classify_leaf(domain.region, domain, leaf) === :outside && return nothing
  quadrature = _assembly_cell_quadrature(domain, leaf, shape)
  return quadrature === nothing ? _cached_tensor_cell_quadrature!(cache, T, shape) : quadrature
end

_uses_full_cell_reference_measure(::Domain, leaf::Int) = true

function _uses_full_cell_reference_measure(domain::PhysicalDomain{D,T},
                                           leaf::Int) where {D,T<:AbstractFloat}
  _classify_leaf(domain.region, domain, leaf) === :inside && return true
  measure = _cell_measure(domain)
  return measure isa FiniteCellExtension && T(measure.alpha) == one(T)
end

function _projection_mass_factor(::Type{T}, degrees::NTuple{D,Int},
                                 modes::AbstractVector{<:NTuple{D,Int}},
                                 quadrature::AbstractQuadrature{D,T}) where {D,T<:AbstractFloat}
  mode_count = length(modes)
  mass = zeros(T, mode_count, mode_count)
  basis_values = ntuple(axis -> Vector{T}(undef, degrees[axis] + 1), D)
  shape_values = Vector{T}(undef, mode_count)

  for point_index in 1:point_count(quadrature)
    ξ = point(quadrature, point_index)
    _fill_leaf_basis!(basis_values, degrees, ξ)
    _fill_projection_shape_values!(shape_values, modes, basis_values)
    weighted = weight(quadrature, point_index)

    for row_index in 1:mode_count
      shape_row = shape_values[row_index]

      for column_index in 1:row_index
        contribution = shape_row * shape_values[column_index] * weighted
        mass[row_index, column_index] += contribution
        row_index == column_index || (mass[column_index, row_index] += contribution)
      end
    end
  end

  return cholesky(Hermitian(mass))
end

# Precompute the parent projection mass matrix and basis modes for one
# h-coarsening candidate. Full-cell masses are cached by degree tuple; cut-cell
# masses depend on the candidate leaf geometry and are factored per candidate.
function _reference_projection_data(space::HpSpace{D,T}, candidate::HCoarseningCandidate{D},
                                    cache::Dict{NTuple{D,Int},
                                                Tuple{Vector{NTuple{D,Int}},Cholesky{T,Matrix{T}}}},
                                    quadrature_cache::Dict{NTuple{D,Int},TensorQuadrature{D,T}}) where {D,
                                                                                                        T<:AbstractFloat}
  degrees = candidate.target_degrees
  full_reference_measure = _uses_full_cell_reference_measure(domain(space), candidate.cell)
  full_reference_measure && haskey(cache, degrees) && return cache[degrees]
  modes = collect(basis_modes(basis_family(space), degrees))
  shape = ntuple(axis -> minimum_gauss_legendre_points(2 * degrees[axis]), D)
  quadrature = _indicator_cell_quadrature!(quadrature_cache, T, domain(space), candidate.cell,
                                           shape)
  quadrature === nothing &&
    throw(ArgumentError("h coarsening candidate parent $(candidate.cell) has no active integration measure"))
  factor = _projection_mass_factor(T, degrees, modes, quadrature)
  data = (modes, factor)
  full_reference_measure && (cache[degrees] = data)
  return data
end

# Build the per-candidate projection references while sharing factorizations for
# repeated target degree tuples. The returned maximum mode count sizes the
# scratch buffers used during candidate scoring.
function _projection_reference_table(space::HpSpace{D,T},
                                     candidates::AbstractVector{HCoarseningCandidate{D}},
                                     quadrature_cache::Dict{NTuple{D,Int},TensorQuadrature{D,T}}) where {D,
                                                                                                         T<:AbstractFloat}
  reference_type = Tuple{Vector{NTuple{D,Int}},Cholesky{T,Matrix{T}}}
  cache = Dict{NTuple{D,Int},reference_type}()
  references = Vector{reference_type}(undef, length(candidates))
  max_mode_count = 0

  for candidate_index in eachindex(candidates)
    reference = _reference_projection_data(space, candidates[candidate_index], cache,
                                           quadrature_cache)
    references[candidate_index] = reference
    max_mode_count = max(max_mode_count, length(reference[1]))
  end

  return references, max_mode_count
end

# The projection quadrature must integrate products between parent target modes
# and child fine modes. Using twice the maximum degree on each axis yields exact
# Gauss-Legendre integration for the polynomial products that appear.
function _projection_quadrature_shape(target_degrees::NTuple{D,Int},
                                      child_degrees::NTuple{D,Int}) where {D}
  return ntuple(axis -> minimum_gauss_legendre_points(2 * max(target_degrees[axis],
                                                              child_degrees[axis])), D)
end

# Projection buffers. Columns correspond to field components and rows to parent
# modes; each candidate uses leading subviews sized by its target basis, so the
# buffers can be reused without allocation across candidates.
struct _ProjectionIndicatorScratch{D,T<:AbstractFloat}
  rhs::Matrix{T}
  coefficients::Matrix{T}
  parent_basis::NTuple{D,Vector{T}}
  child_basis::NTuple{D,Vector{T}}
  component_values::Vector{T}
  shape_values::Vector{T}
end

function _ProjectionIndicatorScratch(::Type{T}, ::Val{D}, max_mode_count::Int,
                                     components::Int) where {D,T<:AbstractFloat}
  return _ProjectionIndicatorScratch{D,T}(Matrix{T}(undef, max_mode_count, components),
                                          Matrix{T}(undef, max_mode_count, components),
                                          ntuple(_ -> T[], D), ntuple(_ -> T[], D),
                                          Vector{T}(undef, components),
                                          Vector{T}(undef, max_mode_count))
end

function _fill_projection_shape_values!(shape_values::AbstractVector{T},
                                        modes::AbstractVector{<:NTuple{D,Int}},
                                        basis_values::NTuple{D,Vector{T}}) where {D,
                                                                                  T<:AbstractFloat}
  length(shape_values) >= length(modes) ||
    throw(ArgumentError("projection shape buffer is too small"))

  for mode_index in eachindex(modes)
    shape_values[mode_index] = _projection_shape_value(modes, basis_values, mode_index)
  end

  return shape_values
end

"""
    projection_coarsening_indicators(state, field, candidates)

Relative local `L²` projection defects for immediate `h`-coarsening candidates.
Smaller values indicate that derefining the candidate parent cell is likely to
be harmless.

The indicator compares the fine representation on the two child leaves with its
`L²` projection onto the candidate parent space and returns the relative defect

`‖u_fine - Π_parent u_fine‖ₗ₂ / ‖u_fine‖ₗ₂`.

Small values therefore indicate that collapsing the children back to the parent
should incur only a small local error.
"""
function projection_coarsening_indicators(state::State{T}, field::AbstractField,
                                          candidates) where {T<:AbstractFloat}
  field_dof_range(field_layout(state), field)
  space = field_space(field)
  checked_candidates = _checked_h_coarsening_candidates(space, candidates)
  indicators = Vector{T}(undef, length(checked_candidates))
  isempty(checked_candidates) && return indicators
  D = dimension(space)
  components = component_count(field)
  component_coefficients = _component_coefficient_views(state, field)
  quadrature_cache = Dict{NTuple{D,Int},TensorQuadrature{D,T}}()
  references, max_mode_count = _projection_reference_table(space, checked_candidates,
                                                           quadrature_cache)
  scratch = _ProjectionIndicatorScratch(T, Val(D), max_mode_count, components)

  for candidate_index in eachindex(checked_candidates)
    candidate = checked_candidates[candidate_index]
    modes, factor = references[candidate_index]
    mode_count = length(modes)
    rhs = @view scratch.rhs[1:mode_count, 1:components]
    coefficients = @view scratch.coefficients[1:mode_count, 1:components]
    shape_values = @view scratch.shape_values[1:mode_count]
    fill!(rhs, zero(T))
    fine_norm = zero(T)
    parent_jacobian = jacobian_determinant_from_biunit_cube(domain(space), candidate.cell)

    # Each child is integrated in its own reference coordinates and each sample
    # is mapped into the candidate parent reference cell before testing against
    # the parent basis. This is what makes h-coarsening indicators robust when
    # the fine representation is genuinely piecewise polynomial on the parent
    # support.
    for child_index in eachindex(candidate.children)
      child = candidate.children[child_index]
      child_compiled = _compiled_leaf(space, child)
      quadrature_shape = _projection_quadrature_shape(candidate.target_degrees,
                                                      child_compiled.degrees)
      quadrature = _indicator_cell_quadrature!(quadrature_cache, T, domain(space), child,
                                               quadrature_shape)
      quadrature === nothing &&
        throw(ArgumentError("h coarsening candidate child $child has no active integration measure"))
      child_jacobian = jacobian_determinant_from_biunit_cube(domain(space), child)

      for point_index in 1:point_count(quadrature)
        ξ_child = point(quadrature, point_index)
        ξ_parent = _child_point_in_parent(space, candidate.cell, child, ξ_child)
        _fill_leaf_basis!(scratch.parent_basis, candidate.target_degrees, ξ_parent)
        _fill_leaf_basis!(scratch.child_basis, child_compiled.degrees, ξ_child)
        _leaf_component_values!(scratch.component_values, child_compiled, component_coefficients,
                                scratch.child_basis)
        _fill_projection_shape_values!(shape_values, modes, scratch.parent_basis)
        weighted = weight(quadrature, point_index) * child_jacobian

        for component in 1:components
          scalar_value = scratch.component_values[component]
          fine_norm += scalar_value * scalar_value * weighted
          weighted_value = scalar_value * weighted

          for mode_index in 1:mode_count
            rhs[mode_index, component] += shape_values[mode_index] * weighted_value
          end
        end
      end
    end

    isfinite(fine_norm) || throw(ArgumentError("h coarsening fine norms must be finite"))
    fine_norm >= zero(T) || throw(ArgumentError("h coarsening fine norms must be non-negative"))

    if fine_norm == zero(T)
      indicators[candidate_index] = zero(T)
      continue
    end

    projection_norm = zero(T)

    # The parent mass factor was built on the reference cell. The right-hand
    # side was accumulated with physical weights, so the parent coefficients are
    # scaled by the parent Jacobian before applying the cached reference-mass
    # factorization.
    for component in 1:components
      @views begin
        rhs_component = rhs[:, component]
        coefficients_component = coefficients[:, component]
        coefficients_component .= rhs_component ./ parent_jacobian
        ldiv!(factor, coefficients_component)

        for mode_index in eachindex(rhs_component)
          projection_norm += rhs_component[mode_index] * coefficients_component[mode_index]
        end
      end
    end

    isfinite(projection_norm) ||
      throw(ArgumentError("h coarsening projection norms must be finite"))
    defect = sqrt(max(zero(T), fine_norm - projection_norm) / fine_norm)
    isfinite(defect) || throw(ArgumentError("h coarsening indicators must be finite"))
    indicators[candidate_index] = defect
  end

  return indicators
end

# Single-tolerance multiresolution adaptivity.

# Cellwise L² energies provide the physical normalization used by the
# multiresolution planner. Modal coefficient ratios and projection defects are
# already relative quantities, but DG jump indicators are trace quantities and
# must be scaled by the local volume energy before the same tolerance can be
# used on CG and DG axes.
function _field_cell_l2_energies(state::State{T}, field::AbstractField) where {T<:AbstractFloat}
  field_dof_range(field_layout(state), field)
  space = field_space(field)
  D = dimension(space)
  energies = zeros(T, active_leaf_count(space))
  components = component_count(field)
  component_coefficients = _component_coefficient_views(state, field)
  basis_scratch = _LeafBasisScratch(T, Val(D))
  component_values = Vector{T}(undef, components)
  quadrature_cache = Dict{NTuple{D,Int},TensorQuadrature{D,T}}()

  for (leaf_index, leaf) in enumerate(snapshot(space).active_leaves)
    compiled = space.compiled_leaves[leaf_index]
    quadrature = _indicator_cell_quadrature!(quadrature_cache, T, domain(space), leaf,
                                             compiled.quadrature_shape)
    quadrature === nothing &&
      throw(ArgumentError("active leaf $leaf has no active integration measure"))

    jacobian = jacobian_determinant_from_biunit_cube(domain(space), leaf)
    energy = zero(T)

    for point_index in 1:point_count(quadrature)
      ξ = point(quadrature, point_index)
      weighted = weight(quadrature, point_index) * jacobian
      _fill_leaf_basis!(basis_scratch.values, compiled.degrees, ξ)
      _leaf_component_values!(component_values, compiled, component_coefficients,
                              basis_scratch.values)

      for component in 1:components
        value = component_values[component]
        energy += value * value * weighted
      end
    end

    isfinite(energy) || throw(ArgumentError("cell L2 indicator energies must be finite"))
    energy >= zero(T) || throw(ArgumentError("cell L2 indicator energies must be non-negative"))
    energies[leaf_index] = energy
  end

  return energies
end

# Convert cell energies into robust normalization denominators. The global
# floor prevents zero-state cells from producing infinite normalized DG jumps
# while still scaling with the magnitude of the represented field.
function _cell_l2_denominators(energies::AbstractVector{T}) where {T<:AbstractFloat}
  global_energy = sum(energies)
  isfinite(global_energy) || throw(ArgumentError("cell L2 indicator energies must be finite"))
  floor_energy = max(global_energy * eps(T), eps(T)^2)
  return [sqrt(max(energy, floor_energy)) for energy in energies]
end

# Normalize face-jump indicators by a cellwise L² scale and by the normal cell
# size so DG jump detail can share the same threshold as modal cell detail.
function _normalized_interface_jump_indicators(state::State{T}, field::AbstractField,
                                               cell_energies::AbstractVector{T}) where {T<:AbstractFloat}
  space = field_space(field)
  raw = interface_jump_indicators(state, field)
  denominators = _cell_l2_denominators(cell_energies)
  result = Vector{NTuple{dimension(space),T}}(undef, active_leaf_count(space))

  for (leaf_index, leaf) in enumerate(snapshot(space).active_leaves)
    result[leaf_index] = ntuple(axis -> raw[leaf_index][axis] *
                                        cell_size(domain(space), leaf, axis) /
                                        denominators[leaf_index], dimension(space))
  end

  _require_finite_axis_indicator_data(result, "normalized interface jump indicators")
  return result
end

# The refinement signal follows the axis continuity. On CG axes the modal
# top-layer ratio is the natural local detail. On DG axes the face jump is the
# more robust h-refinement signal, because a discontinuity may be invisible to a
# purely cell-local modal indicator.
function _continuity_refinement_detail(state::State{T}, field::AbstractField,
                                       p_detail) where {T<:AbstractFloat}
  space = field_space(field)
  D = dimension(space)
  any(axis -> !is_continuous_axis(space, axis), 1:D) || return p_detail
  jump_detail = _normalized_interface_jump_indicators(state, field,
                                                      _field_cell_l2_energies(state, field))

  return [ntuple(axis -> is_continuous_axis(space, axis) ? p_detail[leaf_index][axis] :
                         jump_detail[leaf_index][axis], D)
          for leaf_index in 1:active_leaf_count(space)]
end

# The combined detail is the pointwise maximum of the h- and p-refinement
# details. It is useful for diagnostics and for retaining resolution around
# significant cells; the actual h/p decision still uses the two detail families
# separately.
function _combined_refinement_detail(space::HpSpace{D}, p_detail, h_refinement_detail) where {D}
  return [ntuple(axis -> max(p_detail[leaf_index][axis], h_refinement_detail[leaf_index][axis]), D)
          for leaf_index in 1:active_leaf_count(space)]
end

# Planning needs both the refinement mask data and the candidate-wise
# h-coarsening defects. Keeping them in one object makes the later selection
# stages explicit: per-leaf details decide retention/refinement, while
# candidate details decide whether a parent reconstruction is accurate enough
# to remove its children.
struct _MultiresolutionIndicatorData{D,T<:AbstractFloat,C,V}
  p_detail::Vector{NTuple{D,T}}
  modal_decay::Vector{NTuple{D,T}}
  h_refinement_detail::Vector{NTuple{D,T}}
  refinement_detail::Vector{NTuple{D,T}}
  h_coarsening_candidates::C
  h_coarsening_detail::V
end

# Build only the per-leaf refinement details. This path is used by diagnostics
# and sampled postprocessing as well as by the planner, so it deliberately
# avoids evaluating projection defects for h-coarsening candidates.
function _multiresolution_refinement_data(state::State{T},
                                          field::AbstractField) where {T<:AbstractFloat}
  space = field_space(field)
  modal = _modal_axis_detail_data(state, field)
  p_detail = modal.detail
  h_refinement_detail = _continuity_refinement_detail(state, field, p_detail)
  refinement_detail = _combined_refinement_detail(space, p_detail, h_refinement_detail)
  _require_finite_axis_indicator_data(p_detail, "modal refinement indicators")
  _require_finite_axis_indicator_data(modal.decay, "modal decay indicators")
  _require_finite_axis_indicator_data(h_refinement_detail, "h refinement indicators")
  _require_finite_axis_indicator_data(refinement_detail, "refinement indicators")
  return (; p_detail, modal_decay=modal.decay, h_refinement_detail, refinement_detail)
end

# Full planning data extends diagnostic refinement details with the projection
# defects needed for h-coarsening. This is intentionally separate from
# `multiresolution_indicators`, because sampled postprocessing should not pay
# for candidate projection solves.
function _multiresolution_indicator_data(state::State{T}, field::AbstractField,
                                         limits::AdaptivityLimits) where {T<:AbstractFloat}
  space = field_space(field)
  refinement = _multiresolution_refinement_data(state, field)
  candidates = h_coarsening_candidates(space; limits=limits)
  h_coarsening_detail = projection_coarsening_indicators(state, field, candidates)
  return _MultiresolutionIndicatorData(refinement.p_detail, refinement.modal_decay,
                                       refinement.h_refinement_detail, refinement.refinement_detail,
                                       candidates, h_coarsening_detail)
end

"""
    multiresolution_indicators(state, field; limits=AdaptivityLimits(field_space(field)))

Return the normalized per-leaf, per-axis detail indicators used by
[`adaptivity_plan`](@ref).

The returned field is the union of the normalized h- and p-refinement details.
The p detail is the relative modal energy in the highest polynomial layer. The
h detail follows the continuity of each axis: CG axes use the same modal detail,
while DG axes use normalized interface jumps. Immediate `h`-coarsening
candidates are still checked by local `L²` projection defects inside
[`adaptivity_plan`](@ref), but those removal defects are not part of this
refinement diagnostic.
"""
function multiresolution_indicators(state::State, field::AbstractField;
                                    limits=AdaptivityLimits(field_space(field)))
  space = field_space(field)
  _checked_limits(limits, space)
  return _multiresolution_refinement_data(state, field).refinement_detail
end

# The h/p choice is fallback-based: try the preferred operation first, then the
# other admissible operation before leaving a marked axis unchanged.
function _axis_refinement_choice(space::HpSpace, leaf::Int, axis::Int, prefer_p::Bool,
                                 limits::AdaptivityLimits)
  if prefer_p
    _can_p_refine(space, leaf, axis, limits) && return 1
    _can_h_refine(space, leaf, axis, limits) && return -1
  else
    _can_h_refine(space, leaf, axis, limits) && return -1
    _can_p_refine(space, leaf, axis, limits) && return 1
  end

  return 0
end

# At degree one, top-to-previous modal decay is not a stable regularity
# classifier: a pure linear mode with zero mean would look rough although it is
# smooth. The hp split therefore uses decay only once at least two non-constant
# modal layers are available on that axis.
function _prefer_modal_p_refinement(space::HpSpace, leaf::Int, axis::Int, modal_decay,
                                    smoothness_threshold)
  cell_degrees(space, leaf)[axis] <= 1 && return true
  return modal_decay <= smoothness_threshold
end

# The tolerance marks axes that still carry detail. Modal decay then classifies
# marked modal axes: fast decay prefers p-enrichment, while stalled decay
# prefers h-refinement. DG jumps above tolerance override that classifier and
# remain h-first because discontinuities are primarily geometric
# under-resolution rather than missing polynomial order.
function _multiresolution_refinement_axes(space::HpSpace{D}, data::_MultiresolutionIndicatorData,
                                          tolerance, smoothness_threshold,
                                          limits::AdaptivityLimits{D}) where {D}
  h_refined = _empty_axis_flags(space)
  p_refined = _empty_axis_flags(space)

  for (leaf_index, leaf) in enumerate(snapshot(space).active_leaves)
    h_current = h_refined[leaf_index]
    p_current = p_refined[leaf_index]

    for axis in 1:D
      h_value = data.h_refinement_detail[leaf_index][axis]
      p_value = data.p_detail[leaf_index][axis]
      h_significant = h_value > tolerance
      p_significant = p_value > tolerance
      h_significant || p_significant || continue
      choice = if !is_continuous_axis(space, axis) && h_significant
        _axis_refinement_choice(space, leaf, axis, false, limits)
      elseif p_significant
        prefer_p = _prefer_modal_p_refinement(space, leaf, axis, data.modal_decay[leaf_index][axis],
                                              smoothness_threshold)
        _axis_refinement_choice(space, leaf, axis, prefer_p, limits)
      else
        _axis_refinement_choice(space, leaf, axis, false, limits)
      end

      choice < 0 &&
        (h_current = ntuple(current_axis -> current_axis == axis ? true : h_current[current_axis],
                            D))
      choice > 0 &&
        (p_current = ntuple(current_axis -> current_axis == axis ? true : p_current[current_axis],
                            D))
    end

    h_refined[leaf_index] = h_current
    p_refined[leaf_index] = p_current
  end

  return h_refined, p_refined
end

# Extend h-refinement requests by one face-neighbor ring in the same marked axes.
# This fixed buffer keeps moving transient features from immediately outrunning
# the adapted mesh and reduces isolated refinement/coarsening oscillations.
function _expanded_multiresolution_h_zone(space::HpSpace{D}, h_refined,
                                          limits::AdaptivityLimits{D}) where {D}
  expanded = copy(h_refined)
  grid_data = grid(space)
  space_snapshot = snapshot(space)
  neighbor_lookup = _SnapshotNeighborLookup(space_snapshot)

  for (leaf_index, axes) in enumerate(h_refined)
    any(axes) || continue
    leaf = active_leaf(space, leaf_index)
    leaf_levels = level(grid_data, leaf)

    for face_axis in 1:D
      for side in (LOWER, UPPER)
        for neighbor_leaf in _opposite_active_leaves(neighbor_lookup, leaf, face_axis, side)
          neighbor_index = @inbounds space_snapshot.leaf_to_index[neighbor_leaf]
          neighbor_index == 0 && continue
          neighbor_levels = level(grid_data, neighbor_leaf)
          current = expanded[neighbor_index]

          for axis in 1:D
            axes[axis] || continue
            neighbor_levels[axis] <= leaf_levels[axis] || continue
            _can_h_refine(space, neighbor_leaf, axis, limits) || continue
            current = ntuple(current_axis -> current_axis == axis ? true : current[current_axis], D)
          end

          expanded[neighbor_index] = current
        end
      end
    end
  end

  return expanded
end

# Significant leaves are retained even if they are not themselves modified by the
# fallback h/p decision, because they still contain resolved detail above the
# global tolerance.
function _significant_multiresolution_leaves(space::HpSpace, data::_MultiresolutionIndicatorData,
                                             tolerance)
  significant = falses(active_leaf_count(space))

  for leaf_index in 1:active_leaf_count(space)
    significant[leaf_index] = any(axis -> data.refinement_detail[leaf_index][axis] > tolerance,
                                  1:dimension(space))
  end

  return significant
end

# Coarsening is blocked on changed/significant leaves and on their one-ring face
# neighbors. The neighbor retention mirrors the h-refinement buffer and prevents
# deleting support immediately next to currently resolved detail.
function _multiresolution_h_block_flags(space::HpSpace{D}, h_refined, p_refined,
                                        significant) where {D}
  length(significant) == active_leaf_count(space) ||
    throw(ArgumentError("significant leaf flags must match the active-leaf count"))
  blocked = [any(h_refined[index]) || any(p_refined[index]) || significant[index]
             for index in eachindex(h_refined)]
  space_snapshot = snapshot(space)
  neighbor_lookup = _SnapshotNeighborLookup(space_snapshot)

  for (leaf_index, is_blocked) in enumerate(copy(blocked))
    is_blocked || continue
    leaf = active_leaf(space, leaf_index)

    for axis in 1:D
      for side in (LOWER, UPPER)
        for neighbor_leaf in _opposite_active_leaves(neighbor_lookup, leaf, axis, side)
          neighbor_index = @inbounds space_snapshot.leaf_to_index[neighbor_leaf]
          neighbor_index == 0 && continue
          blocked[neighbor_index] = true
        end
      end
    end
  end

  return blocked
end

# Select immediate h-coarsening moves whose projection defect is below the same
# tolerance used for refinement, excluding all candidates touching blocked
# children. The returned leaf flags prevent p-coarsening on children that are
# about to disappear.
function _multiresolution_h_coarsening_candidates(space::HpSpace{D}, candidates, indicators,
                                                  tolerance, blocked) where {D}
  length(blocked) == active_leaf_count(space) ||
    throw(ArgumentError("blocked leaf flags must match the active-leaf count"))
  selected = HCoarseningCandidate{D}[]
  h_coarsened = falses(active_leaf_count(space))

  for candidate_index in eachindex(candidates)
    candidate = candidates[candidate_index]
    value = float(indicators[candidate_index])
    isfinite(value) || throw(ArgumentError("h coarsening indicators must be finite"))
    value <= tolerance || continue
    any(blocked[_active_leaf_index(space, child)] for child in candidate.children) && continue
    push!(selected, candidate)

    for child in candidate.children
      h_coarsened[_active_leaf_index(space, child)] = true
    end
  end

  return selected, h_coarsened
end

# On leaves not protected by refinement, significance, or h-coarsening, remove
# one polynomial layer on axes whose modal top-layer detail is below tolerance.
function _multiresolution_p_coarsening_axes(space::HpSpace{D}, data::_MultiresolutionIndicatorData,
                                            tolerance, limits::AdaptivityLimits{D},
                                            blocked) where {D}
  p_coarsened = _empty_axis_flags(space)

  for (leaf_index, leaf) in enumerate(snapshot(space).active_leaves)
    blocked[leaf_index] && continue
    current = p_coarsened[leaf_index]

    for axis in 1:D
      data.p_detail[leaf_index][axis] <= tolerance || continue
      _can_p_derefine(space, leaf, axis, limits) || continue
      current = ntuple(current_axis -> current_axis == axis ? true : current[current_axis], D)
    end

    p_coarsened[leaf_index] = current
  end

  return p_coarsened
end

"""
    adaptivity_plan(state, field; tolerance=1.0e-3,
                    smoothness_threshold=0.5,
                    limits=AdaptivityLimits(field_space(field)))

Build an `h`, `p`, or mixed `hp` adaptivity plan from one multiresolution
tolerance and compact h/p policy.

The planner treats adaptivity as coefficient thresholding rather than bulk
marking. It forms normalized FE details with fixed roles: modal top-layer energy
marks resolved detail, modal top-to-previous decay chooses the h/p split for
modal refinement, normalized DG interface jumps override this with h-first
refinement on discontinuous axes, and local `L²` projection defects decide
whether an immediate h-coarsening candidate may be removed. Details above
`tolerance` are retained by refining in `h` or `p`, depending on axis
continuity, modal decay, and admissible limits. Details below `tolerance` may be
removed by derefining in `h` or `p`. The same tolerance is therefore used for
refinement and coarsening.

`smoothness_threshold` is the optional advanced control for the h/p classifier.
On marked modal axes, decay values at or below this threshold are considered
smooth and prefer p-refinement; larger values prefer h-refinement. The default
keeps this regularity heuristic internal for typical use while still allowing
expert adjustment. Degree-one axes do not contain enough modal history for a
stable decay estimate, so they prefer p-refinement unless the active limits force
h-refinement.

The public policy controls remain compact: `tolerance`, `smoothness_threshold`,
and admissible `h`/`p` limits. Pure `h` adaptation is obtained by fixing
`min_p == max_p`; pure `p` adaptation is obtained by fixing
`min_h_level == max_h_level`. The planner also applies a fixed one-ring
retention zone around significant details, following the second-generation
wavelet idea that significant details should keep enough nearby resolution for
transient motion.

Advanced API: this planner is a compact default policy, not a replacement for
problem-specific error estimation. Users who need release-stable adaptation
decisions should build an [`AdaptivityPlan`](@ref) manually or wrap this helper
behind their own application policy.
"""
function adaptivity_plan(state::State, field::AbstractField; tolerance::Real=1.0e-3,
                         smoothness_threshold::Real=0.5,
                         limits=AdaptivityLimits(field_space(field)))
  space = field_space(field)
  checked_limits = _checked_limits(limits, space)
  checked_tolerance = _checked_nonnegative_threshold(tolerance, "tolerance")
  checked_smoothness = _checked_nonnegative_threshold(smoothness_threshold, "smoothness_threshold")
  data = _multiresolution_indicator_data(state, field, checked_limits)
  h_refined, p_refined = _multiresolution_refinement_axes(space, data, checked_tolerance,
                                                          checked_smoothness, checked_limits)
  h_refined = _expanded_multiresolution_h_zone(space, h_refined, checked_limits)
  significant = _significant_multiresolution_leaves(space, data, checked_tolerance)
  blocked = _multiresolution_h_block_flags(space, h_refined, p_refined, significant)
  selected_h, h_coarsened = _multiresolution_h_coarsening_candidates(space,
                                                                     data.h_coarsening_candidates,
                                                                     data.h_coarsening_detail,
                                                                     checked_tolerance, blocked)
  p_blocked = [blocked[index] || h_coarsened[index] for index in eachindex(blocked)]
  p_coarsened = _multiresolution_p_coarsening_axes(space, data, checked_tolerance, checked_limits,
                                                   p_blocked)
  return _adaptivity_plan_from_selections(space, h_refined, p_refined, p_coarsened, selected_h;
                                          limits=checked_limits)
end

function _empty_axis_flags(space::HpSpace{D}) where {D}
  return fill(ntuple(_ -> false, D), active_leaf_count(space))
end

# Convert selected h/p edits into one batched target-space plan. The helper is
# shared by the compact automatic planner and by tests that need to inspect the
# same plan construction path without duplicating degree-change bookkeeping.
function _adaptivity_plan_from_selections(space::HpSpace{D}, h_refinement_axes, p_refinement_axes,
                                          p_coarsening_axes, h_coarsening_candidates;
                                          limits::AdaptivityLimits{D}=AdaptivityLimits(space)) where {D}
  length(p_refinement_axes) == active_leaf_count(space) ||
    throw(ArgumentError("p refinement axes must match the active-leaf count"))
  length(p_coarsening_axes) == active_leaf_count(space) ||
    throw(ArgumentError("p coarsening axes must match the active-leaf count"))
  p_degree_changes = Vector{NTuple{D,Int}}(undef, active_leaf_count(space))

  for leaf_index in eachindex(p_degree_changes)
    refined = p_refinement_axes[leaf_index]
    coarsened = p_coarsening_axes[leaf_index]
    p_degree_changes[leaf_index] = ntuple(axis -> (refined[axis] ? 1 : 0) -
                                                  (coarsened[axis] ? 1 : 0), D)
  end

  return _batched_adaptivity_plan(space, p_degree_changes, h_refinement_axes,
                                  h_coarsening_candidates; limits=limits)
end

# This file moves discrete field data across a `SpaceTransition`. The transfer
# is formulated as a geometric L² projection from source fields to target
# fields: target coefficients are chosen so the target finite-element field
# matches the old field in the target inner product as closely as the target
# space permits.
#
# The code is deliberately independent of the PDE that produced the state. It
# only needs the source/target spaces, the old coefficients, and the overlap
# table compiled by `transition.jl`. That separation is important for adaptive
# workflows: the same transfer path can move affine solutions, nonlinear
# iterates, diagnostics, or auxiliary fields without rebuilding a problem.
#
# There are three execution strategies:
#
# 1. Fully DG target spaces use one dense projection solve per target cell and
#    write coefficients directly because no dof is shared across cells.
# 2. The default CG/mixed path solves the same cell-local projections and then
#    reconciles shared target dofs through small normal-equation components.
# 3. Expert callers may provide a custom `linear_solve`, in which case CG/mixed
#    targets fall back to a global variational projection problem.
#
# Source-to-target h-coarsening is handled by integrating each overlapping
# source leaf separately. The old field is never sampled as if it were one
# polynomial over a coarser target cell; this preserves the correct piecewise
# polynomial content during coarsening.

# State transfer is formulated as an `L²` projection on the target space. The
# local mass operator below provides the symmetric positive definite block that
# defines the target inner product for one field.
struct _TransferMass{F}
  field::F
end

# The transfer source operator evaluates the old state over source-target
# overlap quadrature and injects those values into the right-hand side of the
# projection system. Together with `_TransferMass` this yields the Galerkin
# `L²` projection from the source field to the target field.
struct _TransferSource{F,C,TR,B,P,W}
  field::F
  old_coefficients::C
  transition::TR
  source_basis::B
  target_basis::B
  reference_points::P
  reference_weights::W
end

# Compiled setup for the fully-DG transfer path. It mirrors the affine fallback
# inputs but keeps only target-cell integration data and the maximum dense block
# size needed for reusable scratch storage.
struct _CellwiseDGTransferPlan{D,T<:AbstractFloat,L,C,O,N,TR}
  layout::L
  cells::C
  old_fields::O
  new_fields::N
  transition::TR
  max_local_dofs::Int
end

# Compiled setup for the default CG/mixed transfer path. The cell-local
# projection systems are the same as the DG path, but their local solutions are
# accumulated into a global normal-equation graph because continuous and hanging
# dofs may appear in several local rows with nontrivial coefficients.
struct _LocalProjectionTransferPlan{D,T<:AbstractFloat,L,C,O,N,TR}
  layout::L
  cells::C
  old_fields::O
  new_fields::N
  transition::TR
  max_local_dofs::Int
end

# Dense local buffers are reused while solving cellwise projection systems.
# Fully-DG cells have disjoint global dofs, so the final coefficient writes do
# not need an accumulator.
mutable struct _CellwiseDGTransferScratch{D,T<:AbstractFloat}
  matrix::Matrix{T}
  rhs::Vector{T}
  source_basis::NTuple{D,Vector{T}}
  target_basis::NTuple{D,Vector{T}}
  reference_points::Vector{NTuple{D,T}}
  reference_weights::Vector{T}
end

function _CellwiseDGTransferScratch(::Type{T}, ::Val{D},
                                    local_dof_count::Int) where {D,T<:AbstractFloat}
  return _CellwiseDGTransferScratch{D,T}(Matrix{T}(undef, local_dof_count, local_dof_count),
                                         Vector{T}(undef, local_dof_count), ntuple(_ -> T[], D),
                                         ntuple(_ -> T[], D), NTuple{D,T}[], T[])
end

mutable struct _LocalProjectionTransferScratch{D,T<:AbstractFloat}
  matrix::Matrix{T}
  rhs::Vector{T}
  source_basis::NTuple{D,Vector{T}}
  target_basis::NTuple{D,Vector{T}}
  reference_points::Vector{NTuple{D,T}}
  reference_weights::Vector{T}
end

function _LocalProjectionTransferScratch(::Type{T}, ::Val{D},
                                         local_dof_count::Int) where {D,T<:AbstractFloat}
  return _LocalProjectionTransferScratch{D,T}(Matrix{T}(undef, local_dof_count, local_dof_count),
                                              Vector{T}(undef, local_dof_count),
                                              ntuple(_ -> T[], D), ntuple(_ -> T[], D),
                                              NTuple{D,T}[], T[])
end

# Assemble the element-local mass matrix
#   M_ij = ∫_K φ_i φ_j dΩ
# for the target field, duplicated componentwise for vector-valued fields.
function _assemble_transfer_mass!(local_matrix, values::CellValues, field::AbstractField)
  mode_count = local_mode_count(values, field)
  components = component_count(field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      value_row = shape_value(values, field, point_index, row_mode)

      for col_mode in 1:mode_count
        contribution = value_row * shape_value(values, field, point_index, col_mode) * weighted
        contribution == 0 && continue

        for component in 1:components
          row = local_dof_index(values, field, component, row_mode)
          col = local_dof_index(values, field, component, col_mode)
          local_matrix[row, col] += contribution
        end
      end
    end
  end

  return local_matrix
end

function cell_apply!(local_result, operator::_TransferMass, values::CellValues, local_coefficients)
  field = operator.field
  mode_count = local_mode_count(values, field)
  components = component_count(field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for component in 1:components
      trial_value = zero(eltype(local_result))

      for col_mode in 1:mode_count
        col = local_dof_index(values, field, component, col_mode)
        trial_value += shape_value(values, field, point_index, col_mode) * local_coefficients[col]
      end

      trial_value == 0 && continue

      for row_mode in 1:mode_count
        row = local_dof_index(values, field, component, row_mode)
        local_result[row] += shape_value(values, field, point_index, row_mode) *
                             weighted *
                             trial_value
      end
    end
  end

  return nothing
end

# Evaluate the projection right-hand side
#   b_i = ∫_K u_old φ_i dΩ
# over exact source-target overlap boxes. This is essential for h-coarsening:
# a target cell can cover several source leaves, and the old state may only be
# piecewise polynomial on that target cell.
@inline function _compiled_leaf_shape_value(compiled::_CompiledLeaf{D,T},
                                            basis_values::NTuple{D,Vector{T}},
                                            mode_index::Int) where {D,T<:AbstractFloat}
  mode = @inbounds compiled.local_modes[mode_index]
  value = one(T)

  @inbounds for axis in 1:D
    value *= basis_values[axis][mode[axis]+1]
  end

  return value
end

function _target_overlap_reference_bounds(target_domain::AbstractDomain{D,T}, target_leaf::Int,
                                          source_domain::AbstractDomain{D,T},
                                          source_leaf::Int) where {D,T<:AbstractFloat}
  lower = ntuple(axis -> max(cell_lower(target_domain, target_leaf, axis),
                             cell_lower(source_domain, source_leaf, axis)), D)
  upper = ntuple(axis -> min(cell_upper(target_domain, target_leaf, axis),
                             cell_upper(source_domain, source_leaf, axis)), D)

  for axis in 1:D
    upper[axis] > lower[axis] || return nothing
  end

  return map_to_biunit_cube(target_domain, target_leaf, lower),
         map_to_biunit_cube(target_domain, target_leaf, upper)
end

function _transfer_overlap_quadrature_shape(source_space_data::HpSpace{D}, source_leaf::Int,
                                            target_space_data::HpSpace{D},
                                            target_leaf::Int) where {D}
  source_shape = cell_quadrature_shape(source_space_data, source_leaf)
  target_shape = cell_quadrature_shape(target_space_data, target_leaf)
  return ntuple(axis -> max(source_shape[axis], target_shape[axis]), D)
end

function _append_reference_subcell_tensor_quadrature!(points::Vector{NTuple{D,T}},
                                                      weights::Vector{T}, ::Type{T},
                                                      shape::NTuple{D,Int}, lower::NTuple{D,T},
                                                      upper::NTuple{D,T},
                                                      weight_scale::T) where {D,T<:AbstractFloat}
  quadrature = TensorQuadrature(T, shape)
  center = ntuple(axis -> T(0.5) * (lower[axis] + upper[axis]), D)
  half_size = ntuple(axis -> T(0.5) * (upper[axis] - lower[axis]), D)
  scale = weight_scale * prod(half_size)

  for point_index in 1:point_count(quadrature)
    η = point(quadrature, point_index)
    ξ = ntuple(axis -> muladd(half_size[axis], η[axis], center[axis]), D)
    push!(points, ξ)
    push!(weights, T(weight(quadrature, point_index)) * scale)
  end

  return nothing
end

function _append_physical_overlap_reference_quadrature!(points::Vector{NTuple{D,T}},
                                                        weights::Vector{T},
                                                        domain::PhysicalDomain{D,T}, leaf::Int,
                                                        shape::NTuple{D,Int}, lower::NTuple{D,T},
                                                        upper::NTuple{D,T},
                                                        weight_scale::T) where {D,T<:AbstractFloat}
  classification = _classify_leaf(domain.region, domain, leaf)
  classification == :outside && return nothing
  if classification == :inside
    _append_reference_subcell_tensor_quadrature!(points, weights, T, shape, lower, upper,
                                                 weight_scale)
    return nothing
  end

  base_quadrature = TensorQuadrature(T, shape)
  first_point = length(points) + 1
  _collect_finite_cell_candidates!(points, weights, _background_domain(domain), leaf,
                                   domain.region.classifier, base_quadrature, lower, upper,
                                   domain.region.subdivision_depth)

  for index in first_point:length(weights)
    weights[index] *= weight_scale
  end

  return nothing
end

function _append_transfer_overlap_reference_quadrature!(points::Vector{NTuple{D,T}},
                                                        weights::Vector{T},
                                                        domain::AbstractDomain{D,T}, leaf::Int,
                                                        shape::NTuple{D,Int}, lower::NTuple{D,T},
                                                        upper::NTuple{D,T}) where {D,
                                                                                   T<:AbstractFloat}
  _append_reference_subcell_tensor_quadrature!(points, weights, T, shape, lower, upper, one(T))
  return nothing
end

function _append_transfer_overlap_reference_quadrature!(points::Vector{NTuple{D,T}},
                                                        weights::Vector{T},
                                                        domain::PhysicalDomain{D,T}, leaf::Int,
                                                        shape::NTuple{D,Int}, lower::NTuple{D,T},
                                                        upper::NTuple{D,T}) where {D,
                                                                                   T<:AbstractFloat}
  return _append_transfer_overlap_reference_quadrature!(points, weights, domain, leaf, shape, lower,
                                                        upper, _cell_measure(domain))
end

function _append_transfer_overlap_reference_quadrature!(points::Vector{NTuple{D,T}},
                                                        weights::Vector{T},
                                                        domain::PhysicalDomain{D,T}, leaf::Int,
                                                        shape::NTuple{D,Int}, lower::NTuple{D,T},
                                                        upper::NTuple{D,T},
                                                        ::PhysicalMeasure) where {D,
                                                                                  T<:AbstractFloat}
  # With physical measure, transfer integrates only the physical portion of a
  # cut-cell overlap. This matches the inner product used by assembly on
  # `PhysicalDomain` and avoids transferring data through fictitious volume.
  _append_physical_overlap_reference_quadrature!(points, weights, domain, leaf, shape, lower, upper,
                                                 one(T))
  return nothing
end

function _append_transfer_overlap_reference_quadrature!(points::Vector{NTuple{D,T}},
                                                        weights::Vector{T},
                                                        domain::PhysicalDomain{D,T}, leaf::Int,
                                                        shape::NTuple{D,Int}, lower::NTuple{D,T},
                                                        upper::NTuple{D,T},
                                                        measure::FiniteCellExtension) where {D,
                                                                                             T<:AbstractFloat}
  # Finite-cell extension uses the same split measure as assembly: a small
  # background contribution over the full overlap plus the remaining weight on
  # the physical portion. Transfer therefore respects the stabilization measure
  # instead of silently switching to a pure physical projection.
  alpha = T(measure.alpha)
  alpha == one(T) &&
    return _append_reference_subcell_tensor_quadrature!(points, weights, T, shape, lower, upper,
                                                        one(T))
  alpha > zero(T) &&
    _append_reference_subcell_tensor_quadrature!(points, weights, T, shape, lower, upper, alpha)
  physical_scale = one(T) - alpha
  physical_scale > zero(T) &&
    _append_physical_overlap_reference_quadrature!(points, weights, domain, leaf, shape, lower,
                                                   upper, physical_scale)
  return nothing
end

function _add_transfer_rhs_at_reference_point!(local_rhs, values::CellValues, new_fields::Tuple,
                                               source_coefficients,
                                               source_compiled::_CompiledLeaf{D,T},
                                               target_compiled::_CompiledLeaf{D,T},
                                               source_basis::NTuple{D,Vector{T}},
                                               target_basis::NTuple{D,Vector{T}},
                                               weighted::T) where {D,T<:AbstractFloat}
  for field_index in eachindex(new_fields)
    new_field = new_fields[field_index]
    old_coefficients = source_coefficients[field_index]
    mode_count = local_mode_count(values, new_field)

    for component in 1:component_count(new_field)
      old_value = _leaf_component_value(source_compiled, old_coefficients[component], source_basis)
      weighted_value = old_value * weighted

      for mode_index in 1:mode_count
        shape = _compiled_leaf_shape_value(target_compiled, target_basis, mode_index)
        shape == zero(T) && continue
        row = local_dof_index(values, new_field, component, mode_index)
        local_rhs[row] += weighted_value * shape
      end
    end
  end

  return local_rhs
end

function _assemble_transfer_rhs!(local_rhs, values::CellValues{D,T},
                                 transition::SpaceTransition{D,T}, new_fields::Tuple,
                                 source_coefficients, source_basis::NTuple{D,Vector{T}},
                                 target_basis::NTuple{D,Vector{T}},
                                 reference_points::Vector{NTuple{D,T}},
                                 reference_weights::Vector{T}) where {D,T<:AbstractFloat}
  source_space_data = source_space(transition)
  target_space_data = target_space(transition)
  source_domain = domain(source_space_data)
  target_domain = domain(target_space_data)
  target_leaf = values.leaf
  target_compiled = _compiled_leaf(target_space_data, target_leaf)
  target_jacobian = jacobian_determinant_from_biunit_cube(target_domain, target_leaf)
  first, count = _source_leaf_range_unchecked(transition, target_leaf)

  # For h-coarsening, a target cell may cover several source leaves. Each
  # overlap is quadratured independently, target reference samples are mapped to
  # physical coordinates, and those physical points are then evaluated in the
  # corresponding source leaf. This preserves the old state's discontinuities or
  # fine-scale polynomial pieces inside the coarser target cell.
  for source_row in first:(first+count-1)
    source_leaf = @inbounds transition.source_leaf_data[source_row]
    bounds = _target_overlap_reference_bounds(target_domain, target_leaf, source_domain,
                                              source_leaf)
    bounds === nothing && continue
    lower, upper = bounds
    shape = _transfer_overlap_quadrature_shape(source_space_data, source_leaf, target_space_data,
                                               target_leaf)
    empty!(reference_points)
    empty!(reference_weights)
    _append_transfer_overlap_reference_quadrature!(reference_points, reference_weights,
                                                   target_domain, target_leaf, shape, lower, upper)
    source_compiled = _compiled_leaf(source_space_data, source_leaf)

    for point_index in eachindex(reference_points)
      ξ_target = reference_points[point_index]
      x = map_from_biunit_cube(target_domain, target_leaf, ξ_target)
      ξ_source = map_to_biunit_cube(source_domain, source_leaf, x)
      _fill_leaf_basis!(source_basis, source_compiled.degrees, ξ_source)
      _fill_leaf_basis!(target_basis, target_compiled.degrees, ξ_target)
      weighted = reference_weights[point_index] * target_jacobian
      _add_transfer_rhs_at_reference_point!(local_rhs, values, new_fields, source_coefficients,
                                            source_compiled, target_compiled, source_basis,
                                            target_basis, weighted)
    end
  end

  return local_rhs
end

function cell_rhs!(local_rhs, operator::_TransferSource{F,C,TR,B,P,W},
                   values::CellValues{D,T}) where {D,T<:AbstractFloat,F,C,TR<:SpaceTransition{D,T},
                                                   B,P,W}
  _assemble_transfer_rhs!(local_rhs, values, operator.transition, (operator.field,),
                          (operator.old_coefficients,), operator.source_basis,
                          operator.target_basis, operator.reference_points,
                          operator.reference_weights)
  return nothing
end

# When the target space is fully DG, no local mode shares coefficients across
# cells. The transfer mass matrix is therefore block diagonal with one dense
# block per active target cell, so it is cheaper to solve those local systems
# directly than to assemble one global sparse projection problem.
_is_fully_dg_space(space::HpSpace) = all(kind -> kind === :dg, continuity_policy(space))

# Transfer has three deliberately separate execution paths:
#
# - fully DG targets use independent dense cell solves and may route a custom
#   `linear_solve` into those local systems;
# - the default CG/mixed path uses local cell projections followed by small
#   normal-equation components induced by shared dofs;
# - an explicit custom solver on a CG/mixed target uses the global variational
#   projection hook for users who need to override the transfer solve.
#
# Keeping this as a small internal strategy boundary makes the stable transfer
# core independent of the advanced indicator/planner code below.
abstract type _AbstractTransferStrategy end

struct _CellwiseDGTransferStrategy <: _AbstractTransferStrategy end
struct _LocalProjectionTransferStrategy <: _AbstractTransferStrategy end
struct _VariationalTransferStrategy <: _AbstractTransferStrategy end

const _CELLWISE_DG_TRANSFER = _CellwiseDGTransferStrategy()
const _LOCAL_PROJECTION_TRANSFER = _LocalProjectionTransferStrategy()
const _VARIATIONAL_TRANSFER = _VariationalTransferStrategy()

function _transfer_strategy(transition::SpaceTransition, linear_solve)
  _is_fully_dg_space(target_space(transition)) && return _CELLWISE_DG_TRANSFER
  linear_solve === nothing && return _LOCAL_PROJECTION_TRANSFER
  return _VARIATIONAL_TRANSFER
end

# The fast DG transfer path writes local projection coefficients directly into
# global state storage. That is only valid when every local dof has a unique,
# unit-coefficient global term, so the invariant is checked explicitly.
function _checked_cellwise_single_term_mapping(cell::CellValues)
  local_dofs = cell.local_dof_count
  coefficient_tolerance = 1000 * eps(eltype(cell.single_term_coefficients))
  length(cell.single_term_indices) >= local_dofs ||
    throw(ArgumentError("cellwise DG transfer requires one single-term mapping per local dof"))
  length(cell.single_term_coefficients) >= local_dofs ||
    throw(ArgumentError("cellwise DG transfer requires one single-term coefficient per local dof"))

  for local_dof in 1:local_dofs
    cell.single_term_indices[local_dof] >= 1 ||
      throw(ArgumentError("cellwise DG transfer requires each local dof to map to one global dof"))
    abs(cell.single_term_coefficients[local_dof] - one(eltype(cell.single_term_coefficients))) <=
    coefficient_tolerance ||
      throw(ArgumentError("cellwise DG transfer requires unit local-to-global coefficients"))
  end

  return nothing
end

function _checked_cellwise_transfer_solution(::Type{T}, solution,
                                             local_dofs::Int) where {T<:AbstractFloat}
  solution isa AbstractVector || throw(ArgumentError("linear_solve must return a vector"))
  length(solution) == local_dofs ||
    throw(ArgumentError("linear_solve must return one value per local transfer dof"))

  checked = try
    solution isa AbstractVector{T} ? solution : Vector{T}(solution)
  catch error
    error isa InterruptException && rethrow()
    throw(ArgumentError("linear_solve must return values convertible to the transfer scalar type"))
  end

  _require_one_based_vector(checked, "linear_solve result")
  _require_finite_transfer_solution(checked, "linear_solve result")
  return checked
end

function _require_finite_transfer_solution(solution::AbstractVector{T},
                                           name::AbstractString) where {T<:AbstractFloat}
  for index in eachindex(solution)
    isfinite(solution[index]) || throw(ArgumentError("$name must contain only finite values"))
  end

  return solution
end

@inline function _cellwise_transfer_linear_solve(::Type{T}, local_matrix, local_rhs,
                                                 linear_solve) where {T<:AbstractFloat}
  solution = if linear_solve === nothing
    _regularized_transfer_cholesky_solve!(local_matrix, local_rhs)
  else
    linear_solve(local_matrix, local_rhs)
  end

  return _checked_cellwise_transfer_solution(T, solution, length(local_rhs))
end

function _regularized_transfer_cholesky_solve!(matrix_data::AbstractMatrix{T},
                                               rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  n = _require_square_matrix(matrix_data, "regularized dense Cholesky matrix")
  _require_length(rhs_data, n, "rhs")
  scale = zero(T)

  for index in 1:n
    scale = max(scale, abs(@inbounds matrix_data[index, index]))
  end

  if scale == zero(T)
    fill!(rhs_data, zero(T))
    return rhs_data
  end

  # Cut-cell transfer mass blocks can be semidefinite when the physical cell
  # support does not see every modal direction. A relative shift keeps the
  # bounded local projection path robust without introducing an absolute scale.
  shift = 1000 * eps(T) * scale
  for index in 1:n
    @inbounds matrix_data[index, index] += shift
  end

  _dense_cholesky_factor!(matrix_data)
  return _dense_cholesky_solve!(matrix_data, rhs_data)
end

# Check the field pairs once before choosing a transfer implementation. The
# state layout check also gives a targeted error for missing source fields
# instead of failing later during coefficient evaluation.
function _checked_transfer_fields(transition::SpaceTransition, state::State, old_fields::Tuple,
                                  new_fields::Tuple)
  length(old_fields) == length(new_fields) ||
    throw(ArgumentError("old and new field tuples must have the same length"))
  state_layout = field_layout(state)

  for index in eachindex(old_fields)
    old_field = old_fields[index]
    new_field = new_fields[index]
    old_field isa AbstractField || throw(ArgumentError("old fields must be field descriptors"))
    new_field isa AbstractField || throw(ArgumentError("new fields must be field descriptors"))
    field_space(old_field) === source_space(transition) ||
      throw(ArgumentError("old fields must belong to the transition source space"))
    field_space(new_field) === target_space(transition) ||
      throw(ArgumentError("new fields must belong to the transition target space"))
    component_count(old_field) == component_count(new_field) ||
      throw(ArgumentError("field component counts must match during transfer"))
    field_dof_range(state_layout, old_field)
  end

  return nothing
end

# Compile the target cells used by the direct fully-DG projection and verify
# that each local dof maps to exactly one global coefficient.
function _compile_cellwise_dg_transfer_plan(transition::SpaceTransition{D,T}, old_fields::Tuple,
                                            new_fields::Tuple) where {D,T<:AbstractFloat}
  layout = FieldLayout(new_fields)
  overrides = _cell_quadrature_overrides(layout, ())
  cells = _compile_cells(layout, overrides)
  max_local_dofs = 0

  for cell in cells
    _checked_cellwise_single_term_mapping(cell)
    max_local_dofs = max(max_local_dofs, cell.local_dof_count)
  end

  return _CellwiseDGTransferPlan{D,T,typeof(layout),typeof(cells),typeof(old_fields),
                                 typeof(new_fields),typeof(transition)}(layout, cells, old_fields,
                                                                        new_fields, transition,
                                                                        max_local_dofs)
end

function _compile_local_projection_transfer_plan(transition::SpaceTransition{D,T},
                                                 old_fields::Tuple,
                                                 new_fields::Tuple) where {D,T<:AbstractFloat}
  layout = FieldLayout(new_fields)
  overrides = _cell_quadrature_overrides(layout, ())
  cells = _compile_cells(layout, overrides)
  max_local_dofs = 0

  for cell in cells
    max_local_dofs = max(max_local_dofs, cell.local_dof_count)
  end

  return _LocalProjectionTransferPlan{D,T,typeof(layout),typeof(cells),typeof(old_fields),
                                      typeof(new_fields),typeof(transition)}(layout, cells,
                                                                             old_fields, new_fields,
                                                                             transition,
                                                                             max_local_dofs)
end

function _assemble_cellwise_transfer_matrix!(local_matrix, cell::CellValues, new_fields::Tuple)
  for field in new_fields
    _assemble_transfer_mass!(local_matrix, cell, field)
  end

  return local_matrix
end

# Assemble all target-field RHS blocks for one DG cell. The source leaf lookup
# is shared across field components at a quadrature point; the per-field kernel
# is the same one used by the affine transfer fallback.
function _assemble_cellwise_transfer_rhs!(local_rhs, cell::CellValues,
                                          plan::Union{_CellwiseDGTransferPlan{D,T},
                                                      _LocalProjectionTransferPlan{D,T}},
                                          source_coefficients, scratch) where {D,T<:AbstractFloat}
  return _assemble_transfer_rhs!(local_rhs, cell, plan.transition, plan.new_fields,
                                 source_coefficients, scratch.source_basis, scratch.target_basis,
                                 scratch.reference_points, scratch.reference_weights)
end

# Solve one independent dense projection system per target DG cell and scatter
# the local coefficients directly into the new state vector.
function _transfer_cellwise_dg_state(plan::_CellwiseDGTransferPlan{D,T}, state::State{T};
                                     linear_solve=nothing) where {D,T<:AbstractFloat}
  state_coefficients = zeros(T, dof_count(plan.layout))
  isempty(plan.cells) && return State(plan.layout, state_coefficients)
  scratch = _CellwiseDGTransferScratch(T, Val(D), plan.max_local_dofs)
  source_coefficients = ntuple(index -> _component_coefficient_views(state, plan.old_fields[index]),
                               length(plan.old_fields))

  for cell in plan.cells
    local_dofs = cell.local_dof_count
    matrix_view = view(scratch.matrix, 1:local_dofs, 1:local_dofs)
    rhs_view = view(scratch.rhs, 1:local_dofs)
    fill!(matrix_view, zero(T))
    fill!(rhs_view, zero(T))
    _assemble_cellwise_transfer_matrix!(matrix_view, cell, plan.new_fields)
    _assemble_cellwise_transfer_rhs!(rhs_view, cell, plan, source_coefficients, scratch)
    local_solution = _cellwise_transfer_linear_solve(T, matrix_view, rhs_view, linear_solve)

    for local_dof in 1:local_dofs
      state_coefficients[cell.single_term_indices[local_dof]] = local_solution[local_dof]
    end
  end

  return State(plan.layout, state_coefficients)
end

function _add_transfer_coupling!(couplings::Dict{Tuple{Int,Int},T}, adjacency, first_dof::Int,
                                 second_dof::Int, value::T) where {T<:AbstractFloat}
  first_dof == second_dof && return couplings
  row, col = first_dof < second_dof ? (first_dof, second_dof) : (second_dof, first_dof)
  key = (row, col)

  if !haskey(couplings, key)
    adjacency[row] === nothing && (adjacency[row] = Int[])
    adjacency[col] === nothing && (adjacency[col] = Int[])
    push!(adjacency[row], col)
    push!(adjacency[col], row)
    couplings[key] = value
  else
    couplings[key] += value
  end

  return couplings
end

function _accumulate_transfer_normal_equation!(diagonal::AbstractVector{T}, rhs::AbstractVector{T},
                                               couplings, adjacency, item::_AssemblyValues,
                                               local_solution::AbstractVector{T}) where {T<:AbstractFloat}
  # A local projected coefficient may map to several reduced/global dofs through
  # continuity or hanging-node substitutions. Accumulating the normal equation
  # of that small local relation reconciles all local projections that share a
  # target dof without assembling a full global mass matrix.
  for local_dof in 1:item.local_dof_count
    value = local_solution[local_dof]
    first_term = item.term_offsets[local_dof]
    last_term = item.term_offsets[local_dof+1] - 1

    for row_term in first_term:last_term
      row = item.term_indices[row_term]
      row_coefficient = item.term_coefficients[row_term]
      diagonal[row] += row_coefficient * row_coefficient
      rhs[row] += row_coefficient * value

      for col_term in (row_term+1):last_term
        col = item.term_indices[col_term]
        col_coefficient = item.term_coefficients[col_term]
        _add_transfer_coupling!(couplings, adjacency, row, col, row_coefficient * col_coefficient)
      end
    end
  end

  return rhs
end

function _transfer_component!(component::Vector{Int}, start::Int, adjacency, visited::BitVector)
  empty!(component)
  queue = Int[start]
  visited[start] = true
  cursor = 1

  while cursor <= length(queue)
    dof = queue[cursor]
    cursor += 1
    push!(component, dof)
    neighbors = adjacency[dof]
    neighbors === nothing && continue

    for neighbor in neighbors
      visited[neighbor] && continue
      visited[neighbor] = true
      push!(queue, neighbor)
    end
  end

  return component
end

function _solve_transfer_component!(state_coefficients::AbstractVector{T}, component::Vector{Int},
                                    diagonal::AbstractVector{T}, rhs::AbstractVector{T}, couplings,
                                    adjacency) where {T<:AbstractFloat}
  count = length(component)
  local_index = Dict{Int,Int}()

  for index in 1:count
    local_index[component[index]] = index
  end

  matrix = zeros(T, count, count)
  local_rhs = Vector{T}(undef, count)

  for index in 1:count
    dof = component[index]
    matrix[index, index] = diagonal[dof]
    local_rhs[index] = rhs[dof]
  end

  for row_global in component
    row = local_index[row_global]
    neighbors = adjacency[row_global]
    neighbors === nothing && continue

    for col_global in neighbors
      row_global < col_global || continue
      value = get(couplings, (row_global, col_global), zero(T))
      iszero(value) && continue
      col = local_index[col_global]
      matrix[row, col] = value
      matrix[col, row] = value
    end
  end

  _regularized_transfer_cholesky_solve!(matrix, local_rhs)

  for index in 1:count
    state_coefficients[component[index]] = local_rhs[index]
  end

  return state_coefficients
end

function _finish_transfer_normal_equations!(state_coefficients::AbstractVector{T},
                                            diagonal::AbstractVector{T}, rhs::AbstractVector{T},
                                            couplings, adjacency) where {T<:AbstractFloat}
  visited = falses(length(state_coefficients))
  component = Int[]
  tolerance = 1000 * eps(T)

  # Most target dofs are isolated after local projection and can be written from
  # one scalar normal equation. Only dofs coupled by shared local relations enter
  # the small connected-component solves below.
  for dof in eachindex(state_coefficients)
    visited[dof] && continue
    neighbors = adjacency[dof]

    if neighbors === nothing
      visited[dof] = true
      diagonal[dof] > tolerance && (state_coefficients[dof] = rhs[dof] / diagonal[dof])
      continue
    end

    _transfer_component!(component, dof, adjacency, visited)
    _solve_transfer_component!(state_coefficients, component, diagonal, rhs, couplings, adjacency)
  end

  return state_coefficients
end

function _transfer_local_projection_state(plan::_LocalProjectionTransferPlan{D,T},
                                          state::State{T}) where {D,T<:AbstractFloat}
  state_coefficients = zeros(T, dof_count(plan.layout))
  isempty(plan.cells) && return State(plan.layout, state_coefficients)
  diagonal = zeros(T, length(state_coefficients))
  rhs = zeros(T, length(state_coefficients))
  couplings = Dict{Tuple{Int,Int},T}()
  adjacency = Vector{Union{Nothing,Vector{Int}}}(nothing, length(state_coefficients))
  scratch = _LocalProjectionTransferScratch(T, Val(D), plan.max_local_dofs)
  source_coefficients = ntuple(index -> _component_coefficient_views(state, plan.old_fields[index]),
                               length(plan.old_fields))

  for cell in plan.cells
    local_dofs = cell.local_dof_count
    matrix_view = view(scratch.matrix, 1:local_dofs, 1:local_dofs)
    rhs_view = view(scratch.rhs, 1:local_dofs)
    fill!(matrix_view, zero(T))
    fill!(rhs_view, zero(T))
    _assemble_cellwise_transfer_matrix!(matrix_view, cell, plan.new_fields)
    _assemble_cellwise_transfer_rhs!(rhs_view, cell, plan, source_coefficients, scratch)
    _regularized_transfer_cholesky_solve!(matrix_view, rhs_view)
    _accumulate_transfer_normal_equation!(diagonal, rhs, couplings, adjacency, cell, rhs_view)
  end

  _finish_transfer_normal_equations!(state_coefficients, diagonal, rhs, couplings, adjacency)
  return State(plan.layout, state_coefficients)
end

function _transfer_variational_state(transition::SpaceTransition{D,T}, state::State{T},
                                     old_fields::Tuple, new_fields::Tuple;
                                     linear_solve) where {D,T<:AbstractFloat}
  # The variational path is kept as an expert escape hatch. It builds a normal
  # affine Grico problem for the target mass projection, so user-supplied
  # transfer solvers can reuse the same reduced operator interface as PDE
  # solves.
  problem = AffineProblem(new_fields...)

  for index in eachindex(old_fields)
    new_field = new_fields[index]
    add_cell!(problem, _TransferMass(new_field))
    source_basis = _LeafBasisScratch(T, Val(D)).values
    target_basis = _LeafBasisScratch(T, Val(D)).values
    reference_points = NTuple{D,T}[]
    reference_weights = T[]
    add_cell!(problem,
              _TransferSource(new_field, _component_coefficient_views(state, old_fields[index]),
                              transition, source_basis, target_basis, reference_points,
                              reference_weights))
  end

  plan = compile(problem)
  return disable_polyester_threads() do
    _solve_variational_transfer_plan(plan; linear_solve=linear_solve)
  end
end

function _solve_variational_transfer_plan(plan::AssemblyPlan{D,T};
                                          linear_solve) where {D,T<:AbstractFloat}
  _require_matrix_free_kind(plan, :affine)
  workspace = _ReducedOperatorWorkspace(plan)
  reduced_rhs = zeros(T, reduced_dof_count(plan))
  _reduced_rhs!(reduced_rhs, plan, workspace)
  reduced_values = linear_solve(plan, reduced_rhs; workspace=workspace)
  _require_exact_vector(reduced_values, reduced_dof_count(plan), "linear_solve result")
  eltype(reduced_values) == T ||
    throw(ArgumentError("linear_solve result element type must match the plan scalar type"))
  _require_finite_transfer_solution(reduced_values, "linear_solve result")
  return _state_from_reduced_result(plan, reduced_values)
end

function _transfer_state_with_strategy(::_CellwiseDGTransferStrategy, transition::SpaceTransition,
                                       state::State, old_fields::Tuple, new_fields::Tuple;
                                       linear_solve=nothing)
  plan = _compile_cellwise_dg_transfer_plan(transition, old_fields, new_fields)
  return _transfer_cellwise_dg_state(plan, state; linear_solve=linear_solve)
end

function _transfer_state_with_strategy(::_LocalProjectionTransferStrategy,
                                       transition::SpaceTransition, state::State, old_fields::Tuple,
                                       new_fields::Tuple; linear_solve=nothing)
  plan = _compile_local_projection_transfer_plan(transition, old_fields, new_fields)
  return _transfer_local_projection_state(plan, state)
end

function _transfer_state_with_strategy(::_VariationalTransferStrategy, transition::SpaceTransition,
                                       state::State, old_fields::Tuple, new_fields::Tuple;
                                       linear_solve)
  return _transfer_variational_state(transition, state, old_fields, new_fields; linear_solve)
end

"""
    transfer_state(transition, state, old_fields, new_fields; linear_solve=nothing)
    transfer_state(transition, state, old_field, new_field; linear_solve=nothing)
    transfer_state(transition, state; linear_solve=nothing)

Transfer field coefficients from the source space to the target space of
`transition` by cellwise `L²` projection.

The first forms project explicitly paired old/new fields. The zero-argument
field form recreates the full field layout on the target space via
[`adapted_fields`](@ref) and returns both the new fields and the transferred
state.

The transfer is purely geometric: it depends on the old state and the
source/target spaces, but not on any specific PDE operator. Fully discontinuous
target spaces are projected by independent dense cell solves. For the default
coupled-space path, Grico computes the same cellwise target projection and then
reconciles shared coefficients by the small normal-equation components induced
by continuity and hanging-node substitutions. This keeps transfer bounded and
separate from the PDE linear-solve path. Passing a custom `linear_solve` is an
advanced transfer-only hook: fully DG targets pass dense local projection
systems to the hook, while CG or mixed targets pass the internal variational
projection system.
"""
function transfer_state(transition::SpaceTransition, state::State, old_fields::Tuple,
                        new_fields::Tuple; linear_solve=nothing)
  _checked_transfer_fields(transition, state, old_fields, new_fields)
  strategy = _transfer_strategy(transition, linear_solve)
  return _transfer_state_with_strategy(strategy, transition, state, old_fields, new_fields;
                                       linear_solve=linear_solve)
end

function transfer_state(transition::SpaceTransition, state::State, old_field::AbstractField,
                        new_field::AbstractField; linear_solve=nothing)
  return transfer_state(transition, state, (old_field,), (new_field,); linear_solve=linear_solve)
end

function transfer_state(transition::SpaceTransition, state::State; linear_solve=nothing)
  old_fields = fields(field_layout(state))
  new_fields = adapted_fields(transition, old_fields)
  return new_fields,
         transfer_state(transition, state, old_fields, new_fields; linear_solve=linear_solve)
end

# Mixed-field transfer requires exactly one plan for each source space present
# in the state layout, and all plans must describe the same target active-leaf
# topology. These checks keep the later block transfer deterministic.
function _checked_transition_plans(plans::Tuple, state::State)
  isempty(plans) && throw(ArgumentError("at least one adaptivity plan is required"))
  layout = field_layout(state)
  layout_fields = fields(layout)
  layout_spaces = IdDict{Any,Bool}()

  for field in layout_fields
    layout_spaces[field_space(field)] = true
  end

  plan_by_space = IdDict{Any,AdaptivityPlan}()
  reference_target = nothing
  reference_topology = nothing

  for plan in plans
    plan isa AdaptivityPlan || throw(ArgumentError("plans must be AdaptivityPlan values"))
    source = source_space(plan)
    !haskey(layout_spaces, source) &&
      throw(ArgumentError("every adaptivity plan must match a source space in the state layout"))
    haskey(plan_by_space, source) &&
      throw(ArgumentError("adaptivity plans must use distinct source spaces"))
    plan_by_space[source] = plan

    if isnothing(reference_target)
      reference_target = target_domain(plan)
      reference_topology = _active_leaf_signatures(target_snapshot(plan))
      continue
    end

    _same_adaptivity_geometry(reference_target, target_domain(plan)) ||
      throw(ArgumentError("adaptivity plan targets must share one physical domain and periodic topology"))
    reference_topology == _active_leaf_signatures(target_snapshot(plan)) ||
      throw(ArgumentError("adaptivity plan targets must share the same active-leaf topology"))
  end

  length(plan_by_space) == length(layout_spaces) ||
    throw(ArgumentError("the state layout requires exactly one adaptivity plan per source space"))

  return plan_by_space
end

"""
    transfer_state(plans, state; linear_solve=nothing)

Transfer a mixed-field [`State`](@ref) across one [`AdaptivityPlan`](@ref) per
source space.

This is the layout-level companion to the single-space
`transfer_state(transition, state)` workflow. Each plan is compiled into its own
[`SpaceTransition`](@ref), fields are transferred in groups that share one
source space, and the resulting blocks are stitched back together in the
original layout order. The plans must therefore use distinct source spaces and
describe one common target active-leaf topology so the transferred fields can
again be combined into one [`FieldLayout`](@ref).
"""
function transfer_state(plans::Tuple, state::State; linear_solve=nothing)
  old_fields = fields(field_layout(state))
  plan_by_space = _checked_transition_plans(plans, state)
  space_to_group = IdDict{Any,Int}()
  group_spaces = Any[]
  group_indices = Vector{Vector{Int}}()

  for index in eachindex(old_fields)
    space = field_space(old_fields[index])

    if haskey(space_to_group, space)
      push!(group_indices[space_to_group[space]], index)
    else
      push!(group_spaces, space)
      push!(group_indices, Int[index])
      space_to_group[space] = length(group_spaces)
    end
  end

  new_fields = Vector{AbstractField}(undef, length(old_fields))
  transferred_groups = Vector{Any}(undef, length(group_spaces))
  group_new_fields = Vector{Any}(undef, length(group_spaces))

  for group_index in eachindex(group_spaces)
    indices = group_indices[group_index]
    old_group_fields = ntuple(local_index -> old_fields[indices[local_index]], length(indices))
    group_transition = transition(plan_by_space[group_spaces[group_index]])
    new_group_fields = adapted_fields(group_transition, old_group_fields)
    group_state = transfer_state(group_transition, state, old_group_fields, new_group_fields;
                                 linear_solve=linear_solve)

    for local_index in eachindex(indices)
      new_fields[indices[local_index]] = new_group_fields[local_index]
    end

    group_new_fields[group_index] = new_group_fields
    transferred_groups[group_index] = group_state
  end

  new_layout = FieldLayout(new_fields)
  new_state = State(new_layout)

  for group_index in eachindex(group_spaces)
    group_state = transferred_groups[group_index]

    for field in group_new_fields[group_index]
      field_values(new_state, field) .= field_values(group_state, field)
    end
  end

  return Tuple(new_fields), new_state
end

# This file compiles the geometric bridge between an existing source `HpSpace`
# and a target `HpSpace` described by an `AdaptivityPlan`. The transition object
# is not itself a state-transfer algorithm. Instead, it records the overlap
# relation that transfer, adapted-field recreation, and mixed-space adaptation
# need in order to interpret the same physical domain on two different active
# leaf frontiers.
#
# The important invariant is geometric, not algebraic: every active target leaf
# stores the contiguous slice of active source leaves whose physical cells
# overlap it. Later transfer routines may integrate on that overlap, but this
# file does not evaluate basis functions or solve projection systems.
#
# The implementation is intentionally compact:
# - validate and expose the compressed source-leaf row table,
# - collect source leaves by descending only overlapping parts of the source
#   refinement tree,
# - build derived adaptivity plans for companion spaces in mixed-field problems,
# - compile the final `SpaceTransition` and recreate fields on the target space.

"""
    SpaceTransition

Compiled overlap information between a source space and an adapted target space.

For each active target leaf, the transition stores the contiguous slice of
source leaves whose physical cells overlap that target leaf. This overlap data
is the geometric backbone for state transfer: values on the target space are
evaluated by locating the source leaf that contains each target quadrature
point. The row table is indexed by target active-leaf order, not by raw cell id,
so its size scales with the target frontier rather than append-only grid
history.
"""
struct SpaceTransition{D,T<:AbstractFloat,S<:HpSpace{D,T},N<:HpSpace{D,T},O<:Vector{Int},
                       C<:Vector{Int},V<:Vector{Int}}
  source_space::S
  target_space::N
  source_offsets::O
  source_counts::C
  source_leaf_data::V

  function SpaceTransition(source_space::S, target_space::N, source_offsets::O, source_counts::C,
                           source_leaf_data::V) where {D,T<:AbstractFloat,S<:HpSpace{D,T},
                                                       N<:HpSpace{D,T},O<:Vector{Int},
                                                       C<:Vector{Int},V<:Vector{Int}}
    _check_space_transition_data(source_space, target_space, source_offsets, source_counts,
                                 source_leaf_data)
    return new{D,T,S,N,O,C,V}(source_space, target_space, source_offsets, source_counts,
                              source_leaf_data)
  end
end

function _check_space_transition_data(source_space::HpSpace{D,T}, target_space::HpSpace{D,T},
                                      source_offsets::Vector{Int}, source_counts::Vector{Int},
                                      source_leaf_data::Vector{Int}) where {D,T<:AbstractFloat}
  source_snapshot = snapshot(source_space)
  target_snapshot = snapshot(target_space)
  target_active = target_snapshot.active_leaves
  length(source_offsets) == length(target_active) ||
    throw(ArgumentError("transition source offsets must match the active target leaf count"))
  length(source_counts) == length(target_active) ||
    throw(ArgumentError("transition source counts must match the active target leaf count"))

  cursor = 1

  for target_index in eachindex(target_active)
    first = source_offsets[target_index]
    count = source_counts[target_index]
    count > 0 || throw(ArgumentError("transition target row $target_index has no source leaves"))
    first == cursor || throw(ArgumentError("transition source rows must be stored contiguously"))
    last = first + count - 1
    1 <= first <= last <= length(source_leaf_data) ||
      throw(ArgumentError("transition source row $target_index exceeds source leaf data"))
    target_leaf = target_active[target_index]

    for source_index in first:last
      source_leaf = source_leaf_data[source_index]
      1 <= source_leaf <= stored_cell_count(grid(source_space)) ||
        throw(ArgumentError("transition source leaf $source_leaf is outside the source grid"))
      is_active_leaf(source_snapshot, source_leaf) ||
        throw(ArgumentError("transition source leaf $source_leaf is not active in the source space"))
      _cells_overlap(grid(source_snapshot), source_leaf, grid(target_snapshot), target_leaf) ||
        throw(ArgumentError("transition source leaf $source_leaf does not overlap target leaf $target_leaf"))
    end

    cursor = last + 1
  end

  cursor == length(source_leaf_data) + 1 ||
    throw(ArgumentError("transition source leaf data contains unused entries"))
  return nothing
end

"""
    source_space(transition)

Return the source `HpSpace` linked by `transition`.
"""
source_space(transition::SpaceTransition) = transition.source_space

"""
    target_space(transition)

Return the target `HpSpace` linked by `transition`.
"""
target_space(transition::SpaceTransition) = transition.target_space

# Offsets/counts encode one contiguous source-leaf slice per active target leaf.
# This compact compressed-row style layout avoids allocating small vectors for
# every target leaf.
function _source_leaf_range(transition::SpaceTransition, target_leaf::Integer)
  target = target_space(transition)
  target_snapshot = snapshot(target)
  checked_leaf = _checked_cell(grid(target), target_leaf)
  target_index = checked_leaf <= length(target_snapshot.leaf_to_index) ?
                 @inbounds(target_snapshot.leaf_to_index[checked_leaf]) : 0
  target_index != 0 || throw(ArgumentError("leaf $checked_leaf is not an active target leaf"))
  first, count = _source_leaf_range_at_index_unchecked(transition, target_index)
  count != 0 || throw(ArgumentError("leaf $checked_leaf has no compiled source leaves"))
  return first, count
end

@inline function _source_leaf_range_unchecked(transition::SpaceTransition, target_leaf::Int)
  target_index = @inbounds snapshot(target_space(transition)).leaf_to_index[target_leaf]
  return _source_leaf_range_at_index_unchecked(transition, target_index)
end

@inline function _source_leaf_range_at_index_unchecked(transition::SpaceTransition,
                                                       target_index::Int)
  first = @inbounds transition.source_offsets[target_index]
  count = @inbounds transition.source_counts[target_index]
  return first, count
end

"""
    source_leaves(transition, target_leaf)

Return the active source leaves that overlap one active target leaf.

These leaves are stored in the traversal order used by [`transition`](@ref) and
form the local search set used during state transfer. The returned object is a
non-copying view into the transition data.
"""
function source_leaves(transition::SpaceTransition, target_leaf::Integer)
  first, count = _source_leaf_range(transition, target_leaf)
  return @view transition.source_leaf_data[first:(first+count-1)]
end

# Source/target overlap is collected by descending the source tree only through
# cells whose physical boxes overlap the target leaf. Because both grids are
# dyadic on the same root box, box overlap on logical coordinates is sufficient
# to identify potentially contributing source leaves.
function _collect_source_leaves!(leaves::Vector{Int}, source_snapshot::GridSnapshot{D},
                                 target_grid::CartesianGrid{D}, source_cell::Int,
                                 target_leaf::Int) where {D}
  source_grid = grid(source_snapshot)
  _cells_overlap(source_grid, source_cell, target_grid, target_leaf) || return leaves

  if is_active_leaf(source_snapshot, source_cell)
    push!(leaves, source_cell)
    return leaves
  end

  first = _snapshot_first_child(source_snapshot, source_cell)
  first == NONE && return leaves

  for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
    is_tree_cell(source_snapshot, child) || continue
    _collect_source_leaves!(leaves, source_snapshot, target_grid, child, target_leaf)
  end

  return leaves
end

# A target leaf overlaps only source leaves inside the source subtree rooted at
# the corresponding root cell of the target leaf. Starting the search there
# avoids scanning unrelated roots in multi-root grids.
function _transition_source_leaves!(leaves::Vector{Int}, source_snapshot::GridSnapshot,
                                    target_grid::CartesianGrid, target_leaf::Int)
  empty!(leaves)
  root = target_leaf

  while true
    parent_cell = parent(target_grid, root)
    parent_cell == NONE && break
    root = parent_cell
  end

  _collect_source_leaves!(leaves, source_snapshot, target_grid, root, target_leaf)
  isempty(leaves) && throw(ArgumentError("leaf $target_leaf has no source leaves"))
  return leaves
end

# A derived plan may use a different source space, but it must live on the same
# physical geometry and scalar type as the driver plan. Otherwise the target
# snapshot could not be interpreted as the same adapted domain.
function _checked_derived_plan_space(driver_plan::AdaptivityPlan, space::HpSpace)
  dimension(space) == dimension(driver_plan) ||
    throw(ArgumentError("derived space must have dimension $(dimension(driver_plan))"))
  eltype(origin(domain(space))) == eltype(origin(domain(source_space(driver_plan)))) ||
    throw(ArgumentError("derived space must use the same scalar type as the driver plan"))
  _same_adaptivity_geometry(domain(source_space(driver_plan)), domain(space)) ||
    throw(ArgumentError("derived space must share the driver-plan geometry and root cell counts"))
  return space
end

# Lift one target topology onto a companion space by inheriting the maximal
# degree over every overlapping source leaf. This reproduces refinement
# inheritance and coarsening-by-maximum without assuming leaf-number identity.
function _inherited_target_degrees(space::HpSpace{D}, target_snapshot::GridSnapshot{D},
                                   active::AbstractVector{<:Integer}=target_snapshot.active_leaves) where {D}
  source_snapshot = snapshot(space)
  if grid(target_snapshot) === grid(source_snapshot) && active == source_snapshot.active_leaves
    return copy(space.degree_policy.data)
  end

  target_grid = grid(target_snapshot)
  degrees = Vector{NTuple{D,Int}}(undef, length(active))
  source_leaves = Int[]

  for index in eachindex(active)
    _transition_source_leaves!(source_leaves, source_snapshot, target_grid, active[index])
    degrees[index] = ntuple(axis -> maximum(cell_degrees(space, leaf)[axis]
                                            for leaf in source_leaves), D)
  end

  return degrees
end

# Degree offsets express compact mixed-space relationships such as pressure
# following velocity with one lower order. The offset is applied to the target
# degrees of the already adapted driver plan, not to the original source space.
function _offset_target_degrees(driver_plan::AdaptivityPlan{D},
                                degree_offset::NTuple{D,Int}) where {D}
  active = active_leaves(driver_plan)
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    driver_degrees = cell_degrees(driver_plan, active[index])
    degrees[index] = ntuple(axis -> driver_degrees[axis] + degree_offset[axis], D)
  end

  return degrees
end

"""
    derived_adaptivity_plan(driver_plan, space; degree_offset=nothing,
                            limits=AdaptivityLimits(space))
    derived_adaptivity_plan(driver_plan, field; degree_offset=nothing,
                            limits=AdaptivityLimits(field_space(field)))

Build a companion [`AdaptivityPlan`](@ref) for another space on the target mesh
topology of `driver_plan`.

This is the mixed-space adaptivity bridge for coupled problems whose fields live
on different `HpSpace`s but must still share one target active-leaf topology.
The returned value is an ordinary single-space `AdaptivityPlan`, so all existing
queries such as [`transition`](@ref), [`h_adaptation_axes`](@ref), and
[`adaptivity_summary`](@ref) continue to work unchanged.

If `degree_offset` is left at `nothing`, the companion plan inherits degrees
from its own source space: every target leaf receives the componentwise maximum
degree over the overlapping source leaves of `space`. This is the natural rule
for synchronized pure-`h` adaptation.

If `degree_offset` is provided, the companion target degrees are taken from the
target leaves of `driver_plan` and shifted componentwise by that offset before
the usual `AdaptivityLimits` and continuity checks are applied. This is the
compact way to express relationships such as pressure = velocity - 1.
"""
function derived_adaptivity_plan(driver_plan::AdaptivityPlan{D}, space::HpSpace{D};
                                 degree_offset=nothing, limits=AdaptivityLimits(space)) where {D}
  checked_space = _checked_derived_plan_space(driver_plan, space)
  target = target_domain(driver_plan)
  target_topology = target_snapshot(driver_plan)
  degrees = isnothing(degree_offset) ? _inherited_target_degrees(checked_space, target_topology) :
            _offset_target_degrees(driver_plan, _degree_offset_tuple(degree_offset, D))
  return AdaptivityPlan(checked_space, target, target_topology, degrees; limits=limits)
end

function derived_adaptivity_plan(driver_plan::AdaptivityPlan, field::AbstractField;
                                 degree_offset=nothing, limits=AdaptivityLimits(field_space(field)))
  return derived_adaptivity_plan(driver_plan, field_space(field); degree_offset=degree_offset,
                                 limits=limits)
end

function _copied_transition_target_data(plan::AdaptivityPlan)
  # By default the transition compiles on a non-mutating copy of the target
  # domain. This keeps an already compiled transition independent from later
  # interactive edits to the mutable adaptivity plan.
  copied_domain = copy(target_domain(plan))
  copied_snapshot = _snapshot(grid(copied_domain), target_snapshot(plan).active_leaves)
  return copied_domain, copied_snapshot, copy(plan.target_degrees)
end

function _transition_target_data(plan::AdaptivityPlan, use_compact::Bool)
  use_compact || return _copied_transition_target_data(plan)
  # Compacting is optional because it renumbers target cells. When requested,
  # it removes dead grid storage while preserving the active target frontier and
  # remapping the per-leaf degree table to the compacted cell ids.
  compact_domain, compact_snapshot, old_to_new = compact(target_domain(plan), target_snapshot(plan))
  compact_degrees = _remap_compacted_target_degrees(target_snapshot(plan), plan.target_degrees,
                                                    compact_snapshot, old_to_new)
  return compact_domain, compact_snapshot, compact_degrees
end

function _remap_compacted_target_degrees(source_snapshot::GridSnapshot{D},
                                         source_degrees::AbstractVector{NTuple{D,Int}},
                                         compact_snapshot::GridSnapshot{D},
                                         old_to_new::AbstractVector{<:Integer}) where {D}
  length(source_degrees) == length(source_snapshot.active_leaves) ||
    throw(ArgumentError("target degree data must match the active-leaf count"))
  remapped = Vector{NTuple{D,Int}}(undef, length(compact_snapshot.active_leaves))

  for (source_index, old_leaf) in enumerate(source_snapshot.active_leaves)
    new_leaf = @inbounds old_to_new[old_leaf]
    new_leaf != NONE || throw(ArgumentError("compaction dropped active leaf $old_leaf"))
    new_index = @inbounds compact_snapshot.leaf_to_index[new_leaf]
    new_index != 0 ||
      throw(ArgumentError("compacted snapshot is missing active leaf mapped from $old_leaf"))
    remapped[new_index] = source_degrees[source_index]
  end

  return remapped
end

"""
    transition(plan; compact=false)

Build the `SpaceTransition` from the source space of `plan` to its target
space.

This compiles the current target snapshot and target degree data of `plan` into
a new `HpSpace`, then records for every active target leaf which active source
leaves overlap it. The resulting `SpaceTransition` is the object required for
state transfer and for recreating field layouts on the target space.

If `compact=true`, the target space is compiled on a non-mutating compacted copy
of the target grid. The source space and the plan snapshot remain valid while
the transition uses the compacted target for all new-space data.

By default, the target space is still compiled on a non-mutating copy of the
plan target domain, but retains the plan's full target-grid storage. This keeps
compiled transitions independent of later edits to the mutable plan.

Because the source and target domains share the same physical box and root-grid
layout, the transition can describe source-to-target overlap purely by leaf
relations on dyadic grids.
"""
function transition(plan::AdaptivityPlan{D,T}; compact::Bool=false) where {D,T<:AbstractFloat}
  old_space = source_space(plan)
  new_domain, new_snapshot, target_degrees = _transition_target_data(plan, compact)
  degree_policy = StoredDegrees(new_domain, new_snapshot.active_leaves, target_degrees)
  options = SpaceOptions(basis=basis_family(old_space), degree=degree_policy,
                         quadrature=old_space.quadrature_policy,
                         continuity=continuity_policy(old_space))
  new_space = _compile_snapshot_space(new_domain, new_snapshot, options)
  source_snapshot = snapshot(old_space)
  target_grid = grid(new_space)
  target_active = snapshot(new_space).active_leaves
  source_offsets = zeros(Int, length(target_active))
  source_counts = zeros(Int, length(target_active))
  source_data = Int[]
  source_leaves = Int[]

  # The row table is built in active target-leaf order. Each row is a contiguous
  # slice of `source_data`, which keeps `source_leaves(transition, leaf)` cheap
  # and allocation-free.
  for target_index in eachindex(target_active)
    leaf = target_active[target_index]
    leaves = _transition_source_leaves!(source_leaves, source_snapshot, target_grid, leaf)
    source_offsets[target_index] = length(source_data) + 1
    source_counts[target_index] = length(leaves)
    append!(source_data, leaves)
  end

  return SpaceTransition(old_space, new_space, source_offsets, source_counts, source_data)
end

"""
    adapted_field(transition, field; name=field_name(field))

Recreate `field` on the target space of `transition`.

The field kind, component count, and semantic field identity are preserved; only
the owning space and, optionally, the symbolic field name change. This means an
adapted field represents the same unknown on a new discrete space, so layout and
state lookup by the original field descriptor remain valid after transfer.
"""
function adapted_field(transition::SpaceTransition, field::ScalarField;
                       name::Symbol=field_name(field))
  field_space(field) === source_space(transition) ||
    throw(ArgumentError("field must belong to the transition source space"))
  return _field_with_identity(field, target_space(transition), name)
end

function adapted_field(transition::SpaceTransition, field::VectorField;
                       name::Symbol=field_name(field))
  field_space(field) === source_space(transition) ||
    throw(ArgumentError("field must belong to the transition source space"))
  return _field_with_identity(field, target_space(transition), name)
end

"""
    adapted_fields(transition, fields...)
    adapted_fields(transition, fields::Tuple)
    adapted_fields(transition, layout)

Recreate one or more fields on the target space of `transition`.

Each returned field follows [`adapted_field`](@ref): it lives on the transition
target space but keeps the semantic identity of the corresponding source field.
"""
function adapted_fields(transition::SpaceTransition, fields::Tuple)
  all(field -> field isa AbstractField, fields) ||
    throw(ArgumentError("fields must contain only field descriptors"))
  return ntuple(index -> adapted_field(transition, fields[index]), length(fields))
end

function adapted_fields(transition::SpaceTransition, layout::FieldLayout)
  adapted_fields(transition, fields(layout))
end

function adapted_fields(transition::SpaceTransition, first_field::AbstractField,
                        remaining_fields::AbstractField...)
  return adapted_fields(transition, (first_field, remaining_fields...))
end

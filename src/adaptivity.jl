# This file implements the adaptive mesh/polynomial workflow around `HpSpace`.
# The main responsibilities are:
# 1. describe admissible target discretizations via `AdaptivityPlan`,
# 2. compile source-to-target transfer data via `SpaceTransition`,
# 3. derive problem-independent indicators from modal coefficients, and
# 4. translate indicator fields into concrete `h`, `p`, or mixed `hp` changes.
#
# The guiding design choice is to keep "planning a new space" separate from
# "building that space". An `AdaptivityPlan` therefore stores only a mutable
# target domain and target per-leaf degree data, while the actual target
# `HpSpace` is built later by [`transition`](@ref). This keeps interactive plan
# editing cheap and postpones the more expensive continuity compilation to the
# point where the target space is really needed.
#
# Conceptually, the file contains four connected chapters.
#
# First, it defines the editable target description: admissibility limits,
# mutable plans, and explicit `h`/`p` edit operations.
#
# Second, it defines the compiled bridge from source to target via
# `SpaceTransition`, together with state transfer and field recreation on the
# target space.
#
# Third, it introduces problem-independent indicators derived directly from the
# modal structure of the hp basis. These indicators are intentionally generic:
# they say something about local resolution and smoothness without assuming a
# specific PDE residual or estimator. In the DG setting, that story broadens:
# interior modal decay still matters, but jumps across interfaces also become a
# natural signal of under-resolution.
#
# Fourth, it turns those indicators into concrete `h`, `p`, or mixed `hp`
# changes through bulk marking, thresholding, and candidate selection.
#
# The file is arranged in that same order, so it can be read from manual plan
# editing toward fully automatic adaptivity.

# Adaptivity limits and geometric/topological helper utilities.

# Normalize scalar-or-tuple adaptivity bounds into a per-axis tuple with
# validated entries. The adaptivity API accepts both isotropic limits such as
# `max_p=6` and anisotropic limits such as `max_p=(6, 4)`. Converting both forms
# early keeps the rest of the file free of ad hoc dispatch on user input.
function _adaptivity_limit_tuple(value, D::Int, name::AbstractString, check)
  if value isa Integer
    checked = check(value, name)
    return ntuple(_ -> checked, D)
  end

  value isa Tuple || throw(ArgumentError("$name must be an integer or NTuple{$D,Int}"))
  length(value) == D || throw(ArgumentError("$name must have length $D"))

  return ntuple(axis -> begin
                  component = value[axis]
                  component isa Integer || throw(ArgumentError("$name[$axis] must be an integer"))
                  check(component, "$name[$axis]")
                end, D)
end

# Degree offsets for derived plans follow the same scalar-or-tuple convention as
# adaptivity limits, but offsets may be negative because companion spaces such
# as pressure often track a driver space with a lower polynomial order.
function _degree_offset_tuple(value, D::Int)
  if value isa Integer
    return ntuple(_ -> Int(value), D)
  end

  value isa Tuple || throw(ArgumentError("degree_offset must be an integer or NTuple{$D,Int}"))
  length(value) == D || throw(ArgumentError("degree_offset must have length $D"))

  return ntuple(axis -> begin
                  component = value[axis]
                  component isa Integer ||
                    throw(ArgumentError("degree_offset[$axis] must be an integer"))
                  Int(component)
                end, D)
end

"""
    AdaptivityLimits(dimension; min_h_level=0, max_h_level=typemax(Int),
                     min_p=1, max_p=typemax(Int))

Per-axis admissibility bounds for adaptive `h`- and `p`-changes.

`min_h_level` and `max_h_level` bound the dyadic refinement level on each
logical axis, while `min_p` and `max_p` bound the tensor-product polynomial
degree stored on each active leaf. These limits are interpreted componentwise,
so anisotropic adaptation can be restricted differently on different axes.

The constructor accepts either one scalar, which is broadcast to every axis, or
an `NTuple` with one entry per dimension. The resulting value is used both when
editing an `AdaptivityPlan` manually and when constructing plans from
indicators.
"""
struct AdaptivityLimits{D}
  min_h_level::NTuple{D,Int}
  max_h_level::NTuple{D,Int}
  min_p::NTuple{D,Int}
  max_p::NTuple{D,Int}

  function AdaptivityLimits{D}(min_h_level::NTuple{D,Int}, max_h_level::NTuple{D,Int},
                               min_p::NTuple{D,Int}, max_p::NTuple{D,Int}) where {D}
    D >= 1 || throw(ArgumentError("dimension must be positive"))
    checked_min_h = ntuple(axis -> _checked_nonnegative(min_h_level[axis], "min_h_level[$axis]"), D)
    checked_max_h = ntuple(axis -> _checked_nonnegative(max_h_level[axis], "max_h_level[$axis]"), D)
    checked_min_p = ntuple(axis -> _checked_nonnegative(min_p[axis], "min_p[$axis]"), D)
    checked_max_p = ntuple(axis -> _checked_nonnegative(max_p[axis], "max_p[$axis]"), D)

    for axis in 1:D
      checked_min_h[axis] <= checked_max_h[axis] ||
        throw(ArgumentError("min_h_level[$axis] must not exceed max_h_level[$axis]"))
      checked_min_p[axis] <= checked_max_p[axis] ||
        throw(ArgumentError("min_p[$axis] must not exceed max_p[$axis]"))
    end

    return new{D}(checked_min_h, checked_max_h, checked_min_p, checked_max_p)
  end
end

function AdaptivityLimits(dimension::Integer; min_h_level=0, max_h_level=typemax(Int), min_p=1,
                          max_p=typemax(Int))
  D = _checked_positive(dimension, "dimension")
  return AdaptivityLimits{D}(_adaptivity_limit_tuple(min_h_level, D, "min_h_level",
                                                     _checked_nonnegative),
                             _adaptivity_limit_tuple(max_h_level, D, "max_h_level",
                                                     _checked_nonnegative),
                             _adaptivity_limit_tuple(min_p, D, "min_p", _checked_nonnegative),
                             _adaptivity_limit_tuple(max_p, D, "max_p", _checked_nonnegative))
end

@inline function _default_min_p(continuity_policy::_AxisContinuity{D}) where {D}
  return ntuple(axis -> _is_cg_axis(continuity_policy, axis) ? 1 : 0, D)
end

"""
    AdaptivityLimits(space; min_h_level=0, max_h_level=typemax(Int), min_p=nothing,
                     max_p=typemax(Int))

Build continuity-aware adaptivity limits for one existing `HpSpace`.

If `min_p` is left at `nothing`, the default lower polynomial bound is chosen
axiswise from the compiled continuity policy of `space`: `1` on CG axes and `0`
on DG axes. This matches the admissible degree range of the current space
compiler and is therefore the safest default for automatic planning.
"""
function AdaptivityLimits(space::HpSpace{D}; min_h_level=0, max_h_level=typemax(Int), min_p=nothing,
                          max_p=typemax(Int)) where {D}
  effective_min_p = isnothing(min_p) ? _default_min_p(space.continuity_policy) : min_p
  limits = AdaptivityLimits(dimension(space); min_h_level=min_h_level, max_h_level=max_h_level,
                            min_p=effective_min_p, max_p=max_p)
  return _checked_limits(limits, space)
end

function _checked_limits(limits::AdaptivityLimits{D}, space::HpSpace{D}) where {D}
  for axis in 1:D
    is_continuous_axis(space, axis) &&
      limits.min_p[axis] == 0 &&
      throw(ArgumentError("min_p[$axis] must be at least 1 on :cg axes"))
  end

  return limits
end

function _checked_limits(limits, space::HpSpace{D}) where {D}
  limits isa AdaptivityLimits ||
    throw(ArgumentError("limits must be an AdaptivityLimits{$D} value"))
  throw(ArgumentError("limits must have dimension $D"))
end

# Two cells from different dyadic grids overlap if and only if their logical
# intervals overlap on every axis when both are lifted to a common refinement
# level. Using logical coordinates keeps the test exact and independent of
# floating-point geometry.
function _cells_overlap(first_grid::CartesianGrid{D}, first::Int, second_grid::CartesianGrid{D},
                        second::Int) where {D}
  for axis in 1:D
    first_level = level(first_grid, first, axis)
    second_level = level(second_grid, second, axis)
    comparison_level = max(first_level, second_level)
    first_lower = Int128(logical_coordinate(first_grid, first, axis)) <<
                  (comparison_level - first_level)
    first_upper = (Int128(logical_coordinate(first_grid, first, axis)) + 1) <<
                  (comparison_level - first_level)
    second_lower = Int128(logical_coordinate(second_grid, second, axis)) <<
                   (comparison_level - second_level)
    second_upper = (Int128(logical_coordinate(second_grid, second, axis)) + 1) <<
                   (comparison_level - second_level)
    (first_lower >= second_upper || second_lower >= first_upper) && return false
  end

  return true
end

# Collect the active leaves in the subtree rooted at `cell`. This is used both
# for reporting source-to-target changes and for comparing source leaves with the
# active descendants that replace them in a plan.
function _collect_active_descendants!(leaves::Vector{Int}, grid_data::CartesianGrid, cell::Int)
  is_tree_cell(grid_data, cell) || return leaves

  if is_active_leaf(grid_data, cell)
    push!(leaves, cell)
    return leaves
  end

  first = first_child(grid_data, cell)
  first == NONE && return leaves

  for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
    _collect_active_descendants!(leaves, grid_data, child)
  end

  return leaves
end

# Dense leaf-to-index lookup makes `cell_degrees(plan, leaf)` independent of the
# current active-leaf ordering and avoids repeated dictionary allocations during
# plan editing.
function _target_leaf_lookup(grid_data::CartesianGrid, degrees::AbstractVector{<:Tuple})
  active = active_leaves(grid_data)
  length(degrees) == length(active) ||
    throw(ArgumentError("target degree data must match the active-leaf count"))
  lookup = zeros(Int, stored_cell_count(grid_data))

  for index in eachindex(active)
    lookup[active[index]] = index
  end

  return lookup
end

# Adaptivity plans may only change refinement and degree data, not the physical
# geometry or root-grid layout. This helper enforces that source and target
# domains describe the same physical box and periodic topology, so state
# transfer can be defined by overlap on a common geometric domain.
function _same_adaptivity_geometry(source_domain::Domain, target_domain::Domain)
  root_cell_counts(grid(source_domain)) == root_cell_counts(grid(target_domain)) || return false
  origin(source_domain) == origin(target_domain) || return false
  extent(source_domain) == extent(target_domain) || return false
  periodic_axes(source_domain) == periodic_axes(target_domain) || return false
  return true
end

# Every active target leaf must satisfy the user-specified `h`/`p` bounds. This
# check is applied both when a plan is constructed from raw arrays and after
# derived plan-generation routines have assembled a tentative target state.
function _check_target_leaf(grid_data::CartesianGrid{D}, leaf::Int, degrees::NTuple{D,Int},
                            limits::AdaptivityLimits{D}) where {D}
  for axis in 1:D
    level(grid_data, leaf, axis) >= limits.min_h_level[axis] ||
      throw(ArgumentError("target leaf $leaf violates min_h_level[$axis]"))
    level(grid_data, leaf, axis) <= limits.max_h_level[axis] ||
      throw(ArgumentError("target leaf $leaf violates max_h_level[$axis]"))
    degrees[axis] >= limits.min_p[axis] ||
      throw(ArgumentError("target leaf $leaf violates min_p[$axis]"))
    degrees[axis] <= limits.max_p[axis] ||
      throw(ArgumentError("target leaf $leaf violates max_p[$axis]"))
  end

  return nothing
end

# Mutable adaptivity plans and direct `h`/`p` editing operations.

"""
    AdaptivityPlan(space; limits=AdaptivityLimits(space))

Mutable target-space description for adaptive `h`/`p` refinement and
derefinement.

An `AdaptivityPlan` stores the tentative target domain and the target degree
tuple on each active target leaf, but it does not immediately build the target
`HpSpace`. This separation is important: repeated plan edits such as
`request_h_refinement!` and `request_p_refinement!` only update grid topology
and per-leaf degree metadata, while the expensive continuity compilation is
deferred to [`transition`](@ref).

The source space is immutable within the plan and serves as the reference
configuration for summaries, indicator interpretation, and state transfer. The
plan is therefore best understood as a mutable description of "what the next
space should look like" rather than as a space itself.

Unlike [`HpSpace`](@ref), an `AdaptivityPlan` is intentionally incomplete: it
stores no compiled continuity data, no quadrature policy copies beyond what can
be inherited later, and no transfer operators. Its purpose is to make target
editing cheap.
"""
mutable struct AdaptivityPlan{D,T<:AbstractFloat,S<:HpSpace{D,T},N<:Domain{D,T},
                              V<:Vector{NTuple{D,Int}},I<:Vector{Int},L<:AdaptivityLimits{D}}
  source_space::S
  target_domain::N
  target_degrees::V
  target_leaf_to_index::I
  limits::L

  function AdaptivityPlan{D,T,S,N,V,I,L}(source_space::S, target_domain::N, target_degrees::V,
                                         target_leaf_to_index::I,
                                         limits::L) where {D,T<:AbstractFloat,S<:HpSpace{D,T},
                                                           N<:Domain{D,T},V<:Vector{NTuple{D,Int}},
                                                           I<:Vector{Int},L<:AdaptivityLimits{D}}
    _same_adaptivity_geometry(domain(source_space), target_domain) ||
      throw(ArgumentError("target domain must share the source geometry and root cell counts"))
    stored_cell_count(grid(target_domain)) == length(target_leaf_to_index) ||
      throw(ArgumentError("target leaf lookup length must match the stored target-cell count"))
    active = active_leaves(grid(target_domain))
    length(target_degrees) == length(active) ||
      throw(ArgumentError("target degree data must match the active-leaf count"))

    for index in eachindex(active)
      leaf = active[index]
      target_leaf_to_index[leaf] == index ||
        throw(ArgumentError("target leaf lookup is inconsistent"))
      _check_target_leaf(grid(target_domain), leaf, target_degrees[index], limits)
    end

    return new{D,T,S,N,V,I,L}(source_space, target_domain, target_degrees, target_leaf_to_index,
                              limits)
  end
end

function AdaptivityPlan(source_space::HpSpace{D,T}, target_domain::Domain{D,T},
                        target_degrees::AbstractVector{<:NTuple{D,<:Integer}};
                        limits::AdaptivityLimits{D}=AdaptivityLimits(source_space)) where {D,
                                                                                           T<:AbstractFloat}
  checked_limits = _checked_limits(limits, source_space)
  active = active_leaves(grid(target_domain))
  length(target_degrees) == length(active) ||
    throw(ArgumentError("target degree data must match the active-leaf count"))
  checked = Vector{NTuple{D,Int}}(undef, length(target_degrees))

  for index in eachindex(target_degrees)
    checked[index] = _checked_space_degree_tuple(target_degrees[index],
                                                 source_space.continuity_policy,
                                                 "target_degrees[$index]")
  end

  lookup = _target_leaf_lookup(grid(target_domain), checked)
  for (index, leaf) in enumerate(active)
    _check_target_leaf(grid(target_domain), leaf, checked[index], checked_limits)
  end

  return AdaptivityPlan{D,T,typeof(source_space),typeof(target_domain),typeof(checked),
                        typeof(lookup),typeof(checked_limits)}(source_space, target_domain, checked,
                                                               lookup, checked_limits)
end

function AdaptivityPlan(space::HpSpace{D,T};
                        limits::AdaptivityLimits{D}=AdaptivityLimits(space)) where {D,
                                                                                    T<:AbstractFloat}
  degrees = [cell_degrees(space, leaf) for leaf in active_leaves(space)]
  return AdaptivityPlan(space, copy(domain(space)), degrees; limits=limits)
end

dimension(plan::AdaptivityPlan) = dimension(plan.source_space)

"""
    source_space(plan)

Return the immutable source `HpSpace` from which `plan` starts.
"""
source_space(plan::AdaptivityPlan) = plan.source_space

"""
    target_domain(plan)
    domain(plan)

Return the current target domain stored in `plan`.

This domain carries the tentative `h`-adapted topology. The target polynomial
degrees are stored separately in `plan.target_degrees`.
"""
target_domain(plan::AdaptivityPlan) = plan.target_domain
domain(plan::AdaptivityPlan) = plan.target_domain
grid(plan::AdaptivityPlan) = grid(plan.target_domain)
active_leaf_count(plan::AdaptivityPlan) = active_leaf_count(grid(plan))
active_leaves(plan::AdaptivityPlan) = active_leaves(grid(plan))
active_leaf(plan::AdaptivityPlan, index::Integer) = active_leaf(grid(plan), index)

# Check that `leaf` names one active target leaf in `plan`.
function _checked_target_leaf(plan::AdaptivityPlan, leaf::Integer)
  checked_leaf = _checked_cell(grid(plan), leaf)
  is_active_leaf(grid(plan), checked_leaf) ||
    throw(ArgumentError("leaf $checked_leaf is not an active target leaf"))
  return checked_leaf
end

# Degree lookup on a plan is stored as an active-leaf array plus a dense
# leaf-to-index map. This mirrors the compiled space layout and keeps repeated
# lookup/update operations `O(1)` even while the target grid topology changes.
function cell_degrees(plan::AdaptivityPlan{D}, leaf::Integer) where {D}
  return @inbounds plan.target_degrees[_plan_leaf_index(plan, leaf)]
end

# A plan is empty if it reproduces the source active leaves and degrees exactly.
# This query intentionally compares against source-space leaf numbering rather
# than against object identity of the stored domain copy.
function Base.isempty(plan::AdaptivityPlan)
  source = source_space(plan)
  source_active = active_leaves(source)
  target_active = active_leaves(plan)
  source_active == target_active || return false

  for (index, leaf) in enumerate(source_active)
    cell_degrees(source, leaf) == plan.target_degrees[index] || return false
  end

  return true
end

# Resolve a target leaf number to the packed index inside `plan.target_degrees`.
function _plan_leaf_index(plan::AdaptivityPlan, leaf::Integer)
  return @inbounds plan.target_leaf_to_index[_checked_target_leaf(plan, leaf)]
end

# Normalize per-axis `p` increments/decrements to a validated `NTuple{D,Int}`.
function _checked_degree_steps(steps::NTuple{D,<:Integer}, name::AbstractString) where {D}
  return ntuple(axis -> _checked_nonnegative(steps[axis], "$name[$axis]"), D)
end

# Update one active target leaf by positive `p` increments, subject to `max_p`.
function _apply_p_refinement!(plan::AdaptivityPlan{D}, leaf::Integer,
                              increments::NTuple{D,Int}) where {D}
  checked_leaf = _checked_target_leaf(plan, leaf)
  index = @inbounds plan.target_leaf_to_index[checked_leaf]
  degrees = @inbounds plan.target_degrees[index]
  updated = ntuple(axis -> degrees[axis] + increments[axis], D)

  for axis in 1:D
    updated[axis] <= plan.limits.max_p[axis] ||
      throw(ArgumentError("leaf $checked_leaf reached max_p[$axis]"))
  end

  plan.target_degrees[index] = updated
  return plan
end

# Update one active target leaf by positive `p` decrements, subject to `min_p`.
function _apply_p_derefinement!(plan::AdaptivityPlan{D}, leaf::Integer,
                                decrements::NTuple{D,Int}) where {D}
  checked_leaf = _checked_target_leaf(plan, leaf)
  index = @inbounds plan.target_leaf_to_index[checked_leaf]
  degrees = @inbounds plan.target_degrees[index]
  updated = ntuple(axis -> degrees[axis] - decrements[axis], D)

  for axis in 1:D
    updated[axis] >= plan.limits.min_p[axis] ||
      throw(ArgumentError("leaf $checked_leaf reached min_p[$axis]"))
  end

  plan.target_degrees[index] = updated
  return plan
end

# Materialize the target degree data as a dictionary keyed by leaf number. This
# transient representation is convenient for sequences of splits/collapses,
# because parent and child entries can be inserted and removed directly.
function _degree_map(plan::AdaptivityPlan{D}) where {D}
  active = active_leaves(plan)
  mapping = Dict{Int,NTuple{D,Int}}()

  for index in eachindex(active)
    mapping[active[index]] = plan.target_degrees[index]
  end

  return mapping
end

# Rebuild the target degree vector after topology changes. Refinement and
# derefinement mutate the target grid in place, so the active-leaf ordering and
# lookup table must be regenerated from the surviving mapping.
function _reset_plan_degrees!(plan::AdaptivityPlan{D}, mapping::Dict{Int,NTuple{D,Int}}) where {D}
  active = active_leaves(plan)
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    leaf = active[index]
    haskey(mapping, leaf) || throw(ArgumentError("missing planned degrees for target leaf $leaf"))
    degrees[index] = mapping[leaf]
  end

  plan.target_degrees = degrees
  plan.target_leaf_to_index = _target_leaf_lookup(grid(plan), degrees)
  return plan
end

# Build a leaf => degree map from the source space. Later plan-generation
# routines edit this dictionary under splits/collapses and finally convert it
# back to the packed target arrays expected by `AdaptivityPlan`.
function _source_degree_map(space::HpSpace{D}) where {D}
  mapping = Dict{Int,NTuple{D,Int}}()
  sizehint!(mapping, active_leaf_count(space))

  for leaf in active_leaves(space)
    mapping[leaf] = cell_degrees(space, leaf)
  end

  return mapping
end

# Apply planned `p` increments/decrements before `h` refinement replicates the
# resulting degree tuple to newly created children. This matches the convention
# that a refined child inherits the degree data of the leaf from which it was
# spawned unless later logic changes it again.
function _apply_p_degree_changes!(mapping::Dict{Int,NTuple{D,Int}}, space::HpSpace{D},
                                  p_degree_changes) where {D}
  length(p_degree_changes) == active_leaf_count(space) ||
    throw(ArgumentError("p degree changes must match the active-leaf count"))

  for (leaf_index, leaf) in enumerate(active_leaves(space))
    changes = p_degree_changes[leaf_index]
    any(!=(0), changes) || continue
    haskey(mapping, leaf) ||
      throw(ArgumentError("leaf $leaf is not available for planned p adaptation"))
    degrees = mapping[leaf]
    mapping[leaf] = ntuple(axis -> degrees[axis] + changes[axis], D)
  end

  return mapping
end

# Split one target leaf along the requested axes and duplicate its degree tuple
# onto all created children. Because refinement is dyadic and axis-by-axis, a
# multi-axis refinement is implemented as repeated one-axis splits.
function _apply_h_refinement!(grid_data::CartesianGrid{D}, mapping::Dict{Int,NTuple{D,Int}},
                              leaf::Int, axes::NTuple{D,Bool}) where {D}
  any(axes) || return Int[leaf]
  haskey(mapping, leaf) ||
    throw(ArgumentError("leaf $leaf is not available for planned h refinement"))
  degrees = mapping[leaf]
  delete!(mapping, leaf)
  leaves = Int[leaf]

  for axis in 1:D
    axes[axis] || continue
    next_leaves = Int[]
    sizehint!(next_leaves, 2 * length(leaves))

    for current_leaf in leaves
      first = _split_leaf!(grid_data, current_leaf, axis)
      mapping[first] = degrees
      mapping[first+1] = degrees
      push!(next_leaves, first, first + 1)
    end

    leaves = next_leaves
  end

  return leaves
end

# Construct a plan from batched `p` changes, `h` refinements, and explicit
# `h`-coarsening candidates. The order matters:
# 1. apply derefinements on the copied target grid,
# 2. edit per-leaf polynomial degrees on the surviving source leaves,
# 3. refine marked leaves and inherit those updated degrees to the children.
#
# This produces the same target configuration that one would obtain by
# performing the selected changes conceptually "at once", but avoids repeatedly
# rebuilding plan lookup structures after every individual change.
function _batched_h_adaptivity_plan(space::HpSpace{D,T}, p_degree_changes, h_refinement_axes,
                                    h_coarsening_candidates;
                                    limits::AdaptivityLimits{D}=AdaptivityLimits(space)) where {D,
                                                                                                T<:AbstractFloat}
  length(p_degree_changes) == active_leaf_count(space) ||
    throw(ArgumentError("p degree changes must match the active-leaf count"))
  length(h_refinement_axes) == active_leaf_count(space) ||
    throw(ArgumentError("h refinement axes must match the active-leaf count"))

  target_domain = copy(domain(space))
  target_grid = grid(target_domain)
  mapping = _source_degree_map(space)

  for candidate in h_coarsening_candidates
    _apply_h_derefinement!(target_grid, mapping, candidate)
  end

  _apply_p_degree_changes!(mapping, space, p_degree_changes)

  for (leaf_index, leaf) in enumerate(active_leaves(space))
    axes = h_refinement_axes[leaf_index]
    any(axes) || continue
    _apply_h_refinement!(target_grid, mapping, leaf, axes)
  end

  _finish_refinement_update!(target_grid)
  active = active_leaves(target_grid)
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    leaf = active[index]
    haskey(mapping, leaf) || throw(ArgumentError("missing planned degrees for target leaf $leaf"))
    degrees[index] = mapping[leaf]
  end

  return AdaptivityPlan(space, target_domain, degrees; limits=limits)
end

function _can_h_refine(grid_data::CartesianGrid, leaf::Int, axis::Int, limits::AdaptivityLimits)
  return level(grid_data, leaf, axis) < limits.max_h_level[axis]
end

function _can_h_refine(space::HpSpace, leaf::Int, axis::Int, limits::AdaptivityLimits)
  return _can_h_refine(grid(space), leaf, axis, limits)
end

function _can_p_refine(space::HpSpace, leaf::Int, axis::Int, limits::AdaptivityLimits)
  return cell_degrees(space, leaf)[axis] < limits.max_p[axis]
end

function _can_p_derefine(space::HpSpace, leaf::Int, axis::Int, limits::AdaptivityLimits)
  return cell_degrees(space, leaf)[axis] > limits.min_p[axis]
end

function _can_p_refine(plan::AdaptivityPlan, leaf::Int, axis::Int)
  return cell_degrees(plan, leaf)[axis] < plan.limits.max_p[axis]
end

function _can_p_derefine(plan::AdaptivityPlan, leaf::Int, axis::Int)
  return cell_degrees(plan, leaf)[axis] > plan.limits.min_p[axis]
end

# Immediate `h`-coarsening candidates on the source space.

"""
    HCoarseningCandidate(cell, axis, children, target_degrees)

Immediate `h`-coarsening candidate on one expanded parent cell of a source
space. `target_degrees` are the canonical merged degrees used after
derefinement.

The candidate describes the inverse of a single previous dyadic split. The
children must therefore be the two active leaves created by splitting `cell`
along `axis`. When the candidate is accepted, the parent becomes active again
and receives `target_degrees`, which are chosen as the componentwise maxima of
the child degrees so that no local approximation order is lost purely because
of the merge.
"""
struct HCoarseningCandidate{D}
  cell::Int
  axis::Int
  children::NTuple{_MIDPOINT_CHILD_COUNT,Int}
  target_degrees::NTuple{D,Int}

  function HCoarseningCandidate{D}(cell::Int, axis::Int,
                                   children::NTuple{_MIDPOINT_CHILD_COUNT,Int},
                                   target_degrees::NTuple{D,Int}) where {D}
    checked_cell = _checked_positive(cell, "cell")
    checked_axis = _checked_positive(axis, "axis")
    checked_children = ntuple(index -> _checked_positive(children[index], "children[$index]"),
                              _MIDPOINT_CHILD_COUNT)
    checked_degrees = ntuple(index -> _checked_nonnegative(target_degrees[index],
                                                           "target_degrees[$index]"), D)
    return new{D}(checked_cell, checked_axis, checked_children, checked_degrees)
  end
end

function HCoarseningCandidate(cell::Integer, axis::Integer,
                              children::NTuple{_MIDPOINT_CHILD_COUNT,<:Integer},
                              target_degrees::NTuple{D,<:Integer}) where {D}
  return HCoarseningCandidate{D}(Int(cell), Int(axis),
                                 ntuple(index -> Int(children[index]), _MIDPOINT_CHILD_COUNT),
                                 ntuple(index -> Int(target_degrees[index]), D))
end

function _candidate_target_degrees(space::HpSpace{D},
                                   children::NTuple{_MIDPOINT_CHILD_COUNT,Int}) where {D}
  return ntuple(axis -> maximum(cell_degrees(space, child)[axis] for child in children), D)
end

# A valid coarsening candidate must represent an immediate inverse of one split
# in the current source grid. The checks below exclude partially refined
# children, stale cell numbers, or inconsistent target-degree proposals.
function _checked_h_coarsening_candidate(space::HpSpace{D},
                                         candidate::HCoarseningCandidate{D}) where {D}
  checked_cell = _checked_cell(grid(space), candidate.cell)
  is_expanded(grid(space), checked_cell) ||
    throw(ArgumentError("candidate cell $checked_cell is not expanded"))
  checked_axis = _checked_axis(grid(space), candidate.axis)
  split_axis(grid(space), checked_cell) == checked_axis ||
    throw(ArgumentError("candidate cell $checked_cell is not split along axis $checked_axis"))
  first = first_child(grid(space), checked_cell)
  first == NONE && throw(ArgumentError("candidate cell $checked_cell has no children"))
  expected_children = ntuple(offset -> first + offset - 1, _MIDPOINT_CHILD_COUNT)
  candidate.children == expected_children ||
    throw(ArgumentError("candidate children do not match the children of cell $checked_cell"))
  all(is_active_leaf(grid(space), child) for child in expected_children) ||
    throw(ArgumentError("candidate cell $checked_cell cannot be derefined because not all children are active leaves"))
  expected_degrees = _candidate_target_degrees(space, expected_children)
  candidate.target_degrees == expected_degrees ||
    throw(ArgumentError("candidate target degrees do not match the merged child degrees"))
  return HCoarseningCandidate(checked_cell, checked_axis, expected_children, expected_degrees)
end

function _checked_h_coarsening_candidates(space::HpSpace{D}, candidates) where {D}
  checked = Vector{HCoarseningCandidate{D}}(undef, length(candidates))

  for index in eachindex(candidates)
    candidates[index] isa HCoarseningCandidate{D} ||
      throw(ArgumentError("h coarsening candidates must be HCoarseningCandidate{$D} values"))
    checked[index] = _checked_h_coarsening_candidate(space, candidates[index])
  end

  return checked
end

function _apply_h_derefinement!(grid_data::CartesianGrid{D}, mapping::Dict{Int,NTuple{D,Int}},
                                candidate::HCoarseningCandidate{D}) where {D}
  for child in candidate.children
    delete!(mapping, child)
  end

  _collapse_leaf!(grid_data, candidate.cell)
  mapping[candidate.cell] = candidate.target_degrees
  return mapping
end

"""
    h_coarsening_candidates(space; limits=AdaptivityLimits(space))

Enumerate admissible immediate `h`-coarsening candidates on `space`.

Only immediate parent cells whose two children are both active leaves are
returned. The candidates therefore correspond exactly to derefinement moves that
can be applied without first collapsing deeper descendants.
"""
function h_coarsening_candidates(space::HpSpace{D}; limits=AdaptivityLimits(space)) where {D}
  checked_limits = _checked_limits(limits, space)
  candidates = HCoarseningCandidate{D}[]

  for cell in 1:stored_cell_count(grid(space))
    is_expanded(grid(space), cell) || continue
    axis = split_axis(grid(space), cell)
    level(grid(space), cell, axis) >= checked_limits.min_h_level[axis] || continue
    first = first_child(grid(space), cell)
    first == NONE && continue
    children = ntuple(offset -> first + offset - 1, _MIDPOINT_CHILD_COUNT)
    all(is_active_leaf(grid(space), child) for child in children) || continue
    push!(candidates,
          HCoarseningCandidate(cell, axis, children, _candidate_target_degrees(space, children)))
  end

  return candidates
end

function _default_h_coarsening_candidates(space::HpSpace; limits=AdaptivityLimits(space))
  h_coarsening_candidates(space; limits=limits)
end

# Apply one possibly anisotropic `h` refinement to an active target leaf while
# rebuilding the packed degree arrays only once after all selected axes have
# been processed.
function _refine_target_leaf!(plan::AdaptivityPlan{D}, leaf::Integer,
                              axes::NTuple{D,Bool}) where {D}
  checked_leaf = _checked_target_leaf(plan, leaf)

  for axis in 1:D
    axes[axis] || continue
    _can_h_refine(grid(plan), checked_leaf, axis, plan.limits) ||
      throw(ArgumentError("leaf $checked_leaf reached max_h_level[$axis]"))
  end

  any(axes) || return Int[checked_leaf]
  degrees = cell_degrees(plan, checked_leaf)
  mapping = _degree_map(plan)
  delete!(mapping, checked_leaf)
  leaves = Int[checked_leaf]

  for axis in 1:D
    axes[axis] || continue
    next_leaves = Int[]
    sizehint!(next_leaves, 2 * length(leaves))

    for current_leaf in leaves
      first = refine!(grid(plan), current_leaf, axis)
      mapping[first] = degrees
      mapping[first+1] = degrees
      push!(next_leaves, first, first + 1)
    end

    leaves = next_leaves
  end

  _reset_plan_degrees!(plan, mapping)
  return leaves
end

"""
    request_h_refinement!(plan, leaf, axis)
    request_h_refinement!(plan, leaf, axes)

Refine an active target leaf of `plan` along one or more axes.

The refined children inherit the degree tuple of the refined target leaf. The
plan remains mutable, so subsequent calls may further edit the same target
configuration before a new `HpSpace` is compiled.

The one-axis method returns the first child created by that split. The
anisotropic method returns the tuple-expanded vector of all active children that
replace `leaf` after the requested axis-by-axis refinements have been applied.
"""
function request_h_refinement!(plan::AdaptivityPlan{D}, leaf::Integer, axis::Integer) where {D}
  checked_axis = _checked_axis(grid(plan), axis)
  axes = ntuple(current_axis -> current_axis == checked_axis, D)
  return first(_refine_target_leaf!(plan, leaf, axes))
end

function request_h_refinement!(plan::AdaptivityPlan{D}, leaf::Integer,
                               axes::NTuple{D,Bool}) where {D}
  return _refine_target_leaf!(plan, leaf, axes)
end

"""
    request_h_derefinement!(plan, cell, axis)

Coarsen one expanded target cell back to an active parent leaf.

The merged parent receives the componentwise maximum of the child degrees. This
choice preserves the highest approximation order that was present on either
child and is the natural inverse of degree inheritance under refinement.
"""
function request_h_derefinement!(plan::AdaptivityPlan, cell::Integer, axis::Integer)
  checked_cell = _checked_cell(grid(plan), cell)
  checked_axis = _checked_axis(grid(plan), axis)
  is_expanded(grid(plan), checked_cell) ||
    throw(ArgumentError("cell $checked_cell is not expanded"))
  split_axis(grid(plan), checked_cell) == checked_axis ||
    throw(ArgumentError("cell $checked_cell is not split along axis $checked_axis"))
  level(grid(plan), checked_cell, checked_axis) >= plan.limits.min_h_level[checked_axis] ||
    throw(ArgumentError("cell $checked_cell cannot be derefined below min_h_level[$checked_axis]"))
  first = first_child(grid(plan), checked_cell)
  child_degrees = ntuple(offset -> cell_degrees(plan, first + offset - 1), _MIDPOINT_CHILD_COUNT)
  merged = ntuple(current_axis -> maximum(degrees[current_axis] for degrees in child_degrees),
                  dimension(plan))
  mapping = _degree_map(plan)

  for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
    delete!(mapping, child)
  end

  derefine!(grid(plan), checked_cell)
  mapping[checked_cell] = merged
  _reset_plan_degrees!(plan, mapping)
  return plan
end

"""
    request_p_refinement!(plan, leaf, axis; increment=1)
    request_p_refinement!(plan, leaf, increments)

Increase the target polynomial degree on one active leaf.

The edit is local to the current target leaf of the plan; no neighboring leaves
are changed automatically. Any continuity constraints implied by unequal degrees
are introduced later when the target `HpSpace` is compiled.
"""
function request_p_refinement!(plan::AdaptivityPlan{D}, leaf::Integer, axis::Integer;
                               increment::Integer=1) where {D}
  value = _checked_positive(increment, "increment")
  checked_axis = _checked_axis(grid(plan), axis)
  increments = ntuple(current_axis -> current_axis == checked_axis ? value : 0, D)
  return _apply_p_refinement!(plan, leaf, increments)
end

function request_p_refinement!(plan::AdaptivityPlan{D}, leaf::Integer,
                               increments::NTuple{D,<:Integer}) where {D}
  return _apply_p_refinement!(plan, leaf, _checked_degree_steps(increments, "increments"))
end

"""
    request_p_derefinement!(plan, leaf, axis; decrement=1)
    request_p_derefinement!(plan, leaf, decrements)

Decrease the target polynomial degree on one active leaf.

Derefinement is bounded below by `plan.limits.min_p`. As for refinement, the
operation only edits the stored target degree tuple; continuity enforcement is
deferred to target-space compilation.
"""
function request_p_derefinement!(plan::AdaptivityPlan{D}, leaf::Integer, axis::Integer;
                                 decrement::Integer=1) where {D}
  value = _checked_positive(decrement, "decrement")
  checked_axis = _checked_axis(grid(plan), axis)
  decrements = ntuple(current_axis -> current_axis == checked_axis ? value : 0, D)
  return _apply_p_derefinement!(plan, leaf, decrements)
end

function request_p_derefinement!(plan::AdaptivityPlan{D}, leaf::Integer,
                                 decrements::NTuple{D,<:Integer}) where {D}
  return _apply_p_derefinement!(plan, leaf, _checked_degree_steps(decrements, "decrements"))
end

# Source-to-target transition compilation and field/state transfer.

"""
    SpaceTransition

Compiled overlap information between a source space and an adapted target space.

For each active target leaf, the transition stores the contiguous slice of
source leaves whose physical cells overlap that target leaf. This overlap data
is the geometric backbone for state transfer: values on the target space are
evaluated by locating the source leaf that contains each target quadrature
point.
"""
struct SpaceTransition{D,T<:AbstractFloat,S<:HpSpace{D,T},N<:HpSpace{D,T},O<:Vector{Int},
                       C<:Vector{Int},V<:Vector{Int}}
  source_space::S
  target_space::N
  source_offsets::O
  source_counts::C
  source_leaf_data::V
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
  checked_leaf = _checked_cell(grid(target_space(transition)), target_leaf)
  is_active_leaf(grid(target_space(transition)), checked_leaf) ||
    throw(ArgumentError("leaf $checked_leaf is not an active target leaf"))
  count = @inbounds transition.source_counts[checked_leaf]
  count != 0 || throw(ArgumentError("leaf $checked_leaf has no compiled source leaves"))
  first = @inbounds transition.source_offsets[checked_leaf]
  return first, count
end

"""
    source_leaves(transition, target_leaf)

Return the active source leaves that overlap one active target leaf.

These leaves are stored in the traversal order used by [`transition`](@ref) and
form the local search set used during state transfer.
"""
function source_leaves(transition::SpaceTransition, target_leaf::Integer)
  first, count = _source_leaf_range(transition, target_leaf)
  return transition.source_leaf_data[first:(first+count-1)]
end

# Source/target overlap is collected by descending the source tree only through
# cells whose physical boxes overlap the target leaf. Because both grids are
# dyadic on the same root box, box overlap on logical coordinates is sufficient
# to identify potentially contributing source leaves.
function _collect_source_leaves!(leaves::Vector{Int}, source_grid::CartesianGrid{D},
                                 target_grid::CartesianGrid{D}, source_cell::Int,
                                 target_leaf::Int) where {D}
  _cells_overlap(source_grid, source_cell, target_grid, target_leaf) || return leaves

  if is_active_leaf(source_grid, source_cell)
    push!(leaves, source_cell)
    return leaves
  end

  first = first_child(source_grid, source_cell)
  first == NONE && return leaves

  for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
    _collect_source_leaves!(leaves, source_grid, target_grid, child, target_leaf)
  end

  return leaves
end

# A target leaf overlaps only source leaves inside the source subtree rooted at
# the corresponding root cell of the target leaf. Starting the search there
# avoids scanning unrelated roots in multi-root grids.
function _transition_source_leaves(source_grid::CartesianGrid, target_grid::CartesianGrid,
                                   target_leaf::Int)
  root = target_leaf

  while true
    parent_cell = parent(target_grid, root)
    parent_cell == NONE && break
    root = parent_cell
  end

  leaves = Int[]
  _collect_source_leaves!(leaves, source_grid, target_grid, root, target_leaf)
  isempty(leaves) && throw(ArgumentError("leaf $target_leaf has no source leaves"))
  return leaves
end

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
function _inherited_target_degrees(space::HpSpace{D}, target_domain::Domain{D}) where {D}
  source_grid = grid(space)
  target_grid = grid(target_domain)
  active = active_leaves(target_grid)
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    source_leaves = _transition_source_leaves(source_grid, target_grid, active[index])
    degrees[index] = ntuple(axis -> maximum(cell_degrees(space, leaf)[axis]
                                            for leaf in source_leaves), D)
  end

  return degrees
end

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
  target = copy(target_domain(driver_plan))
  degrees = isnothing(degree_offset) ? _inherited_target_degrees(checked_space, target) :
            _offset_target_degrees(driver_plan, _degree_offset_tuple(degree_offset, D))
  return AdaptivityPlan(checked_space, target, degrees; limits=limits)
end

function derived_adaptivity_plan(driver_plan::AdaptivityPlan, field::AbstractField;
                                 degree_offset=nothing, limits=AdaptivityLimits(field_space(field)))
  return derived_adaptivity_plan(driver_plan, field_space(field); degree_offset=degree_offset,
                                 limits=limits)
end

"""
    transition(plan)

Build the `SpaceTransition` from the source space of `plan` to its target
space.

This compiles the current target domain and target degree data of `plan` into a
new `HpSpace`, then records for every active target leaf which active source
leaves overlap it. The resulting `SpaceTransition` is the object required for
state transfer and for recreating field layouts on the target space.

Because the source and target domains share the same physical box and root-grid
layout, the transition can describe source-to-target overlap purely by leaf
relations on dyadic grids.
"""
function transition(plan::AdaptivityPlan{D,T}) where {D,T<:AbstractFloat}
  old_space = source_space(plan)
  new_domain = copy(target_domain(plan))
  degree_policy = StoredDegrees(new_domain, plan.target_degrees)
  options = SpaceOptions(basis=basis_family(old_space), degree=degree_policy,
                         quadrature=old_space.quadrature_policy,
                         continuity=continuity_policy(old_space))
  new_space = HpSpace(new_domain, options)
  source_grid = grid(old_space)
  target_grid = grid(new_space)
  source_offsets = zeros(Int, stored_cell_count(target_grid))
  source_counts = zeros(Int, stored_cell_count(target_grid))
  source_data = Int[]

  for leaf in active_leaves(new_space)
    leaves = _transition_source_leaves(source_grid, target_grid, leaf)
    source_offsets[leaf] = length(source_data) + 1
    source_counts[leaf] = length(leaves)
    append!(source_data, leaves)
  end

  return SpaceTransition(old_space, new_space, source_offsets, source_counts, source_data)
end

"""
    adapted_field(transition, field; name=field_name(field))

Recreate `field` on the target space of `transition`.

The field kind and component count are preserved; only the owning space and,
optionally, the symbolic field name change.
"""
function adapted_field(transition::SpaceTransition, field::ScalarField;
                       name::Symbol=field_name(field))
  return ScalarField(target_space(transition); name=name)
end

function adapted_field(transition::SpaceTransition, field::VectorField;
                       name::Symbol=field_name(field))
  return VectorField(target_space(transition), component_count(field); name=name)
end

"""
    adapted_fields(transition, fields...)
    adapted_fields(transition, fields::Tuple)
    adapted_fields(transition, layout)

Recreate one or more fields on the target space of `transition`.
"""
function adapted_fields(transition::SpaceTransition, fields::Tuple)
  return ntuple(index -> adapted_field(transition, fields[index]), length(fields))
end

function adapted_fields(transition::SpaceTransition, layout::FieldLayout)
  adapted_fields(transition, fields(layout))
end

function adapted_fields(transition::SpaceTransition, first_field::AbstractField,
                        remaining_fields::AbstractField...)
  return adapted_fields(transition, (first_field, remaining_fields...))
end

# State transfer is formulated as an `L2` projection on the target space. The
# local mass operator below provides the symmetric positive definite block that
# defines the target inner product for one field.
struct _TransferMass{F}
  field::F
end

# The transfer source operator evaluates the old state on target quadrature
# points and injects those values into the right-hand side of the projection
# system. Together with `_TransferMass` this yields the Galerkin `L2`
# projection from the source field to the target field.
struct _TransferSource{F,O,S,TR}
  field::F
  old_field::O
  old_state::S
  transition::TR
end

# Assemble the element-local mass matrix
#   M_ij = ∫_K φ_i φ_j dΩ
# for the target field, duplicated componentwise for vector-valued fields.
function cell_matrix!(local_matrix, operator::_TransferMass, values::CellValues)
  mode_count = local_mode_count(values, operator.field)
  components = component_count(operator.field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      value_row = shape_value(values, operator.field, point_index, row_mode)

      for col_mode in 1:mode_count
        contribution = value_row *
                       shape_value(values, operator.field, point_index, col_mode) *
                       weighted
        contribution == 0 && continue

        for component in 1:components
          row = local_dof_index(values, operator.field, component, row_mode)
          col = local_dof_index(values, operator.field, component, col_mode)
          local_matrix[row, col] += contribution
        end
      end
    end
  end

  return nothing
end

# State transfer repeatedly asks whether a target quadrature point belongs to a
# candidate source leaf. The tolerance keeps roundoff near shared faces from
# creating spurious "point not found" failures.
function _point_in_cell(domain_data::Domain{D,T}, leaf::Int, x::NTuple{D,<:Real};
                        tolerance::T=T(1.0e-12)) where {D,T<:AbstractFloat}
  lower = cell_lower(domain_data, leaf)
  upper = cell_upper(domain_data, leaf)
  return all(lower[axis] - tolerance <= x[axis] <= upper[axis] + tolerance for axis in 1:D)
end

function _source_leaf_at_point(transition::SpaceTransition{D,T}, target_leaf::Int,
                               x::NTuple{D,<:Real}) where {D,T<:AbstractFloat}
  domain_data = domain(source_space(transition))

  for leaf in source_leaves(transition, target_leaf)
    _point_in_cell(domain_data, leaf, x; tolerance=T(1.0e-12)) && return leaf
  end

  throw(ArgumentError("point $x does not lie in any source leaf for target leaf $target_leaf"))
end

# Evaluate the projection right-hand side
#   b_i = ∫_K u_old φ_i dΩ
# by sampling the old state at the physical target quadrature points.
function cell_rhs!(local_rhs, operator::_TransferSource, values::CellValues)
  mode_count = local_mode_count(values, operator.field)
  components = component_count(operator.field)

  for point_index in 1:point_count(values)
    x = point(values, point_index)
    weighted = weight(values, point_index)
    old_value = _transition_value(operator.transition, operator.old_field, operator.old_state,
                                  values.leaf, x)

    for mode_index in 1:mode_count
      shape = shape_value(values, operator.field, point_index, mode_index)
      shape == 0 && continue

      if components == 1
        local_rhs[local_dof_index(values, operator.field, 1, mode_index)] += old_value *
                                                                             shape *
                                                                             weighted
      else
        for component in 1:components
          local_rhs[local_dof_index(values, operator.field, component, mode_index)] += old_value[component] *
                                                                                       shape *
                                                                                       weighted
        end
      end
    end
  end

  return nothing
end

"""
    transfer_state(transition, state, old_fields, new_fields; linear_solve=default_linear_solve)
    transfer_state(transition, state, old_field, new_field; linear_solve=default_linear_solve)
    transfer_state(transition, state; linear_solve=default_linear_solve)

Transfer field coefficients from the source space to the target space of
`transition` by cellwise `L2` projection.

The first forms project explicitly paired old/new fields. The zero-argument
field form recreates the full field layout on the target space via
[`adapted_fields`](@ref) and returns both the new fields and the transferred
state.

The transfer is purely geometric and variational: it depends on the old state,
the source/target spaces, and the chosen linear solve path, but not on any
specific PDE operator.
"""
function transfer_state(transition::SpaceTransition, state::State, old_fields::Tuple,
                        new_fields::Tuple; linear_solve=default_linear_solve)
  length(old_fields) == length(new_fields) ||
    throw(ArgumentError("old and new field tuples must have the same length"))
  problem = AffineProblem(new_fields...)

  for index in eachindex(old_fields)
    old_field = old_fields[index]
    new_field = new_fields[index]
    field_space(old_field) === source_space(transition) ||
      throw(ArgumentError("old fields must belong to the transition source space"))
    field_space(new_field) === target_space(transition) ||
      throw(ArgumentError("new fields must belong to the transition target space"))
    component_count(old_field) == component_count(new_field) ||
      throw(ArgumentError("field component counts must match during transfer"))
    field_dof_range(field_layout(state), old_field)
    add_cell!(problem, _TransferMass(new_field))
    add_cell!(problem, _TransferSource(new_field, old_field, state, transition))
  end

  plan = compile(problem)
  return State(plan, solve(assemble(plan); linear_solve=linear_solve))
end

function transfer_state(transition::SpaceTransition, state::State, old_field::AbstractField,
                        new_field::AbstractField; linear_solve=default_linear_solve)
  return transfer_state(transition, state, (old_field,), (new_field,); linear_solve=linear_solve)
end

function transfer_state(transition::SpaceTransition, state::State;
                        linear_solve=default_linear_solve)
  old_fields = fields(field_layout(state))
  new_fields = adapted_fields(transition, old_fields)
  return new_fields,
         transfer_state(transition, state, old_fields, new_fields; linear_solve=linear_solve)
end

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
      continue
    end

    _same_adaptivity_geometry(reference_target, target_domain(plan)) ||
      throw(ArgumentError("adaptivity plan targets must share one physical domain and periodic topology"))
    active_leaves(grid(reference_target)) == active_leaves(grid(target_domain(plan))) ||
      throw(ArgumentError("adaptivity plan targets must share the same active-leaf topology"))
  end

  length(plan_by_space) == length(layout_spaces) ||
    throw(ArgumentError("the state layout requires exactly one adaptivity plan per source space"))

  return plan_by_space
end

"""
    transfer_state(plans, state; linear_solve=default_linear_solve)

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
function transfer_state(plans::Tuple, state::State; linear_solve=default_linear_solve)
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

# Modal coefficient indicators and smoothness heuristics.

function _local_mode_energy(field::AbstractField, state::State{T}, compiled::_CompiledLeaf,
                            mode_index::Int) where {T<:AbstractFloat}
  energy = zero(T)

  for component in 1:component_count(field)
    amplitude = _local_mode_amplitude(field, state, compiled, component, mode_index)
    energy += amplitude * amplitude
  end

  return energy
end

@inline function _mode_with_axis_value(mode::NTuple{D,<:Integer}, axis::Int, value::Int) where {D}
  return ntuple(current_axis -> current_axis == axis ? value : Int(mode[current_axis]), D)
end

# Degree-zero DG cells only contain a constant mode and therefore have no
# higher modal layer to inspect. Degree-one cells only contain the two endpoint
# modes `ψ₀` and `ψ₁` on each axis. Those do not separate constant and linear
# content by themselves, so for `p = 1` we first transform the endpoint pair
# into its constant/linear combination before extracting the top layer.
function _axis_layer_energies(field::AbstractField, state::State{T}, compiled::_CompiledLeaf{D},
                              axis::Int) where {D,T<:AbstractFloat}
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

      for component in 1:component_count(field)
        lower = _local_mode_amplitude(field, state, compiled, component, lower_index)
        upper = _local_mode_amplitude(field, state, compiled, component, upper_index)
        constant = half * (lower + upper)
        linear = half * (upper - lower)
        previous_energy += constant * constant
        top_energy += linear * linear
      end
    end

    return top_energy, previous_energy
  end

  for mode_index in eachindex(compiled.local_modes)
    mode = compiled.local_modes[mode_index]
    layer = mode[axis]
    layer == degree_value || layer == degree_value - 1 || continue
    amplitude_squared = _local_mode_energy(field, state, compiled, mode_index)

    if layer == degree_value
      top_energy += amplitude_squared
    else
      previous_energy += amplitude_squared
    end
  end

  return top_energy, previous_energy
end

@inline _indicator_squared_norm(value::Number) = value * value
@inline _indicator_squared_norm(value::Tuple) = sum(_indicator_squared_norm(component)
                                                    for component in value)

function _field_only_state(state::State{T}, field::AbstractField) where {T<:AbstractFloat}
  return State(FieldLayout((field,)), field_values(state, field))
end

function _active_leaf_lookup(space::HpSpace)
  lookup = zeros(Int, stored_cell_count(grid(space)))

  for (index, leaf) in enumerate(active_leaves(space))
    lookup[leaf] = index
  end

  return lookup
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

This is the default problem-independent refinement indicator on DG axes. The
idea is the standard DG heuristic: if two neighboring leaves represent a smooth
solution well, then their traces should already agree reasonably closely across
their common face. Large jumps therefore point to under-resolution normal to
that interface.
"""
function interface_jump_indicators(state::State{T}, field::AbstractField) where {T<:AbstractFloat}
  field_state = _field_only_state(state, field)
  space = field_space(field)
  D = dimension(space)
  indicators = zeros(T, active_leaf_count(space), D)
  interfaces = _compile_interfaces(field_layout(field_state))
  isempty(interfaces) && return [ntuple(_ -> zero(T), D) for _ in 1:active_leaf_count(space)]
  lookup = _active_leaf_lookup(space)
  domain_data = domain(space)

  for values in interfaces
    axis = face_axis(values)
    minus_leaf = values.minus_leaf
    plus_leaf = values.plus_leaf
    minus_side = minus(values)
    plus_side = plus(values)
    jump_energy = zero(T)

    for point_index in 1:point_count(values)
      jump_energy += _indicator_squared_norm(jump(value(minus_side, field_state, field,
                                                        point_index),
                                                  value(plus_side, field_state, field, point_index))) *
                     weight(values, point_index)
    end

    minus_degree = cell_degrees(space, minus_leaf)[axis]
    plus_degree = cell_degrees(space, plus_leaf)[axis]
    minus_scale = ((minus_degree + 1)^2) / cell_size(domain_data, minus_leaf, axis)
    plus_scale = ((plus_degree + 1)^2) / cell_size(domain_data, plus_leaf, axis)
    indicators[lookup[minus_leaf], axis] += minus_scale * jump_energy
    indicators[lookup[plus_leaf], axis] += plus_scale * jump_energy
  end

  return [ntuple(axis -> sqrt(indicators[leaf_index, axis]), D)
          for leaf_index in 1:active_leaf_count(space)]
end

function _merge_axis_indicators(space::HpSpace{D}, cg_indicators, dg_indicators) where {D}
  return [ntuple(axis -> is_continuous_axis(space, axis) ? cg_indicators[leaf_index][axis] :
                         dg_indicators[leaf_index][axis], D)
          for leaf_index in 1:active_leaf_count(space)]
end

function _default_refinement_indicators(state::State{T},
                                        field::AbstractField) where {T<:AbstractFloat}
  space = field_space(field)
  has_cg = any(axis -> is_continuous_axis(space, axis), 1:dimension(space))
  has_dg = any(axis -> !is_continuous_axis(space, axis), 1:dimension(space))
  has_cg && !has_dg && return coefficient_indicators(state, field)
  has_dg && !has_cg && return interface_jump_indicators(state, field)
  return _merge_axis_indicators(space, coefficient_indicators(state, field),
                                interface_jump_indicators(state, field))
end

function _default_smoothness_indicators(state::State{T},
                                        field::AbstractField) where {T<:AbstractFloat}
  space = field_space(field)
  decay = coefficient_decay_indicators(state, field)
  any(axis -> !is_continuous_axis(space, axis), 1:dimension(space)) || return decay
  D = dimension(space)

  return [ntuple(axis -> begin
                   degrees = cell_degrees(space, active_leaf(space, leaf_index))
                   !is_continuous_axis(space, axis) && degrees[axis] == 0 ? floatmax(T) :
                   decay[leaf_index][axis]
                 end, D) for leaf_index in 1:active_leaf_count(space)]
end

"""
    coefficient_indicators(state, field)

Per-axis highest-order modal amplitudes on each active leaf. These are the
default modal refinement indicators and remain the default refinement signal on
CG axes.

For each active leaf and each axis, the indicator sums the squared amplitudes of
the local top modal layer and returns the square root of that sum. For
`p_axis ≥ 2`, this is the usual set of modes whose index on that axis equals
the local degree `p_axis`. At `p_axis = 1`, the integrated-Legendre endpoint
pair is first rewritten into constant/linear content and the linear part is
used as the top layer. Large values therefore signal that the highest resolved
polynomial content still carries significant energy, which is a classical
heuristic for requesting further refinement.
"""
function coefficient_indicators(state::State{T}, field::AbstractField) where {T<:AbstractFloat}
  field_dof_range(field_layout(state), field)
  space = field_space(field)
  indicators = Vector{NTuple{dimension(space),T}}(undef, active_leaf_count(space))
  thread_axis_sums = [zeros(T, dimension(space))
                      for _ in 1:max(1, min(Threads.nthreads(), length(space.active_leaves)))]

  _run_chunks_with_scratch!(thread_axis_sums,
                            length(space.active_leaves)) do axis_sums, first_leaf, last_leaf
    for leaf_index in first_leaf:last_leaf
      compiled = space.compiled_leaves[leaf_index]
      for axis in 1:dimension(space)
        top_energy, _ = _axis_layer_energies(field, state, compiled, axis)
        axis_sums[axis] = top_energy
      end

      indicators[leaf_index] = ntuple(axis -> sqrt(axis_sums[axis]), dimension(space))
    end
  end

  return indicators
end

"""
    coefficient_coarsening_indicators(state, field)

Per-axis normalized top-layer modal energy ratios on each active leaf. Smaller
values indicate that one `p`-derefinement step is likely to be harmless on that
axis.

If the modal energy contained in the highest layer is small relative to the
total local modal energy, removing that layer should have little effect on the
local approximation. At `p_axis = 1`, the endpoint pair is interpreted as
constant versus linear content before the top-layer energy is formed. The
returned quantity is `√(E_top / E_total)` per axis.
"""
function coefficient_coarsening_indicators(state::State{T},
                                           field::AbstractField) where {T<:AbstractFloat}
  field_dof_range(field_layout(state), field)
  space = field_space(field)
  indicators = Vector{NTuple{dimension(space),T}}(undef, active_leaf_count(space))
  thread_axis_energy = [zeros(T, dimension(space))
                        for _ in 1:max(1, min(Threads.nthreads(), length(space.active_leaves)))]

  _run_chunks_with_scratch!(thread_axis_energy,
                            length(space.active_leaves)) do axis_energy, first_leaf, last_leaf
    for leaf_index in first_leaf:last_leaf
      compiled = space.compiled_leaves[leaf_index]
      fill!(axis_energy, zero(T))
      total_energy = zero(T)

      for mode_index in eachindex(compiled.local_modes)
        mode = compiled.local_modes[mode_index]
        amplitude_squared = _local_mode_energy(field, state, compiled, mode_index)
        amplitude_squared == zero(T) && continue
        total_energy += amplitude_squared
      end

      for axis in 1:dimension(space)
        axis_energy[axis], _ = _axis_layer_energies(field, state, compiled, axis)
      end

      indicators[leaf_index] = total_energy == zero(T) ? ntuple(_ -> zero(T), dimension(space)) :
                               ntuple(axis -> sqrt(axis_energy[axis] / total_energy),
                                      dimension(space))
    end
  end

  return indicators
end

"""
    coefficient_decay_indicators(state, field)

Per-axis top-to-previous modal layer energy ratios on each active leaf. These
values serve as smoothness indicators for the `h`/`p` split in mixed
adaptivity.

The indicator compares the modal energy in the top layer `p_axis` with the
energy in the preceding layer `p_axis - 1`. At `p_axis = 1`, the endpoint pair
is interpreted as constant versus linear content before forming that ratio.
Small ratios suggest rapid modal decay and hence local smoothness, which favors
`p`-refinement; large ratios suggest slower decay and favor `h`-refinement.
"""
function coefficient_decay_indicators(state::State{T},
                                      field::AbstractField) where {T<:AbstractFloat}
  field_dof_range(field_layout(state), field)
  space = field_space(field)
  decay = Vector{NTuple{dimension(space),T}}(undef, active_leaf_count(space))
  thread_axis_decay = [zeros(T, dimension(space))
                       for _ in 1:max(1, min(Threads.nthreads(), length(space.active_leaves)))]

  _run_chunks_with_scratch!(thread_axis_decay,
                            length(space.active_leaves)) do axis_decay, first_leaf, last_leaf
    for leaf_index in first_leaf:last_leaf
      compiled = space.compiled_leaves[leaf_index]
      fill!(axis_decay, zero(T))

      for axis in 1:dimension(space)
        top_energy, previous_energy = _axis_layer_energies(field, state, compiled, axis)
        axis_decay[axis] = if top_energy == zero(T)
          zero(T)
        elseif previous_energy == zero(T) || previous_energy <= eps(T) * top_energy
          floatmax(T)
        else
          ratio = top_energy / previous_energy
          isfinite(ratio) ? sqrt(ratio) : floatmax(T)
        end
      end

      decay[leaf_index] = ntuple(axis -> axis_decay[axis], dimension(space))
    end
  end

  return decay
end

function _indicator_values(indicator, state::State, field::AbstractField, name::AbstractString)
  applicable(indicator, state, field) ||
    throw(ArgumentError("$name must be callable as $name(state, field)"))
  return indicator(state, field)
end

# The generic adaptivity builders accept user-supplied indicator callbacks. The
# checks here fail early with a descriptive message when a callback has the wrong
# arity or returns data with the wrong shape.
function _candidate_indicator_values(indicator, state::State, field::AbstractField, candidates,
                                     name::AbstractString)
  applicable(indicator, state, field, candidates) ||
    throw(ArgumentError("$name must be callable as $name(state, field, candidates)"))
  return indicator(state, field, candidates)
end

# State-driven plan builders all share the same "optional indicator" pattern:
# only evaluate a coarsening indicator when the corresponding threshold is
# active, so optional features stay dormant without invoking user callbacks.
function _optional_state_indicator_values(indicator, state::State, field::AbstractField, threshold,
                                          name::AbstractString)
  threshold === nothing && return nothing
  return _indicator_values(indicator, state, field, name)
end

# Optional `h`-coarsening additionally needs candidate generation. This helper
# keeps the resolution of default candidates and their indicators consistent
# between pure `h` and mixed `hp` planning.
function _state_h_coarsening_data(state::State, field::AbstractField, candidates, indicator,
                                  threshold, limits)
  threshold === nothing && return nothing, nothing
  space = field_space(field)
  resolved_candidates = isnothing(candidates) ?
                        _default_h_coarsening_candidates(space; limits=limits) : candidates
  resolved_indicators = _candidate_indicator_values(indicator, state, field, resolved_candidates,
                                                    "h_coarsening_indicator")
  return resolved_candidates, resolved_indicators
end

# Marking and candidate-selection helpers shared by `h`, `p`, and `hp` planning.

function _checked_axis_indicator_values(space::HpSpace{D}, indicators,
                                        name::AbstractString) where {D}
  length(indicators) == active_leaf_count(space) ||
    throw(ArgumentError("$name count must match the active-leaf count"))

  for axes in indicators
    axes isa NTuple{D,<:Real} || throw(ArgumentError("$name must be NTuple{$D,<:Real} values"))

    for axis in 1:D
      isfinite(float(axes[axis])) || throw(ArgumentError("$name must be finite"))
    end
  end

  return indicators
end

# Threshold handling is centralized so all plan builders share the same
# admissibility and error semantics for optional coarsening thresholds.
function _checked_nonnegative_threshold(value::Real, name::AbstractString)
  checked = float(value)
  isfinite(checked) || throw(ArgumentError("$name must be finite"))
  checked >= 0 || throw(ArgumentError("$name must be non-negative"))
  return checked
end

function _checked_optional_threshold(value, name::AbstractString)
  value === nothing && return nothing
  return _checked_nonnegative_threshold(value, name)
end

# Dörfler bulk marking selects the smallest set of leaf-axis pairs whose
# squared-indicator contributions account for a fraction `θ` of the total
# indicator mass. Sorting by decreasing contribution implements the standard
# greedy realization of this criterion.
function _marked_axes(space::HpSpace{D}, indicators; threshold::Real=0.5,
                      admissible=(leaf, axis) -> true) where {D}
  theta = float(threshold)
  0 <= theta <= 1 || throw(ArgumentError("threshold must lie in [0, 1]"))
  checked = _checked_axis_indicator_values(space, indicators, "indicators")
  marked = fill(ntuple(_ -> false, D), active_leaf_count(space))
  contributions = Tuple{typeof(theta),Int,Int}[]
  total_indicator = zero(typeof(theta))

  for (leaf_index, axes) in enumerate(checked)
    leaf = active_leaf(space, leaf_index)

    for axis in 1:D
      admissible(leaf, axis) || continue
      contribution = abs2(float(axes[axis]))
      contribution == 0 && continue
      total_indicator += contribution
      push!(contributions, (contribution, leaf_index, axis))
    end
  end

  total_indicator == 0 && return marked
  theta == 0 && return marked
  sort!(contributions; by=entry -> (-entry[1], entry[2], entry[3]))
  target = theta * total_indicator
  accumulated = zero(typeof(theta))

  for (contribution, leaf_index, axis) in contributions
    current = marked[leaf_index]
    marked[leaf_index] = ntuple(current_axis -> current_axis == axis ? true : current[current_axis],
                                D)
    accumulated += contribution
    accumulated >= target && break
  end

  return marked
end

# Coarsening is threshold based rather than bulk based: every admissible
# leaf-axis pair with sufficiently small indicator is selected unless another
# planned modification already blocks that axis or leaf.
function _coarsened_axes(space::HpSpace{D}, indicators; threshold=nothing,
                         threshold_name::AbstractString="p_coarsening_threshold",
                         indicator_name::AbstractString="p coarsening indicators",
                         admissible=(leaf, axis) -> true, blocked=nothing) where {D}
  value = _checked_optional_threshold(threshold, threshold_name)

  if value === nothing
    indicators === nothing ||
      throw(ArgumentError("$threshold_name must be set when $indicator_name are provided"))
    return fill(ntuple(_ -> false, D), active_leaf_count(space))
  end

  indicators === nothing &&
    throw(ArgumentError("$indicator_name must be provided when $threshold_name is set"))
  checked = _checked_axis_indicator_values(space, indicators, indicator_name)
  blocked_axes = isnothing(blocked) ? fill(ntuple(_ -> false, D), active_leaf_count(space)) :
                 blocked
  marked = fill(ntuple(_ -> false, D), active_leaf_count(space))

  for (leaf_index, axes) in enumerate(checked)
    leaf = active_leaf(space, leaf_index)
    current = marked[leaf_index]

    for axis in 1:D
      blocked_axes[leaf_index][axis] && continue
      admissible(leaf, axis) || continue
      float(axes[axis]) <= value || continue
      current = ntuple(current_axis -> current_axis == axis ? true : current[current_axis], D)
    end

    marked[leaf_index] = current
  end

  return marked
end

function _active_leaf_index(space::HpSpace, leaf::Integer)
  checked_leaf = _checked_cell(grid(space), leaf)
  index = @inbounds space.leaf_to_index[checked_leaf]
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
  split = split_axis(grid_data, parent)
  parent_coordinate = logical_coordinate(grid_data, parent, split)
  lower_child = logical_coordinate(grid_data, child, split) == 2 * parent_coordinate

  return ntuple(axis -> begin
                  ξ = T(ξ_child[axis])

                  if axis == split
                    lower_child ? (ξ - one(T)) / T(2) : (ξ + one(T)) / T(2)
                  else
                    ξ
                  end
                end, D)
end

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

# Precompute the parent reference mass matrix and basis modes for a given target
# degree tuple. Candidate scoring repeatedly solves projection systems with the
# same parent degrees, so caching the Cholesky factor avoids redundant dense
# assembly.
function _reference_projection_data(space::HpSpace{D,T}, degrees::NTuple{D,Int},
                                    cache::Dict{NTuple{D,Int},
                                                Tuple{Vector{NTuple{D,Int}},Cholesky{T,Matrix{T}}}}) where {D,
                                                                                                            T<:AbstractFloat}
  haskey(cache, degrees) && return cache[degrees]
  modes = collect(basis_modes(basis_family(space), degrees))
  mode_count = length(modes)
  quadrature = TensorQuadrature(T,
                                ntuple(axis -> minimum_gauss_legendre_points(2 * degrees[axis]), D))
  shape_values = Matrix{T}(undef, mode_count, point_count(quadrature))
  mass = zeros(T, mode_count, mode_count)

  for point_index in 1:point_count(quadrature)
    ξ = point(quadrature, point_index)
    basis_values = ntuple(axis -> _fe_basis_values(ξ[axis], degrees[axis]), D)

    for mode_index in 1:mode_count
      shape_values[mode_index, point_index] = _projection_shape_value(modes, basis_values,
                                                                      mode_index)
    end
  end

  for point_index in 1:point_count(quadrature)
    weighted = weight(quadrature, point_index)

    for row_index in 1:mode_count
      shape_row = shape_values[row_index, point_index]

      for column_index in 1:row_index
        contribution = shape_row * shape_values[column_index, point_index] * weighted
        mass[row_index, column_index] += contribution
        row_index == column_index || (mass[column_index, row_index] += contribution)
      end
    end
  end

  data = (modes, cholesky(Hermitian(mass)))
  cache[degrees] = data
  return data
end

function _projection_reference_table(space::HpSpace{D,T},
                                     candidates::AbstractVector{HCoarseningCandidate{D}}) where {D,
                                                                                                 T<:AbstractFloat}
  reference_type = Tuple{Vector{NTuple{D,Int}},Cholesky{T,Matrix{T}}}
  cache = Dict{NTuple{D,Int},reference_type}()
  references = Vector{reference_type}(undef, length(candidates))
  max_mode_count = 0

  for candidate_index in eachindex(candidates)
    reference = _reference_projection_data(space, candidates[candidate_index].target_degrees, cache)
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

struct _ProjectionIndicatorScratch{T<:AbstractFloat}
  rhs::Matrix{T}
  coefficients::Matrix{T}
end

"""
    projection_coarsening_indicators(state, field, candidates)

Relative local `L2` projection defects for immediate `h`-coarsening candidates.
Smaller values indicate that derefining the candidate parent cell is likely to
be harmless.

The indicator compares the fine representation on the two child leaves with its
`L2` projection onto the candidate parent space and returns the relative defect

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
  components = component_count(field)
  references, max_mode_count = _projection_reference_table(space, checked_candidates)
  worker_count = max(1, min(Threads.nthreads(), length(checked_candidates)))
  scratch = [_ProjectionIndicatorScratch(Matrix{T}(undef, max_mode_count, components),
                                         Matrix{T}(undef, max_mode_count, components))
             for _ in 1:worker_count]

  _with_internal_blas_threads() do
    _run_chunks_with_scratch!(scratch,
                              length(checked_candidates)) do buffers, first_candidate,
                                                             last_candidate
      for candidate_index in first_candidate:last_candidate
        candidate = checked_candidates[candidate_index]
        modes, factor = references[candidate_index]
        mode_count = length(modes)
        rhs = @view buffers.rhs[1:mode_count, 1:components]
        coefficients = @view buffers.coefficients[1:mode_count, 1:components]
        fill!(rhs, zero(T))
        fine_norm = zero(T)
        parent_jacobian = jacobian_determinant_from_biunit_cube(domain(space), candidate.cell)

        for child in candidate.children
          quadrature = TensorQuadrature(T,
                                        _projection_quadrature_shape(candidate.target_degrees,
                                                                     cell_degrees(space, child)))
          child_jacobian = jacobian_determinant_from_biunit_cube(domain(space), child)

          for point_index in 1:point_count(quadrature)
            ξ_child = point(quadrature, point_index)
            ξ_parent = _child_point_in_parent(space, candidate.cell, child, ξ_child)
            basis_values = ntuple(axis -> _fe_basis_values(ξ_parent[axis],
                                                           candidate.target_degrees[axis]),
                                  dimension(space))
            weighted = weight(quadrature, point_index) * child_jacobian
            local_value = _field_value_on_leaf(field, state, child, ξ_child)

            if components == 1
              scalar_value = local_value
              fine_norm += scalar_value * scalar_value * weighted

              for mode_index in 1:mode_count
                rhs[mode_index, 1] += _projection_shape_value(modes, basis_values, mode_index) *
                                      scalar_value *
                                      weighted
              end
            else
              for component in 1:components
                scalar_value = local_value[component]
                fine_norm += scalar_value * scalar_value * weighted

                for mode_index in 1:mode_count
                  rhs[mode_index, component] += _projection_shape_value(modes, basis_values,
                                                                        mode_index) *
                                                scalar_value *
                                                weighted
                end
              end
            end
          end
        end

        if fine_norm == zero(T)
          indicators[candidate_index] = zero(T)
          continue
        end

        projection_norm = zero(T)

        for component in 1:components
          @views begin
            rhs_component = rhs[:, component]
            coefficients_component = coefficients[:, component]
            coefficients_component .= rhs_component ./ parent_jacobian
            _with_serialized_blas() do
              ldiv!(factor, coefficients_component)
            end

            for mode_index in eachindex(rhs_component)
              projection_norm += rhs_component[mode_index] * coefficients_component[mode_index]
            end
          end
        end

        indicators[candidate_index] = sqrt(max(zero(T), fine_norm - projection_norm) / fine_norm)
      end
    end
  end

  return indicators
end

# Mixed `hp` adaptation must avoid selecting derefinement on leaves that are
# already scheduled for any other change. This helper condenses the various
# change markers into one per-leaf "blocked" flag.
function _leaf_change_flags(marked_h, marked_p, coarsened_p)
  return [any(marked_h[index]) || any(>(0), marked_p[index]) || any(coarsened_p[index])
          for index in eachindex(marked_h)]
end

# Candidate selection for `h`-derefinement is independent of bulk marking. The
# candidates have already been structurally validated; the remaining logic only
# filters by threshold, user limits, and conflicts with other requested changes.
function _selected_h_coarsening_candidates(space::HpSpace{D}, candidates, indicators;
                                           threshold=nothing, limits=AdaptivityLimits(space),
                                           blocked=falses(active_leaf_count(space))) where {D}
  value = _checked_optional_threshold(threshold, "h_coarsening_threshold")

  if value === nothing
    (candidates === nothing && indicators === nothing) ||
      throw(ArgumentError("h_coarsening_threshold must be set when h coarsening candidates are provided"))
    return HCoarseningCandidate{D}[]
  end

  candidates === nothing &&
    throw(ArgumentError("h coarsening candidates must be provided when h_coarsening_threshold is set"))
  indicators === nothing &&
    throw(ArgumentError("h coarsening indicators must be provided when h_coarsening_threshold is set"))
  length(blocked) == active_leaf_count(space) ||
    throw(ArgumentError("blocked leaf flags must match the active-leaf count"))
  checked_limits = _checked_limits(limits, space)
  checked_candidates = _checked_h_coarsening_candidates(space, candidates)
  length(indicators) == length(checked_candidates) ||
    throw(ArgumentError("h coarsening indicator count must match the candidate count"))
  selected = HCoarseningCandidate{D}[]

  for index in eachindex(checked_candidates)
    candidate = checked_candidates[index]
    level(grid(space), candidate.cell, candidate.axis) >=
    checked_limits.min_h_level[candidate.axis] ||
      throw(ArgumentError("candidate cell $(candidate.cell) violates min_h_level[$(candidate.axis)]"))
    indicator_value = float(indicators[index])
    isfinite(indicator_value) || throw(ArgumentError("h coarsening indicators must be finite"))
    indicator_value <= value || continue
    any(blocked[_active_leaf_index(space, child)] for child in candidate.children) && continue
    push!(selected, candidate)
  end

  return selected
end

# Automatic `h`, `p`, and mixed `hp` plan construction from indicators.

"""
    h_adaptivity_plan(space, indicators; threshold=0.5,
                      h_coarsening_candidates=nothing,
                      h_coarsening_indicators=nothing,
                      h_coarsening_threshold=nothing,
                      limits=AdaptivityLimits(space))
    h_adaptivity_plan(state, field; threshold=0.5, indicator=nothing,
                      h_coarsening_candidates=nothing,
                      h_coarsening_indicator=projection_coarsening_indicators,
                      h_coarsening_threshold=nothing,
                      limits=AdaptivityLimits(field_space(field)))

Build a pure `h`-adaptivity plan with Dörfler bulk refinement marking and
optional projection-based `h`-derefinement on admissible parent cells.

The refinement side marks leaf-axis pairs whose indicator contributions account
for a fraction `threshold` of the total indicator mass. Optional derefinement is
handled separately via explicit parent-cell candidates and projection-defect
indicators. When no custom `indicator` is supplied, the state-based overload
uses modal indicators on CG axes and interface-jump indicators on DG axes, so
the default refinement signal follows the continuity model of the space rather
than forcing one indicator family everywhere.
"""
function h_adaptivity_plan(space::HpSpace{D}, indicators; threshold::Real=0.5,
                           h_coarsening_candidates=nothing, h_coarsening_indicators=nothing,
                           h_coarsening_threshold=nothing, limits=AdaptivityLimits(space)) where {D}
  checked_limits = _checked_limits(limits, space)
  marked = _marked_axes(space, indicators; threshold=threshold,
                        admissible=(leaf, axis) -> _can_h_refine(space, leaf, axis, checked_limits))
  blocked = [any(marked[index]) for index in eachindex(marked)]
  selected = _selected_h_coarsening_candidates(space, h_coarsening_candidates,
                                               h_coarsening_indicators;
                                               threshold=h_coarsening_threshold,
                                               limits=checked_limits, blocked=blocked)
  p_degree_changes = fill(ntuple(_ -> 0, D), active_leaf_count(space))
  return _batched_h_adaptivity_plan(space, p_degree_changes, marked, selected;
                                    limits=checked_limits)
end

function h_adaptivity_plan(state::State, field::AbstractField; threshold::Real=0.5,
                           indicator=nothing, h_coarsening_candidates=nothing,
                           h_coarsening_indicator=projection_coarsening_indicators,
                           h_coarsening_threshold=nothing,
                           limits=AdaptivityLimits(field_space(field)))
  resolved_indicator = isnothing(indicator) ? _default_refinement_indicators : indicator
  candidates, coarsening = _state_h_coarsening_data(state, field, h_coarsening_candidates,
                                                    h_coarsening_indicator, h_coarsening_threshold,
                                                    limits)
  return h_adaptivity_plan(field_space(field),
                           _indicator_values(resolved_indicator, state, field, "indicator");
                           threshold=threshold, h_coarsening_candidates=candidates,
                           h_coarsening_indicators=coarsening,
                           h_coarsening_threshold=h_coarsening_threshold, limits=limits)
end

"""
    p_adaptivity_plan(space, indicators, p_coarsening_indicators=nothing;
                      threshold=0.5, p_coarsening_threshold=nothing,
                      limits=AdaptivityLimits(space))
    p_adaptivity_plan(state, field; threshold=0.5, indicator=coefficient_indicators,
                      p_coarsening_indicator=coefficient_coarsening_indicators,
                      p_coarsening_threshold=nothing,
                      limits=AdaptivityLimits(field_space(field)))

Build a pure `p`-adaptivity plan with Dörfler bulk refinement marking and
optional modal `p`-derefinement on axes whose coarsening indicator falls below
`p_coarsening_threshold`.

Refinement and derefinement are both axiswise. A leaf may therefore be refined
in one coordinate direction of polynomial degree while being left unchanged in
another.
"""
function p_adaptivity_plan(space::HpSpace{D}, indicators, p_coarsening_indicators=nothing;
                           threshold::Real=0.5, p_coarsening_threshold=nothing,
                           limits=AdaptivityLimits(space)) where {D}
  checked_limits = _checked_limits(limits, space)
  marked = _marked_axes(space, indicators; threshold=threshold,
                        admissible=(leaf, axis) -> _can_p_refine(space, leaf, axis, checked_limits))
  coarsened = _coarsened_axes(space, p_coarsening_indicators; threshold=p_coarsening_threshold,
                              admissible=(leaf, axis) -> _can_p_derefine(space, leaf, axis,
                                                                         checked_limits),
                              blocked=marked)
  plan = AdaptivityPlan(space; limits=checked_limits)

  for leaf_index in eachindex(space.active_leaves)
    leaf = space.active_leaves[leaf_index]
    decrements = ntuple(axis -> coarsened[leaf_index][axis] ? 1 : 0, D)
    increments = ntuple(axis -> marked[leaf_index][axis] ? 1 : 0, D)
    any(>(0), decrements) && request_p_derefinement!(plan, leaf, decrements)
    any(>(0), increments) && request_p_refinement!(plan, leaf, increments)
  end

  return plan
end

function p_adaptivity_plan(state::State, field::AbstractField; threshold::Real=0.5,
                           indicator=coefficient_indicators,
                           p_coarsening_indicator=coefficient_coarsening_indicators,
                           p_coarsening_threshold=nothing,
                           limits=AdaptivityLimits(field_space(field)))
  coarsening = _optional_state_indicator_values(p_coarsening_indicator, state, field,
                                                p_coarsening_threshold, "p_coarsening_indicator")
  return p_adaptivity_plan(field_space(field),
                           _indicator_values(indicator, state, field, "indicator"), coarsening;
                           threshold=threshold, p_coarsening_threshold=p_coarsening_threshold,
                           limits=limits)
end

# Mixed `hp` planning decides one marked axis at a time. Smooth axes prefer
# `p` enrichment, rough axes prefer `h` refinement, and active limits trigger a
# fallback to the other option before leaving the axis unchanged.
function _hp_axis_refinement(space::HpSpace, leaf::Int, axis::Int, prefer_p::Bool,
                             limits::AdaptivityLimits)
  if prefer_p
    _can_p_refine(space, leaf, axis, limits) && return 1
    _can_h_refine(space, leaf, axis, limits) && return -1
    return 0
  end

  _can_h_refine(space, leaf, axis, limits) && return -1
  _can_p_refine(space, leaf, axis, limits) && return 1
  return 0
end

# Collapse the mixed `hp` decision for one source leaf into one pair of target
# updates: a tuple of `h`-refinement axes and a tuple of signed `p` changes.
function _hp_leaf_changes(space::HpSpace{D}, leaf::Int, marked_axes, smoothness_axes,
                          smoothness_threshold::Real, coarsened_axes,
                          limits::AdaptivityLimits{D}) where {D}
  choices = ntuple(axis -> marked_axes[axis] ?
                           _hp_axis_refinement(space, leaf, axis,
                                               float(smoothness_axes[axis]) <= smoothness_threshold,
                                               limits) : 0, D)
  h_axes = ntuple(axis -> choices[axis] < 0, D)
  p_degree_changes = ntuple(axis -> (choices[axis] > 0 ? 1 : 0) - (coarsened_axes[axis] ? 1 : 0), D)
  return h_axes, p_degree_changes
end

"""
    hp_adaptivity_plan(space, indicators, smoothness, p_coarsening_indicators=nothing;
                       threshold=0.5, smoothness_threshold=0.5,
                       p_coarsening_threshold=nothing,
                       h_coarsening_candidates=nothing,
                       h_coarsening_indicators=nothing,
                       h_coarsening_threshold=nothing,
                       limits=AdaptivityLimits(space))
    hp_adaptivity_plan(state, field; threshold=0.5, indicator=nothing,
                       smoothness_indicator=nothing,
                       smoothness_threshold=0.5,
                       p_coarsening_indicator=coefficient_coarsening_indicators,
                       p_coarsening_threshold=nothing,
                       h_coarsening_candidates=nothing,
                       h_coarsening_indicator=projection_coarsening_indicators,
                       h_coarsening_threshold=nothing,
                       limits=AdaptivityLimits(field_space(field)))

Build a mixed `hp`-adaptivity plan. Dörfler marking selects candidate axes, the
smoothness indicator chooses between `p`- and `h`-adaptation, and the optional
coarsening indicators enable modal `p`-derefinement on unmarked leaves and
projection-based `h`-derefinement on unchanged parent cells.

The decision logic is local and fallback-based: when the preferred adaptation
type violates the active limits on an axis, the other type is attempted before
the axis is left unchanged. When no custom refinement indicator is supplied,
the state-based overload uses modal indicators on CG axes and interface-jump
indicators on DG axes. Its default smoothness indicator remains modal but
treats DG `p = 0` axes as rough so mixed `hp` planning prefers `h` there.
"""
function hp_adaptivity_plan(space::HpSpace{D}, indicators, smoothness,
                            p_coarsening_indicators=nothing; threshold::Real=0.5,
                            smoothness_threshold::Real=0.5, p_coarsening_threshold=nothing,
                            h_coarsening_candidates=nothing, h_coarsening_indicators=nothing,
                            h_coarsening_threshold=nothing,
                            limits=AdaptivityLimits(space)) where {D}
  checked_limits = _checked_limits(limits, space)
  marked = _marked_axes(space, indicators; threshold=threshold,
                        admissible=(leaf, axis) -> _can_h_refine(space, leaf, axis,
                                                                 checked_limits) ||
                                                   _can_p_refine(space, leaf, axis, checked_limits))
  checked_smoothness = _checked_axis_indicator_values(space, smoothness, "smoothness indicators")
  checked_smoothness_threshold = _checked_nonnegative_threshold(smoothness_threshold,
                                                                "smoothness_threshold")
  coarsened = _coarsened_axes(space, p_coarsening_indicators; threshold=p_coarsening_threshold,
                              admissible=(leaf, axis) -> _can_p_derefine(space, leaf, axis,
                                                                         checked_limits),
                              blocked=marked)
  planned_h = fill(ntuple(_ -> false, dimension(space)), active_leaf_count(space))
  p_degree_changes = fill(ntuple(_ -> 0, dimension(space)), active_leaf_count(space))

  for leaf_index in eachindex(space.active_leaves)
    leaf = space.active_leaves[leaf_index]
    planned_h[leaf_index], p_degree_changes[leaf_index] = _hp_leaf_changes(space, leaf,
                                                                           marked[leaf_index],
                                                                           checked_smoothness[leaf_index],
                                                                           checked_smoothness_threshold,
                                                                           coarsened[leaf_index],
                                                                           checked_limits)
  end

  blocked = _leaf_change_flags(planned_h, p_degree_changes, coarsened)
  selected = _selected_h_coarsening_candidates(space, h_coarsening_candidates,
                                               h_coarsening_indicators;
                                               threshold=h_coarsening_threshold,
                                               limits=checked_limits, blocked=blocked)
  return _batched_h_adaptivity_plan(space, p_degree_changes, planned_h, selected;
                                    limits=checked_limits)
end

function hp_adaptivity_plan(state::State, field::AbstractField; threshold::Real=0.5,
                            indicator=nothing, smoothness_indicator=nothing,
                            smoothness_threshold::Real=0.5,
                            p_coarsening_indicator=coefficient_coarsening_indicators,
                            p_coarsening_threshold=nothing, h_coarsening_candidates=nothing,
                            h_coarsening_indicator=projection_coarsening_indicators,
                            h_coarsening_threshold=nothing,
                            limits=AdaptivityLimits(field_space(field)))
  resolved_indicator = isnothing(indicator) ? _default_refinement_indicators : indicator
  resolved_smoothness = isnothing(smoothness_indicator) ? _default_smoothness_indicators :
                        smoothness_indicator
  p_coarsening = _optional_state_indicator_values(p_coarsening_indicator, state, field,
                                                  p_coarsening_threshold, "p_coarsening_indicator")
  candidates, h_coarsening = _state_h_coarsening_data(state, field, h_coarsening_candidates,
                                                      h_coarsening_indicator,
                                                      h_coarsening_threshold, limits)
  return hp_adaptivity_plan(field_space(field),
                            _indicator_values(resolved_indicator, state, field, "indicator"),
                            _indicator_values(resolved_smoothness, state, field,
                                              "smoothness_indicator"), p_coarsening;
                            threshold=threshold, smoothness_threshold=smoothness_threshold,
                            p_coarsening_threshold=p_coarsening_threshold,
                            h_coarsening_candidates=candidates,
                            h_coarsening_indicators=h_coarsening,
                            h_coarsening_threshold=h_coarsening_threshold, limits=limits)
end

# Source-space reporting helpers and direct leaf evaluation utilities.

# One internal change summary per source leaf keeps the public query helpers and
# summary reporting consistent while avoiding repeated descendant scans.
struct _SourceLeafChange{D}
  h_axes::NTuple{D,Bool}
  p_degree_changes::NTuple{D,Int}
  p_refined::Bool
  p_derefined::Bool
end

function _source_leaf_change(plan::AdaptivityPlan{D}, leaf::Int) where {D}
  source_grid = grid(source_space(plan))
  source_levels = level(source_grid, leaf)
  source_degrees = cell_degrees(source_space(plan), leaf)
  descendants = Int[]
  _collect_active_descendants!(descendants, grid(plan), leaf)
  isempty(descendants) &&
    return _SourceLeafChange(ntuple(_ -> false, D), ntuple(_ -> 0, D), false, false)
  h_axes = fill(false, D)
  p_degree_changes = fill(typemin(Int), D)
  p_refined = false
  p_derefined = false

  for target_leaf in descendants
    target_levels = level(grid(plan), target_leaf)
    target_degrees = cell_degrees(plan, target_leaf)

    for axis in 1:D
      h_axes[axis] |= target_levels[axis] > source_levels[axis]
      change = target_degrees[axis] - source_degrees[axis]
      p_degree_changes[axis] = max(p_degree_changes[axis], change)
      p_refined |= change > 0
      p_derefined |= change < 0
    end
  end

  return _SourceLeafChange(ntuple(axis -> h_axes[axis], D),
                           ntuple(axis -> p_degree_changes[axis], D), p_refined, p_derefined)
end

function _source_refinement_axes(plan::AdaptivityPlan{D}, leaf::Int) where {D}
  return _source_leaf_change(plan, leaf).h_axes
end

"""
    h_adaptation_axes(plan, leaf)

Return the per-axis `h`-refinement signature of a source active leaf in
`plan`.

The returned tuple reports whether any active target descendant of the source
leaf is more refined than the source leaf along each axis.
"""
function h_adaptation_axes(plan::AdaptivityPlan{D}, leaf::Integer) where {D}
  checked_leaf = _checked_cell(grid(source_space(plan)), leaf)
  is_active_leaf(grid(source_space(plan)), checked_leaf) ||
    throw(ArgumentError("leaf $checked_leaf is not an active source leaf"))
  return _source_refinement_axes(plan, checked_leaf)
end

"""
    p_degree_change(plan, leaf)

Return the maximum per-axis target degree change over the active target
descendants of one source active leaf.

Positive entries indicate `p`-refinement somewhere below the source leaf,
negative entries indicate `p`-derefinement, and zero indicates that no target
descendant changes the degree on that axis.
"""
function p_degree_change(plan::AdaptivityPlan{D}, leaf::Integer) where {D}
  checked_leaf = _checked_cell(grid(source_space(plan)), leaf)
  is_active_leaf(grid(source_space(plan)), checked_leaf) ||
    throw(ArgumentError("leaf $checked_leaf is not an active source leaf"))
  return _source_leaf_change(plan, checked_leaf).p_degree_changes
end

"""
    adaptivity_summary(plan)

Return source-space change counts for an `AdaptivityPlan`.

The summary is reported in terms of source leaves and source expanded cells, so
it measures how the planned target configuration differs from the current source
space rather than summarizing raw target-space counts.
"""
function adaptivity_summary(plan::AdaptivityPlan)
  source = source_space(plan)
  target_grid = grid(plan)
  marked_leaf_count = 0
  h_refinement_leaf_count = 0
  p_refinement_leaf_count = 0
  p_derefinement_leaf_count = 0

  for leaf in active_leaves(source)
    change = _source_leaf_change(plan, leaf)
    is_h = any(change.h_axes)
    marked_leaf_count += (is_h || change.p_refined || change.p_derefined)
    h_refinement_leaf_count += is_h
    p_refinement_leaf_count += change.p_refined
    p_derefinement_leaf_count += change.p_derefined
  end

  h_derefinement_cell_count = 0

  for cell in 1:stored_cell_count(grid(source))
    is_expanded(grid(source), cell) || continue
    cell <= stored_cell_count(target_grid) || continue
    is_active_leaf(target_grid, cell) && (h_derefinement_cell_count += 1)
  end

  return (marked_leaf_count=marked_leaf_count, h_refinement_leaf_count=h_refinement_leaf_count,
          h_derefinement_cell_count=h_derefinement_cell_count,
          p_refinement_leaf_count=p_refinement_leaf_count,
          p_derefinement_leaf_count=p_derefinement_leaf_count)
end

# Direct leaf-local field evaluation used by transfer and projection indicators.

# Evaluate a field directly on one compiled leaf by reconstructing its modal
# expansion on reference coordinates. These helpers keep the transfer and
# projection indicator code independent of the global integration machinery.
function _transition_value(transition::SpaceTransition{D,T}, field::AbstractField, state::State{T},
                           target_leaf::Int, x::NTuple{D,<:Real}) where {D,T<:AbstractFloat}
  leaf = _source_leaf_at_point(transition, target_leaf, x)
  ξ = map_to_biunit_cube(domain(source_space(transition)), leaf, x)
  return _field_value_on_leaf(field, state, leaf, ξ)
end

function _field_value_on_leaf(field::ScalarField, state::State{T}, leaf::Int,
                              ξ::NTuple{D,<:Real}) where {D,T<:AbstractFloat}
  return _field_component_value_on_leaf(field, state, leaf, ξ, 1)
end

function _field_value_on_leaf(field::VectorField, state::State{T}, leaf::Int,
                              ξ::NTuple{D,<:Real}) where {D,T<:AbstractFloat}
  return ntuple(component -> _field_component_value_on_leaf(field, state, leaf, ξ, component),
                component_count(field))
end

function _field_component_value_on_leaf(field::AbstractField, state::State{T}, leaf::Int,
                                        ξ::NTuple{D,<:Real},
                                        component::Int) where {D,T<:AbstractFloat}
  compiled = _compiled_leaf(field_space(field), leaf)
  basis_values = ntuple(axis -> _fe_basis_values(T(ξ[axis]), compiled.degrees[axis]), D)
  coefficients = field_component_values(state, field, component)
  result = zero(T)

  for mode_index in eachindex(compiled.local_modes)
    mode = compiled.local_modes[mode_index]
    shape = one(T)

    for axis in 1:D
      shape *= basis_values[axis][mode[axis]+1]
    end

    shape == zero(T) && continue

    amplitude = _term_amplitude(compiled.term_offsets, compiled.term_indices,
                                compiled.term_coefficients, compiled.single_term_indices,
                                compiled.single_term_coefficients, coefficients, mode_index)
    result += shape * amplitude
  end

  return result
end

function _local_mode_amplitude(field::AbstractField, state::State{T}, compiled::_CompiledLeaf,
                               component::Int, mode_index::Int) where {T<:AbstractFloat}
  coefficients = field_component_values(state, field, component)
  return _term_amplitude(compiled.term_offsets, compiled.term_indices, compiled.term_coefficients,
                         compiled.single_term_indices, compiled.single_term_coefficients,
                         coefficients, mode_index)
end

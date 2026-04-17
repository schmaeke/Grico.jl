# This file implements the adaptive mesh/polynomial workflow around `HpSpace`.
# The main responsibilities are:
# 1. describe admissible target discretizations via `AdaptivityPlan`,
# 2. compile source-to-target transfer data via `SpaceTransition`,
# 3. derive problem-independent FE detail indicators, and
# 4. translate those details into concrete `h`, `p`, or mixed `hp` changes.
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
# Third, it introduces problem-independent detail indicators derived from the hp
# basis and from DG traces. These indicators are intentionally generic: they say
# something about local resolution without assuming a specific PDE residual or
# estimator.
#
# Fourth, it turns those details into concrete `h`, `p`, or mixed `hp` changes
# through one multiresolution-style tolerance, one optional h/p smoothness
# threshold, and admissible bounds on h-level and polynomial degree.
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

# Continuous axes need at least linear endpoint modes to represent conforming
# traces, while fully discontinuous axes may use a constant cell mode. This is
# the continuity-aware default when users do not specify `min_p` explicitly.
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

# Validate a user-supplied limit object against the concrete space. Most bounds
# are purely geometric or polynomial, but CG axes have the additional lower
# degree invariant enforced by the space compiler.
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
function _target_leaf_lookup(grid_data::CartesianGrid, active::AbstractVector{<:Integer},
                             degrees::AbstractVector{<:Tuple})
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
# transfer can be defined by overlap on a common geometric domain. For
# `PhysicalDomain`s the region test is intentionally object identity: copied
# domains share one region object and therefore one classification/quadrature
# cache, while independently constructed regions are treated as distinct
# physical descriptions.
function _same_adaptivity_geometry(source_domain::AbstractDomain, target_domain::AbstractDomain)
  root_cell_counts(grid(source_domain)) == root_cell_counts(grid(target_domain)) || return false
  origin(source_domain) == origin(target_domain) || return false
  extent(source_domain) == extent(target_domain) || return false
  periodic_axes(source_domain) == periodic_axes(target_domain) || return false
  _physical_region(source_domain) === _physical_region(target_domain) || return false
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
mutable struct AdaptivityPlan{D,T<:AbstractFloat,S<:HpSpace{D,T},N<:AbstractDomain{D,T},
                              V<:Vector{NTuple{D,Int}},I<:Vector{Int},L<:AdaptivityLimits{D}}
  source_space::S
  target_domain::N
  target_degrees::V
  target_leaf_to_index::I
  limits::L

  function AdaptivityPlan{D,T,S,N,V,I,L}(source_space::S, target_domain::N, target_degrees::V,
                                         target_leaf_to_index::I,
                                         limits::L) where {D,T<:AbstractFloat,S<:HpSpace{D,T},
                                                           N<:AbstractDomain{D,T},
                                                           V<:Vector{NTuple{D,Int}},I<:Vector{Int},
                                                           L<:AdaptivityLimits{D}}
    _same_adaptivity_geometry(domain(source_space), target_domain) ||
      throw(ArgumentError("target domain must share the source geometry and root cell counts"))
    stored_cell_count(grid(target_domain)) == length(target_leaf_to_index) ||
      throw(ArgumentError("target leaf lookup length must match the stored target-cell count"))
    active = _domain_active_leaves(target_domain)
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

function AdaptivityPlan(source_space::HpSpace{D,T}, target_domain::AbstractDomain{D,T},
                        target_degrees::AbstractVector{<:NTuple{D,<:Integer}};
                        limits::AdaptivityLimits{D}=AdaptivityLimits(source_space)) where {D,
                                                                                           T<:AbstractFloat}
  checked_limits = _checked_limits(limits, source_space)
  active = _domain_active_leaves(target_domain)
  length(target_degrees) == length(active) ||
    throw(ArgumentError("target degree data must match the active-leaf count"))
  checked = Vector{NTuple{D,Int}}(undef, length(target_degrees))

  for index in eachindex(target_degrees)
    checked[index] = _checked_space_degree_tuple(target_degrees[index],
                                                 source_space.continuity_policy,
                                                 "target_degrees[$index]")
  end

  lookup = _target_leaf_lookup(grid(target_domain), active, checked)
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
  target = copy(domain(space))
  degrees = _inherited_target_degrees(space, target)
  return AdaptivityPlan(space, target, degrees; limits=limits)
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
active_leaf_count(plan::AdaptivityPlan) = length(plan.target_degrees)
active_leaves(plan::AdaptivityPlan) = _domain_active_leaves(plan.target_domain)
function active_leaf(plan::AdaptivityPlan, index::Integer)
  @inbounds return active_leaves(plan)[_checked_index(index, active_leaf_count(plan),
                                                      "active leaf")]
end

# Check that `leaf` names one active target leaf in `plan`.
function _checked_target_leaf(plan::AdaptivityPlan, leaf::Integer)
  checked_leaf = _checked_cell(grid(plan), leaf)
  checked_leaf <= length(plan.target_leaf_to_index) &&
  plan.target_leaf_to_index[checked_leaf] != 0 ||
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
  plan.target_leaf_to_index = _target_leaf_lookup(grid(target_domain(plan)), active, degrees)
  return plan
end

# Snapshot source degrees before batched topology edits. The dictionary form is
# temporary and intentionally mirrors the target-grid cell numbers while splits
# and collapses insert or remove leaves.
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
function _batched_adaptivity_plan(space::HpSpace{D,T}, p_degree_changes, h_refinement_axes,
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
  active = _domain_active_leaves(target_domain)
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    leaf = active[index]
    haskey(mapping, leaf) || throw(ArgumentError("missing planned degrees for target leaf $leaf"))
    degrees[index] = mapping[leaf]
  end

  return AdaptivityPlan(space, target_domain, degrees; limits=limits)
end

# These predicates encode admissibility against the active limits only. Keeping
# them separate from indicator logic makes fallback decisions explicit and keeps
# plan construction from silently exceeding user-specified h/p bounds.
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

# Merged h-coarsening parents inherit the componentwise maximum child degree.
# This avoids losing polynomial order as a side effect of removing one mesh
# split; any p-derefinement is planned separately by the modal detail logic.
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

# Normalize a user- or planner-provided candidate list before applying it. This
# keeps the batched plan builder independent of where the candidates came from.
function _checked_h_coarsening_candidates(space::HpSpace{D}, candidates) where {D}
  checked = Vector{HCoarseningCandidate{D}}(undef, length(candidates))

  for index in eachindex(candidates)
    candidates[index] isa HCoarseningCandidate{D} ||
      throw(ArgumentError("h coarsening candidates must be HCoarseningCandidate{$D} values"))
    checked[index] = _checked_h_coarsening_candidate(space, candidates[index])
  end

  return checked
end

# Apply the topological inverse of one dyadic split to the transient degree map.
# The caller is responsible for candidate validation and for rebuilding active
# leaf lookup data after all batched edits have been applied.
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

# A derived plan may use a different source space, but it must live on the same
# physical geometry and scalar type as the driver plan. Otherwise the copied
# target topology could not be interpreted as the same adapted domain.
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
function _inherited_target_degrees(space::HpSpace{D}, target_domain::AbstractDomain{D},
                                   active::AbstractVector{<:Integer}=_domain_active_leaves(target_domain)) where {D}
  source_grid = grid(space)
  target_grid = grid(target_domain)
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    source_leaves = _transition_source_leaves(source_grid, target_grid, active[index])
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

The field kind, component count, and semantic field identity are preserved; only
the owning space and, optionally, the symbolic field name change. This means an
adapted field represents the same unknown on a new discrete space, so layout and
state lookup by the original field descriptor remain valid after transfer.
"""
function adapted_field(transition::SpaceTransition, field::ScalarField;
                       name::Symbol=field_name(field))
  return ScalarField(_field_id(field), target_space(transition), name)
end

function adapted_field(transition::SpaceTransition, field::VectorField;
                       name::Symbol=field_name(field))
  return VectorField(_field_id(field), target_space(transition), component_count(field), name)
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
  return ntuple(index -> adapted_field(transition, fields[index]), length(fields))
end

function adapted_fields(transition::SpaceTransition, layout::FieldLayout)
  adapted_fields(transition, fields(layout))
end

function adapted_fields(transition::SpaceTransition, first_field::AbstractField,
                        remaining_fields::AbstractField...)
  return adapted_fields(transition, (first_field, remaining_fields...))
end

# State transfer is formulated as an `L²` projection on the target space. The
# local mass operator below provides the symmetric positive definite block that
# defines the target inner product for one field.
struct _TransferMass{F}
  field::F
end

# The transfer source operator evaluates the old state on target quadrature
# points and injects those values into the right-hand side of the projection
# system. Together with `_TransferMass` this yields the Galerkin `L²`
# projection from the source field to the target field.
struct _TransferSource{F,C,TR}
  field::F
  old_coefficients::C
  transition::TR
end

# Compiled setup for the fully-DG transfer path. It mirrors the affine fallback
# inputs but keeps only target-cell integration data and the maximum dense block
# size needed for per-worker scratch storage.
struct _CellwiseDGTransferPlan{D,T<:AbstractFloat,L,C,O,N,TR}
  layout::L
  cells::C
  old_fields::O
  new_fields::N
  transition::TR
  max_local_dofs::Int
end

# Dense local buffers are reused by one worker at a time while solving cellwise
# projection systems. Fully-DG cells have disjoint global dofs, so the final
# coefficient writes do not need an accumulator.
mutable struct _CellwiseDGTransferScratch{D,T<:AbstractFloat}
  matrix::Matrix{T}
  rhs::Vector{T}
  source_basis::NTuple{D,Vector{T}}
end

function _CellwiseDGTransferScratch(::Type{T}, ::Val{D},
                                    local_dof_count::Int) where {D,T<:AbstractFloat}
  return _CellwiseDGTransferScratch{D,T}(Matrix{T}(undef, local_dof_count, local_dof_count),
                                         Vector{T}(undef, local_dof_count), ntuple(_ -> T[], D))
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

function cell_matrix!(local_matrix, operator::_TransferMass, values::CellValues)
  _assemble_transfer_mass!(local_matrix, values, operator.field)
  return nothing
end

# State transfer repeatedly asks whether a target quadrature point belongs to a
# candidate source leaf. The tolerance keeps roundoff near shared faces from
# creating spurious "point not found" failures.
function _point_in_cell(domain_data::AbstractDomain{D,T}, leaf::Int, x::NTuple{D,<:Real};
                        tolerance::T=T(1.0e-12)) where {D,T<:AbstractFloat}
  lower = cell_lower(domain_data, leaf)
  upper = cell_upper(domain_data, leaf)
  return all(lower[axis] - tolerance <= x[axis] <= upper[axis] + tolerance for axis in 1:D)
end

# Locate the source leaf that supplies a value at a target quadrature point.
# The transition precomputes the small overlap set, so this search is local to
# one target leaf instead of scanning the full source mesh.
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
function _add_transfer_rhs_at_point!(local_rhs, values::CellValues, old_coefficients,
                                     new_field::AbstractField, source_compiled::_CompiledLeaf{D,T},
                                     source_basis::NTuple{D,Vector{T}}, point_index::Int,
                                     weighted) where {D,T<:AbstractFloat}
  mode_count = local_mode_count(values, new_field)

  for component in 1:component_count(new_field)
    old_value = _leaf_component_value(source_compiled, old_coefficients[component], source_basis)
    weighted_value = old_value * weighted

    for mode_index in 1:mode_count
      shape = shape_value(values, new_field, point_index, mode_index)
      shape == 0 && continue
      row = local_dof_index(values, new_field, component, mode_index)
      local_rhs[row] += weighted_value * shape
    end
  end

  return local_rhs
end

function _assemble_transfer_rhs!(local_rhs, values::CellValues, transition::SpaceTransition{D,T},
                                 new_field::AbstractField,
                                 old_coefficients) where {D,T<:AbstractFloat}
  source_space_data = source_space(transition)
  source_domain = domain(source_space_data)
  source_basis = _LeafBasisScratch(T, Val(D)).values

  for point_index in 1:point_count(values)
    x = point(values, point_index)
    source_leaf = _source_leaf_at_point(transition, values.leaf, x)
    ξ = map_to_biunit_cube(source_domain, source_leaf, x)
    source_compiled = _compiled_leaf(source_space_data, source_leaf)
    _fill_leaf_basis!(source_basis, source_compiled.degrees, ξ)
    weighted = weight(values, point_index)
    _add_transfer_rhs_at_point!(local_rhs, values, old_coefficients, new_field, source_compiled,
                                source_basis, point_index, weighted)
  end

  return local_rhs
end

function cell_rhs!(local_rhs, operator::_TransferSource, values::CellValues)
  _assemble_transfer_rhs!(local_rhs, values, operator.transition, operator.field,
                          operator.old_coefficients)
  return nothing
end

# When the target space is fully DG, no local mode shares coefficients across
# cells. The transfer mass matrix is therefore block diagonal with one dense
# block per active target cell, so it is cheaper to solve those local systems
# directly than to assemble one global sparse projection problem.
_is_fully_dg_space(space::HpSpace) = all(kind -> kind === :dg, continuity_policy(space))

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

@inline function _cellwise_transfer_linear_solve(local_matrix, local_rhs, linear_solve)
  if linear_solve === default_linear_solve
    return local_matrix \ local_rhs
  end

  return linear_solve(local_matrix, local_rhs)
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
                                          plan::_CellwiseDGTransferPlan{D,T}, source_coefficients,
                                          source_basis::NTuple{D,Vector{T}}) where {D,
                                                                                    T<:AbstractFloat}
  transition = plan.transition
  source_space_data = source_space(transition)
  source_domain = domain(source_space_data)

  for point_index in 1:point_count(cell)
    x = point(cell, point_index)
    source_leaf = _source_leaf_at_point(transition, cell.leaf, x)
    ξ = map_to_biunit_cube(source_domain, source_leaf, x)
    source_compiled = _compiled_leaf(source_space_data, source_leaf)
    _fill_leaf_basis!(source_basis, source_compiled.degrees, ξ)
    weighted = weight(cell, point_index)

    for field_index in eachindex(plan.old_fields)
      new_field = plan.new_fields[field_index]
      _add_transfer_rhs_at_point!(local_rhs, cell, source_coefficients[field_index], new_field,
                                  source_compiled, source_basis, point_index, weighted)
    end
  end

  return local_rhs
end

# Solve one independent dense projection system per target DG cell and scatter
# the local coefficients directly into the new state vector.
function _transfer_cellwise_dg_state(plan::_CellwiseDGTransferPlan{D,T}, state::State{T};
                                     linear_solve=default_linear_solve) where {D,T<:AbstractFloat}
  state_coefficients = zeros(T, dof_count(plan.layout))
  isempty(plan.cells) && return State(plan.layout, state_coefficients)
  worker_count = _worker_count(length(plan.cells), 0, 0, 0)
  scratch = [_CellwiseDGTransferScratch(T, Val(D), plan.max_local_dofs) for _ in 1:worker_count]
  source_coefficients = ntuple(index -> _component_coefficient_views(state, plan.old_fields[index]),
                               length(plan.old_fields))

  _run_chunks!(scratch, length(plan.cells), _WORKLOAD_REGULAR) do cache, first_cell, last_cell
    for cell_index in first_cell:last_cell
      cell = @inbounds plan.cells[cell_index]
      local_dofs = cell.local_dof_count
      matrix_view = view(cache.matrix, 1:local_dofs, 1:local_dofs)
      rhs_view = view(cache.rhs, 1:local_dofs)
      fill!(matrix_view, zero(T))
      fill!(rhs_view, zero(T))
      _assemble_cellwise_transfer_matrix!(matrix_view, cell, plan.new_fields)
      _assemble_cellwise_transfer_rhs!(rhs_view, cell, plan, source_coefficients,
                                       cache.source_basis)
      local_solution = _cellwise_transfer_linear_solve(matrix_view, rhs_view, linear_solve)

      for local_dof in 1:local_dofs
        state_coefficients[cell.single_term_indices[local_dof]] = local_solution[local_dof]
      end
    end
  end

  return State(plan.layout, state_coefficients)
end

"""
    transfer_state(transition, state, old_fields, new_fields; linear_solve=default_linear_solve)
    transfer_state(transition, state, old_field, new_field; linear_solve=default_linear_solve)
    transfer_state(transition, state; linear_solve=default_linear_solve)

Transfer field coefficients from the source space to the target space of
`transition` by cellwise `L²` projection.

The first forms project explicitly paired old/new fields. The zero-argument
field form recreates the full field layout on the target space via
[`adapted_fields`](@ref) and returns both the new fields and the transferred
state.

The transfer is purely geometric and variational: it depends on the old state,
the source/target spaces, and the chosen linear solve path, but not on any
specific PDE operator. On fully discontinuous target spaces, the transfer
recognizes that the projection system is cellwise block diagonal and solves one
local dense system per target cell instead of assembling a global sparse
problem.
"""
function transfer_state(transition::SpaceTransition, state::State, old_fields::Tuple,
                        new_fields::Tuple; linear_solve=default_linear_solve)
  _checked_transfer_fields(transition, state, old_fields, new_fields)

  if _is_fully_dg_space(target_space(transition))
    plan = _compile_cellwise_dg_transfer_plan(transition, old_fields, new_fields)
    return _transfer_cellwise_dg_state(plan, state; linear_solve=linear_solve)
  end

  problem = AffineProblem(new_fields...)

  for index in eachindex(old_fields)
    new_field = new_fields[index]
    add_cell!(problem, _TransferMass(new_field))
    add_cell!(problem,
              _TransferSource(new_field, _component_coefficient_views(state, old_fields[index]),
                              transition))
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
    _domain_active_leaves(reference_target) == _domain_active_leaves(target_domain(plan)) ||
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

# Modal, trace, and projection detail indicators.

function _local_mode_amplitude(compiled::_CompiledLeaf, coefficients::AbstractVector,
                               mode_index::Int)
  return _term_amplitude(compiled.term_offsets, compiled.term_indices, compiled.term_coefficients,
                         compiled.single_term_indices, compiled.single_term_coefficients,
                         coefficients, mode_index)
end

function _local_mode_energy(component_coefficients, compiled::_CompiledLeaf{D,T},
                            mode_index::Int) where {D,T<:AbstractFloat}
  energy = zero(T)

  for component in eachindex(component_coefficients)
    amplitude = _local_mode_amplitude(compiled, component_coefficients[component], mode_index)
    energy += amplitude * amplitude
  end

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
function _axis_layer_energies(component_coefficients, compiled::_CompiledLeaf{D,T},
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

      for component in eachindex(component_coefficients)
        lower = _local_mode_amplitude(compiled, component_coefficients[component], lower_index)
        upper = _local_mode_amplitude(compiled, component_coefficients[component], upper_index)
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
    amplitude_squared = _local_mode_energy(component_coefficients, compiled, mode_index)

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

# Interface-jump evaluation keeps one indicator table per worker plus reusable
# basis buffers for both traces of the current face patch. The accumulated
# tables are merged after the parallel pass, so no atomics are needed in the
# inner quadrature loop.
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

function _interface_jump_quadrature_table(::Type{T}, space::HpSpace{D},
                                          specs) where {D,T<:AbstractFloat}
  cache = Dict{Tuple{Int,NTuple{D-1,Int}},TensorQuadrature{D-1,T}}()

  for spec in specs
    minus_leaf, axis, plus_leaf = spec
    minus_compiled = @inbounds space.compiled_leaves[space.leaf_to_index[minus_leaf]]
    plus_compiled = @inbounds space.compiled_leaves[space.leaf_to_index[plus_leaf]]
    shape = _interface_jump_shape(minus_compiled, plus_compiled, axis)
    key = (axis, _face_tangential_shape(shape, axis))
    haskey(cache, key) || (cache[key] = TensorQuadrature(T, key[2]))
  end

  return cache
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
      difference = plus_value - minus_value
      point_jump += difference * difference
    end

    jump_energy += point_jump * weight(quadrature, point_index) * weight_scale
  end

  return jump_energy
end

function _merge_thread_axis_indicators!(target::Matrix{T}, source::Matrix{T}) where {T}
  size(target) == size(source) ||
    throw(ArgumentError("thread-local indicator tables must have matching sizes"))

  @inbounds for index in eachindex(target)
    target[index] += source[index]
  end

  return target
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
  specs = _filtered_upper_face_neighbor_specs(grid(space), space.active_leaves, space.leaf_to_index)
  isempty(specs) && return [ntuple(_ -> zero(T), D) for _ in 1:active_leaf_total]
  domain_data = domain(space)
  component_coefficients = _component_coefficient_views(state, field)
  quadrature_table = _interface_jump_quadrature_table(T, space, specs)
  worker_count = max(1, min(Threads.nthreads(), length(specs)))
  thread_scratch = [_InterfaceJumpScratch(T, Val(D), active_leaf_total) for _ in 1:worker_count]

  _run_chunks_with_scratch!(thread_scratch, length(specs)) do scratch, first_spec, last_spec
    for spec_index in first_spec:last_spec
      minus_leaf, axis, plus_leaf = specs[spec_index]
      minus_leaf_index = @inbounds space.leaf_to_index[minus_leaf]
      plus_leaf_index = @inbounds space.leaf_to_index[plus_leaf]
      minus_compiled = @inbounds space.compiled_leaves[minus_leaf_index]
      plus_compiled = @inbounds space.compiled_leaves[plus_leaf_index]
      shape = _interface_jump_shape(minus_compiled, plus_compiled, axis)
      quadrature = quadrature_table[(axis, _face_tangential_shape(shape, axis))]
      jump_energy = _interface_jump_energy(component_coefficients, minus_compiled, plus_compiled,
                                           domain_data, minus_leaf, plus_leaf, axis, quadrature,
                                           scratch)
      minus_scale = ((minus_compiled.degrees[axis] + 1)^2) /
                    cell_size(domain_data, minus_leaf, axis)
      plus_scale = ((plus_compiled.degrees[axis] + 1)^2) / cell_size(domain_data, plus_leaf, axis)
      scratch.indicators[minus_leaf_index, axis] += minus_scale * jump_energy
      scratch.indicators[plus_leaf_index, axis] += plus_scale * jump_energy
    end
  end

  indicators = thread_scratch[1].indicators

  for worker in 2:length(thread_scratch)
    _merge_thread_axis_indicators!(indicators, thread_scratch[worker].indicators)
  end

  return [ntuple(axis -> sqrt(indicators[leaf_index, axis]), D)
          for leaf_index in 1:active_leaf_total]
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

# Thread-local scratch for modal detail evaluation. The arrays are indexed by
# logical axis and reused across leaf chunks to avoid per-leaf temporary
# allocation.
struct _ModalAxisDetailScratch{T<:AbstractFloat}
  top::Vector{T}
  previous::Vector{T}
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
  thread_scratch = [_ModalAxisDetailScratch(zeros(T, D), zeros(T, D))
                    for _ in 1:max(1, min(Threads.nthreads(), length(space.active_leaves)))]

  _run_chunks_with_scratch!(thread_scratch,
                            length(space.active_leaves)) do scratch, first_leaf, last_leaf
    for leaf_index in first_leaf:last_leaf
      compiled = space.compiled_leaves[leaf_index]
      fill!(scratch.top, zero(T))
      fill!(scratch.previous, zero(T))
      total_energy = zero(T)

      for mode_index in eachindex(compiled.local_modes)
        amplitude_squared = _local_mode_energy(component_coefficients, compiled, mode_index)
        amplitude_squared == zero(T) && continue
        total_energy += amplitude_squared
      end

      for axis in 1:D
        scratch.top[axis], scratch.previous[axis] = _axis_layer_energies(component_coefficients,
                                                                         compiled, axis)
      end

      detail[leaf_index] = total_energy == zero(T) ? ntuple(_ -> zero(T), D) :
                           ntuple(axis -> sqrt(scratch.top[axis] / total_energy), D)
      decay[leaf_index] = ntuple(axis -> _modal_decay_value(scratch.top[axis],
                                                            scratch.previous[axis]), D)
    end
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

# Build the per-candidate projection references while sharing factorizations for
# repeated target degree tuples. The returned maximum mode count sizes the
# thread-local scratch buffers used during candidate scoring.
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

# Thread-local projection buffers. Columns correspond to field components and
# rows to parent modes; each candidate uses leading subviews sized by its target
# basis, so the buffers can be reused without allocation across chunks.
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

function _projection_quadrature_table(::Type{T}, space::HpSpace{D},
                                      candidates::AbstractVector{HCoarseningCandidate{D}}) where {D,
                                                                                                  T<:AbstractFloat}
  cache = Dict{Tuple{NTuple{D,Int},NTuple{D,Int}},TensorQuadrature{D,T}}()

  for candidate in candidates
    for child in candidate.children
      key = (candidate.target_degrees, cell_degrees(space, child))
      haskey(cache, key) || (cache[key] = TensorQuadrature(T,
                                                           _projection_quadrature_shape(candidate.target_degrees,
                                                                                        key[2])))
    end
  end

  return cache
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
  components = component_count(field)
  component_coefficients = _component_coefficient_views(state, field)
  references, max_mode_count = _projection_reference_table(space, checked_candidates)
  quadrature_table = _projection_quadrature_table(T, space, checked_candidates)
  worker_count = max(1, min(Threads.nthreads(), length(checked_candidates)))
  scratch = [_ProjectionIndicatorScratch(T, Val(dimension(space)), max_mode_count, components)
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
        shape_values = @view buffers.shape_values[1:mode_count]
        fill!(rhs, zero(T))
        fine_norm = zero(T)
        parent_jacobian = jacobian_determinant_from_biunit_cube(domain(space), candidate.cell)

        for child in candidate.children
          child_compiled = _compiled_leaf(space, child)
          quadrature = quadrature_table[(candidate.target_degrees, child_compiled.degrees)]
          child_jacobian = jacobian_determinant_from_biunit_cube(domain(space), child)

          for point_index in 1:point_count(quadrature)
            ξ_child = point(quadrature, point_index)
            ξ_parent = _child_point_in_parent(space, candidate.cell, child, ξ_child)
            _fill_leaf_basis!(buffers.parent_basis, candidate.target_degrees, ξ_parent)
            _fill_leaf_basis!(buffers.child_basis, child_compiled.degrees, ξ_child)
            _leaf_component_values!(buffers.component_values, child_compiled,
                                    component_coefficients, buffers.child_basis)
            _fill_projection_shape_values!(shape_values, modes, buffers.parent_basis)
            weighted = weight(quadrature, point_index) * child_jacobian

            for component in 1:components
              scalar_value = buffers.component_values[component]
              fine_norm += scalar_value * scalar_value * weighted
              weighted_value = scalar_value * weighted

              for mode_index in 1:mode_count
                rhs[mode_index, component] += shape_values[mode_index] * weighted_value
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

  for (leaf_index, leaf) in enumerate(active_leaves(space))
    compiled = space.compiled_leaves[leaf_index]
    quadrature = TensorQuadrature(T, compiled.quadrature_shape)
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

    energies[leaf_index] = energy
  end

  return energies
end

# Convert cell energies into robust normalization denominators. The global
# floor prevents zero-state cells from producing infinite normalized DG jumps
# while still scaling with the magnitude of the represented field.
function _cell_l2_denominators(energies::AbstractVector{T}) where {T<:AbstractFloat}
  global_energy = sum(energies)
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

  for (leaf_index, leaf) in enumerate(active_leaves(space))
    result[leaf_index] = ntuple(axis -> raw[leaf_index][axis] *
                                        cell_size(domain(space), leaf, axis) /
                                        denominators[leaf_index], dimension(space))
  end

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
# and VTK output as well as by the planner, so it deliberately avoids evaluating
# projection defects for h-coarsening candidates.
function _multiresolution_refinement_data(state::State{T},
                                          field::AbstractField) where {T<:AbstractFloat}
  space = field_space(field)
  modal = _modal_axis_detail_data(state, field)
  p_detail = modal.detail
  h_refinement_detail = _continuity_refinement_detail(state, field, p_detail)
  refinement_detail = _combined_refinement_detail(space, p_detail, h_refinement_detail)
  return (; p_detail, modal_decay=modal.decay, h_refinement_detail, refinement_detail)
end

# Full planning data extends diagnostic refinement details with the projection
# defects needed for h-coarsening. This is intentionally separate from
# `multiresolution_indicators`, because VTK output should not pay for candidate
# projection solves.
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

  for (leaf_index, leaf) in enumerate(active_leaves(space))
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

  for (leaf_index, axes) in enumerate(h_refined)
    any(axes) || continue
    leaf = active_leaf(space, leaf_index)
    leaf_levels = level(grid_data, leaf)

    for face_axis in 1:D
      for side in (LOWER, UPPER)
        for neighbor_leaf in opposite_active_leaves(grid_data, leaf, face_axis, side)
          neighbor_index = _active_leaf_index(space, neighbor_leaf)
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
  grid_data = grid(space)

  for (leaf_index, is_blocked) in enumerate(copy(blocked))
    is_blocked || continue
    leaf = active_leaf(space, leaf_index)

    for axis in 1:D
      for side in (LOWER, UPPER)
        for neighbor_leaf in opposite_active_leaves(grid_data, leaf, axis, side)
          blocked[_active_leaf_index(space, neighbor_leaf)] = true
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

  for (leaf_index, leaf) in enumerate(active_leaves(space))
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

# Source-space reporting helpers.

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
  h_axes = fill(false, D)
  p_degree_changes = fill(typemin(Int), D)
  p_refined = false
  p_derefined = false
  has_target_descendant = false

  for target_leaf in descendants
    target_leaf <= length(plan.target_leaf_to_index) || continue
    @inbounds plan.target_leaf_to_index[target_leaf] == 0 && continue
    has_target_descendant = true
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

  has_target_descendant ||
    return _SourceLeafChange(ntuple(_ -> false, D), ntuple(_ -> 0, D), false, false)
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
    cell <= length(plan.target_leaf_to_index) &&
      @inbounds(plan.target_leaf_to_index[cell] != 0) &&
      (h_derefinement_cell_count += 1)
  end

  return (marked_leaf_count=marked_leaf_count, h_refinement_leaf_count=h_refinement_leaf_count,
          h_derefinement_cell_count=h_derefinement_cell_count,
          p_refinement_leaf_count=p_refinement_leaf_count,
          p_derefinement_leaf_count=p_derefinement_leaf_count)
end

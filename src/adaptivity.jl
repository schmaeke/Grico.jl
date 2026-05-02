# This source group implements the adaptive mesh/polynomial workflow around `HpSpace`.
# The main responsibilities are:
# 1. describe admissible target discretizations via `AdaptivityPlan`,
# 2. compile source-to-target transfer data via `SpaceTransition`,
# 3. derive problem-independent FE detail indicators, and
# 4. translate those details into concrete `h`, `p`, or mixed `hp` changes.
#
# The guiding design choice is to keep "planning a new space" separate from
# "building that space". An `AdaptivityPlan` therefore stores a target snapshot
# and target per-leaf degree data, while the actual target `HpSpace` is built
# later by [`transition`](@ref). This keeps interactive plan editing cheap and
# postpones the more expensive continuity compilation to the point where the
# target space is really needed.
#
# Conceptually, the implementation contains four connected chapters.
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
# The source files are arranged in that same order: `adaptivity.jl` owns manual
# plan editing and source-space reporting, `transition.jl` compiles the bridge to
# a target space, `transfer.jl` moves state data, and `indicators.jl` contains the
# advanced automatic indicator/planner policy.
#
# Stability: manual plan editing, transition construction, field recreation,
# and state transfer are the stable adaptivity core. Indicator functions and
# the single-tolerance automatic planner are advanced policy helpers: their
# public entry points are supported, but their numerical heuristics may be
# refined as more application evidence is collected.

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

@inline function _checked_int_representable(value::Integer, name::AbstractString)
  typemin(Int) <= value <= typemax(Int) ||
    throw(ArgumentError("$name must be an Int-representable integer"))
  return Int(value)
end

# Degree offsets for derived plans follow the same scalar-or-tuple convention as
# adaptivity limits, but offsets may be negative because companion spaces such
# as pressure often track a driver space with a lower polynomial order.
function _degree_offset_tuple(value, D::Int)
  if value isa Integer
    checked = _checked_int_representable(value, "degree_offset")
    return ntuple(_ -> checked, D)
  end

  value isa Tuple || throw(ArgumentError("degree_offset must be an integer or NTuple{$D,Int}"))
  length(value) == D || throw(ArgumentError("degree_offset must have length $D"))

  return ntuple(axis -> begin
                  component = value[axis]
                  component isa Integer ||
                    throw(ArgumentError("degree_offset[$axis] must be an integer"))
                  _checked_int_representable(component, "degree_offset[$axis]")
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
function _collect_active_descendants!(leaves::Vector{Int}, grid_snapshot::GridSnapshot, cell::Int)
  is_tree_cell(grid_snapshot, cell) || return leaves

  if is_active_leaf(grid_snapshot, cell)
    push!(leaves, cell)
    return leaves
  end

  first = _snapshot_first_child(grid_snapshot, cell)
  first == NONE && return leaves

  for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
    _collect_active_descendants!(leaves, grid_snapshot, child)
  end

  return leaves
end

# Adaptivity plans may only change refinement and degree data, not the physical
# geometry, cell-measure policy, or root-grid layout. This helper enforces that
# source and target domains describe the same physical box and periodic
# topology, so state transfer can be defined by overlap on a common geometric
# domain. For `PhysicalDomain`s the region test is intentionally object
# identity: spaces on the same physical description share one region object and
# therefore one classification/quadrature cache, while independently constructed
# regions are treated as distinct physical descriptions.
function _same_adaptivity_geometry(source_domain::AbstractDomain, target_domain::AbstractDomain)
  root_cell_counts(grid(source_domain)) == root_cell_counts(grid(target_domain)) || return false
  origin(source_domain) == origin(target_domain) || return false
  extent(source_domain) == extent(target_domain) || return false
  periodic_axes(source_domain) == periodic_axes(target_domain) || return false
  _physical_region(source_domain) === _physical_region(target_domain) || return false
  _cell_measure(source_domain) == _cell_measure(target_domain) || return false
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

An `AdaptivityPlan` stores the target domain, an immutable target
`GridSnapshot`, and the target degree tuple on each snapshot leaf, but it does
not immediately build the target `HpSpace`. This separation is important:
repeated plan edits such as `request_h_refinement!` and `request_p_refinement!`
append missing tree data, produce a new target snapshot, and update per-leaf
degree metadata, while the expensive continuity compilation is deferred to
[`transition`](@ref).

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
                              GS<:GridSnapshot{D},V<:Vector{NTuple{D,Int}},I<:Vector{Int},
                              L<:AdaptivityLimits{D}}
  source_space::S
  target_domain::N
  target_snapshot::GS
  target_degrees::V
  target_leaf_to_index::I
  limits::L

  function AdaptivityPlan{D,T,S,N,GS,V,I,L}(source_space::S, target_domain::N, target_snapshot::GS,
                                            target_degrees::V, target_leaf_to_index::I,
                                            limits::L) where {D,T<:AbstractFloat,S<:HpSpace{D,T},
                                                              N<:AbstractDomain{D,T},
                                                              GS<:GridSnapshot{D},
                                                              V<:Vector{NTuple{D,Int}},
                                                              I<:Vector{Int},L<:AdaptivityLimits{D}}
    _same_adaptivity_geometry(domain(source_space), target_domain) ||
      throw(ArgumentError("target domain must share the source geometry and root cell counts"))
    grid(target_snapshot) === grid(target_domain) ||
      throw(ArgumentError("target snapshot must reference the target domain grid"))
    _require_current_snapshot(target_snapshot)
    target_leaf_to_index === target_snapshot.leaf_to_index ||
      throw(ArgumentError("target leaf lookup must be owned by the target snapshot"))
    active = target_snapshot.active_leaves
    length(target_degrees) == length(active) ||
      throw(ArgumentError("target degree data must match the active-leaf count"))

    for index in eachindex(active)
      leaf = active[index]
      target_leaf_to_index[leaf] == index ||
        throw(ArgumentError("target leaf lookup is inconsistent"))
      _check_target_leaf(grid(target_domain), leaf, target_degrees[index], limits)
    end

    return new{D,T,S,N,GS,V,I,L}(source_space, target_domain, target_snapshot, target_degrees,
                                 target_leaf_to_index, limits)
  end
end

function AdaptivityPlan(source_space::HpSpace{D,T}, target_domain::AbstractDomain{D,T},
                        target_snapshot::GridSnapshot{D},
                        target_degrees::AbstractVector{<:NTuple{D,<:Integer}};
                        limits::AdaptivityLimits{D}=AdaptivityLimits(source_space)) where {D,
                                                                                           T<:AbstractFloat}
  checked_limits = _checked_limits(limits, source_space)
  active = target_snapshot.active_leaves
  length(target_degrees) == length(active) ||
    throw(ArgumentError("target degree data must match the active-leaf count"))
  checked = Vector{NTuple{D,Int}}(undef, length(target_degrees))

  for index in eachindex(target_degrees)
    checked[index] = _checked_space_degree_tuple(target_degrees[index],
                                                 source_space.continuity_policy)
  end

  lookup = target_snapshot.leaf_to_index
  for (index, leaf) in enumerate(active)
    _check_target_leaf(grid(target_domain), leaf, checked[index], checked_limits)
  end

  return AdaptivityPlan{D,T,typeof(source_space),typeof(target_domain),typeof(target_snapshot),
                        typeof(checked),typeof(lookup),typeof(checked_limits)}(source_space,
                                                                               target_domain,
                                                                               target_snapshot,
                                                                               checked, lookup,
                                                                               checked_limits)
end

function AdaptivityPlan(space::HpSpace{D,T};
                        limits::AdaptivityLimits{D}=AdaptivityLimits(space)) where {D,
                                                                                    T<:AbstractFloat}
  target = domain(space)
  target_snapshot = snapshot(space)
  degrees = _inherited_target_degrees(space, target_snapshot)
  return AdaptivityPlan(space, target, target_snapshot, degrees; limits=limits)
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
    target_snapshot(plan)

Return the target domain or active-frontier snapshot stored in `plan`.

The domain owns the shared append-only grid and physical geometry. The snapshot
carries the tentative active `h`-adapted topology. The target polynomial degrees
are stored separately in `plan.target_degrees`.
"""
target_domain(plan::AdaptivityPlan) = plan.target_domain
domain(plan::AdaptivityPlan) = plan.target_domain
target_snapshot(plan::AdaptivityPlan) = plan.target_snapshot
grid(plan::AdaptivityPlan) = grid(plan.target_domain)
active_leaf_count(plan::AdaptivityPlan) = length(plan.target_degrees)
active_leaves(plan::AdaptivityPlan) = active_leaves(plan.target_snapshot)

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
# than target object identity.
function Base.isempty(plan::AdaptivityPlan)
  source = source_space(plan)
  source_active = snapshot(source).active_leaves
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

  for axis in 1:D
    increments[axis] <= plan.limits.max_p[axis] - degrees[axis] ||
      throw(ArgumentError("leaf $checked_leaf reached max_p[$axis]"))
  end

  updated = ntuple(axis -> degrees[axis] + increments[axis], D)
  plan.target_degrees[index] = updated
  return plan
end

# Update one active target leaf by positive `p` decrements, subject to `min_p`.
function _apply_p_derefinement!(plan::AdaptivityPlan{D}, leaf::Integer,
                                decrements::NTuple{D,Int}) where {D}
  checked_leaf = _checked_target_leaf(plan, leaf)
  index = @inbounds plan.target_leaf_to_index[checked_leaf]
  degrees = @inbounds plan.target_degrees[index]

  for axis in 1:D
    decrements[axis] <= degrees[axis] - plan.limits.min_p[axis] ||
      throw(ArgumentError("leaf $checked_leaf reached min_p[$axis]"))
  end

  updated = ntuple(axis -> degrees[axis] - decrements[axis], D)
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

# Rebuild the target degree vector after snapshot topology changes. The topology
# layer owns active-frontier construction; adaptivity only filters it through
# domain activity and aligns the degree vector with the resulting snapshot.
function _reset_plan_degrees!(plan::AdaptivityPlan{D}, mapping::Dict{Int,NTuple{D,Int}},
                              target_snapshot::GridSnapshot{D}) where {D}
  plan.target_snapshot = _filter_target_snapshot!(target_domain(plan), target_snapshot, mapping)
  plan.target_leaf_to_index = plan.target_snapshot.leaf_to_index
  active = plan.target_snapshot.active_leaves
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    leaf = active[index]
    haskey(mapping, leaf) || throw(ArgumentError("missing planned degrees for target leaf $leaf"))
    degrees[index] = mapping[leaf]
  end

  plan.target_degrees = degrees
  return plan
end

function _filter_target_snapshot!(target_domain::AbstractDomain{D},
                                  target_snapshot::GridSnapshot{D},
                                  mapping::Dict{Int,NTuple{D,Int}}) where {D}
  grid(target_snapshot) === grid(target_domain) ||
    throw(ArgumentError("target snapshot must reference the target domain grid"))
  active = Int[]

  for leaf in target_snapshot.active_leaves
    if _is_domain_active_leaf(target_domain, leaf)
      haskey(mapping, leaf) || throw(ArgumentError("missing planned degrees for target leaf $leaf"))
      push!(active, leaf)
    else
      delete!(mapping, leaf)
    end
  end

  for leaf in collect(keys(mapping))
    in_snapshot = leaf <= length(target_snapshot.leaf_to_index) &&
                  @inbounds(target_snapshot.leaf_to_index[leaf] != 0)
    in_target = in_snapshot && _is_domain_active_leaf(target_domain, leaf)
    in_target || delete!(mapping, leaf)
  end

  return _snapshot(grid(target_domain), active)
end

# Snapshot source degrees before batched topology edits. The dictionary form is
# temporary and intentionally mirrors the target-grid cell numbers while splits
# and collapses insert or remove leaves.
function _source_degree_map(space::HpSpace{D}) where {D}
  mapping = Dict{Int,NTuple{D,Int}}()
  sizehint!(mapping, active_leaf_count(space))

  for leaf in snapshot(space).active_leaves
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

  for (leaf_index, leaf) in enumerate(snapshot(space).active_leaves)
    changes = p_degree_changes[leaf_index]
    any(!=(0), changes) || continue
    haskey(mapping, leaf) ||
      throw(ArgumentError("leaf $leaf is not available for planned p adaptation"))
    degrees = mapping[leaf]
    mapping[leaf] = ntuple(axis -> degrees[axis] + changes[axis], D)
  end

  return mapping
end

function _apply_h_refinement_mapping!(mapping::Dict{Int,NTuple{D,Int}}, leaf::Int,
                                      leaves::AbstractVector{<:Integer}) where {D}
  haskey(mapping, leaf) ||
    throw(ArgumentError("leaf $leaf is not available for planned h refinement"))
  degrees = mapping[leaf]
  delete!(mapping, leaf)

  for child in leaves
    mapping[Int(child)] = degrees
  end

  return mapping
end

function _batched_h_derefinement_active!(mapping::Dict{Int,NTuple{D,Int}},
                                         source_snapshot::GridSnapshot{D}, candidates) where {D}
  child_to_parent = Dict{Int,Int}()
  emitted_parents = Set{Int}()
  grid_data = grid(source_snapshot)

  for candidate in candidates
    checked_cell = _checked_cell(grid_data, candidate.cell)
    checked_axis = _checked_axis(grid_data, candidate.axis)
    is_expanded(source_snapshot, checked_cell) ||
      throw(ArgumentError("candidate cell $checked_cell is not expanded"))
    _snapshot_structural_split_axis(source_snapshot, checked_cell) == checked_axis ||
      throw(ArgumentError("candidate cell $checked_cell is not split along axis $checked_axis"))
    first = _snapshot_first_child(source_snapshot, checked_cell)
    first == NONE && throw(ArgumentError("candidate cell $checked_cell has no children"))
    expected_children = ntuple(offset -> first + offset - 1, _MIDPOINT_CHILD_COUNT)
    candidate.children == expected_children ||
      throw(ArgumentError("candidate children do not match the children of cell $checked_cell"))

    for child in expected_children
      is_active_leaf(source_snapshot, child) ||
        throw(ArgumentError("candidate cell $checked_cell cannot be derefined because child $child is not an active snapshot leaf"))
      haskey(child_to_parent, child) &&
        throw(ArgumentError("child $child appears in multiple h-coarsening candidates"))
      child_to_parent[child] = checked_cell
    end

    _apply_h_derefinement_mapping!(mapping, candidate)
  end

  active = Int[]
  sizehint!(active,
            length(source_snapshot.active_leaves) - length(child_to_parent) + length(candidates))

  for leaf in source_snapshot.active_leaves
    parent = get(child_to_parent, leaf, NONE)

    if parent == NONE
      push!(active, leaf)
    elseif !(parent in emitted_parents)
      push!(active, parent)
      push!(emitted_parents, parent)
    end
  end

  return active
end

function _batched_h_refinement_active!(mapping::Dict{Int,NTuple{D,Int}}, active::Vector{Int},
                                       space::HpSpace{D}, h_refinement_axes,
                                       limits::AdaptivityLimits{D}) where {D}
  grid_data = grid(space)
  replacements = Dict{Int,Vector{Int}}()
  replacement_extra = 0

  for (leaf_index, leaf) in enumerate(snapshot(space).active_leaves)
    axes = h_refinement_axes[leaf_index]
    any(axes) || continue
    haskey(mapping, leaf) ||
      throw(ArgumentError("leaf $leaf is not available for planned h refinement"))

    for axis in 1:D
      axes[axis] || continue
      _can_h_refine(grid_data, leaf, axis, limits) ||
        throw(ArgumentError("leaf $leaf reached max_h_level[$axis]"))
    end

    leaves = _split_snapshot_leaf!(grid_data, leaf, axes)
    replacements[leaf] = leaves
    replacement_extra += length(leaves) - 1
    _apply_h_refinement_mapping!(mapping, leaf, leaves)
  end

  isempty(replacements) && return active

  refined_active = Int[]
  sizehint!(refined_active, length(active) + replacement_extra)

  for leaf in active
    if haskey(replacements, leaf)
      append!(refined_active, replacements[leaf])
    else
      push!(refined_active, leaf)
    end
  end

  return refined_active
end

# Construct a plan from batched `p` changes, `h` refinements, and explicit
# `h`-coarsening candidates. The order matters:
# 1. apply derefinements on the target snapshot frontier,
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

  target_domain = domain(space)
  source_snapshot = snapshot(space)
  mapping = _source_degree_map(space)
  active = _batched_h_derefinement_active!(mapping, source_snapshot, h_coarsening_candidates)

  _apply_p_degree_changes!(mapping, space, p_degree_changes)
  active = _batched_h_refinement_active!(mapping, active, space, h_refinement_axes, limits)
  target_snapshot = _snapshot(grid(target_domain), active)
  target_snapshot = _filter_target_snapshot!(target_domain, target_snapshot, mapping)
  active = target_snapshot.active_leaves
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    leaf = active[index]
    haskey(mapping, leaf) || throw(ArgumentError("missing planned degrees for target leaf $leaf"))
    degrees[index] = mapping[leaf]
  end

  return AdaptivityPlan(space, target_domain, target_snapshot, degrees; limits=limits)
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
  checked_cell = _checked_positive(cell, "cell")
  checked_axis = _checked_positive(axis, "axis")
  checked_children = ntuple(index -> _checked_positive(children[index], "children[$index]"),
                            _MIDPOINT_CHILD_COUNT)
  checked_degrees = ntuple(index -> _checked_nonnegative(target_degrees[index],
                                                         "target_degrees[$index]"), D)
  return HCoarseningCandidate{D}(checked_cell, checked_axis, checked_children, checked_degrees)
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
  space_snapshot = snapshot(space)
  is_expanded(space_snapshot, checked_cell) ||
    throw(ArgumentError("candidate cell $checked_cell is not expanded"))
  checked_axis = _checked_axis(grid(space), candidate.axis)
  _snapshot_structural_split_axis(space_snapshot, checked_cell) == checked_axis ||
    throw(ArgumentError("candidate cell $checked_cell is not split along axis $checked_axis"))
  first = _snapshot_first_child(space_snapshot, checked_cell)
  first == NONE && throw(ArgumentError("candidate cell $checked_cell has no children"))
  expected_children = ntuple(offset -> first + offset - 1, _MIDPOINT_CHILD_COUNT)
  candidate.children == expected_children ||
    throw(ArgumentError("candidate children do not match the children of cell $checked_cell"))
  all(child -> is_active_leaf(space_snapshot, child) && space_snapshot.leaf_to_index[child] != 0,
      expected_children) ||
    throw(ArgumentError("candidate cell $checked_cell cannot be derefined because not all children are active space leaves"))
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

function _apply_h_derefinement_mapping!(mapping::Dict{Int,NTuple{D,Int}},
                                        candidate::HCoarseningCandidate{D}) where {D}
  for child in candidate.children
    delete!(mapping, child)
  end

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
  space_snapshot = snapshot(space)

  for cell in 1:stored_cell_count(grid(space))
    is_expanded(space_snapshot, cell) || continue
    axis = _snapshot_structural_split_axis(space_snapshot, cell)
    level(grid(space), cell, axis) >= checked_limits.min_h_level[axis] || continue
    first = _snapshot_first_child(space_snapshot, cell)
    first == NONE && continue
    children = ntuple(offset -> first + offset - 1, _MIDPOINT_CHILD_COUNT)
    all(child -> is_active_leaf(space_snapshot, child) && space_snapshot.leaf_to_index[child] != 0,
        children) || continue
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
  mapping = _degree_map(plan)
  refined_snapshot, leaves = _refine_snapshot_leaf!(target_snapshot(plan), checked_leaf, axes)
  _apply_h_refinement_mapping!(mapping, checked_leaf, leaves)

  _reset_plan_degrees!(plan, mapping, refined_snapshot)
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
  level(grid(plan), checked_cell, checked_axis) >= plan.limits.min_h_level[checked_axis] ||
    throw(ArgumentError("cell $checked_cell cannot be derefined below min_h_level[$checked_axis]"))
  derefined_snapshot, children = _derefine_snapshot_cell(target_snapshot(plan), checked_cell,
                                                         checked_axis)
  child_degrees = ntuple(offset -> cell_degrees(plan, children[offset]), _MIDPOINT_CHILD_COUNT)
  merged = ntuple(current_axis -> maximum(degrees[current_axis] for degrees in child_degrees),
                  dimension(plan))
  mapping = _degree_map(plan)

  for child in children
    delete!(mapping, child)
  end
  mapping[checked_cell] = merged
  _reset_plan_degrees!(plan, mapping, derefined_snapshot)
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
  _collect_active_descendants!(descendants, target_snapshot(plan), leaf)
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
  is_active_leaf(snapshot(source_space(plan)), checked_leaf) ||
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
  is_active_leaf(snapshot(source_space(plan)), checked_leaf) ||
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
  source_snapshot = snapshot(source)
  target_grid = grid(plan)
  marked_leaf_count = 0
  h_refinement_leaf_count = 0
  p_refinement_leaf_count = 0
  p_derefinement_leaf_count = 0

  for leaf in source_snapshot.active_leaves
    change = _source_leaf_change(plan, leaf)
    is_h = any(change.h_axes)
    marked_leaf_count += (is_h || change.p_refined || change.p_derefined)
    h_refinement_leaf_count += is_h
    p_refinement_leaf_count += change.p_refined
    p_derefinement_leaf_count += change.p_derefined
  end

  h_derefinement_cell_count = 0

  for cell in 1:stored_cell_count(grid(source))
    is_expanded(source_snapshot, cell) || continue
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

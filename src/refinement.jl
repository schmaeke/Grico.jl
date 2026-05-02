# This file implements the primitive topology edits on a `CartesianGrid`.
#
# The key point is that refinement here is purely topological. These routines
# only mutate the dyadic refinement tree and the topology data derived from it;
# they do not attempt to update polynomial degrees, continuity constraints,
# integration caches, or state vectors. Those transitions belong to higher
# layers such as `space.jl`, `integration.jl`, and `adaptivity.jl`.
#
# The refinement model is intentionally minimal:
# - each operation edits exactly one parent/child relation in the tree,
# - refinement always bisects one axis at its midpoint,
# - derefinement is allowed only when both children are still active leaves,
# - the active-leaf list is rebuilt after every edit,
# - child storage is retained across derefinement so a later re-refinement can
#   reuse an existing same-axis child block or append a different-axis block.
#
# That last point is worth keeping in mind while reading the code: storage and
# live topology are not the same thing. Derefined child blocks may remain in the
# stored arrays even though they no longer belong to the active tree.

# Public topology edits.

"""
    refine!(grid, cell, axis)

Refine the active leaf `cell` by bisecting it along `axis`.

The refinement is anisotropic and dyadic: the selected cell is replaced by two
children whose logical level increases by one only on the chosen axis, while
all other axis levels and coordinates are inherited unchanged. In physical
space, this corresponds to splitting the cell at its midpoint normal to `axis`.

The function updates the refinement tree, rebuilds the active-leaf list,
increments the topology revision counter, and returns the index of the first
child in the newly active child block. The second child is stored at the
consecutive index `first_child + 1`.

`cell` must be an active leaf. The routine does not impose additional mesh
regularity constraints such as 2:1 balance; such policies are handled by
higher-level adaptivity logic when needed.

Operationally, refinement proceeds in two stages:

1. perform the local tree edit by activating a two-child midpoint block,
2. rebuild all topology data derived from the primary tree arrays.
"""
function refine!(grid::CartesianGrid, cell::Integer, axis::Integer)
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  _require_revision_bump(grid)
  first = _split_leaf!(grid, checked_cell, checked_axis)
  _finish_refinement_update!(grid)
  return first
end

# Apply one local midpoint split without rebuilding derived topology data.
# Batched adaptivity uses this primitive to perform several local edits before
# one final rebuild.
function _split_leaf!(grid::CartesianGrid, checked_cell::Int, checked_axis::Int)
  _require_active_leaf(grid, checked_cell)
  child_level = _require_refinable_axis(grid, checked_cell, checked_axis)
  first = _ensure_child_block!(grid, checked_cell)

  for child in _midpoint_child_block(first)
    retained_first = _first_child(grid, child)
    retained_first == NONE || _require_reusable_child_block(grid, child, retained_first)
  end

  _set_parent_split_state!(grid, checked_cell, false, checked_axis)

  for child_offset in 0:1
    _initialize_midpoint_child!(grid, first + child_offset, checked_cell, checked_axis, child_level,
                                child_offset)
  end

  return first
end

"""
    derefine!(grid, cell)

Derefine the expanded tree cell `cell` by collapsing its child block.

The operation is the inverse of one midpoint bisection: the parent cell becomes
active again, its split-axis marker is cleared, the two children are retired
from the active tree, and all derived topology data are rebuilt. The function
returns the mutated `grid`.

`cell` must currently be expanded, and both of its children must still be
active leaves. This prevents derefinement from removing a subtree that has been
refined further below the candidate parent.

The child block is not deleted from storage. It is merely retired from the live
tree so that a later refinement of the same parent can reuse the existing child
records.
"""
function derefine!(grid::CartesianGrid, cell::Integer)
  checked_cell = _checked_cell(grid, cell)
  _require_revision_bump(grid)
  _collapse_leaf!(grid, checked_cell)
  _finish_refinement_update!(grid)
  return grid
end

# Snapshot topology edits.
#
# These helpers are the append-only counterpart to `refine!`/`derefine!`. They
# update only persistent child storage when needed and return a new active
# frontier snapshot. Existing snapshots remain valid because the grid revision
# is not bumped and no existing cell geometry is changed.

function _refine_snapshot_leaf!(snapshot::GridSnapshot{D}, leaf::Integer,
                                axes::NTuple{D,Bool}) where {D}
  _require_current_snapshot(snapshot)
  grid_data = grid(snapshot)
  checked_leaf = _checked_cell(grid_data, leaf)
  _is_active_leaf(snapshot, checked_leaf) ||
    throw(ArgumentError("leaf $checked_leaf is not an active snapshot leaf"))
  any(axes) || return snapshot, Int[checked_leaf]

  refined = _split_snapshot_leaf!(grid_data, checked_leaf, axes)
  active = _replace_snapshot_leaf(snapshot.active_leaves, checked_leaf, refined)
  return _snapshot(grid_data, active), refined
end

function _derefine_snapshot_cell(snapshot::GridSnapshot, cell::Integer, axis::Integer)
  _require_current_snapshot(snapshot)
  grid_data = grid(snapshot)
  checked_cell = _checked_cell(grid_data, cell)
  checked_axis = _checked_axis(grid_data, axis)
  _is_expanded(snapshot, checked_cell) ||
    throw(ArgumentError("cell $checked_cell is not expanded in the snapshot"))
  _snapshot_structural_split_axis(snapshot, checked_cell) == checked_axis ||
    throw(ArgumentError("cell $checked_cell is not split along axis $checked_axis"))
  first = _snapshot_first_child(snapshot, checked_cell)
  first != NONE || throw(ArgumentError("cell $checked_cell is missing its child block"))
  children = ntuple(offset -> first + offset - 1, _MIDPOINT_CHILD_COUNT)

  for child in children
    _is_active_leaf(snapshot, child) ||
      throw(ArgumentError("cell $checked_cell cannot be derefined because child $child is not an active snapshot leaf"))
  end

  active = _replace_snapshot_children(snapshot.active_leaves, checked_cell, children)
  return _snapshot(grid_data, active), children
end

# Collapse one local midpoint split without rebuilding derived topology data.
function _collapse_leaf!(grid::CartesianGrid, checked_cell::Int)
  first = _require_expanded_child_block(grid, checked_cell)
  _require_active_child_block(grid, checked_cell, first)
  _set_parent_split_state!(grid, checked_cell, true, 0)
  _retire_child_block!(grid, first)
  return grid
end

# Rebuild of topology data derived from the primary tree arrays.

# Recompute active-frontier data derived from the primary tree arrays after a
# refinement or derefinement change. Direct neighbors are derived on demand from
# the logical tree, so there is no neighbor table to rebuild here.
function _finish_refinement_update!(grid::CartesianGrid)
  _rebuild_active_leaves!(grid)
  grid.revision += 1
  return grid
end

# Low-level local helpers and invariants.

_midpoint_child_block(first::Int) = first:(first+_MIDPOINT_CHILD_COUNT-1)

function _require_revision_bump(grid::CartesianGrid)
  grid.revision < typemax(UInt) || throw(ArgumentError("topology revision counter overflow"))
  return nothing
end

function _replace_snapshot_leaf(active::AbstractVector{<:Integer}, leaf::Int,
                                replacement::AbstractVector{<:Integer})
  updated = Int[]
  sizehint!(updated, length(active) + length(replacement) - 1)

  for current in active
    Int(current) == leaf || push!(updated, Int(current))
  end

  append!(updated, replacement)
  return updated
end

function _replace_snapshot_children(active::AbstractVector{<:Integer}, parent::Int,
                                    children::NTuple{_MIDPOINT_CHILD_COUNT,Int})
  updated = Int[]
  sizehint!(updated, length(active) - _MIDPOINT_CHILD_COUNT + 1)

  for current in active
    current_int = Int(current)
    current_int in children && continue
    push!(updated, current_int)
  end

  push!(updated, parent)
  return updated
end

function _split_snapshot_leaf!(grid::CartesianGrid{D}, leaf::Int, axes::NTuple{D,Bool}) where {D}
  leaves = Int[leaf]

  for axis in 1:D
    axes[axis] || continue
    next_leaves = Int[]
    sizehint!(next_leaves, 2 * length(leaves))

    for current_leaf in leaves
      first = _append_snapshot_child_block!(grid, current_leaf, axis)
      push!(next_leaves, first, first + 1)
    end

    leaves = next_leaves
  end

  return leaves
end

function _append_snapshot_child_block!(grid::CartesianGrid{D}, cell::Int, axis::Int) where {D}
  existing = _find_child_block(grid, cell, axis)
  existing != NONE && return existing

  child_level = _require_refinable_axis(grid, cell, axis)
  first = _append_child_block!(grid)

  for child_offset in 0:1
    child = first + child_offset
    @inbounds begin
      grid.parent[child] = cell
      grid.split_axis[child] = 0
      grid.active[child] = false
    end

    for current_axis in 1:D
      @inbounds grid.levels[current_axis][child] = grid.levels[current_axis][cell]
      @inbounds grid.coords[current_axis][child] = grid.coords[current_axis][cell]
    end

    @inbounds grid.levels[axis][child] = child_level
    @inbounds grid.coords[axis][child] = 2 * grid.coords[axis][cell] + child_offset
  end

  _link_child_block!(grid, cell, first)
  return first
end

function _require_refinable_axis(grid::CartesianGrid, checked_cell::Int, checked_axis::Int)
  current_level = _level(grid, checked_cell, checked_axis)
  child_level = _checked_refinement_level(current_level, checked_axis)
  _checked_axis_extent(_root_cell_count(grid, checked_axis), child_level, checked_axis)
  current_coord = _logical_coordinate(grid, checked_cell, checked_axis)
  _checked_child_coordinate(current_coord, 0, checked_axis)
  _checked_child_coordinate(current_coord, 1, checked_axis)
  return child_level
end

function _require_active_leaf(grid::CartesianGrid, checked_cell::Int)
  _is_active_leaf(grid, checked_cell) ||
    throw(ArgumentError("cell $checked_cell is not an active leaf"))
  return checked_cell
end

function _require_expanded_child_block(grid::CartesianGrid, checked_cell::Int)
  _is_expanded(grid, checked_cell) || throw(ArgumentError("cell $checked_cell is not expanded"))
  first = _first_child(grid, checked_cell)
  first != NONE || throw(ArgumentError("cell $checked_cell is missing its child block"))
  return _require_child_block_owner(grid, checked_cell, first)
end

function _set_parent_split_state!(grid::CartesianGrid, checked_cell::Int, active::Bool,
                                  split_axis::Int)
  @inbounds begin
    grid.active[checked_cell] = active
    grid.split_axis[checked_cell] = split_axis
  end
  return nothing
end

function _ensure_child_block!(grid::CartesianGrid, checked_cell::Int)
  first = _first_child(grid, checked_cell)

  # Child storage is persistent across derefinement. If this cell was refined
  # before, we can reuse the existing child block instead of allocating again.
  if first == NONE
    first = _append_child_block!(grid)
    grid.first_child[checked_cell] = first
    return first
  end

  return _require_reusable_child_block(grid, checked_cell, first)
end

function _require_child_block_owner(grid::CartesianGrid, checked_cell::Int, first::Int)
  1 <= first <= stored_cell_count(grid) - _MIDPOINT_CHILD_COUNT + 1 ||
    throw(ArgumentError("cell $checked_cell has an invalid child block"))

  for child in _midpoint_child_block(first)
    _parent(grid, child) == checked_cell ||
      throw(ArgumentError("cell $checked_cell has an inconsistent child block"))
  end

  return first
end

function _require_reusable_child_block(grid::CartesianGrid, checked_cell::Int, first::Int)
  _require_child_block_owner(grid, checked_cell, first)

  for child in _midpoint_child_block(first)
    _is_tree_cell(grid, child) &&
      throw(ArgumentError("cell $checked_cell has live retained children"))
  end

  return first
end

function _initialize_midpoint_child!(grid::CartesianGrid{D}, child::Int, checked_cell::Int,
                                     checked_axis::Int, child_level::Int,
                                     child_offset::Int) where {D}
  @inbounds begin
    grid.parent[child] = checked_cell
    grid.split_axis[child] = 0
    grid.active[child] = true
  end

  # Children inherit the full anisotropic level/coordinate state of the
  # parent first, and only the split axis is then updated to reflect the new
  # dyadic subinterval.
  for current_axis in 1:D
    @inbounds grid.levels[current_axis][child] = grid.levels[current_axis][checked_cell]
    @inbounds grid.coords[current_axis][child] = grid.coords[current_axis][checked_cell]
  end

  # The midpoint split doubles the logical coordinate on the refined axis and
  # adds the child offset `0` or `1`, which selects the lower or upper half.
  @inbounds grid.levels[checked_axis][child] = child_level
  @inbounds grid.coords[checked_axis][child] = 2 * grid.coords[checked_axis][checked_cell] +
                                               child_offset
  return nothing
end

function _require_active_child_block(grid::CartesianGrid, checked_cell::Int, first::Int)
  # Derefinement is only valid if the child block is still the active frontier
  # of the tree. If any child has been refined further, collapsing the parent
  # would discard a nontrivial subtree.
  for child in _midpoint_child_block(first)
    _is_active_leaf(grid, child) ||
      throw(ArgumentError("cell $checked_cell cannot be derefined because child $child is not an active leaf"))
  end

  return nothing
end

function _retire_child_block!(grid::CartesianGrid, first::Int)
  # Derefinement retires the children from the live frontier but intentionally
  # leaves their stored topology data in place for possible later reuse.
  for child in _midpoint_child_block(first)
    @inbounds grid.active[child] = false
  end

  return nothing
end

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
# - direct-neighbor tables and active-leaf lists are rebuilt after every edit,
# - child storage is retained across derefinement so a later re-refinement can
#   reuse the same child block.
#
# That last point is worth keeping in mind while reading the code: storage and
# live topology are not the same thing. A derefined child block may remain in
# the stored arrays even though it no longer belongs to the active tree.

# Public topology edits.

"""
    refine!(grid, cell, axis)

Refine the active leaf `cell` by bisecting it along `axis`.

The refinement is anisotropic and dyadic: the selected cell is replaced by two
children whose logical level increases by one only on the chosen axis, while
all other axis levels and coordinates are inherited unchanged. In physical
space, this corresponds to splitting the cell at its midpoint normal to `axis`.

The function updates the refinement tree, rebuilds the direct-neighbor tables
and active-leaf list, increments the topology revision counter, and returns the
index of the first child in the newly active child block. The second child is
stored at the consecutive index `first_child + 1`.

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
  _require_active_leaf(grid, checked_cell)
  first = _split_leaf!(grid, checked_cell, checked_axis)
  _finish_refinement_update!(grid)
  return first
end

# Split and collapse of one local tree edge.

# Perform the purely topological part of one binary midpoint split. The caller
# is responsible for finalizing all derived data structures afterwards.
function _split_leaf!(grid::CartesianGrid{D}, checked_cell::Int, checked_axis::Int) where {D}
  first = _ensure_child_block!(grid, checked_cell)
  _set_parent_split_state!(grid, checked_cell, false, checked_axis)
  _initialize_midpoint_child_block!(grid, first, checked_cell, checked_axis)

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
  _collapse_leaf!(grid, checked_cell)
  _finish_refinement_update!(grid)
  return grid
end

# Collapse one expanded parent back into an active leaf. This only changes the
# primary tree state; neighbor tables, active-leaf enumeration, and the revision
# counter are handled by `_finish_refinement_update!`.
function _collapse_leaf!(grid::CartesianGrid, checked_cell::Int)
  first = _required_child_block!(grid, checked_cell, "derefine")
  _require_active_child_block(grid, checked_cell, first)
  _set_parent_split_state!(grid, checked_cell, true, 0)
  _retire_child_block!(grid, first)

  return grid
end

# Rebuild of topology data derived from the primary tree arrays.

# Recompute all topology data derived from the primary tree arrays after a
# refinement or derefinement change. This keeps the public topology API
# self-consistent and provides a monotone revision counter for cache invalidation.
function _finish_refinement_update!(grid::CartesianGrid)
  _rebuild_direct_neighbors!(grid)
  _rebuild_active_leaves!(grid)
  grid.revision += 1
  return grid
end

# Low-level local helpers and invariants.

@inline _midpoint_child_block(first::Int) = first:(first+_MIDPOINT_CHILD_COUNT-1)

@inline function _require_active_leaf(grid::CartesianGrid, checked_cell::Int)
  is_active_leaf(grid, checked_cell) ||
    throw(ArgumentError("cell $checked_cell is not an active leaf"))
  return checked_cell
end

@inline function _required_child_block!(grid::CartesianGrid, checked_cell::Int,
                                        context::AbstractString)
  is_expanded(grid, checked_cell) || throw(ArgumentError("cell $checked_cell is not expanded"))
  first = first_child(grid, checked_cell)
  first != NONE || throw(ArgumentError("$context cell $checked_cell is missing its child block"))
  return first
end

@inline function _set_parent_split_state!(grid::CartesianGrid, checked_cell::Int, active::Bool,
                                          split_axis::Int)
  @inbounds begin
    grid.active[checked_cell] = active
    grid.split_axis[checked_cell] = split_axis
  end
  return nothing
end

function _ensure_child_block!(grid::CartesianGrid, checked_cell::Int)
  first = first_child(grid, checked_cell)

  # Child storage is persistent across derefinement. If this cell was refined
  # before, we can reuse the existing child block instead of allocating again.
  if first == NONE
    first = _append_child_block!(grid)
    grid.first_child[checked_cell] = first
  end

  return first
end

function _initialize_midpoint_child_block!(grid::CartesianGrid{D}, first::Int, checked_cell::Int,
                                           checked_axis::Int) where {D}
  # Both children live one dyadic level deeper on the split axis, while all
  # tangential levels are inherited unchanged from the parent.
  child_level = level(grid, checked_cell, checked_axis) + 1

  for (child_offset, child) in enumerate(_midpoint_child_block(first))
    _initialize_midpoint_child!(grid, child, checked_cell, checked_axis, child_level,
                                child_offset - 1)
  end

  return nothing
end

function _initialize_midpoint_child!(grid::CartesianGrid{D}, child::Int, checked_cell::Int,
                                     checked_axis::Int, child_level::Int,
                                     child_offset::Int) where {D}
  @inbounds begin
    grid.parent[child] = checked_cell
    grid.first_child[child] = NONE
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
    is_active_leaf(grid, child) ||
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

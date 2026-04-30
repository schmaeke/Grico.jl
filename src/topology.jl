# This file implements the dyadic Cartesian refinement tree used throughout the
# library.
#
# Topology is the purely discrete layer of `Grico.jl`. It answers questions such
# as:
# - which cells currently partition the domain,
# - how a cell is located on the dyadic lattice of each axis,
# - which cell lies across a given face,
# - how to climb from a fine face to a covering coarse face,
# - and how periodic wrapping modifies those relations.
#
# No physical coordinates appear here. Geometry is handled separately in
# `geometry.jl`; this file is responsible only for the combinatorial structure
# of the Cartesian refinement tree.
#
# A few representation choices are central to the rest of the implementation:
# - refinement is anisotropic and always bisects exactly one axis at a time,
# - same-level neighbors are derived from logical cell signatures on demand,
# - higher-level queries recover coverings and active opposite leaves from those
#   derived same-level relations,
# - active leaves are stored explicitly in deterministic sorted order,
# - expanded ancestors remain stored in the tree, while derefined retired cells
#   may remain in storage but no longer belong to the live tree.
#
# That separation between raw storage, the live tree, and the active-leaf
# partition is what later files rely on when they compile continuity, assemble
# interface data, and transfer fields across adaptive updates.

# Public constants and side markers.

"""
    NONE

Sentinel cell index used when a requested topological relation does not exist.

In the refinement tree, valid cells are stored with one-based indices. `NONE`
therefore denotes the absence of a parent, child block, or same-level neighbor.
Public query functions return `NONE` when the corresponding tree relation is not
present, for example at a nonperiodic outer boundary.
"""
const NONE = 0

"""
    LOWER

Identifier for the lower side of an axis-aligned cell face.

Along a chosen axis, `LOWER` denotes the face with the smaller logical
coordinate. It is used throughout the topology, interface, and boundary APIs to
select one of the two sides of a Cartesian cell.
"""
const LOWER = 1

"""
    UPPER

Identifier for the upper side of an axis-aligned cell face.

Along a chosen axis, `UPPER` denotes the face with the larger logical
coordinate. Together with [`LOWER`](@ref), it provides a compact side notation
for neighborhood, boundary, and interface queries.
"""
const UPPER = 2
const _MIDPOINT_CHILD_COUNT = 2

# Core tree representation.

"""
    CartesianGrid(root_counts; periodic=false)

Topological model of a `D`-dimensional Cartesian refinement tree with dyadic
midpoint bisection.

`CartesianGrid` stores only combinatorial information: the root mesh counts,
per-axis refinement levels `ℓₐ`, logical coordinates `iₐ`, parent/child links,
active-leaf flags, periodic wrapping, and a revision counter. Same-level direct
neighbors are derived from the logical tree when queried instead of being stored
per cell. Physical coordinates and measures are handled separately by `Geometry`
and `Domain`; this type represents the discrete refinement topology on which
continuity, interface, and adaptivity logic operate.

Along axis `a`, a cell at level `ℓₐ` occupies one interval in a logical lattice
with `root_counts[a] * 2^ℓₐ` intervals. Refinement splits one axis at a
midpoint, so each expanded cell owns exactly two children. This keeps the
topology dimension-independent while still supporting anisotropic refinement.

Two distinctions are important throughout the file:

- A stored cell is any cell record present in the topology arrays.
- A tree cell is a stored cell that currently belongs to the live refinement
  tree, meaning it is either an active leaf or an expanded ancestor.

Most public algorithms operate on tree cells, and discretization layers operate
specifically on active leaves.
"""
mutable struct CartesianGrid{D}
  root_counts::NTuple{D,Int}
  levels::NTuple{D,Vector{Int}}
  coords::NTuple{D,Vector{Int}}
  parent::Vector{Int}
  first_child::Vector{Int}
  split_axis::Vector{Int}
  active::BitVector
  active_leaves::Vector{Int}
  periodic::NTuple{D,Bool}
  revision::UInt

  # Low-level constructor for already materialized topology arrays. All code
  # paths that build a grid eventually pass through `check_topology`, so the
  # constructor is also the consistency gate for copied or rebuilt grids.
  function CartesianGrid{D}(root_counts::NTuple{D,Int}, levels::NTuple{D,Vector{Int}},
                            coords::NTuple{D,Vector{Int}}, parent::Vector{Int},
                            first_child::Vector{Int}, split_axis::Vector{Int}, active::BitVector,
                            active_leaves::Vector{Int}, periodic::NTuple{D,Bool},
                            revision::UInt) where {D}
    D >= 1 || throw(ArgumentError("dimension must be positive"))
    checked_counts = ntuple(axis -> _checked_positive(root_counts[axis], "root_counts[$axis]"), D)
    grid = new{D}(checked_counts, levels, coords, parent, first_child, split_axis, active,
                  active_leaves, periodic, revision)
    check_topology(grid)
    return grid
  end
end

# Build the unrefined root tree. Root cells are numbered in mixed-radix order
# with axis 1 varying fastest.
function CartesianGrid(root_counts::Tuple{Vararg{Integer,D}}; periodic=false) where {D}
  D >= 1 || throw(ArgumentError("dimension must be positive"))
  checked_counts = ntuple(axis -> _checked_positive(root_counts[axis], "root_counts[$axis]"), D)
  checked_periodic = _checked_periodic_axes(periodic, D)
  cell_total = _checked_root_cell_total(checked_counts)
  levels = ntuple(_ -> zeros(Int, cell_total), D)
  coords = ntuple(_ -> zeros(Int, cell_total), D)
  parent = fill(NONE, cell_total)
  first_child = fill(NONE, cell_total)
  split_axis = zeros(Int, cell_total)
  active = trues(cell_total)
  active_leaves = collect(1:cell_total)
  strides = ntuple(axis -> axis == 1 ? 1 : prod(checked_counts[1:(axis-1)]), D)

  for cell in 1:cell_total
    linear = cell - 1

    @inbounds for axis in 1:D
      coord = fld(linear, strides[axis]) % checked_counts[axis]
      coords[axis][cell] = coord
    end
  end

  return CartesianGrid{D}(checked_counts, levels, coords, parent, first_child, split_axis, active,
                          active_leaves, checked_periodic, zero(UInt))
end

"""
    GridSnapshot

Immutable active-frontier view of a [`CartesianGrid`](@ref).

`GridSnapshot` stores the mesh state that changes between adaptation steps:
the active leaves, dense active-leaf lookup, boundary faces, and active
interfaces. The referenced `CartesianGrid` owns persistent tree data such as
levels, logical coordinates, and parent/child links.

Snapshots are valid only for the grid revision from which they were built.
Normal append-only snapshot adaptation does not change that revision, while
destructive topology edits such as `refine!`, `derefine!`, and `compact!`
invalidate old snapshots by incrementing it.

Snapshot active leaves are stored in deterministic Morton order. Cell ids remain
structural handles owned by the grid; the snapshot order is the traversal order
used by spaces, integration, adaptivity, and output.
"""
struct GridSnapshot{D}
  grid::CartesianGrid{D}
  generation::UInt
  active_leaves::Vector{Int}
  leaf_to_index::Vector{Int}
  boundary_leaf::Vector{Int}
  boundary_axis::Vector{UInt8}
  boundary_side::Vector{UInt8}
  interface_minus::Vector{Int}
  interface_plus::Vector{Int}
  interface_axis::Vector{UInt8}
end

"""
    GridBoundaryFace(leaf, axis, side)

Small value object returned by [`boundary_face_spec`](@ref).
"""
struct GridBoundaryFace
  leaf::Int
  axis::Int
  side::Int
end

"""
    GridInterface(minus, axis, plus)

Small value object returned by [`interface_spec`](@ref).
"""
struct GridInterface
  minus::Int
  axis::Int
  plus::Int
end

"""
    snapshot(grid)

Build a [`GridSnapshot`](@ref) for the current active frontier of `grid`.
"""
snapshot(grid::CartesianGrid{D}) where {D} = _snapshot(grid, active_leaves(grid))

function _snapshot(grid::CartesianGrid{D}, active::AbstractVector{<:Integer}) where {D}
  active_leaves = _morton_ordered_snapshot_leaves(grid, active)
  leaf_to_index = _snapshot_leaf_lookup(grid, active_leaves)
  boundary_leaf, boundary_axis, boundary_side = _snapshot_boundary_data(grid, active_leaves)
  interface_minus, interface_plus, interface_axis = _snapshot_interface_data(grid, active_leaves,
                                                                             leaf_to_index)
  return GridSnapshot{D}(grid, revision(grid), active_leaves, leaf_to_index, boundary_leaf,
                         boundary_axis, boundary_side, interface_minus, interface_plus,
                         interface_axis)
end

# Snapshot traversal order is spatial rather than storage-based. Leaves are
# compared by the Morton code of their lower corner on the finest lattice needed
# by this snapshot; the refinement level and cell id are deterministic
# tie-breakers for anisotropic or otherwise coincident lower corners.
function _morton_ordered_snapshot_leaves(grid::CartesianGrid{D},
                                         active::AbstractVector{<:Integer}) where {D}
  leaves = Int[_checked_cell(grid, leaf) for leaf in active]
  isempty(leaves) && throw(ArgumentError("grid snapshot must contain at least one active leaf"))
  max_levels = _snapshot_max_levels(grid, leaves)
  bit_count = _snapshot_morton_bit_count(grid, max_levels)
  keys = Vector{_SnapshotMortonKey{D}}(undef, length(leaves))

  for index in eachindex(leaves)
    leaf = leaves[index]
    keys[index] = _snapshot_morton_key(grid, leaf, max_levels)
  end

  ordering = sortperm(keys; lt=(left, right) -> _snapshot_morton_isless(left, right, bit_count))
  return leaves[ordering]
end

# Precomputed ordering key used by the Morton comparator. Storing the scaled
# lower corner avoids building large integer Morton codes and keeps comparison
# exact for all refinement levels admitted by the topology checks.
struct _SnapshotMortonKey{D}
  lower::NTuple{D,Int128}
  level::NTuple{D,Int}
  cell::Int
end

function _snapshot_max_levels(grid::CartesianGrid{D}, leaves::AbstractVector{Int}) where {D}
  return ntuple(axis -> maximum(leaf -> _level(grid, leaf, axis), leaves), D)
end

function _snapshot_morton_bit_count(grid::CartesianGrid{D}, max_levels::NTuple{D,Int}) where {D}
  return maximum(axis -> _snapshot_axis_bit_count(grid, max_levels, axis), 1:D)
end

function _snapshot_axis_bit_count(grid::CartesianGrid, max_levels, axis::Int)
  extent = _checked_axis_extent(_root_cell_count(grid, axis), max_levels[axis], axis)
  value = extent - 1
  bits = 0

  while value > 0
    bits += 1
    value >>= 1
  end

  return bits
end

function _snapshot_morton_key(grid::CartesianGrid{D}, leaf::Int,
                              max_levels::NTuple{D,Int}) where {D}
  lower = ntuple(axis -> _scaled_lower_coordinate(grid, leaf, axis, max_levels[axis]), D)
  return _SnapshotMortonKey{D}(lower, _level_tuple(grid, leaf), leaf)
end

function _snapshot_morton_isless(left::_SnapshotMortonKey{D}, right::_SnapshotMortonKey{D},
                                 bit_count::Int) where {D}
  for bit in (bit_count-1):-1:0
    for axis in D:-1:1
      left_bit = !iszero((left.lower[axis] >> bit) & 1)
      right_bit = !iszero((right.lower[axis] >> bit) & 1)
      left_bit == right_bit || return !left_bit && right_bit
    end
  end

  left.level == right.level || return left.level < right.level
  return left.cell < right.cell
end

# Active topology comparisons across independently compacted grids cannot rely
# on cell ids. Logical signatures provide the same information in the common
# root lattice and are ordered with the same Morton policy as snapshots.
function _active_leaf_signatures(grid::CartesianGrid{D},
                                 active::AbstractVector{<:Integer}) where {D}
  ordered = _morton_ordered_snapshot_leaves(grid, active)
  return [(_level_tuple(grid, leaf), _logical_coordinate_tuple(grid, leaf)) for leaf in ordered]
end

function _active_leaf_signatures(snapshot::GridSnapshot)
  _require_current_snapshot(snapshot)
  grid_data = grid(snapshot)
  return [(_level_tuple(grid_data, leaf), _logical_coordinate_tuple(grid_data, leaf))
          for leaf in snapshot.active_leaves]
end

# Build a fresh grid that contains the smallest root-preserving tree needed by
# a snapshot. Every expanded retained parent keeps its full binary child block
# so the normal `CartesianGrid` invariants continue to hold; inactive siblings
# become ordinary grid leaves if they are not part of the snapshot frontier.
function _compact_grid_snapshot(snapshot::GridSnapshot{D}) where {D}
  check_snapshot(snapshot)
  source_grid = grid(snapshot)
  source_stored = stored_cell_count(source_grid)
  root_total = root_cell_total(source_grid)
  old_to_new = zeros(Int, source_stored)
  expanded = _snapshot_expanded_cell_flags(snapshot)
  compact_cell_count = root_total + _MIDPOINT_CHILD_COUNT * count(expanded)
  levels = ntuple(_ -> Int[], D)
  coords = ntuple(_ -> Int[], D)
  parent = Int[]
  first_child = Int[]
  split_axis = Int[]
  active = falses(0)
  sizehint!(parent, compact_cell_count)
  sizehint!(first_child, compact_cell_count)
  sizehint!(split_axis, compact_cell_count)
  sizehint!(active, compact_cell_count)

  for axis in 1:D
    sizehint!(levels[axis], compact_cell_count)
    sizehint!(coords[axis], compact_cell_count)
  end

  for old_root in 1:root_total
    new_root = _append_compact_cell!(source_grid, levels, coords, parent, first_child, split_axis,
                                     active, old_root, NONE)
    new_root == old_root || throw(ArgumentError("compacted grid must preserve root cell ids"))
    old_to_new[old_root] = new_root
  end

  for old_root in 1:root_total
    _compact_snapshot_subtree!(snapshot, expanded, old_to_new, levels, coords, parent, first_child,
                               split_axis, active, old_root)
  end

  active_leaves = Int[]
  sizehint!(active_leaves, length(active))

  for cell in eachindex(active)
    @inbounds active[cell] && push!(active_leaves, cell)
  end

  compact_grid = CartesianGrid{D}(source_grid.root_counts, levels, coords, parent, first_child,
                                  split_axis, active, active_leaves, source_grid.periodic,
                                  zero(UInt))
  compact_active = Int[]
  sizehint!(compact_active, length(snapshot.active_leaves))

  for old_leaf in snapshot.active_leaves
    mapped_leaf = @inbounds old_to_new[old_leaf]
    mapped_leaf != NONE || throw(ArgumentError("compaction dropped active leaf $old_leaf"))
    push!(compact_active, mapped_leaf)
  end

  compact_snapshot = _snapshot(compact_grid, compact_active)
  return compact_snapshot, old_to_new
end

"""
    compact!(grid, snapshot)

Destructively prune `grid` so it keeps only the tree data needed to represent
`snapshot`.

This rewrites non-root cell ids, preserves root ids, bumps the grid revision,
and invalidates every existing snapshot of `grid`, including the input
snapshot. The returned tuple is
`(compacted_snapshot, old_to_new_cell)`, where `compacted_snapshot` is valid for
the compacted `grid` and `old_to_new_cell[old_cell]` is the new cell id or
[`NONE`](@ref) if the old cell was pruned.
"""
function compact!(grid_data::CartesianGrid{D}, active_snapshot::GridSnapshot{D}) where {D}
  active_snapshot.grid === grid_data || throw(ArgumentError("snapshot must reference grid"))
  _require_revision_bump(grid_data)
  compact_snapshot, old_to_new = _compact_grid_snapshot(active_snapshot)
  compact_grid = grid(compact_snapshot)
  grid_data.root_counts = compact_grid.root_counts
  grid_data.levels = compact_grid.levels
  grid_data.coords = compact_grid.coords
  grid_data.parent = compact_grid.parent
  grid_data.first_child = compact_grid.first_child
  grid_data.split_axis = compact_grid.split_axis
  grid_data.active = compact_grid.active
  grid_data.active_leaves = compact_grid.active_leaves
  grid_data.periodic = compact_grid.periodic
  grid_data.revision += 1
  check_topology(grid_data)
  compacted_snapshot = _snapshot(grid_data, compact_snapshot.active_leaves)
  return compacted_snapshot, old_to_new
end

# Mark exactly those cells that are ancestors of the snapshot active leaves.
# Compaction then follows this bitset instead of recursively asking the snapshot
# whether unrelated retained storage happens to contain active descendants.
function _snapshot_expanded_cell_flags(snapshot::GridSnapshot)
  source_grid = grid(snapshot)
  expanded = falses(stored_cell_count(source_grid))

  for leaf in snapshot.active_leaves
    current = _parent(source_grid, leaf)

    while current != NONE
      @inbounds expanded[current] = true
      current = _parent(source_grid, current)
    end
  end

  return expanded
end

function _append_compact_cell!(source_grid::CartesianGrid{D}, levels::NTuple{D,Vector{Int}},
                               coords::NTuple{D,Vector{Int}}, parent::Vector{Int},
                               first_child::Vector{Int}, split_axis::Vector{Int}, active::BitVector,
                               source_cell::Int, parent_cell::Int) where {D}
  new_cell = length(parent) + 1
  push!(parent, parent_cell)
  push!(first_child, NONE)
  push!(split_axis, 0)
  push!(active, true)

  for axis in 1:D
    push!(levels[axis], _level(source_grid, source_cell, axis))
    push!(coords[axis], _logical_coordinate(source_grid, source_cell, axis))
  end

  return new_cell
end

function _compact_snapshot_subtree!(snapshot::GridSnapshot{D}, expanded::BitVector,
                                    old_to_new::Vector{Int}, levels::NTuple{D,Vector{Int}},
                                    coords::NTuple{D,Vector{Int}}, parent::Vector{Int},
                                    first_child::Vector{Int}, split_axis::Vector{Int},
                                    active::BitVector, source_cell::Int) where {D}
  @inbounds expanded[source_cell] || return nothing
  source_grid = grid(snapshot)
  target_cell = @inbounds old_to_new[source_cell]
  target_cell != NONE || throw(ArgumentError("snapshot compaction reached an unmapped tree cell"))
  source_first = _first_child(source_grid, source_cell)
  source_first != NONE ||
    throw(ArgumentError("expanded snapshot cell $source_cell is missing its child block"))
  source_split = _structural_split_axis(source_grid, source_cell)
  source_split != 0 ||
    throw(ArgumentError("expanded snapshot cell $source_cell has no structural split axis"))
  target_first = length(parent) + 1
  @inbounds begin
    first_child[target_cell] = target_first
    split_axis[target_cell] = source_split
    active[target_cell] = false
  end

  for child_offset in 0:(_MIDPOINT_CHILD_COUNT-1)
    source_child = source_first + child_offset
    target_child = _append_compact_cell!(source_grid, levels, coords, parent, first_child,
                                         split_axis, active, source_child, target_cell)
    @inbounds old_to_new[source_child] = target_child
  end

  for child_offset in 0:(_MIDPOINT_CHILD_COUNT-1)
    _compact_snapshot_subtree!(snapshot, expanded, old_to_new, levels, coords, parent, first_child,
                               split_axis, active, source_first + child_offset)
  end

  return nothing
end

# Basic global queries.

"""
    dimension(grid::CartesianGrid)

Return the topological dimension `D` of the refinement tree.

This is the number of coordinate axes carried by the Cartesian root mesh and
therefore the length of per-axis tuples such as levels, logical coordinates,
root counts, and periodicity flags.
"""
dimension(::CartesianGrid{D}) where {D} = D
dimension(::GridSnapshot{D}) where {D} = D

grid(snapshot::GridSnapshot) = snapshot.grid

"""
    root_cell_counts(grid)

Return the number of root cells along each coordinate axis.

These counts define the unrefined tensor-product partition from which all later
dyadic refinement starts.
"""
root_cell_counts(grid::CartesianGrid) = grid.root_counts

"""
    root_cell_count(grid, axis)

Return the number of root cells on a single axis.

If a cell has logical level `ℓ` on that axis, then the corresponding logical
lattice consists of `root_cell_count(grid, axis) * 2^ℓ` intervals there.
"""
@inline function root_cell_count(grid::CartesianGrid, axis::Integer)
  count = dimension(grid)
  @boundscheck 1 <= axis <= count || _throw_index_error(axis, count, "axis")
  return _root_cell_count(grid, Int(axis))
end

"""
    periodic_axes(grid)

Return the per-axis periodicity flags of the root grid.

An entry is `true` exactly when the corresponding outer boundaries are
identified topologically, so neighborhood queries wrap across that axis instead
of terminating at the domain boundary.
"""
periodic_axes(grid::CartesianGrid) = grid.periodic

"""
    is_periodic_axis(grid, axis)

Return `true` when the given axis is periodic.

On a periodic axis, the lower and upper root boundaries are identified, and
queries such as [`neighbor`](@ref), [`covering_neighbor`](@ref), and
[`is_domain_boundary`](@ref) behave accordingly.
"""
@inline function is_periodic_axis(grid::CartesianGrid, axis::Integer)
  count = dimension(grid)
  @boundscheck 1 <= axis <= count || _throw_index_error(axis, count, "axis")
  return _is_periodic_axis(grid, Int(axis))
end

"""
    root_cell_total(grid)

Return the total number of root cells in the unrefined tensor-product mesh.

This is the product of [`root_cell_counts`](@ref) and equals the number of
stored cells in a freshly constructed grid before any refinement occurs.
"""
root_cell_total(grid::CartesianGrid) = _checked_root_cell_total(grid.root_counts)

"""
    stored_cell_count(grid)

Return the number of cell records currently stored in the topology.

This count includes active leaves and expanded ancestor cells that remain part
of the refinement tree. It is therefore a storage quantity, not the number of
currently active finite elements.
"""
stored_cell_count(grid::CartesianGrid) = length(grid.parent)

"""
    revision(grid)

Return the topology revision counter.

The counter is incremented whenever refinement or derefinement rebuilds the tree
connectivity, and also when destructive compaction rewrites stored cell ids.
Higher-level caches can use it to detect when topological data have become
stale.
"""
revision(grid::CartesianGrid) = grid.revision

"""
    check_snapshot(snapshot)

Validate that `snapshot` is current for its grid and that its active-frontier
lookup and derived traversal arrays are internally consistent.
"""
function check_snapshot(snapshot::GridSnapshot)
  _require_current_snapshot(snapshot)
  active = snapshot.active_leaves
  lookup = snapshot.leaf_to_index
  stored = stored_cell_count(grid(snapshot))
  length(lookup) <= stored || throw(ArgumentError("snapshot leaf lookup exceeds grid storage"))

  expected_lookup = zeros(Int, length(lookup))
  for index in eachindex(active)
    leaf = active[index]
    1 <= leaf <= length(lookup) ||
      throw(ArgumentError("snapshot active leaf is outside the snapshot lookup range"))
    @inbounds expected_lookup[leaf] == 0 ||
              throw(ArgumentError("snapshot active leaves must be unique"))
    @inbounds expected_lookup[leaf] = index
  end

  lookup == expected_lookup || throw(ArgumentError("snapshot leaf lookup is inconsistent"))
  active == _morton_ordered_snapshot_leaves(grid(snapshot), active) ||
    throw(ArgumentError("snapshot active leaves are not in Morton order"))
  _check_snapshot_boundary_data(snapshot)
  _check_snapshot_interface_data(snapshot)
  return nothing
end

"""
    active_leaf_count(grid)

Return the number of active leaves in the current refinement tree.

Active leaves are the cells that currently partition the domain and therefore
act as the discrete elements seen by spaces, integration, and assembly.
"""
active_leaf_count(grid::CartesianGrid) = length(grid.active_leaves)
function active_leaf_count(snapshot::GridSnapshot)
  length(_require_current_snapshot(snapshot).active_leaves)
end

"""
    active_leaves(grid)

Return the active leaves as a newly allocated vector.

For a `CartesianGrid`, the returned vector follows structural cell-id order.
For a `GridSnapshot`, it follows the snapshot's deterministic Morton traversal
order. In both cases, the returned vector is independent of the internal
storage, so callers may reorder or modify it without mutating the source.
"""
active_leaves(grid::CartesianGrid) = copy(grid.active_leaves)
active_leaves(snapshot::GridSnapshot) = copy(_require_current_snapshot(snapshot).active_leaves)

@inline _root_cell_count(grid::CartesianGrid, axis::Int) = @inbounds grid.root_counts[axis]

@inline _is_periodic_axis(grid::CartesianGrid, axis::Int) = @inbounds grid.periodic[axis]

@inline _active_leaf(grid::CartesianGrid, index::Int) = @inbounds grid.active_leaves[index]
@inline _active_leaf(snapshot::GridSnapshot, index::Int) = @inbounds snapshot.active_leaves[index]

@inline _level(grid::CartesianGrid, cell::Int, axis::Int) = @inbounds grid.levels[axis][cell]

@inline function _level_tuple(grid::CartesianGrid{D}, cell::Int) where {D}
  return ntuple(axis -> _level(grid, cell, axis), D)
end

@inline _logical_coordinate(grid::CartesianGrid, cell::Int, axis::Int) = @inbounds grid.coords[axis][cell]

@inline function _logical_coordinate_tuple(grid::CartesianGrid{D}, cell::Int) where {D}
  return ntuple(axis -> _logical_coordinate(grid, cell, axis), D)
end

@inline _parent(grid::CartesianGrid, cell::Int) = @inbounds grid.parent[cell]

@inline _first_child(grid::CartesianGrid, cell::Int) = @inbounds grid.first_child[cell]

@inline _split_axis(grid::CartesianGrid, cell::Int) = @inbounds grid.split_axis[cell]

@inline _is_active_leaf(grid::CartesianGrid, cell::Int) = @inbounds grid.active[cell]
@inline function _is_active_leaf(snapshot::GridSnapshot, cell::Int)
  return cell <= length(snapshot.leaf_to_index) && @inbounds(snapshot.leaf_to_index[cell] != 0)
end

@inline _is_expanded(grid::CartesianGrid, cell::Int) = _split_axis(grid, cell) != 0
@inline _is_expanded(snapshot::GridSnapshot, cell::Int) = _snapshot_has_active_descendant(snapshot,
                                                                                          cell)

@inline _is_tree_cell(grid::CartesianGrid, cell::Int) = _is_active_leaf(grid, cell) ||
                                                        _is_expanded(grid, cell)

# Cell-local tree queries.

"""
    active_leaf(grid, index)

Return the `index`-th active leaf of the current tree.

This provides indexed access to the ordering used by [`active_leaves`](@ref)
and [`active_leaf_count`](@ref).
"""
@inline function active_leaf(grid::CartesianGrid, index::Integer)
  count = active_leaf_count(grid)
  @boundscheck 1 <= index <= count || _throw_index_error(index, count, "active leaf")
  return _active_leaf(grid, Int(index))
end

@inline function active_leaf(snapshot::GridSnapshot, index::Integer)
  _require_current_snapshot(snapshot)
  count = active_leaf_count(snapshot)
  @boundscheck 1 <= index <= count || _throw_index_error(index, count, "active leaf")
  return _active_leaf(snapshot, Int(index))
end

"""
    level(grid, cell)
    level(grid, cell, axis)

Return the logical refinement level of a cell, either on all axes or on one
specified axis.

Levels are stored per axis because refinement is anisotropic. Along axis `a`, a
level `ℓₐ` means that the cell lives on a dyadic refinement lattice obtained by
subdividing each root interval into `2^ℓₐ` subintervals.
"""
@inline function level(grid::CartesianGrid{D}, cell::Integer) where {D}
  count = stored_cell_count(grid)
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _level_tuple(grid, Int(cell))
end

@inline function level(grid::CartesianGrid, cell::Integer, axis::Integer)
  cell_count = stored_cell_count(grid)
  axis_count = dimension(grid)
  @boundscheck 1 <= cell <= cell_count || _throw_index_error(cell, cell_count, "cell")
  @boundscheck 1 <= axis <= axis_count || _throw_index_error(axis, axis_count, "axis")
  return _level(grid, Int(cell), Int(axis))
end

"""
    logical_coordinate(grid, cell)
    logical_coordinate(grid, cell, axis)

Return the integer logical coordinate of a cell, either on all axes or on one
specified axis.

Together with [`level`](@ref), the logical coordinate identifies the cell within
the dyadically refined root lattice. On axis `a`, the valid range is
`0:(root_cell_count(grid, a) * 2^ℓₐ - 1)`.
"""
@inline function logical_coordinate(grid::CartesianGrid{D}, cell::Integer) where {D}
  count = stored_cell_count(grid)
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _logical_coordinate_tuple(grid, Int(cell))
end

@inline function logical_coordinate(grid::CartesianGrid, cell::Integer, axis::Integer)
  cell_count = stored_cell_count(grid)
  axis_count = dimension(grid)
  @boundscheck 1 <= cell <= cell_count || _throw_index_error(cell, cell_count, "cell")
  @boundscheck 1 <= axis <= axis_count || _throw_index_error(axis, axis_count, "axis")
  return _logical_coordinate(grid, Int(cell), Int(axis))
end

# Parent/child relations in the refinement tree.

"""
    parent(grid, cell)

Return the parent of `cell`, or [`NONE`](@ref) if `cell` is a root cell.

Parent links define the refinement tree and are used to climb from a fine cell
to coarser coverings when resolving nonmatching neighbors.
"""
@inline function parent(grid::CartesianGrid, cell::Integer)
  count = stored_cell_count(grid)
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _parent(grid, Int(cell))
end

"""
    first_child(grid, cell)

Return the first child of an expanded cell, or [`NONE`](@ref) if the cell is not
currently expanded.

Children are stored in contiguous blocks of length two because refinement always
splits one axis at its midpoint.
"""
@inline function first_child(grid::CartesianGrid, cell::Integer)
  count = stored_cell_count(grid)
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _first_child(grid, Int(cell))
end

"""
    split_axis(grid, cell)

Return the axis along which `cell` was split, or `0` if the cell is not
currently expanded.

This records the anisotropic refinement direction of the most recent split that
produced the current child block.
"""
@inline function split_axis(grid::CartesianGrid, cell::Integer)
  count = stored_cell_count(grid)
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _split_axis(grid, Int(cell))
end

# Face-neighborhood queries.

"""
    neighbor(grid, cell, axis, side)

Return the direct same-level tree neighbor of `cell` across the selected face.

The returned cell, if present, has the same logical level as `cell` and shares
the corresponding full face at that level. If no such same-level tree cell
exists, the function returns [`NONE`](@ref). For nonmatching interfaces, use
[`covering_neighbor`](@ref) or [`opposite_active_leaves`](@ref) instead.
"""
function neighbor(grid::CartesianGrid, cell::Integer, axis::Integer, side::Integer)
  cell_count = stored_cell_count(grid)
  axis_count = dimension(grid)
  @boundscheck 1 <= cell <= cell_count || _throw_index_error(cell, cell_count, "cell")
  @boundscheck 1 <= axis <= axis_count || _throw_index_error(axis, axis_count, "axis")
  checked_side = _checked_side(side)
  return _neighbor(grid, Int(cell), Int(axis), checked_side)
end

"""
    is_active_leaf(grid, cell)

Return `true` if `cell` is an active leaf of the current refinement tree.

Active leaves are the cells that currently partition the domain. They are not
expanded further and therefore represent the elements used by discretization and
integration routines.
"""
@inline function is_active_leaf(grid::CartesianGrid, cell::Integer)
  count = stored_cell_count(grid)
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _is_active_leaf(grid, Int(cell))
end

@inline function is_active_leaf(snapshot::GridSnapshot, cell::Integer)
  _require_current_snapshot(snapshot)
  count = stored_cell_count(grid(snapshot))
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _is_active_leaf(snapshot, Int(cell))
end

"""
    is_expanded(grid, cell)

Return `true` if `cell` has been refined into children.

Expanded cells remain part of the tree as ancestors, but they are no longer
active leaves.
"""
@inline function is_expanded(grid::CartesianGrid, cell::Integer)
  count = stored_cell_count(grid)
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _is_expanded(grid, Int(cell))
end

@inline function is_expanded(snapshot::GridSnapshot, cell::Integer)
  _require_current_snapshot(snapshot)
  count = stored_cell_count(grid(snapshot))
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _is_expanded(snapshot, Int(cell))
end

"""
    is_tree_cell(grid, cell)

Return `true` if `cell` currently belongs to the refinement tree.

This is the union of active leaves and expanded ancestors. It excludes retired
stored cells that no longer participate in the live tree after derefinement.
"""
@inline function is_tree_cell(grid::CartesianGrid, cell::Integer)
  count = stored_cell_count(grid)
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  return _is_tree_cell(grid, Int(cell))
end

@inline function is_tree_cell(snapshot::GridSnapshot, cell::Integer)
  _require_current_snapshot(snapshot)
  count = stored_cell_count(grid(snapshot))
  @boundscheck 1 <= cell <= count || _throw_index_error(cell, count, "cell")
  checked_cell = Int(cell)
  return _is_active_leaf(snapshot, checked_cell) || _is_expanded(snapshot, checked_cell)
end

"""
    is_domain_boundary(grid, cell, axis, side)

Return `true` if the selected face of `cell` lies on a nonperiodic outer
boundary of the current domain.

The query is phrased in terms of covering cells rather than same-level direct
neighbors, so it remains valid on nonmatching meshes. On periodic axes, the
corresponding outer faces are not considered domain boundaries.
"""
function is_domain_boundary(grid::CartesianGrid, cell::Integer, axis::Integer, side::Integer)
  covering_neighbor(grid, cell, axis, side) == NONE
end

"""
    covering_neighbor(grid, cell, axis, side)

Return the coarsest tree cell on the opposite side whose face covers the
selected face patch of `cell`.

Same-level direct neighbors are derived from the logical tree. When the
opposite side is coarser, this function climbs the ancestor chain until it finds
a same-level neighbor that geometrically covers the queried face. The returned
cell may therefore be active or expanded. If no covering cell exists, the
function returns [`NONE`](@ref).
"""
function covering_neighbor(grid::CartesianGrid, cell::Integer, axis::Integer, side::Integer)
  current = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  checked_side = _checked_side(side)
  lookup = _tree_cell_lookup(grid)
  return _covering_neighbor(grid, lookup, current, checked_axis, checked_side)
end

function _covering_neighbor(grid::CartesianGrid, lookup, current::Int, checked_axis::Int,
                            checked_side::Int)
  while true
    # Direct neighbors are resolved on the current tree-cell lookup. When a fine
    # face abuts a coarser cell, we ascend until the missing face segment reaches
    # an ancestor whose opposite side is represented at the same level.
    candidate = _neighbor(grid, lookup, current, checked_axis, checked_side)
    candidate != NONE && return candidate
    current_parent = _parent(grid, current)
    current_parent == NONE && return NONE
    _touches_parent_boundary(grid, current, checked_axis, checked_side) ||
      throw(ArgumentError("cell $current is missing a covering neighbor"))
    current = current_parent
  end
end

"""
    opposite_active_leaves(grid, cell, axis, side)

Return the active leaves whose union covers the opposite side of the selected
face of `cell`.

The result is empty on a nonperiodic domain boundary. Otherwise the returned
active leaves are sorted deterministically in tangential order along the face.
This is the key primitive for building nonmatching face relations in the space
and integration layers.
"""
function opposite_active_leaves(grid::CartesianGrid, cell::Integer, axis::Integer, side::Integer)
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  checked_side = _checked_side(side)
  lookup = _tree_cell_lookup(grid)
  candidate = _covering_neighbor(grid, lookup, checked_cell, checked_axis, checked_side)
  candidate == NONE && return Int[]
  leaves = Int[]

  # First collect the active leaves on the opposite side whose tangential
  # intervals overlap the queried face patch.
  _collect_opposite_active_leaves!(leaves, grid, checked_cell, checked_axis, checked_side,
                                   candidate)
  isempty(leaves) && return leaves

  # Then sort tangentially on a common finest comparison level so later face
  # traversals see a deterministic geometric ordering even across hp mismatch.
  comparison_levels = [maximum(_level(grid, leaf, current_axis)
                               for leaf in Iterators.flatten(((checked_cell,), leaves)))
                       for current_axis in 1:dimension(grid)]
  sort!(leaves;
        by=leaf -> ntuple(current_axis -> current_axis == checked_axis ? 0 :
                                          _scaled_lower_coordinate(grid, leaf, current_axis,
                                                                   comparison_levels[current_axis]),
                          dimension(grid)))
  return leaves
end

function opposite_active_leaves(snapshot::GridSnapshot, cell::Integer, axis::Integer, side::Integer)
  _require_current_snapshot(snapshot)
  grid_data = grid(snapshot)
  checked_cell = _checked_cell(grid_data, cell)
  checked_axis = _checked_axis(grid_data, axis)
  checked_side = _checked_side(side)
  tree_lookup = _snapshot_tree_cell_lookup(grid_data, snapshot.active_leaves)
  return _snapshot_opposite_active_leaves(grid_data, tree_lookup, snapshot.leaf_to_index,
                                          checked_cell, checked_axis, checked_side)
end

function boundary_face_count(snapshot::GridSnapshot)
  return length(_require_current_snapshot(snapshot).boundary_leaf)
end

function interface_count(snapshot::GridSnapshot)
  return length(_require_current_snapshot(snapshot).interface_minus)
end

function boundary_face_spec(snapshot::GridSnapshot, index::Integer)
  _require_current_snapshot(snapshot)
  count = boundary_face_count(snapshot)
  @boundscheck 1 <= index <= count || _throw_index_error(index, count, "boundary face")
  checked_index = Int(index)
  return GridBoundaryFace(@inbounds(snapshot.boundary_leaf[checked_index]),
                          Int(@inbounds(snapshot.boundary_axis[checked_index])),
                          Int(@inbounds(snapshot.boundary_side[checked_index])))
end

function interface_spec(snapshot::GridSnapshot, index::Integer)
  _require_current_snapshot(snapshot)
  count = interface_count(snapshot)
  @boundscheck 1 <= index <= count || _throw_index_error(index, count, "interface")
  checked_index = Int(index)
  return GridInterface(@inbounds(snapshot.interface_minus[checked_index]),
                       Int(@inbounds(snapshot.interface_axis[checked_index])),
                       @inbounds(snapshot.interface_plus[checked_index]))
end

# Copying and validation.

# Copying a grid duplicates the topology arrays while preserving the logical
# structure and periodic metadata. The copied grid is validated via the
# constructor like any other materialized topology object.
function Base.copy(grid::CartesianGrid{D}) where {D}
  return CartesianGrid{D}(grid.root_counts, ntuple(axis -> copy(grid.levels[axis]), D),
                          ntuple(axis -> copy(grid.coords[axis]), D), copy(grid.parent),
                          copy(grid.first_child), copy(grid.split_axis), copy(grid.active),
                          copy(grid.active_leaves), grid.periodic, grid.revision)
end

"""
    check_topology(grid)

Validate the internal consistency of a `CartesianGrid`.

This routine checks that the topology arrays have matching sizes, that every
stored cell satisfies the refinement-tree invariants, that the explicit
`active_leaves` list agrees with the `active` bit vector, and that live tree
cells have unique logical signatures.

It is mainly a debugging, testing, and constructor-validation tool. Public
constructors and topology copies already call it automatically.
"""
# Internal consistency check for the complete topological state. Constructors and
# tests use this to ensure that the refinement arrays, active-leaf list, and
# logical tree structure are internally consistent.
function check_topology(grid::CartesianGrid)
  stored = stored_cell_count(grid)
  all(count >= 1 for count in grid.root_counts) ||
    throw(ArgumentError("root cell counts must be positive"))
  lengths = (length(grid.parent), length(grid.first_child), length(grid.split_axis),
             length(grid.active))
  all(length == stored for length in lengths) ||
    throw(ArgumentError("topology arrays must have matching lengths"))

  for axis in 1:dimension(grid)
    length(grid.levels[axis]) == stored || throw(ArgumentError("level array length mismatch"))
    length(grid.coords[axis]) == stored || throw(ArgumentError("coordinate array length mismatch"))
  end

  computed_active = Int[]

  for cell in 1:stored
    _is_active_leaf(grid, cell) && push!(computed_active, cell)
    _check_topology_cell!(grid, cell)
  end

  computed_active == grid.active_leaves || throw(ArgumentError("active-leaf list is inconsistent"))
  _check_live_tree_structure!(grid)
  _tree_cell_lookup(grid)
  return nothing
end

# Structural rebuild helpers used by refinement and derefinement.

# Append storage for a new binary child block. The block is initialized as two
# active leaves with empty topology links; the caller is responsible for filling
# in the actual parent, level, and coordinate information.
function _append_child_block!(grid::CartesianGrid{D}) where {D}
  first = stored_cell_count(grid) + 1

  for _ in 1:_MIDPOINT_CHILD_COUNT
    push!(grid.parent, NONE)
    push!(grid.first_child, NONE)
    push!(grid.split_axis, 0)
    push!(grid.active, true)

    for axis in 1:D
      push!(grid.levels[axis], 0)
      push!(grid.coords[axis], 0)
    end
  end

  return first
end

# Recompute the sorted active-leaf list from the `active` bit vector. This is
# done after structural topology updates so downstream code can rely on a dense,
# deterministic leaf enumeration.
function _rebuild_active_leaves!(grid::CartesianGrid)
  stored = stored_cell_count(grid)
  empty!(grid.active_leaves)

  for cell in 1:stored
    @inbounds grid.active[cell] && push!(grid.active_leaves, cell)
  end

  return grid
end

# Geometric comparison helpers on the logical lattice.

# Scale interval endpoints to a common finest level so cells with different
# refinement levels can be compared on one integer lattice without roundoff.
function _scaled_lower_coordinate(grid::CartesianGrid, cell::Int, axis::Int, target_level::Int)
  return Int128(_logical_coordinate(grid, cell, axis)) << (target_level - _level(grid, cell, axis))
end

function _scaled_upper_coordinate(grid::CartesianGrid, cell::Int, axis::Int, target_level::Int)
  return (Int128(_logical_coordinate(grid, cell, axis)) + 1) <<
         (target_level - _level(grid, cell, axis))
end

# Determine whether a child touches the queried side of its parent. This is the
# condition under which a missing same-level neighbor may be resolved by moving
# to the parent when searching for a covering cell.
function _touches_parent_boundary(grid::CartesianGrid, cell::Int, axis::Int, side::Int)
  parent_cell = _parent(grid, cell)
  parent_cell == NONE && return false
  split = _split_axis(grid, parent_cell)
  split != axis && return true
  return (_logical_coordinate(grid, cell, axis) & 1) == _side_bit(side)
end

# Two face patches interact only if their intervals overlap in every tangential
# direction. The comparison is performed on a common dyadic lattice so it works
# uniformly across nonmatching refinement levels.
function _tangential_intervals_overlap(grid::CartesianGrid{D}, first::Int, second::Int,
                                       face_axis::Int) where {D}
  for axis in 1:D
    axis == face_axis && continue
    comparison_level = max(_level(grid, first, axis), _level(grid, second, axis))
    first_lower = _scaled_lower_coordinate(grid, first, axis, comparison_level)
    first_upper = _scaled_upper_coordinate(grid, first, axis, comparison_level)
    second_lower = _scaled_lower_coordinate(grid, second, axis, comparison_level)
    second_upper = _scaled_upper_coordinate(grid, second, axis, comparison_level)
    (first_lower >= second_upper || second_lower >= first_upper) && return false
  end

  return true
end

# Recursively descend from a covering neighbor to the active leaves that touch
# the queried face patch. Splits normal to the face select only the adjacent
# child, whereas tangential splits must explore both children because both may
# contribute to the face decomposition.
function _collect_opposite_active_leaves!(leaves::Vector{Int}, grid::CartesianGrid, source::Int,
                                          axis::Int, side::Int, candidate::Int)
  _tangential_intervals_overlap(grid, source, candidate, axis) || return leaves

  if _is_active_leaf(grid, candidate)
    push!(leaves, candidate)
    return leaves
  end

  split = _split_axis(grid, candidate)
  split == 0 && return leaves
  first = _first_child(grid, candidate)

  if split == axis
    # Only the child adjacent to the queried face can touch the source cell when
    # the candidate was split normal to that face.
    child = first + (side == LOWER ? 1 : 0)
    _collect_opposite_active_leaves!(leaves, grid, source, axis, side, child)
    return leaves
  end

  # A tangential split partitions the opposite face patch itself, so both
  # children may contribute active leaves.
  for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
    _collect_opposite_active_leaves!(leaves, grid, source, axis, side, child)
  end

  return leaves
end

# Validation and reconstruction helpers.

# Validate one stored cell record against the refinement-tree invariants. This
# checks logical coordinate ranges, the distinction between active and expanded
# cells, and the consistency of parent/child relationships.
function _check_topology_cell!(grid::CartesianGrid, cell::Int)
  for axis in 1:dimension(grid)
    lvl = grid.levels[axis][cell]
    coord = grid.coords[axis][cell]
    extent = _checked_axis_extent(grid.root_counts[axis], lvl, axis)
    0 <= coord < extent || throw(ArgumentError("logical coordinate out of bounds"))
  end

  _is_active_leaf(grid, cell) &&
    _is_expanded(grid, cell) &&
    throw(ArgumentError("cell cannot be active and expanded"))

  first = _first_child(grid, cell)
  first == NONE || _check_retained_child_block!(grid, cell, first)

  if _is_expanded(grid, cell)
    first != NONE || throw(ArgumentError("expanded cell must own children"))
    checked_axis = _split_axis(grid, cell)
    1 <= checked_axis <= dimension(grid) || throw(ArgumentError("invalid split axis"))

    for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
      _is_tree_cell(grid, child) ||
        throw(ArgumentError("expanded cell child must belong to the live tree"))
    end
  end

  return nothing
end

function _check_retained_child_block!(grid::CartesianGrid, cell::Int, first::Int)
  1 <= first <= stored_cell_count(grid) - _MIDPOINT_CHILD_COUNT + 1 ||
    throw(ArgumentError("retained child block is out of bounds"))

  for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
    grid.parent[child] == cell || throw(ArgumentError("retained child parent mismatch"))
  end

  return nothing
end

function _check_live_tree_structure!(grid::CartesianGrid)
  stored = stored_cell_count(grid)
  root_total = root_cell_total(grid)
  root_total <= stored || throw(ArgumentError("stored topology is missing root cells"))

  for root in 1:root_total
    grid.parent[root] == NONE || throw(ArgumentError("root cell parent must be NONE"))
    _is_tree_cell(grid, root) || throw(ArgumentError("root cell must belong to the live tree"))
  end

  for cell in (root_total+1):stored
    parent_cell = grid.parent[cell]
    parent_cell != NONE || throw(ArgumentError("non-root cell must have a parent"))
    1 <= parent_cell <= stored || throw(ArgumentError("cell parent is out of bounds"))
  end

  for cell in 1:stored
    _is_tree_cell(grid, cell) || continue
    parent_cell = grid.parent[cell]

    if cell <= root_total
      parent_cell == NONE || throw(ArgumentError("root cell parent must be NONE"))
      continue
    end

    parent_cell != NONE || throw(ArgumentError("live non-root cell must have a parent"))
    1 <= parent_cell <= stored || throw(ArgumentError("live cell parent is out of bounds"))
    _is_tree_cell(grid, parent_cell) ||
      throw(ArgumentError("live cell parent must belong to the live tree"))
    _is_expanded(grid, parent_cell) || throw(ArgumentError("live cell parent must be expanded"))
    first = _first_child(grid, parent_cell)
    first <= cell <= first + _MIDPOINT_CHILD_COUNT - 1 ||
      throw(ArgumentError("live child is outside its parent child block"))
  end

  return nothing
end

function _require_current_snapshot(snapshot::GridSnapshot)
  snapshot.generation == revision(grid(snapshot)) ||
    throw(ArgumentError("grid snapshot has been invalidated by a topology generation change"))
  return snapshot
end

function _snapshot_leaf_lookup(grid::CartesianGrid, active::AbstractVector{<:Integer})
  lookup = zeros(Int, stored_cell_count(grid))

  for index in eachindex(active)
    leaf = _checked_cell(grid, active[index])
    lookup[leaf] == 0 || throw(ArgumentError("snapshot active leaves must be unique"))
    lookup[leaf] = index
  end

  return lookup
end

function _snapshot_axis_byte(axis::Int)
  axis <= typemax(UInt8) || throw(ArgumentError("snapshot axis must be UInt8-representable"))
  return UInt8(axis)
end

_snapshot_side_byte(side::Int) = UInt8(_checked_side(side))

function _snapshot_boundary_data(grid::CartesianGrid{D},
                                 active::AbstractVector{<:Integer}) where {D}
  tree_lookup = _snapshot_tree_cell_lookup(grid, active)
  boundary_leaf = Int[]
  boundary_axis = UInt8[]
  boundary_side = UInt8[]
  sizehint!(boundary_leaf, length(active) * D)
  sizehint!(boundary_axis, length(active) * D)
  sizehint!(boundary_side, length(active) * D)

  for leaf in active, axis in 1:D, side in (LOWER, UPPER)
    _snapshot_covering_neighbor(grid, tree_lookup, Int(leaf), axis, side) == NONE || continue
    push!(boundary_leaf, Int(leaf))
    push!(boundary_axis, _snapshot_axis_byte(axis))
    push!(boundary_side, _snapshot_side_byte(side))
  end

  return boundary_leaf, boundary_axis, boundary_side
end

function _snapshot_interface_data(grid::CartesianGrid{D}, active::AbstractVector{<:Integer},
                                  leaf_to_index::AbstractVector{<:Integer}) where {D}
  tree_lookup = _snapshot_tree_cell_lookup(grid, active)
  interface_minus = Int[]
  interface_plus = Int[]
  interface_axis = UInt8[]
  sizehint!(interface_minus, length(active) * D)
  sizehint!(interface_plus, length(active) * D)
  sizehint!(interface_axis, length(active) * D)

  for leaf in active, axis in 1:D
    for other in
        _snapshot_opposite_active_leaves(grid, tree_lookup, leaf_to_index, Int(leaf), axis, UPPER)
      push!(interface_minus, Int(leaf))
      push!(interface_plus, other)
      push!(interface_axis, _snapshot_axis_byte(axis))
    end
  end

  return interface_minus, interface_plus, interface_axis
end

function _snapshot_tree_cell_lookup(grid::CartesianGrid{D},
                                    active::AbstractVector{<:Integer}) where {D}
  lookup = Dict{Tuple{NTuple{D,Int},NTuple{D,Int}},Int}()

  for leaf in active
    current = _checked_cell(grid, leaf)

    while current != NONE
      signature = (_level_tuple(grid, current), _logical_coordinate_tuple(grid, current))
      stored = get(lookup, signature, NONE)
      stored == NONE ||
        stored == current ||
        throw(ArgumentError("snapshot tree cells must have unique logical signatures"))
      lookup[signature] = current
      current = _parent(grid, current)
    end
  end

  return lookup
end

function _snapshot_direct_neighbor(grid::CartesianGrid{D}, tree_lookup, cell::Int,
                                   axis::Int) where {D}
  return _expected_direct_neighbors(grid, tree_lookup, _level_tuple(grid, cell),
                                    _logical_coordinate_tuple(grid, cell), axis)
end

function _snapshot_covering_neighbor(grid::CartesianGrid, tree_lookup, current::Int, axis::Int,
                                     side::Int)
  while true
    lower, upper = _snapshot_direct_neighbor(grid, tree_lookup, current, axis)
    candidate = side == LOWER ? lower : upper
    candidate != NONE && return candidate
    current_parent = _parent(grid, current)
    current_parent == NONE && return NONE
    _snapshot_touches_parent_boundary(grid, current, axis, side) || return NONE
    current = current_parent
  end
end

function _snapshot_opposite_active_leaves(grid::CartesianGrid{D}, tree_lookup,
                                          leaf_to_index::AbstractVector{<:Integer}, cell::Int,
                                          axis::Int, side::Int) where {D}
  candidate = _snapshot_covering_neighbor(grid, tree_lookup, cell, axis, side)
  candidate == NONE && return Int[]
  leaves = Int[]
  _collect_snapshot_opposite_active_leaves!(leaves, grid, tree_lookup, leaf_to_index, cell, axis,
                                            side, candidate)
  isempty(leaves) && return leaves
  comparison_levels = [maximum(_level(grid, leaf, current_axis)
                               for leaf in Iterators.flatten(((cell,), leaves)))
                       for current_axis in 1:D]
  sort!(leaves;
        by=leaf -> ntuple(current_axis -> current_axis == axis ? 0 :
                                          _scaled_lower_coordinate(grid, leaf, current_axis,
                                                                   comparison_levels[current_axis]),
                          D))
  return leaves
end

function _collect_snapshot_opposite_active_leaves!(leaves::Vector{Int}, grid::CartesianGrid,
                                                   tree_lookup,
                                                   leaf_to_index::AbstractVector{<:Integer},
                                                   source::Int, axis::Int, side::Int,
                                                   candidate::Int)
  _tangential_intervals_overlap(grid, source, candidate, axis) || return leaves

  if candidate <= length(leaf_to_index) && @inbounds(leaf_to_index[candidate] != 0)
    push!(leaves, candidate)
    return leaves
  end

  split = _structural_split_axis(grid, candidate)
  split == 0 && return leaves
  first = _first_child(grid, candidate)

  if split == axis
    child = first + (side == LOWER ? 1 : 0)
    _snapshot_cell_in_tree(grid, tree_lookup, child) ||
      throw(ArgumentError("snapshot tree is missing an active face child"))
    _collect_snapshot_opposite_active_leaves!(leaves, grid, tree_lookup, leaf_to_index, source,
                                              axis, side, child)
    return leaves
  end

  for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
    _snapshot_cell_in_tree(grid, tree_lookup, child) || continue
    _collect_snapshot_opposite_active_leaves!(leaves, grid, tree_lookup, leaf_to_index, source,
                                              axis, side, child)
  end

  return leaves
end

function _snapshot_cell_in_tree(grid::CartesianGrid, tree_lookup, cell::Int)
  signature = (_level_tuple(grid, cell), _logical_coordinate_tuple(grid, cell))
  return get(tree_lookup, signature, NONE) == cell
end

function _structural_split_axis(grid::CartesianGrid, cell::Int)
  split = _split_axis(grid, cell)
  split != 0 && return split
  first = _first_child(grid, cell)
  first == NONE && return 0
  child = first
  for axis in 1:dimension(grid)
    if _level(grid, child, axis) == _level(grid, cell, axis) + 1
      return axis
    end
  end
  return 0
end

function _snapshot_touches_parent_boundary(grid::CartesianGrid, cell::Int, axis::Int, side::Int)
  parent_cell = _parent(grid, cell)
  parent_cell == NONE && return false
  split = _structural_split_axis(grid, parent_cell)
  split != axis && return true
  return (_logical_coordinate(grid, cell, axis) & 1) == _side_bit(side)
end

function _check_snapshot_boundary_data(snapshot::GridSnapshot)
  lengths = (length(snapshot.boundary_leaf), length(snapshot.boundary_axis),
             length(snapshot.boundary_side))
  all(length == first(lengths) for length in lengths) ||
    throw(ArgumentError("snapshot boundary arrays must have matching lengths"))
  expected_leaf, expected_axis, expected_side = _snapshot_boundary_data(grid(snapshot),
                                                                        snapshot.active_leaves)
  snapshot.boundary_leaf == expected_leaf ||
    throw(ArgumentError("snapshot boundary leaves are inconsistent"))
  snapshot.boundary_axis == expected_axis ||
    throw(ArgumentError("snapshot boundary axes are inconsistent"))
  snapshot.boundary_side == expected_side ||
    throw(ArgumentError("snapshot boundary sides are inconsistent"))
  return nothing
end

function _check_snapshot_interface_data(snapshot::GridSnapshot)
  lengths = (length(snapshot.interface_minus), length(snapshot.interface_plus),
             length(snapshot.interface_axis))
  all(length == first(lengths) for length in lengths) ||
    throw(ArgumentError("snapshot interface arrays must have matching lengths"))
  expected_minus, expected_plus, expected_axis = _snapshot_interface_data(grid(snapshot),
                                                                          snapshot.active_leaves,
                                                                          snapshot.leaf_to_index)
  snapshot.interface_minus == expected_minus ||
    throw(ArgumentError("snapshot interface minus leaves are inconsistent"))
  snapshot.interface_plus == expected_plus ||
    throw(ArgumentError("snapshot interface plus leaves are inconsistent"))
  snapshot.interface_axis == expected_axis ||
    throw(ArgumentError("snapshot interface axes are inconsistent"))
  return nothing
end

function _snapshot_has_active_descendant(snapshot::GridSnapshot, cell::Int)
  _is_active_leaf(snapshot, cell) && return false
  first = _first_child(grid(snapshot), cell)
  first == NONE && return false

  for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
    (_is_active_leaf(snapshot, child) || _snapshot_has_active_descendant(snapshot, child)) &&
      return true
  end

  return false
end

# Tree cells are uniquely identified by their per-axis levels and logical
# coordinates, so a dictionary on `(level, coordinate)` is enough to recover
# same-level neighbors after arbitrary refinement updates.
function _tree_cell_lookup(grid::CartesianGrid{D}) where {D}
  lookup = Dict{Tuple{NTuple{D,Int},NTuple{D,Int}},Int}()

  for cell in 1:stored_cell_count(grid)
    _is_tree_cell(grid, cell) || continue
    signature = (_level_tuple(grid, cell), _logical_coordinate_tuple(grid, cell))
    haskey(lookup, signature) &&
      throw(ArgumentError("tree cells must have unique logical signatures"))
    lookup[signature] = cell
  end

  return lookup
end

# Recover the expected lower/upper same-level neighbors for one tree-cell
# signature from the logical coordinate step rule and the live tree-cell lookup.
function _expected_direct_neighbors(grid::CartesianGrid{D}, lookup, cell_level::NTuple{D,Int},
                                    cell_coord::NTuple{D,Int}, axis::Int) where {D}
  lower_coord = _neighbor_coordinate(grid, cell_level, cell_coord, axis, LOWER)
  upper_coord = _neighbor_coordinate(grid, cell_level, cell_coord, axis, UPPER)
  lower = isnothing(lower_coord) ? NONE : get(lookup, (cell_level, lower_coord), NONE)
  upper = isnothing(upper_coord) ? NONE : get(lookup, (cell_level, upper_coord), NONE)
  return lower, upper
end

function _expected_direct_neighbors(grid::CartesianGrid{D}, lookup, cell::Int, axis::Int) where {D}
  return _expected_direct_neighbors(grid, lookup, _level_tuple(grid, cell),
                                    _logical_coordinate_tuple(grid, cell), axis)
end

function _neighbor(grid::CartesianGrid, cell::Int, axis::Int, side::Int)
  return _neighbor(grid, _tree_cell_lookup(grid), cell, axis, side)
end

function _neighbor(grid::CartesianGrid, lookup, cell::Int, axis::Int, side::Int)
  _is_tree_cell(grid, cell) || return NONE
  lower, upper = _expected_direct_neighbors(grid, lookup, cell, axis)
  return side == LOWER ? lower : upper
end

function _filtered_upper_face_neighbor_specs(grid::CartesianGrid{D},
                                             active::AbstractVector{<:Integer},
                                             leaf_to_index::AbstractVector{<:Integer}) where {D}
  specs = Tuple{Int,Int,Int}[]

  for leaf in active, axis in 1:D
    for other in opposite_active_leaves(grid, leaf, axis, UPPER)
      @inbounds leaf_to_index[other] == 0 && continue
      push!(specs, (leaf, axis, other))
    end
  end

  return specs
end

# Input normalization and low-level argument checks.

function _checked_root_cell_total(root_counts::NTuple{D,Int}) where {D}
  total = Int128(1)

  for axis in 1:D
    count = root_counts[axis]
    count >= 1 || throw(ArgumentError("root_counts[$axis] must be positive"))
    total *= Int128(count)
    total <= typemax(Int) || throw(ArgumentError("root cell total must be Int-representable"))
  end

  return Int(total)
end

function _checked_axis_extent(root_count::Int, level::Int, axis::Int)
  level >= 0 || throw(ArgumentError("level[$axis] must be non-negative"))
  level < Sys.WORD_SIZE - 1 ||
    throw(ArgumentError("logical extent on axis $axis must be Int-representable"))
  extent = Int128(root_count) << level
  extent <= typemax(Int) ||
    throw(ArgumentError("logical extent on axis $axis must be Int-representable"))
  return Int(extent)
end

function _checked_refinement_level(level::Int, axis::Int)
  level >= 0 || throw(ArgumentError("level[$axis] must be non-negative"))
  level < typemax(Int) ||
    throw(ArgumentError("refinement level on axis $axis must be Int-representable"))
  return level + 1
end

function _checked_child_coordinate(parent_coord::Int, child_offset::Int, axis::Int)
  child_coord = 2 * Int128(parent_coord) + child_offset
  0 <= child_coord <= typemax(Int) ||
    throw(ArgumentError("child coordinate on axis $axis must be Int-representable"))
  return Int(child_coord)
end

# Normalize the periodicity specification to a per-axis tuple and validate its
# shape. This keeps the public constructor flexible while storing one canonical
# representation internally.
function _checked_periodic_axes(periodic, D::Int)
  if periodic isa Bool
    return ntuple(_ -> periodic, D)
  end

  periodic isa Tuple || throw(ArgumentError("periodic must be a Bool or NTuple{$D,Bool}"))
  length(periodic) == D || throw(ArgumentError("periodic must have length $D"))
  return ntuple(axis -> begin
                  value = periodic[axis]
                  value isa Bool || throw(ArgumentError("periodic[$axis] must be a Bool"))
                  value
                end, D)
end

# Step one logical cell across a face on a fixed refinement level. On periodic
# axes the logical coordinate wraps around the root extent; otherwise a step past
# the outer boundary returns `nothing`.
function _neighbor_coordinate(grid::CartesianGrid{D}, cell_level::NTuple{D,Int},
                              cell_coord::NTuple{D,Int}, axis::Int, side::Int) where {D}
  axis_extent = _checked_axis_extent(grid.root_counts[axis], cell_level[axis], axis)
  next_coord = cell_coord[axis] + (side == LOWER ? -1 : 1)

  if !(0 <= next_coord < axis_extent)
    _is_periodic_axis(grid, axis) || return nothing
    next_coord = mod(next_coord, axis_extent)
  end

  return ntuple(current_axis -> current_axis == axis ? next_coord : cell_coord[current_axis], D)
end

_side_bit(side::Int) = side - LOWER

function _checked_cell(grid::CartesianGrid, cell::Integer)
  return _require_index(cell, stored_cell_count(grid), "cell")
end

_checked_axis(grid::CartesianGrid, axis::Integer) = _require_index(axis, dimension(grid), "axis")

function _checked_side(side::Integer)
  (side == LOWER || side == UPPER) || throw(ArgumentError("side must be LOWER or UPPER"))
  return Int(side)
end

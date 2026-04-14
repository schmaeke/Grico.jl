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
# - direct neighbor tables store only same-level tree neighbors,
# - higher-level queries recover coverings and active opposite leaves from those
#   same-level tables,
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
active-leaf flags, same-level direct neighbors, periodic wrapping, and a
revision counter. Physical coordinates and measures are handled separately by
`Geometry` and `Domain`; this type represents the discrete refinement topology
on which continuity, interface, and adaptivity logic operate.

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
  neighbor_lower::NTuple{D,Vector{Int}}
  neighbor_upper::NTuple{D,Vector{Int}}
  periodic::NTuple{D,Bool}
  revision::UInt

  # Low-level constructor for already materialized topology arrays. All code
  # paths that build a grid eventually pass through `check_topology`, so the
  # constructor is also the consistency gate for copied or rebuilt grids.
  function CartesianGrid{D}(root_counts::NTuple{D,Int}, levels::NTuple{D,Vector{Int}},
                            coords::NTuple{D,Vector{Int}}, parent::Vector{Int},
                            first_child::Vector{Int}, split_axis::Vector{Int}, active::BitVector,
                            active_leaves::Vector{Int}, neighbor_lower::NTuple{D,Vector{Int}},
                            neighbor_upper::NTuple{D,Vector{Int}}, periodic::NTuple{D,Bool},
                            revision::UInt) where {D}
    D >= 1 || throw(ArgumentError("dimension must be positive"))
    checked_counts = ntuple(axis -> _checked_positive(root_counts[axis], "root_counts[$axis]"), D)
    grid = new{D}(checked_counts, levels, coords, parent, first_child, split_axis, active,
                  active_leaves, neighbor_lower, neighbor_upper, periodic, revision)
    check_topology(grid)
    return grid
  end
end

# Backward-compatible internal constructor that defaults to a nonperiodic root
# grid when periodic metadata are not supplied explicitly.
function CartesianGrid{D}(root_counts::NTuple{D,Int}, levels::NTuple{D,Vector{Int}},
                          coords::NTuple{D,Vector{Int}}, parent::Vector{Int},
                          first_child::Vector{Int}, split_axis::Vector{Int}, active::BitVector,
                          active_leaves::Vector{Int}, neighbor_lower::NTuple{D,Vector{Int}},
                          neighbor_upper::NTuple{D,Vector{Int}}, revision::UInt) where {D}
  return CartesianGrid{D}(root_counts, levels, coords, parent, first_child, split_axis, active,
                          active_leaves, neighbor_lower, neighbor_upper, ntuple(_ -> false, D),
                          revision)
end

# Build the unrefined root tree and initialize the same-level neighbor tables.
# Root cells are numbered in mixed-radix order with axis 1 varying fastest, so
# the direct neighbors can be reconstructed from strides on the root lattice.
function CartesianGrid(root_counts::NTuple{D,<:Integer}; periodic=false) where {D}
  D >= 1 || throw(ArgumentError("dimension must be positive"))
  checked_counts = ntuple(axis -> _checked_positive(root_counts[axis], "root_counts[$axis]"), D)
  checked_periodic = _checked_periodic_axes(periodic, D)
  cell_total = prod(checked_counts)
  levels = ntuple(_ -> zeros(Int, cell_total), D)
  coords = ntuple(_ -> zeros(Int, cell_total), D)
  parent = fill(NONE, cell_total)
  first_child = fill(NONE, cell_total)
  split_axis = zeros(Int, cell_total)
  active = trues(cell_total)
  active_leaves = collect(1:cell_total)
  neighbor_lower = ntuple(_ -> fill(NONE, cell_total), D)
  neighbor_upper = ntuple(_ -> fill(NONE, cell_total), D)
  strides = ntuple(axis -> axis == 1 ? 1 : prod(checked_counts[1:(axis-1)]), D)

  for cell in 1:cell_total
    linear = cell - 1

    @inbounds for axis in 1:D
      coord = fld(linear, strides[axis]) % checked_counts[axis]
      coords[axis][cell] = coord

      # At the root level, direct neighbors are obtained by stepping one cell on
      # the tensor-product lattice. Periodic axes wrap to the opposite side,
      # while nonperiodic axes use `NONE` to mark the outer boundary.
      if coord == 0
        neighbor_lower[axis][cell] = checked_periodic[axis] ?
                                     cell + (checked_counts[axis] - 1) * strides[axis] : NONE
      else
        neighbor_lower[axis][cell] = cell - strides[axis]
      end

      if coord + 1 == checked_counts[axis]
        neighbor_upper[axis][cell] = checked_periodic[axis] ?
                                     cell - (checked_counts[axis] - 1) * strides[axis] : NONE
      else
        neighbor_upper[axis][cell] = cell + strides[axis]
      end
    end
  end

  return CartesianGrid{D}(checked_counts, levels, coords, parent, first_child, split_axis, active,
                          active_leaves, neighbor_lower, neighbor_upper, checked_periodic,
                          zero(UInt))
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
root_cell_count(grid::CartesianGrid, axis::Integer) = grid.root_counts[_checked_axis(grid, axis)]

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
is_periodic_axis(grid::CartesianGrid, axis::Integer) = grid.periodic[_checked_axis(grid, axis)]

"""
    root_cell_total(grid)

Return the total number of root cells in the unrefined tensor-product mesh.

This is the product of [`root_cell_counts`](@ref) and equals the number of
stored cells in a freshly constructed grid before any refinement occurs.
"""
root_cell_total(grid::CartesianGrid) = prod(grid.root_counts)

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
connectivity. Higher-level caches can use it to detect when topological data
have become stale.
"""
revision(grid::CartesianGrid) = grid.revision

"""
    active_leaf_count(grid)

Return the number of active leaves in the current refinement tree.

Active leaves are the cells that currently partition the domain and therefore
act as the discrete elements seen by spaces, integration, and assembly.
"""
active_leaf_count(grid::CartesianGrid) = length(grid.active_leaves)

"""
    active_leaves(grid)

Return the active leaves of the current tree as a newly allocated vector.

The returned vector is sorted by cell index and is independent of the internal
storage, so callers may reorder or modify it without mutating the grid.
"""
active_leaves(grid::CartesianGrid) = copy(grid.active_leaves)

# Cell-local tree queries.

"""
    active_leaf(grid, index)

Return the `index`-th active leaf of the current tree.

This provides indexed access to the ordering used by [`active_leaves`](@ref)
and [`active_leaf_count`](@ref).
"""
function active_leaf(grid::CartesianGrid, index::Integer)
  @inbounds grid.active_leaves[_checked_index(index, active_leaf_count(grid), "active leaf")]
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
function level(grid::CartesianGrid{D}, cell::Integer) where {D}
  checked = _checked_cell(grid, cell)
  return ntuple(axis -> @inbounds(grid.levels[axis][checked]), D)
end

function level(grid::CartesianGrid, cell::Integer, axis::Integer)
  @inbounds grid.levels[_checked_axis(grid, axis)][_checked_cell(grid, cell)]
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
function logical_coordinate(grid::CartesianGrid{D}, cell::Integer) where {D}
  checked = _checked_cell(grid, cell)
  return ntuple(axis -> @inbounds(grid.coords[axis][checked]), D)
end

function logical_coordinate(grid::CartesianGrid, cell::Integer, axis::Integer)
  @inbounds grid.coords[_checked_axis(grid, axis)][_checked_cell(grid, cell)]
end

# Parent/child relations in the refinement tree.

"""
    parent(grid, cell)

Return the parent of `cell`, or [`NONE`](@ref) if `cell` is a root cell.

Parent links define the refinement tree and are used to climb from a fine cell
to coarser coverings when resolving nonmatching neighbors.
"""
parent(grid::CartesianGrid, cell::Integer) = @inbounds grid.parent[_checked_cell(grid, cell)]

"""
    first_child(grid, cell)

Return the first child of an expanded cell, or [`NONE`](@ref) if the cell is not
currently expanded.

Children are stored in contiguous blocks of length two because refinement always
splits one axis at its midpoint.
"""
function first_child(grid::CartesianGrid, cell::Integer)
  @inbounds grid.first_child[_checked_cell(grid, cell)]
end

"""
    split_axis(grid, cell)

Return the axis along which `cell` was split, or `0` if the cell is not
currently expanded.

This records the anisotropic refinement direction of the most recent split that
produced the current child block.
"""
function split_axis(grid::CartesianGrid, cell::Integer)
  @inbounds grid.split_axis[_checked_cell(grid, cell)]
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
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  checked_side = _checked_side(side)
  return @inbounds (checked_side == LOWER ? grid.neighbor_lower : grid.neighbor_upper)[checked_axis][checked_cell]
end

"""
    is_active_leaf(grid, cell)

Return `true` if `cell` is an active leaf of the current refinement tree.

Active leaves are the cells that currently partition the domain. They are not
expanded further and therefore represent the elements used by discretization and
integration routines.
"""
function is_active_leaf(grid::CartesianGrid, cell::Integer)
  @inbounds grid.active[_checked_cell(grid, cell)]
end

"""
    is_expanded(grid, cell)

Return `true` if `cell` has been refined into children.

Expanded cells remain part of the tree as ancestors, but they are no longer
active leaves.
"""
is_expanded(grid::CartesianGrid, cell::Integer) = split_axis(grid, cell) != 0

"""
    is_tree_cell(grid, cell)

Return `true` if `cell` currently belongs to the refinement tree.

This is the union of active leaves and expanded ancestors. It excludes retired
stored cells that no longer participate in the live tree after derefinement.
"""
function is_tree_cell(grid::CartesianGrid, cell::Integer)
  is_active_leaf(grid, cell) || is_expanded(grid, cell)
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

Direct neighbor tables store only same-level tree neighbors. When the opposite
side is coarser, this function climbs the ancestor chain until it finds a
same-level neighbor that geometrically covers the queried face. The returned
cell may therefore be active or expanded. If no covering cell exists, the
function returns [`NONE`](@ref).
"""
function covering_neighbor(grid::CartesianGrid, cell::Integer, axis::Integer, side::Integer)
  current = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  checked_side = _checked_side(side)

  while true
    # The direct neighbor tables resolve only same-level tree neighbors. When a
    # fine face abuts a coarser cell, we ascend until the missing face segment
    # reaches an ancestor whose opposite side is represented at the same level.
    candidate = @inbounds _neighbor_array(grid, checked_side)[checked_axis][current]
    candidate != NONE && return candidate
    current_parent = @inbounds grid.parent[current]
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
  candidate = covering_neighbor(grid, checked_cell, checked_axis, checked_side)
  candidate == NONE && return Int[]
  leaves = Int[]

  # First collect the active leaves on the opposite side whose tangential
  # intervals overlap the queried face patch.
  _collect_opposite_active_leaves!(leaves, grid, checked_cell, checked_axis, checked_side,
                                   candidate)
  isempty(leaves) && return leaves

  # Then sort tangentially on a common finest comparison level so later face
  # traversals see a deterministic geometric ordering even across hp mismatch.
  comparison_levels = [maximum(level(grid, leaf, current_axis)
                               for leaf in Iterators.flatten(((checked_cell,), leaves)))
                       for current_axis in 1:dimension(grid)]
  sort!(leaves;
        by=leaf -> ntuple(current_axis -> current_axis == checked_axis ? 0 :
                                          _scaled_lower_coordinate(grid, leaf, current_axis,
                                                                   comparison_levels[current_axis]),
                          dimension(grid)))
  return leaves
end

# Copying and validation.

# Copying a grid duplicates the topology arrays while preserving the logical
# structure and periodic metadata. The copied grid is validated via the
# constructor like any other materialized topology object.
function Base.copy(grid::CartesianGrid{D}) where {D}
  return CartesianGrid{D}(grid.root_counts, ntuple(axis -> copy(grid.levels[axis]), D),
                          ntuple(axis -> copy(grid.coords[axis]), D), copy(grid.parent),
                          copy(grid.first_child), copy(grid.split_axis), copy(grid.active),
                          copy(grid.active_leaves),
                          ntuple(axis -> copy(grid.neighbor_lower[axis]), D),
                          ntuple(axis -> copy(grid.neighbor_upper[axis]), D), grid.periodic,
                          grid.revision)
end

"""
    check_topology(grid)

Validate the internal consistency of a `CartesianGrid`.

This routine checks that the topology arrays have matching sizes, that every
stored cell satisfies the refinement-tree invariants, that the explicit
`active_leaves` list agrees with the `active` bit vector, and that the stored
same-level neighbor tables match the logical tree description.

It is mainly a debugging, testing, and constructor-validation tool. Public
constructors and topology copies already call it automatically.
"""
# Internal consistency check for the complete topological state. Constructors and
# tests use this to ensure that the refinement arrays, active-leaf list, and
# direct-neighbor tables all encode the same tree.
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
    length(grid.neighbor_lower[axis]) == stored ||
      throw(ArgumentError("neighbor array length mismatch"))
    length(grid.neighbor_upper[axis]) == stored ||
      throw(ArgumentError("neighbor array length mismatch"))
  end

  computed_active = Int[]

  for cell in 1:stored
    is_active_leaf(grid, cell) && push!(computed_active, cell)
    _check_topology_cell!(grid, cell)
  end

  computed_active == grid.active_leaves || throw(ArgumentError("active-leaf list is inconsistent"))
  _check_direct_neighbor_tables!(grid)
  return nothing
end

# Structural rebuild helpers used by refinement and derefinement.

# Reconstruct root logical coordinates from mixed-radix cell numbering. This is
# mainly useful when a grid is materialized from raw arrays and the root portion
# needs to be reinitialized consistently.
function _initialize_root_coordinates!(grid::CartesianGrid{D}) where {D}
  strides = ntuple(axis -> axis == 1 ? 1 : prod(grid.root_counts[1:(axis-1)]), D)

  for cell in 1:root_cell_total(grid)
    linear = cell - 1

    @inbounds for axis in 1:D
      grid.coords[axis][cell] = fld(linear, strides[axis]) % grid.root_counts[axis]
    end
  end

  return grid
end

# Append storage for a new binary child block. The block is initialized as two
# active leaves with empty topology links; the caller is responsible for filling
# in the actual parent, level, coordinate, and neighbor information.
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
      push!(grid.neighbor_lower[axis], NONE)
      push!(grid.neighbor_upper[axis], NONE)
    end
  end

  return first
end

# Recompute the sorted active-leaf list from the `active` bit vector. This is
# done after structural topology updates so downstream code can rely on a dense,
# deterministic leaf enumeration.
function _rebuild_active_leaves!(grid::CartesianGrid)
  stored = stored_cell_count(grid)
  worker_count = max(1, min(Threads.nthreads(), stored))
  thread_active = [Int[] for _ in 1:worker_count]

  _run_chunks_with_scratch!(thread_active, stored) do leaves, first_cell, last_cell
    for cell in first_cell:last_cell
      @inbounds grid.active[cell] && push!(leaves, cell)
    end
  end

  empty!(grid.active_leaves)

  for leaves in thread_active
    append!(grid.active_leaves, leaves)
  end

  sort!(grid.active_leaves)

  return grid
end

# Rebuild same-level direct-neighbor tables for all live tree cells. The tables
# do not try to resolve hanging faces directly; instead they store exact
# same-level neighbors, and higher-level routines recover coverings and active
# face decompositions from those data.
function _rebuild_direct_neighbors!(grid::CartesianGrid{D}) where {D}
  stored = stored_cell_count(grid)

  _run_chunks!(stored) do first_cell, last_cell
    for cell in first_cell:last_cell
      @inbounds for axis in 1:D
        grid.neighbor_lower[axis][cell] = NONE
        grid.neighbor_upper[axis][cell] = NONE
      end
    end
  end

  lookup = _tree_cell_lookup(grid)

  _run_chunks!(stored) do first_cell, last_cell
    for cell in first_cell:last_cell
      is_tree_cell(grid, cell) || continue
      cell_level = level(grid, cell)
      cell_coord = logical_coordinate(grid, cell)

      @inbounds for axis in 1:D
        lower, upper = _expected_direct_neighbors(grid, lookup, cell_level, cell_coord, axis)
        grid.neighbor_lower[axis][cell] = lower
        grid.neighbor_upper[axis][cell] = upper
      end
    end
  end

  return grid
end

# Geometric comparison helpers on the logical lattice.

function _neighbor_array(grid::CartesianGrid, side::Int)
  return side == LOWER ? grid.neighbor_lower : grid.neighbor_upper
end

# Scale interval endpoints to a common finest level so cells with different
# refinement levels can be compared on one integer lattice without roundoff.
function _scaled_lower_coordinate(grid::CartesianGrid, cell::Int, axis::Int, target_level::Int)
  return Int128(logical_coordinate(grid, cell, axis)) << (target_level - level(grid, cell, axis))
end

function _scaled_upper_coordinate(grid::CartesianGrid, cell::Int, axis::Int, target_level::Int)
  return (Int128(logical_coordinate(grid, cell, axis)) + 1) <<
         (target_level - level(grid, cell, axis))
end

# Determine whether a child touches the queried side of its parent. This is the
# condition under which a missing same-level neighbor may be resolved by moving
# to the parent when searching for a covering cell.
function _touches_parent_boundary(grid::CartesianGrid, cell::Int, axis::Int, side::Int)
  parent_cell = parent(grid, cell)
  parent_cell == NONE && return false
  split = split_axis(grid, parent_cell)
  split != axis && return true
  return (logical_coordinate(grid, cell, axis) & 1) == _side_bit(side)
end

# Two face patches interact only if their intervals overlap in every tangential
# direction. The comparison is performed on a common dyadic lattice so it works
# uniformly across nonmatching refinement levels.
function _tangential_intervals_overlap(grid::CartesianGrid{D}, first::Int, second::Int,
                                       face_axis::Int) where {D}
  for axis in 1:D
    axis == face_axis && continue
    comparison_level = max(level(grid, first, axis), level(grid, second, axis))
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

  if is_active_leaf(grid, candidate)
    push!(leaves, candidate)
    return leaves
  end

  split = split_axis(grid, candidate)
  split == 0 && return leaves
  first = first_child(grid, candidate)

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
    lvl = @inbounds grid.levels[axis][cell]
    coord = @inbounds grid.coords[axis][cell]
    lvl >= 0 || throw(ArgumentError("negative refinement level"))
    0 <= coord < (grid.root_counts[axis] << lvl) ||
      throw(ArgumentError("logical coordinate out of bounds"))
  end

  is_active_leaf(grid, cell) &&
    is_expanded(grid, cell) &&
    throw(ArgumentError("cell cannot be active and expanded"))

  if is_expanded(grid, cell)
    first = first_child(grid, cell)
    first != NONE || throw(ArgumentError("expanded cell must own children"))
    checked_axis = split_axis(grid, cell)
    1 <= checked_axis <= dimension(grid) || throw(ArgumentError("invalid split axis"))

    for child in first:(first+_MIDPOINT_CHILD_COUNT-1)
      @inbounds grid.parent[child] == cell || throw(ArgumentError("child parent mismatch"))
    end
  end

  return nothing
end

# Recompute what the same-level direct-neighbor tables should contain from the
# logical tree description alone, then compare that result with the stored
# tables. This detects inconsistent rebuilds or invalid copied topologies.
function _check_direct_neighbor_tables!(grid::CartesianGrid{D}) where {D}
  lookup = _tree_cell_lookup(grid)

  for cell in 1:stored_cell_count(grid)
    for axis in 1:D
      expected_lower, expected_upper = is_tree_cell(grid, cell) ?
                                       _expected_direct_neighbors(grid, lookup, cell, axis) :
                                       (NONE, NONE)

      @inbounds grid.neighbor_lower[axis][cell] == expected_lower ||
                throw(ArgumentError("lower neighbor table is inconsistent"))
      @inbounds grid.neighbor_upper[axis][cell] == expected_upper ||
                throw(ArgumentError("upper neighbor table is inconsistent"))
    end
  end

  return nothing
end

# Tree cells are uniquely identified by their per-axis levels and logical
# coordinates, so a dictionary on `(level, coordinate)` is enough to recover
# same-level neighbors after arbitrary refinement updates or to validate stored
# neighbor tables against the logical tree description.
function _tree_cell_lookup(grid::CartesianGrid{D}) where {D}
  lookup = Dict{Tuple{NTuple{D,Int},NTuple{D,Int}},Int}()

  for cell in 1:stored_cell_count(grid)
    is_tree_cell(grid, cell) || continue
    lookup[(level(grid, cell), logical_coordinate(grid, cell))] = cell
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
  return _expected_direct_neighbors(grid, lookup, level(grid, cell), logical_coordinate(grid, cell),
                                    axis)
end

# Enumerate each active face interface exactly once by visiting only upper-side
# faces. Downstream code uses these tuples to build continuity relations and
# interface integration data without creating duplicate face pairs.
function _upper_face_neighbor_specs(grid::CartesianGrid{D}) where {D}
  specs = Tuple{Int,Int,Int}[]

  for leaf in active_leaves(grid), axis in 1:D
    for other in opposite_active_leaves(grid, leaf, axis, UPPER)
      push!(specs, (leaf, axis, other))
    end
  end

  return specs
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
  axis_extent = grid.root_counts[axis] << cell_level[axis]
  next_coord = cell_coord[axis] + (side == LOWER ? -1 : 1)

  if !(0 <= next_coord < axis_extent)
    is_periodic_axis(grid, axis) || return nothing
    next_coord = mod(next_coord, axis_extent)
  end

  return ntuple(current_axis -> current_axis == axis ? next_coord : cell_coord[current_axis], D)
end

_side_bit(side::Int) = side - LOWER

function _checked_cell(grid::CartesianGrid, cell::Integer)
  return _checked_index(cell, stored_cell_count(grid), "cell")
end

_checked_axis(grid::CartesianGrid, axis::Integer) = _checked_index(axis, dimension(grid), "axis")

function _checked_side(side::Integer)
  checked = Int(side)
  (checked == LOWER || checked == UPPER) || throw(ArgumentError("side must be LOWER or UPPER"))
  return checked
end

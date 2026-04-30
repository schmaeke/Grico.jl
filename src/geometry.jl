# This file defines the physical geometry layer on top of the Cartesian
# refinement tree.
#
# Whereas `topology.jl` answers purely discrete questions such as which cells
# exist and how they touch, this file answers the corresponding metric
# questions:
# - where a logical cell sits in physical space,
# - how large it is on each axis,
# - what its volume or face measure is,
# - and how points are mapped between reference coordinates and physical
#   coordinates.
#
# The geometry model is deliberately simple: the whole domain is one global
# axis-aligned affine box, and every refined cell inherits its physical
# placement by restricting that affine map to its logical subcell. This gives
# several important consequences:
# - all cell maps remain affine,
# - all Jacobians are diagonal,
# - refinement affects metric factors only through dyadic scaling,
# - and geometry stays completely separate from the refinement-tree logic.
#
# The file is therefore easiest to read in five blocks:
# 1. `Geometry` and `Domain`, which pair the affine box with the topological
#    tree;
# 2. basic accessors for origins, extents, and the underlying grid;
# 3. cell and face metric queries such as bounds, sizes, and measures;
# 4. reference-to-physical maps and their Jacobian factors on `[0, 1]^D` and
#    `[-1, 1]^D`;
# 5. the small internal helpers that derive every metric quantity from the same
#    axis-local affine data.

# Core affine objects.

"""
    Geometry(origin, extent)

Axis-aligned affine geometry for a Cartesian grid.

`Geometry` stores the physical origin `x₀` and extent `L` of a rectangular
domain

  Ω = ∏ₐ [x₀ₐ, x₀ₐ + Lₐ].

It contains no refinement information by itself; instead it provides the affine
metric data that turn the logical cells of a `CartesianGrid` into physical
cells, faces, and coordinate mappings. All coordinate transforms in this file
are diagonal because the geometry is axis-aligned and globally affine.

`origin` and `extent` must have matching dimension, finite entries, and
strictly positive extents on every axis.

`Geometry` is immutable because the physical box is conceptually the fixed
background on which the refinement tree evolves. Adaptive refinement changes
the logical partition of that box, not the box itself.
"""
struct Geometry{D,T<:AbstractFloat}
  origin::NTuple{D,T}
  extent::NTuple{D,T}

  # Validate that the affine box is well-defined before storing it. The
  # topological grid may later refine anisotropically, but the physical base box
  # itself must remain finite and strictly positive on every axis.
  function Geometry{D,T}(origin::NTuple{D,T}, extent::NTuple{D,T}) where {D,T<:AbstractFloat}
    D >= 1 || throw(ArgumentError("dimension must be positive"))
    checked_origin = ntuple(axis -> begin
                              value = origin[axis]
                              isfinite(value) ||
                                throw(ArgumentError("origin[$axis] must be finite"))
                              value
                            end, D)
    checked_extent = ntuple(axis -> begin
                              value = extent[axis]
                              isfinite(value) ||
                                throw(ArgumentError("extent[$axis] must be finite"))
                              value > zero(T) ||
                                throw(ArgumentError("extent[$axis] must be positive"))
                              value
                            end, D)
    return new{D,T}(checked_origin, checked_extent)
  end
end

"""
    Domain(origin, extent, root_counts)
    Domain(grid, geometry)

Pair a Cartesian refinement tree with its physical affine geometry.

`Domain` is the background mesh object of the library. It combines the logical
refinement topology stored in a `CartesianGrid` with the physical affine box
stored in a `Geometry`, so queries such as cell bounds, volumes, face measures,
and reference-to-physical mappings can be expressed directly on one object.

The topology and geometry are intentionally separated: the grid decides how the
domain is refined, while the geometry decides where that refinement lives in
physical space. This keeps refinement, periodicity, and continuity logic
independent of physical scaling.

Most user-facing geometric queries are provided on `Domain` so callers can work
with one combined object instead of passing topology and geometry separately.
Higher layers may then wrap that background domain, for example in
[`PhysicalDomain`](@ref), without changing these geometric queries. The
lower-level `Geometry` plus `CartesianGrid` methods remain available because the
assembly and integration layers sometimes operate on those components directly.
"""
abstract type AbstractDomain{D,T<:AbstractFloat} end

struct Domain{D,T<:AbstractFloat} <: AbstractDomain{D,T}
  grid::CartesianGrid{D}
  geometry::Geometry{D,T}

  # Construct a domain from already materialized topology and geometry data.
  # The grid is revalidated here so downstream geometric queries can assume a
  # consistent logical tree.
  function Domain{D,T}(grid::CartesianGrid{D}, geometry::Geometry{D,T}) where {D,T<:AbstractFloat}
    check_topology(grid)
    return new{D,T}(grid, geometry)
  end
end

function Geometry(origin::Tuple{Vararg{Real,D}}, extent::Tuple{Vararg{Real,D}}) where {D}
  T = float(promote_type(typeof.(origin)..., typeof.(extent)...))
  return Geometry{D,T}(ntuple(axis -> T(origin[axis]), D), ntuple(axis -> T(extent[axis]), D))
end

# Convenience constructor that builds the root grid and the affine box together.
# The `periodic` keyword is forwarded directly to the underlying `CartesianGrid`.
function Domain(origin::Tuple{Vararg{Real,D}}, extent::Tuple{Vararg{Real,D}},
                root_counts::Tuple{Vararg{Integer,D}}; periodic=false) where {D}
  return Domain(CartesianGrid(root_counts; periodic=periodic), Geometry(origin, extent))
end

function Domain(grid::CartesianGrid{D}, geometry::Geometry{D,T}) where {D,T<:AbstractFloat}
  Domain{D,T}(grid, geometry)
end

# Basic paired accessors.

"""
    dimension(geometry)
    dimension(domain)

Return the spatial dimension `D` of the affine box or paired domain.

This equals the number of coordinate axes and therefore the length of tuples
such as origins, extents, logical coordinates, and physical points.
"""
dimension(geometry::Geometry{D}) where {D} = D
dimension(domain::AbstractDomain) = dimension(grid(domain))

"""
    grid(domain)

Return the `CartesianGrid` that stores the logical refinement topology of
`domain`.

Use this when a routine needs tree-level information such as active leaves,
neighbors, or periodicity rather than physical coordinates.
"""
grid(domain::Domain) = domain.grid

"""
    geometry(domain)

Return the `Geometry` that stores the affine physical box of `domain`.

Use this when a routine works with physical coordinates, cell sizes, or
reference-to-physical mappings.
"""
geometry(domain::Domain) = domain.geometry
periodic_axes(domain::AbstractDomain) = periodic_axes(grid(domain))
is_periodic_axis(domain::AbstractDomain, axis::Integer) = is_periodic_axis(grid(domain), axis)

"""
    origin(geometry)
    origin(geometry, axis)
    origin(domain)
    origin(domain, axis)

Return the physical origin of the affine box, either as a full tuple or on one
selected axis.

The origin is the lower corner `x₀` of the rectangular domain
`Ω = ∏ₐ [x₀ₐ, x₀ₐ + Lₐ]`.
"""
origin(geometry::Geometry) = geometry.origin
origin(domain::AbstractDomain) = origin(geometry(domain))
@inline function origin(geometry::Geometry, axis::Integer)
  dimension_count = dimension(geometry)
  @boundscheck 1 <= axis <= dimension_count || _throw_index_error(axis, dimension_count, "axis")
  return @inbounds geometry.origin[Int(axis)]
end
origin(domain::AbstractDomain, axis::Integer) = origin(geometry(domain), axis)

"""
    extent(geometry)
    extent(geometry, axis)
    extent(domain)
    extent(domain, axis)

Return the physical extent of the affine box, either as a full tuple or on one
selected axis.

The extent stores the side lengths `Lₐ` of the rectangular domain.
"""
extent(geometry::Geometry) = geometry.extent
extent(domain::AbstractDomain) = extent(geometry(domain))
@inline function extent(geometry::Geometry, axis::Integer)
  dimension_count = dimension(geometry)
  @boundscheck 1 <= axis <= dimension_count || _throw_index_error(axis, dimension_count, "axis")
  return @inbounds geometry.extent[Int(axis)]
end
extent(domain::AbstractDomain, axis::Integer) = extent(geometry(domain), axis)

# Copy semantics.

# Copying a domain duplicates the topological refinement tree while reusing the
# immutable affine box data. The result is geometrically identical but
# topologically independent.
function Base.copy(domain::Domain{D,T}) where {D,T<:AbstractFloat}
  Domain(copy(domain.grid), domain.geometry)
end

"""
    compact(domain, snapshot)

Build a new domain whose grid keeps only the tree cells needed to represent
`snapshot`.

This is non-mutating: `domain`, its grid, and all snapshots of that grid remain
valid. The returned tuple is `(compacted_domain, compacted_snapshot,
old_to_new_cell)`, where `old_to_new_cell[old_cell]` is the corresponding cell
id in the compacted grid or [`NONE`](@ref) if the old cell was pruned.
"""
function compact(domain::Domain{D,T}, active_snapshot::GridSnapshot{D}) where {D,T<:AbstractFloat}
  grid(active_snapshot) === grid(domain) ||
    throw(ArgumentError("snapshot must reference the domain grid"))
  compact_snapshot, old_to_new = _compact_grid_snapshot(active_snapshot)
  compact_domain = Domain(grid(compact_snapshot), geometry(domain))
  return compact_domain, compact_snapshot, old_to_new
end

# Cell- and face-level metric queries.

"""
    cell_size(domain, cell, axis)

Return the physical side length of `cell` on one axis.

If the root extent on axis `a` is `Lₐ` and the root mesh has `nₐ` cells there,
then a cell with logical level `ℓₐ` has size

  hₐ = Lₐ / (nₐ 2^ℓₐ).

This is the basic metric quantity from which cell bounds, face measures, and
Jacobian factors are derived.
"""
function cell_size(domain::AbstractDomain, cell::Integer, axis::Integer)
  cell_size(geometry(domain), grid(domain), cell, axis)
end

"""
    cell_lower(domain, cell)
    cell_lower(domain, cell, axis)

Return the lower physical corner of `cell`, either as a full tuple or on one
selected axis.

The lower corner is obtained by combining the cell's logical coordinate with
the affine root scaling of the domain.
"""
cell_lower(domain::AbstractDomain, cell::Integer) = cell_lower(geometry(domain), grid(domain), cell)
function cell_lower(domain::AbstractDomain, cell::Integer, axis::Integer)
  cell_lower(geometry(domain), grid(domain), cell, axis)
end

"""
    cell_upper(domain, cell)
    cell_upper(domain, cell, axis)

Return the upper physical corner of `cell`, either as a full tuple or on one
selected axis.

Together with [`cell_lower`](@ref), this describes the closed affine box of the
cell in physical coordinates.
"""
cell_upper(domain::AbstractDomain, cell::Integer) = cell_upper(geometry(domain), grid(domain), cell)
function cell_upper(domain::AbstractDomain, cell::Integer, axis::Integer)
  cell_upper(geometry(domain), grid(domain), cell, axis)
end

"""
    cell_center(domain, cell)
    cell_center(domain, cell, axis)

Return the physical midpoint of `cell`, either as a full tuple or on one
selected axis.

Because the geometry is axis-aligned and affine, the center is simply the
midpoint of the lower and upper bounds on each axis.
"""
function cell_center(domain::AbstractDomain, cell::Integer)
  cell_center(geometry(domain), grid(domain), cell)
end
function cell_center(domain::AbstractDomain, cell::Integer, axis::Integer)
  cell_center(geometry(domain), grid(domain), cell, axis)
end

"""
    cell_volume(domain, cell)

Return the physical volume, area, or length of `cell`, depending on dimension.

For an axis-aligned affine cell, the measure is

  |K| = ∏ₐ hₐ,

where `hₐ` are the per-axis cell sizes.
"""
function cell_volume(domain::AbstractDomain, cell::Integer)
  cell_volume(geometry(domain), grid(domain), cell)
end

"""
    face_measure(domain, cell, axis)

Return the physical measure of the face of `cell` normal to `axis`.

In `D` dimensions this is a `(D-1)`-dimensional measure equal to the product of
the cell sizes on all tangential axes.
"""
function face_measure(domain::AbstractDomain, cell::Integer, axis::Integer)
  face_measure(geometry(domain), grid(domain), cell, axis)
end

# Reference-to-physical maps and Jacobian data on `Domain`.

"""
    map_from_unit_cube(domain, cell, ξ)

Map a reference point `ξ ∈ [0,1]^D` to the physical cell `cell`.

For an affine Cartesian cell with lower corner `xₗ` and side lengths `h`, the
map is

  x(ξ) = xₗ + ξ ⊙ h,

where `⊙` denotes componentwise multiplication.
"""
function map_from_unit_cube(domain::AbstractDomain{D}, cell::Integer, ξ::NTuple{D,<:Real}) where {D}
  map_from_unit_cube(geometry(domain), grid(domain), cell, ξ)
end

"""
    map_from_biunit_cube(domain, cell, ξ)

Map a reference point `ξ ∈ [-1,1]^D` to the physical cell `cell`.

This is the tensor-product affine map commonly used by finite-element basis and
quadrature code:

  x(ξ) = x_c + 1/2 (ξ ⊙ h),

where `x_c` is the cell center and `h` is the vector of cell sizes.
"""
function map_from_biunit_cube(domain::AbstractDomain{D}, cell::Integer,
                              ξ::NTuple{D,<:Real}) where {D}
  map_from_biunit_cube(geometry(domain), grid(domain), cell, ξ)
end

"""
    map_to_biunit_cube(domain, cell, x)

Map a physical point `x` in `cell` back to biunit reference coordinates
`ξ ∈ [-1,1]^D`.

This is the inverse of [`map_from_biunit_cube`](@ref) for the affine geometry
used by this library.
"""
function map_to_biunit_cube(domain::AbstractDomain{D}, cell::Integer, x::NTuple{D,<:Real}) where {D}
  map_to_biunit_cube(geometry(domain), grid(domain), cell, x)
end

"""
    jacobian_diagonal_from_unit_cube(domain, cell, axis)

Return the diagonal Jacobian entry `∂xₐ/∂ξₐ` of the map from `[0,1]^D` to
`cell` on one axis.

Because the geometry is axis-aligned and affine, the Jacobian is diagonal and
this entry equals the cell size on that axis.
"""
function jacobian_diagonal_from_unit_cube(domain::AbstractDomain, cell::Integer, axis::Integer)
  jacobian_diagonal_from_unit_cube(geometry(domain), grid(domain), cell, axis)
end

"""
    jacobian_diagonal_from_biunit_cube(domain, cell, axis)

Return the diagonal Jacobian entry `∂xₐ/∂ξₐ` of the map from `[-1,1]^D` to
`cell` on one axis.

Relative to the unit-cube map, the factor is halved because each biunit
reference interval has length `2`.
"""
function jacobian_diagonal_from_biunit_cube(domain::AbstractDomain, cell::Integer, axis::Integer)
  jacobian_diagonal_from_biunit_cube(geometry(domain), grid(domain), cell, axis)
end

"""
    jacobian_determinant_from_unit_cube(domain, cell)

Return the Jacobian determinant of the affine map from `[0,1]^D` to `cell`.

For the axis-aligned geometry used here, this determinant equals the physical
cell measure.
"""
function jacobian_determinant_from_unit_cube(domain::AbstractDomain, cell::Integer)
  jacobian_determinant_from_unit_cube(geometry(domain), grid(domain), cell)
end

"""
    jacobian_determinant_from_biunit_cube(domain, cell)

Return the Jacobian determinant of the affine map from `[-1,1]^D` to `cell`.

This equals the physical cell measure divided by `2^D`, reflecting the fact
that the biunit reference box has side length `2` on every axis.
"""
function jacobian_determinant_from_biunit_cube(domain::AbstractDomain, cell::Integer)
  jacobian_determinant_from_biunit_cube(geometry(domain), grid(domain), cell)
end

"""
    map_from_unit_cube!(x, domain, cell, ξ)

Write the image of `ξ ∈ [0,1]^D` in `cell` into the preallocated vector `x`.

This is the allocation-free counterpart of [`map_from_unit_cube`](@ref). The
length of `x` must be at least the spatial dimension.
"""
function map_from_unit_cube!(x::AbstractVector, domain::AbstractDomain{D}, cell::Integer,
                             ξ::NTuple{D,<:Real}) where {D}
  map_from_unit_cube!(x, geometry(domain), grid(domain), cell, ξ)
end

"""
    map_from_biunit_cube!(x, domain, cell, ξ)

Write the image of `ξ ∈ [-1,1]^D` in `cell` into the preallocated vector `x`.

This is the allocation-free counterpart of [`map_from_biunit_cube`](@ref). The
length of `x` must be at least the spatial dimension.
"""
function map_from_biunit_cube!(x::AbstractVector, domain::AbstractDomain{D}, cell::Integer,
                               ξ::NTuple{D,<:Real}) where {D}
  map_from_biunit_cube!(x, geometry(domain), grid(domain), cell, ξ)
end

"""
    map_to_biunit_cube!(ξ, domain, cell, x)

Write the inverse biunit reference coordinates of the physical point `x` in
`cell` into the preallocated vector `ξ`.

This is the allocation-free counterpart of [`map_to_biunit_cube`](@ref). The
length of `ξ` must be at least the spatial dimension.
"""
function map_to_biunit_cube!(ξ::AbstractVector, domain::AbstractDomain{D}, cell::Integer,
                             x::NTuple{D,<:Real}) where {D}
  map_to_biunit_cube!(ξ, geometry(domain), grid(domain), cell, x)
end

# Low-level geometry kernels on `Geometry` plus `CartesianGrid`.
#
# All public `Domain` methods above forward to these routines. This keeps the
# actual affine formulas centralized so there is only one implementation of each
# metric or mapping relation.

# All metric quantities are derived from the same separable affine model. Along
# one axis, the root cell size is `extent/root_count`, and each refinement level
# divides that length by `2`. `ldexp` expresses this dyadic scaling exactly.
function cell_size(geometry::Geometry, grid::CartesianGrid, cell::Integer, axis::Integer)
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  return _cell_size_checked(geometry, grid, checked_cell, checked_axis)
end

# The lower corner is the origin shifted by the logical cell index times the
# physical cell size on each axis.
function cell_lower(geometry::Geometry{D,T}, grid::CartesianGrid, cell::Integer) where {D,T}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return ntuple(axis -> _cell_lower_checked(geometry, grid, checked_cell, axis), D)
end

function cell_lower(geometry::Geometry, grid::CartesianGrid, cell::Integer, axis::Integer)
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  return _cell_lower_checked(geometry, grid, checked_cell, checked_axis)
end

# The upper corner uses the same affine formula as the lower corner, shifted by
# one cell width on the logical lattice.
function cell_upper(geometry::Geometry{D,T}, grid::CartesianGrid, cell::Integer) where {D,T}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return ntuple(axis -> _cell_upper_checked(geometry, grid, checked_cell, axis), D)
end

function cell_upper(geometry::Geometry, grid::CartesianGrid, cell::Integer, axis::Integer)
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  return _cell_upper_checked(geometry, grid, checked_cell, checked_axis)
end

# For an affine Cartesian cell, the center is obtained by adding half a cell
# size to the lower bound on each axis.
function cell_center(geometry::Geometry{D,T}, grid::CartesianGrid, cell::Integer) where {D,T}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return ntuple(axis -> _cell_center_checked(geometry, grid, checked_cell, axis), D)
end

function cell_center(geometry::Geometry, grid::CartesianGrid, cell::Integer, axis::Integer)
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  return _cell_center_checked(geometry, grid, checked_cell, checked_axis)
end

# Cell measure is the product of the diagonal affine scaling factors because the
# geometry is axis-aligned and has no cross-axis distortion.
function cell_volume(geometry::Geometry, grid::CartesianGrid, cell::Integer)
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return _cell_size_product_checked(geometry, grid, checked_cell)
end

# A face normal to one axis has measure equal to the product of the tangential
# cell sizes only.
function face_measure(geometry::Geometry, grid::CartesianGrid, cell::Integer, axis::Integer)
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  return _cell_size_product_checked(geometry, grid, checked_cell, checked_axis)
end

# Map `[0,1]^D` into the physical cell by scaling each reference coordinate with
# the corresponding cell size and shifting by the lower corner.
function map_from_unit_cube(geometry::Geometry{D,T}, grid::CartesianGrid, cell::Integer,
                            ξ::NTuple{D,<:Real}) where {D,T}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return ntuple(axis -> begin
                  lower, size = _cell_affine_axis_data(geometry, grid, checked_cell, axis)
                  lower + T(ξ[axis]) * size
                end, D)
end

# Map `[-1,1]^D` into the physical cell using the cell center and half-widths.
function map_from_biunit_cube(geometry::Geometry{D,T}, grid::CartesianGrid, cell::Integer,
                              ξ::NTuple{D,<:Real}) where {D,T}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return ntuple(axis -> begin
                  center, half_size = _cell_center_and_half_size(geometry, grid, checked_cell, axis)
                  center + T(ξ[axis]) * half_size
                end, D)
end

# Invert the biunit affine map axis by axis. The diagonal Jacobian means each
# reference coordinate can be recovered independently.
function map_to_biunit_cube(geometry::Geometry{D,T}, grid::CartesianGrid, cell::Integer,
                            x::NTuple{D,<:Real}) where {D,T}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return ntuple(axis -> begin
                  center, half_size = _cell_center_and_half_size(geometry, grid, checked_cell, axis)
                  (T(x[axis]) - center) / half_size
                end, D)
end

# The affine Jacobian from `[0,1]^D` is diagonal with entries equal to the
# physical cell sizes.
function jacobian_diagonal_from_unit_cube(geometry::Geometry, grid::CartesianGrid, cell::Integer,
                                          axis::Integer)
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  return _cell_size_checked(geometry, grid, checked_cell, checked_axis)
end

# Relative to `[0,1]^D`, the biunit reference interval rescales each axis by an
# additional factor `1/2`.
function jacobian_diagonal_from_biunit_cube(geometry::Geometry, grid::CartesianGrid, cell::Integer,
                                            axis::Integer)
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  checked_axis = _checked_axis(grid, axis)
  return last(_cell_center_and_half_size(geometry, grid, checked_cell, checked_axis))
end

function jacobian_determinant_from_unit_cube(geometry::Geometry, grid::CartesianGrid, cell::Integer)
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return _cell_size_product_checked(geometry, grid, checked_cell)
end

# The determinant for `[-1,1]^D` is obtained by halving every diagonal Jacobian
# entry, hence the factor `2^{-D}`.
function jacobian_determinant_from_biunit_cube(geometry::Geometry{D}, grid::CartesianGrid,
                                               cell::Integer) where {D}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return ldexp(_cell_size_product_checked(geometry, grid, checked_cell), -D)
end

# Allocation-free mapping wrappers.

# The mutating mapping routines fill caller-provided vectors directly from the
# same axis-local affine data as the tuple-returning maps.
function map_from_unit_cube!(x::AbstractVector, geometry::Geometry{D,T}, grid::CartesianGrid,
                             cell::Integer, ξ::NTuple{D,<:Real}) where {D,T}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return _write_checked_axis_values!(x, D, "x") do axis
    lower, size = _cell_affine_axis_data(geometry, grid, checked_cell, axis)
    lower + T(ξ[axis]) * size
  end
end

function map_from_biunit_cube!(x::AbstractVector, geometry::Geometry{D,T}, grid::CartesianGrid,
                               cell::Integer, ξ::NTuple{D,<:Real}) where {D,T}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return _write_checked_axis_values!(x, D, "x") do axis
    center, half_size = _cell_center_and_half_size(geometry, grid, checked_cell, axis)
    center + T(ξ[axis]) * half_size
  end
end

function map_to_biunit_cube!(ξ::AbstractVector, geometry::Geometry{D,T}, grid::CartesianGrid,
                             cell::Integer, x::NTuple{D,<:Real}) where {D,T}
  _require_matching_geometry_grid(geometry, grid)
  checked_cell = _checked_cell(grid, cell)
  return _write_checked_axis_values!(ξ, D, "ξ") do axis
    center, half_size = _cell_center_and_half_size(geometry, grid, checked_cell, axis)
    (T(x[axis]) - center) / half_size
  end
end

# Shared axis-local affine data and tiny helpers.

@inline function _require_matching_geometry_grid(geometry::Geometry, grid::CartesianGrid)
  dimension(geometry) == dimension(grid) || _throw_geometry_grid_dimension_error(geometry, grid)
  return nothing
end

@noinline function _throw_geometry_grid_dimension_error(geometry::Geometry, grid::CartesianGrid)
  throw(ArgumentError("geometry dimension $(dimension(geometry)) must match grid dimension " *
                      "$(dimension(grid))"))
end

# All cell-local geometric queries derive from the same axis-local affine data:
# the physical lower bound and cell size along one axis.
@inline function _cell_affine_axis_data(geometry::Geometry, grid::CartesianGrid, checked_cell::Int,
                                        axis::Int)
  size = _cell_size_checked(geometry, grid, checked_cell, axis)
  lower = @inbounds geometry.origin[axis] + size * _logical_coordinate(grid, checked_cell, axis)
  return lower, size
end

@inline function _cell_size_checked(geometry::Geometry, grid::CartesianGrid, checked_cell::Int,
                                    checked_axis::Int)
  root_size = @inbounds geometry.extent[checked_axis] / _root_cell_count(grid, checked_axis)
  return ldexp(root_size, -_level(grid, checked_cell, checked_axis))
end

@inline function _cell_lower_checked(geometry::Geometry, grid::CartesianGrid, checked_cell::Int,
                                     axis::Int)
  return first(_cell_affine_axis_data(geometry, grid, checked_cell, axis))
end

@inline function _cell_upper_checked(geometry::Geometry, grid::CartesianGrid, checked_cell::Int,
                                     axis::Int)
  lower, size = _cell_affine_axis_data(geometry, grid, checked_cell, axis)
  return lower + size
end

@inline function _cell_center_checked(geometry::Geometry, grid::CartesianGrid, checked_cell::Int,
                                      axis::Int)
  lower, size = _cell_affine_axis_data(geometry, grid, checked_cell, axis)
  return muladd(size, oftype(size, 0.5), lower)
end

@inline function _cell_center_and_half_size(geometry::Geometry, grid::CartesianGrid,
                                            checked_cell::Int, axis::Int)
  lower, size = _cell_affine_axis_data(geometry, grid, checked_cell, axis)
  half_size = oftype(size, 0.5) * size
  return lower + half_size, half_size
end

@inline function _cell_size_product_checked(geometry::Geometry, grid::CartesianGrid,
                                            checked_cell::Int, skipped_axis::Int=0)
  product = one(eltype(geometry.origin))

  @inbounds for axis in 1:dimension(grid)
    axis == skipped_axis && continue
    product *= _cell_size_checked(geometry, grid, checked_cell, axis)
  end

  return product
end

# Mutating affine maps are thin wrappers that fill one caller-provided output
# vector directly from per-axis affine data, without first materializing an
# intermediate tuple.
function _write_checked_axis_values!(value_at_axis, destination::AbstractVector, axis_count::Int,
                                     name::AbstractString)
  _require_length(destination, axis_count, name)

  @inbounds for axis in 1:axis_count
    destination[axis] = value_at_axis(axis)
  end

  return destination
end

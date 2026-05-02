# This file handles codimension-1 embedded geometry. Starting from either an
# implicit zero set or explicit segment geometry, it constructs per-leaf
# reference-surface quadratures that later compilation and assembly treat like
# any other local integration item.
#
# Codimension-0 finite-cell volume support now lives in `regions.jl`, where
# `PhysicalDomain` and `ImplicitRegion` make the physical-volume story a
# first-class domain feature. This file is therefore focused on embedded
# surfaces and the low-level attachment API for explicit custom quadratures.

# Public embedded-geometry types and attachment API.

"""
    SurfaceQuadrature(leaf, quadrature, normals)

Reference-cell quadrature data for an embedded surface piece inside one active
leaf.

`leaf` identifies the active cell that owns the surface piece. `quadrature`
stores quadrature points and weights on the biunit reference cell `[-1, 1]^D`
of that leaf, and `normals` stores one reference-space unit normal per
quadrature point. During problem compilation, these reference data are mapped to
physical surface points, physical unit normals, and physical measure weights.

This type is the low-level attachment format used by embedded-surface assembly.
It can be constructed directly by users or generated automatically from
[`EmbeddedSurface`](@ref) or [`implicit_surface_quadrature`](@ref).

The stored quadrature weights represent reference-surface measure on the
embedded piece, not full-dimensional cell volume. Later compilation maps both
the weights and the normals to physical space.
"""
struct SurfaceQuadrature{D,T<:AbstractFloat,Q<:PointQuadrature{D,T}}
  leaf::Int
  quadrature::Q
  normals::Vector{NTuple{D,T}}

  function SurfaceQuadrature{D,T,Q}(leaf::Int, quadrature::Q,
                                    normals) where {D,T<:AbstractFloat,Q<:PointQuadrature{D,T}}
    return new{D,T,Q}(_checked_positive(leaf, "leaf"), quadrature,
                      _checked_surface_normals(quadrature, normals, T))
  end
end

"""
    SegmentMesh(points, segments)

Piecewise-linear embedded curve geometry in two dimensions.

`points` is a list of physical points in `ℝ²`, and `segments` is a list of
pairs of point indices describing straight line segments between them. When
wrapped in an [`EmbeddedSurface`](@ref), a `SegmentMesh` is intersected with the
active cells of a two-dimensional domain, and each clipped segment piece is
converted into one or more [`SurfaceQuadrature`](@ref) items.
"""
struct SegmentMesh{T<:AbstractFloat}
  points::Vector{NTuple{2,T}}
  segments::Vector{NTuple{2,Int}}

  function SegmentMesh{T}(points::Vector{NTuple{2,T}},
                          segments::Vector{NTuple{2,Int}}) where {T<:AbstractFloat}
    _validate_segment_mesh(points, segments, T)
    return new{T}(points, segments)
  end
end

"""
    EmbeddedSurface(geometry; point_count=2)

High-level embedded-surface description based on a geometry object.

`EmbeddedSurface` delegates the geometric description of an immersed curve or
surface to `geometry` and requests `point_count` quadrature points on each local
surface piece produced during clipping/intersection. The currently supported
geometry backends are file-local methods of `surface_quadratures`, such as the
[`SegmentMesh`](@ref) backend in two dimensions.

This object is intentionally high level: it stores only the geometric recipe.
Actual [`SurfaceQuadrature`](@ref) items are generated later, once a concrete
problem domain and its active leaves are known.
"""
struct EmbeddedSurface{G}
  geometry::G
  point_count::Int

  function EmbeddedSurface{G}(geometry::G, point_count::Int) where {G}
    return new{G}(geometry, _checked_positive(point_count, "point_count"))
  end
end

const _EMBEDDED_SURFACE_TOLERANCE = T -> T(64) * eps(T)

function SurfaceQuadrature(leaf::Integer, quadrature::PointQuadrature{D,T},
                           normals) where {D,T<:AbstractFloat}
  return SurfaceQuadrature{D,T,typeof(quadrature)}(_checked_positive(leaf, "leaf"), quadrature,
                                                   normals)
end

function SurfaceQuadrature(leaf::Integer, quadrature::PointQuadrature{D,T},
                           normal::NTuple{D,<:Real}) where {D,T<:AbstractFloat}
  return SurfaceQuadrature(leaf, quadrature, fill(normal, point_count(quadrature)))
end

function SegmentMesh(points, segments)
  length(points) >= 2 || throw(ArgumentError("segment meshes require at least two points"))
  !isempty(segments) || throw(ArgumentError("segment meshes require at least one segment"))
  T = float(mapreduce(_segment_point_scalar_type, promote_type, points))
  checked_points = Vector{NTuple{2,T}}(undef, length(points))

  for index in eachindex(points)
    point = points[index]
    checked_points[index] = _checked_segment_point(point, T)
  end

  checked_segments = Vector{NTuple{2,Int}}(undef, length(segments))

  for index in eachindex(segments)
    checked_segments[index] = _checked_segment(segments[index], length(points))
  end

  return SegmentMesh{T}(checked_points, checked_segments)
end

function EmbeddedSurface(geometry::G; point_count=2) where {G}
  point_count isa Integer ||
    throw(ArgumentError("point_count must be a positive Int-representable integer"))
  return EmbeddedSurface{G}(geometry, _checked_positive(point_count, "point_count"))
end

dimension(::SegmentMesh) = 2

# Public problem-building hooks for embedded surfaces and custom cut-cell rules.

"""
    add_embedded_surface!(problem, embedded_surface)
    add_embedded_surface!(problem, tag, embedded_surface)

Attach an [`EmbeddedSurface`](@ref) geometry object to `problem`.

During problem compilation, the geometry is converted to one or more
[`SurfaceQuadrature`](@ref) items on the active cells of the problem domain.
These quadratures are then used by embedded-surface operators. The untagged
form registers the geometry for all surface operators; the tagged form
registers it only for surface operators added with the same symbolic `tag`,
for example `:outer`.
"""
function add_embedded_surface!(problem::_AbstractProblem, embedded_surface::EmbeddedSurface)
  push!(_problem_data(problem).embedded_surfaces, _SurfaceAttachment(nothing, embedded_surface))
  return problem
end

function add_embedded_surface!(problem::_AbstractProblem, tag::Symbol,
                               embedded_surface::EmbeddedSurface)
  push!(_problem_data(problem).embedded_surfaces, _SurfaceAttachment(tag, embedded_surface))
  return problem
end

"""
    add_surface_quadrature!(problem, surface)
    add_surface_quadrature!(problem, tag, surface)
    add_surface_quadrature!(problem, leaf, quadrature, normals)
    add_surface_quadrature!(problem, tag, leaf, quadrature, normals)

Attach an explicit embedded-surface quadrature item to `problem`.

This is the low-level path for users who already know the reference-cell
quadrature points and normals on a particular active leaf. The two-argument
method accepts a prebuilt [`SurfaceQuadrature`](@ref); the four-argument method
constructs one from the given reference quadrature data. As with
[`add_embedded_surface!`](@ref), the tagged forms restrict later surface
operators to the same symbolic `tag`, for example `:outer`.
"""
function add_surface_quadrature!(problem::_AbstractProblem, surface::SurfaceQuadrature)
  push!(_problem_data(problem).embedded_surfaces, _SurfaceAttachment(nothing, surface))
  return problem
end

function add_surface_quadrature!(problem::_AbstractProblem, tag::Symbol, surface::SurfaceQuadrature)
  push!(_problem_data(problem).embedded_surfaces, _SurfaceAttachment(tag, surface))
  return problem
end

"""
    add_cell_quadrature!(problem, leaf, quadrature)

Override the default cell quadrature on one active leaf of `problem`.

The supplied quadrature is defined on the biunit reference cell of `leaf`. This
is primarily used together with finite-cell constructions, where the effective
integration domain inside the leaf differs from the full Cartesian cell.

Later compilation validates that `leaf` is active and that all quadrature points
indeed lie in the standard reference cell `[-1, 1]^D`.
"""
function add_cell_quadrature!(problem::_AbstractProblem, leaf::Integer,
                              quadrature::AbstractQuadrature)
  push!(_problem_data(problem).cell_quadratures,
        _CellQuadratureAttachment(_checked_positive(leaf, "leaf"), quadrature))
  return problem
end

function add_surface_quadrature!(problem::_AbstractProblem, leaf::Integer,
                                 quadrature::PointQuadrature, normals)
  return add_surface_quadrature!(problem, SurfaceQuadrature(leaf, quadrature, normals))
end

function add_surface_quadrature!(problem::_AbstractProblem, tag::Symbol, leaf::Integer,
                                 quadrature::PointQuadrature, normals)
  return add_surface_quadrature!(problem, tag, SurfaceQuadrature(leaf, quadrature, normals))
end

# Public constructors for implicit finite-cell and embedded-surface quadratures.

"""
    implicit_surface_quadrature(space, leaf, classifier; subdivision_depth=2,
                                surface_point_count=...)
    implicit_surface_quadrature(domain, leaf, classifier; subdivision_depth=2,
                                surface_point_count=2)

Construct an embedded-surface quadrature from an implicit level-set-like
classifier on one active leaf.

The classifier must return a finite signed real value in physical coordinates.
Its zero level set approximates the embedded surface, and the sign convention is

  classifier(x) ≤ 0  inside/on the surface,
  classifier(x) > 0  outside.

The algorithm recursively subdivides the reference cell, discards subcells that
do not intersect the zero level set, and on terminal cut subcells constructs a
piecewise-linear local surface approximation. In one dimension this reduces to a
root location on the interval; in two dimensions it reduces to a marching-
squares style segment extraction. The result is a [`SurfaceQuadrature`](@ref) in
reference coordinates or `nothing` if the leaf contains no surface piece.

The `space` method chooses a default `surface_point_count` from the cell
quadrature shape already compiled for the leaf, so the default embedded-surface
resolution tracks the local integration order of the space.

Advanced API: the implicit extractor is a convenience backend for simple
one- and two-dimensional level sets. It is useful for prototyping and teaching,
but robust production workflows should prefer explicit [`SurfaceQuadrature`](@ref)
attachments or a dedicated geometry backend when exact geometry handling is
important.
"""
function implicit_surface_quadrature(space::HpSpace{D,T}, leaf::Integer, classifier;
                                     subdivision_depth=2,
                                     surface_point_count=nothing) where {D,T<:AbstractFloat}
  checked_leaf, leaf_index = _checked_active_leaf_index(grid(space), snapshot(space).leaf_to_index,
                                                        leaf, "embedded surface")
  subdivision_depth isa Integer ||
    throw(ArgumentError("subdivision_depth must be a non-negative Int-representable integer"))
  point_count_value = if surface_point_count === nothing
    maximum(space.compiled_leaves[leaf_index].quadrature_shape)
  else
    surface_point_count isa Integer ||
      throw(ArgumentError("surface_point_count must be a positive Int-representable integer"))
    surface_point_count
  end
  checked_depth = _checked_nonnegative(subdivision_depth, "subdivision_depth")
  checked_point_count = _checked_positive(point_count_value, "surface_point_count")
  return _implicit_surface_quadrature_checked(domain(space), checked_leaf, classifier,
                                              checked_depth, checked_point_count)
end

function implicit_surface_quadrature(domain::AbstractDomain{D,T}, leaf::Integer, classifier;
                                     subdivision_depth=2,
                                     surface_point_count=2) where {D,T<:AbstractFloat}
  subdivision_depth isa Integer ||
    throw(ArgumentError("subdivision_depth must be a non-negative Int-representable integer"))
  surface_point_count isa Integer ||
    throw(ArgumentError("surface_point_count must be a positive Int-representable integer"))
  checked_leaf = _checked_cell(grid(domain), leaf)
  _is_current_domain_active_leaf(domain, checked_leaf) ||
    throw(ArgumentError("embedded surfaces can only be built on active leaves"))
  checked_depth = _checked_nonnegative(subdivision_depth, "subdivision_depth")
  checked_point_count = _checked_positive(surface_point_count, "surface_point_count")
  return _implicit_surface_quadrature_checked(domain, checked_leaf, classifier, checked_depth,
                                              checked_point_count)
end

function _implicit_surface_quadrature_checked(domain::AbstractDomain{D,T}, checked_leaf::Int,
                                              classifier, checked_depth::Int,
                                              checked_point_count::Int) where {D,T<:AbstractFloat}
  D <= 2 ||
    throw(ArgumentError("default embedded-surface construction currently supports dimensions 1 and 2"))
  points = NTuple{D,T}[]
  weights = T[]
  normals = NTuple{D,T}[]
  lower = ntuple(_ -> -one(T), D)
  upper = ntuple(_ -> one(T), D)
  rule = D == 1 ? nothing : gauss_legendre_rule(T, checked_point_count)
  _append_embedded_surface_subcells!(points, weights, normals, domain, checked_leaf, classifier,
                                     rule, lower, upper, 0, checked_depth)
  isempty(points) && return nothing
  return SurfaceQuadrature(checked_leaf, PointQuadrature(points, weights), normals)
end
# Embedded-surface extraction on recursively refined cut subcells.

# Recursive embedded-surface extraction. Terminal cut subcells are handled by
# dimension-specific `_append_terminal_embedded_surface!` methods; the recursive
# subdivision itself is shared across dimensions.
function _append_embedded_surface_subcells!(points::Vector{NTuple{D,T}}, weights::Vector{T},
                                            normals::Vector{NTuple{D,T}},
                                            domain::AbstractDomain{D,T}, leaf::Int, classifier,
                                            rule, lower::NTuple{D,T}, upper::NTuple{D,T},
                                            depth::Int, max_depth::Int) where {D,T<:AbstractFloat}
  state = _embedded_surface_state(domain, leaf, classifier, lower, upper, T)
  state == :uniform && return nothing
  state == :degenerate &&
    throw(ArgumentError("embedded-surface classifier is degenerate on a sampled subcell"))

  if depth >= max_depth
    _append_terminal_embedded_surface!(points, weights, normals, domain, leaf, classifier, rule,
                                       lower, upper, T)
    return nothing
  end

  midpoint = _subcell_midpoint(lower, upper, T)

  for child_mask in 0:((1<<D)-1)
    child_lower, child_upper = _subcell_child_bounds(lower, midpoint, upper, child_mask)
    _append_embedded_surface_subcells!(points, weights, normals, domain, leaf, classifier, rule,
                                       child_lower, child_upper, depth + 1, max_depth)
  end

  return nothing
end

# Coarse cut/uniform classification for embedded surfaces. Unlike finite cells,
# uniform means that no zero crossing is detected on the sampled corners and
# center, regardless of which side of the interface the subcell lies on.
function _embedded_surface_state(domain::AbstractDomain{D,T}, leaf::Int, classifier,
                                 lower::NTuple{D,T}, upper::NTuple{D,T},
                                 ::Type{T}) where {D,T<:AbstractFloat}
  minimum_value, maximum_value = _subcell_sample_extrema(domain, leaf, classifier, lower, upper,
                                                         _surface_classifier_value, T)
  tolerance = _EMBEDDED_SURFACE_TOLERANCE(T)
  minimum_value > tolerance && return :uniform
  maximum_value < -tolerance && return :uniform
  abs(minimum_value) <= tolerance && abs(maximum_value) <= tolerance && return :degenerate
  return :cut
end

# Terminal 1D embedded-surface contribution: find a unique root on the interval,
# assign unit reference weight, and orient the normal by the sign of the local
# classifier derivative.
function _append_terminal_embedded_surface!(points::Vector{NTuple{1,T}}, weights::Vector{T},
                                            normals::Vector{NTuple{1,T}},
                                            domain::AbstractDomain{1,T}, leaf::Int, classifier,
                                            ::Nothing, lower::NTuple{1,T}, upper::NTuple{1,T},
                                            ::Type{T}) where {T<:AbstractFloat}
  root = _subcell_root(lower[1], upper[1],
                       _surface_classifier_value(classifier,
                                                 map_from_biunit_cube(domain, leaf, lower), T),
                       _surface_classifier_value(classifier,
                                                 map_from_biunit_cube(domain, leaf, upper), T), T)
  root === nothing && return nothing
  _owns_terminal_surface_root(root, lower[1], upper[1], T) || return nothing
  ξ = (root,)
  push!(points, ξ)
  push!(weights, one(T))
  push!(normals, _surface_point_normal(domain, leaf, classifier, ξ, T))
  return nothing
end

# Terminal 2D embedded-surface contribution: extract straight interface segments
# from the cut reference square and place quadrature points on each segment.
function _append_terminal_embedded_surface!(points::Vector{NTuple{2,T}}, weights::Vector{T},
                                            normals::Vector{NTuple{2,T}},
                                            domain::AbstractDomain{2,T}, leaf::Int, classifier,
                                            rule::GaussLegendreRule{T}, lower::NTuple{2,T},
                                            upper::NTuple{2,T}, ::Type{T}) where {T<:AbstractFloat}
  segments = _terminal_embedded_surface_segments(domain, leaf, classifier, lower, upper, T)

  for (first_point, second_point) in segments
    _append_segment_quadrature!(points, weights, normals, domain, leaf, classifier, rule,
                                first_point, second_point, T)
  end

  return nothing
end

# Embedded-surface classifiers must return a genuine signed real value, because
# normals and zero-crossing locations depend on more than just inside/outside.
function _surface_classifier_value(classifier, x, ::Type{T}) where {T<:AbstractFloat}
  value = classifier(x)
  value isa Bool &&
    throw(ArgumentError("embedded-surface classifiers must return finite signed Real values, not Bool"))
  value isa Real || throw(ArgumentError("embedded-surface classifiers must return Real values"))
  checked = T(value)
  isfinite(checked) || throw(ArgumentError("embedded-surface classifier values must be finite"))
  return checked
end

# Locate the linearized zero crossing on a one-dimensional subcell edge.
function _subcell_root(lower::T, upper::T, lower_value::T, upper_value::T,
                       ::Type{T}) where {T<:AbstractFloat}
  tolerance = _EMBEDDED_SURFACE_TOLERANCE(T)
  abs(lower_value) <= tolerance && return lower
  abs(upper_value) <= tolerance && return upper
  lower_value * upper_value < zero(T) || return nothing
  return muladd(upper - lower, lower_value / (lower_value - upper_value), lower)
end

# Ownership rule used to avoid duplicating roots shared by adjacent terminal
# subcells. Right endpoints are assigned only on the global upper boundary.
function _owns_terminal_surface_root(root::T, lower::T, upper::T,
                                     ::Type{T}) where {T<:AbstractFloat}
  tolerance = _EMBEDDED_SURFACE_TOLERANCE(T)
  root < upper - tolerance && return true
  return abs(upper - one(T)) <= tolerance && abs(root - upper) <= tolerance
end

# Extract straight interface segments from one cut reference square. The corner
# signs define a marching-squares case; ambiguous cases are disambiguated by the
# sign at the square center.
function _terminal_embedded_surface_segments(domain::AbstractDomain{2,T}, leaf::Int, classifier,
                                             lower::NTuple{2,T}, upper::NTuple{2,T},
                                             ::Type{T}) where {T<:AbstractFloat}
  corners = (lower, (upper[1], lower[2]), upper, (lower[1], upper[2]))
  values = ntuple(index -> _surface_classifier_value(classifier,
                                                     map_from_biunit_cube(domain, leaf,
                                                                          corners[index]), T), 4)
  edge_points = Vector{NTuple{2,T}}(undef, 4)
  edge_valid = falses(4)

  for edge in 1:4
    point = _square_edge_intersection(corners, values, edge, T)
    point === nothing && continue
    edge_points[edge] = point
    edge_valid[edge] = true
  end

  center = ntuple(axis -> T(0.5) * (lower[axis] + upper[axis]), 2)
  center_inside = _surface_classifier_value(classifier, map_from_biunit_cube(domain, leaf, center),
                                            T) <= zero(T)
  mask = _marching_square_mask(values)
  edge_pairs = _marching_square_segments(mask, center_inside)
  isempty(edge_pairs) && return Tuple{NTuple{2,T},NTuple{2,T}}[]
  segments = Tuple{NTuple{2,T},NTuple{2,T}}[]

  for (first_edge, second_edge) in edge_pairs
    (edge_valid[first_edge] && edge_valid[second_edge]) || continue
    first_point = edge_points[first_edge]
    second_point = edge_points[second_edge]
    _segment_length(first_point, second_point) > _EMBEDDED_SURFACE_TOLERANCE(T) || continue
    push!(segments, (first_point, second_point))
  end

  return segments
end

# Place Gauss-Legendre quadrature points on one straight reference segment and
# assign a consistent reference normal orientation.
function _append_segment_quadrature!(points::Vector{NTuple{2,T}}, weights::Vector{T},
                                     normals::Vector{NTuple{2,T}}, domain::AbstractDomain{2,T},
                                     leaf::Int, classifier, rule::GaussLegendreRule{T},
                                     first_point::NTuple{2,T}, second_point::NTuple{2,T},
                                     ::Type{T}) where {T<:AbstractFloat}
  midpoint, half_vector, half_length = _segment_rule_geometry(first_point, second_point, T)
  reference_normal = _segment_reference_normal(domain, leaf, classifier, midpoint, half_vector, T)
  _append_segment_rule!(points, weights, normals, rule, midpoint, half_vector, half_length,
                        reference_normal)

  return nothing
end

# Compute the zero crossing of the classifier on one edge of a square by linear
# interpolation of the endpoint values.
function _square_edge_intersection(corners, values, edge::Int, ::Type{T}) where {T<:AbstractFloat}
  edge_pairs = ((1, 2), (2, 3), (3, 4), (4, 1))
  first_corner, second_corner = edge_pairs[edge]
  first_value = values[first_corner]
  second_value = values[second_corner]
  tolerance = _EMBEDDED_SURFACE_TOLERANCE(T)

  abs(first_value) <= tolerance && abs(second_value) <= tolerance && return nothing
  abs(first_value) <= tolerance && return corners[first_corner]
  abs(second_value) <= tolerance && return corners[second_corner]
  first_value * second_value < zero(T) || return nothing
  fraction = first_value / (first_value - second_value)
  return ntuple(axis -> muladd(fraction, corners[second_corner][axis] - corners[first_corner][axis],
                               corners[first_corner][axis]), 2)
end

# Convert corner signs to the standard 4-bit marching-squares mask.
function _marching_square_mask(values)
  ((values[1] <= 0 ? 1 : 0) | (values[2] <= 0 ? 2 : 0) | (values[3] <= 0 ? 4 : 0) |
   (values[4] <= 0 ? 8 : 0))
end

# Lookup table for marching-squares segment connectivity. Ambiguous cases 5 and
# 10 are resolved using the sign at the cell center.
function _marching_square_segments(mask::Int, center_inside::Bool)
  mask == 0 && return ()
  mask == 1 && return ((4, 1),)
  mask == 2 && return ((1, 2),)
  mask == 3 && return ((4, 2),)
  mask == 4 && return ((2, 3),)
  mask == 5 && return center_inside ? ((4, 3), (1, 2)) : ((4, 1), (2, 3))
  mask == 6 && return ((1, 3),)
  mask == 7 && return ((4, 3),)
  mask == 8 && return ((3, 4),)
  mask == 9 && return ((1, 3),)
  mask == 10 && return center_inside ? ((4, 1), (2, 3)) : ((1, 2), (3, 4))
  mask == 11 && return ((2, 3),)
  mask == 12 && return ((4, 2),)
  mask == 13 && return ((1, 2),)
  mask == 14 && return ((4, 1),)
  return ()
end

# Choose a reference-space segment normal that is consistent with the gradient
# direction of the classifier in physical space.
function _segment_reference_normal(domain::AbstractDomain{2,T}, leaf::Int, classifier,
                                   midpoint::NTuple{2,T}, half_vector::NTuple{2,T},
                                   ::Type{T}) where {T<:AbstractFloat}
  tangent = (half_vector[1], half_vector[2])
  length_value = hypot(tangent[1], tangent[2])
  normal = (-tangent[2] / length_value, tangent[1] / length_value)
  gradient = _surface_gradient(domain, leaf, classifier, midpoint, T)
  gradient_magnitude = hypot(gradient[1], gradient[2])
  gradient_magnitude <= _EMBEDDED_SURFACE_TOLERANCE(T) && return normal
  physical_normal = _mapped_surface_normal(domain, leaf, normal)
  return physical_normal[1] * gradient[1] + physical_normal[2] * gradient[2] >= zero(T) ? normal :
         (-normal[1], -normal[2])
end

@inline function _segment_rule_geometry(first_point::NTuple{2,T}, second_point::NTuple{2,T},
                                        ::Type{T}) where {T<:AbstractFloat}
  midpoint = _point_midpoint(first_point, second_point, T)
  half_vector = ntuple(axis -> T(0.5) * (second_point[axis] - first_point[axis]), 2)
  half_length = _segment_length(first_point, second_point) * T(0.5)
  return midpoint, half_vector, half_length
end

function _append_segment_rule!(points::Vector{NTuple{2,T}}, weights::Vector{T},
                               normals::Vector{NTuple{2,T}}, rule::GaussLegendreRule{T},
                               midpoint::NTuple{2,T}, half_vector::NTuple{2,T}, half_length::T,
                               normal::NTuple{2,T}) where {T<:AbstractFloat}
  for point_index in 1:point_count(rule)
    η = point(rule, point_index)[1]
    push!(points, ntuple(axis -> muladd(η, half_vector[axis], midpoint[axis]), 2))
    push!(weights, weight(rule, point_index) * half_length)
    push!(normals, normal)
  end

  return nothing
end

function _segment_rule_quadrature(rule::GaussLegendreRule{T}, midpoint::NTuple{2,T},
                                  half_vector::NTuple{2,T}, half_length::T,
                                  normal::NTuple{2,T}) where {T<:AbstractFloat}
  points = NTuple{2,T}[]
  weights = T[]
  normals = NTuple{2,T}[]
  sizehint!(points, point_count(rule))
  sizehint!(weights, point_count(rule))
  sizehint!(normals, point_count(rule))
  _append_segment_rule!(points, weights, normals, rule, midpoint, half_vector, half_length, normal)
  return PointQuadrature(points, weights), normals
end

# In one dimension, the embedded-surface normal is just the sign of the scalar
# classifier gradient.
function _surface_point_normal(domain::AbstractDomain{1,T}, leaf::Int, classifier, ξ::NTuple{1,T},
                               ::Type{T}) where {T<:AbstractFloat}
  gradient = _surface_gradient(domain, leaf, classifier, ξ, T)
  gradient[1] >= zero(T) ? (one(T),) : (-one(T),)
end

# Approximate the physical gradient of the classifier by centered finite
# differences in physical coordinates.
function _surface_gradient(domain::AbstractDomain{D,T}, leaf::Int, classifier, ξ::NTuple{D,T},
                           ::Type{T}) where {D,T<:AbstractFloat}
  x = map_from_biunit_cube(domain, leaf, ξ)
  return ntuple(axis -> begin
                  step = sqrt(eps(T)) * cell_size(domain, leaf, axis)
                  upper = Base.setindex(x, x[axis] + step, axis)
                  lower = Base.setindex(x, x[axis] - step, axis)
                  (_surface_classifier_value(classifier, upper, T) -
                   _surface_classifier_value(classifier, lower, T)) / (T(2) * step)
                end, D)
end

# Map a reference-space normal to the corresponding physical unit normal.
function _mapped_surface_normal(domain::AbstractDomain{D,T}, leaf::Int,
                                normal::NTuple{D,T}) where {D,T<:AbstractFloat}
  transformed = ntuple(axis -> normal[axis] / jacobian_diagonal_from_biunit_cube(domain, leaf, axis),
                       D)
  magnitude = sqrt(sum(transformed[axis]^2 for axis in 1:D))
  return ntuple(axis -> transformed[axis] / magnitude, D)
end

function _segment_length(first_point::NTuple{2,T},
                         second_point::NTuple{2,T}) where {T<:AbstractFloat}
  hypot(second_point[1] - first_point[1], second_point[2] - first_point[2])
end

# Geometry-specific backend for explicit segment meshes in two dimensions.

# Dispatch point for geometry-specific embedded-surface backends.
function surface_quadratures(surface::EmbeddedSurface, domain::AbstractDomain)
  throw(ArgumentError("embedded-surface geometry $(typeof(surface.geometry)) is not supported for dimension $(dimension(domain))"))
end

function surface_quadratures(surface::EmbeddedSurface, space::HpSpace)
  return _space_surface_quadratures(surface, space)
end

function _space_surface_quadratures(surface::EmbeddedSurface, space::HpSpace)
  throw(ArgumentError("embedded-surface geometry $(typeof(surface.geometry)) is not supported for dimension $(dimension(space))"))
end

function surface_quadratures(surface::EmbeddedSurface{<:SegmentMesh},
                             domain::AbstractDomain{2,T}) where {T<:AbstractFloat}
  return _segment_mesh_surface_quadratures(surface.geometry, domain, _domain_active_leaves(domain),
                                           surface.point_count, T)
end

function _space_surface_quadratures(surface::EmbeddedSurface{<:SegmentMesh},
                                    space::HpSpace{2,T}) where {T<:AbstractFloat}
  return _segment_mesh_surface_quadratures(surface.geometry, domain(space), active_leaves(space),
                                           surface.point_count, T)
end

# Intersect each input segment with the active leaves of the domain, clip the
# segment to each leaf box, and convert the resulting physical segment pieces to
# reference-cell surface quadratures.
function _segment_mesh_surface_quadratures(mesh::SegmentMesh, domain::AbstractDomain{2,T},
                                           leaves::AbstractVector{<:Integer}, point_count::Int,
                                           ::Type{T}) where {T<:AbstractFloat}
  quadratures = SurfaceQuadrature[]

  for segment_index in eachindex(mesh.segments)
    segment = mesh.segments[segment_index]
    first_point = ntuple(axis -> T(mesh.points[segment[1]][axis]), 2)
    second_point = ntuple(axis -> T(mesh.points[segment[2]][axis]), 2)

    for leaf in leaves
      quadrature = _segment_leaf_surface_quadrature(domain, leaf, first_point, second_point,
                                                    point_count, T)
      quadrature === nothing && continue
      push!(quadratures, quadrature)
    end
  end

  return quadratures
end

function _segment_leaf_surface_quadrature(domain::AbstractDomain{2,T}, leaf::Int,
                                          first_point::NTuple{2,T}, second_point::NTuple{2,T},
                                          quadrature_point_count::Int,
                                          ::Type{T}) where {T<:AbstractFloat}
  _segment_bbox_overlaps(domain, leaf, first_point, second_point, T) || return nothing
  clipped = _clip_segment_to_leaf(domain, leaf, first_point, second_point, T)
  clipped === nothing && return nothing
  midpoint = _point_midpoint(clipped[1], clipped[2], T)
  _owns_surface_piece(domain, leaf, midpoint, T) || return nothing
  return _segment_surface_quadrature(domain, leaf, clipped[1], clipped[2], quadrature_point_count,
                                     T)
end

@inline function _leaf_bounds(domain::AbstractDomain{2,T}, leaf::Int) where {T<:AbstractFloat}
  return cell_lower(domain, leaf), cell_upper(domain, leaf)
end

# Cheap bounding-box prefilter before attempting exact segment clipping.
function _segment_bbox_overlaps(domain::AbstractDomain{2,T}, leaf::Int, first_point::NTuple{2,T},
                                second_point::NTuple{2,T}, ::Type{T}) where {T<:AbstractFloat}
  tolerance = _EMBEDDED_SURFACE_TOLERANCE(T)
  segment_lower = (min(first_point[1], second_point[1]), min(first_point[2], second_point[2]))
  segment_upper = (max(first_point[1], second_point[1]), max(first_point[2], second_point[2]))
  leaf_lower, leaf_upper = _leaf_bounds(domain, leaf)
  return segment_lower[1] <= leaf_upper[1] + tolerance &&
         segment_upper[1] >= leaf_lower[1] - tolerance &&
         segment_lower[2] <= leaf_upper[2] + tolerance &&
         segment_upper[2] >= leaf_lower[2] - tolerance
end

function _clip_segment_to_leaf(domain::AbstractDomain{2,T}, leaf::Int, first_point::NTuple{2,T},
                               second_point::NTuple{2,T}, ::Type{T}) where {T<:AbstractFloat}
  lower, upper = _leaf_bounds(domain, leaf)
  return _clip_segment_to_box(first_point, second_point, lower, upper, T)
end

# Liang-Barsky style clipping of a line segment to an axis-aligned box.
function _clip_segment_to_box(first_point::NTuple{2,T}, second_point::NTuple{2,T},
                              lower::NTuple{2,T}, upper::NTuple{2,T},
                              ::Type{T}) where {T<:AbstractFloat}
  tolerance = _EMBEDDED_SURFACE_TOLERANCE(T)
  t_lower = zero(T)
  t_upper = one(T)
  delta = (second_point[1] - first_point[1], second_point[2] - first_point[2])

  for axis in 1:2
    direction = delta[axis]

    if abs(direction) <= tolerance
      first_point[axis] < lower[axis] - tolerance && return nothing
      first_point[axis] > upper[axis] + tolerance && return nothing
      continue
    end

    first_parameter = (lower[axis] - first_point[axis]) / direction
    second_parameter = (upper[axis] - first_point[axis]) / direction
    t_lower = max(t_lower, min(first_parameter, second_parameter))
    t_upper = min(t_upper, max(first_parameter, second_parameter))
    t_lower <= t_upper + tolerance || return nothing
  end

  clipped_first = ntuple(axis -> muladd(t_lower, delta[axis], first_point[axis]), 2)
  clipped_second = ntuple(axis -> muladd(t_upper, delta[axis], first_point[axis]), 2)
  _segment_length(clipped_first, clipped_second) > tolerance || return nothing
  return clipped_first, clipped_second
end

# Ownership rule used to assign clipped segment pieces on shared leaf boundaries
# to exactly one leaf and thereby avoid duplicated surface quadratures.
function _owns_surface_piece(domain::AbstractDomain{2,T}, leaf::Int, point::NTuple{2,T},
                             ::Type{T}) where {T<:AbstractFloat}
  tolerance = _EMBEDDED_SURFACE_TOLERANCE(T)
  leaf_lower, leaf_upper = _leaf_bounds(domain, leaf)

  for axis in 1:2
    point[axis] < leaf_lower[axis] - tolerance && return false
    point[axis] > leaf_upper[axis] + tolerance && return false

    if abs(point[axis] - leaf_upper[axis]) <= tolerance
      domain_upper = origin(domain, axis) + extent(domain, axis)
      abs(leaf_upper[axis] - domain_upper) <= tolerance || return false
    end
  end

  return true
end

# Convert one physical segment piece inside a leaf to reference coordinates and
# place a one-dimensional Gauss-Legendre rule on that reference segment.
function _segment_surface_quadrature(domain::AbstractDomain{2,T}, leaf::Int,
                                     first_point::NTuple{2,T}, second_point::NTuple{2,T},
                                     quadrature_point_count::Int,
                                     ::Type{T}) where {T<:AbstractFloat}
  first_reference = map_to_biunit_cube(domain, leaf, first_point)
  second_reference = map_to_biunit_cube(domain, leaf, second_point)
  rule = gauss_legendre_rule(T, quadrature_point_count)
  midpoint, half_vector, half_length = _segment_rule_geometry(first_reference, second_reference, T)
  reference_normal = _reference_segment_normal(first_reference, second_reference, T)
  quadrature, normals = _segment_rule_quadrature(rule, midpoint, half_vector, half_length,
                                                 reference_normal)
  return SurfaceQuadrature(leaf, quadrature, normals)
end

# Reference-space unit normal of a directed segment.
function _reference_segment_normal(first_point::NTuple{2,T}, second_point::NTuple{2,T},
                                   ::Type{T}) where {T<:AbstractFloat}
  tangent = (second_point[1] - first_point[1], second_point[2] - first_point[2])
  length_value = hypot(tangent[1], tangent[2])
  return (tangent[2] / length_value, -tangent[1] / length_value)
end

# Validation and normalization of user-supplied geometry/quadrature data.

# Validate and convert one segment-mesh point to the internal floating-point
# storage type.
function _segment_point_scalar_type(point)
  length(point) == 2 || throw(ArgumentError("segment-mesh points must be two-dimensional"))
  point[1] isa Real && point[2] isa Real ||
    throw(ArgumentError("segment-mesh point coordinates must be Real values"))
  return promote_type(typeof(point[1]), typeof(point[2]))
end

function _checked_segment_point(point, ::Type{T}) where {T<:AbstractFloat}
  length(point) == 2 || throw(ArgumentError("segment-mesh points must be two-dimensional"))
  values = ntuple(axis -> T(point[axis]), 2)
  all(isfinite, values) || throw(ArgumentError("segment-mesh points must be finite"))
  return values
end

# Validate one segment as a pair of distinct point indices.
function _checked_segment(segment, point_count::Int)
  length(segment) == 2 ||
    throw(ArgumentError("segment-mesh segments must connect two point indices"))
  segment[1] isa Integer && segment[2] isa Integer ||
    throw(ArgumentError("segment-mesh segment indices must be integers"))
  first_index = _require_index(segment[1], point_count, "segment point")
  second_index = _require_index(segment[2], point_count, "segment point")
  first_index != second_index ||
    throw(ArgumentError("segment-mesh segments must have distinct endpoints"))
  return (first_index, second_index)
end

# Validate the full segment-mesh topology and coordinates.
function _validate_segment_mesh(points::Vector{NTuple{2,T}}, segments::Vector{NTuple{2,Int}},
                                ::Type{T}) where {T<:AbstractFloat}
  length(points) >= 2 || throw(ArgumentError("segment meshes require at least two points"))
  !isempty(segments) || throw(ArgumentError("segment meshes require at least one segment"))

  for point in points
    _checked_segment_point(point, T)
  end

  for segment in segments
    _checked_segment(segment, length(points))
  end

  return nothing
end

# Validate and normalize the reference normals attached to a surface quadrature.
function _checked_surface_normals(quadrature::PointQuadrature{D,T}, normals,
                                  ::Type{T}) where {D,T<:AbstractFloat}
  length(normals) == point_count(quadrature) ||
    throw(ArgumentError("surface-quadrature normals must match the quadrature point count"))
  checked = Vector{NTuple{D,T}}(undef, length(normals))

  for index in eachindex(normals)
    checked[index] = _checked_surface_normal(normals[index], D, T)
  end

  return checked
end

# Convert one user-supplied normal to a finite unit vector of the correct
# dimension.
function _checked_surface_normal(normal, dimension_count::Int, ::Type{T}) where {T<:AbstractFloat}
  length(normal) == dimension_count ||
    throw(ArgumentError("surface-quadrature normals must match the spatial dimension"))
  for axis in 1:dimension_count
    normal[axis] isa Real ||
      throw(ArgumentError("surface-quadrature normals must contain Real values"))
  end
  values = ntuple(axis -> T(normal[axis]), dimension_count)
  magnitude = sqrt(sum(values[axis]^2 for axis in 1:dimension_count))
  isfinite(magnitude) || throw(ArgumentError("surface-quadrature normals must be finite"))
  magnitude > zero(T) || throw(ArgumentError("surface-quadrature normals must be non-zero"))
  return ntuple(axis -> values[axis] / magnitude, dimension_count)
end

# This file handles finite-cell and embedded-surface quadrature data. Starting
# from either implicit classifiers or explicit segment-mesh geometry, it
# constructs per-leaf reference quadratures that later compilation, assembly,
# and verification code can treat just like any other local integration item.
#
# There are two closely related stories in this file.
#
# First, finite-cell quadrature replaces the default volume rule on a cut cell by
# a rule that integrates only the physically active part of the cell. The
# implementation recursively subdivides the reference cell, collects candidate
# points on subcells judged to be inside, and then compresses that candidate set
# by moment fitting.
#
# Second, embedded-surface quadrature describes codimension-1 geometry inside a
# leaf. Here the output is not a cell rule but a reference-surface rule with one
# reference normal per quadrature point. These data can come either from an
# implicit zero set or from an explicit segment mesh.
#
# In both cases the crucial design choice is the same: geometry is reduced to
# reference-cell quadratures attached to active leaves. That keeps the later
# integration layer agnostic to how the cut region or embedded surface was
# originally described.
#
# The file is organized accordingly: public attachment types and constructors
# first, then the finite-cell machinery, then the embedded-surface extraction
# algorithms, then geometry-specific backends such as `SegmentMesh`, and
# finally the low-level validation helpers.

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
const _FINITE_CELL_MOMENT_TOLERANCE_SCALE = 1.0e3
const _FINITE_CELL_WEIGHT_PRUNE_RATIO = 1.0e-8
const _FINITE_CELL_MAX_ELIMINATION_ITERATIONS = 1024

function SurfaceQuadrature(leaf::Integer, quadrature::PointQuadrature{D,T},
                           normals) where {D,T<:AbstractFloat}
  return SurfaceQuadrature{D,T,typeof(quadrature)}(Int(leaf), quadrature, normals)
end

function SurfaceQuadrature(leaf::Integer, quadrature::PointQuadrature{D,T},
                           normal::NTuple{D,<:Real}) where {D,T<:AbstractFloat}
  return SurfaceQuadrature(leaf, quadrature, fill(normal, point_count(quadrature)))
end

function SegmentMesh(points, segments)
  length(points) >= 2 || throw(ArgumentError("segment meshes require at least two points"))
  !isempty(segments) || throw(ArgumentError("segment meshes require at least one segment"))
  T = float(mapreduce(point -> promote_type(typeof(point[1]), typeof(point[2])), promote_type,
                      points))
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

function EmbeddedSurface(geometry::G; point_count::Integer=2) where {G}
  return EmbeddedSurface{G}(geometry, Int(point_count))
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
  push!(problem.embedded_surfaces, _SurfaceAttachment(nothing, embedded_surface))
  return problem
end

function add_embedded_surface!(problem::_AbstractProblem, tag::Symbol,
                               embedded_surface::EmbeddedSurface)
  push!(problem.embedded_surfaces, _SurfaceAttachment(tag, embedded_surface))
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
  push!(problem.embedded_surfaces, _SurfaceAttachment(nothing, surface))
  return problem
end

function add_surface_quadrature!(problem::_AbstractProblem, tag::Symbol,
                                 surface::SurfaceQuadrature)
  push!(problem.embedded_surfaces, _SurfaceAttachment(tag, surface))
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
  push!(problem.cell_quadratures, _CellQuadratureAttachment(Int(leaf), quadrature))
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
"""
function implicit_surface_quadrature(space::HpSpace{D,T}, leaf::Integer, classifier;
                                     subdivision_depth::Integer=2,
                                     surface_point_count::Integer=maximum(cell_quadrature_shape(space,
                                                                                                Int(leaf)))) where {D,
                                                                                                                    T<:AbstractFloat}
  return implicit_surface_quadrature(domain(space), leaf, classifier; subdivision_depth,
                                     surface_point_count)
end

function implicit_surface_quadrature(domain::Domain{D,T}, leaf::Integer, classifier;
                                     subdivision_depth::Integer=2,
                                     surface_point_count::Integer=2) where {D,T<:AbstractFloat}
  checked_leaf = _checked_cell(grid(domain), leaf)
  is_active_leaf(grid(domain), checked_leaf) ||
    throw(ArgumentError("embedded surfaces can only be built on active leaves"))
  checked_depth = _checked_nonnegative(subdivision_depth, "subdivision_depth")
  checked_point_count = _checked_positive(surface_point_count, "surface_point_count")
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

"""
    finite_cell_quadrature(space, leaf, classifier; subdivision_depth=2)
    finite_cell_quadrature(domain, leaf, quadrature_shape, classifier; subdivision_depth=2)

Construct a reference-cell quadrature for the part of one active leaf that lies
inside an implicitly defined region.

The classifier is evaluated in physical coordinates and uses the sign convention

  classifier(x) ≤ 0  inside,
  classifier(x) > 0  outside.

If the leaf lies entirely outside, the function returns `nothing`. If it lies
entirely inside, the function returns the standard tensor-product quadrature
corresponding to `quadrature_shape`. Otherwise, the leaf is recursively
subdivided, candidate points are collected on cut subcells, and a reduced
positive-weight quadrature is obtained by moment fitting against a polynomial
test space. The returned quadrature lives on the biunit reference cell of the
leaf and is intended to be used with [`add_cell_quadrature!`](@ref).

The moment-fitting stage targets the full tensor-product polynomial space
associated with `quadrature_shape`, so the reduced rule preserves the moments
that the original tensor rule would integrate exactly on an uncut cell.
"""
function finite_cell_quadrature(space::HpSpace{D,T}, leaf::Integer, classifier;
                                subdivision_depth::Integer=2) where {D,T<:AbstractFloat}
  return finite_cell_quadrature(domain(space), leaf, cell_quadrature_shape(space, leaf), classifier;
                                subdivision_depth)
end

function finite_cell_quadrature(domain::Domain{D,T}, leaf::Integer,
                                quadrature_shape::NTuple{D,<:Integer}, classifier;
                                subdivision_depth::Integer=2) where {D,T<:AbstractFloat}
  checked_leaf = _checked_cell(grid(domain), leaf)
  is_active_leaf(grid(domain), checked_leaf) ||
    throw(ArgumentError("finite-cell quadrature can only be built on active leaves"))
  checked_depth = _checked_nonnegative(subdivision_depth, "subdivision_depth")
  checked_shape = ntuple(axis -> _checked_positive(quadrature_shape[axis],
                                                   "quadrature_shape[$axis]"), D)
  lower = ntuple(_ -> -one(T), D)
  upper = ntuple(_ -> one(T), D)
  classification = _classify_subcell(domain, checked_leaf, classifier, lower, upper, T)
  classification == :outside && return nothing
  base_quadrature = TensorQuadrature(T, checked_shape)

  if classification == :inside
    return _point_quadrature(base_quadrature)
  end

  points = NTuple{D,T}[]
  weights = T[]
  _collect_finite_cell_candidates!(points, weights, domain, checked_leaf, classifier,
                                   base_quadrature, lower, upper, checked_depth)
  isempty(points) && return nothing
  return _moment_fit_finite_cell_quadrature(points, weights, checked_shape)
end

# Finite-cell quadrature construction by subdivision and moment fitting.

# Convert a tensor-product quadrature rule to an explicit point list. This is
# useful because the later finite-cell reduction code operates on mutable point
# and weight arrays rather than on an implicit tensor-product structure.
function _point_quadrature(quadrature::TensorQuadrature{D,T}) where {D,T<:AbstractFloat}
  points = Vector{NTuple{D,T}}(undef, point_count(quadrature))
  weights = Vector{T}(undef, point_count(quadrature))

  for point_index in 1:point_count(quadrature)
    points[point_index] = point(quadrature, point_index)
    weights[point_index] = weight(quadrature, point_index)
  end

  return PointQuadrature(points, weights)
end

@inline function _subcell_midpoint(lower::NTuple{D,T}, upper::NTuple{D,T},
                                   ::Type{T}) where {D,T<:AbstractFloat}
  return ntuple(axis -> T(0.5) * (lower[axis] + upper[axis]), D)
end

@inline function _subcell_corner(lower::NTuple{D,T}, upper::NTuple{D,T},
                                 corner_mask::Int) where {D,T<:AbstractFloat}
  return ntuple(axis -> ((corner_mask >> (axis - 1)) & 1) == 0 ? lower[axis] : upper[axis], D)
end

@inline function _subcell_child_bounds(lower::NTuple{D,T}, midpoint::NTuple{D,T},
                                       upper::NTuple{D,T},
                                       child_mask::Int) where {D,T<:AbstractFloat}
  child_lower = ntuple(axis -> ((child_mask >> (axis - 1)) & 1) == 0 ? lower[axis] : midpoint[axis],
                       D)
  child_upper = ntuple(axis -> ((child_mask >> (axis - 1)) & 1) == 0 ? midpoint[axis] : upper[axis],
                       D)
  return child_lower, child_upper
end

@inline function _point_midpoint(first_point::NTuple{D,T}, second_point::NTuple{D,T},
                                 ::Type{T}) where {D,T<:AbstractFloat}
  return ntuple(axis -> T(0.5) * (first_point[axis] + second_point[axis]), D)
end

# Collect candidate quadrature points for a cut finite cell by recursively
# subdividing the reference cell. Parallel work starts at the first refinement
# level to keep the recursive inner routine simple while still exploiting
# multiple threads on larger searches.
function _collect_finite_cell_candidates!(points::Vector{NTuple{D,T}}, weights::Vector{T},
                                          domain::Domain{D,T}, leaf::Int, classifier,
                                          base_quadrature::TensorQuadrature{D,T},
                                          lower::NTuple{D,T}, upper::NTuple{D,T},
                                          max_depth::Int) where {D,T<:AbstractFloat}
  if max_depth == 0 || Threads.nthreads() == 1
    _collect_finite_cell_candidates_recursive!(points, weights, domain, leaf, classifier,
                                               base_quadrature, lower, upper, 0, max_depth)
    return nothing
  end

  midpoint = _subcell_midpoint(lower, upper, T)
  child_total = 1 << D
  child_points = [NTuple{D,T}[] for _ in 1:child_total]
  child_weights = [T[] for _ in 1:child_total]

  _run_chunks!(child_total) do first_child, last_child
    for child_index in first_child:last_child
      child_mask = child_index - 1
      child_lower, child_upper = _subcell_child_bounds(lower, midpoint, upper, child_mask)
      _collect_finite_cell_candidates_recursive!(child_points[child_index],
                                                 child_weights[child_index], domain, leaf,
                                                 classifier, base_quadrature, child_lower,
                                                 child_upper, 1, max_depth)
    end
  end

  for child_index in 1:child_total
    append!(points, child_points[child_index])
    append!(weights, child_weights[child_index])
  end

  return nothing
end

# Recursive finite-cell candidate generation. Fully inside subcells contribute a
# scaled copy of the base quadrature, fully outside subcells contribute nothing,
# and terminal cut subcells contribute only those quadrature points whose
# classifier values lie on the inside.
function _collect_finite_cell_candidates_recursive!(points::Vector{NTuple{D,T}}, weights::Vector{T},
                                                    domain::Domain{D,T}, leaf::Int, classifier,
                                                    base_quadrature::TensorQuadrature{D,T},
                                                    lower::NTuple{D,T}, upper::NTuple{D,T},
                                                    depth::Int,
                                                    max_depth::Int) where {D,T<:AbstractFloat}
  classification = _classify_subcell(domain, leaf, classifier, lower, upper, T)

  if classification == :inside
    _append_subcell_quadrature!(points, weights, domain, leaf, classifier, base_quadrature, lower,
                                upper, false)
    return nothing
  end

  classification == :outside && return nothing

  if depth >= max_depth
    _append_subcell_quadrature!(points, weights, domain, leaf, classifier, base_quadrature, lower,
                                upper, true)
    return nothing
  end

  midpoint = _subcell_midpoint(lower, upper, T)

  for child_mask in 0:((1<<D)-1)
    child_lower, child_upper = _subcell_child_bounds(lower, midpoint, upper, child_mask)
    _collect_finite_cell_candidates_recursive!(points, weights, domain, leaf, classifier,
                                               base_quadrature, child_lower, child_upper, depth + 1,
                                               max_depth)
  end

  return nothing
end

@inline _finite_cell_moment_tolerance(::Type{T}) where {T<:AbstractFloat} = T(_FINITE_CELL_MOMENT_TOLERANCE_SCALE) *
                                                                            eps(T)

# Reduce an oversized candidate set to a compact positive-weight quadrature by
# matching polynomial moments. The target moment space is the full tensor basis
# associated with the requested quadrature shape.
function _moment_fit_finite_cell_quadrature(points::Vector{NTuple{D,T}}, weights::Vector{T},
                                            quadrature_shape::NTuple{D,Int}) where {D,
                                                                                    T<:AbstractFloat}
  length(points) == length(weights) || throw(ArgumentError("point and weight data must match"))
  isempty(points) && throw(ArgumentError("candidate quadrature must not be empty"))
  degrees = quadrature_shape
  min_point_count = prod(quadrature_shape; init=1)
  length(points) <= min_point_count && return PointQuadrature(points, weights)
  modes = collect(basis_modes(FullTensorBasis(), degrees))
  basis_count = length(modes)
  basis_count <= 1 && return PointQuadrature(points, weights)
  matrix = _finite_cell_moment_matrix(points, degrees, modes)
  moments = matrix * weights
  tolerance = _finite_cell_moment_tolerance(T)
  reduced = _point_elimination(matrix, moments, points, min_point_count, tolerance)
  reduced === nothing && return PointQuadrature(points, weights)
  return reduced
end

# Assemble the moment matrix whose columns are candidate points and whose rows
# are tensor-product polynomial test functions.
function _finite_cell_moment_matrix(points::Vector{NTuple{D,T}}, degrees::NTuple{D,Int},
                                    modes::Vector{NTuple{D,Int}}) where {D,T<:AbstractFloat}
  matrix = Matrix{T}(undef, length(modes), length(points))
  worker_count = max(1, min(Threads.nthreads(), length(points)))
  axis_buffers = [ntuple(axis -> Vector{T}(undef, degrees[axis] + 1), D) for _ in 1:worker_count]

  _run_chunks_with_scratch!(axis_buffers, length(points)) do buffers, first_point, last_point
    for point_index in first_point:last_point
      point_data = points[point_index]

      for axis in 1:D
        legendre_values_and_derivatives!(point_data[axis], degrees[axis], buffers[axis], nothing)
      end

      for mode_index in eachindex(modes)
        mode = modes[mode_index]
        value = one(T)

        @inbounds for axis in 1:D
          value *= buffers[axis][mode[axis]+1]
        end

        matrix[mode_index, point_index] = value
      end
    end
  end

  return matrix
end

# Starting from a feasible positive quadrature, greedily try to remove small-
# weight points while preserving the polynomial moments to the requested
# tolerance.
function _point_elimination(matrix::Matrix{T}, moments::Vector{T}, points::Vector{NTuple{D,T}},
                            min_point_count::Int, tolerance::T) where {D,T<:AbstractFloat}
  full_weights, full_residual = _solve_nonnegative_least_squares(matrix, moments, tolerance)
  full_residual <= tolerance || return nothing
  current = findall(weight -> weight > tolerance, full_weights)
  isempty(current) && return nothing
  current_weights = full_weights[current]

  if length(current) > size(matrix, 1)
    order = sortperm(current_weights; rev=true)
    current = current[order[1:size(matrix, 1)]]
    current_weights, residual = _solve_nonnegative_least_squares(view(matrix, :, current), moments,
                                                                 tolerance)
    residual <= tolerance || return nothing
  end

  previous = copy(current)
  previous_weights = copy(current_weights)

  for _ in 1:_FINITE_CELL_MAX_ELIMINATION_ITERATIONS
    length(current) <= min_point_count && break
    removable = _removable_weight_indices(current_weights, min_point_count, tolerance)
    isempty(removable) && break
    accepted = false

    for local_index in removable
      trial = [current[index] for index in eachindex(current) if index != local_index]
      length(trial) < min_point_count && continue
      trial_weights, residual = _solve_nonnegative_least_squares(view(matrix, :, trial), moments,
                                                                 tolerance)
      residual <= tolerance || continue
      current = trial
      current_weights = trial_weights
      previous = copy(current)
      previous_weights = copy(current_weights)
      accepted = true
      break
    end

    accepted || break
  end

  return PointQuadrature([points[index] for index in previous], previous_weights)
end

# Prefer removing very small weights first; if none are tiny, fall back to the
# smallest weight to keep the elimination process moving.
function _removable_weight_indices(weights::Vector{T}, min_point_count::Int,
                                   tolerance::T) where {T<:AbstractFloat}
  length(weights) > min_point_count || return Int[]
  maximum_weight = maximum(weights)
  threshold = max(tolerance, T(_FINITE_CELL_WEIGHT_PRUNE_RATIO) * maximum_weight)
  removable = findall(weight -> weight <= threshold, weights)
  isempty(removable) && return [argmin(weights)]
  return sort(removable; by=index -> weights[index])
end

# Lawson-Hanson style active-set solver for the nonnegative least-squares
# problem
#
#   minimize ‖A w - m‖₂  subject to  w ≥ 0.
#
# This is the algebraic core of the finite-cell moment fitting step.
function _solve_nonnegative_least_squares(matrix::AbstractMatrix{T}, moments::AbstractVector{T},
                                          tolerance::T) where {T<:AbstractFloat}
  column_count = size(matrix, 2)
  solution = zeros(T, column_count)
  passive = falses(column_count)
  gradient = matrix' * moments
  maximum_iterations = max(8 * column_count, 32)
  iterations = 0

  while iterations < maximum_iterations
    candidate = _nnls_candidate(gradient, passive)
    candidate == 0 && break
    gradient[candidate] > tolerance || break
    passive[candidate] = true

    while true
      passive_indices = findall(passive)
      passive_solution = _least_squares_min_norm(view(matrix, :, passive_indices), moments,
                                                 tolerance)
      trial = zeros(T, column_count)
      trial[passive_indices] = passive_solution

      if all(value > tolerance for value in passive_solution)
        solution = trial
        break
      end

      alpha = one(T)

      for index in passive_indices
        trial[index] > tolerance && continue
        denominator = solution[index] - trial[index]
        abs(denominator) <= tolerance && continue
        alpha = min(alpha, solution[index] / denominator)
      end

      solution .= solution .+ alpha .* (trial .- solution)

      for index in passive_indices
        if solution[index] <= tolerance
          solution[index] = zero(T)
          passive[index] = false
        end
      end

      !any(passive) && break
    end

    residual = moments - matrix * solution
    gradient = matrix' * residual
    iterations += 1
  end

  residual = moments - matrix * solution
  scale = max(norm(moments), one(T))
  return solution, norm(residual) / scale
end

# Select the inactive variable with the largest positive dual gradient.
function _nnls_candidate(gradient::Vector{T}, passive::BitVector) where {T<:AbstractFloat}
  candidate = 0
  best = zero(T)

  for index in eachindex(gradient)
    passive[index] && continue
    gradient[index] > best || continue
    best = gradient[index]
    candidate = index
  end

  return candidate
end

# Minimum-norm least-squares solve based on a truncated singular-value
# decomposition. Small singular values are dropped according to the requested
# tolerance to avoid unstable moment fits.
function _least_squares_min_norm(matrix::AbstractMatrix{T}, rhs::AbstractVector{T},
                                 tolerance::T) where {T<:AbstractFloat}
  isempty(matrix) && return T[]
  factorization = svd(Matrix(matrix); full=false)
  isempty(factorization.S) && return zeros(T, size(matrix, 2))
  cutoff = tolerance * first(factorization.S)
  reduced_rhs = factorization.U' * rhs
  scaled = similar(reduced_rhs)

  for index in eachindex(reduced_rhs)
    singular = factorization.S[index]
    scaled[index] = singular > cutoff ? reduced_rhs[index] / singular : zero(T)
  end

  return factorization.V * scaled
end

# Sample a reference subcell at all corners and at the center, returning the
# minimum and maximum classifier values observed in physical coordinates.
function _subcell_sample_extrema(domain::Domain{D,T}, leaf::Int, classifier, lower::NTuple{D,T},
                                 upper::NTuple{D,T}, classifier_value,
                                 ::Type{T}) where {D,T<:AbstractFloat}
  minimum_value = typemax(T)
  maximum_value = typemin(T)

  for corner_mask in 0:((1<<D)-1)
    ξ = _subcell_corner(lower, upper, corner_mask)
    value = classifier_value(classifier, map_from_biunit_cube(domain, leaf, ξ), T)
    minimum_value = min(minimum_value, value)
    maximum_value = max(maximum_value, value)
  end

  center = _subcell_midpoint(lower, upper, T)
  center_value = classifier_value(classifier, map_from_biunit_cube(domain, leaf, center), T)
  minimum_value = min(minimum_value, center_value)
  maximum_value = max(maximum_value, center_value)
  return minimum_value, maximum_value
end

# Classify a reference subcell as fully inside, fully outside, or cut by
# sampling all corners and the subcell center in physical coordinates.
function _classify_subcell(domain::Domain{D,T}, leaf::Int, classifier, lower::NTuple{D,T},
                           upper::NTuple{D,T}, ::Type{T}) where {D,T<:AbstractFloat}
  minimum_value, maximum_value = _subcell_sample_extrema(domain, leaf, classifier, lower, upper,
                                                         _classifier_value, T)
  maximum_value <= zero(T) && return :inside
  minimum_value > zero(T) && return :outside
  return :cut
end

# Append a scaled copy of the base quadrature on one reference subcell. When
# `filter_points` is true, only the points that lie on the inside of the
# implicit region are kept.
function _append_subcell_quadrature!(points::Vector{NTuple{D,T}}, weights::Vector{T},
                                     domain::Domain{D,T}, leaf::Int, classifier,
                                     base_quadrature::TensorQuadrature{D,T}, lower::NTuple{D,T},
                                     upper::NTuple{D,T},
                                     filter_points::Bool) where {D,T<:AbstractFloat}
  center = _subcell_midpoint(lower, upper, T)
  half_size = ntuple(axis -> T(0.5) * (upper[axis] - lower[axis]), D)
  scale = prod(half_size)

  for point_index in 1:point_count(base_quadrature)
    η = point(base_quadrature, point_index)
    ξ = ntuple(axis -> muladd(half_size[axis], η[axis], center[axis]), D)

    if filter_points
      _classifier_value(classifier, map_from_biunit_cube(domain, leaf, ξ), T) <= zero(T) || continue
    end

    push!(points, ξ)
    push!(weights, weight(base_quadrature, point_index) * scale)
  end

  return nothing
end

# Finite-cell classifiers accept either a signed real value or a Boolean. Booleans
# are interpreted as inside/outside indicators and converted to the signed
# convention used by the rest of the implementation.
function _classifier_value(classifier, x, ::Type{T}) where {T<:AbstractFloat}
  value = classifier(x)

  if value isa Bool
    return value ? -one(T) : one(T)
  end

  value isa Real || throw(ArgumentError("finite-cell classifiers must return Bool or Real values"))
  checked = T(value)
  isfinite(checked) || throw(ArgumentError("finite-cell classifier values must be finite"))
  return checked
end

# Embedded-surface extraction on recursively refined cut subcells.

# Recursive embedded-surface extraction. Terminal cut subcells are handled by
# dimension-specific `_append_terminal_embedded_surface!` methods; the recursive
# subdivision itself is shared across dimensions.
function _append_embedded_surface_subcells!(points::Vector{NTuple{D,T}}, weights::Vector{T},
                                            normals::Vector{NTuple{D,T}}, domain::Domain{D,T},
                                            leaf::Int, classifier, rule, lower::NTuple{D,T},
                                            upper::NTuple{D,T}, depth::Int,
                                            max_depth::Int) where {D,T<:AbstractFloat}
  _embedded_surface_state(domain, leaf, classifier, lower, upper, T) == :uniform && return nothing

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
function _embedded_surface_state(domain::Domain{D,T}, leaf::Int, classifier, lower::NTuple{D,T},
                                 upper::NTuple{D,T}, ::Type{T}) where {D,T<:AbstractFloat}
  minimum_value, maximum_value = _subcell_sample_extrema(domain, leaf, classifier, lower, upper,
                                                         _surface_classifier_value, T)
  tolerance = _EMBEDDED_SURFACE_TOLERANCE(T)
  minimum_value > tolerance && return :uniform
  maximum_value < -tolerance && return :uniform
  return :cut
end

# Terminal 1D embedded-surface contribution: find a unique root on the interval,
# assign unit reference weight, and orient the normal by the sign of the local
# classifier derivative.
function _append_terminal_embedded_surface!(points::Vector{NTuple{1,T}}, weights::Vector{T},
                                            normals::Vector{NTuple{1,T}}, domain::Domain{1,T},
                                            leaf::Int, classifier, ::Nothing, lower::NTuple{1,T},
                                            upper::NTuple{1,T}, ::Type{T}) where {T<:AbstractFloat}
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
                                            normals::Vector{NTuple{2,T}}, domain::Domain{2,T},
                                            leaf::Int, classifier, rule::GaussLegendreRule{T},
                                            lower::NTuple{2,T}, upper::NTuple{2,T},
                                            ::Type{T}) where {T<:AbstractFloat}
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
function _terminal_embedded_surface_segments(domain::Domain{2,T}, leaf::Int, classifier,
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
                                     normals::Vector{NTuple{2,T}}, domain::Domain{2,T}, leaf::Int,
                                     classifier, rule::GaussLegendreRule{T},
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
function _segment_reference_normal(domain::Domain{2,T}, leaf::Int, classifier,
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
function _surface_point_normal(domain::Domain{1,T}, leaf::Int, classifier, ξ::NTuple{1,T},
                               ::Type{T}) where {T<:AbstractFloat}
  gradient = _surface_gradient(domain, leaf, classifier, ξ, T)
  gradient[1] >= zero(T) ? (one(T),) : (-one(T),)
end

# Approximate the physical gradient of the classifier by centered finite
# differences in physical coordinates.
function _surface_gradient(domain::Domain{D,T}, leaf::Int, classifier, ξ::NTuple{D,T},
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
function _mapped_surface_normal(domain::Domain{D,T}, leaf::Int,
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
function surface_quadratures(surface::EmbeddedSurface, domain::Domain)
  throw(ArgumentError("embedded-surface geometry $(typeof(surface.geometry)) is not supported for dimension $(dimension(domain))"))
end

function surface_quadratures(surface::EmbeddedSurface{<:SegmentMesh},
                             domain::Domain{2,T}) where {T<:AbstractFloat}
  return _segment_mesh_surface_quadratures(surface.geometry, domain, surface.point_count, T)
end

# Intersect each input segment with the active leaves of the domain, clip the
# segment to each leaf box, and convert the resulting physical segment pieces to
# reference-cell surface quadratures.
function _segment_mesh_surface_quadratures(mesh::SegmentMesh, domain::Domain{2,T}, point_count::Int,
                                           ::Type{T}) where {T<:AbstractFloat}
  leaves = active_leaves(grid(domain))
  worker_count = max(1, min(Threads.nthreads(), length(mesh.segments)))
  thread_entries = [Tuple{Int,SurfaceQuadrature}[] for _ in 1:worker_count]

  _run_chunks_with_scratch!(thread_entries,
                            length(mesh.segments)) do entries, first_segment, last_segment
    for segment_index in first_segment:last_segment
      segment = mesh.segments[segment_index]
      first_point = ntuple(axis -> T(mesh.points[segment[1]][axis]), 2)
      second_point = ntuple(axis -> T(mesh.points[segment[2]][axis]), 2)

      for leaf in leaves
        quadrature = _segment_leaf_surface_quadrature(domain, leaf, first_point, second_point,
                                                      point_count, T)
        quadrature === nothing && continue
        push!(entries, (segment_index, quadrature))
      end
    end
  end

  segment_quadratures = [SurfaceQuadrature[] for _ in eachindex(mesh.segments)]

  for entries in thread_entries
    for (segment_index, quadrature) in entries
      push!(segment_quadratures[segment_index], quadrature)
    end
  end

  quadratures = SurfaceQuadrature[]

  for local_quadratures in segment_quadratures
    append!(quadratures, local_quadratures)
  end

  return quadratures
end

function _segment_leaf_surface_quadrature(domain::Domain{2,T}, leaf::Int, first_point::NTuple{2,T},
                                          second_point::NTuple{2,T}, quadrature_point_count::Int,
                                          ::Type{T}) where {T<:AbstractFloat}
  _segment_bbox_overlaps(domain, leaf, first_point, second_point, T) || return nothing
  clipped = _clip_segment_to_leaf(domain, leaf, first_point, second_point, T)
  clipped === nothing && return nothing
  midpoint = _point_midpoint(clipped[1], clipped[2], T)
  _owns_surface_piece(domain, leaf, midpoint, T) || return nothing
  return _segment_surface_quadrature(domain, leaf, clipped[1], clipped[2], quadrature_point_count,
                                     T)
end

@inline function _leaf_bounds(domain::Domain{2,T}, leaf::Int) where {T<:AbstractFloat}
  return cell_lower(domain, leaf), cell_upper(domain, leaf)
end

# Cheap bounding-box prefilter before attempting exact segment clipping.
function _segment_bbox_overlaps(domain::Domain{2,T}, leaf::Int, first_point::NTuple{2,T},
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

function _clip_segment_to_leaf(domain::Domain{2,T}, leaf::Int, first_point::NTuple{2,T},
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
function _owns_surface_piece(domain::Domain{2,T}, leaf::Int, point::NTuple{2,T},
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
function _segment_surface_quadrature(domain::Domain{2,T}, leaf::Int, first_point::NTuple{2,T},
                                     second_point::NTuple{2,T}, quadrature_point_count::Int,
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
  first_index = _checked_index(segment[1], point_count, "segment point")
  second_index = _checked_index(segment[2], point_count, "segment point")
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
  values = ntuple(axis -> T(normal[axis]), dimension_count)
  magnitude = sqrt(sum(values[axis]^2 for axis in 1:dimension_count))
  isfinite(magnitude) || throw(ArgumentError("surface-quadrature normals must be finite"))
  magnitude > zero(T) || throw(ArgumentError("surface-quadrature normals must be non-zero"))
  return ntuple(axis -> values[axis] / magnitude, dimension_count)
end

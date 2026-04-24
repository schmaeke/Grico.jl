# This file makes codimension-0 embedded geometry a first-class concept. A
# `PhysicalDomain` wraps the Cartesian background mesh together with a region
# descriptor, and the rest of the library treats that wrapper as an ordinary
# domain whose active leaves are the physical leaves only. For the current
# library scope, the region backend is an implicit classifier with finite-cell
# quadrature support.
#
# Two ideas drive the implementation:
# - the background `Domain` remains the sole owner of topology and affine
#   geometry,
# - the region owns the "which parts of that background are physical?" logic,
#   including cached inside/cut/outside classification and cut-cell quadratures.
#
# This keeps finite-cell support out of example-local quadrature loops. Spaces,
# compilation, and adaptivity can work from one coherent domain abstraction
# instead.

"""
    AbstractPhysicalRegion

Abstract supertype for physical-region descriptions used by
[`PhysicalDomain`](@ref).

Concrete region implementations decide which background leaves are fully inside,
cut, or fully outside the physical domain, and may also provide cached cut-cell
quadratures. The current first-party backend is [`ImplicitRegion`](@ref).
"""
abstract type AbstractPhysicalRegion end

"""
    PhysicalDomain(background, region)

Wrap a Cartesian background `Domain` together with a physical-region
description.

`PhysicalDomain` is the first-class finite-cell domain type of `Grico.jl`. The
background `Domain` still owns the refinement tree and the affine box, while
`region` decides which active leaves are fully physical, cut, or fully
fictitious. When an `HpSpace` is built on a `PhysicalDomain`, fully fictitious
background leaves are dropped automatically and cut-cell quadratures are later
injected automatically during `compile`. Leaves whose physical intersection has
zero volume, such as a cell that only touches the implicit boundary at one
corner, are treated as outside and are therefore trimmed as well.

`PhysicalDomain` does not modify the wrapped background mesh. It only changes
how later discretization layers interpret that mesh.
"""
struct PhysicalDomain{D,T<:AbstractFloat,B<:Domain{D,T},R<:AbstractPhysicalRegion} <:
       AbstractDomain{D,T}
  background::B
  region::R
end

struct _LeafSignature{D,T<:AbstractFloat}
  origin::NTuple{D,T}
  extent::NTuple{D,T}
  root_counts::NTuple{D,Int}
  level::NTuple{D,Int}
  coordinate::NTuple{D,Int}
end

struct _QuadratureSignature{D,T<:AbstractFloat}
  leaf::_LeafSignature{D,T}
  quadrature_shape::NTuple{D,Int}
end

"""
    ImplicitRegion(classifier; subdivision_depth=2)

Implicit physical region described by a classifier in physical coordinates.

The classifier uses the finite-cell sign convention

  classifier(x) < 0  inside,
  classifier(x) = 0  boundary,
  classifier(x) > 0  outside.

Boolean classifiers are also accepted: `true` means inside and `false` means
outside. `ImplicitRegion` caches leaf classifications and cut-cell quadratures
by a geometry-aware signature of each background cell. Reusing the same region
object across copied or adaptively rebuilt domains therefore also reuses those
caches, while domains with different physical boxes get independent cache
entries. For codimension-0 integration, cells with only measure-zero contact to
the physical region are classified as outside so the active-leaf set remains a
true volume discretization.
"""
struct ImplicitRegion{F} <: AbstractPhysicalRegion
  classifier::F
  subdivision_depth::Int
  leaf_classification_cache::Dict{_LeafSignature,Symbol}
  cut_quadrature_cache::Dict{_QuadratureSignature,AbstractQuadrature}
end

function ImplicitRegion(classifier; subdivision_depth::Integer=2)
  checked_depth = _checked_nonnegative(subdivision_depth, "subdivision_depth")
  return ImplicitRegion(classifier, checked_depth, Dict{_LeafSignature,Symbol}(),
                        Dict{_QuadratureSignature,AbstractQuadrature}())
end

grid(domain::PhysicalDomain) = grid(domain.background)
geometry(domain::PhysicalDomain) = geometry(domain.background)

# Physical-domain copies intentionally share the region object. This preserves
# cached leaf classifications and cut quadratures across copied/adapted domains
# and makes "same physical region" an inexpensive identity test elsewhere.
function Base.copy(domain::PhysicalDomain{D,T}) where {D,T<:AbstractFloat}
  return PhysicalDomain(copy(domain.background), domain.region)
end

@inline _background_domain(domain::Domain) = domain
@inline _background_domain(domain::PhysicalDomain) = domain.background

@inline _physical_region(::Domain) = nothing
@inline _physical_region(domain::PhysicalDomain) = domain.region

@inline function _leaf_signature(domain::AbstractDomain{D}, leaf::Int) where {D}
  grid_data = grid(domain)
  return _LeafSignature(origin(domain), extent(domain), root_cell_counts(grid_data),
                        level(grid_data, leaf), logical_coordinate(grid_data, leaf))
end

@inline function _quadrature_signature(domain::AbstractDomain{D}, leaf::Int,
                                       quadrature_shape::NTuple{D,Int}) where {D}
  return _QuadratureSignature(_leaf_signature(domain, leaf), quadrature_shape)
end

function _cached_leaf_classification(region::ImplicitRegion, signature)
  return get(region.leaf_classification_cache, signature, nothing)
end

function _store_leaf_classification!(region::ImplicitRegion, signature, classification::Symbol)
  region.leaf_classification_cache[signature] = classification
  return classification
end

function _cached_cut_quadrature(region::ImplicitRegion, signature)
  return get(region.cut_quadrature_cache, signature, nothing)
end

function _store_cut_quadrature!(region::ImplicitRegion, signature, quadrature)
  region.cut_quadrature_cache[signature] = quadrature
  return quadrature
end

# Domain-level active leaves. Plain background domains keep their full active
# frontier, while physical domains keep only the non-outside portion selected by
# the owning region. The region backend is responsible for providing a usable
# default quadrature on every cut leaf that remains active.
_domain_active_leaves(domain::Domain) = active_leaves(grid(domain))

@inline _is_domain_active_leaf(domain::Domain, leaf::Int) = is_active_leaf(grid(domain), leaf)

function _domain_active_leaves(domain::PhysicalDomain{D,T}) where {D,T<:AbstractFloat}
  grid_data = grid(domain)
  leaves = Int[]
  sizehint!(leaves, active_leaf_count(grid_data))

  for leaf in active_leaves(grid_data)
    _classify_leaf(domain.region, domain, leaf) === :outside || push!(leaves, leaf)
  end

  isempty(leaves) &&
    throw(ArgumentError("physical domains must contain at least one active physical leaf"))
  return leaves
end

@inline function _is_domain_active_leaf(domain::PhysicalDomain, leaf::Int)
  return _classify_leaf(domain.region, domain, leaf) !== :outside
end

function _classify_leaf(region::ImplicitRegion, domain::AbstractDomain{D,T},
                        leaf::Integer) where {D,T<:AbstractFloat}
  grid_data = grid(domain)
  checked_leaf = _checked_cell(grid_data, leaf)
  is_active_leaf(grid_data, checked_leaf) ||
    throw(ArgumentError("physical-region queries require active background leaves"))
  signature = _leaf_signature(domain, checked_leaf)
  cached = _cached_leaf_classification(region, signature)
  cached === nothing || return cached
  lower = ntuple(_ -> -one(T), D)
  upper = ntuple(_ -> one(T), D)
  classification = _classify_subcell(_background_domain(domain), checked_leaf, region.classifier,
                                     lower, upper, T)
  return _store_leaf_classification!(region, signature, classification)
end

function _cut_cell_quadrature(region::ImplicitRegion, domain::AbstractDomain{D,T}, leaf::Integer,
                              quadrature_shape::NTuple{D,<:Integer}) where {D,T<:AbstractFloat}
  grid_data = grid(domain)
  checked_leaf = _checked_cell(grid_data, leaf)
  is_active_leaf(grid_data, checked_leaf) ||
    throw(ArgumentError("finite-cell quadrature can only be built on active leaves"))
  checked_shape = ntuple(axis -> _checked_positive(quadrature_shape[axis],
                                                   "quadrature_shape[$axis]"), D)
  classification = _classify_leaf(region, domain, checked_leaf)
  classification == :outside && return nothing
  base_quadrature = TensorQuadrature(T, checked_shape)
  classification == :inside && return _point_quadrature(base_quadrature)
  signature = _quadrature_signature(domain, checked_leaf, checked_shape)
  cached = _cached_cut_quadrature(region, signature)
  cached === nothing || return cached
  lower = ntuple(_ -> -one(T), D)
  upper = ntuple(_ -> one(T), D)
  points = NTuple{D,T}[]
  weights = T[]
  _collect_finite_cell_candidates!(points, weights, _background_domain(domain), checked_leaf,
                                   region.classifier, base_quadrature, lower, upper,
                                   region.subdivision_depth)
  isempty(points) && return nothing
  quadrature = _moment_fit_finite_cell_quadrature(points, weights, checked_shape)
  return _store_cut_quadrature!(region, signature, quadrature)
end

@inline _default_cell_quadrature(::Domain, leaf::Integer, quadrature_shape) = nothing

function _default_cell_quadrature(domain::PhysicalDomain{D,T}, leaf::Integer,
                                  quadrature_shape::NTuple{D,<:Integer}) where {D,T<:AbstractFloat}
  region = domain.region
  _classify_leaf(region, domain, leaf) === :cut || return nothing
  return _cut_cell_quadrature(region, domain, leaf, quadrature_shape)
end

"""
    finite_cell_quadrature(domain, leaf, quadrature_shape, classifier; subdivision_depth=2)

Construct a reference-cell quadrature for the physical part of one background
leaf defined by `classifier`.

The classifier is evaluated in physical coordinates and follows the convention

  classifier(x) < 0  inside,
  classifier(x) = 0  boundary,
  classifier(x) > 0  outside.

Boolean classifiers are interpreted as indicator functions: `true` means inside
and `false` means outside.

If the leaf is fully outside, the result is `nothing`. If it is fully inside,
the result is the ordinary tensor-product rule corresponding to
`quadrature_shape`. Otherwise a non-negative moment-fitted cut-cell quadrature
is returned on the biunit reference cell of `leaf`.

This is the same finite-cell backend that [`PhysicalDomain`](@ref) uses
automatically during problem compilation.
"""
function finite_cell_quadrature(domain::AbstractDomain{D,T}, leaf::Integer,
                                quadrature_shape::NTuple{D,<:Integer}, classifier;
                                subdivision_depth::Integer=2) where {D,T<:AbstractFloat}
  region = ImplicitRegion(classifier; subdivision_depth=subdivision_depth)
  return _cut_cell_quadrature(region, domain, leaf, quadrature_shape)
end

# Convert a tensor-product quadrature rule to explicit point/weight storage so
# the later finite-cell reduction can work on mutable candidate sets.
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

@inline function _subcell_corner_count(::Val{D}) where {D}
  D < Sys.WORD_SIZE - 1 || throw(ArgumentError("subcell corner count must be Int-representable"))
  return 1 << D
end

@inline _subcell_sample_count(::Val{D}) where {D} = _subcell_corner_count(Val(D)) + 1

function _collect_finite_cell_candidates!(points::Vector{NTuple{D,T}}, weights::Vector{T},
                                          domain::Domain{D,T}, leaf::Int, classifier,
                                          base_quadrature::TensorQuadrature{D,T},
                                          lower::NTuple{D,T}, upper::NTuple{D,T},
                                          max_depth::Int) where {D,T<:AbstractFloat}
  _collect_finite_cell_candidates_recursive!(points, weights, domain, leaf, classifier,
                                             base_quadrature, lower, upper, 0, max_depth)
  return nothing
end

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
  corner_count = _subcell_corner_count(Val(D))

  for child_mask in 0:(corner_count-1)
    child_lower, child_upper = _subcell_child_bounds(lower, midpoint, upper, child_mask)
    _collect_finite_cell_candidates_recursive!(points, weights, domain, leaf, classifier,
                                               base_quadrature, child_lower, child_upper, depth + 1,
                                               max_depth)
  end

  return nothing
end

const _FINITE_CELL_MOMENT_TOLERANCE_SCALE = 1.0e3
const _FINITE_CELL_WEIGHT_PRUNE_RATIO = 1.0e-8
const _FINITE_CELL_MAX_ELIMINATION_ITERATIONS = 1024

@inline _finite_cell_moment_tolerance(::Type{T}) where {T<:AbstractFloat} = T(_FINITE_CELL_MOMENT_TOLERANCE_SCALE) *
                                                                            eps(T)

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

function _finite_cell_moment_matrix(points::Vector{NTuple{D,T}}, degrees::NTuple{D,Int},
                                    modes::Vector{NTuple{D,Int}}) where {D,T<:AbstractFloat}
  matrix = Matrix{T}(undef, length(modes), length(points))
  axis_buffers = ntuple(axis -> Vector{T}(undef, degrees[axis] + 1), D)

  for point_index in eachindex(points)
    point_data = points[point_index]

    for axis in 1:D
      _legendre_values!(point_data[axis], degrees[axis], axis_buffers[axis])
    end

    for mode_index in eachindex(modes)
      mode = modes[mode_index]
      value = one(T)

      @inbounds for axis in 1:D
        value *= axis_buffers[axis][mode[axis]+1]
      end

      matrix[mode_index, point_index] = value
    end
  end

  return matrix
end

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

function _removable_weight_indices(weights::Vector{T}, min_point_count::Int,
                                   tolerance::T) where {T<:AbstractFloat}
  length(weights) > min_point_count || return Int[]
  maximum_weight = maximum(weights)
  threshold = max(tolerance, T(_FINITE_CELL_WEIGHT_PRUNE_RATIO) * maximum_weight)
  removable = findall(weight -> weight <= threshold, weights)
  isempty(removable) && return [argmin(weights)]
  return sort(removable; by=index -> weights[index])
end

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

function _subcell_sample_extrema(domain::AbstractDomain{D,T}, leaf::Int, classifier,
                                 lower::NTuple{D,T}, upper::NTuple{D,T}, classifier_value,
                                 ::Type{T}) where {D,T<:AbstractFloat}
  minimum_value = typemax(T)
  maximum_value = typemin(T)
  corner_count = _subcell_corner_count(Val(D))

  for corner_mask in 0:(corner_count-1)
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

# The subcell classifier samples corners plus the midpoint. This is a compact
# heuristic rather than an exact geometric predicate, but it is consistent with
# the later candidate generation: negative samples provide physical volume
# evidence, positive samples provide outside evidence, and pure zero-contact
# subcells are treated as outside so the active set remains a codimension-0
# volume discretization.
function _classify_subcell(domain::AbstractDomain{D,T}, leaf::Int, classifier, lower::NTuple{D,T},
                           upper::NTuple{D,T}, ::Type{T}) where {D,T<:AbstractFloat}
  has_negative = false
  has_positive = false
  corner_count = _subcell_corner_count(Val(D))

  for corner_mask in 0:(corner_count-1)
    ξ = _subcell_corner(lower, upper, corner_mask)
    value = _classifier_value(classifier, map_from_biunit_cube(domain, leaf, ξ), T)
    has_negative |= value < zero(T)
    has_positive |= value > zero(T)
  end

  center = _subcell_midpoint(lower, upper, T)
  center_value = _classifier_value(classifier, map_from_biunit_cube(domain, leaf, center), T)
  has_negative |= center_value < zero(T)
  has_positive |= center_value > zero(T)

  !has_negative && return :outside
  !has_positive && return :inside
  return :cut
end

function _append_subcell_quadrature!(points::Vector{NTuple{D,T}}, weights::Vector{T},
                                     domain::AbstractDomain{D,T}, leaf::Int, classifier,
                                     base_quadrature::TensorQuadrature{D,T}, lower::NTuple{D,T},
                                     upper::NTuple{D,T},
                                     filter_points::Bool) where {D,T<:AbstractFloat}
  center = _subcell_midpoint(lower, upper, T)
  half_size = ntuple(axis -> T(0.5) * (upper[axis] - lower[axis]), D)
  scale = prod(half_size)

  appended = false

  for point_index in 1:point_count(base_quadrature)
    η = point(base_quadrature, point_index)
    ξ = ntuple(axis -> muladd(half_size[axis], η[axis], center[axis]), D)

    if filter_points
      _classifier_value(classifier, map_from_biunit_cube(domain, leaf, ξ), T) < zero(T) || continue
    end

    push!(points, ξ)
    push!(weights, weight(base_quadrature, point_index) * scale)
    appended = true
  end

  if filter_points && !appended
    _append_subcell_sample_fallback!(points, weights, domain, leaf, classifier, lower, upper, T)
  end

  return nothing
end

# When a terminal cut subcell is so small that none of the tensor quadrature
# points land in the physical part, keep the subcell alive by inserting a tiny
# sample rule on the negative corner/center samples that triggered the `:cut`
# classification. This is intentionally only a last-resort sliver safeguard.
function _append_subcell_sample_fallback!(points::Vector{NTuple{D,T}}, weights::Vector{T},
                                          domain::AbstractDomain{D,T}, leaf::Int, classifier,
                                          lower::NTuple{D,T}, upper::NTuple{D,T},
                                          ::Type{T}) where {D,T<:AbstractFloat}
  sample_total = _subcell_sample_count(Val(D))
  sample_weight = prod(ntuple(axis -> upper[axis] - lower[axis], D)) / sample_total
  center = _subcell_midpoint(lower, upper, T)

  if _classifier_value(classifier, map_from_biunit_cube(domain, leaf, center), T) < zero(T)
    push!(points, center)
    push!(weights, sample_weight)
  end

  corner_count = _subcell_corner_count(Val(D))

  for corner_mask in 0:(corner_count-1)
    ξ = _subcell_corner(lower, upper, corner_mask)
    _classifier_value(classifier, map_from_biunit_cube(domain, leaf, ξ), T) < zero(T) || continue
    push!(points, ξ)
    push!(weights, sample_weight)
  end

  return nothing
end

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

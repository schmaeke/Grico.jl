# This file provides the reference-domain quadrature layer of the library.
#
# In the rest of `Grico.jl`, quadrature is the numerical counterpart of the
# polynomial constructions defined in `polynomials.jl`: once basis functions and
# weak-form terms are known symbolically, quadrature turns them into weighted
# sums over reference points.
#
# The file is organized around three complementary ideas:
# 1. a minimal abstract interface for iterating over weighted points,
# 2. concrete rule families for one-dimensional Gauss-Legendre rules, tensor
#    products of such rules, and general point clouds,
# 3. construction and indexing helpers that keep later integration code generic
#    while still avoiding unnecessary storage.
#
# The mathematically central case is Gauss-Legendre quadrature on `[-1, 1]`.
# Those rules integrate one-dimensional polynomials exactly up to degree
# `2n - 1`, and tensor products of them provide the default cell and face
# quadrature throughout the package. The more general `PointQuadrature` type is
# used when geometry-driven constructions break that tensor structure.

# Numerical tolerances for Newton refinement of Gauss-Legendre roots. The
# tolerance scales with machine precision of the chosen floating-point type,
# while the iteration cap is intentionally generous because rule construction is
# not a dominant runtime hotspot.
const _GAUSS_LEGENDRE_TOLERANCE_SCALE = 1.0e3
const _GAUSS_LEGENDRE_MAX_NEWTON_ITERATIONS = 32

# Abstract interface.

"""
    AbstractQuadrature{D,T}

Abstract supertype for quadrature rules over `D`-dimensional reference domains
with floating-point type `T`.

Concrete subtypes in this library expose a common minimal interface:
[`dimension`](@ref), [`point_count`](@ref), [`point`](@ref), and [`weight`](@ref).
Tensor-product and point-cloud rules both implement this interface so
integration code can be written generically. The interface is intentionally
small because most later algorithms only need to traverse weighted reference
points and should not depend on how those points were generated.
"""
abstract type AbstractQuadrature{D,T<:AbstractFloat} end

# Concrete rule families.

"""
    GaussLegendreRule(points, weights)

One-dimensional Gauss-Legendre quadrature rule on the reference interval
`[-1,1]`.

An `n`-point Gauss-Legendre rule integrates every polynomial of degree at most
`2n - 1` exactly. The points are the roots of the Legendre polynomial `Pₙ`, and
the weights are positive. This rule is the basic one-dimensional building block
for tensor-product quadrature throughout the library.

In practice, this is the canonical reference rule from which the default cell,
face, and many verification quadratures are assembled.
"""
struct GaussLegendreRule{T<:AbstractFloat} <: AbstractQuadrature{1,T}
  points::Vector{T}
  weights::Vector{T}

  # Validate the rule as a genuine quadrature rule on `[-1,1]`: the point and
  # weight vectors must match, all values must be finite, points must lie in the
  # reference interval, and weights must be strictly positive.
  function GaussLegendreRule{T}(points::Vector{T}, weights::Vector{T}) where {T<:AbstractFloat}
    length(points) == length(weights) ||
      throw(ArgumentError("points and weights must have matching lengths"))
    !isempty(points) || throw(ArgumentError("Gauss-Legendre rules require at least one point"))
    all(isfinite, points) || throw(ArgumentError("points must be finite"))
    all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
    all(point -> -one(T) <= point <= one(T), points) ||
      throw(ArgumentError("points must lie in [-1, 1]"))
    all(weight -> weight > zero(T), weights) || throw(ArgumentError("weights must be positive"))
    return new{T}(points, weights)
  end
end

"""
    TensorQuadrature(shape)
    TensorQuadrature(T, shape)
    TensorQuadrature(rules)

Tensor-product quadrature rule on `[-1,1]^D`.

`TensorQuadrature` stores one one-dimensional Gauss-Legendre rule per axis and
forms their Cartesian product. If the per-axis point counts are
`(n₁, …, n_D)`, the total point count is `∏ₐ nₐ`, and the weight of a tensor
point is the product of the corresponding one-dimensional weights.

The constructor `TensorQuadrature(shape)` builds Gauss-Legendre rules with the
requested point counts on each axis. Passing an explicit tuple of rules allows
reusing already constructed one-dimensional rules. The zero-dimensional case
`TensorQuadrature(())` is supported and represents the empty tensor product with
one point `()` and weight `1`.

Internally, tensor points are not materialized as one large coordinate array.
Instead, the rule stores its one-dimensional factors together with mixed-radix
metadata that lets [`point`](@ref), [`coordinate`](@ref), and [`weight`](@ref)
decode any tensor point on demand.
"""
struct TensorQuadrature{D,T<:AbstractFloat} <: AbstractQuadrature{D,T}
  rules::NTuple{D,GaussLegendreRule{T}}
  shape::NTuple{D,Int}
  stride::NTuple{D,Int}
  count::Int
end

"""
    PointQuadrature(points, weights)

Explicit quadrature rule defined by a list of points and weights.

Unlike [`TensorQuadrature`](@ref), this representation imposes no tensor-product
structure. It is useful for embedded, cut-cell, or otherwise custom quadrature
rules where points and weights are generated geometrically rather than by a
standard reference product rule.

This is the general-purpose fallback representation of the package: any
geometrically generated weighted point set can participate in the same
integration layer once it is packaged as a `PointQuadrature`.
"""
struct PointQuadrature{D,T<:AbstractFloat} <: AbstractQuadrature{D,T}
  points::Vector{NTuple{D,T}}
  weights::Vector{T}

  # A point-cloud rule is valid as long as points and weights match in length,
  # the rule is nonempty, and all numerical data are finite.
  function PointQuadrature{D,T}(points::Vector{NTuple{D,T}},
                                weights::Vector{T}) where {D,T<:AbstractFloat}
    length(points) == length(weights) ||
      throw(ArgumentError("points and weights must have matching lengths"))
    !isempty(points) || throw(ArgumentError("point quadrature requires at least one point"))
    all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
    for point in points
      all(isfinite, point) || throw(ArgumentError("quadrature points must be finite"))
    end
    return new{D,T}(points, weights)
  end
end

"""
    dimension(quadrature)

Return the reference-space dimension `D` of the quadrature rule.

For example, a `GaussLegendreRule` has dimension `1`, while a tensor-product
rule over `[-1,1]^3` has dimension `3`. The zero-dimensional tensor-product rule
`TensorQuadrature(())` correspondingly has dimension `0`.
"""
dimension(::AbstractQuadrature{D}) where {D} = D

"""
    point_count(quadrature)

Return the total number of quadrature points.

For tensor-product rules this is the product of the per-axis point counts; for
point-cloud rules it is simply the stored number of weighted points. The
zero-dimensional tensor product has point count `1`, representing the empty
product of one-dimensional rules.
"""
point_count(rule::GaussLegendreRule) = length(rule.points)
point_count(quadrature::TensorQuadrature) = quadrature.count
point_count(quadrature::PointQuadrature) = length(quadrature.weights)

"""
    axis_point_counts(quadrature::TensorQuadrature)

Return the number of quadrature points on each tensor axis.

If `quadrature` is built from one-dimensional rules with `n₁, …, n_D` points,
this function returns `(n₁, …, n_D)`. The product of these counts equals
[`point_count`](@ref).
"""
axis_point_counts(quadrature::TensorQuadrature) = quadrature.shape

"""
    point(quadrature, point_index)

Return the coordinates of one quadrature point.

The point is returned as a tuple of length [`dimension`](@ref). For
one-dimensional Gauss-Legendre rules, the return value is a one-tuple so the
interface matches the multidimensional cases.

For tensor-product rules, points are enumerated in mixed-radix order with axis
1 varying fastest. This is the same tensor ordering convention used elsewhere
in the package for local modal and quadrature data.
"""
function point(rule::GaussLegendreRule{T}, point_index::Integer) where {T}
  return (@inbounds(rule.points[_checked_index(point_index, point_count(rule), "point")]),)
end

function point(quadrature::PointQuadrature{D,T}, point_index::Integer) where {D,T}
  return @inbounds quadrature.points[_checked_index(point_index, point_count(quadrature), "point")]
end

# Tensor points are enumerated in mixed-radix order with axis 1 varying fastest.
# The precomputed strides invert that numbering without materializing the full
# tensor-product point array.
function point(quadrature::TensorQuadrature{D,T}, point_index::Integer) where {D,T}
  linear = _tensor_linear_point_index(quadrature, point_index)
  local_indices = _tensor_local_point_indices(quadrature, linear)
  return ntuple(axis -> @inbounds(quadrature.rules[axis].points[local_indices[axis]]), D)
end

"""
    coordinate(quadrature, point_index, axis)

Return one coordinate of one quadrature point.

This is a convenience wrapper around [`point`](@ref) when only a single axis
coordinate is needed. The specialized tensor-product method avoids constructing
the full point tuple when a caller only needs one coordinate.
"""
function coordinate(quadrature::AbstractQuadrature{D}, point_index::Integer,
                    axis::Integer) where {D}
  return point(quadrature, point_index)[_checked_index(axis, D, "axis")]
end

function coordinate(quadrature::TensorQuadrature, point_index::Integer, axis::Integer)
  checked_axis = _checked_index(axis, dimension(quadrature), "axis")
  linear = _tensor_linear_point_index(quadrature, point_index)
  local_index = _tensor_local_point_index(quadrature, linear, checked_axis)
  return @inbounds quadrature.rules[checked_axis].points[local_index]
end

"""
    weight(quadrature, point_index)

Return the quadrature weight associated with one point.

For tensor-product rules this is the product of the corresponding
one-dimensional weights; for point-cloud rules it is the stored point weight.
For Gauss-Legendre rules on `[-1, 1]`, all weights are strictly positive.
"""
function weight(rule::GaussLegendreRule, point_index::Integer)
  @inbounds rule.weights[_checked_index(point_index, point_count(rule), "point")]
end

function weight(quadrature::PointQuadrature, point_index::Integer)
  @inbounds quadrature.weights[_checked_index(point_index, point_count(quadrature), "point")]
end

# As for the tensor point coordinates, the linear point index is decoded by a
# mixed-radix traversal and the one-dimensional weights are multiplied.
function weight(quadrature::TensorQuadrature{D,T}, point_index::Integer) where {D,T}
  linear = _tensor_linear_point_index(quadrature, point_index)
  local_indices = _tensor_local_point_indices(quadrature, linear)
  value = one(T)

  @inbounds for axis in 1:dimension(quadrature)
    value *= quadrature.rules[axis].weights[local_indices[axis]]
  end

  return value
end

# Exactness bookkeeping for one-dimensional Gauss-Legendre rules.

"""
    gauss_legendre_exact_degree(point_count)

Return the highest polynomial degree integrated exactly by a `point_count`-point
Gauss-Legendre rule on `[-1,1]`.

The exactness degree is `2 * point_count - 1`.
"""
function gauss_legendre_exact_degree(point_count::Integer)
  return 2 * _checked_positive(point_count, "point_count") - 1
end

"""
    minimum_gauss_legendre_points(exact_degree)

Return the smallest number of Gauss-Legendre points needed to integrate all
polynomials of degree at most `exact_degree` exactly.

This is the inverse of [`gauss_legendre_exact_degree`](@ref), rounded up to the
next admissible integer point count. Equivalently, it is the smallest integer
`n` such that `2n - 1 ≥ exact_degree`.
"""
function minimum_gauss_legendre_points(exact_degree::Integer)
  degree = _checked_nonnegative(exact_degree, "exact_degree")
  return max(1, (degree + 2) ÷ 2)
end

# Gauss-Legendre rule construction.

"""
    gauss_legendre_rule(point_count)
    gauss_legendre_rule(T, point_count)

Construct a `point_count`-point Gauss-Legendre rule on `[-1,1]`.

The implementation computes the roots of `Pₙ` by Newton iteration, starting from
the standard asymptotic cosine guesses, and then evaluates the classical weight
formula

  wᵢ = 2 / ((1 - xᵢ²) (Pₙ'(xᵢ))²).

By symmetry, only half of the roots are solved explicitly; the remaining points
and weights are mirrored.

The returned rule stores its points in ascending order from `-1` to `1`.
"""
gauss_legendre_rule(point_count::Integer) = gauss_legendre_rule(Float64, point_count)

function gauss_legendre_rule(::Type{T}, point_count::Integer) where {T<:AbstractFloat}
  count = _checked_positive(point_count, "point_count")
  points = Vector{T}(undef, count)
  weights = Vector{T}(undef, count)
  values = Vector{T}(undef, count + 1)
  derivatives = Vector{T}(undef, count + 1)
  half = cld(count, 2)

  @inbounds for i in 1:half
    # The cosine formula gives the classical asymptotic starting guess for the
    # `i`-th positive root of `Pₙ`. Solving only one half of the spectrum and
    # then mirroring preserves symmetry and halves the Newton work.
    initial = cos(T(pi) * (T(i) - T(0.25)) / (T(count) + T(0.5)))
    x = _gauss_legendre_root!(values, derivatives, count, i, initial)
    w = T(2) / ((one(T) - x * x) * derivatives[end]^2)
    points[i] = -x
    weights[i] = w
    mirror = count - i + 1
    points[mirror] = x
    weights[mirror] = w
  end

  return GaussLegendreRule{T}(points, weights)
end

@inline _gauss_legendre_tolerance(::Type{T}) where {T<:AbstractFloat} = T(_GAUSS_LEGENDRE_TOLERANCE_SCALE) *
                                                                        eps(T)

# Refine one Legendre root by Newton iteration on `Pₙ(x) = 0`. The temporary
# buffers store the full Legendre table and its derivatives up to degree `n`, so
# `values[end]` and `derivatives[end]` are exactly `Pₙ(x)` and `Pₙ'(x)`.
#
# The final re-evaluation after convergence is deliberate: the caller uses the
# derivative at the converged root to build the quadrature weight, so the
# scratch buffers should correspond to the final `x`, not to the previous
# Newton state.
function _gauss_legendre_root!(values::AbstractVector{T}, derivatives::AbstractVector{T},
                               point_count::Int, root_index::Int,
                               initial::T) where {T<:AbstractFloat}
  x = initial
  tolerance = _gauss_legendre_tolerance(T)

  for _ in 1:_GAUSS_LEGENDRE_MAX_NEWTON_ITERATIONS
    _legendre_values_and_derivatives!(x, point_count, values, derivatives)
    derivative = derivatives[end]
    isfinite(derivative) ||
      throw(ArgumentError("Gauss-Legendre root solve produced a non-finite derivative"))
    delta = values[end] / derivative
    isfinite(delta) ||
      throw(ArgumentError("Gauss-Legendre root solve produced a non-finite update"))
    x -= delta

    if abs(delta) <= tolerance
      # Re-evaluate at the converged point so the caller receives consistent
      # final values and derivatives for weight construction.
      _legendre_values_and_derivatives!(x, point_count, values, derivatives)
      return x
    end
  end

  throw(ArgumentError("Gauss-Legendre root $root_index of a $point_count-point rule did not converge"))
end

# Tensor-product and point-cloud constructors.

TensorQuadrature(shape::NTuple{D,<:Integer}) where {D} = TensorQuadrature(Float64, shape)
TensorQuadrature(::Tuple{}) = TensorQuadrature(Float64, ())

# Build a tensor-product rule from per-axis point counts by first constructing
# one-dimensional Gauss-Legendre rules on each axis.
function TensorQuadrature(::Type{T}, shape::NTuple{D,<:Integer}) where {D,T<:AbstractFloat}
  rules = ntuple(axis -> gauss_legendre_rule(T, shape[axis]), D)
  return TensorQuadrature(rules)
end

function TensorQuadrature(::Type{T}, ::Tuple{}) where {T<:AbstractFloat}
  return TensorQuadrature{0,T}((), (), (), 1)
end

# Precompute the mixed-radix metadata used to index tensor-product points and
# weights without materializing the full coordinate grid explicitly. The stride
# convention matches the rest of the package: axis 1 varies fastest.
function TensorQuadrature(rules::NTuple{D,GaussLegendreRule{T}}) where {D,T<:AbstractFloat}
  shape = ntuple(axis -> point_count(rules[axis]), D)
  stride = ntuple(axis -> axis == 1 ? 1 : prod(shape[1:(axis-1)]), D)
  count = prod(shape; init=1)
  return TensorQuadrature{D,T}(rules, shape, stride, count)
end

function PointQuadrature(points::Vector{NTuple{D,T}}, weights::Vector{T}) where {D,T<:AbstractFloat}
  PointQuadrature{D,T}(points, weights)
end

# Mixed-radix tensor indexing utilities.
#
# Public APIs use one-based point indices, but tensor decoding is cleaner in a
# zero-based mixed-radix representation. These helpers convert between those two
# views without allocating temporary arrays.
@inline function _tensor_linear_point_index(quadrature::TensorQuadrature, point_index::Integer)
  return _checked_index(point_index, point_count(quadrature), "point") - 1
end

# Recover one axis-local point index from the zero-based mixed-radix tensor
# counter. Axis 1 varies fastest.
@inline function _tensor_local_point_index(quadrature::TensorQuadrature, linear::Int, axis::Int)
  return fld(linear, quadrature.stride[axis]) % quadrature.shape[axis] + 1
end

@inline function _tensor_local_point_indices(quadrature::TensorQuadrature{D}, linear::Int) where {D}
  return ntuple(axis -> _tensor_local_point_index(quadrature, linear, axis), D)
end

# This file provides the one-dimensional polynomial families used throughout
# the library on the reference interval `[-1, 1]`.
#
# The rest of `Grico.jl` relies on two closely related families:
# 1. the standard Legendre polynomials `Pₙ`, which provide orthogonal modal
#    data, exactness estimates, and the root structure behind Gauss-Legendre
#    quadrature;
# 2. the integrated family `ψₙ`, whose first two functions carry the endpoint
#    traces and whose higher functions vanish at both endpoints.
#
# That second family is the actual finite-element workhorse of the package.
# Tensor products of the `ψₙ` make it easy to distinguish trace-carrying modes
# from purely interior polynomial content, which is exactly the separation used
# later by the continuity compiler in `continuity.jl`. One extra wrinkle enters
# with DG spaces: when a local degree is exactly zero, the compiled finite-
# element basis uses a true cellwise constant mode rather than the lower-endpoint
# trace mode `ψ₀`. The helper routines for that special case live near the middle
# of this file.
#
# Three conventions are worth keeping in mind while reading this file:
# - requesting degree `p` always means "return all modes `0:p`",
# - entry `n + 1` stores the quantity associated with polynomial degree `n`,
# - the mutating routines require `x` and every supplied output buffer to share
#   the same floating-point type, which keeps the hot paths type-stable and
#   allocation-free.

# Public allocating interface.

"""
    legendre_values(x, degree)

Return the Legendre polynomials `P₀(x), …, P_degree(x)` evaluated at `x`.

The family is normalized by `P₀(x) = 1` and `P₁(x) = x` and is orthogonal on
`[-1, 1]` with respect to the usual `L²` inner product. In `Grico.jl` these
polynomials appear in two roles: directly in Gauss-Legendre quadrature
construction, and indirectly as the raw material from which the hierarchical
integrated family is built.

If `degree = p`, the returned vector has length `p + 1`, with entry `n + 1`
storing `Pₙ(x)`.
"""
function legendre_values(x::T, degree::Integer) where {T<:AbstractFloat}
  x, n = _checked_polynomial_problem(x, degree)
  values = _polynomial_output(T, n)
  _legendre_values!(x, n, values)
  return values
end

"""
    legendre_derivatives(x, degree)

Return the derivatives `P₀'(x), …, P_degree'(x)` of the Legendre family at `x`.

The result uses the same indexing convention as [`legendre_values`](@ref): entry
`n + 1` stores the derivative of `Pₙ`. These derivatives enter Newton
iterations for Gauss-Legendre nodes and, after tensor-product assembly, basis-
gradient evaluation on cells, faces, and embedded surfaces.
"""
function legendre_derivatives(x::T, degree::Integer) where {T<:AbstractFloat}
  x, n = _checked_polynomial_problem(x, degree)
  derivatives = _polynomial_output(T, n)
  _legendre_derivatives!(x, n, derivatives)
  return derivatives
end

"""
    integrated_legendre_values(x, degree)

Return the hierarchical integrated Legendre family evaluated at `x`.

This library uses the convention

  ψ₀(x) = (1 - x) / 2,
  ψ₁(x) = (1 + x) / 2,
  ψₙ(x) = (Pₙ(x) - Pₙ₋₂(x)) / √(4n - 2)  for n ≥ 2,

where `Pₙ` denotes the standard Legendre polynomial of degree `n`. The first
two functions are endpoint modes: `ψ₀(-1) = 1`, `ψ₀(1) = 0`, `ψ₁(-1) = 0`, and
`ψ₁(1) = 1`. Every higher mode vanishes at both endpoints and therefore serves
as interior polynomial content. In tensor products, this cleanly separates
trace-carrying modes from purely interior modes, which is why later continuity
code can detect boundary participation simply by checking whether a one-
dimensional mode index is `0` or `1`.

If `degree = p`, the returned vector has length `p + 1`, with entry `n + 1`
storing `ψₙ(x)`.
"""
function integrated_legendre_values(x::T, degree::Integer) where {T<:AbstractFloat}
  x, n = _checked_polynomial_problem(x, degree)
  values = _polynomial_output(T, n)
  _integrated_legendre_values!(x, n, values)
  return values
end

"""
    integrated_legendre_derivatives(x, degree)

Return the derivatives of the hierarchical integrated Legendre family at `x`.

For `n ≥ 2`, the derivative of the interior mode collapses to a scaled
Legendre polynomial,

  ψₙ'(x) = √((2n - 1) / 2) Pₙ₋₁(x),

while `ψ₀'(x) = -1/2` and `ψ₁'(x) = 1/2`. This relation is what makes the
integrated family attractive in practice: gradients can be evaluated directly
from standard Legendre data without numerical differentiation or symbolic
special cases beyond the two endpoint modes.
"""
function integrated_legendre_derivatives(x::T, degree::Integer) where {T<:AbstractFloat}
  x, n = _checked_polynomial_problem(x, degree)
  derivatives = _polynomial_output(T, n)
  _integrated_legendre_derivatives!(x, n, derivatives)
  return derivatives
end

# Public allocation-free interface.

"""
    legendre_values_and_derivatives!(x, degree, values, derivatives)

Write Legendre values and/or derivatives at `x` into preallocated buffers.

At least one of `values` or `derivatives` must be provided. When present, each
buffer must use one-based indexing and have length at least `degree + 1`. The
method signature also requires the buffers to have the same floating-point
element type as `x`, so callers can reuse scratch storage in hot loops without
hidden promotions or allocations.

The indexing convention matches [`legendre_values`](@ref): entry `n + 1`
corresponds to polynomial degree `n`. This routine is the allocation-free core
used in high-frequency code paths such as quadrature construction and basis
evaluation.
"""
function legendre_values_and_derivatives!(x::T, degree::Integer,
                                          values::Union{Nothing,AbstractVector{T}},
                                          derivatives::Union{Nothing,AbstractVector{T}}) where {T<:AbstractFloat}
  x, n = _checked_polynomial_problem(x, degree)
  _check_polynomial_outputs(n, values, derivatives)

  if values === nothing
    _legendre_derivatives!(x, n, derivatives::AbstractVector{T})
  elseif derivatives === nothing
    _legendre_values!(x, n, values::AbstractVector{T})
  else
    _legendre_values_and_derivatives!(x, n, values::AbstractVector{T},
                                      derivatives::AbstractVector{T})
  end

  return nothing
end

"""
    integrated_legendre_values_and_derivatives!(x, degree, values, derivatives)

Write integrated Legendre values and/or derivatives at `x` into preallocated
buffers.

This is the allocation-free counterpart of [`integrated_legendre_values`](@ref)
and [`integrated_legendre_derivatives`](@ref). At least one output buffer must
be provided, every supplied buffer must use one-based indexing and have length
at least `degree + 1`, and the buffers must share the same floating-point
element type as `x`.

Like the nonmutating interface, the function writes the full family
`ψ₀, …, ψ_degree` using the `n + 1 ↔ n` indexing convention.
"""
function integrated_legendre_values_and_derivatives!(x::T, degree::Integer,
                                                     values::Union{Nothing,AbstractVector{T}},
                                                     derivatives::Union{Nothing,AbstractVector{T}}) where {T<:AbstractFloat}
  x, n = _checked_polynomial_problem(x, degree)
  _check_polynomial_outputs(n, values, derivatives)

  if values === nothing
    _integrated_legendre_derivatives!(x, n, derivatives::AbstractVector{T})
  elseif derivatives === nothing
    _integrated_legendre_values!(x, n, values::AbstractVector{T})
  else
    _integrated_legendre_values_and_derivatives!(x, n, values::AbstractVector{T},
                                                 derivatives::AbstractVector{T})
  end

  return nothing
end

# Internal finite-element basis used by compiled `HpSpace` evaluation. For
# `degree >= 1` this matches the hierarchical integrated-Legendre family above.
# For `degree == 0`, however, DG spaces need a true cellwise constant basis
# function rather than the lower endpoint trace mode `ψ₀(x) = (1 - x) / 2`.
# Otherwise a nominally piecewise-constant DG space would still distinguish the
# left and right endpoints of the reference interval, which is not the intended
# one-dimensional degree-zero polynomial space.
function _fe_basis_values(x::T, degree::Integer) where {T<:AbstractFloat}
  x, n = _checked_polynomial_problem(x, degree)
  values = _polynomial_output(T, n)
  _fe_basis_values!(x, n, values)
  return values
end

function _fe_basis_values_and_derivatives!(x::T, degree::Integer,
                                           values::Union{Nothing,AbstractVector{T}},
                                           derivatives::Union{Nothing,AbstractVector{T}}) where {T<:AbstractFloat}
  x, n = _checked_polynomial_problem(x, degree)
  _check_polynomial_outputs(n, values, derivatives)

  if values === nothing
    _fe_basis_derivatives!(x, n, derivatives::AbstractVector{T})
  elseif derivatives === nothing
    _fe_basis_values!(x, n, values::AbstractVector{T})
  else
    _fe_basis_values_and_derivatives!(x, n, values::AbstractVector{T},
                                      derivatives::AbstractVector{T})
  end

  return nothing
end

function _fe_basis_values!(x::T, degree::Int, values::AbstractVector{T}) where {T<:AbstractFloat}
  if degree == 0
    values[1] = one(T)
    return nothing
  end

  return _integrated_legendre_values!(x, degree, values)
end

function _fe_basis_derivatives!(x::T, degree::Int,
                                derivatives::AbstractVector{T}) where {T<:AbstractFloat}
  if degree == 0
    derivatives[1] = zero(T)
    return nothing
  end

  return _integrated_legendre_derivatives!(x, degree, derivatives)
end

function _fe_basis_values_and_derivatives!(x::T, degree::Int, values::AbstractVector{T},
                                           derivatives::AbstractVector{T}) where {T<:AbstractFloat}
  if degree == 0
    values[1] = one(T)
    derivatives[1] = zero(T)
    return nothing
  end

  return _integrated_legendre_values_and_derivatives!(x, degree, values, derivatives)
end

# Validation and buffer checks. Dispatch already guarantees that `x` is a
# floating-point number, and that supplied buffers share its element type. The
# helpers below therefore enforce finiteness, Int-representable nonnegative
# degree values, output presence, buffer sizes, and non-aliasing outputs.
@inline function _checked_polynomial_problem(x::T, degree::Integer) where {T<:AbstractFloat}
  return _checked_polynomial_input(x), _checked_nonnegative(degree, "degree")
end

@inline function _checked_polynomial_input(x::T) where {T<:AbstractFloat}
  isfinite(x) || throw(ArgumentError("x must be finite"))
  return x
end

# Buffer element types are enforced by method dispatch. One-based indexing is
# required because the recurrence kernels write degree `n` at entry `n + 1`.
# Aliasing is rejected because those same kernels use output buffers as
# temporary polynomial storage before writing the requested family.
@inline function _check_polynomial_outputs(degree::Int, values::Union{Nothing,AbstractVector},
                                           derivatives::Union{Nothing,AbstractVector})
  values === nothing &&
    derivatives === nothing &&
    throw(ArgumentError("at least one output buffer is required"))
  values === nothing || _require_one_based_vector(values, "values")
  derivatives === nothing || _require_one_based_vector(derivatives, "derivatives")
  values === nothing || _require_length(values, degree + 1, "values")
  derivatives === nothing || _require_length(derivatives, degree + 1, "derivatives")
  values === nothing ||
    derivatives === nothing ||
    !Base.mightalias(values, derivatives) ||
    throw(ArgumentError("values and derivatives buffers must not alias"))
  return nothing
end

@inline _polynomial_output(::Type{T}, degree::Int) where {T<:AbstractFloat} = Vector{T}(undef,
                                                                                        degree + 1)

# Standard Legendre recurrence.
#
# The defining three-term recurrence is
#
#   Pₙ(x) = ((2n - 1) / n) x Pₙ₋₁(x) - ((n - 1) / n) Pₙ₋₂(x).
#
# The implementation evaluates `P₀, P₁, …, P_degree` in one forward sweep. When
# derivatives are requested, it differentiates the same recurrence in lockstep,
# so values and derivatives are produced together without a second pass.
function _legendre_values!(x::T, degree::Int, values::AbstractVector{T}) where {T<:AbstractFloat}
  values[1] = one(T)

  if degree >= 1
    values[2] = x
  end

  @inbounds for n in 2:degree
    inv_n = inv(T(n))
    values[n+1] = T(2n - 1) * inv_n * x * values[n] - T(n - 1) * inv_n * values[n-1]
  end

  return nothing
end

function _legendre_values_and_derivatives!(x::T, degree::Int, values::AbstractVector{T},
                                           derivatives::AbstractVector{T}) where {T<:AbstractFloat}
  values[1] = one(T)
  derivatives[1] = zero(T)

  if degree >= 1
    values[2] = x
    derivatives[2] = one(T)
  end

  # Differentiate the recurrence in lockstep with the value sweep:
  #   Pₙ' = aₙ (Pₙ₋₁ + x Pₙ₋₁') - bₙ Pₙ₋₂'.
  @inbounds for n in 2:degree
    inv_n = inv(T(n))
    first = T(2n - 1) * inv_n
    second = T(n - 1) * inv_n
    values[n+1] = first * x * values[n] - second * values[n-1]
    derivatives[n+1] = first * (values[n] + x * derivatives[n]) - second * derivatives[n-1]
  end

  return nothing
end

function _legendre_derivatives!(x::T, degree::Int,
                                derivatives::AbstractVector{T}) where {T<:AbstractFloat}
  derivatives[1] = zero(T)
  degree == 0 && return nothing

  derivatives[2] = one(T)

  previous_previous_value = one(T)
  previous_value = x
  previous_previous_derivative = zero(T)
  previous_derivative = one(T)

  @inbounds for n in 2:degree
    inv_n = inv(T(n))
    first = T(2n - 1) * inv_n
    second = T(n - 1) * inv_n
    current_value = first * x * previous_value - second * previous_previous_value
    current_derivative = first * (previous_value + x * previous_derivative) -
                         second * previous_previous_derivative
    derivatives[n+1] = current_derivative

    previous_previous_value = previous_value
    previous_value = current_value
    previous_previous_derivative = previous_derivative
    previous_derivative = current_derivative
  end

  return nothing
end

# Integrated family construction.
#
# The integrated modes are built from already computed Legendre data. The key
# implementation detail is the backward overwrite: `ψₙ` depends on `Pₙ` and
# `Pₙ₋₂`, so writing from high degree down to low degree preserves the source
# data long enough even when the caller's output buffers double as temporary
# storage for the standard family.
function _integrated_legendre_values!(x::T, degree::Int,
                                      values::AbstractVector{T}) where {T<:AbstractFloat}
  _legendre_values!(x, degree, values)

  @inbounds for n in degree:-1:2
    values[n+1] = (values[n+1] - values[n-1]) / sqrt(T(4n - 2))
  end

  values[1] = T(0.5) * (one(T) - x)

  if degree >= 1
    values[2] = T(0.5) * (one(T) + x)
  end

  return nothing
end

function _integrated_legendre_derivatives!(x::T, degree::Int,
                                           derivatives::AbstractVector{T}) where {T<:AbstractFloat}
  _legendre_values!(x, degree, derivatives)

  @inbounds for n in degree:-1:2
    derivatives[n+1] = sqrt(T(2n - 1) / T(2)) * derivatives[n]
  end

  derivatives[1] = -T(0.5)

  if degree >= 1
    derivatives[2] = T(0.5)
  end

  return nothing
end

function _integrated_legendre_values_and_derivatives!(x::T, degree::Int, values::AbstractVector{T},
                                                      derivatives::AbstractVector{T}) where {T<:AbstractFloat}
  _legendre_values!(x, degree, values)

  # The derivative of each interior integrated mode is a scaled Legendre
  # polynomial one degree lower. Values are overwritten afterwards from high to
  # low, so this can safely read the standard Legendre table from `values`.
  @inbounds for n in degree:-1:2
    derivatives[n+1] = sqrt(T(2n - 1) / T(2)) * values[n]
  end

  derivatives[1] = -T(0.5)

  if degree >= 1
    derivatives[2] = T(0.5)
  end

  # Interior modes are normalized differences of Legendre polynomials and
  # therefore vanish at both endpoints. The first two modes are then replaced
  # by the endpoint trace functions `(1 ∓ x) / 2`.
  @inbounds for n in degree:-1:2
    values[n+1] = (values[n+1] - values[n-1]) / sqrt(T(4n - 2))
  end

  values[1] = T(0.5) * (one(T) - x)

  if degree >= 1
    values[2] = T(0.5) * (one(T) + x)
  end

  return nothing
end

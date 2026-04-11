# This file defines admissibility policies for tensor-product modal bases.
#
# The polynomial layer in `polynomials.jl` provides the one-dimensional
# finite-element factors used by the rest of the package. For degrees `pₐ ≥ 1`
# those factors are the hierarchical integrated-Legendre modes
#
#   ψ₀, ψ₁, ψ₂, …
#
# on `[-1, 1]`. On DG axes with `pₐ = 0`, the lone admissible one-dimensional
# factor is instead a true cellwise constant mode. This file answers the next
# question: given one degree `pₐ` per axis, which tensor-product tuples of those
# one-dimensional modes should actually be kept on a cell?
#
# In other words, a basis family is not yet a basis evaluation routine or a dof
# layout. It is the combinatorial rule that decides which modal indices
#
#   (m₁, …, m_D)
#
# are admissible for a degree tuple
#
#   (p₁, …, p_D).
#
# Later files build on top of that rule:
# - `continuity.jl` interprets endpoint indices `0` and `1` as trace-carrying
#   factors on CG axes,
# - `space.jl` compiles the surviving local modes into an hp space with the
#   requested continuity policy,
# - `integration.jl` evaluates the resulting basis functions and gradients.
#
# This file therefore sits at the boundary between one-dimensional polynomial
# data and genuine finite-element space construction. The code is organized in
# four small blocks:
# 1. public basis-family types,
# 2. the iterator over admissible mode tuples,
# 3. admissibility and counting logic for each family,
# 4. mixed-radix utilities for traversing the surrounding tensor-product box.

# Public basis-family types.

"""
    AbstractBasisFamily

Abstract supertype for one-dimensional-to-tensor-product basis-selection
policies.

A basis family does not store basis-function values directly. Instead, it
decides which tensor-product mode indices are admissible for a given tuple of
per-axis polynomial degrees. Higher-level space construction then combines this
mode-selection rule with the library's one-dimensional finite-element factors
to build the actual basis on each cell.

For degrees `pₐ ≥ 1`, those factors are the hierarchical integrated-Legendre
functions. On DG axes with `pₐ = 0`, the lone admissible index `mₐ = 0`
represents a cellwise constant factor instead.

Because the underlying one-dimensional family is hierarchical, these modal
selection rules have a direct geometric interpretation. In particular, the
endpoint indices `0` and `1` correspond to trace-carrying factors, while
indices `≥ 2` correspond to factors that vanish at both endpoints. This is why
mode admissibility here has consequences later for continuity and sparsity.
"""
abstract type AbstractBasisFamily end

"""
    FullTensorBasis()

Full tensor-product basis family.

For per-axis degrees `(p₁, …, p_D)`, this family activates every tensor-product
mode

  (m₁, …, m_D),  with  0 ≤ mₐ ≤ pₐ.

The resulting mode count is `∏ₐ (pₐ + 1)`. This is the standard choice when one
wants the complete anisotropic tensor-product polynomial space associated with
the supplied per-axis degrees.

No coupling restriction is imposed between axes: if a high-order mode is
allowed on each axis individually, then their full tensor product is also kept.
"""
struct FullTensorBasis <: AbstractBasisFamily end

"""
    TrunkBasis()

Trunk basis family with reduced high-order tensor coupling.

This family keeps all endpoint modes `0` and `1` on every axis, but limits how
many genuinely higher-order directions may be combined at once. Concretely, for
degrees `(p₁, …, p_D)` and a mode `(m₁, …, m_D)`, define the retained order

  r(m) = ∑ₐ ρ(mₐ),

with

  ρ(mₐ) = 0  for mₐ ∈ {0, 1},
  ρ(mₐ) = mₐ  for mₐ ≥ 2.

This ignores the endpoint modes and counts only interior polynomial content.
`TrunkBasis` activates exactly those modes with `0 ≤ mₐ ≤ pₐ` and

  r(m) ≤ max(p₁, …, p_D).

This produces a dimension-independent reduction of the full tensor product: the
basis remains rich in one strongly resolved direction while avoiding the full
combinatorial growth of high-order cross terms in many dimensions.

The geometric intuition is that trace-carrying endpoint factors are always kept,
while simultaneous coupling of many genuinely interior high-order directions is
restricted. This often preserves the important anisotropic structure of the
space without paying for every mixed high-order tensor interaction.
"""
struct TrunkBasis <: AbstractBasisFamily end

# Iterator representation for admissible mode tuples.

# Internal iterator state for enumerating admissible tensor-product mode tuples.
# `count` stores the precomputed number of active modes so the iterator reports a
# stable length without having to traverse the admissibility rule each time.
struct BasisModes{D,B<:AbstractBasisFamily}
  basis::B
  degrees::NTuple{D,Int}
  box_count::Int
  count::Int
end

"""
    basis_modes(basis, degrees)

Return an iterator over the active tensor-product mode indices of `basis` for
the supplied per-axis degrees.

Each iterated item is an integer tuple `(m₁, …, m_D)` with one entry per axis.
The iterator traverses the admissible modes in mixed-radix order with axis `1`
varying fastest. This ordering is used consistently throughout the library when
building local mode tables.

Conceptually, the iterator walks the full tensor-product box

  0 ≤ mₐ ≤ pₐ

and filters that box through the admissibility rule of the chosen basis family.
The resulting ordering is deterministic and stable, which is important because
later files use it to assign local mode numbers and sparse local-to-global
expansions.

If some axis has degree zero, then the box on that axis contains only the
single index `mₐ = 0`. On DG spaces this later corresponds to the unique
cellwise constant factor on that axis.
"""
function basis_modes(basis::AbstractBasisFamily, degrees::NTuple{D,<:Integer}) where {D}
  checked_degrees = _checked_degrees(degrees)
  box_count = _basis_mode_box_count(checked_degrees)
  return BasisModes{D,typeof(basis)}(basis, checked_degrees, box_count,
                                     _basis_mode_count(basis, checked_degrees))
end

"""
    basis_mode_count(basis, degrees)

Return the number of active tensor-product modes of `basis` for the supplied
per-axis degrees.

For [`FullTensorBasis`](@ref), this is the full tensor-product count
`∏ₐ (pₐ + 1)`. For [`TrunkBasis`](@ref), it is the number of modes satisfying
the retained-order admissibility condition.

The returned count matches `length(collect(basis_modes(basis, degrees)))`, but
specialized implementations may compute it much more efficiently than explicit
enumeration.

Degree-zero axes still contribute exactly one admissible choice, not zero. This
matters for DG spaces because a cell that is piecewise constant in one
coordinate direction still has one valid tensor-product factor there.
"""
function basis_mode_count(basis::AbstractBasisFamily, degrees::NTuple{D,<:Integer}) where {D}
  return _basis_mode_count(basis, _checked_degrees(degrees))
end

# Counting logic for the different basis families.

# Generic fallback count obtained by explicit enumeration of all tensor-product
# modes. Specialized basis families can override this with a more efficient
# closed-form or dynamic-programming count.
function _basis_mode_count(basis::AbstractBasisFamily, degrees::NTuple{D,Int}) where {D}
  count = 0
  total = _basis_mode_box_count(degrees)

  for state in 0:(total-1)
    mode = _mode_from_linear_index(state, degrees)
    is_active_mode(basis, degrees, mode) && (count += 1)
  end

  return count
end

# The full tensor basis activates every mode within the per-axis degree bounds.
function _basis_mode_count(::FullTensorBasis, degrees::NTuple{D,Int}) where {D}
  return _basis_mode_box_count(degrees)
end

# Count trunk-basis modes without enumerating the full tensor product. The array
# `counts[r+1]` stores how many partial modes on the processed axes have
# retained order `r`, where indices `0` and `1` contribute nothing and every
# interior index `m ≥ 2` contributes its full value `m`. Processing one axis
# updates this distribution by adding either an endpoint choice or one of the
# admissible interior indices on that axis.
function _basis_mode_count(::TrunkBasis, degrees::NTuple{D,Int}) where {D}
  maximum_degree = _trunk_retained_order_limit(degrees)
  counts = zeros(Int, maximum_degree + 1)
  counts[1] = 1

  for axis in 1:D
    next = zeros(Int, maximum_degree + 1)

    for retained_order in 0:maximum_degree
      current = counts[retained_order+1]
      current == 0 && continue
      for value in _axis_mode_values(degrees[axis])
        total_order = retained_order + _trunk_mode_order(value)
        total_order <= maximum_degree || continue
        next[total_order+1] += current
      end
    end

    counts = next
  end

  return sum(counts)
end

# Public admissibility predicates.

"""
    is_active_mode(basis, degrees, mode)

Return `true` if the tensor-product mode `mode` is active in `basis` for the
given per-axis degrees.

The tuple `mode` must have the same length as `degrees`. For
[`FullTensorBasis`](@ref), admissibility is simply the box condition
`0 ≤ mₐ ≤ pₐ`. For [`TrunkBasis`](@ref), the mode must additionally satisfy the
retained-order constraint described in the type documentation.
"""
function is_active_mode(::FullTensorBasis, degrees::NTuple{D,<:Integer},
                        mode::NTuple{D,<:Integer}) where {D}
  checked_degrees = _checked_degrees(degrees)
  return _mode_within_degrees(checked_degrees, mode)
end

function is_active_mode(::TrunkBasis, degrees::NTuple{D,<:Integer},
                        mode::NTuple{D,<:Integer}) where {D}
  checked_degrees = _checked_degrees(degrees)
  _mode_within_degrees(checked_degrees, mode) || return false
  max_degree = _trunk_retained_order_limit(checked_degrees)
  return _trunk_retained_order(mode) <= max_degree
end

# Iterator interface.

Base.IteratorSize(::Type{<:BasisModes}) = Base.HasLength()
Base.eltype(::Type{BasisModes{D,B}}) where {D,B} = NTuple{D,Int}
Base.length(iterator::BasisModes) = iterator.count

# Iterate over the mixed-radix tensor-product box and yield only the modes that
# satisfy the family-specific admissibility rule.
function Base.iterate(iterator::BasisModes{D}, state::Int=0) where {D}
  total = iterator.box_count

  while state < total
    mode = _mode_from_linear_index(state, iterator.degrees)
    state += 1
    is_active_mode(iterator.basis, iterator.degrees, mode) && return mode, state
  end

  return nothing
end

# Mixed-radix utilities and small admissibility helpers.

@inline function _basis_mode_box_count(degrees::NTuple{D,Int}) where {D}
  return prod(axis -> degrees[axis] + 1, 1:D; init=1)
end

# Decode a mixed-radix linear index into one tensor-product mode tuple, using
# axis-specific bases `degrees[axis] + 1`.
function _mode_from_linear_index(index::Int, degrees::NTuple{D,Int}) where {D}
  return ntuple(axis -> begin
                  base = degrees[axis] + 1
                  digit = index % base
                  index = fld(index, base)
                  digit
                end, D)
end

# Quick admissibility test for the axiswise box bounds `0 ≤ mₐ ≤ pₐ`.
function _mode_within_degrees(degrees::NTuple{D,Int}, mode::NTuple{D,<:Integer}) where {D}
  @inbounds for axis in 1:D
    value = Int(mode[axis])
    0 <= value <= degrees[axis] || return false
  end

  return true
end

# Trunk-basis retained order ignores the endpoint modes `0` and `1`; all higher
# mode indices contribute their full value.
@inline _trunk_mode_order(value::Int) = value <= 1 ? 0 : value
@inline _trunk_retained_order_limit(degrees::NTuple{D,Int}) where {D} = maximum(degrees; init=0)

function _trunk_retained_order(mode::NTuple{D,<:Integer}) where {D}
  retained_order = 0

  @inbounds for axis in 1:D
    value = Int(mode[axis])
    value >= 0 || throw(ArgumentError("mode[$axis] must be non-negative"))
    retained_order += _trunk_mode_order(value)
  end

  return retained_order
end

# Enumerate the one-dimensional candidate mode indices on one axis before the
# basis-family admissibility rule couples them across axes.
@inline _axis_mode_values(degree::Int) = 0:degree

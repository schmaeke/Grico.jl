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

# Internal iterator states for enumerating admissible tensor-product mode tuples.
# `count` stores the precomputed number of active modes so each iterator reports
# a stable length without having to traverse the admissibility rule each time.
struct FullTensorBasisModes{D}
  degrees::NTuple{D,Int}
  count::Int
end

struct TrunkBasisModes{D}
  degrees::NTuple{D,Int}
  count::Int
  completion_counts::Vector{Vector{Int}}
end

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

The degree tuple must contain at least one axis. Zero-dimensional basis
selection is rejected until a concrete use case fixes the intended semantics.
"""
function basis_modes(::FullTensorBasis, degrees::NTuple{D,<:Integer}) where {D}
  checked_degrees = _checked_basis_degrees(degrees)
  return FullTensorBasisModes{D}(checked_degrees, _checked_basis_mode_box_count(checked_degrees))
end

function basis_modes(::TrunkBasis, degrees::NTuple{D,<:Integer}) where {D}
  checked_degrees = _checked_basis_degrees(degrees)
  completion_counts = _trunk_completion_counts(checked_degrees)
  return TrunkBasisModes{D}(checked_degrees, last(completion_counts)[end], completion_counts)
end

function basis_modes(basis::AbstractBasisFamily, degrees::NTuple{D,<:Integer}) where {D}
  checked_degrees = _checked_basis_degrees(degrees)
  box_count = _checked_basis_mode_box_count(checked_degrees)
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

The degree tuple must contain at least one axis.
"""
function basis_mode_count(::FullTensorBasis, degrees::NTuple{D,<:Integer}) where {D}
  return _checked_basis_mode_box_count(_checked_basis_degrees(degrees))
end

function basis_mode_count(::TrunkBasis, degrees::NTuple{D,<:Integer}) where {D}
  checked_degrees = _checked_basis_degrees(degrees)
  return _trunk_mode_count(checked_degrees)
end

function basis_mode_count(basis::AbstractBasisFamily, degrees::NTuple{D,<:Integer}) where {D}
  return _basis_mode_count(basis, _checked_basis_degrees(degrees))
end

# Counting logic for the different basis families.

# Generic fallback count obtained by explicit enumeration of all tensor-product
# modes. Specialized basis families can override this with a more efficient
# closed-form or dynamic-programming count.
function _basis_mode_count(basis::AbstractBasisFamily, degrees::NTuple{D,Int}) where {D}
  count = 0
  total = _checked_basis_mode_box_count(degrees)

  for state in 0:(total-1)
    mode = _mode_from_linear_index(state, degrees)
    is_active_mode(basis, degrees, mode) && (count += 1)
  end

  return count
end

# The full tensor basis activates every mode within the per-axis degree bounds.
function _basis_mode_count(::FullTensorBasis, degrees::NTuple{D,Int}) where {D}
  return _checked_basis_mode_box_count(degrees)
end

# Count trunk-basis modes without enumerating the full tensor product.
_basis_mode_count(::TrunkBasis, degrees::NTuple{D,Int}) where {D} = _trunk_mode_count(degrees)

# Public admissibility predicates.

"""
    is_active_mode(basis, degrees, mode)

Return `true` if the tensor-product mode `mode` is active in `basis` for the
given per-axis degrees.

The tuple `mode` must have the same length as `degrees`. For
[`FullTensorBasis`](@ref), admissibility is simply the box condition
`0 ≤ mₐ ≤ pₐ`. For [`TrunkBasis`](@ref), the mode must additionally satisfy the
retained-order constraint described in the type documentation.

The degree and mode tuples must contain at least one axis.
"""
function is_active_mode(::FullTensorBasis, degrees::NTuple{D,<:Integer},
                        mode::NTuple{D,<:Integer}) where {D}
  checked_degrees = _checked_basis_degrees(degrees)
  return _mode_within_degrees(checked_degrees, mode)
end

function is_active_mode(::TrunkBasis, degrees::NTuple{D,<:Integer},
                        mode::NTuple{D,<:Integer}) where {D}
  checked_degrees = _checked_basis_degrees(degrees)
  return _is_trunk_mode_active(checked_degrees, mode)
end

# Iterator interface.

Base.IteratorSize(::Type{<:BasisModes}) = Base.HasLength()
Base.eltype(::Type{BasisModes{D,B}}) where {D,B} = NTuple{D,Int}
Base.length(iterator::BasisModes) = iterator.count

Base.IteratorSize(::Type{<:FullTensorBasisModes}) = Base.HasLength()
Base.eltype(::Type{FullTensorBasisModes{D}}) where {D} = NTuple{D,Int}
Base.length(iterator::FullTensorBasisModes) = iterator.count

Base.IteratorSize(::Type{<:TrunkBasisModes}) = Base.HasLength()
Base.eltype(::Type{TrunkBasisModes{D}}) where {D} = NTuple{D,Int}
Base.length(iterator::TrunkBasisModes) = iterator.count

# Full tensor iteration needs no filtering: every mixed-radix state is active.
function Base.iterate(iterator::FullTensorBasisModes, state::Int=0)
  state < iterator.count || return nothing
  return _mode_from_linear_index(state, iterator.degrees), state + 1
end

# Trunk iteration un-ranks active modes directly in the same ordering that a
# filtered mixed-radix tensor-product traversal would produce. This avoids
# scanning rejected high-order cross terms.
function Base.iterate(iterator::TrunkBasisModes{D}, state::Int=0) where {D}
  state < iterator.count || return nothing
  mode = _trunk_mode_from_rank(Val(D), iterator.degrees, iterator.completion_counts,
                               _trunk_retained_order_limit(iterator.degrees), state)
  return mode, state + 1
end

# Iterate over the mixed-radix tensor-product box and yield only the modes that
# satisfy the family-specific admissibility rule. This is the fallback path for
# custom basis families.
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

function _checked_basis_degrees(degrees::NTuple{D,<:Integer}) where {D}
  D >= 1 || throw(ArgumentError("basis dimension must be positive"))
  return _checked_degrees(degrees)
end

function _checked_basis_mode_box_count(degrees::NTuple{D,Int}) where {D}
  total = Int128(1)

  for axis in 1:D
    choices = Int128(degrees[axis]) + 1
    choices <= typemax(Int) || throw(ArgumentError("basis mode count must be Int-representable"))
    total *= choices
    total <= typemax(Int) || throw(ArgumentError("basis mode count must be Int-representable"))
  end

  return Int(total)
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
    value = mode[axis]
    0 <= value <= degrees[axis] || return false
  end

  return true
end

# Trunk-basis retained order ignores the endpoint modes `0` and `1`; all higher
# mode indices contribute their full value.
@inline _trunk_mode_order(value::Int) = value <= 1 ? 0 : value
@inline _trunk_retained_order_limit(degrees::NTuple{D,Int}) where {D} = maximum(degrees; init=0)

# For every `n`, `completion_counts[n+1][r+1]` stores how many assignments on
# axes `1:n` have retained order at most `r`. The trunk iterator uses these
# cumulative counts to un-rank active modes without visiting inactive tensor
# entries.
function _trunk_mode_count(degrees::NTuple{D,Int}) where {D}
  limit = _trunk_retained_order_limit(degrees)
  limit < typemax(Int) || throw(ArgumentError("basis retained-order limit is too large"))
  current = zeros(Int128, limit + 1)
  next = zeros(Int128, limit + 1)
  current[1] = 1

  for axis in 1:D
    _update_trunk_exact_counts!(next, current, degrees[axis], limit)
    current, next = next, current
  end

  return _checked_total_mode_count(current)
end

function _trunk_completion_counts(degrees::NTuple{D,Int}) where {D}
  limit = _trunk_retained_order_limit(degrees)
  limit < typemax(Int) || throw(ArgumentError("basis retained-order limit is too large"))
  exact = zeros(Int128, limit + 1)
  exact[1] = 1
  completion_counts = Vector{Vector{Int}}(undef, D + 1)
  completion_counts[1] = ones(Int, limit + 1)

  for axis in 1:D
    next = zeros(Int128, limit + 1)
    _update_trunk_exact_counts!(next, exact, degrees[axis], limit)
    exact = next
    completion_counts[axis+1] = _cumulative_mode_counts(exact)
  end

  return completion_counts
end

function _update_trunk_exact_counts!(next::Vector{Int128}, current_counts::Vector{Int128},
                                     degree::Int, limit::Int)
  fill!(next, 0)
  endpoint_choices = degree == 0 ? Int128(1) : Int128(2)

  for retained_order in 0:limit
    current = current_counts[retained_order+1]
    current == 0 && continue
    _add_mode_count!(next, retained_order, current * endpoint_choices)

    for order in 2:min(degree, limit-retained_order)
      _add_mode_count!(next, retained_order + order, current)
    end
  end

  return next
end

function _add_mode_count!(counts::Vector{Int128}, retained_order::Int, increment::Int128)
  updated = counts[retained_order+1] + increment
  updated <= typemax(Int) || throw(ArgumentError("basis mode count must be Int-representable"))
  counts[retained_order+1] = updated
  return counts
end

function _checked_total_mode_count(counts::Vector{Int128})
  total = Int128(0)

  for count in counts
    total += count
    total <= typemax(Int) || throw(ArgumentError("basis mode count must be Int-representable"))
  end

  return Int(total)
end

function _cumulative_mode_counts(exact::Vector{Int128})
  cumulative = Vector{Int}(undef, length(exact))
  running = Int128(0)

  for index in eachindex(exact)
    running += exact[index]
    running <= typemax(Int) || throw(ArgumentError("basis mode count must be Int-representable"))
    cumulative[index] = Int(running)
  end

  return cumulative
end

function _trunk_mode_from_rank(::Val{0}, degrees::NTuple{D,Int}, completion_counts, budget::Int,
                               rank::Int) where {D}
  return ()
end

function _trunk_mode_from_rank(::Val{A}, degrees::NTuple{D,Int}, completion_counts, budget::Int,
                               rank::Int) where {A,D}
  for value in _axis_mode_values(degrees[A])
    order = _trunk_mode_order(value)
    order <= budget || continue
    completions = completion_counts[A][budget-order+1]

    if rank < completions
      return (_trunk_mode_from_rank(Val(A - 1), degrees, completion_counts, budget - order,
                                    rank)..., value)
    end

    rank -= completions
  end

  throw(ArgumentError("invalid trunk basis mode rank"))
end

function _is_trunk_mode_active(degrees::NTuple{D,Int}, mode::NTuple{D,<:Integer}) where {D}
  _mode_within_degrees(degrees, mode) || return false
  limit = _trunk_retained_order_limit(degrees)
  retained_order = 0

  @inbounds for axis in 1:D
    value = Int(mode[axis])
    order = _trunk_mode_order(value)
    order <= limit - retained_order || return false
    retained_order += order
  end

  return true
end

# Enumerate the one-dimensional candidate mode indices on one axis before the
# basis-family admissibility rule couples them across axes.
@inline _axis_mode_values(degree::Int) = 0:degree

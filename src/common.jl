# This file collects small internal utilities that are shared across several
# layers of the library. They are intentionally kept low-level: basic argument
# validation and compact sparse-term evaluation utilities. Centralizing them
# here avoids repeating the same defensive checks and tight inner-loop patterns
# throughout topology, assembly, adaptivity, and post-processing code.

# The checked-value helpers below normalize common integer preconditions at the
# API boundary so the rest of the code can work with plain `Int` values and
# assume non-negativity, positivity, or bounds validity without repeating the
# same checks.
@inline function _checked_nonnegative(value::Integer, name::AbstractString)
  0 <= value <= typemax(Int) ||
    throw(ArgumentError("$name must be a non-negative Int-representable integer"))
  return Int(value)
end

@inline function _checked_positive(value::Integer, name::AbstractString)
  1 <= value <= typemax(Int) ||
    throw(ArgumentError("$name must be a positive Int-representable integer"))
  return Int(value)
end

@noinline function _throw_index_error(index::Integer, upper::Integer, name::AbstractString)
  throw(ArgumentError("$name must be an index in 1:$upper; got $index"))
end

@inline function _require_index(index::Integer, upper::Integer, name::AbstractString)
  1 <= index <= upper || _throw_index_error(index, upper, name)
  return Int(index)
end

function _checked_degrees(degrees::NTuple{D,<:Integer}) where {D}
  return ntuple(axis -> _checked_nonnegative(degrees[axis], "degrees[$axis]"), D)
end

# Require that a reusable output buffer is large enough for the requested write.
# This keeps allocation-free `!` routines honest about their size assumptions.
function _require_length(buffer::AbstractVector, length_required::Int, name::AbstractString)
  length(buffer) >= length_required ||
    throw(ArgumentError("$name must have length at least $length_required"))
  return buffer
end

# Many compiled local basis modes expand to a short affine combination of global
# coefficients. When a local mode depends on exactly one global coefficient, we
# store that information separately so evaluation can bypass the generic short
# sum in the hot path.
function _single_term_metadata(term_offsets::Vector{Int}, term_indices::Vector{Int},
                               term_coefficients::Vector{T}) where {T<:AbstractFloat}
  local_dof_count = length(term_offsets) - 1
  single_term_indices = zeros(Int, local_dof_count)
  single_term_coefficients = zeros(T, local_dof_count)

  for local_dof in 1:local_dof_count
    first_term = term_offsets[local_dof]
    last_term = term_offsets[local_dof+1] - 1
    first_term == last_term || continue
    single_term_indices[local_dof] = term_indices[first_term]
    single_term_coefficients[local_dof] = term_coefficients[first_term]
  end

  return single_term_indices, single_term_coefficients
end

# Evaluate one compiled affine mode amplitude from the global coefficient
# vector. The fast path handles the common single-term case with one multiply,
# while the general path accumulates the short sparse combination described by
# the term-offset structure.
@inline function _term_amplitude(term_offsets::Vector{Int}, term_indices::Vector{Int},
                                 term_coefficients::Vector{T}, single_term_indices::Vector{Int},
                                 single_term_coefficients::Vector{T},
                                 coefficients::AbstractVector{T},
                                 local_dof::Int) where {T<:AbstractFloat}
  single_index = single_term_indices[local_dof]
  if single_index != 0
    return single_term_coefficients[local_dof] * coefficients[single_index]
  end

  value = zero(T)
  @inbounds for term_index in term_offsets[local_dof]:(term_offsets[local_dof+1]-1)
    value += term_coefficients[term_index] * coefficients[term_indices[term_index]]
  end
  return value
end

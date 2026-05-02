# This file collects small internal utilities that are shared across several
# layers of the library. They are intentionally kept low-level: basic argument
# validation and compact sparse-term evaluation utilities. Centralizing them
# here avoids repeating the same defensive checks and tight inner-loop patterns
# throughout topology, assembly, adaptivity, and post-processing code.

macro _threaded_loop(expr)
  # Polyester gives the low-overhead path we want on Linux. On Apple silicon,
  # some user-kernel loops still hit Polyester's cfunction closure limitation,
  # so we use Julia's static scheduler as a portable fallback there.
  threaded_expr = @static if Sys.KERNEL === :Darwin && Sys.ARCH === :aarch64
    quote
      let
        if Base.Threads.nthreads() == 1
          $(expr)
        else
          Base.Threads.@threads :static $(expr)
        end
      end
    end
  else
    quote
      @batch per=thread $(expr)
    end
  end

  return esc(threaded_expr)
end

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

function _require_square_matrix(matrix_data::AbstractMatrix, name::AbstractString)
  size(matrix_data, 1) == size(matrix_data, 2) || throw(ArgumentError("$name must be square"))
  return size(matrix_data, 1)
end

# Compact dense factorizations used for FE-local blocks. These scalar-loop
# kernels deliberately avoid BLAS/LAPACK so they can be called from threaded
# assembly regions without nested native threading.
function _dense_lu_factor!(matrix_data::AbstractMatrix{T},
                           pivots::AbstractVector{Int}) where {T<:AbstractFloat}
  n = _require_square_matrix(matrix_data, "dense LU matrix")
  _require_length(pivots, n, "pivots")

  for column in 1:n
    pivot_row = column
    pivot_magnitude = abs(@inbounds matrix_data[column, column])

    for row in (column+1):n
      magnitude = abs(@inbounds matrix_data[row, column])

      if magnitude > pivot_magnitude
        pivot_magnitude = magnitude
        pivot_row = row
      end
    end

    iszero(pivot_magnitude) && throw(SingularException(column))
    pivots[column] = pivot_row

    if pivot_row != column
      for swap_column in 1:n
        @inbounds matrix_data[column, swap_column], matrix_data[pivot_row, swap_column] = matrix_data[pivot_row,
                                                                                                      swap_column],
                                                                                          matrix_data[column,
                                                                                                      swap_column]
      end
    end

    inverse_pivot = inv(@inbounds matrix_data[column, column])

    for row in (column+1):n
      @inbounds matrix_data[row, column] *= inverse_pivot
      multiplier = @inbounds matrix_data[row, column]

      for trailing_column in (column+1):n
        @inbounds matrix_data[row, trailing_column] -= multiplier *
                                                       matrix_data[column, trailing_column]
      end
    end
  end

  return matrix_data
end

function _apply_lu_pivots!(rhs_data::AbstractVector{T}, pivots::AbstractVector{Int},
                           n::Int=length(rhs_data)) where {T<:AbstractFloat}
  _require_length(rhs_data, n, "rhs")
  _require_length(pivots, n, "pivots")

  for row in 1:n
    pivot_row = @inbounds pivots[row]
    pivot_row == row && continue
    @inbounds rhs_data[row], rhs_data[pivot_row] = rhs_data[pivot_row], rhs_data[row]
  end

  return rhs_data
end

function _apply_lu_pivots!(rhs_data::AbstractMatrix{T}, pivots::AbstractVector{Int},
                           n::Int=size(rhs_data, 1)) where {T<:AbstractFloat}
  size(rhs_data, 1) >= n || throw(ArgumentError("rhs row count must match dense LU factor"))
  _require_length(pivots, n, "pivots")

  for row in 1:n
    pivot_row = @inbounds pivots[row]
    pivot_row == row && continue

    for column in axes(rhs_data, 2)
      @inbounds rhs_data[row, column], rhs_data[pivot_row, column] = rhs_data[pivot_row, column],
                                                                     rhs_data[row, column]
    end
  end

  return rhs_data
end

function _dense_lu_solve!(factor_data::AbstractMatrix{T}, pivots::AbstractVector{Int},
                          rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  n = _require_square_matrix(factor_data, "dense LU factor")
  _require_length(rhs_data, n, "rhs")
  _require_length(pivots, n, "pivots")
  _apply_lu_pivots!(rhs_data, pivots, n)

  for row in 1:n
    value = @inbounds rhs_data[row]

    for column in 1:(row-1)
      @inbounds value -= factor_data[row, column] * rhs_data[column]
    end

    @inbounds rhs_data[row] = value
  end

  for row in n:-1:1
    value = @inbounds rhs_data[row]

    for column in (row+1):n
      @inbounds value -= factor_data[row, column] * rhs_data[column]
    end

    @inbounds rhs_data[row] = value / factor_data[row, row]
  end

  return rhs_data
end

function _dense_lu_solve!(factor_data::AbstractMatrix{T}, pivots::AbstractVector{Int},
                          rhs_data::AbstractMatrix{T}) where {T<:AbstractFloat}
  n = _require_square_matrix(factor_data, "dense LU factor")
  size(rhs_data, 1) == n || throw(ArgumentError("rhs row count must match dense LU factor"))
  _require_length(pivots, n, "pivots")
  _apply_lu_pivots!(rhs_data, pivots, n)

  for right_hand_side in axes(rhs_data, 2)
    for row in 1:n
      value = @inbounds rhs_data[row, right_hand_side]

      for column in 1:(row-1)
        @inbounds value -= factor_data[row, column] * rhs_data[column, right_hand_side]
      end

      @inbounds rhs_data[row, right_hand_side] = value
    end

    for row in n:-1:1
      value = @inbounds rhs_data[row, right_hand_side]

      for column in (row+1):n
        @inbounds value -= factor_data[row, column] * rhs_data[column, right_hand_side]
      end

      @inbounds rhs_data[row, right_hand_side] = value / factor_data[row, row]
    end
  end

  return rhs_data
end

function _dense_cholesky_factor!(matrix_data::AbstractMatrix{T}) where {T<:AbstractFloat}
  n = _require_square_matrix(matrix_data, "dense Cholesky matrix")

  for column in 1:n
    diagonal = @inbounds matrix_data[column, column]

    for previous in 1:(column-1)
      value = @inbounds matrix_data[column, previous]
      diagonal -= value * value
    end

    diagonal > zero(T) || throw(PosDefException(column))
    factor = sqrt(diagonal)
    @inbounds matrix_data[column, column] = factor
    inverse_factor = inv(factor)

    for row in (column+1):n
      value = @inbounds matrix_data[row, column]

      for previous in 1:(column-1)
        @inbounds value -= matrix_data[row, previous] * matrix_data[column, previous]
      end

      @inbounds matrix_data[row, column] = value * inverse_factor
    end
  end

  return matrix_data
end

function _dense_cholesky_solve!(factor_data::AbstractMatrix{T},
                                rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  n = _require_square_matrix(factor_data, "dense Cholesky factor")
  _require_length(rhs_data, n, "rhs")

  for row in 1:n
    value = @inbounds rhs_data[row]

    for column in 1:(row-1)
      @inbounds value -= factor_data[row, column] * rhs_data[column]
    end

    @inbounds rhs_data[row] = value / factor_data[row, row]
  end

  for row in n:-1:1
    value = @inbounds rhs_data[row]

    for column in (row+1):n
      @inbounds value -= factor_data[column, row] * rhs_data[column]
    end

    @inbounds rhs_data[row] = value / factor_data[row, row]
  end

  return rhs_data
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

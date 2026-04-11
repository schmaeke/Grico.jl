# This file collects small internal utilities that are shared across several
# layers of the library. They are intentionally kept low-level: thread/chunk
# scheduling helpers, basic argument validation, and compact sparse-term
# evaluation utilities. Centralizing them here avoids repeating the same
# defensive checks and tight inner-loop patterns throughout topology, assembly,
# adaptivity, and post-processing code.

# These three globals implement a reentrant guard that temporarily forces BLAS
# to run single-threaded while Grico.jl is already parallelizing over Julia
# threads. Without this guard, nested parallelism would oversubscribe the CPU:
# each Julia worker could spawn several BLAS threads again inside dense linear
# algebra kernels, which usually hurts performance rather than helping.
const _INTERNAL_BLAS_GUARD_LOCK = ReentrantLock()
const _INTERNAL_BLAS_GUARD_DEPTH = Ref(0)
const _INTERNAL_BLAS_SAVED_THREADS = Ref(1)
const _INTERNAL_BLAS_CALL_LOCK = ReentrantLock()

# Small lock wrapper used by the BLAS guards below. Keeping the lock/unlock
# pattern in one place makes the guard code easier to read and harder to get
# subtly wrong when nested guard logic changes.
function _with_lock(f::F, lock_object) where {F}
  lock(lock_object)

  try
    return f()
  finally
    unlock(lock_object)
  end
end

# Execute `f()` with BLAS thread count reduced to one whenever the surrounding
# Grico.jl algorithm is already running multi-threaded. The guard is reference
# counted so nested calls remain safe and restore the original BLAS thread count
# only when the outermost guarded region exits.
function _with_internal_blas_threads(f::F) where {F}
  if Threads.nthreads() == 1
    return f()
  end

  _with_lock(_INTERNAL_BLAS_GUARD_LOCK) do
    if _INTERNAL_BLAS_GUARD_DEPTH[] == 0
      _INTERNAL_BLAS_SAVED_THREADS[] = BLAS.get_num_threads()
      _INTERNAL_BLAS_SAVED_THREADS[] == 1 || BLAS.set_num_threads(1)
    end

    _INTERNAL_BLAS_GUARD_DEPTH[] += 1
  end

  try
    return f()
  finally
    _with_lock(_INTERNAL_BLAS_GUARD_LOCK) do
      _INTERNAL_BLAS_GUARD_DEPTH[] -= 1

      if _INTERNAL_BLAS_GUARD_DEPTH[] == 0
        saved = _INTERNAL_BLAS_SAVED_THREADS[]
        saved == 1 || BLAS.set_num_threads(saved)
      end
    end
  end
end

# Some OpenBLAS builds also limit how many distinct caller threads may enter the
# library concurrently, even when BLAS itself is configured to use one worker
# thread. Serializing the actual BLAS/LAPACK entry keeps Grico's outer Julia-level
# parallelism while avoiding allocator corruption on such builds.
function _with_serialized_blas(f::F) where {F}
  if Threads.nthreads() == 1
    return _with_internal_blas_threads(f)
  end

  return _with_lock(_INTERNAL_BLAS_CALL_LOCK) do
    return _with_internal_blas_threads(f)
  end
end

# Run `f(first, last)` on chunks of `1:item_count`, using an atomic next-item
# counter so the same helper works both for balanced and mildly irregular work.
# The chunk size is chosen dynamically to trade scheduling overhead against load
# balancing; the sequential path still goes through the same chunk logic so both
# code paths follow the same semantics.
function _run_chunks!(f::F, item_count::Int) where {F}
  item_count == 0 && return nothing
  worker_count = min(Threads.nthreads(), item_count)
  chunk_size = _dynamic_chunk_size(item_count, worker_count)
  next_item = Threads.Atomic{Int}(1)

  if worker_count == 1
    return _run_chunk_loop!(f, item_count, chunk_size, next_item)
  end

  @sync for _ in 1:worker_count
    Threads.@spawn _run_chunk_loop!(f, item_count, chunk_size, next_item)
  end

  return nothing
end

function _run_chunk_loop!(f::F, item_count::Int, chunk_size::Int,
                          next_item::Threads.Atomic{Int}) where {F}
  while true
    first_item = Threads.atomic_add!(next_item, chunk_size)
    first_item > item_count && return nothing
    last_item = min(item_count, first_item + chunk_size - 1)
    f(first_item, last_item)
  end
end

# Variant of the chunk runner that gives each spawned worker task its own
# scratch object. This is the right tool when the callback needs mutable local
# buffers, because the scratch ownership follows the spawned task directly
# instead of depending on `Threads.threadid()` remaining stable.
function _run_chunks_with_scratch!(f::F, scratch::AbstractVector, item_count::Int) where {F}
  item_count == 0 && return nothing
  worker_count = min(length(scratch), Threads.nthreads(), item_count)
  chunk_size = _dynamic_chunk_size(item_count, worker_count)
  next_item = Threads.Atomic{Int}(1)

  if worker_count == 1
    return _run_chunk_loop_with_scratch!(f, scratch[1], item_count, chunk_size, next_item)
  end

  @sync for worker in 1:worker_count
    Threads.@spawn _run_chunk_loop_with_scratch!(f, scratch[worker], item_count, chunk_size,
                                                 next_item)
  end

  return nothing
end

function _run_chunk_loop_with_scratch!(f::F, scratch, item_count::Int, chunk_size::Int,
                                       next_item::Threads.Atomic{Int}) where {F}
  while true
    first_item = Threads.atomic_add!(next_item, chunk_size)
    first_item > item_count && return nothing
    last_item = min(item_count, first_item + chunk_size - 1)
    f(scratch, first_item, last_item)
  end
end

_dynamic_chunk_size(item_count::Int, worker_count::Int) = max(1, cld(item_count, 8 * worker_count))

# The checked-value helpers below normalize common integer preconditions at the
# API boundary so the rest of the code can work with plain `Int` values and
# assume non-negativity, positivity, or bounds validity without repeating the
# same checks.
@inline function _checked_nonnegative(value::Integer, name::AbstractString)
  checked = Int(value)
  checked >= 0 || throw(ArgumentError("$name must be non-negative"))
  return checked
end

@inline function _checked_positive(value::Integer, name::AbstractString)
  checked = Int(value)
  checked >= 1 || throw(ArgumentError("$name must be positive"))
  return checked
end

@inline function _checked_index(index::Integer, upper::Integer, name::AbstractString)
  checked = Int(index)
  1 <= checked <= upper || throw(BoundsError(Base.OneTo(upper), checked))
  return checked
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

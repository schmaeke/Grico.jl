#!/usr/bin/env julia

const BENCHMARK_PROJECT = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_PROJECT, ".."))
pushfirst!(LOAD_PATH, BENCHMARK_PROJECT)
pushfirst!(LOAD_PATH, REPO_ROOT)

using LinearAlgebra
using Profile
using Printf

using Grico

include(joinpath(@__DIR__, "shared_memory_phase0_cases.jl"))
using .SharedMemoryPhase0Cases

const DEFAULT_PROFILE = "cpu"
const DEFAULT_REPETITIONS = 5
const DEFAULT_ALLOCATION_SAMPLE_RATE = 0.05

function _parse_args(args)
  parsed = Dict{String,String}()

  for arg in args
    startswith(arg, "--") || throw(ArgumentError("unsupported argument `$arg`"))
    key, value = occursin("=", arg) ? split(arg[3:end], "="; limit=2) : (arg[3:end], "true")
    parsed[key] = value
  end

  return parsed
end

function _required(parsed, key)
  haskey(parsed, key) || throw(ArgumentError("missing required argument `--$key=`"))
  return parsed[key]
end

function _find_operation(prepared::PreparedCase, operation_id::AbstractString)
  for operation in prepared.operations
    operation.id == operation_id && return operation
  end

  supported = join((operation.id for operation in prepared.operations), ", ")
  throw(ArgumentError("unknown operation `$operation_id` for case `$(prepared.id)`; supported operations: $supported"))
end

function _header(io, prepared::PreparedCase, operation::OperationSpec, repetitions::Int,
                 profile_kind::AbstractString, sample_rate::Float64)
  println(io, "# Shared-Memory Profile")
  println(io)
  println(io, @sprintf("case: %s", prepared.id))
  println(io, @sprintf("operation: %s", operation.id))
  println(io, @sprintf("profile: %s", profile_kind))
  println(io, @sprintf("threads: %d", Threads.nthreads()))
  println(io, @sprintf("repetitions: %d", repetitions))
  println(io, @sprintf("blas_threads: %d", BLAS.get_num_threads()))
  profile_kind == "alloc" && println(io, @sprintf("allocation_sample_rate: %.4f", sample_rate))
  println(io)
end

function _cpu_profile(io, operation::OperationSpec, repetitions::Int)
  Profile.clear()
  GC.gc()
  operation.setup()
  operation.run()
  Profile.clear()

  Profile.@profile for _ in 1:repetitions
    operation.setup()
    operation.run()
  end

  Profile.print(io; format=:tree, sortedby=:count, maxdepth=24)
  return nothing
end

function _alloc_profile(io, operation::OperationSpec, repetitions::Int, sample_rate::Float64)
  Profile.Allocs.clear()
  GC.gc()
  operation.setup()
  operation.run()
  Profile.Allocs.clear()

  Profile.Allocs.@profile sample_rate=sample_rate begin
    for _ in 1:repetitions
      operation.setup()
      operation.run()
    end
  end

  Profile.Allocs.print(io, Profile.Allocs.fetch())
  return nothing
end

function main(args=ARGS)
  parsed = _parse_args(args)
  case_id = _required(parsed, "case")
  operation_id = _required(parsed, "operation")
  profile_kind = get(parsed, "profile", DEFAULT_PROFILE)
  repetitions = parse(Int, get(parsed, "repetitions", string(DEFAULT_REPETITIONS)))
  sample_rate = parse(Float64, get(parsed, "sample-rate", string(DEFAULT_ALLOCATION_SAMPLE_RATE)))
  output_path = get(parsed, "output",
                    joinpath(BENCHMARK_PROJECT,
                             "profile_$(case_id)_$(operation_id)_$(profile_kind).txt"))
  BLAS.set_num_threads(1)
  prepared = build_phase0_case(case_id)
  operation = _find_operation(prepared, operation_id)
  mkpath(dirname(output_path))

  open(output_path, "w") do io
    _header(io, prepared, operation, repetitions, profile_kind, sample_rate)

    if profile_kind == "cpu"
      _cpu_profile(io, operation, repetitions)
    elseif profile_kind == "alloc"
      _alloc_profile(io, operation, repetitions, sample_rate)
    else
      throw(ArgumentError("profile must be `cpu` or `alloc`"))
    end
  end

  println("wrote $(profile_kind) profile to $output_path")
  return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  main()
end

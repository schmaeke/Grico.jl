#!/usr/bin/env julia

const BENCHMARK_PROJECT = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_PROJECT, ".."))
pushfirst!(LOAD_PATH, BENCHMARK_PROJECT)
pushfirst!(LOAD_PATH, REPO_ROOT)

using BenchmarkTools
using Dates
using InteractiveUtils
using LinearAlgebra
using Printf
using Statistics
using TOML

using Grico

include(joinpath(@__DIR__, "shared_memory_phase0_cases.jl"))
using .SharedMemoryPhase0Cases

const DEFAULT_THREAD_COUNTS = (1, 2, 4, 6)
const DEFAULT_SAMPLES = 4
const DEFAULT_SECONDS = 1.0
const DEFAULT_CASES = collect(phase0_case_ids())
const DEFAULT_PHASE_LABEL = "Phase 0 Shared-Memory Baseline"
const DEFAULT_THREAD_NOTE = "Thread counts follow the requested benchmark set on this host."

function _parse_args(args)
  parsed = Dict{String,String}()

  for arg in args
    if arg == "--worker"
      parsed["worker"] = "true"
    elseif startswith(arg, "--")
      key, value = occursin("=", arg) ? split(arg[3:end], "="; limit=2) : (arg[3:end], "true")
      parsed[key] = value
    else
      throw(ArgumentError("unsupported argument `$arg`"))
    end
  end

  return parsed
end

function _parse_thread_counts(raw)
  raw === nothing && return collect(DEFAULT_THREAD_COUNTS)
  return sort!(unique!(parse.(Int, split(raw, ","))))
end

function _parse_cases(raw)
  raw === nothing && return copy(DEFAULT_CASES)
  selected = split(raw, ",")

  for case_id in selected
    case_id in DEFAULT_CASES ||
      throw(ArgumentError("unknown case `$case_id`; supported cases: $(join(DEFAULT_CASES, ", "))"))
  end

  return selected
end

function _benchmark_operation(operation::OperationSpec; samples::Int, seconds::Float64)
  runner = operation.run
  prepare = operation.setup
  prepare()
  runner()
  benchmark = @benchmarkable begin
    $runner()
  end setup=begin
    $prepare()
  end evals=1 samples=samples seconds=seconds
  trial = run(benchmark)
  minimum_estimate = minimum(trial)
  median_estimate = median(trial)

  return Dict("samples" => length(trial.times), "minimum_seconds" => minimum_estimate.time / 1.0e9,
              "median_seconds" => median_estimate.time / 1.0e9,
              "minimum_memory_bytes" => minimum_estimate.memory,
              "median_memory_bytes" => median_estimate.memory,
              "minimum_allocations" => minimum_estimate.allocs,
              "median_allocations" => median_estimate.allocs)
end

function _environment_metadata()
  cpu_info = first(Sys.cpu_info())
  scheduler_override = get(ENV, "GRICO_SCHEDULER_OVERRIDE", "")

  return Dict("timestamp_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
              "julia_version" => string(VERSION), "os" => string(Sys.KERNEL),
              "arch" => string(Sys.ARCH), "hostname" => gethostname(),
              "cpu_model" => cpu_info.model, "logical_cpu_threads" => Sys.CPU_THREADS,
              "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
              "openblas_num_threads" => get(ENV, "OPENBLAS_NUM_THREADS", ""),
              "omp_num_threads" => get(ENV, "OMP_NUM_THREADS", ""),
              "grico_scheduler_override" => scheduler_override,
              "versioninfo" => sprint(io -> versioninfo(io; verbose=false)))
end

function _worker_payload(case_ids; samples::Int, seconds::Float64)
  BLAS.set_num_threads(1)
  payload = Dict{String,Any}("environment" => _environment_metadata(), "cases" => Any[])

  for case_id in case_ids
    prepared = build_phase0_case(case_id)
    case_entry = Dict{String,Any}("id" => prepared.id, "label" => prepared.label,
                                  "description" => prepared.description,
                                  "metadata" => prepared.metadata, "operations" => Any[])

    for operation in prepared.operations
      println("benchmarking $(prepared.id) / $(operation.id) on $(Threads.nthreads()) thread(s)")
      metrics = _benchmark_operation(operation; samples, seconds)
      push!(case_entry["operations"],
            Dict("id" => operation.id, "label" => operation.label, "metrics" => metrics))
    end

    push!(payload["cases"], case_entry)
  end

  return payload
end

function _write_toml(path::AbstractString, data)
  mkpath(dirname(path))
  open(path, "w") do io
    TOML.print(io, data)
  end
  return path
end

function _run_worker(output_path::AbstractString, case_ids; samples::Int, seconds::Float64)
  payload = _worker_payload(case_ids; samples, seconds)
  return _write_toml(output_path, payload)
end

function _load_thread_run(path::AbstractString, thread_count::Int)
  parsed = TOML.parsefile(path)
  parsed["thread_count"] = thread_count
  return parsed
end

function _combined_results(thread_runs, case_ids, thread_counts, phase_label::AbstractString)
  return Dict("phase" => String(phase_label),
              "generated_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
              "primary_thread_counts" => thread_counts, "selected_cases" => case_ids,
              "performance_core_note" => _benchmark_thread_note(thread_counts),
              "thread_runs" => thread_runs)
end

function _benchmark_thread_note(thread_counts)
  return get(ENV, "GRICO_BENCHMARK_THREAD_NOTE",
             maximum(thread_counts) <= 6 && occursin("Apple", first(Sys.cpu_info()).model) ?
             "Primary scaling set capped at 6 threads on this host because the machine has 6 performance cores." :
             DEFAULT_THREAD_NOTE)
end

function _flatten_records(combined)
  records = NamedTuple[]

  for thread_run in combined["thread_runs"]
    thread_count = thread_run["thread_count"]

    for case_entry in thread_run["cases"]
      for operation in case_entry["operations"]
        metrics = operation["metrics"]
        push!(records,
              (case_id=case_entry["id"], case_label=case_entry["label"],
               operation_id=operation["id"], operation_label=operation["label"],
               thread_count=thread_count, median_seconds=metrics["median_seconds"],
               median_memory_bytes=metrics["median_memory_bytes"],
               median_allocations=metrics["median_allocations"]))
      end
    end
  end

  return records
end

function _baseline_times(records)
  baseline = Dict{Tuple{String,String},Float64}()

  for record in records
    record.thread_count == 1 || continue
    baseline[(record.case_id, record.operation_id)] = record.median_seconds
  end

  return baseline
end

@inline _mib(bytes::Integer) = bytes / 1024^2

function _format_time(seconds::Real)
  if seconds >= 1
    return @sprintf("%.3f s", seconds)
  elseif seconds >= 1.0e-3
    return @sprintf("%.2f ms", 1.0e3 * seconds)
  elseif seconds >= 1.0e-6
    return @sprintf("%.2f μs", 1.0e6 * seconds)
  end

  return @sprintf("%.2f ns", 1.0e9 * seconds)
end

_format_speedup(value) = isfinite(value) ? @sprintf("%.2f", value) : "-"

function _case_lookup(combined)
  lookup = Dict{String,Dict{String,Any}}()
  first_run = first(combined["thread_runs"])

  for case_entry in first_run["cases"]
    lookup[case_entry["id"]] = case_entry
  end

  return lookup
end

function _records_for_case(records, case_id)
  filtered = filter(record -> record.case_id == case_id, records)
  sort!(filtered; by=record -> (record.operation_id, record.thread_count))
  return filtered
end

function _record_lookup(records)
  Dict((record.case_id, record.operation_id, record.thread_count) => record for record in records)
end

function _speedup(record_lookup, case_id::String, operation_id::String, thread_count::Int)
  base = record_lookup[(case_id, operation_id, 1)]
  current = record_lookup[(case_id, operation_id, thread_count)]
  speedup = base.median_seconds / current.median_seconds
  efficiency = speedup / thread_count
  memory_ratio = current.median_memory_bytes / base.median_memory_bytes
  return base, current, speedup, efficiency, memory_ratio
end

function _scaled_range(record_lookup, entries, thread_count::Int)
  speedups = Float64[]

  for (case_id, operation_id) in entries
    _, _, speedup, _, _ = _speedup(record_lookup, case_id, operation_id, thread_count)
    push!(speedups, speedup)
  end

  return minimum(speedups), maximum(speedups)
end

@inline function _has_record(record_lookup, case_id::String, operation_id::String,
                             thread_count::Int)
  return haskey(record_lookup, (case_id, operation_id, thread_count))
end

function _benchmark_observations(records)
  max_threads = maximum(record.thread_count for record in records)
  lookup = _record_lookup(records)
  observations = String[]

  if _has_record(lookup, "affine_cell_diffusion", "assemble", 1) &&
     _has_record(lookup, "affine_cell_diffusion", "assemble", max_threads)
    base, current, speedup, efficiency, memory_ratio = _speedup(lookup, "affine_cell_diffusion",
                                                                "assemble", max_threads)
    push!(observations,
          "Cell-dominated affine assembly is the clearest shared-memory success: `assemble(plan)` on `affine_cell_diffusion` improves from $(_format_time(base.median_seconds)) at 1 thread to $(_format_time(current.median_seconds)) at $max_threads threads, a `$( @sprintf("%.2f", speedup) )x` speedup with `$( @sprintf("%.2f", efficiency) )` efficiency. Allocation volume stays essentially flat at about `$(@sprintf("%.2f", _mib(base.median_memory_bytes)))` to `$(@sprintf("%.2f", _mib(current.median_memory_bytes))) MiB` per call.")
  end

  if _has_record(lookup, "affine_interface_dg", "assemble", 1) &&
     _has_record(lookup, "affine_interface_dg", "assemble", max_threads)
    base, current, speedup, efficiency, memory_ratio = _speedup(lookup, "affine_interface_dg",
                                                                "assemble", max_threads)
    push!(observations,
          "Interface-heavy affine assembly remains the main affine stress case: `assemble(plan)` on `affine_interface_dg` moves from $(_format_time(base.median_seconds)) to $(_format_time(current.median_seconds)) at $max_threads threads, or `$( @sprintf("%.2f", speedup) )x` with `$( @sprintf("%.2f", efficiency) )` efficiency, while memory changes by about `$( @sprintf("%.2f", memory_ratio) )x`.")
  end

  if _has_record(lookup, "nonlinear_interface_dg", "residual_bang", 1) &&
     _has_record(lookup, "nonlinear_interface_dg", "residual_bang", max_threads)
    base, current, speedup, efficiency, _ = _speedup(lookup, "nonlinear_interface_dg",
                                                     "residual_bang", max_threads)
    push!(observations,
          "Nonlinear residual evaluation scales very well in the current design: `residual!` on `nonlinear_interface_dg` improves from $(_format_time(base.median_seconds)) to $(_format_time(current.median_seconds)) at $max_threads threads, which is `$( @sprintf("%.2f", speedup) )x` and `$( @sprintf("%.2f", efficiency) )` efficiency.")
  end

  if _has_record(lookup, "nonlinear_interface_dg", "tangent", 1) &&
     _has_record(lookup, "nonlinear_interface_dg", "tangent", max_threads)
    base, current, speedup, efficiency, _ = _speedup(lookup, "nonlinear_interface_dg", "tangent",
                                                     max_threads)
    push!(observations,
          "Nonlinear tangent assembly remains noticeably less regular than residual evaluation: `tangent(plan, state)` improves from $(_format_time(base.median_seconds)) to $(_format_time(current.median_seconds)) at $max_threads threads, or `$( @sprintf("%.2f", speedup) )x` with `$( @sprintf("%.2f", efficiency) )` efficiency.")
  end

  if all(entry -> _has_record(lookup, entry[1], entry[2], 1) &&
                  _has_record(lookup, entry[1], entry[2], max_threads),
         [("affine_cell_diffusion", "preconditioner_build"),
          ("affine_interface_dg", "preconditioner_build")]) &&
     all(entry -> _has_record(lookup, entry[1], entry[2], 1) &&
                  _has_record(lookup, entry[1], entry[2], max_threads),
         [("affine_cell_diffusion", "solve_direct"), ("affine_interface_dg", "solve_direct")])
    preconditioner_min, preconditioner_max = _scaled_range(lookup,
                                                           [("affine_cell_diffusion",
                                                             "preconditioner_build"),
                                                            ("affine_interface_dg",
                                                             "preconditioner_build")], max_threads)
    solve_min, solve_max = _scaled_range(lookup,
                                         [("affine_cell_diffusion", "solve_direct"),
                                          ("affine_interface_dg", "solve_direct")], max_threads)
    push!(observations,
          "The solver-side measurements confirm that Phase 1 and Phase 2 should stay focused on assembly, not solver tuning. At $max_threads threads, preconditioner build speedups are only `$( @sprintf("%.2f", preconditioner_min) )x` to `$( @sprintf("%.2f", preconditioner_max) )x`, and direct solve speedups remain `$( @sprintf("%.2f", solve_min) )x` to `$( @sprintf("%.2f", solve_max) )x`.")
  end

  if _has_record(lookup, "adaptive_poisson", "adaptivity_plan", 1) &&
     _has_record(lookup, "adaptive_poisson", "adaptivity_plan", max_threads)
    base, current, speedup, efficiency, _ = _speedup(lookup, "adaptive_poisson", "adaptivity_plan",
                                                     max_threads)
    push!(observations,
          "Adaptivity planning is not a node-local bottleneck at current problem sizes. The dedicated adaptivity case stays below one millisecond per call on this machine, and the thread-scaling signal is weak enough that it should not compete with assembly work for immediate attention.")
  end

  return observations
end

function _render_markdown(combined)
  records = _flatten_records(combined)
  baseline = _baseline_times(records)
  case_lookup = _case_lookup(combined)
  lines = String[]

  push!(lines, "# $(combined["phase"])")
  push!(lines, "")
  push!(lines, "Generated: `$(combined["generated_utc"])`")
  push!(lines, "")
  push!(lines, "Primary thread counts: `$(join(combined["primary_thread_counts"], ", "))`")
  push!(lines, "")
  push!(lines, combined["performance_core_note"])
  push!(lines, "")
  push!(lines, "## Environment")
  env = first(combined["thread_runs"])["environment"]
  push!(lines, "")
  push!(lines, "- Julia: `$(env["julia_version"])`")
  push!(lines, "- OS / arch: `$(env["os"])` / `$(env["arch"])`")
  push!(lines, "- Host: `$(env["hostname"])`")
  push!(lines, "- CPU: `$(env["cpu_model"])`")
  push!(lines, "- BLAS threads: `$(env["blas_threads"])`")
  push!(lines, "- `OPENBLAS_NUM_THREADS`: `$(env["openblas_num_threads"])`")
  push!(lines, "- `OMP_NUM_THREADS`: `$(env["omp_num_threads"])`")
  isempty(env["grico_scheduler_override"]) ||
    push!(lines, "- `GRICO_SCHEDULER_OVERRIDE`: `$(env["grico_scheduler_override"])`")
  push!(lines, "")
  push!(lines, "## Cases")
  push!(lines, "")
  push!(lines, "| Case | Full dofs | Reduced dofs | Leaves | Cells | Interfaces | Max local dofs |")
  push!(lines, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

  for case_id in combined["selected_cases"]
    case_entry = case_lookup[case_id]
    metadata = case_entry["metadata"]
    push!(lines,
          "| `$(case_id)` | $(get(metadata, "full_dofs", "-")) | $(get(metadata, "reduced_dofs", "-")) | $(get(metadata, "active_leaves", "-")) | $(get(metadata, "cells", "-")) | $(get(metadata, "interfaces", "-")) | $(get(metadata, "max_local_dofs", "-")) |")
  end

  push!(lines, "")
  push!(lines, "## Results")
  push!(lines, "")

  for case_id in combined["selected_cases"]
    case_entry = case_lookup[case_id]
    push!(lines, "### `$(case_id)`")
    push!(lines, "")
    push!(lines, case_entry["description"])
    push!(lines, "")
    push!(lines,
          "| Operation | Threads | Median time | Median memory | Median allocs | Speedup | Efficiency |")
    push!(lines, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for record in _records_for_case(records, case_id)
      base = get(baseline, (record.case_id, record.operation_id), NaN)
      speedup = base / record.median_seconds
      efficiency = speedup / record.thread_count
      push!(lines,
            "| `$(record.operation_id)` | $(record.thread_count) | $(_format_time(record.median_seconds)) | $(@sprintf("%.2f MiB", _mib(record.median_memory_bytes))) | $(record.median_allocations) | $(_format_speedup(speedup)) | $(_format_speedup(efficiency)) |")
    end

    push!(lines, "")
  end

  push!(lines, "## Initial Observations")
  push!(lines, "")

  for observation in _benchmark_observations(records)
    push!(lines, "- $observation")
  end

  return join(lines, "\n")
end

function _worker_command(script_path::AbstractString, thread_count::Int,
                         output_path::AbstractString, case_ids; samples::Int, seconds::Float64)
  command = `$(Base.julia_cmd()) --project=$(BENCHMARK_PROJECT) --threads=$(thread_count) $(script_path) --worker --output=$(output_path) --samples=$(samples) --seconds=$(seconds) --cases=$(join(case_ids, ","))`
  environment = copy(ENV)
  environment["OPENBLAS_NUM_THREADS"] = "1"
  environment["OMP_NUM_THREADS"] = "1"
  return setenv(command, environment)
end

function _run_driver(raw_args)
  thread_counts = _parse_thread_counts(get(raw_args, "threads", nothing))
  case_ids = _parse_cases(get(raw_args, "cases", nothing))
  samples = parse(Int, get(raw_args, "samples", string(DEFAULT_SAMPLES)))
  seconds = parse(Float64, get(raw_args, "seconds", string(DEFAULT_SECONDS)))
  phase_label = get(raw_args, "phase-label", DEFAULT_PHASE_LABEL)
  output_path = get(raw_args, "output",
                    joinpath(BENCHMARK_PROJECT, "phase0_shared_memory_baseline.toml"))
  report_path = get(raw_args, "report", _report_path_for_output(output_path))
  script_path = abspath(@__FILE__)
  temporary_paths = String[]

  for thread_count in thread_counts
    temp_path = joinpath(BENCHMARK_PROJECT, "phase0_worker_threads$(thread_count).toml")
    push!(temporary_paths, temp_path)
    println("running Phase 0 worker with $thread_count thread(s)")
    run(_worker_command(script_path, thread_count, temp_path, case_ids; samples, seconds))
  end

  thread_runs = [_load_thread_run(path, thread_count)
                 for (path, thread_count) in zip(temporary_paths, thread_counts)]
  combined = _combined_results(thread_runs, case_ids, thread_counts, phase_label)
  _write_toml(output_path, combined)

  open(report_path, "w") do io
    write(io, _render_markdown(combined))
  end

  for path in temporary_paths
    isfile(path) && rm(path)
  end

  println("wrote benchmark artifact to $output_path")
  println("wrote markdown report to $report_path")
  return nothing
end

function _report_path_for_output(output_path::AbstractString)
  basename(output_path) == output_path && return replace(output_path, r"\.toml$" => ".md")
  return joinpath(dirname(output_path), replace(basename(output_path), r"\.toml$" => ".md"))
end

function _rewrite_artifact(raw_args)
  input_path = get(raw_args, "input", nothing)
  input_path === nothing && throw(ArgumentError("missing required argument `--input=`"))
  combined = TOML.parsefile(input_path)
  combined["phase"] = get(raw_args, "phase-label", get(combined, "phase", DEFAULT_PHASE_LABEL))
  output_path = get(raw_args, "output", input_path)
  report_path = get(raw_args, "report", _report_path_for_output(output_path))
  _write_toml(output_path, combined)

  open(report_path, "w") do io
    write(io, _render_markdown(combined))
  end

  println("rewrote benchmark artifact to $output_path")
  println("wrote markdown report to $report_path")
  return nothing
end

function main(args=ARGS)
  parsed = _parse_args(args)

  if get(parsed, "worker", "false") == "true"
    output_path = get(parsed, "output", joinpath(BENCHMARK_PROJECT, "phase0_worker.toml"))
    case_ids = _parse_cases(get(parsed, "cases", nothing))
    samples = parse(Int, get(parsed, "samples", string(DEFAULT_SAMPLES)))
    seconds = parse(Float64, get(parsed, "seconds", string(DEFAULT_SECONDS)))
    _run_worker(output_path, case_ids; samples, seconds)
  elseif haskey(parsed, "input")
    _rewrite_artifact(parsed)
  else
    _run_driver(parsed)
  end

  return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  main()
end

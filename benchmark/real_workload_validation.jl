#!/usr/bin/env julia

const BENCHMARK_PROJECT = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_PROJECT, ".."))
pushfirst!(LOAD_PATH, BENCHMARK_PROJECT)
pushfirst!(LOAD_PATH, REPO_ROOT)

using Dates
using LinearAlgebra
using Printf
using SparseArrays
using Statistics
using TOML

using Grico

const DEFAULT_THREAD_COUNTS = (1, 2, 4, 8, 16)
const DEFAULT_CASES = ("annular_plate_nitsche", "origin_singularity_poisson", "lid_driven_cavity")
const DEFAULT_PROFILE = "validation"
const DEFAULT_REPEATS = 1
const DEFAULT_PHASE_LABEL = "Real-Workload Single-Node Validation"
const DEFAULT_THREAD_NOTE = "Validation runs use the requested physical-core thread counts on this host."

# Minimal command-line parsing shared by driver and worker modes.
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
  raw === nothing && return collect(DEFAULT_CASES)
  selected = split(raw, ",")

  for case_id in selected
    case_id in DEFAULT_CASES ||
      throw(ArgumentError("unknown case `$case_id`; supported cases: $(join(DEFAULT_CASES, ", "))"))
  end

  return selected
end

function _include_example_module(module_name::Symbol, relative_path::AbstractString)
  module_object = Module(module_name)
  Base.include(module_object, joinpath(REPO_ROOT, relative_path))
  return module_object
end

const AnnularExample = _include_example_module(:RealWorkloadAnnularPlateNitsche,
                                               joinpath("examples", "annular_plate_nitsche",
                                                        "benchmarking.jl"))
const OriginExample = _include_example_module(:RealWorkloadOriginSingularityPoisson,
                                              joinpath("examples", "origin_singularity_poisson",
                                                       "benchmarking.jl"))
const LidExample = _include_example_module(:RealWorkloadLidDrivenCavity,
                                           joinpath("examples", "lid_driven_cavity",
                                                    "benchmarking.jl"))

@inline _median(values::Vector{Float64}) = median(values)
@inline _mib(bytes::Real) = bytes / 1024^2
@inline _thread_label(count::Int) = count == 1 ? "thread" : "threads"

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

function _environment_metadata()
  cpu_info = first(Sys.cpu_info())
  return Dict("timestamp_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
              "julia_version" => string(VERSION), "os" => string(Sys.KERNEL),
              "arch" => string(Sys.ARCH), "hostname" => gethostname(),
              "cpu_model" => cpu_info.model, "logical_cpu_threads" => Sys.CPU_THREADS,
              "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
              "openblas_num_threads" => get(ENV, "OPENBLAS_NUM_THREADS", ""),
              "omp_num_threads" => get(ENV, "OMP_NUM_THREADS", ""))
end

# Keep phase timing keys stable across reports so later comparisons can diff
# the same categories even when a workload does not hit every phase.
function _phase_dict()
  return Dict{String,Float64}("setup_seconds" => 0.0, "problem_setup_seconds" => 0.0,
                              "compile_seconds" => 0.0, "assemble_seconds" => 0.0,
                              "solve_seconds" => 0.0, "verify_seconds" => 0.0,
                              "diagnostics_seconds" => 0.0, "adaptivity_seconds" => 0.0,
                              "transfer_seconds" => 0.0)
end

function _trim_breakdown!(breakdown::Dict{String,Float64})
  for key in collect(keys(breakdown))
    breakdown[key] <= 0.0 && delete!(breakdown, key)
  end
  return breakdown
end

function _annular_config(profile::AbstractString)
  if profile == "smoke"
    return (; root_counts=(4, 4), degree=4, segment_count=128, surface_point_count=3,
            fcm_subdivision_depth=5, penalty=AnnularExample.NITSCHE_PENALTY)
  elseif profile == "validation"
    # The shipped annular example assumes every active background leaf
    # intersects the physical domain, so the retained validation case keeps the
    # original `4 x 4` background grid and scales the embedded-boundary work
    # through the segment count instead.
    return (; root_counts=(4, 4), degree=4, segment_count=512, surface_point_count=3,
            fcm_subdivision_depth=7, penalty=AnnularExample.NITSCHE_PENALTY)
  end

  throw(ArgumentError("unsupported profile `$profile`"))
end

function _origin_config(profile::AbstractString)
  if profile == "smoke"
    return (; adaptive_steps=6)
  elseif profile == "validation"
    return (; adaptive_steps=OriginExample.ADAPTIVE_STEPS)
  end

  throw(ArgumentError("unsupported profile `$profile`"))
end

function _lid_config(profile::AbstractString)
  if profile == "smoke"
    return (; root_counts=(8, 8), adaptive_steps=1, max_iters=6, tol=1.0e-4,
            adaptivity_tolerance=LidExample.ADAPTIVITY_TOLERANCE,
            max_h_level=LidExample.MAX_H_LEVEL)
  elseif profile == "validation"
    return (; root_counts=LidExample.ROOT_COUNTS, adaptive_steps=LidExample.ADAPTIVE_STEPS,
            max_iters=LidExample.PICARD_MAX_ITERS, tol=LidExample.PICARD_TOL,
            adaptivity_tolerance=LidExample.ADAPTIVITY_TOLERANCE,
            max_h_level=LidExample.MAX_H_LEVEL)
  end

  throw(ArgumentError("unsupported profile `$profile`"))
end

function _warmup_annular()
  context = AnnularExample.build_annular_plate_nitsche_context(; _annular_config("smoke")...)
  plan = Grico.compile(context.problem)
  system = Grico.assemble(plan)
  values = Grico.solve(system)
  state = Grico.State(plan, values)
  Grico.relative_l2_error(state, context.u, context.exact_solution; plan=plan,
                          cell_quadratures=context.verification_quadratures)
  return nothing
end

function _measure_annular(config)
  breakdown = _phase_dict()
  context = nothing
  breakdown["setup_seconds"] = @elapsed begin
    context = AnnularExample.build_annular_plate_nitsche_context(; config...)
  end
  plan = nothing
  breakdown["compile_seconds"] = @elapsed plan = Grico.compile(context.problem)
  system = nothing
  breakdown["assemble_seconds"] = @elapsed system = Grico.assemble(plan)
  values = nothing
  breakdown["solve_seconds"] = @elapsed values = Grico.solve(system)
  state = Grico.State(plan, values)
  error_value = 0.0
  breakdown["verify_seconds"] = @elapsed begin
    error_value = Grico.relative_l2_error(state, context.u, context.exact_solution; plan=plan,
                                          cell_quadratures=context.verification_quadratures)
  end

  return Dict{String,Any}("config" => Dict("root_counts" => collect(config.root_counts),
                                           "degree" => config.degree,
                                           "segment_count" => config.segment_count,
                                           "surface_point_count" => config.surface_point_count,
                                           "fcm_subdivision_depth" => config.fcm_subdivision_depth,
                                           "penalty" => config.penalty),
                          "breakdown" => _trim_breakdown!(breakdown),
                          "outcome" => Dict("active_leaves" => Grico.active_leaf_count(context.space),
                                            "scalar_dofs" => Grico.scalar_dof_count(context.space),
                                            "reduced_dofs" => size(system.matrix, 1),
                                            "matrix_nnz" => nnz(system.matrix),
                                            "relative_l2_error" => error_value))
end

function _warmup_origin()
  config = _origin_config("smoke")
  context = OriginExample.build_origin_singularity_poisson_context()
  u = context.u

  for step in 0:config.adaptive_steps
    problem = OriginExample.build_origin_singularity_problem(u, context)
    plan = Grico.compile(problem)
    system = Grico.assemble(plan)
    state = Grico.State(plan, Grico.solve(system))
    Grico.relative_l2_error(state, u, context.exact_solution; plan=plan,
                            extra_points=OriginExample.VERIFICATION_EXTRA_POINTS)
    step == config.adaptive_steps && break
    adaptivity_plan = OriginExample.origin_adaptivity_plan(state, u)
    isempty(adaptivity_plan) && break
    u = Grico.adapted_field(Grico.transition(adaptivity_plan), u)
  end

  return nothing
end

function _measure_origin(config)
  breakdown = _phase_dict()
  context = OriginExample.build_origin_singularity_poisson_context()
  u = context.u
  final_system = nothing
  final_error = NaN
  max_leaves = 0
  max_dofs = 0
  completed_steps = 0
  stopped_early = false

  for step in 0:config.adaptive_steps
    problem = nothing
    breakdown["problem_setup_seconds"] += @elapsed problem = OriginExample.build_origin_singularity_problem(u,
                                                                                                            context)

    plan = nothing
    breakdown["compile_seconds"] += @elapsed plan = Grico.compile(problem)
    system = nothing
    breakdown["assemble_seconds"] += @elapsed system = Grico.assemble(plan)
    values = nothing
    breakdown["solve_seconds"] += @elapsed values = Grico.solve(system)
    state = Grico.State(plan, values)
    error_value = 0.0
    breakdown["verify_seconds"] += @elapsed begin
      error_value = Grico.relative_l2_error(state, u, context.exact_solution; plan=plan,
                                            extra_points=OriginExample.VERIFICATION_EXTRA_POINTS)
    end

    final_system = system
    final_error = error_value
    completed_steps = step
    max_leaves = max(max_leaves, Grico.active_leaf_count(Grico.field_space(u)))
    max_dofs = max(max_dofs, Grico.scalar_dof_count(Grico.field_space(u)))
    step == config.adaptive_steps && break

    adaptivity_plan = nothing
    breakdown["adaptivity_seconds"] += @elapsed adaptivity_plan = OriginExample.origin_adaptivity_plan(state,
                                                                                                       u)
    if isempty(adaptivity_plan)
      stopped_early = true
      break
    end

    breakdown["transfer_seconds"] += @elapsed begin
      u = Grico.adapted_field(Grico.transition(adaptivity_plan), u)
    end
  end

  final_space = Grico.field_space(u)
  return Dict{String,Any}("config" => Dict("adaptive_steps" => config.adaptive_steps,
                                           "dimension" => context.dimension,
                                           "initial_degree" => context.initial_degree),
                          "breakdown" => _trim_breakdown!(breakdown),
                          "outcome" => Dict("completed_steps" => completed_steps,
                                            "stopped_early" => stopped_early,
                                            "active_leaves" => Grico.active_leaf_count(final_space),
                                            "scalar_dofs" => Grico.scalar_dof_count(final_space),
                                            "max_active_leaves" => max_leaves,
                                            "max_scalar_dofs" => max_dofs,
                                            "reduced_dofs" => final_system === nothing ? 0 :
                                                              size(final_system.matrix, 1),
                                            "final_relative_l2_error" => final_error,
                                            "final_matrix_nnz" => final_system === nothing ? 0 :
                                                                  nnz(final_system.matrix)))
end

function _warmup_lid()
  config = _lid_config("smoke")
  context = LidExample.build_lid_driven_cavity_context(root_counts=config.root_counts)

  for adaptive_step in 0:config.adaptive_steps
    cycle_tol = adaptive_step == config.adaptive_steps ? config.tol :
                max(config.tol, LidExample.ADAPTIVE_PICARD_TOL)

    for _ in 1:config.max_iters
      context, _, relative_update, _, _ = LidExample.advance_picard_step(context;
                                                                         linear_solve=LidExample.direct_sparse_solve)
      relative_update <= cycle_tol && break
    end

    adaptive_step == config.adaptive_steps && break
    next_context, adaptivity_plan = LidExample.adapt_lid_driven_cavity_context(context;
                                                                               tolerance=config.adaptivity_tolerance,
                                                                               max_h_level=config.max_h_level)
    isempty(adaptivity_plan) && break
    context = next_context
  end

  return nothing
end

function _measure_lid(config)
  breakdown = _phase_dict()
  context = nothing
  breakdown["setup_seconds"] = @elapsed begin
    context = LidExample.build_lid_driven_cavity_context(root_counts=config.root_counts)
  end

  final_update = Inf
  iteration_count = 0
  adaptive_step_count = 0
  max_mixed_dofs = length(Grico.coefficients(context.flow_state))

  for adaptive_step in 0:config.adaptive_steps
    cycle_tol = adaptive_step == config.adaptive_steps ? config.tol :
                max(config.tol, LidExample.ADAPTIVE_PICARD_TOL)

    for iteration in 1:config.max_iters
      context.operator.advecting_state = context.flow_state
      system = nothing
      breakdown["assemble_seconds"] += @elapsed system = Grico.assemble(context.plan)
      candidate_state = nothing
      breakdown["solve_seconds"] += @elapsed begin
        candidate_state = Grico.State(context.plan,
                                      Grico.solve(system;
                                                  linear_solve=LidExample.direct_sparse_solve))
      end
      relative_update = 0.0
      breakdown["diagnostics_seconds"] += @elapsed begin
        context, relative_update, _, _ = LidExample.finalize_picard_step(context, candidate_state)
      end

      max_mixed_dofs = max(max_mixed_dofs, length(Grico.coefficients(context.flow_state)))
      final_update = relative_update
      iteration_count = iteration
      relative_update <= cycle_tol && break
    end

    adaptive_step_count = adaptive_step
    adaptive_step == config.adaptive_steps && break
    next_context = nothing
    adaptivity_plan = nothing
    breakdown["adaptivity_seconds"] += @elapsed begin
      next_context, adaptivity_plan = LidExample.adapt_lid_driven_cavity_context(context;
                                                                                 tolerance=config.adaptivity_tolerance,
                                                                                 max_h_level=config.max_h_level)
    end
    isempty(adaptivity_plan) && break
    context = next_context
  end

  return Dict{String,Any}("config" => Dict("root_counts" => collect(config.root_counts),
                                           "adaptive_steps" => config.adaptive_steps,
                                           "max_iters" => config.max_iters,
                                           "tolerance" => config.tol),
                          "breakdown" => _trim_breakdown!(breakdown),
                          "outcome" => Dict("active_leaves" => Grico.active_leaf_count(context.velocity_space),
                                            "velocity_scalar_dofs" => Grico.scalar_dof_count(context.velocity_space),
                                            "mixed_dofs" => length(Grico.coefficients(context.flow_state)),
                                            "max_mixed_dofs" => max_mixed_dofs,
                                            "adaptive_steps_used" => adaptive_step_count,
                                            "final_picard_iters" => iteration_count,
                                            "final_relative_update" => final_update,
                                            "dg_mass_monitor_l2" => LidExample.dg_mass_monitor_l2(context.plan,
                                                                                                  context.flow_state,
                                                                                                  context.velocity)))
end

function _measure_case(case_id::AbstractString, profile::AbstractString, repeats::Int)
  if case_id == "annular_plate_nitsche"
    description = "Unfitted scalar annulus solve with finite-cell quadrature and embedded Nitsche boundary terms."
    label = "Annular Plate Nitsche"
    warmup! = _warmup_annular
    measure = () -> _measure_annular(_annular_config(profile))
  elseif case_id == "origin_singularity_poisson"
    description = "Adaptive hp Poisson solve with verification and repeated rebuilds on the singular corner problem."
    label = "Origin Singularity Poisson"
    warmup! = _warmup_origin
    measure = () -> _measure_origin(_origin_config(profile))
  elseif case_id == "lid_driven_cavity"
    description = "Adaptive mixed DG lid-driven cavity solve with repeated Picard linearizations and mesh adaptation."
    label = "Lid-Driven Cavity"
    warmup! = _warmup_lid
    measure = () -> _measure_lid(_lid_config(profile))
  else
    throw(ArgumentError("unsupported case `$case_id`"))
  end

  warmup!()
  samples = Dict{String,Any}[]

  for _ in 1:repeats
    GC.gc()
    timed = @timed measure()
    record = timed.value
    record["wall_seconds"] = timed.time
    record["gc_seconds"] = timed.gctime
    record["allocated_bytes"] = timed.bytes
    if hasproperty(timed, :compile_time)
      record["compile_overhead_seconds"] = getproperty(timed, :compile_time)
      record["recompile_overhead_seconds"] = getproperty(timed, :recompile_time)
    end
    push!(samples, record)
  end

  wall_seconds = [sample["wall_seconds"] for sample in samples]
  gc_seconds = [sample["gc_seconds"] for sample in samples]
  allocated_bytes = [Float64(sample["allocated_bytes"]) for sample in samples]
  retained = samples[argmin(wall_seconds)]
  retained["samples"] = length(samples)
  retained["median_wall_seconds"] = _median(wall_seconds)
  retained["median_gc_seconds"] = _median(gc_seconds)
  retained["median_allocated_bytes"] = _median(allocated_bytes)
  retained["label"] = label
  retained["description"] = description
  retained["id"] = case_id
  return retained
end

function _write_toml(path::AbstractString, data)
  mkpath(dirname(path))
  open(path, "w") do io
    TOML.print(io, data)
  end
  return path
end

function _worker_payload(case_ids; profile::AbstractString, repeats::Int)
  BLAS.set_num_threads(1)
  payload = Dict{String,Any}("environment" => _environment_metadata(), "profile" => profile,
                             "repeats" => repeats, "cases" => Any[])

  for case_id in case_ids
    println("benchmarking $case_id on $(Threads.nthreads()) thread(s)")
    push!(payload["cases"], _measure_case(case_id, profile, repeats))
  end

  return payload
end

function _run_worker(output_path::AbstractString, case_ids; profile::AbstractString, repeats::Int)
  payload = _worker_payload(case_ids; profile, repeats)
  return _write_toml(output_path, payload)
end

function _load_thread_run(path::AbstractString, thread_count::Int)
  parsed = TOML.parsefile(path)
  parsed["thread_count"] = thread_count
  return parsed
end

function _combined_results(thread_runs, case_ids, thread_counts, profile::AbstractString,
                           phase_label::AbstractString)
  return Dict("phase" => String(phase_label),
              "generated_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
              "primary_thread_counts" => thread_counts, "selected_cases" => case_ids,
              "profile" => profile, "performance_core_note" => DEFAULT_THREAD_NOTE,
              "thread_runs" => thread_runs)
end

function _record_lookup(combined)
  lookup = Dict{Tuple{String,Int},Dict{String,Any}}()

  for thread_run in combined["thread_runs"]
    for case_entry in thread_run["cases"]
      lookup[(case_entry["id"], thread_run["thread_count"])] = case_entry
    end
  end

  return lookup
end

function _case_lookup(combined)
  lookup = Dict{String,Dict{String,Any}}()
  for case_entry in first(combined["thread_runs"])["cases"]
    lookup[case_entry["id"]] = case_entry
  end
  return lookup
end

function _speedup(lookup, case_id::String, thread_count::Int)
  base = lookup[(case_id, 1)]
  current = lookup[(case_id, thread_count)]
  speedup = base["median_wall_seconds"] / current["median_wall_seconds"]
  return base, current, speedup, speedup / thread_count
end

function _dominant_phase(case_entry::Dict{String,Any})
  breakdown = case_entry["breakdown"]
  best_phase = ""
  best_seconds = -Inf

  for (phase, seconds) in breakdown
    seconds > best_seconds || continue
    best_phase = phase
    best_seconds = seconds
  end

  return best_phase, best_seconds
end

function _findings(combined)
  lookup = _record_lookup(combined)
  max_threads = maximum(combined["primary_thread_counts"])
  findings = String[]

  for case_id in combined["selected_cases"]
    base, current, speedup, efficiency = _speedup(lookup, case_id, max_threads)
    dominant_phase, dominant_seconds = _dominant_phase(current)
    push!(findings,
          "`$case_id` improves from $(_format_time(base["median_wall_seconds"])) at 1 thread to $(_format_time(current["median_wall_seconds"])) at $max_threads $(_thread_label(max_threads)), or `$( @sprintf("%.2f", speedup) )x` with `$( @sprintf("%.2f", efficiency) )` efficiency. The dominant retained phase at $max_threads $(_thread_label(max_threads)) is `$(dominant_phase)` with $(_format_time(dominant_seconds)).")
  end

  return findings
end

function _render_markdown(combined)
  lookup = _record_lookup(combined)
  case_lookup = _case_lookup(combined)
  lines = String[]

  push!(lines, "# $(combined["phase"])")
  push!(lines, "")
  push!(lines, "Generated: `$(combined["generated_utc"])`")
  push!(lines, "")
  push!(lines, "Profile: `$(combined["profile"])`")
  push!(lines, "")
  push!(lines, "Thread counts: `$(join(combined["primary_thread_counts"], ", "))`")
  push!(lines, "")
  push!(lines, combined["performance_core_note"])
  push!(lines, "")
  push!(lines, "## Environment")
  push!(lines, "")
  env = first(combined["thread_runs"])["environment"]
  push!(lines, "- Julia: `$(env["julia_version"])`")
  push!(lines, "- Host: `$(env["hostname"])`")
  push!(lines, "- CPU: `$(env["cpu_model"])`")
  push!(lines, "- BLAS threads: `$(env["blas_threads"])`")
  push!(lines, "")
  push!(lines, "## Cases")
  push!(lines, "")

  for case_id in combined["selected_cases"]
    case_entry = case_lookup[case_id]
    push!(lines, "### `$(case_id)`")
    push!(lines, "")
    push!(lines, case_entry["description"])
    push!(lines, "")
    push!(lines, "Configuration:")
    push!(lines, "")
    for (key, value) in sort!(collect(case_entry["config"]); by=first)
      formatted = value isa Vector ? join(value, ", ") : string(value)
      push!(lines, "- `$key`: `$(formatted)`")
    end
    push!(lines, "")
    push!(lines, "| Threads | Wall time | Speedup | Efficiency | Allocated | GC time |")
    push!(lines, "| ---: | ---: | ---: | ---: | ---: | ---: |")

    for thread_count in combined["primary_thread_counts"]
      record = lookup[(case_id, thread_count)]
      speedup = lookup[(case_id, 1)]["median_wall_seconds"] / record["median_wall_seconds"]
      efficiency = speedup / thread_count
      push!(lines,
            "| $(thread_count) | $(_format_time(record["median_wall_seconds"])) | $( @sprintf("%.2f", speedup) ) | $( @sprintf("%.2f", efficiency) ) | $(@sprintf("%.2f MiB", _mib(record["median_allocated_bytes"]))) | $(_format_time(record["median_gc_seconds"])) |")
    end

    max_record = lookup[(case_id, maximum(combined["primary_thread_counts"]))]
    push!(lines, "")
    push!(lines, "Retained outcome at `$(maximum(combined["primary_thread_counts"]))` threads:")
    push!(lines, "")
    for (key, value) in sort!(collect(max_record["outcome"]); by=first)
      push!(lines, "- `$key`: `$(value)`")
    end
    push!(lines, "")
    push!(lines, "Phase breakdown at `$(maximum(combined["primary_thread_counts"]))` threads:")
    push!(lines, "")
    push!(lines, "| Phase | Time | Share |")
    push!(lines, "| --- | ---: | ---: |")
    total = sum(values(max_record["breakdown"]))

    for (phase, seconds) in sort!(collect(max_record["breakdown"]); by=last, rev=true)
      push!(lines,
            "| `$(phase)` | $(_format_time(seconds)) | $( @sprintf("%.1f%%", 100 * seconds / total) ) |")
    end

    push!(lines, "")
  end

  push!(lines, "## Findings")
  push!(lines, "")
  for finding in _findings(combined)
    push!(lines, "- $finding")
  end

  return join(lines, "\n")
end

function _worker_command(script_path::AbstractString, thread_count::Int,
                         output_path::AbstractString, case_ids; profile::AbstractString,
                         repeats::Int)
  command = `$(Base.julia_cmd()) --project=$(BENCHMARK_PROJECT) --threads=$(thread_count) $(script_path) --worker --output=$(output_path) --profile=$(profile) --repeats=$(repeats) --cases=$(join(case_ids, ","))`
  environment = copy(ENV)
  environment["OPENBLAS_NUM_THREADS"] = "1"
  environment["OMP_NUM_THREADS"] = "1"
  return setenv(command, environment)
end

function _report_path_for_output(output_path::AbstractString)
  basename(output_path) == output_path && return replace(output_path, r"\.toml$" => ".md")
  return joinpath(dirname(output_path), replace(basename(output_path), r"\.toml$" => ".md"))
end

function _run_driver(raw_args)
  thread_counts = _parse_thread_counts(get(raw_args, "threads", nothing))
  case_ids = _parse_cases(get(raw_args, "cases", nothing))
  profile = get(raw_args, "profile", DEFAULT_PROFILE)
  repeats = parse(Int, get(raw_args, "repeats", string(DEFAULT_REPEATS)))
  phase_label = get(raw_args, "phase-label", DEFAULT_PHASE_LABEL)
  output_path = get(raw_args, "output",
                    joinpath(BENCHMARK_PROJECT, "real_workload_validation.toml"))
  report_path = get(raw_args, "report", _report_path_for_output(output_path))
  script_path = abspath(@__FILE__)
  temporary_paths = String[]

  for thread_count in thread_counts
    temp_path = joinpath(BENCHMARK_PROJECT, "real_workload_validation_threads$(thread_count).toml")
    push!(temporary_paths, temp_path)
    println("running real-workload worker with $thread_count thread(s)")
    run(_worker_command(script_path, thread_count, temp_path, case_ids; profile, repeats))
  end

  thread_runs = [_load_thread_run(path, thread_count)
                 for (path, thread_count) in zip(temporary_paths, thread_counts)]
  combined = _combined_results(thread_runs, case_ids, thread_counts, profile, phase_label)
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

function main(args=ARGS)
  parsed = _parse_args(args)

  if get(parsed, "worker", "false") == "true"
    output_path = get(parsed, "output",
                      joinpath(BENCHMARK_PROJECT, "real_workload_validation_worker.toml"))
    case_ids = _parse_cases(get(parsed, "cases", nothing))
    profile = get(parsed, "profile", DEFAULT_PROFILE)
    repeats = parse(Int, get(parsed, "repeats", string(DEFAULT_REPEATS)))
    _run_worker(output_path, case_ids; profile, repeats)
  else
    _run_driver(parsed)
  end

  return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  main()
end

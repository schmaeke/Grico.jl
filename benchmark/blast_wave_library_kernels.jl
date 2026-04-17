#!/usr/bin/env julia

const BENCHMARK_PROJECT = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_PROJECT, ".."))

# Keep the benchmark independent of user/global environments. In particular,
# the blast-wave example should see OrdinaryDiffEq as unavailable unless the
# active project provides it.
empty!(LOAD_PATH)
push!(LOAD_PATH, REPO_ROOT)
push!(LOAD_PATH, "@stdlib")

using Dates
using LinearAlgebra
using Printf
using Statistics
using TOML

using Grico

const DEFAULT_THREAD_COUNTS = (1, 2, 4, 6)
const DEFAULT_REPETITIONS = 5
const DEFAULT_WARMUPS = 1
const DEFAULT_RHS_STEPS = 5
const DEFAULT_ADVANCE_STEPS = 2
const DEFAULT_PHASE_LABEL = "Blast-Wave Library Kernel Baseline"
const DEFAULT_THREAD_NOTE = "Primary scaling set capped at 6 Julia threads for Apple M2 Pro performance-core runs."

ENV["GRICO_BLAST_WAVE_EULER_AUTORUN"] = "0"
include(joinpath(REPO_ROOT, "examples", "blast_wave_euler.jl"))

struct KernelOperation
  id::String
  label::String
  run::Function
  setup::Function
end

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

function _parse_pair(raw::AbstractString)
  parts = split(raw, ",")
  length(parts) == 2 || throw(ArgumentError("expected `a,b`, got `$raw`"))
  return (parse(Int, parts[1]), parse(Int, parts[2]))
end

@inline _mib(bytes::Real) = bytes / 1024^2

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
              "omp_num_threads" => get(ENV, "OMP_NUM_THREADS", ""),
              "ordinarydiffeq_available" => ORDINARYDIFFEQ_AVAILABLE)
end

function _benchmark_config(parsed)
  return (; root_counts=_parse_pair(get(parsed, "root-counts", "16,16")),
          degree=parse(Int, get(parsed, "degree", string(POLYDEG))),
          quadrature_extra_points=parse(Int,
                                        get(parsed, "quadrature-extra-points",
                                            string(QUADRATURE_EXTRA_POINTS))),
          initial_refinement_layers=parse(Int,
                                          get(parsed, "layers",
                                              string(INITIAL_BLAST_REFINEMENT_LAYERS))),
          max_h_level=parse(Int, get(parsed, "max-h-level", string(MAX_H_LEVEL))),
          tolerance=parse(Float64, get(parsed, "tolerance", string(ADAPTIVITY_TOLERANCE))),
          rhs_steps=parse(Int, get(parsed, "rhs-steps", string(DEFAULT_RHS_STEPS))),
          advance_steps=parse(Int, get(parsed, "advance-steps", string(DEFAULT_ADVANCE_STEPS))))
end

function _string_key_dict(nt::NamedTuple)
  return Dict{String,Any}(String(key) => value for (key, value) in pairs(nt))
end

function _config_dict(config)
  return Dict{String,Any}("root_counts" => collect(config.root_counts), "degree" => config.degree,
                          "quadrature_extra_points" => config.quadrature_extra_points,
                          "initial_refinement_layers" => config.initial_refinement_layers,
                          "max_h_level" => config.max_h_level, "tolerance" => config.tolerance,
                          "rhs_steps" => config.rhs_steps, "advance_steps" => config.advance_steps)
end

function _advance_state(context, steps::Int)
  steps >= 0 || throw(ArgumentError("advance steps must be nonnegative"))
  u = copy(coefficients(context.state))
  du = similar(u)
  semi = EulerSemidiscretization(context.spatial_plan, context.conserved, context.state,
                                 context.mass_inverse)

  for step in 1:steps
    euler_rhs!(du, u, semi, (step - 1) * context.dt)
    @. u = u + context.dt * du
  end

  return State(context.spatial_plan, copy(u))
end

function _prepare_case(config)
  BLAS.set_num_threads(1)
  build_timed = @timed build_blast_wave_euler_context(; root_counts=config.root_counts,
                                                      degree=config.degree,
                                                      quadrature_extra_points=config.quadrature_extra_points,
                                                      initial_refinement_layers=config.initial_refinement_layers)
  BLAS.set_num_threads(1)
  context = build_timed.value
  profiled_state = _advance_state(context, config.advance_steps)
  profiled_context = refresh_blast_wave_context(context, profiled_state)
  adaptivity_plan_cache = Ref{Any}(nothing)
  transition_cache = Ref{Any}(nothing)
  new_conserved_cache = Ref{Any}(nothing)
  new_state_cache = Ref{Any}(nothing)
  residual_buffer = zeros(Float64, Grico.dof_count(context.spatial_plan))
  residual_workspace = ResidualWorkspace(context.spatial_plan)
  rhs_seed = copy(coefficients(profiled_state))
  rhs_du = similar(rhs_seed)
  rhs_semi = EulerSemidiscretization(context.spatial_plan, context.conserved, profiled_state,
                                     context.mass_inverse)

  ensure_plan! = () -> begin
    if adaptivity_plan_cache[] === nothing
      adaptivity_plan_cache[] = blast_wave_adaptivity_plan(profiled_context;
                                                           tolerance=config.tolerance,
                                                           max_h_level=config.max_h_level)
    end
    return adaptivity_plan_cache[]
  end

  ensure_transition! = () -> begin
    if transition_cache[] === nothing
      transition_cache[] = transition(ensure_plan!())
    end
    return transition_cache[]
  end

  ensure_new_field! = () -> begin
    if new_conserved_cache[] === nothing
      new_conserved_cache[] = adapted_field(ensure_transition!(), context.conserved)
    end
    return new_conserved_cache[]
  end

  ensure_new_state! = () -> begin
    if new_state_cache[] === nothing
      new_state_cache[] = transfer_state(ensure_transition!(), profiled_state, context.conserved,
                                         ensure_new_field!(); linear_solve=direct_sparse_solve)
    end
    return new_state_cache[]
  end

  no_setup = () -> nothing
  compile_residual_plan_run = () -> euler_residual_plan(context.conserved, context.gamma)
  residual_run = () -> residual!(residual_buffer, context.spatial_plan, profiled_state,
                                 residual_workspace)
  residual_one_shot_run = () -> residual!(residual_buffer, context.spatial_plan, profiled_state)
  rhs_run = function ()
    local_u = copy(rhs_seed)

    for step in 1:config.rhs_steps
      euler_rhs!(rhs_du, local_u, rhs_semi, (step - 1) * context.dt)
      @. local_u = local_u + context.dt * rhs_du
    end

    return nothing
  end
  adaptivity_plan_run = function ()
    adaptivity_plan_cache[] = blast_wave_adaptivity_plan(profiled_context;
                                                         tolerance=config.tolerance,
                                                         max_h_level=config.max_h_level)
    return nothing
  end
  transition_run = function ()
    transition_cache[] = transition(ensure_plan!())
    return nothing
  end
  dg_transfer_run = () -> transfer_state(ensure_transition!(), profiled_state, context.conserved,
                                         ensure_new_field!(); linear_solve=direct_sparse_solve)
  rebuild_context_run = () -> blast_wave_context(ensure_new_field!(), ensure_new_state!();
                                                 gamma=context.gamma, cfl=context.cfl,
                                                 degree=context.degree)
  adapt_all_run = () -> adapt_blast_wave_context(profiled_context; tolerance=config.tolerance,
                                                 max_h_level=config.max_h_level,
                                                 linear_solve=direct_sparse_solve)

  operations = KernelOperation[KernelOperation("compile_residual_plan",
                                               "compile Euler residual AssemblyPlan",
                                               compile_residual_plan_run, no_setup),
                               KernelOperation("residual_bang",
                                               "residual!(buffer, plan, state, workspace)",
                                               residual_run, no_setup),
                               KernelOperation("residual_bang_one_shot",
                                               "residual!(buffer, plan, state)",
                                               residual_one_shot_run, no_setup),
                               KernelOperation("rhs_steps",
                                               "$(config.rhs_steps) euler_rhs! evaluations",
                                               rhs_run, no_setup),
                               KernelOperation("adaptivity_plan",
                                               "blast_wave_adaptivity_plan(context)",
                                               adaptivity_plan_run, no_setup),
                               KernelOperation("transition", "transition(adaptivity_plan)",
                                               transition_run, no_setup),
                               KernelOperation("dg_transfer_state",
                                               "transfer_state on fully DG target", dg_transfer_run,
                                               no_setup),
                               KernelOperation("rebuild_context",
                                               "blast_wave_context after adaptation",
                                               rebuild_context_run, no_setup),
                               KernelOperation("adapt_all", "adapt_blast_wave_context(context)",
                                               adapt_all_run, no_setup)]

  plan = ensure_plan!()
  target_space = transition(plan).target_space
  metadata = Dict{String,Any}("build_seconds" => build_timed.time,
                              "build_allocated_bytes" => build_timed.bytes,
                              "root_counts" => collect(config.root_counts),
                              "degree" => config.degree,
                              "quadrature_extra_points" => config.quadrature_extra_points,
                              "initial_refinement_layers" => config.initial_refinement_layers,
                              "max_h_level" => config.max_h_level, "tolerance" => config.tolerance,
                              "rhs_steps" => config.rhs_steps,
                              "advance_steps" => config.advance_steps,
                              "source_active_leaves" => active_leaf_count(context.space),
                              "source_dofs" => length(coefficients(context.state)),
                              "source_cells" => length(context.spatial_plan.integration.cells),
                              "source_boundary_faces" => length(context.spatial_plan.integration.boundary_faces),
                              "source_interfaces" => length(context.spatial_plan.integration.interfaces),
                              "source_dt" => context.dt,
                              "adaptivity_summary" => _string_key_dict(adaptivity_summary(plan)),
                              "target_active_leaves" => active_leaf_count(target_space),
                              "target_scalar_dofs" => scalar_dof_count(target_space),
                              "target_vector_dofs" => component_count(context.conserved) *
                                                      scalar_dof_count(target_space))

  return (; context, profiled_context, operations, metadata)
end

function _measure_operation(operation::KernelOperation; repetitions::Int, warmups::Int)
  repetitions > 0 || throw(ArgumentError("repetitions must be positive"))
  warmups >= 0 || throw(ArgumentError("warmups must be nonnegative"))

  for _ in 1:warmups
    BLAS.set_num_threads(1)
    operation.setup()
    operation.run()
  end

  times = Float64[]
  gctimes = Float64[]
  bytes = Float64[]

  for _ in 1:repetitions
    BLAS.set_num_threads(1)
    operation.setup()
    GC.gc()
    timed = @timed operation.run()
    push!(times, timed.time)
    push!(gctimes, timed.gctime)
    push!(bytes, Float64(timed.bytes))
  end

  return Dict("repetitions" => repetitions, "warmups" => warmups, "median_seconds" => median(times),
              "minimum_seconds" => minimum(times), "maximum_seconds" => maximum(times),
              "median_gc_seconds" => median(gctimes), "median_allocated_bytes" => median(bytes),
              "minimum_allocated_bytes" => minimum(bytes),
              "maximum_allocated_bytes" => maximum(bytes))
end

function _worker_payload(parsed)
  BLAS.set_num_threads(1)
  repetitions = parse(Int, get(parsed, "repetitions", string(DEFAULT_REPETITIONS)))
  warmups = parse(Int, get(parsed, "warmups", string(DEFAULT_WARMUPS)))
  config = _benchmark_config(parsed)
  prepared = _prepare_case(config)
  BLAS.set_num_threads(1)
  payload = Dict{String,Any}("environment" => _environment_metadata(),
                             "config" => _config_dict(config), "metadata" => prepared.metadata,
                             "operations" => Any[])

  for operation in prepared.operations
    println("benchmarking $(operation.id) on $(Threads.nthreads()) thread(s)")
    metrics = _measure_operation(operation; repetitions, warmups)
    push!(payload["operations"],
          Dict("id" => operation.id, "label" => operation.label, "metrics" => metrics))
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

function _load_thread_run(path::AbstractString, thread_count::Int)
  parsed = TOML.parsefile(path)
  parsed["thread_count"] = thread_count
  return parsed
end

function _combined_results(thread_runs, thread_counts, phase_label::AbstractString)
  return Dict("phase" => String(phase_label),
              "generated_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
              "primary_thread_counts" => thread_counts,
              "performance_core_note" => get(ENV, "GRICO_BENCHMARK_THREAD_NOTE",
                                             DEFAULT_THREAD_NOTE), "thread_runs" => thread_runs)
end

function _operation_lookup(combined)
  lookup = Dict{Tuple{String,Int},Dict{String,Any}}()

  for thread_run in combined["thread_runs"]
    thread_count = Int(thread_run["thread_count"])

    for operation in thread_run["operations"]
      lookup[(operation["id"], thread_count)] = operation
    end
  end

  return lookup
end

function _operation_order(combined)
  return [operation["id"] for operation in first(combined["thread_runs"])["operations"]]
end

function _operation_labels(combined)
  labels = Dict{String,String}()

  for operation in first(combined["thread_runs"])["operations"]
    labels[operation["id"]] = operation["label"]
  end

  return labels
end

function _render_markdown(combined)
  lines = String[]
  thread_counts = [Int(value) for value in combined["primary_thread_counts"]]
  lookup = _operation_lookup(combined)
  labels = _operation_labels(combined)
  operation_ids = _operation_order(combined)
  metadata = first(combined["thread_runs"])["metadata"]
  env = first(combined["thread_runs"])["environment"]

  push!(lines, "# $(combined["phase"])")
  push!(lines, "")
  push!(lines, "Generated: `$(combined["generated_utc"])`")
  push!(lines, "")
  push!(lines, combined["performance_core_note"])
  push!(lines, "")
  push!(lines, "## Environment")
  push!(lines, "")
  push!(lines, "- Julia: `$(env["julia_version"])`")
  push!(lines, "- Host: `$(env["hostname"])`")
  push!(lines, "- CPU: `$(env["cpu_model"])`")
  push!(lines, "- OrdinaryDiffEq visible to benchmark: `$(env["ordinarydiffeq_available"])`")
  push!(lines, "")
  push!(lines, "## Workload")
  push!(lines, "")
  push!(lines, "- Source leaves: `$(metadata["source_active_leaves"])`")
  push!(lines, "- Source dofs: `$(metadata["source_dofs"])`")
  push!(lines,
        "- Source cells/interfaces: `$(metadata["source_cells"])` / `$(metadata["source_interfaces"])`")
  push!(lines, "- Target leaves: `$(metadata["target_active_leaves"])`")
  push!(lines, "- Target vector dofs: `$(metadata["target_vector_dofs"])`")
  push!(lines, "- Adaptivity summary: `$(metadata["adaptivity_summary"])`")
  push!(lines, "")
  push!(lines, "## Kernel Timings")
  push!(lines, "")
  push!(lines, "| Operation | Threads | Median time | Speedup | Allocated | GC time |")
  push!(lines, "| --- | ---: | ---: | ---: | ---: | ---: |")

  for operation_id in operation_ids
    base_seconds = lookup[(operation_id, first(thread_counts))]["metrics"]["median_seconds"]

    for thread_count in thread_counts
      metrics = lookup[(operation_id, thread_count)]["metrics"]
      speedup = base_seconds / metrics["median_seconds"]
      push!(lines,
            "| `$(operation_id)` | $(thread_count) | $(_format_time(metrics["median_seconds"])) | $(@sprintf("%.2f", speedup)) | $(@sprintf("%.2f MiB", _mib(metrics["median_allocated_bytes"]))) | $(_format_time(metrics["median_gc_seconds"])) |")
    end
  end

  push!(lines, "")
  push!(lines, "## Operation Labels")
  push!(lines, "")

  for operation_id in operation_ids
    push!(lines, "- `$(operation_id)`: $(labels[operation_id])")
  end

  return join(lines, "\n")
end

function _worker_command(script_path::AbstractString, thread_count::Int,
                         output_path::AbstractString, parsed)
  args = String["--worker", "--output=$(output_path)",
                "--repetitions=$(get(parsed, "repetitions", string(DEFAULT_REPETITIONS)))",
                "--warmups=$(get(parsed, "warmups", string(DEFAULT_WARMUPS)))",
                "--rhs-steps=$(get(parsed, "rhs-steps", string(DEFAULT_RHS_STEPS)))",
                "--advance-steps=$(get(parsed, "advance-steps", string(DEFAULT_ADVANCE_STEPS)))"]

  for key in
      ("root-counts", "degree", "quadrature-extra-points", "layers", "max-h-level", "tolerance")
    haskey(parsed, key) && push!(args, "--$(key)=$(parsed[key])")
  end

  command = `$(Base.julia_cmd()) --project=$(REPO_ROOT) --threads=$(thread_count) $(script_path) $(args)`
  environment = copy(ENV)
  environment["OPENBLAS_NUM_THREADS"] = "1"
  environment["OMP_NUM_THREADS"] = "1"
  return setenv(command, environment)
end

function _report_path_for_output(output_path::AbstractString)
  basename(output_path) == output_path && return replace(output_path, r"\.toml$" => ".md")
  return joinpath(dirname(output_path), replace(basename(output_path), r"\.toml$" => ".md"))
end

function _run_worker(parsed)
  output_path = get(parsed, "output",
                    joinpath(BENCHMARK_PROJECT, "blast_wave_library_kernels_worker.toml"))
  payload = _worker_payload(parsed)
  _write_toml(output_path, payload)
  println("wrote worker artifact to $output_path")
  return nothing
end

function _run_driver(parsed)
  thread_counts = _parse_thread_counts(get(parsed, "threads", nothing))
  output_path = get(parsed, "output",
                    joinpath(BENCHMARK_PROJECT, "blast_wave_library_kernels.toml"))
  report_path = get(parsed, "report", _report_path_for_output(output_path))
  phase_label = get(parsed, "phase-label", DEFAULT_PHASE_LABEL)
  script_path = abspath(@__FILE__)
  temporary_paths = String[]

  for thread_count in thread_counts
    temp_path = joinpath(BENCHMARK_PROJECT,
                         "blast_wave_library_kernels_threads$(thread_count).toml")
    push!(temporary_paths, temp_path)
    println("running blast-wave library worker with $thread_count thread(s)")
    run(_worker_command(script_path, thread_count, temp_path, parsed))
  end

  thread_runs = [_load_thread_run(path, thread_count)
                 for (path, thread_count) in zip(temporary_paths, thread_counts)]
  combined = _combined_results(thread_runs, thread_counts, phase_label)
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
    _run_worker(parsed)
  else
    _run_driver(parsed)
  end

  return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  main()
end

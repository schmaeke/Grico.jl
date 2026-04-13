#!/usr/bin/env julia

const BENCHMARK_PROJECT = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_PROJECT, ".."))
pushfirst!(LOAD_PATH, BENCHMARK_PROJECT)
pushfirst!(LOAD_PATH, REPO_ROOT)

using LinearAlgebra
using Printf
using SparseArrays
using Statistics
using Dates
using TOML
using Krylov
using Grico

include(joinpath(@__DIR__, "shared_memory_phase0_cases.jl"))
using .SharedMemoryPhase0Cases

const DEFAULT_REPEATS = 3
const DEFAULT_PHASE_LABEL = "Phase 7 Solver Path Cleanup"
const FLOW_ROOT_COUNTS = (16, 16)

struct SolverCandidate
  id::String
  label::String
  build::Function
end

struct SolverCase
  id::String
  label::String
  description::String
  plan::Any
  system::Any
  metadata::Dict{String,Any}
  candidates::Vector{SolverCandidate}
end

struct SolverMeasurement
  id::String
  label::String
  setup_seconds::Float64
  solve_seconds::Float64
  warm_solve_seconds::Float64
  total_seconds::Float64
  iterations::Int
  converged::Bool
  residual::Float64
end

struct DefaultPolicyMeasurement
  label::String
  cold_seconds::Float64
  warm_seconds::Float64
  residual::Float64
end

function _parse_args(args)
  parsed = Dict{String,String}()

  for arg in args
    if startswith(arg, "--")
      key, value = occursin("=", arg) ? split(arg[3:end], "="; limit=2) : (arg[3:end], "true")
      parsed[key] = value
    else
      throw(ArgumentError("unsupported argument `$arg`"))
    end
  end

  return parsed
end

@inline _median(values::Vector{Float64}) = median(values)
@inline _median(values::Vector{Int}) = Int(round(median(values)))

@inline function _format_time(seconds::Real)
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

function _integration_metadata(plan)
  integration = plan.integration
  local_counts = Int[]

  for item in integration.cells
    push!(local_counts, item.local_dof_count)
  end

  for item in integration.boundary_faces
    push!(local_counts, item.local_dof_count)
  end

  for item in integration.interfaces
    push!(local_counts, item.local_dof_count)
  end

  for item in integration.embedded_surfaces
    push!(local_counts, item.local_dof_count)
  end

  return Dict{String,Any}("cells" => length(integration.cells),
                          "boundary_faces" => length(integration.boundary_faces),
                          "interfaces" => length(integration.interfaces),
                          "embedded_surfaces" => length(integration.embedded_surfaces),
                          "max_local_dofs" => isempty(local_counts) ? 0 : maximum(local_counts))
end

function _case_metadata(plan, system; extra::Dict{String,Any}=Dict{String,Any}())
  metadata = _integration_metadata(plan)
  metadata["full_dofs"] = Grico.dof_count(plan)
  metadata["reduced_dofs"] = size(system.matrix, 1)
  metadata["matrix_nnz"] = nnz(system.matrix)
  metadata["symmetric"] = system.symmetric
  metadata["fields"] = Grico.field_count(Grico.field_layout(plan))

  for (key, value) in extra
    metadata[key] = value
  end

  return metadata
end

function _build_affine_cell_case()
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (40, 40))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(3)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, SharedMemoryPhase0Cases.Diffusion(u, 1.0))
  Grico.add_cell!(problem,
                  SharedMemoryPhase0Cases.Source(u, SharedMemoryPhase0Cases._poisson_source_term))

  for axis in 1:2, side in (Grico.LOWER, Grico.UPPER)
    Grico.add_constraint!(problem,
                          Grico.Dirichlet(u, Grico.BoundaryFace(axis, side),
                                          SharedMemoryPhase0Cases._poisson_exact_solution))
  end

  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  metadata = _case_metadata(plan, system;
                            extra=Dict("case_kind" => "affine", "continuity" => "cg", "degree" => 3,
                                       "active_leaves" => length(Grico.active_leaves(space))))
  candidates = SolverCandidate[SolverCandidate("direct", "Sparse direct", () -> nothing),
                               SolverCandidate("amg", "Smoothed Aggregation AMG + CG",
                                               () -> Grico.SmoothedAggregationAMGPreconditioner(min_dofs=0)),
                               SolverCandidate("schwarz", "Additive Schwarz + CG",
                                               () -> Grico.AdditiveSchwarzPreconditioner(min_dofs=0))]
  return SolverCase("affine_cell_diffusion", "Affine Cell-Dominated Diffusion",
                    "Continuous scalar Poisson problem with symmetric volume-dominated assembly.",
                    plan, system, metadata, candidates)
end

function _build_affine_interface_case()
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (40, 40))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2), continuity=:dg))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, SharedMemoryPhase0Cases.MassCoupling(u, u, 1.0))
  Grico.add_interface!(problem, SharedMemoryPhase0Cases.GradientJumpPenalty(u, 0.5))
  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  metadata = _case_metadata(plan, system;
                            extra=Dict("case_kind" => "affine", "continuity" => "dg", "degree" => 2,
                                       "active_leaves" => length(Grico.active_leaves(space))))
  candidates = SolverCandidate[SolverCandidate("direct", "Sparse direct", () -> nothing),
                               SolverCandidate("amg", "Smoothed Aggregation AMG + CG",
                                               () -> Grico.SmoothedAggregationAMGPreconditioner(min_dofs=0)),
                               SolverCandidate("schwarz", "Additive Schwarz + CG",
                                               () -> Grico.AdditiveSchwarzPreconditioner(min_dofs=0))]
  return SolverCase("affine_interface_dg", "Affine Interface-Heavy DG",
                    "Discontinuous symmetric scalar problem with explicit interior interface work.",
                    plan, system, metadata, candidates)
end

let previous = get(ENV, "GRICO_LDC_AUTORUN", nothing)
  ENV["GRICO_LDC_AUTORUN"] = "0"
  include(joinpath(REPO_ROOT, "examples", "lid_driven_cavity.jl"))

  if previous === nothing
    delete!(ENV, "GRICO_LDC_AUTORUN")
  else
    ENV["GRICO_LDC_AUTORUN"] = previous
  end
end

function _build_flow_case()
  context = build_lid_driven_cavity_context(root_counts=FLOW_ROOT_COUNTS)
  flow_preconditioner = Grico.FieldSplitSchurPreconditioner((context.velocity,),
                                                            (context.pressure,); min_dofs=0)
  context, system, _, _, _ = advance_picard_step(context)
  metadata = Dict{String,Any}("reduced_dofs" => size(system.matrix, 1),
                              "matrix_nnz" => nnz(system.matrix), "symmetric" => system.symmetric,
                              "fields" => Grico.field_count(system.layout),
                              "case_kind" => "mixed_flow", "picard_step" => 1,
                              "root_counts" => collect(FLOW_ROOT_COUNTS))
  candidates = SolverCandidate[SolverCandidate("direct", "Sparse direct", () -> nothing),
                               SolverCandidate("ilu", "ILU + GMRES",
                                               () -> Grico.ILUPreconditioner(min_dofs=0)),
                               SolverCandidate("schwarz", "Additive Schwarz + GMRES",
                                               () -> Grico.AdditiveSchwarzPreconditioner(min_dofs=0)),
                               SolverCandidate("fieldsplit", "Field-Split Schur + GMRES",
                                               () -> flow_preconditioner)]
  return SolverCase("lid_driven_cavity_step1", "Lid-Driven Cavity Picard Step 1",
                    "Mixed velocity-pressure reduced system from the DG lid-driven cavity example.",
                    nothing, system, metadata, candidates)
end

_build_cases() = [_build_affine_cell_case(), _build_affine_interface_case(), _build_flow_case()]

function _warmup_case!(case::SolverCase)
  case.plan !== nothing && Grico.assemble(case.plan)
  empty!(case.system.preconditioner_cache)
  try
    Grico.solve(case.system)
  catch
  end
  empty!(case.system.preconditioner_cache)
  Grico._with_internal_blas_threads() do
    direct_operator = Grico._build_ordered_direct_operator(case.system.matrix)
    direct_solution = zeros(eltype(case.system.rhs), size(case.system.matrix, 1))
    ldiv!(direct_solution, direct_operator, case.system.rhs)
  end

  for candidate in case.candidates
    candidate.id == "direct" && continue
    empty!(case.system.preconditioner_cache)
    Grico._with_internal_blas_threads() do
      preconditioner = candidate.build()
      operator = Grico._preconditioner_operator(case.system, preconditioner)
      style = Grico._preconditioned_krylov_style(case.system, preconditioner)
      try
        _run_krylov(case.system, operator, style)
      catch
      end
    end
  end

  empty!(case.system.preconditioner_cache)
  return nothing
end

function _assemble_seconds(plan, repeats::Int)
  plan === nothing && return 0.0
  samples = Float64[]

  for _ in 1:repeats
    push!(samples, @elapsed Grico.assemble(plan))
  end

  return _median(samples)
end

function _reduced_solution(system::Grico.AffineSystem, full_values::AbstractVector{T}) where {T}
  return full_values[system.solve_dofs]
end

function _run_krylov(system::Grico.AffineSystem{T}, operator,
                     ::Grico._SymmetricKrylovStyle) where {T<:AbstractFloat}
  return cg(Symmetric(system.matrix), system.rhs; Grico._cg_krylov_options(system, operator)...)
end

function _run_krylov(system::Grico.AffineSystem{T}, operator,
                     ::Grico._GeneralKrylovStyle) where {T<:AbstractFloat}
  return gmres(system.matrix, system.rhs; Grico._gmres_krylov_options(system, operator)...)
end

function _measure_default_policy(system::Grico.AffineSystem{T},
                                 repeats::Int) where {T<:AbstractFloat}
  resolved = Grico._resolved_preconditioner(system, nothing)
  label = if Grico._preconditioner_is_applicable(system, resolved)
    string(Grico._preconditioner_label(resolved), " + ",
           Grico._krylov_method_name(Grico._preconditioned_krylov_style(system, resolved)))
  else
    "Sparse direct"
  end
  cold_times = Float64[]
  warm_times = Float64[]
  residual = zero(T)

  for _ in 1:repeats
    empty!(system.preconditioner_cache)
    cold_solution = nothing
    push!(cold_times, @elapsed cold_solution = Grico.solve(system))
    residual = Grico._relative_residual_norm(system.matrix, system.rhs,
                                             _reduced_solution(system, cold_solution))

    empty!(system.preconditioner_cache)
    Grico.solve(system)
    push!(warm_times, @elapsed Grico.solve(system))
  end

  return DefaultPolicyMeasurement(label, _median(cold_times), _median(warm_times), residual)
end

function _measure_direct(system::Grico.AffineSystem{T}, repeats::Int) where {T<:AbstractFloat}
  setup_times = Float64[]
  solve_times = Float64[]
  warm_solve_times = Float64[]
  residuals = Float64[]

  for _ in 1:repeats
    Grico._with_internal_blas_threads() do
      operator = nothing
      setup_seconds = @elapsed operator = Grico._build_ordered_direct_operator(system.matrix)
      reduced = zeros(T, size(system.matrix, 1))
      solve_seconds = @elapsed ldiv!(reduced, operator, system.rhs)
      warm_seconds = @elapsed ldiv!(reduced, operator, system.rhs)
      push!(setup_times, setup_seconds)
      push!(solve_times, solve_seconds)
      push!(warm_solve_times, warm_seconds)
      push!(residuals, Grico._relative_residual_norm(system.matrix, system.rhs, reduced))
    end
  end

  return SolverMeasurement("direct", "Sparse direct", _median(setup_times), _median(solve_times),
                           _median(warm_solve_times), _median(setup_times) + _median(solve_times),
                           1, true, _median(residuals))
end

function _measure_preconditioner(system::Grico.AffineSystem{T}, candidate::SolverCandidate,
                                 repeats::Int) where {T<:AbstractFloat}
  setup_times = Float64[]
  solve_times = Float64[]
  warm_solve_times = Float64[]
  iterations = Int[]
  converged = Bool[]
  residuals = Float64[]

  for _ in 1:repeats
    empty!(system.preconditioner_cache)
    Grico._with_internal_blas_threads() do
      preconditioner = candidate.build()
      operator = nothing
      setup_seconds = @elapsed operator = Grico._preconditioner_operator(system, preconditioner)
      style = Grico._preconditioned_krylov_style(system, preconditioner)
      reduced = nothing
      stats = nothing
      solve_seconds = @elapsed reduced, stats = _run_krylov(system, operator, style)
      warm_reduced = nothing
      warm_stats = nothing
      warm_seconds = @elapsed warm_reduced, warm_stats = _run_krylov(system, operator, style)

      push!(setup_times, setup_seconds)
      push!(solve_times, solve_seconds)
      push!(warm_solve_times, warm_seconds)
      push!(iterations, stats.niter)
      push!(converged, stats.solved)
      push!(residuals, Grico._relative_residual_norm(system.matrix, system.rhs, reduced))
      stats.solved == warm_stats.solved || push!(converged, warm_stats.solved)
    end
  end

  converged_value = all(identity, converged)
  return SolverMeasurement(candidate.id, candidate.label, _median(setup_times),
                           _median(solve_times), _median(warm_solve_times),
                           _median(setup_times) + _median(solve_times), _median(iterations),
                           converged_value, _median(residuals))
end

function _measure_case(case::SolverCase, repeats::Int)
  default_policy = _measure_default_policy(case.system, repeats)
  assemble_seconds = _assemble_seconds(case.plan, repeats)
  measurements = SolverMeasurement[]

  for candidate in case.candidates
    if candidate.id == "direct"
      push!(measurements, _measure_direct(case.system, repeats))
    else
      push!(measurements, _measure_preconditioner(case.system, candidate, repeats))
    end
  end

  return Dict{String,Any}("id" => case.id, "label" => case.label, "description" => case.description,
                          "metadata" => case.metadata, "assemble_seconds" => assemble_seconds,
                          "default_policy" => Dict("label" => default_policy.label,
                                                   "cold_seconds" => default_policy.cold_seconds,
                                                   "warm_seconds" => default_policy.warm_seconds,
                                                   "residual" => default_policy.residual),
                          "methods" => [Dict("id" => measurement.id, "label" => measurement.label,
                                             "setup_seconds" => measurement.setup_seconds,
                                             "solve_seconds" => measurement.solve_seconds,
                                             "warm_solve_seconds" => measurement.warm_solve_seconds,
                                             "total_seconds" => measurement.total_seconds,
                                             "iterations" => measurement.iterations,
                                             "converged" => measurement.converged,
                                             "residual" => measurement.residual)
                                        for measurement in measurements])
end

function _render_markdown(report)
  lines = String[]
  push!(lines, "# $(report["phase"])")
  push!(lines, "")
  push!(lines, "Generated: `$(report["generated_utc"])`")
  push!(lines, "")
  push!(lines, "## Environment")
  push!(lines, "")
  env = report["environment"]
  push!(lines, "- Julia: `$(env["julia_version"])`")
  push!(lines, "- Host: `$(env["hostname"])`")
  push!(lines, "- CPU: `$(env["cpu_model"])`")
  push!(lines, "- Julia threads: `$(env["julia_threads"])`")
  push!(lines, "- BLAS threads: `$(env["blas_threads"])`")
  push!(lines, "- `OPENBLAS_NUM_THREADS`: `$(env["openblas_num_threads"])`")
  push!(lines, "- `OMP_NUM_THREADS`: `$(env["omp_num_threads"])`")

  for case in report["cases"]
    push!(lines, "")
    push!(lines, "## `$(case["id"])`")
    push!(lines, "")
    push!(lines, case["description"])
    push!(lines, "")
    push!(lines, "- Reduced dofs: `$(case["metadata"]["reduced_dofs"])`")
    push!(lines, "- Matrix nnz: `$(case["metadata"]["matrix_nnz"])`")
    push!(lines, "- Symmetric: `$(case["metadata"]["symmetric"])`")
    push!(lines, "- Median assembly time: `$(_format_time(case["assemble_seconds"]))`")
    push!(lines, "- Default policy: `$(case["default_policy"]["label"])`")
    push!(lines, "- Default cold solve: `$(_format_time(case["default_policy"]["cold_seconds"]))`")
    push!(lines, "- Default warm solve: `$(_format_time(case["default_policy"]["warm_seconds"]))`")
    push!(lines, "")
    push!(lines,
          "| Method | Setup | Solve | Warm Solve | Cold Total | Assemble + Cold Total | Iter | Converged | Residual |")
    push!(lines, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")

    for method in case["methods"]
      total_with_assembly = case["assemble_seconds"] + method["total_seconds"]
      push!(lines,
            "| `$(method["label"])` | $(_format_time(method["setup_seconds"])) | $(_format_time(method["solve_seconds"])) | $(_format_time(method["warm_solve_seconds"])) | $(_format_time(method["total_seconds"])) | $(_format_time(total_with_assembly)) | $(method["iterations"]) | $(method["converged"] ? "yes" : "no") | $(@sprintf("%.2e", method["residual"])) |")
    end
  end

  return join(lines, "\n")
end

function _write_toml(path, data)
  mkpath(dirname(path))
  open(path, "w") do io
    TOML.print(io, data)
  end
  return path
end

function _report_path_for_output(output_path::AbstractString)
  basename(output_path) == output_path && return replace(output_path, r"\.toml$" => ".md")
  return joinpath(dirname(output_path), replace(basename(output_path), r"\.toml$" => ".md"))
end

function main(args=ARGS)
  parsed = _parse_args(args)
  repeats = parse(Int, get(parsed, "repeats", string(DEFAULT_REPEATS)))
  output_path = get(parsed, "output", joinpath(BENCHMARK_PROJECT, "phase7_solver_paths.toml"))
  report_path = get(parsed, "report", _report_path_for_output(output_path))
  phase_label = get(parsed, "phase-label", DEFAULT_PHASE_LABEL)

  BLAS.set_num_threads(1)
  cases = _build_cases()
  foreach(_warmup_case!, cases)
  report = Dict{String,Any}("phase" => phase_label,
                            "generated_utc" => Dates.format(Dates.now(Dates.UTC),
                                                            dateformat"yyyy-mm-ddTHH:MM:SS"),
                            "environment" => _environment_metadata(),
                            "cases" => [_measure_case(case, repeats) for case in cases])

  _write_toml(output_path, report)

  open(report_path, "w") do io
    write(io, _render_markdown(report))
  end

  println("wrote benchmark artifact to $output_path")
  println("wrote markdown report to $report_path")
  return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  main()
end

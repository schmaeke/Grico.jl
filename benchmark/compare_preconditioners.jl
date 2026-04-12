#!/usr/bin/env julia

const BENCHMARK_PROJECT = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_PROJECT, ".."))
pushfirst!(LOAD_PATH, BENCHMARK_PROJECT)
pushfirst!(LOAD_PATH, REPO_ROOT)

using LinearAlgebra
using Printf
using SparseArrays
using Grico
using IncompleteLU
using Krylov
import AlgebraicMultigrid: ruge_stuben, smoothed_aggregation, aspreconditioner, operator_complexity

const SAMPLE_ITERS = (1, 2, 3)
const FLOW_ROOT_COUNTS = (16, 16)
const ILU_TAUS = (1e-2, 1e-3, 1e-4)

struct SampledFlowSystem{P}
  step::Int
  system::Grico.AffineSystem{Float64,P}
  preconditioner::Grico.FieldSplitSchurPreconditioner
end

struct SolverMeasurement
  label::String
  setup_seconds::Float64
  solve_seconds::Float64
  total_seconds::Float64
  iterations::Int
  converged::Bool
  residual::Float64
  fill::Int
  complexity::Float64
end

# Load the cavity example as a provider of reusable setup helpers while keeping
# its normal top-level run path disabled inside this benchmark harness.
let previous = get(ENV, "GRICO_LDC_AUTORUN", nothing)
  ENV["GRICO_LDC_AUTORUN"] = "0"
  include(joinpath(REPO_ROOT, "examples", "lid_driven_cavity.jl"))

  if previous === nothing
    delete!(ENV, "GRICO_LDC_AUTORUN")
  else
    ENV["GRICO_LDC_AUTORUN"] = previous
  end
end

function _krylov_controls(system::Grico.AffineSystem)
  return (rtol=Grico._default_krylov_reltol(Float64), memory=Grico._default_gmres_restart(system),
          itmax=Grico._default_krylov_maxiter(system))
end

function _relative_residual(system::Grico.AffineSystem, values::AbstractVector{Float64})
  return Grico._relative_residual_norm(system.matrix, system.rhs, values)
end

function _warmup_measurements(sample::SampledFlowSystem)
  _measure_package_preconditioner(sample.system, sample.preconditioner, "warmup")
  _measure_package_preconditioner(sample.system, Grico.AdditiveSchwarzPreconditioner(), "warmup")
  _measure_ilu(sample.system, ILU_TAUS[1])
  _measure_amg(sample.system, ruge_stuben, "warmup")
  _measure_amg(sample.system, smoothed_aggregation, "warmup")
  return nothing
end

function _measure_package_preconditioner(system::Grico.AffineSystem, preconditioner, label::String)
  controls = _krylov_controls(system)
  empty!(system.preconditioner_cache)
  operator = nothing
  solution = zeros(Float64, size(system.matrix, 1))
  stats = nothing
  setup_seconds = @elapsed operator = Grico._preconditioner_operator(system, preconditioner)
  solve_seconds = @elapsed begin
    solution, stats = gmres(system.matrix, system.rhs; N=operator, ldiv=true, rtol=controls.rtol,
                            restart=true, memory=controls.memory, itmax=controls.itmax)
  end
  return SolverMeasurement(label, setup_seconds, solve_seconds, setup_seconds + solve_seconds,
                           stats.niter, stats.solved, _relative_residual(system, solution), 0, 0.0)
end

function _measure_ilu(system::Grico.AffineSystem, τ::Float64)
  controls = _krylov_controls(system)
  ordered_matrix = system.matrix[system.ordering, system.ordering]
  ordered_rhs = system.rhs[system.ordering]
  factor = nothing
  ordered_solution = zeros(Float64, size(ordered_matrix, 1))
  stats = nothing
  setup_seconds = @elapsed factor = ilu(ordered_matrix; τ=τ)
  solve_seconds = @elapsed begin
    ordered_solution, stats = gmres(ordered_matrix, ordered_rhs; N=factor, ldiv=true,
                                    rtol=controls.rtol, restart=true, memory=controls.memory,
                                    itmax=controls.itmax)
  end
  solution = ordered_solution[system.inverse_ordering]
  fill = nnz(factor.L) + nnz(factor.U)
  label = @sprintf("ILU(τ=%.0e)", τ)
  return SolverMeasurement(label, setup_seconds, solve_seconds, setup_seconds + solve_seconds,
                           stats.niter, stats.solved, _relative_residual(system, solution), fill,
                           0.0)
end

function _measure_direct(system::Grico.AffineSystem)
  solution = zeros(Float64, size(system.matrix, 1))
  total_seconds = @elapsed solution = Grico._default_system_direct_solve(system)
  return SolverMeasurement("Sparse direct", 0.0, total_seconds, total_seconds, 1, true,
                           _relative_residual(system, solution), nnz(system.matrix), 0.0)
end

function _measure_amg(system::Grico.AffineSystem, builder, label::String)
  controls = _krylov_controls(system)
  hierarchy = nothing
  preconditioner = nothing
  solution = zeros(Float64, size(system.matrix, 1))
  stats = nothing
  setup_seconds = @elapsed begin
    hierarchy = builder(system.matrix)
    preconditioner = aspreconditioner(hierarchy)
  end
  solve_seconds = @elapsed begin
    solution, stats = gmres(system.matrix, system.rhs; N=preconditioner, ldiv=true,
                            rtol=controls.rtol, restart=true, memory=controls.memory,
                            itmax=controls.itmax)
  end
  return SolverMeasurement(label, setup_seconds, solve_seconds, setup_seconds + solve_seconds,
                           stats.niter, stats.solved, _relative_residual(system, solution), 0,
                           operator_complexity(hierarchy))
end

function sample_flow_systems(sample_steps::Tuple{Vararg{Int}})
  isempty(sample_steps) && throw(ArgumentError("sample_steps must not be empty"))
  issorted(collect(sample_steps)) || throw(ArgumentError("sample_steps must be sorted"))
  first(sample_steps) >= 1 || throw(ArgumentError("sample_steps must be positive"))

  context = build_lid_driven_cavity_context(root_counts=FLOW_ROOT_COUNTS)
  samples = SampledFlowSystem[]

  for step in 1:last(sample_steps)
    flow_preconditioner = FieldSplitSchurPreconditioner((context.velocity,), (context.pressure,))
    context, system, _, _, _ = advance_picard_step(context)
    step in sample_steps && push!(samples, SampledFlowSystem(step, system, flow_preconditioner))
  end

  return samples
end

function compare_on_sample(sample::SampledFlowSystem)
  return vcat([_measure_direct(sample.system),
               _measure_package_preconditioner(sample.system, sample.preconditioner,
                                               "FieldSplitSchur + GMRES"),
               _measure_package_preconditioner(sample.system, Grico.AdditiveSchwarzPreconditioner(),
                                               "AdditiveSchwarz + GMRES"),
               _measure_amg(sample.system, ruge_stuben, "RugeStuben AMG + GMRES"),
               _measure_amg(sample.system, smoothed_aggregation, "SmoothedAgg AMG + GMRES")],
              [_measure_ilu(sample.system, τ) for τ in ILU_TAUS])
end

function _print_measurement(step::Int, system::Grico.AffineSystem, measurement::SolverMeasurement)
  fill_text = measurement.fill == 0 ? "-" : string(measurement.fill)
  complexity_text = measurement.complexity == 0 ? "-" : @sprintf("%.2f", measurement.complexity)
  @printf("  %-24s %7d %9.3f %9.3f %9.3f %7d %9.2e %8s %8s %7s\n", measurement.label,
          size(system.matrix, 1), measurement.setup_seconds, measurement.solve_seconds,
          measurement.total_seconds, measurement.iterations, measurement.residual,
          measurement.converged ? "yes" : "no", fill_text, complexity_text)
  return nothing
end

function main()
  samples = sample_flow_systems(SAMPLE_ITERS)
  isempty(samples) && error("no sample systems collected")
  _warmup_measurements(first(samples))

  println("lid_driven_cavity preconditioner comparison")
  @printf("  sample Picard iterations : %s\n", SAMPLE_ITERS)
  @printf("  ilu taus     : %s\n", ILU_TAUS)
  println("  solver                    dofs     setup s   solve s   total s   iter   residual converged     fill   opcmp")

  for sample in samples
    println()
    @printf("step %d\n", sample.step)

    for measurement in compare_on_sample(sample)
      _print_measurement(sample.step, sample.system, measurement)
    end
  end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  main()
end

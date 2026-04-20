#!/usr/bin/env julia

const BENCHMARK_PROJECT = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_PROJECT, ".."))
pushfirst!(LOAD_PATH, BENCHMARK_PROJECT)
pushfirst!(LOAD_PATH, REPO_ROOT)

get!(ENV, "KMP_DUPLICATE_LIB_OK", "TRUE")
get!(ENV, "OMP_NUM_THREADS", "1")
get!(ENV, "OPENBLAS_NUM_THREADS", "1")

using LinearAlgebra
using Printf
using SparseArrays
using Grico
using IncompleteLU
using Krylov
using MPI
import HYPRE
import MUMPS
import AlgebraicMultigrid: ruge_stuben, smoothed_aggregation, aspreconditioner, operator_complexity
import Grico: cell_matrix!, cell_rhs!

const FLOW_SAMPLE_ITERS = (1, 2, 3)
const FLOW_ROOT_COUNTS = (16, 16)
const SPD_ROOT_COUNTS = (48, 48)
const SPD_DEGREE = 2
const WHOLE_ILU_TAUS = (1e-3, 1e-4)
const SCHWARZ_ALWAYS = Grico.AdditiveSchwarzPreconditioner(min_dofs=0)
const HYPRE_AMG_OPTIONS = (; CoarsenType=6, OldDefault=true, RelaxType=6, NumSweeps=1, PrintLevel=0)

MPI.Initialized() || MPI.Init()
HYPRE.Init(finalize_atexit=false)
BLAS.set_num_threads(1)

struct Measurement
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

struct FlowSample{SF<:Grico.AffineSystem}
  step::Int
  flow_system::SF
  flow_preconditioner::Grico.FieldSplitSchurPreconditioner
end

mutable struct OrderedFactorOperator{T<:AbstractFloat,F}
  factor::F
  ordering::Vector{Int}
  inverse_ordering::Vector{Int}
  ordered_rhs::Vector{T}
  ordered_solution::Vector{T}
  apply_rhs::Vector{T}
end

function LinearAlgebra.ldiv!(result::AbstractVector{T}, operator::OrderedFactorOperator{T},
                             rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  length(result) == length(rhs_data) == length(operator.apply_rhs) ||
    throw(ArgumentError("ordered factor dimensions must match the system"))
  Grico._gather_entries!(operator.ordered_rhs, rhs_data, operator.ordering)
  ldiv!(operator.ordered_solution, operator.factor, operator.ordered_rhs)
  Grico._gather_entries!(result, operator.ordered_solution, operator.inverse_ordering)
  return result
end

function LinearAlgebra.ldiv!(operator::OrderedFactorOperator{T},
                             rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  length(rhs_data) == length(operator.apply_rhs) ||
    throw(ArgumentError("ordered factor dimensions must match the system"))
  copyto!(operator.apply_rhs, rhs_data)
  ldiv!(rhs_data, operator, operator.apply_rhs)
  return rhs_data
end

function build_ordered_ilu_operator(matrix_data::SparseMatrixCSC{T,Int};
                                    τ::Float64) where {T<:AbstractFloat}
  ordering, inverse_ordering = Grico._solve_ordering(matrix_data)
  ordered_matrix = matrix_data[ordering, ordering]
  factor = ilu(ordered_matrix; τ=τ)
  operator = OrderedFactorOperator(factor, ordering, inverse_ordering,
                                   zeros(T, size(matrix_data, 1)), zeros(T, size(matrix_data, 1)),
                                   zeros(T, size(matrix_data, 1)))
  return operator, nnz(factor.L) + nnz(factor.U)
end

# Load the benchmark-facing wrapper for the cavity example. The instructional
# driver remains focused on the example and only runs when invoked as a script.
include(joinpath(REPO_ROOT, "examples", "lid_driven_cavity", "benchmarking.jl"))

struct Diffusion{F}
  field::F
end

function cell_matrix!(local_matrix, operator::Diffusion, values::CellValues)
  dofs = field_dof_range(values, operator.field)
  local_block = view(local_matrix, dofs, dofs)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = values.weights[point_index]

    for row_mode in 1:mode_count
      for col_mode in 1:row_mode
        contribution = zero(eltype(local_matrix))

        for axis in 1:axis_count
          contribution += gradients[axis, row_mode, point_index] *
                          gradients[axis, col_mode, point_index]
        end

        contribution *= weighted
        local_block[row_mode, col_mode] += contribution
        row_mode == col_mode || (local_block[col_mode, row_mode] += contribution)
      end
    end
  end

  return nothing
end

struct Source{F,G}
  field::F
  data::G
end

function cell_rhs!(local_rhs, operator::Source, values::CellValues)
  local_block = view(local_rhs, field_dof_range(values, operator.field))
  shape_table = shape_values(values, operator.field)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = operator.data(values.points[point_index]) * values.weights[point_index]

    for mode_index in 1:mode_count
      local_block[mode_index] += shape_table[mode_index, point_index] * weighted
    end
  end

  return nothing
end

poisson_exact_solution(x) = sinpi(x[1]) * sinpi(x[2])
poisson_source_term(x) = 2 * pi^2 * poisson_exact_solution(x)

function build_spd_problem(u)
  problem = AffineProblem(u)
  add_cell!(problem, Diffusion(u))
  add_cell!(problem, Source(u, poisson_source_term))

  for axis in 1:2, side in (LOWER, UPPER)
    add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, side), poisson_exact_solution))
  end

  return problem
end

function build_spd_case()
  space = HpSpace(Domain((0.0, 0.0), (1.0, 1.0), SPD_ROOT_COUNTS),
                  SpaceOptions(degree=UniformDegree(SPD_DEGREE)))
  u = ScalarField(space; name=:u)
  plan = compile(build_spd_problem(u))
  system = assemble(plan)
  return (; field=u, plan, system)
end

function krylov_controls(system::Grico.AffineSystem)
  return (rtol=Grico._default_krylov_reltol(Float64), memory=Grico._default_gmres_restart(system),
          itmax=Grico._default_krylov_maxiter(system))
end

function relative_residual(system::Grico.AffineSystem, values::AbstractVector{Float64})
  return Grico._relative_residual_norm(system.matrix, system.rhs, values)
end

function measure_direct(system::Grico.AffineSystem; label::String="Sparse direct")
  solution = zeros(Float64, size(system.matrix, 1))
  total_seconds = @elapsed solution = Grico._default_system_direct_solve(system)
  return Measurement(label, 0.0, total_seconds, total_seconds, 1, true,
                     relative_residual(system, solution), nnz(system.matrix), 0.0)
end

function measure_cg(system::Grico.AffineSystem, label::String, builder::Function)
  controls = krylov_controls(system)
  preconditioner = nothing
  fill = 0
  complexity = 0.0
  solution = zeros(Float64, size(system.matrix, 1))
  stats = nothing
  setup_seconds = @elapsed begin
    preconditioner, fill, complexity = builder(system)
  end
  solve_seconds = @elapsed begin
    solution, stats = cg(Symmetric(system.matrix), system.rhs; M=preconditioner, ldiv=true,
                         rtol=controls.rtol, itmax=controls.itmax)
  end
  return Measurement(label, setup_seconds, solve_seconds, setup_seconds + solve_seconds,
                     stats.niter, stats.solved, relative_residual(system, solution), fill,
                     complexity)
end

function measure_gmres(system::Grico.AffineSystem, label::String, builder::Function)
  controls = krylov_controls(system)
  preconditioner = nothing
  fill = 0
  complexity = 0.0
  solution = zeros(Float64, size(system.matrix, 1))
  stats = nothing
  setup_seconds = @elapsed begin
    preconditioner, fill, complexity = builder(system)
  end
  solve_seconds = @elapsed begin
    solution, stats = gmres(system.matrix, system.rhs; N=preconditioner, ldiv=true,
                            rtol=controls.rtol, restart=true, memory=controls.memory,
                            itmax=controls.itmax)
  end
  return Measurement(label, setup_seconds, solve_seconds, setup_seconds + solve_seconds,
                     stats.niter, stats.solved, relative_residual(system, solution), fill,
                     complexity)
end

function hypre_converged(system::Grico.AffineSystem, solution::Vector{Float64},
                         controls)::Tuple{Bool,Float64}
  residual = relative_residual(system, solution)
  tolerance = max(10 * controls.rtol, sqrt(eps(Float64)))
  return residual <= tolerance, residual
end

function build_hypre_context(system::Grico.AffineSystem)
  matrix_data = HYPRE.HYPREMatrix(system.matrix)
  rhs_data = HYPRE.HYPREVector(system.rhs)
  solution_data = HYPRE.HYPREVector(zeros(Float64, size(system.matrix, 1)))
  return matrix_data, rhs_data, solution_data
end

function finalize_if_possible!(object)
  object === nothing && return nothing

  try
    finalize(object)
  catch
  end

  return nothing
end

function measure_hypre(system::Grico.AffineSystem, label::String, builder::Function)
  controls = krylov_controls(system)
  solver = nothing
  matrix_data = nothing
  rhs_data = nothing
  solution_data = nothing
  solution = zeros(Float64, size(system.matrix, 1))
  fill = 0
  complexity = 0.0

  setup_seconds = @elapsed begin
    solver, matrix_data, rhs_data, solution_data, fill, complexity = builder(system, controls)
    HYPRE.Internals.setup_func(solver)(solver, matrix_data, rhs_data, solution_data)
  end

  solve_seconds = @elapsed begin
    HYPRE.Internals.solve_func(solver)(solver, matrix_data, rhs_data, solution_data)
    copy!(solution, solution_data)
  end

  converged, residual = hypre_converged(system, solution, controls)
  iterations = HYPRE.GetNumIterations(solver)
  measurement = Measurement(label, setup_seconds, solve_seconds, setup_seconds + solve_seconds,
                            iterations, converged, residual, fill, complexity)

  if hasproperty(solver, :precond)
    finalize_if_possible!(getproperty(solver, :precond))
  end

  finalize_if_possible!(solution_data)
  finalize_if_possible!(rhs_data)
  finalize_if_possible!(matrix_data)
  finalize_if_possible!(solver)
  return measurement
end

function measure_mumps(system::Grico.AffineSystem; label::String="MUMPS direct",
                       sym::Integer=MUMPS.mumps_unsymmetric)
  mumps = nothing
  solution = zeros(Float64, size(system.matrix, 1))
  fill = 0
  icntl = MUMPS.get_icntl(verbose=false)
  cntl = copy(MUMPS.default_cntl64)

  setup_seconds = @elapsed begin
    mumps = MUMPS.Mumps{Float64}(sym, icntl, cntl)
    MUMPS.factorize!(mumps, system.matrix)
    fill = Int(mumps.infog[29])
  end

  solve_seconds = @elapsed begin
    ldiv!(solution, mumps, system.rhs)
  end

  measurement = Measurement(label, setup_seconds, solve_seconds, setup_seconds + solve_seconds, 1,
                            true, relative_residual(system, solution), fill, 0.0)
  finalize_if_possible!(mumps)
  return measurement
end

function build_hypre_pcg_amg(system::Grico.AffineSystem, controls)
  preconditioner = HYPRE.BoomerAMG(; HYPRE_AMG_OPTIONS..., Tol=0.0, MaxIter=1)
  solver = HYPRE.PCG(; Tol=controls.rtol, MaxIter=controls.itmax, TwoNorm=1, PrintLevel=0,
                     Logging=1, Precond=preconditioner)
  matrix_data, rhs_data, solution_data = build_hypre_context(system)
  return solver, matrix_data, rhs_data, solution_data, 0, 0.0
end

function build_hypre_flexgmres_amg(system::Grico.AffineSystem, controls)
  preconditioner = HYPRE.BoomerAMG(; HYPRE_AMG_OPTIONS..., Tol=0.0, MaxIter=1)
  solver = HYPRE.FlexGMRES(; Tol=controls.rtol, MaxIter=controls.itmax, KDim=controls.memory,
                           PrintLevel=0, Logging=1, Precond=preconditioner)
  matrix_data, rhs_data, solution_data = build_hypre_context(system)
  return solver, matrix_data, rhs_data, solution_data, 0, 0.0
end

function build_hypre_gmres_ilu(system::Grico.AffineSystem, controls)
  preconditioner = HYPRE.ILU(; PrintLevel=0)
  solver = HYPRE.GMRES(; Tol=controls.rtol, MaxIter=controls.itmax, KDim=controls.memory,
                       PrintLevel=0, Logging=1, Precond=preconditioner)
  matrix_data, rhs_data, solution_data = build_hypre_context(system)
  return solver, matrix_data, rhs_data, solution_data, 0, 0.0
end

function build_whole_schwarz(system::Grico.AffineSystem)
  empty!(system.preconditioner_cache)
  return Grico._preconditioner_operator(system, SCHWARZ_ALWAYS), 0, 0.0
end

function build_whole_rs_amg(system::Grico.AffineSystem)
  hierarchy = ruge_stuben(system.matrix)
  return aspreconditioner(hierarchy), 0, operator_complexity(hierarchy)
end

function build_whole_sa_amg(system::Grico.AffineSystem)
  hierarchy = smoothed_aggregation(system.matrix)
  return aspreconditioner(hierarchy), 0, operator_complexity(hierarchy)
end

function build_whole_ilu(system::Grico.AffineSystem, τ::Float64)
  operator, fill = build_ordered_ilu_operator(system.matrix; τ=τ)
  return operator, fill, 0.0
end

function build_block_primary(matrix_data::SparseMatrixCSC{T,Int},
                             topology::Grico._PatchSolveTopology{T},
                             kind::Symbol) where {T<:AbstractFloat}
  if kind == :schwarz
    operator = Grico._build_primary_block_operator(matrix_data, topology, SCHWARZ_ALWAYS)
    return operator, 0, 0.0
  elseif kind == :sa_amg
    hierarchy = smoothed_aggregation(matrix_data)
    return aspreconditioner(hierarchy), 0, operator_complexity(hierarchy)
  elseif kind == :direct
    return Grico._build_ordered_direct_operator(matrix_data), 0, 0.0
  else
    throw(ArgumentError("unsupported primary block kind $kind"))
  end
end

function build_block_schur(matrix_data::SparseMatrixCSC{T,Int},
                           kind::Symbol) where {T<:AbstractFloat}
  if kind == :direct
    return Grico._build_ordered_direct_operator(matrix_data), 0, 0.0
  elseif kind == :sa_amg
    hierarchy = smoothed_aggregation(matrix_data)
    return aspreconditioner(hierarchy), 0, operator_complexity(hierarchy)
  elseif kind == :ilu_1e3
    operator, fill = build_ordered_ilu_operator(matrix_data; τ=1e-3)
    return operator, fill, 0.0
  elseif kind == :ilu_1e4
    operator, fill = build_ordered_ilu_operator(matrix_data; τ=1e-4)
    return operator, fill, 0.0
  else
    throw(ArgumentError("unsupported Schur block kind $kind"))
  end
end

function build_custom_field_split(system::Grico.AffineSystem,
                                  preconditioner::Grico.FieldSplitSchurPreconditioner;
                                  primary_kind::Symbol, schur_kind::Symbol)
  partition = Grico._field_split_partition(system, preconditioner)
  primary_indices = partition.primary_indices
  schur_indices = partition.schur_indices
  primary_matrix = system.matrix[primary_indices, primary_indices]
  coupling12 = system.matrix[primary_indices, schur_indices]
  coupling21 = system.matrix[schur_indices, primary_indices]
  schur_block = system.matrix[schur_indices, schur_indices]
  primary_topology = Grico._restricted_solve_topology(system.solve_topology, primary_indices)
  primary_operator, primary_fill, primary_complexity = build_block_primary(primary_matrix,
                                                                           primary_topology,
                                                                           primary_kind)
  schur_matrix = Grico._approximate_schur_matrix(primary_matrix, coupling12, coupling21,
                                                 schur_block)
  schur_operator, schur_fill, schur_complexity = build_block_schur(schur_matrix, schur_kind)
  T = Float64
  operator = Grico._FieldSplitSchurOperator(copy(primary_indices), copy(schur_indices), coupling21,
                                            primary_operator, schur_operator,
                                            zeros(T, length(primary_indices)),
                                            zeros(T, length(primary_indices)),
                                            zeros(T, length(schur_indices)),
                                            zeros(T, length(schur_indices)),
                                            zeros(T, size(system.matrix, 1)))
  return operator, primary_fill + schur_fill, primary_complexity + schur_complexity
end

function block_builder(preconditioner::Grico.FieldSplitSchurPreconditioner; primary_kind::Symbol,
                       schur_kind::Symbol)
  return system -> build_custom_field_split(system, preconditioner; primary_kind=primary_kind,
                                            schur_kind=schur_kind)
end

function sample_lid_driven_cavity(sample_steps::Tuple{Vararg{Int}})
  isempty(sample_steps) && throw(ArgumentError("sample_steps must not be empty"))
  issorted(collect(sample_steps)) || throw(ArgumentError("sample_steps must be sorted"))
  first(sample_steps) >= 1 || throw(ArgumentError("sample_steps must be positive"))

  context = build_lid_driven_cavity_context(root_counts=FLOW_ROOT_COUNTS)
  samples = FlowSample[]

  for step in 1:last(sample_steps)
    flow_preconditioner = FieldSplitSchurPreconditioner((context.velocity,), (context.pressure,))
    context, flow_system, _, _, _ = advance_picard_step(context)

    if step in sample_steps
      push!(samples, FlowSample(step, flow_system, flow_preconditioner))
    end
  end

  return samples
end

function format_fill(fill::Int)
  fill == 0 && return "-"
  return string(fill)
end

function format_complexity(complexity::Float64)
  complexity == 0 && return "-"
  return @sprintf("%.2f", complexity)
end

function print_measurement(measurement::Measurement, system::Grico.AffineSystem)
  @printf("  %-28s %7d %9.3f %9.3f %9.3f %7d %9.2e %8s %9s %7s\n", measurement.label,
          size(system.matrix, 1), measurement.setup_seconds, measurement.solve_seconds,
          measurement.total_seconds, measurement.iterations, measurement.residual,
          measurement.converged ? "yes" : "no", format_fill(measurement.fill),
          format_complexity(measurement.complexity))
  return nothing
end

function print_header()
  println("  solver                         dofs     setup s   solve s   total s   iter   residual converged      fill   opcmp")
  return nothing
end

function warmup!(spd_case, flow_samples)
  spd_system = spd_case.system
  first_sample = first(flow_samples)
  measure_direct(spd_system)
  measure_mumps(spd_system; label="warmup", sym=MUMPS.mumps_definite)
  measure_cg(spd_system, "warmup", build_whole_schwarz)
  measure_cg(spd_system, "warmup", build_whole_rs_amg)
  measure_cg(spd_system, "warmup", build_whole_sa_amg)
  measure_hypre(spd_system, "warmup", build_hypre_pcg_amg)

  measure_direct(first_sample.flow_system)
  measure_mumps(first_sample.flow_system; label="warmup", sym=MUMPS.mumps_unsymmetric)
  measure_gmres(first_sample.flow_system, "warmup",
                system -> (empty!(system.preconditioner_cache);
                           (Grico._preconditioner_operator(system,
                                                           first_sample.flow_preconditioner), 0,
                            0.0)))
  measure_gmres(first_sample.flow_system, "warmup",
                system -> build_whole_ilu(system, WHOLE_ILU_TAUS[1]))
  measure_gmres(first_sample.flow_system, "warmup",
                system -> build_whole_ilu(system, WHOLE_ILU_TAUS[2]))
  measure_gmres(first_sample.flow_system, "warmup", build_whole_sa_amg)
  measure_gmres(first_sample.flow_system, "warmup",
                block_builder(first_sample.flow_preconditioner; primary_kind=:sa_amg,
                              schur_kind=:direct))
  measure_gmres(first_sample.flow_system, "warmup",
                block_builder(first_sample.flow_preconditioner; primary_kind=:schwarz,
                              schur_kind=:ilu_1e3))
  measure_gmres(first_sample.flow_system, "warmup",
                block_builder(first_sample.flow_preconditioner; primary_kind=:schwarz,
                              schur_kind=:ilu_1e4))
  measure_gmres(first_sample.flow_system, "warmup",
                block_builder(first_sample.flow_preconditioner; primary_kind=:schwarz,
                              schur_kind=:sa_amg))
  measure_hypre(first_sample.flow_system, "warmup", build_hypre_gmres_ilu)
  measure_hypre(first_sample.flow_system, "warmup", build_hypre_flexgmres_amg)
  return nothing
end

function run_spd_experiments(spd_case)
  println()
  println("SPD scalar diffusion")
  @printf("  roots                 : %s\n", SPD_ROOT_COUNTS)
  @printf("  degree                : %d\n", SPD_DEGREE)
  @printf("  matrix symmetry flag  : %s\n", spd_case.system.symmetric)
  print_header()

  measurements = [measure_direct(spd_case.system),
                  measure_mumps(spd_case.system; sym=MUMPS.mumps_definite),
                  measure_cg(spd_case.system, "AdditiveSchwarz + CG", build_whole_schwarz),
                  measure_cg(spd_case.system, "RugeStuben AMG + CG", build_whole_rs_amg),
                  measure_cg(spd_case.system, "SmoothedAgg AMG + CG", build_whole_sa_amg),
                  measure_hypre(spd_case.system, "HYPRE BoomerAMG + PCG", build_hypre_pcg_amg)]

  for measurement in measurements
    print_measurement(measurement, spd_case.system)
  end

  return nothing
end

function run_flow_experiments(flow_samples)
  println()
  println("Mixed lid-driven cavity flow")
  @printf("  sampled Picard iterations : %s\n", FLOW_SAMPLE_ITERS)
  print_header()

  for sample in flow_samples
    println()
    @printf("iteration %d\n", sample.step)
    measurements = [measure_direct(sample.flow_system),
                    measure_mumps(sample.flow_system; sym=MUMPS.mumps_unsymmetric),
                    measure_gmres(sample.flow_system, "FieldSplitSchur + GMRES",
                                  system -> (empty!(system.preconditioner_cache);
                                             (Grico._preconditioner_operator(system,
                                                                             sample.flow_preconditioner),
                                              0, 0.0))),
                    measure_gmres(sample.flow_system, "SmoothedAgg AMG + GMRES",
                                  build_whole_sa_amg),
                    measure_gmres(sample.flow_system,
                                  @sprintf("ILU(τ=%.0e) + GMRES", WHOLE_ILU_TAUS[1]),
                                  system -> build_whole_ilu(system, WHOLE_ILU_TAUS[1])),
                    measure_gmres(sample.flow_system,
                                  @sprintf("ILU(τ=%.0e) + GMRES", WHOLE_ILU_TAUS[2]),
                                  system -> build_whole_ilu(system, WHOLE_ILU_TAUS[2])),
                    measure_gmres(sample.flow_system, "Split(primary AMG, Schur direct)",
                                  block_builder(sample.flow_preconditioner; primary_kind=:sa_amg,
                                                schur_kind=:direct)),
                    measure_gmres(sample.flow_system, "Split(primary Schwarz, Schur AMG)",
                                  block_builder(sample.flow_preconditioner; primary_kind=:schwarz,
                                                schur_kind=:sa_amg)),
                    measure_gmres(sample.flow_system, "Split(primary Schwarz, Schur ILU1e-3)",
                                  block_builder(sample.flow_preconditioner; primary_kind=:schwarz,
                                                schur_kind=:ilu_1e3)),
                    measure_gmres(sample.flow_system, "Split(primary Schwarz, Schur ILU1e-4)",
                                  block_builder(sample.flow_preconditioner; primary_kind=:schwarz,
                                                schur_kind=:ilu_1e4)),
                    measure_hypre(sample.flow_system, "HYPRE GMRES + ILU", build_hypre_gmres_ilu),
                    measure_hypre(sample.flow_system, "HYPRE FlexGMRES + AMG",
                                  build_hypre_flexgmres_amg)]

    for measurement in measurements
      print_measurement(measurement, sample.flow_system)
    end
  end

  return nothing
end

function main()
  spd_case = build_spd_case()
  flow_samples = sample_lid_driven_cavity(FLOW_SAMPLE_ITERS)
  warmup!(spd_case, flow_samples)
  println("solver path experiments")
  run_spd_experiments(spd_case)
  run_flow_experiments(flow_samples)
  return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  main()
end

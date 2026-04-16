#!/usr/bin/env julia

const BENCHMARK_PROJECT = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_PROJECT, ".."))
pushfirst!(LOAD_PATH, BENCHMARK_PROJECT)
pushfirst!(LOAD_PATH, REPO_ROOT)

using BenchmarkTools
using LinearAlgebra
using Printf
using Grico
import Grico: cell_matrix!, cell_rhs!

const DIMENSION = 2
const INITIAL_DEGREE = 2
const SMALL_STEP = 2
const LARGE_STEP = 6
const STRESS_STEP = 8
const ADAPTIVITY_TOLERANCE = 5.0e-2
const MAX_DEGREE = 4
const MAX_H_LEVEL = 5
const SINGULAR_EXPONENT = 0.5
const VERIFICATION_EXTRA_POINTS = 2
const STRESS_ROOT_COUNTS = ntuple(_ -> 8, DIMENSION)
const STRESS_REFINEMENT_PASSES = 6
const STRESS_SEGMENT_ROOT_COUNTS = (48, 48)
const STRESS_SEGMENT_COUNT = 384

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

radius(x) = sqrt(sum(abs2, x))
const SOURCE_FACTOR = -SINGULAR_EXPONENT * (SINGULAR_EXPONENT + DIMENSION - 2)
exact_solution(x) = (r=radius(x); r == 0.0 ? 0.0 : r^SINGULAR_EXPONENT)
source_term(x) = (r=radius(x); r == 0.0 ? 0.0 : SOURCE_FACTOR * r^(SINGULAR_EXPONENT - 2))

function build_space()
  return HpSpace(Domain(ntuple(_ -> 0.0, DIMENSION), ntuple(_ -> 1.0, DIMENSION),
                        ntuple(_ -> 1, DIMENSION)),
                 SpaceOptions(degree=UniformDegree(INITIAL_DEGREE)))
end

function build_problem(u)
  problem = AffineProblem(u)
  add_cell!(problem, Diffusion(u))
  add_cell!(problem, Source(u, source_term))

  for axis in 1:DIMENSION, side in (LOWER, UPPER)
    add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, side), exact_solution))
  end

  return problem
end

function workflow_case(step_target::Int)
  step_target >= 0 || throw(ArgumentError("step_target must be non-negative"))
  space = build_space()
  u = ScalarField(space; name=:u)

  for step in 0:step_target
    problem = build_problem(u)
    plan = compile(problem)
    system = assemble(plan)
    state = State(plan, solve(system))
    step == step_target && return (; field=u, problem, plan, system, state)

    limits = AdaptivityLimits(field_space(u); max_p=MAX_DEGREE, max_h_level=MAX_H_LEVEL)
    next_plan = adaptivity_plan(state, u; tolerance=ADAPTIVITY_TOLERANCE, limits=limits)
    isempty(next_plan) && return (; field=u, problem, plan, system, state)
    u = adapted_field(transition(next_plan), u)
  end

  error("unreachable")
end

function embedded_case()
  domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  classifier = x -> x[1] + x[2] - 0.85
  return (; domain, classifier, leaf=1)
end

function transition_case(step_target::Int)
  case = workflow_case(step_target)
  limits = AdaptivityLimits(field_space(case.field); max_p=MAX_DEGREE, max_h_level=MAX_H_LEVEL)
  next_plan = adaptivity_plan(case.state, case.field; tolerance=ADAPTIVITY_TOLERANCE, limits=limits)
  return (; case..., adaptivity_plan=next_plan)
end

function _uniform_refine!(grid::CartesianGrid{D}, refinement_passes::Int) where {D}
  refinement_passes >= 0 || throw(ArgumentError("refinement_passes must be non-negative"))

  for pass in 1:refinement_passes
    axis = mod1(pass, D)

    for leaf in active_leaves(grid)
      refine!(grid, leaf, axis)
    end
  end

  return grid
end

function uniform_grid_case(root_counts::NTuple{D,Int}, refinement_passes::Int) where {D}
  return _uniform_refine!(CartesianGrid(root_counts), refinement_passes)
end

function uniform_space_case(root_counts::NTuple{D,Int}, refinement_passes::Int) where {D}
  domain = Domain(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D), root_counts)
  _uniform_refine!(grid(domain), refinement_passes)
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(INITIAL_DEGREE)))
  field = ScalarField(space; name=:u)
  return (; space, field)
end

function transition_case(space::HpSpace{D}) where {D}
  adaptivity_plan = AdaptivityPlan(space)

  for (leaf_index, leaf) in enumerate(active_leaves(space))
    request_h_refinement!(adaptivity_plan, leaf, mod1(leaf_index, D))
  end

  return (; space, adaptivity_plan)
end

function topology_case(level_count::Int)
  grid = CartesianGrid((4, 4))

  for level_index in 1:level_count
    leaves = active_leaves(grid)
    axis = isodd(level_index) ? 1 : 2

    for (leaf_index, leaf) in enumerate(leaves)
      isodd(leaf_index + level_index) || continue
      refine!(grid, leaf, axis)
    end
  end

  return grid
end

function segment_surface_case(root_counts::NTuple{2,Int}, segment_count::Int; point_count::Int=3)
  domain = Domain((0.0, 0.0), (1.0, 1.0), root_counts)
  radius = 0.32
  center = (0.5, 0.5)
  points = [ntuple(axis -> center[axis] +
                           radius * (axis == 1 ? cospi(2 * (index - 1) / segment_count) :
                                     sinpi(2 * (index - 1) / segment_count)), 2)
            for index in 1:segment_count]
  segments = [(index, index == segment_count ? 1 : index + 1) for index in 1:segment_count]
  surface = EmbeddedSurface(SegmentMesh(points, segments); point_count=point_count)
  return (; domain, surface)
end

function build_suite()
  suite = BenchmarkGroup()

  suite["core"] = BenchmarkGroup()
  suite["core"]["legendre_small"] = @benchmarkable legendre_values(0.25, 8)
  suite["core"]["legendre_large"] = @benchmarkable legendre_values(0.25, 32)
  suite["core"]["gauss_small"] = @benchmarkable gauss_legendre_rule(8)
  suite["core"]["gauss_large"] = @benchmarkable gauss_legendre_rule(32)
  suite["core"]["basis_small"] = @benchmarkable collect(basis_modes(TrunkBasis(), (4, 4)))
  suite["core"]["basis_large"] = @benchmarkable collect(basis_modes(TrunkBasis(), (6, 6, 6)))

  small = workflow_case(SMALL_STEP)
  large = workflow_case(LARGE_STEP)
  stress = workflow_case(STRESS_STEP)
  small_transition = transition_case(SMALL_STEP)
  large_transition = transition_case(LARGE_STEP)
  stress_setup = uniform_space_case(STRESS_ROOT_COUNTS, STRESS_REFINEMENT_PASSES)
  stress_setup_plan = compile(build_problem(stress_setup.field))
  stress_transition = transition_case(stress_setup.space)
  topology_small = topology_case(3)
  topology_large = topology_case(5)
  topology_stress = uniform_grid_case(STRESS_ROOT_COUNTS, STRESS_REFINEMENT_PASSES)
  segment_small = segment_surface_case((8, 8), 24)
  segment_large = segment_surface_case((20, 20), 96; point_count=5)
  segment_stress = segment_surface_case(STRESS_SEGMENT_ROOT_COUNTS, STRESS_SEGMENT_COUNT;
                                        point_count=7)
  small_adaptivity_limits = AdaptivityLimits(field_space(small.field); max_p=MAX_DEGREE,
                                             max_h_level=MAX_H_LEVEL)
  large_adaptivity_limits = AdaptivityLimits(field_space(large.field); max_p=MAX_DEGREE,
                                             max_h_level=MAX_H_LEVEL)
  stress_adaptivity_limits = AdaptivityLimits(field_space(stress.field); max_p=MAX_DEGREE,
                                              max_h_level=MAX_H_LEVEL)

  suite["workflow"] = BenchmarkGroup()
  suite["workflow"]["small"] = BenchmarkGroup()
  suite["workflow"]["small"]["compile"] = @benchmarkable compile($(small.problem))
  suite["workflow"]["small"]["assemble"] = @benchmarkable assemble($(small.plan))
  suite["workflow"]["small"]["solve"] = @benchmarkable solve($(small.system))
  suite["workflow"]["small"]["verify"] = @benchmarkable relative_l2_error($(small.state),
                                                                          $(small.field),
                                                                          exact_solution;
                                                                          plan=($(small.plan)),
                                                                          extra_points=VERIFICATION_EXTRA_POINTS)
  suite["workflow"]["small"]["refine"] = @benchmarkable adaptivity_plan($(small.state),
                                                                        $(small.field);
                                                                        tolerance=ADAPTIVITY_TOLERANCE,
                                                                        limits=($small_adaptivity_limits))
  suite["workflow"]["large"] = BenchmarkGroup()
  suite["workflow"]["large"]["compile"] = @benchmarkable compile($(large.problem))
  suite["workflow"]["large"]["assemble"] = @benchmarkable assemble($(large.plan))
  suite["workflow"]["large"]["solve"] = @benchmarkable solve($(large.system))
  suite["workflow"]["large"]["verify"] = @benchmarkable relative_l2_error($(large.state),
                                                                          $(large.field),
                                                                          exact_solution;
                                                                          plan=($(large.plan)),
                                                                          extra_points=VERIFICATION_EXTRA_POINTS)
  suite["workflow"]["large"]["refine"] = @benchmarkable adaptivity_plan($(large.state),
                                                                        $(large.field);
                                                                        tolerance=ADAPTIVITY_TOLERANCE,
                                                                        limits=($large_adaptivity_limits))
  suite["workflow"]["stress"] = BenchmarkGroup()
  suite["workflow"]["stress"]["compile"] = @benchmarkable compile($(stress.problem))
  suite["workflow"]["stress"]["assemble"] = @benchmarkable assemble($(stress.plan))
  suite["workflow"]["stress"]["solve"] = @benchmarkable solve($(stress.system))
  suite["workflow"]["stress"]["verify"] = @benchmarkable relative_l2_error($(stress.state),
                                                                           $(stress.field),
                                                                           exact_solution;
                                                                           plan=($(stress.plan)),
                                                                           extra_points=VERIFICATION_EXTRA_POINTS)
  suite["workflow"]["stress"]["refine"] = @benchmarkable adaptivity_plan($(stress.state),
                                                                         $(stress.field);
                                                                         tolerance=ADAPTIVITY_TOLERANCE,
                                                                         limits=($stress_adaptivity_limits))

  suite["setup"] = BenchmarkGroup()
  suite["setup"]["small"] = BenchmarkGroup()
  suite["setup"]["small"]["integration"] = @benchmarkable Grico._compile_integration($(small.plan.layout))
  suite["setup"]["small"]["interior_faces"] = @benchmarkable Grico._compile_interior_faces($(small.field))
  suite["setup"]["small"]["transition"] = @benchmarkable transition($(small_transition.adaptivity_plan))
  suite["setup"]["large"] = BenchmarkGroup()
  suite["setup"]["large"]["integration"] = @benchmarkable Grico._compile_integration($(large.plan.layout))
  suite["setup"]["large"]["interior_faces"] = @benchmarkable Grico._compile_interior_faces($(large.field))
  suite["setup"]["large"]["transition"] = @benchmarkable transition($(large_transition.adaptivity_plan))
  suite["setup"]["stress"] = BenchmarkGroup()
  suite["setup"]["stress"]["integration"] = @benchmarkable Grico._compile_integration($(stress_setup_plan.layout))
  suite["setup"]["stress"]["interior_faces"] = @benchmarkable Grico._compile_interior_faces($(stress_setup.field))
  suite["setup"]["stress"]["transition"] = @benchmarkable transition($(stress_transition.adaptivity_plan))
  suite["setup"]["topology"] = BenchmarkGroup()
  suite["setup"]["topology"]["neighbors_small"] = @benchmarkable Grico._rebuild_direct_neighbors!(grid) setup = (grid = copy($topology_small))
  suite["setup"]["topology"]["neighbors_large"] = @benchmarkable Grico._rebuild_direct_neighbors!(grid) setup = (grid = copy($topology_large))
  suite["setup"]["topology"]["neighbors_stress"] = @benchmarkable Grico._rebuild_direct_neighbors!(grid) setup = (grid = copy($topology_stress))
  suite["setup"]["topology"]["active_small"] = @benchmarkable Grico._rebuild_active_leaves!(grid) setup = (grid = copy($topology_small))
  suite["setup"]["topology"]["active_large"] = @benchmarkable Grico._rebuild_active_leaves!(grid) setup = (grid = copy($topology_large))
  suite["setup"]["topology"]["active_stress"] = @benchmarkable Grico._rebuild_active_leaves!(grid) setup = (grid = copy($topology_stress))

  embedded = embedded_case()
  suite["embedded"] = BenchmarkGroup()
  suite["embedded"]["finite_cell_small"] = @benchmarkable finite_cell_quadrature($(embedded.domain),
                                                                                 $(embedded.leaf),
                                                                                 (3, 3),
                                                                                 $(embedded.classifier);
                                                                                 subdivision_depth=2)
  suite["embedded"]["finite_cell_large"] = @benchmarkable finite_cell_quadrature($(embedded.domain),
                                                                                 $(embedded.leaf),
                                                                                 (5, 5),
                                                                                 $(embedded.classifier);
                                                                                 subdivision_depth=5)
  suite["embedded"]["surface_small"] = @benchmarkable implicit_surface_quadrature($(embedded.domain),
                                                                                  $(embedded.leaf),
                                                                                  $(embedded.classifier);
                                                                                  subdivision_depth=2,
                                                                                  surface_point_count=2)
  suite["embedded"]["surface_large"] = @benchmarkable implicit_surface_quadrature($(embedded.domain),
                                                                                  $(embedded.leaf),
                                                                                  $(embedded.classifier);
                                                                                  subdivision_depth=5,
                                                                                  surface_point_count=3)
  suite["embedded"]["segment_mesh_small"] = @benchmarkable Grico.surface_quadratures($(segment_small.surface),
                                                                                     $(segment_small.domain))
  suite["embedded"]["segment_mesh_large"] = @benchmarkable Grico.surface_quadratures($(segment_large.surface),
                                                                                     $(segment_large.domain))
  suite["embedded"]["segment_mesh_stress"] = @benchmarkable Grico.surface_quadratures($(segment_stress.surface),
                                                                                      $(segment_stress.domain))

  return suite
end

function _requested_thread_counts()
  raw = strip(get(ENV, "GRICO_BENCH_THREADS",
                  string(1, Sys.CPU_THREADS > 1 ? "," * string(min(Sys.CPU_THREADS, 4)) : "")))
  counts = Int[]

  for part in split(raw, ',')
    token = strip(part)
    isempty(token) && continue
    count = parse(Int, token)
    count > 0 || throw(ArgumentError("thread counts must be positive"))
    push!(counts, count)
  end

  isempty(counts) &&
    throw(ArgumentError("GRICO_BENCH_THREADS must request at least one thread count"))
  return unique(counts)
end

function _configure_benchmarktools!()
  params = BenchmarkTools.DEFAULT_PARAMETERS
  params.seconds = parse(Float64, get(ENV, "GRICO_BENCH_SECONDS", "0.2"))
  params.samples = parse(Int, get(ENV, "GRICO_BENCH_SAMPLES", "15"))
  params.evals = 1
  return nothing
end

function _configure_blas!()
  thread_count = parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", string(Threads.nthreads())))
  thread_count > 0 || throw(ArgumentError("OPENBLAS_NUM_THREADS must be positive"))
  BLAS.set_num_threads(thread_count)
  return nothing
end

function _run_child()
  _configure_benchmarktools!()
  _configure_blas!()
  suite = build_suite()
  results = run(suite; verbose=true)
  print_summary(results)

  output_path = strip(get(ENV, "GRICO_BENCH_OUTPUT", ""))

  if !isempty(output_path)
    mkpath(dirname(output_path))
    BenchmarkTools.save(output_path, results)
    println("saved benchmark results to $output_path")
  end

  return nothing
end

function _thread_output_path(path::AbstractString, thread_count::Int)
  isempty(path) && return path
  stem, extension = splitext(path)
  return string(stem, "_t", thread_count, extension)
end

function _spawn_children()
  runner = @__FILE__
  output_path = get(ENV, "GRICO_BENCH_OUTPUT", "")

  for thread_count in _requested_thread_counts()
    println()
    @printf("== Grico benchmarks with JULIA_NUM_THREADS=%d ==\n", thread_count)
    cmd = addenv(`$(Base.julia_cmd()) --project=$REPO_ROOT $runner --child`,
                 "JULIA_NUM_THREADS" => string(thread_count),
                 "OPENBLAS_NUM_THREADS" => string(thread_count),
                 "GRICO_BENCH_SECONDS" => get(ENV, "GRICO_BENCH_SECONDS", "0.2"),
                 "GRICO_BENCH_SAMPLES" => get(ENV, "GRICO_BENCH_SAMPLES", "15"),
                 "GRICO_BENCH_OUTPUT" => _thread_output_path(output_path, thread_count))
    run(cmd)
  end

  return nothing
end

function print_summary(group::BenchmarkGroup, prefix::Vector{String}=String[])
  if isempty(prefix)
    @printf("threads=%d blas=%d\n", Threads.nthreads(), BLAS.get_num_threads())
    println("benchmark median_ms memory_kib allocs")
  end

  for name in sort!(collect(keys(group)))
    entry = group[name]

    if entry isa BenchmarkGroup
      print_summary(entry, [prefix; String(name)])
      continue
    end

    estimate = median(entry)
    label = join([prefix; String(name)], '/')
    @printf("%-34s %10.3f %10.1f %8d\n", label, estimate.time / 1.0e6, estimate.memory / 1024,
            estimate.allocs)
  end

  return nothing
end

function main(arguments)
  if "--child" in arguments
    _run_child()
  elseif isempty(arguments)
    _spawn_children()
  else
    throw(ArgumentError("usage: julia --project=. benchmark/runbenchmarks.jl"))
  end

  return 0
end

exit(main(ARGS))

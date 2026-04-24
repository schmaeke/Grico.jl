using Test
using LinearAlgebra
using Printf
using Grico
import Grico: cell_matrix!, cell_residual!, cell_rhs!, cell_tangent!, face_residual!,
              interface_residual!

function _active_project_has_dependency(name::AbstractString)
  project = Base.active_project()
  project === nothing && return false
  in_deps = false

  for raw_line in eachline(project)
    line = strip(first(split(raw_line, '#'; limit=2)))
    isempty(line) && continue

    if startswith(line, "[") && endswith(line, "]")
      in_deps = line == "[deps]"
      continue
    end

    in_deps && startswith(line, name * " ") && occursin("=", line) && return true
  end

  return false
end

const ORDINARYDIFFEQ_AVAILABLE = if _active_project_has_dependency("OrdinaryDiffEq")
  try
    @eval import OrdinaryDiffEq
    true
  catch
    false
  end
else
  false
end

const BLAST_WAVE_EXAMPLE_DIR = joinpath(@__DIR__, "..", "examples", "blast_wave_euler")
include(joinpath(BLAST_WAVE_EXAMPLE_DIR, "parameters.jl"))
include(joinpath(BLAST_WAVE_EXAMPLE_DIR, "euler_physics.jl"))
include(joinpath(BLAST_WAVE_EXAMPLE_DIR, "projection.jl"))
include(joinpath(BLAST_WAVE_EXAMPLE_DIR, "dg_residual.jl"))
include(joinpath(BLAST_WAVE_EXAMPLE_DIR, "runtime_context.jl"))
include(joinpath(BLAST_WAVE_EXAMPLE_DIR, "output.jl"))
include(joinpath(BLAST_WAVE_EXAMPLE_DIR, "time_driver.jl"))

module NewtonFractalPoissonExample
Base.include(@__MODULE__,
             joinpath(@__DIR__, "..", "examples", "newton_fractal_poisson", "driver.jl"))
end

@testset "Blast-Wave Euler Example" begin
  coarse_context = build_blast_wave_euler_context(; root_counts=(2, 2), degree=1,
                                                  quadrature_extra_points=1, cfl=0.08,
                                                  initial_refinement_layers=0)
  context = build_blast_wave_euler_context(; root_counts=(2, 2), degree=1,
                                           quadrature_extra_points=1, cfl=0.08,
                                           initial_refinement_layers=2)

  @test active_leaf_count(coarse_context.space) == 4
  @test length(coefficients(coarse_context.state)) == 64
  @test origin(coarse_context.domain) == (0.0, 0.0)
  @test extent(coarse_context.domain) == (1.0, 1.0)
  @test active_leaf_count(context.space) > active_leaf_count(coarse_context.space)
  @test length(coefficients(context.state)) > length(coefficients(coarse_context.state))
  @test context.dt > 0.0
  @test !hasproperty(context, :diagnostics)
  @test !hasproperty(context, :mass_plan)
  @test !hasproperty(context, :mass_matrix)
  @test merge_time_grids([0.0, 0.01], [0.0, 0.005, 0.01]) ≈ [0.0, 0.005, 0.01]

  q = conservative_variables(1.0, (0.3, -0.2), 1.0, GAMMA)
  q_reflected = reflective_ghost_state(q, (1.0, 0.0))
  wall_flux = lax_friedrichs_flux(q, q_reflected, (1.0, 0.0), GAMMA)
  @test wall_flux[1] ≈ 0.0 atol = 1.0e-12
  @test wall_flux[4] ≈ 0.0 atol = 1.0e-12

  semi = EulerSemidiscretization(context.spatial_plan, context.conserved, context.state,
                                 context.mass_inverse)
  du = similar(coefficients(context.state))
  euler_rhs!(du, copy(coefficients(context.state)), semi, 0.0)
  @test all(isfinite, du)
  @test any(!iszero, du)

  adapted_context, plan = adapt_blast_wave_context(context; tolerance=0.0, max_h_level=3)
  plan_summary = adaptivity_summary(plan)
  @test plan_summary.h_refinement_leaf_count > 0
  @test active_leaf_count(adapted_context.space) > active_leaf_count(context.space)
  @test length(coefficients(adapted_context.state)) > length(coefficients(context.state))
  @test adapted_context.dt > 0.0
  adaptivity_entry = blast_wave_adaptivity_entry(1, 0.0, context, adapted_context, plan)
  @test adaptivity_entry.after_active_leaves > adaptivity_entry.before_active_leaves
  @test adaptivity_entry.h_refinement_leaf_count == plan_summary.h_refinement_leaf_count

  entry = blast_wave_history_entry(0, 0.0, context)
  @test entry.dt == context.dt
  mktempdir() do directory
    vtk_path = write_blast_wave_vtk(context, entry; output_directory=directory)
    @test isfile(vtk_path)
  end

  if ORDINARYDIFFEQ_AVAILABLE
    result = run_blast_wave_euler_example(; root_counts=(2, 2), degree=1, quadrature_extra_points=1,
                                          cfl=0.08, final_time=0.01, save_interval=0.01,
                                          initial_refinement_layers=2, adapt_interval=0.005,
                                          write_vtk=false, print_summary=false,
                                          adaptivity_tolerance=0.0, max_h_level=3)
    @test !isempty(result.history)
    @test !isempty(result.adaptivity_history)
    @test result.adaptivity_history[1].after_active_leaves >
          result.adaptivity_history[1].before_active_leaves
    @test result.segment_solutions === nothing
    @test last(result.history).time ≈ 0.01 atol = 1.0e-12
  end
end

@testset "Newton Fractal Poisson Example" begin
  result = NewtonFractalPoissonExample.run_newton_fractal_poisson_example(; max_h_level=3,
                                                                          write_vtk=false,
                                                                          print_summary=false)

  @test length(result.history) == 4
  @test first(result.history).active_leaves == 1
  @test last(result.history).max_h_level == 3
  @test last(result.history).active_leaves > first(result.history).active_leaves
  @test first(result.history).h_refinement_leaf_count > 0
  @test any(entry -> entry.p_refinement_leaf_count > 0, result.history)
  @test isfinite(result.final_error)
  @test result.min_degree == NewtonFractalPoissonExample.MIN_DEGREE
  @test result.min_degree <= result.initial_degree <= result.max_degree
  @test length(result.model.roots) == 3
  @test result.model.iterations == NewtonFractalPoissonExample.NEWTON_ITERATIONS
  @test abs(result.exact_solution((0.0, 0.5))) > 1.0e-6
  @test 1 <= NewtonFractalPoissonExample.newton_basin_index(result.model, (0.5, 0.5)) <= 3
  @test isfinite(NewtonFractalPoissonExample.newton_source_value(result.model, (0.5, 0.5)))
  @test_throws ArgumentError begin
    NewtonFractalPoissonExample.build_newton_fractal_poisson_context(; min_degree=4,
                                                                     initial_degree=3, max_degree=5)
  end
end

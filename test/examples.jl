using Test

include("../examples/blast_wave_euler/benchmarking.jl")

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

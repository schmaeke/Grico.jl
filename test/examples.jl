using Test

ENV["GRICO_KH_EULER_AUTORUN"] = "0"
include("../examples/kelvin_helmholtz_euler.jl")

@testset "Kelvin-Helmholtz Euler Example" begin
  context = build_kelvin_helmholtz_euler_context(; root_counts=(2, 2), degree=1,
                                                 quadrature_extra_points=1, cfl=0.1)

  @test active_leaf_count(context.space) == 4
  @test length(coefficients(context.state)) == 64
  @test context.diagnostics.min_density > 0.0
  @test context.diagnostics.min_pressure > 0.0
  @test context.dt > 0.0
  @test !hasproperty(context, :mass_plan)
  @test !hasproperty(context, :mass_matrix)
  @test merge_time_grids([0.0, 0.01], [0.0, 0.005, 0.01]) ≈ [0.0, 0.005, 0.01]

  semi = EulerSemidiscretization(context.spatial_plan, context.conserved, context.state,
                                 context.mass_factorization)
  du = similar(coefficients(context.state))
  euler_rhs!(du, copy(coefficients(context.state)), semi, 0.0)
  @test all(isfinite, du)
  @test any(!iszero, du)

  adapted_context, plan = adapt_kelvin_helmholtz_context(context; threshold=1.0,
                                                         smoothness_threshold=0.5,
                                                         max_h_level=0, max_p_degree=2)
  plan_summary = adaptivity_summary(plan)
  @test plan_summary.p_refinement_leaf_count > 0
  @test plan_summary.h_refinement_leaf_count == 0
  @test active_leaf_count(adapted_context.space) == active_leaf_count(context.space)
  @test length(coefficients(adapted_context.state)) > length(coefficients(context.state))
  @test adapted_context.dt > 0.0

  coarsened_context, coarsen_plan = adapt_kelvin_helmholtz_context(adapted_context; threshold=0.0,
                                                                   smoothness_threshold=0.5,
                                                                   p_coarsening_threshold=1.0,
                                                                   h_coarsening_threshold=1.0,
                                                                   max_h_level=0,
                                                                   max_p_degree=2)
  coarsen_summary = adaptivity_summary(coarsen_plan)
  @test coarsen_summary.p_derefinement_leaf_count > 0
  @test active_leaf_count(coarsened_context.space) == active_leaf_count(adapted_context.space)
  @test length(coefficients(coarsened_context.state)) == length(coefficients(context.state))
  @test coarsened_context.dt > 0.0

  if ORDINARYDIFFEQ_AVAILABLE
    result = run_kelvin_helmholtz_euler_example(; root_counts=(2, 2), degree=1,
                                                quadrature_extra_points=1, cfl=0.08,
                                                final_time=0.01, save_interval=0.01,
                                                adapt_interval=0.01,
                                                write_vtk=false, print_summary=false,
                                                adaptivity_threshold=1.0,
                                                smoothness_threshold=0.5,
                                                p_coarsening_threshold=1.0,
                                                h_coarsening_threshold=1.0,
                                                max_h_level=1, max_p_degree=2)
    @test !isempty(result.history)
    @test result.segment_solutions === nothing
    @test last(result.history).time ≈ 0.01 atol = 1.0e-12
  end
end

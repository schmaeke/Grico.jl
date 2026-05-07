using Test
using Grico

const EXAMPLES_ROOT = joinpath(@__DIR__, "..", "examples")

@testset "Poisson Matrix-Free GMG" begin
  include(joinpath(EXAMPLES_ROOT, "poisson_matrix_free_gmg", "driver.jl"))
  result = run_poisson_matrix_free_gmg_example(; root_counts=(2, 2), degree=2, write_output=false,
                                               print_summary=false)
  @test result.error_value <= 2.0e-2
end

@testset "Adaptive Origin Singularity" begin
  include(joinpath(EXAMPLES_ROOT, "adaptive_origin_singularity", "driver.jl"))
  result = run_adaptive_origin_singularity_example(; adaptive_steps=1, max_degree=3, max_h_level=1,
                                                   write_output=false, print_summary=false)
  @test isfinite(result.final_error)
  @test length(result.history) == 2
  @test last(result.history).active_leaves >= first(result.history).active_leaves
end

@testset "Annular Nitsche FCM" begin
  include(joinpath(EXAMPLES_ROOT, "annular_nitsche_fcm", "driver.jl"))
  result = run_annular_nitsche_fcm_example(; degree=2, segment_count=24, fcm_depth=3,
                                           write_output=false, print_summary=false)
  @test isfinite(result.error_value)
  @test result.error_value <= 0.35
end

@testset "Nonlinear Bratu" begin
  include(joinpath(EXAMPLES_ROOT, "nonlinear_bratu", "driver.jl"))
  result = run_nonlinear_bratu_example(; root_counts=(2, 2), degree=2, λ=0.5, amplitude=0.1,
                                       write_output=false, print_summary=false)
  @test result.error_value <= 2.0e-3
end

@testset "Poisson 1D Makie" begin
  include(joinpath(EXAMPLES_ROOT, "poisson_1d_makie", "driver.jl"))

  if Base.find_package("CairoMakie") === nothing
    @test_throws ArgumentError _load_cairomakie!()
  else
    result = run_poisson_1d_makie_example(; cell_count=1, degree=3, output_directory=mktempdir(),
                                          print_summary=false)
    @test isfile(result.figure_path)
  end
end

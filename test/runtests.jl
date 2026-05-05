using Test
using Grico

@testset verbose = true "Grico" begin
  include("api_surface.jl")
  include("feature_stability.jl")
  include("problem_api.jl")
  include("runtime_contracts.jl")

  @testset verbose = true "Core Numerics" begin
    include("core.jl")
  end

  @testset verbose = true "Mesh And Geometry" begin
    include("topology_geometry.jl")
    include("embedded_geometry.jl")
  end

  @testset verbose = true "Space And Fields" begin
    include("space.jl")
  end

  @testset verbose = true "Matrix-Free Problems" begin
    include("matrix_free.jl")
    include("multigrid.jl")
  end

  @testset verbose = true "Adaptivity" begin
    include("adaptivity.jl")
  end

  @testset verbose = true "Verification And Output" begin
    @testset "Verification" begin
      include("verification.jl")
    end

    @testset "Postprocess" begin
      include("postprocess.jl")
    end

    @testset "VTK" begin
      include("vtk.jl")
    end
  end

  @testset verbose = true "Examples" begin
    include("examples.jl")
  end
end

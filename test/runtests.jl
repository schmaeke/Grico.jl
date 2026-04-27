using Test
using Grico

@testset verbose = true "Grico" begin
  @testset verbose = true "Core Numerics" begin
    include("core.jl")
  end

  @testset verbose = true "Mesh And Geometry" begin
    include("topology_geometry.jl")
  end

  @testset verbose = true "Space And Fields" begin
    include("space.jl")
  end

  @testset verbose = true "Problems, Assembly, And Embedded" begin
    include("assembly.jl")
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

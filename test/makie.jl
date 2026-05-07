using Test
using Grico
import Grico: SampledPostprocess, sample_mesh_skeleton

if Base.find_package("Makie") !== nothing
  using Makie

  @testset "Makie Extension" begin
    extension = Base.get_extension(Grico, :GricoMakieExt)
    extension !== nothing || error("GricoMakieExt did not load")

    domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
    sampled = sample_postprocess(domain; point_data=(marker=x -> x[1] + x[2],), sample_degree=3)
    faces = extension._makie_triangle_faces(sampled.mesh)

    @test size(faces) == (18, 3)
    @test length(unique(vec(faces))) == size(sampled.mesh.points, 2)
    @test faces[1, :] == [1, 2, 6]
    @test faces[2, :] == [6, 5, 1]
    @test_throws ArgumentError extension._makie_component(true, 2, "marker")
    @test_throws ArgumentError extension._makie_component(1.5, 2, "marker")
    @test_throws ArgumentError extension._makie_field_overlay(1, sampled.mesh)

    skeleton1 = sample_mesh_skeleton(Domain((0.0,), (1.0,), (1,)))
    @test_throws ArgumentError extension._makie_field_overlay(skeleton1, sampled.mesh)

    bad_point_data = Pair{String,Union{AbstractVector,AbstractMatrix}}["bad" => [1.0, 2.0]]
    bad_sampled = SampledPostprocess{2,Float64}(sampled.mesh, bad_point_data, sampled.cell_data,
                                                sampled.field_data)
    @test_throws ArgumentError extension._require_makie_sampled_postprocess(bad_sampled)
  end
end

using Test
using Grico
import Grico: SampledPostprocess, postprocess_supported, sample_mesh_skeleton, scalar_dof_count

function _postprocess_throws_argument_message(f, needle::AbstractString)
  try
    f()
  catch exception
    @test exception isa ArgumentError
    @test occursin(needle, sprint(showerror, exception))
    return nothing
  end

  @test false
  return nothing
end

struct ShiftedPostprocessVector{T} <: AbstractVector{T}
  data::Vector{T}
end

Base.size(vector::ShiftedPostprocessVector) = size(vector.data)
Base.axes(vector::ShiftedPostprocessVector) = (0:(length(vector.data)-1),)
Base.IndexStyle(::Type{<:ShiftedPostprocessVector}) = IndexLinear()
Base.getindex(vector::ShiftedPostprocessVector, index::Int) = vector.data[index+1]

function Base.iterate(vector::ShiftedPostprocessVector, state::Int=1)
  state > length(vector.data) && return nothing
  return vector.data[state], state + 1
end

@testset "Postprocess Sampling" begin
  @test postprocess_supported(1)
  @test postprocess_supported(3)
  @test !postprocess_supported(4)
  @test !postprocess_supported(true)
  @test !postprocess_supported(big(typemax(Int)) + 1)

  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [2.0, 2.0])
  copied_domain = copy(domain)
  copied_space = HpSpace(copied_domain,
                         SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  v = ScalarField(copied_space; name=:v)
  mixed_state = State(FieldLayout((u, v)), [2.0, 2.0, 3.0, 3.0])
  physical_region = ImplicitRegion(x -> x[1] - 2.0)
  other_physical_region = ImplicitRegion(x -> x[1] - 3.0)
  physical_domain = PhysicalDomain(copy(domain), physical_region)
  copied_physical_domain = copy(physical_domain)
  other_physical_domain = PhysicalDomain(copy(domain), other_physical_region)
  physical_space = HpSpace(physical_domain,
                           SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  w = ScalarField(physical_space; name=:w)
  physical_state = State(FieldLayout((w,)), [4.0, 4.0])

  geometry = sample_postprocess(domain; point_data=(marker=x -> x[1],),
                                cell_data=(leaf=leaf -> Float64(leaf),
                                           pair=leaf -> (Float64(leaf), Float64(leaf + 1))),
                                field_data=(time=1.5,), subdivisions=2, sample_degree=2)
  geometry_points = Dict(geometry.point_data)
  geometry_cells = Dict(geometry.cell_data)
  geometry_fields = Dict(geometry.field_data)

  @test geometry isa SampledPostprocess{1,Float64}
  @test geometry.mesh.points == reshape([0.0, 0.25, 0.5, 0.75, 1.0], 1, :)
  @test length(geometry.mesh.cell_leaves) == 2
  @test geometry_points["marker"] == [0.0, 0.25, 0.5, 0.75, 1.0]
  @test geometry_cells["leaf"] == [1.0, 1.0]
  @test geometry_cells["pair"] == [1.0 1.0; 2.0 2.0]
  @test geometry_fields["time"] == 1.5

  sampled = sample_postprocess(domain; state=state,
                               point_data=(marker=x -> x[1], shifted=(x, values) -> values.u + x[1],
                                           contextual=(x, values, leaf, ξ) -> values.u + leaf + ξ[1],
                                           tuple_point=(x, values) -> (values.u, x[1])),
                               cell_data=(cell_marker=[4.0], leaf_marker=leaf -> Float64(leaf),
                                          centered_cell=(leaf, x) -> leaf + x[1],
                                          contextual_cell=(leaf, x, ξ) -> leaf + x[1] + ξ[1],
                                          tuple_cell=(leaf, x) -> (leaf, x[1])),
                               field_data=(time=1.5,), subdivisions=2, sample_degree=2)
  points = Dict(sampled.point_data)
  cells = Dict(sampled.cell_data)

  @test Set(keys(points)) == Set(["u", "marker", "shifted", "contextual", "tuple_point"])
  @test points["u"] == fill(2.0, 5)
  @test points["marker"] == [0.0, 0.25, 0.5, 0.75, 1.0]
  @test points["shifted"] == [2.0, 2.25, 2.5, 2.75, 3.0]
  @test points["contextual"] == [2.0, 2.5, 3.0, 3.5, 4.0]
  @test points["tuple_point"] == [2.0 2.0 2.0 2.0 2.0; 0.0 0.25 0.5 0.75 1.0]
  @test cells["cell_marker"] == [4.0, 4.0]
  @test cells["leaf_marker"] == [1.0, 1.0]
  @test cells["centered_cell"] == [1.25, 1.75]
  @test cells["contextual_cell"] == [0.75, 2.25]
  @test cells["tuple_cell"] == [1.0 1.0; 0.25 0.75]

  shifted_leaf_data = ShiftedPostprocessVector([9.0])
  shifted_cell_data = sample_postprocess(domain; cell_data=(shifted=shifted_leaf_data,),
                                         subdivisions=2)
  @test Dict(shifted_cell_data.cell_data)["shifted"] == [9.0, 9.0]

  linear_state = State(FieldLayout((u,)), [0.0, 1.0])
  du_dx = (_x, _values, leaf, ξ) -> field_gradient(linear_state, u, leaf, ξ)[1]
  gradient_sampled = sample_postprocess(linear_state; point_data=(du_dx=du_dx,), subdivisions=2,
                                        sample_degree=2)

  @test field_gradient(linear_state, u, 1, (-1.0,)) == (1.0,)
  @test field_gradient(linear_state, u, 1, (0.0,)) == (1.0,)
  @test Dict(gradient_sampled.point_data)["du_dx"] == ones(5)
  @test_throws ArgumentError field_gradient(linear_state, u, 1, (2.0,))
  @test_throws ArgumentError field_gradient(linear_state, u, 2, 1, (0.0,))
  @test_throws ArgumentError field_gradient(linear_state, u, 1, (0.0, 0.0))

  mixed = sample_postprocess(mixed_state)
  @test haskey(Dict(mixed.point_data), "u")
  @test haskey(Dict(mixed.point_data), "v")

  refined_domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  refine!(grid(refined_domain), 1, 1)
  skeleton = sample_mesh_skeleton(refined_domain)
  skeleton_cells = Dict(skeleton.cell_data)

  @test size(skeleton.points) == (2, 6)
  @test size(skeleton.edges) == (2, 7)
  @test skeleton_cells["h_level"] == fill(1, 7)
  @test Set(skeleton_cells["axis"]) == Set([1, 2])
  @test Set(skeleton_cells["leaf"]) == Set(active_leaves(grid(refined_domain)))

  ordered_domain = Domain((0.0,), (2.0,), (2,))
  refine!(grid(ordered_domain), 2, 1)
  refine!(grid(ordered_domain), 1, 1)
  ordered_space = HpSpace(ordered_domain, SpaceOptions(degree=UniformDegree(0), continuity=:dg))
  ordered_field = ScalarField(ordered_space; name=:ordered)
  ordered_state = State(FieldLayout((ordered_field,)), ones(scalar_dof_count(ordered_space)))
  @test active_leaves(ordered_space) != active_leaves(grid(ordered_domain))
  @test sample_postprocess(ordered_domain; state=ordered_state) isa SampledPostprocess

  _postprocess_throws_argument_message(() -> write_vtk("unloaded", domain), "requires WriteVTK")
  _postprocess_throws_argument_message(() -> write_pvd("unloaded.pvd", String[]),
                                       "requires WriteVTK")
  _postprocess_throws_argument_message(() -> plot_field(sampled, :u), "requires Makie")
  _postprocess_throws_argument_message(() -> plot_field!(nothing, sampled, :u), "requires Makie")
  _postprocess_throws_argument_message(() -> plot_field(state, :u), "requires Makie")
  _postprocess_throws_argument_message(() -> plot_mesh(refined_domain), "requires Makie")
  _postprocess_throws_argument_message(() -> plot_mesh!(nothing, refined_domain), "requires Makie")

  fallback_script = raw"""
using Grico

function _throws_argument_message(f, needle::AbstractString)
  try
    f()
  catch exception
    exception isa ArgumentError || error("expected ArgumentError")
    message = sprint(showerror, exception)
    occursin(needle, message) || error("missing expected message")
    return nothing
  end

  error("expected ArgumentError")
end

domain = Domain((0.0,), (1.0,), (1,))
space = HpSpace(domain, SpaceOptions(degree=UniformDegree(1)))
u = ScalarField(space; name=:u)
state = State(FieldLayout((u,)), zeros(field_dof_count(u)))
sampled = sample_postprocess(state)

_throws_argument_message(() -> write_vtk("unloaded", domain), "requires WriteVTK")
_throws_argument_message(() -> write_vtk("unloaded", sampled), "requires WriteVTK")
_throws_argument_message(() -> write_pvd("unloaded.pvd", String[]), "requires WriteVTK")
_throws_argument_message(() -> plot_field(sampled, :u), "requires Makie")
_throws_argument_message(() -> plot_field!(nothing, sampled, :u), "requires Makie")
_throws_argument_message(() -> plot_mesh(domain), "requires Makie")
_throws_argument_message(() -> plot_mesh!(nothing, domain), "requires Makie")
"""
  fallback_command = `$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) -e $fallback_script`
  @test success(fallback_command)

  @test_throws ArgumentError sample_postprocess(domain; fields=(u,))
  @test sample_postprocess(copied_domain; state=state) isa SampledPostprocess
  @test sample_postprocess(copied_physical_domain; state=physical_state) isa SampledPostprocess
  @test_throws ArgumentError sample_postprocess(refined_domain; state=state)
  @test_throws ArgumentError sample_postprocess(other_physical_domain; state=physical_state)
  @test_throws ArgumentError sample_postprocess(domain; state=state, fields=(1,))
  @test_throws ArgumentError sample_postprocess(domain; point_data=1)
  @test_throws ArgumentError sample_postprocess(domain; point_data=(1 => [1.0, 2.0],))
  @test_throws ArgumentError sample_postprocess(domain; point_data=(empty=x -> (),))
  @test_throws ArgumentError sample_postprocess(domain;
                                                point_data=["marker" => [1.0, 2.0],
                                                            "marker" => [1.0, 2.0]])
  @test_throws ArgumentError sample_postprocess(domain;
                                                cell_data=["leaf" => [1.0], "leaf" => [2.0]])
  @test_throws ArgumentError sample_postprocess(domain;
                                                field_data=["time" => [1.0], "time" => [2.0]])
  @test_throws ArgumentError sample_postprocess(domain; point_data=(bad=[1.0],), subdivisions=2)
  @test_throws ArgumentError sample_postprocess(domain; point_data=(bad=(() -> 1.0),))
  @test_throws ArgumentError sample_postprocess(State(FieldLayout((u,)), [NaN, 1.0]))
  @test_throws ArgumentError sample_postprocess(domain; point_data=(bad=x -> NaN,))
  @test_throws ArgumentError sample_postprocess(domain; point_data=(bad=x -> true,))
  @test_throws ArgumentError sample_postprocess(domain; point_data=(bad=[0.0, Inf],))
  @test_throws ArgumentError sample_postprocess(domain;
                                                point_data=(bad=x -> x[1] == 0.0 ? 1 : 1.5,))
  @test_throws ArgumentError sample_postprocess(domain; cell_data=(bad=leaf -> (1.0, Inf),))
  @test_throws ArgumentError sample_postprocess(domain; subdivisions=0)
  @test_throws ArgumentError sample_postprocess(domain; sample_degree=0)
  _postprocess_throws_argument_message(() -> sample_postprocess(domain; subdivisions=1.5),
                                       "subdivisions must be a positive Int-representable integer")
  _postprocess_throws_argument_message(() -> sample_postprocess(domain; sample_degree=1.5),
                                       "sample_degree must be a positive Int-representable integer")
  _postprocess_throws_argument_message(() -> sample_postprocess(domain; subdivisions=typemax(Int)),
                                       "too many postprocess samples")
  _postprocess_throws_argument_message(() -> sample_postprocess(domain; subdivisions=2,
                                                                sample_degree=typemax(Int)),
                                       "too many postprocess samples")
end

using Test
using Grico

function vtk_xml_attribute(xml::AbstractString, name::AbstractString)
  match_data = match(Regex("$(name)=\"([^\"]+)\""), xml)
  match_data === nothing && error("missing VTK attribute $name")
  return match_data.captures[1]
end

function vtk_file_attribute(xml::AbstractString, name::AbstractString)
  match_data = match(Regex("<VTKFile[^>]*$(name)=\"([^\"]+)\""), xml)
  match_data === nothing && error("missing VTK file attribute $name")
  return match_data.captures[1]
end

function vtk_data_array(xml::AbstractString, name::AbstractString, ::Type{T}) where {T}
  pattern = Regex("<DataArray[^>]*Name=\"$(name)\"[^>]*>\\s*([^<]+?)\\s*</DataArray>", "s")
  match_data = match(pattern, xml)
  match_data === nothing && error("missing VTK data array $name")
  return parse.(T, split(strip(match_data.captures[1])))
end

@testset "VTK Export" begin
  @test vtk_export_supported(1)
  @test vtk_export_supported(3)
  @test !vtk_export_supported(4)

  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [2.0, 2.0])
  copied_domain = copy(domain)
  copied_space = HpSpace(copied_domain,
                         SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  v = ScalarField(copied_space; name=:v)
  mixed_state = State(FieldLayout((u, v)), [2.0, 2.0, 3.0, 3.0])

  mktempdir() do directory
    path = write_vtk(joinpath(directory, "grid"), domain; point_data=(marker=x -> x[1],),
                     cell_data=(leaf=leaf -> Float64(leaf),
                                pair=leaf -> (Float64(leaf), Float64(leaf + 1))),
                     field_data=(time=1.5,), subdivisions=2, export_degree=2, append=false,
                     ascii=true)
    xml = read(path, String)

    @test isfile(path)
    @test occursin("Name=\"marker\"", xml)
    @test occursin("Name=\"leaf\"", xml)
    @test occursin("Name=\"pair\"", xml)
    @test occursin("NumberOfComponents=\"2\"", xml)
    @test occursin("Name=\"time\"", xml)
    @test !occursin("Name=\"u\"", xml)
    @test vtk_data_array(xml, "Points", Float64) ==
          [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.0]
    @test vtk_data_array(xml, "pair", Float64) == [1.0, 2.0, 1.0, 2.0]
  end

  mktempdir() do directory
    path = write_vtk(joinpath(directory, "line"), domain; state=state,
                     point_data=(marker=x -> x[1], shifted=(x, values) -> values.u + x[1],
                                 contextual=(x, values, leaf, ξ) -> values.u + leaf + ξ[1],
                                 tuple_point=(x, values) -> (values.u, x[1])),
                     cell_data=(cell_marker=[4.0], leaf_marker=leaf -> Float64(leaf),
                                centered_cell=(leaf, x) -> leaf + x[1],
                                contextual_cell=(leaf, x, ξ) -> leaf + x[1] + ξ[1],
                                tuple_cell=(leaf, x) -> (Float64(leaf), x[1])),
                     field_data=(time=1.5,), subdivisions=2, export_degree=2, append=false,
                     ascii=true)
    xml = read(path, String)

    @test isfile(path)
    @test endswith(path, ".vtu")
    @test occursin("UnstructuredGrid", xml)
    @test vtk_file_attribute(xml, "version") == "2.2"
    @test vtk_xml_attribute(xml, "NumberOfPoints") == "5"
    @test vtk_xml_attribute(xml, "NumberOfCells") == "2"
    @test occursin("Name=\"u\"", xml)
    @test occursin("Name=\"marker\"", xml)
    @test occursin("Name=\"shifted\"", xml)
    @test occursin("Name=\"contextual\"", xml)
    @test occursin("Name=\"tuple_point\"", xml)
    @test occursin("Name=\"cell_marker\"", xml)
    @test occursin("Name=\"leaf_marker\"", xml)
    @test occursin("Name=\"centered_cell\"", xml)
    @test occursin("Name=\"contextual_cell\"", xml)
    @test occursin("Name=\"tuple_cell\"", xml)
    @test occursin("Name=\"time\"", xml)
    @test vtk_data_array(xml, "Points", Float64) ==
          [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.0]
    @test vtk_data_array(xml, "connectivity", Int) == [0, 2, 1, 2, 4, 3]
    @test vtk_data_array(xml, "types", Int) == [68, 68]
    @test vtk_data_array(xml, "u", Float64) == fill(2.0, 5)
    @test vtk_data_array(xml, "marker", Float64) == [0.0, 0.25, 0.5, 0.75, 1.0]
    @test vtk_data_array(xml, "shifted", Float64) == [2.0, 2.25, 2.5, 2.75, 3.0]
    @test vtk_data_array(xml, "contextual", Float64) == [2.0, 2.5, 3.0, 3.5, 4.0]
    @test vtk_data_array(xml, "tuple_point", Float64) ==
          [2.0, 0.0, 2.0, 0.25, 2.0, 0.5, 2.0, 0.75, 2.0, 1.0]
    @test vtk_data_array(xml, "cell_marker", Float64) == [4.0, 4.0]
    @test vtk_data_array(xml, "leaf_marker", Float64) == [1.0, 1.0]
    @test vtk_data_array(xml, "centered_cell", Float64) == [1.25, 1.75]
    @test vtk_data_array(xml, "contextual_cell", Float64) == [0.75, 2.25]
    @test vtk_data_array(xml, "tuple_cell", Float64) == [1.0, 0.25, 1.0, 0.75]
    @test vtk_data_array(xml, "time", Float64) == [1.5]
  end

  mktempdir() do directory
    path = write_vtk(joinpath(directory, "mixed"), mixed_state; append=false, ascii=true)
    xml = read(path, String)

    @test isfile(path)
    @test occursin("Name=\"u\"", xml)
    @test occursin("Name=\"v\"", xml)
  end

  hex_domain = Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
  mktempdir() do directory
    path = write_vtk(joinpath(directory, "hex"), hex_domain; subdivisions=1, export_degree=2,
                     append=false, ascii=true)
    xml = read(path, String)

    @test vtk_xml_attribute(xml, "NumberOfCells") == "1"
    @test vtk_data_array(xml, "types", Int) == [72]
    @test vtk_data_array(xml, "connectivity", Int) ==
          [1, 3, 9, 7, 19, 21, 27, 25, 2, 6, 8, 4, 20, 24, 26, 22, 10, 12, 18, 16, 13, 15, 11, 17,
           5, 23, 14] .- 1
  end

  refined_domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  refine!(grid(refined_domain), 1, 1)
  refined_space = HpSpace(refined_domain,
                          SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  v = VectorField(refined_space, 2; name=:v)
  refined_state = State(FieldLayout((v,)))

  mktempdir() do directory
    path = write_vtk(joinpath(directory, "quad"), refined_domain; state=refined_state, fields=(v,),
                     export_degree=4, append=false, ascii=true)
    xml = read(path, String)

    @test occursin("NumberOfPoints=\"50\"", xml)
    @test occursin("NumberOfCells=\"2\"", xml)
    @test occursin("Name=\"v\"", xml)
    @test occursin("NumberOfComponents=\"2\"", xml)
    @test length(unique(vtk_data_array(xml, "connectivity", Int))) == 50
  end

  filtered_space = HpSpace(PhysicalDomain(Domain((0.0,), (3.0,), (3,)),
                                          ImplicitRegion(x -> x[1] - 1.5;
                                                         subdivision_depth=1)),
                           SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  filtered_field = ScalarField(filtered_space; name=:filtered)
  filtered_state = State(FieldLayout((filtered_field,)), fill(1.0, scalar_dof_count(filtered_space)))

  mktempdir() do directory
    path = write_vtk(joinpath(directory, "filtered"), filtered_state; append=false, ascii=true)
    xml = read(path, String)

    @test vtk_xml_attribute(xml, "NumberOfCells") == "2"
  end

  mktempdir() do directory
    paths = [write_vtk(joinpath(directory, "series_0000"), state; append=false, ascii=true),
             write_vtk(joinpath(directory, "series_0001"), state; append=false, ascii=true)]
    pvd_path = write_pvd(joinpath(directory, "series.pvd"), paths; timesteps=[0.0, 0.5])
    xml = read(pvd_path, String)

    @test isfile(pvd_path)
    @test occursin("Collection", xml)
    @test occursin("timestep=\"0.0\"", xml)
    @test occursin("timestep=\"0.5\"", xml)
    @test occursin("series_0000.vtu", xml)
    @test occursin("series_0001.vtu", xml)
    @test_throws ArgumentError write_pvd(joinpath(directory, "bad.pvd"), paths; timesteps=[0.0])
  end

  refined_domain = copy(domain)
  refine!(grid(refined_domain), 1, 1)

  mktempdir() do directory
    @test_throws ArgumentError write_vtk(joinpath(directory, "bad"), domain; fields=(u,),
                                         append=false, ascii=true)
    @test isfile(write_vtk(joinpath(directory, "samegeometry"), copied_domain; state=state,
                           append=false, ascii=true))
    @test_throws ArgumentError write_vtk(joinpath(directory, "wrongdomain"), refined_domain;
                                         state=state, append=false, ascii=true)
    @test_throws ArgumentError write_vtk(joinpath(directory, "dup_point"), domain;
                                         point_data=["marker" => [1.0, 2.0],
                                                     "marker" => [1.0, 2.0]], append=false,
                                         ascii=true)
    @test_throws ArgumentError write_vtk(joinpath(directory, "dup_cell"), domain;
                                         cell_data=["leaf" => [1.0], "leaf" => [2.0]], append=false,
                                         ascii=true)
    @test_throws ArgumentError write_vtk(joinpath(directory, "dup_field"), domain;
                                         field_data=["time" => [1.0], "time" => [2.0]],
                                         append=false, ascii=true)
    @test_throws ArgumentError write_vtk(joinpath(directory, "badsize"), domain;
                                         point_data=(bad=[1.0],), subdivisions=2, append=false,
                                         ascii=true)
    @test_throws ArgumentError write_vtk(joinpath(directory, "badcall"), domain;
                                         point_data=(bad=(() -> 1.0),), append=false, ascii=true)
    @test_throws ArgumentError write_vtk(joinpath(directory, "badsub"), domain; subdivisions=0,
                                         append=false, ascii=true)
    @test_throws ArgumentError write_vtk(joinpath(directory, "baddegree"), domain; export_degree=0,
                                         append=false, ascii=true)
  end
end

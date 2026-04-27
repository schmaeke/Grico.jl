function print_lid_driven_cavity_header()
  println("lid_driven_cavity/driver.jl")
  println("  steady discontinuous Galerkin lid-driven cavity")
  @printf("  Reynolds number     : %.1f\n", REYNOLDS_NUMBER)
  @printf("  velocity degree     : %d\n", VELOCITY_DEGREE)
  @printf("  pressure degree     : %d\n", PRESSURE_DEGREE)
  @printf("  adaptive steps      : %d\n", ADAPTIVE_STEPS)
  println("  cycle iter dofs rel-update dg-mass-monitor kinetic-energy")
  return nothing
end

function lid_driven_cavity_point_data()
  return (speed=(x, values) -> sqrt(values.velocity[1]^2 + values.velocity[2]^2),
          horizontal_velocity=(x, values) -> values.velocity[1],
          vertical_velocity=(x, values) -> values.velocity[2])
end

function lid_driven_cavity_cell_data(context)
  current_grid = grid(context.velocity_space)
  return (leaf=leaf -> Float64(leaf), level=leaf -> Float64.(level(current_grid, leaf)),
          velocity_degree=leaf -> Float64.(cell_degrees(context.velocity_space, leaf)),
          pressure_degree=leaf -> Float64.(cell_degrees(context.pressure_space, leaf)))
end

function lid_driven_cavity_field_data(context, iteration_count, final_update)
  return (picard_iterations=Float64(iteration_count), final_relative_update=final_update,
          kinetic_energy=kinetic_energy(context.plan, context.flow_state, context.velocity),
          dg_mass_monitor_l2=dg_mass_monitor_l2(context.plan, context.flow_state, context.velocity),
          Reynolds=REYNOLDS_NUMBER)
end

function write_lid_driven_cavity_vtk(context, iteration_count, final_update)
  # Export both the final fields and enough metadata to understand the adapted
  # mesh afterwards: per-leaf levels, local velocity/pressure degrees, and a few
  # global diagnostic scalars.
  output_directory = joinpath(@__DIR__, "output")
  mkpath(output_directory)
  return write_vtk(joinpath(output_directory, "lid_driven_cavity"), context.flow_state;
                   point_data=lid_driven_cavity_point_data(),
                   cell_data=lid_driven_cavity_cell_data(context),
                   field_data=lid_driven_cavity_field_data(context, iteration_count, final_update),
                   subdivisions=EXPORT_SUBDIVISIONS, sample_degree=EXPORT_DEGREE, append=true,
                   compress=true, ascii=false)
end

function write_lid_driven_cavity_plot(context, iteration_count, final_update)
  CairoMakie = load_lid_driven_cavity_plot_backend()
  output_directory = joinpath(@__DIR__, "output")
  mkpath(output_directory)
  skeleton = sample_mesh_skeleton(context.velocity_space)
  figure = Base.invokelatest(plot_field, context.flow_state, :speed;
                             point_data=lid_driven_cavity_point_data(),
                             cell_data=lid_driven_cavity_cell_data(context),
                             field_data=lid_driven_cavity_field_data(context, iteration_count,
                                                                     final_update),
                             subdivisions=EXPORT_SUBDIVISIONS, sample_degree=EXPORT_DEGREE,
                             mesh=skeleton, colorbar_label="|u|", figure=(size=(720, 640),),
                             axis=(title="Lid-driven cavity speed",))
  figure_path = joinpath(output_directory, "lid_driven_cavity_speed.pdf")
  Base.invokelatest(getproperty(CairoMakie, :save), figure_path, figure)
  return figure_path
end

function load_lid_driven_cavity_plot_backend()
  try
    Base.eval(@__MODULE__, :(using CairoMakie))
  catch error
    occursin("Package CairoMakie", sprint(showerror, error)) || rethrow()
    throw(ArgumentError("Makie figure output requires CairoMakie. Run the example with `julia --project=examples/lid_driven_cavity` after instantiating that environment, or pass `write_plots=false`."))
  end

  return Base.invokelatest(getfield, @__MODULE__, :CairoMakie)
end

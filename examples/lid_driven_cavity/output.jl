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

function write_lid_driven_cavity_vtk(context, iteration_count, final_update)
  # Export both the final fields and enough metadata to understand the adapted
  # mesh afterwards: per-leaf levels, local velocity/pressure degrees, and a few
  # global diagnostic scalars.
  output_directory = joinpath(@__DIR__, "output")
  current_grid = grid(context.velocity_space)
  mkpath(output_directory)
  return write_vtk(joinpath(output_directory, "lid_driven_cavity"), context.flow_state;
                   point_data=(speed=(x, values) -> sqrt(values.velocity[1]^2 +
                                                         values.velocity[2]^2),
                               horizontal_velocity=(x, values) -> values.velocity[1],
                               vertical_velocity=(x, values) -> values.velocity[2]),
                   cell_data=(leaf=leaf -> Float64(leaf),
                              level=leaf -> Float64.(level(current_grid, leaf)),
                              velocity_degree=leaf -> Float64.(cell_degrees(context.velocity_space,
                                                                            leaf)),
                              pressure_degree=leaf -> Float64.(cell_degrees(context.pressure_space,
                                                                            leaf))),
                   field_data=(picard_iterations=Float64(iteration_count),
                               final_relative_update=final_update,
                               kinetic_energy=kinetic_energy(context.plan, context.flow_state,
                                                             context.velocity),
                               dg_mass_monitor_l2=dg_mass_monitor_l2(context.plan,
                                                                     context.flow_state,
                                                                     context.velocity),
                               Reynolds=REYNOLDS_NUMBER), subdivisions=EXPORT_SUBDIVISIONS,
                   export_degree=EXPORT_DEGREE, append=true, compress=true, ascii=false)
end

# Human-facing solve routine for direct execution and interactive exploration.
function run_annular_plate_nitsche_example(;
                                           solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()),
                                           write_vtk=WRITE_VTK, print_summary=true, kwargs...)
  context = build_annular_plate_nitsche_context(; kwargs...)
  state = solve(context.problem; solver)
  error_value = relative_l2_error(state, context.u, context.exact_solution)
  vtk_path = nothing

  if write_vtk
    output_directory = joinpath(@__DIR__, "output")
    current_space = field_space(context.u)
    current_grid = grid(current_space)
    mkpath(output_directory)
    vtk_path = write_vtk(joinpath(output_directory, "annular_plate_nitsche"), state;
                         point_data=(physical=x -> context.is_physical(x) ? 1.0 : 0.0,
                                     abs_error=(x, values) -> context.is_physical(x) ?
                                                              abs(values.u -
                                                                  context.exact_solution(x)) : 0.0),
                         cell_data=(leaf=leaf -> Float64(leaf),
                                    level=leaf -> Float64.(Grico.level(current_grid, leaf)),
                                    degree=leaf -> Float64.(cell_degrees(current_space, leaf))),
                         field_data=(relative_l2_error=error_value,),
                         subdivisions=EXPORT_SUBDIVISIONS, sample_degree=EXPORT_DEGREE, append=true,
                         compress=true, ascii=false)
    print_summary && println("  vtk  $vtk_path")
  end

  if print_summary
    println("annular_plate_nitsche/driver.jl")
    @printf("  degree              : %d\n", context.degree)
    @printf("  active leaves       : %d\n", active_leaf_count(context.space))
    @printf("  scalar dofs         : %d\n", scalar_dof_count(context.space))
    @printf("  relative l2 error   : %.6e\n", error_value)
  end

  return (; context..., state, error_value, vtk_path)
end

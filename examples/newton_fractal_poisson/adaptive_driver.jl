function _format_level_histogram(histogram)
  return join(("$(level):$(count)" for (level, count) in histogram), ",")
end

# Use Julia's direct dense/sparse dispatch by default. The solve hook remains a
# keyword argument so large deep-tree runs can replace it with an iterative or
# external linear solver without changing the example setup.
newton_fractal_direct_solve(matrix, rhs) = matrix \ rhs

# Run the solve-estimate-adapt loop on a single scalar field. Each iteration
# compiles the current mesh, solves the manufactured Poisson problem, records
# `L²` error and refinement diagnostics, optionally writes high-order VTK
# output, and then transfers the field descriptor to the adapted `hp` space.
function run_newton_fractal_poisson_example(; max_h_level=MAX_H_LEVEL, adaptive_steps=max_h_level,
                                            initial_degree=INITIAL_DEGREE, min_degree=MIN_DEGREE,
                                            max_degree=MAX_DEGREE,
                                            quadrature_extra_points=QUADRATURE_EXTRA_POINTS,
                                            adaptivity_tolerance=ADAPTIVITY_TOLERANCE,
                                            adaptivity_smoothness_threshold=ADAPTIVITY_SMOOTHNESS_THRESHOLD,
                                            newton_iterations=NEWTON_ITERATIONS,
                                            plane_scale=NEWTON_PLANE_SCALE,
                                            root_sharpness=NEWTON_ROOT_SHARPNESS,
                                            residual_weight=NEWTON_RESIDUAL_WEIGHT,
                                            residual_contrast=NEWTON_RESIDUAL_CONTRAST,
                                            derivative_regularization=NEWTON_DERIVATIVE_REGULARIZATION,
                                            difference_step=NEWTON_DIFFERENCE_STEP,
                                            linear_solve=newton_fractal_direct_solve,
                                            output_directory=joinpath(@__DIR__, "output"),
                                            write_vtk=WRITE_VTK, sample_degree=EXPORT_DEGREE,
                                            print_summary=true)
  context = build_newton_fractal_poisson_context(; max_h_level, initial_degree, min_degree,
                                                 max_degree, quadrature_extra_points,
                                                 adaptivity_tolerance,
                                                 adaptivity_smoothness_threshold, newton_iterations,
                                                 plane_scale, root_sharpness, residual_weight,
                                                 residual_contrast, derivative_regularization,
                                                 difference_step)
  u = context.u
  history = NamedTuple[]
  vtk_files = String[]
  vtk_steps = Int[]
  vtk_path = nothing
  pvd_path = nothing
  final_plan = nothing
  final_state = nothing
  final_error = NaN

  if print_summary
    println("newton_fractal_poisson/driver.jl")
    println("  step leaves stored max-h dofs rel-l2-error plan level-histogram")
  end

  write_vtk && mkpath(output_directory)

  for step in 0:adaptive_steps
    problem = build_newton_fractal_poisson_problem(u, context)
    assembly_plan = compile(problem)
    state = State(assembly_plan, solve(assemble(assembly_plan); linear_solve))
    error_value = relative_l2_error(state, u, context.exact_solution; plan=assembly_plan,
                                    extra_points=VERIFICATION_EXTRA_POINTS)

    current_space = field_space(u)
    current_grid = grid(current_space)
    current_max_h = newton_fractal_max_h_level(current_space)
    stored_cells = newton_fractal_stored_cell_count(current_space)
    level_histogram = newton_fractal_level_histogram(current_space)

    if step == adaptive_steps || current_max_h >= max_h_level
      adaptivity_plan_value = nothing
      summary = (marked_leaf_count=0, h_refinement_leaf_count=0, h_derefinement_cell_count=0,
                 p_refinement_leaf_count=0, p_derefinement_leaf_count=0)
      step_plan = "done"
      stop_now = true
    else
      adaptivity_plan_value = newton_fractal_adaptivity_plan(state, u, context; max_h_level)
      summary = adaptivity_summary(adaptivity_plan_value)
      step_plan = isempty(adaptivity_plan_value) ? "stop" :
                  "h=$(summary.h_refinement_leaf_count), p=$(summary.p_refinement_leaf_count)"
      stop_now = isempty(adaptivity_plan_value)
    end

    push!(history,
          (; step, active_leaves=active_leaf_count(current_space), stored_cells,
           max_h_level=current_max_h, dofs=scalar_dof_count(current_space), error_value,
           level_histogram, marked_leaf_count=summary.marked_leaf_count,
           h_refinement_leaf_count=summary.h_refinement_leaf_count,
           h_derefinement_cell_count=summary.h_derefinement_cell_count,
           p_refinement_leaf_count=summary.p_refinement_leaf_count,
           p_derefinement_leaf_count=summary.p_derefinement_leaf_count, step_plan))

    if print_summary
      @printf("  %4d %6d %6d %5d %5d %.6e %s %s\n", step, active_leaf_count(current_space),
              stored_cells, current_max_h, scalar_dof_count(current_space), error_value, step_plan,
              _format_level_histogram(level_histogram))
    end

    if write_vtk
      current_domain = current_space.domain
      vtk_path = Grico.write_vtk(joinpath(output_directory,
                                          @sprintf("newton_fractal_poisson_%04d", step)),
                                 current_domain; state,
                                 point_data=(exact=context.exact_solution,
                                             newton_value=x -> Float64(newton_color_value(context.model,
                                                                                          x)),
                                             basin=x -> Float64(newton_basin_index(context.model,
                                                                                   x)),
                                             residual=x -> Float64(newton_residual_indicator(context.model,
                                                                                             x)),
                                             gradient_norm=x -> sqrt(sum(abs2,
                                                                         context.exact_gradient(x))),
                                             abs_error=(x, values) -> abs(values.u -
                                                                          context.exact_solution(x))),
                                 cell_data=(leaf=leaf -> Float64(leaf),
                                            level=leaf -> Float64.(level(current_grid, leaf)),
                                            degree=leaf -> Float64.(cell_degrees(current_space,
                                                                                 leaf)),
                                            basin=leaf -> Float64(newton_basin_index(context.model,
                                                                                     cell_center(current_domain,
                                                                                                 leaf))),
                                            residual=leaf -> Float64(newton_residual_indicator(context.model,
                                                                                               cell_center(current_domain,
                                                                                                           leaf)))),
                                 field_data=(step=Float64(step), max_h_level=Float64(current_max_h),
                                             relative_l2_error=error_value),
                                 subdivisions=EXPORT_SUBDIVISIONS, sample_degree=sample_degree,
                                 append=true, compress=true, ascii=false)
      push!(vtk_files, vtk_path)
      push!(vtk_steps, step)
    end

    final_plan = assembly_plan
    final_state = state
    final_error = error_value

    if stop_now
      if write_vtk && !isempty(vtk_files)
        pvd_path = Grico.write_pvd(joinpath(output_directory, "newton_fractal_poisson.pvd"),
                                   vtk_files; timesteps=vtk_steps)
        print_summary && println("  vtk  $vtk_path")
        print_summary && println("  pvd  $pvd_path")
      end
      break
    end

    space_transition = transition(adaptivity_plan_value)
    u = adapted_field(space_transition, u)
  end

  return (; context..., u, final_plan, final_state, final_error, history, vtk_path, pvd_path)
end

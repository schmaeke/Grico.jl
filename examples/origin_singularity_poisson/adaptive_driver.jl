# Human-facing adaptive driver for the solve-estimate-adapt loop.
function run_origin_singularity_poisson_example(; adaptive_steps=ADAPTIVE_STEPS,
                                                write_vtk=WRITE_VTK, print_summary=true)
  context = build_origin_singularity_poisson_context()
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
    println("origin_singularity_poisson/driver.jl")
    println("  step leaves dofs rel-l2-error plan")
  end

  output_directory = joinpath(@__DIR__, "output")
  write_vtk && mkpath(output_directory)

  for step in 0:adaptive_steps
    problem = build_origin_singularity_problem(u, context)
    plan = compile(problem)
    state = solve(plan; solver=CGSolver(preconditioner=JacobiPreconditioner()))
    error_value = relative_l2_error(state, u, context.exact_solution; plan=plan,
                                    extra_points=VERIFICATION_EXTRA_POINTS)

    if step == adaptive_steps
      step_plan = "done"
      stop_now = true
      adaptivity_plan = nothing
    else
      adaptivity_plan = origin_adaptivity_plan(state, u)
      summary = adaptivity_summary(adaptivity_plan)
      step_plan = isempty(adaptivity_plan) ? "stop" :
                  "h=$(summary.h_refinement_leaf_count), p=$(summary.p_refinement_leaf_count)"
      stop_now = isempty(adaptivity_plan)
    end

    push!(history,
          (; step, active_leaves=active_leaf_count(field_space(u)),
           dofs=scalar_dof_count(field_space(u)), error_value, step_plan))
    print_summary && @printf("  %4d %6d %4d %.6e %s\n", step, active_leaf_count(field_space(u)),
                            scalar_dof_count(field_space(u)), error_value, step_plan)

    if write_vtk
      current_space = field_space(u)
      current_grid = grid(current_space)
      vtk_path = write_vtk(joinpath(output_directory,
                                    @sprintf("origin_singularity_poisson_%04d", step)),
                           current_space.domain; state=state,
                           point_data=(exact=context.exact_solution,
                                       abs_error=(x, values) -> abs(values.u -
                                                                    context.exact_solution(x))),
                           cell_data=(leaf=leaf -> Float64(leaf),
                                      level=leaf -> Float64.(Grico.level(current_grid, leaf)),
                                      degree=leaf -> Float64.(cell_degrees(current_space, leaf))),
                           field_data=(step=Float64(step), relative_l2_error=error_value),
                           subdivisions=EXPORT_SUBDIVISIONS, append=true, compress=true,
                           ascii=false)
      push!(vtk_files, vtk_path)
      push!(vtk_steps, step)
    end

    final_plan = plan
    final_state = state
    final_error = error_value

    if stop_now
      if write_vtk && !isempty(vtk_files)
        pvd_path = write_pvd(joinpath(output_directory, "origin_singularity_poisson.pvd"),
                             vtk_files; timesteps=vtk_steps)
        print_summary && println("  vtk  $vtk_path")
        print_summary && println("  pvd  $pvd_path")
      end
      break
    end

    space_transition = transition(adaptivity_plan)
    u = adapted_field(space_transition, u)
  end

  return (; context..., u, final_plan, final_state, final_error, history, vtk_path, pvd_path)
end

# Human-facing driver used when the example is run directly from the `examples/`
# directory or from the repository root with Julia.
#
# The control structure is:
#
# - outer loop: adapt the mesh a few times,
# - inner loop: converge the Picard iteration on the current mesh.
function run_lid_driven_cavity_example(; max_iters=PICARD_MAX_ITERS, tol=PICARD_TOL,
                                       linear_solve=direct_sparse_solve, write_vtk=WRITE_VTK)
  print_lid_driven_cavity_header()
  context = build_lid_driven_cavity_context()
  final_update = Inf
  iteration_count = 0
  adaptive_step_count = 0

  for adaptive_step in 0:ADAPTIVE_STEPS
    # On intermediate adaptive meshes we allow a looser nonlinear tolerance so
    # the example does not oversolve a mesh that will be replaced immediately.
    cycle_tol = adaptive_step == ADAPTIVE_STEPS ? tol : max(tol, ADAPTIVE_PICARD_TOL)

    for iteration in 1:max_iters
      context, _, relative_update, mass_l2, energy = advance_picard_step(context;
                                                                         linear_solve=linear_solve)
      @printf("  %5d %4d %4d %.6e %.6e %.6e\n", adaptive_step, iteration,
              length(coefficients(context.flow_state)), relative_update, mass_l2, energy)
      final_update = relative_update
      iteration_count = iteration
      relative_update <= cycle_tol && break
    end

    adaptive_step_count = adaptive_step
    adaptive_step == ADAPTIVE_STEPS && break

    # After the nonlinear iteration settles on the current mesh, ask for one
    # velocity-driven DG `h`-adaptation step and transfer both fields if the
    # planner marked anything.
    next_context, adaptivity_plan = adapt_lid_driven_cavity_context(context)
    isempty(adaptivity_plan) && break
    summary = adaptivity_summary(adaptivity_plan)
    @printf("  refine %d marked=%d h+=%d\n", adaptive_step, summary.marked_leaf_count,
            summary.h_refinement_leaf_count)
    context = next_context
  end

  if write_vtk
    vtk_path = write_lid_driven_cavity_vtk(context, iteration_count, final_update)
    println("  vtk  $vtk_path")
  end

  @printf("  active leaves       : %d\n", active_leaf_count(context.velocity_space))
  @printf("  scalar dofs         : %d\n", scalar_dof_count(context.velocity_space))
  @printf("  mixed dofs          : %d\n", length(coefficients(context.flow_state)))
  @printf("  adaptive steps used : %d\n", adaptive_step_count)
  @printf("  final Picard iters  : %d\n", iteration_count)
  @printf("  final rel. update   : %.6e\n", final_update)
  @printf("  dg mass monitor l2  : %.6e\n",
          dg_mass_monitor_l2(context.plan, context.flow_state, context.velocity))
  return (; context, iteration_count, final_update)
end

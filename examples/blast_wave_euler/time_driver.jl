# ---------------------------------------------------------------------------
# 5. Time integration, output, and driver
# ---------------------------------------------------------------------------

# This is the user-facing entry point of the file. It alternates between
#
# 1. fixed-mesh ODE solves,
# 2. optional output writes, and
# 3. optional `h`-adaptation.
#
# The compact history table printed to the terminal tracks mesh size and the
# active explicit timestep. Physical fields are written to VTK for inspection.
function run_blast_wave_euler_example(; root_counts=ROOT_COUNTS, degree=POLYDEG,
                                      quadrature_extra_points=QUADRATURE_EXTRA_POINTS, gamma=GAMMA,
                                      cfl=CFL, final_time=FINAL_TIME, save_interval=SAVE_INTERVAL,
                                      adapt_interval=ADAPT_INTERVAL,
                                      initial_refinement_layers=INITIAL_BLAST_REFINEMENT_LAYERS,
                                      initial_refinement_radius=INITIAL_BLAST_REFINEMENT_RADIUS,
                                      solver=nothing, adaptivity_tolerance=ADAPTIVITY_TOLERANCE,
                                      max_h_level=MAX_H_LEVEL, store_segment_solutions=false,
                                      write_vtk=WRITE_VTK, print_summary=true)
  final_time > 0 || throw(ArgumentError("final_time must be positive"))
  max_h_level >= initial_refinement_layers ||
    throw(ArgumentError("max_h_level must be at least initial_refinement_layers"))
  context = build_blast_wave_euler_context(; root_counts, degree, quadrature_extra_points, gamma,
                                           cfl, initial_refinement_layers,
                                           initial_refinement_radius)
  save_times = saved_times((0.0, final_time), save_interval)
  adapt_times = saved_times((0.0, final_time), adapt_interval)
  times = merge_time_grids(save_times, adapt_times)
  history = NamedTuple[blast_wave_history_entry(0, 0.0, context)]
  adaptivity_history = NamedTuple[]
  segment_solutions = store_segment_solutions ? Any[] : nothing
  vtk_files = String[]
  pvd_path = nothing
  save_index = 2
  adapt_index = 2

  write_vtk && mkpath(joinpath(@__DIR__, "output"))
  write_vtk && push!(vtk_files, write_blast_wave_vtk(context, history[1]; max_h_level=max_h_level))

  if print_summary
    print_blast_wave_header(context, final_time; save_interval=save_interval,
                            adapt_interval=adapt_interval,
                            initial_refinement_layers=initial_refinement_layers,
                            initial_refinement_radius=initial_refinement_radius,
                            adaptivity_tolerance=adaptivity_tolerance, max_h_level=max_h_level)
    print_blast_wave_history_entry(history[1])
  end

  for step in 1:(length(times)-1)
    segment_state, segment_solution = solve_fixed_mesh_segment(context,
                                                               (times[step], times[step + 1]);
                                                               solver=solver,
                                                               return_solution=store_segment_solutions)
    store_segment_solutions && push!(segment_solutions, segment_solution)
    context = refresh_blast_wave_context(context, segment_state)

    while save_index <= length(save_times) && same_time(save_times[save_index], times[step + 1])
      entry = blast_wave_history_entry(length(history), save_times[save_index], context)
      push!(history, entry)
      print_summary && print_blast_wave_history_entry(entry)
      write_vtk && push!(vtk_files, write_blast_wave_vtk(context, entry; max_h_level=max_h_level))
      save_index += 1
    end

    while adapt_index <= length(adapt_times) && same_time(adapt_times[adapt_index], times[step + 1])
      if !same_time(adapt_times[adapt_index], final_time)
        before_context = context
        context, adaptivity_plan = adapt_blast_wave_context(before_context;
                                                            tolerance=adaptivity_tolerance,
                                                            max_h_level=max_h_level)
        adaptivity_entry = blast_wave_adaptivity_entry(step, adapt_times[adapt_index],
                                                       before_context, context, adaptivity_plan)
        push!(adaptivity_history, adaptivity_entry)
      end
      adapt_index += 1
    end
  end

  if write_vtk
    pvd_path = write_pvd(joinpath(@__DIR__, "output", "blast_wave_euler.pvd"), vtk_files;
                         timesteps=[entry.time for entry in history])
    print_summary && println("  vtk  $(last(vtk_files))")
    print_summary && println("  pvd  $pvd_path")
  end

  return (; context, history, adaptivity_history, vtk_files, pvd_path, segment_solutions)
end

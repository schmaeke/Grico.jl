function blast_wave_history_entry(step, time, context)
  return (; step, time=Float64(time), active_leaves=active_leaf_count(context.space),
          dofs=length(coefficients(context.state)), dt=context.dt,
          limited_cells=context.limiter_stats.limited_cell_count,
          limiter_theta=context.limiter_stats.minimum_theta,
          min_density=context.limiter_stats.minimum_density,
          min_pressure=context.limiter_stats.minimum_pressure)
end

sampled_conserved(values, field) = getproperty(values, field_name(field))

function blast_wave_adaptivity_limits(context; max_h_level=MAX_H_LEVEL)
  return AdaptivityLimits(context.space; min_p=context.degree, max_p=context.degree,
                          max_h_level=max_h_level)
end

function blast_wave_adaptivity_entry(step, time, before_context, after_context, plan)
  summary = adaptivity_summary(plan)

  return (; step, time=Float64(time), before_active_leaves=active_leaf_count(before_context.space),
          after_active_leaves=active_leaf_count(after_context.space),
          before_dofs=length(coefficients(before_context.state)),
          after_dofs=length(coefficients(after_context.state)),
          marked_leaf_count=summary.marked_leaf_count,
          h_refinement_leaf_count=summary.h_refinement_leaf_count,
          h_derefinement_cell_count=summary.h_derefinement_cell_count,
          p_refinement_leaf_count=summary.p_refinement_leaf_count,
          p_derefinement_leaf_count=summary.p_derefinement_leaf_count)
end

# Export the active refinement signal as leaf-wise cell data so it is easy to
# inspect where the current adaptive mesh logic sees under-resolution.
function blast_wave_refinement_indicator_data(context; max_h_level=MAX_H_LEVEL)
  limits = blast_wave_adaptivity_limits(context; max_h_level=max_h_level)
  indicators = Grico.multiresolution_indicators(context.state, context.conserved; limits=limits)
  axis_values = Matrix{Float64}(undef, Grico.dimension(context.space), length(indicators))
  norms = Vector{Float64}(undef, length(indicators))

  for leaf_index in eachindex(indicators)
    values = indicators[leaf_index]
    norms[leaf_index] = sqrt(sum(abs2, values))

    for axis in 1:length(values)
      axis_values[axis, leaf_index] = Float64(values[axis])
    end
  end

  return axis_values, norms
end

# Export the current DG state. The point data expose the main physical fields a
# new user typically wants to inspect first: density, velocity, and pressure,
# while the cell data carry leaf metadata and the current adaptivity signal.
function write_blast_wave_vtk(context, entry; output_directory=joinpath(@__DIR__, "output"),
                              max_h_level=MAX_H_LEVEL)
  current_grid = grid(context.domain)
  refinement_indicator, refinement_indicator_norm = blast_wave_refinement_indicator_data(context;
                                                                                         max_h_level=max_h_level)
  return write_vtk(joinpath(output_directory, @sprintf("blast_wave_euler_%04d", entry.step)),
                   context.state; fields=(context.conserved,),
                   point_data=(density=(x, values) -> sampled_conserved(values, context.conserved)[1],
                               log_density=(x, values) -> log(max(sampled_conserved(values,
                                                                                    context.conserved)[1],
                                                                  DENSITY_FLOOR)),
                               velocity=(x, values) -> begin
                                 q = sampled_conserved(values, context.conserved)
                                 velocity(q, context.gamma)
                               end,
                               pressure=(x, values) -> begin
                                 q = sampled_conserved(values, context.conserved)
                                 pressure(q, context.gamma)
                               end),
                   cell_data=(leaf=leaf -> Float64(leaf),
                              level=leaf -> Float64.(Grico.level(current_grid, leaf)),
                              degree=leaf -> Float64.(cell_degrees(context.space, leaf)),
                              refinement_indicator=refinement_indicator,
                              refinement_indicator_norm=refinement_indicator_norm),
                   field_data=(time=entry.time, dt=entry.dt), subdivisions=EXPORT_SUBDIVISIONS,
                   sample_degree=EXPORT_DEGREE, append=true, compress=true, ascii=false)
end

# Print a compact run header so the solver configuration is visible without
# opening the file again while the example is running.
function print_blast_wave_header(context, final_time; save_interval=SAVE_INTERVAL,
                                 adapt_interval=ADAPT_INTERVAL,
                                 adaptivity_tolerance=ADAPTIVITY_TOLERANCE, max_h_level=MAX_H_LEVEL,
                                 initial_refinement_layers=INITIAL_BLAST_REFINEMENT_LAYERS,
                                 initial_refinement_radius=INITIAL_BLAST_REFINEMENT_RADIUS)
  println("blast_wave_euler/driver.jl")
  @printf("  domain             : [%.1f, %.1f] x [%.1f, %.1f] (periodic Sedov setup)\n",
          origin(context.domain, 1), origin(context.domain, 1) + extent(context.domain, 1),
          origin(context.domain, 2), origin(context.domain, 2) + extent(context.domain, 2))
  @printf("  roots              : %d x %d\n", root_cell_counts(grid(context.domain))...)
  @printf("  base degree        : %d\n", context.degree)
  @printf("  cfl                : %.3f\n", context.cfl)
  @printf("  ambient rho / p    : %.2f / %.1e\n", BACKGROUND_DENSITY, BACKGROUND_PRESSURE)
  @printf("  sigma rho / p      : %.3f / %.3f\n", DENSITY_SIGMA, PRESSURE_SIGMA)
  @printf("  positivity limiter : %s\n", context.limiter.enabled ? "on" : "off")
  @printf("  initial h layers   : %d\n", initial_refinement_layers)
  @printf("  initial ref radius : %.3f\n", initial_refinement_radius)
  @printf("  save interval      : %.3f\n", save_interval)
  @printf("  adapt interval     : %.3f\n", adapt_interval)
  @printf("  max h level        : %d\n", max_h_level)
  @printf("  adapt tolerance    : %.2e\n", adaptivity_tolerance)

  @printf("  final time         : %.3f\n", final_time)
  println("  step time leaves dofs dt limited theta min_rho min_p")
end

function print_blast_wave_history_entry(entry)
  @printf("  %4d %.3f %5d %7d %.6e %5d %.3e %.3e %.3e\n", entry.step, entry.time,
          entry.active_leaves, entry.dofs, entry.dt, entry.limited_cells, entry.limiter_theta,
          entry.min_density, entry.min_pressure)
end

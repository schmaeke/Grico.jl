# Shared driver utilities for adaptive nonlinear solve benchmarks.
#
# Nonlinear scripts supply the initial field and residual problem construction.
# This file owns the Newton loop, per-iteration timing, CSV output, plotting,
# warm-up, and the same hp-adaptation diagnostics used by affine benchmarks.

function nonlinear_benchmark_defaults(; newton_iterations=8, newton_tolerance=1.0e-8,
                                      newton_damping=1.0, adaptive_options...)
  defaults = adaptive_benchmark_defaults(; adaptive_options...)
  defaults["newton_iterations"] = newton_iterations
  defaults["newton_tolerance"] = newton_tolerance
  defaults["newton_damping"] = newton_damping
  return defaults
end

function _nonlinear_extra_help(defaults)
  return """
    --newton-iterations N      maximum Newton corrections per cycle (default: $(defaults["newton_iterations"]))
    --newton-tolerance T       nonlinear residual tolerance (default: $(defaults["newton_tolerance"]))
    --newton-damping T         Newton correction damping factor (default: $(defaults["newton_damping"]))
"""
end

function _parse_nonlinear_common_option!(options, args, index::Int)
  value, next_index = _option_value(args, index, "--newton-iterations")

  if value !== nothing
    options["newton_iterations"] = _positive_int(value, "newton-iterations")
    return next_index
  end

  value, next_index = _option_value(args, index, "--newton-tolerance")

  if value !== nothing
    options["newton_tolerance"] = _positive_float(value, "newton-tolerance")
    return next_index
  end

  value, next_index = _option_value(args, index, "--newton-damping")

  if value !== nothing
    options["newton_damping"] = _positive_float(value, "newton-damping")
    return next_index
  end

  return nothing
end

function parse_nonlinear_benchmark_options(args; script_name::AbstractString, defaults,
                                           extra_help::AbstractString="",
                                           (parse_extra_option!)=(_no_extra_option!))
  combined_help = extra_help * _nonlinear_extra_help(defaults)

  combined_parser = function (options, parser_args, index::Int)
    next_index = _parse_nonlinear_common_option!(options, parser_args, index)
    next_index === nothing || return next_index
    return parse_extra_option!(options, parser_args, index)
  end

  return parse_adaptive_benchmark_options(args; script_name, defaults, extra_help=combined_help,
                                          (parse_extra_option!)=combined_parser)
end

function _timed_nonlinear_component!(thunk, rows::Vector{NamedTuple}, cycle::Int,
                                     component::AbstractString)
  timing = @timed thunk()
  push!(rows,
        (; cycle, component=String(component), time_seconds=timing.time, bytes=timing.bytes,
         gc_time_seconds=timing.gctime))
  return timing
end

function _iteration_row(cycle::Int, iteration::Int, residual_norm::Float64, update_norm::Float64,
                        residual_time::Float64, tangent_time::Float64, linear_solve_time::Float64,
                        update_time::Float64, converged::Bool)
  return (; cycle, iteration, residual_norm, update_norm, residual_time_seconds=residual_time,
          tangent_time_seconds=tangent_time, linear_solve_time_seconds=linear_solve_time,
          update_time_seconds=update_time, converged)
end

function _newton_solve!(component_rows, iteration_rows, cycle::Int, plan, state, options)
  tolerance = Float64(options["newton_tolerance"])
  damping = eltype(coefficients(state))(options["newton_damping"])
  correction_count = Int(options["newton_iterations"])
  workspace = ResidualWorkspace(plan)
  reduced_workspace = Grico._ReducedOperatorWorkspace(plan)
  reduced_values = zeros(eltype(coefficients(state)), Grico.reduced_dof_count(plan))
  Grico._compress_reduced!(reduced_values, Grico._reduced_map(plan), coefficients(state))
  residual_vector = zeros(eltype(coefficients(state)), length(reduced_values))
  correction_rhs = similar(residual_vector)
  final_residual_norm = Inf
  converged = false
  iteration_count = 0

  for iteration in 1:correction_count
    iteration_count = iteration
    residual_timing = _timed_nonlinear_component!(component_rows, cycle, "newton_residual") do
      Grico._reduced_residual!(residual_vector, plan, reduced_values, reduced_workspace, workspace)
    end
    final_residual_norm = Float64(norm(residual_vector))

    if final_residual_norm <= tolerance
      converged = true
      push!(iteration_rows,
            _iteration_row(cycle, iteration, final_residual_norm, 0.0, residual_timing.time, 0.0,
                           0.0, 0.0, true))
      final_state = Grico._state_from_reduced!(plan, reduced_workspace, reduced_values)
      copyto!(coefficients(state), coefficients(final_state))
      return (; state, iteration_count, correction_count=iteration - 1,
              residual_norm=final_residual_norm, converged)
    end

    @inbounds for index in eachindex(residual_vector)
      correction_rhs[index] = -residual_vector[index]
    end

    linear_solve_timing = _timed_nonlinear_component!(component_rows, cycle,
                                                      "newton_linear_solve") do
      Grico.default_tangent_linear_solve(plan, reduced_workspace.state, correction_rhs;
                                         workspace=reduced_workspace, residual_workspace=workspace)
    end
    correction = linear_solve_timing.value
    update_norm = Float64(norm(correction))
    update_timing = _timed_nonlinear_component!(component_rows, cycle, "newton_state_update") do
      reduced_values .+= damping .* correction
      nothing
    end

    push!(iteration_rows,
          _iteration_row(cycle, iteration, final_residual_norm, update_norm, residual_timing.time,
                         0.0, linear_solve_timing.time, update_timing.time, false))
  end

  iteration_count = correction_count + 1
  residual_timing = _timed_nonlinear_component!(component_rows, cycle, "newton_residual") do
    Grico._reduced_residual!(residual_vector, plan, reduced_values, reduced_workspace, workspace)
  end
  final_residual_norm = Float64(norm(residual_vector))
  converged = final_residual_norm <= tolerance
  push!(iteration_rows,
        _iteration_row(cycle, iteration_count, final_residual_norm, 0.0, residual_timing.time, 0.0,
                       0.0, 0.0, converged))

  final_state = Grico._state_from_reduced!(plan, reduced_workspace, reduced_values)
  copyto!(coefficients(state), coefficients(final_state))
  return (; state, iteration_count, correction_count, residual_norm=final_residual_norm, converged)
end

function _nonlinear_cycle_summary(cycle::Int, field, plan, solution_l2::Float64, adaptivity,
                                  target_field, newton_result)
  space = field_space(field)
  target_space_value = field_space(target_field)
  summary = adaptivity === nothing ? nothing : adaptivity_summary(adaptivity)
  empty_plan = adaptivity === nothing || isempty(adaptivity)
  min_p_degree, max_p_degree = _p_degree_bounds(space)
  target_min_p_degree, target_max_p_degree = _p_degree_bounds(target_space_value)

  return (; cycle, active_leaves=active_leaf_count(space),
          stored_cells=Grico.stored_cell_count(grid(space)), max_h_level=_max_h_level(space),
          min_p_degree, max_p_degree, dofs=scalar_dof_count(space),
          reduced_dofs=Grico.reduced_dof_count(plan), solution_l2,
          newton_iterations=newton_result.iteration_count,
          newton_corrections=newton_result.correction_count,
          final_residual_norm=newton_result.residual_norm, converged=newton_result.converged,
          marked_leaf_count=summary === nothing ? 0 : summary.marked_leaf_count,
          h_refinement_leaf_count=summary === nothing ? 0 : summary.h_refinement_leaf_count,
          h_derefinement_cell_count=summary === nothing ? 0 : summary.h_derefinement_cell_count,
          p_refinement_leaf_count=summary === nothing ? 0 : summary.p_refinement_leaf_count,
          p_derefinement_leaf_count=summary === nothing ? 0 : summary.p_derefinement_leaf_count,
          target_active_leaves=active_leaf_count(target_space_value), target_min_p_degree,
          target_max_p_degree, target_dofs=scalar_dof_count(target_space_value), stopped=empty_plan)
end

function _print_nonlinear_cycle_row(row)
  @printf("  %3d leaves=%6d dofs=%7d reduced=%7d p=%d:%d l2=%10.3e newton=%2d residual=%9.2e h=%5d p+=%5d target=%6d target_p=%d:%d\n",
          row.cycle, row.active_leaves, row.dofs, row.reduced_dofs, row.min_p_degree,
          row.max_p_degree, row.solution_l2, row.newton_corrections, row.final_residual_norm,
          row.h_refinement_leaf_count, row.p_refinement_leaf_count, row.target_active_leaves,
          row.target_min_p_degree, row.target_max_p_degree)
  return nothing
end

function run_nonlinear_adaptive_cycles(options; build_initial_field, build_problem, output_prefix,
                                       plot_title, solution_reference=0.0, write_files::Bool=true,
                                       print_progress::Bool=true)
  field = build_initial_field(options)
  reference = _solution_reference(solution_reference, options)
  cycle_rows = NamedTuple[]
  component_rows = NamedTuple[]
  iteration_rows = NamedTuple[]
  initial_state = nothing

  for cycle in 1:options["cycles"]
    first_component_row = length(component_rows) + 1
    cycle_start = time()
    problem = _timed!(component_rows, cycle, "problem_setup") do
      build_problem(field, options)
    end
    plan = _timed!(component_rows, cycle, "compile") do
      compile(problem)
    end
    state = initial_state === nothing ? State(plan) : initial_state
    newton_result = _newton_solve!(component_rows, iteration_rows, cycle, plan, state, options)
    solution_l2 = _timed!(component_rows, cycle, "solution_norm") do
      _solution_l2_norm(state, field, plan, reference)
    end
    limits = _adaptivity_limits(field_space(field), options)
    adaptivity = _timed!(component_rows, cycle, "adaptivity_plan") do
      adaptivity_plan(state, field; tolerance=options["tolerance"],
                      smoothness_threshold=options["smoothness_threshold"], limits=limits)
    end

    target_field = field
    initial_state = nothing

    if !isempty(adaptivity)
      transition_data = _timed!(component_rows, cycle, "transition") do
        transition(adaptivity)
      end
      target_field = _timed!(component_rows, cycle, "adapted_field") do
        adapted_field(transition_data, field)
      end
      initial_state = _timed!(component_rows, cycle, "transfer_state") do
        transfer_state(transition_data, state, field, target_field)
      end
    end

    _push_total_row!(component_rows, cycle, first_component_row, time() - cycle_start)
    push!(cycle_rows,
          _nonlinear_cycle_summary(cycle, field, plan, solution_l2, adaptivity, target_field,
                                   newton_result))
    print_progress && _print_nonlinear_cycle_row(last(cycle_rows))
    field = target_field
    isempty(adaptivity) && break
  end

  if write_files
    mkpath(options["output_directory"])
    cycle_path = joinpath(options["output_directory"], "$(output_prefix)_cycles.csv")
    component_path = joinpath(options["output_directory"], "$(output_prefix)_components.csv")
    iteration_path = joinpath(options["output_directory"], "$(output_prefix)_iterations.csv")
    _write_csv(cycle_path, cycle_rows)
    _write_csv(component_path, component_rows)
    _write_csv(iteration_path, iteration_rows)
    options["write_plots"] &&
      _write_plots(options["output_directory"], component_rows, output_prefix, plot_title)
    println("wrote $cycle_path")
    println("wrote $component_path")
    println("wrote $iteration_path")
  end

  return cycle_rows, component_rows, iteration_rows
end

function _warmup_nonlinear_benchmark(options; build_initial_field, build_problem, output_prefix,
                                     plot_title, solution_reference)
  warmup_options = copy(options)
  warmup_options["cycles"] = 1
  warmup_options["output_directory"] = ""
  warmup_options["write_plots"] = false
  warmup_options["warmup"] = false
  run_nonlinear_adaptive_cycles(warmup_options; build_initial_field, build_problem, output_prefix,
                                plot_title, solution_reference, write_files=false,
                                print_progress=false)
  return nothing
end

function run_nonlinear_adaptive_benchmark(args; script_name::AbstractString,
                                          benchmark_title::AbstractString,
                                          output_prefix::AbstractString, defaults,
                                          build_initial_field, build_problem,
                                          solution_reference=0.0, extra_help::AbstractString="",
                                          (parse_extra_option!)=(_no_extra_option!))
  options = parse_nonlinear_benchmark_options(args; script_name, defaults, extra_help,
                                              parse_extra_option!)

  if options["warmup"]
    println("warming up benchmark kernels")
    _warmup_nonlinear_benchmark(options; build_initial_field, build_problem, output_prefix,
                                plot_title=benchmark_title, solution_reference)
  end

  println(benchmark_title)
  @printf("  cycles=%d root_cells=%s degree=%d max_h_level=%s tolerance=%.3e newton=%d tol=%.3e damping=%.3e\n",
          options["cycles"], options["root_cells"], options["degree"],
          _limit_text(options["max_h_level"]), options["tolerance"], options["newton_iterations"],
          options["newton_tolerance"], options["newton_damping"])
  run_nonlinear_adaptive_cycles(options; build_initial_field, build_problem, output_prefix,
                                plot_title=benchmark_title, solution_reference, write_files=true)
  return nothing
end

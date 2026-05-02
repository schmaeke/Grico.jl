# Shared driver utilities for adaptive solve benchmarks.
#
# Benchmark scripts supply the initial field and problem construction. This file
# owns the command-line interface, timing layout, CSV output, plotting, warm-up,
# and per-cycle mesh/space diagnostics.

const GRICO_BENCHMARK_ROOT = normpath(joinpath(@__DIR__, ".."))
GRICO_BENCHMARK_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, GRICO_BENCHMARK_ROOT)

using Grico
using LinearAlgebra
using Printf

function adaptive_benchmark_defaults(; cycles=8, root_cells=(20, 20), degree=1,
                                     quadrature_extra_points=2, max_h_level=12, tolerance=1.0e-5,
                                     smoothness_threshold=0.25,
                                     output_directory=joinpath(@__DIR__, "output"),
                                     write_plots=false, warmup=true, compact_transition=false)
  return Dict{String,Any}("cycles" => cycles, "root_cells" => root_cells, "degree" => degree,
                          "quadrature_extra_points" => quadrature_extra_points,
                          "max_h_level" => max_h_level, "tolerance" => tolerance,
                          "smoothness_threshold" => smoothness_threshold,
                          "output_directory" => output_directory, "write_plots" => write_plots,
                          "warmup" => warmup, "compact_transition" => compact_transition)
end

struct BenchmarkReferenceProvider{F}
  provider::F
end

benchmark_reference_provider(provider) = BenchmarkReferenceProvider(provider)
_solution_reference(reference, options) = reference
_solution_reference(reference::BenchmarkReferenceProvider, options) = reference.provider(options)

function _positive_int(text::AbstractString, name::AbstractString)
  value = tryparse(Int, text)
  value !== nothing && value >= 1 || throw(ArgumentError("$name must be a positive integer"))
  return value
end

function _nonnegative_int(text::AbstractString, name::AbstractString)
  value = tryparse(Int, text)
  value !== nothing && value >= 0 || throw(ArgumentError("$name must be a non-negative integer"))
  return value
end

function _positive_float(text::AbstractString, name::AbstractString)
  value = tryparse(Float64, text)
  value !== nothing && isfinite(value) && value > 0.0 ||
    throw(ArgumentError("$name must be a finite positive number"))
  return value
end

function _nonnegative_float(text::AbstractString, name::AbstractString)
  value = tryparse(Float64, text)
  value !== nothing && isfinite(value) && value >= 0.0 ||
    throw(ArgumentError("$name must be a finite non-negative number"))
  return value
end

function _root_cell_tuple(text::AbstractString)
  parts = split(text, ',')
  length(parts) == 2 || throw(ArgumentError("root-cells must have the form Nx,Ny"))
  return (_positive_int(strip(parts[1]), "root-cells[1]"),
          _positive_int(strip(parts[2]), "root-cells[2]"))
end

function _next_option_value(args, index::Int, option::AbstractString)
  index < length(args) || throw(ArgumentError("$option requires a value"))
  return args[index+1], index + 1
end

function _option_value(args, index::Int, option::AbstractString)
  arg = args[index]
  prefix = option * "="
  startswith(arg, prefix) && return arg[(lastindex(prefix)+1):end], index
  arg == option && return _next_option_value(args, index, option)
  return nothing, index
end

_limit_text(value) = isnothing(value) ? "unrestricted" : string(value)
_root_cells_text(value) = "$(value[1]),$(value[2])"

function _print_help(script_name::AbstractString, defaults, extra_help::AbstractString)
  println("""
  $script_name

  Options:
    --cycles N                 measured solve/adapt cycles (default: $(defaults["cycles"]))
    --root-cells Nx,Ny         initial Cartesian root cells (default: $(_root_cells_text(defaults["root_cells"])))
    --degree P                 initial polynomial degree (default: $(defaults["degree"]))
    --quadrature-extra N       extra cell quadrature points over degree (default: $(defaults["quadrature_extra_points"]))
    --max-h-level N            optional per-axis maximum h level (default: $(_limit_text(defaults["max_h_level"])))
    --unrestricted-h           remove the maximum h-level cap
    --tolerance T              adaptivity detail tolerance (default: $(defaults["tolerance"]))
    --smoothness-threshold T   h/p classifier threshold, kept for diagnostics (default: $(defaults["smoothness_threshold"]))
  """)
  isempty(extra_help) || print(extra_help)
  println("""
    --output DIR               output directory (default: $(defaults["output_directory"]))
    --plots                    write PDF runtime plots with Plots.jl
    --compact-transition       compile adapted target spaces on compacted grids
    --no-warmup                include first-use compilation latency in the measurements
    --help                     show this message
  """)
  return nothing
end

function _parse_common_value_option!(options, args, index::Int)
  value, next_index = _option_value(args, index, "--cycles")

  if value !== nothing
    options["cycles"] = _positive_int(value, "cycles")
    return next_index
  end

  value, next_index = _option_value(args, index, "--root-cells")

  if value !== nothing
    options["root_cells"] = _root_cell_tuple(value)
    return next_index
  end

  value, next_index = _option_value(args, index, "--degree")

  if value !== nothing
    options["degree"] = _positive_int(value, "degree")
    return next_index
  end

  value, next_index = _option_value(args, index, "--quadrature-extra")

  if value !== nothing
    options["quadrature_extra_points"] = _nonnegative_int(value, "quadrature-extra")
    return next_index
  end

  value, next_index = _option_value(args, index, "--max-h-level")

  if value !== nothing
    options["max_h_level"] = _nonnegative_int(value, "max-h-level")
    return next_index
  end

  value, next_index = _option_value(args, index, "--tolerance")

  if value !== nothing
    options["tolerance"] = _nonnegative_float(value, "tolerance")
    return next_index
  end

  value, next_index = _option_value(args, index, "--smoothness-threshold")

  if value !== nothing
    options["smoothness_threshold"] = _nonnegative_float(value, "smoothness-threshold")
    return next_index
  end

  value, next_index = _option_value(args, index, "--output")

  if value !== nothing
    options["output_directory"] = value
    return next_index
  end

  return nothing
end

_no_extra_option!(options, args, index::Int) = nothing

function parse_adaptive_benchmark_options(args; script_name::AbstractString, defaults,
                                          extra_help::AbstractString="",
                                          (parse_extra_option!)=(_no_extra_option!))
  options = copy(defaults)
  index = 1

  while index <= length(args)
    arg = args[index]
    next_index = index

    if arg == "--help" || arg == "-h"
      _print_help(script_name, defaults, extra_help)
      exit(0)
    elseif arg == "--plots"
      options["write_plots"] = true
    elseif arg == "--compact-transition"
      options["compact_transition"] = true
    elseif arg == "--no-warmup"
      options["warmup"] = false
    elseif arg == "--unrestricted-h"
      options["max_h_level"] = nothing
    else
      next_index = _parse_common_value_option!(options, args, index)
      next_index === nothing && (next_index = parse_extra_option!(options, args, index))
      next_index === nothing && throw(ArgumentError("unknown option $arg"))
    end

    index = next_index + 1
  end

  return options
end

function _adaptivity_limits(space::HpSpace, options)
  max_h_level = options["max_h_level"]
  isnothing(max_h_level) && return AdaptivityLimits(space)
  return AdaptivityLimits(space; max_h_level)
end

function _max_h_level(space::HpSpace)
  grid_data = grid(space)
  value = 0

  for leaf in active_leaves(space)
    value = max(value, maximum(level(grid_data, leaf)))
  end

  return value
end

function _p_degree_bounds(space::HpSpace)
  minimum_degree = typemax(Int)
  maximum_degree = 0

  for leaf in active_leaves(space)
    degrees = cell_degrees(space, leaf)
    minimum_degree = min(minimum_degree, minimum(degrees))
    maximum_degree = max(maximum_degree, maximum(degrees))
  end

  return minimum_degree, maximum_degree
end

function _timed!(thunk, rows::Vector{NamedTuple}, cycle::Int, component::AbstractString)
  timing = @timed thunk()
  push!(rows,
        (; cycle, component=String(component), time_seconds=timing.time, bytes=timing.bytes,
         gc_time_seconds=timing.gctime))
  return timing.value
end

function _solution_l2_norm(state::State, field, plan, reference)
  return l2_error(state, field, reference; plan=plan, extra_points=1)
end

function _cycle_summary(cycle::Int, field, plan, solution_l2::Float64, adaptivity, target_field)
  space = field_space(field)
  target_space_value = field_space(target_field)
  summary = adaptivity === nothing ? nothing : adaptivity_summary(adaptivity)
  empty_plan = adaptivity === nothing || isempty(adaptivity)
  min_p_degree, max_p_degree = _p_degree_bounds(space)
  target_min_p_degree, target_max_p_degree = _p_degree_bounds(target_space_value)

  return (; cycle, active_leaves=active_leaf_count(space),
          stored_cells=Grico.stored_cell_count(grid(space)), max_h_level=_max_h_level(space),
          min_p_degree, max_p_degree, dofs=scalar_dof_count(space),
          reduced_dofs=Grico.reduced_dof_count(plan),
          solution_l2, marked_leaf_count=summary === nothing ? 0 : summary.marked_leaf_count,
          h_refinement_leaf_count=summary === nothing ? 0 : summary.h_refinement_leaf_count,
          h_derefinement_cell_count=summary === nothing ? 0 : summary.h_derefinement_cell_count,
          p_refinement_leaf_count=summary === nothing ? 0 : summary.p_refinement_leaf_count,
          p_derefinement_leaf_count=summary === nothing ? 0 : summary.p_derefinement_leaf_count,
          target_active_leaves=active_leaf_count(target_space_value), target_min_p_degree,
          target_max_p_degree, target_dofs=scalar_dof_count(target_space_value), stopped=empty_plan)
end

function _push_total_row!(component_rows, cycle::Int, first_row::Int, total_time::Float64)
  total_bytes = sum(component_rows[index].bytes for index in first_row:length(component_rows))
  total_gc = sum(component_rows[index].gc_time_seconds
                 for index in first_row:length(component_rows))
  push!(component_rows,
        (; cycle, component="total", time_seconds=total_time, bytes=total_bytes,
         gc_time_seconds=total_gc))
  return component_rows
end

function _print_cycle_row(row)
  @printf("  %3d leaves=%6d dofs=%7d reduced=%7d p=%d:%d l2=%10.3e h=%5d p+=%5d target=%6d target_p=%d:%d\n",
          row.cycle, row.active_leaves, row.dofs, row.reduced_dofs, row.min_p_degree,
          row.max_p_degree, row.solution_l2, row.h_refinement_leaf_count,
          row.p_refinement_leaf_count, row.target_active_leaves, row.target_min_p_degree,
          row.target_max_p_degree)
  return nothing
end

function _csv_escape(value)
  text = string(value)
  needs_quotes = any(character -> character in (',', '"', '\n', '\r'), text)
  text = replace(text, "\"" => "\"\"")
  return needs_quotes ? "\"$text\"" : text
end

function _write_csv(path::AbstractString, rows::Vector{NamedTuple})
  open(path, "w") do io
    isempty(rows) && return nothing
    names = propertynames(first(rows))
    println(io, join(names, ","))

    for row in rows
      println(io, join((_csv_escape(getproperty(row, name)) for name in names), ","))
    end
  end

  return path
end

function _component_order(rows)
  preferred = ("problem_setup", "compile", "solve", "newton_residual", "newton_tangent",
               "newton_linear_solve", "newton_state_update", "solution_norm", "adaptivity_plan",
               "transition", "adapted_field", "transfer_state")
  present = Set(row.component for row in rows if row.component != "total")
  ordered = [component for component in preferred if component in present]
  extras = sort!(collect(setdiff(present, Set(ordered))))
  return vcat(ordered, extras)
end

function _runtime_series(rows, cycles, components)
  values = fill(NaN, length(cycles), length(components))

  for (cycle_index, cycle) in enumerate(cycles)
    for (component_index, component) in enumerate(components)
      time_seconds = 0.0
      found = false

      for row in rows
        row.cycle == cycle && row.component == component || continue
        time_seconds += row.time_seconds
        found = true
      end

      found && (values[cycle_index, component_index] = max(time_seconds, eps(Float64)))
    end
  end

  return values
end

function _component_markers(count::Int)
  shapes = (:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :hexagon, :cross, :xcross,
            :pentagon)
  return [shapes[mod1(index, length(shapes))] for index in 1:count]
end

function _load_plots()
  package_id = Base.PkgId(Base.UUID("91a5bcdd-55d7-5caf-9e0b-520d859cae80"), "Plots")

  try
    get!(ENV, "GKSwstype", "100")
    return Base.require(package_id)
  catch error
    message = sprint(showerror, error)
    @warn "Plots.jl is not available; skipping benchmark plots" error=message
    return nothing
  end
end

function _write_plots(output_directory::AbstractString, rows, output_prefix::AbstractString,
                      title::AbstractString)
  plots = _load_plots()
  plots === nothing && return nothing

  cycles = sort!(unique(row.cycle for row in rows))
  components = _component_order(rows)
  values = _runtime_series(rows, cycles, components)
  markers = _component_markers(length(components))
  plot_path = joinpath(output_directory, "$(output_prefix)_components.pdf")

  try
    runtime_plot = nothing

    for component_index in eachindex(components)
      series = values[:, component_index]
      label = components[component_index]
      marker = markers[component_index]

      if runtime_plot === nothing
        runtime_plot = Base.invokelatest(plots.plot, cycles, series; label=label, marker=marker,
                                         linewidth=2, markersize=5, yscale=:log10, xticks=cycles,
                                         xlabel="adaptation cycle", ylabel="seconds", title=title,
                                         legend=:outerright, size=(1100, 650))
      else
        Base.invokelatest(plots.plot!, runtime_plot, cycles, series; label=label, marker=marker,
                          linewidth=2, markersize=5)
      end
    end

    Base.invokelatest(plots.savefig, runtime_plot, plot_path)
  catch error
    message = sprint(showerror, error)
    @warn "Could not write benchmark plots" error=message
    return nothing
  end

  println("wrote $plot_path")
  return plot_path
end

function run_adaptive_cycles(options; build_initial_field, build_problem, output_prefix, plot_title,
                             solution_reference=0.0, write_files::Bool=true,
                             print_progress::Bool=true)
  field = build_initial_field(options)
  reference = _solution_reference(solution_reference, options)
  cycle_rows = NamedTuple[]
  component_rows = NamedTuple[]

  for cycle in 1:options["cycles"]
    # Each cycle measures the public workflow phases separately, then advances
    # the field through the computed adaptation transition.
    first_component_row = length(component_rows) + 1
    cycle_start = time()
    problem = _timed!(component_rows, cycle, "problem_setup") do
      build_problem(field, options)
    end
    plan = _timed!(component_rows, cycle, "compile") do
      compile(problem)
    end
    state = _timed!(component_rows, cycle, "solve") do
      solve(plan; preconditioner=JacobiPreconditioner())
    end
    solution_l2 = _timed!(component_rows, cycle, "solution_norm") do
      _solution_l2_norm(state, field, plan, reference)
    end
    limits = _adaptivity_limits(field_space(field), options)
    adaptivity = _timed!(component_rows, cycle, "adaptivity_plan") do
      adaptivity_plan(state, field; tolerance=options["tolerance"],
                      smoothness_threshold=options["smoothness_threshold"], limits=limits)
    end

    target_field = field

    if !isempty(adaptivity)
      transition_data = _timed!(component_rows, cycle, "transition") do
        transition(adaptivity; compact=options["compact_transition"])
      end
      target_field = _timed!(component_rows, cycle, "adapted_field") do
        adapted_field(transition_data, field)
      end
      _timed!(component_rows, cycle, "transfer_state") do
        transfer_state(transition_data, state, field, target_field)
      end
    end

    _push_total_row!(component_rows, cycle, first_component_row, time() - cycle_start)
    push!(cycle_rows, _cycle_summary(cycle, field, plan, solution_l2, adaptivity, target_field))
    print_progress && _print_cycle_row(last(cycle_rows))
    field = target_field
    isempty(adaptivity) && break
  end

  if write_files
    mkpath(options["output_directory"])
    cycle_path = joinpath(options["output_directory"], "$(output_prefix)_cycles.csv")
    component_path = joinpath(options["output_directory"], "$(output_prefix)_components.csv")
    _write_csv(cycle_path, cycle_rows)
    _write_csv(component_path, component_rows)
    options["write_plots"] &&
      _write_plots(options["output_directory"], component_rows, output_prefix, plot_title)
    println("wrote $cycle_path")
    println("wrote $component_path")
  end

  return cycle_rows, component_rows
end

function _warmup_adaptive_benchmark(options; build_initial_field, build_problem, output_prefix,
                                    plot_title, solution_reference)
  warmup_options = copy(options)
  warmup_options["cycles"] = 1
  warmup_options["output_directory"] = ""
  warmup_options["write_plots"] = false
  warmup_options["warmup"] = false
  run_adaptive_cycles(warmup_options; build_initial_field, build_problem, output_prefix, plot_title,
                      solution_reference, write_files=false, print_progress=false)
  return nothing
end

function run_adaptive_benchmark(args; script_name::AbstractString, benchmark_title::AbstractString,
                                output_prefix::AbstractString, defaults, build_initial_field,
                                build_problem, solution_reference=0.0,
                                extra_help::AbstractString="",
                                (parse_extra_option!)=(_no_extra_option!))
  options = parse_adaptive_benchmark_options(args; script_name, defaults, extra_help,
                                             parse_extra_option!)

  if options["warmup"]
    println("warming up benchmark kernels")
    _warmup_adaptive_benchmark(options; build_initial_field, build_problem, output_prefix,
                               plot_title=benchmark_title, solution_reference)
  end

  println(benchmark_title)
  @printf("  cycles=%d root_cells=%s degree=%d max_h_level=%s tolerance=%.3e compact_transition=%s\n",
          options["cycles"], options["root_cells"], options["degree"],
          _limit_text(options["max_h_level"]), options["tolerance"], options["compact_transition"])
  run_adaptive_cycles(options; build_initial_field, build_problem, output_prefix,
                      plot_title=benchmark_title, solution_reference, write_files=true)
  return nothing
end

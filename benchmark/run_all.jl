# Run the standalone benchmark suite in separate Julia processes.
#
# Each benchmark script owns its problem setup and output files. This driver
# only orchestrates execution, forwards shared options, and writes a compact
# summary CSV for the full run. Running each benchmark in its own process keeps
# compile latency and method definitions isolated in the same way as direct
# script execution.

using Printf

const BENCHMARK_DIRECTORY = @__DIR__
const DEFAULT_OUTPUT_DIRECTORY = joinpath(BENCHMARK_DIRECTORY, "output")

const BENCHMARKS = (;
                    steady_poisson_cg=(script="steady_poisson_adaptivity.jl",
                                       description="continuous-Galerkin sine-source Poisson"),
                    steady_poisson_sipg_dg=(script="steady_poisson_sipg_dg_adaptivity.jl",
                                            description="SIPG discontinuous-Galerkin sine-source Poisson"),
                    nonlinear_bratu_cg=(script="nonlinear_bratu_cg_adaptivity.jl",
                                        description="nonlinear continuous-Galerkin Bratu solve"),
                    annular_nitsche_fcm=(script="annular_nitsche_fcm_adaptivity.jl",
                                         description="annular Nitsche finite-cell immersed boundary"))

const SHARED_VALUE_OPTIONS = Set(["--cycles", "--root-cells", "--degree", "--quadrature-extra",
                                  "--max-h-level", "--tolerance", "--smoothness-threshold"])
const SHARED_FLAG_OPTIONS = Set(["--unrestricted-h"])

_benchmark_names_text() = join((String(name) for name in keys(BENCHMARKS)), ", ")

function _benchmark_descriptions_text()
  buffer = IOBuffer()

  for (name, benchmark) in pairs(BENCHMARKS)
    @printf(buffer, "    %-24s %s\n", String(name), benchmark.description)
  end

  return String(take!(buffer))
end

function _print_help()
  println("""
  run_all.jl

  Run the standalone benchmark suite.

  Options:
    --benchmarks LIST          comma-separated subset (default: all)
                               available: $(_benchmark_names_text())
    --output DIR               output directory passed to every benchmark
                               (default: $DEFAULT_OUTPUT_DIRECTORY)
    --plots                    ask every benchmark to write PDF plots
    --no-warmup                include first-use compilation latency in measurements
    --continue-on-error        run remaining benchmarks after a failure
    --dry-run                  print child commands without executing them
    --help                     show this message

  Available benchmarks:
$(_benchmark_descriptions_text())
  Shared benchmark options are forwarded when present:
    --cycles, --root-cells, --degree, --quadrature-extra, --max-h-level,
    --unrestricted-h, --tolerance, --smoothness-threshold

  Arguments after -- are forwarded to every selected benchmark without
  validation. Use that for benchmark-specific options only when the selected
  benchmark subset supports them.
  """)
  return nothing
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

function _parse_benchmark_names(text::AbstractString)
  names = Symbol[]

  for raw_name in split(text, ',')
    name = Symbol(strip(raw_name))
    haskey(BENCHMARKS, name) || throw(ArgumentError("unknown benchmark $name"))
    push!(names, name)
  end

  isempty(names) && throw(ArgumentError("benchmark list must not be empty"))
  return Tuple(names)
end

function _push_shared_value_option!(forwarded_args, args, index::Int, option::AbstractString)
  value, next_index = _option_value(args, index, option)
  value === nothing && return nothing

  if next_index == index && startswith(args[index], option * "=")
    push!(forwarded_args, args[index])
  else
    push!(forwarded_args, option)
    push!(forwarded_args, value)
  end

  return next_index
end

function _parse_run_all_options(args)
  options = Dict{String,Any}("benchmarks" => Tuple(keys(BENCHMARKS)),
                             "output_directory" => DEFAULT_OUTPUT_DIRECTORY, "plots" => false,
                             "warmup" => true, "continue_on_error" => false, "dry_run" => false,
                             "forwarded_args" => String[])
  index = 1

  while index <= length(args)
    arg = args[index]

    if arg == "--"
      append!(options["forwarded_args"], args[(index+1):end])
      break
    elseif arg == "--help" || arg == "-h"
      _print_help()
      exit(0)
    elseif arg == "--plots"
      options["plots"] = true
    elseif arg == "--no-warmup"
      options["warmup"] = false
    elseif arg == "--continue-on-error"
      options["continue_on_error"] = true
    elseif arg == "--dry-run"
      options["dry_run"] = true
    elseif arg == "--benchmarks" || startswith(arg, "--benchmarks=")
      value, next_index = _option_value(args, index, "--benchmarks")
      options["benchmarks"] = _parse_benchmark_names(value)
      index = next_index
    elseif arg == "--output" || startswith(arg, "--output=")
      value, next_index = _option_value(args, index, "--output")
      options["output_directory"] = value
      index = next_index
    elseif arg in SHARED_FLAG_OPTIONS
      push!(options["forwarded_args"], arg)
    else
      matched = false

      for option in SHARED_VALUE_OPTIONS
        arg == option || startswith(arg, option * "=") || continue
        next_index = _push_shared_value_option!(options["forwarded_args"], args, index, option)
        index = next_index
        matched = true
        break
      end

      matched || throw(ArgumentError("unknown option $arg"))
    end

    index += 1
  end

  return options
end

function _benchmark_arguments(options)
  args = String["--output", options["output_directory"]]
  options["plots"] && push!(args, "--plots")
  options["warmup"] || push!(args, "--no-warmup")
  append!(args, options["forwarded_args"])
  return args
end

function _benchmark_command(benchmark, args)
  script_path = joinpath(BENCHMARK_DIRECTORY, benchmark.script)
  command_words = vcat(Base.julia_cmd().exec, ["--project=$(BENCHMARK_DIRECTORY)", script_path],
                       args)
  return Cmd(command_words)
end

function _csv_escape(value)
  text = string(value)
  needs_quotes = any(character -> character in (',', '"', '\n', '\r'), text)
  text = replace(text, "\"" => "\"\"")
  return needs_quotes ? "\"$text\"" : text
end

function _write_summary(path::AbstractString, rows)
  open(path, "w") do io
    println(io, "benchmark,script,status,elapsed_seconds,command")

    for row in rows
      println(io,
              join((_csv_escape(row.benchmark), _csv_escape(row.script), _csv_escape(row.status),
                    _csv_escape(row.elapsed_seconds), _csv_escape(row.command)), ","))
    end
  end

  return path
end

function _run_benchmark(name::Symbol, benchmark, args; dry_run::Bool)
  command = _benchmark_command(benchmark, args)
  println()
  println("running $name")
  println(command)
  dry_run && return (; status="dry-run", elapsed_seconds=0.0, command=command)

  start_time = time()
  status = "ok"

  try
    run(command)
  catch error
    error isa ProcessFailedException || rethrow()
    status = "failed"
  end

  elapsed = time() - start_time
  @printf("finished %s status=%s elapsed=%.3f s\n", String(name), status, elapsed)
  return (; status, elapsed_seconds=elapsed, command)
end

function main(args=ARGS)
  options = _parse_run_all_options(args)
  benchmark_args = _benchmark_arguments(options)
  rows = NamedTuple[]

  mkpath(options["output_directory"])
  println("Grico benchmark suite")
  println("  output=$(options["output_directory"])")
  println("  benchmarks=$(join((String(name) for name in options["benchmarks"]), ", "))")

  for name in options["benchmarks"]
    benchmark = BENCHMARKS[name]
    result = _run_benchmark(name, benchmark, benchmark_args; dry_run=options["dry_run"])
    push!(rows,
          (; benchmark=String(name), script=benchmark.script, status=result.status,
           elapsed_seconds=result.elapsed_seconds, command=string(result.command)))

    result.status == "failed" && !options["continue_on_error"] && break
  end

  summary_path = joinpath(options["output_directory"], "run_all_summary.csv")
  _write_summary(summary_path, rows)
  println()
  println("wrote $summary_path")

  any(row.status == "failed" for row in rows) && exit(1)
  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

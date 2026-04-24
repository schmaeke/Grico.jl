# Adaptive discontinuous-Galerkin SIPG Poisson benchmark on a unit square with a
# discontinuous sine-interface source term.
#
# Run from the repository root with
#
#   julia --project=benchmark benchmark/steady_poisson_sipg_dg_adaptivity.jl --cycles=5
#
# Add `--plots` to write log-scale runtime PDF plots through Plots.jl. The CSV
# files are always written and are the stable machine-readable benchmark output.

include("adaptive_benchmark_common.jl")
include("sine_poisson_common.jl")
include("sine_poisson_sipg.jl")

const SIPG_DG_POISSON_DEFAULTS = let defaults = adaptive_benchmark_defaults()
  defaults["penalty"] = 20.0
  defaults
end

const SIPG_DG_EXTRA_HELP = """
    --penalty ETA              SIPG penalty multiplier (default: $(SIPG_DG_POISSON_DEFAULTS["penalty"]))
"""

function _parse_sipg_dg_option!(options, args, index::Int)
  value, next_index = _option_value(args, index, "--penalty")

  if value !== nothing
    options["penalty"] = _positive_float(value, "penalty")
    return next_index
  end

  return nothing
end

_sipg_dg_initial_field(options) = sine_interface_poisson_field(options; continuity=:dg)

_sipg_dg_poisson_problem(field, options) = sine_interface_sipg_problem(field, options)

function main(args=ARGS)
  run_adaptive_benchmark(args; script_name="steady_poisson_sipg_dg_adaptivity.jl",
                         benchmark_title="steady Poisson SIPG DG adaptivity benchmark",
                         output_prefix="steady_poisson_sipg_dg", defaults=SIPG_DG_POISSON_DEFAULTS,
                         build_initial_field=_sipg_dg_initial_field,
                         build_problem=_sipg_dg_poisson_problem, extra_help=SIPG_DG_EXTRA_HELP,
                         (parse_extra_option!)=(_parse_sipg_dg_option!))
  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

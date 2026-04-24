# Adaptive continuous-Galerkin Poisson benchmark on a unit square with a
# discontinuous sine-interface source term.
#
# Run from the repository root with
#
#   julia --project=benchmark benchmark/steady_poisson_adaptivity.jl --cycles=5
#
# Add `--plots` to write log-scale runtime PDF plots through Plots.jl. The CSV
# files are always written and are the stable machine-readable benchmark output.

include("adaptive_benchmark_common.jl")
include("sine_poisson_common.jl")

const CG_POISSON_DEFAULTS = adaptive_benchmark_defaults()

_cg_initial_field(options) = sine_interface_poisson_field(options; continuity=:cg)

function _cg_poisson_problem(field, options)
  problem = AffineProblem(field)
  add_sine_interface_poisson_cells!(problem, field)

  for axis in 1:2
    add_constraint!(problem, Dirichlet(field, BoundaryFace(axis, LOWER), 0.0))
    add_constraint!(problem, Dirichlet(field, BoundaryFace(axis, UPPER), 0.0))
  end

  return problem
end

function main(args=ARGS)
  run_adaptive_benchmark(args; script_name="steady_poisson_adaptivity.jl",
                         benchmark_title="steady Poisson CG adaptivity benchmark",
                         output_prefix="steady_poisson", defaults=CG_POISSON_DEFAULTS,
                         build_initial_field=_cg_initial_field, build_problem=_cg_poisson_problem)
  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

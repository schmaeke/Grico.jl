# Adaptive nonlinear continuous-Galerkin Bratu benchmark.
#
# The equation is -Δu + λ exp(u) = f on the unit square with homogeneous
# Dirichlet data. The source f is the same sine-interface indicator used by the
# affine Poisson benchmarks, so the solve combines nonlinear Newton work with
# localized adaptive refinement along a non-smooth right-hand side.

include("adaptive_benchmark_common.jl")
include("nonlinear_benchmark_common.jl")
include("sine_poisson_common.jl")

import Grico: cell_residual!, cell_tangent_apply!

const BRATU_CG_DEFAULTS = nonlinear_benchmark_defaults(; cycles=5, root_cells=(12, 12), degree=2,
                                                       max_h_level=8, tolerance=1.0e-5,
                                                       newton_iterations=8, newton_tolerance=1.0e-8)
BRATU_CG_DEFAULTS["lambda"] = 1.0

const BRATU_CG_EXTRA_HELP = """
    --lambda T                 Bratu reaction strength (default: $(BRATU_CG_DEFAULTS["lambda"]))
"""

struct BratuSineSource{F,T}
  field::F
  lambda::T
end

function _parse_bratu_option!(options, args, index::Int)
  value, next_index = _option_value(args, index, "--lambda")

  if value !== nothing
    options["lambda"] = _positive_float(value, "lambda")
    return next_index
  end

  return nothing
end

_bratu_initial_field(options) = sine_interface_poisson_field(options; continuity=:cg)

function _bratu_problem(field, options)
  problem = ResidualProblem(field)
  add_cell!(problem, BratuSineSource(field, options["lambda"]))

  for axis in 1:2
    add_constraint!(problem, Dirichlet(field, BoundaryFace(axis, LOWER), 0.0))
    add_constraint!(problem, Dirichlet(field, BoundaryFace(axis, UPPER), 0.0))
  end

  return problem
end

# The residual represents ∫Ω ∇v · ∇u + λ exp(u) v - f v dΩ. The discontinuous
# f keeps the adaptivity problem close to the affine Poisson benchmark while the
# exponential reaction makes each cycle exercise residual and tangent assembly.
function cell_residual!(local_residual, operator::BratuSineSource, values::CellValues, state)
  local_block = block(local_residual, values, operator.field)
  shape_table = shape_values(values, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    u_value = value(values, state, operator.field, point_index)
    u_gradient = gradient(values, state, operator.field, point_index)
    reaction_minus_source = operator.lambda * exp(u_value) -
                            sine_interface_source(point(values, point_index))
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      gradient_term = zero(eltype(local_residual))

      for axis in 1:axis_count
        gradient_term += gradients[axis, row_mode, point_index] * u_gradient[axis]
      end

      local_block[row_mode] += (gradient_term +
                                reaction_minus_source * shape_table[row_mode, point_index]) *
                               weighted
    end
  end

  return nothing
end

# The tangent action is the Newton linearization
# ∫Ω ∇v · ∇δu + λ exp(u) v δu dΩ applied directly to the local increment.
function cell_tangent_apply!(local_result, operator::BratuSineSource, values::CellValues, state,
                             local_increment)
  local_block = block(local_result, values, operator.field)
  shape_table = shape_values(values, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    reaction = operator.lambda * exp(value(values, state, operator.field, point_index))
    increment_value = value(values, local_increment, operator.field, point_index)
    increment_gradient = gradient(values, local_increment, operator.field, point_index)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      stiffness = zero(eltype(local_result))

      for axis in 1:axis_count
        stiffness += gradients[axis, row_mode, point_index] * increment_gradient[axis]
      end

      local_block[row_mode] += (stiffness +
                                reaction * shape_table[row_mode, point_index] * increment_value) *
                               weighted
    end
  end

  return nothing
end

function main(args=ARGS)
  run_nonlinear_adaptive_benchmark(args; script_name="nonlinear_bratu_cg_adaptivity.jl",
                                   benchmark_title="nonlinear Bratu CG adaptivity benchmark",
                                   output_prefix="nonlinear_bratu_cg", defaults=BRATU_CG_DEFAULTS,
                                   build_initial_field=_bratu_initial_field,
                                   build_problem=_bratu_problem, extra_help=BRATU_CG_EXTRA_HELP,
                                   (parse_extra_option!)=(_parse_bratu_option!))
  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

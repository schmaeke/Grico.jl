# Adaptive immersed-boundary Poisson benchmark on an annulus.
#
# The physical domain is an annulus cut out of a Cartesian background grid. The
# exact harmonic log-radius solution is imposed weakly on both circular embedded
# boundaries with a symmetric Nitsche term. This benchmark exercises finite-cell
# quadrature, embedded surface quadrature, and unfitted boundary assembly.

include("adaptive_benchmark_common.jl")

import Grico: cell_matrix!, surface_matrix!, surface_rhs!

const ANNULAR_FCM_DEFAULTS = adaptive_benchmark_defaults(; cycles=6, root_cells=(16, 16), degree=2,
                                                         max_h_level=8, tolerance=1.0e-5,
                                                         smoothness_threshold=0.25)
ANNULAR_FCM_DEFAULTS["inner_radius"] = 0.35
ANNULAR_FCM_DEFAULTS["outer_radius"] = 1.0
ANNULAR_FCM_DEFAULTS["segments"] = 160
ANNULAR_FCM_DEFAULTS["surface_points"] = 4
ANNULAR_FCM_DEFAULTS["fcm_depth"] = 4
ANNULAR_FCM_DEFAULTS["penalty"] = 20.0

const ANNULAR_FCM_EXTRA_HELP = """
    --inner-radius R           inner annulus radius (default: $(ANNULAR_FCM_DEFAULTS["inner_radius"]))
    --outer-radius R           outer annulus radius (default: $(ANNULAR_FCM_DEFAULTS["outer_radius"]))
    --segments N               line segments per circle (default: $(ANNULAR_FCM_DEFAULTS["segments"]))
    --surface-points N         embedded segment quadrature points (default: $(ANNULAR_FCM_DEFAULTS["surface_points"]))
    --fcm-depth N              finite-cell subdivision depth (default: $(ANNULAR_FCM_DEFAULTS["fcm_depth"]))
    --penalty ETA              Nitsche penalty multiplier (default: $(ANNULAR_FCM_DEFAULTS["penalty"]))
"""

struct AnnularDiffusion{F}
  field::F
end

struct AnnularNitscheDirichlet{F,G,T}
  field::F
  data::G
  penalty::T
end

function _parse_annular_option!(options, args, index::Int)
  value, next_index = _option_value(args, index, "--inner-radius")

  if value !== nothing
    options["inner_radius"] = _positive_float(value, "inner-radius")
    return next_index
  end

  value, next_index = _option_value(args, index, "--outer-radius")

  if value !== nothing
    options["outer_radius"] = _positive_float(value, "outer-radius")
    return next_index
  end

  value, next_index = _option_value(args, index, "--segments")

  if value !== nothing
    options["segments"] = _positive_int(value, "segments")
    return next_index
  end

  value, next_index = _option_value(args, index, "--surface-points")

  if value !== nothing
    options["surface_points"] = _positive_int(value, "surface-points")
    return next_index
  end

  value, next_index = _option_value(args, index, "--fcm-depth")

  if value !== nothing
    options["fcm_depth"] = _nonnegative_int(value, "fcm-depth")
    return next_index
  end

  value, next_index = _option_value(args, index, "--penalty")

  if value !== nothing
    options["penalty"] = _positive_float(value, "penalty")
    return next_index
  end

  return nothing
end

function _annular_exact_solution(inner_radius, outer_radius)
  denominator = log(inner_radius / outer_radius)
  return x -> log(hypot(x[1], x[2]) / outer_radius) / denominator
end

function _annular_levelset(inner_radius, outer_radius)
  return x -> max(hypot(x[1], x[2]) - outer_radius, inner_radius - hypot(x[1], x[2]))
end

function _circle_points_segments(radius::Float64, segment_count::Int; clockwise::Bool=false)
  points = Vector{NTuple{2,Float64}}(undef, segment_count)

  for index in 1:segment_count
    angle = 2π * (index - 1) / segment_count
    clockwise && (angle = -angle)
    points[index] = (radius * cos(angle), radius * sin(angle))
  end

  segments = [(index, index == segment_count ? 1 : index + 1) for index in 1:segment_count]
  return points, segments
end

function _annular_surface(options)
  inner_radius = Float64(options["inner_radius"])
  outer_radius = Float64(options["outer_radius"])
  segment_count = Int(options["segments"])
  outer_points, outer_segments = _circle_points_segments(outer_radius, segment_count)
  inner_points, inner_segments = _circle_points_segments(inner_radius, segment_count;
                                                         clockwise=true)
  offset = length(outer_points)
  points = vcat(outer_points, inner_points)
  segments = vcat(outer_segments,
                  [(first + offset, second + offset) for (first, second) in inner_segments])
  return EmbeddedSurface(SegmentMesh(points, segments); point_count=options["surface_points"])
end

function _check_annular_options(options)
  inner_radius = options["inner_radius"]
  outer_radius = options["outer_radius"]
  inner_radius < outer_radius ||
    throw(ArgumentError("inner-radius must be smaller than outer-radius"))
  return nothing
end

function _annular_initial_field(options)
  _check_annular_options(options)
  outer_radius = Float64(options["outer_radius"])
  background = Domain((-outer_radius, -outer_radius), (2 * outer_radius, 2 * outer_radius),
                      options["root_cells"])
  levelset = _annular_levelset(options["inner_radius"], options["outer_radius"])
  region = ImplicitRegion(levelset; subdivision_depth=options["fcm_depth"])
  domain = PhysicalDomain(background, region)
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(options["degree"]),
                               quadrature=DegreePlusQuadrature(options["quadrature_extra_points"])))
  return ScalarField(space; name=:u)
end

function _annular_problem(field, options)
  exact = _annular_exact_solution(options["inner_radius"], options["outer_radius"])
  problem = AffineProblem(field)
  add_cell!(problem, AnnularDiffusion(field))
  add_surface!(problem, AnnularNitscheDirichlet(field, exact, options["penalty"]))
  add_embedded_surface!(problem, _annular_surface(options))
  return problem
end

function _annular_reference(options)
  return _annular_exact_solution(options["inner_radius"], options["outer_radius"])
end

function cell_matrix!(local_matrix, operator::AnnularDiffusion, values::CellValues)
  local_block = block(local_matrix, values, operator.field, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      for col_mode in 1:row_mode
        contribution = zero(eltype(local_matrix))

        for axis in 1:axis_count
          contribution += gradients[axis, row_mode, point_index] *
                          gradients[axis, col_mode, point_index]
        end

        contribution *= weighted
        local_block[row_mode, col_mode] += contribution
        row_mode == col_mode || (local_block[col_mode, row_mode] += contribution)
      end
    end
  end

  return nothing
end

function surface_matrix!(local_matrix, operator::AnnularNitscheDirichlet, values::SurfaceValues)
  local_block = block(local_matrix, values, operator.field, operator.field)
  shape_table = shape_values(values, operator.field)
  mode_count = local_mode_count(values, operator.field)
  domain_data = field_space(operator.field).domain
  h = min(cell_size(domain_data, values.leaf, 1), cell_size(domain_data, values.leaf, 2))
  penalty = operator.penalty / h

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)
    normal_data = normal(values, point_index)

    for row_mode in 1:mode_count
      row_value = shape_table[row_mode, point_index]
      row_gradient = shape_gradient(values, operator.field, point_index, row_mode)
      row_flux = row_gradient[1] * normal_data[1] + row_gradient[2] * normal_data[2]

      for col_mode in 1:row_mode
        col_value = shape_table[col_mode, point_index]
        col_gradient = shape_gradient(values, operator.field, point_index, col_mode)
        col_flux = col_gradient[1] * normal_data[1] + col_gradient[2] * normal_data[2]
        contribution = weighted * (-row_value * col_flux - col_value * row_flux +
                                   penalty * row_value * col_value)
        local_block[row_mode, col_mode] += contribution
        row_mode == col_mode || (local_block[col_mode, row_mode] += contribution)
      end
    end
  end

  return nothing
end

function surface_rhs!(local_rhs, operator::AnnularNitscheDirichlet, values::SurfaceValues)
  local_block = block(local_rhs, values, operator.field)
  mode_count = local_mode_count(values, operator.field)
  domain_data = field_space(operator.field).domain
  h = min(cell_size(domain_data, values.leaf, 1), cell_size(domain_data, values.leaf, 2))
  penalty = operator.penalty / h

  @inbounds for point_index in 1:point_count(values)
    boundary_value = operator.data(point(values, point_index))
    weighted = weight(values, point_index)
    normal_data = normal(values, point_index)

    for mode_index in 1:mode_count
      mode_value = shape_value(values, operator.field, point_index, mode_index)
      mode_gradient = shape_gradient(values, operator.field, point_index, mode_index)
      mode_flux = mode_gradient[1] * normal_data[1] + mode_gradient[2] * normal_data[2]
      local_block[mode_index] += weighted * (-boundary_value * mode_flux +
                                             penalty * boundary_value * mode_value)
    end
  end

  return nothing
end

function main(args=ARGS)
  run_adaptive_benchmark(args; script_name="annular_nitsche_fcm_adaptivity.jl",
                         benchmark_title="annular Nitsche FCM adaptivity benchmark",
                         output_prefix="annular_nitsche_fcm", defaults=ANNULAR_FCM_DEFAULTS,
                         build_initial_field=_annular_initial_field, build_problem=_annular_problem,
                         solution_reference=benchmark_reference_provider(_annular_reference),
                         extra_help=ANNULAR_FCM_EXTRA_HELP,
                         (parse_extra_option!)=(_parse_annular_option!))
  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

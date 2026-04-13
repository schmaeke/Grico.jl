using Printf
using Grico
import Grico: cell_matrix!, surface_matrix!, surface_rhs!

# This example is the compact "unfitted methods" tour of the package.
#
# We solve the scalar Laplace problem on the annulus
#
#   Ω = {x ∈ ℝ² : Rᵢ ≤ ‖x‖ ≤ Rₒ}
#
# by embedding the curved physical domain into a Cartesian background mesh. The
# volume bilinear form is integrated with finite-cell quadrature on the cut
# cells, while the Dirichlet boundary condition on the circular boundary is
# imposed weakly by a symmetric Nitsche method on an embedded segment mesh.
#
# The exact harmonic solution is radial,
#
#   u(r) = log(r / Rₒ) / log(Rᵢ / Rₒ),
#
# so Δu = 0 in the annulus, u = 1 on the inner circle, and u = 0 on the outer
# circle. This makes the example a compact demonstration of three ideas at
# once:
#
# 1. finite-cell quadrature on a cut Cartesian background grid,
# 2. embedded-surface assembly on an explicit segment mesh, and
# 3. weak Dirichlet enforcement by Nitsche terms instead of boundary-fitted
#    trace constraints.
#
# File roadmap:
#
# 1. choose the annulus geometry and discretization parameters,
# 2. define the volume and surface weak-form contributions,
# 3. build the background mesh and embedded boundary representation,
# 4. attach finite-cell quadratures on all active leaves,
# 5. solve, verify, and export.

# ---------------------------------------------------------------------------
# 1. Problem geometry and discretization parameters
# ---------------------------------------------------------------------------
#
# The physical domain is the annulus, but the finite-element space lives on the
# enclosing square. Only the quadrature rules "know" which part of each
# Cartesian leaf belongs to the physical domain.
const INNER_RADIUS = 0.35
const OUTER_RADIUS = 1.0
const ROOT_COUNTS = (2, 2)
const DEGREE = 4

# The circular boundary is represented by a polygonal segment mesh. The segment
# count controls the geometric approximation of the circles, while
# `SURFACE_POINT_COUNT` controls the one-dimensional quadrature order used on
# each segment during embedded-surface assembly.
const SEGMENT_COUNT = 128
const SURFACE_POINT_COUNT = 3

# Finite-cell quadrature recursively subdivides cut cells until the physical
# part of the cell is resolved well enough for the moment-fitting procedure.
const FCM_SUBDIVISION_DEPTH = 7

# The symmetric Nitsche penalty scales like η / h. The value here is chosen
# large enough to stabilize the weak Dirichlet enforcement without dominating
# the consistency terms.
const NITSCHE_PENALTY = 40.0

# Optional VTK output settings.
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 4
# Benchmarks may include this file for its reusable builders without running
# the full example. Direct execution keeps the default autorun behavior.
const RUN_ANNULAR_PLATE_NITSCHE = get(ENV, "GRICO_ANNULAR_AUTORUN", "1") == "1"

# ---------------------------------------------------------------------------
# 2. Local weak forms
# ---------------------------------------------------------------------------
#
# The interior operator is the standard Laplace bilinear form. The boundary
# condition is not applied by a fitted boundary trace constraint because the
# boundary is curved and cuts through Cartesian cells. Instead, we add a
# separate surface operator later.

# Standard Laplace bilinear form
#
#   a(v, u) = ∫_Ω ∇v · ∇u dΩ.
#
# The field is scalar, so the local matrix is assembled over one scalar block.
struct Diffusion{F}
  field::F
end

function cell_matrix!(local_matrix, operator::Diffusion, values::CellValues)
  local_block = block(local_matrix, values, operator.field, operator.field)
  mode_count = local_mode_count(values, operator.field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      gradient_row = shape_gradient(values, operator.field, point_index, row_mode)

      for col_mode in 1:mode_count
        gradient_col = shape_gradient(values, operator.field, point_index, col_mode)
        local_block[row_mode, col_mode] += weighted * sum(gradient_row[axis] * gradient_col[axis]
                                                          for axis in eachindex(gradient_row))
      end
    end
  end

  return nothing
end

# Symmetric Nitsche boundary operator for a prescribed Dirichlet datum g. The
# embedded surface contributes
#
#   ∫_Γ (-v ∂ₙu - u ∂ₙv + η h⁻¹ u v) dΓ
#
# to the matrix and
#
#   ∫_Γ (-g ∂ₙv + η h⁻¹ g v) dΓ
#
# to the right-hand side.
#
# The consistency terms recover the weak form of the Dirichlet problem, the
# symmetry term mirrors the first consistency term, and the penalty term restores
# coercivity on unfitted cells.
#
# For a reader new to Nitsche methods, the key point is: the boundary condition
# is enforced weakly by extra integral terms instead of by directly eliminating
# boundary degrees of freedom.
struct NitscheDirichlet{F,G,T}
  field::F
  data::G
  penalty::T
end

function surface_matrix!(local_matrix, operator::NitscheDirichlet, values::SurfaceValues)
  local_block = block(local_matrix, values, operator.field, operator.field)
  mode_count = local_mode_count(values, operator.field)
  domain = field_space(operator.field).domain

  # On the Cartesian background mesh, a simple and robust local length scale is
  # the smaller side length of the leaf that contains the current embedded
  # segment quadrature points. This is the `h` that appears in the Nitsche
  # penalty scaling η / h.
  h = min(cell_size(domain, values.leaf, 1), cell_size(domain, values.leaf, 2))
  penalty = operator.penalty / h

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)
    normal_data = normal(values, point_index)

    for row_mode in 1:mode_count
      value_row = shape_value(values, operator.field, point_index, row_mode)
      gradient_row = shape_gradient(values, operator.field, point_index, row_mode)
      flux_row = sum(gradient_row[axis] * normal_data[axis] for axis in eachindex(normal_data))

      for col_mode in 1:mode_count
        value_col = shape_value(values, operator.field, point_index, col_mode)
        gradient_col = shape_gradient(values, operator.field, point_index, col_mode)
        flux_col = sum(gradient_col[axis] * normal_data[axis] for axis in eachindex(normal_data))
        local_block[row_mode, col_mode] += weighted *
                                           (-value_row * flux_col - value_col * flux_row +
                                            penalty * value_row * value_col)
      end
    end
  end

  return nothing
end

function surface_rhs!(local_rhs, operator::NitscheDirichlet, values::SurfaceValues)
  local_block = block(local_rhs, values, operator.field)
  mode_count = local_mode_count(values, operator.field)
  domain = field_space(operator.field).domain
  h = min(cell_size(domain, values.leaf, 1), cell_size(domain, values.leaf, 2))
  penalty = operator.penalty / h

  for point_index in 1:point_count(values)
    x = point(values, point_index)
    g = operator.data(x)
    weighted = weight(values, point_index)
    normal_data = normal(values, point_index)

    for mode_index in 1:mode_count
      value_i = shape_value(values, operator.field, point_index, mode_index)
      gradient_i = shape_gradient(values, operator.field, point_index, mode_index)
      flux_i = sum(gradient_i[axis] * normal_data[axis] for axis in eachindex(normal_data))
      local_block[mode_index] += weighted * (-g * flux_i + penalty * g * value_i)
    end
  end

  return nothing
end

# ---------------------------------------------------------------------------
# 3. Background mesh and embedded boundary geometry
# ---------------------------------------------------------------------------
#
# The background finite-element mesh is just a square. The curved circles are
# represented separately by a segment mesh. This split is central to unfitted
# methods: geometry and approximation space no longer need to match exactly.
#
# Build a polygonal approximation of a circle as a closed segment mesh. The
# `clockwise` option is used to reverse the inner boundary orientation so the
# resulting two-component boundary has a consistent global orientation. That
# matters because the outward normal on the inner boundary points toward the
# hole, not toward the outer square.
function circle_points_segments(radius, segment_count; clockwise=false)
  angles = range(0.0, 2 * pi; length=segment_count + 1)[1:(end-1)]
  points = [(radius * cos(angle), radius * sin(angle)) for angle in angles]
  clockwise && reverse!(points)
  segments = [(index, index == segment_count ? 1 : index + 1) for index in 1:segment_count]
  return points, segments
end

# Build the full unfitted problem description once so both the example driver
# and the validation harness use the same quadrature, embedded geometry, and
# weak boundary data setup.
function build_annular_plate_nitsche_context(; inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                                             root_counts=ROOT_COUNTS, degree=DEGREE,
                                             segment_count=SEGMENT_COUNT,
                                             surface_point_count=SURFACE_POINT_COUNT,
                                             fcm_subdivision_depth=FCM_SUBDIVISION_DEPTH,
                                             penalty=NITSCHE_PENALTY)
  domain = Domain((-outer_radius, -outer_radius), (2 * outer_radius, 2 * outer_radius), root_counts)
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(degree)))
  u = ScalarField(space; name=:u)

  outer_points, outer_segments = circle_points_segments(outer_radius, segment_count)
  inner_points, inner_segments = circle_points_segments(inner_radius, segment_count;
                                                        clockwise=true)
  offset = length(outer_points)
  boundary_points = vcat(outer_points, inner_points)
  boundary_segments = vcat(outer_segments,
                           [(first + offset, second + offset)
                            for (first, second) in inner_segments])
  boundary = EmbeddedSurface(SegmentMesh(boundary_points, boundary_segments);
                             point_count=surface_point_count)

  exact_solution = x -> log(hypot(x[1], x[2]) / outer_radius) / log(inner_radius / outer_radius)
  annulus_levelset =
    x -> max(hypot(x[1], x[2]) - outer_radius, inner_radius - hypot(x[1], x[2]))
  is_physical = x -> annulus_levelset(x) <= 0.0

  problem = AffineProblem(u)
  add_cell!(problem, Diffusion(u))
  add_surface!(problem, NitscheDirichlet(u, exact_solution, penalty))
  add_embedded_surface!(problem, boundary)

  verification_quadratures = Pair{Int,AbstractQuadrature{2,Float64}}[]

  for leaf in active_leaves(space)
    quadrature = finite_cell_quadrature(space, leaf, annulus_levelset;
                                        subdivision_depth=fcm_subdivision_depth)
    quadrature === nothing &&
      error("annulus setup expected every leaf to intersect the physical domain")
    add_cell_quadrature!(problem, leaf, quadrature)
    push!(verification_quadratures, leaf => quadrature)
  end

  return (; domain, space, u, boundary, exact_solution, annulus_levelset, is_physical, problem,
          verification_quadratures, inner_radius, outer_radius, root_counts, degree,
          segment_count, surface_point_count, fcm_subdivision_depth, penalty)
end

# Human-facing driver used both for direct execution and for benchmarked solves
# that want the exact same problem setup without VTK output.
function run_annular_plate_nitsche_example(; write_vtk=WRITE_VTK, print_summary=true, kwargs...)
  context = build_annular_plate_nitsche_context(; kwargs...)
  plan = compile(context.problem)
  state = State(plan, solve(assemble(plan)))
  error_value = relative_l2_error(state, context.u, context.exact_solution; plan=plan,
                                  cell_quadratures=context.verification_quadratures)
  vtk_path = nothing

  if write_vtk
    output_directory = joinpath(@__DIR__, "output")
    current_space = field_space(context.u)
    current_grid = grid(current_space)
    mkpath(output_directory)
    vtk_path = write_vtk(joinpath(output_directory, "annular_plate_nitsche"), state;
                         point_data=(physical=x -> context.is_physical(x) ? 1.0 : 0.0,
                                     abs_error=(x, values) -> context.is_physical(x) ?
                                                              abs(values.u -
                                                                  context.exact_solution(x)) :
                                                              0.0),
                         cell_data=(leaf=leaf -> Float64(leaf),
                                    level=leaf -> Float64.(level(current_grid, leaf)),
                                    degree=leaf -> Float64.(cell_degrees(current_space, leaf))),
                         field_data=(relative_l2_error=error_value,),
                         subdivisions=EXPORT_SUBDIVISIONS, export_degree=EXPORT_DEGREE,
                         append=true, compress=true, ascii=false)
    print_summary && println("  vtk  $vtk_path")
  end

  if print_summary
    println("annular_plate_nitsche.jl")
    @printf("  degree              : %d\n", context.degree)
    @printf("  active leaves       : %d\n", active_leaf_count(context.space))
    @printf("  scalar dofs         : %d\n", scalar_dof_count(context.space))
    @printf("  relative l2 error   : %.6e\n", error_value)
  end

  return (; context..., plan, state, error_value, vtk_path)
end

RUN_ANNULAR_PLATE_NITSCHE && run_annular_plate_nitsche_example()

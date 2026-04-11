using Printf
using Grico
import Grico: cell_matrix!, surface_matrix!, surface_rhs!

# This example solves the scalar Laplace problem on the annulus
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

# Problem geometry and discretization parameters.
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
  # segment quadrature points.
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

# Build a polygonal approximation of a circle as a closed segment mesh. The
# `clockwise` option is used to reverse the inner boundary orientation so the
# resulting two-component boundary has a consistent global orientation.
function circle_points_segments(radius, segment_count; clockwise=false)
  angles = range(0.0, 2 * pi; length=segment_count + 1)[1:(end-1)]
  points = [(radius * cos(angle), radius * sin(angle)) for angle in angles]
  clockwise && reverse!(points)
  segments = [(index, index == segment_count ? 1 : index + 1) for index in 1:segment_count]
  return points, segments
end

# The Cartesian background domain is the square that encloses the annulus.
domain = Domain((-OUTER_RADIUS, -OUTER_RADIUS), (2 * OUTER_RADIUS, 2 * OUTER_RADIUS), ROOT_COUNTS)
space = HpSpace(domain, SpaceOptions(degree=UniformDegree(DEGREE)))
u = ScalarField(space; name=:u)

# The embedded boundary is the union of the outer and inner circles. Both are
# represented explicitly as segment meshes and concatenated into one
# `EmbeddedSurface`.
outer_points, outer_segments = circle_points_segments(OUTER_RADIUS, SEGMENT_COUNT)
inner_points, inner_segments = circle_points_segments(INNER_RADIUS, SEGMENT_COUNT; clockwise=true)
offset = length(outer_points)
boundary_points = vcat(outer_points, inner_points)
boundary_segments = vcat(outer_segments,
                         [(first + offset, second + offset) for (first, second) in inner_segments])
boundary = EmbeddedSurface(SegmentMesh(boundary_points, boundary_segments);
                           point_count=SURFACE_POINT_COUNT,)

# Exact radial harmonic solution and a level-set description of the physical
# annulus. The level set is negative inside the annulus, zero on the circles,
# and positive outside.
exact_solution = x -> log(hypot(x[1], x[2]) / OUTER_RADIUS) / log(INNER_RADIUS / OUTER_RADIUS)
annulus_levelset = x -> max(hypot(x[1], x[2]) - OUTER_RADIUS, INNER_RADIUS - hypot(x[1], x[2]))
is_physical = x -> annulus_levelset(x) <= 0.0

# Assemble the unfitted problem:
# - Laplace operator in the physical domain,
# - symmetric Nitsche Dirichlet condition on the embedded boundary,
# - explicit embedded-surface geometry.
problem = AffineProblem(u)
add_cell!(problem, Diffusion(u))
add_surface!(problem, NitscheDirichlet(u, exact_solution, NITSCHE_PENALTY))
add_embedded_surface!(problem, boundary)

# Finite-cell quadratures are attached leaf by leaf. The same quadratures are
# also reused for the verification integral so the reported `L²` error is
# measured over the physical annulus rather than over the surrounding square.
verification_quadratures = Pair{Int,AbstractQuadrature{2,Float64}}[]

for leaf in active_leaves(space)
  quadrature = finite_cell_quadrature(space, leaf, annulus_levelset;
                                      subdivision_depth=FCM_SUBDIVISION_DEPTH,)
  quadrature === nothing &&
    error("annulus setup expected every leaf to intersect the physical domain")
  add_cell_quadrature!(problem, leaf, quadrature)
  push!(verification_quadratures, leaf => quadrature)
end

# Solve the linear system and compute the relative `L²` error against the exact
# radial solution.
plan = compile(problem)
state = State(plan, solve(assemble(plan)))
error_value = relative_l2_error(state, u, exact_solution; plan=plan,
                                cell_quadratures=verification_quadratures,)

output_directory = joinpath(@__DIR__, "output")
current_space = field_space(u)
current_grid = grid(current_space)

if WRITE_VTK
  mkpath(output_directory)

  # In the VTK output we export both the discrete solution and a few useful
  # diagnostics: a physical-domain mask on the enclosing square, the pointwise
  # absolute error inside the annulus, and per-cell level/degree metadata.
  vtk_path = write_vtk(joinpath(output_directory, "annular_plate_nitsche"), state;
                       point_data=(physical=x -> is_physical(x) ? 1.0 : 0.0,
                                   abs_error=(x, values) -> is_physical(x) ?
                                                            abs(values.u - exact_solution(x)) : 0.0),
                       cell_data=(leaf=leaf -> Float64(leaf),
                                  level=leaf -> Float64.(level(current_grid, leaf)),
                                  degree=leaf -> Float64.(cell_degrees(current_space, leaf))),
                       field_data=(relative_l2_error=error_value,),
                       subdivisions=EXPORT_SUBDIVISIONS, export_degree=EXPORT_DEGREE, append=true,
                       compress=true, ascii=false,)
  println("  vtk  $vtk_path")
end

println("annular_plate_nitsche.jl")
@printf("  degree              : %d\n", DEGREE)
@printf("  active leaves       : %d\n", active_leaf_count(space))
@printf("  scalar dofs         : %d\n", scalar_dof_count(space))
@printf("  relative l2 error   : %.6e\n", error_value)

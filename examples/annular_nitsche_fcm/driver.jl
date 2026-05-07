using Printf
using Grico
using WriteVTK

# This example demonstrates Grico's unfitted finite-cell path. The physical
# domain is the annulus
#
#   Ω = {x ∈ ℝ² : Rᵢ ≤ ‖x‖₂ ≤ Rₒ},
#
# embedded in a Cartesian background square. Volume integration on cut cells is
# supplied by `PhysicalDomain` and finite-cell quadrature, while the circular
# Dirichlet boundary is represented independently by a polygonal segment mesh.
#
# The exact solution
#
#   uₑ(r) = log(r / Rₒ) / log(Rᵢ / Rₒ)
#
# is harmonic in the annulus, equals one on the inner circle, and equals zero on
# the outer circle. Boundary values are imposed weakly with symmetric Nitsche
# terms written through the accumulator operator API: each callback returns the
# coefficients multiplying v and ∇v at one quadrature point, and Grico handles
# the local projection. This example therefore isolates three advanced features
# in one compact script: cut-cell volume quadrature, explicit embedded surfaces,
# and weak boundary enforcement on an unfitted mesh.

const ANNULUS_INNER_RADIUS = 0.35
const ANNULUS_OUTER_RADIUS = 1.0
const ANNULUS_ROOT_COUNTS = (2, 2)
const ANNULUS_DEGREE = 4
const ANNULUS_SEGMENTS = 128
const ANNULUS_SURFACE_POINTS = 3
const ANNULUS_FCM_DEPTH = 7
const ANNULUS_NITSCHE_PENALTY = 40.0

struct AnnularDiffusion end

struct AnnularNitscheBoundary{F,T}
  exact_solution::F
  penalty::T
end

# The volume term is ordinary diffusion on the physical portion of each finite
# cell. `PhysicalDomain` supplies the cut-cell quadrature, so the accumulator
# callback remains the same as for a fitted Poisson problem.
function Grico.cell_accumulate(::AnnularDiffusion, q, trial, test_component)
  return TestChannels(zero(trial.value), gradient(trial))
end

function Grico.surface_accumulate(operator::AnnularNitscheBoundary, q, trial, test_component)
  # Symmetric Nitsche enforcement of `u = g` adds
  #
  #   -∂ₙu v - u ∂ₙv + η u v
  #
  # to the left-hand side. In accumulator form, the first and penalty terms
  # multiply `v`, while `-u n` multiplies `∇v`.
  h = minimum(cell_size(q))
  η = operator.penalty / h
  normal_value = normal(q)
  trial_normal = normal_component(gradient(trial), normal_value)
  return TestChannels(-trial_normal + η * value(trial),
                      _scale_surface_normal(-value(trial), normal_value))
end

function Grico.surface_rhs_accumulate(operator::AnnularNitscheBoundary, q, test_component)
  # The matching right-hand side is `η g v - g ∂ₙv`. Using the same `η = C / h`
  # scaling as the bilinear term keeps the weak Dirichlet condition stable under
  # h-refinement and different cut-cell sizes.
  g = operator.exact_solution(point(q))
  h = minimum(cell_size(q))
  η = operator.penalty / h
  return TestChannels(η * g, _scale_surface_normal(-g, normal(q)))
end

function _scale_surface_normal(scale, normal_value::NTuple{D,T}) where {D,T}
  ntuple(axis -> scale * normal_value[axis], Val(D))
end

function annulus_circle_segments(radius, segment_count; clockwise=false)
  # `SegmentMesh` normals are determined by segment orientation. The outer
  # circle and inner circle must have opposite orientations so the two boundary
  # components represent the oriented boundary of an annulus.
  segment_count >= 8 || throw(ArgumentError("segment_count must be at least 8"))
  angles = range(0.0, 2π; length=segment_count + 1)[1:(end-1)]
  points = [(radius * cos(angle), radius * sin(angle)) for angle in angles]
  clockwise && reverse!(points)
  segments = [(index, index == segment_count ? 1 : index + 1) for index in 1:segment_count]
  return points, segments
end

function build_annular_nitsche_context(; inner_radius=ANNULUS_INNER_RADIUS,
                                       outer_radius=ANNULUS_OUTER_RADIUS,
                                       root_counts=ANNULUS_ROOT_COUNTS, degree=ANNULUS_DEGREE,
                                       segment_count=ANNULUS_SEGMENTS,
                                       surface_point_count=ANNULUS_SURFACE_POINTS,
                                       fcm_depth=ANNULUS_FCM_DEPTH, penalty=ANNULUS_NITSCHE_PENALTY)
  0 < inner_radius < outer_radius ||
    throw(ArgumentError("inner_radius and outer_radius must satisfy 0 < inner < outer"))
  # The background domain is a simple Cartesian square. The physical annulus is
  # introduced later by the implicit region, which is the finite-cell feature
  # being demonstrated.
  background = Domain((-outer_radius, -outer_radius), (2 * outer_radius, 2 * outer_radius),
                      root_counts)
  outer_points, outer_segments = annulus_circle_segments(outer_radius, segment_count)
  inner_points, inner_segments = annulus_circle_segments(inner_radius, segment_count;
                                                         clockwise=true)
  offset = length(outer_points)
  boundary_points = vcat(outer_points, inner_points)
  boundary_segments = vcat(outer_segments,
                           [(first + offset, second + offset) for (first, second) in inner_segments])
  boundary = Grico.EmbeddedSurface(Grico.SegmentMesh(boundary_points, boundary_segments);
                                   point_count=surface_point_count)

  # The exact harmonic solution is radial, so the source term vanishes in the
  # physical annulus. The level set is negative inside the annulus and positive
  # in the removed disk or outside the outer circle.
  exact_solution = x -> log(hypot(x[1], x[2]) / outer_radius) / log(inner_radius / outer_radius)
  levelset = x -> max(hypot(x[1], x[2]) - outer_radius, inner_radius - hypot(x[1], x[2]))
  is_physical = x -> levelset(x) <= 0.0
  region = Grico.ImplicitRegion(levelset; subdivision_depth=fcm_depth)
  domain = Grico.PhysicalDomain(background, region)
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(degree)))
  u = ScalarField(space; name=:u)

  # The surface operator is registered twice: once as the affine Nitsche
  # bilinear form and once as the boundary load. The explicit embedded surface
  # is attached after the operators so the problem remains readable as
  # "physics first, geometry attachment second".
  problem = AffineProblem(u; operator_class=SPD())
  nitsche = AnnularNitscheBoundary(exact_solution, penalty)
  add_cell_accumulator!(problem, u, u, AnnularDiffusion())
  add_surface_accumulator!(problem, u, u, nitsche)
  add_surface_accumulator!(problem, u, nitsche)
  Grico.add_embedded_surface!(problem, boundary)

  return (; domain, space, u, boundary, exact_solution, levelset, is_physical, problem,
          inner_radius, outer_radius, root_counts, degree, segment_count, surface_point_count,
          fcm_depth, penalty)
end

function run_annular_nitsche_fcm_example(; solver=AutoLinearSolver(), write_output=false,
                                         output_directory=joinpath(@__DIR__, "output"),
                                         export_subdivisions=1, export_degree=ANNULUS_DEGREE,
                                         print_summary=true, kwargs...)
  context = build_annular_nitsche_context(; kwargs...)
  state = solve(context.problem; solver)
  error_value = relative_l2_error(state, context.u, context.exact_solution)
  vtk_path = nothing

  if write_output
    # The exported `physical` point field makes the finite-cell embedding
    # visible in ParaView, while the error field is masked outside the physical
    # annulus so background-cell samples do not distract from the PDE solution.
    mkpath(output_directory)
    current_space = field_space(context.u)
    current_grid = grid(current_space)
    vtk_path = Grico.write_vtk(joinpath(output_directory, "annular_nitsche_fcm"), state;
                               point_data=(physical=x -> context.is_physical(x) ? 1.0 : 0.0,
                                           abs_error=(x, values) -> context.is_physical(x) ?
                                                                    abs(values.u -
                                                                        context.exact_solution(x)) :
                                                                    0.0),
                               cell_data=(leaf=leaf -> Float64(leaf),
                                          level=leaf -> Float64.(Grico.level(current_grid, leaf)),
                                          degree=leaf -> Float64.(Grico.cell_degrees(current_space,
                                                                                     leaf))),
                               field_data=(relative_l2_error=error_value,),
                               subdivisions=export_subdivisions, sample_degree=export_degree,
                               append=true, compress=true, ascii=false)
  end

  if print_summary
    println("annular_nitsche_fcm/driver.jl")
    @printf("  radii             : %.3f %.3f\n", context.inner_radius, context.outer_radius)
    @printf("  degree            : %d\n", context.degree)
    @printf("  active leaves     : %d\n", active_leaf_count(context.space))
    @printf("  scalar dofs       : %d\n", Grico.scalar_dof_count(context.space))
    @printf("  relative L² error : %.6e\n", error_value)
    vtk_path === nothing || println("  vtk               : $vtk_path")
  end

  return (; context..., state, error_value, vtk_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_annular_nitsche_fcm_example()
end

# Build the full unfitted problem description once so the solve step below uses
# one consistent quadrature, embedded geometry, and weak boundary data setup.
function build_annular_plate_nitsche_context(; inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                                             root_counts=ROOT_COUNTS, degree=DEGREE,
                                             segment_count=SEGMENT_COUNT,
                                             surface_point_count=SURFACE_POINT_COUNT,
                                             fcm_subdivision_depth=FCM_SUBDIVISION_DEPTH,
                                             penalty=NITSCHE_PENALTY)
  background = Domain((-outer_radius, -outer_radius), (2 * outer_radius, 2 * outer_radius),
                      root_counts)
  outer_points, outer_segments = circle_points_segments(outer_radius, segment_count)
  inner_points, inner_segments = circle_points_segments(inner_radius, segment_count; clockwise=true)
  offset = length(outer_points)
  boundary_points = vcat(outer_points, inner_points)
  boundary_segments = vcat(outer_segments,
                           [(first + offset, second + offset) for (first, second) in inner_segments])
  boundary = EmbeddedSurface(SegmentMesh(boundary_points, boundary_segments);
                             point_count=surface_point_count)

  exact_solution = x -> log(hypot(x[1], x[2]) / outer_radius) / log(inner_radius / outer_radius)
  annulus_levelset = x -> max(hypot(x[1], x[2]) - outer_radius, inner_radius - hypot(x[1], x[2]))
  is_physical = x -> annulus_levelset(x) <= 0.0
  region = ImplicitRegion(annulus_levelset; subdivision_depth=fcm_subdivision_depth)
  domain = PhysicalDomain(background, region)
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(degree)))
  u = ScalarField(space; name=:u)

  problem = AffineProblem(u)
  add_cell!(problem, Diffusion(u))
  add_surface!(problem, NitscheDirichlet(u, exact_solution, penalty))
  add_embedded_surface!(problem, boundary)

  return (; domain, space, u, boundary, exact_solution, annulus_levelset, is_physical, problem,
          inner_radius, outer_radius, root_counts, degree, segment_count, surface_point_count,
          fcm_subdivision_depth, penalty)
end

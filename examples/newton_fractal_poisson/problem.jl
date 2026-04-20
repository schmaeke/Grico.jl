# The model stores the parameters of the manufactured field. The three roots are
# those of `p(z) = z³ - 1`; `root_values` maps the converged root weights to
# scalar solution values, while the residual parameters add a smooth accent near
# slowly converging Newton points.
struct NewtonFractalModel{T<:AbstractFloat}
  roots::NTuple{3,Complex{T}}
  root_values::NTuple{3,T}
  iterations::Int
  plane_scale::T
  root_sharpness::T
  residual_weight::T
  residual_contrast::T
  derivative_regularization::T
  difference_step::T
end

# Construct the finite Newton model used by the example. The validation keeps
# all smoothing and finite-difference scales positive so the manufactured field
# and its numerical derivatives are well-defined at every quadrature point.
function build_newton_fractal_model(; iterations=NEWTON_ITERATIONS, plane_scale=NEWTON_PLANE_SCALE,
                                    root_sharpness=NEWTON_ROOT_SHARPNESS,
                                    residual_weight=NEWTON_RESIDUAL_WEIGHT,
                                    residual_contrast=NEWTON_RESIDUAL_CONTRAST,
                                    derivative_regularization=NEWTON_DERIVATIVE_REGULARIZATION,
                                    difference_step=NEWTON_DIFFERENCE_STEP)
  iterations >= 1 || throw(ArgumentError("iterations must be positive"))
  plane_scale > 0.0 || throw(ArgumentError("plane_scale must be positive"))
  root_sharpness > 0.0 || throw(ArgumentError("root_sharpness must be positive"))
  residual_weight >= 0.0 || throw(ArgumentError("residual_weight must be nonnegative"))
  residual_contrast > 0.0 || throw(ArgumentError("residual_contrast must be positive"))
  derivative_regularization > 0.0 ||
    throw(ArgumentError("derivative_regularization must be positive"))
  difference_step > 0.0 || throw(ArgumentError("difference_step must be positive"))

  third_root = sqrt(3.0) / 2.0
  roots = (complex(1.0, 0.0), complex(-0.5, third_root), complex(-0.5, -third_root))
  root_values = (-0.85, 0.15, 0.95)
  return NewtonFractalModel{Float64}(roots, root_values, Int(iterations), Float64(plane_scale),
                                     Float64(root_sharpness), Float64(residual_weight),
                                     Float64(residual_contrast), Float64(derivative_regularization),
                                     Float64(difference_step))
end

@inline _newton_polynomial(z) = z^3 - one(z)

# The usual Newton correction is regularized by `|p′(z)|² + ε`. This preserves
# the root basins away from critical points while preventing the finite iteration
# from producing unbounded values near `z = 0`.
@inline function _newton_step(z::Complex{T}, model::NewtonFractalModel{T}) where {T}
  derivative = 3 * z^2
  correction = _newton_polynomial(z) * conj(derivative) /
               (abs2(derivative) + model.derivative_regularization)
  return z - correction
end

# Map the physical square `Ω = [0, 1]²` to a centered square in `ℂ`. The scale
# is chosen in `driver.jl` so all roots of `p(z) = z³ - 1` lie in the sampled
# region.
function _newton_initial_point(model::NewtonFractalModel{T}, x) where {T}
  return complex(model.plane_scale * (T(x[1]) - T(0.5)), model.plane_scale * (T(x[2]) - T(0.5)))
end

# Apply a fixed number of Newton steps. Keeping the iteration count finite makes
# the manufactured field smooth away from thin transition regions instead of
# turning it into a discontinuous basin indicator.
function newton_iterate(model::NewtonFractalModel, x)
  z = _newton_initial_point(model, x)

  for _ in 1:model.iterations
    z = _newton_step(z, model)
  end

  return z
end

# Convert distances to the three roots into smooth weights `ωᵢ` with
# `Σᵢ ωᵢ = 1`. This gives a scalar field with Newton-fractal texture but avoids
# jumps across exact basin boundaries.
function newton_basin_weights(model::NewtonFractalModel{T}, x) where {T}
  z = newton_iterate(model, x)
  scores = ntuple(index -> -model.root_sharpness * abs2(z - model.roots[index]), 3)
  shift = maximum(scores)
  weights = ntuple(index -> exp(scores[index] - shift), 3)
  total = weights[1] + weights[2] + weights[3]
  return ntuple(index -> weights[index] / total, 3)
end

# Return the dominant Newton basin at a point. This diagnostic is written to VTK
# as cell and point data; it is not used in the weak form.
function newton_basin_index(model::NewtonFractalModel, x)
  weights = newton_basin_weights(model, x)
  best = 1

  for index in 2:3
    weights[index] > weights[best] && (best = index)
  end

  return best
end

# The residual indicator `log(1 + |p(z)|)` highlights points that have not
# converged cleanly after the finite Newton iteration. It is scaled separately
# from the root weights so the manufactured solution can retain visible internal
# structure inside a basin.
function newton_residual_indicator(model::NewtonFractalModel, x)
  z = newton_iterate(model, x)
  return log1p(abs(_newton_polynomial(z)))
end

# Combine smooth basin weights with the residual indicator to obtain the exact
# scalar solution used by the Poisson problem.
function newton_color_value(model::NewtonFractalModel{T}, x) where {T}
  weights = newton_basin_weights(model, x)
  basin_value = sum(weights[index] * model.root_values[index] for index in 1:3)
  residual = tanh(model.residual_contrast * newton_residual_indicator(model, x))
  return basin_value + model.residual_weight * residual
end

newton_solution_value(model::NewtonFractalModel{T}, x) where {T} = newton_color_value(model, x)

# This finite-difference source is useful for diagnostics and for comparing with
# the strong load `ℓ(v) = ∫Ω v f dΩ`. The production weak form uses `∇uₑ`
# directly, which avoids applying a discrete Laplacian to a deliberately rough
# field.
function newton_source_value(model::NewtonFractalModel{T}, x) where {T}
  h = model.difference_step
  x0 = T(x[1])
  y0 = T(x[2])
  center = newton_solution_value(model, (x0, y0))
  shifted_x_plus = newton_solution_value(model, (x0 + h, y0))
  shifted_x_minus = newton_solution_value(model, (x0 - h, y0))
  shifted_y_plus = newton_solution_value(model, (x0, y0 + h))
  shifted_y_minus = newton_solution_value(model, (x0, y0 - h))
  laplacian = (shifted_x_plus - 2 * center + shifted_x_minus + shifted_y_plus - 2 * center +
               shifted_y_minus) / h^2
  return -laplacian
end

# The weak manufactured load needs `∇uₑ` at quadrature points. A centered
# finite difference is sufficient here because the model itself is meant to
# include a controlled microscale cutoff.
function newton_solution_gradient(model::NewtonFractalModel{T}, x) where {T}
  h = model.difference_step
  x0 = T(x[1])
  y0 = T(x[2])
  dx = (newton_solution_value(model, (x0 + h, y0)) - newton_solution_value(model, (x0 - h, y0))) /
       (2 * h)
  dy = (newton_solution_value(model, (x0, y0 + h)) - newton_solution_value(model, (x0, y0 - h))) /
       (2 * h)
  return (dx, dy)
end

# Build the initial field, model callbacks, and adaptivity parameters used by
# the solve loop. The polynomial lower bound is explicit because this example
# may reduce the local degree `p` on small nonsmooth leaves while still keeping
# C⁰ traces valid.
function build_newton_fractal_poisson_context(; max_h_level=MAX_H_LEVEL,
                                              initial_degree=INITIAL_DEGREE, min_degree=MIN_DEGREE,
                                              max_degree=MAX_DEGREE,
                                              quadrature_extra_points=QUADRATURE_EXTRA_POINTS,
                                              adaptivity_tolerance=ADAPTIVITY_TOLERANCE,
                                              adaptivity_smoothness_threshold=ADAPTIVITY_SMOOTHNESS_THRESHOLD,
                                              newton_iterations=NEWTON_ITERATIONS,
                                              plane_scale=NEWTON_PLANE_SCALE,
                                              root_sharpness=NEWTON_ROOT_SHARPNESS,
                                              residual_weight=NEWTON_RESIDUAL_WEIGHT,
                                              residual_contrast=NEWTON_RESIDUAL_CONTRAST,
                                              derivative_regularization=NEWTON_DERIVATIVE_REGULARIZATION,
                                              difference_step=NEWTON_DIFFERENCE_STEP)
  min_degree <= initial_degree || throw(ArgumentError("min_degree must not exceed initial_degree"))
  initial_degree <= max_degree || throw(ArgumentError("initial_degree must not exceed max_degree"))

  model = build_newton_fractal_model(; iterations=newton_iterations, plane_scale, root_sharpness,
                                     residual_weight, residual_contrast, derivative_regularization,
                                     difference_step)
  domain = Domain(ntuple(_ -> 0.0, DIMENSION), ntuple(_ -> 1.0, DIMENSION),
                  ntuple(_ -> 1, DIMENSION))
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(initial_degree),
                               quadrature=DegreePlusQuadrature(quadrature_extra_points)))
  u = ScalarField(space; name=:u)
  exact_solution = x -> newton_solution_value(model, x)
  exact_gradient = x -> newton_solution_gradient(model, x)
  source_term = x -> newton_source_value(model, x)

  return (; dimension=DIMENSION, max_h_level, initial_degree, min_degree, max_degree,
          quadrature_extra_points, adaptivity_tolerance, adaptivity_smoothness_threshold, model,
          exact_solution, exact_gradient, source_term, space, u)
end

# Assemble the manufactured Poisson problem. The right-hand side is written as
# `ℓ(v) = ∫Ω ∇v · ∇uₑ dΩ`, and the exact nonzero trace `uₑ|∂Ω` is imposed on
# every physical boundary face through strong Dirichlet projection.
function build_newton_fractal_poisson_problem(u, context)
  problem = AffineProblem(u)
  add_cell!(problem, NewtonFractalPoissonDiffusion(u))
  add_cell!(problem, NewtonFractalPoissonGradientLoad(u, context.exact_gradient))

  for axis in 1:context.dimension
    add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, LOWER), context.exact_solution))
    add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, UPPER), context.exact_solution))
  end

  return problem
end

# The automatic planner uses the multiresolution detail indicators from the
# current state. `min_p` and `max_p` keep the example in a controlled mixed-`hp`
# regime, while `max_h_level` forces the dyadic depth `ℓ` under study.
function newton_fractal_adaptivity_plan(state, u, context; tolerance=context.adaptivity_tolerance,
                                        smoothness_threshold=context.adaptivity_smoothness_threshold,
                                        max_h_level=context.max_h_level)
  limits = AdaptivityLimits(field_space(u); min_p=context.min_degree, max_p=context.max_degree,
                            max_h_level=max_h_level)
  return adaptivity_plan(state, u; tolerance, smoothness_threshold, limits)
end

# Report the deepest dyadic level `ℓ` currently present in any active leaf. This
# is the stopping criterion used by the example driver.
function newton_fractal_max_h_level(space)
  grid_data = grid(space)
  return maximum((maximum(level(grid_data, leaf)) for leaf in active_leaves(space)); init=0)
end

# Count active leaves by their deepest coordinate direction level. The histogram
# is a compact terminal diagnostic for whether the refinement tree actually
# reaches the intended depth.
function newton_fractal_level_histogram(space)
  grid_data = grid(space)
  counts = Dict{Int,Int}()

  for leaf in active_leaves(space)
    leaf_level = maximum(level(grid_data, leaf))
    counts[leaf_level] = get(counts, leaf_level, 0) + 1
  end

  return sort!(collect(counts); by=first)
end

newton_fractal_stored_cell_count(space) = Grico.stored_cell_count(grid(space))

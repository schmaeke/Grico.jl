# ---------------------------------------------------------------------------
# 2. Exact solution and matching source term
# ---------------------------------------------------------------------------
#
# Because the exact solution is known, we can both prescribe exact boundary
# data and measure the true discretization error after each adaptive step.
#
# The prefactor
#
#   -α (α + d - 2)
#
# is the Laplacian coefficient for r^α in dimension d. Keeping the manufactured
# data in one helper ensures that boundary data, source terms, and error
# measurement all refer to the same mathematical solution.
function origin_solution_data(; dimension=DIMENSION, singular_exponent=SINGULAR_EXPONENT)
  source_factor = -singular_exponent * (singular_exponent + dimension - 2)
  exact_solution = x -> (r=sqrt(sum(abs2, x)); r == 0.0 ? 0.0 : r^singular_exponent)
  source_term = x -> (r=sqrt(sum(abs2, x));
                      r == 0.0 ? 0.0 : source_factor * r^(singular_exponent - 2))
  return (; dimension, singular_exponent, source_factor, exact_solution, source_term)
end

# Build the reusable field descriptor and manufactured data used by the
# adaptive solve below.
function build_origin_singularity_poisson_context(; dimension=DIMENSION,
                                                  initial_degree=INITIAL_DEGREE,
                                                  singular_exponent=SINGULAR_EXPONENT)
  manufactured = origin_solution_data(; dimension, singular_exponent)
  space = HpSpace(Domain(ntuple(_ -> 0.0, dimension), ntuple(_ -> 1.0, dimension),
                         ntuple(_ -> 1, dimension)),
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(initial_degree)))
  u = ScalarField(space; name=:u)
  return (; manufactured..., initial_degree, space, u)
end

function build_origin_singularity_problem(u, context)
  problem = AffineProblem(u; operator_class=SPD())
  add_cell_bilinear!(problem, u, u) do q, v, w
    inner(grad(v), grad(w))
  end
  add_cell_linear!(problem, u) do q, v
    value(v) * context.source_term(point(q))
  end

  for axis in 1:context.dimension
    add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, UPPER), context.exact_solution))
  end

  return problem
end

function origin_adaptivity_plan(state, u)
  limits = AdaptivityLimits(field_space(u); max_p=MAX_DEGREE, max_h_level=MAX_H_LEVEL)
  return Grico.adaptivity_plan(state, u; tolerance=ADAPTIVITY_TOLERANCE, limits=limits)
end

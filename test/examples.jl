using Test
using Grico

@testset "Matrix-Free Poisson Example" begin
  domain = Domain((0.0,), (1.0,), (2,))
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(2)))
  u = ScalarField(space; name=:u)

  problem = AffineProblem(u; operator_class=SPD())
  add_cell_bilinear!(problem, u, u) do q, v, w
    grad(v)[1] * grad(w)[1]
  end
  add_cell_linear!(problem, u) do q, v
    2.0 * value(v)
  end
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, LOWER), 1.0))
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, UPPER), 1.0))

  plan = compile(problem)
  state = solve(plan; solver=CGSolver(preconditioner=JacobiPreconditioner()))

  @test l2_error(state, u, x -> x[1] * (1.0 - x[1]) + 1.0; plan) <= 1.0e-10
end

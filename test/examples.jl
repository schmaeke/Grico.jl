using Test
using Grico

struct _ExampleDiffusion{F}
  field::F
end

struct _ExampleSource{F}
  field::F
end

function Grico.cell_apply!(local_result, operator::_ExampleDiffusion, values::CellValues,
                           local_coefficients)
  mode_count = local_mode_count(values, operator.field)

  for point_index in 1:point_count(values)
    gradient_value = gradient(values, local_coefficients, operator.field, point_index)
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      row = local_dof_index(values, operator.field, 1, mode_index)
      test_gradient = shape_gradient(values, operator.field, point_index, mode_index)
      local_result[row] += test_gradient[1] * gradient_value[1] * weighted
    end
  end

  return nothing
end

function Grico.cell_diagonal!(local_diagonal, operator::_ExampleDiffusion, values::CellValues)
  mode_count = local_mode_count(values, operator.field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      row = local_dof_index(values, operator.field, 1, mode_index)
      gradient_value = shape_gradient(values, operator.field, point_index, mode_index)
      local_diagonal[row] += gradient_value[1] * gradient_value[1] * weighted
    end
  end

  return nothing
end

function Grico.cell_rhs!(local_rhs, operator::_ExampleSource, values::CellValues)
  block_data = block(local_rhs, values, operator.field)

  for point_index in 1:point_count(values)
    weighted = 2.0 * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, operator.field)
      block_data[mode_index] += shape_value(values, operator.field, point_index, mode_index) *
                                weighted
    end
  end

  return nothing
end

@testset "Matrix-Free Poisson Example" begin
  domain = Domain((0.0,), (1.0,), (2,))
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(2)))
  u = ScalarField(space; name=:u)

  problem = AffineProblem(u)
  add_cell!(problem, _ExampleDiffusion(u))
  add_cell!(problem, _ExampleSource(u))
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, LOWER), 1.0))
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, UPPER), 1.0))

  plan = compile(problem)
  state = solve(plan; preconditioner=JacobiPreconditioner())

  @test l2_error(state, u, x -> x[1] * (1.0 - x[1]) + 1.0; plan) <= 1.0e-10
end

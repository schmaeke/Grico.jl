using Test
using Grico

const VERIFICATION_TOL = 1.0e-10

struct _VerificationDiffusion{F,T}
  field::F
  coefficient::T
end

function Grico.cell_apply!(local_result, operator::_VerificationDiffusion, values::Grico.CellValues,
                           local_coefficients)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    weighted = Grico.weight(values, point_index)
    gradient_value = Grico.gradient(values, local_coefficients, operator.field, point_index)

    for row_mode in 1:mode_count
      gradient_row = Grico.shape_gradient(values, operator.field, point_index, row_mode)
      local_result[Grico.local_dof_index(values, operator.field, 1, row_mode)] += operator.coefficient *
                                                                                  sum(gradient_row[axis] *
                                                                                      gradient_value[axis]
                                                                                      for axis in
                                                                                          eachindex(gradient_row)) *
                                                                                  weighted
    end
  end

  return nothing
end

function Grico.cell_diagonal!(local_diagonal, operator::_VerificationDiffusion,
                              values::Grico.CellValues)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    weighted = Grico.weight(values, point_index)

    for mode_index in 1:mode_count
      gradient_value = Grico.shape_gradient(values, operator.field, point_index, mode_index)
      local_diagonal[Grico.local_dof_index(values, operator.field, 1, mode_index)] += operator.coefficient *
                                                                                      sum(value *
                                                                                          value
                                                                                          for value in
                                                                                              gradient_value) *
                                                                                      weighted
    end
  end

  return nothing
end

struct _VerificationSource{F,G}
  field::F
  data::G
end

struct _VerificationExactCallable end

(::_VerificationExactCallable)(x) = x[1] * (1.0 - x[1]) + 1.0

function Grico.cell_rhs!(local_rhs, operator::_VerificationSource, values::Grico.CellValues)
  local_block = Grico.block(local_rhs, values, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    weighted = operator.data(Grico.point(values, point_index)) * Grico.weight(values, point_index)

    for mode_index in 1:mode_count
      local_block[mode_index] += Grico.shape_value(values, operator.field, point_index,
                                                   mode_index) * weighted
    end
  end

  return nothing
end

function _verification_left_half_reference_quadrature()
  midpoint = -0.5
  half_length = 0.5
  offset = half_length / sqrt(3.0)
  return Grico.PointQuadrature([(midpoint - offset,), (midpoint + offset,)],
                               [half_length, half_length])
end

@testset "Scalar Relative L2 Error" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, _VerificationDiffusion(u, 1.0))
  Grico.add_cell!(problem, _VerificationSource(u, x -> 2.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 1.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 1.0))

  plan = Grico.compile(problem)
  state = Grico.solve(plan; solver=Grico.CGSolver(preconditioner=Grico.JacobiPreconditioner()))
  exact(x) = x[1] * (1.0 - x[1]) + 1.0

  @test Grico.l2_error(state, u, exact; plan=plan) <= VERIFICATION_TOL
  @test Grico.relative_l2_error(state, u, exact; plan=plan) <= VERIFICATION_TOL
  @test Grico.l2_error(state, u, exact) <= VERIFICATION_TOL
  @test Grico.relative_l2_error(state, u, exact) <= VERIFICATION_TOL
  @test Grico.l2_error(state, u, _VerificationExactCallable(); plan=plan) <= VERIFICATION_TOL
  @test Grico.relative_l2_error(state, u, _VerificationExactCallable(); plan=plan) <=
        VERIFICATION_TOL
end

@testset "Vector Relative L2 Error" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  v = Grico.VectorField(space, 2; name=:v)
  state = Grico.State(Grico.FieldLayout((v,)))

  @test Grico.l2_error(state, v, (1.0, 2.0)) ≈ sqrt(5.0) atol = VERIFICATION_TOL
  @test Grico.relative_l2_error(state, v, (1.0, 2.0)) ≈ 1.0 atol = VERIFICATION_TOL
end

@testset "Verification Quadrature Override" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  state = Grico.State(Grico.FieldLayout((u,)))
  quadrature = _verification_left_half_reference_quadrature()

  @test Grico.l2_error(state, u, 1.0; cell_quadratures=(1 => quadrature,)) ≈ sqrt(0.5) atol = VERIFICATION_TOL
  @test Grico.l2_error(state, u, 1.0; cell_quadratures=(Int32(1) => quadrature,)) ≈ sqrt(0.5) atol = VERIFICATION_TOL
  float32_quadrature = Grico.PointQuadrature([(Float32(0.0),)], [Float32(2.0)])
  @test Grico.l2_error(state, u, 1.0; cell_quadratures=(1 => float32_quadrature,)) ≈ 1.0 atol = VERIFICATION_TOL
  @test Grico.relative_l2_error(state, u, 1.0; cell_quadratures=(1 => quadrature,)) ≈ 1.0 atol = VERIFICATION_TOL
  @test Grico.l2_error(state, u, [1.0]) ≈ 1.0 atol = VERIFICATION_TOL
  @test_throws ArgumentError Grico.l2_error(state, u, (1.0, 2.0))
  @test_throws ArgumentError Grico.l2_error(state, u, [1.0, 2.0])
  @test Grico.relative_l2_error(state, u, 0.0) == 0.0
  Grico.coefficients(state) .= 1.0
  @test Grico.relative_l2_error(state, u, 0.0) == Inf
  @test_throws ArgumentError Grico.l2_error(state, u, 1.0;
                                            cell_quadratures=(1 => quadrature, 1 => quadrature))
  @test_throws ArgumentError Grico.l2_error(state, u, 1.0; cell_quadratures=(1.5 => quadrature,))
  wrong_dimension = Grico.PointQuadrature([(0.0, 0.0)], [1.0])
  @test_throws ArgumentError Grico.l2_error(state, u, 1.0; cell_quadratures=(1 => wrong_dimension,))
end

@testset "Physical Verification Quadrature Enrichment" begin
  background = Grico.Domain((0.0,), (1.0,), (1,))
  domain = Grico.PhysicalDomain(background,
                                Grico.ImplicitRegion(x -> x[1] - 0.5; subdivision_depth=1))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  state = Grico.State(Grico.FieldLayout((u,)))
  expected = sqrt(0.5)

  @test Grico.l2_error(state, u, 1.0) ≈ expected atol = VERIFICATION_TOL
  @test Grico.l2_error(state, u, 1.0; extra_points=1) ≈ expected atol = VERIFICATION_TOL

  extended_domain = Grico.PhysicalDomain(background,
                                         Grico.ImplicitRegion(x -> x[1] - 0.5; subdivision_depth=1);
                                         cell_measure=Grico.FiniteCellExtension(0.2))
  extended_space = Grico.HpSpace(extended_domain,
                                 Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                    degree=Grico.UniformDegree(1)))
  extended_field = Grico.ScalarField(extended_space; name=:u)
  problem = Grico.AffineProblem(extended_field)
  plan = Grico.compile(problem)
  extended_state = Grico.State(plan)

  @test sum(plan.integration.cells[1].weights) ≈ 0.6 atol = VERIFICATION_TOL
  @test Grico.l2_error(extended_state, extended_field, 1.0; plan) ≈ expected atol = VERIFICATION_TOL
  @test Grico.l2_error(extended_state, extended_field, 1.0; extra_points=1) ≈ expected atol = VERIFICATION_TOL
end

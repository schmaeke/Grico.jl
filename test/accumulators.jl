using Test
using Grico
import Grico: OperatorWorkspace, ResidualWorkspace

struct _AccumulatorDiffusionReaction{T}
  diffusion::T
  reaction::T
end

struct _AccumulatorVectorAffine{T}
  diagonal_mass::T
  offdiagonal_mass::T
  diagonal_diffusion::T
  offdiagonal_diffusion::T
end

struct _AccumulatorBratu end

struct _AccumulatorVectorSemilinear end

struct _AccumulatorCellLoad{T}
  scale::T
end

struct _AccumulatorOneSidedAffine{T}
  mass::T
  test_normal::T
  trial_normal::T
  diffusion::T
end

struct _AccumulatorOneSidedLoad{T}
  value::T
  normal::T
end

struct _AccumulatorInterfaceAffine{T}
  penalty::T
  test_normal::T
  trial_normal::T
end

struct _AccumulatorInterfaceLoad{T}
  jump_value::T
  average_normal::T
end

struct _AccumulatorOneSidedQuadratic end

struct _AccumulatorInterfaceQuadratic end

_accumulator_allocation_limit(serial_limit::Integer) = max(serial_limit, 1_000 * Threads.nthreads())

@inline function _scale_tuple(scale, tuple::NTuple{D,T}) where {D,T}
  return ntuple(axis -> scale * tuple[axis], Val(D))
end

@inline _zero_tuple(tuple::NTuple{D,T}) where {D,T} = ntuple(_ -> zero(T), Val(D))

function Grico.cell_accumulate(operator::_AccumulatorDiffusionReaction, q, trial, test_component)
  return TestChannels(operator.reaction * trial.value,
                      _scale_tuple(operator.diffusion, trial.gradient))
end

function Grico.cell_accumulate(operator::_AccumulatorVectorAffine, q, trial, test_component)
  same_component = test_component == trial.component
  mass = same_component ? operator.diagonal_mass : operator.offdiagonal_mass
  diffusion = same_component ? operator.diagonal_diffusion : operator.offdiagonal_diffusion
  return TestChannels(mass * trial.value, _scale_tuple(diffusion, trial.gradient))
end

function Grico.cell_rhs_accumulate(operator::_AccumulatorCellLoad, q, test_component)
  return operator.scale * (1 + point(q)[1] + point(q)[2])
end

function _one_sided_affine_channels(operator::_AccumulatorOneSidedAffine, q, trial)
  normal_value = normal(q)
  trial_normal = normal_component(gradient(trial), normal_value)
  value_coefficient = operator.mass * value(trial) + operator.trial_normal * trial_normal
  gradient_coefficient = _scale_tuple(operator.test_normal * value(trial), normal_value) .+
                         _scale_tuple(operator.diffusion, gradient(trial))
  return TestChannels(value_coefficient, gradient_coefficient)
end

function Grico.boundary_accumulate(operator::_AccumulatorOneSidedAffine, q, trial, test_component)
  return _one_sided_affine_channels(operator, q, trial)
end

function Grico.surface_accumulate(operator::_AccumulatorOneSidedAffine, q, trial, test_component)
  return _one_sided_affine_channels(operator, q, trial)
end

function _one_sided_load_channels(operator::_AccumulatorOneSidedLoad, q)
  return TestChannels(operator.value, _scale_tuple(operator.normal, normal(q)))
end

function Grico.boundary_rhs_accumulate(operator::_AccumulatorOneSidedLoad, q, test_component)
  return _one_sided_load_channels(operator, q)
end

function Grico.surface_rhs_accumulate(operator::_AccumulatorOneSidedLoad, q, test_component)
  return _one_sided_load_channels(operator, q)
end

function Grico.interface_accumulate(operator::_AccumulatorInterfaceAffine, q, trial, test_component)
  normal_value = normal(q)
  jump_value = jump(value(trial))
  average_normal = average(normal_component(gradient(trial), normal_value))
  test_normal = _scale_tuple(0.5 * operator.test_normal * jump_value, normal_value)
  value_coefficient = operator.penalty * jump_value + operator.trial_normal * average_normal
  return TraceTestChannels(TestChannels(-value_coefficient, test_normal),
                           TestChannels(value_coefficient, test_normal))
end

function Grico.interface_rhs_accumulate(operator::_AccumulatorInterfaceLoad, q, test_component)
  normal_value = normal(q)
  gradient_coefficient = _scale_tuple(0.5 * operator.average_normal, normal_value)
  return TraceTestChannels(TestChannels(-operator.jump_value, gradient_coefficient),
                           TestChannels(operator.jump_value, gradient_coefficient))
end

function Grico.cell_residual_accumulate(::_AccumulatorBratu, q, state, test_component)
  return TestChannels(exp(state.value), state.gradient)
end

function Grico.cell_tangent_accumulate(::_AccumulatorBratu, q, state, increment, test_component)
  return TestChannels(exp(state.value) * increment.value, increment.gradient)
end

function Grico.cell_residual_accumulate(::_AccumulatorVectorSemilinear, q, state, test_component)
  state_value = value(state)
  norm2 = inner(state_value, state_value)
  return TestChannels((norm2 + 0.5) * value(state, test_component), gradient(state, test_component))
end

function Grico.cell_tangent_accumulate(::_AccumulatorVectorSemilinear, q, state, increment,
                                       test_component)
  state_value = value(state)
  norm2 = inner(state_value, state_value)
  increment_component = component(increment)
  reaction = 2 * value(state, test_component) * value(state, increment_component) +
             (test_component == increment_component ? norm2 + 0.5 : 0.0)
  increment_gradient = gradient(increment)
  gradient_coefficient = test_component == increment_component ? increment_gradient :
                         _zero_tuple(increment_gradient)
  return TestChannels(reaction * value(increment), gradient_coefficient)
end

_one_sided_quadratic_residual(q, state) = TestChannels(value(state)^2, gradient(state))

function _one_sided_quadratic_tangent(q, state, increment)
  return TestChannels(2 * value(state) * value(increment), gradient(increment))
end

function Grico.boundary_residual_accumulate(::_AccumulatorOneSidedQuadratic, q, state,
                                            test_component)
  return _one_sided_quadratic_residual(q, state)
end

function Grico.surface_residual_accumulate(::_AccumulatorOneSidedQuadratic, q, state,
                                           test_component)
  return _one_sided_quadratic_residual(q, state)
end

function Grico.boundary_tangent_accumulate(::_AccumulatorOneSidedQuadratic, q, state, increment,
                                           test_component)
  return _one_sided_quadratic_tangent(q, state, increment)
end

function Grico.surface_tangent_accumulate(::_AccumulatorOneSidedQuadratic, q, state, increment,
                                          test_component)
  return _one_sided_quadratic_tangent(q, state, increment)
end

function Grico.interface_residual_accumulate(::_AccumulatorInterfaceQuadratic, q, state,
                                             test_component)
  jump_value = jump(value(state))
  return TraceTestChannels(TestChannels(-jump_value^2, _zero_tuple(normal(q))),
                           TestChannels(jump_value^2, _zero_tuple(normal(q))))
end

function Grico.interface_tangent_accumulate(::_AccumulatorInterfaceQuadratic, q, state, increment,
                                            test_component)
  coefficient = 2 * jump(value(state)) * jump(value(increment))
  return TraceTestChannels(TestChannels(-coefficient, _zero_tuple(normal(q))),
                           TestChannels(coefficient, _zero_tuple(normal(q))))
end

function _reduced_matrix(plan)
  return Grico._assemble_reduced_operator_matrix(plan,
                                                 Grico._ReducedOperatorWorkspace(plan).scratch)
end

function _finite_difference_tangent(plan, coefficients, increment)
  state = State(plan, coefficients)
  residual_base = residual(plan, state)
  step = 1.0e-7
  perturbed = State(plan, coefficients .+ step .* increment)
  return (residual(plan, perturbed) .- residual_base) ./ step
end

@testset "Accumulator Operators" begin
  domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  tensor_space = HpSpace(domain,
                         SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2),
                                      continuity=:dg))
  scalar = ScalarField(tensor_space; name=:u)
  coefficients = [sin(0.23 * index) for index in 1:field_dof_count(scalar)]

  accumulator_problem = AffineProblem(scalar; operator_class=SPD())
  add_cell_accumulator!(accumulator_problem, scalar, scalar,
                        _AccumulatorDiffusionReaction(1.75, 0.25))
  accumulator_plan = compile(accumulator_problem)

  accumulator_matrix = _reduced_matrix(accumulator_plan)
  @test apply(accumulator_plan, coefficients) ≈ accumulator_matrix * coefficients
  @test accumulator_matrix ≈ accumulator_matrix'

  workspace = OperatorWorkspace(accumulator_plan)
  result = zeros(field_dof_count(scalar))
  @test @inferred(apply!(result, accumulator_plan, coefficients, workspace)) === result
  apply_allocations = @allocated apply!(result, accumulator_plan, coefficients, workspace)
  @test apply_allocations <= _accumulator_allocation_limit(1_000)

  vector_field = VectorField(tensor_space, 2; name=:velocity)
  vector_coefficients = [cos(0.17 * index) for index in 1:field_dof_count(vector_field)]
  vector_problem = AffineProblem(vector_field; operator_class=NonsymmetricOperator())
  add_cell_accumulator!(vector_problem, vector_field, vector_field,
                        _AccumulatorVectorAffine(2.0, -0.2, 1.1, 0.35))
  vector_plan = compile(vector_problem)
  @test apply(vector_plan, vector_coefficients) ≈ _reduced_matrix(vector_plan) * vector_coefficients

  cell_load_problem = AffineProblem(scalar)
  add_cell_accumulator!(cell_load_problem, scalar, _AccumulatorCellLoad(0.35))
  cell_load_plan = compile(cell_load_problem)
  @test all(isfinite, rhs(cell_load_plan))

  boundary_operator = _AccumulatorOneSidedAffine(1.1, 0.2, 0.3, 0.4)
  boundary_load = _AccumulatorOneSidedLoad(0.7, -0.25)
  boundary_problem = AffineProblem(scalar; operator_class=NonsymmetricOperator())
  add_boundary_accumulator!(boundary_problem, BoundaryFace(1, UPPER), scalar, scalar,
                            boundary_operator)
  add_boundary_accumulator!(boundary_problem, BoundaryFace(1, UPPER), scalar, boundary_load)
  boundary_plan = compile(boundary_problem)
  @test apply(boundary_plan, coefficients) ≈ _reduced_matrix(boundary_plan) * coefficients
  @test all(isfinite, rhs(boundary_plan))

  surface_operator = _AccumulatorOneSidedAffine(0.9, -0.15, 0.25, 0.2)
  surface_load = _AccumulatorOneSidedLoad(-0.4, 0.35)
  surface_quadrature = Grico.PointQuadrature([(0.0, 0.0)], [0.5])
  surface_normals = [(1.0, 0.0)]
  surface_problem = AffineProblem(scalar; operator_class=NonsymmetricOperator())
  Grico.add_surface_quadrature!(surface_problem,
                                Grico.SurfaceQuadrature(1, surface_quadrature, surface_normals))
  add_surface_accumulator!(surface_problem, scalar, scalar, surface_operator)
  add_surface_accumulator!(surface_problem, scalar, surface_load)
  surface_plan = compile(surface_problem)
  @test apply(surface_plan, coefficients) ≈ _reduced_matrix(surface_plan) * coefficients
  @test all(isfinite, rhs(surface_plan))

  interface_domain = Domain((0.0, 0.0), (1.0, 1.0), (2, 1))
  interface_space = HpSpace(interface_domain,
                            SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2),
                                         continuity=:dg))
  interface_field = ScalarField(interface_space; name=:trace)
  interface_coefficients = [sin(0.21 * index) for index in 1:field_dof_count(interface_field)]
  interface_operator = _AccumulatorInterfaceAffine(1.4, -0.3, 0.2)
  interface_load = _AccumulatorInterfaceLoad(0.6, -0.45)
  interface_problem = AffineProblem(interface_field; operator_class=NonsymmetricOperator())
  add_interface_accumulator!(interface_problem, interface_field, interface_field,
                             interface_operator)
  add_interface_accumulator!(interface_problem, interface_field, interface_load)
  interface_plan = compile(interface_problem)
  @test apply(interface_plan, interface_coefficients) ≈
        _reduced_matrix(interface_plan) * interface_coefficients
  @test all(isfinite, rhs(interface_plan))

  nonlinear_trace_domain = Domain((0.0,), (1.0,), (2,))
  nonlinear_trace_space = HpSpace(nonlinear_trace_domain,
                                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2),
                                               continuity=:dg))
  nonlinear_trace_field = ScalarField(nonlinear_trace_space; name=:u)
  nonlinear_trace_problem = ResidualProblem(nonlinear_trace_field)
  add_boundary_accumulator!(nonlinear_trace_problem, BoundaryFace(1, UPPER), nonlinear_trace_field,
                            nonlinear_trace_field, _AccumulatorOneSidedQuadratic())
  Grico.add_surface_quadrature!(nonlinear_trace_problem,
                                Grico.SurfaceQuadrature(1, Grico.PointQuadrature([(0.0,)], [0.25]),
                                                        [(1.0,)]))
  add_surface_accumulator!(nonlinear_trace_problem, nonlinear_trace_field, nonlinear_trace_field,
                           _AccumulatorOneSidedQuadratic())
  add_interface_accumulator!(nonlinear_trace_problem, nonlinear_trace_field, nonlinear_trace_field,
                             _AccumulatorInterfaceQuadratic())
  nonlinear_trace_plan = compile(nonlinear_trace_problem)
  nonlinear_trace_coefficients = [0.07 * sin(0.31 * index)
                                  for index in 1:field_dof_count(nonlinear_trace_field)]
  nonlinear_trace_increment = [cos(0.27 * index)
                               for index in 1:field_dof_count(nonlinear_trace_field)]
  nonlinear_trace_state = State(nonlinear_trace_plan, nonlinear_trace_coefficients)
  nonlinear_trace_tangent = tangent_apply(nonlinear_trace_plan, nonlinear_trace_state,
                                          nonlinear_trace_increment)
  @test nonlinear_trace_tangent ≈
        _finite_difference_tangent(nonlinear_trace_plan, nonlinear_trace_coefficients,
                                   nonlinear_trace_increment) rtol = 2.0e-6 atol = 2.0e-6

  bratu_problem = ResidualProblem(scalar; operator_class=SPD())
  add_cell_accumulator!(bratu_problem, scalar, scalar, _AccumulatorBratu())
  bratu_plan = compile(bratu_problem)
  bratu_coefficients = [0.1 * sin(0.19 * index) for index in 1:field_dof_count(scalar)]
  bratu_increment = [cos(0.29 * index) for index in 1:field_dof_count(scalar)]
  bratu_state = State(bratu_plan, bratu_coefficients)
  bratu_tangent = tangent_apply(bratu_plan, bratu_state, bratu_increment)
  @test bratu_tangent ≈ _finite_difference_tangent(bratu_plan, bratu_coefficients, bratu_increment) rtol = 1.0e-6 atol = 1.0e-6

  residual_workspace = ResidualWorkspace(bratu_plan)
  residual_result = zeros(field_dof_count(scalar))
  tangent_result = zeros(field_dof_count(scalar))
  @test @inferred(residual!(residual_result, bratu_plan, bratu_state, residual_workspace)) ===
        residual_result
  @test @inferred(tangent_apply!(tangent_result, bratu_plan, bratu_state, bratu_increment,
                                 residual_workspace)) === tangent_result
  residual_allocations = @allocated residual!(residual_result, bratu_plan, bratu_state,
                                              residual_workspace)
  tangent_allocations = @allocated tangent_apply!(tangent_result, bratu_plan, bratu_state,
                                                  bratu_increment, residual_workspace)
  @test residual_allocations <= _accumulator_allocation_limit(1_000)
  @test tangent_allocations <= _accumulator_allocation_limit(1_000)

  vector_residual_problem = ResidualProblem(vector_field)
  add_cell_accumulator!(vector_residual_problem, vector_field, vector_field,
                        _AccumulatorVectorSemilinear())
  vector_residual_plan = compile(vector_residual_problem)
  vector_state_coefficients = [0.08 * sin(0.11 * index)
                               for index in 1:field_dof_count(vector_field)]
  vector_increment = [cos(0.13 * index) for index in 1:field_dof_count(vector_field)]
  vector_state = State(vector_residual_plan, vector_state_coefficients)
  vector_tangent = tangent_apply(vector_residual_plan, vector_state, vector_increment)
  @test vector_tangent ≈ _finite_difference_tangent(vector_residual_plan, vector_state_coefficients,
                                                    vector_increment) rtol = 2.0e-6 atol = 2.0e-6
end

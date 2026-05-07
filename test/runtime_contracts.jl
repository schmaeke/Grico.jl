using Test
using Grico
import Grico: OperatorWorkspace, ResidualWorkspace, add_cell!, add_cell_quadrature!, block,
              field_gradient, local_dof_index, local_mode_count, normal_gradient, point_count,
              shape_value, target_space, tensor_axis_values, tensor_values

struct _RuntimeBadWeightQuadrature <: Grico.AbstractQuadrature{1,Float64} end

Grico.point_count(::_RuntimeBadWeightQuadrature) = 1
Grico.point(::_RuntimeBadWeightQuadrature, ::Integer) = (0.0,)
Grico.weight(::_RuntimeBadWeightQuadrature, ::Integer) = NaN

struct _RuntimeMassOperator{F}
  field::F
end

struct _RuntimeScaledIdentity{T}
  scale::T
end

function Grico.cell_rhs!(local_rhs, operator::_RuntimeMassOperator, values)
  field = operator.field

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_rhs[row] += shape_value(values, field, point_index, mode_index) * weighted
    end
  end

  return nothing
end

function Grico.cell_apply!(local_result, operator::_RuntimeMassOperator, values, local_coefficients)
  field = operator.field

  for point_index in 1:point_count(values)
    field_value = value(values, local_coefficients, field, point_index)
    weighted = weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_result[row] += shape_value(values, field, point_index, mode_index) *
                           field_value *
                           weighted
    end
  end

  return nothing
end

function Grico.cell_apply!(local_result, operator::_RuntimeScaledIdentity, values,
                           local_coefficients)
  for index in eachindex(local_coefficients)
    local_result[index] += operator.scale * local_coefficients[index]
  end

  return nothing
end

function Grico.cell_diagonal!(local_diagonal, operator::_RuntimeScaledIdentity, values)
  for index in eachindex(local_diagonal)
    local_diagonal[index] += operator.scale
  end

  return nothing
end

function Grico.cell_matrix!(local_matrix, operator::_RuntimeScaledIdentity, values)
  for index in axes(local_matrix, 1)
    local_matrix[index, index] += operator.scale
  end

  return nothing
end

struct _RuntimeQuadraticOperator{F,T}
  field::F
  target::T
end

function Grico.cell_residual!(local_residual, operator::_RuntimeQuadraticOperator, values, state)
  field = operator.field

  for point_index in 1:point_count(values)
    field_value = value(values, state, field, point_index)
    weighted = (field_value * field_value - operator.target) * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_residual[row] += shape_value(values, field, point_index, mode_index) * weighted
    end
  end

  return nothing
end

function Grico.cell_tangent_apply!(local_result, operator::_RuntimeQuadraticOperator, values, state,
                                   local_increment)
  field = operator.field

  for point_index in 1:point_count(values)
    field_value = value(values, state, field, point_index)
    increment_value = value(values, local_increment, field, point_index)
    weighted = 2 * field_value * increment_value * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_result[row] += shape_value(values, field, point_index, mode_index) * weighted
    end
  end

  return nothing
end

function _runtime_allocated(call)
  call()
  GC.gc()
  return @allocated call()
end

@testset "Runtime Contracts" begin
  @testset "Shared Memory Runtime Boundary" begin
    @test Grico._runtime_worker_count(Grico._SHARED_MEMORY_CPU_BACKEND) == Base.Threads.nthreads()
    @test Grico._runtime_uses_polyester(Grico._SharedMemoryCPUBackend) isa Bool
  end

  @testset "Affine Matrix-Free Workspace" begin
    domain = Domain((0.0, 0.0), (1.0, 1.0), (2, 2))
    space = HpSpace(domain,
                    SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2), continuity=:dg))
    u = ScalarField(space; name=:u)
    problem = AffineProblem(u)
    add_cell!(problem, _RuntimeMassOperator(u))
    plan = compile(problem)
    workspace = OperatorWorkspace(plan)
    coefficients_data = ones(field_dof_count(u))
    result = zeros(field_dof_count(u))
    cell = plan.integration.cells[1]

    @test @inferred(value(cell, coefficients_data, u, 1)) isa Float64
    @test @inferred(gradient(cell, coefficients_data, u, 1)) isa NTuple{2,Float64}
    @test @inferred(rhs!(result, plan, workspace)) === result
    @test @inferred(apply!(result, plan, coefficients_data, workspace)) === result

    rhs_allocations = _runtime_allocated(() -> rhs!(result, plan, workspace))
    apply_allocations = _runtime_allocated(() -> apply!(result, plan, coefficients_data, workspace))
    fresh_apply_allocations = _runtime_allocated(() -> apply!(result, plan, coefficients_data))

    @test rhs_allocations <= 10_000
    @test apply_allocations <= 1_000
    @test apply_allocations < fresh_apply_allocations

    @test_throws ArgumentError rhs!(zeros(field_dof_count(u) + 1), plan, workspace)
    @test_throws ArgumentError apply!(result, plan, ones(field_dof_count(u) + 1), workspace)
    @test_throws ArgumentError apply!(view(coefficients_data, :), plan, coefficients_data,
                                      workspace)
  end

  @testset "Assembly Preserves Tiny Contributions" begin
    tiny = 1.0e-14
    domain = Domain((0.0,), (1.0,), (1,))
    space = HpSpace(domain, SpaceOptions(degree=UniformDegree(0), continuity=:dg))
    u = ScalarField(space; name=:u)
    problem = AffineProblem(u)
    add_cell!(problem, _RuntimeScaledIdentity(tiny))
    plan = compile(problem)
    reduced_workspace = Grico._ReducedOperatorWorkspace(plan)
    diagonal = zeros(Grico.reduced_dof_count(plan))

    @test apply(plan, [1.0]) == [tiny]
    @test Grico._reduced_diagonal!(diagonal, plan, reduced_workspace)
    @test diagonal == [tiny]
    @test Grico._assemble_reduced_operator_matrix(plan, reduced_workspace.scratch) ==
          reshape([tiny], 1, 1)

    expansion = Grico._ReducedExpansion(Float64)
    expansion.reduced[1] = tiny
    map = Grico._pack_reduced_operator_map(Float64, 1, [1], [1], _ -> expansion)
    full = zeros(1)
    Grico._expand_reduced!(full, map, [1.0])
    @test full == [tiny]
  end

  @testset "Integration API Contracts" begin
    domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
    space = HpSpace(domain,
                    SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2), continuity=:dg))
    u = ScalarField(space; name=:u)
    problem = AffineProblem(u)
    add_cell!(problem, _RuntimeMassOperator(u))
    plan = compile(problem)
    state = State(plan, ones(field_dof_count(u)))
    cell = plan.integration.cells[1]
    face = plan.integration.boundary_faces[1]
    tensor = tensor_values(cell, u)
    other = ScalarField(space; name=:other)
    wrong_state = State(FieldLayout((other, u)), ones(2 * field_dof_count(u)))

    @test_throws ArgumentError value(cell, wrong_state, u, 1)
    @test_throws ArgumentError gradient(cell, wrong_state, u, 1)
    @test_throws ArgumentError normal_gradient(face, wrong_state, u, 1)

    @test_throws ArgumentError value(cell, Float64[], u, 1)
    @test_throws ArgumentError gradient(cell, Float64[], u, 1)
    @test_throws ArgumentError normal_gradient(face, Float64[], u, 1)
    @test_throws ArgumentError block(Float64[], cell, u)

    @test_throws ArgumentError Grico.point(cell, true)
    @test_throws ArgumentError weight(cell, true)
    @test_throws ArgumentError shape_value(cell, u, true, 1)
    @test_throws ArgumentError local_dof_index(cell, u, true, 1)
    @test_throws ArgumentError tensor_axis_values(tensor, true)
    @test_throws ArgumentError field_gradient(state, u, true, (0.0, 0.0))
    @test_throws ArgumentError field_gradient(state, u, 1, (true, true))

    quadrature_domain = Domain((0.0,), (1.0,), (1,))
    quadrature_space = HpSpace(quadrature_domain, SpaceOptions(degree=UniformDegree(1)))
    q = ScalarField(quadrature_space; name=:q)
    quadrature_problem = AffineProblem(q)
    add_cell!(quadrature_problem, _RuntimeMassOperator(q))
    add_cell_quadrature!(quadrature_problem, 1, _RuntimeBadWeightQuadrature())

    @test_throws ArgumentError compile(quadrature_problem)
  end

  @testset "Residual Matrix-Free Workspace" begin
    domain = Domain((0.0, 0.0), (1.0, 1.0), (2, 2))
    space = HpSpace(domain,
                    SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1), continuity=:dg))
    u = ScalarField(space; name=:u)
    problem = ResidualProblem(u)
    add_cell!(problem, _RuntimeQuadraticOperator(u, 1.0))
    plan = compile(problem)
    workspace = ResidualWorkspace(plan)
    state = State(plan, fill(0.5, field_dof_count(u)))
    increment = ones(field_dof_count(u))
    result = zeros(field_dof_count(u))

    @test @inferred(residual!(result, plan, state, workspace)) === result
    @test @inferred(tangent_apply!(result, plan, state, increment, workspace)) === result

    residual_allocations = _runtime_allocated(() -> residual!(result, plan, state, workspace))
    tangent_allocations = _runtime_allocated(() -> tangent_apply!(result, plan, state, increment,
                                                                  workspace))

    @test residual_allocations <= 1_000
    @test tangent_allocations <= 1_000

    @test_throws ArgumentError residual!(coefficients(state), plan, state, workspace)
    @test_throws ArgumentError tangent_apply!(coefficients(state), plan, state, increment,
                                              workspace)
    @test_throws ArgumentError tangent_apply!(view(increment, :), plan, state, increment, workspace)
    @test_throws ArgumentError tangent_apply!(result, plan, state, ones(field_dof_count(u) + 1),
                                              workspace)
  end

  @testset "Transfer Allocation Smoke" begin
    domain = Domain((0.0, 0.0), (1.0, 1.0), (2, 2))
    space = HpSpace(domain,
                    SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1), continuity=:dg))
    u = ScalarField(space; name=:u)
    state = State(FieldLayout((u,)), ones(field_dof_count(u)))
    plan = AdaptivityPlan(space)
    request_h_refinement!(plan, first(active_leaves(space)), 1)
    transition_data = transition(plan)
    new_u = adapted_field(transition_data, u)
    transferred = transfer_state(transition_data, state, u, new_u)

    @test field_space(new_u) === target_space(transition_data)
    @test fields(field_layout(transferred)) == (new_u,)

    transfer_allocations = _runtime_allocated() do
      transfer_state(transition_data, state, u, new_u)
    end
    @test transfer_allocations <= 500_000
  end
end

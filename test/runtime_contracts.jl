using Test
using Grico
import Grico: target_space

struct _RuntimeMassOperator{F}
  field::F
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

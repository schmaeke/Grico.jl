using Test
using Grico
import Grico: IdentityPreconditioner, JacobiPreconditioner, KernelScratch, PointQuadrature,
              SurfaceQuadrature, add_boundary!, add_cell!, add_interface!, add_surface!,
              add_surface_quadrature!, block, cell_apply!, is_full_tensor, local_dof_index,
              local_mode_count, point_count, scratch_matrix, scratch_vector, shape_gradient,
              shape_value, tensor_degrees, tensor_gradient!, tensor_interpolate!, tensor_mode_count,
              tensor_point_count, tensor_project_gradient!, tensor_quadrature_shape, tensor_values

_threaded_allocation_limit(serial_limit::Integer) = max(serial_limit, 1_000 * Threads.nthreads())

struct _MatrixFreeIdentity end

function Grico.cell_apply!(local_result, ::_MatrixFreeIdentity, values, local_coefficients)
  local_result .+= local_coefficients
  return nothing
end

function Grico.cell_diagonal!(local_diagonal, ::_MatrixFreeIdentity, values)
  local_diagonal .+= 1
  return nothing
end

function Grico.cell_rhs!(local_rhs, ::_MatrixFreeIdentity, values)
  local_rhs .+= 1
  return nothing
end

struct _MatrixFreeNoDiagonalIdentity end

function Grico.cell_apply!(local_result, ::_MatrixFreeNoDiagonalIdentity, values,
                           local_coefficients)
  local_result .+= local_coefficients
  return nothing
end

function Grico.cell_rhs!(local_rhs, ::_MatrixFreeNoDiagonalIdentity, values)
  local_rhs .+= 1
  return nothing
end

struct _MatrixFreeMassAction{F}
  field::F
end

struct _MatrixFreeDiffusion{F,T}
  field::F
  coefficient::T
end

struct _MatrixFreeTensorDiffusion{F,T}
  field::F
  coefficient::T
end

struct _MatrixFreeBoundaryMass{F,T}
  field::F
  coefficient::T
end

struct _MatrixFreeInterfaceJump{F,T}
  field::F
  coefficient::T
end

struct _MatrixFreeSurfaceMass{F,T}
  field::F
  coefficient::T
end

struct _MatrixFreeQuadraticReaction{F,T}
  field::F
  target::T
end

struct _CellAccumulatorFunction{F}
  f::F
end

struct _CellRhsAccumulatorFunction{F}
  f::F
end

struct _CellResidualAccumulatorFunction{R,T}
  residual::R
  tangent::T
end

struct _BoundaryAccumulatorFunction{F}
  f::F
end

struct _InterfaceAccumulatorFunction{F}
  f::F
end

struct _SurfaceAccumulatorFunction{F}
  f::F
end

function Grico.cell_accumulate(operator::_CellAccumulatorFunction, q, trial, test_component)
  return operator.f(q, trial, test_component)
end

function Grico.cell_rhs_accumulate(operator::_CellRhsAccumulatorFunction, q, test_component)
  return operator.f(q, test_component)
end

function Grico.cell_residual_accumulate(operator::_CellResidualAccumulatorFunction, q, state,
                                        test_component)
  return operator.residual(q, state, test_component)
end

function Grico.cell_tangent_accumulate(operator::_CellResidualAccumulatorFunction, q, state,
                                       increment, test_component)
  return operator.tangent(q, state, increment, test_component)
end

function Grico.boundary_accumulate(operator::_BoundaryAccumulatorFunction, q, trial, test_component)
  return operator.f(q, trial, test_component)
end

function Grico.interface_accumulate(operator::_InterfaceAccumulatorFunction, q, trial,
                                    test_component)
  return operator.f(q, trial, test_component)
end

function Grico.surface_accumulate(operator::_SurfaceAccumulatorFunction, q, trial, test_component)
  return operator.f(q, trial, test_component)
end

@inline function _scale_tuple(scale, tuple::NTuple{D,T}) where {D,T}
  return ntuple(axis -> scale * tuple[axis], Val(D))
end

@inline function _add_tuple(first::NTuple{D}, second::NTuple{D}) where {D}
  return ntuple(axis -> first[axis] + second[axis], Val(D))
end

@inline function _with_axis_increment(gradient_value::NTuple{D,T}, scale,
                                      axis_index::Int) where {D,T}
  return ntuple(axis -> gradient_value[axis] + (axis == axis_index ? T(scale) : zero(T)), Val(D))
end

@inline _zero_tuple(tuple::NTuple{D,T}) where {D,T} = ntuple(_ -> zero(T), Val(D))

function _vector_semilinear_operator(shift)
  residual = (q, state, test_component) -> begin
    state_value = value(state)
    state_norm2 = inner(state_value, state_value)
    return TestChannels((state_norm2 + shift) * value(state, test_component),
                        gradient(state, test_component))
  end
  tangent = (q, state, increment, test_component) -> begin
    increment_component = component(increment)
    state_value = value(state)
    state_norm2 = inner(state_value, state_value)
    increment_value = value(increment)
    reaction_scale = 2 * value(state, test_component) * value(state, increment_component) +
                     (test_component == increment_component ? state_norm2 + shift :
                      zero(state_norm2))
    gradient_coefficient = test_component == increment_component ? gradient(increment) :
                           _zero_tuple(gradient(increment))
    return TestChannels(reaction_scale * increment_value, gradient_coefficient)
  end
  return _CellResidualAccumulatorFunction(residual, tangent)
end

function Grico.cell_apply!(local_result, operator::_MatrixFreeMassAction, values,
                           local_coefficients)
  field = operator.field

  for point_index in 1:point_count(values)
    field_value = value(values, local_coefficients, field, point_index)
    weighted = weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_result[row] += shape_value(values, field, point_index, mode_index) *
                           weighted *
                           field_value
    end
  end

  return nothing
end

function Grico.cell_diagonal!(local_diagonal, operator::_MatrixFreeMassAction, values)
  field = operator.field

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      shape = shape_value(values, field, point_index, mode_index)
      local_diagonal[row] += shape * shape * weighted
    end
  end

  return nothing
end

function Grico.cell_apply!(local_result, operator::_MatrixFreeDiffusion, values, local_coefficients)
  field = operator.field
  mode_count = local_mode_count(values, field)

  for point_index in 1:point_count(values)
    field_gradient_value = gradient(values, local_coefficients, field, point_index)
    weighted = operator.coefficient * weight(values, point_index)

    for mode_index in 1:mode_count
      row = local_dof_index(values, field, 1, mode_index)
      test_gradient = shape_gradient(values, field, point_index, mode_index)
      local_result[row] += _tuple_dot(field_gradient_value, test_gradient) * weighted
    end
  end

  return nothing
end

function Grico.cell_diagonal!(local_diagonal, operator::_MatrixFreeDiffusion, values)
  field = operator.field
  mode_count = local_mode_count(values, field)

  for point_index in 1:point_count(values)
    weighted = operator.coefficient * weight(values, point_index)

    for mode_index in 1:mode_count
      row = local_dof_index(values, field, 1, mode_index)
      test_gradient = shape_gradient(values, field, point_index, mode_index)
      local_diagonal[row] += _tuple_dot(test_gradient, test_gradient) * weighted
    end
  end

  return nothing
end

function Grico.cell_apply!(local_result, operator::_MatrixFreeTensorDiffusion, values,
                           local_coefficients, scratch::KernelScratch)
  field = operator.field
  tensor = tensor_values(values, field)
  tensor !== nothing ||
    return Grico.cell_apply!(local_result, _MatrixFreeDiffusion(field, operator.coefficient),
                             values, local_coefficients)
  @assert is_full_tensor(tensor)
  local_input = block(local_coefficients, values, field)
  local_output = block(local_result, values, field)
  gradients = scratch_matrix(scratch, 1, length(tensor_degrees(tensor)), tensor_point_count(tensor))
  tensor_gradient!(gradients, tensor, local_input, scratch)

  for point_index in 1:tensor_point_count(tensor)
    weighted = operator.coefficient * weight(values, point_index)

    for axis in 1:size(gradients, 1)
      gradients[axis, point_index] *= weighted
    end
  end

  tensor_project_gradient!(local_output, tensor, gradients, scratch)
  return nothing
end

function Grico.face_apply!(local_result, operator::_MatrixFreeBoundaryMass, values,
                           local_coefficients)
  field = operator.field

  for point_index in 1:point_count(values)
    field_value = value(values, local_coefficients, field, point_index)
    weighted = operator.coefficient * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_result[row] += shape_value(values, field, point_index, mode_index) *
                           field_value *
                           weighted
    end
  end

  return nothing
end

function Grico.face_diagonal!(local_diagonal, operator::_MatrixFreeBoundaryMass, values)
  field = operator.field

  for point_index in 1:point_count(values)
    weighted = operator.coefficient * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      shape = shape_value(values, field, point_index, mode_index)
      local_diagonal[row] += shape * shape * weighted
    end
  end

  return nothing
end

function Grico.interface_apply!(local_result, operator::_MatrixFreeInterfaceJump, values,
                                local_coefficients)
  field = operator.field
  minus_values = minus(values)
  plus_values = plus(values)
  minus_block = block(local_result, minus_values, field)
  plus_block = block(local_result, plus_values, field)

  for point_index in 1:point_count(values)
    jump_value = jump(value(minus_values, local_coefficients, field, point_index),
                      value(plus_values, local_coefficients, field, point_index))
    weighted = operator.coefficient * weight(values, point_index)

    for mode_index in 1:local_mode_count(minus_values, field)
      minus_block[mode_index] -= shape_value(minus_values, field, point_index, mode_index) *
                                 jump_value *
                                 weighted
    end

    for mode_index in 1:local_mode_count(plus_values, field)
      plus_block[mode_index] += shape_value(plus_values, field, point_index, mode_index) *
                                jump_value *
                                weighted
    end
  end

  return nothing
end

function Grico.interface_diagonal!(local_diagonal, operator::_MatrixFreeInterfaceJump, values)
  field = operator.field
  minus_values = minus(values)
  plus_values = plus(values)
  minus_block = block(local_diagonal, minus_values, field)
  plus_block = block(local_diagonal, plus_values, field)

  for point_index in 1:point_count(values)
    weighted = operator.coefficient * weight(values, point_index)

    for mode_index in 1:local_mode_count(minus_values, field)
      shape = shape_value(minus_values, field, point_index, mode_index)
      minus_block[mode_index] += shape * shape * weighted
    end

    for mode_index in 1:local_mode_count(plus_values, field)
      shape = shape_value(plus_values, field, point_index, mode_index)
      plus_block[mode_index] += shape * shape * weighted
    end
  end

  return nothing
end

function Grico.surface_apply!(local_result, operator::_MatrixFreeSurfaceMass, values,
                              local_coefficients)
  field = operator.field

  for point_index in 1:point_count(values)
    field_value = value(values, local_coefficients, field, point_index)
    weighted = operator.coefficient * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_result[row] += shape_value(values, field, point_index, mode_index) *
                           field_value *
                           weighted
    end
  end

  return nothing
end

function Grico.surface_diagonal!(local_diagonal, operator::_MatrixFreeSurfaceMass, values)
  field = operator.field

  for point_index in 1:point_count(values)
    weighted = operator.coefficient * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      shape = shape_value(values, field, point_index, mode_index)
      local_diagonal[row] += shape * shape * weighted
    end
  end

  return nothing
end

function Grico.cell_residual!(local_residual, operator::_MatrixFreeQuadraticReaction, values, state)
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

function Grico.cell_tangent_apply!(local_result, operator::_MatrixFreeQuadraticReaction, values,
                                   state, local_increment)
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

function _tuple_dot(first::Tuple, second::Tuple)
  sum(first[index] * second[index] for index in eachindex(first))
end

function _local_reference_mass!(local_matrix, field, values)
  block_data = block(local_matrix, values, field, field)
  mode_count = local_mode_count(values, field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      row_shape = shape_value(values, field, point_index, row_mode)

      for col_mode in 1:mode_count
        block_data[row_mode, col_mode] += row_shape *
                                          shape_value(values, field, point_index, col_mode) *
                                          weighted
      end
    end
  end

  return local_matrix
end

function _local_reference_diffusion!(local_matrix, field, values, coefficient)
  block_data = block(local_matrix, values, field, field)
  mode_count = local_mode_count(values, field)

  for point_index in 1:point_count(values)
    weighted = coefficient * weight(values, point_index)

    for row_mode in 1:mode_count
      row_gradient = shape_gradient(values, field, point_index, row_mode)

      for col_mode in 1:mode_count
        col_gradient = shape_gradient(values, field, point_index, col_mode)
        block_data[row_mode, col_mode] += _tuple_dot(row_gradient, col_gradient) * weighted
      end
    end
  end

  return local_matrix
end

function _reference_cell_action(plan, coefficients, local_matrix_function)
  result = zeros(eltype(coefficients), length(coefficients))

  for cell in plan.integration.cells
    local_matrix = zeros(eltype(coefficients), cell.local_dof_count, cell.local_dof_count)
    local_matrix_function(local_matrix, cell)
    local_coefficients = _reference_local_coefficients(cell, coefficients)

    for local_row in 1:cell.local_dof_count
      value = zero(eltype(coefficients))

      for local_col in 1:cell.local_dof_count
        value += local_matrix[local_row, local_col] * local_coefficients[local_col]
      end

      _reference_scatter!(result, cell, local_row, value)
    end
  end

  return result
end

function _reference_local_coefficients(item, coefficients)
  local_coefficients = zeros(eltype(coefficients), item.local_dof_count)

  for local_dof in 1:item.local_dof_count
    value = zero(eltype(coefficients))

    for term_index in item.term_offsets[local_dof]:(item.term_offsets[local_dof+1]-1)
      value += item.term_coefficients[term_index] * coefficients[item.term_indices[term_index]]
    end

    local_coefficients[local_dof] = value
  end

  return local_coefficients
end

function _reference_scatter!(result, item, local_row, value)
  iszero(value) && return result

  for term_index in item.term_offsets[local_row]:(item.term_offsets[local_row+1]-1)
    result[item.term_indices[term_index]] += item.term_coefficients[term_index] * value
  end

  return result
end

function _reference_reduced_diagonal(plan)
  workspace = Grico._ReducedOperatorWorkspace(plan)
  count = Grico.reduced_dof_count(plan)
  basis = zeros(count)
  response = zeros(count)
  diagonal = zeros(count)

  for index in 1:count
    basis[index] = 1.0
    Grico._reduced_apply!(response, plan, basis, workspace)
    diagonal[index] = response[index]
    basis[index] = 0.0
  end

  return diagonal
end

function _kernel_reduced_diagonal(plan)
  diagonal = zeros(Grico.reduced_dof_count(plan))
  selected = Grico._reduced_diagonal!(diagonal, plan, Grico._ReducedOperatorWorkspace(plan))
  return selected, diagonal
end

@testset "Matrix-free affine operators" begin
  domain = Domain((0.0,), (1.0,), (1,))
  dg_space = HpSpace(domain, SpaceOptions(degree=UniformDegree(1), continuity=:dg))
  dg_field = ScalarField(dg_space; name=:u)

  identity_problem = AffineProblem(dg_field)
  add_cell!(identity_problem, _MatrixFreeIdentity())
  identity_state = solve(identity_problem)
  @test coefficients(identity_state) ≈ [1.0, 1.0]

  no_diagonal_problem = AffineProblem(dg_field)
  add_cell!(no_diagonal_problem, _MatrixFreeNoDiagonalIdentity())
  no_diagonal_plan = compile(no_diagonal_problem)
  no_diagonal_workspace = Grico._ReducedOperatorWorkspace(no_diagonal_plan)
  no_diagonal_operator = Grico._ReducedAffineOperator(no_diagonal_plan, no_diagonal_workspace)
  @test Grico._compile_preconditioner(JacobiPreconditioner(), no_diagonal_operator) isa
        Grico._JacobiCompiledPreconditioner
  counting_operator = Grico._CountingReducedOperator(no_diagonal_operator, Ref(0))
  @test Grico._compile_preconditioner(IdentityPreconditioner(), counting_operator) isa
        Grico._IdentityCompiledPreconditioner
  no_diagonal_state = solve(no_diagonal_plan;
                            solver=CGSolver(preconditioner=JacobiPreconditioner()))
  @test coefficients(no_diagonal_state) ≈ [1.0, 1.0]

  mass_problem = AffineProblem(dg_field)
  add_cell!(mass_problem, _MatrixFreeMassAction(dg_field))
  mass_plan = compile(mass_problem)
  @test apply(mass_plan, ones(2)) ≈ [0.5, 0.5]
  @test apply(mass_plan, [0.25, -0.5]) ≈ _reference_cell_action(mass_plan, [0.25, -0.5],
                                                                (local_matrix, values) -> _local_reference_mass!(local_matrix,
                                                                                                                 dg_field, values))
  mass_diagonal_selected, mass_diagonal = _kernel_reduced_diagonal(mass_plan)
  @test mass_diagonal_selected
  @test mass_diagonal ≈ _reference_reduced_diagonal(mass_plan)

  accumulator_mass_problem = AffineProblem(dg_field; operator_class=SPD())
  add_cell_accumulator!(accumulator_mass_problem, dg_field, dg_field,
                        _CellAccumulatorFunction((q, trial, test_component) -> value(trial)))
  add_cell_accumulator!(accumulator_mass_problem, dg_field,
                        _CellRhsAccumulatorFunction((q, test_component) -> 1.0))
  accumulator_mass_plan = compile(accumulator_mass_problem)
  @test apply(accumulator_mass_plan, [0.25, -0.5]) ≈ apply(mass_plan, [0.25, -0.5])
  @test rhs(accumulator_mass_plan) ≈ [0.5, 0.5]
  accumulator_mass_matrix = Grico._assemble_reduced_operator_matrix(accumulator_mass_plan,
                                                                    Grico._ReducedOperatorWorkspace(accumulator_mass_plan).scratch)
  @test accumulator_mass_matrix * [0.25, -0.5] ≈ apply(accumulator_mass_plan, [0.25, -0.5])

  diffusion_problem = AffineProblem(dg_field)
  add_cell!(diffusion_problem, _MatrixFreeDiffusion(dg_field, 2.0))
  diffusion_plan = compile(diffusion_problem)
  diffusion_coefficients = [0.25, -0.5]
  @test apply(diffusion_plan, diffusion_coefficients) ≈
        _reference_cell_action(diffusion_plan, diffusion_coefficients,
                               (local_matrix, values) -> _local_reference_diffusion!(local_matrix,
                                                                                     dg_field,
                                                                                     values, 2.0))
  diffusion_diagonal_selected, diffusion_diagonal = _kernel_reduced_diagonal(diffusion_plan)
  @test diffusion_diagonal_selected
  @test diffusion_diagonal ≈ _reference_reduced_diagonal(diffusion_plan)

  accumulator_diffusion_problem = AffineProblem(dg_field; operator_class=SPD())
  add_cell_accumulator!(accumulator_diffusion_problem, dg_field, dg_field,
                        _CellAccumulatorFunction((q, trial, test_component) -> TestChannels(zero(value(trial)),
                                                                                            _scale_tuple(2.0,
                                                                                                         gradient(trial)))))
  accumulator_diffusion_plan = compile(accumulator_diffusion_problem)
  @test apply(accumulator_diffusion_plan, diffusion_coefficients) ≈
        apply(diffusion_plan, diffusion_coefficients)
  accumulator_diffusion_diagonal_selected, accumulator_diffusion_diagonal = _kernel_reduced_diagonal(accumulator_diffusion_plan)
  @test accumulator_diffusion_diagonal_selected
  @test accumulator_diffusion_diagonal ≈ _reference_reduced_diagonal(accumulator_diffusion_plan)

  tensor_domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  tensor_space = HpSpace(tensor_domain,
                         SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2),
                                      continuity=:dg))
  tensor_field = ScalarField(tensor_space; name=:u)
  tensor_problem = AffineProblem(tensor_field)
  add_cell!(tensor_problem, _MatrixFreeTensorDiffusion(tensor_field, 1.75))
  tensor_plan = compile(tensor_problem)
  tensor_coefficients = [sin(0.3 * index) for index in 1:field_dof_count(tensor_field)]
  @test apply(tensor_plan, tensor_coefficients) ≈
        _reference_cell_action(tensor_plan, tensor_coefficients,
                               (local_matrix, values) -> _local_reference_diffusion!(local_matrix,
                                                                                     tensor_field,
                                                                                     values, 1.75))
  accumulator_tensor_problem = AffineProblem(tensor_field; operator_class=SPD())
  add_cell_accumulator!(accumulator_tensor_problem, tensor_field, tensor_field,
                        _CellAccumulatorFunction((q, trial, test_component) -> TestChannels(zero(value(trial)),
                                                                                            _scale_tuple(1.75,
                                                                                                         gradient(trial)))))
  accumulator_tensor_plan = compile(accumulator_tensor_problem)
  @test apply(accumulator_tensor_plan, tensor_coefficients) ≈
        apply(tensor_plan, tensor_coefficients)
  tensor_item = tensor_plan.integration.cells[1]
  tensor_data = tensor_values(tensor_item, tensor_field)
  @test tensor_data !== nothing
  @test is_full_tensor(tensor_data)
  @test tensor_degrees(tensor_data) == (2, 2)
  @test tensor_quadrature_shape(tensor_data) == (3, 3)
  tensor_scratch = KernelScratch(Float64)
  tensor_input = scratch_vector(tensor_scratch, 1, tensor_mode_count(tensor_data))
  copyto!(tensor_input, tensor_coefficients)
  tensor_point_values = zeros(Float64, tensor_point_count(tensor_data))
  tensor_interpolate!(tensor_point_values, tensor_data, tensor_input, tensor_scratch, 2)
  @test tensor_point_values ≈ [value(tensor_item, tensor_coefficients, tensor_field, point_index)
                               for point_index in 1:point_count(tensor_item)]
  tensor_points = tensor_point_count(tensor_data)

  accumulator_tensor_mass_problem = AffineProblem(tensor_field; operator_class=SPD())
  add_cell_accumulator!(accumulator_tensor_mass_problem, tensor_field, tensor_field,
                        _CellAccumulatorFunction((q, trial, test_component) -> value(trial)))
  accumulator_tensor_mass_plan = compile(accumulator_tensor_mass_problem)
  accumulator_tensor_mass_scratch = KernelScratch(Float64)
  accumulator_tensor_mass_output = zeros(field_dof_count(tensor_field))
  cell_apply!(accumulator_tensor_mass_output, accumulator_tensor_mass_plan.cell_operators[1],
              accumulator_tensor_mass_plan.integration.cells[1], tensor_coefficients,
              accumulator_tensor_mass_scratch)
  @test size(accumulator_tensor_mass_scratch.matrices[1]) == (2, tensor_points)

  accumulator_tensor_diffusion_scratch = KernelScratch(Float64)
  accumulator_tensor_diffusion_output = zeros(field_dof_count(tensor_field))
  cell_apply!(accumulator_tensor_diffusion_output, accumulator_tensor_plan.cell_operators[1],
              accumulator_tensor_plan.integration.cells[1], tensor_coefficients,
              accumulator_tensor_diffusion_scratch)
  @test length(accumulator_tensor_diffusion_scratch.vectors[5]) == tensor_points

  semilinear_problem = ResidualProblem(tensor_field; operator_class=SPD())
  add_cell_accumulator!(semilinear_problem, tensor_field, tensor_field,
                        _CellResidualAccumulatorFunction((q, state, test_component) -> TestChannels(value(state)^2 -
                                                                                                    0.25,
                                                                                                    gradient(state)),
                                                         (q, state, increment, test_component) -> TestChannels(2 *
                                                                                                               value(state) *
                                                                                                               value(increment),
                                                                                                               gradient(increment))))
  semilinear_plan = compile(semilinear_problem)
  semilinear_state = State(semilinear_plan, tensor_coefficients)
  semilinear_increment = [cos(0.19 * index) for index in 1:field_dof_count(tensor_field)]
  semilinear_residual = residual(semilinear_plan, semilinear_state)
  semilinear_tangent = tangent_apply(semilinear_plan, semilinear_state, semilinear_increment)
  finite_difference_step = 1.0e-7
  perturbed_state = State(semilinear_plan,
                          tensor_coefficients .+ finite_difference_step .* semilinear_increment)
  finite_difference_tangent = (residual(semilinear_plan, perturbed_state) .- semilinear_residual) ./
                              finite_difference_step
  @test semilinear_tangent ≈ finite_difference_tangent rtol = 1.0e-6 atol = 1.0e-6

  semilinear_residual_scratch = KernelScratch(Float64)
  semilinear_local_residual = zeros(field_dof_count(tensor_field))
  Grico.cell_residual!(semilinear_local_residual, semilinear_plan.cell_operators[1],
                       semilinear_plan.integration.cells[1], semilinear_state,
                       semilinear_residual_scratch)
  @test size(semilinear_residual_scratch.matrices[2]) == (tensor_points, 1)
  @test size(semilinear_residual_scratch.matrices[3]) == (tensor_points, 2)
  @test size(semilinear_residual_scratch.matrices[4]) == (2, tensor_points)
  @test size(semilinear_residual_scratch.matrices[5]) == (tensor_points, 1)
  @test size(semilinear_residual_scratch.matrices[6]) == (tensor_points, 2)

  semilinear_tangent_scratch = KernelScratch(Float64)
  semilinear_local_tangent = zeros(field_dof_count(tensor_field))
  Grico.cell_tangent_apply!(semilinear_local_tangent, semilinear_plan.cell_operators[1],
                            semilinear_plan.integration.cells[1], semilinear_state,
                            semilinear_increment, semilinear_tangent_scratch)
  @test size(semilinear_tangent_scratch.matrices[2]) == (tensor_points, 1)
  @test size(semilinear_tangent_scratch.matrices[3]) == (tensor_points, 2)
  @test size(semilinear_tangent_scratch.matrices[4]) == (2, tensor_points)
  @test size(semilinear_tangent_scratch.matrices[5]) == (tensor_points, 1)
  @test size(semilinear_tangent_scratch.matrices[6]) == (tensor_points, 2)
  @test size(semilinear_tangent_scratch.matrices[7]) == (2, tensor_points)

  vector_tensor_field = VectorField(tensor_space, 2; name=:velocity)
  vector_tensor_coefficients = [sin(0.17 * index) + 0.1 * cos(0.31 * index)
                                for index in 1:field_dof_count(vector_tensor_field)]

  vector_semilinear_problem = ResidualProblem(vector_tensor_field)
  add_cell_accumulator!(vector_semilinear_problem, vector_tensor_field, vector_tensor_field,
                        _vector_semilinear_operator(-0.25))
  vector_semilinear_plan = compile(vector_semilinear_problem)
  vector_semilinear_state = State(vector_semilinear_plan, vector_tensor_coefficients)
  vector_semilinear_increment = [cos(0.13 * index)
                                 for index in 1:field_dof_count(vector_tensor_field)]
  vector_semilinear_residual = residual(vector_semilinear_plan, vector_semilinear_state)
  vector_semilinear_tangent = tangent_apply(vector_semilinear_plan, vector_semilinear_state,
                                            vector_semilinear_increment)
  vector_perturbed_state = State(vector_semilinear_plan,
                                 vector_tensor_coefficients .+
                                 finite_difference_step .* vector_semilinear_increment)
  vector_finite_difference_tangent = (residual(vector_semilinear_plan, vector_perturbed_state) .-
                                      vector_semilinear_residual) ./ finite_difference_step
  @test vector_semilinear_tangent ≈ vector_finite_difference_tangent rtol = 2.0e-6 atol = 2.0e-6

  vector_tensor_problem = AffineProblem(vector_tensor_field; operator_class=NonsymmetricOperator())
  add_cell_accumulator!(vector_tensor_problem, vector_tensor_field, vector_tensor_field,
                        _CellAccumulatorFunction((q, trial, test_component) -> begin
                                                   trial_component = component(trial)
                                                   mass_scale = test_component == trial_component ?
                                                                2.0 : -0.25
                                                   diffusion_scale = test_component ==
                                                                     trial_component ? 1.1 : 0.35
                                                   trial_gradient = gradient(trial)
                                                   value_coefficient = mass_scale * value(trial) +
                                                                       0.07 *
                                                                       trial_gradient[test_component]
                                                   gradient_coefficient = _with_axis_increment(_scale_tuple(diffusion_scale,
                                                                                                            trial_gradient),
                                                                                               0.05 *
                                                                                               value(trial),
                                                                                               trial_component)
                                                   return TestChannels(value_coefficient,
                                                                       gradient_coefficient)
                                                 end))
  vector_tensor_plan = compile(vector_tensor_problem)
  vector_tensor_matrix = Grico._assemble_reduced_operator_matrix(vector_tensor_plan,
                                                                 Grico._ReducedOperatorWorkspace(vector_tensor_plan).scratch)
  @test apply(vector_tensor_plan, vector_tensor_coefficients) ≈
        vector_tensor_matrix * vector_tensor_coefficients
  vector_tensor_diagonal_selected, vector_tensor_diagonal = _kernel_reduced_diagonal(vector_tensor_plan)
  @test vector_tensor_diagonal_selected
  @test vector_tensor_diagonal ≈ _reference_reduced_diagonal(vector_tensor_plan)

  trunk_vector_space = HpSpace(tensor_domain, SpaceOptions(degree=UniformDegree(2), continuity=:dg))
  trunk_vector_field = VectorField(trunk_vector_space, 2; name=:trunk_velocity)
  trunk_vector_coefficients = [cos(0.19 * index) for index in 1:field_dof_count(trunk_vector_field)]

  trunk_vector_semilinear_problem = ResidualProblem(trunk_vector_field)
  add_cell_accumulator!(trunk_vector_semilinear_problem, trunk_vector_field, trunk_vector_field,
                        _vector_semilinear_operator(0.5))
  trunk_vector_semilinear_plan = compile(trunk_vector_semilinear_problem)
  @test !is_full_tensor(tensor_values(trunk_vector_semilinear_plan.integration.cells[1],
                                      trunk_vector_field))
  trunk_vector_semilinear_state = State(trunk_vector_semilinear_plan, trunk_vector_coefficients)
  trunk_vector_semilinear_increment = [sin(0.23 * index)
                                       for index in 1:field_dof_count(trunk_vector_field)]
  trunk_vector_semilinear_residual = residual(trunk_vector_semilinear_plan,
                                              trunk_vector_semilinear_state)
  trunk_vector_semilinear_tangent = tangent_apply(trunk_vector_semilinear_plan,
                                                  trunk_vector_semilinear_state,
                                                  trunk_vector_semilinear_increment)
  trunk_vector_perturbed_state = State(trunk_vector_semilinear_plan,
                                       trunk_vector_coefficients .+
                                       finite_difference_step .* trunk_vector_semilinear_increment)
  trunk_vector_finite_difference_tangent = (residual(trunk_vector_semilinear_plan,
                                                     trunk_vector_perturbed_state) .-
                                            trunk_vector_semilinear_residual) ./
                                           finite_difference_step
  @test trunk_vector_semilinear_tangent ≈ trunk_vector_finite_difference_tangent rtol = 3.0e-6 atol = 3.0e-6

  trunk_vector_problem = AffineProblem(trunk_vector_field; operator_class=NonsymmetricOperator())
  add_cell_accumulator!(trunk_vector_problem, trunk_vector_field, trunk_vector_field,
                        _CellAccumulatorFunction((q, trial, test_component) -> begin
                                                   trial_component = component(trial)
                                                   if test_component == trial_component
                                                     return TestChannels(value(trial),
                                                                         gradient(trial))
                                                   end
                                                   return TestChannels(0.2 *
                                                                       gradient(trial)[test_component],
                                                                       _zero_tuple(gradient(trial)))
                                                 end))
  trunk_vector_plan = compile(trunk_vector_problem)
  trunk_vector_matrix = Grico._assemble_reduced_operator_matrix(trunk_vector_plan,
                                                                Grico._ReducedOperatorWorkspace(trunk_vector_plan).scratch)
  @test apply(trunk_vector_plan, trunk_vector_coefficients) ≈
        trunk_vector_matrix * trunk_vector_coefficients

  vector_3d_domain = Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
  vector_3d_space = HpSpace(vector_3d_domain,
                            SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2),
                                         continuity=:dg))
  vector_3d_field = VectorField(vector_3d_space, 3; name=:displacement)
  vector_3d_problem = AffineProblem(vector_3d_field; operator_class=NonsymmetricOperator())
  add_cell_accumulator!(vector_3d_problem, vector_3d_field, vector_3d_field,
                        _CellAccumulatorFunction((q, trial, test_component) -> begin
                                                   trial_component = component(trial)
                                                   coupling = test_component == trial_component ?
                                                              1.4 : 0.2
                                                   trial_gradient = gradient(trial)
                                                   value_coefficient = 0.04 *
                                                                       trial_gradient[test_component]
                                                   gradient_coefficient = _with_axis_increment(_scale_tuple(coupling,
                                                                                                            trial_gradient),
                                                                                               0.03 *
                                                                                               value(trial),
                                                                                               trial_component)
                                                   return TestChannels(value_coefficient,
                                                                       gradient_coefficient)
                                                 end))
  vector_3d_plan = compile(vector_3d_problem)
  vector_3d_coefficients = [sin(0.07 * index) for index in 1:field_dof_count(vector_3d_field)]
  vector_3d_matrix = Grico._assemble_reduced_operator_matrix(vector_3d_plan,
                                                             Grico._ReducedOperatorWorkspace(vector_3d_plan).scratch)
  @test apply(vector_3d_plan, vector_3d_coefficients) ≈ vector_3d_matrix * vector_3d_coefficients

  boundary_problem = AffineProblem(dg_field)
  add_boundary!(boundary_problem, BoundaryFace(1, UPPER), _MatrixFreeBoundaryMass(dg_field, 3.0))
  boundary_plan = compile(boundary_problem)
  boundary_diagonal_selected, boundary_diagonal = _kernel_reduced_diagonal(boundary_plan)
  @test boundary_diagonal_selected
  @test boundary_diagonal ≈ _reference_reduced_diagonal(boundary_plan)
  @test apply(boundary_plan, [0.25, -0.5]) ≈ [0.0, -1.5]

  accumulator_boundary_problem = AffineProblem(dg_field; operator_class=SPD())
  add_boundary_accumulator!(accumulator_boundary_problem, BoundaryFace(1, UPPER), dg_field,
                            dg_field,
                            _BoundaryAccumulatorFunction((q, trial, test_component) -> 3.0 *
                                                                                       value(trial)))
  accumulator_boundary_plan = compile(accumulator_boundary_problem)
  @test apply(accumulator_boundary_plan, [0.25, -0.5]) ≈ apply(boundary_plan, [0.25, -0.5])

  vector_boundary_domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  vector_boundary_space = HpSpace(vector_boundary_domain,
                                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(3),
                                               continuity=:dg))
  vector_boundary_field = VectorField(vector_boundary_space, 2; name=:velocity)
  vector_boundary_problem = AffineProblem(vector_boundary_field;
                                          operator_class=NonsymmetricOperator())
  add_boundary_accumulator!(vector_boundary_problem, BoundaryFace(1, UPPER), vector_boundary_field,
                            vector_boundary_field,
                            _BoundaryAccumulatorFunction((q, trial, test_component) -> begin
                                                           trial_component = component(trial)
                                                           coupling = test_component ==
                                                                      trial_component ? 1.3 : -0.25
                                                           normal_value = normal(q)
                                                           trial_gradient = gradient(trial)
                                                           trial_normal = normal_component(trial_gradient,
                                                                                           normal_value)
                                                           value_coefficient = coupling *
                                                                               value(trial) +
                                                                               0.15 * trial_normal
                                                           gradient_coefficient = _with_axis_increment(_scale_tuple(0.2 *
                                                                                                                    value(trial),
                                                                                                                    normal_value),
                                                                                                       0.03 *
                                                                                                       trial_gradient[test_component],
                                                                                                       trial_component)
                                                           return TestChannels(value_coefficient,
                                                                               gradient_coefficient)
                                                         end))
  vector_boundary_plan = compile(vector_boundary_problem)
  vector_boundary_coefficients = [sin(0.13 * index)
                                  for index in 1:field_dof_count(vector_boundary_field)]
  vector_boundary_matrix = Grico._assemble_reduced_operator_matrix(vector_boundary_plan,
                                                                   Grico._ReducedOperatorWorkspace(vector_boundary_plan).scratch)
  @test apply(vector_boundary_plan, vector_boundary_coefficients) ≈
        vector_boundary_matrix * vector_boundary_coefficients

  interface_domain = Domain((0.0,), (1.0,), (2,))
  interface_space = HpSpace(interface_domain, SpaceOptions(degree=UniformDegree(1), continuity=:dg))
  interface_field = ScalarField(interface_space; name=:u)
  interface_problem = AffineProblem(interface_field)
  add_interface!(interface_problem, _MatrixFreeInterfaceJump(interface_field, 2.0))
  interface_plan = compile(interface_problem)
  interface_diagonal_selected, interface_diagonal = _kernel_reduced_diagonal(interface_plan)
  @test interface_diagonal_selected
  @test interface_diagonal ≈ _reference_reduced_diagonal(interface_plan)

  accumulator_interface_problem = AffineProblem(interface_field; operator_class=SPD())
  add_interface_accumulator!(accumulator_interface_problem, interface_field, interface_field,
                             _InterfaceAccumulatorFunction((q, trial, test_component) -> begin
                                                             jump_value = jump(value(trial))
                                                             return TraceTestChannels(TestChannels(-2.0 *
                                                                                                   jump_value,
                                                                                                   _zero_tuple(normal(q))),
                                                                                      TestChannels(2.0 *
                                                                                                   jump_value,
                                                                                                   _zero_tuple(normal(q))))
                                                           end))
  accumulator_interface_plan = compile(accumulator_interface_problem)
  accumulator_interface_coefficients = [0.25, -0.5, 0.75, -0.125]
  @test apply(accumulator_interface_plan, accumulator_interface_coefficients) ≈
        apply(interface_plan, accumulator_interface_coefficients)

  accumulator_interface_penalty_problem = AffineProblem(interface_field; operator_class=SPD())
  add_interface_accumulator!(accumulator_interface_penalty_problem, interface_field,
                             interface_field,
                             _InterfaceAccumulatorFunction((q, trial, test_component) -> begin
                                                             coefficient = minimum(cell_size(q,
                                                                                             interface_field)) *
                                                                           jump(value(trial))
                                                             return TraceTestChannels(TestChannels(-coefficient,
                                                                                                   _zero_tuple(normal(q))),
                                                                                      TestChannels(coefficient,
                                                                                                   _zero_tuple(normal(q))))
                                                           end))
  accumulator_interface_penalty_plan = compile(accumulator_interface_penalty_problem)
  @test apply(accumulator_interface_penalty_plan, ones(field_dof_count(interface_field))) ≈
        zeros(field_dof_count(interface_field))

  vector_interface_domain = Domain((0.0, 0.0), (1.0, 1.0), (2, 1))
  vector_interface_space = HpSpace(vector_interface_domain,
                                   SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(3),
                                                continuity=:dg))
  vector_interface_field = VectorField(vector_interface_space, 2; name=:velocity)
  vector_interface_problem = AffineProblem(vector_interface_field;
                                           operator_class=NonsymmetricOperator())
  add_interface_accumulator!(vector_interface_problem, vector_interface_field,
                             vector_interface_field,
                             _InterfaceAccumulatorFunction((q, trial, test_component) -> begin
                                                             trial_component = component(trial)
                                                             coupling = test_component ==
                                                                        trial_component ? 2.0 : 0.35
                                                             normal_value = normal(q)
                                                             jump_value = jump(value(trial))
                                                             trial_normal = average(normal_component(gradient(trial),
                                                                                                     normal_value))
                                                             value_coefficient = coupling *
                                                                                 jump_value +
                                                                                 0.2 * trial_normal
                                                             shared_normal_gradient = _scale_tuple(0.15 *
                                                                                                   jump_value,
                                                                                                   normal_value)
                                                             average_trial_gradient = average(gradient(trial))
                                                             minus_gradient = _add_tuple(shared_normal_gradient,
                                                                                         _scale_tuple(-0.04,
                                                                                                      average_trial_gradient))
                                                             plus_gradient = _add_tuple(shared_normal_gradient,
                                                                                        _scale_tuple(0.04,
                                                                                                     average_trial_gradient))
                                                             return TraceTestChannels(TestChannels(-value_coefficient,
                                                                                                   minus_gradient),
                                                                                      TestChannels(value_coefficient,
                                                                                                   plus_gradient))
                                                           end))
  vector_interface_plan = compile(vector_interface_problem)
  vector_interface_coefficients = [cos(0.09 * index)
                                   for index in 1:field_dof_count(vector_interface_field)]
  vector_interface_matrix = Grico._assemble_reduced_operator_matrix(vector_interface_plan,
                                                                    Grico._ReducedOperatorWorkspace(vector_interface_plan).scratch)
  @test apply(vector_interface_plan, vector_interface_coefficients) ≈
        vector_interface_matrix * vector_interface_coefficients

  surface_problem = AffineProblem(dg_field)
  surface_quadrature = PointQuadrature([(0.0,)], [1.0])
  add_surface_quadrature!(surface_problem, SurfaceQuadrature(1, surface_quadrature, [(1.0,)]))
  add_surface!(surface_problem, _MatrixFreeSurfaceMass(dg_field, 5.0))
  surface_plan = compile(surface_problem)
  surface_diagonal_selected, surface_diagonal = _kernel_reduced_diagonal(surface_plan)
  @test surface_diagonal_selected
  @test surface_diagonal ≈ _reference_reduced_diagonal(surface_plan)

  accumulator_surface_problem = AffineProblem(dg_field; operator_class=SPD())
  add_surface_quadrature!(accumulator_surface_problem,
                          SurfaceQuadrature(1, surface_quadrature, [(1.0,)]))
  add_surface_accumulator!(accumulator_surface_problem, dg_field, dg_field,
                           _SurfaceAccumulatorFunction((q, trial, test_component) -> 5.0 *
                                                                                     value(trial)))
  accumulator_surface_plan = compile(accumulator_surface_problem)
  @test apply(accumulator_surface_plan, [0.25, -0.5]) ≈ apply(surface_plan, [0.25, -0.5])

  vector_surface_domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  vector_surface_space = HpSpace(vector_surface_domain,
                                 SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(3),
                                              continuity=:dg))
  vector_surface_field = VectorField(vector_surface_space, 2; name=:velocity)
  vector_surface_problem = AffineProblem(vector_surface_field;
                                         operator_class=NonsymmetricOperator())
  vector_surface_quadrature = Grico.TensorQuadrature(Float64, (1, 4))
  vector_surface_normals = fill((1.0, 0.0), point_count(vector_surface_quadrature))
  add_surface_quadrature!(vector_surface_problem,
                          SurfaceQuadrature(1, vector_surface_quadrature, vector_surface_normals))
  add_surface_accumulator!(vector_surface_problem, vector_surface_field, vector_surface_field,
                           _SurfaceAccumulatorFunction((q, trial, test_component) -> begin
                                                         trial_component = component(trial)
                                                         coupling = test_component ==
                                                                    trial_component ? 1.1 : -0.2
                                                         normal_value = normal(q)
                                                         trial_gradient = gradient(trial)
                                                         trial_normal = normal_component(trial_gradient,
                                                                                         normal_value)
                                                         value_coefficient = coupling *
                                                                             value(trial) +
                                                                             0.17 * trial_normal
                                                         gradient_coefficient = _with_axis_increment(_scale_tuple(0.15 *
                                                                                                                  value(trial),
                                                                                                                  normal_value),
                                                                                                     0.02 *
                                                                                                     trial_gradient[test_component],
                                                                                                     trial_component)
                                                         return TestChannels(value_coefficient,
                                                                             gradient_coefficient)
                                                       end))
  vector_surface_plan = compile(vector_surface_problem)
  vector_surface_coefficients = [sin(0.11 * index)
                                 for index in 1:field_dof_count(vector_surface_field)]
  vector_surface_matrix = Grico._assemble_reduced_operator_matrix(vector_surface_plan,
                                                                  Grico._ReducedOperatorWorkspace(vector_surface_plan).scratch)
  @test apply(vector_surface_plan, vector_surface_coefficients) ≈
        vector_surface_matrix * vector_surface_coefficients
  surface_diagnostics = Grico.operator_diagnostics(vector_surface_plan; repetitions=1)
  @test surface_diagnostics.embedded_surfaces == 1
  @test surface_diagnostics.apply_seconds_per_call >= 0
  @test surface_diagnostics.apply_bytes_per_call <= _threaded_allocation_limit(1_000)
  @test surface_diagnostics.reduced_apply_seconds_per_call >= 0
  @test eltype(Grico._diagnostic_sample_vector(Float32, 3)) === Float32

  cg_space = HpSpace(domain, SpaceOptions(degree=UniformDegree(1)))
  cg_field = ScalarField(cg_space; name=:u)
  dirichlet_problem = AffineProblem(cg_field)
  add_cell!(dirichlet_problem, _MatrixFreeIdentity())
  add_constraint!(dirichlet_problem, Dirichlet(cg_field, BoundaryFace(1, LOWER), 2.0))
  dirichlet_state = solve(dirichlet_problem)
  @test coefficients(dirichlet_state) ≈ [2.0, 1.0]
  dirichlet_diagonal_selected, dirichlet_diagonal = _kernel_reduced_diagonal(compile(dirichlet_problem))
  @test dirichlet_diagonal_selected
  @test dirichlet_diagonal ≈ [1.0]
  jacobi_dirichlet_state = solve(dirichlet_problem;
                                 solver=CGSolver(preconditioner=JacobiPreconditioner()))
  @test coefficients(jacobi_dirichlet_state) ≈ [2.0, 1.0]

  constant_space = HpSpace(domain, SpaceOptions(degree=UniformDegree(0), continuity=:dg))
  constant_field = ScalarField(constant_space; name=:u)
  mean_problem = AffineProblem(constant_field)
  add_cell!(mean_problem, _MatrixFreeIdentity())
  add_constraint!(mean_problem, MeanValue(constant_field, 2.0))
  mean_state = solve(mean_problem)
  @test coefficients(mean_state) ≈ [2.0]

  nonlinear_problem = ResidualProblem(constant_field)
  add_cell!(nonlinear_problem, _MatrixFreeQuadraticReaction(constant_field, 4.0))
  nonlinear_plan = compile(nonlinear_problem)
  nonlinear_initial = State(nonlinear_plan, [1.0])
  nonlinear_state = solve(nonlinear_plan; initial_state=nonlinear_initial,
                          relative_tolerance=1.0e-12, absolute_tolerance=1.0e-12)
  @test coefficients(nonlinear_state) ≈ [2.0] atol = 1.0e-10
end

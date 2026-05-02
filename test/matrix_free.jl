using Test
using Grico

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

struct _MatrixFreeMassAction{F}
  field::F
end

struct _MatrixFreeDiffusion{F,T}
  field::F
  coefficient::T
end

function Grico.cell_apply!(local_result, operator::_MatrixFreeMassAction, values,
                           local_coefficients)
  field = operator.field

  for point_index in 1:point_count(values)
    field_value = value(values, local_coefficients, field, point_index)
    weighted = weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_result[row] += shape_value(values, field, point_index, mode_index) * weighted *
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

function Grico.cell_apply!(local_result, operator::_MatrixFreeDiffusion, values,
                           local_coefficients)
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

_tuple_dot(first::Tuple, second::Tuple) = sum(first[index] * second[index] for index in eachindex(first))

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

  mass_problem = AffineProblem(dg_field)
  add_cell!(mass_problem, _MatrixFreeMassAction(dg_field))
  mass_plan = compile(mass_problem)
  @test apply(mass_plan, ones(2)) ≈ [0.5, 0.5]
  @test apply(mass_plan, [0.25, -0.5]) ≈
        _reference_cell_action(mass_plan, [0.25, -0.5],
                               (local_matrix, values) -> _local_reference_mass!(local_matrix,
                                                                                 dg_field,
                                                                                 values))
  mass_diagonal_selected, mass_diagonal = _kernel_reduced_diagonal(mass_plan)
  @test mass_diagonal_selected
  @test mass_diagonal ≈ _reference_reduced_diagonal(mass_plan)

  diffusion_problem = AffineProblem(dg_field)
  add_cell!(diffusion_problem, _MatrixFreeDiffusion(dg_field, 2.0))
  diffusion_plan = compile(diffusion_problem)
  diffusion_coefficients = [0.25, -0.5]
  @test apply(diffusion_plan, diffusion_coefficients) ≈
        _reference_cell_action(diffusion_plan, diffusion_coefficients,
                               (local_matrix, values) -> _local_reference_diffusion!(local_matrix,
                                                                                     dg_field,
                                                                                     values,
                                                                                     2.0))
  diffusion_diagonal_selected, diffusion_diagonal = _kernel_reduced_diagonal(diffusion_plan)
  @test diffusion_diagonal_selected
  @test diffusion_diagonal ≈ _reference_reduced_diagonal(diffusion_plan)

  cg_space = HpSpace(domain, SpaceOptions(degree=UniformDegree(1)))
  cg_field = ScalarField(cg_space; name=:u)
  dirichlet_problem = AffineProblem(cg_field)
  add_cell!(dirichlet_problem, _MatrixFreeIdentity())
  add_constraint!(dirichlet_problem, Dirichlet(cg_field, BoundaryFace(1, LOWER), 2.0))
  dirichlet_state = solve(dirichlet_problem)
  @test coefficients(dirichlet_state) ≈ [2.0, 1.0]
  dirichlet_diagonal_selected, dirichlet_diagonal =
    _kernel_reduced_diagonal(compile(dirichlet_problem))
  @test dirichlet_diagonal_selected
  @test dirichlet_diagonal ≈ [1.0]
  jacobi_dirichlet_state = solve(dirichlet_problem; preconditioner=JacobiPreconditioner())
  @test coefficients(jacobi_dirichlet_state) ≈ [2.0, 1.0]

  constant_space = HpSpace(domain, SpaceOptions(degree=UniformDegree(0), continuity=:dg))
  constant_field = ScalarField(constant_space; name=:u)
  mean_problem = AffineProblem(constant_field)
  add_cell!(mean_problem, _MatrixFreeIdentity())
  add_constraint!(mean_problem, MeanValue(constant_field, 2.0))
  mean_state = solve(mean_problem)
  @test coefficients(mean_state) ≈ [2.0]
end

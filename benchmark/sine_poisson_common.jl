# Shared sine-interface Poisson ingredients used by the CG and DG benchmarks.

import Grico: cell_apply!, cell_diagonal!, cell_rhs!

struct SineInterfacePoissonDiffusion{F}
  field::F
  use_tensor::Bool
end

SineInterfacePoissonDiffusion(field) = SineInterfacePoissonDiffusion(field, true)

struct SineInterfacePoissonSource{F}
  field::F
  use_tensor::Bool
end

SineInterfacePoissonSource(field) = SineInterfacePoissonSource(field, true)

function sine_interface_poisson_field(options; continuity)
  domain = Domain((0.0, 0.0), (1.0, 1.0), options["root_cells"])
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(options["degree"]),
                               quadrature=DegreePlusQuadrature(options["quadrature_extra_points"]),
                               continuity=continuity))
  return ScalarField(space; name=:u)
end

function add_sine_interface_poisson_cells!(problem, field)
  add_sine_interface_poisson_cells!(problem, field, true)
end

function add_sine_interface_poisson_cells!(problem, field, use_tensor::Bool)
  add_cell!(problem, SineInterfacePoissonDiffusion(field, use_tensor))
  add_cell!(problem, SineInterfacePoissonSource(field, use_tensor))
  return problem
end

function _copy_scalar_tensor_input!(result, values, field, local_coefficients)
  range = field_dof_range(values, field)

  @inbounds for mode_index in eachindex(result)
    result[mode_index] = local_coefficients[first(range)+mode_index-1]
  end

  return result
end

function _add_scalar_tensor_output!(local_result, values, field, contribution)
  range = field_dof_range(values, field)

  @inbounds for mode_index in eachindex(contribution)
    local_result[first(range)+mode_index-1] += contribution[mode_index]
  end

  return local_result
end

# Apply the scalar Poisson stiffness block ∫Ω ∇u · ∇v dΩ.
function cell_apply!(local_result, operator::SineInterfacePoissonDiffusion, values::CellValues,
                     local_coefficients)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)
    field_gradient_value = gradient(values, local_coefficients, operator.field, point_index)

    for row_mode in 1:mode_count
      contribution = zero(eltype(local_result))

      for axis in 1:axis_count
        contribution += gradients[axis, row_mode, point_index] * field_gradient_value[axis]
      end

      local_result[local_dof_index(values, operator.field, 1, row_mode)] += contribution * weighted
    end
  end

  return nothing
end

function cell_apply!(local_result, operator::SineInterfacePoissonDiffusion, values::CellValues,
                     local_coefficients, scratch::KernelScratch)
  tensor = tensor_values(values, operator.field)
  if !operator.use_tensor || tensor === nothing || !is_full_tensor(tensor)
    return cell_apply!(local_result, operator, values, local_coefficients)
  end

  return _cell_apply_tensor_diffusion!(local_result, operator, values, local_coefficients, tensor,
                                       scratch)
end

function _cell_apply_tensor_diffusion!(local_result, operator::SineInterfacePoissonDiffusion,
                                       values::CellValues, local_coefficients,
                                       tensor::TensorProductValues{D,T},
                                       scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  gradients = scratch_matrix(scratch, 1, length(tensor_degrees(tensor)), tensor_point_count(tensor))
  input = scratch_vector(scratch, length(tensor_degrees(tensor)) + 2, tensor_mode_count(tensor))
  _copy_scalar_tensor_input!(input, values, operator.field, local_coefficients)
  tensor_gradient!(gradients, tensor, input, scratch)

  @inbounds for point_index in 1:tensor_point_count(tensor)
    weighted = weight(values, point_index)

    for axis in 1:size(gradients, 1)
      gradients[axis, point_index] *= weighted
    end
  end

  contribution = scratch_vector(scratch, length(tensor_degrees(tensor)) + 3,
                                tensor_mode_count(tensor))
  fill!(contribution, zero(eltype(contribution)))
  tensor_project_gradient!(contribution, tensor, gradients, scratch)
  _add_scalar_tensor_output!(local_result, values, operator.field, contribution)
  return nothing
end

function cell_diagonal!(local_diagonal, operator::SineInterfacePoissonDiffusion, values::CellValues)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      contribution = zero(eltype(local_diagonal))

      for axis in 1:axis_count
        gradient_value = gradients[axis, mode_index, point_index]
        contribution += gradient_value * gradient_value
      end

      local_diagonal[local_dof_index(values, operator.field, 1, mode_index)] += contribution *
                                                                                weighted
    end
  end

  return nothing
end

# Assemble ∫Ω f v dΩ with the discontinuous source evaluated at physical
# quadrature points. The jump across the sine interface drives adaptive
# refinement without prescribing whether the planner should use h or p changes.
function cell_rhs!(local_rhs, operator::SineInterfacePoissonSource, values::CellValues)
  local_block = block(local_rhs, values, operator.field)
  shape_table = shape_values(values, operator.field)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = sine_interface_source(point(values, point_index)) * weight(values, point_index)

    for mode_index in 1:mode_count
      local_block[mode_index] += shape_table[mode_index, point_index] * weighted
    end
  end

  return nothing
end

function cell_rhs!(local_rhs, operator::SineInterfacePoissonSource, values::CellValues,
                   scratch::KernelScratch)
  tensor = tensor_values(values, operator.field)
  if !operator.use_tensor || tensor === nothing || !is_full_tensor(tensor)
    return cell_rhs!(local_rhs, operator, values)
  end

  return _cell_rhs_tensor_source!(local_rhs, operator, values, tensor, scratch)
end

function _cell_rhs_tensor_source!(local_rhs, operator::SineInterfacePoissonSource,
                                  values::CellValues, tensor::TensorProductValues{D,T},
                                  scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  weighted_values = scratch_vector(scratch, length(tensor_degrees(tensor)) + 2,
                                   tensor_point_count(tensor))

  @inbounds for point_index in 1:tensor_point_count(tensor)
    weighted_values[point_index] = sine_interface_source(point(values, point_index)) *
                                   weight(values, point_index)
  end

  contribution = scratch_vector(scratch, length(tensor_degrees(tensor)) + 3,
                                tensor_mode_count(tensor))
  fill!(contribution, zero(eltype(contribution)))
  tensor_project!(contribution, tensor, weighted_values, scratch)
  _add_scalar_tensor_output!(local_rhs, values, operator.field, contribution)
  return nothing
end

function sine_interface_source(x)
  interface_height = 0.5 + 0.25 * sinpi(2 * x[1])
  return x[2] < interface_height ? 1.0 : 0.0
end

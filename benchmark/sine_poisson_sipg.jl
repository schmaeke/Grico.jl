# Symmetric interior-penalty DG operator for the sine-interface Poisson case.

import Grico: face_apply!, face_diagonal!, interface_apply!, interface_diagonal!

struct SineInterfaceSIPGPoisson{F,T}
  field::F
  penalty::T
end

function sine_interface_sipg_problem(field, options)
  operator = SineInterfaceSIPGPoisson(field, options["penalty"])
  problem = AffineProblem(field)
  add_sine_interface_poisson_cells!(problem, field, get(options, "tensor_kernels", true))
  add_interface!(problem, operator)

  for axis in 1:2
    add_boundary!(problem, BoundaryFace(axis, LOWER), operator)
    add_boundary!(problem, BoundaryFace(axis, UPPER), operator)
  end

  return problem
end

# The SIPG operator below uses the convention [w] = w⁻ - w⁺ with the interface
# normal pointing from the minus leaf to the plus leaf.
@inline _trace_jump_sign(is_plus::Bool) = is_plus ? -1.0 : 1.0

@inline function _interface_length_scale(field, minus_leaf, plus_leaf, axis)
  domain_data = field_space(field).domain
  h_minus = cell_size(domain_data, minus_leaf, axis)
  h_plus = cell_size(domain_data, plus_leaf, axis)
  return 2.0 * h_minus * h_plus / (h_minus + h_plus)
end

@inline function _interface_penalty_scale(field, minus_leaf, plus_leaf, axis)
  h = _interface_length_scale(field, minus_leaf, plus_leaf, axis)
  degree_value = max(cell_degrees(field_space(field), minus_leaf)[axis],
                     cell_degrees(field_space(field), plus_leaf)[axis])
  return (degree_value + 1)^2 / h
end

@inline function _boundary_penalty_scale(field, leaf, axis)
  h = cell_size(field_space(field).domain, leaf, axis)
  degree_value = cell_degrees(field_space(field), leaf)[axis]
  return (degree_value + 1)^2 / h
end

# Apply the SIPG interior-face contribution
#
#   - {∂ₙu} [v] - {∂ₙv} [u] + σ [u] [v],
#
# where σ = penalty * (p + 1)^2 / h. The action is evaluated directly against
# the local coefficient vector instead of materializing the face matrix.
function interface_apply!(local_result, operator::SineInterfaceSIPGPoisson, values::InterfaceValues,
                          local_coefficients)
  T = eltype(local_result)
  field = operator.field
  minus_values = minus(values)
  plus_values = plus(values)
  minus_modes = local_mode_count(minus_values, field)
  plus_modes = local_mode_count(plus_values, field)
  blocks = (block(local_result, minus_values, field), block(local_result, plus_values, field))
  penalty_scale = T(_interface_penalty_scale(field, values.minus_leaf, values.plus_leaf,
                                             values.axis))::T
  penalty = T(operator.penalty) * penalty_scale

  @inbounds for point_index in 1:point_count(values)
    minus_value = value(minus_values, local_coefficients, field, point_index)
    plus_value = value(plus_values, local_coefficients, field, point_index)
    trial_jump = minus_value - plus_value
    trial_average_flux = 0.5 *
                         (normal_gradient(minus_values, local_coefficients, field, point_index) +
                          normal_gradient(plus_values, local_coefficients, field, point_index))
    weighted = weight(values, point_index)

    for row_side in 1:2
      row_values = row_side == 1 ? minus_values : plus_values
      row_modes = row_side == 1 ? minus_modes : plus_modes
      row_sign = _trace_jump_sign(row_side == 2)
      local_block = blocks[row_side]

      for row_mode in 1:row_modes
        row_shape = shape_value(row_values, field, point_index, row_mode)
        row_normal_gradient = shape_normal_gradient(row_values, field, point_index, row_mode)
        row_jump = row_sign * row_shape
        row_average_flux = 0.5 * row_normal_gradient
        contribution = -row_jump * trial_average_flux - row_average_flux * trial_jump +
                       penalty * row_jump * trial_jump
        local_block[row_mode] += contribution * weighted
      end
    end
  end

  return nothing
end

function interface_diagonal!(local_diagonal, operator::SineInterfaceSIPGPoisson,
                             values::InterfaceValues)
  T = eltype(local_diagonal)
  field = operator.field
  minus_values = minus(values)
  plus_values = plus(values)
  blocks = (block(local_diagonal, minus_values, field), block(local_diagonal, plus_values, field))
  penalty_scale = T(_interface_penalty_scale(field, values.minus_leaf, values.plus_leaf,
                                             values.axis))::T
  penalty = T(operator.penalty) * penalty_scale

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for side in 1:2
      side_values = side == 1 ? minus_values : plus_values
      sign = _trace_jump_sign(side == 2)
      local_block = blocks[side]

      for mode_index in 1:local_mode_count(side_values, field)
        shape = shape_value(side_values, field, point_index, mode_index)
        flux = shape_normal_gradient(side_values, field, point_index, mode_index)
        local_block[mode_index] += (penalty * shape * shape - sign * shape * flux) * weighted
      end
    end
  end

  return nothing
end

# Homogeneous Dirichlet data are imposed weakly by the boundary analogue of the
# SIPG form. Since g = 0, only the operator action remains.
function face_apply!(local_result, operator::SineInterfaceSIPGPoisson, values::FaceValues,
                     local_coefficients)
  T = eltype(local_result)
  field = operator.field
  local_block = block(local_result, values, field)
  mode_count = local_mode_count(values, field)
  penalty_scale = T(_boundary_penalty_scale(field, values.leaf, values.axis))::T
  penalty = T(operator.penalty) * penalty_scale

  @inbounds for point_index in 1:point_count(values)
    trial_value = value(values, local_coefficients, field, point_index)
    trial_normal_gradient = normal_gradient(values, local_coefficients, field, point_index)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      row_shape = shape_value(values, field, point_index, row_mode)
      row_normal_gradient = shape_normal_gradient(values, field, point_index, row_mode)
      contribution = -row_shape * trial_normal_gradient - row_normal_gradient * trial_value +
                     penalty * row_shape * trial_value
      local_block[row_mode] += contribution * weighted
    end
  end

  return nothing
end

function face_diagonal!(local_diagonal, operator::SineInterfaceSIPGPoisson, values::FaceValues)
  T = eltype(local_diagonal)
  field = operator.field
  local_block = block(local_diagonal, values, field)
  mode_count = local_mode_count(values, field)
  penalty_scale = T(_boundary_penalty_scale(field, values.leaf, values.axis))::T
  penalty = T(operator.penalty) * penalty_scale

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      shape = shape_value(values, field, point_index, mode_index)
      flux = shape_normal_gradient(values, field, point_index, mode_index)
      local_block[mode_index] += (penalty * shape * shape - 2 * shape * flux) * weighted
    end
  end

  return nothing
end

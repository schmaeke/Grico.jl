# Symmetric interior-penalty DG operator for the sine-interface Poisson case.

import Grico: face_matrix!, interface_matrix!

struct SineInterfaceSIPGPoisson{F,T}
  field::F
  penalty::T
end

function sine_interface_sipg_problem(field, options)
  operator = SineInterfaceSIPGPoisson(field, options["penalty"])
  problem = AffineProblem(field)
  add_sine_interface_poisson_cells!(problem, field)
  add_interface!(problem, operator)

  for axis in 1:2
    add_boundary!(problem, BoundaryFace(axis, LOWER), operator)
    add_boundary!(problem, BoundaryFace(axis, UPPER), operator)
  end

  return problem
end

# The weak form below uses the convention [w] = w⁻ - w⁺ with the interface
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

# Assemble the SIPG interior-face contribution
#
#   - {∂ₙu} [v] - {∂ₙv} [u] + σ [u] [v],
#
# where σ = penalty * (p + 1)^2 / h. The side loops keep matching and hanging
# interfaces in one path because the plus and minus traces may have different
# local mode counts.
function interface_matrix!(local_matrix, operator::SineInterfaceSIPGPoisson,
                           values::InterfaceValues)
  field = operator.field
  minus_values = minus(values)
  plus_values = plus(values)
  minus_modes = local_mode_count(minus_values, field)
  plus_modes = local_mode_count(plus_values, field)
  blocks = ((block(local_matrix, minus_values, field, minus_values, field),
             block(local_matrix, minus_values, field, plus_values, field)),
            (block(local_matrix, plus_values, field, minus_values, field),
             block(local_matrix, plus_values, field, plus_values, field)))
  penalty = operator.penalty *
            _interface_penalty_scale(field, values.minus_leaf, values.plus_leaf, values.axis)

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_side in 1:2
      row_values = row_side == 1 ? minus_values : plus_values
      row_modes = row_side == 1 ? minus_modes : plus_modes
      row_sign = _trace_jump_sign(row_side == 2)

      for row_mode in 1:row_modes
        row_shape = shape_value(row_values, field, point_index, row_mode)
        row_normal_gradient = shape_normal_gradient(row_values, field, point_index, row_mode)
        row_jump = row_sign * row_shape
        row_average_flux = 0.5 * row_normal_gradient

        for col_side in 1:2
          col_values = col_side == 1 ? minus_values : plus_values
          col_modes = col_side == 1 ? minus_modes : plus_modes
          col_sign = _trace_jump_sign(col_side == 2)
          local_block = blocks[row_side][col_side]

          for col_mode in 1:col_modes
            col_shape = shape_value(col_values, field, point_index, col_mode)
            col_normal_gradient = shape_normal_gradient(col_values, field, point_index, col_mode)
            col_jump = col_sign * col_shape
            col_average_flux = 0.5 * col_normal_gradient
            contribution = -row_jump * col_average_flux - row_average_flux * col_jump +
                           penalty * row_jump * col_jump
            local_block[row_mode, col_mode] += contribution * weighted
          end
        end
      end
    end
  end

  return nothing
end

# Homogeneous Dirichlet data are imposed weakly by the boundary analogue of the
# SIPG form. Since g = 0, only the matrix contribution remains.
function face_matrix!(local_matrix, operator::SineInterfaceSIPGPoisson, values::FaceValues)
  field = operator.field
  local_block = block(local_matrix, values, field, field)
  mode_count = local_mode_count(values, field)
  penalty = operator.penalty * _boundary_penalty_scale(field, values.leaf, values.axis)

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      row_shape = shape_value(values, field, point_index, row_mode)
      row_normal_gradient = shape_normal_gradient(values, field, point_index, row_mode)

      for col_mode in 1:mode_count
        col_shape = shape_value(values, field, point_index, col_mode)
        col_normal_gradient = shape_normal_gradient(values, field, point_index, col_mode)
        contribution = -row_shape * col_normal_gradient - row_normal_gradient * col_shape +
                       penalty * row_shape * col_shape
        local_block[row_mode, col_mode] += contribution * weighted
      end
    end
  end

  return nothing
end

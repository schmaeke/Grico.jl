# ---------------------------------------------------------------------------
# 1. Local weak-form building blocks
# ---------------------------------------------------------------------------
#
# Grico examples define variational forms through small local operators. For
# Poisson we need the usual diffusion bilinear form and the volume load term.

# Standard diffusion bilinear form
#
#   a(v, u) = ∫_Ω ∇v · ∇u dΩ.
#
# The matrix-free action evaluates the diffusion bilinear form directly against
# the local coefficient vector.
struct Diffusion{F}
  field::F
end

function cell_apply!(local_result, operator::Diffusion, values::CellValues, local_coefficients)
  local_block = block(local_result, values, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    trial_gradient = gradient(values, local_coefficients, operator.field, point_index)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      contribution = zero(eltype(local_result))

      for axis in 1:axis_count
        contribution += gradients[axis, row_mode, point_index] * trial_gradient[axis]
      end

      local_block[row_mode] += contribution * weighted
    end
  end

  return nothing
end

function cell_diagonal!(local_diagonal, operator::Diffusion, values::CellValues)
  local_block = block(local_diagonal, values, operator.field)
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

      local_block[mode_index] += contribution * weighted
    end
  end

  return nothing
end

# Load functional
#
#   ℓ(v) = ∫_Ω f v dΩ.
#
# The structure is the same: loop over quadrature points, evaluate the physical
# coefficient data there, and accumulate into the local vector with the test
# basis values.
struct Source{F,G}
  field::F
  data::G
end

function cell_rhs!(local_rhs, operator::Source, values::CellValues)
  local_block = block(local_rhs, values, operator.field)
  shape_table = shape_values(values, operator.field)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = operator.data(point(values, point_index)) * weight(values, point_index)

    for mode_index in 1:mode_count
      local_block[mode_index] += shape_table[mode_index, point_index] * weighted
    end
  end

  return nothing
end

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
# The local matrix is symmetric, so the implementation assembles only the lower
# triangle and mirrors it.
struct Diffusion{F}
  field::F
end

function cell_matrix!(local_matrix, operator::Diffusion, values::CellValues)
  local_block = block(local_matrix, values, operator.field, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      for col_mode in 1:row_mode
        contribution = zero(eltype(local_matrix))

        for axis in 1:axis_count
          contribution += gradients[axis, row_mode, point_index] *
                          gradients[axis, col_mode, point_index]
        end

        contribution *= weighted
        local_block[row_mode, col_mode] += contribution
        row_mode == col_mode || (local_block[col_mode, row_mode] += contribution)
      end
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

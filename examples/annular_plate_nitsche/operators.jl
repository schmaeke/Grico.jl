# ---------------------------------------------------------------------------
# 2. Local weak forms
# ---------------------------------------------------------------------------
#
# The interior operator is the standard Laplace bilinear form. The boundary
# condition is not applied by a fitted boundary trace constraint because the
# boundary is curved and cuts through Cartesian cells. Instead, we add a
# separate surface operator later.

# Standard Laplace bilinear form
#
#   a(v, u) = ∫_Ω ∇v · ∇u dΩ.
#
# The field is scalar, so the local matrix is assembled over one scalar block.
struct Diffusion{F}
  field::F
end

function cell_matrix!(local_matrix, operator::Diffusion, values::CellValues)
  local_block = block(local_matrix, values, operator.field, operator.field)
  mode_count = local_mode_count(values, operator.field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      gradient_row = shape_gradient(values, operator.field, point_index, row_mode)

      for col_mode in 1:mode_count
        gradient_col = shape_gradient(values, operator.field, point_index, col_mode)
        local_block[row_mode, col_mode] += weighted * sum(gradient_row[axis] * gradient_col[axis]
                                                          for axis in eachindex(gradient_row))
      end
    end
  end

  return nothing
end

# Symmetric Nitsche boundary operator for a prescribed Dirichlet datum g. The
# embedded surface contributes
#
#   ∫_Γ (-v ∂ₙu - u ∂ₙv + η h⁻¹ u v) dΓ
#
# to the matrix and
#
#   ∫_Γ (-g ∂ₙv + η h⁻¹ g v) dΓ
#
# to the right-hand side.
#
# The consistency terms recover the weak form of the Dirichlet problem, the
# symmetry term mirrors the first consistency term, and the penalty term restores
# coercivity on unfitted cells.
#
# For a reader new to Nitsche methods, the key point is: the boundary condition
# is enforced weakly by extra integral terms instead of by directly eliminating
# boundary degrees of freedom.
struct NitscheDirichlet{F,G,T}
  field::F
  data::G
  penalty::T
end

function surface_matrix!(local_matrix, operator::NitscheDirichlet, values::SurfaceValues)
  local_block = block(local_matrix, values, operator.field, operator.field)
  mode_count = local_mode_count(values, operator.field)
  domain = field_space(operator.field).domain

  # On the Cartesian background mesh, a simple and robust local length scale is
  # the smaller side length of the leaf that contains the current embedded
  # segment quadrature points. This is the `h` that appears in the Nitsche
  # penalty scaling η / h.
  h = min(cell_size(domain, values.leaf, 1), cell_size(domain, values.leaf, 2))
  penalty = operator.penalty / h

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)
    normal_data = normal(values, point_index)

    for row_mode in 1:mode_count
      value_row = shape_value(values, operator.field, point_index, row_mode)
      gradient_row = shape_gradient(values, operator.field, point_index, row_mode)
      flux_row = sum(gradient_row[axis] * normal_data[axis] for axis in eachindex(normal_data))

      for col_mode in 1:mode_count
        value_col = shape_value(values, operator.field, point_index, col_mode)
        gradient_col = shape_gradient(values, operator.field, point_index, col_mode)
        flux_col = sum(gradient_col[axis] * normal_data[axis] for axis in eachindex(normal_data))
        local_block[row_mode, col_mode] += weighted *
                                           (-value_row * flux_col - value_col * flux_row +
                                            penalty * value_row * value_col)
      end
    end
  end

  return nothing
end

function surface_rhs!(local_rhs, operator::NitscheDirichlet, values::SurfaceValues)
  local_block = block(local_rhs, values, operator.field)
  mode_count = local_mode_count(values, operator.field)
  domain = field_space(operator.field).domain
  h = min(cell_size(domain, values.leaf, 1), cell_size(domain, values.leaf, 2))
  penalty = operator.penalty / h

  for point_index in 1:point_count(values)
    x = point(values, point_index)
    g = operator.data(x)
    weighted = weight(values, point_index)
    normal_data = normal(values, point_index)

    for mode_index in 1:mode_count
      value_i = shape_value(values, operator.field, point_index, mode_index)
      gradient_i = shape_gradient(values, operator.field, point_index, mode_index)
      flux_i = sum(gradient_i[axis] * normal_data[axis] for axis in eachindex(normal_data))
      local_block[mode_index] += weighted * (-g * flux_i + penalty * g * value_i)
    end
  end

  return nothing
end

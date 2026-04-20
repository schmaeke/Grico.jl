# ---------------------------------------------------------------------------
# 2. Mass matrix and initial projection
# ---------------------------------------------------------------------------

# The initial condition is inserted into the DG space via an `L²` projection.
# That means we assemble the element mass matrix
#
#   Mᵢⱼ = ∫_K φᵢ φⱼ dx,
#
# and the projected right-hand side
#
#   bᵢ = ∫_K φᵢ q₀ dx,
#
# then solve `M q_h(0) = b`.
struct MassMatrix{F}
  field::F
end

function _assemble_mass_block!(local_matrix, field, values)
  local_block = block(local_matrix, values, field, field)
  mode_count = local_mode_count(values, field)
  components = component_count(field)
  shape_table = shape_values(values, field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      row_value = shape_table[row_mode, point_index]

      for col_mode in 1:mode_count
        contribution = row_value * shape_table[col_mode, point_index] * weighted
        contribution == 0 && continue

        for component in 1:components
          row = component_local_index(mode_count, component, row_mode)
          col = component_local_index(mode_count, component, col_mode)
          local_block[row, col] += contribution
        end
      end
    end
  end

  return nothing
end

function cell_matrix!(local_matrix, operator::MassMatrix, values::CellValues)
  _assemble_mass_block!(local_matrix, operator.field, values)
end
function cell_tangent!(local_matrix, operator::MassMatrix, values::CellValues, state::State)
  _assemble_mass_block!(local_matrix, operator.field, values)
end

struct ProjectionSource{F,G}
  field::F
  data::G
end

# Assemble the load vector for the `L²` projection of the analytic blast-wave
# state onto the DG basis.
function cell_rhs!(local_rhs, operator::ProjectionSource, values::CellValues)
  local_block = block(local_rhs, values, operator.field)
  mode_count = local_mode_count(values, operator.field)
  shape_table = shape_values(values, operator.field)

  for point_index in 1:point_count(values)
    state_value = operator.data(point(values, point_index))
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      shape = shape_table[mode_index, point_index]
      scale = shape * weighted
      scale == 0 && continue

      for component in 1:component_count(operator.field)
        row = component_local_index(mode_count, component, mode_index)
        local_block[row] += state_value[component] * scale
      end
    end
  end

  return nothing
end

direct_sparse_solve(matrix_data, rhs_data) = matrix_data \ rhs_data

# Build the projected initial DG state `q_h(x, 0)`.
function project_initial_condition(field, data; linear_solve=direct_sparse_solve)
  problem = AffineProblem(field)
  add_cell!(problem, MassMatrix(field))
  add_cell!(problem, ProjectionSource(field, data))
  system = assemble(compile(problem))
  return State(system, Grico.solve(system; linear_solve=linear_solve))
end

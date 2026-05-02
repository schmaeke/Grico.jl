# ---------------------------------------------------------------------------
# 2. Mass matrix and initial projection
# ---------------------------------------------------------------------------

# The initial condition is inserted into the DG space via an `L²` projection.
# That means we apply the element mass operator
#
#   (M q)ᵢ = ∫_K φᵢ q dx,
#
# and the projected right-hand side
#
#   bᵢ = ∫_K φᵢ q₀ dx,
#
# then solve `M q_h(0) = b`.
struct MassMatrix{F}
  field::F
end

function _apply_mass_block!(local_result, field, values, local_coefficients)
  local_block = block(local_result, values, field)
  mode_count = local_mode_count(values, field)
  components = component_count(field)
  shape_table = shape_values(values, field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for component in 1:components
      trial_value = zero(eltype(local_result))

      for col_mode in 1:mode_count
        col = component_local_index(mode_count, component, col_mode)
        trial_value += shape_table[col_mode, point_index] * local_coefficients[col]
      end

      trial_value == 0 && continue

      for row_mode in 1:mode_count
        row = component_local_index(mode_count, component, row_mode)
        local_block[row] += shape_table[row_mode, point_index] * trial_value * weighted
      end
    end
  end

  return nothing
end

function _copy_component_tensor_input!(result, values, field, component, local_coefficients)
  range = field_dof_range(values, field)
  mode_count = length(result)

  @inbounds for mode_index in 1:mode_count
    local_row = component_local_index(mode_count, component, mode_index)
    result[mode_index] = local_coefficients[first(range)+local_row-1]
  end

  return result
end

function _add_component_tensor_output!(local_result, values, field, component, contribution)
  range = field_dof_range(values, field)
  mode_count = length(contribution)

  @inbounds for mode_index in 1:mode_count
    local_row = component_local_index(mode_count, component, mode_index)
    local_result[first(range)+local_row-1] += contribution[mode_index]
  end

  return local_result
end

function cell_apply!(local_result, operator::MassMatrix, values::CellValues, local_coefficients)
  _apply_mass_block!(local_result, operator.field, values, local_coefficients)
end

function cell_apply!(local_result, operator::MassMatrix, values::CellValues, local_coefficients,
                     scratch::KernelScratch)
  tensor = tensor_values(values, operator.field)
  if tensor === nothing || !is_full_tensor(tensor)
    return cell_apply!(local_result, operator, values, local_coefficients)
  end

  mode_count = tensor_mode_count(tensor)
  point_count_value = tensor_point_count(tensor)
  input = scratch_vector(scratch, 4, mode_count)
  point_values = scratch_vector(scratch, 5, point_count_value)
  contribution = scratch_vector(scratch, 6, mode_count)

  for component in 1:component_count(operator.field)
    _copy_component_tensor_input!(input, values, operator.field, component, local_coefficients)
    tensor_interpolate!(point_values, tensor, input, scratch)

    @inbounds for point_index in 1:point_count_value
      point_values[point_index] *= weight(values, point_index)
    end

    fill!(contribution, zero(eltype(contribution)))
    tensor_project!(contribution, tensor, point_values, scratch)
    _add_component_tensor_output!(local_result, values, operator.field, component, contribution)
  end

  return nothing
end

function cell_diagonal!(local_diagonal, operator::MassMatrix, values::CellValues)
  local_block = block(local_diagonal, values, operator.field)
  mode_count = local_mode_count(values, operator.field)
  components = component_count(operator.field)
  shape_table = shape_values(values, operator.field)

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      shape = shape_table[mode_index, point_index]
      contribution = shape * shape * weighted

      for component in 1:components
        row = component_local_index(mode_count, component, mode_index)
        local_block[row] += contribution
      end
    end
  end

  return nothing
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

function cell_rhs!(local_rhs, operator::ProjectionSource, values::CellValues,
                   scratch::KernelScratch)
  tensor = tensor_values(values, operator.field)
  if tensor === nothing || !is_full_tensor(tensor)
    return cell_rhs!(local_rhs, operator, values)
  end

  mode_count = tensor_mode_count(tensor)
  point_count_value = tensor_point_count(tensor)
  weighted_values = scratch_vector(scratch, 4, point_count_value)
  contribution = scratch_vector(scratch, 5, mode_count)

  for component in 1:component_count(operator.field)
    @inbounds for point_index in 1:point_count_value
      weighted_values[point_index] = operator.data(point(values, point_index))[component] *
                                     weight(values, point_index)
    end

    fill!(contribution, zero(eltype(contribution)))
    tensor_project!(contribution, tensor, weighted_values, scratch)
    _add_component_tensor_output!(local_rhs, values, operator.field, component, contribution)
  end

  return nothing
end

# Build the projected initial DG state `q_h(x, 0)`.
function project_initial_condition(field, data; linear_solve=Grico.default_linear_solve)
  problem = AffineProblem(field)
  add_cell!(problem, MassMatrix(field))
  add_cell!(problem, ProjectionSource(field, data))
  plan = compile(problem)
  return solve(plan; linear_solve=linear_solve, preconditioner=JacobiPreconditioner())
end

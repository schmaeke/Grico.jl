# ---------------------------------------------------------------------------
# 3. One steady Oseen step
# ---------------------------------------------------------------------------
#
# Picard iteration for the steady Navier-Stokes equations freezes the advecting
# velocity at the previous iterate. The resulting linear problem is an Oseen
# problem.
#
# In this example the local operator collects all pieces of that linearized DG
# formulation:
#
# - cell terms for viscosity, convection, incompressibility, and grad-div,
# - interface terms for SIPG diffusion, upwind convection, and mixed coupling,
# - boundary terms for weak wall data.
#
# One steady Oseen step with frozen advecting velocity `w`.
#
# The linearized momentum equation is
#
#   ∇ · (w ⊗ u) - νΔu + ∇p = 0,
#
# while incompressibility is enforced in mixed form together with a mild broken
# grad-div stabilization and an optional pressure-jump stabilization.
#
# The operator is mutable so the compiled plan can be reused across Picard
# iterations: only the lagged advecting state changes, not the integration
# tables, sparsity pattern, or boundary/interface topology.
mutable struct SteadyDGOseen{U,P,S,T,G}
  velocity::U
  pressure::P
  advecting_state::S
  viscosity::T
  velocity_penalty::T
  pressure_jump_penalty::T
  normal_flux_penalty::T
  divergence_penalty::T
  boundary_data::G
end

function cell_matrix!(local_matrix, operator::SteadyDGOseen{U,P,S,T},
                      values::CellValues) where {U,P,S,T}
  velocity = operator.velocity
  pressure = operator.pressure
  velocity_block = block(local_matrix, values, velocity, velocity)
  velocity_pressure = block(local_matrix, values, velocity, pressure)
  pressure_velocity = block(local_matrix, values, pressure, velocity)
  velocity_modes = local_mode_count(values, velocity)
  pressure_modes = local_mode_count(values, pressure)

  for point_index in 1:point_count(values)
    advecting = value(values, operator.advecting_state, velocity, point_index)
    advecting_gradient = gradient(values, operator.advecting_state, velocity, point_index)
    advecting_divergence = advecting_gradient[1][1] + advecting_gradient[2][2]
    weighted = weight(values, point_index)

    for row_component in 1:2
      for row_mode in 1:velocity_modes
        row = velocity_index(velocity_modes, row_component, row_mode)
        row_shape = shape_value(values, velocity, point_index, row_mode)
        row_gradient = shape_gradient(values, velocity, point_index, row_mode)
        row_divergence = row_gradient[row_component]

        for col_component in 1:2
          for col_mode in 1:velocity_modes
            col = velocity_index(velocity_modes, col_component, col_mode)
            col_shape = shape_value(values, velocity, point_index, col_mode)
            col_gradient = shape_gradient(values, velocity, point_index, col_mode)
            contribution = operator.divergence_penalty *
                           row_divergence *
                           col_gradient[col_component]

            if row_component == col_component
              contribution += operator.viscosity * dot2(row_gradient, col_gradient)
              contribution -= col_shape * dot2(advecting, row_gradient)
              contribution -= advecting_divergence * row_shape * col_shape
            end

            velocity_block[row, col] += contribution * weighted
          end
        end

        for col_mode in 1:pressure_modes
          velocity_pressure[row, col_mode] -= row_divergence *
                                              shape_value(values, pressure, point_index, col_mode) *
                                              weighted
        end
      end
    end

    for row_mode in 1:pressure_modes
      row_shape = shape_value(values, pressure, point_index, row_mode)

      for col_component in 1:2
        for col_mode in 1:velocity_modes
          col = velocity_index(velocity_modes, col_component, col_mode)
          pressure_velocity[row_mode, col] += row_shape *
                                              shape_gradient(values, velocity, point_index,
                                                             col_mode)[col_component] *
                                              weighted
        end
      end
    end
  end

  return nothing
end

function interface_matrix!(local_matrix, operator::SteadyDGOseen, values::InterfaceValues)
  # This is the interior-face DG coupling. Conceptually it contains four parts:
  #
  # 1. symmetric interior-penalty diffusion for velocity,
  # 2. upwind transport for the frozen convective term,
  # 3. velocity-pressure coupling across the face,
  # 4. optional pressure-jump stabilization.
  #
  # The nested side/field loops keep minus and plus traces in one uniform code
  # path, which is verbose but avoids duplicating nearly identical formulas.
  velocity = operator.velocity
  pressure = operator.pressure
  minus_values = minus(values)
  plus_values = plus(values)
  minus_velocity_modes = local_mode_count(minus_values, velocity)
  plus_velocity_modes = local_mode_count(plus_values, velocity)
  minus_pressure_modes = local_mode_count(minus_values, pressure)
  plus_pressure_modes = local_mode_count(plus_values, pressure)

  velocity_velocity = ((block(local_matrix, minus_values, velocity, minus_values, velocity),
                        block(local_matrix, minus_values, velocity, plus_values, velocity)),
                       (block(local_matrix, plus_values, velocity, minus_values, velocity),
                        block(local_matrix, plus_values, velocity, plus_values, velocity)))
  velocity_pressure = ((block(local_matrix, minus_values, velocity, minus_values, pressure),
                        block(local_matrix, minus_values, velocity, plus_values, pressure)),
                       (block(local_matrix, plus_values, velocity, minus_values, pressure),
                        block(local_matrix, plus_values, velocity, plus_values, pressure)))
  pressure_velocity = ((block(local_matrix, minus_values, pressure, minus_values, velocity),
                        block(local_matrix, minus_values, pressure, plus_values, velocity)),
                       (block(local_matrix, plus_values, pressure, minus_values, velocity),
                        block(local_matrix, plus_values, pressure, plus_values, velocity)))
  pressure_pressure = ((block(local_matrix, minus_values, pressure, minus_values, pressure),
                        block(local_matrix, minus_values, pressure, plus_values, pressure)),
                       (block(local_matrix, plus_values, pressure, minus_values, pressure),
                        block(local_matrix, plus_values, pressure, plus_values, pressure)))

  penalty_scale = interface_penalty_scale(velocity, values.minus_leaf, values.plus_leaf,
                                          values.axis)
  face_size = interface_length_scale(velocity, values.minus_leaf, values.plus_leaf, values.axis)
  viscous_penalty = operator.velocity_penalty * operator.viscosity * penalty_scale
  normal_penalty = operator.normal_flux_penalty * penalty_scale
  pressure_penalty = operator.pressure_jump_penalty *
                     face_size *
                     (max(cell_degrees(field_space(pressure), values.minus_leaf)[values.axis],
                          cell_degrees(field_space(pressure), values.plus_leaf)[values.axis]) + 1)^2

  for point_index in 1:point_count(values)
    normal_data = normal(values)
    minus_advecting = value(minus_values, operator.advecting_state, velocity, point_index)
    plus_advecting = value(plus_values, operator.advecting_state, velocity, point_index)
    normal_speed = dot2(average(minus_advecting, plus_advecting), normal_data)
    weighted = weight(values, point_index)

    for row_side_index in 1:2
      row_values = row_side_index == 1 ? minus_values : plus_values
      row_velocity_modes = row_side_index == 1 ? minus_velocity_modes : plus_velocity_modes
      row_pressure_modes = row_side_index == 1 ? minus_pressure_modes : plus_pressure_modes
      row_sign = trace_jump_sign(row_side_index == 2)

      for row_component in 1:2
        for row_mode in 1:row_velocity_modes
          row_index = velocity_index(row_velocity_modes, row_component, row_mode)
          row_shape = shape_value(row_values, velocity, point_index, row_mode)
          row_normal_gradient = shape_normal_gradient(row_values, velocity, point_index, row_mode)
          row_normal_jump = row_sign * normal_data[row_component] * row_shape

          for col_side_index in 1:2
            col_values = col_side_index == 1 ? minus_values : plus_values
            col_velocity_modes = col_side_index == 1 ? minus_velocity_modes : plus_velocity_modes
            col_sign = trace_jump_sign(col_side_index == 2)
            velocity_block = velocity_velocity[row_side_index][col_side_index]
            pressure_block = velocity_pressure[row_side_index][col_side_index]

            for col_component in 1:2
              for col_mode in 1:col_velocity_modes
                col_index = velocity_index(col_velocity_modes, col_component, col_mode)
                col_shape = shape_value(col_values, velocity, point_index, col_mode)
                col_normal_gradient = shape_normal_gradient(col_values, velocity, point_index,
                                                            col_mode)
                contribution = normal_penalty *
                               row_normal_jump *
                               (col_sign * normal_data[col_component] * col_shape)

                if row_component == col_component
                  contribution -= 0.5 *
                                  operator.viscosity *
                                  row_sign *
                                  row_shape *
                                  col_normal_gradient
                  contribution -= 0.5 *
                                  operator.viscosity *
                                  row_normal_gradient *
                                  col_sign *
                                  col_shape
                  contribution += viscous_penalty * (row_sign * row_shape) * (col_sign * col_shape)

                  if (normal_speed >= 0.0 && col_side_index == 1) ||
                     (normal_speed < 0.0 && col_side_index == 2)
                    contribution += normal_speed * row_sign * row_shape * col_shape
                  end
                end

                velocity_block[row_index, col_index] += contribution * weighted
              end
            end

            col_pressure_modes = col_side_index == 1 ? minus_pressure_modes : plus_pressure_modes

            for col_mode in 1:col_pressure_modes
              col_shape = shape_value(col_values, pressure, point_index, col_mode)
              pressure_block[row_index, col_mode] += 0.5 * row_normal_jump * col_shape * weighted
            end
          end
        end
      end

      for row_mode in 1:row_pressure_modes
        row_shape = shape_value(row_values, pressure, point_index, row_mode)
        row_jump = row_sign * row_shape

        for col_side_index in 1:2
          col_values = col_side_index == 1 ? minus_values : plus_values
          col_velocity_modes = col_side_index == 1 ? minus_velocity_modes : plus_velocity_modes
          col_pressure_modes = col_side_index == 1 ? minus_pressure_modes : plus_pressure_modes
          col_sign = trace_jump_sign(col_side_index == 2)
          velocity_block = pressure_velocity[row_side_index][col_side_index]
          pressure_block = pressure_pressure[row_side_index][col_side_index]

          for col_component in 1:2
            for col_mode in 1:col_velocity_modes
              col_index = velocity_index(col_velocity_modes, col_component, col_mode)
              col_shape = shape_value(col_values, velocity, point_index, col_mode)
              velocity_block[row_mode, col_index] -= 0.5 *
                                                     row_shape *
                                                     (col_sign *
                                                      normal_data[col_component] *
                                                      col_shape) *
                                                     weighted
            end
          end

          for col_mode in 1:col_pressure_modes
            col_shape = shape_value(col_values, pressure, point_index, col_mode)
            pressure_block[row_mode, col_mode] += pressure_penalty *
                                                  row_jump *
                                                  (col_sign * col_shape) *
                                                  weighted
          end
        end
      end
    end
  end

  return nothing
end

function face_matrix!(local_matrix, operator::SteadyDGOseen, values::FaceValues)
  # Boundary faces use the same DG philosophy as interfaces, except that the
  # "exterior" trace is replaced by prescribed wall data. The matrix therefore
  # contains the terms multiplying the unknown interior trace.
  velocity = operator.velocity
  pressure = operator.pressure
  velocity_block = block(local_matrix, values, velocity, velocity)
  velocity_pressure = block(local_matrix, values, velocity, pressure)
  pressure_velocity = block(local_matrix, values, pressure, velocity)
  velocity_modes = local_mode_count(values, velocity)
  pressure_modes = local_mode_count(values, pressure)
  penalty_scale = boundary_penalty_scale(velocity, values.leaf, values.axis)
  viscous_penalty = operator.velocity_penalty * operator.viscosity * penalty_scale
  normal_penalty = operator.normal_flux_penalty * penalty_scale

  for point_index in 1:point_count(values)
    normal_data = normal(values)
    advecting = value(values, operator.advecting_state, velocity, point_index)
    normal_speed = dot2(advecting, normal_data)
    weighted = weight(values, point_index)

    for row_component in 1:2
      for row_mode in 1:velocity_modes
        row_index = velocity_index(velocity_modes, row_component, row_mode)
        row_shape = shape_value(values, velocity, point_index, row_mode)
        row_normal_gradient = shape_normal_gradient(values, velocity, point_index, row_mode)
        row_normal_trace = normal_data[row_component] * row_shape

        for col_component in 1:2
          for col_mode in 1:velocity_modes
            col_index = velocity_index(velocity_modes, col_component, col_mode)
            col_shape = shape_value(values, velocity, point_index, col_mode)
            col_normal_gradient = shape_normal_gradient(values, velocity, point_index, col_mode)
            contribution = normal_penalty *
                           row_normal_trace *
                           normal_data[col_component] *
                           col_shape

            if row_component == col_component
              contribution -= operator.viscosity * row_shape * col_normal_gradient
              contribution -= operator.viscosity * row_normal_gradient * col_shape
              contribution += viscous_penalty * row_shape * col_shape
              normal_speed >= 0.0 && (contribution += normal_speed * row_shape * col_shape)
            end

            velocity_block[row_index, col_index] += contribution * weighted
          end
        end

        for col_mode in 1:pressure_modes
          velocity_pressure[row_index, col_mode] += row_normal_trace *
                                                    shape_value(values, pressure, point_index,
                                                                col_mode) *
                                                    weighted
        end
      end
    end

    for row_mode in 1:pressure_modes
      row_shape = shape_value(values, pressure, point_index, row_mode)

      for col_component in 1:2
        for col_mode in 1:velocity_modes
          col_index = velocity_index(velocity_modes, col_component, col_mode)
          pressure_velocity[row_mode, col_index] -= row_shape *
                                                    normal_data[col_component] *
                                                    shape_value(values, velocity, point_index,
                                                                col_mode) *
                                                    weighted
        end
      end
    end
  end

  return nothing
end

function face_rhs!(local_rhs, operator::SteadyDGOseen, values::FaceValues)
  # The right-hand side contains the pieces involving the prescribed boundary
  # velocity `g`. This is the weak analogue of saying "set u = g on the wall".
  velocity = operator.velocity
  pressure = operator.pressure
  velocity_block = block(local_rhs, values, velocity)
  pressure_block = block(local_rhs, values, pressure)
  velocity_modes = local_mode_count(values, velocity)
  pressure_modes = local_mode_count(values, pressure)
  penalty_scale = boundary_penalty_scale(velocity, values.leaf, values.axis)
  viscous_penalty = operator.velocity_penalty * operator.viscosity * penalty_scale
  normal_penalty = operator.normal_flux_penalty * penalty_scale

  for point_index in 1:point_count(values)
    x = point(values, point_index)
    g = operator.boundary_data(x)
    normal_data = normal(values)
    advecting = value(values, operator.advecting_state, velocity, point_index)
    normal_speed = dot2(advecting, normal_data)
    weighted = weight(values, point_index)

    for row_component in 1:2
      for row_mode in 1:velocity_modes
        row_index = velocity_index(velocity_modes, row_component, row_mode)
        row_shape = shape_value(values, velocity, point_index, row_mode)
        row_normal_gradient = shape_normal_gradient(values, velocity, point_index, row_mode)
        row_normal_trace = normal_data[row_component] * row_shape
        contribution = -operator.viscosity * g[row_component] * row_normal_gradient +
                       viscous_penalty * g[row_component] * row_shape +
                       normal_penalty * dot2(g, normal_data) * row_normal_trace
        normal_speed < 0.0 && (contribution += normal_speed * g[row_component] * row_shape)
        velocity_block[row_index] += contribution * weighted
      end
    end

    for row_mode in 1:pressure_modes
      pressure_block[row_mode] += shape_value(values, pressure, point_index, row_mode) *
                                  dot2(g, normal_data) *
                                  weighted
    end
  end

  return nothing
end

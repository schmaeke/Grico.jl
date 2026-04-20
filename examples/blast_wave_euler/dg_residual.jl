# ---------------------------------------------------------------------------
# 3. Semidiscrete Euler residual
# ---------------------------------------------------------------------------

# After projection, the example advances the semidiscrete DG system
#
#   M dq_h/dt = R(q_h),
#
# where `R(q_h)` contains cell integrals of `∇φ · F(q_h)` and face integrals
# of the numerical flux `F̂`.
struct CompressibleEulerVolume{F,T}
  field::F
  gamma::T
end

struct CompressibleEulerInterface{F,T}
  field::F
  gamma::T
end

struct ReflectiveEulerWall{F,T}
  field::F
  gamma::T
end

# Cell term:
#
#   ∫_K ∇φ · F(q_h) dx.
#
# The sign convention here matches the rest of Grico's residual assembly, so
# the face terms below enter with their corresponding boundary/interface signs.
function cell_residual!(local_rhs, operator::CompressibleEulerVolume, values::CellValues,
                        state::State)
  local_block = block(local_rhs, values, operator.field)
  mode_count = local_mode_count(values, operator.field)

  for point_index in 1:point_count(values)
    q = value(values, state, operator.field, point_index)
    flux = euler_flux(q, operator.gamma)
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      gradient_data = shape_gradient(values, operator.field, point_index, mode_index)

      for component in 1:component_count(operator.field)
        row = component_local_index(mode_count, component, mode_index)
        local_block[row] += dot2(flux[component], gradient_data) * weighted
      end
    end
  end

  return nothing
end

# Interior-face term:
#
#   -∫_f φ⁻ F̂(q⁻, q⁺, n) ds   on the minus side,
#   +∫_f φ⁺ F̂(q⁻, q⁺, n) ds   on the plus side.
function interface_residual!(local_rhs, operator::CompressibleEulerInterface,
                             values::InterfaceValues, state::State)
  minus_values = minus(values)
  plus_values = plus(values)
  minus_block = block(local_rhs, minus_values, operator.field)
  plus_block = block(local_rhs, plus_values, operator.field)
  minus_modes = local_mode_count(minus_values, operator.field)
  plus_modes = local_mode_count(plus_values, operator.field)
  normal_data = normal(values)

  for point_index in 1:point_count(values)
    q_minus = value(minus_values, state, operator.field, point_index)
    q_plus = value(plus_values, state, operator.field, point_index)
    numerical_flux = lax_friedrichs_flux(q_minus, q_plus, normal_data, operator.gamma)
    weighted = weight(values, point_index)

    for mode_index in 1:minus_modes
      shape = shape_value(minus_values, operator.field, point_index, mode_index)
      scale = shape * weighted

      for component in 1:component_count(operator.field)
        row = component_local_index(minus_modes, component, mode_index)
        minus_block[row] -= numerical_flux[component] * scale
      end
    end

    for mode_index in 1:plus_modes
      shape = shape_value(plus_values, operator.field, point_index, mode_index)
      scale = shape * weighted

      for component in 1:component_count(operator.field)
        row = component_local_index(plus_modes, component, mode_index)
        plus_block[row] += numerical_flux[component] * scale
      end
    end
  end

  return nothing
end

# Boundary-face term. On every wall we build a mirrored exterior state and feed
# it into the same Rusanov flux as the interior interfaces. On `x = 0` and
# `y = 0`, this is the symmetry condition; on `x = 1` and `y = 1`, it is a
# reflective slip wall.
function face_residual!(local_rhs, operator::ReflectiveEulerWall, values::FaceValues, state::State)
  local_block = block(local_rhs, values, operator.field)
  mode_count = local_mode_count(values, operator.field)
  normal_data = normal(values)

  for point_index in 1:point_count(values)
    q_interior = value(values, state, operator.field, point_index)
    q_exterior = reflective_ghost_state(q_interior, normal_data)
    numerical_flux = lax_friedrichs_flux(q_interior, q_exterior, normal_data, operator.gamma)
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      shape = shape_value(values, operator.field, point_index, mode_index)
      scale = shape * weighted

      for component in 1:component_count(operator.field)
        row = component_local_index(mode_count, component, mode_index)
        local_block[row] -= numerical_flux[component] * scale
      end
    end
  end

  return nothing
end

# Compile one reusable residual plan that contains the cell, interior-face, and
# wall contributions of the blast-wave problem.
function euler_residual_plan(field, gamma)
  problem = ResidualProblem(field)
  add_cell!(problem, CompressibleEulerVolume(field, gamma))
  add_interface!(problem, CompressibleEulerInterface(field, gamma))

  for axis in 1:2, side in (LOWER, UPPER)
    add_boundary!(problem, BoundaryFace(axis, side), ReflectiveEulerWall(field, gamma))
  end

  return compile(problem)
end

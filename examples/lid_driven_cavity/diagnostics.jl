# ---------------------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------------------
#
# The cavity example prints a few numbers after each Picard step. They are not
# all rigorous error estimators; they are quick health checks that help a reader
# see whether the nonlinear iteration is settling down and whether the DG mixed
# solution remains reasonably divergence-free.

# A few compact diagnostics are enough to make the Picard iteration informative:
# the kinetic energy of the current cavity vortex and a DG incompressibility
# monitor combining cellwise divergence with normal-velocity jumps.
#
# This monitor is an absolute DG quantity, not a mesh-independent physical error
# norm. Its main purpose here is to show whether the Picard sequence settles
# onto a stable mixed solution.
function kinetic_energy(plan, state, velocity)
  total = 0.0

  for cell in plan.integration.cells
    for point_index in 1:point_count(cell)
      velocity_value = value(cell, state, velocity, point_index)
      total += 0.5 * squared_norm2(velocity_value) * weight(cell, point_index)
    end
  end

  return total
end

function broken_divergence_l2(plan, state, velocity)
  total = 0.0

  for cell in plan.integration.cells
    for point_index in 1:point_count(cell)
      gradients = gradient(cell, state, velocity, point_index)
      divergence = gradients[1][1] + gradients[2][2]
      total += divergence^2 * weight(cell, point_index)
    end
  end

  return sqrt(total)
end

function normal_velocity_jump_l2(plan, state, velocity)
  total = 0.0

  for item in plan.integration.interfaces
    minus_values = minus(item)
    plus_values = plus(item)

    for point_index in 1:point_count(item)
      minus_velocity = value(minus_values, state, velocity, point_index)
      plus_velocity = value(plus_values, state, velocity, point_index)
      jump_normal = normal_component(jump(minus_velocity, plus_velocity), normal(item))
      total += jump_normal^2 * weight(item, point_index)
    end
  end

  return sqrt(total)
end

function dg_mass_monitor_l2(plan, state, velocity)
  hypot(broken_divergence_l2(plan, state, velocity), normal_velocity_jump_l2(plan, state, velocity))
end

# ---------------------------------------------------------------------------
# 1. Euler helpers and blast-wave data
# ---------------------------------------------------------------------------

# We store the conservative variables `q = (ρ, ρu, ρv, E)` because they are the
# natural unknowns for the Euler equations. Many flux formulas are simpler in
# primitive variables `(ρ, u, v, p)`, so the first few helpers just move between
# these two views of the same state.
@inline dot2(a, b) = a[1] * b[1] + a[2] * b[2]
@inline squared_norm2(a) = dot2(a, a)
@inline component_local_index(mode_count::Int, component::Int, mode::Int) = (component - 1) *
                                                                            mode_count + mode

# Recover `(ρ, u, v, p)` from the conservative state while applying the small
# positivity floors used throughout the example.
@inline function primitive_variables(q, gamma)
  rho = max(q[1], DENSITY_FLOOR)
  inv_rho = inv(rho)
  velocity_data = (q[2] * inv_rho, q[3] * inv_rho)
  kinetic = 0.5 * (q[2] * velocity_data[1] + q[3] * velocity_data[2])
  pressure_value = max((gamma - 1.0) * (q[4] - kinetic), PRESSURE_FLOOR)
  return rho, velocity_data, pressure_value
end

@inline pressure(q, gamma) = primitive_variables(q, gamma)[3]
@inline velocity(q, gamma) = primitive_variables(q, gamma)[2]

@inline function sound_speed(q, gamma)
  rho, _, pressure_value = primitive_variables(q, gamma)
  return sqrt(gamma * pressure_value / rho)
end

# Reassemble the conservative variables after specifying `ρ`, `(u, v)`, and
# `p`. For ideal-gas Euler, the total energy is
#
#   E = p / (γ - 1) + ρ |u|² / 2.
@inline function conservative_variables(rho, velocity_data, pressure_value, gamma)
  kinetic = 0.5 * rho * squared_norm2(velocity_data)
  energy = pressure_value / (gamma - 1.0) + kinetic
  return (rho, rho * velocity_data[1], rho * velocity_data[2], energy)
end

# The physical Euler flux `F(q)` is stored component-wise as a tuple of
# spatial flux vectors, one for each conserved variable.
function euler_flux(q, gamma)
  _, velocity_data, pressure_value = primitive_variables(q, gamma)
  velocity1 = velocity_data[1]
  velocity2 = velocity_data[2]
  momentum1 = q[2]
  momentum2 = q[3]
  energy = q[4]
  return ((momentum1, momentum2), (momentum1 * velocity1 + pressure_value, momentum1 * velocity2),
          (momentum2 * velocity1, momentum2 * velocity2 + pressure_value),
          ((energy + pressure_value) * velocity1, (energy + pressure_value) * velocity2))
end

# Face integrals only need `F(q) · n`, so this helper evaluates the physical
# normal flux directly.
function flux_dot_normal(q, normal_data, gamma)
  _, velocity_data, pressure_value = primitive_variables(q, gamma)
  normal_velocity = dot2(velocity_data, normal_data)
  energy = q[4]
  return (q[1] * normal_velocity, q[2] * normal_velocity + pressure_value * normal_data[1],
          q[3] * normal_velocity + pressure_value * normal_data[2],
          (energy + pressure_value) * normal_velocity)
end

# We close interfaces and boundaries with a local Lax-Friedrichs
# (Rusanov) flux,
#
#   F̂(q⁻, q⁺, n) = 1/2 (F(q⁻)·n + F(q⁺)·n - α (q⁺ - q⁻)),
#
# where `α` is the largest one-sided acoustic wave speed.
function lax_friedrichs_flux(q_minus, q_plus, normal_data, gamma)
  flux_minus = flux_dot_normal(q_minus, normal_data, gamma)
  flux_plus = flux_dot_normal(q_plus, normal_data, gamma)
  speed_minus = abs(dot2(velocity(q_minus, gamma), normal_data)) + sound_speed(q_minus, gamma)
  speed_plus = abs(dot2(velocity(q_plus, gamma), normal_data)) + sound_speed(q_plus, gamma)
  alpha = max(speed_minus, speed_plus)
  return ntuple(component -> 0.5 * (flux_minus[component] + flux_plus[component] -
                                    alpha * (q_plus[component] - q_minus[component])), 4)
end

# A reflective wall or symmetry plane flips only the normal momentum. Density
# and total energy stay unchanged, and tangential momentum is preserved.
@inline function reflective_ghost_state(q, normal_data)
  reflected_normal_momentum = q[2] * normal_data[1] + q[3] * normal_data[2]
  reflected_momentum1 = q[2] - 2.0 * reflected_normal_momentum * normal_data[1]
  reflected_momentum2 = q[3] - 2.0 * reflected_normal_momentum * normal_data[2]
  return (q[1], reflected_momentum1, reflected_momentum2, q[4])
end

"""
    blast_wave_initial_condition(x; gamma=GAMMA)

Sedov blast-wave data on `[-1.5, 1.5]^2`.

The state starts from rest in a periodic box. Density and pressure are
initialized as Gaussian concentrations in a homogeneous ambient medium,
following equation (22) of Rueda-Ramírez and Gassner,
arXiv:2102.06017v1. The paper uses this smooth initialization to generate a
strong expanding shock without imposing a discontinuous initial condition.
"""
function blast_wave_initial_condition(x; gamma=GAMMA, center=BLAST_CENTER,
                                      ambient_density=BACKGROUND_DENSITY,
                                      ambient_pressure=BACKGROUND_PRESSURE,
                                      density_sigma=DENSITY_SIGMA, pressure_sigma=PRESSURE_SIGMA)
  density_sigma > 0 || throw(ArgumentError("density_sigma must be positive"))
  pressure_sigma > 0 || throw(ArgumentError("pressure_sigma must be positive"))
  radius_squared = (x[1] - center[1])^2 + (x[2] - center[2])^2
  density_value = ambient_density +
                  inv(4π * density_sigma^2) * exp(-0.5 * radius_squared / density_sigma^2)
  pressure_value = ambient_pressure +
                   (gamma - 1.0) *
                   inv(4π * pressure_sigma^2) *
                   exp(-0.5 * radius_squared / pressure_sigma^2)
  return conservative_variables(density_value, (0.0, 0.0), pressure_value, gamma)
end

# Before time integration starts, we seed a few layers of `h`-refinement around
# the Gaussian concentration so the outgoing shock starts from a locally refined
# mesh.
@inline function squared_distance_to_box(point, lower, upper)
  distance = 0.0

  for axis in 1:2
    if point[axis] < lower[axis]
      delta = lower[axis] - point[axis]
      distance += delta^2
    elseif point[axis] > upper[axis]
      delta = point[axis] - upper[axis]
      distance += delta^2
    end
  end

  return distance
end

@inline function cell_intersects_disk(domain, leaf, center, radius)
  radius >= 0 || throw(ArgumentError("radius must be nonnegative"))
  lower = cell_lower(domain, leaf)
  upper = cell_upper(domain, leaf)
  return squared_distance_to_box(center, lower, upper) <= radius^2
end

function pre_refine_blast_wave_space(space; layers=INITIAL_BLAST_REFINEMENT_LAYERS,
                                     center=BLAST_CENTER, radius=INITIAL_BLAST_REFINEMENT_RADIUS)
  layers >= 0 || throw(ArgumentError("layers must be nonnegative"))
  radius >= 0 || throw(ArgumentError("radius must be nonnegative"))
  layers == 0 && return space

  source_grid = grid(space)
  base_max_level = maximum(Grico.level(source_grid, leaf, axis)
                           for leaf in active_leaves(space), axis in 1:dimension(space))
  plan = AdaptivityPlan(space; limits=AdaptivityLimits(space; max_h_level=base_max_level + layers))

  for _ in 1:layers
    leaves_to_refine = Int[]

    for leaf in active_leaves(plan)
      cell_intersects_disk(plan.target_domain, leaf, center, radius) &&
        push!(leaves_to_refine, leaf)
    end

    isempty(leaves_to_refine) && break

    for leaf in leaves_to_refine
      request_h_refinement!(plan, leaf, (true, true))
    end
  end

  return transition(plan).target_space
end

using Printf
using Grico
import Grico: cell_matrix!, cell_rhs!, h_adaptation_axes, interface_matrix!, source_leaves,
              stored_cell_count, target_space

# Two-dimensional Kelvin-Helmholtz shear-layer example.
#
# Physical model:
#   We solve the incompressible Navier-Stokes equations on Ω = [0, 1]²,
#
#     ∂ₜu + (u · ∇)u - νΔu + ∇p = 0,
#                    ∇ · u      = 0,
#
#   with periodic boundary conditions in x and Dirichlet velocity data on the
#   top and bottom boundaries y = 0 and y = 1 chosen to preserve the background
#   shear profile there. Two horizontal shear layers are placed near y = 0.25
#   and y = 0.75. A small vertical perturbation seeds the Kelvin-Helmholtz
#   instability, after which the layers roll up into vortical billows.
#
# Discretization strategy:
#   The flow uses continuous equal-order H¹ spaces for velocity and pressure.
#   That choice is compact and easy to read, but it is not inf-sup stable by
#   itself, so the example adds a small pressure-gradient stabilization term.
#   A grad-div term further discourages compressibility errors, and a mild
#   interior-face penalty regularizes jumps of velocity normal derivatives across
#   hp interfaces. Time integration is a semi-implicit Oseen step: the advecting
#   velocity is frozen from the previous step, so each update solves one linear
#   saddle-point system instead of a nonlinear Newton problem.
#
# Scalar transport:
#   A passive concentration field c tracks how strongly the two streams mix. It
#   solves
#
#     ∂ₜc + u · ∇c - κΔc = 0
#
#   with SUPG-style streamline stabilization. The concentration is also used as
#   the h-adaptivity indicator because the strongest mixing structures occur near
#   its sharp transition layers.
#
# Reading guide:
#   The file is organized in the same order as the algorithm:
#   1. Define parameters and initial data.
#   2. Assemble reusable local operators for projection, flow, and transport.
#   3. Build plans, project the initial state, and enter the time loop.
#   4. At each step, measure diagnostics, adapt, write output, then advance the
#      flow solve followed by the passive scalar solve.

# Mesh and polynomial order for the initial tensor-product grid. The example
# starts on a modest uniform mesh, applies a small deterministic y-only
# prerefinement around the two shear layers, and then lets h-adaptivity
# concentrate resolution later where the scalar interface folds into thin
# filaments.
const ROOT_COUNTS = (24, 24)
const PREREFINEMENT_LEVELS = 5
const DEGREE = 2

# Temporal resolution. A smaller Δt is important here because the instability
# develops through advection-dominated shear roll-up, and we want the linearized
# Oseen step to track that evolution without excessive temporal diffusion.
const TIME_STEP = 0.00025
const STEP_COUNT = 800
const ADAPTIVITY_INTERVAL = 1

# Flow and stabilization parameters:
#   ν   = kinematic viscosity
#   δₚ  = pressure-gradient stabilization strength
#   γ   = grad-div coefficient
#   η   = interior-face penalty coefficient
#   κ   = passive-scalar diffusivity
#
# In this benchmark ν and κ are both small enough to permit thin layers, but not
# so small that the run becomes dominated by unresolved underdiffused oscillation.
const VISCOSITY = 7.5e-8
const PRESSURE_STABILIZATION = 5.0e-4
const GRAD_DIV = 5.0e-2
const INTERFACE_PENALTY = 2.5e-2
const MIXTURE_DIFFUSIVITY = 3.0e-4

# h-adaptivity controls based on the scalar field. Cells above the refinement
# threshold are candidates for splitting, cells below the coarsening threshold
# can be merged, and MAX_H_LEVEL prevents refinement from cascading forever once
# thin scalar filaments appear.
const H_REFINEMENT_THRESHOLD = 0.5
const H_COARSENING_THRESHOLD = 2.0e-5
const MAX_H_LEVEL = 5

# Shear-layer and perturbation parameters. SHEAR_WIDTH controls the thickness of
# the tanh transition. FLOW_SPEED sets the characteristic horizontal velocity.
# The perturbation is localized around the two layers and sinusoidal in x so the
# roll-up starts from a clean, well-understood mode.
const SHEAR_LAYER_CENTERS = (0.25, 0.75)
const SHEAR_WIDTH = 0.00005
const FLOW_SPEED = 4.0
const PERTURBATION_AMPLITUDE = 5.0e-2
const PERTURBATION_WIDTH = 0.08

# Output settings only affect post-processing. They are separated from the model
# parameters so it is obvious that changing them does not alter the numerics.
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 2

# Tiny helpers used inside quadrature loops. They keep the local assembly code
# uncluttered while still compiling to simple scalar arithmetic.
@inline dot2(a, b) = a[1] * b[1] + a[2] * b[2]
@inline squared_norm2(a) = dot2(a, a)

# The root mesh is much coarser than the nominal tanh width, so we sharpen the
# projected initial condition by refining narrow horizontal bands around the two
# shear centers before the space is built. Refinement is anisotropic in y
# because the layers are thin only across the shear, not along the streamwise x
# direction where the data are already periodic and smooth.
function apply_shear_prerefinement!(domain)
  PREREFINEMENT_LEVELS <= 0 && return domain

  grid_data = grid(domain)
  band_half_width = 4.0 * SHEAR_WIDTH

  for _ in 1:PREREFINEMENT_LEVELS
    marked = Int[]

    for leaf in active_leaves(grid_data)
      lower = cell_lower(domain, leaf, 2)
      upper = cell_upper(domain, leaf, 2)
      intersects_band = any(center -> lower <= center + band_half_width &&
                                      center - band_half_width <= upper, SHEAR_LAYER_CENTERS)
      intersects_band && push!(marked, leaf)
    end

    isempty(marked) && break

    for leaf in marked
      refine!(grid_data, leaf, 2)
    end
  end

  return domain
end

# The base flow consists of two opposed tanh shear layers. Far below the lower
# layer and far above the upper layer the horizontal velocity is approximately 0;
# between them it is approximately FLOW_SPEED. The result is a central strip
# moving to the right relative to the outer fluid, which is the classic
# Kelvin-Helmholtz setup.
#
# The vertical perturbation is intentionally much smaller than the horizontal
# speed. It does not define the instability by itself; it merely breaks the
# exact symmetry so the unstable shear mode has something to amplify.
function initial_velocity(x)
  lower_center, upper_center = SHEAR_LAYER_CENTERS
  layer_1 = tanh((x[2] - lower_center) / SHEAR_WIDTH)
  layer_2 = tanh((x[2] - upper_center) / SHEAR_WIDTH)
  horizontal = FLOW_SPEED * (layer_1 - layer_2 - 1.0)
  bump_1 = exp(-((x[2] - lower_center) / PERTURBATION_WIDTH)^2)
  bump_2 = exp(-((x[2] - upper_center) / PERTURBATION_WIDTH)^2)
  vertical = PERTURBATION_AMPLITUDE * sin(8.0 * pi * x[1]) * (bump_1 + bump_2)
  return (horizontal, vertical)
end

# The passive scalar labels the central stream with c ≈ 1 and the outer fluid
# with c ≈ 0. Because it shares the same tanh transitions as the velocity shear,
# it is an effective marker for where the most dynamically interesting interfaces
# live. Later, as the flow stretches and folds the interface, c becomes a useful
# adaptivity indicator and visualization field.
function initial_concentration(x)
  lower_center, upper_center = SHEAR_LAYER_CENTERS
  layer_1 = tanh((x[2] - lower_center) / SHEAR_WIDTH)
  layer_2 = tanh((x[2] - upper_center) / SHEAR_WIDTH)
  return 0.5 * (layer_1 - layer_2)
end

# Helper for turning scalar or vector-valued data callbacks into per-component
# entries during projection assembly.
@inline function data_component(value, component_total::Int, component::Int)
  if component_total == 1
    value isa Tuple && return value[1]
    value isa AbstractVector && return value[1]
    return value
  end

  return value[component]
end

# L² projection building block. The matrix assembled here is
#
#   coefficient ⋅ ∫ φᵢ φⱼ dΩ
#
# for each component of the target field. Reusing the same tiny operator for
# both velocity and concentration keeps the example focused on the interesting
# fluid operators instead of duplicating projection code.
struct FieldProjection{F,T}
  field::F
  coefficient::T
end

function cell_matrix!(local_matrix, operator::FieldProjection, values::CellValues)
  field = operator.field
  mode_count = local_mode_count(values, field)

  for point_index in 1:point_count(values)
    # `weight` already contains the Jacobian determinant and quadrature weight,
    # so this is the physical-cell measure dΩ at the current point.
    weighted = operator.coefficient * weight(values, point_index)

    for component in 1:component_count(field)
      for row_mode in 1:mode_count
        row = local_dof_index(values, field, component, row_mode)
        shape_row = shape_value(values, field, point_index, row_mode)

        for col_mode in 1:mode_count
          col = local_dof_index(values, field, component, col_mode)
          local_matrix[row, col] += shape_row *
                                    shape_value(values, field, point_index, col_mode) *
                                    weighted
        end
      end
    end
  end

  return nothing
end

# Right-hand side for the same L² projection:
#
#   ∫ f φᵢ dΩ
#
# where `data` may return a scalar or vector depending on the field.
struct FieldProjectionSource{F,D}
  field::F
  data::D
end

function cell_rhs!(local_rhs, operator::FieldProjectionSource, values::CellValues)
  field = operator.field
  mode_count = local_mode_count(values, field)

  for point_index in 1:point_count(values)
    target = operator.data(point(values, point_index))
    weighted = weight(values, point_index)

    for component in 1:component_count(field)
      value_component = data_component(target, component_count(field), component) * weighted

      for mode_index in 1:mode_count
        row = local_dof_index(values, field, component, mode_index)
        local_rhs[row] += shape_value(values, field, point_index, mode_index) * value_component
      end
    end
  end

  return nothing
end

# One semi-implicit Navier-Stokes step. The state `old_state` provides the
# lagged advecting velocity uⁿ, while the unknowns are uⁿ⁺¹ and pⁿ⁺¹.
# With test functions v and q, the weak form assembled below is
#
#   (Δt⁻¹uⁿ⁺¹, v)
#   + ((uⁿ · ∇)uⁿ⁺¹, v)
#   + ν(∇uⁿ⁺¹, ∇v)
#   - (pⁿ⁺¹, ∇ · v)
#   + (q, ∇ · uⁿ⁺¹)
#   + δₚ(∇pⁿ⁺¹, ∇q)
#   + γ(∇ · uⁿ⁺¹, ∇ · v)
#   = (Δt⁻¹uⁿ, v).
#
# Students can read this as "backward Euler plus frozen advection". Advanced
# readers may recognize it as a low-order Oseen linearization with symmetric
# pressure stabilization and grad-div control.
mutable struct OseenStep{U,P,S,T}
  velocity::U
  pressure::P
  old_state::S
  inv_dt::T
  viscosity::T
  pressure_stabilization::T
  grad_div::T
end

function cell_matrix!(local_matrix, operator::OseenStep{U,P,S,T},
                      values::CellValues) where {U,P,S,T}
  velocity = operator.velocity
  pressure = operator.pressure
  velocity_modes = local_mode_count(values, velocity)
  pressure_modes = local_mode_count(values, pressure)

  for point_index in 1:point_count(values)
    # Freeze the advecting velocity at uⁿ, evaluated at the quadrature point.
    advecting = value(values, operator.old_state, velocity, point_index)
    weighted = weight(values, point_index)

    for row_component in 1:2
      for row_mode in 1:velocity_modes
        row = local_dof_index(values, velocity, row_component, row_mode)
        row_shape = shape_value(values, velocity, point_index, row_mode)
        row_gradient = shape_gradient(values, velocity, point_index, row_mode)
        row_divergence = row_gradient[row_component]

        for col_component in 1:2
          for col_mode in 1:velocity_modes
            col = local_dof_index(values, velocity, col_component, col_mode)
            col_shape = shape_value(values, velocity, point_index, col_mode)
            col_gradient = shape_gradient(values, velocity, point_index, col_mode)

            # grad-div couples every velocity component through the divergence.
            contribution = operator.grad_div * row_divergence * col_gradient[col_component]

            if row_component == col_component
              # Because each scalar basis function belongs to one component, the
              # component-wise mass, diffusion, and advection terms only act on
              # matching components. Cross-component coupling still enters
              # through grad-div and the pressure block.
              contribution += operator.inv_dt * row_shape * col_shape
              contribution += operator.viscosity * dot2(row_gradient, col_gradient)
              contribution += row_shape * dot2(advecting, col_gradient)
            end

            local_matrix[row, col] += contribution * weighted
          end
        end

        for col_mode in 1:pressure_modes
          # -(p, ∇ · v)
          col = local_dof_index(values, pressure, 1, col_mode)
          local_matrix[row, col] -= row_divergence *
                                    shape_value(values, pressure, point_index, col_mode) *
                                    weighted
        end
      end
    end

    for row_mode in 1:pressure_modes
      row = local_dof_index(values, pressure, 1, row_mode)
      row_shape = shape_value(values, pressure, point_index, row_mode)
      row_gradient = shape_gradient(values, pressure, point_index, row_mode)

      for col_component in 1:2
        for col_mode in 1:velocity_modes
          # +(q, ∇ · u)
          col = local_dof_index(values, velocity, col_component, col_mode)
          col_gradient = shape_gradient(values, velocity, point_index, col_mode)
          local_matrix[row, col] += row_shape * col_gradient[col_component] * weighted
        end
      end

      for col_mode in 1:pressure_modes
        # Pressure-gradient stabilization for equal-order H¹/H¹ spaces. This is
        # not a physical diffusion term; it is a consistency-preserving device
        # used to suppress the checkerboard pressure modes that otherwise appear
        # when velocity and pressure orders match.
        col = local_dof_index(values, pressure, 1, col_mode)
        local_matrix[row, col] += operator.pressure_stabilization *
                                  dot2(row_gradient,
                                       shape_gradient(values, pressure, point_index, col_mode)) *
                                  weighted
      end
    end
  end

  return nothing
end

# Passive-scalar step. The scalar is advanced after the new velocity has been
# computed, so each transport solve uses the most recent flow field. The weak
# form is backward Euler advection-diffusion with SUPG:
#
#   (Δt⁻¹cⁿ⁺¹, w)
#   + (uⁿ⁺¹ · ∇cⁿ⁺¹, w)
#   + κ(∇cⁿ⁺¹, ∇w)
#   + Σ_K τ_K (uⁿ⁺¹ · ∇w, Δt⁻¹cⁿ⁺¹ + uⁿ⁺¹ · ∇cⁿ⁺¹)
#   = (Δt⁻¹cⁿ, w)
#   + Σ_K τ_K (uⁿ⁺¹ · ∇w, Δt⁻¹cⁿ).
#
# The extra streamline term stabilizes high-Pe advection without smearing the
# solution isotropically as much as plain artificial diffusion would.
mutable struct TransportStep{C,U,SC,SV,T}
  concentration::C
  velocity::U
  concentration_state::SC
  velocity_state::SV
  inv_dt::T
  diffusivity::T
end

# A standard residual-based estimate for the SUPG parameter:
#
#   τ ≈ (Δt⁻² + (2|u|/h)² + (4κ/h²)²)^(-1/2).
#
# The three pieces correspond to transient, advective, and diffusive scales.
# When advection dominates, τ behaves like h/(2|u|); when diffusion dominates it
# shrinks accordingly, preventing the stabilization from becoming too strong.
@inline function transport_tau(operator::TransportStep, values::CellValues, advecting)
  domain_data = field_space(operator.concentration).domain
  cell_width = min(cell_size(domain_data, values.leaf, 1), cell_size(domain_data, values.leaf, 2))
  speed = sqrt(squared_norm2(advecting))
  advective = speed == 0.0 ? 0.0 : (2.0 * speed / cell_width)^2
  diffusive = (4.0 * operator.diffusivity / cell_width^2)^2
  return inv(sqrt(operator.inv_dt^2 + advective + diffusive))
end

function cell_matrix!(local_matrix, operator::TransportStep{C,U,SC,SV,T},
                      values::CellValues) where {C,U,SC,SV,T}
  concentration = operator.concentration
  velocity = operator.velocity
  mode_count = local_mode_count(values, concentration)

  for point_index in 1:point_count(values)
    # Evaluate the current velocity directly on this leaf. The transport solve
    # may be rebuilt after h-adaptation, so we do not assume cached quadrature
    # data from a previous mesh.
    advecting = field_value_on_leaf(velocity, operator.velocity_state, values.leaf,
                                    point(values, point_index))
    tau = transport_tau(operator, values, advecting)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      row = local_dof_index(values, concentration, 1, row_mode)
      row_shape = shape_value(values, concentration, point_index, row_mode)
      row_gradient = shape_gradient(values, concentration, point_index, row_mode)
      streamline_row = dot2(advecting, row_gradient)

      for col_mode in 1:mode_count
        col = local_dof_index(values, concentration, 1, col_mode)
        col_shape = shape_value(values, concentration, point_index, col_mode)
        col_gradient = shape_gradient(values, concentration, point_index, col_mode)
        streamline_col = operator.inv_dt * col_shape + dot2(advecting, col_gradient)

        # The last product is the SUPG contribution:
        #   τ(u · ∇w, Δt⁻¹c + u · ∇c)
        local_matrix[row, col] += (operator.inv_dt * row_shape * col_shape +
                                   operator.diffusivity * dot2(row_gradient, col_gradient) +
                                   row_shape * dot2(advecting, col_gradient) +
                                   tau * streamline_row * streamline_col) * weighted
      end
    end
  end

  return nothing
end

function cell_rhs!(local_rhs, operator::TransportStep{C,U,SC,SV,T},
                   values::CellValues) where {C,U,SC,SV,T}
  concentration = operator.concentration
  velocity = operator.velocity
  mode_count = local_mode_count(values, concentration)

  for point_index in 1:point_count(values)
    advecting = field_value_on_leaf(velocity, operator.velocity_state, values.leaf,
                                    point(values, point_index))
    previous_concentration = value(values, operator.concentration_state, concentration, point_index)
    tau = transport_tau(operator, values, advecting)
    weighted_value = operator.inv_dt * previous_concentration * weight(values, point_index)

    for row_mode in 1:mode_count
      row = local_dof_index(values, concentration, 1, row_mode)
      row_shape = shape_value(values, concentration, point_index, row_mode)
      row_gradient = shape_gradient(values, concentration, point_index, row_mode)
      streamline_row = dot2(advecting, row_gradient)

      # This is the SUPG-modified history term:
      #   (w + τu · ∇w, Δt⁻¹cⁿ).
      local_rhs[row] += (row_shape + tau * streamline_row) * weighted_value
    end
  end

  return nothing
end

# The Oseen right-hand side is simply the old velocity scaled by Δt⁻¹, i.e. the
# "history" term from backward Euler.
function cell_rhs!(local_rhs, operator::OseenStep{U,P,S,T}, values::CellValues) where {U,P,S,T}
  velocity = operator.velocity
  velocity_modes = local_mode_count(values, velocity)

  for point_index in 1:point_count(values)
    previous_velocity = value(values, operator.old_state, velocity, point_index)
    weighted = operator.inv_dt * weight(values, point_index)

    for component in 1:2
      value_component = previous_velocity[component] * weighted

      for mode_index in 1:velocity_modes
        row = local_dof_index(values, velocity, component, mode_index)
        local_rhs[row] += shape_value(values, velocity, point_index, mode_index) * value_component
      end
    end
  end

  return nothing
end

# Interior-face penalty on jumps of the velocity normal derivative:
#
#   η h_F² ∫_F ⟦∂ₙu⟧ · ⟦∂ₙv⟧ dS.
#
# This is a continuous-interior-penalty style regularization. It is deliberately
# mild: the goal is not to convert the method into DG, but to damp mesh-scale
# oscillations that can appear when equal-order continuous spaces are driven hard
# by advection and local refinement.
struct VelocityJumpPenalty{U,T}
  velocity::U
  coefficient::T
end

# For a scalar basis function φ, ∂ₙφ = ∇φ · n on the current face.
@inline function _normal_gradient(values, field, point_index, mode_index)
  gradient_value = shape_gradient(values, field, point_index, mode_index)
  return dot2(gradient_value, normal(values))
end

function interface_matrix!(local_matrix, operator::VelocityJumpPenalty, values::InterfaceValues)
  velocity = operator.velocity
  minus_values = minus(values)
  plus_values = plus(values)
  minus_minus = Grico.block(local_matrix, minus_values, velocity, minus_values, velocity)
  minus_plus = Grico.block(local_matrix, minus_values, velocity, plus_values, velocity)
  plus_minus = Grico.block(local_matrix, plus_values, velocity, minus_values, velocity)
  plus_plus = Grico.block(local_matrix, plus_values, velocity, plus_values, velocity)
  minus_mode_count = local_mode_count(minus_values, velocity)
  plus_mode_count = local_mode_count(plus_values, velocity)
  domain_data = field_space(velocity).domain
  face_size = 0.5 * (cell_size(domain_data, values.minus_leaf, values.axis) +
                     cell_size(domain_data, values.plus_leaf, values.axis))

  for point_index in 1:point_count(values)
    # The h_F² scaling keeps the penalty dimensionally consistent with the
    # volume Laplacian term for this H¹-based formulation.
    weighted = operator.coefficient * face_size^2 * weight(values, point_index)

    for row_component in 1:2
      row_offset = row_component == 1 ? 0 : minus_mode_count
      plus_row_offset = row_component == 1 ? 0 : plus_mode_count

      for row_mode in 1:minus_mode_count
        minus_row_gradient = _normal_gradient(minus_values, velocity, point_index, row_mode)

        for col_mode in 1:minus_mode_count
          minus_col_gradient = _normal_gradient(minus_values, velocity, point_index, col_mode)
          minus_minus[row_offset+row_mode, row_offset+col_mode] += minus_row_gradient *
                                                                   minus_col_gradient *
                                                                   weighted
        end

        for col_mode in 1:plus_mode_count
          plus_col_gradient = _normal_gradient(plus_values, velocity, point_index, col_mode)
          minus_plus[row_offset+row_mode, plus_row_offset+col_mode] -= minus_row_gradient *
                                                                       plus_col_gradient *
                                                                       weighted
        end
      end

      for row_mode in 1:plus_mode_count
        plus_row_gradient = _normal_gradient(plus_values, velocity, point_index, row_mode)

        for col_mode in 1:minus_mode_count
          minus_col_gradient = _normal_gradient(minus_values, velocity, point_index, col_mode)
          plus_minus[plus_row_offset+row_mode, row_offset+col_mode] -= plus_row_gradient *
                                                                       minus_col_gradient *
                                                                       weighted
        end

        for col_mode in 1:plus_mode_count
          plus_col_gradient = _normal_gradient(plus_values, velocity, point_index, col_mode)
          plus_plus[plus_row_offset+row_mode, plus_row_offset+col_mode] += plus_row_gradient *
                                                                           plus_col_gradient *
                                                                           weighted
        end
      end
    end
  end

  return nothing
end

# Merge several field states into one block state for output or coupled solves.
# This avoids recomputing fields that already live in separate plans.
function combine_states(layout, pairs...)
  combined = zeros(Float64, Grico.dof_count(layout))

  for (field, state) in pairs
    combined[field_dof_range(layout, field)] .= field_values(state, field)
  end

  return State(layout, combined)
end

# Diagnostics:
#   kinetic_energy  = 1/2 ∫ |u|² dΩ
#   enstrophy       = 1/2 ∫ ω² dΩ, with ω = ∂ₓu_y - ∂ᵧu_x in 2D
#   divergence_l2   = ‖∇ · u‖_{L²}
#   concentration_range = min/max sampled at integration points
#
# In a well-resolved incompressible run, energy typically decays, enstrophy
# rises during filament formation and then diffuses, and divergence_l2 should
# stay small relative to the flow scales.
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

function enstrophy(plan, state, velocity)
  total = 0.0

  for cell in plan.integration.cells
    for point_index in 1:point_count(cell)
      gradients = gradient(cell, state, velocity, point_index)
      vorticity = gradients[2][1] - gradients[1][2]
      total += 0.5 * vorticity^2 * weight(cell, point_index)
    end
  end

  return total
end

function divergence_l2(plan, state, velocity)
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

function concentration_range(plan, state, concentration)
  minimum_value = Inf
  maximum_value = -Inf

  for cell in plan.integration.cells
    for point_index in 1:point_count(cell)
      value_at_point = value(cell, state, concentration, point_index)
      minimum_value = min(minimum_value, value_at_point)
      maximum_value = max(maximum_value, value_at_point)
    end
  end

  return minimum_value, maximum_value
end

# Manual point evaluation on a specific leaf. This is useful after mesh
# adaptation, when the transport operator needs the advecting velocity on the
# current leaf but only has the state vector. The basis is reconstructed from the
# tensor-product integrated Legendre tables used by the Grico package.
function field_component_value_on_leaf(field, state, leaf, x, component)
  space = field_space(field)
  ξ = map_to_biunit_cube(space.domain, leaf, x)
  one_dimensional = ntuple(axis -> integrated_legendre_values(ξ[axis],
                                                              cell_degrees(space, leaf)[axis]),
                           length(ξ))
  values = field_component_values(state, field, component)
  result = 0.0

  for mode in local_modes(space, leaf)
    local_value = 1.0

    for axis in 1:length(ξ)
      local_value *= one_dimensional[axis][mode[axis]+1]
    end

    for term in mode_terms(space, leaf, mode)
      result += local_value * term.second * values[term.first]
    end
  end

  return result
end

function field_value_on_leaf(field::VectorField, state, leaf, x)
  return ntuple(component -> field_component_value_on_leaf(field, state, leaf, x, component),
                component_count(field))
end

# Project the analytic velocity onto the discrete space instead of interpolating
# nodally. L² projection is robust for hp spaces and naturally handles the vector
# field component-by-component. Only the top and bottom boundaries are fixed
# strongly; the left and right sides are periodic at the mesh/space level.
function build_velocity_projection_plan(velocity)
  problem = AffineProblem(velocity)
  add_cell!(problem, FieldProjection(velocity, 1.0))
  add_cell!(problem, FieldProjectionSource(velocity, initial_velocity))

  for side in (LOWER, UPPER)
    add_constraint!(problem, Dirichlet(velocity, BoundaryFace(2, side), initial_velocity))
  end

  return compile(problem)
end

# The concentration projection is simpler because the scalar is only used as an
# interior marker; no Dirichlet conditions are needed for its initial state.
function build_concentration_projection_plan(concentration)
  problem = AffineProblem(concentration)
  add_cell!(problem, FieldProjection(concentration, 1.0))
  add_cell!(problem, FieldProjectionSource(concentration, initial_concentration))
  return compile(problem)
end

# Build and compile one linearized flow step on the current mesh. We keep the
# operator object so its frozen state can be updated in-place from step to step.
function build_flow_step_plan(velocity, pressure, flow_state)
  step_operator = OseenStep(velocity, pressure, flow_state, inv(TIME_STEP), VISCOSITY,
                            PRESSURE_STABILIZATION, GRAD_DIV)
  problem = AffineProblem(velocity, pressure)
  add_cell!(problem, step_operator)
  add_interface!(problem, VelocityJumpPenalty(velocity, INTERFACE_PENALTY))

  # Pressure is only defined up to a constant, so pin its mean to zero.
  add_constraint!(problem, MeanValue(pressure, 0.0))

  for side in (LOWER, UPPER)
    add_constraint!(problem, Dirichlet(velocity, BoundaryFace(2, side), initial_velocity))
  end

  return step_operator, compile(problem)
end

# Build the scalar transport plan on the current mesh, again retaining the
# operator so the lagged concentration and current velocity can be refreshed.
function build_transport_plan(concentration, velocity, concentration_state, flow_state)
  transport_operator = TransportStep(concentration, velocity, concentration_state, flow_state,
                                     inv(TIME_STEP), MIXTURE_DIFFUSIVITY)
  problem = AffineProblem(concentration)
  add_cell!(problem, transport_operator)
  return transport_operator, compile(problem)
end

# All fields start on the same hp space. Using one shared mesh is convenient for
# this example because the concentration-driven adaptivity then immediately
# benefits the velocity and pressure discretization as well. The x direction is
# periodic, so the left and right edges are identified by the library rather
# than constrained manually in this example. Before building the space, we
# prerefine the two horizontal shear bands so the initial L² projection resolves
# the thin tanh transitions more faithfully than the coarse root grid would.
domain = Domain((0.0, 0.0), (1.0, 1.0), ROOT_COUNTS; periodic=(true, false))
apply_shear_prerefinement!(domain)
space = HpSpace(domain, SpaceOptions(degree=UniformDegree(DEGREE)))
velocity = VectorField(space, 2; name=:velocity)
pressure = ScalarField(space; name=:pressure)
concentration = ScalarField(space; name=:concentration)

# Build initial states. Velocity and concentration come from L² projection of
# the analytic setup, while pressure starts from zero and is determined by the
# first flow solve up to the imposed mean-value constraint.
projection_plan = build_velocity_projection_plan(velocity)
velocity_state = State(projection_plan, solve(assemble(projection_plan)))

flow_layout = FieldLayout((velocity, pressure))
flow_state = combine_states(flow_layout, velocity => velocity_state,
                            pressure => State(FieldLayout((pressure,))))

concentration_plan = build_concentration_projection_plan(concentration)
concentration_state = State(concentration_plan, solve(assemble(concentration_plan)))

output_layout = FieldLayout((velocity, pressure, concentration))
step_operator, step_plan = build_flow_step_plan(velocity, pressure, flow_state)
transport_operator, transport_plan = build_transport_plan(concentration, velocity,
                                                          concentration_state, flow_state)

const RUN_KELVIN_HELMHOLTZ = get(ENV, "GRICO_KH_AUTORUN", "1") == "1"

if RUN_KELVIN_HELMHOLTZ
  # Print a compact run summary so the user can relate later diagnostics and VTK
  # output to the chosen parameter set.
  println("kelvin_helmholtz.jl")
  println("  two-dimensional Kelvin-Helmholtz shear layer")
  println("  equal-order H1/H1, semi-implicit Oseen stepping with passive-scalar mixing")
  println("  step time leaves scalar-dofs kinetic-energy enstrophy div-l2 c-min c-max adapt")
  @printf("  roots              : %s\n", ROOT_COUNTS)
  @printf("  prerefinement      : %d y-levels around each shear layer\n", PREREFINEMENT_LEVELS)
  @printf("  scalar dofs        : %d\n", scalar_dof_count(space))
  @printf("  velocity dofs      : %d\n", field_dof_count(velocity))
  @printf("  pressure dofs      : %d\n", field_dof_count(pressure))
  @printf("  concentration dofs : %d\n", field_dof_count(concentration))
  @printf("  h adaptivity       : every %d steps, refine=%.2f, coarsen=%.3e, max level=%d\n",
          ADAPTIVITY_INTERVAL, H_REFINEMENT_THRESHOLD, H_COARSENING_THRESHOLD, MAX_H_LEVEL)

  output_directory = joinpath(@__DIR__, "output", "KHI-2")
  vtk_files = String[]
  vtk_times = Float64[]
  WRITE_VTK && mkpath(output_directory)

  # The `let` block keeps the time-marching state local while still allowing
  # rebinding after mesh adaptation. The loop order is:
  #   1. Measure diagnostics on the current discrete state.
  #   2. Optionally mark and build an h-adaptivity transition from the scalar.
  #   3. Write output for the current time.
  #   4. Transfer states if the mesh changes.
  #   5. Advance the Oseen flow step.
  #   6. Advance the passive-scalar transport step with the new velocity.
  #
  # Step 0 is therefore the projected initial condition, not the result of one
  # time update. This is often the most useful convention for transient examples.
  let velocity = velocity,
    pressure = pressure,
    concentration = concentration,
    flow_preconditioner = FieldSplitSchurPreconditioner((velocity,), (pressure,)),
    flow_state = flow_state,
    concentration_state = concentration_state,
    output_layout = output_layout,
    step_operator = step_operator,
    step_plan = step_plan,
    transport_operator = transport_operator,
    transport_plan = transport_plan

    for step in 0:STEP_COUNT
      current_time = step * TIME_STEP
      energy = kinetic_energy(step_plan, flow_state, velocity)
      current_enstrophy = enstrophy(step_plan, flow_state, velocity)
      div_l2 = divergence_l2(step_plan, flow_state, velocity)
      concentration_min, concentration_max = concentration_range(transport_plan,
                                                                 concentration_state, concentration)
      current_space = field_space(velocity)
      current_grid = grid(current_space)
      h_adaptation = zeros(Float64, stored_cell_count(current_grid))
      adaptation_note = "-"
      adaptivity_plan = nothing
      space_transition = nothing

      if step > 0 && step < STEP_COUNT && step % ADAPTIVITY_INTERVAL == 0
        # Adaptivity is driven by the scalar because it cleanly highlights the
        # sheared interface. In practice this also resolves the velocity billows,
        # since the strongest vorticity structures remain tied to the same layers.
        limits = AdaptivityLimits(current_space; max_h_level=MAX_H_LEVEL)
        adaptivity_plan = h_adaptivity_plan(concentration_state, concentration;
                                            threshold=H_REFINEMENT_THRESHOLD,
                                            h_coarsening_threshold=H_COARSENING_THRESHOLD,
                                            limits=limits)

        if isempty(adaptivity_plan)
          adaptation_note = "h=0"
        else
          summary = adaptivity_summary(adaptivity_plan)
          adaptation_note = @sprintf("h+=%d,h-=%d", summary.h_refinement_leaf_count,
                                     summary.h_derefinement_cell_count)
          space_transition = transition(adaptivity_plan)

          # Positive values mark leaves that will be refined in the current mesh.
          for leaf in active_leaves(current_space)
            any(h_adaptation_axes(adaptivity_plan, leaf)) && (h_adaptation[leaf] = 1.0)
          end

          # Negative values mark parent cells that disappear through derefinement.
          for target_leaf in active_leaves(target_space(space_transition))
            sources = source_leaves(space_transition, target_leaf)
            length(sources) > 1 || continue

            for source_leaf in sources
              h_adaptation[source_leaf] = -1.0
            end
          end
        end
      end

      output_state = combine_states(output_layout, velocity => flow_state, pressure => flow_state,
                                    concentration => concentration_state)

      @printf("  %4d %5.2f %5d %6d %.6e %.6e %.3e %.3e %.3e %s\n", step, current_time,
              active_leaf_count(current_space), scalar_dof_count(current_space), energy,
              current_enstrophy, div_l2, concentration_min, concentration_max, adaptation_note)

      if WRITE_VTK
        vtk_path = write_vtk(joinpath(output_directory, @sprintf("kelvin_helmholtz_%04d", step)),
                             output_state;
                             point_data=(speed=(x, values) -> sqrt(values.velocity[1]^2 +
                                                                   values.velocity[2]^2),
                                         # 4c(1 - c) is near 0 in the unmixed
                                         # phases c ≈ 0 or c ≈ 1 and near 1 in a
                                         # perfectly mixed 50/50 layer.
                                         mixed_fraction=(x, values) -> 4.0 *
                                                                       values.concentration *
                                                                       (1.0 - values.concentration),
                                         vertical_velocity=(x, values) -> values.velocity[2],
                                         # Helpful for comparing the evolving flow
                                         # against the initial horizontal profile.
                                         shear_reference=x -> initial_velocity(x)[1]),
                             cell_data=(leaf=leaf -> Float64(leaf),
                                        level=leaf -> Float64.(level(current_grid, leaf)),
                                        degree=leaf -> Float64.(cell_degrees(current_space, leaf)),
                                        h_adaptation=leaf -> h_adaptation[leaf]),
                             field_data=(time=current_time, kinetic_energy=energy,
                                         enstrophy=current_enstrophy, divergence_l2=div_l2,
                                         concentration_min=concentration_min,
                                         concentration_max=concentration_max),
                             subdivisions=EXPORT_SUBDIVISIONS, export_degree=EXPORT_DEGREE,
                             append=true, compress=true, ascii=false)
        push!(vtk_files, vtk_path)
        push!(vtk_times, current_time)
      end

      step == STEP_COUNT && break

      if !isnothing(adaptivity_plan) && !isempty(adaptivity_plan)
        # After adaptation we must move every field and state to the target mesh,
        # then rebuild the compiled operators because local DOF layouts changed.
        space_transition = something(space_transition, transition(adaptivity_plan))
        new_velocity, new_pressure, new_concentration = adapted_fields(space_transition, velocity,
                                                                       pressure, concentration)
        flow_state = transfer_state(space_transition, flow_state, (velocity, pressure),
                                    (new_velocity, new_pressure))
        concentration_state = transfer_state(space_transition, concentration_state, concentration,
                                             new_concentration)
        velocity, pressure, concentration = new_velocity, new_pressure, new_concentration
        flow_preconditioner = FieldSplitSchurPreconditioner((velocity,), (pressure,))
        output_layout = FieldLayout((velocity, pressure, concentration))
        step_operator, step_plan = build_flow_step_plan(velocity, pressure, flow_state)
        transport_operator, transport_plan = build_transport_plan(concentration, velocity,
                                                                  concentration_state, flow_state)
      end

      # Advance flow first so the scalar uses uⁿ⁺¹ rather than uⁿ. This ordering
      # makes the mixing diagnostic visually track the freshly updated billows.
      step_operator.old_state = flow_state
      # The mixed velocity-pressure Oseen step benefits from a semantic field
      # split: velocity stays on the primary block, while pressure is treated by
      # the Schur correction block.
      flow_state = State(step_plan, solve(assemble(step_plan); preconditioner=flow_preconditioner))
      transport_operator.velocity_state = flow_state
      transport_operator.concentration_state = concentration_state
      concentration_state = State(transport_plan, solve(assemble(transport_plan)))
    end
  end

  # Collect the individual time slices into a ParaView-readable time series.
  if WRITE_VTK
    pvd_path = write_pvd(joinpath(output_directory, "kelvin_helmholtz.pvd"), vtk_files;
                         timesteps=vtk_times)
    println("  pvd  $pvd_path")
  end
end

using LinearAlgebra
using Printf
using Grico
import Grico: cell_matrix!, face_matrix!, face_rhs!, interface_matrix!

# This is the package's main discontinuous Galerkin flow example.
#
# It solves the steady lid-driven cavity problem on the unit square
#
#   Ω = [0, 1]²,
#
# with incompressible velocity-pressure unknowns `(u, p)` satisfying
#
#   (u · ∇)u - νΔu + ∇p = 0,
#               ∇ · u   = 0,
#
# together with the classical cavity boundary data
#
#   u = (1, 0)  on y = 1,
#   u = (0, 0)  on the remaining walls.
#
# The top corners carry a jump in the prescribed tangential velocity. This is a
# natural fit for a DG demo: all wall data are imposed weakly through boundary
# operators, so no continuous trace space has to resolve the corner singularity.
#
# Discretization strategy:
#
# 1. velocity uses a discontinuous vector-valued tensor-product space,
# 2. pressure uses a lower-order DG space on the same active-leaf topology,
# 3. viscous terms use symmetric interior penalty,
# 4. the convective term is linearized by Picard iteration and discretized with
#    an upwind numerical flux,
# 5. wall data are imposed weakly on all four physical faces, and
# 6. a mean-value pressure constraint removes the constant-pressure null space.
#
# Each Picard step therefore solves one steady Oseen problem with the advecting
# velocity frozen from the previous iterate. The example keeps the Reynolds
# number modest so this fixed-point iteration converges robustly from the zero
# initial guess while still producing a recognizable cavity vortex.
#
# File roadmap:
#
# 1. choose physical, DG, and nonlinear-solver parameters,
# 2. define the interior, interface, and boundary contributions of one steady
#    Oseen step,
# 3. define a few physically meaningful diagnostics,
# 4. build reusable spaces, fields, and compiled plans,
# 5. run Picard iteration and adaptive mesh refinement,
# 6. export the final state to VTK.

# ---------------------------------------------------------------------------
# 1. Global parameters
# ---------------------------------------------------------------------------
#
# Mesh and approximation order. The example starts from a coarse uniform mesh
# and lets a few DG jump-indicator refinement cycles find the lid singularities
# automatically.
const ROOT_COUNTS = (16, 16)
const ADAPTIVE_STEPS = 4
const ADAPTIVITY_THRESHOLD = 0.9
const MAX_H_LEVEL = 3
const VELOCITY_DEGREE = 2
const PRESSURE_DEGREE = 1
const QUADRATURE_EXTRA_POINTS = 1

# Flow and DG stabilization parameters.
const REYNOLDS_NUMBER = 100.0
const LID_SPEED = 1.0
const VISCOSITY = LID_SPEED / REYNOLDS_NUMBER
const VELOCITY_PENALTY = 6.0
const NORMAL_FLUX_PENALTY = 1.0
const PRESSURE_JUMP_PENALTY = 0.0
const DIVERGENCE_PENALTY = 5.0e-3

# Picard iteration controls.
const PICARD_MAX_ITERS = 24
const PICARD_TOL = 1.0e-6
const ADAPTIVE_PICARD_TOL = 1.0e-4
const PICARD_RELAXATION = 0.85

# Optional VTK output settings.
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 3
# Benchmarks may include this file for its helper functions without running the
# full example. Direct execution keeps the default autorun behavior.
const RUN_LID_DRIVEN_CAVITY = get(ENV, "GRICO_LDC_AUTORUN", "1") == "1"

# ---------------------------------------------------------------------------
# 2. Small algebra helpers
# ---------------------------------------------------------------------------
#
# These are just tiny utilities used repeatedly inside the local operator loops.
# Keeping them separate makes the weak-form code below easier to read.

# Tiny helpers used inside quadrature loops.
@inline dot2(a, b) = a[1] * b[1] + a[2] * b[2]
@inline squared_norm2(a) = dot2(a, a)
@inline velocity_index(mode_count::Int, component::Int, mode::Int) = (component - 1) * mode_count +
                                                                     mode

# Grico's generic `jump(minus, plus)` helper returns `plus - minus`, but the DG
# bilinear forms below are written in the more common trace convention
# [w] = w⁻ - w⁺ on an interface with normal pointing from minus to plus.
@inline trace_jump_sign(is_plus::Bool) = is_plus ? -1.0 : 1.0

@inline function interface_length_scale(field, minus_leaf, plus_leaf, axis)
  domain_data = field_space(field).domain
  h_minus = cell_size(domain_data, minus_leaf, axis)
  h_plus = cell_size(domain_data, plus_leaf, axis)
  return 2.0 * h_minus * h_plus / (h_minus + h_plus)
end

@inline boundary_length_scale(field, leaf, axis) = cell_size(field_space(field).domain, leaf, axis)

@inline function interface_penalty_scale(field, minus_leaf, plus_leaf, axis)
  h = interface_length_scale(field, minus_leaf, plus_leaf, axis)
  degree_value = max(cell_degrees(field_space(field), minus_leaf)[axis],
                     cell_degrees(field_space(field), plus_leaf)[axis])
  return (degree_value + 1)^2 / h
end

@inline function boundary_penalty_scale(field, leaf, axis)
  h = boundary_length_scale(field, leaf, axis)
  degree_value = cell_degrees(field_space(field), leaf)[axis]
  return (degree_value + 1)^2 / h
end

# The lid data are intentionally discontinuous at the top corners. The weak DG
# wall treatment handles that without requiring a globally continuous boundary
# trace. Side and bottom walls remain no-slip.
cavity_velocity(x) = x[2] >= 1.0 - 1.0e-12 ? (LID_SPEED, 0.0) : (0.0, 0.0)

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

direct_sparse_solve(matrix_data, rhs_data) = matrix_data \ rhs_data

# ---------------------------------------------------------------------------
# 5. Space and plan builders
# ---------------------------------------------------------------------------
#
# Velocity and pressure both use DG trunk bases, but pressure is one order lower
# than velocity. This is the classical `p/p-1` mixed pairing used in many DG
# incompressible-flow discretizations.

function build_velocity_space(domain)
  return HpSpace(domain,
                 SpaceOptions(basis=TrunkBasis(), degree=UniformDegree(VELOCITY_DEGREE),
                              quadrature=DegreePlusQuadrature(QUADRATURE_EXTRA_POINTS),
                              continuity=:dg))
end

function build_pressure_space(domain)
  return HpSpace(domain,
                 SpaceOptions(basis=TrunkBasis(), degree=UniformDegree(PRESSURE_DEGREE),
                              quadrature=DegreePlusQuadrature(QUADRATURE_EXTRA_POINTS),
                              continuity=:dg))
end

function build_flow_plan(velocity, pressure, advecting_state)
  # Build one compiled linear Oseen problem. The important design detail is that
  # the operator stores `advecting_state` mutably, so later Picard steps can
  # reuse the same compiled plan and only swap the lagged velocity field values.
  operator = SteadyDGOseen(velocity, pressure, advecting_state, VISCOSITY, VELOCITY_PENALTY,
                           PRESSURE_JUMP_PENALTY, NORMAL_FLUX_PENALTY, DIVERGENCE_PENALTY,
                           cavity_velocity)
  problem = AffineProblem(velocity, pressure)
  add_cell!(problem, operator)
  add_interface!(problem, operator)

  for axis in 1:2, side in (LOWER, UPPER)
    add_boundary!(problem, BoundaryFace(axis, side), operator)
  end

  add_constraint!(problem, MeanValue(pressure, 0.0))
  return operator, compile(problem)
end

# Build the fixed mesh, fields, and compiled Oseen plan used throughout the
# cavity solve. The returned context is also reused by the benchmark scripts,
# which is why the example exposes helpers instead of keeping everything inside
# one top-level script body.
function _lid_driven_cavity_context(velocity, pressure, flow_state)
  velocity_space = field_space(velocity)
  pressure_space = field_space(pressure)
  domain = velocity_space.domain
  flow_layout = field_layout(flow_state)
  operator, plan = build_flow_plan(velocity, pressure, flow_state)
  return (; domain, velocity_space, pressure_space, velocity, pressure, flow_layout, flow_state,
          operator, plan)
end

function build_lid_driven_cavity_context(domain::Domain)
  velocity_space = build_velocity_space(domain)
  pressure_space = build_pressure_space(domain)
  velocity = VectorField(velocity_space, 2; name=:velocity)
  pressure = ScalarField(pressure_space; name=:pressure)
  flow_layout = FieldLayout((velocity, pressure))
  flow_state = State(flow_layout)
  return _lid_driven_cavity_context(velocity, pressure, flow_state)
end

function build_lid_driven_cavity_context(; root_counts=ROOT_COUNTS)
  build_lid_driven_cavity_context(Domain((0.0, 0.0), (1.0, 1.0), root_counts))
end

# Advance one under-relaxed Picard step and return both the updated context and
# the assembled Oseen system so external drivers can benchmark alternative
# linear solver paths on the exact matrix used at that iteration.
function advance_picard_step(context; linear_solve=direct_sparse_solve)
  # 1. update the lagged advecting state in the reusable operator,
  # 2. assemble and solve the linear Oseen system,
  # 3. under-relax the new iterate for robustness,
  # 4. report a relative update and DG diagnostics.
  context.operator.advecting_state = context.flow_state
  system = assemble(context.plan)
  candidate_state = State(context.plan, solve(system; linear_solve=linear_solve))
  relaxed_coefficients = PICARD_RELAXATION == 1.0 ? coefficients(candidate_state) :
                         (1.0 - PICARD_RELAXATION) .* coefficients(context.flow_state) .+
                         PICARD_RELAXATION .* coefficients(candidate_state)
  relaxed_state = PICARD_RELAXATION == 1.0 ? candidate_state :
                  State(context.flow_layout, relaxed_coefficients)
  velocity_update = norm(field_values(relaxed_state, context.velocity) -
                         field_values(context.flow_state, context.velocity))
  relative_update = velocity_update / max(norm(field_values(relaxed_state, context.velocity)), 1.0)
  mass_l2 = dg_mass_monitor_l2(context.plan, relaxed_state, context.velocity)
  energy = kinetic_energy(context.plan, relaxed_state, context.velocity)
  next_context = (; context..., flow_state=relaxed_state)
  return next_context, system, relative_update, mass_l2, energy
end

# Refine the velocity-pressure mesh from the default DG jump indicator and
# transfer the current discrete state to the new spaces.
#
# The important point here is that the mesh change is driven from the velocity
# field, but the pressure space must follow the same new active-leaf topology.
# `derived_adaptivity_plan` expresses exactly that relationship at the library
# level, and `transfer_state((velocity_plan, pressure_plan), ...)` then moves
# the mixed state to the new pair of spaces in one consistent operation.
function adapt_lid_driven_cavity_context(context; threshold=ADAPTIVITY_THRESHOLD,
                                         max_h_level=MAX_H_LEVEL)
  limits = AdaptivityLimits(context.velocity_space; max_h_level=max_h_level)
  velocity_plan = h_adaptivity_plan(context.flow_state, context.velocity; threshold=threshold,
                                    limits=limits)
  isempty(velocity_plan) && return context, velocity_plan
  pressure_plan = derived_adaptivity_plan(velocity_plan, context.pressure;
                                          limits=AdaptivityLimits(context.pressure_space;
                                                                  max_h_level=max_h_level))
  (new_velocity, new_pressure), new_flow_state = transfer_state((velocity_plan, pressure_plan),
                                                                context.flow_state;
                                                                linear_solve=direct_sparse_solve)
  return _lid_driven_cavity_context(new_velocity, new_pressure, new_flow_state), velocity_plan
end

function print_lid_driven_cavity_header()
  println("lid_driven_cavity.jl")
  println("  steady discontinuous Galerkin lid-driven cavity")
  @printf("  Reynolds number     : %.1f\n", REYNOLDS_NUMBER)
  @printf("  velocity degree     : %d\n", VELOCITY_DEGREE)
  @printf("  pressure degree     : %d\n", PRESSURE_DEGREE)
  @printf("  adaptive steps      : %d\n", ADAPTIVE_STEPS)
  println("  cycle iter dofs rel-update dg-mass-monitor kinetic-energy")
  return nothing
end

function write_lid_driven_cavity_vtk(context, iteration_count, final_update)
  # Export both the final fields and enough metadata to understand the adapted
  # mesh afterwards: per-leaf levels, local velocity/pressure degrees, and a few
  # global diagnostic scalars.
  output_directory = joinpath(@__DIR__, "output")
  current_grid = grid(context.velocity_space)
  mkpath(output_directory)
  return write_vtk(joinpath(output_directory, "lid_driven_cavity"), context.flow_state;
                   point_data=(speed=(x, values) -> sqrt(values.velocity[1]^2 +
                                                         values.velocity[2]^2),
                               horizontal_velocity=(x, values) -> values.velocity[1],
                               vertical_velocity=(x, values) -> values.velocity[2]),
                   cell_data=(leaf=leaf -> Float64(leaf),
                              level=leaf -> Float64.(level(current_grid, leaf)),
                              velocity_degree=leaf -> Float64.(cell_degrees(context.velocity_space,
                                                                            leaf)),
                              pressure_degree=leaf -> Float64.(cell_degrees(context.pressure_space,
                                                                            leaf))),
                   field_data=(picard_iterations=Float64(iteration_count),
                               final_relative_update=final_update,
                               kinetic_energy=kinetic_energy(context.plan, context.flow_state,
                                                             context.velocity),
                               dg_mass_monitor_l2=dg_mass_monitor_l2(context.plan,
                                                                     context.flow_state,
                                                                     context.velocity),
                               Reynolds=REYNOLDS_NUMBER), subdivisions=EXPORT_SUBDIVISIONS,
                   export_degree=EXPORT_DEGREE, append=true, compress=true, ascii=false)
end

# Human-facing driver used when the example is run directly from the `examples/`
# directory or from the repository root with Julia.
#
# The control structure is:
#
# - outer loop: adapt the mesh a few times,
# - inner loop: converge the Picard iteration on the current mesh.
function run_lid_driven_cavity_example(; max_iters=PICARD_MAX_ITERS, tol=PICARD_TOL,
                                       linear_solve=direct_sparse_solve, write_vtk=WRITE_VTK)
  print_lid_driven_cavity_header()
  context = build_lid_driven_cavity_context()
  final_update = Inf
  iteration_count = 0
  adaptive_step_count = 0

  for adaptive_step in 0:ADAPTIVE_STEPS
    # On intermediate adaptive meshes we allow a looser nonlinear tolerance so
    # the example does not oversolve a mesh that will be replaced immediately.
    cycle_tol = adaptive_step == ADAPTIVE_STEPS ? tol : max(tol, ADAPTIVE_PICARD_TOL)

    for iteration in 1:max_iters
      context, _, relative_update, mass_l2, energy = advance_picard_step(context;
                                                                         linear_solve=linear_solve)
      @printf("  %5d %4d %4d %.6e %.6e %.6e\n", adaptive_step, iteration,
              length(coefficients(context.flow_state)), relative_update, mass_l2, energy)
      final_update = relative_update
      iteration_count = iteration
      relative_update <= cycle_tol && break
    end

    adaptive_step_count = adaptive_step
    adaptive_step == ADAPTIVE_STEPS && break

    # After the nonlinear iteration settles on the current mesh, ask for one
    # velocity-driven DG `h`-adaptation step and transfer both fields if the
    # planner marked anything.
    next_context, adaptivity_plan = adapt_lid_driven_cavity_context(context)
    isempty(adaptivity_plan) && break
    summary = adaptivity_summary(adaptivity_plan)
    @printf("  refine %d marked=%d h+=%d\n", adaptive_step, summary.marked_leaf_count,
            summary.h_refinement_leaf_count)
    context = next_context
  end

  if write_vtk
    vtk_path = write_lid_driven_cavity_vtk(context, iteration_count, final_update)
    println("  vtk  $vtk_path")
  end

  @printf("  active leaves       : %d\n", active_leaf_count(context.velocity_space))
  @printf("  scalar dofs         : %d\n", scalar_dof_count(context.velocity_space))
  @printf("  mixed dofs          : %d\n", length(coefficients(context.flow_state)))
  @printf("  adaptive steps used : %d\n", adaptive_step_count)
  @printf("  final Picard iters  : %d\n", iteration_count)
  @printf("  final rel. update   : %.6e\n", final_update)
  @printf("  dg mass monitor l2  : %.6e\n",
          dg_mass_monitor_l2(context.plan, context.flow_state, context.velocity))
  return (; context, iteration_count, final_update)
end

RUN_LID_DRIVEN_CAVITY && run_lid_driven_cavity_example()

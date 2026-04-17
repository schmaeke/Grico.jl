using LinearAlgebra
using Printf
using Grico
import Grico: cell_matrix!, cell_residual!, cell_rhs!, cell_tangent!, face_residual!,
              interface_residual!

const ORDINARYDIFFEQ_IMPORT_ERROR = let
  try
    @eval import OrdinaryDiffEq
    nothing
  catch err
    sprint(showerror, err)
  end
end

const ORDINARYDIFFEQ_AVAILABLE = ORDINARYDIFFEQ_IMPORT_ERROR === nothing

# This example solves the compressible Euler equations
#
#   ∂ₜq + ∇ · F(q) = 0,
#
# for the conservative state
#
#   q = (ρ, ρu, ρv, E),
#
# on the quarter domain
#
#   Ω = [0, 1]².
#
# The setup is a quarter-domain reduction of the standard symmetric blast-wave
# problem on `[-1, 1]²`: the faces `x = 0` and `y = 0` are symmetry planes,
# while `x = 1` and `y = 1` are the physical walls of the box. A smooth
# overpressure region is centered at the origin, so the solution remains
# symmetric and the quarter model reproduces the full-box dynamics at lower
# cost.
#
# The file is written as a guided walkthrough for new users:
# 1. choose the physical and numerical parameters,
# 2. define the Euler helper routines and the smooth blast-wave profile,
# 3. assemble the DG mass matrix, projection, and residual operators,
# 4. estimate stable timesteps and build the runtime context,
# 5. adapt the mesh in `h`,
# 6. integrate the semidiscrete system in time, and
# 7. write VTK output and print a compact run summary.
#
# The pressure jump is intentionally smoothed because this example does not yet
# include a positivity limiter. Small density and pressure floors are therefore
# retained as a last line of defense against unphysical roundoff excursions.

# Geometry and DG discretization.
const DOMAIN_ORIGIN = (0.0, 0.0)
const DOMAIN_EXTENT = (1.0, 1.0)
# The blast is centered on the two symmetry planes, so the quarter-domain model
# is exactly the mirrored restriction of the full symmetric blast.
const BLAST_CENTER = DOMAIN_ORIGIN
const ROOT_COUNTS = (16, 16)
const POLYDEG = 1
const QUADRATURE_EXTRA_POINTS = 1

# Gas model and time-integration controls.
const GAMMA = 1.4
const CFL = 0.025
const FINAL_TIME = 4.0
const SAVE_INTERVAL = 0.0125
const ADAPT_INTERVAL = 0.00125 / 2

# Adaptivity controls. This example uses pure `h`-adaptivity and keeps the
# polynomial degree fixed.
const ADAPTIVITY_TOLERANCE = 2.5e-2
const MAX_H_LEVEL = 5

# Physical floors and blast parameters.
const DENSITY_FLOOR = 1.0e-12
const PRESSURE_FLOOR = 1.0e-12
const BACKGROUND_DENSITY = 1.0
const INNER_PRESSURE = 3.0
const OUTER_PRESSURE = 1.0
const BLAST_RADIUS = 0.2
const BLAST_TRANSITION_WIDTH = 0.03
const INITIAL_BLAST_REFINEMENT_LAYERS = 4
const INITIAL_BLAST_REFINEMENT_RADIUS = BLAST_RADIUS + 2.0 * BLAST_TRANSITION_WIDTH

# Output controls.
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 1

# The example auto-runs only when it is executed directly. Tests disable this
# by setting `GRICO_BLAST_WAVE_EULER_AUTORUN=0` before including the file.
const RUN_BLAST_WAVE_EULER = get(ENV, "GRICO_BLAST_WAVE_EULER_AUTORUN", "1") == "1"

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

function require_ordinarydiffeq()
  ORDINARYDIFFEQ_AVAILABLE && return nothing
  message = "This example requires OrdinaryDiffEq.jl on the current Julia LOAD_PATH. " *
            "Install it in your user environment with `julia -e 'using Pkg; " *
            "Pkg.add(\"OrdinaryDiffEq\")'` and rerun the example."
  ORDINARYDIFFEQ_IMPORT_ERROR === nothing ||
    (message *= " OrdinaryDiffEq could not be loaded: $(ORDINARYDIFFEQ_IMPORT_ERROR)")
  throw(ArgumentError(message))
end

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

Smooth quarter-domain blast-wave data on `[0, 1]^2`.

The state starts from rest with uniform density and a pressure bump centered at
the origin. Together with reflective walls on all four faces, this reproduces
the symmetric full-box blast on `[-1, 1]^2`. The transition is smoothed with a
tanh profile so the example remains reasonably robust without a limiter.
"""
function blast_wave_initial_condition(x; gamma=GAMMA, center=BLAST_CENTER,
                                      density=BACKGROUND_DENSITY, inner_pressure=INNER_PRESSURE,
                                      outer_pressure=OUTER_PRESSURE, radius=BLAST_RADIUS,
                                      transition_width=BLAST_TRANSITION_WIDTH)
  transition_width > 0 || throw(ArgumentError("transition_width must be positive"))
  r = hypot(x[1] - center[1], x[2] - center[2])
  blend = 0.5 * (1.0 - tanh((r - radius) / transition_width))
  pressure_value = outer_pressure + (inner_pressure - outer_pressure) * blend
  return conservative_variables(density, (0.0, 0.0), pressure_value, gamma)
end

# Before time integration starts, we seed a few layers of `h`-refinement around
# the blast so the sharp pressure transition is resolved from the first step.
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
  base_max_level = maximum(level(source_grid, leaf, axis)
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

# ---------------------------------------------------------------------------
# 4. Diagnostics, CFL control, and runtime context
# ---------------------------------------------------------------------------

# These diagnostics are cheap global health checks that we print during the run:
# minimum density, minimum pressure, total mass, total energy, and the largest
# local characteristic speed.
function euler_diagnostics(plan, state, field, gamma)
  min_density = Inf
  min_pressure = Inf
  max_wave_speed = 0.0
  total_mass = 0.0
  total_energy = 0.0

  for item in plan.integration.cells
    for point_index in 1:point_count(item)
      q = value(item, state, field, point_index)
      rho, velocity_data, pressure_value = primitive_variables(q, gamma)
      weighted = weight(item, point_index)
      wave_speed = hypot(velocity_data[1], velocity_data[2]) + sqrt(gamma * pressure_value / rho)
      min_density = min(min_density, q[1])
      min_pressure = min(min_pressure,
                         (gamma - 1.0) *
                         (q[4] - 0.5 * (q[2]^2 + q[3]^2) / max(q[1], DENSITY_FLOOR)))
      max_wave_speed = max(max_wave_speed, wave_speed)
      total_mass += q[1] * weighted
      total_energy += q[4] * weighted
    end
  end

  return (; min_density, min_pressure, max_wave_speed, total_mass, total_energy)
end

# The explicit timestep estimate follows the usual DG scaling
#
#   Δt ≈ CFL / max_K ((2p + 1) Σₐ (|uₐ| + c) / hₐ),
#
# with `c = √(γ p / ρ)`.
function suggest_timestep(plan, state, field, gamma; cfl=CFL)
  cfl > 0 || throw(ArgumentError("cfl must be positive"))
  space = field_space(field)
  max_rate = 0.0

  for item in plan.integration.cells
    cell_sizes = ntuple(axis -> cell_size(space.domain, item.leaf, axis), 2)
    degree_value = maximum(cell_degrees(space, item.leaf))
    scale = 2 * degree_value + 1

    for point_index in 1:point_count(item)
      rho, velocity_data, pressure_value = primitive_variables(value(item, state, field,
                                                                     point_index), gamma)
      sound = sqrt(gamma * pressure_value / rho)
      local_rate = scale *
                   sum((abs(velocity_data[axis]) + sound) / cell_sizes[axis] for axis in 1:2)
      max_rate = max(max_rate, local_rate)
    end
  end

  return cfl / max_rate
end

struct CellwiseMassBlock{F,V}
  factorization::F
  global_dofs::V
end

struct CellwiseMassInverse{B,V}
  blocks::B
  workspace::V
end

# On a DG mesh, the mass matrix is block diagonal by cell. We exploit that
# structure and invert one small dense block per active cell.
function local_mass_matrix(values::CellValues, field)
  mode_count = local_mode_count(values, field)
  components = component_count(field)
  local_matrix = zeros(eltype(weight(values, 1)), components * mode_count, components * mode_count)
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
          local_matrix[row, col] += contribution
        end
      end
    end
  end

  return local_matrix
end

function build_cellwise_mass_inverse(plan, field)
  T = eltype(weight(first(plan.integration.cells), 1))
  blocks = CellwiseMassBlock{Cholesky{T,Matrix{T}},Vector{Int}}[]
  max_local_dofs = 0

  for item in plan.integration.cells
    local_dofs = item.single_term_indices
    all(isone, item.single_term_coefficients) ||
      throw(ArgumentError("cellwise mass inverse requires one-term DG dofs"))
    local_matrix = local_mass_matrix(item, field)
    push!(blocks, CellwiseMassBlock(cholesky(Symmetric(local_matrix)), collect(local_dofs)))
    max_local_dofs = max(max_local_dofs, length(local_dofs))
  end

  return CellwiseMassInverse(blocks, zeros(T, max_local_dofs))
end

# Apply the precomputed cellwise inverse to the assembled residual to obtain
# the time derivative `dq_h/dt`.
function apply_mass_inverse!(du, mass_inverse::CellwiseMassInverse, rhs)
  fill!(du, zero(eltype(du)))

  for block in mass_inverse.blocks
    local_dof_count = length(block.global_dofs)
    local_rhs = view(mass_inverse.workspace, 1:local_dof_count)

    for local_dof in 1:local_dof_count
      local_rhs[local_dof] = rhs[block.global_dofs[local_dof]]
    end

    ldiv!(block.factorization, local_rhs)

    for local_dof in 1:local_dof_count
      du[block.global_dofs[local_dof]] = local_rhs[local_dof]
    end
  end

  return du
end

# Bundle the runtime objects that the driver updates after each solve/adapt
# segment. The context carries both the DG state and the cheap diagnostics
# derived from it.
function blast_wave_context(conserved, state; gamma=GAMMA, cfl=CFL, degree=POLYDEG)
  spatial_plan = euler_residual_plan(conserved, gamma)
  mass_inverse = build_cellwise_mass_inverse(spatial_plan, conserved)
  diagnostics = euler_diagnostics(spatial_plan, state, conserved, gamma)
  dt = suggest_timestep(spatial_plan, state, conserved, gamma; cfl)
  return (; domain=field_space(conserved).domain, space=field_space(conserved), conserved, gamma,
          cfl, degree, mass_inverse, spatial_plan, state, diagnostics, dt)
end

function refresh_blast_wave_context(context, state=context.state)
  diagnostics = euler_diagnostics(context.spatial_plan, state, context.conserved, context.gamma)
  dt = suggest_timestep(context.spatial_plan, state, context.conserved, context.gamma;
                        cfl=context.cfl)
  return merge(context, (; state, diagnostics, dt))
end

function build_blast_wave_euler_context(; root_counts=ROOT_COUNTS, degree=POLYDEG,
                                        quadrature_extra_points=QUADRATURE_EXTRA_POINTS,
                                        gamma=GAMMA, cfl=CFL,
                                        initial_refinement_layers=INITIAL_BLAST_REFINEMENT_LAYERS,
                                        initial_refinement_radius=INITIAL_BLAST_REFINEMENT_RADIUS)
  domain = Domain(DOMAIN_ORIGIN, DOMAIN_EXTENT, root_counts)
  coarse_space = HpSpace(domain,
                         SpaceOptions(basis=TrunkBasis(), degree=UniformDegree(degree),
                                      quadrature=DegreePlusQuadrature(quadrature_extra_points),
                                      continuity=:dg))
  space = pre_refine_blast_wave_space(coarse_space; layers=initial_refinement_layers,
                                      center=BLAST_CENTER, radius=initial_refinement_radius)
  conserved = VectorField(space, 4; name=:conserved)
  initial_state = project_initial_condition(conserved,
                                            x -> blast_wave_initial_condition(x; gamma,
                                                                              center=BLAST_CENTER))
  return blast_wave_context(conserved, initial_state; gamma, cfl, degree)
end

# The example follows the library-level multiresolution planner: one normalized
# tolerance controls both refinement and derefinement, while fixed degree limits
# make this a pure `h` experiment.
function blast_wave_adaptivity_plan(context; tolerance=ADAPTIVITY_TOLERANCE,
                                    max_h_level=MAX_H_LEVEL)
  limits = AdaptivityLimits(context.space; min_p=context.degree, max_p=context.degree,
                            max_h_level=max_h_level)
  return adaptivity_plan(context.state, context.conserved; tolerance=tolerance, limits=limits)
end

# Mesh adaptation is performed between fixed-mesh ODE segments. The current DG
# state is used as an error indicator, then transferred to the adapted mesh.
function adapt_blast_wave_context(context; tolerance=ADAPTIVITY_TOLERANCE, max_h_level=MAX_H_LEVEL,
                                  linear_solve=direct_sparse_solve)
  plan = blast_wave_adaptivity_plan(context; tolerance=tolerance, max_h_level=max_h_level)

  if isempty(plan)
    return context, plan
  end

  space_transition = transition(plan)
  new_conserved = adapted_field(space_transition, context.conserved)
  new_state = transfer_state(space_transition, context.state, context.conserved, new_conserved;
                             linear_solve=linear_solve)
  return blast_wave_context(new_conserved, new_state; gamma=context.gamma, cfl=context.cfl,
                            degree=context.degree), plan
end

# `EulerSemidiscretization` is the object handed to `OrdinaryDiffEq.jl`. It owns
# the reusable work arrays needed to evaluate `du = M⁻¹ R(u)`.
struct EulerSemidiscretization{P,F,S,V,W,M}
  plan::P
  field::F
  state::S
  residual_vector::V
  residual_workspace::W
  mass_inverse::M
end

function EulerSemidiscretization(plan, field, initial_state, mass_inverse)
  runtime_state = State(plan, copy(coefficients(initial_state)))
  residual_vector = similar(coefficients(initial_state))
  residual_workspace = ResidualWorkspace(plan)
  return EulerSemidiscretization(plan, field, runtime_state, residual_vector, residual_workspace,
                                 mass_inverse)
end

# Copy the ODE state into the reusable `State`, assemble the DG residual, and
# apply the cellwise inverse mass matrix.
function euler_rhs!(du, u, semi::EulerSemidiscretization, t)
  semi.state.coefficients .= u
  residual!(semi.residual_vector, semi.plan, semi.state, semi.residual_workspace)
  apply_mass_inverse!(du, semi.mass_inverse, semi.residual_vector)
  return nothing
end

# Integrate one time segment on a fixed mesh. Mesh changes happen only between
# segments so that the ODE solve itself sees a time-independent semidiscrete
# operator.
function solve_fixed_mesh_segment(context, tspan; solver=nothing, return_solution=false)
  semi = EulerSemidiscretization(context.spatial_plan, context.conserved, context.state,
                                 context.mass_inverse)
  problem = OrdinaryDiffEq.ODEProblem(euler_rhs!, copy(coefficients(context.state)), tspan, semi)
  resolved_solver = solver === nothing ?
                    OrdinaryDiffEq.CarpenterKennedy2N54(williamson_condition=false) : solver
  step_size = min(context.dt, tspan[2] - tspan[1])
  solution = OrdinaryDiffEq.solve(problem, resolved_solver; dt=step_size, adaptive=false,
                                  tstops=[tspan[2]], saveat=[tspan[2]], save_start=false,
                                  save_everystep=false)
  final_state = State(context.spatial_plan, copy(only(solution.u)))
  return return_solution ? (final_state, solution) : (final_state, nothing)
end

function saved_times(tspan, save_interval)
  save_interval > 0 || throw(ArgumentError("save_interval must be positive"))
  first_time, final_time = tspan
  times = collect(first_time:save_interval:final_time)
  isempty(times) && return [first_time, final_time]
  first(times) == first_time || pushfirst!(times, first_time)
  isapprox(last(times), final_time; atol=100 * eps(final_time + 1.0)) || push!(times, final_time)
  return times
end

@inline function same_time(a, b)
  scale = max(abs(a), abs(b)) + 1.0
  return isapprox(a, b; atol=100 * eps(scale))
end

function merge_time_grids(grids...)
  merged = sort!(vcat(grids...))
  unique_times = Float64[]

  for time in merged
    (isempty(unique_times) || !same_time(time, unique_times[end])) && push!(unique_times, time)
  end

  return unique_times
end

sampled_conserved(values, field) = getproperty(values, field_name(field))

function blast_wave_history_entry(step, time, context, initial_diagnostics)
  diagnostics = context.diagnostics
  mass_scale = max(abs(initial_diagnostics.total_mass), 1.0)
  energy_scale = max(abs(initial_diagnostics.total_energy), 1.0)
  return (; step, time=Float64(time), active_leaves=active_leaf_count(context.space),
          dofs=length(coefficients(context.state)), min_density=diagnostics.min_density,
          min_pressure=diagnostics.min_pressure, max_wave_speed=diagnostics.max_wave_speed,
          total_mass=diagnostics.total_mass, total_energy=diagnostics.total_energy,
          relative_mass_drift=(diagnostics.total_mass - initial_diagnostics.total_mass) /
                              mass_scale,
          relative_energy_drift=(diagnostics.total_energy - initial_diagnostics.total_energy) /
                                energy_scale)
end

function blast_wave_adaptivity_limits(context; max_h_level=MAX_H_LEVEL)
  return AdaptivityLimits(context.space; min_p=context.degree, max_p=context.degree,
                          max_h_level=max_h_level)
end

function blast_wave_adaptivity_entry(step, time, before_context, after_context, plan)
  summary = adaptivity_summary(plan)

  return (; step, time=Float64(time), before_active_leaves=active_leaf_count(before_context.space),
          after_active_leaves=active_leaf_count(after_context.space),
          before_dofs=length(coefficients(before_context.state)),
          after_dofs=length(coefficients(after_context.state)),
          marked_leaf_count=summary.marked_leaf_count,
          h_refinement_leaf_count=summary.h_refinement_leaf_count,
          h_derefinement_cell_count=summary.h_derefinement_cell_count,
          p_refinement_leaf_count=summary.p_refinement_leaf_count,
          p_derefinement_leaf_count=summary.p_derefinement_leaf_count)
end

# Export the active refinement signal as leaf-wise cell data so it is easy to
# inspect where the current adaptive mesh logic sees under-resolution.
function blast_wave_refinement_indicator_data(context; max_h_level=MAX_H_LEVEL)
  limits = blast_wave_adaptivity_limits(context; max_h_level=max_h_level)
  indicators = Grico.multiresolution_indicators(context.state, context.conserved; limits=limits)
  axis_values = Matrix{Float64}(undef, dimension(context.space), length(indicators))
  norms = Vector{Float64}(undef, length(indicators))

  for leaf_index in eachindex(indicators)
    values = indicators[leaf_index]
    norms[leaf_index] = sqrt(sum(abs2, values))

    for axis in 1:length(values)
      axis_values[axis, leaf_index] = Float64(values[axis])
    end
  end

  return axis_values, norms
end

# Export the current DG state. The point data expose the main physical fields a
# new user typically wants to inspect first: density, velocity, and pressure,
# while the cell data carry leaf metadata and the current adaptivity signal.
function write_blast_wave_vtk(context, entry; output_directory=joinpath(@__DIR__, "output"),
                              max_h_level=MAX_H_LEVEL)
  current_grid = grid(context.domain)
  refinement_indicator, refinement_indicator_norm = blast_wave_refinement_indicator_data(context;
                                                                                         max_h_level=max_h_level)
  return write_vtk(joinpath(output_directory, @sprintf("blast_wave_euler_%04d", entry.step)),
                   context.state; fields=(context.conserved,),
                   point_data=(density=(x, values) -> sampled_conserved(values, context.conserved)[1],
                               velocity=(x, values) -> begin
                                 q = sampled_conserved(values, context.conserved)
                                 velocity(q, context.gamma)
                               end,
                               pressure=(x, values) -> begin
                                 q = sampled_conserved(values, context.conserved)
                                 pressure(q, context.gamma)
                               end),
                   cell_data=(leaf=leaf -> Float64(leaf),
                              level=leaf -> Float64.(level(current_grid, leaf)),
                              degree=leaf -> Float64.(cell_degrees(context.space, leaf)),
                              refinement_indicator=refinement_indicator,
                              refinement_indicator_norm=refinement_indicator_norm),
                   field_data=(time=entry.time, min_density=entry.min_density,
                               min_pressure=entry.min_pressure,
                               relative_mass_drift=entry.relative_mass_drift,
                               relative_energy_drift=entry.relative_energy_drift),
                   subdivisions=EXPORT_SUBDIVISIONS, export_degree=EXPORT_DEGREE, append=true,
                   compress=true, ascii=false)
end

# Print a compact run header so the solver configuration is visible without
# opening the file again while the example is running.
function print_blast_wave_header(context, final_time; save_interval=SAVE_INTERVAL,
                                 adapt_interval=ADAPT_INTERVAL,
                                 adaptivity_tolerance=ADAPTIVITY_TOLERANCE, max_h_level=MAX_H_LEVEL,
                                 initial_refinement_layers=INITIAL_BLAST_REFINEMENT_LAYERS,
                                 initial_refinement_radius=INITIAL_BLAST_REFINEMENT_RADIUS)
  println("blast_wave_euler.jl")
  @printf("  domain             : [%.1f, %.1f] x [%.1f, %.1f] (quarter model with symmetry)\n",
          origin(context.domain, 1), origin(context.domain, 1) + extent(context.domain, 1),
          origin(context.domain, 2), origin(context.domain, 2) + extent(context.domain, 2))
  @printf("  roots              : %d x %d\n", root_cell_counts(grid(context.domain))...)
  @printf("  base degree        : %d\n", context.degree)
  @printf("  cfl                : %.3f\n", context.cfl)
  @printf("  blast radius       : %.3f\n", BLAST_RADIUS)
  @printf("  p_in / p_out       : %.2f / %.2f\n", INNER_PRESSURE, OUTER_PRESSURE)
  @printf("  transition width   : %.3f\n", BLAST_TRANSITION_WIDTH)
  @printf("  initial h layers   : %d\n", initial_refinement_layers)
  @printf("  initial ref radius : %.3f\n", initial_refinement_radius)
  @printf("  save interval      : %.3f\n", save_interval)
  @printf("  adapt interval     : %.3f\n", adapt_interval)
  @printf("  max h level        : %d\n", max_h_level)
  @printf("  adapt tolerance    : %.2e\n", adaptivity_tolerance)

  @printf("  final time         : %.3f\n", final_time)
  println("  step time leaves dofs min(rho) min(p) rel-mass rel-energy max-wave")
end

function print_blast_wave_history_entry(entry)
  @printf("  %4d %.3f %5d %5d %.6e %.6e %.6e %.6e %.6e\n", entry.step, entry.time,
          entry.active_leaves, entry.dofs, entry.min_density, entry.min_pressure,
          entry.relative_mass_drift, entry.relative_energy_drift, entry.max_wave_speed)
end

# ---------------------------------------------------------------------------
# 5. Time integration, output, and driver
# ---------------------------------------------------------------------------

# This is the user-facing entry point of the file. It alternates between
#
# 1. fixed-mesh ODE solves,
# 2. optional output writes, and
# 3. optional `h`-adaptation.
#
# The history table printed to the terminal lets users monitor the most
# important conservation and positivity diagnostics while the example runs.
function run_blast_wave_euler_example(; root_counts=ROOT_COUNTS, degree=POLYDEG,
                                      quadrature_extra_points=QUADRATURE_EXTRA_POINTS, gamma=GAMMA,
                                      cfl=CFL, final_time=FINAL_TIME, save_interval=SAVE_INTERVAL,
                                      adapt_interval=ADAPT_INTERVAL,
                                      initial_refinement_layers=INITIAL_BLAST_REFINEMENT_LAYERS,
                                      initial_refinement_radius=INITIAL_BLAST_REFINEMENT_RADIUS,
                                      solver=nothing, adaptivity_tolerance=ADAPTIVITY_TOLERANCE,
                                      max_h_level=MAX_H_LEVEL, store_segment_solutions=false,
                                      write_vtk=WRITE_VTK, print_summary=true)
  require_ordinarydiffeq()
  final_time > 0 || throw(ArgumentError("final_time must be positive"))
  max_h_level >= initial_refinement_layers ||
    throw(ArgumentError("max_h_level must be at least initial_refinement_layers"))
  context = build_blast_wave_euler_context(; root_counts, degree, quadrature_extra_points, gamma,
                                           cfl, initial_refinement_layers,
                                           initial_refinement_radius)
  initial_diagnostics = context.diagnostics
  save_times = saved_times((0.0, final_time), save_interval)
  adapt_times = saved_times((0.0, final_time), adapt_interval)
  times = merge_time_grids(save_times, adapt_times)
  history = NamedTuple[blast_wave_history_entry(0, 0.0, context, initial_diagnostics)]
  adaptivity_history = NamedTuple[]
  segment_solutions = store_segment_solutions ? Any[] : nothing
  vtk_files = String[]
  pvd_path = nothing
  save_index = 2
  adapt_index = 2

  write_vtk && mkpath(joinpath(@__DIR__, "output"))
  write_vtk && push!(vtk_files, write_blast_wave_vtk(context, history[1]; max_h_level=max_h_level))

  if print_summary
    print_blast_wave_header(context, final_time; save_interval=save_interval,
                            adapt_interval=adapt_interval,
                            initial_refinement_layers=initial_refinement_layers,
                            initial_refinement_radius=initial_refinement_radius,
                            adaptivity_tolerance=adaptivity_tolerance, max_h_level=max_h_level)
    print_blast_wave_history_entry(history[1])
  end

  for step in 1:(length(times)-1)
    segment_state, segment_solution = solve_fixed_mesh_segment(context,
                                                               (times[step], times[step + 1]);
                                                               solver=solver,
                                                               return_solution=store_segment_solutions)
    store_segment_solutions && push!(segment_solutions, segment_solution)
    context = refresh_blast_wave_context(context, segment_state)

    while save_index <= length(save_times) && same_time(save_times[save_index], times[step + 1])
      entry = blast_wave_history_entry(length(history), save_times[save_index], context,
                                       initial_diagnostics)
      push!(history, entry)
      print_summary && print_blast_wave_history_entry(entry)
      write_vtk && push!(vtk_files, write_blast_wave_vtk(context, entry; max_h_level=max_h_level))
      save_index += 1
    end

    while adapt_index <= length(adapt_times) && same_time(adapt_times[adapt_index], times[step + 1])
      if !same_time(adapt_times[adapt_index], final_time)
        before_context = context
        context, adaptivity_plan = adapt_blast_wave_context(before_context;
                                                            tolerance=adaptivity_tolerance,
                                                            max_h_level=max_h_level)
        adaptivity_entry = blast_wave_adaptivity_entry(step, adapt_times[adapt_index],
                                                       before_context, context, adaptivity_plan)
        push!(adaptivity_history, adaptivity_entry)
      end
      adapt_index += 1
    end
  end

  if write_vtk
    pvd_path = write_pvd(joinpath(@__DIR__, "output", "blast_wave_euler.pvd"), vtk_files;
                         timesteps=[entry.time for entry in history])
    print_summary && println("  vtk  $(last(vtk_files))")
    print_summary && println("  pvd  $pvd_path")
  end

  return (; context, history, adaptivity_history, vtk_files, pvd_path, segment_solutions)
end

RUN_BLAST_WAVE_EULER && run_blast_wave_euler_example()

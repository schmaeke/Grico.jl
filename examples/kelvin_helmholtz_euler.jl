const APPLEACCELERATE_AVAILABLE = let
  try
    @eval import AppleAccelerate
    true
  catch err
    err isa ArgumentError || rethrow()
    false
  end
end

const THREADPINNING_AVAILABLE = let
  try
    @eval import ThreadPinning
    true
  catch err
    err isa ArgumentError || rethrow()
    false
  end
end

THREADPINNING_AVAILABLE && ThreadPinning.pinthreads(:cores)

using LinearAlgebra
using Printf
using Grico
import Grico: cell_matrix!, cell_residual!, cell_rhs!, cell_tangent!, interface_residual!

const ORDINARYDIFFEQ_AVAILABLE = let
    try
        @eval import OrdinaryDiffEq
        true
    catch err
        err isa ArgumentError || rethrow()
        false
    end
end

# This example is the package's first transient "real-world" DG demo.
#
# It solves the compressible Euler equations
#
#   ∂ₜq + ∇ · F(q) = 0,   q = (ρ, ρu, ρv, E),
#
# on the periodic box
#
#   Ω = [-1, 1]²,
#
# using one equal-order discontinuous Galerkin space for all four conservative
# variables. The physical setup follows the smooth Kelvin-Helmholtz test case
# used in
#
# - Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
#   A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM
#   Discretizations of the Euler Equations
#   [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
#
# The goal here is not to reproduce Trixi's specific DGSEM flux, shock
# capturing, and AMR stack. Instead, the example shows how to assemble a
# semidiscrete nonlinear DG operator in Grico, project physically meaningful
# initial data, and then hand the resulting ODE system to SciML's
# `OrdinaryDiffEq.jl` when that optional package is present on the user's Julia
# `LOAD_PATH`.
#
# Mesh adaptation is intentionally kept simple and readable: the example solves
# between the union of save and adaptation checkpoints, saves the physical state
# on one schedule, asks Grico for one mixed `hp`-adaptivity plan on another,
# combines the default DG jump refinement indicator with modal smoothness-based
# `h`/`p` decisions, allows both `h`- and `p`-derefinement, transfers the
# solution to the new mesh, and then continues time integration on that adapted
# space.
#
# Since this example intentionally omits a positivity limiter, the Euler helper
# routines apply tiny density/pressure floors inside the flux evaluation to keep
# the demo robust on coarse meshes. The printed diagnostics still report the
# unfloored minima seen by the discrete polynomial.
#
# File roadmap:
#
# 1. choose physical and DG parameters,
# 2. define the conservative/primitive Euler helpers,
# 3. define the cell/interface DG residual and the consistent mass matrix,
# 4. project the Kelvin-Helmholtz initial condition,
# 5. build a semidiscrete ODE right-hand side,
# 6. add mixed `hp`-adaptation with refinement and derefinement,
# 7. integrate successive fixed-mesh ODE segments with `OrdinaryDiffEq.jl` and
#    export time snapshots to VTK.

const ROOT_COUNTS = (16, 16)
const POLYDEG = 3
const QUADRATURE_EXTRA_POINTS = 2
const GAMMA = 1.4
const CFL = 0.06
const FINAL_TIME = 4.0
const SAVE_INTERVAL = 0.1
const ADAPT_INTERVAL = 0.2
const ADAPTIVITY_THRESHOLD = 0.8
const SMOOTHNESS_THRESHOLD = 0.15
const P_COARSENING_THRESHOLD = 2.5e-2
const H_COARSENING_THRESHOLD = 2.5e-4
const MAX_H_LEVEL = 2
const MIN_P_DEGREE = POLYDEG - 1
const MAX_P_DEGREE = POLYDEG + 1
const DENSITY_FLOOR = 1.0e-12
const PRESSURE_FLOOR = 1.0e-12
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 4
const RUN_KELVIN_HELMHOLTZ_EULER = get(ENV, "GRICO_KH_EULER_AUTORUN", "1") == "1"

# ---------------------------------------------------------------------------
# 1. Small Euler helpers
# ---------------------------------------------------------------------------

@inline dot2(a, b) = a[1] * b[1] + a[2] * b[2]
@inline squared_norm2(a) = dot2(a, a)
@inline component_local_index(mode_count::Int, component::Int, mode::Int) = (component - 1) * mode_count +
                                                                             mode

function require_ordinarydiffeq()
    ORDINARYDIFFEQ_AVAILABLE && return nothing
    throw(ArgumentError("This example requires OrdinaryDiffEq.jl on the current Julia LOAD_PATH. " *
                      "Install it in your user environment with `julia -e 'using Pkg; " *
                      "Pkg.add(\"OrdinaryDiffEq\")'` and rerun the example."))
end

@inline function primitive_variables(q, gamma)
    rho = max(q[1], DENSITY_FLOOR)
    inv_rho = inv(rho)
    velocity = (q[2] * inv_rho, q[3] * inv_rho)
    kinetic = 0.5 * (q[2] * velocity[1] + q[3] * velocity[2])
    pressure = max((gamma - 1.0) * (q[4] - kinetic), PRESSURE_FLOOR)
    return rho, velocity, pressure
end

@inline pressure(q, gamma) = primitive_variables(q, gamma)[3]
@inline velocity(q, gamma) = primitive_variables(q, gamma)[2]
@inline function sound_speed(q, gamma)
    rho, _, pressure_value = primitive_variables(q, gamma)
    return sqrt(gamma * pressure_value / rho)
end

@inline function conservative_variables(rho, velocity_data, pressure_value, gamma)
    kinetic = 0.5 * rho * squared_norm2(velocity_data)
    energy = pressure_value / (gamma - 1.0) + kinetic
    return (rho, rho * velocity_data[1], rho * velocity_data[2], energy)
end

function euler_flux(q, gamma)
    _, velocity_data, pressure_value = primitive_variables(q, gamma)
    velocity1 = velocity_data[1]
    velocity2 = velocity_data[2]
    momentum1 = q[2]
    momentum2 = q[3]
    energy = q[4]
    return ((momentum1, momentum2),
          (momentum1 * velocity1 + pressure_value, momentum1 * velocity2),
          (momentum2 * velocity1, momentum2 * velocity2 + pressure_value),
          ((energy + pressure_value) * velocity1, (energy + pressure_value) * velocity2))
end

function flux_dot_normal(q, normal_data, gamma)
    _, velocity_data, pressure_value = primitive_variables(q, gamma)
    normal_velocity = dot2(velocity_data, normal_data)
    energy = q[4]
    return (q[1] * normal_velocity,
          q[2] * normal_velocity + pressure_value * normal_data[1],
          q[3] * normal_velocity + pressure_value * normal_data[2],
          (energy + pressure_value) * normal_velocity)
end

function lax_friedrichs_flux(q_minus, q_plus, normal_data, gamma)
    flux_minus = flux_dot_normal(q_minus, normal_data, gamma)
    flux_plus = flux_dot_normal(q_plus, normal_data, gamma)
    speed_minus = abs(dot2(velocity(q_minus, gamma), normal_data)) + sound_speed(q_minus, gamma)
    speed_plus = abs(dot2(velocity(q_plus, gamma), normal_data)) + sound_speed(q_plus, gamma)
    alpha = max(speed_minus, speed_plus)
    return ntuple(component -> 0.5 * (flux_minus[component] + flux_plus[component] -
                                    alpha * (q_plus[component] - q_minus[component])), 4)
end

"""
    kelvin_helmholtz_initial_condition(x; gamma=GAMMA)

Smooth Kelvin-Helmholtz instability data on `[-1, 1]^2`.

This follows the setup quoted above from Rueda-Ramírez and Gassner
([arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)), but the surrounding
DG discretization is Grico's own equal-order modal DG formulation.
"""
function kelvin_helmholtz_initial_condition(x; gamma=GAMMA)
  slope = 15.0
  shear_profile = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * shear_profile
  velocity_data = (0.5 * (shear_profile - 1.0), 0.1 * sinpi(2.0 * x[1]))
  pressure_value = 1.0
  return conservative_variables(rho, velocity_data, pressure_value, gamma)
end

# ---------------------------------------------------------------------------
# 2. Mass matrix and initial projection
# ---------------------------------------------------------------------------

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

cell_matrix!(local_matrix, operator::MassMatrix, values::CellValues) =
  _assemble_mass_block!(local_matrix, operator.field, values)
cell_tangent!(local_matrix, operator::MassMatrix, values::CellValues, state::State) =
  _assemble_mass_block!(local_matrix, operator.field, values)

struct ProjectionSource{F,G}
  field::F
  data::G
end

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

function mass_tangent_plan(field)
  problem = ResidualProblem(field)
  add_cell!(problem, MassMatrix(field))
  return compile(problem)
end

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

struct CompressibleEulerVolume{F,T}
  field::F
  gamma::T
end

struct CompressibleEulerInterface{F,T}
  field::F
  gamma::T
end

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

function euler_residual_plan(field, gamma)
  problem = ResidualProblem(field)
  add_cell!(problem, CompressibleEulerVolume(field, gamma))
  add_interface!(problem, CompressibleEulerInterface(field, gamma))
  return compile(problem)
end

# ---------------------------------------------------------------------------
# 4. Context builders and diagnostics
# ---------------------------------------------------------------------------

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
      min_pressure = min(min_pressure, (gamma - 1.0) * (q[4] - 0.5 * (q[2]^2 + q[3]^2) /
                                                                max(q[1], DENSITY_FLOOR)))
      max_wave_speed = max(max_wave_speed, wave_speed)
      total_mass += q[1] * weighted
      total_energy += q[4] * weighted
    end
  end

  return (; min_density, min_pressure, max_wave_speed, total_mass, total_energy)
end

function suggest_timestep(plan, state, field, gamma; cfl=CFL)
  cfl > 0 || throw(ArgumentError("cfl must be positive"))
  space = field_space(field)
  max_rate = 0.0

  for item in plan.integration.cells
    cell_sizes = ntuple(axis -> cell_size(space.domain, item.leaf, axis), 2)
    degree_value = maximum(cell_degrees(space, item.leaf))
    scale = 2 * degree_value + 1

    for point_index in 1:point_count(item)
      rho, velocity_data, pressure_value = primitive_variables(value(item, state, field, point_index),
                                                               gamma)
      sound = sqrt(gamma * pressure_value / rho)
      local_rate = scale * sum((abs(velocity_data[axis]) + sound) / cell_sizes[axis] for axis in 1:2)
      max_rate = max(max_rate, local_rate)
    end
  end

  return cfl / max_rate
end

function kelvin_helmholtz_context(conserved, state; gamma=GAMMA, cfl=CFL, degree=POLYDEG)
  mass_plan = mass_tangent_plan(conserved)
  mass_factorization = let
    mass_matrix = tangent(mass_plan, State(mass_plan))
    cholesky(Symmetric(mass_matrix))
  end
  spatial_plan = euler_residual_plan(conserved, gamma)
  diagnostics = euler_diagnostics(spatial_plan, state, conserved, gamma)
  dt = suggest_timestep(spatial_plan, state, conserved, gamma; cfl)
  return (; domain=field_space(conserved).domain, space=field_space(conserved), conserved, gamma, cfl,
          degree, mass_factorization, spatial_plan, state, diagnostics, dt)
end

function refresh_kelvin_helmholtz_context(context, state=context.state)
  diagnostics = euler_diagnostics(context.spatial_plan, state, context.conserved, context.gamma)
  dt = suggest_timestep(context.spatial_plan, state, context.conserved, context.gamma; cfl=context.cfl)
  return merge(context, (; state, diagnostics, dt))
end

function build_kelvin_helmholtz_euler_context(; root_counts=ROOT_COUNTS, degree=POLYDEG,
                                              quadrature_extra_points=QUADRATURE_EXTRA_POINTS,
                                              gamma=GAMMA, cfl=CFL)
  domain = Domain((-1.0, -1.0), (2.0, 2.0), root_counts; periodic=true)
  space = HpSpace(domain,
                  SpaceOptions(basis=TrunkBasis(), degree=UniformDegree(degree),
                               quadrature=DegreePlusQuadrature(quadrature_extra_points),
                               continuity=:dg))
  conserved = VectorField(space, 4; name=:conserved)
  initial_state = project_initial_condition(conserved,
                                            x -> kelvin_helmholtz_initial_condition(x; gamma))
  return kelvin_helmholtz_context(conserved, initial_state; gamma, cfl, degree)
end

function adapt_kelvin_helmholtz_context(context; threshold=ADAPTIVITY_THRESHOLD,
                                        smoothness_threshold=SMOOTHNESS_THRESHOLD,
                                        p_coarsening_threshold=P_COARSENING_THRESHOLD,
                                        h_coarsening_threshold=H_COARSENING_THRESHOLD,
                                        max_h_level=MAX_H_LEVEL, min_p_degree=MIN_P_DEGREE, max_p_degree=MAX_P_DEGREE,
                                        linear_solve=direct_sparse_solve)
  max_p_degree >= context.degree ||
    throw(ArgumentError("max_p_degree must be at least the base degree $(context.degree)"))
  limits = AdaptivityLimits(context.space; min_p=context.degree, max_h_level=max_h_level,
                            max_p=max_p_degree)
  plan = hp_adaptivity_plan(context.state, context.conserved; threshold=threshold,
                            smoothness_threshold=smoothness_threshold,
                            p_coarsening_threshold=p_coarsening_threshold,
                            h_coarsening_threshold=h_coarsening_threshold, limits=limits)

  if isempty(plan)
    return context, plan
  end

  space_transition = transition(plan)
  new_conserved = adapted_field(space_transition, context.conserved)
  new_state = transfer_state(space_transition, context.state, context.conserved, new_conserved;
                             linear_solve=linear_solve)
  return kelvin_helmholtz_context(new_conserved, new_state; gamma=context.gamma, cfl=context.cfl,
                                  degree=context.degree), plan
end

struct EulerSemidiscretization{P,F,S,V,M}
  plan::P
  field::F
  state::S
  residual_vector::V
  mass_factorization::M
end

function EulerSemidiscretization(plan, field, initial_state, mass_factorization)
  runtime_state = State(plan, copy(coefficients(initial_state)))
  residual_vector = similar(coefficients(initial_state))
  return EulerSemidiscretization(plan, field, runtime_state, residual_vector, mass_factorization)
end

function euler_rhs!(du, u, semi::EulerSemidiscretization, t)
  semi.state.coefficients .= u
  residual!(semi.residual_vector, semi.plan, semi.state)
  ldiv!(du, semi.mass_factorization, semi.residual_vector)
  return nothing
end

function solve_fixed_mesh_segment(context, tspan; solver=nothing)
  semi = EulerSemidiscretization(context.spatial_plan, context.conserved, context.state,
                                 context.mass_factorization)
  problem = OrdinaryDiffEq.ODEProblem(euler_rhs!, copy(coefficients(context.state)), tspan, semi)
  resolved_solver = solver === nothing ? OrdinaryDiffEq.CarpenterKennedy2N54(williamson_condition=false) : solver
  step_size = min(context.dt, tspan[2] - tspan[1])
  solution = OrdinaryDiffEq.solve(problem, resolved_solver; dt=step_size, adaptive=false,
                                  tstops=[tspan[2]], saveat=[tspan[2]], save_start=false,
                                  save_everystep=false)
  final_state = State(context.spatial_plan, copy(only(solution.u)))
  return final_state, semi, solution
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

function history_entry(step, time, context, initial_diagnostics)
  diagnostics = context.diagnostics
  mass_scale = max(abs(initial_diagnostics.total_mass), 1.0)
  energy_scale = max(abs(initial_diagnostics.total_energy), 1.0)
  return (; step, time=Float64(time), active_leaves=active_leaf_count(context.space),
          dofs=length(coefficients(context.state)), min_density=diagnostics.min_density,
          min_pressure=diagnostics.min_pressure, max_wave_speed=diagnostics.max_wave_speed,
          total_mass=diagnostics.total_mass, total_energy=diagnostics.total_energy,
          relative_mass_drift=(diagnostics.total_mass - initial_diagnostics.total_mass) / mass_scale,
          relative_energy_drift=(diagnostics.total_energy - initial_diagnostics.total_energy) /
energy_scale)
end

function write_kelvin_helmholtz_vtk(context, entry; output_directory=joinpath(@__DIR__, "output"))
  current_grid = grid(context.domain)
  return write_vtk(joinpath(output_directory, @sprintf("kelvin_helmholtz_euler_%04d", entry.step)),
                   context.state; fields=(context.conserved,),
                   point_data=(density = (x, values) -> sampled_conserved(values, context.conserved)[1],
                               velocity = (x, values) -> begin
                                 q = sampled_conserved(values, context.conserved)
                                 velocity(q, context.gamma)
                               end,
                               pressure = (x, values) -> begin
                                 q = sampled_conserved(values, context.conserved)
                                 pressure(q, context.gamma)
                               end),
                   cell_data=(leaf = leaf -> Float64(leaf),
                              level = leaf -> Float64.(level(current_grid, leaf)),
                              degree = leaf -> Float64.(cell_degrees(context.space, leaf))),
                   field_data=(time = entry.time, min_density = entry.min_density,
                               min_pressure = entry.min_pressure,
                               relative_mass_drift = entry.relative_mass_drift,
                               relative_energy_drift = entry.relative_energy_drift),
                   subdivisions=EXPORT_SUBDIVISIONS, export_degree=EXPORT_DEGREE, append=true,
                   compress=true, ascii=false)
end

function print_kelvin_helmholtz_header(context, final_time; save_interval=SAVE_INTERVAL,
                                       adapt_interval=ADAPT_INTERVAL,
                                       adaptivity_threshold=ADAPTIVITY_THRESHOLD,
                                       smoothness_threshold=SMOOTHNESS_THRESHOLD,
                                       p_coarsening_threshold=P_COARSENING_THRESHOLD,
                                       h_coarsening_threshold=H_COARSENING_THRESHOLD,
                                       max_h_level=MAX_H_LEVEL,
                                       max_p_degree=MAX_P_DEGREE)
  println("kelvin_helmholtz_euler.jl")
  @printf("  roots              : %d x %d\n", root_cell_counts(grid(context.domain))...)
  @printf("  base degree        : %d\n", context.degree)
  @printf("  max p degree       : %d\n", max_p_degree)
  @printf("  cfl                : %.3f\n", context.cfl)
  @printf("  save interval      : %.3f\n", save_interval)
  @printf("  adapt interval     : %.3f\n", adapt_interval)
  @printf("  max h level        : %d\n", max_h_level)
  @printf("  refinement thresh. : %.2f\n", adaptivity_threshold)
  @printf("  smoothness thresh. : %.2f\n", smoothness_threshold)
  @printf("  p coarsening thr.  : %.2e\n", p_coarsening_threshold)
  @printf("  h coarsening thr.  : %.2e\n", h_coarsening_threshold)
  @printf("  final time         : %.3f\n", final_time)
  println("  step time leaves dofs min(rho) min(p) rel-mass rel-energy max-wave")
end

function print_kelvin_helmholtz_history_entry(entry)
  @printf("  %4d %.3f %5d %5d %.6e %.6e %.6e %.6e %.6e\n", entry.step, entry.time,
          entry.active_leaves, entry.dofs, entry.min_density, entry.min_pressure,
          entry.relative_mass_drift, entry.relative_energy_drift, entry.max_wave_speed)
end

function print_kelvin_helmholtz_adaptivity(step, plan)
  summary = adaptivity_summary(plan)
  @printf("  adapt after %4d marked=%d h+=%d h-=%d p+=%d p-=%d\n", step,
          summary.marked_leaf_count, summary.h_refinement_leaf_count,
          summary.h_derefinement_cell_count, summary.p_refinement_leaf_count,
          summary.p_derefinement_leaf_count)
end

# ---------------------------------------------------------------------------
# 5. Human-facing driver
# ---------------------------------------------------------------------------

function run_kelvin_helmholtz_euler_example(; root_counts=ROOT_COUNTS, degree=POLYDEG,
                                            quadrature_extra_points=QUADRATURE_EXTRA_POINTS,
                                            gamma=GAMMA, cfl=CFL, final_time=FINAL_TIME,
                                            save_interval=SAVE_INTERVAL,
                                            adapt_interval=ADAPT_INTERVAL,
                                            solver=nothing,
                                            adaptivity_threshold=ADAPTIVITY_THRESHOLD,
                                            smoothness_threshold=SMOOTHNESS_THRESHOLD,
                                            p_coarsening_threshold=P_COARSENING_THRESHOLD,
                                            h_coarsening_threshold=H_COARSENING_THRESHOLD,
                                            max_h_level=MAX_H_LEVEL,
                                            max_p_degree=MAX_P_DEGREE,
                                            store_segment_solutions=false,
                                            write_vtk=WRITE_VTK,
                                            print_summary=true)
  require_ordinarydiffeq()
  final_time > 0 || throw(ArgumentError("final_time must be positive"))
  max_p_degree >= degree ||
    throw(ArgumentError("max_p_degree must be at least the base degree $degree"))
  context = build_kelvin_helmholtz_euler_context(; root_counts, degree, quadrature_extra_points,
                                                 gamma, cfl)
  initial_diagnostics = context.diagnostics
  save_times = saved_times((0.0, final_time), save_interval)
  adapt_times = saved_times((0.0, final_time), adapt_interval)
  times = merge_time_grids(save_times, adapt_times)
  history = NamedTuple[history_entry(0, 0.0, context, initial_diagnostics)]
  segment_solutions = store_segment_solutions ? Any[] : nothing
  vtk_files = String[]
  pvd_path = nothing
  save_index = 2
  adapt_index = 2

  write_vtk && mkpath(joinpath(@__DIR__, "output"))
  write_vtk && push!(vtk_files, write_kelvin_helmholtz_vtk(context, history[1]))

  if print_summary
    print_kelvin_helmholtz_header(context, final_time; save_interval=save_interval,
                                  adapt_interval=adapt_interval,
                                  adaptivity_threshold=adaptivity_threshold,
                                  smoothness_threshold=smoothness_threshold,
                                  p_coarsening_threshold=p_coarsening_threshold,
                                  h_coarsening_threshold=h_coarsening_threshold,
                                  max_h_level=max_h_level, max_p_degree=max_p_degree)
    print_kelvin_helmholtz_history_entry(history[1])
  end

  for step in 1:(length(times) - 1)
    segment_state, semi, segment_solution = solve_fixed_mesh_segment(context, (times[step], times[step + 1]);
                                                                     solver=solver)
    store_segment_solutions && push!(segment_solutions, segment_solution)
    context = refresh_kelvin_helmholtz_context(context, segment_state)

    while save_index <= length(save_times) && same_time(save_times[save_index], times[step + 1])
      entry = history_entry(length(history), save_times[save_index], context, initial_diagnostics)
      push!(history, entry)
      print_summary && print_kelvin_helmholtz_history_entry(entry)
      write_vtk && push!(vtk_files, write_kelvin_helmholtz_vtk(context, entry))
      save_index += 1
    end

    while adapt_index <= length(adapt_times) && same_time(adapt_times[adapt_index], times[step + 1])
      if !same_time(adapt_times[adapt_index], final_time)
        context, adaptivity_plan = adapt_kelvin_helmholtz_context(context; threshold=adaptivity_threshold,
                                                                  smoothness_threshold=smoothness_threshold,
                                                                  p_coarsening_threshold=p_coarsening_threshold,
                                                                  h_coarsening_threshold=h_coarsening_threshold,
                                                                  max_h_level=max_h_level,
                                                                  max_p_degree=max_p_degree)
        print_summary && !isempty(adaptivity_plan) && print_kelvin_helmholtz_adaptivity(step,
                                                                                         adaptivity_plan)
      end
      adapt_index += 1
    end
  end

  if write_vtk
    pvd_path = write_pvd(joinpath(@__DIR__, "output", "kelvin_helmholtz_euler.pvd"), vtk_files;
                         timesteps=[entry.time for entry in history])
    print_summary && println("  vtk  $(last(vtk_files))")
    print_summary && println("  pvd  $pvd_path")
  end

  return (; context, history, vtk_files, pvd_path, segment_solutions)
end

RUN_KELVIN_HELMHOLTZ_EULER && run_kelvin_helmholtz_euler_example()

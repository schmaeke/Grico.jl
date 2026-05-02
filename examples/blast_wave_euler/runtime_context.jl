# ---------------------------------------------------------------------------
# 4. CFL control and runtime context
# ---------------------------------------------------------------------------

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

struct PositivityLimiter{T}
  gamma::T
  density_floor::T
  pressure_floor::T
  enabled::Bool
end

# The limiter reports how much correction was needed during the most recent
# application. `minimum_theta < 1` means at least one cell polynomial was scaled
# toward its conservative mean; `minimum_theta == 0` means a transferred cell
# mean itself had to be repaired to the admissible set.
mutable struct PositivityLimiterStats{T}
  limited_cell_count::Int
  minimum_theta::T
  minimum_density::T
  minimum_pressure::T
end

function PositivityLimiter(gamma; density_floor=POSITIVITY_DENSITY_FLOOR,
                           pressure_floor=POSITIVITY_PRESSURE_FLOOR,
                           enabled=POSITIVITY_LIMITER_ENABLED)
  T = typeof(float(gamma))
  return PositivityLimiter{T}(T(gamma), T(density_floor), T(pressure_floor), enabled)
end

function _empty_limiter_stats(::Type{T}) where {T<:AbstractFloat}
  return PositivityLimiterStats(0, one(T), T(Inf), T(Inf))
end

function _cell_mean_state(values::CellValues, state::State, field, gamma)
  component_count(field) == 4 ||
    throw(ArgumentError("Euler positivity limiter expects four conserved components"))
  T = eltype(weight(values, 1))
  density_mean = zero(T)
  momentum1_mean = zero(T)
  momentum2_mean = zero(T)
  energy_mean = zero(T)
  volume = zero(T)

  for point_index in 1:point_count(values)
    q = value(values, state, field, point_index)
    weighted = weight(values, point_index)
    volume += weighted
    density_mean += q[1] * weighted
    momentum1_mean += q[2] * weighted
    momentum2_mean += q[3] * weighted
    energy_mean += q[4] * weighted
  end

  mean_state = (density_mean / volume, momentum1_mean / volume, momentum2_mean / volume,
                energy_mean / volume)
  mean_density = mean_state[1]
  mean_pressure = pressure(mean_state, gamma)
  return mean_state, mean_density, mean_pressure
end

function _constant_mode_coefficient(mode, mean_state, component)
  for axis_mode in mode
    axis_mode <= 1 || return zero(eltype(mean_state))
  end

  return mean_state[component]
end

function _density_limiter_theta(theta, mean_density, density_value, density_floor)
  density_value >= density_floor && return theta
  mean_density > density_floor ||
    throw(ArgumentError("cell mean density is below the positivity floor"))
  denominator = mean_density - density_value
  denominator > 0 || return zero(theta)
  return min(theta, (mean_density - density_floor) / denominator)
end

function _pressure_limiter_theta(theta, mean_state, point_state, gamma, pressure_floor)
  pressure(point_state, gamma) >= pressure_floor && return theta
  pressure(mean_state, gamma) > pressure_floor ||
    throw(ArgumentError("cell mean pressure is below the positivity floor"))

  lower = zero(theta)
  upper = theta

  for _ in 1:32
    midpoint = 0.5 * (lower + upper)
    candidate = (mean_state[1] + midpoint * (point_state[1] - mean_state[1]),
                 mean_state[2] + midpoint * (point_state[2] - mean_state[2]),
                 mean_state[3] + midpoint * (point_state[3] - mean_state[3]),
                 mean_state[4] + midpoint * (point_state[4] - mean_state[4]))

    if pressure(candidate, gamma) >= pressure_floor
      lower = midpoint
    else
      upper = midpoint
    end
  end

  return lower
end

function _admissible_mean_state(mean_state, gamma, density_floor, pressure_floor)
  repaired_density = max(mean_state[1], density_floor)
  velocity_data = mean_state[1] > density_floor ?
                  (mean_state[2] / mean_state[1], mean_state[3] / mean_state[1]) :
                  (zero(repaired_density), zero(repaired_density))
  repaired_pressure = max(pressure(mean_state, gamma), pressure_floor)
  return conservative_variables(repaired_density, velocity_data, repaired_pressure, gamma)
end

# The integrated-Legendre basis represents constants by assigning the same
# value to every tensor product of endpoint modes and zero to all interior
# modes, since `(ψ₀ + ψ₁)^D = 1`. This lets the limiter scale a DG polynomial
# around its cell mean directly in coefficient space.
function _limit_cell_coefficients!(coefficients_data, item::CellValues, field, mean_state, theta)
  tensor = tensor_values(item, field)
  tensor !== nothing && is_full_tensor(tensor) ||
    throw(ArgumentError("positivity limiter requires full tensor-product DG cells"))
  modes = tensor_local_modes(tensor)
  mode_count = length(modes)
  component_total = component_count(field)
  local_range = field_dof_range(item, field)

  for component in 1:component_total
    for mode_index in 1:mode_count
      local_row = component_local_index(mode_count, component, mode_index)
      dof = item.single_term_indices[first(local_range)+local_row-1]
      mean_coefficient = _constant_mode_coefficient(modes[mode_index], mean_state, component)
      coefficients_data[dof] = mean_coefficient +
                               theta * (coefficients_data[dof] - mean_coefficient)
    end
  end

  return coefficients_data
end

# This is a Zhang-Shu style cell-average limiter adapted to Grico's modal basis.
# It preserves the conservative cell mean whenever that mean is admissible and
# scales only the high-order part of the polynomial. The mean-repair branch is a
# robustness fallback for projection/transfer artifacts after local h-adaptation;
# it is deliberately visible through `minimum_theta == 0` in the reported stats.
function apply_positivity_limiter!(coefficients_data, plan, field, limiter::PositivityLimiter)
  T = eltype(coefficients_data)
  stats = _empty_limiter_stats(T)
  limiter.enabled || return stats
  state = State(plan, coefficients_data)

  for item in plan.integration.cells
    mean_state, mean_density, mean_pressure = _cell_mean_state(item, state, field, limiter.gamma)
    theta = one(T)
    minimum_density = T(Inf)
    minimum_pressure = T(Inf)

    if mean_density <= limiter.density_floor || mean_pressure <= limiter.pressure_floor
      repaired_mean = _admissible_mean_state(mean_state, limiter.gamma, limiter.density_floor,
                                             limiter.pressure_floor)
      stats.limited_cell_count += 1
      stats.minimum_theta = zero(T)
      stats.minimum_density = min(stats.minimum_density, mean_density)
      stats.minimum_pressure = min(stats.minimum_pressure, mean_pressure)
      _limit_cell_coefficients!(coefficients_data, item, field, repaired_mean, zero(T))
      continue
    end

    for point_index in 1:point_count(item)
      point_state = value(item, state, field, point_index)
      density_value = point_state[1]
      pressure_value = pressure(point_state, limiter.gamma)
      minimum_density = min(minimum_density, density_value)
      minimum_pressure = min(minimum_pressure, pressure_value)
      theta = _density_limiter_theta(theta, mean_density, density_value, limiter.density_floor)
      theta = _pressure_limiter_theta(theta, mean_state, point_state, limiter.gamma,
                                      limiter.pressure_floor)
    end

    stats.minimum_density = min(stats.minimum_density, minimum_density)
    stats.minimum_pressure = min(stats.minimum_pressure, minimum_pressure)

    if theta < one(T)
      stats.limited_cell_count += 1
      stats.minimum_theta = min(stats.minimum_theta, theta)
      _limit_cell_coefficients!(coefficients_data, item, field, mean_state, max(theta, zero(T)))
    end
  end

  return stats
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

  workspace = zeros(T, max_local_dofs)
  return CellwiseMassInverse(blocks, workspace)
end

# Apply the precomputed cellwise inverse to the assembled residual to obtain
# the time derivative `dq_h/dt`. Each DG cell block is independent, and the
# serial traversal reuses one local RHS buffer for all dense solves.
function apply_mass_inverse!(du, mass_inverse::CellwiseMassInverse, rhs)
  fill!(du, zero(eltype(du)))
  blocks = mass_inverse.blocks
  workspace = mass_inverse.workspace

  for block in blocks
    local_dof_count = length(block.global_dofs)
    local_rhs = view(workspace, 1:local_dof_count)

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
# segment. The context carries only data needed to advance the semidiscrete
# system; visual diagnostics are recovered from VTK output rather than from a
# separate runtime quadrature pass.
function blast_wave_context(conserved, state; gamma=GAMMA, cfl=CFL, degree=POLYDEG,
                            limiter=PositivityLimiter(gamma), limiter_stats=nothing)
  spatial_plan = euler_residual_plan(conserved, gamma)
  limiter_stats === nothing &&
    (limiter_stats = apply_positivity_limiter!(coefficients(state), spatial_plan, conserved,
                                               limiter))
  mass_inverse = build_cellwise_mass_inverse(spatial_plan, conserved)
  dt = suggest_timestep(spatial_plan, state, conserved, gamma; cfl)
  return (; domain=field_space(conserved).domain, space=field_space(conserved), conserved, gamma,
          cfl, degree, limiter, limiter_stats, mass_inverse, spatial_plan, state, dt)
end

function refresh_blast_wave_context(context, state=context.state;
                                    limiter_stats=context.limiter_stats)
  dt = suggest_timestep(context.spatial_plan, state, context.conserved, context.gamma;
                        cfl=context.cfl)
  return merge(context, (; state, limiter_stats, dt))
end

function build_blast_wave_euler_context(; root_counts=ROOT_COUNTS, degree=POLYDEG,
                                        quadrature_extra_points=QUADRATURE_EXTRA_POINTS,
                                        gamma=GAMMA, cfl=CFL,
                                        initial_refinement_layers=INITIAL_BLAST_REFINEMENT_LAYERS,
                                        initial_refinement_radius=INITIAL_BLAST_REFINEMENT_RADIUS)
  domain = Domain(DOMAIN_ORIGIN, DOMAIN_EXTENT, root_counts; periodic=PERIODIC_AXES)
  coarse_space = HpSpace(domain,
                         SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(degree),
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
                                  linear_solve=Grico.default_linear_solve)
  plan = blast_wave_adaptivity_plan(context; tolerance=tolerance, max_h_level=max_h_level)

  if isempty(plan)
    return context, plan
  end

  space_transition = transition(plan)
  new_conserved = adapted_field(space_transition, context.conserved)
  new_state = transfer_state(space_transition, context.state, context.conserved, new_conserved;
                             linear_solve=linear_solve)
  spatial_plan = euler_residual_plan(new_conserved, context.gamma)
  limiter_stats = apply_positivity_limiter!(coefficients(new_state), spatial_plan, new_conserved,
                                            context.limiter)
  return blast_wave_context(new_conserved, new_state; gamma=context.gamma, cfl=context.cfl,
                            degree=context.degree, limiter=context.limiter, limiter_stats), plan
end

# `EulerSemidiscretization` is the object handed to `OrdinaryDiffEq.jl`. It owns
# the reusable work arrays needed to evaluate `du = M⁻¹ R(u)`.
struct EulerSemidiscretization{P,F,S,V,W,M,L,R}
  plan::P
  field::F
  state::S
  residual_vector::V
  residual_workspace::W
  mass_inverse::M
  limiter::L
  limiter_stats::R
end

function EulerSemidiscretization(plan, field, initial_state, mass_inverse, limiter)
  runtime_state = State(plan, copy(coefficients(initial_state)))
  residual_vector = similar(coefficients(initial_state))
  residual_workspace = ResidualWorkspace(plan)
  limiter_stats = Ref(_empty_limiter_stats(eltype(residual_vector)))
  return EulerSemidiscretization(plan, field, runtime_state, residual_vector, residual_workspace,
                                 mass_inverse, limiter, limiter_stats)
end

# Copy the ODE state into the reusable `State`, evaluate the DG residual, and
# apply the cellwise inverse mass matrix.
function euler_rhs!(du, u, semi::EulerSemidiscretization, t)
  semi.state.coefficients .= u
  residual!(semi.residual_vector, semi.plan, semi.state, semi.residual_workspace)
  apply_mass_inverse!(du, semi.mass_inverse, semi.residual_vector)
  return nothing
end

function limit_euler_solution!(integrator)
  semi = integrator.p
  semi.limiter_stats[] = apply_positivity_limiter!(integrator.u, semi.plan, semi.field,
                                                   semi.limiter)
  return nothing
end

# Integrate one time segment on a fixed mesh. Mesh changes happen only between
# segments so that the ODE solve itself sees a time-independent semidiscrete
# operator.
function solve_fixed_mesh_segment(context, tspan; solver=nothing, return_solution=false)
  semi = EulerSemidiscretization(context.spatial_plan, context.conserved, context.state,
                                 context.mass_inverse, context.limiter)
  problem = OrdinaryDiffEq.ODEProblem(euler_rhs!, copy(coefficients(context.state)), tspan, semi)
  resolved_solver = solver === nothing ? OrdinaryDiffEq.Tsit5() : solver
  step_size = min(context.dt, tspan[2] - tspan[1])
  limiter_callback = OrdinaryDiffEq.DiscreteCallback((u, t, integrator) -> true,
                                                     limit_euler_solution!;
                                                     save_positions=(false, false))
  solution = OrdinaryDiffEq.solve(problem, resolved_solver; dt=step_size, adaptive=false,
                                  tstops=[tspan[2]], saveat=[tspan[2]], save_start=false,
                                  save_everystep=false, callback=limiter_callback)
  final_state = State(context.spatial_plan, copy(only(solution.u)))
  limiter_stats = apply_positivity_limiter!(coefficients(final_state), context.spatial_plan,
                                            context.conserved, context.limiter)
  return return_solution ? (final_state, solution, limiter_stats) :
         (final_state, nothing, limiter_stats)
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

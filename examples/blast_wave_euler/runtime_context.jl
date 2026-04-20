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
  workspaces::V
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

  workspaces = [zeros(T, max_local_dofs) for _ in 1:Threads.maxthreadid()]
  return CellwiseMassInverse(blocks, workspaces)
end

# Apply the precomputed cellwise inverse to the assembled residual to obtain
# the time derivative `dq_h/dt`. Each DG cell block is independent, so the
# block solves can run in parallel as long as each worker owns its RHS buffer
# and BLAS is kept single-threaded inside the threaded region.
function apply_mass_inverse!(du, mass_inverse::CellwiseMassInverse, rhs)
  fill!(du, zero(eltype(du)))
  blocks = mass_inverse.blocks
  workspaces = mass_inverse.workspaces

  Grico._with_internal_blas_threads() do
    Threads.@threads :static for block_index in eachindex(blocks)
      block = blocks[block_index]
      local_dof_count = length(block.global_dofs)
      local_rhs = view(workspaces[Threads.threadid()], 1:local_dof_count)

      for local_dof in 1:local_dof_count
        local_rhs[local_dof] = rhs[block.global_dofs[local_dof]]
      end

      ldiv!(block.factorization, local_rhs)

      for local_dof in 1:local_dof_count
        du[block.global_dofs[local_dof]] = local_rhs[local_dof]
      end
    end
  end

  return du
end

# Bundle the runtime objects that the driver updates after each solve/adapt
# segment. The context carries only data needed to advance the semidiscrete
# system; visual diagnostics are recovered from VTK output rather than from a
# separate runtime quadrature pass.
function blast_wave_context(conserved, state; gamma=GAMMA, cfl=CFL, degree=POLYDEG)
  spatial_plan = euler_residual_plan(conserved, gamma)
  mass_inverse = build_cellwise_mass_inverse(spatial_plan, conserved)
  dt = suggest_timestep(spatial_plan, state, conserved, gamma; cfl)
  return (; domain=field_space(conserved).domain, space=field_space(conserved), conserved, gamma,
          cfl, degree, mass_inverse, spatial_plan, state, dt)
end

function refresh_blast_wave_context(context, state=context.state)
  dt = suggest_timestep(context.spatial_plan, state, context.conserved, context.gamma;
                        cfl=context.cfl)
  return merge(context, (; state, dt))
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

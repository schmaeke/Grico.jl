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
# cavity solve. Returning an explicit context keeps the driver readable: the
# nonlinear loop can update a compact bundle instead of rebuilding the problem
# description at every line.
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

# Finish one Picard linear solve by applying the example's under-relaxation and
# by evaluating the DG diagnostics reported in the iteration table.
function finalize_picard_step(context, candidate_state)
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
  return next_context, relative_update, mass_l2, energy
end

# Advance one under-relaxed Picard step and return both the updated context and
# the assembled Oseen system. Returning the system makes the algebraic solve
# explicit without obscuring the nonlinear fixed-point update.
function advance_picard_step(context; linear_solve=direct_sparse_solve)
  # 1. update the lagged advecting state in the reusable operator,
  # 2. assemble and solve the linear Oseen system,
  # 3. under-relax the new iterate for robustness,
  # 4. report a relative update and DG diagnostics.
  context.operator.advecting_state = context.flow_state
  system = assemble(context.plan)
  candidate_state = State(context.plan, solve(system; linear_solve=linear_solve))
  next_context, relative_update, mass_l2, energy = finalize_picard_step(context, candidate_state)
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
function adapt_lid_driven_cavity_context(context; tolerance=ADAPTIVITY_TOLERANCE,
                                         max_h_level=MAX_H_LEVEL)
  limits = AdaptivityLimits(context.velocity_space; min_p=VELOCITY_DEGREE, max_p=VELOCITY_DEGREE,
                            max_h_level=max_h_level)
  velocity_plan = adaptivity_plan(context.flow_state, context.velocity; tolerance=tolerance,
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

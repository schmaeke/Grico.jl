using Test
using Grico
import Grico: adaptivity_plan, coefficient_coarsening_indicators, h_adaptation_axes,
              h_coarsening_candidates, interface_jump_indicators, is_domain_boundary, level,
              local_modes, map_to_biunit_cube, multiresolution_indicators, p_degree_change,
              periodic_axes, projection_coarsening_indicators, source_leaves, snapshot,
              target_space, derived_adaptivity_plan

const ADAPTIVITY_TOL = 1.0e-8

function _field_value_at_point(field, state, x, component::Int=1)
  space = field_space(field)
  domain = Grico.domain(space)
  leaves = active_leaves(space)
  tolerance = 1.0e-12
  leaf_index = findfirst(leaves) do current_leaf
    lower = cell_lower(domain, current_leaf)
    upper = cell_upper(domain, current_leaf)
    all(lower[axis] - tolerance <= x[axis] <= upper[axis] + tolerance for axis in eachindex(x))
  end
  leaf_index === nothing && error("point $x does not lie on an active leaf")
  leaf = leaves[leaf_index]
  ξ = map_to_biunit_cube(domain, leaf, x)
  one_dimensional = ntuple(axis -> Grico._fe_basis_values(ξ[axis], cell_degrees(space, leaf)[axis]),
                           length(x))
  values = field_component_values(state, field, component)
  result = 0.0

  for mode in local_modes(space, leaf)
    local_value = 1.0

    for axis in 1:length(x)
      local_value *= one_dimensional[axis][mode[axis]+1]
    end

    for term in mode_terms(space, leaf, mode)
      result += local_value * term.second * values[term.first]
    end
  end

  return result
end

@testset "Adaptivity Plan And Transition" begin
  domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  plan = AdaptivityPlan(space)

  @test isempty(plan)
  request_h_refinement!(plan, 1, 1)
  request_h_refinement!(plan, 2, 2)
  @test !isempty(plan)
  @test h_adaptation_axes(plan, 1) == (true, true)
  @test p_degree_change(plan, 1) == (0, 0)

  space_transition = transition(plan)
  refined = target_space(space_transition)

  @test active_leaf_count(refined) == 3
  @test all(source_leaves(space_transition, leaf) == [1] for leaf in active_leaves(refined))
  @test all(cell_degrees(refined, leaf) == (2, 2) for leaf in active_leaves(refined))
end

@testset "Physical Domain Adaptivity" begin
  background = Domain((0.0,), (3.0,), (3,))
  region = ImplicitRegion(x -> x[1] - 1.5; subdivision_depth=1)
  domain = PhysicalDomain(background, region)
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(1)))
  plan = AdaptivityPlan(space)

  @test active_leaves(space) == [1, 2]
  @test active_leaves(plan) == [1, 2]
  @test active_leaves(target_space(transition(plan))) == [1, 2]

  request_h_refinement!(plan, 2, 1)
  summary = adaptivity_summary(plan)
  refined = target_space(transition(plan))

  @test summary.marked_leaf_count == 1
  @test summary.h_refinement_leaf_count == 1
  @test active_leaves(refined) == [1, 4]

  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), collect(1.0:scalar_dof_count(space)))
  automatic_plan = adaptivity_plan(state, u; tolerance=0.0,
                                   limits=AdaptivityLimits(space; min_p=1, max_p=1, max_h_level=1))
  automatic_active = active_leaves(target_space(transition(automatic_plan)))
  @test length(automatic_active) == 3
  @test all(leaf -> level(grid(space), leaf) == (1,), automatic_active)
end

@testset "Periodic Transition" begin
  domain = Domain((0.0,), (1.0,), (2,); periodic=true)
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), collect(1.0:scalar_dof_count(space)))

  plan = AdaptivityPlan(space)
  request_h_refinement!(plan, 2, 1)
  space_transition = transition(plan)
  refined = target_space(space_transition)
  new_u = adapted_field(space_transition, u)
  transferred = transfer_state(space_transition, state, u, new_u)

  @test periodic_axes(refined) == (true,)
  @test !is_domain_boundary(grid(refined), 1, 1, LOWER)
  @test !is_domain_boundary(grid(refined), 4, 1, UPPER)

  for x in ((0.0,), (0.125,), (0.5,), (0.875,), (1.0,))
    @test _field_value_at_point(u, state, x) ≈ _field_value_at_point(new_u, transferred, x) atol = ADAPTIVITY_TOL
  end
end

@testset "Scalar State Transfer" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(3)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [0.3, -0.8, 0.45, 1.1])

  plan = AdaptivityPlan(space)
  request_h_refinement!(plan, 1, 1)
  space_transition = transition(plan)
  new_u = adapted_field(space_transition, u)
  transferred = transfer_state(space_transition, state, u, new_u)

  @test field_space(new_u) === target_space(space_transition)
  @test field_dof_range(field_layout(transferred), u) ==
        field_dof_range(field_layout(transferred), new_u)
  @test field_values(transferred, u) == field_values(transferred, new_u)

  for x in ((0.0,), (0.125,), (0.25,), (0.5,), (0.75,), (1.0,))
    @test _field_value_at_point(u, state, x) ≈ _field_value_at_point(new_u, transferred, x) atol = ADAPTIVITY_TOL
  end
end

@testset "Transfer Strategy Boundary" begin
  cg_domain = Domain((0.0,), (1.0,), (1,))
  cg_space = HpSpace(cg_domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  cg_plan = AdaptivityPlan(cg_space)
  request_h_refinement!(cg_plan, 1, 1)
  cg_transition = transition(cg_plan)
  custom_solve = (A, b) -> A \ b

  @test Grico._transfer_strategy(cg_transition, Grico.default_linear_solve) ===
        Grico._LOCAL_PROJECTION_TRANSFER
  @test Grico._transfer_strategy(cg_transition, custom_solve) === Grico._VARIATIONAL_TRANSFER

  dg_domain = Domain((0.0,), (1.0,), (1,))
  dg_space = HpSpace(dg_domain,
                     SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0),
                                  continuity=:dg))
  dg_plan = AdaptivityPlan(dg_space)
  request_h_refinement!(dg_plan, 1, 1)
  dg_transition = transition(dg_plan)

  @test Grico._transfer_strategy(dg_transition, Grico.default_linear_solve) ===
        Grico._CELLWISE_DG_TRANSFER
  @test Grico._transfer_strategy(dg_transition, custom_solve) === Grico._CELLWISE_DG_TRANSFER
end

@testset "Manual Transfer And Indicator Policy Separation" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [0.3, -0.8, 0.45])

  manual_plan = AdaptivityPlan(space)
  request_h_refinement!(manual_plan, 1, 1)
  heuristic_plan = adaptivity_plan(state, u; tolerance=0.0,
                                   limits=AdaptivityLimits(space; min_h_level=0,
                                                           max_h_level=0, min_p=2,
                                                           max_p=3))

  @test h_adaptation_axes(manual_plan, 1) == (true,)
  @test p_degree_change(manual_plan, 1) == (0,)
  @test h_adaptation_axes(heuristic_plan, 1) == (false,)
  @test p_degree_change(heuristic_plan, 1) == (1,)

  manual_transition = transition(manual_plan)
  new_u = adapted_field(manual_transition, u)
  transferred = transfer_state(manual_transition, state, u, new_u)

  for x in ((0.0,), (0.125,), (0.25,), (0.5,), (0.75,), (1.0,))
    @test _field_value_at_point(u, state, x) ≈ _field_value_at_point(new_u, transferred, x) atol = ADAPTIVITY_TOL
  end
end

@testset "Coarsening State Transfer" begin
  coarse_domain = Domain((0.0,), (1.0,), (1,))
  coarse_space = HpSpace(coarse_domain,
                         SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(3)))
  coarse_u = ScalarField(coarse_space; name=:u)
  coarse_state = State(FieldLayout((coarse_u,)), [0.3, -0.8, 0.45, 1.1])

  refine_plan = AdaptivityPlan(coarse_space)
  request_h_refinement!(refine_plan, 1, 1)
  refine_transition = transition(refine_plan)
  fine_u = adapted_field(refine_transition, coarse_u)
  fine_state = transfer_state(refine_transition, coarse_state, coarse_u, fine_u)

  coarsen_plan = AdaptivityPlan(target_space(refine_transition))
  request_h_derefinement!(coarsen_plan, 1, 1)
  coarsen_transition = transition(coarsen_plan)
  recovered_u = adapted_field(coarsen_transition, fine_u; name=:u)
  recovered_state = transfer_state(coarsen_transition, fine_state, fine_u, recovered_u)
  compact_coarsen_transition = transition(coarsen_plan; compact=true)
  compact_recovered_u = adapted_field(compact_coarsen_transition, fine_u; name=:u)
  compact_recovered_state = transfer_state(compact_coarsen_transition, fine_state, fine_u,
                                           compact_recovered_u)

  @test active_leaf_count(target_space(coarsen_transition)) == 1
  @test source_leaves(coarsen_transition, 1) == [2, 3]
  @test Grico.stored_cell_count(grid(target_space(coarsen_transition))) == 3
  @test active_leaf_count(target_space(compact_coarsen_transition)) == 1
  @test Grico.stored_cell_count(grid(target_space(compact_coarsen_transition))) == 1
  @test active_leaves(target_space(compact_coarsen_transition)) == [1]
  @test source_leaves(compact_coarsen_transition, 1) == [2, 3]
  @test Grico.check_space(target_space(compact_coarsen_transition)) === nothing

  for x in ((0.0,), (0.125,), (0.25,), (0.5,), (0.75,), (1.0,))
    @test _field_value_at_point(coarse_u, coarse_state, x) ≈
          _field_value_at_point(recovered_u, recovered_state, x) atol = ADAPTIVITY_TOL
    @test _field_value_at_point(coarse_u, coarse_state, x) ≈
          _field_value_at_point(compact_recovered_u, compact_recovered_state, x) atol = ADAPTIVITY_TOL
  end
end

@testset "Alternating Snapshot H Refinement Axes" begin
  domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1), continuity=:dg))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), collect(1.0:scalar_dof_count(space)))

  x_plan = AdaptivityPlan(space)
  request_h_refinement!(x_plan, 1, 1)
  x_transition = transition(x_plan)
  x_space = target_space(x_transition)
  x_u = adapted_field(x_transition, u)
  x_state = transfer_state(x_transition, state, u, x_u)

  coarsen_plan = AdaptivityPlan(x_space)
  request_h_derefinement!(coarsen_plan, 1, 1)
  coarsen_transition = transition(coarsen_plan)
  coarse_again = target_space(coarsen_transition)
  coarse_u = adapted_field(coarsen_transition, x_u)
  coarse_state = transfer_state(coarsen_transition, x_state, x_u, coarse_u)

  y_plan = AdaptivityPlan(coarse_again)
  request_h_refinement!(y_plan, 1, 2)
  y_transition = transition(y_plan)
  y_space = target_space(y_transition)
  y_u = adapted_field(y_transition, coarse_u)
  y_state = transfer_state(y_transition, coarse_state, coarse_u, y_u)

  @test active_leaf_count(y_space) == 2
  @test all(leaf -> level(grid(y_space), leaf) == (0, 1), active_leaves(y_space))

  for x in ((0.2, 0.2), (0.7, 0.3), (0.4, 0.8))
    @test _field_value_at_point(u, state, x) ≈ _field_value_at_point(y_u, y_state, x) atol = ADAPTIVITY_TOL
  end
end

@testset "Projection H Coarsening Indicators" begin
  coarse_domain = Domain((0.0,), (1.0,), (1,))
  coarse_space = HpSpace(coarse_domain,
                         SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(3)))
  coarse_u = ScalarField(coarse_space; name=:u)
  coarse_state = State(FieldLayout((coarse_u,)), [0.3, -0.8, 0.45, 1.1])

  refine_plan = AdaptivityPlan(coarse_space)
  request_h_refinement!(refine_plan, 1, 1)
  refine_transition = transition(refine_plan)
  fine_u = adapted_field(refine_transition, coarse_u)
  fine_state = transfer_state(refine_transition, coarse_state, coarse_u, fine_u)
  candidates = h_coarsening_candidates(field_space(fine_u))
  indicators = projection_coarsening_indicators(fine_state, fine_u, candidates)
  coarsening_tol = 10 * sqrt(eps(Float64))

  @test length(candidates) == 1
  @test indicators[1] <= coarsening_tol
end

@testset "Transfer State Linear Solve Hook" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(3)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [0.3, -0.8, 0.45, 1.1])

  plan = AdaptivityPlan(space)
  request_h_refinement!(plan, 1, 1)
  space_transition = transition(plan)
  new_u = adapted_field(space_transition, u)
  calls = Ref(0)
  transferred = transfer_state(space_transition, state, u, new_u;
                               linear_solve=(A, b; kwargs...) -> begin
                                 calls[] += 1
                                 if A isa Grico.AssemblyPlan
                                   return Grico.default_linear_solve(A, b; kwargs...)
                                 end
                                 return A \ b
                               end)

  @test calls[] == 1

  for x in ((0.0,), (0.125,), (0.25,), (0.5,), (0.75,), (1.0,))
    @test _field_value_at_point(u, state, x) ≈ _field_value_at_point(new_u, transferred, x) atol = ADAPTIVITY_TOL
  end
end

@testset "P Adaptation Transfer" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [0.5, -0.25, 1.0])

  refine_plan = AdaptivityPlan(space)
  request_p_refinement!(refine_plan, 1, 1)
  @test p_degree_change(refine_plan, 1) == (1,)

  refine_transition = transition(refine_plan)
  enriched_u = adapted_field(refine_transition, u)
  enriched_state = transfer_state(refine_transition, state, u, enriched_u)

  coarsen_plan = AdaptivityPlan(target_space(refine_transition))
  request_p_derefinement!(coarsen_plan, 1, 1)
  @test p_degree_change(coarsen_plan, 1) == (-1,)

  coarsen_transition = transition(coarsen_plan)
  recovered_u = adapted_field(coarsen_transition, enriched_u; name=:u)
  recovered_state = transfer_state(coarsen_transition, enriched_state, enriched_u, recovered_u)

  @test active_leaf_count(target_space(refine_transition)) == 1
  @test cell_degrees(target_space(refine_transition), 1) == (3,)
  @test cell_degrees(target_space(coarsen_transition), 1) == (2,)

  for x in ((0.0,), (0.125,), (0.5,), (0.875,), (1.0,))
    @test _field_value_at_point(u, state, x) ≈
          _field_value_at_point(recovered_u, recovered_state, x) atol = ADAPTIVITY_TOL
  end
end

@testset "Multi-Field State Transfer" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  s = ScalarField(space; name=:s)
  v = VectorField(space, 2; name=:v)
  coefficients = [0.25, -0.5, 1.0, -1.0, 0.75, -0.25, 0.5, 0.125, -0.375]
  state = State(FieldLayout((s, v)), coefficients)

  plan = AdaptivityPlan(space)
  request_h_refinement!(plan, 1, 1)
  space_transition = transition(plan)
  new_fields, transferred = transfer_state(space_transition, state)
  new_s, new_v = new_fields

  @test field_space(new_s) === target_space(space_transition)
  @test field_space(new_v) === target_space(space_transition)
  @test component_count(new_v) == 2

  for x in ((0.0,), (0.2,), (0.5,), (0.8,), (1.0,))
    @test _field_value_at_point(s, state, x) ≈ _field_value_at_point(new_s, transferred, x) atol = ADAPTIVITY_TOL
    @test _field_value_at_point(v, state, x, 1) ≈ _field_value_at_point(new_v, transferred, x, 1) atol = ADAPTIVITY_TOL
    @test _field_value_at_point(v, state, x, 2) ≈ _field_value_at_point(new_v, transferred, x, 2) atol = ADAPTIVITY_TOL
  end
end

@testset "Derived Adaptivity Plans" begin
  domain = Domain((0.0,), (1.0,), (1,))
  velocity_space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  pressure_space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  pressure = ScalarField(pressure_space; name=:p)
  driver_plan = AdaptivityPlan(velocity_space)

  request_p_refinement!(driver_plan, 1, 1)
  request_h_refinement!(driver_plan, 1, 1)

  inherited_plan = derived_adaptivity_plan(driver_plan, pressure)
  offset_plan = derived_adaptivity_plan(driver_plan, pressure_space; degree_offset=-1)

  @test active_leaves(inherited_plan) == active_leaves(driver_plan)
  @test all(cell_degrees(inherited_plan, leaf) == (1,) for leaf in active_leaves(inherited_plan))
  @test all(cell_degrees(offset_plan, leaf) == (2,) for leaf in active_leaves(offset_plan))
  @test_throws ArgumentError derived_adaptivity_plan(driver_plan, pressure_space;
                                                     degree_offset=(0, 0))
  @test_throws ArgumentError derived_adaptivity_plan(driver_plan, pressure_space;
                                                     degree_offset=big(typemax(Int)) + 1)
end

@testset "Mixed-Space State Transfer" begin
  domain = Domain((0.0,), (1.0,), (1,))
  velocity_space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  pressure_space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  velocity = ScalarField(velocity_space; name=:u)
  pressure = ScalarField(pressure_space; name=:p)
  state = State(FieldLayout((velocity, pressure)), [0.3, -0.8, 0.45, 1.1, -0.35])

  velocity_plan = AdaptivityPlan(velocity_space)
  request_p_refinement!(velocity_plan, 1, 1)
  request_h_refinement!(velocity_plan, 1, 1)
  pressure_plan = derived_adaptivity_plan(velocity_plan, pressure; degree_offset=-1)
  velocity_transition = transition(velocity_plan)
  pressure_transition = transition(pressure_plan)
  new_fields, transferred = transfer_state((pressure_plan, velocity_plan), state)
  new_velocity, new_pressure = new_fields

  @test active_leaves(field_space(new_velocity)) == active_leaves(target_space(velocity_transition))
  @test active_leaves(field_space(new_pressure)) == active_leaves(target_space(pressure_transition))
  @test all(cell_degrees(field_space(new_velocity), leaf) ==
            cell_degrees(target_space(velocity_transition), leaf)
            for leaf in active_leaves(field_space(new_velocity)))
  @test all(cell_degrees(field_space(new_pressure), leaf) ==
            cell_degrees(target_space(pressure_transition), leaf)
            for leaf in active_leaves(field_space(new_pressure)))

  for x in ((0.0,), (0.125,), (0.25,), (0.5,), (0.75,), (1.0,))
    @test _field_value_at_point(velocity, state, x) ≈
          _field_value_at_point(new_velocity, transferred, x) atol = ADAPTIVITY_TOL
    @test _field_value_at_point(pressure, state, x) ≈
          _field_value_at_point(new_pressure, transferred, x) atol = ADAPTIVITY_TOL
  end

  @test_throws ArgumentError transfer_state((velocity_plan,), state)
  @test_throws ArgumentError transfer_state((velocity_plan, pressure_plan, pressure_plan), state)
  mismatched_pressure_plan = AdaptivityPlan(pressure_space)
  @test_throws ArgumentError transfer_state((velocity_plan, mismatched_pressure_plan), state)
  @test_throws ArgumentError adapted_field(velocity_transition, pressure)
  @test_throws ArgumentError adapted_fields(velocity_transition, (velocity, "bad"))
  @test_throws ArgumentError transfer_state(velocity_transition, state, ("bad",), (new_velocity,))
end

@testset "Mixed Hp Transfer" begin
  domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [0.2, -0.1, 0.5, 0.3, -0.25, 0.15, 0.4, -0.2, 0.1])

  plan = AdaptivityPlan(space)
  request_p_refinement!(plan, 1, 2)
  request_h_refinement!(plan, 1, 1)
  space_transition = transition(plan)
  new_u = adapted_field(space_transition, u)
  transferred = transfer_state(space_transition, state, u, new_u)

  @test active_leaf_count(target_space(space_transition)) == 2
  @test all(cell_degrees(target_space(space_transition), leaf) == (2, 3)
            for leaf in active_leaves(target_space(space_transition)))

  for x in ((0.0, 0.0), (0.2, 0.3), (0.5, 0.5), (0.8, 0.1), (1.0, 1.0))
    @test _field_value_at_point(u, state, x) ≈ _field_value_at_point(new_u, transferred, x) atol = 1.0e-7
  end
end

@testset "Tuple Plan Edits" begin
  domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  plan = AdaptivityPlan(space)

  leaves = request_h_refinement!(plan, 1, (true, true))
  @test length(leaves) == 4
  @test active_leaf_count(plan) == 4
  @test h_adaptation_axes(plan, 1) == (true, true)
  @test all(cell_degrees(plan, leaf) == (2, 2) for leaf in active_leaves(plan))

  target_leaf = active_leaves(plan)[1]
  request_p_refinement!(plan, target_leaf, (1, 2))
  @test cell_degrees(plan, target_leaf) == (3, 4)
  request_p_derefinement!(plan, target_leaf, (1, 2))
  @test cell_degrees(plan, target_leaf) == (2, 2)
  @test p_degree_change(plan, 1) == (0, 0)
end

@testset "Adaptivity Limits" begin
  domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  limits = AdaptivityLimits(2; min_h_level=(0, 0), max_h_level=(1, 0), min_p=(2, 1), max_p=(2, 3))
  plan = AdaptivityPlan(space; limits=limits)

  request_h_refinement!(plan, 1, 1)
  @test_throws ArgumentError request_h_refinement!(plan, 2, 1)
  @test_throws ArgumentError request_h_refinement!(plan, 2, 2)
  @test_throws ArgumentError request_p_refinement!(plan, 2, 1)
  request_p_refinement!(plan, 2, 2)
  @test_throws ArgumentError request_p_refinement!(plan, 2, 2; increment=2)
  request_p_derefinement!(plan, 2, 2)
  @test_throws ArgumentError request_p_derefinement!(plan, 2, 1)

  derefine_limits = AdaptivityLimits(2; min_h_level=(1, 0), max_h_level=(1, 0), min_p=(2, 1),
                                     max_p=(2, 3))
  derefine_plan = AdaptivityPlan(target_space(transition(plan)); limits=derefine_limits)
  @test_throws ArgumentError request_h_derefinement!(derefine_plan, 1, 1)
end

@testset "Modal Detail At P1" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  u = ScalarField(space; name=:u)

  constant_values = zeros(scalar_dof_count(space))

  for term in mode_terms(space, 1, (0,))
    constant_values[term.first] += term.second
  end

  for term in mode_terms(space, 1, (1,))
    constant_values[term.first] += term.second
  end

  constant_state = State(FieldLayout((u,)), constant_values)
  @test coefficient_coarsening_indicators(constant_state, u)[1][1] ≈ 0.0 atol = ADAPTIVITY_TOL
  constant_plan = adaptivity_plan(constant_state, u; tolerance=0.0,
                                  limits=AdaptivityLimits(space; max_h_level=0, max_p=2))
  @test isempty(constant_plan)

  linear_values = zeros(scalar_dof_count(space))

  for term in mode_terms(space, 1, (0,))
    linear_values[term.first] -= term.second
  end

  for term in mode_terms(space, 1, (1,))
    linear_values[term.first] += term.second
  end

  linear_state = State(FieldLayout((u,)), linear_values)
  @test coefficient_coarsening_indicators(linear_state, u)[1][1] ≈ inv(sqrt(2)) atol = ADAPTIVITY_TOL
  linear_plan = adaptivity_plan(linear_state, u; tolerance=0.0,
                                limits=AdaptivityLimits(space; max_h_level=0, max_p=2))
  @test h_adaptation_axes(linear_plan, 1) == (false,)
  @test p_degree_change(linear_plan, 1) == (1,)
end

@testset "DG P0 Adaptivity Support" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0), continuity=:dg))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [2.5])

  @test AdaptivityLimits(space).min_p == (0,)
  @test coefficient_coarsening_indicators(state, u)[1][1] ≈ 0.0 atol = ADAPTIVITY_TOL

  refine_plan = AdaptivityPlan(space)
  request_p_refinement!(refine_plan, 1, 1)
  refine_transition = transition(refine_plan)
  refined = target_space(refine_transition)
  refined_u = adapted_field(refine_transition, u)
  refined_state = transfer_state(refine_transition, state, u, refined_u)

  @test cell_degrees(refined, 1) == (1,)
  for x in ((0.0,), (0.25,), (0.5,), (1.0,))
    @test _field_value_at_point(refined_u, refined_state, x) ≈ 2.5 atol = ADAPTIVITY_TOL
  end

  coarsen_plan = AdaptivityPlan(refined)
  request_p_derefinement!(coarsen_plan, 1, 1)
  coarsen_transition = transition(coarsen_plan)
  coarsened = target_space(coarsen_transition)
  coarsened_u = adapted_field(coarsen_transition, refined_u)
  coarsened_state = transfer_state(coarsen_transition, refined_state, refined_u, coarsened_u)

  @test cell_degrees(coarsened, 1) == (0,)
  for x in ((0.0,), (0.3,), (0.8,), (1.0,))
    @test _field_value_at_point(coarsened_u, coarsened_state, x) ≈ 2.5 atol = ADAPTIVITY_TOL
  end
end

@testset "DG Transfer Uses Cellwise Solve Path" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0), continuity=:dg))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [2.5])

  refine_plan = AdaptivityPlan(space)
  request_h_refinement!(refine_plan, 1, 1)
  refine_transition = transition(refine_plan)
  refined = target_space(refine_transition)
  refined_u = adapted_field(refine_transition, u)
  calls = Ref(0)
  refined_state = transfer_state(refine_transition, state, u, refined_u;
                                 linear_solve=(A, b) -> begin
                                   calls[] += 1
                                   return A \ b
                                 end)

  @test calls[] == active_leaf_count(refined)
  @test_throws ArgumentError transfer_state(refine_transition, state, u, refined_u;
                                            linear_solve=(A, b) -> 1.0)
  @test_throws ArgumentError transfer_state(refine_transition, state, u, refined_u;
                                            linear_solve=(A, b) -> zeros(size(A, 1) + 1))
  @test_throws ArgumentError transfer_state(refine_transition, state, u, refined_u;
                                            linear_solve=(A, b) -> fill("bad", size(A, 1)))
  for x in ((0.0,), (0.2,), (0.5,), (0.8,), (1.0,))
    @test _field_value_at_point(refined_u, refined_state, x) ≈ 2.5 atol = ADAPTIVITY_TOL
  end
end

@testset "DG Jump Refinement Defaults" begin
  domain = Domain((0.0,), (1.0,), (2,))
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0), continuity=:dg))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [1.0, 3.0])
  jumps = interface_jump_indicators(state, u)

  @test jumps[1][1] > 0.0
  @test jumps[1][1] ≈ jumps[2][1] atol = ADAPTIVITY_TOL
  v = ScalarField(space; name=:v)
  mixed_state = State(FieldLayout((v, u)), [10.0, 20.0, 1.0, 3.0])
  mixed_jumps = interface_jump_indicators(mixed_state, u)
  @test mixed_jumps[1][1] ≈ jumps[1][1] atol = ADAPTIVITY_TOL
  @test mixed_jumps[2][1] ≈ jumps[2][1] atol = ADAPTIVITY_TOL

  jump_plan = adaptivity_plan(state, u; tolerance=0.0,
                              limits=AdaptivityLimits(space; min_p=0, max_p=0, max_h_level=1))
  @test h_adaptation_axes(jump_plan, 1) == (true,)
  @test h_adaptation_axes(jump_plan, 2) == (true,)
  @test p_degree_change(jump_plan, 1) == (0,)
  @test p_degree_change(jump_plan, 2) == (0,)
end

@testset "Multiresolution Adaptivity Planning" begin
  jump_domain = Domain((0.0,), (1.0,), (2,))
  jump_space = HpSpace(jump_domain,
                       SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0),
                                    continuity=:dg))
  jump_u = ScalarField(jump_space; name=:u)
  jump_state = State(FieldLayout((jump_u,)), [1.0, 3.0])
  jump_limits = AdaptivityLimits(jump_space; min_p=0, max_p=0, max_h_level=1)
  jump_details = multiresolution_indicators(jump_state, jump_u; limits=jump_limits)

  @test jump_details[1][1] > 0.0
  @test jump_details[2][1] > 0.0

  jump_plan = adaptivity_plan(jump_state, jump_u; tolerance=0.0, limits=jump_limits)
  @test h_adaptation_axes(jump_plan, 1) == (true,)
  @test h_adaptation_axes(jump_plan, 2) == (true,)

  coarse_domain = Domain((0.0,), (1.0,), (1,))
  coarse_space = HpSpace(coarse_domain,
                         SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0),
                                      continuity=:dg))
  coarse_u = ScalarField(coarse_space; name=:u)
  coarse_state = State(FieldLayout((coarse_u,)), [2.0])
  refine_plan = AdaptivityPlan(coarse_space)
  request_h_refinement!(refine_plan, 1, 1)
  refine_transition = transition(refine_plan)
  refined_u = adapted_field(refine_transition, coarse_u)
  refined_state = transfer_state(refine_transition, coarse_state, coarse_u, refined_u)
  refined_space = field_space(refined_u)
  coarsen_limits = AdaptivityLimits(refined_space; min_p=0, max_p=0, max_h_level=1)
  coarsen_plan = adaptivity_plan(refined_state, refined_u; tolerance=10 * sqrt(eps(Float64)),
                                 limits=coarsen_limits)

  @test adaptivity_summary(coarsen_plan).h_derefinement_cell_count == 1

  saturated_domain = Domain((0.0,), (1.0,), (1,))
  saturated_space = HpSpace(saturated_domain,
                            SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0),
                                         continuity=:dg))
  saturated_u = ScalarField(saturated_space; name=:u)
  split_plan = AdaptivityPlan(saturated_space)
  request_h_refinement!(split_plan, 1, 1)
  split_transition = transition(split_plan)
  split_space = target_space(split_transition)
  split_u = adapted_field(split_transition, saturated_u)
  saturated_plan = AdaptivityPlan(split_space)

  for leaf in active_leaves(split_space)
    request_h_refinement!(saturated_plan, leaf, 1)
  end

  saturated_transition = transition(saturated_plan)
  saturated_u = adapted_field(saturated_transition, split_u)
  saturated_space = field_space(saturated_u)
  saturated_state = State(FieldLayout((saturated_u,)), [2.0, 2.0, 3.0, 3.0])
  saturated_limits = AdaptivityLimits(saturated_space; min_p=0, max_p=0, max_h_level=2)
  saturated_candidates = h_coarsening_candidates(saturated_space; limits=saturated_limits)

  @test all(<(1.0e-2),
            projection_coarsening_indicators(saturated_state, saturated_u, saturated_candidates))

  retained_plan = adaptivity_plan(saturated_state, saturated_u; tolerance=1.0e-2,
                                  limits=saturated_limits)
  @test adaptivity_summary(retained_plan).h_derefinement_cell_count == 0

  p_domain = Domain((0.0,), (1.0,), (1,))
  p_space = HpSpace(p_domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  p_u = ScalarField(p_space; name=:u)
  p_values = zeros(scalar_dof_count(p_space))

  for term in mode_terms(p_space, 1, (0,))
    p_values[term.first] -= term.second
  end

  for term in mode_terms(p_space, 1, (1,))
    p_values[term.first] += term.second
  end

  p_state = State(FieldLayout((p_u,)), p_values)
  p_limits = AdaptivityLimits(p_space; min_h_level=0, max_h_level=0, min_p=1, max_p=2)
  p_plan = adaptivity_plan(p_state, p_u; tolerance=0.0, limits=p_limits)

  @test h_adaptation_axes(p_plan, 1) == (false,)
  @test p_degree_change(p_plan, 1) == (1,)

  hp_limits = AdaptivityLimits(p_space; min_h_level=0, max_h_level=1, min_p=1, max_p=2)
  hp_plan = adaptivity_plan(p_state, p_u; tolerance=0.0, limits=hp_limits)

  @test h_adaptation_axes(hp_plan, 1) == (false,)
  @test p_degree_change(hp_plan, 1) == (1,)

  h_only_limits = AdaptivityLimits(p_space; min_p=1, max_p=1, max_h_level=1)
  h_only_plan = adaptivity_plan(p_state, p_u; tolerance=0.0, limits=h_only_limits)

  @test h_adaptation_axes(h_only_plan, 1) == (true,)
  @test p_degree_change(h_only_plan, 1) == (0,)

  dg_p_space = HpSpace(p_domain,
                       SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1),
                                    continuity=:dg))
  dg_p_u = ScalarField(dg_p_space; name=:u)
  dg_p_values = zeros(scalar_dof_count(dg_p_space))

  for term in mode_terms(dg_p_space, 1, (0,))
    dg_p_values[term.first] -= term.second
  end

  for term in mode_terms(dg_p_space, 1, (1,))
    dg_p_values[term.first] += term.second
  end

  dg_p_state = State(FieldLayout((dg_p_u,)), dg_p_values)
  dg_p_limits = AdaptivityLimits(dg_p_space; min_h_level=0, max_h_level=1, min_p=1, max_p=2)
  dg_p_plan = adaptivity_plan(dg_p_state, dg_p_u; tolerance=0.0, limits=dg_p_limits)

  @test h_adaptation_axes(dg_p_plan, 1) == (false,)
  @test p_degree_change(dg_p_plan, 1) == (1,)
end

@testset "DG Jump Indicators On Hanging Interface" begin
  mesh_domain = Domain((0.0,), (1.0,), (2,))
  coarse = HpSpace(mesh_domain,
                   SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0), continuity=:dg))
  base_u = ScalarField(coarse; name=:u)
  refine_plan = AdaptivityPlan(coarse)
  request_h_refinement!(refine_plan, 1, 1)
  refine_transition = transition(refine_plan)
  refined = target_space(refine_transition)
  u = adapted_field(refine_transition, base_u)
  refined_domain = field_space(u).domain
  values = Vector{Float64}(undef, active_leaf_count(refined))

  for (leaf_index, leaf) in enumerate(active_leaves(refined))
    values[leaf_index] = cell_upper(refined_domain, leaf, 1) <= 0.5 ? 1.0 : 3.0
  end

  state = State(FieldLayout((u,)), values)
  jumps = interface_jump_indicators(state, u)

  for (leaf_index, leaf) in enumerate(active_leaves(refined))
    cell_upper(refined_domain, leaf, 1) <= 0.25 &&
      (@test jumps[leaf_index][1] ≈ 0.0 atol = ADAPTIVITY_TOL)
    cell_upper(refined_domain, leaf, 1) > 0.25 && (@test jumps[leaf_index][1] > 0.0)
  end
end

@testset "Mixed Continuity Adaptivity Defaults" begin
  domain = Domain((0.0, 0.0), (2.0, 2.0), (2, 2))
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=AxisDegrees((1, 0)),
                               continuity=(:cg, :dg)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [1.0, 1.0, 1.0, 3.0, 3.0, 3.0])

  @test AdaptivityLimits(space).min_p == (1, 0)
  @test interface_jump_indicators(state, u) == [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]

  plan = adaptivity_plan(state, u; tolerance=0.0,
                         limits=AdaptivityLimits(space; min_p=(1, 0), max_p=(1, 0), max_h_level=1))
  for leaf in active_leaves(space)
    @test h_adaptation_axes(plan, leaf) == (false, true)
  end
end

@testset "Coefficient Coarsening Indicators" begin
  domain = Domain((0.0,), (1.0,), (3,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(3)))
  u = ScalarField(space; name=:u)
  values = zeros(scalar_dof_count(space))

  for term in mode_terms(space, 1, (2,))
    values[term.first] = 1 / term.second
  end

  for term in mode_terms(space, 1, (3,))
    values[term.first] = 0.1 / term.second
  end

  for term in mode_terms(space, 2, (2,))
    values[term.first] = 1 / term.second
  end

  for term in mode_terms(space, 2, (3,))
    values[term.first] = 0.1 / term.second
  end

  for term in mode_terms(space, 3, (2,))
    values[term.first] = 0.1 / term.second
  end

  for term in mode_terms(space, 3, (3,))
    values[term.first] = 1 / term.second
  end

  state = State(FieldLayout((u,)), values)
  coarsening = coefficient_coarsening_indicators(state, u)

  @test 0.0 <= coarsening[1][1] < coarsening[3][1] <= 1.0

  plan = adaptivity_plan(state, u; tolerance=0.2,
                         limits=AdaptivityLimits(space; min_h_level=0, max_h_level=0, min_p=2,
                                                 max_p=3))
  @test p_degree_change(plan, 1) == (-1,)
  @test p_degree_change(plan, 2) == (0,)
  @test p_degree_change(plan, 3) == (0,)

  hp_limits = AdaptivityLimits(space; min_h_level=0, max_h_level=1, min_p=2, max_p=4)
  hp_plan = adaptivity_plan(state, u; tolerance=0.05, limits=hp_limits)
  @test h_adaptation_axes(hp_plan, 1) == (false,)
  @test p_degree_change(hp_plan, 1) == (1,)
  @test h_adaptation_axes(hp_plan, 3) == (true,)
  @test p_degree_change(hp_plan, 3) == (0,)

  p_biased_plan = adaptivity_plan(state, u; tolerance=0.05, smoothness_threshold=20.0,
                                  limits=hp_limits)
  @test h_adaptation_axes(p_biased_plan, 3) == (false,)
  @test p_degree_change(p_biased_plan, 3) == (1,)
end

@testset "Planner Limits And Validation" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [0.1, 0.2, 0.3])

  @test_throws ArgumentError AdaptivityLimits(1; min_h_level=(0, 0))
  @test_throws ArgumentError AdaptivityLimits(1; min_h_level=1, max_h_level=0)
  @test_throws ArgumentError AdaptivityLimits(1; min_p=2, max_p=1)
  domain_snapshot = snapshot(grid(domain))
  @test_throws ArgumentError AdaptivityPlan(space, domain, domain_snapshot, NTuple{1,Int}[])
  @test_throws ArgumentError AdaptivityPlan(space, domain, domain_snapshot, [(0,)])
  @test_throws ArgumentError Grico.HCoarseningCandidate(big(typemax(Int)) + 1, 1, (1, 2), (1,))
  dg_space = HpSpace(domain,
                     SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0), continuity=:dg))
  @test AdaptivityPlan(dg_space, domain, domain_snapshot, [(0,)]) isa AdaptivityPlan
  @test_throws ArgumentError AdaptivityLimits(space; min_p=0)
  @test_throws ArgumentError adaptivity_plan(state, u; tolerance=NaN)
  @test_throws ArgumentError adaptivity_plan(state, u; tolerance=-1.0)
  @test_throws ArgumentError adaptivity_plan(state, u; smoothness_threshold=NaN)
  @test_throws ArgumentError adaptivity_plan(state, u; smoothness_threshold=-1.0)
  @test_throws ArgumentError adaptivity_plan(state, u; limits=AdaptivityLimits(2))

  revision_limited_domain = Domain((0.0,), (1.0,), (1,))
  revision_limited_space = HpSpace(revision_limited_domain,
                                   SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  Grico.grid(revision_limited_space).revision = typemax(UInt)
  @test_throws ArgumentError Grico._batched_adaptivity_plan(revision_limited_space, [(0,)],
                                                            [(false,)],
                                                            Grico.HCoarseningCandidate{1}[])

  overflow_plan = AdaptivityPlan(space)
  @test_throws ArgumentError request_p_refinement!(overflow_plan, 1, 1; increment=typemax(Int))
  @test_throws ArgumentError request_p_refinement!(overflow_plan, 1, (typemax(Int),))
  @test_throws ArgumentError request_p_derefinement!(overflow_plan, 1, 1; decrement=typemax(Int))

  plan = AdaptivityPlan(space)
  request_h_refinement!(plan, 1, 1)
  summary = adaptivity_summary(plan)
  @test summary.marked_leaf_count == 1
  @test summary.h_refinement_leaf_count == 1
  @test summary.h_derefinement_cell_count == 0
  @test summary.p_refinement_leaf_count == 0
  @test summary.p_derefinement_leaf_count == 0

  refined_transition = transition(plan)
  refined_space = target_space(refined_transition)
  coarsen_plan = AdaptivityPlan(refined_space)
  request_h_derefinement!(coarsen_plan, 1, 1)
  request_p_refinement!(coarsen_plan, 1, 1)
  request_p_derefinement!(coarsen_plan, 1, 1)
  coarsen_summary = adaptivity_summary(coarsen_plan)
  @test coarsen_summary.h_derefinement_cell_count == 1

  new_v = adapted_field(refined_transition, VectorField(space, 2; name=:v))
  @test_throws ArgumentError transfer_state(refined_transition, state, u, new_v)
end

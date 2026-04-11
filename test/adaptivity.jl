using Test
using Grico
import Grico: coefficient_coarsening_indicators, coefficient_decay_indicators,
              coefficient_indicators, h_adaptation_axes, h_coarsening_candidates,
              interface_jump_indicators, is_domain_boundary, p_degree_change,
              projection_coarsening_indicators, source_leaves, target_space

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

  @test active_leaf_count(target_space(coarsen_transition)) == 1
  @test source_leaves(coarsen_transition, 1) == [2, 3]

  for x in ((0.0,), (0.125,), (0.25,), (0.5,), (0.75,), (1.0,))
    @test _field_value_at_point(coarse_u, coarse_state, x) ≈
          _field_value_at_point(recovered_u, recovered_state, x) atol = ADAPTIVITY_TOL
  end
end

@testset "Automatic H Coarsening" begin
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

  plan = h_adaptivity_plan(fine_state, fine_u; threshold=0.0, h_coarsening_threshold=coarsening_tol)
  summary = adaptivity_summary(plan)
  @test summary.h_derefinement_cell_count == 1
  @test active_leaf_count(target_space(transition(plan))) == 1
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
                               linear_solve=(A, b) -> begin
                                 calls[] += 1
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
    @test _field_value_at_point(u, state, x) ≈ _field_value_at_point(new_u, transferred, x) atol = 5.0e-8
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

@testset "H Adaptivity Planning" begin
  domain = Domain((0.0, 0.0), (1.0, 1.0), (2, 1))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=AxisDegrees((3, 2))))
  u = ScalarField(space; name=:u)
  values = zeros(scalar_dof_count(space))

  for term in mode_terms(space, 1, (3, 2))
    values[term.first] = 1 / term.second
  end

  state = State(FieldLayout((u,)), values)
  indicators = coefficient_indicators(state, u)

  @test indicators[1][1] > 0
  @test indicators[1][2] > 0
  @test indicators[2] == (0.0, 0.0)

  plan = h_adaptivity_plan(space, indicators; threshold=1.0)
  @test h_adaptation_axes(plan, 1) == (true, true)
  @test h_adaptation_axes(plan, 2) == (false, false)
  @test p_degree_change(plan, 1) == (0, 0)

  auto_plan = h_adaptivity_plan(state, u; threshold=1.0)
  @test h_adaptation_axes(auto_plan, 1) == (true, true)
  @test h_adaptation_axes(auto_plan, 2) == (false, false)
end

@testset "Coefficient Indicators At P1" begin
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
  @test coefficient_indicators(constant_state, u)[1][1] ≈ 0.0 atol = ADAPTIVITY_TOL
  @test coefficient_coarsening_indicators(constant_state, u)[1][1] ≈ 0.0 atol = ADAPTIVITY_TOL
  @test coefficient_decay_indicators(constant_state, u)[1][1] ≈ 0.0 atol = ADAPTIVITY_TOL
  @test h_adaptation_axes(h_adaptivity_plan(constant_state, u; threshold=1.0), 1) == (false,)

  linear_values = zeros(scalar_dof_count(space))

  for term in mode_terms(space, 1, (0,))
    linear_values[term.first] -= term.second
  end

  for term in mode_terms(space, 1, (1,))
    linear_values[term.first] += term.second
  end

  linear_state = State(FieldLayout((u,)), linear_values)
  @test coefficient_indicators(linear_state, u)[1][1] ≈ 1.0 atol = ADAPTIVITY_TOL
  @test h_adaptation_axes(h_adaptivity_plan(linear_state, u; threshold=1.0), 1) == (true,)
end

@testset "DG P0 Adaptivity Support" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0), continuity=:dg))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [2.5])

  @test AdaptivityLimits(space).min_p == (0,)
  @test coefficient_indicators(state, u)[1][1] ≈ 0.0 atol = ADAPTIVITY_TOL
  @test coefficient_coarsening_indicators(state, u)[1][1] ≈ 0.0 atol = ADAPTIVITY_TOL
  @test coefficient_decay_indicators(state, u)[1][1] ≈ 0.0 atol = ADAPTIVITY_TOL

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

@testset "DG Jump Refinement Defaults" begin
  domain = Domain((0.0,), (1.0,), (2,))
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0), continuity=:dg))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [1.0, 3.0])
  modal = coefficient_indicators(state, u)
  jumps = interface_jump_indicators(state, u)

  @test modal[1][1] ≈ 0.0 atol = ADAPTIVITY_TOL
  @test modal[2][1] ≈ 0.0 atol = ADAPTIVITY_TOL
  @test jumps[1][1] > 0.0
  @test jumps[1][1] ≈ jumps[2][1] atol = ADAPTIVITY_TOL
  v = ScalarField(space; name=:v)
  mixed_state = State(FieldLayout((v, u)), [10.0, 20.0, 1.0, 3.0])
  mixed_jumps = interface_jump_indicators(mixed_state, u)
  @test mixed_jumps[1][1] ≈ jumps[1][1] atol = ADAPTIVITY_TOL
  @test mixed_jumps[2][1] ≈ jumps[2][1] atol = ADAPTIVITY_TOL

  default_h = h_adaptivity_plan(state, u; threshold=1.0)
  @test h_adaptation_axes(default_h, 1) == (true,)
  @test h_adaptation_axes(default_h, 2) == (true,)

  modal_h = h_adaptivity_plan(state, u; indicator=coefficient_indicators, threshold=1.0)
  @test h_adaptation_axes(modal_h, 1) == (false,)
  @test h_adaptation_axes(modal_h, 2) == (false,)

  default_hp = hp_adaptivity_plan(state, u; threshold=1.0, smoothness_threshold=0.5)
  @test h_adaptation_axes(default_hp, 1) == (true,)
  @test h_adaptation_axes(default_hp, 2) == (true,)
  @test p_degree_change(default_hp, 1) == (0,)
  @test p_degree_change(default_hp, 2) == (0,)
end

@testset "Dorfler Bulk Marking" begin
  domain = Domain((0.0,), (1.0,), (2,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(1)))
  indicators = [(0.2,), (1.0,)]

  focused = h_adaptivity_plan(space, indicators; threshold=0.5)
  @test h_adaptation_axes(focused, 1) == (false,)
  @test h_adaptation_axes(focused, 2) == (true,)

  full = h_adaptivity_plan(space, indicators; threshold=1.0)
  @test h_adaptation_axes(full, 1) == (true,)
  @test h_adaptation_axes(full, 2) == (true,)
end

@testset "Coefficient Coarsening Indicators" begin
  domain = Domain((0.0,), (1.0,), (2,))
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
    values[term.first] = 0.1 / term.second
  end

  for term in mode_terms(space, 2, (3,))
    values[term.first] = 1 / term.second
  end

  state = State(FieldLayout((u,)), values)
  coarsening = coefficient_coarsening_indicators(state, u)

  @test 0.0 <= coarsening[1][1] < coarsening[2][1] <= 1.0

  plan = p_adaptivity_plan(space, [(0.0,), (0.0,)], coarsening; threshold=0.0,
                           p_coarsening_threshold=0.2)
  @test p_degree_change(plan, 1) == (-1,)
  @test p_degree_change(plan, 2) == (0,)
end

@testset "Custom Indicator Hooks" begin
  domain = Domain((0.0,), (1.0,), (2,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), zeros(scalar_dof_count(space)))
  refinement = (_state, _field) -> [(0.0,), (1.0,)]
  smoothness = (_state, _field) -> [(1.0,), (0.0,)]
  p_coarsening = (_state, _field) -> [(0.0,), (1.0,)]

  plan = hp_adaptivity_plan(state, u; indicator=refinement, smoothness_indicator=smoothness,
                            smoothness_threshold=0.5, p_coarsening_indicator=p_coarsening,
                            p_coarsening_threshold=0.1, threshold=1.0)
  @test h_adaptation_axes(plan, 1) == (false,)
  @test p_degree_change(plan, 1) == (-1,)
  @test h_adaptation_axes(plan, 2) == (false,)
  @test p_degree_change(plan, 2) == (1,)

  precomputed = hp_adaptivity_plan(space, [(1.0,), (0.0,)], [(1.0,), (0.0,)]; threshold=1.0,
                                   smoothness_threshold=0.5)
  @test h_adaptation_axes(precomputed, 1) == (true,)
  @test p_degree_change(precomputed, 1) == (0,)
  @test h_adaptation_axes(precomputed, 2) == (false,)

  coarse_domain = Domain((0.0,), (1.0,), (1,))
  coarse_space = HpSpace(coarse_domain,
                         SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  coarse_u = ScalarField(coarse_space; name=:u)
  coarse_state = State(FieldLayout((coarse_u,)), [0.2, -0.1, 0.6])
  refine_plan = AdaptivityPlan(coarse_space)
  request_h_refinement!(refine_plan, 1, 1)
  refine_transition = transition(refine_plan)
  fine_u = adapted_field(refine_transition, coarse_u)
  fine_state = transfer_state(refine_transition, coarse_state, coarse_u, fine_u)
  candidates = h_coarsening_candidates(field_space(fine_u))
  h_coarsening = (_state, _field, _candidates) -> [0.0]

  coarsened = h_adaptivity_plan(fine_state, fine_u; threshold=0.0,
                                h_coarsening_candidates=candidates,
                                h_coarsening_indicator=h_coarsening, h_coarsening_threshold=0.1)
  @test adaptivity_summary(coarsened).h_derefinement_cell_count == 1
end

@testset "Coefficient Hp Planning" begin
  domain = Domain((0.0,), (1.0,), (2,))
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
    values[term.first] = 0.1 / term.second
  end

  for term in mode_terms(space, 2, (3,))
    values[term.first] = 1 / term.second
  end

  state = State(FieldLayout((u,)), values)
  decay = coefficient_decay_indicators(state, u)
  @test decay[1][1] < 1
  @test decay[2][1] > 1

  p_plan = p_adaptivity_plan(state, u; threshold=1.0)
  @test p_degree_change(p_plan, 1) == (1,)
  @test p_degree_change(p_plan, 2) == (1,)

  hp_plan = hp_adaptivity_plan(state, u; threshold=1.0, smoothness_threshold=0.5)
  @test p_degree_change(hp_plan, 1) == (1,)
  @test h_adaptation_axes(hp_plan, 1) == (false,)
  @test p_degree_change(hp_plan, 2) == (0,)
  @test h_adaptation_axes(hp_plan, 2) == (true,)
end

@testset "Hp Planning Fallback" begin
  domain = Domain((0.0,), (1.0,), (2,))
  refine!(grid(domain), 1, 1)
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(),
                               degree=ByLeafDegrees((_, leaf) -> leaf == 2 ? (3,) : (2,))))
  limits = AdaptivityLimits(1; max_h_level=1, max_p=3)
  plan = hp_adaptivity_plan(space, [(1.0,), (1.0,), (0.0,)], [(0.0,), (1.0,), (0.0,)];
                            threshold=1.0, smoothness_threshold=0.5, limits=limits)

  @test h_adaptation_axes(plan, 2) == (true,)
  @test p_degree_change(plan, 2) == (0,)
  @test h_adaptation_axes(plan, 3) == (false,)
  @test p_degree_change(plan, 3) == (1,)
  @test h_adaptation_axes(plan, 4) == (false,)
  @test p_degree_change(plan, 4) == (0,)
end

@testset "Planner Limits And Validation" begin
  domain = Domain((0.0,), (1.0,), (1,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(2)))
  u = ScalarField(space; name=:u)
  state = State(FieldLayout((u,)), [0.1, 0.2, 0.3])

  @test_throws ArgumentError AdaptivityLimits(1; min_h_level=(0, 0))
  @test_throws ArgumentError AdaptivityLimits(1; min_h_level=1, max_h_level=0)
  @test_throws ArgumentError AdaptivityLimits(1; min_p=2, max_p=1)
  @test_throws ArgumentError AdaptivityPlan(space, domain, NTuple{1,Int}[])
  @test_throws ArgumentError AdaptivityPlan(space, domain, [(0,)])
  dg_space = HpSpace(domain,
                     SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(0), continuity=:dg))
  @test AdaptivityPlan(dg_space, domain, [(0,)]) isa AdaptivityPlan
  @test_throws ArgumentError AdaptivityLimits(space; min_p=0)
  @test_throws ArgumentError h_adaptivity_plan(space, [(Inf,)])
  @test_throws ArgumentError h_adaptivity_plan(space, [(0.0,)]; h_coarsening_threshold=0.1)
  @test_throws ArgumentError p_adaptivity_plan(space, [(NaN,)])
  @test_throws ArgumentError p_adaptivity_plan(space, [(0.0,)]; p_coarsening_threshold=0.1)
  @test_throws ArgumentError p_adaptivity_plan(state, u;
                                               p_coarsening_indicator=(_state, _field) -> [(NaN,)],
                                               p_coarsening_threshold=0.1)
  @test_throws ArgumentError hp_adaptivity_plan(state, u; indicator=(_state, _field) -> [(NaN,)])
  @test_throws ArgumentError hp_adaptivity_plan(state, u;
                                                smoothness_indicator=(_state, _field) -> [(NaN,)])
  @test_throws ArgumentError h_adaptivity_plan(state, u;
                                               h_coarsening_indicator=(_state, _field, _candidates) -> [NaN],
                                               h_coarsening_threshold=0.1)

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

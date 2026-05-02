using Test
using Grico

const SPACE_TOL = 1.0e-12

function _space_value(space, leaf, ξ, coefficients)
  one_dimensional = ntuple(axis -> Grico._fe_basis_values(ξ[axis],
                                                          Grico.cell_degrees(space, leaf)[axis]),
                           length(ξ))
  value = 0.0

  for mode in Grico.local_modes(space, leaf)
    local_value = 1.0

    for axis in 1:length(ξ)
      local_value *= one_dimensional[axis][mode[axis]+1]
    end

    for term in Grico.mode_terms(space, leaf, mode)
      value += local_value * term.second * coefficients[term.first]
    end
  end

  return value
end

function _test_term_vector(actual::Vector{<:Pair}, expected::Vector{<:Pair}; atol::Real=SPACE_TOL)
  @test length(actual) == length(expected)

  for index in eachindex(expected)
    @test actual[index].first == expected[index].first
    @test actual[index].second ≈ expected[index].second atol = atol
  end
end

@testset "Policy Validation" begin
  @test Grico.UniformDegree(0) isa Grico.UniformDegree
  @test Grico.AxisDegrees((0, 1)) isa Grico.AxisDegrees{2}
  @test Grico.AxisDegrees{2}((0, 1)) isa Grico.AxisDegrees{2}
  too_large = big(typemax(Int)) + 1
  @test_throws ArgumentError Grico.UniformDegree(too_large)
  @test_throws ArgumentError Grico.AxisDegrees((too_large,))
  @test_throws ArgumentError Grico.DegreePlusQuadrature(too_large)
  @test_throws ArgumentError Grico.DegreePlusQuadrature(-1)
  @test_throws ArgumentError Grico.SpaceOptions(continuity=:foo)
  @test_throws ArgumentError Grico.SpaceOptions(continuity=(:cg, :foo))
  @test_throws ArgumentError Grico.SpaceOptions(continuity=())
  @test_throws ArgumentError Grico.SpaceOptions(continuity=(:cg, 1))

  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))
  compiled = only(space.compiled_leaves)
  space.compiled_leaves[1] = Grico._CompiledLeaf(compiled.leaf, compiled.degrees,
                                                 compiled.support_shape, compiled.local_modes,
                                                 compiled.mode_lookup, compiled.term_offsets,
                                                 compiled.term_indices, compiled.term_coefficients,
                                                 compiled.single_term_indices,
                                                 compiled.single_term_coefficients, (0,))
  @test_throws ArgumentError Grico.check_space(space)

  lookup_space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))
  lookup_compiled = only(lookup_space.compiled_leaves)
  bad_lookup = copy(lookup_compiled.mode_lookup)
  fill!(bad_lookup, 0)
  lookup_space.compiled_leaves[1] = Grico._CompiledLeaf(lookup_compiled.leaf,
                                                        lookup_compiled.degrees,
                                                        lookup_compiled.support_shape,
                                                        lookup_compiled.local_modes, bad_lookup,
                                                        lookup_compiled.term_offsets,
                                                        lookup_compiled.term_indices,
                                                        lookup_compiled.term_coefficients,
                                                        lookup_compiled.single_term_indices,
                                                        lookup_compiled.single_term_coefficients,
                                                        lookup_compiled.quadrature_shape)
  @test_throws ArgumentError Grico.check_space(lookup_space)

  offset_space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))
  offset_compiled = only(offset_space.compiled_leaves)
  bad_offsets = copy(offset_compiled.term_offsets)
  bad_offsets[3] = bad_offsets[2]
  offset_space.compiled_leaves[1] = Grico._CompiledLeaf(offset_compiled.leaf,
                                                        offset_compiled.degrees,
                                                        offset_compiled.support_shape,
                                                        offset_compiled.local_modes,
                                                        offset_compiled.mode_lookup, bad_offsets,
                                                        offset_compiled.term_indices,
                                                        offset_compiled.term_coefficients,
                                                        offset_compiled.single_term_indices,
                                                        offset_compiled.single_term_coefficients,
                                                        offset_compiled.quadrature_shape)
  @test_throws ArgumentError Grico.check_space(offset_space)
end

@testset "Continuity Policy API" begin
  domain_2d = Grico.Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  default_space = Grico.HpSpace(domain_2d, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))
  tuple_space = Grico.HpSpace(domain_2d,
                              Grico.SpaceOptions(degree=Grico.UniformDegree(2),
                                                 continuity=(:cg, :cg)))
  dg_space = Grico.HpSpace(domain_2d,
                           Grico.SpaceOptions(degree=Grico.UniformDegree(2), continuity=:dg))
  mixed_space = Grico.HpSpace(domain_2d,
                              Grico.SpaceOptions(degree=Grico.AxisDegrees((1, 0)),
                                                 continuity=(:cg, :dg)))

  @test Grico.continuity_policy(default_space) == (:cg, :cg)
  @test Grico.continuity_policy(tuple_space) == (:cg, :cg)
  @test Grico.continuity_policy(dg_space) == (:dg, :dg)
  @test Grico.continuity_policy(mixed_space) == (:cg, :dg)
  @test Grico.continuity_kind(default_space, 1) == :cg
  @test Grico.continuity_kind(default_space, 2) == :cg
  @test Grico.continuity_kind(dg_space, 1) == :dg
  @test Grico.continuity_kind(dg_space, 2) == :dg
  @test Grico.continuity_kind(mixed_space, 1) == :cg
  @test Grico.continuity_kind(mixed_space, 2) == :dg
  @test Grico.is_continuous_axis(default_space, 1)
  @test Grico.is_continuous_axis(default_space, 2)
  @test !Grico.is_continuous_axis(dg_space, 1)
  @test !Grico.is_continuous_axis(dg_space, 2)
  @test Grico.is_continuous_axis(mixed_space, 1)
  @test !Grico.is_continuous_axis(mixed_space, 2)
  @test_throws ArgumentError Grico.continuity_kind(default_space, 3)
  @test_throws ArgumentError Grico.is_mode_active(default_space, 1, (0,))
  @test_throws ArgumentError Grico.is_mode_active(default_space, 1, (0, 0.0))
  @test_throws ArgumentError Grico.mode_terms(default_space, 1, (0,))
  @test_throws ArgumentError Grico.mode_terms(default_space, 1, [0, 0])
  @test_throws ArgumentError Grico.HpSpace(domain_2d,
                                           Grico.SpaceOptions(degree=Grico.UniformDegree(0),
                                                              continuity=:cg))
  @test Grico.HpSpace(domain_2d,
                      Grico.SpaceOptions(degree=Grico.UniformDegree(0), continuity=:dg)) isa
        Grico.HpSpace

  @test_throws ArgumentError Grico.HpSpace(domain_2d,
                                           Grico.SpaceOptions(degree=Grico.AxisDegrees((0, 1)),
                                                              continuity=(:cg, :dg)))

  domain_1d = Grico.Domain((0.0,), (1.0,), (1,))
  @test_throws ArgumentError Grico.HpSpace(domain_1d,
                                           Grico.SpaceOptions(degree=Grico.UniformDegree(2),
                                                              continuity=(:cg, :cg)))

  snapshot_domain = Grico.Domain((0.0,), (1.0,), (1,))
  snapshot_space = Grico.HpSpace(snapshot_domain, Grico.SpaceOptions(degree=Grico.UniformDegree(1)))
  Grico.refine!(Grico.grid(snapshot_domain), 1, 1)
  @test Grico.active_leaves(snapshot_space) == [1]
  @test Grico.active_leaves(Grico.grid(snapshot_space)) == [1]
  @test Grico.snapshot(snapshot_space) isa Grico.GridSnapshot{1}
  @test Grico.grid(Grico.snapshot(snapshot_space)) === Grico.grid(snapshot_space)
  @test Grico.check_snapshot(Grico.snapshot(snapshot_space)) === nothing
  @test Grico.boundary_face_count(Grico.snapshot(snapshot_space)) == 2
  @test Grico.interface_count(Grico.snapshot(snapshot_space)) == 0
  @test Grico.check_space(snapshot_space) === nothing
end

@testset "Physical Domains" begin
  background = Grico.Domain((0.0,), (3.0,), (3,))
  region = Grico.ImplicitRegion(x -> x[1] - 1.5; subdivision_depth=1)
  domain = Grico.PhysicalDomain(background, region)

  default_space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(1)))

  @test Grico.active_leaves(default_space) == [1, 2]
  @test Grico.active_leaves(Grico.snapshot(default_space)) == [1, 2]
  @test Grico.check_snapshot(Grico.snapshot(default_space)) === nothing
  @test Grico.interface_count(Grico.snapshot(default_space)) == 1

  sliver_background = Grico.Domain((0.0,), (1.0,), (2,))
  sliver_region = Grico.ImplicitRegion(x -> x[1] <= 0.05 || x[1] >= 0.75; subdivision_depth=0)
  sliver_domain = Grico.PhysicalDomain(sliver_background, sliver_region)
  sliver_space = Grico.HpSpace(sliver_domain, Grico.SpaceOptions(degree=Grico.UniformDegree(1)))

  @test Grico.active_leaves(sliver_space) == [1, 2]

  shared_region = Grico.ImplicitRegion(x -> x[1] - 0.5; subdivision_depth=0)
  left_domain = Grico.PhysicalDomain(Grico.Domain((0.0,), (1.0,), (1,)), shared_region)
  extended_domain = Grico.PhysicalDomain(Grico.Domain((0.0,), (1.0,), (1,)), shared_region;
                                         cell_measure=Grico.FiniteCellExtension(0.1))
  shifted_domain = Grico.PhysicalDomain(Grico.Domain((2.0,), (3.0,), (1,)), shared_region)

  @test Grico._domain_active_leaves(left_domain) == [1]
  @test Grico._domain_active_leaves(copy(left_domain)) == [1]
  @test Grico._cell_measure(copy(extended_domain)) == Grico.FiniteCellExtension(0.1)
  @test_throws ArgumentError Grico._domain_active_leaves(shifted_domain)

  left_space = Grico.HpSpace(left_domain, Grico.SpaceOptions(degree=Grico.UniformDegree(1)))
  extended_space = Grico.HpSpace(extended_domain, Grico.SpaceOptions(degree=Grico.UniformDegree(1)))
  @test_throws ArgumentError Grico.FieldLayout((Grico.ScalarField(left_space),
                                                Grico.ScalarField(extended_space)))
end

@testset "Field And State Validation" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  v = Grico.VectorField(space, 2; name=:v)
  layout = Grico.FieldLayout((u,))
  mixed_layout = Grico.FieldLayout((u, v))
  slot = only(layout.slots)
  state = Grico.State(layout, [0.1, -0.2, 0.3])
  mixed_state = Grico.State(mixed_layout, collect(0.1:0.1:0.9))

  @test Grico.field_dof_range(mixed_layout, u) == 1:3
  @test Grico.field_dof_range(mixed_layout, v) == 4:9
  @test Grico.field_component_range(mixed_layout, v, 1) == 4:6
  @test Grico.field_component_range(mixed_layout, v, 2) == 7:9
  @test collect(Grico.field_values(mixed_state, v)) ≈ collect(0.4:0.1:0.9)
  @test collect(Grico.field_component_values(mixed_state, v, 2)) ≈ collect(0.7:0.1:0.9)

  too_large = big(typemax(Int)) + 1
  @test_throws ArgumentError Grico.VectorField(space, too_large)
  @test_throws ArgumentError Grico.VectorField(1, space, too_large, :v)
  @test_throws ArgumentError Grico.VectorField(1, space, 0, :v)
  @test_throws ArgumentError Grico.FieldLayout((1,))
  @test_throws ArgumentError Grico.FieldLayout((u, u))
  @test_throws ArgumentError typeof(layout)(typeof(layout.slots)(), 0)
  @test_throws ArgumentError typeof(layout)([slot], slot.dof_count - 1)
  @test_throws ArgumentError typeof(layout)([Grico._FieldSlot(slot.field, slot.space, 2,
                                                              slot.scalar_dof_count,
                                                              slot.dof_count)], slot.dof_count)
  @test_throws ArgumentError Grico.State(layout, [1, 2, 3])
  @test_throws ArgumentError typeof(state)(layout, [0.1, -0.2])
  @test_throws ArgumentError Grico.field_dof_range(layout, v)
  @test_throws ArgumentError Grico.field_values(state, v)

  first_region = Grico.ImplicitRegion(x -> x[1] - 0.25; subdivision_depth=1)
  second_region = Grico.ImplicitRegion(x -> x[1] - 0.75; subdivision_depth=1)
  first_physical = Grico.PhysicalDomain(Grico.Domain((0.0,), (1.0,), (1,)), first_region)
  second_physical = Grico.PhysicalDomain(Grico.Domain((0.0,), (1.0,), (1,)), second_region)
  first_field = Grico.ScalarField(Grico.HpSpace(first_physical); name=:first)
  second_field = Grico.ScalarField(Grico.HpSpace(second_physical); name=:second)
  @test Grico.active_leaves(Grico.field_space(first_field)) ==
        Grico.active_leaves(Grico.field_space(second_field))
  @test_throws ArgumentError Grico.FieldLayout((first_field, second_field))
end

@testset "HpSpace Single Cell" begin
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.AxisDegrees((2, 1))))

  @test Grico.scalar_dof_count(space) == 6
  @test Grico.local_mode_count(space, 1) == 6
  @test Grico.support_shape(space, 1) == (3, 2)
  @test Grico.cell_degrees(space, 1) == (2, 1)
  @test Grico.cell_quadrature_shape(space, 1) == (3, 2)
  @test Grico.global_cell_quadrature_shape(space) == (3, 2)
  @test Grico.is_mode_active(space, 1, (2, 1))
  @test length(Grico.mode_terms(space, 1, (2, 1))) == 1
  @test only(Grico.mode_terms(space, 1, (2, 1))).second ≈ 1.0 atol = SPACE_TOL
  @test Grico.check_space(space) === nothing
end

@testset "Compiled Leaf Regression" begin
  single_cell_domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  single_cell_space = Grico.HpSpace(single_cell_domain,
                                    Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                       degree=Grico.AxisDegrees((2, 1))))

  @test Grico.scalar_dof_count(single_cell_space) == 6
  _test_term_vector(Grico.mode_terms(single_cell_space, 1, (0, 0)), [1 => 1.0])
  _test_term_vector(Grico.mode_terms(single_cell_space, 1, (1, 0)), [2 => 1.0])
  _test_term_vector(Grico.mode_terms(single_cell_space, 1, (2, 0)), [3 => 1.0])
  _test_term_vector(Grico.mode_terms(single_cell_space, 1, (0, 1)), [4 => 1.0])
  _test_term_vector(Grico.mode_terms(single_cell_space, 1, (1, 1)), [5 => 1.0])
  _test_term_vector(Grico.mode_terms(single_cell_space, 1, (2, 1)), [6 => 1.0])

  hanging_domain = Grico.Domain((0.0, 0.0), (2.0, 1.0), (2, 1))
  hanging_grid = Grico.grid(hanging_domain)
  Grico.refine!(hanging_grid, 2, 2)
  hanging_space = Grico.HpSpace(hanging_domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))

  @test Grico.scalar_dof_count(hanging_space) == 16
  _test_term_vector(Grico.mode_terms(hanging_space, 3, (0, 1)),
                    [2 => 0.5, 5 => 0.5, 8 => -0.6123724356957945])
  _test_term_vector(Grico.mode_terms(hanging_space, 4, (0, 0)),
                    [2 => 0.5, 5 => 0.5, 8 => -0.6123724356957945])
end

@testset "DG Space Compilation" begin
  uniform_domain = Grico.Domain((0.0,), (1.0,), (2,))
  uniform_space = Grico.HpSpace(uniform_domain,
                                Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                   degree=Grico.UniformDegree(3), continuity=:dg))

  @test Grico.scalar_dof_count(uniform_space) == 8
  @test Grico.scalar_dof_count(uniform_space) == sum(Grico.local_mode_count(uniform_space, leaf)
                                                     for leaf in Grico.active_leaves(uniform_space))
  _test_term_vector(Grico.mode_terms(uniform_space, 1, (1,)), [2 => 1.0])
  _test_term_vector(Grico.mode_terms(uniform_space, 2, (0,)), [5 => 1.0])

  p0_space = Grico.HpSpace(uniform_domain,
                           Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                              degree=Grico.UniformDegree(0), continuity=:dg))
  @test Grico.scalar_dof_count(p0_space) == 2
  @test Grico.cell_quadrature_shape(p0_space, 1) == (1,)
  @test Grico.support_shape(p0_space, 1) == (1,)
  @test Grico.local_mode_count(p0_space, 1) == 1
  _test_term_vector(Grico.mode_terms(p0_space, 1, (0,)), [1 => 1.0])
  _test_term_vector(Grico.mode_terms(p0_space, 2, (0,)), [2 => 1.0])
  @test _space_value(p0_space, 1, (-1.0,), [2.5, -1.0]) ≈ 2.5 atol = SPACE_TOL
  @test _space_value(p0_space, 1, (0.0,), [2.5, -1.0]) ≈ 2.5 atol = SPACE_TOL
  @test _space_value(p0_space, 1, (1.0,), [2.5, -1.0]) ≈ 2.5 atol = SPACE_TOL

  minimal_quadrature_space = Grico.HpSpace(uniform_domain,
                                           Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                              degree=Grico.UniformDegree(0),
                                                              quadrature=Grico.DegreePlusQuadrature(0),
                                                              continuity=:dg))
  @test Grico.cell_quadrature_shape(minimal_quadrature_space, 1) == (1,)
  @test Grico.check_space(minimal_quadrature_space) === nothing

  anisotropic_domain = Grico.Domain((0.0, 0.0), (2.0, 1.0), (2, 1))
  anisotropic_space = Grico.HpSpace(anisotropic_domain,
                                    Grico.SpaceOptions(degree=Grico.ByLeafDegrees((domain, leaf) -> leaf ==
                                                                                                    1 ?
                                                                                                    (1,
                                                                                                     1) :
                                                                                                    (1,
                                                                                                     3)),
                                                       continuity=:dg))
  @test Grico.scalar_dof_count(anisotropic_space) ==
        sum(Grico.local_mode_count(anisotropic_space, leaf)
            for leaf in Grico.active_leaves(anisotropic_space))

  hanging_domain = Grico.Domain((0.0, 0.0), (2.0, 1.0), (2, 1))
  hanging_grid = Grico.grid(hanging_domain)
  Grico.refine!(hanging_grid, 2, 2)
  hanging_space = Grico.HpSpace(hanging_domain,
                                Grico.SpaceOptions(degree=Grico.UniformDegree(2), continuity=:dg))

  @test Grico.scalar_dof_count(hanging_space) == 24
  @test Grico.scalar_dof_count(hanging_space) == sum(Grico.local_mode_count(hanging_space, leaf)
                                                     for leaf in Grico.active_leaves(hanging_space))
  _test_term_vector(Grico.mode_terms(hanging_space, 3, (0, 1)), [12 => 1.0])
  _test_term_vector(Grico.mode_terms(hanging_space, 4, (0, 0)), [17 => 1.0])
  @test Grico.check_space(hanging_space) === nothing
end

@testset "Mixed Space Compilation" begin
  domain = Grico.Domain((0.0, 0.0), (2.0, 2.0), (2, 2))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.AxisDegrees((1, 0)), continuity=(:cg, :dg)))

  @test Grico.scalar_dof_count(space) == 6
  _test_term_vector(Grico.mode_terms(space, 1, (0, 0)), [1 => 1.0])
  _test_term_vector(Grico.mode_terms(space, 2, (0, 0)), [2 => 1.0])
  _test_term_vector(Grico.mode_terms(space, 1, (1, 0)), [2 => 1.0])
  _test_term_vector(Grico.mode_terms(space, 2, (1, 0)), [3 => 1.0])
  _test_term_vector(Grico.mode_terms(space, 3, (0, 0)), [4 => 1.0])
  _test_term_vector(Grico.mode_terms(space, 4, (0, 0)), [5 => 1.0])
  _test_term_vector(Grico.mode_terms(space, 3, (1, 0)), [5 => 1.0])
  _test_term_vector(Grico.mode_terms(space, 4, (1, 0)), [6 => 1.0])
  @test Grico.check_space(space) === nothing

  coefficients = collect(1.0:Grico.scalar_dof_count(space))

  for y in (-1.0, -0.25, 0.5, 1.0)
    @test _space_value(space, 1, (1.0, y), coefficients) ≈
          _space_value(space, 2, (-1.0, y), coefficients) atol = SPACE_TOL
    @test _space_value(space, 3, (1.0, y), coefficients) ≈
          _space_value(space, 4, (-1.0, y), coefficients) atol = SPACE_TOL
  end

  discontinuous = zeros(Grico.scalar_dof_count(space))
  discontinuous[1] = 1.0
  @test abs(_space_value(space, 1, (-1.0, 1.0), discontinuous) -
            _space_value(space, 3, (-1.0, -1.0), discontinuous)) > SPACE_TOL

  flipped = Grico.HpSpace(domain,
                          Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                             degree=Grico.AxisDegrees((0, 1)),
                                             continuity=(:dg, :cg)))
  @test Grico.scalar_dof_count(flipped) == 6
  _test_term_vector(Grico.mode_terms(flipped, 1, (0, 0)), [1 => 1.0])
  _test_term_vector(Grico.mode_terms(flipped, 3, (0, 0)), [2 => 1.0])
  _test_term_vector(Grico.mode_terms(flipped, 1, (0, 1)), [2 => 1.0])
  _test_term_vector(Grico.mode_terms(flipped, 3, (0, 1)), [5 => 1.0])
  @test Grico.check_space(flipped) === nothing

  mismatch_domain = Grico.Domain((0.0, 0.0), (2.0, 1.0), (2, 1))
  mismatch_space = Grico.HpSpace(mismatch_domain,
                                 Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                    degree=Grico.ByLeafDegrees((domain, leaf) -> leaf ==
                                                                                                 1 ?
                                                                                                 (1,
                                                                                                  0) :
                                                                                                 (1,
                                                                                                  2)),
                                                    continuity=(:cg, :dg)))
  @test Grico.check_space(mismatch_space) === nothing
  mismatch_coefficients = collect(1.0:Grico.scalar_dof_count(mismatch_space))

  for y in (-1.0, -0.5, 0.0, 0.5, 1.0)
    @test _space_value(mismatch_space, 1, (1.0, y), mismatch_coefficients) ≈
          _space_value(mismatch_space, 2, (-1.0, y), mismatch_coefficients) atol = SPACE_TOL
  end
end

@testset "Same-Level P Mismatch" begin
  domain = Grico.Domain((0.0, 0.0), (2.0, 1.0), (2, 1))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(degree=Grico.ByLeafDegrees((domain, leaf) -> leaf == 1 ?
                                                                                        (1, 1) :
                                                                                        (1, 3))))

  @test Grico.scalar_dof_count(space) == 8
  @test Grico.support_shape(space, 1) == (2, 2)
  @test Grico.support_shape(space, 2) == (2, 4)
  @test !Grico.is_mode_active(space, 2, (0, 2))
  @test Grico.is_mode_active(space, 2, (1, 3))
  @test Grico.global_cell_quadrature_shape(space) == (2, 4)
  @test Grico.check_space(space) === nothing

  coefficients = collect(1.0:Grico.scalar_dof_count(space))

  for y in (-1.0, -0.25, 0.5)
    @test _space_value(space, 1, (1.0, y), coefficients) ≈
          _space_value(space, 2, (-1.0, y), coefficients) atol = SPACE_TOL
  end

  domain_3d = Grico.Domain((0.0, 0.0, 0.0), (2.0, 1.0, 1.0), (2, 1, 1))
  space_3d = Grico.HpSpace(domain_3d,
                           Grico.SpaceOptions(degree=Grico.ByLeafDegrees((domain, leaf) -> leaf ==
                                                                                           1 ?
                                                                                           (1, 1,
                                                                                            1) :
                                                                                           (1, 3, 2))))
  coefficients_3d = collect(1.0:Grico.scalar_dof_count(space_3d))

  for yz in ((-0.5, -0.5), (-0.25, 0.25), (0.75, -0.25))
    y, z = yz
    @test _space_value(space_3d, 1, (1.0, y, z), coefficients_3d) ≈
          _space_value(space_3d, 2, (-1.0, y, z), coefficients_3d) atol = SPACE_TOL
  end
end

@testset "Mixed Hanging Continuity" begin
  domain = Grico.Domain((0.0, 0.0), (2.0, 1.0), (2, 1))
  grid = Grico.grid(domain)
  first_child = Grico.refine!(grid, 2, 2)
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(degree=Grico.UniformDegree(2), continuity=(:cg, :dg)))
  @test Grico.check_space(space) === nothing

  function wrapped_value(y, coefficients)
    return y < 0 ? _space_value(space, first_child, (-1.0, 2 * y + 1), coefficients) :
           _space_value(space, first_child + 1, (-1.0, 2 * y - 1), coefficients)
  end

  coefficients = collect(1.0:Grico.scalar_dof_count(space))

  for y in (-0.75, -0.25, 0.25, 0.75)
    @test _space_value(space, 1, (1.0, y), coefficients) ≈ wrapped_value(y, coefficients) atol = SPACE_TOL
  end

  p0_tangential_domain = Grico.Domain((0.0, 0.0), (2.0, 1.0), (2, 1))
  p0_tangential_grid = Grico.grid(p0_tangential_domain)
  p0_first_child = Grico.refine!(p0_tangential_grid, 2, 2)
  p0_space = Grico.HpSpace(p0_tangential_domain,
                           Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                              degree=Grico.AxisDegrees((1, 0)),
                                              continuity=(:cg, :dg)))
  @test Grico.check_space(p0_space) === nothing

  function p0_wrapped_value(y, coefficients)
    return y < 0 ? _space_value(p0_space, p0_first_child, (-1.0, 2 * y + 1), coefficients) :
           _space_value(p0_space, p0_first_child + 1, (-1.0, 2 * y - 1), coefficients)
  end

  p0_coefficients = collect(1.0:Grico.scalar_dof_count(p0_space))

  for y in (-0.75, -0.25, 0.25, 0.75)
    @test _space_value(p0_space, 1, (1.0, y), p0_coefficients) ≈
          p0_wrapped_value(y, p0_coefficients) atol = SPACE_TOL
  end
end

@testset "Float32 Hanging Continuity" begin
  constant_restriction = Grico._affine_restriction_matrix(0, 0, 1, 0, Float32)
  @test constant_restriction isa Matrix{Float32}
  @test constant_restriction == ones(Float32, 1, 1)

  domain = Grico.Domain((0.0f0, 0.0f0), (2.0f0, 1.0f0), (2, 1))
  grid = Grico.grid(domain)
  first_child = Grico.refine!(grid, 2, 2)
  space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(3)))
  @test Grico.check_space(space) === nothing

  terms = Grico.mode_terms(space, first_child, (0, 1))
  @test !isempty(terms)
  @test all(term -> term.second isa Float32, terms)

  function wrapped_value(y, coefficients)
    return y < 0 ? _space_value(space, first_child, (-1.0f0, 2 * y + 1), coefficients) :
           _space_value(space, first_child + 1, (-1.0f0, 2 * y - 1), coefficients)
  end

  coefficients = Float32.(1:Grico.scalar_dof_count(space))

  for y in Float32[-0.75, -0.25, 0.25, 0.75]
    @test _space_value(space, 1, (1.0f0, y), coefficients) ≈ wrapped_value(y, coefficients) atol = 5.0e-5
  end
end

@testset "Periodic Continuity" begin
  ring_domain = Grico.Domain((0.0,), (1.0,), (2,); periodic=true)
  ring_space = Grico.HpSpace(ring_domain,
                             Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                degree=Grico.UniformDegree(1)))
  @test Grico.scalar_dof_count(ring_space) == 2
  @test Grico.check_space(ring_space) === nothing

  mixed_periodic_domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (2, 1);
                                       periodic=(true, false))
  mixed_periodic_space = Grico.HpSpace(mixed_periodic_domain,
                                       Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                          degree=Grico.AxisDegrees((1, 0)),
                                                          continuity=(:cg, :dg)))
  @test Grico.check_space(mixed_periodic_space) === nothing
  mixed_periodic_coefficients = collect(1.0:Grico.scalar_dof_count(mixed_periodic_space))
  @test _space_value(mixed_periodic_space, 1, (-1.0, 0.0),
                     mixed_periodic_coefficients) ≈
        _space_value(mixed_periodic_space, 2, (1.0, 0.0),
                     mixed_periodic_coefficients) atol = SPACE_TOL
  @test _space_value(mixed_periodic_space, 1, (1.0, 0.0),
                     mixed_periodic_coefficients) ≈
        _space_value(mixed_periodic_space, 2, (-1.0, 0.0),
                     mixed_periodic_coefficients) atol = SPACE_TOL

  single_domain = Grico.Domain((0.0,), (1.0,), (1,); periodic=true)
  single_space = Grico.HpSpace(single_domain,
                               Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                  degree=Grico.UniformDegree(3)))
  coefficients = collect(1.0:Grico.scalar_dof_count(single_space))
  @test _space_value(single_space, 1, (-1.0,), coefficients) ≈
        _space_value(single_space, 1, (1.0,), coefficients) atol = SPACE_TOL

  hanging_domain = Grico.Domain((0.0, 0.0), (2.0, 1.0), (2, 1); periodic=(true, false))
  hanging_grid = Grico.grid(hanging_domain)
  first_child = Grico.refine!(hanging_grid, 2, 2)
  hanging_space = Grico.HpSpace(hanging_domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))
  @test Grico.check_space(hanging_space) === nothing

  function wrapped_value(y, coefficients)
    return y < 0 ? _space_value(hanging_space, first_child, (1.0, 2 * y + 1), coefficients) :
           _space_value(hanging_space, first_child + 1, (1.0, 2 * y - 1), coefficients)
  end

  hanging_coefficients = collect(1.0:Grico.scalar_dof_count(hanging_space))

  for y in (-0.75, -0.25, 0.25, 0.75)
    @test _space_value(hanging_space, 1, (-1.0, y), hanging_coefficients) ≈
          wrapped_value(y, hanging_coefficients) atol = SPACE_TOL
  end
end

@testset "One-Dimensional Hanging" begin
  domain = Grico.Domain((0.0,), (2.0,), (2,))
  grid = Grico.grid(domain)
  first_child = Grico.refine!(grid, 1, 1)
  right_child = first_child + 1
  space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(3)))

  @test Grico.scalar_dof_count(space) == 10
  @test Grico.local_modes(space, 2) == [(0,), (1,), (2,), (3,)]
  @test Grico.local_modes(space, right_child) == [(0,), (1,), (2,), (3,)]
  @test_throws ArgumentError Grico.cell_degrees(space, 1)
  @test_throws ArgumentError Grico.mode_terms(space, 1, (0,))
  @test Grico.check_space(space) === nothing

  coefficients = collect(1.0:Grico.scalar_dof_count(space))
  @test _space_value(space, right_child, (1.0,), coefficients) ≈
        _space_value(space, 2, (-1.0,), coefficients) atol = SPACE_TOL
end

@testset "Two-Dimensional Hanging" begin
  domain = Grico.Domain((0.0, 0.0), (2.0, 1.0), (2, 1))
  grid = Grico.grid(domain)
  first_child = Grico.refine!(grid, 2, 2)
  space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))

  @test Grico.scalar_dof_count(space) == 16
  @test Grico.check_space(space) === nothing

  function right_value(y, coefficients)
    return y < 0 ? _space_value(space, first_child, (-1.0, 2 * y + 1), coefficients) :
           _space_value(space, first_child + 1, (-1.0, 2 * y - 1), coefficients)
  end

  coefficients = collect(1.0:Grico.scalar_dof_count(space))

  for y in (-0.75, -0.25, 0.25, 0.75)
    @test _space_value(space, 1, (1.0, y), coefficients) ≈ right_value(y, coefficients) atol = SPACE_TOL
  end

  for dof in 1:Grico.scalar_dof_count(space)
    unit = zeros(Grico.scalar_dof_count(space))
    unit[dof] = 1.0

    for y in (-0.75, -0.25, 0.25, 0.75)
      @test _space_value(space, 1, (1.0, y), unit) ≈ right_value(y, unit) atol = SPACE_TOL
    end
  end
end

@testset "Three-Dimensional Hanging" begin
  domain = Grico.Domain((0.0, 0.0, 0.0), (2.0, 1.0, 1.0), (2, 1, 1))
  grid = Grico.grid(domain)
  first_child = Grico.refine!(grid, 2, 2)
  lower_front = Grico.refine!(grid, first_child, 3)
  upper_front = Grico.refine!(grid, first_child + 1, 3)
  space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))

  @test Grico.scalar_dof_count(space) == 50
  @test Grico.check_space(space) === nothing

  function right_value(y, z, coefficients)
    leaf = y < 0 ? (z < 0 ? lower_front : lower_front + 1) : (z < 0 ? upper_front : upper_front + 1)
    ξ = (-1.0, y < 0 ? 2 * y + 1 : 2 * y - 1, z < 0 ? 2 * z + 1 : 2 * z - 1)
    return _space_value(space, leaf, ξ, coefficients)
  end

  coefficients = collect(1.0:Grico.scalar_dof_count(space))
  sample_points = ((-0.75, -0.75), (-0.25, -0.5), (-0.25, 0.5), (0.25, -0.5), (0.75, 0.75))

  for (y, z) in sample_points
    @test _space_value(space, 1, (1.0, y, z), coefficients) ≈ right_value(y, z, coefficients) atol = SPACE_TOL
  end

  for dof in 1:Grico.scalar_dof_count(space)
    unit = zeros(Grico.scalar_dof_count(space))
    unit[dof] = 1.0

    for (y, z) in sample_points
      @test _space_value(space, 1, (1.0, y, z), unit) ≈ right_value(y, z, unit) atol = SPACE_TOL
    end
  end
end

@testset "Four-Dimensional Hanging" begin
  domain = Grico.Domain((0.0, 0.0, 0.0, 0.0), (2.0, 1.0, 1.0, 1.0), (2, 1, 1, 1))
  grid = Grico.grid(domain)
  leaves = [2]

  for axis in 2:4
    next_leaves = Int[]

    for leaf in leaves
      first_child = Grico.refine!(grid, leaf, axis)
      append!(next_leaves, first_child:(first_child+1))
    end

    leaves = next_leaves
  end

  space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))
  @test Grico.check_space(space) === nothing

  function right_value(y, z, w, coefficients)
    y_bit = y > 0 ? 1 : 0
    z_bit = z > 0 ? 1 : 0
    w_bit = w > 0 ? 1 : 0
    leaf = leaves[1+w_bit+2*z_bit+4*y_bit]
    ξ = (-1.0, y_bit == 0 ? 2 * y + 1 : 2 * y - 1, z_bit == 0 ? 2 * z + 1 : 2 * z - 1,
         w_bit == 0 ? 2 * w + 1 : 2 * w - 1)
    return _space_value(space, leaf, ξ, coefficients)
  end

  coefficients = collect(1.0:Grico.scalar_dof_count(space))
  sample_points = ((-0.75, -0.75, -0.75), (-0.25, -0.5, 0.5), (0.25, 0.5, -0.5), (0.75, 0.75, 0.75))

  for (y, z, w) in sample_points
    @test _space_value(space, 1, (1.0, y, z, w), coefficients) ≈ right_value(y, z, w, coefficients) atol = SPACE_TOL
  end

  for dof in 1:Grico.scalar_dof_count(space)
    unit = zeros(Grico.scalar_dof_count(space))
    unit[dof] = 1.0

    for (y, z, w) in sample_points
      @test _space_value(space, 1, (1.0, y, z, w), unit) ≈ right_value(y, z, w, unit) atol = SPACE_TOL
    end
  end
end

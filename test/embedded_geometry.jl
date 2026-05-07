using Test
using Grico
import Grico: EmbeddedSurface, SegmentMesh, SurfaceQuadrature, point_count

const EMBEDDED_GEOMETRY_TOL = 1.0e-10

function _argument_message_contains(f, needle::AbstractString)
  try
    f()
  catch exception
    @test exception isa ArgumentError
    @test occursin(needle, sprint(showerror, exception))
    return nothing
  end

  @test false
  return nothing
end

@testset "Physical Domain Cache Sharing" begin
  region = ImplicitRegion(x -> x[1] - 0.5; subdivision_depth=1)
  domain = PhysicalDomain(Domain((0.0,), (1.0,), (2,)), region)

  @test Grico._domain_active_leaves(domain) == [1]
  @test length(region.leaf_classification_cache) == 2

  copied = copy(domain)
  @test copied.region === region
  @test Grico._domain_active_leaves(copied) == [1]
  @test length(region.leaf_classification_cache) == 2

  compacted, compacted_snapshot, _ = Grico.compact(domain, Grico.snapshot(grid(domain)))
  @test compacted.region === region
  @test Grico.check_snapshot(compacted_snapshot) === nothing
  @test Grico._domain_active_leaves(compacted) == [1]

  shifted = PhysicalDomain(Domain((2.0,), (1.0,), (2,)), region)
  @test_throws ArgumentError Grico._domain_active_leaves(shifted)
  @test length(region.leaf_classification_cache) > 2
end

@testset "Finite Cell Robustness" begin
  domain = Domain((0.0,), (1.0,), (1,))

  @test Grico.finite_cell_quadrature(domain, 1, (2,), x -> 1.0) === nothing
  @test Grico.finite_cell_quadrature(domain, 1, (2,), x -> x[1]) === nothing
  @test Grico.finite_cell_quadrature(domain, 1, (2,), x -> -1.0) isa Grico.TensorQuadrature

  sliver = Grico.finite_cell_quadrature(domain, 1, (1,), x -> x[1] - 1.0e-6; subdivision_depth=0)
  @test sliver !== nothing
  @test point_count(sliver) >= 1
  @test sum(weight(sliver, index) for index in 1:point_count(sliver)) > 0.0

  domain3 = Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
  half = Grico.finite_cell_quadrature(domain3, 1, (2, 2, 2), x -> x[1] - 0.5; subdivision_depth=1)
  @test half !== nothing
  @test Grico.dimension(half) == 3
  @test sum(weight(half, index) for index in 1:point_count(half)) ≈ 4.0 atol = EMBEDDED_GEOMETRY_TOL

  _argument_message_contains(() -> ImplicitRegion(x -> -1.0; subdivision_depth=1.5),
                             "subdivision_depth")
  _argument_message_contains(() -> Grico.finite_cell_quadrature(domain, 1, (1,), x -> -1.0;
                                                                subdivision_depth=1.5),
                             "subdivision_depth")
  _argument_message_contains(() -> Grico.ImplicitRegion(x -> -1.0; subdivision_depth=typemax(Int)),
                             "subdivision_depth")
  _argument_message_contains(() -> Grico.finite_cell_quadrature(domain, 1, (1,), x -> x[1] - 0.5;
                                                                subdivision_depth=21),
                             "subdivision_depth")

  calls = Ref(0)
  extension_region = Grico.ImplicitRegion(x -> (calls[] += 1; x[1] - 0.5); subdivision_depth=4)
  extension_domain = Grico.PhysicalDomain(domain, extension_region;
                                          cell_measure=Grico.FiniteCellExtension(1.0))
  @test Grico._assembly_cell_quadrature(extension_domain, 1, (2,)) === nothing
  @test calls[] == 0
  @test isempty(extension_region.leaf_classification_cache)
  @test isempty(extension_region.cut_quadrature_cache)
end

@testset "Embedded Surface Dimension Boundaries" begin
  line = SegmentMesh([(0.0, 0.0), (1.0, 1.0)], [(1, 2)])
  one_dimensional = Domain((0.0,), (1.0,), (1,))
  three_dimensional = Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))

  _argument_message_contains(() -> Grico.surface_quadratures(EmbeddedSurface(line), one_dimensional),
                             "dimension 1")
  _argument_message_contains(() -> Grico.implicit_surface_quadrature(three_dimensional, 1,
                                                                     x -> x[1] - 0.5),
                             "dimensions 1 and 2")

  planar = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  quadratures = Grico.surface_quadratures(EmbeddedSurface(line; point_count=2), planar)
  @test length(quadratures) == 1
  @test only(quadratures) isa SurfaceQuadrature{2}
end

@testset "Embedded Surface Input Validation" begin
  rule = Grico.PointQuadrature([(0.0,)], [1.0])
  @test SurfaceQuadrature(1, rule, [(2.0,)]).normals == [(1.0,)]

  _argument_message_contains(() -> SurfaceQuadrature(1, rule, [(true,)]), "not Bool")
  _argument_message_contains(() -> SurfaceQuadrature(1, Grico.PointQuadrature([(0.0,)], [-1.0]),
                                                     [(1.0,)]), "positive finite")
  _argument_message_contains(() -> SurfaceQuadrature(1, Grico.PointQuadrature([(0.0,)], [0.0]),
                                                     [(1.0,)]), "positive finite")

  _argument_message_contains(() -> SegmentMesh([(false, 0.0), (1.0, 0.0)], [(1, 2)]), "not Bool")
  _argument_message_contains(() -> SegmentMesh([(0.0, 0.0), (1.0, 0.0)], [(true, 2)]), "not Bool")
  _argument_message_contains(() -> SegmentMesh([(0.0, 0.0), (0.0, 0.0)], [(1, 2)]),
                             "positive geometric length")
  _argument_message_contains(() -> SegmentMesh([(0.0, 0.0), (1.0, 0.0)], [(1, 2), (2, 1)]),
                             "duplicate")
  _argument_message_contains(() -> SegmentMesh([(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
                                               [(1, 2), (3, 4)]), "duplicate")
end

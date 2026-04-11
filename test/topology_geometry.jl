using Test
using Grico

const TOPOLOGY_TOL = 1.0e-12

@testset "Cartesian Grid" begin
  for root_counts in ((3,), (2, 2), (2, 1, 2), (2, 1, 1, 2))
    grid = Grico.CartesianGrid(root_counts)
    @test Grico.dimension(grid) == length(root_counts)
    @test Grico.root_cell_counts(grid) == root_counts
    @test Grico.root_cell_total(grid) == prod(root_counts)
    @test Grico.stored_cell_count(grid) == prod(root_counts)
    @test Grico.active_leaf_count(grid) == prod(root_counts)
    @test Grico.active_leaves(grid) == collect(1:prod(root_counts))
    @test Grico.check_topology(grid) === nothing
  end

  grid = Grico.CartesianGrid((2, 1))
  @test Grico.level(grid, 1) == (0, 0)
  @test Grico.level(grid, 2) == (0, 0)
  @test Grico.logical_coordinate(grid, 1) == (0, 0)
  @test Grico.logical_coordinate(grid, 2) == (1, 0)
  @test Grico.neighbor(grid, 1, 1, Grico.UPPER) == 2
  @test Grico.neighbor(grid, 2, 1, Grico.LOWER) == 1
  @test Grico.neighbor(grid, 1, 2, Grico.LOWER) == Grico.NONE
  @test Grico.is_domain_boundary(grid, 1, 2, Grico.LOWER)
  @test Grico.active_leaf(grid, 2) == 2

  first = Grico.refine!(grid, 1, 1)
  @test first == 3
  @test Grico.revision(grid) == 1
  @test Grico.stored_cell_count(grid) == 4
  @test Grico.active_leaves(grid) == [2, 3, 4]
  @test !Grico.is_active_leaf(grid, 1)
  @test Grico.is_expanded(grid, 1)
  @test Grico.parent(grid, 3) == 1
  @test Grico.parent(grid, 4) == 1
  @test Grico.level(grid, 3) == (1, 0)
  @test Grico.level(grid, 4) == (1, 0)
  @test Grico.logical_coordinate(grid, 3) == (0, 0)
  @test Grico.logical_coordinate(grid, 4) == (1, 0)
  @test Grico.neighbor(grid, 3, 1, Grico.UPPER) == 4
  @test Grico.covering_neighbor(grid, 4, 1, Grico.UPPER) == 2
  @test Grico.opposite_active_leaves(grid, 2, 1, Grico.LOWER) == [4]
  @test Grico.check_topology(grid) === nothing

  Grico.derefine!(grid, 1)
  @test Grico.revision(grid) == 2
  @test Grico.active_leaves(grid) == [1, 2]
  @test Grico.is_active_leaf(grid, 1)
  @test !Grico.is_expanded(grid, 1)
  @test Grico.check_topology(grid) === nothing

  second_first = Grico.refine!(grid, 1, 1)
  @test second_first == first
  @test Grico.stored_cell_count(grid) == 4
  Grico.derefine!(grid, 1)
  @test Grico.check_topology(grid) === nothing

  tangential_grid = Grico.CartesianGrid((2, 1))
  Grico.refine!(tangential_grid, 2, 2)
  @test Grico.covering_neighbor(tangential_grid, 1, 1, Grico.UPPER) == 2
  @test Grico.opposite_active_leaves(tangential_grid, 1, 1, Grico.UPPER) == [3, 4]
  @test Grico.check_topology(tangential_grid) === nothing

  periodic = Grico.CartesianGrid((2, 1); periodic=(true, false))
  @test Grico.periodic_axes(periodic) == (true, false)
  @test Grico.is_periodic_axis(periodic, 1)
  @test !Grico.is_periodic_axis(periodic, 2)
  @test Grico.neighbor(periodic, 1, 1, Grico.LOWER) == 2
  @test Grico.neighbor(periodic, 2, 1, Grico.UPPER) == 1
  @test !Grico.is_domain_boundary(periodic, 1, 1, Grico.LOWER)
  @test !Grico.is_domain_boundary(periodic, 2, 1, Grico.UPPER)
  @test Grico.is_domain_boundary(periodic, 1, 2, Grico.LOWER)

  Grico.refine!(periodic, 2, 2)
  @test Grico.covering_neighbor(periodic, 1, 1, Grico.LOWER) == 2
  @test Grico.opposite_active_leaves(periodic, 1, 1, Grico.LOWER) == [3, 4]
  @test Grico.check_topology(periodic) === nothing

  self_periodic = Grico.CartesianGrid((1,); periodic=true)
  @test Grico.neighbor(self_periodic, 1, 1, Grico.LOWER) == 1
  @test Grico.neighbor(self_periodic, 1, 1, Grico.UPPER) == 1
  @test !Grico.is_domain_boundary(self_periodic, 1, 1, Grico.LOWER)
  @test !Grico.is_domain_boundary(self_periodic, 1, 1, Grico.UPPER)

  one_dimensional = Grico.CartesianGrid((2,))
  first = Grico.refine!(one_dimensional, 1, 1)
  Grico.refine!(one_dimensional, first + 1, 1)
  @test Grico.neighbor(one_dimensional, 6, 1, Grico.UPPER) == Grico.NONE
  @test Grico.covering_neighbor(one_dimensional, 6, 1, Grico.UPPER) == 2
  @test Grico.opposite_active_leaves(one_dimensional, 2, 1, Grico.LOWER) == [6]

  two_dimensional = Grico.CartesianGrid((2, 1))
  first = Grico.refine!(two_dimensional, 2, 2)
  Grico.refine!(two_dimensional, first, 2)
  @test Grico.neighbor(two_dimensional, 5, 1, Grico.LOWER) == Grico.NONE
  @test Grico.covering_neighbor(two_dimensional, 5, 1, Grico.LOWER) == 1
  @test Grico.opposite_active_leaves(two_dimensional, 1, 1, Grico.UPPER) == [5, 6, 4]

  three_dimensional = Grico.CartesianGrid((2, 1, 1))
  first = Grico.refine!(three_dimensional, 2, 2)
  lower_front = Grico.refine!(three_dimensional, first, 3)
  Grico.refine!(three_dimensional, first + 1, 3)
  @test Grico.neighbor(three_dimensional, lower_front, 1, Grico.LOWER) == Grico.NONE
  @test Grico.covering_neighbor(three_dimensional, lower_front, 1, Grico.LOWER) == 1
  @test Grico.opposite_active_leaves(three_dimensional, 1, 1, Grico.UPPER) == [5, 6, 7, 8]
end

@testset "Topology Validation" begin
  @test_throws ArgumentError Grico.CartesianGrid((0,))

  grid = Grico.CartesianGrid((1,))
  levels = ntuple(axis -> copy(grid.levels[axis]), 1)
  coords = ntuple(axis -> copy(grid.coords[axis]), 1)
  neighbor_lower = ntuple(axis -> copy(grid.neighbor_lower[axis]), 1)
  neighbor_upper = ntuple(axis -> copy(grid.neighbor_upper[axis]), 1)
  neighbor_upper[1][1] = 1
  @test_throws ArgumentError Grico.CartesianGrid{1}(grid.root_counts, levels, coords,
                                                    copy(grid.parent), copy(grid.first_child),
                                                    copy(grid.split_axis), copy(grid.active),
                                                    copy(grid.active_leaves), neighbor_lower,
                                                    neighbor_upper, grid.revision)

  broken_grid = copy(grid)
  broken_grid.neighbor_upper[1][1] = 1
  geometry = Grico.Geometry((0.0,), (1.0,))
  @test_throws ArgumentError Grico.Domain{1,Float64}(broken_grid, geometry)

  @test_throws BoundsError Grico.neighbor(grid, 1, 2, Grico.LOWER)
  @test_throws ArgumentError Grico.neighbor(grid, 1, 1, 0)
  @test_throws BoundsError Grico.covering_neighbor(grid, 1, 2, Grico.LOWER)
  @test_throws ArgumentError Grico.covering_neighbor(grid, 1, 1, 0)
  @test_throws BoundsError Grico.refine!(grid, 1, 2)
  @test_throws BoundsError Grico.refine!(grid, 1, 0)

  refined = Grico.CartesianGrid((1,))
  first = Grico.refine!(refined, 1, 1)
  @test_throws ArgumentError Grico.refine!(refined, 1, 1)
  @test_throws ArgumentError Grico.derefine!(refined, first)

  nested = Grico.CartesianGrid((1,))
  first = Grico.refine!(nested, 1, 1)
  Grico.refine!(nested, first, 1)
  @test_throws ArgumentError Grico.derefine!(nested, 1)
end

@testset "Geometry And Domain" begin
  grid = Grico.CartesianGrid((3, 2))
  geometry = Grico.Geometry((1.0, -2.0), (6.0, 8.0))

  @test Grico.origin(geometry) == (1.0, -2.0)
  @test Grico.extent(geometry) == (6.0, 8.0)
  @test Grico.cell_lower(geometry, grid, 2) == (3.0, -2.0)
  @test Grico.cell_upper(geometry, grid, 2) == (5.0, 2.0)
  @test Grico.cell_center(geometry, grid, 2) == (4.0, 0.0)
  @test Grico.cell_size(geometry, grid, 2, 1) == 2.0
  @test Grico.cell_size(geometry, grid, 2, 2) == 4.0
  @test Grico.cell_volume(geometry, grid, 2) == 8.0
  @test Grico.face_measure(geometry, grid, 2, 1) == 4.0
  @test Grico.face_measure(geometry, grid, 2, 2) == 2.0

  first = Grico.refine!(grid, 1, 1)
  lower_right_child = first + 1
  @test Grico.cell_lower(geometry, grid, lower_right_child) == (2.0, -2.0)
  @test Grico.cell_upper(geometry, grid, lower_right_child) == (3.0, 2.0)
  @test collect(Grico.map_from_unit_cube(geometry, grid, lower_right_child, (0.25, 0.25))) ≈
        [2.25, -1.0] atol = TOPOLOGY_TOL
  @test collect(Grico.map_from_biunit_cube(geometry, grid, lower_right_child, (0.5, -0.5))) ≈
        [2.75, -1.0] atol = TOPOLOGY_TOL
  @test collect(Grico.map_to_biunit_cube(geometry, grid, lower_right_child, (2.75, -1.0))) ≈
        [0.5, -0.5] atol = TOPOLOGY_TOL
  @test Grico.jacobian_diagonal_from_unit_cube(geometry, grid, lower_right_child, 1) == 1.0
  @test Grico.jacobian_diagonal_from_biunit_cube(geometry, grid, lower_right_child, 1) == 0.5
  @test Grico.jacobian_determinant_from_unit_cube(geometry, grid, lower_right_child) == 4.0
  @test Grico.jacobian_determinant_from_biunit_cube(geometry, grid, lower_right_child) == 1.0

  physical = zeros(2)
  reference = zeros(2)
  Grico.map_from_unit_cube!(physical, geometry, grid, lower_right_child, (0.25, 0.25))
  @test physical ≈ [2.25, -1.0] atol = TOPOLOGY_TOL
  Grico.map_from_biunit_cube!(physical, geometry, grid, lower_right_child, (0.5, -0.25))
  Grico.map_to_biunit_cube!(reference, geometry, grid, lower_right_child, Tuple(physical))
  @test reference ≈ [0.5, -0.25] atol = TOPOLOGY_TOL

  domain = Grico.Domain((-1.0, 2.0), (6.0, 8.0), (3, 2))
  @test Grico.dimension(domain) == 2
  @test Grico.origin(domain) == (-1.0, 2.0)
  @test Grico.extent(domain) == (6.0, 8.0)
  @test Grico.root_cell_counts(Grico.grid(domain)) == (3, 2)
  @test Grico.cell_lower(domain, 5) == (1.0, 6.0)
  @test Grico.cell_upper(domain, 5) == (3.0, 10.0)
  @test Grico.cell_center(domain, 5) == (2.0, 8.0)

  copied = copy(domain)
  Grico.refine!(Grico.grid(copied), 1, 1)
  @test Grico.stored_cell_count(Grico.grid(domain)) == 6
  @test Grico.stored_cell_count(Grico.grid(copied)) == 8
  @test Grico.check_topology(Grico.grid(domain)) === nothing
  @test Grico.check_topology(Grico.grid(copied)) === nothing

  periodic_domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (2, 1); periodic=(true, false))
  @test Grico.periodic_axes(periodic_domain) == (true, false)
  @test Grico.is_periodic_axis(periodic_domain, 1)
  periodic_copy = copy(periodic_domain)
  @test Grico.periodic_axes(periodic_copy) == (true, false)
end

@testset "Geometry Validation" begin
  @test_throws ArgumentError Grico.Geometry((0.0,), (0.0,))
  @test_throws ArgumentError Grico.Geometry{1,Float64}((0.0,), (0.0,))
  @test_throws ArgumentError Grico.Geometry{1,Float64}((NaN,), (1.0,))
end

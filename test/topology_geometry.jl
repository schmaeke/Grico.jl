using Test
using Grico

const TOPOLOGY_TOL = 1.0e-12

function _topology_access_allocation(grid)
  value = 0

  for index in 1:Grico.active_leaf_count(grid)
    leaf = Grico.active_leaf(grid, index)
    value += Grico.level(grid, leaf, 1)
    value += Grico.logical_coordinate(grid, leaf, 1)
  end

  return value
end

function _geometry_access_allocation(physical, geometry, grid, cell, reference)
  value = Grico.cell_size(geometry, grid, cell, 1)
  value += Grico.jacobian_diagonal_from_biunit_cube(geometry, grid, cell, 1)
  value += Grico.jacobian_determinant_from_biunit_cube(geometry, grid, cell)
  Grico.map_from_biunit_cube!(physical, geometry, grid, cell, reference)
  return value + physical[1]
end

function _snapshot_boundary_specs(snapshot)
  specs = Set{Tuple{Int,Int,Int}}()

  for index in 1:Grico.boundary_face_count(snapshot)
    spec = Grico.boundary_face_spec(snapshot, index)
    push!(specs, (spec.leaf, spec.axis, spec.side))
  end

  return specs
end

function _snapshot_interface_specs(snapshot)
  specs = Set{Tuple{Int,Int,Int}}()

  for index in 1:Grico.interface_count(snapshot)
    spec = Grico.interface_spec(snapshot, index)
    push!(specs, (spec.minus, spec.axis, spec.plus))
  end

  return specs
end

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

  allocation_grid = Grico.CartesianGrid((2, 2))
  _topology_access_allocation(allocation_grid)
  @test @allocated(_topology_access_allocation(allocation_grid)) == 0

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

  nested_reuse = Grico.CartesianGrid((1,))
  child_first = Grico.refine!(nested_reuse, 1, 1)
  grandchild_first = Grico.refine!(nested_reuse, child_first, 1)
  @test Grico.stored_cell_count(nested_reuse) == 5
  Grico.derefine!(nested_reuse, child_first)
  Grico.derefine!(nested_reuse, 1)
  child_first_again = Grico.refine!(nested_reuse, 1, 1)
  @test child_first_again == child_first
  @test Grico.first_child(nested_reuse, child_first_again) == grandchild_first
  grandchild_first_again = Grico.refine!(nested_reuse, child_first_again, 1)
  @test grandchild_first_again == grandchild_first
  @test Grico.stored_cell_count(nested_reuse) == 5
  @test Grico.check_topology(nested_reuse) === nothing

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

@testset "Grid Snapshots" begin
  grid = Grico.CartesianGrid((2, 1))
  grid_snapshot = Grico.snapshot(grid)
  @test grid_snapshot isa Grico.GridSnapshot{2}
  @test Grico.grid(grid_snapshot) === grid
  @test Grico.dimension(grid_snapshot) == 2
  @test grid_snapshot.generation == Grico.revision(grid)
  @test Grico.active_leaf_count(grid_snapshot) == 2
  @test Grico.active_leaves(grid_snapshot) == [1, 2]
  @test Grico.active_leaf(grid_snapshot, 2) == 2
  @test Grico.is_active_leaf(grid_snapshot, 1)
  @test !Grico.is_expanded(grid_snapshot, 1)
  @test Grico.check_snapshot(grid_snapshot) === nothing

  expected_boundary = Set([(1, 1, Grico.LOWER), (1, 2, Grico.LOWER), (1, 2, Grico.UPPER),
                           (2, 1, Grico.UPPER), (2, 2, Grico.LOWER), (2, 2, Grico.UPPER)])
  @test Grico.boundary_face_count(grid_snapshot) == length(expected_boundary)
  @test _snapshot_boundary_specs(grid_snapshot) == expected_boundary
  @test Grico.interface_count(grid_snapshot) == 1
  @test _snapshot_interface_specs(grid_snapshot) == Set([(1, 1, 2)])

  periodic = Grico.CartesianGrid((2, 1); periodic=(true, false))
  periodic_snapshot = Grico.snapshot(periodic)
  @test Grico.check_snapshot(periodic_snapshot) === nothing
  @test _snapshot_boundary_specs(periodic_snapshot) ==
        Set([(1, 2, Grico.LOWER), (1, 2, Grico.UPPER), (2, 2, Grico.LOWER), (2, 2, Grico.UPPER)])
  @test _snapshot_interface_specs(periodic_snapshot) == Set([(1, 1, 2), (2, 1, 1)])

  self_periodic = Grico.CartesianGrid((1,); periodic=true)
  self_periodic_snapshot = Grico.snapshot(self_periodic)
  @test Grico.boundary_face_count(self_periodic_snapshot) == 0
  @test _snapshot_interface_specs(self_periodic_snapshot) == Set([(1, 1, 1)])

  morton_grid = Grico.CartesianGrid((2,))
  Grico.refine!(morton_grid, 2, 1)
  Grico.refine!(morton_grid, 1, 1)
  morton_snapshot = Grico.snapshot(morton_grid)
  @test Grico.active_leaves(morton_grid) == [3, 4, 5, 6]
  @test Grico.active_leaves(morton_snapshot) == [5, 6, 3, 4]
  @test [Grico.logical_coordinate(morton_grid, leaf, 1)
         for leaf in Grico.active_leaves(morton_snapshot)] == [0, 1, 2, 3]
  @test Grico.check_snapshot(morton_snapshot) === nothing

  refined = Grico.CartesianGrid((1,))
  stale_snapshot = Grico.snapshot(refined)
  first = Grico.refine!(refined, 1, 1)
  @test first == 2
  @test_throws ArgumentError Grico.check_snapshot(stale_snapshot)
  @test_throws ArgumentError Grico.active_leaf_count(stale_snapshot)
  @test_throws ArgumentError Grico.boundary_face_count(stale_snapshot)

  refined_snapshot = Grico.snapshot(refined)
  @test Grico.active_leaves(refined_snapshot) == [2, 3]
  @test Grico.is_expanded(refined_snapshot, 1)
  @test !Grico.is_active_leaf(refined_snapshot, 1)
  @test Grico.check_snapshot(refined_snapshot) === nothing

  append_only = Grico.CartesianGrid((1, 1))
  source_snapshot = Grico.snapshot(append_only)
  source_generation = Grico.revision(append_only)
  target_snapshot, children = Grico._refine_snapshot_leaf!(source_snapshot, 1, (true, false))
  @test children == [2, 3]
  @test Grico.revision(append_only) == source_generation
  @test Grico.stored_cell_count(append_only) == 3
  @test Grico.active_leaves(source_snapshot) == [1]
  @test Grico.check_snapshot(source_snapshot) === nothing
  @test Grico.active_leaves(target_snapshot) == [2, 3]
  @test Grico.is_expanded(target_snapshot, 1)

  coarsened_snapshot, coarsened_children = Grico._derefine_snapshot_cell(target_snapshot, 1, 1)
  @test coarsened_children == (2, 3)
  @test Grico.revision(append_only) == source_generation
  @test Grico.active_leaves(coarsened_snapshot) == [1]
  @test Grico.check_snapshot(target_snapshot) === nothing
  @test Grico.check_snapshot(coarsened_snapshot) === nothing

  tuple_refined_snapshot, tuple_children = Grico._refine_snapshot_leaf!(coarsened_snapshot, 1,
                                                                        (true, true))
  @test length(tuple_children) == 4
  @test Grico.active_leaves(tuple_refined_snapshot) == [4, 6, 5, 7]
  @test Set(Grico.active_leaves(tuple_refined_snapshot)) == Set(tuple_children)
  @test all(leaf -> Grico.level(append_only, leaf) == (1, 1), tuple_children)
  @test Grico.check_snapshot(coarsened_snapshot) === nothing
  @test Grico.check_snapshot(tuple_refined_snapshot) === nothing

  axis_reuse_grid = Grico.CartesianGrid((1, 1))
  axis_source = Grico.snapshot(axis_reuse_grid)
  axis_x_refined, _ = Grico._refine_snapshot_leaf!(axis_source, 1, (true, false))
  axis_coarsened, _ = Grico._derefine_snapshot_cell(axis_x_refined, 1, 1)
  axis_y_refined, axis_y_children = Grico._refine_snapshot_leaf!(axis_coarsened, 1, (false, true))
  @test length(axis_y_children) == 2
  @test all(leaf -> Grico.level(axis_reuse_grid, leaf) == (0, 1), axis_y_children)
  @test Grico.check_snapshot(axis_source) === nothing
  @test Grico.check_snapshot(axis_x_refined) === nothing
  @test Grico.check_snapshot(axis_coarsened) === nothing
  @test Grico.check_snapshot(axis_y_refined) === nothing

  append_domain = Grico.Domain(append_only, Grico.Geometry((0.0, 0.0), (1.0, 1.0)))
  compact_source_domain, compact_source_snapshot, source_old_to_new = Grico.compact(append_domain,
                                                                                    source_snapshot)
  compact_source_grid = Grico.grid(compact_source_domain)
  @test Grico.stored_cell_count(compact_source_grid) == 1
  @test Grico.active_leaves(compact_source_snapshot) == [1]
  @test source_old_to_new[1] == 1
  @test all(iszero, source_old_to_new[2:end])
  @test Grico.check_snapshot(source_snapshot) === nothing
  @test Grico.check_snapshot(compact_source_snapshot) === nothing

  compact_target_domain, compact_target_snapshot, target_old_to_new = Grico.compact(append_domain,
                                                                                    target_snapshot)
  compact_target_grid = Grico.grid(compact_target_domain)
  @test Grico.stored_cell_count(compact_target_grid) == 3
  @test Grico.active_leaves(compact_target_snapshot) == [2, 3]
  @test target_old_to_new[1:3] == [1, 2, 3]
  @test all(iszero, target_old_to_new[4:end])
  @test Grico.check_topology(compact_target_grid) === nothing
  @test Grico.check_snapshot(target_snapshot) === nothing
  @test Grico.check_snapshot(compact_target_snapshot) === nothing

  destructive = Grico.CartesianGrid((1,))
  destructive_source = Grico.snapshot(destructive)
  destructive_refined, _ = Grico._refine_snapshot_leaf!(destructive_source, 1, (true,))
  destructive_coarsened, _ = Grico._derefine_snapshot_cell(destructive_refined, 1, 1)
  @test Grico.stored_cell_count(destructive) == 3
  compacted_snapshot, destructive_old_to_new = Grico.compact!(destructive, destructive_coarsened)
  @test Grico.stored_cell_count(destructive) == 1
  @test Grico.active_leaves(destructive) == [1]
  @test Grico.active_leaves(compacted_snapshot) == [1]
  @test destructive_old_to_new == [1, 0, 0]
  @test Grico.check_snapshot(compacted_snapshot) === nothing
  @test_throws ArgumentError Grico.check_snapshot(destructive_source)
  @test_throws ArgumentError Grico.check_snapshot(destructive_refined)
  @test_throws ArgumentError Grico.check_snapshot(destructive_coarsened)
  @test_throws ArgumentError Grico.compact!(Grico.CartesianGrid((1,)), compacted_snapshot)
end

@testset "Topology Validation" begin
  @test_throws ArgumentError Grico.CartesianGrid((0,))
  @test_throws ArgumentError Grico.CartesianGrid((typemax(Int), 2))
  @test Grico.root_cell_counts(Grico.CartesianGrid((1, big(2)))) == (1, 2)

  grid = Grico.CartesianGrid((1,))
  @test_throws ArgumentError Grico.neighbor(grid, 1, 2, Grico.LOWER)
  @test_throws ArgumentError Grico.neighbor(grid, 1, 1, 0)
  @test_throws ArgumentError Grico.neighbor(grid, 1, 1, big(typemax(Int)) + 1)
  @test_throws ArgumentError Grico.covering_neighbor(grid, 1, 2, Grico.LOWER)
  @test_throws ArgumentError Grico.covering_neighbor(grid, 1, 1, 0)
  @test_throws ArgumentError Grico.refine!(grid, 1, 2)
  @test_throws ArgumentError Grico.refine!(grid, 1, 0)

  refined = Grico.CartesianGrid((1,))
  first = Grico.refine!(refined, 1, 1)
  @test_throws ArgumentError Grico.refine!(refined, 1, 1)
  @test_throws ArgumentError Grico.derefine!(refined, first)

  nested = Grico.CartesianGrid((1,))
  first = Grico.refine!(nested, 1, 1)
  Grico.refine!(nested, first, 1)
  @test_throws ArgumentError Grico.derefine!(nested, 1)

  frontier = Grico.CartesianGrid((1,))
  first = Grico.refine!(frontier, 1, 1)
  Grico.derefine!(frontier, 1)
  frontier_active = copy(frontier.active)
  frontier_active[first] = true
  @test_throws ArgumentError Grico.CartesianGrid{1}(frontier.root_counts,
                                                    ntuple(axis -> copy(frontier.levels[axis]), 1),
                                                    ntuple(axis -> copy(frontier.coords[axis]), 1),
                                                    copy(frontier.parent),
                                                    copy(frontier.first_child),
                                                    copy(frontier.split_axis), frontier_active,
                                                    [1, first], frontier.periodic,
                                                    frontier.revision)

  childless = Grico.CartesianGrid((1,))
  childless_active = copy(childless.active)
  childless_active[1] = false
  childless_first_child = copy(childless.first_child)
  childless_first_child[1] = 2
  childless_split_axis = copy(childless.split_axis)
  childless_split_axis[1] = 1
  @test_throws ArgumentError Grico.CartesianGrid{1}(childless.root_counts,
                                                    ntuple(axis -> copy(childless.levels[axis]), 1),
                                                    ntuple(axis -> copy(childless.coords[axis]), 1),
                                                    copy(childless.parent), childless_first_child,
                                                    childless_split_axis, childless_active, Int[],
                                                    childless.periodic, childless.revision)

  duplicate = Grico.CartesianGrid((2,))
  duplicate_coords = (copy(duplicate.coords[1]),)
  duplicate_coords[1][2] = 0
  @test_throws ArgumentError Grico.CartesianGrid{1}(duplicate.root_counts,
                                                    ntuple(axis -> copy(duplicate.levels[axis]), 1),
                                                    duplicate_coords, copy(duplicate.parent),
                                                    copy(duplicate.first_child),
                                                    copy(duplicate.split_axis),
                                                    copy(duplicate.active),
                                                    copy(duplicate.active_leaves),
                                                    duplicate.periodic, duplicate.revision)

  retained = Grico.CartesianGrid((1,))
  retained_first_child = copy(retained.first_child)
  retained_first_child[1] = 999
  @test_throws ArgumentError Grico.CartesianGrid{1}(retained.root_counts,
                                                    ntuple(axis -> copy(retained.levels[axis]), 1),
                                                    ntuple(axis -> copy(retained.coords[axis]), 1),
                                                    copy(retained.parent), retained_first_child,
                                                    copy(retained.split_axis),
                                                    copy(retained.active),
                                                    copy(retained.active_leaves), retained.periodic,
                                                    retained.revision)

  retained_parent = Grico.CartesianGrid((1,))
  retained_first = Grico.refine!(retained_parent, 1, 1)
  Grico.derefine!(retained_parent, 1)
  retained_parent_array = copy(retained_parent.parent)
  retained_parent_array[retained_first] = retained_first + 1
  @test_throws ArgumentError Grico.CartesianGrid{1}(retained_parent.root_counts,
                                                    ntuple(axis -> copy(retained_parent.levels[axis]),
                                                           1),
                                                    ntuple(axis -> copy(retained_parent.coords[axis]),
                                                           1), retained_parent_array,
                                                    copy(retained_parent.first_child),
                                                    copy(retained_parent.split_axis),
                                                    copy(retained_parent.active),
                                                    copy(retained_parent.active_leaves),
                                                    retained_parent.periodic,
                                                    retained_parent.revision)

  corrupted = Grico.CartesianGrid((1,))
  corrupted.first_child[1] = 999
  previous_revision = Grico.revision(corrupted)
  previous_stored = Grico.stored_cell_count(corrupted)
  @test_throws ArgumentError Grico.refine!(corrupted, 1, 1)
  @test Grico.is_active_leaf(corrupted, 1)
  @test Grico.split_axis(corrupted, 1) == 0
  @test Grico.revision(corrupted) == previous_revision
  @test Grico.stored_cell_count(corrupted) == previous_stored

  revision_limited = Grico.CartesianGrid((1,))
  revision_limited.revision = typemax(UInt)
  @test_throws ArgumentError Grico.refine!(revision_limited, 1, 1)
  @test Grico.is_active_leaf(revision_limited, 1)
  @test Grico.split_axis(revision_limited, 1) == 0
  @test Grico.stored_cell_count(revision_limited) == 1
  @test Grico.revision(revision_limited) == typemax(UInt)

  expanded_revision_limited = Grico.CartesianGrid((1,))
  first = Grico.refine!(expanded_revision_limited, 1, 1)
  expanded_revision_limited.revision = typemax(UInt)
  @test_throws ArgumentError Grico.derefine!(expanded_revision_limited, 1)
  @test Grico.is_expanded(expanded_revision_limited, 1)
  @test Grico.is_active_leaf(expanded_revision_limited, first)
  @test Grico.is_active_leaf(expanded_revision_limited, first + 1)
  @test Grico.revision(expanded_revision_limited) == typemax(UInt)

  deep = Grico.CartesianGrid((1,))
  leaf = 1

  for _ in 1:62
    first = Grico.refine!(deep, leaf, 1)
    leaf = first + 1
  end

  previous_revision = Grico.revision(deep)
  previous_stored = Grico.stored_cell_count(deep)
  @test_throws ArgumentError Grico.refine!(deep, leaf, 1)
  @test Grico.revision(deep) == previous_revision
  @test Grico.stored_cell_count(deep) == previous_stored
  @test Grico.check_topology(deep) === nothing
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

  oversized = fill(-99.0, 3)
  @test Grico.map_from_biunit_cube!(oversized, geometry, grid, lower_right_child, (0.5, -0.5)) ===
        oversized
  @test oversized[1:2] ≈ [2.75, -1.0] atol = TOPOLOGY_TOL
  @test oversized[3] == -99.0

  _geometry_access_allocation(physical, geometry, grid, lower_right_child, (0.25, -0.25))
  @test @allocated(_geometry_access_allocation(physical, geometry, grid, lower_right_child,
                                               (0.25, -0.25))) == 0

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

  mixed_geometry = Grico.Geometry((0, 1.0), (1, 2.0))
  @test mixed_geometry isa Grico.Geometry{2,Float64}
  mixed_domain = Grico.Domain((0, 1.0), (1, 2.0), (1, big(2)))
  @test Grico.origin(mixed_domain) == (0.0, 1.0)
  @test Grico.root_cell_counts(Grico.grid(mixed_domain)) == (1, 2)
end

@testset "Geometry Validation" begin
  @test_throws ArgumentError Grico.Geometry((0.0,), (0.0,))
  @test_throws ArgumentError Grico.Geometry{1,Float64}((0.0,), (0.0,))
  @test_throws ArgumentError Grico.Geometry{1,Float64}((NaN,), (1.0,))

  one_dimensional_geometry = Grico.Geometry((0.0,), (1.0,))
  two_dimensional_geometry = Grico.Geometry((0.0, 0.0), (1.0, 1.0))
  one_dimensional_grid = Grico.CartesianGrid((1,))
  two_dimensional_grid = Grico.CartesianGrid((1, 1))

  @test_throws ArgumentError Grico.cell_size(two_dimensional_geometry, one_dimensional_grid, 1, 1)
  @test_throws ArgumentError Grico.cell_lower(two_dimensional_geometry, one_dimensional_grid, 1)
  @test_throws ArgumentError Grico.cell_lower(one_dimensional_geometry, two_dimensional_grid, 1, 1)
  @test_throws ArgumentError Grico.cell_upper(two_dimensional_geometry, one_dimensional_grid, 1)
  @test_throws ArgumentError Grico.cell_center(two_dimensional_geometry, one_dimensional_grid, 1)
  @test_throws ArgumentError Grico.cell_volume(two_dimensional_geometry, one_dimensional_grid, 1)
  @test_throws ArgumentError Grico.face_measure(two_dimensional_geometry, one_dimensional_grid, 1,
                                                1)
  @test_throws ArgumentError Grico.map_from_biunit_cube(two_dimensional_geometry,
                                                        one_dimensional_grid, 1, (0.0, 0.0))
  @test_throws ArgumentError Grico.map_to_biunit_cube(two_dimensional_geometry,
                                                      one_dimensional_grid, 1, (0.5, 0.5))
  @test_throws ArgumentError Grico.jacobian_diagonal_from_biunit_cube(two_dimensional_geometry,
                                                                      one_dimensional_grid, 1, 1)
  @test_throws ArgumentError Grico.jacobian_determinant_from_biunit_cube(two_dimensional_geometry,
                                                                         one_dimensional_grid, 1)
  @test_throws ArgumentError Grico.map_from_biunit_cube!(zeros(2), two_dimensional_geometry,
                                                         one_dimensional_grid, 1, (0.0, 0.0))
end

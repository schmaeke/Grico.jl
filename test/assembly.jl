using LinearAlgebra
using SparseArrays
using Test
using Grico

const ASSEMBLY_TOL = 1.0e-10

struct OutOfReferenceEmbeddedGeometry end

function Grico._space_surface_quadratures(surface::Grico.EmbeddedSurface{OutOfReferenceEmbeddedGeometry},
                                          space::Grico.HpSpace{1,T}) where {T<:AbstractFloat}
  quadrature = Grico.PointQuadrature([(T(2),)], [one(T)])
  return [Grico.SurfaceQuadrature(1, quadrature, [(one(T),)])]
end

struct Diffusion{F,T}
  field::F
  coefficient::T
end

function Grico.cell_matrix!(local_matrix, operator::Diffusion, values::Grico.CellValues)
  block = Grico.block(local_matrix, values, operator.field, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    weight_value = Grico.weight(values, point_index)

    for row_mode in 1:mode_count
      grad_row = Grico.shape_gradient(values, operator.field, point_index, row_mode)

      for col_mode in 1:mode_count
        grad_col = Grico.shape_gradient(values, operator.field, point_index, col_mode)
        block[row_mode, col_mode] += operator.coefficient *
                                     sum(grad_row[axis] * grad_col[axis]
                                         for axis in eachindex(grad_row)) *
                                     weight_value
      end
    end
  end

  return nothing
end

struct Source{F,G}
  field::F
  f::G
end

function Grico.cell_rhs!(local_rhs, operator::Source, values::Grico.CellValues)
  block = Grico.block(local_rhs, values, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    source_value = operator.f(Grico.point(values, point_index))
    weighted = source_value * Grico.weight(values, point_index)

    for mode_index in 1:mode_count
      block[mode_index] += Grico.shape_value(values, operator.field, point_index, mode_index) *
                           weighted
    end
  end

  return nothing
end

struct BoundaryLoad{F,T}
  field::F
  value::T
end

function Grico.face_rhs!(local_rhs, operator::BoundaryLoad, values::Grico.FaceValues)
  block = Grico.block(local_rhs, values, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    weighted = operator.value * Grico.weight(values, point_index)

    for mode_index in 1:mode_count
      block[mode_index] += Grico.shape_value(values, operator.field, point_index, mode_index) *
                           weighted
    end
  end

  return nothing
end

struct EmbeddedPointLoad{F,T}
  field::F
  value::T
end

function Grico.surface_rhs!(local_rhs, operator::EmbeddedPointLoad, values::Grico.SurfaceValues)
  block = Grico.block(local_rhs, values, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    weighted = operator.value *
               Grico.weight(values, point_index) *
               Grico.normal(values, point_index)[1]

    for mode_index in 1:mode_count
      block[mode_index] += Grico.shape_value(values, operator.field, point_index, mode_index) *
                           weighted
    end
  end

  return nothing
end

struct MassCoupling{F,G,T}
  test_field::F
  trial_field::G
  coefficient::T
end

function Grico.cell_matrix!(local_matrix, operator::MassCoupling, values::Grico.CellValues)
  block = Grico.block(local_matrix, values, operator.test_field, operator.trial_field)
  test_modes = Grico.local_mode_count(values, operator.test_field)
  trial_modes = Grico.local_mode_count(values, operator.trial_field)

  for point_index in 1:Grico.point_count(values)
    weighted = operator.coefficient * Grico.weight(values, point_index)

    for row_mode in 1:test_modes
      shape_row = Grico.shape_value(values, operator.test_field, point_index, row_mode)

      for col_mode in 1:trial_modes
        shape_col = Grico.shape_value(values, operator.trial_field, point_index, col_mode)
        block[row_mode, col_mode] += shape_row * shape_col * weighted
      end
    end
  end

  return nothing
end

struct InterfaceStamp{F}
  field::F
end

function Grico.interface_matrix!(local_matrix, operator::InterfaceStamp,
                                 values::Grico.InterfaceValues)
  minus_values = Grico.minus(values)
  plus_values = Grico.plus(values)
  minus_minus = Grico.block(local_matrix, minus_values, operator.field, minus_values,
                            operator.field)
  minus_plus = Grico.block(local_matrix, minus_values, operator.field, plus_values, operator.field)
  plus_minus = Grico.block(local_matrix, plus_values, operator.field, minus_values, operator.field)
  plus_plus = Grico.block(local_matrix, plus_values, operator.field, plus_values, operator.field)
  minus_minus .= [1.0 2.0;
                  3.0 4.0]
  minus_plus .= [5.0 6.0;
                 7.0 8.0]
  plus_minus .= [9.0 10.0;
                 11.0 12.0]
  plus_plus .= [13.0 14.0;
                15.0 16.0]
  return nothing
end

function Grico.interface_rhs!(local_rhs, operator::InterfaceStamp, values::Grico.InterfaceValues)
  minus_block = Grico.block(local_rhs, Grico.minus(values), operator.field)
  plus_block = Grico.block(local_rhs, Grico.plus(values), operator.field)
  minus_block .= [1.0, 2.0]
  plus_block .= [3.0, 4.0]
  return nothing
end

struct GradientJumpPenalty{F,T}
  field::F
  coefficient::T
end

function Grico.interface_matrix!(local_matrix, operator::GradientJumpPenalty,
                                 values::Grico.InterfaceValues)
  minus_values = Grico.minus(values)
  plus_values = Grico.plus(values)
  minus_minus = Grico.block(local_matrix, minus_values, operator.field, minus_values,
                            operator.field)
  minus_plus = Grico.block(local_matrix, minus_values, operator.field, plus_values, operator.field)
  plus_minus = Grico.block(local_matrix, plus_values, operator.field, minus_values, operator.field)
  plus_plus = Grico.block(local_matrix, plus_values, operator.field, plus_values, operator.field)
  minus_modes = Grico.local_mode_count(minus_values, operator.field)
  plus_modes = Grico.local_mode_count(plus_values, operator.field)

  for point_index in 1:Grico.point_count(values)
    weighted = operator.coefficient * Grico.weight(values, point_index)

    for row_mode in 1:minus_modes
      row_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                 row_mode)

      for col_mode in 1:minus_modes
        col_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                   col_mode)
        minus_minus[row_mode, col_mode] += row_gradient * col_gradient * weighted
      end

      for col_mode in 1:plus_modes
        col_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                   col_mode)
        minus_plus[row_mode, col_mode] -= row_gradient * col_gradient * weighted
      end
    end

    for row_mode in 1:plus_modes
      row_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index, row_mode)

      for col_mode in 1:minus_modes
        col_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                   col_mode)
        plus_minus[row_mode, col_mode] -= row_gradient * col_gradient * weighted
      end

      for col_mode in 1:plus_modes
        col_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                   col_mode)
        plus_plus[row_mode, col_mode] += row_gradient * col_gradient * weighted
      end
    end
  end

  return nothing
end

struct NonlinearGradientJump{F}
  field::F
end

function Grico.interface_residual!(local_rhs, operator::NonlinearGradientJump,
                                   values::Grico.InterfaceValues, state::Grico.State)
  minus_values = Grico.minus(values)
  plus_values = Grico.plus(values)
  minus_block = Grico.block(local_rhs, minus_values, operator.field)
  plus_block = Grico.block(local_rhs, plus_values, operator.field)
  minus_modes = Grico.local_mode_count(minus_values, operator.field)
  plus_modes = Grico.local_mode_count(plus_values, operator.field)

  for point_index in 1:Grico.point_count(values)
    jump = Grico.jump(Grico.normal_gradient(minus_values, state, operator.field, point_index),
                      Grico.normal_gradient(plus_values, state, operator.field, point_index))
    weighted = jump^3 * Grico.weight(values, point_index)

    for mode_index in 1:minus_modes
      minus_block[mode_index] -= Grico.shape_normal_gradient(minus_values, operator.field,
                                                             point_index, mode_index) * weighted
    end

    for mode_index in 1:plus_modes
      plus_block[mode_index] += Grico.shape_normal_gradient(plus_values, operator.field,
                                                            point_index, mode_index) * weighted
    end
  end

  return nothing
end

function Grico.interface_tangent!(local_matrix, operator::NonlinearGradientJump,
                                  values::Grico.InterfaceValues, state::Grico.State)
  minus_values = Grico.minus(values)
  plus_values = Grico.plus(values)
  minus_minus = Grico.block(local_matrix, minus_values, operator.field, minus_values,
                            operator.field)
  minus_plus = Grico.block(local_matrix, minus_values, operator.field, plus_values, operator.field)
  plus_minus = Grico.block(local_matrix, plus_values, operator.field, minus_values, operator.field)
  plus_plus = Grico.block(local_matrix, plus_values, operator.field, plus_values, operator.field)
  minus_modes = Grico.local_mode_count(minus_values, operator.field)
  plus_modes = Grico.local_mode_count(plus_values, operator.field)

  for point_index in 1:Grico.point_count(values)
    jump = Grico.jump(Grico.normal_gradient(minus_values, state, operator.field, point_index),
                      Grico.normal_gradient(plus_values, state, operator.field, point_index))
    weighted = 3.0 * jump^2 * Grico.weight(values, point_index)

    for row_mode in 1:minus_modes
      row_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                 row_mode)

      for col_mode in 1:minus_modes
        col_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                   col_mode)
        minus_minus[row_mode, col_mode] += row_gradient * col_gradient * weighted
      end

      for col_mode in 1:plus_modes
        col_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                   col_mode)
        minus_plus[row_mode, col_mode] -= row_gradient * col_gradient * weighted
      end
    end

    for row_mode in 1:plus_modes
      row_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index, row_mode)

      for col_mode in 1:minus_modes
        col_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                   col_mode)
        plus_minus[row_mode, col_mode] -= row_gradient * col_gradient * weighted
      end

      for col_mode in 1:plus_modes
        col_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                   col_mode)
        plus_plus[row_mode, col_mode] += row_gradient * col_gradient * weighted
      end
    end
  end

  return nothing
end

struct NonlinearReaction{F,T}
  field::F
  source::T
end

function Grico.cell_residual!(local_rhs, operator::NonlinearReaction, values::Grico.CellValues,
                              state::Grico.State)
  block = Grico.block(local_rhs, values, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    u = Grico.value(values, state, operator.field, point_index)
    weighted = (u^2 - operator.source) * Grico.weight(values, point_index)

    for mode_index in 1:mode_count
      block[mode_index] += Grico.shape_value(values, operator.field, point_index, mode_index) *
                           weighted
    end
  end

  return nothing
end

function Grico.cell_tangent!(local_matrix, operator::NonlinearReaction, values::Grico.CellValues,
                             state::Grico.State)
  block = Grico.block(local_matrix, values, operator.field, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    u = Grico.value(values, state, operator.field, point_index)
    weighted = 2.0 * u * Grico.weight(values, point_index)

    for row_mode in 1:mode_count
      shape_row = Grico.shape_value(values, operator.field, point_index, row_mode)

      for col_mode in 1:mode_count
        shape_col = Grico.shape_value(values, operator.field, point_index, col_mode)
        block[row_mode, col_mode] += shape_row * shape_col * weighted
      end
    end
  end

  return nothing
end

struct NonlinearBoundaryReaction{F,T}
  field::F
  target::T
end

function Grico.face_residual!(local_rhs, operator::NonlinearBoundaryReaction,
                              values::Grico.FaceValues, state::Grico.State)
  block = Grico.block(local_rhs, values, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    u = Grico.value(values, state, operator.field, point_index)
    weighted = (u^2 - operator.target) * Grico.weight(values, point_index)

    for mode_index in 1:mode_count
      block[mode_index] += Grico.shape_value(values, operator.field, point_index, mode_index) *
                           weighted
    end
  end

  return nothing
end

function Grico.face_tangent!(local_matrix, operator::NonlinearBoundaryReaction,
                             values::Grico.FaceValues, state::Grico.State)
  block = Grico.block(local_matrix, values, operator.field, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  for point_index in 1:Grico.point_count(values)
    u = Grico.value(values, state, operator.field, point_index)
    weighted = 2.0 * u * Grico.weight(values, point_index)

    for row_mode in 1:mode_count
      shape_row = Grico.shape_value(values, operator.field, point_index, row_mode)

      for col_mode in 1:mode_count
        shape_col = Grico.shape_value(values, operator.field, point_index, col_mode)
        block[row_mode, col_mode] += shape_row * shape_col * weighted
      end
    end
  end

  return nothing
end

function _field_value(field, state, leaf, ξ, component::Int=1)
  space = Grico.field_space(field)
  one_dimensional = ntuple(axis -> Grico._fe_basis_values(ξ[axis],
                                                          Grico.cell_degrees(space, leaf)[axis]),
                           length(ξ))
  values = Grico.field_component_values(state, field, component)
  result = 0.0

  for mode in Grico.local_modes(space, leaf)
    local_value = 1.0

    for axis in 1:length(ξ)
      local_value *= one_dimensional[axis][mode[axis]+1]
    end

    for term in Grico.mode_terms(space, leaf, mode)
      result += local_value * term.second * values[term.first]
    end
  end

  return result
end

function _dirichlet_state(plan::Grico.AssemblyPlan)
  coefficients = zeros(Float64, Grico.dof_count(plan))

  for index in eachindex(plan.dirichlet.fixed_dofs)
    coefficients[plan.dirichlet.fixed_dofs[index]] = plan.dirichlet.fixed_values[index]
  end

  for row in plan.dirichlet.rows
    value = row.rhs

    for index in eachindex(row.indices)
      value += row.coefficients[index] * coefficients[row.indices[index]]
    end

    coefficients[row.pivot] = value
  end

  return Grico.State(plan, coefficients)
end

function _embedded_surface_measure(surface::Grico.SurfaceQuadrature, domain)
  total = 0.0

  for point_index in 1:Grico.point_count(surface.quadrature)
    normal_data = surface.normals[point_index]
    det_jacobian = Grico.jacobian_determinant_from_biunit_cube(domain, surface.leaf)
    mapped_normal = ntuple(axis -> Grico.jacobian_diagonal_from_biunit_cube(domain, surface.leaf,
                                                                            axis) *
                                   normal_data[axis], length(normal_data))
    scale = det_jacobian / sqrt(sum(mapped_normal[axis]^2 for axis in eachindex(mapped_normal)))
    total += Grico.weight(surface.quadrature, point_index) * scale
  end

  return total
end

@testset "Affine Cell Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_cell!(problem, Source(u, x -> 2.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 1.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 1.0))

  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  state = Grico.State(plan, Grico.solve(system))
  exact(x) = x * (1.0 - x) + 1.0
  @test size(Grico.matrix(system), 1) == 1

  for leaf in Grico.active_leaves(space), ξ in (-1.0, -0.25, 0.25, 1.0)
    x = Grico.map_from_biunit_cube(domain, leaf, (ξ,))[1]
    @test _field_value(u, state, leaf, (ξ,)) ≈ exact(x) atol = ASSEMBLY_TOL
  end
end

@testset "Boundary Face Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_boundary!(problem, Grico.BoundaryFace(1, Grico.UPPER), BoundaryLoad(u, 1.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 0.0))

  plan = Grico.compile(problem)
  face = first(plan.integration.boundary_faces)
  @test_throws ArgumentError Grico.normal(face, 0)
  system = Grico.assemble(plan)
  state = Grico.State(plan, Grico.solve(system))

  for leaf in Grico.active_leaves(space), ξ in (-1.0, 0.0, 1.0)
    x = Grico.map_from_biunit_cube(domain, leaf, (ξ,))[1]
    @test _field_value(u, state, leaf, (ξ,)) ≈ x atol = ASSEMBLY_TOL
  end
end

@testset "Interface Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_interface!(problem, InterfaceStamp(u))

  plan = Grico.compile(problem)
  @test length(plan.integration.interfaces) == 1
  interface = plan.integration.interfaces[1]
  @test interface.minus_leaf == 1
  @test interface.plus_leaf == 2
  @test Grico.face_axis(interface) == 1
  @test Grico.normal(interface) == (1.0,)
  @test_throws ArgumentError Grico.normal(interface, 0)
  @test_throws ArgumentError Grico.normal(Grico.minus(interface), 0)
  system = Grico.assemble(plan)
  assembled = Matrix(Grico.matrix(system))
  assembled_rhs = Grico.rhs(system)

  local_matrix = [1.0 2.0 5.0 6.0;
                  3.0 4.0 7.0 8.0;
                  9.0 10.0 13.0 14.0;
                  11.0 12.0 15.0 16.0]
  local_rhs = [1.0, 2.0, 3.0, 4.0]
  local_to_global = [1, 2, 2, 3]
  expected_matrix = zeros(3, 3)
  expected_rhs = zeros(3)

  for local_row in eachindex(local_to_global)
    expected_rhs[local_to_global[local_row]] += local_rhs[local_row]

    for local_col in eachindex(local_to_global)
      expected_matrix[local_to_global[local_row], local_to_global[local_col]] += local_matrix[local_row,
                                                                                              local_col]
    end
  end

  @test assembled ≈ expected_matrix atol = ASSEMBLY_TOL
  @test assembled_rhs ≈ expected_rhs atol = ASSEMBLY_TOL
end

@testset "Filtered Interface Assembly" begin
  background = Grico.Domain((0.0,), (3.0,), (3,))
  domain = Grico.PhysicalDomain(background,
                                Grico.ImplicitRegion(x -> x[1] - 1.5; subdivision_depth=1))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1), continuity=:dg))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_interface!(problem, InterfaceStamp(u))

  plan = Grico.compile(problem)
  @test length(plan.integration.interfaces) == 1
  interface = only(plan.integration.interfaces)
  @test interface.minus_leaf == 1
  @test interface.plus_leaf == 2
end

@testset "DG Interface Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1), continuity=:dg))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_interface!(problem, InterfaceStamp(u))

  plan = Grico.compile(problem)
  @test length(plan.integration.interfaces) == 1
  interface = only(plan.integration.interfaces)
  @test interface.minus_leaf == 1
  @test interface.plus_leaf == 2
  @test Grico.face_axis(interface) == 1
  @test Grico.normal(interface) == (1.0,)

  system = Grico.assemble(plan)
  assembled = Matrix(Grico.matrix(system))
  assembled_rhs = Grico.rhs(system)
  expected_matrix = [1.0 2.0 5.0 6.0;
                     3.0 4.0 7.0 8.0;
                     9.0 10.0 13.0 14.0;
                     11.0 12.0 15.0 16.0]
  expected_rhs = [1.0, 2.0, 3.0, 4.0]

  @test assembled ≈ expected_matrix atol = ASSEMBLY_TOL
  @test assembled_rhs ≈ expected_rhs atol = ASSEMBLY_TOL
end

@testset "DG P0 Cell Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(0), continuity=:dg))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, MassCoupling(u, u, 1.0))
  Grico.add_cell!(problem, Source(u, _ -> 1.0))

  system = Grico.assemble(Grico.compile(problem))
  @test Matrix(Grico.matrix(system)) ≈ [0.5 0.0;
                                        0.0 0.5] atol = ASSEMBLY_TOL
  @test Grico.rhs(system) ≈ [0.5, 0.5] atol = ASSEMBLY_TOL
end

@testset "DG P0 Minimum Quadrature Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(0),
                                           quadrature=Grico.DegreePlusQuadrature(0),
                                           continuity=:dg))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, MassCoupling(u, u, 1.0))
  Grico.add_cell!(problem, Source(u, _ -> 1.0))

  system = Grico.assemble(Grico.compile(problem))
  @test Matrix(Grico.matrix(system)) ≈ [0.5 0.0;
                                        0.0 0.5] atol = ASSEMBLY_TOL
  @test Grico.rhs(system) ≈ [0.5, 0.5] atol = ASSEMBLY_TOL
end

@testset "DG Interface Helpers" begin
  @test Grico.jump(1.0, 4.0) == 3.0
  @test Grico.average(1.0, 4.0) == 2.5
  @test Grico.jump((1.0, 2.0), (4.0, 8.0)) == (3.0, 6.0)
  @test Grico.average((1.0, 2.0), (5.0, 8.0)) == (3.0, 5.0)
  @test Grico.average(((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0))) ==
        ((3.0, 4.0), (5.0, 6.0))
  @test Grico.normal_component((2.0, -1.0), (3.0, 4.0)) == 2.0
  @test Grico.normal_component(((2.0, -1.0), (1.0, 0.5)), (3.0, 4.0)) == (2.0, 5.0)

  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1), continuity=:dg))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_interface!(problem, InterfaceStamp(u))
  plan = Grico.compile(problem)
  interface = only(plan.integration.interfaces)
  minus_values = Grico.minus(interface)
  plus_values = Grico.plus(interface)
  state = Grico.State(plan, [1.0, 2.0, 3.0, 4.0])

  minus_value = Grico.value(minus_values, state, u, 1)
  plus_value = Grico.value(plus_values, state, u, 1)
  @test Grico.jump(minus_value, plus_value) ≈ plus_value - minus_value atol = ASSEMBLY_TOL
  @test Grico.average(minus_value, plus_value) ≈ (minus_value + plus_value) / 2 atol = ASSEMBLY_TOL
  @test Grico.shape_normal_gradient(minus_values, u, 1, 1) ≈
        Grico.normal_component(Grico.shape_gradient(minus_values, u, 1, 1),
                               Grico.normal(minus_values, 1)) atol = ASSEMBLY_TOL
  @test Grico.normal_gradient(minus_values, state, u, 1) ≈
        Grico.normal_component(Grico.gradient(minus_values, state, u, 1),
                               Grico.normal(minus_values, 1)) atol = ASSEMBLY_TOL
end

@testset "Periodic Interface Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,); periodic=true)
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_interface!(problem, InterfaceStamp(u))

  plan = Grico.compile(problem)
  @test isempty(plan.integration.boundary_faces)
  @test length(plan.integration.interfaces) == 2

  first_interface = plan.integration.interfaces[1]
  second_interface = plan.integration.interfaces[2]
  @test first_interface.minus_leaf == 1
  @test first_interface.plus_leaf == 2
  @test second_interface.minus_leaf == 2
  @test second_interface.plus_leaf == 1
  @test Grico.normal(first_interface) == (1.0,)
  @test Grico.normal(second_interface) == (1.0,)
  @test Grico.point(first_interface, 1) == (0.5,)
  @test Grico.point(Grico.plus(second_interface), 1) == (0.0,)

  system = Grico.assemble(plan)
  assembled = Matrix(Grico.matrix(system))
  assembled_rhs = Grico.rhs(system)

  local_matrix = [1.0 2.0 5.0 6.0;
                  3.0 4.0 7.0 8.0;
                  9.0 10.0 13.0 14.0;
                  11.0 12.0 15.0 16.0]
  local_rhs = [1.0, 2.0, 3.0, 4.0]
  expected_matrix = zeros(2, 2)
  expected_rhs = zeros(2)

  for local_to_global in ([1, 2, 2, 1], [2, 1, 1, 2])
    for local_row in eachindex(local_to_global)
      expected_rhs[local_to_global[local_row]] += local_rhs[local_row]

      for local_col in eachindex(local_to_global)
        expected_matrix[local_to_global[local_row], local_to_global[local_col]] += local_matrix[local_row,
                                                                                                local_col]
      end
    end
  end

  @test assembled ≈ expected_matrix atol = ASSEMBLY_TOL
  @test assembled_rhs ≈ expected_rhs atol = ASSEMBLY_TOL
end

@testset "Periodic Boundary Validation" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,); periodic=true)
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)

  @test_throws ArgumentError Grico.add_boundary!(problem, Grico.BoundaryFace(1, Grico.LOWER),
                                                 BoundaryLoad(u, 1.0))
  @test_throws ArgumentError Grico.add_constraint!(problem,
                                                   Grico.Dirichlet(u,
                                                                   Grico.BoundaryFace(1,
                                                                                      Grico.UPPER),
                                                                   0.0))
end

@testset "Interface Terms Disable Static Condensation" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)

  baseline_problem = Grico.AffineProblem(u)
  Grico.add_cell!(baseline_problem, Diffusion(u, 1.0))
  Grico.add_constraint!(baseline_problem,
                        Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 0.0))
  Grico.add_constraint!(baseline_problem,
                        Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 0.0))
  baseline_system = Grico.assemble(Grico.compile(baseline_problem))
  @test size(Grico.matrix(baseline_system), 1) == 1

  interface_problem = Grico.AffineProblem(u)
  Grico.add_cell!(interface_problem, Diffusion(u, 1.0))
  Grico.add_interface!(interface_problem, GradientJumpPenalty(u, 1.0))
  Grico.add_constraint!(interface_problem,
                        Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 0.0))
  Grico.add_constraint!(interface_problem,
                        Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 0.0))
  interface_system = Grico.assemble(Grico.compile(interface_problem))
  @test size(Grico.matrix(interface_system), 1) == 3
  @test Matrix(Grico.matrix(interface_system)) ≈ transpose(Matrix(Grico.matrix(interface_system))) atol = ASSEMBLY_TOL
end

@testset "Dirichlet On Refined Boundary" begin
  domain = Grico.Domain((0.0, 0.0, 0.0), (2.0, 1.0, 1.0), (2, 1, 1))
  grid = Grico.grid(domain)
  first_child = Grico.refine!(grid, 2, 2)
  lower_front = Grico.refine!(grid, first_child, 3)
  upper_front = Grico.refine!(grid, first_child + 1, 3)
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 2.0))

  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  state = Grico.State(plan, Grico.solve(system))
  assembled = Matrix(Grico.matrix(system))

  @test assembled ≈ transpose(assembled) atol = ASSEMBLY_TOL

  for leaf in (lower_front, lower_front + 1, upper_front, upper_front + 1)
    for yz in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))
      @test _field_value(u, state, leaf, (1.0, yz[1], yz[2])) ≈ 2.0 atol = ASSEMBLY_TOL
    end
  end
end

@testset "Dirichlet Projection Rank Tolerance" begin
  matrix = Diagonal([300.0, 4.0e-6])
  rhs = [0.0, 6.0e-6]
  constraint_matrix, targets, tolerance = Grico._dirichlet_constraint_system(Matrix(matrix), rhs,
                                                                             :u)
  coefficients = constraint_matrix \ targets

  @test tolerance < matrix[2, 2]
  @test matrix * coefficients ≈ rhs atol = ASSEMBLY_TOL
  @test_throws ArgumentError Grico._dirichlet_constraint_system([1.0 0.0; 0.0 0.0], [0.0, 1.0e-6],
                                                                :u)
end

@testset "Dirichlet On Selected Refined Boundary Subset" begin
  domain = Grico.Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (3, 3, 3))
  grid = Grico.grid(domain)

  for (leaf, axis) in ((9, 1), (21, 2), (6, 1), (3, 1), (16, 2), (26, 1), (12, 3), (38, 3))
    Grico.refine!(grid, leaf, axis)
  end

  space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  boundary_data(x) = sum(x)
  problem = Grico.AffineProblem(u)

  for axis in 1:3
    Grico.add_constraint!(problem,
                          Grico.Dirichlet(u, Grico.BoundaryFace(axis, Grico.UPPER), boundary_data))
  end

  plan = Grico.compile(problem)
  state = _dirichlet_state(plan)
  @test !isempty(plan.dirichlet.fixed_dofs) || !isempty(plan.dirichlet.rows)

  for face in plan.integration.boundary_faces
    Grico.face_side(face) == Grico.UPPER || continue

    for point_index in 1:Grico.point_count(face)
      expected = boundary_data(Grico.point(face, point_index))
      @test Grico.value(face, state, u, point_index) ≈ expected atol = 1.0e-9
    end
  end
end

@testset "Dirichlet On Selected Refined Boundary Subset Solve" begin
  domain = Grico.Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (3, 3, 3))
  grid = Grico.grid(domain)

  for (leaf, axis) in ((9, 1), (21, 2), (6, 1), (3, 1), (16, 2), (26, 1), (12, 3), (38, 3))
    Grico.refine!(grid, leaf, axis)
  end

  space = Grico.HpSpace(domain, Grico.SpaceOptions(degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  exact(x) = sum(value^2 for value in x)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_cell!(problem, Source(u, _ -> -6.0))

  for axis in 1:3
    Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(axis, Grico.UPPER), exact))
  end

  plan = Grico.compile(problem)
  state = Grico.State(plan, Grico.solve(Grico.assemble(plan)))
  @test Grico.relative_l2_error(state, u, exact; extra_points=2) <= 1.0e-9
end

@testset "Dirichlet On Full Boundary" begin
  domain = Grico.Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(3)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))

  for axis in 1:3, side in (Grico.LOWER, Grico.UPPER)
    Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(axis, side), 2.0))
  end

  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  state = Grico.State(plan, Grico.solve(system))

  for ξ1 in (-1.0, 0.0, 1.0), ξ2 in (-1.0, 0.0, 1.0), ξ3 in (-1.0, 0.0, 1.0)
    @test _field_value(u, state, 1, (ξ1, ξ2, ξ3)) ≈ 2.0 atol = ASSEMBLY_TOL
  end
end

@testset "Vector Dirichlet Component Selection" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.VectorField(space, 3; name=:u)

  shorthand_problem = Grico.AffineProblem(u)
  Grico.add_constraint!(shorthand_problem,
                        Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), (1.0, 2.0, 3.0)))
  shorthand_state = _dirichlet_state(Grico.compile(shorthand_problem))

  explicit_problem = Grico.AffineProblem(u)
  Grico.add_constraint!(explicit_problem,
                        Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), (1, 2, 3),
                                        (1.0, 2.0, 3.0)))
  explicit_state = _dirichlet_state(Grico.compile(explicit_problem))

  for component in 1:3
    @test _field_value(u, shorthand_state, 1, (-1.0,), component) ≈
          _field_value(u, explicit_state, 1, (-1.0,), component) atol = ASSEMBLY_TOL
  end

  problem = Grico.AffineProblem(u)
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 1, -2.0))
  Grico.add_constraint!(problem,
                        Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 2,
                                        (10.0, 1.5, 30.0)))
  Grico.add_constraint!(problem,
                        Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), (1, 3),
                                        x -> (x[1], -x[1])))
  state = _dirichlet_state(Grico.compile(problem))

  @test _field_value(u, state, 1, (-1.0,), 1) ≈ -2.0 atol = ASSEMBLY_TOL
  @test _field_value(u, state, 1, (-1.0,), 2) ≈ 1.5 atol = ASSEMBLY_TOL
  @test _field_value(u, state, 1, (-1.0,), 3) ≈ 0.0 atol = ASSEMBLY_TOL
  @test _field_value(u, state, 1, (1.0,), 1) ≈ 1.0 atol = ASSEMBLY_TOL
  @test _field_value(u, state, 1, (1.0,), 2) ≈ 0.0 atol = ASSEMBLY_TOL
  @test _field_value(u, state, 1, (1.0,), 3) ≈ -1.0 atol = ASSEMBLY_TOL
end

@testset "Multi-Field Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  p = Grico.ScalarField(space; name=:p)
  base_problem = Grico.AffineProblem(u)
  Grico.add_cell!(base_problem, MassCoupling(u, u, 1.0))
  base_matrix = Matrix(Grico.matrix(Grico.assemble(Grico.compile(base_problem))))

  problem = Grico.AffineProblem(u, p)
  Grico.add_cell!(problem, MassCoupling(u, u, 1.0))
  Grico.add_cell!(problem, MassCoupling(p, p, 2.0))
  Grico.add_cell!(problem, MassCoupling(u, p, 3.0))
  Grico.add_cell!(problem, MassCoupling(p, u, 4.0))

  system = Grico.assemble(Grico.compile(problem))
  assembled = Matrix(Grico.matrix(system))
  layout = Grico.field_layout(system)
  u_range = Grico.field_dof_range(layout, u)
  p_range = Grico.field_dof_range(layout, p)

  @test assembled[u_range, u_range] ≈ base_matrix atol = ASSEMBLY_TOL
  @test assembled[p_range, p_range] ≈ 2.0 .* base_matrix atol = ASSEMBLY_TOL
  @test assembled[u_range, p_range] ≈ 3.0 .* base_matrix atol = ASSEMBLY_TOL
  @test assembled[p_range, u_range] ≈ 4.0 .* base_matrix atol = ASSEMBLY_TOL
end

@testset "Mixed Vector Field Evaluation" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  velocity = Grico.VectorField(space, 2; name=:velocity)
  pressure = Grico.ScalarField(space; name=:pressure)
  vector_plan = Grico.compile(Grico.AffineProblem(velocity))
  vector_state = Grico.State(vector_plan, collect(1.0:Grico.dof_count(vector_plan)))
  vector_cell = only(vector_plan.integration.cells)
  vector_face = first(vector_plan.integration.boundary_faces)

  @test @inferred(Grico.value(vector_cell, vector_state, velocity, 1)) isa NTuple{2,Float64}
  @test @inferred(Grico.gradient(vector_cell, vector_state, velocity, 1)) isa
        NTuple{2,NTuple{1,Float64}}
  @test @inferred(Grico.normal_gradient(vector_face, vector_state, velocity, 1)) isa
        NTuple{2,Float64}

  plan = Grico.compile(Grico.AffineProblem(velocity, pressure))
  state = Grico.State(plan, collect(1.0:Grico.dof_count(plan)))
  cell = only(plan.integration.cells)

  function expected_gradient(field, state, ξ, component)
    derivatives = Grico.integrated_legendre_derivatives(ξ[1], Grico.cell_degrees(space, 1)[1])
    values = Grico.field_component_values(state, field, component)
    result = 0.0

    for mode in Grico.local_modes(space, 1)
      local_derivative = 2.0 * derivatives[mode[1]+1]

      for term in Grico.mode_terms(space, 1, mode)
        result += local_derivative * term.second * values[term.first]
      end
    end

    return result
  end

  for point_index in 1:Grico.point_count(cell)
    ξ = (2.0 * Grico.point(cell, point_index)[1] - 1.0,)
    velocity_value = Grico.value(cell, state, velocity, point_index)
    velocity_gradient = Grico.gradient(cell, state, velocity, point_index)

    @test velocity_value[1] ≈ _field_value(velocity, state, 1, ξ, 1) atol = ASSEMBLY_TOL
    @test velocity_value[2] ≈ _field_value(velocity, state, 1, ξ, 2) atol = ASSEMBLY_TOL
    @test Grico.value(cell, state, pressure, point_index) ≈ _field_value(pressure, state, 1, ξ, 1) atol = ASSEMBLY_TOL
    @test velocity_gradient[1][1] ≈ expected_gradient(velocity, state, ξ, 1) atol = ASSEMBLY_TOL
    @test velocity_gradient[2][1] ≈ expected_gradient(velocity, state, ξ, 2) atol = ASSEMBLY_TOL
    @test Grico.gradient(cell, state, pressure, point_index)[1] ≈
          expected_gradient(pressure, state, ξ, 1) atol = ASSEMBLY_TOL
  end

  face = first(plan.integration.boundary_faces)
  for point_index in 1:Grico.point_count(face)
    actual = Grico.normal_gradient(face, state, velocity, point_index)
    expected = Grico.normal_component(Grico.gradient(face, state, velocity, point_index),
                                      Grico.normal(face, point_index))
    @test length(actual) == length(expected)
    @test all(isapprox(actual[index], expected[index]; atol=ASSEMBLY_TOL)
              for index in eachindex(actual))
  end
end

@testset "Mean Value Constraint" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_constraint!(problem, Grico.MeanValue(u, 2.0))

  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  state = Grico.State(plan, Grico.solve(system))

  for leaf in Grico.active_leaves(space), ξ in (-1.0, 0.0, 1.0)
    @test _field_value(u, state, leaf, (ξ,)) ≈ 2.0 atol = ASSEMBLY_TOL
  end
end

@testset "Mean Value Constraint On Physical Domain" begin
  background = Grico.Domain((0.0,), (1.0,), (1,))
  domain = Grico.PhysicalDomain(background,
                                Grico.ImplicitRegion(x -> x[1] - 0.5; subdivision_depth=1))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_constraint!(problem, Grico.MeanValue(u, 2.0))

  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  state = Grico.State(plan, Grico.solve(system))
  cell = only(plan.integration.cells)
  measure = sum(Grico.weight(cell, point_index) for point_index in 1:Grico.point_count(cell))
  average = sum(Grico.value(cell, state, u, point_index) * Grico.weight(cell, point_index)
                for point_index in 1:Grico.point_count(cell)) / measure

  @test measure ≈ 0.5 atol = ASSEMBLY_TOL
  @test average ≈ 2.0 atol = ASSEMBLY_TOL
end

@testset "Custom Linear Solve Hook" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_cell!(problem, Source(u, x -> 2.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 1.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 1.0))

  system = Grico.assemble(Grico.compile(problem))
  calls = Ref(0)
  state = Grico.State(system, Grico.solve(system; linear_solve=(A, b) -> begin
                                            calls[] += 1
                                            return A \ b
                                          end))

  @test calls[] == 1
  @test_throws ArgumentError Grico.AdditiveSchwarzPreconditioner(min_dofs=-1)
  @test_throws ArgumentError Grico.SmoothedAggregationAMGPreconditioner(min_dofs=-1)
  @test_throws ArgumentError Grico.ILUPreconditioner(tau=-1.0)
  @test_throws ArgumentError Grico.FieldSplitSchurPreconditioner((), (u,))
  @test_throws ArgumentError Grico.FieldSplitSchurPreconditioner((u,), (u,))
  @test_throws ArgumentError Grico.solve(system; linear_solve=(A, b) -> A \ b,
                                         preconditioner=Grico.AdditiveSchwarzPreconditioner())
  @test_throws ArgumentError Grico.solve(system; linear_solve=(A, b) -> 1.0)
  @test_throws ArgumentError Grico.solve(system; linear_solve=(A, b) -> zeros(size(A, 1) + 1))
  @test_throws ArgumentError Grico.solve(system; linear_solve=(A, b) -> fill("bad", size(A, 1)))
  @test_throws ArgumentError Grico.solve(system;
                                         preconditioner=Grico.FieldSplitSchurPreconditioner((u,),
                                                                                            (Grico.ScalarField(space;
                                                                                                               name=:other),)))

  for leaf in Grico.active_leaves(space), ξ in (-1.0, -0.25, 0.25, 1.0)
    x = Grico.map_from_biunit_cube(domain, leaf, (ξ,))[1]
    @test _field_value(u, state, leaf, (ξ,)) ≈ x * (1.0 - x) + 1.0 atol = ASSEMBLY_TOL
  end
end

@testset "Initial Solution Hook" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_cell!(problem, Source(u, x -> 2.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 1.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 1.0))

  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  full_guess = collect(1.0:Grico.dof_count(plan))
  reduced_guess = full_guess[system.solve_dofs]
  expected = reduced_guess[system.ordering]
  seen = Ref{Vector{Float64}}()

  state = Grico.State(system,
                      Grico.solve(system; initial_solution=full_guess,
                                  linear_solve=(A, b, x0) -> begin
                                    seen[] = copy(x0)
                                    return A \ b
                                  end))
  @test seen[] ≈ expected atol = ASSEMBLY_TOL

  state_guess = Grico.State(plan, full_guess)
  seen[] = Float64[]
  state_from_state = Grico.State(system,
                                 Grico.solve(system; initial_solution=state_guess,
                                             linear_solve=(A, b, x0) -> begin
                                               seen[] = copy(x0)
                                               return A \ b
                                             end))
  @test seen[] ≈ expected atol = ASSEMBLY_TOL
  @test Grico.coefficients(state_from_state) ≈ Grico.coefficients(state) atol = ASSEMBLY_TOL
  @test_throws ArgumentError Grico.solve(system; initial_solution=1.0)
  @test_throws ArgumentError Grico.solve(system; initial_solution=fill("bad", length(full_guess)))

  compact_domain = Grico.Domain((0.0,), (1.0,), (1,))
  compact_space = Grico.HpSpace(compact_domain,
                                Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                   degree=Grico.UniformDegree(2)))
  w = Grico.ScalarField(compact_space; name=:w)
  compact_problem = Grico.AffineProblem(w)
  Grico.add_cell!(compact_problem, Diffusion(w, 1.0))
  Grico.add_constraint!(compact_problem,
                        Grico.Dirichlet(w, Grico.BoundaryFace(1, Grico.LOWER), 0.0))
  Grico.add_constraint!(compact_problem,
                        Grico.Dirichlet(w, Grico.BoundaryFace(1, Grico.UPPER), 0.0))
  compact_plan = Grico.compile(compact_problem)
  compact_system = Grico.assemble(compact_plan)
  incompatible_plan = Grico.AdaptivityPlan(compact_space)
  Grico.request_p_derefinement!(incompatible_plan, 1, 1)
  Grico.request_h_refinement!(incompatible_plan, 1, 1)
  incompatible_transition = Grico.transition(incompatible_plan)
  incompatible_w = Grico.adapted_field(incompatible_transition, w)
  incompatible_state = Grico.State(Grico.FieldLayout((incompatible_w,)),
                                   zeros(Grico.scalar_dof_count(Grico.field_space(incompatible_w))))

  @test Grico.dof_count(Grico.field_layout(incompatible_state)) == Grico.dof_count(compact_plan)
  @test_throws ArgumentError Grico.solve(compact_system; initial_solution=incompatible_state)
end

@testset "Default Symmetric AMG Solve" begin
  domain = Grico.Domain((0.0,), (1.0,), (2048,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_cell!(problem, Source(u, x -> 2.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 1.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 1.0))

  system = Grico.assemble(Grico.compile(problem))
  @test isempty(system.preconditioner_cache)
  state = Grico.State(system, Grico.solve(system))
  @test haskey(system.preconditioner_cache, Grico.SmoothedAggregationAMGPreconditioner())
  guided_state = Grico.State(system,
                             Grico.solve(system;
                                         initial_solution=fill(0.5, size(Grico.matrix(system), 1))))
  @test norm(Grico.coefficients(guided_state) - Grico.coefficients(state)) /
        norm(Grico.coefficients(state)) <= 1.0e-6

  for leaf in (1, 512, 1024, 1536, 2048), ξ in (-1.0, 0.0, 1.0)
    x = Grico.map_from_biunit_cube(domain, leaf, (ξ,))[1]
    @test _field_value(u, state, leaf, (ξ,)) ≈ x * (1.0 - x) + 1.0 atol = 5.0e-7
  end
end

@testset "Three-Dimensional Additive Schwarz Coarse Compression" begin
  domain = Grico.Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (6, 6, 6))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_cell!(problem, Source(u, x -> 1.0))

  for axis in 1:3
    Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(axis, Grico.LOWER), 0.0))
    Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(axis, Grico.UPPER), 0.0))
  end

  system = Grico.assemble(Grico.compile(problem))
  preconditioner = Grico._preconditioner_operator(system, Grico.AdditiveSchwarzPreconditioner())
  @test preconditioner ===
        Grico._preconditioner_operator(system, Grico.AdditiveSchwarzPreconditioner())
  @test rank(Matrix(preconditioner.coarse_prolongation)) ==
        size(preconditioner.coarse_prolongation, 2)

  reduced_values, converged = Grico._preconditioned_krylov_solve(system,
                                                                 Grico.AdditiveSchwarzPreconditioner())
  @test converged
  @test norm(Grico.matrix(system) * reduced_values - Grico.rhs(system)) / norm(Grico.rhs(system)) <=
        1.0e-6
  repeated_values, repeated_converged = Grico._preconditioned_krylov_solve(system,
                                                                           Grico.AdditiveSchwarzPreconditioner(),
                                                                           reduced_values)
  @test repeated_converged
  @test norm(repeated_values - reduced_values) / norm(reduced_values) <= 1.0e-8

  direct_state = Grico.State(system, Grico.solve(system; linear_solve=(A, b) -> A \ b))
  explicit_state = Grico.State(system,
                               Grico.solve(system;
                                           preconditioner=Grico.AdditiveSchwarzPreconditioner()))
  schwarz_state = Grico.State(system, reduced_values)
  @test norm(Grico.coefficients(explicit_state) - Grico.coefficients(direct_state)) /
        norm(Grico.coefficients(direct_state)) <= 1.0e-6
  @test norm(Grico.coefficients(schwarz_state) - Grico.coefficients(direct_state)) /
        norm(Grico.coefficients(direct_state)) <= 1.0e-6
end

@testset "Default Unsymmetric ILU Solve" begin
  domain = Grico.Domain((0.0,), (1.0,), (768,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  v = Grico.ScalarField(space; name=:v)
  problem = Grico.AffineProblem(u, v)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_cell!(problem, Diffusion(v, 1.0))
  Grico.add_cell!(problem, MassCoupling(u, v, 0.25))
  Grico.add_cell!(problem, Source(u, x -> 1.0 + x[1]))
  Grico.add_cell!(problem, Source(v, x -> 2.0 - x[1]))

  for field in (u, v)
    Grico.add_constraint!(problem, Grico.Dirichlet(field, Grico.BoundaryFace(1, Grico.LOWER), 0.0))
    Grico.add_constraint!(problem, Grico.Dirichlet(field, Grico.BoundaryFace(1, Grico.UPPER), 0.0))
  end

  system = Grico.assemble(Grico.compile(problem))
  @test !system.symmetric
  @test isempty(system.preconditioner_cache)

  explicit_ilu = Grico.ILUPreconditioner(min_dofs=0)
  reduced_values, converged = Grico._preconditioned_krylov_solve(system, explicit_ilu)
  @test converged
  @test norm(Grico.matrix(system) * reduced_values - Grico.rhs(system)) / norm(Grico.rhs(system)) <=
        1.0e-6

  default_state = Grico.State(system, Grico.solve(system))
  @test haskey(system.preconditioner_cache, Grico._DIRECT_SOLVE_CACHE_KEY)
  @test !haskey(system.preconditioner_cache, Grico.ILUPreconditioner())
  direct_operator = Grico._direct_operator(system)
  @test direct_operator === Grico._direct_operator(system)
  direct_state = Grico.State(system, Grico.solve(system; linear_solve=(A, b) -> A \ b))
  guided_state = Grico.State(system,
                             Grico.solve(system;
                                         initial_solution=fill(0.5, size(Grico.matrix(system), 1))))
  ilu_state = Grico.State(system, reduced_values)
  @test norm(Grico.coefficients(default_state) - Grico.coefficients(direct_state)) /
        norm(Grico.coefficients(direct_state)) <= 1.0e-6
  @test norm(Grico.coefficients(guided_state) - Grico.coefficients(direct_state)) /
        norm(Grico.coefficients(direct_state)) <= 1.0e-6
  @test norm(Grico.coefficients(ilu_state) - Grico.coefficients(direct_state)) /
        norm(Grico.coefficients(direct_state)) <= 1.0e-6
end

@testset "Field-Split Schur Solve" begin
  domain = Grico.Domain((0.0,), (1.0,), (768,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  p = Grico.ScalarField(space; name=:p)
  problem = Grico.AffineProblem(u, p)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_cell!(problem, Diffusion(p, 0.1))
  Grico.add_cell!(problem, MassCoupling(u, p, 1.0))
  Grico.add_cell!(problem, MassCoupling(p, u, 1.0))
  Grico.add_cell!(problem, Source(u, x -> 1.0 + x[1]))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 0.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 0.0))
  Grico.add_constraint!(problem, Grico.MeanValue(p, 0.0))

  system = Grico.assemble(Grico.compile(problem))
  preconditioner = Grico.FieldSplitSchurPreconditioner((u,), (p,))
  @test isempty(system.preconditioner_cache)

  reduced_values, converged = Grico._preconditioned_krylov_solve(system, preconditioner)
  @test converged
  @test norm(Grico.matrix(system) * reduced_values - Grico.rhs(system)) / norm(Grico.rhs(system)) <=
        1.0e-6

  default_state = Grico.State(system, Grico.solve(system; preconditioner=preconditioner))
  @test haskey(system.preconditioner_cache, preconditioner)
  direct_state = Grico.State(system, Grico.solve(system; linear_solve=(A, b) -> A \ b))
  guided_state = Grico.State(system,
                             Grico.solve(system; preconditioner=preconditioner,
                                         initial_solution=fill(0.5, size(Grico.matrix(system), 1))))
  schur_state = Grico.State(system, reduced_values)
  @test norm(Grico.coefficients(default_state) - Grico.coefficients(direct_state)) /
        norm(Grico.coefficients(direct_state)) <= 1.0e-6
  @test norm(Grico.coefficients(guided_state) - Grico.coefficients(direct_state)) /
        norm(Grico.coefficients(direct_state)) <= 1.0e-6
  @test norm(Grico.coefficients(schur_state) - Grico.coefficients(direct_state)) /
        norm(Grico.coefficients(direct_state)) <= 1.0e-6
end

@testset "Nonlinear Cell Residual And Tangent" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.ResidualProblem(u)
  Grico.add_cell!(problem, NonlinearReaction(u, 4.0))
  Grico.add_constraint!(problem, Grico.MeanValue(u, 2.0))

  plan = Grico.compile(problem)
  @test plan.assembly_structure.affine === nothing
  @test plan.assembly_structure.tangent === nothing
  @test_throws ArgumentError Grico.assemble(plan)
  state = Grico.State(plan)
  fill!(Grico.coefficients(state), 1.5)
  residual_workspace = Grico.ResidualWorkspace(plan)
  workspace_residual = zeros(Float64, Grico.dof_count(plan))
  Grico.residual!(workspace_residual, plan, state, residual_workspace)
  @test workspace_residual ≈ Grico.residual(plan, state)

  other_plan = Grico.compile(problem)
  other_workspace = Grico.ResidualWorkspace(other_plan)
  @test_throws ArgumentError Grico.residual!(workspace_residual, plan, state, other_workspace)

  for _ in 1:5
    r = Grico.residual!(workspace_residual, plan, state, residual_workspace)
    K = Grico.tangent(plan, state)
    @test plan.assembly_structure.tangent !== nothing
    Grico.coefficients(state) .+= K \ (-r)
  end

  @test norm(Grico.residual!(workspace_residual, plan, state, residual_workspace)) <= 1.0e-11

  for leaf in Grico.active_leaves(space), ξ in (-1.0, 0.0, 1.0)
    @test _field_value(u, state, leaf, (ξ,)) ≈ 2.0 atol = ASSEMBLY_TOL
  end
end

@testset "Nonlinear Boundary Residual And Tangent" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.ResidualProblem(u)
  Grico.add_boundary!(problem, Grico.BoundaryFace(1, Grico.UPPER),
                      NonlinearBoundaryReaction(u, 1.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 0.0))

  plan = Grico.compile(problem)
  state = Grico.State(plan)
  fill!(Grico.coefficients(state), 0.5)

  for _ in 1:6
    r = Grico.residual(plan, state)
    K = Grico.tangent(plan, state)
    Grico.coefficients(state) .+= K \ (-r)
  end

  @test norm(Grico.residual(plan, state)) <= 1.0e-11
  @test _field_value(u, state, 1, (-1.0,)) ≈ 0.0 atol = ASSEMBLY_TOL
  @test _field_value(u, state, 1, (1.0,)) ≈ 1.0 atol = ASSEMBLY_TOL
end

@testset "Nonlinear Interface Residual And Tangent" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.ResidualProblem(u)
  Grico.add_interface!(problem, NonlinearGradientJump(u))

  plan = Grico.compile(problem)
  state = Grico.State(plan)
  Grico.coefficients(state) .= [0.3, -0.8, 0.5, 1.1, -0.4]
  direction = [0.6, -0.2, 0.9, -0.7, 0.4]
  residual_value = Grico.residual(plan, state)
  tangent_value = Grico.tangent(plan, state)
  epsilon = 1.0e-7
  state_plus = Grico.State(plan, Grico.coefficients(state) .+ epsilon .* direction)
  state_minus = Grico.State(plan, Grico.coefficients(state) .- epsilon .* direction)
  finite_difference = (Grico.residual(plan, state_plus) - Grico.residual(plan, state_minus)) /
                      (2 * epsilon)

  @test tangent_value * direction ≈ finite_difference atol = 1.0e-6 rtol = 1.0e-6
  @test norm(residual_value) > 0.0
end

function _left_half_reference_quadrature()
  midpoint = -0.5
  half_length = 0.5
  offset = half_length / sqrt(3.0)
  return Grico.PointQuadrature([(midpoint - offset,), (midpoint + offset,)],
                               [half_length, half_length])
end

@testset "Cell Quadrature Override Affine" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, MassCoupling(u, u, 1.0))
  Grico.add_cell_quadrature!(problem, 1, _left_half_reference_quadrature())

  assembled = Matrix(Grico.matrix(Grico.assemble(Grico.compile(problem))))
  exact = [7/24 1/12;
           1/12 1/24]

  @test assembled ≈ exact atol = ASSEMBLY_TOL
end

@testset "Cell Quadrature Override Nonlinear" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.ResidualProblem(u)
  Grico.add_cell!(problem, NonlinearReaction(u, 4.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 2.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 2.0))
  Grico.add_cell_quadrature!(problem, 1, _left_half_reference_quadrature())

  plan = Grico.compile(problem)
  state = Grico.State(plan)
  fill!(Grico.coefficients(state), 0.5)

  residual_norm = Inf

  for _ in 1:10
    r = Grico.residual(plan, state)
    residual_norm = norm(r)
    residual_norm <= 1.0e-10 && break
    K = Grico.tangent(plan, state)
    Grico.coefficients(state) .+= K \ (-r)
  end

  residual_norm = norm(Grico.residual(plan, state))
  @test residual_norm <= 1.0e-10

  for ξ in (-1.0, 0.0, 1.0)
    @test _field_value(u, state, 1, (ξ,)) ≈ 2.0 atol = 1.0e-8
  end
end

@testset "Finite Cell Quadrature Builder" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  quadrature = Grico.finite_cell_quadrature(domain, 1, (2,), x -> x[1] - 0.5; subdivision_depth=1)
  @test quadrature !== nothing
  @test quadrature isa Grico.PointQuadrature{1,Float64}
  @test Grico.point_count(quadrature) <= 3
  @test sum(Grico.weight(quadrature, i) for i in 1:Grico.point_count(quadrature)) *
        Grico.jacobian_determinant_from_biunit_cube(domain, 1) ≈ 0.5 atol = ASSEMBLY_TOL

  reference_measure = zero(Float64)
  reference_first_moment = zero(Float64)
  reference_second_moment = zero(Float64)

  for point_index in 1:Grico.point_count(quadrature)
    ξ = Grico.point(quadrature, point_index)[1]
    weight_value = Grico.weight(quadrature, point_index)
    @test weight_value > 0.0
    @test -1.0 <= ξ <= 0.0
    reference_measure += weight_value
    reference_first_moment += weight_value * ξ
    reference_second_moment += weight_value * ξ^2
  end

  @test reference_measure ≈ 1.0 atol = ASSEMBLY_TOL
  @test reference_first_moment ≈ -0.5 atol = ASSEMBLY_TOL
  @test reference_second_moment ≈ 1 / 3 atol = ASSEMBLY_TOL
  @test Grico.finite_cell_quadrature(domain, 1, (2,), x -> 2.0 - x[1]; subdivision_depth=1) ===
        nothing
  @test Grico.finite_cell_quadrature(domain, 1, (2,), x -> 0.0; subdivision_depth=1) === nothing

  boundary_sample = Grico.finite_cell_quadrature(domain, 1, (1,), x -> x[1] - 0.5;
                                                 subdivision_depth=0)
  @test boundary_sample !== nothing
  @test sum(Grico.weight(boundary_sample, i) for i in 1:Grico.point_count(boundary_sample)) < 2.0

  bool_quadrature = Grico.finite_cell_quadrature(domain, 1, (1,), _ -> true)
  @test bool_quadrature isa Grico.PointQuadrature{1,Float64}
  @test sum(Grico.weight(bool_quadrature, i) for i in 1:Grico.point_count(bool_quadrature)) ≈ 2.0 atol = ASSEMBLY_TOL
  @test Grico.finite_cell_quadrature(domain, 1, (1,), _ -> false) === nothing

  shared_region = Grico.ImplicitRegion(x -> x[1] - 0.5; subdivision_depth=1)
  q64 = Grico._cut_cell_quadrature(shared_region, domain, 1, (2,))
  domain32 = Grico.Domain((0.0f0,), (1.0f0,), (1,))
  q32 = Grico._cut_cell_quadrature(shared_region, domain32, 1, (2,))
  @test q64 isa Grico.PointQuadrature{1,Float64}
  @test q32 isa Grico.PointQuadrature{1,Float32}

  @test !Base.ismutable(shared_region)
  @test Grico._subcell_corner_count(Val(2)) == 4
  @test Grico._subcell_sample_count(Val(2)) == 5
  @test_throws ArgumentError Grico._subcell_corner_count(Val(Sys.WORD_SIZE - 1))
end

@testset "Physical Domain Compile" begin
  background = Grico.Domain((0.0,), (3.0,), (3,))
  domain = Grico.PhysicalDomain(background,
                                Grico.ImplicitRegion(x -> x[1] - 1.5; subdivision_depth=1))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, MassCoupling(u, u, 1.0))
  plan = Grico.compile(problem)

  @test Grico.active_leaves(space) == [1, 2]
  @test length(plan.integration.cells) == 2
  @test plan.integration.cells[1].leaf == 1
  @test plan.integration.cells[2].leaf == 2
  @test length(plan.integration.cells[1].weights) == 3
  @test sum(plan.integration.cells[2].weights) ≈ 0.5 atol = ASSEMBLY_TOL
end

@testset "Finite Cell Extension Measure" begin
  background = Grico.Domain((0.0,), (1.0,), (1,))
  region = Grico.ImplicitRegion(x -> x[1] - 0.5; subdivision_depth=1)
  physical_domain = Grico.PhysicalDomain(background, region)
  extended_domain = Grico.PhysicalDomain(background, region;
                                         cell_measure=Grico.FiniteCellExtension(0.2))

  @test_throws ArgumentError Grico.FiniteCellExtension(-0.1)
  @test_throws ArgumentError Grico.FiniteCellExtension(1.1)
  @test_throws ArgumentError Grico.FiniteCellExtension(Inf)

  physical_space = Grico.HpSpace(physical_domain,
                                 Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                    degree=Grico.UniformDegree(1)))
  extended_space = Grico.HpSpace(extended_domain,
                                 Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                    degree=Grico.UniformDegree(1)))
  physical_field = Grico.ScalarField(physical_space; name=:u)
  extended_field = Grico.ScalarField(extended_space; name=:u)
  physical_problem = Grico.AffineProblem(physical_field)
  extended_problem = Grico.AffineProblem(extended_field)
  Grico.add_cell!(physical_problem, MassCoupling(physical_field, physical_field, 1.0))
  Grico.add_cell!(extended_problem, MassCoupling(extended_field, extended_field, 1.0))
  physical_plan = Grico.compile(physical_problem)
  extended_plan = Grico.compile(extended_problem)

  @test sum(physical_plan.integration.cells[1].weights) ≈ 0.5 atol = ASSEMBLY_TOL
  @test sum(extended_plan.integration.cells[1].weights) ≈ 0.6 atol = ASSEMBLY_TOL
end

@testset "Finite Cell Default Backend Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  quadrature = Grico.finite_cell_quadrature(Grico.domain(space), 1,
                                            Grico.cell_quadrature_shape(space, 1), x -> x[1] - 0.5;
                                            subdivision_depth=1)
  @test quadrature !== nothing

  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, MassCoupling(u, u, 1.0))
  Grico.add_cell_quadrature!(problem, 1, quadrature)

  assembled = Matrix(Grico.matrix(Grico.assemble(Grico.compile(problem))))
  exact = [7/24 1/12;
           1/12 1/24]

  @test assembled ≈ exact atol = ASSEMBLY_TOL
end

@testset "Surface Quadrature Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_surface!(problem, EmbeddedPointLoad(u, 1.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), 0.0))
  Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.UPPER), 0.0))
  surface = Grico.implicit_surface_quadrature(space, 2, x -> x[1] - 0.5; subdivision_depth=1)
  @test surface !== nothing
  Grico.add_surface_quadrature!(problem, surface)

  state = Grico.State(Grico.compile(problem), Grico.solve(Grico.assemble(Grico.compile(problem))))
  exact(x) = x <= 0.5 ? 0.5 * x : 0.5 * (1.0 - x)

  for leaf in Grico.active_leaves(space), ξ in (-1.0, 0.0, 1.0)
    x = Grico.map_from_biunit_cube(domain, leaf, (ξ,))[1]
    @test _field_value(u, state, leaf, (ξ,)) ≈ exact(x) atol = ASSEMBLY_TOL
  end
end

@testset "Tagged Surface Quadrature Assembly" begin
  domain = Grico.Domain((0.0,), (1.0,), (2,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  quadrature = Grico.PointQuadrature([(0.0,)], [1.0])
  left_surface = Grico.SurfaceQuadrature(1, quadrature, [(1.0,)])
  right_surface = Grico.SurfaceQuadrature(2, quadrature, [(1.0,)])

  tagged_problem = Grico.AffineProblem(u)
  @test Grico.add_surface!(tagged_problem, :left, EmbeddedPointLoad(u, 2.0)) === tagged_problem
  @test Grico.add_surface!(tagged_problem, :right, EmbeddedPointLoad(u, 3.0)) === tagged_problem
  @test Grico.add_surface_quadrature!(tagged_problem, :left, left_surface) === tagged_problem
  @test Grico.add_surface_quadrature!(tagged_problem, :right, right_surface) === tagged_problem
  tagged_plan = Grico.compile(tagged_problem)
  @test isconcretetype(eltype(tagged_plan.integration.embedded_surfaces))
  @test [item.tag for item in tagged_plan.integration.embedded_surfaces] == [:left, :right]
  tagged_system = Grico.assemble(tagged_plan)
  @test Grico.rhs(tagged_system) ≈ [1.0, 2.5, 1.5] atol = ASSEMBLY_TOL

  untagged_problem = Grico.AffineProblem(u)
  Grico.add_surface!(untagged_problem, EmbeddedPointLoad(u, 1.0))
  Grico.add_surface_quadrature!(untagged_problem, :left, left_surface)
  Grico.add_surface_quadrature!(untagged_problem, :right, right_surface)
  untagged_system = Grico.assemble(Grico.compile(untagged_problem))
  @test Grico.rhs(untagged_system) ≈ [0.5, 1.0, 0.5] atol = ASSEMBLY_TOL
end

@testset "Implicit Surface Quadrature 1D" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  surface = Grico.implicit_surface_quadrature(domain, 1, x -> x[1] - 0.35; subdivision_depth=3)
  @test surface !== nothing
  @test Grico.point_count(surface.quadrature) == 1
  @test Grico.point(surface.quadrature, 1)[1] ≈ -0.3 atol = ASSEMBLY_TOL
  @test Grico.weight(surface.quadrature, 1) ≈ 1.0 atol = ASSEMBLY_TOL
  @test surface.normals[1] == (1.0,)
end

@testset "Implicit Surface Quadrature 2D Straight Interface" begin
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  surface = Grico.implicit_surface_quadrature(domain, 1, x -> x[1] + x[2] - 0.85;
                                              subdivision_depth=0, surface_point_count=2)
  @test surface !== nothing
  @test Grico.point_count(surface.quadrature) == 2
  expected_measure = sqrt(2.0) * 0.85
  expected_normal = (inv(sqrt(2.0)), inv(sqrt(2.0)))
  measure = _embedded_surface_measure(surface, domain)
  @test measure ≈ expected_measure atol = ASSEMBLY_TOL

  for point_index in 1:Grico.point_count(surface.quadrature)
    @test sum(surface.normals[point_index][axis] * expected_normal[axis] for axis in 1:2) >
          1 - 1.0e-10
  end
end

@testset "Implicit Surface Quadrature 2D Curved Interface" begin
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  radius = 0.3
  classifier(x) = hypot(x[1] - 0.5, x[2] - 0.5) - radius
  coarse = Grico.implicit_surface_quadrature(domain, 1, classifier; subdivision_depth=2,
                                             surface_point_count=2)
  fine = Grico.implicit_surface_quadrature(domain, 1, classifier; subdivision_depth=5,
                                           surface_point_count=2)
  @test coarse !== nothing
  @test fine !== nothing
  exact_measure = 2 * pi * radius
  coarse_error = abs(_embedded_surface_measure(coarse, domain) - exact_measure)
  fine_error = abs(_embedded_surface_measure(fine, domain) - exact_measure)
  @test fine_error < coarse_error
  @test fine_error < 5.0e-2
end

@testset "Segment Mesh Embedded Surface Compile" begin
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  mesh = Grico.SegmentMesh([(0.1, 0.2), (0.9, 0.2)], [(1, 2)])
  surface = Grico.EmbeddedSurface(mesh; point_count=2)
  problem = Grico.AffineProblem(u)
  @test Grico.add_embedded_surface!(problem, :segment, surface) === problem
  plan = Grico.compile(problem)
  items = plan.integration.embedded_surfaces

  @test length(items) == 1
  @test isconcretetype(eltype(items))
  item = only(items)
  @test item.tag == :segment
  @test sum(Grico.weight(item, point_index) for point_index in 1:Grico.point_count(item)) ≈ 0.8 atol = ASSEMBLY_TOL

  for point_index in 1:Grico.point_count(item)
    normal_data = Grico.normal(item, point_index)
    @test normal_data[1] ≈ 0.0 atol = ASSEMBLY_TOL
    @test normal_data[2] ≈ -1.0 atol = ASSEMBLY_TOL
  end
end

@testset "Segment Mesh Embedded Surface Ownership" begin
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (2, 1))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  mesh = Grico.SegmentMesh([(0.5, 0.0), (0.5, 1.0)], [(1, 2)])
  surface = Grico.EmbeddedSurface(mesh; point_count=2)
  problem = Grico.AffineProblem(u)
  Grico.add_embedded_surface!(problem, surface)
  plan = Grico.compile(problem)
  items = plan.integration.embedded_surfaces

  @test length(items) == 1
  @test only(items).leaf == 2
  @test sum(Grico.weight(only(items), point_index)
            for point_index in 1:Grico.point_count(only(items))) ≈ 1.0 atol = ASSEMBLY_TOL
end

@testset "Embedded Validation" begin
  mesh = Grico.SegmentMesh([(0.0, 0.0), (1.0, 0.0)], [(1, 2)])
  huge_index = big(typemax(Int)) + 1
  @test_throws ArgumentError Grico.SegmentMesh{Float64}([(0.0, 0.0)], NTuple{2,Int}[(1, 1)])
  @test_throws ArgumentError Grico.SegmentMesh{Float64}([(0.0, 0.0), (1.0, 0.0)],
                                                        NTuple{2,Int}[(1, 1)])
  @test_throws ArgumentError Grico.SegmentMesh([(0.0,), (1.0,)], [(1, 2)])
  @test_throws ArgumentError Grico.SegmentMesh([("bad", 0.0), (1.0, 0.0)], [(1, 2)])
  @test_throws ArgumentError Grico.SegmentMesh([(0.0, 0.0), (1.0, 0.0)], [(1.5, 2)])
  @test_throws ArgumentError Grico.EmbeddedSurface{typeof(mesh)}(mesh, 0)
  @test_throws ArgumentError Grico.EmbeddedSurface(mesh; point_count=huge_index)
  @test_throws ArgumentError Grico.EmbeddedSurface(mesh; point_count=1.5)

  domain = Grico.Domain((0.0,), (1.0,), (1,))
  @test_throws ArgumentError Grico.implicit_surface_quadrature(domain, 1, _ -> true)
  @test_throws ArgumentError Grico.implicit_surface_quadrature(domain, 1, _ -> 0.0)
  @test_throws ArgumentError Grico.implicit_surface_quadrature(domain, 1, x -> x[1];
                                                               subdivision_depth=1.5)
  @test_throws ArgumentError Grico.implicit_surface_quadrature(domain, 1, x -> x[1];
                                                               surface_point_count=1.5)

  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  line_quadrature = Grico.PointQuadrature([(0.0,)], [1.0])
  @test_throws ArgumentError Grico.SurfaceQuadrature(huge_index, line_quadrature, [(1.0,)])
  @test_throws ArgumentError Grico.SurfaceQuadrature(1, line_quadrature, [("bad",)])
  @test_throws ArgumentError Grico.add_cell_quadrature!(problem, huge_index, line_quadrature)
  @test_throws ArgumentError Grico.implicit_surface_quadrature(space, huge_index, x -> x[1])
  @test_throws ArgumentError Grico.implicit_surface_quadrature(space, 1, x -> x[1];
                                                               subdivision_depth=1.5)
  @test_throws ArgumentError Grico.implicit_surface_quadrature(space, 1, x -> x[1];
                                                               surface_point_count=1.5)
  bad_surface_problem = Grico.AffineProblem(u)
  Grico.add_embedded_surface!(bad_surface_problem,
                              Grico.EmbeddedSurface(OutOfReferenceEmbeddedGeometry()))
  @test_throws ArgumentError Grico.compile(bad_surface_problem)
  quadrature = Grico.PointQuadrature([(0.0, 0.0)], [1.0])
  Grico.add_surface_quadrature!(problem, Grico.SurfaceQuadrature(1, quadrature, [(1.0, 0.0)]))
  @test_throws ArgumentError Grico.compile(problem)
end

@testset "Problem Builder Validation" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(1)))
  u = Grico.ScalarField(space; name=:u)
  dg_space = Grico.HpSpace(domain,
                           Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                              degree=Grico.UniformDegree(1), continuity=:dg))
  w = Grico.ScalarField(dg_space; name=:w)
  other_domain = Grico.Domain((0.0,), (1.0,), (1,))
  other_space = Grico.HpSpace(other_domain,
                              Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                 degree=Grico.UniformDegree(1)))
  v = Grico.ScalarField(other_space; name=:v)
  incompatible_domain = Grico.Domain((0.0,), (2.0,), (1,))
  incompatible_space = Grico.HpSpace(incompatible_domain,
                                     Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                        degree=Grico.UniformDegree(1)))
  incompatible = Grico.ScalarField(incompatible_space; name=:incompatible)
  vec = Grico.VectorField(space, 2; name=:vec)

  huge_index = big(typemax(Int)) + 1
  @test_throws ArgumentError Grico.BoundaryFace(0, Grico.LOWER)
  @test_throws ArgumentError Grico.BoundaryFace(huge_index, Grico.LOWER)
  @test_throws ArgumentError Grico.BoundaryFace(1, huge_index)
  @test_throws ArgumentError Grico.AffineProblem(u, u)
  @test_throws ArgumentError Grico.ResidualProblem(u, u)
  @test_throws ArgumentError Grico.AffineProblem(u, incompatible)
  @test_throws ArgumentError Grico.ResidualProblem(u, incompatible)

  problem = Grico.AffineProblem(u)
  @test_throws ArgumentError Grico.add_boundary!(problem, Grico.BoundaryFace(2, Grico.LOWER),
                                                 BoundaryLoad(u, 1.0))
  @test_throws ArgumentError Grico.add_constraint!(problem,
                                                   Grico.Dirichlet(v,
                                                                   Grico.BoundaryFace(1,
                                                                                      Grico.LOWER),
                                                                   0.0))
  @test_throws ArgumentError Grico.add_constraint!(Grico.AffineProblem(w),
                                                   Grico.Dirichlet(w,
                                                                   Grico.BoundaryFace(1,
                                                                                      Grico.LOWER),
                                                                   0.0))
  @test_throws ArgumentError Grico.Dirichlet(v, Grico.BoundaryFace(1, Grico.LOWER), 2, 0.0)
  @test_throws ArgumentError Grico.Dirichlet(vec, Grico.BoundaryFace(1, Grico.LOWER), 0, 0.0)
  @test_throws ArgumentError Grico.Dirichlet(vec, Grico.BoundaryFace(1, Grico.LOWER), 3, 0.0)
  @test_throws ArgumentError Grico.Dirichlet(vec, Grico.BoundaryFace(1, Grico.LOWER), huge_index,
                                             0.0)
  @test_throws ArgumentError Grico.Dirichlet(vec, Grico.BoundaryFace(1, Grico.LOWER), (huge_index,),
                                             0.0)
  @test_throws ArgumentError Grico.Dirichlet(vec, Grico.BoundaryFace(1, Grico.LOWER), (2, 1),
                                             (0.0, 0.0))
  @test_throws ArgumentError Grico.Dirichlet(vec, Grico.BoundaryFace(1, Grico.LOWER), (1, 1),
                                             (0.0, 0.0))
  bad_dirichlet_problem = Grico.AffineProblem(u)
  Grico.add_constraint!(bad_dirichlet_problem,
                        Grico.Dirichlet(u, Grico.BoundaryFace(1, Grico.LOWER), "bad"))
  @test_throws ArgumentError Grico.compile(bad_dirichlet_problem)
  bad_vector_dirichlet_problem = Grico.AffineProblem(vec)
  Grico.add_constraint!(bad_vector_dirichlet_problem,
                        Grico.Dirichlet(vec, Grico.BoundaryFace(1, Grico.LOWER), (1, 2),
                                        (0.0, "bad")))
  @test_throws ArgumentError Grico.compile(bad_vector_dirichlet_problem)
  mixed_domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
  mixed_space = Grico.HpSpace(mixed_domain,
                              Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                 degree=Grico.AxisDegrees((1, 0)),
                                                 continuity=(:cg, :dg)))
  m = Grico.ScalarField(mixed_space; name=:m)
  mixed_problem = Grico.AffineProblem(m)
  @test Grico.add_constraint!(mixed_problem,
                              Grico.Dirichlet(m, Grico.BoundaryFace(1, Grico.LOWER), 0.0)) ===
        mixed_problem
  Grico.add_boundary!(mixed_problem, Grico.BoundaryFace(2, Grico.LOWER), BoundaryLoad(m, 1.0))
  @test Grico.compile(mixed_problem) isa Grico.AssemblyPlan
  @test_throws ArgumentError Grico.add_constraint!(Grico.AffineProblem(m),
                                                   Grico.Dirichlet(m,
                                                                   Grico.BoundaryFace(2,
                                                                                      Grico.LOWER),
                                                                   0.0))
  @test_throws ArgumentError Grico.add_constraint!(problem, Grico.MeanValue(v, 0.0))
  @test Grico.constrain!(problem, Grico.MeanValue(u, 0.0)) === problem
  @test only(problem.mean_constraints).target == 0.0
  bad_mean_problem = Grico.AffineProblem(u)
  Grico.add_constraint!(bad_mean_problem, Grico.MeanValue(u, "bad"))
  @test_throws ArgumentError Grico.compile(bad_mean_problem)
  bad_vector_mean_problem = Grico.AffineProblem(vec)
  Grico.add_constraint!(bad_vector_mean_problem, Grico.MeanValue(vec, (0.0, "bad")))
  @test_throws ArgumentError Grico.compile(bad_vector_mean_problem)

  raw = Grico.AffineProblem(Grico.AbstractField[u], Any[], Grico._BoundaryContribution[], Any[],
                            Grico._SurfaceContribution[], Grico._CellQuadratureAttachment[],
                            Grico._SurfaceAttachment[],
                            Grico.Dirichlet[Grico.Dirichlet(v, Grico.BoundaryFace(1, Grico.LOWER),
                                                            0.0)], Grico.MeanValue[])
  @test_throws ArgumentError Grico.compile(raw)

  raw_dg = Grico.AffineProblem(Grico.AbstractField[w], Any[], Grico._BoundaryContribution[], Any[],
                               Grico._SurfaceContribution[], Grico._CellQuadratureAttachment[],
                               Grico._SurfaceAttachment[],
                               Grico.Dirichlet[Grico.Dirichlet(w,
                                                               Grico.BoundaryFace(1, Grico.LOWER),
                                                               0.0)], Grico.MeanValue[])
  @test_throws ArgumentError Grico.compile(raw_dg)

  overlap_domain = Grico.Domain((0.0,), (1.0,), (1,))
  overlap_space = Grico.HpSpace(overlap_domain,
                                Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                                   degree=Grico.UniformDegree(1)))
  overlap = Grico.VectorField(overlap_space, 2; name=:overlap)
  overlap_problem = Grico.AffineProblem(overlap)
  Grico.add_constraint!(overlap_problem,
                        Grico.Dirichlet(overlap, Grico.BoundaryFace(1, Grico.LOWER), 1, 0.0))
  Grico.add_constraint!(overlap_problem,
                        Grico.Dirichlet(overlap, Grico.BoundaryFace(1, Grico.LOWER), (1, 2),
                                        (0.0, 0.0)))
  @test_throws ArgumentError Grico.compile(overlap_problem)

  tagged_problem = Grico.AffineProblem(u)
  @test Grico.add_surface!(tagged_problem, :missing, EmbeddedPointLoad(u, 1.0)) === tagged_problem
  quadrature = Grico.PointQuadrature([(0.0,)], [1.0])
  @test Grico.add_surface_quadrature!(tagged_problem, :other,
                                      Grico.SurfaceQuadrature(1, quadrature, [(1.0,)])) ===
        tagged_problem
  @test_throws ArgumentError Grico.compile(tagged_problem)

  quadrature = Grico.PointQuadrature([(0.0, 0.0)], [1.0])
  @test_throws ArgumentError Grico.SurfaceQuadrature(0, quadrature, [(1.0, 0.0)])
  @test_throws ArgumentError Grico.SurfaceQuadrature(1, quadrature, [(0.0, 0.0)])
  @test_throws ArgumentError Grico.SurfaceQuadrature(1, quadrature, NTuple{2,Float64}[])
  @test_throws ArgumentError Grico.SurfaceQuadrature{2,Float64,typeof(quadrature)}(1, quadrature,
                                                                                   [(0.0, 0.0)])
end

@testset "State Layout Validation" begin
  domain = Grico.Domain((0.0,), (1.0,), (1,))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2)))
  u = Grico.ScalarField(space; name=:u)
  v = Grico.ScalarField(space; name=:v)
  problem = Grico.ResidualProblem(u)
  Grico.add_cell!(problem, NonlinearReaction(u, 1.0))
  plan = Grico.compile(problem)
  foreign_state = Grico.State(Grico.FieldLayout((v,)), fill(0.5, Grico.dof_count(plan)))

  @test_throws ArgumentError Grico.residual(plan, foreign_state)
  @test_throws ArgumentError Grico.tangent(plan, foreign_state)
end

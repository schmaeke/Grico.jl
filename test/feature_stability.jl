using Test
using Grico
import Grico: JacobiPreconditioner, add_cell!, default_tangent_linear_solve,
              implicit_surface_quadrature, local_dof_index, local_mode_count, point_count,
              sample_mesh_skeleton, shape_value

struct _StabilityQuadraticReaction{F,T}
  field::F
  target::T
end

struct _StabilityScaledAffine{T}
  scale::T
  rhs::T
end

function Grico.cell_apply!(local_result, operator::_StabilityScaledAffine, values,
                           local_coefficients)
  for index in eachindex(local_coefficients)
    local_result[index] += operator.scale * local_coefficients[index]
  end

  return nothing
end

function Grico.cell_diagonal!(local_diagonal, operator::_StabilityScaledAffine, values)
  for index in eachindex(local_diagonal)
    local_diagonal[index] += operator.scale
  end

  return nothing
end

function Grico.cell_rhs!(local_rhs, operator::_StabilityScaledAffine, values)
  for index in eachindex(local_rhs)
    local_rhs[index] += operator.rhs
  end

  return nothing
end

function Grico.cell_residual!(local_residual, operator::_StabilityQuadraticReaction, values, state)
  field = operator.field

  for point_index in 1:point_count(values)
    field_value = value(values, state, field, point_index)
    weighted = (field_value * field_value - operator.target) * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_residual[row] += shape_value(values, field, point_index, mode_index) * weighted
    end
  end

  return nothing
end

function Grico.cell_tangent_apply!(local_result, operator::_StabilityQuadraticReaction, values,
                                   state, local_increment)
  field = operator.field

  for point_index in 1:point_count(values)
    field_value = value(values, state, field, point_index)
    increment_value = value(values, local_increment, field, point_index)
    weighted = 2 * field_value * increment_value * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_result[row] += shape_value(values, field, point_index, mode_index) * weighted
    end
  end

  return nothing
end

function _throws_argument_message(f, needle::AbstractString)
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

@testset "Feature Stability" begin
  @testset "Advanced API Is Qualified" begin
    for name in (:adaptivity_plan, :finite_cell_quadrature, :implicit_surface_quadrature)
      @test Base.ispublic(Grico, name)
      @test !Base.isexported(Grico, name)
    end
  end

  @testset "Unsupported Dimensions" begin
    domain4 = Domain(ntuple(_ -> 0.0, 4), ntuple(_ -> 1.0, 4), ntuple(_ -> 1, 4))
    _throws_argument_message(() -> sample_postprocess(domain4), "dimension between 1 and 3")
    _throws_argument_message(() -> sample_mesh_skeleton(domain4), "dimension between 1 and 3")

    domain3 = Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
    _throws_argument_message(() -> implicit_surface_quadrature(domain3, 1, x -> x[1] - 0.5),
                             "dimensions 1 and 2")
  end

  @testset "Embedded Surface Active Frontiers" begin
    inactive_domain = Domain((0.0,), (1.0,), (1,))
    refine!(grid(inactive_domain), 1, 1)
    _throws_argument_message(() -> implicit_surface_quadrature(inactive_domain, 1, x -> x[1] - 0.5),
                             "active leaves")

    physical_background = Domain((0.0,), (1.0,), (1,))
    refine!(grid(physical_background), 1, 1)
    physical_domain = PhysicalDomain(physical_background,
                                     ImplicitRegion(x -> -1.0; subdivision_depth=0))
    _throws_argument_message(() -> implicit_surface_quadrature(physical_domain, 1, x -> x[1] - 0.5),
                             "active leaves")

    virtual_domain = Domain((0.0,), (1.0,), (1,))
    virtual_snapshot, children = Grico._refine_snapshot_leaf!(Grico.snapshot(grid(virtual_domain)),
                                                              1, (true,))
    _throws_argument_message(() -> implicit_surface_quadrature(virtual_domain, first(children),
                                                               x -> x[1] - 0.25), "active leaves")

    virtual_space = Grico._compile_snapshot_space(virtual_domain, virtual_snapshot,
                                                  SpaceOptions(degree=UniformDegree(1)))
    @test Grico.check_space(virtual_space) === nothing
    @test implicit_surface_quadrature(virtual_space, first(children), x -> x[1] - 0.25) !== nothing
  end

  @testset "Extension Fallbacks" begin
    domain = Domain((0.0,), (1.0,), (1,))
    space = HpSpace(domain, SpaceOptions(degree=UniformDegree(1)))
    u = ScalarField(space; name=:u)
    state = State(FieldLayout((u,)), zeros(field_dof_count(u)))

    _throws_argument_message(() -> write_vtk("unloaded", domain), "requires WriteVTK")
    _throws_argument_message(() -> write_pvd("unloaded.pvd", String[]), "requires WriteVTK")
    _throws_argument_message(() -> plot_field(state, :u), "requires Makie")
    _throws_argument_message(() -> plot_mesh(domain), "requires Makie")
  end

  @testset "Default Nonlinear Solve Policy" begin
    domain = Domain((0.0,), (1.0,), (1,))
    space = HpSpace(domain, SpaceOptions(degree=UniformDegree(0), continuity=:dg))
    u = ScalarField(space; name=:u)
    problem = ResidualProblem(u)
    add_cell!(problem, _StabilityQuadraticReaction(u, 4.0))
    plan = compile(problem)
    state = State(plan, [1.0])

    _throws_argument_message(() -> default_tangent_linear_solve(plan, state, [0.0];
                                                                preconditioner=JacobiPreconditioner()),
                             "tangent preconditioning is not implemented")
    _throws_argument_message(() -> solve(plan; solver=FGMRESSolver(), initial_state=state),
                             "residual solves use linear_solve")
    _throws_argument_message(() -> solve(plan; initial_state=state,
                                         linear_solve=(args...; kwargs...) -> [1.0, 99.0]),
                             "Newton correction must have length 1")
  end

  @testset "Scaled Linear Solve Contracts" begin
    domain = Domain((0.0,), (1.0,), (1,))
    space = HpSpace(domain, SpaceOptions(degree=UniformDegree(0), continuity=:dg))
    u = ScalarField(space; name=:u)

    tiny_rhs_problem = AffineProblem(u)
    add_cell!(tiny_rhs_problem, _StabilityScaledAffine(1.0, 1.0e-14))
    @test only(coefficients(solve(tiny_rhs_problem))) == 1.0e-14

    tiny_jacobi_problem = AffineProblem(u)
    add_cell!(tiny_jacobi_problem, _StabilityScaledAffine(1.0e-14, 1.0e-14))
    jacobi_state = solve(tiny_jacobi_problem;
                         solver=CGSolver(preconditioner=JacobiPreconditioner()))
    @test only(coefficients(jacobi_state)) ≈ 1.0

    tiny_fgmres_problem = AffineProblem(u; operator_class=NonsymmetricOperator())
    add_cell!(tiny_fgmres_problem, _StabilityScaledAffine(1.0e-14, 1.0e-14))
    fgmres_state = solve(tiny_fgmres_problem; solver=FGMRESSolver())
    @test only(coefficients(fgmres_state)) ≈ 1.0

    _throws_argument_message(() -> solve(tiny_rhs_problem; relative_tolerance=-1.0),
                             "relative_tolerance")
    _throws_argument_message(() -> solve(tiny_rhs_problem; absolute_tolerance=Inf),
                             "absolute_tolerance")
  end
end

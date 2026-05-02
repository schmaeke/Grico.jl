using Test
using Grico

struct _ProblemApiBoundaryMarker end

struct _ProblemApiSurfaceLoad{F,T}
  field::F
  value::T
end

function Grico.surface_rhs!(local_rhs, operator::_ProblemApiSurfaceLoad, values)
  field = operator.field

  for point_index in 1:point_count(values)
    weighted = operator.value * weight(values, point_index)

    for mode_index in 1:local_mode_count(values, field)
      row = local_dof_index(values, field, 1, mode_index)
      local_rhs[row] += shape_value(values, field, point_index, mode_index) * weighted
    end
  end

  return nothing
end

function _throws_problem_argument_message(f, needle::AbstractString)
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

@testset "Problem API" begin
  @testset "Storage Is Internal" begin
    domain = Domain((0.0,), (1.0,), (1,))
    space = HpSpace(domain, SpaceOptions(degree=UniformDegree(1)))
    u = ScalarField(space; name=:u)
    problem = AffineProblem(u)

    @test fields(problem) == (u,)
    @test field_count(problem) == 1
    @test propertynames(problem) == ()
    _throws_problem_argument_message(() -> getproperty(problem, :cell_operators),
                                     "problem storage")
    _throws_problem_argument_message(() -> setproperty!(problem, :fields, AbstractField[]),
                                     "problem storage")
  end

  @testset "Constructor And Constraint Validation" begin
    domain = Domain((0.0,), (1.0,), (1,))
    space = HpSpace(domain, SpaceOptions(degree=UniformDegree(1)))
    u = ScalarField(space; name=:u)
    v = ScalarField(space; name=:v)
    other_domain = Domain((0.0, 0.0), (1.0, 1.0), (1, 1))
    other_space = HpSpace(other_domain, SpaceOptions(degree=UniformDegree(1)))
    w = ScalarField(other_space; name=:w)

    @test_throws ArgumentError AffineProblem()
    @test_throws ArgumentError AffineProblem(u, u)
    @test_throws ArgumentError AffineProblem(u, w)

    problem = AffineProblem(u)
    @test add_cell!(problem, _ProblemApiBoundaryMarker()) === problem
    @test_throws ArgumentError add_constraint!(problem, Dirichlet(v, BoundaryFace(1, LOWER), 0.0))

    periodic_domain = Domain((0.0,), (1.0,), (1,); periodic=true)
    periodic_space = HpSpace(periodic_domain, SpaceOptions(degree=UniformDegree(1)))
    periodic_field = ScalarField(periodic_space; name=:periodic)
    periodic_problem = AffineProblem(periodic_field)
    @test_throws ArgumentError add_boundary!(periodic_problem, BoundaryFace(1, LOWER),
                                             _ProblemApiBoundaryMarker())
    @test_throws ArgumentError add_constraint!(periodic_problem,
                                               Dirichlet(periodic_field,
                                                         BoundaryFace(1, LOWER), 0.0))

    dg_space = HpSpace(domain, SpaceOptions(degree=UniformDegree(0), continuity=:dg))
    dg_field = ScalarField(dg_space; name=:dg)
    @test_throws ArgumentError add_constraint!(AffineProblem(dg_field),
                                               Dirichlet(dg_field, BoundaryFace(1, LOWER), 0.0))
  end

  @testset "Contribution Types And Surface Tags" begin
    domain = Domain((0.0,), (1.0,), (1,))
    space = HpSpace(domain, SpaceOptions(degree=UniformDegree(0), continuity=:dg))
    u = ScalarField(space; name=:u)

    boundary_problem = AffineProblem(u)
    boundary_operator = _ProblemApiBoundaryMarker()
    add_boundary!(boundary_problem, BoundaryFace(1, LOWER), boundary_operator)
    boundary_plan = compile(boundary_problem)
    @test fieldtype(typeof(boundary_plan.boundary_operators[1]), :operator) ===
          typeof(boundary_operator)

    quadrature = PointQuadrature([(0.0,)], [1.0])
    surface = SurfaceQuadrature(1, quadrature, [(1.0,)])
    selected_problem = AffineProblem(u)
    add_surface_quadrature!(selected_problem, :selected, surface)
    add_surface_quadrature!(selected_problem, :ignored, surface)
    selected_operator = _ProblemApiSurfaceLoad(u, 3.0)
    add_surface!(selected_problem, :selected, selected_operator)
    selected_plan = compile(selected_problem)
    selected_rhs = rhs(selected_plan)

    @test fieldtype(typeof(selected_plan.surface_operators[1]), :operator) ===
          typeof(selected_operator)
    @test selected_rhs[1] > 0

    all_problem = AffineProblem(u)
    add_surface_quadrature!(all_problem, :selected, surface)
    add_surface_quadrature!(all_problem, :ignored, surface)
    add_surface!(all_problem, selected_operator)
    @test rhs(compile(all_problem)) ≈ 2 .* selected_rhs

    missing_tag_problem = AffineProblem(u)
    add_surface_quadrature!(missing_tag_problem, :selected, surface)
    add_surface!(missing_tag_problem, :missing, selected_operator)
    @test_throws ArgumentError compile(missing_tag_problem)
  end
end

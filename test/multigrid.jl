using Test
using Grico

struct _MGIdentity end

function Grico.cell_apply!(local_result, ::_MGIdentity, values, local_coefficients)
  copyto!(local_result, local_coefficients)
  return nothing
end

function Grico.cell_rhs!(local_rhs, ::_MGIdentity, values)
  fill!(local_rhs, 1.0)
  return nothing
end

function _mg_identity_problem(; continuity=:dg, degree=3, cells=(2,), refine=nothing)
  domain = Domain((0.0,), (1.0,), cells)
  refine === nothing || refine(grid(domain))
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(degree), continuity=continuity))
  field = ScalarField(space; name=:u)
  problem = AffineProblem(field; operator_class=SPD())
  add_cell!(problem, _MGIdentity())
  return problem, field
end

function _mg_identity_problem_2d(; continuity=:cg, degree=1, cells=(2, 1), refine=nothing)
  domain = Domain((0.0, 0.0), (1.0, 1.0), cells)
  refine === nothing || refine(grid(domain))
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(degree), continuity=continuity))
  field = ScalarField(space; name=:u)
  problem = AffineProblem(field; operator_class=SPD())
  add_cell!(problem, _MGIdentity())
  return problem, field
end

function _test_transfer_adjoint(transfer)
  coarse = [sin(0.2 * index) for index in 1:Grico.reduced_dof_count(transfer.coarse_plan)]
  fine = [cos(0.3 * index) for index in 1:Grico.reduced_dof_count(transfer.fine_plan)]
  prolonged = zeros(length(fine))
  restricted = zeros(length(coarse))
  Grico._prolongate_reduced_add!(prolonged, transfer, coarse)
  Grico._restrict_reduced!(restricted, transfer, fine)
  return sum(prolonged .* fine) ≈ sum(coarse .* restricted)
end

@testset "Geometric multigrid" begin
  dg_problem, dg_field = _mg_identity_problem(continuity=:dg, degree=3)
  dg_state = solve(dg_problem;
                   solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))
  @test coefficients(dg_state) ≈ ones(field_dof_count(dg_field))

  cg_problem, cg_field = _mg_identity_problem(continuity=:cg, degree=3)
  add_constraint!(cg_problem, Dirichlet(cg_field, BoundaryFace(1, LOWER), 2.0))
  cg_state = solve(cg_problem;
                   solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))
  @test first(coefficients(cg_state)) ≈ 2.0
  @test coefficients(cg_state)[2:end] ≈ ones(field_dof_count(cg_field) - 1)

  hierarchy = Grico._compile_geometric_multigrid(dg_problem, GeometricMultigridPreconditioner())
  @test length(hierarchy.levels) == 2
  @test hierarchy.coarse_solver isa Grico._DenseCholeskyCoarseSolver
  @test _test_transfer_adjoint(only(hierarchy.transfers))

  weak_problem, weak_field = _mg_identity_problem(continuity=:dg, degree=3)
  weak_problem = AffineProblem(weak_field; operator_class=SPD())
  add_cell_bilinear!(weak_problem, weak_field, weak_field) do q, v, w
    value(v) * value(w)
  end
  add_cell_linear!(weak_problem, weak_field) do q, v
    value(v)
  end
  weak_hierarchy = Grico._compile_geometric_multigrid(weak_problem, GeometricMultigridPreconditioner())
  @test weak_hierarchy.coarse_solver isa Grico._DenseCholeskyCoarseSolver
  weak_gmg = solve(weak_problem;
                   solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))
  weak_cg = solve(compile(weak_problem);
                  solver=CGSolver(preconditioner=JacobiPreconditioner()),
                  relative_tolerance=1.0e-12)
  @test coefficients(weak_gmg) ≈ coefficients(weak_cg) atol = 1.0e-8
  @test_throws ArgumentError solve(compile(weak_problem);
                                   solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))

  nonsym_domain = Domain((0.0,), (1.0,), (2,))
  nonsym_space = HpSpace(nonsym_domain,
                         SpaceOptions(degree=UniformDegree(3), continuity=:dg))
  nonsym_field = ScalarField(nonsym_space; name=:u)
  nonsym_problem = AffineProblem(nonsym_field; operator_class=NonsymmetricOperator())
  add_cell_bilinear!(nonsym_problem, nonsym_field, nonsym_field) do q, v, w
    value(v) * value(w) + 0.1 * value(v) * grad(w)[1]
  end
  add_cell_linear!(nonsym_problem, nonsym_field) do q, v
    value(v)
  end
  nonsym_plan = compile(nonsym_problem)
  nonsym_workspace = Grico._ReducedOperatorWorkspace(nonsym_plan)
  nonsym_matrix = Grico._assemble_reduced_operator_matrix(nonsym_plan,
                                                          nonsym_workspace.scratch)
  nonsym_rhs = rhs(nonsym_plan)
  nonsym_hierarchy = Grico._compile_geometric_multigrid(nonsym_problem,
                                                        GeometricMultigridPreconditioner())
  @test nonsym_hierarchy.coarse_solver isa Grico._DenseLUCoarseSolver
  nonsym_state = solve(nonsym_problem;
                       solver=FGMRESSolver(preconditioner=GeometricMultigridPreconditioner()),
                       relative_tolerance=1.0e-12)
  @test coefficients(nonsym_state) ≈ nonsym_matrix \ nonsym_rhs atol = 1.0e-8
  nonsym_auto = solve(nonsym_problem; solver=AutoLinearSolver(), relative_tolerance=1.0e-12)
  @test coefficients(nonsym_auto) ≈ nonsym_matrix \ nonsym_rhs atol = 1.0e-8

  low_order_problem, = _mg_identity_problem(continuity=:dg, degree=1)
  @test coefficients(solve(low_order_problem; solver=AutoLinearSolver())) ≈ ones(4)

  h_problem, h_field = _mg_identity_problem(continuity=:dg, degree=1, cells=(1,),
                                            refine=grid -> begin
                                              first = refine!(grid, 1, 1)
                                              refine!(grid, first + 1, 1)
                                            end)
  h_hierarchy = Grico._compile_geometric_multigrid(h_problem, GeometricMultigridPreconditioner())
  @test map(level -> Grico.reduced_dof_count(level.plan), h_hierarchy.levels) == [2, 4, 6]
  @test all(_test_transfer_adjoint, h_hierarchy.transfers)
  @test isapprox(coefficients(solve(h_problem;
                                    solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))),
                 ones(field_dof_count(h_field)); atol=1.0e-8)
  @test isapprox(coefficients(solve(h_problem; solver=AutoLinearSolver())),
                 ones(field_dof_count(h_field)); atol=1.0e-8)

  hp_problem, = _mg_identity_problem(continuity=:dg, degree=3, cells=(1,),
                                     refine=grid -> refine!(grid, 1, 1))
  hp_hierarchy = Grico._compile_geometric_multigrid(hp_problem, GeometricMultigridPreconditioner())
  @test map(level -> Grico.reduced_dof_count(level.plan), hp_hierarchy.levels) == [2, 4, 8]
  @test all(_test_transfer_adjoint, hp_hierarchy.transfers)

  hanging_problem, hanging_field = _mg_identity_problem_2d(continuity=:cg, degree=1,
                                                           refine=grid -> refine!(grid, 1, 1))
  hanging_hierarchy = Grico._compile_geometric_multigrid(hanging_problem,
                                                         GeometricMultigridPreconditioner())
  @test length(hanging_hierarchy.levels) == 2
  @test _test_transfer_adjoint(only(hanging_hierarchy.transfers))
  @test coefficients(solve(hanging_problem;
                           solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))) ≈
        ones(field_dof_count(hanging_field))
end

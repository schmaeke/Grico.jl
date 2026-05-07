using Test
using Grico
import Grico: JacobiPreconditioner, add_cell!

struct _MGIdentity end
struct _MGMass end
struct _MGNonsymmetricMassGradient end

function Grico.cell_apply!(local_result, ::_MGIdentity, values, local_coefficients)
  copyto!(local_result, local_coefficients)
  return nothing
end

function Grico.cell_rhs!(local_rhs, ::_MGIdentity, values)
  fill!(local_rhs, 1.0)
  return nothing
end

Grico.cell_accumulate(::_MGMass, q, trial, test_component) = value(trial)

Grico.cell_rhs_accumulate(::_MGMass, q, test_component) = 1.0

function Grico.cell_accumulate(::_MGNonsymmetricMassGradient, q, trial, test_component)
  return value(trial) + 0.1 * gradient(trial)[1]
end

Grico.cell_rhs_accumulate(::_MGNonsymmetricMassGradient, q, test_component) = 1.0

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
  @test_throws ArgumentError GeometricMultigridPreconditioner(smoother_damping=Inf)
  @test_throws ArgumentError GeometricMultigridPreconditioner(coarse_relative_tolerance=Inf)
  @test_throws ArgumentError GeometricMultigridPreconditioner(coarse_absolute_tolerance=NaN)
  @test !Grico._matrix_is_symmetric([1.0e-14 0.0; 1.0e-10 1.0e-14])
  @test !Grico._matrix_is_symmetric([1.0 NaN; NaN 1.0])
  gmres_coefficients = zeros(2)
  Grico._gmres_smoother_upper_triangular_solve!(gmres_coefficients, [1.0e-14 0.0; 0.0 1.0],
                                                [1.0, 2.0], 2)
  @test gmres_coefficients ≈ [1.0e14, 2.0]
  @test_throws ArgumentError Grico._gmres_smoother_upper_triangular_solve!(zeros(2),
                                                                           [0.0 0.0; 0.0 1.0],
                                                                           [1.0, 2.0], 2)

  dg_problem, dg_field = _mg_identity_problem(continuity=:dg, degree=3)
  dg_state = solve(dg_problem; solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))
  @test coefficients(dg_state) ≈ ones(field_dof_count(dg_field))

  cg_problem, cg_field = _mg_identity_problem(continuity=:cg, degree=3)
  add_constraint!(cg_problem, Dirichlet(cg_field, BoundaryFace(1, LOWER), 2.0))
  cg_state = solve(cg_problem; solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))
  @test first(coefficients(cg_state)) ≈ 2.0
  @test coefficients(cg_state)[2:end] ≈ ones(field_dof_count(cg_field) - 1)

  hierarchy = Grico._compile_geometric_multigrid(dg_problem, GeometricMultigridPreconditioner())
  @test length(hierarchy.levels) == 2
  @test hierarchy.coarse_solver isa Grico._DenseCholeskyCoarseSolver
  @test length(hierarchy.level_rhs) == length(hierarchy.levels) - 1
  @test length(hierarchy.level_solution) == length(hierarchy.levels) - 1
  @test isempty(hierarchy.gmres_smoothers)
  @test _test_transfer_adjoint(only(hierarchy.transfers))

  mass_problem, mass_field = _mg_identity_problem(continuity=:dg, degree=3)
  mass_problem = AffineProblem(mass_field; operator_class=SPD())
  add_cell_accumulator!(mass_problem, mass_field, mass_field, _MGMass())
  add_cell_accumulator!(mass_problem, mass_field, _MGMass())
  mass_hierarchy = Grico._compile_geometric_multigrid(mass_problem,
                                                      GeometricMultigridPreconditioner())
  @test mass_hierarchy.coarse_solver isa Grico._DenseCholeskyCoarseSolver
  mass_gmg = solve(mass_problem; solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))
  mass_cg = solve(compile(mass_problem); solver=CGSolver(preconditioner=JacobiPreconditioner()),
                  relative_tolerance=1.0e-12)
  @test coefficients(mass_gmg) ≈ coefficients(mass_cg) atol = 1.0e-8
  @test_throws ArgumentError solve(mass_problem;
                                   solver=CGSolver(preconditioner=GeometricMultigridPreconditioner(pre_smoothing_steps=1,
                                                                                                   post_smoothing_steps=2)))
  @test_throws ArgumentError solve(mass_problem;
                                   solver=CGSolver(preconditioner=GeometricMultigridPreconditioner(smoother=:gmres)))
  @test_throws ArgumentError solve(compile(mass_problem);
                                   solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))

  threshold_domain = Domain((0.0, 0.0), (1.0, 1.0), (22, 22))
  threshold_space = HpSpace(threshold_domain, SpaceOptions(degree=UniformDegree(2), continuity=:cg))
  threshold_field = ScalarField(threshold_space; name=:u)
  threshold_problem = AffineProblem(threshold_field; operator_class=SPD())
  add_cell_accumulator!(threshold_problem, threshold_field, threshold_field, _MGMass())
  threshold_hierarchy = Grico._compile_geometric_multigrid(threshold_problem,
                                                           GeometricMultigridPreconditioner())
  @test Grico.reduced_dof_count(threshold_hierarchy.levels[1].plan) > 512
  @test threshold_hierarchy.coarse_solver isa Grico._KrylovCoarseSolver
  threshold_direct = Grico._compile_geometric_multigrid(threshold_problem,
                                                        GeometricMultigridPreconditioner(coarse_direct_dof_limit=4096))
  @test threshold_direct.coarse_solver isa Grico._DenseCholeskyCoarseSolver

  nonsym_domain = Domain((0.0,), (1.0,), (2,))
  nonsym_space = HpSpace(nonsym_domain, SpaceOptions(degree=UniformDegree(3), continuity=:dg))
  nonsym_field = ScalarField(nonsym_space; name=:u)
  nonsym_problem = AffineProblem(nonsym_field; operator_class=NonsymmetricOperator())
  add_cell_accumulator!(nonsym_problem, nonsym_field, nonsym_field, _MGNonsymmetricMassGradient())
  add_cell_accumulator!(nonsym_problem, nonsym_field, _MGNonsymmetricMassGradient())
  nonsym_plan = compile(nonsym_problem)
  nonsym_workspace = Grico._ReducedOperatorWorkspace(nonsym_plan)
  nonsym_matrix = Grico._assemble_reduced_operator_matrix(nonsym_plan, nonsym_workspace.scratch)
  nonsym_rhs = rhs(nonsym_plan)
  nonsym_hierarchy = Grico._compile_geometric_multigrid(nonsym_problem,
                                                        GeometricMultigridPreconditioner())
  @test nonsym_hierarchy.coarse_solver isa Grico._DenseLUCoarseSolver
  @test nonsym_hierarchy.smoother == :gmres
  @test length(nonsym_hierarchy.gmres_smoothers) == length(nonsym_hierarchy.levels) - 1
  @test_throws ArgumentError Grico._compile_geometric_multigrid(nonsym_problem,
                                                                GeometricMultigridPreconditioner(smoother=:gmres,
                                                                                                 smoother_restart=1))
  @test Grico._default_affine_preconditioner(nonsym_problem) isa JacobiPreconditioner
  @test_throws ArgumentError solve(nonsym_problem;
                                   solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))
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

  float32_domain = Domain((0.0f0,), (1.0f0,), (2,))
  float32_space = HpSpace(float32_domain, SpaceOptions(degree=UniformDegree(3), continuity=:dg))
  float32_field = ScalarField(float32_space; name=:u)
  float32_problem = AffineProblem(float32_field; operator_class=SPD())
  add_cell!(float32_problem, _MGIdentity())
  float32_state = solve(float32_problem;
                        solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()))
  @test eltype(coefficients(float32_state)) === Float32
  @test coefficients(float32_state) ≈ ones(Float32, field_dof_count(float32_field))

  float32_h_domain = Domain((0.0f0,), (1.0f0,), (1,))
  refine!(grid(float32_h_domain), 1, 1)
  float32_h_space = HpSpace(float32_h_domain,
                            SpaceOptions(degree=UniformDegree(16), continuity=:dg))
  float32_h_field = ScalarField(float32_h_space; name=:u)
  float32_h_problem = AffineProblem(float32_h_field; operator_class=SPD())
  add_cell!(float32_h_problem, _MGIdentity())
  float32_hierarchy = Grico._compile_geometric_multigrid(float32_h_problem,
                                                         GeometricMultigridPreconditioner(min_degree=16,
                                                                                          max_levels=2))
  @test any(coefficient -> 0 < abs(coefficient) <= 1000eps(Float32),
            only(float32_hierarchy.transfers).coefficients)

  diagnostics = Grico.multigrid_diagnostics(mass_problem;
                                            preconditioner=GeometricMultigridPreconditioner(),
                                            repetitions=1)
  @test diagnostics.levels == 2
  @test diagnostics.smoother == :jacobi
  @test diagnostics.coarse_solver == :dense_cholesky
  @test diagnostics.vcycle_seconds_per_call >= 0
end

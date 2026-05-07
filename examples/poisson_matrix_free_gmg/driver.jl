using Printf
using Grico
using WriteVTK

# This is the smallest complete matrix-free solver example in the repository.
# It solves the manufactured Poisson problem
#
#   -Δu = f  in Ω = (0, 1)²,      u = 0 on ∂Ω,
#
# with exact solution
#
#   uₑ(x, y) = sin(πx) sin(πy).
#
# The Poisson operator is defined by local accumulator callbacks. No dense global
# stiffness matrix is formed: `solve` applies the affine operator through the
# compiled matrix-free plan and uses geometric multigrid as the preconditioner
# for CG. This is the intended default path for supported affine SPD problems on
# Cartesian domains.

const DEFAULT_ROOT_COUNTS = (4, 4)
const DEFAULT_DEGREE = 4
const DEFAULT_EXPORT_DEGREE = DEFAULT_DEGREE

struct PoissonDiffusion end

struct PoissonLoad{F}
  source::F
end

# The scalar Poisson operator is the simplest accumulator example. Returning the
# reconstructed trial gradient as the test-gradient coefficient represents
# `∫Ω ∇v · ∇u dΩ`; there is no coefficient multiplying `v`.
function Grico.cell_accumulate(::PoissonDiffusion, q, trial, test_component)
  return TestChannels(zero(value(trial)), gradient(trial))
end

# The right-hand side is factored into a tiny operator object so the same
# registration pattern works for manufactured loads, measured data, or closures
# over physical parameters.
Grico.cell_rhs_accumulate(operator::PoissonLoad, q, test_component) = operator.source(point(q))

poisson_exact(x) = prod(sinpi(x[axis]) for axis in 1:2)

# Since `-∂² sin(πx) = π² sin(πx)` in each coordinate direction, the two-
# dimensional source is `2π² uₑ`.
poisson_source(x) = 2 * π^2 * poisson_exact(x)

function build_poisson_matrix_free_gmg_problem(; root_counts=DEFAULT_ROOT_COUNTS,
                                               degree=DEFAULT_DEGREE)
  # The full-tensor basis and uniform degree keep the geometry and polynomial
  # space compatible with the fast tensor-product matrix-free kernels and the
  # rediscretized GMG hierarchy.
  domain = Domain((0.0, 0.0), (1.0, 1.0), root_counts)
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(degree)))
  u = ScalarField(space; name=:u)

  # Declaring the affine problem as SPD lets the automatic and explicit solver
  # policies choose CG-compatible preconditioning. The example passes an
  # explicit GMG preconditioner below to make the intended solver path visible.
  problem = AffineProblem(u; operator_class=SPD())
  add_cell_accumulator!(problem, u, u, PoissonDiffusion())
  add_cell_accumulator!(problem, u, PoissonLoad(poisson_source))

  # Homogeneous Dirichlet conditions match the manufactured sine solution on
  # all sides of the unit square and remove the constant nullspace.
  for axis in 1:2, side in (LOWER, UPPER)
    add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, side), 0.0))
  end

  return (; domain, space, u, problem)
end

function run_poisson_matrix_free_gmg_example(; root_counts=DEFAULT_ROOT_COUNTS,
                                             degree=DEFAULT_DEGREE,
                                             solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()),
                                             write_output=false,
                                             output_directory=joinpath(@__DIR__, "output"),
                                             export_degree=DEFAULT_EXPORT_DEGREE,
                                             print_summary=true)
  context = build_poisson_matrix_free_gmg_problem(; root_counts, degree)
  # `solve` compiles a matrix-free plan internally. The operator is never
  # assembled globally; local accumulator kernels provide action, diagonal
  # information, and rediscretized coarse operators for GMG.
  state = solve(context.problem; solver)
  error_value = l2_error(state, context.u, poisson_exact)
  vtk_path = nothing

  if write_output
    # File output is optional because the core example should stay quick in the
    # test suite. When enabled, the VTK point data samples both the exact
    # manufactured solution and the absolute error.
    mkpath(output_directory)
    vtk_path = Grico.write_vtk(joinpath(output_directory, "poisson_matrix_free_gmg"), state;
                               point_data=(exact=poisson_exact,
                                           abs_error=(x, values) -> abs(values.u - poisson_exact(x))),
                               field_data=(l2_error=error_value,), sample_degree=export_degree,
                               append=true, compress=true, ascii=false)
  end

  if print_summary
    println("poisson_matrix_free_gmg/driver.jl")
    @printf("  root cells        : %d x %d\n", root_counts...)
    @printf("  degree            : %d\n", degree)
    @printf("  active leaves     : %d\n", active_leaf_count(context.space))
    @printf("  scalar dofs       : %d\n", Grico.scalar_dof_count(context.space))
    @printf("  L² error          : %.6e\n", error_value)
    vtk_path === nothing || println("  vtk               : $vtk_path")
  end

  return (; context..., state, error_value, vtk_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_poisson_matrix_free_gmg_example()
end

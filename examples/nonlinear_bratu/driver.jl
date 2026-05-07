using Printf
using Grico
using WriteVTK

# This example demonstrates Grico's accumulator operator interface without
# adding adaptivity or embedded geometry. We solve a manufactured Bratu-type
# problem
#
#   -Δu + λ exp(u) = f  in Ω = (0, 1)²,      u = 0 on ∂Ω,
#
# with exact solution
#
#   uₑ(x, y) = A sin(πx) sin(πy).
#
# The residual contributes
#
#   R(u; v) = ∫Ω ∇v · ∇u dΩ + ∫Ω λ exp(u) v dΩ - ∫Ω f v dΩ,
#
# and the tangent applies the Newton linearization
#
#   J(u)[δu; v] = ∫Ω ∇v · ∇δu dΩ + ∫Ω λ exp(u) v δu dΩ.
#
# The accumulator callbacks return only the coefficients multiplying v and ∇v
# at one quadrature point. Grico owns the basis projection, matrix-free tangent
# application, diagonal path, and tensor-product sum-factorization kernels.

const BRATU_ROOT_COUNTS = (4, 4)
const BRATU_DEGREE = 3
const BRATU_LAMBDA = 1.0
const BRATU_AMPLITUDE = 0.2

struct BratuAccumulator{T,F}
  λ::T
  source::F
end

function Grico.cell_residual_accumulate(operator::BratuAccumulator, q, state, test_component)
  # The residual is written as coefficient data for `v` and `∇v`. The value
  # coefficient contains the nonlinear reaction `λ exp(u)` and the load `-f`;
  # the gradient coefficient is `∇u`, giving the diffusion term
  # `∫Ω ∇v · ∇u dΩ`.
  return TestChannels(operator.λ * exp(state.value) - operator.source(point(q)), state.gradient)
end

function Grico.cell_tangent_accumulate(operator::BratuAccumulator, q, state, increment,
                                       test_component)
  # Newton's method needs the Jacobian action on an increment `δu`. The
  # derivative of `λ exp(u)` is `λ exp(u) δu`, and the diffusion derivative is
  # simply `∇δu`.
  return TestChannels(operator.λ * exp(state.value) * increment.value, increment.gradient)
end

bratu_exact(x; amplitude=BRATU_AMPLITUDE) = amplitude * sinpi(x[1]) * sinpi(x[2])

function bratu_source(x; λ=BRATU_LAMBDA, amplitude=BRATU_AMPLITUDE)
  # For the manufactured state, `-Δuₑ = 2π² uₑ`. Adding `λ exp(uₑ)` gives the
  # right-hand side that makes `uₑ` the exact solution of the nonlinear problem.
  exact = bratu_exact(x; amplitude)
  return 2 * π^2 * exact + λ * exp(exact)
end

function build_nonlinear_bratu_problem(; root_counts=BRATU_ROOT_COUNTS, degree=BRATU_DEGREE,
                                       λ=BRATU_LAMBDA, amplitude=BRATU_AMPLITUDE)
  # This example deliberately uses a fitted Cartesian domain and homogeneous
  # Dirichlet constraints so the nonlinear accumulator API is the only advanced
  # feature in the setup.
  domain = Domain((0.0, 0.0), (1.0, 1.0), root_counts)
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(degree)))
  u = ScalarField(space; name=:u)
  exact = x -> bratu_exact(x; amplitude)
  source = x -> bratu_source(x; λ, amplitude)

  # The tangent of this scalar Bratu problem is SPD for the chosen positive λ,
  # so the nonlinear solver can use the same matrix-free SPD linear-solver path
  # as affine elliptic examples.
  problem = ResidualProblem(u; operator_class=SPD())
  add_cell_accumulator!(problem, u, u, BratuAccumulator(λ, source))

  for axis in 1:2, side in (LOWER, UPPER)
    add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, side), 0.0))
  end

  return (; domain, space, u, exact, source, problem, λ, amplitude)
end

function run_nonlinear_bratu_example(; root_counts=BRATU_ROOT_COUNTS, degree=BRATU_DEGREE,
                                     λ=BRATU_LAMBDA, amplitude=BRATU_AMPLITUDE, write_output=false,
                                     output_directory=joinpath(@__DIR__, "output"),
                                     print_summary=true)
  context = build_nonlinear_bratu_problem(; root_counts, degree, λ, amplitude)
  # Compiling once exposes the reusable nonlinear residual and tangent plan.
  # The zero initial state is a simple and deterministic Newton starting point
  # for the small manufactured amplitudes used by the example and tests.
  plan = compile(context.problem)
  initial_state = State(plan, zeros(Grico.field_dof_count(context.u)))
  state = solve(plan; initial_state, maxiter=10, relative_tolerance=1.0e-10,
                absolute_tolerance=1.0e-10, linear_relative_tolerance=1.0e-10)
  error_value = l2_error(state, context.u, context.exact)
  vtk_path = nothing

  if write_output
    # The optional VTK file records the exact solution and pointwise absolute
    # error. This keeps the example useful as a visual Newton smoke test without
    # forcing file output during automated tests.
    mkpath(output_directory)
    vtk_path = Grico.write_vtk(joinpath(output_directory, "nonlinear_bratu"), state;
                               point_data=(exact=context.exact,
                                           abs_error=(x, values) -> abs(values.u - context.exact(x))),
                               field_data=(l2_error=error_value, λ=Float64(λ)),
                               sample_degree=degree, append=true, compress=true, ascii=false)
  end

  if print_summary
    println("nonlinear_bratu/driver.jl")
    @printf("  root cells        : %d x %d\n", root_counts...)
    @printf("  degree            : %d\n", degree)
    @printf("  λ                 : %.3f\n", λ)
    @printf("  scalar dofs       : %d\n", Grico.scalar_dof_count(context.space))
    @printf("  L² error          : %.6e\n", error_value)
    vtk_path === nothing || println("  vtk               : $vtk_path")
  end

  return (; context..., plan, initial_state, state, error_value, vtk_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_nonlinear_bratu_example()
end

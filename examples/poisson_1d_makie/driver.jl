using Printf
using Grico

# This optional plotting example solves
#
#   -u″ = cos(2πx)  on Ω = (0, 1),      u(0) = u(1) = 0,
#
# and renders both `u_h` and ∂ₓu_h through the Makie postprocessing extension.
# The example is intentionally one-dimensional: it keeps the finite-element
# setup close to the minimal workflow while showing how sampled point data can
# be supplied without writing a VTK file.
#
# CairoMakie is loaded lazily inside the driver so the rest of the repository can
# parse and test the example set without installing a graphics stack. Run this
# directory with `julia --project=. driver.jl` after resolving its Project.toml.

const MAKIE_CELL_COUNT = 2
const MAKIE_DEGREE = 8
const MAKIE_SAMPLE_SUBDIVISIONS = 4

struct MakiePoissonDiffusion end

# The same small operator object is used for both the bilinear form and the load
# callback below. For the bilinear form, returning `(0, ∂ₓu)` makes Grico
# accumulate `∫Ω ∂ₓv ∂ₓu dx`.
function Grico.cell_accumulate(::MakiePoissonDiffusion, q, trial, test_component)
  return TestChannels(zero(value(trial)), gradient(trial))
end

# In one dimension the manufactured load is just the scalar value multiplying
# the test function. Returning a number uses the accumulator shorthand for
# `TestChannels(f, 0)`.
Grico.cell_rhs_accumulate(::MakiePoissonDiffusion, q, test_component) = cos(2π * point(q)[1])

function _load_cairomakie!()
  # CairoMakie is an optional example dependency. Loading it lazily keeps
  # `include("driver.jl")` cheap for test environments that only verify the
  # package extension boundary.
  Base.find_package("CairoMakie") === nothing &&
    throw(ArgumentError("poisson_1d_makie requires CairoMakie in the active environment"))
  Base.eval(@__MODULE__, :(using CairoMakie))
  return CairoMakie
end

function build_poisson_1d_makie_problem(; cell_count=MAKIE_CELL_COUNT, degree=MAKIE_DEGREE)
  # A high polynomial degree on two cells gives a smooth plot with very little
  # code and demonstrates that the matrix-free path is not restricted to low
  # order discretizations.
  domain = Domain((0.0,), (1.0,), (cell_count,))
  space = HpSpace(domain, SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(degree)))
  u = ScalarField(space; name=:u)

  # The two accumulator registrations distinguish the bilinear operator from
  # the load vector even though the same Julia object provides both callback
  # methods.
  problem = AffineProblem(u; operator_class=SPD())
  add_cell_accumulator!(problem, u, u, MakiePoissonDiffusion())
  add_cell_accumulator!(problem, u, MakiePoissonDiffusion())
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, LOWER), 0.0))
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, UPPER), 0.0))

  return (; domain, space, u, problem)
end

function poisson_1d_makie_figure(state, field; sample_subdivisions=MAKIE_SAMPLE_SUBDIVISIONS,
                                 sample_degree=MAKIE_DEGREE)
  CairoMakie = _load_cairomakie!()
  figure = CairoMakie.Figure(size=(720, 620))
  solution_axis = CairoMakie.Axis(figure[1, 1]; ylabel="u",
                                  title="One-dimensional Poisson solution")
  gradient_axis = CairoMakie.Axis(figure[2, 1]; xlabel="x", ylabel="∂ₓu")
  # `plot_field!` can sample any point-data callback with access to the leaf and
  # reference coordinate `ξ`. Here we expose the derivative by evaluating the
  # already solved state in reference coordinates and converting through Grico's
  # field-gradient API.
  derivative_data = (du_dx=(_x, _values, leaf, ξ) -> field_gradient(state, field, leaf, ξ)[1],)

  Grico.plot_field!(solution_axis, state, :u; subdivisions=sample_subdivisions,
                    sample_degree=sample_degree)
  Grico.plot_field!(gradient_axis, state, :du_dx; point_data=derivative_data,
                    subdivisions=sample_subdivisions, sample_degree=sample_degree)
  CairoMakie.linkxaxes!(solution_axis, gradient_axis)
  return figure
end

function run_poisson_1d_makie_example(; cell_count=MAKIE_CELL_COUNT, degree=MAKIE_DEGREE,
                                      sample_subdivisions=MAKIE_SAMPLE_SUBDIVISIONS,
                                      sample_degree=degree,
                                      output_directory=joinpath(@__DIR__, "output"),
                                      solver=CGSolver(preconditioner=GeometricMultigridPreconditioner()),
                                      print_summary=true)
  context = build_poisson_1d_makie_problem(; cell_count, degree)
  # The default solver is intentionally explicit here: this example shows that
  # the same matrix-free GMG solve can feed interactive or file-based
  # postprocessing.
  state = solve(context.problem; solver)
  figure = poisson_1d_makie_figure(state, context.u; sample_subdivisions, sample_degree)
  CairoMakie = _load_cairomakie!()

  mkpath(output_directory)
  figure_path = joinpath(output_directory, "poisson_1d_solution.pdf")
  CairoMakie.save(figure_path, figure)

  if print_summary
    println("poisson_1d_makie/driver.jl")
    @printf("  cells             : %d\n", cell_count)
    @printf("  degree            : %d\n", degree)
    println("  figure            : $figure_path")
  end

  return (; context..., state, figure, figure_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_poisson_1d_makie_example()
end

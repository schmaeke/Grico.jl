using Grico
using CairoMakie

# This example is intentionally close to the README problem: solve
# -u'' = cos(2 π x) on [0, 1] with homogeneous Dirichlet data and render the sampled
# finite-element solution directly through the Makie postprocessing extension.

function run_poisson_1d_makie_example(; cell_count=2, degree=8, sample_subdivisions=4,
                                      sample_degree=degree,
                                      output_directory=joinpath(@__DIR__, "output"))
  domain = Domain((0.0,), (1.0,), (cell_count,))
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(degree)))
  u = ScalarField(space; name=:u)

  problem = AffineProblem(u; operator_class=SPD())
  add_cell_bilinear!(problem, u, u) do q, v, w
    grad(v)[1] * grad(w)[1]
  end
  add_cell_linear!(problem, u) do q, v
    value(v) * cos(2.0 * pi * point(q)[1])
  end
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, LOWER), 0.0))
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, UPPER), 0.0))

  plan = compile(problem)
  state = solve(plan; solver=CGSolver(preconditioner=JacobiPreconditioner()))
  figure = poisson_figure(state, u; sample_subdivisions, sample_degree)

  mkpath(output_directory)
  figure_path = joinpath(output_directory, "poisson_1d_solution.pdf")
  CairoMakie.save(figure_path, figure)
  println("poisson_1d_makie/driver.jl")
  println("  figure $figure_path")
  return (; domain, space, field=u, plan, state, figure_path)
end

function poisson_figure(state, field; sample_subdivisions, sample_degree)
  figure = Figure(size=(720, 620))
  solution_axis = Axis(figure[1, 1]; ylabel="u", title="One-dimensional Poisson solution")
  gradient_axis = Axis(figure[2, 1]; xlabel="x", ylabel="du/dx")
  derivative_data = (du_dx=(_x, _values, leaf, ξ) -> field_gradient(state, field, leaf, ξ)[1],)

  plot_field!(solution_axis, state, :u; subdivisions=sample_subdivisions,
              sample_degree=sample_degree)
  plot_field!(gradient_axis, state, :du_dx; point_data=derivative_data,
              subdivisions=sample_subdivisions, sample_degree=sample_degree)
  linkxaxes!(solution_axis, gradient_axis)
  return figure
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_poisson_1d_makie_example()
end

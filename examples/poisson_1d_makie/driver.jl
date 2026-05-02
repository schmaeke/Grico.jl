using Grico
using CairoMakie
import Grico: cell_apply!, cell_rhs!

# This example is intentionally close to the README problem: solve
# -u'' = cos(2 π x) on [0, 1] with homogeneous Dirichlet data and render the sampled
# finite-element solution directly through the Makie postprocessing extension.
struct Diffusion{F,T}
  field::F
  kappa::T
end

function cell_apply!(local_result, op::Diffusion, values::CellValues, local_coefficients)
  mode_count = local_mode_count(values, op.field)

  for q in 1:point_count(values)
    weighted_gradient = op.kappa * gradient(values, local_coefficients, op.field, q)[1] *
                        weight(values, q)

    for i in 1:mode_count
      grad_i = shape_gradient(values, op.field, q, i)
      row = local_dof_index(values, op.field, 1, i)
      local_result[row] += grad_i[1] * weighted_gradient
    end
  end

  return nothing
end

struct Source{F,G}
  field::F
  f::G
end

function cell_rhs!(local_rhs, op::Source, values::CellValues)
  b = block(local_rhs, values, op.field)
  mode_count = local_mode_count(values, op.field)

  for q in 1:point_count(values)
    w = op.f(point(values, q)) * weight(values, q)

    for i in 1:mode_count
      b[i] += shape_value(values, op.field, q, i) * w
    end
  end

  return nothing
end

function run_poisson_1d_makie_example(; cell_count=2, degree=8, sample_subdivisions=4,
                                      sample_degree=degree,
                                      output_directory=joinpath(@__DIR__, "output"))
  domain = Domain((0.0,), (1.0,), (cell_count,))
  space = HpSpace(domain, SpaceOptions(degree=UniformDegree(degree)))
  u = ScalarField(space; name=:u)

  problem = AffineProblem(u)
  add_cell!(problem, Diffusion(u, 1.0))
  add_cell!(problem, Source(u, x -> cos(2.0 * pi * x[1])))
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, LOWER), 0.0))
  add_constraint!(problem, Dirichlet(u, BoundaryFace(1, UPPER), 0.0))

  plan = compile(problem)
  state = solve(plan; preconditioner=JacobiPreconditioner())
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

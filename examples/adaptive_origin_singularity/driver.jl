using Printf
using Grico
using WriteVTK

# This example is the main adaptive finite-element tour of Grico. It solves
#
#   -Δu = f  in Ω = (0, 1)²,
#
# with manufactured solution
#
#   uₑ(x) = r(x)^α,      r(x) = ‖x‖₂,      α = 1/2.
#
# The solution is continuous but has an unbounded gradient at the origin. Away
# from that corner the field is smooth, so p-enrichment is useful; close to the
# singularity local h-refinement is the correct response. This makes the problem
# a compact demonstration of mixed hp adaptation, exact-error reporting, and
# source-to-target space transitions.
#
# Boundary conditions are imposed strongly on the upper faces `xₐ = 1`. The
# lower faces are left natural; for this radial solution ∂u/∂xₐ vanishes on
# `xₐ = 0` away from the singular corner.

const ORIGIN_DIMENSION = 2
const ORIGIN_EXPONENT = 0.5
const ORIGIN_INITIAL_DEGREE = 2
const ORIGIN_MAX_DEGREE = 4
const ORIGIN_MAX_H_LEVEL = 5
const ORIGIN_ADAPTIVITY_TOLERANCE = 5.0e-2
const ORIGIN_DEFAULT_STEPS = 6

struct OriginDiffusion end

struct OriginLoad{F}
  source::F
end

# The cell operator is the Laplace bilinear form. In accumulator notation the
# callback returns the coefficients of `v` and `∇v`; for
# `∫Ω ∇v · ∇u dΩ`, the value coefficient is zero and the gradient coefficient is
# the reconstructed trial gradient.
function Grico.cell_accumulate(::OriginDiffusion, q, trial, test_component)
  return TestChannels(zero(value(trial)), gradient(trial))
end

# The load is stored as a callable so the manufactured source can be changed
# without changing the operator registration code.
Grico.cell_rhs_accumulate(operator::OriginLoad, q, test_component) = operator.source(point(q))

function origin_solution_data(; dimension=ORIGIN_DIMENSION, exponent=ORIGIN_EXPONENT)
  # For `u = r^α` in `d` dimensions, `Δu = α(α + d - 2) r^(α-2)` away from the
  # origin. The singular point is assigned finite placeholder values because it
  # lies on the domain corner; it should not be sampled as an interior
  # quadrature point, but the guard keeps diagnostics and plotting robust.
  source_factor = -exponent * (exponent + dimension - 2)
  exact_solution = x -> begin
    radius = sqrt(sum(abs2, x))
    radius == 0.0 ? 0.0 : radius^exponent
  end
  source_term = x -> begin
    radius = sqrt(sum(abs2, x))
    radius == 0.0 ? 0.0 : source_factor * radius^(exponent - 2)
  end
  return (; dimension, exponent, source_factor, exact_solution, source_term)
end

function build_origin_singularity_context(; dimension=ORIGIN_DIMENSION,
                                          initial_degree=ORIGIN_INITIAL_DEGREE,
                                          exponent=ORIGIN_EXPONENT)
  data = origin_solution_data(; dimension, exponent)
  # The initial mesh intentionally starts with a single Cartesian root cell in
  # each coordinate direction. This keeps the first solve simple and makes the
  # subsequent h-refinement pattern around the singular corner easy to inspect.
  domain = Domain(ntuple(_ -> 0.0, dimension), ntuple(_ -> 1.0, dimension),
                  ntuple(_ -> 1, dimension))
  # A full-tensor basis exercises the tensorized matrix-free kernels on every
  # cell where the local polynomial space remains rectangular.
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(initial_degree)))
  u = ScalarField(space; name=:u)
  return (; data..., initial_degree, domain, space, u)
end

function build_origin_singularity_problem(u, context)
  problem = AffineProblem(u; operator_class=SPD())
  add_cell_accumulator!(problem, u, u, OriginDiffusion())
  add_cell_accumulator!(problem, u, OriginLoad(context.source_term))

  # Only the upper faces are prescribed strongly. The lower faces keep the
  # natural boundary condition, which is compatible with the radial manufactured
  # solution away from the singular corner.
  for axis in 1:context.dimension
    add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, UPPER), context.exact_solution))
  end

  return problem
end

function origin_adaptivity_plan(state, u; max_degree=ORIGIN_MAX_DEGREE,
                                max_h_level=ORIGIN_MAX_H_LEVEL,
                                tolerance=ORIGIN_ADAPTIVITY_TOLERANCE)
  # The automatic plan combines Grico's multiresolution indicators with hard
  # per-field limits. The limits are important in an example because they keep
  # the refinement sequence bounded and reproducible across machines.
  limits = AdaptivityLimits(field_space(u); max_p=max_degree, max_h_level=max_h_level)
  return Grico.adaptivity_plan(state, u; tolerance, limits)
end

function run_adaptive_origin_singularity_example(; adaptive_steps=ORIGIN_DEFAULT_STEPS,
                                                 max_degree=ORIGIN_MAX_DEGREE,
                                                 max_h_level=ORIGIN_MAX_H_LEVEL,
                                                 tolerance=ORIGIN_ADAPTIVITY_TOLERANCE,
                                                 solver=AutoLinearSolver(), write_output=false,
                                                 output_directory=joinpath(@__DIR__, "output"),
                                                 export_subdivisions=1, export_degree=3,
                                                 print_summary=true)
  context = build_origin_singularity_context()
  u = context.u
  history = NamedTuple[]
  vtk_files = String[]
  vtk_steps = Int[]
  final_state = nothing
  final_problem = nothing
  final_error = NaN
  pvd_path = nothing

  print_summary && println("adaptive_origin_singularity/driver.jl")
  print_summary && println("  step leaves dofs rel-L²-error plan")
  write_output && mkpath(output_directory)

  for step in 0:adaptive_steps
    # The field `u` may be replaced by an adapted field at the end of each
    # iteration, so the problem is rebuilt from the current field every step.
    # This also reattaches constraints to the active space instead of carrying
    # stale integration plans across a mesh transition.
    problem = build_origin_singularity_problem(u, context)
    state = solve(problem; solver)
    error_value = relative_l2_error(state, u, context.exact_solution)
    current_space = field_space(u)

    # The final requested step reports the solved state but does not construct
    # another plan. Earlier steps stop as soon as the indicator marks nothing,
    # which keeps the function useful as both a fixed-step demo and a tolerance
    # controlled solve.
    if step == adaptive_steps
      plan = nothing
      step_plan = "done"
      stop_now = true
    else
      plan = origin_adaptivity_plan(state, u; max_degree, max_h_level, tolerance)
      summary = adaptivity_summary(plan)
      step_plan = isempty(plan) ? "stop" :
                  "h=$(summary.h_refinement_leaf_count), p=$(summary.p_refinement_leaf_count)"
      stop_now = isempty(plan)
    end

    push!(history,
          (; step, active_leaves=active_leaf_count(current_space),
           dofs=Grico.scalar_dof_count(current_space), error_value, step_plan))

    if print_summary
      @printf("  %4d %6d %5d %.6e %s\n", step, active_leaf_count(current_space),
              Grico.scalar_dof_count(current_space), error_value, step_plan)
    end

    if write_output
      # VTK output samples both nodal point data and cell metadata. The leaf
      # level and polynomial degree fields are especially useful for checking
      # whether h-refinement concentrates near the origin while p-enrichment is
      # used on smoother cells.
      current_grid = grid(current_space)
      vtk_path = Grico.write_vtk(joinpath(output_directory,
                                          @sprintf("adaptive_origin_singularity_%04d", step)),
                                 state;
                                 point_data=(exact=context.exact_solution,
                                             abs_error=(x, values) -> abs(values.u -
                                                                          context.exact_solution(x))),
                                 cell_data=(leaf=leaf -> Float64(leaf),
                                            level=leaf -> Float64.(Grico.level(current_grid, leaf)),
                                            degree=leaf -> Float64.(Grico.cell_degrees(current_space,
                                                                                       leaf))),
                                 field_data=(step=Float64(step), relative_l2_error=error_value),
                                 subdivisions=export_subdivisions, sample_degree=export_degree,
                                 append=true, compress=true, ascii=false)
      push!(vtk_files, vtk_path)
      push!(vtk_steps, step)
    end

    final_problem = problem
    final_state = state
    final_error = error_value

    if stop_now
      if write_output && !isempty(vtk_files)
        pvd_path = Grico.write_pvd(joinpath(output_directory, "adaptive_origin_singularity.pvd"),
                                   vtk_files; timesteps=vtk_steps)
      end
      break
    end

    # `transition(plan)` contains the source-to-target space map. Constructing
    # the adapted field through this object preserves field identity while
    # moving the next solve to the refined hp space.
    u = adapted_field(Grico.transition(plan), u)
  end

  return (; context..., u, final_problem, final_state, final_error, history, pvd_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_adaptive_origin_singularity_example()
end

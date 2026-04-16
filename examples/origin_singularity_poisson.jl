using Printf
using Grico
import Grico: cell_matrix!, cell_rhs!

# This example is meant to be the simplest "read it from top to bottom" tour of
# adaptive finite elements in Grico.
#
# We solve a Poisson problem on the unit hypercube,
#
#   Ω = [0, 1]ᵈ,
#
# with a known exact solution
#
#   u(x) = r(x)^α,   r(x) = ‖x‖,
#
# where `α = 0.5` in the default configuration. This function is continuous, but
# its gradient blows up at the origin. Numerically that means:
#
# - the solution is smooth away from the origin, so high-order polynomials are
#   attractive there,
# - but near the origin the singular corner is better resolved by local mesh
#   refinement.
#
# This combination makes the problem a classical benchmark for adaptive
# h/p finite-element methods.
#
# The source term is chosen so that
#
#   -Δu = f,   f(r) = -α (α + d - 2) r^(α - 2),
#
# for `r > 0`. At the origin we define both `u` and `f` by their limiting
# finite values used in the discrete code path.
#
# Boundary conditions are mixed:
#
# - on the upper faces `x_a = 1`, the exact Dirichlet trace is imposed;
# - on the lower faces `x_a = 0`, no boundary operator is added, so the weak
#   form uses the natural Neumann condition.
#
# For the radial exact solution this is consistent because ∂u/∂x_a = 0 whenever
# `x_a = 0`.
#
# File roadmap:
#
# 1. choose the model parameters,
# 2. define the local cell operators,
# 3. define the exact solution and source term,
# 4. build a very coarse initial hp space,
# 5. solve, estimate, adapt, and visualize.

# The code is written dimension-independently, but the default is now 2D so a
# new reader can inspect the VTK output directly.
const DIMENSION = 2
const INITIAL_DEGREE = 2
const ADAPTIVE_STEPS = 20

# Single-tolerance adaptivity controls. The planner marks modal detail with one
# tolerance, uses its internal modal-decay classifier for the h/p split, and
# respects the explicit degree and h-level limits below.
const ADAPTIVITY_TOLERANCE = 5.0e-2
const MAX_DEGREE = 4
const MAX_H_LEVEL = 5

# With the default 2D configuration, VTK output is produced after every solve.
# The guard remains dimension-aware so the same file still works if one changes
# `DIMENSION` manually.
const WRITE_VTK = DIMENSION <= 3
const EXPORT_SUBDIVISIONS = 1
# Benchmarks may include this file for its reusable builders without running
# the full adaptive loop. Direct execution keeps the default autorun behavior.
const RUN_ORIGIN_SINGULARITY_POISSON = get(ENV, "GRICO_ORIGIN_SINGULARITY_AUTORUN", "1") == "1"

# Singularity strength `α` in u = r^α.
const SINGULAR_EXPONENT = 0.5

# Optional quadrature enrichment for the verification integral.
const VERIFICATION_EXTRA_POINTS = 0

# ---------------------------------------------------------------------------
# 1. Local weak-form building blocks
# ---------------------------------------------------------------------------
#
# Grico examples define variational forms through small local operators. For
# Poisson we need the usual diffusion bilinear form and the volume load term.

# Standard diffusion bilinear form
#
#   a(v, u) = ∫_Ω ∇v · ∇u dΩ.
#
# The local matrix is symmetric, so the implementation assembles only the lower
# triangle and mirrors it.
struct Diffusion{F}
  field::F
end

function cell_matrix!(local_matrix, operator::Diffusion, values::CellValues)
  local_block = block(local_matrix, values, operator.field, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      for col_mode in 1:row_mode
        contribution = zero(eltype(local_matrix))

        for axis in 1:axis_count
          contribution += gradients[axis, row_mode, point_index] *
                          gradients[axis, col_mode, point_index]
        end

        contribution *= weighted
        local_block[row_mode, col_mode] += contribution
        row_mode == col_mode || (local_block[col_mode, row_mode] += contribution)
      end
    end
  end

  return nothing
end

# Load functional
#
#   ℓ(v) = ∫_Ω f v dΩ.
#
# The structure is the same: loop over quadrature points, evaluate the physical
# coefficient data there, and accumulate into the local vector with the test
# basis values.
struct Source{F,G}
  field::F
  data::G
end

function cell_rhs!(local_rhs, operator::Source, values::CellValues)
  local_block = block(local_rhs, values, operator.field)
  shape_table = shape_values(values, operator.field)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = operator.data(point(values, point_index)) * weight(values, point_index)

    for mode_index in 1:mode_count
      local_block[mode_index] += shape_table[mode_index, point_index] * weighted
    end
  end

  return nothing
end

# ---------------------------------------------------------------------------
# 2. Exact solution and matching source term
# ---------------------------------------------------------------------------
#
# Because the exact solution is known, we can both prescribe exact boundary
# data and measure the true discretization error after each adaptive step.
#
# The prefactor
#
#   -α (α + d - 2)
#
# is the Laplacian coefficient for r^α in dimension d. The example exposes this
# manufactured data through a helper so both the direct example driver and the
# benchmark harness use the exact same problem definition.
function origin_solution_data(; dimension=DIMENSION, singular_exponent=SINGULAR_EXPONENT)
  source_factor = -singular_exponent * (singular_exponent + dimension - 2)
  exact_solution = x -> (r=sqrt(sum(abs2, x)); r == 0.0 ? 0.0 : r^singular_exponent)
  source_term = x -> (r=sqrt(sum(abs2, x));
                      r == 0.0 ? 0.0 : source_factor * r^(singular_exponent - 2))
  return (; dimension, singular_exponent, source_factor, exact_solution, source_term)
end

# Build the reusable field descriptor and manufactured data used by the example
# driver and by the validation benchmarks.
function build_origin_singularity_poisson_context(; dimension=DIMENSION,
                                                  initial_degree=INITIAL_DEGREE,
                                                  singular_exponent=SINGULAR_EXPONENT)
  manufactured = origin_solution_data(; dimension, singular_exponent)
  space = HpSpace(Domain(ntuple(_ -> 0.0, dimension), ntuple(_ -> 1.0, dimension),
                         ntuple(_ -> 1, dimension)),
                  SpaceOptions(degree=UniformDegree(initial_degree)))
  u = ScalarField(space; name=:u)
  return (; manufactured..., initial_degree, space, u)
end

function build_origin_singularity_problem(u, context)
  problem = AffineProblem(u)
  add_cell!(problem, Diffusion(u))
  add_cell!(problem, Source(u, context.source_term))

  for axis in 1:context.dimension
    add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, UPPER), context.exact_solution))
  end

  return problem
end

function origin_adaptivity_plan(state, u)
  limits = AdaptivityLimits(field_space(u); max_p=MAX_DEGREE, max_h_level=MAX_H_LEVEL)
  return adaptivity_plan(state, u; tolerance=ADAPTIVITY_TOLERANCE, limits=limits)
end

# Human-facing adaptive driver used both when the example is run directly and
# when benchmarks need the same solve/adapt loop without VTK output.
function run_origin_singularity_poisson_example(; adaptive_steps=ADAPTIVE_STEPS,
                                                write_vtk=WRITE_VTK, print_summary=true)
  context = build_origin_singularity_poisson_context()
  u = context.u
  history = NamedTuple[]
  vtk_files = String[]
  vtk_steps = Int[]
  vtk_path = nothing
  pvd_path = nothing
  final_plan = nothing
  final_state = nothing
  final_error = NaN

  if print_summary
    println("origin_singularity_poisson.jl")
    println("  step leaves dofs rel-l2-error plan")
  end

  output_directory = joinpath(@__DIR__, "output")
  write_vtk && mkpath(output_directory)

  for step in 0:adaptive_steps
    problem = build_origin_singularity_problem(u, context)
    plan = compile(problem)
    state = State(plan, solve(assemble(plan)))
    error_value = relative_l2_error(state, u, context.exact_solution; plan=plan,
                                    extra_points=VERIFICATION_EXTRA_POINTS)

    if step == adaptive_steps
      step_plan = "done"
      stop_now = true
      adaptivity_plan = nothing
    else
      adaptivity_plan = origin_adaptivity_plan(state, u)
      summary = adaptivity_summary(adaptivity_plan)
      step_plan = isempty(adaptivity_plan) ? "stop" :
                  "h=$(summary.h_refinement_leaf_count), p=$(summary.p_refinement_leaf_count)"
      stop_now = isempty(adaptivity_plan)
    end

    push!(history,
          (; step, active_leaves=active_leaf_count(field_space(u)),
           dofs=scalar_dof_count(field_space(u)), error_value, step_plan))
    print_summary && @printf("  %4d %6d %4d %.6e %s\n", step, active_leaf_count(field_space(u)),
                            scalar_dof_count(field_space(u)), error_value, step_plan)

    if write_vtk
      current_space = field_space(u)
      current_grid = grid(current_space)
      vtk_path = write_vtk(joinpath(output_directory,
                                    @sprintf("origin_singularity_poisson_%04d", step)),
                           current_space.domain; state=state,
                           point_data=(exact=context.exact_solution,
                                       abs_error=(x, values) -> abs(values.u -
                                                                    context.exact_solution(x))),
                           cell_data=(leaf=leaf -> Float64(leaf),
                                      level=leaf -> Float64.(level(current_grid, leaf)),
                                      degree=leaf -> Float64.(cell_degrees(current_space, leaf))),
                           field_data=(step=Float64(step), relative_l2_error=error_value),
                           subdivisions=EXPORT_SUBDIVISIONS, append=true, compress=true,
                           ascii=false)
      push!(vtk_files, vtk_path)
      push!(vtk_steps, step)
    end

    final_plan = plan
    final_state = state
    final_error = error_value

    if stop_now
      if write_vtk && !isempty(vtk_files)
        pvd_path = write_pvd(joinpath(output_directory, "origin_singularity_poisson.pvd"),
                             vtk_files; timesteps=vtk_steps)
        print_summary && println("  vtk  $vtk_path")
        print_summary && println("  pvd  $pvd_path")
      end
      break
    end

    space_transition = transition(adaptivity_plan)
    u = adapted_field(space_transition, u)
  end

  return (; context..., u, final_plan, final_state, final_error, history, vtk_path, pvd_path)
end

RUN_ORIGIN_SINGULARITY_POISSON && run_origin_singularity_poisson_example()

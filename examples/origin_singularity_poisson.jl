using Printf
using Grico
import Grico: cell_matrix!, cell_rhs!

# This example is meant to be the simplest "read it from top to bottom" tour of
# adaptive `hp` finite elements in Grico.
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
# This combination makes the problem a classical benchmark for adaptive `hp`
# methods.
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

# Dörfler bulk-marking parameter for refinement and the modal smoothness
# threshold that decides whether a marked axis should prefer `p`- or
# `h`-adaptation.
const MARK_THRESHOLD = 0.9
const SMOOTHNESS_THRESHOLD = 0.3

# With the default 2D configuration, VTK output is produced after every solve.
# The guard remains dimension-aware so the same file still works if one changes
# `DIMENSION` manually.
const WRITE_VTK = DIMENSION <= 3
const EXPORT_SUBDIVISIONS = 1

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
# is the Laplacian coefficient for r^α in dimension d.
radius(x) = sqrt(sum(abs2, x))
source_factor = -SINGULAR_EXPONENT * (SINGULAR_EXPONENT + DIMENSION - 2)
exact_solution = x -> (r=radius(x); r == 0.0 ? 0.0 : r^SINGULAR_EXPONENT)
source_term = x -> (r=radius(x); r == 0.0 ? 0.0 : source_factor * r^(SINGULAR_EXPONENT - 2))

# ---------------------------------------------------------------------------
# 3. Initial hp space
# ---------------------------------------------------------------------------
#
# We start from one single Cartesian root cell with a moderate polynomial
# degree. This is deliberately very coarse. The point is to let the adaptivity
# algorithm decide where geometric refinement and where polynomial enrichment
# are needed.
space = HpSpace(Domain(ntuple(_ -> 0.0, DIMENSION), ntuple(_ -> 1.0, DIMENSION),
                       ntuple(_ -> 1, DIMENSION)),
                SpaceOptions(degree=UniformDegree(INITIAL_DEGREE)))

println("origin_singularity_poisson.jl")
println("  step leaves dofs rel-l2-error plan")

let u = ScalarField(space; name=:u)
  output_directory = joinpath(@__DIR__, "output")
  mkpath(output_directory)
  vtk_files = String[]
  vtk_steps = Int[]

  # Each loop iteration solves the problem on the current hp space, measures
  # the exact error, asks the built-in planner for the next target space, and
  # then recreates the field on that target space.
  for step in 0:ADAPTIVE_STEPS
    # Assemble the current Poisson problem on the present hp space:
    #
    #   find u_h in V_h such that
    #     ∫_Ω ∇v_h · ∇u_h dΩ = ∫_Ω v_h f dΩ
    #
    # for all test functions v_h, together with Dirichlet data on the upper
    # faces.
    problem = AffineProblem(u)
    add_cell!(problem, Diffusion(u))
    add_cell!(problem, Source(u, source_term))

    # Dirichlet data are imposed only on the upper faces. The lower faces are
    # left natural, which matches the exact radial solution as discussed above.
    for axis in 1:DIMENSION
      add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, UPPER), exact_solution))
    end

    plan = compile(problem)

    # For this small scalar example the default sparse direct solve is a good
    # fit and keeps the script short.
    state = State(plan, solve(assemble(plan)))

    # Compare the discrete solution against the exact one in the relative
    # `L²` norm. This is not an estimator; it is a true verification quantity
    # available only because we chose a manufactured solution.
    error_value = relative_l2_error(state, u, exact_solution; plan=plan,
                                    extra_points=VERIFICATION_EXTRA_POINTS,)

    if step == ADAPTIVE_STEPS
      step_plan = "done"
      stop_now = true
    else
      # The built-in `hp` planner combines:
      #
      # - a refinement indicator, which says "this region looks underresolved",
      # - and a smoothness indicator, which says whether `p` or `h` is the
      #   better next move on the marked leaves.
      #
      # Very roughly:
      #
      # - smooth region  -> favor `p`,
      # - singular region -> favor `h`.
      adaptivity_plan = hp_adaptivity_plan(state, u; threshold=MARK_THRESHOLD,
                                           smoothness_threshold=SMOOTHNESS_THRESHOLD,)
      summary = adaptivity_summary(adaptivity_plan)
      step_plan = isempty(adaptivity_plan) ? "stop" :
                  "h=$(summary.h_refinement_leaf_count), p=$(summary.p_refinement_leaf_count)"
      stop_now = isempty(adaptivity_plan)
    end

    @printf("  %4d %6d %4d %.6e %s\n", step, active_leaf_count(field_space(u)),
            scalar_dof_count(field_space(u)), error_value, step_plan,)

    # Export every adaptive step so one can inspect how the mesh and polynomial
    # degree evolve. The point data show both the exact solution and the
    # pointwise absolute error. The cell data show the local refinement level
    # and polynomial degree on each active leaf.
    if WRITE_VTK
      current_space = field_space(u)
      current_grid = grid(current_space)
      vtk_path = write_vtk(joinpath(output_directory,
                                    @sprintf("origin_singularity_poisson_%04d", step)),
                           current_space.domain; state=state,
                           point_data=(exact=exact_solution,
                                       abs_error=(x, values) -> abs(values.u - exact_solution(x))),
                           cell_data=(leaf=leaf -> Float64(leaf),
                                      level=leaf -> Float64.(level(current_grid, leaf)),
                                      degree=leaf -> Float64.(cell_degrees(current_space, leaf))),
                           field_data=(step=Float64(step), relative_l2_error=error_value),
                           subdivisions=EXPORT_SUBDIVISIONS, append=true, compress=true,
                           ascii=false,)
      push!(vtk_files, vtk_path)
      push!(vtk_steps, step)
    end

    if stop_now
      if WRITE_VTK
        vtk_path = vtk_files[end]
        pvd_path = write_pvd(joinpath(output_directory, "origin_singularity_poisson.pvd"),
                             vtk_files; timesteps=vtk_steps,)
        println("  vtk  $vtk_path")
        println("  pvd  $pvd_path")
      end

      break
    end

    # This example resolves each adapted problem from scratch rather than
    # transferring the previous discrete state. The space transition is still
    # needed, because it knows how to recreate the field descriptor on the new
    # hp space produced by the adaptivity plan. In larger nonlinear problems one
    # would often transfer the old state as an initial guess; here we keep the
    # example focused on the space-adaptation mechanism itself.
    space_transition = transition(adaptivity_plan)
    u = adapted_field(space_transition, u)
  end
end

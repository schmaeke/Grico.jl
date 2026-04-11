using Printf
using Grico
import Grico: cell_matrix!, cell_rhs!

# This example studies a classical singular Poisson solution on the unit
# hypercube,
#
#   Ω = [0, 1]ᵈ,
#
# with exact solution
#
#   u(x) = r(x)^α,   r(x) = ‖x‖,
#
# where `α = 0.5` in the default configuration. The solution is continuous but
# has a singular gradient at the origin, so it is a standard benchmark for
# adaptive `hp` methods: away from the origin the solution is smooth and favors
# `p`-refinement, while near the singularity geometric `h`-refinement is more
# effective.
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

const DIMENSION = 4
const INITIAL_DEGREE = 2
const ADAPTIVE_STEPS = 20

# Dörfler bulk-marking parameter for refinement and the modal smoothness
# threshold that decides whether a marked axis should prefer `p`- or
# `h`-adaptation.
const MARK_THRESHOLD = 0.9
const SMOOTHNESS_THRESHOLD = 0.3

# VTK output is only attempted in spatial dimensions up to three, because the
# current exporter supports only 1D/2D/3D domains.
const EXPORT_SUBDIVISIONS = 1

# Singularity strength `α` in u = r^α.
const SINGULAR_EXPONENT = 0.5

# Optional quadrature enrichment for the verification integral.
const VERIFICATION_EXTRA_POINTS = 0

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

# Radial exact solution and matching source term. The prefactor
#
#   -α (α + d - 2)
#
# is the Laplacian coefficient for r^α in dimension d.
radius(x) = sqrt(sum(abs2, x))
source_factor = -SINGULAR_EXPONENT * (SINGULAR_EXPONENT + DIMENSION - 2)
exact_solution = x -> (r=radius(x); r == 0.0 ? 0.0 : r^SINGULAR_EXPONENT)
source_term = x -> (r=radius(x); r == 0.0 ? 0.0 : source_factor * r^(SINGULAR_EXPONENT - 2))

# Start from a single root cell with uniform degree. This deliberately coarse
# initial space makes the later `hp` decisions easy to see.
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

  for step in 0:ADAPTIVE_STEPS
    # Assemble the current Poisson problem on the present hp space.
    problem = AffineProblem(u)
    add_cell!(problem, Diffusion(u))
    add_cell!(problem, Source(u, source_term))

    # Dirichlet data are imposed only on the upper faces. The lower faces are
    # left natural, which matches the exact radial solution as discussed above.
    for axis in 1:DIMENSION
      add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, UPPER), exact_solution))
    end

    plan = compile(problem)
    state = State(plan, solve(assemble(plan)))
    error_value = relative_l2_error(state, u, exact_solution; plan=plan,
                                    extra_points=VERIFICATION_EXTRA_POINTS,)

    if step == ADAPTIVE_STEPS
      step_plan = "done"
      stop_now = true
    else
      # The built-in `hp` planner combines a refinement indicator with a modal
      # smoothness indicator. In smooth regions it tends to raise polynomial
      # degree, while near the singular corner it prefers geometric refinement.
      adaptivity_plan = hp_adaptivity_plan(state, u; threshold=MARK_THRESHOLD,
                                           smoothness_threshold=SMOOTHNESS_THRESHOLD,)
      summary = adaptivity_summary(adaptivity_plan)
      step_plan = isempty(adaptivity_plan) ? "stop" :
                  "h=$(summary.h_refinement_leaf_count), p=$(summary.p_refinement_leaf_count)"
      stop_now = isempty(adaptivity_plan)
    end

    @printf("  %4d %6d %4d %.6e %s\n", step, active_leaf_count(field_space(u)),
            scalar_dof_count(field_space(u)), error_value, step_plan,)

    # VTK export is skipped in dimension four and higher because the current
    # VTK writer only handles 1D/2D/3D domains.
    if DIMENSION < 4
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
      if DIMENSION < 4
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
    # needed to rebuild the field on the target hp space produced by the
    # adaptivity plan.
    space_transition = transition(adaptivity_plan)
    u = adapted_field(space_transition, u)
  end
end

using Printf
using Grico
import Grico: cell_matrix!, cell_rhs!, h_adaptation_axes, p_degree_change, source_leaves,
              stored_cell_count, target_space

# This example solves a Poisson problem whose exact solution contains a steep
# but smooth radial layer around a prescribed shock radius,
#
#   u(r) = atan(α (r - R)) + atan(α R).
#
# The additive offset makes u(0) = 0. The parameter `α` controls the layer
# sharpness: larger values produce a narrower transition around `r = R`. This
# type of solution is a useful `hp` benchmark because the layer is still smooth,
# so the method can trade off geometric refinement against increased polynomial
# degree instead of dealing with a true discontinuity.
#
# As in the origin-singularity example, the problem is posed on the unit
# hypercube with Dirichlet conditions on the upper faces `x_a = 1` and natural
# Neumann conditions on the lower faces `x_a = 0`. For this radial exact
# solution the lower-face normal derivative vanishes because ∂u/∂x_a is
# proportional to x_a.

const DIMENSION = 3
const ROOT_COUNTS = ntuple(_ -> 5, DIMENSION)
const INITIAL_DEGREE = 2
const ADAPTIVE_STEPS = 10

# `threshold` drives Dörfler bulk marking, while `smoothness_threshold` steers
# the split between `h` and `p` on the marked axes. The additional
# `H_COARSENING_THRESHOLD` activates projection-based geometric derefinement when
# previously refined regions no longer need fine cells.
const MARK_THRESHOLD = 0.9
const SMOOTHNESS_THRESHOLD = 0.05
const H_COARSENING_THRESHOLD = 25.0e-2

const WRITE_VTK = false
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 4

# Layer parameters in u(r) = atan(α (r - R)) + atan(α R).
const ALPHA = 80.0
const SHOCK_RADIUS = 0.9

const VERIFICATION_EXTRA_POINTS = 0

# Standard Laplace bilinear form
#
#   a(v, u) = ∫_Ω ∇v · ∇u dΩ.
#
# The local matrix is symmetric, so only one triangle is assembled explicitly.
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
#   ℓ(v) = ∫_Ω f v dΩ,
#
# where f is derived analytically from the radial Laplacian of the exact
# arctangent profile.
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

radius(x) = sqrt(sum(abs2, x))

# The offset makes u(0) = 0. The source term comes from the radial identity
#
#   Δu(r) = u''(r) + (d - 1) r⁻¹ u'(r),
#
# applied to the arctangent profile.
exact_offset = atan(ALPHA * SHOCK_RADIUS)
exact_solution = x -> atan(ALPHA * (radius(x) - SHOCK_RADIUS)) + exact_offset
source_term = x -> begin
  r = radius(x)
  r == 0.0 && return 0.0
  shift = r - SHOCK_RADIUS
  numerator = ALPHA * (2 * ALPHA^2 * SHOCK_RADIUS * shift - (DIMENSION - 1) -
                       ALPHA^2 * (DIMENSION - 3) * shift^2)
  denominator = r * (1 + ALPHA^2 * shift^2)^2
  return numerator / denominator
end

# Start from a moderately coarse Cartesian background grid with uniform degree.
space = HpSpace(Domain(ntuple(_ -> 0.0, DIMENSION), ntuple(_ -> 1.0, DIMENSION), ROOT_COUNTS),
                SpaceOptions(degree=UniformDegree(INITIAL_DEGREE)))

println("shock_problem.jl")
println("  linear solver: default")
println("  step leaves dofs rel-l2-error plan")

let u = ScalarField(space; name=:u)
  output_directory = joinpath(@__DIR__, "output")
  vtk_files = String[]
  vtk_steps = Int[]

  WRITE_VTK && mkpath(output_directory)

  for step in 0:ADAPTIVE_STEPS
    problem = AffineProblem(u)
    add_cell!(problem, Diffusion(u))
    add_cell!(problem, Source(u, source_term))

    # Dirichlet data are prescribed on the upper faces only; the lower faces
    # remain natural and are satisfied exactly by the radial profile.
    for axis in 1:DIMENSION
      add_constraint!(problem, Dirichlet(u, BoundaryFace(axis, UPPER), exact_solution))
    end

    plan = compile(problem)
    state = State(plan, solve(assemble(plan)))
    error_value = relative_l2_error(state, u, exact_solution;
                                    extra_points=VERIFICATION_EXTRA_POINTS,)

    current_space = field_space(u)
    current_grid = grid(current_space)

    # These arrays are purely diagnostic. They are exported to VTK so that one
    # can visualize where the planner requested `h` refinement/derefinement and
    # where it requested `p` refinement/derefinement.
    h_adaptation = zeros(Float64, stored_cell_count(current_grid))
    p_adaptation = zeros(Float64, stored_cell_count(current_grid))
    space_transition = nothing

    if step == ADAPTIVE_STEPS
      step_plan = "done"
      stop_now = true
    else
      adaptivity_plan = hp_adaptivity_plan(state, u; threshold=MARK_THRESHOLD,
                                           smoothness_threshold=SMOOTHNESS_THRESHOLD,
                                           h_coarsening_threshold=H_COARSENING_THRESHOLD,)
      summary = adaptivity_summary(adaptivity_plan)
      stop_now = isempty(adaptivity_plan)
      space_transition = stop_now ? nothing : transition(adaptivity_plan)

      for leaf in active_leaves(current_space)
        any(h_adaptation_axes(adaptivity_plan, leaf)) && (h_adaptation[leaf] = 1.0)

        degree_change = p_degree_change(adaptivity_plan, leaf)

        if any(>(0), degree_change)
          p_adaptation[leaf] = 1.0
        elseif any(<(0), degree_change)
          p_adaptation[leaf] = -1.0
        end
      end

      if !isnothing(space_transition)
        # If several source leaves map to one target leaf, the planner has
        # performed `h`-derefinement. We mark the contributing source leaves with
        # `-1` so this becomes visible in post-processing.
        for target_leaf in active_leaves(target_space(space_transition))
          sources = source_leaves(space_transition, target_leaf)
          length(sources) > 1 || continue

          for source_leaf in sources
            h_adaptation[source_leaf] = -1.0
          end
        end
      end

      step_plan = isempty(adaptivity_plan) ? "stop" :
                  @sprintf("marked=%d, h+=%d, h-=%d, p+=%d, p-=%d", summary.marked_leaf_count,
                           summary.h_refinement_leaf_count, summary.h_derefinement_cell_count,
                           summary.p_refinement_leaf_count, summary.p_derefinement_leaf_count,)
    end

    @printf("  %4d %6d %4d %.6e %s\n", step, active_leaf_count(current_space),
            scalar_dof_count(current_space), error_value, step_plan,)

    if WRITE_VTK
      vtk_path = write_vtk(joinpath(output_directory, @sprintf("shock_problem_%04d", step)), state;
                           point_data=(exact=exact_solution,
                                       abs_error=(x, values) -> abs(values.u - exact_solution(x))),
                           cell_data=(leaf=leaf -> Float64(leaf),
                                      level=leaf -> Float64.(level(current_grid, leaf)),
                                      degree=leaf -> Float64.(cell_degrees(current_space, leaf)),
                                      h_adaptation=leaf -> h_adaptation[leaf],
                                      p_adaptation=leaf -> p_adaptation[leaf]),
                           field_data=(step=Float64(step), relative_l2_error=error_value),
                           subdivisions=EXPORT_SUBDIVISIONS, export_degree=EXPORT_DEGREE,
                           append=true, compress=true, ascii=false,)
      push!(vtk_files, vtk_path)
      push!(vtk_steps, step)
    end

    if stop_now
      if WRITE_VTK
        vtk_path = vtk_files[end]
        pvd_path = write_pvd(joinpath(output_directory, "shock_problem.pvd"), vtk_files;
                             timesteps=vtk_steps,)
        println("  vtk  $vtk_path")
        println("  pvd  $pvd_path")
      end
      break
    end

    # As in the origin-singularity example, we rebuild the field on the target
    # space but resolve the next discrete system from scratch rather than
    # transferring the previous state.
    space_transition = something(space_transition, transition(adaptivity_plan))
    u = adapted_field(space_transition, u)
  end
end

module SharedMemoryPhase0Cases

using LinearAlgebra
using SparseArrays
using Grico

import Grico: cell_matrix!, cell_residual!, cell_rhs!, cell_tangent!, face_residual!,
              face_tangent!, interface_matrix!, interface_residual!, interface_tangent!

export OperationSpec, PreparedCase, build_phase0_case, phase0_case_ids

const DIRECT_LINEAR_SOLVE = (matrix, rhs) -> matrix \ rhs

const ADAPTIVITY_THRESHOLD = 0.35
const SMOOTHNESS_THRESHOLD = 0.45
const SINGULAR_EXPONENT = 0.5
const SOURCE_FACTOR = -SINGULAR_EXPONENT * (SINGULAR_EXPONENT + 2 - 2)

struct OperationSpec
  id::String
  label::String
  run::Function
  setup::Function
end

struct PreparedCase
  id::String
  label::String
  description::String
  metadata::Dict{String,Any}
  operations::Vector{OperationSpec}
end

phase0_case_ids() = ("affine_cell_diffusion", "affine_interface_dg", "nonlinear_interface_dg",
                     "adaptive_poisson")

struct Diffusion{F,T}
  field::F
  coefficient::T
end

function cell_matrix!(local_matrix, operator::Diffusion, values::CellValues)
  block = Grico.block(local_matrix, values, operator.field, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  @inbounds for point_index in 1:Grico.point_count(values)
    weighted = operator.coefficient * Grico.weight(values, point_index)

    for row_mode in 1:mode_count
      grad_row = Grico.shape_gradient(values, operator.field, point_index, row_mode)

      for col_mode in 1:mode_count
        grad_col = Grico.shape_gradient(values, operator.field, point_index, col_mode)
        block[row_mode, col_mode] += sum(grad_row[axis] * grad_col[axis]
                                         for axis in eachindex(grad_row)) * weighted
      end
    end
  end

  return nothing
end

struct Source{F,G}
  field::F
  f::G
end

function cell_rhs!(local_rhs, operator::Source, values::CellValues)
  block = Grico.block(local_rhs, values, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  @inbounds for point_index in 1:Grico.point_count(values)
    weighted = operator.f(Grico.point(values, point_index)) * Grico.weight(values, point_index)

    for mode_index in 1:mode_count
      block[mode_index] += Grico.shape_value(values, operator.field, point_index, mode_index) *
                           weighted
    end
  end

  return nothing
end

struct MassCoupling{F,G,T}
  test_field::F
  trial_field::G
  coefficient::T
end

function cell_matrix!(local_matrix, operator::MassCoupling, values::CellValues)
  block = Grico.block(local_matrix, values, operator.test_field, operator.trial_field)
  test_modes = Grico.local_mode_count(values, operator.test_field)
  trial_modes = Grico.local_mode_count(values, operator.trial_field)

  @inbounds for point_index in 1:Grico.point_count(values)
    weighted = operator.coefficient * Grico.weight(values, point_index)

    for row_mode in 1:test_modes
      shape_row = Grico.shape_value(values, operator.test_field, point_index, row_mode)

      for col_mode in 1:trial_modes
        shape_col = Grico.shape_value(values, operator.trial_field, point_index, col_mode)
        block[row_mode, col_mode] += shape_row * shape_col * weighted
      end
    end
  end

  return nothing
end

struct GradientJumpPenalty{F,T}
  field::F
  coefficient::T
end

function interface_matrix!(local_matrix, operator::GradientJumpPenalty, values::InterfaceValues)
  minus_values = Grico.minus(values)
  plus_values = Grico.plus(values)
  minus_minus = Grico.block(local_matrix, minus_values, operator.field, minus_values,
                            operator.field)
  minus_plus = Grico.block(local_matrix, minus_values, operator.field, plus_values,
                           operator.field)
  plus_minus = Grico.block(local_matrix, plus_values, operator.field, minus_values,
                           operator.field)
  plus_plus = Grico.block(local_matrix, plus_values, operator.field, plus_values,
                          operator.field)
  minus_modes = Grico.local_mode_count(minus_values, operator.field)
  plus_modes = Grico.local_mode_count(plus_values, operator.field)

  @inbounds for point_index in 1:Grico.point_count(values)
    weighted = operator.coefficient * Grico.weight(values, point_index)

    for row_mode in 1:minus_modes
      row_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                 row_mode)

      for col_mode in 1:minus_modes
        col_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                   col_mode)
        minus_minus[row_mode, col_mode] += row_gradient * col_gradient * weighted
      end

      for col_mode in 1:plus_modes
        col_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                   col_mode)
        minus_plus[row_mode, col_mode] -= row_gradient * col_gradient * weighted
      end
    end

    for row_mode in 1:plus_modes
      row_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                 row_mode)

      for col_mode in 1:minus_modes
        col_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                   col_mode)
        plus_minus[row_mode, col_mode] -= row_gradient * col_gradient * weighted
      end

      for col_mode in 1:plus_modes
        col_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                   col_mode)
        plus_plus[row_mode, col_mode] += row_gradient * col_gradient * weighted
      end
    end
  end

  return nothing
end

struct NonlinearReaction{F,T}
  field::F
  source::T
end

function cell_residual!(local_rhs, operator::NonlinearReaction, values::CellValues,
                        state::State)
  block = Grico.block(local_rhs, values, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  @inbounds for point_index in 1:Grico.point_count(values)
    u = Grico.value(values, state, operator.field, point_index)
    weighted = (u^2 - operator.source) * Grico.weight(values, point_index)

    for mode_index in 1:mode_count
      block[mode_index] += Grico.shape_value(values, operator.field, point_index, mode_index) *
                           weighted
    end
  end

  return nothing
end

function cell_tangent!(local_matrix, operator::NonlinearReaction, values::CellValues,
                       state::State)
  block = Grico.block(local_matrix, values, operator.field, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  @inbounds for point_index in 1:Grico.point_count(values)
    u = Grico.value(values, state, operator.field, point_index)
    weighted = 2.0 * u * Grico.weight(values, point_index)

    for row_mode in 1:mode_count
      shape_row = Grico.shape_value(values, operator.field, point_index, row_mode)

      for col_mode in 1:mode_count
        shape_col = Grico.shape_value(values, operator.field, point_index, col_mode)
        block[row_mode, col_mode] += shape_row * shape_col * weighted
      end
    end
  end

  return nothing
end

struct NonlinearBoundaryReaction{F,T}
  field::F
  target::T
end

function face_residual!(local_rhs, operator::NonlinearBoundaryReaction, values::FaceValues,
                        state::State)
  block = Grico.block(local_rhs, values, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  @inbounds for point_index in 1:Grico.point_count(values)
    u = Grico.value(values, state, operator.field, point_index)
    weighted = (u^2 - operator.target) * Grico.weight(values, point_index)

    for mode_index in 1:mode_count
      block[mode_index] += Grico.shape_value(values, operator.field, point_index, mode_index) *
                           weighted
    end
  end

  return nothing
end

function face_tangent!(local_matrix, operator::NonlinearBoundaryReaction, values::FaceValues,
                       state::State)
  block = Grico.block(local_matrix, values, operator.field, operator.field)
  mode_count = Grico.local_mode_count(values, operator.field)

  @inbounds for point_index in 1:Grico.point_count(values)
    u = Grico.value(values, state, operator.field, point_index)
    weighted = 2.0 * u * Grico.weight(values, point_index)

    for row_mode in 1:mode_count
      shape_row = Grico.shape_value(values, operator.field, point_index, row_mode)

      for col_mode in 1:mode_count
        shape_col = Grico.shape_value(values, operator.field, point_index, col_mode)
        block[row_mode, col_mode] += shape_row * shape_col * weighted
      end
    end
  end

  return nothing
end

struct NonlinearGradientJump{F}
  field::F
end

function interface_residual!(local_rhs, operator::NonlinearGradientJump, values::InterfaceValues,
                             state::State)
  minus_values = Grico.minus(values)
  plus_values = Grico.plus(values)
  minus_block = Grico.block(local_rhs, minus_values, operator.field)
  plus_block = Grico.block(local_rhs, plus_values, operator.field)
  minus_modes = Grico.local_mode_count(minus_values, operator.field)
  plus_modes = Grico.local_mode_count(plus_values, operator.field)

  @inbounds for point_index in 1:Grico.point_count(values)
    jump = Grico.jump(Grico.normal_gradient(minus_values, state, operator.field, point_index),
                      Grico.normal_gradient(plus_values, state, operator.field, point_index))
    weighted = jump^3 * Grico.weight(values, point_index)

    for mode_index in 1:minus_modes
      minus_block[mode_index] -= Grico.shape_normal_gradient(minus_values, operator.field,
                                                             point_index, mode_index) * weighted
    end

    for mode_index in 1:plus_modes
      plus_block[mode_index] += Grico.shape_normal_gradient(plus_values, operator.field,
                                                            point_index, mode_index) * weighted
    end
  end

  return nothing
end

function interface_tangent!(local_matrix, operator::NonlinearGradientJump, values::InterfaceValues,
                            state::State)
  minus_values = Grico.minus(values)
  plus_values = Grico.plus(values)
  minus_minus = Grico.block(local_matrix, minus_values, operator.field, minus_values,
                            operator.field)
  minus_plus = Grico.block(local_matrix, minus_values, operator.field, plus_values,
                           operator.field)
  plus_minus = Grico.block(local_matrix, plus_values, operator.field, minus_values,
                           operator.field)
  plus_plus = Grico.block(local_matrix, plus_values, operator.field, plus_values,
                          operator.field)
  minus_modes = Grico.local_mode_count(minus_values, operator.field)
  plus_modes = Grico.local_mode_count(plus_values, operator.field)

  @inbounds for point_index in 1:Grico.point_count(values)
    jump = Grico.jump(Grico.normal_gradient(minus_values, state, operator.field, point_index),
                      Grico.normal_gradient(plus_values, state, operator.field, point_index))
    weighted = 3.0 * jump^2 * Grico.weight(values, point_index)

    for row_mode in 1:minus_modes
      row_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                 row_mode)

      for col_mode in 1:minus_modes
        col_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                   col_mode)
        minus_minus[row_mode, col_mode] += row_gradient * col_gradient * weighted
      end

      for col_mode in 1:plus_modes
        col_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                   col_mode)
        minus_plus[row_mode, col_mode] -= row_gradient * col_gradient * weighted
      end
    end

    for row_mode in 1:plus_modes
      row_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                 row_mode)

      for col_mode in 1:minus_modes
        col_gradient = Grico.shape_normal_gradient(minus_values, operator.field, point_index,
                                                   col_mode)
        plus_minus[row_mode, col_mode] -= row_gradient * col_gradient * weighted
      end

      for col_mode in 1:plus_modes
        col_gradient = Grico.shape_normal_gradient(plus_values, operator.field, point_index,
                                                   col_mode)
        plus_plus[row_mode, col_mode] += row_gradient * col_gradient * weighted
      end
    end
  end

  return nothing
end

@inline function _state_pattern(index::Int)
  return 0.35 * sin(0.017 * index) + 0.25 * cos(0.011 * index)
end

@inline function _poisson_exact_solution(x)
  sinpi(x[1]) * sinpi(x[2])
end

@inline function _poisson_source_term(x)
  2 * pi^2 * _poisson_exact_solution(x)
end

@inline function _singular_exact_solution(x)
  radius = sqrt(sum(abs2, x))
  radius == 0.0 ? 0.0 : radius^SINGULAR_EXPONENT
end

@inline function _singular_source_term(x)
  radius = sqrt(sum(abs2, x))
  radius == 0.0 ? 0.0 : SOURCE_FACTOR * radius^(SINGULAR_EXPONENT - 2)
end

function _integration_metadata(plan)
  integration = plan.integration
  local_counts = Int[]

  for item in integration.cells
    push!(local_counts, item.local_dof_count)
  end

  for item in integration.boundary_faces
    push!(local_counts, item.local_dof_count)
  end

  for item in integration.interfaces
    push!(local_counts, item.local_dof_count)
  end

  for item in integration.embedded_surfaces
    push!(local_counts, item.local_dof_count)
  end

  return Dict{String,Any}("cells" => length(integration.cells),
                          "boundary_faces" => length(integration.boundary_faces),
                          "interfaces" => length(integration.interfaces),
                          "embedded_surfaces" => length(integration.embedded_surfaces),
                          "max_local_dofs" => isempty(local_counts) ? 0 :
                                              maximum(local_counts))
end

function _prepared_metadata(plan; system=nothing, extra::Dict{String,Any}=Dict{String,Any}())
  metadata = _integration_metadata(plan)
  metadata["full_dofs"] = Grico.dof_count(plan)
  metadata["fields"] = Grico.field_count(Grico.field_layout(plan))

  if system !== nothing
    metadata["reduced_dofs"] = size(Grico.matrix(system), 1)
    metadata["matrix_nnz"] = nnz(Grico.matrix(system))
  end

  for (key, value) in extra
    metadata[key] = value
  end

  return metadata
end

function _affine_cell_case()
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (40, 40))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(3)))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, Diffusion(u, 1.0))
  Grico.add_cell!(problem, Source(u, _poisson_source_term))

  for axis in 1:2, side in (Grico.LOWER, Grico.UPPER)
    Grico.add_constraint!(problem, Grico.Dirichlet(u, Grico.BoundaryFace(axis, side),
                                                   _poisson_exact_solution))
  end

  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  solution = Grico.solve(system; linear_solve=DIRECT_LINEAR_SOLVE)
  state = Grico.State(plan, solution)
  preconditioner = Grico.AdditiveSchwarzPreconditioner(min_dofs=0)
  metadata = _prepared_metadata(plan; system,
                                extra=Dict("case_kind" => "affine",
                                           "continuity" => "cg",
                                           "degree" => 3,
                                           "root_counts" => [40, 40],
                                           "active_leaves" => length(Grico.active_leaves(space))))

  operations = OperationSpec[
    OperationSpec("assemble", "assemble(plan)", () -> Grico.assemble(plan), () -> nothing),
    OperationSpec("preconditioner_build", "build AdditiveSchwarzPreconditioner",
                  () -> Grico._preconditioner_operator(system, preconditioner),
                  () -> empty!(system.preconditioner_cache)),
    OperationSpec("solve_direct", "solve(system; linear_solve=A\\\\b)",
                  () -> Grico.solve(system; linear_solve=DIRECT_LINEAR_SOLVE), () -> nothing),
    OperationSpec("adaptivity_plan", "hp_adaptivity_plan(state, u)",
                  () -> Grico.hp_adaptivity_plan(state, u; threshold=ADAPTIVITY_THRESHOLD,
                                                 smoothness_threshold=SMOOTHNESS_THRESHOLD),
                  () -> nothing),
  ]

  return PreparedCase("affine_cell_diffusion", "Affine Cell-Dominated Diffusion",
                      "Continuous scalar Poisson problem with volume-dominated affine assembly.",
                      metadata, operations)
end

function _affine_interface_case()
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (40, 40))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2),
                                           continuity=:dg))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.AffineProblem(u)
  Grico.add_cell!(problem, MassCoupling(u, u, 1.0))
  Grico.add_interface!(problem, GradientJumpPenalty(u, 0.5))
  plan = Grico.compile(problem)
  system = Grico.assemble(plan)
  preconditioner = Grico.AdditiveSchwarzPreconditioner(min_dofs=0)
  metadata = _prepared_metadata(plan; system,
                                extra=Dict("case_kind" => "affine",
                                           "continuity" => "dg",
                                           "degree" => 2,
                                           "root_counts" => [40, 40],
                                           "active_leaves" => length(Grico.active_leaves(space))))

  operations = OperationSpec[
    OperationSpec("assemble", "assemble(plan)", () -> Grico.assemble(plan), () -> nothing),
    OperationSpec("preconditioner_build", "build AdditiveSchwarzPreconditioner",
                  () -> Grico._preconditioner_operator(system, preconditioner),
                  () -> empty!(system.preconditioner_cache)),
    OperationSpec("solve_direct", "solve(system; linear_solve=A\\\\b)",
                  () -> Grico.solve(system; linear_solve=DIRECT_LINEAR_SOLVE), () -> nothing),
  ]

  return PreparedCase("affine_interface_dg", "Affine Interface-Heavy DG Mass + Jump",
                      "Discontinuous scalar mass problem with explicit interior interface work.",
                      metadata, operations)
end

function _nonlinear_interface_case()
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (28, 28))
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(2),
                                           continuity=:dg))
  u = Grico.ScalarField(space; name=:u)
  problem = Grico.ResidualProblem(u)
  Grico.add_cell!(problem, NonlinearReaction(u, 0.2))
  Grico.add_interface!(problem, NonlinearGradientJump(u))

  for axis in 1:2, side in (Grico.LOWER, Grico.UPPER)
    target = axis == 2 && side == Grico.UPPER ? 1.0 : 0.0
    Grico.add_boundary!(problem, Grico.BoundaryFace(axis, side),
                        NonlinearBoundaryReaction(u, target))
  end

  plan = Grico.compile(problem)
  state = Grico.State(plan)

  for index in eachindex(Grico.coefficients(state))
    Grico.coefficients(state)[index] = _state_pattern(index)
  end

  residual_buffer = zeros(Float64, Grico.dof_count(plan))
  metadata = _prepared_metadata(plan;
                                extra=Dict("case_kind" => "nonlinear",
                                           "continuity" => "dg",
                                           "degree" => 2,
                                           "root_counts" => [28, 28],
                                           "active_leaves" => length(Grico.active_leaves(space))))

  operations = OperationSpec[
    OperationSpec("residual_bang", "residual!(buffer, plan, state)",
                  () -> Grico.residual!(residual_buffer, plan, state), () -> nothing),
    OperationSpec("tangent", "tangent(plan, state)", () -> Grico.tangent(plan, state),
                  () -> nothing),
  ]

  return PreparedCase("nonlinear_interface_dg", "Nonlinear DG Residual + Tangent",
                      "Nonlinear discontinuous problem with cell, boundary, and interface terms.",
                      metadata, operations)
end

function _adaptive_benchmark_domain()
  domain = Grico.Domain((0.0, 0.0), (1.0, 1.0), (8, 8))
  grid_data = Grico.grid(domain)

  for pass in 1:2
    axis = isodd(pass) ? 1 : 2

    for leaf in Grico.active_leaves(grid_data)
      center = Grico.cell_center(domain, leaf)
      distance = abs(center[1] - 0.5) + abs(center[2] - 0.5)
      distance <= (pass == 1 ? 0.55 : 0.35) || continue
      Grico.refine!(grid_data, leaf, axis)
    end
  end

  return domain
end

function _adaptive_poisson_case()
  domain = _adaptive_benchmark_domain()
  space = Grico.HpSpace(domain,
                        Grico.SpaceOptions(basis=Grico.FullTensorBasis(),
                                           degree=Grico.UniformDegree(3)))
  u = Grico.ScalarField(space; name=:u)
  state = Grico.State(Grico.FieldLayout((u,)))

  for index in eachindex(Grico.coefficients(state))
    Grico.coefficients(state)[index] = _state_pattern(index)
  end

  metadata = Dict{String,Any}("case_kind" => "adaptivity",
                              "continuity" => "cg",
                              "degree" => 3,
                              "root_counts" => [8, 8],
                              "manual_refinement_passes" => 2,
                              "active_leaves" => length(Grico.active_leaves(space)),
                              "full_dofs" => length(Grico.coefficients(state)))

  operations = OperationSpec[
    OperationSpec("adaptivity_plan", "hp_adaptivity_plan(state, u)",
                  () -> Grico.hp_adaptivity_plan(state, u; threshold=ADAPTIVITY_THRESHOLD,
                                                 smoothness_threshold=SMOOTHNESS_THRESHOLD),
                  () -> nothing),
  ]

  return PreparedCase("adaptive_poisson", "Adaptive Planning On Refined Mesh",
                      "Adaptivity-planning benchmark on a deterministic manually refined mesh.",
                      metadata, operations)
end

function build_phase0_case(id::AbstractString)
  if id == "affine_cell_diffusion"
    return _affine_cell_case()
  elseif id == "affine_interface_dg"
    return _affine_interface_case()
  elseif id == "nonlinear_interface_dg"
    return _nonlinear_interface_case()
  elseif id == "adaptive_poisson"
    return _adaptive_poisson_case()
  end

  supported = join(phase0_case_ids(), ", ")
  throw(ArgumentError("unknown Phase 0 case `$id`; supported cases: $supported"))
end

end

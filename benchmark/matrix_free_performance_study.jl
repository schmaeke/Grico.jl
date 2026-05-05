# Matrix-free performance study for realistic benchmark cases.
#
# This script is intentionally separate from the regular benchmark drivers. The
# regular drivers stay focused on user-visible workflows, while this file uses a
# small amount of internal instrumentation to split matrix-free runtime into the
# phases and hot kernels that matter for subsequent performance work.

const PERF_ROOT = normpath(joinpath(@__DIR__, ".."))
PERF_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, PERF_ROOT)

using Grico
using Printf

module _PerfCGPoisson
include(joinpath(@__DIR__, "steady_poisson_adaptivity.jl"))
end

module _PerfSIPGDG
include(joinpath(@__DIR__, "steady_poisson_sipg_dg_adaptivity.jl"))
end

module _PerfBratuCG
include(joinpath(@__DIR__, "nonlinear_bratu_cg_adaptivity.jl"))
end

module _PerfAnnularFCM
include(joinpath(@__DIR__, "annular_nitsche_fcm_adaptivity.jl"))
end

const CASE_ORDER = (:cg_poisson, :sipg_dg, :bratu_cg, :annular_fcm)

function _sample_vector(::Type{T}, count::Int) where {T<:AbstractFloat}
  values = Vector{T}(undef, count)

  @inbounds for index in 1:count
    values[index] = T(sin(0.017 * index) + 0.5 * cos(0.031 * index))
  end

  return values
end

function _csv_escape(value)
  text = string(value)
  text = replace(text, "\"" => "\"\"")
  return any(character -> character in (',', '"', '\n', '\r'), text) ? "\"$text\"" : text
end

function _write_csv(path::AbstractString, rows::Vector{NamedTuple})
  mkpath(dirname(path))

  open(path, "w") do io
    isempty(rows) && return nothing
    names = propertynames(first(rows))
    println(io, join(names, ","))

    for row in rows
      println(io, join((_csv_escape(getproperty(row, name)) for name in names), ","))
    end
  end

  return path
end

function _push_row!(rows::Vector{NamedTuple}, case_name, category, component, repetitions,
                    total_seconds, total_bytes, gc_seconds; value=NaN, note="")
  calls = max(repetitions, 1)
  push!(rows,
        (; case=String(case_name), category=String(category), component=String(component),
         repetitions=Int(repetitions), total_seconds=Float64(total_seconds),
         seconds_per_call=Float64(total_seconds) / calls, total_bytes=Int(total_bytes),
         bytes_per_call=Float64(total_bytes) / calls, gc_seconds=Float64(gc_seconds),
         value=Float64(value), note=String(note)))
  return rows
end

function _push_metric!(rows::Vector{NamedTuple}, case_name, category, component, value; note="")
  return _push_row!(rows, case_name, category, component, 0, 0.0, 0, 0.0; value, note)
end

function _measure!(thunk, rows::Vector{NamedTuple}, case_name, category, component,
                   repetitions::Int; note="")
  repetitions >= 1 || throw(ArgumentError("repetitions must be positive"))
  GC.gc()
  last_value = nothing
  timing = @timed begin
    for _ in 1:repetitions
      last_value = thunk()
    end
  end
  _push_row!(rows, case_name, category, component, repetitions, timing.time, timing.bytes,
             timing.gctime; note)
  return last_value
end

function _measure_attempt!(thunk, rows::Vector{NamedTuple}, case_name, category, component; note="")
  GC.gc()
  success = Ref(true)
  message = Ref(String(note))
  value = Ref{Any}(nothing)
  timing = @timed begin
    try
      value[] = thunk()
    catch error
      success[] = false
      message[] = isempty(message[]) ? sprint(showerror, error) :
                  string(message[], "; ", sprint(showerror, error))
    end
  end
  _push_row!(rows, case_name, category, component, 1, timing.time, timing.bytes, timing.gctime;
             note=message[])
  return success[], value[]
end

function _case_options(defaults, pairs...)
  options = copy(defaults)

  for pair in pairs
    options[String(pair.first)] = pair.second
  end

  options["cycles"] = 1
  options["write_plots"] = false
  options["warmup"] = false
  return options
end

function _case_config(case::Symbol, preset::Symbol)
  smoke = preset === :smoke
  large = preset === :large

  case === :cg_poisson && return (; name=case, nonlinear=false, module_ref=_PerfCGPoisson,
                                  defaults=_PerfCGPoisson.CG_POISSON_DEFAULTS,
                                  options=_case_options(_PerfCGPoisson.CG_POISSON_DEFAULTS,
                                                        :root_cells => smoke ? (8, 8) : large ? (96, 96) : (48, 48),
                                                        :degree => smoke ? 2 : 3, :quadrature_extra_points => 2,
                                                        :max_h_level => 8, :tolerance => 1.0e-5),
                                  build_initial_field=_PerfCGPoisson._cg_initial_field,
                                  build_problem=_PerfCGPoisson._cg_poisson_problem)

  case === :sipg_dg && return (; name=case, nonlinear=false, module_ref=_PerfSIPGDG,
                               defaults=_PerfSIPGDG.SIPG_DG_POISSON_DEFAULTS,
                               options=_case_options(_PerfSIPGDG.SIPG_DG_POISSON_DEFAULTS,
                                                     :root_cells => smoke ? (6, 6) : large ? (64, 64) : (32, 32),
                                                     :degree => smoke ? 2 : 3, :quadrature_extra_points => 2,
                                                     :max_h_level => 8, :tolerance => 1.0e-5),
                               build_initial_field=_PerfSIPGDG._sipg_dg_initial_field,
                               build_problem=_PerfSIPGDG._sipg_dg_poisson_problem)

  case === :bratu_cg && return (; name=case, nonlinear=true, module_ref=_PerfBratuCG,
                                defaults=_PerfBratuCG.BRATU_CG_DEFAULTS,
                                options=_case_options(_PerfBratuCG.BRATU_CG_DEFAULTS,
                                                      :root_cells => smoke ? (6, 6) : large ? (64, 64) : (32, 32),
                                                      :degree => smoke ? 2 : 3, :quadrature_extra_points => 2,
                                                      :max_h_level => 8, :tolerance => 1.0e-5, :newton_iterations => 8,
                                                      :newton_tolerance => 1.0e-8, :newton_damping => 1.0),
                                build_initial_field=_PerfBratuCG._bratu_initial_field,
                                build_problem=_PerfBratuCG._bratu_problem)

  case === :annular_fcm && return (; name=case, nonlinear=false, module_ref=_PerfAnnularFCM,
                                   defaults=_PerfAnnularFCM.ANNULAR_FCM_DEFAULTS,
                                   options=_case_options(_PerfAnnularFCM.ANNULAR_FCM_DEFAULTS,
                                                         :root_cells => smoke ? (8, 8) : large ? (32, 32) : (16, 16),
                                                         :degree => smoke ? 2 : 3, :quadrature_extra_points => 2,
                                                         :max_h_level => 8, :tolerance => 1.0e-5,
                                                         :segments => smoke ? 64 : large ? 256 : 128, :surface_points => 4,
                                                         :fcm_depth => smoke ? 2 : 4, :penalty => 1000.0),
                                   build_initial_field=_PerfAnnularFCM._annular_initial_field,
                                   build_problem=_PerfAnnularFCM._annular_problem)

  throw(ArgumentError("unknown case $case"))
end

function _warmup_options(options)
  warmup = copy(options)
  warmup["root_cells"] = (4, 4)
  warmup["degree"] = min(Int(options["degree"]), 2)
  warmup["max_h_level"] = min(something(options["max_h_level"], 4), 4)

  haskey(warmup, "segments") && (warmup["segments"] = min(Int(warmup["segments"]), 32))
  haskey(warmup, "fcm_depth") && (warmup["fcm_depth"] = min(Int(warmup["fcm_depth"]), 2))
  haskey(warmup, "newton_iterations") && (warmup["newton_iterations"] = 2)
  return warmup
end

_affine_preconditioner(config) = config.name === :annular_fcm ? nothing : JacobiPreconditioner()

function _affine_solver(config)
  preconditioner = _affine_preconditioner(config)
  return preconditioner === nothing ? CGSolver() : CGSolver(preconditioner=preconditioner)
end

function _affine_maxiter(config, reduced_count::Int)
  config.name === :sipg_dg && return 500
  config.name === :annular_fcm && return 500
  return max(1_000, 2 * reduced_count)
end

function _warmup_case(config)
  options = _warmup_options(config.options)
  field = config.build_initial_field(options)
  problem = config.build_problem(field, options)
  plan = compile(problem)
  T = eltype(coefficients(State(plan)))

  if config.nonlinear
    state = State(plan)
    result = zeros(T, Grico.dof_count(plan))
    increment = _sample_vector(T, Grico.dof_count(plan))
    workspace = ResidualWorkspace(plan)
    residual!(result, plan, state, workspace)
    tangent_apply!(result, plan, state, increment, workspace)
    state = solve(plan; maxiter=2, absolute_tolerance=T(1.0e-6), relative_tolerance=T(1.0e-6),
                  linear_maxiter=max(100, 2 * Grico.reduced_dof_count(plan)))
  else
    vector = _sample_vector(T, Grico.dof_count(plan))
    result = zeros(T, Grico.dof_count(plan))
    rhs!(result, plan)
    apply!(result, plan, vector)
    state = solve(plan; solver=_affine_solver(config),
                  maxiter=_affine_maxiter(config, Grico.reduced_dof_count(plan)))
  end

  limits = config.module_ref._adaptivity_limits(field_space(field), options)
  adaptivity = Grico.adaptivity_plan(state, field; tolerance=options["tolerance"],
                                     smoothness_threshold=options["smoothness_threshold"], limits)

  if !isempty(adaptivity)
    transition_data = transition(adaptivity; compact=get(options, "compact_transition", false))
    target_field = adapted_field(transition_data, field)
    transfer_state(transition_data, state, field, target_field)
  end

  return nothing
end

function _first_batch_item(batches, items)
  isempty(batches) && return nothing
  batch = first(batches)
  isempty(batch.item_indices) && return nothing
  return batch, items[first(batch.item_indices)]
end

function _time_local_affine_kernel!(rows, case_name, component, batch, item, operators, hook,
                                    repetitions::Int, ::Type{T}) where {T<:AbstractFloat}
  isempty(operators) && return nothing
  local_input = _sample_vector(T, batch.local_dof_count)
  local_output = zeros(T, batch.local_dof_count)
  scratch = KernelScratch(T)

  _measure!(rows, case_name, "micro", component, repetitions) do
    fill!(local_output, zero(T))

    for operator in operators
      hook(local_output, operator, item, local_input, scratch)
    end

    nothing
  end

  return nothing
end

function _time_local_rhs_kernel!(rows, case_name, component, batch, item, operators, hook,
                                 repetitions::Int, ::Type{T}) where {T<:AbstractFloat}
  isempty(operators) && return nothing
  local_rhs = zeros(T, batch.local_dof_count)
  scratch = KernelScratch(T)

  _measure!(rows, case_name, "micro", component, repetitions) do
    fill!(local_rhs, zero(T))

    for operator in operators
      hook(local_rhs, operator, item, scratch)
    end

    nothing
  end

  return nothing
end

function _time_local_residual_kernel!(rows, case_name, component, batch, item, operators, hook,
                                      state, repetitions::Int, ::Type{T}) where {T<:AbstractFloat}
  isempty(operators) && return nothing
  local_rhs = zeros(T, batch.local_dof_count)
  scratch = KernelScratch(T)

  _measure!(rows, case_name, "micro", component, repetitions) do
    fill!(local_rhs, zero(T))

    for operator in operators
      hook(local_rhs, operator, item, state, scratch)
    end

    nothing
  end

  return nothing
end

function _time_local_tangent_kernel!(rows, case_name, component, batch, item, operators, hook,
                                     state, repetitions::Int, ::Type{T}) where {T<:AbstractFloat}
  isempty(operators) && return nothing
  local_increment = _sample_vector(T, batch.local_dof_count)
  local_output = zeros(T, batch.local_dof_count)
  scratch = KernelScratch(T)

  _measure!(rows, case_name, "micro", component, repetitions) do
    fill!(local_output, zero(T))

    for operator in operators
      hook(local_output, operator, item, state, local_increment, scratch)
    end

    nothing
  end

  return nothing
end

function _filtered_operators(operators, batch)
  return [operators[index].operator for index in batch.operator_indices]
end

function _measure_affine_local_kernels!(rows, case_name, plan, repetitions::Int)
  T = eltype(coefficients(State(plan)))
  traversal = plan.traversal_plan

  data = _first_batch_item(traversal.cell_batches, plan.integration.cells)
  if data !== nothing
    batch, item = data
    _time_local_rhs_kernel!(rows, case_name, "local_cell_rhs", batch, item, plan.cell_operators,
                            cell_rhs!, repetitions, T)
    _time_local_affine_kernel!(rows, case_name, "local_cell_apply", batch, item,
                               plan.cell_operators, cell_apply!, repetitions, T)
  end

  data = _first_batch_item(traversal.boundary_batches, plan.integration.boundary_faces)
  if data !== nothing
    batch, item = data
    operators = _filtered_operators(plan.boundary_operators, batch)
    _time_local_rhs_kernel!(rows, case_name, "local_boundary_rhs", batch, item, operators,
                            face_rhs!, repetitions, T)
    _time_local_affine_kernel!(rows, case_name, "local_boundary_apply", batch, item, operators,
                               face_apply!, repetitions, T)
  end

  data = _first_batch_item(traversal.interface_batches, plan.integration.interfaces)
  if data !== nothing
    batch, item = data
    _time_local_rhs_kernel!(rows, case_name, "local_interface_rhs", batch, item,
                            plan.interface_operators, interface_rhs!, repetitions, T)
    _time_local_affine_kernel!(rows, case_name, "local_interface_apply", batch, item,
                               plan.interface_operators, interface_apply!, repetitions, T)
  end

  data = _first_batch_item(traversal.surface_batches, plan.integration.embedded_surfaces)
  if data !== nothing
    batch, item = data
    operators = _filtered_operators(plan.surface_operators, batch)
    _time_local_rhs_kernel!(rows, case_name, "local_surface_rhs", batch, item, operators,
                            surface_rhs!, repetitions, T)
    _time_local_affine_kernel!(rows, case_name, "local_surface_apply", batch, item, operators,
                               surface_apply!, repetitions, T)
  end

  return nothing
end

function _measure_residual_local_kernels!(rows, case_name, plan, state, repetitions::Int)
  T = eltype(coefficients(State(plan)))
  traversal = plan.traversal_plan

  data = _first_batch_item(traversal.cell_batches, plan.integration.cells)
  if data !== nothing
    batch, item = data
    _time_local_residual_kernel!(rows, case_name, "local_cell_residual", batch, item,
                                 plan.cell_operators, cell_residual!, state, repetitions, T)
    _time_local_tangent_kernel!(rows, case_name, "local_cell_tangent_apply", batch, item,
                                plan.cell_operators, cell_tangent_apply!, state, repetitions, T)
  end

  data = _first_batch_item(traversal.boundary_batches, plan.integration.boundary_faces)
  if data !== nothing
    batch, item = data
    operators = _filtered_operators(plan.boundary_operators, batch)
    _time_local_residual_kernel!(rows, case_name, "local_boundary_residual", batch, item, operators,
                                 face_residual!, state, repetitions, T)
    _time_local_tangent_kernel!(rows, case_name, "local_boundary_tangent_apply", batch, item,
                                operators, face_tangent_apply!, state, repetitions, T)
  end

  data = _first_batch_item(traversal.interface_batches, plan.integration.interfaces)
  if data !== nothing
    batch, item = data
    _time_local_residual_kernel!(rows, case_name, "local_interface_residual", batch, item,
                                 plan.interface_operators, interface_residual!, state, repetitions,
                                 T)
    _time_local_tangent_kernel!(rows, case_name, "local_interface_tangent_apply", batch, item,
                                plan.interface_operators, interface_tangent_apply!, state,
                                repetitions, T)
  end

  data = _first_batch_item(traversal.surface_batches, plan.integration.embedded_surfaces)
  if data !== nothing
    batch, item = data
    operators = _filtered_operators(plan.surface_operators, batch)
    _time_local_residual_kernel!(rows, case_name, "local_surface_residual", batch, item, operators,
                                 surface_residual!, state, repetitions, T)
    _time_local_tangent_kernel!(rows, case_name, "local_surface_tangent_apply", batch, item,
                                operators, surface_tangent_apply!, state, repetitions, T)
  end

  return nothing
end

function _counting_affine_linear_solve(plan::AssemblyPlan{D,T}, reduced_rhs::AbstractVector{T};
                                       workspace, preconditioner=nothing,
                                       relative_tolerance=sqrt(eps(T)), absolute_tolerance=zero(T),
                                       maxiter=max(1_000, 2 * length(reduced_rhs)),
                                       initial_solution=nothing,
                                       counter=Ref(0)) where {D,T<:AbstractFloat}
  operator = Grico._CountingReducedOperator(Grico._ReducedAffineOperator(plan, workspace),
                                            counter)
  policy = preconditioner === nothing ? IdentityPreconditioner() : preconditioner
  compiled_preconditioner = Grico._compile_preconditioner(policy, operator)
  return Grico._cg_solve(operator, reduced_rhs, compiled_preconditioner;
                         relative_tolerance=T(relative_tolerance),
                         absolute_tolerance=T(absolute_tolerance), maxiter, initial_solution)
end

function _counting_tangent_linear_solve(plan::AssemblyPlan{D,T}, state::State{T},
                                        reduced_rhs::AbstractVector{T}; workspace,
                                        residual_workspace, preconditioner=nothing,
                                        relative_tolerance=sqrt(eps(T)), absolute_tolerance=zero(T),
                                        maxiter=max(1_000, 2 * length(reduced_rhs)),
                                        initial_solution=nothing,
                                        counter=Ref(0)) where {D,T<:AbstractFloat}
  preconditioner === nothing ||
    throw(ArgumentError("tangent counting solve expects identity preconditioning"))
  copyto!(workspace.full_state, coefficients(state))
  operator = Grico._CountingReducedOperator(Grico._ReducedTangentOperator(plan, workspace,
                                                                          residual_workspace),
                                            counter)
  compiled_preconditioner = Grico._compile_preconditioner(IdentityPreconditioner(), operator)
  return Grico._cg_solve(operator, reduced_rhs, compiled_preconditioner;
                         relative_tolerance=T(relative_tolerance),
                         absolute_tolerance=T(absolute_tolerance), maxiter, initial_solution)
end

function _measure_adaptivity!(rows, config, case_name, field, plan, state)
  limits = config.module_ref._adaptivity_limits(field_space(field), config.options)
  adaptivity = _measure!(rows, case_name, "phase", "adaptivity_plan", 1) do
    Grico.adaptivity_plan(state, field; tolerance=config.options["tolerance"],
                          smoothness_threshold=config.options["smoothness_threshold"], limits)
  end

  isempty(adaptivity) && return nothing

  transition_data = _measure!(rows, case_name, "phase", "transition", 1) do
    transition(adaptivity; compact=get(config.options, "compact_transition", false))
  end
  target_field = _measure!(rows, case_name, "phase", "adapted_field", 1) do
    adapted_field(transition_data, field)
  end
  transfer_success, _ = _measure_attempt!(rows, case_name, "phase", "transfer_state") do
    transfer_state(transition_data, state, field, target_field)
  end
  _push_metric!(rows, case_name, "metadata", "transfer_succeeded", transfer_success ? 1.0 : 0.0)

  return nothing
end

function _measure_affine_case!(rows, config, repetitions::Int, local_repetitions::Int)
  case_name = config.name
  options = config.options
  field = _measure!(rows, case_name, "phase", "problem_setup_field", 1) do
    config.build_initial_field(options)
  end
  problem = _measure!(rows, case_name, "phase", "problem_setup_operator", 1) do
    config.build_problem(field, options)
  end
  plan = _measure!(rows, case_name, "phase", "compile", 1) do
    compile(problem)
  end

  T = eltype(coefficients(State(plan)))
  _push_metric!(rows, case_name, "metadata", "active_leaves", active_leaf_count(field_space(field)))
  _push_metric!(rows, case_name, "metadata", "dofs", Grico.dof_count(plan))
  _push_metric!(rows, case_name, "metadata", "reduced_dofs", Grico.reduced_dof_count(plan))

  full_rhs = zeros(T, Grico.dof_count(plan))
  coefficients_data = _sample_vector(T, Grico.dof_count(plan))
  full_result = zeros(T, Grico.dof_count(plan))
  workspace = Grico._ReducedOperatorWorkspace(plan)
  reduced_values = _sample_vector(T, Grico.reduced_dof_count(plan))
  reduced_result = zeros(T, length(reduced_values))

  _measure!(rows, case_name, "phase", "rhs!", repetitions) do
    rhs!(full_rhs, plan)
  end
  _measure!(rows, case_name, "phase", "apply!", repetitions) do
    apply!(full_result, plan, coefficients_data)
  end
  _measure!(rows, case_name, "micro", "reduced_expand", repetitions) do
    Grico._expand_reduced!(workspace.full_input, Grico._reduced_map(plan), reduced_values;
                           include_shift=false)
  end
  _measure!(rows, case_name, "micro", "reduced_project", repetitions) do
    Grico._project_reduced!(reduced_result, Grico._reduced_map(plan), workspace.full_output)
  end
  _measure!(rows, case_name, "micro", "reduced_apply", repetitions) do
    Grico._reduced_apply!(reduced_result, plan, reduced_values, workspace)
  end
  _measure!(rows, case_name, "micro", "reduced_rhs", repetitions) do
    Grico._reduced_rhs!(reduced_result, plan, workspace)
  end

  diagonal_available = Ref(false)
  _measure!(rows, case_name, "micro", "reduced_diagonal", repetitions) do
    diagonal_available[] = Grico._reduced_diagonal!(reduced_result, plan, workspace)
  end
  _push_metric!(rows, case_name, "metadata", "jacobi_diagonal_available",
                diagonal_available[] ? 1.0 : 0.0)

  _measure_affine_local_kernels!(rows, case_name, plan, local_repetitions)

  operator_calls = Ref(0)
  preconditioner = _affine_preconditioner(config)
  solve_success, state = _measure_attempt!(rows, case_name, "phase", "solve") do
    solve_workspace = Grico._ReducedOperatorWorkspace(plan)
    solve_rhs = zeros(T, Grico.reduced_dof_count(plan))
    Grico._reduced_rhs!(solve_rhs, plan, solve_workspace)
    solve_values = _counting_affine_linear_solve(plan, solve_rhs;
                                                 workspace=solve_workspace,
                                                 preconditioner=preconditioner,
                                                 maxiter=_affine_maxiter(config,
                                                                         length(reduced_values)),
                                                 counter=operator_calls)
    Grico._state_from_reduced_result(plan, solve_values)
  end
  _push_metric!(rows, case_name, "metadata", "cg_operator_applications", operator_calls[])
  _push_metric!(rows, case_name, "metadata", "solve_succeeded", solve_success ? 1.0 : 0.0)
  solve_success || return nothing

  _measure_adaptivity!(rows, config, case_name, field, plan, state)
  return nothing
end

function _measure_residual_case!(rows, config, repetitions::Int, local_repetitions::Int)
  case_name = config.name
  options = config.options
  field = _measure!(rows, case_name, "phase", "problem_setup_field", 1) do
    config.build_initial_field(options)
  end
  problem = _measure!(rows, case_name, "phase", "problem_setup_operator", 1) do
    config.build_problem(field, options)
  end
  plan = _measure!(rows, case_name, "phase", "compile", 1) do
    compile(problem)
  end

  T = eltype(coefficients(State(plan)))
  _push_metric!(rows, case_name, "metadata", "active_leaves", active_leaf_count(field_space(field)))
  _push_metric!(rows, case_name, "metadata", "dofs", Grico.dof_count(plan))
  _push_metric!(rows, case_name, "metadata", "reduced_dofs", Grico.reduced_dof_count(plan))

  state = State(plan)
  residual_workspace = ResidualWorkspace(plan)
  residual_result = zeros(T, Grico.dof_count(plan))
  tangent_result = zeros(T, Grico.dof_count(plan))
  full_increment = _sample_vector(T, Grico.dof_count(plan))
  reduced_workspace = Grico._ReducedOperatorWorkspace(plan)
  reduced_values = zeros(T, Grico.reduced_dof_count(plan))
  reduced_increment = _sample_vector(T, length(reduced_values))
  reduced_result = zeros(T, length(reduced_values))

  _measure!(rows, case_name, "phase", "residual!", repetitions) do
    residual!(residual_result, plan, state, residual_workspace)
  end
  _measure!(rows, case_name, "phase", "tangent_apply!", repetitions) do
    tangent_apply!(tangent_result, plan, state, full_increment, residual_workspace)
  end
  _measure!(rows, case_name, "micro", "reduced_residual", repetitions) do
    Grico._reduced_residual!(reduced_result, plan, reduced_values, reduced_workspace,
                             residual_workspace)
  end
  _measure!(rows, case_name, "micro", "reduced_tangent_apply", repetitions) do
    Grico._reduced_tangent_apply!(reduced_result, plan, reduced_increment, reduced_workspace,
                                  residual_workspace)
  end

  _measure_residual_local_kernels!(rows, case_name, plan, state, local_repetitions)

  Grico._reduced_residual!(reduced_result, plan, reduced_values, reduced_workspace,
                           residual_workspace)
  correction_rhs = similar(reduced_result)
  @inbounds for index in eachindex(reduced_result)
    correction_rhs[index] = -reduced_result[index]
  end
  operator_calls = Ref(0)
  _measure!(rows, case_name, "phase", "tangent_linear_solve", 1) do
    _counting_tangent_linear_solve(plan, reduced_workspace.state, correction_rhs;
                                   workspace=reduced_workspace,
                                   residual_workspace=residual_workspace,
                                   relative_tolerance=sqrt(eps(T)), absolute_tolerance=zero(T),
                                   maxiter=max(1_000, 2 * length(correction_rhs)),
                                   counter=operator_calls)
  end
  _push_metric!(rows, case_name, "metadata", "cg_operator_applications", operator_calls[])

  solved_state = State(plan)
  newton_result = Ref{Any}(nothing)
  _measure!(rows, case_name, "phase", "solve", 1) do
    component_rows = NamedTuple[]
    iteration_rows = NamedTuple[]
    newton_result[] = config.module_ref._newton_solve!(component_rows, iteration_rows, 1, plan,
                                                       solved_state, options)
  end
  result = newton_result[]
  _push_metric!(rows, case_name, "metadata", "newton_iterations", result.iteration_count)
  _push_metric!(rows, case_name, "metadata", "newton_corrections", result.correction_count)
  _push_metric!(rows, case_name, "metadata", "newton_converged", result.converged ? 1.0 : 0.0)
  _push_metric!(rows, case_name, "metadata", "newton_residual_norm", result.residual_norm)

  _measure_adaptivity!(rows, config, case_name, field, plan, solved_state)
  return nothing
end

function _parse_cases(text)
  cases = Symbol[]

  for raw in split(text, ',')
    name = Symbol(strip(raw))
    name in CASE_ORDER || throw(ArgumentError("unknown case $name"))
    push!(cases, name)
  end

  return cases
end

function _parse_args(args)
  options = Dict{String,Any}("output" => joinpath(@__DIR__, "output",
                                                  "matrix_free_performance_study.csv"),
                             "cases" => collect(CASE_ORDER), "repetitions" => 3,
                             "local_repetitions" => 500, "warmup" => true, "preset" => :real,
                             "tensor_kernels" => true, "degree" => nothing)
  index = 1

  while index <= length(args)
    arg = args[index]

    if arg == "--help" || arg == "-h"
      println("""
      matrix_free_performance_study.jl

      Options:
        --output PATH              CSV output path
        --cases LIST               comma-separated case list ($(join(CASE_ORDER, ",")))
        --repetitions N            whole-plan repetitions for apply/residual micro phases
        --local-repetitions N      local-kernel repetitions
        --preset NAME              real, large, or smoke (default: real)
        --degree P                 override the preset polynomial degree
        --shape-kernels            use the full shape-table benchmark kernels
        --tensor-kernels           use tensor-product benchmark kernels when available
        --no-warmup                include first-use Julia compilation latency
        --help                     show this message
      """)
      exit(0)
    elseif arg == "--no-warmup"
      options["warmup"] = false
    elseif arg == "--shape-kernels"
      options["tensor_kernels"] = false
    elseif arg == "--tensor-kernels"
      options["tensor_kernels"] = true
    elseif startswith(arg, "--output=")
      options["output"] = split(arg, "=", limit=2)[2]
    elseif arg == "--output"
      index < length(args) || throw(ArgumentError("--output requires a path"))
      index += 1
      options["output"] = args[index]
    elseif startswith(arg, "--cases=")
      options["cases"] = _parse_cases(split(arg, "=", limit=2)[2])
    elseif arg == "--cases"
      index < length(args) || throw(ArgumentError("--cases requires a list"))
      index += 1
      options["cases"] = _parse_cases(args[index])
    elseif startswith(arg, "--repetitions=")
      options["repetitions"] = parse(Int, split(arg, "=", limit=2)[2])
    elseif arg == "--repetitions"
      index < length(args) || throw(ArgumentError("--repetitions requires a value"))
      index += 1
      options["repetitions"] = parse(Int, args[index])
    elseif startswith(arg, "--local-repetitions=")
      options["local_repetitions"] = parse(Int, split(arg, "=", limit=2)[2])
    elseif arg == "--local-repetitions"
      index < length(args) || throw(ArgumentError("--local-repetitions requires a value"))
      index += 1
      options["local_repetitions"] = parse(Int, args[index])
    elseif startswith(arg, "--preset=")
      options["preset"] = Symbol(split(arg, "=", limit=2)[2])
    elseif arg == "--preset"
      index < length(args) || throw(ArgumentError("--preset requires a value"))
      index += 1
      options["preset"] = Symbol(args[index])
    elseif startswith(arg, "--degree=")
      options["degree"] = parse(Int, split(arg, "=", limit=2)[2])
    elseif arg == "--degree"
      index < length(args) || throw(ArgumentError("--degree requires a value"))
      index += 1
      options["degree"] = parse(Int, args[index])
    else
      throw(ArgumentError("unknown option $arg"))
    end

    index += 1
  end

  options["repetitions"] >= 1 || throw(ArgumentError("--repetitions must be positive"))
  options["local_repetitions"] >= 1 || throw(ArgumentError("--local-repetitions must be positive"))
  options["preset"] in (:real, :large, :smoke) ||
    throw(ArgumentError("--preset must be real, large, or smoke"))
  options["degree"] === nothing ||
    options["degree"] >= 0 ||
    throw(ArgumentError("--degree must be nonnegative"))
  return options
end

function main(args=ARGS)
  options = _parse_args(args)
  rows = NamedTuple[]

  @printf("matrix-free performance study: preset=%s repetitions=%d local_repetitions=%d\n",
          options["preset"], options["repetitions"], options["local_repetitions"])

  for case in options["cases"]
    config = _case_config(case, options["preset"])
    config.options["tensor_kernels"] = options["tensor_kernels"]
    options["degree"] === nothing || (config.options["degree"] = options["degree"])

    if options["warmup"]
      @printf("warming up %s\n", case)
      _warmup_case(config)
    end

    @printf("measuring %s\n", case)

    if config.nonlinear
      _measure_residual_case!(rows, config, options["repetitions"], options["local_repetitions"])
    else
      _measure_affine_case!(rows, config, options["repetitions"], options["local_repetitions"])
    end
  end

  path = _write_csv(options["output"], rows)
  println("wrote $path")
  return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

# Boundary and interface weak-form sum-factorization benchmark.
#
# The script measures vector-valued trace bilinear forms on full tensor-product
# bases. It isolates the boundary-face and two-sided interface kernels through
# local apply timings and also records their effect on full and reduced operator
# application.
#
# Example helios run:
#
#   JULIA_NUM_THREADS=16 julia --project=benchmark benchmark/trace_weakform_sumfactorization.jl

const TRACE_WEAKFORM_ROOT = normpath(get(ENV, "GRICO_BENCHMARK_ROOT", joinpath(@__DIR__, "..")))
TRACE_WEAKFORM_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, TRACE_WEAKFORM_ROOT)

using Grico
using Printf

const TRACE_WEAKFORM_DEFAULT_OUTPUT = joinpath(@__DIR__, "output",
                                               "trace_weakform_sumfactorization.csv")

function _option(args, name, default)
  prefix = string(name, "=")

  for (index, arg) in pairs(args)
    arg == name && return index < length(args) ? args[index+1] : default
    startswith(arg, prefix) && return split(arg, "=", limit=2)[2]
  end

  return default
end

function _csv_escape(value)
  text = string(value)
  text = replace(text, "\"" => "\"\"")
  return any(character -> character in (',', '"', '\n', '\r'), text) ? "\"$text\"" : text
end

function _resolve_output_path(path::AbstractString, default_file::AbstractString)
  lowercase(splitext(path)[2]) == ".csv" && return path
  return joinpath(path, default_file)
end

function _write_csv(path::AbstractString, rows::Vector{NamedTuple})
  resolved = _resolve_output_path(path, basename(TRACE_WEAKFORM_DEFAULT_OUTPUT))
  mkpath(dirname(resolved))

  open(resolved, "w") do io
    isempty(rows) && return nothing
    names = propertynames(first(rows))
    println(io, join(names, ","))

    for row in rows
      println(io, join((_csv_escape(getproperty(row, name)) for name in names), ","))
    end
  end

  return resolved
end

function _push_row!(rows::Vector{NamedTuple}, case_name, category, component, repetitions, timing;
                    value=NaN, note="")
  calls = max(repetitions, 1)
  push!(rows,
        (; threads=Threads.nthreads(), case=String(case_name), category=String(category),
         component=String(component), repetitions=Int(repetitions),
         total_seconds=Float64(timing.time), seconds_per_call=Float64(timing.time) / calls,
         total_bytes=Int(timing.bytes), bytes_per_call=Float64(timing.bytes) / calls,
         gc_seconds=Float64(timing.gctime), value=Float64(value), note=String(note)))
  return rows
end

function _push_metric!(rows::Vector{NamedTuple}, case_name, category, component, value; note="")
  timing = (; time=0.0, bytes=0, gctime=0.0)
  return _push_row!(rows, case_name, category, component, 0, timing; value, note)
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
  _push_row!(rows, case_name, category, component, repetitions, timing; note)
  return last_value
end

function _sample_vector(::Type{T}, count::Int) where {T<:AbstractFloat}
  values = Vector{T}(undef, count)

  @inbounds for index in 1:count
    values[index] = T(sin(0.017 * index) + 0.15 * cos(0.031 * index))
  end

  return values
end

function _trace_space(; dim::Int, cells::Int, degree::Int)
  domain = Domain(ntuple(_ -> 0.0, dim), ntuple(_ -> 1.0, dim), ntuple(_ -> cells, dim))
  return HpSpace(domain,
                 SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(degree),
                              continuity=:dg))
end

function _boundary_problem(; dim::Int=2, cells::Int=32, degree::Int=4, components::Int=4)
  space = _trace_space(; dim, cells, degree)
  u = VectorField(space, components; name=:u)
  problem = AffineProblem(u; operator_class=NonsymmetricOperator())

  for axis in 1:dim, side in (LOWER, UPPER)
    add_boundary_bilinear!(problem, BoundaryFace(axis, side), u, u) do q, v, w
      test_component = component(v)
      trial_component = component(w)
      test_axis = mod1(test_component, dim)
      trial_axis = mod1(trial_component, dim)
      coupling = test_component == trial_component ? 1.7 : -0.2
      coupling * value(v) * value(w) +
      0.18 * normal_gradient(v) * value(w) +
      0.21 * value(v) * normal_gradient(w) +
      0.04 * grad(v)[trial_axis] * grad(w)[test_axis]
    end
  end

  return problem, u
end

function _interface_problem(; dim::Int=2, cells::Int=32, degree::Int=4, components::Int=4)
  space = _trace_space(; dim, cells, degree)
  u = VectorField(space, components; name=:u)
  problem = AffineProblem(u; operator_class=NonsymmetricOperator())

  add_interface_bilinear!(problem, u, u) do q, v, w
    test_component = component(v)
    trial_component = component(w)
    coupling = test_component == trial_component ? 2.0 : 0.3
    coupling * jump(value(v)) * jump(value(w)) +
    0.24 * average(normal_gradient(v)) * jump(value(w)) +
    0.19 * jump(value(v)) * average(normal_gradient(w)) +
    0.05 * inner(jump(grad(v)), average(grad(w)))
  end

  return problem, u
end

function _surface_problem(; dim::Int=2, cells::Int=32, degree::Int=4, components::Int=4)
  space = _trace_space(; dim, cells, degree)
  u = VectorField(space, components; name=:u)
  problem = AffineProblem(u; operator_class=NonsymmetricOperator())
  shape = ntuple(axis -> axis == 1 ? 1 : degree + 2, dim)
  quadrature = Grico.TensorQuadrature(Float64, shape)
  normals = fill(ntuple(axis -> axis == 1 ? 1.0 : 0.0, dim), point_count(quadrature))

  for leaf in active_leaves(field_space(u))
    add_surface_quadrature!(problem, SurfaceQuadrature(leaf, quadrature, normals))
  end

  add_surface_bilinear!(problem, u, u) do q, v, w
    test_component = component(v)
    trial_component = component(w)
    test_axis = mod1(test_component, dim)
    trial_axis = mod1(trial_component, dim)
    coupling = test_component == trial_component ? 1.4 : -0.25
    coupling * value(v) * value(w) +
    0.17 * normal_gradient(v) * value(w) +
    0.16 * value(v) * normal_gradient(w) +
    0.04 * grad(v)[trial_axis] * grad(w)[test_axis]
  end

  return problem, u
end

function _first_boundary_operator(plan)
  isempty(plan.integration.boundary_faces) &&
    throw(ArgumentError("benchmark problem has no boundary faces"))
  isempty(plan.boundary_operators) &&
    throw(ArgumentError("benchmark problem has no boundary operators"))
  return plan.integration.boundary_faces[1], plan.boundary_operators[1].operator
end

function _first_interface_operator(plan)
  isempty(plan.integration.interfaces) &&
    throw(ArgumentError("benchmark problem has no interfaces"))
  isempty(plan.interface_operators) &&
    throw(ArgumentError("benchmark problem has no interface operators"))
  return plan.integration.interfaces[1], plan.interface_operators[1]
end

function _first_surface_operator(plan)
  isempty(plan.integration.embedded_surfaces) &&
    throw(ArgumentError("benchmark problem has no embedded surfaces"))
  isempty(plan.surface_operators) &&
    throw(ArgumentError("benchmark problem has no surface operators"))
  return plan.integration.embedded_surfaces[1], plan.surface_operators[1].operator
end

function _local_apply!(local_output, case_name, operator, item, local_input, scratch)
  if case_name == :boundary_trace_full_tensor
    face_apply!(local_output, operator, item, local_input, scratch)
  elseif case_name == :surface_trace_full_tensor
    surface_apply!(local_output, operator, item, local_input, scratch)
  else
    interface_apply!(local_output, operator, item, local_input, scratch)
  end
end

function _benchmark_operator_case!(rows, case_name, builder, first_operator; dim, cells, degree,
                                   components, repetitions, local_repetitions)
  problem, field = _measure!(rows, case_name, "phase", "problem_setup", 1) do
    builder(; dim, cells, degree, components)
  end
  plan = _measure!(rows, case_name, "phase", "compile", 1) do
    compile(problem)
  end
  T = eltype(coefficients(State(plan)))
  input = _sample_vector(T, Grico.dof_count(plan))
  output = zeros(T, Grico.dof_count(plan))
  workspace = Grico._ReducedOperatorWorkspace(plan)
  reduced_input = _sample_vector(T, Grico.reduced_dof_count(plan))
  reduced_output = zeros(T, length(reduced_input))
  item, operator = first_operator(plan)
  local_input = _sample_vector(T, item.local_dof_count)
  local_output = zeros(T, item.local_dof_count)
  scratch = KernelScratch(T)

  _push_metric!(rows, case_name, "metadata", "active_leaves", active_leaf_count(field_space(field)))
  _push_metric!(rows, case_name, "metadata", "boundary_faces",
                length(plan.integration.boundary_faces))
  _push_metric!(rows, case_name, "metadata", "interfaces", length(plan.integration.interfaces))
  _push_metric!(rows, case_name, "metadata", "embedded_surfaces",
                length(plan.integration.embedded_surfaces))
  _push_metric!(rows, case_name, "metadata", "dofs", Grico.dof_count(plan))
  _push_metric!(rows, case_name, "metadata", "reduced_dofs", Grico.reduced_dof_count(plan))
  _push_metric!(rows, case_name, "metadata", "degree", degree)
  _push_metric!(rows, case_name, "metadata", "components", components)

  apply!(output, plan, input)
  Grico._reduced_apply!(reduced_output, plan, reduced_input, workspace)
  _local_apply!(local_output, case_name, operator, item, local_input, scratch)

  _measure!(rows, case_name, "phase", "apply!", repetitions) do
    apply!(output, plan, input)
  end
  _measure!(rows, case_name, "micro", "reduced_apply", repetitions) do
    Grico._reduced_apply!(reduced_output, plan, reduced_input, workspace)
  end
  _measure!(rows, case_name, "micro", "local_apply", local_repetitions) do
    fill!(local_output, zero(T))
    _local_apply!(local_output, case_name, operator, item, local_input, scratch)
  end

  return rows
end

function _print_help()
  println("""
  trace_weakform_sumfactorization.jl

  Options:
    --output PATH              CSV output path
    --dim N                    spatial dimension (default: 2)
    --cells N                  root cells per axis (default: 32)
    --degree P                 polynomial degree (default: 4)
    --components N             vector components (default: 4)
    --repetitions N            global repetitions (default: 5)
    --local-repetitions N      local kernel repetitions (default: 1000)
    --case NAME                boundary, interface, surface, or all (default: all)
    --help                     show this message
  """)
  return nothing
end

function main(args=ARGS)
  if any(arg -> arg == "--help" || arg == "-h", args)
    _print_help()
    return nothing
  end

  output = _option(args, "--output", TRACE_WEAKFORM_DEFAULT_OUTPUT)
  dim = parse(Int, _option(args, "--dim", "2"))
  cells = parse(Int, _option(args, "--cells", "32"))
  degree = parse(Int, _option(args, "--degree", "4"))
  components = parse(Int, _option(args, "--components", "4"))
  repetitions = parse(Int, _option(args, "--repetitions", "5"))
  local_repetitions = parse(Int, _option(args, "--local-repetitions", "1000"))
  case_option = lowercase(_option(args, "--case", "all"))

  dim >= 1 || throw(ArgumentError("--dim must be positive"))
  cells >= 2 || throw(ArgumentError("--cells must be at least 2"))
  degree >= 1 || throw(ArgumentError("--degree must be positive"))
  components >= 1 || throw(ArgumentError("--components must be positive"))
  repetitions >= 1 || throw(ArgumentError("--repetitions must be positive"))
  local_repetitions >= 1 || throw(ArgumentError("--local-repetitions must be positive"))
  case_option in ("boundary", "interface", "surface", "both", "all") ||
    throw(ArgumentError("--case must be boundary, interface, surface, both, or all"))

  @printf("trace weak-form benchmark: threads=%d dim=%d cells=%d degree=%d components=%d case=%s\n",
          Threads.nthreads(), dim, cells, degree, components, case_option)
  rows = NamedTuple[]

  if case_option in ("boundary", "both", "all")
    _benchmark_operator_case!(rows, :boundary_trace_full_tensor, _boundary_problem,
                              _first_boundary_operator; dim, cells, degree, components, repetitions,
                              local_repetitions)
  end

  if case_option in ("interface", "both", "all")
    _benchmark_operator_case!(rows, :interface_trace_full_tensor, _interface_problem,
                              _first_interface_operator; dim, cells, degree, components,
                              repetitions, local_repetitions)
  end

  if case_option in ("surface", "all")
    _benchmark_operator_case!(rows, :surface_trace_full_tensor, _surface_problem,
                              _first_surface_operator; dim, cells, degree, components, repetitions,
                              local_repetitions)
  end

  path = _write_csv(output, rows)
  println("wrote $path")
  return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

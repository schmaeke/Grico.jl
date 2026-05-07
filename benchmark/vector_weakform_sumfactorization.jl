# Vector-valued weak-form sum-factorization benchmark.
#
# The script measures a component-coupled vector cell bilinear form on a full
# tensor-product basis. It is meant to isolate the performance effect of the
# weak-form lowering itself: local cell apply, global apply!, and reduced apply.
#
# Example helios run:
#
#   JULIA_NUM_THREADS=16 julia --project=benchmark benchmark/vector_weakform_sumfactorization.jl

const VECTOR_WEAKFORM_ROOT = normpath(get(ENV, "GRICO_BENCHMARK_ROOT", joinpath(@__DIR__, "..")))
VECTOR_WEAKFORM_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, VECTOR_WEAKFORM_ROOT)

using Grico
using Printf

const VECTOR_WEAKFORM_DEFAULT_OUTPUT = joinpath(@__DIR__, "output",
                                                "vector_weakform_sumfactorization.csv")

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
  resolved = _resolve_output_path(path, basename(VECTOR_WEAKFORM_DEFAULT_OUTPUT))
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
    values[index] = T(sin(0.011 * index) + 0.2 * cos(0.037 * index))
  end

  return values
end

function _basis_family(name::AbstractString)
  lowered = lowercase(strip(name))
  lowered == "full" && return FullTensorBasis()
  lowered == "trunk" && return TrunkBasis()
  throw(ArgumentError("--basis must be full or trunk"))
end

function _vector_problem(; dim::Int=2, cells::Int=32, degree::Int=4, components::Int=4,
                         basis_name::AbstractString="full")
  origin = ntuple(_ -> 0.0, dim)
  widths = ntuple(_ -> 1.0, dim)
  domain = Domain(origin, widths, ntuple(_ -> cells, dim))
  space = HpSpace(domain,
                  SpaceOptions(basis=_basis_family(basis_name), degree=UniformDegree(degree),
                               continuity=:dg))
  u = VectorField(space, components; name=:u)
  problem = AffineProblem(u; operator_class=NonsymmetricOperator())

  add_cell_bilinear!(problem, u, u) do q, v, w
    test_component = component(v)
    trial_component = component(w)
    test_axis = mod1(test_component, dim)
    trial_axis = mod1(trial_component, dim)
    mass_scale = test_component == trial_component ? 2.0 : -0.15
    diffusion_scale = test_component == trial_component ? 1.1 : 0.35
    mass_scale * value(v) * value(w) +
    diffusion_scale * inner(grad(v), grad(w)) +
    0.05 * grad(v)[trial_axis] * value(w) +
    0.07 * value(v) * grad(w)[test_axis]
  end

  return problem, u
end

function _first_cell_operator(plan)
  isempty(plan.integration.cells) && throw(ArgumentError("benchmark problem has no cells"))
  isempty(plan.cell_operators) && throw(ArgumentError("benchmark problem has no cell operators"))
  return plan.integration.cells[1], plan.cell_operators[1]
end

function _benchmark_case!(rows; dim, cells, degree, components, basis_name, repetitions,
                          local_repetitions)
  case_name = lowercase(basis_name) == "trunk" ? :vector_weakform_trunk_tensor_box :
              :vector_weakform_full_tensor
  problem, field = _measure!(rows, case_name, "phase", "problem_setup", 1) do
    _vector_problem(; dim, cells, degree, components, basis_name)
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
  item, operator = _first_cell_operator(plan)
  local_input = _sample_vector(T, length(input) ÷ active_leaf_count(field_space(field)))
  local_output = zeros(T, length(local_input))
  scratch = KernelScratch(T)

  _push_metric!(rows, case_name, "metadata", "active_leaves", active_leaf_count(field_space(field)))
  _push_metric!(rows, case_name, "metadata", "dofs", Grico.dof_count(plan))
  _push_metric!(rows, case_name, "metadata", "reduced_dofs", Grico.reduced_dof_count(plan))
  _push_metric!(rows, case_name, "metadata", "degree", degree)
  _push_metric!(rows, case_name, "metadata", "components", components)

  apply!(output, plan, input)
  Grico._reduced_apply!(reduced_output, plan, reduced_input, workspace)
  cell_apply!(local_output, operator, item, local_input, scratch)

  _measure!(rows, case_name, "phase", "apply!", repetitions) do
    apply!(output, plan, input)
  end
  _measure!(rows, case_name, "micro", "reduced_apply", repetitions) do
    Grico._reduced_apply!(reduced_output, plan, reduced_input, workspace)
  end
  _measure!(rows, case_name, "micro", "local_cell_apply", local_repetitions) do
    fill!(local_output, zero(T))
    cell_apply!(local_output, operator, item, local_input, scratch)
  end

  return rows
end

function _print_help()
  println("""
  vector_weakform_sumfactorization.jl

  Options:
    --output PATH              CSV output path
    --dim N                    spatial dimension (default: 2)
    --cells N                  root cells per axis (default: 32)
    --degree P                 polynomial degree (default: 4)
    --components N             vector components (default: 4)
    --basis NAME               full or trunk (default: full)
    --repetitions N            global repetitions (default: 5)
    --local-repetitions N      local kernel repetitions (default: 1000)
    --help                     show this message
  """)
  return nothing
end

function main(args=ARGS)
  if any(arg -> arg == "--help" || arg == "-h", args)
    _print_help()
    return nothing
  end

  output = _option(args, "--output", VECTOR_WEAKFORM_DEFAULT_OUTPUT)
  dim = parse(Int, _option(args, "--dim", "2"))
  cells = parse(Int, _option(args, "--cells", "32"))
  degree = parse(Int, _option(args, "--degree", "4"))
  components = parse(Int, _option(args, "--components", "4"))
  basis_name = lowercase(_option(args, "--basis", "full"))
  repetitions = parse(Int, _option(args, "--repetitions", "5"))
  local_repetitions = parse(Int, _option(args, "--local-repetitions", "1000"))

  dim >= 1 || throw(ArgumentError("--dim must be positive"))
  cells >= 1 || throw(ArgumentError("--cells must be positive"))
  degree >= 1 || throw(ArgumentError("--degree must be positive"))
  components >= 1 || throw(ArgumentError("--components must be positive"))
  basis_name in ("full", "trunk") || throw(ArgumentError("--basis must be full or trunk"))
  repetitions >= 1 || throw(ArgumentError("--repetitions must be positive"))
  local_repetitions >= 1 || throw(ArgumentError("--local-repetitions must be positive"))

  @printf("vector weak-form benchmark: threads=%d dim=%d cells=%d degree=%d components=%d basis=%s\n",
          Threads.nthreads(), dim, cells, degree, components, basis_name)
  rows = NamedTuple[]
  _benchmark_case!(rows; dim, cells, degree, components, basis_name, repetitions, local_repetitions)
  path = _write_csv(output, rows)
  println("wrote $path")
  return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

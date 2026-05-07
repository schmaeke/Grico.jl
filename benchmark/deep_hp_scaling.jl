# Deep local hp multigrid benchmark.
#
# This script stresses the hierarchy with a single deeply refined dyadic chain
# and deliberately nonuniform polynomial degrees. It complements uniform
# `hpmg_scaling.jl` by exposing traversal balance, hanging-interface density,
# transfer construction, and coarse-level behavior on locally graded meshes.

const DEEP_HP_ROOT = normpath(joinpath(@__DIR__, ".."))
DEEP_HP_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, DEEP_HP_ROOT)

using Grico
using LinearAlgebra
using Printf

const DEEP_HP_DEFAULT_OUTPUT = joinpath(@__DIR__, "output", "deep_hp_scaling.csv")

function _option(args, name, default)
  prefix = string(name, "=")

  for (index, arg) in pairs(args)
    arg == name && return index < length(args) ? args[index+1] : default
    startswith(arg, prefix) && return split(arg, "=", limit=2)[2]
  end

  return default
end

function _resolve_output_path(path::AbstractString, default_file::AbstractString)
  lowercase(splitext(path)[2]) == ".csv" && return path
  return joinpath(path, default_file)
end

function _csv_escape(value)
  text = replace(string(value), "\"" => "\"\"")
  return any(character -> character in (',', '"', '\n', '\r'), text) ? "\"$text\"" : text
end

function _write_csv(path::AbstractString, rows::Vector{NamedTuple})
  resolved = _resolve_output_path(path, basename(DEEP_HP_DEFAULT_OUTPUT))
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

function _refine_deep_chain!(domain::Domain{D}, depth::Int) where {D}
  grid_data = grid(domain)
  leaf = first(active_leaves(grid_data))

  for step in 1:depth
    axis = mod1(step, D)
    first_child = refine!(grid_data, leaf, axis)
    leaf = first_child + 1
  end

  return domain
end

function _deep_hp_problem(; dim::Int=2, root_cells::Int=1, depth::Int=10, degree::Int=5)
  domain = Domain(ntuple(_ -> 0.0, dim), ntuple(_ -> 1.0, dim), ntuple(_ -> root_cells, dim))
  _refine_deep_chain!(domain, depth)
  degree_policy = ByLeafDegrees() do current_domain, leaf
    grid_data = grid(current_domain)
    leaf_level = maximum(Grico.level(grid_data, leaf, axis) for axis in 1:dim)
    tangential_degree = max(1, degree - min(leaf_level, max(degree - 1, 0)))
    return ntuple(axis -> axis == mod1(leaf_level + 1, dim) ? degree : tangential_degree, dim)
  end
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=degree_policy, continuity=:cg))
  u = ScalarField(space; name=:u)
  problem = AffineProblem(u; operator_class=SPD())

  add_cell_bilinear!(problem, u, u) do q, v, w
    inner(grad(v), grad(w)) + value(v) * value(w)
  end
  add_cell_linear!(problem, u) do q, v
    value(v)
  end

  return problem, u
end

function _relative_residual(operator, values, rhs)
  tmp = similar(rhs)
  Grico._apply_operator!(tmp, operator, values)

  @inbounds for index in eachindex(tmp)
    tmp[index] -= rhs[index]
  end

  return norm(tmp) / max(norm(rhs), one(eltype(rhs)))
end

function _benchmark_case!(rows; dim, root_cells, depth, degree, repetitions, rtol, maxiter)
  case_name = :deep_local_hp_gmg
  problem, field = _measure!(rows, case_name, "phase", "problem_setup", 1) do
    _deep_hp_problem(; dim, root_cells, depth, degree)
  end
  policy = GeometricMultigridPreconditioner(; p_sequence=:bisect, max_levels=32)
  hierarchy = _measure!(rows, case_name, "phase", "compile_hpmg", 1) do
    Grico._compile_geometric_multigrid(problem, policy)
  end
  finest = hierarchy.levels[end]
  rhs_data = zeros(eltype(coefficients(State(finest.plan))), Grico.reduced_dof_count(finest.plan))
  Grico._reduced_rhs!(rhs_data, finest.plan, finest.workspace)
  solution = zeros(eltype(rhs_data), length(rhs_data))

  _push_metric!(rows, case_name, "metadata", "active_leaves", active_leaf_count(field_space(field)))
  _push_metric!(rows, case_name, "metadata", "dofs", Grico.dof_count(finest.plan))
  _push_metric!(rows, case_name, "metadata", "reduced_dofs", length(rhs_data))
  _push_metric!(rows, case_name, "metadata", "levels", length(hierarchy.levels))
  _push_metric!(rows, case_name, "metadata", "root_cells", root_cells)
  _push_metric!(rows, case_name, "metadata", "finest_degree", degree)
  _push_metric!(rows, case_name, "metadata", "depth", depth)

  _measure!(rows, case_name, "micro", "hpmg_vcycle", repetitions) do
    Grico._apply_preconditioner!(solution, hierarchy, rhs_data)
  end

  solve_ref = Ref{Any}(nothing)
  success = Ref(true)
  message = Ref("")
  timing = @timed begin
    try
      solve_ref[] = Grico._cg_solve(finest.operator, rhs_data, hierarchy;
                                    relative_tolerance=eltype(rhs_data)(rtol),
                                    absolute_tolerance=zero(eltype(rhs_data)), maxiter,
                                    initial_solution=nothing)
    catch error
      success[] = false
      message[] = sprint(showerror, error)
    end
  end
  residual = success[] ? _relative_residual(finest.operator, solve_ref[], rhs_data) : NaN
  _push_row!(rows, case_name, "phase", "cg_gmg_solve", 1, timing; value=residual,
             note=success[] ? "" : message[])
  _push_metric!(rows, case_name, "metadata", "solve_succeeded", success[] ? 1.0 : 0.0)
  return rows
end

function _print_help()
  println("""
  deep_hp_scaling.jl

  Options:
    --output PATH          CSV file or output directory
    --dim N                spatial dimension (default: 2)
    --root-cells N         root cells per axis (default: 1)
    --depth N              local refinement chain depth (default: 10)
    --degree P             maximum polynomial degree (default: 5)
    --repetitions N        V-cycle repetitions (default: 5)
    --rtol VALUE           solve relative tolerance (default: 1e-8)
    --maxiter N            maximum CG iterations (default: 300)
    --help                 show this message
  """)
  return nothing
end

function main(args=ARGS)
  if any(arg -> arg == "--help" || arg == "-h", args)
    _print_help()
    return nothing
  end

  output = _option(args, "--output", DEEP_HP_DEFAULT_OUTPUT)
  dim = parse(Int, _option(args, "--dim", "2"))
  root_cells = parse(Int, _option(args, "--root-cells", "1"))
  depth = parse(Int, _option(args, "--depth", "10"))
  degree = parse(Int, _option(args, "--degree", "5"))
  repetitions = parse(Int, _option(args, "--repetitions", "5"))
  rtol = parse(Float64, _option(args, "--rtol", "1e-8"))
  maxiter = parse(Int, _option(args, "--maxiter", "300"))

  dim >= 1 || throw(ArgumentError("--dim must be positive"))
  root_cells >= 1 || throw(ArgumentError("--root-cells must be positive"))
  depth >= 1 || throw(ArgumentError("--depth must be positive"))
  degree >= 1 || throw(ArgumentError("--degree must be positive"))
  repetitions >= 1 || throw(ArgumentError("--repetitions must be positive"))
  rtol > 0 || throw(ArgumentError("--rtol must be positive"))
  maxiter >= 1 || throw(ArgumentError("--maxiter must be positive"))

  @printf("deep hp benchmark: threads=%d dim=%d root_cells=%d depth=%d degree=%d\n",
          Threads.nthreads(), dim, root_cells, depth, degree)
  rows = NamedTuple[]
  _benchmark_case!(rows; dim, root_cells, depth, degree, repetitions, rtol, maxiter)
  path = _write_csv(output, rows)
  println("wrote $path")
  return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

# hp geometric multigrid scaling benchmark.
#
# The production CPU target is the physical-core count on helios:
#
#   JULIA_NUM_THREADS=16 julia --project=benchmark benchmark/hpmg_scaling.jl
#
# Use 32 threads only as an explicit SMT experiment. The script records the
# active Julia thread count in every row, so sweeps are best run by launching one
# Julia process per thread count.

const HPMG_BENCHMARK_ROOT = normpath(joinpath(@__DIR__, ".."))
HPMG_BENCHMARK_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, HPMG_BENCHMARK_ROOT)

using Grico
using LinearAlgebra
using Printf

const HPMG_DEFAULT_OUTPUT = joinpath(@__DIR__, "output", "hpmg_scaling.csv")

function _option(args, name, default)
  prefix = string(name, "=")

  for (index, arg) in pairs(args)
    arg == name && return index < length(args) ? args[index+1] : default
    startswith(arg, prefix) && return split(arg, "=", limit=2)[2]
  end

  return default
end

function _parse_bool(text)
  lowered = lowercase(strip(String(text)))
  lowered in ("true", "yes", "1", "on") && return true
  lowered in ("false", "no", "0", "off") && return false
  throw(ArgumentError("expected a boolean value, got $text"))
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
    values[index] = T(sin(0.013 * index) + 0.25 * cos(0.037 * index))
  end

  return values
end

function _refine_leaf_all_axes!(grid_data, leaf::Int, ::Val{D}) where {D}
  leaves = Int[leaf]

  for axis in 1:D
    next_leaves = Int[]
    sizehint!(next_leaves, 2 * length(leaves))

    for current in leaves
      first_child = refine!(grid_data, current, axis)
      push!(next_leaves, first_child, first_child + 1)
    end

    leaves = next_leaves
  end

  return leaves
end

function _uniform_refine!(domain::Domain{D}, depth::Int) where {D}
  grid_data = grid(domain)

  for _ in 1:depth
    leaves = copy(active_leaves(grid_data))

    for leaf in leaves
      _refine_leaf_all_axes!(grid_data, leaf, Val(D))
    end
  end

  return domain
end

function _laplace_mass_problem(; dim::Int=2, depth::Int=6, degree::Int=4, nonsymmetric::Bool=false)
  origin = ntuple(_ -> 0.0, dim)
  widths = ntuple(_ -> 1.0, dim)
  domain = Domain(origin, widths, ntuple(_ -> 1, dim))
  _uniform_refine!(domain, depth)
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(degree),
                               continuity=:cg))
  u = ScalarField(space; name=nonsymmetric ? :w : :u)
  problem = AffineProblem(u; operator_class=nonsymmetric ? NonsymmetricOperator() : SPD())

  if nonsymmetric
    add_cell_bilinear!(problem, u, u) do q, v, w
      inner(grad(v), grad(w)) + value(v) * value(w) + 0.05 * value(v) * grad(w)[1]
    end
  else
    add_cell_bilinear!(problem, u, u) do q, v, w
      inner(grad(v), grad(w)) + value(v) * value(w)
    end
  end

  add_cell_linear!(problem, u) do q, v
    value(v)
  end

  return problem
end

function _relative_residual(operator, values, rhs)
  tmp = similar(rhs)
  Grico._apply_operator!(tmp, operator, values)

  @inbounds for index in eachindex(tmp)
    tmp[index] -= rhs[index]
  end

  return norm(tmp) / max(norm(rhs), one(eltype(rhs)))
end

function _time_solve!(rows, case_name, hierarchy, rhs; nonsymmetric::Bool, rtol, maxiter)
  finest = hierarchy.levels[end]
  counter = Ref(0)
  operator = Grico._CountingReducedOperator(finest.operator, counter)
  success = Ref(true)
  message = Ref("")
  values_ref = Ref{Any}(nothing)
  T = eltype(rhs)

  GC.gc()
  timing = @timed begin
    try
      if nonsymmetric
        values_ref[] = Grico._fgmres_solve(operator, rhs, hierarchy; restart=30,
                                           relative_tolerance=T(rtol), absolute_tolerance=zero(T),
                                           maxiter, initial_solution=nothing)
      else
        values_ref[] = Grico._cg_solve(operator, rhs, hierarchy; relative_tolerance=T(rtol),
                                       absolute_tolerance=zero(T), maxiter,
                                       initial_solution=nothing)
      end
    catch error
      success[] = false
      message[] = sprint(showerror, error)
    end
  end

  residual = success[] ? _relative_residual(finest.operator, values_ref[], rhs) : NaN
  note = success[] ? "" : message[]
  component = nonsymmetric ? "fgmres_gmg_solve" : "cg_gmg_solve"
  _push_row!(rows, case_name, "phase", component, 1, timing; value=residual, note)
  _push_metric!(rows, case_name, "metadata", "outer_operator_applications", counter[])
  _push_metric!(rows, case_name, "metadata", "solve_succeeded", success[] ? 1.0 : 0.0)
  return values_ref[]
end

function _warm_solve!(hierarchy, rhs; nonsymmetric::Bool, rtol)
  finest = hierarchy.levels[end]
  counter = Ref(0)
  operator = Grico._CountingReducedOperator(finest.operator, counter)
  T = eltype(rhs)

  try
    if nonsymmetric
      Grico._fgmres_solve(operator, rhs, hierarchy; restart=30, relative_tolerance=T(rtol),
                          absolute_tolerance=zero(T), maxiter=1, initial_solution=nothing)
    else
      Grico._cg_solve(operator, rhs, hierarchy; relative_tolerance=T(rtol),
                      absolute_tolerance=zero(T), maxiter=1, initial_solution=nothing)
    end
  catch error
    error isa ArgumentError || rethrow()
  end

  return nothing
end

function _benchmark_case!(rows::Vector{NamedTuple}, case_name; dim, depth, degree, nonsymmetric,
                          repetitions, rtol, maxiter)
  problem = _measure!(rows, case_name, "phase", "problem_setup", 1) do
    _laplace_mass_problem(; dim, depth, degree, nonsymmetric)
  end

  mg_policy = GeometricMultigridPreconditioner(; p_sequence=:bisect, coarse_direct_dof_limit=512,
                                               pre_smoothing_steps=2, post_smoothing_steps=2)
  hierarchy = _measure!(rows, case_name, "phase", "compile_hpmg", 1) do
    Grico._compile_geometric_multigrid(problem, mg_policy)
  end

  finest = hierarchy.levels[end]
  T = eltype(finest.residual)
  rhs = zeros(T, Grico.reduced_dof_count(finest.plan))
  Grico._reduced_rhs!(rhs, finest.plan, finest.workspace)
  level_sizes = join((Grico.reduced_dof_count(level.plan) for level in hierarchy.levels), ";")
  _push_metric!(rows, case_name, "metadata", "levels", length(hierarchy.levels); note=level_sizes)
  _push_metric!(rows, case_name, "metadata", "fine_dofs", Grico.dof_count(finest.plan))
  _push_metric!(rows, case_name, "metadata", "fine_reduced_dofs",
                Grico.reduced_dof_count(finest.plan))
  finest_field = first(fields(field_layout(State(finest.plan))))
  _push_metric!(rows, case_name, "metadata", "active_leaves",
                active_leaf_count(field_space(finest_field)))

  full_input = _sample_vector(T, Grico.dof_count(finest.plan))
  full_output = zeros(T, length(full_input))
  reduced_input = _sample_vector(T, length(rhs))
  reduced_output = zeros(T, length(rhs))
  vcycle_rhs = _sample_vector(T, length(rhs))
  vcycle_output = zeros(T, length(rhs))

  apply!(full_output, finest.plan, full_input)
  Grico._reduced_apply!(reduced_output, finest.plan, reduced_input, finest.workspace)
  Grico._apply_preconditioner!(vcycle_output, hierarchy, vcycle_rhs)
  _warm_solve!(hierarchy, rhs; nonsymmetric, rtol)

  _measure!(rows, case_name, "phase", "apply!", repetitions) do
    apply!(full_output, finest.plan, full_input)
  end
  _measure!(rows, case_name, "micro", "reduced_apply", repetitions) do
    Grico._reduced_apply!(reduced_output, finest.plan, reduced_input, finest.workspace)
  end
  _measure!(rows, case_name, "phase", "hpmg_vcycle", repetitions) do
    Grico._apply_preconditioner!(vcycle_output, hierarchy, vcycle_rhs)
  end
  _time_solve!(rows, case_name, hierarchy, rhs; nonsymmetric, rtol, maxiter)
  return rows
end

function _warmup!()
  rows = NamedTuple[]
  _benchmark_case!(rows, :warmup_spd_2d; dim=2, depth=2, degree=2, nonsymmetric=false,
                   repetitions=1, rtol=1.0e-6, maxiter=50)
  _benchmark_case!(rows, :warmup_nonsym_2d; dim=2, depth=2, degree=2, nonsymmetric=true,
                   repetitions=1, rtol=1.0e-6, maxiter=50)
  return nothing
end

function _print_help()
  println("""
  hpmg_scaling.jl

  Options:
    --output PATH       CSV output path
    --depth N           uniform refinement depth from one root cell per axis (default: 6)
    --degree P          polynomial degree (default: 4)
    --repetitions N     repetitions for apply/V-cycle phases (default: 5)
    --rtol TOL          Krylov relative tolerance (default: 1.0e-8)
    --maxiter N         Krylov iteration cap (default: 300)
    --nonsym BOOL       also run nonsymmetric FGMRES+GMG case (default: true)
    --no-warmup         include first-use method compilation in measured phases
    --help              show this message

  Launch with JULIA_NUM_THREADS=16 for the helios physical-core target. Use 32
  threads only when intentionally measuring SMT behavior.
  """)
  return nothing
end

function main(args=ARGS)
  if any(arg -> arg == "--help" || arg == "-h", args)
    _print_help()
    return nothing
  end

  output = _option(args, "--output", HPMG_DEFAULT_OUTPUT)
  depth = parse(Int, _option(args, "--depth", "6"))
  degree = parse(Int, _option(args, "--degree", "4"))
  repetitions = parse(Int, _option(args, "--repetitions", "5"))
  rtol = parse(Float64, _option(args, "--rtol", "1.0e-8"))
  maxiter = parse(Int, _option(args, "--maxiter", "300"))
  run_nonsym = _parse_bool(_option(args, "--nonsym", "true"))
  warmup = !any(arg -> arg == "--no-warmup", args)

  depth >= 0 || throw(ArgumentError("--depth must be nonnegative"))
  degree >= 1 || throw(ArgumentError("--degree must be positive"))
  repetitions >= 1 || throw(ArgumentError("--repetitions must be positive"))
  maxiter >= 1 || throw(ArgumentError("--maxiter must be positive"))
  rtol >= 0 || throw(ArgumentError("--rtol must be nonnegative"))

  @printf("hpMG scaling: threads=%d depth=%d degree=%d repetitions=%d\n", Threads.nthreads(), depth,
          degree, repetitions)
  warmup && _warmup!()

  rows = NamedTuple[]
  _benchmark_case!(rows, :spd_cg_hpmg_2d; dim=2, depth, degree, nonsymmetric=false, repetitions,
                   rtol, maxiter)

  if run_nonsym
    _benchmark_case!(rows, :nonsym_fgmres_hpmg_2d; dim=2, depth, degree, nonsymmetric=true,
                     repetitions, rtol, maxiter)
  end

  path = _write_csv(output, rows)
  println("wrote $path")
  return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

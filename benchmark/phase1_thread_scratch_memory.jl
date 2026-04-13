#!/usr/bin/env julia

const BENCHMARK_PROJECT = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_PROJECT, ".."))
pushfirst!(LOAD_PATH, REPO_ROOT)

using Dates
using Printf
using TOML

using Grico

const DEFAULT_INPUT = joinpath(BENCHMARK_PROJECT, "phase1_threaded_assembly_memory.toml")
const DEFAULT_OUTPUT = joinpath(BENCHMARK_PROJECT, "phase1_thread_scratch_memory.toml")
const DEFAULT_REPORT = joinpath(BENCHMARK_PROJECT, "phase1_thread_scratch_memory.md")
const DEFAULT_THREAD_COUNTS = [1, 2, 4, 6, 16]
const DEFAULT_PROJECTED_RHS_DOFS = [10^5, 10^6, 10^7]

function _parse_args(args)
  parsed = Dict{String,String}()

  for arg in args
    startswith(arg, "--") || throw(ArgumentError("unsupported argument `$arg`"))
    key, value = occursin("=", arg) ? split(arg[3:end], "="; limit=2) : (arg[3:end], "true")
    parsed[key] = value
  end

  return parsed
end

function _parse_int_list(raw, default)
  raw === nothing && return copy(default)
  return sort!(unique!(parse.(Int, split(raw, ","))))
end

function _case_entries(artifact)
  first_run = first(artifact["thread_runs"])
  entries = Dict{String,Any}[]

  for case_entry in first_run["cases"]
    metadata = case_entry["metadata"]
    operations = Set(operation["id"] for operation in case_entry["operations"])
    full_dofs = get(metadata, "full_dofs", nothing)
    max_local_dofs = get(metadata, "max_local_dofs", nothing)

    if full_dofs === nothing || max_local_dofs === nothing
      continue
    end

    if "assemble" in operations
      rhs_target_dofs = get(metadata, "reduced_dofs", full_dofs)
      push!(entries,
            Dict("case_id" => case_entry["id"],
                 "case_label" => case_entry["label"],
                 "operation_id" => "assemble",
                 "rhs_target_dofs" => rhs_target_dofs,
                 "max_local_dofs" => max_local_dofs))
    end

    if "residual_bang" in operations
      push!(entries,
            Dict("case_id" => case_entry["id"],
                 "case_label" => case_entry["label"],
                 "operation_id" => "residual_bang",
                 "rhs_target_dofs" => full_dofs,
                 "max_local_dofs" => max_local_dofs))
    end
  end

  sort!(entries; by=entry -> (entry["case_id"], entry["operation_id"]))
  return entries
end

function _entry_metrics(entry)
  local_dofs = Int(entry["max_local_dofs"])
  rhs_target_dofs = Int(entry["rhs_target_dofs"])
  scratch = Grico._ThreadScratch(Float64, local_dofs)
  new_scratch_bytes = Base.summarysize(scratch)
  removed_dense_rhs_bytes = sizeof(Float64) * rhs_target_dofs
  return Dict("case_id" => entry["case_id"],
              "case_label" => entry["case_label"],
              "operation_id" => entry["operation_id"],
              "rhs_target_dofs" => rhs_target_dofs,
              "max_local_dofs" => local_dofs,
              "new_scratch_bytes" => new_scratch_bytes,
              "removed_dense_rhs_bytes" => removed_dense_rhs_bytes)
end

function _projected_metrics(rhs_target_dofs::Int)
  return Dict("rhs_target_dofs" => rhs_target_dofs,
              "removed_dense_rhs_bytes" => sizeof(Float64) * rhs_target_dofs)
end

function _format_bytes(bytes::Integer)
  value = float(bytes)

  if value >= 1024^3
    return @sprintf("%.2f GiB", value / 1024^3)
  elseif value >= 1024^2
    return @sprintf("%.2f MiB", value / 1024^2)
  elseif value >= 1024
    return @sprintf("%.2f KiB", value / 1024)
  end

  return @sprintf("%d B", bytes)
end

function _write_toml(path::AbstractString, data)
  mkpath(dirname(path))
  open(path, "w") do io
    TOML.print(io, data)
  end
  return path
end

function _render_markdown(results)
  lines = String[]
  thread_counts = [Int(value) for value in results["thread_counts"]]
  push!(lines, "# Phase 1 Threaded Assembly Memory")
  push!(lines, "")
  push!(lines, "Generated: `$(results["generated_utc"])`")
  push!(lines, "")
  push!(lines, "Input artifact: `$(basename(results["input_artifact"]))`")
  push!(lines, "")
  push!(lines, "## Summary")
  push!(lines, "")
  push!(lines,
        "- `_ThreadScratch` no longer stores one dense global RHS vector per worker.")
  push!(lines,
        "- Persistent scratch size now depends on `max_local_dofs`, not on the global RHS target length.")
  push!(lines,
        "- The new `rhs_rows` / `rhs_values` buffers are sparse chunk-local accumulators that are flushed after each claimed chunk instead of staying dense for the full solve space.")
  push!(lines, "")
  push!(lines, "## Measured Case-Level Effect")
  push!(lines, "")
  header = ["Case / operation", "RHS target dofs", "Max local dofs", "New scratch per worker",
            "Dense RHS removed per worker"]
  append!(header, ["Dense RHS removed at $(threads) threads" for threads in thread_counts])
  push!(lines, "| $(join(header, " | ")) |")
  separator = vcat(["---", "---:", "---:", "---:", "---:"], ["---:" for _ in thread_counts])
  push!(lines, "| $(join(separator, " | ")) |")

  for entry in results["case_metrics"]
    removed = entry["removed_dense_rhs_bytes"]
    row = ["`$(entry["case_id"]) / $(entry["operation_id"])`", string(entry["rhs_target_dofs"]),
           string(entry["max_local_dofs"]), _format_bytes(entry["new_scratch_bytes"]),
           _format_bytes(removed)]
    append!(row, [_format_bytes(threads * removed) for threads in thread_counts])
    push!(lines, "| $(join(row, " | ")) |")
  end

  push!(lines, "")
  push!(lines, "## Projected Dense RHS Overhead Removed")
  push!(lines, "")
  projection_header = ["RHS target dofs"]
  append!(projection_header, ["Removed at $(threads) thread$(threads == 1 ? "" : "s")"
                              for threads in thread_counts])
  push!(lines, "| $(join(projection_header, " | ")) |")
  projection_separator = vcat(["---:"], ["---:" for _ in thread_counts])
  push!(lines, "| $(join(projection_separator, " | ")) |")

  for projection in results["projected_metrics"]
    removed = projection["removed_dense_rhs_bytes"]
    row = [string(projection["rhs_target_dofs"])]
    append!(row, [_format_bytes(threads * removed) for threads in thread_counts])
    push!(lines, "| $(join(row, " | ")) |")
  end

  push!(lines, "")
  push!(lines, "## Interpretation")
  push!(lines, "")
  push!(lines,
        "- On the current Phase 0 benchmark cases, the removed dense RHS vector is small compared with the total COO and sparse-construction allocation volume, so median per-call memory barely moves.")
  push!(lines,
        "- The architectural gain is that the persistent RHS path is no longer `O(thread_count × rhs_target_dofs)`. Baseline thread-local scratch now scales with the local integration size, while temporary RHS storage scales with recently claimed work and is cleared at chunk boundaries.")
  push!(lines,
        "- This is a prerequisite for larger one-node runs, but it does not replace the need for Phase 2. The remaining dominant memory traffic still comes from COO triplet growth and `sparse(...)` reconstruction.")
  return join(lines, "\n")
end

function main(args=ARGS)
  parsed = _parse_args(args)
  input_path = get(parsed, "input", DEFAULT_INPUT)
  output_path = get(parsed, "output", DEFAULT_OUTPUT)
  report_path = get(parsed, "report", DEFAULT_REPORT)
  thread_counts = _parse_int_list(get(parsed, "threads", nothing), DEFAULT_THREAD_COUNTS)
  projected_rhs_dofs = _parse_int_list(get(parsed, "rhs-dofs", nothing),
                                       DEFAULT_PROJECTED_RHS_DOFS)
  artifact = TOML.parsefile(input_path)
  case_metrics = [_entry_metrics(entry) for entry in _case_entries(artifact)]
  projected_metrics = [_projected_metrics(rhs_target_dofs) for rhs_target_dofs in projected_rhs_dofs]
  results = Dict("generated_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
                 "input_artifact" => abspath(input_path),
                 "thread_counts" => thread_counts,
                 "case_metrics" => case_metrics,
                 "projected_metrics" => projected_metrics)
  _write_toml(output_path, results)

  open(report_path, "w") do io
    write(io, _render_markdown(results))
  end

  println("wrote phase 1 scratch-memory artifact to $output_path")
  println("wrote phase 1 scratch-memory report to $report_path")
  return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  main()
end

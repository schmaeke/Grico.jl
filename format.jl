#!/usr/bin/env julia

const PROJECT_ROOT = @__DIR__
const SKIP_DIRECTORIES = Set([".git", "old", "tmp", "output"])
const _FORMATTER_ERROR = Ref{Any}(nothing)
const _FORMATTER_AVAILABLE = try
  @eval using JuliaFormatter
  true
catch error
  _FORMATTER_ERROR[] = error
  false
end

function _parse_check_mode(arguments)
  isempty(arguments) && return false
  length(arguments) == 1 && arguments[1] == "--check" && return true
  throw(ArgumentError("usage: julia format.jl [--check]"))
end

function _formatter_files(root::AbstractString)
  files = String[]

  for (current_root, dirs, names) in walkdir(root)
    filter!(dir -> !(dir in SKIP_DIRECTORIES), dirs)

    for name in sort!(collect(names))
      endswith(name, ".jl") || continue
      push!(files, joinpath(current_root, name))
    end
  end

  return sort!(files)
end

function _check_formatted(path::AbstractString)
  original = read(path, String)

  mktemp() do temp_path, io
    write(io, original)
    close(io)
    JuliaFormatter.format(temp_path; overwrite=true, verbose=false)
    return original == read(temp_path, String)
  end
end

function main()
  if !_FORMATTER_AVAILABLE
    println(stderr, "JuliaFormatter is not available.")
    println(stderr, "Install it in your default Julia environment or this project, then rerun:")
    println(stderr, "  julia -e 'import Pkg; Pkg.add(\"JuliaFormatter\")'")
    println(stderr, "Original error: ", sprint(showerror, _FORMATTER_ERROR[]))
    return 1
  end

  check_only = _parse_check_mode(ARGS)
  files = _formatter_files(PROJECT_ROOT)
  isempty(files) && return 0

  if check_only
    unformatted = String[]

    for path in files
      _check_formatted(path) || push!(unformatted, path)
    end

    if !isempty(unformatted)
      println(stderr, "The following files are not correctly formatted:")

      for path in unformatted
        println(stderr, "  ", relpath(path, PROJECT_ROOT))
      end

      println(stderr, "Run `julia format.jl` to format the project.")
      return 1
    end

    println("Formatting check passed.")
    return 0
  end

  for path in files
    JuliaFormatter.format(path; overwrite=true, verbose=true)
  end

  return 0
end

exit(main())

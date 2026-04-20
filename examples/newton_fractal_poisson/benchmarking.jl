# Benchmark and validation code include this wrapper when they need the example
# definitions without executing the driver. Running `driver.jl` directly remains
# the user-facing entry point.

Base.include(@__MODULE__, joinpath(@__DIR__, "driver.jl"))

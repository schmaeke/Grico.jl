# This file is only for benchmark and validation harnesses.
#
# It is deliberately not part of the instructional example. The actual example
# starts in `driver.jl`; this wrapper only loads the same definitions so
# benchmark code can call the setup and Picard-step routines without executing
# the example run.

Base.include(@__MODULE__, joinpath(@__DIR__, "driver.jl"))

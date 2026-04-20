# This file is only for benchmark and validation harnesses.
#
# It is deliberately not part of the instructional example. The actual example
# starts in `driver.jl`; this wrapper only loads the same definitions so
# benchmark code can call the setup, residual, adaptation, and output routines
# without executing the example run.

using LinearAlgebra
using Printf
using Grico
import Grico: cell_matrix!, cell_residual!, cell_rhs!, cell_tangent!, face_residual!,
              interface_residual!

const ORDINARYDIFFEQ_AVAILABLE = false

Base.include(@__MODULE__, joinpath(@__DIR__, "parameters.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "euler_physics.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "projection.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "dg_residual.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "runtime_context.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "output.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "time_driver.jl"))

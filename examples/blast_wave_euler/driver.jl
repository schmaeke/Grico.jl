using LinearAlgebra
using Printf
using Grico
import OrdinaryDiffEq
import Grico: cell_matrix!, cell_residual!, cell_rhs!, cell_tangent!, face_residual!,
              interface_residual!

# This example solves the compressible Euler equations
#
#   ∂ₜq + ∇ · F(q) = 0,
#
# for the conservative state
#
#   q = (ρ, ρu, ρv, E),
#
# on the quarter domain
#
#   Ω = [0, 1]².
#
# The setup is a quarter-domain reduction of the standard symmetric blast-wave
# problem on `[-1, 1]²`: the faces `x = 0` and `y = 0` are symmetry planes,
# while `x = 1` and `y = 1` are the physical walls of the box. A smooth
# overpressure region is centered at the origin, so the solution remains
# symmetric and the quarter model reproduces the full-box dynamics at lower
# cost.
#
# Directory organization:
#
# - `driver.jl` loads the example environment and runs the time loop.
# - `parameters.jl` selects physical, DG, adaptivity, and output parameters.
# - `euler_physics.jl` defines conservative-variable helpers and the smooth blast profile.
# - `projection.jl` assembles the DG mass matrix and projects the initial state.
# - `dg_residual.jl` defines the semidiscrete volume, interface, and wall residuals.
# - `runtime_context.jl` builds timestep estimates, mass inverses, contexts, adaptation, and ODE helpers.
# - `output.jl` creates history entries, VTK data, and compact terminal summaries.
# - `time_driver.jl` runs the fixed-mesh time segments interleaved with adaptation.

# The pressure jump is intentionally smoothed because this example does not yet
# include a positivity limiter. Small density and pressure floors are therefore
# retained as a last line of defense against unphysical roundoff excursions.

Base.include(@__MODULE__, joinpath(@__DIR__, "parameters.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "euler_physics.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "projection.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "dg_residual.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "runtime_context.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "output.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "time_driver.jl"))

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_blast_wave_euler_example()
end

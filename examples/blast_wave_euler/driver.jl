using LinearAlgebra
using Printf
using Grico
using WriteVTK
import OrdinaryDiffEq
import Grico: cell_residual!, face_residual!, interface_residual!

# This example solves the compressible Euler equations
#
#   ∂ₜq + ∇ · F(q) = 0,
#
# for the conservative state
#
#   q = (ρ, ρu, ρv, E),
#
# on the periodic Sedov domain
#
#   Ω = [-1.5, 1.5]².
#
# The physical and geometric setup follows the Sedov blast test in
# Rueda-Ramírez and Gassner, arXiv:2102.06017v1, Section 5.2: the initial gas
# is at rest, density and pressure are Gaussian concentrations in a homogeneous
# ambient medium, and all outer faces are periodic. The discretization remains
# Grico-specific: adaptive Cartesian `h`-refinement, matrix-free residuals, and
# a modal DG basis are used instead of the paper's DGSEM/FV subcell blend.
#
# Directory organization:
#
# - `driver.jl` loads the example environment and runs the time loop.
# - `parameters.jl` selects physical, DG, adaptivity, and output parameters.
# - `euler_physics.jl` defines conservative-variable helpers and the Sedov blast profile.
# - `projection.jl` applies the DG mass operator and projects the initial state.
# - `dg_residual.jl` defines the semidiscrete volume, interface, and wall residuals.
# - `runtime_context.jl` builds timestep estimates, mass inverses, contexts, adaptation, and ODE helpers.
# - `output.jl` creates history entries, VTK data, and compact terminal summaries.
# - `time_driver.jl` runs the fixed-mesh time segments interleaved with adaptation.

# A cell-average positivity limiter is applied after accepted ODE steps. It
# preserves the local conservative mean and scales troubled DG polynomials
# toward that mean, which gives this example the same practical role as the
# paper's positivity fallback without adding a second FV discretization path.

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

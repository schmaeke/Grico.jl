using LinearAlgebra
using Printf
using Grico
import Grico: cell_matrix!, face_matrix!, face_rhs!, interface_matrix!

# This is the package's main discontinuous Galerkin flow example.
#
# It solves the steady lid-driven cavity problem on the unit square
#
#   Ω = [0, 1]²,
#
# with incompressible velocity-pressure unknowns `(u, p)` satisfying
#
#   (u · ∇)u - νΔu + ∇p = 0,
#               ∇ · u   = 0,
#
# together with the classical cavity boundary data
#
#   u = (1, 0)  on y = 1,
#   u = (0, 0)  on the remaining walls.
#
# The top corners carry a jump in the prescribed tangential velocity. This is a
# natural fit for a DG demo: all wall data are imposed weakly through boundary
# operators, so no continuous trace space has to resolve the corner singularity.
#
# Discretization strategy:
#
# 1. velocity uses a discontinuous vector-valued tensor-product space,
# 2. pressure uses a lower-order DG space on the same active-leaf topology,
# 3. viscous terms use symmetric interior penalty,
# 4. the convective term is linearized by Picard iteration and discretized with
#    an upwind numerical flux,
# 5. wall data are imposed weakly on all four physical faces, and
# 6. a mean-value pressure constraint removes the constant-pressure null space.
#
# Each Picard step therefore solves one steady Oseen problem with the advecting
# velocity frozen from the previous iterate. The example keeps the Reynolds
# number modest so this fixed-point iteration converges robustly from the zero
# initial guess while still producing a recognizable cavity vortex.
#
# Directory organization:
#
# - `driver.jl` selects the physical, DG, and nonlinear-solver parameters.
# - `algebra.jl` contains small trace, scaling, and wall-data helpers used by the operators.
# - `oseen_operator.jl` defines the cell, interface, and boundary terms for one Oseen step.
# - `diagnostics.jl` computes the kinetic-energy and DG incompressibility monitors.
# - `spaces_and_picard.jl` builds spaces, fields, compiled plans, Picard updates, and adaptation.
# - `output.jl` prints the run header and writes the final VTK data.
# - `picard_driver.jl` runs the adaptive Picard loop.

# ---------------------------------------------------------------------------
# 1. Global parameters
# ---------------------------------------------------------------------------
#
# Mesh and approximation order. The example starts from a coarse uniform mesh
# and lets the single-tolerance DG adaptivity planner find the lid
# singularities automatically while keeping the polynomial degree fixed.
const ROOT_COUNTS = (16, 16)
const ADAPTIVE_STEPS = 4
const ADAPTIVITY_TOLERANCE = 5.0e-2
const MAX_H_LEVEL = 3
const VELOCITY_DEGREE = 2
const PRESSURE_DEGREE = 1
const QUADRATURE_EXTRA_POINTS = 1

# Flow and DG stabilization parameters.
const REYNOLDS_NUMBER = 100.0
const LID_SPEED = 1.0
const VISCOSITY = LID_SPEED / REYNOLDS_NUMBER
const VELOCITY_PENALTY = 6.0
const NORMAL_FLUX_PENALTY = 1.0
const PRESSURE_JUMP_PENALTY = 0.0
const DIVERGENCE_PENALTY = 5.0e-3

# Picard iteration controls.
const PICARD_MAX_ITERS = 24
const PICARD_TOL = 1.0e-6
const ADAPTIVE_PICARD_TOL = 1.0e-4
const PICARD_RELAXATION = 0.85

# Optional VTK output settings.
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 3

Base.include(@__MODULE__, joinpath(@__DIR__, "algebra.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "oseen_operator.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "diagnostics.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "spaces_and_picard.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "output.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "picard_driver.jl"))

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_lid_driven_cavity_example()
end

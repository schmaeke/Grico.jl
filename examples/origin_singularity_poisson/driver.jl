using Printf
using Grico
using WriteVTK
import Grico: cell_apply!, cell_diagonal!, cell_rhs!

# This example is meant to be the simplest "read it from top to bottom" tour of
# adaptive finite elements in Grico.
#
# We solve a Poisson problem on the unit hypercube,
#
#   Ω = [0, 1]ᵈ,
#
# with a known exact solution
#
#   u(x) = r(x)^α,   r(x) = ‖x‖,
#
# where `α = 0.5` in the default configuration. This function is continuous, but
# its gradient blows up at the origin. Numerically that means:
#
# - the solution is smooth away from the origin, so high-order polynomials are
#   attractive there,
# - but near the origin the singular corner is better resolved by local mesh
#   refinement.
#
# This combination makes the problem a classical benchmark for adaptive
# h/p finite-element methods.
#
# The source term is chosen so that
#
#   -Δu = f,   f(r) = -α (α + d - 2) r^(α - 2),
#
# for `r > 0`. At the origin we define both `u` and `f` by their limiting
# finite values used in the discrete code path.
#
# Boundary conditions are mixed:
#
# - on the upper faces `x_a = 1`, the exact Dirichlet trace is imposed;
# - on the lower faces `x_a = 0`, no boundary operator is added, so the weak
#   form uses the natural Neumann condition.
#
# For the radial exact solution this is consistent because ∂u/∂x_a = 0 whenever
# `x_a = 0`.
#
# Directory organization:
#
# - `driver.jl` selects the parameters and loads the pieces in reading order.
# - `operators.jl` defines the local diffusion and load operators.
# - `problem.jl` defines the manufactured data, space, boundary data, and adaptivity plan.
# - `adaptive_driver.jl` performs the solve-estimate-adapt loop and optional VTK export.

# The code is written dimension-independently, but the default is now 2D so a
# new reader can inspect the VTK output directly.
const DIMENSION = 2
const INITIAL_DEGREE = 2
const ADAPTIVE_STEPS = 20

# Single-tolerance adaptivity controls. The planner marks modal detail with one
# tolerance, uses its internal modal-decay classifier for the h/p split, and
# respects the explicit degree and h-level limits below.
const ADAPTIVITY_TOLERANCE = 5.0e-2
const MAX_DEGREE = 4
const MAX_H_LEVEL = 5

# With the default 2D configuration, VTK output is produced after every solve.
# The guard remains dimension-aware so the same file still works if one changes
# `DIMENSION` manually.
const WRITE_VTK = DIMENSION <= 3
const EXPORT_SUBDIVISIONS = 1

# Singularity strength `α` in u = r^α.
const SINGULAR_EXPONENT = 0.5

# Optional quadrature enrichment for the verification integral.
const VERIFICATION_EXTRA_POINTS = 0

Base.include(@__MODULE__, joinpath(@__DIR__, "operators.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "problem.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "adaptive_driver.jl"))

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_origin_singularity_poisson_example()
end

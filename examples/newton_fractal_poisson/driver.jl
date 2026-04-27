using Printf
using Grico
using WriteVTK
import Grico: cell_matrix!, cell_rhs!

# This example solves a manufactured Poisson problem on the unit square,
#
#   -Δu = f,  u = uₑ on ∂Ω.
#
# The exact solution `uₑ` is obtained by mapping each point `x ∈ Ω` to a point
# `z ∈ ℂ`, applying a finite Newton iteration for `p(z) = z³ - 1`, and coloring
# the result by smooth root-proximity weights. This produces a continuous
# Newton-fractal-like field with nested smooth and nonsmooth regions. The
# nonzero trace `uₑ|∂Ω` is imposed strongly so the run also exercises the
# Dirichlet projection machinery on deep mixed-`hp` boundary meshes.

const DIMENSION = 2
const INITIAL_DEGREE = 3
const MIN_DEGREE = 1
const MAX_DEGREE = 5
const QUADRATURE_EXTRA_POINTS = 4

# The driver performs one solve-adapt cycle per admissible dyadic level `ℓ`.
# The root grid has `ℓ = 0`, so `MAX_H_LEVEL` is also the default number of
# adaptive transitions.
const MAX_H_LEVEL = 10

# `NEWTON_PLANE_SCALE = 3` maps Ω to `[-1.5, 1.5]² ⊂ ℂ`, which places all three
# roots of `p(z) = z³ - 1` inside the sampled region. The derivative
# regularization keeps the Newton update finite near `z = 0`. The manufactured
# load is assembled from a finite-difference approximation of `∇uₑ`, so the
# difference step also defines the smallest scale resolved by the model.
const NEWTON_ITERATIONS = 9
const NEWTON_PLANE_SCALE = 3.0
const NEWTON_ROOT_SHARPNESS = 32.0
const NEWTON_RESIDUAL_WEIGHT = 0.25
const NEWTON_RESIDUAL_CONTRAST = 1.5
const NEWTON_DERIVATIVE_REGULARIZATION = 1.0e-5
const NEWTON_DIFFERENCE_STEP = 4.0e-4

const ADAPTIVITY_TOLERANCE = 8.0e-2
const ADAPTIVITY_SMOOTHNESS_THRESHOLD = 0.8

# Optional verification and VTK output settings.
const VERIFICATION_EXTRA_POINTS = 1
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 3

Base.include(@__MODULE__, joinpath(@__DIR__, "operators.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "problem.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "adaptive_driver.jl"))

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_newton_fractal_poisson_example()
end

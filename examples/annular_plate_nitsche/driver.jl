using Printf
using Grico
import Grico: cell_matrix!, surface_matrix!, surface_rhs!

# This example is the compact "unfitted methods" tour of the package.
#
# We solve the scalar Laplace problem on the annulus
#
#   Ω = {x ∈ ℝ² : Rᵢ ≤ ‖x‖ ≤ Rₒ}
#
# by embedding the curved physical domain into a Cartesian background mesh. The
# volume bilinear form is integrated with finite-cell quadrature on the cut
# cells, while the Dirichlet boundary condition on the circular boundary is
# imposed weakly by a symmetric Nitsche method on an embedded segment mesh.
#
# The exact harmonic solution is radial,
#
#   u(r) = log(r / Rₒ) / log(Rᵢ / Rₒ),
#
# so Δu = 0 in the annulus, u = 1 on the inner circle, and u = 0 on the outer
# circle. This makes the example a compact demonstration of three ideas at
# once:
#
# 1. finite-cell quadrature on a cut Cartesian background grid,
# 2. embedded-surface assembly on an explicit segment mesh, and
# 3. weak Dirichlet enforcement by Nitsche terms instead of boundary-fitted
#    trace constraints.
#
# Directory organization:
#
# - `driver.jl` selects the annulus parameters and loads the mathematical pieces.
# - `operators.jl` defines the volume Laplace form and symmetric Nitsche boundary terms.
# - `geometry.jl` builds the polygonal circle meshes used for the embedded boundary.
# - `problem.jl` assembles the unfitted physical-domain problem description.
# - `solve_and_export.jl` solves, verifies, prints a summary, and optionally writes VTK output.

# ---------------------------------------------------------------------------
# 1. Problem geometry and discretization parameters
# ---------------------------------------------------------------------------
#
# The physical domain is the annulus, but the finite-element space still starts
# from the enclosing square. `PhysicalDomain` trims leaves that are fully
# outside the annulus and lets compilation inject finite-cell quadratures on
# the remaining cut leaves automatically.
const INNER_RADIUS = 0.35
const OUTER_RADIUS = 1.0
const ROOT_COUNTS = (2, 2)
const DEGREE = 4

# The circular boundary is represented by a polygonal segment mesh. The segment
# count controls the geometric approximation of the circles, while
# `SURFACE_POINT_COUNT` controls the one-dimensional quadrature order used on
# each segment during embedded-surface assembly.
const SEGMENT_COUNT = 128
const SURFACE_POINT_COUNT = 3

# Finite-cell quadrature recursively subdivides cut cells until the physical
# part of the cell is resolved well enough for the moment-fitting procedure.
const FCM_SUBDIVISION_DEPTH = 7

# The symmetric Nitsche penalty scales like η / h. The value here is chosen
# large enough to stabilize the weak Dirichlet enforcement without dominating
# the consistency terms.
const NITSCHE_PENALTY = 40.0

# Optional VTK output settings.
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 4

Base.include(@__MODULE__, joinpath(@__DIR__, "operators.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "geometry.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "problem.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "solve_and_export.jl"))

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
  run_annular_plate_nitsche_example()
end

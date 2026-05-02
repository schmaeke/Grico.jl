# ---------------------------------------------------------------------------
# 1. Problem parameters
# ---------------------------------------------------------------------------
#
# These constants are kept separate from the executable driver so benchmark and
# test wrappers can load the reusable setup code without importing the
# example-only ODE package.

# Geometry and DG discretization. The physical setup follows the Sedov blast
# test of Rueda-Ramírez and Gassner, arXiv:2102.06017v1, Section 5.2: a full
# periodic square with a Gaussian density and pressure concentration at the
# origin. The paper uses 64² elements and degree 7; the defaults below are
# deliberately smaller so the example remains usable as an interactive driver,
# while the physical and geometric data match the paper.
const DOMAIN_ORIGIN = (-1.5, -1.5)
const DOMAIN_EXTENT = (3.0, 3.0)
const PERIODIC_AXES = (true, true)
const BLAST_CENTER = (0.0, 0.0)
const ROOT_COUNTS = (16, 16)
const POLYDEG = 1
const QUADRATURE_EXTRA_POINTS = 1

# Gas model and time-integration controls.
const GAMMA = 1.4
const CFL = 0.05
const FINAL_TIME = 10.0
const SAVE_INTERVAL = 0.25
const ADAPT_INTERVAL = 0.025

# Adaptivity controls. This example uses pure `h`-adaptivity and keeps the
# polynomial degree fixed.
const ADAPTIVITY_TOLERANCE = 2.5e-2
const MAX_H_LEVEL = 2

# Physical floors, positivity limiter controls, and blast parameters.
const DENSITY_FLOOR = 1.0e-12
const PRESSURE_FLOOR = 1.0e-12
const POSITIVITY_DENSITY_FLOOR = 1.0e-10
const POSITIVITY_PRESSURE_FLOOR = 1.0e-10
const POSITIVITY_LIMITER_ENABLED = true
const BACKGROUND_DENSITY = 1.0
const BACKGROUND_PRESSURE = 1.0e-5
const DENSITY_SIGMA = 0.25
const PRESSURE_SIGMA = 0.15
const INITIAL_BLAST_REFINEMENT_LAYERS = 2
const INITIAL_BLAST_REFINEMENT_RADIUS = 3.0 * max(DENSITY_SIGMA, PRESSURE_SIGMA)

# Output controls.
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = POLYDEG

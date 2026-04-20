# ---------------------------------------------------------------------------
# 1. Problem parameters
# ---------------------------------------------------------------------------
#
# These constants are kept separate from the executable driver so benchmark and
# test wrappers can load the reusable setup code without importing the
# example-only ODE package.

# Geometry and DG discretization.
const DOMAIN_ORIGIN = (0.0, 0.0)
const DOMAIN_EXTENT = (1.0, 1.0)
# The blast is centered on the two symmetry planes, so the quarter-domain model
# is exactly the mirrored restriction of the full symmetric blast.
const BLAST_CENTER = DOMAIN_ORIGIN
const ROOT_COUNTS = (16, 16)
const POLYDEG = 1
const QUADRATURE_EXTRA_POINTS = 1

# Gas model and time-integration controls.
const GAMMA = 1.4
const CFL = 0.025
const FINAL_TIME = 4.0
const SAVE_INTERVAL = 0.0125
const ADAPT_INTERVAL = 0.00125 / 2

# Adaptivity controls. This example uses pure `h`-adaptivity and keeps the
# polynomial degree fixed.
const ADAPTIVITY_TOLERANCE = 2.5e-2
const MAX_H_LEVEL = 5

# Physical floors and blast parameters.
const DENSITY_FLOOR = 1.0e-12
const PRESSURE_FLOOR = 1.0e-12
const BACKGROUND_DENSITY = 1.0
const INNER_PRESSURE = 3.0
const OUTER_PRESSURE = 1.0
const BLAST_RADIUS = 0.2
const BLAST_TRANSITION_WIDTH = 0.03
const INITIAL_BLAST_REFINEMENT_LAYERS = 4
const INITIAL_BLAST_REFINEMENT_RADIUS = BLAST_RADIUS + 2.0 * BLAST_TRANSITION_WIDTH

# Output controls.
const WRITE_VTK = true
const EXPORT_SUBDIVISIONS = 1
const EXPORT_DEGREE = 1

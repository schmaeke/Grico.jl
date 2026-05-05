# ---------------------------------------------------------------------------
# 2. Mass matrix and initial projection
# ---------------------------------------------------------------------------

# The initial condition is inserted into the DG space via an `L²` projection.
# That means we apply the element mass operator
#
#   (M q)ᵢ = ∫_K φᵢ q dx,
#
# and the projected right-hand side
#
#   bᵢ = ∫_K φᵢ q₀ dx,
#
# then solve `M q_h(0) = b`. The projection is a diagonal block mass form in
# field-component space, so it can be written directly with Grico's weak-form
# API.

# Build the projected initial DG state `q_h(x, 0)`.
function project_initial_condition(field, data; solver=AutoLinearSolver())
  problem = AffineProblem(field; operator_class=SPD())
  add_cell_bilinear!(problem, field, field) do q, v, w
    component(v) == component(w) ? value(v) * value(w) : zero(value(v) * value(w))
  end
  add_cell_linear!(problem, field) do q, v
    data(point(q))[component(v)] * value(v)
  end
  return solve(problem; solver=solver)
end

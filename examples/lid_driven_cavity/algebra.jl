# ---------------------------------------------------------------------------
# 2. Small algebra helpers
# ---------------------------------------------------------------------------
#
# These are just tiny utilities used repeatedly inside the local operator loops.
# Keeping them separate makes the weak-form code below easier to read.

# Tiny helpers used inside quadrature loops.
@inline dot2(a, b) = a[1] * b[1] + a[2] * b[2]
@inline squared_norm2(a) = dot2(a, a)
@inline velocity_index(mode_count::Int, component::Int, mode::Int) = (component - 1) * mode_count +
                                                                     mode

# Grico's generic `jump(minus, plus)` helper returns `plus - minus`, but the DG
# bilinear forms below are written in the more common trace convention
# [w] = w⁻ - w⁺ on an interface with normal pointing from minus to plus.
@inline trace_jump_sign(is_plus::Bool) = is_plus ? -1.0 : 1.0

@inline function interface_length_scale(field, minus_leaf, plus_leaf, axis)
  domain_data = field_space(field).domain
  h_minus = cell_size(domain_data, minus_leaf, axis)
  h_plus = cell_size(domain_data, plus_leaf, axis)
  return 2.0 * h_minus * h_plus / (h_minus + h_plus)
end

@inline boundary_length_scale(field, leaf, axis) = cell_size(field_space(field).domain, leaf, axis)

@inline function interface_penalty_scale(field, minus_leaf, plus_leaf, axis)
  h = interface_length_scale(field, minus_leaf, plus_leaf, axis)
  degree_value = max(cell_degrees(field_space(field), minus_leaf)[axis],
                     cell_degrees(field_space(field), plus_leaf)[axis])
  return (degree_value + 1)^2 / h
end

@inline function boundary_penalty_scale(field, leaf, axis)
  h = boundary_length_scale(field, leaf, axis)
  degree_value = cell_degrees(field_space(field), leaf)[axis]
  return (degree_value + 1)^2 / h
end

# The lid data are intentionally discontinuous at the top corners. The weak DG
# wall treatment handles that without requiring a globally continuous boundary
# trace. Side and bottom walls remain no-slip.
cavity_velocity(x) = x[2] >= 1.0 - 1.0e-12 ? (LID_SPEED, 0.0) : (0.0, 0.0)

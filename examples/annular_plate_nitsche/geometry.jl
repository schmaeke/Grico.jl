# ---------------------------------------------------------------------------
# 3. Background mesh and embedded boundary geometry
# ---------------------------------------------------------------------------
#
# The background finite-element mesh is just a square. The curved circles are
# represented separately by a segment mesh. This split is central to unfitted
# methods: geometry and approximation space no longer need to match exactly.
#
# Build a polygonal approximation of a circle as a closed segment mesh. The
# `clockwise` option is used to reverse the inner boundary orientation so the
# resulting two-component boundary has a consistent global orientation. That
# matters because the outward normal on the inner boundary points toward the
# hole, not toward the outer square.
function circle_points_segments(radius, segment_count; clockwise=false)
  angles = range(0.0, 2 * pi; length=segment_count + 1)[1:(end-1)]
  points = [(radius * cos(angle), radius * sin(angle)) for angle in angles]
  clockwise && reverse!(points)
  segments = [(index, index == segment_count ? 1 : index + 1) for index in 1:segment_count]
  return points, segments
end

module GricoMakieExt

using Grico
using Makie

# This extension is intentionally a thin plotting adapter. The core package
# owns sampling, field evaluation, and dataset validation; Makie-specific code
# only chooses a plot primitive and converts Grico's sampled arrays to the
# coordinate and connectivity layouts expected by Makie.

# The high-level plotting methods follow the same contract as VTK export:
# sample a state/reference pair through Grico's backend-neutral postprocessing
# API, then hand the sampled arrays to the small Makie adapter below.
function Grico.plot_field(state::Grico.State, name::Union{Symbol,AbstractString}; kwargs...)
  return Grico.plot_field(_makie_state_reference(state), name; state=state, kwargs...)
end

function Grico.plot_field(reference::Union{Grico.HpSpace{D,T},Grico.AbstractDomain{D,T}},
                          name::Union{Symbol,AbstractString};
                          state::Union{Nothing,Grico.State}=nothing, fields=nothing, point_data=(),
                          cell_data=(), field_data=(), subdivisions::Integer=1,
                          sample_degree::Integer=1, mesh=false,
                          kwargs...) where {D,T<:AbstractFloat}
  data = _makie_sample_postprocess(reference; state=state, fields=fields, point_data=point_data,
                                   cell_data=cell_data, field_data=field_data,
                                   subdivisions=subdivisions, sample_degree=sample_degree)
  mesh_overlay = mesh === true ? Grico.sample_mesh_skeleton(reference) : mesh
  return Grico.plot_field(data, name; mesh=mesh_overlay, kwargs...)
end

# Construct a standalone figure for an already sampled point dataset. The
# low-level sampled-data method is the backend boundary: callers that need to
# reuse one sampling pass can pass `SampledPostprocess` directly, while the
# higher-level methods above produce the same object internally.
function Grico.plot_field(data::Grico.SampledPostprocess, name::Union{Symbol,AbstractString};
                          component=nothing, mesh=false, figure=(;), axis=(;), colorbar=true,
                          colorbar_label=String(name), mesh_color=:white, kwargs...)
  sampled_mesh = data.mesh
  _require_makie_dimension(sampled_mesh)
  figure_object = Makie.Figure(; figure...)
  axis_object = _makie_field_axis(figure_object[1, 1], sampled_mesh, String(name), axis)
  plot_object = Grico.plot_field!(axis_object, data, name; component=component, kwargs...)

  mesh === false ||
    _plot_mesh_overlay!(axis_object, mesh === true ? sampled_mesh : mesh, mesh_color)
  colorbar &&
    _makie_field_has_colorbar(sampled_mesh) &&
    Makie.Colorbar(figure_object[1, 2], plot_object; label=colorbar_label)
  Makie.autolimits!(axis_object)
  return figure_object
end

# Mutating field plots preserve Makie's normal workflow: users create their own
# figure/axis layout, and Grico only adds one sampled dataset to the supplied
# axis. Sampling keywords are intentionally identical to the non-mutating method.
function Grico.plot_field!(axis, state::Grico.State, name::Union{Symbol,AbstractString}; kwargs...)
  return Grico.plot_field!(axis, _makie_state_reference(state), name; state=state, kwargs...)
end

function Grico.plot_field!(axis, reference::Union{Grico.HpSpace{D,T},Grico.AbstractDomain{D,T}},
                           name::Union{Symbol,AbstractString};
                           state::Union{Nothing,Grico.State}=nothing, fields=nothing, point_data=(),
                           cell_data=(), field_data=(), subdivisions::Integer=1,
                           sample_degree::Integer=1, kwargs...) where {D,T<:AbstractFloat}
  data = _makie_sample_postprocess(reference; state=state, fields=fields, point_data=point_data,
                                   cell_data=cell_data, field_data=field_data,
                                   subdivisions=subdivisions, sample_degree=sample_degree)
  return Grico.plot_field!(axis, data, name; kwargs...)
end

# In one dimension, the sampled point order is already the physical line order
# inside each active leaf, so scalar datasets can be drawn directly as a line.
function Grico.plot_field!(axis, data::Grico.SampledPostprocess{1},
                           name::Union{Symbol,AbstractString}; component=nothing, color=:black,
                           linewidth=2.0, kwargs...)
  values = _makie_point_values(data, String(name), component)
  return Makie.lines!(axis, vec(data.mesh.points[1, :]), values; color=color, linewidth=linewidth,
                      kwargs...)
end

# In two dimensions, the sampled tensor-product cells are rendered as triangles.
# The triangulation deliberately keeps duplicate interface points so discontinuous
# hp fields are not visually smoothed across active-leaf boundaries.
function Grico.plot_field!(axis, data::Grico.SampledPostprocess{2},
                           name::Union{Symbol,AbstractString}; component=nothing, colormap=:RdBu,
                           kwargs...)
  sampled_mesh = data.mesh
  values = _makie_point_values(data, String(name), component)
  vertices = _makie_vertices(sampled_mesh)
  faces = _makie_triangle_faces(sampled_mesh)
  return Makie.mesh!(axis, vertices, faces; color=values, colormap=colormap, kwargs...)
end

# Mesh plotting uses the active-leaf skeleton by default. This exposes the true
# adaptive topology rather than the finer sampled visualization cells used for a
# field plot.
function Grico.plot_mesh(reference::Union{Grico.HpSpace{D,T},Grico.AbstractDomain{D,T}};
                         kwargs...) where {D,T<:AbstractFloat}
  return Grico.plot_mesh(Grico.sample_mesh_skeleton(reference); kwargs...)
end

function Grico.plot_mesh(state::Grico.State; kwargs...)
  return Grico.plot_mesh(_makie_state_reference(state); kwargs...)
end

function Grico.plot_mesh(skeleton::Grico.SampledMeshSkeleton; figure=(;), axis=(;), kwargs...)
  _require_makie_dimension(skeleton)
  figure_object = Makie.Figure(; figure...)
  axis_object = _makie_mesh_axis(figure_object[1, 1], skeleton, axis)
  Grico.plot_mesh!(axis_object, skeleton; kwargs...)
  Makie.autolimits!(axis_object)
  return figure_object
end

function Grico.plot_mesh(mesh::Grico.SampledMesh; figure=(;), axis=(;), kwargs...)
  _require_makie_dimension(mesh)
  figure_object = Makie.Figure(; figure...)
  axis_object = _makie_mesh_axis(figure_object[1, 1], mesh, axis)
  Grico.plot_mesh!(axis_object, mesh; kwargs...)
  Makie.autolimits!(axis_object)
  return figure_object
end

function Grico.plot_mesh!(axis, reference::Union{Grico.HpSpace{D,T},Grico.AbstractDomain{D,T}};
                          kwargs...) where {D,T<:AbstractFloat}
  return Grico.plot_mesh!(axis, Grico.sample_mesh_skeleton(reference); kwargs...)
end

function Grico.plot_mesh!(axis, state::Grico.State; kwargs...)
  return Grico.plot_mesh!(axis, _makie_state_reference(state); kwargs...)
end

function Grico.plot_mesh!(axis, skeleton::Grico.SampledMeshSkeleton; color=:black, linewidth=0.75,
                          kwargs...)
  _require_makie_dimension(skeleton)
  return Makie.linesegments!(axis, _makie_segments(skeleton); color=color, linewidth=linewidth,
                             kwargs...)
end

function Grico.plot_mesh!(axis, mesh::Grico.SampledMesh; color=:black, linewidth=0.5, kwargs...)
  _require_makie_dimension(mesh)
  return Makie.linesegments!(axis, _makie_segments(mesh); color=color, linewidth=linewidth,
                             kwargs...)
end

function _plot_mesh_overlay!(axis, skeleton::Grico.SampledMeshSkeleton, color)
  return Grico.plot_mesh!(axis, skeleton; color=color, linewidth=0.65)
end

function _plot_mesh_overlay!(axis, mesh::Grico.SampledMesh, color)
  return Grico.plot_mesh!(axis, mesh; color=color, linewidth=0.35)
end

# A state carries only fields and coefficients, not a direct plotting geometry.
# Its first field identifies the common reference space guaranteed by the field
# layout checks in the core package.
function _makie_state_reference(state::Grico.State)
  layout = Grico.field_layout(state)
  return Grico.field_space(first(Grico.fields(layout)))
end

# Keep high-level Makie methods as pure consumers of the backend-neutral
# sampling API. All validation of fields, callbacks, dimensions, and sample
# counts therefore remains centralized in `sample_postprocess`.
function _makie_sample_postprocess(reference; state::Union{Nothing,Grico.State}=nothing,
                                   fields=nothing, point_data=(), cell_data=(), field_data=(),
                                   subdivisions::Integer=1, sample_degree::Integer=1)
  return Grico.sample_postprocess(reference; state=state, fields=fields, point_data=point_data,
                                  cell_data=cell_data, field_data=field_data,
                                  subdivisions=subdivisions, sample_degree=sample_degree)
end

# Makie can render the sampled representation directly for one- and
# two-dimensional domains. Higher-dimensional sampled data remains useful for
# file export, but it has no canonical scalar field plot here.
function _require_makie_dimension(mesh::Union{Grico.SampledMesh{D},Grico.SampledMeshSkeleton{D}}) where {D}
  1 <= D <= 2 ||
    throw(ArgumentError("Makie postprocessing currently supports one- and two-dimensional samples"))
  return nothing
end

function _makie_field_axis(position, mesh::Grico.SampledMesh{1}, name::String, axis)
  return Makie.Axis(position; xlabel="x", ylabel=name, axis...)
end

function _makie_field_axis(position, mesh::Grico.SampledMesh{2}, name::String, axis)
  return Makie.Axis(position; aspect=Makie.DataAspect(), xlabel="x", ylabel="y", axis...)
end

_makie_field_has_colorbar(::Grico.SampledMesh{1}) = false
_makie_field_has_colorbar(::Grico.SampledMesh{2}) = true

function _makie_mesh_axis(position, mesh::Union{Grico.SampledMesh{1},Grico.SampledMeshSkeleton{1}},
                          axis)
  return Makie.Axis(position; xlabel="x", ylabel="", axis...)
end

function _makie_mesh_axis(position, mesh::Union{Grico.SampledMesh{2},Grico.SampledMeshSkeleton{2}},
                          axis)
  return Makie.Axis(position; aspect=Makie.DataAspect(), xlabel="x", ylabel="y", axis...)
end

# Makie receives one scalar value per plotted vertex. Vector-valued Grico point
# datasets are stored as component-major matrices, so users select the displayed
# component explicitly through `component=<index>`.
function _makie_point_values(data::Grico.SampledPostprocess, name::String, component)
  dataset = _makie_dataset(data.point_data, name)

  if dataset isa AbstractVector
    component === nothing ||
      throw(ArgumentError("scalar point dataset $name does not accept component"))
    return dataset
  end

  dataset isa AbstractMatrix ||
    throw(ArgumentError("point dataset $name must be a vector or matrix"))
  component === nothing &&
    throw(ArgumentError("vector-valued point dataset $name requires component=<index>"))
  component_index = _makie_component(component, size(dataset, 1), name)
  return vec(dataset[component_index, :])
end

function _makie_dataset(datasets, name::String)
  for (dataset_name, values) in datasets
    dataset_name == name && return values
  end

  throw(ArgumentError("missing point dataset $name"))
end

function _makie_component(component::Integer, component_count::Int, name::String)
  1 <= component <= component_count ||
    throw(ArgumentError("component for point dataset $name must be in 1:$component_count"))
  return Int(component)
end

# Makie accepts two-dimensional vertex coordinates as an `N × 2` array here,
# while Grico stores sampled coordinates as `D × N` columns to match the rest of
# the postprocessing backends.
function _makie_vertices(mesh::Grico.SampledMesh{2,T}) where {T}
  vertices = Matrix{T}(undef, size(mesh.points, 2), 2)

  for point_index in axes(mesh.points, 2)
    vertices[point_index, 1] = mesh.points[1, point_index]
    vertices[point_index, 2] = mesh.points[2, point_index]
  end

  return vertices
end

# Split every sampled quadrilateral subcell into two triangles. The leaf-wise
# point numbering is duplicated across interfaces, which lets Makie show DG
# jumps without accidentally interpolating across neighboring leaves.
function _makie_triangle_faces(mesh::Grico.SampledMesh{2})
  faces = Matrix{Int}(undef, 2 * length(mesh.cell_leaves), 3)
  point_stride = mesh.point_stride
  local_point_count = point_stride^2
  cell_indices = CartesianIndices((mesh.subdivisions, mesh.subdivisions))
  row = 1

  for leaf_index in eachindex(mesh.leaf_data)
    point_offset = (leaf_index - 1) * local_point_count

    for I in cell_indices
      start = ((I[1] - 1) * mesh.sample_degree + 1, (I[2] - 1) * mesh.sample_degree + 1)
      lower_left = _makie_local_point(point_offset, point_stride, start)
      lower_right = _makie_local_point(point_offset, point_stride,
                                       (start[1] + mesh.sample_degree, start[2]))
      upper_right = _makie_local_point(point_offset, point_stride,
                                       (start[1] + mesh.sample_degree,
                                        start[2] + mesh.sample_degree))
      upper_left = _makie_local_point(point_offset, point_stride,
                                      (start[1], start[2] + mesh.sample_degree))
      faces[row, 1] = lower_left
      faces[row, 2] = lower_right
      faces[row, 3] = upper_right
      faces[row+1, 1] = upper_right
      faces[row+1, 2] = upper_left
      faces[row+1, 3] = lower_left
      row += 2
    end
  end

  return faces
end

function _makie_local_point(point_offset::Int, point_stride::Int, index::NTuple{2,Int})
  return point_offset + index[1] + (index[2] - 1) * point_stride
end

# The active-leaf skeleton and the sampled mesh share the same line-segment
# drawing path. One-dimensional coordinates are lifted to y=0 so they can be
# rendered by Makie's two-coordinate segment primitive.
function _makie_segments(skeleton::Grico.SampledMeshSkeleton{2})
  segments = Vector{Tuple{Makie.Point2f,Makie.Point2f}}(undef, size(skeleton.edges, 2))

  for edge_index in axes(skeleton.edges, 2)
    first = skeleton.edges[1, edge_index]
    second = skeleton.edges[2, edge_index]
    segments[edge_index] = (_makie_point(skeleton.points, first),
                            _makie_point(skeleton.points, second))
  end

  return segments
end

function _makie_segments(skeleton::Grico.SampledMeshSkeleton{1})
  segments = Vector{Tuple{Makie.Point2f,Makie.Point2f}}(undef, size(skeleton.edges, 2))

  for edge_index in axes(skeleton.edges, 2)
    first = skeleton.edges[1, edge_index]
    second = skeleton.edges[2, edge_index]
    segments[edge_index] = (_makie_point(skeleton.points, first),
                            _makie_point(skeleton.points, second))
  end

  return segments
end

function _makie_segments(mesh::Grico.SampledMesh{2})
  point_stride = mesh.point_stride
  local_point_count = point_stride^2
  edge_total = length(mesh.leaf_data) * 2 * point_stride * (point_stride - 1)
  segments = Vector{Tuple{Makie.Point2f,Makie.Point2f}}(undef, edge_total)
  offset = 1

  for leaf_index in eachindex(mesh.leaf_data)
    point_offset = (leaf_index - 1) * local_point_count

    for j in 1:point_stride, i in 1:(point_stride-1)
      first = _makie_local_point(point_offset, point_stride, (i, j))
      second = _makie_local_point(point_offset, point_stride, (i + 1, j))
      segments[offset] = (_makie_point(mesh.points, first), _makie_point(mesh.points, second))
      offset += 1
    end

    for i in 1:point_stride, j in 1:(point_stride-1)
      first = _makie_local_point(point_offset, point_stride, (i, j))
      second = _makie_local_point(point_offset, point_stride, (i, j + 1))
      segments[offset] = (_makie_point(mesh.points, first), _makie_point(mesh.points, second))
      offset += 1
    end
  end

  return segments
end

function _makie_segments(mesh::Grico.SampledMesh{1})
  point_stride = mesh.point_stride
  local_point_count = point_stride
  edge_total = length(mesh.leaf_data) * (point_stride - 1)
  segments = Vector{Tuple{Makie.Point2f,Makie.Point2f}}(undef, edge_total)
  offset = 1

  for leaf_index in eachindex(mesh.leaf_data)
    point_offset = (leaf_index - 1) * local_point_count

    for i in 1:(point_stride-1)
      first = point_offset + i
      second = point_offset + i + 1
      segments[offset] = (_makie_point(mesh.points, first), _makie_point(mesh.points, second))
      offset += 1
    end
  end

  return segments
end

# Convert one sampled coordinate column to Makie's two-dimensional point type.
# One-dimensional meshes are embedded on the x-axis so the same segment plot
# machinery can render both line meshes and planar meshes.
function _makie_point(points::AbstractMatrix, index::Int)
  return Makie.Point2f(points[1, index], size(points, 1) == 1 ? 0 : points[2, index])
end

end

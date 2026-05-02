module GricoWriteVTKExt

using Grico
using WriteVTK: MeshCell, VTKCellData, VTKCellTypes, VTKFieldData, VTKPointData,
                multiblock_add_block, vtk_grid, vtk_multiblock

# This extension is the only layer that knows about WriteVTK's concrete grid,
# dataset, and XML conventions. The core package supplies fully sampled arrays
# and one-based local connectivity; this file converts that representation to
# the VTK cell ordering and file structure expected by ParaView-compatible tools.

const _VtkArrayData = Union{AbstractVector,AbstractMatrix}

# Public VTK entry points mirror `sample_postprocess`: users may export a whole
# state, a chosen reference plus state/dataset callbacks, or an already sampled
# object. All numerical sampling stays in Grico's core; this extension only
# translates the sampled arrays and connectivity into WriteVTK calls.
function Grico.write_vtk(path::AbstractString, state::Grico.State; kwargs...)
  layout = Grico.field_layout(state)
  space = Grico.field_space(Grico.fields(layout)[1])
  return Grico.write_vtk(path, space; state=state, kwargs...)
end

function Grico.write_vtk(path::AbstractString, space::Grico.HpSpace{D,T};
                         state::Union{Nothing,Grico.State}=nothing, fields=nothing, point_data=(),
                         cell_data=(), field_data=(), subdivisions=1, sample_degree=1,
                         mesh::Bool=false, vtk_kwargs...) where {D,T<:AbstractFloat}
  return _write_vtk(path, space; state=state, fields=fields, point_data=point_data,
                    cell_data=cell_data, field_data=field_data, subdivisions=subdivisions,
                    sample_degree=sample_degree, mesh=mesh, vtk_kwargs...)
end

function Grico.write_vtk(path::AbstractString, domain_data::Grico.AbstractDomain{D,T};
                         state::Union{Nothing,Grico.State}=nothing, fields=nothing, point_data=(),
                         cell_data=(), field_data=(), subdivisions=1, sample_degree=1,
                         mesh::Bool=false, vtk_kwargs...) where {D,T<:AbstractFloat}
  return _write_vtk(path, domain_data; state=state, fields=fields, point_data=point_data,
                    cell_data=cell_data, field_data=field_data, subdivisions=subdivisions,
                    sample_degree=sample_degree, mesh=mesh, vtk_kwargs...)
end

function Grico.write_vtk(path::AbstractString, data::Grico.SampledPostprocess; mesh::Bool=false,
                         skeleton=nothing, vtk_kwargs...)
  _require_vtk_dimension(data.mesh)
  grid_kwargs = _vtk_grid_kwargs(vtk_kwargs)

  if mesh
    skeleton = _require_vtk_skeleton(data, skeleton)
    return _write_vtk_multiblock(path, data, skeleton, grid_kwargs)
  end

  return only(_write_vtk_solution_grid(path, data, grid_kwargs))
end

# Shared implementation for state/reference export. Sampling and validation are
# delegated to Grico's backend-neutral postprocessing layer, while this adapter
# only decides whether to write a single solution grid or a multiblock file with
# an additional active-leaf mesh skeleton.
function _write_vtk(path::AbstractString, reference; state::Union{Nothing,Grico.State}=nothing,
                    fields=nothing, point_data=(), cell_data=(), field_data=(),
                    subdivisions=1, sample_degree=1, mesh::Bool=false, vtk_kwargs...)
  data = Grico.sample_postprocess(reference; state=state, fields=fields, point_data=point_data,
                                  cell_data=cell_data, field_data=field_data,
                                  subdivisions=subdivisions, sample_degree=sample_degree)
  grid_kwargs = _vtk_grid_kwargs(vtk_kwargs)

  if mesh
    skeleton = Grico.sample_mesh_skeleton(reference)
    return _write_vtk_multiblock(path, data, skeleton, grid_kwargs)
  end

  return only(_write_vtk_solution_grid(path, data, grid_kwargs))
end

function _require_vtk_dimension(mesh::Union{Grico.SampledMesh{D},
                                            Grico.SampledMeshSkeleton{D}}) where {D}
  Grico.postprocess_supported(D) ||
    throw(ArgumentError("VTK export requires sampled data with dimension between 1 and 3"))
  return nothing
end

function _require_vtk_skeleton(data::Grico.SampledPostprocess{D},
                               skeleton::Grico.SampledMeshSkeleton{D}) where {D}
  _require_vtk_dimension(skeleton)
  return skeleton
end

function _require_vtk_skeleton(data::Grico.SampledPostprocess{D},
                               ::Grico.SampledMeshSkeleton{S}) where {D,S}
  throw(ArgumentError("mesh skeleton dimension $S does not match sampled data dimension $D"))
end

function _require_vtk_skeleton(::Grico.SampledPostprocess, skeleton)
  skeleton === nothing &&
    throw(ArgumentError("mesh=true with sampled data requires skeleton=sample_mesh_skeleton(...)"))
  throw(ArgumentError("mesh=true with sampled data requires a SampledMeshSkeleton"))
end

# WriteVTK's default VTK XML version may change with package versions. Grico pins
# the default to `:latest` here so callers get modern Lagrange-cell support
# unless they deliberately override `vtkversion`.
function _vtk_grid_kwargs(vtk_kwargs)
  return haskey(vtk_kwargs, :vtkversion) ? (; vtk_kwargs...) : (; vtk_kwargs..., vtkversion=:latest)
end

# Write the sampled solution grid as one ordinary `.vtu` file. The VTK adapter
# converts Grico's backend-neutral sampled mesh into VTK Lagrange cells and pads
# coordinates to the three-component point matrix required by WriteVTK.
function _write_vtk_solution_grid(path::AbstractString, data::Grico.SampledPostprocess, grid_kwargs)
  mesh = data.mesh
  _require_vtk_dimension(mesh)
  return vtk_grid(path, _vtk_point_matrix(mesh), _vtk_lagrange_cells(mesh); grid_kwargs...) do vtk
    _write_vtk_datasets!(vtk, data.point_data, data.cell_data, data.field_data)
  end
end

# With `mesh=true`, the user-facing result is one `.vtm` file. Its first block
# is the sampled solution grid and its second block is the true active-leaf mesh
# skeleton. The child `.vtu` files live next to the wrapper and are named from
# the same stem so a time series remains easy to manage.
function _write_vtk_multiblock(path::AbstractString, data::Grico.SampledPostprocess,
                               skeleton::Grico.SampledMeshSkeleton, grid_kwargs)
  _require_vtk_dimension(data.mesh)
  _require_vtk_dimension(skeleton)
  base_path = _vtk_multiblock_base_path(path)
  vtm = vtk_multiblock(base_path)

  solution = vtk_grid(_vtk_block_path(base_path, "solution"), _vtk_point_matrix(data.mesh),
                      _vtk_lagrange_cells(data.mesh); grid_kwargs...)
  _write_vtk_datasets!(solution, data.point_data, data.cell_data, data.field_data)
  multiblock_add_block(vtm, solution, "solution")

  mesh_block = vtk_grid(_vtk_block_path(base_path, "mesh"), _vtk_point_matrix(skeleton),
                        _vtk_line_cells(skeleton); grid_kwargs...)
  _write_vtk_datasets!(mesh_block, Pair{String,_VtkArrayData}[], skeleton.cell_data,
                       Pair{String,Any}[])
  multiblock_add_block(vtm, mesh_block, "mesh")

  return first(close(vtm))
end

function _vtk_multiblock_base_path(path::AbstractString)
  stem, extension = splitext(path)
  extension in (".vtu", ".vtm") && return stem
  return path
end

function _vtk_block_path(base_path::AbstractString, suffix::AbstractString)
  directory = dirname(base_path)
  stem = basename(base_path)
  return joinpath(directory, "$(stem)_$suffix")
end

# PVD files are small XML collection wrappers around previously written VTK
# files. They are kept in the extension because the format is ParaView-specific,
# but the implementation deliberately avoids depending on WriteVTK internals.
function Grico.write_pvd(path::AbstractString, vtk_files::AbstractVector{<:AbstractString};
                         timesteps=nothing)
  values = timesteps === nothing ? collect(eachindex(vtk_files)) .- 1 : collect(timesteps)
  length(values) == length(vtk_files) ||
    throw(ArgumentError("timesteps length $(length(values)) does not match file count $(length(vtk_files))"))

  open(path, "w") do io
    println(io, "<?xml version=\"1.0\"?>")
    println(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">")
    println(io, "  <Collection>")

    for index in eachindex(vtk_files)
      relative_path = relpath(vtk_files[index], dirname(path))
      escaped_step = _vtk_xml_attribute(values[index])
      escaped_path = _vtk_xml_attribute(relative_path)
      println(io,
              "    <DataSet timestep=\"$escaped_step\" group=\"\" part=\"0\" file=\"$escaped_path\"/>")
    end

    println(io, "  </Collection>")
    println(io, "</VTKFile>")
  end

  return path
end

# Attribute values are escaped locally because the PVD writer is intentionally a
# minimal collection-file writer rather than a dependency on a full XML library.
function _vtk_xml_attribute(value)
  text = replace(string(value), "&" => "&amp;")
  text = replace(text, "\"" => "&quot;")
  text = replace(text, "<" => "&lt;")
  text = replace(text, ">" => "&gt;")
  return replace(text, "'" => "&apos;")
end

# Grico stores point, cell, and field datasets in one common list-of-pairs form.
# WriteVTK distinguishes those locations by tag objects on assignment, so this
# routine is the single location mapping for all solution-grid writes.
function _write_vtk_datasets!(vtk, point_datasets, cell_datasets, field_datasets)
  for (name, data) in point_datasets
    vtk[name, VTKPointData()] = data
  end

  for (name, data) in cell_datasets
    vtk[name, VTKCellData()] = data
  end

  for (name, data) in field_datasets
    vtk[name, VTKFieldData()] = data
  end

  return vtk
end

function _vtk_point_matrix(mesh::Grico.SampledMesh{D,T}) where {D,T<:AbstractFloat}
  return _vtk_point_matrix(mesh.points, T)
end

function _vtk_point_matrix(skeleton::Grico.SampledMeshSkeleton{D,T}) where {D,T<:AbstractFloat}
  return _vtk_point_matrix(skeleton.points, T)
end

# WriteVTK expects unstructured-grid points as a `3 × N` coordinate matrix even
# for one- and two-dimensional domains. Missing coordinates are padded by this
# adapter instead of being stored in Grico's backend-neutral sampled data.
function _vtk_point_matrix(points::AbstractMatrix{T}, ::Type{T}) where {T<:AbstractFloat}
  dimension = size(points, 1)
  matrix = Matrix{T}(undef, 3, size(points, 2))

  for point_index in axes(points, 2)
    @inbounds for axis in 1:dimension
      matrix[axis, point_index] = points[axis, point_index]
    end

    @inbounds for axis in (dimension+1):3
      matrix[axis, point_index] = zero(T)
    end
  end

  return matrix
end

# Skeleton overlays are written as ordinary VTK line cells. The skeleton already
# stores deduplicated one-based vertex indices, so no additional connectivity
# transformation is needed here.
function _vtk_line_cells(skeleton::Grico.SampledMeshSkeleton)
  cells = Vector{MeshCell}(undef, size(skeleton.edges, 2))

  for edge_index in axes(skeleton.edges, 2)
    cells[edge_index] = MeshCell(VTKCellTypes.VTK_LINE,
                                 [skeleton.edges[1, edge_index], skeleton.edges[2, edge_index]])
  end

  return cells
end

# Reconstruct the VTK Lagrange cells from the structured point numbering stored
# in `SampledMesh`. Points are contiguous by leaf and lexicographically ordered
# in the same tensor-product traversal used by the sampler.
function _vtk_lagrange_cells(mesh::Grico.SampledMesh{D}) where {D}
  local_point_count = mesh.point_stride^D
  cell_indices = CartesianIndices(ntuple(_ -> mesh.subdivisions, D))
  cells = Vector{MeshCell}(undef, length(mesh.cell_leaves))

  for leaf_index in eachindex(mesh.leaf_data)
    point_offset = (leaf_index - 1) * local_point_count
    cell_offset = (leaf_index - 1) * mesh.cells_per_leaf
    local_points = collect((point_offset+1):(point_offset+local_point_count))
    local_cell_offset = 1

    for I in cell_indices
      cells[cell_offset+local_cell_offset] = _vtk_cell(local_points, mesh.point_stride,
                                                       mesh.sample_degree, I, Val(D))
      local_cell_offset += 1
    end
  end

  return cells
end

# The three `_vtk_cell` specializations assemble VTK connectivity in the exact
# vertex/edge/face/interior ordering required by VTK Lagrange cells. The helper
# routines below append interior points on lines, planes, and volumes after the
# corner points, mirroring VTK's canonical ordering.
function _vtk_cell(local_points::AbstractVector{Int}, point_stride::Int, degree::Int,
                   index::CartesianIndex{1}, ::Val{1})
  start = ((index[1] - 1) * degree + 1,)
  connectivity = Vector{Int}(undef, degree + 1)
  offset = 1
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride, start)
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride, (start[1] + degree,))
  _vtk_push_line!(connectivity, offset, local_points, point_stride, start, 1, degree)
  return MeshCell(VTKCellTypes.VTK_LAGRANGE_CURVE, connectivity)
end

function _vtk_cell(local_points::AbstractVector{Int}, point_stride::Int, degree::Int,
                   index::CartesianIndex{2}, ::Val{2})
  start = ((index[1] - 1) * degree + 1, (index[2] - 1) * degree + 1)
  connectivity = Vector{Int}(undef, (degree + 1)^2)
  offset = 1
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride, start)
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1] + degree, start[2]))
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1] + degree, start[2] + degree))
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1], start[2] + degree))
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride, start, 1, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1] + degree, start[2]), 2, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1], start[2] + degree), 1, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride, start, 2, degree)
  _vtk_push_plane!(connectivity, offset, local_points, point_stride, start, 1, 2, degree)
  return MeshCell(VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL, connectivity)
end

function _vtk_cell(local_points::AbstractVector{Int}, point_stride::Int, degree::Int,
                   index::CartesianIndex{3}, ::Val{3})
  start = ((index[1] - 1) * degree + 1, (index[2] - 1) * degree + 1, (index[3] - 1) * degree + 1)
  connectivity = Vector{Int}(undef, (degree + 1)^3)
  offset = 1
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride, start)
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1] + degree, start[2], start[3]))
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1] + degree, start[2] + degree, start[3]))
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1], start[2] + degree, start[3]))
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1], start[2], start[3] + degree))
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1] + degree, start[2], start[3] + degree))
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1] + degree, start[2] + degree, start[3] + degree))
  offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                            (start[1], start[2] + degree, start[3] + degree))
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride, start, 1, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1] + degree, start[2], start[3]), 2, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1], start[2] + degree, start[3]), 1, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride, start, 2, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1], start[2], start[3] + degree), 1, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1] + degree, start[2], start[3] + degree), 2, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1], start[2] + degree, start[3] + degree), 1, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1], start[2], start[3] + degree), 2, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride, start, 3, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1] + degree, start[2], start[3]), 3, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1] + degree, start[2] + degree, start[3]), 3, degree)
  offset = _vtk_push_line!(connectivity, offset, local_points, point_stride,
                           (start[1], start[2] + degree, start[3]), 3, degree)
  offset = _vtk_push_plane!(connectivity, offset, local_points, point_stride, start, 2, 3, degree)
  offset = _vtk_push_plane!(connectivity, offset, local_points, point_stride,
                            (start[1] + degree, start[2], start[3]), 2, 3, degree)
  offset = _vtk_push_plane!(connectivity, offset, local_points, point_stride, start, 1, 3, degree)
  offset = _vtk_push_plane!(connectivity, offset, local_points, point_stride,
                            (start[1], start[2] + degree, start[3]), 1, 3, degree)
  offset = _vtk_push_plane!(connectivity, offset, local_points, point_stride, start, 1, 2, degree)
  offset = _vtk_push_plane!(connectivity, offset, local_points, point_stride,
                            (start[1], start[2], start[3] + degree), 1, 2, degree)
  _vtk_push_volume!(connectivity, offset, local_points, point_stride, start, degree)
  return MeshCell(VTKCellTypes.VTK_LAGRANGE_HEXAHEDRON, connectivity)
end

# The push helpers append one geometric entity at a time to a VTK Lagrange-cell
# connectivity array. They keep the dimension-specific cell routines readable
# while preserving VTK's required ordering: first vertices, then edge interiors,
# then face interiors, then volume interiors.
function _vtk_push_point!(connectivity::AbstractVector{Int}, offset::Int,
                          local_points::AbstractVector{Int}, point_stride::Int,
                          index::NTuple{D,Int}) where {D}
  connectivity[offset] = _vtk_local_point(local_points, point_stride, index)
  return offset + 1
end

function _vtk_push_line!(connectivity::AbstractVector{Int}, offset::Int,
                         local_points::AbstractVector{Int}, point_stride::Int, start::NTuple{D,Int},
                         axis::Int, degree::Int) where {D}
  for delta in 1:(degree-1)
    offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                              _vtk_shift_index(start, axis, delta))
  end

  return offset
end

function _vtk_push_plane!(connectivity::AbstractVector{Int}, offset::Int,
                          local_points::AbstractVector{Int}, point_stride::Int,
                          start::NTuple{D,Int}, inner_axis::Int, outer_axis::Int,
                          degree::Int) where {D}
  for outer in 1:(degree-1), inner in 1:(degree-1)
    offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                              _vtk_shift_index(_vtk_shift_index(start, inner_axis, inner),
                                               outer_axis, outer))
  end

  return offset
end

function _vtk_push_volume!(connectivity::AbstractVector{Int}, offset::Int,
                           local_points::AbstractVector{Int}, point_stride::Int,
                           start::NTuple{3,Int}, degree::Int)
  for k in 1:(degree-1), j in 1:(degree-1), i in 1:(degree-1)
    offset = _vtk_push_point!(connectivity, offset, local_points, point_stride,
                              (start[1] + i, start[2] + j, start[3] + k))
  end

  return offset
end

function _vtk_shift_index(index::NTuple{D,Int}, axis::Int, delta::Int) where {D}
  ntuple(i -> i == axis ? index[i] + delta : index[i], D)
end

# Convert one tensor-product local index to the one-based sampled point number
# used by WriteVTK. Grico's sampled points are ordered with axis 1 as the fastest
# varying index, matching Julia's column-major Cartesian traversal.
function _vtk_local_point(local_points::AbstractVector{Int}, point_stride::Int,
                          index::NTuple{D,Int}) where {D}
  linear_index = 1
  stride = 1

  @inbounds for axis in 1:D
    linear_index += (index[axis] - 1) * stride
    stride *= point_stride
  end

  return local_points[linear_index]
end

end

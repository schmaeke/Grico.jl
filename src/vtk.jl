# This file exports discrete hp fields and auxiliary datasets to VTK/ParaView.
# In contrast to assembly and verification, which integrate over one quadrature
# rule at a time, the export path must build an explicit piecewise-polynomial
# mesh that an external visualization tool can read. The strategy is therefore:
#
# 1. choose which discrete fields should appear in the output,
# 2. sample those fields on a tensor-product grid on every active leaf,
# 3. evaluate optional user-provided point, cell, and field datasets on the
#    same exported mesh, and
# 4. package the result as VTK Lagrange cells in the ordering expected by
#    WriteVTK and ParaView.
#
# Two export parameters control the balance between compactness and visual
# fidelity. `export_degree` is the polynomial degree of each emitted VTK
# Lagrange cell, while `subdivisions` controls how many such VTK cells are used
# per active hp leaf along each axis. Increasing `export_degree` raises the
# polynomial order of one exported cell; increasing `subdivisions` splits one hp
# leaf into more VTK cells. Neighboring hp leaves are exported independently, so
# points are intentionally duplicated across interfaces. This avoids imposing a
# single conforming VTK point numbering on locally refined meshes and keeps the
# export logic aligned with the leaf-wise structure used throughout the library.

# Internal export-mesh storage shared by field sampling and custom datasets.

# `_VtkMesh` is the compiled export mesh shared by field sampling and dataset
# evaluation. It stores physical point coordinates, the owning leaf and
# reference coordinates of every sampled point, and the per-cell leaf/reference
# metadata needed for cell-centered custom datasets.
struct _VtkMesh{D,T,C}
  points::Matrix{T}
  point_leaves::Vector{Int}
  point_references::Matrix{T}
  cells::Vector{C}
  leaf_data::Vector{Int}
  cell_leaves::Vector{Int}
  cell_references::Matrix{T}
  cell_centers::Matrix{T}
  cells_per_leaf::Int
end

const _VtkArrayData = Union{AbstractVector,AbstractMatrix}

# Export capability checks and top-level file writers.

"""
    vtk_export_supported(dimension)

Return whether VTK export is available for a given domain dimension.

The current export path supports one-, two-, and three-dimensional domains,
which map naturally to VTK curve, quadrilateral, and hexahedral Lagrange cells.
Higher-dimensional spaces are not representable in the VTK file format used
here.
"""
vtk_export_supported(dimension::Integer) = 1 <= Int(dimension) <= 3

@inline function _vtk_column_tuple(matrix::AbstractMatrix, index::Int, ::Val{D}) where {D}
  return ntuple(axis -> matrix[axis, index], D)
end

"""
    write_vtk(path, state; kwargs...)
    write_vtk(path, domain; state=nothing, fields=nothing, point_data=(), cell_data=(),
              field_data=(), subdivisions=1, export_degree=1, vtk_kwargs...)

Export sampled field data and optional custom datasets to a VTK file.

The `state` form infers the domain from the fields stored in the state layout.
The `domain` form exports geometry even without a state and optionally samples a
selected subset of fields from `state`.

Each active leaf is sampled on a tensor-product grid of
`subdivisions * export_degree + 1` points per axis. The sampled points are then
packaged into `subdivisions^D` VTK Lagrange cells per leaf, each of polynomial
degree `export_degree`. This separates geometric resolution from polynomial
output degree: increasing `subdivisions` produces more cells, while increasing
`export_degree` increases the polynomial order within each exported cell.

Additional datasets may be attached as:

- `point_data`: values on sampled VTK points,
- `cell_data`: values on exported VTK cells, and
- `field_data`: global metadata arrays written as VTK field data.

Point and cell datasets may be given directly as arrays or as callables that
are evaluated on sampled points or cells. The accepted callable signatures are
described by the internal dataset normalization rules in this file.

The returned value is the path of the written VTK file as returned by
`WriteVTK.vtk_grid`. The export is leaf-wise: every active hp leaf contributes
its own structured block of sampled points and `subdivisions^D` VTK Lagrange
cells, even when neighboring leaves touch geometrically. This is the natural
representation for locally refined high-order spaces because it preserves the
piecewise structure of the discrete field without forcing a separate global mesh
merging stage.
"""
function write_vtk(path::AbstractString, state::State; kwargs...)
  layout = field_layout(state)
  domain_data = field_space(fields(layout)[1]).domain
  return write_vtk(path, domain_data; state=state, kwargs...)
end

function write_vtk(path::AbstractString, domain_data::Domain{D,T};
                   state::Union{Nothing,State}=nothing, fields=nothing, point_data=(), cell_data=(),
                   field_data=(), subdivisions::Integer=1, export_degree::Integer=1,
                   vtk_kwargs...) where {D,T<:AbstractFloat}
  return _with_internal_blas_threads() do
    vtk_export_supported(D) ||
      throw(ArgumentError("VTK export requires a domain dimension between 1 and 3"))
    subdivision_count = _checked_positive(subdivisions, "subdivisions")
    vtk_degree = _checked_positive(export_degree, "export_degree")
    vtk_fields = _vtk_fields(domain_data, state, fields)
    mesh = _vtk_mesh_data(domain_data, subdivision_count, vtk_degree)
    sampled_fields = _sample_vtk_fields(vtk_fields, state, mesh, T)
    point_datasets = _vtk_point_datasets(vtk_fields, sampled_fields, point_data, mesh)
    cell_datasets = _vtk_cell_datasets(cell_data, mesh)
    field_datasets = _checked_vtk_datasets(field_data, "field")
    _require_vtk_dataset_sizes(point_datasets, size(mesh.points, 2), "point")
    _require_vtk_dataset_sizes(cell_datasets, length(mesh.cells), "cell")
    grid_kwargs = haskey(vtk_kwargs, :vtkversion) ? (; vtk_kwargs...) :
                  (; vtk_kwargs..., vtkversion=:latest)
    saved_files = vtk_grid(path, mesh.points, mesh.cells; grid_kwargs...) do vtk
      _write_vtk_datasets!(vtk, point_datasets, cell_datasets, field_datasets)
    end
    return only(saved_files)
  end
end

"""
    write_pvd(path, vtk_files; timesteps=nothing)

Write a ParaView collection file that references an ordered sequence of VTK
outputs.

The collection records the files in the order given by `vtk_files`. If
`timesteps` is omitted, integer times `0, 1, …` are used. Otherwise
`timesteps[index]` is written as the timestep attached to `vtk_files[index]`.

The file references are written relative to the directory containing `path`, so
the collection remains portable when the whole output directory is moved
together.
"""
function write_pvd(path::AbstractString, vtk_files::AbstractVector{<:AbstractString};
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
      println(io,
              "    <DataSet timestep=\"$(values[index])\" group=\"\" part=\"0\" file=\"$relative_path\"/>")
    end

    println(io, "  </Collection>")
    println(io, "</VTKFile>")
  end

  return path
end

# Field selection and direct sampling of hp state data on VTK points.

# Resolve which state fields should be exported and validate that they all belong
# to the provided domain/state combination. The exported point-data names are
# derived from `field_name(field)` and must therefore be unique.
function _vtk_fields(domain_data::Domain, state::Union{Nothing,State}, selected_fields)
  if state === nothing
    _empty_vtk_field_selector(selected_fields) && return AbstractField[]
    throw(ArgumentError("field export requires a state"))
  end

  layout = field_layout(state)
  available = collect(fields(layout))
  field_list = if selected_fields === nothing
    available
  elseif selected_fields isa AbstractField
    AbstractField[selected_fields]
  elseif selected_fields isa Tuple || selected_fields isa AbstractVector
    AbstractField[field for field in selected_fields]
  else
    throw(ArgumentError("fields must be a field, tuple of fields, vector of fields, or nothing"))
  end

  names = Set{String}()

  for field in field_list
    field in available || throw(ArgumentError("field does not belong to the provided state"))
    field_space(field).domain === domain_data ||
      throw(ArgumentError("VTK field export requires fields defined on the provided Domain"))
    name = string(field_name(field))
    name in names && throw(ArgumentError("duplicate VTK point-data name $name"))
    push!(names, name)
  end

  return field_list
end

_empty_vtk_field_selector(::Nothing) = true
_empty_vtk_field_selector(fields::Tuple) = isempty(fields)
_empty_vtk_field_selector(fields::AbstractVector) = isempty(fields)
_empty_vtk_field_selector(::Any) = false

# Combine sampled state fields with user-specified point datasets into one list
# of VTK point arrays. User datasets are allowed to depend on the already
# sampled field values, so the sampled-field NamedTuple is passed along.
function _vtk_point_datasets(vtk_fields, sampled_fields, point_data, mesh::_VtkMesh)
  datasets = Pair{String,_VtkArrayData}[]
  names = Set{String}()

  for field in vtk_fields
    name = string(field_name(field))
    push!(datasets, name => sampled_fields[field_name(field)])
    push!(names, name)
  end

  for (name, data) in _vtk_datasets(point_data)
    name in names && throw(ArgumentError("duplicate VTK point-data name $name"))
    push!(datasets, name => _vtk_point_dataset(data, sampled_fields, mesh))
    push!(names, name)
  end

  return datasets
end

# Cell datasets are similar to point datasets, but they are evaluated once per
# exported VTK cell rather than once per sampled point.
function _vtk_cell_datasets(cell_data, mesh::_VtkMesh)
  datasets = Pair{String,_VtkArrayData}[]

  for (name, data) in _checked_vtk_datasets(cell_data, "cell")
    push!(datasets, name => _vtk_cell_dataset(data, mesh))
  end

  return datasets
end

# Sample all requested exported fields on the compiled VTK points. The resulting
# NamedTuple is keyed by symbolic field name and stores one vector (scalar field)
# or matrix (vector field) per field.
function _sample_vtk_fields(vtk_fields, state::Union{Nothing,State}, mesh::_VtkMesh{D,T},
                            ::Type{T}) where {D,T<:AbstractFloat}
  (state === nothing || isempty(vtk_fields)) && return NamedTuple()
  names = Tuple(field_name(field) for field in vtk_fields)
  values = tuple((_sample_vtk_field(field, state, mesh, T) for field in vtk_fields)...)
  return NamedTuple{names}(values)
end

# Field sampling evaluates the hp expansion directly at the exported reference
# points associated with each VTK node. For vector fields the sampled array is
# stored in VTK's expected component-major matrix layout.
function _sample_vtk_field(field::AbstractField, state::State{T}, mesh::_VtkMesh{D,T},
                           ::Type{T}) where {D,T<:AbstractFloat}
  point_total = length(mesh.point_leaves)
  sampled = _vtk_field_samples(T, component_count(field), point_total)

  _run_chunks!(point_total) do first_point, last_point
    for point_index in first_point:last_point
      ξ = _vtk_column_tuple(mesh.point_references, point_index, Val(D))
      value = _field_value_on_leaf(field, state, mesh.point_leaves[point_index], ξ)
      _vtk_store_field_sample!(sampled, value, point_index)
    end
  end

  return sampled
end

function _vtk_field_samples(::Type{T}, component_total::Int,
                            point_total::Int) where {T<:AbstractFloat}
  return component_total == 1 ? Vector{T}(undef, point_total) :
         Matrix{T}(undef, component_total, point_total)
end

@inline function _vtk_store_field_sample!(sampled::AbstractVector, value, point_index::Int)
  sampled[point_index] = value
  return sampled
end

function _vtk_store_field_sample!(sampled::AbstractMatrix, value, point_index::Int)
  @inbounds for component in axes(sampled, 1)
    sampled[component, point_index] = value[component]
  end

  return sampled
end

# Point and cell dataset normalization, callable evaluation, and validation.

# Point datasets may be provided either as arrays with one tuple per exported
# point or as callables. Callable point datasets can depend on
# 1. the physical point `x`,
# 2. `x` plus a NamedTuple of sampled field values at that point, or
# 3. `x`, sampled values, the owning leaf, and the reference coordinate `ξ`.
# This gives custom diagnostics access to both geometric and discrete field data
# without exposing the rest of the export internals.
function _vtk_point_context(sampled_fields, mesh::_VtkMesh{D}, point_index::Int) where {D}
  x = _vtk_column_tuple(mesh.points, point_index, Val(D))
  values = _vtk_point_values(sampled_fields, point_index)
  leaf = mesh.point_leaves[point_index]
  ξ = _vtk_column_tuple(mesh.point_references, point_index, Val(D))
  return x, values, leaf, ξ
end

function _vtk_point_dataset_value(data, x, values, leaf, ξ)
  applicable(data, x, values, leaf, ξ) && return data(x, values, leaf, ξ)
  applicable(data, x, values) && return data(x, values)
  applicable(data, x) && return data(x)
  throw(ArgumentError("point datasets must be vectors, matrices, or callables accepting x, x+values, or x+values+leaf+reference"))
end

function _vtk_point_dataset(data, sampled_fields, mesh::_VtkMesh{D}) where {D}
  data isa _VtkArrayData && return data

  return _collect_vtk_samples(length(mesh.point_leaves), "point",
                              point_index -> begin
                                x, values, leaf, ξ = _vtk_point_context(sampled_fields, mesh,
                                                                        point_index)
                                return _vtk_point_dataset_value(data, x, values, leaf, ξ)
                              end)
end

# Cell datasets follow the same idea, but are evaluated on exported subcells
# rather than nodal points. Callable signatures may depend on the source leaf,
# the physical center of the exported cell, and its reference-space center.
function _vtk_cell_context(mesh::_VtkMesh{D}, cell_index::Int) where {D}
  leaf = mesh.cell_leaves[cell_index]
  x = _vtk_column_tuple(mesh.cell_centers, cell_index, Val(D))
  ξ = _vtk_column_tuple(mesh.cell_references, cell_index, Val(D))
  return leaf, x, ξ
end

function _vtk_cell_dataset_value(data, leaf, x, ξ)
  applicable(data, leaf, x, ξ) && return data(leaf, x, ξ)
  applicable(data, leaf, x) && return data(leaf, x)
  applicable(data, leaf) && return data(leaf)
  throw(ArgumentError("cell datasets must be vectors, matrices, or callables accepting leaf, leaf+x, or leaf+x+reference"))
end

function _vtk_cell_array_dataset(data::_VtkArrayData, mesh::_VtkMesh)
  tuple_count = _vtk_tuple_count(data)
  return tuple_count == length(mesh.leaf_data) ?
         _expand_vtk_leaf_dataset(data, mesh.cells_per_leaf) : data
end

function _vtk_cell_dataset(data, mesh::_VtkMesh{D}) where {D}
  data isa _VtkArrayData && return _vtk_cell_array_dataset(data, mesh)

  return _collect_vtk_samples(length(mesh.cells), "cell",
                              cell_index -> begin
                                leaf, x, ξ = _vtk_cell_context(mesh, cell_index)
                                return _vtk_cell_dataset_value(data, leaf, x, ξ)
                              end)
end

# Repackage sampled exported fields into a NamedTuple of scalar/vector values at
# one point so user-provided point-data callables can depend on already sampled
# state fields.
function _vtk_point_values(sampled_fields::NamedTuple, point_index::Int)
  names = propertynames(sampled_fields)
  values = ntuple(index -> _vtk_sample_value(getfield(sampled_fields, names[index]), point_index),
                  length(names))
  return NamedTuple{names}(values)
end

_vtk_sample_value(values::AbstractVector, point_index::Int) = values[point_index]
function _vtk_sample_value(values::AbstractMatrix, point_index::Int)
  ntuple(component -> values[component, point_index], size(values, 1))
end

# The public API accepts several convenient dataset container formats. These
# helpers normalize them to a common `Vector{Pair{String,Any}}` representation
# and validate name uniqueness before writing.
_vtk_datasets(::Nothing) = Pair{String,Any}[]
_vtk_datasets(data::Pair) = [String(first(data)) => last(data)]
function _vtk_datasets(data::NamedTuple)
  [String(name) => getfield(data, name) for name in propertynames(data)]
end
_vtk_datasets(data::AbstractDict) = [String(name) => value for (name, value) in pairs(data)]

function _vtk_datasets(data::Union{AbstractVector,Tuple})
  all(item -> item isa Pair, data) ||
    throw(ArgumentError("VTK datasets must be provided as a NamedTuple, Dict, Pair, or vector/tuple of Pair values"))
  return [String(first(item)) => last(item) for item in data]
end

function _checked_vtk_datasets(data, location::AbstractString)
  datasets = _vtk_datasets(data)
  names = Set{String}()

  for (name, _) in datasets
    name in names && throw(ArgumentError("duplicate VTK $location-data name $name"))
    push!(names, name)
  end

  return datasets
end

# Transfer the normalized datasets into the WriteVTK handle using the correct
# VTK data-location tag.
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

# Array-valued point and cell datasets must match the exported point/cell counts.
# This is checked after all automatic field sampling and leaf-data expansion have
# been resolved.
function _require_vtk_dataset_sizes(datasets, count::Int, location::AbstractString)
  for (name, data) in datasets
    _vtk_tuple_count(data) == count ||
      throw(ArgumentError("$location dataset $name must have $count tuples"))
  end
end

_vtk_tuple_count(data::AbstractVector) = length(data)
_vtk_tuple_count(data::AbstractMatrix) = size(data, 2)

# Cell datasets are allowed to be specified either per exported cell or per
# active leaf. In the latter case the values are replicated to the
# `cells_per_leaf` exported subcells belonging to that leaf.
function _expand_vtk_leaf_dataset(data::AbstractVector, cells_per_leaf::Int)
  expanded = similar(data, length(data) * cells_per_leaf)

  for leaf_index in eachindex(data), local_cell in 1:cells_per_leaf
    expanded[(leaf_index-1)*cells_per_leaf+local_cell] = data[leaf_index]
  end

  return expanded
end

function _expand_vtk_leaf_dataset(data::AbstractMatrix, cells_per_leaf::Int)
  expanded = similar(data, size(data, 1), size(data, 2) * cells_per_leaf)

  for leaf_index in axes(data, 2), local_cell in 1:cells_per_leaf
    expanded[:, (leaf_index-1)*cells_per_leaf+local_cell] = data[:, leaf_index]
  end

  return expanded
end

# Evaluate a point/cell dataset callable on all samples and pack the result into
# the vector or matrix format expected by WriteVTK. The first sample determines
# whether the dataset is scalar- or vector-valued, and all later samples must
# match that shape.
function _collect_vtk_samples(sample_count::Int, location::AbstractString, sample)
  first_value = sample(1)
  values = _vtk_collected_sample_buffer(first_value, sample_count, location)
  _vtk_store_collected_sample!(values, first_value, 1, location)

  if sample_count > 1
    _run_chunks!(sample_count - 1) do first_item, last_item
      for index in (first_item+1):(last_item+1)
        _vtk_store_collected_sample!(values, sample(index), index, location)
      end
    end
  end

  return values
end

function _vtk_collected_sample_buffer(first_value::Number, sample_count::Int, ::AbstractString)
  return Vector{typeof(first_value)}(undef, sample_count)
end

function _vtk_collected_sample_buffer(first_value::Union{Tuple,AbstractVector}, sample_count::Int,
                                      ::AbstractString)
  return Matrix{eltype(first_value)}(undef, length(first_value), sample_count)
end

function _vtk_collected_sample_buffer(::Any, ::Int, location::AbstractString)
  throw(ArgumentError("$location dataset callables must return scalars, tuples, or vectors"))
end

@inline function _vtk_store_collected_sample!(values::AbstractVector, current::Number, index::Int,
                                              ::AbstractString)
  values[index] = current
  return values
end

function _vtk_store_collected_sample!(values::AbstractMatrix, current::Union{Tuple,AbstractVector},
                                      index::Int, location::AbstractString)
  length(current) == size(values, 1) ||
    throw(ArgumentError("$location dataset callables must return one fixed tuple/vector size"))

  @inbounds for component in axes(values, 1)
    values[component, index] = current[component]
  end

  return values
end

function _vtk_store_collected_sample!(values, ::Any, ::Int, location::AbstractString)
  throw(ArgumentError("$location dataset callables must return scalars, tuples, or vectors"))
end

# Tensor-product export-mesh construction on active hp leaves.

# Build the tensor-product VTK export mesh for the current active leaves. Every
# active leaf is sampled independently on a structured grid in reference space,
# then mapped to physical coordinates. The resulting points are not shared
# between neighboring leaves; duplicating them keeps the connectivity compact and
# allows each exported cell to remain a self-contained Lagrange cell, which is
# the most robust choice for high-order output.
function _vtk_mesh_data(domain_data::Domain{D,T}, subdivisions::Int,
                        export_degree::Int) where {D,T<:AbstractFloat}
  leaf_data = active_leaves(grid(domain_data))
  leaf_count = length(leaf_data)
  point_resolution = subdivisions * export_degree
  point_stride = point_resolution + 1
  local_point_count = point_stride^D
  total_point_count = leaf_count * local_point_count
  total_cells = leaf_count * subdivisions^D
  cells_per_leaf = subdivisions^D
  points = Matrix{T}(undef, 3, total_point_count)
  point_leaves = Vector{Int}(undef, total_point_count)
  point_references = Matrix{T}(undef, D, total_point_count)
  cell_leaves = Vector{Int}(undef, total_cells)
  cell_references = Matrix{T}(undef, D, total_cells)
  cell_centers = Matrix{T}(undef, D, total_cells)
  sample_connectivity = _vtk_cell(fill(1, local_point_count), point_stride, export_degree,
                                  CartesianIndex(ntuple(_ -> 1, D)), Val(D))
  cells = Vector{typeof(sample_connectivity)}(undef, total_cells)
  point_indices = CartesianIndices(ntuple(_ -> point_stride, D))
  cell_indices = CartesianIndices(ntuple(_ -> subdivisions, D))
  _run_chunks!(leaf_count) do first_leaf, last_leaf
    reference_coordinates = Vector{T}(undef, D)
    cell_reference_coordinates = Vector{T}(undef, D)
    local_points = Vector{Int}(undef, local_point_count)

    for leaf_index in first_leaf:last_leaf
      leaf = leaf_data[leaf_index]
      point_offset = (leaf_index - 1) * local_point_count
      cell_offset = (leaf_index - 1) * cells_per_leaf
      local_point_offset = 1

      for I in point_indices
        @inbounds for axis in 1:D
          reference_coordinates[axis] = _vtk_reference_coordinate(I[axis], point_resolution, T)
        end

        mapped = map_from_biunit_cube(domain_data, leaf, Tuple(reference_coordinates))
        point_index = point_offset + local_point_offset
        point_leaves[point_index] = leaf

        @inbounds for axis in 1:D
          points[axis, point_index] = mapped[axis]
          point_references[axis, point_index] = reference_coordinates[axis]
        end

        @inbounds for axis in (D+1):3
          points[axis, point_index] = zero(T)
        end

        local_points[local_point_offset] = point_index
        local_point_offset += 1
      end

      local_cell_offset = 1

      for I in cell_indices
        cell_index = cell_offset + local_cell_offset

        @inbounds for axis in 1:D
          cell_reference_coordinates[axis] = _vtk_subcell_center_coordinate(I[axis], subdivisions,
                                                                            T)
          cell_references[axis, cell_index] = cell_reference_coordinates[axis]
        end

        mapped_center = map_from_biunit_cube(domain_data, leaf, Tuple(cell_reference_coordinates))

        @inbounds for axis in 1:D
          cell_centers[axis, cell_index] = mapped_center[axis]
        end

        cell_leaves[cell_index] = leaf
        cells[cell_index] = _vtk_cell(local_points, point_stride, export_degree, I, Val(D))
        local_cell_offset += 1
      end
    end
  end

  return _VtkMesh{D,T,eltype(cells)}(points, point_leaves, point_references, cells, leaf_data,
                                     cell_leaves, cell_references, cell_centers, cells_per_leaf)
end

# Dimension-specific VTK Lagrange connectivity construction.

# The three `_vtk_cell` specializations assemble VTK connectivity in the exact
# vertex/edge/face/interior ordering required by VTK Lagrange cells. The helper
# routines below append interior points on lines, planes, and volumes after the
# corner points, mirroring the canonical ordering described by VTK.
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

# Map the integer tensor-product sample indices on one leaf to biunit-cube
# reference coordinates. Points use the nodal grid, while cell centers use the
# midpoint of each exported subcell.
function _vtk_reference_coordinate(index::Int, subdivisions::Int,
                                   ::Type{T}) where {T<:AbstractFloat}
  T(2 * (index - 1)) / T(subdivisions) - one(T)
end

function _vtk_subcell_center_coordinate(index::Int, subdivisions::Int,
                                        ::Type{T}) where {T<:AbstractFloat}
  T(2 * index - 1) / T(subdivisions) - one(T)
end

# Dimension-independent tensor-index helpers for connectivity assembly.

# The push helpers linearize tensor-product indices into the local point array
# and append the corresponding point numbers to the VTK connectivity array. They
# encode the dimension-independent bookkeeping behind the explicit VTK ordering
# used in `_vtk_cell`.
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

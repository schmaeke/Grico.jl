# This file builds backend-neutral postprocessing data from Grico fields and
# domains. Assembly and verification usually work one quadrature rule at a time;
# visualization and file export instead need an explicit sampled representation
# of the piecewise-polynomial hp field. The routines below choose compatible
# fields, sample them leaf by leaf on a tensor-product grid, evaluate optional
# user datasets on the same samples, and return ordinary Julia arrays that
# output backends can consume without knowing how hp fields are stored.
#
# The sampled representation is intentionally leaf-wise. Neighboring active
# leaves duplicate interface points instead of sharing a conforming global
# visualization mesh. This preserves the discontinuous or locally refined
# polynomial structure that Grico actually solved on and keeps backend adapters
# from having to merge incompatible local polynomial grids.

const PostprocessArray = Union{AbstractVector,AbstractMatrix}

"""
    SampledMesh{D,T}

Store a leaf-wise sampled mesh used by postprocessing backends.

The matrix `points` has size `D × point_count` and stores physical coordinates.
The matrix `point_references` has size `D × point_count` and stores the
corresponding leaf-local reference coordinate `ξ` in the biunit cube
`[-1, 1]^D`. Cell metadata is stored at the sampled subcell centers in
`cell_centers` and `cell_references`.

Points are deliberately duplicated across active-leaf interfaces. This makes
the sampled object a faithful representation of the piecewise hp structure and
leaves backend-specific point merging, if desired, to downstream code.
"""
struct SampledMesh{D,T}
  points::Matrix{T}
  point_leaves::Vector{Int}
  point_references::Matrix{T}
  leaf_data::Vector{Int}
  cell_leaves::Vector{Int}
  cell_references::Matrix{T}
  cell_centers::Matrix{T}
  point_stride::Int
  subdivisions::Int
  sample_degree::Int
  cells_per_leaf::Int
end

"""
    SampledMeshSkeleton{D,T}

Store a deduplicated active-leaf mesh skeleton for overlays.

The skeleton stores physical vertex coordinates in a `D × vertex_count` matrix
and edge connectivity in a `2 × edge_count` matrix of one-based vertex indices.
Its `cell_data` arrays describe the skeleton edges; the default metadata names
are `"leaf"`, `"axis"`, and `"h_level"`.

Conforming edges share vertices, while hanging interfaces keep both coarse and
fine edges. This makes refinement structure visible without forcing a conforming
edge graph.
"""
struct SampledMeshSkeleton{D,T}
  points::Matrix{T}
  edges::Matrix{Int}
  cell_data::Vector{Pair{String,PostprocessArray}}
end

"""
    SampledPostprocess{D,T}

Collect sampled mesh geometry and datasets for postprocessing backends.

`point_data` arrays have one tuple per sampled point, `cell_data` arrays have
one tuple per sampled subcell, and `field_data` stores global metadata. Scalar
datasets are vectors, while vector-valued datasets are matrices whose columns
are sample tuples.
"""
struct SampledPostprocess{D,T}
  mesh::SampledMesh{D,T}
  point_data::Vector{Pair{String,PostprocessArray}}
  cell_data::Vector{Pair{String,PostprocessArray}}
  field_data::Vector{Pair{String,Any}}
end

"""
    postprocess_supported(dimension)

Return whether Grico can build sampled postprocessing data for `dimension`.

The current sampled representation supports one-, two-, and three-dimensional
domains. Higher-dimensional domains may be valid for numerical work, but the
standard visualization backends targeted by Grico do not have a common direct
representation for them.
"""
postprocess_supported(dimension::Integer) = 1 <= dimension <= 3

"""
    sample_postprocess(state; kwargs...)
    sample_postprocess(space; state=nothing, fields=nothing, point_data=(),
                       cell_data=(), field_data=(), subdivisions=1, sample_degree=1)
    sample_postprocess(domain; state=nothing, fields=nothing, point_data=(),
                       cell_data=(), field_data=(), subdivisions=1, sample_degree=1)

Sample fields and auxiliary datasets on a leaf-wise tensor-product mesh.

The `state` form infers the geometric reference from the first field in the
state layout. The `space` form uses one `HpSpace` as the reference for geometry
and field compatibility checks. The `domain` form samples background geometry
and may also sample selected fields from `state`.

Every active leaf is sampled on `subdivisions * sample_degree + 1` points per
axis and contributes `subdivisions^D` sampled subcells. The sampled subcells are
not a finite-element discretization; they are a postprocessing grid used by VTK,
Makie, or other consumers. Point datasets may be arrays or callables accepting
`x`, `(x, values)`, or `(x, values, leaf, ξ)`. Cell datasets may be arrays or
callables accepting `leaf`, `(leaf, x)`, or `(leaf, x, ξ)`, where `x` is a
physical coordinate and `ξ` is the leaf-local reference coordinate. Point
dataset callbacks that need derivatives can capture a field and call
[`field_gradient`](@ref) with the supplied `leaf` and `ξ`.
"""
function sample_postprocess(state::State; kwargs...)
  layout = field_layout(state)
  space = field_space(fields(layout)[1])
  return sample_postprocess(space; state=state, kwargs...)
end

function sample_postprocess(space::HpSpace{D,T}; state::Union{Nothing,State}=nothing,
                            fields=nothing, point_data=(), cell_data=(), field_data=(),
                            subdivisions::Integer=1,
                            sample_degree::Integer=1) where {D,T<:AbstractFloat}
  return _sample_postprocess(space; state=state, fields=fields, point_data=point_data,
                             cell_data=cell_data, field_data=field_data, subdivisions=subdivisions,
                             sample_degree=sample_degree)
end

function sample_postprocess(domain_data::AbstractDomain{D,T}; state::Union{Nothing,State}=nothing,
                            fields=nothing, point_data=(), cell_data=(), field_data=(),
                            subdivisions::Integer=1,
                            sample_degree::Integer=1) where {D,T<:AbstractFloat}
  return _sample_postprocess(domain_data; state=state, fields=fields, point_data=point_data,
                             cell_data=cell_data, field_data=field_data, subdivisions=subdivisions,
                             sample_degree=sample_degree)
end

function _sample_postprocess(reference; state::Union{Nothing,State}=nothing, fields=nothing,
                             point_data=(), cell_data=(), field_data=(), subdivisions::Integer=1,
                             sample_degree::Integer=1)
  reference_domain = _postprocess_reference_domain(reference)
  D = dimension(reference_domain)
  T = eltype(origin(reference_domain))
  postprocess_supported(D) ||
    throw(ArgumentError("postprocessing requires a domain dimension between 1 and 3"))
  subdivision_count = _checked_positive(subdivisions, "subdivisions")
  checked_sample_degree = _checked_positive(sample_degree, "sample_degree")
  sampled_fields = _postprocess_fields(reference, state, fields)
  sampled_mesh = _sampled_mesh(reference, subdivision_count, checked_sample_degree)
  field_samples = _sample_postprocess_fields(sampled_fields, state, sampled_mesh, T)
  point_datasets = _postprocess_point_datasets(sampled_fields, field_samples, point_data,
                                               sampled_mesh)
  cell_datasets = _postprocess_cell_datasets(cell_data, sampled_mesh)
  field_datasets = _checked_postprocess_datasets(field_data, "field")
  _require_postprocess_dataset_sizes(point_datasets, size(sampled_mesh.points, 2), "point")
  _require_postprocess_dataset_sizes(cell_datasets, length(sampled_mesh.cell_leaves), "cell")
  return SampledPostprocess{D,T}(sampled_mesh, point_datasets, cell_datasets, field_datasets)
end

"""
    write_vtk(args...; kwargs...)

Write sampled Grico postprocessing data through the optional WriteVTK backend.

This symbol is defined by Grico so user code can rely on a stable name. The
actual implementation is provided by the `GricoWriteVTKExt` package extension
and is loaded when `WriteVTK` is loaded in the Julia session. The extension
implements the shared postprocessing API: a `State`, `HpSpace`, or domain
reference together with optional `fields`, `point_data`, `cell_data`,
`field_data`, `subdivisions`, and `sample_degree` keywords.
"""
function write_vtk(args...; kwargs...)
  throw(ArgumentError("VTK export requires WriteVTK. Load it with `using WriteVTK`."))
end

"""
    write_pvd(path, vtk_files; timesteps=nothing)

Write a ParaView collection file through the optional WriteVTK backend.

The implementation is provided by the `GricoWriteVTKExt` package extension and
is loaded when `WriteVTK` is loaded in the Julia session.
"""
function write_pvd(args...; kwargs...)
  throw(ArgumentError("PVD export requires WriteVTK. Load it with `using WriteVTK`."))
end

"""
    plot_field(args...; kwargs...)

Plot a sampled Grico field through the optional Makie backend.

This symbol is an extension hook. Load a Makie backend such as `CairoMakie`,
`GLMakie`, or `WGLMakie` to make plotting methods available. The Makie
extension draws one-dimensional scalar data as line plots and two-dimensional
scalar data as sampled surface-color plots. It implements the same sampled
postprocessing API as `write_vtk`: call `plot_field(state, name; ...)`, call
`plot_field(space_or_domain, name; state=state, ...)`, or pass an already
sampled [`SampledPostprocess`](@ref) object.
"""
function plot_field(args...; kwargs...)
  throw(ArgumentError("field plotting requires Makie. Load a Makie backend such as `using CairoMakie`."))
end

"""
    plot_field!(args...; kwargs...)

Add a sampled Grico field plot to an existing backend plot object.

This symbol is an extension hook. Load a Makie backend such as `CairoMakie`,
`GLMakie`, or `WGLMakie` to make mutating plotting methods available. The
mutating methods accept the same state/reference sampling keywords as
[`plot_field`](@ref).
"""
function plot_field!(args...; kwargs...)
  throw(ArgumentError("field plotting requires Makie. Load a Makie backend such as `using CairoMakie`."))
end

"""
    plot_mesh(args...; kwargs...)

Plot a Grico sampled mesh or active-leaf mesh skeleton through the optional
Makie backend.

This symbol is an extension hook. Load a Makie backend such as `CairoMakie`,
`GLMakie`, or `WGLMakie` to make plotting methods available.
"""
function plot_mesh(args...; kwargs...)
  throw(ArgumentError("mesh plotting requires Makie. Load a Makie backend such as `using CairoMakie`."))
end

"""
    plot_mesh!(args...; kwargs...)

Add a Grico sampled mesh or active-leaf mesh skeleton to an existing backend
plot object.

This symbol is an extension hook. Load a Makie backend such as `CairoMakie`,
`GLMakie`, or `WGLMakie` to make mutating plotting methods available.
"""
function plot_mesh!(args...; kwargs...)
  throw(ArgumentError("mesh plotting requires Makie. Load a Makie backend such as `using CairoMakie`."))
end

@inline function _postprocess_column_tuple(matrix::AbstractMatrix, index::Int, ::Val{D}) where {D}
  return ntuple(axis -> matrix[axis, index], D)
end

# Field selection and direct sampling of hp state data on sampled points.

# Postprocessing needs one common geometric, topological, and physical-domain
# description. Participating fields may come from compatible spaces built on
# equal but not identical domain objects, for example after independently
# compiled adaptive transfers on several spaces.
_postprocess_reference_domain(space::HpSpace) = domain(space)
_postprocess_reference_domain(domain_data::AbstractDomain) = domain_data

_postprocess_reference_active_leaves(space::HpSpace) = active_leaves(space)
function _postprocess_reference_active_leaves(domain_data::AbstractDomain)
  _morton_ordered_snapshot_leaves(grid(domain_data), _domain_active_leaves(domain_data))
end

function _postprocess_matching_domain(space::HpSpace, reference)
  reference_domain = _postprocess_reference_domain(reference)
  space_domain = domain(space)
  root_cell_counts(grid(space_domain)) == root_cell_counts(grid(reference_domain)) || return false
  snapshot(space).active_leaves == _postprocess_reference_active_leaves(reference) || return false
  origin(space_domain) == origin(reference_domain) || return false
  extent(space_domain) == extent(reference_domain) || return false
  periodic_axes(space_domain) == periodic_axes(reference_domain) || return false
  _physical_region(space_domain) === _physical_region(reference_domain) || return false
  _cell_measure(space_domain) == _cell_measure(reference_domain) || return false
  return true
end

# Resolve which state fields should be exported and validate that they all belong
# to the provided domain/state combination. The exported point-data names are
# derived from `field_name(field)` and must therefore be unique.
function _postprocess_fields(reference, state::Union{Nothing,State}, selected_fields)
  if state === nothing
    _empty_postprocess_field_selector(selected_fields) && return AbstractField[]
    throw(ArgumentError("field export requires a state"))
  end

  layout = field_layout(state)
  available = collect(fields(layout))
  field_list = if selected_fields === nothing
    available
  elseif selected_fields isa AbstractField
    AbstractField[selected_fields]
  elseif selected_fields isa Tuple || selected_fields isa AbstractVector
    all(field -> field isa AbstractField, selected_fields) ||
      throw(ArgumentError("fields must contain only AbstractField values"))
    AbstractField[field for field in selected_fields]
  else
    throw(ArgumentError("fields must be a field, tuple of fields, vector of fields, or nothing"))
  end

  names = Set{String}()

  for field in field_list
    field in available || throw(ArgumentError("field does not belong to the provided state"))
    _postprocess_matching_domain(field_space(field), reference) ||
      throw(ArgumentError("postprocess field sampling requires fields defined on the provided export reference"))
    name = string(field_name(field))
    name in names && throw(ArgumentError("duplicate point-data name $name"))
    push!(names, name)
  end

  return field_list
end

_empty_postprocess_field_selector(::Nothing) = true
_empty_postprocess_field_selector(fields::Tuple) = isempty(fields)
_empty_postprocess_field_selector(fields::AbstractVector) = isempty(fields)
_empty_postprocess_field_selector(::Any) = false

# Combine sampled state fields with user-specified point datasets into one list
# of sampled point arrays. User datasets are allowed to depend on the already
# sampled field values, so the sampled-field NamedTuple is passed along.
function _postprocess_point_datasets(postprocess_fields, sampled_fields, point_data,
                                     mesh::SampledMesh)
  datasets = Pair{String,PostprocessArray}[]
  names = Set{String}()

  for field in postprocess_fields
    name = string(field_name(field))
    push!(datasets, name => sampled_fields[field_name(field)])
    push!(names, name)
  end

  for (name, data) in _checked_postprocess_datasets(point_data, "point")
    name in names && throw(ArgumentError("duplicate point-data name $name"))
    push!(datasets, name => _postprocess_point_dataset(data, sampled_fields, mesh))
    push!(names, name)
  end

  return datasets
end

# Cell datasets are similar to point datasets, but they are evaluated once per
# sampled cell rather than once per sampled point.
function _postprocess_cell_datasets(cell_data, mesh::SampledMesh)
  datasets = Pair{String,PostprocessArray}[]

  for (name, data) in _checked_postprocess_datasets(cell_data, "cell")
    push!(datasets, name => _postprocess_cell_dataset(data, mesh))
  end

  return datasets
end

# Sample all requested exported fields on the sampled points. The resulting
# NamedTuple is keyed by symbolic field name and stores one vector (scalar field)
# or matrix (vector field) per field.
function _sample_postprocess_fields(postprocess_fields, state::Union{Nothing,State},
                                    mesh::SampledMesh{D,T}, ::Type{T}) where {D,T<:AbstractFloat}
  (state === nothing || isempty(postprocess_fields)) && return NamedTuple()
  names = Tuple(field_name(field) for field in postprocess_fields)
  values = tuple((_sample_postprocess_field(field, state, mesh, T) for field in postprocess_fields)...)
  return NamedTuple{names}(values)
end

# Field sampling evaluates the hp expansion directly at the exported reference
# points associated with each sampled point. For vector fields the sampled array is
# stored in component-major matrix layout.
struct _PostprocessFieldSampleScratch{D,T<:AbstractFloat}
  basis::_LeafBasisScratch{D,T}
  component_values::Vector{T}
end

function _PostprocessFieldSampleScratch(::Type{T}, ::Val{D},
                                        components::Int) where {D,T<:AbstractFloat}
  return _PostprocessFieldSampleScratch{D,T}(_LeafBasisScratch(T, Val(D)),
                                             Vector{T}(undef, components))
end

function _sample_postprocess_field(field::AbstractField, state::State{T}, mesh::SampledMesh{D,T},
                                   ::Type{T}) where {D,T<:AbstractFloat}
  point_total = length(mesh.point_leaves)
  components = component_count(field)
  sampled = _postprocess_field_samples(T, components, point_total)
  component_coefficients = _component_coefficient_views(state, field)
  scratch = _PostprocessFieldSampleScratch(T, Val(D), components)

  for point_index in 1:point_total
    ξ = _postprocess_column_tuple(mesh.point_references, point_index, Val(D))
    compiled = _compiled_leaf(field_space(field), mesh.point_leaves[point_index])
    _fill_leaf_basis!(scratch.basis.values, compiled.degrees, ξ)
    _leaf_component_values!(scratch.component_values, compiled, component_coefficients,
                            scratch.basis.values)
    _postprocess_store_field_sample!(sampled, scratch.component_values, point_index)
  end

  return sampled
end

function _postprocess_field_samples(::Type{T}, component_total::Int,
                                    point_total::Int) where {T<:AbstractFloat}
  return component_total == 1 ? Vector{T}(undef, point_total) :
         Matrix{T}(undef, component_total, point_total)
end

@inline function _postprocess_store_field_sample!(sampled::AbstractVector{T},
                                                  values::AbstractVector{T},
                                                  point_index::Int) where {T<:AbstractFloat}
  sampled[point_index] = values[1]
  return sampled
end

function _postprocess_store_field_sample!(sampled::AbstractMatrix{T}, values::AbstractVector{T},
                                          point_index::Int) where {T<:AbstractFloat}
  @inbounds for component in axes(sampled, 1)
    sampled[component, point_index] = values[component]
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
function _postprocess_point_context(sampled_fields, mesh::SampledMesh{D},
                                    point_index::Int) where {D}
  x = _postprocess_column_tuple(mesh.points, point_index, Val(D))
  values = _postprocess_point_values(sampled_fields, point_index)
  leaf = mesh.point_leaves[point_index]
  ξ = _postprocess_column_tuple(mesh.point_references, point_index, Val(D))
  return x, values, leaf, ξ
end

function _postprocess_point_dataset_value(data, x, values, leaf, ξ)
  applicable(data, x, values, leaf, ξ) && return data(x, values, leaf, ξ)
  applicable(data, x, values) && return data(x, values)
  applicable(data, x) && return data(x)
  throw(ArgumentError("point datasets must be vectors, matrices, or callables accepting x, x+values, or x+values+leaf+reference"))
end

function _postprocess_point_dataset(data, sampled_fields, mesh::SampledMesh{D}) where {D}
  data isa PostprocessArray && return data

  return _collect_postprocess_samples(length(mesh.point_leaves), "point",
                                      point_index -> begin
                                        x, values, leaf, ξ = _postprocess_point_context(sampled_fields,
                                                                                        mesh,
                                                                                        point_index)
                                        return _postprocess_point_dataset_value(data, x, values,
                                                                                leaf, ξ)
                                      end)
end

# Cell datasets follow the same idea, but are evaluated on exported subcells
# rather than nodal points. Callable signatures may depend on the source leaf,
# the physical center of the exported cell, and its reference-space center.
function _postprocess_cell_context(mesh::SampledMesh{D}, cell_index::Int) where {D}
  leaf = mesh.cell_leaves[cell_index]
  x = _postprocess_column_tuple(mesh.cell_centers, cell_index, Val(D))
  ξ = _postprocess_column_tuple(mesh.cell_references, cell_index, Val(D))
  return leaf, x, ξ
end

function _postprocess_cell_dataset_value(data, leaf, x, ξ)
  applicable(data, leaf, x, ξ) && return data(leaf, x, ξ)
  applicable(data, leaf, x) && return data(leaf, x)
  applicable(data, leaf) && return data(leaf)
  throw(ArgumentError("cell datasets must be vectors, matrices, or callables accepting leaf, leaf+x, or leaf+x+reference"))
end

function _postprocess_cell_array_dataset(data::PostprocessArray, mesh::SampledMesh)
  tuple_count = _postprocess_tuple_count(data)
  return tuple_count == length(mesh.leaf_data) ?
         _expand_postprocess_leaf_dataset(data, mesh.cells_per_leaf) : data
end

function _postprocess_cell_dataset(data, mesh::SampledMesh{D}) where {D}
  data isa PostprocessArray && return _postprocess_cell_array_dataset(data, mesh)

  return _collect_postprocess_samples(length(mesh.cell_leaves), "cell",
                                      cell_index -> begin
                                        leaf, x, ξ = _postprocess_cell_context(mesh, cell_index)
                                        return _postprocess_cell_dataset_value(data, leaf, x, ξ)
                                      end)
end

# Repackage sampled exported fields into a NamedTuple of scalar/vector values at
# one point so user-provided point-data callables can depend on already sampled
# state fields.
function _postprocess_point_values(sampled_fields::NamedTuple, point_index::Int)
  names = propertynames(sampled_fields)
  values = ntuple(index -> _postprocess_sample_value(getfield(sampled_fields, names[index]),
                                                     point_index), length(names))
  return NamedTuple{names}(values)
end

_postprocess_sample_value(values::AbstractVector, point_index::Int) = values[point_index]
function _postprocess_sample_value(values::AbstractMatrix, point_index::Int)
  ntuple(component -> values[component, point_index], size(values, 1))
end

# The public API accepts several convenient dataset container formats. These
# helpers normalize them to a common `Vector{Pair{String,Any}}` representation
# and validate name uniqueness before writing.
_postprocess_dataset_name(name::Union{AbstractString,Symbol}, ::AbstractString) = String(name)
function _postprocess_dataset_name(name, location::AbstractString)
  throw(ArgumentError("postprocess $location-data names must be strings or symbols"))
end

_postprocess_datasets(::Nothing, ::AbstractString) = Pair{String,Any}[]
function _postprocess_datasets(data::Pair, location::AbstractString)
  [_postprocess_dataset_name(first(data), location) => last(data)]
end
function _postprocess_datasets(data::NamedTuple, location::AbstractString)
  [_postprocess_dataset_name(name, location) => getfield(data, name)
   for name in propertynames(data)]
end
function _postprocess_datasets(data::AbstractDict, location::AbstractString)
  [_postprocess_dataset_name(name, location) => value for (name, value) in pairs(data)]
end

function _postprocess_datasets(data::Union{AbstractVector,Tuple}, location::AbstractString)
  all(item -> item isa Pair, data) ||
    throw(ArgumentError("postprocess $location-data datasets must be provided as a NamedTuple, Dict, Pair, or vector/tuple of Pair values"))
  return [_postprocess_dataset_name(first(item), location) => last(item) for item in data]
end

function _postprocess_datasets(_, location::AbstractString)
  throw(ArgumentError("postprocess $location-data datasets must be provided as a NamedTuple, Dict, Pair, or vector/tuple of Pair values"))
end

function _checked_postprocess_datasets(data, location::AbstractString)
  datasets = _postprocess_datasets(data, location)
  names = Set{String}()

  for (name, _) in datasets
    name in names && throw(ArgumentError("duplicate postprocess $location-data name $name"))
    push!(names, name)
  end

  return datasets
end

# Array-valued point and cell datasets must match the exported point/cell counts.
# This is checked after all automatic field sampling and leaf-data expansion have
# been resolved.
function _require_postprocess_dataset_sizes(datasets, count::Int, location::AbstractString)
  for (name, data) in datasets
    _postprocess_tuple_count(data) == count ||
      throw(ArgumentError("$location dataset $name must have $count tuples"))
  end
end

_postprocess_tuple_count(data::AbstractVector) = length(data)
_postprocess_tuple_count(data::AbstractMatrix) = size(data, 2)

# Cell datasets are allowed to be specified either per exported cell or per
# active leaf. In the latter case the values are replicated to the
# `cells_per_leaf` exported subcells belonging to that leaf.
function _expand_postprocess_leaf_dataset(data::AbstractVector, cells_per_leaf::Int)
  expanded = similar(data, length(data) * cells_per_leaf)

  for leaf_index in eachindex(data), local_cell in 1:cells_per_leaf
    expanded[(leaf_index-1)*cells_per_leaf+local_cell] = data[leaf_index]
  end

  return expanded
end

function _expand_postprocess_leaf_dataset(data::AbstractMatrix, cells_per_leaf::Int)
  expanded = similar(data, size(data, 1), size(data, 2) * cells_per_leaf)

  for leaf_index in axes(data, 2), local_cell in 1:cells_per_leaf
    expanded[:, (leaf_index-1)*cells_per_leaf+local_cell] = data[:, leaf_index]
  end

  return expanded
end

# Evaluate a point/cell dataset callable on all samples and pack the result into
# the vector or matrix format expected by postprocessing. The first sample determines
# whether the dataset is scalar- or vector-valued, and all later samples must
# match that shape.
function _collect_postprocess_samples(sample_count::Int, location::AbstractString, sample)
  first_value = sample(1)
  values = _postprocess_collected_sample_buffer(first_value, sample_count, location)
  _postprocess_store_collected_sample!(values, first_value, 1, location)

  if sample_count > 1
    for index in 2:sample_count
      _postprocess_store_collected_sample!(values, sample(index), index, location)
    end
  end

  return values
end

function _postprocess_collected_sample_buffer(first_value::Number, sample_count::Int,
                                              ::AbstractString)
  return Vector{typeof(first_value)}(undef, sample_count)
end

function _postprocess_collected_sample_buffer(first_value::Union{Tuple,AbstractVector},
                                              sample_count::Int, location::AbstractString)
  !isempty(first_value) ||
    throw(ArgumentError("$location dataset callables must return at least one tuple/vector component"))
  return Matrix{_postprocess_sample_component_type(first_value)}(undef, length(first_value),
                                                                 sample_count)
end

function _postprocess_collected_sample_buffer(::Any, ::Int, location::AbstractString)
  throw(ArgumentError("$location dataset callables must return scalars, tuples, or vectors"))
end

_postprocess_sample_component_type(first_value::Tuple) = Base.promote_typeof(first_value...)
_postprocess_sample_component_type(first_value::AbstractVector) = eltype(first_value)

@inline function _postprocess_store_collected_sample!(values::AbstractVector, current::Number,
                                                      index::Int, ::AbstractString)
  values[index] = current
  return values
end

function _postprocess_store_collected_sample!(values::AbstractMatrix,
                                              current::Union{Tuple,AbstractVector}, index::Int,
                                              location::AbstractString)
  length(current) == size(values, 1) ||
    throw(ArgumentError("$location dataset callables must return one fixed tuple/vector size"))

  @inbounds for component in axes(values, 1)
    values[component, index] = current[component]
  end

  return values
end

function _postprocess_store_collected_sample!(values, ::Any, ::Int, location::AbstractString)
  throw(ArgumentError("$location dataset callables must return scalars, tuples, or vectors"))
end

# Tensor-product sampled-mesh construction on active hp leaves.

# Build the tensor-product sampled mesh for the current active leaves. Every
# active leaf is sampled independently on a structured grid in reference space
# and mapped to physical coordinates. The resulting points are not shared across
# leaf interfaces, which keeps the sampled data aligned with the hp topology.
function _sampled_mesh(reference, subdivisions::Int, sample_degree::Int)
  return _sampled_mesh(_postprocess_reference_domain(reference),
                       _postprocess_reference_active_leaves(reference), subdivisions, sample_degree)
end

function _sampled_mesh(domain_data::AbstractDomain{D,T}, leaf_data::AbstractVector{<:Integer},
                       subdivisions::Int, sample_degree::Int) where {D,T<:AbstractFloat}
  leaf_count = length(leaf_data)
  point_resolution = subdivisions * sample_degree
  point_stride = point_resolution + 1
  local_point_count = point_stride^D
  total_point_count = leaf_count * local_point_count
  total_cells = leaf_count * subdivisions^D
  cells_per_leaf = subdivisions^D
  points = Matrix{T}(undef, D, total_point_count)
  point_leaves = Vector{Int}(undef, total_point_count)
  point_references = Matrix{T}(undef, D, total_point_count)
  cell_leaves = Vector{Int}(undef, total_cells)
  cell_references = Matrix{T}(undef, D, total_cells)
  cell_centers = Matrix{T}(undef, D, total_cells)
  point_indices = CartesianIndices(ntuple(_ -> point_stride, D))
  cell_indices = CartesianIndices(ntuple(_ -> subdivisions, D))
  reference_coordinates = Vector{T}(undef, D)
  cell_reference_coordinates = Vector{T}(undef, D)

  for leaf_index in 1:leaf_count
    leaf = leaf_data[leaf_index]
    point_offset = (leaf_index - 1) * local_point_count
    cell_offset = (leaf_index - 1) * cells_per_leaf
    local_point_offset = 1

    for I in point_indices
      @inbounds for axis in 1:D
        reference_coordinates[axis] = _postprocess_reference_coordinate(I[axis], point_resolution,
                                                                        T)
      end

      mapped = map_from_biunit_cube(domain_data, leaf, Tuple(reference_coordinates))
      point_index = point_offset + local_point_offset
      point_leaves[point_index] = leaf

      @inbounds for axis in 1:D
        points[axis, point_index] = mapped[axis]
        point_references[axis, point_index] = reference_coordinates[axis]
      end

      local_point_offset += 1
    end

    local_cell_offset = 1

    for I in cell_indices
      cell_index = cell_offset + local_cell_offset

      @inbounds for axis in 1:D
        cell_reference_coordinates[axis] = _postprocess_subcell_center_coordinate(I[axis],
                                                                                  subdivisions, T)
        cell_references[axis, cell_index] = cell_reference_coordinates[axis]
      end

      mapped_center = map_from_biunit_cube(domain_data, leaf, Tuple(cell_reference_coordinates))

      @inbounds for axis in 1:D
        cell_centers[axis, cell_index] = mapped_center[axis]
      end

      cell_leaves[cell_index] = leaf
      local_cell_offset += 1
    end
  end

  return SampledMesh{D,T}(points, point_leaves, point_references, Int.(leaf_data), cell_leaves,
                          cell_references, cell_centers, point_stride, subdivisions, sample_degree,
                          cells_per_leaf)
end

"""
    sample_mesh_skeleton(space)
    sample_mesh_skeleton(domain)

Sample the active-leaf mesh skeleton for plotting or file-export overlays.

The returned [`SampledMeshSkeleton`](@ref) contains deduplicated physical
vertices and one line segment per visible active-leaf edge. Conforming leaf
interfaces share vertices, while hanging interfaces keep both the coarse edge
and the finer subedges so the refinement structure remains visible.
"""
function sample_mesh_skeleton(space::HpSpace{D,T}) where {D,T<:AbstractFloat}
  postprocess_supported(D) ||
    throw(ArgumentError("mesh-skeleton sampling requires a domain dimension between 1 and 3"))
  return _sample_mesh_skeleton(space)
end

function sample_mesh_skeleton(domain_data::AbstractDomain{D,T}) where {D,T<:AbstractFloat}
  postprocess_supported(D) ||
    throw(ArgumentError("mesh-skeleton sampling requires a domain dimension between 1 and 3"))
  return _sample_mesh_skeleton(domain_data)
end

# True active-leaf mesh skeleton construction uses one global dyadic vertex map,
# so conforming leaf edges share endpoints. Equal edges are deduplicated, while
# hanging interfaces intentionally keep both the coarse edge and finer subedges.
function _sample_mesh_skeleton(reference)
  domain_data = _postprocess_reference_domain(reference)
  leaf_data = _postprocess_reference_active_leaves(reference)
  return _sample_mesh_skeleton(domain_data, leaf_data)
end

function _sample_mesh_skeleton(domain_data::AbstractDomain{D,T},
                               leaf_data::AbstractVector{<:Integer}) where {D,T<:AbstractFloat}
  grid_data = grid(domain_data)
  max_levels = _mesh_skeleton_levels(grid_data, leaf_data, Val(D))
  vertex_lookup = Dict{NTuple{D,Int128},Int}()
  point_data = NTuple{D,T}[]
  edge_lookup = Set{NTuple{2,Int}}()
  edge_data = NTuple{2,Int}[]
  edge_leaves = Int[]
  edge_axes = Int[]
  edge_levels = Int[]

  for leaf in leaf_data
    checked_leaf = _checked_cell(grid_data, leaf)

    for axis in 1:D
      for side_index in 0:(2^(D-1)-1)
        lower_corner = _mesh_skeleton_edge_corner(axis, side_index, 0, Val(D))
        upper_corner = _mesh_skeleton_edge_corner(axis, side_index, 1, Val(D))
        lower_key = _mesh_skeleton_vertex_key(grid_data, checked_leaf, lower_corner, max_levels)
        upper_key = _mesh_skeleton_vertex_key(grid_data, checked_leaf, upper_corner, max_levels)
        lower_vertex = _mesh_skeleton_vertex_index!(vertex_lookup, point_data, domain_data,
                                                    lower_key, max_levels)
        upper_vertex = _mesh_skeleton_vertex_index!(vertex_lookup, point_data, domain_data,
                                                    upper_key, max_levels)
        edge_key = lower_vertex <= upper_vertex ? (lower_vertex, upper_vertex) :
                   (upper_vertex, lower_vertex)
        edge_key in edge_lookup && continue
        push!(edge_lookup, edge_key)
        push!(edge_data, edge_key)
        push!(edge_leaves, checked_leaf)
        push!(edge_axes, axis)
        push!(edge_levels, maximum(level(grid_data, checked_leaf)))
      end
    end
  end

  cell_datasets = Pair{String,PostprocessArray}["leaf" => edge_leaves, "axis" => edge_axes,
                                                "h_level" => edge_levels]
  return SampledMeshSkeleton{D,T}(_postprocess_point_matrix(point_data, Val(D), T),
                                  _postprocess_edge_matrix(edge_data), cell_datasets)
end

# Choose the finest logical lattice needed to represent all active-leaf
# vertices. All skeleton vertices are keyed on this common dyadic lattice so
# shared geometric vertices receive the same sampled point number.
function _mesh_skeleton_levels(grid_data::CartesianGrid, leaf_data, ::Val{D}) where {D}
  isempty(leaf_data) && return ntuple(_ -> 0, D)
  return ntuple(axis -> maximum(level(grid_data, leaf, axis) for leaf in leaf_data), D)
end

# Enumerate one corner pair of an active leaf edge. The edge axis selects the
# moving coordinate, while `side_index` encodes the fixed side bits on all
# tangential axes.
function _mesh_skeleton_edge_corner(edge_axis::Int, side_index::Int, endpoint::Int,
                                    ::Val{D}) where {D}
  return ntuple(axis -> axis == edge_axis ? endpoint :
                        (side_index >> (axis < edge_axis ? axis - 1 : axis - 2)) & 1, D)
end

# Convert a leaf-local corner into a global integer vertex key. The topology
# helpers perform the same endpoint scaling used for neighbor comparisons,
# which keeps the sampled skeleton aligned with the refinement-tree semantics.
function _mesh_skeleton_vertex_key(grid_data::CartesianGrid{D}, leaf::Int, corner::NTuple{D,Int},
                                   max_levels::NTuple{D,Int}) where {D}
  return ntuple(axis -> corner[axis] == 0 ?
                        _scaled_lower_coordinate(grid_data, leaf, axis, max_levels[axis]) :
                        _scaled_upper_coordinate(grid_data, leaf, axis, max_levels[axis]), D)
end

# Look up or insert the sampled point for one logical vertex. The dictionary is
# the connectivity bridge that makes conforming leaf interfaces share point
# numbers in the skeleton.
function _mesh_skeleton_vertex_index!(lookup::Dict{NTuple{D,Int128},Int},
                                      points::Vector{NTuple{D,T}}, domain_data::AbstractDomain{D,T},
                                      key::NTuple{D,Int128},
                                      max_levels::NTuple{D,Int}) where {D,T<:AbstractFloat}
  index = get(lookup, key, 0)
  index != 0 && return index
  point = _mesh_skeleton_vertex_point(domain_data, key, max_levels)
  push!(points, point)
  lookup[key] = length(points)
  return length(points)
end

# Map one global dyadic vertex key to physical coordinates. This repeats the
# affine background-domain formula directly instead of mapping through an
# arbitrary owning leaf, so equal logical vertices are converted to bitwise equal
# physical points.
function _mesh_skeleton_vertex_point(domain_data::AbstractDomain{D,T}, key::NTuple{D,Int128},
                                     max_levels::NTuple{D,Int}) where {D,T<:AbstractFloat}
  root_counts = root_cell_counts(grid(domain_data))
  return ntuple(axis -> origin(domain_data, axis) +
                        ldexp(extent(domain_data, axis) * T(key[axis]) / T(root_counts[axis]),
                              -max_levels[axis]), D)
end

# Pack skeleton point tuples into the same `D × N` column-major coordinate
# layout used by sampled solution points.
function _postprocess_point_matrix(points::Vector{NTuple{D,T}}, ::Val{D},
                                   ::Type{T}) where {D,T<:AbstractFloat}
  matrix = Matrix{T}(undef, D, length(points))

  for (index, point) in enumerate(points)
    @inbounds for axis in 1:D
      matrix[axis, index] = point[axis]
    end
  end

  return matrix
end

function _postprocess_edge_matrix(edges::Vector{NTuple{2,Int}})
  matrix = Matrix{Int}(undef, 2, length(edges))

  for (index, edge) in enumerate(edges)
    matrix[1, index] = edge[1]
    matrix[2, index] = edge[2]
  end

  return matrix
end

# Map the integer tensor-product sample indices on one leaf to biunit-cube
# reference coordinates. Points use the nodal grid, while cell centers use the
# midpoint of each sampled subcell.
function _postprocess_reference_coordinate(index::Int, resolution::Int,
                                           ::Type{T}) where {T<:AbstractFloat}
  T(2 * (index - 1)) / T(resolution) - one(T)
end

function _postprocess_subcell_center_coordinate(index::Int, subdivisions::Int,
                                                ::Type{T}) where {T<:AbstractFloat}
  T(2 * index - 1) / T(subdivisions) - one(T)
end

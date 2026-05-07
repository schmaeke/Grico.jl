# This file contains the setup side of the operator-evaluation layer:
# 1. immutable compiled plan data,
# 2. compilation from editable problem descriptions to `AssemblyPlan`, and
# 3. compilation of Dirichlet and mean-value constraints.
#
# The preceding files establish the ingredients of a discretization:
# - `space.jl` compiles one hp space with the requested per-axis continuity,
# - `fields.jl` groups that space into named unknown blocks,
# - `problem.jl` stores operators and constraints in an editable problem
#   description,
# - and `integration.jl` prepares reusable local evaluation data.
#
# The job of this file is to freeze all of that into one immutable
# `AssemblyPlan`. The plan is the exact algebraic snapshot on which runtime
# matrix-free applications and nonlinear evaluations operate. It records not
# only operators and integration data, but also the compiled form of strong
# Dirichlet constraints and mean-value constraints.
#
# In that sense, `problem.jl` describes which local operators should be applied,
# whereas this file determines how those declarations are translated into
# algebraic rows and eliminated degrees of freedom.
#
# Two global algebraic transformations dominate that translation.
#
# First, strong Dirichlet data are not imposed pointwise on raw nodal values.
# Instead, the code projects the prescribed boundary data onto the active trace
# space carried by the compiled field basis. The outcome is a finite-dimensional
# boundary system whose solution expresses trace dofs as either fixed values or
# affine functions of remaining free dofs.
#
# Second, mean-value constraints are turned into explicit global linear
# functionals integrated against the already compiled basis. Those rows survive
# into the later nonlinear residual/tangent path as ordinary algebraic
# equations.
#
# The file is organized in the same order:
# compiled constraint/plan data first, then the overall compilation pipeline,
# then Dirichlet compilation, and finally mean-value compilation.

# Compiled plan and constraint data structures.

# Compiled row of a linear constraint of the form
#
#   u[pivot] = rhs - Σᵢ coefficients[i] * u[indices[i]]
#
# or, equivalently for residual/tangent evaluation,
#
#   Σᵢ coefficients[i] * u[indices[i]] - rhs = 0
#
# with `pivot` chosen as the algebraic row used to represent the constraint.
struct _CompiledLinearConstraint{T<:AbstractFloat}
  pivot::Int
  indices::Vector{Int}
  coefficients::Vector{T}
  rhs::T
end

# Internal affine relation used for Dirichlet elimination and static
# condensation. The pivot dof is reconstructed from reduced dofs and a constant
# shift after the reduced system has been solved.
struct _CompiledAffineRelation{T<:AbstractFloat}
  pivot::Int
  indices::Vector{Int}
  coefficients::Vector{T}
  rhs::T
end

# Compiled strong Dirichlet data split into:
# - dofs fixed to explicit values,
# - and affine relations for trace dofs that depend on remaining free dofs.
struct _CompiledDirichlet{T<:AbstractFloat}
  fixed_dofs::Vector{Int}
  fixed_values::Vector{T}
  rows::Vector{_CompiledAffineRelation{T}}
end

# Bit masks describing which global rows/dofs are fixed, algebraically
# eliminated, or reserved as explicit constraint rows.
struct _ConstraintMasks{T<:AbstractFloat}
  fixed::BitVector
  fixed_values::Vector{T}
  eliminated::BitVector
  constraint_rows::BitVector
  blocked_rows::BitVector
end

# Batched traversal metadata used by runtime operator and nonlinear evaluation.
# Items that share the same local kernel signature are grouped together so the
# runtime passes can reuse the same local buffer shape and avoid repeated
# selector/filter work inside the hottest loops.
struct _KernelBatch
  item_indices::Vector{Int}
  local_dof_count::Int
  color_ranges::Vector{UnitRange{Int}}
end

struct _FilteredKernelBatch
  item_indices::Vector{Int}
  local_dof_count::Int
  operator_indices::Vector{Int}
  color_ranges::Vector{UnitRange{Int}}
end

struct _TraversalPlan
  cell_batches::Vector{_KernelBatch}
  boundary_batches::Vector{_FilteredKernelBatch}
  interface_batches::Vector{_KernelBatch}
  surface_batches::Vector{_FilteredKernelBatch}
  boundary_operator_lookup::Vector{Vector{Int}}
  surface_operator_lookup::Dict{_SurfaceTag,Vector{Int}}
end

"""
    AssemblyPlan

Compiled operator data for a problem on a fixed field layout and mesh state.

An `AssemblyPlan` stores the validated [`FieldLayout`](@ref), the compiled local
integration data, the operator collections, and the compiled Dirichlet and
mean-value constraints. It is the reusable object that separates the expensive
setup phase from repeated matrix-free applications or nonlinear evaluations.

For affine problems, [`apply!`](@ref) applies the compiled operator action
directly to coefficient vectors. For nonlinear problems, [`residual!`](@ref)
evaluates the compiled residual equations directly on the full field layout.

The plan is tied to the current field layout, active-leaf set, and current
constraint data. If the mesh, space, or problem definition changes, a new plan
must be compiled.
"""
struct AssemblyPlan{D,T<:AbstractFloat,CO<:Tuple,BO<:Tuple,IO<:Tuple,SO<:Tuple,I,OC<:_OperatorClass,
                    S}
  layout::FieldLayout{D,T}
  cell_operators::CO
  boundary_operators::BO
  interface_operators::IO
  surface_operators::SO
  integration::I
  operator_class::OC
  dirichlet::_CompiledDirichlet{T}
  mean_constraints::Vector{_CompiledLinearConstraint{T}}
  constraint_masks::_ConstraintMasks{T}
  traversal_plan::_TraversalPlan
  assembly_structure::S
end

"""
    compile(problem)

Validate a problem description and compile its field layout, integration data,
and strong/linear constraints into an [`AssemblyPlan`](@ref).

Compilation performs all setup steps that depend only on the problem
description, the current space, and the current mesh state: local integration
tables are generated, Dirichlet data are projected to trace dofs, mean-value
constraints are compiled, and the operator collections are frozen into tuples
for efficient later traversal.

For Dirichlet data, "projected to trace dofs" means that the prescribed
boundary values are matched in the `L²` sense on the active trace basis of the
selected field. This is what allows the later operator layer to eliminate
strong boundary data in a basis-agnostic way, even on hanging interfaces and
high-order trace spaces.

The returned plan fixes its mesh, fields, operators, constraints, traversal
data, and reduced-space maps. Runtime workspaces are separate objects, so a
plan can be reused for repeated evaluations without mutating its compiled
problem data.
"""
compile(problem::AffineProblem) = _compile_problem_description(problem, Val(:affine))

compile(problem::ResidualProblem) = _compile_problem_description(problem, Val(:residual))

function _compile_problem_description(problem::_AbstractProblem, assembly_kind)
  _validate_problem_data(problem)
  data = _problem_data(problem)
  return _compile_problem(data.fields, Tuple(data.cell_operators), Tuple(data.boundary_operators),
                          Tuple(data.interface_operators), Tuple(data.surface_operators),
                          data.cell_quadratures, data.embedded_surfaces, data.dirichlet_constraints,
                          data.mean_constraints, data.operator_class, assembly_kind)
end

# Main compilation pipeline from problem description to immutable plan.

# After geometric compilation, surface selectors can be validated against the
# actual embedded-surface quadratures that will participate in evaluation. This
# catches misspelled or otherwise dead symbolic tags at `compile` time.
function _validate_surface_operators(surface_operators, surfaces)
  available_tags = Set{Symbol}()

  for surface in surfaces
    surface.tag === nothing && continue
    push!(available_tags, surface.tag)
  end

  for wrapped in surface_operators
    wrapped.tag === nothing && continue
    wrapped.tag in available_tags && continue
    throw(ArgumentError("surface operator targets embedded-surface tag $(repr(wrapped.tag)), but compilation produced no embedded surface quadratures with that tag"))
  end

  return nothing
end

# Shared compilation path for affine and residual problems. The resulting plan
# is structurally the same; later operator evaluation decides which operator
# callbacks are used.
function _compile_problem(fields, cell_operators, boundary_operators, interface_operators,
                          surface_operators, cell_quadratures, embedded_surfaces,
                          dirichlet_constraints, mean_constraints, operator_class, assembly_kind)
  layout = _field_layout(fields)
  integration = _compile_integration(layout, cell_quadratures, embedded_surfaces;
                                     include_interfaces=(!isempty(interface_operators)))
  _validate_surface_operators(surface_operators, integration.embedded_surfaces)
  compiled_dirichlet = _compile_dirichlet(layout, integration.boundary_faces, dirichlet_constraints)
  compiled_mean_constraints = _compile_mean_constraints(layout, integration.cells,
                                                        compiled_dirichlet, mean_constraints)
  constraint_masks = _constraint_masks(dof_count(layout), compiled_dirichlet,
                                       compiled_mean_constraints)
  traversal_plan = _compile_traversal_plan(dimension(layout), integration, boundary_operators,
                                           interface_operators, surface_operators)
  assembly_structure = _compile_assembly_structure(assembly_kind, layout, cell_operators,
                                                   boundary_operators, interface_operators,
                                                   surface_operators, integration,
                                                   compiled_dirichlet, compiled_mean_constraints,
                                                   constraint_masks)
  return AssemblyPlan(layout, cell_operators, boundary_operators, interface_operators,
                      surface_operators, integration, operator_class, compiled_dirichlet,
                      compiled_mean_constraints, constraint_masks, traversal_plan,
                      assembly_structure)
end

# Compile traversal metadata for operator and nonlinear evaluation. Items are
# batched by local kernel signature and then greedily colored so each color can
# scatter directly into the global vector without write conflicts.
function _compile_traversal_plan(dimension::Int, integration::_CompiledIntegration,
                                 boundary_operators, interface_operators, surface_operators)
  boundary_lookup = _boundary_operator_lookup(dimension, boundary_operators)
  surface_lookup = _surface_operator_lookup(integration.embedded_surfaces, surface_operators)
  cell_batches = _compile_kernel_batches(integration.cells, _cell_kernel_signature)
  interface_batches = isempty(interface_operators) ? _KernelBatch[] :
                      _compile_kernel_batches(integration.interfaces, _interface_kernel_signature)
  boundary_batches = isempty(boundary_operators) ? _FilteredKernelBatch[] :
                     _compile_filtered_batches(integration.boundary_faces, _face_kernel_signature,
                                               face -> boundary_lookup[_boundary_lookup_slot(face.axis,
                                                                                             face.side)])
  surface_batches = isempty(surface_operators) ? _FilteredKernelBatch[] :
                    _compile_filtered_batches(integration.embedded_surfaces,
                                              _surface_kernel_signature,
                                              surface -> _surface_operator_indices(surface_lookup,
                                                                                   surface.tag))
  return _TraversalPlan(cell_batches, boundary_batches, interface_batches, surface_batches,
                        boundary_lookup, surface_lookup)
end

@inline _boundary_lookup_slot(axis::Int, side::Int) = 2 * (axis - 1) + side

function _boundary_operator_lookup(dimension::Int, boundary_operators)
  lookup = [Int[] for _ in 1:(2*dimension)]

  for operator_index in eachindex(boundary_operators)
    wrapped = boundary_operators[operator_index]
    push!(lookup[_boundary_lookup_slot(wrapped.boundary.axis, wrapped.boundary.side)],
          operator_index)
  end

  return lookup
end

function _surface_operator_lookup(surfaces, surface_operators)
  wildcard = Int[]
  tagged = Dict{Symbol,Vector{Int}}()

  for operator_index in eachindex(surface_operators)
    wrapped = surface_operators[operator_index]

    if wrapped.tag === nothing
      push!(wildcard, operator_index)
    else
      push!(get!(tagged, wrapped.tag, Int[]), operator_index)
    end
  end

  lookup = Dict{_SurfaceTag,Vector{Int}}(nothing => copy(wildcard))
  available_tags = Set{_SurfaceTag}()
  push!(available_tags, nothing)

  for surface in surfaces
    push!(available_tags, surface.tag)
  end

  for tag in available_tags
    tag === nothing && continue
    indices = copy(wildcard)
    tagged_indices = get(tagged, tag, nothing)
    tagged_indices === nothing || append!(indices, tagged_indices)
    lookup[tag] = indices
  end

  return lookup
end

@inline function _surface_operator_indices(lookup::Dict{_SurfaceTag,Vector{Int}}, tag::_SurfaceTag)
  return get(lookup, tag, lookup[nothing])
end

function _compile_kernel_batches(items, signature_fn)
  batch_indices = Vector{Vector{Int}}()
  batch_local_dofs = Int[]
  lookup = Dict{Any,Int}()

  for item_index in eachindex(items)
    item = items[item_index]
    signature = signature_fn(item)
    batch_index = get!(lookup, signature) do
      push!(batch_indices, Int[])
      push!(batch_local_dofs, item.local_dof_count)
      return length(batch_indices)
    end

    push!(batch_indices[batch_index], item_index)
  end

  batches = _KernelBatch[]

  for index in eachindex(batch_indices)
    colored_indices, color_ranges = _color_item_indices(items, batch_indices[index])
    push!(batches, _KernelBatch(colored_indices, batch_local_dofs[index], color_ranges))
  end

  return batches
end

function _compile_filtered_batches(items, signature_fn, operators_fn)
  batch_indices = Vector{Vector{Int}}()
  batch_local_dofs = Int[]
  batch_operator_indices = Vector{Vector{Int}}()
  lookup = Dict{Any,Int}()

  for item_index in eachindex(items)
    item = items[item_index]
    operator_indices = operators_fn(item)
    isempty(operator_indices) && continue
    signature = signature_fn(item)
    key = (signature, Tuple(operator_indices))
    batch_index = get!(lookup, key) do
      push!(batch_indices, Int[])
      push!(batch_local_dofs, item.local_dof_count)
      push!(batch_operator_indices, copy(operator_indices))
      return length(batch_indices)
    end

    push!(batch_indices[batch_index], item_index)
  end

  batches = _FilteredKernelBatch[]

  for index in eachindex(batch_indices)
    colored_indices, color_ranges = _color_item_indices(items, batch_indices[index])
    push!(batches,
          _FilteredKernelBatch(colored_indices, batch_local_dofs[index],
                               batch_operator_indices[index], color_ranges))
  end

  return batches
end

function _color_item_indices(items, item_indices::Vector{Int})
  color_dofs = BitSet[]
  color_items = Vector{Int}[]

  for item_index in item_indices
    item = @inbounds items[item_index]
    color_index = _find_disjoint_color(color_dofs, item)

    if color_index == 0
      push!(color_dofs, BitSet())
      push!(color_items, Int[])
      color_index = length(color_items)
    end

    _add_item_dofs!(color_dofs[color_index], item)
    push!(color_items[color_index], item_index)
  end

  colored_indices = Int[]
  color_ranges = UnitRange{Int}[]

  for indices in color_items
    first_index = length(colored_indices) + 1
    append!(colored_indices, indices)
    push!(color_ranges, first_index:length(colored_indices))
  end

  return colored_indices, color_ranges
end

function _find_disjoint_color(color_dofs::Vector{BitSet}, item::_AssemblyValues)
  for color_index in eachindex(color_dofs)
    _item_dofs_are_disjoint(color_dofs[color_index], item) && return color_index
  end

  return 0
end

function _item_dofs_are_disjoint(dofs::BitSet, item::_AssemblyValues)
  for local_dof in 1:item.local_dof_count
    for term_index in _local_term_range(item, local_dof)
      item.term_indices[term_index] in dofs && return false
    end
  end

  return true
end

function _add_item_dofs!(dofs::BitSet, item::_AssemblyValues)
  for local_dof in 1:item.local_dof_count
    for term_index in _local_term_range(item, local_dof)
      push!(dofs, item.term_indices[term_index])
    end
  end

  return dofs
end

@inline function _field_kernel_signature(data::_FieldValues)
  return (data.field_id, _field_component_count(data), data.scalar_dof_count, data.local_mode_count,
          length(data.term_indices), length(data.term_offsets))
end

function _field_tuple_signature(fields::Tuple)
  return ntuple(index -> _field_kernel_signature(fields[index]), length(fields))
end

function _cell_kernel_signature(item::CellValues)
  return (:cell, point_count(item), item.local_dof_count, Tuple(item.interior_local_dofs),
          _field_tuple_signature(item.fields))
end

function _face_kernel_signature(item::FaceValues)
  return (:face, item.axis, item.side, point_count(item), item.local_dof_count,
          _field_tuple_signature(item.fields))
end

function _surface_kernel_signature(item::SurfaceValues)
  return (:surface, item.tag, point_count(item), item.local_dof_count,
          _field_tuple_signature(item.fields))
end

function _interface_kernel_signature(item::InterfaceValues)
  return (:interface, item.axis, point_count(item), item.local_dof_count,
          _field_tuple_signature(item.minus_fields), _field_tuple_signature(item.plus_fields))
end

# Build global bit masks describing which dofs are fixed, eliminated, or reserved
# as explicit constraint rows.
function _constraint_masks(ndofs::Int, dirichlet::_CompiledDirichlet{T},
                           mean_constraints::Vector{_CompiledLinearConstraint{T}}) where {T<:AbstractFloat}
  fixed = falses(ndofs)
  fixed_values = zeros(T, ndofs)
  eliminated = falses(ndofs)
  constraint_rows = falses(ndofs)

  for index in eachindex(dirichlet.fixed_dofs)
    dof = dirichlet.fixed_dofs[index]
    fixed[dof] = true
    fixed_values[dof] = dirichlet.fixed_values[index]
  end

  for row in dirichlet.rows
    eliminated[row.pivot] = true
  end

  for constraint in mean_constraints
    constraint_rows[constraint.pivot] = true
  end

  return _ConstraintMasks(fixed, fixed_values, eliminated, constraint_rows,
                          eliminated .| constraint_rows)
end

# Dirichlet compilation by boundary projection onto the active trace space.

# Compile all Dirichlet constraints by projecting the prescribed boundary data
# onto the active trace space of each affected field.
function _compile_dirichlet(layout::FieldLayout{D,T}, boundary_faces,
                            constraints) where {D,T<:AbstractFloat}
  isempty(constraints) && return _CompiledDirichlet{T}(Int[], T[], _CompiledAffineRelation{T}[])
  fixed_dofs = Int[]
  fixed_values = T[]
  rows = _CompiledAffineRelation{T}[]

  for slot in layout.slots
    field_constraints = _field_constraints(constraints, slot.field)
    isempty(field_constraints) && continue
    dofs, matrix, rhs = _boundary_projection_system(layout, slot, boundary_faces, field_constraints)
    isempty(dofs) &&
      throw(ArgumentError("Dirichlet constraint does not match any boundary face for field $(field_name(slot.field))"))
    slot_fixed_dofs, slot_fixed_values, slot_rows = _dirichlet_relations(dofs, Matrix(matrix), rhs,
                                                                         field_name(slot.field))
    append!(fixed_dofs, slot_fixed_dofs)
    append!(fixed_values, slot_fixed_values)
    append!(rows, slot_rows)
  end

  if !isempty(fixed_dofs)
    permutation = sortperm(fixed_dofs)
    fixed_dofs = fixed_dofs[permutation]
    fixed_values = fixed_values[permutation]
  end

  sort!(rows; by=row -> row.pivot)
  return _CompiledDirichlet(fixed_dofs, fixed_values, rows)
end

# Collect the Dirichlet constraints that apply to one field.
function _field_constraints(constraints, field::AbstractField)
  field_id = _field_id(field)
  matched = Dirichlet[]

  for constraint in constraints
    _field_id(constraint.field) == field_id || continue
    push!(matched, constraint)
  end

  return matched
end

# Find the Dirichlet constraints that apply on one boundary face for one already
# field-filtered constraint set. Multiple constraints are allowed on the same
# face only when they target disjoint component subsets.
@inline function _matches(face::FaceValues, boundary::BoundaryFace)
  return face.axis == boundary.axis && face.side == boundary.side
end

function _matching_dirichlets(constraints, face::FaceValues, component_total::Int)
  matched = Int[]
  covered = falses(component_total)

  for index in eachindex(constraints)
    constraint = constraints[index]
    _matches(face, constraint.boundary) || continue

    for component in constraint.components
      !covered[component] ||
        throw(ArgumentError("multiple Dirichlet constraints target overlapping components on the same boundary face"))
      covered[component] = true
    end

    push!(matched, index)
  end

  return matched
end

# Assemble the boundary projection system whose solution expresses the active
# trace dofs of one field in terms of the prescribed Dirichlet data. In
# mathematical terms, this builds the Gram system
#
#   ∫Γ (Σᵢ αᵢ ϕᵢ) ϕⱼ dΓ = ∫Γ g ϕⱼ dΓ
#
# on the active trace basis `{ϕᵢ}` of the selected field and boundary set.
function _boundary_projection_system(layout::FieldLayout{D,T}, slot::_FieldSlot{D,T},
                                     boundary_faces, constraints) where {D,T<:AbstractFloat}
  selected_face_indices = Int[]
  constraint_matches = falses(length(constraints))
  component_total = component_count(slot.field)

  for face_index in eachindex(boundary_faces)
    face = boundary_faces[face_index]
    matched = _matching_dirichlets(constraints, face, component_total)
    isempty(matched) && continue
    push!(selected_face_indices, face_index)

    for index in matched
      constraint_matches[index] = true
    end
  end

  all(constraint_matches) ||
    throw(ArgumentError("Dirichlet constraint does not match any boundary face for field $(field_name(slot.field))"))
  isempty(selected_face_indices) && return Int[], zeros(T, 0, 0), T[]
  selected_faces = boundary_faces[selected_face_indices]
  operators = [_BoundaryContribution(constraint.boundary,
                                     _DirichletProjection(slot.field, constraint.components,
                                                          constraint.data))
               for constraint in constraints]
  max_local_dofs = maximum(face.local_dof_count for face in selected_faces)
  boundary_lookup = _boundary_operator_lookup(dimension(layout), operators)
  boundary_batches = _compile_filtered_batches(selected_faces, _face_kernel_signature,
                                               face -> boundary_lookup[_boundary_lookup_slot(face.axis,
                                                                                             face.side)])
  slot_range = field_dof_range(layout, slot.field)
  involved = _boundary_projection_dofs(slot_range, selected_faces)
  isempty(involved) &&
    throw(ArgumentError("Dirichlet projection found no active trace dofs for field $(field_name(slot.field))"))
  slot_to_compact = zeros(Int, length(slot_range))

  for compact_index in eachindex(involved)
    slot_to_compact[involved[compact_index]] = compact_index
  end

  compact_matrix = zeros(T, length(involved), length(involved))
  compact_rhs = zeros(T, length(involved))
  local_matrix = zeros(T, max_local_dofs, max_local_dofs)
  local_rhs = zeros(T, max_local_dofs)
  dofs = [first(slot_range) + local_dof - 1 for local_dof in involved]
  _assemble_boundary_projection!(compact_matrix, compact_rhs, slot_range, slot_to_compact,
                                 local_matrix, local_rhs, boundary_batches, selected_faces,
                                 operators)
  return dofs, compact_matrix, compact_rhs
end

function _boundary_projection_dofs(slot_range::UnitRange{Int}, faces)
  active = falses(length(slot_range))

  for face in faces
    for local_dof in 1:face.local_dof_count
      for term_index in _local_term_range(face, local_dof)
        slot_index = _slot_index(slot_range, face.term_indices[term_index])
        slot_index == 0 || (active[slot_index] = true)
      end
    end
  end

  return findall(active)
end

function _assemble_boundary_projection!(compact_matrix::AbstractMatrix{T},
                                        compact_rhs::AbstractVector{T}, slot_range::UnitRange{Int},
                                        slot_to_compact::Vector{Int},
                                        local_matrix::AbstractMatrix{T},
                                        local_rhs::AbstractVector{T},
                                        batches::Vector{_FilteredKernelBatch}, faces,
                                        operators) where {T<:AbstractFloat}
  for batch in batches
    matrix_view = view(local_matrix, 1:batch.local_dof_count, 1:batch.local_dof_count)
    rhs_view = view(local_rhs, 1:batch.local_dof_count)

    for batch_face_index in eachindex(batch.item_indices)
      face = @inbounds faces[batch.item_indices[batch_face_index]]
      fill!(matrix_view, zero(T))
      fill!(rhs_view, zero(T))

      for operator_index in batch.operator_indices
        wrapped = @inbounds operators[operator_index]
        _face_projection_matrix!(matrix_view, wrapped.operator, face)
        _face_projection_rhs!(rhs_view, wrapped.operator, face)
      end

      _scatter_projection!(compact_matrix, compact_rhs, slot_range, slot_to_compact, face,
                           matrix_view, rhs_view)
    end
  end

  return nothing
end

@inline function _local_term_range(item::_AssemblyValues, local_dof::Int)
  return item.term_offsets[local_dof]:(item.term_offsets[local_dof+1]-1)
end

@inline function _slot_index(slot_range::UnitRange{Int}, global_dof::Int)
  first_dof = first(slot_range)
  last_dof = last(slot_range)
  return first_dof <= global_dof <= last_dof ? global_dof - first_dof + 1 : 0
end

function _scatter_projection!(matrix_data::AbstractMatrix{T}, rhs_data::AbstractVector{T},
                              slot_range::UnitRange{Int}, slot_to_compact::Vector{Int},
                              item::_AssemblyValues, local_matrix::AbstractMatrix{T},
                              local_rhs::AbstractVector{T}) where {T<:AbstractFloat}
  for local_row in 1:item.local_dof_count
    rhs_value = local_rhs[local_row]

    for row_term in _local_term_range(item, local_row)
      row_slot = _slot_index(slot_range, item.term_indices[row_term])
      row_slot == 0 && continue
      row = slot_to_compact[row_slot]
      row == 0 && continue
      row_coefficient = item.term_coefficients[row_term]
      iszero(row_coefficient) && continue
      iszero(rhs_value) || (rhs_data[row] += row_coefficient * rhs_value)

      for local_col in 1:item.local_dof_count
        coefficient = local_matrix[local_row, local_col]
        iszero(coefficient) && continue

        for col_term in _local_term_range(item, local_col)
          col_slot = _slot_index(slot_range, item.term_indices[col_term])
          col_slot == 0 && continue
          col = slot_to_compact[col_slot]
          col == 0 && continue
          col_coefficient = item.term_coefficients[col_term]
          iszero(col_coefficient) && continue
          matrix_data[row, col] += row_coefficient * coefficient * col_coefficient
        end
      end
    end
  end

  return nothing
end

# Interpret user-supplied Dirichlet data as either scalar or vector-valued
# boundary values for the selected component subset. Vector-valued data may
# either match the selected subset or the full field component count.
@inline function _selected_component_index(components::Tuple{Vararg{Int}}, component::Int)
  for index in eachindex(components)
    components[index] == component && return index
  end

  throw(ArgumentError("selected component $component is not part of this Dirichlet constraint"))
end

@noinline function _throw_dirichlet_data_conversion_error(value, ::Type{T}) where {T<:AbstractFloat}
  throw(ArgumentError("Dirichlet data entries must be convertible to $T; got $(typeof(value))"))
end

function _dirichlet_data_value(value, ::Type{T}) where {T<:AbstractFloat}
  try
    return T(value)
  catch
    _throw_dirichlet_data_conversion_error(value, T)
  end
end

function _dirichlet_component_value(data, x, component::Int, components::Tuple{Vararg{Int}},
                                    component_total::Int, ::Type{T}) where {T<:AbstractFloat}
  value = applicable(data, x) ? data(x) : data
  selected_total = length(components)
  selected_index = _selected_component_index(components, component)

  if value isa Tuple || value isa AbstractVector
    length(value) == selected_total && return _dirichlet_data_value(value[selected_index], T)
    length(value) == component_total && return _dirichlet_data_value(value[component], T)
    throw(ArgumentError("Dirichlet data must match the selected component count or the full field component count"))
  end

  selected_total == 1 ||
    throw(ArgumentError("Dirichlet data for multiple selected components must return a tuple or vector"))
  return _dirichlet_data_value(value, T)
end

# Mean-value constraint compilation as explicit linear equations.

# Compile mean-value constraints as explicit linear rows. Any contributions from
# fixed Dirichlet dofs are moved to the right-hand side, and a unique free pivot
# is chosen for each scalar constraint row.
function _compile_mean_constraints(layout::FieldLayout{D,T}, cells,
                                   dirichlet::_CompiledDirichlet{T},
                                   constraints) where {D,T<:AbstractFloat}
  isempty(constraints) && return _CompiledLinearConstraint{T}[]
  blocked = Set(dirichlet.fixed_dofs)
  union!(blocked, row.pivot for row in dirichlet.rows)
  fixed_values = Dict(dirichlet.fixed_dofs[index] => dirichlet.fixed_values[index]
                      for index in eachindex(dirichlet.fixed_dofs))
  domain_measure_value = _cell_integration_measure(cells, T)
  compiled = _CompiledLinearConstraint{T}[]
  used_pivots = Set{Int}()

  for constraint in constraints
    slot = layout.slots[_field_slot_index(layout, constraint.field)]
    row_vectors = _field_mean_rows(slot, cells)
    targets = _mean_targets(constraint.target, component_count(slot.field), T)

    for component in 1:component_count(slot.field)
      pairs = row_vectors[component]
      indices = Int[]
      coefficients = T[]
      rhs = targets[component] * domain_measure_value

      for pair in pairs
        if haskey(fixed_values, pair.first)
          rhs -= pair.second * fixed_values[pair.first]
        else
          push!(indices, pair.first)
          push!(coefficients, pair.second)
        end
      end

      isempty(indices) && throw(ArgumentError("mean-value constraint leaves no free dofs"))
      free_indices = [index for index in indices if !(index in blocked)]
      isempty(free_indices) && throw(ArgumentError("mean-value constraint leaves no free dofs"))
      pivot = minimum(free_indices)
      !(pivot in used_pivots) || throw(ArgumentError("mean-value pivots must be unique"))
      push!(used_pivots, pivot)
      push!(compiled, _CompiledLinearConstraint(pivot, indices, coefficients, rhs))
    end
  end

  return compiled
end

function _cell_integration_measure(cells, ::Type{T}) where {T<:AbstractFloat}
  measure = zero(T)

  for cell in cells
    for weight_value in cell.weights
      measure += weight_value
    end
  end

  return measure
end

# Auxiliary operator used to build the boundary L² projection system for
# Dirichlet data.
struct _DirichletProjection{F,C,G}
  field::F
  components::C
  data::G
end

# Boundary mass matrix on the active trace space of one field.
function _face_projection_matrix!(local_matrix, operator::_DirichletProjection, values::FaceValues)
  block_data = block(local_matrix, values, operator.field, operator.field)
  data = _field_values(values, operator.field)
  mode_count = data.local_mode_count

  for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for component in operator.components
      offset = _field_component_offset(data, component)

      for row_mode in 1:mode_count
        shape_row = data.values[row_mode, point_index]

        for col_mode in 1:mode_count
          block_data[offset+row_mode, offset+col_mode] += shape_row *
                                                          data.values[col_mode, point_index] *
                                                          weighted
        end
      end
    end
  end

  return nothing
end

# Right-hand side of the boundary projection system against the prescribed
# Dirichlet data.
function _face_projection_rhs!(local_rhs, operator::_DirichletProjection{F,C,G},
                               values::FaceValues) where {F,C,G}
  block_data = block(local_rhs, values, operator.field)
  data = _field_values(values, operator.field)
  mode_count = data.local_mode_count
  value_type = eltype(block_data)

  for point_index in 1:point_count(values)
    point_data = point(values, point_index)
    weighted = weight(values, point_index)

    for component in operator.components
      target = _dirichlet_component_value(operator.data, point_data, component, operator.components,
                                          _field_component_count(data), value_type)
      offset = _field_component_offset(data, component)

      for mode_index in 1:mode_count
        block_data[offset+mode_index] += data.values[mode_index, point_index] * weighted * target
      end
    end
  end

  return nothing
end

# Numerical rank tolerance for the boundary projection Gram matrix. The cutoff
# is relative to the actual matrix scale. Physical boundary measures may be
# very small, and a fixed absolute floor would turn valid trace constraints into
# apparent null modes merely because the domain has small units.
function _boundary_projection_rank_tolerance(::Type{T}, scale::T,
                                             dimension::Integer) where {T<:AbstractFloat}
  relative_scale = abs(scale)
  iszero(relative_scale) && return zero(T)
  return 1000 * eps(T) * relative_scale * max(Int(dimension), 1)
end

# Convert the projected boundary system to either explicitly fixed dofs or affine
# relations between trace dofs. The goal is to move from the abstract boundary
# coefficient vector `α` to row-wise elimination formulas that later evaluation
# can apply directly.
function _dirichlet_relations(dofs::Vector{Int}, matrix_data::AbstractMatrix{T},
                              rhs_data::AbstractVector{T},
                              field_name::Symbol) where {T<:AbstractFloat}
  constraint_matrix, targets, tolerance = _dirichlet_constraint_system(matrix_data, rhs_data,
                                                                       field_name)
  constraint_count = size(constraint_matrix, 1)
  constraint_count == 0 && return Int[], T[], _CompiledAffineRelation{T}[]

  factorization = qr(constraint_matrix, ColumnNorm())
  pivot_columns = factorization.p[1:constraint_count]
  free_columns = Int[]
  pivot_mask = falses(length(dofs))

  for column in pivot_columns
    pivot_mask[column] = true
  end

  for column in eachindex(dofs)
    pivot_mask[column] || push!(free_columns, column)
  end

  pivot_matrix = Matrix(constraint_matrix[:, pivot_columns])
  shift_values = pivot_matrix \ targets
  fixed_dofs = Int[]
  fixed_values = T[]
  rows = _CompiledAffineRelation{T}[]

  if isempty(free_columns)
    append!(fixed_dofs, dofs[pivot_columns])
    append!(fixed_values, shift_values)
    return fixed_dofs, fixed_values, rows
  end

  free_indices = dofs[free_columns]
  coupling = -(pivot_matrix \ Matrix(constraint_matrix[:, free_columns]))

  for row_index in 1:constraint_count
    indices = Int[]
    coefficients = T[]

    for free_index in eachindex(free_indices)
      coefficient = coupling[row_index, free_index]
      abs(coefficient) > tolerance || continue
      push!(indices, free_indices[free_index])
      push!(coefficients, coefficient)
    end

    pivot = dofs[pivot_columns[row_index]]
    value = shift_values[row_index]

    if isempty(indices)
      push!(fixed_dofs, pivot)
      push!(fixed_values, value)
    else
      push!(rows, _CompiledAffineRelation(pivot, indices, coefficients, value))
    end
  end

  return fixed_dofs, fixed_values, rows
end

# Spectral analysis of the boundary projection Gram system. Positive eigenmodes
# define independent trace constraints; near-null modes are checked for data
# consistency and then discarded. This is the robust way to handle trace bases
# that contain algebraic dependencies or vanish on the selected boundary subset:
# only the positive-eigenvalue subspace represents genuinely controllable trace
# directions.
function _dirichlet_constraint_system(matrix_data::AbstractMatrix{T}, rhs_data::AbstractVector{T},
                                      field_name::Symbol) where {T<:AbstractFloat}
  decomposition = eigen(Symmetric(matrix_data))
  order = sortperm(decomposition.values; rev=true)
  eigenvalues = decomposition.values[order]
  basis = decomposition.vectors[:, order]
  scale = maximum(abs, eigenvalues; init=zero(T))
  tolerance = _boundary_projection_rank_tolerance(T, scale, size(matrix_data, 1))
  constraint_count = count(>(tolerance), eigenvalues)
  transformed_rhs = transpose(basis) * rhs_data
  rhs_tolerance = tolerance * max(norm(rhs_data), one(T))

  for index in (constraint_count+1):length(transformed_rhs)
    abs(transformed_rhs[index]) <= rhs_tolerance ||
      throw(ArgumentError("Dirichlet data on field $field_name is inconsistent on the selected boundary faces"))
  end

  constraint_count == 0 && return zeros(T, 0, size(matrix_data, 1)), T[], tolerance
  constraint_matrix = transpose(@view basis[:, 1:constraint_count])
  targets = transformed_rhs[1:constraint_count] ./ eigenvalues[1:constraint_count]
  return constraint_matrix, targets, tolerance
end

# Assemble the coefficient rows representing the integral mean of one field over
# the whole domain.
function _field_mean_rows(slot::_FieldSlot{D,T}, cells) where {D,T<:AbstractFloat}
  rows = [Dict{Int,T}() for _ in 1:component_count(slot.field)]

  for cell in cells
    data = _field_values(cell, slot.field)

    for component in 1:_field_component_count(data)
      row = rows[component]
      local_offset = _field_component_offset(data, component)

      for mode_index in 1:data.local_mode_count
        local_dof = local_offset + mode_index

        for point_index in 1:point_count(cell)
          weighted = data.values[mode_index, point_index] * weight(cell, point_index)

          for term_index in _field_term_range(data, local_dof)
            global_dof = data.term_indices[term_index]
            row[global_dof] = get(row, global_dof, zero(T)) +
                              data.term_coefficients[term_index] * weighted
          end
        end
      end
    end
  end

  return [sort!(collect(row); by=first) for row in rows]
end

@inline function _field_term_range(data::_FieldValues, local_dof::Int)
  return data.term_offsets[local_dof]:(data.term_offsets[local_dof+1]-1)
end

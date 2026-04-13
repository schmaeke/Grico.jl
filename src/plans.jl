# This file contains the setup side of the assembly layer:
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
# assembly and nonlinear evaluations operate. It records not only operators and
# integration data, but also the compiled form of strong Dirichlet constraints
# and mean-value constraints.
#
# In that sense, `problem.jl` describes what should be assembled, whereas this
# file determines how those declarations are translated into algebraic rows and
# eliminated degrees of freedom.
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
# functionals assembled against the already compiled basis. Those rows survive
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
# or, equivalently for residual/tangent assembly,
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

"""
    AssemblyPlan

Compiled assembly data for a problem on a fixed field layout and mesh state.

An `AssemblyPlan` stores the validated [`FieldLayout`](@ref), the compiled local
integration data, the operator collections, and the compiled Dirichlet and
mean-value constraints. It is the reusable object that separates the expensive
setup phase from repeated assembly or nonlinear evaluations.

For affine problems, `assemble(plan)` turns the plan into an [`AffineSystem`](@ref).
For nonlinear problems, the same plan supplies the local data needed by
[`residual`](@ref) and [`tangent`](@ref).

The plan is tied to the current field layout, active-leaf set, and current
constraint data. If the mesh, space, or problem definition changes, a new plan
must be compiled.
"""
struct AssemblyPlan{D,T<:AbstractFloat}
  layout::FieldLayout{D,T}
  cell_operators::Tuple
  boundary_operators::Tuple
  interface_operators::Tuple
  surface_operators::Tuple
  integration
  dirichlet::_CompiledDirichlet{T}
  mean_constraints::Vector{_CompiledLinearConstraint{T}}
  constraint_masks::_ConstraintMasks{T}
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
selected field. This is what allows the later assembly layer to eliminate
strong boundary data in a basis-agnostic way, even on hanging interfaces and
high-order trace spaces.

The returned plan is immutable in the sense that later assembly reads from it
but does not rewrite its structural data. Recompilation is therefore the
boundary between the editable setup phase and the repeated runtime phase.
"""
function compile(problem::_AbstractProblem)
  return _with_internal_blas_threads() do
    _validate_problem_data(problem)
    data = _problem_data(problem)
    _compile_problem(data.fields, Tuple(data.cell_operators), Tuple(data.boundary_operators),
                     Tuple(data.interface_operators), Tuple(data.surface_operators),
                     data.cell_quadratures, data.embedded_surfaces, data.dirichlet_constraints,
                     data.mean_constraints)
  end
end

# Main compilation pipeline from problem description to immutable plan.

# After geometric compilation, surface selectors can be validated against the
# actual embedded-surface quadratures that will participate in assembly. This
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
# is structurally the same; later assembly/evaluation decides which operator
# callbacks are used.
function _compile_problem(fields, cell_operators, boundary_operators, interface_operators,
                          surface_operators, cell_quadratures, embedded_surfaces,
                          dirichlet_constraints, mean_constraints)
  layout = _field_layout(fields)
  integration = _compile_integration(layout, cell_quadratures, embedded_surfaces;
                                     include_interfaces=(!isempty(interface_operators)))
  _validate_surface_operators(surface_operators, integration.embedded_surfaces)
  compiled_dirichlet = _compile_dirichlet(layout, integration.boundary_faces, dirichlet_constraints)
  compiled_mean_constraints = _compile_mean_constraints(layout, integration.cells,
                                                        compiled_dirichlet, mean_constraints)
  constraint_masks = _constraint_masks(dof_count(layout), compiled_dirichlet,
                                       compiled_mean_constraints)
  return AssemblyPlan{dimension(layout),eltype(origin(field_space(layout.slots[1].field)))}(layout,
                                                                                            cell_operators,
                                                                                            boundary_operators,
                                                                                            interface_operators,
                                                                                            surface_operators,
                                                                                            integration,
                                                                                            compiled_dirichlet,
                                                                                            compiled_mean_constraints,
                                                                                            constraint_masks)
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
  selected_faces = FaceValues[]
  constraint_matches = falses(length(constraints))
  component_total = component_count(slot.field)

  for face in boundary_faces
    matched = _matching_dirichlets(constraints, face, component_total)
    isempty(matched) && continue
    push!(selected_faces, face)

    for index in matched
      constraint_matches[index] = true
    end
  end

  all(constraint_matches) ||
    throw(ArgumentError("Dirichlet constraint does not match any boundary face for field $(field_name(slot.field))"))
  isempty(selected_faces) && return Int[], spzeros(T, 0, 0), T[]
  operators = [_BoundaryContribution(constraint.boundary,
                                     _DirichletProjection(slot.field, constraint.components,
                                                          constraint.data))
               for constraint in constraints]
  ndofs = dof_count(layout)
  max_local_dofs = maximum(face.local_dof_count for face in selected_faces)
  scratch = [_ThreadScratch(T, ndofs, max_local_dofs)
             for _ in 1:min(Threads.nthreads(), length(selected_faces))]
  fixed = falses(ndofs)
  fixed_values = zeros(T, ndofs)
  pivot_rows = falses(ndofs)
  _assemble_boundary_pass!(scratch, selected_faces, operators, fixed, fixed_values, pivot_rows)
  rows = Int[]
  cols = Int[]
  values = T[]
  rhs_data = zeros(T, ndofs)

  for cache in scratch
    append!(rows, cache.rows)
    append!(cols, cache.cols)
    append!(values, cache.values)
    rhs_data .+= cache.global_rhs
  end

  slot_range = field_dof_range(layout, slot.field)
  full_matrix = sparse(rows, cols, values, ndofs, ndofs)
  slot_matrix = full_matrix[slot_range, slot_range]
  slot_rhs = rhs_data[slot_range]
  involved = _active_projection_dofs(slot_matrix, slot_rhs)
  isempty(involved) &&
    throw(ArgumentError("Dirichlet projection found no active trace dofs for field $(field_name(slot.field))"))
  dofs = [first(slot_range) + local_dof - 1 for local_dof in involved]
  return dofs, slot_matrix[involved, involved], slot_rhs[involved]
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

function _dirichlet_component_value(data, x, component::Int, components::Tuple{Vararg{Int}},
                                    component_total::Int,
                                    ::Type{T}) where {T<:AbstractFloat}
  value = data isa Function ? data(x) : data
  selected_total = length(components)
  selected_index = _selected_component_index(components, component)

  if value isa Tuple || value isa AbstractVector
    length(value) == selected_total && return T(value[selected_index])
    length(value) == component_total && return T(value[component])
    throw(ArgumentError("Dirichlet data must match the selected component count or the full field component count"))
  end

  selected_total == 1 || throw(ArgumentError("Dirichlet data for multiple selected components must return a tuple or vector"))
  return T(value)
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
  domain_measure_value = sum(cell_volume(layout.slots[1].space.domain, cell.leaf) for cell in cells)
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

# Auxiliary operator used to assemble the boundary L² projection system for
# Dirichlet data.
struct _DirichletProjection{F,C,G}
  field::F
  components::C
  data::G
end

# Boundary mass matrix on the active trace space of one field.
function face_matrix!(local_matrix, operator::_DirichletProjection, values::FaceValues)
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
function face_rhs!(local_rhs, operator::_DirichletProjection{F,C,G},
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
                                          data.component_count, value_type)
      offset = _field_component_offset(data, component)

      for mode_index in 1:mode_count
        block_data[offset+mode_index] += data.values[mode_index, point_index] * weighted * target
      end
    end
  end

  return nothing
end

# Identify the trace dofs that are actually active in the projection system.
function _active_projection_dofs(matrix_data::SparseMatrixCSC{T,Int},
                                 rhs_data::AbstractVector{T}) where {T<:AbstractFloat}
  active = falses(size(matrix_data, 1))
  rows, cols, values = findnz(matrix_data)
  tolerance = 1000 * eps(T)

  for index in eachindex(values)
    abs(values[index]) > tolerance || continue
    active[rows[index]] = true
    active[cols[index]] = true
  end

  for index in eachindex(rhs_data)
    abs(rhs_data[index]) > tolerance && (active[index] = true)
  end

  return findall(active)
end

# Numerical rank tolerance for the boundary projection Gram matrix.
function _boundary_projection_rank_tolerance(::Type{T}, scale::T) where {T<:AbstractFloat}
  return sqrt(eps(T)) * max(scale, one(T))
end

# Convert the projected boundary system to either explicitly fixed dofs or affine
# relations between trace dofs. The goal is to move from the abstract boundary
# coefficient vector `α` to row-wise elimination formulas that later assembly
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
  tolerance = _boundary_projection_rank_tolerance(T, scale)
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

    for component in 1:data.component_count
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

# Interpret a scalar or vector mean target with the correct component count.
function _mean_targets(target, component_total::Int, ::Type{T}) where {T<:AbstractFloat}
  if component_total == 1
    return (T(target),)
  end

  target isa Tuple ||
    target isa AbstractVector ||
    throw(ArgumentError("vector-valued mean target must be a tuple or vector"))
  length(target) == component_total ||
    throw(ArgumentError("mean target must match the field component count"))
  return ntuple(index -> T(target[index]), component_total)
end

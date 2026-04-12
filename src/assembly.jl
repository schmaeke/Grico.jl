# This file owns the runtime side of the assembly layer:
# 1. reduced affine-system assembly from an `AssemblyPlan`,
# 2. nonlinear residual and tangent evaluation on the full field layout, and
# 3. the structural metadata needed by the default solve path.
#
# If `plans.jl` is the chapter that freezes the problem description into an
# immutable algebraic blueprint, this file is the chapter that executes that
# blueprint.
#
# There are two distinct runtime views of the same compiled problem.
#
# First, affine assembly produces a reduced linear system. Strong Dirichlet
# conditions are eliminated, optional static condensation removes suitable
# cell-interior modes, and mean-value constraints are inserted as explicit rows.
# The output is an `AffineSystem` together with the data needed to reconstruct a
# full state from the reduced solve vector.
#
# Second, nonlinear residual and tangent evaluation operate on the full field
# layout rather than on the reduced affine system. Here the same local
# integration data and operator callbacks are reused, but the compiled
# constraints reappear as explicit residual equations and tangent rows.
#
# The affine path has one especially important additional idea: whenever a cell
# local basis splits into trace-coupled dofs and purely interior dofs, the
# interior block may be eliminated by a local Schur complement. That is the
# static-condensation step used below. It reduces the global solve size without
# changing the represented discrete solution, because the eliminated cell-
# interior amplitudes are reconstructed afterwards from the retained dofs.
#
# Most of the file is therefore about moving between three levels of algebraic
# description:
# - local cell/face/interface/surface contributions,
# - reduced solve-space data after eliminations and condensation,
# - and full-layout coefficient vectors used for states and nonlinear
#   evaluations.
#
# The file is organized in that order: reduced-system metadata first, then the
# public `AffineSystem` API, then affine assembly, then residual/tangent
# evaluation, then the shared scattering kernels, and finally the reconstruction
# and solve-topology helpers.

# Reduced-system metadata and reconstruction storage.

# Sparse storage of the reduced affine reconstruction rows used after Dirichlet
# elimination and static condensation. For each eliminated global dof, the
# interval `row_offsets[dof]:(row_offsets[dof+1]-1)` stores its coefficients in
# the reduced basis.
struct _ReducedAffineRows{T<:AbstractFloat}
  eliminated::BitVector
  shifts::Vector{T}
  row_offsets::Vector{Int}
  indices::Vector{Int}
  coefficients::Vector{T}
end

@inline function _affine_row_pointer_range(rows::_ReducedAffineRows, dof::Int)
  return rows.row_offsets[dof]:(rows.row_offsets[dof+1]-1)
end

function _affine_row_pairs(rows::_ReducedAffineRows{T}, dof::Int) where {T<:AbstractFloat}
  return Pair{Int,T}[rows.indices[pointer] => rows.coefficients[pointer]
                     for pointer in _affine_row_pointer_range(rows, dof)]
end

@inline function _accumulate_reduced_row!(row::Dict{Int,T}, reduced_dof::Int,
                                          coefficient::T) where {T<:AbstractFloat}
  row[reduced_dof] = get(row, reduced_dof, zero(T)) + coefficient
  return nothing
end

function _sorted_reduced_row(row::Dict{Int,T}) where {T<:AbstractFloat}
  indices = sort!(collect(keys(row)))
  return Pair{Int,T}[index => row[index] for index in indices]
end

function _packed_reduced_affine_rows(::Type{T}, eliminated::BitVector, shifts::Vector{T},
                                     rows::Dict{Int,Vector{Pair{Int,T}}}) where {T<:AbstractFloat}
  ndofs = length(eliminated)
  row_offsets = ones(Int, ndofs + 1)

  for dof in 1:ndofs
    pairs = get(rows, dof, nothing)
    row_offsets[dof+1] = row_offsets[dof] + (pairs === nothing ? 0 : length(pairs))
  end

  indices = Vector{Int}(undef, row_offsets[end] - 1)
  coefficients = Vector{T}(undef, row_offsets[end] - 1)

  for dof in 1:ndofs
    pairs = get(rows, dof, nothing)
    pairs === nothing && continue
    pointer = row_offsets[dof]

    for pair in pairs
      indices[pointer] = pair.first
      coefficients[pointer] = pair.second
      pointer += 1
    end
  end

  return _ReducedAffineRows(eliminated, shifts, row_offsets, indices, coefficients)
end

# Per-cell static-condensation plan splitting local dofs into kept trace-coupled
# dofs and eliminated interior dofs.
struct _StaticCondensationPlan
  kept_local_dofs::Vector{Int}
  eliminated_local_dofs::Vector{Int}
  eliminated_global_dofs::Vector{Int}
end

# Topological data for the default iterative solve path: local Schwarz patches
# and a geometric coarse prolongation.
struct _PatchSolveTopology{T<:AbstractFloat}
  leaf_patches::Vector{Vector{Int}}
  coarse_prolongation::SparseMatrixCSC{T,Int}
end

"""
    AffineSystem

Reduced affine linear system together with the data needed to reconstruct the
full state.

`AffineSystem` stores the reduced sparse matrix and right-hand side obtained
after applying Dirichlet elimination, optional static condensation, and
mean-value constraints. It also stores the reconstruction map and constant shift
needed to recover the full coefficient vector from the reduced solution.

The object additionally caches structural data for the default solve path,
including fill-reducing orderings, patch topology for preconditioners built on
cell-local supports, and lazily built preconditioner instances keyed by their
configuration.

Algebraically, the full coefficient vector is reconstructed as

  u_full = shift + reconstruction * u_reduced.

The reduced matrix and right-hand side therefore act only on the unconstrained
solve dofs retained after Dirichlet elimination, optional static condensation,
and explicit constraint-row insertion.
"""
struct AffineSystem{T<:AbstractFloat,L<:FieldLayout}
  layout::L
  matrix::SparseMatrixCSC{T,Int}
  rhs::Vector{T}
  solve_dofs::Vector{Int}
  solve_topology::_PatchSolveTopology{T}
  reconstruction::SparseMatrixCSC{T,Int}
  shift::Vector{T}
  ordering::Vector{Int}
  inverse_ordering::Vector{Int}
  symmetric::Bool
  preconditioner_cache::Dict{Any,Any}
end

# Per-thread scratch storage used during assembly and nonlinear evaluations.
# Reusing these buffers avoids repeated allocation of local dense matrices,
# vectors, and sparse triplet work arrays for every cell/face/interface item.
mutable struct _ThreadScratch{T<:AbstractFloat}
  matrix::Matrix{T}
  factor_matrix::Matrix{T}
  solve_matrix::Matrix{T}
  rhs::Vector{T}
  work_rhs::Vector{T}
  rows::Vector{Int}
  cols::Vector{Int}
  values::Vector{T}
  shift_rows::Vector{Int}
  shift_values::Vector{T}
  reconstruction_rows::Vector{Int}
  reconstruction_cols::Vector{Int}
  reconstruction_values::Vector{T}
  global_rhs::Vector{T}
end

# Allocate one scratch bundle sized for the largest local integration item.
function _ThreadScratch(::Type{T}, ndofs::Int, local_dof_count::Int) where {T<:AbstractFloat}
  return _ThreadScratch(Matrix{T}(undef, local_dof_count, local_dof_count),
                        Matrix{T}(undef, local_dof_count, local_dof_count),
                        Matrix{T}(undef, local_dof_count, local_dof_count),
                        Vector{T}(undef, local_dof_count), Vector{T}(undef, local_dof_count), Int[],
                        Int[], T[], Int[], T[], Int[], Int[], T[], zeros(T, ndofs))
end

# Public accessors and constructors on assembled systems and plans.

field_layout(plan::AssemblyPlan) = plan.layout
field_layout(system::AffineSystem) = system.layout
dof_count(plan::AssemblyPlan) = dof_count(plan.layout)

"""
    matrix(system)

Return the reduced sparse matrix stored in `system`.

The matrix acts on the reduced solve vector, not on the full coefficient vector
of the underlying field layout.
"""
matrix(system::AffineSystem) = system.matrix

"""
    rhs(system)

Return the reduced right-hand side stored in `system`.

The returned vector is ordered consistently with [`matrix`](@ref) and with the
reduced solve dofs of the system.
"""
rhs(system::AffineSystem) = system.rhs

# `State(system, coefficients)` accepts either a full-layout coefficient vector
# or a reduced solve vector. In the latter case it reconstructs the full state
# through the stored affine map.
State(plan::AssemblyPlan) = State(plan.layout)
function State(system::AffineSystem{T}, coefficients::AbstractVector{T}) where {T<:AbstractFloat}
  length(coefficients) == dof_count(system.layout) && return State(system.layout, coefficients)
  length(coefficients) == length(system.solve_dofs) ||
    throw(ArgumentError("coefficient vector length must match either the reduced system or full layout"))
  return State(system.layout, _expand_system_values(system, coefficients))
end
function State(plan::AssemblyPlan, coefficients::AbstractVector{T}) where {T<:AbstractFloat}
  State(plan.layout, coefficients)
end

# Affine assembly from compiled plans to reduced linear systems.

"""
    assemble(problem)
    assemble(plan)

Assemble an [`AffineSystem`](@ref) from a problem or from a previously compiled
plan.

Assembly produces the reduced linear system obtained after applying strong
Dirichlet constraints, optional static condensation of interior cell dofs, and
mean-value constraints. The returned system retains the reconstruction data
needed to recover the full state after solving.

When static condensation is active, the returned matrix is the global system
for the retained dofs after local Schur-complement elimination of eligible
cell-interior modes. The eliminated amplitudes are not lost; they are stored in
the reconstruction map of the returned [`AffineSystem`](@ref).
"""
assemble(problem::AffineProblem) = assemble(compile(problem))

function assemble(plan::AssemblyPlan{D,T}) where {D,T<:AbstractFloat}
  return _with_internal_blas_threads() do
    ndofs = dof_count(plan)
    masks = plan.constraint_masks
    fixed = masks.fixed
    fixed_values = masks.fixed_values
    eliminated = masks.eliminated
    constraint_rows = masks.constraint_rows

    # Decide which cell-local interior dofs can be statically condensed. This is
    # only done when no interface operators are present, because interface terms
    # may couple the would-be eliminated trace data in a way that the purely
    # cell-local condensation formula would miss.
    condensed, condensation = _condensation_data(plan, fixed, masks.blocked_rows)
    reduced_index, solve_dofs = _reduced_dof_index(fixed, condensed, eliminated)
    dirichlet_affine = _reduce_dirichlet(plan.dirichlet, reduced_index, fixed, fixed_values)
    solve_topology = _solve_topology(plan, condensation, reduced_index, solve_dofs)
    shift = copy(dirichlet_affine.shifts)

    scratch = _scratch_buffers(T, length(solve_dofs), plan.integration)
    _assemble_leaf_affine_pass!(scratch, plan, condensation, reduced_index, fixed, fixed_values,
                                eliminated, constraint_rows, dirichlet_affine)
    _assemble_reduced_pass!(scratch, plan.integration.interfaces, plan.interface_operators,
                            reduced_index, fixed, fixed_values, eliminated, constraint_rows,
                            dirichlet_affine, interface_matrix!, interface_rhs!)

    # Merge per-thread sparse triplets and right-hand-side contributions, then
    # append the explicitly retained mean-value constraint rows in reduced form.
    rows = Int[]
    cols = Int[]
    values = T[]
    reduced_rhs = zeros(T, length(solve_dofs))
    condensed_rows = _reconstruction_rows(dirichlet_affine)

    for cache in scratch
      append!(rows, cache.rows)
      append!(cols, cache.cols)
      append!(values, cache.values)
      reduced_rhs .+= cache.global_rhs

      for index in eachindex(cache.shift_rows)
        shift[cache.shift_rows[index]] += cache.shift_values[index]
      end

      for index in eachindex(cache.reconstruction_rows)
        row = cache.reconstruction_rows[index]
        pair = cache.reconstruction_cols[index] => cache.reconstruction_values[index]
        push!(get!(condensed_rows, row, Pair{Int,T}[]), pair)
      end
    end

    for constraint in
        _reduce_mean_constraints(plan.mean_constraints, reduced_index, shift, condensed_rows)
      for index in eachindex(constraint.indices)
        push!(rows, constraint.pivot)
        push!(cols, constraint.indices[index])
        push!(values, constraint.coefficients[index])
      end

      reduced_rhs[constraint.pivot] = constraint.rhs
    end

    reduced_matrix = sparse(rows, cols, values, length(solve_dofs), length(solve_dofs))
    reconstruction = _reconstruction_matrix(T, ndofs, solve_dofs, condensed_rows)
    ordering, inverse_ordering = _solve_ordering(reduced_matrix)
    return AffineSystem(plan.layout, reduced_matrix, reduced_rhs, solve_dofs, solve_topology,
                        reconstruction, shift, ordering, inverse_ordering,
                        _is_symmetric_matrix(reduced_matrix), Dict{Any,Any}())
  end
end

# Symmetric reverse Cuthill-McKee ordering used to reduce fill in the default
# direct solve path.
function _solve_ordering(matrix_data::SparseMatrixCSC)
  ndofs = size(matrix_data, 1)
  ordering = ndofs == 0 ? Int[] : symrcm(matrix_data; sortbydeg=false)
  inverse_ordering = zeros(Int, ndofs)

  for ordered_index in eachindex(ordering)
    inverse_ordering[ordering[ordered_index]] = ordered_index
  end

  return ordering, inverse_ordering
end

# Numerical symmetry test used to decide whether the Cholesky/CG-based solve
# path is admissible.
function _is_symmetric_matrix(matrix_data::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
  size(matrix_data, 1) == size(matrix_data, 2) || return false
  nnz(matrix_data) == 0 && return true
  delta = matrix_data - transpose(matrix_data)
  norm_matrix = opnorm(matrix_data, Inf)
  tolerance = max(norm_matrix, one(T)) * size(matrix_data, 1) * 1000 * eps(T)
  return opnorm(delta, Inf) <= tolerance
end

# Nonlinear residual and tangent evaluation on the full field layout.

"""
    residual(plan, state)
    residual!(result, plan, state)

Assemble the nonlinear residual associated with `plan` at the given `state`.

The residual is formed on the full field layout, not on the reduced affine
solve space. In addition to the operator contributions, it includes explicit
equations enforcing compiled Dirichlet and mean-value constraints.

The mutating form writes into `result`; the allocating form returns a newly
allocated vector.
"""
function residual(plan::AssemblyPlan{D,T}, state::State{T}) where {D,T<:AbstractFloat}
  result = zeros(T, dof_count(plan))
  residual!(result, plan, state)
  return result
end

"""
    residual!(result, plan, state)

Overwrite `result` with the nonlinear residual associated with `plan` at
`state`.
"""
function residual!(result::AbstractVector{T}, plan::AssemblyPlan{D,T},
                   state::State{T}) where {D,T<:AbstractFloat}
  _with_internal_blas_threads() do
    _check_state(plan, state)
    _require_length(result, dof_count(plan), "result")
    masks = plan.constraint_masks
    fixed = masks.fixed
    skipped_rows = masks.blocked_rows
    scratch = _scratch_buffers(T, dof_count(plan), plan.integration)
    _reset_scratch!(scratch)
    _residual_pass!(scratch, plan.integration.cells, plan.cell_operators, state, fixed,
                    skipped_rows, cell_residual!)
    _boundary_residual_pass!(scratch, plan.integration.boundary_faces, plan.boundary_operators,
                             state, fixed, skipped_rows)
    _residual_pass!(scratch, plan.integration.interfaces, plan.interface_operators, state, fixed,
                    skipped_rows, interface_residual!)
    _surface_residual_pass!(scratch, plan.integration.embedded_surfaces, plan.surface_operators,
                            state, fixed, skipped_rows)
    fill!(result, zero(T))

    # Sum all unconstrained operator contributions from the thread-local buffers.
    for cache in scratch
      result .+= cache.global_rhs
    end

    # Dirichlet and mean-value constraints enter the nonlinear residual as
    # explicit algebraic equations on their designated pivot rows.
    for index in eachindex(plan.dirichlet.fixed_dofs)
      dof = plan.dirichlet.fixed_dofs[index]
      result[dof] = state.coefficients[dof] - plan.dirichlet.fixed_values[index]
    end

    for row in plan.dirichlet.rows
      value = state.coefficients[row.pivot] - row.rhs

      for index in eachindex(row.indices)
        value -= row.coefficients[index] * state.coefficients[row.indices[index]]
      end

      result[row.pivot] = value
    end

    for constraint in plan.mean_constraints
      result[constraint.pivot] = -constraint.rhs

      for index in eachindex(constraint.indices)
        result[constraint.pivot] += constraint.coefficients[index] *
                                    state.coefficients[constraint.indices[index]]
      end
    end

    return result
  end
end

"""
    tangent(plan, state)
    tangent!(matrix, plan, state)

Assemble the tangent matrix of the nonlinear residual associated with `plan` at
`state`.

As with [`residual`](@ref), the tangent is assembled on the full field layout
and includes explicit rows for Dirichlet and mean-value constraints. The
mutating form overwrites an existing sparse matrix with the newly assembled
tangent.
"""
function tangent(plan::AssemblyPlan{D,T}, state::State{T}) where {D,T<:AbstractFloat}
  return _with_internal_blas_threads() do
    _check_state(plan, state)
    masks = plan.constraint_masks
    fixed = masks.fixed
    skipped_rows = masks.blocked_rows
    scratch = _scratch_buffers(T, dof_count(plan), plan.integration)
    _reset_scratch!(scratch)
    _tangent_pass!(scratch, plan.integration.cells, plan.cell_operators, state, fixed, skipped_rows,
                   cell_tangent!)
    _boundary_tangent_pass!(scratch, plan.integration.boundary_faces, plan.boundary_operators,
                            state, fixed, skipped_rows)
    _tangent_pass!(scratch, plan.integration.interfaces, plan.interface_operators, state, fixed,
                   skipped_rows, interface_tangent!)
    _surface_tangent_pass!(scratch, plan.integration.embedded_surfaces, plan.surface_operators,
                           state, fixed, skipped_rows)
    rows = Int[]
    cols = Int[]
    values = T[]

    for cache in scratch
      append!(rows, cache.rows)
      append!(cols, cache.cols)
      append!(values, cache.values)
    end

    # Constraint rows are appended explicitly because they do not arise from the
    # operator callbacks. Fixed Dirichlet rows contribute identity equations,
    # affine Dirichlet rows contribute `u_pivot - Σ cᵢ u_i - rhs = 0`, and
    # mean-value rows contribute the dense average constraints.
    for index in eachindex(plan.dirichlet.fixed_dofs)
      dof = plan.dirichlet.fixed_dofs[index]
      push!(rows, dof)
      push!(cols, dof)
      push!(values, one(T))
    end

    for row in plan.dirichlet.rows
      push!(rows, row.pivot)
      push!(cols, row.pivot)
      push!(values, one(T))

      for index in eachindex(row.indices)
        push!(rows, row.pivot)
        push!(cols, row.indices[index])
        push!(values, -row.coefficients[index])
      end
    end

    for constraint in plan.mean_constraints
      for index in eachindex(constraint.indices)
        push!(rows, constraint.pivot)
        push!(cols, constraint.indices[index])
        push!(values, constraint.coefficients[index])
      end
    end

    return sparse(rows, cols, values, dof_count(plan), dof_count(plan))
  end
end

"""
    tangent!(matrix, plan, state)

Overwrite `matrix` with the tangent matrix associated with `plan` at `state`.
"""
function tangent!(matrix_data::SparseMatrixCSC{T,Int}, plan::AssemblyPlan{D,T},
                  state::State{T}) where {D,T<:AbstractFloat}
  assembled = tangent(plan, state)
  size(matrix_data) == size(assembled) ||
    throw(ArgumentError("target sparse matrix shape must match the assembled tangent"))
  resize!(matrix_data.colptr, length(assembled.colptr))
  resize!(matrix_data.rowval, length(assembled.rowval))
  resize!(matrix_data.nzval, length(assembled.nzval))
  copyto!(matrix_data.colptr, assembled.colptr)
  copyto!(matrix_data.rowval, assembled.rowval)
  copyto!(matrix_data.nzval, assembled.nzval)
  return matrix_data
end

# Shared local traversal routines for affine, residual, and tangent passes.

# Main affine assembly pass over cells together with any boundary-face and
# embedded-surface terms that are local to the same leaf. The key point is that
# all contributions living on one leaf are accumulated before static
# condensation is attempted, so the local Schur complement sees the complete
# cell-local algebra available at that stage. Only afterwards is the retained
# local system scattered to the reduced global system.
function _assemble_leaf_affine_pass!(scratch, plan::AssemblyPlan{D,T}, condensation, reduced_index,
                                     fixed, fixed_values, eliminated, constraint_rows,
                                     dirichlet_affine::_ReducedAffineRows{T}) where {D,
                                                                                     T<:AbstractFloat}
  cells = plan.integration.cells
  leaves = plan.integration.leaves
  boundary_faces = plan.integration.boundary_faces
  embedded_surfaces = plan.integration.embedded_surfaces
  tolerance = 1000 * eps(T)

  _run_chunks!(scratch, length(cells)) do cache, first_leaf, last_leaf
    for leaf_index in first_leaf:last_leaf
      cell = @inbounds cells[leaf_index]
      leaf = @inbounds leaves[leaf_index]
      condensation_plan = @inbounds condensation[leaf_index]
      _prepare_scratch!(cache, cell.local_dof_count)
      local_matrix = view(cache.matrix, 1:cell.local_dof_count, 1:cell.local_dof_count)
      local_rhs = view(cache.rhs, 1:cell.local_dof_count)
      fill!(local_matrix, zero(T))
      fill!(local_rhs, zero(T))

      for operator in plan.cell_operators
        cell_matrix!(local_matrix, operator, cell)
        cell_rhs!(local_rhs, operator, cell)
      end

      for face_index in leaf.boundary_faces
        face = @inbounds boundary_faces[face_index]

        for wrapped in plan.boundary_operators
          _matches(face, wrapped.boundary) || continue
          face_matrix!(local_matrix, wrapped.operator, face)
          face_rhs!(local_rhs, wrapped.operator, face)
        end
      end

      for surface_index in leaf.embedded_surfaces
        surface = @inbounds embedded_surfaces[surface_index]

        for wrapped in plan.surface_operators
          _matches(surface, wrapped.tag) || continue
          surface_matrix!(local_matrix, wrapped.operator, surface)
          surface_rhs!(local_rhs, wrapped.operator, surface)
        end
      end

      _static_condense_affine!(cache, cell, condensation_plan, local_matrix, local_rhs,
                               reduced_index, fixed, fixed_values, dirichlet_affine, tolerance)
      _scatter_affine_reduced!(cache, cell, condensation_plan.kept_local_dofs, local_matrix,
                               local_rhs, reduced_index, fixed, fixed_values, eliminated,
                               constraint_rows, dirichlet_affine)
    end
  end

  return nothing
end

# Generic affine assembly pass for local items whose full local dof block is
# scattered directly to the global system.
function _assemble_pass!(scratch, items, operators, fixed, fixed_values, pivot_rows, matrix_hook,
                         rhs_hook)
  _run_chunks!(scratch, length(items)) do cache, first_item, last_item
    for item_index in first_item:last_item
      item = @inbounds items[item_index]
      _prepare_scratch!(cache, item.local_dof_count)
      local_matrix = view(cache.matrix, 1:item.local_dof_count, 1:item.local_dof_count)
      local_rhs = view(cache.rhs, 1:item.local_dof_count)
      fill!(local_matrix, zero(eltype(cache.values)))
      fill!(local_rhs, zero(eltype(cache.global_rhs)))

      for operator in operators
        matrix_hook(local_matrix, operator, item)
        rhs_hook(local_rhs, operator, item)
      end

      _scatter_affine!(cache, item, local_matrix, local_rhs, fixed, fixed_values, pivot_rows)
    end
  end

  return nothing
end

# Reduced affine assembly pass used for interface operators, whose contributions
# bypass the cell-local static-condensation step but still have to respect the
# reduced-system indexing induced by constraints and condensed dofs.
function _assemble_reduced_pass!(scratch, items, operators, reduced_index, fixed, fixed_values,
                                 eliminated, constraint_rows,
                                 dirichlet_affine::_ReducedAffineRows{T}, matrix_hook,
                                 rhs_hook) where {T<:AbstractFloat}
  _run_chunks!(scratch, length(items)) do cache, first_item, last_item
    for item_index in first_item:last_item
      item = @inbounds items[item_index]
      _prepare_scratch!(cache, item.local_dof_count)
      local_matrix = view(cache.matrix, 1:item.local_dof_count, 1:item.local_dof_count)
      local_rhs = view(cache.rhs, 1:item.local_dof_count)
      fill!(local_matrix, zero(T))
      fill!(local_rhs, zero(T))

      for operator in operators
        matrix_hook(local_matrix, operator, item)
        rhs_hook(local_rhs, operator, item)
      end

      _scatter_affine_reduced!(cache, item, Base.OneTo(item.local_dof_count), local_matrix,
                               local_rhs, reduced_index, fixed, fixed_values, eliminated,
                               constraint_rows, dirichlet_affine)
    end
  end

  return nothing
end

# Specialized boundary-face traversal that first filters operators by the target
# boundary selector before calling the standard face hooks.
function _assemble_boundary_pass!(scratch, faces, operators, fixed, fixed_values, pivot_rows)
  _run_chunks!(scratch, length(faces)) do cache, first_face, last_face
    for face_index in first_face:last_face
      face = @inbounds faces[face_index]
      _prepare_scratch!(cache, face.local_dof_count)
      local_matrix = view(cache.matrix, 1:face.local_dof_count, 1:face.local_dof_count)
      local_rhs = view(cache.rhs, 1:face.local_dof_count)
      fill!(local_matrix, zero(eltype(cache.values)))
      fill!(local_rhs, zero(eltype(cache.global_rhs)))

      for wrapped in operators
        _matches(face, wrapped.boundary) || continue
        face_matrix!(local_matrix, wrapped.operator, face)
        face_rhs!(local_rhs, wrapped.operator, face)
      end

      _scatter_affine!(cache, face, local_matrix, local_rhs, fixed, fixed_values, pivot_rows)
    end
  end

  return nothing
end

# Specialized embedded-surface traversal using the surface operator hooks and
# optional symbolic surface-tag filtering.
function _assemble_surface_pass!(scratch, surfaces, operators, fixed, fixed_values, pivot_rows)
  _run_chunks!(scratch, length(surfaces)) do cache, first_surface, last_surface
    for surface_index in first_surface:last_surface
      surface = @inbounds surfaces[surface_index]
      _prepare_scratch!(cache, surface.local_dof_count)
      local_matrix = view(cache.matrix, 1:surface.local_dof_count, 1:surface.local_dof_count)
      local_rhs = view(cache.rhs, 1:surface.local_dof_count)
      fill!(local_matrix, zero(eltype(cache.values)))
      fill!(local_rhs, zero(eltype(cache.global_rhs)))

      for wrapped in operators
        _matches(surface, wrapped.tag) || continue
        surface_matrix!(local_matrix, wrapped.operator, surface)
        surface_rhs!(local_rhs, wrapped.operator, surface)
      end

      _scatter_affine!(cache, surface, local_matrix, local_rhs, fixed, fixed_values, pivot_rows)
    end
  end

  return nothing
end

# Generic nonlinear residual pass on a collection of local integration items.
function _residual_pass!(scratch, items, operators, state, fixed, pivot_rows, residual_hook)
  _run_chunks!(scratch, length(items)) do cache, first_item, last_item
    for item_index in first_item:last_item
      item = @inbounds items[item_index]
      _prepare_scratch!(cache, item.local_dof_count)
      local_rhs = view(cache.rhs, 1:item.local_dof_count)
      fill!(local_rhs, zero(eltype(cache.global_rhs)))

      for operator in operators
        residual_hook(local_rhs, operator, item, state)
      end

      _scatter_residual!(cache, item, local_rhs, fixed, pivot_rows)
    end
  end

  return nothing
end

# Boundary-face residual traversal with boundary filtering.
function _boundary_residual_pass!(scratch, faces, operators, state, fixed, pivot_rows)
  _run_chunks!(scratch, length(faces)) do cache, first_face, last_face
    for face_index in first_face:last_face
      face = @inbounds faces[face_index]
      _prepare_scratch!(cache, face.local_dof_count)
      local_rhs = view(cache.rhs, 1:face.local_dof_count)
      fill!(local_rhs, zero(eltype(cache.global_rhs)))

      for wrapped in operators
        _matches(face, wrapped.boundary) || continue
        face_residual!(local_rhs, wrapped.operator, face, state)
      end

      _scatter_residual!(cache, face, local_rhs, fixed, pivot_rows)
    end
  end

  return nothing
end

# Embedded-surface residual traversal with optional surface-tag filtering.
function _surface_residual_pass!(scratch, surfaces, operators, state, fixed, pivot_rows)
  _run_chunks!(scratch, length(surfaces)) do cache, first_surface, last_surface
    for surface_index in first_surface:last_surface
      surface = @inbounds surfaces[surface_index]
      _prepare_scratch!(cache, surface.local_dof_count)
      local_rhs = view(cache.rhs, 1:surface.local_dof_count)
      fill!(local_rhs, zero(eltype(cache.global_rhs)))

      for wrapped in operators
        _matches(surface, wrapped.tag) || continue
        surface_residual!(local_rhs, wrapped.operator, surface, state)
      end

      _scatter_residual!(cache, surface, local_rhs, fixed, pivot_rows)
    end
  end

  return nothing
end

# Generic nonlinear tangent pass on a collection of local integration items.
function _tangent_pass!(scratch, items, operators, state, fixed, pivot_rows, tangent_hook)
  _run_chunks!(scratch, length(items)) do cache, first_item, last_item
    for item_index in first_item:last_item
      item = @inbounds items[item_index]
      _prepare_scratch!(cache, item.local_dof_count)
      local_matrix = view(cache.matrix, 1:item.local_dof_count, 1:item.local_dof_count)
      fill!(local_matrix, zero(eltype(cache.values)))

      for operator in operators
        tangent_hook(local_matrix, operator, item, state)
      end

      _scatter_tangent!(cache, item, local_matrix, fixed, pivot_rows)
    end
  end

  return nothing
end

# Boundary-face tangent traversal with boundary filtering.
function _boundary_tangent_pass!(scratch, faces, operators, state, fixed, pivot_rows)
  _run_chunks!(scratch, length(faces)) do cache, first_face, last_face
    for face_index in first_face:last_face
      face = @inbounds faces[face_index]
      _prepare_scratch!(cache, face.local_dof_count)
      local_matrix = view(cache.matrix, 1:face.local_dof_count, 1:face.local_dof_count)
      fill!(local_matrix, zero(eltype(cache.values)))

      for wrapped in operators
        _matches(face, wrapped.boundary) || continue
        face_tangent!(local_matrix, wrapped.operator, face, state)
      end

      _scatter_tangent!(cache, face, local_matrix, fixed, pivot_rows)
    end
  end

  return nothing
end

# Embedded-surface tangent traversal with optional surface-tag filtering.
function _surface_tangent_pass!(scratch, surfaces, operators, state, fixed, pivot_rows)
  _run_chunks!(scratch, length(surfaces)) do cache, first_surface, last_surface
    for surface_index in first_surface:last_surface
      surface = @inbounds surfaces[surface_index]
      _prepare_scratch!(cache, surface.local_dof_count)
      local_matrix = view(cache.matrix, 1:surface.local_dof_count, 1:surface.local_dof_count)
      fill!(local_matrix, zero(eltype(cache.values)))

      for wrapped in operators
        _matches(surface, wrapped.tag) || continue
        surface_tangent!(local_matrix, wrapped.operator, surface, state)
      end

      _scatter_tangent!(cache, surface, local_matrix, fixed, pivot_rows)
    end
  end

  return nothing
end

# Check that the thread-local dense buffers are large enough for the current
# local integration item.
function _prepare_scratch!(scratch::_ThreadScratch, local_dof_count::Int)
  size(scratch.matrix, 1) >= local_dof_count ||
    throw(ArgumentError("thread scratch matrix is too small for the local item"))
  size(scratch.factor_matrix, 1) >= local_dof_count ||
    throw(ArgumentError("thread scratch factor matrix is too small for the local item"))
  size(scratch.solve_matrix, 1) >= local_dof_count ||
    throw(ArgumentError("thread scratch solve matrix is too small for the local item"))
  length(scratch.rhs) >= local_dof_count ||
    throw(ArgumentError("thread scratch rhs is too small for the local item"))
  length(scratch.work_rhs) >= local_dof_count ||
    throw(ArgumentError("thread scratch work rhs is too small for the local item"))
  return scratch
end

# Allocate one scratch bundle per worker, sized by the largest local item over
# all cells/faces/interfaces/surfaces in the compiled integration cache.
function _scratch_buffers(::Type{T}, ndofs::Int,
                          integration::_CompiledIntegration) where {T<:AbstractFloat}
  worker_count = _worker_count(length(integration.cells), length(integration.boundary_faces),
                               length(integration.interfaces),
                               length(integration.embedded_surfaces))
  max_local_dofs = _max_local_dof_count(integration)
  return [_ThreadScratch(T, ndofs, max_local_dofs) for _ in 1:worker_count]
end

# Choose how many worker-local scratch bundles to allocate.
function _worker_count(cell_count::Int, face_count::Int, interface_count::Int, surface_count::Int)
  total = max(cell_count, face_count, interface_count, surface_count, 1)
  return min(Threads.nthreads(), total)
end

# Reset all triplet and right-hand-side accumulators in the thread-local
# scratch buffers before a new assembly/evaluation pass.
function _reset_scratch!(scratch)
  for cache in scratch
    empty!(cache.rows)
    empty!(cache.cols)
    empty!(cache.values)
    empty!(cache.shift_rows)
    empty!(cache.shift_values)
    empty!(cache.reconstruction_rows)
    empty!(cache.reconstruction_cols)
    empty!(cache.reconstruction_values)
    fill!(cache.global_rhs, zero(eltype(cache.global_rhs)))
  end

  return scratch
end

# Parallel chunk scheduler shared by the various assembly passes.
function _run_chunks!(f::F, scratch, item_count::Int) where {F}
  item_count == 0 && return nothing
  worker_count = min(length(scratch), item_count)
  chunk_size = _dynamic_chunk_size(item_count, worker_count)
  next_item = Threads.Atomic{Int}(1)

  if worker_count == 1
    _run_claimed_chunks!(f, scratch[1], next_item, chunk_size, item_count)
    return nothing
  end

  @sync for worker in 1:worker_count
    Threads.@spawn _run_claimed_chunks!(f, scratch[worker], next_item, chunk_size, item_count)
  end

  return nothing
end

# Claim work chunks atomically and execute the caller-supplied local traversal.
function _run_claimed_chunks!(f::F, cache, next_item::Threads.Atomic{Int}, chunk_size::Int,
                              item_count::Int) where {F}
  while true
    first_item = Threads.atomic_add!(next_item, chunk_size)
    first_item > item_count && return nothing
    last_item = min(item_count, first_item + chunk_size - 1)
    f(cache, first_item, last_item)
  end
end

# Scattering local algebra to full or reduced global storage.

# Scatter a fully assembled local affine contribution to the full global system.
# Each local test/trial function may itself expand into several global dofs due
# to continuity constraints, so scattering loops over the sparse local term maps
# stored in the compiled integration item.
function _scatter_affine!(cache::_ThreadScratch{T}, item::_AssemblyValues,
                          local_matrix::AbstractMatrix{T}, local_rhs::AbstractVector{T},
                          fixed::BitVector, fixed_values::Vector{T},
                          pivot_rows::BitVector) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for local_row in 1:item.local_dof_count
    rhs_value = local_rhs[local_row]
    row_first = item.term_offsets[local_row]
    row_last = item.term_offsets[local_row+1] - 1

    if abs(rhs_value) > tolerance
      @inbounds for row_term_index in row_first:row_last
        row = item.term_indices[row_term_index]
        fixed[row] && continue
        pivot_rows[row] && continue
        cache.global_rhs[row] += item.term_coefficients[row_term_index] * rhs_value
      end
    end

    for local_col in 1:item.local_dof_count
      coefficient = local_matrix[local_row, local_col]
      abs(coefficient) > tolerance || continue
      col_first = item.term_offsets[local_col]
      col_last = item.term_offsets[local_col+1] - 1

      @inbounds for row_term_index in row_first:row_last
        row = item.term_indices[row_term_index]
        fixed[row] && continue
        pivot_rows[row] && continue
        row_scale = item.term_coefficients[row_term_index] * coefficient

        for col_term_index in col_first:col_last
          global_col = item.term_indices[col_term_index]
          contribution = row_scale * item.term_coefficients[col_term_index]
          abs(contribution) > tolerance || continue

          if fixed[global_col]
            cache.global_rhs[row] -= contribution * fixed_values[global_col]
          else
            push!(cache.rows, row)
            push!(cache.cols, global_col)
            push!(cache.values, contribution)
          end
        end
      end
    end
  end

  return nothing
end

# Scatter a local affine contribution directly to the reduced system, resolving
# condensed and Dirichlet-eliminated dofs on the fly through the reduced affine
# reconstruction rows.
function _scatter_affine_reduced!(cache::_ThreadScratch{T}, item::_AssemblyValues, local_dofs,
                                  local_matrix::AbstractMatrix{T}, local_rhs::AbstractVector{T},
                                  reduced_index::Vector{Int}, fixed::BitVector,
                                  fixed_values::Vector{T}, eliminated::BitVector,
                                  constraint_rows::BitVector,
                                  dirichlet_affine::_ReducedAffineRows{T}) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for local_row in local_dofs
    rhs_value = local_rhs[local_row]
    row_first = item.term_offsets[local_row]
    row_last = item.term_offsets[local_row+1] - 1

    if abs(rhs_value) > tolerance
      @inbounds for row_term_index in row_first:row_last
        row = item.term_indices[row_term_index]
        fixed[row] && continue
        constraint_rows[row] && continue
        row_value = item.term_coefficients[row_term_index] * rhs_value

        if eliminated[row]
          for pointer in _affine_row_pointer_range(dirichlet_affine, row)
            contribution = row_value * dirichlet_affine.coefficients[pointer]
            abs(contribution) > tolerance || continue
            cache.global_rhs[dirichlet_affine.indices[pointer]] += contribution
          end
        else
          row_reduced = reduced_index[row]
          row_reduced != 0 ||
            throw(ArgumentError("local row $row does not belong to the reduced system"))
          cache.global_rhs[row_reduced] += row_value
        end
      end
    end

    for local_col in local_dofs
      coefficient = local_matrix[local_row, local_col]
      abs(coefficient) > tolerance || continue
      col_first = item.term_offsets[local_col]
      col_last = item.term_offsets[local_col+1] - 1

      @inbounds for row_term_index in row_first:row_last
        row = item.term_indices[row_term_index]
        fixed[row] && continue
        row_scale = item.term_coefficients[row_term_index] * coefficient

        if eliminated[row]
          for pointer in _affine_row_pointer_range(dirichlet_affine, row)
            row_reduced = dirichlet_affine.indices[pointer]
            reduced_row_scale = row_scale * dirichlet_affine.coefficients[pointer]
            abs(reduced_row_scale) > tolerance || continue

            for col_term_index in col_first:col_last
              contribution = reduced_row_scale * item.term_coefficients[col_term_index]
              abs(contribution) > tolerance || continue
              _scatter_reduced_column!(cache, row_reduced, contribution,
                                       item.term_indices[col_term_index], reduced_index, fixed,
                                       fixed_values, dirichlet_affine, tolerance)
            end
          end
          continue
        end

        constraint_rows[row] && continue
        row_reduced = reduced_index[row]
        row_reduced != 0 ||
          throw(ArgumentError("local row $row does not belong to the reduced system"))

        for col_term_index in col_first:col_last
          contribution = row_scale * item.term_coefficients[col_term_index]
          abs(contribution) > tolerance || continue
          _scatter_reduced_column!(cache, row_reduced, contribution,
                                   item.term_indices[col_term_index], reduced_index, fixed,
                                   fixed_values, dirichlet_affine, tolerance)
        end
      end
    end
  end

  return nothing
end

# Scatter a local nonlinear residual contribution to the full global residual,
# skipping rows reserved for explicit constraints.
function _scatter_residual!(cache::_ThreadScratch{T}, item::_AssemblyValues,
                            local_rhs::AbstractVector{T}, fixed::BitVector,
                            pivot_rows::BitVector) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for local_row in 1:item.local_dof_count
    rhs_value = local_rhs[local_row]
    abs(rhs_value) > tolerance || continue

    @inbounds for row_term_index in item.term_offsets[local_row]:(item.term_offsets[local_row+1]-1)
      row = item.term_indices[row_term_index]
      fixed[row] && continue
      pivot_rows[row] && continue
      cache.global_rhs[row] += item.term_coefficients[row_term_index] * rhs_value
    end
  end

  return nothing
end

# Scatter a local nonlinear tangent contribution to the full global tangent.
function _scatter_tangent!(cache::_ThreadScratch{T}, item::_AssemblyValues,
                           local_matrix::AbstractMatrix{T}, fixed::BitVector,
                           pivot_rows::BitVector) where {T<:AbstractFloat}
  tolerance = 1000 * eps(T)

  for local_row in 1:item.local_dof_count
    row_first = item.term_offsets[local_row]
    row_last = item.term_offsets[local_row+1] - 1

    for local_col in 1:item.local_dof_count
      coefficient = local_matrix[local_row, local_col]
      abs(coefficient) > tolerance || continue
      col_first = item.term_offsets[local_col]
      col_last = item.term_offsets[local_col+1] - 1

      @inbounds for row_term_index in row_first:row_last
        row = item.term_indices[row_term_index]
        fixed[row] && continue
        pivot_rows[row] && continue
        row_scale = item.term_coefficients[row_term_index] * coefficient

        for col_term_index in col_first:col_last
          contribution = row_scale * item.term_coefficients[col_term_index]
          abs(contribution) > tolerance || continue
          push!(cache.rows, row)
          push!(cache.cols, item.term_indices[col_term_index])
          push!(cache.values, contribution)
        end
      end
    end
  end

  return nothing
end

# Accumulate one contribution to an eliminated/condensed affine relation while
# simultaneously accounting for fixed Dirichlet values and reduced-system
# indexing.
function _accumulate_affine_relation!(row::Dict{Int,T}, contribution::T, global_dof::Int,
                                      reduced_index::Vector{Int}, fixed::BitVector,
                                      fixed_values::Vector{T},
                                      dirichlet_affine::_ReducedAffineRows{T},
                                      tolerance::T) where {T<:AbstractFloat}
  abs(contribution) > tolerance || return zero(T)

  if fixed[global_dof]
    return contribution * fixed_values[global_dof]
  end

  reduced_dof = reduced_index[global_dof]

  if reduced_dof != 0
    row[reduced_dof] = get(row, reduced_dof, zero(T)) + contribution
    return zero(T)
  end

  dirichlet_affine.eliminated[global_dof] ||
    throw(ArgumentError("global dof $global_dof does not belong to the reduced system"))

  shift_value = contribution * dirichlet_affine.shifts[global_dof]

  for pointer in _affine_row_pointer_range(dirichlet_affine, global_dof)
    reduced_dof = dirichlet_affine.indices[pointer]
    reduced_contribution = contribution * dirichlet_affine.coefficients[pointer]
    abs(reduced_contribution) > tolerance || continue
    _accumulate_reduced_row!(row, reduced_dof, reduced_contribution)
  end

  return shift_value
end

# Scatter one column contribution into the reduced system, resolving fixed and
# eliminated columns through the reduced affine representation when necessary.
function _scatter_reduced_column!(cache::_ThreadScratch{T}, row_reduced::Int, contribution::T,
                                  global_dof::Int, reduced_index::Vector{Int}, fixed::BitVector,
                                  fixed_values::Vector{T}, dirichlet_affine::_ReducedAffineRows{T},
                                  tolerance::T) where {T<:AbstractFloat}
  if fixed[global_dof]
    cache.global_rhs[row_reduced] -= contribution * fixed_values[global_dof]
    return nothing
  end

  col_reduced = reduced_index[global_dof]

  if col_reduced != 0
    push!(cache.rows, row_reduced)
    push!(cache.cols, col_reduced)
    push!(cache.values, contribution)
    return nothing
  end

  dirichlet_affine.eliminated[global_dof] ||
    throw(ArgumentError("local column $global_dof does not belong to the reduced system"))

  shift_value = contribution * dirichlet_affine.shifts[global_dof]
  abs(shift_value) > tolerance && (cache.global_rhs[row_reduced] -= shift_value)

  for pointer in _affine_row_pointer_range(dirichlet_affine, global_dof)
    reduced_contribution = contribution * dirichlet_affine.coefficients[pointer]
    abs(reduced_contribution) > tolerance || continue
    push!(cache.rows, row_reduced)
    push!(cache.cols, dirichlet_affine.indices[pointer])
    push!(cache.values, reduced_contribution)
  end

  return nothing
end

# Static condensation and default solve-topology construction.

# Build a cell-local static-condensation plan by eliminating interior modes that
# map one-to-one to unconstrained global dofs. Only such modes can be safely
# condensed without first introducing additional local algebra.
function _static_condensation(cells, fixed::BitVector, pivot_rows::BitVector)
  plans = Vector{_StaticCondensationPlan}(undef, length(cells))
  condensed = falses(length(fixed))

  for cell_index in eachindex(cells)
    cell = cells[cell_index]
    eliminated_local_dofs = Int[]
    eliminated_global_dofs = Int[]
    eliminated_local = falses(cell.local_dof_count)

    for local_dof in cell.interior_local_dofs
      global_dof = cell.single_term_indices[local_dof]
      global_dof != 0 ||
        throw(ArgumentError("cell interior modes must map to a single global dof for static condensation"))
      coefficient = cell.single_term_coefficients[local_dof]
      abs(coefficient - one(typeof(coefficient))) <= 1000 * eps(typeof(coefficient)) ||
        throw(ArgumentError("cell interior modes must map to a unit global dof for static condensation"))
      (fixed[global_dof] || pivot_rows[global_dof]) && continue
      push!(eliminated_local_dofs, local_dof)
      push!(eliminated_global_dofs, global_dof)
      eliminated_local[local_dof] = true
      condensed[global_dof] = true
    end

    kept_local_dofs = Int[]

    for local_dof in 1:cell.local_dof_count
      eliminated_local[local_dof] || push!(kept_local_dofs, local_dof)
    end

    plans[cell_index] = _StaticCondensationPlan(kept_local_dofs, eliminated_local_dofs,
                                                eliminated_global_dofs)
  end

  return condensed, plans
end

# Degenerate "no condensation" plan used when interface couplings prevent purely
# local elimination.
function _identity_condensation(cells, ndofs::Int)
  plans = Vector{_StaticCondensationPlan}(undef, length(cells))

  for cell_index in eachindex(cells)
    cell = cells[cell_index]
    plans[cell_index] = _StaticCondensationPlan(collect(1:cell.local_dof_count), Int[], Int[])
  end

  return falses(ndofs), plans
end

# Static condensation is disabled whenever interface operators are present,
# because then a cell-interior elimination based only on the cell matrix would
# generally miss off-cell couplings.
function _condensation_data(plan::AssemblyPlan, fixed::BitVector, pivot_rows::BitVector)
  isempty(plan.interface_operators) &&
    return _static_condensation(plan.integration.cells, fixed, pivot_rows)
  return _identity_condensation(plan.integration.cells, length(fixed))
end

# Build the geometric information used by the default iterative solve path. Each
# reduced cell contributes one Schwarz patch, and the coarse space is built from
# multilinear root-grid vertex functions.
function _solve_topology(plan::AssemblyPlan{D,T}, condensation, reduced_index::Vector{Int},
                         solve_dofs::Vector{Int}) where {D,T<:AbstractFloat}
  cells = plan.integration.cells
  basic_patches = Vector{Vector{Int}}(undef, length(cells))
  leaf_patches = Vector{Vector{Int}}(undef, length(cells))
  space = plan.layout.slots[1].space
  grid_data = grid(space)
  domain_data = space.domain
  support_sums = zeros(T, D, length(solve_dofs))
  support_counts = zeros(Int, length(solve_dofs))

  for cell_index in eachindex(cells)
    cell = cells[cell_index]
    patch = _leaf_patch_dofs(cell, condensation[cell_index].kept_local_dofs, reduced_index)
    basic_patches[cell_index] = patch
    leaf_patches[cell_index] = patch
    center = cell_center(domain_data, cell.leaf)

    for reduced_dof in patch
      support_counts[reduced_dof] += 1

      for axis in 1:D
        support_sums[axis, reduced_dof] += center[axis]
      end
    end
  end

  return _PatchSolveTopology{T}(leaf_patches,
                                _geometric_coarse_prolongation(T, plan.layout, domain_data,
                                                               grid_data, support_sums,
                                                               support_counts, solve_dofs))
end

# Collect the reduced dofs touched by the specified local cell dofs.
function _leaf_patch_dofs(cell::CellValues, local_dofs, reduced_index::Vector{Int})
  patch = Int[]

  for local_dof in local_dofs
    @inbounds for term_index in cell.term_offsets[local_dof]:(cell.term_offsets[local_dof+1]-1)
      reduced_dof = reduced_index[cell.term_indices[term_index]]
      reduced_dof == 0 && continue
      push!(patch, reduced_dof)
    end
  end

  return sort!(unique!(patch))
end

# Build a low-order geometric coarse prolongation by interpolating each reduced
# dof to the multilinear basis on the root-grid vertices of its field component.
function _geometric_coarse_prolongation(::Type{T}, layout::FieldLayout{D,T},
                                        domain_data::Domain{D,T}, grid_data::CartesianGrid{D},
                                        support_sums::Matrix{T}, support_counts::Vector{Int},
                                        solve_dofs::Vector{Int}) where {D,T<:AbstractFloat}
  isempty(solve_dofs) && return spzeros(T, 0, 0)
  root_counts = root_cell_counts(grid_data)
  vertex_counts = ntuple(axis -> root_counts[axis] + 1, D)
  vertices_per_component = prod(vertex_counts)
  total_components = sum(component_count(slot.field) for slot in layout.slots)
  rows = Int[]
  cols = Int[]
  values = T[]

  for reduced_dof in eachindex(solve_dofs)
    support_counts[reduced_dof] == 0 && continue
    point = ntuple(axis -> support_sums[axis, reduced_dof] / support_counts[reduced_dof], D)
    component = _component_slot(layout, solve_dofs[reduced_dof])
    _append_root_basis_weights!(rows, cols, values, reduced_dof, component, point, domain_data,
                                root_counts, vertex_counts, vertices_per_component)
  end

  return _compress_coarse_prolongation(T, length(solve_dofs), rows, cols, values,
                                       total_components * vertices_per_component)
end

# Append the multilinear root-grid vertex weights associated with one support
# point.
function _append_root_basis_weights!(rows::Vector{Int}, cols::Vector{Int}, values::Vector{T},
                                     reduced_dof::Int, component::Int, point::NTuple{D,T},
                                     domain_data::Domain{D,T}, root_counts::NTuple{D,Int},
                                     vertex_counts::NTuple{D,Int},
                                     vertices_per_component::Int) where {D,T<:AbstractFloat}
  lower = Vector{Int}(undef, D)
  λ = Vector{T}(undef, D)

  for axis in 1:D
    root_width = extent(domain_data, axis) / root_counts[axis]
    scaled = (point[axis] - origin(domain_data, axis)) / root_width

    if scaled <= zero(T)
      lower[axis] = 0
      λ[axis] = zero(T)
    elseif scaled >= T(root_counts[axis])
      lower[axis] = root_counts[axis] - 1
      λ[axis] = one(T)
    else
      lower[axis] = min(root_counts[axis] - 1, floor(Int, scaled))
      λ[axis] = scaled - lower[axis]
    end
  end

  for mask in 0:((1<<D)-1)
    weight = one(T)
    vertex = ntuple(axis -> begin
                      bit = (mask >> (axis - 1)) & 1
                      weight *= bit == 0 ? (one(T) - λ[axis]) : λ[axis]
                      lower[axis] + bit
                    end, D)
    weight == zero(T) && continue
    column = (component - 1) * vertices_per_component + _root_vertex_index(vertex, vertex_counts)
    push!(rows, reduced_dof)
    push!(cols, column)
    push!(values, weight)
  end

  return nothing
end

# Mixed-radix flattening of a root-grid vertex index.
function _root_vertex_index(vertex::NTuple{D,Int}, vertex_counts::NTuple{D,Int}) where {D}
  index = 1
  stride = 1

  for axis in 1:D
    index += vertex[axis] * stride
    stride *= vertex_counts[axis]
  end

  return index
end

# Remove unused coarse-space columns and renumber the remaining ones densely.
function _compress_coarse_prolongation(::Type{T}, ndofs::Int, rows::Vector{Int}, cols::Vector{Int},
                                       values::Vector{T}, max_column::Int) where {T<:AbstractFloat}
  isempty(cols) && return spzeros(T, ndofs, 0)
  used_columns = falses(max_column)

  for column in cols
    used_columns[column] = true
  end

  remap = zeros(Int, max_column)
  column_count = 0

  for column in eachindex(used_columns)
    used_columns[column] || continue
    column_count += 1
    remap[column] = column_count
  end

  for index in eachindex(cols)
    cols[index] = remap[cols[index]]
  end

  return sparse(rows, cols, values, ndofs, column_count)
end

# Map a global dof to its field-component slot in the concatenated layout.
function _component_slot(layout::FieldLayout, global_dof::Int)
  component_offset = 0

  for slot in layout.slots
    first_dof = slot.offset
    last_dof = slot.offset + slot.dof_count - 1

    if first_dof <= global_dof <= last_dof
      local_dof = global_dof - first_dof
      return component_offset + fld(local_dof, slot.scalar_dof_count) + 1
    end

    component_offset += component_count(slot.field)
  end

  throw(ArgumentError("global dof $global_dof does not belong to the field layout"))
end

# Apply local static condensation to one cell matrix/rhs and record the affine
# reconstruction rows that recover the eliminated interior global dofs from the
# reduced trace dofs.
function _static_condense_affine!(cache::_ThreadScratch{T}, item::CellValues,
                                  plan::_StaticCondensationPlan, local_matrix::AbstractMatrix{T},
                                  local_rhs::AbstractVector{T}, reduced_index::Vector{Int},
                                  fixed::BitVector, fixed_values::Vector{T},
                                  dirichlet_affine::_ReducedAffineRows{T},
                                  tolerance::T) where {T<:AbstractFloat}
  interior_count = length(plan.eliminated_local_dofs)
  interior_count == 0 && return nothing
  kept_count = length(plan.kept_local_dofs)
  factor_matrix = view(cache.factor_matrix, 1:interior_count, 1:interior_count)
  solve_matrix = view(cache.solve_matrix, 1:interior_count, 1:kept_count)
  work_rhs = view(cache.work_rhs, 1:interior_count)

  for row_index in 1:interior_count
    local_row = plan.eliminated_local_dofs[row_index]
    work_rhs[row_index] = local_rhs[local_row]

    for col_index in 1:interior_count
      factor_matrix[row_index, col_index] = local_matrix[local_row,
                                                         plan.eliminated_local_dofs[col_index]]
    end

    for col_index in 1:kept_count
      solve_matrix[row_index, col_index] = local_matrix[local_row, plan.kept_local_dofs[col_index]]
    end
  end

  _with_serialized_blas() do
    factorization = lu!(factor_matrix)
    ldiv!(factorization, work_rhs)
    ldiv!(factorization, solve_matrix)
  end

  for kept_row_index in 1:kept_count
    local_row = plan.kept_local_dofs[kept_row_index]

    for kept_col_index in 1:kept_count
      correction = zero(T)

      for interior_index in 1:interior_count
        correction += local_matrix[local_row, plan.eliminated_local_dofs[interior_index]] *
                      solve_matrix[interior_index, kept_col_index]
      end

      local_matrix[local_row, plan.kept_local_dofs[kept_col_index]] -= correction
    end

    correction = zero(T)

    for interior_index in 1:interior_count
      correction += local_matrix[local_row, plan.eliminated_local_dofs[interior_index]] *
                    work_rhs[interior_index]
    end

    local_rhs[local_row] -= correction
  end

  for interior_index in 1:interior_count
    global_dof = plan.eliminated_global_dofs[interior_index]
    shift_value = work_rhs[interior_index]
    row = Dict{Int,T}()

    for kept_index in 1:kept_count
      local_coefficient = -solve_matrix[interior_index, kept_index]
      abs(local_coefficient) > tolerance || continue

      local_dof = plan.kept_local_dofs[kept_index]

      for term_index in item.term_offsets[local_dof]:(item.term_offsets[local_dof+1]-1)
        shift_value += _accumulate_affine_relation!(row,
                                                    local_coefficient *
                                                    item.term_coefficients[term_index],
                                                    item.term_indices[term_index], reduced_index,
                                                    fixed, fixed_values, dirichlet_affine,
                                                    tolerance)
      end
    end

    push!(cache.shift_rows, global_dof)
    push!(cache.shift_values, shift_value)

    for (reduced_dof, coefficient) in row
      abs(coefficient) > tolerance || continue
      push!(cache.reconstruction_rows, global_dof)
      push!(cache.reconstruction_cols, reduced_dof)
      push!(cache.reconstruction_values, coefficient)
    end
  end

  return nothing
end

# Check that a state is defined on the same field layout as the compiled plan.
function _check_state(plan::AssemblyPlan, state::State)
  _matching_layout(plan.layout, state.layout) ||
    throw(ArgumentError("state layout does not match the compiled plan"))
  return state
end

# Reduction of compiled constraints and reconstruction of full-layout states.

# Reduce Dirichlet affine rows to reduced-system coordinates and collect the
# associated constant shifts.
function _reduce_dirichlet(dirichlet::_CompiledDirichlet{T}, reduced_index::Vector{Int},
                           fixed::BitVector, fixed_values::Vector{T}) where {T<:AbstractFloat}
  ndofs = length(reduced_index)
  eliminated = falses(ndofs)
  shifts = copy(fixed_values)
  temp_rows = Dict{Int,Vector{Pair{Int,T}}}()

  for row in dirichlet.rows
    eliminated[row.pivot] = true
    reduced_row = Dict{Int,T}()
    shift_value = row.rhs

    for index in eachindex(row.indices)
      global_dof = row.indices[index]
      coefficient = row.coefficients[index]

      if fixed[global_dof]
        shift_value += coefficient * fixed_values[global_dof]
        continue
      end

      reduced_dof = reduced_index[global_dof]
      reduced_dof != 0 ||
        throw(ArgumentError("Dirichlet free trace dof $global_dof does not belong to the reduced system"))
      _accumulate_reduced_row!(reduced_row, reduced_dof, coefficient)
    end

    shifts[row.pivot] = shift_value

    pairs = _sorted_reduced_row(reduced_row)
    isempty(pairs) || (temp_rows[row.pivot] = pairs)
  end

  return _packed_reduced_affine_rows(T, eliminated, shifts, temp_rows)
end

# Initialize the reconstruction-row dictionary with the reduced Dirichlet
# relations.
function _reconstruction_rows(dirichlet_affine::_ReducedAffineRows{T}) where {T<:AbstractFloat}
  rows = Dict{Int,Vector{Pair{Int,T}}}()

  for dof in eachindex(dirichlet_affine.eliminated)
    dirichlet_affine.eliminated[dof] || continue
    pairs = _affine_row_pairs(dirichlet_affine, dof)
    isempty(pairs) || (rows[dof] = pairs)
  end

  return rows
end

# Build the map from full global dofs to reduced solve dofs.
function _reduced_dof_index(fixed::BitVector, condensed::BitVector, eliminated::BitVector)
  solve_dofs = Int[]
  reduced_index = zeros(Int, length(fixed))

  for dof in eachindex(fixed)
    fixed[dof] && continue
    condensed[dof] && continue
    eliminated[dof] && continue
    push!(solve_dofs, dof)
    reduced_index[dof] = length(solve_dofs)
  end

  return reduced_index, solve_dofs
end

# Assemble the sparse reconstruction matrix `R` such that
#
#   full_values = shift + R * reduced_values.
function _reconstruction_matrix(::Type{T}, ndofs::Int, solve_dofs::Vector{Int},
                                condensed_rows::Dict{Int,Vector{Pair{Int,T}}}) where {T<:AbstractFloat}
  rows = Int[]
  cols = Int[]
  coefficients = T[]
  entry_count = length(solve_dofs) +
                sum(length(pairs) for pairs in Base.values(condensed_rows); init=0)
  sizehint!(rows, entry_count)
  sizehint!(cols, entry_count)
  sizehint!(coefficients, entry_count)

  for reduced_dof in eachindex(solve_dofs)
    push!(rows, solve_dofs[reduced_dof])
    push!(cols, reduced_dof)
    push!(coefficients, one(T))
  end

  for (global_dof, pairs) in condensed_rows
    for pair in pairs
      push!(rows, global_dof)
      push!(cols, pair.first)
      push!(coefficients, pair.second)
    end
  end

  return sparse(rows, cols, coefficients, ndofs, length(solve_dofs))
end

# Push mean-value constraints through the same reduced-system elimination and
# reconstruction machinery used for Dirichlet rows and static condensation.
function _reduce_mean_constraints(constraints::Vector{_CompiledLinearConstraint{T}},
                                  reduced_index::Vector{Int}, shift::Vector{T},
                                  condensed_rows::Dict{Int,Vector{Pair{Int,T}}}) where {T<:AbstractFloat}
  isempty(constraints) && return _CompiledLinearConstraint{T}[]
  reduced = _CompiledLinearConstraint{T}[]

  for constraint in constraints
    reduced_pivot = reduced_index[constraint.pivot]
    reduced_pivot != 0 || throw(ArgumentError("mean-value pivot must remain in the reduced system"))
    row = Dict{Int,T}()
    rhs = constraint.rhs

    for index in eachindex(constraint.indices)
      global_dof = constraint.indices[index]
      coefficient = constraint.coefficients[index]
      rhs -= coefficient * shift[global_dof]
      reduced_dof = reduced_index[global_dof]

      if reduced_dof != 0
        _accumulate_reduced_row!(row, reduced_dof, coefficient)
        continue
      end

      pairs = get(condensed_rows, global_dof, nothing)
      pairs === nothing && continue

      for pair in pairs
        _accumulate_reduced_row!(row, pair.first, coefficient * pair.second)
      end
    end

    pairs = _sorted_reduced_row(row)
    isempty(pairs) && throw(ArgumentError("mean-value constraint leaves no reduced dofs"))
    push!(reduced,
          _CompiledLinearConstraint(reduced_pivot, [pair.first for pair in pairs],
                                    T[pair.second for pair in pairs], rhs))
  end

  return reduced
end

# Reconstruct the full coefficient vector from the reduced solution.
function _expand_system_values(system::AffineSystem{T},
                               reduced_values::AbstractVector{T}) where {T<:AbstractFloat}
  length(reduced_values) == size(system.reconstruction, 2) ||
    throw(ArgumentError("reduced coefficient vector length must match the assembled system size"))
  values = copy(system.shift)
  values .+= system.reconstruction * reduced_values
  return values
end

# Boundary-face selector match.
function _matches(face::FaceValues, boundary::BoundaryFace)
  face.axis == boundary.axis && face.side == boundary.side
end

# Embedded-surface selector match.
_matches(::SurfaceValues, ::Nothing) = true
_matches(surface::SurfaceValues, tag::Symbol) = surface.tag === tag

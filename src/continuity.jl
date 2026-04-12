# This file builds the inter-element conformity structure of an hp space on a
# Cartesian refinement tree.
#
# Conceptually, this is the algebraic heart of the package. Earlier files decide
# which local tensor-product modes exist on each leaf. Later files want a
# compiled `HpSpace` whose local modes are either coupled continuously across
# interfaces (`:cg`) or left independent across them (`:dg`). This file performs
# that translation.
#
# The hierarchical one-dimensional finite-element factors explain why the
# continuity algebra can be organized cleanly:
# - on axes with degree `p ≥ 1`, indices `0` and `1` are endpoint modes and
#   therefore carry trace data,
# - indices `≥ 2` vanish at both endpoints and are interior in that direction,
# - on DG axes with `p = 0`, the single local factor is a cellwise constant mode
#   that remains leaf-local because no continuity coupling is imposed there.
#
# The key mathematical object is the trace space on one face. On matching faces,
# continuity means that two neighboring leaves represent the same coefficient in
# that common trace space. On hanging faces, the two traces first have to be
# restricted to a common overlap patch on the finest logical lattice; only then
# can their coefficients be compared. This is why the implementation repeatedly
# moves between local tensor-product modes, face-local trace modes, and sparse
# algebraic constraint rows.
#
# For spaces with at least one continuous axis, the compiler runs in six phases:
# 1. freeze the active leaves, lift them to a common finest logical lattice, and
#    cache their local admissible mode patterns;
# 2. decide, from the requested continuity policy, which local traces should
#    enter a coupling algebra at all;
# 3. introduce one provisional boundary variable for every participating local
#    boundary trace;
# 4. enumerate all active face pairs, merge matching traces by union-find, and
#    assemble hanging-face continuity rows on common overlap patches;
# 5. eliminate the resulting sparse boundary-variable graph and assign global
#    scalar dofs to the surviving free trace representatives;
# 6. compile every leaf into a sparse expansion from local modes to finalized
#    global scalar dofs.
#
# For fully discontinuous spaces, the same leaf-pattern infrastructure is still
# used, but the face-coupling phases are skipped and every active local mode
# receives an independent global scalar dof. For mixed per-axis CG/DG spaces,
# only traces on CG axes enter the global coupling algebra; DG-axis traces
# remain leaf-local.
#
# Core data structures shared across the compilation phases.

# Internal continuity-policy abstraction used by space compilation. The public
# API exposes per-axis `:cg`/`:dg` choices, and the compiler threads the
# resulting normalized policy explicitly so the same stages handle full and
# mixed spaces.
abstract type _AbstractContinuityPolicy end

struct _AxisContinuity{D} <: _AbstractContinuityPolicy
  kinds::NTuple{D,Symbol}

  function _AxisContinuity{D}(kinds::NTuple{D,Symbol}) where {D}
    all(kind -> kind === :cg || kind === :dg, kinds) ||
      throw(ArgumentError("continuity policy axes must be either :cg or :dg"))
    return new{D}(kinds)
  end
end

@inline _is_cg_axis(policy::_AxisContinuity, axis::Int) = policy.kinds[axis] === :cg
@inline _is_fully_dg(policy::_AxisContinuity) = all(kind -> kind === :dg, policy.kinds)

# A face overlap patch represented on the common finest logical lattice used by
# `_SpaceBuildState`. Along the face-normal axis, `lower == upper` stores the
# shared face coordinate; along tangential axes, `[lower, upper)` is the overlap
# interval on which two traces must agree.
struct _FacePiece{D}
  lower::NTuple{D,Int}
  upper::NTuple{D,Int}
end

# Sparse linear row over provisional boundary variables or finalized global
# scalar dofs. The representation is intentionally simple because rows are small,
# heavily mutated during substitution, and repeatedly cleaned of near-zero terms.
mutable struct _ConstraintRow{T<:AbstractFloat}
  variables::Vector{Int}
  coefficients::Vector{T}
end

# One provisional boundary variable together with the local tensor-product mode
# whose face trace it represents.
struct _BoundaryVariable{D}
  variable::Int
  mode::NTuple{D,Int}
end

# Cached mode pattern for one tuple of cell degrees. Many leaves share the same
# degree tuple, so the local-mode enumeration and boundary-mode extraction are
# cached once and reused.
struct _LeafPattern{D}
  support_shape::NTuple{D,Int}
  local_modes::Vector{NTuple{D,Int}}
  boundary_modes::Vector{NTuple{D,Int}}
end

# Fully compiled local basis data for one active leaf. The `term_*` arrays
# encode each active local mode as a sparse expansion in global scalar dofs, and
# `single_term_*` stores a fast path for the common case where a mode maps to
# only one global dof.
struct _CompiledLeaf{D,T<:AbstractFloat}
  leaf::Int
  degrees::NTuple{D,Int}
  support_shape::NTuple{D,Int}
  local_modes::Vector{NTuple{D,Int}}
  mode_lookup::Vector{Int}
  term_offsets::Vector{Int}
  term_indices::Vector{Int}
  term_coefficients::Vector{T}
  single_term_indices::Vector{Int}
  single_term_coefficients::Vector{T}
  quadrature_shape::NTuple{D,Int}
end

# Global build state shared across all phases of continuity compilation. In
# particular:
# - `leaf_lower`/`leaf_upper` store every active leaf interval on a common
#   finest dyadic lattice, so geometric comparisons are exact integer
#   operations.
# - `boundary_lookup` maps a boundary local mode to its provisional variable id.
# - `face_boundary_modes` gives face-local access to the subset of boundary
#   variables that can appear on one specific leaf face.
# - `boundary_terms` eventually stores the finalized expansion of every
#   provisional boundary variable in terms of global scalar dofs.
mutable struct _SpaceBuildState{D,T<:AbstractFloat,B}
  grid::CartesianGrid{D}
  basis::B
  active_leaves::Vector{Int}
  leaf_to_index::Vector{Int}
  leaf_degrees::Vector{NTuple{D,Int}}
  leaf_patterns::Vector{_LeafPattern{D}}
  leaf_lower::Vector{NTuple{D,Int}}
  leaf_upper::Vector{NTuple{D,Int}}
  finest_levels::NTuple{D,Int}
  boundary_lookup::Vector{Dict{NTuple{D,Int},Int}}
  face_boundary_modes::Vector{Vector{_BoundaryVariable{D}}}
  boundary_terms::Vector{Vector{Pair{Int,T}}}
  boundary_var_count::Int
  next_global_dof::Int
end

# Phase 1. Freeze the active leaves and build reusable local-pattern metadata.

# Build the static geometric and combinatorial data needed for continuity
# compilation. The active leaves are frozen, leaf intervals are lifted to the
# common finest logical lattice, and repeated degree patterns are cached so the
# later compilation phases can work with integer interval arithmetic and shared
# local mode tables.
function _build_space_state(domain::Domain{D,T}, basis::B,
                            leaf_degrees::Vector{NTuple{D,Int}}) where {D,T<:AbstractFloat,B}
  grid_data = grid(domain)
  active = active_leaves(grid_data)
  length(leaf_degrees) == length(active) ||
    throw(ArgumentError("leaf degree data must match the active-leaf count"))
  finest_levels = ntuple(axis -> maximum(level(grid_data, leaf, axis) for leaf in active; init=0),
                         D)
  leaf_to_index = zeros(Int, stored_cell_count(grid_data))
  leaf_patterns = Vector{_LeafPattern{D}}(undef, length(active))
  leaf_lower = Vector{NTuple{D,Int}}(undef, length(active))
  leaf_upper = Vector{NTuple{D,Int}}(undef, length(active))
  pattern_cache = Dict{NTuple{D,Int},_LeafPattern{D}}()

  for leaf_index in eachindex(active)
    leaf = active[leaf_index]
    leaf_to_index[leaf] = leaf_index
    degrees = leaf_degrees[leaf_index]
    leaf_patterns[leaf_index] = get!(pattern_cache, degrees) do
      _leaf_pattern(basis, degrees)
    end
    # Store every active leaf as a half-open interval on the finest logical
    # lattice. This makes overlap tests and relative-position calculations exact
    # even when neighboring leaves live on different refinement levels.
    leaf_lower[leaf_index] = ntuple(axis -> logical_coordinate(grid_data, leaf, axis) <<
                                            (finest_levels[axis] - level(grid_data, leaf, axis)), D)
    leaf_upper[leaf_index] = ntuple(axis -> (logical_coordinate(grid_data, leaf, axis) + 1) <<
                                            (finest_levels[axis] - level(grid_data, leaf, axis)), D)
  end

  boundary_lookup = [Dict{NTuple{D,Int},Int}() for _ in eachindex(active)]
  face_boundary_modes = [Vector{_BoundaryVariable{D}}() for _ in 1:(length(active)*D*2)]
  return _SpaceBuildState(grid_data, basis, active, leaf_to_index, leaf_degrees, leaf_patterns,
                          leaf_lower, leaf_upper, finest_levels, boundary_lookup,
                          face_boundary_modes, Vector{Vector{Pair{Int,T}}}(), 0, 1)
end

# Phase 2. Enumerate the trace modes that actually participate in coupling.
#
# The cached leaf patterns from `_build_space_state` are continuity-agnostic.
# This stage decides whether any of those local traces should enter a global
# face algebra at all. Fully DG spaces deliberately skip that work, while any
# policy with at least one CG axis introduces provisional variables only for the
# local modes that actually participate in CG trace coupling.
function _enumerate_space_modes(domain::Domain{D,T}, basis::B, leaf_degrees::Vector{NTuple{D,Int}},
                                continuity_policy::_AxisContinuity{D}) where {D,T<:AbstractFloat,B}
  state = _build_space_state(domain, basis, leaf_degrees)
  _enumerate_trace_modes!(state, continuity_policy)
  return state
end

function _enumerate_trace_modes!(state::_SpaceBuildState{D,T},
                                 continuity_policy::_AxisContinuity{D}) where {D,T<:AbstractFloat}
  if _is_fully_dg(continuity_policy)
    nothing
  else
    _build_boundary_variables!(state, continuity_policy)
  end
  return state
end

# Phase 3. Introduce provisional variables for all local boundary traces.

# Introduce one provisional variable for every local mode that really enters the
# CG trace algebra. Modes that only touch DG axes keep their own independent
# cell-local dof and therefore never need a provisional boundary representative.
function _build_boundary_variables!(state::_SpaceBuildState{D,T},
                                    continuity_policy::_AxisContinuity{D}) where {D,
                                                                                  T<:AbstractFloat}
  next_variable = 1

  for leaf_index in eachindex(state.active_leaves)
    lookup = state.boundary_lookup[leaf_index]

    for mode in state.leaf_patterns[leaf_index].boundary_modes
      _mode_requires_trace_coupling(continuity_policy, mode) || continue
      variable = next_variable
      lookup[mode] = variable
      for axis in 1:D
        # In the integrated Legendre basis, index 0 is the lower endpoint mode
        # and index 1 is the upper endpoint mode. Modes with index ≥ 2 vanish at
        # both endpoints and therefore do not contribute to that face trace.
        side_mode = mode[axis]
        _is_cg_axis(continuity_policy, axis) && side_mode <= 1 || continue
        push!(_face_boundary_modes(state, leaf_index, axis, side_mode + LOWER),
              _BoundaryVariable(variable, mode))
      end
      next_variable += 1
    end
  end

  state.boundary_var_count = next_variable - 1
  state.boundary_terms = [Pair{Int,T}[] for _ in 1:state.boundary_var_count]
  return state
end

# Phase 6. Compile finalized per-leaf local-to-global expansions.
#
# After the trace algebra has either been resolved (CG) or intentionally skipped
# (DG), each active leaf is converted into the sparse runtime format used by
# evaluation and assembly.

# Compile one active leaf after the boundary constraint system has been reduced
# to explicit trace expansions. Purely interior modes receive fresh global dofs
# immediately, while boundary-carrying modes inherit the already computed
# expansion of their provisional boundary variables.
function _compile_leaf(state::_SpaceBuildState{D,T}, leaf_index::Int,
                       continuity_policy::_AxisContinuity{D},
                       quadrature_policy) where {D,T<:AbstractFloat}
  leaf = state.active_leaves[leaf_index]
  degrees = state.leaf_degrees[leaf_index]
  pattern = state.leaf_patterns[leaf_index]
  support_shape = pattern.support_shape
  row_terms = Dict{NTuple{D,Int},Dict{Int,T}}()

  for mode in pattern.local_modes
    # Modes whose endpoint content lies only on DG axes are independent local
    # unknowns. Only modes that touch at least one CG axis participate in the
    # trace-coupling algebra.
    coefficients = _mode_coefficients(state, leaf_index, continuity_policy, mode, T)
    isempty(coefficients) || (row_terms[mode] = coefficients)
  end

  local_modes_data = NTuple{D,Int}[]
  term_offsets = Int[1]
  term_indices = Int[]
  term_coefficients = T[]

  for mode in pattern.local_modes
    # Compress the dictionary representation into the sorted sparse arrays used
    # at runtime by evaluation and assembly.
    coefficients = get(row_terms, mode, nothing)
    coefficients === nothing && continue
    roots = sort!(collect(keys(coefficients)))
    isempty(roots) && continue
    push!(local_modes_data, mode)

    for root in roots
      coefficient = coefficients[root]
      abs(coefficient) <= _constraint_tolerance(T) && continue
      push!(term_indices, root)
      push!(term_coefficients, coefficient)
    end

    term_offsets[end] <= length(term_indices) + 1 ||
      throw(ArgumentError("invalid local term ordering"))
    push!(term_offsets, length(term_indices) + 1)
  end

  mode_lookup = zeros(Int, prod(support_shape))

  for mode_index in eachindex(local_modes_data)
    mode_lookup[_flatten_mode(local_modes_data[mode_index], support_shape)] = mode_index
  end

  quadrature_shape = _quadrature_shape(quadrature_policy, degrees)
  single_term_indices, single_term_coefficients = _single_term_metadata(term_offsets, term_indices,
                                                                        term_coefficients)
  return _CompiledLeaf(leaf, degrees, support_shape, local_modes_data, mode_lookup, term_offsets,
                       term_indices, term_coefficients, single_term_indices,
                       single_term_coefficients, quadrature_shape)
end

# Finalize every active leaf after the continuity algebra has been settled. The
# output is the runtime representation used by `HpSpace`: one compiled leaf
# record per active cell plus the componentwise maximum quadrature shape needed
# anywhere in the space.
function _finalize_compiled_leaves(state::_SpaceBuildState{D,T},
                                   continuity_policy::_AxisContinuity{D},
                                   quadrature_policy) where {D,T<:AbstractFloat}
  compiled = Vector{_CompiledLeaf{D,T}}(undef, length(state.active_leaves))
  global_quadrature_shape = ntuple(_ -> 1, D)

  for leaf_index in eachindex(state.active_leaves)
    compiled_leaf = _compile_leaf(state, leaf_index, continuity_policy, quadrature_policy)
    compiled[leaf_index] = compiled_leaf
    global_quadrature_shape = _merge_quadrature_shapes(global_quadrature_shape,
                                                       compiled_leaf.quadrature_shape)
  end

  return compiled, global_quadrature_shape
end

# Phases 4 and 5. Build and eliminate the global boundary-trace constraint
# system.
#
# Matching faces contribute simple identifications of provisional variables.
# Hanging faces contribute linear rows. After those two ingredients have been
# assembled, the remaining work is sparse elimination on the boundary-variable
# graph. The conceptual outcome is a basis change from provisional face-local
# trace variables to a smaller set of genuine global boundary dofs.

# Compile the complete trace continuity system on the CG axes of the requested
# policy. DG-axis interfaces are intentionally skipped here because their traces
# remain independent leaf-local unknowns.
function _compile_boundary_terms!(state::_SpaceBuildState{D,T},
                                  continuity_policy::_AxisContinuity{D}) where {D,T<:AbstractFloat}
  parent = collect(1:state.boundary_var_count)
  zeroed = falses(state.boundary_var_count)
  face_pairs = _internal_face_pairs(state, continuity_policy)

  # First collapse all matching-face identities and detect trace variables that
  # cannot survive because their tangential content is not representable on the
  # common face space.
  _merge_matching_faces!(parent, zeroed, state, face_pairs)
  roots = _boundary_roots!(parent)
  zeroed_roots = _zeroed_roots(roots, zeroed)

  # Then assemble explicit linear rows only for genuinely hanging interfaces,
  # and append one-variable zero rows for trace representatives that were marked
  # incompatible in the merge pass.
  rows = _build_hanging_face_rows!(state, roots, face_pairs)
  _append_zeroed_root_rows!(rows, roots, zeroed_roots, T)

  # The remaining algebra is solved component-wise on the variable graph. This
  # yields pivot expressions for constrained representatives and identifies the
  # root representatives that survive as free trace dofs.
  component_rows = _constraint_components(rows, state.boundary_var_count)
  components = findall(!isempty, component_rows)
  pivots, component_orders = _eliminate_boundary_components!(component_rows, components,
                                                             state.boundary_var_count, T)
  free_dofs = _assign_free_boundary_dofs!(state, roots, zeroed_roots, pivots)

  # Finally convert every provisional boundary variable into its explicit sparse
  # expansion in global scalar dofs.
  _initialize_root_boundary_terms!(state.boundary_terms, roots, zeroed_roots, pivots, free_dofs, T)
  _finalize_boundary_components!(state.boundary_terms, pivots, component_orders, free_dofs, T)
  _propagate_root_boundary_terms!(state.boundary_terms, roots)
  return state
end

# Dispatch the face-coupling phase according to the requested continuity policy.
# Fully DG spaces skip the boundary-trace algebra, while all other policies run
# it only on interfaces normal to CG axes.
function _compile_face_coupling!(state::_SpaceBuildState{D,T},
                                 continuity_policy::_AxisContinuity{D}) where {D,T<:AbstractFloat}
  if _is_fully_dg(continuity_policy)
    nothing
  else
    _compile_boundary_terms!(state, continuity_policy)
  end
  return state
end

# Enumerate every active interface patch exactly once, using active-leaf indices
# rather than raw cell ids so later arrays can be addressed directly. Only
# interfaces normal to CG axes participate in the continuity algebra.
function _internal_face_pairs(state::_SpaceBuildState{D},
                              continuity_policy::_AxisContinuity{D}) where {D}
  pairs = Tuple{Int,Int,Int,Int}[]

  for (leaf, axis, other) in _upper_face_neighbor_specs(state.grid)
    _is_cg_axis(continuity_policy, axis) || continue
    push!(pairs,
          (@inbounds(state.leaf_to_index[leaf]), @inbounds(state.leaf_to_index[other]), axis,
           UPPER))
  end

  return pairs
end

# Build and cache the active local-mode set for one degree tuple, then extract
# the subset that can carry nonzero trace data on at least one boundary face.
# This classification depends only on the basis and the degree tuple, not yet on
# the later continuity policy.
function _leaf_pattern(basis::AbstractBasisFamily, degrees::NTuple{D,Int}) where {D}
  support_shape = ntuple(axis -> degrees[axis] + 1, D)
  local_modes = collect(basis_modes(basis, degrees))
  boundary_modes = NTuple{D,Int}[]

  for mode_index in eachindex(local_modes)
    mode = local_modes[mode_index]
    _is_boundary_mode(mode) && push!(boundary_modes, mode)
  end

  return _LeafPattern(support_shape, local_modes, boundary_modes)
end

# Matching-face merge and hanging-face row assembly.

# Process all interfaces that geometrically coincide face-to-face. Compatible
# trace variables are merged by union-find, while tangential modes that are not
# representable in the common face space are marked as zero.
function _merge_matching_faces!(parent::Vector{Int}, zeroed::BitVector, state::_SpaceBuildState{D},
                                face_pairs) where {D}
  for (first_leaf_index, second_leaf_index, axis, first_side) in face_pairs
    second_side, common_degrees, matching = _face_pair_metadata(state, first_leaf_index,
                                                                second_leaf_index, axis, first_side)
    _mark_incompatible_face_modes!(zeroed, state, first_leaf_index, axis, first_side,
                                   common_degrees)
    _mark_incompatible_face_modes!(zeroed, state, second_leaf_index, axis, second_side,
                                   common_degrees)
    matching || continue
    _merge_matching_face!(parent, state, first_leaf_index, second_leaf_index, axis, first_side,
                          second_side, common_degrees)
  end

  return parent
end

# Build continuity rows for all nonmatching hanging interfaces. Each row states
# that one trace mode in the common overlap space has identical coefficients
# when expanded from either side.
function _build_hanging_face_rows!(state::_SpaceBuildState{D,T}, roots::Vector{Int},
                                   face_pairs) where {D,T<:AbstractFloat}
  isempty(face_pairs) && return _ConstraintRow{T}[]
  restriction_cache = Dict{NTuple{4,Int},Matrix{T}}()
  rows = _ConstraintRow{T}[]

  for (first_leaf_index, second_leaf_index, axis, first_side) in face_pairs
    second_side, common_degrees, matching = _face_pair_metadata(state, first_leaf_index,
                                                                second_leaf_index, axis, first_side)
    matching && continue
    _add_hanging_face_rows!(rows, restriction_cache, roots, state, first_leaf_index,
                            second_leaf_index, axis, first_side, second_side, common_degrees)
  end

  return rows
end

# Assemble all continuity rows contributed by one nonmatching face pair. The two
# traces are both restricted to the same overlap patch and then tested against
# the basis of the common face space.
function _add_hanging_face_rows!(rows::Vector{_ConstraintRow{T}},
                                 restriction_cache::Dict{NTuple{4,Int},Matrix{T}},
                                 roots::Vector{Int}, state::_SpaceBuildState{D,T},
                                 first_leaf_index::Int, second_leaf_index::Int, axis::Int,
                                 first_side::Int, second_side::Int,
                                 common_degrees::NTuple{D,Int}) where {D,T<:AbstractFloat}
  piece = _face_piece(state, first_leaf_index, second_leaf_index, axis, first_side)

  # Each basis mode of the common face space contributes one scalar continuity
  # equation equating the restricted coefficient seen from the two sides.
  for common_mode in basis_modes(state.basis, common_degrees)
    row = _ConstraintRow{T}(Int[], T[])
    _accumulate_face_row!(row, restriction_cache, roots, state, first_leaf_index, axis, first_side,
                          piece, common_degrees, common_mode, one(T))
    _accumulate_face_row!(row, restriction_cache, roots, state, second_leaf_index, axis,
                          second_side, piece, common_degrees, common_mode, -one(T))
    _cleanup_boundary_row!(row, T)
    _row_isempty(row) || push!(rows, row)
  end

  return rows
end

# On a geometrically matching face, continuity is enforced by simple variable
# identification: the same trace mode on both sides is the same global unknown.
function _merge_matching_face!(parent::Vector{Int}, state::_SpaceBuildState{D},
                               first_leaf_index::Int, second_leaf_index::Int, axis::Int,
                               first_side::Int, second_side::Int,
                               common_degrees::NTuple{D,Int}) where {D}
  first_lookup = state.boundary_lookup[first_leaf_index]
  second_lookup = state.boundary_lookup[second_leaf_index]
  first_side_mode = _side_mode(first_side)
  second_side_mode = _side_mode(second_side)

  for common_mode in basis_modes(state.basis, common_degrees)
    first_mode = ntuple(current_axis -> current_axis == axis ? first_side_mode :
                                        common_mode[current_axis], D)
    second_mode = ntuple(current_axis -> current_axis == axis ? second_side_mode :
                                         common_mode[current_axis], D)
    first_variable = get(first_lookup, first_mode, 0)
    second_variable = get(second_lookup, second_mode, 0)
    (first_variable == 0 || second_variable == 0) && continue
    _union_boundary_variables!(parent, first_variable, second_variable)
  end

  return parent
end

# Any local face mode whose tangential content lies outside the common face
# space cannot participate in a continuous trace across that interface, so its
# coefficient must vanish there.
function _mark_incompatible_face_modes!(zeroed::BitVector, state::_SpaceBuildState{D},
                                        leaf_index::Int, axis::Int, side::Int,
                                        common_degrees::NTuple{D,Int}) where {D}
  for boundary_variable in _face_boundary_modes(state, leaf_index, axis, side)
    _compatible_face_mode(state.basis, common_degrees, axis, boundary_variable.mode) === nothing ||
      continue
    zeroed[boundary_variable.variable] = true
  end

  return zeroed
end

# Add one side of one hanging-face continuity equation to a sparse row. The
# tensor-product trace coefficient factorizes into one-dimensional restriction
# coefficients on the tangential axes, so the total coefficient is their
# product.
function _accumulate_face_row!(row::_ConstraintRow{T},
                               restriction_cache::Dict{NTuple{4,Int},Matrix{T}}, roots::Vector{Int},
                               state::_SpaceBuildState{D,T}, leaf_index::Int, face_axis::Int,
                               side::Int, piece::_FacePiece{D}, common_degrees::NTuple{D,Int},
                               common_mode::NTuple{D,Int}, sign::T) where {D,T<:AbstractFloat}
  for boundary_variable in _face_boundary_modes(state, leaf_index, face_axis, side)
    mode = boundary_variable.mode
    _compatible_face_mode(state.basis, common_degrees, face_axis, mode) === nothing && continue
    coefficient = _face_restriction_coefficient(restriction_cache, state, leaf_index, face_axis,
                                                piece, mode, common_mode, common_degrees)
    abs(coefficient) <= _constraint_tolerance(T) && continue
    _row_add!(row, roots[boundary_variable.variable], sign * coefficient)
  end

  return row
end

# Recover the already finalized expansion of one provisional boundary variable.
# A copy is returned because the caller may combine or filter coefficients when
# assembling the local sparse representation of a compiled leaf.
function _boundary_mode_terms(state::_SpaceBuildState{D,T}, leaf_index::Int,
                              mode::NTuple{D,Int}) where {D,T<:AbstractFloat}
  variable = get(state.boundary_lookup[leaf_index], mode, 0)
  variable != 0 || throw(ArgumentError("missing boundary variable for local mode"))
  terms = Dict{Int,T}()

  for term in state.boundary_terms[variable]
    terms[term.first] = term.second
  end

  return terms
end

# Decide whether one local tensor-product mode really needs the boundary-trace
# algebra. Only modes that touch at least one CG axis through an endpoint factor
# participate in global continuity; every other mode is leaf-local and can
# immediately receive its own scalar dof.
@inline function _mode_requires_trace_coupling(continuity_policy::_AxisContinuity{D},
                                               mode::NTuple{D,<:Integer}) where {D}
  for axis in 1:D
    _is_cg_axis(continuity_policy, axis) && mode[axis] <= 1 && return true
  end

  return false
end

@inline function _independent_mode_terms(state::_SpaceBuildState{D,T}, ::NTuple{D,<:Integer},
                                         ::Type{T}) where {D,T<:AbstractFloat}
  dof = state.next_global_dof
  state.next_global_dof += 1
  return Dict{Int,T}(dof => one(T))
end

# Resolve one local mode either through the compiled boundary-variable expansion
# or, if it never participates in a CG trace, by assigning it a fresh
# independent scalar dof directly.
function _mode_coefficients(state::_SpaceBuildState{D,T}, leaf_index::Int,
                            continuity_policy::_AxisContinuity{D}, mode::NTuple{D,Int},
                            ::Type{T}) where {D,T<:AbstractFloat}
  _mode_requires_trace_coupling(continuity_policy, mode) ||
    return _independent_mode_terms(state, mode, T)
  return _boundary_mode_terms(state, leaf_index, mode)
end

# Compute the exact logical overlap patch of two adjacent faces on the common
# finest dyadic lattice. This patch is the geometric support on which hanging
# continuity has to be enforced.
function _face_piece(state::_SpaceBuildState{D}, first_leaf_index::Int, second_leaf_index::Int,
                     axis::Int, side::Int) where {D}
  face_coordinate = side == LOWER ? state.leaf_lower[first_leaf_index][axis] :
                    state.leaf_upper[first_leaf_index][axis]
  lower = ntuple(current_axis -> current_axis == axis ? face_coordinate :
                                 max(state.leaf_lower[first_leaf_index][current_axis],
                                     state.leaf_lower[second_leaf_index][current_axis]), D)
  upper = ntuple(current_axis -> current_axis == axis ? face_coordinate :
                                 min(state.leaf_upper[first_leaf_index][current_axis],
                                     state.leaf_upper[second_leaf_index][current_axis]), D)

  for current_axis in 1:D
    current_axis == axis && continue
    lower[current_axis] < upper[current_axis] ||
      throw(ArgumentError("adjacent leaves do not share a positive-measure face overlap"))
  end

  return _FacePiece(lower, upper)
end

# Compute the tensor-product coefficient with which one local trace mode
# contributes to one common face mode on the overlap patch. Every tangential
# direction contributes one scalar restriction factor, so the full coefficient is
# obtained by multiplication.
function _face_restriction_coefficient(restriction_cache::Dict{NTuple{4,Int},Matrix{T}},
                                       state::_SpaceBuildState{D,T}, leaf_index::Int,
                                       face_axis::Int, piece::_FacePiece{D}, mode::NTuple{D,Int},
                                       common_mode::NTuple{D,Int},
                                       common_degrees::NTuple{D,Int}) where {D,T<:AbstractFloat}
  coefficient = one(T)

  for axis in 1:D
    axis == face_axis && continue
    column = _dyadic_restriction_column!(restriction_cache, state,
                                         state.leaf_lower[leaf_index][axis],
                                         state.leaf_upper[leaf_index][axis], piece.lower[axis],
                                         piece.upper[axis], mode[axis], common_degrees[axis], axis)
    coefficient *= column[common_mode[axis]+1]
    abs(coefficient) <= _constraint_tolerance(T) && return zero(T)
  end

  return coefficient
end

# Determine whether a boundary mode can appear in the common face space and, if
# so, return the corresponding tangential mode tuple. The face-normal component
# is replaced by `0` as a placeholder because the common trace space lives only
# in the tangential directions.
function _compatible_face_mode(basis::AbstractBasisFamily, common_degrees::NTuple{D,Int},
                               face_axis::Int, mode::NTuple{D,Int}) where {D}
  for axis in 1:D
    axis == face_axis && continue
    mode[axis] <= common_degrees[axis] || return nothing
  end

  common_mode = ntuple(axis -> axis == face_axis ? 0 : mode[axis], D)
  return is_active_mode(basis, common_degrees, common_mode) ? common_mode : nothing
end

# Shared metadata for one active face pair. Both the matching-face merge pass
# and the hanging-face row builder need the opposite side, common face degrees,
# and the matching-vs-hanging classification.
function _face_pair_metadata(state::_SpaceBuildState{D}, first_leaf_index::Int,
                             second_leaf_index::Int, axis::Int, first_side::Int) where {D}
  second_side = _opposite_side(first_side)
  common_degrees = _common_face_degrees(state, first_leaf_index, second_leaf_index, axis)
  matching = _matching_face(state, first_leaf_index, second_leaf_index, axis)
  return second_side, common_degrees, matching
end

# Sparse row algebra for boundary constraints.

@inline _pivot_expression(pivots::Dict{Int,_ConstraintRow{T}}, variable::Int) where {T<:AbstractFloat} = get(pivots,
                                                                                                             variable,
                                                                                                             nothing)
@inline _pivot_expression(pivots::AbstractVector{<:Union{Nothing,_ConstraintRow{T}}}, variable::Int) where {T<:AbstractFloat} = pivots[variable]

# Expand one pivot expression into `row` with the given coefficient factor.
function _accumulate_scaled_row!(row::_ConstraintRow{T}, expression::_ConstraintRow{T},
                                 factor::T) where {T<:AbstractFloat}
  return _accumulate_row!(row, expression, factor)
end

# Replace the first pivot variable still present in `row` by its stored
# expression. Returns `true` when one substitution was performed.
function _substitute_boundary_step!(row::_ConstraintRow{T}, pivots) where {T<:AbstractFloat}
  for index in eachindex(row.variables)
    expression = _pivot_expression(pivots, row.variables[index])
    expression === nothing && continue
    factor = row.coefficients[index]
    _remove_variable_at!(row, index)
    _accumulate_scaled_row!(row, expression, factor)
    return true
  end

  return false
end

# Substitute already eliminated pivot variables from `row` until only free
# variables remain. This is a sparse symbolic/numeric elimination step on small
# constraint rows.
function _substitute_boundary_row!(row::_ConstraintRow{T}, pivots,
                                   ::Type{T}) where {T<:AbstractFloat}
  while true
    changed = _substitute_boundary_step!(row, pivots)
    _cleanup_boundary_row!(row, T)
    changed || return row
  end
end

# Remove coefficients below the numerical tolerance. This keeps rows compact and
# prevents the elimination phase from carrying along numerical noise that should
# be interpreted as exact cancellation.
function _cleanup_boundary_row!(row::_ConstraintRow{T}, ::Type{T}) where {T<:AbstractFloat}
  tolerance = _constraint_tolerance(T)
  index = 1

  while index <= length(row.variables)
    if abs(row.coefficients[index]) <= tolerance
      _remove_variable_at!(row, index)
    else
      index += 1
    end
  end

  return row
end

# Expand one eliminated pivot expression all the way to finalized global scalar
# dofs by recursively substituting already finalized boundary terms.
function _accumulate_row!(row::_ConstraintRow{T}, source::_ConstraintRow{T},
                          factor::T=one(T)) where {T<:AbstractFloat}
  for index in eachindex(source.variables)
    _row_add!(row, source.variables[index], factor * source.coefficients[index])
  end

  return row
end

function _accumulate_pairs!(row::_ConstraintRow{T}, pairs::AbstractVector{<:Pair{Int,T}},
                            factor::T=one(T)) where {T<:AbstractFloat}
  for term in pairs
    _row_add!(row, term.first, factor * term.second)
  end

  return row
end

function _sorted_row_pairs(row::_ConstraintRow{T}) where {T<:AbstractFloat}
  pairs = Pair{Int,T}[row.variables[index] => row.coefficients[index]
                      for index in eachindex(row.variables)]
  sort!(pairs; by=first)
  return pairs
end

function _finalize_boundary_terms!(expression::_ConstraintRow{T},
                                   boundary_terms::Vector{Vector{Pair{Int,T}}},
                                   free_dofs::Vector{Int}, ::Type{T}) where {T<:AbstractFloat}
  coefficients = _ConstraintRow{T}(Int[], T[])

  for index in eachindex(expression.variables)
    variable = expression.variables[index]
    factor = expression.coefficients[index]
    abs(factor) <= _constraint_tolerance(T) && continue
    terms = boundary_terms[variable]

    # An empty term list means the variable was constrained to zero. Otherwise
    # the already finalized term list is recursively accumulated into the current
    # expression.
    if isempty(terms)
      free_dofs[variable] == 0 ||
        throw(ArgumentError("boundary term expansion expected variable $variable to be constrained"))
      continue
    end

    _accumulate_pairs!(coefficients, terms, factor)
  end

  _cleanup_boundary_row!(coefficients, T)
  return _sorted_row_pairs(coefficients)
end

# Shared face-space metadata.

# The common face space keeps the face-normal direction fixed and uses the
# minimum tangential degree seen on the two sides, because only that tangential
# polynomial content can be represented exactly on both traces.
function _common_face_degrees(state::_SpaceBuildState{D}, first_leaf_index::Int,
                              second_leaf_index::Int, axis::Int) where {D}
  return ntuple(current_axis -> current_axis == axis ? 0 :
                                min(state.leaf_degrees[first_leaf_index][current_axis],
                                    state.leaf_degrees[second_leaf_index][current_axis]), D)
end

# A face pair is geometrically matching if its tangential intervals coincide
# exactly. Such faces can be merged directly; otherwise they require hanging-face
# restriction rows on the overlap patch.
function _matching_face(state::_SpaceBuildState{D}, first_leaf_index::Int, second_leaf_index::Int,
                        axis::Int) where {D}
  for current_axis in 1:D
    current_axis == axis && continue
    state.leaf_lower[first_leaf_index][current_axis] ==
    state.leaf_lower[second_leaf_index][current_axis] || return false
    state.leaf_upper[first_leaf_index][current_axis] ==
    state.leaf_upper[second_leaf_index][current_axis] || return false
  end

  return true
end

# Component-wise elimination and finalization of the boundary graph.

# Partition the constraint rows into connected components of the variable graph.
# Different components can be eliminated independently and, in this file, in
# parallel.
function _constraint_components(rows::Vector{_ConstraintRow{T}},
                                variable_count::Int) where {T<:AbstractFloat}
  parent = collect(1:variable_count)

  for row in rows
    length(row.variables) <= 1 && continue
    first_variable = row.variables[1]

    for index in 2:length(row.variables)
      _union_boundary_variables!(parent, first_variable, row.variables[index])
    end
  end

  components = [_ConstraintRow{T}[] for _ in 1:variable_count]

  for row in rows
    _row_isempty(row) && continue
    component = _boundary_root!(parent, row.variables[1])
    push!(components[component], row)
  end

  return components
end

# Eliminate each connected component of the boundary-variable constraint graph.
# Every surviving row chooses one pivot variable and rewrites it as an affine
# combination of the remaining variables in that component. The pivot order is
# stored so the final expansion step can later process the dependencies in
# reverse topological order.
function _eliminate_boundary_components!(component_rows::Vector{Vector{_ConstraintRow{T}}},
                                         components::Vector{Int}, variable_count::Int,
                                         ::Type{T}) where {T<:AbstractFloat}
  component_orders = Vector{Vector{Int}}(undef, length(components))
  component_pivots = Vector{Vector{Pair{Int,_ConstraintRow{T}}}}(undef, length(components))

  Threads.@threads for component_index in eachindex(components)
    local_pivots = Dict{Int,_ConstraintRow{T}}()
    pivot_order = Int[]

    for row in component_rows[components[component_index]]
      _substitute_boundary_row!(row, local_pivots, T)
      _row_isempty(row) && continue
      # Choosing the largest variable id as pivot gives a deterministic
      # elimination order and leaves lower-numbered representatives free when
      # possible.
      pivot, pivot_coefficient = _pivot_variable(row)
      _remove_variable!(row, pivot)
      scale = -inv(pivot_coefficient)
      expression = _ConstraintRow{T}(Int[], T[])

      # After removing the pivot term, the remaining row represents
      #
      #   pivot_coefficient * x_pivot + Σ aⱼ xⱼ = 0.
      #
      # Multiplying by `-1 / pivot_coefficient` rewrites this as an explicit
      # affine expression for the pivot variable in terms of the remaining
      # variables.
      _accumulate_row!(expression, row, scale)
      _cleanup_boundary_row!(expression, T)
      local_pivots[pivot] = expression
      push!(pivot_order, pivot)
    end

    component_orders[component_index] = pivot_order
    component_pivots[component_index] = Pair{Int,_ConstraintRow{T}}[pivot => local_pivots[pivot]
                                                                    for pivot in pivot_order]
  end

  pivots = Vector{Union{Nothing,_ConstraintRow{T}}}(nothing, variable_count)

  for pairs in component_pivots
    for pair in pairs
      pivots[pair.first] = pair.second
    end
  end

  return pivots, component_orders
end

# Finalize every component by expanding pivots in reverse elimination order.
# Reversing the local pivot order guarantees that when one pivot is expanded, all
# variables it depends on already have finalized boundary-term expressions.
function _finalize_boundary_components!(boundary_terms::Vector{Vector{Pair{Int,T}}},
                                        pivots::Vector{Union{Nothing,_ConstraintRow{T}}},
                                        component_orders::Vector{Vector{Int}},
                                        free_dofs::Vector{Int}, ::Type{T}) where {T<:AbstractFloat}
  Threads.@threads for component_index in eachindex(component_orders)
    for pivot in Iterators.reverse(component_orders[component_index])
      boundary_terms[pivot] = _finalize_boundary_terms!(pivots[pivot]::_ConstraintRow{T},
                                                        boundary_terms, free_dofs, T)
    end
  end

  return boundary_terms
end

# Root initialization and propagation after elimination.

# Variables that are incompatible with the common face space are constrained to
# zero by appending explicit one-variable rows for the surviving root
# representatives.
function _append_zeroed_root_rows!(rows::Vector{_ConstraintRow{T}}, roots::Vector{Int},
                                   zeroed_roots::BitVector, ::Type{T}) where {T<:AbstractFloat}
  for variable in eachindex(roots)
    _is_root_variable(roots, variable) || continue
    zeroed_roots[variable] || continue
    push!(rows, _ConstraintRow{T}(Int[variable], T[one(T)]))
  end

  return rows
end

# Every root variable that survives merging, is not zeroed, and is not chosen
# as an elimination pivot becomes an independent global scalar dof.
function _assign_free_boundary_dofs!(state::_SpaceBuildState, roots::Vector{Int},
                                     zeroed_roots::BitVector,
                                     pivots::AbstractVector{<:Union{Nothing,_ConstraintRow}})
  free_dofs = zeros(Int, state.boundary_var_count)

  for variable in 1:state.boundary_var_count
    _is_free_boundary_root(roots, zeroed_roots, pivots, variable) || continue
    free_dofs[variable] = state.next_global_dof
    state.next_global_dof += 1
  end

  return free_dofs
end

# Seed root representatives with either their free-dof identity expression or an
# empty constrained expression. Non-root variables are filled later by copying
# their finalized root expansion.
function _initialize_root_boundary_terms!(boundary_terms::Vector{Vector{Pair{Int,T}}},
                                          roots::Vector{Int}, zeroed_roots::BitVector,
                                          pivots::AbstractVector{<:Union{Nothing,_ConstraintRow{T}}},
                                          free_dofs::Vector{Int},
                                          ::Type{T}) where {T<:AbstractFloat}
  for variable in eachindex(boundary_terms)
    _is_root_variable(roots, variable) || continue
    boundary_terms[variable] = _initial_root_boundary_terms(zeroed_roots, pivots, free_dofs,
                                                            variable, T)
  end

  return boundary_terms
end

function _initial_root_boundary_terms(zeroed_roots::BitVector,
                                      pivots::AbstractVector{<:Union{Nothing,_ConstraintRow{T}}},
                                      free_dofs::Vector{Int}, variable::Int,
                                      ::Type{T}) where {T<:AbstractFloat}
  if zeroed_roots[variable] || pivots[variable] !== nothing
    return Pair{Int,T}[]
  end

  return Pair{Int,T}[free_dofs[variable] => one(T)]
end

# Once the root representatives have been finalized, all merged variables share
# that same expansion verbatim.
function _propagate_root_boundary_terms!(boundary_terms, roots::Vector{Int})
  for variable in eachindex(boundary_terms)
    root = roots[variable]
    root == variable && continue
    boundary_terms[variable] = boundary_terms[root]
  end

  return boundary_terms
end

# Tiny predicates and sparse-row primitives.

@inline _row_isempty(row::_ConstraintRow) = isempty(row.variables)
@inline _is_root_variable(roots::Vector{Int}, variable::Int) = roots[variable] == variable
@inline function _is_free_boundary_root(roots::Vector{Int}, zeroed_roots::BitVector, pivots,
                                        variable::Int)
  return _is_root_variable(roots, variable) &&
         !zeroed_roots[variable] &&
         pivots[variable] === nothing
end

# Select the pivot variable for one row. The row is not sorted, so the pivot is
# chosen by inspecting the stored variable ids directly.
function _pivot_variable(row::_ConstraintRow)
  pivot_index = argmax(row.variables)
  return row.variables[pivot_index], row.coefficients[pivot_index]
end

# Add one coefficient into a sparse row, merging duplicates in place.
function _row_add!(row::_ConstraintRow{T}, variable::Int, coefficient::T) where {T<:AbstractFloat}
  for index in eachindex(row.variables)
    row.variables[index] == variable || continue
    row.coefficients[index] += coefficient
    return row
  end

  push!(row.variables, variable)
  push!(row.coefficients, coefficient)
  return row
end

# Remove one variable from a sparse row if present.
function _remove_variable!(row::_ConstraintRow, variable::Int)
  for index in eachindex(row.variables)
    row.variables[index] == variable || continue
    _remove_variable_at!(row, index)
    return row
  end

  return row
end

# Delete the `index`-th stored term from the sparse row.
function _remove_variable_at!(row::_ConstraintRow, index::Int)
  deleteat!(row.variables, index)
  deleteat!(row.coefficients, index)
  return row
end

# Union-find root query with path compression.
@inline function _boundary_root!(parent::Vector{Int}, variable::Int)
  root = variable

  while parent[root] != root
    root = parent[root]
  end

  while parent[variable] != variable
    next = parent[variable]
    parent[variable] = root
    variable = next
  end

  return root
end

# Union-find helpers for equivalence classes of matching trace variables.

# Union two boundary-variable equivalence classes and return the chosen root.
# The smaller variable id is kept as representative to make the result
# deterministic across runs.
function _union_boundary_variables!(parent::Vector{Int}, first::Int, second::Int)
  root_first = _boundary_root!(parent, first)
  root_second = _boundary_root!(parent, second)
  root_first == root_second && return root_first

  if root_first < root_second
    parent[root_second] = root_first
    return root_first
  end

  parent[root_first] = root_second
  return root_second
end

# Lift the variable-level zero flags to their union-find representatives.
function _zeroed_roots(roots::Vector{Int}, zeroed::BitVector)
  zeroed_roots = falses(length(zeroed))

  for variable in eachindex(zeroed)
    zeroed[variable] || continue
    zeroed_roots[roots[variable]] = true
  end

  return zeroed_roots
end

# Compute the union-find representative of every boundary variable.
function _boundary_roots!(parent::Vector{Int})
  roots = similar(parent)

  for variable in eachindex(parent)
    roots[variable] = _boundary_root!(parent, variable)
  end

  return roots
end

# Dyadic one-dimensional restriction operators for hanging faces.

# Return one cached restriction column that maps a source one-dimensional trace
# basis function onto the target overlap interval. The cache key is purely
# dyadic: source basis index, target degree, relative refinement level `δ`, and
# child offset inside the source interval.
function _dyadic_restriction_column!(restriction_cache::Dict{NTuple{4,Int},Matrix{T}},
                                     state::_SpaceBuildState{D,T}, source_lower::Int,
                                     source_upper::Int, target_lower::Int, target_upper::Int,
                                     source_index::Int, target_degree::Int,
                                     axis::Int) where {D,T<:AbstractFloat}
  source_index <= target_degree ||
    throw(ArgumentError("target degree must represent the source trace exactly"))
  delta, relative = _relative_interval_position(source_lower, source_upper, target_lower,
                                                target_upper, state.finest_levels[axis])
  matrix = get!(restriction_cache, (source_index, target_degree, delta, relative)) do
    # The cache stores the whole restriction matrix for this dyadic embedding,
    # even though callers typically request only one source column at a time.
    # This pays off because many trace modes on the same interface reuse the
    # same affine subinterval geometry.
    _affine_restriction_matrix(source_index, target_degree, delta, relative, T)
  end
  return view(matrix, :, source_index + 1)
end

# Describe the target interval as a dyadic subinterval of the source interval.
# `delta` is the refinement-level difference and `relative` is the integer
# offset of the target inside the subdivided source interval.
function _relative_interval_position(source_lower::Int, source_upper::Int, target_lower::Int,
                                     target_upper::Int, finest_level::Int)
  source_level, source_coord = _interval_level_and_coord(source_lower, source_upper, finest_level)
  target_level, target_coord = _interval_level_and_coord(target_lower, target_upper, finest_level)
  delta = target_level - source_level
  relative = target_coord - (source_coord << delta)
  return delta, relative
end

# Recover the dyadic level and logical coordinate of one interval represented on
# the common finest lattice. The interval length is always a power of two in
# this code path, so trailing zeros identify the coarser dyadic level exactly.
function _interval_level_and_coord(lower::Int, upper::Int, finest_level::Int)
  length_interval = upper - lower
  level = finest_level - trailing_zeros(length_interval)
  coord = lower >>> trailing_zeros(length_interval)
  return level, coord
end

# Build the one-dimensional affine restriction matrix from a source trace basis
# on one interval to a target basis on a dyadic subinterval.
#
# If `η ∈ [-1,1]` are target reference coordinates and `ξ` are source reference
# coordinates, then the subinterval embedding has the affine form
#
#   ξ = 2^{-δ} η + shift,
#
# where `δ` is the dyadic refinement difference and `shift` encodes the child
# position. The matrix columns are computed by evaluating the restricted source
# basis at `target_degree + 1` Gauss points and solving the square collocation
# system in the target basis. Because both sides lie in the same finite
# polynomial space and the points are distinct, this reconstruction is exact.
function _affine_restriction_matrix(source_degree::Int, target_degree::Int, delta::Int,
                                    relative::Int, ::Type{T}) where {T<:AbstractFloat}
  scale = ldexp(one(T), -delta)
  shift = delta == 0 ? zero(T) : muladd(scale, T(2 * relative + 1), -one(T))
  required_degree = max(1, source_degree)
  target_degree >= required_degree ||
    throw(ArgumentError("target degree must represent the source degree exactly"))

  point_total = target_degree + 1
  rule = gauss_legendre_rule(point_total)
  basis_matrix = Matrix{T}(undef, point_total, point_total)

  for point_index in 1:point_total
    # Build the collocation matrix of the target basis on the target reference
    # points once, then reuse its factorization for all source columns.
    values = integrated_legendre_values(coordinate(rule, point_index, 1), target_degree)

    for basis_index in 1:point_total
      basis_matrix[point_index, basis_index] = values[basis_index]
    end
  end

  factorization = lu(basis_matrix)
  restriction = Matrix{T}(undef, target_degree + 1, source_degree + 1)

  for source_index in 0:source_degree
    rhs = Vector{T}(undef, point_total)

    for point_index in 1:point_total
      η = coordinate(rule, point_index, 1)
      ξ = muladd(scale, η, shift)
      # Evaluate the source basis function after pulling the target point back
      # to source reference coordinates.
      rhs[point_index] = integrated_legendre_values(ξ, source_index)[source_index+1]
    end

    restriction[:, source_index+1] = factorization \ rhs
  end

  return restriction
end

# Small indexing and tolerance helpers.

# Face-local lookup for the provisional boundary variables that can contribute to
# one specific leaf face.
@inline function _face_boundary_modes(state::_SpaceBuildState{D}, leaf_index::Int, axis::Int,
                                      side::Int) where {D}
  return @inbounds state.face_boundary_modes[(((leaf_index-1)*D+(axis-1))<<1)+side]
end

# Numerical tolerance used to interpret cancellation in the continuity algebra.
@inline _constraint_tolerance(::Type{T}) where {T<:AbstractFloat} = 1000 * eps(T)
# A tensor-product mode is a boundary mode if at least one one-dimensional factor
# is an endpoint mode of the integrated Legendre basis.
@inline _is_boundary_mode(mode) = any(index <= 1 for index in mode)
# Convert the face side identifier `LOWER/UPPER` to the corresponding endpoint
# mode index `0/1`.
@inline _side_mode(side::Int) = side - LOWER
# Return the opposite face side on the neighboring cell.
@inline _opposite_side(side::Int) = side == LOWER ? UPPER : LOWER

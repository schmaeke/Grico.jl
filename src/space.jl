# This file builds compiled hp spaces from domains, basis families, degree
# policies, and continuity data.
#
# Earlier files answer three separate questions:
# - `basis.jl`: which local tensor-product modes are admissible for a degree
#   tuple,
# - `continuity.jl`: how local boundary modes on neighboring leaves are either
#   identified/constrained or left independent under the requested continuity
#   policy,
# - `quadrature.jl`: how many integration points are associated with a chosen
#   local degree.
#
# This file is where those ingredients become the user-facing compiled space
# object. In particular, `HpSpace` turns
#
#   domain + basis family + degree policy + quadrature policy + continuity policy
#
# into
#
#   active leaves + compiled local modes + global scalar dofs + local quadrature
#   shapes.
#
# That makes `space.jl` the public façade over the lower-level continuity
# algebra. Readers should think of it in four blocks:
# 1. policy types describing degrees and quadrature sizes,
# 2. the immutable compiled `HpSpace` object,
# 3. public queries into compiled local modes and their global dof expansions,
# 4. internal materialization, compilation, and validation helpers.

# Public policy types.

"""
    AbstractDegreePolicy

Abstract supertype for policies that assign polynomial degrees to active leaves
of an `HpSpace`.

Degree policies are evaluated against a `Domain` and determine the tuple of
one-dimensional polynomial degrees used on each active cell. They separate the
question "which local degree should this leaf use?" from the later continuity
and compilation machinery that turns those choices into a compiled space with
the requested inter-element continuity.

In other words, a degree policy describes local approximation order, not the
global dof numbering. The same degree policy can be compiled with different
basis families or quadrature policies.

The policy itself does not decide whether degree zero is admissible. That is a
property of the later continuity-aware space compilation: DG axes may use
degree zero, while CG axes are required to use degree at least one so they can
carry endpoint trace data.
"""
abstract type AbstractDegreePolicy end

"""
    AbstractQuadraturePolicy

Abstract supertype for policies that choose cell quadrature sizes from local
polynomial degrees.

Quadrature policies are used when an `HpSpace` is compiled. They determine how
many quadrature points are associated with each active leaf and therefore how
much exactness or overintegration is used by default during later assembly and
verification.

These policies are purely local: they see the degree tuple of one leaf and
return the tensor-product quadrature shape to be used there.
"""
abstract type AbstractQuadraturePolicy end

"""
    UniformDegree(degree)

Degree policy that assigns the same polynomial degree on every axis of every
active leaf.

If `degree = p`, then each active cell receives the degree tuple

  (p, …, p).

This is the simplest way to build a globally uniform `p`-space. It is often the
right starting point when introducing a new problem or when one wants to study
the behavior of a basis family without anisotropic degree variation.
"""
struct UniformDegree <: AbstractDegreePolicy
  degree::Int

  # Degree policies permit zero so DG spaces can represent piecewise constants.
  # The later `HpSpace` compilation step enforces that continuous axes still use
  # degree at least one.
  UniformDegree(degree::Int) = new(_checked_nonnegative(degree, "degree"))
end

UniformDegree(degree::Integer) = UniformDegree(Int(degree))

"""
    AxisDegrees(degrees)

Degree policy with one fixed degree per coordinate axis.

If `degrees = (p₁, …, p_D)`, then every active leaf receives exactly that tuple.
This is useful for anisotropic spaces where one wants, for example, higher
order in one direction and lower order in another while still keeping the same
degree pattern on every cell.
"""
struct AxisDegrees{D} <: AbstractDegreePolicy
  degrees::NTuple{D,Int}

  AxisDegrees{D}(degrees::NTuple{D,Int}) where {D} = new{D}(_checked_degree_tuple(degrees))
end

function AxisDegrees(degrees::NTuple{D,<:Integer}) where {D}
  return AxisDegrees{D}(ntuple(axis -> Int(degrees[axis]), D))
end

"""
    ByLeafDegrees(f)

Degree policy defined by a user callback `f(domain, leaf)`.

The callback must return an integer tuple of length `D` containing the per-axis
degrees of the requested active leaf. This is the most flexible policy and is
intended for user-driven anisotropic degree assignment.

The callback is evaluated on the active leaves of the current domain, so it can
react to geometric position, refinement level, leaf index, periodicity, or any
other information accessible from the `Domain`.
"""
struct ByLeafDegrees{F} <: AbstractDegreePolicy
  f::F
end

# Internal degree-policy representation used after materialization.

# Internal materialized degree policy. `HpSpace` converts all public degree
# policies to this stored representation so the later compilation steps can read
# leaf degrees by direct indexing rather than repeatedly re-evaluating a policy.
struct StoredDegrees{D} <: AbstractDegreePolicy
  leaf_to_index::Vector{Int}
  data::Vector{NTuple{D,Int}}
end

"""
    DegreePlusQuadrature(extra_points)

Quadrature policy that chooses `max(pₐ + extra_points, 1)` points on each axis
of a cell with local degree tuple `(p₁, …, p_D)`.

This simple rule couples quadrature resolution directly to local polynomial
degree. For tensor-product Gauss-Legendre quadrature, increasing the point count
by one per axis increases the exact polynomial degree by two on that axis.

This is the default policy because it tracks local approximation order in a
simple and predictable way while still allowing modest overintegration through
`extra_points`. The lower bound of one point per axis also means DG `p = 0`
cells still receive a valid quadrature rule.
"""
struct DegreePlusQuadrature <: AbstractQuadraturePolicy
  extra_points::Int

  DegreePlusQuadrature(extra_points::Int) = new(_checked_nonnegative(extra_points, "extra_points"))
end

DegreePlusQuadrature(extra_points::Integer) = DegreePlusQuadrature(Int(extra_points))

"""
    SpaceOptions(; basis=TrunkBasis(), degree=UniformDegree(1),
                   quadrature=DegreePlusQuadrature(1), continuity=:cg)

Bundle the basis family, degree policy, and quadrature policy used to build an
`HpSpace`.

This object collects the user-facing choices that define a compiled space:

- which local tensor-product mode family is admissible,
- which polynomial degree tuple each active leaf receives,
- which quadrature sizes are associated with those local degrees,
- and which inter-element continuity policy is requested.

The defaults correspond to a trunk basis, uniform degree `1`, and one extra
Gauss-Legendre point per axis relative to the local degree, together with full
`:cg` continuity.

The `continuity` keyword accepts either one symbol, applied on every axis, or a
tuple with one continuity kind per axis. The symbols mean:

- `:cg`: neighboring leaves share one continuous trace across interfaces normal
  to that axis,
- `:dg`: leaves keep independent trace coefficients across those interfaces.
"""
struct SpaceOptions{B<:AbstractBasisFamily,D<:AbstractDegreePolicy,Q<:AbstractQuadraturePolicy,C}
  basis::B
  degree::D
  quadrature::Q
  continuity::C
end

@inline function _checked_continuity_kind(kind::Symbol, context::AbstractString)
  (kind === :cg || kind === :dg) && return kind
  throw(ArgumentError("$context must be :cg or :dg"))
end

function _checked_continuity_spec(spec)
  if spec isa Symbol
    return _checked_continuity_kind(spec, "continuity")
  elseif spec isa Tuple
    isempty(spec) && throw(ArgumentError("continuity tuple must not be empty"))
    all(kind -> kind isa Symbol, spec) ||
      throw(ArgumentError("continuity tuple entries must be symbols"))
    return ntuple(axis -> _checked_continuity_kind(spec[axis], "continuity[$axis]"), length(spec))
  end

  throw(ArgumentError("continuity must be :cg, :dg, or a tuple of those symbols"))
end

function _normalized_continuity_policy(spec::Symbol, ::Val{D}) where {D}
  checked = _checked_continuity_kind(spec, "continuity")
  return _AxisContinuity{D}(ntuple(_ -> checked, D))
end

function _normalized_continuity_policy(spec::NTuple{N,Symbol}, ::Val{D}) where {N,D}
  N == D || throw(ArgumentError("continuity tuple length must match the space dimension"))
  return _AxisContinuity{D}(spec)
end

function SpaceOptions(; basis::AbstractBasisFamily=TrunkBasis(),
                      degree::AbstractDegreePolicy=UniformDegree(1),
                      quadrature::AbstractQuadraturePolicy=DegreePlusQuadrature(1), continuity=:cg)
  checked_continuity = _checked_continuity_spec(continuity)
  return SpaceOptions{typeof(basis),typeof(degree),typeof(quadrature),typeof(checked_continuity)}(basis,
                                                                                                  degree,
                                                                                                  quadrature,
                                                                                                  checked_continuity)
end

# Public compiled-space object.

"""
    HpSpace(domain, options=SpaceOptions())

Compile a dimension-independent high-order finite-element space on a
Cartesian `Domain`.

`HpSpace` is the public compiled space object of the library. It stores the
active leaves, the chosen basis family, the materialized degree and quadrature
policies, the normalized per-axis continuity policy, and for each active leaf a
sparse expansion of local tensor-product modes into global scalar degrees of
freedom. The actual continuity algebra that produces these expansions is
implemented in `continuity.jl`; this type exposes the compiled result in a form
used by fields, integration, assembly, and adaptivity.

The continuity policy is axiswise: full `:cg`, full `:dg`, and mixed per-axis
tuples are all compiled through the same representation. Degree zero is allowed
only on DG axes.

Two viewpoints are useful when reading the rest of this file:

- Locally, each active leaf has a set of admissible tensor-product modes and a
  sparse expansion of each such mode into global scalar dofs.
- Globally, `HpSpace` is just the collection of those leaf-local expansions plus
  the total scalar dof count and quadrature metadata derived from them.

On axes with `:cg` continuity, boundary-carrying local modes on neighboring
leaves are identified or constrained so that traces agree across matching and
hanging interfaces. On axes with `:dg` continuity, no such coupling is imposed
and the corresponding trace content stays leaf-local. The public query
interface in this file intentionally hides those implementation details behind
one compiled representation.
"""
struct HpSpace{D,T<:AbstractFloat,B<:AbstractBasisFamily,DG<:AbstractDegreePolicy,
               Q<:AbstractQuadraturePolicy,C<:_AbstractContinuityPolicy}
  domain::Domain{D,T}
  basis::B
  degree_policy::DG
  quadrature_policy::Q
  continuity_policy::C
  active_leaves::Vector{Int}
  leaf_to_index::Vector{Int}
  compiled_leaves::Vector{_CompiledLeaf{D,T}}
  scalar_dof_count::Int
  global_quadrature_shape::NTuple{D,Int}
end

# Compile the full space in three stages:
# 1. materialize per-leaf degree data and enumerate local/trace modes under the
#    selected continuity policy,
# 2. compile the resulting face-coupling relations,
# 3. finalize each leaf into sparse local-to-global mode expansions and record
#    the quadrature requirements induced by the quadrature policy.
function HpSpace(domain::Domain{D,T},
                 options::SpaceOptions=SpaceOptions()) where {D,T<:AbstractFloat}
  raw_degree_policy = StoredDegrees(domain, options.degree)
  continuity_policy = _normalized_continuity_policy(options.continuity, Val(D))
  checked_leaf_degrees = Vector{NTuple{D,Int}}(undef, length(raw_degree_policy.data))

  for index in eachindex(raw_degree_policy.data)
    checked_leaf_degrees[index] = _checked_space_degree_tuple(raw_degree_policy.data[index],
                                                              continuity_policy,
                                                              "leaf_degrees[$index]")
  end

  degree_policy = StoredDegrees{D}(raw_degree_policy.leaf_to_index, checked_leaf_degrees)
  active, leaf_to_index, compiled, scalar_dofs, global_quadrature_shape = _compile_space_data(domain,
                                                                                              options.basis,
                                                                                              degree_policy.data,
                                                                                              continuity_policy,
                                                                                              options.quadrature)
  return HpSpace(domain, options.basis, degree_policy, options.quadrature, continuity_policy,
                 active, leaf_to_index, compiled, scalar_dofs, global_quadrature_shape)
end

# Public query layer on compiled spaces.

# Most geometric and topological queries simply delegate to the underlying
# domain. `HpSpace` stores only the additional basis, degree, quadrature, and
# compiled local-mode information layered on top of that domain.
dimension(space::HpSpace) = dimension(space.domain)
domain(space::HpSpace) = space.domain
grid(space::HpSpace) = grid(space.domain)
geometry(space::HpSpace) = geometry(space.domain)
origin(space::HpSpace) = origin(space.domain)
extent(space::HpSpace) = extent(space.domain)
periodic_axes(space::HpSpace) = periodic_axes(domain(space))
is_periodic_axis(space::HpSpace, axis::Integer) = is_periodic_axis(domain(space), axis)

"""
    basis_family(space)

Return the basis family used to compile `space`.

This is typically either [`FullTensorBasis`](@ref) or [`TrunkBasis`](@ref), but
the API is written in terms of the abstract basis-family interface so code can
reason about admissible local modes without committing to a particular family.
"""
basis_family(space::HpSpace) = space.basis

"""
    continuity_policy(space)

Return the normalized per-axis continuity policy of `space`.

The result is an `NTuple{D,Symbol}` with one entry per coordinate axis. Each
entry is currently either `:cg` or `:dg`.

This is the normalized form stored inside `HpSpace`, regardless of whether the
user originally passed one symbol or a full tuple to [`SpaceOptions`](@ref).
"""
function continuity_policy(space::HpSpace{D}) where {D}
  ntuple(axis -> space.continuity_policy.kinds[axis], D)
end

"""
    continuity_kind(space, axis)

Return the continuity kind requested on one axis of `space`.

The queried axis is the axis normal to the interfaces whose coupling behavior
is being described.
"""
function continuity_kind(space::HpSpace, axis::Integer)
  checked_axis = _checked_positive(axis, "axis")
  checked_axis <= dimension(space) ||
    throw(ArgumentError("axis must not exceed the space dimension"))
  return @inbounds space.continuity_policy.kinds[checked_axis]
end

"""
    is_continuous_axis(space, axis)

Return `true` if `space` is continuous across interfaces normal to `axis`.

This is exactly the predicate `continuity_kind(space, axis) === :cg`, provided
as a convenience because later code often wants to branch on "continuous versus
discontinuous on this axis" rather than inspect the raw symbol directly.
"""
is_continuous_axis(space::HpSpace, axis::Integer) = continuity_kind(space, axis) === :cg

"""
    active_leaf_count(space)

Return the number of active leaves on which `space` is compiled.

This equals the active-leaf count of the underlying grid, but the function is
useful at the space level because the compiled leaf data and local mode tables
are stored in that same ordering.
"""
active_leaf_count(space::HpSpace) = length(space.active_leaves)

"""
    active_leaves(space)

Return the active leaves of `space` as a newly allocated vector.

The returned ordering matches the compilation order of the stored local leaf
data. In particular, the `i`-th entry corresponds to the `i`-th compiled leaf
inside the space.
"""
active_leaves(space::HpSpace) = copy(space.active_leaves)

"""
    active_leaf(space, index)

Return the `index`-th active leaf of `space`.

This provides indexed access to the compilation order used by
[`active_leaves`](@ref) and [`active_leaf_count`](@ref).
"""
function active_leaf(space::HpSpace, index::Integer)
  @inbounds space.active_leaves[_checked_index(index, active_leaf_count(space), "active leaf")]
end

"""
    scalar_dof_count(space)

Return the total number of global scalar degrees of freedom in `space`.

Vector-valued fields built on `space` multiply this count by their component
count, but the underlying scalar space always uses this many independent global
unknowns.
"""
scalar_dof_count(space::HpSpace) = space.scalar_dof_count

"""
    global_cell_quadrature_shape(space)

Return the componentwise maximum quadrature shape over all active leaves of
`space`.

If different leaves use different local degree tuples, then the associated local
quadrature shapes may differ as well. This function reports the smallest tensor
shape that dominates all cell-local quadrature shapes componentwise.
"""
global_cell_quadrature_shape(space::HpSpace) = space.global_quadrature_shape

"""
    cell_degrees(space, leaf)

Return the per-axis polynomial degrees of the active leaf `leaf`.

The returned tuple is the degree data actually used when compiling that leaf,
after the user-facing degree policy has been materialized.
"""
cell_degrees(space::HpSpace, leaf::Integer) = _compiled_leaf(space, leaf).degrees

"""
    support_shape(space, leaf)

Return the tensor shape `(p₁ + 1, …, p_D + 1)` of the local mode box on `leaf`.

This is the full tensor-product shape before inactive modes are filtered out by
the chosen basis family.
"""
support_shape(space::HpSpace, leaf::Integer) = _compiled_leaf(space, leaf).support_shape

"""
    cell_quadrature_shape(space, leaf)

Return the per-axis quadrature shape assigned to the active leaf `leaf`.

This is the tensor-product point count tuple produced by the quadrature policy
from the local degree tuple of that leaf.
"""
cell_quadrature_shape(space::HpSpace, leaf::Integer) = _compiled_leaf(space, leaf).quadrature_shape

"""
    local_mode_count(space, leaf)

Return the number of active local tensor-product modes on `leaf`.

This count depends on both the degree tuple of the leaf and the chosen basis
family.
"""
local_mode_count(space::HpSpace, leaf::Integer) = length(_compiled_leaf(space, leaf).local_modes)

"""
    local_modes(space, leaf)

Return the active local tensor-product mode tuples on `leaf`.

Each mode is an integer tuple `(m₁, …, m_D)` indexing the integrated Legendre
factor used on each axis for degrees `pₐ >= 1`. On DG axes with `pₐ = 0`, the
single mode `mₐ = 0` instead refers to the cellwise constant finite-element
factor.

The returned ordering is the compiled local mode ordering used by evaluation,
assembly, and `mode_terms`.
"""
local_modes(space::HpSpace, leaf::Integer) = copy(_compiled_leaf(space, leaf).local_modes)

"""
    is_mode_active(space, leaf, mode)

Return `true` if `mode` is an active local tensor-product mode on `leaf`.

This checks both the box bounds induced by the local degrees and the filtering
rule of the chosen basis family.
"""
function is_mode_active(space::HpSpace, leaf::Integer, mode::NTuple{D,<:Integer}) where {D}
  compiled = _compiled_leaf(space, leaf)
  return _mode_lookup(compiled, mode) != 0
end

"""
    mode_terms(space, leaf, mode)

Return the sparse expansion of one active local mode on `leaf` in terms of
global scalar degrees of freedom.

The result is a vector of `global_dof => coefficient` pairs. For purely
interior modes this is typically a single coefficient `1`, while boundary and
hanging-interface modes may expand into several global dofs due to continuity
constraints. On DG-only trace content, most active local modes simply map to
one independent scalar dof. Modes that participate in CG trace coupling may
instead inherit nontrivial sparse expansions from the continuity compiler.

This is the most direct public view of the continuity compilation performed in
`continuity.jl`.
"""
function mode_terms(space::HpSpace{D,T}, leaf::Integer,
                    mode::NTuple{D,<:Integer}) where {D,T<:AbstractFloat}
  compiled = _compiled_leaf(space, leaf)
  mode_index = _mode_lookup(compiled, mode)
  mode_index != 0 || throw(ArgumentError("mode is not active on this leaf"))
  return Pair{Int,T}[compiled.term_indices[term_index] => compiled.term_coefficients[term_index]
                     for term_index in _mode_term_range(compiled, mode_index)]
end

"""
    check_space(space)

Validate the internal consistency of a compiled `HpSpace`.

This checks that the active-leaf list and lookup tables agree, that every
compiled leaf stores coherent sparse local-mode data, that all quadrature shapes
are positive, and that every global scalar dof is referenced by at least one
local mode expansion. The function throws an `ArgumentError` if an inconsistency
is detected and otherwise returns `nothing`.

The same checks apply to continuous, discontinuous, and mixed spaces: only the
structure of the stored local mode expansions differs.
"""
function check_space(space::HpSpace)
  length(space.active_leaves) == length(space.compiled_leaves) ||
    throw(ArgumentError("compiled leaf data must match the active-leaf list"))
  stored_cell_count(grid(space)) == length(space.leaf_to_index) ||
    throw(ArgumentError("leaf lookup length mismatch"))
  all(axis -> space.global_quadrature_shape[axis] > 0, 1:dimension(space)) ||
    throw(ArgumentError("global quadrature shape entries must be positive"))

  used = falses(space.scalar_dof_count)

  for leaf_index in eachindex(space.active_leaves)
    _check_compiled_leaf!(space, leaf_index, used)
  end

  all(used) || throw(ArgumentError("some scalar dofs are unused"))
  return nothing
end

# Degree-policy materialization.

# Degree-policy evaluation for the common uniform case.
function _leaf_degrees(policy::UniformDegree, domain::Domain{D}, leaf::Int) where {D}
  ntuple(_ -> policy.degree, D)
end

# Axis degrees are already stored in the exact format the compiler needs.
_leaf_degrees(policy::AxisDegrees{D}, domain::Domain{D}, leaf::Int) where {D} = policy.degrees

# Evaluate a user callback and validate that it returns a full nonnegative degree
# tuple of the correct dimension.
function _leaf_degrees(policy::ByLeafDegrees, domain::Domain{D}, leaf::Int) where {D}
  value = policy.f(domain, leaf)
  value isa NTuple{D,<:Integer} ||
    throw(ArgumentError("custom degree policy must return an NTuple{$D,Int}"))
  return _checked_degree_tuple(value)
end

# Stored degree policies are resolved by direct lookup on the active-leaf index
# table built when the policy was materialized.
function _leaf_degrees(policy::StoredDegrees{D}, domain::Domain{D}, leaf::Int) where {D}
  _, leaf_index = _checked_active_leaf_index(grid(domain), policy.leaf_to_index, leaf,
                                             "degree-policy")
  return @inbounds policy.data[leaf_index]
end

# Materialize an arbitrary public degree policy on the current active-leaf set.
function StoredDegrees(domain::Domain{D}, policy::AbstractDegreePolicy) where {D}
  active = active_leaves(grid(domain))
  degrees = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    degrees[index] = _leaf_degrees(policy, domain, active[index])
  end

  return StoredDegrees(domain, degrees)
end

# Low-level constructor from explicit per-leaf degree tuples.
function StoredDegrees(domain::Domain{D}, degrees::AbstractVector{<:NTuple{D,<:Integer}}) where {D}
  grid_data = grid(domain)
  active = active_leaves(grid_data)
  length(degrees) == length(active) ||
    throw(ArgumentError("degree data must match the active-leaf count"))
  leaf_to_index = _active_leaf_lookup(grid_data, active)
  checked = Vector{NTuple{D,Int}}(undef, length(active))

  for index in eachindex(active)
    checked[index] = _checked_degree_tuple(degrees[index])
  end

  return StoredDegrees{D}(leaf_to_index, checked)
end

# Quadrature-policy evaluation and full space compilation.

# The default quadrature policy simply adds a fixed offset to every local degree
# component.
function _quadrature_shape(policy::DegreePlusQuadrature, degrees::NTuple{D,Int}) where {D}
  return ntuple(axis -> max(degrees[axis] + policy.extra_points, 1), D)
end

# Compile the continuity system and all leaf-local sparse mode expansions for
# one `HpSpace` construction. The work splits into three conceptual stages:
# 1. freeze the active leaves and cache their local mode patterns,
# 2. apply the requested continuity policy, which means either solving the
#    global trace algebra (`:cg`) or intentionally skipping it (`:dg`),
# 3. finalize every active leaf into the sparse runtime format used by
#    evaluation and assembly while also accumulating the global quadrature
#    envelope induced by the quadrature policy.
#
# The active-leaf ordering, lookup table, and global quadrature envelope are
# produced together so the public constructor stays focused on wiring the final
# immutable `HpSpace`. Algebraically, this is where a set of leaf-local modal
# bases becomes one sparse global space description.
function _compile_space_data(domain::Domain{D,T}, basis::B, leaf_degrees::Vector{NTuple{D,Int}},
                             continuity_policy::_AxisContinuity{D},
                             quadrature_policy) where {D,T<:AbstractFloat,B<:AbstractBasisFamily}
  grid_data = grid(domain)
  active = active_leaves(grid_data)
  state = _enumerate_space_modes(domain, basis, leaf_degrees, continuity_policy)
  _compile_face_coupling!(state, continuity_policy)
  leaf_to_index = _active_leaf_lookup(grid_data, active)
  compiled, global_quadrature_shape = _finalize_compiled_leaves(state, continuity_policy,
                                                                quadrature_policy)

  return active, leaf_to_index, compiled, state.next_global_dof - 1, global_quadrature_shape
end

# Validation and small lookup helpers.

# Active-leaf lookup tables are used both by stored degree policies and by the
# compiled `HpSpace` itself. They map raw leaf ids to the active-leaf ordering
# used by the stored vectors.
function _active_leaf_lookup(grid_data::CartesianGrid, active::AbstractVector{<:Integer})
  leaf_to_index = zeros(Int, stored_cell_count(grid_data))

  for index in eachindex(active)
    leaf_to_index[active[index]] = index
  end

  return leaf_to_index
end

@inline function _merge_quadrature_shapes(current::NTuple{D,Int}, next::NTuple{D,Int}) where {D}
  ntuple(axis -> max(current[axis], next[axis]), D)
end

# Validate one degree tuple that has already been checked to have the correct
# dimension and integer element type.
@inline function _checked_degree_tuple(degrees::NTuple{D,<:Integer}) where {D}
  return ntuple(axis -> _checked_nonnegative(degrees[axis], "degrees[$axis]"), D)
end

@inline function _checked_space_degree(degree::Integer, continuity_policy::_AxisContinuity,
                                       axis::Int, name::AbstractString)
  checked = _checked_nonnegative(degree, name)
  _is_cg_axis(continuity_policy, axis) &&
    checked == 0 &&
    throw(ArgumentError("$name must be at least 1 on :cg axis $axis"))
  return checked
end

function _checked_space_degree_tuple(degrees::NTuple{D,<:Integer},
                                     continuity_policy::_AxisContinuity{D},
                                     name::AbstractString) where {D}
  return ntuple(axis -> _checked_space_degree(degrees[axis], continuity_policy, axis,
                                              "$name[$axis]"), D)
end

# Validate one compiled leaf and mark all referenced global scalar dofs as used.
function _check_compiled_leaf!(space::HpSpace, leaf_index::Int, used::BitVector)
  leaf = space.active_leaves[leaf_index]
  compiled = space.compiled_leaves[leaf_index]
  compiled.leaf == leaf || throw(ArgumentError("compiled leaf mismatch"))
  space.leaf_to_index[leaf] == leaf_index || throw(ArgumentError("leaf lookup mismatch"))
  length(compiled.term_offsets) == length(compiled.local_modes) + 1 ||
    throw(ArgumentError("term offsets must match the local-mode count"))
  compiled.term_offsets[end] == length(compiled.term_indices) + 1 ||
    throw(ArgumentError("invalid term offsets"))
  length(compiled.term_coefficients) == length(compiled.term_indices) ||
    throw(ArgumentError("term coefficient count must match term index count"))
  length(compiled.single_term_indices) == length(compiled.local_modes) ||
    throw(ArgumentError("single-term lookup length mismatch"))
  length(compiled.single_term_coefficients) == length(compiled.local_modes) ||
    throw(ArgumentError("single-term coefficient length mismatch"))
  all(axis -> compiled.quadrature_shape[axis] > 0, 1:dimension(space)) ||
    throw(ArgumentError("cell quadrature shape entries must be positive"))
  _mark_compiled_leaf_dofs!(used, compiled, space.scalar_dof_count)
  return nothing
end

function _mark_compiled_leaf_dofs!(used::BitVector, compiled::_CompiledLeaf, scalar_dof_count::Int)
  for mode_index in eachindex(compiled.local_modes)
    for term_index in _mode_term_range(compiled, mode_index)
      global_dof = compiled.term_indices[term_index]
      1 <= global_dof <= scalar_dof_count || throw(ArgumentError("global dof out of bounds"))
      used[global_dof] = true
    end
  end

  return used
end

# Resolve one active leaf to its compiled leaf data.
function _compiled_leaf(space::HpSpace, leaf::Integer)
  _, leaf_index = _checked_active_leaf_index(grid(space), space.leaf_to_index, leaf, "space")
  return @inbounds space.compiled_leaves[leaf_index]
end

@inline function _checked_active_leaf_index(grid_data::CartesianGrid,
                                            leaf_to_index::AbstractVector{<:Integer}, leaf::Integer,
                                            context::AbstractString)
  checked_leaf = _checked_cell(grid_data, leaf)
  leaf_index = @inbounds leaf_to_index[checked_leaf]
  leaf_index != 0 || throw(ArgumentError("$context leaf $checked_leaf is not active"))
  return checked_leaf, leaf_index
end

@inline function _mode_term_range(compiled::_CompiledLeaf, mode_index::Integer)
  first_term = @inbounds compiled.term_offsets[mode_index]
  last_term = @inbounds compiled.term_offsets[mode_index+1] - 1
  return first_term:last_term
end

# Lookup one local mode inside the dense tensor-product support box. The lookup
# table stores zero for inactive modes and the positive local-mode index for
# active ones.
function _mode_lookup(compiled::_CompiledLeaf{D}, mode::NTuple{D,<:Integer}) where {D}
  _mode_within_degrees(compiled.support_shape .- 1, mode) || return 0
  return @inbounds compiled.mode_lookup[_flatten_mode(mode, compiled.support_shape)]
end

# Flatten one tensor-product mode tuple into the one-based mixed-radix index
# used by `mode_lookup`. The stride convention matches `basis.jl`: axis 1 varies
# fastest.
function _flatten_mode(mode::NTuple{D,<:Integer}, shape::NTuple{D,Int}) where {D}
  index = 1
  stride = 1

  for axis in 1:D
    index += Int(mode[axis]) * stride
    stride *= shape[axis]
  end

  return index
end

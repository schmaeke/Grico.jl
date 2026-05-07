# This file implements the first-party weak-form operator layer. It is the
# preferred affine-problem API for users who want to write variational forms
# directly instead of defining low-level local kernels.
#
# A weak-form callback describes one local integrand, for example
# `inner(grad(v), grad(w))` or `jump(value(v)) * jump(value(w))`. The code below
# lowers that single mathematical expression to the three representations used
# elsewhere in the matrix-free stack:
#
# 1. local matrices for diagnostics, exact small solves, and reference checks,
# 2. local diagonal entries for Jacobi-like components and tests,
# 3. allocation-free local matrix-vector products for production apply paths.
#
# Bilinear callbacks must be genuinely linear in both the test proxy `v` and
# the trial proxy `w`. The implementation enforces this with symbolic marker
# terms rather than by trusting documentation: invalid forms such as `value(v)^2`,
# `value(w)^2`, `value(v) + value(w)`, or constants fail with `ArgumentError`
# before they can silently produce a path-dependent operator.

# These closed internal containers attach a user callback to the geometric
# integration region where it acts. Cell, boundary, and embedded-surface forms
# share the same one-sided local evaluation machinery, while interface forms
# operate on two trace sides and therefore use dedicated kernels below.
abstract type _LocalBilinearForm end
abstract type _LocalLinearForm end

struct _CellBilinearForm{F,TF<:AbstractField,UF<:AbstractField} <: _LocalBilinearForm
  form::F
  test_field::TF
  trial_field::UF
end

struct _BoundaryBilinearForm{F,TF<:AbstractField,UF<:AbstractField} <: _LocalBilinearForm
  form::F
  test_field::TF
  trial_field::UF
end

struct _SurfaceBilinearForm{F,TF<:AbstractField,UF<:AbstractField} <: _LocalBilinearForm
  form::F
  test_field::TF
  trial_field::UF
end

struct _InterfaceBilinearForm{F,TF<:AbstractField,UF<:AbstractField}
  form::F
  test_field::TF
  trial_field::UF
end

struct _CellLinearForm{F,TF<:AbstractField} <: _LocalLinearForm
  form::F
  test_field::TF
end

struct _BoundaryLinearForm{F,TF<:AbstractField} <: _LocalLinearForm
  form::F
  test_field::TF
end

struct _SurfaceLinearForm{F,TF<:AbstractField} <: _LocalLinearForm
  form::F
  test_field::TF
end

struct _InterfaceLinearForm{F,TF<:AbstractField}
  form::F
  test_field::TF
end

# A quadrature-point proxy keeps the public callback vocabulary independent of
# the concrete `CellValues`, `FaceValues`, `InterfaceValues`, or `SurfaceValues`
# container. All geometric accessors dispatch through `values` and `point_index`.
struct _WeakQuadraturePoint{V}
  values::V
  point_index::Int
end

# Basis proxies are used when a kernel is explicitly iterating test or trial
# modes, for example while forming a dense local matrix. Linear RHS forms use
# `_WeakBasisFunction`, while bilinear forms split test and trial basis proxies
# so the symbolic algebra can detect dependence on each argument separately.
struct _WeakBasisFunction{V,F<:AbstractField}
  values::V
  field::F
  point_index::Int
  component::Int
  mode_index::Int
end

struct _WeakTestBasisFunction{V,F<:AbstractField}
  values::V
  field::F
  point_index::Int
  component::Int
  mode_index::Int
end

struct _WeakTrialBasisFunction{V,F<:AbstractField}
  values::V
  field::F
  point_index::Int
  component::Int
  mode_index::Int
end

# During matrix-free application, a bilinear trial argument represents the
# discrete trial field u_h rather than one basis function. This is equivalent to
# multiplying the assembled local matrix by the coefficient vector for genuine
# bilinear forms, and it avoids the quadratic trial-mode loop in the hot apply
# path.
struct _WeakTrialFunction{D,T<:AbstractFloat,F<:AbstractField,N}
  field::F
  component::Int
  value::T
  gradient::NTuple{D,T}
  normal_gradient::N
end

# Tensorized trial proxies avoid materializing `u_h` and `∇u_h` scalars until a
# callback actually asks for them. The value and gradient work arrays may have
# zero columns when the symbolic prepass proves that the corresponding channel
# is unused.
struct _WeakTensorTrialFunction{D,T<:AbstractFloat,F<:AbstractField,N,PV<:AbstractVector{T},
                                PG<:AbstractMatrix{T}}
  field::F
  component::Int
  point_index::Int
  point_values::PV
  point_gradients::PG
  normal::N
end

# Tensorized weak-form paths evaluate the form once per quadrature point with a
# symbolic test function. The resulting `_WeakTestTerm` stores the coefficients
# multiplying v_h and ∇v_h so that the contribution can be projected with
# sum-factorized tensor kernels.
struct _WeakTestFunction{D,T<:AbstractFloat,F<:AbstractField,N}
  field::F
  component::Int
  normal::N
end

# A test term is the affine functional α v_h + β · ∇v_h at one quadrature point.
# Nonlinear dependence on the test function is rejected explicitly because such
# a form cannot be represented by a bilinear matrix-free operator.
struct _WeakTestTerm{D,T<:AbstractFloat} <: Number
  value_coefficient::T
  gradient_coefficients::NTuple{D,T}
end

# Trial terms mark whether the callback requested `w`, `∇w`, or ∂ₙw. The boolean
# type parameters are compile-time channel flags used by tensorized apply paths
# to skip interpolation or gradient evaluation that cannot contribute.
struct _WeakTrialTerm{T<:AbstractFloat,V,G,N} <: Number
  value::T
end

# Bilinear terms store the coefficients of the symbolic test functional after it
# has been multiplied by a trial marker. The value coefficient multiplies `v_h`;
# each gradient coefficient multiplies one component of `∇v_h`. The trial flags
# are propagated so tensorized apply kernels know which trial channels to build.
struct _WeakBilinearTerm{D,T<:AbstractFloat,V,G,N} <: Number
  value_coefficient::T
  gradient_coefficients::NTuple{D,T}
end

# Interface quantities are represented as oriented minus/plus trace pairs. The
# orientation matches `InterfaceValues`: `jump(pair)` is `plus - minus`, and
# `average(pair)` is the arithmetic mean of the two sides.
struct _WeakTracePair{M,P}
  minus::M
  plus::P
end

# Trace basis and function proxies carry both side evaluation objects plus a
# side selector. A value on the inactive side is represented by a structurally
# matching zero so ordinary trace algebra can be reused in callback code.
struct _WeakTraceBasisFunction{MV,PV,F<:AbstractField}
  minus_values::MV
  plus_values::PV
  field::F
  point_index::Int
  component::Int
  mode_index::Int
  plus_side::Bool
end

struct _WeakTraceTestBasisFunction{MV,PV,F<:AbstractField}
  minus_values::MV
  plus_values::PV
  field::F
  point_index::Int
  component::Int
  mode_index::Int
  plus_side::Bool
end

struct _WeakTraceTrialBasisFunction{MV,PV,F<:AbstractField}
  minus_values::MV
  plus_values::PV
  field::F
  point_index::Int
  component::Int
  mode_index::Int
  plus_side::Bool
end

struct _WeakTraceTestFunction{D,T<:AbstractFloat,F<:AbstractField,N}
  field::F
  component::Int
  plus_side::Bool
  normal::N
end

struct _WeakTraceTrialFunction{M,P,F<:AbstractField}
  field::F
  component::Int
  minus::M
  plus::P
end

"""
    add_cell_bilinear!(problem, test_field, trial_field, form)
    add_cell_bilinear!(form, problem, test_field, trial_field)

Add a cell bilinear weak form to an affine problem.

The callback `form(q, v, w)` represents the cell integrand of a local bilinear
form, integrated over every active cell with the quadrature attached to the
test and trial spaces. The proxies `v` and `w` are the current test and trial
field components. Use [`value`](@ref), [`grad`](@ref), [`gradient`](@ref), and
[`component`](@ref) to write expressions such as
`inner(grad(v), grad(w))` or `value(v) * value(w)`.

The callback must be linear in both `v` and `w`. Grico checks this contract when
the operator is applied or locally assembled and throws `ArgumentError` for
constant, one-sided, affine, or nonlinear expressions. The function validates
that both fields belong to `problem`, mutates `problem`, and returns it.
"""
function add_cell_bilinear!(problem::AffineProblem, test_field::AbstractField,
                            trial_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "cell bilinear test")
  _check_problem_field(data.fields, trial_field, "cell bilinear trial")
  add_cell!(problem, _CellBilinearForm(form, test_field, trial_field))
  return problem
end

function add_cell_bilinear!(form, problem::AffineProblem, test_field::AbstractField,
                            trial_field::AbstractField)
  return add_cell_bilinear!(problem, test_field, trial_field, form)
end

"""
    add_cell_linear!(problem, test_field, form)
    add_cell_linear!(form, problem, test_field)

Add a cell linear weak form to an affine problem.

The callback `form(q, v)` represents a cell load integrand, integrated over
every active cell and accumulated into the right-hand side block for
`test_field`. It may depend on the quadrature point `q`, physical coordinates,
and the current test component `v`, but it has no trial argument and therefore
does not contribute to the affine operator matrix.

The function validates that `test_field` belongs to `problem`, mutates
`problem`, and returns it.
"""
function add_cell_linear!(problem::AffineProblem, test_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "cell linear test")
  add_cell!(problem, _CellLinearForm(form, test_field))
  return problem
end

function add_cell_linear!(form, problem::AffineProblem, test_field::AbstractField)
  return add_cell_linear!(problem, test_field, form)
end

"""
    add_boundary_bilinear!(problem, boundary, test_field, trial_field, form)
    add_boundary_bilinear!(form, problem, boundary, test_field, trial_field)

Add a boundary-face bilinear weak form to an affine problem.

The callback `form(q, v, w)` represents an integrand on the physical
`boundary`. The quadrature point `q` supports the usual point and weight
accessors plus [`normal`](@ref); the test and trial proxies additionally support
[`normal_gradient`](@ref), which evaluates ∂ₙv or ∂ₙw on the selected face.

Boundary bilinear forms are intended for weak boundary terms such as Nitsche
penalties, Robin conditions, and DG fluxes on exterior faces. The callback must
be linear in both `v` and `w`; invalid bilinear structure is rejected during
assembly or application. The function validates the fields and boundary
attachment through the normal problem machinery, mutates `problem`, and returns
it.
"""
function add_boundary_bilinear!(problem::AffineProblem, boundary::BoundaryFace,
                                test_field::AbstractField, trial_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "boundary bilinear test")
  _check_problem_field(data.fields, trial_field, "boundary bilinear trial")
  add_boundary!(problem, boundary, _BoundaryBilinearForm(form, test_field, trial_field))
  return problem
end

function add_boundary_bilinear!(form, problem::AffineProblem, boundary::BoundaryFace,
                                test_field::AbstractField, trial_field::AbstractField)
  return add_boundary_bilinear!(problem, boundary, test_field, trial_field, form)
end

"""
    add_boundary_linear!(problem, boundary, test_field, form)
    add_boundary_linear!(form, problem, boundary, test_field)

Add a boundary-face linear weak form to an affine problem.

The callback `form(q, v)` represents a load integrand on the physical
`boundary`. It is useful for natural boundary data and weak imposition terms
that contribute only to the right-hand side. The point proxy provides the face
normal, and the test proxy supports value, gradient, normal-gradient, and
component access.

The function validates the field and boundary through the normal problem
machinery, mutates `problem`, and returns it.
"""
function add_boundary_linear!(problem::AffineProblem, boundary::BoundaryFace,
                              test_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "boundary linear test")
  add_boundary!(problem, boundary, _BoundaryLinearForm(form, test_field))
  return problem
end

function add_boundary_linear!(form, problem::AffineProblem, boundary::BoundaryFace,
                              test_field::AbstractField)
  return add_boundary_linear!(problem, boundary, test_field, form)
end

"""
    add_surface_bilinear!(problem, test_field, trial_field, form)
    add_surface_bilinear!(problem, tag, test_field, trial_field, form)
    add_surface_bilinear!(form, problem, test_field, trial_field)
    add_surface_bilinear!(form, problem, tag, test_field, trial_field)

Add an embedded-surface bilinear weak form to an affine problem.

The callback `form(q, v, w)` is integrated over embedded-surface quadrature
items previously attached to `problem`. Use the untagged method for all
embedded surfaces, or the `tag` method to restrict the form to a named surface
quadrature family.

Surface forms use the same one-sided weak-form vocabulary as boundary forms:
`q` supplies the surface point, weight, and normal, while `v` and `w` supply
value, gradient, normal-gradient, and component access. The callback must be
linear in both field proxies. The function validates the fields, mutates
`problem`, and returns it.
"""
function add_surface_bilinear!(problem::AffineProblem, test_field::AbstractField,
                               trial_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface bilinear test")
  _check_problem_field(data.fields, trial_field, "surface bilinear trial")
  add_surface!(problem, _SurfaceBilinearForm(form, test_field, trial_field))
  return problem
end

function add_surface_bilinear!(problem::AffineProblem, tag::Symbol, test_field::AbstractField,
                               trial_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface bilinear test")
  _check_problem_field(data.fields, trial_field, "surface bilinear trial")
  add_surface!(problem, tag, _SurfaceBilinearForm(form, test_field, trial_field))
  return problem
end

function add_surface_bilinear!(form, problem::AffineProblem, test_field::AbstractField,
                               trial_field::AbstractField)
  return add_surface_bilinear!(problem, test_field, trial_field, form)
end

function add_surface_bilinear!(form, problem::AffineProblem, tag::Symbol, test_field::AbstractField,
                               trial_field::AbstractField)
  return add_surface_bilinear!(problem, tag, test_field, trial_field, form)
end

"""
    add_surface_linear!(problem, test_field, form)
    add_surface_linear!(problem, tag, test_field, form)
    add_surface_linear!(form, problem, test_field)
    add_surface_linear!(form, problem, tag, test_field)

Add an embedded-surface linear weak form to an affine problem.

The callback `form(q, v)` contributes a right-hand-side integrand over embedded
surface quadrature items. The optional `tag` restricts the contribution to one
named surface family; otherwise the form is attached to all embedded-surface
quadrature data in the problem.

Surface linear forms are appropriate for forcing, flux, or constraint terms
that live on a codimension-one embedded surface and do not depend on a trial
field. The function validates the field, mutates `problem`, and returns it.
"""
function add_surface_linear!(problem::AffineProblem, test_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface linear test")
  add_surface!(problem, _SurfaceLinearForm(form, test_field))
  return problem
end

function add_surface_linear!(problem::AffineProblem, tag::Symbol, test_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface linear test")
  add_surface!(problem, tag, _SurfaceLinearForm(form, test_field))
  return problem
end

function add_surface_linear!(form, problem::AffineProblem, test_field::AbstractField)
  return add_surface_linear!(problem, test_field, form)
end

function add_surface_linear!(form, problem::AffineProblem, tag::Symbol, test_field::AbstractField)
  return add_surface_linear!(problem, tag, test_field, form)
end

"""
    add_interface_bilinear!(problem, test_field, trial_field, form)
    add_interface_bilinear!(form, problem, test_field, trial_field)

Add an interface bilinear weak form to an affine problem.

The callback `form(q, v, w)` is integrated over interior interfaces. Unlike
cell and boundary forms, `value(v)`, `grad(v)`, and `normal_gradient(v)` return
oriented minus/plus trace pairs. Combine trace quantities with [`jump`](@ref),
[`average`](@ref), [`avg`](@ref), [`minus`](@ref), and [`plus`](@ref). The
orientation is fixed by `InterfaceValues`: `jump(pair)` is `plus(pair) -
minus(pair)`.

Interface bilinear forms are the natural representation for DG fluxes, interior
penalty terms, and weak coupling across nonmatching hp interfaces. The callback
must be linear in both the test and trial traces. The function validates both
fields, mutates `problem`, and returns it.
"""
function add_interface_bilinear!(problem::AffineProblem, test_field::AbstractField,
                                 trial_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "interface bilinear test")
  _check_problem_field(data.fields, trial_field, "interface bilinear trial")
  add_interface!(problem, _InterfaceBilinearForm(form, test_field, trial_field))
  return problem
end

function add_interface_bilinear!(form, problem::AffineProblem, test_field::AbstractField,
                                 trial_field::AbstractField)
  return add_interface_bilinear!(problem, test_field, trial_field, form)
end

"""
    add_interface_linear!(problem, test_field, form)
    add_interface_linear!(form, problem, test_field)

Add an interface linear weak form to an affine problem.

The callback `form(q, v)` contributes a right-hand-side integrand over interior
interfaces. As for interface bilinear forms, field accessors on `v` return
minus/plus trace pairs that can be combined with `jump`, `average`, `minus`, and
`plus`.

This is intended for source or flux data that live on interfaces but do not
depend on a trial field. The function validates the field, mutates `problem`,
and returns it.
"""
function add_interface_linear!(problem::AffineProblem, test_field::AbstractField, form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "interface linear test")
  add_interface!(problem, _InterfaceLinearForm(form, test_field))
  return problem
end

function add_interface_linear!(form, problem::AffineProblem, test_field::AbstractField)
  return add_interface_linear!(problem, test_field, form)
end

# Geometric callback accessors delegate to the underlying integration container.
# Interface cell sizes are deliberately conservative: for penalty scalings on a
# nonmatching interface, the characteristic local length is the smaller of the
# two adjacent cells on each axis.
@inline point(q::_WeakQuadraturePoint) = point(q.values, q.point_index)
@inline weight(q::_WeakQuadraturePoint) = weight(q.values, q.point_index)
@inline coordinate(q::_WeakQuadraturePoint, axis::Integer) = point(q)[axis]
@inline normal(q::_WeakQuadraturePoint) = normal(q.values, q.point_index)
@inline cell_size(q::_WeakQuadraturePoint{<:InterfaceValues}, field::AbstractField) = ntuple(axis -> cell_size(q,
                                                                                                               field,
                                                                                                               axis),
                                                                                             dimension(field_space(field)))
@inline function cell_size(q::_WeakQuadraturePoint{<:InterfaceValues}, field::AbstractField,
                           axis::Integer)
  domain_data = field_space(field).domain
  minus_size = cell_size(domain_data, q.values.minus_leaf, axis)
  plus_size = cell_size(domain_data, q.values.plus_leaf, axis)
  return min(minus_size, plus_size)
end
@inline cell_size(q::_WeakQuadraturePoint, field::AbstractField) = ntuple(axis -> cell_size(field_space(field).domain,
                                                                                            q.values.leaf,
                                                                                            axis),
                                                                          dimension(field_space(field)))
@inline cell_size(q::_WeakQuadraturePoint, field::AbstractField, axis::Integer) = cell_size(field_space(field).domain,
                                                                                            q.values.leaf,
                                                                                            axis)

@inline value(basis::_WeakBasisFunction) = shape_value(basis.values, basis.field, basis.point_index,
                                                       basis.mode_index)

@inline function value(basis::_WeakTestBasisFunction)
  shape = shape_value(basis.values, basis.field, basis.point_index, basis.mode_index)
  return _basis_value_test_term(typeof(shape), Val(dimension(field_space(basis.field))), shape)
end

@inline value(basis::_WeakTrialBasisFunction) = _trial_value_term(shape_value(basis.values,
                                                                              basis.field,
                                                                              basis.point_index,
                                                                              basis.mode_index))

@inline _trial_value_term(value::T) where {T<:AbstractFloat} = _WeakTrialTerm{T,true,false,false}(value)
@inline _trial_gradient_term(value::T) where {T<:AbstractFloat} = _WeakTrialTerm{T,false,true,
                                                                                 false}(value)
@inline _trial_normal_term(value::T) where {T<:AbstractFloat} = _WeakTrialTerm{T,false,false,true}(value)

@inline function _basis_value_test_term(::Type{T}, ::Val{D}, value::T) where {D,T<:AbstractFloat}
  return _WeakTestTerm(value, ntuple(_ -> zero(T), Val(D)))
end

@inline function _basis_gradient_test_term(::Type{T}, ::Val{D}, axis::Int,
                                           value::T) where {D,T<:AbstractFloat}
  return _WeakTestTerm(zero(T), ntuple(current -> current == axis ? value : zero(T), Val(D)))
end

# The following accessor methods are the semantic bridge between user callback
# syntax and the internal lowering. Linear forms receive numeric basis values.
# Bilinear test arguments produce symbolic test functionals, and trial arguments
# produce symbolic trial markers or reconstructed trial-field values depending
# on the kernel path.
@inline value(trial::_WeakTrialFunction) = _trial_value_term(trial.value)

@inline function value(trial::_WeakTensorTrialFunction)
  return _trial_value_term(@inbounds trial.point_values[trial.point_index])
end

@inline function value(test::_WeakTestFunction{D,T}) where {D,T<:AbstractFloat}
  return _WeakTestTerm(one(T), ntuple(_ -> zero(T), Val(D)))
end

@inline function value(test::_WeakTraceTestFunction{D,T}) where {D,T<:AbstractFloat}
  active_value = _WeakTestTerm(one(T), ntuple(_ -> zero(T), Val(D)))
  inactive_value = zero(active_value)
  return test.plus_side ? _WeakTracePair(inactive_value, active_value) :
         _WeakTracePair(active_value, inactive_value)
end

@inline function value(trial::_WeakTraceTrialFunction)
  return _WeakTracePair(value(trial.minus), value(trial.plus))
end

@inline function value(basis::_WeakTraceBasisFunction)
  active_value = _trace_shape_value(basis)
  inactive_value = zero(active_value)
  return basis.plus_side ? _WeakTracePair(inactive_value, active_value) :
         _WeakTracePair(active_value, inactive_value)
end

@inline function value(basis::_WeakTraceTestBasisFunction)
  active_shape = _trace_shape_value(basis)
  active_value = _basis_value_test_term(typeof(active_shape),
                                        Val(dimension(field_space(basis.field))), active_shape)
  inactive_value = zero(active_value)
  return basis.plus_side ? _WeakTracePair(inactive_value, active_value) :
         _WeakTracePair(active_value, inactive_value)
end

@inline function value(basis::_WeakTraceTrialBasisFunction)
  active_value = _trial_value_term(_trace_shape_value(basis))
  inactive_value = zero(active_value)
  return basis.plus_side ? _WeakTracePair(inactive_value, active_value) :
         _WeakTracePair(active_value, inactive_value)
end

@inline function gradient(basis::_WeakBasisFunction)
  return shape_gradient(basis.values, basis.field, basis.point_index, basis.mode_index)
end

@inline function gradient(basis::_WeakTestBasisFunction)
  gradient_data = shape_gradient(basis.values, basis.field, basis.point_index, basis.mode_index)
  T = typeof(gradient_data[1])
  D = length(gradient_data)
  return ntuple(axis -> _basis_gradient_test_term(T, Val(D), axis, gradient_data[axis]), Val(D))
end

@inline function gradient(basis::_WeakTrialBasisFunction)
  return map(_trial_gradient_term,
             shape_gradient(basis.values, basis.field, basis.point_index, basis.mode_index))
end

@inline gradient(trial::_WeakTrialFunction) = map(_trial_gradient_term, trial.gradient)

@inline function gradient(trial::_WeakTensorTrialFunction{D,T}) where {D,T<:AbstractFloat}
  return ntuple(axis -> _trial_gradient_term(@inbounds trial.point_gradients[axis,
                                                                             trial.point_index]),
                Val(D))
end

@inline function gradient(test::_WeakTestFunction{D,T}) where {D,T<:AbstractFloat}
  return ntuple(axis -> _WeakTestTerm(zero(T),
                                      ntuple(current -> current == axis ? one(T) : zero(T), Val(D))),
                Val(D))
end

@inline function gradient(test::_WeakTraceTestFunction{D,T}) where {D,T<:AbstractFloat}
  active_gradient = ntuple(axis -> _WeakTestTerm(zero(T),
                                                 ntuple(current -> current == axis ? one(T) :
                                                                   zero(T), Val(D))), Val(D))
  inactive_gradient = map(zero, active_gradient)
  return test.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline function gradient(trial::_WeakTraceTrialFunction)
  return _WeakTracePair(gradient(trial.minus), gradient(trial.plus))
end

@inline function gradient(basis::_WeakTraceBasisFunction)
  active_gradient = _trace_shape_gradient(basis)
  inactive_gradient = _zero_like(active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline function gradient(basis::_WeakTraceTestBasisFunction)
  active_gradient_data = _trace_shape_gradient(basis)
  T = typeof(active_gradient_data[1])
  D = length(active_gradient_data)
  active_gradient = ntuple(axis -> _basis_gradient_test_term(T, Val(D), axis,
                                                             active_gradient_data[axis]), Val(D))
  inactive_gradient = map(zero, active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline function gradient(basis::_WeakTraceTrialBasisFunction)
  active_gradient = map(_trial_gradient_term, _trace_shape_gradient(basis))
  inactive_gradient = _zero_like(active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

"""
    grad(v)
    ∇(v)

Return the physical gradient of a weak-form field proxy.

Inside cell, boundary, surface, and interface weak forms, `grad(v)` evaluates
`∇v` at the current quadrature point. For scalar fields the result is a tuple
with one entry per spatial axis. For vector-valued fields, each component is
handled by the component proxy supplied to the callback; use
[`component`](@ref) when the form needs component-dependent coefficients.

On interface callbacks the result is a minus/plus trace pair. Use `jump`,
`average`, `minus`, or `plus` before combining it with one-sided quantities.
For example, `inner(jump(grad(v)), average(grad(w)))` contracts the physical
gradient jump of the test trace with the average trial gradient.

`∇` is the unicode alias for the same operation and is intended for source code
that mirrors the mathematical weak form.
"""
@inline grad(basis::_WeakBasisFunction) = gradient(basis)
@inline grad(basis::_WeakTestBasisFunction) = gradient(basis)
@inline grad(basis::_WeakTrialBasisFunction) = gradient(basis)
@inline grad(trial::_WeakTrialFunction) = gradient(trial)
@inline grad(trial::_WeakTensorTrialFunction) = gradient(trial)
@inline grad(test::_WeakTestFunction) = gradient(test)
@inline grad(basis::_WeakTraceBasisFunction) = gradient(basis)
@inline grad(basis::_WeakTraceTestBasisFunction) = gradient(basis)
@inline grad(basis::_WeakTraceTrialBasisFunction) = gradient(basis)
@inline grad(test::_WeakTraceTestFunction) = gradient(test)
@inline grad(trial::_WeakTraceTrialFunction) = gradient(trial)

"""
    ∇(v)

Unicode alias for [`grad`](@ref).
"""
@inline ∇(basis::_WeakBasisFunction) = gradient(basis)
@inline ∇(basis::_WeakTestBasisFunction) = gradient(basis)
@inline ∇(basis::_WeakTrialBasisFunction) = gradient(basis)
@inline ∇(trial::_WeakTrialFunction) = gradient(trial)
@inline ∇(trial::_WeakTensorTrialFunction) = gradient(trial)
@inline ∇(test::_WeakTestFunction) = gradient(test)
@inline ∇(basis::_WeakTraceBasisFunction) = gradient(basis)
@inline ∇(basis::_WeakTraceTestBasisFunction) = gradient(basis)
@inline ∇(basis::_WeakTraceTrialBasisFunction) = gradient(basis)
@inline ∇(test::_WeakTraceTestFunction) = gradient(test)
@inline ∇(trial::_WeakTraceTrialFunction) = gradient(trial)

@inline function normal_gradient(basis::_WeakBasisFunction)
  return shape_normal_gradient(basis.values, basis.field, basis.point_index, basis.mode_index)
end

@inline function normal_gradient(basis::_WeakTestBasisFunction)
  gradient_data = shape_gradient(basis.values, basis.field, basis.point_index, basis.mode_index)
  normal_data = _point_normal(basis.values, basis.point_index)
  T = typeof(gradient_data[1])
  D = length(gradient_data)
  return _WeakTestTerm(zero(T), ntuple(axis -> gradient_data[axis] * normal_data[axis], Val(D)))
end

@inline function normal_gradient(basis::_WeakTrialBasisFunction)
  return _trial_normal_term(shape_normal_gradient(basis.values, basis.field, basis.point_index,
                                                  basis.mode_index))
end

@inline function normal_gradient(trial::_WeakTrialFunction{D,T,F,T}) where {D,T<:AbstractFloat,
                                                                            F<:AbstractField}
  return _trial_normal_term(trial.normal_gradient)
end

function normal_gradient(::_WeakTrialFunction{D,T,F,Nothing}) where {D,T<:AbstractFloat,
                                                                     F<:AbstractField}
  throw(ArgumentError("normal_gradient is only available for weak-form trial functions on faces, interfaces, and embedded surfaces"))
end

@inline function normal_gradient(trial::_WeakTensorTrialFunction{D,T,F,NTuple{D,T}}) where {D,
                                                                                            T<:AbstractFloat,
                                                                                            F<:AbstractField}
  return normal_component(gradient(trial), trial.normal)
end

function normal_gradient(::_WeakTensorTrialFunction{D,T,F,Nothing}) where {D,T<:AbstractFloat,
                                                                           F<:AbstractField}
  throw(ArgumentError("normal_gradient is only available for weak-form trial functions on faces, interfaces, and embedded surfaces"))
end

function normal_gradient(::_WeakTestFunction{D,T,F,Nothing}) where {D,T<:AbstractFloat,
                                                                    F<:AbstractField}
  throw(ArgumentError("normal_gradient is not available in tensorized cell weak-form projection"))
end

@inline function normal_gradient(test::_WeakTestFunction{D,T,F,N}) where {D,T<:AbstractFloat,
                                                                          F<:AbstractField,N}
  return _WeakTestTerm(zero(T), test.normal)
end

@inline function normal_gradient(test::_WeakTraceTestFunction{D,T}) where {D,T<:AbstractFloat}
  active_gradient = _WeakTestTerm(zero(T), test.normal)
  inactive_gradient = zero(active_gradient)
  return test.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline function normal_gradient(trial::_WeakTraceTrialFunction)
  return _WeakTracePair(normal_gradient(trial.minus), normal_gradient(trial.plus))
end

@inline function normal_gradient(basis::_WeakTraceBasisFunction)
  active_gradient = _trace_shape_normal_gradient(basis)
  inactive_gradient = zero(active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline function normal_gradient(basis::_WeakTraceTestBasisFunction)
  values = _trace_values(basis)
  gradient_data = shape_gradient(values, basis.field, basis.point_index, basis.mode_index)
  normal_data = _point_normal(values, basis.point_index)
  T = typeof(gradient_data[1])
  D = length(gradient_data)
  active_gradient = _WeakTestTerm(zero(T),
                                  ntuple(axis -> gradient_data[axis] * normal_data[axis], Val(D)))
  inactive_gradient = zero(active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline function normal_gradient(basis::_WeakTraceTrialBasisFunction)
  active_gradient = _trial_normal_term(_trace_shape_normal_gradient(basis))
  inactive_gradient = zero(active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

"""
    component(v)

Return the field component represented by a weak-form proxy.

For scalar fields this is always `1`. For vector fields, weak-form callbacks
are evaluated component by component, and `component(v)` gives the one-based
component index of the current test, trial, or trace proxy. This keeps
component-coupled forms explicit without requiring users to write separate
operators for each component block.
"""
@inline component(basis::_WeakBasisFunction) = basis.component
@inline component(basis::_WeakTestBasisFunction) = basis.component
@inline component(basis::_WeakTrialBasisFunction) = basis.component
@inline component(trial::_WeakTrialFunction) = trial.component
@inline component(trial::_WeakTensorTrialFunction) = trial.component
@inline component(test::_WeakTestFunction) = test.component
@inline component(basis::_WeakTraceBasisFunction) = basis.component
@inline component(basis::_WeakTraceTestBasisFunction) = basis.component
@inline component(basis::_WeakTraceTrialBasisFunction) = basis.component
@inline component(test::_WeakTraceTestFunction) = test.component
@inline component(trial::_WeakTraceTrialFunction) = trial.component

@inline function _zero_test_term(::Type{T}, ::Val{D}) where {D,T<:AbstractFloat}
  return _WeakTestTerm(zero(T), ntuple(_ -> zero(T), Val(D)))
end

@inline function _zero_bilinear_term(::Type{T}, ::Val{D}) where {D,T<:AbstractFloat}
  return _WeakBilinearTerm{D,T,false,false,false}(zero(T), ntuple(_ -> zero(T), Val(D)))
end

@inline _trial_uses_value(::Type{<:_WeakTrialTerm{T,V,G,N}}) where {T,V,G,N} = V
@inline _trial_uses_gradient(::Type{<:_WeakTrialTerm{T,V,G,N}}) where {T,V,G,N} = G
@inline _trial_uses_normal(::Type{<:_WeakTrialTerm{T,V,G,N}}) where {T,V,G,N} = N
@inline _trial_uses_value(::Type{<:_WeakBilinearTerm{D,T,V,G,N}}) where {D,T,V,G,N} = V
@inline _trial_uses_gradient(::Type{<:_WeakBilinearTerm{D,T,V,G,N}}) where {D,T,V,G,N} = G
@inline _trial_uses_normal(::Type{<:_WeakBilinearTerm{D,T,V,G,N}}) where {D,T,V,G,N} = N

@inline function _combine_trial_term(a::_WeakTrialTerm{T,V1,G1,N1}, b::_WeakTrialTerm{T,V2,G2,N2},
                                     value::T) where {T<:AbstractFloat,V1,G1,N1,V2,G2,N2}
  return _WeakTrialTerm{T,V1 || V2,G1 || G2,N1 || N2}(value)
end

@inline function _combine_bilinear_term(a::_WeakBilinearTerm{D,T,V1,G1,N1},
                                        b::_WeakBilinearTerm{D,T,V2,G2,N2}, value_coefficient::T,
                                        gradient_coefficients::NTuple{D,T}) where {D,
                                                                                   T<:AbstractFloat,
                                                                                   V1,G1,N1,V2,G2,
                                                                                   N2}
  return _WeakBilinearTerm{D,T,V1 || V2,G1 || G2,N1 || N2}(value_coefficient, gradient_coefficients)
end

# Symbolic terms subtype `Number` so ordinary callback code can use scalar
# algebra, `inner`, and `normal_component` without a separate expression tree.
# The methods below preserve only operations that remain linear in the required
# proxy arguments. They reject nonlinear, affine, constant, and one-sided terms
# at the point where the user expression makes the invalid structure visible.
@inline Base.zero(term::_WeakTestTerm{D,T}) where {D,T<:AbstractFloat} = _zero_test_term(T, Val(D))
@inline Base.zero(::Type{_WeakTestTerm{D,T}}) where {D,T<:AbstractFloat} = _zero_test_term(T,
                                                                                           Val(D))
@inline Base.zero(term::_WeakTrialTerm{T,V,G,N}) where {T<:AbstractFloat,V,G,N} = _WeakTrialTerm{T,
                                                                                                 V,
                                                                                                 G,
                                                                                                 N}(zero(T))
@inline Base.zero(::Type{_WeakTrialTerm{T,V,G,N}}) where {T<:AbstractFloat,V,G,N} = _WeakTrialTerm{T,
                                                                                                   V,
                                                                                                   G,
                                                                                                   N}(zero(T))
@inline Base.zero(term::_WeakBilinearTerm{D,T,V,G,N}) where {D,T<:AbstractFloat,V,G,N} = _WeakBilinearTerm{D,
                                                                                                           T,
                                                                                                           V,
                                                                                                           G,
                                                                                                           N}(zero(T),
                                                                                                              ntuple(_ -> zero(T),
                                                                                                                     Val(D)))
@inline Base.zero(::Type{_WeakBilinearTerm{D,T,V,G,N}}) where {D,T<:AbstractFloat,V,G,N} = _WeakBilinearTerm{D,
                                                                                                             T,
                                                                                                             V,
                                                                                                             G,
                                                                                                             N}(zero(T),
                                                                                                                ntuple(_ -> zero(T),
                                                                                                                       Val(D)))

@inline Base.iszero(term::_WeakTestTerm) = iszero(term.value_coefficient) &&
                                           all(iszero, term.gradient_coefficients)
@inline Base.iszero(term::_WeakTrialTerm) = iszero(term.value)
@inline Base.iszero(term::_WeakBilinearTerm) = iszero(term.value_coefficient) &&
                                               all(iszero, term.gradient_coefficients)

@inline function Base.:+(a::_WeakTestTerm{D,T}, b::_WeakTestTerm{D,T}) where {D,T<:AbstractFloat}
  return _WeakTestTerm(a.value_coefficient + b.value_coefficient,
                       ntuple(axis -> a.gradient_coefficients[axis] + b.gradient_coefficients[axis],
                              Val(D)))
end

@inline function Base.:+(a::_WeakTrialTerm{T}, b::_WeakTrialTerm{T}) where {T<:AbstractFloat}
  return _combine_trial_term(a, b, a.value + b.value)
end

@inline function Base.:+(a::_WeakBilinearTerm{D,T},
                         b::_WeakBilinearTerm{D,T}) where {D,T<:AbstractFloat}
  return _combine_bilinear_term(a, b, a.value_coefficient + b.value_coefficient,
                                ntuple(axis -> a.gradient_coefficients[axis] +
                                               b.gradient_coefficients[axis], Val(D)))
end

@inline Base.:-(a::_WeakTestTerm{D,T}) where {D,T<:AbstractFloat} = -one(T) * a
@inline Base.:-(a::_WeakTrialTerm{T}) where {T<:AbstractFloat} = -one(T) * a
@inline Base.:-(a::_WeakBilinearTerm{D,T}) where {D,T<:AbstractFloat} = -one(T) * a

@inline Base.:-(a::_WeakTestTerm{D,T}, b::_WeakTestTerm{D,T}) where {D,T<:AbstractFloat} = a + (-b)
@inline Base.:-(a::_WeakTrialTerm{T}, b::_WeakTrialTerm{T}) where {T<:AbstractFloat} = a + (-b)
@inline Base.:-(a::_WeakBilinearTerm{D,T}, b::_WeakBilinearTerm{D,T}) where {D,T<:AbstractFloat} = a +
                                                                                                   (-b)

@inline function Base.:+(a::_WeakTestTerm, b::Number)
  iszero(b) && return a
  throw(ArgumentError("cell bilinear form produced a nonzero term independent of the test function"))
end

@inline Base.:+(a::Number, b::_WeakTestTerm) = b + a

@inline function Base.:+(a::_WeakTrialTerm, b::Number)
  iszero(b) && return a
  throw(ArgumentError("weak bilinear form is affine rather than linear in the trial function"))
end

@inline Base.:+(a::Number, b::_WeakTrialTerm) = b + a

@inline function Base.:+(a::_WeakBilinearTerm, b::Number)
  iszero(b) && return a
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test and trial functions"))
end

@inline Base.:+(a::Number, b::_WeakBilinearTerm) = b + a

@inline function Base.:+(a::_WeakBilinearTerm, b::_WeakTestTerm)
  iszero(b) && return a
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the trial function"))
end

@inline Base.:+(a::_WeakTestTerm, b::_WeakBilinearTerm) = b + a

@inline function Base.:+(a::_WeakBilinearTerm, b::_WeakTrialTerm)
  iszero(b) && return a
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test function"))
end

@inline Base.:+(a::_WeakTrialTerm, b::_WeakBilinearTerm) = b + a

@inline function Base.:+(a::_WeakTestTerm, b::_WeakTrialTerm)
  iszero(a) && return b
  iszero(b) && return a
  throw(ArgumentError("weak bilinear form produced separate nonzero test-only and trial-only terms"))
end

@inline Base.:+(a::_WeakTrialTerm, b::_WeakTestTerm) = b + a

@inline function Base.:-(a::_WeakTestTerm, b::Number)
  iszero(b) && return a
  throw(ArgumentError("cell bilinear form produced a nonzero term independent of the test function"))
end

@inline function Base.:-(a::Number, b::_WeakTestTerm)
  iszero(a) && return -b
  throw(ArgumentError("cell bilinear form produced a nonzero term independent of the test function"))
end

@inline function Base.:-(a::_WeakTrialTerm, b::Number)
  iszero(b) && return a
  throw(ArgumentError("weak bilinear form is affine rather than linear in the trial function"))
end

@inline function Base.:-(a::Number, b::_WeakTrialTerm)
  iszero(a) && return -b
  throw(ArgumentError("weak bilinear form is affine rather than linear in the trial function"))
end

@inline function Base.:-(a::_WeakBilinearTerm, b::Number)
  iszero(b) && return a
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test and trial functions"))
end

@inline function Base.:-(a::Number, b::_WeakBilinearTerm)
  iszero(a) && return -b
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test and trial functions"))
end

@inline Base.:-(a::_WeakBilinearTerm, b::_WeakTestTerm) = a + (-b)
@inline Base.:-(a::_WeakTestTerm, b::_WeakBilinearTerm) = a + (-b)
@inline Base.:-(a::_WeakBilinearTerm, b::_WeakTrialTerm) = a + (-b)
@inline Base.:-(a::_WeakTrialTerm, b::_WeakBilinearTerm) = a + (-b)
@inline Base.:-(a::_WeakTestTerm, b::_WeakTrialTerm) = a + (-b)
@inline Base.:-(a::_WeakTrialTerm, b::_WeakTestTerm) = a + (-b)

@inline function Base.:*(a::_WeakTestTerm{D,T}, b::Number) where {D,T<:AbstractFloat}
  scale = T(b)
  return _WeakTestTerm(scale * a.value_coefficient,
                       ntuple(axis -> scale * a.gradient_coefficients[axis], Val(D)))
end

@inline Base.:*(a::Number, b::_WeakTestTerm) = b * a

@inline function Base.:*(a::_WeakTrialTerm{T,V,G,N}, b::Number) where {T<:AbstractFloat,V,G,N}
  return _WeakTrialTerm{T,V,G,N}(T(b) * a.value)
end

@inline Base.:*(a::Number, b::_WeakTrialTerm) = b * a

@inline function Base.:*(a::_WeakBilinearTerm{D,T,V,G,N},
                         b::Number) where {D,T<:AbstractFloat,V,G,N}
  scale = T(b)
  return _WeakBilinearTerm{D,T,V,G,N}(scale * a.value_coefficient,
                                      ntuple(axis -> scale * a.gradient_coefficients[axis], Val(D)))
end

@inline Base.:*(a::Number, b::_WeakBilinearTerm) = b * a

@inline function Base.:*(a::_WeakTestTerm{D,T},
                         b::_WeakTrialTerm{T,V,G,N}) where {D,T<:AbstractFloat,V,G,N}
  return _WeakBilinearTerm{D,T,V,G,N}(b.value * a.value_coefficient,
                                      ntuple(axis -> b.value * a.gradient_coefficients[axis],
                                             Val(D)))
end

@inline Base.:*(a::_WeakTrialTerm, b::_WeakTestTerm) = b * a

function Base.:*(::_WeakTestTerm, ::_WeakTestTerm)
  throw(ArgumentError("cell bilinear form is nonlinear in the test function"))
end

function Base.:*(::_WeakTrialTerm, ::_WeakTrialTerm)
  throw(ArgumentError("weak bilinear form is nonlinear in the trial function"))
end

function Base.:*(::_WeakBilinearTerm, ::Union{_WeakTestTerm,_WeakTrialTerm,_WeakBilinearTerm})
  throw(ArgumentError("weak bilinear form is nonlinear in the test or trial function"))
end

Base.:*(a::Union{_WeakTestTerm,_WeakTrialTerm}, b::_WeakBilinearTerm) = b * a

@inline Base.muladd(a::_WeakTestTerm, b::Number, c::_WeakTestTerm) = a * b + c
@inline Base.muladd(a::_WeakTrialTerm, b::Number, c::_WeakTrialTerm) = a * b + c
@inline Base.muladd(a::_WeakBilinearTerm, b::Number, c::_WeakBilinearTerm) = a * b + c
@inline Base.muladd(a::_WeakTestTerm, b::Number, c::Number) = a * b + c
@inline Base.muladd(a::_WeakTrialTerm, b::Number, c::Number) = a * b + c
@inline Base.muladd(a::_WeakBilinearTerm, b::Number, c::Number) = a * b + c

@inline Base.:/(a::_WeakTestTerm{D,T}, b::Number) where {D,T<:AbstractFloat} = a * inv(T(b))
@inline Base.:/(a::_WeakTrialTerm{T}, b::Number) where {T<:AbstractFloat} = a * inv(T(b))
@inline Base.:/(a::_WeakBilinearTerm{D,T}, b::Number) where {D,T<:AbstractFloat} = a * inv(T(b))

function Base.:/(::Number, ::_WeakTrialTerm)
  throw(ArgumentError("weak bilinear form is nonlinear in the trial function"))
end

function Base.:/(::Number, ::_WeakTestTerm)
  throw(ArgumentError("cell bilinear form is nonlinear in the test function"))
end

function Base.:/(::Number, ::_WeakBilinearTerm)
  throw(ArgumentError("weak bilinear form is nonlinear in the test or trial function"))
end

function _throw_nonlinear_weak_quotient()
  throw(ArgumentError("weak bilinear form is nonlinear in the test or trial function"))
end

Base.:/(::_WeakTestTerm, ::_WeakTestTerm) = _throw_nonlinear_weak_quotient()
Base.:/(::_WeakTestTerm, ::_WeakTrialTerm) = _throw_nonlinear_weak_quotient()
Base.:/(::_WeakTestTerm, ::_WeakBilinearTerm) = _throw_nonlinear_weak_quotient()
Base.:/(::_WeakTrialTerm, ::_WeakTestTerm) = _throw_nonlinear_weak_quotient()
Base.:/(::_WeakTrialTerm, ::_WeakTrialTerm) = _throw_nonlinear_weak_quotient()
Base.:/(::_WeakTrialTerm, ::_WeakBilinearTerm) = _throw_nonlinear_weak_quotient()
Base.:/(::_WeakBilinearTerm, ::_WeakTestTerm) = _throw_nonlinear_weak_quotient()
Base.:/(::_WeakBilinearTerm, ::_WeakTrialTerm) = _throw_nonlinear_weak_quotient()
Base.:/(::_WeakBilinearTerm, ::_WeakBilinearTerm) = _throw_nonlinear_weak_quotient()

function Base.literal_pow(::typeof(^), a::_WeakTrialTerm, ::Val{P}) where {P}
  P == 1 && return a
  throw(ArgumentError("weak bilinear form is nonlinear in the trial function"))
end

function Base.literal_pow(::typeof(^), a::_WeakTestTerm, ::Val{P}) where {P}
  P == 1 && return a
  throw(ArgumentError("cell bilinear form is nonlinear in the test function"))
end

function Base.literal_pow(::typeof(^), a::_WeakBilinearTerm, ::Val{P}) where {P}
  P == 1 && return a
  throw(ArgumentError("weak bilinear form is nonlinear in the test or trial function"))
end

@inline Base.conj(a::_WeakTestTerm) = a
@inline Base.conj(a::_WeakTrialTerm) = a
@inline Base.conj(a::_WeakBilinearTerm) = a

"""
    inner(a, b)
    a ⋅ b

Return the Euclidean inner product used in weak-form callbacks.

For scalar arguments this is ordinary multiplication. For tuples, `inner`
contracts matching components and is therefore the usual `a · b` product for
gradients. For interface trace pairs it applies the product independently on
the minus and plus sides, allowing expressions such as
`inner(jump(grad(v)), average(grad(w)))`.

The operation is deliberately algebraic: it does not conjugate its first
argument and it expects matching tuple lengths. This matches the real-valued
finite-element forms supported by the current matrix-free kernels.
"""
@inline inner(a::Number, b::Number) = a * b

@inline inner(a::NTuple{N,<:Number}, b::NTuple{N,<:Number}) where {N} = _tuple_inner(a, b, Val(N))

@inline function _tuple_inner(a::NTuple{N,<:Number}, b::NTuple{N,<:Number}, ::Val{N}) where {N}
  result = zero(a[1] * b[1])

  for index in 1:N
    result += a[index] * b[index]
  end

  return result
end

@inline inner(a::_WeakTracePair, b::_WeakTracePair) = _WeakTracePair(inner(a.minus, b.minus),
                                                                     inner(a.plus, b.plus))
"""
    a ⋅ b

Unicode infix dot product for weak-form trace pairs.

For trace pairs, `a ⋅ b` applies [`inner`](@ref) independently on the minus and
plus sides and returns another trace pair. Ordinary scalar and tuple dot
products use `LinearAlgebra.dot`; this method extends the same notation to the
interface proxies used by `add_interface_bilinear!` and `add_interface_linear!`.
"""
@inline LinearAlgebra.dot(a::_WeakTracePair, b::_WeakTracePair) = inner(a, b)
@inline minus(pair::_WeakTracePair) = pair.minus
@inline plus(pair::_WeakTracePair) = pair.plus
@inline jump(pair::_WeakTracePair) = jump(pair.minus, pair.plus)
@inline average(pair::_WeakTracePair) = average(pair.minus, pair.plus)

"""
    avg(pair)
    avg(minus_value, plus_value)

Short alias for [`average`](@ref) in interface weak forms.

The function returns the arithmetic average across the two traces of an
interface quantity. It is provided as a compact notation for standard DG
expressions such as fluxes and penalty terms, while [`average`](@ref) remains
the descriptive spelling.
"""
@inline avg(pair::_WeakTracePair) = average(pair)
@inline avg(minus_value, plus_value) = average(minus_value, plus_value)

@inline _zero_like(value::Number) = zero(value)
@inline _zero_like(value::Tuple) = map(_zero_like, value)

@inline function _trace_values(basis::_WeakTraceBasisFunction)
  return basis.plus_side ? basis.plus_values : basis.minus_values
end

@inline function _trace_values(basis::_WeakTraceTestBasisFunction)
  return basis.plus_side ? basis.plus_values : basis.minus_values
end

@inline function _trace_values(basis::_WeakTraceTrialBasisFunction)
  return basis.plus_side ? basis.plus_values : basis.minus_values
end

@inline function _trace_shape_value(basis::Union{_WeakTraceBasisFunction,
                                                 _WeakTraceTestBasisFunction,
                                                 _WeakTraceTrialBasisFunction})
  values = _trace_values(basis)
  return shape_value(values, basis.field, basis.point_index, basis.mode_index)
end

@inline function _trace_shape_gradient(basis::Union{_WeakTraceBasisFunction,
                                                    _WeakTraceTestBasisFunction,
                                                    _WeakTraceTrialBasisFunction})
  values = _trace_values(basis)
  return shape_gradient(values, basis.field, basis.point_index, basis.mode_index)
end

@inline function _trace_shape_normal_gradient(basis::Union{_WeakTraceBasisFunction,
                                                           _WeakTraceTestBasisFunction,
                                                           _WeakTraceTrialBasisFunction})
  values = _trace_values(basis)
  return shape_normal_gradient(values, basis.field, basis.point_index, basis.mode_index)
end

function cell_matrix!(local_matrix::AbstractMatrix{T}, operator::_CellBilinearForm, values,
                      scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_matrix!(local_matrix, operator, values)
  return local_matrix
end

function face_matrix!(local_matrix::AbstractMatrix{T}, operator::_BoundaryBilinearForm, values,
                      scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_matrix!(local_matrix, operator, values)
  return local_matrix
end

function surface_matrix!(local_matrix::AbstractMatrix{T}, operator::_SurfaceBilinearForm, values,
                         scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_matrix!(local_matrix, operator, values)
  return local_matrix
end

function cell_apply!(local_result::AbstractVector{T}, operator::_CellBilinearForm, values,
                     local_coefficients::AbstractVector{T},
                     scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_apply!(local_result, operator, values, local_coefficients, scratch)
  return nothing
end

function face_apply!(local_result::AbstractVector{T}, operator::_BoundaryBilinearForm, values,
                     local_coefficients::AbstractVector{T},
                     scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_apply!(local_result, operator, values, local_coefficients, scratch)
  return nothing
end

function surface_apply!(local_result::AbstractVector{T}, operator::_SurfaceBilinearForm, values,
                        local_coefficients::AbstractVector{T},
                        scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_apply!(local_result, operator, values, local_coefficients, scratch)
  return nothing
end

function cell_diagonal!(local_diagonal::AbstractVector{T}, operator::_CellBilinearForm, values,
                        scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_diagonal!(local_diagonal, operator, values)
  return nothing
end

function face_diagonal!(local_diagonal::AbstractVector{T}, operator::_BoundaryBilinearForm, values,
                        scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_diagonal!(local_diagonal, operator, values)
  return nothing
end

function surface_diagonal!(local_diagonal::AbstractVector{T}, operator::_SurfaceBilinearForm,
                           values, scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_diagonal!(local_diagonal, operator, values)
  return nothing
end

function cell_rhs!(local_rhs::AbstractVector{T}, operator::_CellLinearForm, values,
                   scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_linear_rhs!(local_rhs, operator, values)
  return nothing
end

function face_rhs!(local_rhs::AbstractVector{T}, operator::_BoundaryLinearForm, values,
                   scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_linear_rhs!(local_rhs, operator, values)
  return nothing
end

function surface_rhs!(local_rhs::AbstractVector{T}, operator::_SurfaceLinearForm, values,
                      scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_linear_rhs!(local_rhs, operator, values)
  return nothing
end

function interface_matrix!(local_matrix::AbstractMatrix{T}, operator::_InterfaceBilinearForm,
                           values, scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_interface_bilinear_matrix!(local_matrix, operator, values)
  return local_matrix
end

function interface_apply!(local_result::AbstractVector{T}, operator::_InterfaceBilinearForm, values,
                          local_coefficients::AbstractVector{T},
                          scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_interface_bilinear_apply!(local_result, operator, values, local_coefficients, scratch)
  return nothing
end

function interface_diagonal!(local_diagonal::AbstractVector{T}, operator::_InterfaceBilinearForm,
                             values, scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_interface_bilinear_diagonal!(local_diagonal, operator, values)
  return nothing
end

function interface_rhs!(local_rhs::AbstractVector{T}, operator::_InterfaceLinearForm, values,
                        scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_interface_linear_rhs!(local_rhs, operator, values)
  return nothing
end

# Dense local matrices and diagonals intentionally use symbolic test and trial
# basis proxies as well. This keeps the bilinearity contract identical to the
# matrix-free apply path instead of letting invalid forms pass in diagnostic
# assembly but fail during production application.
function _accumulate_cell_bilinear_matrix!(local_matrix::AbstractMatrix{T},
                                           operator::_LocalBilinearForm,
                                           values) where {T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_component in 1:component_count(test_field)
      for test_mode in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, test_component, test_mode)
        test = _WeakTestBasisFunction(values, test_field, point_index, test_component, test_mode)

        for trial_component in 1:component_count(trial_field)
          for trial_mode in 1:local_mode_count(values, trial_field)
            column = local_dof_index(values, trial_field, trial_component, trial_mode)
            trial = _WeakTrialBasisFunction(values, trial_field, point_index, trial_component,
                                            trial_mode)
            local_matrix[row, column] += weighted *
                                         _weak_bilinear_value(operator.form(q, test, trial))
          end
        end
      end
    end
  end

  return local_matrix
end

# The one-sided bilinear apply dispatcher selects the strongest available
# lowering. Full tensor-product test and trial data use sum factorization on
# both sides; a tensor trial with a non-tensor test keeps only the exact
# test-basis loop; all other cases reconstruct the trial field pointwise.
function _accumulate_cell_bilinear_apply!(local_result::AbstractVector{T},
                                          operator::_LocalBilinearForm, values,
                                          local_coefficients::AbstractVector{T},
                                          scratch::KernelScratch{T}) where {T<:AbstractFloat}
  tensor = tensor_values(values, operator.trial_field)

  if tensor !== nothing
    return _accumulate_cell_bilinear_apply_tensor_trial!(local_result, operator, values,
                                                         local_coefficients, tensor, scratch)
  end

  return _accumulate_cell_bilinear_apply_reconstructed_trial!(local_result, operator, values,
                                                              local_coefficients)
end

# Reconstruct the scalar value, physical gradient, and, on normal-bearing
# integration items, normal derivative of one trial component at one quadrature
# point. This represents the local finite-element function `u_h`, not a basis
# function.
function _trial_function(values::_FieldEvaluationValues, local_coefficients::AbstractVector{T},
                         field::AbstractField, component::Int,
                         point_index::Int) where {T<:AbstractFloat}
  data = _field_values(values, field)
  value_data = _local_field_value_component(data, local_coefficients, component, point_index)
  gradient_data = _local_field_gradient(data, local_coefficients, component, point_index)
  normal_data = _normal_trial_gradient(values, data, local_coefficients, component, point_index)
  return _WeakTrialFunction(field, component, value_data, gradient_data, normal_data)
end

@inline _normal_trial_gradient(values, data, local_coefficients, component, point_index) = nothing

@inline function _normal_trial_gradient(values::_NormalEvaluationValues, data::_FieldValues{D,T},
                                        local_coefficients::AbstractVector{T}, component::Int,
                                        point_index::Int) where {D,T<:AbstractFloat}
  return _local_field_normal_gradient(data, local_coefficients, component, point_index,
                                      _point_normal(values, point_index))
end

@inline function _trace_trial_function(minus_values::_FieldEvaluationValues,
                                       plus_values::_FieldEvaluationValues,
                                       local_coefficients::AbstractVector{T}, field::AbstractField,
                                       component::Int, point_index::Int) where {T<:AbstractFloat}
  minus_trial = _trial_function(minus_values, local_coefficients, field, component, point_index)
  plus_trial = _trial_function(plus_values, local_coefficients, field, component, point_index)
  return _WeakTraceTrialFunction(field, component, minus_trial, plus_trial)
end

@inline function _weak_test_function(::Val{D}, ::Type{T}, field::AbstractField,
                                     component::Int) where {D,T<:AbstractFloat}
  return _WeakTestFunction{D,T,typeof(field),Nothing}(field, component, nothing)
end

@inline function _weak_test_function(values::_NormalEvaluationValues, ::Val{D}, ::Type{T},
                                     field::AbstractField, component::Int,
                                     point_index::Int) where {D,T<:AbstractFloat}
  normal_data = _point_normal(values, point_index)
  return _WeakTestFunction{D,T,typeof(field),typeof(normal_data)}(field, component, normal_data)
end

@inline function _weak_test_function(values, ::Val{D}, ::Type{T}, field::AbstractField,
                                     component::Int, point_index::Int) where {D,T<:AbstractFloat}
  return _weak_test_function(Val(D), T, field, component)
end

@inline function _weak_trace_test_function(::Val{D}, ::Type{T}, field::AbstractField,
                                           component::Int, plus_side::Bool,
                                           normal_data::NTuple{D,T}) where {D,T<:AbstractFloat}
  return _WeakTraceTestFunction{D,T,typeof(field),typeof(normal_data)}(field, component, plus_side,
                                                                       normal_data)
end

# Symbolic trial probes use unit value and unit gradient data. Their numerical
# values are irrelevant; only the propagated type flags matter. Evaluating the
# user callback on these probes lets tensorized kernels decide whether they must
# interpolate `u_h`, compute `∇u_h`, or both.
@inline function _symbolic_trial_function(values, ::Val{D}, ::Type{T}, field::AbstractField,
                                          component::Int,
                                          point_index::Int) where {D,T<:AbstractFloat}
  normal_data = values isa _NormalEvaluationValues ? one(T) : nothing
  return _WeakTrialFunction(field, component, one(T), ntuple(_ -> one(T), Val(D)), normal_data)
end

@inline function _symbolic_trace_trial_function(minus_values::_InterfaceSideValues{D,T},
                                                plus_values::_InterfaceSideValues{D,T},
                                                field::AbstractField, component::Int,
                                                point_index::Int) where {D,T<:AbstractFloat}
  minus_trial = _symbolic_trial_function(minus_values, Val(D), T, field, component, point_index)
  plus_trial = _symbolic_trial_function(plus_values, Val(D), T, field, component, point_index)
  return _WeakTraceTrialFunction(field, component, minus_trial, plus_trial)
end

@inline function _merge_trial_channel_flags(current::Tuple{Bool,Bool}, term::_WeakBilinearTerm)
  value_used = current[1] || _trial_uses_value(typeof(term))
  gradient_used = current[2] ||
                  _trial_uses_gradient(typeof(term)) ||
                  _trial_uses_normal(typeof(term))
  return (value_used, gradient_used)
end

@inline function _merge_trial_channel_flags(current::Tuple{Bool,Bool}, term::_WeakTestTerm)
  _test_value_coefficient(term)
  return current
end

@inline function _merge_trial_channel_flags(current::Tuple{Bool,Bool}, term::_WeakTrialTerm)
  _test_value_coefficient(term)
  return current
end

@inline function _merge_trial_channel_flags(current::Tuple{Bool,Bool}, term::Number)
  _test_value_coefficient(term)
  return current
end

# Scan a tensorizable cell form once with symbolic trial probes. This prepass is
# deliberately done per local integration item because user callbacks may branch
# on component, point data, or geometric information. It preserves correctness
# while avoiding unnecessary value or gradient interpolation in the hot loop.
function _cell_tensor_trial_channels(operator::_LocalBilinearForm, values, ::Val{D},
                                     ::Type{T}) where {D,T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  channels = (false, false)

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)

    for trial_component in 1:component_count(trial_field)
      trial = _symbolic_trial_function(values, Val(D), T, trial_field, trial_component, point_index)

      for test_component in 1:component_count(test_field)
        test = _weak_test_function(values, Val(D), T, test_field, test_component, point_index)
        channels = _merge_trial_channel_flags(channels, operator.form(q, test, trial))
      end
    end
  end

  return channels
end

# Interface tensorization performs the same channel analysis on both trace
# sides. A normal derivative uses the gradient channel, because ∂ₙw is computed
# from `∇w · n` after the trial gradient has been sum-factorized.
function _interface_tensor_trial_channels(operator::_InterfaceBilinearForm, values,
                                          minus_values::_InterfaceSideValues{D,T},
                                          plus_values::_InterfaceSideValues{D,T}) where {D,
                                                                                         T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  normal_data = normal(values)
  channels = (false, false)

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)

    for trial_component in 1:component_count(trial_field)
      trial = _symbolic_trace_trial_function(minus_values, plus_values, trial_field,
                                             trial_component, point_index)

      for test_component in 1:component_count(test_field)
        minus_test = _weak_trace_test_function(Val(D), T, test_field, test_component, false,
                                               normal_data)
        channels = _merge_trial_channel_flags(channels, operator.form(q, minus_test, trial))
        plus_test = _weak_trace_test_function(Val(D), T, test_field, test_component, true,
                                              normal_data)
        channels = _merge_trial_channel_flags(channels, operator.form(q, plus_test, trial))
      end
    end
  end

  return channels
end

# Apply a bilinear form after reconstructing each trial component u_hᶜ at the
# quadrature point. This is the general non-tensor fallback: it preserves the
# component-by-component weak-form semantics but removes the trial-mode loop.
function _accumulate_cell_bilinear_apply_reconstructed_trial!(local_result::AbstractVector{T},
                                                              operator::_LocalBilinearForm, values,
                                                              local_coefficients::AbstractVector{T}) where {T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)

    for trial_component in 1:component_count(trial_field)
      trial = _trial_function(values, local_coefficients, trial_field, trial_component, point_index)

      for test_component in 1:component_count(test_field)
        for test_mode in 1:local_mode_count(values, test_field)
          row = local_dof_index(values, test_field, test_component, test_mode)
          test = _WeakTestBasisFunction(values, test_field, point_index, test_component, test_mode)
          local_result[row] += weighted * _weak_bilinear_value(operator.form(q, test, trial))
        end
      end
    end
  end

  return local_result
end

# Tensor-product kernels work on dense per-axis mode boxes. A compiled field may
# store only active modes from that box, so this routine scatters one component's
# local coefficients into the tensor layout expected by interpolation and
# gradient kernels.
function _copy_tensor_component_box_coefficients!(result::AbstractVector{T},
                                                  tensor::TensorProductValues{D,T}, values,
                                                  field::AbstractField, component::Int,
                                                  local_coefficients::AbstractVector{T}) where {D,
                                                                                                T<:AbstractFloat}
  data = _field_values(values, field)
  offset = first(data.block) + _field_component_offset(data, component) - 1
  fill!(result, zero(T))
  box_shape = tensor_mode_shape(tensor)

  @inbounds for mode_index in 1:tensor_mode_count(tensor)
    mode = tensor.local_modes[mode_index]
    box_indices = ntuple(axis -> mode[axis] + 1, Val(D))
    box_index = _tensor_shape_linear_index(box_indices, box_shape)
    result[box_index] = local_coefficients[offset+mode_index]
  end

  return result
end

# The transpose operation gathers a tensor-box contribution back to the compact
# local dof ordering used by the field layout. This is the adjoint of
# `_copy_tensor_component_box_coefficients!` with respect to the stored local
# coefficient numbering.
function _add_tensor_component_box_output!(local_result::AbstractVector{T}, values,
                                           field::AbstractField, component::Int,
                                           tensor::TensorProductValues{D,T},
                                           contribution::AbstractVector{T}) where {D,
                                                                                   T<:AbstractFloat}
  data = _field_values(values, field)
  offset = first(data.block) + _field_component_offset(data, component) - 1
  box_shape = tensor_mode_shape(tensor)

  @inbounds for mode_index in 1:tensor_mode_count(tensor)
    mode = tensor.local_modes[mode_index]
    box_indices = ntuple(axis -> mode[axis] + 1, Val(D))
    box_index = _tensor_shape_linear_index(box_indices, box_shape)
    local_result[offset+mode_index] += contribution[box_index]
  end

  return local_result
end

@inline _tensor_trial_normal(values, point_index) = nothing

@inline function _tensor_trial_normal(values::_NormalEvaluationValues, point_index::Int)
  return _point_normal(values, point_index)
end

@inline function _tensor_trial_function(values, ::Val{D}, field::AbstractField, point_index::Int,
                                        component::Int, point_values::AbstractVector{T},
                                        point_gradients::AbstractMatrix{T}) where {D,
                                                                                   T<:AbstractFloat}
  normal_data = _tensor_trial_normal(values, point_index)
  return _WeakTensorTrialFunction{D,T,typeof(field),typeof(normal_data),typeof(point_values),
                                  typeof(point_gradients)}(field, component, point_index,
                                                           point_values, point_gradients,
                                                           normal_data)
end

@inline function _tensor_trace_trial_function(minus_values::_InterfaceSideValues{D,T},
                                              plus_values::_InterfaceSideValues{D,T},
                                              field::AbstractField, point_index::Int,
                                              component::Int, minus_point_values::AbstractVector{T},
                                              plus_point_values::AbstractVector{T},
                                              minus_point_gradients::AbstractMatrix{T},
                                              plus_point_gradients::AbstractMatrix{T}) where {D,
                                                                                              T<:AbstractFloat}
  minus_trial = _tensor_trial_function(minus_values, Val(D), field, point_index, component,
                                       minus_point_values, minus_point_gradients)
  plus_trial = _tensor_trial_function(plus_values, Val(D), field, point_index, component,
                                      plus_point_values, plus_point_gradients)
  return _WeakTraceTrialFunction(field, component, minus_trial, plus_trial)
end

function _accumulate_cell_bilinear_apply_tensor_trial!(local_result::AbstractVector{T},
                                                       operator::_LocalBilinearForm, values,
                                                       local_coefficients::AbstractVector{T},
                                                       tensor::TensorProductValues{D,T},
                                                       scratch::KernelScratch{T}) where {D,
                                                                                         T<:AbstractFloat}
  test_tensor = tensor_values(values, operator.test_field)
  if test_tensor !== nothing
    return _accumulate_cell_bilinear_apply_tensor_project!(local_result, operator, values,
                                                           local_coefficients, test_tensor, tensor,
                                                           scratch)
  end

  return _accumulate_cell_bilinear_apply_tensor_test_basis!(local_result, operator, values,
                                                            local_coefficients, tensor, scratch)
end

# Tensorize only the trial side when the test field does not provide a full
# tensor-product mode box. This still avoids the expensive trial-mode loop, but
# it keeps the exact basis loop on the test side.
function _accumulate_cell_bilinear_apply_tensor_test_basis!(local_result::AbstractVector{T},
                                                            operator::_LocalBilinearForm, values,
                                                            local_coefficients::AbstractVector{T},
                                                            tensor::TensorProductValues{D,T},
                                                            scratch::KernelScratch{T}) where {D,
                                                                                              T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  use_trial_value, use_trial_gradient = _cell_tensor_trial_channels(operator, values, Val(D), T)
  trial_coefficients = scratch_vector(scratch, D + 2, tensor_mode_box_count(tensor))
  point_values = scratch_vector(scratch, D + 3, use_trial_value ? tensor_point_count(tensor) : 0)
  point_gradients = scratch_matrix(scratch, 1, D,
                                   use_trial_gradient ? tensor_point_count(tensor) : 0)

  for trial_component in 1:component_count(trial_field)
    _copy_tensor_component_box_coefficients!(trial_coefficients, tensor, values, trial_field,
                                             trial_component, local_coefficients)
    use_trial_value && tensor_box_interpolate!(point_values, tensor, trial_coefficients, scratch)
    use_trial_gradient && tensor_box_gradient!(point_gradients, tensor, trial_coefficients, scratch)

    for point_index in 1:point_count(values)
      q = _WeakQuadraturePoint(values, point_index)
      weighted = weight(q)
      trial = _tensor_trial_function(values, Val(D), trial_field, point_index, trial_component,
                                     point_values, point_gradients)

      for test_component in 1:component_count(test_field)
        for test_mode in 1:local_mode_count(values, test_field)
          row = local_dof_index(values, test_field, test_component, test_mode)
          test = _WeakTestBasisFunction(values, test_field, point_index, test_component, test_mode)
          local_result[row] += weighted * _weak_bilinear_value(operator.form(q, test, trial))
        end
      end
    end
  end

  return local_result
end

# Project one symbolic test component back to the local coefficient block. The
# pointwise matrices are stored component-major by column so each component can
# be copied into the contiguous buffers expected by the tensor projection
# helpers without allocating inside the cell kernel.
function _project_tensor_test_component!(local_result::AbstractVector{T}, values,
                                         field::AbstractField, component::Int,
                                         tensor::TensorProductValues{D,T},
                                         weighted_values_by_component::AbstractMatrix{T},
                                         weighted_gradients_by_component::AbstractMatrix{T},
                                         scratch::KernelScratch{T}) where {D,T<:AbstractFloat}
  point_total = tensor_point_count(tensor)
  contribution = scratch_vector(scratch, D + 4, tensor_mode_box_count(tensor))
  weighted_values = scratch_vector(scratch, D + 5, point_total)
  weighted_gradients = scratch_matrix(scratch, 4, D, point_total)
  fill!(contribution, zero(T))
  has_value = false
  has_gradient = false

  @inbounds for point_index in 1:point_total
    value_coefficient = weighted_values_by_component[point_index, component]
    weighted_values[point_index] = value_coefficient
    has_value |= !iszero(value_coefficient)

    for axis in 1:D
      source_column = (component - 1) * D + axis
      gradient_coefficient = weighted_gradients_by_component[point_index, source_column]
      weighted_gradients[axis, point_index] = gradient_coefficient
      has_gradient |= !iszero(gradient_coefficient)
    end
  end

  has_value && tensor_box_project!(contribution, tensor, weighted_values, scratch)
  has_gradient && tensor_box_project_gradient!(contribution, tensor, weighted_gradients, scratch)
  _add_tensor_component_box_output!(local_result, values, field, component, tensor, contribution)
  return local_result
end

# Extract the coefficients of the symbolic test functional. A valid bilinear
# term has already combined a test marker with a trial marker. Any remaining
# nonzero pure test, pure trial, or scalar term means the callback is not a
# bilinear form, so these accessors are the final contract boundary before
# projection or dense local assembly.
@inline _test_value_coefficient(term::_WeakBilinearTerm) = term.value_coefficient

@inline function _test_value_coefficient(term::_WeakTestTerm)
  iszero(term) && return term.value_coefficient
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the trial function"))
end

@inline function _test_value_coefficient(term::_WeakTrialTerm)
  iszero(term) && return term.value
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test function"))
end

@inline function _test_value_coefficient(term::Number)
  iszero(term) && return term
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test function"))
end

@inline _test_gradient_coefficient(term::_WeakBilinearTerm, axis::Int) = term.gradient_coefficients[axis]

@inline function _test_gradient_coefficient(term::_WeakTestTerm, axis::Int)
  iszero(term) && return term.gradient_coefficients[axis]
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the trial function"))
end

@inline function _test_gradient_coefficient(term::_WeakTrialTerm, axis::Int)
  iszero(term) && return term.value
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test function"))
end

@inline function _test_gradient_coefficient(term::Number, axis::Int)
  iszero(term) && return term
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test function"))
end

@inline function _weak_bilinear_value(term::_WeakTrialTerm)
  iszero(term) && return term.value
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test function"))
end

@inline function _weak_bilinear_value(term::_WeakBilinearTerm)
  return term.value_coefficient + sum(term.gradient_coefficients)
end

@inline function _weak_bilinear_value(term::_WeakTestTerm)
  iszero(term) && return term.value_coefficient
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the trial function"))
end

@inline function _weak_bilinear_value(term::Number)
  iszero(term) && return term
  throw(ArgumentError("weak bilinear form produced a nonzero term independent of the test or trial function"))
end

# Scalar cell forms use the same symbolic-test representation as vector forms,
# but they can keep the weighted value and gradient data in the contiguous
# buffers consumed directly by the tensor projection helpers.
function _accumulate_cell_bilinear_apply_tensor_project_scalar!(local_result::AbstractVector{T},
                                                                operator::_LocalBilinearForm,
                                                                values,
                                                                local_coefficients::AbstractVector{T},
                                                                test_tensor::TensorProductValues{D,
                                                                                                 T},
                                                                trial_tensor::TensorProductValues{D,
                                                                                                  T},
                                                                scratch::KernelScratch{T}) where {D,
                                                                                                  T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  use_trial_value, use_trial_gradient = _cell_tensor_trial_channels(operator, values, Val(D), T)
  trial_coefficients = scratch_vector(scratch, D + 2, tensor_mode_box_count(trial_tensor))
  trial_values = scratch_vector(scratch, D + 3,
                                use_trial_value ? tensor_point_count(trial_tensor) : 0)
  weighted_values = scratch_vector(scratch, D + 4, tensor_point_count(test_tensor))
  contribution = scratch_vector(scratch, D + 5, tensor_mode_box_count(test_tensor))
  trial_gradients = scratch_matrix(scratch, 1, D,
                                   use_trial_gradient ? tensor_point_count(trial_tensor) : 0)
  weighted_gradients = scratch_matrix(scratch, 2, D, tensor_point_count(test_tensor))
  _copy_tensor_component_box_coefficients!(trial_coefficients, trial_tensor, values, trial_field, 1,
                                           local_coefficients)
  use_trial_value &&
    tensor_box_interpolate!(trial_values, trial_tensor, trial_coefficients, scratch)
  use_trial_gradient &&
    tensor_box_gradient!(trial_gradients, trial_tensor, trial_coefficients, scratch)
  fill!(contribution, zero(T))
  has_test_value = false
  has_test_gradient = false

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)
    test = _weak_test_function(values, Val(D), T, test_field, 1, point_index)
    trial = _tensor_trial_function(values, Val(D), trial_field, point_index, 1, trial_values,
                                   trial_gradients)
    term = operator.form(q, test, trial)
    value_coefficient = T(_test_value_coefficient(term))
    weighted_values[point_index] = weighted * value_coefficient
    has_test_value |= !iszero(value_coefficient)

    for axis in 1:D
      gradient_coefficient = T(_test_gradient_coefficient(term, axis))
      weighted_gradients[axis, point_index] = weighted * gradient_coefficient
      has_test_gradient |= !iszero(gradient_coefficient)
    end
  end

  has_test_value && tensor_box_project!(contribution, test_tensor, weighted_values, scratch)
  has_test_gradient &&
    tensor_box_project_gradient!(contribution, test_tensor, weighted_gradients, scratch)
  _add_tensor_component_box_output!(local_result, values, test_field, 1, test_tensor, contribution)
  return local_result
end

# Fully tensorized cell apply for scalar and vector fields. Each trial
# component is reconstructed by sum factorization; the weak form is then
# evaluated against one symbolic test component at a time and projected with the
# transpose tensor kernels. This represents the same local matrix-vector product
# as the basis-loop implementation for bilinear forms.
function _accumulate_cell_bilinear_apply_tensor_project!(local_result::AbstractVector{T},
                                                         operator::_LocalBilinearForm, values,
                                                         local_coefficients::AbstractVector{T},
                                                         test_tensor::TensorProductValues{D,T},
                                                         trial_tensor::TensorProductValues{D,T},
                                                         scratch::KernelScratch{T}) where {D,
                                                                                           T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  if component_count(test_field) == 1 && component_count(trial_field) == 1
    return _accumulate_cell_bilinear_apply_tensor_project_scalar!(local_result, operator, values,
                                                                  local_coefficients, test_tensor,
                                                                  trial_tensor, scratch)
  end

  trial_coefficients = scratch_vector(scratch, D + 2, tensor_mode_box_count(trial_tensor))
  use_trial_value, use_trial_gradient = _cell_tensor_trial_channels(operator, values, Val(D), T)
  trial_values = scratch_vector(scratch, D + 3,
                                use_trial_value ? tensor_point_count(trial_tensor) : 0)
  trial_gradients = scratch_matrix(scratch, 1, D,
                                   use_trial_gradient ? tensor_point_count(trial_tensor) : 0)
  test_components = component_count(test_field)
  weighted_values = scratch_matrix(scratch, 2, tensor_point_count(test_tensor), test_components)
  weighted_gradients = scratch_matrix(scratch, 3, tensor_point_count(test_tensor),
                                      D * test_components)
  fill!(weighted_values, zero(T))
  fill!(weighted_gradients, zero(T))

  for trial_component in 1:component_count(trial_field)
    _copy_tensor_component_box_coefficients!(trial_coefficients, trial_tensor, values, trial_field,
                                             trial_component, local_coefficients)
    use_trial_value &&
      tensor_box_interpolate!(trial_values, trial_tensor, trial_coefficients, scratch)
    use_trial_gradient &&
      tensor_box_gradient!(trial_gradients, trial_tensor, trial_coefficients, scratch)

    for point_index in 1:point_count(values)
      q = _WeakQuadraturePoint(values, point_index)
      weighted = weight(q)
      trial = _tensor_trial_function(values, Val(D), trial_field, point_index, trial_component,
                                     trial_values, trial_gradients)

      for test_component in 1:test_components
        test = _weak_test_function(values, Val(D), T, test_field, test_component, point_index)
        term = operator.form(q, test, trial)
        weighted_values[point_index, test_component] += weighted * T(_test_value_coefficient(term))

        for axis in 1:D
          target_column = (test_component - 1) * D + axis
          weighted_gradients[point_index, target_column] += weighted *
                                                            T(_test_gradient_coefficient(term,
                                                                                         axis))
        end
      end
    end
  end

  for test_component in 1:test_components
    _project_tensor_test_component!(local_result, values, test_field, test_component, test_tensor,
                                    weighted_values, weighted_gradients, scratch)
  end

  return local_result
end

function _accumulate_cell_bilinear_diagonal!(local_diagonal::AbstractVector{T},
                                             operator::_LocalBilinearForm,
                                             values) where {T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  _field_id(test_field) == _field_id(trial_field) || return local_diagonal

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_component in 1:component_count(test_field)
      for test_mode in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, test_component, test_mode)
        test = _WeakTestBasisFunction(values, test_field, point_index, test_component, test_mode)
        trial = _WeakTrialBasisFunction(values, trial_field, point_index, test_component, test_mode)
        local_diagonal[row] += weighted * _weak_bilinear_value(operator.form(q, test, trial))
      end
    end
  end

  return local_diagonal
end

function _accumulate_cell_linear_rhs!(local_rhs::AbstractVector{T}, operator::_LocalLinearForm,
                                      values) where {T<:AbstractFloat}
  test_field = operator.test_field

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_component in 1:component_count(test_field)
      for test_mode in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, test_component, test_mode)
        test = _WeakBasisFunction(values, test_field, point_index, test_component, test_mode)
        local_rhs[row] += weighted * operator.form(q, test)
      end
    end
  end

  return local_rhs
end

function _accumulate_interface_bilinear_matrix!(local_matrix::AbstractMatrix{T},
                                                operator::_InterfaceBilinearForm,
                                                values) where {T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  minus_values = minus(values)
  plus_values = plus(values)

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_side in (false, true)
      test_values = test_side ? plus_values : minus_values

      for test_component in 1:component_count(test_field)
        for test_mode in 1:local_mode_count(test_values, test_field)
          row = local_dof_index(test_values, test_field, test_component, test_mode)
          test = _WeakTraceTestBasisFunction(minus_values, plus_values, test_field, point_index,
                                             test_component, test_mode, test_side)

          for trial_side in (false, true)
            trial_values = trial_side ? plus_values : minus_values

            for trial_component in 1:component_count(trial_field)
              for trial_mode in 1:local_mode_count(trial_values, trial_field)
                column = local_dof_index(trial_values, trial_field, trial_component, trial_mode)
                trial = _WeakTraceTrialBasisFunction(minus_values, plus_values, trial_field,
                                                     point_index, trial_component, trial_mode,
                                                     trial_side)
                local_matrix[row, column] += weighted *
                                             _weak_bilinear_value(operator.form(q, test, trial))
              end
            end
          end
        end
      end
    end
  end

  return local_matrix
end

function _accumulate_interface_bilinear_apply!(local_result::AbstractVector{T},
                                               operator::_InterfaceBilinearForm, values,
                                               local_coefficients::AbstractVector{T},
                                               scratch::KernelScratch{T}) where {T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  minus_values = minus(values)
  plus_values = plus(values)
  minus_trial_tensor = tensor_values(minus_values, trial_field)
  plus_trial_tensor = tensor_values(plus_values, trial_field)
  minus_test_tensor = tensor_values(minus_values, test_field)
  plus_test_tensor = tensor_values(plus_values, test_field)

  if minus_trial_tensor !== nothing &&
     plus_trial_tensor !== nothing &&
     minus_test_tensor !== nothing &&
     plus_test_tensor !== nothing
    return _accumulate_interface_bilinear_apply_tensor_project!(local_result, operator, values,
                                                                local_coefficients, minus_values,
                                                                plus_values, minus_test_tensor,
                                                                plus_test_tensor,
                                                                minus_trial_tensor,
                                                                plus_trial_tensor, scratch)
  end

  return _accumulate_interface_bilinear_apply_reconstructed_trial!(local_result, operator, values,
                                                                   local_coefficients, minus_values,
                                                                   plus_values)
end

# General interface fallback: reconstruct both one-sided traces of u_h once per
# quadrature point and trial component, then keep the exact test-basis loop.
function _accumulate_interface_bilinear_apply_reconstructed_trial!(local_result::AbstractVector{T},
                                                                   operator::_InterfaceBilinearForm,
                                                                   values,
                                                                   local_coefficients::AbstractVector{T},
                                                                   minus_values,
                                                                   plus_values) where {T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)

    for trial_component in 1:component_count(trial_field)
      trial = _trace_trial_function(minus_values, plus_values, local_coefficients, trial_field,
                                    trial_component, point_index)

      for test_side in (false, true)
        test_values = test_side ? plus_values : minus_values

        for test_component in 1:component_count(test_field)
          for test_mode in 1:local_mode_count(test_values, test_field)
            row = local_dof_index(test_values, test_field, test_component, test_mode)
            test = _WeakTraceTestBasisFunction(minus_values, plus_values, test_field, point_index,
                                               test_component, test_mode, test_side)
            local_result[row] += weighted * _weak_bilinear_value(operator.form(q, test, trial))
          end
        end
      end
    end
  end

  return local_result
end

# Fully tensorized two-sided trace apply. Each side is interpolated from its
# local tensor coefficients, the interface form is evaluated with a symbolic
# trace test on one side, and the resulting value/gradient coefficients are
# projected back with the transpose tensor kernels for that side.
function _accumulate_interface_bilinear_apply_tensor_project!(local_result::AbstractVector{T},
                                                              operator::_InterfaceBilinearForm,
                                                              values,
                                                              local_coefficients::AbstractVector{T},
                                                              minus_values::_InterfaceSideValues{D,
                                                                                                 T},
                                                              plus_values::_InterfaceSideValues{D,
                                                                                                T},
                                                              minus_test_tensor::TensorProductValues{D,
                                                                                                     T},
                                                              plus_test_tensor::TensorProductValues{D,
                                                                                                    T},
                                                              minus_trial_tensor::TensorProductValues{D,
                                                                                                      T},
                                                              plus_trial_tensor::TensorProductValues{D,
                                                                                                     T},
                                                              scratch::KernelScratch{T}) where {D,
                                                                                                T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  trial_components = component_count(trial_field)
  test_components = component_count(test_field)
  point_total = point_count(values)
  use_trial_value, use_trial_gradient = _interface_tensor_trial_channels(operator, values,
                                                                         minus_values, plus_values)
  minus_trial_coefficients = scratch_vector(scratch, D + 2,
                                            tensor_mode_box_count(minus_trial_tensor))
  plus_trial_coefficients = scratch_vector(scratch, D + 3, tensor_mode_box_count(plus_trial_tensor))
  minus_trial_values = scratch_vector(scratch, D + 4,
                                      use_trial_value ? tensor_point_count(minus_trial_tensor) : 0)
  plus_trial_values = scratch_vector(scratch, D + 5,
                                     use_trial_value ? tensor_point_count(plus_trial_tensor) : 0)
  minus_trial_gradients = scratch_matrix(scratch, 1, D,
                                         use_trial_gradient ?
                                         tensor_point_count(minus_trial_tensor) : 0)
  plus_trial_gradients = scratch_matrix(scratch, 2, D,
                                        use_trial_gradient ? tensor_point_count(plus_trial_tensor) :
                                        0)
  minus_weighted_values = scratch_matrix(scratch, 3, point_total, test_components)
  plus_weighted_values = scratch_matrix(scratch, 7, point_total, test_components)
  minus_weighted_gradients = scratch_matrix(scratch, 5, point_total, D * test_components)
  plus_weighted_gradients = scratch_matrix(scratch, 6, point_total, D * test_components)
  normal_data = normal(values)
  fill!(minus_weighted_values, zero(T))
  fill!(plus_weighted_values, zero(T))
  fill!(minus_weighted_gradients, zero(T))
  fill!(plus_weighted_gradients, zero(T))

  for trial_component in 1:trial_components
    _copy_tensor_component_box_coefficients!(minus_trial_coefficients, minus_trial_tensor,
                                             minus_values, trial_field, trial_component,
                                             local_coefficients)
    _copy_tensor_component_box_coefficients!(plus_trial_coefficients, plus_trial_tensor,
                                             plus_values, trial_field, trial_component,
                                             local_coefficients)
    use_trial_value &&
      tensor_box_interpolate!(minus_trial_values, minus_trial_tensor, minus_trial_coefficients,
                              scratch)
    use_trial_value &&
      tensor_box_interpolate!(plus_trial_values, plus_trial_tensor, plus_trial_coefficients,
                              scratch)
    use_trial_gradient &&
      tensor_box_gradient!(minus_trial_gradients, minus_trial_tensor, minus_trial_coefficients,
                           scratch)
    use_trial_gradient &&
      tensor_box_gradient!(plus_trial_gradients, plus_trial_tensor, plus_trial_coefficients,
                           scratch)

    for point_index in 1:point_total
      q = _WeakQuadraturePoint(values, point_index)
      weighted = weight(q)
      trial = _tensor_trace_trial_function(minus_values, plus_values, trial_field, point_index,
                                           trial_component, minus_trial_values, plus_trial_values,
                                           minus_trial_gradients, plus_trial_gradients)

      for test_component in 1:test_components
        minus_test = _weak_trace_test_function(Val(D), T, test_field, test_component, false,
                                               normal_data)
        minus_term = operator.form(q, minus_test, trial)
        minus_weighted_values[point_index, test_component] += weighted *
                                                              T(_test_value_coefficient(minus_term))

        plus_test = _weak_trace_test_function(Val(D), T, test_field, test_component, true,
                                              normal_data)
        plus_term = operator.form(q, plus_test, trial)
        plus_weighted_values[point_index, test_component] += weighted *
                                                             T(_test_value_coefficient(plus_term))

        for axis in 1:D
          target_column = (test_component - 1) * D + axis
          minus_weighted_gradients[point_index, target_column] += weighted *
                                                                  T(_test_gradient_coefficient(minus_term,
                                                                                               axis))
          plus_weighted_gradients[point_index, target_column] += weighted *
                                                                 T(_test_gradient_coefficient(plus_term,
                                                                                              axis))
        end
      end
    end
  end

  for test_component in 1:test_components
    _project_tensor_test_component!(local_result, minus_values, test_field, test_component,
                                    minus_test_tensor, minus_weighted_values,
                                    minus_weighted_gradients, scratch)
    _project_tensor_test_component!(local_result, plus_values, test_field, test_component,
                                    plus_test_tensor, plus_weighted_values, plus_weighted_gradients,
                                    scratch)
  end

  return local_result
end

function _accumulate_interface_bilinear_diagonal!(local_diagonal::AbstractVector{T},
                                                  operator::_InterfaceBilinearForm,
                                                  values) where {T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  minus_values = minus(values)
  plus_values = plus(values)

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_side in (false, true)
      test_values = test_side ? plus_values : minus_values

      for test_component in 1:component_count(test_field)
        for test_mode in 1:local_mode_count(test_values, test_field)
          row = local_dof_index(test_values, test_field, test_component, test_mode)
          test = _WeakTraceTestBasisFunction(minus_values, plus_values, test_field, point_index,
                                             test_component, test_mode, test_side)

          for trial_side in (false, true)
            trial_values = trial_side ? plus_values : minus_values

            for trial_component in 1:component_count(trial_field)
              for trial_mode in 1:local_mode_count(trial_values, trial_field)
                column = local_dof_index(trial_values, trial_field, trial_component, trial_mode)
                row == column || continue
                trial = _WeakTraceTrialBasisFunction(minus_values, plus_values, trial_field,
                                                     point_index, trial_component, trial_mode,
                                                     trial_side)
                local_diagonal[row] += weighted *
                                       _weak_bilinear_value(operator.form(q, test, trial))
              end
            end
          end
        end
      end
    end
  end

  return local_diagonal
end

function _accumulate_interface_linear_rhs!(local_rhs::AbstractVector{T},
                                           operator::_InterfaceLinearForm,
                                           values) where {T<:AbstractFloat}
  test_field = operator.test_field
  minus_values = minus(values)
  plus_values = plus(values)

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_side in (false, true)
      test_values = test_side ? plus_values : minus_values

      for test_component in 1:component_count(test_field)
        for test_mode in 1:local_mode_count(test_values, test_field)
          row = local_dof_index(test_values, test_field, test_component, test_mode)
          test = _WeakTraceBasisFunction(minus_values, plus_values, test_field, point_index,
                                         test_component, test_mode, test_side)
          local_rhs[row] += weighted * operator.form(q, test)
        end
      end
    end
  end

  return local_rhs
end

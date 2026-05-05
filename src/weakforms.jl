# First-party weak-form operators.
#
# These operators are the preferred affine problem API. A user supplies the
# mathematical form once; the implementation below lowers it to matrix-free
# application, local matrices, and diagonal data.

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

struct _WeakQuadraturePoint{V}
  values::V
  point_index::Int
end

struct _WeakBasisFunction{V,F<:AbstractField}
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

# The tensorized cell path evaluates the form once per quadrature point with a
# symbolic test function. The resulting `_WeakTestTerm` stores the coefficients
# multiplying v_h and ∇v_h so that the contribution can be projected with
# sum-factorized tensor kernels.
struct _WeakTestFunction{D,T<:AbstractFloat,F<:AbstractField}
  field::F
  component::Int
end

# A test term is the affine functional α v_h + β · ∇v_h at one quadrature point.
# Nonlinear dependence on the test function is rejected explicitly because such
# a form cannot be represented by a bilinear matrix-free operator.
struct _WeakTestTerm{D,T<:AbstractFloat} <: Number
  value_coefficient::T
  gradient_coefficients::NTuple{D,T}
end

struct _WeakTracePair{M,P}
  minus::M
  plus::P
end

struct _WeakTraceBasisFunction{MV,PV,F<:AbstractField}
  minus_values::MV
  plus_values::PV
  field::F
  point_index::Int
  component::Int
  mode_index::Int
  plus_side::Bool
end

"""
    add_cell_bilinear!(problem, test_field, trial_field, form)
    add_cell_bilinear!(form, problem, test_field, trial_field)

Add a cell bilinear weak form to an affine problem.

The `do`-block form receives `(q, v, w)`, where `q` is the quadrature point,
`v` is a test-basis proxy, and `w` is a trial-basis proxy. Use `value(v)`,
`grad(v)`, `gradient(v)`, `point(q)`, and `weight(q)` inside the form.
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

The `do`-block form receives `(q, v)`, where `q` is the quadrature point and
`v` is a test-basis proxy.
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

Add an embedded-surface bilinear weak form to an affine problem.
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

Add an embedded-surface linear weak form to an affine problem.
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

The `do`-block receives `(q, v, w)`. `value(v)`, `grad(v)`, and
`normal_gradient(v)` are trace pairs; use `jump(...)`, `average(...)`, `avg(...)`,
`minus(...)`, and `plus(...)` to combine them.
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

@inline point(q::_WeakQuadraturePoint) = point(q.values, q.point_index)
@inline weight(q::_WeakQuadraturePoint) = weight(q.values, q.point_index)
@inline coordinate(q::_WeakQuadraturePoint, axis::Integer) = point(q)[axis]
@inline normal(q::_WeakQuadraturePoint) = normal(q.values, q.point_index)
@inline cell_size(q::_WeakQuadraturePoint, field::AbstractField) = ntuple(axis -> cell_size(field_space(field).domain,
                                                                                            q.values.leaf,
                                                                                            axis),
                                                                          dimension(field_space(field)))
@inline cell_size(q::_WeakQuadraturePoint, field::AbstractField, axis::Integer) = cell_size(field_space(field).domain,
                                                                                            q.values.leaf,
                                                                                            axis)

@inline value(basis::_WeakBasisFunction) = shape_value(basis.values, basis.field, basis.point_index,
                                                       basis.mode_index)

@inline value(trial::_WeakTrialFunction) = trial.value

@inline function value(test::_WeakTestFunction{D,T}) where {D,T<:AbstractFloat}
  return _WeakTestTerm(one(T), ntuple(_ -> zero(T), Val(D)))
end

@inline function value(basis::_WeakTraceBasisFunction)
  active_value = _trace_shape_value(basis)
  inactive_value = zero(active_value)
  return basis.plus_side ? _WeakTracePair(inactive_value, active_value) :
         _WeakTracePair(active_value, inactive_value)
end

@inline function gradient(basis::_WeakBasisFunction)
  return shape_gradient(basis.values, basis.field, basis.point_index, basis.mode_index)
end

@inline gradient(trial::_WeakTrialFunction) = trial.gradient

@inline function gradient(test::_WeakTestFunction{D,T}) where {D,T<:AbstractFloat}
  return ntuple(axis -> _WeakTestTerm(zero(T),
                                      ntuple(current -> current == axis ? one(T) : zero(T), Val(D))),
                Val(D))
end

@inline function gradient(basis::_WeakTraceBasisFunction)
  active_gradient = _trace_shape_gradient(basis)
  inactive_gradient = _zero_like(active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline grad(basis::_WeakBasisFunction) = gradient(basis)
@inline grad(trial::_WeakTrialFunction) = gradient(trial)
@inline grad(test::_WeakTestFunction) = gradient(test)
@inline grad(basis::_WeakTraceBasisFunction) = gradient(basis)
@inline ∇(basis::_WeakBasisFunction) = gradient(basis)
@inline ∇(trial::_WeakTrialFunction) = gradient(trial)
@inline ∇(test::_WeakTestFunction) = gradient(test)
@inline ∇(basis::_WeakTraceBasisFunction) = gradient(basis)

@inline function normal_gradient(basis::_WeakBasisFunction)
  return shape_normal_gradient(basis.values, basis.field, basis.point_index, basis.mode_index)
end

@inline function normal_gradient(trial::_WeakTrialFunction{D,T,F,T}) where {D,T<:AbstractFloat,
                                                                            F<:AbstractField}
  return trial.normal_gradient
end

function normal_gradient(::_WeakTrialFunction{D,T,F,Nothing}) where {D,T<:AbstractFloat,
                                                                     F<:AbstractField}
  throw(ArgumentError("normal_gradient is only available for weak-form trial functions on faces, interfaces, and embedded surfaces"))
end

function normal_gradient(::_WeakTestFunction)
  throw(ArgumentError("normal_gradient is not available in tensorized cell weak-form projection"))
end

@inline function normal_gradient(basis::_WeakTraceBasisFunction)
  active_gradient = _trace_shape_normal_gradient(basis)
  inactive_gradient = zero(active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline component(basis::_WeakBasisFunction) = basis.component
@inline component(trial::_WeakTrialFunction) = trial.component
@inline component(test::_WeakTestFunction) = test.component
@inline component(basis::_WeakTraceBasisFunction) = basis.component

@inline function _zero_test_term(::Type{T}, ::Val{D}) where {D,T<:AbstractFloat}
  return _WeakTestTerm(zero(T), ntuple(_ -> zero(T), Val(D)))
end

@inline Base.zero(term::_WeakTestTerm{D,T}) where {D,T<:AbstractFloat} = _zero_test_term(T, Val(D))
@inline Base.zero(::Type{_WeakTestTerm{D,T}}) where {D,T<:AbstractFloat} = _zero_test_term(T,
                                                                                           Val(D))

@inline function Base.:+(a::_WeakTestTerm{D,T}, b::_WeakTestTerm{D,T}) where {D,T<:AbstractFloat}
  return _WeakTestTerm(a.value_coefficient + b.value_coefficient,
                       ntuple(axis -> a.gradient_coefficients[axis] + b.gradient_coefficients[axis],
                              Val(D)))
end

@inline Base.:-(a::_WeakTestTerm{D,T}) where {D,T<:AbstractFloat} = -one(T) * a

@inline Base.:-(a::_WeakTestTerm{D,T}, b::_WeakTestTerm{D,T}) where {D,T<:AbstractFloat} = a + (-b)

@inline function Base.:+(a::_WeakTestTerm, b::Number)
  iszero(b) && return a
  throw(ArgumentError("cell bilinear form produced a nonzero term independent of the test function"))
end

@inline Base.:+(a::Number, b::_WeakTestTerm) = b + a

@inline function Base.:-(a::_WeakTestTerm, b::Number)
  iszero(b) && return a
  throw(ArgumentError("cell bilinear form produced a nonzero term independent of the test function"))
end

@inline function Base.:-(a::Number, b::_WeakTestTerm)
  iszero(a) && return -b
  throw(ArgumentError("cell bilinear form produced a nonzero term independent of the test function"))
end

@inline function Base.:*(a::_WeakTestTerm{D,T}, b::Number) where {D,T<:AbstractFloat}
  scale = T(b)
  return _WeakTestTerm(scale * a.value_coefficient,
                       ntuple(axis -> scale * a.gradient_coefficients[axis], Val(D)))
end

@inline Base.:*(a::Number, b::_WeakTestTerm) = b * a

function Base.:*(::_WeakTestTerm, ::_WeakTestTerm)
  throw(ArgumentError("cell bilinear form is nonlinear in the test function"))
end

@inline Base.:/(a::_WeakTestTerm{D,T}, b::Number) where {D,T<:AbstractFloat} = a * inv(T(b))

@inline Base.conj(a::_WeakTestTerm) = a

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
@inline LinearAlgebra.dot(a::_WeakTracePair, b::_WeakTracePair) = inner(a, b)
@inline minus(pair::_WeakTracePair) = pair.minus
@inline plus(pair::_WeakTracePair) = pair.plus
@inline jump(pair::_WeakTracePair) = jump(pair.minus, pair.plus)
@inline average(pair::_WeakTracePair) = average(pair.minus, pair.plus)
@inline avg(pair::_WeakTracePair) = average(pair)
@inline avg(minus_value, plus_value) = average(minus_value, plus_value)

@inline _zero_like(value::Number) = zero(value)
@inline _zero_like(value::Tuple) = map(_zero_like, value)

@inline function _trace_values(basis::_WeakTraceBasisFunction)
  return basis.plus_side ? basis.plus_values : basis.minus_values
end

@inline function _trace_shape_value(basis::_WeakTraceBasisFunction)
  values = _trace_values(basis)
  return shape_value(values, basis.field, basis.point_index, basis.mode_index)
end

@inline function _trace_shape_gradient(basis::_WeakTraceBasisFunction)
  values = _trace_values(basis)
  return shape_gradient(values, basis.field, basis.point_index, basis.mode_index)
end

@inline function _trace_shape_normal_gradient(basis::_WeakTraceBasisFunction)
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
  _accumulate_interface_bilinear_apply!(local_result, operator, values, local_coefficients)
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
        test = _WeakBasisFunction(values, test_field, point_index, test_component, test_mode)

        for trial_component in 1:component_count(trial_field)
          for trial_mode in 1:local_mode_count(values, trial_field)
            column = local_dof_index(values, trial_field, trial_component, trial_mode)
            trial = _WeakBasisFunction(values, trial_field, point_index, trial_component,
                                       trial_mode)
            local_matrix[row, column] += weighted * operator.form(q, test, trial)
          end
        end
      end
    end
  end

  return local_matrix
end

function _accumulate_cell_bilinear_apply!(local_result::AbstractVector{T},
                                          operator::_LocalBilinearForm, values,
                                          local_coefficients::AbstractVector{T},
                                          scratch::KernelScratch{T}) where {T<:AbstractFloat}
  # Vector-valued trial fields keep the basis loop because component-coupled
  # forms need the exact per-component matrix semantics.
  component_count(operator.trial_field) == 1 ||
    return _accumulate_cell_bilinear_apply_by_basis!(local_result, operator, values,
                                                     local_coefficients)
  tensor = tensor_values(values, operator.trial_field)

  if tensor !== nothing && is_full_tensor(tensor)
    return _accumulate_cell_bilinear_apply_tensor_trial!(local_result, operator, values,
                                                         local_coefficients, tensor, scratch)
  end

  return _accumulate_cell_bilinear_apply_scalar_trial!(local_result, operator, values,
                                                       local_coefficients)
end

function _accumulate_cell_bilinear_apply_by_basis!(local_result::AbstractVector{T},
                                                   operator::_LocalBilinearForm, values,
                                                   local_coefficients::AbstractVector{T}) where {T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)

    for test_component in 1:component_count(test_field)
      for test_mode in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, test_component, test_mode)
        test = _WeakBasisFunction(values, test_field, point_index, test_component, test_mode)
        value_sum = zero(T)

        for trial_component in 1:component_count(trial_field)
          for trial_mode in 1:local_mode_count(values, trial_field)
            column = local_dof_index(values, trial_field, trial_component, trial_mode)
            coefficient = local_coefficients[column]
            coefficient == zero(T) && continue
            trial = _WeakBasisFunction(values, trial_field, point_index, trial_component,
                                       trial_mode)
            value_sum += coefficient * operator.form(q, test, trial)
          end
        end

        local_result[row] += weighted * value_sum
      end
    end
  end

  return local_result
end

function _trial_function(values::_FieldEvaluationValues, local_coefficients::AbstractVector{T},
                         field::AbstractField, point_index::Int) where {T<:AbstractFloat}
  data = _field_values(values, field)
  value_data = _local_field_value_component(data, local_coefficients, 1, point_index)
  gradient_data = _local_field_gradient(data, local_coefficients, 1, point_index)
  normal_data = _normal_trial_gradient(values, data, local_coefficients, point_index)
  return _WeakTrialFunction(field, 1, value_data, gradient_data, normal_data)
end

@inline _normal_trial_gradient(values, data, local_coefficients, point_index) = nothing

@inline function _normal_trial_gradient(values::_NormalEvaluationValues, data::_FieldValues{D,T},
                                        local_coefficients::AbstractVector{T},
                                        point_index::Int) where {D,T<:AbstractFloat}
  return _local_field_normal_gradient(data, local_coefficients, 1, point_index,
                                      _point_normal(values, point_index))
end

function _accumulate_cell_bilinear_apply_scalar_trial!(local_result::AbstractVector{T},
                                                       operator::_LocalBilinearForm, values,
                                                       local_coefficients::AbstractVector{T}) where {T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)
    trial = _trial_function(values, local_coefficients, trial_field, point_index)

    for test_component in 1:component_count(test_field)
      for test_mode in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, test_component, test_mode)
        test = _WeakBasisFunction(values, test_field, point_index, test_component, test_mode)
        local_result[row] += weighted * operator.form(q, test, trial)
      end
    end
  end

  return local_result
end

function _copy_scalar_tensor_trial_coefficients!(result::AbstractVector{T}, values,
                                                 field::AbstractField,
                                                 local_coefficients::AbstractVector{T}) where {T<:AbstractFloat}
  data = _field_values(values, field)
  offset = first(data.block) + _field_component_offset(data, 1) - 1

  @inbounds for mode_index in eachindex(result)
    result[mode_index] = local_coefficients[offset+mode_index]
  end

  return result
end

@inline function _tensor_trial_function(::Val{D}, field::AbstractField, point_index::Int,
                                        point_values::AbstractVector{T},
                                        point_gradients::AbstractMatrix{T}) where {D,
                                                                                   T<:AbstractFloat}
  gradient_data = ntuple(axis -> (@inbounds point_gradients[axis, point_index]), Val(D))
  value_data = @inbounds point_values[point_index]
  return _WeakTrialFunction(field, 1, value_data, gradient_data, nothing)
end

function _accumulate_cell_bilinear_apply_tensor_trial!(local_result::AbstractVector{T},
                                                       operator::_CellBilinearForm, values,
                                                       local_coefficients::AbstractVector{T},
                                                       tensor::TensorProductValues{D,T},
                                                       scratch::KernelScratch{T}) where {D,
                                                                                         T<:AbstractFloat}
  test_tensor = tensor_values(values, operator.test_field)
  if component_count(operator.test_field) == 1 &&
     test_tensor !== nothing &&
     is_full_tensor(test_tensor)
    return _accumulate_cell_bilinear_apply_tensor_project!(local_result, operator, values,
                                                           local_coefficients, test_tensor, tensor,
                                                           scratch)
  end

  return _accumulate_cell_bilinear_apply_tensor_test_basis!(local_result, operator, values,
                                                            local_coefficients, tensor, scratch)
end

function _accumulate_cell_bilinear_apply_tensor_test_basis!(local_result::AbstractVector{T},
                                                            operator::_CellBilinearForm, values,
                                                            local_coefficients::AbstractVector{T},
                                                            tensor::TensorProductValues{D,T},
                                                            scratch::KernelScratch{T}) where {D,
                                                                                              T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  trial_coefficients = scratch_vector(scratch, D + 2, tensor_mode_count(tensor))
  point_values = scratch_vector(scratch, D + 3, tensor_point_count(tensor))
  point_gradients = scratch_matrix(scratch, 1, D, tensor_point_count(tensor))
  _copy_scalar_tensor_trial_coefficients!(trial_coefficients, values, trial_field,
                                          local_coefficients)
  tensor_interpolate!(point_values, tensor, trial_coefficients, scratch)
  tensor_gradient!(point_gradients, tensor, trial_coefficients, scratch)

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)
    trial = _tensor_trial_function(Val(D), trial_field, point_index, point_values, point_gradients)

    for test_component in 1:component_count(test_field)
      for test_mode in 1:local_mode_count(values, test_field)
        row = local_dof_index(values, test_field, test_component, test_mode)
        test = _WeakBasisFunction(values, test_field, point_index, test_component, test_mode)
        local_result[row] += weighted * operator.form(q, test, trial)
      end
    end
  end

  return local_result
end

function _add_scalar_tensor_output!(local_result::AbstractVector{T}, values, field::AbstractField,
                                    contribution::AbstractVector{T}) where {T<:AbstractFloat}
  data = _field_values(values, field)
  offset = first(data.block) + _field_component_offset(data, 1) - 1

  @inbounds for mode_index in eachindex(contribution)
    local_result[offset+mode_index] += contribution[mode_index]
  end

  return local_result
end

@inline _test_value_coefficient(term::_WeakTestTerm) = term.value_coefficient

@inline function _test_value_coefficient(term::Number)
  iszero(term) && return term
  throw(ArgumentError("cell bilinear form produced a nonzero term independent of the test function"))
end

@inline _test_gradient_coefficient(term::_WeakTestTerm, axis::Int) = term.gradient_coefficients[axis]

@inline function _test_gradient_coefficient(term::Number, axis::Int)
  iszero(term) && return term
  throw(ArgumentError("cell bilinear form produced a nonzero term independent of the test function"))
end

function _accumulate_cell_bilinear_apply_tensor_project!(local_result::AbstractVector{T},
                                                         operator::_CellBilinearForm, values,
                                                         local_coefficients::AbstractVector{T},
                                                         test_tensor::TensorProductValues{D,T},
                                                         trial_tensor::TensorProductValues{D,T},
                                                         scratch::KernelScratch{T}) where {D,
                                                                                           T<:AbstractFloat}
  test_field = operator.test_field
  trial_field = operator.trial_field
  trial_coefficients = scratch_vector(scratch, D + 2, tensor_mode_count(trial_tensor))
  trial_values = scratch_vector(scratch, D + 3, tensor_point_count(trial_tensor))
  weighted_values = scratch_vector(scratch, D + 4, tensor_point_count(test_tensor))
  contribution = scratch_vector(scratch, D + 5, tensor_mode_count(test_tensor))
  trial_gradients = scratch_matrix(scratch, 1, D, tensor_point_count(trial_tensor))
  weighted_gradients = scratch_matrix(scratch, 2, D, tensor_point_count(test_tensor))
  _copy_scalar_tensor_trial_coefficients!(trial_coefficients, values, trial_field,
                                          local_coefficients)
  tensor_interpolate!(trial_values, trial_tensor, trial_coefficients, scratch)
  tensor_gradient!(trial_gradients, trial_tensor, trial_coefficients, scratch)
  fill!(contribution, zero(T))

  for point_index in 1:point_count(values)
    q = _WeakQuadraturePoint(values, point_index)
    weighted = weight(q)
    test = _WeakTestFunction{D,T,typeof(test_field)}(test_field, 1)
    trial = _tensor_trial_function(Val(D), trial_field, point_index, trial_values, trial_gradients)
    term = operator.form(q, test, trial)
    weighted_values[point_index] = weighted * T(_test_value_coefficient(term))

    for axis in 1:D
      weighted_gradients[axis, point_index] = weighted * T(_test_gradient_coefficient(term, axis))
    end
  end

  tensor_project!(contribution, test_tensor, weighted_values, scratch)
  tensor_project_gradient!(contribution, test_tensor, weighted_gradients, scratch)
  _add_scalar_tensor_output!(local_result, values, test_field, contribution)
  return local_result
end

function _accumulate_cell_bilinear_apply_tensor_trial!(local_result::AbstractVector{T},
                                                       operator::_LocalBilinearForm, values,
                                                       local_coefficients::AbstractVector{T},
                                                       tensor::TensorProductValues,
                                                       scratch::KernelScratch{T}) where {T<:AbstractFloat}
  return _accumulate_cell_bilinear_apply_scalar_trial!(local_result, operator, values,
                                                       local_coefficients)
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
        test = _WeakBasisFunction(values, test_field, point_index, test_component, test_mode)
        trial = _WeakBasisFunction(values, trial_field, point_index, test_component, test_mode)
        local_diagonal[row] += weighted * operator.form(q, test, trial)
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
          test = _WeakTraceBasisFunction(minus_values, plus_values, test_field, point_index,
                                         test_component, test_mode, test_side)

          for trial_side in (false, true)
            trial_values = trial_side ? plus_values : minus_values

            for trial_component in 1:component_count(trial_field)
              for trial_mode in 1:local_mode_count(trial_values, trial_field)
                column = local_dof_index(trial_values, trial_field, trial_component, trial_mode)
                trial = _WeakTraceBasisFunction(minus_values, plus_values, trial_field, point_index,
                                                trial_component, trial_mode, trial_side)
                local_matrix[row, column] += weighted * operator.form(q, test, trial)
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
                                               local_coefficients::AbstractVector{T}) where {T<:AbstractFloat}
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
          test = _WeakTraceBasisFunction(minus_values, plus_values, test_field, point_index,
                                         test_component, test_mode, test_side)
          value_sum = zero(T)

          for trial_side in (false, true)
            trial_values = trial_side ? plus_values : minus_values

            for trial_component in 1:component_count(trial_field)
              for trial_mode in 1:local_mode_count(trial_values, trial_field)
                column = local_dof_index(trial_values, trial_field, trial_component, trial_mode)
                coefficient = local_coefficients[column]
                coefficient == zero(T) && continue
                trial = _WeakTraceBasisFunction(minus_values, plus_values, trial_field, point_index,
                                                trial_component, trial_mode, trial_side)
                value_sum += coefficient * operator.form(q, test, trial)
              end
            end
          end

          local_result[row] += weighted * value_sum
        end
      end
    end
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
          test = _WeakTraceBasisFunction(minus_values, plus_values, test_field, point_index,
                                         test_component, test_mode, test_side)

          for trial_side in (false, true)
            trial_values = trial_side ? plus_values : minus_values

            for trial_component in 1:component_count(trial_field)
              for trial_mode in 1:local_mode_count(trial_values, trial_field)
                column = local_dof_index(trial_values, trial_field, trial_component, trial_mode)
                row == column || continue
                trial = _WeakTraceBasisFunction(minus_values, plus_values, trial_field, point_index,
                                                trial_component, trial_mode, trial_side)
                local_diagonal[row] += weighted * operator.form(q, test, trial)
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

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
                                test_field::AbstractField, trial_field::AbstractField,
                                form)
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

function add_surface_bilinear!(problem::AffineProblem, tag::Symbol,
                               test_field::AbstractField, trial_field::AbstractField,
                               form)
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

function add_surface_bilinear!(form, problem::AffineProblem, tag::Symbol,
                               test_field::AbstractField, trial_field::AbstractField)
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

function add_surface_linear!(problem::AffineProblem, tag::Symbol, test_field::AbstractField,
                             form)
  data = _problem_data(problem)
  _check_problem_field(data.fields, test_field, "surface linear test")
  add_surface!(problem, tag, _SurfaceLinearForm(form, test_field))
  return problem
end

function add_surface_linear!(form, problem::AffineProblem, test_field::AbstractField)
  return add_surface_linear!(problem, test_field, form)
end

function add_surface_linear!(form, problem::AffineProblem, tag::Symbol,
                             test_field::AbstractField)
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

@inline value(basis::_WeakBasisFunction) =
  shape_value(basis.values, basis.field, basis.point_index, basis.mode_index)

@inline function value(basis::_WeakTraceBasisFunction)
  active_value = _trace_shape_value(basis)
  inactive_value = zero(active_value)
  return basis.plus_side ? _WeakTracePair(inactive_value, active_value) :
         _WeakTracePair(active_value, inactive_value)
end

@inline function gradient(basis::_WeakBasisFunction)
  return shape_gradient(basis.values, basis.field, basis.point_index, basis.mode_index)
end

@inline function gradient(basis::_WeakTraceBasisFunction)
  active_gradient = _trace_shape_gradient(basis)
  inactive_gradient = _zero_like(active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline grad(basis::_WeakBasisFunction) = gradient(basis)
@inline grad(basis::_WeakTraceBasisFunction) = gradient(basis)
@inline ∇(basis::_WeakBasisFunction) = gradient(basis)
@inline ∇(basis::_WeakTraceBasisFunction) = gradient(basis)

@inline function normal_gradient(basis::_WeakBasisFunction)
  return shape_normal_gradient(basis.values, basis.field, basis.point_index, basis.mode_index)
end

@inline function normal_gradient(basis::_WeakTraceBasisFunction)
  active_gradient = _trace_shape_normal_gradient(basis)
  inactive_gradient = zero(active_gradient)
  return basis.plus_side ? _WeakTracePair(inactive_gradient, active_gradient) :
         _WeakTracePair(active_gradient, inactive_gradient)
end

@inline component(basis::_WeakBasisFunction) = basis.component
@inline component(basis::_WeakTraceBasisFunction) = basis.component

@inline inner(a::Number, b::Number) = a * b

@inline function inner(a::NTuple{N,<:Number}, b::NTuple{N,<:Number}) where {N}
  return _tuple_inner(a, b, Val(N))
end

@inline function _tuple_inner(a::NTuple{N,<:Number}, b::NTuple{N,<:Number},
                             ::Val{N}) where {N}
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

function cell_matrix!(local_matrix::AbstractMatrix{T}, operator::_CellBilinearForm,
                      values, scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_matrix!(local_matrix, operator, values)
  return local_matrix
end

function face_matrix!(local_matrix::AbstractMatrix{T}, operator::_BoundaryBilinearForm,
                      values, scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_matrix!(local_matrix, operator, values)
  return local_matrix
end

function surface_matrix!(local_matrix::AbstractMatrix{T}, operator::_SurfaceBilinearForm,
                         values, scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_matrix!(local_matrix, operator, values)
  return local_matrix
end

function cell_apply!(local_result::AbstractVector{T}, operator::_CellBilinearForm,
                     values, local_coefficients::AbstractVector{T},
                     scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_apply!(local_result, operator, values, local_coefficients)
  return nothing
end

function face_apply!(local_result::AbstractVector{T}, operator::_BoundaryBilinearForm,
                     values, local_coefficients::AbstractVector{T},
                     scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_apply!(local_result, operator, values, local_coefficients)
  return nothing
end

function surface_apply!(local_result::AbstractVector{T}, operator::_SurfaceBilinearForm,
                        values, local_coefficients::AbstractVector{T},
                        scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_apply!(local_result, operator, values, local_coefficients)
  return nothing
end

function cell_diagonal!(local_diagonal::AbstractVector{T}, operator::_CellBilinearForm,
                        values, scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_cell_bilinear_diagonal!(local_diagonal, operator, values)
  return nothing
end

function face_diagonal!(local_diagonal::AbstractVector{T}, operator::_BoundaryBilinearForm,
                        values, scratch::KernelScratch{T}) where {T<:AbstractFloat}
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

function interface_apply!(local_result::AbstractVector{T}, operator::_InterfaceBilinearForm,
                          values, local_coefficients::AbstractVector{T},
                          scratch::KernelScratch{T}) where {T<:AbstractFloat}
  _accumulate_interface_bilinear_apply!(local_result, operator, values, local_coefficients)
  return nothing
end

function interface_diagonal!(local_diagonal::AbstractVector{T},
                             operator::_InterfaceBilinearForm, values,
                             scratch::KernelScratch{T}) where {T<:AbstractFloat}
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
                                          operator::_LocalBilinearForm,
                                          values,
                                          local_coefficients::AbstractVector{T}) where {
                                                                                         T<:AbstractFloat}
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

function _accumulate_cell_bilinear_diagonal!(local_diagonal::AbstractVector{T},
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
            row == column || continue
            trial = _WeakBasisFunction(values, trial_field, point_index, trial_component,
                                       trial_mode)
            local_diagonal[row] += weighted * operator.form(q, test, trial)
          end
        end
      end
    end
  end

  return local_diagonal
end

function _accumulate_cell_linear_rhs!(local_rhs::AbstractVector{T},
                                      operator::_LocalLinearForm,
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
                trial = _WeakTraceBasisFunction(minus_values, plus_values, trial_field,
                                                point_index, trial_component, trial_mode,
                                                trial_side)
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
                                               operator::_InterfaceBilinearForm,
                                               values,
                                               local_coefficients::AbstractVector{T}) where {
                                                                                            T<:AbstractFloat}
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
                trial = _WeakTraceBasisFunction(minus_values, plus_values, trial_field,
                                                point_index, trial_component, trial_mode,
                                                trial_side)
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
                trial = _WeakTraceBasisFunction(minus_values, plus_values, trial_field,
                                                point_index, trial_component, trial_mode,
                                                trial_side)
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

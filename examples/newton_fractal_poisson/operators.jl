# These operators evaluate the scalar Poisson weak form for the manufactured
# problem. Their names are example-specific so benchmark and test harnesses can
# include several examples in the same Julia session without method ambiguity.

# The diffusion operator contributes the bilinear form
# `a(v, u) = ∫Ω ∇v · ∇u dΩ`. The local kernel evaluates its matrix-free action
# directly against the local coefficient vector.
struct NewtonFractalPoissonDiffusion{F}
  field::F
end

function cell_apply!(local_result, operator::NewtonFractalPoissonDiffusion, values::CellValues,
                     local_coefficients)
  local_block = block(local_result, values, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    trial_gradient = gradient(values, local_coefficients, operator.field, point_index)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      contribution = zero(eltype(local_result))

      for axis in 1:axis_count
        contribution += gradients[axis, row_mode, point_index] * trial_gradient[axis]
      end

      local_block[row_mode] += contribution * weighted
    end
  end

  return nothing
end

function cell_diagonal!(local_diagonal, operator::NewtonFractalPoissonDiffusion, values::CellValues)
  local_block = block(local_diagonal, values, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      contribution = zero(eltype(local_diagonal))

      for axis in 1:axis_count
        gradient_value = gradients[axis, mode_index, point_index]
        contribution += gradient_value * gradient_value
      end

      local_block[mode_index] += contribution * weighted
    end
  end

  return nothing
end

# This strong source operator represents `ℓ(v) = ∫Ω v f dΩ`. The main example
# uses the gradient-load formulation below, but this operator is kept beside the
# finite-difference source term so the same manufactured data can be inspected
# with the more direct Poisson right-hand side.
struct NewtonFractalPoissonSource{F,G}
  field::F
  data::G
end

function cell_rhs!(local_rhs, operator::NewtonFractalPoissonSource, values::CellValues)
  local_block = block(local_rhs, values, operator.field)
  shape_table = shape_values(values, operator.field)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = operator.data(point(values, point_index)) * weight(values, point_index)

    for mode_index in 1:mode_count
      local_block[mode_index] += shape_table[mode_index, point_index] * weighted
    end
  end

  return nothing
end

# The gradient-load operator assembles `ℓ(v) = ∫Ω ∇v · ∇uₑ dΩ`. Together with
# the exact Dirichlet trace `uₑ|∂Ω`, this gives the same weak problem as
# `-Δu = f` without requiring a second finite difference of the Newton-fractal
# field during assembly.
struct NewtonFractalPoissonGradientLoad{F,G}
  field::F
  gradient::G
end

function cell_rhs!(local_rhs, operator::NewtonFractalPoissonGradientLoad, values::CellValues)
  local_block = block(local_rhs, values, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    exact_gradient = operator.gradient(point(values, point_index))
    weighted = weight(values, point_index)

    for mode_index in 1:mode_count
      contribution = zero(eltype(local_rhs))

      for axis in 1:axis_count
        contribution += gradients[axis, mode_index, point_index] * exact_gradient[axis]
      end

      local_block[mode_index] += contribution * weighted
    end
  end

  return nothing
end

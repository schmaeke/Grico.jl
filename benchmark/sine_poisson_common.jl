# Shared sine-interface Poisson ingredients used by the CG and DG benchmarks.

import Grico: cell_matrix!, cell_rhs!

struct SineInterfacePoissonDiffusion{F}
  field::F
end

struct SineInterfacePoissonSource{F}
  field::F
end

function sine_interface_poisson_field(options; continuity)
  domain = Domain((0.0, 0.0), (1.0, 1.0), options["root_cells"])
  space = HpSpace(domain,
                  SpaceOptions(basis=FullTensorBasis(), degree=UniformDegree(options["degree"]),
                               quadrature=DegreePlusQuadrature(options["quadrature_extra_points"]),
                               continuity=continuity))
  return ScalarField(space; name=:u)
end

function add_sine_interface_poisson_cells!(problem, field)
  add_cell!(problem, SineInterfacePoissonDiffusion(field))
  add_cell!(problem, SineInterfacePoissonSource(field))
  return problem
end

# Assemble the scalar Poisson stiffness block ∫Ω ∇u · ∇v dΩ. The local matrix is
# filled symmetrically because the operator is self-adjoint and scalar.
function cell_matrix!(local_matrix, operator::SineInterfacePoissonDiffusion, values::CellValues)
  local_block = block(local_matrix, values, operator.field, operator.field)
  gradients = shape_gradients(values, operator.field)
  axis_count = size(gradients, 1)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = weight(values, point_index)

    for row_mode in 1:mode_count
      for col_mode in 1:row_mode
        contribution = zero(eltype(local_matrix))

        for axis in 1:axis_count
          contribution += gradients[axis, row_mode, point_index] *
                          gradients[axis, col_mode, point_index]
        end

        contribution *= weighted
        local_block[row_mode, col_mode] += contribution
        row_mode == col_mode || (local_block[col_mode, row_mode] += contribution)
      end
    end
  end

  return nothing
end

# Assemble ∫Ω f v dΩ with the discontinuous source evaluated at physical
# quadrature points. The jump across the sine interface drives adaptive
# refinement without prescribing whether the planner should use h or p changes.
function cell_rhs!(local_rhs, operator::SineInterfacePoissonSource, values::CellValues)
  local_block = block(local_rhs, values, operator.field)
  shape_table = shape_values(values, operator.field)
  mode_count = local_mode_count(values, operator.field)

  @inbounds for point_index in 1:point_count(values)
    weighted = sine_interface_source(point(values, point_index)) * weight(values, point_index)

    for mode_index in 1:mode_count
      local_block[mode_index] += shape_table[mode_index, point_index] * weighted
    end
  end

  return nothing
end

function sine_interface_source(x)
  interface_height = 0.5 + 0.25 * sinpi(2 * x[1])
  return x[2] < interface_height ? 1.0 : 0.0
end

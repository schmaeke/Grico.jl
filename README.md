# Grico.jl

High-order finite elements on adaptive Cartesian grids.

`Grico.jl` is a compact Julia toolbox for dimension-independent finite elements
on dyadically refined, axis-aligned Cartesian domains. It is built for
high-order tensor-product discretizations where mesh topology, space
construction, assembly, solve support, adaptivity, and post-processing should
live in one small, readable stack.

Grico is intentionally specialized. It is not a general unstructured-mesh FEM
framework. Its core model is affine Cartesian geometry with anisotropic dyadic
`h`-refinement and tensor-product polynomial spaces.

## Current Feature Set

- Dimension-independent Cartesian grids and domains, including periodic axes.
- Local anisotropic `h`, `p`, and mixed `hp` refinement and derefinement.
- High-order tensor-product spaces via `HpSpace`, with `FullTensorBasis` and
  `TrunkBasis`.
- Per-axis continuous/discontinuous coupling, including full `:cg`, full `:dg`,
  and mixed axiswise policies.
- Scalar and vector fields, multi-field layouts, Dirichlet constraints, and
  mean-value constraints.
- Operator-based assembly for cell, boundary, interface, and embedded-surface
  contributions.
- Affine and residual problems, including nonlinear residual and tangent
  assembly.
- Built-in solve support with sparse direct solves, Krylov methods,
  smoothed-aggregation AMG, ILU, additive Schwarz, and field-split Schur
  preconditioning.
- State transfer across adaptive space changes.
- Embedded geometry support through finite-cell quadrature, implicit surface
  quadrature, and explicit segment-mesh surfaces.
- `L²` and relative `L²` verification helpers, backend-neutral
  postprocessing samples, and optional VTK/PVD export through `WriteVTK`.

## Installation

Grico currently targets Julia `1.12`.

```julia
using Pkg
Pkg.add(url="https://github.com/schmaeke/Grico.jl.git")
```

From a local checkout:

```bash
julia --project=.
```

The package root intentionally does not track a `Manifest.toml`. Dependency
resolution for normal package development therefore comes from
[`Project.toml`](Project.toml). The [`benchmark/`](benchmark) environment keeps
its own [`Project.toml`](benchmark/Project.toml), while local manifests and
generated benchmark reports stay untracked.

## Workflow

1. Build a `Domain` on a Cartesian grid.
2. Compile an `HpSpace` from basis, degree, quadrature, and continuity choices.
3. Define fields and add local operators to an `AffineProblem` or
   `ResidualProblem`.
4. `compile`, `assemble`, and `solve`.
5. Optionally verify, sample/export, or build a new adaptive space with
   `adaptivity_plan`.

## Minimal Example

```julia
using Grico
import Grico: cell_matrix!, cell_rhs!

struct Diffusion{F,T}
  field::F
  kappa::T
end

function cell_matrix!(local_matrix, op::Diffusion, values::CellValues)
  A = block(local_matrix, values, op.field, op.field)
  mode_count = local_mode_count(values, op.field)

  for q in 1:point_count(values)
    w = weight(values, q)

    for i in 1:mode_count
      grad_i = shape_gradient(values, op.field, q, i)

      for j in 1:mode_count
        grad_j = shape_gradient(values, op.field, q, j)
        A[i, j] += op.kappa * sum(grad_i[a] * grad_j[a] for a in eachindex(grad_i)) * w
      end
    end
  end

  return nothing
end

struct Source{F,G}
  field::F
  f::G
end

function cell_rhs!(local_rhs, op::Source, values::CellValues)
  b = block(local_rhs, values, op.field)
  mode_count = local_mode_count(values, op.field)

  for q in 1:point_count(values)
    w = op.f(point(values, q)) * weight(values, q)

    for i in 1:mode_count
      b[i] += shape_value(values, op.field, q, i) * w
    end
  end

  return nothing
end

domain = Domain((0.0,), (1.0,), (8,))
space = HpSpace(domain, SpaceOptions(degree=UniformDegree(2)))
u = ScalarField(space; name=:u)

problem = AffineProblem(u)
add_cell!(problem, Diffusion(u, 1.0))
add_cell!(problem, Source(u, x -> 1.0))
add_constraint!(problem, Dirichlet(u, BoundaryFace(1, LOWER), 0.0))
add_constraint!(problem, Dirichlet(u, BoundaryFace(1, UPPER), 0.0))

plan = compile(problem)
state = State(plan, solve(assemble(plan)))
```

## Scope Notes

- Grico is currently built around axis-aligned affine Cartesian geometry.
- Continuity policies are currently package-level `:cg` and `:dg`.
- Postprocessing samples and VTK export are intended for 1D, 2D, and 3D output.
- VTK/PVD output is provided by an optional package extension; add `WriteVTK`
  to the active environment and load it before calling `write_vtk` or
  `write_pvd`.
- Makie figure output is provided by an optional package extension; add and
  load a Makie backend such as `CairoMakie` before calling `plot_field` or
  `plot_mesh`.
- Output backends share the same postprocessing inputs: a state or geometric
  reference, optional point/cell/field datasets, and the sampling controls
  `subdivisions` and `sample_degree`.

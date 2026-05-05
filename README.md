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
- Operator-based matrix-free evaluation for cell, boundary, interface, and
  embedded-surface contributions.
- Affine and residual problems, including nonlinear residual and tangent-action
  evaluation.
- Built-in reduced CG solves for affine matrix-free operators, with optional
  diagonal Jacobi data supplied by local kernels.
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
3. Define fields and add weak forms to an `AffineProblem` or
   `ResidualProblem`.
4. `compile`, apply matrix-free operators directly, and `solve`.
5. Optionally verify, sample/export, or build a new adaptive space with
   `adaptivity_plan`.

## Minimal Example

```julia
using Grico

domain = Domain((0.0,), (1.0,), (8,))
space = HpSpace(domain, SpaceOptions(degree=UniformDegree(2)))
u = ScalarField(space; name=:u)

problem = AffineProblem(u; operator_class=SPD())
add_cell_bilinear!(problem, u, u) do q, v, w
  inner(grad(v), grad(w))
end
add_cell_linear!(problem, u) do q, v
  value(v)
end
add_constraint!(problem, Dirichlet(u, BoundaryFace(1, LOWER), 0.0))
add_constraint!(problem, Dirichlet(u, BoundaryFace(1, UPPER), 0.0))

plan = compile(problem)
state = solve(plan; solver=CGSolver(preconditioner=JacobiPreconditioner()))
```

"""
    Grico

High-order finite elements on adaptive Cartesian grids.

`Grico.jl` implements dimension-independent high-order finite elements on
axis-aligned Cartesian meshes with local anisotropic `h`, `p`, and mixed `hp`
refinement. The compiled `HpSpace` supports fully continuous (`:cg`), fully
discontinuous (`:dg`), and mixed per-axis inter-element coupling. The package
is deliberately organized as a small stack of layers:

1. one-dimensional polynomial and quadrature tools,
2. Cartesian refinement topology and affine geometry,
3. admissible local bases and inter-element continuity compilation,
4. fields, local integration data, and operator descriptions,
5. matrix-free operator evaluation, solves, adaptivity, verification, and output.

This file is the architectural introduction to that stack. Its `include` order
is intentionally both the dependency order of the implementation and the
recommended reading order for a new contributor. Reading the files from top to
bottom should therefore mirror the mathematical construction of the library:
start with one-dimensional ingredients, build a dyadic mesh and a physical
domain, compile an hp space on the active leaves with the requested
inter-element continuity, attach fields and operators, then apply, solve,
adapt, and postprocess.

Two complementary viewpoints are useful while reading the package. One is
local: on each active leaf, the code builds tensor-product polynomial modes,
quadrature rules, and local weak-form contributions. The other is global: those
leaf-local quantities are then tied together by topology, continuity policy,
and global degree-of-freedom numbering. Much of the architecture exists
to keep those two viewpoints separate and explicit.

# Reading Guide

The source tree is easiest to understand in the following conceptual blocks:

1. `common.jl`
   Shared internal validation, indexing, and small utility helpers used
   throughout the package.
2. `polynomials.jl`, `quadrature.jl`
   One-dimensional numerical building blocks: Legendre-related modal data and
   quadrature rules that later become tensor-product finite-element machinery.
3. `topology.jl`, `geometry.jl`, `refinement.jl`
   The discrete mesh model. `topology.jl` owns the logical dyadic refinement
   tree, `geometry.jl` adds the affine embedding into physical space, and
   `refinement.jl` mutates that tree.
4. `basis.jl`, `continuity.jl`, `space.jl`, `fields.jl`
   The finite-element core. These files define admissible local mode families,
   compile inter-element continuity ranging from global `C⁰` coupling to
   leaf-local DG independence, materialize the public `HpSpace`, and lay out
   concrete fields and states on that space.
5. `problem.jl`, `embedded.jl`, `integration.jl`, `plans.jl`, `assembly.jl`,
   `solve.jl`, `adaptivity.jl`, `transition.jl`, `transfer.jl`,
   `indicators.jl`
   The execution layer. It describes weak-form operators, specialized
   quadrature constructions, compiled local evaluation data, operator plans,
   solve support, manual adaptivity, source-to-target transitions, state
   transfer, and advanced automatic indicators.
6. `verification.jl`, `postprocess.jl`
   Postprocessing helpers for error measurement and backend-neutral sampling.
   Concrete output backends such as VTK and plotting packages are integrated
   through package extensions.

# Public Workflow

A typical user-facing workflow follows the same structure:

1. build a `CartesianGrid` and either a background `Domain` or a `PhysicalDomain`,
2. choose basis, degree, quadrature, and continuity options and compile an
   `HpSpace`,
3. define fields and a `FieldLayout`,
4. describe a problem through local cell, face, interface, or surface
   operators,
5. `compile`, `apply!` or `solve`,
6. optionally verify, export, or adapt and transfer the resulting state.

The exports grouped below follow these same layers. Internal helper code is
kept in earlier files and usually remains unexported so that the public API
tracks the main abstractions rather than every implementation detail.

# Feature Stability

The first mature API distinguishes stable workflow features from advanced
policy features. The exported tier is the ordinary user workflow: grids,
domains, hp spaces, fields, problem builders, local operator callbacks,
matrix-free plans and application, solve hooks, manual adaptivity and transfer,
verification, sampled postprocessing, and extension entry points.

Qualified public names are supported but intentionally advanced. This includes
low-level topology and space inspection, polynomial/quadrature diagnostics,
automatic adaptivity indicators and planning, finite-cell moment-fitted
quadrature construction, implicit embedded-surface extraction, postprocessing
data containers, and default nonlinear tangent solve internals. These names are
used as `Grico.name` or imported explicitly, which keeps `using Grico` compact
while still making important expert hooks available.

Features not listed as exported or qualified public are implementation details.
They may change while the internal design is matured.
"""
module Grico

using LinearAlgebra
using Polyester: @batch, disable_polyester_threads

# Part I. Internal utilities shared throughout the implementation. This file is
# intentionally included first because nearly every later layer depends on its
# validation, indexing, and tuple-manipulation helpers.
include("common.jl")

# Part II. One-dimensional numerical ingredients. These files define the modal
# polynomial data and quadrature rules from which the tensor-product finite-
# element machinery is built.
include("polynomials.jl")
public integrated_legendre_derivatives, integrated_legendre_values,
       integrated_legendre_values_and_derivatives!, legendre_derivatives, legendre_values,
       legendre_values_and_derivatives!

include("quadrature.jl")
export PointQuadrature, point, point_count, weight
public AbstractQuadrature, GaussLegendreRule, TensorQuadrature, axis_point_counts, coordinate,
       dimension, gauss_legendre_exact_degree, gauss_legendre_rule, minimum_gauss_legendre_points

# Part III. Discrete mesh structure and its affine embedding into physical
# space. Topology comes first, geometry builds on top of it, and refinement
# mutates that topological tree.
include("topology.jl")
export CartesianGrid, LOWER, NONE, UPPER, active_leaf_count, active_leaves
public GridSnapshot, active_leaf, boundary_face_count, boundary_face_spec, check_snapshot,
       check_topology, compact!, covering_neighbor, first_child, interface_count, interface_spec,
       is_active_leaf, is_domain_boundary, is_expanded, is_periodic_axis, is_tree_cell, level,
       logical_coordinate, neighbor, opposite_active_leaves, parent, periodic_axes, root_cell_count,
       root_cell_counts, root_cell_total, snapshot, split_axis, stored_cell_count

include("geometry.jl")
export Domain, cell_center, cell_lower, cell_size, cell_upper, cell_volume, extent, grid, origin
public AbstractDomain, Geometry, compact, face_measure, map_from_biunit_cube, map_to_biunit_cube

include("refinement.jl")
export derefine!, refine!

# Part IV. Finite-element space construction. These files move from admissible
# local basis modes to compiled hp spaces with configurable continuity and
# finally to concrete field/state storage on those spaces.
include("basis.jl")
export FullTensorBasis, TrunkBasis
public AbstractBasisFamily, basis_mode_count, basis_modes, is_active_mode

include("regions.jl")
export FiniteCellExtension, ImplicitRegion, PhysicalDomain, PhysicalMeasure
public finite_cell_quadrature

include("continuity.jl")

include("space.jl")
export AxisDegrees, ByLeafDegrees, DegreePlusQuadrature, HpSpace, SpaceOptions, UniformDegree,
       cell_degrees, local_mode_count, mode_terms, scalar_dof_count
public AbstractDegreePolicy, AbstractQuadraturePolicy, basis_family, cell_quadrature_shape,
       check_space, continuity_kind, continuity_policy, global_cell_quadrature_shape,
       is_continuous_axis, is_mode_active, local_modes, support_shape

include("fields.jl")
export AbstractField, FieldLayout, ScalarField, State, VectorField, coefficients, component_count,
       field_component_values, field_count, field_dof_count, field_dof_range, field_layout,
       field_name, field_space, field_values, fields
public field_component_range

# Part V. Problem definition and execution. Once a space and fields exist, these
# files describe weak-form operators, compile reusable local evaluation data,
# apply matrix-free operators, solve them, and transition states across adaptive
# mesh or degree changes.
include("problem.jl")
export AffineProblem, BoundaryFace, Dirichlet, GeneralOperator, MeanValue, ResidualProblem, SPD,
       add_boundary!, add_cell!, add_constraint!, add_interface!, add_surface!, cell_apply!,
       cell_diagonal!, cell_matrix!, cell_residual!, cell_rhs!, cell_tangent_apply!, constrain!,
       face_apply!, face_diagonal!, face_matrix!, face_residual!, face_rhs!, face_tangent_apply!,
       interface_apply!, interface_diagonal!, interface_matrix!, interface_residual!,
       interface_rhs!, interface_tangent_apply!, operator_class, surface_apply!,
       surface_diagonal!, surface_matrix!, surface_residual!, surface_rhs!,
       surface_tangent_apply!, KernelScratch, scratch_matrix, scratch_vector
public AbstractOperatorClass

include("embedded.jl")
export EmbeddedSurface, SegmentMesh, SurfaceQuadrature, add_cell_quadrature!, add_embedded_surface!,
       add_surface_quadrature!
public implicit_surface_quadrature

include("integration.jl")
export CellValues, FaceValues, SurfaceValues, average, block, face_axis, face_side, jump,
       local_dof_index, normal, normal_component, normal_gradient, gradient, field_gradient,
       InterfaceValues, minus, plus, shape_gradient, shape_gradients, shape_normal_gradient,
       shape_value, shape_values, TensorProductValues, tensor_axis_gradients, tensor_axis_values,
       tensor_degrees, tensor_gradient!, tensor_interpolate!, tensor_local_modes, tensor_mode_count,
       tensor_mode_shape, tensor_point_count, tensor_project!, tensor_project_gradient!,
       tensor_quadrature_shape, tensor_values, value, is_full_tensor

include("weakforms.jl")
export add_boundary_bilinear!, add_boundary_linear!, add_cell_bilinear!, add_cell_linear!,
       add_interface_bilinear!, add_interface_linear!, add_surface_bilinear!,
       add_surface_linear!, avg, component, grad, inner, ∇, ⋅

include("plans.jl")
export AssemblyPlan, compile

include("assembly.jl")
export OperatorWorkspace, ResidualWorkspace, apply, apply!, residual, residual!, rhs, rhs!,
       tangent_apply, tangent_apply!

include("solve.jl")
export AutoLinearSolver, CGSolver, IdentityPreconditioner, JacobiPreconditioner, solve
public AbstractLinearSolver, AbstractPreconditioner, default_tangent_linear_solve

include("multigrid.jl")
export GeometricMultigridSolver

include("adaptivity.jl")
export AdaptivityLimits, AdaptivityPlan, adaptivity_summary, request_h_derefinement!,
       request_h_refinement!, request_p_derefinement!, request_p_refinement!
public HCoarseningCandidate, h_adaptation_axes, h_coarsening_candidates, p_degree_change,
       source_space

include("transition.jl")
export adapted_field, adapted_fields, transition
public SpaceTransition, derived_adaptivity_plan, source_leaves, target_space

include("transfer.jl")
export transfer_state

include("indicators.jl")
public adaptivity_plan, coefficient_coarsening_indicators, interface_jump_indicators,
       multiresolution_indicators, projection_coarsening_indicators

# Part VI. Postprocessing, verification, and output.
include("verification.jl")
export l2_error, relative_l2_error

include("postprocess.jl")
export plot_field, plot_field!, plot_mesh, plot_mesh!, sample_postprocess, write_pvd, write_vtk
public SampledMesh, SampledMeshSkeleton, SampledPostprocess, postprocess_supported,
       sample_mesh_skeleton

end

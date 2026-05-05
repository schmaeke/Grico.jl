using Test
import Grico

# This test intentionally freezes the first mature API boundary. Exported names
# are meant for ordinary `using Grico` workflows; qualified public names are
# supported as `Grico.name` but should not be injected into every user namespace.
const EXPECTED_EXPORTED = Set(Symbol[:AbstractField, :AdaptivityLimits, :AdaptivityPlan,
                                     :AffineProblem, :AssemblyPlan, :AutoLinearSolver,
                                     :AxisDegrees, :BoundaryFace,
                                     :ByLeafDegrees, :CartesianGrid, :CellValues,
                                     :CGSolver, :DegreePlusQuadrature, :Dirichlet, :Domain,
                                     :EmbeddedSurface,
                                     :FGMRESSolver, :FaceValues, :FieldLayout,
                                     :FiniteCellExtension, :FullTensorBasis, :GeneralOperator,
                                     :GeometricMultigridPreconditioner, :HpSpace,
                                     :IdentityPreconditioner, :ImplicitRegion, :IndefiniteOperator,
                                     :InterfaceValues,
                                     :JacobiPreconditioner, :KernelScratch, :LOWER, :MeanValue,
                                     :NONE, :NonsymmetricOperator, :OperatorWorkspace,
                                     :PhysicalDomain, :PhysicalMeasure, :PointQuadrature,
                                     :ResidualProblem, :ResidualWorkspace,
                                     :SPD, :ScalarField, :SegmentMesh, :SpaceOptions, :State,
                                     :SurfaceQuadrature, :SurfaceValues, :TensorProductValues,
                                     :TrunkBasis, :UPPER, :UniformDegree, :VectorField,
                                     :active_leaf_count, :active_leaves, :adapted_field,
                                     :adapted_fields, :adaptivity_summary, :add_boundary!,
                                     :add_boundary_bilinear!, :add_boundary_linear!, :add_cell!,
                                     :add_cell_bilinear!, :add_cell_linear!,
                                     :add_cell_quadrature!, :add_constraint!,
                                     :add_embedded_surface!, :add_interface!,
                                     :add_interface_bilinear!, :add_interface_linear!,
                                     :add_surface!, :add_surface_bilinear!,
                                     :add_surface_linear!, :add_surface_quadrature!, :apply,
                                     :apply!, :average, :avg, :block, :cell_apply!,
                                     :cell_center, :cell_degrees, :cell_diagonal!, :cell_lower,
                                     :cell_matrix!, :cell_residual!, :cell_rhs!, :cell_size,
                                     :cell_tangent_apply!, :cell_upper, :cell_volume, :coefficients,
                                     :compile, :component, :component_count, :constrain!,
                                     :derefine!, :extent, :face_apply!, :face_axis,
                                     :face_diagonal!, :face_matrix!, :face_residual!,
                                     :face_rhs!, :face_side, :face_tangent_apply!,
                                     :field_component_values, :field_count, :field_dof_count,
                                     :field_dof_range, :field_gradient, :field_layout, :field_name,
                                     :field_space, :field_values, :fields, :grad, :gradient,
                                     :grid, :inner, :interface_apply!, :interface_diagonal!,
                                     :interface_matrix!, :interface_residual!, :interface_rhs!,
                                     :interface_tangent_apply!, :is_full_tensor, :jump, :l2_error,
                                     :local_dof_index, :local_mode_count, :minus, :mode_terms,
                                     :normal, :normal_component, :normal_gradient,
                                     :operator_class, :origin, :plot_field, :plot_field!,
                                     :plot_mesh, :plot_mesh!,
                                     :plus, :point, :point_count, :refine!, :relative_l2_error,
                                     :request_h_derefinement!, :request_h_refinement!,
                                     :request_p_derefinement!, :request_p_refinement!, :residual,
                                     :residual!, :rhs, :rhs!, :sample_postprocess,
                                     :scalar_dof_count, :scratch_matrix, :scratch_vector,
                                     :shape_gradient, :shape_gradients, :shape_normal_gradient,
                                     :shape_value, :shape_values, :solve, :surface_apply!,
                                     :surface_diagonal!, :surface_matrix!, :surface_residual!,
                                     :surface_rhs!, :surface_tangent_apply!, :tangent_apply,
                                     :tangent_apply!,
                                     :tensor_axis_gradients, :tensor_axis_values, :tensor_degrees,
                                     :tensor_gradient!, :tensor_interpolate!, :tensor_local_modes,
                                     :tensor_mode_count, :tensor_mode_shape, :tensor_point_count,
                                     :tensor_project!, :tensor_project_gradient!,
                                     :tensor_quadrature_shape, :tensor_values, :transfer_state,
                                     :transition, :value, :weight, :write_pvd, :write_vtk, :∇,
                                     :⋅])

const EXPECTED_QUALIFIED_PUBLIC = Set(Symbol[:AbstractBasisFamily, :AbstractDegreePolicy,
                                             :AbstractDomain, :AbstractLinearSolver,
                                             :AbstractOperatorClass, :AbstractPreconditioner,
                                             :AbstractQuadrature,
                                             :AbstractQuadraturePolicy, :GaussLegendreRule,
                                             :Geometry, :GridSnapshot, :HCoarseningCandidate,
                                             :SampledMesh, :SampledMeshSkeleton,
                                             :SampledPostprocess, :SpaceTransition,
                                             :TensorQuadrature, :active_leaf, :adaptivity_plan,
                                             :axis_point_counts, :basis_family, :basis_mode_count,
                                             :basis_modes, :boundary_face_count,
                                             :boundary_face_spec, :cell_quadrature_shape,
                                             :check_snapshot, :check_space, :check_topology,
                                             :coefficient_coarsening_indicators, :compact,
                                             :compact!, :continuity_kind, :continuity_policy,
                                             :coordinate, :covering_neighbor,
                                             :default_tangent_linear_solve,
                                             :derived_adaptivity_plan, :dimension, :face_measure,
                                             :field_component_range, :finite_cell_quadrature,
                                             :first_child, :gauss_legendre_exact_degree,
                                             :gauss_legendre_rule, :global_cell_quadrature_shape,
                                             :h_adaptation_axes, :h_coarsening_candidates,
                                             :integrated_legendre_derivatives,
                                             :integrated_legendre_values,
                                             :integrated_legendre_values_and_derivatives!,
                                             :implicit_surface_quadrature, :interface_count,
                                             :interface_jump_indicators, :interface_spec,
                                             :is_active_leaf, :is_active_mode, :is_continuous_axis,
                                             :is_domain_boundary, :is_expanded, :is_mode_active,
                                             :is_periodic_axis, :is_tree_cell,
                                             :legendre_derivatives, :legendre_values,
                                             :legendre_values_and_derivatives!, :level,
                                             :local_modes, :logical_coordinate,
                                             :map_from_biunit_cube, :map_to_biunit_cube,
                                             :minimum_gauss_legendre_points,
                                             :multiresolution_indicators, :neighbor,
                                             :opposite_active_leaves, :p_degree_change, :parent,
                                             :periodic_axes, :postprocess_supported,
                                             :projection_coarsening_indicators, :root_cell_count,
                                             :root_cell_counts, :root_cell_total,
                                             :sample_mesh_skeleton, :snapshot, :source_leaves,
                                             :source_space, :split_axis, :stored_cell_count,
                                             :support_shape, :target_space])

function _set_delta(current::Set{Symbol}, expected::Set{Symbol})
  return (extra=sort!(collect(setdiff(current, expected))),
          missing=sort!(collect(setdiff(expected, current))))
end

@testset "API Surface" begin
  current_public = Set(filter(!=(:Grico), names(Grico; all=false)))
  current_exported = Set(name for name in current_public if Base.isexported(Grico, name))
  current_qualified_public = setdiff(current_public, current_exported)

  @test _set_delta(current_exported, EXPECTED_EXPORTED) == (extra=Symbol[], missing=Symbol[])
  @test _set_delta(current_qualified_public, EXPECTED_QUALIFIED_PUBLIC) ==
        (extra=Symbol[], missing=Symbol[])
  @test isempty(intersect(EXPECTED_EXPORTED, EXPECTED_QUALIFIED_PUBLIC))
  @test all(name -> Base.ispublic(Grico, name), union(EXPECTED_EXPORTED, EXPECTED_QUALIFIED_PUBLIC))
  @test !any(name -> Base.isexported(Grico, name), EXPECTED_QUALIFIED_PUBLIC)
end

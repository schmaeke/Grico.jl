using Test
using LinearAlgebra
using Grico

const TOL = 1.0e-12

function _polynomial_eval_allocation()
  values = zeros(Float64, 6)
  derivatives = zeros(Float64, 6)
  return _polynomial_eval_allocation(values, derivatives)
end

function _polynomial_eval_allocation(values, derivatives)
  Grico.legendre_values_and_derivatives!(0.25, 5, values, derivatives)
  Grico.integrated_legendre_values_and_derivatives!(0.25, 5, values, derivatives)
  return nothing
end

function _integrated_polynomial_value_allocation(values)
  Grico.integrated_legendre_values_and_derivatives!(0.25, 5, values, nothing)
  return nothing
end

function _polynomial_derivative_allocation(derivatives)
  Grico.legendre_values_and_derivatives!(0.25, 5, nothing, derivatives)
  Grico.integrated_legendre_values_and_derivatives!(0.25, 5, nothing, derivatives)
  return nothing
end

function _polynomial_kernel_allocation(values, derivatives)
  Grico._legendre_values!(0.25, 5, values)
  Grico._legendre_derivatives!(0.25, 5, derivatives)
  Grico._legendre_values_and_derivatives!(0.25, 5, values, derivatives)
  Grico._integrated_legendre_values!(0.25, 5, values)
  Grico._integrated_legendre_derivatives!(0.25, 5, derivatives)
  Grico._integrated_legendre_values_and_derivatives!(0.25, 5, values, derivatives)
  Grico._fe_basis_values!(0.25, 5, values)
  Grico._fe_basis_derivatives!(0.25, 5, derivatives)
  Grico._fe_basis_values_and_derivatives!(0.25, 5, values, derivatives)
  return nothing
end

function _integrate_monomial(rule, degree::Int)
  value = 0.0

  for point_index in 1:Grico.point_count(rule)
    x = Grico.point(rule, point_index)[1]
    value += Grico.weight(rule, point_index) * x^degree
  end

  return value
end

function _exact_interval_monomial(degree::Int)
  isodd(degree) && return 0.0
  return 2.0 / (degree + 1)
end

function _integrate_tensor_monomial(quadrature, exponents)
  value = 0.0

  for point_index in 1:Grico.point_count(quadrature)
    x = Grico.point(quadrature, point_index)
    monomial = 1.0

    @inbounds for axis in eachindex(exponents)
      monomial *= x[axis]^exponents[axis]
    end

    value += Grico.weight(quadrature, point_index) * monomial
  end

  return value
end

function _exact_tensor_monomial(exponents)
  value = 1.0

  @inbounds for exponent in exponents
    isodd(exponent) && return 0.0
    value *= 2.0 / (exponent + 1)
  end

  return value
end

_quadrature_tolerance(::Type{Float32}) = 5.0e-5

_quadrature_tolerance(::Type{Float64}) = 1.0e-12

@testset "Dense Local Kernels" begin
  matrix_data = [0.0 2.0 1.0; 1.0 -2.0 -3.0; 2.0 3.0 1.0]
  rhs = [1.0, -2.0, 0.5]
  factor = copy(matrix_data)
  pivots = zeros(Int, 3)
  solution = copy(rhs)
  Grico._dense_lu_factor!(factor, pivots)
  Grico._dense_lu_solve!(factor, pivots, solution)
  @test solution ≈ matrix_data \ rhs atol = TOL

  rhs_matrix = [1.0 0.0; -2.0 3.0; 0.5 -1.0]
  factor = copy(matrix_data)
  matrix_solution = copy(rhs_matrix)
  Grico._dense_lu_factor!(factor, pivots)
  Grico._dense_lu_solve!(factor, pivots, matrix_solution)
  @test matrix_solution ≈ matrix_data \ rhs_matrix atol = TOL

  @test_throws SingularException Grico._dense_lu_factor!([1.0 2.0; 2.0 4.0], zeros(Int, 2))

  spd_matrix = [4.0 1.0 2.0; 1.0 3.0 0.5; 2.0 0.5 5.0]
  spd_rhs = [1.0, 2.0, -1.0]
  cholesky_factor = copy(spd_matrix)
  cholesky_solution = copy(spd_rhs)
  Grico._dense_cholesky_factor!(cholesky_factor)
  Grico._dense_cholesky_solve!(cholesky_factor, cholesky_solution)
  @test cholesky_solution ≈ spd_matrix \ spd_rhs atol = TOL

  @test_throws PosDefException Grico._dense_cholesky_factor!([1.0 2.0; 2.0 1.0])
end

@testset "Polynomial Core" begin
  @test Grico.legendre_values(0.5, 2) ≈ [1.0, 0.5, -0.125] atol = TOL
  @test Grico.legendre_derivatives(0.5, 2) ≈ [0.0, 1.0, 1.5] atol = TOL
  @test Grico.integrated_legendre_values(-1.0, 4) ≈ [1.0, 0.0, 0.0, 0.0, 0.0] atol = TOL
  @test Grico.integrated_legendre_values(1.0, 4) ≈ [0.0, 1.0, 0.0, 0.0, 0.0] atol = TOL
  @test Grico.integrated_legendre_derivatives(0.0, 1) ≈ [-0.5, 0.5] atol = TOL

  values = zeros(Float32, 3)
  derivatives = zeros(Float32, 3)
  Grico.legendre_values_and_derivatives!(0.5f0, 2, values, derivatives)
  @test values ≈ Float32[1.0, 0.5, -0.125] atol = 1.0f-6
  @test derivatives ≈ Float32[0.0, 1.0, 1.5] atol = 1.0f-6
  derivative_only = zeros(Float64, 3)
  Grico.legendre_values_and_derivatives!(0.5, 2, nothing, derivative_only)
  @test derivative_only ≈ [0.0, 1.0, 1.5] atol = TOL
  Grico.integrated_legendre_values_and_derivatives!(0.0, 2, nothing, derivative_only)
  @test derivative_only ≈ [-0.5, 0.5, 0.0] atol = TOL

  _polynomial_eval_allocation()
  alloc_values = zeros(Float64, 6)
  alloc_derivatives = zeros(Float64, 6)
  _polynomial_eval_allocation(alloc_values, alloc_derivatives)
  _integrated_polynomial_value_allocation(alloc_values)
  _polynomial_derivative_allocation(alloc_derivatives)
  _polynomial_kernel_allocation(alloc_values, alloc_derivatives)
  @test @allocated(_polynomial_eval_allocation(alloc_values, alloc_derivatives)) == 0
  @test @allocated(_integrated_polynomial_value_allocation(alloc_values)) == 0
  @test @allocated(_polynomial_derivative_allocation(alloc_derivatives)) == 0
  @test @allocated(_polynomial_kernel_allocation(alloc_values, alloc_derivatives)) == 0
  @test_throws MethodError Grico.legendre_values_and_derivatives!(0.5, 2, values, derivatives)
  aliased = zeros(Float64, 3)
  @test_throws ArgumentError Grico.legendre_values_and_derivatives!(0.5, 2, aliased, aliased)
  @test_throws ArgumentError Grico.integrated_legendre_values_and_derivatives!(0.5, 2, aliased,
                                                                               aliased)

  @test_throws ArgumentError Grico.legendre_values(0.0, -1)
  @test_throws ArgumentError Grico.legendre_values(0.0, big(typemax(Int)) + 1)
  @test_throws ArgumentError Grico.integrated_legendre_values(0.0, -1)
  @test_throws ArgumentError Grico.legendre_values(Inf, 2)
  @test_throws ArgumentError Grico.legendre_derivatives(NaN, 2)
  @test_throws ArgumentError Grico.integrated_legendre_values(Inf, 2)
  @test_throws ArgumentError Grico.integrated_legendre_derivatives(NaN, 2)
  @test_throws ArgumentError Grico.legendre_values_and_derivatives!(Inf, 2, alloc_values,
                                                                    alloc_derivatives)
  @test_throws ArgumentError Grico.integrated_legendre_values_and_derivatives!(NaN, 2, alloc_values,
                                                                               alloc_derivatives)
end

@testset "Basis Modes" begin
  full = collect(Grico.basis_modes(Grico.FullTensorBasis(), (2, 1)))
  @test length(full) == 6
  @test (2, 1) in full

  trunk = collect(Grico.basis_modes(Grico.TrunkBasis(), (2, 2)))
  @test length(trunk) == 8
  @test (0, 2) in trunk
  @test (1, 2) in trunk
  @test (2, 1) in trunk
  @test !((2, 2) in trunk)

  @test Grico.basis_mode_count(Grico.FullTensorBasis(), (3, 2, 1)) == 24
  @test Grico.is_active_mode(Grico.TrunkBasis(), (3, 3, 3), (1, 1, 3))
  @test !Grico.is_active_mode(Grico.TrunkBasis(), (3, 3, 3), (2, 2, 1))

  for degrees in ((3,), (2, 2), (2, 1, 2), (2, 1, 1, 2))
    modes = collect(Grico.basis_modes(Grico.FullTensorBasis(), degrees))
    @test length(modes) == prod(degrees .+ 1)
  end

  @test Grico.is_active_mode(Grico.TrunkBasis(), (2,), (2,))
  @test Grico.is_active_mode(Grico.TrunkBasis(), (2, 2, 2), (1, 1, 2))
  @test !Grico.is_active_mode(Grico.TrunkBasis(), (2, 2, 2), (2, 2, 1))
  @test Grico.is_active_mode(Grico.TrunkBasis(), (3, 2, 1, 2), (1, 2, 1, 1))
  @test !Grico.is_active_mode(Grico.TrunkBasis(), (3, 2, 1, 2), (2, 2, 1, 2))
  @test Grico.basis_mode_count(Grico.TrunkBasis(), (1, 3)) ==
        length(collect(Grico.basis_modes(Grico.TrunkBasis(), (1, 3))))

  trunk_ordered = collect(Grico.basis_modes(Grico.TrunkBasis(), (4, 3, 2)))
  full_ordered = collect(Grico.basis_modes(Grico.FullTensorBasis(), (4, 3, 2)))
  @test trunk_ordered ==
        filter(mode -> Grico.is_active_mode(Grico.TrunkBasis(), (4, 3, 2), mode), full_ordered)

  for (basis, degrees) in ((Grico.FullTensorBasis(), (4,)), (Grico.FullTensorBasis(), (2, 3, 1)),
                           (Grico.TrunkBasis(), (3, 3)), (Grico.TrunkBasis(), (3, 2, 1, 2)))
    iterator = Grico.basis_modes(basis, degrees)
    collected = collect(iterator)
    @test length(iterator) == length(collected)
    @test Grico.basis_mode_count(basis, degrees) == length(collected)
  end

  oversized_mode = (big(typemax(Int)) + 1,)
  @test !Grico.is_active_mode(Grico.FullTensorBasis(), (2,), oversized_mode)
  @test !Grico.is_active_mode(Grico.TrunkBasis(), (2,), oversized_mode)
  @test !Grico.is_active_mode(Grico.TrunkBasis(), (typemax(Int) - 1, typemax(Int) - 1),
                              (typemax(Int) - 1, typemax(Int) - 1))
  @test_throws ArgumentError Grico.basis_modes(Grico.FullTensorBasis(), ())
  @test_throws ArgumentError Grico.basis_mode_count(Grico.TrunkBasis(), ())
  @test_throws ArgumentError Grico.is_active_mode(Grico.FullTensorBasis(), (), ())
  @test_throws ArgumentError Grico.basis_mode_count(Grico.FullTensorBasis(), (typemax(Int),))
  @test_throws ArgumentError Grico.basis_modes(Grico.FullTensorBasis(), (typemax(Int),))
  @test_throws ArgumentError Grico.basis_mode_count(Grico.TrunkBasis(), (typemax(Int),))
  @test_throws ArgumentError Grico.basis_modes(Grico.TrunkBasis(), (typemax(Int),))
end

@testset "Quadrature" begin
  rule = Grico.gauss_legendre_rule(3)
  @test Grico.dimension(rule) == 1
  @test Grico.point_count(rule) == 3
  @test Grico.gauss_legendre_exact_degree(3) == 5
  @test Grico.minimum_gauss_legendre_points(5) == 3
  @test Grico.gauss_legendre_exact_degree(typemax(Int) ÷ 2 + 1) == typemax(Int)
  @test Grico.minimum_gauss_legendre_points(typemax(Int)) == typemax(Int) ÷ 2 + 1
  @test_throws ArgumentError Grico.gauss_legendre_exact_degree(typemax(Int) ÷ 2 + 2)
  @test_throws ArgumentError Grico.minimum_gauss_legendre_points(big(typemax(Int)) + 1)

  manual_rule = Grico.GaussLegendreRule([0.0], [2.0])
  @test manual_rule isa Grico.GaussLegendreRule{Float64}
  @test Grico.point(manual_rule, 1) == (0.0,)
  @test Grico.weight(manual_rule, 1) == 2.0

  for degree in 0:5
    @test _integrate_monomial(rule, degree) ≈ _exact_interval_monomial(degree) atol = 1.0e-12
  end

  quadrature = Grico.TensorQuadrature((2, 3))
  @test Grico.dimension(quadrature) == 2
  @test Grico.point_count(quadrature) == 6
  @test Grico.axis_point_counts(quadrature) == (2, 3)
  @test Grico.coordinate(quadrature, 4, 1) ≈ Grico.point(quadrature, 4)[1] atol = TOL
  @test Grico.coordinate(quadrature, 4, 2) ≈ Grico.point(quadrature, 4)[2] atol = TOL
  rule_x = Grico.gauss_legendre_rule(2)
  rule_y = Grico.gauss_legendre_rule(3)
  @test Grico.point(quadrature, 4) == (Grico.point(rule_x, 2)[1], Grico.point(rule_y, 2)[1])
  @test Grico.weight(quadrature, 4) ≈ Grico.weight(rule_x, 2) * Grico.weight(rule_y, 2) atol = TOL

  for exponents in ((0, 0), (2, 0), (0, 4), (1, 2), (2, 3))
    exact = _exact_tensor_monomial(exponents)
    atol = exact == 0.0 ? 1.0e-12 : 1.0e-12 * abs(exact)
    @test _integrate_tensor_monomial(quadrature, exponents) ≈ exact atol = atol
  end

  point_quadrature = Grico.PointQuadrature([(0.0, -1.0), (1.0, 0.0), (0.5, 1.0)], [0.25, 0.5, 1.25])
  @test Grico.dimension(point_quadrature) == 2
  @test Grico.point_count(point_quadrature) == 3
  @test Grico.coordinate(point_quadrature, 2, 1) ≈ 1.0 atol = TOL
  @test Grico.coordinate(point_quadrature, 3, 2) ≈ 1.0 atol = TOL
  @test sum(Grico.weight(point_quadrature, i) for i in 1:Grico.point_count(point_quadrature)) ≈ 2.0 atol = TOL

  empty_quadrature = Grico.TensorQuadrature(())
  @test Grico.dimension(empty_quadrature) == 0
  @test Grico.point_count(empty_quadrature) == 1
  @test Grico.point(empty_quadrature, 1) == ()
  @test Grico.weight(empty_quadrature, 1) == 1.0

  quadrature_3d = Grico.TensorQuadrature((2, 2, 2))
  @test _integrate_tensor_monomial(quadrature_3d, (1, 2, 3)) ≈ _exact_tensor_monomial((1, 2, 3)) atol = 1.0e-12

  quadrature_4d = Grico.TensorQuadrature((2, 2, 2, 2))
  @test _integrate_tensor_monomial(quadrature_4d, (0, 2, 1, 2)) ≈
        _exact_tensor_monomial((0, 2, 1, 2)) atol = 1.0e-12

  @test_throws ArgumentError Grico.PointQuadrature([(0.0,)], Float64[])
  @test_throws ArgumentError Grico.gauss_legendre_rule(0)
  @test_throws ArgumentError Grico.GaussLegendreRule{Float64}([2.0], [1.0])
  @test_throws ArgumentError Grico.GaussLegendreRule{Float64}([0.0], [-1.0])

  for (T, point_count) in ((Float32, 8), (Float64, 8), (Float64, 16))
    high_order_rule = Grico.gauss_legendre_rule(T, point_count)
    tolerance = _quadrature_tolerance(T)

    for point_index in 1:Grico.point_count(high_order_rule)
      mirror = Grico.point_count(high_order_rule) - point_index + 1
      @test Grico.point(high_order_rule, point_index)[1] ≈ -Grico.point(high_order_rule, mirror)[1] atol = tolerance
      @test Grico.weight(high_order_rule, point_index) ≈ Grico.weight(high_order_rule, mirror) atol = tolerance
    end

    for degree in 0:Grico.gauss_legendre_exact_degree(point_count)
      @test _integrate_monomial(high_order_rule, degree) ≈ _exact_interval_monomial(degree) atol = tolerance
    end
  end
end

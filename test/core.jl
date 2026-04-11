using Test
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

  _polynomial_eval_allocation()
  alloc_values = zeros(Float64, 6)
  alloc_derivatives = zeros(Float64, 6)
  _polynomial_eval_allocation(alloc_values, alloc_derivatives)
  _integrated_polynomial_value_allocation(alloc_values)
  @test @allocated(_polynomial_eval_allocation(alloc_values, alloc_derivatives)) == 0
  @test @allocated(_integrated_polynomial_value_allocation(alloc_values)) == 0
  @test_throws MethodError Grico.legendre_values_and_derivatives!(0.5, 2, values, derivatives)

  @test_throws ArgumentError Grico.legendre_values(0.0, -1)
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

  for (basis, degrees) in ((Grico.FullTensorBasis(), (4,)), (Grico.FullTensorBasis(), (2, 3, 1)),
                           (Grico.TrunkBasis(), (3, 3)), (Grico.TrunkBasis(), (3, 2, 1, 2)))
    iterator = Grico.basis_modes(basis, degrees)
    collected = collect(iterator)
    @test length(iterator) == length(collected)
    @test Grico.basis_mode_count(basis, degrees) == length(collected)
  end
end

@testset "Quadrature" begin
  rule = Grico.gauss_legendre_rule(3)
  @test Grico.dimension(rule) == 1
  @test Grico.point_count(rule) == 3
  @test Grico.gauss_legendre_exact_degree(3) == 5
  @test Grico.minimum_gauss_legendre_points(5) == 3

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

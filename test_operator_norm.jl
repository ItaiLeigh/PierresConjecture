using Test
include("operator_norm.jl")
using .OperatorNorm

@testset "operator_norm basic properties" begin
    D = 4
    M = randn(ComplexF64, D, D)
    norm_val = OperatorNorm.operator_norm(M)
    @test isapprox(norm_val, opnorm(M, 2); atol=1e-12)
    @test norm_val >= 0
end

@testset "build_total_operator shape and type" begin
    K, d, D = 2, 2, 2
    U = [Matrix{ComplexF64}(I, D, D) for _ in 1:K]
    Γ = [Matrix{ComplexF64}(I, d, d) for _ in 1:K]
    H = build_total_operator(U, Γ)
    @test size(H, 1) == d*D*D
    @test size(H, 2) == d*D*D
    @test eltype(H) == ComplexF64
end

@testset "build_adjusted_total_operator shape and type" begin
    K, d, D = 2, 2, 2
    U = [Matrix{ComplexF64}(I, D, D) for _ in 1:K]
    Γ = [Matrix{ComplexF64}(I, d, d) for _ in 1:K]
    H = build_adjusted_total_operator(U, Γ)
    @test size(H, 1) == d*D*D*D*D
    @test size(H, 2) == d*D*D*D*D
    @test eltype(H) == ComplexF64
end

@testset "constants exported" begin
    @test typeof(OperatorNorm.commute_designated_register) == String
    @test typeof(OperatorNorm.commute_control_register) == String
    @test typeof(OperatorNorm.commute_erase1) == String
    @test typeof(OperatorNorm.block_commute_half) == String
end

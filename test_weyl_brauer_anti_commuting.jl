using Test
include("weyl_brauer_anti_commuting.jl")
using .WeylBrauerAntiCommuting

@testset "gamma_matrices basic properties" begin
    K, d = 3, 8
    Γ = gamma_matrices(K, d)
    @test length(Γ) == K
    @test all(size(G) == (d, d) for G in Γ)
    @test all(ishermitian(G) for G in Γ)
    @test all(all(abs.(eigvals(G)) .≈ 1.0) for G in Γ)
end

@testset "gamma_matrices anti-commutation" begin
    K, d = 3, 8
    Γ = gamma_matrices(K, d)
    for i = 1:K, j = 1:i-1
        anti = norm(Γ[i] * Γ[j] + Γ[j] * Γ[i])
        @test isapprox(anti, 0.0; atol=1e-12)
    end
end

@testset "gamma_matrices errors" begin
    @test_throws ErrorException gamma_matrices(5, 4)  # d=4, max K=4
    @test_throws ErrorException gamma_matrices(3, 6)  # d not power of 2
end

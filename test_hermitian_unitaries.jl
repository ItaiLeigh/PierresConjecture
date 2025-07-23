using Test
include("hermitian_unitaries.jl")
using .HermitianUnitaries

@testset "random_hermitian_unitary" begin
    D = 4
    U = HermitianUnitaries.random_hermitian_unitary(D)
    @test size(U) == (D, D)
    @test ishermitian(U)
    eigvals_U = eigen(U).values
    @test all(abs.(eigvals_U) .≈ 1.0)
end

@testset "project_back_to_hermitian_unitary" begin
    D = 4
    U = randn(ComplexF64, D, D)
    U_projected = HermitianUnitaries.project_back_to_hermitian_unitary(U)
    @test ishermitian(U_projected)
    eigvals_U_projected = eigen(U_projected).values
    @test all(abs.(eigvals_U_projected) .≈ 1.0)
    @test size(U_projected) == size(U)
end

@testset "draw_U_k" begin
    K, D = 3, 4
    U_list = HermitianUnitaries.draw_U_k(K, D)
    @test length(U_list) == K
    @test all(size(U) == (D, D) for U in U_list)
    @test all(ishermitian(U) for U in U_list)
    @test all(all(abs.(eigen(U).values) .≈ 1.0) for U in U_list)
end

@testset "draw_H_k" begin
    K, D = 3, 4
    eig_bound = 2.0
    H = HermitianUnitaries.draw_H_k(K, D, eig_bound)
    @test size(H) == (D, D)
    @test ishermitian(H)
    eigvals_H = eigen(H).values
    @test all(abs.(eigvals_H) .<= eig_bound)
end

using Test
using Symbolics
using LinearAlgebra

include("symbolic_differentiation.jl")
using .SymbolicDifferentiation

@testset "symbolic_optimization_poly output types and shapes" begin
    D = 2
    d = 2
    K = 2
    Γ = [Matrix{Float64}(I, d, d) for _ in 1:K]
    target, U, x = SymbolicDifferentiation.symbolic_optimization_poly(D, Γ)
    @test typeof(target) <: Symbolics.Num
    @test isa(U, Vector)
    @test isa(x, Array)
    @test size(U[1]) == (D, D)
    @test length(x) == d*D*D
end

@testset "evaluate_symbolic substitution" begin
    @variables x1 x2
    poly = x1 + x2^2
    val1 = SymbolicDifferentiation.evaluate_symbolic([x1, x2], [0.0, 0.0], poly)
    @test val1 == 0
    val2 = SymbolicDifferentiation.evaluate_symbolic([x1, x2], [1.0, 2.0], poly)
    @test val2 == 5.0
    U = [Symbolics.variables(Symbol("u$k"), 1:2, 1:2) for k in 1:2]
    inputs = [fill(1.0, 2, 2), fill(2.0, 2, 2)]
    entry_sum = sum(sum(U))
    val3 = SymbolicDifferentiation.evaluate_symbolic(U, inputs, entry_sum)
    @test val3 == sum(sum(inputs))
end

@testset "gradient computation" begin
    D = 2
    d = 2
    K = 2
    Γ = [Matrix{Float64}(I, d, d) for _ in 1:K]
    target, U_sym, x_sym = SymbolicDifferentiation.symbolic_optimization_poly(D, Γ)
    U_val = [ones(D,D) for _ in 1:K]
    x_val = ones(Float64, d*D*D)
    grad_U, grad_x = SymbolicDifferentiation.symbolic_optimization_grad(D, Γ)
    val_target = SymbolicDifferentiation.evaluate_symbolic(U_sym, U_val, target)
    val_target = SymbolicDifferentiation.evaluate_symbolic(x_sym, x_val, val_target)
    val_gradient = [SymbolicDifferentiation.evaluate_symbolic(U_sym, U_val, grad_Ui) for grad_Ui in grad_U] 
    val_gradient = [SymbolicDifferentiation.evaluate_symbolic(x_sym, x_val, val_gradient_i) for val_gradient_i in val_gradient]

    @test isa(grad_U, Vector)
    @test isa(grad_x, Vector)
end


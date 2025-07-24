module SymbolicDifferentiation

using Symbolics
using LinearAlgebra
using Random
Random.seed!(123)

# Symbolic variables currently support computations in R, not in C.

function symbolic_optimization_poly(D::Int, Γ::Vector{Matrix{Float64}})
    K = length(Γ)
    d = size(Γ[1], 1)
    U = [Symbolics.variables(Symbol("u$k"), 1:D, 1:D) for k in 1:K]
    x = Symbolics.variables(:x, 1:d*D*D)

    Id = Matrix{Float64}(I, d, d)
    ID = Matrix{Float64}(I, D, D)

    C = zeros(Symbolics.Num, d*D*D, d*D*D)

    for k in 1:K
        Uk = U[k]
        Γk = Γ[k]
        term1 = kron(Γk, kron(Uk, ID) + kron(ID, Uk))
        term2 = kron(Id, kron(Uk, Uk))
        C += term1 + term2
    end
    return x' * C * x, U, x # when we move to complex variables, change ' to adjoint.
end

function symbolic_optimization_grad(D::Int, Γ::Vector{Matrix{Float64}})
    target, U, x = symbolic_optimization_poly(D, Γ)
    U_flat = vcat([vec(Ui) for Ui in U]...)
    grad_U = Symbolics.gradient(target, U_flat)
    grad_x = Symbolics.gradient(target, x)
    return grad_U, grad_x
end

function evaluate_symbolic(
    U_symbol::Vector{Matrix{Symbolics.Num}},
    U_value::Vector{Matrix{Float64}},
    expression::Symbolics.Num
)
    U_sym_flat = vcat([vec(U) for U in U_symbol]...)
    U_val_flat = vcat([vec(U) for U in U_value]...)
    subs_dict = Dict()
    for (var, val) in zip(U_sym_flat, U_val_flat)
        subs_dict[var] = val
    end
    return Symbolics.substitute(expression, subs_dict)
end

function evaluate_symbolic(
    x_symbol::Vector{Symbolics.Num},
    x_value::Vector{Float64},
    expression::Symbolics.Num
)
    x_sym_flat = vec(x_symbol)
    x_val_flat = vec(x_value)
    subs_dict = Dict()
    for (var, val) in zip(x_symbol, x_value)
        subs_dict[var] = val
    end
    return Symbolics.substitute(expression, subs_dict)
end

export symbolic_optimization_poly, evaluate_symbolic, symbolic_optimization_grad

end # module SymbolicDifferentiation
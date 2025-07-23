module HermitianUnitaries

using LinearAlgebra
using Random


function round_to_hermitian(U::Matrix{ComplexF64}; atol=1e-12)
    if !isapprox(U, U', atol=atol)
        @warn "Input matrix is not Hermitian within tolerance, but will be symmetrized."
    end
    return (U + U') / 2
end

function random_hermitian_unitary(D::Int)
    Q, _ = qr(randn(ComplexF64, D, D))  # Random unitary
    diag_vals = rand([-1.0, 1.0], D)    # ±1 eigenvalues
    U = Q * Diagonal(diag_vals) * Q'
    return round_to_hermitian(U)
end

function draw_U_k(K::Int, D::Int; shared::Bool=false, commuting::Bool=false)
    # WBs = gamma_matrices(K, D)
    # V, _ = qr(randn(ComplexF64, D, D))  # Random unitary
    # return [V'*WBs[i]*V for i in 1:K]
    # return WBs
    if shared
        U = random_hermitian_unitary(D)
        return [U for _ in 1:K]
    elseif commuting
        # Shared eigenbasis
        Q, _ = qr(randn(ComplexF64, D, D))  # random unitary
        return [round_to_hermitian(Q * Diagonal(rand([-1.0, 1.0], D)) * Q') for _ in 1:K]
    else
        Hs = [random_hermitian_unitary(D) for _ in 1:K]
        #println("||H_1-H_2|| = ", opnorm(Hs[1]-Hs[2], 2))
        #println("||H_1H_2-H_2H_1|| = ", opnorm((Hs[1]*Hs[2])-(Hs[2]*Hs[1]), 2))
        return Hs
    end
end


function rand_floats(a, b, n)
    a .+ (b - a) .* rand(n)
end

function draw_H_k(K::Int, D::Int, eig_bound::Float64)
    Q, _ = qr(randn(ComplexF64, D, D))  # Random unitary
    diag_vals = rand_floats(-eig_bound, eig_bound, D)    # eigenvalues
    return round_to_hermitian(Q * Diagonal(diag_vals) * Q')
end

function project_back_to_hermitian_unitary(U::Matrix{ComplexF64})
    # Make U Hermitian
    U = (U + U') / 2
    eigvals_U, eigvecs_U = eigen(U)
    return round_to_hermitian(eigvecs_U * Diagonal(sign.(eigvals_U)) * eigvecs_U')
end

export random_hermitian_unitary, project_back_to_hermitian_unitary, draw_U_k, draw_H_k

end  # module HermitianUnitaries
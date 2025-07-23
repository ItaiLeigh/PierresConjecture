using LinearAlgebra, Random, Statistics
using ProgressMeter
using Plots


# --------------------------
# Part 1: Hermitian Unitaries U_k ∈ ℂ^{D×D}
# --------------------------

function random_hermitian_unitary(D::Int)
    Q, _ = qr(randn(ComplexF64, D, D))  # Random unitary
    diag_vals = rand([-1.0, 1.0], D)    # ±1 eigenvalues
    return Q * Diagonal(diag_vals) * Q'
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
        return [Q * Diagonal(rand([-1.0, 1.0], D)) * Q' for _ in 1:K]
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

function draw_H_k(K::Int, D::Int, eig_bound::ComplexF64)
    Q, _ = qr(randn(ComplexF64, D, D))  # Random unitary
    diag_vals = rand_floats(-eig_bound, eig_bound, D)    # eigenvalues
    return Q * Diagonal(diag_vals) * Q'
end

function project_back_to_hermitian_unitary(U::Matrix{ComplexF64})
    # Make U Hermitian
    U = (U + U') / 2
    eigvals_U, eigvecs_U = eigen(U)
    return eigvecs_U * Diagonal(sign.(eigvals_U)) * eigvecs_U'
end

using Test

@testset "project_back_to_hermitian_unitary tests" begin
    # Test 1: Check if the function returns a Hermitian matrix
    U = randn(ComplexF64, 4, 4)
    U_projected = project_back_to_hermitian_unitary(U)
    @test all(U_projected .≈ U_projected')

    # Test 2: Check if the eigenvalues are ±1
    eigvals_U_projected = eigen(U_projected).values
    @test all(abs.(eigvals_U_projected) .≈ 1.0)

    # Test 3: Check if the function preserves the shape of the matrix
    @test size(U_projected) == size(U)

    # Test 4: Check if the function works for a Hermitian matrix input
    U_hermitian = (U + U') / 2
    U_projected_hermitian = project_back_to_hermitian_unitary(U_hermitian)
    @test all(U_projected_hermitian ≈ U_projected_hermitian')
    @test all(abs.(eigen(U_projected_hermitian).values) .≈ 1.0)
end





# --------------------------
# Part 2: Weyl–Brauer Anti-commuting Γ_k ∈ ℂ^{d×d}
# --------------------------

function s_x()
    ComplexF64[0 1; 1 0]
end

function s_y()
    ComplexF64[0 -im; im 0]
end

function s_z()
    ComplexF64[1 0; 0 -1]
end

function I2()
    ComplexF64[1 0; 0 1]
end

function tensor_self(A::Matrix{ComplexF64}, j::Int)
    if j < 1
        return ComplexF64[1.0]  # scalar 1
    end
    result = A
    for _ in 2:j
        result = kron(result, A)
    end
    return result
end

# Returns the i-th Weyl–Brauer generator in dimension d = 2^m
function operator_n_i(d::Int, i::Int)
    m = Int(log2(d))
    if i < 1 || i > 2m
        error("Index i=$i out of range for d=$d (max 2m = $(2m))")
    end
    base = iseven(i) ? s_y() : s_x()
    pref_len = div(i - 1, 2)
    prefix = tensor_self(s_z(), pref_len)
    suffix = tensor_self(I2(), m - 1 - pref_len)
    return kron(kron(prefix, base), suffix)
end

function gamma_matrices(K::Int, d::Int)
    if log2(d) % 1 != 0
        error("Dimension d=$d must be a power of 2")
    end
    m = Int(log2(d))
    max_K = 2 * m
    if K > max_K
        error("Cannot construct $K anti-commuting Hermitian unitaries in dimension $d (max is $max_K).")
    end
    return [operator_n_i(d, i) for i in 1:K]
end

# --------------------------
# Part 3: Large Operator & Operator Norm
# --------------------------

function build_total_operator(U::Vector{Matrix{ComplexF64}}, Γ::Vector{Matrix{ComplexF64}})
    K = length(U)
    d, D = size(Γ[1], 1), size(U[1], 1)
    Id = Matrix{ComplexF64}(I, d, d)
    ID = Matrix{ComplexF64}(I, D, D)

    total = zeros(ComplexF64, d*D*D, d*D*D)

    for k in 1:K
        term1 = kron(Γ[k], kron(U[k], ID) + kron(ID, U[k]))
        term2 = kron(Id, kron(U[k], U[k]))
        total += term1 + term2
        # total += term1 
    end

    return total
end

function direct_sum(A::Matrix{ComplexF64}, B::Matrix{ComplexF64})
    m, n = size(A)
    p, q = size(B)

    C = zeros(eltype(A), m+p, n+q)
    C[1:m, 1:n] = A
    C[m+1:m+p, n+1:n+q] = B

    return C
end

const commute_designated_register = "commute_designated_register"
# (adjusted_profile="commute_designated_register":) Using U'_1:=U_1\otimes ID and U'_i:=ID\otimes U_i for i>1, to check if the norm has to increase
#Looks like it does only grow by this commutisation operation!
const commute_control_register = "commute_control_register"
# (adjusted_profile="commute_control_register":) Using U'_1:=C_0U_1 U'_i:=C_1U_i where C_0U and C_1U are the controlled versions of U, controlled on either 0 or 1 in the additional qubit.
#In this case there WAS a negative difference, albeit a very small one that might just be a rounding error (-1.7....e-15). But this repeats...
const commute_erase1 = "commute_erase1"
# (adjusted_profile="commute_erase1":) Using U'_1:=Id or -Id, depending on the maximal absolute-value eigenvalue's eginevector's reyleigh quotient's sign of \Gamma1\otimes(U_1\otimes Id + Id\otimes U_1) + Id\otimes U_1\otimes U_1
const block_commute_half = "block_commute_half"
# (adjusted_profile="block_commute_half":) Using U'_i=C_0U_i for i<k/2 and U'_i=C_1U_i for the rest
function build_adjusted_total_operator(U::Vector{Matrix{ComplexF64}}, Γ::Vector{Matrix{ComplexF64}}, adjusted_profile::String=commute_designated_register)
    K = length(U)
    d, D = size(Γ[1], 1), size(U[1], 1)
    Id = Matrix{ComplexF64}(I, d, d)
    ID = Matrix{ComplexF64}(I, D, D)
    D_adjusted = 0
    if adjusted_profile == commute_designated_register
        global D_adjusted = D*D
    elseif adjusted_profile == commute_control_register
        global D_adjusted = 2*D
    elseif adjusted_profile == commute_erase1
        global D_adjusted = D
    elseif adjusted_profile == block_commute_half
        global D_adjusted = 2*D
    end
    # D_adjusted = designated_register ? D*D : 2*D
    IDD = Matrix{ComplexF64}(I, D_adjusted, D_adjusted)

    total = zeros(ComplexF64, d*D_adjusted*D_adjusted, d*D_adjusted*D_adjusted)

    function map1(U1::Matrix{ComplexF64})
        if adjusted_profile == commute_designated_register
            return kron(U1, ID)
        elseif adjusted_profile == commute_control_register
            return direct_sum(U1, ID)
        elseif adjusted_profile == commute_erase1
            op_norm_with_ID = operator_norm(build_total_operator([[IDD]; U[2:end]], Γ))
            op_norm_with_minusID = operator_norm(build_total_operator([[-IDD]; U[2:end]], Γ))
            println("The larger operator norm is given by: ", op_norm_with_ID < op_norm_with_minusID ? "-ID" : "+ID")
            return op_norm_with_ID < op_norm_with_minusID ? -IDD : IDD
        elseif adjusted_profile == block_commute_half
            return kron(U1, ID)
        end
    end

    function mapi(Ui::Matrix{ComplexF64})
        if adjusted_profile == commute_designated_register
            return kron(ID, Ui)
        elseif adjusted_profile == commute_control_register
            return direct_sum(ID, Ui)
        elseif adjusted_profile == commute_erase1
            return Ui
        elseif adjusted_profile == block_commute_half
            #TODO: get the i and k as well to decide which control to add to Ui and add the correct one
        end
    end
    
    term1 = kron(Γ[1], kron(map1(U[1]), IDD) + kron(IDD, map1(U[1])))
    term2 = kron(Id, kron(map1(U[1]), map1(U[1])))
    total += term1 + term2

    for k in 2:K
        term1 = kron(Γ[k], kron(mapi(U[k]), IDD) + kron(IDD, mapi(U[k])))
        term2 = kron(Id, kron(mapi(U[k]), mapi(U[k])))
        total += term1 + term2
        # total += term1 
    end

    return total
end

function operator_norm(M::Matrix{ComplexF64})
    return opnorm(M, 2)
end



# --------------------------
# Main Experiment
# --------------------------

function approx_subset(subset, superset, ϵ)
    all(x -> any(y -> abs(x - y) ≤ ϵ, superset), subset)
end

function run_experiments(K::Int, d::Int, D::Int; m::Int = 100, shared_U::Bool = false, commuting::Bool = false, adjusted_profile::String=commute_designated_register)
    println("Sanity check: Anti-commutation of Gamma matrices")
    Γ = gamma_matrices(K, d)
    for i = 1:K, j = 1:i-1
        c = norm(Γ[i] * Γ[j] + Γ[j] * Γ[i])
        println("||Γ[$i]Γ[$j] + Γ[$j]Γ[$i]|| = $c")
    end

    println("\nRunning $m experiments...")
    norms = Float64[]
    norms_adjusted = Float64[]
    @showprogress 1 "Computing operator norms..." for trial in 1:m
        U = draw_U_k(K, D; shared=shared_U, commuting=commuting)
        H = build_total_operator(U, Γ)
        H_adjusted = build_adjusted_total_operator(U, Γ, adjusted_profile)
        push!(norms, operator_norm(H))
        push!(norms_adjusted, operator_norm(H_adjusted))
    end

    println("\nResults over $m experiments:")
    println("  Max norm     = ", maximum(norms))
    println("  Min norm     = ", minimum(norms))
    println("  Median norm  = ", median(norms))
    println("  Average norm = ", mean(norms))

    println("\nResults over $m adjusted experiments:")
    println("  Max norm     = ", maximum(norms_adjusted))
    println("  Min norm     = ", minimum(norms_adjusted))
    println("  Median norm  = ", median(norms_adjusted))
    println("  Average norm = ", mean(norms_adjusted))

    println("\nComparing to adjusted operators:")
    println(" Max norm difference = ", maximum([a-b for (a,b) in zip(norms_adjusted, norms)]))
    println(" Min norm difference = ", minimum([a-b for (a,b) in zip(norms_adjusted, norms)]))

    # Plotting
    ymin = minimum(norms) - 1
    ymax = maximum(norms) + 1
    ymin = minimum([ymin, 2 * sqrt(K)-0.1, K-0.1])
    threshold = K + 2 * sqrt(K)
    threshold2 = 2 * sqrt(K)
    ymax = maximum([ymax, threshold+0.1])
    plot(
        1:m, norms,
        seriestype = :scatter,
        xlabel = "Trial",
        ylabel = "Operator Norm",
        title = "Operator Norms over $m Experiments",
        label = "Observed Norms",
        legend = :topright,
        markersize = 3,
        ylims = (ymin, ymax),
    )
    hline!([
        K,
        K+2,
        threshold,
        threshold2
    ],
    label = ["K = $K", "K + 2 = $(K+2)", "K + 2√K = $(round(threshold, digits=3))", "2√K = $(round(threshold2, digits=3))"],
    linestyle = [:dash :dash :dash :dash],
    color = [:green :red :blue :black])
end

# --------------------------
# Entry Point
# --------------------------

K, d, D = 3, 4, 4 #2, 2, 25
run_experiments(K, d, D; m=300, shared_U=false, commuting=false)
#run_experiments(K, d, D; m=200, shared_U=false, commuting=false, adjusted_profile=commute_erase1)

# Γ = gamma_matrices(K, d)
# U = draw_U_k(K, D; shared=false, commuting=false)
# H = build_total_operator(U, Γ)
# H_adjusted = build_adjusted_total_operator(U, Γ)
##included = approx_subset(eigvals(H), eigvals(H_adjusted), 0.01)
##println("Are the original operator's eigenvalues contained in the adjusted one's? $included") #false!
# random_vec = randn(ComplexF64, d*D*D)
# random_vec ./= norm(random_vec)
# random_vec_adjusted = zeros(ComplexF64, d*D*D*D*D)
# function kron_to_index(i::Int, j::Int, l::Int, adjusted::Bool=false)
#     if not adjusted
#         return (i-1)*
#     end
# for i in 1:d
#     for j in 1:D
#         for l in 1:D
#             random_vec_adjusted[(i*D*D*D*D) + (j*D*D*D)]
#         end
#     end
# end

#Checking if the general conjecture of making one Hermitian-unitary commute with the rest, when in tensor-product with Hermitians only increases the operator norm.
#It does not! It can strictly decrease it!
# ID = Matrix{ComplexF64}(I, D, D)
# dif = 0
# A1 = 0
# B = 0
# A2 = 0
# while dif >= 0
#     global A1 = draw_U_k(K, D; shared=false, commuting=false)
#     global B = draw_H_k(K, D, ComplexF64(3.0*K)) #draw_U_k(K,D; shared=false, commuting=false) + draw_U_k(K,D; shared=false, commuting=false)
#     global A2 = [kron(ID, a) for a in A1]
#     global A2[1] = kron(A1[1], ID)
#     global dif = operator_norm(sum(pair -> kron(pair[1], pair[2]), zip(A2, B))) - operator_norm(sum(pair -> kron(pair[1],pair[2]), zip(A1, B)))
#     # if dif <= 0.000001
#     #     break
#     # end
# end
# println("The difference ||C'||-||C|| = $dif")
# println("A1 = $A1")
# println("B = $B")

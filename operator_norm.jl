module OperatorNorm

using LinearAlgebra

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

export operator_norm, commute_designated_register, commute_control_register, commute_erase1, block_commute_half, build_total_operator, build_adjusted_total_operator

end  # module OperatorNorm

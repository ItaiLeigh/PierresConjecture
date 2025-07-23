module WeylBrauerAntiCommuting

using LinearAlgebra

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

export gamma_matrices

end # module WeylBrauerAntiCommuting
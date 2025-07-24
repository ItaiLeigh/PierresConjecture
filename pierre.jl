using LinearAlgebra, Random, Statistics
using ProgressMeter
using Plots

include("hermitian_unitaries.jl")
include("weyl_brauer_anti_commuting.jl")
include("operator_norm.jl")

using .HermitianUnitaries
using .OperatorNorm
using .WeylBrauerAntiCommuting

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
        push!(norms, OperatorNorm.operator_norm(H))
        push!(norms_adjusted, OperatorNorm.operator_norm(H_adjusted))
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
run_experiments(K, d, D; m=50, shared_U=false, commuting=false)

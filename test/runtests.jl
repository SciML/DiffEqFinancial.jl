using Pkg
using DiffEqFinancial, Statistics, StochasticDiffEq
using Test

const GROUP = get(ENV, "GROUP", "all")

function activate_nopre_env()
    Pkg.activate("nopre")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

@testset "Interface Compatibility" begin
    include("interface_tests.jl")
end

@testset "DiffEqFinancial.jl" begin
    if GROUP == "all" || GROUP == "core"
        @testset "Core Tests" begin
            u0 = [1.0; 0.5]
            σ = 0.25
            prob = HestonProblem(1.0, 1.0, σ, 1.0, 1.0, u0, (0.0, 1.0))
            sol = solve(prob, SRIW1(), adaptive = false, dt = 1 / 100)

            prob = BlackScholesProblem((t) -> t^2, (t, u) -> 1.0, σ, 0.5, (0.0, 1.0))
            sol = solve(prob, SRIW1())

            r = 0.03
            sigma = 0.2
            S0 = 100
            t = 0
            T = 1.0
            days = 252
            dt = 1 / days

            prob = GeometricBrownianMotionProblem(r, sigma, S0, (t, T))
            sol = solve(prob, EM(); dt = dt)
            monte_prob = EnsembleProblem(prob)
            sol = solve(monte_prob, EM(); dt = dt, trajectories = 1000000)
            us = [sol[i].u for i in eachindex(sol)]
            simulated = mean(us)

            tsteps = collect(0:dt:T)
            expected = S0 * exp.(r * tsteps)
            testerr = sum(abs2.(simulated .- expected))
            @test testerr < 2.5e-1
        end

        # Interface tests for type genericity
        @testset "Interface Compatibility" begin
            include("interface_tests.jl")
        end
    end

    if GROUP == "all" || GROUP == "nopre"
        activate_nopre_env()
        @testset "Explicit Imports" begin
            include("nopre/explicit_imports.jl")
        end
        @testset "JET Tests" begin
            include("nopre/jet_tests.jl")
        end
        @testset "AllocCheck - Zero Allocations" begin
            include("nopre/alloc_tests.jl")
        end
    end
end

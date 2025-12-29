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

        @testset "Analytical Solutions" begin
            @testset "GBM analytical functions" begin
                μ = 0.05
                σ = 0.2
                u0 = 100.0
                t = 1.0

                # Test at t=0 (should return initial value for mean, 0 for variance)
                @test gbm_mean(μ, u0, 0.0) ≈ u0
                @test gbm_variance(μ, σ, u0, 0.0) ≈ 0.0 atol = 1.0e-15

                # Test analytical formulas
                @test gbm_mean(μ, u0, t) ≈ u0 * exp(μ * t)
                @test gbm_variance(μ, σ, u0, t) ≈ u0^2 * exp(2μ * t) * (exp(σ^2 * t) - 1)
                @test gbm_std(μ, σ, u0, t) ≈ sqrt(gbm_variance(μ, σ, u0, t))

                # Test broadcasting with array of times
                times = [0.0, 0.5, 1.0, 2.0]
                means = gbm_mean.(μ, u0, times)
                @test length(means) == 4
                @test means[1] ≈ u0

                # Monte Carlo validation
                prob = GeometricBrownianMotionProblem(μ, σ, u0, (0.0, t))
                monte_prob = EnsembleProblem(prob)
                sol = solve(monte_prob, EM(); dt = 0.01, trajectories = 10000)
                final_values = [sol[i].u[end] for i in eachindex(sol)]
                simulated_mean = mean(final_values)
                simulated_var = var(final_values)

                analytical_mean = gbm_mean(μ, u0, t)
                analytical_var = gbm_variance(μ, σ, u0, t)

                # Allow 5% relative error for Monte Carlo
                @test abs(simulated_mean - analytical_mean) / analytical_mean < 0.05
                @test abs(simulated_var - analytical_var) / analytical_var < 0.15
            end

            @testset "OU analytical functions" begin
                a = 0.5
                r = 1.0
                σ = 0.3
                u0 = 0.5
                t = 2.0

                # Test at t=0
                @test ou_mean(a, r, u0, 0.0) ≈ u0
                @test ou_variance(a, σ, 0.0) ≈ 0.0 atol = 1.0e-15

                # Test analytical formulas
                @test ou_mean(a, r, u0, t) ≈ r + (u0 - r) * exp(-a * t)
                @test ou_variance(a, σ, t) ≈ (σ^2 / (2a)) * (1 - exp(-2a * t))
                @test ou_std(a, σ, t) ≈ sqrt(ou_variance(a, σ, t))

                # Test stationary values
                @test ou_stationary_mean(r) == r
                @test ou_stationary_variance(a, σ) ≈ σ^2 / (2a)

                # Test convergence to stationary distribution
                large_t = 100.0
                @test ou_mean(a, r, u0, large_t) ≈ ou_stationary_mean(r) atol = 1.0e-10
                @test ou_variance(a, σ, large_t) ≈ ou_stationary_variance(a, σ) atol = 1.0e-10

                # Monte Carlo validation
                prob = OrnsteinUhlenbeckProblem(a, r, σ, u0, (0.0, t))
                monte_prob = EnsembleProblem(prob)
                sol = solve(monte_prob, EM(); dt = 0.01, trajectories = 10000)
                final_values = [sol[i].u[end] for i in eachindex(sol)]
                simulated_mean = mean(final_values)
                simulated_var = var(final_values)

                analytical_mean = ou_mean(a, r, u0, t)
                analytical_var = ou_variance(a, σ, t)

                @test abs(simulated_mean - analytical_mean) < 0.05
                @test abs(simulated_var - analytical_var) / analytical_var < 0.15
            end

            @testset "CIR analytical functions" begin
                κ = 0.5
                θ = 0.04
                σ = 0.1
                u0 = 0.03
                t = 2.0

                # Verify Feller condition for test parameters
                @test 2κ * θ >= σ^2  # Should satisfy Feller condition

                # Test at t=0
                @test cir_mean(κ, θ, u0, 0.0) ≈ u0
                @test cir_variance(κ, θ, σ, u0, 0.0) ≈ 0.0 atol = 1.0e-15

                # Test analytical formulas
                exp_κt = exp(-κ * t)
                exp_2κt = exp(-2κ * t)
                expected_mean = θ + (u0 - θ) * exp_κt
                expected_var = u0 * (σ^2 / κ) * (exp_κt - exp_2κt) +
                    (θ * σ^2 / (2κ)) * (1 - exp_κt)^2

                @test cir_mean(κ, θ, u0, t) ≈ expected_mean
                @test cir_variance(κ, θ, σ, u0, t) ≈ expected_var
                @test cir_std(κ, θ, σ, u0, t) ≈ sqrt(expected_var)

                # Test stationary values
                @test cir_stationary_mean(θ) == θ
                @test cir_stationary_variance(κ, θ, σ) ≈ θ * σ^2 / (2κ)

                # Test convergence to stationary distribution
                large_t = 100.0
                @test cir_mean(κ, θ, u0, large_t) ≈ cir_stationary_mean(θ) atol = 1.0e-10

                # Monte Carlo validation
                prob = CIRProblem(κ, θ, σ, u0, (0.0, t))
                monte_prob = EnsembleProblem(prob)
                sol = solve(monte_prob, EM(); dt = 0.001, trajectories = 10000)
                final_values = [sol[i].u[end] for i in eachindex(sol)]
                simulated_mean = mean(final_values)
                simulated_var = var(final_values)

                analytical_mean = cir_mean(κ, θ, u0, t)
                analytical_var = cir_variance(κ, θ, σ, u0, t)

                @test abs(simulated_mean - analytical_mean) / analytical_mean < 0.05
                @test abs(simulated_var - analytical_var) / analytical_var < 0.2
            end

            @testset "Black-Scholes log-price functions" begin
                r = 0.05
                σ = 0.2
                u0 = log(100.0)  # log of initial price
                t = 1.0

                # Test at t=0
                @test bs_log_mean(r, σ, u0, 0.0) ≈ u0
                @test bs_log_variance(σ, 0.0) ≈ 0.0 atol = 1.0e-15

                # Test analytical formulas
                @test bs_log_mean(r, σ, u0, t) ≈ u0 + (r - σ^2 / 2) * t
                @test bs_log_variance(σ, t) ≈ σ^2 * t
                @test bs_log_std(σ, t) ≈ σ * sqrt(t)

                # Test broadcasting
                times = [0.25, 0.5, 1.0]
                variances = bs_log_variance.(σ, times)
                @test variances ≈ σ^2 .* times
            end

            @testset "Heston analytical functions" begin
                μ = 0.05
                κ = 2.0
                Θ = 0.04
                σ = 0.3
                u0_S = 100.0
                u0_v = 0.04
                t = 1.0

                # Test Heston mean (should match GBM mean since volatility has zero mean increment)
                @test heston_mean(μ, u0_S, t) ≈ u0_S * exp(μ * t)
                @test heston_mean(μ, u0_S, 0.0) ≈ u0_S

                # Test variance process mean (should match CIR mean)
                @test heston_variance_mean(κ, Θ, u0_v, t) ≈ cir_mean(κ, Θ, u0_v, t)
                @test heston_variance_mean(κ, Θ, u0_v, 0.0) ≈ u0_v

                # Test variance of variance process (should match CIR variance)
                @test heston_variance_variance(κ, Θ, σ, u0_v, t) ≈
                    cir_variance(κ, Θ, σ, u0_v, t)
            end
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

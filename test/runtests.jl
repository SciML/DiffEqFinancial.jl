using DiffEqFinancial, Statistics, StochasticDiffEq
using Test
using JET

@testset "Explicit Imports" begin
    include("explicit_imports.jl")
end

@testset "DiffEqFinancial.jl" begin
u0 = [1.0; 0.5]
σ = 0.25
prob = HestonProblem(1.0, 1.0, σ, 1.0, 1.0, u0, (0.0, 1.0))
sol = solve(prob, SRIW1(), adaptive = false, dt = 1 / 100)

prob = BlackScholesProblem((t) -> t^2, (t, u) -> 1.0, σ, 0.5, (0.0, 1.0))
sol = solve(prob, SRIW1())

r = 0.03
sigma = 0.2
S0 = 100
t=0
T=1.0
days = 252
dt = 1/days

prob = GeometricBrownianMotionProblem(r, sigma, S0, (t, T))
sol = solve(prob, EM(); dt = dt)
monte_prob = EnsembleProblem(prob)
sol = solve(monte_prob, EM(); dt = dt, trajectories = 1000000)
us=[sol[i].u for i in eachindex(sol)]
simulated = mean(us)

tsteps = collect(0:dt:T)
expected = S0 * exp.(r * tsteps)
testerr = sum(abs2.(simulated .- expected))
@test testerr < 2e-1

end # testset DiffEqFinancial.jl

@testset "JET static analysis" begin
    # Test that key problem constructors are type-stable using @test_opt
    # target_modules filters to only report issues from DiffEqFinancial itself,
    # ignoring upstream SciMLBase/DiffEqBase issues

    @testset "HestonProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) HestonProblem(
            1.0, 1.0, 0.25, 1.0, 1.0, [1.0, 0.5], (0.0, 1.0))
    end

    @testset "GeometricBrownianMotionProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) GeometricBrownianMotionProblem(
            0.03, 0.2, 100.0, (0.0, 1.0))
    end

    @testset "OrnsteinUhlenbeckProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) OrnsteinUhlenbeckProblem(
            0.1, 0.5, 0.2, 1.0, (0.0, 1.0))
    end

    @testset "CIRProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) CIRProblem(
            0.1, 0.5, 0.2, 0.4, (0.0, 1.0))
    end

    @testset "CIRNoise type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) CIRNoise(
            0.1, 0.5, 0.2, 0.0, 0.4)
    end
end

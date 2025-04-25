using DiffEqFinancial, Statistics, StochasticDiffEq, Distributions
using Test

# write your own tests here
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
us = [sol.u[i].u for i in eachindex(sol)]
simulated = mean(us)

tsteps = collect(0:dt:T)
expected = S0 * exp.(r * tsteps)
testerr = sum(abs2.(simulated .- expected))
@test testerr < 2e-1

κ, θ, σ, u0, tspan =  0.30, 0.04, 0.15, 0.2, (0.0, 1.0)
prob = CIRProblem(κ, θ, σ, u0, tspan)
sol = solve(prob, EM(); dt = dt)
monte_prob = EnsembleProblem(prob)
sol = solve(monte_prob, EM(); dt = dt, trajectories = 1000000)
us = [sol.u[i].u for i in eachindex(sol)]
simulated = mean(us)

d = 4 * κ * θ / σ^2  # Degrees of freedom
λ(t) = 4 * κ * exp(-κ * t) * u0 / (σ^2 * (-expm1(-κ * t)))  # Noncentrality parameter
c(t) = σ^2 * (-expm1(-κ * t)) / (4 * κ)  # Scaling factor
dist(t) = c(t) * Distributions.mean(NoncentralChisq(d, λ(t)))

tsteps = collect(dt:dt:T) 
expected = dist.(tsteps)
testerr = sum(abs2.(simulated[2:end] .- expected))
@test testerr < 2e-1
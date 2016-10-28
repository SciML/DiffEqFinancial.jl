using FinancialModels, StochasticDiffEq
using Base.Test

# write your own tests here
u₀ = [1.;0.5]
σ =  0.25
prob = HestonProblem(1.,1.,σ,1.,1.,u₀)
sol = solve(prob)

prob = BlackScholesProblem((t)->t^2,(t,u)->1.,σ,0.5)
sol = solve(prob)

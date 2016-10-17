using FinancialModels
using Base.Test

# write your own tests here
u₀ = [1.;0.5]
σ =  0.25
prob = HestonProblem(1.,1.,σ,1.,1.,u₀)
sol = solve(prob)

prob = BlackScholesProblem((t)->t^2,1.,σ,u₀)
sol = solve(prob)

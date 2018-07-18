using DiffEqFinancial, StochasticDiffEq
using Test

# write your own tests here
u0 = [1.;0.5]
σ =  0.25
prob = HestonProblem(1.,1.,σ,1.,1.,u0,(0.0,1.0))
sol = solve(prob,SRIW1(),adaptive=false,dt=1/10)

prob = BlackScholesProblem((t)->t^2,(t,u)->1.,σ,0.5,(0.0,1.0))
sol = solve(prob,SRIW1())

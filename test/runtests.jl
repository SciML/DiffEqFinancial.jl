using FinancialModels
using Base.Test

# write your own tests here
prob = HestonProblem(1.,1.,1.,1.,1.,[1.;1.])
sol = solve(prob)

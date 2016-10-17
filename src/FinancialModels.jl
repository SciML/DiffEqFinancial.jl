module FinancialModels
  using Reexport
  @reexport using DifferentialEquations
  import DifferentialEquations: solve

  include("problems.jl")

  export HestonProblem, BlackScholesProblem

end # module

__precompile__()

module DiffEqFinancial
  using DiffEqBase, DiffEqNoiseProcess
  import DiffEqBase: solve

  import RandomNumbers: Xorshifts


  include("problems.jl")

  export HestonProblem, BlackScholesProblem, GeneralizedBlackScholesProblem,
  ExtendedOrnsteinUhlenbeckProblem, OrnsteinUhlenbeckProblem, GeometricBrownianMotionProblem,
  MfStateProblem

end # module

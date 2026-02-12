module DiffEqFinancial

using DiffEqBase: AbstractSDEProblem, CallbackSet, SDEFunction, SDEProblem
using DiffEqNoiseProcess: CorrelatedWienerProcess!, NoiseProcess
using Distributions: NoncentralChisq
using Markdown: @doc_str

import RandomNumbers: Xorshifts

include("problems.jl")
include("analytics.jl")

export HestonProblem, BlackScholesProblem, GeneralizedBlackScholesProblem,
    ExtendedOrnsteinUhlenbeckProblem, OrnsteinUhlenbeckProblem,
    GeometricBrownianMotionProblem,
    MfStateProblem,
    CIRProblem,
    CIRNoise

# Analytical solutions and moment functions
export gbm_mean, gbm_variance, gbm_std,
    ou_mean, ou_variance, ou_std, ou_stationary_mean, ou_stationary_variance,
    cir_mean, cir_variance, cir_std, cir_stationary_mean, cir_stationary_variance,
    bs_log_mean, bs_log_variance, bs_log_std,
    heston_mean, heston_variance_mean, heston_variance_variance

end # module

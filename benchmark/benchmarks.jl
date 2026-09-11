using DiffEqFinancial, StochasticDiffEq, BenchmarkTools

const SUITE = BenchmarkGroup()

# =============================================================================
# Financial problem solves
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

# Heston stochastic volatility model
u0 = [1.0; 0.5]
heston = HestonProblem(1.0, 1.0, 0.25, 1.0, 1.0, u0, (0.0, 1.0))
SUITE["solve"]["heston"] = @benchmarkable solve(
    $heston, SRIW1(); adaptive = false, dt = 1 / 100
)

# Geometric Brownian motion (Black-Scholes)
gbm_prob = GeneralizedBlackScholesProblem(
    t -> 0.01, t -> 0.0, (t, S) -> 0.25, 0.25, 100.0, (0.0, 1.0)
)
SUITE["solve"]["black_scholes"] = @benchmarkable solve(
    $gbm_prob, EM(); adaptive = false, dt = 0.01
)

# Monte Carlo bundle of GBM paths
SUITE["solve"]["gbm_ensemble"] = @benchmarkable solve(
    $(EnsembleProblem(gbm_prob)), EM(); adaptive = false, dt = 0.01,
    trajectories = 1000
)

# =============================================================================
# Analytic helpers
# =============================================================================

SUITE["helpers"] = BenchmarkGroup()

SUITE["helpers"]["gbm_mean"] = @benchmarkable gbm_mean(0.05, 100.0, 1.0)
SUITE["helpers"]["gbm_variance"] = @benchmarkable gbm_variance(
    0.05, 0.25, 100.0, 1.0
)

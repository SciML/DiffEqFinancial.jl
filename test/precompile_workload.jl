using DiffEqFinancial
using Test

@testset "Precompile workload API" begin
    problem = GeometricBrownianMotionProblem(0.03, 0.2, 100.0, (0.0, 1.0))
    @test problem.u0 == 100.0
    @test gbm_mean(0.03, 100.0, 0.0) == 100.0
    @test cir_stationary_mean(0.04) == 0.04
end

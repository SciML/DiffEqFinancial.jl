using DiffEqFinancial
using JET
using Test

@testset "JET static analysis" begin
    # Test that key problem constructors are type-stable using @test_opt
    # target_modules filters to only report issues from DiffEqFinancial itself,
    # ignoring upstream SciMLBase/DiffEqBase issues

    @testset "HestonProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) HestonProblem(
            1.0, 1.0, 0.25, 1.0, 1.0, [1.0, 0.5], (0.0, 1.0)
        )
    end

    @testset "BlackScholesProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) BlackScholesProblem(
            t -> t^2, (t, u) -> 1.0, 0.25, 0.5, (0.0, 1.0)
        )
    end

    @testset "GeneralizedBlackScholesProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) GeneralizedBlackScholesProblem(
            t -> 0.05, t -> 0.01, (t, u) -> 1.0, 0.2, 0.5, (0.0, 1.0)
        )
    end

    @testset "GeometricBrownianMotionProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) GeometricBrownianMotionProblem(
            0.03, 0.2, 100.0, (0.0, 1.0)
        )
    end

    @testset "OrnsteinUhlenbeckProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) OrnsteinUhlenbeckProblem(
            0.1, 0.5, 0.2, 1.0, (0.0, 1.0)
        )
    end

    @testset "ExtendedOrnsteinUhlenbeckProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) ExtendedOrnsteinUhlenbeckProblem(
            0.1, t -> 0.5 + 0.1 * t, 0.2, 1.0, (0.0, 1.0)
        )
    end

    @testset "MfStateProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) MfStateProblem(
            0.1, t -> 0.2 * (1.0 + 0.1 * t), 0.0, (0.0, 1.0)
        )
    end

    @testset "CIRProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) CIRProblem(
            0.1, 0.5, 0.2, 0.4, (0.0, 1.0)
        )
    end

    @testset "CIRNoise type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) CIRNoise(
            0.1, 0.5, 0.2, 0.0, 0.4
        )
    end

    @testset "Analytical functions type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) gbm_mean(0.05, 100.0, 1.0)
        @test_opt target_modules = (DiffEqFinancial,) gbm_variance(0.05, 0.2, 100.0, 1.0)
        @test_opt target_modules = (DiffEqFinancial,) gbm_std(0.05, 0.2, 100.0, 1.0)
        @test_opt target_modules = (DiffEqFinancial,) ou_mean(0.5, 1.0, 0.5, 2.0)
        @test_opt target_modules = (DiffEqFinancial,) ou_variance(0.5, 0.3, 2.0)
        @test_opt target_modules = (DiffEqFinancial,) ou_std(0.5, 0.3, 2.0)
        @test_opt target_modules = (DiffEqFinancial,) ou_stationary_mean(1.0)
        @test_opt target_modules = (DiffEqFinancial,) ou_stationary_variance(0.5, 0.3)
        @test_opt target_modules = (DiffEqFinancial,) cir_mean(0.5, 0.04, 0.03, 2.0)
        @test_opt target_modules = (DiffEqFinancial,) cir_variance(0.5, 0.04, 0.1, 0.03, 2.0)
        @test_opt target_modules = (DiffEqFinancial,) cir_std(0.5, 0.04, 0.1, 0.03, 2.0)
        @test_opt target_modules = (DiffEqFinancial,) cir_stationary_mean(0.04)
        @test_opt target_modules = (DiffEqFinancial,) cir_stationary_variance(0.5, 0.04, 0.1)
        @test_opt target_modules = (DiffEqFinancial,) bs_log_mean(0.05, 0.2, 4.6, 1.0)
        @test_opt target_modules = (DiffEqFinancial,) bs_log_variance(0.2, 1.0)
        @test_opt target_modules = (DiffEqFinancial,) bs_log_std(0.2, 1.0)
        @test_opt target_modules = (DiffEqFinancial,) heston_mean(0.05, 100.0, 1.0)
        @test_opt target_modules = (DiffEqFinancial,) heston_variance_mean(2.0, 0.04, 0.04, 1.0)
        @test_opt target_modules = (DiffEqFinancial,) heston_variance_variance(
            2.0, 0.04, 0.3, 0.04, 1.0
        )
    end
end

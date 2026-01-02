using DiffEqFinancial
using JET
using Test

@testset "JET static analysis" begin
    # Test that key problem constructors are type-stable using @test_opt
    # target_modules filters to only report issues from DiffEqFinancial itself,
    # ignoring upstream SciMLBase/DiffEqBase issues

    @testset "HestonProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) HestonProblem(
            1.0, 1.0, 0.25, 1.0, 1.0, [1.0, 0.5], (0.0, 1.0))
    end

    @testset "BlackScholesProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) BlackScholesProblem(
            t -> t^2, (t, u) -> 1.0, 0.25, 0.5, (0.0, 1.0))
    end

    @testset "GeneralizedBlackScholesProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) GeneralizedBlackScholesProblem(
            t -> 0.05, t -> 0.01, (t, u) -> 1.0, 0.2, 0.5, (0.0, 1.0))
    end

    @testset "GeometricBrownianMotionProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) GeometricBrownianMotionProblem(
            0.03, 0.2, 100.0, (0.0, 1.0))
    end

    @testset "OrnsteinUhlenbeckProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) OrnsteinUhlenbeckProblem(
            0.1, 0.5, 0.2, 1.0, (0.0, 1.0))
    end

    @testset "ExtendedOrnsteinUhlenbeckProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) ExtendedOrnsteinUhlenbeckProblem(
            0.1, t -> 0.5 + 0.1 * t, 0.2, 1.0, (0.0, 1.0))
    end

    @testset "MfStateProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) MfStateProblem(
            0.1, t -> 0.2 * (1.0 + 0.1 * t), 0.0, (0.0, 1.0))
    end

    @testset "CIRProblem type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) CIRProblem(
            0.1, 0.5, 0.2, 0.4, (0.0, 1.0))
    end

    @testset "CIRNoise type stability" begin
        @test_opt target_modules = (DiffEqFinancial,) CIRNoise(
            0.1, 0.5, 0.2, 0.0, 0.4)
    end
end

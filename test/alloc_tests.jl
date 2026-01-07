using DiffEqFinancial
using AllocCheck
using Test

@testset "AllocCheck - Drift/Diffusion Functions" begin
    # These tests verify that the drift and diffusion functions used in solving
    # do not allocate memory, which is critical for performance.

    @testset "HestonProblem" begin
        u0 = [1.0, 0.5]
        prob = HestonProblem(1.0, 1.0, 0.25, 1.0, 1.0, u0, (0.0, 1.0))
        du = zeros(2)
        u = [1.0, 0.5]

        # Test drift function
        @check_allocs test_heston_drift(f, du, u) = (f(du, u, nothing, 0.0); nothing)
        @test (test_heston_drift(prob.f.f, du, u); true)

        # Test diffusion function
        @check_allocs test_heston_diff(g, du, u) = (g(du, u, nothing, 0.0); nothing)
        @test (test_heston_diff(prob.f.g, du, u); true)
    end

    @testset "GeometricBrownianMotionProblem" begin
        prob = GeometricBrownianMotionProblem(0.03, 0.2, 100.0, (0.0, 1.0))

        @check_allocs test_gbm_drift(f, u) = f(u, nothing, 0.0)
        @test (test_gbm_drift(prob.f.f, 100.0); true)

        @check_allocs test_gbm_diff(g, u) = g(u, nothing, 0.0)
        @test (test_gbm_diff(prob.f.g, 100.0); true)
    end

    @testset "CIRProblem" begin
        prob = CIRProblem(0.1, 0.5, 0.2, 0.4, (0.0, 1.0))

        @check_allocs test_cir_drift(f, u) = f(u, nothing, 0.0)
        @test (test_cir_drift(prob.f.f, 0.4); true)

        @check_allocs test_cir_diff(g, u) = g(u, nothing, 0.0)
        @test (test_cir_diff(prob.f.g, 0.4); true)
    end

    @testset "OrnsteinUhlenbeckProblem" begin
        prob = OrnsteinUhlenbeckProblem(0.1, 0.5, 0.2, 1.0, (0.0, 1.0))

        @check_allocs test_ou_drift(f, u) = f(u, nothing, 0.0)
        @test (test_ou_drift(prob.f.f, 1.0); true)

        @check_allocs test_ou_diff(g, u) = g(u, nothing, 0.0)
        @test (test_ou_diff(prob.f.g, 1.0); true)
    end

    @testset "ExtendedOrnsteinUhlenbeckProblem" begin
        b_func = t -> 0.5 + 0.1 * t
        prob = ExtendedOrnsteinUhlenbeckProblem(0.1, b_func, 0.2, 1.0, (0.0, 1.0))

        @check_allocs test_eou_drift(f, u) = f(u, nothing, 0.5)
        @test (test_eou_drift(prob.f.f, 1.0); true)

        @check_allocs test_eou_diff(g, u) = g(u, nothing, 0.5)
        @test (test_eou_diff(prob.f.g, 1.0); true)
    end

    @testset "MfStateProblem" begin
        sigma_func = t -> 0.2 * (1.0 + 0.1 * t)
        prob = MfStateProblem(0.1, sigma_func, 0.0, (0.0, 1.0))

        @check_allocs test_mf_drift(f, u) = f(u, nothing, 0.5)
        @test (test_mf_drift(prob.f.f, 0.0); true)

        @check_allocs test_mf_diff(g, u) = g(u, nothing, 0.5)
        @test (test_mf_diff(prob.f.g, 0.0); true)
    end

    @testset "GeneralizedBlackScholesProblem" begin
        # Use constant callables for allocation testing
        r_func = t -> 0.05
        q_func = t -> 0.01
        theta_func = (t, u) -> 0.2
        prob = GeneralizedBlackScholesProblem(r_func, q_func, theta_func, 0.2, 0.5, (0.0, 1.0))

        @check_allocs test_gbs_drift(f, u) = f(u, nothing, 0.5)
        @test (test_gbs_drift(prob.f.f, 0.5); true)

        @check_allocs test_gbs_diff(g, u) = g(u, nothing, 0.5)
        @test (test_gbs_diff(prob.f.g, 0.5); true)
    end

    @testset "BlackScholesProblem" begin
        # BlackScholesProblem wraps GeneralizedBlackScholesProblem with q(t) = 0
        r_func = t -> 0.05
        theta_func = (t, u) -> 0.2
        prob = BlackScholesProblem(r_func, theta_func, 0.2, 0.5, (0.0, 1.0))

        @check_allocs test_bs_drift(f, u) = f(u, nothing, 0.5)
        @test (test_bs_drift(prob.f.f, 0.5); true)

        @check_allocs test_bs_diff(g, u) = g(u, nothing, 0.5)
        @test (test_bs_diff(prob.f.g, 0.5); true)
    end
end

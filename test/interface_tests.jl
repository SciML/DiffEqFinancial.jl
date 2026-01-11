using Test
using DiffEqFinancial
using StochasticDiffEq

@testset "Interface Compatibility" begin
    @testset "BigFloat support" begin
        @testset "GeometricBrownianMotionProblem" begin
            μ = BigFloat(0.03)
            σ = BigFloat(0.2)
            u0 = BigFloat(100.0)
            tspan = (BigFloat(0.0), BigFloat(1.0))
            prob = GeometricBrownianMotionProblem(μ, σ, u0, tspan)
            @test prob.u0 isa BigFloat
            sol = solve(prob, EM(), dt = BigFloat(0.01))
            @test sol.u[end] isa BigFloat
        end

        @testset "OrnsteinUhlenbeckProblem" begin
            a = BigFloat(0.5)
            r = BigFloat(1.0)
            σ = BigFloat(0.1)
            u0 = BigFloat(0.0)
            tspan = (BigFloat(0.0), BigFloat(1.0))
            prob = OrnsteinUhlenbeckProblem(a, r, σ, u0, tspan)
            @test prob.u0 isa BigFloat
            sol = solve(prob, EM(), dt = BigFloat(0.01))
            @test sol.u[end] isa BigFloat
        end

        @testset "ExtendedOrnsteinUhlenbeckProblem" begin
            a = BigFloat(0.5)
            b = t -> BigFloat(1.0) + BigFloat(0.1) * t
            σ = BigFloat(0.1)
            u0 = BigFloat(0.0)
            tspan = (BigFloat(0.0), BigFloat(1.0))
            prob = ExtendedOrnsteinUhlenbeckProblem(a, b, σ, u0, tspan)
            @test prob.u0 isa BigFloat
            sol = solve(prob, EM(), dt = BigFloat(0.01))
            @test sol.u[end] isa BigFloat
        end

        @testset "CIRProblem" begin
            κ = BigFloat(0.5)
            θ = BigFloat(0.1)
            σ = BigFloat(0.1)
            u0 = BigFloat(0.05)
            tspan = (BigFloat(0.0), BigFloat(1.0))
            prob = CIRProblem(κ, θ, σ, u0, tspan)
            @test prob.u0 isa BigFloat
            sol = solve(prob, EM(), dt = BigFloat(0.01))
            @test sol.u[end] isa BigFloat
        end

        @testset "MfStateProblem" begin
            a = BigFloat(0.5)
            σ = t -> BigFloat(0.1) * t
            u0 = BigFloat(0.0)
            tspan = (BigFloat(0.0), BigFloat(1.0))
            prob = MfStateProblem(a, σ, u0, tspan)
            @test prob.u0 isa BigFloat
            # Test that drift function returns correct type (was returning Int before fix)
            @test prob.f(u0, nothing, BigFloat(0.5)) isa BigFloat
            sol = solve(prob, EM(), dt = BigFloat(0.01))
            @test sol.u[end] isa BigFloat
        end

        @testset "BlackScholesProblem" begin
            r = t -> BigFloat(0.05)
            Θ = (t, u) -> BigFloat(1.0)
            σ = BigFloat(0.2)
            u0 = BigFloat(0.5)
            tspan = (BigFloat(0.0), BigFloat(1.0))
            prob = BlackScholesProblem(r, Θ, σ, u0, tspan)
            @test prob.u0 isa BigFloat
            sol = solve(prob, EM(), dt = BigFloat(0.01))
            @test sol.u[end] isa BigFloat
        end
    end

    @testset "Float32 support" begin
        @testset "HestonProblem" begin
            u0 = Float32[1.0, 0.5]
            μ = Float32(1.0)
            κ = Float32(1.0)
            Θ = Float32(0.25)
            σ = Float32(1.0)
            ρ = Float32(0.5)
            tspan = (Float32(0.0), Float32(1.0))
            prob = HestonProblem(μ, κ, Θ, σ, ρ, u0, tspan)
            @test eltype(prob.u0) == Float32
            sol = solve(prob, EM(), dt = Float32(0.01))
            @test eltype(sol.u[end]) == Float32
        end

        @testset "GeometricBrownianMotionProblem" begin
            μ = Float32(0.03)
            σ = Float32(0.2)
            u0 = Float32(100.0)
            tspan = (Float32(0.0), Float32(1.0))
            prob = GeometricBrownianMotionProblem(μ, σ, u0, tspan)
            @test prob.u0 isa Float32
            sol = solve(prob, EM(), dt = Float32(0.01))
            @test sol.u[end] isa Float32
        end
    end
end

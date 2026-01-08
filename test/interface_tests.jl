using DiffEqFinancial
using StochasticDiffEq
using Test

@testset "Interface Compatibility" begin
    @testset "Type genericity - Float32" begin
        # HestonProblem with Float32
        u0 = Float32[1.0, 0.5]
        prob = HestonProblem(
            Float32(1.0), Float32(1.0), Float32(0.25),
            Float32(1.0), Float32(1.0), u0,
            (Float32(0.0), Float32(1.0))
        )
        sol = solve(prob, EM(), adaptive = false, dt = Float32(0.1))
        @test eltype(sol.u[end]) == Float32

        # GeometricBrownianMotionProblem with Float32
        prob2 = GeometricBrownianMotionProblem(
            Float32(0.03), Float32(0.2), Float32(100.0),
            (Float32(0.0), Float32(1.0))
        )
        sol2 = solve(prob2, EM(), dt = Float32(0.01))
        @test eltype(sol2.u) == Float32

        # OrnsteinUhlenbeckProblem with Float32
        prob3 = OrnsteinUhlenbeckProblem(
            Float32(1.0), Float32(0.5), Float32(0.1),
            Float32(0.0), (Float32(0.0), Float32(1.0))
        )
        sol3 = solve(prob3, EM(), dt = Float32(0.01))
        @test eltype(sol3.u) == Float32

        # CIRProblem with Float32
        prob4 = CIRProblem(
            Float32(0.5), Float32(0.04), Float32(0.05),
            Float32(0.02), (Float32(0.0), Float32(1.0))
        )
        sol4 = solve(prob4, EM(), dt = Float32(0.01))
        @test eltype(sol4.u) == Float32

        # MfStateProblem with Float32
        prob5 = MfStateProblem(
            Float32(0.1), t -> Float32(0.2),
            Float32(0.0), (Float32(0.0), Float32(1.0))
        )
        sol5 = solve(prob5, EM(), dt = Float32(0.01))
        @test eltype(sol5.u) == Float32
    end

    @testset "Type genericity - BigFloat" begin
        # GeometricBrownianMotionProblem with BigFloat
        prob = GeometricBrownianMotionProblem(
            BigFloat(0.03), BigFloat(0.2), BigFloat(100.0),
            (BigFloat(0.0), BigFloat(1.0))
        )
        sol = solve(prob, EM(), dt = BigFloat(0.01))
        @test eltype(sol.u) == BigFloat

        # OrnsteinUhlenbeckProblem with BigFloat
        prob2 = OrnsteinUhlenbeckProblem(
            BigFloat(1.0), BigFloat(0.5), BigFloat(0.1),
            BigFloat(0.0), (BigFloat(0.0), BigFloat(1.0))
        )
        sol2 = solve(prob2, EM(), dt = BigFloat(0.01))
        @test eltype(sol2.u) == BigFloat

        # CIRProblem with BigFloat
        prob3 = CIRProblem(
            BigFloat(0.5), BigFloat(0.04), BigFloat(0.05),
            BigFloat(0.02), (BigFloat(0.0), BigFloat(1.0))
        )
        sol3 = solve(prob3, EM(), dt = BigFloat(0.01))
        @test eltype(sol3.u) == BigFloat

        # MfStateProblem with BigFloat
        prob4 = MfStateProblem(
            BigFloat(0.1), t -> BigFloat(0.2),
            BigFloat(0.0), (BigFloat(0.0), BigFloat(1.0))
        )
        sol4 = solve(prob4, EM(), dt = BigFloat(0.01))
        @test eltype(sol4.u) == BigFloat
    end
end

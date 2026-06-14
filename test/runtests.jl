using Pkg
using SafeTestsets
using Test
using DiffEqFinancial, Statistics, StochasticDiffEq

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    @testset "DiffEqFinancial.jl" begin
        @safetestset "Core Tests" begin
            include("core_tests.jl")
        end

        # Interface tests for type genericity
        @safetestset "Interface Compatibility" begin
            include("interface_tests.jl")
        end

        @safetestset "Analytical Solutions" begin
            include("analytical_tests.jl")
        end
    end
end

if GROUP == "All" || GROUP == "QA"
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @testset "Quality Assurance" begin
        include(joinpath(@__DIR__, "qa", "qa.jl"))
    end
end

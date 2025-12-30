using ExplicitImports
using DiffEqFinancial
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(DiffEqFinancial) === nothing
    @test check_no_stale_explicit_imports(DiffEqFinancial) === nothing
end

using SafeTestsets

@safetestset "Explicit Imports" begin
    include("explicit_imports.jl")
end
@safetestset "JET Tests" begin
    include("jet_tests.jl")
end
@safetestset "AllocCheck - Zero Allocations" begin
    include("alloc_tests.jl")
end

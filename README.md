# DiffEqFinancial.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://github.com/SciML/DiffEqFinancial.jl/workflows/CI/badge.svg)](https://github.com/SciML/DiffEqFinancial.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/JuliaDiffEq/FinancialModels.jl/badge.svg)](https://coveralls.io/github/JuliaDiffEq/DiffEqFinancial.jl)
[![codecov](https://codecov.io/gh/JuliaDiffEq/FinancialModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaDiffEq/DiffEqFinancial.jl)

This repository holds problem definitions for common financial models for use in
the DifferentialEquations ecosystem. The goal is to be a feature-complete set of
solvers for the types of problems found in libraries like QuantLib. For example,
one can easily specify, solve, and plot the solution for a trajectory of a Heston
process via the commands:

```julia
prob = HestonProblem(μ,κ,Θ,σ,ρ,u₀)
sol = solve(prob)
plot(sol)
```

Full documentation is in the
[DifferentialEquations.jl models documentation](https://docs.sciml.ai/DiffEqDocs/stable/)

These solvers use DifferentialEquations.jl, meaning that high-performance and
high order algorithms are available.

The project is looking for contributors who would like to implement more models.
If you're interested and need help getting started, talk to Chris Rackauckas in
the [Gitter chat](https://gitter.im/JuliaDiffEq/Lobby).

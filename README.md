# DiffEqFinancial.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/JuliaDiffEq/DiffEqFinancial.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/DiffEqFinancial.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/a5ynd3ypb3tejqa2?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/diffeqfinancial-jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaDiffEq/FinancialModels.jl/badge.svg)](https://coveralls.io/github/JuliaDiffEq/DiffEqFinancial.jl)
[![codecov](https://codecov.io/gh/JuliaDiffEq/FinancialModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaDiffEq/DiffEqFinancial.jl)
[![DiffEqFinancial](http://pkg.julialang.org/badges/DiffEqFinancial_0.5.svg)](http://pkg.julialang.org/?pkg=DiffEqFinancial)
[![DiffEqFinancial](http://pkg.julialang.org/badges/DiffEqFinancial_0.6.svg)](http://pkg.julialang.org/?pkg=DiffEqFinancial)

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
[DifferentialEquations.jl models documentation](http://docs.juliadiffeq.org/latest/models/financial.html)

These solvers use DifferentialEquations.jl, meaning that high-performance and
high order algorithms are available.

The project is looking for contributors who would like to implement more models.
If you're interested and need help getting started, talk to Chris Rackauckas in
the [Gitter chat](https://gitter.im/JuliaDiffEq/Lobby).

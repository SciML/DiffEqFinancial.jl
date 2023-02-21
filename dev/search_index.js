var documenterSearchIndex = {"docs":
[{"location":"diffeqfinancial/#API","page":"API","title":"API","text":"","category":"section"},{"location":"diffeqfinancial/","page":"API","title":"API","text":"Modules = [DiffEqFinancial]","category":"page"},{"location":"diffeqfinancial/#DiffEqFinancial.BlackScholesProblem","page":"API","title":"DiffEqFinancial.BlackScholesProblem","text":"d ln S(t) = (r(t) - fracΘ(tS)^22)dt + σ dW_t\n\nSolves for log S(t).\n\n\n\n","category":"type"},{"location":"diffeqfinancial/#DiffEqFinancial.ExtendedOrnsteinUhlenbeckProblem-NTuple{5, Any}","page":"API","title":"DiffEqFinancial.ExtendedOrnsteinUhlenbeckProblem","text":"dx = a(b(t)-x)dt + σ dW_t\n\n\n\n","category":"method"},{"location":"diffeqfinancial/#DiffEqFinancial.GeneralizedBlackScholesProblem-NTuple{6, Any}","page":"API","title":"DiffEqFinancial.GeneralizedBlackScholesProblem","text":"d ln S(t) = (r(t) - q(t) - fracΘ(tS)^22)dt + σ dW_t\n\nSolves for log S(t).\n\n\n\n","category":"method"},{"location":"diffeqfinancial/#DiffEqFinancial.GeometricBrownianMotionProblem-NTuple{4, Any}","page":"API","title":"DiffEqFinancial.GeometricBrownianMotionProblem","text":"dx = μ dt + σ dW_t\n\n\n\n","category":"method"},{"location":"diffeqfinancial/#DiffEqFinancial.HestonProblem-NTuple{7, Any}","page":"API","title":"DiffEqFinancial.HestonProblem","text":"dS = μSdt + sqrtvSdW_1\ndv = κ(Θ-v)dt + σsqrtvdW_2\ndW_1 dW_2 = ρ dt\n\n\n\n","category":"method"},{"location":"diffeqfinancial/#DiffEqFinancial.MfStateProblem-NTuple{4, Any}","page":"API","title":"DiffEqFinancial.MfStateProblem","text":"dx = σ(t)e^at dW_t\n\n\n\n","category":"method"},{"location":"diffeqfinancial/#DiffEqFinancial.OrnsteinUhlenbeckProblem-NTuple{5, Any}","page":"API","title":"DiffEqFinancial.OrnsteinUhlenbeckProblem","text":"dx = a(r-x)dt + σ dW_t\n\n\n\n","category":"method"},{"location":"#DiffEqFinancial.jl","page":"Home","title":"DiffEqFinancial.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This repository holds problem definitions for common financial models for use in the DifferentialEquations ecosystem. The goal is to be a feature-complete set of solvers for the types of problems found in libraries like QuantLib. For example, one can easily specify, solve, and plot the solution for a trajectory of a Heston process via the commands:","category":"page"},{"location":"","page":"Home","title":"Home","text":"prob = HestonProblem(μ,κ,Θ,σ,ρ,u₀)\nsol = solve(prob)\nplot(sol)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Full documentation is in the DifferentialEquations.jl models documentation","category":"page"},{"location":"","page":"Home","title":"Home","text":"These solvers use DifferentialEquations.jl, meaning that high-performance and high order algorithms are available.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The project is looking for contributors who would like to implement more models. If you're interested and need help getting started, talk to Chris Rackauckas in the Gitter chat.","category":"page"},{"location":"#Contributing","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Please refer to the SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages for guidance on PRs, issues, and other matters relating to contributing to SciML.\nSee the SciML Style Guide for common coding practices and other style decisions.\nThere are a few community forums:\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Slack\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Zulip\nOn the Julia Discourse forums\nSee also SciML Community page","category":"page"},{"location":"#Reproducibility","page":"Home","title":"Reproducibility","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>and using this machine and Julia version.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using InteractiveUtils # hide\nversioninfo() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status(;mode = PKGMODE_MANIFEST) # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"You can also download the\n<a href=\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TOML\nversion = TOML.parse(read(\"../../Project.toml\",String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\",String))[\"name\"]\nlink = \"https://github.com/SciML/\"*name*\".jl/tree/gh-pages/v\"*version*\"/assets/Manifest.toml\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"\">manifest</a> file and the\n<a href=\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TOML\nversion = TOML.parse(read(\"../../Project.toml\",String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\",String))[\"name\"]\nlink = \"https://github.com/SciML/\"*name*\".jl/tree/gh-pages/v\"*version*\"/assets/Project.toml\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"\">project</a> file.","category":"page"}]
}

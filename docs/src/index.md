# DiffEqFinancial.jl

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

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
- There are a few community forums:
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Slack](https://julialang.org/slack/)
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
    - On the [Julia Discourse forums](https://discourse.julialang.org)
    - See also [SciML Community page](https://sciml.ai/community/)


## Reproducibility
```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```
```@example
using Pkg # hide
Pkg.status() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>and using this machine and Julia version.</summary>
```
```@example
using InteractiveUtils # hide
versioninfo() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```
```@example
using Pkg # hide
Pkg.status(;mode = PKGMODE_MANIFEST) # hide
```
```@raw html
</details>
```
```@raw html
You can also download the
<a href="
```
```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Manifest.toml"
Markdown.parse(link)
```
```@raw html
">manifest</a> file and the
<a href="
```
```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Project.toml"
Markdown.parse(link)
```
```@raw html
">project</a> file.
```

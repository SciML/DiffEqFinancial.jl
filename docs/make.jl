using Documenter, DiffEqFinancial

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(
    sitename = "DiffEqFinancial.jl",
    authors = "Chris Rackauckas",
    modules = [DiffEqFinancial],
    clean = true,
    doctest = false,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DiffEqFinancial/stable/"
    ),
    pages = pages
)

deploydocs(repo = "github.com/SciML/DiffEqFinancial.jl.git"; push_preview = true)

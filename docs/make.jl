using SpeciesDistributionModels
using Documenter

DocMeta.setdocmeta!(SpeciesDistributionModels, :DocTestSetup, :(using SpeciesDistributionModels); recursive=true)

makedocs(;
    modules=[SpeciesDistributionModels],
    authors="Tiem van der Deure <tvd@sund.ku.dk>, Rafael Schouten <rafaelschouten@gmail.com>",
    repo="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/{commit}{path}#{line}",
    sitename="SpeciesDistributionModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tiemvanderdeure.github.io/SpeciesDistributionModels.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tiemvanderdeure/SpeciesDistributionModels.jl",
    devbranch="master",
)

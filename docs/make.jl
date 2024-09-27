using Documenter, DocumenterVitepress
using SpeciesDistributionModels

makedocs(;
    modules=[SpeciesDistributionModels],
    authors= "Tiem van der Deure <tvd@sund.ku.dk>, Rafael Schouten <rafaelschouten@gmail.com>",
    repo="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl",
    sitename="SpeciesDistributionModels.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl",
        devurl = "dev",
        devbranch = "master",
        deploy_url = "tiemvanderdeure.github.io/SpeciesDistributionModels.jl",
    ),
    warnonly = true,
)

deploydocs(; 
    repo="github.com/tiemvanderdeure/SpeciesDistributionModels.jl",
    target = "build", # this is where Vitepress stores its output
    branch = "gh-pages",
    devbranch = "master",
    push_preview = true
)
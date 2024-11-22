using Documenter, DocumenterVitepress
using SpeciesDistributionModels

makedocs(;
    modules=[SpeciesDistributionModels],
    authors= "Tiem van der Deure <tvd@sund.ku.dk>, Rafael Schouten <rafaelschouten@gmail.com>",
    sitename="SpeciesDistributionModels.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/tiemvanderdeure/SpeciesDistributionModels.jl",
        devurl = "dev",
        devbranch = "main",
    ),
    warnonly = true
)

deploydocs(; 
    repo="github.com/tiemvanderdeure/SpeciesDistributionModels.jl",
    target = "build", # this is where Vitepress stores its output
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true
)
module SpeciesDistributionModelsMakieExt
    using Makie, SpeciesDistributionModels
    import SpeciesDistributionModels as SDM
    import SpeciesDistributionModels: interactive_evaluation
    import Statistics, Loess
    import DimensionalData as DD

    include("plotrecipes.jl")
end
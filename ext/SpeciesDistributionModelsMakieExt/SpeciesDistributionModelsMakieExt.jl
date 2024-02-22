module SpeciesDistributionModelsMakieExt
    using Makie, SpeciesDistributionModels
    import SpeciesDistributionModels: model_names, machine_evaluations
    include("plotrecipes.jl")
    export interactive_response_curves, interactive_evaluation, _boxplot
end
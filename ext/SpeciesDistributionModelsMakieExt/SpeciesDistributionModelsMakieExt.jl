module SpeciesDistributionModelsMakieExt
    using Makie, SpeciesDistributionModels
    import SpeciesDistributionModels as SDM
    import SpeciesDistributionModels: model_keys, machine_evaluations, sdm_machines, data, 
        _conf_mats_from_thresholds
    import SpeciesDistributionModels: interactive_evaluation
    import Statistics, Loess

    include("plotrecipes.jl")

end
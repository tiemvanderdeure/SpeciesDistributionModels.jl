module SpeciesDistributionModels

import Tables, StatsBase, Statistics, StatsAPI, StatsModels, LinearAlgebra
import MLJBase, StatisticalMeasures, StatisticalMeasuresBase, ScientificTypesBase, CategoricalArrays
import GLM, PrettyTables, Rasters, EvoTrees, DecisionTree, Makie, Shapley, Loess

using MLJBase: pdf

using Rasters: Raster, RasterStack, Band

using Makie: Toggle, Label, scatter!, lines!, Axis, Figure, GridLayout, lift

using ScientificTypesBase: Continuous, OrderedFactor, Multiclass, Count

export SDMensemble, predict, sdm_ensemble, select, machines, machine_keys, shap, 
    interactive_evaluation, interactive_response_curves,
    remove_collinear

include("data_utils.jl")
include("collinearity.jl")
include("models.jl")
include("ensemble.jl")
include("predict.jl")
include("explain.jl")
include("evaluate.jl")
include("interface.jl")
include("plots.jl")

end

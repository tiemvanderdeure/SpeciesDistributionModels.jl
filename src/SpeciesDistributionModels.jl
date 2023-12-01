module SpeciesDistributionModels

using MLJ 

import Tables, StatsBase, Statistics
import GLM, PrettyTables, Rasters, EvoTrees, DecisionTree, Makie, Shapley, Loess

using Rasters: Raster, RasterStack

using Makie: Toggle, Label, scatter!, lines!, Axis, Figure, GridLayout, lift

import CategoricalArrays.CategoricalArray

export SDMensemble, predict, sdm, select, machines, machine_keys, shap, 
    interactive_evaluation, interactive_response_curves

include("models.jl")
include("ensemble.jl")
include("predict.jl")
include("explain.jl")
include("plots.jl")

end

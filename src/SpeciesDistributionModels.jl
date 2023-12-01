module SpeciesDistributionModels

using MLJ 

import GLM, Tables, StatsBase, PrettyTables, Rasters, EvoTrees, DecisionTree, Shapley

using Rasters: Raster, RasterStack

import CategoricalArrays.CategoricalArray

export SDMensemble, predict, sdm, select, machines, machine_keys, shap

include("models.jl")
include("ensemble.jl")
include("predict.jl")
include("explain.jl")

end



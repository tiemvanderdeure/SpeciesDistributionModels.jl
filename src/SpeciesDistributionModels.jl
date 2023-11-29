module SpeciesDistributionModels

using MLJ

import GLM, Tables, StatsBase, PrettyTables, Rasters, EvoTrees, DecisionTree, Makie

using Makie: Toggle, Label, lines!, Axis, Figure, GridLayout, lift

using Rasters: Raster, RasterStack

import CategoricalArrays.CategoricalArray

export SDMensemble, predict, sdm, select, machines, machine_keys, interactive_evaluation

include("models.jl")
include("ensemble.jl")
include("predict.jl")
include("plots.jl")

end

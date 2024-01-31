module SpeciesDistributionModels

import Tables, StatsBase, Statistics, StatsAPI, StatsModels, LinearAlgebra
import MLJBase, StatisticalMeasures, StatisticalMeasuresBase, ScientificTypesBase, CategoricalArrays
import GLM, PrettyTables, Rasters, EvoTrees, DecisionTree, Shapley, Loess

using MLJBase: pdf

using Rasters: Raster, RasterStack

using ScientificTypesBase: Continuous, OrderedFactor, Multiclass, Count

export SDMensemble, predict, sdm, select, machines, machine_keys, shap, 
    remove_collinear

# export stubs for extensions
export interactive_response_curves, interactive_evaluation


include("collinearity.jl")
include("models.jl")
include("ensemble.jl")
include("predict.jl")
include("explain.jl")
include("evaluate.jl")

end

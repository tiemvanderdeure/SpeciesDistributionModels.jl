module SpeciesDistributionModels

import Tables, StatsBase, Statistics, StatsAPI, StatsModels, LinearAlgebra, Random, ThreadsX
import MLJBase, StatisticalMeasures, StatisticalMeasuresBase, ScientificTypesBase, CategoricalArrays
import GLM, PrettyTables, Rasters, EvoTrees, DecisionTree, Shapley, Loess

using MLJBase: pdf
using Rasters: Raster, RasterStack, Band
using ScientificTypesBase: Continuous, OrderedFactor, Multiclass, Count
using ComputationalResources: CPU1, CPUThreads
using Makie: Toggle, Label, scatter!, lines!, Axis, Figure, GridLayout, lift

using ScientificTypesBase: Continuous, OrderedFactor, Multiclass, Count

export SDMensemble, predict, sdm, select, machines, machine_keys,
    remove_collinear,
    explain, variable_importance, ShapleyValues,
    SDMmachineExplanation, SDMgroupExplanation, SDMensembleExplanation,
    SDMmachineEvaluation, SDMgroupEvaluation, SDMensembleEvaluation

include("data_utils.jl")
# export stubs for extensions
export interactive_response_curves, interactive_evaluation


include("collinearity.jl")
include("models.jl")
include("ensemble.jl")
include("predict.jl")
include("explain/explain.jl")
include("explain/shapley.jl")
include("evaluate.jl")
include("interface.jl")

end

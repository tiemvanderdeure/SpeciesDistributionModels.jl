module SpeciesDistributionModels

import Tables, StatsBase, Statistics, StatsAPI, StatsModels, LinearAlgebra, Random, ThreadsX
import MLJBase, StatisticalMeasures, StatisticalMeasuresBase, ScientificTypesBase, CategoricalArrays
import GLM, PrettyTables, Rasters, EvoTrees, DecisionTree, Shapley, Loess

using MLJBase: pdf
using Rasters: Raster, RasterStack, Band
using ScientificTypesBase: Continuous, OrderedFactor, Multiclass, Count
using ComputationalResources: CPU1, CPUThreads, AbstractCPU

using ScientificTypesBase: Continuous, OrderedFactor, Multiclass, Count
using StatisticalMeasures: auc, kappa, sensitivity, selectivity, accuracy
import MLJBase: StratifiedCV, CV, Holdout, ResamplingStrategy

export SDMensemble, predict, sdm, select, machines, machine_keys,
    remove_collinear,
    explain, variable_importance, ShapleyValues,
    SDMmachineExplanation, SDMgroupExplanation, SDMensembleExplanation,
    SDMmachineEvaluation, SDMgroupEvaluation, SDMensembleEvaluation

# re-export
export auc, kappa, sensitivity, selectivity, accuracy,
    Continuous, OrderedFactor, Multiclass, Count,
    StratifiedCV, CV, Holdout, ResamplingStrategy

include("data_utils.jl")
include("resample.jl")
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
include("extensions.jl")

end

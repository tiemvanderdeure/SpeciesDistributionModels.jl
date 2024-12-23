module SpeciesDistributionModels

import Tables, StatsBase, Statistics, StatsAPI, StatsModels, LinearAlgebra, Random, ThreadsX
import MLJBase, StatisticalMeasures, CategoricalArrays
import GLM, PrettyTables, Rasters, Shapley, Loess, Distances
import GeoInterface as GI

using Rasters: Raster, RasterStack, Band, DD
using ComputationalResources: CPU1, CPUThreads, AbstractCPU, CPUProcesses
using StatisticalMeasures: auc, kappa, sensitivity, selectivity, accuracy, StatisticalMeasuresBase
using ScientificTypesBase: Continuous, OrderedFactor, Multiclass, Count
import MLJBase: StratifiedCV, CV, Holdout, ResamplingStrategy, Machine, Probabilistic, pdf

export SDMensemble, predict, sdm, sdmdata, select, machines, machine_keys,
    remove_collinear, thin,
    explain, variable_importance, ShapleyValues,
    SDMmachineExplanation, SDMgroupExplanation, SDMensembleExplanation,
    SDMmachineEvaluation, SDMgroupEvaluation, SDMensembleEvaluation

# re-export
export auc, kappa, sensitivity, selectivity, accuracy,
    Continuous, OrderedFactor, Multiclass, Count,
    StratifiedCV, CV, Holdout, ResamplingStrategy
#include("learningnetwork.jl")
include("data_utils.jl")
include("resample.jl")
# export stubs for extensions
export interactive_response_curves, interactive_evaluation

include("collinearity.jl")
include("ensemble.jl")
include("predict.jl")
include("explain/explain.jl")
include("explain/shapley.jl")
include("evaluate.jl")
include("interface.jl")
include("extensions.jl")
include("thin.jl")

end

module SpeciesDistributionModels

import Tables, StatsBase, Statistics, StatsAPI, StatsModels, LinearAlgebra, Random, ThreadsX
import MLJBase, StatisticalMeasures, CategoricalArrays
import GLM, PrettyTables, Rasters, Shapley, Loess, Distances
import GeoInterface as GI
import DimensionalData as DD
import Rasters as RA
import ConstructionBase

using ComputationalResources: CPU1, CPUThreads, AbstractCPU, CPUProcesses
using StatisticalMeasures: auc, kappa, sensitivity, selectivity, accuracy, StatisticalMeasuresBase
using ScientificTypesBase: Continuous, OrderedFactor, Multiclass, Count
import MLJBase: StratifiedCV, CV, Holdout, ResamplingStrategy, Machine, Probabilistic, pdf
import DimensionalData: DimArray, DimStack, Dim, broadcast_dims

export SDMensemble, SDMexplanation, SDMevaluation,
    predict, sdm, sdmdata, select, machines,
    remove_collinear, thin,
    explain, variable_importance, ShapleyValues

# re-export
export auc, kappa, sensitivity, selectivity, accuracy,
    Continuous, OrderedFactor, Multiclass, Count,
    StratifiedCV, CV, Holdout, ResamplingStrategy

# export stubs for extensions
export interactive_response_curves, interactive_evaluation

# Data handling and types
include("data_utils.jl")
include("dimtypes.jl")

# Interface
include("interface.jl")
include("extensions.jl")

# Utilities
include("thin.jl")
include("collinearity.jl")

# Core functionality
include("ensemble.jl")
include("predict.jl")
include("evaluate.jl")

# Explain
include("explain/explain.jl")
include("explain/shapley.jl")

end

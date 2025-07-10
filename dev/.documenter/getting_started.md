
# Getting started with SpeciesDistributionModels.jl {#Getting-started-with-SpeciesDistributionModels.jl}

This package is under active development. Be advised that it may change at any time.

## Installation {#Installation}

This package is not registered yet, but can easily be installed directly from GitHub.

```julia
] add github.com/tiemvanderdeure/SpeciesDistributionModels.jl
```


## Core functionality {#Core-functionality}

A species distribution modelling workflow would typically consist of some data handling, then fitting a model, then evaluating it, and finally projecting to spatial data.

In this package, the main data handling tool is the [sdmdata](/api#SpeciesDistributionModels.sdmdata-Tuple{Any,%20Any}) function, which takes two Tables.jl-compatible data objects (e.g. DataFrames) as input and returns an `SDMdata` object. You can also specify a resampling strategy or select a subset of variables in this step.

Next, this object and a `NamedTuple` of models is passed to the [sdm](/api#SpeciesDistributionModels.sdm-Tuple{Any,%20Any}) function to fit the models. The models can be any object that implements the MLJ interface and is compatible with binary categorical data. See the [MLJ model registry](https://juliaai.github.io/MLJ.jl/dev/model_browser/#Classification) for a list of available models.

A fit ensemble can then be passed to functions like `SDM.evaluate` and `SDM.predict`. 

## Types {#Types}

Most types in this package are named acoording to the same nested structure with three levels; machine, group, and ensemble. This does for fitted models ([SDMmachine](@ref), [SDMgroup](@ref), and [SDMensemble](@ref)), evaluations of those models ([SDMmachineEvaluation](@ref) etc.), and mdoel explanations ([SDMmachineExplanation](@ref) etc.). The meaning of machine, group, and ensemble in this package is as follows:
- machine: a single instance of one particular model fit using one particular set of data.
  
- group: one or more machines, which are instances of the same model, but may be fit on different sets of data (e.g. resampling folds)
  
- ensemble: one or more groups, which each use different models.
  

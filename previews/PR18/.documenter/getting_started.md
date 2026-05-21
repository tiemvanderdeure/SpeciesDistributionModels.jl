
# Getting started with SpeciesDistributionModels.jl {#Getting-started-with-SpeciesDistributionModels.jl}

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

## Dimensional types {#Dimensional-types}

Most main objects in this package are built on DimensionalData.jl. [sdm](/api#SpeciesDistributionModels.sdm-Tuple{Any,%20Any}) returns an [SDMensemble](/api#SpeciesDistributionModels.SDMensemble), which is an `AbstractDimArray` subtype with `:model` and `:fold` dimensions. It is thus easy to understand what `machine`s (from MLJ) each fit ensemble contains, or to subset an ensemble.

Similarly `SDM.evalute` returns [`SDMevaluation`](/api#SpeciesDistributionModels.SDMevaluation) and `SDM.explain` returns [`SDMexplanation`](/api#SpeciesDistributionModels.SDMexplanation), which both are `AbstractDimStack` subtypes that preserve the same `:model` and `:fold` dims.

`SDM.predict` returns a DimArray or Raster, where dimensions are again preserved. Existing functionality from DimensionalData can then be used to summarize outputs. For instance, the mean suitability predicted can be generated with `mean(x; dims = (:model, :fold))`.

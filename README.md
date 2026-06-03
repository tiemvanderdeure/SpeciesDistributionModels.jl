++README.md
# SpeciesDistributionModels.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tiemvanderdeure.github.io/SpeciesDistributionModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tiemvanderdeure.github.io/SpeciesDistributionModels.jl/dev/)
[![Build Status](https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tiemvanderdeure/SpeciesDistributionModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tiemvanderdeure/SpeciesDistributionModels.jl)

SpeciesDistributionModels.jl aims to provide a flexible and easy-to-use pipeline for fitting, evaluation, and using species distribution models. It is based on the MLJ ecosystem and integrated with Rasters.jl.

## Core interface functions

SpeciesDistributionModels.jl provides a small, consistent interface around common tasks in presence/background SDM workflows. It has functions for handling input data, fitting ensembles, evaluating and understanding those models, and predicting to raster data.

- construct SDM input objects with `sdmdata` (wraps presence and background tables + resampling scheme),
- build and fit ensembles of MLJ models with `sdm` (returns an `SDMensemble`),
- evaluate fitted ensembles with `evaluate`,
- compute explanations with `explain`, and
- make predictions to tables or to raster stacks with `predict`.

The package is closely integrated with DimensionalData.jl. `SDMensemble` is an `AbstractDimArray` subtype with `:model` and `:fold` dimensions, and `SDMevaluation`/`SDMexplanation` are `AbstractDimStack` subtypes built from the same ensemble dimensions. This means fitted results, evaluations, and explanations all preserve dimensional metadata and integrate smoothly with DimensionalData workflows.

Fitting an ensemble is as simple as:
```
using SpeciesDistributionModels
using MLJGLMInterface: LinearBinaryClassifier
using Maxnet: MaxnetBinaryClassifier
n = 100
backgrounddata = (a = rand(n), b = rand(n), c = rand(n))
presencedata = (a = rand(n), b = rand(n).^2, c = sqrt.(rand(n)))
data = sdmdata(presencedata, backgrounddata; resampler = CV())
models = (lm = LinearBinaryClassifier(), maxnet = MaxnetBinaryClassifier())
sdm(data, models)
```

This returns an SDMensemble object, which clearly states the dimensions it has. 
```
┌ 2×6 SDMensemble{Machine, 2} ┐
├─────────────────────────────┴────────────────────── dims ┐
  ↓ model Categorical{Symbol} [:lm, :maxnet] Unordered,
  → fold Sampled{Int64} 1:6 ForwardOrdered Regular Points
└──────────────────────────────────────────────────────────┘
 :lm      LinearBinaryClassifier(fit_intercept = true, …)
 :maxnet  MaxnetBinaryClassifier(features = , …)
 ```

## Integration with MLJ
This package uses MLJ under the hood, and therefore any MLJ-compatible classification model can be used in an ensemble. This includes popular models such as random forest, boosted regression trees, and Maxnet.

See the [MLJ website](https://juliaml.ai/machines) for all the available models.
 
## Usage example
See the documentation for a full example of a workflow from downloading data to generating predictions.
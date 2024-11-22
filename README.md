# SpeciesDistributionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tiemvanderdeure.github.io/SpeciesDistributionModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tiemvanderdeure.github.io/SpeciesDistributionModels.jl/dev/)
[![Build Status](https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tiemvanderdeure/SpeciesDistributionModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tiemvanderdeure/SpeciesDistributionModels.jl)

SpeciesDistributionModels.jl aims to provide a flexible and easy-to-use pipeline for fitting, evaluation, and using species distribution models. It is based on the MLJ ecosystem and integrated with Rasters.jl.

This package is still very much work in progress.

## Usage
The following example models the distribution of _Anopheles nili_, a malaria vector mosquito distributed in the tropical regions of Africa.

We'll use several other packages in addition to SpeciesDistributionModels
```julia
using SpeciesDistributionModels, Rasters, RasterDataSources, GBIF2, LibGEOS, ArchGDAL, StatsBase
import GeometryOps as GO
import SpeciesDistributionModels as SDM
```

We start by extracting occurrence records from GBIF using `GBIF2`. 
```julia
# extract occurrence records for Anopheles rufipes
nili = species_match("Anopheles nili")
occurrences = occurrence_search(nili, hasCoordinate = true, year = (1990, 2020), limit = 1000)
presence_points = unique(occurrences.geometry)
```

The region of interest will be defined by a convex hull around our presence points, buffered by 5 degrees.
```julia
roi = GO.buffer(GO.convex_hull(presence_points), 5)
```

Now we'll load bioclimatic variables from WorldClim, using `RasterDataSources` and `Rasters`.
```julia
# load bioclimatic variables
bio = crop(RasterStack(WorldClim{BioClim}); to = roi)

```

The `extract` function from `Rasters` makes it easy to extract the variables at our presence points.
```julia
# get values at presence points
presences = extract(bio, presence_points; skipmissing = true, geometry = false)
```

For background (pseudo-absence) points, we draw 300 points uniformly distributed within the region of interest.
```julia
# get a raster with dimensions of bio that is true for all cells within roi and where bio is not missing
roi_raster = rasterize(roi; to = bio, fill = true, missingval = false) .* Rasters.boolmask(bio)
# draw random cells in the region of interest
background_points = sample(DimIndices(roi_raster), weights(roi_raster), 300)
# get bioclimatic variables at the background locations
background = map(p -> bio[p...], background_points)
```

Let's choose some settings for modelling. Here we're using 4 different models, resample using 3-fold stratified cross-validation, and use 3 predictor variables.
```julia
# choose models, a resampler, and predictor variables
models = [SDM.linear_model(), SDM.random_forest(), SDM.random_forest(; max_depth = 3), SDM.boosted_regression_tree()]
resampler = StratifiedCV(; nfolds = 3)
predictors = (:bio1, :bio7, :bio12)
```

Now we're ready to run the models. We construct an ensemble using the `sdm` function.
```julia
# run models and construct ensemble
ensemble = sdm(presences, background; models, resampler, predictors)
```
When an ensemble is printed it shows some basic information.
```
SDMensemble with 12 machines across 4 groups
Occurence data: Presence-Absence with 27 presences and 300 absences 
Predictors: bio1 (Continuous), bio7 (Continuous), bio12 (Continuous)
┌────────────────┬──────────────┬──────────┐
│          model │    resampler │ machines │
├────────────────┼──────────────┼──────────┤
│         Linear │ StratifiedCV │        3 │
│ RandomForest_1 │ StratifiedCV │        3 │
│ RandomForest_2 │ StratifiedCV │        3 │
│        EvoTree │ StratifiedCV │        3 │
└────────────────┴──────────────┴──────────┘
```

To see how our models performed, use the `evaluate` function. Here we use it with default settings.
```julia
evaluation = SDM.evaluate(ensemble)
```
```
SDMensembleEvaluation with 4 performance measures
train
┌────────────────┬──────────┬──────────┬───────────┬──────────┐
│          model │ accuracy │      auc │  log_loss │    kappa │
│         Symbol │  Float64 │  Float64 │   Float64 │  Float64 │
├────────────────┼──────────┼──────────┼───────────┼──────────┤
│         Linear │ 0.912844 │ 0.692685 │  0.265474 │ 0.179984 │
│ RandomForest_1 │      1.0 │      1.0 │ 0.0619646 │      1.0 │
│ RandomForest_2 │ 0.955657 │ 0.936343 │  0.288469 │ 0.665568 │
│        EvoTree │      1.0 │      1.0 │ 0.0115611 │      1.0 │
└────────────────┴──────────┴──────────┴───────────┴──────────┘
test
┌────────────────┬──────────┬──────────┬──────────┬──────────┐
│          model │ accuracy │      auc │ log_loss │    kappa │
│         Symbol │  Float64 │  Float64 │  Float64 │  Float64 │
├────────────────┼──────────┼──────────┼──────────┼──────────┤
│         Linear │ 0.908257 │ 0.672963 │ 0.273263 │ 0.170727 │
│ RandomForest_1 │ 0.966361 │ 0.876296 │ 0.178809 │ 0.738267 │
│ RandomForest_2 │ 0.938838 │ 0.773333 │ 0.649535 │ 0.404121 │
│        EvoTree │  0.95107 │ 0.875185 │  0.25651 │ 0.630953 │
└────────────────┴──────────┴──────────┴──────────┴──────────┘
```

Finally, we predict back to a raster, taking the simple mean of all models to generate a final prediction.
```julia
pr = SDM.predict(ensemble, bio; reducer = mean)
```
```
252×166 Raster{Union{Missing, Float64},2} with dimensions: 
  X Projected{Float64} LinRange{Float64}(-17.1667, 24.6667, 252) ForwardOrdered Regular 
Intervals{Start} crs: WellKnownText,
  Y Projected{Float64} LinRange{Float64}(17.3333, -10.1667, 166) ReverseOrdered Regular 
Intervals{Start} crs: WellKnownText
extent: Extent(X = (-17.166666666666686, 24.83333333333334), Y = (-10.166666666666671, 17.500000000000004))
missingval: missing
crs: GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]
parent:
           17.3333      17.1667      …  -9.83333     -10.0         -10.1667
 -17.1667    missing      missing         missing       missing       missing
 -17.0       missing      missing         missing       missing       missing
 -16.8333    missing      missing         missing       missing       missing
   ⋮                                 ⋱
  24.1667   0.00273958   0.00528863  …   0.00558025    0.00578908    0.00613832
  24.3333   0.00357797   0.00445221      0.00637058    0.00660174    0.00547219
  24.5      0.00610894   0.00443861      0.00616782    0.00583453    0.00602396
  24.6667   0.00440761   0.00360355      0.00649026    0.00615647    0.00665752
```

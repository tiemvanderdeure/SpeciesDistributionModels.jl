---
---

# Example of a full species distribution modelling workflow {#Example-of-a-full-species-distribution-modelling-workflow}

This example fits and projects species distribution models for the tree species _Eucaplytus regnans_ from scratch.

## Environmental data {#Environmental-data}

We start by downloading environmental data. We use the [Rasters](https://github.com/rafaqz/Rasters.jl) package to handle raster data, and RasterDataSources.jl to automatically download Worldclim data.

In this example, we just download bioclimatic variables 1 and 12 (annual mean temperature and annual mean precipitation). We then mask to the country borders of Australia.

```julia
using Rasters, RasterDataSources, ArchGDAL, NaturalEarth, DataFrames
bio = RasterStack(WorldClim{BioClim}, (1,12))
countries = naturalearth("ne_10m_admin_0_countries") |> DataFrame
australia = subset(countries, :NAME => ByRow(==("Australia"))).geometry
bio_aus = Rasters.trim(mask(bio; with = australia)[X = 110 .. 156, Y = -45 .. -10])
```


```ansi
[90m┌ [39m[38;5;209m244[39m×[38;5;32m198[39m RasterStack[90m ┐[39m
[90m├─────────────────────┴────────────────────────────────────────────────── dims ┐[39m
  [38;5;209m↓ [39m[38;5;209mX[39m Projected{Float64} [38;5;209m112.99999999999999:0.16666666666666666:153.49999999999997[39m [38;5;244mForwardOrdered[39m [38;5;244mRegular[39m [38;5;244mIntervals{Start}[39m,
  [38;5;32m→ [39m[38;5;32mY[39m Projected{Float64} [38;5;32m-10.833333333333332:-0.16666666666666666:-43.666666666666664[39m [38;5;244mReverseOrdered[39m [38;5;244mRegular[39m [38;5;244mIntervals{Start}[39m
[90m├────────────────────────────────────────────────────────────────────── layers ┤[39m
[38;5;37m  :bio1 [39m[90m eltype: [39mUnion{Missing, Float32}[90m dims: [39m[38;5;209mX[39m, [38;5;32mY[39m[90m size: [39m[38;5;209m244[39m×[38;5;32m198[39m
[38;5;37m  :bio12[39m[90m eltype: [39mUnion{Missing, Float32}[90m dims: [39m[38;5;209mX[39m, [38;5;32mY[39m[90m size: [39m[38;5;209m244[39m×[38;5;32m198[39m
[90m├────────────────────────────────────────────────────────────────────── raster ┤[39m
[90m  missingval: [39mmissing
[90m  extent: [39mExtent(X = (112.99999999999999, 153.66666666666663), Y = (-43.666666666666664, -10.666666666666666))
[90m  crs: [39mGEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.25722...
[90m└──────────────────────────────────────────────────────────────────────────────┘[39m
```


Let's plot this data to see what it looks like.

```julia
using CairoMakie
Rasters.rplot(bio_aus)
```

![](gtybvxs.png){width=600px height=450px}

## Occurrence data {#Occurrence-data}

Next, we use [GBIF2.jl](https://github.com/rafaqz/GBIF2.jl) to download occurrence records for this species. We use the [thin](/api#SpeciesDistributionModels.thin) function in this package to weed out occurrences that are very close to each other, using a cut-off of 5km.

```julia
using GBIF2, SpeciesDistributionModels
sp = species_match("Eucalyptus regnans")
occurrences_raw = occurrence_search(sp; year = (1970,2000), country = "AU", hasCoordinate = true, limit = 2000)
occurrences = thin(occurrences_raw.geometry, 5000)
```


```ansi
1267-element Vector{Tuple{Float64, Float64}}:
 (145.98298, -38.33349)
 (147.255219, -41.237844)
 (147.497946, -41.301006)
 (146.35, -37.36667)
 (147.899871, -41.147461)
 (146.782197, -41.376415)
 (146.779808, -41.37551)
 (146.24194, -38.52083)
 (146.3125, -38.56111)
 (146.45583, -38.46889)
 ⋮
 (146.634191, -42.309965)
 (147.970709, -41.134905)
 (147.985049, -41.145683)
 (147.795927, -41.204164)
 (145.791172, -41.193094)
 (145.30611, -37.52194)
 (146.16306, -37.82861)
 (146.16306, -37.83)
 (146.16139, -37.82861)
```


## Background points {#Background-points}

Next, we sample random points to use as background points.

Let's plot both the occurrence and background points to see where _Eucalyptus regnans_ is found.

```julia
using StatsBase # to active Rasters.sample
bg_data = Rasters.sample(bio_aus, 500; skipmissing = true, geometry = (X,Y))
bg_points = getproperty.(bg_data, :geometry)
fig, ax, pl = plot(bio_aus.bio1)
scatter!(ax, occurrences; color = :red)
scatter!(ax, bg_points; color = :grey)
fig
```

![](hfbycac.png){width=600px height=450px}

## Handling data {#Handling-data}

SpeciesDistributionModels.jl has a [sdmdata](/api#SpeciesDistributionModels.sdmdata) function to handle input data. It takes tabular presence and background data as inputs, such as what is returned by `Rasters.extract` and `Rasters.sample`.

```julia
using SpeciesDistributionModels
p_data = extract(bio_aus, occurrences; skipmissing = true)
data = sdmdata(p_data, bg_data; resampler = CV(nfolds = 3))
```


```ansi
SDMdata object with [0m[1m1250[22m presence points and [0m[1m500[22m absence points. 
 
[0m[1mResampling: [22m
Data is divided into 3 folds using resampling strategy CV(nfolds = 3, …).
┌──────┬─────────┬────────┐
│[1m fold [0m│[1m # train [0m│[1m # test [0m│
├──────┼─────────┼────────┤
│    1 │    1166 │    584 │
│    2 │    1167 │    583 │
│    3 │    1167 │    583 │
└──────┴─────────┴────────┘
[0m[1mPredictor variables: [22m
┌───────┬────────────┬─────────┐
│[22m names [0m│[22m scitypes   [0m│[22m types   [0m│
├───────┼────────────┼─────────┤
│ bio1  │ Continuous │ Float32 │
│ bio12 │ Continuous │ Float32 │
└───────┴────────────┴─────────┘
Also contains geometry data
```


## Fitting an ensemble {#Fitting-an-ensemble}

Now that we have our `data` object with presence and background data, we can fit our ensemble. The `sdm` function fits a whole ensemble, taking two arguments: a data object and a `NamedTuple` with models the ensemble should have. This can be any MLJ-compatible model. In this case, we use Maxnet, boosted regression trees (from the EvoTrees.jl package), and a GLM.

```julia
using Maxnet: MaxnetBinaryClassifier
using EvoTrees: EvoTreeClassifier
using MLJGLMInterface: LinearBinaryClassifier
models = (
  maxnet = MaxnetBinaryClassifier(),
  brt = EvoTreeClassifier(),
  glm = LinearBinaryClassifier()
)

ensemble = sdm(data, models)
```


```ansi
[90m┌ [39m[38;5;209m3[39m×[38;5;32m3[39m DimArray{MLJBase.Machine{M, OM, true} where {M, OM}, 2}[90m ┐[39m
[90m├─────────────────────────────────────────────────────── dims ┤[39m
  [38;5;209m↓ [39m[38;5;209mmodel[39m Categorical{Symbol} [38;5;209m[:maxnet, …, :glm][39m [38;5;244mUnordered[39m,
  [38;5;32m→ [39m[38;5;32mfold[39m Sampled{Int64} [38;5;32m1:3[39m [38;5;244mForwardOrdered[39m [38;5;244mRegular[39m [38;5;244mPoints[39m
[90m└─────────────────────────────────────────────────────────────┘[39m
 [38;5;209m:maxnet[39m  MaxnetBinaryClassifier(features = , …)
 [38;5;209m:brt[39m     EvoTreeClassifier(loss = mlogloss, …)
 [38;5;209m:glm[39m     LinearBinaryClassifier(fit_intercept = true, …)
```


## Evaluating an ensemble {#Evaluating-an-ensemble}

We can evaluate the entire ensemble using any metric from [StatisticalMeasures.jl](https://github.com/JuliaAI/StatisticalMeasures.jl).

```julia
import SpeciesDistributionModels as SDM
ev = SDM.evaluate(ensemble; measures = (; auc, accuracy))
```


```ansi
SDMevaluation with dimensions:
[90m┌ [39m[38;5;209m3[39m×[38;5;32m3[39m×[38;5;81m2[39m×[38;5;204m2[39m DimStack[90m ┐[39m
[90m├──────────────────┴───────────────────────────────────────────────────── dims ┐[39m
  [38;5;209m↓ [39m[38;5;209mmodel[39m Categorical{Symbol} [38;5;209m[:maxnet, …, :glm][39m [38;5;244mUnordered[39m,
  [38;5;32m→ [39m[38;5;32mfold[39m Sampled{Int64} [38;5;32m1:3[39m [38;5;244mForwardOrdered[39m [38;5;244mRegular[39m [38;5;244mPoints[39m,
  [38;5;81m↗ [39m[38;5;81mdataset[39m Categorical{Symbol} [38;5;81m[:train, :test][39m [38;5;244mUnordered[39m,
  [38;5;204m⬔ [39m[38;5;204mmeasure[39m Categorical{Symbol} [38;5;204m[:auc, :accuracy][39m [38;5;244mUnordered[39m
[90m├────────────────────────────────────────────────────────────────────── layers ┤[39m
[38;5;37m  :score    [39m[90m eltype: [39mFloat64[90m dims: [39m[38;5;209mmodel[39m, [38;5;32mfold[39m, [38;5;81mdataset[39m, [38;5;204mmeasure[39m[90m size: [39m[38;5;209m3[39m×[38;5;32m3[39m×[38;5;81m2[39m×[38;5;204m2[39m
[38;5;37m  :threshold[39m[90m eltype: [39mUnion{Missing, Float64}[90m dims: [39m[38;5;209mmodel[39m, [38;5;32mfold[39m, [38;5;81mdataset[39m, [38;5;204mmeasure[39m[90m size: [39m[38;5;209m3[39m×[38;5;32m3[39m×[38;5;81m2[39m×[38;5;204m2[39m


Mean training performance:
 [38;5;209m↓[39m [38;5;32m→[39m        [38;5;32m:auc[39m      [38;5;32m:accuracy[39m
  [38;5;209m:maxnet[39m  0.992683  0.991715
  [38;5;209m:brt[39m     0.993891  0.993715
  [38;5;209m:glm[39m     0.985387  0.990286

Mean test performance:
 [38;5;209m↓[39m [38;5;32m→[39m        [38;5;32m:auc[39m      [38;5;32m:accuracy[39m
  [38;5;209m:maxnet[39m  0.992634  0.992002
  [38;5;209m:brt[39m     0.99391   0.993716
  [38;5;209m:glm[39m     0.985328  0.990287
```


## Predicting {#Predicting}

Next, we predict climatic suitability of the species throughout Australia using `SpeciesDistributionModels.predict`.  By specifying the `reducer` argument, only one value is returned for each grid cell. By default, the dimensions of the ensemble are retained and the predicted value for each ensemble member is 

```julia
pred = SDM.predict(ensemble, bio_aus; reducer = mean)
plot(pred; colorrange = (0,1))
```

![](mzvpsnb.png){width=600px height=450px}

## Understanding the model {#Understanding-the-model}

To understand variable importance and responses, use `SDM.explain`, which takes an ensemble as the first argument and a method keyword argument. Currently, the only implemented method is Shapley values from the [Shapley.jl](https://gitlab.com/ExpandingMan/Shapley.jl) package.

```julia
expl = SDM.explain(ensemble; method = ShapleyValues(8))
variable_importance(expl)
```


```ansi
(bio1 = 0.2917428592905032, bio12 = 0.08045275850100682)
```


We can also interactively plot the model explanation to get response curves.

```julia
interactive_response_curves(expl)
```

![](hhzjiys.png){width=600px height=450px}

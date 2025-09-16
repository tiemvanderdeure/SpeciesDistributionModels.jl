
# Example of a full species distribution modelling workflow {#Example-of-a-full-species-distribution-modelling-workflow}

This example fits and projects species distribution models for the tree species _Eucaplytus regnans_ from scratch.

## Environmental data {#Environmental-data}

We start by downloading environmental data. We use the [Rasters](www.github.com/rafaqz/Rasters.jl) package to handle raster data, and RasterDataSources.jl to automatically download Worldclim data.

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
  [38;5;32m→ [39m[38;5;32mY[39m Projected{Float64} [38;5;32m-10.833333333333334:-0.16666666666666666:-43.666666666666664[39m [38;5;244mReverseOrdered[39m [38;5;244mRegular[39m [38;5;244mIntervals{Start}[39m
[90m├────────────────────────────────────────────────────────────────────── layers ┤[39m
[38;5;37m  :bio1 [39m[90m eltype: [39mUnion{Missing, Float32}[90m dims: [39m[38;5;209mX[39m, [38;5;32mY[39m[90m size: [39m[38;5;209m244[39m×[38;5;32m198[39m
[38;5;37m  :bio12[39m[90m eltype: [39mUnion{Missing, Float32}[90m dims: [39m[38;5;209mX[39m, [38;5;32mY[39m[90m size: [39m[38;5;209m244[39m×[38;5;32m198[39m
[90m├────────────────────────────────────────────────────────────────────── raster ┤[39m
[90m  missingval: [39mmissing
[90m  extent: [39mExtent(X = (112.99999999999999, 153.66666666666663), Y = (-43.666666666666664, -10.666666666666668))
[90m  crs: [39mGEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.25722...
[90m└──────────────────────────────────────────────────────────────────────────────┘[39m
```


Let&#39;s plot this data to see what it looks like.

```julia
using CairoMakie
Rasters.rplot(bio_aus)
```

![](lgflzka.png){width=600px height=450px}

## Occurrence data {#Occurrence-data}

Next, we use [GBIF2.jl](www.github.com/rafaqz/GBIF2.jl) to download occurrence records for this species. We use the [thin](/api#SpeciesDistributionModels.thin-Tuple{Any,%20Any}) function in this package to weed out occurrences that are very close to each other, using a cut-off of 5km.

```julia
using GBIF2, SpeciesDistributionModels
sp = species_match("Eucalyptus regnans")
occurrences_raw = occurrence_search(sp; year = (1970,2000), country = "AU", hasCoordinate = true, limit = 2000)
occurrences = thin(occurrences_raw.geometry, 5000)
```


```ansi
1249-element Vector{Tuple{Float64, Float64}}:
 (145.98298, -38.33349)
 (147.497946, -41.301006)
 (145.19167, -37.41639)
 (146.35, -37.36667)
 (147.899871, -41.147461)
 (146.779808, -41.37551)
 (146.19722, -38.56889)
 (146.30667, -38.56472)
 (146.24194, -38.52083)
 (146.30333, -38.57611)
 ⋮
 (146.634191, -42.309965)
 (147.970709, -41.134905)
 (147.985049, -41.145683)
 (145.791172, -41.193094)
 (145.30611, -37.52194)
 (146.16306, -37.82861)
 (146.16306, -37.83)
 (146.16139, -37.82861)
 (146.15806, -37.83)
```


## Background points {#Background-points}

Next, we sample random points to use as background points.

Let&#39;s plot both the occurrence and background points to see where _Eucalyptus regnans_ is found.

```julia
using StatsBase # to active Rasters.sample
bg_data = Rasters.sample(bio_aus, 500; skipmissing = true, geometry = (X,Y))
bg_points = getproperty.(bg_data, :geometry)
fig, ax, pl = plot(bio_aus.bio1)
scatter!(ax, occurrences; color = :red)
scatter!(ax, bg_points; color = :grey)
fig
```

![](hmcitmt.png){width=600px height=450px}

## Handling data {#Handling-data}

SpeciesDistributionModels.jl has a [sdmdata](/api#SpeciesDistributionModels.sdmdata-Tuple{Any,%20Any}) function to handle input data. It takes tabular presence and background data as inputs, such as what is returned by `Rasters.extract` and `Rasters.sample`.

```julia
using SpeciesDistributionModels
p_data = extract(bio_aus, occurrences; skipmissing = true)
data = sdmdata(p_data, bg_data; resampler = CV(nfolds = 3))
```


```ansi
SDMdata object with [0m[1m1234[22m presence points and [0m[1m500[22m absence points. 
 
[0m[1mResampling: [22m
Data is divided into 3 folds using resampling strategy CV(nfolds = 3, …).
┌──────┬─────────┬────────┐
│[1m fold [0m│[1m # train [0m│[1m # test [0m│
├──────┼─────────┼────────┤
│    1 │    1156 │    578 │
│    2 │    1156 │    578 │
│    3 │    1156 │    578 │
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
[90m┌ [39m[38;5;209m3[39m×[38;5;32m3[39m SDMensemble{MLJBase.Machine, 2}[90m ┐[39m
[90m├─────────────────────────────────────┴──────────────── dims ┐[39m
  [38;5;209m↓ [39m[38;5;209mmodel[39m Categorical{Symbol} [38;5;209m[:maxnet, …, :glm][39m [38;5;244mUnordered[39m,
  [38;5;32m→ [39m[38;5;32mfold[39m Sampled{Int64} [38;5;32m1:3[39m [38;5;244mForwardOrdered[39m [38;5;244mRegular[39m [38;5;244mPoints[39m
[90m└────────────────────────────────────────────────────────────┘[39m
 [38;5;209m:maxnet[39m  …  MaxnetBinaryClassifier(features = , …)
 [38;5;209m:brt[39m        EvoTrees.EvoTreeClassifier
 - loss: mlogloss
 - metric: mlogloss
 - nrounds: 100
 - bagging_size: 1
 - early_stopping_rounds: 9223372036854775807
 - L2: 1.0
 - lambda: 0.0
 - gamma: 0.0
 - eta: 0.1
 - max_depth: 6
 - min_weight: 1.0
 - rowsample: 1.0
 - colsample: 1.0
 - nbins: 64
 - alpha: 0.5
 - tree_type: binary
 - rng: Random.MersenneTwister(123, (0, 60120, 59118, 150))
 - device: cpu

 [38;5;209m:glm[39m        LinearBinaryClassifier(fit_intercept = true, …)
```


## Evaluating an ensemble {#Evaluating-an-ensemble}

We can evaluate the entire ensemble using any metric from [StatisticalMeasures.jl](https://github.com/JuliaAI/StatisticalMeasures.jl).

```julia
import SpeciesDistributionModels as SDM
ev = SDM.evaluate(ensemble; measures = (; auc, accuracy))
```


```ansi
SDMensembleEvaluation with dimensions:
[90m┌ [39m[38;5;209m3[39m×[38;5;32m3[39m×[38;5;81m2[39m×[38;5;204m2[39m SDMensembleEvaluation[90m ┐[39m
[90m├───────────────────────────────┴──────────────────────────────────────── dims ┐[39m
  [38;5;209m↓ [39m[38;5;209mmodel[39m Categorical{Symbol} [38;5;209m[:maxnet, …, :glm][39m [38;5;244mUnordered[39m,
  [38;5;32m→ [39m[38;5;32mfold[39m Sampled{Int64} [38;5;32m1:3[39m [38;5;244mForwardOrdered[39m [38;5;244mRegular[39m [38;5;244mPoints[39m,
  [38;5;81m↗ [39m[38;5;81mdataset[39m Categorical{Symbol} [38;5;81m[:train, :test][39m [38;5;244mReverseOrdered[39m,
  [38;5;204m⬔ [39m[38;5;204mmeasure[39m Categorical{Symbol} [38;5;204m[:auc, :accuracy][39m [38;5;244mReverseOrdered[39m
[90m├────────────────────────────────────────────────────────────────────── layers ┤[39m
[38;5;37m  :score    [39m[90m eltype: [39mFloat64[90m dims: [39m[38;5;209mmodel[39m, [38;5;32mfold[39m, [38;5;81mdataset[39m, [38;5;204mmeasure[39m[90m size: [39m[38;5;209m3[39m×[38;5;32m3[39m×[38;5;81m2[39m×[38;5;204m2[39m
[38;5;37m  :threshold[39m[90m eltype: [39mUnion{Missing, Float64}[90m dims: [39m[38;5;209mmodel[39m, [38;5;32mfold[39m, [38;5;81mdataset[39m, [38;5;204mmeasure[39m[90m size: [39m[38;5;209m3[39m×[38;5;32m3[39m×[38;5;81m2[39m×[38;5;204m2[39m
[38;5;37m  :ensemble [39m[90m eltype: [39mMLJBase.Machine[90m dims: [39m[38;5;209mmodel[39m, [38;5;32mfold[39m, [38;5;81mdataset[39m, [38;5;204mmeasure[39m[90m size: [39m[38;5;209m3[39m×[38;5;32m3[39m×[38;5;81m2[39m×[38;5;204m2[39m


Mean training performance:
 [38;5;209m↓[39m [38;5;32m→[39m        [38;5;32m:auc[39m      [38;5;32m:accuracy[39m
  [38;5;209m:maxnet[39m  0.997655  0.995386
  [38;5;209m:brt[39m     0.997411  0.99654
  [38;5;209m:glm[39m     0.995593  0.995098

Mean test performance:
 [38;5;209m↓[39m [38;5;32m→[39m        [38;5;32m:auc[39m      [38;5;32m:accuracy[39m
  [38;5;209m:maxnet[39m  0.997454  0.995963
  [38;5;209m:brt[39m     0.997235  0.997116
  [38;5;209m:glm[39m     0.995362  0.995386
```


## Predicting {#Predicting}

Next, we the climatic suitability of the species throughout Australia using `SpeciesDistributionModels.predict`. We can specify a `reducer` argument to get a single value, instead of a prediction for each member in the ensemble.

```@example test
pred = SDM.predict(ensemble, bio_aus; reducer = mean)
plot(pred; colorrange = (0,1))
```


## Understanding the model {#Understanding-the-model}

[SDM.explain](@ref) offers tools to estimate the contribution and response curves for each variable. Currently, the only implemented method is Shapley values from the [Shapley.jl](www.gitlab.com/ExpandingMan/Shapley.jl) package.

```@example test
expl = SDM.explain(ensemble; method = ShapleyValues(8))
variable_importance(expl)
```


We can also interactively plot the model explanation to get response curves.

```@example test
interactive_response_curves(expl)
```


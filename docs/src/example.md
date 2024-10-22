# Example of a full species distribution modelling workflow
This example fits and projects species distribution models for the tree species _Eucaplytus regnans_ from scratch.

## Environmental data
We start by downloading environmental data. We use the [Rasters](www.github.com/rafaqz/Rasters.jl) package to handle raster data, and RasterDataSources.jl to automatically download Worldclim data.

In this example, we just download bioclimatic variables 1 and 12 (annual mean temperature and annual mean precipitation). We then mask to the country borders of Australia.

```@example test
using Rasters, RasterDataSources, ArchGDAL, NaturalEarth, DataFrames
bio = RasterStack(WorldClim{BioClim}, (1,12))
countries = naturalearth("ne_10m_admin_0_countries") |> DataFrame
australia = subset(countries, :NAME => ByRow(==("Australia"))).geometry
bio_aus = Rasters.trim(mask(bio; with = australia)[X = 110 .. 156, Y = -45 .. -10])
```

Let's plot this data to see what it looks like.

```@example test
using CairoMakie
Rasters.rplot(bio_aus)
```

## Occurrence data
Next, we use [GBIF2.jl](www.github.com/rafaqz/GBIF2.jl) to download occurrence records for this species. We use the [thin](@ref) function in this package to weed out occurrences that are very close to each other, using a cut-off of 5km.

```@example test
using GBIF2, SpeciesDistributionModels
sp = species_match("Eucalyptus regnans")
occurrences_raw = occurrence_search(sp; year = (1970,2000), country = "AU", hasCoordinate = true, limit = 2000)
occurrences = thin(occurrences_raw.geometry, 5000)
```

## Background points
Next, we sample random points to use as background points.

From Rasters 0.12.1 on, this will be easier with Rasters.sample! 

Let's plot both the occurrence and background points to see where _Eucalyptus regnans_ is found.

```@example test
using StatsBase
bg_indices = sample(findall(boolmask(bio_aus)), 500)
bg_points = DimPoints(bio_aus)[bg_indices]
fig, ax, pl = plot(bio_aus.bio1)
scatter!(ax, occurrences; color = :red)
scatter!(ax, bg_points; color = :grey)
fig
```

## Handling data
SpeciesDistributionModels.jl has a [sdmdata](@ref) function to handle input data. It takes tabular presence and background data as inputs, such as what is returned by `Rasters.extract` and `Rasters.sample`.

```@example test
using SpeciesDistributionModels
p_data = extract(bio_aus, occurrences; skipmissing = true)
bg_data = bio_aus[bg_indices]
data = sdmdata(p_data, bg_data; resampler = CV(nfolds = 3))
```

## Fitting an ensemble
Now that we have our `data` object with presence and background data, we can fit our ensemble. The `sdm` function fits a whole ensemble, taking two arguments: a data object and a `NamedTuple` with models the ensemble should have. This can be any MLJ-compatible model. In this case, we use Maxnet, boosted regression trees (from the EvoTrees.jl package), and a GLM.

```@example test
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

## Evaluating an ensemble
We can evaluate the entire ensemble using any metric from [StatisticalMeasures.jl](https://github.com/JuliaAI/StatisticalMeasures.jl).

```@example test
import SpeciesDistributionModels as SDM
ev = SDM.evaluate(ensemble; measures = (; auc, accuracy))
```

## Predicting
Next, we the climatic suitability of the species throughout Australia using `SpeciesDistributionModels.predict`. We can specify a `reducer` argument to get a single value, instead of a prediction for each member in the ensemble.

```@example test
pred = SDM.predict(ensemble, bio_aus; reducer = mean)
plot(pred; colorrange = (0,1))
```

## Understanding the model
[SDM.explain](@ref) offers tools to estimate the contribution and response curves for each variable. Currently, the only implemented method is Shapley values from the [Shapley.jl](www.gitlab.com/ExpandingMan/Shapley.jl) package.

```@example test
expl = SDM.explain(ensemble; method = ShapleyValues(8))
variable_importance(expl)
```

We can also interactively plot the model explanation to get response curves.
```@example test
interactive_response_curves(expl)
```
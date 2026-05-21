


## Index {#Index}
- [`SpeciesDistributionModels.SDMensemble`](#SpeciesDistributionModels.SDMensemble)
- [`SpeciesDistributionModels.SDMevaluation`](#SpeciesDistributionModels.SDMevaluation)
- [`SpeciesDistributionModels.SDMexplainMethod`](#SpeciesDistributionModels.SDMexplainMethod)
- [`SpeciesDistributionModels.SDMexplanation`](#SpeciesDistributionModels.SDMexplanation)
- [`SpeciesDistributionModels.ShapleyValues`](#SpeciesDistributionModels.ShapleyValues)
- [`SpeciesDistributionModels.evaluate`](#SpeciesDistributionModels.evaluate-Tuple{Any})
- [`SpeciesDistributionModels.explain`](#SpeciesDistributionModels.explain-Tuple{SDMensemble})
- [`SpeciesDistributionModels.predict`](#SpeciesDistributionModels.predict-Tuple{SDMensemble,%20Any})
- [`SpeciesDistributionModels.remove_collinear`](#SpeciesDistributionModels.remove_collinear-Tuple{Any})
- [`SpeciesDistributionModels.sdm`](#SpeciesDistributionModels.sdm-Tuple{Any,%20Any})
- [`SpeciesDistributionModels.sdmdata`](#SpeciesDistributionModels.sdmdata-Tuple{Any,%20Any})
- [`SpeciesDistributionModels.thin`](#SpeciesDistributionModels.thin-Tuple{Any,%20Any})


## Reference - Exported functions {#Reference-Exported-functions}
<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.SDMensemble' href='#SpeciesDistributionModels.SDMensemble'><span class="jlbinding">SpeciesDistributionModels.SDMensemble</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



SDMensemble &lt;: AbstractDimArray

An ensemble of fitted `MLJ` `Machine` objects, returned by the [sdm](/api#SpeciesDistributionModels.sdm-Tuple{Any,%20Any}) function. The ensemble is stored as a `DimArray` with `:model` and `:fold` dimensions, where each element is a fitted `Machine`. The ensemble's metadata includes the original `SDMdata` used to fit the models; use `sdmdata(ensemble)` to access it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/dimtypes.jl#L11-L15" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.SDMevaluation' href='#SpeciesDistributionModels.SDMevaluation'><span class="jlbinding">SpeciesDistributionModels.SDMevaluation</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



SDMevaluation &lt;: AbstractDimStack

An object containing evaluation results produced from an `SDMensemble`, returned by `SDM.evaluate`. The stack preserves the ensemble's `:model` and `:fold` dimensions and additionally has a `:dataset` dimension with values `:train`, `:test`, and optionally `:validation`, as well as a `:measure` dimension with the names of the evaluation measures used. An `SDMevaluation` has two layers `:score` (the value of the evaluation measure) and `:threshold` (the threshold at which the measure is optimal, for threshold-dependent measures; otherwise `missing`).

Use `sdm(ev)` to retrieve the originating ensemble and `sdmdata(ev)` for the associated training data.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/dimtypes.jl#L34-L41" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.SDMexplanation' href='#SpeciesDistributionModels.SDMexplanation'><span class="jlbinding">SpeciesDistributionModels.SDMexplanation</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



SDMexplanation &lt;: AbstractDimStack

An object that stores model explanation outputs (for example per-feature Shapley values). The stack retains the originating ensemble's `:model` and `:fold` dimensions. The layers correspond to predictor variables the ensemble is trained on.

The `method` field records which explanation algorithm was used.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/dimtypes.jl#L56-L62" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.ShapleyValues' href='#SpeciesDistributionModels.ShapleyValues'><span class="jlbinding">SpeciesDistributionModels.ShapleyValues</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ShapleyValues(algorithm::Shapley.Algorithm)
ShapleyValues(N::Integer; threaded = true, rng = Random.GLOBAL_RNG)
```


Use to specify use Shapley values as method in [`explain`](/api#SpeciesDistributionModels.explain-Tuple{SDMensemble}). If an integer `N`, and optionally `threaded` and `rng` is supplied, `MonteCarlo` sampling is used, where `N` is the number of iterations (samples). More samples will result in more accurate results,  but will take more time to compute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/explain/shapley.jl#L1-L9" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.explain-Tuple{SDMensemble}' href='#SpeciesDistributionModels.explain-Tuple{SDMensemble}'><span class="jlbinding">SpeciesDistributionModels.explain</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
explain(ensemble::SDMensemble; method, [data], [predictors])
```


Generate response curves for `ensemble`.

**Keywords**
- `method` is the algorithm to use. See ShapleyValues
  
- `data` is the data to use to generate response curves, and defaults to the data used to train the ensemble
  
- `predictors`: which predictors to generate response curves for. Defaults to all variables in `data`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/interface.jl#L115-L125" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.predict-Tuple{SDMensemble, Any}' href='#SpeciesDistributionModels.predict-Tuple{SDMensemble, Any}'><span class="jlbinding">SpeciesDistributionModels.predict</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
predict(SDMobject, newdata; clamp = false, threaded = false)
```


Use an `SDMensemble` to predict habitat suitability for some data. Predictions preserve the ensemble's `:model` and `:fold` dimensions, so the returned result carries the same dimensional metadata as the input.

`newdata` can be either a `RasterStack`, or a Tables.jl-compatible object. It must have all predictor variables used to train the models in its columns (or layers in case of a RasterStack).

**Keywords**
- `clamp`: if `true`, the predictions are clamped to the interval seen during training of `SDMobject`. Defaults to `false`
  
- `threaded`: if `true`, run multithreaded. Defaults to `true`.
  
- `reducer`: optionally provide a `Function` to summarize the output. The function should take a vector of values and return a single value. Typical examples are `Statistics.mean` or `Statistics.median`.
  

**Returns**

If `newdata` is a `RasterStack`, `predict` returns a `Raster`; otherwise, it returns a `DimArray`. In both cases the result preserves the `:model` and `:fold` dimensions of the input ensemble. Habitat suitability represented by a floating-point number between 0 and 1.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/interface.jl#L130-L146" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.remove_collinear-Tuple{Any}' href='#SpeciesDistributionModels.remove_collinear-Tuple{Any}'><span class="jlbinding">SpeciesDistributionModels.remove_collinear</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
remove_collinear(data; method, silent = false)
```


Removes strongly correlated variables in `data`, until correlation is below a threshold specified in `method`.

`method` can currently be either `Gvif`, `Vif` or `Pearson`, which use GVIF, VIF, or Pearson's r, respectively. GVIF and VIF are similar method, but GVIF includes categorical variables whereas VIF ignores them.

To run without showing information about collinearity scores, set `silent = true`.

**Example**

```julia
julia> import SpeciesDistributionModels as SDM
julia> mydata = (a = 1:100, b = sqrt.(1:100), c = rand(100))
julia> SDM.remove_collinear(mydata; method = SDM.Vif(10))
[ Info: a has highest GVIF of 28.367942095054225
[ Info: Removing a, 2 variables remaining
[ Info: b has highest GVIF of 1.0077618445543057
[ Info: All variables are below threshold, returning remaining variables
(:b, :c)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/collinearity.jl#L21-L43" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.sdm-Tuple{Any, Any}' href='#SpeciesDistributionModels.sdm-Tuple{Any, Any}'><span class="jlbinding">SpeciesDistributionModels.sdm</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
sdm(data, models; [resampler], [predictors], [verbosity])
```


Construct an ensemble.

**Arguments**

`data`: an `SDMdata` object `models`: a `NamedTuple` with the models to be used in the ensemble.

**Keywords**
- `models`: a `Vector` of the models to be used in the ensemble. All models must be MLJ-supported Classifiers. 
  
- For a full list of supported models, see https://alan-turing-institute.github.io/MLJ.jl/stable/model_browser/#Classification
  
- `predictors`: a `Vector` of `Symbols` with the names of the predictor values to be used. By default, all pdf
  
- `verbosity`: an `Int` value that regulates how much information is printed.
  
- `cache`: is passed to `MLJBase.machine`. Specify cache=false to prioritize memory management over speed.
  
- `scitype_check_level`: is passed to `MLJBase.machine`. Specify scitype_check_level=0 to disable scitype checking.
  

**Example**

```julia
using SpeciesDistributionModels, Maxnet, MLJGLMInterface
mydata = sdmdata(presences, absences; resampler = CV(nfolds = 5))
models = (maxnet = MaxnetBinaryClassifier(), glm = LinearBinaryClassifier())
ensemble = sdm(mydata, models)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/interface.jl#L43-L67" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.sdmdata-Tuple{Any, Any}' href='#SpeciesDistributionModels.sdmdata-Tuple{Any, Any}'><span class="jlbinding">SpeciesDistributionModels.sdmdata</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
sdmdata(presences, absences; resampler, predictors)
sdmdata(X, y::BitVector; resampler, predictors)
```


Construct an `SDMdata` object from species `presences` and `absences`.  Alternatively, from a table with predictor variables `X` and a `BitVector` `y`, where `false` represents absence and `true` represents presence.

**Keywords**
- `resampler`: The resampling strategy to be used. Should be a `MLJBase.ResamplingStrategy`, or a `Vector` of `Tuple`s with the form `(train, test)`.    Defaults to `NoResampling()`. If `resampler` is a `CV`, `shuffle` is internally set to `true`.
  
- `predictors`: a `Tuple` of `Symbols` with the names of the predictor values to be used. By default, all predictor variables in `X`,  or all predictor variables in both `presences` and `absences` are used..
  

**Returns**

An `SDMdata` object containing the data provided. This object can be used to construct an `SDMensemble`.

**Example**

```julia
using Rasters, SpeciesDistributionModels
A = rand(10,10)
B = rand(10,10)
st = RasterStack((a=A, b=B), (X, Y); missingval=(a=missing,b=missing))

presence_points = [(1, 1), (2, 2), (3, 3), (4, 4)]
absence_points = [(5, 5), (6, 6), (7, 7), (8, 8)]

p = extract(st, presence_points)
a = extract(st, absence_points)

mydata = sdmdata(p, a; resampler = CV(nfolds = 2)) # 2-fold cross validation
mydata2 = sdmdata([p; a], [trues(4); falses(4)]; resampler = [([1,2],[5,6]), ([3,4], [7,8])]) # provide resampling rows
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/interface.jl#L1-L33" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.thin-Tuple{Any, Any}' href='#SpeciesDistributionModels.thin-Tuple{Any, Any}'><span class="jlbinding">SpeciesDistributionModels.thin</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



````julia
thin([rng], x, cutoff; distance = Haversine(), [geometrycolumn])

Thin spatial data by removing points that are closer than `cutoff` distance
to the nearest other point in the dataset.

## Arguments
- `rng`: a random number generator. The default is `Random.GLOBAL_RNG()`.
- `x`: an `AbstractVector` that iterates points, or a table with a `:geometry` column.
- `cutoff`: the distance threshold in units of `distance`.
## Keywords
- `distance`: the distance metric used to calculate distances between points. The default
is `Haversine()`, which uses the Haversine formula to calculate the distance between coordinates in meter units.
- `geometrycolumn`: the name of the column in the table that contains the points, if `x` is a table. Usually defaults to `:geometry`.

## Example
```jldoctest
using SpeciesDistributionModels, Distances
# a vector that iteratores points
geometries = [(0,0), (1,0), (0,0.000001)]
# thin to 1000 meters
thin(geometries, 1000)
# thin to 1 degree
thin(Xoshiro(123), geometries, 1; distance = Euclidean())

# output
2-element Vector{Tuple{Int64, Real}}:
(0, 0)
(1, 0)
```
````



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/thin.jl#L1-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Reference - Internal functions {#Reference-Internal-functions}
<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.SDMexplainMethod' href='#SpeciesDistributionModels.SDMexplainMethod'><span class="jlbinding">SpeciesDistributionModels.SDMexplainMethod</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



`SDMexplainMethod`

Abstract marker type for explanation method descriptors. Currently the only method implemented is [ShapleyValues](/api#SpeciesDistributionModels.ShapleyValues).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/dimtypes.jl#L49-L53" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.evaluate-Tuple{Any}' href='#SpeciesDistributionModels.evaluate-Tuple{Any}'><span class="jlbinding">SpeciesDistributionModels.evaluate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
evaluate(x; measures, train = true, test = true, [validation])
```


Evaluate `x`, which should be an `SDMensemble`, by applying the measures provided to the data used to build the ensemble, and return an evaluation object.

**Keywords**
- `measures` is a `NamedTuple` of measures. The keys are used to identify the measures.
  

This defaults to using auc, log_loss, and kappa. For threshold-dependent measures, the highest score as well as the threshold at which the highest scores is reached are reported. A list of measurse is available here: https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/#aliases. However, note that not all measures are useful.
- `train`: a `Bool` that indicates whether to evaluate on the training data. Defaults to `true`
  
- `test`: a `Bool` that indicates whether to evaluate on the test data.Defaults to `true`
  
- `validation`: optionally, provide a separate validation dataset. 
  

Validation data should be a `Tuple` with presences as the first field and absences as the second.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/d01b92382c1ad8d95156a5b300f4a0fdd492d791/src/interface.jl#L78-L92" target="_blank" rel="noreferrer">source</a></Badge>

</details>


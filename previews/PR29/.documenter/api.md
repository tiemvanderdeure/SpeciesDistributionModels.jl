


## Basic workflow {#Basic-workflow}
<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.sdmdata' href='#SpeciesDistributionModels.sdmdata'><span class="jlbinding">SpeciesDistributionModels.sdmdata</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sdmdata(s::Union{SDMensemble, SDMevaluation, SDMexplanation})
```


Return the `SDMdata` object associated with an ensemble, evaluation, or explanation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/dimtypes.jl#L115-L119" target="_blank" rel="noreferrer">source</a></Badge>



```julia
sdmdata(presences, absences; resampler, predictors)
sdmdata(X, y::BitVector; resampler, predictors)
```


Construct an `SDMdata` object from species `presences` and `absences`. Alternatively, from a table with predictor variables `X` and a `BitVector` `y`, where `false` represents absence and `true` represents presence.

**Keywords**
- `resampler`: The resampling strategy to be used. Should be a `MLJBase.ResamplingStrategy`,   or a `Vector` of `Tuple`s with the form `(train, test)`. Defaults to `NoResampling()`.   If `resampler` is a `CV`, `shuffle` is internally set to `true`.
  
- `predictors`: a `Tuple` of `Symbols` with the names of the predictor variables to be used.   By default, all predictor variables in `X`, or all predictor variables in both   `presences` and `absences` are used.
  

**Returns**

An `SDMdata` object containing the data provided. This object can be used to construct an [`SDMensemble`](/api#SpeciesDistributionModels.SDMensemble).

**Example**

```julia
using Rasters, SpeciesDistributionModels
A = rand(10,10)
B = rand(10,10)
st = RasterStack((a=A, b=B), (X, Y); missingval=missing)

presence_points = [(1, 1), (2, 2), (3, 3), (4, 4)]
absence_points = [(5, 5), (6, 6), (7, 7), (8, 8)]

p = extract(st, presence_points; skipmissing = true)
a = extract(st, absence_points; skipmissing = true)

mydata = sdmdata(p, a; resampler = CV(nfolds = 2)) # 2-fold cross validation
mydata2 = sdmdata([p; a], [trues(4); falses(4)]; predictors = (:a,))

# output

SDMdata object with 4 presence points and 4 absence points. 
 
Resampling: 
Data is divided into 1 folds using resampling strategy NoResampling().
┌──────┬─────────┬────────┐
│ fold │ # train │ # test │
├──────┼─────────┼────────┤
│    1 │       8 │      0 │
└──────┴─────────┴────────┘
Predictor variables: 
┌───────┬────────────┬─────────┐
│ names │ scitypes   │ types   │
├───────┼────────────┼─────────┤
│ a     │ Continuous │ Float64 │
└───────┴────────────┴─────────┘
Also contains geometry data
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/interface.jl#L1-L56" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.sdm' href='#SpeciesDistributionModels.sdm'><span class="jlbinding">SpeciesDistributionModels.sdm</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sdm(s::Union{SDMevaluation, SDMexplanation})
```


Return the originating `SDMensemble` from an evaluation or explanation object.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/dimtypes.jl#L108-L112" target="_blank" rel="noreferrer">source</a></Badge>



```julia
sdm(data, models; verbosity = 0, cache = true, scitype_check_level = 1, threaded = false)
```


Construct an [`SDMensemble`](/api#SpeciesDistributionModels.SDMensemble) from data and models.

**Arguments**
- `data`: an [`SDMdata`](/api#SpeciesDistributionModels.SDMdata) object
  
- `models`: a `NamedTuple` with the models to be used in the ensemble. All models must be   MLJ-supported Classifiers.
  

**Keywords**
- `verbosity`: an `Int` value that regulates how much information is printed. Defaults to `0`.
  
- `cache`: passed to `MLJBase.machine`. Set `cache=false` to prioritize memory management over speed.
  
- `scitype_check_level`: passed to `MLJBase.machine`. Set to `0` to disable scitype checking.
  
- `threaded`: if `true`, run multithreaded. Defaults to `false`.
  

**Returns**

An [`SDMensemble`](/api#SpeciesDistributionModels.SDMensemble) containing fitted models.

**Example**

```julia
using SpeciesDistributionModels
import Maxnet: MaxnetBinaryClassifier
import MLJGLMInterface: LinearBinaryClassifier
presences = (a = rand(100), b = rand(100)); background = (a = rand(100).^2, b = sqrt.(rand(100)))
mydata = sdmdata(presences, background; resampler = CV(nfolds = 5))
models = (maxnet = MaxnetBinaryClassifier(), glm = LinearBinaryClassifier())
ensemble = sdm(mydata, models)
```


See also: [`SDMensemble`](/api#SpeciesDistributionModels.SDMensemble), [`sdmdata`](/api#SpeciesDistributionModels.sdmdata), [`evaluate`](/api#SpeciesDistributionModels.evaluate).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/interface.jl#L66-L97" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.evaluate' href='#SpeciesDistributionModels.evaluate'><span class="jlbinding">SpeciesDistributionModels.evaluate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
evaluate(x; measures = (...), train = true, test = true, validation = ())
```


Evaluate `x`, which should be an [`SDMensemble`](/api#SpeciesDistributionModels.SDMensemble), by applying the measures provided to the data used to build the ensemble, and return an [`SDMevaluation`](/api#SpeciesDistributionModels.SDMevaluation) object.

**Keywords**
- `measures`: a `NamedTuple` of measures. The keys are used to identify the measures.   Defaults to `(accuracy = StatisticalMeasures.accuracy, auc = StatisticalMeasures.auc,   log_loss = StatisticalMeasures.log_loss, kappa = StatisticalMeasures.kappa)`.   For threshold-dependent measures, the highest score as well as the threshold at which   the highest score is reached are reported.   A list of measures is available here: https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/#aliases
  
- `train`: a `Bool` that indicates whether to evaluate on the training data. Defaults to `true`.
  
- `test`: a `Bool` that indicates whether to evaluate on the test data. Defaults to `true`.
  
- `validation`: optionally, provide a separate validation dataset as a `Tuple` with presences   as the first field and absences as the second.
  

**Returns**

An [`SDMevaluation`](/api#SpeciesDistributionModels.SDMevaluation) object containing evaluation scores.

**Example**

```julia
using SpeciesDistributionModels
mydata = sdmdata(presences, absences; resampler = CV(nfolds = 5))
models = (glm = LinearBinaryClassifier(),)
ensemble = sdm(mydata, models)
eval_results = evaluate(ensemble)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/interface.jl#L108-L137" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.predict' href='#SpeciesDistributionModels.predict'><span class="jlbinding">SpeciesDistributionModels.predict</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
predict(ensemble::SDMensemble, newdata; clamp = false, threaded = false, reducer = nothing, dims = DD.dims(ensemble))
```


Use an [`SDMensemble`](/api#SpeciesDistributionModels.SDMensemble) to predict habitat suitability for `newdata`.

Predictions preserve the ensemble's `:model` and `:fold` dimensions, so the returned result carries the same dimensional metadata as the input.

`newdata` can be either a `RasterStack`, or a Tables.jl-compatible object. It must have all predictor variables used to train the models in its columns (or layers in case of a `RasterStack`).

**Keywords**
- `clamp`: if `true`, the predictions are clamped to the interval seen during training.   Defaults to `false`.
  
- `threaded`: if `true`, run multithreaded. Defaults to `false`.
  
- `reducer`: optionally provide a `Function` to summarize the output. The function should   take a vector of values and return a single value. Typical examples are `Statistics.mean`   or `Statistics.median`.
  
- `dims`: the dimensions to preserve from the ensemble. Defaults to `DD.dims(ensemble)`.
  

**Returns**
- If `newdata` is a `RasterStack`, returns a `Raster`
  
- Otherwise, returns a `DimArray`
  

In both cases the result preserves the `:model` and `:fold` dimensions of the input ensemble. Predictions are habitat suitability values represented as floating-point numbers between 0 and 1.

**Example**

```julia
using SpeciesDistributionModels
using MLJGLMInterface: LinearBinaryClassifier
presences = (a = rand(100), b = rand(100))
background = (a = rand(100).^2, b = sqrt.(rand(100)))
mydata = sdmdata(presences, background; resampler = CV(; nfolds = 3))
models = (glm = LinearBinaryClassifier(),)
ensemble = sdm(mydata, models)
preds = predict(ensemble, presences)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/interface.jl#L185-L224" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.explain' href='#SpeciesDistributionModels.explain'><span class="jlbinding">SpeciesDistributionModels.explain</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
explain(ensemble::SDMensemble; method, data = sdmdata(ensemble).predictor, predictors = keys(data))
```


Generate explanations for `ensemble` using the specified method.

Currently supports Shapley value-based explanations via [`ShapleyValues`](/api#SpeciesDistributionModels.ShapleyValues).

**Keywords**
- `method`: the explanation algorithm to use. See [`ShapleyValues`](/api#SpeciesDistributionModels.ShapleyValues).
  
- `data`: the data to use for generating explanations. Defaults to the data used to train the ensemble.
  
- `predictors`: which predictors to generate explanations for. Defaults to all variables in `data`.
  

**Returns**

An [`SDMexplanation`](/api#SpeciesDistributionModels.SDMexplanation) object containing explanation values.

**Example**

```julia
using SpeciesDistributionModels, Shapley
expl = explain(ensemble; method = ShapleyValues(100))
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/interface.jl#L160-L180" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Utility {#Utility}
<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.variable_importance' href='#SpeciesDistributionModels.variable_importance'><span class="jlbinding">SpeciesDistributionModels.variable_importance</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
variable_importance(expl::SDMexplanation)
```


Compute variable importance from an [`SDMexplanation`](/api#SpeciesDistributionModels.SDMexplanation) object.

By default, variable importance is the absolute mean value for each variable across all Shapley values.

**Returns**

A `NamedTuple` with variable importance scores.

**Example**

```julia
using SpeciesDistributionModels, Shapley
ensemble = sdm(mydata, models) # see sdm documentation for an example
expl = explain(ensemble; method = ShapleyValues(100))
vi = variable_importance(expl)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/explain/explain.jl#L1-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.thin' href='#SpeciesDistributionModels.thin'><span class="jlbinding">SpeciesDistributionModels.thin</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
thin([rng], x, cutoff; distance = Haversine(), geometrycolumn = nothing)
```


Thin spatial data by removing points that are closer than `cutoff` distance to the nearest other point in the dataset.

**Arguments**
- `rng`: a random number generator. Defaults to `Random.GLOBAL_RNG()`.
  
- `x`: an `AbstractVector` that iterates points, or a table with a `:geometry` column.
  
- `cutoff`: the distance threshold in units of `distance`.
  

**Keywords**
- `distance`: the distance metric used to calculate distances between points. Defaults to   `Haversine()`, which uses the Haversine formula to calculate distance in meters.
  
- `geometrycolumn`: the name of the column in the table that contains the points, if `x` is   a table. Defaults to `:geometry`.
  

**Returns**
- If `x` is a table: a subset of the table with thinned rows
  
- If `x` is an `AbstractVector`: a vector of points with nearby points removed
  

**Example**

```julia
using SpeciesDistributionModels, Distances
# a vector that iterates points
geometries = [(0.0,0.0), (1.0,0.0), (0.0,0.01)]
# thin to 1000 meters
thin(geometries, 1000)
# thin to 1 degree
thin(geometries, 1; distance = Euclidean())

# output
2-element Vector{Tuple{Float64, Float64}}:
 (0.0, 0.0)
 (1.0, 0.0)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/thin.jl#L1-L37" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.remove_collinear' href='#SpeciesDistributionModels.remove_collinear'><span class="jlbinding">SpeciesDistributionModels.remove_collinear</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
remove_collinear(data; method, silent = false)
```


Remove strongly correlated variables in `data` until correlation is below a threshold specified in `method`.

**Arguments**
- `data`: a table-like object with predictor variables
  

**Keywords**
- `method`: the collinearity method to use. Can be `Gvif`, `Vif`, or `Pearson`, which use   GVIF, VIF, or Pearson's r, respectively.
  
- `silent`: if `true`, suppress informational messages. Defaults to `false`.
  

**Returns**

A tuple of `Symbol`s representing the variable names that remain after removing strongly correlated variables.

**Notes**

GVIF and VIF are similar methods, but GVIF includes categorical variables whereas VIF ignores them.

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



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/collinearity.jl#L21-L53" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Plotting {#Plotting}
<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.interactive_response_curves' href='#SpeciesDistributionModels.interactive_response_curves'><span class="jlbinding">SpeciesDistributionModels.interactive_response_curves</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
interactive_response_curves
```


Interactive response curve visualization. Requires Makie extension.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/extensions.jl#L8-L12" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.interactive_evaluation' href='#SpeciesDistributionModels.interactive_evaluation'><span class="jlbinding">SpeciesDistributionModels.interactive_evaluation</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
interactive_evaluation
```


Interactive evaluation visualization for SDM results. Requires Makie extension.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/extensions.jl#L1-L5" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.boxplot' href='#SpeciesDistributionModels.boxplot'><span class="jlbinding">SpeciesDistributionModels.boxplot</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
boxplot
```


Create a boxplot visualization. Requires Makie extension.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/extensions.jl#L15-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Types {#Types}
<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.SDMdata' href='#SpeciesDistributionModels.SDMdata'><span class="jlbinding">SpeciesDistributionModels.SDMdata</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SDMdata
A struct to hold data for species distribution modeling. Contains the predictor variables, 
    response variable, geometry (if available), and resampling information.

See [`sdmdata`](@ref).
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/data_utils.jl#L8-L14" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.SDMensemble' href='#SpeciesDistributionModels.SDMensemble'><span class="jlbinding">SpeciesDistributionModels.SDMensemble</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SDMensemble <: AbstractDimArray
```


An ensemble of fitted `MLJ` `Machine` objects, returned by the [`sdm`](/api#SpeciesDistributionModels.sdm) function.

The ensemble is stored as a `DimArray` with `:model` and `:fold` dimensions, where each  element is a fitted `Machine`. The ensemble's metadata includes the original `SDMdata`  used to fit the models; use `sdmdata(ensemble)` to access it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/dimtypes.jl#L11-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.SDMevaluation' href='#SpeciesDistributionModels.SDMevaluation'><span class="jlbinding">SpeciesDistributionModels.SDMevaluation</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SDMevaluation <: AbstractDimStack
```


An object containing evaluation results produced from an [`SDMensemble`](/api#SpeciesDistributionModels.SDMensemble),  returned by [`evaluate`](/api#SpeciesDistributionModels.evaluate).

The stack preserves the ensemble's `:model` and `:fold` dimensions and additionally has a `:dataset` dimension with values `:train`, `:test`, and optionally `:validation`, as well as a `:measure` dimension with the names of the evaluation measures used. 

An `SDMevaluation` has two layers:
- `:score`: the value of the evaluation measure
  
- `:threshold`: the threshold at which the measure is optimal (for threshold-dependent measures; otherwise `missing`)
  

Use `sdm(ev)` to retrieve the originating ensemble and `sdmdata(ev)` for the associated training data.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/dimtypes.jl#L38-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.SDMexplanation' href='#SpeciesDistributionModels.SDMexplanation'><span class="jlbinding">SpeciesDistributionModels.SDMexplanation</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SDMexplanation <: AbstractDimStack
```


An object that stores model explanation outputs (for example per-feature Shapley values).

The stack retains the originating ensemble's `:model` and `:fold` dimensions. The layers correspond to predictor variables the ensemble is trained on.

The `method` field records which explanation algorithm was used.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/dimtypes.jl#L70-L79" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.ShapleyValues' href='#SpeciesDistributionModels.ShapleyValues'><span class="jlbinding">SpeciesDistributionModels.ShapleyValues</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ShapleyValues
```


A method descriptor for computing Shapley values for model explanations.

Use to specify Shapley values as method in [`explain`](/api#SpeciesDistributionModels.explain).

**Constructors**

```julia
ShapleyValues(algorithm::Shapley.Algorithm)
ShapleyValues(N::Integer; threaded = false, rng = Random.GLOBAL_RNG)
```


If an integer `N` is provided, `MonteCarlo` sampling is used, where `N` is the number  of iterations (samples). More samples yield more accurate results but take more time.

**Keywords (for Integer constructor)**
- `threaded`: if `true`, use multithreading. Defaults to `false`.
  
- `rng`: random number generator. Defaults to `Random.GLOBAL_RNG`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/explain/shapley.jl#L1-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='SpeciesDistributionModels.SDMexplainMethod' href='#SpeciesDistributionModels.SDMexplainMethod'><span class="jlbinding">SpeciesDistributionModels.SDMexplainMethod</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SDMexplainMethod
```


Abstract marker type for explanation method descriptors. Currently the only method implemented is [`ShapleyValues`](/api#SpeciesDistributionModels.ShapleyValues).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/blob/150df8296ee3cf70c975999c7314fd5866520eb4/src/dimtypes.jl#L62-L67" target="_blank" rel="noreferrer">source</a></Badge>

</details>


"""
    sdmdata(presences, absences; resampler, predictors)
    sdmdata(X, y::BitVector; resampler, predictors)

Construct an `SDMdata` object from species `presences` and `absences`.
Alternatively, from a table with predictor variables `X` and a `BitVector` `y`,
where `false` represents absence and `true` represents presence.

## Keywords
- `resampler`: The resampling strategy to be used. Should be a `MLJBase.ResamplingStrategy`,
    or a `Vector` of `Tuple`s with the form `(train, test)`. Defaults to `NoResampling()`.
    If `resampler` is a `CV`, `shuffle` is internally set to `true`.
- `predictors`: a `Tuple` of `Symbols` with the names of the predictor variables to be used.
    By default, all predictor variables in `X`, or all predictor variables in both
    `presences` and `absences` are used.

## Returns
An `SDMdata` object containing the data provided. This object can be used to construct
an [`SDMensemble`](@ref).

## Example
```jldoctest
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
"""
function sdmdata(
    presences,
    absences;
    resampler = NoResampling(),#MLJBase.CV(; nfolds = 5, shuffle = true),
    predictors = nothing,
)
    _sdmdata(presences, absences, resampler, predictors)
end

"""
    sdm(data, models; verbosity = 0, cache = true, scitype_check_level = 1, threaded = false)

Construct an [`SDMensemble`](@ref) from data and models.

## Arguments
- `data`: an [`SDMdata`](@ref) object
- `models`: a `NamedTuple` with the models to be used in the ensemble. All models must be
    MLJ-supported Classifiers.

## Keywords
- `verbosity`: an `Int` value that regulates how much information is printed. Defaults to `0`.
- `cache`: passed to `MLJBase.machine`. Set `cache=false` to prioritize memory management over speed.
- `scitype_check_level`: passed to `MLJBase.machine`. Set to `0` to disable scitype checking.
- `threaded`: if `true`, run multithreaded. Defaults to `false`.

## Returns
An [`SDMensemble`](@ref) containing fitted models.

## Example
```julia
using SpeciesDistributionModels
import Maxnet: MaxnetBinaryClassifier
import MLJGLMInterface: LinearBinaryClassifier
presences = (a = rand(100), b = rand(100)); background = (a = rand(100).^2, b = sqrt.(rand(100)))
mydata = sdmdata(presences, background; resampler = CV(nfolds = 5))
models = (maxnet = MaxnetBinaryClassifier(), glm = LinearBinaryClassifier())
ensemble = sdm(mydata, models)
```

See also: [`SDMensemble`](@ref), [`sdmdata`](@ref), [`evaluate`](@ref).
"""
function sdm(
    data, models;
    verbosity = 0,
    cache = true,
    scitype_check_level = 1,
    threaded = false
)
    _sdm(data, models, verbosity, cache, scitype_check_level, threaded)
end

"""
    evaluate(x; measures = (...), train = true, test = true, validation = ())

Evaluate `x`, which should be an [`SDMensemble`](@ref), by applying the measures provided to
the data used to build the ensemble, and return an [`SDMevaluation`](@ref) object.

## Keywords
- `measures`: a `NamedTuple` of measures. The keys are used to identify the measures.
    Defaults to `(accuracy = StatisticalMeasures.accuracy, auc = StatisticalMeasures.auc,
    log_loss = StatisticalMeasures.log_loss, kappa = StatisticalMeasures.kappa)`.
    For threshold-dependent measures, the highest score as well as the threshold at which
    the highest score is reached are reported.
    A list of measures is available here: https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/#aliases
- `train`: a `Bool` that indicates whether to evaluate on the training data. Defaults to `true`.
- `test`: a `Bool` that indicates whether to evaluate on the test data. Defaults to `true`.
- `validation`: optionally, provide a separate validation dataset as a `Tuple` with presences
    as the first field and absences as the second.

## Returns
An [`SDMevaluation`](@ref) object containing evaluation scores.

## Example
```julia
using SpeciesDistributionModels
mydata = sdmdata(presences, absences; resampler = CV(nfolds = 5))
models = (glm = LinearBinaryClassifier(),)
ensemble = sdm(mydata, models)
eval_results = evaluate(ensemble)
```
"""
function evaluate( # Define this as an extension of MLJBase.evaluate??
    x;
    measures = (; 
        StatisticalMeasures.accuracy,
        StatisticalMeasures.auc, 
        StatisticalMeasures.log_loss, 
        StatisticalMeasures.kappa
    ),
    train = true,
    test = true,
    validation::Tuple = ()
)
    test || train || isempty(validation) || error("No data to test. Either test or train must be true, or validation data must be provided")
    if !isempty(validation)
        X, y = _predictor_response_from_presence_absence(validation[1],validation[2], predictorkeys(sdmdata(x)))
        validation = (X, y)
    end

    _evaluate(x, measures, train, test, validation)
end


"""
    explain(ensemble::SDMensemble; method, data = sdmdata(ensemble).predictor, predictors = keys(data))

Generate explanations for `ensemble` using the specified method.

Currently supports Shapley value-based explanations via [`ShapleyValues`](@ref).

## Keywords
- `method`: the explanation algorithm to use. See [`ShapleyValues`](@ref).
- `data`: the data to use for generating explanations. Defaults to the data used to train the ensemble.
- `predictors`: which predictors to generate explanations for. Defaults to all variables in `data`.

## Returns
An [`SDMexplanation`](@ref) object containing explanation values.

## Example
```julia
using SpeciesDistributionModels, Shapley
expl = explain(ensemble; method = ShapleyValues(100))
```
"""
function explain(e::SDMensemble; method, data = sdmdata(e).predictor, predictors = keys(data))
    _explain(e, method, data, predictors)
end

"""
    predict(ensemble::SDMensemble, newdata; clamp = false, threaded = false, reducer = nothing, dims = DD.dims(ensemble))

Use an [`SDMensemble`](@ref) to predict habitat suitability for `newdata`.

Predictions preserve the ensemble's `:model` and `:fold` dimensions, so the returned
result carries the same dimensional metadata as the input.

`newdata` can be either a `RasterStack`, or a Tables.jl-compatible object. It must have
all predictor variables used to train the models in its columns (or layers in case
of a `RasterStack`).

## Keywords
- `clamp`: if `true`, the predictions are clamped to the interval seen during training.
    Defaults to `false`.
- `threaded`: if `true`, run multithreaded. Defaults to `false`.
- `reducer`: optionally provide a `Function` to summarize the output. The function should
    take a vector of values and return a single value. Typical examples are `Statistics.mean`
    or `Statistics.median`.
- `dims`: the dimensions to preserve from the ensemble. Defaults to `DD.dims(ensemble)`.

## Returns
- If `newdata` is a `RasterStack`, returns a `Raster`
- Otherwise, returns a `DimArray`

In both cases the result preserves the `:model` and `:fold` dimensions of the input ensemble.
Predictions are habitat suitability values represented as floating-point numbers between 0 and 1.

## Example
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
"""
function predict(e::SDMensemble, d; clamp = false, threaded = false, reducer = nothing, dims = DD.dims(e))
    _reformat_and_predict(e, d, clamp, threaded, reducer, dims)
end



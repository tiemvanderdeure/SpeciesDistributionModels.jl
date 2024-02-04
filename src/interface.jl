"""
    sdm_ensemble(presences, absences; models, [resampler], [predictors], [verbosity])

Construct an ensemble with input data specified in `presences` and `absences`.

The first input argument is species presences and the second (pseudo-)absences. Both presence and absence data must be Tables-compatible (e.g., a `DataFrame`, a `Vector` of `NamedTuple`, but not an `Array`)

## Keywords
`models`: a `Vector` of the models to be used in the ensemble. All models must be MLJ-supported Classifiers. 
For a full list of supported models, see https://alan-turing-institute.github.io/MLJ.jl/stable/model_browser/#Classification
`resampler`: The resampling strategy to be used. Defaults to 5-fold cross validation.
`predictors`: a `Vector` of `Symbols` with the names of the predictor values to be used. By default, all pdf
`verbosity`: an `Int` value that regulates how much information is printed.

## Example

"""
function sdm_ensemble(
    presences,
    absences;
    models,
    resampler = MLJBase.CV(; nfolds = 5, shuffle = true),
    predictors = _get_predictor_names(presences, absences),
    verbosity = 0
)

    predictors = collect(predictors)

    # Check the predictor values are valid
    :geometry in [predictors] && error("Predictors cannot be called :geometry")
    Base.intersect(predictors, Tables.schema(presences).names) == predictors || 
        error("The presence data does not contain all predictors specified")
    Base.intersect(predictors, Tables.schema(absences).names) == predictors || 
        error("The absence data does not contain all predictors specified")

    _fit_sdm_ensemble(presences, absences, models, [resampler], predictors, verbosity)
end

"""
    evaluate(ensemble::SDMensemble; measures)

Evaluate an ensemble, by applying the measures provided to the data used to built an ensemble, and return an `SDMensembleEvaluation`.

`measures` is a `NamedTuple` of measures. The keys are used to identify the measures.
This defaults to using auc, log_loss, and kappa.

For threshold-dependent measures, the highest score as well as the threshold at which the highest scores is reached are reported.

A list of measurse is available here: https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/#aliases. However, note that not all measures are useful.
"""
function evaluate( # Define this as an extension of MLJBase.evaluate??
    ensemble,
    measures = (; 
        auc = StatisticalMeasures.auc, 
        log_loss = StatisticalMeasures.log_loss, 
        kappa = StatisticalMeasures.kappa
    )
)
    _evaluate_ensemble(ensemble, measures)
end


"""
    explain(ensemble::SDMensemble; method, [predictors])

Generate response curves for all or some of the predictors used in the ensemble.

`method` defaults to Shapley() and 

"""
function explain end

"""
    predict(SDMobject, newdata; [reducer], [by_group])

Use an SDMmachine, group, or ensemble, to predict habitat suitability for some data, optionally summarized for the entire ensemble, or for each `SDMgroup`.

`newdata` can be either a `RasterStack`, or some other data which must be compatible with Tables.jl. It must have all predictor variables used to train the models in its columns (or layers in case of a RasterStack).

Optionally provide a function to summarize the output as the `reducer` argument. This would typically be `Statistics.mean` or `Statistics.median`.
If `by_group` is set to `true`, the data is reduced for each `SDMgroup`, if it is set to `false` (the default), it reduced across the entire ensemble.

If `newdata` is a `RasterStack`, the `predict` returns a `Raster`; otherwise, it returns a `NamedTuple` of `Vectors`
Habitat suitability is always reported as a floating-point number between 0 and 1.
"""
function predict(m::SDMmachine, d)
    _reformat_and_predict(m, d)
end
function predict(g::SDMgroup, d; reducer = nothing)
    _reformat_and_predict(g, d, reducer)
end
function predict(e::SDMensemble, d; reducer = nothing, by_group = false)
    by_group && isnothing(reducer) && error("If by_group is true, reducer must be specified")
    _reformat_and_predict(e, d, reducer, by_group)
end




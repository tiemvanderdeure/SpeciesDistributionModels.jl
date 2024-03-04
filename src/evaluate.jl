struct SDMmachineEvaluation
    sdm_machine::SDMmachine
    measures::NamedTuple
    results::NamedTuple
end

struct SDMgroupEvaluation <: AbstractVector{SDMmachineEvaluation}
    machine_evaluations::Vector{SDMmachineEvaluation}
    group::SDMgroup
    measures
    results
end

struct SDMensembleEvaluation <: AbstractVector{SDMgroupEvaluation}
    group_evaluations::Vector{SDMgroupEvaluation}
    ensemble::SDMensemble
    measures
    results
end

ScoreType = NamedTuple{(:score, :threshold), Tuple{Float64, Union{Missing, Float64}}}

SDMevaluation = Union{SDMmachineEvaluation, SDMgroupEvaluation, SDMensembleEvaluation}
SDMgroupOrEnsembleEvaluation = Union{SDMgroupEvaluation, SDMensembleEvaluation}

# Basic operations on evaluate objects
Base.getindex(ensemble::SDMensembleEvaluation, i) = ensemble.group_evaluations[i]
Base.getindex(group::SDMgroupEvaluation, i) = group.machine_evaluations[i]

Base.size(ensemble::SDMensembleEvaluation) = Base.size(ensemble.group_evaluations)
Base.size(group::SDMgroupEvaluation) = Base.size(group.machine_evaluations)

evaluation_sets(macheval::SDMmachineEvaluation) = keys(macheval.results)
evaluation_sets(e::SDMgroupOrEnsembleEvaluation) = evaluation_sets(e[1])
"""
    machine_evaluations(eval)

    Get the scores for each machine in an evaluation, which can be either an 
    `SDMgroupEvaluation` or an `SDMensembleEvaluation`.

    The return type is a nested structure of `NamedTuple`s. 
    The `NamedTuple` returned has keys corresponding to the evaluation datasets (a subset of :train, :test, and :validation), 
    which each have keys corresponding to the measures specified in [`evaluate`](@ref).

    ## Example
    ```julia
    evaluation = SDM.evaluate(ensemble; measures = (; accuracy, auc))
    machine_aucs = SDM.machine_evaluations(evaluation).train.auc
    ```
"""
machine_evaluations

function machine_evaluations(groupeval::SDMgroupEvaluation)
    sets = evaluation_sets(groupeval)
    map(sets) do set
        map(keys(groupeval.measures)) do key
            map(groupeval) do e
                e.results[set][key].score
            end
        end |> NamedTuple{keys(groupeval.measures)}
    end |> NamedTuple{sets}
end
function machine_evaluations(ensembleeval::SDMensembleEvaluation)
    sets = keys(ensembleeval[1][1].results)
    map(sets) do set
        map(keys(ensembleeval.measures)) do key
            mapreduce(vcat, ensembleeval) do groupeval
                map(groupeval) do e
                    e.results[set][key].score
                end
            end
        end |> NamedTuple{keys(ensembleeval.measures)}
    end |> NamedTuple{sets}
end

## Show methods
function Base.show(io::IO, mime::MIME"text/plain", evaluation::SDMmachineEvaluation)
    println(io, "SDMmachineEvaluation")

    measures = collect(keys(evaluation.measures))
    sets = evaluation_sets(evaluation)
    scores = map(sets) do s
        round.(getfield.(collect(evaluation.results[s]), :score); digits = 2)
    end
    
    table_cols = hcat(measures, scores...)
    header = (["measure"; string.(sets)...])
    PrettyTables.pretty_table(io, table_cols; header = header)
end

function Base.show(io::IO, mime::MIME"text/plain", evaluation::SDMgroupEvaluation)
    measures = collect(keys(evaluation.measures))
    train_scores, test_scores = machine_evaluations(evaluation)
    folds = getfield.(evaluation.group, :fold)

    println(io, "$(typeof(evaluation)) with $(length(measures)) performance measures")

    println(io, "Testing data")
    PrettyTables.pretty_table(io, merge((; fold = folds),  test_scores))
    println(io, "Training data")
    PrettyTables.pretty_table(io, merge((; fold = folds),  train_scores))
end

function Base.show(io::IO, mime::MIME"text/plain", evaluation::SDMensembleEvaluation)
    measures = collect(keys(evaluation.measures))
    models = getfield.(evaluation.ensemble, :model_name)

    # get scores from each group
    scores = machine_evaluations.(evaluation)
    # get mean test and train from each group for each measure.
    # then invert to a namedtuple where measures are keys
    println(io, "$(typeof(evaluation)) with $(length(measures)) performance measures")

    for k in keys(scores[1])
        println(io, string(k))
        s = map(scores) do score
            map(Statistics.mean, getfield(score, k))
        end |> Tables.columntable
        PrettyTables.pretty_table(io, merge((; model = models),  s))
    end
end


## Core evaluator
# internal method to get a vector of scores from y_hats, ys, and a namedtuple of measures
function _evaluate(y_hat::MLJBase.UnivariateFiniteArray, y::CategoricalArrays.CategoricalArray, measures)
    kinds_of_proxy = map(StatisticalMeasuresBase.kind_of_proxy, measures)
    
    # if any are literal targets (threshold-dependent), compute the confusion matrices outside the loop
    if any(map(kind -> kind == StatisticalMeasures.LearnAPI.LiteralTarget(), kinds_of_proxy))
        scores = pdf.(y_hat, true)
        thresholds = unique(scores)
        conf_mats = _conf_mats_from_thresholds(scores, y, thresholds)
    else
        conf_mats = nothing
    end

    map(measures, kinds_of_proxy) do measure, kind
        # If the measures is threshold independent
        if kind == StatisticalMeasures.LearnAPI.Distribution()
            return ScoreType((score = measure(y_hat, y), threshold = missing))
        else # else the measure uses thresholds    
            # find the max value and corresponding threshold for measure
            all_scores = measure.(conf_mats)
            max_score = findmax(all_scores)
            return ScoreType((score = max_score[1], threshold = thresholds[max_score[2]]))
        end
    end
end

function _conf_mats_from_thresholds(scores, y, thresholds)
    levels = [false, true]
    # use the internal method to avoid constructing indexer every time
    indexer = StatisticalMeasures.LittleDict(levels[i] => i for i in eachindex(levels)) |> StatisticalMeasures.freeze
    broadcast(thresholds) do t
        y_ = boolean_categorical(scores .>= t)
        StatisticalMeasures.ConfusionMatrices._confmat(y_, y, indexer, levels, true)
    end    
end

#### Evaluate methods ####
_ev_predict(sdm_machine, data::AbstractVector{<:Integer}) = MLJBase.predict(sdm_machine.machine; rows = data)
_ev_predict(sdm_machine, data) = MLJBase.predict(sdm_machine, data)
# Evaluate a single SDMmachine
function _evaluate(sdm_machine::SDMmachine, measures::NamedTuple, train, test, validation)
    machdata = data(sdm_machine)
    # set up namedtuple with data/rows, throw out if nothing/false
    d_X = (;)
    d_y = (;)
    if train
        d_X = merge(d_X, (;train = Tables.subset(machdata.predictor, sdm_machine.train_rows)))
        d_y = merge(d_y, (;train = machdata.response[sdm_machine.train_rows]))
    end
    if test
        d_X = merge(d_X, (;test =  Tables.subset(machdata.predictor, sdm_machine.test_rows)))
        d_y = merge(d_y, (;test = machdata.response[sdm_machine.test_rows]))
    end
    if !isempty(validation)
        d_X = merge(d_X, (;validation = validation[1]))
        d_y = merge(d_y, (;validation = validation[2]))
    end

    results = map(d_X, d_y) do X, y
        y_hat = MLJBase.predict(sdm_machine.machine, X)
        _evaluate(y_hat, y, measures)
    end

    return SDMmachineEvaluation(sdm_machine, measures, results)
end

# Evaluate a group
function _evaluate(group::SDMgroup, measures, train, test, validation)
    machine_evaluations = map(m -> (_evaluate(m, measures, train, test, validation)), group)

    # average group prediction
    p = predict(group, data(group).predictor, reducer = Statistics.mean)
    y_hat = MLJBase.UnivariateFinite(boolean_categorical([false, true]), p, augment = true)

    y = data(group).response
    group_evaluation = _evaluate(y_hat, y, measures)

    return SDMgroupEvaluation(
        machine_evaluations,
        group,
        measures,
        group_evaluation
    )
end

function _evaluate(ensemble::SDMensemble, measures, train, test, validation)
    group_evaluations = map(m -> (_evaluate(m, measures, train, test, validation)), ensemble)

    # average ensemble prediction
    p = predict(ensemble, data(ensemble).predictor, reducer = Statistics.mean)
    y_hat = MLJBase.UnivariateFinite(boolean_categorical([false, true]), p, augment = true)
    y = data(ensemble).response
    ensemble_evaluation = _evaluate(y_hat, y, measures)
    
    return SDMensembleEvaluation(
        group_evaluations,
        ensemble,
        measures,
        ensemble_evaluation
    )
end
struct SDMmachineEvaluation
    sdm_machine::SDMmachine
    measures::NamedTuple
    results::NamedTuple
end

struct SDMgroupEvaluation <: AbstractVector{SDMmachineEvaluation}
    machine_evaluations::Vector{SDMmachineEvaluation}
    group::SDMgroup
end

struct SDMensembleEvaluation <: AbstractVector{SDMgroupEvaluation}
    group_evaluations::Vector{SDMgroupEvaluation}
    ensemble::SDMensemble
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

measures(macheval::SDMmachineEvaluation) = macheval.measures
measures(e::SDMgroupOrEnsembleEvaluation) = measures(e[1])
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
        map(keys(measures(groupeval))) do key
            map(groupeval) do e
                e.results[set][key].score
            end
        end |> NamedTuple{keys(measures(groupeval))}
    end |> NamedTuple{sets}
end
function machine_evaluations(ensembleeval::SDMensembleEvaluation)
    sets = keys(ensembleeval[1][1].results)
    map(sets) do set
        map(keys(measures(ensembleeval))) do key
            mapreduce(vcat, ensembleeval) do groupeval
                map(groupeval) do e
                    e.results[set][key].score
                end
            end
        end |> NamedTuple{keys(measures(ensembleeval))}
    end |> NamedTuple{sets}
end

## Show methods
function Base.show(io::IO, mime::MIME"text/plain", evaluation::SDMmachineEvaluation)
    println(io, "SDMmachineEvaluation")

    measure_names = collect(keys(measures(evaluation)))
    sets = evaluation_sets(evaluation)
    scores = map(sets) do s
        round.(getfield.(collect(evaluation.results[s]), :score); digits = 2)
    end
    
    table_cols = hcat(measure_names, scores...)
    header = (["measure"; string.(sets)...])
    PrettyTables.pretty_table(io, table_cols; header = header)
end

function Base.show(io::IO, mime::MIME"text/plain", evaluation::SDMgroupEvaluation)
    measure_names = collect(keys(measures(evaluation)))
    train_scores, test_scores = machine_evaluations(evaluation)
    folds = getfield.(evaluation.group, :fold)

    println(io, "$(typeof(evaluation)) with $(length(measure_names)) performance measures")

    println(io, "Testing data")
    PrettyTables.pretty_table(io, merge((; fold = folds),  test_scores))
    println(io, "Training data")
    PrettyTables.pretty_table(io, merge((; fold = folds),  train_scores))
end

function Base.show(io::IO, mime::MIME"text/plain", evaluation::SDMensembleEvaluation)
    measurekeys = collect(keys(measures(evaluation)))
    models = collect(modelkeys(evaluation.ensemble))

    # get scores from each group
    machine_scores = machine_evaluations.(evaluation)
    ensemble_scores = map(e -> map(e -> e.score, e), evaluation.results)
    scores = vcat(machine_scores, ensemble_scores)
    # get mean test and train from each group for each measure.
    # then invert to a namedtuple where measures are keys
    println(io, "$(typeof(evaluation)) with $(length(measurekeys)) performance measures")

    for k in keys(scores[1])
        println(io, string(k))
        s = map(scores) do score
            map(Statistics.mean, getfield(score, k))
        end |> Tables.columntable
        PrettyTables.pretty_table(io, merge((; model = [models; "ensemble"]),  s))
    end
end


## Core evaluator
# internal method to get a vector of scores from y_hats, ys, and a namedtuple of measures
function _evaluate(y_hat::MLJBase.UnivariateFiniteArray, y::AbstractArray{<:CategoricalArrays.CategoricalValue}, measures)
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
    # preallocate y_
    y_ = boolean_categorical(falses(size(scores)...)) 
    broadcast(thresholds) do t
        broadcast!(>=(t), y_, scores) 
        StatisticalMeasures.ConfusionMatrices._confmat(y_, y, indexer, levels, true)
    end    
end

#### Evaluate methods ####
_ev_predict(sdm_machine, data::AbstractVector{<:Integer}) = MLJBase.predict(sdm_machine.machine; rows = data)
_ev_predict(sdm_machine, data) = MLJBase.predict(sdm_machine, data)
# Evaluate a single SDMmachine

function _ev_ydata(sdm_machine, train, test, validation)
    machdata = data(sdm_machine)
    d_y = (;)
    if train 
        d_y = merge(d_y, (;train = view(machdata.response, _gettrainrows(machdata, sdm_machine.fold))))
    end
    if test
        d_y = merge(d_y, (;test = view(machdata.response, _gettestrows(machdata, sdm_machine.fold))))
    end
    if !isempty(validation)
        d_y = merge(d_y, (;validation = validation[2]))
    end
    return d_y
end

function _ev_predict(sdm_machine::SDMmachine, train, test, validation)
    machdata = data(sdm_machine)
    # set up namedtuple with data/rows, throw out if nothing/false
    d_y = (;)
    if train
        d_y = merge(d_y, (;train = MLJBase.predict(sdm_machine.machine; rows = _gettrainrows(machdata, sdm_machine.fold))))
    end
    if test
        d_y = merge(d_y, (; test = MLJBase.predict(sdm_machine.machine; rows = _gettestrows(machdata, sdm_machine.fold))))
    end
    if !isempty(validation)
        d_y = merge(d_y, (; validation = MLJBase.predict(sdm_machine.machine, validation[1])))
    end
    return d_y
end

function _evaluate(sdm_machine::SDMmachine, y_hats::NamedTuple, ys::NamedTuple, measures::NamedTuple)
    results = map(y_hats, ys) do y_hat, y
        _evaluate(y_hat, y, measures)
    end
    return SDMmachineEvaluation(sdm_machine, measures, results)
end

function _evaluate(sdm_machine::SDMmachine, measures::NamedTuple, train, test, validation)
    y_hat = _ev_predict(sdm_machine, train, test, validation)
    y = _ev_ydata(sdm_machine, train, test, validation)
    _evaluate(sdm_machine, y_hat, y, measures)
end

# Evaluate a group
function _evaluate(group::SDMgroup, measures, train::Bool, test::Bool, validation::Tuple)
    machine_evaluations = map(m -> (_evaluate(m, measures, train, test, validation)), group)

    return SDMgroupEvaluation(
        machine_evaluations,
        group
    )
end

function _evaluate(group::SDMgroup, yhat, y, measures::NamedTuple)
    machine_evaluations = map((m, yhat_, y_) -> _evaluate(m, yhat_, y_, measures), group, yhat, y)

    return SDMgroupEvaluation(
        machine_evaluations,
        group
    )
end



function _evaluate(ensemble::SDMensemble, measures, train, test, validation, reducer = Statistics.mean)
    # works only if all groups have the same number of folds, for now
    
    # get y_hats
    y_hats = map(g -> map(m -> _ev_predict(m, train, test, validation), g), ensemble)
    ys = map(m -> _ev_ydata(m, train, test, validation), first(ensemble))
    # this whole thing returns the reduced (usually mean) for each fold, across model, for test/train/validation
    y_hats_reduced = map(y_hats...) do y...
        map(y...) do y_...
            p = map(x -> pdf.(x, true), y_) # probabilities from multivariatefinites
            ps_red = map((p...) -> reducer(p), p...)
            MLJBase.UnivariateFinite(boolean_categorical([false, true]), ps_red, augment = true)
        end
    end

    group_evaluations = map((g, y_hat) -> (_evaluate(g, y_hat, ys, measures)), ensemble, y_hats)

    # average ensemble prediction
    ensemble_evaluation = [
        map((y_hat, y) -> _evaluate(y_hat, y, measures), y_hats_reduced[i], ys[i])
        for i in eachindex(y_hats_reduced) 
    ]

    # invert the structure so it becomes a namedtuple (train/test) of namedtuples (measures) containing vectors (folds)
    sets = keys(ys[1])
    ensemble_evaluation = map(sets) do set
        map(keys(measures)) do key
            map((:score, :threshold)) do s
                map(ensemble_evaluation) do e
                    e[set][key][s]
                end 
            end |> NamedTuple{(:score, :threshold)}
        end |> NamedTuple{keys(measures)}
    end |> NamedTuple{sets}
    
    return SDMensembleEvaluation(
        group_evaluations,
        ensemble,
        ensemble_evaluation
    )
end
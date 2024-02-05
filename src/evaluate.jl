struct SDMmachineEvaluation
    sdm_machine::SDMmachine
    measures
    results
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

SDMgroupOrEnsembleEvaluation = Union{SDMgroupEvaluation, SDMensembleEvaluation}

ScoreType = NamedTuple{(:score, :threshold), Tuple{Float64, Union{Missing, Float64}}}

Base.getindex(ensemble::SDMensembleEvaluation, i) = ensemble.group_evaluations[i]
Base.getindex(group::SDMgroupEvaluation, i) = group.machine_evaluations[i]

Base.size(ensemble::SDMensembleEvaluation) = Base.size(ensemble.group_evaluations)
Base.size(group::SDMgroupEvaluation) = Base.size(group.machine_evaluations)

function machine_evaluations(groupeval::SDMgroupEvaluation; mean = false)
    map((:train, :test)) do set
        map(keys(groupeval.measures)) do key
            r = map(groupeval) do e
                e.results[set][key].score
            end

            if mean
                Statistics.mean(r)
            else
                r
            end

        end |> NamedTuple{keys(groupeval.measures)}
    end |> NamedTuple{(:train, :test)}
end

function machine_evaluations(ensembleeval::SDMensembleEvaluation; mean = false)
    map((:train, :test)) do set
        map(keys(ensembleeval.measures)) do key
            r = mapreduce(vcat, ensembleeval) do groupeval
                map(groupeval) do e
                    e.results[set][key].score
                end
            end

            if mean
                Statistics.mean(r)
            else
                r
            end

        end |> NamedTuple{keys(ensembleeval.measures)}
    end |> NamedTuple{(:train, :test)}
end

## Show methods
function Base.show(io::IO, mime::MIME"text/plain", evaluation::SDMmachineEvaluation)
    println(io, "SDMmachineEvaluation")

    measures = collect(keys(evaluation.measures))
    train_scores = round.(getfield.(collect(evaluation.results.train), :score); digits = 2)
    test_scores = round.(getfield.(collect(evaluation.results.test), :score); digits = 2)

    table_cols = hcat(measures, train_scores, test_scores)
    header = (["measure", "train_score", "test_score"])
    PrettyTables.pretty_table(io, table_cols; header = header)
end

function Base.show(io::IO, mime::MIME"text/plain", evaluation::SDMgroupOrEnsembleEvaluation)
    measures = collect(keys(evaluation.measures))
    train_scores, test_scores = machine_evaluations(evaluation, mean = true)
    
    group_scores = map(measures) do key
        evaluation.results[key].score
    end

    println(io, "$(typeof(evaluation)) with $(length(measures)) performance measures")

    table_cols = hcat(measures, collect(group_scores), collect(train_scores), collect(test_scores))
    header = (["measure", "performance of avg", "avg. train performance", "avg. test performance"])
    PrettyTables.pretty_table(io, table_cols; header = header)
    
end

## Core evaluator
# internal method to get a vector of scores from y_hats, ys, and a namedtuple of measures
function _evaluate(y_hat::MLJBase.UnivariateFiniteArray, y::CategoricalArrays.CategoricalArray, measures)
    map(measures) do measure
        # If the measures is threshold independent
        if StatisticalMeasuresBase.kind_of_proxy(measure) == StatisticalMeasures.LearnAPI.Distribution()
            return ScoreType((score = measure(y_hat, y), threshold = missing))
        else # else the measure uses thresholds
            # first get all possible thresholded values
            scores = pdf.(y_hat, true)
            thresholds = unique(scores)
            thresholded_scores = map(t -> CategoricalArrays.categorical(scores .>= t, levels = [false, true]), thresholds)
            
            # find the max value and corresponding threshold for measure
            all_scores = measure.(thresholded_scores, Ref(y))
            max_score = findmax(all_scores)
            return ScoreType((score = max_score[1], threshold = thresholds[max_score[2]]))
        end
    end
end

# Evaluate a single SDMmachine
function _evaluate(sdm_machine::SDMmachine, measures::NamedTuple)
    results = map((train = sdm_machine.train_rows, test = sdm_machine.test_rows)) do rows
        y_hat = MLJBase.predict(sdm_machine.machine, rows = rows)
        y = data(sdm_machine).response[rows]
        _evaluate(y_hat, y, measures)
    end

    return SDMmachineEvaluation(sdm_machine, measures, results)
end

# Evaluate a group
function _evaluate(group::SDMgroup, measures)
    machine_evaluations = map(m -> (evaluate(m; measures = measures)), group)

    # average group prediction
    y_hat = mapreduce(+, machines(group)) do mach 
        MLJBase.predict(mach) 
    end / length(group)

    y = data(group).response
    group_evaluation = _evaluate(y_hat, y, measures)

    return SDMgroupEvaluation(
        machine_evaluations,
        group,
        measures,
        group_evaluation
    )
end

function _evaluate(ensemble::SDMensemble, measures)
    group_evaluations = map(m -> (evaluate(m; measures = measures)), ensemble)

    # average ensemble prediction
    y_hat = mapreduce(+, machines(ensemble)) do mach 
        MLJBase.predict(mach)
    end / n_machines(ensemble)    

    y = data(ensemble).response
    ensemble_evaluation = _evaluate(y_hat, y, measures)
    
    return SDMensembleEvaluation(
        group_evaluations,
        ensemble,
        measures,
        ensemble_evaluation
    )
end
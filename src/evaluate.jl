struct SDMmachineEvaluation
    ensemble::SDMensemble
    measures
    results
end

struct SDMgroupEvaluation
    ensemble::SDMensemble
    measures
    results
end

struct SDMensembleEvaluation
    groupevaluations::SDMgroupEvaluation
    ensemble::SDMensemble
    measures
    results
end

ScoreType = NamedTuple{(:score, :threshold), Tuple{Float64, Union{Missing, Float64}}}

function ensemble_auc(ensemble)
    mapreduce(vcat, keys(ensemble.models)) do key
        mean([model.auc for model in ensemble.trained_models if model.model_key == key])
    end
end

function auc(ensemble)
    mapreduce(vcat, keys(ensemble.models)) do key
        Statistics.mean([model.auc for model in ensemble.trained_models if model.model_key == key])
    end
end



function evaluate_ensemble(
    ensemble::SDMensemble, 
    measures = (; 
        auc = StatisticalMeasures.auc, 
        log_loss = StatisticalMeasures.log_loss, 
        kappa = StatisticalMeasures.kappa)
    )

    results = map((train = :train, test = :test)) do rows

        map(ensemble.trained_models) do model
            mach = model.machine

            y_hat = MLJBase.predict(mach, rows = model[rows])
            y = ensemble.data.response[model[rows]]
            
            map(measures) do measure
                # If the measures is threshold in dependent
                if StatisticalMeasures.StatisticalMeasuresBase.kind_of_proxy(measure) == StatisticalMeasures.LearnAPI.Distribution()
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
    end

    return SDMevaluation(ensemble, measurements, results)
end

function getmeasure(evaluation::SDMensembleEvaluation, measure = :auc; train_or_test = :test)
    mapreduce(vcat, evaluation.results[:test]) do result
        result[measure]
    end
end


using MLJBase
folds = MLJBase.train_test_pairs(MLJBase.CV(; nfolds = 6, shuffle = true), 1:400, rand(400)) ## get indices

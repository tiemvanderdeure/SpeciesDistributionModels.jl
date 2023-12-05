struct SDMevaluation
    ensemble::SDMensemble
    measures
    results
end

ScoreType = NamedTuple{(:score, :threshold), Tuple{Float64, Union{Missing, Float64}}}

function evaluate_ensemble(ensemble::SDMensemble, measures = (; auc, log_loss, kappa))

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

function getmeasure(evaluation::SDMevaluation, measure = :auc; train_or_test = :test)
    mapreduce(vcat, evaluation.results[:test]) do result
        result[measure]
    end
end


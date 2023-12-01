struct SDMshapley
    values::Vector{<:NamedTuple} # Contains all shap values for all models
    importances::Vector{<:NamedTuple} # Contais mean absolute shap for each variable for each model
    ensemble::SDMensemble
    summary
end

Base.size(shap::SDMshapley) = Base.size(shap.values)
Base.length(shap::SDMshapley) = Base.length(shap.values)

function Base.show(io::IO, mime::MIME"text/plain", shap::SDMshapley)
    println(io, "Shapley evaluation for SDM ensemble with $(Base.length(shap)) models")

    println(io, "Mean feature importance:")
    Base.show(io, mime, shap.summary)
end

function shap(ensemble; parallelism = Shapley.CPUThreads(), n_samples = 50)
    shapvalues = map(ensemble.trained_models) do model
        Shapley.shapley(
            x -> Float64.(MLJBase.pdf.(MLJBase.predict(model.machine, x), true)), # some ml models return float32s - where to handle this?
            Shapley.MonteCarlo(parallelism, n_samples), 
            ensemble.data.predictor
        )
    end

    importances = map(vals -> map(val -> mapreduce(abs, +, val) / Base.length(val), vals), shapvalues)

    summary = NamedTuple(var => mapreduce(x -> getfield(x, var), +, importances) / Base.length(importances) for var in ensemble.predictors)

    return SDMshapley(shapvalues, importances, ensemble, summary)
end




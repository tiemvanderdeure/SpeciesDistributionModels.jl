# Type definitions are in dimtypes.jl
function _sdm(
    data::SDMdata,
    models::NamedTuple, 
    verbosity::Int,
    cache, 
    scitype_check_level,
    threaded::Bool
)
    ensemble = _initialize_ensemble(data, models, cache, scitype_check_level)
    _fit!(ensemble, threaded; verbosity)
end

function _initialize_ensemble(data, models::NamedTuple, cache, scitype_check_level)
    # set up dimensions
    modeldim = DD.Categorical(collect(keys(models)); order = DD.Unordered()) |> Dim{:model}
    folddim = Dim{:fold}(1:length(data.traintestpairs))
    dims = (modeldim, folddim)

    # initialize the models
    machines = broadcast(DD.DimPoints(dims)) do (model, fold)
        MLJBase.machine(models[model], data.predictor, data.response; cache, scitype_check_level = scitype_check_level * (fold == 1))
    end |> DimArray{Machine}
    
    return SDMensemble(machines, data)    
end

function _fit!(e::SDMensemble, threaded::Bool; verbosity)
    @maybe_threads threaded for (m, d) in zip(e, DD.DimPoints(DD.dims(e, (:fold, :model))))
        MLJBase.fit!(m; rows = sdmdata(e).traintestpairs[d[1]][1], verbosity)
    end
    return e
end

# Utility function used in show
function models(ensemble::SDMensemble)
    if DD.hasdim(ensemble, :fold)
        models(view(ensemble, fold = 1))
    elseif DD.hasdim(ensemble, :model) 
        getfield.(ensemble, :model)
    else
        DD.DimVector([first(ensemble).model], DD.refdims(ensemble, :model))
    end
end
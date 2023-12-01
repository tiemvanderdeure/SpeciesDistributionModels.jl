function predict(ensemble::SDMensemble, data::NamedTuple)
    # take data based on variable names
    data_ = data[ensemble.predictors]

    map(machines(ensemble)) do mach
        # predict each machine and get the probability of true
        Float64.(MLJBase.predict(mach, data_).prob_given_ref[2])
    end
end

function predict(ensemble::SDMensemble, data::Rasters.RasterStack)
    preds = Tuple(ensemble.predictors)

    # Check dimensions match and variables exist
    data1 = data[first(preds)]
    dims1 = Rasters.dims(data1)
    if ~all(p -> Rasters.dims(data[p]) == dims1, preds) error("Dimensions of data do not match") end

    # Find missing values -- maybe add this as method to RasterStack?
    missings = falses(dims1)
    for l in data[preds]
        missings .|= l .=== Rasters.missingval(l)
    end

    # Take non-missing data and convert to namedtuple of vectors
    data_ = NamedTuple{preds}(map(p -> data[p][.~missings], preds))

    # Reformat data to named tuple of vectors
    @time data_ = NamedTuple{Tuple(ensemble.predictors)}([vec(data[pre]) for pre in preds])

    # Allocate Raster to save results
    outraster = Raster(fill(NaN, (dims1..., Rasters.Band(machine_keys(ensemble)))); missingval = NaN, crs = Rasters.crs(data1))

    for (i, mach) in enumerate(machines(ensemble))
        # predict each machine and get the probability of true
        @views outraster[Rasters.Band(i)][.~missings] .= MLJBase.predict(mach, data_).prob_given_ref[2]
    end

    return outraster
end

function predict(ensemble::SDMensemble, rows::Symbol)
    y_hat_y = map(ensemble.trained_models) do model
        y_hat = MLJBase.predict(model.machine, rows = model[rows])
        y = ensemble.data.response[model[rows]]
        return (;y_hat, y)
    end

    return NamedTuple{Tuple(machine_keys(ensemble))}(y_hat_y)
end
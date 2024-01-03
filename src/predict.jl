function _unsafe_predict(mach::SDMmachine, data)
    CategoricalDistributions.pdf.(MLJBase.predict(mach.machine, data), true)
end


function predict(mach::SDMmachine, d)
    data_ = Tables.columntable(d)[keys(data(mach).predictor)] 
    _unsafe_predict(mach, data_)
end

function predict(s::SDMgroupOrEnsemble, d)
    data_ = Tables.columntable(d)[keys(data(s).predictor)] 

    mapreduce(hcat, sdm_machines(s)) do mach
        _unsafe_predict(mach, data_)
    end
end

#= on pause until RasterStacks are compatible with Tables.jl
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
=#

# inernal convenience function to predict just train or test rows for each machine
function _predict(s::SDMgroupOrEnsemble, rows::Symbol)
    y_hat_y = map(sdm_machines(s)) do sdm_mach
        y_hat = MLJBase.predict(sdm_mach.machine, rows = sdm_mach[rows])
        y = data(s).response[sdm_mach[rows]]
        return (;y_hat, y)
    end

    return (y_hat_y)
end
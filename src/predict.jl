#### Helper functions ####

# Reformat data so that it can be used in predict. Different models use different data types
_reformat_data(m::SDMmachine, d) = MLJBase.reformat(m.machine.old_model, Tables.columntable(d)[keys(data(m).predictor)])[1]

#### _predict methods ####
# _predict uses already-reformatted data.
# _reformat_and_predict methods first reformat and then call _predict

# Machine-level _predict method. All other _predict methods eventually call this
function _predict(m::SDMmachine, data)
    # predict
    prediction = MLJBase.predict(m.machine.old_model, m.machine.fitresult, data)
    # convert to Floats
    MLJBase.pdf.(prediction, true)
end

function _reformat_and_predict(m::SDMmachine, data)
    _predict(m, _reformat_data(m, data))
end

## Group level _predict methods
function _predict(g::SDMgroup, data, ::Nothing)
    pr = map(m -> _predict(m, data), g)
    return NamedTuple{Tuple(machine_keys(g))}(pr)
end

function _predict(g::SDMgroup, data, reducer::Function)
    pr = map(m -> _predict(m, data), g)
    return map((d...) -> reducer(d), pr...)
end

function _reformat_and_predict(g::SDMgroup, data, reducer)
    _predict(g, _reformat_data(first(g), data), reducer)
end

# ensemble-level methods
# For ensemble, there are no _predict methods. Datas has to be reformatted for each group
function _reformat_and_predict(e::SDMensemble, data, ::Nothing, ::Bool)
    mapreduce(
        g -> _reformat_and_predict(g, data, nothing), 
        merge, 
        e
    )
end

function _reformat_and_predict(e::SDMensemble, data, reducer::Function, by_group::Bool)
    if by_group
        # pass the reducer to each group, then combine into a namedtuple
        group_pr = (map(g -> _reformat_and_predict(g, data, reducer), e))
        NamedTuple{Tuple(model_names(e))}(group_pr)
    else
        # predict without reducing, then apply the reducer
        pr = mapreduce(g -> _reformat_and_predict(g, data, nothing), merge, e)
        map((d...) -> reducer(d), pr...)
    end
end

# Dispatch on RasterStacks
_reformat_and_predict(e::SDMensemble, rs::Rasters.AbstractRasterStack, reducer::Function, by_group::Bool) = 
    _reformat_and_predict_raster(e, rs, reducer, by_group)
_reformat_and_predict(g::SDMgroup, rs::Rasters.AbstractRasterStack, reducer::Union{<:Function, <:Nothing}) = 
    _reformat_and_predict_raster(g, rs, reducer)
_reformat_and_predict(m::SDMmachine, rs::Rasters.AbstractRasterStack) =
    _reformat_and_predict_raster(m, rs)

function _reformat_and_predict_raster(s::Union{<:SDMensemble, SDMgroup, SDMmachine}, rs::Rasters.AbstractRasterStack, args...)
    missing_mask = _missingmask(rs)
    d = rs[missing_mask]
    pr =  _reformat_and_predict(s, d, args...)
    return _build_raster(missing_mask, pr)
end

# Build Raster with the models/machines in Band dimension
function _build_raster(missing_mask::Rasters.AbstractRaster, pr::NamedTuple)
    r_dims = (Rasters.dims(missing_mask)..., Band(collect(keys(pr))))
    T = eltype(first(pr)) # usually Float64, but could be something else depending on the reducer
    output = Raster(Array{Union{Missing, T}}(missing, size(r_dims)); dims = r_dims)
    for k in keys(pr)
        @views output[Band = Rasters.At(k)][missing_mask] .= pr[k]
    end
    return output
end

# Build Raster with no additional layer
function _build_raster(missing_mask::Rasters.AbstractRaster, pr::Vector)
    r_dims = Rasters.dims(missing_mask)
    T = eltype(pr) # usually Float64, but could be something else depending on the reducer
    output = Raster(Array{Union{Missing, T}}(missing, size(r_dims)), dims = r_dims)
    output[missing_mask] .= pr
    return output
end

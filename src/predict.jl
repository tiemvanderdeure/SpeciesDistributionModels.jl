#### Helper functions ####

# Reformat data so that it can be used in predict. Different models use different data types
function _reformat_data(m::SDMmachine, d, clamp::Bool)
    traindata = data(m).predictor
    newdata = Tables.columntable(d)[keys(traindata)]
    if clamp
        for k in keys(traindata)
            if !(MLJBase.scitype(traindata[k]) <: AbstractVector{<:MLJBase.Finite}) # if data is categorical, don't clamp
                newdata[k] .= Base.clamp.(newdata[k], Base.extrema(traindata[k])...)
            end
        end
    end

    return MLJBase.reformat(m.machine.old_model, newdata)[1]
end
#### _predict methods ####
# _predict uses already-reformatted data.
# _reformat_and_predict methods first reformat and then call _predict

# Machine-level _predict method. All other _predict methods eventually call this
function _predict(m::SDMmachine, data)
    # predict
    prediction = MLJBase.predict(m.machine.old_model, m.machine.fitresult, data)
    # convert to Floats
    return MLJBase.pdf.(prediction, true)
end

function _reformat_and_predict(m::SDMmachine, data, clamp)
    _predict(m, _reformat_data(m, data, clamp))
end

## Group level _predict methods
function _predict(g::SDMgroup, data, ::Nothing, resource::AbstractCPU)
    pr = _map(resource)(m -> _predict(m, data), g)
    return NamedTuple{Tuple(machine_keys(g))}(pr)
end

function _reduce(reducer::F, pr::AbstractVector{<:Vector{<:AbstractFloat}}) where F
    map((d...) -> reducer(d), pr...) 
end
function _predict(g::SDMgroup, data, reducer::Function, resource::AbstractCPU) 
    pr = _map(resource)(m -> _predict(m, data), g)
    return  map((d...) -> reducer(d), pr...)
end

function _reformat_and_predict(g::SDMgroup, data, clamp, reducer::F, resource::AbstractCPU) where F
    _predict(g, _reformat_data(first(g), data, clamp), reducer, resource)
end

# ensemble-level methods
# For ensemble, there are no _predict methods. Datas has to be reformatted for each group
function _reformat_and_predict(e::SDMensemble, data, clamp::Bool, ::Nothing, ::Bool, resource::AbstractCPU)
    pr = _map(resource)(
        g -> _reformat_and_predict(g, data, clamp, nothing, resource), 
        e
    )
    merge(pr...)
end

function _reformat_and_predict(e::SDMensemble, data, clamp::Bool, reducer::Function, by_group::Bool, resource::AbstractCPU)
    if by_group
        # pass the reducer to each group, then combine into a namedtuple
        group_pr = _map(resource)(g -> _reformat_and_predict(g, data, clamp::Bool, reducer, resource), e)
        NamedTuple{Tuple(model_names(e))}(group_pr)
    else
        # predict without reducing, then apply the reducer
        pr = _map(resource)(g -> _reformat_and_predict(g, data, clamp::Bool, nothing, resource), e)
        pr_m = merge(pr...)
        map((d...) -> reducer(d), pr_m...)
    end
end

# Dispatch on RasterStacks
_reformat_and_predict(e::SDMensemble, rs::Rasters.AbstractRasterStack, clamp::Bool, reducer::Function, by_group::Bool, resource::AbstractCPU) = 
    _reformat_and_predict_raster(e, rs, clamp, reducer, by_group, resource)
_reformat_and_predict(g::SDMgroup, rs::Rasters.AbstractRasterStack, clamp::Bool, reducer::Union{<:Function, <:Nothing}, resource::AbstractCPU) = 
    _reformat_and_predict_raster(g, rs, clamp, reducer, resource)
_reformat_and_predict(m::SDMmachine, rs::Rasters.AbstractRasterStack, clamp::Bool, resource::AbstractCPU) =
    _reformat_and_predict_raster(m, rs, clamp, resource)

function _reformat_and_predict_raster(s::Union{<:SDMensemble, SDMgroup, SDMmachine}, rs::Rasters.AbstractRasterStack, args...)
    missing_mask = Rasters.boolmask(rs)
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

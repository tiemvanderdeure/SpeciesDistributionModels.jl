function _reformat_and_predict(e, d, clamp, args...)
    Tables.istable(d) || throw(ArgumentError("data is a $(typeof(d)), wich is not a Tables.jl-compatible table"))
    data = _select_features(sdmdata(e), d)
    output = _allocate_prediction(data, DD.dims(d), DD.dims(e))
    _predict!(output, data, e, args...)
end

function _select_features(d::SDMdata, x)
    cols = Tables.Columns(x)
    colnames = Tables.columnnames(cols)
    for key in predictorkeys(d)
        key in colnames || throw(ArgumentError("data is missing predictor variable $key"))
    end
    return NamedTuple(K => Tables.getcolumn(cols, K) for K in predictorkeys(d))
end
function _select_features(d::SDMdata, rs::Rasters.AbstractRasterStack)
    layernames = Rasters.name(rs)
    for key in predictorkeys(d)
        key in layernames || throw(ArgumentError("data is missing predictor variable $key"))
    end
    return rs[predictorkeys(d)]
end

_allocate_prediction(data, ::Nothing, ::Nothing) = _allocate_prediction(data, (Dim{:row}(1:Tables.rowcount(data)),))
_allocate_prediction(data, ::Nothing, ensdims) = _allocate_prediction(data, (Dim{:row}(1:Tables.rowcount(data)), ensdims...))
_allocate_prediction(data, dims, ensdims::Nothing) = _allocate_prediction(data, dims)
_allocate_prediction(data, dims, ensdims) = _allocate_prediction(data, (dims..., ensdims...))
_allocate_prediction(data, dims::DD.DimTuple) = zeros(dims)
function _allocate_prediction(st::RA.AbstractRasterStack, dims::DD.DimTuple)
    data = fill!(RA.Raster{Union{Missing, Float64}}(undef, dims), missing) 
    DD.rebuild(first(DD.layers(st)); dims, data, name = DD.NoName())
end

function _predict!(output::RA.AbstractRaster, data::RA.AbstractRasterStack, e::SDMensemble, threaded::Bool)
    bm = Rasters.boolmask(data)
    outputslices = @views [output[I][bm] for I in DD.DimIndices(DD.dims(e))]
    _predict!(outputslices, data[bm], e, threaded)
    return output
end
function _predict!(output::RA.AbstractRaster, data::RA.AbstractRasterStack, e)
    bm = Rasters.boolmask(data)
    @views _predict!(output[bm], data[bm], e)
    return output
end
function _predict!(output::DimArray, data, e::SDMensemble, threaded::Bool)
    @maybe_threads threaded for I in DD.DimIndices(e)
        _predict!(view(output, I), data, e[I])
    end
    return output
end
function _predict!(output::AbstractArray, data, m)
    prediction = MLJBase.predict(m, data)
    vec(output) .= MLJBase.pdf.(prediction, true)
    return output
end
# view(output, I) can result in a 0-dimension array
_predict!(output::DimArray{<:AbstractArray, 0}, data, m) = _predict!(first(output), data, m)
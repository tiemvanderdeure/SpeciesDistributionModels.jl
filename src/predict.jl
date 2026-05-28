function _reformat_and_predict(e, d, clamp, threaded, reducer, dims)
    Tables.istable(d) || throw(ArgumentError("data is a $(typeof(d)), wich is not a Tables.jl-compatible table"))
    data = _select_features(sdmdata(e), d, clamp)
    output = _allocate_prediction(data, DD.dims(d), DD.dims(e))
    _predict!(output, data, e, threaded)
    return _maybe_reduce(output, reducer, dims)
end

_maybe_reduce(x, ::Nothing, dims) = x
_maybe_reduce(x, f, dims) = f(x; dims)

function _select_features(d::SDMdata, x, clamp)
    cols = Tables.Columns(x)
    colnames = Tables.columnnames(cols)
    return map(predictorkeys(d)) do key
        key in colnames || throw(ArgumentError("data is missing predictor variable $key"))
        col = Tables.getcolumn(cols, key)
        # maybe clamp - only mutate in place if table does not have column access
        clamp || return col
        Tables.columnaccess(x) ? clamp(col, extrema(d.predictor[key])...) : clamp!(col, extrema(d.predictor[key])...)
    end |> NamedTuple{predictorkeys(d)}
end
# Specialize for RasterStacks to avoid conversion to namedtuple of vectors
function _select_features(d::SDMdata, rs::Rasters.AbstractRasterStack, clamp)
    layernames = Rasters.name(rs)
    for key in predictorkeys(d)
        key in layernames || throw(ArgumentError("data is missing predictor variable $key"))
    end
    data = return rs[predictorkeys(d)]
    clamp || return data
    maplayers(rs) do layer
        clamp(layer, extrema(d.predictor[RasterName(layer)])...) # never mutate
    end
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
function _predict!(output::DD.AbstractDimArray, data, e::SDMensemble, threaded::Bool)
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
_predict!(output::DD.AbstractDimArray{<:AbstractArray, 0}, data, m) = _predict!(first(output), data, m)
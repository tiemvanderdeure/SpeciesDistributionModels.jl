struct SDMensemble{N,D,R,A<:AbstractArray{<:Machine}} <: DD.AbstractDimArray{Machine,N,D,A}
    data::A
    dims::D
    refdims::R
    name::Symbol
    sdmdata::SDMdata
end
function SDMensemble(machines::A, dims::D, refdims::R, name::Symbol, sdmdata::SDMdata) where {A<:AbstractArray{<:Any, N}, D, R} where N
    SDMensemble{N,D,R,A}(machines, dims, refdims, name, sdmdata) 
end
SDMensemble(machines::DD.DimArray{<:Machine}, sdmdata) =
    SDMensemble(parent(machines), DD.dims(machines), DD.refdims(machines), DD.name(machines), sdmdata)

    # TODO Maybe make a special metadata that stores sdmdata
DD.metadata(e::SDMensemble, args...) = DD.NoMetadata()
DimArray(A::SDMensemble) = DimArray(parent(A), DD.dims(A); refdims = DD.refdims(A), name = DD.name(A))
function DD.rebuild(A::SDMensemble; data, dims = DD.dims(A), refdims = DD.refdims(A), name = DD.name(A), kw...)
    if eltype(data) <: Machine
        SDMensemble(data, dims, refdims, name, A.sdmdata)
    else
        DD.rebuild(DimArray(A); data, dims, refdims, name, kw...)
    end
end

@inline function DD.rebuild(
    A::SDMensemble, data, dims::Tuple, refdims, name, metadata
) 
    SDMensemble(data, dims, refdims, name, A.sdmdata)
end

# easy access to the fields of the type
sdmdata(s::SDMensemble) = s.sdmdata
machines(s::SDMensemble) = parent(s)

function models(ensemble::SDMensemble)
    if DD.hasdim(ensemble, :fold)
        models(view(ensemble, fold = 1))
    elseif DD.hasdim(ensemble, :model) 
        getfield.(ensemble, :model)
    else
        DimArray([model(first(ensemble))], DD.refdims(ensemble, :model))
    end
end

_folddim(ensemble::SDMensemble) = DD.hasdim(ensemble, :fold) ? DD.dims(ensemble, :fold) : DD.refdims(ensemble, :fold)
_modeldim(ensemble::SDMensemble) = DD.hasdim(ensemble, :model) ? DD.dims(ensemble, :model) : DD.refdims(ensemble, :model)

## Show methods
function Base.show(io::IO, mime::MIME"text/plain", ensemble::SDMensemble)
    lines, blockwidth = DD.show_main(io, mime, ensemble)
    # Printing the array data is optional, subtypes can
    # show other things here instead.
    ds = displaysize(io)
    ctx = IOContext(io, :blockwidth => blockwidth, :displaysize => (ds[1] - lines, ds[2]))
    DD.show_after(ctx, mime, models(ensemble))
end

## Table interface
Tables.istable(::Type{SDMensemble}) = true
Tables.schema(ensemble::SDMensemble) = Tables.schema(info(ensemble))
Tables.rows(ensemble::SDMensemble) = Tables.rows(info(ensemble))
Tables.columns(ensemble::SDMensemble) = Tables.columns(info(ensemble))

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
    modeldim = Dim{:model}(collect(keys(models)))
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
const ENSEMBLEKEYS = (:machine, :model, :trainrows, :testrows)
const ENSEMBLETYPES = Tuple{<:Machine, <:MLJBase.Probabilistic, <:AbstractVector{<:Integer}, <:AbstractVector{<:Integer}}
const SDMMACHINETUPLETYPE = NamedTuple{ENSEMBLEKEYS, <:ENSEMBLETYPES}
const SDMENSEMBLESTACKTYPE{N} = NamedTuple{
    ENSEMBLEKEYS,
    <:Tuple{Array{<:Machine, N}, Vector{<:MLJBase.Probabilistic}, Vector{<:Vector{<:Integer}}, Vector{<:Vector{<:Integer}}}
}

struct SDMmachine
    machine::Machine
    model::Symbol
    fold::Int
    data::SDMdata
end

struct SDMensemble{N, D} <: DD.AbstractBasicDimArray{SDMmachine, N, D}
    machines::DD.DimArray{<:Machine, N, D}
    data::SDMdata

    function SDMensemble(machines::DD.DimArray{<:Machine, N, D}, data) where {N, D}
        new{N, D}(machines, data)
    end
end

SDMmachineOrEnsemble = Union{SDMmachine, SDMensemble}

# easy access to the fields of the type
sdmdata(s::SDMmachineOrEnsemble) = s.data
machines(s::SDMensemble) = s.machines

models(m::SDMmachine)::MLJBase.Probabilistic = m.machine.model
models(ensemble::SDMensemble{2}) = models(ensemble[fold = 1])
models(ensemble::SDMensemble{1, <:Tuple{<:Dim{:fold}}}) = DimArray([model(first(ensemble))], DD.refdims(ensemble, :model))
models(ensemble::SDMensemble{1, <:Tuple{<:Dim{:model}}}) = model.(ensemble)

"""
    model(s)
    Similar to `models(s)`, but returns a model object rather than 1-length Array, whenever appropriate.
"""
function model(s::SDMmachineOrEnsemble)
    m = models(s)
    m isa DimArray && length(m) == 1 ? first(m) : m
end

# for easier exploration
Base.propertynames(::SDMmachineOrEnsemble) = (ENSEMBLEKEYS..., :data)

# iterating though an ensemble returns machines
function _getindex(ensemble::SDMensemble, I...; kw...)
    obj = Base.getindex(machines(ensemble), I...; kw...)
    if obj isa DD.AbstractDimArray
        SDMensemble(obj, sdmdata(ensemble))
    elseif obj isa Machine
        # TODO: This would be much easier if refdims would be a Table column
        refdims = DD.refdims(ensemble)
        if isempty(refdims)
            model, fold = getindex(DD.DimPoints(DD.dims(ensemble)), I...; kw...)
        elseif refdims isa Tuple{Dim{:fold}}
            model = first(getindex(DD.DimPoints(DD.dims(ensemble)), I...; kw...))
            fold = first(DD.lookup(DD.dims(refdims, :fold)))
        elseif refdims isa Tuple{Dim{:model}}
            fold = first(getindex(DD.DimPoints(DD.dims(ensemble)), I...; kw...))
            model = first(DD.lookup(DD.dims(refdims, :model)))
        end

        SDMmachine(obj, model, fold, sdmdata(ensemble))
    end
end
# To disambiguate
Base.getindex(e::SDMensemble, I...; kw...) = _getindex(e, I...; kw...)
Base.getindex(e::SDMensemble, d1::DD.DimensionalIndices; kw...) = _getindex(e, d1; kw...)
Base.getindex(e::SDMensemble, d1::DD.DimIndices; kw...) = _getindex(e, d1; kw...)
Base.getindex(e::SDMensemble, d1::DD.DimensionalIndices, d2::DD.DimensionalIndices, D::DD.DimensionalIndices...; kw...) = _getindex(e, d1,d2, D...; kw...)

# indexing by symbol returns the modelDimIndices
Base.getindex(ensemble::SDMensemble, s::Symbol) = ensemble[model = DD.At(s)]
# to solve ambiguity

# Define dims.
DD.dims(e::SDMensemble, args...) = DD.dims(machines(e), args...)
DD.refdims(e::SDMensemble, args...) = DD.refdims(machines(e), args...)

## Show methods
function Base.show(io::IO, mime::MIME"text/plain", ensemble::SDMensemble)
    println(io, "Trained SDMensemble")
    nfolds = DD.hasdim(ensemble, :fold) ? length(DD.dims(ensemble, Dim{:fold})) : 1
    println(io, "Contains $(length(ensemble)) SDMmachines across $nfolds resampling folds\n")
    println(io, "Uses the following models:")   
    DD.print_array(io, mime, models(ensemble))
end

function Base.show(io::IO, mime::MIME"text/plain", mach::SDMmachine)
    println(io, "Trained SDMmachine\n")
    println(io, "Uses model type:")
    Base.show(io, mime, mach.machine.model)
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
    @maybe_threads threaded for mach in e
        _fit!(mach; verbosity)
    end
    return e
end
function _fit!(m::SDMmachine; kw...)
    MLJBase.fit!(m.machine; rows = m.data.traintestpairs[m.fold][1], kw...)
    return m
end
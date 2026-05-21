########## Define dimensional types and basic DD methods
struct SDMmetadata{X} <: DD.Lookups.AbstractMetadata{Nothing, X}
    sdmdata::SDMdata
    metadata::X
end
SDMmetadata(sdmdata::SDMdata) = SDMmetadata(sdmdata, NamedTuple())
sdmdata(m::SDMmetadata) = m.sdmdata
DD.val(m::SDMmetadata) = m.metadata

### SDMensemble
"""
SDMensemble <: AbstractDimArray

An ensemble of fitted `MLJ` `Machine` objects, returned by the [sdm](@ref) function. The ensemble is stored as a `DimArray` with `:model` and `:fold` dimensions, where each element is a fitted `Machine`. The ensemble's metadata includes the original `SDMdata` used to fit the models; use `sdmdata(ensemble)` to access it.
"""
struct SDMensemble{T<:Machine,N,D,A<:DD.AbstractDimArray{T,N,D}} <: DD.AbstractDimArray{T,N,D,A}
    parent::A
end
SDMensemble(machines::DD.DimArray{<:Machine}, sdmdata::SDMdata) =
    SDMensemble(DD.rebuild(machines; metadata = SDMmetadata(sdmdata, NamedTuple())))

function DD.rebuild(A::SDMensemble; kw...)
    newparent = DD.rebuild(parent(A); kw...)
    if eltype(newparent) <: Machine
        SDMensemble(newparent)
    else
        newparent
    end
end
DD.rebuild(A::SDMensemble, data, dims::Tuple=DD.dims(A), refdims=DD.refdims(A), name=DD.name(A)) =
     DD.rebuild(A; data, dims, refdims, name)

### SDMevaluation
"""
SDMevaluation <: AbstractDimStack

An object containing evaluation results produced from an `SDMensemble`, returned by `SDM.evaluate`.
The stack preserves the ensemble's `:model` and `:fold` dimensions and additionally has a `:dataset` dimension with values `:train`, `:test`, and optionally `:validation`, as well as a `:measure` dimension with the names of the evaluation measures used. An `SDMevaluation` has two layers `:score` (the value of the evaluation measure) and `:threshold` (the threshold at which the measure is optimal, for threshold-dependent measures; otherwise `missing`).

Use `sdm(ev)` to retrieve the originating ensemble and `sdmdata(ev)` for the associated training data.
"""
struct SDMevaluation{K,T,N,L,A<:DD.DimStack{K,T,N,L}} <: DD.AbstractDimStack{K,T,N,L}
    parent::A
    ensemble::SDMensemble
end

## SDMexplanation

"""
`SDMexplainMethod`

Abstract marker type for explanation method descriptors. Currently the only method implemented is [ShapleyValues](@ref).
"""
abstract type SDMexplainMethod end

"""
SDMexplanation <: AbstractDimStack

An object that stores model explanation outputs (for example per-feature Shapley values). The stack retains the originating ensemble's `:model` and `:fold` dimensions. The layers correspond to predictor variables the ensemble is trained on.
    
The `method` field records which explanation algorithm was used.
"""
struct SDMexplanation{K,T,N,L,A<:DD.DimStack{K,T,N,L}} <: DD.AbstractDimStack{K,T,N,L}
    parent::A
    ensemble::SDMensemble
    method::SDMexplainMethod
end

method(ex::SDMexplanation) = getfield(ex, :method)

##### Forward essential DD methods
SDMDimStackorArray = Union{SDMensemble, SDMevaluation, SDMexplanation}
SDMDimStack = Union{SDMevaluation, SDMexplanation}

# Define parent
DD.parent(ex::SDMDimStackorArray) = getfield(ex, :parent)

# Forward dimensional methods to parent
for f in [:data, :dims, :refdims, :metadata, :name]
    @eval begin
        DD.$(f)(s::SDMDimStackorArray) = DD.$(f)(getfield(s, :parent))
    end
end
for f in [:layerdims, :layermetadata]
    @eval begin
        DD.$(f)(s::SDMDimStack) = DD.$(f)(parent(s))
    end
end

# Accessors
sdm(s::SDMDimStack) = getfield(s, :ensemble)
sdmdata(s::SDMDimStack) = sdmdata(sdm(s))
sdmdata(s::SDMensemble) = sdmdata(DD.metadata(s))

# rebuild sliced for SDMDimStacks
function DD.rebuild(s::SDMDimStack; data, dims=DD.dims(s), refdims=DD.refdims(s),
    layerdims=DD.layerdims(s), metadata=DD.metadata(s), layermetadata=DD.layermetadata(s)
)
    newparent = DD.rebuild(parent(s); data, dims, refdims, layerdims, metadata, layermetadata)
    DD.rebuild(s; parent = newparent)
end

_rebuild(x::SDMevaluation; parent = parent(x), ensemble = sdm(x)) =
    SDMevaluation(parent, ensemble)
_rebuild(x::SDMexplanation; parent = parent(x), ensemble = sdm(x), method = method(x)) =
    SDMexplanation(parent, ensemble, method)



function DD.rebuildsliced(f::Function, s::SDMDimStack, layers::NamedTuple, I)
    layerdims = DD.unrolled_map(DD.basedims, layers)
    dims, refdims = DD.slicedims(f, s, I)
    seldims = map(DD.dims(s), I) do d, i
        DD.rebuild(d, i)
    end
    ensemble = f(sdm(s), DD.commondims(seldims, DD.dims(sdm(s))));
    st = DD.rebuild(parent(s); data=DD.unrolled_map(parent, layers), dims, refdims, layerdims)
    _rebuild(s; parent = st, ensemble)
end
function DD.rebuildsliced(f::Function, s::SDMDimStack, layers::Tuple, I)
    layerdims = NamedTuple{keys(s)}(unrolled_map(basedims, layers))
    dims, refdims = slicedims(f, s, I)
    ensemble = sdm(s)[I]
    st = rebuild(parent(s); data=unrolled_map(parent, layers), dims, refdims, layerdims)
    rebuild(s; parent = st, ensemble)
end

#### Show methods
function Base.show(io::IO, mime::MIME"text/plain", ensemble::SDMensemble)
    lines, blockwidth = DD.show_main(io, mime, getfield(ensemble, :parent))
    ds = displaysize(io)
    ctx = IOContext(io, :blockwidth => blockwidth, :displaysize => (ds[1] - lines, ds[2]))
    DD.show_after(ctx, mime, models(ensemble))
end

function Base.show(io::IO, mime::MIME"text/plain", ev::SDMevaluation)
    meanscores = Statistics.mean(ev.score, dims = :fold)[fold = 1]
    _, displaywidth = displaysize(io)
    blockwidth = displaywidth
    io = IOContext(io, :dim_brackets => false)
    println(io, "SDMevaluation with dimensions:")
    lines, blockwidth = DD.show_main(io, mime, parent(ev))
    println(io, "\n\nMean training performance:")
    DD.print_array(io, mime, meanscores[dataset = DD.At(:train)])
    println(io, "\n\nMean test performance:")
    DD.print_array(io, mime, meanscores[dataset = DD.At(:test)])
end

function Base.show(io::IO, mime::MIME"text/plain", ex::SDMexplanation)
    lines, blockwidth = DD.show_main(io, mime, ex)
    ds = displaysize(io)
    ctx = IOContext(io, :blockwidth => blockwidth, :displaysize => (ds[1] - lines, ds[2]))
    #DD.show_after(ctx, mime, method(ex))
end

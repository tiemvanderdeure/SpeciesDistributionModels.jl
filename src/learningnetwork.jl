using MLJBase

import EvoTrees: EvoTreeClassifier
import MLJModelInterface
MMI = MLJModelInterface
import Tables

mutable struct Reducer <: MMI.Probabilistic
    reducing_function::Function    
end

function MMI.fit(m::Reducer, verbosity::Int64, X, y) 
    (MMI.classes(y), nothing , nothing)
end
function MMI.predict(m::Reducer, classes, X)
    mat = MMI.matrix(X)
    y = map(row -> m.reducing_function(row), eachrow(mat))
    MMI.UnivariateFinite(classes, y, augment = true)
end

# Similar to FeatureSelector(), but also handles RasterStacks
mutable struct SdmDataProcessor
    features::Vector{Symbol}
end

SdmDataProcessor() = SdmDataProcessor([])

mutable struct SdmEnsemble{
    modelnames,
} <: ProbabilisticNetworkComposite
    models::Vector{Supervised}
    reducer::Probabilistic
    cache::Bool
    acceleration::MLJBase.AbstractResource
    function SdmEnsemble(
        modelnames,
        models,
        reducer,
        cache = false,
        acceleration = CPU1()
    )
        map(models) do m
            MLJBase.check_ismodel(m, spelling=true)
        end
        return new{modelnames}(
            models,
            Reducer(reducer),
            cache,
            acceleration
        )
    end
 end

function SdmEnsemble(; reducer = mean, cache = false, acceleration = CPU1(), named_models...)
    nt = NamedTuple(named_models)
    modelnames = keys(nt)
    models = collect(nt)
    SdmEnsemble(modelnames, models, reducer, cache, acceleration)
end

Base.propertynames(::SdmEnsemble{modelnames}) where modelnames =
 tuple(:reducer, :cache, :acceleration, modelnames...)

function Base.getproperty(stack::SdmEnsemble{modelnames}, name::Symbol) where modelnames
    for f = [:reducer, :cache, :acceleration]
        name === f && return getfield(stack, f)
    end
    models = getfield(stack, :models)
    for j in eachindex(modelnames)
        name === modelnames[j] && return models[j]
    end
    error("type Stack has no property $name")
end

function Base.setproperty!(stack::SdmEnsemble, f::Symbol, v::Function)
    if f === :reducer 
        return setfield!(stack, f, Reducer(v))
    else
        ty = fieldtype(typeof(x), f)
        val = v isa ty ? v : convert(ty, v)
        return setfield!(x, f, val)
    end
end

function setproperty!(x, f::Symbol, v)
    ty = fieldtype(typeof(x), f)
    val = v isa ty ? v : convert(ty, v)
    return setfield!(x, f, val)
end

pdf_true_transform(ŷ::Node) = node(ŷ -> pdf.(ŷ, true), ŷ)
# Creating a namedtuple node!
Base.NamedTuple{names}(args::AbstractNode...) where names = node((args...) -> NamedTuple{names}(args), args...)
Base.eachrow(X::AbstractNode) = node(Base.eachrow, X)


function MLJBase.prefit(ensemble::SdmEnsemble{modelnames}, verbosity::Int, X, y) where modelnames
    Xs = source(X)
    ys = source(y)

    reducer = machine(:reducer, Xs, ys, cache = ensemble.cache)

    Zpred = map(modelnames) do symbolic_model
        model = getproperty(ensemble, symbolic_model)
        mach = machine(symbolic_model, Xs, ys, cache=ensemble.cache)
        ypred = predict(mach, Xs)
        ypred_float = pdf_true_transform(ypred)#, typeof(model), target_scitype(model))
    end 
    Ztable = MLJBase.table(hcat(Zpred...))
    me = predict(reducer, Ztable)


    #Zpred = NamedTuple{modelnames}(Zpred...)
    rows = eachrow(Ztable)

    (;
     predict = me,
     acceleration=ensemble.acceleration,
     )
end

ensemble = SdmEnsemble(reducer = mean, tree = EvoTreeClassifier(), tree2 = EvoTreeClassifier())
X = (a = rand(10), b = rand(10))
y = categorical(rand(Bool, 10))
m = machine(ensemble, X, y)
fit!(m)
t = predict(m, X)


UnivariateFinite(t, MMI.classes(y), augment = true)

ensemble.reducer =  median# Reducer(median)

predict_joint
NamedTuple{modelnames}(t)

n = MLJBase.prefit(ensemble, 0, X, y).predict.args[1].args[1].args[1]

MLJBase.glb |> methods
myhcat(args::AbstractNode...) = node(hcat, args...)


NamedTuple{names}(itr) where {names} = NamedTuple{names}(Tuple(itr))


NamedTuple{modelnames}(Zprednode...)

nodecollect(Zprednode...)
nodecollect(X::AbstractNode...) = node(collect, X...)



@edit MLJBase.glb(MLJBase.prefit(ensemble, 0, X, y).predict)

Zprednode = (MLJBase.prefit(ensemble, 0, X, y).predict)
modelnames = (:tree,)
node(Zprednode -> NamedTuple{modelnames}(Zprednode), Zprednode)

namedtuplenode2((:tree,), (Zprednode,)...)

getmodels(ensemble)

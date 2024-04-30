import MLJBase, MLJModels, Tables, Rasters
import MLJModelInterface as MMI
import MLJBase: Node, node, source, Machine, machine

### SdmPreProcessor
# Similar to FeatureSelector(),  but also handles RasterStacks, and always ignores :geometry features
mutable struct SdmPreProcessor <: MLJBase.Unsupervised
    features::Vector{Symbol}
    clamp::Bool
end
function MMI.fit(transformer::SdmPreProcessor, verbosity::Int, X)
    all_features = Tables.schema(X).names

    if isempty(transformer.features)
        features = collect(all_features)
        filter!(feature -> feature != :geometry, features)
    else
        for feature in transformer.features
            feature in all_features ||
            throw(ArgumentError("Feature $feature is expliclitly selected, but not found in the training data"))
        end
        in(:geometry, transformer.features) &&
        throw(ArgumentError("Features cannot be called :geometry"))
        features = transformer.features |> collect
    end

    cols = Tables.columns(X)
    extrema = NamedTuple(feature => Base.extrema(Tables.getcolumn(cols, feature)) for feature in features)

    report = NamedTuple()

    return (features, extrema), nothing, report
end
function MMI.transform(p::SdmPreProcessor, (features, extrema), X)
    all(e -> e in Tables.schema(X).names, features) ||
    throw(ArgumentError("Supplied frame does not admit previously selected features."))

    if X isa Rasters.RasterStack
        rs = X[Tuple(features)]
        if p.clamp 
            rs = map(rs, extrema) do r, ex
                clamp.(r, ex...)
            end
        end
        return view(rs, Rasters.boolmask(rs))
    else
        cols = MMI.selectcols(X, features)
        if p.clamp 
            cols = map(cols, extrema) do c, ex
                clamp.(c, ex...)
            end
        end
        return cols
    end
end
MMI.clean!(transformer::SdmPreProcessor) = ""

### SdmPostProcessor
# Reduces and constructs a Raster
mutable struct SdmPostProcessor <: MLJBase.Unsupervised
    reducing_function::Function    
end
function MMI.fit(transformer::SdmPostProcessor, verbosity::Int)
    report = NamedTuple()
    fitresult = nothing
    return fitresult, nothing, report
end

function MMI.transform(transformer::SdmPostProcessor, _, X, indata)
    X_true = map(x -> x.prob_given_ref[2], X)
    if indata isa Base.SubArray #this is the case if the provided data is a RasterStack
        dims = Rasters.dims(indata.parent.parent) # indata.parent.parent is the original RasterStack
        y = Base.fill!(Raster{Union{Missing, Float64}}(undef, dims), missing)
        y_view = Base.view(y, Base.parentindices(indata)...)
        map!((cols...) -> Float64(transformer.reducing_function(cols)), y_view, X_true...)
    else
        y = map((cols...) -> transformer.reducing_function(cols), X_true...)
       # y = MLJBase.UnivariateFinite(categorical([false, true]), y, augment = true)
    end
    return y
end

MMI.clean!(transformer::SdmPostProcessor) = ""

### SdmModel - just a single model with a preprocessor and postprocessor
mutable struct SdmModel{
    modelname
} <: MLJBase.ProbabilisticNetworkComposite
    model::MLJBase.Supervised
    preprocessor::SdmPreProcessor
    postprocessor::SdmPostProcessor
end


### Ensemble
mutable struct SdmEnsemble{
    modelnames,
} <: MLJBase.ProbabilisticNetworkComposite
    models::Vector{MLJBase.Supervised}
    preprocessor::SdmPreProcessor
    postprocessor::SdmPostProcessor
    standardize::Bool
    cache::Bool
    acceleration::MLJBase.AbstractResource
end

## Constructors
 function SdmEnsemble(; features = Symbol[], reducing_function = mean, standardize = true, cache = false, acceleration = CPU1(), named_models...)
    nt = NamedTuple(named_models)
    modelnames = keys(nt)
    models = collect(nt)
    map(models) do m
        MLJBase.check_ismodel(m, spelling=true)
    end
    return SdmEnsemble{modelnames}(
        models,
        SdmPreProcessor(features, true),
        SdmPostProcessor(reducing_function),
        standardize,
        cache,
        acceleration
    )
end

Base.propertynames(::SdmEnsemble{modelnames}) where modelnames =
 tuple(:reducing_function, :features, :cache, :acceleration, :preprocessor, :postprocessor, modelnames...)

function Base.getproperty(stack::SdmEnsemble{modelnames}, name::Symbol) where modelnames
    if name in modelnames
        models = getfield(stack, :models)
        for j in eachindex(modelnames)
            name === modelnames[j] && return models[j]
        end
    elseif name === :features
        return getfield(getfield(stack, :preprocessor), name)
    elseif name === :reducing_function
        return getfield(getfield(stack, :postprocessor), name)
    else
        return getfield(stack, name)
    end
end

function Base.setproperty!(stack::SdmEnsemble, f::Symbol, v)
    if f === :features
        return setfield!(stack.preprocessor, f, v)
    elseif f === :reducing_function
        return setfield!(stack.postprocessor, f, v)
    else
        ty = fieldtype(typeof(stack), f)
        val = v isa ty ? v : convert(ty, v)
        return setfield!(stack, f, val)
    end
end

# Creating a namedtuple node!
#Base.NamedTuple{names}(args::NTuple{<:Any, Node}) where names = node((args...) -> NamedTuple{names}(args), args...)
nodenamedtuple(keys, values::NTuple{<:Any, Node}) = node((values...) -> NamedTuple{keys}(values), values...)
function MLJBase.prefit(ensemble::SdmEnsemble{modelnames}, verbosity::Int, X, y) where modelnames
    Xs = source(X)
    ys = source(y)

    preprocessor = machine(:preprocessor, Xs, cache = ensemble.cache)
    postprocessor = machine(:postprocessor, cache = ensemble.cache)

    t = MLJBase.transform(preprocessor, Xs)

    if ensemble.standardize
        standardizer = machine(MLJModels.Standardizer(), t, cache = ensemble.cache)
        tnew = MLJBase.transform(standardizer, t)
    else
        tnew = t
    end

    Zpred = map(modelnames) do symbolic_model
        mach = machine(symbolic_model, tnew, ys, cache=ensemble.cache)
        MLJBase.predict(mach, tnew)
    end 

    Ztable = nodenamedtuple(modelnames, Zpred)#MLJBase.table(hcat(Zpred...))
    y = MLJBase.transform(postprocessor, Ztable, t)

    return merge(
        NamedTuple{modelnames}(Zpred),

        (        Ztable = Ztable, predict = y, acceleration=ensemble.acceleration, preprocessor, postprocessor),
    )
end

## Hack the evaluate method like with a pipeline?
function MLJBase.evaluate!(m::Machine{<:SdmEnsemble}; kw...)
    pipe = MLJBase.Pipeline(
        m.model, 
        (y -> MLJBase.UnivariateFinite(MLJBase.categorical([false, true]), y, augment = true)), 
        prediction_type = :probabilistic)
    mach_ev = machine(pipe, m.args...)
    MLJBase.evaluate!(mach_ev; kw...)
end
const evaluationkeys = (:score, :threshold)
const ScoreType = NamedTuple{evaluationkeys, Tuple{Float64, Union{Missing, Float64}}}

struct SDMensembleEvaluation{T,N,L,D<:DD.DimStack} <: DD.AbstractDimStack{(:score, :threshold, :ensemble), T, N, L}
    data::D
end
function SDMensembleEvaluation(stack::DD.AbstractDimStack{(:score, :threshold)}, ensemble::SDMensemble)
    ds = DD.DimStack(
        (score = stack.score, threshold = stack.threshold, ensemble = ensemble),
        metadata = DD.metadata(ensemble)
    )
    SDMensembleEvaluation(ds, sdmdata(ensemble))
end
SDMensembleEvaluation(stack::DD.AbstractDimStack{(:score, :threshold, :ensemble), T, N, L}) where {T,N,L} =
    SDMensembleEvaluation{T,N,L, typeof(stack)}(stack)

Base.@constprop :aggressive Base.@propagate_inbounds function Base.getindex(ev::SDMensembleEvaluation, key::Symbol)
    if key === :ensemble
        SDMensemble(parent(ev)[key])
    else
        DD.DimArray(parent(ev)[key], DD.dims(ev), DD.refdims(ev), key, DD.NoMetadata())
    end
end

function DD.rebuild(ev::SDMensembleEvaluation; kw...)
    parent = DD.rebuild(parent(ev); kw...)
    if keys(parent) === (:score, :threshold, :ensemble)
        SDMensembleEvaluation(parent)
    else
        parent
    end
end

DD.parent(s::SDMensembleEvaluation) = getfield(s, :data)

sdmdata(ev::SDMensembleEvaluation) = metadata(ev).sdmdata

for f in [:data, :dims, :refdims, :metadata, :layerdims, :layermetadata]
    @eval begin
        DD.$(f)(ds::SDMensembleEvaluation) = DD.$(f)(parent(ds))
    end
end

function Base.show(io::IO, mime::MIME"text/plain", ev::SDMensembleEvaluation)
    meanscores = Statistics.mean(ev.score, dims = :fold)[fold = 1]
    _, displaywidth = displaysize(io)
    blockwidth = displaywidth
    io = IOContext(io, :dim_brackets => false)
    println(io, "SDMensembleEvaluation with dimensions:")
    lines, blockwidth = DD.show_main(io, mime, ev)
    #DD.print_dims_block(io, mime, DD.dims(ev); displaywidth, blockwidth)
    println(io, "\n\nMean training performance:")
    DD.print_array(io, mime, meanscores[dataset = DD.At(:train)])
    println(io, "\n\nMean test performance:")
    DD.print_array(io, mime, meanscores[dataset = DD.At(:test)])
end

## TODO: make this much smoother and more understandable code
function _getrows(data::SDMdata, set::Symbol, fold)
    if set == :train
        data.traintestpairs[fold][1]
    elseif set == :test
        data.traintestpairs[fold][2]
    end
end
function _getrows(ds::Tuple{Dim{:dataset}, Dim{:fold}}, ensemble::SDMensemble)
    data = sdmdata(ensemble)
    rows = broadcast(DD.DimPoints(ds)) do (t, f)
        _getrows(data, t, f)
    end
end
function _getrows(ds::Tuple{Dim{:dataset}}, ensemble::SDMensemble)
    data = sdmdata(ensemble)
    f = first(folds(ensemble))
    rows = broadcast(DD.DimPoints(ds)) do (t,)
        _getrows(data, t, f)
    end
end

function _evaluate(ensemble::SDMensemble, measures::NamedTuple, train::Bool, test::Bool, validation::Tuple)
    data = sdmdata(ensemble)
    measuredim = DD.Categorical(collect(keys(measures)); order = DD.Unordered()) |> Dim{:measure}
    datasets = Symbol[]
    train && push!(datasets, :train)
    test && push!(datasets, :test)
    isempty(validation) || push!(datasets, :validation)
    dataset_dim = DD.Categorical(datasets; order = DD.Unordered()) |> Dim{:dataset}

    alldims = (
        DD.dims(ensemble)..., 
        dataset_dim, 
        measuredim
    )

    # get a DimArray with row indices for each dataset and fold (if applicable)
    rows = _getrows(DD.dims(alldims, (:dataset, :fold)), ensemble)
    x = broadcast(rows) do r
        if isnothing(r) # this is the case for validation data only!
            validation[1]
        else 
            map(p -> p[r], data.predictor)
        end
    end
    y = broadcast(rows) do r 
        if isnothing(r)
            validation[2]
        else 
            getindex(data.response, r)
        end
    end

    predictions = DimArray{MLJBase.UnivariateFiniteVector}(undef, DD.dims(alldims, (:fold, :model, :dataset)))
    DD.broadcast_dims!(predictions, ensemble, x) do m, x
        MLJBase.predict(m, x)
    end

    # if any are literal targets (threshold-dependent), compute the confusion matrices outside the loop
    anyliteral = any(map(m -> StatisticalMeasuresBase.kind_of_proxy(m) isa StatisticalMeasures.LearnAPI.LiteralTarget, measures))
    thresholds_confmats = broadcast_dims(predictions, y) do p, y
        if anyliteral
            scores = pdf.(p, true)
            thresholds = unique(scores)
            (thresholds, _conf_mats_from_thresholds(scores, y, thresholds))
        else
            (nothing, nothing)
        end
    end

    # pre-allocate the evaluation stack - its layers are `scores` and `thresholds`
    evaluationstack = DimStack((score = zeros(alldims), threshold = DD.DimArray{Union{Missing, Float64}}(undef, alldims)))
    # evaluate - replace with a broadcast_dims! in the future?
    for I in DD.DimIndices(alldims)
        evaluationstack[I] = _apply_measure(
            predictions[DD.commondims(I, DD.dims(predictions))], 
            y[DD.commondims(I, DD.dims(y))], 
            thresholds_confmats[DD.commondims(I, DD.dims(thresholds_confmats))],
            measures[DD.dims(I, :measure).val]
        )
    end
    ev = DimStack(merge(evaluationstack, (; ensemble)), metadata = DD.metadata(ensemble))
    return SDMensembleEvaluation(ev)
end

function _apply_measure(y_hat::MLJBase.UnivariateFiniteVector, y::MLJBase.CategoricalVector, (thresholds, conf_mats), measure)
    if StatisticalMeasuresBase.kind_of_proxy(measure) isa StatisticalMeasures.LearnAPI.LiteralTarget
        # in this case the measure is threshold-dependent and we use the precomputed confusion matrices
        score, idx = findmax(measure, conf_mats)
        # return the maximum score and corresponding threshold
        return (score, thresholds[idx])
    else
        # in this case the measure is threshold-independent and we just compute the score
        # and return 'missing' for the threshold
        return (measure(y_hat, y), missing)
    end
end

## hacky way of computing confusion matrices that is much faster
function _conf_mats_from_thresholds(scores, y, thresholds)
    levels = [false, true]
    # use the internal method to avoid constructing indexer every time
    indexer = StatisticalMeasures.LittleDict(levels[i] => i for i in eachindex(levels)) |> StatisticalMeasures.freeze
    # preallocate y_
    y_ = boolean_categorical(falses(size(scores)...)) 
    broadcast(thresholds) do t
        broadcast!(>=(t), y_, scores) 
        StatisticalMeasures.ConfusionMatrices._confmat(y_, y, indexer, levels, true)
    end    
end

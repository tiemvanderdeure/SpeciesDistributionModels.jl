abstract type SDMexplainMethod end

struct SDMexplanation{K,T,N,L,A<:DD.DimStack,A2<:SDMensemble} <: DD.AbstractDimStack{K,T,N,L}
    data::A
    ensemble::A2
    method
end
function SDMexplanation(stack::DD.AbstractDimStack{K,T,N,L}, ensemble, method) where {K,T,N,L}
    SDMexplanation{K,T,N,L,typeof(stack), typeof(ensemble)}(stack, ensemble, method)
end

DD.parent(ex::SDMexplanation) = getfield(ex, :data)
sdm(ex::SDMexplanation) =  getfield(ex, :ensemble)
sdmdata(ex::SDMexplanation) = sdmdata(ensemble(ex))
method(ex::SDMexplanation) = getfield(ex, :method)

for f in [:data, :dims, :refdims, :metadata, :layerdims, :layermetadata]
    @eval begin
        DD.$(f)(ds::SDMexplanation) = DD.$(f)(parent(ds))
    end
end
# When rebuilding, make sure to make dims match for the ensemble
function DD.rebuild(ex::SDMexplanation; dims, refdims, kw...)
    parent = DD.rebuild(DD.parent(ex); dims, refdims, kw...)
    ensemble = sdm(ex)
    ensemble = isempty(refdims) ? ensemble : ensemble[first(DD.DimSelectors(refdims))]
    ensemble = dims === DD.dims(ensemble) ? ensemble : ensemble[DD.DimSelectors(dims)]
    SDMexplanation(parent, ensemble, method(ex))
end

## By default, variable importance is absolute mean for value for each variable in shapvalues
function variable_importance(expl::SDMexplanation)
    DD.maplayers(vals -> Statistics.mean(Statistics.mean.(abs, vals)), expl)
end

function _explain(ensemble::SDMensemble, method, d::NamedTuple{K}, predictors) where K
    expl = map(ensemble) do mach
        _explain(mach, method, d, predictors)
    end
    expl_stack = NamedTuple(k => getindex.(expl, k) for k in K) |> DimStack
    return SDMexplanation(expl_stack, ensemble, method)
end

#summary = NamedTuple(var => mapreduce(x -> getfield(x, var), +, importances) / Base.length(importances) for var in ensemble.predictors)



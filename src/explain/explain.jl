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

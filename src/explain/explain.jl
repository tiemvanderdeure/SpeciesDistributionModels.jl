"""
    variable_importance(expl::SDMexplanation)

Compute variable importance from an [`SDMexplanation`](@ref) object.

By default, variable importance is the absolute mean value for each variable across
all Shapley values.

## Returns
A `NamedTuple` with variable importance scores.

## Example
```julia
using SpeciesDistributionModels, Shapley
ensemble = sdm(mydata, models) # see sdm documentation for an example
expl = explain(ensemble; method = ShapleyValues(100))
vi = variable_importance(expl)
```
"""
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

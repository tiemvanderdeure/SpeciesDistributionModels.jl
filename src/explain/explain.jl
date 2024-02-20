abstract type SDMexplainMethod end

# Type definitions for explanation objects
struct SDMmachineExplanation
    machine::SDMmachine
    method::SDMexplainMethod
    values::NamedTuple # Contains values
    data::NamedTuple # Contains the data used to explain
end

struct SDMgroupExplanation <: AbstractVector{SDMmachineExplanation}
    group::SDMgroup
    machine_explanations::Vector{SDMmachineExplanation}
end

struct SDMensembleExplanation <: AbstractVector{SDMgroupExplanation}
    ensemble::SDMensemble
    group_explanations::Vector{SDMgroupExplanation}
end

#### Basic operations on Explanation objects ####
Base.getindex(e::SDMgroupExplanation, i::Integer) = e.machine_explanations[i]
Base.getindex(e::SDMensembleExplanation, i::Integer) = e.group_explanations[i]
Base.size(e::SDMgroupExplanation) = Base.size(e.machine_explanations)
Base.size(e::SDMensembleExplanation) = Base.size(e.group_explanations)

variables(e::SDMmachineExplanation) = keys(e.values)
data(e::SDMmachineExplanation) = e.data
method(e::SDMmachineExplanation) = e.method

for f in [:data, :variables, :method]
    @eval $f(e::Union{SDMgroupExplanation, SDMensembleExplanation}) = $f(first(e))
end

machine_explanations(e::SDMgroupExplanation) = e.machine_explanations
machine_explanations(e::SDMensembleExplanation) = reduce(vcat, machine_explanations.(e))


#### Show methods ####
function Base.show(io::IO, mime::MIME"text/plain", expl::SDMmachineExplanation)
    println(io, "$(typeof(expl)) using method $(typeof(method(expl)))")
end
function Base.show(io::IO, mime::MIME"text/plain", expl::SDMgroupExplanation)
    println(io, "$(typeof(expl)) using method $(typeof(method(expl)))")
end
function Base.show(io::IO, mime::MIME"text/plain", expl::SDMensembleExplanation)
    println(io, "$(typeof(expl)) using method $(typeof(method(expl)))")
end


## By default, variable importance is absolute mean for value for each variable in shapvalues
function variable_importance(expl::SDMmachineExplanation)
    map(vals -> Statistics.mean(abs, vals), expl.values)
end

function variable_importance(expl::Union{SDMgroupExplanation, SDMensembleExplanation})
    group_var_imp = map(variable_importance, machine_explanations(expl))
    map(Statistics.mean, Tables.columntable(group_var_imp))
end

#summary = NamedTuple(var => mapreduce(x -> getfield(x, var), +, importances) / Base.length(importances) for var in ensemble.predictors)



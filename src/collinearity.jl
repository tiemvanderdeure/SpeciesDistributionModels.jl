abstract type AbstractCollinearityMethod end

struct Gvif <: AbstractCollinearityMethod
    threshold
    remove_perfectly_collinear
end
Gvif(; threshold, remove_perfectly_collinear = true) = Gvif(threshold, remove_perfectly_collinear)

struct Vif <: AbstractCollinearityMethod
    threshold
    remove_perfectly_collinear
end
Vif(; threshold, remove_perfectly_collinear = true) = Vif(threshold, remove_perfectly_collinear)

struct Pearson <: AbstractCollinearityMethod
    threshold
end
Pearson(; threshold) = Pearson(threshold)

# Need to add a method for RasterStack, unless it will be Tables.jl compatible
"""
    remove_collinear(data; method, silent = false)

Removes strongly correlated variables in `data`, until correlation is below a threshold specified in `method`.

`method` can currently be either `Gvif`, `Vif` or `Pearson`, which use GVIF, VIF, or Pearson's r, respectively.
GVIF and VIF are similar method, but GVIF includes categorical variables whereas VIF ignores them.

To run without showing information about collinearity scores, set `silent = true`.

## Example
```julia
julia> import SpeciesDistributionModels as SDM
julia> mydata = (a = 1:100, b = sqrt.(1:100), c = rand(100))
julia> SDM.remove_collinear(mydata; method = SDM.Vif(10))
[ Info: a has highest GVIF of 28.367942095054225
[ Info: Removing a, 2 variables remaining
[ Info: b has highest GVIF of 1.0077618445543057
[ Info: All variables are below threshold, returning remaining variables
(:b, :c)
```

"""
function remove_collinear(data; method, silent::Bool = false) 
    schema = Tables.schema(data)
    datakeys = schema.names
    iscategorical = collect(schema.types .<: CategoricalArrays.CategoricalValue)
    _remove_collinear(data, datakeys, method, ~silent, iscategorical)
end

_remove_collinear(data, datakeys, v::Vif, verbose, iscategorical) = (_vifstep(data, datakeys[.~iscategorical], v.threshold, verbose, StatsAPI.vif, v.remove_perfectly_collinear)..., datakeys[iscategorical]...)
_remove_collinear(data, datakeys, v::Gvif, verbose, iscategorical) = _vifstep(data, datakeys, v.threshold, verbose, StatsAPI.gvif, v.remove_perfectly_collinear)
_remove_collinear(data, datakeys, p::Pearson, verbose, iscategorical) = (_pearsonstep(data, datakeys[.~iscategorical], p.threshold, verbose)..., datakeys[iscategorical]...)

function _vifstep(data, datakeys, threshold, verbose, vifmethod, remove_perfectly_collinear)
    highest_vif = threshold + 1.
    while highest_vif > threshold && length(datakeys) > 1
        # make a custom implementation of gvif that works without the useless model
        m = GLM.lm(StatsModels.FormulaTerm(StatsModels.term(1), StatsModels.term.(datakeys)), data) 
        vifresult = vifmethod(m) 
        maxvif = Base.findmax(vifresult)

        if isnan(maxvif[1])
            if remove_perfectly_collinear
                @warn "VIF calculation returned NaN, looking for variables with perfect collinearity."
                perfectly_collinear_var = findfirst(isnan.(StatsModels.stderror(m)[2:end]))
                if isnothing(perfectly_collinear_var)
                    error("Found no perfectly collinear variables")
                else
                    maxvif = (Inf, perfectly_collinear_var)
                    @info "Removing perfectly collinear variable. To disable this behaviour, set remove_perfectly_collinear to false."
                end
            else 
                error("VIF calculation returned NaN. If some variables have perfect collinearity, try running with remove_perfectly_collinear = true.")
            end
        end
        
        if verbose
            @info "$(datakeys[maxvif[2]]) has highest VIF score: $(maxvif[1])"
        end
        
        highest_vif = maxvif[1]

        if highest_vif > threshold
            if verbose
                @info "Removing $(datakeys[maxvif[2]]), $(length(datakeys)-1) variables remaining"
            end
           datakeys = datakeys[Base.setdiff(1:length(datakeys), maxvif[2])] # not very elegant!
        end 
    end

    if verbose
        @info "All variables are below threshold, returning remaining variables"
    end

    return datakeys
end

# to break ties it j
function _pearsonstep(data, datakeys, threshold, verbose)
    data = Tables.columntable(data)[datakeys]
    datamatrix = reduce(hcat, data)
    c = abs.(Statistics.cor(datamatrix) - LinearAlgebra.I)
    correlated_vars_idx = findall(LinearAlgebra.LowerTriangular(c) .> threshold)
    if verbose
        @info "Found $(length(correlated_vars_idx)) pairs of highly correlated vairables"
        for idx in correlated_vars_idx
            println("$(keys(data)[idx.I[1]]) ~ $(keys(data)[idx.I[2]]): $(c[idx])")
        end
    end

    correlated_vars = Tuple.(correlated_vars_idx)
    vars_to_remove = Int[]
    while correlated_vars != []
        cm = mapreduce(x -> collect(x), vcat, correlated_vars) |> StatsBase.countmap # count how often each occurs
        to_remove = findmax(cm)[2]
        correlated_vars = [c for c in correlated_vars if ~in(to_remove, c)]
        append!(vars_to_remove, to_remove)
    end
    if verbose
        @info "Removed $(length(vars_to_remove)) variables: $(keys(data)[vars_to_remove]). Returning remaining variables"
    end
    vars_remaining = keys(data)[setdiff(1:length(keys(data)), vars_to_remove)]
    return vars_remaining
end
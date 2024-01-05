abstract type AbstractCollinearityMethod end

struct Vif <: AbstractCollinearityMethod
    threshold
end
Vif(; threshold) = Vif(threshold)

struct Pearson <: AbstractCollinearityMethod
    threshold
end
Pearson(; threshold) = Pearson(threshold)

# Need to add a method for RasterStack, unless it will be Tables.jl compatible

function remove_collinear(data; max_size = 10_000, method, verbose::Bool = true)
    rows = Tables.rowtable(data)
    nrows = length(rows)
    if max_size < nrows
         data = rows[StatsBase.randperm(nrows)[1:max_size]]
    end
    _remove_collinear(data, method, verbose)
end

_remove_collinear(data, v::Vif, verbose) = _vifstep(data, v.threshold, verbose)
_remove_collinear(data, p::Pearson, verbose) = _pearsonstep(data, p.threshold, verbose)

function _vifstep(data, threshold, verbose)
    datakeys = Tables.schema(data).names
    highest_vif = threshold + 1.
    while highest_vif > threshold 
        # make a custom implementation of gvif that works without the useless model
        m = GLM.lm(StatsModels.FormulaTerm(StatsModels.term(1), StatsModels.term.(datakeys)), data) 
        vifresult = StatsAPI.gvif(m) 
        maxvif = Base.findmax(vifresult)
        highest_vif = maxvif[1]
        if verbose
            @info "$(datakeys[maxvif[2]]) has highest GVIF of $(maxvif[1])"
        end
        if highest_vif > threshold
            if verbose
                @info "Removing $(datakeys[maxvif[2]]), $(length(datakeys)-1) variables remaining"
            end
           datakeys = datakeys[filter(x -> x != maxvif[2], 1:length(datakeys))] # not very elegant!
        end 
    end

    if verbose
        @info "All variables are below threshold, returning remaining variables"
    end

    return datakeys
end

# to break ties it j
function _pearsonstep(data, threshold, verbose)
    data = Tables.columntable(data)
    datamatrix = reduce(hcat, data)
    c = abs.(Statistics.cor(datamatrix) - LinearAlgebra.I)
    correlated_vars_idx = findall(LinearAlgebra.LowerTriangular(c) .> threshold)
    if verbose
        @info "Found $(length(correlated_vars_idx)) correlated variable pairs"
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
    vars_to_remove
    vars_remaining = keys(data)[setdiff(1:length(keys(data)), vars_to_remove)]
    return vars_remaining
end
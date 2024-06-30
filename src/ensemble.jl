# Machines have 1 machine, plus metadata
struct SDMmachine{M<:MLJBase.Probabilistic} 
    machine::Machine
    data::SDMdata
    fold::Int

    SDMmachine(m::Machine{M}, data, fold) where M = new{M}(m, data, fold)
end

# Groups have multiple machines with identical model and resampler
struct SDMgroup <: AbstractVector{SDMmachine}
    sdm_machines::Vector{SDMmachine}
    model
    model_key
end

# Ensembles have multiple groups with potentially different models and different resamplers, but identical data
struct SDMensemble{K}
    groups::NamedTuple{K, <:Tuple{Vararg{SDMgroup}}} # Contains the trained models

    SDMensemble(groups::NamedTuple{K}) where K = new{K}(groups)
end

function Base.getproperty(ensemble::SDMensemble{K}, s::Symbol) where K
    if s in K
        getfield(ensemble.groups, s)
    else
        getfield(ensemble, s)
    end
end


SDMgroupOrEnsemble = Union{SDMgroup, SDMensemble}

# args field stores exactly the data as it is given to the machine
data(mach::SDMmachine) = mach.data
data(s::SDMgroupOrEnsemble) = data(s[1])
model(g::SDMgroup) = g.model
models(e::SDMensemble) = map(g -> g.model, groups(e))
modelkeys(e::SDMensemble{K}) where K = K

Base.iterate(ensemble::SDMensemble) = Base.iterate(ensemble.groups)
Base.iterate(ensemble::SDMensemble, i) = Base.iterate(ensemble.groups, i)
Base.getindex(ensemble::SDMensemble, i) = ensemble.groups[i]
Base.getindex(group::SDMgroup, i) = group.sdm_machines[i]
Base.length(ensemble::SDMensemble) = Base.length(ensemble.groups)
Base.size(group::SDMgroup) = size(group.sdm_machines)
n_machines(ensemble::SDMensemble) = mapreduce(group -> length(group.sdm_machines), +, ensemble)

groups(ensemble::SDMensemble) = ensemble.groups
machines(group::SDMgroup) = map(m -> m.machine, group)
sdm_machines(group::SDMgroup) = group.sdm_machines

# machine_key generates a unique key for a machine
machine_keys(group::SDMgroup) = [Symbol("$(group.model_key)_fold$(m.fold)") for m in group]

# A bunch of functions are applied to an ensemble by applying to each group and reducing with vcat
for f in (:machines, :machine_keys, :sdm_machines)
    @eval ($f)(ensemble::SDMensemble) = mapreduce(group -> ($f)(group), vcat, ensemble)
end

model_names(ensemble) = getfield.(ensemble.groups, :model_name)

## Select methods
# Function to convienently select some models from groups or ensembles
function select(group::SDMgroup, machine_indices::AbstractVector{<:Integer})
    if length(machine_indices) == 0
        return nothing
    else 
        return SDMgroup(
            group.sdm_machines[machine_indices],
            group.model,
            group.resampler,
            group.model_name,
        )
    end
end

function select(ensemble::SDMensemble, group_indices::AbstractVector{<:Integer})
    if length(group_indices) == 0
        return nothing
    else 
        return SDMensemble(
            ensemble.groups[group_indices],
        )
    end
end

function select(ensemble::SDMensemble; machines)
    if length(machines) == 0
        return nothing
    else 
        Statistics.maximum(machines) <= n_machines(ensemble) || throw(BoundsError(ensemble, machines))

        j = 0
        groups = SDMgroup[]
        groups = map(ensemble) do group
            i = j
            j = i + length(group)
            select(group, filter(x -> x > i && x <= j, machines) .- i)
        end

        return SDMensemble(
            filter(!Base.isnothing, groups),
        )
    end
end

## Show methods
function Base.show(io::IO, mime::MIME"text/plain", ensemble::SDMensemble{K}) where K
    d = ensemble[1][1].data#data(ensemble)
    
    println(io, "trained SDMensemble, containing $(n_machines(ensemble)) SDMmachines across $(Base.length(ensemble)) SDMgroups \n")

    print("Uses the following models: \n")   
    for k in K
        modeltype = MLJBase.name(model(ensemble[k]))
        printstyled(io, k, color = :blue)
        print(io, " => $modeltype. \n")
    end
end


#=
    data = ensemble[1][1].data#data(ensemble)
    responsedata = response(data)

    n_presence = sum(responsedata .== true)
    n_absence = sum(length(responsedata) - n_presence)

    sc = MLJBase.schema(predictor(data))
    sci = sc.scitypes
    nam = sc.names

    println(io, "Occurence data: Presence-Absence with $n_presence presences and $n_absence absences")
    println(io, "Predictors: $(join(["$key ($scitype)" for (key, scitype) in zip(nam, sci)], ", "))")

    resampler_names = MLJBase.name.(getfield.(ensemble.groups, :resampler))
    n_models = Base.length.(ensemble.groups)
    table_cols = hcat(K, resampler_names, n_models)
    header = (["model", "resampler", "machines"])
    PrettyTables.pretty_table(io, table_cols; header = header)
    =#


function Base.show(io::IO, mime::MIME"text/plain", group::SDMgroup)
    println(io, "trained SDMgroup, containing $(length(group.sdm_machines)) SDMmachines")
    println(io, "name: $(group.model_name)")
    println(io, "model type: $(MLJBase.name(group.model))")
end

function Base.show(io::IO, mime::MIME"text/plain", mach::SDMmachine)
    println(io, "trained SDMmachine")
    println(io, "fold number: $(mach.fold)")
    println(io, "model type: $(MLJBase.name(mach.machine.model))")
end

## Table interface !! Is this still valid?
Tables.istable(::Type{SDMensemble}) = true
Tables.schema(ensemble::SDMensemble) = Tables.schema(ensemble.groups[1])
Tables.rows(ensemble::SDMensemble) = Tables.rows(ensemble.groups)
Tables.columns(ensemble::SDMensemble) = Tables.columns(ensemble.groups)

# Turns models into a NamedTuple with unique keys
function _givenames(models::Vector)
    names = map(models) do model
        Base.replace(MLJBase.name(model), "Classifier"=>"", "Binary"=>"")
    end
    for (name, n) in StatsBase.countmap(names)
        if n > 1
            names[names .== name] = name .* "_" .* string.(1:n)
        end
    end
    return NamedTuple{Tuple(Symbol.(names))}(models)
end

function _fit_sdm_model(data::SDMdata, model::Probabilistic, fold, train, test, verbosity, cache, scitype_check_level)
    scitype_check_level = scitype_check_level * fold == 1
    mach = MLJBase.machine(model, predictor(data), data.response; cache, scitype_check_level)
    MLJBase.fit!(mach; rows = train, verbosity)
    return SDMmachine(mach, data, fold)
end

function _fit_sdm_group(
    data,
    model, 
    model_key,
    verbosity,
    cache,
    scitype_check_level,
    cpu_backend
    )

    machines = _map(cpu_backend)(enumerate(traintestpairs(data))) do (f, (train, test))
        _fit_sdm_model(data, model, f, train, test, verbosity, cache, scitype_check_level)
    end

    return SDMgroup(machines, model, model_key)

end

function _sdm(
    data::SDMdata,
    models::NamedTuple{K}, 
    verbosity::Int,
    cache, 
    scitype_check_level,
    threaded::Bool
) where K
    backend = cpu_backend(threaded)
    sdm_groups = _map(backend)(keys(models)) do model_key
        model = models[model_key]
        _fit_sdm_group(
            data,
            model, 
            model_key,
            verbosity,
            cache,
            scitype_check_level,
            backend
        )
    end |> NamedTuple{keys(models)}

    return SDMensemble(sdm_groups)
end
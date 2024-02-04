# Machines have 1 machine, plus metadata
struct SDMmachine
    machine
    predictors::NTuple{<:Any, Symbol}
    fold
    train_rows
    test_rows
end

# Groups have multiple machines with identical model and resampler
struct SDMgroup <: AbstractVector{SDMmachine}
    sdm_machines::Vector{SDMmachine}
    model
    resampler
    model_name
    resampler_name
end

# Ensembles have multiple groups with potentially different models and different resamplers, but identical data
struct SDMensemble <: AbstractVector{SDMgroup}
    groups::Vector{SDMgroup} # Contains the trained models
end

SDMgroupOrEnsemble = Union{SDMgroup, SDMensemble}

# Rather than storing it in ensemble, access it like this? It looks like args stores exactly the data as it is given to the machine
data(mach::SDMmachine) = (predictor = mach.machine.args[1].data, response = mach.machine.args[2].data)
data(group::SDMgroup) = data(group[1])
data(ensemble::SDMensemble) = data(ensemble[1])



#Base.getproperty(ensemble::SDMensemble, key::Symbol) = getproperty.(ensemble, key)
# getproperty directly into models???

Base.getindex(ensemble::SDMensemble, i) = ensemble.groups[i]
Base.getindex(group::SDMgroup, i) = group.sdm_machines[i]
Base.size(ensemble::SDMensemble) = size(ensemble.groups)
Base.size(group::SDMgroup) = size(group.sdm_machines)
n_machines(ensemble::SDMensemble) = mapreduce(group -> length(group.sdm_machines), +, ensemble)

machines(group::SDMgroup) = map(m -> m.machine, group)

sdm_machines(group::SDMgroup) = group.sdm_machines

# machine_key generates a unique key for a machine
machine_keys(group::SDMgroup) = [Symbol("$(group.model_name)_$(group.resampler_name)_$(m.fold)") for m in group]

# A bunch of functions are applied to an ensemble by applying to each group and reducing with vcat
for f in (:machines, :machine_keys, :sdm_machines)
    @eval ($f)(ensemble::SDMensemble) = mapreduce(group -> ($f)(group), vcat, ensemble)
end

model_names(ensemble) = getfield.(ensemble.groups, :model_name)

## Select methods
# Function to convienently select some models from groups or ensembles
function select(group::SDMgroup, machine_indices::Vector{<:Int})
    if length(machine_indices) == 0
        return nothing
    else 
        return SDMgroup(
            group.sdm_machines[machine_indices],
            group.model,
            group.resampler,
            group.model_name,
            group.resampler_name
        )
    end
end

function select(ensemble::SDMensemble, group_indices::Vector{<:Int})
    if length(group_indices) == 0
        return nothing
    else 
        return SDMensemble(
            ensemble.groups[group_indices],
        )
    end
end

function select(ensemble::SDMensemble; machines::Vector{<:Int})
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
function Base.show(io::IO, mime::MIME"text/plain", ensemble::SDMensemble)
    n_presence = sum(data(ensemble).response .== true)
    n_absence = sum(data(ensemble).response .== false)

    sc = MLJBase.schema(data(ensemble).predictor)
    sci = sc.scitypes
    nam = sc.names

    println(io, "SDMensemble with $(n_machines(ensemble)) machines across $(Base.length(ensemble)) groups")
    println(io, "Occurence data: Presence-Absence with $n_presence presences and $n_absence absences")
    println(io, "Predictors: $(join(["$key ($scitype)" for (key, scitype) in zip(nam, sci)], ", "))")

    m_names = model_names(ensemble)
    resampler_names = getfield.(ensemble.groups, :resampler_name)
    n_models = Base.length.(ensemble.groups)
    table_cols = hcat(m_names, resampler_names, n_models)
    header = (["model", "resampler", "machines"])
    PrettyTables.pretty_table(io, table_cols; header = header)
    
end

function Base.show(io::IO, mime::MIME"text/plain", group::SDMgroup)
    println(io, "SDMgroup with $(Base.length(group)) machines")
    println(io, "Model $(group.model_name) and resampler $(group.resampler_name)")
end


## Table interface !! Is this still valid?
Tables.istable(::Type{SDMensemble}) = true
Tables.schema(ensemble::SDMensemble) = Tables.schema(ensemble.groups[1])
Tables.rows(ensemble::SDMensemble) = Tables.rows(ensemble.groups)
Tables.columns(ensemble::SDMensemble) = Tables.columns(ensemble.groups)

# Turns models into a NamedTuple with unique keys
function _givenames(models::Vector)
    names = map(models) do model
        Base.replace(MLJBase.name(model), r"Classifier$"=>"", r"Binary$"=>"")
    end
    for (name, n) in StatsBase.countmap(names)
        if n > 1
            names[names .== name] = name .* "_" .* string.(1:n)
        end
    end
    return NamedTuple{Tuple(Symbol.(names))}(models)
end

function _fit_sdm_model(predictor_values::NamedTuple, response_values, model, fold, train, test, verbosity)
    mach = MLJBase.machine(model, predictor_values, response_values)
    MLJBase.fit!(mach; rows = train, verbosity = verbosity)
    return SDMmachine(mach, keys(predictor_values), fold, train, test)
end

function _fit_sdm_group(
    predictor_values::NamedTuple, 
    response_values, 
    model, 
    resampler, 
    folds,
    model_name, 
    resampler_name,
    verbosity
    )

    machines = map(enumerate(folds)) do (f, (train, test))
        _fit_sdm_model(predictor_values, response_values, model, f, train, test, verbosity)
    end

    return SDMgroup(machines, model, resampler, model_name, resampler_name)

end

function _fit_sdm_ensemble(
    presences, 
    absence, 
    models, 
    resamplers,
    predictors::Vector{Symbol},
    verbosity::Int = 0
)
    @assert Tables.istable(presences) && Tables.istable(absence)

    n_presence = Base.length(Tables.rows(presences)) ##
    n_absence = Base.length(Tables.rows(absence))
    n_total = n_presence + n_absence

    # merge presence and absence data into one namedtuple of vectors
    predictor_values = NamedTuple{Tuple(predictors)}([[Tables.columns(absence)[var]; Tables.columns(presences)[var]] for var in predictors])
    response_values = CategoricalArrays.categorical(
        [falses(n_absence); trues(n_presence)]; 
        levels = [false, true], ordered = false
    )

    models_ = _givenames(models)
    resamplers_ = _givenames(resamplers)
    
    sdm_groups = mapreduce(vcat, collect(keys(resamplers_))) do resampler_key
        resampler = resamplers_[resampler_key]
        folds = MLJBase.train_test_pairs(resampler, 1:n_total, response_values) ## get indices
        map(collect(keys(models_))) do model_key
            model = models_[model_key]
            _fit_sdm_group(
                predictor_values, 
                response_values, 
                model, 
                resampler, 
                folds,
                model_key, 
                resampler_key,
                verbosity
            )
        end
    end

    return SDMensemble(sdm_groups)

end
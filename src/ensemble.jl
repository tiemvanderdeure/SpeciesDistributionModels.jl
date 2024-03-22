# Machines have 1 machine, plus metadata
struct SDMmachine
    machine::MLJBase.Machine
    predictors::NTuple{<:Any, Symbol}
    fold::Int
    train_rows::Vector{Int}
    test_rows::Vector{Int}
end

# Groups have multiple machines with identical model and resampler
struct SDMgroup <: AbstractVector{SDMmachine}
    sdm_machines::Vector{SDMmachine}
    model
    resampler
    model_name
end

# Ensembles have multiple groups with potentially different models and different resamplers, but identical data
struct SDMensemble <: AbstractVector{SDMgroup}
    groups::Vector{SDMgroup} # Contains the trained models
end

SDMgroupOrEnsemble = Union{SDMgroup, SDMensemble}

# args field stores exactly the data as it is given to the machine
data(mach::SDMmachine) = (predictor = mach.machine.args[1].data, response = mach.machine.args[2].data)
data(s::SDMgroupOrEnsemble) = data(s[1])
predictors(mach::SDMmachine) = mach.predictors
predictors(s::SDMgroupOrEnsemble) = predictors(s[1])


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
machine_keys(group::SDMgroup) = [Symbol("$(group.model_name)_fold$(m.fold)") for m in group]

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
    resampler_names = MLJBase.name.(getfield.(ensemble.groups, :resampler))
    n_models = Base.length.(ensemble.groups)
    table_cols = hcat(m_names, resampler_names, n_models)
    header = (["model", "resampler", "machines"])
    PrettyTables.pretty_table(io, table_cols; header = header)
    
end

function Base.show(io::IO, mime::MIME"text/plain", group::SDMgroup)
    println(io, "SDMgroup with $(Base.length(group)) machines")
    println(io, "Model $(group.model_name) and resampler $(MLJBase.name(group.resampler))")
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

function _fit_sdm_model(predictor_values::NamedTuple, response_values, model, fold, train, test, verbosity)
    mach = MLJBase.machine(model, predictor_values, response_values, scitype_check_level= fold == 1)
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
    verbosity,
    cpu_backend
    )

    machines = _map(cpu_backend)(enumerate(folds)) do (f, (train, test))
        _fit_sdm_model(predictor_values, response_values, model, f, train, test, verbosity)
    end

    return SDMgroup(machines, model, resampler, model_name)

end

function _sdm(
    presences,
    absences,
    models,
    resampler = MLJBase.CV(; nfolds = 5, shuffle = true),
    predictors = _get_predictor_names(presences, absences),
    verbosity = 0,
    threaded = false
)
    X, y = _predictor_response_from_presence_absence(presences, absences, predictors)
    # handle geometries separately. In the future, we might use geometries somehow, 
    # e.g. to add spatial resampling, or define a scatter! recipe.
    # Right now geometries are not stored/used
    :geometry ∈ predictors && error("Predictors cannot be called :geometry")
    #if :geometry ∈ Tables.columnnames(presences) && :geometry ∈ Tables.columnnames(absences)
    #    geometries = [vcat(Tables.getcolumn(presences, :geometry), Tables.getcolumn(absences, :geometry))]
    #else
    #    geometries = nothing
    #end

    _sdm(X, y, models, resampler, verbosity, threaded)
end

function _sdm(
    X, 
    y::CategoricalArrays.CategoricalArray{Bool},
    models::Vector{<:MLJBase.Model}, 
    resampler::MLJBase.ResamplingStrategy,
    predictors,
    verbosity::Int,
    threaded::Bool
)
    train_test_rows = MLJBase.train_test_pairs(resampler, 1:length(y), y)
    _fit_sdm_ensemble(X, y; models, resampler, train_test_rows, predictors, verbosity, threaded)
end

function _sdm(
    X,
    y::CategoricalArrays.CategoricalArray{Bool},
    models::Vector{<:MLJBase.Model},
    train_test_rows::Vector{<:Tuple{AbstractVector{<:Integer}, AbstractVector{<:Integer}}},
    predictors,
    verbosity::Int,
    threaded::Bool
)
    _fit_sdm_ensemble(X, y; models, resampler = CustomRows(), train_test_rows, predictors, verbosity, threaded)
end

function _fit_sdm_ensemble(X, y; models, resampler, train_test_rows, predictors, verbosity, threaded)
    cols = Tables.columns(X)
    X_ = NamedTuple{Tuple(predictors)}([Tables.getcolumn(cols, pred) for pred in predictors])
    _fit_sdm_ensemble(X_, y, models, resampler, train_test_rows, verbosity, threaded::Bool)
end

function _fit_sdm_ensemble(X, y, models, resampler, folds, verbosity, threaded::Bool)
    models = _givenames(models)
    backend = cpu_backend(threaded)
    sdm_groups = _map(backend)(collect(keys(models))) do model_key
        model = models[model_key]
        _fit_sdm_group(
            X, 
            y, 
            model, 
            resampler, 
            folds,
            model_key, 
            verbosity,
            backend
        )
    end

    return SDMensemble(sdm_groups)
end
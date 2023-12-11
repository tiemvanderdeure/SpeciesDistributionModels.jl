# Machines have 1 machine, plus metadata
struct SDMmachine
    machine
    predictors::NTuple{<:Any, Symbol}
    fold
    train
    test
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
    data
end

#Base.getproperty(ensemble::SDMensemble, key::Symbol) = getproperty.(ensemble, key)
# getproperty directly into models???

Base.getindex(ensemble::SDMensemble, i) = ensemble.groups[i]
Base.getindex(group::SDMgroup, i) = group.sdm_machines[i]
Base.size(ensemble::SDMensemble) = size(ensemble.groups)
Base.size(group::SDMgroup) = size(group.sdm_machines)

#trained_models(ensemble) = getfield(ensemble, :trained_models)
machines(ensemble::SDMensemble) = mapreduce(group -> group.sdm_machines.machine, vcat, ensemble.group)
sdm_machines(ensemble::SDMensemble) = mapreduce(group -> group.sdm_machines, vcat, ensemble.group)
#machine_keys(ensemble::SDMensemble) = getfield.(ensemble.trained_models, :machine_key)

# Symbol(String(model_key) * "_" * String(resampler_key) * "_" * string(f))

# Function to convienently select some models from the ensemble
function select(ensemble::SDMensemble, indices::Vector{Int})
    trained_models = ensemble.trained_models[indices]

    model_keys = unique(getfield.(trained_models, :model_key))
    resampler_keys = unique(getfield.(trained_models, :resampler_key))

    return SDMensemble(
        trained_models, 
        ensemble.predictors, 
        ensemble.n_presences, 
        ensemble.n_absences, 
        ensemble.models[model_keys], 
        ensemble.resamplers[resampler_keys],
        ensemble.data
    )
end

function select(
    ensemble::SDMensemble; 
    models = keys(ensemble.models), 
    resamplers = keys(ensemble.resamplers),
    folds = 1:1000)

    select(
        ensemble,
        findall(
            m -> m.model_key in models && m.resampler_key in resamplers && m.fold in folds, 
            ensemble.trained_models
        )
    )
end

function Base.show(io::IO, mime::MIME"text/plain", ensemble::SDMensemble)
    println(io, "SDMensemble with $(Base.length(ensemble)) groups")

    model_names = getfield.(ensemble.groups, :model_name)
    resampler_names = getfield.(ensemble.groups, :resampler_name)
    n_models = Base.length.(ensemble.groups)
    data = hcat(model_names, resampler_names, n_models)
    header = (["model", "resampler", "number of models"])
    PrettyTables.pretty_table(io, data; header = header)

    #println(io, "model keys: ", keys(ensemble.models))
    #show(io, mime, Tables.rows(ensemble))
end

Tables.istable(::Type{SDMensemble}) = true
Tables.schema(ensemble::SDMensemble) = Tables.schema(ensemble.trained_models[1])
Tables.rows(ensemble::SDMensemble) = Tables.rows(ensemble.trained_models)
Tables.columns(ensemble::SDMensemble) = Tables.columns(ensemble.trained_models)

# Turns models into a NamedTuple with unique keys
function _givenames(models::Vector)
    names = map(models) do model
        replace(MLJBase.name(model), r"Classifier$"=>"")
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

function sdm(
    presences, 
    absence, 
    models, 
    resamplers;
    var_keys::Vector{Symbol} = intersect(Tables.schema(absence).names, Tables.schema(presences).names),
    scitypes::Vector{DataType} = [MLJBase.scitype(Tables.schema(presences).types) for key in var_keys],
    verbosity::Int = 0
)
    @assert Tables.istable(presences) && Tables.istable(absence)

    n_presence = Base.length(Tables.rows(presences)) ##
    n_absence = Base.length(Tables.rows(absence))
    n_total = n_presence + n_absence

    # merge presence and absence data into one namedtuple of vectors
    predictor_values = NamedTuple{Tuple(var_keys)}([[Tables.columns(absence)[var]; Tables.columns(presences)[var]] for var in var_keys])
    response_values = CategoricalArray(
        [falses(n_absence); trues(n_presence)]; 
        levels = [false, true], ordered = true
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

    return SDMensemble(
        sdm_groups, 
        (predictor = predictor_values, response = response_values)
    )

end
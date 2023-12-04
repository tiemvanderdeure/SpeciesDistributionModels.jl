struct SDMensemble <: AbstractVector{NamedTuple}
    trained_models::Vector{<:NamedTuple} # Contains the trained models
    predictors::Vector{Symbol} # Vector of names of predictor variables # Better tuple of vector???
    n_presences::Int
    n_absences::Int
    models::NamedTuple
    resamplers::NamedTuple
    data
    #folds::Vector{<:Vector}
end

#Base.getproperty(ensemble::SDMensemble, key::Symbol) = getproperty.(ensemble, key)
# getproperty directly into models???

Base.getindex(ensemble::SDMensemble, i) = ensemble.trained_models[i]
Base.size(ensemble::SDMensemble) = size(ensemble.trained_models)

#trained_models(ensemble) = getfield(ensemble, :trained_models)
machines(ensemble::SDMensemble) = getfield.(ensemble.trained_models, :machine)
machine_keys(ensemble::SDMensemble) = getfield.(ensemble.trained_models, :machine_key)

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
    println(io, "SDMensemble with $(Base.length(ensemble)) models")

    println(io, "Model performance:")

    aucs = auc_by_model(ensemble)
    data = hcat(collect(keys(ensemble.models)), aucs)
    header = (["model_key", "auc"])
    PrettyTables.pretty_table(io, data; header = header)

    #println(io, "model keys: ", keys(ensemble.models))
    #show(io, mime, Tables.rows(ensemble))
end

Tables.istable(::Type{SDMensemble}) = true
Tables.schema(ensemble::SDMensemble) = Tables.schema(ensemble.trained_models[1])
Tables.rows(ensemble::SDMensemble) = Tables.rows(ensemble.trained_models)
Tables.columns(ensemble::SDMensemble) = Tables.columns(ensemble.trained_models)

# Turns models into a NamedTuple with unique keys
function givenames(models::Vector)
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

function auc_by_model(ensemble)
    mapreduce(vcat, keys(ensemble.models)) do key
        Statistics.mean([model.auc for model in ensemble.trained_models if model.model_key == key])
    end
end

function sdm(
    presences, 
    absence, 
    models, 
    resamplers;
    var_keys::Vector{Symbol} = [key for key in Tables.schema(absence).names if in(key, Tables.schema(presences).names)],
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
        levels = [false, true], ordered = true)

    models_ = givenames(models)
    resamplers_ = givenames(resamplers)

    trained_models = mapreduce(vcat, keys(resamplers_)) do resampler_key
        resampler = resamplers_[resampler_key]
        folds = MLJBase.train_test_pairs(resampler, 1:n_total, response_values) ## get indices
        mapreduce(vcat, keys(models_)) do model_key
            model = models_[model_key]
            map(enumerate(folds)) do (f, (train, test))
                mach = MLJBase.machine(model, predictor_values, response_values)
                MLJBase.fit!(mach; rows = train, verbosity = verbosity)
                y_hat = MLJBase.predict(mach, rows = test)
                auc = StatisticalMeasures.auc(y_hat, response_values[test])
                machine_key = Symbol(String(model_key) * "_" * String(resampler_key) * "_" * string(f))
                return (; machine = mach, auc = auc, model_key, resampler_key, fold = f, machine_key, train, test)
                # Probably make a Type for this
            end
        end
    end

    return SDMensemble(
        trained_models, 
        var_keys, 
        n_presence, 
        n_absence, 
        models_, 
        resamplers_,
        (predictor = predictor_values, response = response_values)
        )
end

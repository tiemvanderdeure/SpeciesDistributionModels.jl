### Miscelanious utilities to deal with data issues such as names, missing values

# Convert a BitArray to a CategoricalArray. Faster and type-stable version of `categorical`
function boolean_categorical(A::BitArray{N}) where N 
    CategoricalArrays.CategoricalArray{Bool, N, UInt8}(A, levels=[false, true], ordered=false)
end
boolean_categorical(A::AbstractVector{Bool}) = boolean_categorical(BitArray(A))

function _get_predictor_names(p, a)
    predictors = Base.intersect(Tables.schema(a).names, Tables.schema(p).names)
    predictors = filter!(!=(:geometry), predictors) # geometry is never a variable
    length(predictors) > 0 || error("Presence and absence data have no common variable names - can't fit the ensemble.")
    return predictors
end


function _predictor_response_from_presence_absence(presences, absences, predictors)
    p_columns = Tables.columns(presences)
    a_columns = Tables.columns(absences) 
    n_presence = Tables.rowcount(p_columns)
    n_absence = Tables.rowcount(a_columns)

    # merge presence and absence data into one namedtuple of vectors
    predictor_values = NamedTuple{Tuple(predictors)}([[a_columns[var]; p_columns[var]] for var in predictors])
    response_values = boolean_categorical([falses(n_absence); trues(n_presence)])
    return predictor_values, response_values
end


cpu_backend(threaded) = threaded ? CPUThreads() : CPU1()
_map(::CPU1) = Base.map
_map(::CPUThreads) = ThreadsX.map
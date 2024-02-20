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

_map(::CPU1) = Base.map
_map(::CPUThreads) = ThreadsX.map

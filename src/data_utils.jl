### Miscelanious utilities to deal with data issues such as names, missing values

function _get_predictor_names(p, a)
    predictors = Base.intersect(Tables.schema(a).names, Tables.schema(p).names)
    predictors = filter!(!=(:geometry), predictors) # geometry is never a variable
    length(predictors) > 0 || error("Presence and absence data have no common variable names - can't fit the ensemble.")
    return predictors
end


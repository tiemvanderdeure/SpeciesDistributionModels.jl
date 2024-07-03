"""
    ShapleyValues(algorithm::Shapley.Algorithm)
    ShapleyValues(N::Integer; threaded = true, rng = Random.GLOBAL_RNG)

Use to specify use Shapley values as method in [`explain`](@ref).
If an integer `N`, and optionally `threaded` and `rng` is supplied, `MonteCarlo` sampling is used,
where `N` is the number of iterations (samples). More samples will result in more accurate results, 
but will take more time to compute.
"""
struct ShapleyValues <: SDMexplainMethod
    algorithm::Shapley.Algorithm
end
# Default to MonteCarlo algorithm with 100 samples
function ShapleyValues(N::Integer; threaded = false, rng = Random.GLOBAL_RNG)
    resource = threaded ? CPUThreads() : CPU1()
    algorithm = Shapley.MonteCarlo(resource, N, rng)
    ShapleyValues(algorithm)
end

function _explain(mach::SDMmachine, method::ShapleyValues, d, predictors)
    shapvalues = map(predictors) do predictor
        Shapley.shapley(
            x -> _reformat_and_predict(mach, x, false), # some ml models return float32s - where to handle this?
            method.algorithm, 
            d,
            predictor,
            d
        )
    end |> NamedTuple{predictors}
    return SDMmachineExplanation(mach, method, shapvalues, d)
end

function _explain(group::SDMgroup, method::ShapleyValues, d, predictors)
    machine_explanations = map(group) do mach
        _explain(mach, method, d, predictors)
    end
    return SDMgroupExplanation(group, machine_explanations)
end

function _explain(ensemble::SDMensemble, method::ShapleyValues, d, predictors)
    group_explanations = map(ensemble) do group
        _explain(group, method, d, predictors)
    end
    
    return SDMensembleExplanation(ensemble, group_explanations)
end

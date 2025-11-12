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

function _explain(mach::Machine, method::ShapleyValues, d, predictors)
    map(predictors) do predictor
        Shapley.shapley(
            x -> MLJBase.pdf.(MLJBase.predict(mach, x), true), # some ml models return float32s - where to handle this?
            method.algorithm, 
            d,
            predictor,
            d
        )
    end |> NamedTuple{predictors}
end


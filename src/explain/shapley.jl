"""
    ShapleyValues

A method descriptor for computing Shapley values for model explanations.

Use to specify Shapley values as method in [`explain`](@ref).

## Constructors

    ShapleyValues(algorithm::Shapley.Algorithm)
    ShapleyValues(N::Integer; threaded = false, rng = Random.GLOBAL_RNG)

If an integer `N` is provided, `MonteCarlo` sampling is used, where `N` is the number
 of iterations (samples). More samples yield more accurate results but take more time.

## Keywords (for Integer constructor)
- `threaded`: if `true`, use multithreading. Defaults to `false`.
- `rng`: random number generator. Defaults to `Random.GLOBAL_RNG`.
"""
struct ShapleyValues <: SDMexplainMethod
    algorithm::Shapley.Algorithm
end
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


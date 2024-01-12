using SpeciesDistributionModels, Random, CategoricalArrays
import SpeciesDistributionModels as SDM

import GLM: Distributions
rand(Distributions.Binomial(3, 0.5))

# some mock data
Random.seed!(0)
n = 500
backgrounddata = (a = rand(n), b = rand(n), c = categorical(rand(0:3, n)))
presencedata = (a = rand(n), b = rand(n).^2, c = categorical(rand(Distributions.Binomial(3, 0.5), n)))

using Test

@testset "SpeciesDistributionModels.jl" begin
    # Write your tests here.
    presences = (a = rand(200), b = rand(200), c = rand(200))
    absence =  [(a = rand(), b = rand()*2, c = rand().^2) for _ in 1:500]
    models = [SDM.random_forest(), SDM.random_forest(; max_depth = 3), SDM.linear_model(), SDM.boosted_regression_tree()]
    resamplers = [SDM.MLJ.CV(; shuffle = true, nfolds = 5)]

    ensemble = sdm(
        presences, absence, 
        models, 
        resamplers
    )

    evaluation = evaluate(ensemble)
end

@testset "collinearity" begin
    # mock data with a collinearity problem
    Random.seed!(1)
    data_with_collinearity = merge(backgrounddata, (; d = data.a .+ rand(n), e = data.a .+ rand(n)))

    rm_col_vif = remove_collinear(data_with_collinearity; method = SDM.Vif(; threshold = 2.), verbose = false)
    rm_col_pearson = remove_collinear(data_with_collinearity; method = SDM.Pearson(; threshold = 0.65), verbose = false)
    @test rm_col_vif == (:b, :c, :d, :e)
    @test rm_col_pearson == (:b, :c, :d, :e)

end
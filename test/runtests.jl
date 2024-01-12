using SpeciesDistributionModels, CategoricalArrays
import SpeciesDistributionModels as SDM

import GLM: Distributions

# some mock data
n = 500
backgrounddata = (a = rand(n), b = rand(n), c = categorical(rand(0:3, n)))
presencedata = (a = rand(n), b = rand(n).^2, c = categorical(rand(Distributions.Binomial(3, 0.5), n)))

using Test

@testset "SpeciesDistributionModels.jl" begin
    models = [SDM.random_forest(), SDM.random_forest(; max_depth = 3), SDM.linear_model(), SDM.boosted_regression_tree()]
    resamplers = [SDM.MLJBase.CV(; shuffle = true, nfolds = 5)]

    ensemble = sdm(
        presencedata, backgrounddata, 
        models, 
        resamplers
    )

    evaluation = SDM.evaluate(ensemble)
end

@testset "collinearity" begin
    # mock data with a collinearity problem
    data_with_collinearity = merge(backgrounddata, (; d = backgrounddata.a .+ rand(n), e = backgrounddata.a .+ rand(n)))

    rm_col_vif = remove_collinear(data_with_collinearity; method = SDM.Vif(; threshold = 2.), verbose = false)
    rm_col_pearson = remove_collinear(data_with_collinearity; method = SDM.Pearson(; threshold = 0.65), verbose = false)
    @test rm_col_vif == (:b, :c, :d, :e)
    @test rm_col_pearson == (:b, :c, :d, :e)
end
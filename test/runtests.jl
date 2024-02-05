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

    ensemble = sdm_ensemble(
        presencedata, backgrounddata;
        models = models, resampler = SDM.MLJBase.CV(; shuffle = true, nfolds = 5)
    )

    evaluation = SDM.evaluate(ensemble)

    pr1 = SDM.predict(ensemble, backgrounddata)
    pr2 = SDM.predict(ensemble, backgrounddata; reducer = maximum)
    pr3 = SDM.predict(ensemble, backgrounddata; reducer = maximum, by_group = true)

    # Need much better tests
    @test evaluation isa SDM.SDMensembleEvaluation
    @test evaluation[1] isa SDM.SDMgroupEvaluation
    @test evaluation[1][1] isa SDM.SDMmachineEvaluation
    
    @test pr2 isa Vector
    @test collect(keys(pr1)) == SDM.machine_keys(ensemble)
    @test collect(keys(pr3)) == SDM.model_names(ensemble)

    @test_throws ArgumentError SDM.predict(ensemble, backgrounddata.a)
    @test_throws Exception SDM.predict(ensemble, backgrounddata[(:a,)])
    @test_throws Exception SDM.predict(ensemble, backgrounddata; by_group = true)
end

@testset "collinearity" begin
    # mock data with a collinearity problem
    data_with_collinearity = merge(backgrounddata, (; d = backgrounddata.a .+ rand(n), e = backgrounddata.a .+ rand(n), f = f = categorical(rand(Distributions.Binomial(3, 0.5), 500))    ))

    rm_col_gvif = remove_collinear(data_with_collinearity; method = SDM.Gvif(; threshold = 2.), silent = true)
    rm_col_vif = remove_collinear(data_with_collinearity; method = SDM.Vif(; threshold = 2.), silent = true)
    rm_col_pearson = remove_collinear(data_with_collinearity; method = SDM.Pearson(; threshold = 0.65), silent = true)
    @test rm_col_gvif == (:b, :c, :d, :e, :f)
    @test rm_col_vif == (:b, :d, :e, :c, :f)
    @test rm_col_pearson == (:b, :d, :e, :c, :f)

    data_with_perfect_collinearity = (a = [1,2,3], b = [1,2,3])
    Test.@test_throws Exception remove_collinear(data_with_perfect_collinearity; method = SDM.Gvif(; threshold = 2., remove_perfectly_collinear = false), silent = true)
    @test remove_collinear(data_with_perfect_collinearity; method = SDM.Gvif(; threshold = 2.), silent = true) == (:a, )
    @test remove_collinear(data_with_perfect_collinearity; method = SDM.Pearson(; threshold = 0.65), silent = true) == (:a, )
end
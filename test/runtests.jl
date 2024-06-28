using SpeciesDistributionModels, MLJBase, Tables
import SpeciesDistributionModels as SDM
using StableRNGs, Distributions, Test
using Makie

rng = StableRNG(0)
using Random; rng = Random.GLOBAL_RNG
# some mock data
n = 100
backgrounddata = (a = rand(rng, n), b = rand(rng, n), c = rand(rng, n))
presencedata = (a = rand(rng, n), b = rand(rng, n).^2, c = sqrt.(rand(rng, n)))

@testset "SpeciesDistributionModels.jl" begin
    models = [
        SDM.random_forest(; rng)
        SDM.random_forest(; max_depth = 3, rng)
        SDM.linear_model()
        SDM.boosted_regression_tree(; rng)
    ]

    ensemble = sdm(
        presencedata, backgrounddata;
        models = models, 
        resampler = SDM.MLJBase.CV(; shuffle = true, nfolds = 5, rng), 
        threaded = false
    )
    # alternative sdm method
    x = map(presencedata, backgrounddata) do p, b
        [p; b]
    end
    y = [trues(Tables.rowcount(presencedata)); falses(Tables.rowcount(backgrounddata))]
    ensemble2 = sdm(x, y; models, resampler = SDM.NoResampling())
    
    evaluation = SDM.evaluate(ensemble; validation = (presencedata, backgrounddata))
    evaluation2 = SDM.evaluate(ensemble)
    @test evaluation isa SDM.SDMensembleEvaluation
    @test evaluation[1] isa SDM.SDMgroupEvaluation
    @test evaluation[1][1] isa SDM.SDMmachineEvaluation
    @test evaluation.measures isa NamedTuple
    mach_evals = SDM.machine_evaluations(evaluation)
    @test mach_evals isa NamedTuple{(:train, :test, :validation)}
    @test mach_evals.train isa NamedTuple{(keys(evaluation.measures))}

    machine_aucs = SDM.machine_evaluations(evaluation).test.auc

    pr1 = SDM.predict(ensemble, backgrounddata)
    pr2 = SDM.predict(ensemble, backgrounddata; reducer = maximum)
    pr3 = SDM.predict(ensemble, backgrounddata; reducer = x -> sum(x .> 0.5), by_group = true)

    @test pr2 isa Vector
    @test collect(keys(pr1)) == SDM.machine_keys(ensemble)
    @test collect(keys(pr3)) == SDM.model_names(ensemble)
    eltype(pr3) == Vector{Int64}

    @test_throws ArgumentError SDM.predict(ensemble, backgrounddata.a)
    @test_throws Exception SDM.predict(ensemble, backgrounddata[(:a,)])
    @test_throws Exception SDM.predict(ensemble, backgrounddata; by_group = true)

    # explain
    expl = explain(ensemble; method = ShapleyValues(10; rng))
    varimp = variable_importance(expl)
    @test varimp.b > varimp.a
    @test varimp.c > varimp.a

    # plots
    interactive_evaluation(ensemble, thresholds = 0:0.001:1)
    interactive_response_curves(expl)
    boxplot(evaluation, :auc)
end

@testset "collinearity" begin
    # mock data with a collinearity problem
    data_with_collinearity = merge(backgrounddata, (; d = backgrounddata.a .+ rand(rng, n), e = backgrounddata.a .+ rand(rng, n), f = f = categorical(rand(Distributions.Binomial(3, 0.5), n))))

    rm_col_gvif = remove_collinear(data_with_collinearity; method = SDM.Gvif(; threshold = 2.), silent = false)
    rm_col_vif = remove_collinear(data_with_collinearity; method = SDM.Vif(; threshold = 2.), silent = true)
    rm_col_pearson = remove_collinear(data_with_collinearity; method = SDM.Pearson(; threshold = 0.65), silent = true)
    @test rm_col_gvif == (:b, :c, :d, :e, :f)
    @test rm_col_vif == (:b, :c, :d, :e, :f)
    @test rm_col_pearson == (:b, :c, :d, :e, :f)

    data_with_perfect_collinearity = (a = [1,2,3], b = [1,2,3])
    Test.@test_throws Exception remove_collinear(data_with_perfect_collinearity; method = SDM.Gvif(; threshold = 2., remove_perfectly_collinear = false), silent = true)
    @test remove_collinear(data_with_perfect_collinearity; method = SDM.Gvif(; threshold = 2.), silent = true) == (:a, )
    @test remove_collinear(data_with_perfect_collinearity; method = SDM.Pearson(; threshold = 0.65), silent = true) == (:a, )
end

using SpeciesDistributionModels, MLJBase, Tables
import SpeciesDistributionModels as SDM

import EvoTrees: EvoTreeClassifier
    using Statistics
    using Rasters: RasterStack, Raster
    xdim = Rasters.X(1:10); ydim = Rasters.Y(1:10)
    rs = RasterStack([Raster(rand(xdim,ydim)) for i in 1:2]...; name = [:a, :b])

    X = (a = rand(100), b = rand(Float32, 100) .+ 2)
    y = MLJBase.categorical(rand(Bool, 100))
    ensemble = SDM.SdmEnsemble(tree = EvoTreeClassifier(), tree2 = EvoTreeClassifier(); reducing_function = mean, cache =true)

    mach = machine(ensemble, X, y)
    MLJBase.fit!(sp; force = true)
    MLJBase.predict(sp, rs)

    ev = MLJBase.evaluate!(sp, measure = auc)
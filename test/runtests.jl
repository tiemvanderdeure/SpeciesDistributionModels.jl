using SpeciesDistributionModels, MLJBase, MLJModels, Tables
import SpeciesDistributionModels as SDM
import CategoricalArrays as CA
using StableRNGs, Distributions, Test
using Makie
using Rasters

using MLJGLMInterface: LinearBinaryClassifier
using EvoTrees: EvoTreeClassifier
using MLJDecisionTreeInterface: RandomForestClassifier

rng = StableRNG(0)
#using Random; rng = Random.GLOBAL_RNG

# some mock data
n = 100
backgrounddata = (a = rand(rng, n), b = rand(rng, n), c = rand(rng, n))
presencedata = (a = rand(rng, n), b = rand(rng, n).^2, c = sqrt.(rand(rng, n)))

@testset "SDM data" begin
    geometry = tuple.(rand(rng, n), rand(rng, n))
    bg_with_geometry = merge(backgrounddata, (; geometry))
    pres_with_geometry = merge(presencedata, (; geometry))
    # to construct with one big table and bools
    x = map(presencedata, backgrounddata) do p, b
        [p; b]
    end
    x_with_geometry = merge(x, (; geometry = vcat(geometry, geometry)))
    y = [trues(Tables.rowcount(presencedata)); falses(Tables.rowcount(backgrounddata))]
 
    tabletypes = (Tables.rowtable, Tables.columntable)
    for tabletype in tabletypes
        ## data without geometries
        data = sdmdata(tabletype(presencedata), tabletype(backgrounddata); resampler = CV(rng = StableRNG(0), nfolds = 5))
        data2 = sdmdata(tabletype(x), y; resampler = CV(rng = StableRNG(0), nfolds = 5, shuffle = true))

        @test isnothing(data.geometry)
        @test length(data.traintestpairs) == 5
        @test data.traintestpairs == data2.traintestpairs # shuffling even if not specified
        
        ## data with geometries
        data3 = sdmdata(tabletype(pres_with_geometry), tabletype(bg_with_geometry); resampler = CV(rng = StableRNG(0), nfolds = 5))
        @test data3.geometry == vcat(geometry,geometry)
        @test data.predictor == data3.predictor
        @test data.traintestpairs == data3.traintestpairs
        @test data.response == data3.response

        ## Without resampling
        data4 = sdmdata(tabletype(presencedata), tabletype(backgrounddata))
        @test data4.traintestpairs == [(Base.OneTo(200), [])]

        ## Passing predictor keys
        data5 = sdmdata(tabletype(presencedata), tabletype(backgrounddata); predictors = (:a, :b), resampler = CV(rng = StableRNG(0), nfolds = 5))
        data6 = sdmdata(tabletype(pres_with_geometry), tabletype(bg_with_geometry); predictors = (:a, :b), resampler = CV(rng = StableRNG(0), nfolds = 5))
        data7 = sdmdata(tabletype(x), y; predictors = (:a, :b), resampler = CV(rng = StableRNG(0), nfolds = 5))
        data8 = sdmdata(tabletype(x_with_geometry), y; predictors = (:a, :b), resampler = CV(rng = StableRNG(0), nfolds = 5))
        @test keys(data5.predictor) == keys(data6.predictor) == keys(data7.predictor) == keys(data8.predictor) == (:a, :b)
        @test data6.geometry == data8.geometry == vcat(geometry, geometry)
        @test data.traintestpairs == data6.traintestpairs == data7.traintestpairs == data8.traintestpairs

        for d in (data, data2, data5, data6, data7)
            ys = CA.unwrap.(d.response)
            @test d.predictor.a[ys] == presencedata.a
            @test d.predictor.a[.!ys] == backgrounddata.a 
        end
    end
end

@testset "SpeciesDistributionModels.jl" begin
    ## data
    data = sdmdata(presencedata, backgrounddata; resampler = CV(nfolds = 5, shuffle = true))

    ## ensemble
    models = (
        rf = RandomForestClassifier(; rng),
        rf2 = OneHotEncoder() |> RandomForestClassifier(; max_depth = 3, rng),
        lm = LinearBinaryClassifier(),
        brt = EvoTreeClassifier(; rng)
    )

    ensemble = sdm(data, models;
        threaded = false
    )

    evaluation = SDM.evaluate(ensemble; validation = (presencedata, backgrounddata))
    evaluation2 = SDM.evaluate(ensemble)
    @test evaluation isa SDM.SDMensembleEvaluation
    @test evaluation[1] isa SDM.SDMgroupEvaluation
    @test evaluation[1][1] isa SDM.SDMmachineEvaluation
    @test SDM.measures(evaluation) isa NamedTuple
    mach_evals = SDM.machine_evaluations(evaluation)
    @test mach_evals isa NamedTuple{(:train, :test, :validation)}
    @test mach_evals.train isa NamedTuple{(keys(SDM.measures(evaluation)))}

    machine_aucs = SDM.machine_evaluations(evaluation).test.auc

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

data = sdmdata(presencedata, backgrounddata; resampler = CV(nfolds = 5, shuffle = true))
ensemble = sdm(data, (; lm = LinearBinaryClassifier()))

@testset "predict" begin
    pr1 = SDM.predict(ensemble, backgrounddata)
    pr2 = SDM.predict(ensemble, backgrounddata; reducer = maximum)
    pr3 = SDM.predict(ensemble, backgrounddata; reducer = x -> sum(x .> 0.5), by_group = true)

    @test pr2 isa Vector
    @test collect(keys(pr1)) == SDM.machine_keys(ensemble)
    @test (keys(pr3)) == SDM.model_keys(ensemble)
    eltype(pr3) == Vector{Int64}

    @test_throws ArgumentError SDM.predict(ensemble, backgrounddata.a)
    @test_throws ArgumentError SDM.predict(ensemble, backgrounddata[(:a,)])
    @test_throws Exception SDM.predict(ensemble, backgrounddata; by_group = true)

    ## to a Raster
    ds = (X(1:100), Y(1:100))
    rs = RasterStack((a = rand(ds), b = rand(ds), c = rand(ds)), missingval = 0.0)
    # make the first value missing
    rs[1] = (a = 0, b = 0, c = 0)
    raspr = SDM.predict(ensemble, rs)
    @test all(ismissing, raspr[X=1, Y=1])
    # all values should be between 0 and 1
    extr = extrema(skipmissing(raspr))
    @test extr[1] > 0 && extr[2] < 1
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


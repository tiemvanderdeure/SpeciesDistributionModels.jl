using SpeciesDistributionModels

rng = StableRNG(0)

# some mock data
n = 100
backgrounddata = (a = rand(rng, n), b = rand(rng, n), c = rand(rng, n))
presencedata = (a = rand(rng, n), b = rand(rng, n).^2, c = sqrt.(rand(rng, n)))

geometry = tuple.(rand(rng, n), rand(rng, n))
bg_with_geometry = merge(backgrounddata, (; geometry))
pres_with_geometry = merge(presencedata, (; geometry))

@testset "SDM data" begin
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

@testset "SDM data with user-provided train-test pairs" begin
    cvdata = sdmdata(presencedata, backgrounddata; resampler = CV(rng = StableRNG(0), nfolds = 4))
    pairs = cvdata.traintestpairs
    data = sdmdata(presencedata, backgrounddata; resampler = pairs)
    @test data.traintestpairs == pairs
    @test isnothing(data.resampler)
    @test data.predictor == cvdata.predictor
    @test data.response == cvdata.response
    # also with a table and a response vector, and with geometry
    x = map(vcat, backgrounddata, presencedata)
    y = [falses(n); trues(n)]
    @test sdmdata(x, y; resampler = pairs).traintestpairs == pairs
    @test sdmdata(pres_with_geometry, bg_with_geometry; resampler = pairs).geometry == vcat(geometry, geometry)
    # a tuple of pairs works too
    @test sdmdata(presencedata, backgrounddata; resampler = Tuple(pairs)).traintestpairs == pairs
    # show mentions user-provided pairs
    @test occursin("user-provided", sprint(show, MIME"text/plain"(), data))
    # invalid input
    @test_throws ArgumentError sdmdata(presencedata, backgrounddata; resampler = [(1:150, 151:201)])
    @test_throws ArgumentError sdmdata(presencedata, backgrounddata; resampler = Tuple{Vector{Int}, Vector{Int}}[])
    # an ensemble fitted on user-provided pairs is identical to one fitted on the strategy
    models = (glm = LinearBinaryClassifier(),)
    ev1 = SDM.evaluate(sdm(data, models); measures = (; auc))
    ev2 = SDM.evaluate(sdm(cvdata, models); measures = (; auc))
    @test ev1.score == ev2.score
end

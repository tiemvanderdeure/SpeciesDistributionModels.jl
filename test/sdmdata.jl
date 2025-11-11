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
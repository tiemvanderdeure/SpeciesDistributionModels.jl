@testset "predict" begin
    @testset "To a table" begin
        # predict to a table of background points
        pr = SDM.predict(ensemble, backgrounddata)

        # The prediction result should carry the ensemble dimensions (model, fold)
        @test all(hasdim(pr, (:model, :fold, :row)))
        @test dims(pr, (:model, :fold)) === dims(ensemble)

        # incorrect inputs should error
        @test_throws ArgumentError SDM.predict(ensemble, backgrounddata.a)
        @test_throws ArgumentError SDM.predict(ensemble, backgrounddata[(:a,)])

        # by_group prediction is not implemented for this ensemble -> should throw
        @test_throws Exception SDM.predict(ensemble, backgrounddata; by_group = true)

        # Extra columns and column order do not matter
        extradata = merge(backgrounddata, (; d= backgrounddata.a))
        misordereddata = backgrounddata[(:b,:a,:c)]

        @test pr == SDM.predict(ensemble, extradata) == SDM.predict(ensemble, misordereddata)

        # Threading works
        @test pr == SDM.predict(ensemble, backgrounddata; threaded=  true)
    end

    @testset "to a RasterStack" begin
        ## predict to a RasterStack
        ds = (X(1:50), Y(1:50))
        rs = RasterStack((a = rand(ds), b = rand(ds), c = rand(ds)), missingval = 0.0)
        rs[1] = (a = 0, b = 0, c = 0)
        rs2 = replace_missing(rs)

        # set the first cell to missing across layers to simulate mask
        raspr = SDM.predict(ensemble, rs)
        raspr2 = SDM.predict(ensemble, rs2)

        # result should be a Raster (per-layer raster output) and should carry ensemble dims
        @test raspr isa Raster
        @test all(hasdim(raspr, (:model, :fold)))
        @test dims(raspr, (:model, :fold)) === dims(ensemble)
        @test dims(raspr, (X, Y)) === dims(rs)

        # masked cells should remain missing
        @test all(ismissing, raspr[X=1, Y=1])
        @test isequal(raspr, raspr2)

        # non-missing values should be probabilities in (0,1)
        vals = skipmissing(raspr)
        extr = extrema(collect(vals))
        @test extr[1] >= 0.0 && extr[2] <= 1.0
    end
end

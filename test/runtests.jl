TestEnv.activate()
using SpeciesDistributionModels, MLJBase, Tables
import SpeciesDistributionModels as SDM
import CategoricalArrays as CA
import StatisticalMeasures
using StableRNGs, Distributions, Test
using Makie
using Rasters

using MLJGLMInterface: LinearBinaryClassifier
using EvoTrees: EvoTreeClassifier
using MLJDecisionTreeInterface: RandomForestClassifier
import MLJTransforms: OneHotEncoder

rng = StableRNG(0)

# some mock data
n = 100
backgrounddata = (a = rand(rng, n), b = rand(rng, n), c = rand(rng, n))
presencedata = (a = rand(rng, n), b = rand(rng, n).^2, c = sqrt.(rand(rng, n)))
data = sdmdata(presencedata, backgrounddata; resampler = CV(nfolds = 5, shuffle = true))

include("tools.jl")
include("sdmdata.jl")
include("fit.jl")


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



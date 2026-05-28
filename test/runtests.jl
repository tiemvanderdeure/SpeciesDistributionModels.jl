using SpeciesDistributionModels, MLJBase, Tables
import SpeciesDistributionModels as SDM
import CategoricalArrays as CA
import StatisticalMeasures
using StableRNGs, Distributions, Test
using Rasters

using MLJGLMInterface: LinearBinaryClassifier
using MLJDecisionTreeInterface: RandomForestClassifier
using Maxnet: MaxnetBinaryClassifier
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
include("predict.jl")
include("plotting.jl")
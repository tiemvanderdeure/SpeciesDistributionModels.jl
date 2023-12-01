using Revise

using SpeciesDistributionModels, GLMakie
import SpeciesDistributionModels as SDM

presences = (a = rand(200), b = rand(200), c = rand(200))
background = (a = rand(200), b = sqrt.(rand(200)), c = rand(200).^2)

models = [
    SDM.linear_model(),
    SDM.boosted_regression_tree(),
    SDM.random_forest(),
    SDM.random_forest(; n_trees = 10, max_depth = 3)]

ensemble = sdm(presences, background, models, [SDM.MLJBase.CV(; shuffle = true)])

interactive_evaluation(ensemble)

shapley = shap(ensemble; n_samples = 5)

interactive_response_curves(shapley)

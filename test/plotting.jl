ensemble = sdm(data, (lm = LinearBinaryClassifier(), maxnet = MaxnetBinaryClassifier()))
evaluation = SDM.evaluate(ensemble; validation = (presencedata, backgrounddata))
expl = explain(ensemble; method = ShapleyValues(10; rng))

# plots
@testset "plotting" begin
    interactive_evaluation(ensemble, thresholds = 0:0.001:1);
    interactive_response_curves(expl);
    SDM.boxplot(evaluation, :auc);
end
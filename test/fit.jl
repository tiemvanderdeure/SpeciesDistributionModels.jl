using MLJGLMInterface: LinearBinaryClassifier
using EvoTrees: EvoTreeClassifier
using MLJDecisionTreeInterface: RandomForestClassifier
import MLJTransforms: OneHotEncoder

@testset "sdm" begin
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

    # plots
    interactive_evaluation(ensemble, thresholds = 0:0.001:1)
    interactive_response_curves(expl)
    boxplot(evaluation, :auc)
end

ensemble = sdm(data, (; lm = LinearBinaryClassifier()))

@testset "Evaluate" begin
    evaluation = SDM.evaluate(ensemble; validation = (presencedata, backgrounddata))
    evaluation2 = SDM.evaluate(ensemble; measures = (; auc = StatisticalMeasures.AreaUnderCurve()))
    @test evaluation isa SDM.SDMensembleEvaluation
    @test all(hasdim(evaluation, (:model, :fold, :dataset, :measure)))
    @test evaluation[1] isa NamedTuple{(:score, :threshold, :ensemble)}
    @test mach_evals.train isa NamedTuple{(keys(SDM.measures(evaluation)))}

    @test isequal(lookup(evaluation, :dataset), [:train, :test, :validation])
    @test isequal(lookup(evaluation2, :dataset), [:train, :test])

    @test isequal(lookup(evaluation, :measure), [:accuracy, :auc, :log_loss, :kappa])
    @test isequal(lookup(evaluation2, :measure), [:auc])

    @test evaluation[model = 1, fold = 1, dataset = At(:train), measure = At(:kappa)].threshold

end

evaluation[model = 1, fold = 1]

@testset "Explain" begin
    expl = explain(ensemble; method = ShapleyValues(10; rng))
    varimp = variable_importance(expl)
    @test varimp.b > varimp.a
    @test varimp.c > varimp.a
end
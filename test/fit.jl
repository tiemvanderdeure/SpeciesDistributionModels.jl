@testset "sdm" begin
    ## ensemble
    models = (
        rf = RandomForestClassifier(; rng),
        rf2 = OneHotEncoder() |> RandomForestClassifier(; max_depth = 3, rng),
        lm = LinearBinaryClassifier(),
        maxnet = MaxnetBinaryClassifier()
    )

    ensemble = sdm(data, models;
        threaded = false
    )

    @test ensemble isa SDM.SDMensemble
    @test ensemble[1] === ensemble[1,1] === ensemble[model = At(:rf), fold = At(1)]
    @test ensemble[1] isa Machine

    @test !DimensionalData.isordered(dims(ensemble, :model))
    @test all(hasdim(ensemble, (:model, :fold)))
    @test isequal(lookup(ensemble, :model), collect(keys(models)))
end

ensemble = sdm(data, (; lm = LinearBinaryClassifier()))

@testset "Evaluate" begin
    evaluation = SDM.evaluate(ensemble; validation = (presencedata, backgrounddata))
    evaluation2 = SDM.evaluate(ensemble; measures = (; auc = StatisticalMeasures.AreaUnderCurve()))

    # check the type
    @test evaluation isa SDM.SDMensembleEvaluation

    # indexing works
    @test evaluation[1] isa NamedTuple{(:score, :threshold, :ensemble)}
    @test evaluation[model = At(:lm), fold = At(1), dataset = At(:train), measure = At(:auc)] isa NamedTuple{(:score, :threshold, :ensemble)}

    # dimensions are correct
    @test all(hasdim(evaluation, (:model, :fold, :dataset, :measure)))
    @test dims(evaluation, (:model, :fold)) === dims(ensemble)

    @test isequal(lookup(evaluation, :dataset), [:train, :test, :validation])
    @test isequal(lookup(evaluation2, :dataset), [:train, :test])

    @test isequal(lookup(evaluation, :measure), [:accuracy, :auc, :log_loss, :kappa])
    @test isequal(lookup(evaluation2, :measure), [:auc])

    @test evaluation[model=1, fold=1, dataset=At(:train), measure=At(:kappa)].threshold isa Float64
    @test evaluation[model=1, fold=1, dataset=At(:train), measure=At(:auc)].threshold |> ismissing
end

@testset "Explain" begin
    expl = explain(ensemble; method = ShapleyValues(10; rng))

    # indexing works
    @test expl[1] isa NamedTuple{(:a, :b, :c)}
    @test expl[model = At(:lm), fold = At(1)] === expl[1]
    expl1 = expl[fold = At([1,2]), model = 1]

    # dimensions are correct
    @test dims(expl) === dims(ensemble)
    @test dims(expl1) == dims(sdm(expl1)) 
    @test refdims(expl1) == refdims(sdm(expl1)) 

    # Variable importance
    varimp = variable_importance(expl)
    @test varimp.b > varimp.a
    @test varimp.c > varimp.a
end

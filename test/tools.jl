
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
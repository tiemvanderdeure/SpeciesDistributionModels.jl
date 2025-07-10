# Convert a BitArray to a CategoricalArray. Faster and type-stable version of `categorical`
BooleanCategorical{N} = CategoricalArrays.CategoricalArray{Bool, N, UInt8} where N
function boolean_categorical(A::BitArray{N}) where N 
    BooleanCategorical{N}(A, levels=[false, true], ordered=false)
end
boolean_categorical(A::AbstractVector{Bool}) = boolean_categorical(BitArray(A))


struct SDMdata{K}
    predictor::NamedTuple
    response::CategoricalArrays.CategoricalArray
    geometry::Union{Nothing, Vector}
    traintestpairs::MLJBase.TrainTestPairs
    resampler::Union{Nothing, MLJBase.ResamplingStrategy}

    function SDMdata(predictor::P, response, geometry, traintestpairs, resampler) where P<:NamedTuple{K} where K
        new{K}(predictor, response, geometry, traintestpairs, resampler)
    end
end

function Base.show(io::IO, mime::MIME"text/plain", data::SDMdata{K}) where K
    y = response(data)
    print(io, "SDMdata object with ")
    printstyled(io, sum(y), bold = true)
    print(io, " presence points and ")
    printstyled(io, length(y) - sum(y), bold = true)
    print(io, " absence points. \n \n")

    printstyled(io, "Resampling: \n", bold = true)
    println(io, "Data is divided into $(nfolds(data)) folds using resampling strategy $(resampler(data)).")

    n_presences = length.(getindex.(traintestpairs(data), 1))
    n_absences = length.(getindex.(traintestpairs(data), 2))
    table_cols = hcat(1:nfolds(data), n_presences, n_absences)
    header = (["fold", "# train", "# test"])
    PrettyTables.pretty_table(io, table_cols; header = header)

    printstyled(io, "Predictor variables: \n", bold = true)
    Base.show(io, mime, MLJBase.schema(predictor(data)))

    if isnothing(geometry(data)) 
        print(io, "Does not contain geometry data")
    else
        print(io, "Also contains geometry data")
    end
end


_gettrainrows(d::SDMdata, i) = d.traintestpairs[i][1]
_gettestrows(d::SDMdata, i) = d.traintestpairs[i][2]
predictor(d::SDMdata) = d.predictor
predictorkeys(d::SDMdata{K}) where K = K 
response(d::SDMdata) = convert(AbstractArray{Bool}, d.response)
geometry(d::SDMdata) = d.geometry
traintestpairs(d::SDMdata) = d.traintestpairs
resampler(d::SDMdata) = d.resampler
nfolds(d::SDMdata) = length(d.traintestpairs)

# if input is two tables of presences and absences
function _sdmdata(presences, absences, resampler, predictorkeys)
    Tables.istable(presences) || throw(ArgumentError("Presences must be a Tables.jl-compatible table"))
    Tables.istable(absences) || throw(ArgumentError("Absences must be a Tables.jl-compatible table"))
    # check that both tables have the same number of rows
    p_columns = Tables.columns(presences)
    a_columns = Tables.columns(absences) 
    n_presence = Tables.rowcount(p_columns)
    n_absence = Tables.rowcount(a_columns)
    if isnothing(predictorkeys)
        predictorkeys = Tuple(Base.intersect(keys(p_columns), keys(a_columns)))
        length(predictorkeys) > 0 || throw(ArgumentError("Presence and absence data have no common variable names - can't fit the ensemble."))
    else
        if haskey(p_columns, :geometry) && haskey(a_columns, :geometry)
            predictorkeys = (predictorkeys..., :geometry)
        end
    end
    # merge presence and absence data into one namedtuple of vectors
    X = NamedTuple(var => [a_columns[var]; p_columns[var]] for var in predictorkeys)
    y = boolean_categorical([falses(n_absence); trues(n_presence)])
    
    _sdmdata(X, y, resampler)
end
# in case input is a table with bools for presence/absence
function _sdmdata(X, response::BitVector, resampler, predictorkeys)
    Tables.istable(X) || throw(ArgumentError("X must be a Tables.jl-compatible table"))
    _sdmdata(Tables.columntable(X), response, resampler, predictorkeys)
end
_sdmdata(X::Tables.ColumnTable, response::BitVector, resampler, ::Nothing) = 
    _sdmdata(X, response, resampler, Tables.columnnames(X))
function _sdmdata(X::Tables.ColumnTable, y::BitVector, resampler, predictorkeys::Tuple)
    Tables.rowcount(X) == length(y) || error("Number of rows in predictors and response do not match")
    predictorkeys = haskey(X, :geometry) ? (predictorkeys..., :geometry) : predictorkeys
    _sdmdata(X[predictorkeys], boolean_categorical(y), resampler)
end
function _sdmdata(
    X::Tables.ColumnTable, 
    y::BooleanCategorical, 
    resampler::MLJBase.ResamplingStrategy, 
)
    # hack to allways shuffle CV
    resampler = resampler isa CV ? ConstructionBase.setproperties(resampler, (; shuffle = true)) : resampler
    traintestpairs = MLJBase.train_test_pairs(resampler, eachindex(y), X, y)
    geometries = :geometry ∈ keys(X) ? Tables.getcolumn(X, :geometry) : nothing
    X = Base.structdiff(X, NamedTuple{(:geometry,)})
    SDMdata(X, y, geometries, traintestpairs, resampler)
end

cpu_backend(threaded) = threaded ? CPUThreads() : CPU1()
_map(::CPU1) = Base.map
_map(::CPUThreads) = ThreadsX.map


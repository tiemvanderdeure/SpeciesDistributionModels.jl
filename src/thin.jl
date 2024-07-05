"""
    thin(x, cutoff; distance = Haversine(), [rng])

    Thin spatial data by removing points that are closer than `cutoff` distance
    to the nearest other point in the dataset.

    ## Arguments
    - `x`: an `AbstractVector` that iterates points, or a table with a `:geometry` column.
    - `cutoff`: the distance threshold in units of `distance`.
    ## Keywords
    - `distance`: the distance metric used to calculate distances between points. The default
    is `Haversine()`, which uses the Haversine formula to calculate the distance between coordinates in meter units.
    - `rng`: a random number generator. The default is `Random.GLOBAL_RNG()`.

    ## Example
    ```jldoctest
    using SpeciesDistributionModels, Distances
    # a vector that iteratores points
    geometries = [(0,0), (1,0), (0,0.000001)]
    # thin to 1000 meters
    thin(geometries, 1000)
    # thin to 1 degree
    thin(geometries, 1; distance = Euclidean(), rng = Xoshiro(123))

    # output
    2-element Vector{Tuple{Int64, Real}}:
    (0, 0)
    (1, 0)
    ```
"""
function thin(x, cutoff; distance = Distances.Haversine(), rng = Random.GLOBAL_RNG)
    if Tables.istable(x)
        geoms = Tables.getcolumn(x, :geometry)
        indices = _thin(geoms, cutoff, distance, rng)
        return Tables.subset(x, indices)
    elseif x isa AbstractVector
        _geoms = unique(x)
        indices = _thin(_geoms, cutoff, distance, rng)
        return _geoms[indices]
    else
        throw(ArgumentError("x must be an AbstractVector or a Table with a :geometry column."))
    end
end

function _thin(geoms::AbstractVector, cutoff::Real, distance::Distances.Metric, rng::Random.AbstractRNG)
    dist_matrix = Distances.pairwise(distance, geoms)

    dist_mask = dist_matrix .< cutoff
    dist_sum = vec(sum(dist_mask; dims = 1))
    s = size(dist_sum, 1)

    indices = collect(1:s)

    for i in 1:s
        m = maximum(dist_sum)
        if m == 1
            break
        else
            drop = rand(rng, findall(dist_sum .== m))
            dist_sum .-= @view dist_mask[drop, :]
            drop_indices = 1:s .!= drop
            dist_mask = @view dist_mask[drop_indices, drop_indices]
            dist_sum = @view dist_sum[drop_indices]
            s -= 1
            indices = @view indices[drop_indices]
        end
    end
    return indices
end

"""
    thin([rng], x, cutoff; distance = Haversine(), geometrycolumn = nothing)

Thin spatial data by removing points that are closer than `cutoff` distance to the nearest
other point in the dataset.

## Arguments
- `rng`: a random number generator. Defaults to `Random.GLOBAL_RNG()`.
- `x`: an `AbstractVector` that iterates points, or a table with a `:geometry` column.
- `cutoff`: the distance threshold in units of `distance`.

## Keywords
- `distance`: the distance metric used to calculate distances between points. Defaults to
    `Haversine()`, which uses the Haversine formula to calculate distance in meters.
- `geometrycolumn`: the name of the column in the table that contains the points, if `x` is
    a table. Defaults to `:geometry`.

## Returns
- If `x` is a table: a subset of the table with thinned rows
- If `x` is an `AbstractVector`: a vector of points with nearby points removed

## Example
```jldoctest; setup = :(using Random; Random.seed!(123))
using SpeciesDistributionModels, Distances
# a vector that iterates points
geometries = [(0,0), (1,0), (0,0.000001)]
# thin to 1000 meters
thin(geometries, 1000)
# thin to 1 degree
thin(geometries, 1; distance = Euclidean())

# output
2-element Vector{Tuple{Int64, Real}}:
(0, 0)
(1, 0)
```
"""
thin(x, cutoff; kw...) = thin(Random.GLOBAL_RNG, x, cutoff; kw...)
function thin(rng::Random.AbstractRNG, x, cutoff; distance = Distances.Haversine(), geometrycolumn = nothing) # = first(GI.geometrycolumn(x))
    if !(x isa AbstractVector{<:GI.NamedTuplePoint}) && Tables.istable(x)
        geomcol = isnothing(geometrycolumn) ? first(GI.geometrycolumns(data)) : geometrycolumn
        geoms = Tables.getcolumn(Tables.Columns(x), geomcol)
        indices = _thin(rng, geoms, cutoff, distance)
        return Tables.subset(x, indices)
    elseif x isa AbstractVector
        geoms = unique(x)
        indices = _thin(rng, geoms, cutoff, distance)
        return geoms[indices]
    else
        throw(ArgumentError("x must be an AbstractVector or a Table with a :$geometrycolumn column."))
    end
end

function _thin(rng::Random.AbstractRNG, geoms::AbstractVector, cutoff::Real, distance::Distances.Metric)
    dist_matrix = Distances.pairwise(distance, geoms)
    dist_mask = dist_matrix .< cutoff
    dist_sum = vec(sum(dist_mask; dims = 1))
    l = length(dist_sum)
    keep_mask = trues(l)

    for _ in 1:l
        m = 1
        for i in 1:l
            if keep_mask[i]
                m = max(m, dist_sum[i])
            end
        end
        if m == 1
            break
        else
            drop = rand(rng, findall(dist_sum .== m))
            keep_mask[drop] = false
            for j in 1:l
                if keep_mask[j] && dist_mask[j, drop]
                    dist_sum[j] -= 1
                end
            end
        end
    end
    
    return keep_mask
end

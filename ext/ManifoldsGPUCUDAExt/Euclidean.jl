using LinearAlgebra

"""
    exp!(M::PowerManifold{ℝ, <:Euclidean, ...}, q, p, X)

GPU-native exponential on batched Euclidean space.
Strategy: flat manifold — geodesic is q = p + X (pure broadcasting).
"""
function ManifoldsBase.exp!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, N},
        p::CuArray{T, N},
        X::CuArray{T, N},
    ) where {T <: Real, N}
    q .= p .+ X
    return q
end

"""
    log!(M::PowerManifold{ℝ, <:Euclidean, ...}, X, p, q)

GPU-native logarithm on batched Euclidean space.
Strategy: flat manifold — log is X = q - p (pure broadcasting).
"""
function ManifoldsBase.log!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        X::CuArray{T, N},
        p::CuArray{T, N},
        q::CuArray{T, N},
    ) where {T <: Real, N}
    X .= q .- p
    return X
end

"""
    distance(M::PowerManifold{ℝ, <:Euclidean, ...}, p, q)

GPU-native distance on batched Euclidean space.
Strategy: fused broadcast avoids temp array from default log + norm chain.
"""
function ManifoldsBase.distance(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, N},
        q::CuArray{T, N},
    ) where {T <: Real, N}
    return sqrt(sum((p .- q) .^ 2))
end

"""
    inner(M::PowerManifold{ℝ, <:Euclidean, ...}, p, X, Y)

GPU-native inner product on batched Euclidean space.
Strategy: fused broadcast dot product, summed over all dimensions.
"""
function ManifoldsBase.inner(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        ::CuArray{T, N},
        X::CuArray{T, N},
        Y::CuArray{T, N},
    ) where {T <: Real, N}
    return sum(X .* Y)
end

"""
    norm(M::PowerManifold{ℝ, <:Euclidean, ...}, p, X)

GPU-native norm on batched Euclidean space.
Strategy: fused broadcast, avoids intermediate from default sqrt(inner(...)).
"""
function LinearAlgebra.norm(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        ::CuArray{T, N},
        X::CuArray{T, N},
    ) where {T <: Real, N}
    return sqrt(sum(X .^ 2))
end

"""
    parallel_transport_to!(M::PowerManifold{ℝ, <:Euclidean, ...}, Y, p, X, q)

GPU-native parallel transport on batched Euclidean space.
Strategy: identity transport (Y .= X), avoids PowerManifold per-element loop.
"""
function ManifoldsBase.parallel_transport_to!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        Y::CuArray{T, N},
        ::CuArray{T, N},
        X::CuArray{T, N},
        ::CuArray{T, N},
    ) where {T <: Real, N}
    Y .= X
    return Y
end

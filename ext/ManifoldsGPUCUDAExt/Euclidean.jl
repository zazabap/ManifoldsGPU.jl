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

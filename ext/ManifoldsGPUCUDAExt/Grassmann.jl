function ManifoldsBase.inner(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        Y::CuArray{T, 3},
    ) where {T <: Real}
    return dot(X, Y)
end

function ManifoldsBase.norm(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Real}
    return sqrt(dot(X, X))
end

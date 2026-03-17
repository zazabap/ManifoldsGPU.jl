function ManifoldsBase.inner(
        ::PowerManifold{ℝ, <:Sphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T},
        X::CuArray{T},
        Y::CuArray{T},
    ) where {T <: Real}
    return dot(X, Y)
end

function ManifoldsBase.norm(
        ::PowerManifold{ℝ, <:Sphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T},
        X::CuArray{T},
    ) where {T <: Real}
    return sqrt(dot(X, X))
end

using LinearAlgebra

"""
    exp!(M::PowerManifold{..., <:GeneralUnitaryMatrices}, q, p, X)

GPU-accelerated exponential map for batched `GeneralUnitaryMatrices` (including
`UnitaryMatrices` and `OrthogonalMatrices`).

Computes `q .= p * exp(X)` using the on-device Taylor-based `_matrix_exp_gpu`
for the matrix exponential, followed by a single `gemm_strided_batched` call
for the batched matrix multiply.
"""
function ManifoldsBase.exp!(
        M::PowerManifold{
            𝔽,
            <:Manifolds.GeneralUnitaryMatrices{<:Any, 𝔽},
            <:Tuple,
            ArrayPowerRepresentation,
        },
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {𝔽, T <: Number}
    E = _matrix_exp_gpu(X)
    q .= CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, E)
    return q
end

"""
    retract_polar_fused!(M::PowerManifold{..., <:GeneralUnitaryMatrices}, q, p, X, t)

GPU-accelerated polar retraction for batched `GeneralUnitaryMatrices`.

Computes `q = p + p * (t * X)` then projects onto the manifold via the polar
factor of the SVD: `q = U * V'`. Uses `gesvdj!` for batched on-device SVD,
with a CPU fallback for matrices exceeding cuSOLVER size limits.
"""
function ManifoldsBase.retract_polar_fused!(
        ::PowerManifold{
            𝔽,
            <:Manifolds.GeneralUnitaryMatrices{<:Any, 𝔽},
            <:Tuple,
            ArrayPowerRepresentation,
        },
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
    ) where {𝔽, T <: Number}
    q .= p .+ CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, T(t) .* X)

    # SVD polar factor: q = U * V'
    try
        U, _, V = CUDA.CUSOLVER.gesvdj!('V', q)
        q .= CUDA.CUBLAS.gemm_strided_batched('N', 'C', U, V)
    catch e
        if e isa ArgumentError
            # CPU fallback: gesvdj! fails for matrices larger than supported size
            batch = size(q, 3)
            for i in 1:batch
                q_i = copy(Array(@view q[:, :, i]))
                s = svd!(q_i)
                @view(q[:, :, i]) .= s.U * s.Vt
            end
        else
            rethrow()
        end
    end

    return q
end

"""
    retract_fused!(M::PowerManifold{..., <:GeneralUnitaryMatrices}, q, p, X, t, ::PolarRetraction)

Dispatches to [`retract_polar_fused!`](@ref) for `PolarRetraction` on GPU.
"""
function ManifoldsBase.retract_fused!(
        M::PowerManifold{
            𝔽,
            <:Manifolds.GeneralUnitaryMatrices{<:Any, 𝔽},
            <:Tuple,
            ArrayPowerRepresentation,
        },
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
        ::PolarRetraction,
    ) where {𝔽, T <: Number}
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, t)
end

"""
    project!(M::PowerManifold{..., <:GeneralUnitaryMatrices{..., AbsoluteDeterminantOneMatrixType}}, q, p)

GPU-accelerated point projection for batched `OrthogonalMatrices` / `UnitaryMatrices`
(manifolds with `AbsoluteDeterminantOneMatrixType`).

Projects an arbitrary matrix onto the manifold via the polar factor of the SVD:
`q = U * V'`. Uses `gesvdj!` for batched on-device SVD, with a CPU fallback
for matrices exceeding cuSOLVER size limits.
"""
function ManifoldsBase.project!(
        ::PowerManifold{
            𝔽,
            <:Manifolds.GeneralUnitaryMatrices{
                <:Any,
                𝔽,
                Manifolds.AbsoluteDeterminantOneMatrixType,
            },
            <:Tuple,
            ArrayPowerRepresentation,
        },
        q::CuArray{T, 3},
        p::CuArray{T, 3},
    ) where {𝔽, T <: Number}
    q .= p

    # SVD polar factor: q = U * V'
    try
        U, _, V = CUDA.CUSOLVER.gesvdj!('V', q)
        q .= CUDA.CUBLAS.gemm_strided_batched('N', 'C', U, V)
    catch e
        if e isa ArgumentError
            # CPU fallback: gesvdj! fails for matrices larger than supported size
            batch = size(q, 3)
            for i in 1:batch
                q_i = copy(Array(@view q[:, :, i]))
                s = svd!(q_i)
                @view(q[:, :, i]) .= s.U * s.Vt
            end
        else
            rethrow()
        end
    end

    return q
end

"""
    project!(M::PowerManifold{..., <:GeneralUnitaryMatrices}, Y, p, X)

GPU-accelerated tangent vector projection for batched `GeneralUnitaryMatrices`.

Computes the skew-symmetric (skew-Hermitian) part of `p' * X`:
`Y = (p' * X - X' * p) / 2` using two `gemm_strided_batched` calls.
"""
function ManifoldsBase.project!(
        ::PowerManifold{
            𝔽,
            <:Manifolds.GeneralUnitaryMatrices{<:Any, 𝔽},
            <:Tuple,
            ArrayPowerRepresentation,
        },
        Y::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {𝔽, T <: Number}
    A = CUDA.CUBLAS.gemm_strided_batched('C', 'N', p, X)    # p' * X
    B = CUDA.CUBLAS.gemm_strided_batched('C', 'N', X, p)    # X' * p
    Y .= (A .- B) ./ 2
    return Y
end

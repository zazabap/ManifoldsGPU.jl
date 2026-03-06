function _matrix_exp_gpu(A::CuArray{T, 2}) where {T <: Real}
    E = _matrix_exp_gpu(reshape(A, size(A, 1), size(A, 2), 1))
    return reshape(E, size(A))
end

function _matrix_exp_gpu(A::CuArray{T, 2}) where {T <: Complex}
    E = _matrix_exp_gpu(reshape(A, size(A, 1), size(A, 2), 1))
    return reshape(E, size(A))
end

function _matrix_exp_gpu(A::CuArray{T, 3}) where {T <: Real}
    n, m, batch = size(A)
    n == m || throw(DimensionMismatch("matrix exponential requires square matrices, got ($n, $m, $batch)"))

    I_n = reshape(CuArray(Matrix{T}(I, n, n)), n, n, 1)
    I_batch = similar(A)
    I_batch .= I_n

    maxabs = maximum(abs, A)
    theta = T <: Float32 ? T(1) : T(1 // 2)
    s = max(0, ceil(Int, log2(float(maxabs * n / theta))))
    As = A ./ T(2)^s

    order = T <: Float32 ? 18 : 30
    E = copy(I_batch)
    term = copy(I_batch)
    for j in 1:order
        term = CUDA.CUBLAS.gemm_strided_batched('N', 'N', term, As)
        term ./= T(j)
        E .+= term
    end

    for _ in 1:s
        E = CUDA.CUBLAS.gemm_strided_batched('N', 'N', E, E)
    end

    return E
end

function _matrix_exp_gpu(A::CuArray{T, 3}) where {T <: Complex}
    n, m, batch = size(A)
    n == m ||
        throw(DimensionMismatch("matrix exponential requires square matrices, got ($n, $m, $batch)"))

    RT = real(T)
    I_n = reshape(CuArray(Matrix{T}(I, n, n)), n, n, 1)
    I_batch = similar(A)
    I_batch .= I_n

    maxabs = maximum(abs, A)
    theta = RT <: Float32 ? RT(1) : RT(1 // 2)
    s = max(0, ceil(Int, log2(float(real(maxabs) * n / theta))))
    As = A ./ T(2)^s

    order = RT <: Float32 ? 18 : 30
    E = copy(I_batch)
    term = copy(I_batch)
    for j in 1:order
        term = CUDA.CUBLAS.gemm_strided_batched('N', 'N', term, As)
        term ./= T(j)
        E .+= term
    end

    for _ in 1:s
        E = CUDA.CUBLAS.gemm_strided_batched('N', 'N', E, E)
    end

    return E
end

"""
    _matrix_log_gpu(A::CuArray{T, 2})
    _matrix_log_gpu(A::CuArray{T, 3})

GPU matrix logarithm via per-slice eigendecomposition.

For each slice `A[:,:,i]`, computes `eigen(A_i)` via cuSOLVER `geev!`, then
reconstructs `log(A_i) = V * Diag(log(λ)) * V⁻¹`. Real inputs are promoted
to complex for eigendecomposition, then the real part is taken.

Note: No batched `geev!` exists in cuSOLVER, so this loops over the batch
dimension with sequential kernel launches (all computation stays on GPU).
"""
function _matrix_log_gpu(A::CuArray{T, 2}) where {T <: Real}
    L = _matrix_log_gpu(reshape(A, size(A, 1), size(A, 2), 1))
    return reshape(L, size(A))
end

function _matrix_log_gpu(A::CuArray{T, 2}) where {T <: Complex}
    L = _matrix_log_gpu(reshape(A, size(A, 1), size(A, 2), 1))
    return reshape(L, size(A))
end

function _matrix_log_gpu(A::CuArray{T, 3}) where {T <: Complex}
    n, m, batch = size(A)
    n == m ||
        throw(DimensionMismatch("matrix logarithm requires square matrices, got ($n, $m, $batch)"))
    result = similar(A)
    for i in 1:batch
        A_i = A[:, :, i]
        F = eigen(A_i)
        log_slice = F.vectors * Diagonal(log.(F.values)) * inv(F.vectors)
        result[:, :, i] .= log_slice
    end
    return result
end

function _matrix_log_gpu(A::CuArray{T, 3}) where {T <: Real}
    CT = complex(T)
    Ac = CuArray{CT}(A)
    logAc = _matrix_log_gpu(Ac)
    return real.(logAc)
end

using CUDA
using Manifolds, ManifoldsBase, Random, Test

@testset "CUDA: Stiefel exp Float64" begin
    Random.seed!(45)
    M = Stiefel(4, 2)
    MP = PowerManifold(M, 5)

    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test is_point(MP, Array(Y_cu))
    @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "CUDA: Stiefel exp batched stress Float64" begin
    Random.seed!(42)
    M = Stiefel(8, 4)
    MP = PowerManifold(M, 64)

    for _ in 1:6
        p = rand(MP)
        X = 0.25 * rand(MP; vector_at = p)
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)
        Y_cu_h = Array(Y_cu)

        @test is_point(MP, Y_cu_h)
        @test isapprox(MP, p, Y_cu_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end
end

@testset "CUDA: Stiefel exp batched stress Float32" begin
    Random.seed!(43)
    M = Stiefel(8, 4)
    MP = PowerManifold(M, 64)

    for _ in 1:6
        p = Float32.(rand(MP))
        X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)
        Y_cu_h = Array(Y_cu)

        @test is_point(MP, Y_cu_h)
        @test isapprox(MP, p, Y_cu_h, Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end
end

@testset "CUDA: Stiefel PolarRetraction Float64" begin
    Random.seed!(46)
    M = Stiefel(8, 4)
    MP = PowerManifold(M, 64)
    t = 0.3

    p = rand(MP)
    X = rand(MP; vector_at = p)
    q_cpu = similar(p)
    ManifoldsBase.retract_fused!(MP, q_cpu, p, X, t, PolarRetraction())

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    q_cu = similar(p_cu)
    ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, PolarRetraction())
    q_cu_h = Array(q_cu)

    @test is_point(MP, q_cu_h)
    @test isapprox(MP, p, q_cu_h, q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "CUDA: Stiefel PolarRetraction Float32" begin
    Random.seed!(47)
    M = Stiefel(8, 4)
    MP = PowerManifold(M, 64)
    t = Float32(0.3)

    p = Float32.(rand(MP))
    X = Float32.(rand(MP; vector_at = p))
    q_cpu = similar(p)
    ManifoldsBase.retract_fused!(MP, q_cpu, p, X, t, PolarRetraction())

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    q_cu = similar(p_cu)
    ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, PolarRetraction())
    q_cu_h = Array(q_cu)

    @test is_point(MP, q_cu_h)
    @test isapprox(MP, p, q_cu_h, q_cpu; atol = 2.0f-5, rtol = 2.0f-5)
end

@testset "CUDA: Stiefel PolarRetraction fallback stress" begin
    Random.seed!(48)
    M = Stiefel(48, 16)
    MP = PowerManifold(M, 8)
    t = 0.2

    for _ in 1:4
        p = rand(MP)
        X = 0.2 * rand(MP; vector_at = p)
        q_cpu = similar(p)
        ManifoldsBase.retract_fused!(MP, q_cpu, p, X, t, PolarRetraction())

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        q_cu = similar(p_cu)
        ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, PolarRetraction())
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        # gesvdj! fails for n>32; falls back to per-slice svd! via cuSOLVER.
        # Slightly looser tolerance than the batched GPU path.
        @test isapprox(MP, p, q_cu_h, q_cpu; atol = 2.0e-12, rtol = 2.0e-12)
    end
end

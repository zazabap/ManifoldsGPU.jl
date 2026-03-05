using CUDA
using Manifolds, ManifoldsBase, Random, Test

@testset "CUDA: Grassmann exp Float64" begin
    Random.seed!(50)
    M = Grassmann(6, 3)
    MP = PowerManifold(M, 32)

    p = rand(MP)
    X = 0.25 .* rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test is_point(MP, Array(Y_cu))
    # Grassmann points are subspaces (equivalence classes); use 3-arg isapprox
    # (manifold-aware, checks geodesic distance) rather than element-wise comparison.
    # CPU LAPACK and GPU cuSOLVER Jacobi may produce different n×k representatives
    # of the same subspace when singular values are nearly degenerate.
    @test isapprox(MP, Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "CUDA: Grassmann exp Float32" begin
    Random.seed!(51)
    M = Grassmann(6, 3)
    MP = PowerManifold(M, 32)

    p = Float32.(rand(MP))
    X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test is_point(MP, Array(Y_cu))
    # Grassmann: 3-arg isapprox for subspace comparison (see Float64 testset comment).
    @test isapprox(MP, Array(Y_cu), Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
end

@testset "CUDA: Grassmann PolarRetraction" begin
    Random.seed!(52)
    M = Grassmann(6, 3)
    MP = PowerManifold(M, 32)
    t = 0.3

    p = rand(MP)
    X = rand(MP; vector_at = p)
    q_cpu = similar(p)
    ManifoldsBase.retract_fused!(MP, q_cpu, p, X, t, PolarRetraction())

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    q_cu = similar(p_cu)
    ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, PolarRetraction())

    @test is_point(MP, Array(q_cu))
    @test isapprox(MP, p, Array(q_cu), q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "CUDA: Grassmann exp stress Float64" begin
    Random.seed!(53)
    M = Grassmann(8, 4)
    MP = PowerManifold(M, 128)

    for _ in 1:4
        p = rand(MP)
        X = 0.25 .* rand(MP; vector_at = p)
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)

        @test is_point(MP, Array(Y_cu))
        # Grassmann: 3-arg isapprox for subspace comparison (see exp Float64 comment).
        @test isapprox(MP, Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end
end

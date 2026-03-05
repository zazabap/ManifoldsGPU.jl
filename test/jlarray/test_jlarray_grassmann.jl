using JLArrays, GPUArrays
using Manifolds, ManifoldsBase, Random, Test

GPUArrays.allowscalar(true)

@testset "JLArray: Grassmann exp Float64" begin
    M = Grassmann(6, 3)
    MP = PowerManifold(M, 8)

    Random.seed!(50)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test is_point(MP, Array(Y_jl))
    # Grassmann points are subspaces; use 3-arg isapprox (manifold-aware, checks geodesic
    # distance). CPU and GPU SVD may produce different orthonormal representatives of the
    # same subspace when singular values are nearly degenerate.
    @test isapprox(MP, Array(Y_jl), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "JLArray: Grassmann exp Float32" begin
    M = Grassmann(6, 3)
    MP = PowerManifold(M, 8)

    Random.seed!(51)
    p = Float32.(rand(MP))
    X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test is_point(MP, Array(Y_jl))
    # Grassmann: 3-arg isapprox for subspace comparison (see Float64 testset comment).
    @test isapprox(MP, Array(Y_jl), Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
end

@testset "JLArray: Grassmann PolarRetraction" begin
    M = Grassmann(6, 3)
    MP = PowerManifold(M, 8)
    t = 0.3

    Random.seed!(52)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    q_cpu = similar(p)
    ManifoldsBase.retract_fused!(MP, q_cpu, p, X, t, PolarRetraction())

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    q_jl = similar(p_jl)
    ManifoldsBase.retract_fused!(MP, q_jl, p_jl, X_jl, t, PolarRetraction())

    @test is_point(MP, Array(q_jl))
    @test isapprox(MP, p, Array(q_jl), q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

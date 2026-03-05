using JLArrays, GPUArrays
using Manifolds, ManifoldsBase, Random, Test

# Allow scalar indexing: CPU fallback implementations in Manifolds.jl use scalar
# operations (e.g. hvcat in StiefelEuclideanMetric.jl). This does not mask any
# GPU-correctness issue because CuArray paths use dedicated GPU kernels.
GPUArrays.allowscalar(true)

@testset "JLArray: Stiefel exp" begin
    M = Stiefel(4, 2)
    MP = PowerManifold(M, 5)

    Random.seed!(42)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test is_point(MP, Array(Y_jl))
    @test isapprox(MP, p, Array(Y_jl), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "JLArray: Stiefel exp Float32" begin
    M = Stiefel(8, 4)
    MP = PowerManifold(M, 16)

    Random.seed!(43)
    p = Float32.(rand(MP))
    X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test is_point(MP, Array(Y_jl))
    @test isapprox(MP, p, Array(Y_jl), Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
end

@testset "JLArray: Stiefel PolarRetraction" begin
    M = Stiefel(8, 4)
    MP = PowerManifold(M, 16)
    t = 0.3

    Random.seed!(44)
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

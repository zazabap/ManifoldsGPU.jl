using CUDA
using Manifolds, ManifoldsBase, Random, Test

@testset "CUDA: SPD exp Float64" begin
    Random.seed!(90)
    M = SymmetricPositiveDefinite(4)
    MP = PowerManifold(M, 16)

    p = rand(MP)
    X = 0.2 .* rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test is_point(MP, Array(Y_cu))
    @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0e-12, rtol = 2.0e-12)
end

@testset "CUDA: SPD exp Float32" begin
    Random.seed!(91)
    M = SymmetricPositiveDefinite(4)
    MP = PowerManifold(M, 16)

    p = Float32.(rand(MP))
    X = Float32(0.2) .* Float32.(rand(MP; vector_at = p))
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test is_point(MP, Array(Y_cu))
    @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
end

using CUDA
using Manifolds, ManifoldsBase, Random, Test

@testset "CUDA: UnitaryMatrices exp ComplexF64" begin
    Random.seed!(80)
    M = UnitaryMatrices(3)
    MP = PowerManifold(M, 32)

    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test is_point(MP, Array(Y_cu))
    @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0e-12, rtol = 2.0e-12)
end

@testset "CUDA: UnitaryMatrices exp ComplexF32" begin
    Random.seed!(81)
    M = UnitaryMatrices(3)
    MP = PowerManifold(M, 32)

    p = ComplexF32.(rand(MP))
    X = ComplexF32.(rand(MP; vector_at = p))
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test is_point(MP, Array(Y_cu))
    @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0f-4, rtol = 2.0f-4)
end

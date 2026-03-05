using CUDA
using Manifolds, ManifoldsBase, Random, Test

@testset "CUDA: Euclidean exp Float64" begin
    Random.seed!(70)
    MP = PowerManifold(Euclidean(4), 1024)

    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test isapprox(Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "CUDA: Euclidean log Float64" begin
    Random.seed!(71)
    MP = PowerManifold(Euclidean(4), 1024)

    p = rand(MP)
    q = rand(MP)
    X_cpu = log(MP, p, q)

    p_cu = CuArray(p)
    q_cu = CuArray(q)
    X_cu = log(MP, p_cu, q_cu)

    @test isapprox(Array(X_cu), X_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

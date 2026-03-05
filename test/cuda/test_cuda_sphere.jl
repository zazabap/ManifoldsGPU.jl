using CUDA
using Manifolds, ManifoldsBase, Random, Test

@testset "CUDA: Sphere exp Float64" begin
    Random.seed!(60)
    M = Sphere(3)
    MP = PowerManifold(M, 256)

    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test is_point(MP, Array(Y_cu))
    @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "CUDA: Sphere exp Float32" begin
    Random.seed!(61)
    M = Sphere(3)
    MP = PowerManifold(M, 256)

    p = Float32.(rand(MP))
    X = Float32(0.5) .* Float32.(rand(MP; vector_at = p))
    Y_cpu = exp(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = exp(MP, p_cu, X_cu)

    @test is_point(MP, Array(Y_cu))
    @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
end

@testset "CUDA: Sphere log Float64" begin
    Random.seed!(62)
    M = Sphere(3)
    MP = PowerManifold(M, 256)

    p = rand(MP)
    q = rand(MP)
    X_cpu = log(MP, p, q)

    p_cu = CuArray(p)
    q_cu = CuArray(q)
    X_cu = log(MP, p_cu, q_cu)

    @test isapprox(Array(X_cu), X_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

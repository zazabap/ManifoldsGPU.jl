using CUDA
using Manifolds, ManifoldsBase, LinearAlgebra, Random, Test

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

@testset "CUDA: Euclidean distance Float64" begin
    Random.seed!(72)
    MP = PowerManifold(Euclidean(4), 1024)

    p = rand(MP)
    q = rand(MP)
    d_cpu = distance(MP, p, q)

    p_cu = CuArray(p)
    q_cu = CuArray(q)
    d_cu = distance(MP, p_cu, q_cu)

    @test isapprox(d_cu, d_cpu; atol = 1.0e-12, rtol = 1.0e-12)
end

@testset "CUDA: Euclidean inner Float64" begin
    Random.seed!(73)
    MP = PowerManifold(Euclidean(4), 1024)

    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y = rand(MP; vector_at = p)
    val_cpu = inner(MP, p, X, Y)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    Y_cu = CuArray(Y)
    val_cu = inner(MP, p_cu, X_cu, Y_cu)

    @test isapprox(val_cu, val_cpu; atol = 1.0e-12, rtol = 1.0e-12)
end

@testset "CUDA: Euclidean norm Float64" begin
    Random.seed!(74)
    MP = PowerManifold(Euclidean(4), 1024)

    p = rand(MP)
    X = rand(MP; vector_at = p)
    val_cpu = norm(MP, p, X)

    p_cu = CuArray(p)
    X_cu = CuArray(X)
    val_cu = norm(MP, p_cu, X_cu)

    @test isapprox(val_cu, val_cpu; atol = 1.0e-12, rtol = 1.0e-12)
end

@testset "CUDA: Euclidean parallel_transport_to Float64" begin
    Random.seed!(75)
    MP = PowerManifold(Euclidean(4), 1024)

    p = rand(MP)
    q = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = parallel_transport_to(MP, p, X, q)

    p_cu = CuArray(p)
    q_cu = CuArray(q)
    X_cu = CuArray(X)
    Y_cu = similar(X_cu)
    parallel_transport_to!(MP, Y_cu, p_cu, X_cu, q_cu)

    @test isapprox(Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

using JLArrays, GPUArrays
using Manifolds, ManifoldsBase, LinearAlgebra, Random, Test

GPUArrays.allowscalar(true)

@testset "JLArray: Euclidean exp Float64" begin
    MP = PowerManifold(Euclidean(4), 32)

    Random.seed!(70)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test isapprox(Array(Y_jl), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "JLArray: Euclidean log Float64" begin
    MP = PowerManifold(Euclidean(4), 32)

    Random.seed!(71)
    p = rand(MP)
    q = rand(MP)
    X_cpu = log(MP, p, q)

    p_jl = JLArray(p)
    q_jl = JLArray(q)
    X_jl = log(MP, p_jl, q_jl)

    @test isapprox(Array(X_jl), X_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "JLArray: Euclidean distance Float64" begin
    MP = PowerManifold(Euclidean(4), 32)

    Random.seed!(72)
    p = rand(MP)
    q = rand(MP)
    d_cpu = distance(MP, p, q)

    p_jl = JLArray(p)
    q_jl = JLArray(q)
    d_jl = distance(MP, p_jl, q_jl)

    @test isapprox(d_jl, d_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "JLArray: Euclidean inner Float64" begin
    MP = PowerManifold(Euclidean(4), 32)

    Random.seed!(73)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y = rand(MP; vector_at = p)
    val_cpu = inner(MP, p, X, Y)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = JLArray(Y)
    val_jl = inner(MP, p_jl, X_jl, Y_jl)

    @test isapprox(val_jl, val_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "JLArray: Euclidean norm Float64" begin
    MP = PowerManifold(Euclidean(4), 32)

    Random.seed!(74)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    val_cpu = norm(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    val_jl = norm(MP, p_jl, X_jl)

    @test isapprox(val_jl, val_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "JLArray: Euclidean parallel_transport_to Float64" begin
    MP = PowerManifold(Euclidean(4), 32)

    Random.seed!(75)
    p = rand(MP)
    q = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = parallel_transport_to(MP, p, X, q)

    p_jl = JLArray(p)
    q_jl = JLArray(q)
    X_jl = JLArray(X)
    Y_jl = similar(X_jl)
    parallel_transport_to!(MP, Y_jl, p_jl, X_jl, q_jl)

    @test isapprox(Array(Y_jl), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

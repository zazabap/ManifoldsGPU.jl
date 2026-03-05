using JLArrays, GPUArrays
using Manifolds, ManifoldsBase, Random, Test

GPUArrays.allowscalar(true)

@testset "JLArray: Sphere exp Float64" begin
    M = Sphere(3)
    MP = PowerManifold(M, 16)

    Random.seed!(60)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test is_point(MP, Array(Y_jl))
    @test isapprox(MP, p, Array(Y_jl), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "JLArray: Sphere exp Float32" begin
    M = Sphere(3)
    MP = PowerManifold(M, 16)

    Random.seed!(61)
    p = Float32.(rand(MP))
    X = Float32(0.5) .* Float32.(rand(MP; vector_at = p))
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test is_point(MP, Array(Y_jl))
    @test isapprox(MP, p, Array(Y_jl), Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
end

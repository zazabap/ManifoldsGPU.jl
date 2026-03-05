using JLArrays, GPUArrays
using Manifolds, ManifoldsBase, Random, Test

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

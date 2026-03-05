using JLArrays, GPUArrays
using Manifolds, ManifoldsBase, Random, Test

GPUArrays.allowscalar(true)

@testset "JLArray: UnitaryMatrices exp ComplexF64" begin
    M = UnitaryMatrices(3)
    MP = PowerManifold(M, 8)

    Random.seed!(80)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test is_point(MP, Array(Y_jl))
    @test isapprox(MP, p, Array(Y_jl), Y_cpu; atol = 2.0e-12, rtol = 2.0e-12)
end

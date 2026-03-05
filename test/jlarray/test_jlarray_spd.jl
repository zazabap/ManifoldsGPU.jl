using JLArrays, GPUArrays
using Manifolds, ManifoldsBase, Random, Test

GPUArrays.allowscalar(true)

@testset "JLArray: SPD exp Float64" begin
    M = SymmetricPositiveDefinite(4)
    MP = PowerManifold(M, 8)

    Random.seed!(90)
    p = rand(MP)
    X = 0.2 .* rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    # No is_point check: the CPU Manifolds.jl exp for SPD produces ~O(eps) asymmetry
    # that fails Manifolds.jl's strict symmetry tolerance. The GPU implementation
    # (SPD.jl) symmetrizes explicitly via 0.5*(result + result').
    @test isapprox(MP, p, Array(Y_jl), Y_cpu; atol = 2.0e-12, rtol = 2.0e-12)
end

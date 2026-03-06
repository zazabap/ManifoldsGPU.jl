using GPUArrays

@testset "Euclidean JLArray" begin
    # JLArray tests verify numerical correctness without GPU hardware.
    # Scalar indexing is allowed because the CuArray-specific overrides
    # do not dispatch on JLArray; the default ManifoldsBase PowerManifold
    # path loops per element. CUDA tests verify no scalar indexing.
    GPUArrays.allowscalar(true)

    @testset "exp! and log!" begin
        Random.seed!(50)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        X = randn(8, 4, 64)

        Y_cpu = exp(MP, p, X)

        p_jl = JLArray(p)
        X_jl = JLArray(X)
        Y_jl = exp(MP, p_jl, X_jl)
        @test is_point(MP, Array(Y_jl))
        @test isapprox(Array(Y_jl), Y_cpu; atol = 2.0e-14)

        V_cpu = log(MP, p, Y_cpu)

        V_jl = log(MP, p_jl, JLArray(Y_cpu))
        @test isapprox(Array(V_jl), V_cpu; atol = 2.0e-14)
    end

    @testset "exp! and log! Float32" begin
        Random.seed!(51)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = Float32.(randn(8, 4, 64))
        X = Float32.(randn(8, 4, 64))

        Y_cpu = exp(MP, p, X)

        p_jl = JLArray(p)
        X_jl = JLArray(X)
        Y_jl = exp(MP, p_jl, X_jl)
        @test is_point(MP, Array(Y_jl))
        @test isapprox(Array(Y_jl), Y_cpu; atol = 2.0f-5)

        V_cpu = log(MP, p, Y_cpu)

        V_jl = log(MP, p_jl, JLArray(Y_cpu))
        @test isapprox(Array(V_jl), V_cpu; atol = 2.0f-5)
    end

    @testset "distance, inner, norm" begin
        Random.seed!(52)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        q = randn(8, 4, 64)
        X = randn(8, 4, 64)
        Y = randn(8, 4, 64)

        p_jl = JLArray(p)
        q_jl = JLArray(q)
        X_jl = JLArray(X)
        Y_jl = JLArray(Y)

        d_cpu = distance(MP, p, q)
        d_jl = distance(MP, p_jl, q_jl)
        @test isapprox(d_jl, d_cpu; atol = 1.0e-12)

        v_cpu = inner(MP, p, X, Y)
        v_jl = inner(MP, p_jl, X_jl, Y_jl)
        @test isapprox(v_jl, v_cpu; atol = 1.0e-12)

        n_cpu = norm(MP, p, X)
        n_jl = norm(MP, p_jl, X_jl)
        @test isapprox(n_jl, n_cpu; atol = 1.0e-12)
    end

    @testset "parallel_transport_to!" begin
        Random.seed!(53)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        q = randn(8, 4, 64)
        X = randn(8, 4, 64)

        Z_cpu = similar(X)
        parallel_transport_to!(MP, Z_cpu, p, X, q)

        p_jl = JLArray(p)
        q_jl = JLArray(q)
        X_jl = JLArray(X)
        Z_jl = similar(X_jl)
        parallel_transport_to!(MP, Z_jl, p_jl, X_jl, q_jl)

        @test isapprox(Array(Z_jl), Z_cpu; atol = 2.0e-14)
    end

    GPUArrays.allowscalar(false)
end

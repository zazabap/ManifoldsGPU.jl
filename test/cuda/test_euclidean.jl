@testset "Euclidean CUDA" begin
    @testset "exp! and log!" begin
        Random.seed!(50)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        X = randn(8, 4, 64)

        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)
        @test isapprox(Array(Y_cu), Y_cpu; atol = 2.0e-14)

        V_cpu = log(MP, p, Y_cpu)

        V_cu = log(MP, p_cu, CuArray(Y_cpu))
        @test isapprox(Array(V_cu), V_cpu; atol = 2.0e-14)
    end

    @testset "exp! and log! Float32" begin
        Random.seed!(51)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = Float32.(randn(8, 4, 64))
        X = Float32.(randn(8, 4, 64))

        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)
        @test isapprox(Array(Y_cu), Y_cpu; atol = 2.0f-5)

        V_cpu = log(MP, p, Y_cpu)

        V_cu = log(MP, p_cu, CuArray(Y_cpu))
        @test isapprox(Array(V_cu), V_cpu; atol = 2.0f-5)
    end

    @testset "distance, inner, norm" begin
        Random.seed!(52)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        q = randn(8, 4, 64)
        X = randn(8, 4, 64)
        Y = randn(8, 4, 64)

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        X_cu = CuArray(X)
        Y_cu = CuArray(Y)

        d_cpu = distance(MP, p, q)
        d_gpu = distance(MP, p_cu, q_cu)
        @test isapprox(d_gpu, d_cpu; atol = 1.0e-12)

        v_cpu = inner(MP, p, X, Y)
        v_gpu = inner(MP, p_cu, X_cu, Y_cu)
        @test isapprox(v_gpu, v_cpu; atol = 1.0e-12)

        n_cpu = norm(MP, p, X)
        n_gpu = norm(MP, p_cu, X_cu)
        @test isapprox(n_gpu, n_cpu; atol = 1.0e-12)
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

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        X_cu = CuArray(X)
        Z_cu = similar(X_cu)
        parallel_transport_to!(MP, Z_cu, p_cu, X_cu, q_cu)

        @test isapprox(Array(Z_cu), Z_cpu; atol = 2.0e-14)
    end
end

@testset "Sphere CUDA" begin
    @testset "inner and norm" begin
        Random.seed!(70)

        M = Sphere(7)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        X = rand(MP; vector_at = p)
        Y = rand(MP; vector_at = p)

        i_cpu = inner(MP, p, X, Y)
        n_cpu = norm(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = CuArray(Y)

        i_gpu = inner(MP, p_cu, X_cu, Y_cu)
        n_gpu = norm(MP, p_cu, X_cu)

        @test isapprox(i_gpu, i_cpu; atol = 1.0e-10, rtol = 1.0e-10)
        @test isapprox(n_gpu, n_cpu; atol = 1.0e-10, rtol = 1.0e-10)
    end

    @testset "inner and norm Float32" begin
        Random.seed!(71)

        M = Sphere(7)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        X = Float32.(rand(MP; vector_at = p))
        Y = Float32.(rand(MP; vector_at = p))

        i_cpu = inner(MP, p, X, Y)
        n_cpu = norm(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = CuArray(Y)

        i_gpu = inner(MP, p_cu, X_cu, Y_cu)
        n_gpu = norm(MP, p_cu, X_cu)

        @test isapprox(i_gpu, i_cpu; atol = 1.0f-4, rtol = 1.0f-4)
        @test isapprox(n_gpu, n_cpu; atol = 1.0f-4, rtol = 1.0f-4)
    end
end

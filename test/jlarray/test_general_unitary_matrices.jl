using GPUArrays

@testset "GeneralUnitaryMatrices JLArray" begin
    GPUArrays.allowscalar(true)

    # Test 1: Rotations exp! (Float64)
    @testset "Rotations exp! Float64" begin
        Random.seed!(42)

        M = Rotations(4)
        MP = PowerManifold(M, 16)

        for _ in 1:3
            p = rand(MP)
            X = 0.25 * rand(MP; vector_at = p)
            Y_cpu = exp(MP, p, X)

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            Y_jl = exp(MP, p_jl, X_jl)
            Y_jl_h = Array(Y_jl)

            @test is_point(MP, Y_jl_h)
            @test isapprox(Y_jl_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end

    # Test 2: Rotations exp! (Float32)
    @testset "Rotations exp! Float32" begin
        Random.seed!(43)

        M = Rotations(4)
        MP = PowerManifold(M, 16)

        for _ in 1:3
            p = Float32.(rand(MP))
            X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
            Y_cpu = exp(MP, p, X)

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            Y_jl = exp(MP, p_jl, X_jl)
            Y_jl_h = Array(Y_jl)

            @test is_point(MP, Y_jl_h)
            @test isapprox(Y_jl_h, Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
        end
    end

    # Test 3: UnitaryMatrices exp! (Float64 complex)
    @testset "UnitaryMatrices exp! ComplexF64" begin
        Random.seed!(44)

        M = UnitaryMatrices(4)
        MP = PowerManifold(M, 16)

        for _ in 1:3
            p = rand(MP)
            X = 0.25 * rand(MP; vector_at = p)
            Y_cpu = exp(MP, p, X)

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            Y_jl = exp(MP, p_jl, X_jl)
            Y_jl_h = Array(Y_jl)

            @test is_point(MP, Y_jl_h)
            @test isapprox(Y_jl_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end

    # Test 4: Rotations project! tangent (Float64)
    @testset "Rotations project! tangent Float64" begin
        Random.seed!(45)

        M = Rotations(4)
        MP = PowerManifold(M, 16)

        p = rand(MP)
        X = randn(size(p)...)

        Y_cpu = similar(X)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(Y_cpu, :, :, i), view(p, :, :, i), view(X, :, :, i)
            )
        end

        p_jl = JLArray(p)
        X_jl = JLArray(X)
        Y_jl = similar(X_jl)
        ManifoldsBase.project!(MP, Y_jl, p_jl, X_jl)
        Y_jl_h = Array(Y_jl)

        @test isapprox(Y_jl_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    # Test 5: UnitaryMatrices project! tangent (ComplexF64)
    @testset "UnitaryMatrices project! tangent ComplexF64" begin
        Random.seed!(46)

        M = UnitaryMatrices(4)
        MP = PowerManifold(M, 16)

        p = rand(MP)
        X = randn(ComplexF64, size(p)...)

        Y_cpu = similar(X)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(Y_cpu, :, :, i), view(p, :, :, i), view(X, :, :, i)
            )
        end

        p_jl = JLArray(p)
        X_jl = JLArray(X)
        Y_jl = similar(X_jl)
        ManifoldsBase.project!(MP, Y_jl, p_jl, X_jl)
        Y_jl_h = Array(Y_jl)

        @test isapprox(Y_jl_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    # Test 6: Rotations log! (Float64)
    @testset "Rotations log! Float64" begin
        Random.seed!(60)

        M = Rotations(4)
        MP = PowerManifold(M, 16)

        for _ in 1:3
            p = rand(MP)
            q = rand(MP)
            X_cpu = log(MP, p, q)

            p_jl = JLArray(p)
            q_jl = JLArray(q)
            X_jl = log(MP, p_jl, q_jl)
            X_jl_h = Array(X_jl)

            @test is_vector(MP, p, X_jl_h; atol = 1.0e-10)
            @test isapprox(X_jl_h, X_cpu; atol = 2.0e-10, rtol = 2.0e-10)
        end
    end

    # Test 7: Rotations log! (Float32)
    @testset "Rotations log! Float32" begin
        Random.seed!(61)

        M = Rotations(4)
        MP = PowerManifold(M, 16)

        for _ in 1:3
            p = Float32.(rand(MP))
            q = Float32.(rand(MP))
            X_cpu = log(MP, p, q)

            p_jl = JLArray(p)
            q_jl = JLArray(q)
            X_jl = log(MP, p_jl, q_jl)
            X_jl_h = Array(X_jl)

            @test is_vector(MP, p, X_jl_h; atol = 1.0f-3)
            @test isapprox(X_jl_h, X_cpu; atol = 2.0f-3, rtol = 2.0f-3)
        end
    end

    # NOTE: UnitaryMatrices log! JLArray test is skipped because JLArray hits a
    # map() ambiguity in GPUArrays/LinearAlgebra for log(UpperTriangular(JLArray)).
    # UnitaryMatrices log! is tested in test/cuda/test_general_unitary_matrices.jl.

    GPUArrays.allowscalar(false)
end

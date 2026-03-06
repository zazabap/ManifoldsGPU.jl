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

    GPUArrays.allowscalar(false)
end

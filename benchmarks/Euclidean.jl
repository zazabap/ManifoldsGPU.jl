include("common.jl")

using LinearAlgebra

function _setup_euclidean_data(; n::Int, batch::Int, seed::Int)
    Random.seed!(seed)

    M = Euclidean(n)
    MP = PowerManifold(M, batch)

    p_cpu = Float32.(rand(MP))
    q_cpu = Float32.(rand(MP))
    X_cpu = Float32.(rand(MP; vector_at = p_cpu))
    Y_cpu = Float32.(rand(MP; vector_at = p_cpu))

    p_gpu = CuArray(p_cpu)
    q_gpu = CuArray(q_cpu)
    X_gpu = CuArray(X_cpu)
    Y_gpu = CuArray(Y_cpu)

    return (; MP, p_cpu, q_cpu, X_cpu, Y_cpu, p_gpu, q_gpu, X_gpu, Y_gpu)
end

function benchmark_euclidean_exp(; n::Int = 64, batch::Int = 4096, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, batch = batch, seed = seed)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> exp(data.MP, data.p_cpu, data.X_cpu),
        () -> CUDA.@sync exp(data.MP, data.p_gpu, data.X_gpu);
        samples = samples,
    )

    relerr = begin
        r_cpu = exp(data.MP, data.p_cpu, data.X_cpu)
        r_gpu = Array(CUDA.@sync exp(data.MP, data.p_gpu, data.X_gpu))
        norm(r_cpu .- r_gpu) / max(norm(r_cpu), eps(Float32))
    end

    return _print_results(;
        name = "Euclidean exp", n = n, k = 1, batch = batch, samples = samples,
        cpu_all = cpu_all, gpu_all = gpu_all, cpu_ms = cpu_ms, gpu_ms = gpu_ms,
        relerr = relerr, relerr_label = "||cpu - gpu||/||cpu||",
    )
end

function benchmark_euclidean_log(; n::Int = 64, batch::Int = 4096, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, batch = batch, seed = seed)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> log(data.MP, data.p_cpu, data.q_cpu),
        () -> CUDA.@sync log(data.MP, data.p_gpu, data.q_gpu);
        samples = samples,
    )

    relerr = begin
        r_cpu = log(data.MP, data.p_cpu, data.q_cpu)
        r_gpu = Array(CUDA.@sync log(data.MP, data.p_gpu, data.q_gpu))
        norm(r_cpu .- r_gpu) / max(norm(r_cpu), eps(Float32))
    end

    return _print_results(;
        name = "Euclidean log", n = n, k = 1, batch = batch, samples = samples,
        cpu_all = cpu_all, gpu_all = gpu_all, cpu_ms = cpu_ms, gpu_ms = gpu_ms,
        relerr = relerr, relerr_label = "||cpu - gpu||/||cpu||",
    )
end

function benchmark_euclidean_distance(; n::Int = 64, batch::Int = 4096, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, batch = batch, seed = seed)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> distance(data.MP, data.p_cpu, data.q_cpu),
        () -> CUDA.@sync distance(data.MP, data.p_gpu, data.q_gpu);
        samples = samples,
    )

    relerr = begin
        r_cpu = distance(data.MP, data.p_cpu, data.q_cpu)
        r_gpu = CUDA.@sync distance(data.MP, data.p_gpu, data.q_gpu)
        abs(r_cpu - r_gpu) / max(abs(r_cpu), eps(Float32))
    end

    return _print_results(;
        name = "Euclidean distance", n = n, k = 1, batch = batch, samples = samples,
        cpu_all = cpu_all, gpu_all = gpu_all, cpu_ms = cpu_ms, gpu_ms = gpu_ms,
        relerr = relerr, relerr_label = "|cpu - gpu|/|cpu|",
    )
end

function benchmark_euclidean_inner(; n::Int = 64, batch::Int = 4096, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, batch = batch, seed = seed)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> inner(data.MP, data.p_cpu, data.X_cpu, data.Y_cpu),
        () -> CUDA.@sync inner(data.MP, data.p_gpu, data.X_gpu, data.Y_gpu);
        samples = samples,
    )

    relerr = begin
        r_cpu = inner(data.MP, data.p_cpu, data.X_cpu, data.Y_cpu)
        r_gpu = CUDA.@sync inner(data.MP, data.p_gpu, data.X_gpu, data.Y_gpu)
        abs(r_cpu - r_gpu) / max(abs(r_cpu), eps(Float32))
    end

    return _print_results(;
        name = "Euclidean inner", n = n, k = 1, batch = batch, samples = samples,
        cpu_all = cpu_all, gpu_all = gpu_all, cpu_ms = cpu_ms, gpu_ms = gpu_ms,
        relerr = relerr, relerr_label = "|cpu - gpu|/|cpu|",
    )
end

function benchmark_euclidean_norm(; n::Int = 64, batch::Int = 4096, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, batch = batch, seed = seed)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> norm(data.MP, data.p_cpu, data.X_cpu),
        () -> CUDA.@sync norm(data.MP, data.p_gpu, data.X_gpu);
        samples = samples,
    )

    relerr = begin
        r_cpu = norm(data.MP, data.p_cpu, data.X_cpu)
        r_gpu = CUDA.@sync norm(data.MP, data.p_gpu, data.X_gpu)
        abs(r_cpu - r_gpu) / max(abs(r_cpu), eps(Float32))
    end

    return _print_results(;
        name = "Euclidean norm", n = n, k = 1, batch = batch, samples = samples,
        cpu_all = cpu_all, gpu_all = gpu_all, cpu_ms = cpu_ms, gpu_ms = gpu_ms,
        relerr = relerr, relerr_label = "|cpu - gpu|/|cpu|",
    )
end

function benchmark_euclidean_parallel_transport(; n::Int = 64, batch::Int = 4096, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, batch = batch, seed = seed)
    Z_cpu = similar(data.X_cpu)
    Z_gpu = similar(data.X_gpu)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> parallel_transport_to!(data.MP, Z_cpu, data.p_cpu, data.X_cpu, data.q_cpu),
        () -> CUDA.@sync parallel_transport_to!(data.MP, Z_gpu, data.p_gpu, data.X_gpu, data.q_gpu);
        samples = samples,
    )

    relerr = begin
        parallel_transport_to!(data.MP, Z_cpu, data.p_cpu, data.X_cpu, data.q_cpu)
        CUDA.@sync parallel_transport_to!(data.MP, Z_gpu, data.p_gpu, data.X_gpu, data.q_gpu)
        norm(Z_cpu .- Array(Z_gpu)) / max(norm(Z_cpu), eps(Float32))
    end

    return _print_results(;
        name = "Euclidean parallel_transport_to!", n = n, k = 1, batch = batch, samples = samples,
        cpu_all = cpu_all, gpu_all = gpu_all, cpu_ms = cpu_ms, gpu_ms = gpu_ms,
        relerr = relerr, relerr_label = "||cpu - gpu||/||cpu||",
    )
end

function main()
    n = _parse_arg(1, 64)
    batch = _parse_arg(2, 4096)
    samples = _parse_arg(3, 6)

    println("Euclidean benchmarks: n=$n, batch=$batch, samples=$samples")
    println()
    benchmark_euclidean_exp(; n = n, batch = batch, samples = samples)
    println()
    benchmark_euclidean_log(; n = n, batch = batch, samples = samples)
    println()
    benchmark_euclidean_distance(; n = n, batch = batch, samples = samples)
    println()
    benchmark_euclidean_inner(; n = n, batch = batch, samples = samples)
    println()
    benchmark_euclidean_norm(; n = n, batch = batch, samples = samples)
    println()
    return benchmark_euclidean_parallel_transport(; n = n, batch = batch, samples = samples)
end

main()

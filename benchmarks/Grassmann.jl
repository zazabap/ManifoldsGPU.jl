# Grassmann-specific benchmarks.
# Run standalone: julia --project benchmarks/Grassmann.jl [n] [k] [batch] [samples]
# Or included by main.jl for the combined benchmark suite.

if !@isdefined(_benchmark_exp)
    include(joinpath(@__DIR__, "utils.jl"))
end

function benchmark_grassmann(; n::Int, k::Int, batch::Int, scale::Float32, samples::Int, seed::Int)
    data = _setup_data(
        Grassmann(n, k);
        batch = batch,
        scale = scale,
        seed = seed,
        point_type = Float32,
        use_power_manifold = true,
    )

    MB = data.MB
    manifold_label = "PowerManifold(Grassmann($n, $k), $batch)"
    results = NamedTuple[]

    println("=== Grassmann benchmarks ===")
    println("Point element type: $(eltype(data.p_cpu))")
    println()

    # exp — uses distance-based error because GPU polar decomposition and CPU QR
    # give different matrix representatives of the same subspace.
    push!(results, _benchmark_exp(; MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label, error_fn = _subspace_error))
    println()

    push!(results, _benchmark_inner(; MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, Y_cpu = data.Y_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, Y_gpu = data.Y_gpu, samples = samples, manifold_label = manifold_label))
    println()

    push!(results, _benchmark_norm(; MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label))
    println()

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = _parse_arg(1, 32)
    k = _parse_arg(2, 16)
    batch = _parse_arg(3, 2048)
    samples = _parse_arg(4, 6)

    println("Running Grassmann benchmarks: n=$n, k=$k, batch=$batch, samples=$samples")
    println()

    results = benchmark_grassmann(; n = n, k = k, batch = batch, scale = 0.2f0, samples = samples, seed = 1234)

    println(generate_markdown_summary_table(results))
end

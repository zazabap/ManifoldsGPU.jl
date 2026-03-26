# GeneralUnitaryMatrices-specific benchmarks (Rotations + UnitaryMatrices).
# Run standalone: julia --project benchmarks/GeneralUnitaryMatrices.jl [n] [batch] [samples]
# Or included by main.jl for the combined benchmark suite.

if !@isdefined(_benchmark_exp)
    include(joinpath(@__DIR__, "utils.jl"))
end

function benchmark_rotations(; n::Int, batch::Int, scale::Float32, t::Float32, samples::Int, seed::Int)
    data = _setup_data(
        Rotations(n);
        batch = batch,
        scale = scale,
        seed = seed,
        point_type = Float32,
        use_power_manifold = true,
    )

    MB = data.MB
    manifold_label = "PowerManifold(Rotations($n), $batch)"
    results = NamedTuple[]

    println("=== Rotations benchmarks ===")
    println("Point element type: $(eltype(data.p_cpu))")
    println()

    push!(results, _benchmark_exp(; MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label))
    println()

    push!(results, _benchmark_log!(; MP = MB, p_cpu = data.p_cpu, q_cpu = data.q_cpu, p_gpu = data.p_gpu, q_gpu = data.q_gpu, X_cpu = data.X_cpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label))
    println()

    push!(results, _benchmark_retraction(PolarRetraction(); MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, t = t, samples = samples, manifold_label = manifold_label))
    println()

    return results
end

function benchmark_unitary(; n::Int, batch::Int, scale::Float32, samples::Int, seed::Int)
    data = _setup_data(
        UnitaryMatrices(n);
        batch = batch,
        scale = scale,
        seed = seed,
        point_type = ComplexF32,
        use_power_manifold = true,
    )

    MB = data.MB
    manifold_label = "PowerManifold(UnitaryMatrices($n), $batch)"
    results = NamedTuple[]

    println("=== UnitaryMatrices benchmarks ===")
    println("Point element type: $(eltype(data.p_cpu))")
    println()

    push!(results, _benchmark_exp(; MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label))
    println()

    push!(results, _benchmark_log!(; MP = MB, p_cpu = data.p_cpu, q_cpu = data.q_cpu, p_gpu = data.p_gpu, q_gpu = data.q_gpu, X_cpu = data.X_cpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label))
    println()

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = _parse_arg(1, 16)
    batch = _parse_arg(2, 2048)
    samples = _parse_arg(3, 6)

    println("Running GeneralUnitaryMatrices benchmarks: n=$n, batch=$batch, samples=$samples")
    println()

    all_results = NamedTuple[]
    append!(all_results, benchmark_rotations(; n = n, batch = batch, scale = 0.2f0, t = 0.3f0, samples = samples, seed = 1234))
    append!(all_results, benchmark_unitary(; n = n, batch = batch, scale = 0.2f0, samples = samples, seed = 1235))

    println(generate_markdown_summary_table(all_results))
end

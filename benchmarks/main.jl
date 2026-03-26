# Combined benchmark suite for all manifolds.
# Run: julia --project benchmarks/main.jl [n] [k] [batch] [samples]
#
# Per-manifold benchmarks can also be run standalone:
#   julia --project benchmarks/Grassmann.jl [n] [k] [batch] [samples]
#   julia --project benchmarks/GeneralUnitaryMatrices.jl [n] [batch] [samples]

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "GeneralUnitaryMatrices.jl"))
include(joinpath(@__DIR__, "Grassmann.jl"))

function benchmark_stiefel_retractions(; n::Int, k::Int, batch::Int, scale::Float32, t::Float32, samples::Int, seed::Int)
    data = _setup_data(
        Stiefel(n, k);
        batch = batch,
        scale = scale,
        seed = seed,
        point_type = Float32,
        use_power_manifold = true,
    )

    MB = data.MB
    manifold_label = "PowerManifold(Stiefel($n, $k), $batch)"
    results = NamedTuple[]

    println("=== Stiefel retraction benchmarks ===")
    println("Point element type: $(eltype(data.p_cpu))")
    println()

    push!(results, _benchmark_retraction(ExponentialRetraction(); MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, t = t, samples = samples, manifold_label = manifold_label))
    println()

    push!(results, _benchmark_retraction(PolarRetraction(); MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, t = t, samples = samples, manifold_label = manifold_label))
    println()

    return results
end

function main()
    n = _parse_arg(1, 32)
    k = _parse_arg(2, 16)
    batch = _parse_arg(3, 2048)
    samples = _parse_arg(4, 6)
    scale = 0.2f0
    t = 0.3f0
    seed = 1234

    println("Running with n=$n, k=$k, batch=$batch, samples=$samples")
    println()

    all_results = NamedTuple[]

    append!(all_results, benchmark_manifold("Euclidean($n, $k, $batch)", Euclidean(n, k, batch); batch = batch, scale = scale, samples = samples, seed = seed, point_type = Float32, use_power_manifold = false))

    append!(all_results, benchmark_manifold("Sphere($(n - 1))", Sphere(n - 1); batch = batch, scale = scale, samples = samples, seed = seed + 1, point_type = Float32))

    append!(all_results, benchmark_rotations(; n = n, batch = batch, scale = scale, t = t, samples = samples, seed = seed + 2))

    append!(all_results, benchmark_unitary(; n = n, batch = batch, scale = scale, samples = samples, seed = seed + 3))

    append!(all_results, benchmark_grassmann(; n = n, k = k, batch = batch, scale = scale, samples = samples, seed = seed + 4))

    append!(all_results, benchmark_stiefel_retractions(; n = n, k = k, batch = batch, scale = scale, t = t, samples = samples, seed = seed + 5))

    markdown_table = generate_markdown_summary_table(all_results)
    println("=== Markdown summary table ===")
    println("Device: ", CUDA.name(CUDA.device()), ", eltype: Float32/ComplexF32")
    println(markdown_table)

    return markdown_table
end

main()

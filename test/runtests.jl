using ManifoldsGPU
using Manifolds, ManifoldsBase
using Random, Test

# JLArray tests: always run (no hardware required).
# Files matching test_jlarray_*.jl are auto-discovered in test/jlarray/.
for f in sort(readdir(joinpath(@__DIR__, "jlarray"); join = true))
    if startswith(basename(f), "test_jlarray_")
        include(f)
    end
end

# CUDA tests: only run when CUDA hardware is available.
# Files matching test_cuda_*.jl are auto-discovered in test/cuda/.
using CUDA
if CUDA.functional()
    for f in sort(readdir(joinpath(@__DIR__, "cuda"); join = true))
        if startswith(basename(f), "test_cuda_")
            include(f)
        end
    end
else
    @info "CUDA not available, skipping CUDA tests"
end

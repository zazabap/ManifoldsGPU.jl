using ManifoldsGPU
using Manifolds, ManifoldsBase
using Random, Test

# JLArray tests: always run (no hardware required).
include(joinpath(@__DIR__, "jlarray", "test_jlarray_euclidean.jl"))
include(joinpath(@__DIR__, "jlarray", "test_jlarray_stiefel.jl"))

# CUDA tests: only run when CUDA hardware is available.
using CUDA
if CUDA.functional()
    include(joinpath(@__DIR__, "cuda", "test_cuda_euclidean.jl"))
    include(joinpath(@__DIR__, "cuda", "test_cuda_stiefel.jl"))
else
    @info "CUDA not available, skipping CUDA tests"
end

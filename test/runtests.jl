using ManifoldsGPU
using Test
using Random

using ManifoldsBase, Manifolds

# JLArray tests: always run (no hardware required)
include("jlarray_tests.jl")

# CUDA tests: only run when CUDA is available
using CUDA
if CUDA.functional()
    include("cuda_tests.jl")
else
    @info "CUDA not available, skipping CUDA-specific tests"
end

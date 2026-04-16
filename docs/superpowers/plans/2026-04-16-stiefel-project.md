# Stiefel `project!` (point + tangent) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add batched GPU overrides for Stiefel point projection and tangent vector projection on `PowerManifold` with `CuArray`.

**Architecture:** Two new `ManifoldsBase.project!` methods in the CUDA extension's `Stiefel.jl`. Point projection reuses the existing `_polar_project_gpu!` helper (SVD-based polar decomposition). Tangent projection uses `gemm_strided_batched` for `p'X`, `permutedims` for symmetrization, and in-place `gemm_strided_batched!` for the final accumulation.

**Tech Stack:** Julia, CUDA.jl (cuBLAS `gemm_strided_batched`, cuSOLVER `gesvdj!`/`gesvda!`), ManifoldsBase, Manifolds

**Spec:** `docs/superpowers/specs/2026-04-16-stiefel-project-design.md`

---

### Task 1: Add `project!` (point) to Stiefel.jl

**Files:**
- Modify: `ext/ManifoldsGPUCUDAExt/Stiefel.jl` (insert before `retract_polar_fused!`, after `norm`)

- [ ] **Step 1: Add point projection method**

Insert this function after the `norm` method (after line 49) and before `retract_polar_fused!` (line 51) in `ext/ManifoldsGPUCUDAExt/Stiefel.jl`:

```julia
function ManifoldsBase.project!(
        ::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
    ) where {T <: Real}
    q .= p
    return _polar_project_gpu!(q)
end
```

- [ ] **Step 2: Verify formatting**

Run:
```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "ext/"])'
```
Expected: No changes (code already follows Runic style).

- [ ] **Step 3: Commit**

```bash
git add ext/ManifoldsGPUCUDAExt/Stiefel.jl
git commit -m "add GPU project! (point) for Stiefel"
```

---

### Task 2: Add `project!` (tangent) to Stiefel.jl

**Files:**
- Modify: `ext/ManifoldsGPUCUDAExt/Stiefel.jl` (insert after point projection, before `retract_polar_fused!`)

- [ ] **Step 1: Add tangent projection method**

Insert this function immediately after the point projection method added in Task 1:

```julia
function ManifoldsBase.project!(
        ::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        Y::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Real}
    A = CUDA.CUBLAS.gemm_strided_batched('C', 'N', p, X)    # p'X, k×k×batch
    sym = A .+ permutedims(A, (2, 1, 3))                     # A + A'
    Y .= X
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', T(-0.5), p, sym, one(T), Y)
    return Y
end
```

Formula: `Y = X - 0.5 * p * (p'X + X'p)`. The `permutedims(A, (2, 1, 3))` transposes each k×k slice of the batch, giving `A' = (p'X)'= X'p`.

- [ ] **Step 2: Verify formatting**

Run:
```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "ext/"])'
```
Expected: No changes.

- [ ] **Step 3: Commit**

```bash
git add ext/ManifoldsGPUCUDAExt/Stiefel.jl
git commit -m "add GPU project! (tangent) for Stiefel"
```

---

### Task 3: Add CUDA tests for point projection

**Files:**
- Modify: `test/cuda/test_stiefel.jl` (append test blocks inside `@testset "Stiefel CUDA"`, before the closing `end`)

- [ ] **Step 1: Add Float64 point projection test**

Append before the final `end` of `@testset "Stiefel CUDA"` (before line 177):

```julia
    @testset "project! point Float64" begin
        Random.seed!(76)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 32)

        p = randn(size(rand(MP))...)

        q_cpu = similar(p)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(q_cpu, :, :, i), view(p, :, :, i)
            )
        end

        p_cu = CuArray(p)
        q_cu = similar(p_cu)
        ManifoldsBase.project!(MP, q_cu, p_cu)
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        @test isapprox(q_cu_h, q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end
```

- [ ] **Step 2: Add Float32 point projection test**

Append immediately after the Float64 test:

```julia
    @testset "project! point Float32" begin
        Random.seed!(77)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 32)

        p = Float32.(randn(size(rand(MP))...))

        q_cpu = similar(p)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(q_cpu, :, :, i), view(p, :, :, i)
            )
        end

        p_cu = CuArray(p)
        q_cu = similar(p_cu)
        ManifoldsBase.project!(MP, q_cu, p_cu)
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        @test isapprox(q_cu_h, q_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end
```

- [ ] **Step 3: Commit**

```bash
git add test/cuda/test_stiefel.jl
git commit -m "add CUDA tests for Stiefel project! (point)"
```

---

### Task 4: Add CUDA tests for tangent projection

**Files:**
- Modify: `test/cuda/test_stiefel.jl` (append after point projection tests)

- [ ] **Step 1: Add Float64 tangent projection test**

Append after the point projection tests:

```julia
    @testset "project! tangent Float64" begin
        Random.seed!(78)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        X = randn(size(p)...)

        Y_cpu = similar(X)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(Y_cpu, :, :, i), view(p, :, :, i), view(X, :, :, i)
            )
        end

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = similar(X_cu)
        ManifoldsBase.project!(MP, Y_cu, p_cu, X_cu)
        Y_cu_h = Array(Y_cu)

        @test isapprox(Y_cu_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end
```

- [ ] **Step 2: Add Float32 tangent projection test**

Append immediately after:

```julia
    @testset "project! tangent Float32" begin
        Random.seed!(79)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        X = Float32.(randn(size(p)...))

        Y_cpu = similar(X)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(Y_cpu, :, :, i), view(p, :, :, i), view(X, :, :, i)
            )
        end

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = similar(X_cu)
        ManifoldsBase.project!(MP, Y_cu, p_cu, X_cu)
        Y_cu_h = Array(Y_cu)

        @test isapprox(Y_cu_h, Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end
```

- [ ] **Step 3: Commit**

```bash
git add test/cuda/test_stiefel.jl
git commit -m "add CUDA tests for Stiefel project! (tangent)"
```

---

### Task 5: Format and verify

- [ ] **Step 1: Run Runic formatter on all changed files**

```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "ext/", "test/"])'
```

- [ ] **Step 2: Run JLArray tests to verify no regressions**

```bash
julia --project test/runtests.jl
```
Expected: All existing tests pass. The new Stiefel `project!` methods won't be exercised (no GPU), but this confirms imports and module loading work.

- [ ] **Step 3: Commit any formatting changes (if any)**

```bash
git add -A && git diff --cached --quiet || git commit -m "apply Runic formatting"
```

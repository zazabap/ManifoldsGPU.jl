# Stiefel `project!` (point + tangent) GPU Overrides

**Date:** 2026-04-16
**Issue:** JuliaManifolds/ManifoldsGPU.jl#5 (step 5)
**Depends on:** Existing Stiefel GPU overrides (exp!, retract_polar_fused!, inner, norm)

## Goal

Add batched GPU overrides for Stiefel point projection and tangent vector projection on `PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation}` with `CuArray{T, 3}` where `T <: Real`. These replace ManifoldsBase's sequential `get_iterator` loop with batched GPU operations.

## Operations

### 1. `project!` (point) — project n×k matrix onto Stiefel(n, k)

**CPU formula:** `q = U * V'` from SVD of `p` (polar decomposition)

**GPU implementation:**
```julia
q .= p
_polar_project_gpu!(q)
```

Reuses existing `_polar_project_gpu!` from helpers.jl (batched SVD via gesvdj!/gesvda! fallback). Identical to Grassmann point projection — both manifolds use the same polar decomposition formula.

### 2. `project!` (tangent) — project ambient vector into tangent space at `p`

**CPU formula:** `Y = X - p * sym(p'X)` where `sym(A) = (A + A') / 2`

The tangent space constraint for Stiefel is `p'Y + Y'p = 0` (skew-symmetry of `p'Y`).

**GPU implementation:**
```julia
A = gemm_strided_batched('C', 'N', p, X)           # p'X, k×k×batch
sym = A .+ permutedims(A, (2, 1, 3))                # A + A' (symmetrize each slice)
Y .= X
gemm_strided_batched!('N', 'N', T(-0.5), p, sym, one(T), Y)  # Y = X - 0.5*p*(A+A')
```

Three GPU operations: one GEMM for `p'X`, one `permutedims` + broadcast add for symmetrization, one in-place GEMM for the final accumulation.

**Note:** Grassmann tangent projection is `Y = X - p*(p'X)` (simpler — no symmetrization needed because Grassmann's tangent constraint is `p'X = 0`). Stiefel's tangent constraint `p'X + X'p = 0` only requires the symmetric part to vanish, hence the `sym()` operator.

## Dispatch signatures

Both methods dispatch on:
- Manifold: `PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation}`
- Arrays: `CuArray{T, 3} where {T <: Real}`

This matches the existing `exp!`/`inner`/`norm`/`retract_polar_fused!` signatures in Stiefel.jl.

## Files changed

### `ext/ManifoldsGPUCUDAExt/Stiefel.jl`
Add 2 functions (~20 lines) before `retract_polar_fused!`:
- `ManifoldsBase.project!(M, q, p)` — point projection
- `ManifoldsBase.project!(M, Y, p, X)` — tangent projection

No new imports needed — `ManifoldsBase` is already available via `using ManifoldsBase`, and the methods use the fully-qualified `ManifoldsBase.project!` syntax.

### `test/cuda/test_stiefel.jl`
Add 4 test blocks (~60 lines) to the existing `@testset "Stiefel CUDA"`:
- `project! point Float64` — `randn(n, k, batch)` arbitrary ambient matrix, project, verify `is_point(MP, ...)` and compare GPU vs CPU slice-by-slice (`atol = 2e-14`)
- `project! point Float32` — same at lower precision (`atol = 2f-5`)
- `project! tangent Float64` — valid `p` from `rand(MP)`, arbitrary `X` from `randn(...)`, project, compare GPU vs CPU slice-by-slice (`atol = 2e-14`)
- `project! tangent Float32` — same at lower precision (`atol = 2f-5`)

CPU reference is computed slice-by-slice using the single-manifold `ManifoldsBase.project!(M, view(...), view(...))` to avoid relying on the PowerManifold loop (which is what we're overriding).

### No JLArray tests
Both operations use cuBLAS (`gemm_strided_batched`) or cuSOLVER (`gesvdj!`), so they are CUDA-only. This is consistent with the existing Stiefel test structure (no JLArray tests for exp! or retract either — only GeneralUnitaryMatrices has JLArray tests).

## What is NOT included

- Complex Stiefel (`T <: Complex`) — out of scope per issue #5
- `log!` — delegates to iterative shooting via `StiefelSubmersionMetric`; hard to GPU-ify
- `parallel_transport_to!`, `inverse_retract_polar!`, `inverse_retract_qr!` — listed as nice-to-have in issue #5
- `retract_qr_fused!` — step 6 in the roadmap (Cholesky-QR approach)
- Benchmarks — can be added separately

## Branch

Feature branch `feat/stiefel-project` from `main`.

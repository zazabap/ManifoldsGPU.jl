# CLAUDE.md

## Project Overview

GPU-accelerated manifold operations for the JuliaManifolds ecosystem.
Provides CUDA extensions for Manifolds.jl manifolds, enabling batched
operations on PowerManifolds via CuArrays.

## Skills

- [check-code-quality](skills/check-code-quality/SKILL.md) — Review Julia + GPU code for correctness, type stability, and AD safety.
- [create-pr](skills/create-pr/SKILL.md) — Create a pull request from the current branch.
- [add-manifold](skills/add-manifold/SKILL.md) — Step-by-step guide for implementing GPU ops for a new manifold.

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                      # Run all tests
julia --project=benchmarks benchmarks/main.jl 32 16 2048 6        # Run benchmark
julia --project=. -e 'using Pkg; Pkg.status()'                    # Check dependencies
git config core.hooksPath .githooks                           # one-time: activate Runic pre-commit hook after clone
julia --project=. -e 'using Runic; for f in filter(f -> endswith(f, ".jl"), readdir(".", join=true)); Runic.format_file(f; inplace=true); end'  # format all .jl files manually
```

## Git Safety

- **NEVER force push** (`git push --force`, `git push -f`). Absolute rule, no exceptions.
- **PRs go to the fork only.** Always create PRs on the forked repository (`origin`). Never open PRs directly against the upstream main repo — those are submitted manually by the maintainer.

## Architecture

### Extension Structure

All GPU code lives in `ext/ManifoldsGPUCUDAExt/`:

```
ext/ManifoldsGPUCUDAExt/
├── ManifoldsGPUCUDAExt.jl          # Entry point, includes all manifold files
├── helpers.jl                      # Shared: _matrix_exp_gpu, _matrix_log_gpu, _batched_inv_gpu, _batched_sqrtm_gpu
├── Stiefel.jl                      # exp!, retract_polar_fused! + retract_fused! for PowerManifold(Stiefel)
├── Grassmann.jl                    # exp!, retract_polar_fused! + retract_fused! for PowerManifold(Grassmann)
├── GeneralUnitaryMatrices.jl       # exp!, retract_polar_fused!, retract_fused!, project! for PowerManifold(GeneralUnitaryMatrices)
```

### Dispatch Pattern

Methods dispatch on `CuArray` element type and dimensionality:
- `CuArray{T, 3}` — batched (PowerManifold with ArrayPowerRepresentation)
- `CuArray{T, 2}` — single manifold element

The extension dispatches on `PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation}`
with `CuArray` point types.

**Two-level retraction structure:** `retract_polar_fused!` is the implementation;
`retract_fused!(M, q, p, X, t, PolarRetraction())` dispatches to it. New retraction
types each get their own `retract_<type>_fused!` function.

### Key Design Decisions

1. `_matrix_exp_gpu` in `helpers.jl` is shared across manifolds — uses Scaling & Squaring
   with Taylor series (not `exp()`) to stay fully on-device
2. `PowerManifold` + `ArrayPowerRepresentation` is the primary target: pack batch dim as
   3rd axis → single `gemm_strided_batched` call
3. Extension never imports Manifolds.jl internals — only public ManifoldsBase.jl API
4. No `ManoptCUDAExt` needed — Manopt.jl `_produce_type` fix (PR #579) handles GPU
   allocation when solvers are initialized with a GPU point
5. **Only add `PowerManifold` GPU overrides for manifolds that genuinely need batched
   CUBLAS/cuSOLVER calls** (Stiefel, Grassmann, UnitaryMatrices). Do NOT add overrides
   for manifolds where the operations are pure broadcasting (e.g., Euclidean, Sphere) —
   those work on CuArrays already via equivalent non-power types:
   - `PowerManifold(Euclidean(n), k)` → use `Euclidean(n, k)` instead
   - `PowerManifold(Sphere(n-1), k)` → use `Oblique(n, k)` instead

### Manopt.jl Integration

```julia
# Correct: pass GPU point to stepsize constructor
step = ArmijoLinesearch(M, p_gpu)
gradient_descent(M, f, grad_f, p_gpu; stepsize=step)

# Wrong: CPU allocation, won't propagate to GPU
step = ArmijoLinesearch(M)
```

### Implemented Manifolds

| Manifold              | GPU ops implemented    | CUDA primitive                        |
|-----------------------|------------------------|---------------------------------------|
| Stiefel               | `exp!`, `retract!`     | `gemm_strided_batched`, `gesvdj!`     |
| Grassmann             | `exp!`, `retract!`     | `gemm_strided_batched`, `gesvdj!`     |
| GeneralUnitaryMatrices| `exp!`, `log!`, `retract!`, `project!` | `gemm_strided_batched`, `gesvdj!`, `getrf/getri` + Taylor series|
| SymmetricPositiveDefinite | `exp!`             | CPU eigendecomp per slice (serial)    |

### Manifolds That Do NOT Need PowerManifold Overrides

These manifolds have equivalent non-power types that work with CuArrays via broadcasting:

| PowerManifold form              | Use instead       | Why                                    |
|---------------------------------|--------------------|----------------------------------------|
| `PowerManifold(Euclidean(n), k)`| `Euclidean(n, k)`  | Same operations, just broadcasting     |
| `PowerManifold(Sphere(n-1), k)` | `Oblique(n, k)`   | Columns are unit vectors, same `exp!`  |

## Conventions

### File Naming

- Extension files: `ext/ManifoldsGPUCUDAExt/<ManifoldName>.jl` The ManifoldName should be found in the original [repo](https://github.com/JuliaManifolds/Manifolds.jl/tree/master/src/manifolds)
- JLArray tests: `test/jlarray/test_<manifold_name>.jl` (CI-safe)
- CUDA tests: `test/cuda/test_<manifold_name>.jl` (requires hardware)
- Benchmark scripts: `benchmarks/<ManifoldName>.jl`

### Naming Conventions

- **Functions**: snake_case — `exp!`, `retract_fused!`, `_matrix_exp_gpu`
- **Internal helpers**: underscore prefix — `_matrix_exp_gpu`, `_setup_stiefel_data`
- **Type parameters**: single uppercase letters — `T`, `M`

### Code Style

- Docstrings with `"""..."""` for public overrides explaining the GPU strategy used
- `@assert` for dimension checks before GPU kernel calls
- Explicit `CUDA.@sync` in benchmarks; not in library code

## Testing Requirements

### Coverage

New GPU methods need both a JLArray test and a CUDA test.
JLArray tests **must** pass with `GPUArrays.allowscalar(false)` — needing `allowscalar(true)` means the override isn't dispatching.

### Key Testing Patterns

- `Random.seed!(42)` for reproducibility
- Compare GPU result to CPU reference: `@test isapprox(Array(Y_gpu), Y_cpu; atol=...)`
- Verify manifold membership: `@test is_point(MP, Array(Y_gpu))`
- Test Float32 and Float64 separately (different tolerances)
- JLArray tests go in `test/jlarray/` (runs in CI)
- CUDA tests go in `test/cuda/` (manual only)

### Tolerances

- Float64: `atol=2e-14, rtol=2e-14`
- Float32: `atol=2e-5, rtol=2e-5`

## Documentation Locations

- `docs/plans/` — Design documents and implementation plans
- `benchmarks/` — Performance benchmarks

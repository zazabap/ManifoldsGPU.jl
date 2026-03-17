# ManifoldsGPU

```@docs
ManifoldsGPU.ManifoldsGPU
```

## Benchmarks

Device: NVIDIA GeForce RTX 5070 Ti, eltype: Float32/ComplexF32

| Manifold | Operation | CPU median [ms] | GPU median [ms] | Speedup CPU/GPU | Relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| Euclidean(32, 16, 2048) | exp | 0.54 | 0.18 | 2.93 | 0.0 |
| Euclidean(32, 16, 2048) | log! | 0.46 | 0.19 | 2.44 | 0.0 |
| Euclidean(32, 16, 2048) | inner | 0.28 | 0.14 | 1.99 | 1.871e-7 |
| Euclidean(32, 16, 2048) | norm | 0.16 | 0.19 | 0.87 | 8.423e-8 |
| Euclidean(32, 16, 2048) | project! | 0.36 | 0.14 | 2.66 | 0.0 |
| PowerManifold(Sphere(31), 2048) | exp | 0.05 | 47.61 | 0.0 | 6.877e-8 |
| PowerManifold(Sphere(31), 2048) | log! | 0.17 | 84.27 | 0.0 | 4.262e-8 |
| PowerManifold(Sphere(31), 2048) | inner | 0.02 | 0.14 | 0.16 | 5.86e-7 |
| PowerManifold(Sphere(31), 2048) | norm | 0.03 | 0.15 | 0.21 | 1.064e-7 |
| PowerManifold(Sphere(31), 2048) | project! | 0.04 | 41.05 | 0.0 | 2.813e-8 |
| PowerManifold(Rotations(32), 2048) | exp | 41.4 | 2.42 | 17.14 | 2.594e-6 |
| PowerManifold(Rotations(32), 2048) | log! | 557.82 | 72.72 | 7.67 | 9.157e-5 |
| PowerManifold(Rotations(32), 2048) | inner | 0.58 | 0.15 | 3.81 | 4.708e-6 |
| PowerManifold(Rotations(32), 2048) | norm | 1.51 | 0.14 | 10.53 | 1.109e-6 |
| PowerManifold(Rotations(32), 2048) | project! | 22.73 | 0.26 | 87.15 | 3.644e-7 |
| PowerManifold(UnitaryMatrices(32), 2048) | exp | 89.11 | 7.68 | 11.6 | 1.957e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | log! | 775.64 | 72.03 | 10.77 | 0.0001844 |
| PowerManifold(UnitaryMatrices(32), 2048) | inner | 1.16 | 75.78 | 0.02 | 5.979e-5 |
| PowerManifold(UnitaryMatrices(32), 2048) | norm | 1.87 | 45.72 | 0.04 | 1.516e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | project! | 33.5 | 0.38 | 88.74 | 5.512e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | exp | 74.28 | 1064.36 | 0.07 | 8.275e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | log! | 68.93 | 1033.57 | 0.07 | 1.854e-5 |
| PowerManifold(Grassmann(32, 16), 2048) | inner | 0.28 | 0.16 | 1.77 | 8.056e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | norm | 0.85 | 0.15 | 5.5 | 1.848e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | project! | 1.15 | 54.21 | 0.02 | 1.329e-7 |
| PowerManifold(Stiefel(32, 16), 2048) | exp(ExponentialRetraction) | 76.59 | 3.56 | 21.54 | 1.164e-6 |
| PowerManifold(Stiefel(32, 16), 2048) | retract_fused!(PolarRetraction) | 45.23 | 2.97 | 15.24 | 1.37e-6 |

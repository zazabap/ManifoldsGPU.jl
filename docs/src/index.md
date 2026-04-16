# ManifoldsGPU

```@docs
ManifoldsGPU.ManifoldsGPU
```

## Benchmarks

Device: NVIDIA GeForce RTX 3090, eltype: Float32/ComplexF32

| Manifold | Operation | CPU median [ms] | GPU median [ms] | Speedup CPU/GPU | Relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| Euclidean(32, 16, 2048) | exp | 1.09 | 0.11 | 9.62 | 0.0 |
| Euclidean(32, 16, 2048) | log! | 1.05 | 0.13 | 8.03 | 0.0 |
| Euclidean(32, 16, 2048) | inner | 0.58 | 0.15 | 3.9 | 4.679e-7 |
| Euclidean(32, 16, 2048) | norm | 0.67 | 0.09 | 7.44 | 8.423e-8 |
| Euclidean(32, 16, 2048) | project! | 0.54 | 0.07 | 7.4 | 0.0 |
| PowerManifold(Sphere(31), 2048) | exp | 0.15 | 56.4 | 0.0 | 6.877e-8 |
| PowerManifold(Sphere(31), 2048) | log! | 0.28 | 112.0 | 0.0 | 4.262e-8 |
| PowerManifold(Sphere(31), 2048) | inner | 0.06 | 0.1 | 0.6 | 1.074e-6 |
| PowerManifold(Sphere(31), 2048) | norm | 0.08 | 0.11 | 0.77 | 0.0 |
| PowerManifold(Sphere(31), 2048) | project! | 0.11 | 56.74 | 0.0 | 2.813e-8 |
| PowerManifold(Rotations(32), 2048) | exp | 122.64 | 5.94 | 20.65 | 2.594e-6 |
| PowerManifold(Rotations(32), 2048) | log! | 3128.55 | 125.03 | 25.02 | 9.157e-5 |
| PowerManifold(Rotations(32), 2048) | inner | 1.33 | 0.13 | 10.63 | 5.008e-6 |
| PowerManifold(Rotations(32), 2048) | norm | 3.22 | 0.13 | 24.07 | 9.784e-7 |
| PowerManifold(Rotations(32), 2048) | project! | 124.76 | 0.46 | 273.38 | 3.644e-7 |
| PowerManifold(Rotations(32), 2048) | retract_fused!(PolarRetraction) | 298.95 | 5.61 | 53.32 | 2.554e-6 |
| PowerManifold(Rotations(32), 2048) | retract_fused!(QRRetraction) | 471.96 | 1.58 | 298.33 | 3.307e-7 |
| PowerManifold(UnitaryMatrices(32), 2048) | exp | 248.92 | 13.31 | 18.7 | 1.95e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | log! | 4388.76 | 120.95 | 36.29 | 0.0001548 |
| PowerManifold(UnitaryMatrices(32), 2048) | inner | 2.84 | 85.34 | 0.03 | 6.639e-5 |
| PowerManifold(UnitaryMatrices(32), 2048) | norm | 5.8 | 54.56 | 0.11 | 1.253e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | project! | 155.5 | 0.38 | 407.37 | 5.512e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | exp | 201.87 | 6.47 | 31.19 | 7.042e-5 |
| PowerManifold(Grassmann(32, 16), 2048) | log! | 312.38 | 4.06 | 76.95 | 2.287e-5 |
| PowerManifold(Grassmann(32, 16), 2048) | inner | 0.68 | 0.11 | 6.25 | 1.549e-6 |
| PowerManifold(Grassmann(32, 16), 2048) | norm | 1.77 | 0.1 | 17.54 | 3.696e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | project! | 4.1 | 0.21 | 19.69 | 1.303e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | retract_fused!(PolarRetraction) | 110.93 | 3.37 | 32.94 | 1.332e-6 |
| PowerManifold(Grassmann(32, 16), 2048) | retract_fused!(QRRetraction) | 53.98 | 1.38 | 39.23 | 0.2119 |
| PowerManifold(Stiefel(32, 16), 2048) | exp(ExponentialRetraction) | 211.41 | 7.11 | 29.73 | 1.164e-6 |
| PowerManifold(Stiefel(32, 16), 2048) | retract_fused!(PolarRetraction) | 117.31 | 3.35 | 35.01 | 1.373e-6 |
| PowerManifold(Stiefel(32, 16), 2048) | retract_fused!(QRRetraction) | 55.93 | 1.27 | 44.02 | 1.943e-7 |

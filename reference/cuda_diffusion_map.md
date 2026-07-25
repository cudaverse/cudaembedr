# Diffusion-map-style embedding

Pairwise distances can use the `cudalearnr` CUDA path. Kernel
construction and eigendecomposition currently run on the CPU.

## Usage

``` r
cuda_diffusion_map(
  x,
  n_components = 2L,
  sigma = NULL,
  diffusion_time = 1,
  metric = c("euclidean", "cosine"),
  device = c("auto", "cuda", "cpu")
)
```

## Arguments

- x:

  Numeric observation-by-feature matrix or compatible cudaverse result.

- n_components:

  Output dimensions.

- sigma:

  Gaussian kernel bandwidth. Defaults to the median positive pairwise
  distance.

- diffusion_time:

  Non-negative diffusion time exponent.

- metric:

  Euclidean or cosine distance.

- device:

  Device passed to
  [`cudalearnr::cuda_distance()`](https://rdrr.io/pkg/cudalearnr/man/cuda_distance.html).

## Value

A `cuda_embedding` with the stable fields documented by
[`cuda_umap()`](https://cudaverse.github.io/cudaembedr/reference/cuda_umap.md),
stage-level distance/kernel/eigendecomposition provenance, and an
additional `eigenvalues` element.

## Examples

``` r
cuda_diffusion_map(
  matrix(rnorm(120), 40, 3),
  n_components = 2,
  device = "cpu"
)
#> <cuda_embedding method=diffusion observations=40 dimensions=2 backend=base-eigen compute_device=cpu>
```

# UMAP embedding

UMAP currently uses the CPU `uwot` backend. GPU-aware cudaverse inputs
are accepted and their source device is retained in the result metadata.

## Usage

``` r
cuda_umap(
  x,
  n_components = 2L,
  n_neighbors = 15L,
  min_dist = 0.1,
  metric = "euclidean",
  n_epochs = NULL,
  seed = NULL,
  ...
)
```

## Arguments

- x:

  Numeric observation-by-feature matrix or compatible cudaverse result.

- n_components:

  Output dimensions.

- n_neighbors:

  Number of nearest neighbours.

- min_dist:

  Minimum UMAP distance.

- metric:

  Distance metric passed to
  [`uwot::umap()`](https://jlmelville.github.io/uwot/reference/umap.html).

- n_epochs:

  Optional training epochs.

- seed:

  Optional random seed.

- ...:

  Additional arguments passed to
  [`uwot::umap()`](https://jlmelville.github.io/uwot/reference/umap.html).

## Value

A `cuda_embedding` list containing `coordinates`, `method`, `backend`,
`compute_device`, per-stage `compute_stages`, source metadata, and
algorithm `parameters`.

## Examples

``` r
if (requireNamespace("uwot", quietly = TRUE)) {
  cuda_umap(matrix(rnorm(120), 40, 3), n_neighbors = 5, seed = 1)
}
#> <cuda_embedding method=umap observations=40 dimensions=2 backend=uwot compute_device=cpu>
```

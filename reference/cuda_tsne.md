# t-SNE embedding

t-SNE currently uses the CPU `Rtsne` backend.

## Usage

``` r
cuda_tsne(x, n_components = 2L, perplexity = 30, theta = 0.5, seed = NULL, ...)
```

## Arguments

- x:

  Numeric observation-by-feature matrix or compatible cudaverse result.

- n_components:

  Output dimensions.

- perplexity:

  t-SNE perplexity.

- theta:

  Barnes-Hut accuracy/speed trade-off.

- seed:

  Optional random seed.

- ...:

  Additional arguments passed to
  [`Rtsne::Rtsne()`](https://rdrr.io/pkg/Rtsne/man/Rtsne.html).

## Value

A `cuda_embedding`; see
[`cuda_umap()`](https://cudaverse.github.io/cudaembedr/reference/cuda_umap.md)
for the stable result fields.

## Examples

``` r
if (requireNamespace("Rtsne", quietly = TRUE)) {
  cuda_tsne(matrix(rnorm(120), 40, 3), perplexity = 5, seed = 1)
}
#> <cuda_embedding method=tsne observations=40 dimensions=2 backend=Rtsne compute_device=cpu>
```

# cudaembedr

`cudaembedr` is the embedding layer of the **cudaverse**. It gives UMAP,
t-SNE, and diffusion-map-style embeddings a common result structure and
accepts matrices, `cuda_pca` results, `cudatensor` objects, and
`cudacell_workflow` results.

## Current backends

- UMAP: `uwot` on CPU;
- t-SNE: `Rtsne` on CPU;
- diffusion map: CUDA-aware pairwise distances through `cudalearnr`,
  followed by CPU kernel construction and eigendecomposition.

Each result reports `source_device`, `compute_device`, and `backend`.
Diffusion-map results also report `compute_stages`, separating distance,
kernel, and eigendecomposition provenance. A CUDA distance followed by
CPU kernel/eigendecomposition is therefore reported as
`compute_device = "hybrid"`, never as an entirely GPU-native workflow.

## Installation

``` r

# install.packages("pak")
pak::pak("cudaverse/cudaembedr")
```

The diffusion-map implementation works with the base eigensolver and
uses `RSpectra` automatically when installed. UMAP and t-SNE are
optional adapters:

``` r

pak::pak(c("uwot", "Rtsne", "RSpectra"))
```

Installing those packages does not make UMAP or t-SNE GPU-native; their
current backends remain CPU implementations as listed above.

## Example

``` r

library(cudalearnr)
library(cudaembedr)

pca <- cuda_pca(matrix(rnorm(1000), 100, 10), n_components = 5)
embedding <- cuda_diffusion_map(pca, n_components = 2)

embedding
embedding_coordinates(embedding)
```

See the cudaverse [end-to-end
workflow](https://github.com/cudaverse/.github/blob/main/WORKFLOW.md)
for a complete sparse-counts-to-embedding example.

For installation, device verification, hybrid-stage interpretation, and
common failures, see the cudaverse [GPU setup and troubleshooting
guide](https://github.com/cudaverse/.github/blob/main/GPU_SETUP.md).

## License

MIT © Yaoxiang Li

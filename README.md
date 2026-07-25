# cudaembedr

`cudaembedr` is the embedding layer of the **cudaverse**. It gives UMAP,
t-SNE, and diffusion-map-style embeddings a common result structure and accepts
matrices, `cuda_pca` results, `cudatensor` objects, and `cudacell_workflow`
results.

## Current backends

- UMAP: `uwot` on CPU;
- t-SNE: `Rtsne` on CPU;
- diffusion map: CUDA-aware pairwise distances through `cudalearnr`, followed
  by CPU kernel construction and eigendecomposition.

Each result reports `source_device`, `compute_device`, and `backend`.
Diffusion-map results also report `compute_stages`, separating distance,
kernel, and eigendecomposition provenance. A CUDA distance followed by CPU
kernel/eigendecomposition is therefore reported as `compute_device =
"hybrid"`, never as an entirely GPU-native workflow.

## Installation

```r
# install.packages("pak")
pak::pak("cudaverse/cudaembedr")
```

## Example

```r
library(cudalearnr)
library(cudaembedr)

pca <- cuda_pca(matrix(rnorm(1000), 100, 10), n_components = 5)
embedding <- cuda_diffusion_map(pca, n_components = 2)

embedding
embedding_coordinates(embedding)
```

## License

MIT © Yaoxiang Li

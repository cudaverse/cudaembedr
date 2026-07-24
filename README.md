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

Each result reports `source_device`, `compute_device`, and `backend`. This keeps
the public API ready for future device-resident implementations without
misrepresenting the current execution path.

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

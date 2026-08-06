# cudaembedr

> **Archived:** This package has been incorporated into
> [`cudaverse`](https://github.com/cudaverse/cudaverse). Install and load
> `cudaverse`, then continue using the same embedding functions. This
> repository remains available as development history and receives no new
> features or releases.

`cudaembedr` is the embedding layer of the **cudaverse**. It gives UMAP,
t-SNE, and diffusion-map-style embeddings a common result structure and accepts
matrices, `cuda_pca` results, `cudatensor` objects, and `cudacell_workflow`
results. It also accepts a `SingleCellExperiment` reduced dimension without
copying assays or cell metadata into a package-specific container.

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

| Method | Source provenance | Current compute provenance |
|---|---|---|
| UMAP | retained from the input | CPU through `uwot` |
| t-SNE | retained from the input | CPU through `Rtsne` |
| Diffusion map | retained from the input | distance on CPU or CUDA; kernel and eigendecomposition on CPU |

The [backend and provenance article](https://cudaverse.github.io/cudaembedr/articles/backend-provenance.html)
walks through source versus current stages, diffusion-map hybrid execution,
quadratic memory requirements, and the optional CUDA hardware gate.

## Installation

```r
# install.packages("pak")
pak::pak("cudaverse/cudaembedr")
```

The diffusion-map implementation works with the base eigensolver and uses
`RSpectra` automatically when installed. UMAP and t-SNE are optional adapters:

```r
pak::pak(c("uwot", "Rtsne", "RSpectra"))
```

Installing those packages does not make UMAP or t-SNE GPU-native; their
current backends remain CPU implementations as listed above.

`SingleCellExperiment` support is optional and can be installed from
Bioconductor:

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
BiocManager::install("SingleCellExperiment")
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

Embedding coordinate row names retain the input observation identifiers.
This also holds when the input is a `cuda_pca` or `cudacell_workflow`, so cell
names remain aligned from counts through the final low-dimensional result.

## SingleCellExperiment input

Pass a reduced dimension directly instead of extracting its matrix by hand:

```r
embedding <- cuda_diffusion_map(
  sce,
  reduced_dim = "PCA",
  n_components = 2,
  device = "cpu"
)
```

When `reduced_dim = NULL`, `cudaembedr` first uses
`metadata(sce)$cudacellr$reduced_dim`, then a uniquely named `"PCA"`.
Non-PCA choices are never guessed, even if only one exists; pass its name
explicitly. The error lists the available names.

When the selected reduced dimension is the output named by a cudacellr
metadata record, that record also carries the upstream stage provenance and
aggregate source compute device into the embedding. Selecting a different
reduced dimension never borrows that lineage. `source_provenance` describes
the selected input's upstream work, while `cuda_provenance(embedding)`
describes only the current embedding. Coordinate row names always follow
`colnames(sce)`. A generic or unrelated SCE reduced dimension reports
`source_device` and `source_compute_device` as `"unknown"`.

See the cudaverse
[end-to-end workflow](https://github.com/cudaverse/.github/blob/main/WORKFLOW.md)
for a complete sparse-counts-to-embedding example.

For installation, device verification, hybrid-stage interpretation, and common
failures, see the cudaverse
[GPU setup and troubleshooting guide](https://github.com/cudaverse/.github/blob/main/GPU_SETUP.md).

## License

MIT © Yaoxiang Li

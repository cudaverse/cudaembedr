# Backend and provenance

`cudaembedr` keeps upstream work separate from the work performed by the
current embedding:

- **source provenance** describes the input object, such as the stages
  used to compute a PCA result;
- **current compute provenance** describes the embedding stages returned
  by
  [`cuda_provenance()`](https://cudaverse.github.io/cudaembedr/reference/cuda_provenance.md).

This distinction prevents a CUDA source from being mistaken for an
end-to-end CUDA embedding.

## A runnable CPU diffusion map

This example computes PCA and the diffusion map on the CPU. Row names
are preserved through both results.

``` r

library(cudaembedr)

set.seed(1)
observations <- matrix(rnorm(40 * 5), 40, 5)
rownames(observations) <- paste0("observation_", seq_len(nrow(observations)))

pca <- cudalearnr::cuda_pca(
  observations,
  n_components = 3,
  device = "cpu"
)
embedding <- cuda_diffusion_map(
  pca,
  n_components = 2,
  device = "cpu"
)

embedding
#> <cuda_embedding method=diffusion observations=40 dimensions=2 backend=base-eigen compute_device=cpu>
head(embedding_coordinates(embedding))
#>                 DIFFUSION1    DIFFUSION2
#> observation_1 -0.001416858 -0.0009952801
#> observation_2 -0.004355441 -0.0028505600
#> observation_3  0.001478039 -0.0031071846
#> observation_4 -0.007913494  0.0076251042
#> observation_5  0.001198100 -0.0024159201
#> observation_6 -0.010683413 -0.0053600315
embedding$source_device
#> [1] "cpu"
embedding$source_compute_device
#> [1] "cpu"
embedding$source_provenance
#> <cuda_provenance schema=cudaverse-stage/1 stages=2 compute=cpu>
#>          stage requested_device device backend selection_reason fallback
#>  preprocessing              cpu    cpu   stats     explicit_cpu    FALSE
#>  decomposition              cpu    cpu   stats     explicit_cpu    FALSE
#>  output_device
#>            cpu
#>            cpu
cuda_provenance(embedding)
#> <cuda_provenance schema=cudaverse-stage/1 stages=3 compute=cpu>
#>               stage requested_device device    backend   selection_reason
#>            distance              cpu    cpu       base       explicit_cpu
#>              kernel        fixed-cpu    cpu       base algorithm_cpu_only
#>  eigendecomposition        fixed-cpu    cpu base-eigen algorithm_cpu_only
#>  fallback output_device
#>     FALSE           cpu
#>     FALSE           cpu
#>     FALSE           cpu
```

`embedding$source_provenance` contains the upstream PCA stages.
`cuda_provenance(embedding)` contains only the current distance, kernel,
and eigendecomposition stages. With `device = "cpu"`, all three current
stages use the CPU and `embedding$compute_device` is `"cpu"`.

UMAP and t-SNE use fixed CPU backends (`uwot` and `Rtsne`,
respectively). Their source metadata can still describe an upstream CUDA
result, but their current embedding stage remains CPU.

## Use a SingleCellExperiment reduced dimension

`cudaembedr` can read a cell-by-component matrix directly from a
`SingleCellExperiment`. The returned coordinate rows follow the SCE
column names, so users do not need to extract a reduced dimension and
restore cell identifiers manually.

``` r

if (requireNamespace("SingleCellExperiment", quietly = TRUE)) {
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = t(observations))
  )
  SingleCellExperiment::reducedDim(sce, "PCA") <- pca$x

  sce_embedding <- cuda_diffusion_map(
    sce,
    reduced_dim = "PCA",
    n_components = 2,
    device = "cpu"
  )

  identical(
    rownames(embedding_coordinates(sce_embedding)),
    colnames(sce)
  )
}
#> [1] TRUE
```

When `reduced_dim = NULL`, selection is deterministic:

1.  use `metadata(sce)$cudacellr$reduced_dim` when cudacellr recorded
    one;
2.  otherwise use a unique reduced dimension named `"PCA"`;
3.  otherwise require an explicit `reduced_dim`.

Non-PCA choices are never guessed, even if only one exists. The error
lists the available names. Explicit `reduced_dim` always wins. A valid
cudacellr metadata record becomes `source_provenance` and
`source_compute_device` only when the selected reduced dimension is the
output named by that record, without being mixed into the current
embedding stages. Selecting a different reduced dimension reports
unknown source devices instead of borrowing unrelated lineage.

When a generic SCE has no cudacellr provenance record, both
source-device fields are `"unknown"`; the package does not infer how an
existing reduced dimension was computed.

## Why a CUDA diffusion map is hybrid

`device` in
[`cuda_diffusion_map()`](https://cudaverse.github.io/cudaembedr/reference/cuda_diffusion_map.md)
controls the pairwise-distance stage. Kernel construction and
eigendecomposition are currently fixed to the CPU. The truthful stage
summary is therefore:

| Distance stage | Kernel stage | Eigendecomposition stage | `compute_device` |
|----------------|--------------|--------------------------|------------------|
| CPU            | CPU          | CPU                      | `"cpu"`          |
| CUDA           | CPU          | CPU                      | `"hybrid"`       |

The result’s top-level `backend` names the eigendecomposition backend
(`base-eigen` or `RSpectra`). Use
[`cuda_provenance()`](https://cudaverse.github.io/cudaembedr/reference/cuda_provenance.md)
when the backend and device of every stage matter.

## Quadratic memory is still the limiting factor

Diffusion maps currently materialize a complete pairwise-distance
matrix, followed by dense kernel and normalized matrices. One
double-precision `n`-by-`n` matrix alone requires approximately
`8 * n^2` bytes:

``` r

one_dense_matrix_gib <- function(n) {
  8 * as.double(n)^2 / 1024^3
}

data.frame(
  observations = c(1e3, 1e4, 5e4),
  one_dense_matrix_GiB = round(
    one_dense_matrix_gib(c(1e3, 1e4, 5e4)),
    2
  )
)
#>   observations one_dense_matrix_GiB
#> 1         1000                 0.01
#> 2        10000                 0.75
#> 3        50000                18.63
```

Peak memory is several times this single-matrix estimate because
multiple dense intermediates coexist. CUDA can accelerate distance
computation, but it does not remove the `O(n^2)` host-memory requirement
of the current implementation. Estimate memory before running large data
and prefer methods that do not materialize all pairwise distances when
`n` is large.

## Optional CUDA execution and the hardware gate

The next block runs only when a usable CUDA backend exists. It requests
CUDA explicitly and verifies the hybrid contract: CUDA distance, CPU
kernel, and CPU eigendecomposition.

The cudaverse NVIDIA workflow sets `CUDAVERSE_REQUIRE_CUDA=true`. Under
that gate, an unavailable CUDA runtime is an error rather than an
allowed skip.

``` r

require_cuda <- identical(
  tolower(Sys.getenv("CUDAVERSE_REQUIRE_CUDA", "false")),
  "true"
)
cuda_ready <- cudatensr::cuda_available()

if (require_cuda && !cuda_ready) {
  stop("The hardware gate requires a usable CUDA backend.")
}

if (cuda_ready) {
  embedding_cuda <- cuda_diffusion_map(
    observations,
    n_components = 2,
    device = "cuda"
  )
  provenance_cuda <- cuda_provenance(embedding_cuda)
  stage_devices <- stats::setNames(
    provenance_cuda$device,
    provenance_cuda$stage
  )

  stopifnot(
    identical(embedding_cuda$compute_device, "hybrid"),
    identical(unname(stage_devices["distance"]), "cuda"),
    identical(unname(stage_devices["kernel"]), "cpu"),
    identical(unname(stage_devices["eigendecomposition"]), "cpu")
  )

  provenance_cuda
}
```

Use `device = "cuda"` when CUDA is mandatory. Use `device = "auto"` only
when the recorded CPU fallback is acceptable. In either case,
[`cuda_provenance()`](https://cudaverse.github.io/cudaembedr/reference/cuda_provenance.md)
is the authoritative record of what actually ran.

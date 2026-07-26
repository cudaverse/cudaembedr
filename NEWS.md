# cudaembedr 0.3.0

- UMAP, t-SNE, and diffusion maps now accept `SingleCellExperiment` inputs
  through a named reduced dimension. Automatic selection prefers the
  cudacellr metadata record, then a standard `"PCA"` entry, and never guesses
  a non-PCA choice.
- Embeddings created from a `SingleCellExperiment` preserve cell identifiers
  and retain cudacellr source provenance and aggregate compute metadata when
  the selected reduced dimension is the recorded cudacellr output. Unrelated
  reduced dimensions never borrow that provenance.
- A generic SCE without cudacellr provenance reports its source device as
  `"unknown"` rather than guessing that its reduced dimension was computed on
  the CPU.

# cudaembedr 0.2.0

- Embedding stages now use the shared validated provenance schema, including
  requested device, actual device, backend, fallback reason, and output device.
- Embeddings retain the full provenance and aggregate compute device of
  compatible cudaverse inputs.
- Diffusion maps preserve the original device request and continue to report
  CUDA distance plus CPU kernel/eigendecomposition as hybrid execution.
- UMAP, t-SNE, and diffusion results now retain their complete normalized
  explicit parameters.

# cudaembedr 0.1.2

- Added an end-to-end single-cell workflow test proving that cell identifiers
  remain attached to embedding coordinates.

# cudaembedr 0.1.1

- Embedding results now expose stable per-stage `compute_stages` provenance.
- Diffusion maps correctly report `compute_device = "hybrid"` when CUDA
  distance calculation is followed by CPU kernel construction and
  eigendecomposition.
- Corrected the t-SNE `...` documentation to reference `Rtsne::Rtsne()`.
